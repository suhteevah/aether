//! matt-voice / FR-17.14-extra-qlora-bwd — backward through a frozen
//! QUANTIZED linear `y = W x`: `dx = Wᵀ · dy`.
//!
//! This is the GPU primitive that unblocks LoRA fine-tuning backprop for
//! Qwen2.5-7B: to flow the loss *through* a frozen quantized base linear
//! back to the LoRA adapters on the previous layer you need `dx = Wᵀ·dy`,
//! where `W` is a Q4_K / Q6_K device buffer.
//!
//! Parity strategy: dequant a REAL Qwen2.5-7B weight tensor on the CPU
//! (the trusted `aether_dequant_q4_k_m` / `aether_dequant_q6_k`), compute
//! the reference `dx = Wᵀ·dy` with plain f32 loops over the SAME
//! dequantised W, then run the GPU op `aether_op_quant_matmul_backward_lhs_f32_cuda`
//! against the same raw GGUF u8 bytes and assert max-abs-diff < 1e-3.
//!
//! Gated on the Qwen blob existing (skips if absent, like qwen25_paged_parity).
//
// roadmap: P18

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_dequant_q4_k_m, aether_dequant_q6_k,
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_quant_matmul_backward_lhs_f32_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

/// Run the parity check against one real GGUF tensor.
///
/// `tensor_name` must be a quantized 2-D weight whose inner (column)
/// dimension is a multiple of 256 — Qwen2.5-7B's projection weights are
/// 3584-wide (= 14 super-blocks), so a contiguous prefix of `n_out` rows
/// forms a clean `[n_out, n_in]` sub-matrix in the GGUF byte stream.
///
/// `block_bytes` is 144 for Q4_K (dt=12), 210 for Q6_K (dt=14).
unsafe fn parity_for(tensor_name: &[u8], expect_dt: c_int, block_bytes: usize) {
    aether_dev_init();
    let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
    assert!(h >= 0, "gguf open failed");
    let idx = aether_gguf_find_tensor_by_name(h, tensor_name.as_ptr() as i64, tensor_name.len() as c_int);
    assert!(idx >= 0, "tensor {:?} not found", std::str::from_utf8(tensor_name).unwrap());
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    assert_eq!(dt, expect_dt, "unexpected dtype for {:?}", std::str::from_utf8(tensor_name).unwrap());

    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    // Qwen2.5-7B projection weights are square-ish [d, d] with d=3584=14*256.
    // Derive the row width from the total element count assuming a square
    // matrix; fall back gracefully if it isn't.
    let n_in_full = 3584usize;
    assert_eq!(n_elems % n_in_full, 0, "tensor width not 3584; got {} elems", n_elems);
    let n_rows_full = n_elems / n_in_full;
    assert_eq!(n_in_full % 256, 0, "n_in not a multiple of 256");

    // Sub-matrix: first n_out rows, full n_in columns. Rows are contiguous
    // super-blocks in the GGUF stream, so this is a clean byte prefix.
    let n_out = 64usize.min(n_rows_full);
    let n_in = n_in_full;
    let blocks_per_row = n_in / 256;
    let n_blocks = n_out * blocks_per_row;
    let n_bytes = n_blocks * block_bytes;

    let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;
    let bytes: Vec<u8> = std::slice::from_raw_parts(dptr, n_bytes).to_vec();

    // --- CPU: dequant the sub-matrix W [n_out, n_in] row-major ---
    let mut w_cpu = vec![0.0f32; n_out * n_in];
    match dt {
        12 => { aether_dequant_q4_k_m(bytes.as_ptr() as *const c_void, w_cpu.as_mut_ptr() as *mut c_void, n_blocks as c_int); }
        14 => { aether_dequant_q6_k(bytes.as_ptr() as *const c_void, w_cpu.as_mut_ptr() as *mut c_void, n_blocks as c_int); }
        _  => unreachable!(),
    }

    // Synthetic upstream gradient dy [n_out].
    let dy: Vec<f32> = (0..n_out).map(|o| (((o as f32) * 0.137).sin()) * 0.5 + 0.1).collect();

    // --- CPU reference: dx[i] = Σ_o W[o,i] * dy[o] ---
    let mut dx_ref = vec![0.0f32; n_in];
    for o in 0..n_out {
        let dyo = dy[o];
        let row = &w_cpu[o * n_in..(o + 1) * n_in];
        for i in 0..n_in {
            dx_ref[i] += row[i] * dyo;
        }
    }

    // --- GPU: same raw bytes → dx via the new op ---
    let d_w   = aether_dev_alloc_u8(n_bytes as c_int);
    let d_dy  = aether_dev_alloc_f32(n_out as c_int);
    let d_dx  = aether_dev_alloc_f32(n_in as c_int);
    assert!(d_w != 0 && d_dy != 0 && d_dx != 0, "device alloc failed");
    aether_dev_h2d_u8(bytes.as_ptr() as i64, d_w, n_bytes as c_int);
    aether_dev_h2d_f32(dy.as_ptr() as i64, d_dy, n_out as c_int);

    let rc = aether_op_quant_matmul_backward_lhs_f32_cuda(
        d_w, dt, d_dy, d_dx, n_out as c_int, n_in as c_int,
    );
    assert_eq!(rc, 0, "quant_matmul_backward_lhs returned {}", rc);
    aether_dev_sync();

    let mut dx_gpu = vec![0.0f32; n_in];
    aether_dev_d2h_f32(d_dx, dx_gpu.as_mut_ptr() as i64, n_in as c_int);

    aether_dev_free_u8(d_w);
    aether_dev_free_f32(d_dy);
    aether_dev_free_f32(d_dx);
    aether_gguf_close(h);

    // --- compare ---
    let mut max_diff = 0.0f32;
    let mut worst_i = 0usize;
    for i in 0..n_in {
        let d = (dx_gpu[i] - dx_ref[i]).abs();
        if d > max_diff { max_diff = d; worst_i = i; }
    }
    eprintln!(
        "[qlora-bwd {}] dt={} n_out={} n_in={} -> dx=Wᵀ·dy  max|gpu-cpu|={:.3e} at i={}",
        std::str::from_utf8(tensor_name).unwrap(), dt, n_out, n_in, max_diff, worst_i,
    );
    eprintln!("  cpu dx[..4]: {:?}", &dx_ref[..4]);
    eprintln!("  gpu dx[..4]: {:?}", &dx_gpu[..4]);
    assert!(max_diff < 1e-3, "quant matmul backward parity exceeded 1e-3: {:.3e}", max_diff);
}

#[test]
fn quant_matmul_backward_q4k_real_qwen25() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B blob not present");
        return;
    }
    // attn_q.weight is Q4_K (dt=12) in matt-voice's Q4_K_M Qwen2.5-7B.
    unsafe { parity_for(b"blk.0.attn_q.weight", 12, 144); }
}

#[test]
fn quant_matmul_backward_q6k_real_qwen25() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B blob not present");
        return;
    }
    // attn_v.weight is Q6_K (dt=14) in matt-voice's Q4_K_M Qwen2.5-7B.
    unsafe { parity_for(b"blk.0.attn_v.weight", 14, 210); }
}

// ===========================================================================
// matt-voice FR-17.14-extra-qlora-bwd — IQ3_XXS (dt=18) backward parity.
//
// roadmap: P18
//
// IQ3_XXS is the target base quant for 70B QLoRA training (Llama-3.3-70B
// IQ3_XXS ≈ 27.5 GB fits the GPU pool). kokonoe has no IQ3_XXS GGUF (only a
// Q4_K_M Qwen2.5-7B), so this case SYNTHESISES random IQ3_XXS-packed bytes,
// dequants on CPU with the same unpacking the GPU kernel uses, computes the
// reference dx = Wᵀ·dy with plain f32 loops, then runs the GPU op against the
// raw packed bytes and asserts max|gpu-cpu| < 1e-3.
// ===========================================================================

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};

const IQ3XXS_QK: usize = 256;
const IQ3XXS_BYTES_PER_BLOCK: usize = 98;

// llama.cpp's ksigns_iq2xs: 7-bit index → 8-bit sign pattern.
const KSIGNS_IQ2XS: [u8; 128] = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
];

// llama.cpp's iq3xxs_grid: 8-bit index → packed 4-uint8 quant pattern (u32).
const IQ3XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

fn iq3xxs_rng(state: &mut u64) -> u32 {
    let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
    *state = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z >> 32) ^ z) as u32
}

/// Build one synthetic 98-byte IQ3_XXS block + its CPU dequant.
fn iq3xxs_synth_block(seed: u64) -> ([u8; IQ3XXS_BYTES_PER_BLOCK], [f32; IQ3XXS_QK]) {
    let mut state = seed.wrapping_add(1);
    let mut bytes = [0u8; IQ3XXS_BYTES_PER_BLOCK];
    let d = 0.07f32;
    let d_bits = unsafe { aether_f32_to_f16(d) } as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;
    for i in 0..64 {
        bytes[2 + i] = (iq3xxs_rng(&mut state) & 0xFF) as u8;
    }
    for ib32 in 0..8 {
        let scale = (iq3xxs_rng(&mut state) & 0xF) as u32;
        let s0 = (iq3xxs_rng(&mut state) & 0x7F) as u32;
        let s1 = (iq3xxs_rng(&mut state) & 0x7F) as u32;
        let s2 = (iq3xxs_rng(&mut state) & 0x7F) as u32;
        let s3 = (iq3xxs_rng(&mut state) & 0x7F) as u32;
        let aux32 = s0 | (s1 << 7) | (s2 << 14) | (s3 << 21) | (scale << 28);
        let off = 2 + 64 + 4 * ib32;
        bytes[off + 0] = (aux32 & 0xFF) as u8;
        bytes[off + 1] = ((aux32 >> 8) & 0xFF) as u8;
        bytes[off + 2] = ((aux32 >> 16) & 0xFF) as u8;
        bytes[off + 3] = ((aux32 >> 24) & 0xFF) as u8;
    }
    let dq = iq3xxs_dequant_block(&bytes);
    (bytes, dq)
}

/// CPU reference dequant — mirrors the dequant_iq3_xxs GPU kernel exactly.
fn iq3xxs_dequant_block(bytes: &[u8; IQ3XXS_BYTES_PER_BLOCK]) -> [f32; IQ3XXS_QK] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let qs = &bytes[2..66];
    let sas = &bytes[66..98];
    let mut out = [0f32; IQ3XXS_QK];
    for ib32 in 0..8 {
        let aux32 = (sas[4 * ib32 + 0] as u32)
            | ((sas[4 * ib32 + 1] as u32) << 8)
            | ((sas[4 * ib32 + 2] as u32) << 16)
            | ((sas[4 * ib32 + 3] as u32) << 24);
        let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize] as u32;
            let grid1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 0] as usize];
            let grid2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
            for j in 0..4 {
                let q0 = ((grid1 >> (8 * j)) & 0xFF) as u32;
                let q1 = ((grid2 >> (8 * j)) & 0xFF) as u32;
                let s0 = if (signs & (1 << (j + 0))) != 0 { -1.0 } else { 1.0 };
                let s1 = if (signs & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                out[32 * ib32 + 8 * l + j + 0] = db * (q0 as f32) * s0;
                out[32 * ib32 + 8 * l + j + 4] = db * (q1 as f32) * s1;
            }
        }
    }
    out
}

#[test]
fn quant_matmul_backward_iq3_xxs_synthetic() {
    // No GGUF dependency — pure synthetic IQ3_XXS bytes, always runs.
    unsafe { assert_eq!(aether_dev_init(), 0); }

    let n_out = 64usize;
    let n_in = 1024usize;           // 4 super-blocks per row, 256-aligned
    let blocks_per_row = n_in / IQ3XXS_QK;
    let n_blocks = n_out * blocks_per_row;
    let n_bytes = n_blocks * IQ3XXS_BYTES_PER_BLOCK;

    // Pack W [n_out, n_in] row-major (GGUF natural order: each row is its
    // own run of super-blocks) and capture the CPU-dequantised f32 W.
    let mut w_packed = vec![0u8; n_bytes];
    let mut w_cpu = vec![0f32; n_out * n_in];
    for o in 0..n_out {
        for b in 0..blocks_per_row {
            let (bytes, dq) = iq3xxs_synth_block(o as u64 * 1000 + b as u64);
            let off = (o * blocks_per_row + b) * IQ3XXS_BYTES_PER_BLOCK;
            w_packed[off..off + IQ3XXS_BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..IQ3XXS_QK { w_cpu[o * n_in + b * IQ3XXS_QK + i] = dq[i]; }
        }
    }

    // Synthetic upstream gradient dy [n_out].
    let dy: Vec<f32> = (0..n_out).map(|o| (((o as f32) * 0.137).sin()) * 0.5 + 0.1).collect();

    // CPU reference dx[i] = Σ_o W[o,i] * dy[o].
    let mut dx_ref = vec![0f32; n_in];
    for o in 0..n_out {
        let dyo = dy[o];
        let row = &w_cpu[o * n_in..(o + 1) * n_in];
        for i in 0..n_in { dx_ref[i] += row[i] * dyo; }
    }

    unsafe {
        let d_w  = aether_dev_alloc_u8(n_bytes as c_int);
        let d_dy = aether_dev_alloc_f32(n_out as c_int);
        let d_dx = aether_dev_alloc_f32(n_in as c_int);
        assert!(d_w != 0 && d_dy != 0 && d_dx != 0, "device alloc failed");
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, d_w, n_bytes as c_int);
        aether_dev_h2d_f32(dy.as_ptr() as i64, d_dy, n_out as c_int);

        let rc = aether_op_quant_matmul_backward_lhs_f32_cuda(
            d_w, 18, d_dy, d_dx, n_out as c_int, n_in as c_int,
        );
        assert_eq!(rc, 0, "quant_matmul_backward_lhs (IQ3_XXS) returned {}", rc);
        aether_dev_sync();

        let mut dx_gpu = vec![0f32; n_in];
        aether_dev_d2h_f32(d_dx, dx_gpu.as_mut_ptr() as i64, n_in as c_int);

        aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_dy);
        aether_dev_free_f32(d_dx);

        let mut max_diff = 0.0f32;
        let mut worst_i = 0usize;
        for i in 0..n_in {
            let d = (dx_gpu[i] - dx_ref[i]).abs();
            if d > max_diff { max_diff = d; worst_i = i; }
        }
        let n_finite = dx_gpu.iter().filter(|x| x.is_finite()).count();
        eprintln!(
            "[qlora-bwd IQ3_XXS synthetic] dt=18 n_out={} n_in={} -> dx=Wᵀ·dy  max|gpu-cpu|={:.3e} at i={} finite={}/{}",
            n_out, n_in, max_diff, worst_i, n_finite, n_in,
        );
        eprintln!("  cpu dx[..4]: {:?}", &dx_ref[..4]);
        eprintln!("  gpu dx[..4]: {:?}", &dx_gpu[..4]);
        assert_eq!(n_finite, n_in, "non-finite values in IQ3_XXS dx output");
        assert!(max_diff < 1e-3, "IQ3_XXS quant matmul backward parity exceeded 1e-3: {:.3e}", max_diff);
    }
}
