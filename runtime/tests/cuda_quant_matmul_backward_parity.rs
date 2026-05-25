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
