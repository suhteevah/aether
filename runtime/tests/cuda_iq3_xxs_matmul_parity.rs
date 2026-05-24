//! IQ3_XXS fused matmul parity (FR-17-extra-iq3_xxs-fwd).
//!
//! cnc's GLM-4.7-flash-UD-IQ3_XXS uses IQ3_XXS for nearly every weight
//! tensor.  Format: 98-byte 256-element block (2-byte f16 scale + 64-byte
//! codebook indices + 32-byte scales_and_signs).  Dequant uses a 256-entry
//! grid lookup + 128-entry sign-pattern lookup with a 4-bit per-sub-block
//! scale offset.
//!
//! This test constructs synthetic IQ3_XXS-encoded weights with random
//! grid indices, random signs, and known scales, then verifies the GPU
//! matmul matches a CPU reference that runs the same dequant.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_iq3_xxs_matmul_seq1_cuda,
};

const QK: usize = 256;
const BYTES_PER_BLOCK: usize = 98;

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

/// Build a synthetic 98-byte IQ3_XXS block from random grid/sign/scale picks.
/// Returns (block_bytes, dequantised_values).
fn synth_block(seed: u64) -> ([u8; BYTES_PER_BLOCK], [f32; QK]) {
    let mut state = seed.wrapping_add(1);
    let mut next_u32 = || {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z >> 32) ^ z) as u32
    };
    let mut bytes = [0u8; BYTES_PER_BLOCK];
    // f16 d ≈ 0.07 (typical for IQ3 weights).
    let d = 0.07f32;
    let d_bits = unsafe { aether_f32_to_f16(d) } as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;
    // 64 codebook indices: random in [0, 256).
    for i in 0..64 {
        bytes[2 + i] = (next_u32() & 0xFF) as u8;
    }
    // 32 bytes scales_and_signs: 8 × u32, each = scale<<28 | 4×7-bit sign idx.
    for ib32 in 0..8 {
        let scale = (next_u32() & 0xF) as u32;  // 4-bit scale
        let s0 = (next_u32() & 0x7F) as u32;
        let s1 = (next_u32() & 0x7F) as u32;
        let s2 = (next_u32() & 0x7F) as u32;
        let s3 = (next_u32() & 0x7F) as u32;
        let aux32 = s0 | (s1 << 7) | (s2 << 14) | (s3 << 21) | (scale << 28);
        let off = 2 + 64 + 4 * ib32;
        bytes[off + 0] = (aux32 & 0xFF) as u8;
        bytes[off + 1] = ((aux32 >> 8) & 0xFF) as u8;
        bytes[off + 2] = ((aux32 >> 16) & 0xFF) as u8;
        bytes[off + 3] = ((aux32 >> 24) & 0xFF) as u8;
    }
    let dq = dequant_iq3_xxs_block(&bytes);
    (bytes, dq)
}

/// CPU reference dequant — mirrors the GPU kernel exactly.
fn dequant_iq3_xxs_block(bytes: &[u8; BYTES_PER_BLOCK]) -> [f32; QK] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let qs = &bytes[2..66];
    let sas = &bytes[66..98];
    let mut out = [0f32; QK];
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

fn deterministic(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n).map(|_| {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let u = (((z >> 32) ^ z) as u32) as f32 / 4_294_967_296.0;
        (u * 2.0 - 1.0) * scale
    }).collect()
}

#[test]
fn iq3_xxs_matches_cpu_dequant_small() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 32;
    let k = 256;  // exactly one IQ3_XXS block per row
    let n_blocks = k / QK;
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let (bytes, dq) = synth_block(row as u64 * 1000 + b as u64);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..QK { w_dequant[row * k + b * QK + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 11, 1.0);
    let mut cpu_out = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += a[i] * w_dequant[row * k + i];
        }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_iq3_xxs_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0, "fused_iq3_xxs rc={}", rc);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_out.iter().filter(|x| x.is_finite()).count();
        println!("[iq3_xxs] n={} k={} max_diff={:.3e} finite={}/{}",
            n, k, max_diff, n_finite, n);
        assert_eq!(n_finite, n, "non-finite values in IQ3_XXS output");
        assert!(max_diff < 1e-4,
            "IQ3_XXS matmul diverged from CPU reference ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn iq3_xxs_matches_at_glm_47_flash_class_shape() {
    // GLM-4.7-flash uses 256-aligned d_model dimensions throughout (it's
    // built on the deepseek2 spec where d_model = 2048 in the small variant
    // — though GLM-4.7-flash itself may differ; this test just exercises
    // an attn-class shape with 8 IQ3_XXS blocks per row.
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 2048;
    let k = 2048;
    let n_blocks = k / QK;
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let (bytes, dq) = synth_block(row as u64 * 1000 + b as u64);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..QK { w_dequant[row * k + b * QK + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 23, 1.0);
    let mut cpu_out = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += a[i] * w_dequant[row * k + i];
        }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_iq3_xxs_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[iq3_xxs-2048] max_diff={:.3e}", max_diff);
        assert!(max_diff < 1e-2,
            "IQ3_XXS at GLM-class shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}
