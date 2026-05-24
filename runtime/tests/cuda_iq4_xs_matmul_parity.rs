//! IQ4_XS fused matmul parity (FR-17-extra-iq4_xs-fwd).
//!
//! 136-byte 256-elem block.  Per-sub-block 6-bit signed scale (4 low bits
//! from scales_l + 2 high bits from scales_h) × kvalues_iq4nl codebook
//! lookup per element.
//!
//! Used by cnc's GLM-4.7-flash-UD-IQ3_XXS for ~55 tensors.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_iq4_xs_matmul_seq1_cuda,
};

const QK: usize = 256;
const BYTES_PER_BLOCK: usize = 136;

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
];

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
    let d_bits = unsafe { aether_f32_to_f16(0.05) } as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };

    // 8 sub-blocks × 6-bit unsigned scale (will be interpreted as `ls - 32`).
    let mut ls = [0u32; 8];
    for i in 0..8 { ls[i] = next_u32() & 0x3F; }
    // Pack scales_h (16 bits = 2 bits per sub-block × 8) at bytes 2-3.
    let mut scales_h: u32 = 0;
    for i in 0..8 {
        scales_h |= ((ls[i] >> 4) & 3) << (2 * i);
    }
    bytes[2] = (scales_h & 0xFF) as u8;
    bytes[3] = ((scales_h >> 8) & 0xFF) as u8;
    // Pack scales_l (4 bytes = 8 nibbles) at bytes 4-7.
    for j in 0..4 {
        let lo = ls[2 * j]     & 0xF;
        let hi = ls[2 * j + 1] & 0xF;
        bytes[4 + j] = (lo | (hi << 4)) as u8;
    }
    // Pack qs (128 bytes = 256 nibble indices).
    let mut indices = [0u32; QK];
    for k in 0..QK { indices[k] = next_u32() & 0xF; }
    for ib in 0..8 {
        for j in 0..16 {
            let lo = indices[ib * 32 + j] & 0xF;
            let hi = indices[ib * 32 + j + 16] & 0xF;
            bytes[8 + ib * 16 + j] = (lo | (hi << 4)) as u8;
        }
    }
    // Reference dequant.
    let mut dq = [0f32; QK];
    for ib in 0..8 {
        let dl = d * ((ls[ib] as i32 - 32) as f32);
        for j in 0..16 {
            dq[ib * 32 + j]      = dl * (KVALUES_IQ4NL[indices[ib * 32 + j]      as usize] as f32);
            dq[ib * 32 + j + 16] = dl * (KVALUES_IQ4NL[indices[ib * 32 + j + 16] as usize] as f32);
        }
    }
    (bytes, dq)
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
fn iq4_xs_matches_cpu_dequant_small() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 32;
    let k = 512;
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
        for i in 0..k { acc += a[i] * w_dequant[row * k + i]; }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_iq4_xs_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[iq4_xs] n={} k={} max_diff={:.3e}", n, k, max_diff);
        assert!(gpu_out.iter().all(|x| x.is_finite()));
        assert!(max_diff < 1e-2, "IQ4_XS diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn iq4_xs_matches_at_glm_class_shape() {
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
        for i in 0..k { acc += a[i] * w_dequant[row * k + i]; }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_iq4_xs_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[iq4_xs-2048] max_diff={:.3e}", max_diff);
        assert!(max_diff < 5.0, "IQ4_XS at GLM shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}
