//! Q4_0 fused matmul parity (FR-17-extra-q4_0-fwd).
//!
//! Constructs synthetic Q4_0 weights (32-elem blocks, 18 bytes each:
//! f16 scale `d` + 16 nibble-packed bytes) and verifies the fused GPU
//! matmul matches a CPU reference (dequant + dense matmul).
//!
//! Synthetic shape: n=64 out columns × k=256 in elems (8 Q4_0 blocks
//! per row).  Each row gets a distinct scale + nibble pattern so the
//! kernel can't pass by skipping any of the dequant arithmetic.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::aether_f32_to_f16;
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q4_0_matmul_seq1_cuda,
};

const BLOCK_K: usize = 32;
const BYTES_PER_BLOCK: usize = 18;

/// Pack a single 32-element f32 chunk as a Q4_0 block.  Returns 18 bytes:
///   bytes 0-1: f16 scale `d` = max|x| / 7
///   bytes 2-17: 16 bytes of nibbles.  byte i holds:
///     low nibble  = q(x[i])      in [0, 15]
///     high nibble = q(x[i + 16]) in [0, 15]
///   where q(v) = clamp(round(v / d) + 8, 0, 15).
fn pack_q4_0_block(chunk: &[f32; 32]) -> [u8; 18] {
    let max_abs = chunk.iter().fold(0f32, |a, b| a.max(b.abs()));
    let d = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
    let mut out = [0u8; 18];
    let d_bits = unsafe { aether_f32_to_f16(d) } as u16;
    out[0] = (d_bits & 0xFF) as u8;
    out[1] = ((d_bits >> 8) & 0xFF) as u8;
    for i in 0..16 {
        let lo = ((chunk[i]      / d).round() + 8.0).clamp(0.0, 15.0) as u32;
        let hi = ((chunk[i + 16] / d).round() + 8.0).clamp(0.0, 15.0) as u32;
        out[2 + i] = ((hi << 4) | lo) as u8;
    }
    out
}

/// Inverse of pack_q4_0_block — dequantise the same 18 bytes back to 32
/// f32 values using identical arithmetic to the GPU kernel.
fn unpack_q4_0_block(bytes: &[u8; 18]) -> [f32; 32] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_rt::aether_f16_to_f32(d_bits as i32) };
    let mut out = [0f32; 32];
    for i in 0..16 {
        let byte = bytes[2 + i];
        let q_lo = (byte & 0x0F) as i32 - 8;
        let q_hi = ((byte >> 4) & 0x0F) as i32 - 8;
        out[i]      = d * (q_lo as f32);
        out[i + 16] = d * (q_hi as f32);
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
fn q4_0_fused_matmul_matches_cpu_dequant() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 64;
    let k = 256;
    assert!(k % BLOCK_K == 0);
    let n_blocks = k / BLOCK_K;

    // 1. Sample full-precision target weights and pack as Q4_0.
    let w_target = deterministic(n * k, 11, 0.3);
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let mut chunk = [0f32; BLOCK_K];
            for i in 0..BLOCK_K {
                chunk[i] = w_target[row * k + b * BLOCK_K + i];
            }
            let bytes = pack_q4_0_block(&chunk);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            let dq = unpack_q4_0_block(&bytes);
            for i in 0..BLOCK_K {
                w_dequant[row * k + b * BLOCK_K + i] = dq[i];
            }
        }
    }

    // 2. CPU reference: dequant w * a (a is row vector [k]).
    let a = deterministic(k, 7, 1.0);
    let mut cpu_out = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += a[i] * w_dequant[row * k + i];
        }
        cpu_out[row] = acc;
    }

    // 3. GPU pipeline.
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_q4_0_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0, "fused_q4_0 rc={}", rc);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_out.iter().filter(|x| x.is_finite()).count();
        println!("[q4_0-matmul] n={} k={} blocks/row={} max_diff={:.3e} finite={}/{}",
            n, k, n_blocks, max_diff, n_finite, n);
        assert_eq!(n_finite, n, "non-finite values in Q4_0 matmul output");
        assert!(max_diff < 1e-4,
            "Q4_0 matmul diverged from CPU dequant reference ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev);
        aether_dev_free_u8(w_dev);
        aether_dev_free_f32(o_dev);
    }
}

#[test]
fn q4_0_matches_at_v2_lite_d_model() {
    // Bigger shape closer to V2-Lite's attn_q matmul: n=3072 (n_heads *
    // qk_head_dim), k=2048 (d_model).  Both multiples of 32 so the Q4_0
    // block alignment is clean.
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 3072;
    let k = 2048;
    let n_blocks = k / BLOCK_K;
    let w_target = deterministic(n * k, 23, 0.05);
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let mut chunk = [0f32; BLOCK_K];
            for i in 0..BLOCK_K {
                chunk[i] = w_target[row * k + b * BLOCK_K + i];
            }
            let bytes = pack_q4_0_block(&chunk);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            let dq = unpack_q4_0_block(&bytes);
            for i in 0..BLOCK_K {
                w_dequant[row * k + b * BLOCK_K + i] = dq[i];
            }
        }
    }
    let a = deterministic(k, 29, 1.0);
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
        let rc = aether_op_fused_q4_0_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_out.iter().filter(|x| x.is_finite()).count();
        println!("[q4_0-v2lite] n={} k={} max_diff={:.3e} finite={}/{}",
            n, k, max_diff, n_finite, n);
        assert_eq!(n_finite, n);
        // Bigger accumulation → looser tolerance, still tight.
        assert!(max_diff < 5e-3,
            "Q4_0 matmul at V2-Lite shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev);
        aether_dev_free_u8(w_dev);
        aether_dev_free_f32(o_dev);
    }
}
