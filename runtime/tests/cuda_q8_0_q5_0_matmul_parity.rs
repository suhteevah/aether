//! Q8_0 + Q5_0 fused matmul parity (FR-17-extra-{q8_0,q5_0}-fwd).
//!
//! cnc's V2-Lite Q4_K_M uses Q8_0 for the dense ffn_down (d_in=10944)
//! and Q5_0 for half of the ffn_down_exps (d_in=1408).  Both formats
//! have 32-element blocks so they naturally handle d_ff dims that
//! aren't 256-aligned.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q8_0_matmul_seq1_cuda,
    aether_op_fused_q5_0_matmul_seq1_cuda,
};

const BLOCK_K: usize = 32;

// ---------- Q8_0 pack/unpack (34 bytes per 32-elem block) ----------

fn pack_q8_0_block(chunk: &[f32; 32]) -> [u8; 34] {
    let max_abs = chunk.iter().fold(0f32, |a, b| a.max(b.abs()));
    let d = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let mut out = [0u8; 34];
    let d_bits = unsafe { aether_f32_to_f16(d) } as u16;
    out[0] = (d_bits & 0xFF) as u8;
    out[1] = ((d_bits >> 8) & 0xFF) as u8;
    for i in 0..32 {
        let q = (chunk[i] / d).round().clamp(-127.0, 127.0) as i8;
        out[2 + i] = q as u8;
    }
    out
}

fn unpack_q8_0_block(bytes: &[u8; 34]) -> [f32; 32] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let mut out = [0f32; 32];
    for i in 0..32 {
        let q = bytes[2 + i] as i8;
        out[i] = d * (q as f32);
    }
    out
}

// ---------- Q5_0 pack/unpack (22 bytes per 32-elem block) ----------

fn pack_q5_0_block(chunk: &[f32; 32]) -> [u8; 22] {
    let max_abs = chunk.iter().fold(0f32, |a, b| a.max(b.abs()));
    let d = if max_abs > 0.0 { max_abs / 15.0 } else { 1.0 };
    let mut out = [0u8; 22];
    let d_bits = unsafe { aether_f32_to_f16(d) } as u16;
    out[0] = (d_bits & 0xFF) as u8;
    out[1] = ((d_bits >> 8) & 0xFF) as u8;
    let mut qh: u32 = 0;
    for k in 0..32 {
        let q = ((chunk[k] / d).round() + 16.0).clamp(0.0, 31.0) as u32;
        let lo4 = q & 0x0F;
        let hi1 = (q >> 4) & 0x01;
        if hi1 != 0 { qh |= 1 << k; }
        // Pack low nibble: elem k in [0, 16) → low nibble of ql[k]
        //                  elem k in [16,32) → high nibble of ql[k-16]
        if k < 16 {
            out[6 + k] = (out[6 + k] & 0xF0) | (lo4 as u8);
        } else {
            let idx = k - 16;
            out[6 + idx] = (out[6 + idx] & 0x0F) | ((lo4 as u8) << 4);
        }
    }
    out[2] = (qh & 0xFF) as u8;
    out[3] = ((qh >> 8) & 0xFF) as u8;
    out[4] = ((qh >> 16) & 0xFF) as u8;
    out[5] = ((qh >> 24) & 0xFF) as u8;
    out
}

fn unpack_q5_0_block(bytes: &[u8; 22]) -> [f32; 32] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let qh = (bytes[2] as u32)
        | ((bytes[3] as u32) << 8)
        | ((bytes[4] as u32) << 16)
        | ((bytes[5] as u32) << 24);
    let mut out = [0f32; 32];
    for i in 0..16 {
        let byte = bytes[6 + i];
        let q_lo = ((byte & 0x0F) as u32 | (((qh >> i) & 1) << 4)) as i32 - 16;
        let q_hi = (((byte >> 4) & 0x0F) as u32 | (((qh >> (i + 16)) & 1) << 4)) as i32 - 16;
        out[i]      = d * (q_lo as f32);
        out[i + 16] = d * (q_hi as f32);
    }
    out
}

// ---------- Helpers ----------

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

fn cpu_matmul(a: &[f32], w_dequant: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += a[i] * w_dequant[row * k + i];
        }
        out[row] = acc;
    }
    out
}

// ---------- Q8_0 tests ----------

#[test]
fn q8_0_matches_cpu_dequant() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 64;
    let k = 256;
    let n_blocks = k / BLOCK_K;
    const BPB: usize = 34;

    let w_target = deterministic(n * k, 11, 0.3);
    let mut w_packed = vec![0u8; n * n_blocks * BPB];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let mut chunk = [0f32; BLOCK_K];
            for i in 0..BLOCK_K { chunk[i] = w_target[row * k + b * BLOCK_K + i]; }
            let bytes = pack_q8_0_block(&chunk);
            let off = (row * n_blocks + b) * BPB;
            w_packed[off..off + BPB].copy_from_slice(&bytes);
            let dq = unpack_q8_0_block(&bytes);
            for i in 0..BLOCK_K { w_dequant[row * k + b * BLOCK_K + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 7, 1.0);
    let cpu_out = cpu_matmul(&a, &w_dequant, n, k);

    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BPB) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BPB) as i32);
        let rc = aether_op_fused_q8_0_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[q8_0] n={} k={} max_diff={:.3e}", n, k, max_diff);
        assert!(max_diff < 1e-4, "Q8_0 diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn q8_0_matches_at_v2_lite_ffn_down_dense() {
    // V2-Lite layer 0 dense ffn_down: n=2048 (d_model), k=10944 (d_ff).
    // The 10944 dim is NOT a multiple of 256 (Q4_K super-block) but IS a
    // multiple of 32 (Q8_0 block size).  This is the exact shape llama.cpp
    // picks Q8_0 for in V2-Lite Q4_K_M.
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 2048;
    let k = 10944;
    assert_eq!(k % BLOCK_K, 0);
    let n_blocks = k / BLOCK_K;
    const BPB: usize = 34;

    let w_target = deterministic(n * k, 31, 0.02);
    let mut w_packed = vec![0u8; n * n_blocks * BPB];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let mut chunk = [0f32; BLOCK_K];
            for i in 0..BLOCK_K { chunk[i] = w_target[row * k + b * BLOCK_K + i]; }
            let bytes = pack_q8_0_block(&chunk);
            let off = (row * n_blocks + b) * BPB;
            w_packed[off..off + BPB].copy_from_slice(&bytes);
            let dq = unpack_q8_0_block(&bytes);
            for i in 0..BLOCK_K { w_dequant[row * k + b * BLOCK_K + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 37, 1.0);
    let cpu_out = cpu_matmul(&a, &w_dequant, n, k);

    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BPB) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BPB) as i32);
        let rc = aether_op_fused_q8_0_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[q8_0-v2lite] n={} k={} max_diff={:.3e}", n, k, max_diff);
        assert!(max_diff < 1e-2,
            "Q8_0 at V2-Lite ffn_down shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

// ---------- Q5_0 tests ----------

#[test]
fn q5_0_matches_cpu_dequant() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 64;
    let k = 256;
    let n_blocks = k / BLOCK_K;
    const BPB: usize = 22;

    let w_target = deterministic(n * k, 13, 0.3);
    let mut w_packed = vec![0u8; n * n_blocks * BPB];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let mut chunk = [0f32; BLOCK_K];
            for i in 0..BLOCK_K { chunk[i] = w_target[row * k + b * BLOCK_K + i]; }
            let bytes = pack_q5_0_block(&chunk);
            let off = (row * n_blocks + b) * BPB;
            w_packed[off..off + BPB].copy_from_slice(&bytes);
            let dq = unpack_q5_0_block(&bytes);
            for i in 0..BLOCK_K { w_dequant[row * k + b * BLOCK_K + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 7, 1.0);
    let cpu_out = cpu_matmul(&a, &w_dequant, n, k);

    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BPB) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BPB) as i32);
        let rc = aether_op_fused_q5_0_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[q5_0] n={} k={} max_diff={:.3e}", n, k, max_diff);
        assert!(max_diff < 1e-4, "Q5_0 diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn q5_0_matches_at_v2_lite_expert_ffn_down() {
    // V2-Lite per-expert ffn_down: n=2048 (d_model), k=1408 (expert_ff_dim).
    // 1408 is NOT a multiple of 256 (Q4_K) but IS a multiple of 32 (Q5_0).
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 2048;
    let k = 1408;
    assert_eq!(k % BLOCK_K, 0);
    let n_blocks = k / BLOCK_K;
    const BPB: usize = 22;

    let w_target = deterministic(n * k, 41, 0.05);
    let mut w_packed = vec![0u8; n * n_blocks * BPB];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let mut chunk = [0f32; BLOCK_K];
            for i in 0..BLOCK_K { chunk[i] = w_target[row * k + b * BLOCK_K + i]; }
            let bytes = pack_q5_0_block(&chunk);
            let off = (row * n_blocks + b) * BPB;
            w_packed[off..off + BPB].copy_from_slice(&bytes);
            let dq = unpack_q5_0_block(&bytes);
            for i in 0..BLOCK_K { w_dequant[row * k + b * BLOCK_K + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 43, 1.0);
    let cpu_out = cpu_matmul(&a, &w_dequant, n, k);

    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BPB) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BPB) as i32);
        let rc = aether_op_fused_q5_0_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[q5_0-v2lite] n={} k={} max_diff={:.3e}", n, k, max_diff);
        assert!(max_diff < 5e-3,
            "Q5_0 at V2-Lite expert ffn_down shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}
