//! Q6_K MoE expert-variant matmul parity (item (d) — qwen3moe expert dtypes).
//!
//! The expert kernel `fused_q6_k_expert_matmul_seq1` decodes Q6_K super-blocks
//! (210 bytes / 256 elems) from ONE slice of a concatenated MoE expert weight
//! buffer.  The slice begins at `expert_idx * n_out * blocks_per_row * 210`
//! bytes from `w_base`.
//!
//! Parity strategy: GPU-vs-GPU.  The standalone `fused_q6k_matmul_seq1_v2`
//! kernel run against expert E's slice in isolation must produce the same
//! output (within f32 reduction-order tolerance) as the expert kernel run
//! against the full concatenated buffer at `expert_idx = E`.
//!
//! The two kernels use different launch geometries (dense = warp-per-row;
//! expert = CTA-per-row, 256-thread block-stride), so the per-row summation
//! order differs and the result matches only up to f32 non-associativity
//! rounding — checked with a small relative tolerance.
//!
//! Random ql/qh/scale bytes are fine — every byte pattern decodes to a
//! deterministic 256-element f32 vector via the standard Q6_K dequant, so we
//! don't need a "valid" GGUF block; we only need both kernels to see the same
//! input.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::aether_f32_to_f16;
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q6_k_expert_matmul_seq1_cuda,
};

const QK: usize = 256;
const BYTES_PER_BLOCK: usize = 210;

fn next_u32(state: &mut u64) -> u32 {
    let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
    *state = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z >> 32) ^ z) as u32
}

/// Build a synthetic 210-byte Q6_K block — random ql(128)/qh(64)/scales(16),
/// small d at bytes 208..210.
fn synth_block(seed: u64) -> [u8; BYTES_PER_BLOCK] {
    let mut state = seed.wrapping_add(1);
    let mut bytes = [0u8; BYTES_PER_BLOCK];
    for i in 0..192 { bytes[i] = (next_u32(&mut state) & 0xFF) as u8; } // ql + qh
    for i in 0..16  { bytes[192 + i] = (next_u32(&mut state) & 0xFF) as u8; } // scales (signed)
    let d = 0.05f32;
    let d_bits = aether_f32_to_f16(d) as u16;
    bytes[208] = (d_bits & 0xFF) as u8;
    bytes[209] = ((d_bits >> 8) & 0xFF) as u8;
    bytes
}

fn deterministic(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n).map(|_| {
        let u = (next_u32(&mut state) as f32) / 4_294_967_296.0;
        (u * 2.0 - 1.0) * scale
    }).collect()
}

fn build_concat_experts(
    n_experts: usize, n_out: usize, blocks_per_row: usize, seed_base: u64,
) -> Vec<u8> {
    let mut buf = vec![0u8; n_experts * n_out * blocks_per_row * BYTES_PER_BLOCK];
    for e in 0..n_experts {
        for row in 0..n_out {
            for b in 0..blocks_per_row {
                let off = ((e * n_out + row) * blocks_per_row + b) * BYTES_PER_BLOCK;
                let seed = seed_base
                    + (e as u64) * 1_000_000
                    + (row as u64) * 1_000
                    + (b as u64);
                let bytes = synth_block(seed);
                buf[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            }
        }
    }
    buf
}

fn expert_slice(concat: &[u8], expert_idx: usize, n_out: usize, blocks_per_row: usize) -> &[u8] {
    let stride = n_out * blocks_per_row * BYTES_PER_BLOCK;
    let off = expert_idx * stride;
    &concat[off..off + stride]
}

unsafe fn run_standalone(x_host: &[f32], w_host: &[u8], n_out: usize, n_blocks: usize) -> Vec<f32> {
    let k = n_blocks * QK;
    let a_dev = aether_dev_alloc_f32(k as i32);
    let w_dev = aether_dev_alloc_u8(w_host.len() as i32);
    let o_dev = aether_dev_alloc_f32(n_out as i32);
    aether_dev_h2d_f32(x_host.as_ptr() as i64, a_dev, k as i32);
    aether_dev_h2d_u8(w_host.as_ptr() as i64, w_dev, w_host.len() as i32);
    let rc = aether_op_fused_q6k_matmul_seq1_v2_cuda(
        a_dev, w_dev, o_dev, n_out as i32, n_blocks as i32);
    assert_eq!(rc, 0, "standalone Q6_K rc={}", rc);
    aether_dev_sync();
    let mut out = vec![0f32; n_out];
    aether_dev_d2h_f32(o_dev, out.as_mut_ptr() as i64, n_out as i32);
    aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    out
}

unsafe fn run_expert(
    x_host: &[f32], w_concat: &[u8],
    n_out: usize, blocks_per_row: usize, expert_idx: usize,
) -> Vec<f32> {
    let k = blocks_per_row * QK;
    let a_dev = aether_dev_alloc_f32(k as i32);
    let w_dev = aether_dev_alloc_u8(w_concat.len() as i32);
    let o_dev = aether_dev_alloc_f32(n_out as i32);
    aether_dev_h2d_f32(x_host.as_ptr() as i64, a_dev, k as i32);
    aether_dev_h2d_u8(w_concat.as_ptr() as i64, w_dev, w_concat.len() as i32);
    let rc = aether_op_fused_q6_k_expert_matmul_seq1_cuda(
        a_dev, w_dev, o_dev, n_out as i32, blocks_per_row as i32, expert_idx as i32);
    assert_eq!(rc, 0, "expert Q6_K rc={}", rc);
    aether_dev_sync();
    let mut out = vec![0f32; n_out];
    aether_dev_d2h_f32(o_dev, out.as_mut_ptr() as i64, n_out as i32);
    aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    out
}

/// max relative error: |a-b| / (1 + max(|a|,|b|)) over all rows.
fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs() / (1.0 + x.abs().max(y.abs())))
        .fold(0f32, f32::max)
}

#[test]
fn q6_k_expert_matches_standalone_small() {
    assert_eq!(aether_dev_init(), 0);
    // Tiny shape: 3 experts, n_out=32, blocks_per_row=1 (k=256).
    let n_experts = 3;
    let n_out = 32;
    let blocks_per_row = 1;
    let k = blocks_per_row * QK;
    let concat = build_concat_experts(n_experts, n_out, blocks_per_row, 7);
    let x = deterministic(k, 11, 1.0);
    for e in 0..n_experts {
        let slice = expert_slice(&concat, e, n_out, blocks_per_row);
        let y_ref  = unsafe { run_standalone(&x, slice,  n_out, blocks_per_row) };
        let y_test = unsafe { run_expert    (&x, &concat, n_out, blocks_per_row, e) };
        let rel = max_rel(&y_ref, &y_test);
        let n_finite = y_test.iter().filter(|v| v.is_finite()).count();
        println!("[q6_k-expert small] e={} max_rel={:.3e} finite={}/{}",
            e, rel, n_finite, n_out);
        assert_eq!(n_finite, n_out, "non-finite output for expert {}", e);
        assert!(rel < 1e-4,
            "expert kernel (e={}) diverged from standalone (rel {:.3e})", e, rel);
    }
}

#[test]
fn q6_k_expert_matches_standalone_qwen3moe_class() {
    assert_eq!(aether_dev_init(), 0);
    // Qwen3-30B-A3B MoE expert gate/up shape: n_in = d_model = 2048,
    // n_out = expert_ff_dim = 768.  Two experts proves the
    // expert_offset_blocks math works at scale.
    let n_experts = 2;
    let n_out = 768;
    let blocks_per_row = 2048 / QK;  // = 8
    let k = blocks_per_row * QK;
    let concat = build_concat_experts(n_experts, n_out, blocks_per_row, 19);
    let x = deterministic(k, 23, 1.0);
    for e in 0..n_experts {
        let slice = expert_slice(&concat, e, n_out, blocks_per_row);
        let y_ref  = unsafe { run_standalone(&x, slice,  n_out, blocks_per_row) };
        let y_test = unsafe { run_expert    (&x, &concat, n_out, blocks_per_row, e) };
        let rel = max_rel(&y_ref, &y_test);
        let n_finite = y_test.iter().filter(|v| v.is_finite()).count();
        println!("[q6_k-expert qwen3moe] e={} n={} k={} max_rel={:.3e} finite={}/{}",
            e, n_out, k, rel, n_finite, n_out);
        assert_eq!(n_finite, n_out, "non-finite output for expert {}", e);
        assert!(rel < 1e-3,
            "expert kernel (e={}) diverged at qwen3moe shape (rel {:.3e})", e, rel);
    }
}
