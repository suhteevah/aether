//! IQ4_XS MoE expert-variant matmul parity (FR-17-extra-moe-quant-dispatch-iq4xs).
//!
//! The expert kernel decodes IQ4_XS super-blocks (136 bytes / 256 elems) from
//! ONE slice of a concatenated MoE expert weight buffer.  The slice begins
//! at `expert_idx * n_out * blocks_per_row * 136` bytes from `w_base`.
//!
//! Parity strategy: GPU-vs-GPU, mirroring the IQ3_S expert parity test.  The
//! standalone `fused_iq4_xs_matmul_seq1` kernel run against expert E's slice
//! in isolation must produce the same output (within f32 reduction-order
//! tolerance) as the expert kernel run against the full concatenated buffer
//! at `expert_idx = E`.
//!
//! When `blocks_per_row == 1` the per-output-row summation order is identical
//! across both kernels → bit-identical.  When `blocks_per_row > 1` the
//! standalone sums sequentially in one thread while the expert kernel
//! parallelises across threads + warp-reduces, so the result differs by f32
//! non-associativity rounding (typically <1e-2 at GLM-class shapes).
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::aether_f32_to_f16;
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_iq4_xs_matmul_seq1_cuda,
    aether_op_fused_iq4_xs_expert_matmul_seq1_cuda,
};

const QK: usize = 256;
const BYTES_PER_BLOCK: usize = 136;

fn next_u32(state: &mut u64) -> u32 {
    let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
    *state = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z >> 32) ^ z) as u32
}

/// Build a synthetic 136-byte IQ4_XS block — random scales_h/scales_l/qs,
/// small d.  Any 136 bytes decode deterministically because the kvalues
/// codebook is a fixed 16-entry lookup; both kernels will see the same bytes.
fn synth_block(seed: u64) -> [u8; BYTES_PER_BLOCK] {
    let mut state = seed.wrapping_add(1);
    let mut bytes = [0u8; BYTES_PER_BLOCK];
    let d = 0.05f32;
    let d_bits = aether_f32_to_f16(d) as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;
    // 2-byte scales_h
    let sh = next_u32(&mut state) as u16;
    bytes[2] = (sh & 0xFF) as u8;
    bytes[3] = ((sh >> 8) & 0xFF) as u8;
    // 4-byte scales_l
    for i in 0..4 { bytes[4 + i] = (next_u32(&mut state) & 0xFF) as u8; }
    // 128-byte qs
    for i in 0..128 { bytes[8 + i] = (next_u32(&mut state) & 0xFF) as u8; }
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
    let rc = aether_op_fused_iq4_xs_matmul_seq1_cuda(
        a_dev, w_dev, o_dev, n_out as i32, n_blocks as i32);
    assert_eq!(rc, 0, "standalone IQ4_XS rc={}", rc);
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
    let rc = aether_op_fused_iq4_xs_expert_matmul_seq1_cuda(
        a_dev, w_dev, o_dev, n_out as i32, blocks_per_row as i32, expert_idx as i32);
    assert_eq!(rc, 0, "expert IQ4_XS rc={}", rc);
    aether_dev_sync();
    let mut out = vec![0f32; n_out];
    aether_dev_d2h_f32(o_dev, out.as_mut_ptr() as i64, n_out as i32);
    aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    out
}

#[test]
fn iq4_xs_expert_matches_standalone_small() {
    assert_eq!(aether_dev_init(), 0);
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
        let max_diff = y_ref.iter().zip(y_test.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = y_test.iter().filter(|v| v.is_finite()).count();
        println!("[iq4_xs-expert small] e={} max_diff={:.3e} finite={}/{}",
            e, max_diff, n_finite, n_out);
        assert_eq!(n_finite, n_out, "non-finite output for expert {}", e);
        assert_eq!(max_diff, 0.0,
            "expert kernel (e={}) diverged from standalone ({:.3e})", e, max_diff);
    }
}

#[test]
fn iq4_xs_expert_matches_standalone_glm_class() {
    assert_eq!(aether_dev_init(), 0);
    // GLM-4.7-flash MoE gate/up shape: n_in=2048, n_out=1536.
    let n_experts = 2;
    let n_out = 1536;
    let blocks_per_row = 2048 / QK;  // = 8
    let k = blocks_per_row * QK;
    let concat = build_concat_experts(n_experts, n_out, blocks_per_row, 19);
    let x = deterministic(k, 23, 1.0);
    for e in 0..n_experts {
        let slice = expert_slice(&concat, e, n_out, blocks_per_row);
        let y_ref  = unsafe { run_standalone(&x, slice,  n_out, blocks_per_row) };
        let y_test = unsafe { run_expert    (&x, &concat, n_out, blocks_per_row, e) };
        let max_diff = y_ref.iter().zip(y_test.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = y_test.iter().filter(|v| v.is_finite()).count();
        println!("[iq4_xs-expert glm] e={} n={} k={} max_diff={:.3e} finite={}/{}",
            e, n_out, k, max_diff, n_finite, n_out);
        assert_eq!(n_finite, n_out, "non-finite output for expert {}", e);
        assert!(max_diff < 1e-2,
            "expert kernel (e={}) diverged at GLM shape ({:.3e})", e, max_diff);
    }
}
