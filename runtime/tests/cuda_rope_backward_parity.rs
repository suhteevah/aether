//! matt-voice FR-18.6-real leg 2 — RoPE backward GPU parity.
//!
//! roadmap: P18
//!
//! RoPE rotates each (i, i+head_dim/2) pair by R(theta)=[[c,-s],[s,c]]. It is
//! orthogonal and parameter-free, so backward applies the transpose R^T. Two
//! checks:
//!   1. Round-trip: rope_backward(rope_forward(x)) == x (R^T R = I). For an
//!      orthogonal linear op this is exactly the statement that the backward
//!      is the correct adjoint.
//!   2. Direct: GPU rope_backward(dy) matches a from-scratch CPU transpose.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_rope_apply_f32_cuda,
    aether_op_rope_apply_backward_f32_cuda,
};

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }).collect()
}

/// CPU transpose rotation: dx = R^T dy.
fn cpu_rope_backward(dy: &[f32], seq: usize, n_heads: usize, head_dim: usize, base: f32, pos_start: i32) -> Vec<f32> {
    let mut dx = dy.to_vec();
    let hd_half = head_dim / 2;
    for t in 0..seq {
        for h in 0..n_heads {
            let base_off = (t * n_heads + h) * head_dim;
            for i in 0..hd_half {
                let pos = (t as i32 + pos_start) as f32;
                let exp = -2.0 * i as f32 / head_dim as f32;
                let theta = pos * base.powf(exp);
                let (c, s) = (theta.cos(), theta.sin());
                let (i0, i1) = (base_off + i, base_off + i + hd_half);
                let (y0, y1) = (dy[i0], dy[i1]);
                dx[i0] = y0 * c + y1 * s;
                dx[i1] = -y0 * s + y1 * c;
            }
        }
    }
    dx
}

#[test]
fn rope_backward_matches_cpu_and_round_trips() {
    aether_dev_init();
    let seq = 4usize;
    let n_heads = 5usize;
    let head_dim = 64usize;
    let base = 10000.0f32;
    let pos_start = 0i32;
    let n = seq * n_heads * head_dim;

    let x = fill(7, n);

    // --- Check 2: GPU backward vs CPU transpose on an independent dy.
    let dy = fill(11, n);
    let cpu_dx = cpu_rope_backward(&dy, seq, n_heads, head_dim, base, pos_start);

    let g_buf = aether_dev_alloc_f32(n as c_int);
    assert!(g_buf >= 0);
    unsafe {
        aether_dev_h2d_f32(dy.as_ptr() as i64, g_buf, n as c_int);
        let rc = aether_op_rope_apply_backward_f32_cuda(
            g_buf, seq as c_int, n_heads as c_int, head_dim as c_int, base, pos_start);
        assert_eq!(rc, 0, "rope_apply_backward returned {}", rc);
        aether_dev_sync();
    }
    let mut gpu_dx = vec![0.0f32; n];
    unsafe { aether_dev_d2h_f32(g_buf, gpu_dx.as_mut_ptr() as i64, n as c_int); aether_dev_sync(); }
    let max_dx = gpu_dx.iter().zip(&cpu_dx).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    // --- Check 1: round-trip forward then backward recovers x.
    let rt_buf = aether_dev_alloc_f32(n as c_int);
    assert!(rt_buf >= 0);
    unsafe {
        aether_dev_h2d_f32(x.as_ptr() as i64, rt_buf, n as c_int);
        aether_op_rope_apply_f32_cuda(rt_buf, seq as c_int, n_heads as c_int, head_dim as c_int, base, pos_start);
        aether_op_rope_apply_backward_f32_cuda(rt_buf, seq as c_int, n_heads as c_int, head_dim as c_int, base, pos_start);
        aether_dev_sync();
    }
    let mut rt = vec![0.0f32; n];
    unsafe {
        aether_dev_d2h_f32(rt_buf, rt.as_mut_ptr() as i64, n as c_int);
        aether_dev_sync();
        aether_dev_free_f32(g_buf); aether_dev_free_f32(rt_buf);
    }
    let max_rt = rt.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    eprintln!("[rope_bwd parity] seq={} heads={} hd={} max|dx-cpu|={:.3e} round-trip max|diff|={:.3e}",
        seq, n_heads, head_dim, max_dx, max_rt);
    assert!(max_dx < 1e-4, "dx vs CPU: {:.3e} >= 1e-4", max_dx);
    assert!(max_rt < 1e-4, "round-trip identity: {:.3e} >= 1e-4", max_rt);
}
