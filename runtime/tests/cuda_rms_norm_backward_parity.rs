//! matt-voice FR-18.6-real leg 2 — RMSNorm backward GPU parity.
//!
//! roadmap: P18
//!
//! qwen3 (and every Llama-family) block applies RMSNorm twice per layer; the
//! pipeline-parallel trainer must backprop through it. cuda.rs had RMSNorm
//! forward but no backward — this verifies the new `rms_norm_bwd_dx` +
//! `rms_norm_bwd_gamma` kernels against a from-scratch CPU reference on
//! synthetic data (no GGUF needed, runs on any CUDA box).
//!
//! Forward: inv = 1/sqrt(mean(x^2)+eps); y[i] = x[i]*inv*gamma[i].
//! Backward (g[i]=dy[i]*gamma[i], dot=sum g[i]*x[i]):
//!   dx[j]     = inv*(g[j] - x[j]*dot*inv^2/d)
//!   dgamma[j] = sum_rows dy[r,j]*x[r,j]*inv(r)

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_rms_norm_backward_dx_f32_cuda,
    aether_op_rms_norm_backward_gamma_f32_cuda,
};

/// Deterministic pseudo-random fill in [-1, 1).
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n)
        .map(|_| {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn cpu_rms_norm_backward(
    x: &[f32], gamma: &[f32], dy: &[f32], eps: f32, rows: usize, d: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut dx = vec![0.0f32; rows * d];
    let mut dgamma = vec![0.0f32; d];
    for r in 0..rows {
        let xr = &x[r * d..(r + 1) * d];
        let dyr = &dy[r * d..(r + 1) * d];
        let sumsq: f32 = xr.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sumsq / d as f32 + eps).sqrt();
        let dot: f32 = (0..d).map(|i| dyr[i] * gamma[i] * xr[i]).sum();
        let inv2 = inv * inv;
        for j in 0..d {
            let g = dyr[j] * gamma[j];
            dx[r * d + j] = inv * (g - xr[j] * dot * inv2 / d as f32);
            dgamma[j] += dyr[j] * xr[j] * inv;
        }
    }
    (dx, dgamma)
}

#[test]
fn rms_norm_backward_matches_cpu() {
    aether_dev_init();
    let rows = 5usize;
    let d = 320usize; // AetherLM-Tiny width; not 256-aligned on purpose
    let eps = 1e-5f32;

    let x = fill(1, rows * d);
    let gamma = fill(2, d);
    let dy = fill(3, rows * d);

    let (cpu_dx, cpu_dg) = cpu_rms_norm_backward(&x, &gamma, &dy, eps, rows, d);

    // Device buffers.
    let dx_buf = aether_dev_alloc_f32((rows * d) as c_int);
    let g_buf = aether_dev_alloc_f32(d as c_int);
    let dy_buf = aether_dev_alloc_f32((rows * d) as c_int);
    let out_dx = aether_dev_alloc_f32((rows * d) as c_int);
    let inv_buf = aether_dev_alloc_f32(rows as c_int);
    let dg_buf = aether_dev_alloc_f32(d as c_int);
    assert!(dx_buf >= 0 && g_buf >= 0 && dy_buf >= 0 && out_dx >= 0 && inv_buf >= 0 && dg_buf >= 0);

    unsafe {
        aether_dev_h2d_f32(x.as_ptr() as i64, dx_buf, (rows * d) as c_int);
        aether_dev_h2d_f32(gamma.as_ptr() as i64, g_buf, d as c_int);
        aether_dev_h2d_f32(dy.as_ptr() as i64, dy_buf, (rows * d) as c_int);

        let rc1 = aether_op_rms_norm_backward_dx_f32_cuda(
            dx_buf, g_buf, dy_buf, out_dx, inv_buf, eps, rows as c_int, d as c_int);
        assert_eq!(rc1, 0, "rms_norm_backward_dx returned {}", rc1);
        let rc2 = aether_op_rms_norm_backward_gamma_f32_cuda(
            dx_buf, dy_buf, inv_buf, dg_buf, rows as c_int, d as c_int);
        assert_eq!(rc2, 0, "rms_norm_backward_gamma returned {}", rc2);
        aether_dev_sync();
    }

    let mut gpu_dx = vec![0.0f32; rows * d];
    let mut gpu_dg = vec![0.0f32; d];
    unsafe {
        aether_dev_d2h_f32(out_dx, gpu_dx.as_mut_ptr() as i64, (rows * d) as c_int);
        aether_dev_d2h_f32(dg_buf, gpu_dg.as_mut_ptr() as i64, d as c_int);
        aether_dev_sync();
        aether_dev_free_f32(dx_buf); aether_dev_free_f32(g_buf);
        aether_dev_free_f32(dy_buf); aether_dev_free_f32(out_dx);
        aether_dev_free_f32(inv_buf); aether_dev_free_f32(dg_buf);
    }

    let max_dx = gpu_dx.iter().zip(&cpu_dx).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_dg = gpu_dg.iter().zip(&cpu_dg).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("[rms_norm_bwd parity] rows={} d={} max|dx diff|={:.3e} max|dgamma diff|={:.3e}",
        rows, d, max_dx, max_dg);
    assert!(max_dx < 1e-4, "dx parity: {:.3e} >= 1e-4", max_dx);
    assert!(max_dg < 1e-4, "dgamma parity: {:.3e} >= 1e-4", max_dg);
}
