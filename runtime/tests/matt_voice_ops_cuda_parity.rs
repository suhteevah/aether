//! Verify the new matt-voice CUDA kernels (RMSNorm, RoPE, GQA, SiLU,
//! mul_inplace, add_inplace, bias_add) produce the same outputs as
//! the CPU reference implementations in `ops::*` within float
//! tolerance.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_op_rms_norm_f32_cuda, aether_op_rope_apply_f32_cuda,
    aether_op_gqa_repeat_kv_f32_cuda, aether_op_silu_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_bias_add_f32_cuda,
};
use aether_rt::ops as cpu_ops;

fn close_all(gpu: &[f32], cpu: &[f32], tol: f32) -> bool {
    if gpu.len() != cpu.len() { return false; }
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (&g, &c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (g - c).abs();
        if d > worst { worst = d; worst_i = i; }
    }
    if worst >= tol {
        eprintln!("[mismatch] worst at {}: gpu={} cpu={} diff={}", worst_i,
                  gpu[worst_i], cpu[worst_i], worst);
        return false;
    }
    true
}

#[test]
fn rms_norm_cuda_matches_cpu() {
    unsafe {
        aether_dev_init();
        let rows = 4usize;
        let d = 64usize;
        let x: Vec<f32> = (0..rows * d).map(|i| ((i as f32) * 0.013 - 0.5)).collect();
        let gamma: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let mut cpu_out = vec![0.0f32; rows * d];
        cpu_ops::rms_norm_f32(x.as_ptr(), gamma.as_ptr(), 1e-5,
            cpu_out.as_mut_ptr(), rows, d);

        let dx = aether_dev_alloc_f32((rows * d) as c_int);
        let dg = aether_dev_alloc_f32(d as c_int);
        let dy = aether_dev_alloc_f32((rows * d) as c_int);
        aether_dev_h2d_f32(x.as_ptr() as i64, dx, (rows * d) as c_int);
        aether_dev_h2d_f32(gamma.as_ptr() as i64, dg, d as c_int);
        let rc = aether_op_rms_norm_f32_cuda(dx, dg, dy, 1e-5, rows as c_int, d as c_int);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0.0f32; rows * d];
        aether_dev_d2h_f32(dy, gpu_out.as_mut_ptr() as i64, (rows * d) as c_int);
        aether_dev_free_f32(dx); aether_dev_free_f32(dg); aether_dev_free_f32(dy);

        assert!(close_all(&gpu_out, &cpu_out, 1e-4),
            "RMSNorm GPU vs CPU mismatch");
    }
}

#[test]
fn rope_cuda_matches_cpu() {
    unsafe {
        aether_dev_init();
        let seq = 4usize;
        let n_heads = 8usize;
        let head_dim = 16usize;
        let base = 10000.0f32;
        let pos_start = 0usize;
        let n = seq * n_heads * head_dim;

        let mut x_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01) - 0.3).collect();
        let x_orig = x_cpu.clone();
        cpu_ops::rope_apply_f32(x_cpu.as_mut_ptr(), seq, n_heads, head_dim, base, pos_start);

        let dx = aether_dev_alloc_f32(n as c_int);
        aether_dev_h2d_f32(x_orig.as_ptr() as i64, dx, n as c_int);
        let rc = aether_op_rope_apply_f32_cuda(dx, seq as c_int, n_heads as c_int,
            head_dim as c_int, base, pos_start as c_int);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut x_gpu = vec![0.0f32; n];
        aether_dev_d2h_f32(dx, x_gpu.as_mut_ptr() as i64, n as c_int);
        aether_dev_free_f32(dx);

        assert!(close_all(&x_gpu, &x_cpu, 1e-4), "RoPE GPU vs CPU mismatch");
    }
}

#[test]
fn gqa_repeat_cuda_matches_cpu() {
    unsafe {
        aether_dev_init();
        let seq = 3usize;
        let n_kv = 2usize;
        let head_dim = 8usize;
        let n_q = 6usize;  // factor 3
        let kv_in: Vec<f32> = (0..seq * n_kv * head_dim).map(|i| (i as f32) * 0.1).collect();
        let mut cpu_out = vec![0.0f32; seq * n_q * head_dim];
        cpu_ops::gqa_repeat_kv_f32(kv_in.as_ptr(), cpu_out.as_mut_ptr(),
            seq, n_kv, head_dim, n_q);

        let dx = aether_dev_alloc_f32((seq * n_kv * head_dim) as c_int);
        let dy = aether_dev_alloc_f32((seq * n_q * head_dim) as c_int);
        aether_dev_h2d_f32(kv_in.as_ptr() as i64, dx, (seq * n_kv * head_dim) as c_int);
        aether_op_gqa_repeat_kv_f32_cuda(dx, dy, seq as c_int, n_kv as c_int,
            head_dim as c_int, n_q as c_int);
        aether_dev_sync();
        let mut gpu_out = vec![0.0f32; seq * n_q * head_dim];
        aether_dev_d2h_f32(dy, gpu_out.as_mut_ptr() as i64, (seq * n_q * head_dim) as c_int);
        aether_dev_free_f32(dx); aether_dev_free_f32(dy);

        assert!(close_all(&gpu_out, &cpu_out, 1e-6), "GQA repeat mismatch");
    }
}

#[test]
fn silu_cuda_matches_cpu() {
    unsafe {
        aether_dev_init();
        let n = 256usize;
        let mut x_cpu: Vec<f32> = (0..n).map(|i| ((i as f32) - 128.0) * 0.05).collect();
        let x_orig = x_cpu.clone();
        cpu_ops::silu_f32(x_cpu.as_mut_ptr(), n);

        let dx = aether_dev_alloc_f32(n as c_int);
        aether_dev_h2d_f32(x_orig.as_ptr() as i64, dx, n as c_int);
        aether_op_silu_f32_cuda(dx, n as c_int);
        aether_dev_sync();
        let mut x_gpu = vec![0.0f32; n];
        aether_dev_d2h_f32(dx, x_gpu.as_mut_ptr() as i64, n as c_int);
        aether_dev_free_f32(dx);

        assert!(close_all(&x_gpu, &x_cpu, 1e-5), "SiLU mismatch");
    }
}

#[test]
fn mul_add_inplace_bias_add_cuda() {
    unsafe {
        aether_dev_init();
        let n = 64usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..n).map(|i| (n - i) as f32 * 0.07).collect();

        // mul_inplace: x *= y
        let mut cpu = a.clone();
        for i in 0..n { cpu[i] *= b[i]; }
        let dx = aether_dev_alloc_f32(n as c_int);
        let dy = aether_dev_alloc_f32(n as c_int);
        aether_dev_h2d_f32(a.as_ptr() as i64, dx, n as c_int);
        aether_dev_h2d_f32(b.as_ptr() as i64, dy, n as c_int);
        aether_op_mul_inplace_f32_cuda(dx, dy, n as c_int);
        aether_dev_sync();
        let mut gpu = vec![0.0f32; n];
        aether_dev_d2h_f32(dx, gpu.as_mut_ptr() as i64, n as c_int);
        assert!(close_all(&gpu, &cpu, 1e-5), "mul_inplace mismatch");

        // add_inplace: x += y
        let mut cpu2 = a.clone();
        for i in 0..n { cpu2[i] += b[i]; }
        aether_dev_h2d_f32(a.as_ptr() as i64, dx, n as c_int);
        aether_op_add_inplace_f32_cuda(dx, dy, n as c_int);
        aether_dev_sync();
        aether_dev_d2h_f32(dx, gpu.as_mut_ptr() as i64, n as c_int);
        assert!(close_all(&gpu, &cpu2, 1e-5), "add_inplace mismatch");

        // bias_add: x[r, c] += bias[c]
        let rows = 8usize;
        let cols = 8usize;
        let mat: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
        let bias: Vec<f32> = (0..cols).map(|c| (c as f32) * 10.0).collect();
        let mut cpu3 = mat.clone();
        for r in 0..rows {
            for c in 0..cols { cpu3[r * cols + c] += bias[c]; }
        }
        let dm = aether_dev_alloc_f32((rows * cols) as c_int);
        let db = aether_dev_alloc_f32(cols as c_int);
        aether_dev_h2d_f32(mat.as_ptr() as i64, dm, (rows * cols) as c_int);
        aether_dev_h2d_f32(bias.as_ptr() as i64, db, cols as c_int);
        aether_op_bias_add_f32_cuda(dm, db, rows as c_int, cols as c_int);
        aether_dev_sync();
        let mut gpu3 = vec![0.0f32; rows * cols];
        aether_dev_d2h_f32(dm, gpu3.as_mut_ptr() as i64, (rows * cols) as c_int);
        assert!(close_all(&gpu3, &cpu3, 1e-5), "bias_add mismatch");

        aether_dev_free_f32(dx); aether_dev_free_f32(dy);
        aether_dev_free_f32(dm); aether_dev_free_f32(db);
    }
}
