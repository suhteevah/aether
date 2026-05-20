//! Compare Q6_K fused matmul kernel against CPU dequant + matmul for
//! blk.3.attn_v.weight specifically (where the autoregressive chain
//! produces NaN values).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q6_k, aether_op_matmul_f32,
};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

#[test]
#[ignore]
fn q6k_blk3_v_proj() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);

        // Diagnose blk.3.attn_v.weight specifically -- the V proj that
        // blew up to 2.19e9 in the autoregressive chain.
        let needle = b"blk.3.attn_v.weight";
        let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
        assert!(idx >= 0);
        assert_eq!(aether_gguf_get_tensor_dtype(h, idx), 14, "not Q6_K");
        let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
        let n_blocks = n_elems / 256;
        let n_rows = 512;  // d_out (V proj output dim)
        let n_cols = 3584; // d_in
        let blocks_per_row = n_cols / 256;
        assert_eq!(n_blocks, n_rows * blocks_per_row);
        let n_bytes = n_blocks * 210;
        let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;
        let bytes: Vec<u8> = std::slice::from_raw_parts(dptr, n_bytes).to_vec();

        // CPU dequant
        let mut w_dq = vec![0.0f32; n_elems];
        aether_dequant_q6_k(bytes.as_ptr() as *const c_void,
            w_dq.as_mut_ptr() as *mut c_void, n_blocks as c_int);

        let mut max_abs = 0.0f32;
        let mut sum = 0.0f64;
        for &v in &w_dq {
            let a = v.abs();
            if a > max_abs { max_abs = a; }
            sum += v as f64;
        }
        eprintln!("[blk.3.attn_v.weight CPU dequant]");
        eprintln!("  n_rows={}, n_cols={}, n_elems={}", n_rows, n_cols, n_elems);
        eprintln!("  max_abs={:.3e}, mean={:.3e}", max_abs, sum / n_elems as f64);
        eprintln!("  first 8 values: {:?}", &w_dq[..8]);

        // Now run the SAME computation using realistic input (small magnitude)
        // and see what cuBLAS-sgemm produces vs fused Q6_K v2.
        let k = n_cols;
        let n = n_rows;

        // Synthetic x_norm input similar to what the model produces (max ~5)
        let a_host: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.001 - 0.5) * 5.0).collect();

        // Path 1: CPU reference
        let mut out_cpu = vec![0.0f32; n];
        // out[ni] = sum_k a[k] * w_dq[ni, k]
        for ni in 0..n {
            let mut acc = 0.0f32;
            for k_i in 0..k {
                acc += a_host[k_i] * w_dq[ni * k + k_i];
            }
            out_cpu[ni] = acc;
        }
        let mut cpu_max = 0.0f32;
        for &v in &out_cpu { let a = v.abs(); if a > cpu_max { cpu_max = a; } }
        eprintln!("[CPU x @ W_v] max_abs={:.3e}, first 4: {:?}", cpu_max, &out_cpu[..4]);

        // Path 2: GPU fused Q6_K v2
        let d_a = aether_dev_alloc_f32(k as c_int);
        let d_w_u8 = aether_dev_alloc_u8(bytes.len() as c_int);
        let d_out = aether_dev_alloc_f32(n as c_int);
        aether_dev_h2d_f32(a_host.as_ptr() as i64, d_a, k as c_int);
        aether_dev_h2d_u8(bytes.as_ptr() as i64, d_w_u8, bytes.len() as c_int);
        let rc = aether_op_fused_q6k_matmul_seq1_v2_cuda(
            d_a, d_w_u8, d_out, n as c_int, blocks_per_row as c_int,
        );
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut out_gpu = vec![0.0f32; n];
        aether_dev_d2h_f32(d_out, out_gpu.as_mut_ptr() as i64, n as c_int);

        let mut gpu_max = 0.0f32;
        let mut nan_count = 0;
        for &v in &out_gpu {
            if v.is_nan() { nan_count += 1; continue; }
            let a = v.abs();
            if a > gpu_max { gpu_max = a; }
        }
        eprintln!("[GPU fused Q6_K v2] max_abs={:.3e}, nan={}, first 4: {:?}",
            gpu_max, nan_count, &out_gpu[..4]);

        // Diff
        let mut max_diff = 0.0f32;
        let mut bad = 0usize;
        for (i, (c, g)) in out_cpu.iter().zip(out_gpu.iter()).enumerate() {
            if g.is_nan() { bad += 1; continue; }
            let d = (c - g).abs();
            if d > max_diff { max_diff = d; }
            if d > 1e-3 { bad += 1; }
        }
        eprintln!("[diff] max_diff={:.3e}, bad={} out of {}", max_diff, bad, n);

        aether_dev_free_f32(d_a);
        aether_dev_free_u8(d_w_u8);
        aether_dev_free_f32(d_out);
        aether_gguf_close(h);
    }
}
