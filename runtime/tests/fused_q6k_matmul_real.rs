//! Fused Q6_K matmul v2 — parity vs CPU + bench on real Qwen2.5
//! blk.0.attn_v.weight (Q6_K, shape [512, 3584]).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q6_k,
};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_matmul_f32_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

#[test]
#[ignore]
fn fused_q6k_matmul_real_qwen25() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);

        let shapes: &[(&str, usize, usize)] = &[
            ("blk.0.attn_v.weight",   512,    3584),   // Q6_K, V proj
            ("blk.0.ffn_down.weight", 3584,   18944),  // Q6_K, ffn_down
            ("output.weight",         152064, 3584),   // Q6_K, lm_head -- big!
        ];

        eprintln!("\n{:<28} {:>7} {:>6}  {:>9} {:>9}  {:>6}  {:>10}",
            "tensor", "n", "k", "cuBLAS_us", "v2_us", "speedup", "max_diff");
        eprintln!("{}", "-".repeat(90));

        for &(name, n_rows, n_cols) in shapes {
            let needle = name.as_bytes();
            let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
            assert!(idx >= 0);
            assert_eq!(aether_gguf_get_tensor_dtype(h, idx), 14, "{} not Q6_K", name);
            let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
            let n_blocks_total = n_elems / 256;
            let blocks_per_row = n_cols / 256;
            assert_eq!(n_blocks_total, n_rows * blocks_per_row);
            let n_bytes = n_blocks_total * 210;
            let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;
            let w_bytes: Vec<u8> = std::slice::from_raw_parts(dptr, n_bytes).to_vec();

            let k = n_cols;
            let n = n_rows;

            let a_host: Vec<f32> = (0..k).map(|i| (i as f32) * 0.001 - 0.5).collect();

            // === Path 1: CPU dequant + transpose + cuBLAS ===
            let mut w_natural = vec![0.0f32; n_rows * n_cols];
            aether_dequant_q6_k(w_bytes.as_ptr() as *const c_void,
                w_natural.as_mut_ptr() as *mut c_void, n_blocks_total as c_int);
            let mut w_matmul = vec![0.0f32; n_rows * n_cols];
            for i_out in 0..n_rows {
                for i_in in 0..n_cols {
                    w_matmul[i_in * n_rows + i_out] = w_natural[i_out * n_cols + i_in];
                }
            }

            let d_a = aether_dev_alloc_f32(k as c_int);
            let d_w = aether_dev_alloc_f32((n_rows * n_cols) as c_int);
            let d_out_cublas = aether_dev_alloc_f32(n as c_int);
            aether_dev_h2d_f32(a_host.as_ptr() as i64, d_a, k as c_int);
            aether_dev_h2d_f32(w_matmul.as_ptr() as i64, d_w, (n_rows * n_cols) as c_int);

            aether_op_matmul_f32_cuda(d_a, d_w, d_out_cublas, 1, k as c_int, n as c_int);
            aether_dev_sync();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                aether_op_matmul_f32_cuda(d_a, d_w, d_out_cublas, 1, k as c_int, n as c_int);
            }
            aether_dev_sync();
            let cublas_us = t0.elapsed().as_micros() / 10;

            let mut out_cublas = vec![0.0f32; n];
            aether_dev_d2h_f32(d_out_cublas, out_cublas.as_mut_ptr() as i64, n as c_int);

            // === Path 2: fused Q6_K v2 ===
            let d_w_u8 = aether_dev_alloc_u8(w_bytes.len() as c_int);
            let d_out_v2 = aether_dev_alloc_f32(n as c_int);
            aether_dev_h2d_u8(w_bytes.as_ptr() as i64, d_w_u8, w_bytes.len() as c_int);

            aether_op_fused_q6k_matmul_seq1_v2_cuda(d_a, d_w_u8, d_out_v2,
                n as c_int, blocks_per_row as c_int);
            aether_dev_sync();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                aether_op_fused_q6k_matmul_seq1_v2_cuda(d_a, d_w_u8, d_out_v2,
                    n as c_int, blocks_per_row as c_int);
            }
            aether_dev_sync();
            let v2_us = t0.elapsed().as_micros() / 10;

            let mut out_v2 = vec![0.0f32; n];
            aether_dev_d2h_f32(d_out_v2, out_v2.as_mut_ptr() as i64, n as c_int);

            let mut max_diff = 0.0f32;
            for (a, b) in out_cublas.iter().zip(out_v2.iter()) {
                let d = (a - b).abs();
                if d > max_diff { max_diff = d; }
            }
            let speedup = cublas_us as f32 / v2_us.max(1) as f32;
            eprintln!("{:<28} {:>7} {:>6}  {:>9} {:>9}  {:>5.2}x  {:>10.3e}",
                name, n_rows, n_cols, cublas_us, v2_us, speedup, max_diff);

            assert!(max_diff < 1e-2, "{} v2 max_diff {} > 1e-2", name, max_diff);

            aether_dev_free_f32(d_a);
            aether_dev_free_f32(d_w);
            aether_dev_free_f32(d_out_cublas);
            aether_dev_free_f32(d_out_v2);
            aether_dev_free_u8(d_w_u8);
        }

        eprintln!("{}", "-".repeat(90));
        aether_gguf_close(h);
    }
}
