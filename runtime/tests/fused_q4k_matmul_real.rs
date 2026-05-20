//! Fused Q4_K matmul kernel: parity vs CPU reference + benchmark vs
//! cuBLAS (dequant + sgemm) on real Qwen2.5 matmul shapes.
//!
//! The fused kernel reads Q4_K bytes directly + dequants inline +
//! accumulates. Saves the full 4x dequant->f32 write + cuBLAS f32
//! read round-trip through VRAM.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m,
};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_matmul_f32_cuda,
    aether_op_dequant_q4_k_m_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

unsafe fn q4k_raw_bytes(h: i64, name: &str) -> (Vec<u8>, usize, usize) {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0, "tensor {} not found", name);
    assert_eq!(aether_gguf_get_tensor_dtype(h, idx), 12, "{} is not Q4_K", name);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let n_blocks_total = n_elems / 256;
    let n_bytes = n_blocks_total * 144;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;
    let bytes = std::slice::from_raw_parts(dptr, n_bytes).to_vec();
    (bytes, n_blocks_total, n_elems)
}

/// Run the fused-Q4K-matmul on one matmul shape and compare against
/// the cuBLAS reference (dequant -> sgemm). Returns (max_diff, mean_diff).
unsafe fn run_one(name: &str, n_rows: usize, n_cols: usize, h: i64) -> (f32, f32, u128, u128) {
    let (w_bytes, n_blocks_total, _n_elems) = q4k_raw_bytes(h, name);
    // The tensor has n_rows output rows, each row has k = n_cols quants = n_cols/256 super-blocks
    let blocks_per_row = n_cols / 256;
    assert_eq!(n_blocks_total, n_rows * blocks_per_row);
    let k = n_cols;
    let n = n_rows;

    // Synthetic activation: deterministic small values
    let a_host: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.001 - 0.5)).collect();

    // === Path 1: dequant ALL weights on host + cuBLAS sgemm with full f32 ===
    // Dequant on CPU + transpose to matmul layout for cuBLAS
    let mut w_dequant_natural = vec![0.0f32; n_rows * n_cols];
    aether_dequant_q4_k_m(w_bytes.as_ptr() as *const c_void,
        w_dequant_natural.as_mut_ptr() as *mut c_void, n_blocks_total as c_int);
    // Transpose [n, k] -> [k, n]
    let mut w_matmul = vec![0.0f32; n_rows * n_cols];
    for i_out in 0..n_rows {
        for i_in in 0..n_cols {
            w_matmul[i_in * n_rows + i_out] = w_dequant_natural[i_out * n_cols + i_in];
        }
    }

    let d_a = aether_dev_alloc_f32(k as c_int);
    let d_w = aether_dev_alloc_f32((n_rows * n_cols) as c_int);
    let d_out_cublas = aether_dev_alloc_f32(n as c_int);
    aether_dev_h2d_f32(a_host.as_ptr() as i64, d_a, k as c_int);
    aether_dev_h2d_f32(w_matmul.as_ptr() as i64, d_w, (n_rows * n_cols) as c_int);

    // Warmup
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

    // === Path 2: fused Q4K matmul ===
    let d_w_u8 = aether_dev_alloc_u8(w_bytes.len() as c_int);
    let d_out_fused = aether_dev_alloc_f32(n as c_int);
    aether_dev_h2d_u8(w_bytes.as_ptr() as i64, d_w_u8, w_bytes.len() as c_int);

    // Warmup
    aether_op_fused_q4k_matmul_seq1_cuda(d_a, d_w_u8, d_out_fused, n as c_int, blocks_per_row as c_int);
    aether_dev_sync();

    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        aether_op_fused_q4k_matmul_seq1_cuda(d_a, d_w_u8, d_out_fused, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();
    let fused_us = t0.elapsed().as_micros() / 10;

    let mut out_fused = vec![0.0f32; n];
    aether_dev_d2h_f32(d_out_fused, out_fused.as_mut_ptr() as i64, n as c_int);

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    for (a, b) in out_cublas.iter().zip(out_fused.iter()) {
        let d = (a - b).abs();
        if d > max_diff { max_diff = d; }
        sum_diff += d;
    }
    let mean_diff = sum_diff / n as f32;

    aether_dev_free_f32(d_a);
    aether_dev_free_f32(d_w);
    aether_dev_free_f32(d_out_cublas);
    aether_dev_free_f32(d_out_fused);
    aether_dev_free_u8(d_w_u8);

    (max_diff, mean_diff, cublas_us, fused_us)
}

#[test]
#[ignore]  // ~30s release with cuda
fn fused_q4k_matmul_real_qwen25_bench() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // (name, n_rows = d_out, n_cols = d_in) for all Q4_K matmuls
        // in a Qwen2.5 block.
        let shapes: &[(&str, usize, usize)] = &[
            ("blk.0.attn_q.weight",      3584,  3584),
            ("blk.0.attn_k.weight",      512,   3584),
            ("blk.0.attn_output.weight", 3584,  3584),
            ("blk.0.ffn_gate.weight",    18944, 3584),
            ("blk.0.ffn_up.weight",      18944, 3584),
        ];

        eprintln!("\n{:<32} {:>10} {:>10}  {:>10} {:>10} {:>8} {:>10}",
            "tensor", "n", "k", "cuBLAS_us", "fused_us", "speedup", "max_diff");
        eprintln!("{}", "-".repeat(96));
        for &(name, n_rows, n_cols) in shapes {
            let (max_d, mean_d, cublas_us, fused_us) = run_one(name, n_rows, n_cols, h);
            let speedup = cublas_us as f32 / fused_us.max(1) as f32;
            eprintln!("{:<32} {:>10} {:>10}  {:>10} {:>10} {:>7.2}x {:>10.3e}",
                name, n_rows, n_cols, cublas_us, fused_us, speedup, max_d);

            // Output values can differ slightly because the dot product
            // sum order differs between cuBLAS (sgemm) and our
            // accumulate-into-single-thread kernel. Tolerance: absolute
            // < 1e-2, relative < 1e-3 for a typical Qwen-magnitude value.
            assert!(max_d < 1e-2, "{} max_diff {} > 1e-2", name, max_d);
            assert!(mean_d < 1e-3, "{} mean_diff {} > 1e-3", name, mean_d);
        }
        eprintln!("{}", "-".repeat(96));
        eprintln!("(speedup > 1.0 means fused is faster than dequant->cuBLAS)");

        aether_gguf_close(h);
    }
}
