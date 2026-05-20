//! Per-matmul-shape microbenchmark. With CUDA graphs we've eliminated
//! launch overhead -- now the GPU compute dominates. Measure each
//! Qwen2.5 matmul shape in isolation so we know which kernels need
//! kernel-level (SASS) tuning.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::time::Instant;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_alloc_u8,
    aether_dev_h2d_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
};

const D_MODEL: usize = 3584;
const D_KV: usize = 512;
const D_FF: usize = 18944;
const VOCAB: usize = 152064;

fn random_q4k(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let n_bytes = n_outputs * blocks_per_row * 144;
    let mut out = vec![0u8; n_bytes];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 144;
        out[off + 0] = 0x47; out[off + 1] = 0x21;
        out[off + 2] = 0x47; out[off + 3] = 0x19;
    }
    out
}
fn random_q6k(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let n_bytes = n_outputs * blocks_per_row * 210;
    let mut out = vec![0u8; n_bytes];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    // For Q6_K, d is at byte offset 208..210.
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 210;
        out[off + 208] = 0x47; out[off + 209] = 0x21;  // d = 0.01
    }
    out
}

unsafe fn bench_q4k(label: &str, n: usize, blocks_per_row: usize, n_iters: usize) {
    let k = blocks_per_row * 256;
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 1e-3 - 1.0).collect();
    let w = random_q4k(n, blocks_per_row, 0xCAFE);
    let d_a = aether_dev_alloc_f32(k as c_int);
    let d_w = aether_dev_alloc_u8(w.len() as c_int);
    let d_out = aether_dev_alloc_f32(n as c_int);
    aether_dev_h2d_f32(a.as_ptr() as i64, d_a, k as c_int);
    aether_dev_h2d_u8(w.as_ptr() as i64, d_w, w.len() as c_int);

    // Warmup
    for _ in 0..3 {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_w, d_out, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();

    let t = Instant::now();
    for _ in 0..n_iters {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_w, d_out, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();
    let elapsed = t.elapsed().as_secs_f64();
    let per_iter_us = elapsed * 1e6 / n_iters as f64;
    let bytes = n * blocks_per_row * 144;
    let gb_s = (bytes as f64 / 1e9) / (per_iter_us / 1e6);
    eprintln!("  Q4_K {:<20}: {:5} outs x {:2} K-tiles | {:7.2} us | {:5.1} GB/s | {:6.1} MB/op",
        label, n, blocks_per_row, per_iter_us, gb_s, bytes as f64 / 1e6);
}
unsafe fn bench_q6k(label: &str, n: usize, blocks_per_row: usize, n_iters: usize) {
    let k = blocks_per_row * 256;
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 1e-3 - 1.0).collect();
    let w = random_q6k(n, blocks_per_row, 0xBEEF);
    let d_a = aether_dev_alloc_f32(k as c_int);
    let d_w = aether_dev_alloc_u8(w.len() as c_int);
    let d_out = aether_dev_alloc_f32(n as c_int);
    aether_dev_h2d_f32(a.as_ptr() as i64, d_a, k as c_int);
    aether_dev_h2d_u8(w.as_ptr() as i64, d_w, w.len() as c_int);

    for _ in 0..3 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(d_a, d_w, d_out, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();
    let t = Instant::now();
    for _ in 0..n_iters {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(d_a, d_w, d_out, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();
    let elapsed = t.elapsed().as_secs_f64();
    let per_iter_us = elapsed * 1e6 / n_iters as f64;
    let bytes = n * blocks_per_row * 210;
    let gb_s = (bytes as f64 / 1e9) / (per_iter_us / 1e6);
    eprintln!("  Q6_K {:<20}: {:5} outs x {:2} K-tiles | {:7.2} us | {:5.1} GB/s | {:6.1} MB/op",
        label, n, blocks_per_row, per_iter_us, gb_s, bytes as f64 / 1e6);
}
unsafe fn bench_ffn(label: &str, n: usize, blocks_per_row: usize, n_iters: usize) {
    let k = blocks_per_row * 256;
    let a: Vec<f32> = (0..k).map(|i| (i as f32) * 1e-3 - 1.0).collect();
    let wg = random_q4k(n, blocks_per_row, 0xAAAA);
    let wu = random_q4k(n, blocks_per_row, 0xBBBB);
    let d_a = aether_dev_alloc_f32(k as c_int);
    let d_wg = aether_dev_alloc_u8(wg.len() as c_int);
    let d_wu = aether_dev_alloc_u8(wu.len() as c_int);
    let d_out = aether_dev_alloc_f32(n as c_int);
    aether_dev_h2d_f32(a.as_ptr() as i64, d_a, k as c_int);
    aether_dev_h2d_u8(wg.as_ptr() as i64, d_wg, wg.len() as c_int);
    aether_dev_h2d_u8(wu.as_ptr() as i64, d_wu, wu.len() as c_int);

    for _ in 0..3 {
        aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(d_a, d_wg, d_wu, d_out, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();
    let t = Instant::now();
    for _ in 0..n_iters {
        aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(d_a, d_wg, d_wu, d_out, n as c_int, blocks_per_row as c_int);
    }
    aether_dev_sync();
    let elapsed = t.elapsed().as_secs_f64();
    let per_iter_us = elapsed * 1e6 / n_iters as f64;
    let bytes = n * blocks_per_row * 144 * 2;  // gate + up
    let gb_s = (bytes as f64 / 1e9) / (per_iter_us / 1e6);
    eprintln!("  Q4_K-FFN {:<16}: {:5} outs x {:2} K-tiles | {:7.2} us | {:5.1} GB/s | {:6.1} MB/op",
        label, n, blocks_per_row, per_iter_us, gb_s, bytes as f64 / 1e6);
}

#[test]
#[ignore]
fn matmul_per_shape() {
    unsafe {
        aether_dev_init();
        eprintln!("\nPer-shape matmul bench (RTX 3070 Ti, after CUDA graphs land):\n");
        eprintln!("Shape (out x in)              Time      GB/s     Bytes");

        // Q4_K shapes used in autoregressive:
        bench_q4k("Q proj (D_MODEL)",   D_MODEL, D_MODEL/256, 100);  // 3584x3584
        bench_q4k("K/V proj (D_KV)",    D_KV,    D_MODEL/256, 100);  // 512x3584
        bench_q4k("O proj (D_MODEL)",   D_MODEL, D_MODEL/256, 100);  // 3584x3584
        bench_q4k("FFN single (D_FF)",  D_FF,    D_MODEL/256, 100);  // 18944x3584 (legacy)
        bench_q4k("down (D_MODEL)",     D_MODEL, D_FF/256,    100);  // 3584x18944
        bench_q4k("lm_head (VOCAB)",    VOCAB,   D_MODEL/256, 20);   // 152064x3584
        // Q6_K shapes:
        bench_q6k("V proj (D_KV)",      D_KV,    D_MODEL/256, 100);
        bench_q6k("down (D_MODEL)",     D_MODEL, D_FF/256,    100);
        bench_q6k("lm_head (VOCAB)",    VOCAB,   D_MODEL/256, 20);
        // Fused FFN:
        bench_ffn("gate+up+silu+mul",   D_FF,    D_MODEL/256, 100);
    }
}
