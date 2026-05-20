//! Parity + bench for the small-N matmul variants (32-thread CTAs).

#![cfg(feature = "cuda")]

#![allow(non_snake_case)]

use std::os::raw::c_int;
use std::time::Instant;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_alloc_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_matmul_seq1_smallN_cuda,
    aether_op_fused_q6k_matmul_seq1_smallN_cuda,
};

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
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 210;
        out[off + 208] = 0x47; out[off + 209] = 0x21;
    }
    out
}

#[test]
#[ignore]
fn q4k_smallN_matches_v2() {
    unsafe {
        aether_dev_init();
        const N: usize = 512;
        const BLOCKS: c_int = 14;
        const K: usize = (BLOCKS as usize) * 256;

        let a: Vec<f32> = (0..K).map(|i| (i as f32) * 1e-3 - 1.0).collect();
        let w = random_q4k(N, BLOCKS as usize, 0xCAFE);
        let d_a = aether_dev_alloc_f32(K as c_int);
        let d_w = aether_dev_alloc_u8(w.len() as c_int);
        let d_v2 = aether_dev_alloc_f32(N as c_int);
        let d_sm = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_f32(a.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(w.as_ptr() as i64, d_w, w.len() as c_int);

        assert_eq!(0, aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_w, d_v2, N as c_int, BLOCKS));
        assert_eq!(0, aether_op_fused_q4k_matmul_seq1_smallN_cuda(d_a, d_w, d_sm, N as c_int, BLOCKS));
        aether_dev_sync();
        let mut v2 = vec![0.0f32; N]; let mut sm = vec![0.0f32; N];
        aether_dev_d2h_f32(d_v2, v2.as_mut_ptr() as i64, N as c_int);
        aether_dev_d2h_f32(d_sm, sm.as_mut_ptr() as i64, N as c_int);

        let mut max_diff = 0.0f32; let mut max_rel = 0.0f32;
        for i in 0..N {
            let d = (v2[i] - sm[i]).abs();
            if d > max_diff { max_diff = d; }
            let rel = if v2[i].abs() > 1e-6 { d / v2[i].abs() } else { 0.0 };
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("[Q4_K smallN parity] max_diff={:.3e} max_rel={:.3e}", max_diff, max_rel);
        assert!(max_rel < 1e-3, "Q4_K smallN diverges from v2");
    }
}

#[test]
#[ignore]
fn q6k_smallN_matches_v2() {
    unsafe {
        aether_dev_init();
        const N: usize = 512;
        const BLOCKS: c_int = 14;
        const K: usize = (BLOCKS as usize) * 256;

        let a: Vec<f32> = (0..K).map(|i| (i as f32) * 1e-3 - 1.0).collect();
        let w = random_q6k(N, BLOCKS as usize, 0xBEEF);
        let d_a = aether_dev_alloc_f32(K as c_int);
        let d_w = aether_dev_alloc_u8(w.len() as c_int);
        let d_v2 = aether_dev_alloc_f32(N as c_int);
        let d_sm = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_f32(a.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(w.as_ptr() as i64, d_w, w.len() as c_int);

        assert_eq!(0, aether_op_fused_q6k_matmul_seq1_v2_cuda(d_a, d_w, d_v2, N as c_int, BLOCKS));
        assert_eq!(0, aether_op_fused_q6k_matmul_seq1_smallN_cuda(d_a, d_w, d_sm, N as c_int, BLOCKS));
        aether_dev_sync();
        let mut v2 = vec![0.0f32; N]; let mut sm = vec![0.0f32; N];
        aether_dev_d2h_f32(d_v2, v2.as_mut_ptr() as i64, N as c_int);
        aether_dev_d2h_f32(d_sm, sm.as_mut_ptr() as i64, N as c_int);

        let mut max_diff = 0.0f32; let mut max_rel = 0.0f32;
        for i in 0..N {
            let d = (v2[i] - sm[i]).abs();
            if d > max_diff { max_diff = d; }
            let rel = if v2[i].abs() > 1e-6 { d / v2[i].abs() } else { 0.0 };
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("[Q6_K smallN parity] max_diff={:.3e} max_rel={:.3e}", max_diff, max_rel);
        assert!(max_rel < 1e-3, "Q6_K smallN diverges from v2");
    }
}

#[test]
#[ignore]
fn bench_kv_smallN_vs_v2() {
    unsafe {
        aether_dev_init();
        const N: usize = 512;
        const BLOCKS: c_int = 14;
        const K: usize = (BLOCKS as usize) * 256;
        const ITERS: usize = 200;

        let a: Vec<f32> = (0..K).map(|i| (i as f32) * 1e-3 - 1.0).collect();
        let w = random_q4k(N, BLOCKS as usize, 0xCAFE);
        let d_a = aether_dev_alloc_f32(K as c_int);
        let d_w = aether_dev_alloc_u8(w.len() as c_int);
        let d_out = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_f32(a.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(w.as_ptr() as i64, d_w, w.len() as c_int);

        // Warm
        for _ in 0..5 { aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_w, d_out, N as c_int, BLOCKS); }
        aether_dev_sync();
        let t = Instant::now();
        for _ in 0..ITERS { aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_w, d_out, N as c_int, BLOCKS); }
        aether_dev_sync();
        let v2_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        for _ in 0..5 { aether_op_fused_q4k_matmul_seq1_smallN_cuda(d_a, d_w, d_out, N as c_int, BLOCKS); }
        aether_dev_sync();
        let t = Instant::now();
        for _ in 0..ITERS { aether_op_fused_q4k_matmul_seq1_smallN_cuda(d_a, d_w, d_out, N as c_int, BLOCKS); }
        aether_dev_sync();
        let sm_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let bytes = (N * BLOCKS as usize * 144) as f64;
        let v2_gbps = bytes / 1e9 / (v2_us / 1e6);
        let sm_gbps = bytes / 1e9 / (sm_us / 1e6);
        eprintln!("[K/V proj 512x3584 Q4_K]");
        eprintln!("  v2       : {:6.2} us = {:5.1} GB/s", v2_us, v2_gbps);
        eprintln!("  smallN   : {:6.2} us = {:5.1} GB/s  ({:.2}x)", sm_us, sm_gbps, v2_us / sm_us);
    }
}
