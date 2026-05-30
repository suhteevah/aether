//! Weight-reuse batched Q6_K matmul parity + throughput (batched-MMVQ Phase).
//!
//! `fused_q6k_matmul_seqB_v3` computes `batch` independent output rows against
//! ONE shared Q6_K weight matrix, dequantizing each 210-byte super-block exactly
//! once and reusing it across rows.  It replaces the per-row fallback in
//! `matmul_batched` (which re-read each Q6_K weight `batch` times via `batch`
//! separate `seq1_v2` launches).  Q6_K weights are `ffn_down` + `attn_v` in
//! Qwen2.5-7B, the dominant batched-decode cost — so this is the lever that took
//! the N=8 server aggregate 23.1 → 34.1 tok/s (+47%).
//!
//! `seqB_matches_b_sequential_seq1` (always runs): for every batch size 1..=8,
//! asserts the batched kernel is BIT-IDENTICAL to `batch` sequential `seq1_v2`
//! calls.  The dequant + FMA order per (row, output) is unchanged, so equality
//! must be EXACT — any cross-row mixing or layout drift diverges it.
//!
//! `seqB_throughput_bench` (#[ignore], opt-in): times the batched kernel against
//! B serial `seq1_v2` calls at Qwen2.5-7B `ffn_down` shape and prints the speedup.
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seqB_v3_cuda,
    aether_op_fused_q6k_matmul_seqB_fp16_cuda,
};

/// Random Q6_K weight bytes (210/super-block) with a fixed small f16 `d` per
/// block (at byte offset 208/209) so the dequant stays finite — avoids NaN/inf
/// while keeping the quants/scales fully random to stress every layout path.
fn random_q6k_bytes(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = vec![0u8; n_outputs * blocks_per_row * 210];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 210;
        // d (little-endian f16 at [208],[209]) = 0x2147 ≈ 0.0103.
        out[off + 208] = 0x47;
        out[off + 209] = 0x21;
    }
    out
}

#[test]
fn seqB_matches_b_sequential_seq1() {
    unsafe {
        assert_eq!(0, aether_dev_init(), "CUDA init");
        const N: usize = 512;            // output rows (e.g. attn_v d_kv)
        const N_BLOCKS: c_int = 4;
        const K: usize = (N_BLOCKS as usize) * 256;

        let w_host = random_q6k_bytes(N, N_BLOCKS as usize, 0xC0FF_EE17u64);
        let d_w = aether_dev_alloc_u8(w_host.len() as c_int);
        aether_dev_h2d_u8(w_host.as_ptr() as i64, d_w, w_host.len() as c_int);

        let d_a1 = aether_dev_alloc_f32(K as c_int);
        let d_o1 = aether_dev_alloc_f32(N as c_int);

        // Every batch size the scheduler can form (1..=MAX_BATCH=8).
        for batch in 1usize..=8 {
            // Distinct activation per row → any cross-row mixing diverges.
            let mut a_rows: Vec<Vec<f32>> = Vec::with_capacity(batch);
            for b in 0..batch {
                a_rows.push((0..K)
                    .map(|i| (((i + b * 37) as f32) * 0.0009 - 0.25).sin() * 0.5)
                    .collect());
            }

            // Reference: `batch` separate seq1_v2 calls.
            let mut ref_out = vec![0.0f32; batch * N];
            for b in 0..batch {
                aether_dev_h2d_f32(a_rows[b].as_ptr() as i64, d_a1, K as c_int);
                assert_eq!(0, aether_op_fused_q6k_matmul_seq1_v2_cuda(
                    d_a1, d_w, d_o1, N as c_int, N_BLOCKS));
                aether_dev_sync();
                aether_dev_d2h_f32(d_o1, ref_out[b * N..(b + 1) * N].as_mut_ptr() as i64, N as c_int);
            }

            // Candidate: ONE batched seqB call.  a is [batch * K] row-major.
            let mut a_batch = Vec::with_capacity(batch * K);
            for b in 0..batch { a_batch.extend_from_slice(&a_rows[b]); }
            let d_ab = aether_dev_alloc_f32((batch * K) as c_int);
            let d_ob = aether_dev_alloc_f32((batch * N) as c_int);
            aether_dev_h2d_f32(a_batch.as_ptr() as i64, d_ab, (batch * K) as c_int);
            assert_eq!(0, aether_op_fused_q6k_matmul_seqB_v3_cuda(
                d_ab, d_w, d_ob, N as c_int, N_BLOCKS, batch as c_int));
            aether_dev_sync();
            let mut cand_out = vec![0.0f32; batch * N];
            aether_dev_d2h_f32(d_ob, cand_out.as_mut_ptr() as i64, (batch * N) as c_int);

            let mut max_diff = 0.0f32;
            for i in 0..batch * N {
                let d = (ref_out[i] - cand_out[i]).abs();
                if d > max_diff { max_diff = d; }
            }
            println!("[q6k seqB] batch={} n={} max_abs_diff vs seq1×{} = {:.3e}",
                batch, N, batch, max_diff);
            // Dequant + FMA order identical to seq1_v2 → must be EXACTLY bit-identical.
            assert_eq!(max_diff, 0.0,
                "batch={}: seqB diverged from sequential seq1_v2 (max_abs={})", batch, max_diff);

            aether_dev_free_f32(d_ab);
            aether_dev_free_f32(d_ob);
        }

        aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_a1);
        aether_dev_free_f32(d_o1);
    }
}

/// fp16 half2 variant (P100 lever): accuracy vs the fp32 seqB (NOT bit-exact —
/// fp16 multiply + per-super-block fp16 partials introduce small error, but the
/// long-K + warp reductions are fp32, so it must stay within a tight tolerance)
/// + isolated speed at the ffn_down shape (B=8).  Decides whether fp16 is worth
/// integrating before touching the e2e serving path.
#[test]
fn seqB_fp16_accuracy_and_speed() {
    unsafe {
        assert_eq!(0, aether_dev_init(), "CUDA init");
        // ffn_down shape: n_out = d_model = 3584, n_in = d_ff = 18944 = 74 blocks.
        const N: usize = 3584;
        const N_BLOCKS: c_int = 74;
        const K: usize = (N_BLOCKS as usize) * 256;
        const BATCH: usize = 8;

        let w_host = random_q6k_bytes(N, N_BLOCKS as usize, 0xF00D_5EEDu64);
        let a_batch: Vec<f32> = (0..(BATCH * K))
            .map(|i| ((i as f32) * 0.00025 - 0.12).sin() * 0.5).collect();
        let d_w = aether_dev_alloc_u8(w_host.len() as c_int);
        let d_ab = aether_dev_alloc_f32((BATCH * K) as c_int);
        let d_o32 = aether_dev_alloc_f32((BATCH * N) as c_int);
        let d_o16 = aether_dev_alloc_f32((BATCH * N) as c_int);
        aether_dev_h2d_u8(w_host.as_ptr() as i64, d_w, w_host.len() as c_int);
        aether_dev_h2d_f32(a_batch.as_ptr() as i64, d_ab, (BATCH * K) as c_int);

        // --- accuracy: fp16 seqB vs fp32 seqB (the fp32 path is bit-exact to per-row) ---
        assert_eq!(0, aether_op_fused_q6k_matmul_seqB_v3_cuda(d_ab, d_w, d_o32, N as c_int, N_BLOCKS, BATCH as c_int));
        assert_eq!(0, aether_op_fused_q6k_matmul_seqB_fp16_cuda(d_ab, d_w, d_o16, N as c_int, N_BLOCKS, BATCH as c_int));
        aether_dev_sync();
        let mut o32 = vec![0.0f32; BATCH * N];
        let mut o16 = vec![0.0f32; BATCH * N];
        aether_dev_d2h_f32(d_o32, o32.as_mut_ptr() as i64, (BATCH * N) as c_int);
        aether_dev_d2h_f32(d_o16, o16.as_mut_ptr() as i64, (BATCH * N) as c_int);
        let mut max_abs = 0.0f32;
        let mut max_mag = 0.0f32;
        for i in 0..BATCH * N {
            let d = (o32[i] - o16[i]).abs();
            if d > max_abs { max_abs = d; }
            if o32[i].abs() > max_mag { max_mag = o32[i].abs(); }
        }
        let rel = if max_mag > 0.0 { max_abs / max_mag } else { 0.0 };
        println!("[q6k fp16] accuracy vs fp32 seqB: max_abs={:.3e} max_mag={:.3e} rel={:.3e}",
            max_abs, max_mag, rel);
        assert!(rel < 0.05,
            "fp16 drifted too far from fp32 (rel={:.3e} >= 0.05) — fp32-accumulator design may be wrong", rel);

        // --- speed: fp16 vs fp32 seqB at B=8 ---
        const ITERS: usize = 200;
        for _ in 0..40 {
            aether_op_fused_q6k_matmul_seqB_fp16_cuda(d_ab, d_w, d_o16, N as c_int, N_BLOCKS, BATCH as c_int);
            aether_op_fused_q6k_matmul_seqB_v3_cuda(d_ab, d_w, d_o32, N as c_int, N_BLOCKS, BATCH as c_int);
        }
        aether_dev_sync();
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            aether_op_fused_q6k_matmul_seqB_v3_cuda(d_ab, d_w, d_o32, N as c_int, N_BLOCKS, BATCH as c_int);
        }
        aether_dev_sync();
        let fp32_us = t0.elapsed().as_secs_f64() / ITERS as f64 * 1e6;
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            aether_op_fused_q6k_matmul_seqB_fp16_cuda(d_ab, d_w, d_o16, N as c_int, N_BLOCKS, BATCH as c_int);
        }
        aether_dev_sync();
        let fp16_us = t1.elapsed().as_secs_f64() / ITERS as f64 * 1e6;
        println!("[q6k fp16 bench] n={} n_blocks={} batch={} iters={}", N, N_BLOCKS, BATCH, ITERS);
        println!("  fp32 seqB_v3: {:.2} µs/step", fp32_us);
        println!("  fp16 seqB:    {:.2} µs/step", fp16_us);
        println!("  fp16 speedup vs fp32: {:.2}× ({})", fp32_us / fp16_us,
            if fp16_us < fp32_us { "WIN" } else { "no win — fp32 stays" });

        aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_ab); aether_dev_free_f32(d_o32); aether_dev_free_f32(d_o16);
    }
}

#[test]
#[ignore = "perf microbench — run explicitly with --ignored --nocapture"]
fn seqB_throughput_bench() {
    unsafe {
        assert_eq!(0, aether_dev_init(), "CUDA init");
        // Qwen2.5-7B ffn_down: n_out = d_model = 3584, n_in = d_ff = 18944 = 74
        // super-blocks of 256.  This is the dominant Q6_K batched matmul.
        const N: usize = 3584;
        const N_BLOCKS: c_int = 74;
        const K: usize = (N_BLOCKS as usize) * 256;
        const BATCH: usize = 8;
        const ITERS: usize = 200;

        let a_batch: Vec<f32> = (0..(BATCH * K))
            .map(|i| ((i as f32) * 0.0002 - 0.1).sin() * 0.4).collect();
        let w_host = random_q6k_bytes(N, N_BLOCKS as usize, 0x2468_ACE0u64);
        let d_w = aether_dev_alloc_u8(w_host.len() as c_int);
        let d_ab = aether_dev_alloc_f32((BATCH * K) as c_int);
        let d_ob = aether_dev_alloc_f32((BATCH * N) as c_int);
        let d_o1 = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_u8(w_host.as_ptr() as i64, d_w, w_host.len() as c_int);
        aether_dev_h2d_f32(a_batch.as_ptr() as i64, d_ab, (BATCH * K) as c_int);

        // Warm-up (boost clocks + first-launch PTX JIT).
        for _ in 0..40 {
            aether_op_fused_q6k_matmul_seqB_v3_cuda(d_ab, d_w, d_ob, N as c_int, N_BLOCKS, BATCH as c_int);
            aether_op_fused_q6k_matmul_seq1_v2_cuda(d_ab, d_w, d_o1, N as c_int, N_BLOCKS);
        }
        aether_dev_sync();

        // Serial baseline: BATCH seq1_v2 launches per step (the per-row fallback).
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            for _b in 0..BATCH {
                aether_op_fused_q6k_matmul_seq1_v2_cuda(d_ab, d_w, d_o1, N as c_int, N_BLOCKS);
            }
        }
        aether_dev_sync();
        let serial = t0.elapsed().as_secs_f64();

        // Batched: ONE seqB call per step.
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            aether_op_fused_q6k_matmul_seqB_v3_cuda(d_ab, d_w, d_ob, N as c_int, N_BLOCKS, BATCH as c_int);
        }
        aether_dev_sync();
        let batched = t1.elapsed().as_secs_f64();

        let serial_us = serial / ITERS as f64 * 1e6;
        let batched_us = batched / ITERS as f64 * 1e6;
        println!("[q6k seqB bench] n={} n_blocks={} batch={} iters={}", N, N_BLOCKS, BATCH, ITERS);
        println!("  serial  ({}× seq1_v2): {:.2} µs/step", BATCH, serial_us);
        println!("  batched (1× seqB_v3):  {:.2} µs/step", batched_us);
        println!("  speedup: {:.2}×", serial_us / batched_us);
        assert!(batched_us < serial_us,
            "batched ({:.2}µs) not faster than serial ({:.2}µs)", batched_us, serial_us);

        aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_ab); aether_dev_free_f32(d_ob); aether_dev_free_f32(d_o1);
    }
}
