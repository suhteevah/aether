//! Weight-reuse batched Q4_K matmul parity + throughput (FR-19.5-extra-deep
//! Phase 2b-2a).
//!
//! `fused_q4k_matmul_seqB_v3` computes `batch` independent output rows
//! against ONE shared Q4_K weight matrix, dequantizing each weight block
//! exactly once and reusing it across rows.  This is the decode throughput
//! lever for continuous batching: at batch=1 the matmul is DRAM-bandwidth-
//! bound on the weights, so amortizing the weight load over B rows scales
//! FMA throughput ~B× until compute-bound.
//!
//! `seqB_matches_b_sequential_seq1` (always runs): asserts the batched
//! kernel is BIT-IDENTICAL to `batch` sequential `seq1_v3` calls — the FMA
//! order per (row, output) is unchanged, so equality must be exact.
//!
//! `seqB_throughput_bench` (#[ignore], opt-in): times the batched kernel
//! against B serial seq1_v3 calls at Qwen2.5-7B d_model=3584 and prints the
//! measured speedup.
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_fused_q4k_matmul_seq1_v3_cuda,
    aether_op_fused_q4k_matmul_seqB_v3_cuda,
};

/// Random Q4_K weight bytes with fixed d/dmin per block (avoids NaN/inf).
fn random_q4k_bytes(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = vec![0u8; n_outputs * blocks_per_row * 144];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 144;
        out[off] = 0x47; out[off + 1] = 0x21;     // d    ≈ 0.01
        out[off + 2] = 0x47; out[off + 3] = 0x19; // dmin ≈ 0.005
    }
    out
}

#[test]
fn seqB_matches_b_sequential_seq1() {
    unsafe {
        assert_eq!(0, aether_dev_init(), "CUDA init");
        const N: usize = 512;
        const N_BLOCKS: c_int = 4;
        const K: usize = (N_BLOCKS as usize) * 256;
        const BATCH: usize = 4;

        // Distinct activation per row so any cross-row mixing in the
        // batched kernel would diverge the output.
        let mut a_rows: Vec<Vec<f32>> = Vec::with_capacity(BATCH);
        for b in 0..BATCH {
            a_rows.push((0..K)
                .map(|i| (((i + b * 31) as f32) * 0.0007 - 0.3).sin() * 0.5)
                .collect());
        }
        let w_host = random_q4k_bytes(N, N_BLOCKS as usize, 0xA5A5_1234u64);

        let d_w = aether_dev_alloc_u8(w_host.len() as c_int);
        aether_dev_h2d_u8(w_host.as_ptr() as i64, d_w, w_host.len() as c_int);

        // Reference: BATCH separate seq1_v3 calls.
        let mut ref_out = vec![0.0f32; BATCH * N];
        let d_a1 = aether_dev_alloc_f32(K as c_int);
        let d_o1 = aether_dev_alloc_f32(N as c_int);
        for b in 0..BATCH {
            aether_dev_h2d_f32(a_rows[b].as_ptr() as i64, d_a1, K as c_int);
            assert_eq!(0, aether_op_fused_q4k_matmul_seq1_v3_cuda(d_a1, d_w, d_o1, N as c_int, N_BLOCKS));
            aether_dev_sync();
            aether_dev_d2h_f32(d_o1, ref_out[b * N..(b + 1) * N].as_mut_ptr() as i64, N as c_int);
        }

        // Candidate: ONE batched call.  a is [BATCH * K] row-major.
        let mut a_batch = Vec::with_capacity(BATCH * K);
        for b in 0..BATCH { a_batch.extend_from_slice(&a_rows[b]); }
        let d_ab = aether_dev_alloc_f32((BATCH * K) as c_int);
        let d_ob = aether_dev_alloc_f32((BATCH * N) as c_int);
        aether_dev_h2d_f32(a_batch.as_ptr() as i64, d_ab, (BATCH * K) as c_int);
        assert_eq!(0, aether_op_fused_q4k_matmul_seqB_v3_cuda(
            d_ab, d_w, d_ob, N as c_int, N_BLOCKS, BATCH as c_int));
        aether_dev_sync();
        let mut cand_out = vec![0.0f32; BATCH * N];
        aether_dev_d2h_f32(d_ob, cand_out.as_mut_ptr() as i64, (BATCH * N) as c_int);

        let mut max_diff = 0.0f32;
        for i in 0..BATCH * N {
            let d = (ref_out[i] - cand_out[i]).abs();
            if d > max_diff { max_diff = d; }
        }
        println!("[q4k seqB] batch={} n={} max_abs_diff vs seq1×{} = {:.3e}",
            BATCH, N, BATCH, max_diff);
        // FMA order identical → must be EXACTLY bit-identical.
        assert_eq!(max_diff, 0.0,
            "batched kernel diverged from sequential seq1 (max_abs={})", max_diff);

        aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_a1); aether_dev_free_f32(d_o1);
        aether_dev_free_f32(d_ab); aether_dev_free_f32(d_ob);
    }
}

#[test]
#[ignore = "perf microbench — run explicitly with --ignored --nocapture"]
fn seqB_throughput_bench() {
    unsafe {
        assert_eq!(0, aether_dev_init(), "CUDA init");
        // Qwen2.5-7B d_model = 3584 = 14 super-blocks of 256.
        const N: usize = 3584;
        const N_BLOCKS: c_int = 14;
        const K: usize = (N_BLOCKS as usize) * 256;
        const BATCH: usize = 4;
        const ITERS: usize = 400;

        let a_batch: Vec<f32> = (0..(BATCH * K))
            .map(|i| ((i as f32) * 0.0003 - 0.1).sin() * 0.4).collect();
        let w_host = random_q4k_bytes(N, N_BLOCKS as usize, 0x1357_9BDFu64);
        let d_w = aether_dev_alloc_u8(w_host.len() as c_int);
        let d_ab = aether_dev_alloc_f32((BATCH * K) as c_int);
        let d_ob = aether_dev_alloc_f32((BATCH * N) as c_int);
        let d_o1 = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_u8(w_host.as_ptr() as i64, d_w, w_host.len() as c_int);
        aether_dev_h2d_f32(a_batch.as_ptr() as i64, d_ab, (BATCH * K) as c_int);

        // Warm-up (GPU boost clocks ramp; first launches JIT-compile PTX).
        for _ in 0..40 {
            aether_op_fused_q4k_matmul_seqB_v3_cuda(d_ab, d_w, d_ob, N as c_int, N_BLOCKS, BATCH as c_int);
            aether_op_fused_q4k_matmul_seq1_v3_cuda(d_ab, d_w, d_o1, N as c_int, N_BLOCKS);
        }
        aether_dev_sync();

        // Serial baseline: BATCH seq1_v3 launches per "step" (what the
        // scheduler does today — one matmul launch per slot).  Each launch
        // re-reads the full weight matrix from DRAM.
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            for _b in 0..BATCH {
                aether_op_fused_q4k_matmul_seq1_v3_cuda(d_ab, d_w, d_o1, N as c_int, N_BLOCKS);
            }
        }
        aether_dev_sync();
        let serial = t0.elapsed().as_secs_f64();

        // Batched: ONE seqB call per step.
        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            aether_op_fused_q4k_matmul_seqB_v3_cuda(d_ab, d_w, d_ob, N as c_int, N_BLOCKS, BATCH as c_int);
        }
        aether_dev_sync();
        let batched = t1.elapsed().as_secs_f64();

        let serial_us = serial / ITERS as f64 * 1e6;
        let batched_us = batched / ITERS as f64 * 1e6;
        println!("[q4k seqB bench] n={} batch={} iters={}", N, BATCH, ITERS);
        println!("  serial  ({}× seq1_v3): {:.2} µs/step", BATCH, serial_us);
        println!("  batched (1× seqB_v3):  {:.2} µs/step", batched_us);
        println!("  speedup: {:.2}×", serial_us / batched_us);
        // The batched kernel must be at least somewhat faster than B serial
        // launches; if it regressed, the weight-reuse thesis is wrong.
        assert!(batched_us < serial_us,
            "batched ({:.2}µs) not faster than serial ({:.2}µs)", batched_us, serial_us);

        aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_ab); aether_dev_free_f32(d_ob); aether_dev_free_f32(d_o1);
    }
}
