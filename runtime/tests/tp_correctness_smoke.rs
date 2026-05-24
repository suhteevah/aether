//! Tensor-parallel correctness smoke tests.
//!
//! These tests run on every machine — they don't require multiple GPUs.
//! They verify the sharding-plan math is correct (so when the multi-
//! context cuda.rs refactor lands the TP runtime is built on a proven
//! foundation).
//!
//! Multi-GPU end-to-end smoke is `#[ignore]`'d until the cuda.rs multi-
//! context refactor lands.  Run with `cargo test --release --features nccl
//! tp_dual_gpu_real -- --ignored` on a 2-GPU box once that lands.

#![cfg(feature = "cuda")]

use aether_rt::tensor_parallel::{
    BlockShardPlan, HeadShard, NcclAvailability,
    all_reduce_sum_host, concat_rows_host, matmul_nt_host,
    partition_heads, partition_rows, probe_nccl_availability,
    shard_w_by_cols, shard_w_by_rows, shard_x_by_cols,
};

#[test]
fn partition_rows_round_trip_qwen2_5_7b_shapes() {
    // Qwen2.5-7B: 28 Q heads / 4 KV heads, d_model=3584, d_ff=18944.
    for n in [28usize, 4, 3584, 18944, 152064] {
        for w in [1usize, 2, 4, 8] {
            let p = partition_rows(n, w);
            assert_eq!(p.len(), w);
            let total: usize = p.iter().map(|(_, l)| *l).sum();
            assert_eq!(total, n, "n={} w={}", n, w);
        }
    }
}

#[test]
fn partition_heads_qwen2_5_7b_tp2() {
    let p = partition_heads(28, 4, 2);
    assert_eq!(p.len(), 2);
    assert_eq!(p[0].q_head_count, 14);
    assert_eq!(p[1].q_head_count, 14);
    assert_eq!(p[0].kv_head_count, 2);
    assert_eq!(p[1].kv_head_count, 2);
    assert!(!p[0].kv_replicated);
}

#[test]
fn partition_heads_replicates_kv_when_undersized() {
    // GLM-4.7-flash MLA-absorbed: 20 Q heads, effective 1 KV head per
    // rank — KV gets replicated.
    let p = partition_heads(20, 1, 2);
    assert!(p[0].kv_replicated);
    assert!(p[1].kv_replicated);
    assert_eq!(p[0].q_head_count, 10);
    assert_eq!(p[1].q_head_count, 10);
}

/// TP=1 column-parallel must be bit-identical to the unsharded matmul.
#[test]
fn column_parallel_tp1_is_bit_identical() {
    let n_in = 256usize;
    let n_out = 64usize;
    let w: Vec<f32> = (0..n_in * n_out).map(|i| ((i as f32) * 0.001).sin()).collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.003).cos()).collect();
    let y_full = matmul_nt_host(&x, &w, n_in, n_out);
    let shards = shard_w_by_rows(&w, n_in, n_out, 1);
    assert_eq!(shards.len(), 1);
    let y = matmul_nt_host(&x, &shards[0], n_in, n_out);
    assert_eq!(y, y_full, "TP=1 column-parallel must be bit-identical");
}

/// TP=1 row-parallel must be bit-identical to the unsharded matmul.
#[test]
fn row_parallel_tp1_is_bit_identical() {
    let n_in = 256usize;
    let n_out = 64usize;
    let w: Vec<f32> = (0..n_in * n_out).map(|i| ((i as f32) * 0.001).sin()).collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.003).cos()).collect();
    let y_full = matmul_nt_host(&x, &w, n_in, n_out);
    let w_s = shard_w_by_cols(&w, n_in, n_out, 1);
    let x_s = shard_x_by_cols(&x, 1);
    assert_eq!(w_s.len(), 1);
    assert_eq!(x_s.len(), 1);
    let mut y = vec![0.0f32; n_out];
    for i in 0..n_out {
        let mut acc = 0.0;
        for k in 0..n_in { acc += x_s[0][k] * w_s[0][i * n_in + k]; }
        y[i] = acc;
    }
    let y_reduced = all_reduce_sum_host(&[y]);
    assert_eq!(y_reduced, y_full, "TP=1 row-parallel must be bit-identical");
}

/// TP=4 column-parallel: shard W by rows, run per-rank partial matmul,
/// concat row-slices → must recover the unsharded matmul to f32 noise.
#[test]
fn column_parallel_tp4_reconstructs() {
    let n_in = 256usize;
    let n_out = 128usize;
    let world_size = 4usize;
    let w: Vec<f32> = (0..n_in * n_out).map(|i| ((i as f32) * 0.0017).sin()).collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.029).cos()).collect();
    let y_full = matmul_nt_host(&x, &w, n_in, n_out);

    let shards = shard_w_by_rows(&w, n_in, n_out, world_size);
    let parts = partition_rows(n_out, world_size);
    let mut partials = Vec::new();
    for (r, &(_s, len)) in parts.iter().enumerate() {
        partials.push(matmul_nt_host(&x, &shards[r], n_in, len));
    }
    let y_concat = concat_rows_host(&partials);

    assert_eq!(y_concat.len(), y_full.len());
    for i in 0..y_full.len() {
        assert!((y_concat[i] - y_full[i]).abs() < 1e-5,
            "concat mismatch at {}: got {} vs expected {}",
            i, y_concat[i], y_full[i]);
    }
}

/// TP=4 row-parallel: shard W by cols, shard X by the same axis, run per
/// rank partial matmul, all-reduce SUM → recover unsharded matmul.
#[test]
fn row_parallel_tp4_reconstructs() {
    let n_in = 256usize;
    let n_out = 128usize;
    let world_size = 4usize;
    let w: Vec<f32> = (0..n_in * n_out).map(|i| ((i as f32) * 0.0023).sin()).collect();
    let x: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.041).cos()).collect();
    let y_full = matmul_nt_host(&x, &w, n_in, n_out);

    let w_s = shard_w_by_cols(&w, n_in, n_out, world_size);
    let x_s = shard_x_by_cols(&x, world_size);
    let parts = partition_rows(n_in, world_size);

    let mut partials = Vec::new();
    for (r, &(_s, len)) in parts.iter().enumerate() {
        let mut y_part = vec![0.0f32; n_out];
        for i in 0..n_out {
            let mut acc = 0.0;
            for k in 0..len {
                acc += x_s[r][k] * w_s[r][i * len + k];
            }
            y_part[i] = acc;
        }
        partials.push(y_part);
    }
    let y_reduced = all_reduce_sum_host(&partials);

    assert_eq!(y_reduced.len(), y_full.len());
    for i in 0..y_full.len() {
        assert!((y_reduced[i] - y_full[i]).abs() < 1e-4,
            "row-parallel mismatch at {}: got {} vs expected {}",
            i, y_reduced[i], y_full[i]);
    }
}

/// `BlockShardPlan` for Qwen2.5-7B TP=2 must sum to the full block dims.
#[test]
fn block_shard_plan_qwen25_7b_tp2_invariants() {
    let p0 = BlockShardPlan::new(0, 2, 3584, 18944, 28, 4, 128);
    let p1 = BlockShardPlan::new(1, 2, 3584, 18944, 28, 4, 128);
    assert_eq!(p0.d_ff_local + p1.d_ff_local, 18944);
    assert_eq!(p0.heads.q_head_count + p1.heads.q_head_count, 28);
    assert_eq!(p0.heads.kv_head_count + p1.heads.kv_head_count, 4);
    assert_eq!(p0.d_q_local + p1.d_q_local, 28 * 128);
    assert_eq!(p0.d_kv_local + p1.d_kv_local, 4 * 128);
}

/// Llama-3-8B-style TP=4 case: 32 Q heads, 8 KV heads, head_dim=128.
#[test]
fn block_shard_plan_llama_3_8b_tp4_invariants() {
    let plans: Vec<BlockShardPlan> = (0..4)
        .map(|r| BlockShardPlan::new(r, 4, 4096, 14336, 32, 8, 128))
        .collect();
    let d_ff_total: usize = plans.iter().map(|p| p.d_ff_local).sum();
    let q_total: usize = plans.iter().map(|p| p.heads.q_head_count).sum();
    let kv_total: usize = plans.iter().map(|p| p.heads.kv_head_count).sum();
    assert_eq!(d_ff_total, 14336);
    assert_eq!(q_total, 32);
    assert_eq!(kv_total, 8);
    for p in &plans {
        assert!(!p.heads.kv_replicated);
        assert_eq!(p.heads.q_head_count, 8);
        assert_eq!(p.heads.kv_head_count, 2);
        assert_eq!(p.d_q_local, 8 * 128);
        assert_eq!(p.d_ff_local, 14336 / 4);
    }
}

/// NCCL availability probe must never panic and must report a defined
/// status.  This is the call serve.rs uses to gate --tp 2.
#[test]
fn nccl_probe_is_safe() {
    let a = probe_nccl_availability();
    // Each variant is fine — just verify the variant is exhaustively
    // matchable (i.e. the enum hasn't drifted).
    let _label = match a {
        NcclAvailability::Available { n_devices: _ } => "available",
        NcclAvailability::NotEnoughDevices { n_devices: _ } => "not_enough",
        NcclAvailability::FeatureNotCompiled => "feature_off",
        NcclAvailability::ProbeFailed => "probe_failed",
    };
}

/// HeadShard equality / Debug-fmt sanity.
#[test]
fn head_shard_eq() {
    let a = HeadShard {
        q_head_start: 0, q_head_count: 14,
        kv_head_start: 0, kv_head_count: 2,
        kv_replicated: false,
    };
    let b = a;
    assert_eq!(a, b);
}

/// **The critical bit-identity gate**: on a machine where the file is
/// present, build a TpSession with `world_size = 1` against a real GGUF
/// and verify a single decode step matches a fresh QwenSession decode
/// step on the same prompt.
///
/// Gated behind `AETHER_TP_GGUF` env var so it only runs when a model is
/// available.  CI / kokonoe machines without a GGUF skip transparently.
#[test]
fn tp1_decode_matches_non_tp_decode_when_gguf_available() {
    let gguf = match std::env::var("AETHER_TP_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[skip] AETHER_TP_GGUF not set; pointing it at any \
                Qwen2.5/Llama Q4_K_M GGUF runs the bit-identity gate.");
            return;
        }
    };
    if !std::path::Path::new(&gguf).exists() {
        eprintln!("[skip] AETHER_TP_GGUF={} does not exist on this machine", gguf);
        return;
    }

    let prompt = vec![1usize, 2, 3, 4, 5];

    let mut base = aether_rt::serving::QwenSession::new(&gguf)
        .expect("baseline QwenSession::new");
    let base_first = base.generate(&prompt, 4, None);

    let mut tp = aether_rt::tensor_parallel::TpSession::new(&gguf, 1)
        .expect("TpSession::new tp=1");
    assert_eq!(tp.effective_world_size, 1);
    let tp_first = tp.generate(&prompt, 4, None);

    assert_eq!(base_first, tp_first,
        "TP=1 must produce bit-identical token ids to non-TP path");
}

/// Real 2-GPU TP smoke.  Marked `#[ignore]` because the cuda.rs multi-
/// context refactor (TP_GAPS::CUDA_MULTI_CONTEXT) is not yet shipped —
/// TpSession currently falls back to single-GPU even with 2 GPUs
/// visible.  Once the refactor lands, drop the ignore + run with
/// `cargo test --release --features nccl tp_dual_gpu_real_2 -- --ignored`.
#[test]
#[ignore]
fn tp_dual_gpu_real_2() {
    let gguf = std::env::var("AETHER_TP_GGUF")
        .expect("AETHER_TP_GGUF must point at a Q4_K_M model");
    let mut tp = aether_rt::tensor_parallel::TpSession::new(&gguf, 2)
        .expect("TpSession::new tp=2");
    // After the multi-context refactor this will be 2.
    assert_eq!(tp.effective_world_size, 2,
        "TP=2 must actually shard once cuda.rs is multi-context.  Today \
         it falls back to 1; this test exists to fail loudly when that \
         changes.");
    let out = tp.generate(&[1, 2, 3, 4, 5], 8, None);
    assert!(!out.is_empty());
}
