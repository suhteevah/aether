//! Deterministic crash repro for the N≥4 batched-decode illegal-address.
//!
//! The server crash (CUDA_ERROR_ILLEGAL_ADDRESS) is timing-dependent: it fires
//! at N=4 but the effective batch size the scheduler forms depends on request
//! arrival. This test removes that confound by driving
//! `step_logits_for_batch` DIRECTLY at a FIXED batch size `b` (env
//! AETHER_REPRO_B, default sweeps 2..=8), single-threaded, so each `b` is
//! exercised exactly. If a given `b` triggers the OOB it will panic here with a
//! Rust backtrace naming the faulting `aether_op_*` wrapper (run with
//! RUST_BACKTRACE=1; add CUDA_LAUNCH_BLOCKING=1 to make the faulting kernel's
//! own `.expect("launch X")` panic by name).
//!
//! NOT a parity assertion — purely "does this batch size crash". Skipped
//! without `--features cuda` AND without the GGUF (AETHER_QWEN25_GGUF).
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use aether_rt::serving::{QwenSession, SharedKvPool, MAX_SEQ};
use std::sync::Arc;

const DEFAULT_GGUF: &str = "C:/Users/Matt/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

fn gguf_path() -> Option<String> {
    let p = std::env::var("AETHER_QWEN25_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    if std::path::Path::new(&p).exists() { Some(p) } else {
        eprintln!("[crash-repro] GGUF not at {} — skipping", p);
        None
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; best = i; } }
    best
}

struct Slot {
    page_table_host: Vec<i32>,
    owned_blocks: Vec<i32>,
    next_pos: i32,
    last: usize,
}

/// Run `b` slots in lockstep for `steps` decode ticks. Heterogeneous prompt
/// lengths → heterogeneous positions/cur_seq (stresses the hetero kernels).
fn run_batch(sess: &mut QwenSession, b: usize, steps: usize, n_logical: usize) {
    eprintln!("[crash-repro] === b={} ===", b);
    // b distinct prompts of VARYING length (1..=b+3 tokens) → varied positions.
    let base: Vec<usize> = vec![9707, 11, 1879, 0, 785, 3974, 13, 264, 220, 17, 20, 4666];
    let mut slots: Vec<Slot> = Vec::new();
    for i in 0..b {
        let len = 3 + (i % 5);                       // 3..=7 tokens
        let prompt: Vec<usize> = (0..len).map(|k| base[(i + k) % base.len()]).collect();
        let mut s = Slot {
            page_table_host: vec![-1i32; n_logical.max(1)],
            owned_blocks: Vec::new(),
            next_pos: 0,
            last: 0,
        };
        sess.prefill_for_slot(&mut s.page_table_host, &mut s.owned_blocks,
            &mut s.next_pos, &prompt).expect("prefill");
        s.last = *prompt.last().unwrap();
        slots.push(s);
    }
    for step in 0..steps {
        for s in slots.iter_mut() {
            sess.slot_ensure_block(s.next_pos, &mut s.page_table_host,
                &mut s.owned_blocks).expect("ensure");
        }
        let page_tables: Vec<Vec<i32>> = slots.iter().map(|s| s.page_table_host.clone()).collect();
        let last_ids: Vec<usize> = slots.iter().map(|s| s.last).collect();
        let mut positions: Vec<i32> = slots.iter().map(|s| s.next_pos).collect();
        let logits = sess.step_logits_for_batch(&page_tables, &last_ids, &mut positions);
        for (i, s) in slots.iter_mut().enumerate() {
            s.next_pos = positions[i];
            s.last = argmax(&logits[i]);
        }
        if step == 0 || step == steps - 1 {
            eprintln!("[crash-repro]   b={} step {} OK (positions {:?})",
                b, step, slots.iter().map(|s| s.next_pos).collect::<Vec<_>>());
        }
    }
    for s in slots.iter_mut() { sess.slot_release_blocks(&mut s.owned_blocks); }
    eprintln!("[crash-repro] b={} survived {} steps", b, steps);
}

/// Reproduce the SERVER lifecycle that the fixed-b sweep omits: warmup (which
/// captures the CUDA graph), decode a persistent slot ALONE via the single-slot
/// graph path, then fuse it with fresh slots in the batched path, then retire
/// the fresh slots (pool churn) — looped. This is the untested interaction:
/// graph single-slot ↔ imperative batched, sharing one KV pool + block
/// allocator. AETHER_NO_GRAPH toggles the single-slot path (graph vs imperative)
/// to bisect. AETHER_REPRO_ROUNDS sets the loop count (default 16).
#[test]
fn batched_graph_churn_repro() {
    let Some(path) = gguf_path() else { return; };
    let rounds: usize = std::env::var("AETHER_REPRO_ROUNDS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);

    let block_size = 4i32;
    let n_logical = ((MAX_SEQ as i32 + block_size - 1) / block_size) as usize;
    let total_blocks = 8 * n_logical as i32;

    let cfg = { QwenSession::new(&path).expect("probe").cfg.clone() };
    assert!(cfg.kv_lora_rank == 0 && cfg.n_experts == 0, "needs dense arch");
    let pool: Arc<SharedKvPool> =
        SharedKvPool::new_for_shape(total_blocks, block_size, cfg.n_layers, cfg.d_kv);
    let mut sess = QwenSession::new_paged_with_pool(&path, pool.clone()).expect("pooled");
    assert!(sess.is_batchable(), "must be batchable");

    // Match the server: warmup captures the graph (unless AETHER_NO_GRAPH=1).
    eprintln!("[graph-churn] warmup (captures graph unless NO_GRAPH)...");
    sess.warmup(4);

    let prompta: Vec<usize> = vec![9707, 11, 1879, 0, 785];
    let mut a = Slot { page_table_host: vec![-1i32; n_logical], owned_blocks: Vec::new(),
                       next_pos: 0, last: 0 };
    sess.prefill_for_slot(&mut a.page_table_host, &mut a.owned_blocks, &mut a.next_pos, &prompta)
        .expect("prefill A");
    a.last = *prompta.last().unwrap();

    for round in 0..rounds {
        // 1. Persistent slot A decodes ALONE via the single-slot graph path.
        for _ in 0..3 {
            sess.slot_ensure_block(a.next_pos, &mut a.page_table_host, &mut a.owned_blocks)
                .expect("ensure A");
            let logits = sess.step_logits_for_slot(&a.page_table_host, &mut a.next_pos, a.last);
            a.last = argmax(&logits);
        }
        // 2. Admit 3 FRESH slots → batched b=4 (mix with A mid-life at high pos).
        let mut fresh: Vec<Slot> = Vec::new();
        for j in 0..3 {
            let p: Vec<usize> = vec![785usize, 3974, 13, 264][..(2 + j)].to_vec();
            let mut s = Slot { page_table_host: vec![-1i32; n_logical], owned_blocks: Vec::new(),
                               next_pos: 0, last: 0 };
            sess.prefill_for_slot(&mut s.page_table_host, &mut s.owned_blocks, &mut s.next_pos, &p)
                .expect("prefill fresh");
            s.last = *p.last().unwrap();
            fresh.push(s);
        }
        for _ in 0..5 {
            // A + the 3 fresh, all in one batch (heterogeneous positions: A high, fresh low)
            sess.slot_ensure_block(a.next_pos, &mut a.page_table_host, &mut a.owned_blocks).expect("ensure A2");
            for s in fresh.iter_mut() {
                sess.slot_ensure_block(s.next_pos, &mut s.page_table_host, &mut s.owned_blocks).expect("ensure fresh");
            }
            let mut pts: Vec<Vec<i32>> = vec![a.page_table_host.clone()];
            let mut lids: Vec<usize> = vec![a.last];
            let mut poss: Vec<i32> = vec![a.next_pos];
            for s in fresh.iter() { pts.push(s.page_table_host.clone()); lids.push(s.last); poss.push(s.next_pos); }
            let logits = sess.step_logits_for_batch(&pts, &lids, &mut poss);
            a.next_pos = poss[0]; a.last = argmax(&logits[0]);
            for (k, s) in fresh.iter_mut().enumerate() { s.next_pos = poss[k+1]; s.last = argmax(&logits[k+1]); }
        }
        // 3. Retire the fresh slots (pool churn: free their blocks back).
        for s in fresh.iter_mut() { sess.slot_release_blocks(&mut s.owned_blocks); }
        eprintln!("[graph-churn] round {} OK (A pos={})", round, a.next_pos);
        if a.next_pos as usize > MAX_SEQ - 64 { break; }   // stay well under cap
    }
    sess.slot_release_blocks(&mut a.owned_blocks);
    eprintln!("[graph-churn] PASS — {} rounds survived (graph={})",
        rounds, if std::env::var("AETHER_NO_GRAPH").is_ok() { "OFF" } else { "ON" });
}

#[test]
fn batched_decode_crash_sweep() {
    let Some(path) = gguf_path() else { return; };
    let steps: usize = std::env::var("AETHER_REPRO_STEPS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(24);

    let block_size = 4i32;
    let n_logical = ((MAX_SEQ as i32 + block_size - 1) / block_size) as usize;
    let total_blocks = 8 * n_logical as i32;

    let cfg = { QwenSession::new(&path).expect("probe").cfg.clone() };
    assert!(cfg.kv_lora_rank == 0 && cfg.n_experts == 0, "needs dense arch (Qwen2.5)");

    let pool: Arc<SharedKvPool> =
        SharedKvPool::new_for_shape(total_blocks, block_size, cfg.n_layers, cfg.d_kv);
    let mut sess = QwenSession::new_paged_with_pool(&path, pool.clone()).expect("pooled");
    assert!(sess.is_batchable(), "must be batchable");

    // Sweep b. A single env AETHER_REPRO_B pins one size; default sweeps 2..=8.
    let bs: Vec<usize> = match std::env::var("AETHER_REPRO_B").ok().and_then(|s| s.parse().ok()) {
        Some(b) => vec![b],
        None => vec![2, 3, 4, 5, 6, 7, 8],
    };
    for &b in &bs {
        run_batch(&mut sess, b, steps, n_logical);
    }
    eprintln!("[crash-repro] PASS — all batch sizes {:?} survived {} steps", bs, steps);
}
