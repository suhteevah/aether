//! Batched-decode end-to-end parity (FR-19.5-extra-deep Phase 2b-2b).
//!
//! This is the witness for the continuous-batching throughput win: it proves
//! that `QwenSession::step_logits_for_batch` — which fuses B requests at
//! heterogeneous decode positions into ONE forward pass (Q4_K weight-reuse
//! seqB matmul + per-request hetero RoPE/append/attention) — produces the
//! SAME greedy token stream per request as B independent serial
//! `step_logits_for_slot` decodes.
//!
//! Both paths share one `QwenSession` + `SharedKvPool`; each logical request
//! owns its own `page_table_host` / `owned_blocks` (exactly as the scheduler
//! drives them).  We run two distinct prompts, decode N greedy steps each way,
//! and assert the per-request token lists match.
//!
//! Skipped without `--features cuda` AND without the Qwen2.5-7B GGUF.
//! AETHER_QWEN25_GGUF overrides the path; default is matt-voice's ollama blob.
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use aether_rt::serving::{QwenSession, SharedKvPool, MAX_SEQ};
use std::sync::Arc;

const DEFAULT_GGUF: &str = "C:/Users/Matt/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

fn gguf_path() -> Option<String> {
    let p = std::env::var("AETHER_QWEN25_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    if std::path::Path::new(&p).exists() {
        Some(p)
    } else {
        eprintln!("[batched-parity] GGUF not at {} — skipping", p);
        None
    }
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v { best_v = v; best = i; }
    }
    best
}

/// One independent decode slot: its own page table + owned blocks + position.
struct Slot {
    page_table_host: Vec<i32>,
    owned_blocks: Vec<i32>,
    next_pos: i32,
    last: usize,
}

impl Slot {
    /// Pre-size the page table to `n_logical` (matches the scheduler's
    /// `SessionSlot::new`); the plain i32 h2d requires the host page-table
    /// length to equal the device buffer length.
    fn new(n_logical: usize) -> Self {
        Slot {
            page_table_host: vec![-1i32; n_logical.max(1)],
            owned_blocks: Vec::new(),
            next_pos: 0,
            last: 0,
        }
    }
}

#[test]
fn batched_decode_matches_serial_per_slot() {
    let Some(path) = gguf_path() else { return; };

    // Two distinct prompts → two independent sequences.
    let prompts: [Vec<usize>; 2] = [
        vec![9707usize, 11, 1879, 0],   // "Hello, world!"
        vec![785usize, 3974, 13],       // a different short prompt
    ];
    let steps = 6usize;

    let block_size = 4i32;
    let n_logical = (MAX_SEQ as i32 + block_size - 1) / block_size;
    let total_blocks = 8 * n_logical;

    // ---- Phase 1: serial reference via the proven single-stream `generate()`
    // path (the same greedy decode `qwen25_paged_parity` validates).  A
    // separate non-pooled session, dropped before the pooled one loads so the
    // two 7B models never coexist on the 8 GB card. ----
    let mut serial_out: Vec<Vec<usize>> = Vec::new();
    {
        eprintln!("[batched-parity] loading reference (new_paged) session...");
        let mut ref_sess = QwenSession::new_paged(&path).expect("reference session");
        for prompt in prompts.iter() {
            let toks = ref_sess.generate(prompt, steps, None);
            eprintln!("[batched-parity] serial(generate) seq -> {:?}", toks);
            serial_out.push(toks);
        }
        // ref_sess dropped here → frees ~4.7 GB before the pooled load.
    }

    // ---- Phase 2: batched — decode both sequences in lockstep, fused, on a
    // pooled session. ----
    eprintln!("[batched-parity] loading pooled session...");
    let cfg = {
        let probe = QwenSession::new(&path).expect("probe session");
        probe.cfg.clone()
    };
    assert!(cfg.kv_lora_rank == 0 && cfg.n_experts == 0,
        "this parity test targets the standard dense arch (Qwen2.5)");
    let pool: Arc<SharedKvPool> =
        SharedKvPool::new_for_shape(total_blocks, block_size, cfg.n_layers, cfg.d_kv);
    let mut sess = QwenSession::new_paged_with_pool(&path, pool.clone())
        .expect("pooled session");
    assert!(sess.is_batchable(), "session must report batchable for Qwen2.5");

    let mut slots: Vec<Slot> = Vec::new();
    for prompt in prompts.iter() {
        let mut s = Slot::new(n_logical as usize);
        sess.prefill_for_slot(&mut s.page_table_host, &mut s.owned_blocks,
            &mut s.next_pos, prompt).expect("prefill batched");
        s.last = *prompt.last().unwrap();
        slots.push(s);
    }
    let mut batched_out: Vec<Vec<usize>> = vec![Vec::new(); slots.len()];
    for _ in 0..steps {
        for s in slots.iter_mut() {
            sess.slot_ensure_block(s.next_pos, &mut s.page_table_host,
                &mut s.owned_blocks).expect("ensure batched");
        }
        let page_tables: Vec<Vec<i32>> =
            slots.iter().map(|s| s.page_table_host.clone()).collect();
        let last_ids: Vec<usize> = slots.iter().map(|s| s.last).collect();
        let mut positions: Vec<i32> = slots.iter().map(|s| s.next_pos).collect();
        let logits_batch = sess.step_logits_for_batch(
            &page_tables, &last_ids, &mut positions);
        for (i, s) in slots.iter_mut().enumerate() {
            s.next_pos = positions[i];
            let id = argmax(&logits_batch[i]);
            batched_out[i].push(id);
            s.last = id;
        }
    }
    for s in slots.iter_mut() { sess.slot_release_blocks(&mut s.owned_blocks); }
    for (i, toks) in batched_out.iter().enumerate() {
        eprintln!("[batched-parity] batched seq {} -> {:?}", i, toks);
    }

    // ---- Assert: per-request token streams identical over the common
    // prefix (generate() may stop early at EOS; batched always runs `steps`). ----
    for i in 0..prompts.len() {
        let n = serial_out[i].len().min(batched_out[i].len());
        assert!(n >= 1, "request {}: no tokens to compare", i);
        assert_eq!(&serial_out[i][..n], &batched_out[i][..n],
            "request {}: batched decode diverged from serial generate()\n  serial : {:?}\n  batched: {:?}",
            i, serial_out[i], batched_out[i]);
    }
    eprintln!("[batched-parity] PASS — {} requests token-identical \
        (batched == single-stream generate)", prompts.len());
}
