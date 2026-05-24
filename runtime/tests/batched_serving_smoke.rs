//! FR-19.5-extra-deep — BatchScheduler smoke test.
//!
//! Spawns 4 concurrent `generate` calls against a real GGUF-backed
//! `QwenSession` (wrapped in a `BatchScheduler` over a `SharedKvPool`)
//! and asserts each call returns a non-empty id stream.
//!
//! Gated behind `#[ignore]` because a real model file is needed.  Run
//! explicitly when a GGUF is on disk and the build was made with
//! `--features cuda`:
//!
//!     export AETHER_TEST_GGUF=$HOME/path/to/qwen2.5-3b-q4_k_m.gguf
//!     cargo test --release --features cuda -p aether_rt \
//!         batched_serving_smoke -- --ignored --nocapture
//!
//! The path is read from `AETHER_TEST_GGUF`; absence skips with a clear
//! note rather than failing.
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use std::sync::mpsc;
use std::thread;

use aether_rt::batched_serving::{
    open_for_serve, BatchRequest, SessionSlot,
};
use aether_rt::serving::SamplingParams;

#[test]
#[ignore = "requires GGUF on disk via AETHER_TEST_GGUF"]
fn batched_serving_smoke_four_concurrent_requests() {
    let gguf = match std::env::var("AETHER_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!("[batched_serving_smoke] AETHER_TEST_GGUF not set, skipping");
            return;
        }
    };
    if !std::path::Path::new(&gguf).exists() {
        eprintln!("[batched_serving_smoke] {} missing on disk, skipping", gguf);
        return;
    }

    // 4 slots × 8 blocks/slot × 4 tokens/block = 128-token capacity.
    // Sized for short smoke prompts.
    const MAX_CONCURRENT: usize = 4;
    const BLOCKS_PER_SLOT: i32 = 8;
    const BLOCK_SIZE: i32 = 4;

    let (sched, pool) = open_for_serve(&gguf, MAX_CONCURRENT, BLOCK_SIZE, BLOCKS_PER_SLOT)
        .expect("open_for_serve");
    eprintln!("[smoke] pool allocated; {} blocks, block_size={}",
        pool.n_blocks, pool.block_size);

    // Submit 4 concurrent requests with different sampling seeds so
    // their RNG paths don't accidentally collide.
    let sched = std::sync::Arc::new(sched);
    let mut joins: Vec<thread::JoinHandle<Result<Vec<usize>, String>>> = Vec::new();
    for i in 0..MAX_CONCURRENT {
        let sched = sched.clone();
        let j = thread::spawn(move || {
            let params = SamplingParams {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                seed: Some(0xC0FFEE_0000 + i as u64),
                logit_bias: std::collections::HashMap::new(),
            };
            // Use trivial prompt id list — any Qwen GGUF has these.
            // The smoke test doesn't validate output quality, just
            // that the scheduler runs N forwards without deadlock.
            let prompt_ids: Vec<usize> = vec![1, 2, 3, 4];
            sched.generate_blocking(prompt_ids, 8, None, params, Vec::new())
        });
        joins.push(j);
    }

    let mut ok_count = 0usize;
    for (i, j) in joins.into_iter().enumerate() {
        match j.join().unwrap() {
            Ok(ids) => {
                eprintln!("[smoke] req {} -> {} tokens: {:?}", i, ids.len(), ids);
                assert!(!ids.is_empty(), "request {} returned empty stream", i);
                ok_count += 1;
            }
            Err(e) => {
                panic!("request {} failed: {}", i, e);
            }
        }
    }
    assert_eq!(ok_count, MAX_CONCURRENT,
        "expected {} successful requests, got {}", MAX_CONCURRENT, ok_count);
    eprintln!("[smoke] all {} concurrent requests completed", MAX_CONCURRENT);
}

/// Lightweight sanity check that doesn't need a real model — just
/// constructs and drops a `SessionSlot` to verify default-initialization
/// and `n_logical=0` handling.  Runs in every test invocation.
#[test]
fn session_slot_default_ctor_runs() {
    let mut slot = SessionSlot::new(8);
    assert_eq!(slot.page_table_host.len(), 8);
    assert!(slot.page_table_host.iter().all(|&x| x == -1));
    assert!(slot.owned_blocks.is_empty());
    assert_eq!(slot.next_pos, 0);
    assert!(slot.generated.is_empty());

    // Mutate to confirm fields are pub.
    slot.max_tokens = 16;
    slot.stop_token = Some(99);
    slot.stop_strings.push("STOP".into());
    assert_eq!(slot.max_tokens, 16);
}

/// Verify a BatchRequest can be constructed via the public API (compile-
/// only test — no scheduler is created).  Confirms the field set is
/// stable across the public boundary.
#[test]
fn batch_request_public_api_compiles() {
    let (_tx, _rx) = mpsc::channel::<Result<Vec<usize>, String>>();
    let _req = BatchRequest {
        prompt_ids: vec![1, 2, 3],
        max_tokens: 8,
        stop_token: Some(42),
        params: SamplingParams::greedy(),
        stop_strings: vec!["</s>".into()],
        done: _tx,
    };
}
