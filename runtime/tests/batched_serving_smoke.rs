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
    open_for_serve, BatchRequest, SessionSlot, StreamEvent,
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
        stream_tx: None,
    };
}

/// Compile + variant sanity for the streaming event type.  No scheduler
/// is created;  just constructs each `StreamEvent` variant and matches
/// on it to lock the public shape.  Runs in every invocation.
#[test]
fn stream_event_public_api_compiles() {
    let events = vec![
        StreamEvent::Token { id: 7, piece: "hi".into() },
        StreamEvent::Done { generated: vec![7, 8] },
        StreamEvent::Error("boom".into()),
    ];
    let mut tokens = 0;
    let mut done = 0;
    let mut errs = 0;
    for ev in events {
        match ev {
            StreamEvent::Token { id, piece } => { assert_eq!(id, 7); assert_eq!(piece, "hi"); tokens += 1; }
            StreamEvent::Done { generated } => { assert_eq!(generated, vec![7, 8]); done += 1; }
            StreamEvent::Error(e) => { assert_eq!(e, "boom"); errs += 1; }
        }
    }
    assert_eq!((tokens, done, errs), (1, 1, 1));
}

/// Gated real-GGUF streaming-equivalence test.  Submits one greedy
/// (deterministic) request via `submit_streaming`, collects the per-
/// token `Token` pieces, and asserts:
///   1. exactly one terminal `Done` arrives (no `Error`),
///   2. the concatenated streamed pieces equal `decode_ids(generated)`
///      from the `Done` event — i.e. streaming sees the same tokens, in
///      order, as the blocking path would have buffered.
#[test]
#[ignore = "requires GGUF on disk via AETHER_TEST_GGUF"]
fn batched_streaming_pieces_match_full_decode() {
    let gguf = match std::env::var("AETHER_TEST_GGUF") {
        Ok(p) => p,
        Err(_) => { eprintln!("[stream_smoke] AETHER_TEST_GGUF not set, skipping"); return; }
    };
    if !std::path::Path::new(&gguf).exists() {
        eprintln!("[stream_smoke] {} missing, skipping", gguf);
        return;
    }

    let (sched, _pool) = open_for_serve(&gguf, 2, 4, 16).expect("open_for_serve");

    // Greedy (temperature=0) → deterministic; no seed needed.
    let rx = sched.submit_streaming(
        vec![1, 2, 3, 4], 12, None, SamplingParams::greedy(), Vec::new(),
    ).expect("submit_streaming");

    let mut streamed = String::new();
    let mut streamed_ids: Vec<usize> = Vec::new();
    let mut final_ids: Option<Vec<usize>> = None;
    let mut n_token_events = 0usize;
    while let Ok(ev) = rx.recv() {
        match ev {
            StreamEvent::Token { id, piece } => {
                streamed.push_str(&piece);
                streamed_ids.push(id);
                n_token_events += 1;
            }
            StreamEvent::Done { generated } => { final_ids = Some(generated); break; }
            StreamEvent::Error(e) => panic!("stream errored: {}", e),
        }
    }

    let final_ids = final_ids.expect("no terminal Done event");
    eprintln!("[stream_smoke] {} token events, {} final ids", n_token_events, final_ids.len());
    assert!(!final_ids.is_empty(), "no tokens generated");
    // PRIMARY invariant: token events arrive 1:1 and in-order with the
    // generated id list (stop-string trimming only shrinks the final
    // list, and this prompt has no stop strings).  This is the real
    // proof that streaming sees the same decode as the blocking path.
    assert_eq!(streamed_ids, final_ids,
        "streamed ids diverge from final id list");
    // SECONDARY (best-effort) text check: per-token `decode_ids` does
    // `from_utf8_lossy`, so a multi-byte UTF-8 char split across two
    // tokens yields U+FFFD in the streamed concatenation but resolves
    // in a single full decode.  Only assert byte-exact text equality
    // when neither side contains a replacement char (no split occurred).
    let full = sched.decode_ids(&final_ids);
    if !streamed.contains('\u{FFFD}') && !full.contains('\u{FFFD}') {
        assert_eq!(streamed, full,
            "streamed text != full decode\n  stream: {:?}\n  full:   {:?}", streamed, full);
        eprintln!("[stream_smoke] streamed text matches full decode: {:?}", full);
    } else {
        eprintln!("[stream_smoke] UTF-8 split across token boundary (expected for some \
                   byte-level BPE outputs); id-stream equality already verified");
    }
}
