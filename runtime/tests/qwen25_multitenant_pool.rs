//! Multi-tenant SharedKvPool parity test (FR-19.4-extra-tenant).
//!
//! Creates ONE `SharedKvPool` and binds TWO `PagedQwenSession` instances to it.
//! Each session runs a different prompt through real Qwen2.5-7B inference and
//! must produce token IDs identical to running the SAME prompt through a fresh
//! single-tenant `QwenSession::new`.  This proves:
//!   1. The pool's per-layer GPU buffers are correctly shared across sessions
//!      (kvs[layer].{k,v}_cache = pool.pool_{k,v}(layer)).
//!   2. Per-session page tables keep the sessions' KV state independent —
//!      session A's writes go to its allocated blocks, session B's to its own.
//!   3. Dynamic block allocation walks correctly across token-boundary
//!      crossings (ensure_block_for_position).
//!   4. Drop returns blocks to the pool (verified via n_allocated() before/after).
//!
//! Skipped without `--features cuda` AND without the Qwen2.5-7B GGUF.
//!
//! roadmap: P19.4

#![cfg(feature = "cuda")]

use aether_rt::serving::{QwenSession, SharedKvPool};

const DEFAULT_GGUF: &str = "C:/Users/Matt/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

fn gguf_path() -> Option<String> {
    let p = std::env::var("AETHER_QWEN25_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    if std::path::Path::new(&p).exists() { Some(p) } else {
        eprintln!("[mt-pool] GGUF not at {} — skipping", p);
        None
    }
}

#[test]
fn shared_pool_two_sessions_identical_to_single_tenant() {
    let Some(path) = gguf_path() else { return; };

    let prompt_a = [9707usize, 11, 1879, 0];      // "Hello, world!"
    let prompt_b = [40usize, 1079, 264, 220, 17]; // different starter
    let max_tokens = 6usize;

    // Reference run: two fresh single-tenant sessions in sequence.
    eprintln!("[mt-pool] reference: running prompt A through single-tenant session...");
    let t0 = std::time::Instant::now();
    let ref_a;
    {
        let mut s = QwenSession::new(&path).expect("ref session A");
        ref_a = s.generate(&prompt_a, max_tokens, None);
    }
    let ref_b;
    {
        let mut s = QwenSession::new(&path).expect("ref session B");
        ref_b = s.generate(&prompt_b, max_tokens, None);
    }
    eprintln!("[mt-pool] reference A: {:?} ({:?})", ref_a, &ref_a[..]);
    eprintln!("[mt-pool] reference B: {:?} ({:?})", ref_b, &ref_b[..]);
    eprintln!("[mt-pool] reference total: {:.2}s", t0.elapsed().as_secs_f32());

    // Multi-tenant run: ONE pool, TWO paged sessions.
    eprintln!("[mt-pool] multi-tenant: allocating pool (32 blocks × 4 tokens = 128 token capacity)...");
    let pool = SharedKvPool::new(32, 4);
    assert_eq!(pool.n_allocated(), 0, "pool starts empty");

    let mut sess_a = QwenSession::new_paged_with_pool(&path, pool.clone())
        .expect("paged session A");
    eprintln!("[mt-pool] sess_a constructed; pool allocated={}", pool.n_allocated());
    assert!(pool.n_allocated() >= 1, "sess_a should own at least 1 block");

    let mut sess_b = QwenSession::new_paged_with_pool(&path, pool.clone())
        .expect("paged session B");
    eprintln!("[mt-pool] sess_b constructed; pool allocated={}", pool.n_allocated());
    assert!(pool.n_allocated() >= 2, "both sessions should own >=1 block each");

    // Run both sessions; the prompt+max_tokens=10 > block_size*2=8 forces at
    // least one boundary crossing -> ensure_block_for_position runs.
    let mt_a = sess_a.generate(&prompt_a, max_tokens, None);
    let mt_b = sess_b.generate(&prompt_b, max_tokens, None);
    eprintln!("[mt-pool] multi-tenant A: {:?}", mt_a);
    eprintln!("[mt-pool] multi-tenant B: {:?}", mt_b);
    let alloc_during = pool.n_allocated();
    eprintln!("[mt-pool] pool allocated mid-test: {}", alloc_during);

    assert_eq!(mt_a, ref_a,
        "session A's tokens diverged from single-tenant reference:\n  ref: {:?}\n  mt:  {:?}", ref_a, mt_a);
    assert_eq!(mt_b, ref_b,
        "session B's tokens diverged from single-tenant reference:\n  ref: {:?}\n  mt:  {:?}", ref_b, mt_b);

    // Drop both sessions and verify the pool reclaims their blocks.
    drop(sess_a);
    drop(sess_b);
    assert_eq!(pool.n_allocated(), 0,
        "pool should be fully reclaimed after both sessions drop");

    eprintln!("[mt-pool] PASS — 2 sessions over 1 shared pool, both bit-identical to single-tenant baseline");
}
