//! FR-19.5-extra-deep — Continuous-batching scheduler for aether-serve.
//!
//! Up to `max_concurrent` chat requests share a single `QwenSession` +
//! `SharedKvPool`.  A background worker thread drives a multiplexed
//! decode loop: each tick the worker walks all active slots, runs ONE
//! forward step against the slot's own page-table + position, samples
//! per the slot's own RNG / penalty / stop-string state, and returns
//! generated tokens via a per-request `oneshot` channel.
//!
//! Architecture (Phase 1):
//!
//!   ┌────────────────────────────────────────────────────────────┐
//!   │ aether-serve HTTP thread (one per accepted connection)     │
//!   │   → BatchScheduler::submit(req) → blocks on done.recv()    │
//!   └─────────────────────┬──────────────────────────────────────┘
//!                         │
//!         ┌───────────────▼────────────────┐
//!         │ shared:                         │
//!         │   pending (Mutex<VecDeque>),    │
//!         │   active  (Mutex<Vec<Slot>>),   │
//!         │   notify  (Condvar)             │
//!         └───────────────┬────────────────┘
//!                         │
//!   ┌─────────────────────▼─────────────────────────────────────┐
//!   │ scheduler worker thread (single thread, owns the GPU)      │
//!   │   loop:                                                    │
//!   │     - admit ≤ max_concurrent new requests (prefill each)   │
//!   │     - for each active slot:                                │
//!   │         lock session → step_logits_for_slot → unlock       │
//!   │         sample / penalty / stop-check (no lock)            │
//!   │     - retire any slot that hit stop / max_tokens           │
//!   └────────────────────────────────────────────────────────────┘
//!
//! Phase-1 limits documented in the parent task description:
//!   - Prefill is per-request (NOT batched).  Decode tick is multiplexed
//!     across slots.  Batched-prefill is a follow-on FR.
//!   - No speculative decoding, no prefix caching.
//!   - All slots run on the same GPU (the model is single-GPU); cross-
//!     GPU sharding is orthogonal.
//!
//! Critical correctness gates (enforced by SessionSlot design):
//!   - Per-slot xorshift RNG state, never shared.
//!   - Per-slot `seen` counts for repetition penalty.
//!   - Per-slot stop_token + stop_strings checking.
//!   - Per-slot decoded `running_text` for the stop-string tail match.
//!   - GPU mutex held ONLY during the actual forward+H2D — sampling and
//!     bookkeeping run lock-free.

#![cfg(feature = "cuda")]

use std::collections::{HashMap, VecDeque};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;

use crate::serving::{
    apply_logit_bias, apply_repetition_penalty, argmax_external,
    sample_from_logits_v2, seed_rng_external,
    QwenSession, SamplingParams, SharedKvPool,
};

/// One in-flight chat request.  All mutable per-request state lives
/// here; the scheduler walks active slots round-robin under the GPU
/// mutex but does sampling + stop-checking outside the lock.
pub struct SessionSlot {
    /// Slot-local logical→physical page table mirror.  H2D'd into the
    /// session's shared `page_table_dev` at the top of each decode
    /// tick for THIS slot.
    pub page_table_host: Vec<i32>,
    /// Physical block IDs (in the shared pool) this slot currently
    /// holds.  Returned to the pool on slot retirement.
    pub owned_blocks: Vec<i32>,
    /// Next decode position the slot will consume.
    pub next_pos: i32,
    /// The token id the slot will feed into the next decode step
    /// (initially the last prompt token; thereafter the last sampled).
    pub last_token: usize,
    /// Generated suffix (does NOT include prompt).
    pub generated: Vec<usize>,
    /// Sampler config — per-slot so each request can have its own
    /// temperature, seed, logit_bias, etc.
    pub params: SamplingParams,
    /// Optional token id that ends generation early.
    pub stop_token: Option<usize>,
    /// Decoded stop strings.  Same semantics as `generate_sampled_v2`.
    pub stop_strings: Vec<String>,
    /// Decoded text accumulator for stop-string tail match.
    pub running_text: String,
    /// Maximum tokens this request will produce.
    pub max_tokens: usize,
    /// Per-slot RNG state (xorshift64).  Initialised from
    /// `params.seed` if present, else seeded per-slot.
    pub rng: u64,
    /// Per-slot repetition-penalty token counts.
    pub seen: HashMap<usize, u32>,
    /// Optional stream callback: token + decoded piece per generated
    /// token.  Phase-1 just buffers; the streaming HTTP path is left
    /// to the existing single-session streaming code.
    pub stream_callback: Option<Box<dyn FnMut(usize, &str) + Send>>,
}

impl SessionSlot {
    /// Construct an empty slot.  Caller fills `params`, `stop_token`,
    /// `stop_strings`, `max_tokens` after this returns and BEFORE
    /// submitting the slot.
    pub fn new(n_logical: usize) -> Self {
        Self {
            page_table_host: vec![-1i32; n_logical.max(1)],
            owned_blocks: Vec::new(),
            next_pos: 0,
            last_token: 0,
            generated: Vec::with_capacity(64),
            params: SamplingParams::greedy(),
            stop_token: None,
            stop_strings: Vec::new(),
            running_text: String::new(),
            max_tokens: 0,
            rng: 0,
            seen: HashMap::new(),
            stream_callback: None,
        }
    }
}

/// Caller-visible request envelope.  Constructed by the HTTP handler,
/// submitted to the scheduler, awaited on `done` for the final
/// generated id list (Phase 1 returns the full list on completion;
/// streaming uses the optional callback inside `SessionSlot`).
pub struct BatchRequest {
    pub prompt_ids: Vec<usize>,
    pub max_tokens: usize,
    pub stop_token: Option<usize>,
    pub params: SamplingParams,
    pub stop_strings: Vec<String>,
    /// One-shot channel that receives the generated token ids when the
    /// slot retires.  `Err` if the scheduler couldn't admit the request
    /// (e.g. pool exhausted, prefill failed) or shut down mid-flight.
    pub done: mpsc::Sender<Result<Vec<usize>, String>>,
}

/// Internal shared state between the scheduler worker and submitters.
struct Shared {
    pending: Mutex<VecDeque<BatchRequest>>,
    /// Wakes the worker when a request is submitted OR when the
    /// scheduler is shutting down.
    notify: Condvar,
    /// Lifecycle flag — set by `Drop::drop` to tell the worker to exit.
    shutdown: Mutex<bool>,
}

/// Continuous-batching scheduler.  Drop releases the worker thread.
pub struct BatchScheduler {
    shared: Arc<Shared>,
    worker: Option<thread::JoinHandle<()>>,
    max_concurrent: usize,
    /// Shared session — held by both the scheduler worker (for forward
    /// passes) and by HTTP threads (for short tokenizer / chat-template
    /// calls that don't touch the GPU).  The scheduler worker holds
    /// the lock ONLY during the forward pass + H2D / D2H; sampling and
    /// per-slot bookkeeping run lock-free.
    pub session: Arc<Mutex<QwenSession>>,
}

impl BatchScheduler {
    /// Build a scheduler around a pool-backed session.  The session
    /// MUST have been constructed via `QwenSession::new_paged_with_pool`
    /// (else `is_pool_backed()` returns false and this returns Err).
    ///
    /// `max_concurrent` caps how many requests run concurrently;
    /// excess submissions queue in FIFO order.  The scheduler picks
    /// new requests up as soon as a slot retires.
    pub fn new(
        session: QwenSession,
        max_concurrent: usize,
    ) -> Result<Self, String> {
        if max_concurrent == 0 {
            return Err("max_concurrent must be >= 1".into());
        }
        if !session.is_pool_backed() {
            return Err(
                "BatchScheduler requires a pool-backed session (build via \
                 QwenSession::new_paged_with_pool)".into());
        }
        let session = Arc::new(Mutex::new(session));
        let shared = Arc::new(Shared {
            pending: Mutex::new(VecDeque::new()),
            notify: Condvar::new(),
            shutdown: Mutex::new(false),
        });
        let worker_shared = shared.clone();
        let worker_session = session.clone();
        let worker = thread::Builder::new()
            .name("aether-batch-sched".into())
            .spawn(move || run_worker(worker_shared, worker_session, max_concurrent))
            .map_err(|e| format!("spawn scheduler worker: {}", e))?;
        Ok(Self {
            shared,
            worker: Some(worker),
            max_concurrent,
            session,
        })
    }

    /// Configured per-instance concurrency cap.
    pub fn max_concurrent(&self) -> usize { self.max_concurrent }

    /// Submit a request.  Non-blocking on the caller (the scheduler
    /// worker performs the actual prefill + decode).  The HTTP handler
    /// then blocks on `req.done` to receive the final id list.
    ///
    /// Returns Err if the scheduler has already been shut down.
    pub fn submit(&self, req: BatchRequest) -> Result<(), String> {
        if *self.shared.shutdown.lock().unwrap() {
            return Err("scheduler shut down".into());
        }
        self.shared.pending.lock().unwrap().push_back(req);
        self.shared.notify.notify_one();
        Ok(())
    }

    /// Convenience: submit + block until completion.  Mirrors the
    /// single-session `generate_sampled_v2` interface.
    pub fn generate_blocking(
        &self,
        prompt_ids: Vec<usize>,
        max_tokens: usize,
        stop_token: Option<usize>,
        params: SamplingParams,
        stop_strings: Vec<String>,
    ) -> Result<Vec<usize>, String> {
        let (tx, rx) = mpsc::channel();
        self.submit(BatchRequest {
            prompt_ids, max_tokens, stop_token,
            params, stop_strings,
            done: tx,
        })?;
        rx.recv().map_err(|e| format!("scheduler done channel: {}", e))?
    }
}

impl Drop for BatchScheduler {
    fn drop(&mut self) {
        *self.shared.shutdown.lock().unwrap() = true;
        self.shared.notify.notify_all();
        if let Some(j) = self.worker.take() {
            // Best-effort: if a request mid-decode is panicking the
            // worker thread will still unwind cleanly.
            let _ = j.join();
        }
    }
}

// --------------------------------------------------------------------
// Worker
// --------------------------------------------------------------------

/// Block size used for slot page tables — derived from the session's
/// paged_cfg at admission time.  Cached locally to avoid re-fetching.
struct SlotEntry {
    slot: SessionSlot,
    done: mpsc::Sender<Result<Vec<usize>, String>>,
}

fn run_worker(
    shared: Arc<Shared>,
    session: Arc<Mutex<QwenSession>>,
    max_concurrent: usize,
) {
    // Read once — the model shape doesn't change after construction.
    let n_logical = session.lock().unwrap().paged_n_logical();
    let mut active: Vec<SlotEntry> = Vec::with_capacity(max_concurrent);

    loop {
        // 1. Shutdown check.
        if *shared.shutdown.lock().unwrap() {
            let mut sess = session.lock().unwrap();
            for mut e in active.drain(..) {
                sess.slot_release_blocks(&mut e.slot.owned_blocks);
                let _ = e.done.send(Err("scheduler shut down mid-flight".into()));
            }
            drop(sess);
            let mut g = shared.pending.lock().unwrap();
            while let Some(req) = g.pop_front() {
                let _ = req.done.send(Err("scheduler shut down before admission".into()));
            }
            break;
        }

        // 2. Admit new requests up to max_concurrent.
        while active.len() < max_concurrent {
            let req_opt = shared.pending.lock().unwrap().pop_front();
            let Some(req) = req_opt else { break; };
            match admit_request(&session, n_logical, req) {
                Ok(entry) => active.push(entry),
                Err(()) => { /* admit_request already signalled done with the error */ }
            }
        }

        // 3. If still idle, wait for a submission (or shutdown).
        if active.is_empty() {
            let g = shared.pending.lock().unwrap();
            if g.is_empty() && !*shared.shutdown.lock().unwrap() {
                // Drop the returned guard immediately — we only want
                // the condvar wait side-effect, not to hold the lock.
                let _guard = shared.notify.wait(g).unwrap();
            }
            continue;
        }

        // 4. One decode tick per active slot.  GPU lock is per-slot so
        //    sampling + stop-string decode for slot N can run while
        //    slot N+1's HTTP encoder holds the session for tokenizer
        //    work — minimizes scheduler-vs-encoder contention.
        let mut retired: Vec<usize> = Vec::new();
        for (i, entry) in active.iter_mut().enumerate() {
            let done = match step_slot(&session, &mut entry.slot) {
                Ok(d) => d,
                Err(e) => {
                    let _ = entry.done.send(Err(e));
                    retired.push(i);
                    continue;
                }
            };
            if done {
                retired.push(i);
            }
        }

        // 5. Retire finished slots (reverse-iterate to keep indices valid).
        for &idx in retired.iter().rev() {
            let mut entry = active.swap_remove(idx);
            {
                let sess = session.lock().unwrap();
                sess.slot_release_blocks(&mut entry.slot.owned_blocks);
            }
            let result = std::mem::take(&mut entry.slot.generated);
            let _ = entry.done.send(Ok(result));
        }
    }
}

/// Prefill the request onto a fresh slot, register it as active.
fn admit_request(
    session: &Arc<Mutex<QwenSession>>,
    n_logical: usize,
    req: BatchRequest,
) -> Result<SlotEntry, ()> {
    let BatchRequest {
        prompt_ids, max_tokens, stop_token,
        params, stop_strings, done,
    } = req;

    if prompt_ids.is_empty() {
        let _ = done.send(Err("prompt_ids empty".into()));
        return Err(());
    }
    let vocab = session.lock().unwrap().vocab();
    if let Some(&bad) = prompt_ids.iter().find(|&&i| i >= vocab) {
        let _ = done.send(Err(format!(
            "prompt_ids contains token id {} out of vocab (vocab_size={})",
            bad, vocab)));
        return Err(());
    }
    if max_tokens == 0 {
        let _ = done.send(Ok(Vec::new()));
        return Err(());
    }

    let mut slot = SessionSlot::new(n_logical);
    slot.rng = params.seed.unwrap_or_else(seed_rng_external);
    if slot.rng == 0 { slot.rng = seed_rng_external(); }
    slot.params = params;
    slot.stop_token = stop_token;
    slot.stop_strings = stop_strings;
    slot.max_tokens = max_tokens;
    slot.last_token = *prompt_ids.last().unwrap();

    // Phase-1 prefill: per-slot serial prefill, GPU lock held for the
    // duration of THIS request's prefill.  Other admitted slots can't
    // decode during prefill — Phase-2 will batch the prefill itself.
    let prefill_err = {
        let mut sess = session.lock().unwrap();
        sess.prefill_for_slot(
            &mut slot.page_table_host,
            &mut slot.owned_blocks,
            &mut slot.next_pos,
            &prompt_ids,
        )
    };
    if let Err(e) = prefill_err {
        let mut sess = session.lock().unwrap();
        sess.slot_release_blocks(&mut slot.owned_blocks);
        let _ = done.send(Err(format!("prefill failed: {}", e)));
        return Err(());
    }

    Ok(SlotEntry { slot, done })
}

/// One decode tick for one slot.  Returns Ok(true) when the slot is
/// done (hit stop / max / max-seq), Ok(false) otherwise.  Sampling and
/// stop-checking are deliberately OUTSIDE the GPU-critical section
/// (which is just the `step_logits_for_slot` call).
fn step_slot(
    session: &Arc<Mutex<QwenSession>>,
    slot: &mut SessionSlot,
) -> Result<bool, String> {
    // Pool-block alloc + GPU forward live under one lock acquisition —
    // they're cheap to keep grouped and the alloc reads pool state
    // anyway (small Mutex inside SharedKvPool).
    let mut logits = {
        let mut sess = session.lock().unwrap();
        if let Err(e) = sess.slot_ensure_block(
            slot.next_pos, &mut slot.page_table_host, &mut slot.owned_blocks,
        ) {
            return Err(format!("pool exhausted at pos {}: {}", slot.next_pos, e));
        }
        sess.step_logits_for_slot(
            &slot.page_table_host,
            &mut slot.next_pos,
            slot.last_token,
        )
    };
    let max_pos = session.lock().unwrap().max_pos();

    // Per-slot logit biases + repetition penalties + sample.
    if !slot.params.logit_bias.is_empty() {
        apply_logit_bias(&mut logits, &slot.params.logit_bias);
    }
    if slot.params.presence_penalty != 0.0 || slot.params.frequency_penalty != 0.0 {
        apply_repetition_penalty(&mut logits, &slot.seen,
            slot.params.presence_penalty, slot.params.frequency_penalty);
    }
    let id = if slot.params.temperature <= 0.0 {
        argmax_external(&logits)
    } else {
        sample_from_logits_v2(&mut logits,
            slot.params.temperature, slot.params.top_p, slot.params.top_k,
            &mut slot.rng)
    };

    // Stop-token check happens BEFORE the token is recorded (matching
    // the legacy generate_sampled_v2 semantics).
    if Some(id) == slot.stop_token {
        return Ok(true);
    }

    *slot.seen.entry(id).or_insert(0) += 1;
    slot.generated.push(id);

    // Stop-strings: decode just the new piece, append to running_text,
    // check tail-match.  Decoding ONE id is cheap.
    if !slot.stop_strings.is_empty() {
        let piece = session.lock().unwrap().decode_ids(&[id]);
        slot.running_text.push_str(&piece);
        let mut hit_len: Option<usize> = None;
        for s in &slot.stop_strings {
            if slot.running_text.ends_with(s) {
                hit_len = Some(s.len());
                break;
            }
        }
        if let Some(slen) = hit_len {
            let target = slot.running_text.len().saturating_sub(slen);
            let mut cur_text = slot.running_text.clone();
            while !slot.generated.is_empty() && cur_text.len() > target {
                slot.generated.pop();
                cur_text = session.lock().unwrap().decode_ids(&slot.generated);
            }
            return Ok(true);
        }
    }

    // Optional streaming hook.
    if let Some(cb) = slot.stream_callback.as_mut() {
        let piece = session.lock().unwrap().decode_ids(&[id]);
        cb(id, &piece);
    }

    slot.last_token = id;

    // Termination checks.
    if slot.generated.len() >= slot.max_tokens { return Ok(true); }
    if slot.next_pos >= max_pos { return Ok(true); }

    Ok(false)
}

// --------------------------------------------------------------------
// Public constructor helper — wires the standard `QwenSession +
// SharedKvPool` setup that aether-serve uses.  Keeps the trainer bin
// dependency-free of pool-allocation internals.
// --------------------------------------------------------------------

/// Convenience: open a GGUF + allocate a `SharedKvPool` sized for at
/// least `max_concurrent * MAX_SEQ` tokens, build the BatchScheduler,
/// and return both the scheduler and the kept-alive pool handle.  The
/// pool is held by the scheduler's session as well, but returning it
/// here lets callers introspect pool-utilization metrics if they want.
pub fn open_for_serve(
    gguf_path: &str,
    max_concurrent: usize,
    block_size: i32,
    blocks_per_slot: i32,
) -> Result<(BatchScheduler, Arc<SharedKvPool>), String> {
    // Build a probe session to read shape metadata for pool sizing.
    // This is cheap relative to the real load — we discard it after
    // reading n_layers + d_kv.  Phase-2 could refactor SharedKvPool to
    // be shape-inferred during open(), but for now a brief probe is
    // simpler than threading an extra config struct.
    let probe = QwenSession::new(gguf_path)?;
    let cfg = probe.cfg.clone();
    drop(probe);

    let total_blocks = (max_concurrent as i32) * blocks_per_slot;
    let pool = SharedKvPool::new_for_shape(
        total_blocks, block_size, cfg.n_layers, cfg.d_kv);
    let session = QwenSession::new_paged_with_pool(gguf_path, pool.clone())?;
    let sched = BatchScheduler::new(session, max_concurrent)?;
    Ok((sched, pool))
}

// --------------------------------------------------------------------
// Public helpers — let the HTTP layer do tokenizer work + chat-template
// rendering without leaking the internal Arc<Mutex<QwenSession>> shape.
// These all briefly lock the shared session; the scheduler worker is
// also locking it for forward passes, so high-volume encode storms will
// serialize against decode.  In practice encoding is sub-millisecond
// and only happens once per request.
// --------------------------------------------------------------------

impl BatchScheduler {
    /// Encode arbitrary UTF-8 text → token ids using the session's BPE
    /// tokenizer.  Delegates to `QwenSession::encode_text` under the
    /// shared lock; returns empty vec if the tokenizer wasn't loaded.
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        self.session.lock().unwrap().encode_text(text)
    }

    /// Encode text with chat-template special markers preserved.
    pub fn encode_text_with_specials(&self, text: &str) -> Vec<usize> {
        self.session.lock().unwrap().encode_text_with_specials(text)
    }

    /// Render messages through the session's loaded chat template (or
    /// a per-arch fallback) → wire text.  Returns None if no template
    /// is available; caller should plain-text-encode in that case.
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> Option<String> {
        self.session.lock().unwrap().apply_chat_template(messages)
    }

    /// Decode token ids → UTF-8 text.
    pub fn decode_ids(&self, ids: &[usize]) -> String {
        self.session.lock().unwrap().decode_ids(ids)
    }

    /// Model vocab size — for prompt_id validation in HTTP handlers.
    pub fn vocab(&self) -> usize {
        self.session.lock().unwrap().vocab()
    }

    /// Configured EOS token id (or `-1` if metadata absent).
    pub fn eos_token(&self) -> i32 {
        self.session.lock().unwrap().eos_token
    }
}
