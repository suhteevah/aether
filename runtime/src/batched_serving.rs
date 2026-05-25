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

/// Per-token streaming event delivered from the scheduler worker thread
/// to a blocked HTTP handler over an `mpsc` channel.  One `Token` is
/// sent per generated token (the stop-token itself is NOT emitted,
/// matching `generate_sampled_v2` semantics);  exactly one terminal
/// `Done` or `Error` follows.  The HTTP layer drains the receiver,
/// writing an SSE `data:` chunk per `Token`, then `[DONE]` on `Done`.
pub enum StreamEvent {
    /// A freshly sampled token id + its decoded UTF-8 piece.
    Token { id: usize, piece: String },
    /// Generation finished cleanly (stop token / stop string / max
    /// tokens / max sequence).  Carries the full generated id list so
    /// the HTTP layer can compute usage counts without a second pass.
    Done { generated: Vec<usize> },
    /// Generation aborted (pool exhausted, prefill failure, shutdown).
    Error(String),
}

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
    /// token.  In-process consumers (tests) can use this;  the HTTP
    /// streaming path uses `stream_tx` instead (cross-thread channel).
    pub stream_callback: Option<Box<dyn FnMut(usize, &str) + Send>>,
    /// Optional cross-thread streaming channel.  When `Some`, the
    /// scheduler worker sends a `StreamEvent::Token` per generated
    /// token and a terminal `Done`/`Error` on retirement.  Used by the
    /// SSE HTTP handler to stream deltas as they're produced.
    pub stream_tx: Option<mpsc::Sender<StreamEvent>>,
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
            stream_tx: None,
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
    /// Optional per-token streaming channel.  When `Some`, the
    /// scheduler emits a `StreamEvent::Token` for each generated token
    /// and a terminal `Done`/`Error`.  The `done` channel is still
    /// fired on retirement (carrying the same id list / error), so a
    /// streaming caller may either drain `stream_rx` for deltas or
    /// block on `done` for the final list — both are consistent.
    pub stream_tx: Option<mpsc::Sender<StreamEvent>>,
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
            stream_tx: None,
        })?;
        rx.recv().map_err(|e| format!("scheduler done channel: {}", e))?
    }

    /// Submit a streaming request.  Returns a `Receiver<StreamEvent>`
    /// the caller drains for per-token deltas;  the worker sends one
    /// `Token` per generated token and exactly one terminal
    /// `Done`/`Error`.  The `done` one-shot still fires on retirement
    /// (its result is identical to the terminal stream event) but a
    /// streaming caller normally only reads the stream channel.
    ///
    /// Returns Err if the scheduler has already shut down.
    pub fn submit_streaming(
        &self,
        prompt_ids: Vec<usize>,
        max_tokens: usize,
        stop_token: Option<usize>,
        params: SamplingParams,
        stop_strings: Vec<String>,
    ) -> Result<mpsc::Receiver<StreamEvent>, String> {
        let (stream_tx, stream_rx) = mpsc::channel();
        // The `done` sender is required by the request envelope but the
        // streaming caller ignores it (terminal info arrives via the
        // stream channel).  A throwaway channel keeps the worker's
        // retirement path uniform.
        let (done_tx, _done_rx) = mpsc::channel();
        self.submit(BatchRequest {
            prompt_ids, max_tokens, stop_token,
            params, stop_strings,
            done: done_tx,
            stream_tx: Some(stream_tx),
        })?;
        Ok(stream_rx)
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
                let msg = "scheduler shut down mid-flight".to_string();
                if let Some(tx) = &e.slot.stream_tx {
                    let _ = tx.send(StreamEvent::Error(msg.clone()));
                }
                let _ = e.done.send(Err(msg));
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
        // Each retired entry carries its outcome: Ok(()) for a clean
        // finish, Err(msg) for a fault.  Terminal delivery (to both the
        // `done` one-shot and the optional `stream_tx`) happens once, in
        // step 5 — never double-send.
        let mut retired: Vec<(usize, Result<(), String>)> = Vec::new();
        // FR-19.5-extra-deep Phase 2b-2b — fuse all active slots into ONE
        // batched GPU tick when the model supports it (standard attn + dense
        // FFN) and ≥2 slots are in flight.  Q4_K weights dequant once and
        // apply to all rows (1.9× @ b=4); attention/RoPE/append run as single
        // per-request hetero launches.  Sampling stays per-slot, lock-free,
        // via the shared `consume_slot_logits`.  Falls back to the serial
        // per-slot loop for batch=1 or non-batchable arches (MLA / MoE / flex).
        let use_batched = active.len() >= 2 && session.lock().unwrap().is_batchable();
        if use_batched {
            // One lock acquisition: ensure each slot's block, gather inputs,
            // run the fused forward, write advanced positions back.
            let batch_result: Result<Vec<Vec<f32>>, (usize, String)> = {
                let mut sess = session.lock().unwrap();
                let mut ensure_err = None;
                for (i, entry) in active.iter_mut().enumerate() {
                    if let Err(e) = sess.slot_ensure_block(
                        entry.slot.next_pos,
                        &mut entry.slot.page_table_host,
                        &mut entry.slot.owned_blocks,
                    ) {
                        ensure_err = Some((i, format!(
                            "pool exhausted at pos {}: {}", entry.slot.next_pos, e)));
                        break;
                    }
                }
                match ensure_err {
                    Some(e) => Err(e),
                    None => {
                        let page_tables: Vec<Vec<i32>> =
                            active.iter().map(|e| e.slot.page_table_host.clone()).collect();
                        let last_ids: Vec<usize> =
                            active.iter().map(|e| e.slot.last_token).collect();
                        let mut positions: Vec<i32> =
                            active.iter().map(|e| e.slot.next_pos).collect();
                        let out = sess.step_logits_for_batch(
                            &page_tables, &last_ids, &mut positions);
                        for (entry, p) in active.iter_mut().zip(positions.iter()) {
                            entry.slot.next_pos = *p;
                        }
                        Ok(out)
                    }
                }
            };
            match batch_result {
                Ok(all) => {
                    for (i, (entry, logits)) in
                        active.iter_mut().zip(all.into_iter()).enumerate()
                    {
                        match consume_slot_logits(&session, &mut entry.slot, logits) {
                            Ok(true)  => retired.push((i, Ok(()))),
                            Ok(false) => { /* slot continues */ }
                            Err(e)    => retired.push((i, Err(e))),
                        }
                    }
                }
                // A block-allocation failure aborts THIS slot only; the others
                // simply tick on the next scheduler iteration (no token lost).
                Err((i, msg)) => retired.push((i, Err(msg))),
            }
        } else {
            for (i, entry) in active.iter_mut().enumerate() {
                match step_slot(&session, &mut entry.slot) {
                    Ok(true)  => retired.push((i, Ok(()))),
                    Ok(false) => { /* slot continues */ }
                    Err(e)    => retired.push((i, Err(e))),
                }
            }
        }

        // 5. Retire finished slots (reverse-iterate to keep indices valid).
        for (idx, outcome) in retired.into_iter().rev() {
            let mut entry = active.swap_remove(idx);
            {
                let sess = session.lock().unwrap();
                sess.slot_release_blocks(&mut entry.slot.owned_blocks);
            }
            match outcome {
                Ok(()) => {
                    let result = std::mem::take(&mut entry.slot.generated);
                    if let Some(tx) = &entry.slot.stream_tx {
                        let _ = tx.send(StreamEvent::Done { generated: result.clone() });
                    }
                    let _ = entry.done.send(Ok(result));
                }
                Err(e) => {
                    if let Some(tx) = &entry.slot.stream_tx {
                        let _ = tx.send(StreamEvent::Error(e.clone()));
                    }
                    let _ = entry.done.send(Err(e));
                }
            }
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
        params, stop_strings, done, stream_tx,
    } = req;

    // Admission errors must reach BOTH the `done` one-shot (legacy
    // blocking callers) and the stream channel (SSE callers), since a
    // streaming caller only drains `stream_rx`.
    let fail = |msg: String, stream_tx: &Option<mpsc::Sender<StreamEvent>>,
                done: &mpsc::Sender<Result<Vec<usize>, String>>| {
        let _ = done.send(Err(msg.clone()));
        if let Some(tx) = stream_tx { let _ = tx.send(StreamEvent::Error(msg)); }
    };

    if prompt_ids.is_empty() {
        fail("prompt_ids empty".into(), &stream_tx, &done);
        return Err(());
    }
    let vocab = session.lock().unwrap().vocab();
    if let Some(&bad) = prompt_ids.iter().find(|&&i| i >= vocab) {
        fail(format!(
            "prompt_ids contains token id {} out of vocab (vocab_size={})",
            bad, vocab), &stream_tx, &done);
        return Err(());
    }
    if max_tokens == 0 {
        let _ = done.send(Ok(Vec::new()));
        if let Some(tx) = &stream_tx { let _ = tx.send(StreamEvent::Done { generated: Vec::new() }); }
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
    slot.stream_tx = stream_tx;

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
        let msg = format!("prefill failed: {}", e);
        let _ = done.send(Err(msg.clone()));
        if let Some(tx) = &slot.stream_tx { let _ = tx.send(StreamEvent::Error(msg)); }
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
    let logits = {
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
    consume_slot_logits(session, slot, logits)
}

/// FR-19.5-extra-deep Phase 2b-2b — the per-slot post-logits half of a
/// decode tick: logit_bias + repetition penalty, sample, stop-token +
/// stop-string + max-tokens + max-seq checks, and stream/callback emit.
/// Shared verbatim by the serial (`step_slot`) and batched
/// (`step_logits_for_batch`) paths so sampling semantics are identical
/// regardless of how the logits were produced.  Caller must NOT hold the
/// session lock (this re-acquires it for `max_pos` / `decode_ids`).
fn consume_slot_logits(
    session: &Arc<Mutex<QwenSession>>,
    slot: &mut SessionSlot,
    mut logits: Vec<f32>,
) -> Result<bool, String> {
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

    // Decode the new piece ONCE if any consumer needs it: stop-string
    // matching, the in-process callback, or the cross-thread stream.
    let needs_piece = !slot.stop_strings.is_empty()
        || slot.stream_callback.is_some()
        || slot.stream_tx.is_some();
    let piece = if needs_piece {
        session.lock().unwrap().decode_ids(&[id])
    } else {
        String::new()
    };

    // Emit the streaming token BEFORE the stop-string retraction below.
    // This matches the legacy single-session SSE path
    // (handle_completion_streaming_t), which also chunks the piece then
    // breaks — the already-streamed text isn't retracted, only the
    // final `done` id list is trimmed.
    if let Some(cb) = slot.stream_callback.as_mut() {
        cb(id, &piece);
    }
    if let Some(tx) = &slot.stream_tx {
        let _ = tx.send(StreamEvent::Token { id, piece: piece.clone() });
    }

    // Stop-strings: append the new piece to running_text, check
    // tail-match, trim the final id list if hit.
    if !slot.stop_strings.is_empty() {
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
