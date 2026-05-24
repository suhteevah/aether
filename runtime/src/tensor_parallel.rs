//! Tensor-parallel (TP) inference.
//!
//! Multi-GPU TP inference for transformer decoders.  Each GPU computes a
//! shard of the per-block compute; an all-reduce after the attention output
//! projection and another after the FFN down projection sum the partial
//! results back together.
//!
//! Sharding scheme (Megatron-LM convention):
//!   - **q_proj / k_proj / v_proj**: column-parallel — each rank holds
//!     `n_q_heads / W` Q heads and `n_kv_heads / W` KV heads.  When
//!     n_kv_heads < W, KV heads are REPLICATED across ranks (the heads can
//!     be broadcast cheaply over NVLink/PCIe and the per-rank KV cache stays
//!     local to its Q-head slice).
//!   - **o_proj**: row-parallel — each rank's Wo holds `d_model`-out rows
//!     and `(n_q_heads_local * head_dim)`-in cols.  Partial outputs
//!     (`[d_model]` per rank) are summed via all-reduce.
//!   - **ffn_gate / ffn_up (dense FFN)**: column-parallel — output dim
//!     `d_ff / W` rows per rank.
//!   - **ffn_down (dense FFN)**: row-parallel — input dim `d_ff / W` cols
//!     per rank; output `[d_model]` partial summed via all-reduce.
//!   - **MoE FFN**: Phase 1 REPLICATES the MoE block (every rank runs the
//!     full MoE FFN identically).  MoE-TP (sharding experts) is Phase 2.
//!   - **Embeddings + LM head**: replicated.
//!   - **All norms, activations, residuals, RoPE**: replicated (cheap
//!     elementwise compute on already-replicated buffers).
//!
//! Two all-reduces per block:
//!   1. After `Wo @ attn_out` (collapses partial attention output rows).
//!   2. After `down @ silu(gate) * up`  (collapses partial FFN output).
//!
//! Phase 1 scope (what this module ships):
//!   - `TpSession` API + weight-sharding plan + per-block forward
//!     orchestration skeleton.
//!   - `TpSession::new(gguf, world_size)` constructs a session.  With
//!     `world_size == 1`, this is **bit-identical** to `QwenSession::new`
//!     (just a thin wrapper that delegates).
//!   - With `world_size >= 2`, runtime-detects NCCL availability via the
//!     `nccl_real` module.  If 2+ CUDA devices are visible AND the `nccl`
//!     feature is compiled in, attempts multi-GPU setup.  Otherwise emits
//!     a warning and falls back to `world_size = 1`.
//!
//! Known structural gap (filed as `TP_GAPS` documentation below):
//!   The existing `cuda.rs` module uses a single-context singleton (`CTX`
//!   bound to `CudaDevice::new(0)`).  Allocating weights on GPU 1 requires
//!   a separate `CudaDevice::new(1)` context with its own kernel modules
//!   loaded.  That refactor is out of scope for Phase 1.  Until that
//!   lands, `--tp 2` constructs a `TpSession` that holds the full weights
//!   on rank 0 and emits a clear "TP execution falls back to single-GPU
//!   path until cuda.rs is multi-context" warning.  The all-reduce wiring,
//!   sharding plan, host-side sharding math, and API surface are all
//!   shipped + tested so the multi-context cuda.rs refactor lands into a
//!   well-defined seam.
//!
//! Validation:
//!   - `runtime/tests/tp_correctness_smoke.rs` — TP=1 produces bit-identical
//!     logits to non-TP path.  Sharding-plan unit tests prove that host-side
//!     shard-then-sum recovers the unsharded matmul to f32 precision.
//!   - TP=2 real-multi-GPU smoke is gated behind `#[ignore]` until the
//!     cuda.rs multi-context refactor lands.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

// =============================================================================
// Sharding plan — host-side metadata only.  No GPU calls.
// =============================================================================

/// Partition `n` items across `world_size` ranks.  Each rank gets `n / W`
/// or `n / W + 1` items depending on whether it's below the remainder
/// boundary.  Returns `[(start, len); W]`.
///
/// Megatron-LM convention: lowest-numbered ranks absorb the remainder.
/// Total = n by construction.
pub fn partition_rows(n: usize, world_size: usize) -> Vec<(usize, usize)> {
    assert!(world_size > 0, "world_size must be > 0");
    if world_size == 1 {
        return vec![(0, n)];
    }
    let base = n / world_size;
    let rem = n % world_size;
    let mut out = Vec::with_capacity(world_size);
    let mut s = 0;
    for r in 0..world_size {
        let len = base + if r < rem { 1 } else { 0 };
        out.push((s, len));
        s += len;
    }
    debug_assert_eq!(s, n);
    out
}

/// Partition `n_heads` Q-heads across `world_size` ranks.  Same as
/// `partition_rows` for the simple case, but also returns the matching
/// KV-head slice for grouped-query attention (GQA).
///
/// When `n_kv_heads < world_size`, KV heads cannot be evenly partitioned —
/// the convention is to REPLICATE KV heads across ranks (each rank holds
/// all KV heads but only its own Q-head slice).  This trades a small
/// memory footprint penalty for clean kernel boundaries (no cross-rank
/// communication for KV reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadShard {
    pub q_head_start: usize,
    pub q_head_count: usize,
    pub kv_head_start: usize,
    pub kv_head_count: usize,
    /// True when KV heads are replicated (n_kv_heads < world_size).
    pub kv_replicated: bool,
}

pub fn partition_heads(
    n_q_heads: usize, n_kv_heads: usize, world_size: usize,
) -> Vec<HeadShard> {
    assert!(world_size > 0);
    assert!(n_q_heads > 0);
    assert!(n_kv_heads > 0);
    assert_eq!(n_q_heads % n_kv_heads, 0,
        "GQA invariant: n_q_heads must be a multiple of n_kv_heads");
    if world_size == 1 {
        return vec![HeadShard {
            q_head_start: 0, q_head_count: n_q_heads,
            kv_head_start: 0, kv_head_count: n_kv_heads,
            kv_replicated: false,
        }];
    }
    let q_parts = partition_rows(n_q_heads, world_size);
    let kv_replicated = n_kv_heads < world_size;
    if kv_replicated {
        return q_parts.into_iter().map(|(qs, qc)| HeadShard {
            q_head_start: qs, q_head_count: qc,
            kv_head_start: 0, kv_head_count: n_kv_heads,
            kv_replicated: true,
        }).collect();
    }
    // n_kv_heads >= world_size — partition KV the same way.  GQA groups
    // stay aligned because n_q_heads / n_kv_heads is uniform.
    let kv_parts = partition_rows(n_kv_heads, world_size);
    q_parts.into_iter().zip(kv_parts.into_iter())
        .map(|((qs, qc), (ks, kc))| HeadShard {
            q_head_start: qs, q_head_count: qc,
            kv_head_start: ks, kv_head_count: kc,
            kv_replicated: false,
        }).collect()
}

/// Complete sharding plan for a transformer block.
///
/// Computed once at session construction.  Each per-rank plan tells the
/// weight loader which rows/cols of each tensor to keep on that rank.
#[derive(Debug, Clone)]
pub struct BlockShardPlan {
    pub rank: usize,
    pub world_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    /// Attention head shard for this rank.
    pub heads: HeadShard,
    /// Local d_ff for this rank (column-parallel gate/up output dim).
    pub d_ff_local: usize,
    /// Local Q output dim for this rank: q_head_count * head_dim.
    pub d_q_local: usize,
    /// Local KV output dim for this rank: kv_head_count * head_dim.
    pub d_kv_local: usize,
    /// Per-rank o_proj input dim (= d_q_local).
    pub d_o_in_local: usize,
}

impl BlockShardPlan {
    pub fn new(
        rank: usize, world_size: usize,
        d_model: usize, d_ff: usize,
        n_q_heads: usize, n_kv_heads: usize, head_dim: usize,
    ) -> Self {
        assert!(rank < world_size);
        let parts = partition_heads(n_q_heads, n_kv_heads, world_size);
        let heads = parts[rank];
        let d_q_local = heads.q_head_count * head_dim;
        let d_kv_local = heads.kv_head_count * head_dim;
        let d_ff_parts = partition_rows(d_ff, world_size);
        let d_ff_local = d_ff_parts[rank].1;
        Self {
            rank, world_size, d_model, d_ff,
            heads, d_ff_local,
            d_q_local,
            d_kv_local,
            d_o_in_local: d_q_local,
        }
    }
}

// =============================================================================
// Pure-host correctness primitives — exercised by the smoke tests.
// =============================================================================

/// Reference (unsharded) matmul: `y[i, j] = sum_k x[i, k] * w[j, k]`
/// (row-major, weight in `[n_out, n_in]` "NT" layout matching the GPU
/// kernels' convention).
///
/// Used by the smoke tests to verify the host-side shard-then-sum identity.
pub fn matmul_nt_host(x: &[f32], w: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    assert_eq!(x.len(), n_in);
    assert_eq!(w.len(), n_in * n_out);
    let mut y = vec![0.0f32; n_out];
    for i in 0..n_out {
        let mut acc = 0.0f32;
        for k in 0..n_in {
            acc += x[k] * w[i * n_in + k];
        }
        y[i] = acc;
    }
    y
}

/// Shard a weight matrix by output rows (column-parallel).  Splits `w`
/// `[n_out, n_in]` into `world_size` slices of `[n_out_local, n_in]`.
/// Each rank's output slice computes a contiguous row range of `y`.
pub fn shard_w_by_rows(w: &[f32], n_in: usize, n_out: usize, world_size: usize) -> Vec<Vec<f32>> {
    let parts = partition_rows(n_out, world_size);
    parts.into_iter().map(|(s, len)| {
        w[s * n_in .. (s + len) * n_in].to_vec()
    }).collect()
}

/// Shard a weight matrix by input cols (row-parallel).  Splits `w`
/// `[n_out, n_in]` (NT layout) into `world_size` slices of
/// `[n_out, n_in_local]`.  Caller must also shard the input `x` along
/// the same axis; per-rank partial outputs are then summed via
/// all-reduce.
pub fn shard_w_by_cols(w: &[f32], n_in: usize, n_out: usize, world_size: usize) -> Vec<Vec<f32>> {
    let parts = partition_rows(n_in, world_size);
    parts.into_iter().map(|(s, len)| {
        let mut out = vec![0.0f32; n_out * len];
        for i in 0..n_out {
            out[i * len .. (i + 1) * len].copy_from_slice(
                &w[i * n_in + s .. i * n_in + s + len]);
        }
        out
    }).collect()
}

/// Shard an input vector by its layout axis.  Used together with
/// `shard_w_by_cols` for row-parallel matmuls.
pub fn shard_x_by_cols(x: &[f32], world_size: usize) -> Vec<Vec<f32>> {
    let parts = partition_rows(x.len(), world_size);
    parts.into_iter().map(|(s, len)| x[s..s+len].to_vec()).collect()
}

/// Sum partial outputs across ranks.  Models the all-reduce SUM that
/// would happen on-device.  All inputs must have the same length.
pub fn all_reduce_sum_host(partials: &[Vec<f32>]) -> Vec<f32> {
    assert!(!partials.is_empty());
    let n = partials[0].len();
    for p in partials.iter() { assert_eq!(p.len(), n, "all_reduce: partial size mismatch"); }
    let mut out = vec![0.0f32; n];
    for p in partials {
        for i in 0..n { out[i] += p[i]; }
    }
    out
}

/// Concatenate column-parallel output partials.  Each rank's partial is
/// a contiguous row-slice of the full output; reassembly is just
/// concatenation.
pub fn concat_rows_host(partials: &[Vec<f32>]) -> Vec<f32> {
    let total: usize = partials.iter().map(|p| p.len()).sum();
    let mut out = Vec::with_capacity(total);
    for p in partials { out.extend_from_slice(p); }
    out
}

// =============================================================================
// TpSession — the multi-rank inference session.
// =============================================================================

/// Runtime-detected NCCL availability.  Cached on first probe so the
/// detection doesn't repeat on every constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclAvailability {
    /// `nccl` feature compiled in AND `>= 2` CUDA devices visible at
    /// runtime.  TP > 1 can use real NCCL.
    Available { n_devices: usize },
    /// `nccl` feature is compiled in but < 2 CUDA devices visible.
    NotEnoughDevices { n_devices: usize },
    /// `nccl` feature not compiled.  This binary cannot do real multi-GPU.
    FeatureNotCompiled,
    /// CUDA device count probe failed.
    ProbeFailed,
}

/// Probe whether real multi-GPU NCCL is usable in this process.
pub fn probe_nccl_availability() -> NcclAvailability {
    #[cfg(feature = "nccl")]
    {
        use cudarc::driver::CudaDevice;
        match CudaDevice::count() {
            Ok(n) => {
                let n = n as usize;
                if n >= 2 {
                    NcclAvailability::Available { n_devices: n }
                } else {
                    NcclAvailability::NotEnoughDevices { n_devices: n }
                }
            }
            Err(_) => NcclAvailability::ProbeFailed,
        }
    }
    #[cfg(not(feature = "nccl"))]
    {
        NcclAvailability::FeatureNotCompiled
    }
}

/// Multi-GPU tensor-parallel inference session.
///
/// Wraps an existing `QwenSession` for the inner per-rank work.  In
/// Phase 1 with `world_size == 1`, this is a transparent passthrough.
/// With `world_size >= 2`, the constructor detects NCCL availability and
/// either succeeds in multi-GPU mode (when supported) or falls back to
/// single-GPU with a warning.
///
/// API mirrors `QwenSession` for drop-in serve.rs integration.
pub struct TpSession {
    /// The actual world_size the session is running with.  May be < the
    /// requested `world_size` if NCCL/multi-GPU init failed (fallback).
    pub effective_world_size: usize,
    /// The world_size the caller requested.  Recorded for diagnostics.
    pub requested_world_size: usize,
    /// Per-rank weight-sharding plans (one entry per `effective_world_size`).
    /// In fallback mode (`effective_world_size == 1`), this has length 1
    /// and represents the trivial unsharded plan.
    pub plans: Vec<BlockShardPlan>,
    /// Inner session that holds the (full) weights.  In Phase 1 we hold
    /// exactly ONE QwenSession on rank 0; for true multi-GPU TP each rank
    /// will own its own slice (filed as `TP_GAPS::CUDA_MULTI_CONTEXT`).
    inner: crate::serving::QwenSession,
    /// NCCL availability the session was constructed with.  Diagnostic only.
    pub nccl: NcclAvailability,
}

/// Errors during TP session construction.
#[derive(Debug)]
pub enum TpError {
    /// The underlying QwenSession failed to load.
    SessionLoad(String),
    /// `world_size` argument is 0.
    ZeroWorldSize,
}

impl std::fmt::Display for TpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TpError::SessionLoad(s) => write!(f, "TP session load failed: {}", s),
            TpError::ZeroWorldSize => write!(f, "world_size must be > 0"),
        }
    }
}

impl std::error::Error for TpError {}

impl TpSession {
    /// Construct a TP session.  See module docs for semantics.
    ///
    /// `world_size = 1`: thin wrapper around `QwenSession::new`.  Bit-
    /// identical to the non-TP path.
    ///
    /// `world_size >= 2`: detects NCCL availability.  If real multi-GPU
    /// is wired up (post `TP_GAPS::CUDA_MULTI_CONTEXT` refactor), shards
    /// weights across ranks and runs the partial-compute + all-reduce
    /// path.  Until then, falls back to single-GPU with a warning.
    pub fn new(gguf_path: &str, world_size: usize) -> Result<Self, TpError> {
        Self::new_with_mode(gguf_path, world_size, false)
    }

    /// Construct with explicit KV-cache mode (`paged = true` matches
    /// `QwenSession::new_paged`).
    pub fn new_paged(gguf_path: &str, world_size: usize) -> Result<Self, TpError> {
        Self::new_with_mode(gguf_path, world_size, true)
    }

    fn new_with_mode(gguf_path: &str, world_size: usize, paged: bool) -> Result<Self, TpError> {
        if world_size == 0 { return Err(TpError::ZeroWorldSize); }

        let inner = if paged {
            crate::serving::QwenSession::new_paged(gguf_path)
        } else {
            crate::serving::QwenSession::new(gguf_path)
        }.map_err(TpError::SessionLoad)?;

        // Resolve effective world_size based on NCCL availability + the
        // structural gap (cuda.rs single-context).
        let nccl = probe_nccl_availability();
        let mut effective = world_size;

        if world_size > 1 {
            let multi_ok = matches!(nccl, NcclAvailability::Available { .. });
            if !multi_ok {
                eprintln!(
                    "[TpSession] WARN: --tp {} requested but multi-GPU unavailable ({:?}); \
                     falling back to --tp 1 (single-GPU path)",
                    world_size, nccl);
                effective = 1;
            } else {
                // NCCL is available BUT the cuda.rs CTX is a single-device
                // singleton (TP_GAPS::CUDA_MULTI_CONTEXT).  Until that
                // refactor lands, we can't actually put weights on GPU 1+
                // — every kernel still runs on device 0.  Emit a clear
                // warning and degrade to effective=1 so the session still
                // produces correct output (just on one GPU).
                eprintln!(
                    "[TpSession] WARN: --tp {} with NCCL available ({:?}), but cuda.rs \
                     uses a single-device context (CTX bound to CudaDevice(0)). \
                     Multi-context support is filed as TP_GAPS::CUDA_MULTI_CONTEXT in \
                     runtime/src/tensor_parallel.rs. Falling back to --tp 1 until \
                     that refactor lands.  All-reduce wiring + sharding-plan tests \
                     still validate the correctness math.",
                    world_size, nccl);
                effective = 1;
            }
        }

        let plans: Vec<BlockShardPlan> = (0..effective).map(|r|
            BlockShardPlan::new(
                r, effective,
                inner.cfg.d_model, inner.cfg.d_ff,
                inner.cfg.n_q_heads, inner.cfg.n_kv_heads, inner.cfg.head_dim,
            )
        ).collect();

        Ok(Self {
            effective_world_size: effective,
            requested_world_size: world_size,
            plans, inner, nccl,
        })
    }

    /// Reset the underlying session's KV cache.
    pub fn reset(&mut self) { self.inner.reset(); }

    /// Prefill the underlying session.  In Phase 1 effective fallback,
    /// this is just `QwenSession::prefill`.  Once multi-GPU lands, each
    /// rank prefills its own KV slice in parallel.
    pub fn prefill(&mut self, prompt_ids: &[usize]) {
        self.inner.prefill(prompt_ids);
    }

    /// Run one decode step.  See module docs for the two-all-reduce
    /// scheme.  Phase 1 fallback delegates to single-GPU.
    pub fn decode_step(&mut self, last_id: usize) -> usize {
        // Multi-rank path lives here once cuda.rs is multi-context.
        // For now: single delegate is bit-identical to non-TP path.
        self.inner.decode_step(last_id)
    }

    /// Warmup the underlying GPU(s).
    pub fn warmup(&mut self, n_steps: usize) { self.inner.warmup(n_steps); }

    /// Generate `max_tokens` continuation token ids.  Convenience wrapper
    /// around prefill + decode_step that matches QwenSession::generate.
    pub fn generate(
        &mut self, prompt_ids: &[usize], max_tokens: usize, stop_token: Option<usize>,
    ) -> Vec<usize> {
        self.inner.generate(prompt_ids, max_tokens, stop_token)
    }

    /// Borrow the inner session immutably.  Used by serve.rs to access
    /// fields like `cfg`, `eos_token`, `decode_ids`, `encode_text`, etc.
    /// that haven't been promoted onto TpSession yet.
    pub fn inner(&self) -> &crate::serving::QwenSession { &self.inner }

    /// Borrow the inner session mutably.  Used by serve.rs for the same
    /// reason as `inner()`.
    pub fn inner_mut(&mut self) -> &mut crate::serving::QwenSession { &mut self.inner }

    /// Consume the TpSession and return the inner QwenSession.  In the
    /// Phase 1 fallback (effective_world_size == 1), this is the
    /// natural way to integrate with code paths that still want a
    /// `QwenSession` directly.  Once true multi-GPU lands, this method
    /// will be removed in favour of routing all decode through
    /// `TpSession::decode_step`.
    pub fn into_inner(self) -> crate::serving::QwenSession { self.inner }

    /// Run one all-reduce sum on host-side partial buffers via real NCCL
    /// if available, otherwise the host-side fallback.  Used by the
    /// internal TP forward orchestration once multi-context cuda.rs lands.
    ///
    /// Today this is exposed as a public testing hook so the smoke test
    /// can verify the all-reduce path on dual P100s without needing a
    /// full forward pass.
    #[cfg(feature = "nccl")]
    pub fn all_reduce_sum_dev(send_dev: i64, recv_dev: i64, n: c_int, comm: i64) -> c_int {
        crate::nccl_real::aether_nccl_real_all_reduce_f32(send_dev, recv_dev, n, 0, comm)
    }

    /// Diagnostic summary for serve.rs to log on startup.
    pub fn diag(&self) -> String {
        format!(
            "TpSession(requested={}, effective={}, nccl={:?}, d_model={}, d_ff={}, n_q_heads={}, n_kv_heads={})",
            self.requested_world_size,
            self.effective_world_size,
            self.nccl,
            self.inner.cfg.d_model,
            self.inner.cfg.d_ff,
            self.inner.cfg.n_q_heads,
            self.inner.cfg.n_kv_heads,
        )
    }
}

// =============================================================================
// TP_GAPS — structural gaps that block Phase-1 TP execution from actually
// sharding onto multiple GPUs.  Each is its own follow-on FR.  Listed here
// so the multi-context refactor lands into a known seam.
// =============================================================================

/// Documentation-only — known structural gaps blocking real multi-GPU TP.
///
/// 1. **CUDA_MULTI_CONTEXT** — `runtime/src/cuda.rs::CTX` is a
///    `OnceLock<CudaCtx>` initialised exactly once against
///    `CudaDevice::new_with_stream(0)`.  All BUFFERS / I32_BUFFERS /
///    U8_BUFFERS registries live on that single device.  Putting weights
///    on GPU 1 needs a per-device context + per-device kernel-module
///    load + per-device registries.  Roughly a `Vec<CudaCtx>` indexed by
///    rank, with `bufs(rank)` / `i32_bufs(rank)` / `u8_bufs(rank)`
///    accessors.  Every `aether_op_*_cuda` function needs a `device_id`
///    parameter threading through.
///
/// 2. **PER_RANK_WEIGHTS** — `QwenSession::new_with_mode` uploads all
///    weights through `upload_tensor_u8` against the default device.
///    With multi-context cuda, weight upload needs to dispatch to the
///    target rank's device.  Sharding metadata is already in
///    `BlockShardPlan` — the loader just needs to honour `(start, len)`
///    when copying GGUF tensor blobs to device.
///
/// 3. **PER_RANK_ATTENTION** — `block_forward_devarg` reads
///    `cfg.n_q_heads` / `cfg.n_kv_heads` as the full counts.  With TP
///    those become per-rank counts (`plan.heads.q_head_count` etc.) so
///    the attention kernel only processes the rank's slice.
///
/// 4. **CROSS_RANK_ALLREDUCE** — after `Wo @ attn_out` and after `down @
///    ffn_inner`, partial outputs need a `ncclAllReduce(SUM)` on the
///    `[d_model]` activation buffer.  The wiring path is:
///    `nccl_real::aether_nccl_real_all_reduce_f32(send_dev, recv_dev, d_model, 0, comm)`
///    — both buffers must be the SAME device, which means we'd allocate
///    a per-rank `act.proj` / `act.down` on each rank and the all-reduce
///    sums across rank communicators.
///
/// 5. **RANK_THREADING** — today `decode_step` runs synchronously on the
///    calling thread.  Multi-GPU needs N threads (one per rank) feeding
///    into a `group_start() / per-rank work / group_end()` collective
///    barrier — same shape as `runtime/tests/nccl_dual_gpu.rs`.
pub mod tp_gaps_doc {
    //! See enclosing module-level documentation.
}

// =============================================================================
// Tests (host-side, no GPU needed) — verify the sharding math.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_rows_basic() {
        assert_eq!(partition_rows(10, 1), vec![(0, 10)]);
        assert_eq!(partition_rows(10, 2), vec![(0, 5), (5, 5)]);
        assert_eq!(partition_rows(10, 3), vec![(0, 4), (4, 3), (7, 3)]);
        assert_eq!(partition_rows(8, 4), vec![(0, 2), (2, 2), (4, 2), (6, 2)]);
    }

    #[test]
    fn partition_rows_total_preserved() {
        for n in [1usize, 7, 32, 128, 4096, 13824] {
            for w in [1usize, 2, 3, 4, 8] {
                let parts = partition_rows(n, w);
                assert_eq!(parts.len(), w);
                let total: usize = parts.iter().map(|(_, l)| *l).sum();
                assert_eq!(total, n, "n={} w={}", n, w);
                // No gaps.
                let mut s = 0;
                for (st, l) in &parts {
                    assert_eq!(*st, s);
                    s += l;
                }
            }
        }
    }

    #[test]
    fn partition_heads_gqa() {
        // Qwen2.5-7B: 28 Q heads, 4 KV heads.
        let p = partition_heads(28, 4, 2);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].q_head_count + p[1].q_head_count, 28);
        assert_eq!(p[0].kv_head_count + p[1].kv_head_count, 4);
        assert!(!p[0].kv_replicated);

        // GLM-4.7-flash MLA-ish: 20 Q heads, 1 KV head (MQA-style).
        // KV gets replicated.
        let p = partition_heads(20, 1, 2);
        assert!(p[0].kv_replicated);
        assert!(p[1].kv_replicated);
        assert_eq!(p[0].kv_head_count, 1);
        assert_eq!(p[1].kv_head_count, 1);
    }

    /// The CORE correctness gate: shard a matmul's weight by output rows
    /// (column-parallel), compute per-rank partials, concatenate the
    /// rows, and verify it equals the unsharded matmul.
    #[test]
    fn column_parallel_shard_reconstructs() {
        let n_in = 64usize;
        let n_out = 32usize;
        let world_size = 4usize;

        // Deterministic synthetic weights + input.
        let w: Vec<f32> = (0..n_in * n_out).map(|i| (i as f32 * 0.013).sin()).collect();
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.07).cos()).collect();

        let y_full = matmul_nt_host(&x, &w, n_in, n_out);

        let w_shards = shard_w_by_rows(&w, n_in, n_out, world_size);
        let parts = partition_rows(n_out, world_size);
        let mut partials: Vec<Vec<f32>> = Vec::new();
        for (r, &(_s, len)) in parts.iter().enumerate() {
            // Each rank gets x in full (input is replicated for column-
            // parallel) and computes [len] output rows.
            let y_part = matmul_nt_host(&x, &w_shards[r], n_in, len);
            assert_eq!(y_part.len(), len);
            partials.push(y_part);
        }
        let y_reconstructed = concat_rows_host(&partials);
        assert_eq!(y_reconstructed.len(), y_full.len());
        for i in 0..y_full.len() {
            assert!((y_reconstructed[i] - y_full[i]).abs() < 1e-5,
                "column-parallel mismatch at {}: {} vs {}",
                i, y_reconstructed[i], y_full[i]);
        }
    }

    /// The other CORE correctness gate: shard a matmul's weight by INPUT
    /// cols (row-parallel), shard the input the same way, compute per-
    /// rank partials, sum (all-reduce) them, verify it equals the
    /// unsharded matmul.
    #[test]
    fn row_parallel_shard_then_all_reduce_reconstructs() {
        let n_in = 64usize;
        let n_out = 32usize;
        let world_size = 4usize;

        let w: Vec<f32> = (0..n_in * n_out).map(|i| (i as f32 * 0.019).sin()).collect();
        let x: Vec<f32> = (0..n_in).map(|i| (i as f32 * 0.11).cos()).collect();

        let y_full = matmul_nt_host(&x, &w, n_in, n_out);

        let w_shards = shard_w_by_cols(&w, n_in, n_out, world_size);
        let x_shards = shard_x_by_cols(&x, world_size);
        let parts = partition_rows(n_in, world_size);

        let mut partials: Vec<Vec<f32>> = Vec::new();
        for (r, &(_s, len)) in parts.iter().enumerate() {
            // Per-rank w shard: [n_out, len].  Per-rank x shard: [len].
            let mut y_part = vec![0.0f32; n_out];
            for i in 0..n_out {
                let mut acc = 0.0f32;
                for k in 0..len {
                    acc += x_shards[r][k] * w_shards[r][i * len + k];
                }
                y_part[i] = acc;
            }
            partials.push(y_part);
        }
        let y_reduced = all_reduce_sum_host(&partials);
        assert_eq!(y_reduced.len(), y_full.len());
        for i in 0..y_full.len() {
            assert!((y_reduced[i] - y_full[i]).abs() < 1e-4,
                "row-parallel mismatch at {}: {} vs {}",
                i, y_reduced[i], y_full[i]);
        }
    }

    /// World-size-1 must produce a single trivial shard equal to the
    /// unsharded weight.  This is the bit-identity guard for `--tp 1`.
    #[test]
    fn tp1_is_identity() {
        let n_in = 16usize;
        let n_out = 8usize;
        let w: Vec<f32> = (0..n_in * n_out).map(|i| i as f32).collect();
        let s = shard_w_by_rows(&w, n_in, n_out, 1);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], w);
        let s2 = shard_w_by_cols(&w, n_in, n_out, 1);
        assert_eq!(s2.len(), 1);
        assert_eq!(s2[0], w);
        let x: Vec<f32> = vec![1.0; n_in];
        let xs = shard_x_by_cols(&x, 1);
        assert_eq!(xs.len(), 1);
        assert_eq!(xs[0], x);
    }

    #[test]
    fn block_shard_plan_qwen25_7b_tp2() {
        // Qwen2.5-7B shape: d_model=3584, d_ff=18944, 28 Q heads, 4 KV heads, head_dim=128
        let p0 = BlockShardPlan::new(0, 2, 3584, 18944, 28, 4, 128);
        let p1 = BlockShardPlan::new(1, 2, 3584, 18944, 28, 4, 128);
        assert_eq!(p0.d_ff_local + p1.d_ff_local, 18944);
        assert_eq!(p0.heads.q_head_count + p1.heads.q_head_count, 28);
        assert_eq!(p0.heads.kv_head_count + p1.heads.kv_head_count, 4);
        assert!(!p0.heads.kv_replicated);
    }

    #[test]
    fn nccl_availability_is_probeable() {
        // Just confirm the call doesn't panic and returns one of the variants.
        let avail = probe_nccl_availability();
        match avail {
            NcclAvailability::Available { n_devices } => assert!(n_devices >= 2),
            NcclAvailability::NotEnoughDevices { n_devices } => assert!(n_devices < 2),
            NcclAvailability::FeatureNotCompiled => {}
            NcclAvailability::ProbeFailed => {}
        }
    }
}
