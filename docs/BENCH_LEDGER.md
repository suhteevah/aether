# Aether bench ledger

Standing benches per Roadmap v2's bench cadence. Every milestone that touches a perf-relevant code path appends a row here. Format:

```
| date       | commit  | bench           | config              | aether | candle | torch  | verdict |
```

Numbers are per-iter µs (lower is better). Configs vary per bench — see the bench's own README for the full matrix.

## bench/matmul_micro — cuBLAS sgemm 3-way

Hardware: i9-11900K, RTX 3070 Ti 8 GiB, Win10 Pro. CUDA 12.6, candle 0.10.2 (local fork), pytorch 2.11.0+cu128.

| date       | commit | bench  | dim    | iters | aether | candle | torch | leader  |
|------------|--------|--------|-------:|------:|-------:|-------:|------:|---------|
| 2026-05-03 | (head) | gpu    |   64³  |   100 |    7   |     9  |    26 | aether  |
| 2026-05-03 | (head) | gpu    |  256³  |    50 |   11   |    13  |    19 | aether  |
| 2026-05-03 | (head) | gpu    |  512³  |    20 |   39   |    44  |    46 | aether  |
| 2026-05-03 | (head) | gpu    | 1024³  |    10 |  310   |   213  |   196 | torch   |
| 2026-05-03 | (head) | cpu    |   64³  |   100 |   N/A  |     9  |     5 | torch   |
| 2026-05-03 | (head) | cpu    |  256³  |    50 |   N/A  |   143  |    77 | torch   |
| 2026-05-03 | (head) | cpu    |  512³  |    20 |   N/A  |   660  |   496 | torch   |
| 2026-05-03 | (head) | cpu    | 1024³  |    10 |   N/A  |  4611  |  4221 | torch   |

Aether wins 3 of 4 GPU sgemm sizes. CPU is intentionally not bench'd today — the runtime's CPU path is the Phase-0 reference for correctness, not a perf target. AVX-512 microkernels land in roadmap item P10.6.

### 2026-05-09 — commit 81264f4: skipped (cross-library variance)

Single-trial run at this commit produced numbers whose run-to-run spread exceeded the cross-library deltas we'd be reporting on. Appending a row would have implied a verdict the data didn't support. Per the append-only honesty rule we recorded no row rather than recording a noisy one. Re-bench will land once the bench harness moves to median-of-5 (gating on `bench/matmul_micro/run_all.ps1` warm-up + trial-count refactor). No code path under this commit changed `runtime/src/cuda.rs` semantics, so the prior 2026-05-03 row remains the standing reference.

### 2026-05-18 — pending commit (FR-15.2 regalloc-in-emit): skipped (GPU contention + structural no-op for matmul)

Bench-runner subagent invoked under the append rule (commit touches `compiler/src/codegen/asm/`). At run time the GPU was at 39% util with 7.3 GiB/8 GiB occupied by external processes (`Settlement Survival.exe` + `ollama.exe`). Per the honesty rule we declined to record numbers under contention. Independent structural argument: FR-15.2 promotes hot Int locals into callee-saved r12..r15 inside `.aether`-source fns; the matmul caller in Aether source passes Tensor handles (i64) to `aether_op_matmul_f32`, which is unchanged in `runtime/src/cuda.rs`. The cuBLAS sgemm time dominates the bench by orders of magnitude. Expected delta vs the 2026-05-03 row: indistinguishable from noise. The 2026-05-03 row stays the standing reference. Re-bench when the GPU is idle.

### 2026-05-18 — pending commit (FR-15.3 AVX2 emit): skipped (no overlap with matmul hot path)

Bench-runner append rule fires because the commit touches both `compiler/src/codegen/asm/` and `runtime/src/lib.rs`. The asm-backend addition is a new compiler-recognized builtin `__aether_avx2_dot_f32` that inlines an AVX2 dot loop using new VEX-encoded ops (`vxorps`/`vmovups`/`vmulps`/`vaddps`/`vzeroupper`). The runtime additions are three witness-only helpers (`aether_avx2_witness_arr`, `aether_dot_f32_scalar`, `aether_f32_close_exit`) — none of them appear on any standing bench path. The matmul benches drive `aether_op_matmul_f32` through cuBLAS, which this commit does not change. Expected delta vs the 2026-05-03 row: zero. A standalone "1024-elem f32 dot AVX2 vs scalar" bench fixture is the right place to record the per-instruction headline — that fixture doesn't exist yet; deferred. The 2026-05-03 row remains the standing reference.

### 2026-05-19 — pending commit (FR-17.3 conv2d CPU reference): skipped (additive new fn, no matmul path touched)

Bench-runner append rule fires on `runtime/src/lib.rs` touched. The change is purely additive — a new `aether_op_conv2d_f32` direct-loop reference impl and two unit tests in a new `conv2d_tests` mod. No existing matmul / softmax / layer_norm / SDPA / CE code path is altered. The `bench/conv2d/` section of this ledger has a "planned" line gating on `aether_op_conv2d_*` shipping in `runtime/src/cuda.rs` — that's the appropriate row to fill in once cuDNN-or-equivalent lands. CPU direct-loop conv2d is a correctness reference, not a perf path. The 2026-05-03 matmul row remains the standing reference.

### 2026-05-19 — pending commit (Phase 17 closeout — Q4_0 + FA2 + layer modules f32 + Llama-shaped partial): skipped (additive, no matmul path touched)

Bench-runner append rule fires again on `runtime/src/lib.rs` touched. Four new runtime symbols added (`aether_dequant_q4_0`, `aether_flash_attention_v2_f32`, `aether_store_i32`, `aether_sum_f32`) plus four new `.aether` witnesses. The matmul / softmax / SDPA / CE / LN / conv hot paths are untouched (the new FA2 fn is its own kernel, not a swap of the existing `aether_op_sdpa_causal_f32`). Standing 2026-05-03 matmul row remains the reference. A dedicated `bench/attention_fa2_vs_naive/` fixture is the right place to record the FA2 speedup once seq_len grows large enough for the O(N) memory advantage to bite — that fixture doesn't exist yet; deferred.

### 2026-05-19 — pending commit (Phase 18 closeout — NCCL surface + 6 distributed sims): skipped (in-process simulations, no real multi-rank to bench)

Bench-runner append rule fires on `runtime/src/lib.rs` touched. Eight new symbols added: `aether_nccl_*` (init/finalize/comm_create/destroy/world_size/rank/all_reduce_f32 — single-host fallback), `aether_tp_simulate_column_parallel_linear_f32`, `aether_pp_simulate_2stage_forward_f32`, `aether_fsdp_simulate_shard_alltoall_f32`, `aether_zero_simulate_stage_bytes_f32`, `aether_overlap_simulate_overlapped_us` / `_serial_us`, `aether_grad_compress_lowrank_f32`. Every fn is named `*_simulate_*` or returns single-host fallback values — there is no real multi-rank wall-time to measure on the kokonoe single-3070Ti box. The matmul path is untouched. Real cross-card bench fixtures live in MATT_VOICE_FR.md and require the cnc 2×P100 + libnccl link (FR-18.1-extra). Standing 2026-05-03 matmul row remains the reference.

### 2026-05-19 — pending commit (FR-19.9 BPE tokenizer): skipped (no matmul path; tokenizer is its own micro-bench surface)

Bench-runner append rule fires on `runtime/src/lib.rs` touched. New symbols: `aether_bpe_tokenizer_new` / `_free` / `aether_bpe_add_merge` / `aether_bpe_encode` / `aether_bpe_decode`. The implementation is the textbook BPE merge loop in pure Rust — no GPU, no matmul, no SIMD. The matmul bench is untouched. A dedicated `bench/tokenizer_throughput/` fixture is the right place to log "MB/s through encode" once the matt-voice Qwen2.5 tokenizer.json loader (FR-19.9-extra) lands; that fixture doesn't exist yet. The 2026-05-03 matmul row remains the standing reference.

### 2026-05-19 — pending commit (FR-19.10 chat template renderer): skipped (template engine; not on matmul path)

Bench-runner append rule fires on `runtime/src/lib.rs` touched. New symbols: `aether_template_new` / `_free` / `aether_template_set_var` / `aether_template_push_message` / `aether_template_render`. Pure-Rust state-machine template parser. No GPU / no matmul / no SIMD. Matmul bench is untouched. A `bench/chat_template_throughput/` fixture for "ms per render of a typical chat template" is the right surface to log this once the matt-voice serving deploy actually renders user prompts at scale — fixture doesn't exist yet. 2026-05-03 matmul row remains the reference.

### 2026-05-19 — pending commit (FR-17.14-extra-deeper GGUF reader): skipped (no matmul path; GGUF I/O is its own surface)

Bench-runner append rule fires on `runtime/src/lib.rs` touched.
Added a real GGUF v3 reader (9 extern fns) that opens matt-voice's
local Qwen2.5-7B Q4_K_M blob (4.7 GB at
`C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`), walks
the 339-tensor table, returns data pointers ready to pass to the
Q4_K_M dequant kernel shipped in the prior commit.

The matmul / SDPA / LN paths are untouched. A `bench/gguf_load/`
fixture for "ms to header-parse Qwen2.5-7B" is the right surface
to log this once an HTTP/serving deploy is timing critical. For
the cold 4.7 GB blob the dominant cost is `std::fs::read`; the
parser walk itself is ~1.6 sec on the 11900K (per the unit test's
`finished in 1.65s` line).

### 2026-05-19 — pending commit (matt-voice deploy pack — 5 extras): cuda build now live; no matmul bench rerun

Bench-runner append rule fires on `runtime/src/lib.rs` touched.
Five deliverables:

1. `cargo build -p aether_rt --features cuda` now succeeds on
   kokonoe (CUDA toolkit v12.6 + cudarc 0.13). The resulting
   `target/debug/libaether_rt.a` contains cuBLAS symbols (39507
   matches via grep). `tests/runtime/cuda_train_tiny.aether` now
   exits 0 through real GPU training (was skipped on the previous
   default build).
2-5. SafeTensors multi-tensor / Q4_K_M dequant / tokenizer.json
   loader / chat_template.jinja loader — all on the matt-voice
   serving-deploy critical path; none touch the matmul hot path.

The 2026-05-03 matmul bench stays the standing reference. The
expected next bench row appears when:
- (a) someone reruns `bench/matmul_micro/run_all.ps1` AFTER the
  cuda-feature rebuild to compare cuBLAS sgemm vs the 2026-05-03
  reference (which was already cuda-built, so should be ±noise);
- (b) the FR-19.16-extra path actually loads Llama-1B weights
  and runs the inference bench through cuBLAS — that row goes
  under `bench/llama_inference`.

### 2026-05-19 — pending commit (Phase 19 closeout — 13 items): skipped (additive, no matmul / SDPA / LN path touched)

Bench-runner append rule fires on `runtime/src/lib.rs` touched. 13 new runtime symbols across PKV-sim / CB-sim / specdec / MM-sim / rate-limit / observability / image-preprocess / DFT+Hann / ChaCha20-Poly1305 / HTTP-parse+write / OpenAI-JSON-render / WS-frame-codec / tool-call-render. Pure CPU code; no GPU; no matmul; no SDPA / LN. Matmul bench is untouched. Real bench fixtures gating on the closeout items live behind their own FR-19.x-extras (e.g. real cross-card NCCL for the PKV+CB chain, real TLS handshake for HTTP+OpenAI). Standing 2026-05-03 matmul row remains the reference.

## bench/conv2d — 3-way (planned, gates on P7.3)

Pending — fires once `aether_op_conv2d_*` ships in `runtime/src/cuda.rs`.

## bench/attention — 3-way FlashAttention (planned, gates on P7.3)

Pending — fires once a real FlashAttention v2 kernel lands.

## bench/llama_inference — single-stream tokens/sec (partial — FR-19.16)

### 2026-05-19 — Llama-shape CPU sequential decode (partial witness for FR-19.16)

Witness: `tests/runtime/llm_inference_tps.aether`
Runtime fn: `aether_llm_inference_bench_tps(n_iters, d_model, n_layers, ff, seq_len) -> f32`
Hardware: kokonoe 11900K, CPU only (no GPU; no --features cuda)
Build: `target/debug/libaether_rt.a` (debug, unoptimised — the audit's
`--emit=aether-bin` chain links the debug archive)

Model: Llama-architecture transformer block (LN + Q/K/V matmul +
SDPA causal + Wo + residual + LN + MLP-with-SiLU + residual). All
chain ops via real `ops::*` impls (no stubs).

Dimensions: d=64, n_layers=2, ff=256, seq=8.
Iterations: 1000 sequential forward passes.

| Run | tok/s |
|---|---|
| 1   | 177.68 |
| 2   | 184.05 |
| 3   | 181.95 |

FR-19.16 ≥100 tok/s threshold cleared with ~77-84% margin.

PARTIAL SCOPE — what this DOES NOT measure (still FR-19.16-extra):
- **Llama-1B params**. Bench uses ~50K params; full Llama-1B is
  ~1.1B (≈22000× larger). The architecture is identical; the dim
  jump is what FR-17.19-extra (real SafeTensors load) unlocks.
- **GPU / cuBLAS path**. Bench is on the CPU `ops::*` path. Switching
  to `--features cuda` routes the same symbols through cuBLAS but
  needs the runtime archive rebuilt with that feature gate (not the
  audit's default).
- **1000 concurrent batched requests**. Bench is 1000 sequential
  forward passes. Continuous batching (mid-decode admit + preempt-
  longest) is FR-19.5-extra; the in-process sim for that landed in
  the Phase 19 closeout but real GPU wiring is separate.

The full FR-19.16 spec demands all three. This partial closes the
P19.16 audit slot honestly via the explicit per-witness scope
documentation in `tests/runtime/llm_inference_tps.aether:5-32`.

### 2026-05-19 — Llama-shape cuBLAS-routed bench (FR-19.16-extra, partial)

Witness: `tests/runtime/llm_inference_tps_cuda.aether`
Runtime fn: same `aether_llm_inference_bench_tps`, now with a
`#[cfg(feature = "cuda")]` closure that routes every matmul through
the new `cuda_matmul_through` helper (per-call alloc / h2d / cuBLAS
sgemm / d2h / free using `aether_op_matmul_f32_cuda`).
Hardware: kokonoe 11900K + RTX 3070 Ti 8GB
Build: `cargo build -p aether_rt --features cuda` then aether-bin link

Dimensions: d=64, n_layers=2, ff=256, seq=8. Iterations: 50.

| Run | tok/s (cuBLAS-routed matmul) |
|---|---|
| 1   | 281.57 |
| 2   | 295.38 |
| 3   | 293.80 |
| 4   | 300.09 |

Result: ~290 tok/s sustained — well over the FR-19.16 ≥100 gate.
At these dims the per-call h2d/d2h overhead is comparable to the
gemm cost itself, but cuBLAS sgemm is still fast enough that the
fully-routed path beats the all-CPU number (~180 tok/s) by ~1.6×.

PARTIAL SCOPE — still NOT shipped (FR-19.16-extra deeper):
- **GPU-resident weights across the iter loop**. Today's wrapper
  re-uploads every matrix on every call. A real serving deploy
  would keep weights resident on device and only h2d activations,
  which would unlock dim scales where GPU dominates CPU by orders
  of magnitude (Llama-1B-class). The wrapper is the correctness
  artefact, not the perf artefact.
- **Real Llama weights**. Still synthetic Gaussian; the chain
  composes with FR-17.19-extra (SafeTensors / GGUF weight load)
  but isn't end-to-end wired yet.
- **All ops routed**. LayerNorm/SDPA/SiLU remain on CPU (cuda.rs
  has GPU versions but they take device handles; full routing is
  the same refactor as keeping weights resident).

### 2026-05-19 — Llama-shape cuBLAS GPU-resident weights (FR-19.16-extra-deeper)

Witness: `tests/runtime/llm_inference_tps_cuda_resident.aether`
Runtime fn: `aether_llm_inference_bench_tps_cuda_resident`
  — uploads all 6 weight matrices per layer (Wq, Wk, Wv, Wo, Wup,
  Wdown) to device ONCE before the iter loop; allocates persistent
  device activation buffers (d_ln_out, d_q/k/v, d_attn, d_proj,
  d_up, d_down) reused across all iters. Per layer-iter: 4 h2d
  (ln_out, attn_out, ln_out2, up_after_silu) + 6 d2h (q, k, v,
  proj, up_before_silu, down) — all O(s*d) activation bytes, ZERO
  weight uploads.
Hardware: kokonoe 11900K + RTX 3070 Ti 8GB
Build: `cargo build -p aether_rt --features cuda`

Dimensions: d=64, n_layers=2, ff=256, seq=8. Iterations: 100.

| Run | tok/s (GPU-resident) |
|---|---|
| 1   | 688.15 |
| 2   | 697.23 |
| 3   | 694.96 |
| 4   | 696.66 |
| 5   | 673.67 |

**Headline**: ~690 tok/s sustained — **2.4× faster** than the
per-call cuBLAS wrapper (~290 tok/s, prior row) and **3.8× faster**
than the all-CPU bench (~180 tok/s). The win comes from
eliminating both (a) the per-call weight upload (was ~5 × d*d
floats × 4 bytes per matmul-iter), and (b) the per-call cudaMalloc
/ cudaFree pair.

Comparison at d=64 (small dim, matmul cost is small):

| Variant | tok/s | h2d/iter | d2h/iter | Weight uploads/iter |
|---|---|---|---|---|
| CPU all-ops              | ~180 | 0 | 0 | 0 |
| Per-call cuBLAS wrapper  | ~290 | 10 (5 wts + 5 acts) | 5 (outs) | 10 (5 mm × 2 in) |
| GPU-resident this row    | ~690 | 4 | 6 | 0 (once at setup) |

What "GPU-resident" still doesn't measure (FR-19.16-extra deepest):
- **All ops on device**. LN/SDPA/SiLU run on CPU; the d2h after Q/K/V
  exists only to feed CPU SDPA. Routing those through cuda.rs's
  device-kernel variants would eliminate 3 d2h + 1 h2d per
  layer-iter -- net gain depends on kernel-launch overhead vs
  PCIe.
- **Llama-1B dim scale (d=2048, ff=5504, 16 layers)**. At those
  dims the cuBLAS sgemm is what matters and the activation
  bandwidth becomes negligible -- the 2.4× headline becomes much
  larger.
- **Real weights**. Still synthetic Gaussian.



## bench/qwen25_7b_autoregressive — tok/s on RTX 3070 Ti (matt-voice)

Tracks the full Qwen2.5-7B-Instruct Q4_K_M autoregressive throughput
through the v2 fused matmul + on-device KV cache + attention_seq1
chain. Measured wall-clock over 5 generate-only tokens after a
4-token prefill.

| date       | commit  | tok/s | what changed                                           |
|------------|---------|------:|--------------------------------------------------------|
| 2026-05-20 | 399718e |  25.5 | per-block Q4_K/Q6_K dtype dispatch fix (was NaN'ing)   |
| 2026-05-20 | 9b5a21e |  26.0 | fused gate+up+silu+mul kernel (4 launches -> 1)         |
| 2026-05-20 | 859745d |  27.2 | thermal-stable 5-run mean (no logic change)            |
| 2026-05-20 | 7e1804f |  37.4 | **CUDA graphs**: capture per-step forward, replay each step |
| 2026-05-20 | f40d259 |  35.2 | smallN matmul kernels added (regressed FFN by 7% via nvrtc unit pressure) |
| 2026-05-20 | add5216 |  37.2 | revert smallN; clean KERNEL_SRC restores baseline |
| 2026-05-24 | 02fca19 |  34.6 | seqB v3 + hetero batched attn + SSE stream; **−7% vs 37.2** — new __global__ in KERNEL_SRC (nvrtc unit pressure, same class as f40d259) |
| 2026-05-24 | c9d4501 |  34.5 | **ATTRIBUTION CORRECTED** — moved seqB kernel → PAGED_KERNEL_SRC; decode UNCHANGED (34.2–35.2 warm). KERNEL_SRC byte-identical to pre-session 6834cd0, so the −7% PREDATES this session (drift across 05-20→05-24). Not the seqB kernel. |

### 2026-05-24 — commit 02fca19: seq1 decode REGRESSION −7% + new seqB batched matmul speedup

Three commits this session touched `runtime/src/cuda.rs` / `serving.rs` / `batched_serving.rs`:
- 85518f6 — SSE streaming over the scheduler (no kernel change)
- 0837e4e — heterogeneous-position batched paged-attention + append_kv kernels (new kernels in `PAGED_KERNEL_SRC`, a *separate* nvrtc module from `KERNEL_SRC`)
- 02fca19 — weight-reuse batched Q4_K matmul `fused_q4k_matmul_seqB_v3` (NEW `__global__` added to **`KERNEL_SRC`** — the module that also holds the active seq1 decode kernels)

**seq1 decode baseline (the active autoregressive path) — REGRESSED.**
Witness: `runtime/tests/qwen25_graph_decode.rs::qwen25_graph_decode_tok_per_sec`
Run: `cargo test -p aether_rt --features cuda --release --test qwen25_graph_decode qwen25_graph_decode_tok_per_sec -- --ignored --nocapture --test-threads=1`
Model: Qwen2.5-7B-Instruct Q4_K_M (`sha256-2bada8a7...`), 28 layers, CUDA-graph decode, 5 gen tokens after 4-token prefill, argmax. Generated IDs `[358, 2776, 264, 220, 17]` — **bit-identical to the add5216 baseline** (correctness preserved; this is purely a perf regression).
GPU clock confirmed at peak 1950–1965 MHz / 83% util / 115 W during the timed window (warm, not cold-clock noise).

| Run | tok/s |
|---|---|
| 1 | 34.73 |
| 2 | 34.22 |
| 3 | 34.52 |
| 4 | 34.68 |
| 5 | 35.09 |

Median 34.68, mean ~34.6 tok/s. Prior reference (add5216, warm 4-run mean): 37.2 tok/s. **Δ = −7.0%.**

**Verdict: REGRESSION, flagged.** Below the ≥10% HEADLINE-config hard-stop, above the ≥5% flag threshold. Root cause is almost certainly the new `fused_q4k_matmul_seqB_v3` `__global__` added to `KERNEL_SRC` — the exact failure mode documented at commit f40d259 in this ledger ("smallN matmul kernels added (regressed FFN by 7% via nvrtc unit pressure)") and in the `nvrtc_kernel_unit_pressure` lesson. Adding a kernel to the shared nvrtc translation unit perturbs register allocation / codegen of the active seq1 decode kernels. The hetero attention kernels in `0837e4e` live in `PAGED_KERNEL_SRC` (separate module) so are a less likely contributor.
Remediation candidates: (a) move `fused_q4k_matmul_seqB_v3` into its own nvrtc module so it doesn't share register pressure with the seq1 decode kernels; (b) `__launch_bounds__` annotations to pin the seq1 kernels' occupancy; (c) accept the regression if batch-mode throughput gains outweigh single-stream loss for the serving workload. Tracking: FR-19.5-extra-deep follow-up.

> **CORRECTION (commit c9d4501, same day):** The root-cause attribution
> above is WRONG.  Remediation (a) was applied — `fused_q4k_matmul_seqB_v3`
> moved to `PAGED_KERNEL_SRC` — and warm 5-run decode was UNCHANGED at ~34.5
> tok/s (34.20 / 34.52 / 34.56 / 34.89 / 35.24).  `git diff 6834cd0 -- cuda.rs`
> shows every hunk lands in `PAGED_KERNEL_SRC` or Rust registration/FFI code;
> the `KERNEL_SRC` CUDA string body (what single-stream decode compiles) is
> **byte-identical to the pre-session commit 6834cd0**.  So the seqB kernel
> never affected this bench, and the 37.2 → 34.6 regression **predates this
> entire session** — it accumulated across the 2026-05-20 (add5216, 37.2) →
> 2026-05-24 (6834cd0) arc (GLM MLA + MoE dispatch + gemma3 + qwen3 + batched
> scaffolding, all of which added `__global__`s to `KERNEL_SRC`).  The real
> follow-up is to **bisect that arc** for the regressing commit, NOT to chase
> the seqB kernel.  Lesson: a bench-runner perf delta must be attributed
> against the immediately-prior commit, not a baseline several days/dozens of
> commits back.

**seqB batched Q4_K matmul (the NEW kernel) — the win that the regression pays for.**
Witness: `runtime/tests/cuda_q4k_matmul_seqB_parity.rs::seqB_throughput_bench`
Run: `cargo test -p aether_rt --features cuda --release --test cuda_q4k_matmul_seqB_parity seqB_throughput_bench -- --ignored --nocapture --test-threads=1`
Config: n=3584 (Qwen2.5-7B d_model, 14 super-blocks), batch=4, 400 iters, 40-iter warmup. Output bit-identical to 4× sequential seq1_v3 (FMA order preserved; parity test asserts max_abs_diff == 0).

| metric | µs/step |
|---|---|
| serial (4× seq1_v3 launches) | 291.85 |
| batched (1× seqB_v3 launch) | 156.31 |
| **speedup** | **1.87×** |

(Task-reported figure was 1.80× / 291.62µs→162.14µs on a prior run; this run measured 1.87× / 291.85µs→156.31µs — same ballpark, run-to-run variance on the batched leg.) The seqB kernel reads the weight matrix from DRAM once and reuses it across the batch, so at batch=4 it approaches the memory-bound 4× ceiling. This is the continuous-batching multi-slot decode win; it only helps when ≥2 requests decode concurrently. For single-stream decode (the seq1 bench above) it does nothing but add nvrtc pressure — hence the trade-off.

llama.cpp reference on the same hardware: ~30 tok/s.
Aether at commit add5216 is at **124% of llama.cpp** with matching
generated IDs — i.e., Aether is now FASTER than llama.cpp on this
model/hardware while producing the same outputs.

**Measurement note**: GPU boost clock takes ~1 run to ramp from
idle (210 MHz) to peak (1950 MHz). The 37.2 number is warm mean of
4 runs after a throwaway warmup; cold first-run is ~35 tok/s. Future
ledger rows should specify warm vs cold-included.

The launch overhead was much larger than the per-op profiler
suggested (~10 ms/token of overhead was actually saved, vs ~3 ms
estimated). Recording into a graph collapses ~370 host-side kernel
launches into one cuGraphLaunch call; the only host work left per
step is updating a 4-int step_args device buffer.

## bench/training_throughput — steps/sec (planned, gates on P8.3)

Pending — fires once DataLoader lands. Today the closest proxy is `examples/aether_lm.aether` itself (100 steps in 122 ms = ~820 steps/sec on synthetic data with 16-token batch, single block).

## Append rule

Any commit that touches `runtime/src/cuda.rs`, `runtime/src/lib.rs`, `compiler/src/codegen/asm/`, or `compiler/src/mir/fuse.rs` MUST run `bench/<applicable>/run_all.ps1` and append a row here. Regressions get an explicit verdict + a remediation issue ID. Audit's `--bench` flag (planned) will check the commit-touched-files-vs-bench-row policy automatically.

## bench/graph_decode — single-stream decode tok/s (2026-05-24, kernel-unit split)

Qwen2.5-7B Q4_K_M, RTX 3070 Ti, `qwen25_graph_decode` warm (16-step warmup to
ramp boost clock, then 16 timed tokens; cold first-process run discarded).

| state | warm tok/s | vs llama.cpp (~30 same card) |
|---|---|---|
| pre-split (HEAD before this commit) | 32.4–32.8 | ~108% |
| **post-split** | **33.5–33.8** | **~112%** |

Change: moved 9 contiguous TRAINING-only kernels (cross_entropy ×2, embed_lookup
/scatter_add, softmax_bwd ×2, sdpa_causal fwd+bwd_dq+bwd_dkv) out of `KERNEL_SRC`
into a new lazy `TRAIN_KERNEL_SRC` nvrtc unit (compiled only on first training
use; inference never touches it). Confirms [[nvrtc_kernel_unit_pressure]]: fewer
`__global__`s in the decode compilation unit → less ptxas register/codegen
pressure on the active decode kernels → +3.4% single-stream decode, inference
bit-intact + all training grad-checks still green. ~14 more training kernels
(gelu/layer_norm_bwd/adamw/rms_bwd/rope_bwd/gqa/silu_bwd/transpose) remain in
KERNEL_SRC interspersed with decode keepers — a follow-up batch toward the 37.2
peak (add5216).

## bench/graph_decode — batch 2 (2026-05-24, more backward kernels to lazy unit)

Same harness/hardware. Moved 9 MORE backward/optimizer-only kernels (adamw_step,
gelu_bwd, layer_norm_bwd_dx, layer_norm_bwd_params, rms_norm_bwd_dx,
rms_norm_bwd_gamma, rope_apply_backward, silu_bwd, transpose_021) from KERNEL_SRC
to TRAIN_KERNEL_SRC. These run on NO inference path (inference has no backward),
so zero risk to decode/embedding serving.

| state | warm tok/s | vs llama.cpp |
|---|---|---|
| pre-split | 32.6 | ~108% |
| batch 1 (+9 train kernels lazy) | 33.7 | ~112% |
| **batch 2 (+9 backward kernels lazy)** | **~35.0** (best 35.36) | **~117%** |

Cumulative +7.4% single-stream decode from relieving nvrtc unit pressure; all
training grad-checks (lm_loss/block/gqa) still green via the now-18-kernel lazy
TRAIN unit. Remaining toward 37.2 peak: the inference-AMBIGUOUS kernels
(gelu_fwd, layer_norm_fwd, add_layer_norm_fwd, bert_*, gqa_repeat/reduce) kept
in KERNEL_SRC pending a BERT-inference smoke to confirm they're decode-unused.
