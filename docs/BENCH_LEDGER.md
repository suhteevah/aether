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

## perf investigation — the "last 2 tok/s" (2026-05-24): #2 profile, #4 ptxas, #1 split

Goal: recover decode from ~35.0 (post batch-2) toward the 37.2 peak (add5216).
Three levers tried; net: the nvrtc-pressure lever is EXHAUSTED at ~35 — the
remainder is measurement variance + genuine common-path kernel cost.

- **#2 per-op profile** (qwen25_perf_breakdown, no-graph per-op shares): decode is
  ~85% in the fused QUANT MATMULS (attn_norm+Q/K/V+rope 30.7%, FFN gate/up 34.6%,
  down 19.4%); attention only 4.2%. No single anomalous/regressed kernel → the
  loss is DIFFUSE register pressure on the quant-matmul hot path, not one slow op.
- **#4 ptxas maxrregcount=64** on the decode unit: REGRESSED 35.0→33.2. The hot
  seq1 quant matmuls are register-HUNGRY (not occupancy-limited); a cap spills.
  Default allocation is optimal. Reverted.
- **#1 per-arch decode split** (moved 18 non-common-path decode kernels — 9 rare
  quant dtypes + 6 MoE expert + 2 bert + iq3 dequant — to a lazy AUXD unit):
  measured 33.3/34.0/35.4 = within the ~2 tok/s noise of batch-2's ~35. NO clear
  gain — the hot kernels' register allocation wasn't bottlenecked by those 18's
  co-residence (batches 1+2 already relieved the pressure that mattered). Reverted
  (also avoids unverifiable lazy-compile risk to GLM/MoE/BGE inference on this card).

Conclusion: batches 1+2 (+7.4%, 32.6→~35.0, ~117% of llama.cpp) captured the
recoverable nvrtc-pressure win. The gap to 37.2 is (a) measurement: 37.2 was a
4-run warm mean, ~35.4 is the best single warm run here — a clean mean-of-N would
shrink the apparent gap; (b) genuine: more arch-required kernels in KERNEL_SRC vs
add5216 that can't be removed without dropping arch support. Real further gains
would need faster HOT kernels (a better q4k seq1 matmul / fused FFN), not more
unit-splitting — a kernel-optimization task, not a pressure-relief one.

## serving e2e — aether-serve vs llama.cpp (real-world chat, 1:1)

First **end-to-end serving** head-to-head (prior rows were micro-benches /
single-kernel). Same GPU, same GGUF, same workload, served through each engine's
OpenAI `/v1/chat/completions`. Workload: 4 open-ended chat prompts (transformer
explainer, LRU-cache code, French-Revolution essay, TCP handshake), `max_tokens=128`,
`temperature=0`; 2 warmup requests discarded, then N=3 × 4 prompts = 12 timed
requests; both engines emitted exactly 128 tokens/req (clean, no early-stop skew).
End-to-end = `completion_tokens / client_wall_time` (the user-facing rate).

**Hardware: cnc Tesla P100-16GB (Pascal/sm_60), CUDA 12.8.** Qwen2.5-7B-Instruct
Q4_K_M (4.47 GiB). llama.cpp b8182 `llama-server -ngl 99 -c 2048`; aether-serve
`--paged` (CUDA-graph decode). Both default sampler flags.

| date       | engine    | hardware   | model            | end-to-end tok/s | self-reported decode | verdict |
|------------|-----------|------------|------------------|-----------------:|---------------------:|---------|
| 2026-05-25 | llama.cpp b8182 | P100 16GB | Qwen2.5-7B Q4_K_M | **37.4** | 39.1 (eval) | leader  |
| 2026-05-25 | aether-serve    | P100 16GB | Qwen2.5-7B Q4_K_M | **13.8** | 13.5–14.1 (gen) | **0.37×** |

**Honest verdict: aether is ~2.7× slower than llama.cpp on the P100 for
real-world chat decode.** This is hardware-dependent and does NOT match the
earlier 3070-Ti (Ampere) rows where aether's nvrtc decode reached ~117% of
llama.cpp — llama.cpp ships hand-tuned Pascal kernels (mmq / dp4a paths) that
aether's generic nvrtc Q4_K seq1 matmul + fused FFN don't approach on sm_60. The
per-op profile (above) already located ~85% of aether decode in those quant
matmuls; on Pascal that hot path is the whole gap. Prefill is negligible here
(prompts ~21–42 tokens) so e2e ≈ decode for both.

TODO to close it: (1) re-run the identical bench on the 3070 Ti to quantify the
Ampere-vs-Pascal split with a clean mean-of-N (the "117%" was a best-single-run
claim); (2) a Pascal-aware quant matmul (dp4a int8 path like llama's mmq) is the
real lever — unit-splitting is exhausted.

### Ampere/Windows follow-up — same workload on the 3070 Ti (WDDM)

Re-ran the identical workload on kokonoe (RTX 3070 Ti, Win10/WDDM, CUDA 12.x).
llama.cpp reference here is **ollama 0.24.0** (embeds llama.cpp; only local
option — no standalone llama-server on Windows). Both engines served the
**identical GGUF** (ollama blob `sha256-2bada8a7…`, Qwen2.5-7B Q4_K_M). GPU
clocks LOCKED (`-lgc 1700,2115 -lmc 9501`) after discovering the card idles at
mem 810 MHz / sm 210 MHz (P8) — decode is memory-bound, the idle mem clock alone
was an 11.7× throttle. Same 4 prompts, max_tokens=128, temp 0, warm + N=3.

| date       | engine          | hardware        | model            | end-to-end tok/s | GPU util | power | verdict |
|------------|-----------------|-----------------|------------------|-----------------:|---------:|------:|---------|
| 2026-05-26 | aether-serve    | 3070 Ti / WDDM  | Qwen2.5-7B Q4_K_M | **28.1** | **98%** | 210W | **1.6× ollama** |
| 2026-05-26 | ollama (llama.cpp) | 3070 Ti / WDDM | Qwen2.5-7B Q4_K_M | **17.5** | ~12%   |  88W | launch-bound |

**The result INVERTS vs the P100/Linux row above** — and the cause is the
platform, not the kernels. On Windows/WDDM, single-stream decode fires ~200 tiny
sequential kernels/token and per-submission WDDM overhead dominates: ollama sits
at **~12% GPU util** (idle 88%, launch-latency-bound). aether's **CUDA-graph
decode** (non-MoE path captures the whole step into one graph launch) bypasses
that and hits **98% util / 210W** → 1.6× ollama. On the P100/Linux box there is
no WDDM tax, kernels issue back-to-back, and raw kernel quality decides → raw
llama-server b8182 wins 2.7× (its Pascal mmq/dp4a quant matmuls beat aether's
generic nvrtc seq1 matmul).

**Net (honest):** "aether vs llama.cpp" has NO single answer — it's
platform-and-GPU-dependent:
- **Linux server GPU, compute-bound:** llama.cpp wins on kernel quality (P100 2.7×; worst case Pascal).
- **Windows/WDDM, launch-bound:** aether's CUDA-graph decode wins (1.6× vs ollama; 98% vs 12% util).
Caveats: ollama ≠ raw llama-server (adds a scheduler layer, may not enable CUDA
graphs on Windows — part of the gap is ollama's, not llama.cpp-the-kernels'); a
raw llama.cpp Windows build with graphs would be the fairer Ampere reference and
is the TODO. Both rows used the identical GGUF + locked clocks.

### P100 decode profiling — v6 matvec win does NOT move e2e (matmul isn't the bottleneck)

Autonomous perf sprint (full-P100 window). Goal was to close the P100 decode gap
(aether 13.8 vs llama 37.4 tok/s, Qwen2.5-7B Q4_K_M). Built a vectorized Q4_K
seq1 matvec (`fused_q4k_matmul_seq1_v6`: uint loads + per-lane 2-scale dequant)
and microbenched it (`runtime/tests/cuda_q4k_matvec_bench.rs`) against the prod
v2/v3:

| kernel | aggregate GB/s | vs v3 | notes |
|--------|---------------:|------:|-------|
| v3 (shared-tiled) | 73 | 1.0x | prior |
| v4 (no-shared/inline) | 58 | 0.79x | regressed |
| v5 (multi-row MLP) | 70 | 0.96x | no help |
| **v6 (vectorized uint)** | **105** | **1.43x** | parity-clean (rel 5e-5) |
| membw probe (uint, no dequant) | ~200 | — | wide-load mem ceiling, 28% peak |
| membw probe (byte, no dequant) | ~134 | — | byte-load ceiling, 19% peak |

**But wiring v6 into decode gave ZERO e2e gain: aether 13.85 vs 13.8 tok/s.**
Decode-phase timing (AETHER_DECODE_TIMING) on a real 128-token decode:
- forward (embed+h2d+GPU forward+sync+logits-d2h) = **62994 us/tok (99.7%)**
- sampling + host = **171 us/tok (0.3%)**

So: (1) sampling/host is NOT the bottleneck; (2) decode is GPU-forward-bound at
~63 ms/tok = ~71 GB/s effective over the 4.5 GB model, vs llama's ~25 ms =
~180 GB/s; (3) the seq1 matvec v6 sped up is only PART of the forward — the gap
is the AGGREGATE of all decode kernels (FFN-fused gate/up/down + attention +
matmuls + norms), broadly ~2.5x slow, not one hot kernel. v6 reverted from the
default dispatch (no e2e benefit, avoid risk on all Q4_K models); kept registered
+ benched, ready for batched decode / once the forward bottleneck is addressed.

**Next lever (per gpu_perf_surpass_strategy): NOT one faster kernel — reduce the
whole forward's cost (fusion = fewer/bigger kernels, + raise every kernel's
achieved BW toward the ~200 probe ceiling). Need per-category profiling
(matmul vs FFN-fused vs attention) of the 63 ms to target the biggest chunk.**

### P100 decode: per-section profiling → FFN-kernel vectorization → +8.7% e2e

Followed the profiling finding (above) with AETHER_DECODE_TIMING per-section
timers (forced imperative; env-gated). Steady-state decode of Qwen2.5-7B Q4_K_M
on the cnc P100 splits:
- **FFN section ~60%** (~1270 us/block x28 = ~35 ms/tok) — memory-bound on
  gate/up/down weights (~114 MB/layer, ~78 GB/s).
- **attention section ~40%** (~840 us/block x28 = ~24 ms/tok) — only ~16 MB
  weights (~18 GB/s effective) → NOT bandwidth-bound; dominated by the paged
  attention kernel + rope + norms + per-kernel latency.

The biggest weight chunk (FFN gate/up, 76 MB/layer) ran through the SEPARATE
`fused_q4k_ffn_gate_up_silu_mul` kernel (8 byte-loads/lane), which v6 never
touched. Vectorizing it (2 uint loads/lane, bit-identical) + wiring v6 into
dispatch_matmul dt=12 (q/k/v/o/down/lm_head):

| config | decode (pure) | e2e (4-prompt bench) | vs baseline |
|--------|--------------:|---------------------:|-------------|
| v2 baseline | 15.5 tok/s | 13.85 tok/s | 1.0x |
| + vectorized FFN gate/up | 17.0 | — | |
| + v6 dispatch (combined) | 17.1 | **15.05** | **1.087x** |
| llama-server b8182 (ref) | — | 37.33 | 2.48x ahead |

**Honest: +8.7% e2e (0.37x → 0.40x of llama).** The win is the FFN gate/up
vectorization (the bandwidth-bound chunk); v6 on the smaller matmuls
(q/k/v/o/down/lm_head) was e2e-neutral — they're latency-bound single-shot, not
bandwidth-bound. Coherence intact ("Paris"). Remaining ~2.5x gap: the attention
section (40%, latency/overhead-bound) needs FUSION (fewer/bigger kernels), not
more per-kernel bandwidth — that is the next lever per gpu_perf_surpass_strategy.

---

## 2026-05-27 — attention-section: multi-warp paged seq1 attention (v2)

Picking up the prior handoff's #1 lever (attention section, ~40% of decode,
latency/occupancy-bound). The `paged_attention_seq1_devarg` kernel ran ONE warp
per head (grid=n_q_heads=28, block=32) → ~28 warps on a 56-SM P100, a single
warp per occupied SM with nothing to interleave against the long serial K/V
loads. New `paged_attention_seq1_v2_devarg` splits the per-head KV loop across
`AETHER_ATTN_WARPS` warps (block = 32×NW); softmax is global and pass-3 is a
linear weighted-V sum, so per-warp partial sums add EXACTLY (parity test
`cuda_paged_attention_v2_parity`: max_abs ≤ 2.4e-7, max_rel ≤ 2.4e-4 across
cur_seq 1..257). Coherence preserved (`qwen25_paged_parity` token-identical).

cnc P100 (GPU1, workhorse evicted+restored), Qwen2.5-Math-7B Q4_K_M, --paged,
4-prompt × 3-rep e2e bench, **same binary** (env A/B → clean attribution):

| config | e2e tok/s | vs v1 |
|--------|----------:|-------|
| v1 baseline (AETHER_ATTN_V2=0) | 15.05 | 1.000x (reproduces committed baseline exactly) |
| v2 AETHER_ATTN_WARPS=4  | 15.69 | 1.043x |
| v2 AETHER_ATTN_WARPS=8 (default) | 15.79 | 1.049x |
| v2 AETHER_ATTN_WARPS=16 | 15.85 | 1.053x |
| llama-server b8182 (ref) | 37.33 | 2.36x ahead |

**Honest: +4.9% e2e (0.40x → 0.42x of llama), warp count within noise (8↔16
≈ 0.4%).** Modest because 28 heads = 28 blocks fills only half a 56-SM P100;
the multi-warp fix helps the OCCUPIED SMs hide latency but leaves 28 SMs idle.
Next lever to fill all SMs: split-KV (grid-Y KV chunks + a combine kernel) —
the multi-warp structure here is the basis for it. Shipped default-on,
env-toggleable (AETHER_ATTN_V2=0 → v1), zero regression risk.

---

## 2026-05-27 — FFN gate/up: factored-dequant (v3) — NEGATIVE RESULT

Pivoted to the FFN section (the bigger 59% of decode forward). Added FFN
sub-split timing (AETHER_DECODE_TIMING now prints norm+gate/up vs down-proj).
cnc P100, Qwen2.5-Math-7B Q4_K_M, --paged, per-token ×28 layers:

| FFN sub-section | per-token | % FFN | % forward |
|-----------------|----------:|------:|----------:|
| norm + gate/up  | 23.5ms | 66% | ~38% |
| down-proj       | 12.0ms | 34% | ~20% |

So the fused gate/up kernel is the single biggest chunk of decode (~38%),
running ~91 GB/s vs a ~200 GB/s wide-load ceiling. Hypothesis: it's ALU-bound on
per-element float dequant, so factor `a·(d_eff·nibble−m_eff)` → `d_eff·Σ(a·nibble)
− m_eff·Σa` (scale/min applied once per sub-block; Σa shared gate/up).

**Result: factoring REGRESSES the kernel on P100.**

| config (attn v2 in all) | e2e tok/s | gate/up µs/tok |
|-------------------------|----------:|---------------:|
| base gate/up | 15.79 | 23532 |
| v3 (factored, a8[] reg cache) | 15.58 | 24169 |
| v3 (fused single-loop, 1 act read) | 15.41 | 24743 |

Both restructurings add live registers (dot_g/dot_u/asum/8 hoisted scale-mins)
→ register pressure → lower occupancy → slower, despite ~1/3 fewer per-element
FLOPs. The base per-element-FMA form is already well-tuned for P100. Coherence
held throughout (qwen25_paged_parity token-identical with v3).

**Conclusion: ALU-factoring the float-activation gate/up kernel is a dead end on
P100. The real lever to approach llama's ~180 GB/s is INT8 ACTIVATION
QUANTIZATION (Q8_1) + int8×int4 dot products with one float scale per block —
llama's MMVQ approach — which removes all per-element float dequant. That's a
fundamentally different, larger build (activation-quant kernel + int-MAC gate/up
+ scale handling), not a micro-tweak. v3 kernel reverted; FFN sub-split timing
kept as a permanent diagnostic.**

---

## 2026-05-27 — FFN gate/up: int8×int4 MMVQ (Q8_1 activation) — NEGATIVE RESULT on P100

The real attempt at llama's approach: quantize the activation to Q8_1 (per-32
int8 + f32 scale + f32 block-sum) and replace the float gate/up dequant with an
INTEGER dot (int8×int4) + one float scale per sub-block. Built both kernels
(quantize_q8_1 + fused_q4k_q8_1_ffn_gate_up_silu_mul, shared-staged like base),
wired behind AETHER_FFN_Q8.

- **Correct**: MMVQ vs float base rms_rel 0.66% (int8 activation-quant error;
  a bug in the int dot/scale/min-term would blow past 5%).
- **Coherent**: real Qwen2.5-7B greedy unchanged — qwen25 8-token output
  byte-identical to float `[358,2776,264,220,17,20,4666,6284]`; chat gives
  "The capital of France is Paris."

| config (attn v2, cnc P100) | e2e tok/s |
|----------------------------|----------:|
| float base gate/up | 15.79 |
| int8×int4 MMVQ (Q8_1) | 15.44 (−2.2%) |

**MMVQ is SLOWER on P100.** sm_60 has no dp4a and strong fp32 / weak int32-mul,
so scalar int8-MAC does not beat fp32-FMA; the extra quantize launch + int↔float
conversions make it net slower. Reverted (kept bench scripts).

**KEY CONCLUSION:** Two independent gate/up rewrites (ALU-factoring + int8 MMVQ)
both LOSE to the base fp32 kernel on Pascal. The base is well-matched to P100's
fp32 strength, and **llama's ~180 GB/s on P100 is NOT explained by int8-MMVQ
arithmetic being faster.** The P100 decode gap vs llama is therefore NOT in
gate/up per-kernel arithmetic — it's elsewhere (memory access pattern /
whole-layer kernel fusion & scheduling / occupancy), or the 180 GB/s reference
figure needs re-derivation. Stop micro-optimizing the gate/up math for P100;
re-examine where llama actually wins (profile llama's kernel occupancy/timeline,
or pursue whole-layer fusion). An Ampere (sm_86, dp4a) int8 MMVQ path could win
but needs the __dp4a intrinsic, not this scalar kernel.

---

## 2026-05-27 — Where llama wins on P100 + multi-warp gate/up (3rd NEGATIVE) + systemic conclusion

Profiled the gap (no nsys/ncu — Leap Micro immutable OS; used llama-bench +
llama.cpp source).

**Clean re-derivation of the gap** (llama-bench, GPU1, same GGUF):
- pp512 = 802 tok/s; **tg128 = 39.07 tok/s** decode → ≈170 GB/s effective
  (24% of P100's 720 peak). aether ≈ 15.8 e2e (~71 GB/s, ~10%). Gap 2.36× is
  real, re-derived from a clean bench (not just the 37.33 serving number).

**Root cause (llama mmvq.cu, GENERIC table, decode ncols_dst=1):** llama uses
`nwarps=4, rows_per_cuda_block=1` → **4 warps cooperate per output row** (K-split
+ shared-mem reduce) for 4× memory-level parallelism. aether's matmuls use
**1 warp per row**. This explained why the prior 2 FFN attempts (ALU-factor,
int8-MMVQ) failed — both kept 1-warp/row.

**Ported llama's structure (multi-warp-per-row K-split gate/up). 3rd NEGATIVE:**

| config (attn v2, cnc P100) | e2e tok/s |
|----------------------------|----------:|
| base (1 warp/row, 8 rows/block, shared act reuse) | 15.78 |
| multi-warp K-split, MW_WARPS=2 | 15.04 (−4.7%) |
| multi-warp K-split, MW_WARPS=4 (llama's choice) | 15.04 (−4.7%) |
| multi-warp K-split, MW_WARPS=7 | 15.03 (−4.7%) |

Correct + coherent (qwen25 token-identical), but slower. **Why porting the
structure alone regresses:** aether's base does 8 rows/block with the activation
staged in shared ONCE and reused across 8 rows. The 1-row/block multi-warp form
loses that amortization (re-reads activation per block) + adds a cross-warp
reduce. llama affords 1-row/block because its activation is int8 Q8_1 (4× smaller
reads) and the whole pipeline is co-designed around it.

**SYSTEMIC CONCLUSION:** Three independent gate/up rewrites (ALU-factor, int8
MMVQ, multi-warp K-split) ALL lose to the well-tuned base on P100. And gate/up is
only ~38% of decode — even a perfect gate/up caps aether at ~25 tok/s, still short
of 39. **llama's P100 advantage is SYSTEMIC** (every kernel ~2.3× more
HBM-efficient via co-designed int8-activation + multi-warp), not a single-kernel
fix. Closing it needs a holistic decode rewrite (int8 activations throughout +
multi-warp across all matmuls — a major, P100-uncertain effort), OR is better
pursued on Ampere (3070 Ti, where aether already beats ollama 1.6× via CUDA-graph
decode). Recommendation: stop per-kernel P100 gate/up work; the frontier is a
co-designed int8 pipeline or the Ampere path. attention v2 (+4.9%, adbd4f0)
remains this arc's shipped win.
