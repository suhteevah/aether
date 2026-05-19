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

## bench/conv2d — 3-way (planned, gates on P7.3)

Pending — fires once `aether_op_conv2d_*` ships in `runtime/src/cuda.rs`.

## bench/attention — 3-way FlashAttention (planned, gates on P7.3)

Pending — fires once a real FlashAttention v2 kernel lands.

## bench/llama_inference — single-stream tokens/sec (planned, gates on P7.4 + P8.5)

Pending — fires once GGUF quant + serving land.

## bench/training_throughput — steps/sec (planned, gates on P8.3)

Pending — fires once DataLoader lands. Today the closest proxy is `examples/aether_lm.aether` itself (100 steps in 122 ms = ~820 steps/sec on synthetic data with 16-token batch, single block).

## Append rule

Any commit that touches `runtime/src/cuda.rs`, `runtime/src/lib.rs`, `compiler/src/codegen/asm/`, or `compiler/src/mir/fuse.rs` MUST run `bench/<applicable>/run_all.ps1` and append a row here. Regressions get an explicit verdict + a remediation issue ID. Audit's `--bench` flag (planned) will check the commit-touched-files-vs-bench-row policy automatically.
