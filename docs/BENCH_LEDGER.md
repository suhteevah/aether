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
