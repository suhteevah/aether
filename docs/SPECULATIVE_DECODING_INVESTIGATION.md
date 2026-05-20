# Speculative decoding — investigation

**Date:** 2026-05-20
**Status:** Investigation only. No implementation in this branch.
**Baseline:** Qwen2.5-7B-Instruct Q4_K_M, RTX 3070 Ti, **37.22 tok/s** warm.

## Summary

Speculative decoding could **theoretically push Aether to ~80-90 tok/s on Qwen2.5-7B (≈2.4x current)**, but requires building a multi-token forward path (seq>1 attention + matmul kernels) which today doesn't exist. The empirical break-even analysis shows the naive "just verify with repeated seq=1 launches" approach cannot win at any acceptance rate ≤ 1.0. Recommended only if the multi-token infrastructure is needed for ANOTHER reason (batched serving, larger contexts, parallel sampling) — otherwise the ~5-8 day effort is hard to justify against the +60-100% upside.

## What speculative decoding is

Single-stream LLM decoding is bandwidth-bound: every token requires streaming the full ~4 GB of weights through DRAM once. Speculative decoding amortizes that across multiple candidate tokens:

1. A small **draft model** (e.g., Qwen2.5-0.5B, ~14x smaller than 7B) proposes the next N tokens cheaply, one at a time.
2. The big **target model** verifies all N candidates in a SINGLE forward pass — same weight bytes streamed once, N rows of input processed simultaneously.
3. The verification computes target logits at positions [t, t+1, ..., t+N-1]. Greedy comparison against draft proposals yields an accepted prefix of length r·N (where r is acceptance probability per position).
4. The accepted prefix gets committed; verification rejected one is replaced with the target's own sample; the chain continues.

If verification cost ≈ single-token cost (the bandwidth-bound case), per-accepted-token cost drops from 26.8 ms to `26.8 / (r·N)` ms. For typical r=0.6, N=4: 11.2 ms/token = **89 tok/s**.

## Empirical break-even analysis

`runtime/tests/spec_dec_naive_verify_bench.rs` measures the cost of the naive approach: verify N candidates by running the existing seq=1 CUDA graph N times.

```
seq=1 forward:   26.28 ms = 38.0 tok/s
seq=4 verify:   105.09 ms (4.00x single)
seq=8 verify:   208.61 ms (7.94x single)
```

Cost scales **linearly** in N. Break-even acceptance rate (for naive verify to match baseline):
- N=4: need r > 99.96% (impossible)
- N=8: need r > 99.21% (impossible)

**Naive repeat-launch verification mathematically cannot win at any realistic acceptance rate.** Real multi-token kernels are required.

## Architecture gap

Aether's current kernel suite is heavily seq=1 specialized. The matt-voice deploy was built around `seq=1` autoregressive decoding only. What's missing:

| Kernel | Current state | Needed for spec-dec |
|---|---|---|
| `fused_q4k_matmul_seq1_v2` | input [1, D_in] → output [1, D_out] | seq>1: [N, D_in] → [N, D_out] |
| `fused_q6k_matmul_seq1_v2` | same | same |
| `fused_q4k_ffn_gate_up_silu_mul` | seq=1 | seq>1 batched FFN |
| `attention_seq1` / `_devarg` | Q is [n_q_heads * head_dim] | Q is [N, n_q_heads * head_dim], attend [cur_seq + N] |
| `rope_apply_devarg` | seq=1 at one pos | seq=N at consecutive positions |
| `append_kv_devarg` | append 1 (K,V) row | append N (K,V) rows |

That's **6 kernels to refactor or duplicate**. Each needs:
- New CUDA kernel C source
- Rust wrapper
- Parity test against the seq=1 path
- Adjustment of step_args layout (now needs pos_start + N, plus per-N rope positions)
- New CUDA graph capture for the seq=N forward

Plus orchestration:
- Draft model load (Qwen2.5-0.5B Q4_K_M is 398 MB in ollama registry — needs to be loaded, separate KV cache)
- Two separate graphs (seq=1 for draft, seq=N for verify), maybe a 3rd for the initial prefill
- Acceptance comparison logic (compare draft argmax to target argmax at each verify position)
- KV cache rollback for rejected tokens (truncate cache back to last-accepted)
- Token sampling: when verify rejects a position, sample from target's distribution at that position

## Effort estimate

| Task | Effort |
|---|---|
| seqN matmul kernels (Q4_K + Q6_K + FFN) | 2 days |
| seqN attention kernel | 1 day |
| seqN RoPE + append_kv variants | 0.5 day |
| Draft model load + GGUF parsing for 0.5B | 0.5 day |
| Two-graph capture + orchestration | 1 day |
| Verification + KV rollback | 1 day |
| Parity tests, correctness validation | 1 day |
| Tuning + bench | 0.5-1 day |
| **Total** | **~7-8 days** |

## Expected speedup

Acceptance rate depends heavily on draft/target alignment and prompt type. Published numbers:
- Same-family draft (Qwen2.5-0.5B drafting for 7B): typical r ∈ [0.55, 0.75] (lossless: target prediction always wins when divergent)
- Greedy decoding: tighter alignment, higher r
- Stochastic / temperature > 0: lower r (~0.4-0.6)

Conservative scenarios:

| N | r | Verify cost | Draft cost (0.5B at ~3ms each) | Per accepted token | tok/s |
|---|---|---|---|---|---|
| 4 | 0.5 | ~28 ms (1.05x) | 12 ms | 28+12 / 2 = 20 ms | **50** |
| 4 | 0.6 | ~28 ms | 12 ms | 40 / 2.4 = 16.7 ms | **60** |
| 4 | 0.7 | ~28 ms | 12 ms | 40 / 2.8 = 14.3 ms | **70** |
| 8 | 0.5 | ~30 ms | 24 ms | 54 / 4 = 13.5 ms | **74** |
| 8 | 0.65 | ~30 ms | 24 ms | 54 / 5.2 = 10.4 ms | **96** |

**Realistic expectation: 1.5x — 2.4x speedup, putting Aether at 55-90 tok/s on Qwen2.5-7B.**

Note: these assume the seq=N forward stays roughly bandwidth-bound (verify cost grows only marginally with N, not linearly). If our kernels can't achieve that — e.g., if shmem pressure forces lower occupancy at seq=N — the verify cost grows toward N·single, killing the speedup. The kernel implementation needs care.

## Risks

1. **Multi-token matmul kernels regress seq=1**: From the prior session, adding new `__global__` kernels to `KERNEL_SRC` can regress active kernels via nvrtc unit pressure. Each new seqN kernel risks slowing down the seq=1 hot path it's meant to complement. Mitigation: maybe ship the seqN kernels in a SEPARATE nvrtc module instantiated only when spec-dec is enabled.
2. **CUDA graph state across draft/verify**: With two graphs sharing the same KV cache buffers, capture must avoid recording stale pointer states. Probably needs separate device buffers per graph.
3. **Acceptance rate may be lower than published**: For matt-voice's use case (creative voice-driven prompts) the acceptance rate could be 0.3-0.5, putting realistic speedup at 1.2-1.5x rather than 2x.
4. **Draft model accuracy mismatch on edge cases**: Q4_K_M quantization of a 0.5B model is lossy enough that some prompts may produce drafts that DIVERGE from target consistently. Lossless verification handles this correctly (just falls back to target sampling) but throughput suffers.

## Alternative: batched serving

The multi-token kernel infrastructure required for speculative decoding is **the same as what's needed for batched serving** (process B independent requests through one forward pass, sharing weight reads). If matt-voice's deployment target is multi-user (Telegram bot, web API), batched serving gives:
- 4x throughput at B=4 with ~linear cost
- Per-request latency stays the same (single user feels no change)
- All gains come from amortizing weight reads across batches

vs speculative decoding:
- 1.5-2.4x throughput on single-user latency
- Per-token latency drops (single user sees faster output)
- Gains come from amortizing weight reads across speculative tokens

Both share kernel infrastructure. **Whichever the deployment target needs more, the other comes nearly for free.**

## Recommendation

**Defer speculative decoding unless one of:**
1. matt-voice deployment specifically needs single-stream latency improvement (e.g., real-time voice synthesis demand). In that case, speculative is the cleanest path.
2. Batched serving is also wanted, in which case the work is shared and the bundle becomes a clear high-value investment (multi-day effort, opens both speculative + batched at once).

**Otherwise, the current 37.22 tok/s (124% of llama.cpp) is already the strong state**, and the next-highest-value work is:
- Stability/correctness improvements
- The matt-voice deploy critical path (TLS, serving HTTP, OpenAI-compat, etc.)
- Multi-host (cnc 2× P100) workloads where the kernel reuse is also relevant

If pursued, the minimum-risk path is: build the seqN matmul + attention kernels FIRST as a separate compile unit, prove they don't regress seq=1 baseline, THEN add the spec-dec orchestration layer.

## Files in this investigation

- `runtime/tests/spec_dec_naive_verify_bench.rs` — empirically establishes that naive repeat-launch verify can't win at r ≤ 1.0
- This document — analysis + effort estimate + recommendation

No production code changed.
