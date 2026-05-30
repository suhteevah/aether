# Aether FRs / notes — driven by the claudeai workbench (verification + serving-parity)

**Source project:** `J:\claudeai\` (the brain+body workbench / OpenClaw fleet operator + per-arch verification + parity research)
**Started:** 2026-05-29
**Owner of this list:** maintained by Claude on the claudeai-workbench side. This is the inbox for cross-cutting findings the workbench surfaces about aether-as-served-substrate that the aether dev session should have — **not** an attempt to do aether engineering from here (the aether tree has its own dev agent; this side is hands-off the source).
**Sibling lists:** `OPENCLAW_FR.md` (inference-rollout driven), `MATT_VOICE_FR.md` (LoRA-training driven), `ANTCOLONY_FR.md` (RL-trainer driven). Shared concerns are cross-linked, not duplicated.

> Note channel, not a work queue. The aether dev agent owns priority + sequencing. These are dropped here so they survive across sessions and aren't lost in the workbench's memory only.

---

## Context: there is no single "aether vs llama.cpp" serving number (2026-05-28)

From the dev agent's 1:1 bench (`docs/BENCH_LEDGER.md`, HEADs `8e3c058` P100 / `6bc03c0` 3070 Ti). Identical GGUF (Qwen2.5-7B Q4_K_M), 4 prompts, locked clocks. The result is **platform-dependent**:

- **cnc P100 / Linux (the actual fleet HW): llama-server b8182 wins ~2.7×.** Compute-bound; kernel quality decides. llama.cpp's Pascal mmq quant matmuls beat aether's generic nvrtc seq1 matmul.
- **kokonoe 3070 Ti / Windows WDDM: aether wins ~1.6× vs ollama** (98% util / 210W vs 12% / 88W) because aether's CUDA-graph decode bypasses the per-launch WDDM tax.

**Operational consequence for the fleet:** until aether reaches P100 serving parity, llama.cpp stays the production fleet backend. aether's current edge is sovereignty + training (cross-GPU PP QLoRA), not raw P100 serving throughput. See `OPENCLAW_FR.md` (the cutover gate) and `[[reference_aether_parity_context]]` / `[[reference_aether_per_arch_verification]]` in claudeai memory.

## FR-CLW-1 (clarification, HIGH value / low cost) — the P100 parity lever is **fp16, NOT dp4a**

If/when the P100 quant-matmul parity work proceeds, target the right kernel so no effort is spent on an instruction the card doesn't have:

- cnc P100s are **GP100 / compute capability 6.0**. The `__dp4a` int8 dot-product intrinsic is **sm_61+** (P40 / P4 / GTX-10xx — the *consumer/inference* Pascal). **P100 (sm_60) does not have dp4a.** llama.cpp's mmq on sm_60 already falls back off dp4a.
- P100's real edge is **native 2:1 FP16** (full-rate `half2` — it's the FP16/FP64 HPC Pascal).
- Right target = **fp16-MAC into an fp32 accumulator** tiled quant matmul (dequant → fp16, `half2` multiply-add, **fp32 accumulator** for the long-K reductions at gate/up 18944 and lm_head). Pure-fp16 accumulation over thousands of terms drifts → the vocab-1 / NaN territory the V2-Lite/GLM chases already lived in.
- Anyone reading a BENCH_LEDGER row that says "dp4a int8 path like llama's mmq" should read it as shorthand for "llama.cpp's mmq *approach*"; on sm_60 the concrete kernel is the fp16 fallback, not dp4a. (Matt confirmed the target is fp16.)

## FR-CLW-2 (open verification gap) — honest Ampere reference bench is still PENDING

The 3070 Ti "aether wins ~1.6×" row is measured **vs ollama, not vs raw llama.cpp-the-engine** — and that gap may be ollama's wrapper, not the engine's:

- **llama.cpp has had CUDA graphs since 2024** (NVIDIA-contributed, Alan Gray): `GGML_CUDA_USE_GRAPHS` is **default ON in any CUDA build** (kill switch `GGML_CUDA_DISABLE_GRAPHS=1`). Same decode-graph mechanism aether's decode path uses. **Don't reinvent it.**
- Caveats: graphs only engage for single-token decode (batch=1); llama.cpp **disables graphs for MoE** (variable expert routing changes node count) — so qwen3moe / GLM-flash don't benefit on llama.cpp either. A clean Ampere comparison is dense models (Qwen 7B/14B).
- **Pending action (offered to Matt, awaiting go):** grab a raw prebuilt `llama-bXXXX-bin-win-cuda` release (graphs default-on), confirm graphs engage (util + log), re-run the dev agent's locked-clock harness vs aether on the 3070 Ti. **If raw llama.cpp also hits ~98% util, the "1.6× Ampere win" evaporates** and the ledger row should be re-stated as "vs ollama" only. This bench runs from the claudeai/kokonoe side when Matt greenlights it; flagging here so the ledger's Ampere row carries the caveat until then.

## Cross-ref — per-arch verification (claudeai-side, 2026-05-25)

The workbench independently smoke-verified the deployed `aether-serve` per architecture before the dev agent closed the gaps. Recorded at `[[reference_aether_per_arch_verification]]` (claudeai memory). All gaps (V2-Lite, qwen3moe, gemma3, codestral) were closed by the dev agent by 2026-05-26 — listed here only so the dev session knows the workbench has a matching record and the matrix is considered CLOSED on this side too.

## Sources (CUDA-graphs research)

- NVIDIA: *Optimizing llama.cpp AI Inference with CUDA Graphs* — developer.nvidia.com/blog
- llama.cpp #6763 (CUDA Graphs impl), #15013 (perf discussion), #5178 (Windows low-util issue)
