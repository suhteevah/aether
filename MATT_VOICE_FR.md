# Aether Feature Requests — driven by matt-voice QLoRA training

**Source project:** `J:\matt-voice\` (Discord-scraped corpus → fine-tuned LoRA that writes in Matt's actual voice)
**Started:** 2026-05-19
**Owner of this list:** maintained as Claude works on matt-voice training; updated whenever a candle feature is used that Aether doesn't have yet.
**Sibling list:** `J:\aether\ANTCOLONY_FR.md` (RL-trainer driven). Shared dependencies (Tensor, Autograd, CUDA, AdamW, safetensors) are intentionally duplicated; cross-link as items land.

## How this list is used

The matt-voice project is using **candle** (`J:\candle-src\`, fork at `suhteevah/candle:matt-voice-lora`) because Aether doesn't yet support QLoRA training over a Qwen2.5-arch GGUF base on a Pascal GPU. The candle fork already shipped end-to-end 7B QLoRA on an 8 GB Ampere card and is now running on a P100 via fp16. **Each candle feature below is a feature Aether should add so we can swap candle out for Aether eventually.**

The killer unlock that justifies this whole swap is **multi-GPU training**. Candle's `qwen-lora-train` is single-GPU. Aether's `#[distributed(world_size=N, backend="nccl", algorithm="ring")]` attribute is the language-native answer — once Phase 17 (tensor stack) and Phase 18 (distributed) land, both cnc P100s (28 GB combined) become usable for a single 14B or 32B training job. That's the qualitative jump that's invisible in candle today.

When Aether ships a feature listed here, mark it `[done]` and (if applicable) note the Aether commit / module that implements it.

When the matt-voice trainer encounters a NEW candle dependency not yet listed, **append it to this file with a citation** (which trainer module / config knob needed it and why).

---

## Current dependencies (as the candle qwen-lora-train binary is used)

### Quantized base model (QLoRA core)

- [ ] **GGUF reader** — single-file `gguf_file::Content::read` parity. Must parse Qwen2.5-7B-Instruct Q4_K_M (`/opt/matt-voice/models/qwen2.5-7b-q4km.gguf`). 28 layers, hidden=3584, heads=28, kv_heads=4. Multi-shard GGUF not required.
- [ ] **Q4_K_M dequantize** — the heavy lift. Pascal needs both an on-the-fly path (dequant during matmul) and a `dequantize_f16` specialized kernel (candle has both; lm_head uses pre-dequant to f16 to avoid transient OOM). Other quant types we may want: Q3_K_M, Q5_K_M, Q8_0, F16, BF16.
- [ ] **QMatMul** — quantized weight × fp16/bf16 activation matmul. Forward AND backward (the backward path was the candle-fork's main contribution; see [`QMatmulBwdOp` in candle-core/src/quantized/mod.rs:926`]).
- [ ] **HF tokenizer.json parser** — BPE / sentencepiece / tiktoken from a `tokenizer.json`. Qwen uses the same tokenizer across 0.5B → 72B. Reference path: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/.../tokenizer.json`.

### LoRA training primitives

- [ ] **Low-rank adapter modules** — `LoraLinear { lora_A: Tensor[r, in], lora_B: Tensor[out, r], scale: alpha/rank }`. Wraps a frozen base linear (or QMatMul).
- [ ] **Target-module dispatch** — by name string (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`). Attached at model-construction time per layer.
- [ ] **Trainable-only autograd** — frozen base weights produce no gradient; LoRA A/B accumulate.
- [ ] **PEFT-compatible safetensors export** — adapter weights keyed by HuggingFace PEFT naming convention so the result is loadable from Python tooling.
- [ ] **Adapter merge** — fold trained LoRA into the base for inference (`merge_adapters_into_base`).

### Memory-saving tricks (what makes 7B fit on small VRAM)

- [ ] **Gradient checkpointing at DecoderLayer boundary** — drops most activations, re-runs forward during backward. Candle's implementation has the "last layer must NOT be detached" gotcha; the fix is `Tensor::backward_into(&mut GradStore, Option<Tensor>)` for composable backward.
- [ ] **Fused softmax + analytical backward** — single fused kernel, detached y cache for backward.
- [ ] **Fused RMSNorm + analytical backward** — same shape, detached y cache.
- [ ] **(Rejected, do not implement)** Fused RoPE — candle measured -7% tok/s vs `rope_slow` once GC is enabled (cached y try_clone dominates the saving). Listed here for record; skip.
- [ ] `Tensor::backward_into` + `GradStore::remove_by_id` — composable backward primitives that enable the above.

### Distributed training (Phase 18 — the multi-GPU unlock)

This is the section that justifies the whole "swap candle for aether" framing.

- [ ] **FR-18.1** — Own NCCL bindings (M). Direct calls into libnccl; no Python wrapper. Gates everything below.
- [ ] **FR-18.2-extra** — Multi-rank wiring (today's collectives are single-rank passthroughs). Real `all_reduce` across ranks.
- [ ] **FR-18.4** — FSDP (L). Shard adapter optimizer state + adapter parameters across ranks. For QLoRA the frozen base stays whole on each rank; only the (tiny) adapter state needs sharding. **For matt-voice this is overkill** — the LoRA adapter is < 20 MB.
- [ ] **FR-18.5** — Tensor parallelism (Megatron-style) (L). Split each linear's weight matrix column-wise across ranks. For Qwen-7B on 2× P100: split attention heads + MLP intermediate dim across cards. **Most useful for matt-voice on current hardware** — keeps activations smaller per card.
- [ ] **FR-18.6** — Pipeline parallelism, 1F1B (L). Split decoder layers across ranks. For Qwen-7B (28 layers): rank 0 gets layers 0-13, rank 1 gets layers 14-27. **The realistic config for matt-voice on 2 P100s** — each card holds half the base weights. Unlocks 14B and 32B base models that don't fit on either card alone.
- [ ] **FR-18.7** — ZeRO-1/2/3 (L). Optimizer-state sharding, gradient sharding, parameter sharding. For QLoRA mostly unnecessary (only adapters are trainable).
- [ ] **FR-18.8** — Compute/comm overlap via CUDA streams (M). Hide all_reduce latency behind backward pass.
- [ ] **FR-18.9** — Gradient compression (PowerSGD-class) (M). Bandwidth saver on PCIe-link P100s. Useful if all_reduce shows up in profile.
- [ ] **FR-18.10** — Multi-host RDMA (skip — single-host cnc only).
- [ ] **FR-18.11** — 8-GPU Llama-7B training (skip — we have 2× P100, not 8).

### Optimizer

- [ ] **AdamW** — `lr`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`, `weight_decay=0.0` (frozen base means weight_decay irrelevant). LoRA defaults: `lr=2e-4`, no warmup, no schedule for the matt-voice baseline.
- [ ] **`clip_grad_norm_`** — standard global norm clip. Not currently used in matt-voice baseline but expected for stability at bigger configs.
- [ ] **LR scheduler** — linear warmup + cosine decay (optional; baseline is constant lr).
- [ ] **`optimizer.step()` + `optimizer.zero_grad()`** — same as antcolony.

### Forward pass — Qwen2.5 architecture

- [ ] **Qwen2 model graph** — RMSNorm → (Q,K,V proj → RoPE → causal attention → O proj) → residual → RMSNorm → SwiGLU MLP (gate_proj, up_proj, down_proj) → residual → repeat × num_layers → final RMSNorm → lm_head.
- [ ] **Grouped-query attention** — `kv_heads < num_heads`. Qwen2.5-7B has 28 heads, 4 kv-heads (factor 7 GQA). Needs proper KV-head broadcast.
- [ ] **RoPE precompute + apply** — `precompute_freqs_cis(head_dim, rope_freq, context_length, device)`. Use `rope_slow` (not fused, see above).
- [ ] **Causal mask** — standard upper-triangular.
- [ ] **Cross-entropy loss with label mask** — train only on the assistant-response tokens, not the user-prompt context. The matt-voice JSONL schema `{"context": ..., "matt": ...}` becomes a single sequence where only the `matt` portion contributes to loss.

### Dataset

- [ ] **JSONL streaming reader** — one `{"context": str, "matt": str}` per line. 46,407 pairs at `J:\matt-voice\training-data\matt-voice.jsonl` (14 MB).
- [ ] **Tokenize-on-load with label-mask construction** — context tokens get label=-100 (ignored in loss); response tokens get themselves shifted by one as label.
- [ ] **Right-truncate at `max_seq_len`** (128 baseline, 512 maxed). No padding-mask within batch since batch=1.

### Quantization-Aware Cross-Entropy (memory)

- [ ] **Tiled cross-entropy** — chunked CE for vocab-sized logits to avoid `[B, L, V]` peak of ~300 MB at Qwen vocab size. Currently disabled in candle (`--ce-chunk-size 0`); a known stall when > 0. Memory ceiling unlock if fixed.

### Checkpointing

- [ ] **`--save-every N`** — write `<output-dir>/checkpoints/step_N/adapter_model.safetensors + adapter_config.json` every N optimizer steps. Step counter persisted.
- [ ] **`--resume-from <ckpt-dir>`** — load adapter weights + step counter. Optimizer momentum/variance does NOT need to round-trip (candle's choice — adapter is the only thing that has to survive).

### Bench harness

- [ ] **`--benchmark`** — 3 warmup + 20 measured optimizer steps, single-line `BENCH {...}` JSON output with `tokens_per_sec`, `step_ms_median`, `step_ms_p95`, `peak_vram_mb`, `mean_gpu_util`. Reproducible measurement protocol — refuses to land an optimization that doesn't move at least one metric without regressing another. (Candle's bench had a multi-GPU `-i` bug that was fixed in the 2026-05-19 session; Aether's must `-i` from the start.)

### Persistence + tokenizer cache

- [ ] `safetensors` reader/writer for adapter state.
- [ ] HuggingFace cache path conventions (`~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<hash>/...`).

### CUDA backend (Pascal + Ampere matters here)

- [ ] **fp16 native** — sm_53+. Required on Pascal (P100 has no native bf16).
- [ ] **bf16 native** — sm_80+ (Ampere). Auto-detect via `CudaDevice::compute_cap()`; pick fp16 below 80, bf16 at 80+. Candle's qwen-lora-train hardcoded BF16 originally; the 2026-05-19 session added a `Device::supports_bf16_native()` probe + `--base-dtype auto`. Aether should ship the same auto-detect from day one.
- [ ] **sm_60 (Pascal) kernel coverage** — gate sm_70+ ops behind `__CUDA_ARCH__ >= 700`, gate sm_80+ ops behind `>= 800`, stub WMMA on sub-Volta. Otherwise builds silently lack kernels.
- [ ] **cublas gemm** + **on-the-fly dequant matmul** for Q4_K_M.
- [ ] **Device selection** via `CUDA_VISIBLE_DEVICES` env (don't bypass it).
- [ ] **gcc-13 host compiler** as default when CUDA toolkit rejects gcc-15 (Leap Micro / current SUSE rolling).

---

## Acceptance witnesses — what proves matt-voice is unblocked on Aether

These are concrete, single-command witnesses. Each one corresponds to an Aether milestone where matt-voice gets a real benefit.

| Witness | Aether milestones gated | Outcome |
|---|---|---|
| `aether-train --gguf qwen2.5-7b-q4km.gguf --dataset matt-voice.jsonl --rank 8 --target q_proj,v_proj --max-seq-len 128 --max-steps 100` finishes with loss declining and an adapter saved | Phase 17 (tensor stack), GGUF reader, QMatMul fwd+bwd, LoRA, AdamW, fp16 backend | matt-voice baseline runs on Aether at single-GPU. Direct candle-parity check. |
| Same command on the **16 GB P100** (sm_60), `nvidia-smi` shows ≥10 GB used and 99% util sustained | Pascal/fp16 path correctness (no silent CPU fallback) | matt-voice runs on the current cnc hardware via Aether. |
| `aether-train ... --distributed world_size=2 --algorithm pp --max-seq-len 512` finishes with loss declining on **both P100s** at once | FR-18.1 (NCCL), FR-18.2-extra (multi-rank), FR-18.6 (PP/1F1B), Phase 18 in general | **The unlock.** First multi-GPU matt-voice training job. Both cards working a single step. |
| `aether-train --gguf qwen2.5-14b.gguf ... --distributed world_size=2 --algorithm pp` reaches step 100 without OOM | Above + GGUF Q4_K_M for 14B + activation memory budget | 14B at full quality becomes accessible. Currently impossible on either single P100. |
| `aether-train --gguf qwen2.5-32b.gguf ... --distributed world_size=2 --algorithm pp` reaches step 100 without OOM | Above + careful per-stage memory split | 32B becomes accessible — the size where matt-voice meaningfully outperforms anything an open API offers. |

The first two witnesses are the immediate goal post-Phase 17. The third is the Phase 18 milestone. The fourth and fifth are the qualitative justification for the whole exercise.

### Bigger-model target: Qwen3-32B dense (verified on cnc 2026-05-24 eve)

Confirmed by Matt; verified by inspecting cnc, NOT taken on faith:
- **Base GGUF staged + complete**: `cnc:/opt/matt-voice/models/Qwen3-32B-Q4_K_M.gguf` = **19 GB** (download finished 17:55; `dl.log` shows the curl run completing). Tokenizer/config at `cnc:/opt/matt-voice/models/qwen/Qwen-Qwen3-32B/`.
- **Arch (from that dir's config.json)**: 64 hidden layers, hidden 5120, 64 q / 8 kv heads (GQA), head_dim 128, intermediate 25600, vocab 151936, rope_theta 1e6, **no sliding window**, `tie_word_embeddings=false`. 64 layers → clean **32/32 pipeline split**.
- **Why this is the aether driver (not just a candle job)**: the 7B LoRA flow is already DONE via candle single-GPU (`matt-voice-7b-{v1,v2-r64,longform}.lora.gguf` exist on cnc). **Qwen3-32B Q4_K_M (19 GB) does NOT fit either P100** (cnc GPUs are 12 GB + 16 GB), so it CANNOT be trained single-GPU and DP-replication doesn't help. It requires **model-splitting training** — that is the capability only aether is being built to provide.
- **cnc GPU state at check time**: GPU0 = 12 GB (10.5 used by the workhorse, 1.6 free), GPU1 = 16 GB free. A real 32B PP run needs the workhorse stopped to free GPU0 (coordinate via openclaw `main`, B-approval).

**Gate (aether-side, to be built):** (1) **FR-18.6-real pipeline parallelism (1F1B)** to span the 64 layers across the two P100s; (2) **qwen3 GPU forward+backward in the trainer** (`trainer/model.rs` is a CPU reference net today) — reuse the verified `QwenSession` inference forward as the base; (3) **QLoRA against the quantized base**: forward delta-add of LoRA adapters + backprop through the frozen quantized linears. Pieces (3) partially landed 2026-05-24 eve: LoRA adapter math + DP adapter all-reduce (`trainer/src/lora.rs`, `lora_dp.rs`, CPU finite-diff verified) and the QMatMul-backward kernel `aether_op_quant_matmul_backward_lhs_f32_cuda` (`dx = Wᵀ·dy` through frozen Q4_K/Q6_K, GPU parity 3e-8 vs CPU). The big remaining build is PP/1F1B.

### STRETCH target: Llama-3.3-70B (fit analysis, 2026-05-24 eve)

Matt wants 70B "if we can fit it." For QLoRA the frozen quantized base dominates VRAM (adapters/grads/AdamW are tiny; activations bounded by checkpointing) → it's base-size vs pool-VRAM. Pool: cnc 2×P100 = 12+16 = **28 GB**; + kokonoe 3070 Ti 8 GB + satibook 3050 (~6–8 GB) ≈ **~42 GB total**, ~36 GB usable after CUDA context + activation working set. 70B GGUF sizes:

| Quant | ~Size | cnc-only 28 GB | full pool ~42 GB | aether kernel |
|---|---|---|---|---|
| Q4_K_M | 42.5 GB | ❌ | ❌ | ✅ |
| Q3_K_M | 34.6 GB | ❌ | ⚠️ tight | ✅ |
| **IQ3_XXS** | **27.5 GB** | ❌ | ✅ ~14 GB headroom | ✅ fwd shipped |
| Q2_K | 26.4 GB | ⚠️ ~2 GB | ✅ | ❌ |
| IQ2_XS | 21 GB | ✅ ~7 GB | ✅ | ❌ |
| IQ1_M | 16.8 GB | ✅ | ✅ | ❌ |

**Verdict: 70B fits only at low-bit, not Q4_K.** Two targets (Matt: "both"):
1. **IQ3_XXS (~27.5 GB), full-pool PP** — reuses the shipped IQ3_XXS forward; needs the QMatMul-backward extended to the IQ3_XXS dtype (one arm; its GPU dequant exists). **Recommended first** (least new code).
2. **IQ2_XS (~21 GB), cnc-only PP** — simpler topology (no cross-host bubble) but needs BRAND-NEW IQ2 quant kernels (forward + dequant + backward). Larger build.

The PP/1F1B machinery is identical for 32B/70B — build PP once, prove on 32B (fits cnc easily), scale to 70B by swapping model + quant. Caveat: P100s are Pascal (~9 TFLOPS fp32, no fast fp16) + cross-host PP bubble → 70B QLoRA will *fit* but be *slow*. No 70B GGUF staged yet (cnc /opt has 482 GB free; Llama-3.3-70B chat template already present).

---

## Aether-equivalent already in place (existing capability)

(Move items here as Aether features land. Format: `- [done] <feature> — `<aether commit / module>` (date)`.)

- (nothing yet — Phase 17 at 55%, Phase 18 at 9% per 2026-05-09 audit; the matt-voice path waits on both)

---

## Notes on the bigger picture

Aether's pitch for matt-voice is the same as for any LLM-training workload: a from-scratch language that emits a single static binary, with `#[distributed]` as a language attribute rather than a framework. Three concrete wins over candle if Aether matures:

1. **Multi-GPU is one attribute, not a fork-and-rewrite.** Today matt-voice on two P100s = "write pipeline-parallel code in the candle fork over a weekend". On Aether it's `#[distributed(world_size=2, algorithm="pp")]` on the train function. This is the killer feature.
2. **No CUDA toolchain dependency at run time.** Candle on Windows needs MSVC `cl.exe` because nvcc on Windows mandates an MSVC host compiler. The matt-voice candle work happened on Linux (cnc) to dodge this; the kokonoe Windows build needed VS BuildTools + careful PATH. If Aether's CUDA codegen is self-contained (own PTX emitter / cuBLAS calls only / no nvcc), this whole class of pain goes away. (See the parallel point in `ANTCOLONY_FR.md`.)
3. **First-class autodiff with explicit tape.** Easier to reason about than candle's `apply_op1` + `CustomOp1::bwd` indirection — and `--strip-comments` means the production binary has zero documentation overhead.

For the matt-voice training itself, candle is fine **as long as it's single-GPU**. The push to Aether is justified by the need for multi-GPU and (eventually) bigger base models. Until Phase 18 lands, this list is a roadmap, not a blocker.

---

## Per-commit log of what was added to this file

(Append here as the matt-voice training adds new requirements.)

- 2026-05-19 — Initial list seeded by Claude after the first real P100 training run launched on cnc. Items reflect the candle features actually being used by the in-progress 7B QLoRA run (`/opt/matt-voice/train-7b-p100.sh`, PID 944096, 1000-step run started 08:23 UTC). Distributed section pulled directly from `NEXT-UP.md` FR-18.* entries so cross-reference stays clean.
