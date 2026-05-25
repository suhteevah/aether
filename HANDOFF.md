# Aether — Session Handoff

## Last Updated — 2026-05-25 (V2-Lite bisect — 3 fixes shipped, GLM coherent, V2-Lite isolated to YaRN path)
🟢 Big session on cnc P100s (standing GPU access granted). 4 commits pushed
(`84ef3a5`, `ffaf9c7`, `37d5050`, + this YaRN-diag `f31a4c0`). Each rebuilt on
cnc GPU1 (workhorse evicted+restored every time, all reversible, workhorse
active now).

**HEADLINE WINS:**
- **GLM-4.7-flash now CLEARLY coherent** ("The capital of France is Paris.",
  valid Python) — was only loosely token-witnessed before. The MoE gating fix
  (sigmoid + scale 1.8 + norm, all read from GGUF) did it.
- **3 real bugs fixed + verified**: (1) YaRN ramp `84ef3a5`; (2) MoE gating
  `ffaf9c7` (read expert_gating_func/weights_norm/weights_scale; softmax|sigmoid
  over all experts; +GgufMeta::Bool; pure unit tests); (3) deepseek2 norm
  default=false `37d5050`.

**V2-Lite STILL incoherent** after all three (degenerate-repetitive). Bug is now
TIGHTLY ISOLATED:
- GLM (absorbed MLA, NON-yarn rope) coherent → MoE + embed + norms + lm_head +
  o_proj + paged_attention_mla + the absorbed path all CORRECT.
- `cuda_mla_e2e_synthetic` (non-absorbed MLA step, NON-yarn, hand-CPU ref) PASSES
  → the non-absorbed assembly kernels (assemble_k/extract_v/split/kv_b) + attn
  core are correct at short positions with plain rope.
- **The ONLY V2-Lite-unique code exercised by NOTHING else = the YaRN rope path**
  (`mla_rope_q_partial_yarn` / `mla_rope_k_shared_yarn`). GLM has yarn INACTIVE
  (rope=1e6, no YaRN log line); the synthetic test uses plain rope. So yarn is
  unwitnessed-in-integration. My ramp fix passes `cuda_yarn_rope_parity`, but
  that test is the ONLY yarn check and can't catch a llama.cpp divergence my own
  reference shares.

**4th fix `7e24763` (done): llama.cpp deepseek2 YaRN scaling matched VERBATIM**
(fetched src/models/deepseek2.cpp + ggml rope_yarn). rope cos/sin mscale=1.3689
(was 1.0); attention kq_scale mscale=1.4046 → 0.1424 (was HF-style 0.1147). Aether
had matched HF, not the GGUF's llama.cpp runtime. yarn parity 4/4.

**PROMPT_IDS BYPASS REFRAME (key):** clean ids `[100000,549,6077,280,7239,317]`
greedy → `" is is is is is is is is is the the the"`. REAL frequent words in a
repetition loop, NOT char-garbage. So after 4 fixes V2-Lite went random "eu)" →
"\n\n\n" → word-repetition. TWO residual problems, now separated:
 (A) FORWARD: flat-attention repetition even on clean ids (real tokens, can't
     progress past "is"). Closer but not right.
 (B) TOKENIZER/TEMPLATE: chat path is WORSE than prompt_ids (adds char-garbage)
     — a SEPARATE bug from the forward.

**Phase 4.5 reached — 4 fixes, partial convergence. STOP incremental tweaks.**
DEFINITIVE next step (NOT yet done — fresh effort, GPU access granted):
1. **llama.cpp greedy + intermediate reference on the SAME 6 ids.** b8182
   `llama-cli -no-cnv` SEGFAULTS on CPU → try `llama-completion` or `llama-server`
   (likely needs GPU; evict workhorse). If llama.cpp → "Paris", dump its
   position-0 logits + per-layer attn and diff vs Aether (`AETHER_DUMP_MLA=1`,
   `AETHER_DUMP_BLOCKS=1`) to localize the flat-attention bug. This is the only
   way left — formula-derivation from references is exhausted.
2. NEOX-vs-NORM rope: deepseek2 is NEOX in llama.cpp (Aether matches) — checked,
   not the bug. But the deepseek q_pe/k_pe weight-layout permute (HF interleave
   vs GGUF) is unverified; the synthetic test is tautological on rope pairing.
3. Separately debug the tokenizer/template (B) — compare aether encode_text +
   deepseek chat template vs llama-tokenize on the rendered prompt.
4. Re-smoke qwen3moe (insurance; gating default unchanged for it).

**Honest status: 4 real fixes shipped (GLM now coherent — big win), V2-Lite
improved but NOT solved. Do not claim V2-Lite done.**

**mscale AUDIT (done):** attention softmax mscale²=1.5896 confirmed correct via
load log; HF cos/sin _mscale=1.0 no-op for V2-Lite. Diagnostic log added
(`[QwenSession] YaRN:` + `MoE:`).

Original entry that kicked this off:
🟢 Picked up handoff next-item #1 (V2-Lite MLA-non-absorbed forward bisect).
Found the YaRN ramp bug by line-by-line comparison vs the HF/llama.cpp reference.

**The bug** (`runtime/src/cuda.rs` `yarn_scale_factor`): two compounding errors —
(1) correction-dim bounds named backwards, so `denom = max(i_high - i_low, 1e-3)`
clamped a negative diff to 1e-3 and collapsed the smooth interpolation ramp into
a misplaced step; (2) call sites passed `2*i` (full-dim index) instead of the
rotary PAIR index `i`. Net: ~10 of 32 rope freq pairs got wrong scaling (9–50°
rotation error even at short positions). **V2-Lite is the ONLY model on the YaRN
path** — GLM-absorbed + qwen2.5 don't hit it → explains why it was V2-Lite-only.
[[v2_lite_yarn_ramp_bug]]

**Why it survived:** `cuda_yarn_rope_parity.rs` was tautological (CPU "reference"
== kernel's buggy formula). Rewrote with an independent HF-math reference +
`yarn_scale_factor_reference_anchors` (hand-computed: scale(0)=1.0, scale(16)=0.55,
scale(23)=0.025) so it can't silently re-encode the bug.

**Verified (kokonoe 3070 Ti):** 4/4 cuda_yarn_rope_parity (anchors + GPU parity
1e-7 + factor=1 identity); qwen25_paged_parity unregressed (standard rope path
untouched).

**NOT yet confirmed:** actual V2-Lite *coherence*. The fix is in the nvrtc kernel
SOURCE STRING, so the cnc binary must be REBUILT (`strings` won't show it). The
decisive smoke needs a cnc GPU slot — scripted at `scratch/v2_lite_yarn_smoke.sh`
(evict workhorse, V2-Lite Q4_K_M, --paged, chat-completion request → expect
coherent text, not "eu)"). The pre-fix "incoherent" run may ALSO have predated
the deepseek2 chat-template split, so the smoke is the real arbiter.

**Re-smoke GLM too (insurance):** the absorbed path (`serving.rs:1587,1605`) also
calls these yarn kernels when `yarn_active`. Fix is safe by reference-equivalence
(now matches llama.cpp, which serves GLM coherently), but re-run the GLM witness
after the cnc rebuild to be sure.

**mscale — AUDITED, correct for V2-Lite (commit `f31a4c0`, diagnostic only):**
chased the "missing rope cos/sin mscale?" question against HF DeepSeek-V2.
(1) Attention softmax scale: Aether's `base_scale*mscale²` with
`mscale = 1 + 0.0707*ln(40) = 1.2608` (mscale²=1.589) EXACTLY matches HF.
(2) cos/sin `_mscale` = `yarn_get_mscale(s,mscale)/yarn_get_mscale(s,mscale_all_dim)`
= 1.0 for V2-Lite because `mscale == mscale_all_dim == 0.707` — a genuine no-op,
correctly omitted. (3) Only residual risk: the GGUF `yarn_log_multiplier` key
not being read → mscale silently 1.0. Added a YaRN dump at session load
(`[QwenSession] YaRN: factor=.. log_mul=.. -> mscale=.. mscale^2=..`) so the
smoke output proves the read. **No mscale code change was warranted.**

**Next:** (1) push + cnc V2-Lite coherence smoke (+ GLM re-smoke) — CHECK the new
`[QwenSession] YaRN:` log shows `log_mul=0.0707 mscale^2≈1.589` (if it shows
`log_mul=0`, that IS the bug — trivial GGUF-key fix); (2) qwen3moe forward bisect
(still open from prior handoff); (3) the deferred 8h matt-voice run.

---

## Last Updated — 2026-05-25 (per-arch session B — gemma3 CLOSED end-to-end)
🟢 Picked up handoff items (a)+(b)+(d), then drove gemma3 all the way to
coherent text serving. 4 commits (e5da06a, dca1510, 39bc261, d820cc8), all
pushed to origin/main; cnc /opt/aether fast-forwarded + rebuilt each step.

**Headline: gemma-3-12b serves coherent text end-to-end on cnc.**
- "List three primary colors, separated by commas." → **"Red, Yellow, Blue"**
  (prompt_ids byte-identical to `llama-tokenize`: 18 ids).
- "What is the capital of France?" → **"The capital of France is Paris."**
Two bugs, both fixed:
1. **Forward** — gemma3 per-layer local/global attention + dual RoPE base
   (5 local rope=10k/win=1024 : 1 global rope=1M/full, `il%6==5`). Validated
   first via `prompt_ids` (greedy → "Red,Yellow,Blue"). logit-softcap +
   query_pre_attn_scalar were RED HERRINGS (gemma2-only / no-op for 12b);
   RMSNorm `(1+w)` is baked into the GGUF — plain rms_norm stays correct.
   [[gemma3_dual_rope_per_layer_swa]]
2. **Tokenizer** — gemma is SentencePiece (`tokenizer.ggml.model=="llama"`),
   aether had GPT-2 byte-BPE only. Built an SPM encoder. KEY: SPM is NOT
   Unigram-Viterbi (Viterbi over-segments — gemma scores reward short pieces:
   `user`=-1870 < `us`-111+`er`-3); llama.cpp uses iterative **bigram-merge**.
   Reimplemented that way → byte-exact with llama-tokenize. Plus `▁`→space
   detok + BOS prepend + f32-array (scores) GGUF getter.
   [[spm_bigram_merge_tokenizer]]

**Also shipped (the original a/b/d):**
- **(b)** deepseek2 chat-template split — GLM (absorbed) vs DeepSeek-Coder-V2
  (non-absorbed) via `is_mla_absorbed()`. V2-Lite no longer gets GLM markers
  (prompt_tokens 15, correct). 6 unit tests. [[deepseek2_two_chat_templates]]
- **(d)** Q6_K (dt=14) MoE expert matmul kernel + dispatch + 2 GPU parity
  tests. qwen3moe smoke ran experts with zero dtype panic.

**Per-arch ledger now:**
- ✅ witness-ready: qwen2.5, GLM, BGE, IQ3_M-class, **gemma3 (NEW — coherent text)**
- 🟡 forward still incoherent (separate, NOT started): **V2-Lite** (template now
  correct → blocker is the MLA-non-absorbed FORWARD) + **qwen3moe** (Q6_K no
  panic, ChatML correct → forward rambles). Both need an `AETHER_DUMP_BLOCKS`/
  `AETHER_DUMP_LOGITS` NaN-bisect of their forwards — open-ended GPU debugging.

**cnc state:** all smokes ran on GPU1 with the workhorse evicted+restored
(watchdog + explicit restart each time); workhorse + scout both active now.
Local: workspace builds clean (bare + cuda); audit 0 errors; new unit tests
(6 template + 3 gemma3-layer + 4 SPM + 2 Q6_K-parity) all green.
`AETHER_DUMP_TOK=1` dumps final prompt ids. `llama-tokenize` at
`cnc:/opt/llama/llama-b8182/` is the tokenizer ground-truth.

**Next:** (1) V2-Lite MLA-non-absorbed forward bisect; (2) qwen3moe forward
bisect; (3) wire SPM-arch detection into the per-model chat path if any other
SPM model is hosted; (4) the deferred 8h matt-voice run.

---

## Last Updated — 2026-05-25 (session end, prior block)
🟢 Two big threads landed; both have clean carry-forward. (Older dated sections
below are chronological; this block is the canonical current state.)

**1. matt-voice REAL 32B QLoRA pipeline — VALIDATED, 8h run deferred.**
Full path built: embed real tokens → 64 Qwen3-32B layers split across 2×P100 →
real Q6_K lm_head → next-token CE → adapter backprop. 20-step real-data run
(matt-voice ChatML corpus, 2.38M tokens) loss **4.485 → 1.732**, no OOM, clean
exit. Launch-ready; operator deferred the 8h run (slot lapsed). Recipe: `pp-qlora-worker
--data <ids.bin> --t 32 --splits 26,38 --microbatch 1 --lora-rank 16 --alpha 32
--lr 2e-4 --save <p> --ckpt-every 200`; stop all 3 cnc services first (re-grant
(B) via openclaw main per-day). Full detail: memory `matt_voice_real_training_validated`.

**2. Per-arch round — shared vocab-1 root cause FOUND + fixed.**
A free GGUF dtype probe (no bisect) found `dequant_embd_row` hardcoded Q4_K while
`token_embd` is Q6_K(gemma3)/Q3_K(qwen3moe)/IQ3_S(IQ3_M) → garbage embeddings →
the vocab-1 NaN-pin across all three. Fix (`b5397fa`): dtype-aware embedding dequant
+ host `aether_dequant_q3_k`/`aether_dequant_iq3_s`. Payoff: **IQ3_M fully coherent**,
**qwen3moe forward fixed** (real tokens), **gemma3 narrowed** (no longer pad-pinned).
Also shipped: Q3_K upload arm, Q3_K+Q5_K MoE expert kernels, gemma3 tied-embed
lm_head + embed scale, aether-serve port-collision fix (:8081→:18913).

**Per-arch ledger:**
- witness-ready: qwen2.5, GLM (MLA-absorbed), BGE, **IQ3_M-class** (R1-Distill coherent)
- forward-fixed, needs chat-template wiring in serving: qwen3moe, V2-Lite
- parked (narrowed to ~3 gemma-specifics): gemma3 forward — kokonoe code, no eviction to find

**What's next (prioritized):** (a) gemma3 gemma-specifics bisect (logit softcap /
query_pre_attn_scalar / sliding-window) — kokonoe code; (b) per-model chat-template
wiring in serving.rs to flip qwen3moe+V2-Lite to witness-ready; (c) the 8h matt-voice
run when scheduled; (d) qwen3moe expert mix may need more dtypes (Q6_K dt=14) — chase
if a re-smoke surfaces it. Build: `cargo build --workspace --release` clean;
`--features cuda` clean. Fleet fully restored, all evictions reversible.

---

## 2026-05-25 — PER-ARCH VERIFICATION (live cnc smokes, NOT "fully ready")
Ran real decode/embedding smokes against the deployed `aether-serve` binary on
cnc P100s (workhorse evicted from GPU1 for the window, restored after; no
training was running). Methodology + full notes in memory
`reference_aether_per_arch_verification.md` + driver `cnc:/tmp/aether-perarch-smoke.sh`.

| Arch | Model (quant) | Result |
|---|---|---|
| `qwen2`/2.5 | Qwen2.5-Math-7B Q4_K_M | ✅ coherent |
| `deepseek2` MLA-**absorbed** (GLM) | glm-4.7-flash IQ3_XXS | ✅ coherent |
| `bert`/bge | bge-large-en-v1.5 f16 | ✅ 1024-dim, L2=1.0000, finite (`/v1/embeddings`, `--bge-gguf`) |
| `deepseek2` MLA non-absorbed (V2-Lite) | DeepSeek-Coder-V2-Lite Q4_K_M | 🟡 decodes real tokens, incoherent — retest w/ chat template |
| `llama` | codestral-22b **IQ3_M** | ❌ NaN-pins to vocab-1 (32767×); forward broken for this quant + text-encode gap |
| `gemma3` | gemma-3-12b Q4_K_M | ❌ panic `output.weight not found` (tied embeddings; need token_embd lm_head fallback @ serving.rs:2148) |
| `qwen3moe` | Qwen3-30B-A3B **Q3_K_M** | ❌ panic `unsupported dtype 11` — Q3_K missing from `upload_tensor_u8`/`_opt` (761/843). **~2-line fix**: `11 => { let nb=n_elems/256; (nb, nb*110) }`. Kernel+dispatch already present. |
| `qwen3` dense | Qwen3-32B Q4_K_M (19G) | ⛔ exceeds single P100; only 2×P100 PP training exercises it |

**aether-serve.service crashloop FIXED:** unit had `--port 8081` (collides with
workhorse llama-server) → 66 restarts of load-warm-die. Changed to `--port 18913`;
NRestarts=0, listening, decode verified. Runs alongside workhorse on GPU0.

### What's next (prioritized fixes from this verification)
1. **qwen3moe — TRIVIAL (~2 lines):** add Q3_K (dt=11) arm to `upload_tensor_u8`
   AND `upload_tensor_u8_opt` (serving.rs:761/843):
   `11 => { let nb = n_elems / 256; (nb, nb * 110) }` (Q3_K super-block = 110B/256
   elems). The matmul kernel (`fused_q3_k_matmul_seq1`) + `dispatch_matmul` dt=11
   arm already exist — only the upload helpers reject it. Rebuild aether-serve on
   cnc (`RUSTC_WRAPPER= cargo build --release --features cuda`, ~35s) + re-smoke
   Qwen3-30B-A3B-Q3_K_M. Unblocks the entire qwen3moe arch.
2. **V2-Lite coherence — retest, probably not a bug:** DeepSeek-Coder-V2-Lite
   decoded real non-NaN tokens but incoherent output ("\n2)Pi_ -Ca"). Re-smoke with
   the model's chat template applied (or known-good prompt_ids). Confirm before
   filing as an MLA-non-absorbed forward bug.
3. **gemma3 — MEDIUM (~15-30 LOC):** lm_head loader at serving.rs:2148 panics on
   missing `output.weight`. Gemma3 ties embeddings → fall back to `token_embd.weight`
   as lm_head when `output.weight` is absent. Then re-smoke gemma-3-12b.
4. **codestral/llama IQ3_M NaN — HARD/open-ended:** forward emits NaN → vocab-1
   pinning. Bisect like the GLM NaN chase (AETHER_DUMP_BLOCKS etc.). Likely
   IQ3_M-quant-specific; "llama" arch may be fine on a Q4_K model — verify with a
   Q4_K llama GGUF before assuming the arch path is broken.
5. **qwen3 dense (32B) single-card decode** stays HW-blocked unless a smaller qwen3
   GGUF is staged; the 2×P100 PP training path already exercises the arch.

## Last Updated
2026-05-25 (**matt-voice REAL 32B QLoRA training pipeline VALIDATED end-to-end on
2×P100. Built the real path (embed real tokens -> 64 qwen3 layers split 26/38 ->
real Q6_K lm_head -> next-token CE -> adapter backprop), tokenized the matt-voice
corpus (28513 ChatML examples, 2.38M tokens, Qwen3 tokenizer), and confirmed a
20-step real-data run: loss 4.485→1.732, adapter |B| 0→37440, no OOM, clean exit.
The actual 8h fine-tune DEFERRED (slot lapsed to next day) — pipeline + working
config launch-ready. All 3 cnc services restarted + health-verified.**

### matt-voice real-training: how to launch the deferred 8h run
- Binary: `trainer/src/bin/pp_qlora_worker.rs` `--data <ids.bin>` mode. Stage:
  `trainer/src/qwen_qlora_stage.rs` (embed_tokens / load_lm_head / lm_head_loss /
  save_adapters). Corpus already tokenized at
  `cnc:/opt/matt-voice/training-data/matt-voice-longform.ids.bin`.
- **Working memory config (DO NOT exceed):** `--t 32 --splits 26,38 --microbatch 1
  --lora-rank 16 --alpha 32 --lr 2e-4`. GPU0 (12GB) holds 26 layers; GPU1 (16GB)
  holds 38 + the **chunked** Q6_K lm_head (16 vocab-chunks, ~190MB transient — the
  whole-weight 3.1GB dequant OOM'd). T=64 OOM'd GPU0 on step 1 (fragmentation at
  the edge); T=32 has headroom.
- **cnc coordination:** stop ALL THREE services first (openclaw main re-grants (B)
  per-day): `openclaw-inference-scout` + `aether-serve` (GPU0) + `openclaw-inference-
  workhorse` (GPU1). Relaunch: `systemctl start` all three. The workhorse on GPU1
  is easy to miss — it came up after main's "GPU1 free" read and OOM'd rank1.
- **Tuning for quality before the long run:** the 20-step loss was noisy
  (4.5→7.4→…→1.7) from microbatch=1 single-sample gradients. Add grad accumulation
  (microbatch≥4) or lower lr (~1e-4) for a smoother real run. Adapter checkpoints
  via `--save <path> --ckpt-every 200` (writes `<path>.rank{0,1}.step{N}`); a
  .lora.gguf merge/export tool is still TODO.

## (2026-05-24 LATE) PERF: decode tok/s recovery via nvrtc-unit split. Measured
Aether was already ~108%% of llama.cpp on the 3070 Ti (never behind — the "1:1"
premise was off); the lost ground was Aether's own 37.2 peak regression
([[nvrtc_kernel_unit_pressure]]). Split KERNEL_SRC → lazy TRAIN_KERNEL_SRC in 2
batches (18 training/backward kernels moved out of the decode unit): decode
32.6 → ~35.0 tok/s (+7.4%%, ~117%% of llama.cpp). Inference bit-intact, all
grad-checks green. Commits 72f41e3 + 3af8d9f, BENCH_LEDGER rows appended.
REMAINING to 37.2: inference-AMBIGUOUS kernels (gelu_fwd, layer_norm_fwd,
add_layer_norm_fwd, bert_*, gqa_repeat/reduce) — need a BERT-embedding smoke to
confirm decode-unused before moving. Decode bench now warm+configurable
(16-step warmup, MAX_SEQ 128).**)

## (prior) Last Updated
2026-05-24 (**FR-18.6-real LEG 3 FINISHED — the matt-voice Qwen3-32B unlock is
DONE. Full 64-layer Qwen3-32B-Q4_K_M QLoRA-trained across 2×P100 on cnc via
Aether pipeline parallelism: both ranks loaded (28/36 split, no OOM), both GPUs
100%% alternating through the pipe, 8 steps loss 5.70→3.09, 67M LoRA params live
across both stages, exit 0/0, clean trap restart of GPU-0 services. All 3 legs
of FR-18.6-real now closed. ~12 commits this session.**)

## Project Status — FR-18.6-real ALL THREE LEGS CLOSED
🟢 **Leg 3 done.** The full path is proven on real hardware:
- **mmap GGUF loader** (`6a2b94b`) — `aether_gguf_open` now mmaps (was a whole-file
  `std::fs::read` that OOM-killed the 19 GB load on cnc's 15 GB box). `GgufBlob`
  enum, unix mmap / owned-Vec fallback, Derefs to `[u8]` (all sites unchanged).
- **QwenQLoraStage** (`efe557a`) — frozen quant base + trainable LoRA adapters,
  full-seq fwd/bwd, GQA, Qwen3 per-head Q/K norm. Validated on real 7B AND 32B.
- **pp-qlora-worker** (`f8...`/`--splits`) — N-way PP QLoRA worker; uneven split
  (28 on the 12 GB P100, 36 on the 16 GB).
- **Witnessed on cnc 2×P100**: full 64-layer Qwen3-32B, 28/36 split, both GPUs
  100%, loss 5.70→3.09 (synthetic-objective smoke — noisy by design), adapter |B|
  0→46k/59k both stages, exit 0/0. Every cnc cycle reversible (restart trap +
  openclaw-main (B) coordination; ~8 min blackout, services verified restored).

**What a REAL matt-voice fine-tune needs beyond this smoke** (machinery is all
proven): swap the synthetic loss head for the GGUF lm_head + real tokenized
matt-voice corpus + a DataLoader; tune lr/steps/microbatch (the smoke loss is
noisy from microbatch=1 + random targets); save the trained adapters to a
.lora.gguf. None of that is new machinery — it's data + a save path.

## (prior) Project Status — leg 2 CLOSED
🟢 The 4 leg-2 finishers the prior handoff listed as NEXT are all shipped +
finite-diff/loss-curve witnessed:

| # | Finisher | Commit | Witness |
|---|---|---|---|
| 1 | **GQA** repeat(fwd)/reduce(bwd) | `ae334c3` | new `gqa_reduce_kv_grad` kernel; GQA block (n_q=4,n_kv=2) grad-check max rel 2.35e-2 |
| 2 | **LM-loss** wrapper (embed+lm_head+CE) | `663ae30` | new `embed_lookup`/`embed_scatter_add`; full LM loss=3.045, all 12 tensors max rel 2.04e-3 |
| 4 | **GPU pipeline Stage** | `33b4ae0` | `QwenBlockStage` trains through `run_1f1b`, loss 2.91→1.72, grad-accum across microbatches |
| 3 | **QLoRA** proj (frozen base + adapter) | `b239f5d` | adapter dA/dB finite-diff max rel 7.7e-4 |

Build: `cargo build --workspace --release` clean (non-cuda); all 5 new tests
green under `--features cuda` on the 3070 Ti. MHA capstone + qwen25_paged_parity
unregressed.

**Leg 3 — PP machinery DONE + witnessed on target hardware (`9c1704b`):**
`pp-qwen-worker` (trainer/src/bin/pp_qwen_worker.rs) runs one rank as its own
PROCESS (the runtime CudaCtx is a process-global singleton → ranks can't be
threads; that's why the in-process witness was world_size=1). `connect_pipeline`
TCP rendezvous + run_1f1b drive a layer-split qwen3 block stack.
`scripts/pp_2rank_witness.sh` asserts the 2-process pipelined run is BIT-IDENTICAL
to the single-process reference (loss 4.369193→0.344829; rank paramsums add to the
1-proc total; rank0 warmup=1 = real 1F1B overlap). **Verified on BOTH the RTX
3070 Ti (kokonoe) AND a cnc Tesla P100** (built on cnc EXIT_0, ran on the free
GPU1, zero service disruption). Because PP transport is TCP with host staging
(d2h→TCP→h2d), 2-procs-1-GPU and 2-procs-2-GPU exercise the *identical* code path
— so the physical-GPU-split path is already covered.

**Leg 3 — what REMAINS for a literal Qwen3-32B matt-voice run (NOT yet built):**
1. **GGUF-quant + QLoRA stage**: `QwenBlockStage` today loads f32 RANDOM weights.
   For 32B, build a variant that loads real GGUF quant tensors into device
   buffers, swaps the f32 base matmuls for the proven quant kernels
   (`fused_*_matmul_seq1` fwd + `quant_matmul_backward_lhs` for dx through the
   frozen base), and injects the QLoRA adapter (math proven in finisher #3,
   `b239f5d`) after each q/k/v/o/gate/up/down proj. GQA repeat/reduce (#1)
   handles n_kv<n_q. This is the real fine-tune stage — a substantial build,
   validate against the 19 GB GGUF on cnc.
2. **Free a 2nd full P100**: 32B per-side needs a dedicated card. cnc GPU1 is
   currently free (16 GB); GPU0 has matt-voice (~10.5/12 GB). The 64-layer
   32/32 split across 2 cards needs the second card freed — coordinate the
   workhorse/GPU0 tenants via openclaw main (B-approval).
- Multi-layer-per-stage already supported (`QwenBlockStage::build` takes a layer
  range); witnesses used 4 layers split 2+2.

---

## (prior) Last Updated
2026-05-24 (**FR-18.6-real PP push — leg 1 (1F1B machinery) DONE+witnessed;
leg 2 DONE: all qwen3-block kernels (rms/rope/sdpa fwd+bwd, silu bwd,
transpose) + CAPSTONE composition grad-check (qwen3 block fwd+bwd, finite-diff
verified). 11 commits this session, all on RTX 3070 Ti, ALL COMMITTED.
Plus Mistral Small 24B explicit-head_dim support (side quest).**)

## Project Status
🟢 Pipeline parallelism (the matt-voice Qwen3-32B unlock) decomposed into 3
legs (see [[fr_18_6_pp_three_legs]] memory). **Leg 1 + leg-2 backward kernels
shipped this session:**

| Commit | What | Witness |
|---|---|---|
| `5a3342b` | FR-17.14 IQ3_XXS standalone dequant for QLoRA backward (dt=18) | parity 3e-5 vs CPU; 70B base-quant target |
| `efc928f` | **Leg 1**: 1F1B PP machinery `trainer/src/pipeline.rs` (connect_pipeline TCP rendezvous + Stage trait + run_1f1b) | 2-rank localhost-TCP, param max\|diff\|=0.000e0 vs single-process ref |
| `dd4a257` | **Leg 2a**: RMSNorm backward CUDA (rms_norm_bwd_dx/gamma) | parity 1.19e-7 / 2.38e-7 |
| `1de3c6b` | **Leg 2b**: RoPE backward CUDA (rope_apply_backward) | parity 1.19e-7 + round-trip identity |
| `0e0bd5f` | **Leg 2c**: causal SDPA backward CUDA (sdpa_causal_bwd_dq/dkv) | parity 3e-8/6e-8/2e-7 vs CPU |
| `332bda3` | **Leg 2d**: full-seq causal SDPA FORWARD (emits [bh,s,s] attn probs training needs) | parity 1.19e-7/5.96e-8 |
| `1fe33be` | **Leg 2e**: SiLU backward (SwiGLU) | parity 1.19e-7 |
| `0b29652` | **Leg 2f + CAPSTONE**: transpose_021 + **qwen3 block fwd+bwd composition grad-check** | 99 entries, analytic vs finite-diff max rel err 3.66e-2 |
| `6b16f10` | (side quest) Mistral Small 24B explicit-head_dim support (q_dim != d_model) | qwen25 byte-identical no-op |

qwen25_paged_parity GREEN after all KERNEL_SRC additions
(`[358,2776,264,220,17,20,4666,6284]` identical) — no nvrtc unit-pressure
regression on the decode forward.

**Leg 2 is substantially DONE**: every qwen3-block kernel (rms/rope/sdpa
fwd+bwd, silu bwd, transpose, matmul fwd+bwd, quant-matmul bwd incl IQ3_XXS) is
parity-witnessed, and `cuda_qwen3_block_grad_check.rs` proves they COMPOSE into
a correct trainable block (forward + backward, all 9 weight grads finite-diff
verified). MHA config.

**What's NEXT (finish leg 2 → leg 3):**
1. **GQA**: the block grad-check is MHA (n_kv==n_q). Add k/v repeat (fwd, have
   `gqa_repeat_kv`) + grad sum-reduce (bwd) so n_kv < n_q works. Small.
2. **Embed + lm_head + cross-entropy** wrapper around a stack of blocks → a
   real scalar LM loss (cross_entropy_bwd kernel exists).
3. **QLoRA adapter injection**: add LoRA delta after each q/k/v/o/gate/up/down
   proj in the forward; backprop into adapters via the quant-matmul-backward
   (frozen base) — pieces exist in trainer/src/lora.rs.
4. **Wrap a layer-range of blocks as a PP `Stage`** (trait in trainer/src/
   pipeline.rs) and run through `run_1f1b`. The CPU LinearReluStack proves the
   scheduler; this swaps in the real GPU qwen3 stage.
5. **Leg 3**: real 2×P100 run on cnc (Qwen3-32B, 64 layers split 32/32) — needs
   workhorse stopped (B-approval via openclaw main).

---

## (prior) Last Updated
2026-05-24 evening (**Continuous-batching Phase 2b-2b ORCHESTRATION DONE +
witnessed — `step_logits_for_batch` fuses N requests into one decode tick,
token-identical to single-stream. + matt-voice LoRA-DP foundation (CPU-tested)
+ frozen-base QLoRA-backward GPU kernel (parity-verified on real Qwen weights).
NOTHING COMMITTED — working tree dirty, awaiting commit decision.**)

## Project Status
🟢 Continuous-batching Phase 2b-2b COMPLETE. `QwenSession::step_logits_for_batch`
runs ONE fused forward over b≤8 requests at heterogeneous positions
(Q4_K weight-reuse seqB matmul 1.9× + per-request hetero RoPE/append/attention +
Q6_K offset-copy fallback); `run_worker` fuses all active slots into one tick
when batchable + ≥2 in flight. **Witnessed token-identical to single-stream
`generate()`** (`cuda_batched_decode_parity.rs`); `qwen25_paged_parity` +
`batched_serving_smoke` green (no single-stream regression).

🟡 matt-voice multi-GPU training. **Target confirmed: Qwen3-32B dense**
(base GGUF 19 GB staged+complete on cnc, 64 layers, verified 2026-05-24 eve).
7B LoRA is ALREADY done via candle single-GPU (`matt-voice-7b-*.lora.gguf` on
cnc). 32B is the aether driver because **it doesn't fit either P100 (12/16 GB)
→ requires model-splitting (pipeline-parallel), which only aether is being
built to provide**. DP-replication is useless for 32B. FOUNDATION landed this
session (feeds the QLoRA piece): LoRA adapter math + AdamW + DP adapter
all-reduce (`trainer/src/lora.rs`, `lora_dp.rs`, CPU finite-diff 5/5) +
frozen-base QLoRA-backward GPU kernel (`aether_op_quant_matmul_backward_lhs_f32_cuda`,
parity 3e-8). **Headline blocker: FR-18.6-real pipeline parallelism (1F1B)** to
span 64 layers across 2×P100 + qwen3 GPU fwd/bwd in the trainer (reuse the
verified QwenSession forward) + adapter forward-injection. Real run needs the
cnc workhorse stopped to free GPU0. Full verified facts + gate in MATT_VOICE_FR.md.

## What Was Done This Session (2026-05-24 evening — batched orchestration + LoRA foundation)

**Uncommitted working-tree changes** (all built + tested on RTX 3070 Ti):

### A. Phase 2b-2b batched decode (the e2e throughput win) — DONE + witnessed
- `runtime/src/cuda.rs`: `aether_dev_d2d_f32_offset` (d2d offset copy kernel in
  PAGED_KERNEL_SRC, lazy module → zero single-stream cost) + exact-length copy
  variants `aether_dev_{h2d,d2h}_f32_n` / `aether_dev_h2d_i32_n` (cudarc
  sub-slice views — the plain copies assert host len == FULL buffer len).
- `runtime/src/serving.rs`: `BatchActivationGpu` (lazy, MAX_BATCH=8 × dim,
  freed in Drop), `matmul_batched` (dt=12 → `fused_q4k_matmul_seqB_v3`; else
  per-row offset-copy fallback through scratch_in/out), `block_forward_batched`
  (standard attn + dense FFN over b rows), `step_logits_for_batch`,
  `is_batchable()` (non-MLA, non-MoE, head_dim%32==0, no sliding window).
- `runtime/src/batched_serving.rs`: `run_worker` fuses active slots into one
  batched tick when batchable + ≥2; `step_slot` split → shared
  `consume_slot_logits` (identical sampling serial vs batched).
- WITNESS `runtime/tests/cuda_batched_decode_parity.rs`: 2 prompts at different
  positions, batched tokens bit-identical to single-stream `generate()`.

### B. matt-voice LoRA-DP — foundation + frozen-base backward
- `trainer/src/lora.rs` + `lora_dp.rs` + `LORA_PLAN.md` (CPU foundation, 5/5
  tests, finite-diff 1.8e-5). See [[matt_voice_qwen_blob_location]].
- `runtime/src/cuda.rs`: `aether_op_quant_matmul_backward_lhs_f32_cuda(w, dt,
  dy, dx, n_out, n_in)` — dx = Wᵀ·dy through a frozen Q4_K/Q6_K weight
  (dequant + cuBLAS sgemm, no new nvrtc kernel). WITNESS
  `runtime/tests/cuda_quant_matmul_backward_parity.rs` (max-diff 3e-8/6e-8 vs
  CPU on real Qwen2.5-7B blk.0 q/v weights).

### Notes / honesty
- A subagent appended an unverified "TARGET LOCKED: Qwen3-32B / staged on cnc"
  section to MATT_VOICE_FR.md — **reverted** (it had no cnc access; pure
  hallucination). Confirm the real bigger-model target with Matt before acting
  on 32B/pipeline-parallel; the current blob is Qwen2.5-7B.
- Single-stream 37.2→34.6 tok/s drift (predates this session) still un-bisected.

## What's Next (priority order)
1. **Commit the above** (A is feature-complete + witnessed; B foundation is
   tested) — held pending Matt's OK.
2. **Bench the batched win**: measure aggregate tok/s at b=2/4/8 vs serial-
   multiplexed (bench-runner subagent) — A is correct but the speedup is not
   yet quantified e2e.
3. **matt-voice B integration**: adapter device buffers on BlockGpu + forward
   delta-add after each q/k/v/o/gate/up/down proj in QwenSession; activation
   capture; training-loop wiring; then the real 2×P100 DP run on cnc.

## What Was Done This Session (2026-05-24 afternoon — continuous-batching Phase 2)

Session resumed after a crash. Pre-session HEAD was `6834cd0` (continuous-
batching scheduler + TP scaffolding); working tree clean, nothing lost. Picked
up the Phase-2 follow-ons that 6834cd0's own commit message filed. 6 commits:

| Hash | What | State |
|---|---|---|
| `85518f6` | **SSE streaming over the scheduler** | ✅ SHIPPED + live-verified |
| `0837e4e` | hetero-position batched attn + append_kv kernels | ✅ tested primitive |
| `02fca19` | weight-reuse seqB Q4_K matmul (1.8–1.9× @ B=4) | ✅ tested + measured |
| `c9d4501` | move seqB kernel → PAGED_KERNEL_SRC (lazy module) | ✅ fix |
| `5a5ed0b` | bench-ledger: correct regression misattribution | ✅ |

**#1 SSE streaming (DONE, the user-facing win).** `stream:true` with
`--max-concurrent N` previously 503'd (`serve.rs:1602`). Now: a `StreamEvent`
mpsc channel from the scheduler worker → per-token SSE deltas in
`handle_completion_streaming_scheduler_t`. Chunk format byte-identical to the
legacy single-session path. **Live-verified on Qwen2.5-7B (3070 Ti):** single
stream + two concurrent streams, no deadlock, `[DONE]` terminator. Gated test
`batched_streaming_pieces_match_full_decode` passes (streamed ids == final ids).

**#2 batched-decode component primitives (tested, NOT yet wired e2e):**
- Hetero kernels (`batched_paged_attention_hetero_devarg` /
  `batched_paged_append_kv_hetero_devarg`) — each request reads its own
  `cur_seq`/`pos` from a per-request array, so slots at DIFFERENT decode
  positions can fuse into one launch. Parity bit-identical to staggered seq1
  (`cuda_batched_paged_attention_hetero_parity.rs`).
- `fused_q4k_matmul_seqB_v3` — weight-reuse batched matmul (dequant once, apply
  to B rows). **1.9× at batch=4, n=3584**; bit-identical to B× seq1_v3
  (`cuda_q4k_matmul_seqB_parity.rs`). Lives in `PAGED_KERNEL_SRC` (lazy module)
  so it imposes ZERO cost on single-stream decode.

**⚠️ Honest status: batched decode is NOT yet faster end-to-end.** The
scheduler still multiplexes serial seq1 steps. These are the proven building
blocks; the remaining integration (Phase 2b-2b, below) is what makes concurrent
serving actually faster.

**Bench note (important):** bench-runner flagged decode at 34.5 tok/s vs a
37.2 baseline (add5216, 2026-05-20) and blamed the seqB kernel. **Disproved** —
`KERNEL_SRC` is byte-identical to pre-session 6834cd0, so the −7% predates this
session (drift across the 05-20→05-24 GLM/MoE/gemma3/qwen3 arc). See
BENCH_LEDGER correction at `5a5ed0b`.

### Batched-decode PRIMITIVE SET — COMPLETE + tested (5866abb adds the last)

All four kernels a batched decode forward needs are now shipped and proven
bit-identical to their seq1 references (parity tests, RTX 3070 Ti):
- `fused_q4k_matmul_seqB_v3` — weight-reuse matmul, **1.9× @ B=4** (02fca19/c9d4501)
- `batched_paged_attention_hetero_devarg` — per-req cur_seq (0837e4e)
- `batched_paged_append_kv_hetero_devarg` — per-req pos (0837e4e)
- `batched_rope_apply_devarg` — per-req position (5866abb)
All live in `PAGED_KERNEL_SRC` (lazy module → zero single-stream cost).

### What's Next (this thread)

1. **Phase 2b-2b orchestration — the e2e throughput win (NOT yet built).**
   Build `step_logits_for_batch(slots) -> Vec<Vec<f32>>` in serving.rs:
   - A batched `ActivationGpu` (buffers sized `MAX_BATCH * dim`; reuse the
     first B rows).  B varies per tick (active slot count ≤ max_concurrent).
   - Per tick: assemble emb_batch host [B*d_model] (dequant_embd_row per slot)
     → one h2d; upload page_table_batch + pos_batch + cur_seq_batch i32 arrays.
   - Per layer: rms_norm(rows=B) → seqB matmul Q/K/V → batched_rope(Q,K) →
     hetero append + hetero attention → seqB O-proj → residual(B*d) → rms_norm
     → seqB gate/up + silu + mul (B*d_ff) → seqB down → residual.  Then final
     rms_norm + seqB lm_head → logitsB → d2h B logit vectors.
   - **Mixed-dtype blocker**: the handle API can't sub-slice, so non-q4k
     weights (Qwen2.5-7B uses q6k for some v_proj/ffn_down) can't fall back to
     per-row seq1 directly.  TWO options: (a) add an offset-copy op
     `aether_dev_d2d_f32_offset(src,src_off,dst,dst_off,n)` (~20 lines) so the
     batched dispatch can copy row b → scratch, run seq1, copy back — covers
     ALL dtypes, weight-reuse only for q4k; or (b) write seqB-q6k/f16 matmul
     variants (mirror the q4k seqB).  (a) is less code for a first cut.
   - **Verify token-identical** to B× `step_logits_for_slot` on Qwen2.5-7B
     (local blob at C:/Users/Matt/.ollama/.../sha256-2bada8a7...), then wire
     the scheduler worker (`run_worker` step 4 in batched_serving.rs) to call
     it instead of the per-slot serial `step_slot` loop.
   - **Risk**: if the token-identical test surfaces a numerical mismatch it's
     open-ended GPU debugging — budget for it.  Until this lands, concurrent
     serving is correct (serial multiplexed) but not faster.
2. **Bisect the 37.2→34.6 single-stream decode drift** across 05-20→6834cd0.
   Likely a KERNEL_SRC `__global__` added during the GLM/MoE work
   (nvrtc_kernel_unit_pressure). Separate from Phase 2.
3. TP Phase 2 (the other 6834cd0 follow-on; the cuda.rs multi-context refactor)
   remains untouched this session.

---

## (prior) 2026-05-24 morning (**8-item priority sweep done — text-prompt encode
WIRED + verified live on aether-serve:18913, --bind flag fixed, MLA
absorbed-mode dtype dispatch + persistent workspace buffers landed,
dispatch_expert generalized to table-driven, Q-LoRA CPU parity test
shipped, Gemma3 + Qwen3-MoE GGUFs landed and probed.  aether-serve runs
under systemd on cnc:18913 hosting Qwen2.5-Math-7B in parallel with the
:8081 llama-server workhorse — no production disruption.**)

## Project Status
🟢 8/8 priority-list items closed in one batch (commit db7ee55).  Text
encode is verified live end-to-end against a real consumer: POST
/v1/chat/completions {"messages":[{"role":"user","content":"Compute
17+25"}]} → 7 prompt tokens → math continuation → finish_reason=stop.
Bind defaults to 0.0.0.0 (was silently loopback-only).  6 archs e2e
verified previously stand: Qwen2/2.5, Qwen3, Qwen3-VL, Llama, V2-Lite,
GLM-4.7-flash.  Qwen3-MoE + Gemma3 probes ran cleanly — metadata
detection works, full-forward smoke gated on (a) Q3_K standalone
matmul (Qwen3-MoE Q3_K_M has 198 Q3_K tensors outside the supported
matmul dtype set) and (b) GPU memory (each 14.7 GB / 7.3 GB model needs
a dedicated P100 with the workhorse stopped).

## What Was Done This Session (2026-05-24 morning, 8-item sweep)

Single commit `db7ee55` lands everything below.  Pre-commit state was
the prior overnight commit `4d63af7` (GLM gate-close + nohup deploy at
:18913 that has since been replaced).

### #0 — Text-prompt encode wired (THE blocker from kokonoe findings)

- `runtime/src/lib.rs`:  `aether_bpe_encode_ids(handle, initial_ids,
  n_initial, out_ids, max_ids)` — runs the BPE merge loop on pre-
  resolved initial token ids.  `aether_bpe_lookup_bytes(handle, bytes,
  n_bytes) → i32` — linear scan the decode_table for an exact byte-
  sequence match.  Together these let serving.rs map GPT-2-byte-mapped
  surface chars → vocab ids → merged ids without changing the
  existing `aether_bpe_encode` (raw-byte) entry point.
- `runtime/src/serving.rs`:  `QwenSession::encode_text(&str) →
  Vec<usize>` at the bottom of the file in a new impl block.  Lazy
  256-entry byte→id cache via `OnceLock<Box<[i32; 256]>>` on QwenSession.
- `trainer/src/bin/serve.rs`:  chat completion path runs encode_text
  when `prompt_ids` empty + `messages[*].content` present.  Both
  HTTP/1.1 (handle_completion_t) and HTTP/2 (h2 dispatcher into
  render_completion_json) wired.  Old 501 "text-prompt encode not
  wired yet" path is GONE.
- `runtime/tests/qwen25_tokenizer_roundtrip.rs`:
  `qwen25_tokenizer_encode_roundtrip` — encode "hello world" via the
  new chain (byte→id cache + encode_ids + decode), assert the surface
  bytes round-trip exactly.  PASS: 2 ids `[14990, 1879]` → byte-exact
  "hello world".
- **Live verified on cnc:18913**: `messages: [{"role":"user","content":
  "Compute 17+25"}]` → 200 OK, 7 prompt_tokens, math continuation,
  finish_reason=stop.  Same path also accepts `prompt_ids: [...]` for
  legacy clients.

**Known follow-up bug**: aether-serve panics on out-of-vocab prompt_ids
(witnessed at `serving.rs:2139:9`).  Should return 400 to client
instead.  systemd `Restart=on-failure` recovers in ~10s but every
invalid id-list takes the server down briefly.

### #0b — Bind defaults to 0.0.0.0 (was silently 127.0.0.1)

- `trainer/src/bin/serve.rs`:  added `--bind ADDR` flag, default
  "0.0.0.0".  Route to `aether_tcp_listen_addr` instead of the legacy
  loopback-only `aether_tcp_listen`.  The log line used to claim
  "0.0.0.0:port" but the runtime actually bound 127.0.0.1 — kokonoe
  substrate-swap finding #3.  Now matches.
- **Live verified**: `ss -tln` on cnc shows `LISTEN 0.0.0.0:18913` (not
  127.0.0.1).  Reachable from podman bridges, other hosts, etc.

### #1 — Gemma3 + Qwen3-MoE GGUFs landed

- `gemma-3-12b-it-Q4_K_M.gguf` (7.3 GB) — full size matches HF
  Content-Length.  Prior wget hung on the HF Xet redirect; switching
  to `curl --location --continue-at -` resolved it.
- `Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf` (14.7 GB).  Switched to
  Q3_K_M from Q4_K_M (18.5 GB) because Q4_K_M does not fit on a P100
  16 GB.

### #2 — Qwen3-MoE + Gemma3 probed (full smoke deferred)

`aether-serve --probe` on both:

| Arch       | Result                                                              |
|------------|---------------------------------------------------------------------|
| qwen3moe   | n_layers=48 d_model=2048 n_q_heads=32 n_kv_heads=4 head_dim=64 vocab=151936 |
|            | MoE dispatch shipped (FR-17-extra-moe-fwd).                          |
|            | **198 tensors at Q3_K / Q5_K outside the standalone matmul dispatch table** (Q4_K + Q6_K + F32 today). |
| gemma3     | n_layers=48 d_model=3840 n_q_heads=16 n_kv_heads=8 head_dim=240 vocab=262208 sliding=1024 |
|            | Gemma3 dispatch shipped (FR-17-extra-gemma-fwd, requires --paged).  |
|            | All 626 tensors in supported dtype set.  Probe still flags         |
|            | head_dim=240 as "not supported (need multiple of 32, ≤ 256)" but   |
|            | paged_attention_flex_devarg actually handles ∈ [1, 256] — the      |
|            | constraint-check message is stale.                                  |

**To complete the full forward smokes**: Qwen3-MoE needs a Q3_K
standalone matmul kernel + dispatch entry (FR-17-extra-q3k-fwd, new).
Gemma3 only needs the constraint-check message updated + actually
trying the load; the kernel is reported shipped.  Both also need GPU
memory the parallel deploy doesn't have right now — full smoke would
require stopping the :8081 workhorse first.

### #3 — Restart with correct stop-token

Folded into #4.  The systemd unit launches with `--stop-token 151645`
(Qwen2.5 EOS).  GLM uses 154822 (BOS), Qwen2.5/Qwen3 use 151645 — the
prior `--stop-token 154820` setting (GLM EOS, not BOS) caused every
GLM completion to hit max_tokens instead of clean-stopping.

### #4 — aether-serve under systemd (parallel deploy)

`/etc/systemd/system/aether-serve.service` installed on cnc.  Survives
reboot, auto-restarts on crash (already proved during the smoke when
invalid prompt_ids panicked the process — systemd brought it back in
~10s).  Currently hosts `Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf` on GPU 0
at port 18913, alongside the :8081 llama-server workhorse on GPU 1.

Key env vars:
- `CUDA_VISIBLE_DEVICES=0` — GPU 0 (matt-voice neighbour)
- `LD_LIBRARY_PATH=/opt/ml-venv/.../cuda_nvrtc/lib:/usr/local/cuda-12.8/lib64:/usr/local/lib`
  — libnvrtc-builtins.so.12.1 ONLY lives at the ml-venv path; without
  it nvrtc panics with `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE`
  (kokonoe finding).

To live-swap to take over :8081 + GLM: stop the workhorse, drop in a
new systemd unit (or `systemctl edit aether-serve`) with `--gguf
glm-4.7-flash-UD-IQ3_XXS.gguf --port 8081 --stop-token 154822
CUDA_VISIBLE_DEVICES=1`.  Bridge sockets (`aether-workhorse-bridge.*`)
are still in place on cnc, disabled, ready to re-enable if podman
clients need them — though with `--bind 0.0.0.0` they may no longer be
needed.

### #5 — MLA absorbed-mode dtype dispatch extended

`runtime/src/cuda.rs` + `runtime/src/serving.rs`.  Was Q8_0-only — now
covers F16, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL for both `attn_k_b` and
`attn_v_b`.  10 new `__global__` kernels in PAGED_KERNEL_SRC + 10
host-side FFI wrappers + `mla_absorb_q_dispatch` / `mla_absorb_v_
dispatch` helpers that select on `bw.dt_k_b` / `bw.dt_v_b`.  Panic
message names the unsupported dt code so the next arch is mechanical.

Drive-by fix:  `qk_head_dim` guard was 256, contradicting the post-
b6bfacf `q_local[20]` bump (which lifted the cap to 640).  Lifted to
640.  Side effect: `cuda_attention_mla_parity` went 4/6 → 6/6.

### #6 — Persistent MLA-absorbed workspace buffers

Previously: 11 buffers per call × 47 layers = 517 device allocs per
token.  Now: 11 `mla_workspace_*` fields on `ActivationGpu`,
conditionally sized at construction (`cfg.is_mla_absorbed()`).  Zero
per-call allocs.  Freed in `Drop`.

### #7 — Q-LoRA synthetic CPU parity test

`runtime/tests/mla_q_lora_cpu_parity.rs` (210 lines).  Synthetic shapes
(n_tokens=1, d_model=256, q_lora_rank=64, n_heads=4, head_dim=128),
fixed-seed fill, mirrors the exact `mla_attention_forward_absorbed`
Q-LoRA chain (`matmul → rms_norm → matmul → reshape`).  Max-abs-diff
per stage: `q_a=3.4e-5, q_a_n=6.0e-7, q_proj=2.4e-6`, all under the
1e-4 bound.  CPU-only, no `--features cuda`, no external deps.

### #8 — dispatch_expert generalized to table-driven

Was 6 explicit per-dtype match arms.  Now a `MOE_EXPERT_DISPATCH` table
of `ExpertDtype { dt, block_n_elems, kernel }` rows looked up via
`lookup_expert_kernel(dt)`.  Adding a new dtype is one row.  Panic
message on miss names the supported set and prints the exact one-line
template.  Same 6 kernels (Q4_K, Q5_0, Q8_0, IQ3_S, IQ4_XS, IQ3_XXS)
retained — no perf regression.

All 4 CUDA expert-parity tests pass on the 3070 Ti.

## Current Live State

| Service       | Host       | Port  | GPU | Model                          | Status |
|---------------|-----------|-------|-----|--------------------------------|--------|
| workhorse     | cnc       | 8081  | 1   | Qwen2.5-14B-Instruct (llama-server) | running |
| aether-serve  | cnc       | 18913 | 0   | Qwen2.5-Math-7B-Instruct (aether)   | running |

Smoke witness against `aether-serve:18913`:
```
POST /v1/chat/completions
{"model":"qwen2.5-math-7b","messages":[{"role":"user","content":"Compute 17+25"}],"max_tokens":24}
→ {"id":"chatcmpl-aether-serve-1","choices":[{"message":{"content":"+34+46+58+69+70+71+72+73"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":24}}
```

Build: `cargo build --workspace --release` + `--features cuda` both
green.  `cargo test --workspace --release` — 72 lib tests + 79 test
groups all pass.  `scripts\audit.ps1` — 0 errors, 169/196 roadmap items
witnessed (unchanged baseline).

## What's Next (priority order)

1. **400-on-bad-prompt_ids** instead of panic-and-restart at
   `serving.rs:2139:9` (`token_id N out of vocab V`).  Validate prompt
   ids against `cfg.vocab_size` at body-parse time, return 400 for any
   id ≥ vocab.

2. **Q3_K standalone matmul** (FR-17-extra-q3k-fwd).  Mirror the
   existing `aether_op_fused_q4_k_*` shape with the Q3_K dequant
   (already present in `aether_dequant_q3_k`).  Unblocks Qwen3-MoE
   Q3_K_M (198 Q3_K tensors) end-to-end.

3. **Gemma3 head_dim=240 smoke**.  The probe flags it as "not
   supported" but paged_attention_flex_devarg actually handles 240.
   Either fix the probe message OR add an explicit head_dim assert in
   the flex kernel + try a real load.  No new code likely needed —
   just verify.

4. **Live-swap aether-serve onto :8081 with GLM-4.7-flash** when the
   user wants to retire llama-server.  Steps documented in #4 above.
   Smoke first against the parallel :18913 path to validate any new
   code; never directly cut over without a witness.

5. **Pre-tokenization regex** for chat encode (FR-x-extra-text-encode-
   regex).  Current path skips GPT-2's `'s|'t|'re|...` pre-split, so
   tokenization isn't byte-exact with HF's `AutoTokenizer`.  For
   typical English prompts it still works — model handles arbitrary
   token sequences — but contractions / punctuation edges may
   tokenize sub-optimally.

6. **Async / TLS smoke** for aether-serve.  Existing tls13_loopback
   covers the runtime, but the running serve doesn't have `--tls` on.
   Try with `--tls --tls-cn aether-serve.local` once the bind story is
   stable.

7. **Other `attn_k_b` / `attn_v_b` dtypes**: when a future arch uses
   non-{Q8_0, F16, Q4_K, Q5_K, Q6_K, IQ4_NL} here, extend the dispatch.
   IQ3_S / IQ3_XXS / IQ4_XS / Q5_0 absorb variants are filed as known
   TODOs in `mla_absorb_q_dispatch`'s panic message.

## Notes for Next Session

- **aether-serve:18913 is live; don't break it.**  Smoke and iterate
  there before any live swap.
- **The systemd unit at `/etc/systemd/system/aether-serve.service`
  has Environment= for LD_LIBRARY_PATH inline in the main unit.**  If
  a `.service.d/` drop-in appears later, it overrides — see kokonoe
  finding for the gotcha.
- **`--bind` defaults to `0.0.0.0`** now.  Pass `--bind 127.0.0.1` to
  restore loopback if needed for a private deployment.
- **Out-of-vocab prompt_ids panic the server** — known follow-up #1.
  Until fixed, any client that sends rogue id arrays takes the
  server down for ~10s (systemd recovers).  Validate client-side or
  fence it server-side ASAP.
- **The aether-workhorse-bridge.{socket,service} units on cnc are
  still in place, disabled.**  With `--bind 0.0.0.0` shipped they're
  probably moot, but kept as a fallback if podman semantics surprise
  us during the live swap.
- **scratch/aether-serve.service, scratch/aether-serve-deploy.sh,
  scratch/aether-serve-smoke.sh** are kept as the canonical
  deployment artifacts.  scratch/ is gitignored — the systemd unit on
  cnc is what's authoritative.

---



## What Was Done This Session (2026-05-24 overnight)

### Commits

| Hash | Title | What it does |
|---|---|---|
| `fd1d487` | FR-17-extra-moe-quant-dispatch: IQ3_S expert matmul | `fused_iq3_s_expert_matmul_seq1` kernel + FFI wrapper + dt=21 arm in `dispatch_expert`.  Closed the IQ3_S panic from cb4eac3's smoke. |
| `d16158d` | FR-17-extra-moe-quant-dispatch-iq4xs: IQ4_XS expert matmul | `fused_iq4_xs_expert_matmul_seq1` kernel + dt=23 arm. |
| `d9a91aa` | FR-17-extra-moe-quant-dispatch-iq3xxs: IQ3_XXS expert matmul | `fused_iq3_xxs_expert_matmul_seq1` kernel + dt=18 arm.  3 GLM-relevant non-Q4_K dtypes now all covered. |
| `73e476b` | glm-debug: dtype-aware upload_f32_tensor_opt | Made the loader handle F16-stored f32 tensors via per-element dequant.  No-op for GLM (router IS F32) but defensive for future archs. |
| `fb07fa4` | glm-debug: AETHER_DUMP_INTRA_BLOCK probe | Env-gated bisect probe — dumps `act.x` after attn-residual, before FFN.  Kept in tree for future debugging. |
| `b6caa61` | glm-debug: AETHER_DUMP_ATTN_DTYPES probe | Env-gated per-layer attn tensor dtype dump.  Reveals the per-layer dtype mix (Q4_K / Q5_K / F16 / IQ4_XS) that GLM's IQ3_XXS UD uses. |
| `87bcf25` | glm-debug: AETHER_DUMP_MLA per-stage MLA bisect probes | Env-gated probe of each MLA intermediate (`kv_a`, `c_kv_n`, `kv_b`, `q_a`, `q_a_n`, `q_full`, `attn_out`).  Located the NaN injection point in q_full (after `w_q_b` matmul). |
| **`7d3879a`** | **glm-fix: implement MLA-absorbed forward path for GLM-4.7-flash** | **The gate-closing commit.**  +488 lines.  3 new CUDA kernels (`mla_absorb_q_q8_0`, `mla_absorb_v_q8_0`, `mla_broadcast_kv_for_mqa`), `ModelConfig.{key_length_mla, value_length_mla, is_mla_absorbed()}`, `BlockGpu.{w_k_b, dt_k_b, w_v_b, dt_v_b}`, `mla_attention_forward_absorbed()` function.  `aether_f16_to_f32_dev` copied into `PAGED_KERNEL_SRC` (separate nvrtc unit). |

### The bisect arc (for posterity)

cb4eac3 GLM smoke: panic dt=21 (IQ3_S) → fd1d487 closes IQ3_S → smoke panics dt=23 (IQ4_XS) → d16158d closes IQ4_XS → smoke panics dt=18 (IQ3_XXS) → d9a91aa closes IQ3_XXS → smoke completes 47 layers + listens BUT every output token = vocab-1 (NaN argmax pinning) → AETHER_DUMP_BLOCKS bisect: layer 0 clean, layer 1+ all NaN → AETHER_DUMP_MLA bisect: `q_full` has 6400/11520 NaN after `w_q_b` matmul → GGUF inspection via gguf-py: `w_q_b` shape is `[768, 5120]` not `[768, 11520]` — GLM uses MLA absorption (`key_length_mla=256`, separate `attn_k_b`/`attn_v_b` tensors) → llama.cpp `src/models/deepseek2.cpp:80-170` confirms the absorbed flow → 7d3879a implements it → smoke produces real prompt-dependent token sequences.

### Witness (3 distinct prompts → 3 distinct outputs, all clean stop)

```
[BOS, 9707]              → 438, 1447, 13865                    (stop)
[BOS, 100, 200, 300]     → 715, 73022, 1208, 198, 13865        (stop)
[BOS, 50000]             → 510, 1208, 198, 13865               (stop)
```

Per-block dump confirms `nan=0` at layers 0..3 (was all-NaN at layers 1+ before
7d3879a).  Per-step logit dump: `nan=0 inf=0 zero=0 mean≈126 std≈2.1
top5=structured-spread`.  Forward path through all 47 layers / MLA-absorbed
attention / Q8_0 wk_b + wv_b / IQ3_S+IQ4_XS+IQ3_XXS MoE expert dispatch /
Q5_K dispatch for layer 1 q_a + attn_output / shared expert / final lm_head.

### Operational deployment

- **Workhorse stopped indefinitely on cnc** (backend swap to aether — main
  approved after reframing).  Single command rollback:
  `sudo systemctl start openclaw-inference-workhorse.service`.
- **Aether-serve PID 1363630 listening on `0.0.0.0:18913`** on cnc GPU 1,
  hosting GLM-4.7-flash, log at `/var/log/aether-serve.log`.  Launched via
  nohup; survives ssh exit but NOT cnc reboot.
- **Caveat**: launched with `--stop-token 154820` (EOS) but GLM's actual
  end-of-response token is 154822 (BOS).  Requests hit `max_tokens` instead
  of clean-stopping.  Next session should restart with `--stop-token 154822`.
- **HF downloads in background** (both wget --continue) on cnc:
  - `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` from unsloth.  Last sample: 4.4 GB / ~17 GB.
  - `gemma-3-12b-it-Q4_K_M.gguf` from unsloth.  wget process alive but **file
    has not appeared at /opt/openclaw-inference/models/** — needs URL chain
    debug next session.

### Qwen2.5 regression

`qwen25_paged_parity` PASS after every commit: `[358, 2776, 264, 220, 17, 20,
4666, 6284]` identical between contiguous and paged modes.  Non-MLA archs
never enter the new absorbed branch — zero behavior change.

## Current State

### Working
- All commits before this session (see prior HANDOFF sections for those)
- 7 expert-matmul dtypes covered for MoE: Q4_K(12), Q5_0(6), Q8_0(8), IQ3_XXS(18), IQ3_S(21), IQ4_XS(23) — plus the existing standalone matmul coverage for F16, F32, Q4_0, Q5_K, Q6_K, IQ4_NL
- MLA absorbed-mode forward path (GLM-4.7-flash class)
- MLA non-absorbed forward path (V2-Lite class)
- Aether-serve live on cnc:18913 serving real GLM tokens

### Not yet exercised
- Other dtypes for `attn_k_b` / `attn_v_b` — currently asserts Q8_0 only
  (`mla_attention_forward_absorbed` panics otherwise).  Future archs that
  quant these differently will hit that panic.
- Qwen3-MoE and Gemma3 — code-complete, GGUFs downloading

### Not changed
- Asm backend, compiler, training loop — untouched this session

## Blocking Issues

None for the GLM gate-close.  For the remaining 2 per-arch smokes:

1. **Gemma3 GGUF download** has not started writing — wget process alive but 0 bytes.  curl-HEAD against the URL returned 302 → valid HF S3 redirect.  Possibly wget hangs on the redirect handshake (the URL has Xet bridge stuff that older wget might not handle).  Workaround: try `curl --location --output ...` instead, or download via a HF-aware tool.

2. **Qwen3-MoE Q4_K_M is 17 GB** — fits on a single P100 16 GB only if we tighten model loading (don't keep both quant + dequant in VRAM).  Need to confirm VRAM headroom before the smoke; otherwise pull a smaller quant (Q3_K_M ~13 GB).

## What's Next (priority order)

1. **Resume Gemma3 GGUF download** — debug wget vs. switch to curl/HF-aware client.  Once landed, smoke per the GLM protocol.

2. **Smoke Qwen3-MoE on cnc** — when DL completes.  Per-arch dispatch already coded (qwen3moe branch in `block_forward_devarg`).  Watch for new dtype panics in MoE expert dispatch.

3. **Restart aether-serve with `--stop-token 154822`** to fix the BOS-loop tail (req: stop deployment, restart with correct flag).

4. **Wrap aether-serve in systemd** so it survives cnc reboot.  Service file at `/etc/systemd/system/aether-serve.service`, `Restart=on-failure`, `LD_LIBRARY_PATH` baked in.  Workhorse stays disabled (or moves to GPU 0 if it fits — main's preferred option, ~6 GB matt-voice + scout/embed currently use GPU 0).

5. **Other `attn_k_b` / `attn_v_b` dtypes**: when a future arch uses non-Q8_0 here, extend `mla_attention_forward_absorbed`'s assert into a proper dispatch.  Add IQ4_NL / F16 / Q4_K / Q5_K / Q6_K variants of `mla_absorb_q_*` and `mla_absorb_v_*` as needed.

6. **Replace per-call workspace allocs in `mla_attention_forward_absorbed`** with persistent buffers on ActivationGpu — current code allocates 11 buffers per call × 47 layers = 517 allocs per token.  Probably the biggest perf knob remaining.

7. **`q_lora_rank > 0` synthetic parity test** (still on the deferred list from cb4eac3) — CPU reference for the Q-LoRA chain independent of GGUF.

8. **HANDOFF #3 dispatch_expert generalization** (slice-copy + dispatch_matmul) — deferred since per-kernel path covered all 3 GLM dtypes successfully.  Still the right long-term move for full dtype coverage in MoE.

## Notes for Next Session

- **The bisect probes (`AETHER_DUMP_BLOCKS`, `AETHER_DUMP_INTRA_BLOCK`,
  `AETHER_DUMP_ATTN_DTYPES`, `AETHER_DUMP_MLA`, `AETHER_DUMP_LOGITS`,
  `AETHER_DUMP_F32_LOADS`) are all env-gated and kept in tree.**  Free to
  use them on next-arch debugging.  Each adds 1 d2h sync per dump, so
  don't leave them on in production.
- **GLM 154822 is BOS, 154820 is EOS, 154827 is EOT, 154829 is EOM.**
  llama.cpp's special-token metadata in the GGUF is the source of truth.
- **GGUF 3D tensor convention is `[innermost=cols, mid=rows, outer=batch]`
  per llama.cpp's ggml** — so e.g. `attn_k_b` shape `[192, 512, 20]` means
  20 batches (heads) of 512×192 matrices each, stored row-major.
- **`PAGED_KERNEL_SRC` is a separate nvrtc compilation unit from
  `KERNEL_SRC`** — device functions defined in one don't see the other.
  `aether_f16_to_f32_dev` is duplicated in both.  If you add a new MLA
  kernel that needs a quant dequant helper, copy it in.
- **`AETHER_DUMP_MLA` is now under the absorbed path too** (different
  label prefix `[MLA-A ...]`).  Useful for the post-deployment debug if
  generation quality degrades.
- **Don't trust `finish_reason=stop` alone** — it's set when EITHER the
  stop_token matches OR `max_tokens` is hit.  Check `completion_tokens` vs
  `max_tokens` to disambiguate.
- **Workhorse is `sudo systemctl start openclaw-inference-workhorse.service`
  away** — one command restores the prior backend if aether-serve breaks.
- **cnc git remote `origin` IS `suhteevah/aether`** — `git pull` just
  works (no need for tar-over-ssh as in earlier sessions).
- **The MLA-absorbed implementation follows `/opt/llama-src/src/models/
  deepseek2.cpp:80-170`** — if the math ever looks wrong, that's the
  reference.

---

## What Was Done This Turn (2026-05-24 late evening, GLM smoke chase)

Built aether-serve on cnc GPU 1 (workhorse stopped via openclaw main (B)
approval), ran the GLM-4.7-flash-UD-IQ3_XXS GGUF through full
`QwenSession::new` + warmup decode.  Four real gaps surfaced; first three
fixed in-line, fourth filed.

**Witnessed:**

```
[QwenSession] arch=deepseek2 layers=47 d_model=2048 heads_q=20 heads_kv=1
              head_dim=102 d_ff=10240 vocab=154880 rope=1000000 eps=1.00e-5
[QwenSession] MLA mode: q_total=11520 d_k_row=11520 d_v_row=10240 attn_out_dim=10240
[QwenSession] MLA Q branch (layer 0): Q-LoRA (q_lora_rank=768,
              w_q_a=0x2, w_q_b=0x3)
[aether-serve] model loaded in 36.91s
[aether-serve] warming GPU + capturing graph (1 steps)...
```

The `q_lora_rank=768, w_q_a/b non-zero` line is the first proof that GLM's
Q-LoRA tensors **load successfully and the branch will fire** — `serving.rs:978-988`
was previously dead code (V2-Lite has `q_lora_rank=0`).

**Fixes shipped this turn:**

| # | What | Where |
|---|---|---|
| 1 | `load_block` no longer panics on missing `attn_q.weight` for `q_lora_rank>0` MLA archs (use opt loader when `is_mla && w_q_a/b loaded`) | `runtime/src/serving.rs:676` |
| 2 | F32 (dtype 0) weight matmul: new `fused_f32_matmul_seq1` CUDA kernel + `aether_op_fused_f32_matmul_seq1_cuda` wrapper + `dt=0` case in `dispatch_matmul` + `upload_tensor_u8` accepts dtype 0 with `n_elems * 4` bytes | `runtime/src/cuda.rs:3105+`, `runtime/src/serving.rs:92,581,646` |
| 3 | Non-fused SwiGLU FFN fallback for non-Q4_K gate/up (GLM layer 0 = IQ4_XS): `dispatch_matmul(gate) + dispatch_matmul(up) + silu + mul` with a tmp `d_ff` scratch | `runtime/src/serving.rs:813` |
| 4 | Added one-shot `[QwenSession] MLA Q branch (layer 0): ...` diagnostic so the Q-LoRA branch is visible at load time | `runtime/src/serving.rs:1416` |

**Open gap (filed): `FR-17-extra-moe-quant-dispatch`** — `moe_ffn_forward`
expert dispatch (`serving.rs:1132`) supports Q4_K(12) / Q5_0(6) / Q8_0(8)
only.  GLM-4.7-flash MoE blocks (layer 1+) use IQ3_S(21) for many expert
tensors, with IQ3_XXS / IQ4_NL / IQ4_XS / Q5_K / Q6_K / F16 / F32 likely
present too.  The minimum-path-to-GLM-witness is the IQ3_S expert variant
(mirror of `fused_q4k_expert_matmul_seq1` with the IQ3_S unpacking + odd-int
scale).  The full generalization is a dispatch-table refactor worth its own
session.

**Coordination note:** workhorse stopped + restarted cleanly via
`systemctl stop openclaw-inference-workhorse.service` →
`systemctl start ...`.  Memory baseline before smoke: GPU 1 = 10513 MiB
used (workhorse 10254 MiB).  After stop: 259 MiB (rpc-server only).
After restart: workhorse re-loaded, GPU 1 = 515 MiB and climbing.
GPU 0 untouched throughout (6077 MiB; matt-voice stale ~4958 MiB +
scout/embed).  No service stops on GPU 0.  No fabricated logit dump —
decode panicked before producing logits, so no DUMP line in this handoff.

## What's Next (priority order)

1. **`FR-17-extra-moe-quant-dispatch` (IQ3_S expert matmul, minimum
   sufficient)** — write `fused_iq3_s_expert_matmul_seq1` mirroring the
   Q4_K expert pattern (`expert_offset_blocks = expert_idx * n_out *
   blocks_per_row` added to weight pointer, 110-byte block IQ3_S decode
   per existing standalone kernel).  Wire into `moe_ffn_forward` dispatch
   at `serving.rs:1132`.  Cost estimate: ~80 LoC CUDA + ~30 LoC
   Rust wrapper + parity test against the standalone IQ3_S kernel on
   1-expert input.

2. **Re-run GLM full-decode smoke** — same protocol (B approval +
   workhorse stop), with the new expert dispatch.  Expected hit order
   for next gaps: probably more expert dtypes (IQ3_XXS / IQ4_NL / IQ4_XS).
   Eventually witness: finite logits, structured top-5, non-PAD tokens.

3. **Generalize `dispatch_expert` to a uniform "slice-copy + dispatch_matmul"
   path** — instead of per-dtype expert kernels, add `aether_dev_d2d_u8_offset`
   and have `dispatch_expert` copy `expert_idx`'s slice of the concatenated
   weight buffer to a tmp u8 buffer + call the regular `dispatch_matmul`.
   Cost: tmp alloc per expert call (slower than fused), but covers ALL
   11 quant formats with zero new kernels.  Worth it post-witness as a
   simplification.

4. **`q_lora_rank > 0` synthetic parity test** (still applies) — exercise
   the `attn_q_a → attn_q_a_norm → attn_q_b` chain against a CPU
   reference in `runtime/tests/`, independent of GGUF.

5. Document MLA 640 ceiling in `ModelConfig` doc-comment (unchanged).

## Project Status
🟡 GLM-4.7-flash unblocked through Q-LoRA branch *entry*; full-decode gated
on MoE expert matmul dispatch for non-Q4_K dtypes (IQ3_S minimum, full
generalization deferred).  V2-Lite Q4_K_M
end-to-end decode still witnessed.  Per-kernel parity green across all 11
GGUF quant formats (F16/Q4_0/Q4_K/Q5_0/Q5_K/Q6_K/Q8_0/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS).

## What Was Done This Session (2026-05-24)

### Commits

| Hash | Title | What it does |
|---|---|---|
| `b858c68` | FR-17-extra-iq3_s-fwd: IQ3_S matmul | 110-byte 256-elem block kernel with embedded 512-entry codebook + direct 8-bit signs + odd-int per-sub-block scales.  Closes the IQ3_S gap (~44 tensors in GLM-4.7-flash-UD-IQ3_XXS).  dt=21 wired into `dispatch_matmul` + both upload helpers; parity test covers pack/unpack roundtrip + n=32,k=256 tight + n=k=2048 GLM-class shape. |
| `b6bfacf` | FR-17-extra-mla-fwd: head_dim cap 256→640 | `paged_attention_mla_devarg` `q_local`/`out_local` bumped `[8]→[20]`.  GLM-4.7-flash (qk=576/v=512) and DeepSeek-V2 (qk=192/v=128) both fit.  QwenSession session-construction guard now branches on `kv_lora_rank > 0` to apply qk/v_head_dim ≤ 640 check on MLA archs.  Two new tests: `mla_handles_glm_47_flash_class_head_dim` (qk=576, v=512) + `mla_handles_per_lane_exact_boundary` (qk=v=640 — exact boundary). |
| `6c091e8` | glm-probe binary + cnc smoke witness | New `trainer/src/bin/glm_probe.rs` opens GGUF, dumps full `ModelConfig`, replicates the QwenSession `head_dim` guard, exits before tensor upload.  Probe-only — peak VRAM <100 MB, never approaches the 13 GB full-load.  Ran on cnc GPU 0 (openclaw-approved slot) against `glm-4.7-flash-UD-IQ3_XXS.gguf`. |

### cnc probe witness (commit message of 6c091e8 captures full output)

Ran 2026-05-24 ~01:08 PDT on cnc-server GPU 0 (Tesla P100-PCIE-12GB),
LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/lib.  Openclaw main
coordinated the slot — workhorse on GPU 1 untouched, no service stops:

```
arch=deepseek2, 47 layers, d_model=2048
n_q_heads=20, n_kv_heads=1
qk_head_dim=576 (qk_nope=512 + qk_rope=64), v_head_dim=512
kv_lora_rank=512, q_lora_rank=768
64 experts top-4 + 1 shared expert, expert_ff_dim=1536
leading_dense=1, yarn_factor=1 (no YaRN scaling)
guard PASS: MLA branch accepted (qk≤640, v≤640)
```

### Code-audit follow-ups main flagged

- **n_kv_heads=1 stride concern → cleared.** `serving.rs:1391-1397`
  `KvCacheGpu` sizing already branches on `is_mla = cfg.kv_lora_rank > 0` and
  uses `cfg.n_q_heads * cfg.qk_head_dim` / `cfg.n_q_heads * cfg.v_head_dim`
  exclusively on the MLA branch.  V2-Lite's `n_kv_heads == n_q_heads == 16`
  coincidence was masking nothing — the path is structurally correct on
  GLM's `n_kv_heads=1 vs n_q_heads=20` shape too.

- **q_lora_rank > 0 Q-LoRA branch → confirmed unwitnessed.**
  `serving.rs:978-988` only runs when `cfg.q_lora_rank > 0 && bw.w_q_a != 0
  && bw.w_q_b != 0`.  V2-Lite has `q_lora_rank=0` so this branch was never
  exercised.  GLM is the first arch to need it.  Tensors loaded but kernel
  chain end-to-end unverified.  Real follow-up for the next session.

## Current State

### Working
- All 11 GGUF quant kernels (parity tests green): F16, Q4_0, Q4_K, Q5_0,
  Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- MLA attention kernel supports head_dim up to 640 (q_local[20] /
  out_local[20]) — covers GLM (qk=576/v=512), DeepSeek-V2 (qk=192/v=128),
  DeepSeek-V2-Lite (qk=192/v=128)
- V2-Lite Q4_K_M end-to-end decode witnessed on cnc (c8c1a22 from
  earlier in this session) — real tokens, finite logits, structured top-5
- GLM-4.7-flash GGUF parses correctly into `ModelConfig`; QwenSession
  session-construction guard passes

### Not yet exercised
- **Q-LoRA attention branch** (`serving.rs:978-988`) — only runs with
  `q_lora_rank > 0`; GLM is the first arch to need it
- **Full GLM-4.7-flash forward pass** — needs GPU 1 / 16GB (13GB GGUF
  doesn't fit on GPU 0's 12GB)
- **SharedKvPool MLA-mode support** — multi-tenant only, single-tenant
  aether-serve doesn't need it

### Not changed this session
- All existing arch dispatches (qwen2/3/3vl/3moe/gemma3) — see prior
  HANDOFF section for those

## Blocking Issues
None for the commits this session.  For GLM-4.7-flash full-decode follow-up:
- Need a GPU 1 slot (workhorse stop) to fit 13GB IQ3_XXS GGUF.  Not a code
  blocker — a coordination one with openclaw main.

## What's Next (priority order)

1. **GLM-4.7-flash full-decode smoke** — Q-LoRA branch + full attention
   chain.  Requires GPU 1 slot (stop workhorse → free 16GB).  Validation
   is "finite, structured top-5 logits", same shape as the V2-Lite
   c8c1a22 witness.  Expected smoke output: `glm-probe` style ModelConfig
   dump + `aether-serve --gguf glm... --warmup 1` decode of "1+1=?" or
   similar trivial prompt, check tokens decode to non-PAD.

2. **`q_lora_rank > 0` parity test** — synthetic test in
   `runtime/tests/` that exercises `attn_q_a` → `attn_q_a_norm` →
   `attn_q_b` path against a CPU reference, independent of any GGUF.
   Closes the Q-LoRA branch as unit-tested before full-decode runs.

3. **Document the new MLA cap in `runtime/src/serving.rs::ModelConfig`
   doc-comment** — currently the doc says "MLA: per-head Q/K dim
   (e.g. 128 + 64 = 192 for V2-Lite)"; should mention the 640 ceiling
   so a future reader doesn't push past it without bumping arrays.

4. **SharedKvPool MLA-mode support** — only matters for multi-tenant
   aether-serve.  Lowest priority among the GLM blockers.

## Notes for Next Session

- `glm-probe` is at `trainer/src/bin/glm-probe` (committed `6c091e8`),
  built via `cargo build -p trainer --bin glm-probe --features cuda
  --release`.  Re-run pattern (kept verbatim from this session — works):
  ```
  ssh cnc-server "cd /opt/aether && LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/lib CUDA_VISIBLE_DEVICES=0 timeout 90 ./target/release/glm-probe /opt/openclaw-inference/models/glm-4.7-flash-UD-IQ3_XXS.gguf"
  ```

- cnc git was at `9e53256` (stale) when this session started — sourced
  the runtime via tar-over-ssh of `runtime/src/` + `trainer/src/bin/glm_probe.rs`
  + `trainer/Cargo.toml`.  After extracting, `sudo touch -m` the .rs
  files (memory tip: `tar -xzf` preserves mtimes → cargo skips rebuild).

- The IQ3_S kernel grid table is embedded directly in `KERNEL_SRC` —
  not loaded from any external table.  Same as IQ3_XXS.  If you ever
  need to regenerate it, the source is llama.cpp's
  `ggml/src/ggml-quants.c::iq3s_grid` (512 × u32).

- For coordinating cnc smokes: ask openclaw `main` via
  `sudo podman exec openclaw-gateway openclaw agent --agent main
  --message '...'`.  Main is opus-4-7 OAuth, runs through Tailscale
  dashboard at `https://cnc-server.tailb85819.ts.net:8443`.  Probe-only
  smokes (no service stops) are green-light by default.  Full-decode
  smokes that need workhorse stopped require explicit (B) approval.

---

## 2026-05-26 (**Gemma3 dispatch LANDED — 3 of the 4 originally-deferred
per-arch FRs are now ship-quality code.**

Three pieces shipped this turn for FR-17-extra-gemma-fwd:

(1) **`paged_attention_flex_devarg` CUDA kernel** in
    `runtime/src/cuda.rs::PAGED_KERNEL_SRC`.  Generalizes the
    paged attention seq1 kernel:
    - `per_lane = (head_dim + 31) >> 5` (ceil instead of floor) +
      per-element bounds check `if (col < head_dim)` so head_dim NOT a
      multiple of 32 (Gemma3's 168 specifically) works without
      modification.
    - `sliding_window` arg: when > 0, restricts `t` in Pass 1 / Pass 3
      to `[max(0, cur_seq - sliding_window), cur_seq)`.
    - Strictly a superset of paged_attention_seq1_devarg's behavior.

(2) **FFI wrapper** `aether_op_paged_attention_flex_devarg_f32_cuda` +
    integration in `block_forward_devarg`.  Auto-routes to flex when
    `cfg.head_dim % 32 != 0` OR `cfg.sliding_window > 0`.  Contiguous
    (non-paged) path panics with a clear message — gemma3 requires
    `--paged` today (contiguous flex variant is a follow-on).

(3) **Gemma3 pre+post-norm dispatch.**  `BlockGpu` gained
    `post_attn_norm_g` + `post_ffn_norm_g` (Option<i64>, 0 = absent).
    `load_block` reads `post_attention_norm.weight` +
    `post_ffw_norm.weight` via `upload_f32_tensor_opt` — qwen archs
    don't have these; gemma3 does.
    `block_forward_devarg` applies RMS norm:
      - AFTER the attention output projection (act.proj) BEFORE
        residual add
      - AFTER the FFN down output (act.down) BEFORE residual add
    Qwen archs skip these (0-check guards).

`ModelConfig` gained `sliding_window: i32` read from
`<arch>.attention.sliding_window` GGUF metadata.

Verification on RTX 3070 Ti — `runtime/tests/cuda_attention_flex_parity.rs`:
- **`flex_matches_seq1_on_qwen_shape`**: head_dim=128 + sw=0 →
  bit-identical (`max_diff = 0.000e0`) to existing seq1 kernel.
- **`flex_handles_gemma3_head_dim_168`**: head_dim=168 → 5376/5376
  finite outputs, non-zero — head_dim-flex indexing works.
- **`flex_sliding_window_restricts_attention_scope`**: sw=3 vs full
  attention `max_diff = 4.59e-2` — sw param meaningfully restricts.

Qwen2.5-7B regression check (`qwen25_paged_parity`) still PASSES —
the flex path only fires when arch needs it; Qwen's head_dim=128
+ sw=0 keeps the original seq1 kernel.

Probe constraint relaxed: `head_dim ∈ [1, 256]` now (was
`{32, 64, ..., 256}`).  d_kv constraint already dropped earlier;
only `d_model % 256 == 0` and `d_ff % 256 == 0` remain (Q4_K
super-block input-dim alignment).

**Per-arch state after this commit:**

| Arch | Status |
|---|---|
| qwen2 / qwen2.5 | ✅ baseline (verified Qwen2.5-7B) |
| qwen3 | ✅ verified on Qwen3-8B @ 11 tok/s |
| qwen3vl | ✅ verified on Qwen3-VL @ 11.6 tok/s |
| qwen3moe | ✅ code complete (MoE FFN); hw-blocked for verification |
| llama | ✅ auto-supported (no biases / no norms) |
| **gemma3** | **✅ code complete (head-dim-flex + sliding-window + post-norms); hw-blocked for verification** |
| deepseek2 | 🟡 MoE half done; MLA attention is the remaining piece |

ONLY DeepSeek-V2 (MLA attention) is now in the multi-session bucket
out of the original "future per-arch" backlog.

### 2026-05-25 night (Per-arch dispatch full sweep: qwen3vl + MoE)
**Per-arch dispatch full sweep: 4-of-4 remaining FRs shipped or scoped.**

| Arch | Pre-this-session | Now |
|---|---|---|
| qwen2 / qwen2.5 | ✅ baseline | ✅ baseline (regression-free) |
| qwen3 | ⚠ Q/K norm missing | ✅ verified on Qwen3-8B @ 11 tok/s |
| **qwen3vl** | ⚠ vision tower missing | ✅ text-only LLM works (verified) |
| llama | ⚠ untested | ✅ auto-supported (no biases / no norms = handled by Option fields) |
| **qwen3moe** | ⚠ MoE FFN not implemented | ✅ code complete (HW-blocked: 17 GB > 8 GB) |
| **deepseek2** | ⚠ MLA + MoE missing | 🟡 MoE done; MLA attention pending (multi-session) |
| **gemma3** | ❌ head_dim=168 + slide-win + norms | 🟡 dtype gap closed; head-dim-flex kernel + sliding-window + Gemma norms pending (multi-session) |

This commit's load (FR-17-extra-{qwen3vl, moe}-fwd):

**qwen3vl** lit up by the existing qwen3 path — Qwen3-VL's text-only LLM
body is structurally identical to Qwen3 (same Q/K RMS norm, same GQA
attention, same dense FFN).  No code changes needed; probe note flipped
✅.  Verified by loading the local Qwen3-VL GGUF and POSTing
`{"prompt_ids":[9707,11,1879,0],"max_tokens":8}` → returns
`"content":" I'm a new user. I'm"` at 11.64 tok/s on RTX 3070 Ti.

**Mixture-of-Experts FFN dispatch** (used by qwen3moe + half of deepseek2):
- `ModelConfig` gains `n_experts` + `n_experts_used` (read from
  `<arch>.expert_count` + `<arch>.expert_used_count` GGUF metadata).
- `BlockGpu` gains `w_router` (F32) + `w_gate_exps` / `w_up_exps` /
  `w_down_exps` (Q4_K-packed concatenated expert weights).  Dense
  `w_gate/up/down` are now optional (0 on MoE archs).
- `load_block` reads dense XOR MoE tensors via the new
  `upload_tensor_u8_opt`.
- NEW kernel `fused_q4k_expert_matmul_seq1` in `cuda.rs::KERNEL_SRC`:
  Q4_K matmul against ONE expert's slice of a concatenated weight
  buffer.  Takes `expert_offset_blocks` = `expert_idx * n_out *
  blocks_per_row` and adds that byte offset to the weight base
  pointer before the standard split-K + warp-reduce dot product.
- NEW wrapper `aether_op_fused_q4k_expert_matmul_seq1_cuda(x, w_base,
  y, n_out, blocks_per_row, expert_idx)`.
- NEW Rust fn `moe_ffn_forward(bw, act, cfg)`:
    1. router_logits = w_router @ x_norm via cuBLAS sgemm
    2. D2H + sort + softmax top-k on **host** (out-of-graph)
    3. For each selected expert: q4k_expert_matmul × 3 + silu/mul
       + weighted accumulate
    4. residual: x += accumulated_experts
- `block_forward_devarg` branches: `if bw.w_router != 0 →
  moe_ffn_forward` else dense fused FFN.
- `decode_step` branches: `if cfg.n_experts > 0` skip the captured CUDA
  graph and run `run_forward_imperative` instead (host-side top-k can't
  be captured into a graph).

Constraint relaxations:
- Dropped the over-eager `d_kv % 256 == 0` check.  Q4_K alignment only
  matters on the input dimension of a matmul (`d_model` for Q/K/V/O/LM,
  `d_ff` for down).  Output dims are launch-per-row; no alignment needed.
- Added a separate `d_ff % 256 == 0` check (was implicitly required
  by the dense fused FFN kernel; now explicit).

Verification on Qwen2.5-7B regression: `qwen25_paged_parity` still
passes bit-identically [358, 2776, 264, 220, 17, 20, 4666, 6284] — the
MoE branches never fire when n_experts=0 / w_router=0.

REMAINING MULTI-SESSION FRs:
- `FR-17-extra-mla-fwd` (deepseek2 attention): KV is projected into a
  low-dim latent space, decompressed via a separate `attn_kv_a` +
  `attn_kv_b` tensor pair.  Needs a new attention kernel with the
  latent-decompression step.  MoE FFN half already shipped.
- `FR-17-extra-gemma-fwd` (gemma3): three blockers compounding —
    (1) head_dim=168 needs a per_lane-flexible attention kernel
        (current kernel assumes head_dim is a multiple of 32).
    (2) Sliding-window attention scope (only attend to last N positions).
    (3) Gemma3-specific pre+post-attention RMSNorm layout (vs Qwen's
        single attn_norm).
  Each tractable individually; the combined forward path is its own
  refactor.

Both blocked on >8 GB GPU for end-to-end verification anyway — the cnc
2×P100 box (32 GB each) is the natural next target.

### 2026-05-25 evening (Qwen3-8B end-to-end)
**Qwen3-8B running end-to-end through Aether — first non-Qwen2.5 model.**

Per-arch dispatch for `qwen3` shipped AND verified on real GPU.  Output
of `aether-serve --gguf <Qwen3-8B>` + curl POST:

```
[aether-serve] loading GGUF: ...sha256-a3de86cd...
[QwenSession] arch=qwen3 layers=36 d_model=4096 heads_q=32 heads_kv=8
              head_dim=128 d_ff=12288 vocab=151936 rope=1000000 eps=1e-6
[aether-serve] model loaded in 3.39s
POST /v1/chat/completions {"prompt_ids":[9707,11,1879,0],"max_tokens":8}
→ "content":" I'm a new user. How can"
→ 11.41 tok/s
```

Components landed this turn:

(1) **F16 weight loading + matmul kernel** (FR-17-extra-f16-fwd).  Qwen3
    GGUFs store some tensors (V projections, in this case) as raw F16
    (dtype=1, 2 bytes/elem) while the rest are Q4_K_M.
    `upload_tensor_u8` now accepts dtype 1 alongside 12/14 and copies
    n_elems × 2 raw bytes to the device.
    New CUDA kernel `fused_f16_matmul_seq1` in KERNEL_SRC does the dot
    product against F16 weights via `aether_f16_to_f32_dev` per element,
    256-thread CTAs with warp-reduce + cross-warp reduce.

(2) **Per-tensor dtype dispatch** (FR-17-extra-mixed-quant).
    `BlockGpu` gained `dt_q/dt_k/dt_v/dt_o/dt_gate/dt_up/dt_down` fields.
    New helper `dispatch_matmul(x, w, dt, y, n_out, n_in)` routes by
    dtype to the right kernel (Q4_K→fused_q4k, Q6_K→fused_q6k,
    F16→fused_f16).  All Q/K/V/O/down matmuls + LM head now route
    through this helper.  Mixed-precision GGUFs (any combination of
    Q4_K + Q6_K + F16 weights across the block tensors) work as long as
    gate+up are both Q4_K (the fused gate_up_silu_mul kernel is still
    Q4_K-only, panics otherwise — a non-fused fallback is the next stage).

(3) **Qwen3 arch dispatch** (FR-17-extra-qwen3-fwd).
    `BlockGpu` gained `attn_q_norm_g / attn_k_norm_g` (Option<i64> as 0/non-0).
    `load_block` reads `attn_q_norm.weight` + `attn_k_norm.weight` via
    `upload_f32_tensor_opt` (returns 0 if absent; qwen2 has none).
    `block_forward_devarg` applies per-head RMS norm to Q and K when the
    gain tensors are present (rows=n_heads, d=head_dim).  Skips
    `bias_add` for Q/K/V when the bias tensor is absent (qwen3 has none).
    Llama-style models (no biases, no Q/K norm) are now automatically
    supported by this same dispatch.

Regression checks (RTX 3070 Ti):
- `qwen25_paged_parity` and `qwen25_multitenant_pool` both still pass
  bit-identically.  Qwen2.5-7B's tokens unchanged.
- The F16 dispatch branch only activates when `dt == 1`; Qwen2.5 GGUFs
  hit `dt == 12 || 14` and continue using the original fused kernels.

**Architecture compatibility status after this commit:**

| Model | Arch | Status |
|---|---|---|
| Qwen2.5-7B | qwen2 | ✅ baseline working |
| Qwen2.5-14B | qwen2 | ✅ would work (not local to test) |
| **Qwen3-8B** | qwen3 | **✅ generating real text @ 11 tok/s on 3070 Ti** |
| Qwen3-VL | qwen3vl | ⚠ qwen3 LLM works; vision tower path needed |
| Llama (any size) | llama | ✅ should work (no biases/norms — handled by optional fields) |
| Qwen3-Coder 30B | qwen3moe | ⚠ MoE FFN kernel needed |
| DeepSeek-Coder-V2 | deepseek2 | ⚠ MLA attention + MoE |
| Gemma3 27B | gemma3 | ❌ head_dim=168 + sliding-window |

### 2026-05-25 (Bigger-models foundation: runtime ModelConfig)
**Bigger-models foundation LANDED: runtime ModelConfig from
GGUF metadata + multi-arch probe.**  Every hardcoded Qwen2.5-7B
dimension in `runtime/src/serving.rs` is now a runtime parameter read
from the GGUF metadata at session construction.

**New: `ModelConfig::from_gguf(handle)`** parses `general.architecture`
+ the `<arch>.embedding_length`, `<arch>.block_count`,
`<arch>.attention.head_count`, `<arch>.attention.head_count_kv`,
`<arch>.feed_forward_length`, `<arch>.rope.freq_base`,
`<arch>.attention.layer_norm_rms_epsilon` keys.  Falls back to 7B
defaults for any missing key.

**Refactor surface:**
- `block_forward_devarg` now takes `cfg: &ModelConfig, max_seq: usize`
  and threads every kernel-launch dim through it.
- `SharedKvPool::new_for_shape(n_blocks, block_size, n_layers, d_kv)`
  sizes per-layer pools from runtime shape; legacy `new(...)` shortcut
  preserved for Qwen2.5-7B callers.
- Every `aether_dev_alloc_f32(D_MODEL)` / `(D_KV)` / `(D_FF)` /
  `(VOCAB)` etc. inside `new_with_mode` swapped to `cfg.*`.
- `dequant_embd_row`, `decode_step`, `prefill`, `capture_graph_now` all
  now use `self.cfg.*`.

**Kernel constraints enforced at load time** (rejects with a clear error
instead of producing NaN):
- `head_dim` must be multiple of 32 and ≤ 256 (attention_seq1's
  `per_lane = head_dim >> 5` layout caps out at 8 slots × 32 lanes).
- `n_q_heads` must be divisible by `n_kv_heads` (GQA invariant).
- `d_model` and `d_kv` must be multiples of 256 (Q4_K super-block).

**Regression check:** Qwen2.5-7B bit-identical to pre-refactor —
`qwen25_paged_parity` + `qwen25_multitenant_pool` tests both green.

**`aether-serve --probe --gguf <path>`** opens any GGUF, prints the
detected ModelConfig + shape-constraint + arch-compat status, exits.
Used to enumerate locally-available models:

| Model | Arch | Layers | d_model | KV | head_dim | Status |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | `qwen2` | 28 | 3584 | 4 | 128 | ✅ ready |
| Qwen2.5-14B (not local) | `qwen2` | 48 | 5120 | 8 | 128 | ✅ would load with same kernels |
| Qwen3-8B | `qwen3` | 36 | 4096 | 8 | 128 | ⚠ Q/K RMS norm tensors not yet loaded |
| Qwen3-VL | `qwen3vl` | 36 | 4096 | 8 | 128 | ⚠ qwen3 changes + vision tower |
| Qwen3-Coder 30B | `qwen3moe` | 48 | 2048 | 4 | 64 | ⚠ MoE FFN — needs router + per-expert dispatch |
| DeepSeek-Coder-V2 16B | `deepseek2` | 27 | 2048 | 16 | 128 | ⚠ MLA attention + MoE FFN |
| Gemma3 27B | `gemma3` | 62 | 5376 | 16 | 168 | ❌ head_dim=168 fails constraint; needs kernel re-derive |

**Bottom line:** the runtime-shape refactor unblocks any **Qwen2.5
variant** (14B, 32B) — same kernels, different dims.  Per-arch work
needed for everything else:
- `qwen3*` → load + plumb the per-head Q/K RMS norm tensors
- `*moe` → MoE FFN kernel (gate_exps/up_exps/down_exps + token routing)
- `deepseek2` → MLA attention kernel (latent KV projection)
- `gemma3` → relax head_dim constraint or add per-arch kernel variants

Filing each as separate FRs.

### 2026-05-24 night (Batched-forward stage 1)
**Batched-forward stage 1 LANDED + bigger-models scope captured.**

**Stage 1 — batched paged attention kernel.**
`runtime/src/cuda.rs::batched_paged_attention_seqB_devarg` fans across
`(blockIdx.x = head, blockIdx.y = request_idx)` so one launch processes B
concurrent requests against the shared K/V pool, with each request
indexing its own row of `page_table_batch` (row stride = a runtime arg).
All B requests share `cur_seq` (synchronous batched step).

`runtime/tests/cuda_batched_paged_attention_parity.rs` verifies bit-
identical output to running `paged_attention_seq1_devarg` B times
sequentially: **`max_abs_diff = 0.000e0`** for both requests with
DIFFERENT page tables (request A maps to physical blocks [0, 1],
request B maps to [3, 5] within the same shared pool).  Passes on RTX
3070 Ti.

**What remains for the full BatchedQwenSession.**  The attention kernel
is the most-complex piece and is now done; the other batched kernels
needed are:

| Op | Status | Notes |
|---|---|---|
| `paged_attention_seqB` | ✅ shipped this commit | B queries, B page tables, 1 launch |
| `rms_norm_fwd` | ⏳ | already takes `rows`; just FFI plumbing for `rows=B` |
| `rope_apply_devarg` | ⏳ | already takes `seq`; FFI plumbing for `seq=B` |
| `paged_append_kv_seqB` | ⏳ | similar shape — B (k_new, v_new) pairs against B page tables |
| `bias_add_f32`, `add_inplace_f32` | ⏳ | already row-loop-friendly |
| `fused_q4k_matmul_seqB_v2` | ⏳ **biggest piece** | needs the fused dequant-+-matmul kernel re-derived for B rows; dequant becomes amortized across batch |
| `fused_q6k_matmul_seqB_v2` | ⏳ | mirrors Q4_K path |
| `fused_q4k_ffn_gate_up_silu_mul_seqB` | ⏳ | gate+up+SiLU+mul fused, batch dim |
| LM head `fused_q4k|q6k_matmul` | ⏳ | shared shape with the Q/K/V projections |

The matmul batching is the hard part — the existing seq=1 fused kernels
are dense.  Estimated work: 1-2 weeks for a clean BatchedQwenSession that
runs Qwen2.5-7B B-batched with the same correctness/perf as the seq=1
contiguous path.  The architectural design + the attention kernel + the
shared-pool foundation are now all in place; the remaining lift is
mostly kernel-by-kernel.

**Bigger models — refactor scope captured.**
Per the OPENCLAW_FR.md operator-side audit (consumer-side concerns
section, dated 2026-05-23), the next deployment shapes are:

- **Qwen2.5-14B**: 48 decoder blocks (vs 28 for 7B); other shape dims
  potentially different.  Blocks on FR-17-extra-14b-e2e.  Per-block
  Q4_K/Q6_K dtype dispatch may need re-enumeration; the 7B path
  encountered a NaN-at-block-3 regression that was root-caused to
  exactly this mixed-precision pattern (per
  `memory/qwen25_q4km_mixed_precision_per_block_dtype.md`) — 14B will
  have its own per-block dtype pattern that needs verification.
- **Qwen2.5-32B-Q3_K_M**: even larger; will likely need
  tensor-parallel inference (FR-18-extra-tp-infer-bench) since 32B Q3_K_M
  weights exceed a single 3070 Ti's 8 GB by ~2-3x.
- **bge-large (embeddings)**: BERT-style encoder, not decoder; requires
  FR-17-extra-bert-fwd (different forward shape — no autoregressive
  KV cache, single-pass over full sequence).  Plus
  FR-19-extra-embed-endpoint for the `/v1/embeddings` OpenAI route.

The common pre-requisite for all three is **generalizing QwenSession's
hardcoded constants** (`N_LAYERS=28`, `D_MODEL=3584`, `N_Q_HEADS=28`,
`N_KV_HEADS=4`, `HEAD_DIM=128`, `D_FF=18944`, `VOCAB=152064`) into
runtime parameters read from the GGUF metadata.  Today these are `const`
in `serving.rs:57-67`; making them runtime values is a moderate refactor
that touches every kernel-launch site.  Subsequently, per-arch dispatch
(Qwen2.5 vs Llama vs BERT) requires either a per-arch session type or
a config-driven forward pass.

Filing these as future-session FRs.  The deployment surface as it
stands today serves Qwen2.5-7B; bigger and other-shape models are
multi-session work.

**aether-serve from this commit onward** exposes the batched attention
kernel via its FFI but does NOT yet route through it (BatchedQwenSession
is the missing layer).  Single-tenant single-request decode unchanged
at 30-34 tok/s on RTX 3070 Ti.

### 2026-05-24 evening (SharedKvPool multi-tenancy)
**SharedKvPool multi-tenancy LANDED — true vLLM-shape foundation.** `runtime/src/serving.rs::SharedKvPool` owns per-layer
(`N_LAYERS × 2 = 56`) GPU buffers + a free-block bitmap.
`Arc<SharedKvPool>` shared across multiple `QwenSession`s on the same
model; each session binds via `QwenSession::new_paged_with_pool(path, pool)`,
holds its own `page_table_dev` mapping logical block index -> physical
block id, and grows the page table dynamically on token-boundary
crossings (`ensure_block_for_position`).  `Drop` returns the session's
blocks to the pool's free list.

Multi-tenant parity verified on RTX 3070 Ti — `runtime/tests/qwen25_multitenant_pool.rs`
creates ONE pool (32 blocks × 4 tokens = 128 token capacity), binds TWO
PagedQwenSessions to it, runs different prompts through each, and
asserts each session's token IDs match a single-tenant `QwenSession::new`
reference exactly:
- session A (`[9707,11,1879,0]`): `[358, 2776, 264, 220, 17, 20]` ✓
- session B (`[40,1079,264,220,17]`): `[19, 1042, 2310, 8593, 13, 358]` ✓
Pool tracking: 0 → 2 (init) → 6 (grew during decode) → 0 (reclaimed on Drop).

aether-serve wired with `--pool-blocks N` CLI flag (implies `--paged`).
End-to-end TLS + h2-ALPN + shared pool + real Qwen verified:
```
$ aether-serve --tls --pool-blocks 32 --gguf <Qwen2.5-7B> --warmup 4
[aether-serve] allocating SharedKvPool: 32 blocks × 4 tokens = 128 token capacity
[aether-serve] model loaded in 2.17s (paged KV mode)
...
POST /v1/chat/completions -> "content":" I'm a 25-year-old"
```

The remaining "true vLLM-shape" future work — a single batched-forward
kernel that decodes N concurrent requests in ONE launch (vs the current
serialize-on-Mutex pattern) — needs a rewrite of every kernel in
QwenSession to accept a batch dimension.  That's a 1-2 week project; the
SharedKvPool foundation it builds on top of is now in place.

### 2026-05-24 (ALL deferred polish items LANDED)
**ALL deferred polish items LANDED.  aether-hosting is fully
production-shaped.**  Three pieces flipped from "deferred" to "shipped":

(1) **ALPN extension in TLS 1.3** (RFC 7301).  `TlsServerSession::new_with_alpn`
    accepts a supported-protocols list; ClientHello parser extracts the
    client's ALPN list (ext 0x0010); ServerHello path moved to put the chosen
    protocol in **EncryptedExtensions** (per RFC 8446 §4.4); negotiated value
    surfaced via `negotiated_alpn()` on both server and client sides.
    Verified with `openssl s_client -alpn h2,http/1.1` -> `ALPN protocol: h2`;
    `-alpn http/1.1` -> `ALPN protocol: http/1.1`.  3 loopback tests:
    `tls13_alpn_negotiation` (server-pref-wins), `tls13_alpn_h2_only`,
    `tls13_alpn_no_overlap` (no extension emitted -> both sides None).
    aether-serve's `TlsStream::accept` now advertises `["h2","http/1.1"]`,
    correctly negotiating h2 for HTTP/2-capable clients.

(2) **Paged-KV inside QwenSession** (FR-19.4-extra-deep).  `QwenSession::new_paged`
    swaps the contiguous K/V cache for the paged kernels
    (`paged_append_kv_devarg`, `paged_attention_seq1_devarg`) with an
    identity-mapping page table (block_size=4, n_blocks=MAX_SEQ/4=8).
    Parity test `runtime/tests/qwen25_paged_parity.rs` loads the real
    Qwen2.5-7B GGUF twice — once contiguous, once paged — runs the same
    `[9707,11,1879,0]` prompt + 8 max_tokens through each, and asserts
    identical token IDs.  Verified on RTX 3070 Ti:
    - contiguous: `[358, 2776, 264, 220, 17, 20, 4666, 6284]` in 0.389s
    - paged:      `[358, 2776, 264, 220, 17, 20, 4666, 6284]` in 0.387s
    Bit-identical at the token-ID level across the full forward chain
    (RMS norm + Q/K/V matmul + RoPE + paged append_kv + paged
    attention_seq1 + O matmul + residual + FFN + LM head + argmax).
    Speed within noise — the paged kernels add one int load per K/V row
    which amortizes against the matmul cost.

(3) **TlsClientSession cert chain verifier**.  `TlsClientSession::new_full`
    takes a list of trusted Ed25519 SPKI pubkeys.  When non-empty, the
    flight processor (a) requires the server's leaf cert SPKI to match
    one of them, AND (b) calls `verify_self_signed_ed25519_cert(cert_der)`
    which extracts the TBS DER + signature, walks to SPKI, runs
    `aether_ed25519_verify(spki_pubkey, tbs_der, sig)`, and confirms the
    cert is signed by its own SPKI key.  Two loopback tests:
    `tls13_trust_anchor_positive` (correct anchor -> handshake succeeds),
    `tls13_trust_anchor_negative` (bogus anchor -> client rejects with
    "server cert pubkey not in trust anchors").  Empty trust_anchors list
    = trust-on-first-use (the prior behavior).

**Aether-hosting is now production-complete.**  One binary, one process,
multi-protocol (HTTP/1.1 + HTTP/2 via ALPN-negotiated h2 or h2c-preface
auto-detect), TLS 1.3 with from-scratch crypto + handshake + ALPN, multi-
threaded accept, real Qwen2.5-7B forward through CUDA-graph-captured paged
or contiguous KV (bit-identical), and outbound-TLS cert validation via
caller-configured trust anchors.

### 2026-05-23 night (Aether-hosting deployment COMPLETE + HTTP/2 (h2c) wired)
**Aether-hosting deployment COMPLETE + HTTP/2 (h2c) wired.**
On top of the multi-threaded TLS 1.3 + HTTP/1.1 hosting landed earlier, the
aether-serve binary now AUTO-DETECTS HTTP/2 (h2c, prior-knowledge mode) on
each incoming connection by peeking the 24-byte preface
`PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n`.  Both protocols are served from the
same listener:

- HTTP/1.1: handled by the existing `handle_request` -> JSON parser flow.
- HTTP/2:   handled by new `handle_request_h2`:
  * Sends server SETTINGS, ACKs client SETTINGS
  * HEADERS frame -> HPACK decode (literal + static-table-indexed forms)
    -> dispatch by :method + :path
  * DATA frame -> accumulate body bytes
  * PING -> echo as ACK
  * Response written as HEADERS+DATA frames with status code via the
    static table.

Verified via `scripts/h2c_smoke.py` (hand-rolled 200-line Python h2c
client, no external deps):

```
$ python scripts/h2c_smoke.py 18600 GET /health
  -> :status 200, DATA "ok"
$ python scripts/h2c_smoke.py 18600 GET /v1/models
  -> :status 200, DATA {"object":"list","data":[{"id":"qwen2.5-7b-instruct",...}]}
$ python scripts/h2c_smoke.py 18600 POST /v1/chat/completions '{"prompt_ids":[1,2,3,4],"max_tokens":4}'
  -> :status 200, DATA <OpenAI completion JSON>
```

Items deferred for future sessions (polish, not blocking):
- ALPN extension in `tls13` to negotiate h2-over-TLS during the
  handshake (today h2 works over plaintext TCP, h1 works over TLS).
- Paged-KV swap inside QwenSession (kernels parity-verified;
  integration needs a batched-forward kernel to materialise the
  multi-tenant decode win).
- TlsClientSession cert chain / root store.

The hosting story as-shipped: ONE binary, ONE process, ZERO external
server libraries.  Listens on a port; auto-dispatches per-connection
between HTTP/1.1 and HTTP/2 (prior-knowledge); under both, TLS 1.3 is
terminated server-side by Aether's own from-scratch crypto + handshake
state machine; per-request decoding through CUDA-graph-captured Qwen2.5
forward at 23 tok/s warm; OpenAI-shaped JSON returned.

**Aether-hosting: complete.**

### 2026-05-23 night earlier (Aether-hosting baseline — multi-threaded TLS + real LLM)
**Aether-hosting deployment COMPLETE.** Multi-threaded
accept in aether-serve: thread per accept, `ServerState` in `Arc`, QwenSession
mutex serializes the actual GPU decode (one forward pass at a time on the
single GPU) while TLS handshakes, HTTP parsing, BPE decode, and JSON
rendering happen concurrently across threads.  Verified end-to-end:

**Single-request real-LLM via TLS:**
- `target/release/aether-serve.exe --tls --port 18566 --gguf <path> --warmup 4`
- Real Qwen2.5-7B model loads (3.28s GGUF parse), warmup captures CUDA graph (0.31s)
- `openssl s_client -tls1_3 -ciphersuites TLS_CHACHA20_POLY1305_SHA256 -groups X25519 -sigalgs ed25519`
  POSTs `/v1/chat/completions` with `prompt_ids=[9707,11,1879,0]`
- Server decrypts the request through `TlsServerSession.feed`, runs HTTP
  parser on the cleartext, decodes 28 real tokens through the
  CUDA-graph-captured Qwen forward at **23.17 tok/s warm**, renders OpenAI
  JSON, encrypts + sends back via `TlsServerSession.send_app_data`.
- Client receives: `"content":" I'm a 25-year-old software developer from
  the UK, and I'm currently working on a project that involves creating a
  web application"`.

**3 concurrent real-LLM TLS clients (different prompts):**
- prompt A (`[9707,11,1879,0]`): `"content":" I'm a 25"`
- prompt B (`[1939,279,16164]`): `"content":" function is not working as expected"`
- prompt C (`[4340,374,12541]`): `"content":" a sentence?\\n\\n\\\""`
- First request 10.87 tok/s (cold GPU clock ramp), next two 20.10 / 20.21
  tok/s as the GPU stayed warm.  All three handshakes + HTTP parsing
  happened in parallel; the 3 forward passes serialize on the QwenSession
  mutex.

The complete deployment chain — TCP → TLS 1.3 (Aether's own crypto +
handshake state machine) → HTTP/1.1 (Aether's own parser) → JSON body
parse → BPE prompt-id ingestion → CUDA forward (Aether's runtime via
cudarc → cuBLAS + nvrtc-compiled paged-ready kernels) → BPE detokenize →
OpenAI-shaped JSON → TLS 1.3 record-layer seal → TCP — runs as ONE binary,
ONE process, single self-hosted runtime, zero external server libraries.

Items left for future sessions (polish, not blocking):
- HTTP/2 server-side wire (h2c preface + SETTINGS handshake; the frame codec
  + HPACK + Huffman decoder already exist in `runtime/src/http2.rs`).
  Requires ALPN in `tls13` to negotiate h2 over TLS.
- Wiring the paged-KV kernels into `QwenSession` as a swappable backend.
  The kernels are parity-verified vs the contiguous reference on RTX 3070 Ti
  (`cuda_paged_kv_parity.rs`); for single-request serving the contiguous
  path is faster, the real payoff requires also building a batched-forward
  kernel that processes N requests per launch.
- Cert chain / root store for outbound HTTPS (`TlsClientSession` validates
  self-signed only today).

### 2026-05-23 late evening (Real continuous-batching scheduler — FR-19.5-extra)
**Real continuous-batching scheduler — FR-19.5-extra.**
Pool-aware `BatchScheduler` in `runtime/src/lib.rs` that owns per-request page
tables on top of the FR-19.4-extra paged KV primitives.  FFI:
`aether_batch_sched_{new, destroy, admit, record_token, finish, reap,
n_active, pagetable_for, position}` — all returning i64 to dodge the
asm-backend i32 sign-extend gap.  Admit allocates the initial physical
blocks the prompt needs + a page table; record_token advances the position
and lazily allocates a new block when the next write crosses a block
boundary; reap collects finished requests and returns their blocks to the
pool.  Witness `tests/runtime/batch_scheduler.aether` exits 42 — admits
2 requests, exhausts capacity (3rd rejected), records 4 tokens through
req=10, reaps it back to the pool, ends with 1 active and 1 allocated
block (matching req=11's remaining state).

### 2026-05-23 evening (Real paged KV cache landing — FR-19.4-extra)
**Real paged KV cache landing — FR-19.4-extra.** Built on top of
the existing CPU-side block allocator (`aether_pkv_*` from FR-19.4): added per-request
**virtual page table** primitive (`aether_pkv_pagetable_*`) plus two new **GPU
kernels** (`paged_append_kv_devarg`, `paged_attention_seq1_devarg`) that walk the
page table on every K/V read and write.  Kernels compiled in a SEPARATE nvrtc
unit (`aether_paged_kernels`) so they don't apply register-allocator pressure
on the main `KERNEL_SRC` kernels (per the `nvrtc_kernel_unit_pressure.md`
memory).  Parity test (`runtime/tests/cuda_paged_kv_parity.rs`) runs on the
RTX 3070 Ti and proves:
1. With **identity page-table mapping** the paged kernels produce
   bit-identical output (`max_abs_diff < 1e-5`) to the contiguous
   `append_kv_devarg` + `attention_seq1_devarg` reference, across the full
   Qwen2.5 shape (n_q_heads=28, n_kv_heads=4, head_dim=128, cur_seq=7).
2. With a **permuted page table** (`logical [0,1] → physical [2,0]`) the
   attention output is unchanged — proving the kernel correctly walks the
   indirection rather than reading the raw pool layout.
`.aether` audit witness `tests/runtime/paged_kv_pagetable.aether` exercises
the page-table FFI surface (new/set/get/len/destroy + i64-returning get to
sidestep the asm-backend i32-sign-extend gap per
`asm_backend_known_gaps.md`).  Audit clean.  Integration into `QwenSession`
deferred — single-request serving doesn't benefit from page tables, the
payoff is when paged KV combines with the continuous-batching scheduler
(real implementation is FR-19.5-extra).

### 2026-05-23 morning (Path D landing — TLS 1.3 + HTTP/2 + aether-serve HTTPS)
**Path D landing — full TLS 1.3 + HTTP/2 + aether-serve HTTPS interop.**
Pickup from "go on 1" → completed every tractable Path D item.  Highlights:
(1) **aether-serve has a working `--tls` mode**: generates fresh Ed25519 + X25519
on startup via `aether_random_bytes` (BCryptGenRandom on Windows), self-signs
an X.509 cert, drives a `TlsStream` adapter over each accepted TCP socket,
runs HTTP/1.1 on the decrypted bytes.  GET /health, GET /v1/models, and POST
/v1/chat/completions all round-trip through TLS with `openssl s_client`
(RFC-conformant interop verified end-to-end — `Cipher is TLS_CHACHA20_POLY1305_SHA256`
+ `Peer Temp Key: X25519` + cert chain validated + real HTTP/1.1 response body
decrypted).  Schannel-backed curl on Windows refuses Ed25519 sigs
(`SEC_E_ALGORITHM_MISMATCH`) — a known Windows Schannel limitation, not a
server bug.
(2) **TlsClientSession** is now a public streaming state machine (mirrors
TlsServerSession), plus FFI (`aether_tls13_client_*`).  Loopback test
`tls13_client_session_loopback` drives a full handshake + 2 app-data records
through both `TlsServerSession` and `TlsClientSession` end-to-end.
(3) **HTTP/2 + HPACK shipped** (`runtime/src/http2.rs`, ~600 LOC).  Frame
codec (DATA, HEADERS, SETTINGS, WINDOW_UPDATE, PING, GOAWAY, RST_STREAM),
HPACK static table (61 entries), HPACK integer codec, Huffman DECODER
(RFC 7541 Appendix B — full 257-symbol table; decodes
`www.example.com` per §C.4.1).  6 unit tests pass; `.aether` witness
`tests/runtime/http2_hpack.aether` exits 42 via
`aether_http2_self_loopback_smoke`.
(4) **TLS 1.3 record layer made streaming-tolerant** — `TlsServerSession.feed`
now buffers partial records across calls (essential for real TCP delivery
which doesn't honor record boundaries).
(5) **Caught a real Poly1305 bug en route** — RFC 7539 §2.8.2 vector test
exposed unmasked spurious bits in the 5×26→4×32 limb conversion at MAC
finalization (round-trip tests pass because the bug was self-consistent).
Fix: mask each `h_full_i` to 32 bits before the final s-addition.  Without
this fix, openssl rejects with `bad_record_mac` on the first encrypted
record.
(6) **Real RNG primitive** `aether_random_bytes` shipped (BCryptGenRandom
on Windows, `/dev/urandom` on Unix).  No external deps.
Items DEFERRED (multi-session): real GPU-backed paged KV cache + continuous
batching require kernel changes to attention to walk page tables at seq>1
reads; the existing FR-19.4/FR-19.5 simulations stay closed.  Audit 169/196
unchanged (all this is FR-x-extra per the parent-tag convention).)

### 2026-05-22 evening (prior — TLS 1.3 handshake landing)
**TLS 1.3 server-side handshake state machine SHIPPED — FR-19.1-extra.**
`runtime/src/tls13.rs` implements RFC 8446 server end-to-end on top of Aether's
from-scratch crypto: TLSPlaintext/TLSCiphertext record layer with per-record
nonce derivation (IV XOR seq big-endian) + AAD construction, key schedule
(Early/Handshake/Master + 4 traffic secrets per §7.1), handshake message
codecs (parse ClientHello, encode ServerHello/EncryptedExtensions/Certificate/
CertificateVerify/Finished, parse client Finished), minimal X.509 self-signed
Ed25519 cert generator (OID 1.3.101.112, full DER tree), and a
`TlsServerSession` state-machine driver. Profile: TLS_CHACHA20_POLY1305_SHA256
+ X25519 + Ed25519 only; no PSK / 0-RTT / resumption / ALPN. Witness:
`runtime/tests/tls13_loopback.rs` (in-process server + matching client; full
handshake + 2 app-data records round-trip; tamper test rejects flipped server
flight). `.aether` audit witness: `tests/runtime/tls13_handshake.aether`
exits 42 via `aether_tls13_self_loopback_smoke`. FFI surface:
`aether_tls13_server_new/feed/take_outbound/send/is_done/free`. Audit clean.

### 2026-05-22 morning (prior in same day)
**aether-serve generates real Qwen text via HTTP**. End-to-end Qwen2.5-7B hosting on localhost: `aether-serve --gguf <path> --warmup 6` → curl POST `/v1/chat/completions` returns OpenAI-shaped JSON with real generated text at 24 tok/s warm (capped by GPU power state on bursty single-request workloads; same kernels reach 37 tok/s warm in the sustained `qwen25_graph_decode` bench). SSE streaming (`"stream":true`) emits per-token delta chunks. Plus FR-19.1-extra crypto primitives (SHA-256/512, HMAC, HKDF, X25519, Ed25519) — the foundation underneath the TLS 1.3 handshake shipped this evening.

### 2026-05-21 (prior session)
**37.22 tok/s warm** = 124% of llama.cpp on RTX 3070 Ti / Qwen2.5-7B. Standing wins from that session: per-block Q4_K/Q6_K dtype dispatch (NaN fix), fused FFN kernel (gate+up+silu+mul in 1 launch), CUDA graph capture for autoregressive decode (+37% throughput). Investigated speculative decoding (theoretical 1.6-2.6x but needs ~7-8 days of seq>1 kernel work; deferred). Path E step 11 attempt (self-host compiler if/else) blocked on Aether-compiler 8-arg fn bug — documented in docs/PATH_E_STATUS.md.

## Project Status
🟢 **Audit: 169/196 (86%) — 10 of 19 phases at 100%**. matt-voice's
serving-deploy critical path within Aether's language + runtime is
materially complete: GGUF reader + Q4_K_M dequant + cuda routing
live, tokenizer.json + chat_template loaders, SafeTensors multi-
tensor parser, **TLS 1.3 server-side handshake state machine LANDED
this evening (FR-19.1-extra, on top of the crypto primitives from
this morning's session)**. Remaining gates: HTTP/2 + HPACK, real-
world TLS interop (vs curl/openssl), real RNG, cert-chain verifier
(for client-side), and hardware-binding NCCL on cnc 2× P100.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  ← +3 (FR-15.{1,2,3} earlier today)
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   20/20 witnessed (100%) ← closed today
Phase 18:    9/11 witnessed (81%)  ← +7 (only hardware-blocked remain)
Phase 19:   16/16 witnessed (100%) ← closed today
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    169/196 (86%)
```

Workspace tests: **61 lib tests pass** (up from 59 — +2 crypto: ed25519_rfc8032_test_1, sha512_known_vectors), 134+ workspace integration tests still pass.
Honesty scan: 0 todo / 0 unimplemented / 4 known-OK stubs (unchanged).

## What Was Done This Session (2026-05-22 evening, TLS 1.3 handshake)

User direction: "go on 1" — Path D, full TLS 1.3 handshake state machine
on top of the prior session's RFC-verified crypto primitives.

### `runtime/src/tls13.rs` — full TLS 1.3 server, no external deps

Single-file module (~1100 LOC) implementing RFC 8446 server profile:

| Layer | What it does |
|---|---|
| `record::*` | TLSPlaintext / TLSCiphertext codec; `build_nonce(iv, seq)` = IV XOR right-aligned seq big-endian; `build_aad` = type(0x17) ‖ 0x0303 ‖ u16 len; `seal_record` / `open_record` thin wrappers over `aead_chacha20_poly1305_{seal,open}` (newly exposed Rust API in lib.rs); `parse_plaintext` for cleartext records (ClientHello, ServerHello). |
| `key_schedule::Schedule` | Early Secret = HKDF-Extract(0, 0); Handshake Secret = HKDF-Extract(Derive-Secret(es, "derived", ""), DHE); Master Secret = HKDF-Extract(Derive-Secret(hs, "derived", ""), 0). `derive_secret(secret, label, transcript)` = HKDF-Expand-Label(secret, label, SHA-256(transcript), 32). Per-direction handshake + app traffic secrets all derived per §7.1. `traffic_keys(secret)` = HKDF-Expand-Label(., "key", 32) + (., "iv", 12). `finished_mac(base_key, transcript)` = HMAC-SHA256(HKDF-Expand-Label(base_key, "finished", 32), SHA-256(transcript)). |
| `handshake::*` | `wrap(msg_type, body)` adds the 4-byte (type ‖ u24 len) header. `parse_client_hello` walks `legacy_version=0x0303 ‖ random[32] ‖ sid<u8> ‖ cipher_suites<u16> ‖ comp<u8> ‖ extensions<u16>`; extracts supported_versions (0x002b), supported_groups (0x000a), signature_algorithms (0x000d), key_share (0x0033 — X25519 entry). `encode_server_hello` emits the same with supported_versions(=tls 1.3) + key_share(x25519, server_pub). `encode_encrypted_extensions` = `[0, 0]` (empty list). `encode_certificate` = `u8(0) ‖ u24(list_len) ‖ u24(cert_len) ‖ cert_der ‖ u16(0)`. `encode_certificate_verify` = `u16(sig_alg) ‖ u16(64) ‖ ed25519_sig`. `encode_finished(verify_data)` = the 32-byte VD. `server_cert_verify_signed_content(transcript)` = `0x20*64 ‖ "TLS 1.3, server CertificateVerify" ‖ 0x00 ‖ SHA-256(transcript)` per §4.4.3. Internal `Reader` for zero-copy parsing. |
| `x509::self_sign_ed25519(seed, cn, serial) -> (cert_der, pub_key)` | Builds TBSCertificate as SEQUENCE { version=v3, serial INTEGER, sigAlg, issuer (Name with CN), validity (UTCTime 240101000000Z..490101000000Z), subject (Name with CN), SPKI (Ed25519 OID 1.3.101.112 ‖ BIT STRING(0x00 ‖ pub_key)) } then signs TBS via `aether_ed25519_sign`, wraps Certificate = SEQUENCE { tbs, sigAlg, signature BIT STRING }. Returns DER bytes + the 32-byte pub key for the matching session. |
| `TlsServerSession` | State machine: `ExpectClientHello -> SentServerFlight -> Connected -> Closed`. `feed(input)` parses incoming records by current state; on CH it emits ServerHello (TLSPlaintext) + one encrypted record containing {EE, Cert, CV, Finished}; on encrypted Finished it verifies the MAC, transitions to Connected. `send_app_data(plaintext)` seals into a TLSCiphertext app-data record and appends to outbound. `take_outbound()` drains the response queue. CCS records tolerated and skipped (RFC 8446 §5). |
| `client_for_test::TestClient` | Matching client harness in the same file (used by tests, kept compiled because it's small and gated only by being inside `pub mod`). `build_client_hello_record` crafts a wire-correct ClientHello with our cipher suite/group/sig-alg/key_share. `process_server_flight_and_build_client_finished` parses ServerHello, derives DHE shared, decrypts the encrypted flight, extracts the Ed25519 SPKI from the Certificate, verifies the CertificateVerify signature against the transcript, recomputes server Finished MAC, and constructs the client's own Finished record. |
| FFI surface | `aether_tls13_server_new`, `_feed`, `_take_outbound`, `_send`, `_is_done`, `_free`, and `_self_loopback_smoke` (one-shot in-process loopback returning 42 on success). All `#[no_mangle] extern "C"`, callable from `.aether`. |

### Witnesses

`runtime/tests/tls13_loopback.rs` — two integration tests:
- `tls13_full_loopback` — CH (149 B) → server flight (503 B) → client Finished (58 B) → server `Connected` → 1 server→client + 1 client→server app-data record (37 B plaintext, 59 B on-the-wire each direction). All bytes verified.
- `tls13_tampered_server_flight_rejected` — bit-flip one byte inside the encrypted second record; client must reject (CV signature fail or MAC mismatch).

`tests/runtime/tls13_handshake.aether` — `// roadmap: P19.1`, calls `aether_tls13_self_loopback_smoke()` and forwards its exit code. Exits 42.

Audit: still 169/196 (FR-19.1-extra reuses the parent tag per the
`fr_x_extra_tag_convention.md` convention — depth-over-breadth).

### lib.rs surgery

- Made `chacha20_xor`, `poly1305_mac`, `poly1305_key_gen` `pub(crate)`.
- Added `aead_chacha20_poly1305_seal` and `aead_chacha20_poly1305_open` Rust
  helpers (mirror of the FFI functions but with Vec / Option return).
- Added `pub mod tls13;`.

### Architectural choices worth remembering

- **One encrypted record for the whole server flight**. EE‖Cert‖CV‖Finished
  concatenated as the inner plaintext of a single TLSCiphertext. RFC permits
  fragmentation; we don't, because everything fits within the 16 KB record limit.
- **No ChangeCipherSpec emit**. We tolerate-and-skip incoming CCS (some
  clients send one for middlebox compat) but don't emit one ourselves. RFC
  §5: "MAY send", we choose MAY-NOT.
- **No HelloRetryRequest**. ClientHello MUST come with an X25519 key_share in
  the first message. Real clients all do this for X25519 because the group
  is universally supported.
- **No padding** in TLSCiphertext.  RFC §5.4 makes padding optional.
- **DER builder is direct write** — no full ASN.1 lib. ~100 LOC.

### What's still missing on the D-path

In dependency order from "first to tackle next session":
1. **HTTP/2 framing + HPACK** (M-L). Independent of TLS. SETTINGS / HEADERS / DATA / WINDOW_UPDATE / PING + static HPACK table. h2c over plaintext TCP works as a first slice; h2-over-TLS slots on top of `TlsServerSession`.
2. **Real-world interop** (M). Drive a TLS connection from `curl --tlsv1.3` against `aether-serve` — currently the test loopback proves the bytes are correct against our own client implementation. A round-trip against curl/openssl/rustls would catch any wire-format drift.
3. **Wire `TlsServerSession` into `aether-serve`** (M). Today the binary uses plain HTTP/1.1. The clean integration: per-accept-on-listener spawn → wrap socket → drive TLS handshake → on Connected, plumb decrypted bytes into the existing HTTP/1.1 parser. Pure additive.
4. **TlsClientSession** (M). The client half (which we have inline in `client_for_test`) deserves a clean public API for outbound calls.
5. **Real RNG** (S). Today `server_random` / `x25519_priv` come from the caller; production needs `getrandom`-equivalent. Single-platform Win32 `BCryptGenRandom` works.
6. **Cert chain / root store** (L). Today: self-signed only. Real internet needs a trusted-root verifier (or a CA-signed leaf, with a manual root pre-loaded).
7. **Resumption / 0-RTT / PSK** (XL). Not blocking the D-path proof; multi-session.

The 1100 LOC of `tls13.rs` is the load-bearing core. Items 1-5 are each
similar weight or smaller. Path D is roughly half-done by code volume.

---

## What Was Done Prior in 2026-05-22 (morning — real-LLM hosting)

User direction: "lets try to closeout the rest of the hosting for real llama in aether"
+ "models downloaded now for smoke testing" + "in theory llama.cpp and aether are a switch flip away"
+ explicit scope: "Full Path D (TLS 1.3 + HTTP/2)".

### aether-serve now generates real Qwen2.5-7B text via OpenAI-compat HTTP

`trainer/src/bin/serve.rs` was previously a STUB that returned
`[aether-serve stub: integrate qwen25_autoregressive_cuda forward here]`.
This session wired the real forward chain + tokenizer + JSON + SSE
streaming. End state:

```
target/release/aether-serve.exe \
  --port 8181 \
  --model qwen2.5-7b-instruct \
  --gguf "C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7..." \
  --warmup 6

# In another shell:
curl -X POST http://localhost:8181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_ids":[9707,11,1879,0],"max_tokens":32}'

→ {"id":"chatcmpl-aether-serve-1","object":"chat.completion",
   "model":"qwen2.5-7b-instruct",
   "choices":[{"index":0,"message":{"role":"assistant",
     "content":" I'm a 25-year-old software developer from the UK,
                and I'm currently working on a project that involves
                creating a web application"},
     "finish_reason":"stop"}],
   "usage":{"prompt_tokens":4,"completion_tokens":28}}
```

Generated IDs `[358, 2776, 264, 220, 17, ...]` are BIT-IDENTICAL to
the prior session's reference `qwen25_graph_decode.rs` test.

Pieces shipped:
1. **`runtime/src/serving.rs`** — `QwenSession` struct factored from
   the reference impl. Owns 28 BlockGpu + KvCache + activation
   buffers + step_args + a captured CUDA graph + the loaded BPE
   tokenizer (vocab 152064, merges 151387) + EOS token id. Methods:
   `new(gguf_path)`, `prefill(&[ids])`, `decode_step(last_id)`,
   `generate(&prompt, n, stop)`, `decode_ids(&ids) → String`,
   `warmup(N)`. Drop frees all GPU handles.
2. **`runtime/src/serving.rs` decode_ids** — uses GPT-2 byte fixup
   (RFC convention `Ġ` ↔ 0x20 etc.) so generated bytes round-trip
   to readable text. Returns "" if tokenizer absent.
3. **`trainer/src/bin/serve.rs`** — JSON body parser (`prompt_ids`,
   `max_tokens`, `stream`), endpoint dispatch (`GET /health`,
   `GET /v1/models`, `POST /v1/chat/completions`,
   `POST /v1/completions`), JSON-escape for content, EOS auto-stop,
   `--warmup N` flag.
4. **SSE streaming** — when `stream:true`, response uses
   `Content-Type: text/event-stream` + chunked transfer encoding,
   emits one `data: {choices:[{delta:{content:"..."}}]}\n\n` per
   generated token, terminates with `data: [DONE]\n\n`. Verified
   via `curl -N`.
5. **`trainer` cuda feature** — new `cuda = ["aether_rt/cuda"]`
   passthrough; `cargo build --release -p trainer --features cuda`
   produces a release binary with cuBLAS + nvrtc routing.

### Performance reality check

The 37.22 tok/s figure from prior session was a SUSTAINED bench
that holds the GPU at P2/1950MHz across N decode steps. The serve
binary's bursty workload pattern (request → idle → request) lets
the GPU drop into P5/P8 power states between requests (210 MHz)
and the FIRST request after idle takes ~200 ms per token while
the GPU climbs back to P2.

Mitigation: `--warmup N` runs N synthetic decode steps on startup
to drive the GPU into P2 and pre-capture the graph. Subsequent
requests within a few seconds of the warmup (or each other) sustain
24-37 tok/s. Long-idle gaps still trigger the cold-clock penalty.

The right production fix is `nvidia-smi -pm 1` (persistence mode)
+ `nvidia-smi -ac <mem>,<gfx>` (lock clocks) — admin commands
outside Aether's scope. Documented but not enforced.

### FR-19.1-extra: TLS 1.3 crypto primitives (foundation)

Path D requires a full TLS 1.3 handshake stack from scratch
(per Aether's no-external-deps rule). This session shipped the
crypto foundation, all RFC-verified against test vectors:

| Primitive | Source | Test vector |
|---|---|---|
| SHA-256 | FIPS 180-4 | `abc`, `""`, 56-byte boundary |
| SHA-512 | FIPS 180-4 | `abc`, `""` |
| HMAC-SHA256 | RFC 2104 | RFC 4231 §4.2 test case 1 |
| HKDF-Extract | RFC 5869 | RFC 5869 §A.1 test case 1 |
| HKDF-Expand | RFC 5869 | RFC 5869 §A.1 test case 1 |
| HKDF-Expand-Label | RFC 8446 §7.1 | Shape + determinism smoke |
| X25519 (scalar_mult / derive_public / shared_secret) | RFC 7748 | §5.2 first vector + DH roundtrip |
| Ed25519 (derive_public / sign / verify) | RFC 8032 | §7.1 TEST 1 |

ABI surface (all `#[no_mangle] extern "C"`):
- `aether_sha256`, `aether_sha512`
- `aether_hmac_sha256`
- `aether_hkdf_extract`, `aether_hkdf_expand`, `aether_tls13_hkdf_expand_label`
- `aether_x25519_derive_public`, `aether_x25519_shared_secret`
- `aether_ed25519_derive_public`, `aether_ed25519_sign`, `aether_ed25519_verify`

Internal implementations: radix-2^51 field arithmetic over GF(2^255-19),
twisted-Edwards point ops (X, Y, Z, T extended coords),
double-and-add scalar mult (non-constant-time; FR-x-extra for ct),
sc_reduce via long-division (verify-only path; Barrett-form
constant-time is FR-x-extra). Curve constants (d, sqrt(-1), base
point) computed/derived at runtime from byte forms or self-checked
against `x^2 == -1`.

### Bugs caught in-flight

1. **X25519 fe_from_bytes over-masked limb 4 to 50 bits** — total
   read was 254 bits (off by 1 from input bit 254). Fix: read full
   51 bits for all 5 limbs (255 total; input bit 255 ignored implicitly).
2. **fe_pow_p_plus_3_over_8 final mul was by z instead of z^2** —
   gave z^(2^252 - 3) instead of z^(2^252 - 2). Fix: multiply by z2.
3. **Initial "compute sqrt(-1) via fe_pow_p_plus_3_over_8(-1)" returns 1**
   (since `(-1)^(even exponent) == 1`). Fix: use `2^((p-1)/4)` with
   a hardcoded constant + self-check at first access.
4. **Ed25519 verify subtracted k*A instead of adding** — sign was
   inverted in the negation logic. Fix: drop the negate step, just
   compute `R + k*A` directly.

### Decode-side text round-trip caveat

`aether_bpe_encode` (text → ids) is byte-level and DOES NOT match
HF's GPT-2 unicode-char-level BPE for Qwen. Decode round-trips
fine (proven). For now, clients must send `prompt_ids` (token ids);
the `messages[].content` text path returns 501 with a clear error.
Encode parity is FR-19.9-extra-deeper (multi-session).

## What Was Done Prior Session (2026-05-21)

Twelve commits, pushed to `origin/main`. The arc is matt-voice perf
optimization from the broken NaN starting state to 124% of llama.cpp
on Qwen2.5-7B, plus two investigations (speculative decoding,
self-host step 11) that didn't ship code but produced actionable
plans + memory.

```
399718e fix(matt-voice): per-block Q4_K/Q6_K dtype dispatch
9b5a21e perf(matt-voice): fused FFN kernel (gate+up+silu+mul in 1 launch)
859745d perf(matt-voice): byte-once Q4_K matmul v3 (kept alt, NOT on hot path)
1682bfe docs: HANDOFF + NEXT-UP + BENCH_LEDGER for 27.22 tok/s baseline
7e1804f perf(matt-voice): CUDA graph capture for autoregressive decode -> 37.35 tok/s
5aaf3a4 docs: HANDOFF + NEXT-UP + BENCH_LEDGER for 37.35 tok/s graph baseline
f40d259 perf(matt-voice): small-N matmul kernel explored, not promoted
add5216 perf(matt-voice): revert smallN matmul kernels (regress FFN via nvrtc unit pressure)
a3aa6ef docs: kernel-asm exploration learnings + 37.22 tok/s warm baseline
ef94fa3 docs: speculative decoding investigation -- analysis, not implementation
b23d661 NEXT-UP: speculative decoding investigated, deferred per user direction
62e18aa docs: Path E (self-host) status -- step 11 blocked on 8-arg fn bug
```

### The big win: CUDA graphs (commit 7e1804f)
After the prior session shipped fused matmul kernels and on-device
KV cache + attention to reach 25 tok/s, this session pushed end-to-end
through:
1. **NaN bisect (399718e)**: per-block Q4_K/Q6_K dtype dispatch.
   Qwen2.5-7B Q4_K_M is mixed-precision; V proj and ffn_down switch
   between Q4_K (144 B blocks) and Q6_K (210 B blocks) per layer.
   Hardcoding from block 0's dtype made block 3's V proj read garbage.
   Fix: store dt_v + dt_down per BlockGpu, dispatch matmul kernel by
   stored dtype. Generated IDs `[358, 2776, 264, 220, 17]` match
   cuBLAS reference exactly. **25.5 tok/s.**

2. **Fused FFN kernel (9b5a21e)**: replaces 4 kernel launches per
   layer (gate matmul + up matmul + silu + mul_inplace) with 1.
   Gate and up share x_norm; one kernel computes both, applies
   silu(gate)*up, writes one output. Parity bit-identical
   (`max_diff = 0`). **+7% throughput: 25.5 -> 27.2 tok/s warm mean.**

3. **CUDA graph capture (7e1804f)**: per-token forward gets recorded
   into one CUDA graph at first decode step; subsequent steps replay
   the graph with just a 4-int step_args h2d update. Compresses
   ~370 kernel launches per token into one `cuGraphLaunch`. Three
   pieces:
   - **Device-arg kernel variants** of rope_apply, append_kv,
     attention_seq1 that read pos/cur_seq from device memory
   - **Raw cudarc::driver::sys** bindings to cuStreamBeginCapture_v2,
     cuStreamEndCapture, cuGraphInstantiateWithFlags, cuGraphLaunch
   - **CudaDevice::new_with_stream()** instead of new() — the legacy
     default stream cannot be captured (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)
   **+37% throughput: 27.2 -> 37.35 tok/s = 124% of llama.cpp's ~30.**

### Kernel-asm exploration (a3aa6ef arc)
User asked to push further on the "assembly aspect" for more tok/s.
Per-shape matmul bench (matmul_per_shape_bench.rs) found:
- K/V proj at 512-out shape runs at 4.3% of peak BW (worst offender)
- FFN at 18944-out runs at 39% (dominant cost, ~9 ms/token)
- Q/O at 21%, down at 29%, lm_head at 38-46%

Tried 3 kernel-level wins:
1. **smallN matmul** (32-thread CTAs for K/V): 1.32x in isolation,
   end-to-end REGRESSION (~5%). SM scheduling fragments when mixing
   CTA sizes. Reverted.
2. **Interleaved FFN gate+up FMA**: 1.02x in isolation, end-to-end
   noise/regression. Reverted.
3. **Byte-once v3 matmul** (deeper from prior session): also slower
   end-to-end. Kept as alternate, not on hot path.

The unifying lesson: adding `__global__` kernels to KERNEL_SRC
regresses the existing actively-used ones by 5-7% via nvrtc unit
pressure (shared register allocation analysis). Removing the smallN
kernels restored the 37.35 baseline. **Treat KERNEL_SRC as load-
bearing — additions are not free.**

Plus discovered GPU boost-clock cold-start phantom: first run after
idle is ~5% slow while clocks ramp 210 -> 1950 MHz. Don't include
the cold run in N-run means.

### Speculative decoding investigation (ef94fa3, b23d661)
User: "Investigate speculative decoding". Empirical bench
(spec_dec_naive_verify_bench.rs) proved: naive verify by re-launching
the seq=1 graph N times scales **linearly** in N (4.00x at N=4).
**Break-even acceptance rate for N=4 is 99.96% — mathematically
impossible.** Speculative decoding requires real seq>1 kernels.

Full architecture analysis in docs/SPECULATIVE_DECODING_INVESTIGATION.md:
6 kernels need seq>1 variants + draft model integration + verification
orchestration. ~7-8 days for production quality. Expected speedup
1.6-2.6x (55-90 tok/s).

User decision: defer. The 37.22 tok/s baseline is already strong;
ship the matt-voice critical path instead.

### Path E step 11 attempt (62e18aa)
User: "go on 5" => Path E self-host compiler. Bootstrap step 10 (a
baby aetherc in Aether-source that emits real x86-64 .s files) is
shipped; step 11 was meant to add `if/else` + comparison operators.

Blocked on a real **Aether asm-backend bug**: 8-arg recursive fn
signatures access-violate at `popq %rbp; ret` epilogue, even on
step-10's working input the moment any fn gets an 8th arg. Bisection
ruled out the new code; the bug is purely about 8-arg signatures and
outgoing-arg space interfering with caller-frame locals.

Investigation captured in docs/PATH_E_STATUS.md with three concrete
next-step options. Memory updated at memory/asm_backend_known_gaps.md.

### New memories captured
- `qwen25_q4km_mixed_precision_per_block_dtype.md` — V/ffn_down dtype varies per layer
- `cuda_graphs_pattern.md` — non-default stream + devarg kernels + raw cudarc sys
- `nvrtc_kernel_unit_pressure.md` — KERNEL_SRC is load-bearing; unused kernels hurt active ones
- `gpu_boost_clock_warmup.md` — 5% cold-start phantom; discard run 1 from N-run means

### Stale-but-still-relevant prior session details

### Path A complete (FR-15.{1,2,3}) — earlier in session
- SSA-driven opt pipeline rewrites AST at --O1
- Regalloc plan drives callee-saved r12..r15 promotion
- AVX2 emit via aether_asm + recognised `__aether_avx2_dot_f32` builtin
- 23/23 honesty-auditor claims verified across the three commits

### Phase 17 closed to 100%
- conv2d CPU direct-loop reference (P17.3)
- Q4_0 GGUF dequant
- FlashAttention v2 (blocked online-softmax, matches naive SDPA)
- Real f32 Linear + LayerNorm witness
- Partial Llama-shape architecture forward (explicit partial scope)

### Phase 18 closed to 81% (only hardware-blocked items remain)
- NCCL FFI surface (single-host fallback; -1 sentinel on ws>1)
- Pipeline-parallel 1F1B sim (matt-voice's 2×P100 unlock per
  MATT_VOICE_FR.md)
- Tensor-parallel column-parallel Linear sim
- FSDP shard+alltoall sim
- ZeRO-1/2/3 staged sharding sim
- Compute/comm overlap sim
- Gradient compression shape

Only FR-18.10 (multi-host RDMA) + FR-18.11 (8-GPU) remain; both
hardware-binding per NEXT-UP §2 PARKED.

### Phase 19 closed to 100%
**13 items in one batch** (paged KV / continuous batching /
speculative decoding / multi-model / tool calling / rate-limit /
observability / vision / speech / ChaCha20-Poly1305 / HTTP/1.1 /
OpenAI shape / WS frame), plus **FR-19.9 BPE tokenizer** + **FR-
19.10 chat template renderer** + the **FR-19.16 partial tok/s bench
at 177 tok/s on Llama-shape** (well over the ≥100 threshold).

### matt-voice deploy pack — 5 FR-x-extras (commit 3283015)
- `cargo build -p aether_rt --features cuda` succeeds on kokonoe;
  libaether_rt.a has 39507 cuBLAS-symbol matches; cuda_train_tiny
  goes from skipped → real GPU train exit=0.
- FR-17.19-extra: SafeTensors multi-tensor parser (n_tensors /
  get_shape / get_dtype with the F32/F16/BF16/I32/I16/U8/I64 enum).
- FR-17.14-extra: Q4_K_M dequant kernel (real ggml 144-byte
  super-block layout; the format matt-voice's Qwen2.5-7B uses).
- FR-19.9-extra: HF tokenizer.json loader (hand-walks vocab + merges
  with explicit HF id preservation — essential for model weight
  indexing).
- FR-19.10-extra: chat_template.jinja file loader (wraps std::fs::read
  → render).
- Plus `aether_copy_cstr` helper: copies NUL-terminated `Expr::StrLit`
  literals from .rdata into heap buffers. Major witness-readability
  win — see `memory/aether_copy_cstr_pattern.md`.

### Real GGUF reader for Qwen2.5-7B (commit 172f423)
**The story**: user said Llama-1B weights were downloaded but
couldn't recall location. Searched extensively — no Llama-3-1B on
this machine. What IS local: matt-voice's actual base model,
Qwen2.5-7B-Instruct Q4_K_M, in ollama's blob store as 4.7 GB at
`C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`.

User picked: use the local Qwen2.5-7B. Built a real GGUF v3 reader
that walks all 339 tensors. Witness verifies tensor 0 is
`token_embd.weight` at dtype 12 (= Q4_K = matt-voice's quant
format). Together with the FR-17.14-extra Q4_K_M dequant kernel
already shipped, Aether can now READ real Qwen2.5-7B weight bytes.

Three primitive layers shipped together:
1. `aether_gguf_open` + 8 walker fns
2. `aether_dequant_q4_k_m` (Qwen2.5's exact format)
3. cuBLAS routing via `--features cuda`

honesty-auditor verdicts across the session: **51/51 claims
verified, zero false** across 9 commits.

## Current State

**Working (deploy — the new headline):**
- **`aether-serve` produces real Qwen2.5-7B text via OpenAI HTTP**
  - `POST /v1/chat/completions` (single-shot JSON OR `stream:true` SSE)
  - `GET /v1/models`, `GET /health`
  - JSON-escape, EOS auto-stop, warmup on startup
  - 24-37 tok/s warm (dependent on GPU power state — see HANDOFF
    "Performance reality check" subsection)
- Generated IDs bit-identical to `qwen25_graph_decode.rs` reference

**Working (matt-voice perf — prior session headline):**
- **37.22 tok/s warm mean on Qwen2.5-7B Q4_K_M / RTX 3070 Ti = 124% of llama.cpp**
- Generated IDs bit-identical to cuBLAS reference: `[358, 2776, 264, 220, 17]`
- Full forward pass via on-device fused matmul + on-device KV cache
  + GPU attention kernel, all wrapped in one CUDA graph that replays
  per decode step
- `runtime/tests/qwen25_graph_decode.rs` is the standing benchmark
- Per-shape diagnostic: `runtime/tests/matmul_per_shape_bench.rs`

**Working (TLS 1.3 crypto foundation — this session):**
- 7 RFC-verified primitives: SHA-256, SHA-512, HMAC-SHA256, HKDF
  (Extract+Expand+Expand-Label), X25519 (derive+shared), Ed25519
  (derive+sign+verify), ChaCha20-Poly1305 (prior).
- ABI: `aether_sha256/512`, `aether_hmac_sha256`, `aether_hkdf_*`,
  `aether_tls13_hkdf_expand_label`, `aether_x25519_*`, `aether_ed25519_*`
- Full TLS 1.3 cipher suite TLS_CHACHA20_POLY1305_SHA256 is now
  buildable from these primitives. Remaining: handshake state
  machine, record layer, X.509 cert handling.

**Working (broader project, unchanged from prior session):**
- 169/196 audit-tagged witnesses pass (no change — crypto adds are
  library funcs not roadmap line items).
- 61 lib tests pass (+2 from prior session: ed25519, sha512).
- 10 phases at 100% (6-14 + 17 + 19).
- `cargo build -p aether_rt --features cuda` succeeds; cuBLAS path live.
- Real Qwen2.5-7B Q4_K_M GGUF readable via `aether_gguf_*` extern surface.
- matt-voice's serving-deploy critical path's LANGUAGE work:
  - ✅ BPE algorithm + chat template engine
  - ✅ tokenizer.json + chat_template.jinja file loaders
  - ✅ Q4_K_M dequant + GGUF reader (and now mixed-precision-aware)
  - ✅ SafeTensors multi-tensor parser
  - ✅ cuda runtime path live (cuBLAS sgemm + nvrtc kernels + CUDA graphs)
  - ✅ Full forward pass through real Qwen2.5 weights at scale
  - ✅ **OpenAI-compat HTTP server returns real text (this session)**
  - ✅ **TLS 1.3 crypto primitives RFC-verified (this session)**
  - ⏳ FR-19.1-extra TLS 1.3 handshake state machine (XL, partial — primitives done)
  - ⏳ FR-19.2-extra HTTP/2 framing + HPACK
  - ⏳ FR-19.9-extra-deeper text→ids encode with GPT-2 unicode BPE
  - ⏳ FR-19.4-extra real paged KV cache (block allocator)
  - ⏳ FR-19.5-extra continuous batching scheduler

**Honest scaffold-vs-shipped notes** (updated):
- Phase 19's FR-19.16 ships a PARTIAL tok/s bench (177 tok/s on
  Llama-shape, NOT real Llama-1B). The witness header carves out
  the partial scope explicitly. The full Llama-1B target is
  FR-19.16-extra.
- The GGUF reader reads bytes only — no forward pass through real
  weights yet. The next gate (weight → dequant → matmul chain at
  every transformer layer) is multi-session.
- The "simulations" in Phase 18 are correctly named `*_simulate_*`
  so the simulation status is visible at the call site. Real
  multi-rank needs FR-18.1-extra libnccl + a second GPU.
- 4 known-OK stubs unchanged (mir/fuse.rs:53, mir/spec.rs:161,
  runtime_pe/src/lib.rs:59 + :443).
- Audit count for FR-x-extra tags doesn't advance (parent tag
  reuses) — design intent, not regression.

## Blocking Issues

None on the matt-voice perf critical path — the 37.22 tok/s baseline
is solid and tested. Remaining deploy items + active blockers:

- **Path E step 11 (self-host if/else)** — blocked on Aether asm
  backend bug: 8-arg fn signatures crash on `popq %rbp; ret`
  epilogue. See `docs/PATH_E_STATUS.md` for the investigation and
  three workaround options. Workaround needed before continuing
  the self-host bootstrap chain.
- **FR-19.1-extra full TLS 1.3** — XL multi-session work for real
  HTTPS serving. Not blocking the local 37 tok/s state.
- **FR-19.16-extra deploy as HTTP server** — composite of TLS +
  HTTP + serving. Multi-session.
- **FR-18.1-extra real libnccl** — needs the cnc 2× P100 box
  (kokonoe is single-GPU).

## What's Next

For "switch flip from llama.cpp": the **basic** hosting path is now
done (single-shot + SSE). The remaining Path D pieces, in dependency
order:

1. **TLS 1.3 handshake state machine** (XL, ~1-2 sessions). With the
   crypto primitives in place, this is mostly:
   - Record layer (TLSPlaintext + TLSCiphertext wrappers — ChaCha20-
     Poly1305 already encrypts the records)
   - Handshake messages: ClientHello, ServerHello, EncryptedExtensions,
     Certificate, CertificateVerify, Finished
   - Key schedule: early_secret, handshake_secret, master_secret +
     derive_secret + 8 per-direction traffic keys (HKDF-Expand-Label
     already shipped — just call it 8 times with the right labels)
   - Self-signed cert generation (Ed25519 keypair + minimal X.509 DER)
   - State machine driver: drive a TLS connection from accept to
     handshake_finished
2. **HTTP/2 framing + HPACK** (M-L). Independent of TLS — h2c
   over plain TCP is fine for the in-flight increment. HEADERS /
   DATA / SETTINGS / WINDOW_UPDATE / PING + static HPACK table.
3. **Real paged KV cache** (M) — block allocator over a fixed GPU
   pool, virtual-page mapping per request. Replaces today's
   single-resident KV per QwenSession. Unlocks multi-request
   serving without growing per-request VRAM.
4. **Continuous batching scheduler** (M) — decode N requests per
   step rather than one. Sits on top of paged KV cache.

Three alternative directions (orthogonal to D-path):

1. **D-path** (TLS / HTTP/2 / batching) — see above.
2. **E-path (self-host)**: fix the 8-arg fn bug in the asm backend,
   then continue bootstrap steps. M-L for the fix, then incremental.
3. **Spec-decode build-out** (deferred prior session): seq>1 kernel
   suite + draft model. ~7-8 days. 60-90 tok/s upside.

### Operational note for next-session smoke testing

When measuring `aether-serve` perf or running benches:
- Run `nvidia-smi -pm 1` and `nvidia-smi --query-gpu=pstate` to
  verify GPU is in persistence mode at P0/P2 before measuring.
- Use `--warmup 6` (or higher) on `aether-serve` to drive GPU
  into P2 before serving real requests.
- Discard the FIRST run after spawn from any N-run mean
  (cold-clock penalty per `memory/gpu_boost_clock_warmup.md`).
- Reference standing bench is `cargo test --release --features cuda
  -p aether_rt --test qwen25_graph_decode -- --ignored --nocapture`
  — this should report 27-37 tok/s on a hot GPU. If it reports
  <10 tok/s, the GPU is in P5/P8 and serve will be slow too.

(Continuing from prior session's option list — unchanged.)

2. **If extending CUDA perf work on matt-voice**: the FFN at
   39% peak BW is the dominant cost. Further gains require either
   PTX/SASS-level kernel work (high risk per this session's
   lessons) or moving to tensor-cores (XL, requires F16 reorg).
   Diminishing returns; ~37 tok/s is already past llama.cpp.

3. **If picking up Path E step 11**: easiest path is the
   step-11-lite variant (comparison ops only, no labels, doesn't
   need 8 args). Real fix is auditing
   `compiler/src/codegen/asm/mod.rs` outgoing-arg-area layout.

## Notes for Next Session

- **The 37.22 tok/s number is WARM mean.** First run after GPU idle
  shows ~35 tok/s while clocks ramp 210 → 1950 MHz. Always
  pre-warm before measuring. `nvidia-smi --query-gpu=clocks.current.graphics`
  confirms boost state per iteration.
- **KERNEL_SRC is load-bearing.** Adding new `__global__` kernels
  to `runtime/src/cuda.rs::KERNEL_SRC` can regress existing active
  kernels by 5-7% via nvrtc shared register-allocation analysis.
  Always re-run `matmul_per_shape_bench.rs` + the full graph
  decode after KERNEL_SRC edits.
- **In-isolation kernel benchmarks lie.** smallN was 1.32x in
  isolation but -5% end-to-end. Always validate via end-to-end
  `qwen25_graph_decode.rs` before promoting.
- **CUDA graph capture requires non-default stream.** `CudaDevice::new()`
  uses the legacy null stream; `cuStreamBeginCapture_v2` rejects
  it. Already switched to `CudaDevice::new_with_stream()` in cuda.rs.
- **Per-block dtype dispatch matters for Q4_K_M**. Qwen2.5-7B has
  V proj and ffn_down switching between Q4_K and Q6_K per layer.
  Hardcoding from block 0 → NaN at first mismatched layer. See
  `memory/qwen25_q4km_mixed_precision_per_block_dtype.md`.
- **bench/qwen25_7b_autoregressive section in docs/BENCH_LEDGER.md**
  is the canonical perf history for this model. Append rows when
  this number moves.

## FR-18.1-extra — Real libnccl cross-card (LANDED)

Aether's runtime now supports REAL cross-GPU NCCL collectives,
verified end-to-end on cnc-server's 2× P100 box.

New surface (gated `--features nccl`):
- `runtime/src/nccl_real.rs` — `aether_nccl_real_init_multi_gpu(n)`
  wraps `ncclCommInitAll`; `_get_handle(rank)` / `_all_reduce_f32` /
  `_comm_world_size` / `_comm_rank` / `_finalize` round out the
  surface. Plus a Rust-side `comm_at(i)` accessor for integration
  tests that need to drive cudarc's typed API directly.
- `runtime/tests/nccl_dual_gpu.rs` — `Comm::from_devices(2)`,
  group_start/end, all_reduce sum: rank 0 sends 1.0s, rank 1 sends
  2.0s, both ranks see 3.0s. Verified on 2× P100.
- `runtime/tests/nccl_dual_gpu_dp_step.rs` — data-parallel
  training step: each rank computes gradient on its own shard
  (rank 0=1.0, rank 1=3.0), all_reduce sum, divide by ws → mean=2.0,
  identical SGD update on both ranks. The matt-voice unlock shape.

nvidia-smi confirmation: a single test process appears on BOTH GPU
UUIDs simultaneously (`bb77bda0...` first P100, `17bd0d20...` second
P100) with ~260+318 MiB allocations — physically proving cross-card
data exchange.

NCCL compatibility note: ollama's bundled libnccl 2.29 dropped sm_60
(Pascal) kernels and fails with "named symbol not found" on the
P100s. Aether links against libnccl 2.21.5+cuda12.4 from the local
fish-speech venv via `/usr/local/lib/libnccl.so.2` symlink. Documented
in NEXT-UP.

## On-device KV cache + attention kernel LANDED (correctness still WIP)

User: "Go on attention correctness". Shipped:

### Kernels
- `append_kv` -- writes new K/V step into the per-block KV cache
  at position `pos`.
- `attention_seq1` -- one warp per Q head, lanes cooperatively compute
  scores via warp-reduce, softmax (max + exp+sum + normalise), then
  aggregate V_cache by softmax weights. Dynamic shared mem sized
  for `cur_seq * 4` bytes.
- Wrappers `aether_op_append_kv_f32_cuda` +
  `aether_op_attention_seq1_f32_cuda`.

### Verification (in isolation)
`runtime/tests/attention_seq1_parity.rs` — Q/K_cache/V_cache with
known values, n_q=28 / n_kv=4 / head_dim=128 / cur_seq=7. GPU
matches CPU reference within **max_diff = 7.15e-7**.

### Wiring into autoregressive_fused
`runtime/tests/qwen25_autoregressive_fused.rs` now uses the real
GPU attention kernel + per-block KV cache (MAX_SEQ=32 device
buffers per block, 28 caches). Speed: **25 tok/s**.

### NaN at block 3: ROOT-CAUSED + FIXED

Bisecting per-op magnitudes inside block 3 revealed
`[V + bias] max_abs=2.192e9 nan=true` — V proj's output was garbage,
not "FP drift accumulating across blocks".

`q6k_blk3_diagnose.rs` failed with `assertion left==right: 12 vs 14`:
**`blk.3.attn_v.weight` is Q4_K (12), not Q6_K (14)**. Qwen2.5-7B
Q4_K_M is **mixed-precision** — V proj and ffn_down switch between
Q4_K and Q6_K per layer:

- V proj Q6_K on blocks [0,1,2,5,9,11,14,17,20,23,24,25,26,27],
  Q4_K on the rest.
- ffn_down Q6_K on [0,1,2,5,7,12,14,17,20,23,24,25,26,27],
  Q4_K on the rest.
- Q/K/O/gate/up always Q4_K; lm_head Q6_K.

`qwen25_per_block_dtypes.rs` enumerates the full table.

The fused Q4_K and Q6_K kernels have **different super-block
layouts** (144 B vs 210 B per 256-elem block) — dispatching to the
wrong one reads completely garbled weights, hence the 2.19e9 / NaN.

**Fix** (commit pending): `BlockGpu` now stores `dt_v: i32` and
`dt_down: i32` captured from `aether_gguf_get_tensor_dtype` at
upload time; `upload_tensor_u8` returns the dtype as a 3rd element.
The forward pass dispatches V proj and ffn_down matmul on the
stored dtype:

```rust
if bw.dt_v == 14 {
    aether_op_fused_q6k_matmul_seq1_v2_cuda(...);
} else {
    aether_op_fused_q4k_matmul_seq1_v2_cuda(...);
}
```

Same dispatch for ffn_down on `bw.dt_down` and for lm_head on `lm_dt`.

**Result** (verified locally just now):
- Block 3 V + bias now `max_abs=1.688e0` (sane, was 2.19e9).
- All 28 blocks produce finite activations end-to-end.
- Generated IDs `[358, 2776, 264, 220, 17]` — matches the
  cuBLAS-routed reference exactly.
- Speed sustained at **25.53 tok/s** (39.2 ms/token, 5 decoded
  tokens after a 4-token prefill at 37.9 ms/token).

llama.cpp's Q4_K_M dispatch table for the same model reports
~30 tok/s on a 3070 Ti; Aether is at **85% of llama.cpp throughput
with sane outputs**. Correctness gap is closed.

Pitfall captured in memory at
`memory/qwen25_q4km_mixed_precision_per_block_dtype.md`.

## End-to-end fused autoregressive: 24 tok/s MEASURED on RTX 3070 Ti

User: "Go till the cuda tuning is complete". Continued the v2 line
through Q6_K fused matmul + end-to-end measurement.

### Shipped in this batch
- **Q6_K fused matmul v2** (`fused_q6k_matmul_seq1_v2` kernel +
  `aether_op_fused_q6k_matmul_seq1_v2_cuda` wrapper). Same warp-
  per-output split-K design as Q4_K v2 but for 210-byte Q6_K
  super-blocks. Used for V proj + ffn_down + lm_head.
- **End-to-end test** `qwen25_autoregressive_fused.rs` wires v2
  kernels into the full Qwen forward + lm_head + sampling chain.

### Q6_K v2 measured perf
```
tensor                       n      k    cuBLAS    v2  speedup  max_diff
blk.0.attn_v.weight        512  3584      26us  55us  0.47x   9.5e-7
blk.0.ffn_down.weight     3584 18944     475us 262us  1.81x   9.3e-5
output.weight (lm_head) 152064  3584    3692us 1580us 2.34x   2.9e-6
```

The **lm_head matmul (largest single matmul) saves 2.1 ms per token**.

### End-to-end measurement (RTX 3070 Ti, release build)

| Stage | Time |
|---|---|
| Upload 28 blocks Q4_K+Q6_K bytes | 0.68 s |
| Upload output_norm + lm_head | 0.75 s |
| Prefill 4 tokens | 158 ms (39.6 ms/token) |
| Generate 5 tokens | 207 ms |
| **Per-token cost** | **41.4 ms** |
| **tok/s** | **24.17** |
| llama.cpp reference (same hw) | ~30 tok/s |
| **Speedup vs prior CPU baseline** | **96× from 0.25 tok/s** |

### Honest caveat: attention stub

The current end-to-end uses a SHORTCUT for attention: at seq=1 the
attn_out is approximated rather than computing real
softmax(Q·K^T) over a growing KV cache. The 24 tok/s figure
reflects the real cost of all matmuls + non-matmul ops +
kernel-launch overhead, but the GENERATED IDS are not meaningful
(they all argmax to 152063 = PAD because activations are wrong
without real attention).

Adding real attention with on-device KV cache is the remaining
correctness step. The attention matmul (Q·K^T softmax · V) at
typical context lengths is a small matmul: 1-2 ms/step. So full
correct inference should be **~43-45 ms/token = 22-23 tok/s** --
still in llama.cpp's range.

### What's complete and what's open

| Layer | Status |
|---|---|
| Q4_K/Q6_K resident weights | ✅ |
| GPU kernels for RMSNorm/RoPE/GQA/SiLU/add/mul/bias | ✅ |
| Fused Q4_K matmul v2 (split-K) | ✅ |
| Fused Q6_K matmul v2 (split-K) | ✅ |
| End-to-end measurement (24 tok/s) | ✅ |
| **Real attention with on-device KV cache** | ⏳ open |
| Tensor-core wmma path | ⏳ open (additional 2-3x possible) |
| Flash attention | ⏳ open |
| Q4_K kernel small-N (attn_k) tuning | ⏳ open (minor) |

The CUDA tuning track is materially complete at 24 tok/s. The
remaining "real attention + KV cache on GPU" is a correctness
fix, not a perf one (its cost is small).

## Fused Q4_K matmul v2 LANDED — split-K warp-reduce, 2.7x cuBLAS

User: "Go on the v2 kernel". Closed.

### Design
- CTA = 8 warps × 32 threads = 256 threads. Processes 8 output cols.
- Each WARP owns one output. 32 lanes cooperatively process the K
  dimension (each lane = 8 quants of the 256-quant super-block).
- After all K-tiles: `__shfl_down_sync` warp-reduce the 32 partials,
  lane 0 writes the output.
- A tile (256 f32) loaded once per K-tile via shared mem, all 8 warps
  share the same A.
- 2-way branch divergence (is_hi=0 lanes 0-3, is_hi=1 lanes 4-7,
  alternating) -- NVCC predicates with SEL, no penalty.

### Measured perf (RTX 3070 Ti, release)
```
tensor                       n      k    cuBLAS    v1    v2    v2_sp
blk.0.attn_q.weight       3584   3584      97us  131us   53us  1.83x
blk.0.attn_k.weight        512   3584      20us  100us   38us  0.53x
blk.0.attn_output.weight  3584   3584      95us  124us   51us  1.86x
blk.0.ffn_gate.weight    18944   3584     500us  396us  184us  2.72x
blk.0.ffn_up.weight      18944   3584     479us  397us  182us  2.63x
```

- v2 beats cuBLAS on 4 of 5 shapes (loses only on tiny attn_k where
  cuBLAS sgemm launch overhead is already at the floor).
- v2 is 2-2.5x faster than v1 across the board.
- v2 is **MORE accurate than v1**: max_diff 4.9e-6 vs v1's 1.4e-5.
  Warp-reduce sums in tree order which is numerically tighter than
  v1's sequential accumulate.

### Per-block matmul cost on Qwen2.5
| Path | Per-block cost |
|---|---|
| pure CPU | 5790 ms |
| cuBLAS dequant+sgemm | 1.78 ms |
| v1 fused | ~1.6 ms |
| **v2 fused (split-K)** | **1.04 ms** |

Per-block reduction with v2: 42% vs cuBLAS. Per-token extrapolation
(28 blocks + lm_head + misc):
- Prior cuBLAS-routed: ~4 sec/token
- v2 fused estimated: **~89 ms/token = 11 tok/s**
- That's **45× faster** than prior baseline
- Remaining gap to llama.cpp's ~30 tok/s: **2.7×**

### Remaining gap to llama.cpp parity

| Gap | Estimated speedup |
|---|---|
| Q6_K fused matmul (V proj + ffn_down + lm_head) | 1.5-2× |
| Tensor-core wmma path (sm_8.0+) | 2-3× |
| Flash attention (long seq) | 1.5× |
| Better tiling at small N | 1.2× |
| **Composite remaining gap** | **~3×** |

Closing the rest is incremental CUDA tuning, not new system design.

## Fused Q4_K matmul kernel v1 LANDED

User: "Go on fused kernel". Shipped a working fused Q4_K matmul
kernel that reads Q4_K bytes directly + dequants inline +
accumulates fma. No f32 transient buffer needed.

### Design (v1)
- CTA layout: BLOCK_N = 32 output columns per CTA
- One thread per output column (best for large N, weak for small N)
- Per K-tile (256 quants):
  - Cooperatively load 256 floats of A into shared mem (8 loads/thread,
    fully coalesced)
  - Each thread reads its own super-block of W (144 bytes), dequants
    inline, accumulates 256 fma's
  - Sync between K-tiles
- 1 KB shared memory per CTA (just A tile)
- Wrapper: `aether_op_fused_q4k_matmul_seq1_cuda(a_dev_f32, w_dev_u8,
  out_dev_f32, n, n_blocks)`

### Correctness verified
`runtime/tests/fused_q4k_matmul_real.rs` runs all 5 Q4_K matmul
shapes of Qwen2.5 block 0 and compares against `dequant -> cuBLAS
sgemm` reference. **All shapes match within 1.4e-5 absolute** (sum-
order differences between cuBLAS and our accumulate-into-thread
kernel).

### Measured perf on RTX 3070 Ti
```
tensor                       n      k    cuBLAS_us  fused_us  speedup
blk.0.attn_q.weight        3584   3584         100       127    0.79x
blk.0.attn_k.weight         512   3584          33       100    0.33x
blk.0.attn_output.weight   3584   3584          94       123    0.76x
blk.0.ffn_gate.weight     18944   3584         478       397    1.20x
blk.0.ffn_up.weight       18944   3584         475       396    1.20x
```

The v1 kernel is **faster than cuBLAS only on large-N FFN matmuls**.
For small N (attention projections), each output gets one thread
which under-utilizes the GPU (3584 threads << 6144 cores). A v2
kernel needs split-K reduction (multiple threads per output's dot
product, then warp-reduce) for small-N matmul wins.

### What the v1 kernel really unblocks: full-resident inference

Without fused matmul, every matmul needs a transient f32 dequant
buffer = 870 MB per block per matmul. With fused matmul, Q4_K
bytes go directly to f32 outputs — no transient. This is THE
enabler for keeping all 28 blocks of Qwen2.5-7B resident in 8 GB
VRAM throughout autoregressive generation.

### Remaining open FRs for matt-voice production speed

1. **Split-K Q4_K kernel (v2)** — multiple threads per output's
   dot product, warp-reduce. Estimated 2-3x more speedup on
   attention matmuls. ~100 LOC of careful CUDA.
2. **Tensor-core path for sm_8.0+** — RTX 3070 Ti is sm_8.6.
   Cast Q4_K dequant output to f16 + use wmma::fragment for the
   accumulate. Estimated 4-8x more speedup over current. Larger
   CUDA effort (~200-500 LOC).
3. **Q6_K fused matmul** — V proj + ffn_down. Same pattern as
   Q4_K but with the 210-byte super-block layout.
4. **Full end-to-end autoregressive measurement** with the fused
   kernel wired into the existing qwen25_autoregressive_cuda.rs.
   Quantify the real tok/s on Qwen2.5-7B inference.

## Q4_K + Q6_K dequant on GPU LANDED (memory enabler for matt-voice deploy)

User: "Get it to production speed". Plan was Q4_K-on-GPU to close
the llama.cpp gap. This session: the FOUNDATION (u8 device
registry + Q4_K/Q6_K dequant kernels + parity tests). The fused
dequant+matmul kernel that finishes the perf story remains the
single biggest remaining FR.

### Shipped
- `aether_dev_alloc_u8` / `_h2d_u8` / `_d2h_u8` / `_free_u8` --
  device byte buffer registry (parallel to f32 + i32 registries).
- `dequant_q4_k_m` CUDA kernel: 256 threads/CTA, 1 output per
  thread. Reads f16 d + dmin + 12 packed scales + 128 nibble-
  packed quants per 144-byte super-block.
- `dequant_q6_k` CUDA kernel: same shape, 210-byte super-blocks
  with ql[128] + qh[64] + scales[16] + f16 d.
- Both expose `aether_op_dequant_q*_f32_cuda(blocks_u8, out_f32,
  n_blocks)` wrappers.

### Bit-exact vs CPU on real Qwen2.5 weights
- `q4_k_dequant_cuda_parity.rs`: synth block + real
  blk.0.attn_q.weight first 4 super-blocks. `max_diff = 0`.
- `q6_k_dequant_cuda_parity.rs`: real blk.0.attn_v.weight
  (7168 super-blocks, 1.4M elements). `max_diff = 0`.
- `qwen25_block_forward_q4k_resident.rs`: 5 Q4_K tensors of
  Qwen block 0 verified bit-exact. **Total 84 MB Q4_K vs 623 MB
  f32 = 7.4x less PCIe** per block.

### Memory accounting (RTX 3070 Ti 8 GB)
- All 28 blocks of Qwen2.5-7B as Q4_K + Q6_K: ~6 GB **fits** with
  room for activations + KV cache.
- All 28 blocks as f32 dequant'd: ~24 GB **does NOT fit**.

Q4_K/Q6_K-on-GPU is what makes 8 GB VRAM enough to hold the
entire model. This is the unblock for matt-voice inference at
production scale.

### The final perf finisher (Q4_K matmul fusion) -- still open

Today: per matmul we dequant Q4_K -> transient f32 -> cuBLAS sgemm.
The transient f32 buffer is 4x the Q4_K size and the cuBLAS sgemm
runs over the bloated f32 (same as before). What llama.cpp does:
fused dequant+matmul where each CTA loads a tile of Q4_K bytes
into shared mem, dequants inline, then does sgemm-style accumulate
against the activation matrix in registers.

Engineering: ~500-1000 LOC of careful CUDA (tile sizing, shared
memory layout, warp-level math, tensor-core path on sm_8.0+).
Estimated ~10-50x additional speedup on top of today's bit-exact
dequant. That's the final piece for matt-voice production speed.

Tagged FR-17.14-extra-deepest in NEXT-UP. The Q4_K-on-GPU
foundation shipped this session is the prerequisite.

## GPU-native Qwen block forward LANDED — 115x speedup

User asked: "address major inference on gpu bugs/FR to bring to
parity with llama.cpp". Identified the dominant gap: every
non-matmul op in qwen forward was bouncing activations through
CPU (h2d/d2h per op). Shipped 5 new device kernels closing that:

### New CUDA kernels (added to KERNEL_SRC + ctx)
- `rms_norm_fwd` — RMSNorm `y = x * gamma / sqrt(mean(x^2) + eps)`
- `rope_apply`  — rotary in-place, half-half pair, Qwen-style
- `gqa_repeat_kv` — broadcast n_kv -> n_q heads (parallel copy)
- `silu_inplace` — `x = x / (1 + exp(-x))`
- `mul_inplace` / `add_inplace` / `bias_add` — element-wise

### New extern wrappers in runtime/src/cuda.rs
- `aether_op_rms_norm_f32_cuda(x, gamma, out, eps, rows, d)`
- `aether_op_rope_apply_f32_cuda(x, seq, n_heads, head_dim, base, pos_start)`
- `aether_op_gqa_repeat_kv_f32_cuda(in, out, seq, n_kv, head_dim, n_q)`
- `aether_op_silu_f32_cuda(x, n)`
- `aether_op_mul_inplace_f32_cuda(x, y, n)`
- `aether_op_add_inplace_f32_cuda(x, y, n)` (residual)
- `aether_op_bias_add_f32_cuda(x, bias, rows, cols)`

### Verification
- 5 parity tests (`matt_voice_ops_cuda_parity.rs`): every new
  kernel matches its CPU reference within 1e-4 / 1e-5.
- `qwen25_block_forward_full_gpu.rs` runs the entire block 0
  forward of real Qwen2.5-7B with all ops on device. Matches
  the CPU reference within `max_diff=2.956e-5` across all
  14336 output elements.

### Measured (release build, 11900K + RTX 3070 Ti)

| Phase | Time |
|---|---|
| CPU reference forward (block 0, seq=4) | 5.79 s |
| GPU all-on-device forward (block 0, seq=4) | **0.05 s** |
| Speedup vs CPU | **115×** |
| One-time h2d weight upload (per block) | 0.27 s |

For autoregressive generation per token:
- Prior cuBLAS-routed: ~4 s/token (per-op CPU bouncing dominated)
- Estimated new: ~1.4 s/token (28 blocks × 50 ms + h2d weight load)
- llama.cpp Q4_K-on-GPU reference: ~30 tok/s = 33 ms/token

### Remaining llama.cpp parity gap: Q4_K-on-GPU

The dominant remaining bug to close llama.cpp parity is **fused
dequant + matmul for Q4_K weights**. Today Aether dequantises
Q4_K_M block weights to f32 on the host then h2d's the f32:
- Memory: 870 MB f32 per block vs 217 MB Q4_K (4x)
- Bandwidth: 870 MB h2d per block vs 217 MB Q4_K (4x)
- Plus: cuBLAS sgemm on f32, not the fused dequant+matmul that
  llama.cpp implements

Implementing this is the single biggest remaining FR for matt-voice
production deploy. ~50× speedup expected. Scope:
1. Allocate Q4_K_M blocks (raw bytes) on device, not dequant'd f32
2. Custom CUDA kernel: dequant Q4_K block into shared mem, then
   sgemm-style matmul in same kernel
3. Apply to all `aether_op_matmul_f32_cuda` callsites that read
   Q4_K weights (full transformer forward)

Once shipped, matt-voice on RTX 3070 Ti should hit ~30 tok/s,
which is competitive with llama.cpp on the same hardware.

## FR-18.10 UNPARKED — 3-host TCP/IP all-reduce works

User said "we need 18.10 and 18.11" with kokonoe + cnc + satibook
available. All 3 hosts on 192.168.168.x LAN + Tailscale.

**Hardware pool:**
| Host | OS | LAN IP | GPU | VRAM |
|---|---|---|---|---|
| kokonoe | Windows | 192.168.168.121 | RTX 3070 Ti | 8 GB |
| cnc-server | Linux | 192.168.168.100 | 2× P100 | 12+16 GB |
| satibook | Windows | 192.168.168.200 | RTX 3050 Laptop | 6 GB |

Total **4 GPUs across 3 hosts, ~42 GB combined**.

**Shipped:**
- `aether_tcp_listen_addr(addr, n_addr, port)` -- bind to any
  interface (was 127.0.0.1-only).
- `aether_tcp_connect_host(host, n_host, port)` -- connect to any
  host (was 127.0.0.1-only).
- `trainer/src/bin/allreduce.rs` -- new `aether-allreduce` binary.
  Rank 0 = rendezvous server (listens 0.0.0.0), ranks 1..N-1 =
  clients. Per all-reduce: rank 0 collects buffers from each peer,
  computes sum, broadcasts.

**3-host run verified:**
```
kokonoe rank 0 (value=1) -> received [7.0, 7.0, ...]
cnc     rank 1 (value=2) -> received [7.0, 7.0, ...], RTT 158 ms
satibook rank 2 (value=4) -> received [7.0, 7.0, ...], RTT 25 ms
Sum: 1 + 2 + 4 = 7 ✓ across all ranks
```

**Operational notes documented in memory:**
- `three_host_pool_setup.md` -- inventory, firewall fix
  (`netsh advfirewall firewall add rule name="aether-N"
  dir=in action=allow protocol=TCP localport=N`), build commands
  per host.

## FR-18.11 partial — 4-GPU shape, 8-GPU still gated by hardware

The original FR-18.11 spec is "8-GPU Llama-7B training". With 4
GPUs in the pool, the full 8-card witness can't run. But all
PROTOCOL components are proven:
- Multi-host all-reduce (above)
- Per-host data-parallel (`nccl_dual_gpu_dp_step`, `_resident`)
- KV cache + autoregressive (`qwen25_autoregressive_cuda`)
- Block-streaming weight load (`qwen25_full_inference`)

Combining into a 4-host distributed training run that trains
matt-voice across all 4 GPUs is the next-session bridge.

## aether-serve HTTP binary + LoRA apply LANDED

### LoRA apply
`runtime/src/ops.rs::apply_lora_f32` + extern wrapper
`aether_op_apply_lora_f32`: in-place update of an Aether matmul-
layout weight by `W += scale * A^T @ B^T`. PEFT-compatible
convention (A=[rank,d_in], B=[d_out,rank]).

Tests:
- Unit: zero LoRA = no-op identity; matmul-layout math matches
  direct computation for a 2x3 case with hand-traced expected.
- `runtime/tests/qwen25_lora_apply.rs`: applies a synthetic rank-8
  LoRA to REAL Qwen2.5 blk.0.attn_q.weight (3584×3584 Q4_K dequant).
  Probes 4 specific (i_in, i_out) cells; each delta matches the
  scale*A^T@B^T direct computation to 1e-4. Frobenius norm grows
  by 0.61% (LoRA had measurable effect). 12.84M elements in 0.02s.

### HTTP server
`trainer/src/bin/serve.rs`: new `aether-serve` binary. Listens on
the chosen port, accepts HTTP requests, parses with
`aether_http_parse_request`, renders OpenAI-shape JSON via
`aether_openai_render_completion`, sends back via
`aether_http_write_response_200` + `aether_tcp_send`.

Build + run:
```
cargo build -p trainer --bin aether-serve
target/debug/aether-serve --port 8080 --model matt-voice
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_ids":[9707,11,1879,0],"max_tokens":5}'
```

Returns proper OpenAI JSON:
```json
{
  "id":"chatcmpl-aether-serve-1","object":"chat.completion",
  "model":"matt-voice",
  "choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":N,"completion_tokens":M}
}
```

**Status:** The HTTP wire-up is complete. The `content` field
currently returns a stub message; integrating the real Qwen2.5-7B
autoregressive forward chain (from `qwen25_autoregressive_cuda.rs`)
into `handle_request` is ~300 LOC of mechanical copy + JSON body
parsing for `prompt_ids` / `max_tokens`. Tracked as the final
ship gate.

`runtime/tests/aether_serve_http_wireup.rs` end-to-end-tests the
full HTTP loop without the 24 GB model load: TCP listen → accept
→ parse → render → respond → verify. Runs in ~100 ms.

## cuBLAS-routed autoregressive generation LANDED

`runtime/tests/qwen25_autoregressive_cuda.rs` -- same shape as the
CPU autoregressive test but every matmul routes through cuBLAS via
a per-call host-pointer wrapper (`matmul_via_cublas`: dev_alloc /
h2d / sgemm / d2h / free).

Per-token cost: **53s (CPU) -> 4s (cuBLAS) -- 13x speedup**.
Prefill: 206s -> 4.5s -- 45x. Total 4 prompt + 5 generated:
501s -> 106s.

CORRECTNESS: generated IDs are byte-identical between CPU and
cuBLAS runs: [9707, 11, 1879, 0, 358, 2776, 264, 220, 17]. Same
logits to 3 decimals. Strong determinism signal across backends.

Non-matmul ops (RMSNorm/RoPE/GQA/SiLU/attention) stay on CPU.
Routing all of them through GPU would be FR-x-extra-deeper but the
13x matmul-only speedup is already enough to make Aether-Qwen
inference usable for matt-voice serving.

## Tokenizer integration LANDED (decode side)

`runtime/tests/qwen25_tokenizer_roundtrip.rs` -- loads Qwen2.5's
embedded tokenizer (152064 vocab + 151386 merges + BOS/EOS ids)
from GGUF metadata into `aether_bpe_tokenizer`. Decodes the
autoregressive output to actual coherent English:

```
[decode] "Hello,Ġworld!ĠI'mĠaĠ2" (surface)
[decode] "Hello, world! I'm a 2" (real text)
```

Token IDs from the autoregressive run, when decoded with the
GPT-2 byte fixup, produce **real coherent text**. This is the
matt-voice deploy proof: real prompt -> real Qwen inference -> real
generated text, all through Aether's runtime.

New runtime surface (4 GGUF metadata accessors):
- `aether_gguf_get_metadata_u32(handle, key, key_len) -> i64`
- `aether_gguf_get_metadata_string(handle, key, key_len, out, max) -> i32`
- `aether_gguf_get_metadata_array_string_n(handle, key, key_len) -> i32`
- `aether_gguf_get_metadata_array_string_get(handle, key, key_len, idx, out, max) -> i32`

GGUF parser refactor: previously the metadata KV table was SKIPPED
during parse. Now U32, String, and StringArray values are captured
into a `HashMap<String, GgufMeta>` on the GgufFile struct. All
other types still skip (memory hygiene).

**Known limitation: encode** (text -> IDs) requires unicode-char-
level initial split because Qwen uses GPT-2 BPE with the bytes-to-
unicode mapping. Aether's `aether_bpe_encode` does byte-level
initial split. For matt-voice's inference deploy this is fine
(user tokenizes externally; Aether takes IDs); but full in-Aether
text-in-text-out requires extending the encoder. FR-x-extra.

## Autoregressive generation LANDED

`runtime/tests/qwen25_autoregressive_gen.rs` produces multi-token
generated output from a 4-token prompt through real Qwen2.5-7B
with **per-block KV cache**.

Architecture:
- `BlockWeights`: all 12 matmul-targeted weights + biases per block
  loaded + transposed ONCE before generation (~24 GB f32 total for
  all 28 blocks).
- `KvCache`: per-block `Vec<f32>` storage for past K and V
  activations. Grows by `seq * D_KV` per call.
- `block_forward_kv`: handles BOTH prefill (seq=prompt_len) and
  per-step (seq=1) modes. Q/K/V projection on the new tokens only;
  K/V appended to cache; attention reads from cache.

End-to-end run:
```
[all blocks loaded]   71.74s -- 28 blocks * ~2.6s each
[lm_head xpose]        6.62s
[prefill]            206.00s -- 4 tokens through 28 blocks
[gen 1/4]             53.04s -- next_id=2776 logit=13.374
[gen 2/4]             53.83s -- next_id=264  logit=12.557
[gen 3/4]             53.66s -- next_id=220  logit=15.310
[gen 4/4]             54.27s -- next_id=17   logit=27.127
[total]              501.61s
Generated IDs: [9707, 11, 1879, 0, 358, 2776, 264, 220, 17]
                 (prompt 4 tokens)  (5 generated)
```

Per-token cost dropped from 270s (full re-forward) to **53s** with
KV cache -- a 5x speedup. The model produces increasingly confident
predictions (logit 10.3 → 27.1) which is normal autoregressive
behaviour as context extends.

Test is `#[ignore]`d (~8 min run). Invoke explicitly:
```
cargo test -p aether_rt --release --test qwen25_autoregressive_gen \
  -- --ignored --nocapture
```

## What's still left for shipping matt-voice end-to-end

Two items from the prior 4-item list LANDED this session (GPU-
routed inference + tokenizer decode). Remaining gap:

1. **(DONE)** Tokenizer integration -- 152064 vocab + 151386 merges
   loaded from GGUF metadata; decode verified producing real
   English ("Hello, world! I'm a 2"). Encode side has a known
   limitation (unicode-char-level BPE not yet wired); FR-x-extra.
2. **HTTP server wrap** -- new `trainer/src/bin/serve.rs` that
   loops `accept -> parse_request -> tokenize_external_or_skip ->
   forward_loop -> render_completion`. The TCP + HTTP + OpenAI
   render fns are all shipped. Mostly composition.
3. **LoRA adapter loading** -- matt-voice's adapter from candle.
   Each block's W becomes W + (B @ A) * (alpha/rank). SafeTensors
   loader is shipped; needs an "apply lora" helper that mutates
   loaded block weights in-place.
4. **(DONE)** GPU-routed inference -- cuBLAS for every matmul,
   13x speedup, byte-identical IDs. Full-GPU (all ops on device)
   is FR-x-extra-deeper.

So the remaining matt-voice ship work is: HTTP server binary + LoRA
adapter loader + (optionally) full-text encoder. ~1 focused session.

## Full Qwen2.5-7B inference LANDED

`runtime/tests/qwen25_full_inference.rs` stacks the block-0 forward
28 times via a streaming-dequant loop. End-to-end inference through
Aether's runtime: tokens → 28 blocks → final_norm → lm_head → argmax.

Measured 270s total on 11900K release-build for a 4-token forward:
- 28 blocks × ~9s each (FFN-dominated)
- lm_head load + transpose: 5.2s
- lm_head matmul [4, 3584] @ [3584, 152064]: 14.8s

Argmax predictions for input [9707, 11, 1879, 0] ("Hello, world!"-ish):
```
pos 0 (token 9707) -> argmax 358, logit 8.04  (range -18.98..8.04)
pos 1 (token 11)   -> argmax 358, logit 10.11
pos 2 (token 1879) -> argmax 2219, logit 13.33
pos 3 (token 0)    -> argmax 358, logit 10.26
```

All logits finite, no NaN/Inf, argmax IDs in vocab. Token 358 is
"" I"" in Qwen's BPE -- a very common next-token prediction.

Test is `#[ignore]`d by default; explicit invocation:
```
cargo test -p aether_rt --release --test qwen25_full_inference -- --ignored --nocapture
```

## GPU-resident DP step LANDED

`runtime/tests/nccl_dual_gpu_resident.rs` proves that weights live
on each P100's device across N optimization steps -- the matt-voice
training-deploy shape minus QMatMul.

Per step (W never touches host):
1. compute_grad kernel on device: `grad = 2 * (W - target)`
2. NCCL all_reduce across ranks (already device-resident)
3. sgd_step kernel on device: `W -= (lr / world_size) * grad`
4. (Loss d2h only every 10 steps for logging, NOT training-critical)

Verified on cnc 2× P100:
```
[gpu-resident-dp ws=2] step=   0 loss=0.056009
[gpu-resident-dp ws=2] step=  10 loss=0.000646
[gpu-resident-dp ws=2] step=  20 loss=0.000007
[gpu-resident-dp ws=2] step=  30 loss=0.000000
[gpu-resident-dp] W on device for 50 steps, ranks byte-identical
```

nvidia-smi: PID 2560518 on BOTH GPU UUIDs at 368 MiB each.

Three CUDA kernels (compute_grad, sgd_step, sq_diff) JIT-compiled
via cudarc's nvrtc per device. Note: cudarc's nvrtc binding wants
`libnvrtc-builtins.so.12.1` even when the system CUDA is 12.8 --
symlink from torch's nvidia-cuda-nvrtc pip wheel. Documented in
`memory/cnc_nvrtc_builtins_path.md`.

## Real Qwen2.5-7B block forward LANDED

`runtime/tests/qwen25_block_forward.rs` runs the FULL decoder block 0
forward pass on matt-voice's actual Qwen2.5-7B Q4_K_M/Q6_K GGUF.

End-to-end chain through Aether's runtime:
1. Open GGUF (matt-voice 4.7 GB blob in ollama blob store)
2. Find + dequantise all 13 block-0 tensors:
   - 4 F32 (norms + 3 biases)
   - 7 Q4_K (Wq, Wk, Wo, ffn_gate, ffn_up, attn_k.bias-x, token_embd)
   - 2 Q6_K (Wv, ffn_down) -- new aether_dequant_q6_k kernel
3. Transpose weights (GGUF stores [d_in inner, d_out outer]; matmul
   wants [d_in, d_out])
4. Lookup 4 token-embedding rows -> X[4, 3584]
5. attn_norm RMSNorm (new aether_op_rms_norm_f32)
6. Q/K/V proj matmuls (uneven dims: Q[seq,3584], K/V[seq,512])
7. RoPE on Q+K (new aether_op_rope_apply_f32, Qwen base=1e6)
8. GQA repeat K/V from 4 KV heads to 28 Q heads (new
   aether_op_gqa_repeat_kv_f32)
9. Causal SDPA over 28 heads
10. Output proj + residual
11. ffn_norm RMSNorm
12. SwiGLU MLP: matmul(gate) + matmul(up) -> silu(gate)*up ->
    matmul(down) + residual

Output: max_abs=5.88, sum=-7.02, no NaN/Inf, total time 9.36s in
release build on 11900K. Per-token L2 norms differ (row0=29.0,
row3=16.4) -- attention mixed information across positions, not a
trivial pass-through.

Three new ops shipped this session:
- `aether_op_rms_norm_f32(x, gamma, eps, out, rows, d)` -- the
  Qwen/Llama-style normalisation (no beta).
- `aether_op_rope_apply_f32(x, seq, n_heads, head_dim, base, pos_start)`
  -- rotary embeddings in-place, llama-style "half-half" pair
  layout.
- `aether_op_gqa_repeat_kv_f32(in, out, seq, n_kv, head_dim, n_q)`
  -- GQA broadcast.

Two new GGUF helpers:
- `aether_gguf_find_tensor_by_name(handle, name, n)` -- linear-scan
  lookup; avoids manually iterating all 339 tensors.
- `aether_gguf_get_tensor_n_elems(handle, i)` -- product of dims.

Plus the Q6_K dequantisation kernel
(`aether_dequant_q6_k`) ported from ggml's reference decoder.

5 new unit tests for the three ops (RMSNorm scale invariance + unit
gamma; RoPE pos-0 identity + L2 norm preservation; GQA repeat) +
1 Q6_K dequant test against real Qwen V-proj weights (max_abs=0.023).

## Dual-P100 training LANDED

`aether-train --world-size 2 --features nccl` trains a model across
both P100s end-to-end via NCCL gradient all-reduce. Verified on
cnc-server:

```
[aether-train-dp ws=2] step=    0 loss=5.5676 lr=3.00e-4 elapsed=0.1s
[aether-train-dp ws=2] step=   50 loss=2.1998 lr=2.68e-3 elapsed=3.4s
[aether-train-dp ws=2] step=  100 loss=0.1044 lr=1.62e-3 elapsed=6.8s
[aether-train-dp ws=2] step=  150 loss=0.0420 lr=4.84e-4 elapsed=10.1s
[aether-train-dp ws=2] step=  199 loss=0.0307 lr=2.05e-7 elapsed=13.3s
[aether-train-dp] final params identical across ranks: true (sampled first 8 of 85504)
```

nvidia-smi proof: aether-train PID 2044863 visible on BOTH GPU
UUIDs simultaneously (368 MiB on `bb77bda0...` first P100,
368 MiB on `17bd0d20...` second P100). Real cross-card data
exchange via libnccl 2.21.5 ncclAllReduce.

Shape of the loop:
- `trainer/src/dp.rs::train_dp` — N Model instances (same seed),
  per-step: per-rank batch shard, forward/backward, h2d host grads,
  group_start/per-rank all_reduce(Sum)/group_end, d2h, scale by
  1/world_size, AdamW. Final-state invariant check confirms ranks
  end with identical params.
- `trainer/Cargo.toml` `nccl` feature forwards to `aether_rt/nccl`
  + adds cudarc 0.13 for the typed group_start/end API.
- `trainer/src/main.rs --world-size N` dispatches to train_dp.

## What's Next

Items 1 + 2 from the prior "What's Next" + the GPU-resident
weights variant + FR-18.1-extra + dual-P100 training all shipped
this session. Remaining matt-voice deploy work:

1. **(DONE)** Multi-rank training-loop bringup. `aether-train
   --world-size 2 --features nccl` works on cnc 2× P100; loss
   declines, params identical at convergence.
1b. **(DONE)** Full-model multi-block stack -- 28 decoder blocks +
   final_norm + lm_head ship in `qwen25_full_inference.rs`. Streaming
   dequant (~870 MB peak). Argmax predictions verified.
1c. **(DONE)** GPU-resident DP step -- W on device across iters,
   loss converges to 0 on cnc 2× P100. The matt-voice training-
   deploy shape minus QMatMul.
2. **Pipeline-parallel 1F1B real impl** (matt-voice §FR-18.6).
   Today's `aether_pp_simulate_2stage_forward_f32` is in-process.
   Real PP needs send/recv between adjacent ranks via the
   `Comm::send` / `Comm::recv` surface in cudarc::nccl. Foundation
   shipped; scheduling is the remaining work.
3. **All ops on device** (FR-19.16-extra deepest). LN / SDPA / SiLU
   still run CPU-side and drive the per-iter h2d/d2h pattern in
   `aether_llm_inference_bench_tps_cuda_resident`. Route those
   through cuda.rs's existing device-kernel variants.
4. **Forward-pass over a whole transformer block on real Qwen2.5
   weights**. The one-block witness proves dequant → matmul
   composes. Next is iterating the chain through Q/K/V/O matmuls +
   attention + MLP for one full block.
5. **Llama-1B-scale dims** (d=2048, ff=5504, 16 layers). At those
   dims cuBLAS sgemm dominates and the resident bench headline
   grows much larger.
6. **Phase 15 leftovers**: FR-15.7 (SWP), FR-15.10 (hand-asm gate
   for the v4 SHIP perf claim).
7. **Phase 16 leftovers**: proc-macros, Drop, slice/str primitives.

NOTE on TCP test flake: `tests::tcp_send_recv_loopback` at
`runtime/src/lib.rs:3492` fails ~1/3 runs with "accept returned -1"
under Windows firewall / port contention. Unrelated to any 2026-05-19
work. If the audit-agent protocol re-runs it during honesty-auditor
and reports it as failing, it's a known flake.

Phase 18's remaining 2 are hardware-blocked; not next-session
attackable.

## Notes for Next Session

- **`aether_copy_cstr` is the new go-to** for passing string
  literals to extern fns. The witness footprint shrinks dramatically
  (90 lines → 50 lines is typical). See
  `memory/aether_copy_cstr_pattern.md`.
- **GGUF v3 has a BOOL=1-byte pitfall** at value type 7. My parser
  groups it correctly with u8/i8 in the 1-byte size branch;
  preserving that grouping is critical when adding new GGUF features.
  See `memory/gguf_v3_bool_pitfall.md`.
- **matt-voice's actual Qwen2.5-7B blob** is at
  `C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`
  (4.7 GB Q4_K_M). The GGUF reader walks it cleanly; tensor 0 is
  `token_embd.weight`. See `memory/matt_voice_qwen_blob_location.md`.
- **FR-x-extra tags reuse the parent's primary roadmap ID** —
  audit count doesn't advance, but the work is real. The
  honesty-auditor checks impl + non-claims carve-outs, not the
  audit-count delta. See `memory/fr_x_extra_tag_convention.md`.
- **`--features cuda` build is now active.** New witnesses that
  exercise the GPU should tag `// requires: cuda`. The audit's
  runtime_check.rs detects cuda via "cublas"/"cudart6" symbol
  presence in libaether_rt.a (now present).
- **Llama-1B was NOT found locally** despite the user thinking it
  was downloaded. If matt-voice deploy needs Llama-3.2-1B specifically
  (vs the local Qwen2.5-7B), that's a 1.3 GB HF download (auth-
  gated). Most matt-voice work targets Qwen anyway per
  `MATT_VOICE_FR.md`.
- **honesty-auditor protocol still applies**. 51/51 claims this
  session held up — keep using it on perf-relevant or claim-heavy
  work.
- **No Python for tooling.** Same as always.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime (default): `cargo build -p aether_rt`
- Build runtime (cuda): `cargo build -p aether_rt --features cuda`
- GGUF walk witness: `cargo run --bin aetherc -- tests/runtime/gguf_qwen25_walk.aether --emit=aether-bin -o scratch/gguf.exe`
- tok/s bench: `tests/runtime/llm_inference_tps.aether` (~177 tok/s on Llama-shape)
- Qwen2.5 blob path: `C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`
- matt-voice FR list: `MATT_VOICE_FR.md`
- ant-brain FR list: `ANTCOLONY_FR.md`
- v4 FR queue: `NEXT-UP.md`

## Commits this session (all pushed to origin/main)

```
32784f7 → 172f423   (9 commits)
141/196 → 169/196   (+28 audit slots, 86% coverage)
Phases at 100%: 6-14 + 17 + 19  (10 phases)
honesty-auditor claims: 51/51 verified
```

---

## External findings — substrate-swap attempt (kokonoe session, 2026-05-24 ~04:30-09:50 PDT)

A separate Claude session on kokonoe attempted Phase 1 of the openclaw-inference fleet substrate swap (workhorse: llama-server → aether-serve hosting GLM-4.7-flash UD-IQ3_XXS). **Swap was reverted at the user's call.** Surfacing the findings here so next aether-side session has them before re-attempting.

### TL;DR

The swap was **technically operational** — aether-serve hosted GLM under systemd, routed end-to-end through LiteLLM — but **functionally unusable for chat consumers** because aether-serve's text→token encoding isn't wired. Every LiteLLM consumer (school teams, redteam adversaries, anything passing `messages: [...]`) got `text-prompt encode not wired yet; pass prompt_ids (token ids) for now`. Anthropic OAuth fallback also broken (separate ~8h-cliff issue). Workhorse reverted to llama-server.

### Three concrete blockers to fix before retrying

**1. aether-serve text-prompt encoding (THE blocker).** `POST /v1/chat/completions` with `{"messages":[{"role":"user","content":"hi"}]}` against GLM-4.7-flash returns `text-prompt encode not wired yet; pass prompt_ids (token ids) for now`. Validated against the live :8081 deployment and against the earlier :18913 nohup launch — same error path. Per the commit history, WordPiece + BPE exist for BERT (`92a5054`, `8985d85`) but the chat path for GLM/Qwen/DeepSeek doesn't tokenize text. Either wire the tokenizer into the chat-completion handler, or add a tokenizer sidecar in front.

**2. cnc systemd `.service.d/` drop-in overrides main unit `Environment=` silently.** Each `openclaw-inference-<name>.service` has a drop-in at `/etc/systemd/system/openclaw-inference-<name>.service.d/ld-library-path.conf` that was originally set to `LD_LIBRARY_PATH=/opt/llama/llama-b8182:/usr/local/cuda-12/lib64` (llama.cpp's bundled libs). **Drop-ins win over main-unit `Environment=` lines**, so rewriting the main unit's LD_LIBRARY_PATH and `daemon-reload`-ing did NOT change the effective env. Cost three crash-loops before `systemctl cat <unit>` (which shows ALL files contributing in apply order) revealed it. When retrying, edit the drop-in, not the main unit:

```ini
# /etc/systemd/system/openclaw-inference-workhorse.service.d/ld-library-path.conf
[Service]
# aether-serve needs NVRTC builtins from the ml-venv pip-installed nvidia package
# (libnvrtc-builtins.so.12.1 lives only there; not in any system path)
Environment=LD_LIBRARY_PATH=/opt/ml-venv/lib/python3.13/site-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda-12.8/lib64:/usr/local/lib
```

The `libnvrtc-builtins.so.12.1` lives ONLY at the ml-venv path above. Without it in LD_LIBRARY_PATH, aether-serve panics at `runtime/src/cuda.rs:3876:43` with `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE`.

**3. aether-serve binds `127.0.0.1` only, despite `--help` claiming `0.0.0.0`.** `--help` says "Listens on 0.0.0.0:port for OpenAI-compat /v1/chat/completions." Actual binding (`ss -tlnp` on PID 1646339 + the earlier :18913 nohup launch): `127.0.0.1:8081` only. No `--host` flag. Binary `strings` contains both `127.0.0.1` and `0.0.0.0`, so this looks like an actual aether-serve binary bug, not a missed CLI option. **This blocks LiteLLM in podman bridge mode from reaching aether-serve**, because `host.containers.internal → 10.88.0.1` (podman0 bridge) which can't reach host loopback.

**Workaround that worked: `systemd-socket-proxyd` bridge.** Two unit files left in place on cnc, **disabled**, ready to re-enable on retry:

- `/etc/systemd/system/aether-workhorse-bridge.socket` — listens on `10.88.0.1:8081`
- `/etc/systemd/system/aether-workhorse-bridge.service` — runs `/usr/lib/systemd/systemd-socket-proxyd 127.0.0.1:8081`

Reusable for scout + embed bridges by copying with port substitution. (socat would be cleaner but isn't installed on MicroOS and needs `transactional-update + reboot`.)

### Side-effect: ~14-min cold load due to memory pressure

cnc has **15 GiB RAM**, not 8 (the "8 GB box" line in earlier aether-session handoffs likely refers to one P100's VRAM, not host memory). GLM-4.7-flash UD-IQ3_XXS takes **~10 GiB resident + ~2 GiB swap** during load while scout + embed llama-servers also hold host RAM. The process sits in state `D` (disk sleep) for most of the 14 minutes; don't kill it — watch GPU VRAM grow as the progress signal. **Stop scout + embed before starting the aether workhorse next time** — frees ~5 GiB RAM and should cut load to a few minutes.

### Independent issue: Anthropic OAuth fallback was down at smoke time

LiteLLM tried to fall back to `claude-sonnet-4-6` when aether-serve returned 501; that also failed: `upstream connect error or disconnect/reset before headers. reset reason: connection termination`. Consistent with the OpenClaw fleet's ~8h OAuth refresh cliff (see kokonoe memory `reference_openclaw_oauth_refresh_400`). Refresh the fleet main's OAuth before retrying so consumers cleanly fall back during any future swap attempt.

### Retry checklist

1. **Wire text-prompt encoding in aether-serve** — the only real blocker. (Without this, the swap is operational but every chat consumer 501s.)
2. **Refresh Anthropic OAuth** on fleet main (`openclaw agents add main`) so fallbacks work during the retry window.
3. **Stop scout + embed before starting aether workhorse** — cuts cold-load from ~14 min to ~3 min.
4. **Update the `.service.d/` drop-in**, not the main unit, for LD_LIBRARY_PATH (the ml-venv NVRTC path above).
5. **Re-enable the bridge socket**: `sudo systemctl enable --now aether-workhorse-bridge.socket`.
6. **Smoke via LiteLLM with `model: llama-3.3-70b-versatile`** (the consumer-facing alias; raw `qwen2.5-14b` is the upstream name and gets rejected by LiteLLM's `model_list`).

### Files left on cnc (kept for retry)

- `/etc/systemd/system/openclaw-inference-workhorse.service.bak.1779635911` — original llama-server unit (current main was restored from this)
- `/etc/systemd/system/openclaw-inference-workhorse.service.d/ld-library-path.conf.bak.1779636381` — original llama-server drop-in (current was restored from this)
- `/etc/systemd/system/aether-workhorse-bridge.{socket,service}` — disabled, ready to re-enable

### Concrete fleet → aether endpoint map (validated at smoke time)

```
LiteLLM container (bridge mode, 10.88.0.2)
  → host.containers.internal (10.88.0.1, podman0)
  → systemd-socket-proxyd on 10.88.0.1:8081
  → aether-serve on 127.0.0.1:8081
  → GLM-4.7-flash UD-IQ3_XXS on P100 #2 (CUDA_VISIBLE_DEVICES=1, ~12.7 GiB VRAM)
```

The route works. The cargo doesn't survive aether-serve's tokenization gap.

## 2026-05-25 — PER-ARCH FIX RESULTS (post Matt's verification; 2 partial-blackout cnc smoke windows)

Acted on the queue. aether-serve rebuilt on cnc with 2 fixes; batch re-smoked on GPU1 (workhorse-only evict, scout+aether-serve stayed up):

| # | Fix shipped | Re-smoke result | State |
|---|---|---|---|
| 1 qwen3moe | Q3_K dt=11 upload arm (`upload_tensor_u8`+`_opt`, commit c8f882c) | **LOADS now** (was upload panic); NEXT gap: MoE **expert** matmul lacks Q3_K → panic serving.rs:1802 | 🟡 advanced |
| 3 gemma3 | tied-embeddings → `token_embd` lm_head fallback (serving.rs:2150) | **LOADS now** (was output.weight panic); forward emits pad/vocab-1 | 🟡 advanced |
| 2 V2-Lite | (retest only) | **NOT a bug** — 13 real varied tokens; incoherent only w/o chat template | ✅ closed |
| 4 codestral/IQ3_M | (isolation probe) | **IQ3_M QUANT is the bug, not arch** — R1-Distill-Qwen-32B-IQ3_M (qwen2 arch, fine on Q4_K) ALSO NaN-pins vocab-1. codestral has a separate tokenizer vocab-byte-10 gap. | 🔍 root-caused |
| 5 qwen3-dense | (none — HW) | No small qwen3-dense GGUF exists; 32B covered by 2×P100 PP path | ⛔ HW-blocked |

### Remaining work (post-fix)
- **qwen3moe expert Q3_K kernel** (~80 LoC, mirror IQ3_S/IQ4_XS expert kernels): new `fused_q3_k_expert_matmul_seq1` in cuda.rs + `ExpertDtype { dt:11, block_n_elems:256, kernel:... }` row in `MOE_EXPERT_DISPATCH`. The panic msg at serving.rs:1802 gives the exact template. Does NOT exist yet (grep cuda.rs = 0).
- **gemma3 forward** (open-ended bisect, cnc): loads but pad/vocab-1 → gemma3-specific path (pre/post norms, attn logit softcap, embedding scale) not fully right. Bisect like the GLM NaN chase (AETHER_DUMP_BLOCKS).
- **IQ3_M quant** (the high-value one): one fix clears ALL IQ3_M models. IQ3_M is NOT in the dispatch dtype set (12/8/6/21/23/18 + matmul dtypes) — it's a distinct GGUF quant (≈ mixed IQ3_S/IQ3_XXS). Either it's silently mis-dispatched → NaN, or needs its own dequant+matmul kernel. Investigate which dt code IQ3_M maps to + whether a kernel exists.
- **V2-Lite**: just needs the chat template applied in the serving path (or known-good prompt_ids) — not a forward fix.

## 2026-05-25 — PER-ARCH ROUND: SESSION-END STATE (after shipping the fixes)

Worked the queue to its natural stopping point. **5 partial-blackout workhorse-only
evictions** (scout+aether-serve stayed up each time), all restored+health-checked.

| Arch | State | Detail |
|---|---|---|
| qwen2.5, GLM(MLA-absorbed), BGE | ✅ **witness-ready** | coherent decode/embeddings (earlier smokes) |
| V2-Lite | ✅ **not-a-bug** | real varied tokens; incoherent only w/o chat template — close |
| qwen3-dense 32B | ⛔ **HW-blocked** | no small GGUF; covered by 2×P100 PP path |
| **qwen3moe** | 🟡 **dispatch DONE, forward parked** | Q3_K(dt=11)+Q5_K(dt=13) expert kernels shipped → full MoE forward now RUNS (17 tok, no panic). Output vocab-1/pad → forward bug. |
| **gemma3** | 🟡 **load DONE, forward parked** | tied-embed lm_head fallback + embedding ×√d_model shipped → loads. Output vocab-1/pad → forward bug (next: token_embd dtype — dequant_embd_row hardcodes Q4_K host dequant; + logit softcap). |
| **IQ3_M (codestral/R1-distill)** | 🔍 **root-caused, parked** | IQ3_M *quant* bug, NOT arch (R1-Distill is qwen2-arch, fine on Q4_K, NaN on IQ3_M). codestral also has a tokenizer vocab-byte-10 gap. |

### THE pattern (high-value next-session lead)
**qwen3moe + gemma3 + IQ3_M all show the SAME vocab-1/pad NaN-pin** at the logits.
The dispatch/load fixes are all shipped; what remains is a **forward-correctness
bisect**, and the shared symptom suggests **one root cause, not three** — worth a
single `AETHER_DUMP_BLOCKS`/`AETHER_DUMP_LOGITS` bisect (GLM-gate-close pattern) to
find where NaN/degeneracy enters, likely a code path the working archs (qwen2.5/GLM)
don't hit. Code work on kokonoe; eviction smokes only to verify.

### Commits this round
c8f882c Q3_K upload arm · (gemma3 lm_head) · (gemma3 embed scale) · Q3_K expert kernel · Q5_K expert kernel

## 2026-05-25 — HIGH-VALUE LEAD RESOLVED: shared vocab-1 root cause = wrong token_embd dtype

Found via a FREE GGUF dtype probe (no GPU/eviction): `dequant_embd_row` hardcoded
Q4_K, but `token_embd` is **Q6_K (gemma3) / Q3_K (qwen3moe) / IQ3_S (R1-distill/IQ3_M)**
→ garbage embeddings → the vocab-1/pad NaN-pin across all three. (qwen2.5/GLM worked:
their token_embd is Q4_K.) Fix: dtype-aware `dequant_embd_row` + new host dequants
`aether_dequant_q3_k` + `aether_dequant_iq3_s` (incl the 512-entry iq3s_grid),
CPU ports of the device decoders.

**Payoff smoke (one fix, three arches):**
| arch | before | after |
|---|---|---|
| IQ3_M (R1-Distill-Qwen-32B-IQ3_M) | `[PAD152063]` | **"The ocean is a vast and vital"** — ✅ FULLY COHERENT |
| qwen3moe (Qwen3-30B-A3B-Q3_K_M) | `[PAD151935]×17` | real varied tokens — forward FIXED (rambly only w/o chat template) |
| gemma3 (gemma-3-12b) | `[PAD262207]` | `"T"`+stop — no longer pad-pinned, but **still degenerate → needs MORE** |

### Updated per-arch ledger
- **witness-ready:** qwen2.5, GLM(MLA-absorbed), BGE, **+ IQ3_M-class** (R1-Distill coherent).
- **forward-fixed, pending chat-template wiring:** qwen3moe, V2-Lite (both real tokens; serving path needs per-model chat template applied for coherent instruct output).
- **still parked (narrowed):** gemma3 — embed dtype was necessary but not sufficient; remaining gemma-specifics to check: attention logit soft-cap, `query_pre_attn_scalar` scaling, per-layer sliding-window alternation. Code work on kokonoe.

### Commits (root-cause arc)
Q3_K upload arm · gemma3 lm_head fallback · gemma3 embed scale · Q3_K expert kernel ·
Q5_K expert kernel · **dtype-aware token_embd dequant (the root-cause fix)**
