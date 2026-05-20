# Aether — Session Handoff

## Last Updated
2026-05-19 (FR-18.1-extra LANDED: real cross-card NCCL on cnc 2× P100; data-parallel training-step witness verified end-to-end)

## Project Status
🟢 **Audit: 169/196 (86%) — 10 of 19 phases at 100%**. matt-voice's
serving-deploy critical path within Aether's language + runtime is
materially complete: GGUF reader + Q4_K_M dequant + cuda routing
live, tokenizer.json + chat_template loaders, SafeTensors multi-
tensor parser. Remaining gates are multi-session work (full TLS
1.3, real forward pass through real weights at scale) or hardware-
binding (libnccl cross-card on cnc 2× P100).

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

Workspace tests: 134+ unit tests pass.
Honesty scan: 0 todo / 0 unimplemented / 4 known-OK stubs (unchanged).

## What Was Done This Session

Nine commits, pushed to `origin/main`:

```
32784f7 Path A FR-15.1: SSA-driven opt pipeline rewrites AST at --O1
ffb2336 Path A FR-15.2: regalloc plan drives r12..r15 promotion
8cae67c Path A FR-15.3: AVX2 emit via aether_asm + dot builtin
e5fa443 Path C FR-17.3: conv2d CPU direct-loop reference
976dbce Phase 17 closeout: Q4_0 + FA2 + layer modules f32 + Llama partial
a8214f6 Phase 18 closeout: NCCL + PP + TP + FSDP + ZeRO + overlap + grad_compress
499c49e Phase 19 kickoff: FR-19.9 byte-level BPE tokenizer
ace5367 Phase 19 advance: FR-19.10 Jinja-lite chat template
a1ddb5f Phase 19 closeout: 13 items (PKV/CB/specdec/MM/tool/rate/obs/vision/speech/ChaCha20/HTTP/OpenAI/WS)
217934d Phase 19 100%: FR-19.16 partial tok/s bench (177 tok/s, ≥100 ✓)
3283015 matt-voice deploy pack: 5 FR-x-extras (cuda + SafeTensors + Q4_K + tokenizer.json + chat_template.jinja)
172f423 FR-17.14-extra-deeper: real GGUF reader walks Qwen2.5-7B
```

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

**Working:**
- 169/196 audit-tagged witnesses pass.
- 10 phases at 100% (6-14 + 17 + 19).
- `cargo build -p aether_rt --features cuda` succeeds; cuBLAS path
  live for any `// requires: cuda` witness.
- Real Qwen2.5-7B Q4_K_M GGUF readable via `aether_gguf_*` extern
  surface.
- matt-voice's serving-deploy critical path's LANGUAGE work:
  - ✅ BPE algorithm + chat template engine
  - ✅ tokenizer.json + chat_template.jinja file loaders
  - ✅ Q4_K_M dequant + GGUF reader
  - ✅ SafeTensors multi-tensor parser
  - ✅ cuda runtime path live (cuBLAS sgemm + nvrtc kernels)
  - ⏳ Full forward pass through real Qwen2.5 weights at scale
  - ⏳ FR-19.1-extra full TLS 1.3 handshake (XL)
  - ⏳ FR-18.1-extra real libnccl link (hardware-binding)
  - ⏳ FR-19.16-extra Llama-1B at 100 tok/s on 3070 Ti (composite)

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

None on the kokonoe-local side. Remaining gates are:
- **FR-19.1-extra full TLS 1.3** — XL effort, multi-session.
- **FR-17.19-extra-deeper Llama-1B real weights** — needs the
  ~1.3 GB Llama-3.2-1B SafeTensors download (auth-gated HF) OR
  use the local Qwen2.5-7B (already loaded, but bigger model).
- **FR-19.16-extra real ≥100 tok/s on real Llama-1B / 3070 Ti**
  — composite of the above + cuda matmul wiring through the
  dequant chain.
- **FR-18.1-extra real libnccl** — needs the cnc 2× P100 box
  (kokonoe is single-GPU).

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

## What's Next

Items 1 + 2 from the prior "What's Next" + the GPU-resident
weights variant + FR-18.1-extra all shipped this session. Remaining
matt-voice deploy work:

1. **Multi-rank training-loop bringup** on the new NCCL surface. The
   FR-18.1-extra witnesses prove all_reduce + per-rank gradient
   exchange work; the next step is plugging that into the trainer
   crate so `cargo run --bin aether-train --features nccl` actually
   trains a model across both P100s. Need: multi-process spawn (one
   process per rank) OR a single-process multi-stream variant.
   Simplest first cut: data-parallel mode + AdamW + the train_tiny
   shape from existing CPU witnesses.
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
