# Aether — Session Handoff

## Last Updated
2026-05-19 (matt-voice deploy pack — 5 FR-x-extras shipped)

## Project Status
🟢 **Audit: 169/196 (86%).** Audit count unchanged from prior
commit (the 4 code extras tag already-witnessed primary slots —
the right kind of "depth over breadth" progress). Substantive
deliverables: cuda runtime build now live, 4 matt-voice-critical
FR-x-extras shipped, all honesty-auditor verified.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  — unchanged
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   20/20 witnessed (100%) — unchanged
Phase 18:    9/11 witnessed (81%)  — unchanged
Phase 19:   16/16 witnessed (100%) — unchanged
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    169/196 (86%)
```

## What Was Done This Session

The user requested: "Target all of those relevant extras". The
5-item matt-voice deploy pack:

### 1. cuda runtime build (configuration win)

`cargo build -p aether_rt --features cuda` succeeds on kokonoe.
CUDA toolkit v12.6 detected; cudarc 0.13 with feature
`cuda-12060` builds clean. The resulting libaether_rt.a is now
~64MB (up from ~5MB) with 39507 cuBLAS-symbol matches.

Practical proof: `cuda_train_tiny.aether` now exits 0 through
REAL cuBLAS sgemm + nvrtc-JIT'd custom kernels (it was skipped
on the previous default build). This unlocks the cuda-routed
path for ALL existing `// requires: cuda` witnesses.

### 2. FR-17.19-extra — SafeTensors multi-tensor parser

3 new extern fns in `runtime/src/lib.rs`:
- `aether_safetensors_n_tensors(buf, len) -> c_int`
- `aether_safetensors_get_shape(buf, len, name, n_name, out_dims, max_dims) -> c_int`
- `aether_safetensors_get_dtype(buf, len, name, n_name) -> c_int`
  (enum: 0=F32, 1=F16, 2=BF16, 3=I32, 4=I16, 5=U8, 6=I64)

Walks the SafeTensors JSON header with depth-tracking brace/quote
state machine. The existing single-tensor `safetensors_get_tensor_f32`
covers the read path; these new fns cover the metadata-discovery
path matt-voice needs to walk Qwen2.5's `model.safetensors`.

Witness `tests/runtime/safetensors_multi.aether` (tag P17.19)
builds a 2-tensor blob via `aether_copy_cstr` (new helper) +
`aether_byte_set`, runs all 3 lookups, exits 42.

### 3. FR-17.14-extra — Q4_K_M dequant kernel

Real ggml Q4_K_M block layout:
- 144 bytes per super-block of 256 quants
- bytes 0-2: f16 d (super-block scale)
- bytes 2-4: f16 dmin (super-block min)
- bytes 4-16: 12 bytes of packed 6-bit scales (8) + mins (8)
- bytes 16-144: 128 bytes of 4-bit quants

`q4k_get_scale_min` replicates ggml-quants.c's `get_scale_min_k4`
6-bit packing (j<4: bottom 6 bits of scales[j]; j>=4: composite
from byte j+4 and j-4 with `>>6<<4` upper bits). Dequant per
sub-block: `val = d * sc * q - dmin * m`.

Witness `tests/runtime/q4_k_dequant.aether` (tag P17.14) hand-
builds a Q4_K block with d=1.0, dmin=0, sub-block-0 scale=1,
min=0, then verifies sub-block 0's 32 outputs equal `l & 0xF`
for l in 0..32. Exits 42.

matt-voice's Qwen2.5-7B uses Q4_K_M exactly.

### 4. FR-19.9-extra — HF tokenizer.json loader

3 new extern fns:
- `aether_bpe_add_token_with_id(handle, token_id, bytes, n)`
- `aether_bpe_add_merge_by_id(handle, left_id, right_id, rank, merged_id)`
- `aether_tokenizer_json_load(handle, json, n) -> n_merges_loaded`

Hand-walks the HF `{"model":{"vocab":{...},"merges":[[...],...]}}`
JSON shape. Registers tokens at their EXPLICIT HF ids (essential
for matt-voice's Qwen2.5 weight indexing — the model's embedding
matrix is keyed by token id). Returns merges-loaded count or
-1/-2 on parse/lookup failure.

Witness `tests/runtime/tokenizer_json_load.aether` (tag P19.9)
loads a tiny 8-token/4-merge tokenizer JSON. Exits 42 iff
n_merges == 4.

### 5. FR-19.10-extra — chat_template.jinja file loader

`aether_template_render_from_file` reads the template from disk
via `std::fs::read` + dispatches to `aether_template_render`.

Witness `tests/runtime/chat_template_from_file.aether` (tag
P19.10) writes a template to disk via `aether_write_file`, reads
it back via `_render_from_file`, verifies the rendered output
is "[user: hi]". Exits 42.

### Plus: `aether_copy_cstr` helper

`aether_copy_cstr(dst, cstr, max) -> n_bytes_copied`. Copies a
NUL-terminated string (the form `Expr::StrLit` lowers to in the
asm backend, sitting in `.rdata` at a leaq-resolved address)
into a heap buffer. Unblocks witnesses that need to pass multi-
character literals to extern fns without per-byte
`aether_byte_set` calls. Used by 3 of the 4 new witnesses above.

### honesty-auditor

7/7 claims verified. The 5-item batch is honest:
- cuda build proven by 39507 cuBLAS-symbol matches +
  cuda_train_tiny exit=0 (REAL loss decrease, not a stub).
- 4 deepenings have real runtime impls (no stub/todo/
  unimplemented anywhere).
- Each witness exits 42 with byte-exact / value-exact asserts.
- Audit count correctly unchanged (FR-x-extras don't inflate
  the witness count by design).

## Current State

**Working:**
- 169/196 audit-tagged witnesses pass.
- `cargo build -p aether_rt --features cuda` succeeds.
- Witnesses tagged `// requires: cuda` now route through cuBLAS.
- matt-voice's Qwen2.5 deploy critical path:
  - ✅ FR-19.9 BPE algorithm
  - ✅ FR-19.10 chat template engine
  - ✅ FR-19.9-extra tokenizer.json loader
  - ✅ FR-19.10-extra chat_template.jinja file loader
  - ✅ FR-17.14-extra Q4_K_M dequant (Qwen2.5-7B Q4_K_M format)
  - ✅ FR-17.19-extra SafeTensors multi-tensor parser
  - ✅ cuda runtime path live (cuBLAS sgemm + nvrtc kernels)
  - ⏳ FR-19.1-extra full TLS 1.3 handshake (XL, multi-session)
  - ⏳ FR-17.19-extra-deeper real 1.3 GiB Llama-1B weight bundle
  - ⏳ FR-19.5-extra real continuous batching via cuda matmul
  - ⏳ FR-18.1-extra real libnccl link (needs 2nd GPU on cnc)
  - ⏳ FR-19.16-extra Llama-1B at 100 tok/s on 3070 Ti (gate
       composite of above)

**Honest scope notes:**
- The 4 deepenings are EXTRAS — they extend, not replace, the
  primary FR witnesses. Each tagged with the parent FR's primary
  ID; audit count unchanged.
- The cuda build is a configuration win, not a code shippable.
  The `cuda_train_tiny` test exit=0 is the practical proof.
- Real Llama-1B inference at 100 tok/s on the 3070 Ti still
  requires: download + memory-map the 1.3 GiB weight bundle,
  cuda matmul through the SafeTensors-loaded tensors, real
  continuous batching wiring. Multi-session work.

## Blocking Issues

None for the language. Remaining gates are multi-session XL
items (full TLS handshake) or hardware-binding (libnccl on the
cnc 2× P100 path; the 3070 Ti at-scale Llama-1B bench requires
the weight bundle download + cuda-route the bench fn).

## What's Next

`NEXT-UP.md` is the queue. Path to matt-voice deploy via Aether:

1. **FR-19.1-extra TLS 1.3 (XL, multi-session)** — ChaCha20-
   Poly1305 already shipped; need HMAC-SHA256 + X25519 + Ed25519
   + AES-GCM + handshake state machine. The long pole.
2. **FR-17.19-extra weight-bundle path** — download Llama-1B
   safetensors / 1.3 GiB, mmap, route through the SafeTensors
   multi-tensor parser shipped today. M-L effort.
3. **FR-19.5-extra real continuous batching** — wire the in-
   process sim shipped in Phase 19 closeout to a real
   cuda-routed forward loop. M effort.
4. **FR-19.16-extra Llama-1B at 100 tok/s** — composite of the
   above. Bench appends to BENCH_LEDGER's `bench/llama_inference`
   section.
5. **FR-18.1-extra libnccl link** — hardware-binding (cnc 2× P100).

Phases still under 100%:
- Phase 15: 8/10 (FR-15.7 SWP, FR-15.10 hand-asm gate)
- Phase 16: 22/25 (proc-macros, Drop, slice/str)
- Phase 18: 9/11 (hardware-blocked only)
- Phase 20: 7/10 (self-hosted asm emitter XL)
- Phase 21: 4/10 (Mach-O/ELF/ARM/WASM)
- Phase 22: 6/10 (LSP, DAP, fuzzing)
- Phase 23: 2/6 (synthesis)
- Phase 24: 7/10 (sanitizers, hot-reload)

## Notes for Next Session

- **`aether_copy_cstr` is the new go-to for passing string
  literals to extern fns.** Use it instead of per-byte
  `aether_byte_set` chains. The witness footprint shrinks
  dramatically — `safetensors_multi.aether` went from 90 lines
  of byte_set chains to 50 lines of `copy_cstr` calls.
- **`Expr::StrLit` lowers to `leaq sym(%rip), %rax`** (an i64
  pointer to a NUL-terminated string in .rdata). Pass that
  pointer to any runtime fn that takes an i64 cstr arg; the
  runtime side does strlen + read.
- **cuda build is now active.** New witnesses that should run on
  GPU should tag `// requires: cuda`. The audit's
  `runtime_check.rs` detects this via "cublas"/"cudart6" in
  libaether_rt.a (now present).
- **FR-x-extra tagging convention**: deepenings of an existing
  FR tag the parent's primary ID. The audit dedupes, so the
  witness count is unchanged. The substantive progress lives in
  the runtime impl + the second witness file.
- **matt-voice critical path** (per `MATT_VOICE_FR.md`): all
  the Aether-side LANGUAGE work for matt-voice's Qwen2.5
  deploy is now in place. Remaining is hardware/network-
  binding (real Llama weights, real TLS, real libnccl).
- **No Python for tooling.** Same as always.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build runtime (default): `cargo build -p aether_rt`
- Build runtime (cuda): `cargo build -p aether_rt --features cuda`
- SafeTensors multi witness: `cargo run --bin aetherc -- tests/runtime/safetensors_multi.aether --emit=aether-bin -o scratch/stm.exe`
- Q4_K dequant: `tests/runtime/q4_k_dequant.aether`
- tokenizer.json load: `tests/runtime/tokenizer_json_load.aether`
- chat_template file load: `tests/runtime/chat_template_from_file.aether`
- matt-voice FR list: `MATT_VOICE_FR.md`
- ant-brain FR list: `ANTCOLONY_FR.md`
- v4 FR queue: `NEXT-UP.md`

## Commits this session

```
e5fa443 Path C FR-17.3: conv2d CPU direct-loop reference
976dbce Phase 17 closeout: Q4_0 + FA2 + layer modules f32 + Llama-shaped partial
a8214f6 Phase 18 closeout: NCCL surface + PP/TP/FSDP/ZeRO/overlap/grad_compress sims
499c49e Phase 19 kickoff: FR-19.9 byte-level BPE tokenizer
ace5367 Phase 19 advance: FR-19.10 Jinja-lite chat template renderer
a1ddb5f Phase 19 closeout: 13 items
217934d Phase 19 100%: FR-19.16 partial — Llama-shape tok/s bench >=100
(pending) matt-voice deploy pack: 5 FR-x-extras (cuda + SafeTensors + Q4_K + tokenizer.json + chat_template loader)
```
