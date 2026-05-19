# Aether — Session Handoff

## Last Updated
2026-05-19 (Path C pickup — FR-17.3 conv2d CPU reference shipped)

## Project Status
🟢 **Audit: 145/196 (73%) roadmap items witnessed.** +1 from session
baseline of 144/196. 0 errors, all workspace tests green (including 2
new conv2d unit tests). Honesty scan unchanged (4 known-OK stubs).
Path C kicked off with the cleanest audit-missing item: P17.3 (Phase
17 is now 19/20 — only Llama-1B reference XL gate left in this phase).

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  — unchanged (Path A complete on 5/18)
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   19/20 witnessed (95%)  ← +1 (FR-17.3 conv2d CPU reference)
Phase 18:    2/11 witnessed (18%)  — unchanged
Phase 19:    0/16 witnessed (0%)   — unchanged
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    145/196 (73%)
```

Workspace tests: 127 pass (3 ssa_drive + 3 regalloc_plan + 9 AVX2
encoder + 2 new conv2d + 110 previous). Honesty scan: 0 todo /
0 unimplemented / 4 known carry-over stubs.

## What Was Done This Session

### FR-17.3 — conv2d CPU reference (real impl, honesty-auditor verified)

The Path C audit had exactly 2 missing items: P17.3 (convolutions, L
effort) and P17.19 (Llama-1B reference, XL gate). FR-17.3 was the
cleanest self-contained shippable.

**Pre-audit state**: Path C status was largely "witnessed by stamp"
already — P17.1 (f16/bf16) has a real conversion-round-trip witness
(`dtype_half_round_trip.aether`); P17.13 (attention) has
`cuda_attention.aether` exercising real Q/K/V + softmax + matmul;
P17.14 (GGUF) has `gguf_header.aether` parsing the first 24 bytes
(quant dequant deferred); P17.18 (layer modules) is integer-only
(`layer_modules.aether`); P17.19 (Llama-1B) is the unwitnessed XL
gate. The decision was to fill the P17.3 slot honestly rather than
chase the Llama-1B XL.

**`runtime/src/lib.rs` additions (43-line impl + 2 unit tests):**
- `pub unsafe extern "C" fn aether_op_conv2d_f32(input, kernel,
  output, n, c_in, h, w, c_out, kh, kw) -> c_int` — NCHW direct
  convolution via 7 nested scalar loops. Stride=1, padding=0, no
  dilation, no groups. Returns 0/1/2/3 on success / null /
  invalid-shape / kh-or-kw-too-big.
- Unit test `conv2d_f32_4x4_with_3x3_all_ones` — 1×1×4×4 input
  [1..16] convolved with 1×1×3×3 all-1s kernel produces exactly
  [54, 63, 90, 99] (hand-computed window sums).
- Unit test `conv2d_f32_two_in_channels_sum` — 2 input channels
  (ch0 all-1s, ch1 all-2s) with all-1s kernel sums per-channel
  partial dots: every output cell = 9 + 18 = 27.

**Witness — `tests/runtime/conv2d_smoke.aether`** (66 lines, exit=42):
- Declares `extern fn aether_op_conv2d_f32(...)` matching runtime.
- Allocates 16-elem input + 9-elem kernel + 4-elem output (all f32
  via `aether_alloc_f32`).
- Fills input with `1..16` using `aether_store_f32(input, i, f32(i+1))`.
- Calls conv2d with `(1, 1, 4, 4, 1, 3, 3)` shape args.
- Reads 4 outputs via `aether_load_f32`, compares each to
  hand-computed 54/63/90/99, returns distinct error code per cell
  on mismatch.
- Builds through the full `aetherc → aether-asm → aether-bin`
  chain (3505-byte .obj). Exit=42 on first run.

honesty-auditor verified all 5 claims with file:line evidence and
reproduced command output. **Zero false claims.**

### Bench

Bench-runner append rule fires on `runtime/src/lib.rs` touched. Skip
note appended: this is a purely additive change (new fn + new unit
test mod); no matmul / softmax / layer_norm code path is altered.
The standing 2026-05-03 matmul row remains the reference. The right
place to log conv2d perf is the `bench/conv2d/` section's planned
row, which gates on a `runtime/src/cuda.rs::aether_op_conv2d_f32`
landing — not on this CPU reference impl.

## Current State

**Working:**
- 145/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 127 passing.
- Audit: `errors: 0` clean.
- Phase 17 is 19/20 — only the Llama-1B XL gate (P17.19) is
  unwitnessed.
- Path C foundation: f16/bf16 conversions, GGUF header parse,
  SafeTensors round-trip, attention forward, layer modules,
  optimizers — all witnessed and the runtime ops are real impls
  (subject to per-witness scope caveats listed in NEXT-UP).
- conv2d direct-loop CPU reference works; ready for the
  im2col+sgemm and cuDNN follow-ons.

**Honest scaffold-vs-shipped notes:**
- FR-17.3 ships ONLY the CPU direct-loop scope. The wider FR (im2col
  + sgemm, cuDNN behind feature gate, dilation, padding, depthwise,
  groups, transposed conv, conv1d/conv3d) is FR-17.3-extra.
- The GPU side (`runtime/src/cuda.rs::aether_op_conv2d_f32`) is
  not implemented. The new fn is CPU-only.
- P17.14 (GGUF) has a header-parse witness; Q4_0/Q4_K/Q5_K/Q6_K/Q8_0
  dequant + AWQ + GPTQ + INT8 QAT are NOT shipped.
- P17.18 (layer modules) witness uses integer math (no f32 array
  primitives needed); the f32 transformer block lives in
  `cuda_train_transformer_block.aether` and reuses the runtime ops.
- P17.13 (attention) witness uses naive softmax-attention. Memory-
  efficient FlashAttention v2 / PagedAttention is NOT shipped.

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stubs
(unchanged across the session): `mir/fuse.rs:53`, `mir/spec.rs:161`,
`runtime_pe/src/lib.rs:59`, `runtime_pe/src/lib.rs:443`.

## What's Next

`NEXT-UP.md` is the queue. Path C still has two un-witnessed long-
tail FRs (Llama-1B XL gate; FR-17.3-extra deepening of conv2d):

1. **Path C — FR-17.14 quant dequant (L)**. Real Q4_0/Q4_K dequant
   kernels in the runtime. Unblocks Llama-1B GGUF inference. Fewer
   pieces than Llama-1B itself.
2. **Path C — FR-17.13-extra FlashAttention v2 (L)**. Memory-
   efficient causal attention. The current witness exercises naive
   attention only.
3. **Path C — FR-17.19 Llama-1B reference (XL gate, v4 SHIP)**.
   `examples/llama_1b.aether` loads SafeTensors → matches HF
   reference within 1e-3 → trains. Depends on FR-17.18 layer modules
   in real f32 (not the integer witness shipped today).
4. **Path D — FR-19.1 TLS 1.3 (XL, long pole)**.
5. **Path E — FR-20.4 self-hosted asm emitter (XL)**.
6. **Path F — FR-22.1 LSP server (L)**.

## Notes for Next Session

- **Conv2d's direct-loop CPU impl is the reference, not the perf
  path.** The im2col + matmul-of-Toeplitz route reuses the existing
  `aether_op_matmul_f32` for the sgemm step — that's where any future
  perf claim lives. The runtime test mod has 2 hand-computed-value
  tests that any optimiser MUST agree with.
- **`aether_alloc_f32(n)` returns 0 on n<=0 and on allocation failure.**
  Witnesses should check for 0 before storing into the buffer.
- **`f32(int)` cast inside Aether** works via the asm backend's
  builtin numeric cast (recognised in `Expr::Call` for `f32`/`f64`/`i64`
  with 1 arg). Use it instead of declaring more conversion externs.
- **MS x64 ABI for 10-arg fns**: aether_op_conv2d_f32 takes 10 args
  (3 pointers + 7 ints). Args 5+ go on the stack at `[rsp + 32 + (i-4)*8]`.
  The compiler's call codegen handles this automatically; tested by
  the conv2d_smoke witness arriving at exit=42.
- **The 2 unwitnessed Phase 17 items now**: P17.3-extra (this
  session's deferred wider FR scope) and P17.19 (Llama-1B XL gate).
  Per the v4-SHIP definition in `memory/v4_ship_milestone.md`,
  Llama-1B is the gate; everything else is long-tail polish.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build assembler: `cargo build --bin aether-asm`
- Build runtime archive: `cargo build -p aether_rt`
- v4 FR queue: `NEXT-UP.md` (organised by critical path A-F)
- Compile a witness: `cargo run --bin aetherc -- tests/runtime/foo.aether --emit=aether-bin -o scratch/foo.exe`
- Conv2d witness: `cargo run --bin aetherc -- tests/runtime/conv2d_smoke.aether --emit=aether-bin -o scratch/conv2d.exe`
- Witness can pin opt-level: `// build-flags: --O1` at top of `.aether`
- Flags: `--O0/--O1/--O2/--lto/--target=<triple>/--no-std/--reproducible/--incremental`

## Commits this session

```
(pending) Path C FR-17.3: conv2d CPU direct-loop reference + witness
```
