# Aether — Session Handoff

## Last Updated
2026-05-02

## Project Status
🟢 **Audit clean. 17/17 runtime end-to-end tests pass. f32 lands in the asm backend; Aether code can now express SSE2 arithmetic, ucomiss compares, and call into libaether_rt via FFI. Loss curve still drops 5.564 → 1.679 in the AetherLM-Nano CPU smoke.**

The canonical numbers are whatever `scripts\audit.ps1` prints — that's the source of truth, not this file. Re-run it before claiming anything.

## What Was Done This Session

### Core deliverable: f32 in the asm backend (item #22 on the critical path)

- `aether_asm/src/encode.rs`
  - New `XmmReg` enum (Xmm0..Xmm7).
  - 4 new `CondCode` variants for unsigned/ucomiss compares (`A`, `B`, `Ae`, `Be`).
  - 11 new SSE2 instructions: `MovssRbpDispToXmm`, `MovssXmmToRbpDisp`, `MovssRipSymToXmm`, `MovssXmmXmm`, `AddssXmmXmm`, `SubssXmmXmm`, `MulssXmmXmm`, `DivssXmmXmm`, `UcomissXmmXmm`, `MovssRspToXmm`, `MovssXmmToRsp`.
  - 2 new encoder unit tests (`sse_arithmetic_encodings`, `movss_rip_has_relocation`) — byte-exact vs. Intel SDM.
- `aether_asm/src/parse.rs`
  - `.byte` directive (for f32 constant tables).
  - `parse_xmm`, `parse_rip_mem` helpers.
  - `(%rsp)` mem-operand recognition for spill slots.
  - `seta`/`setb`/`setae`/`setbe` mnemonics for unsigned setcc.
  - Size accounting for all new instructions.
- `compiler/src/codegen/asm/mod.rs`
  - New public `TyKind { Int, F32 }` enum.
  - `Locals` gained `types: HashMap<String, TyKind>` and `float_consts: Vec<f32>`.
  - `intern_f32` for per-fn-unique `.LF_<fnname>_<n>` labels.
  - `emit_expr_value` now returns `TyKind` so callers know which register to read.
  - `Stmt::Let { ty: Some(Ty::Named("f32")) }` puts the value in xmm0 and stores via `movss`.
  - `Expr::FloatLit` interns the bit pattern in `.rdata` and emits `movss .LF_<...>(%rip), %xmm0`.
  - `Expr::Ident` of an f32 local loads from its slot via `movss`.
  - `Expr::Bin` with f32 operands: spill lhs to `(%rsp)`, eval rhs, reload to `xmm1`, run `addss`/`subss`/`mulss`/`divss`.
  - f32 comparisons emit `ucomiss` + unsigned setcc (`seta`/`setb`/`setae`/`setbe`/`sete`/`setne`) and return `TyKind::Int`.
  - `Bin::Assign`, `If`, `Block`, all loops, `Unary`, `Call` updated to thread `TyKind` and reject mismatched-type Bin operands.

### Test coverage

- `tests/runtime/f32_compare.aether` (exit=7) — `1.5 + 2.5 == 4.0` → tail-expr branch.
- `tests/runtime/f32_arith.aether` (exit=42) — `((10.0 * 4.5) - 3.0) / 1.0 > 41.5`, mixes Mul + Sub + Div + ucomiss.

### Bookkeeping

- Renamed the project spec file `handoff.md` → `SPEC.md` so the conventional `HANDOFF.md` (this file) is free for session state. Updated references in `CLAUDE.md`, `README.md`.

## Current State

### Working (verified by `scripts\audit.ps1` this session)

- Full Aether-only compile chain: `aetherc --emit=aether-bin` → x86-64 asm (ours) → COFF .obj (`aether_asm`, ours) → linked .exe.
- Aether language surface compiled by the asm backend: `let` (with optional type annotation), `let mut`, `Bin::Assign`, ints + arithmetic (Add/Sub/Mul/Div/Mod with idivq+cqo), comparisons (Eq/Ne/Lt/Gt/Le/Ge → bool), unary `-x` and `!x`, `if/else`, `for i in lo..hi`, `while cond`, `break`, `continue`, `&local`, multi-arg FFI calls, `println(STR)`, **f32 literals + arithmetic + ucomiss compares**.
- libaether_rt linkage from `--emit=aether-bin`: extern fns named `aether_*` resolve. Verified by `ffi_self_check`, `ffi_tape_push`, `ffi_buffer`, `for_ffi_tape`, `nested_loops`.
- AetherLM-Nano trains on CPU through libaether_rt: 5.564 → 1.679 in 40 steps.
- 40/40 unit tests pass; 9/9 golden artifacts match; 8/8 conformance cases pass; 17/17 runtime cases pass.

### Stubbed / explicitly Phase-N

- `aether_op_*` runtime bodies are real f32 CPU implementations; cuBLAS/cuDNN swap is Phase 1.
- `aether_op_all_reduce_sum_f32` and the higher-level `aether_dist_all_reduce` still no-op (`/* Phase 2 — NCCL */`).
- The `trainer/` Rust crate is bootstrap; it's what aetherc Phase 1 emits from `examples/aether_lm.aether`.
- `aether_asm` only encodes the instruction subset aetherc emits today (~30 mnemonics). Missing: most general addressing modes, all xmm8–xmm15, all f64 / int-vector / AVX. Not blocking; widen on demand.
- The system linker is the last external tool in `--emit=aether-bin`. Phase-5 self-hosted PE32+ writer drops it.

### Not yet wired

- f32 ↔ int casts (`cvtsi2ss`, `cvtss2si`).
- f32 arg passing to FFI (must use xmm0–xmm3 in MS x64; today the asm backend rejects float args).
- f32 fn return values.
- f64.
- Struct field access (`x.field`).
- Arrays (`[T; N]`).
- Nested calls in args (the asm backend rejects).

## Blocking Issues

None.

## What's Next (priority order)

The CLAUDE.md "Critical Path" section lists 27 numbered steps; items 1–22 are done. The live items:

1. **#22 expansion — f32 casts + FFI float args + f32 returns + f64.** `cvtsi2ss xmm0, eax` (`F3 0F 2A C0`), `cvtss2si eax, xmm0` (`F3 0F 2D C0`); rework `Expr::Call` arg-passing to interleave xmm regs by position; teach `emit_fn` to keep the tail value in xmm0 when the declared return type is `f32`. Add `f64` by swapping the SSE prefix from `F3` to `F2` (sd suffix). Each is small once the type-tracking plumbing is in place.

2. **#23 — struct field access** (`x.field`). Layout: each struct is a contiguous block of f32/i64 slots in the same arena as the struct local. `count_locals` needs to recurse into struct types and sum their slot counts. `Expr::Field { recv, name }` looks up the field's offset and emits a slot read at the right disp.

3. **Arrays** (`[T; N]`) and `lhs[i]`. Stack-allocated, fixed size known at lex. Arms the runtime test `examples/aether_lm.aether` to pass real f32 buffers to `aether_op_*`.

4. **#24 — self-hosted linker.** PE32+ writer in `aether_asm/`: DOS stub, PE/COFF headers, `.idata` import table for msvcrt's `puts` + libaether_rt's `aether_*` symbols, base-relocations, IAT. After this lands the toolchain has zero external deps for static binaries.

5. **#25 — real cuBLAS/cuDNN backend in `runtime/`.** Replace each `aether_op_*` body with a CUDA implementation. The Rust crate stays a thin shim. Use `cudarc` or hand-rolled FFI; differential-test against `model/aether_lm.py` before swapping.

6. **#26 — first real training run on 3070 Ti**, once #22 expansion + #23 + #25 land. Compile `examples/aether_lm.aether --emit=aether-bin`, link with the cuBLAS-backed runtime, run on the 3070 Ti, assert loss curve matches the PyTorch oracle within tolerance.

7. **#27 — self-host the compiler.** Rewrite `compiler/`, `aether_asm/`, `runtime/` in Aether. Drops Rust from the entire stack.

## Notes for Next Session

- **Run `scripts\audit.ps1` first.** Don't take any number in this file at face value; the audit prints the live numbers (SLOC, honesty scan, tests, goldens, conformance, runtime end-to-end). 0 errors == clean.
- **The audit is the truth.** Every claim must be backed by an audit dimension. The file `tools/audit/src/scan.rs` defines what counts as a "stub return"; the heuristic is conservative (only matches `fn ... { 0 }` one-liners with optional `/* */` comments) but lets f32-bit literal lines escape. That works today, but if you add new patterns for stubs, update the heuristic to keep finding them.
- **Rebuild golden expected files only when codegen changes are intentional.** Run `aether-audit --update-golden` to re-prime — the next audit will diff against the new files.
- **`aether_asm` instruction-size tables in `parse.rs::synthetic_text_size` MUST stay in sync with the encoder.** A mismatch silently corrupts forward-jump rel32 displacements. Every new `Instr` variant needs an entry there.
- **Comments are stripped at lex time, irreversibly.** No `--keep-comments` flag exists. The user has reaffirmed this multiple times — do not add a debug escape hatch.
- **Bootstrap reality:** Rust is the implementation language for the compiler, the assembler, the runtime, and the trainer. Aether self-hosts in Phase 5 (#27). Until then, "no language deps" means "no Python, no candle, no torch, no JAX, no XLA" — Rust-as-bootstrap stays.
- **The user's candle fork lives at `J:\candle-src`.** Production-grade work (7B QLoRA on 8GB at 5.5 tok/s, TDPI loop, flash-attn-v3, etc.). Aether **does not depend on it**; Aether's runtime calls cuBLAS/cuDNN directly via C ABI. The candle fork is informational — it tells us which ops the runtime needs to expose.
- **Smoke runs the AetherLM-Nano training** end-to-end through libaether_rt: `scripts\smoke.ps1`. If you change the runtime ops, watch the loss curve there.
- **Don't run subagents to "explore" before reading this file + CLAUDE.md.** The prior-session knowledge is dense. Read both, then act.
