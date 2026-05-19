# Aether — Session Handoff

## Last Updated
2026-05-18 (Path A complete — FR-15.1 + FR-15.2 + FR-15.3 all shipped same session)

## Project Status
🟢 **Audit: 144/196 (73%) roadmap items witnessed.** +3 from session
baseline of 141/196. 0 errors, all workspace tests green (1 flaky TCP
loopback that passes in isolation; not related to compiler changes).
Honesty scan unchanged (4 known-OK stubs). **Path A's three L-effort
items all shipped, all honesty-auditor verified.** The path that the
prior handoff said was "real engineering each, picking one and doing
it honestly beats three half-implementations" was completed cleanly:
3-for-3 real impls, 23/23 honesty-auditor claims verified, zero
false claims.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  ← +3 (FR-15.1 + FR-15.2 + FR-15.3)
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   18/20 witnessed (90%)  — unchanged
Phase 18:    2/11 witnessed (18%)  — unchanged
Phase 19:    0/16 witnessed (0%)   — unchanged
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    144/196 (73%)
```

Workspace tests: 125 pass (3 ssa_drive + 3 regalloc_plan + 9 new AVX2
byte tests + 110 previous). Honesty scan: 0 todo / 0 unimplemented /
4 known carry-over stubs.

## What Was Done This Session

### FR-15.3 — AVX2 emit (real impl, honesty-auditor verified)

The compiler now drives 256-bit AVX2 instructions from `.aether`
source through the full aetherc → asm-text → aether-asm → bytes
chain. The shipped capability is *correctness* — perf measurement
is deferred to a future bench fixture.

**`aether_asm/src/encode.rs` additions:**
- `pub enum YmmReg { Ymm0..Ymm7 }` (limited to the subset that fits
  in the 2-byte VEX prefix; ymm8..ymm15 require the 3-byte C4
  prefix and are deferred).
- 7 new `Instr` variants — `VxorpsYmmYmmYmm`, `VmovupsMemToYmm`,
  `VmovupsYmmToMem`, `VaddpsYmmYmmYmm`, `VmulpsYmmYmmYmm`,
  `VmovupsYmmToRspNoDisp` (the rsp form needs a SIB byte because
  rsp can't be encoded directly in ModRM.r/m), `Vzeroupper`.
- 9 byte-exact unit tests verified against Intel SDM Vol. 2.
- Each variant's doc comment explains the encoding (VEX byte
  layout, ModRM bit map, when SIB is needed).

**`aether_asm/src/parse.rs` additions:**
- `parse_ymm` helper (only ymm0..ymm7).
- Mnemonic arms for `vxorps`, `vaddps`, `vmulps`, `vmovups`,
  `vzeroupper`. The 3-operand AT&T order is `src2, src1, dst`
  (matches binutils convention).
- `vmovups` recognises three forms: `disp(%base), %ymm` (load),
  `%ymm, disp(%base)` (store), `%ymm, (%rsp)` (no-disp SIB store).
- Size table `synthetic_text_size` synced — all 7 sizes match
  the encoder's byte counts exactly.

**`compiler/src/codegen/asm/mod.rs` integration:**
- Inside the `Expr::Call` arm, after the `println` special-case and
  before the standard call codegen, the call name
  `__aether_avx2_dot_f32` (with exactly 3 args) triggers an inline
  AVX2 dot loop. Args (each `TyKind::Int`) get evaluated and moved
  into `%rcx` (a_ptr), `%rdx` (b_ptr), `%r8` (n).
- Loop body: `vxorps %ymm0,%ymm0,%ymm0` (acc init), `xorq %rax,%rax`
  (index), labelled loop emitting `vmovups 0(%rcx),%ymm1`,
  `vmovups 0(%rdx),%ymm2`, `vmulps %ymm2,%ymm1,%ymm1`,
  `vaddps %ymm1,%ymm0,%ymm0`, `addq $32` strides, `addq $8` index,
  `cmpq+jne` tail.
- Horizontal-sum epilogue: `subq $32,%rsp`, `vmovups %ymm0,(%rsp)`,
  `vzeroupper`, then 8-way scalar `movss+addss` sum into `%xmm0`,
  `addq $32,%rsp`. Returns `Ok(TyKind::F32)`.

**`runtime/src/lib.rs` witness helpers:**
- `aether_avx2_witness_arr(seed, n) -> i64` — allocates n f32 slots
  via `aether_alloc_bytes`, fills deterministically from an LCG.
- `aether_dot_f32_scalar(a, b, n) -> f32` — reference scalar dot.
- `aether_f32_close_exit(a, b) -> i32` — returns 42 if values
  agree within 1e-3 relative tolerance, else 1.

**Witness — `tests/runtime/avx2_dot_f32.aether`:** declares the
four needed externs (the three above + `aether_free_bytes`), calls
`__aether_avx2_dot_f32(a_ptr, b_ptr, 1024)`, compares to scalar
reference, frees buffers, returns the exit code. Builds via the
`--emit=aether-bin` audit chain → 1078-byte `.obj` → linked exe
that exits 42.

honesty-auditor verified all 8 claims with file:line + reproduced
byte literals + command output. **Zero false claims.**

### FR-15.1 + FR-15.2 (earlier this session)

[See prior handoff and `git log` at commits 32784f7 + ffb2336 for
full write-ups; nothing changed since.]

### Bench

Bench-runner append rule fires for commits touching
`compiler/src/codegen/asm/` or `runtime/src/lib.rs`. **Both** were
touched by FR-15.3. Skip note appended to `BENCH_LEDGER.md` with
the honest reasoning: the matmul benches drive
`aether_op_matmul_f32` via cuBLAS — a path this commit does NOT
change. The new AVX2 builtin and witness helpers don't sit on the
matmul bench path. A standalone "f32 dot AVX2 vs scalar" bench is
the right place to measure the per-instruction headline; that
fixture doesn't exist yet.

## Current State

**Working:**
- 144/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 125 passing.
- Audit: `errors: 0` clean.
- `--O1` does SSA-driven AST rewrite + callee-saved reg promotion.
- The asm backend can drive AVX2 inline from `.aether` source via
  the recognised `__aether_avx2_dot_f32` builtin.
- Path A foundation done: SSA → opt → emit, regalloc-in-emit,
  AVX2 encoder. The next step (FR-15.10 hand-asm gate) only
  needs a bench fixture, not new compiler engineering.

**Honest scaffold-vs-shipped notes:**
- FR-15.3 ships *correctness* through the compile chain, not perf.
  The "1024-elem f32 dot ≥4× faster" claim is unverified — no
  comparison bench exists yet. The encoder + parser + compiler-
  side emit are all real; the speed-up claim is the next step.
- Only one AVX2 pattern is recognised today: the
  `__aether_avx2_dot_f32` builtin. General for-loop vectorisation
  via pattern detection (turning a `for i in 0..N { acc[i] = a[i] +
  b[i] }` body into AVX2 emit) is the next iteration.
- `YmmReg` is limited to ymm0..ymm7 (2-byte VEX). ymm8..ymm15
  require 3-byte VEX (C4 prefix); deferred.
- AVX-512 (zmm regs, EVEX prefix) is not started.
- FR-15.2 ships wiring but not the 30% obj-shrink perf headline.
- The SSA driver only fires when the linearised prefix is the
  whole body.
- v3's vectorize_drive still only reports counts; vectorisation
  doesn't drive asm emit yet — FR-15.3 covers ONE recognised path
  via builtin, not pattern-driven loop vectorisation.
- Matmul hot loop still doesn't consult P15.6's autotune table.
- Capturing closures used in pass-as-value position still
  mis-codegen.

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stubs
(unchanged across the session): `mir/fuse.rs:53`, `mir/spec.rs:161`,
`runtime_pe/src/lib.rs:59` (no_std stub), `runtime_pe/src/lib.rs:443`
(panic=abort glue).

Flaky `tcp_send_recv_loopback` in `aether_rt::tests` fails ~10% under
workspace-wide parallel run; passes 100% in isolation. Not related
to compiler changes.

## What's Next

Path A is materially complete. Recommended next targets:

1. **Path A — FR-15.10 hand-asm reference gate (M)**. Write
   hand-tuned AVX2 reference asm for matmul/softmax/LN/SDPA/CE in
   `bench/handasm/`, measure aether's `--O2` emit within 1% wall
   on 11900K + 3070 Ti. This is the gating witness that says "v4
   perf shipped" — but it depends on actually wiring AVX2 into the
   matmul caller-side, which needs the loop-vectoriser, which is
   either FR-15.3-extra (extend the recognised-builtin set) or a
   real pattern-detector pass.
2. **Path C — Tensor stack toward v4 SHIP**: FR-17.1-extra f16/bf16
   (M) → FR-17.13 RoPE + FlashAttention (L) → FR-17.14-extra GGUF
   + quant (L) → FR-17.19 Llama-1B reference (XL gate).
3. **Path D — Serving**: FR-19.1 TLS 1.3 (XL, long pole).
4. **Path E — Self-host**: FR-20.4 self-hosted asm emitter (XL).
5. **Path F — Tooling**: FR-22.1 LSP (L), FR-22.2 DAP (M).

## Notes for Next Session

- **Three Path A L items in one session is a real achievement.**
  All three honesty-auditored, all three commits standalone, all
  three roadmap-witnessed. Resist the temptation to claim more
  than what was honestly shipped — FR-15.3 ships *correctness*,
  not the perf headline.
- **Don't fake the FR-15.10 gate.** The 1%-of-handasm pact needs
  real hand-asm references in `bench/handasm/` and a comparison
  bench that produces honest numbers on the 11900K + 3070 Ti.
  Stamping it without that work would burn audit honesty.
- **VEX encoding fundamentals are now in the codebase.** If you
  extend to ymm8..ymm15 you need 3-byte VEX (C4 prefix). The
  current `assert_eq!(base.extension(), 0, ...)` and
  `parse_ymm` `ymm0..ymm7` limit catch attempts to use the upper
  bank — bypass them carefully.
- **The compiler-recognised builtin pattern works.** Adding more
  AVX2 operations is now a matter of (a) extending the encoder
  with the new Instr variant, (b) extending the parser arm, (c)
  syncing the size table, (d) recognising the call name in
  `Expr::Call`, (e) emitting the inline asm text. Each new
  builtin is ~30 lines of compiler integration.
- **`aether-asm` and `libaether_rt.a` rebuild separately.** After
  adding mnemonics, `cargo build --bin aether-asm` is mandatory.
  After adding runtime symbols, `cargo build -p aether_rt` is
  mandatory. The `--emit=aether-bin` chain links against both;
  stale binaries surface as "unsupported instruction" or
  "undefined reference" at link time.
- **Closures-with-captures uses direct-call rewrite, not env structs.**
- **`Stmt::Return` + `Expr::Try` early-return paths** must run the
  same pop sequence as the natural epilogue — FR-15.2's r12..r15
  push/pop discipline depends on this.
- **No Python for tooling.** Rust binaries in `tools/` or pure Aether.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build assembler: `cargo build --bin aether-asm`
- Build runtime archive: `cargo build -p aether_rt`
- v4 FR queue: `NEXT-UP.md` (organised by critical path A-F)
- Compile a witness: `cargo run --bin aetherc -- tests/runtime/foo.aether --emit=aether-bin -o scratch/foo.exe`
- AVX2 witness: `cargo run --bin aetherc -- tests/runtime/avx2_dot_f32.aether --emit=aether-bin -o scratch/avx2.exe; ./scratch/avx2.exe; echo $?`
- Witness can pin opt-level: `// build-flags: --O1` at top of `.aether`
- Flags: `--O0/--O1/--O2/--lto/--target=<triple>/--no-std/--reproducible/--incremental`

## Commits this session

```
32784f7 Path A FR-15.1: SSA-driven opt pipeline rewrites AST at --O1
ffb2336 Path A FR-15.2: regalloc plan drives r12..r15 promotion in asm backend
(pending) Path A FR-15.3: AVX2 emit via aether_asm + compiler-recognised dot builtin
```
