# Aether — Session Handoff

## Last Updated
2026-05-18 (Path A continuation — FR-15.1 + FR-15.2 shipped same session)

## Project Status
🟢 **Audit: 143/196 (72%) roadmap items witnessed.** +2 from session
baseline of 141/196. 0 errors, all workspace tests green (1 flaky TCP
loopback that passes in isolation; not related to compiler changes).
Honesty scan unchanged (4 known-OK stubs). Path A advanced two L-effort
items in one session — A1 (SSA-driven emit) and A2 (regalloc-in-emit) —
both honestly shipped, both honesty-auditor verified.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    7/10 witnessed (70%)  ← +2 (FR-15.1 + FR-15.2)
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   18/20 witnessed (90%)  — unchanged
Phase 18:    2/11 witnessed (18%)  — unchanged
Phase 19:    0/16 witnessed (0%)   — unchanged
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    143/196 (72%)
```

Workspace tests: 116 pass (3 ssa_drive + 3 regalloc_plan + 110 others).
Honesty scan: 0 todo / 0 unimplemented / 4 known carry-over stubs.

## What Was Done This Session

### FR-15.2 — regalloc-in-emit (real impl, honesty-auditor verified)

The existing `mir::regalloc::Allocator` ran at `--O1` but only reported
`(reg_count, spill_count)` to stderr — its plan never reached the asm
backend. This was the second of FR-15.{1,2,3}'s three "real engineering
each" items, and the unblocker for A3 (AVX2/AVX-512 emit, FR-15.3,
which needs SIMD-resident locals to place its vectors into).

**`compiler/src/mir/regalloc_plan.rs` (new, 414 lines, 3 unit tests):**
- `pub fn plan_program(prog: &Program) -> PlanMap` returns the per-fn
  `HashMap<String, HashMap<String, u8>>` — local name → callee-saved
  physical reg id in {12, 13, 14, 15}.
- Walks each fn body collecting decl + last-use indices via the same
  live-range shape `mir::regalloc_drive` already uses, then runs
  `Allocator::new(vec![12, 13, 14, 15]).allocate(...)`.
- Exclusions: address-taken locals (`&x` / `&mut x`), composite-typed
  lets (struct / tuple / array / Tensor / non-primitive Named),
  shadowed re-declarations, uninit (`value: None`) lets. Extern fns
  skipped entirely.

**`compiler/src/codegen/asm/mod.rs` changes:**
- New public entrypoint `pub fn emit_with_plan(p, plan)` consumes the
  plan; the old `pub fn emit(p)` is a thin wrapper passing an empty plan
  (preserves `--O0` byte-compat).
- `Locals` struct grew `reg_map: HashMap<String, u8>` and
  `saved_regs: Vec<u8>`. `frame_bytes()` adds 8 when the saved-reg
  push count is odd to keep rsp 16-aligned after the subq.
- Prologue: `pushq %rN` per saved reg AFTER `pushq %rbp` and BEFORE
  the __chkstk/subq probe.
- Epilogue: `popq %rN` per saved reg in reverse order AFTER the addq
  and BEFORE `popq %rbp`. Three early-return paths (`Stmt::Return(Some)`,
  `Stmt::Return(None)`, `Expr::Try` Err-branch) all run the same pop
  sequence — without it, callee-saved regs leak out clobbered.
- Ident read site: when `kind == Int` and `reg_map.get(name)` hits,
  emit `movq %rN, %rax` instead of `movq disp(%rbp), %rax`. Returns
  early — never touches the stack for promoted reads.
- Stmt::Let int store path AND Bin::Assign int store path: after the
  canonical stack store `movq %rax, slot`, emit `movq slot, %rN`
  (load-from-slot write-through). **Critical detail**: copying
  `%rax → %rN` would race the peephole at lines 233-243, which
  collapses `movq $imm, %rax; movq %rax, slot` into `movq $imm, slot`
  and leaves %rax stale. Loading from the slot is correct regardless
  of whether peephole fired.

**`compiler/src/main.rs`:** plan computed at `--O1+` (empty otherwise),
forwarded to all 4 `codegen::asm::emit_with_plan(&prog, &plan)` call
sites (C / Asm / AetherBin / PeBin / AsmBin emit paths). Stderr line
reads `[aetherc] P15.2 regalloc plan: N fn(s), K local(s) promoted to
r12..r15`.

**Witness — `tests/runtime/regalloc_in_emit.aether`:** 4 hot Int
locals (a/b/c/d), straight-line body with 16 reads in pairwise products
and a final sum. Exits 42. Empirical evidence:

| metric                                | --O0 | --O1 |
|---------------------------------------|------|------|
| `movq disp(%rbp), %rax` count         |   16 |    1 |
| `movq %r1[2-5], %rax` count           |    0 |   15 |
| asm lines                             |  103 |  115 |
| stderr "regalloc plan" line           |  n/a | `1 fn(s), 4 local(s) promoted` |

15 of 16 reads converted from stack to reg. The asm count grew 12 lines
at --O1 (4 pushes + 4 pops + 4 initial reg loads); for this toy body
the prologue/epilogue overhead outweighs the per-read savings. On
`cuda_train_transformer_block.aether` the shrink is 21670 → 21632 bytes
(0.18%), far from the FR's aspirational 30% target — the planner is
deliberately conservative (Tensor handles and method calls excluded)
and that file's hot path is exactly the excluded shape. The shipped
capability is the FOUNDATIONAL MACHINERY (plan computed, threaded,
consumed; hot Int locals demonstrably in r12..r15; correctness
preserved across calls), not the perf headline.

honesty-auditor verified all 8 claims against file:line and reproduced
command output. **Zero false claims.**

### FR-15.1 — SSA-driven opt pipeline (real impl, earlier this session)

[See prior handoff in `git log` at commit 32784f7 for the full FR-15.1
write-up; one safety fix applied since: the SSA driver now ONLY fires
when the linearised prefix is the entire body — when a non-linearisable
suffix follows (a for-loop, an if, a call), DCE could legitimately drop
a let that the suffix still references. This came up while writing
the FR-15.2 witness, which has `let a=1; let b=2; ...; for i in 0..14 {
acc = acc + a + b + ...; }` — at --O1 the SSA pass would have nuked
a/b/c/d as "unused" because their only uses are inside the for-loop
that lives in the suffix. Fix: `if prefix_len < body.stmts.len() { return; }`
gates the optimisation. The ssa_emit_drives_asm witness still works
because its body is pure lets + tail (no suffix).]

### Bench

Bench-runner invoked per the BENCH_LEDGER append rule (commit touches
`compiler/src/codegen/asm/`). GPU was at 39% util with 7.3 GiB occupied
by `Settlement Survival.exe` + `ollama.exe`; subagent correctly
declined to record numbers under contention, matching the 2026-05-09
"skipped — variance" precedent. Structural argument: FR-15.2 affects
caller fns in `.aether` source only; `aether_op_matmul_f32` in
`runtime/src/cuda.rs` is unchanged; cuBLAS sgemm time dominates the
bench by orders of magnitude. Expected delta vs 2026-05-03 reference:
indistinguishable from noise. Skip note appended to ledger; the
2026-05-03 row remains the standing reference.

## Current State

**Working:**
- 143/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 116 passing.
- Audit: `errors: 0` clean.
- `--O1` now does SSA-driven AST rewriting AND callee-saved reg
  promotion for hot Int locals.
- FR-15.{1,2} foundation is in place for FR-15.3 (AVX2/AVX-512 emit) —
  the regalloc plan can be extended to assign SIMD locals into
  ymm/zmm regs once the encoder gains the AVX2 opcodes.

**Honest scaffold-vs-shipped notes:**
- FR-15.2 ships the wiring but not the perf claim. The FR spec's "30%
  obj shrink on `cuda_train_transformer_block`" target is 0.18% in
  practice. The asm-backend headroom is there for hot Int code; the
  Tensor-handle-heavy programs that dominate Aether's test suite
  benefit very little.
- SSA driver linearisation requires the body to be a let-prefix +
  optional tail. A real SSA-aware suffix analysis is the next iteration.
- v3's vectorize_drive still only reports counts; vectorisation doesn't
  drive asm emission yet (FR-15.3 carries that L work).
- Matmul hot loop still doesn't consult P15.6's autotune table.
- Capturing closures used in pass-as-value position still mis-codegen.

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stubs:
- `compiler/src/mir/fuse.rs:53` — `fn_marker` unused-arg helper.
- `compiler/src/mir/spec.rs:161` — `_scaffold_param_unused` helper.
- `runtime_pe/src/lib.rs:59` — `aether_autodiff_accumulate` (no_std stub).
- `runtime_pe/src/lib.rs:443` — `rust_eh_personality` (panic=abort glue).

Flaky `tcp_send_recv_loopback` in `aether_rt::tests` fails ~10% of the
time under workspace-wide parallel run; passes 100% in isolation. Not
related to compiler changes; pre-existing TCP loopback timing.

## What's Next

`NEXT-UP.md` is the queue. Path A is one L-effort step away from done.

1. **Path A — FR-15.3 AVX2/AVX-512 emit (L)**. The final A-path L item.
   Needs new encoder ops in `aether_asm/`:
   `vmovups`/`vaddps`/`vmulps`/`vfmadd231ps`/`vbroadcastss` (256-bit
   first; 512-bit later). Behind `--target-cpu={skylake-avx512,znver4}`.
   The regalloc plan now has somewhere to put SIMD-resident values —
   extend the pool to ymm/zmm regs when target-cpu requests it.
   Witness target: 1024-element f32 dot product ≥4× faster at --O1
   vs --O0.
2. **Path A — FR-15.10 hand-asm reference (M, gate)**. Hand-written
   reference asm for matmul/softmax/LN/SDPA/CE in `bench/handasm/`;
   measure ≤1% gap. This closes Path A.
3. **Path C — Tensor stack toward v4 SHIP**: FR-17.1-extra f16/bf16
   (M) → FR-17.13 RoPE + FlashAttention (L) → FR-17.14-extra GGUF +
   quant (L) → FR-17.19 Llama-1B reference (XL gate).
4. **Path D — Serving**: FR-19.1 TLS 1.3 (XL, long pole).
5. **Path E — Self-host**: FR-20.4 self-hosted asm emitter (XL).
6. **Path F — Tooling**: FR-22.1 LSP (L), FR-22.2 DAP (M).

## Notes for Next Session

- **Honest scope is the rule.** FR-15.2's "30% shrink" target wasn't hit;
  the handoff and the honesty-auditor's verdict say so plainly. The
  shipped capability is the foundational machinery, not the perf headline.
  Future sessions should NOT claim "matmul 30% faster" based on this
  commit.
- **--O0 byte-compat is preserved.** `emit(p)` wraps `emit_with_plan(p,
  &empty)`; the empty plan path is byte-identical to the pre-FR-15.2
  baseline. Don't change that without a witness proving it.
- **Write-through pattern**: hot-reg Int locals use `movq slot, %rN`
  (load-from-slot) NOT `movq %rax, %rN` (rax-to-reg). The peephole at
  lines 233-243 makes the latter unsafe — see the comment at the
  Stmt::Let write-through site for the full reason.
- **Callee-saved reg push count parity**: when the planner assigns an
  odd number of regs, `frame_bytes()` adds 8 to keep rsp 16-aligned
  after the subq. Forget this and your fn will crash on the first
  `movaps` / xmm spill.
- **`Stmt::Return` + `Expr::Try` early-return paths** must run the same
  pop sequence as the natural epilogue. The codebase has 3 ret-emit
  sites; touching one without the other 2 silently corrupts callee
  state. Search for `popq %rbp` in `codegen/asm/mod.rs` to find them all.
- **Closures-with-captures uses direct-call rewrite, not env structs.**
  See `memory/closures_captures_pattern.md`. Pass-as-value of capturing
  closures is silently mis-codegen'd today.
- **`aether-bin` linking against libaether_rt.a needs `cargo build -p
  aether_rt`** after adding new runtime symbols.
- **No Python for tooling.** Rust binaries in `tools/` or pure Aether.
- **Path A's last L item (FR-15.3 AVX2) is real engineering.** Don't
  bag it alongside FR-15.10 (the gate witness). Pick one.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime archive: `cargo build -p aether_rt`
- v4 FR queue: `NEXT-UP.md` (organised by critical path A-F)
- Compile a witness: `cargo run --bin aetherc -- tests/runtime/foo.aether --emit=aether-bin -o scratch/foo.exe`
- Compile SSA+regalloc on: add `--O1` (stderr reports both pipelines)
- Witness can pin opt-level: `// build-flags: --O1` at top of `.aether`
- Flags: `--O0/--O1/--O2/--lto/--target=<triple>/--no-std/--reproducible/--incremental`

## Commits this session

```
32784f7 Path A FR-15.1: SSA-driven opt pipeline rewrites AST at --O1
(pending) Path A FR-15.2: regalloc-in-emit (r12..r15 callee-saved promotion)
```
