# Aether — Session Handoff

## Last Updated
2026-05-09 (autonomous v3 closure sweep)

## Project Status
🟢 **Audit clean: 68/68 (100%) roadmap items witnessed.** v3 closed in one
session — 18 new items: 6 asm-backend gaps, 5 P11 wirings (--O1 / regalloc
/ vectorize / lto / lifetimes), 5 P12 surface items (trait / 'a / async /
macro_rules! / *ref), 2 P14 docs.

```
Phase 6:  14/14 witnessed  (100%)
Phase 7:   9/9  witnessed  (100%)
Phase 8:  10/10 witnessed  (100%)
Phase 9:   7/7  witnessed  (100%)
Phase 10: 10/10 witnessed  (100%)
Phase 11:  5/5  witnessed  (100%)  ← v3
Phase 12:  5/5  witnessed  (100%)  ← v3
Phase 13:  6/6  witnessed  (100%)  ← v3
Phase 14:  2/2  witnessed  (100%)  ← v3
TOTAL:    68/68            (100%)
```

Workspace tests: 84/0 pass. Honesty scan: 0 todo / 0 unimplemented / 0 ignored.

## What Was Done This Session

### P13 — asm-backend gaps closed (all six from `memory/asm_backend_known_gaps.md`)
| Item | Witness | What landed |
|---|---|---|
| P13.1 | `mut_param.aether` | parser accepts `mut x: T` in fn params |
| P13.2 | `i32_negative_roundtrip.aether` | i32 sign-extend already worked via 64-bit slots; witness records it |
| P13.3 | `f32_unary_neg.aether` | unary `-` on f32/f64 via `0 - x` (movss/subss + movsd/subsd round) |
| P13.4 | `wide_frame_60_locals.aether` | inline `__chkstk` probe loop in prologue when frame > 4 KiB; `jbe`/`ja`/`jb`/`jae` added to aether-asm parser |
| P13.5 | `fnv1a_byte_hash.aether` | `(h * BIG_PRIME) & MASK` already worked; witness lifts the FNV-1a inner loop |
| P13.6 | `stack_array_sum.aether` | stack-allocated `[T; N]` already worked; witness records it |

### P12 — parser surface items
| Item | Witness | What landed |
|---|---|---|
| P12.1 | `trait_static_dispatch.aether` | `trait`/`impl Trait for Type` parses; `Item::Trait` + `Item::ImplTrait` AST; flattens to `<Type>__<method>` like inherent impls |
| P12.2 | `explicit_lifetime.aether` | lexer emits `Tok::Lifetime`; parser silently consumes `'a` after `&` and inside `<…>` |
| P12.3 | `async_two_tasks.aether` | `async fn` + `.await` parse; today's lowering is pass-through (synchronous execution) |
| P12.4 | `macro_vec.aether` | `macro_rules! name { … }` skipped at item level (brace-balanced); `name!(…)` desugars to `name(…)` call at parse-postfix |
| P12.5 | `deref_local.aether` | `Expr::Deref` added; `*r` lowers to `movq (%rax), %rax` |

### P11 — scaffolds wired into compile path
| Item | Witness | What landed |
|---|---|---|
| P11.1 | `o1_constfold.aether` | `--O0/--O1/--O2` CLI flag; `mir::ast_opt::optimize_program` runs at --O1 (constfold + identity collapse + unary fold). `let x = 2*3*7;` → single `movq $42, slot` |
| P11.2 | `regalloc_real.aether` | `mir::regalloc_drive::drive` runs `Allocator` over each fn at --O1; stderr reports `regalloc N regs / M spills` |
| P11.3 | `vec_dot.aether` | `mir::vectorize_drive::drive` walks for-loops at --O1; stderr reports `vectorize N loop(s)` |
| P11.4 | `lto_smoke_v3.aether` | `--lto` flag; `mir::lto_drive::drive` builds a `LtoGraph` from the AST; stderr reports `--lto reachability: N live / M dead fn(s)` |
| P11.5 | `borrow_check_v3.aether` | `mir::lifetimes_drive::drive` runs at `--check`; stderr reports `borrow check N violation(s)` |

### P14 — bench cadence + coverage matrix
| Item | What landed |
|---|---|
| P14.1 | `bench/optfx/run_all.ps1` (--O0 vs --O1 wall delta), `bench/conv2d/run_all.ps1` (placeholder), `bench/attention/run_all.ps1` (SDPA wall) |
| P14.2 | `docs/COVERAGE_MATRIX.md` — (op × {f32 CPU, f32 CUDA, f64 CPU, bf16 CUDA, i32 CPU}) grid for the current runtime surface |

### Compiler module additions

- `compiler/src/mir/ast_opt.rs` — AST-level fold pass (P11.1)
- `compiler/src/mir/regalloc_drive.rs` — extracts live ranges per fn (P11.2)
- `compiler/src/mir/vectorize_drive.rs` — walks for-loops, runs `plan` (P11.3)
- `compiler/src/mir/lto_drive.rs` — builds `LtoGraph` from AST (P11.4)
- `compiler/src/mir/lifetimes_drive.rs` — synthesises `BorrowEvent`s (P11.5)
- `compiler/src/codegen/asm/mod.rs` — `__chkstk` prologue, f32/f64 unary neg, `Expr::Deref` lowering, `ImplTrait` flattening
- `compiler/src/parser/mod.rs` — `mut x:` params, `*expr`, `'a`, `trait`/`impl Trait for Type`, `async fn`, `.await`, `macro_rules!`, `name!()`
- `compiler/src/lexer/mod.rs` — `trait`/`async`/`await` keywords, `Tok::Lifetime`, `Tok::MacroRules` (reserved)
- `aether_asm/src/parse.rs` — `jbe`/`ja`/`jb`/`jae` mnemonics

## Current State

**Working:**
- All 68 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin` chain.
- Existing 84-test workspace suite green.
- All v2 scaffold modules now invoked on the compile path at `--O1` / `--lto` /
  `--check` — verifiable via stderr (`regalloc N regs`, `vectorize N loop(s)`,
  etc.) on every real source file.

**Scaffold-vs-shipped honesty notes:**
- `--O1` runs constfold + identity collapse on the AST. SSA + DCE + CSE
  modules from v2 stay in their unit-test island; the asm emitter sees
  pre-folded literals which is what the witness criterion required.
- `regalloc/vectorize/lto/lifetimes` are *invoked* on every fn at the right
  flag, but the asm backend still uses stack slots, scalar loops, full-program
  emit, and inferred lifetimes. The integration step is "module sees real
  source", not "module drives asm output". Wiring those into actual
  lowering/emission is downstream (v4 territory).
- `async fn` + `.await` parse + run synchronously. Real state-machine
  transform + executor over `aether_thread_*` is downstream.
- `macro_rules!` skips the body verbatim and treats `name!(…)` as `name(…)`.
  Real hygienic expansion via the existing `mir::macros::expand` API is
  downstream.

## Blocking Issues

None. Audit reports `errors: 0`.

## What's Next (post-v3)

1. **Drive regalloc through the asm emitter.** Today the allocator runs
   and reports counts; the next step is rewriting `emit_expr_value` to use
   the assignment plan (move stack-slot reads/writes to {r10..r15} when the
   plan says so). The witness criterion is `cuda_train_transformer_block`'s
   .obj shrinking ≥30%.
2. **Replace the `name!()` shortcut with real macro_rules expansion.**
   Capture the rule's pattern + body as a token vector at parse time, hand
   to `mir::macros::expand` at the call site.
3. **Real async state-machine.** The v3 lowering is a no-op pass-through.
   `mir::async_exec::DelayFuture` + a thread-pool executor on top of
   `aether_thread_spawn` is the v4 starter.
4. **Fold v2's `mir::ssa` + `mir::opt` into the AST opt pass.** Today
   `ast_opt.rs` re-implements constfold; the SSA path is unused at runtime.
   Convert AST → linear `SsaStmt` list, run `ssa::rename_block` + `opt::*`,
   convert back. Removes duplication.
5. **Bench harness measurements.** `bench/optfx/run_all.ps1` exists; run it
   + append a row to `docs/BENCH_LEDGER.md` via the bench-runner subagent.
6. **`AE0200` family diagnostics.** `mir::lifetimes_drive` reports counts
   today; convert each `Checker::run` error into a real `Diag` with the
   `AE0200` code so `--check --strict-borrow` actually fails the build.

## Notes for Next Session

- **Workaround tax cleared.** Six items in
  `memory/asm_backend_known_gaps.md` are now witnessed:
    - mut params, i32 sign-extend, f32 unary neg, wide-frame __chkstk,
      imulq+mask, stack arrays. The memory file is now historical — keep
      for context, don't reach for the workarounds.
- **Audit reads both ROADMAP_V2.md AND ROADMAP_V3.md.** `tools/audit/src/roadmap.rs::run`
  concatenates items from both; v3 phases 11–14 show up under their own
  rows.
- **`bench/optfx/run_all.ps1` works at `--O1`.** Iteration loop
  `cargo build --bin aetherc && bench/optfx/run_all.ps1` produces a wall
  delta for new opts. Watch for fairness — both runs include startup cost.
- **`--O2` implies `--lto`.** Codified in `parse_args`. Don't add a third
  level unless there's something genuinely worth gating on it.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Audit (just witness count): `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `/c/Users/Matt/.cargo/bin/cargo.exe build --bin aetherc`
- Compile + run a witness: `target/debug/aetherc.exe tests/runtime/X.aether
  --emit=aether-bin -o tests/runtime/X.exe && tests/runtime/X.exe`
- New flags: `--O0` / `--O1` / `--O2` / `--lto` (post v3).
