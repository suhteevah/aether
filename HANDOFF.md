# Aether — Session Handoff

## Last Updated
2026-05-18 (Path A continuation — FR-15.1 SSA-driven emit shipped)

## Project Status
🟢 **Audit: 142/196 (72%) roadmap items witnessed.** +1 from
baseline of 141/196. 0 errors, all workspace tests green, honesty scan
unchanged (4 known-OK stubs). Path A unblocked at A1: FR-15.1 (the
gate for A2 and A3) is honestly shipped — SSA pipeline now drives a
real AST rewrite under `--O1`.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    6/10 witnessed (60%)  ← +1 (FR-15.1 SSA-driven emit)
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   18/20 witnessed (90%)  — unchanged
Phase 18:    2/11 witnessed (18%)  — unchanged (NCCL bindings needed)
Phase 19:    0/16 witnessed (0%)   — unchanged (TLS 1.3 needed)
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    142/196 (72%)
```

Workspace tests: 113 pass (3 new ssa_drive unit tests + 110 previous).
Honesty scan: 0 todo / 0 unimplemented / 4 known carry-over stubs.

## What Was Done This Session

### FR-15.1 — SSA-driven opt pipeline (real impl, honesty-auditor verified)

The existing SSA scaffold (`mir/ssa.rs::rename_block`) and opt passes
(`mir/opt.rs::{const_fold, strength_reduce, cse, dce}`) operated on a
parallel string-keyed IR but never affected codegen — that's the
"v3 drives report counts; they don't drive asm emission yet" gap
called out in the prior handoff.

**`compiler/src/mir/ssa_drive.rs` (new, 343 lines, 3 unit tests):**

- `pub fn drive(prog: &mut Program) -> Report` mutates the AST in-place.
- For each fn, walks the *leading run of pure let-bindings* (`Stmt::Let`
  with simple expr: leaf or `lhs op rhs` over Ident/IntLit for
  `op ∈ {Add, Sub, Mul, Shl}`) plus an optional tail expression.
- Linearises that prefix into the SSA `(lhs, op, Vec<rhs>)` shape;
  bails out at the first non-linearisable stmt (calls, ifs, loops,
  field access, method calls — left untouched).
- Pipeline: `rename_block` → `const_fold` → `strength_reduce` → `cse`
  → `dce_preserve_tail` (drops unused lhs but keeps the synthetic
  TAIL_SENTINEL stmt alive).
- Materialises the optimised SsaStmt list back into AST `Stmt::Let`s
  + (optional) `body.tail`. SSA-rename suffixes (`_<digits>`) stripped
  on materialise; CSE-aliased rhs operands resolve to the surviving
  original-name binding.
- `Report { fns_processed, stmts_in, stmts_out }` printed via stderr.

**`compiler/src/main.rs`:** call to `mir::ssa_drive::drive(&mut prog)`
threaded between the inline+ast_opt step and the regalloc/vectorize
drives inside `if args.opt_level >= 1`. Stderr message now reads
`[aetherc] --O1 ast-opt applied; inlined N call(s); ssa N fn(s)
X→Y stmts; regalloc … ; vectorize …`. **--O0 byte-compat preserved**
(the SSA driver never runs at `--O0`).

**`tools/audit/src/runtime_check.rs`:** new `build_flags: Vec<String>`
field on `RuntimeCase`, `// build-flags: <args...>` line parser
(whitespace-split), forwarded ahead of `--emit=...` in the aetherc
invocation. Cleanly orthogonal to `// expect:` / `// build-mode:` /
`// requires:`. This is what lets the witness opt into `--O1`.

**Witness — `tests/runtime/ssa_emit_drives_asm.aether`:** exercises all
four opts in a single body (constfold + strength_reduce + cse + dce);
`// build-flags: --O1` forces opt-level; tail = 42. Empirical evidence:

| metric              | --O0 | --O1 |
|---------------------|------|------|
| asm lines           |   48 |   35 |
| `imulq` count       |    2 |    0 |
| `shlq` count        |    0 |    1 |
| stderr "ssa" report |  n/a | `1 fn(s) 6→4 stmts` |

honesty-auditor verified 7 claims against file:line, command output,
and the audit's roadmap-progress diff. No false claims.

### Bench

Not required by the BENCH_LEDGER append rule (which triggers on
changes to `runtime/src/cuda.rs`, `runtime/src/lib.rs`,
`compiler/src/codegen/asm/`, or `compiler/src/mir/fuse.rs`). FR-15.1
touches only the AST-rewrite path before emit; the matmul hot loop and
runtime are untouched. Bench-runner subagent not invoked for this
commit per the rule.

## Current State

**Working:**
- 142/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 113 passing (+3 new SSA driver unit tests).
- Audit: `errors: 0` clean.
- `--O1` now emits visibly tighter asm for any fn whose body starts
  with a run of pure-arithmetic lets (constant-folded values, mul-by-
  pow2 reductions, duplicate-compute collapses, dead-binding drops).
- Foundation in place for A2 (regalloc-in-emit) and A3 (AVX2/AVX-512)
  to consume the SSA-flavoured intermediate without redoing the
  linearisation work.

**Honest scaffold-vs-shipped notes (carried over, still true):**
- The SSA driver linearises *the prefix of pure lets* only — calls,
  ifs, loops, field/method access break the run. That's still a big
  subset (every example file's main starts with a let-cascade), but
  it's a subset. Extending the linearisable set is iterative.
- v3's regalloc/vectorize drives still only report counts, they don't
  drive register assignment in `emit_expr_value`. That's FR-15.2.
- Matmul hot loop still doesn't consult P15.6's autotune table.
- Capturing closures used in pass-as-value position still mis-codegen.
- Macros and async parser surface lands; expansion / state machines
  remain pass-through (FR-16.{8,9,22}).

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stubs:
- `compiler/src/mir/fuse.rs:53` — `fn_marker` unused-arg helper.
- `compiler/src/mir/spec.rs:161` — `_scaffold_param_unused` helper.
- `runtime_pe/src/lib.rs:59` — `aether_autodiff_accumulate` (no_std stub).
- `runtime_pe/src/lib.rs:443` — `rust_eh_personality` (panic=abort glue).

## What's Next

`NEXT-UP.md` is the queue. Recommended attack order (highest leverage first):

1. **Path A continues — FR-15.2 regalloc-in-emit** is now the natural
   next L-effort step. The hook: thread the existing `mir::regalloc::
   Allocator` plan (already computed at `--O1`) into the asm backend's
   `emit_expr_value` so that frequently-read locals stay in r10..r15
   across loop bodies. Today every local re-loads from `disp(%rbp)`.
   Witness target: a fn whose body uses the same local 10+ times in a
   for-loop body sees ≥30% smaller .obj at `--O1` vs `--O0`.
2. **Path A — FR-15.3 AVX2/AVX-512 emit** after A2.
3. **Path C — Tensor stack toward v4 SHIP**: FR-17.1-extra f16/bf16
   (M) → FR-17.13 RoPE + FlashAttention (L) → FR-17.14-extra GGUF +
   quant (L) → FR-17.19 Llama-1B reference (XL gate).
4. **Path D — Serving**: FR-19.1 TLS 1.3 (XL, long pole).
5. **Path E — Self-host**: FR-20.4 self-hosted asm emitter (XL).
6. **Path F — Tooling**: FR-22.1 LSP (L), FR-22.2 DAP (M).

## Notes for Next Session

- **Honest scope is the rule.** FR-15.1 specifically did NOT claim to
  rewrite the asm backend itself — the SSA driver mutates the AST,
  then the existing AST-walking emit consumes the rewritten AST. That
  earns the "SSA drives asm emit" claim because asm output observably
  changes (imulq → shlq, fewer lines), but anyone claiming the asm
  backend now consumes SSA directly would be lying. A2 is the work
  that puts the regalloc plan into the emit path itself.
- **--O0 byte-compat means: the SSA driver only runs at --O1+.** Don't
  change the gate (`if args.opt_level >= 1`) without a witness that
  proves --O0 output is still bit-identical to today.
- **`// build-flags:` is the new audit knob.** Use it for any future
  witness that needs a specific opt-level or codegen flag to
  demonstrate the FR. Don't reach for it to hide a regression at
  --O0; the default-no-flags audit run is still the integrity check.
- **Closures-with-captures uses direct-call rewrite, not env structs.**
  See `memory/closures_captures_pattern.md`. Pass-as-value of capturing
  closures is silently mis-codegen'd today; the next layer is env-struct.
- **`aether-bin` linking against libaether_rt.a needs `cargo build -p
  aether_rt`** after adding new runtime symbols. The static archive is
  what `--emit=aether-bin` links against; without rebuilding it,
  undefined-reference link errors surface at link time.
- **No Python for tooling.** Rust binaries in `tools/` or pure Aether
  are the on-mandate path.
- **Path A's 2 remaining L items (FR-15.2 regalloc-in-emit + FR-15.3
  AVX2) are real engineering each.** Don't try to bag both in one
  run; FR-15.2 is the bigger dependency unblocker (FR-15.3 needs
  somewhere to put the SIMD-resident values, which is FR-15.2's job).

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime archive: `cargo build -p aether_rt`
- v4 FR queue: `NEXT-UP.md` (organised by critical path A-F)
- Compile a witness: `cargo run --bin aetherc -- tests/runtime/foo.aether --emit=aether-bin -o scratch/foo.exe`
- Compile SSA-on: add `--O1` (driver reports `ssa N fn(s) X→Y stmts`)
- Witness can pin opt-level: `// build-flags: --O1` at top of `.aether`
- Flags: `--O0/--O1/--O2/--lto/--target=<triple>/--no-std/--reproducible/--incremental`

## Commits this session

(Single commit lands at the end of this session — see `git log` after push.)
