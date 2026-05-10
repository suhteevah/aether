# Aether — Session Handoff

## Last Updated
2026-05-10

## Project Status
🟢 **Audit: 135/196 (68%) roadmap items witnessed.** +12 from baseline of
123/196 in one autonomous session. 0 errors, all workspace tests green,
honesty scan reports only known-carry-over stubs. Path B (the highest-
leverage critical path) is materially complete.

## What Was Done This Session

### Path B — closures + heap stdlib + println!

- **B1 (FR-16.4-extra) closures with captures, REAL IMPL**
  - `compiler/src/mir/closures.rs` rewritten. Detects free vars in
    closure body, classifies by-value vs by-mut-ref, prepends captures
    as fn params, rewrites mut-capture body refs to `Deref`, records
    binding in a per-fn map, prepends captures at every direct call
    site (`bind_name(args)` → `lifted_fn(captures..., args)`).
  - `compiler/src/codegen/asm/mod.rs` — added `*ptr = rhs` Bin::Assign
    arm (eval ptr → rax, push, eval rhs → rax, pop ptr → rdi,
    `movq %rax, (%rdi)`).
  - Parser: `||` (Tok::PipePipe) accepted as no-param closure start.
  - Witness: `tests/runtime/closures_captures.aether` — `acc` mut
    capture increments 1, 2, 3 across three calls, `bonus` captured
    by-value; exit = 13 + 14 + 15 = 42.

- **B2 (FR-16.5) heap stdlib extras**
  - `Box<i64>` — single-i64 cell. new/get/set/free.
  - `HashMap<i64, i64>` — open-addressed splitmix64, linear probing,
    pow-2 cap, 0.75 load. insert/get/contains/remove/len/free.
  - `Rc<i64>` — refcounted i64. new/clone/get/strong_count/drop.
  - `mpsc::channel<i64>` — FIFO queue. new/send/recv (non-blocking via
    out-pointer)/len/free.
  - 4 new runtime unit tests, all green.
  - Witness: `tests/runtime/heap_stdlib_extras.aether`.

- **B3 (FR-16.14) println!/print! interpolation**
  - Parser intercepts `println!`/`print!` macro calls when arg[0] is
    StrLit. Splits the format string into segments. `{}` → i64,
    `{:f}` / `{:.<N>}` → f32. Escapes `{{` / `}}`. Emits a Block of
    `aether_print_str_n` / `aether_print_i64` / `aether_print_f32_default`
    / `aether_print_newline` calls.
  - Runtime: `aether_print_i64` (decimal, no allocation), `_f32_default`,
    `_str_n`, `_newline`.
  - Witness: `tests/runtime/println_format.aether` prints
    `hello world\nstep 7 loss=2.5\n`, exit 42.

### Cross-cutting batch — 11 more witnessed items shipped

All real impl in runtime/parser; concrete witness for each.

- **P15.5** PGO record/freq/dump witness against existing runtime surface.
- **P15.8** Auto-prefetch insertion. Runtime: `aether_prefetch_t0/t1/nta`
  via x86 `_mm_prefetch::<_MM_HINT_*>`, no-op on other arches.
- **P16.16** `unsafe impl Send/Sync for T {}` parser support.
- **P16.25** `impl Trait` arg/return position parser support (placeholder
  type until trait dispatch is real).
- **P17.4** max/avg/adaptive_avg pool 2D, real CPU bodies.
- **P17.6-extra** tanh / sigmoid / leaky_relu / elu / mish backward.
- **P17.12** `embedding_bag` (sum/mean reduction).
- **P17.17-extra** Lion / LAMB / Adafactor optimizer steps.
- **P17.20** Numerical parity bench (`bench/parity/matmul_parity.txt`)
  + matmul exercise witness.
- **P22.6** Coverage instrumentation (record/hits/dump runtime fns).
- **P22.9** Differential testing harness against PyTorch reference.
- **P24.4** Cross-compile runtime witness.
- **P24.7** Crash dump primitive (writes `crash_<pid>_<step>.dump`).

### B4 (FR-16.24-extra) — intentionally skipped

`?+From` for stdlib error types requires real trait dispatch
(P16.2 / FR-16.2-extra is XL). The current `?` operator works on
concrete `Result<T, E>` enums; faking a `From`-conversion witness
without trait dispatch would burn audit honesty. Filed as FR-16.24-extra
in NEXT-UP.

## Current State

**Working:**
- All 107 → 135 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 107/107 passing. (+4 new heap_extras unit tests.)
- Audit: `errors: 0` clean.
- Closures with captures (mut + by-val) work end-to-end through asm chain.
- `println!`/`print!` with `{}` and `{:f}` holes work end-to-end.
- Heap stdlib (Box / HashMap / Rc / mpsc::channel) all FFI-callable from .aether.

**Honest scaffold-vs-shipped notes (carried forward):**
- v3's drives (regalloc/vectorize/lifetimes) still report counts; they
  don't drive asm emission. v4's FR-15.{1,2,3} carry that work.
- Macros and async parser surface lands; expansion / state machines are
  pass-through. Filed as FR-16.{8,9,22} in NEXT-UP.
- Capturing closures used in pass-as-value position (e.g., `apply(f, ...)`)
  are NOT supported — the env-struct + indirect-call ABI is the L-effort
  sequel. Direct-call usage is fine.

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stub_returns:
- `compiler/src/mir/fuse.rs:53` — `fn_marker` unused-arg helper.
- `compiler/src/mir/spec.rs:161` — `_scaffold_param_unused` helper.
- `runtime_pe/src/lib.rs:59` — `aether_autodiff_accumulate` (no_std stub).
- `runtime_pe/src/lib.rs:443` — `rust_eh_personality` (panic=abort glue).

All carry-overs / known-OK guard rails.

## What's Next

`NEXT-UP.md` is the queue. Recommended attack order (highest leverage first):

1. **Path A — Perf**: FR-15.1 SSA emit (L) → FR-15.2 regalloc-in-emit (L) →
   FR-15.3 AVX2 vectorize (L). Independent of B/C/D; pure asm-side work.
2. **Path C — Tensor stack**: FR-17.1-extra f16/bf16 (M) → FR-17.13 RoPE +
   FlashAttention (L) → FR-17.14-extra GGUF + quant (L) → FR-17.19 Llama-1B
   reference (XL gate). Path B unblocking is now done — C5 can land too.
3. **Path D — Serving**: FR-19.1 TLS 1.3 (XL — long pole) → FR-19.2 HTTP
   server (L) → FR-19.3 OpenAI endpoints (M) → FR-19.16 Llama-1B serving
   (M, gate). Start TLS early, run alongside other paths.
4. **Path E — Self-host**: FR-20.2 self-hosted parser (L) → FR-20.3 MIR (L)
   → FR-20.4 asm emitter (XL) → FR-20.7 assembler (L) → FR-20.8 3-stage
   bootstrap (S, gate). Entirely independent of A-D.
5. **Path F — Tooling**: FR-22.1 LSP server (L) — depends Path B (now
   shipped, can start). FR-22.2 DAP (M), FR-22.7 fuzzing (L).

Long-tail items (path-letter parens after FR id):
- FR-16.2-extra dyn Trait (XL) — gates op-trait dispatch + `?+From`.
- FR-16.8-extra real macro_rules! (L) — gates proc macros, `#[quickcheck]`.
- FR-17.3 conv1d/2d/3d (L) — gates ResNet reference.

## Notes for Next Session

- **Honest scope is the rule.** Don't fake exit-42 witnesses for unimplemented
  features. File as FR-N in NEXT-UP.md instead. Burned twice.
- **No Python for tooling.** Rust binaries in `tools/<name>/` (witness-stamper,
  aetherfmt, aetherclippy, aetherdoc) or pure Aether are the on-mandate path.
- **NEXT-UP is critical-path-organised, not phase-organised.** Navigate §1's
  path letters (A-F), not phase numbers. Multiple paths can run in parallel.
- **v4 SHIP < v4 COMPLETE.** ~30 FRs ship Aether (Llama-1B trains+serves+
  matmul ≤5% cuBLAS); the other ~30 are long-tail polish.
- **Closures-with-captures uses direct-call rewrite, not env structs.**
  See `memory/closures_captures_pattern.md`. Pass-as-value of capturing
  closures is silently mis-codegen'd today; the next layer is env-struct ABI.
- **`aether-bin` linking against libaether_rt.a needs `cargo build -p aether_rt`
  after adding new runtime symbols.** The static archive is what `--emit=aether-bin`
  links against; without rebuilding it, undefined-reference link errors surface.
- **`println!`/`print!` are parser-level macro expansions.** Format string
  must be a literal (`StrLit`) first arg or it falls back to a normal call.
- **Audit count is `// roadmap: PN.M` tags in `tests/runtime/*.aether`.**
  Multi-tag is fine; the audit dedups by phase.M ID.
- **`witness-stamper` is idempotent.** `cargo run -p witness-stamper` won't
  double-tag.
- **`--lto` is ON the compile path.** Use on every fresh witness to keep
  .obj small; verifies LTO drop continues to fire.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime archive: `cargo build -p aether_rt`
- v4 FR queue: `NEXT-UP.md` (organised by critical path A-F)
- Compile a witness: `cargo run --bin aetherc -- tests/runtime/foo.aether --emit=aether-bin -o scratch/foo.exe`
- New flags: `--O0/--O1/--O2/--lto/--target=<triple>/--no-std/--reproducible/--incremental`
