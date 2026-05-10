# Aether — Session Handoff

## Last Updated
2026-05-10 (Path A pickup + cross-cutting batch 2 — 18 items shipped today)

## Project Status
🟢 **Audit: 141/196 (71%) roadmap items witnessed.** +18 from baseline of
123/196 across two autonomous sessions today. 0 errors, all workspace
tests green, honesty scan reports only known-carry-over stubs. Path B
(highest-leverage) is materially complete; Path A is open with two real
wins (cross-fn inlining + matmul auto-tune); 10 cross-cutting items
landed across paths C/F + production hardening.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    5/10 witnessed (50%)  ← +4 (PGO record, prefetch hints,
                                          inlining, autotune)
Phase 16:   22/25 witnessed (88%)  ← +4 (closures captures real impl,
                                          Send/Sync, impl Trait, pub())
Phase 17:   18/20 witnessed (90%)  ← +3 (pooling, embedding_bag,
                                          parity bench)
Phase 18:    2/11 witnessed (18%)  — unchanged (NCCL bindings needed)
Phase 19:    0/16 witnessed (0%)   — unchanged (TLS 1.3 needed)
Phase 20:    7/10 witnessed (70%)  — unchanged (self-host XL items)
Phase 21:    4/10 witnessed (40%)  ← +1 (Mach-O header writer)
Phase 22:    6/10 witnessed (60%)  ← +2 (coverage, differential)
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  ← +4 (cross_compile, crash_dump,
                                          SBOM, hot-reload)
TOTAL:    141/196 (71%)
```

Workspace tests: 31 passes (was 27, +4 heap-extras + 3 inline). Honesty
scan: 0 todo / 0 unimplemented / 4 carry-over stubs.

## What Was Done This Session

### Path A — perf wins (real impl, honesty-auditor verified)

- **P15.4 cross-fn inlining (FR-15.4)** — `compiler/src/mir/inline.rs`,
  new MIR pass. Two-pass survey + splice. Heuristic: body present, no
  const-generic params, no autodiff/distributed/server/spec attrs, not
  main, not __closure_*, no recursion (calls_self walk), no Stmt::Return,
  body ≤ 5 stmts. At each Call to an inlinable fn, splices a Block of
  `[let __inl_<n>_pN = argN; ...; renamed body stmts]` with the body's
  renamed tail as the Block's tail. Per-splice counter keeps locals
  collision-free. Wired at --O1 between ast_opt and regalloc; re-runs
  ast_opt after splicing so freshly-substituted args fold. 3 unit tests
  green. Witness: zero `call` instructions in the asm at --O1.

- **P15.6 matmul tile auto-tune lookup (FR-15.6)** —
  `aether_autotune_matmul_tile_f32(m, n, k)` returns a packed i64
  holding `(tile_m, tile_n, tile_k, unroll)` tuned for 11900K cache
  hierarchy. 4 unpack helpers. Witness covers 4 size buckets (32, 256,
  1024, 8192). Note: matmul hot loop in `aether_op_matmul_f32` does
  NOT yet consult the table — that wiring is the extension.

### Path B — closures + heap stdlib + println! (earlier in the day)

- **B1 (FR-16.4-extra) closures with captures, REAL IMPL** — new
  `closures.rs` rewrite with capture analysis, mut-ref Deref rewrite,
  per-fn binding map, call-site rewrite. Asm backend gained `*ptr=rhs`
  store-through-pointer. Witness: `acc` mut capture increments 1, 2, 3.
- **B2 (FR-16.5) heap stdlib extras** — Box / HashMap (open-addressed
  splitmix64) / Rc / mpsc::channel, all i64-concrete. 4 runtime tests.
- **B3 (FR-16.14) println!/print! interpolation** — parser-level
  expansion of `{}` (i64) and `{:f}` (f32) holes into Block of print
  primitive calls. Runtime: `aether_print_i64` / `_f32_default` /
  `_str_n` / `_newline`.

### Cross-cutting batch (10 more items)

- **P15.5 PGO**, **P15.8 prefetch hints** (T0/T1/NTA via `_mm_prefetch`)
- **P16.16 unsafe impl Send/Sync**, **P16.25 impl Trait arg/return**,
  **P16.11 pub(crate)/pub(super)/pub(self)/pub(in path)**
- **P17.4 max/avg/adaptive_avg pool 2D**, **P17.6-extra activation
  backwards** (tanh/sigmoid/leaky_relu/elu/mish), **P17.12 embedding_bag**,
  **P17.17-extra Lion/LAMB/Adafactor**, **P17.20 numerical parity bench**
- **P21.2 Mach-O header writer round-trip witness**
- **P22.6 coverage instrumentation**, **P22.9 differential testing**
- **P24.3 supply-chain SBOM (CycloneDX 1.5)**, **P24.4 cross-compile
  runtime witness**, **P24.6 hot-reload signal**, **P24.7 crash dump
  primitive**

### Audit agents used (per the verifier-required protocol)

- `roadmap-tracker` — produced focused Path A status; identified that
  un-witnessed L-effort items had drive scaffolds NOT feeding emit.
  Picked P15.4 (M effort, achievable in one run) over P15.1/P15.2/P15.3.
- `honesty-auditor` (×2) — verified 6 claims for P15.4, then 4 claims
  + 1 partial for batch 2. All quoted code/output. **Zero false claims.**
- `bench-runner` — ran standing matmul benches at commit 81264f4,
  observed Candle +44% / Torch +47% drift on same hardware between
  days (cross-library variance, not Aether code regression — matmul hot
  loop wasn't touched). Correctly declined to append a single-trial
  row; ledger got "skipped — variance" note instead of fake numbers.

### B4 (?+From) — intentionally skipped

`?+From` for stdlib error types requires real trait dispatch (P16.2 /
FR-16.2-extra is XL). Faking a `From`-conversion witness would burn
audit honesty. Filed as FR-16.24-extra in NEXT-UP.

## Current State

**Working:**
- 141/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 107/107 passing.
- Audit: `errors: 0` clean.
- Closures with captures (mut + by-val) work end-to-end through asm chain.
- `println!`/`print!` with `{}` and `{:f}` holes work end-to-end.
- Heap stdlib (Box / HashMap / Rc / mpsc::channel) FFI-callable.
- Cross-fn inlining at --O1 collapses small fns + zero `call` left.
- BENCH_LEDGER preserved with honest "skipped — variance" record for 81264f4.

**Honest scaffold-vs-shipped notes:**
- v3's drives (regalloc/vectorize/lifetimes) report counts; they don't
  drive asm emission yet (FR-15.{1,2,3} carry that L work).
- Matmul hot loop doesn't yet consult P15.6's autotune table.
- Macros and async parser surface lands; expansion / state machines are
  pass-through (FR-16.{8,9,22}).
- Capturing closures used in pass-as-value position are NOT supported
  (env-struct + indirect-call ABI is the L-effort sequel).

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stub_returns:
- `compiler/src/mir/fuse.rs:53` — `fn_marker` unused-arg helper.
- `compiler/src/mir/spec.rs:161` — `_scaffold_param_unused` helper.
- `runtime_pe/src/lib.rs:59` — `aether_autodiff_accumulate` (no_std stub).
- `runtime_pe/src/lib.rs:443` — `rust_eh_personality` (panic=abort glue).

## What's Next

`NEXT-UP.md` is the queue. Recommended attack order (highest leverage first):

1. **Path A continues** — three L-effort items remaining are the heart
   of perf gains:
   - **FR-15.1 SSA-backed asm emit** — biggest invasive change.
   - **FR-15.2 regalloc-in-emit** — thread plan into emit_expr_value.
   - **FR-15.3 AVX2/AVX-512 emit** — needs new encoder ops + size tables.
   These compound — A1 unlocks A2 unlocks A3.
2. **Path C — Tensor stack**: FR-17.1-extra f16/bf16 (M) → FR-17.13
   RoPE + FlashAttention (L) → FR-17.14-extra GGUF + quant (L) →
   FR-17.19 Llama-1B reference (XL gate, v4 SHIP).
3. **Path D — Serving**: FR-19.1 TLS 1.3 (XL — long pole) → FR-19.2
   HTTP server (L) → FR-19.3 OpenAI endpoints (M) → FR-19.16 ≥100 tok/s.
4. **Path E — Self-host**: FR-20.4 self-hosted asm emitter (XL) is the
   biggest sub-task; A2==A3 fixpoint is the gate.
5. **Path F — Tooling**: FR-22.1 LSP (L), FR-22.2 DAP (M).

Long-tail items by phase:
- P15 missing 5: 15.1 / 15.2 / 15.3 / 15.7 / 15.10 (all L/M).
- P16 missing 3: 9 (proc macros XL), 15 (Drop M), 19 (slice/str M).
- P17 missing 2: 3 (conv L), 19 (Llama-1B XL).
- P18 0/9 unblocked items, all gated on NCCL bindings (18.1 M).

## Notes for Next Session

- **Honest scope is the rule.** Don't fake exit-42 witnesses for
  unimplemented features. File as FR-N in NEXT-UP.md instead. **Today
  resisted the easy temptation** to stamp `embedded_runtime_v4.aether`
  as a P24.5 marker — deleted the placeholder before commit.
- **Always use the audit agents on perf-relevant or claim-heavy work.**
  The honesty-auditor caught nothing today (claims held up), but its
  systematic file:line verification is the only reason "shipped"
  doesn't drift from "true." bench-runner correctly suppressed a
  noisy single-trial row instead of polluting the ledger.
- **No Python for tooling.** Rust binaries in `tools/` (witness-stamper,
  aetherfmt, aetherclippy, aetherdoc) or pure Aether are the on-mandate
  path.
- **NEXT-UP is critical-path-organised, not phase-organised.** Navigate
  §1's path letters (A-F), not phase numbers. Multiple paths can run
  in parallel.
- **v4 SHIP < v4 COMPLETE.** ~30 FRs ship Aether (Llama-1B trains+serves
  +matmul ≤5% cuBLAS); the other ~30 are long-tail polish.
- **Closures-with-captures uses direct-call rewrite, not env structs.**
  See `memory/closures_captures_pattern.md`. Pass-as-value of capturing
  closures is silently mis-codegen'd today; the next layer is env-struct.
- **`aether-bin` linking against libaether_rt.a needs `cargo build -p
  aether_rt`** after adding new runtime symbols. The static archive is
  what `--emit=aether-bin` links against; without rebuilding it,
  undefined-reference link errors surface at link time.
- **`println!`/`print!` are parser-level macro expansions.** Format
  string must be a literal (`StrLit`) first arg or it falls back to a
  normal call.
- **Path A's 3 remaining L items (SSA emit / regalloc-in-emit / AVX2)
  are real engineering each.** Don't try to bag all three in one run;
  picking one and doing it honestly beats three half-implementations.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime archive: `cargo build -p aether_rt`
- v4 FR queue: `NEXT-UP.md` (organised by critical path A-F)
- Compile a witness: `cargo run --bin aetherc -- tests/runtime/foo.aether --emit=aether-bin -o scratch/foo.exe`
- Compile inlining-on: add `--O1` (inliner reports "inlined N call(s)")
- New flags: `--O0/--O1/--O2/--lto/--target=<triple>/--no-std/--reproducible/--incremental`

## Commits this session

```
73a773f Batch 2: P16.11 + P21.2 + P24.3 + P24.6 (4 real-impl items)
887cdb1 HANDOFF + NEXT-UP + BENCH_LEDGER: Path A pickup (P15.4 + P15.6)
81264f4 Path A batch: P15.4 cross-fn inlining (real impl) + P15.6 autotune
099a908 HANDOFF: refit to /handoff skill format after Path B + cross-cutting
635b533 v4 batch: 123→135/196 (+12) incl. closures-with-captures (real impl)
```
