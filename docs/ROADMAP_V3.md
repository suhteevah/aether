# Aether Roadmap v3 — wire the scaffolds, close the asm gaps

**Provenance**: written 2026-05-09. Roadmap v2 closed 50/50 witnessed (commit `fa087c7`). Nine compiler modules (`mir::ssa`, `opt`, `regalloc`, `vectorize`, `lto`, `traits`, `lifetimes`, `async_exec`, `macros`) ship with green unit tests but **do not drive codegen** — the AST-walking emitter in `compiler/src/codegen/asm/mod.rs` is still what every `.aether` test goes through. v3's mandate: turn witness-grade items into shipped-feature items, plus close the asm-backend gaps that forced workarounds in the v2 sweep.

> **Goal**: every v2 scaffold is invoked on the compile path under a flag; every v2 workaround listed in `HANDOFF.md` is gone; bench ledger gains rows for at least three new optimisations with measured deltas.

## Cross-cutting rules (carried from v2)

1. **Audit monotone** — `target/debug/aether-audit.exe` must stay 50/50+ green; v3 items add new tagged witnesses, never delete old ones.
2. **Honesty scan green** — no new `todo!()`/`unimplemented!()`; spec-mode for stubs.
3. **Bench every perf-relevant item** — bench-runner subagent appends to `docs/BENCH_LEDGER.md` after each merge that touches asm/runtime/fuse paths.
4. **Subagent ownership** — each item names the watcher that gates merge. No item closes without that subagent's sign-off.

## Subagent watcher team (who owns what)

| Subagent | Owns |
|---|---|
| `roadmap-tracker` | session-start status, item count |
| `witness-test-author` | drafts the tagged `tests/runtime/<name>.aether` per item |
| `bench-runner` | appends `BENCH_LEDGER.md` row after asm/runtime/fuse touches |
| `coverage-matrix` | (op, dtype, device) table after Phase 11 work |
| `honesty-auditor` | every external claim cross-referenced before it ships |

---

# Phase 11 — Wire the v2 scaffolds into codegen

## 11.1 `--O1` flag drives `mir::ssa` + `mir::opt` (M)
- New CLI flag `--O0` (default) / `--O1` / `--O2`.
- At `--O1`: per-fn linear stmt list → `ssa::rename_block` → `opt::const_fold` → `opt::strength_reduce` → `opt::dce` → `opt::cse` → asm emitter.
- Preserve existing `--O0` behaviour byte-for-byte (golden artifacts unchanged at `--O0`).
- **Witness** (`witness-test-author`): `tests/runtime/o1_constfold.aether` — fn whose body is `let x = 2 * 3 * 7;` compiles to a single `movq $42` immediate at `--O1` (verify via `--emit=asm` grep), and exits 42.
- **Bench gate** (`bench-runner`): ≥3% wall improvement on `bench/optfx/run_all.ps1` (new) over a representative scalar microbench.
- Roadmap tag: `P11.1`.

## 11.2 Real linear-scan integration (L, depends 11.1)
- Drive `mir::regalloc::Allocator` over the SSA'd fn; map virtual regs to {`r10`, `r11`, `r12`, `r13`, `r14`, `r15`} (callee-saved tracked + spilled to stack on call boundaries per MS x64 ABI).
- Spill code reuses the existing stack-slot allocator.
- Hot locals stay in regs across loop bodies.
- **Witness**: `cuda_train_transformer_block.aether` .obj shrinks ≥30% (per v2 7.0 criterion line in v2). Tagged `regalloc_real`, `P11.2`.
- **Bench gate**: `bench/matmul_micro/` regresses no more than 1%; `bench/optfx/scalar_inner/` improves ≥10%.

## 11.3 Loop vectorizer (M, depends 11.2)
- `mir::vectorize::VectorPlan` plugged into the lowering of `for i in 0..N { … }` when:
  - body has no calls,
  - all stores hit a single contiguous tensor,
  - trip count is constant or runtime-known.
- Emit AVX2 (256-bit `vpaddd`/`vmulps`/etc.) by default; AVX-512 path behind `--target-cpu=skylake-avx512`.
- Scalar remainder loop for the tail.
- **Witness**: `tests/runtime/vec_dot.aether` — 1024-element f32 dot product runs ≥4× faster at `--O1` vs `--O0` on the 11900K. Tagged `P11.3`.

## 11.4 Cross-crate LTO (S, depends 11.1)
- `--lto` flag walks `mir::lto::Reachability` from `extern fn aether_*` + `main` roots; drops dead `pub fn` from final .obj.
- Runs after 11.1's per-fn opts.
- **Witness**: `examples/aether_lm.exe` shrinks ≥15% with `--lto`. Tagged `P11.4`.

## 11.5 Wire `mir::lifetimes` into `--check` (M)
- After parse, before MIR autodiff: synthesize an MIR borrow CFG and feed events to `lifetimes::Checker`.
- Surface failures as `AE0200`/`AE0201`/`AE0202` (already-defined codes).
- Mode: `--check --strict-borrow` opts in; default warn-only for one release to ease migration.
- **Witness**: keep `borrow_check.aether` green; add `tests/aether/negative/expect_AE0200_mut_alias.aether` to conformance.

---

# Phase 12 — Parser surface for v2's invisible features

## 12.1 `trait` + `impl Trait for Type` (L)
- Lex: `trait` keyword (already reserved? verify in `lexer/mod.rs`).
- Parse: `trait Foo { fn bar(&self) -> i32; }`, `impl Foo for T { … }`.
- Wire `mir::traits::Resolver` to AST collection; build dispatch table at MIR.
- Static dispatch via monomorphization; `dyn Trait` deferred to v4.
- **Witness**: `tests/runtime/trait_static_dispatch.aether` — two structs impl same trait, call site picks the right body. Tagged `P12.1`.

## 12.2 `'a` lifetime annotations (M, depends 12.1 partial)
- Parse `&'a T` and `&'a mut T` in `Ty::Ref`.
- Today's `&T` keeps inferring an anonymous lifetime; explicit `'a` flows through to `mir::lifetimes::Region`.
- **Witness**: `tests/runtime/explicit_lifetime.aether`. Tagged `P12.2`.

## 12.3 `async fn` parser + state-machine lowering (L)
- Parse `async fn foo() -> T` and `expr.await`.
- Lower per `mir::async_exec::DelayFuture` shape: each `await` becomes a state in a synthesized enum; the body becomes a `poll` impl.
- Real executor: thread pool over `aether_thread_spawn` + work-stealing deque.
- **Witness**: `tests/runtime/async_two_tasks.aether` — two `async fn` reading from disk in parallel, joined; finishes <2× single-task time. Tagged `P12.3`.

## 12.4 `macro_rules!` parser (L)
- Lex `macro_rules! name { (pat) => { body }; }`.
- Hand to `mir::macros::expand` at parse-end (pre-MIR).
- Limit v3 scope: token-tree literal patterns + `$x:expr`/`$x:ident`/`$x:tt` fragment kinds; full repetition (`$($x:tt),*`) deferred unless trivial.
- **Witness**: `tests/runtime/macro_vec.aether` — user-defined `vec![1, 2, 3]` expands to a `Vec`-shaped chain. Tagged `P12.4`.

## 12.5 `*ref` deref expression (S)
- Parser today rejects unary `*` on a ref (HANDOFF.md note). Add `Expr::Deref(Box<Expr>)`; lower to `movq (%reg), %rax` for refs into stack slots.
- **Witness**: `tests/runtime/deref_local.aether` — `let x = 5; let r = &x; *r` returns 5. Tagged `P12.5`.

---

# Phase 13 — Close the asm-backend gaps (`memory/asm_backend_known_gaps.md`)

Each item below is a workaround that bit during the v2 sweep. They block straight-line porting of larger programs.

## 13.1 `mut` on fn params (S)
- `fn f(mut x: i64) { x += 1; … }` parses + emits the param's stack slot as writable.
- **Witness**: `tests/runtime/mut_param.aether`. Tagged `P13.1`.

## 13.2 i32 sign-extend on load (S)
- Loads from i32 stack slots into 64-bit regs use `movsxd` not `movl`.
- Lex/parser already accept `i32`; only codegen path is broken.
- **Witness**: `tests/runtime/i32_negative_roundtrip.aether` — store -7 as i32, load + add 50 → 43. Tagged `P13.2`.

## 13.3 f32 unary `-` (S)
- `-x` for f32 lowers to `xorps xmm, [sign_mask]`; sign mask interned in `.rdata` once per fn.
- **Witness**: `tests/runtime/f32_unary_neg.aether`. Tagged `P13.3`.

## 13.4 Wide-frame stability (M)
- Symptom: SIGSEGV when a single fn accumulates ~40+ `let`/byte_set sequences.
- Suspect: `sub rsp, imm32` or shadow-region overlap when frame > 4 KiB (Win32 needs `__chkstk` for >4 KiB stack growth).
- Fix: emit `__chkstk` thunk call when prologue requires `>4096` bytes; thunk is a 30-byte hand-rolled probe loop in `aether_rt`.
- **Witness**: `tests/runtime/wide_frame_60_locals.aether` — fn with 60 i64 locals, each used. Tagged `P13.4`.

## 13.5 Multiply-then-mask without segfault (S)
- `(h * BIG_PRIME) & MASK` segfaulted under asm; FNV-1a witness fell back to byte-sum.
- Suspect: 64-bit `imul` overflow path, or wrong reg width on the mask `andq`.
- Fix: ensure `imulq r, r` lowering uses 64-bit form (`REX.W + 0F AF`), not `imull`.
- **Witness**: `tests/runtime/fnv1a_byte_hash.aether` — re-enable the FNV-1a inner loop from `cargo_manifest.aether`. Tagged `P13.5`.

## 13.6 Stack-allocated arrays (M)
- `let buf: [i64; 16];` allocates 128 bytes contiguous, indexed via `disp32(%rbp, %rax, 8)`.
- Today every "array" goes through `aether_alloc_bytes` (heap).
- **Witness**: `tests/runtime/stack_array_sum.aether` — sum 0..16 in a stack-resident array, returns 120. Tagged `P13.6`.

---

# Phase 14 — Bench cadence + coverage matrix

## 14.1 New bench harnesses
- `bench/optfx/` — scalar microbench suite (constant fold, CSE, DCE wins).
- `bench/conv2d/` — 64x3x224x224 ResNet-style first conv (Aether vs Candle vs PyTorch).
- `bench/attention/` — 1x16x2048x64 causal attention (Aether SDPA vs PyTorch SDPA).
- Each ships a `run_all.ps1` mirroring `bench/matmul_micro/`.
- **Owner**: `bench-runner` appends a row per harness on first run + after any perf-relevant commit.

## 14.2 Coverage matrix snapshot (S)
- `coverage-matrix` subagent runs after 13.x lands; output committed at `docs/COVERAGE_MATRIX.md`.
- Grid: rows = ops in `runtime/src/lib.rs`, cols = (f32, f64, bf16, f16) × (CPU, CUDA).
- Hard-fails the v3 close criteria if any op claimed in roadmap text is missing from the grid.

---

## Done criteria for v3

1. `--O1` flag exists; `bench/optfx/` shows ≥3% on at least one row; SSA + opt + DCE + CSE all touched in compile path (verifiable via `--emit=mir` diff at `--O0` vs `--O1`).
2. `cuda_train_transformer_block.aether` .obj shrinks ≥30% via real regalloc.
3. Every gap in `memory/asm_backend_known_gaps.md` has a green tagged witness.
4. `mir::traits` invoked on every parse (not just unit tests); `trait_static_dispatch.aether` green.
5. `mir::lifetimes` invoked on every `--check`; `expect_AE0200_mut_alias.aether` failing with the right code.
6. `mir::async_exec`, `mir::macros` reached from real source files (not synthetic test inputs).
7. `docs/BENCH_LEDGER.md` has ≥6 new rows post-v3-start; `honesty-auditor` signs each one.
8. Audit count ≥ 60/60.

## Suggested execution order

1. **13.1, 13.2, 13.3, 13.5, 12.5** — small asm fixes; clear out the workaround tax first (1 evening, parallelisable subagents).
2. **13.4, 13.6** — frame stability + stack arrays; unblocks larger fns.
3. **11.1** — `--O1` plumbing; the rest of phase 11 stacks on it.
4. **11.2 → 11.3 → 11.4** — sequential, each leans on the prior.
5. **12.1 → 12.2** — trait + lifetime parsers in one push.
6. **11.5** — wire lifetimes once 12.2's `'a` lands.
7. **12.3, 12.4** — async + macros; XL items, can run in parallel.
8. **14.x** — bench + coverage drumbeat; runs throughout.

History calibration: v2 priced these scaffolds at L/XL each and we wrote the first 9 modules + 24 unit tests + 22 witnesses in one evening. Treat the M/L tags as upper bounds; expect 3-5× faster.
