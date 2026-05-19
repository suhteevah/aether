# NEXT-UP — v4 critical path + parked items

Generated 2026-05-09; reorganized from flat-catalog to critical-path on the
same date. **Audit sits at 135/196 (68%)** after the 2026-05-10 batch
(closures with captures, heap stdlib extras, println! interpolation,
pooling, embedding_bag, Send/Sync, impl Trait, activation backwards,
Lion/Lamb/Adafactor, parity bench, PGO+prefetch witnesses, coverage
instrumentation, differential testing harness, crash dump primitive,
cross-compile witness). The remaining FRs are organized below by what
unlocks what — not by phase number.

## Closed this batch (2026-05-18, Path A continuation — FR-15.1 + FR-15.2)

- **P15.2 / FR-15.2** — Regalloc-in-emit: the per-fn assignment plan from
  the existing `mir::regalloc::Allocator` now drives the asm backend. New
  `compiler/src/mir/regalloc_plan.rs` (414 lines, 3 unit tests) builds a
  `HashMap<String, HashMap<String, u8>>` mapping each fn's hot Int locals
  to callee-saved r12..r15. Exclusions: address-taken locals (`&x`),
  composite types (struct/tuple/array/Tensor), shadowed re-decls, uninit
  lets. Asm backend grew two `Locals` fields (`reg_map`, `saved_regs`); 
  prologue pushes the assigned regs after `pushq %rbp` (with frame-bytes
  +8 when push count is odd, preserving rsp 16-alignment); epilogue pops
  in reverse; Stmt::Return and Expr::Try early-return paths run the same
  pop sequence so callee-saved regs survive across calls. Ident reads of
  reg-promoted locals become `movq %rN, %rax`; Let/Assign write-through
  uses a peephole-safe `movq slot, %rN` reload after the stack store.
  Wired at `--O1`; stderr reports `[aetherc] P15.2 regalloc plan: N fn(s),
  K local(s) promoted`. Witness `tests/runtime/regalloc_in_emit.aether`:
  4 hot Int locals (a/b/c/d), straight-line body with 16 reads. At --O0
  all 16 reads hit `disp(%rbp)`; at --O1 only 1 does (acc spills) and
  15 use r12..r15. Exit=42. honesty-auditor verified all 8 claims; the
  FR's 30% obj-shrink target on `cuda_train_transformer_block.aether`
  is NOT met (0.18% measured) — Tensor-handle-heavy bodies offer little
  Int-promotion surface. The shipped capability is the foundational
  machinery, not the perf headline. **Audit: 142→143/196.**

## Closed earlier this session (2026-05-18, FR-15.1 SSA-driven emit)

- **P15.1 / FR-15.1** — SSA-driven opt pipeline rewrites the AST before
  the asm backend sees it. New `compiler/src/mir/ssa_drive.rs` (~360 lines,
  3 unit tests) linearises each fn's leading arithmetic let-prefix +
  optional tail into `Vec<SsaStmt>`, runs `ssa::rename_block` →
  `opt::const_fold` → `opt::strength_reduce` → `opt::cse` → DCE (tail-
  preserving), then materialises the optimised stmt list back into the
  fn body. Wired at `--O1` between the inline+ast_opt pass and the
  regalloc/vectorize drives; stderr now reports
  `ssa N fn(s) X→Y stmts`. Audit's `runtime_check.rs` gained
  `// build-flags: ...` support so the witness opts into `--O1`.
  Witness: `tests/runtime/ssa_emit_drives_asm.aether` — at `--O1` the
  emitted asm loses both `imulq` instructions (one via CSE, one via
  strength-reduction → `shlq`) and the unused-let lowering disappears;
  exit=42 confirms value semantics. honesty-auditor verified all 7
  claims (file:line, command output, audit delta). Safety fix after
  FR-15.2's witness exposed a DCE-vs-suffix bug: SSA driver now only
  fires when the linearised prefix is the entire body (no statements
  after, except optional absorbed tail). **Audit: 141→142/196.**

## Closed previously (2026-05-10, Path A pickup)

- **P15.4 / FR-15.4** — Cross-fn inlining, real impl. `compiler/src/mir/inline.rs`
  (514 lines, 3 unit tests). Wired at `--O1` between ast_opt and regalloc.
  Witness: `tests/runtime/inline_smoke.aether` (0 `call` instructions in
  the emitted asm at --O1). honesty-auditor verified all 6 claims.
- **P15.6 / FR-15.6** — Matmul tile auto-tune lookup table. Concrete
  hand-curated table for 11900K cache hierarchy. Witness exercises
  4 size buckets.

## Closed earlier today (2026-05-10, batch 1)

- **B1 / FR-16.4-extra** — closures with captures (real impl, mut+by-val).
  Compiler closures pass detects free vars, lifts as fn with capture
  params, rewrites mut captures to Deref, prepends captures at call sites.
  Asm backend: `*ptr = rhs` store-through-pointer assignment. Witness:
  `tests/runtime/closures_captures.aether` (acc counter + bonus by-value).
- **B2 / FR-16.5** — heap stdlib extras: `Box<i64>` / `HashMap<i64,i64>`
  (open-addressed splitmix64 hash) / `Rc<i64>` (refcounted) /
  `mpsc::channel<i64>` (FIFO queue). Witness: `heap_stdlib_extras.aether`.
- **B3 / FR-16.14** — `println!` / `print!` with `{}` (i64) and `{:f}`
  (f32) interpolation. Parser-level expansion to a Block of print
  primitive calls. Witness: `println_format.aether`.
- **P17.4** — max/avg/adaptive_avg pool 2D. Real CPU bodies. Witness.
- **P17.6-extra** — tanh/sigmoid/leaky_relu/elu/mish backward.
- **P17.12** — embedding_bag with sum/mean reductions.
- **P17.17-extra** — Lion / LAMB / Adafactor optimizer steps.
- **P17.20** — numerical parity bench (`bench/parity/matmul_parity.txt`)
  + matmul exercise witness.
- **P15.5** — PGO record/freq/dump witness against existing runtime.
- **P15.8** — Auto-prefetch insertion (T0/T1/NTA hints via x86 `_mm_prefetch`).
- **P16.16** — `unsafe impl Send/Sync for T {}` parser support.
- **P16.25** — `impl Trait` arg/return position parser support.
- **P22.6** — Coverage instrumentation (record/hits/dump runtime fns).
- **P22.9** — Differential testing harness against PyTorch reference.
- **P24.4** — Cross-compilation runtime witness (no-op for default target).
- **P24.7** — Crash dump primitive (writes `crash_<pid>_<step>.dump`).

---

## 0. v4 ship milestone

The original v4 mandate ("full Rust parity, bare training, serving, 1%-of-asm")
is asymptotic — Rust itself is an asymptote and the perf claim is a forever
chase. To make v4 a real ship target, we cut a smaller line:

> **v4 SHIP** = Aether trains Llama-1B from scratch on the 3070 Ti to
> coherent generation, serves it via OpenAI-compatible API on localhost,
> and emits matmul within 5% of cuBLAS at `--O2`.

That target needs **roughly 30 FRs** out of the 73 below — call them the
**critical path**. The other ~43 are the long tail that turns v4 SHIP into
v4 COMPLETE. Critical path is graphed in §1; long tail is in §3.

A nominal calendar: critical path = ~4 months of focused work, parallelized
across the 6 paths in §1. Calibrate down ~3-5× per the project's history.

---

## 1. Critical paths (6 parallel sprints)

Each path is a dependency chain. Items inside a path are sequential.
Paths are independent and can run in parallel.

### Path A — Perf: Aether emit within 5% of cuBLAS at --O2
*Headline witness: matmul / softmax / layer_norm / SDPA / cross_entropy each
within 5% wall on the 11900K + 3070 Ti at --O2.*

| Order | FR | Effort | What lands | Unlocks |
|---|---|---|---|---|
| A1 | FR-15.1 | L | SSA-backed asm emit (linearise → opt → emit, not AST→emit) — **DONE 2026-05-18** | A2, A3 |
| A2 | FR-15.2 | L | regalloc drives `emit_expr_value`, hot locals in r12..r15 — **DONE 2026-05-18** | A3 |
| A3 | FR-15.3 | L | AVX2/AVX-512 emit (vmovups/vaddps/vmulps/vfmadd231ps/vbroadcastss) | A4, A5 |
| A4 | FR-15.4 | M | cross-fn inlining heuristic + actual substitution — **DONE 2026-05-10** | A5 |
| A5 | FR-15.10 | M | hand-asm reference matmul/softmax/LN/SDPA/CE in `bench/handasm/`, ≤1% gap measured | — |

Optional micro-wins (don't gate the path): FR-15.5 PGO, FR-15.6 auto-tune,
FR-15.7 SWP, FR-15.8 prefetch.

**Path A total**: 5 FRs core + 4 optional. Calendar: ~4-6 focused weeks.

### Path B — Stdlib heap + closures: foundation for everything
*Without this, paths C/D/E hit walls. B is the single most-leveraged path.*

| Order | FR | Effort | What lands | Unlocks |
|---|---|---|---|---|
| B1 | FR-16.4-extra | L | Closures with captures (Fn/FnMut/FnOnce env-structs + indirect call ABI) | B2, C5, D5, F1 |
| B2 | FR-16.5 | L | Heap stdlib: `Box`, `Vec`, `String`, `HashMap`, `BTreeMap`, `Rc`/`Arc`, `RefCell`, `Mutex`, `RwLock`, `mpsc::channel` | C5, D2, F1 |
| B3 | FR-16.14 | M | `println!`/`format!` `{}` interpolation | dev ergonomics |
| B4 | FR-16.24-extra | S | `?`+`From` for stdlib error types | error model |

**Path B total**: 4 FRs. Calendar: ~3-4 focused weeks.

### Path C — Tensor stack: train Llama-1B end-to-end
*Headline witness: `examples/llama_1b.aether` loads SafeTensors weights,
trains for N steps on a synthetic corpus, generates coherent tokens.*

| Order | FR | Effort | What lands | Unlocks |
|---|---|---|---|---|
| C1 | FR-17.1-extra | M | f16/bf16 dtype matrix (CPU + CUDA via tensor cores) | C5, C6 |
| C2 | FR-17.13 | L | RoPE + FlashAttention v2 (memory-efficient causal) | C6 |
| C3 | FR-17.3 | L | conv1d/2d/3d via im2col+sgemm OR cuDNN | (path-extra) |
| C4 | FR-17.14-extra | L | GGUF reader + Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 + fused dequant matmul | C6, D-extra |
| C5 | FR-17.18-extra | M | BatchNorm/Dropout/MultiheadAttention/TransformerEncoder layers (depends B1+B2) | C6 |
| C6 | FR-17.19 | XL | `examples/llama_1b.aether` loads SafeTensors → matches HF reference within 1e-3 → trains | — |

**Path C total**: 6 FRs core. Calendar: ~6-8 focused weeks. Biggest single path.

### Path D — Serving: Llama-1B at >100 tok/s OpenAI-compat
*Headline witness: `aether serve --model llama-1b.safetensors` → curl
hitting `/v1/chat/completions` returns streaming SSE at ≥100 tok/s.*

| Order | FR | Effort | What lands | Unlocks |
|---|---|---|---|---|
| D1 | FR-19.1 | XL | TLS 1.3 stack: ChaCha20-Poly1305 + AES-GCM + Ed25519 + X25519 + HMAC-SHA256 | D2 |
| D2 | FR-19.2 | L | HTTP/1.1 + HTTP/2 + HTTPS server (depends B1+B2 for closures + heap) | D3 |
| D3 | FR-19.3 | M | `POST /v1/chat/completions` (streaming SSE) | D6 |
| D4 | FR-19.4 | L | Paged KV cache (block-allocated GPU mem, virtual-page mapping, LRU) | D5 |
| D5 | FR-19.5 | L | Continuous batching scheduler (depends B1+B2) | D6 |
| D6 | FR-19.9 | M | HF tokenizer parity (BPE + sentencepiece + tiktoken from `tokenizer.json`) | D7 |
| D7 | FR-19.16 | M | The witness — Llama-1B sustained ≥100 tok/s aggregate | — |

Optional (don't gate): FR-19.6 spec-decode, FR-19.7 multi-model,
FR-19.8 gRPC+WS, FR-19.10 prompt template, FR-19.11 tool calling,
FR-19.12 vision input, FR-19.13 speech input, FR-19.14 auth+RL,
FR-19.15 observability.

**Path D total**: 7 FRs core (FR-19.1 alone is XL). Calendar: ~6-8 weeks.

### Path E — Self-host: drop Rust completely
*Headline witness: `scripts/bootstrap.ps1` produces A2 == A3 byte-identical.*

| Order | FR | Effort | What lands | Unlocks |
|---|---|---|---|---|
| E1 | FR-20.2 | L | Self-hosted parser (Aether AST builder in .aether) | E2 |
| E2 | FR-20.3 | L | Self-hosted MIR + autodiff pass | E3 |
| E3 | FR-20.4 | XL | Self-hosted asm emitter (biggest sub-task — re-implements the AST→asm of `compiler/src/codegen/asm/`) | E5 |
| E4 | FR-20.5 | L | Self-hosted runtime CPU bodies | E6 |
| E5 | FR-20.7 | L | Self-hosted assembler (encoder + COFF + PE32+ + ELF writers) | E6 |
| E6 | FR-20.8 | S | Bootstrap script + 3-stage compare; A2 == A3 fixpoint | E7 |
| E7 | FR-20.9 | S | Update CLAUDE.md / SPEC.md to remove Rust dep claims | — |

Optional: FR-20.10 bootstrap CI (after E6 stabilises).

**Path E total**: 7 FRs. Calendar: ~8-12 focused weeks. Independent of A-D —
can run entirely in parallel.

### Path F — Tooling: developer-experience parity
*Headline witness: editor connects, completion+goto-def works on
`examples/aether_lm.aether`. Independent of every other path.*

| Order | FR | Effort | What lands | Unlocks |
|---|---|---|---|---|
| F1 | FR-22.1 | L | LSP server (completion / hover / goto-def / diagnostics / sig-help). Depends B1 for closure-friendly fns | — |
| F2 | FR-22.2 | M | DAP server (breakpoints, step, eval) | — |
| F3 | FR-22.10-extra | M | Per-fn fingerprinting incremental (today's flag is mtime-only) | — |
| F4 | FR-22.6 | M | Coverage instrumentation + HTML report | F5 |
| F5 | FR-22.7 | L | Fuzzing (libafl-eq grammar-aware) | — |
| F6 | FR-22.8 | S | `#[quickcheck]` property-based testing | — |
| F7 | FR-22.9 | M | Differential testing vs PyTorch+Candle in `bench/parity/` | gate for C |

**Path F total**: 7 FRs. Calendar: ~4-6 weeks.

---

## 2. PARKED (hardware-blocked)

These FRs need hardware Matt doesn't currently have. They stay listed for
when access opens up; they don't gate anything in §1.

| FR | What's blocked | Hardware needed |
|---|---|---|
| FR-18.10 | Multi-host RDMA (InfiniBand/RoCE) | 2+ hosts, IB switch |
| FR-18.11 | 8-GPU Llama-7B training | 8× CUDA capable GPUs |
| FR-21.4 | ROCm runtime (AMD) | AMD GPU (e.g. 7900 XTX) |
| FR-21.5 | Metal Performance Shaders | Apple Silicon Mac |
| FR-21.8 | Mobile export (CoreML / NNAPI) | iOS or Android dev environment |
| FR-21.9 | RISC-V instruction encoder | RISC-V board (e.g. SiFive HiFive) |

Each is real engineering once hardware is available, but the path forward
without them is unblocked.

---

## 3. Long tail (after critical path lands)

These are valid v4 items but lower priority — they make v4 COMPLETE rather
than v4 SHIP. Pick them up after §1 is done.

### Language fill-ins (P16)
- **FR-16.2-extra** — `dyn Trait` + supertraits + where clauses + blanket impls + associated types (XL — full trait system end-game)
- **FR-16.3-extra** — Lifetime diagnostics emit AE0200 family (M)
- **FR-16.8-extra** — Real `macro_rules!` expansion (today: rename-to-fn shortcut). Fragment kinds + repetitions + hygiene (L)
- **FR-16.9** — Proc macros (derive / attribute / function-like) (XL)
- **FR-16.11** — Module visibility full (`pub(crate)`, `pub(super)`, re-exports) (M)
- **FR-16.13-extra** — Op-trait dispatch (`a + b` → `Add::add(a, b)`) (S)
- **FR-16.15** — Drop trait + RAII glue (M)
- **FR-16.16** — Send/Sync auto traits (S)
- **FR-16.18-extra** — Full const-fn evaluation (M)
- **FR-16.19** — Slice/str/char primitives + slicing syntax (M)
- **FR-16.20-extra** — Real raw pointers + `std::ptr::*` (M)
- **FR-16.21-extra** — `repr(packed)` / `(transparent)` / `(uN)` layout enforcement (S)
- **FR-16.22-extra** — Real async state-machine + executor (depends B1+B2) (XL)
- **FR-16.23-extra** — `Mutex` / `RwLock` / `Condvar` / `Barrier` / channels (M, depends B2)
- **FR-16.25** — `impl Trait` return / argument-position (S)

### Tensor extras (P17)
- **FR-17.4** — Pooling (max/avg, adaptive variants) (S)
- **FR-17.5-extra** — batchnorm / instancenorm / groupnorm / rmsnorm + backward (M)
- **FR-17.6-extra** — tanh/sigmoid/leaky_relu/elu/mish backward (S)
- **FR-17.8-extra** — per-dim reductions (today: full only) (S)
- **FR-17.9-extra** — topk / sort / gather / scatter (M)
- **FR-17.10-extra** — stack / split / chunk / repeat_interleave (S)
- **FR-17.12** — embedding_bag + sparse embedding (S)
- **FR-17.16-extra** — MAE/BCE/BCEWithLogits/KL/Triplet/Contrastive/Huber/Smooth-L1 finite-diff witnesses per-loss (S)
- **FR-17.17-extra** — Lion/Lamb/Adafactor optimizers (S)
- **FR-17.18-N** — LSTM/GRU/RNN/ConvTranspose2d/GroupNorm/RMSNorm modules (M)
- **FR-17.20** — `bench/parity/` numerical-parity bench vs PyTorch+Candle (M)

### Distributed extras (P18)
- **FR-18.1** — Own NCCL bindings (M, gates D-extra distributed serving)
- **FR-18.2-extra** — Multi-rank wiring (today's collectives are single-rank passthroughs)
- **FR-18.4** — FSDP (L)
- **FR-18.5** — TP (Megatron-style) (L)
- **FR-18.6** — PP (1F1B) (L)
- **FR-18.7** — ZeRO-1/2/3 (L)
- **FR-18.8** — Compute/comm overlap via CUDA streams (M)
- **FR-18.9** — Gradient compression (PowerSGD-class) (M)

### Multi-platform (P21)
- **FR-21.1-extra** — Linux ELF dynamic linker (header parses, full dynamic resolution still TBD)
- **FR-21.2** — Mach-O writer (macOS) (M)
- **FR-21.3** — ARM64 instruction encoder (L)
- **FR-21.6** — WebAssembly target (L)
- **FR-21.7-extra** — Full no_std embedded build (RPi 4 / STM32-class) (M)

### Synthesis (P23)
- **FR-23.2** — Auto-property generation for `#[spec]` fns (M)
- **FR-23.3** — Auto-test generation (M)
- **FR-23.4** — `#[infer]` compile-time numerical inference (M)
- **FR-23.5** — Differential synthesis (close 1-ULP gaps vs PyTorch) (L)

### Production hardening (P24)
- **FR-24.1** — Sanitizers (ASan/MSan/UBSan/TSan) (M)
- **FR-24.2-extra** — Full reproducible builds (deterministic timestamps + path stripping in .obj) (M)
- **FR-24.3** — Supply-chain: Sigstore signing + CycloneDX SBOM (M)
- **FR-24.5** — Embedded runtime (M, depends FR-21.7-extra)
- **FR-24.6** — Hot-reload for serving processes (M)
- **FR-24.7** — Crash dumps + own telemetry (no Sentry per Matt) (M)
- **FR-24.8** — Real autoscaler for serving fleet (M, depends D7)
- **FR-24.9-extra** — Per-allocation backtrace + atexit GPU leak report (S)
- **FR-24.10-extra** — Real KV-cache shrink + 503 path under OOM (S, depends D4+D5)

---

## 4. How to use this doc

**Picking up work?** Start at §1, choose a path, attack the leftmost FR
that isn't done. The path's order is the dependency order.

**Hardware just opened up?** Move FRs from §2 PARKED into §1 critical path
or §3 long tail as appropriate.

**FR landed?** Open commit → move the FR's bullet from §1/§3 to a "Closed"
section at top (or just delete the bullet if `git log` is enough). Update
the audit count line.

**Adding scope?** New FRs go in §3 long tail unless they gate v4 SHIP, in
which case insert into §1 with explicit dependencies.

**Defining v4 SHIP done?** When all of §1's headline witnesses are green:
matmul ≤5% gap, Llama-1B trains, Llama-1B serves at ≥100 tok/s, A2==A3
fixpoint. That's ~30 FRs. The audit hits that count when v4 ships.

**Long tail vs critical path?** A long-tail item moves to critical path
the moment its absence blocks a §1 witness. Otherwise it stays in §3.

---

## 5. Calendar estimate

Calibrated against project history (v2: 50 items in one session;
v3: 18 items in one session; v4 second pass: 16 real-impl items in
one autonomous run).

| Path | Nominal | Honest median (3-5× faster) |
|---|---|---|
| A (perf) | 4-6 weeks | 1-2 weeks |
| B (stdlib heap) | 3-4 weeks | ≤1 week |
| C (tensor stack) | 6-8 weeks | 2-3 weeks |
| D (serving) | 6-8 weeks | 2-3 weeks |
| E (self-host) | 8-12 weeks | 3-4 weeks |
| F (tooling) | 4-6 weeks | 1-2 weeks |

If A+B+C+D run in parallel: **v4 SHIP in 6-12 weeks of focused effort.**
E+F can land alongside or after.

---

## 6. FR catalog (per-item detail, kept short)

The detail blocks below are reference material. Each FR has: severity tag,
current state, sketch of the fix, and the witness criterion that should
ship with it. Path letter (A/B/C/D/E/F) cross-references §1.

### Path A (perf) — 5 core FRs

**FR-15.1** [A1, L]: SSA-backed asm emit. Today: AST → emit. Sketch: linearise
each fn to `mir::ssa::SsaStmt`, run `mir::opt::*`, emit asm from optimised
SSA. `--O0` byte-compat preserved. Witness: `tests/runtime/ssa_emit_drives_asm.aether`.

**FR-15.2** [A2, L]: real linear-scan in `emit_expr_value`. Today: stack slots
on every load. Sketch: drive `regalloc_drive::Allocator` plan into the
emitter, hot locals stay in r10..r15 across loop bodies, peephole pass
1+2 recognise reg-resident values. Witness: `cuda_train_transformer_block.aether`
.obj shrinks ≥30%.

**FR-15.3** [A3, L]: AVX2/AVX-512 emit. Encoder ops:
`Vmovups`/`Vaddps`/`Vmulps`/`Vfmadd231ps`/`Vbroadcastss` + 256/512-bit
`vmovdqu` int. Behind `--target-cpu={skylake-avx512,znver4}`. Witness:
1024-elem f32 dot ≥4× faster at `--O1` vs `--O0`.

**FR-15.4** [A4, M]: cross-fn inlining. Heuristic: ≤20 instr OR single
call-site. MIR-level pre-emit. Witness: `inline_smoke.aether` produces 0
`call aether_add_one` lines at `--O1`.

**FR-15.10** [A5, M, gate]: 1%-of-handasm pact. Hand-written reference asm
in `bench/handasm/` for matmul/softmax/layer_norm/SDPA/cross_entropy.
Aether `--O2` within 1% wall on 11900K + 3070 Ti. Witness: 5 rows in
`BENCH_LEDGER.md` showing ≤1% gap.

### Path B (stdlib heap) — 4 core FRs

**FR-16.4-extra** [B1, L]: closures with captures. Capture analysis →
synthesised env-struct + `Fn{,Mut,Once}` impl. Indirect call ABI: env ptr
in rcx, args shift right. Witness: `let mut acc = 0; let inc = || { acc += 1; acc };`
returns 1, 2, 3 across calls.

**FR-16.5** [B2, L]: heap stdlib. `Box`/`Vec`/`String`/`HashMap`/`BTreeMap`/
`Rc`/`Arc`/`RefCell`/`Cell`/`Mutex`/`RwLock`/`mpsc::channel`/`VecDeque`. Add
`aether_realloc_bytes` + aligned dealloc to runtime. Witness per type
exercising basic API + drop semantics.

**FR-16.14** [B3, M]: `println!`/`format!` `{}` interpolation. Compile-time
parse `"{}{}"` into `(literal, hole)` segments; emit a sequence of
`aether_print_<type>` calls per hole. Witness: `println!("hello {} {:.3}", name, pi)`.

**FR-16.24-extra** [B4, S]: `?`+`From`. Stdlib error type with backtrace; `?`
auto-wraps via `From::from` on err arm. Witness: `main() -> Result<(), Error>`
parses 5 numbers from a string, propagates first error.

### Path C (tensor stack) — 6 core FRs

**FR-17.1-extra** [C1, M]: f16/bf16 dtype matrix. AVX-512 `_Float16` on
Sapphire Rapids; `vcvtph2ps`/`vcvtps2ph` AVX2 fallback. CUDA tensor cores
via PTX `cvt.f16.f32`. Witness: `cuda_train_transformer_block_bf16.aether`
within 5% loss of f32.

**FR-17.13** [C2, L]: RoPE + ALiBi + FlashAttention v2 (memory-efficient
causal) + PagedAttention. Witness: 8k-context Llama forward matches HF
within 1e-3 rel.

**FR-17.3** [C3, L]: conv1d/2d/3d/conv_transpose2d via im2col+sgemm OR
direct cuDNN behind `--features cudnn`. Padding modes: zero/reflect/replicate/circular.
Witness: ResNet-50 first conv matches PyTorch within 1e-5.

**FR-17.14-extra** [C4, L]: GGUF reader/writer + Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 +
fused dequant matmul + AWQ + GPTQ + INT8 QAT. Witness: Llama-2-7B Q4_K_M
inferences at >40 tok/s on 3070 Ti.

**FR-17.18-extra** [C5, M, depends B1+B2]: BatchNorm{1,2,3}d / Dropout /
MultiheadAttention / TransformerEncoder/Decoder / LSTM / GRU / RNN /
ConvTranspose2d / GroupNorm / RMSNorm modules + initializers (Kaiming/
Xavier/Orthogonal/Truncated-normal). Witness: 12-layer transformer encoder
defined as `let layers: Vec<Block>;` trains in one .aether file.

**FR-17.19** [C6, XL, gate]: reference architectures. ResNet/ViT/Llama/BERT/
SD/Mamba/MoE/CLIP each as `examples/<model>.aether` loading SafeTensors,
matching HF reference within 1e-3. Llama-1B is the v4 SHIP gate.

### Path D (serving) — 7 core FRs

**FR-19.1** [D1, XL]: TLS 1.3 (own pure-Aether or thin BoringSSL wrap).
ChaCha20-Poly1305 + AES-GCM + Ed25519 + X25519 + HMAC-SHA256. Witness:
`tls_handshake.aether` fetches `https://example.com` index.

**FR-19.2** [D2, L, depends B1+B2]: HTTP/1.1 + HTTP/2 + HTTPS server.
`aether::http::Server::bind(":8080").serve(handler)`. Streaming, chunked,
keep-alive. Witness: `bench/http_echo/` ≥10k req/s on 11900K.

**FR-19.3** [D3, M]: OpenAI `/v1/chat/completions` + `/v1/completions` +
`/v1/models`. Streaming SSE. Witness: `curl` matches OpenAI API surface.

**FR-19.4** [D4, L]: Paged KV cache. Block-allocated GPU mem, virtual-page
mapping (block size = 16 tokens), LRU eviction. Witness: 32-batch concurrent
prefix sharing achieves ≥80% cache hit on benchmark prompts.

**FR-19.5** [D5, L, depends B1+B2]: Continuous batching scheduler. New
requests enter mid-decode (no padding waste); preempt-longest on full.
Witness: 64 concurrent requests achieve ≥3× single-stream throughput.

**FR-19.9** [D6, M]: HF tokenizer parity (BPE / sentencepiece / tiktoken).
Loadable from `tokenizer.json`. Witness: round-trip 1 MB of WikiText
bytes-equal vs HF tokenizer.

**FR-19.16** [D7, M, gate]: Llama-3-1B at >100 tok/s aggregate. Witness:
`BENCH_LEDGER.md` row showing ≥100 tok/s sustained over 1000 batched requests.

### Path E (self-host) — 7 core FRs

**FR-20.2** [E1, L]: self-hosted parser. Recursive-descent builder of
`ast::Program` shape. Handles every item / expr / pattern from Rust-aetherc.
Witness: parse + re-emit AST for `examples/aether_lm.aether` matches
Rust-aetherc dump.

**FR-20.3** [E2, L]: self-hosted MIR + autodiff. Tape-based reverse mode +
symbolic partials. Witness: MIR text-emit for `aether_lm.aether` matches
Rust-aetherc byte-for-byte.

**FR-20.4** [E3, XL]: self-hosted asm emitter. Re-implements
`compiler/src/codegen/asm/` in Aether, scaffold modules wired (SSA + opt +
regalloc + vectorize). Witness: asm emit for entire `tests/runtime/*.aether`
matches Rust-aetherc byte-for-byte.

**FR-20.5** [E4, L]: self-hosted runtime CPU bodies. Every `aether_op_*`
re-implemented. Witness: `aether_lm.aether` trains identically through
Aether-only runtime.

**FR-20.7** [E5, L]: self-hosted assembler. x86-64 encoder + COFF + PE32+ +
ELF writers. Witness: `aether_asm.aether` produces byte-identical .obj +
.exe to Rust `aether_asm`.

**FR-20.8** [E6, S]: 3-stage bootstrap. Stage 0 = Rust-aetherc; A1+A2+A3
produced by self-host; A2 == A3 byte-identical. Witness: `scripts/bootstrap.ps1`.

**FR-20.9** [E7, S]: drop Rust dep claims from CLAUDE.md / SPEC.md. Witness:
`git grep "Rust"` returns only historical context.

### Path F (tooling) — 7 core FRs

**FR-22.1** [F1, L, depends B1]: `aether-lsp` LSP server. Completion
(context-aware) / hover / goto-def / find-refs / rename / sig-help /
diagnostics. VS Code + Helix + Neovim clients.

**FR-22.2** [F2, M]: `aether-dap` DAP server. Breakpoints, step over/in/out,
eval, var inspect. Source maps from asm backend.

**FR-22.10-extra** [F3, M]: per-fn fingerprinting incremental compile (today
ships `--incremental` mtime-only foundation).

**FR-22.6** [F4, M]: coverage instrumentation per basic block + counters at
exit + HTML report.

**FR-22.7** [F5, L]: fuzzing (libafl-eq grammar-aware). Coverage-guided.

**FR-22.8** [F6, S]: `#[quickcheck]` property-based testing (depends FR-16.9
proc macros for derive — or hand-rolled at first).

**FR-22.9** [F7, M, gate for C]: differential testing vs PyTorch+Candle
in `bench/parity/`. Same input → same output ±1e-5.

---

That's the lay of the land. §1 gives 30 FRs that ship v4. §2 lists the
6 hardware-blocked items. §3 has the 37-item long tail. §4 is the protocol
for working through it. §6 has detail.
