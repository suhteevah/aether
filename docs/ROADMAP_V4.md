# Aether Roadmap v4 — full Rust parity, bare training, serving, 1%-of-asm perf

**Provenance**: written 2026-05-09. v3 closed 68/68 (100%) — every v2 scaffold module is invoked at the right flag, every asm-backend gap from `memory/asm_backend_known_gaps.md` has a tagged witness, parser surface for `trait`/`'a`/`async`/`macro_rules!`/`*ref` lands. **What v3 did NOT do**: the scaffold modules report counts to stderr but the asm emitter still lowers stack-slot scalars; macro/async/lifetime semantics are pass-throughs; the heap stdlib doesn't exist; conv/pool/quant/distributed/serving don't exist; we still ship Rust binaries.

> **Goal**: Aether is at full feature parity with shipping Rust, the runtime trains and serves real models end-to-end with no Python or C++ in the path, and the compiler-emitted machine code lands within 1% of hand-tuned x86-64 assembly on the kernels that matter. v4 is the work that gets us there.

## Effort scale (calibrated, not nominal)

- **S** = 1 evening of focused push
- **M** = 2-3 sessions
- **L** = ≤2 weeks
- **XL** = ≤1 month
- Calibrate down 3-5× — history says so. v3 priced 18 L/XL items; landed in one session. The audit-witness-only path ratchets these way down.

## Cross-cutting rules

1. **Audit monotone, never red.** Currently 68/68. v4 items add tagged witnesses; v4 close = ≥150/150.
2. **Honesty scan green.** No new `todo!()` / `unimplemented!()`. Keep guard-rail panics minimal and explicit.
3. **Bench every perf claim.** `bench-runner` subagent appends a row to `docs/BENCH_LEDGER.md` after any commit touching `runtime/src/cuda.rs`, `runtime/src/lib.rs`, `compiler/src/codegen/asm/`, or `compiler/src/mir/fuse.rs`.
4. **Self-hosted bootstrap is the primary client.** Every feature SHOULD have a witness compilable by the eventual self-hosted aetherc, not just by Rust-aetherc.
5. **Scaffold-vs-shipped honesty.** When an item lands behind a flag and exercises a scaffold without driving codegen, the witness comment says so explicitly. v3's drives are the model.

## Watcher team (carries forward from v2/v3)

| Subagent | Owns |
|---|---|
| `roadmap-tracker` | session-start + per-claim status |
| `witness-test-author` | drafts each `tests/runtime/<name>.aether` |
| `bench-runner` | `BENCH_LEDGER.md` rows after perf-relevant commits |
| `coverage-matrix` | `docs/COVERAGE_MATRIX.md` after any new (op, dtype, device) lands |
| `honesty-auditor` | every external claim cross-referenced before it ships |

---

# Phase 15 — Real codegen (the 1%-of-asm play)

The v3 scaffolds *report counts*. v4 makes them *drive emit*.

## 15.1 SSA-backed asm emit (L)
- Linearise each fn into `mir::ssa::SsaStmt`, run `mir::opt::{const_fold, strength_reduce, dce, cse}`, then emit asm from the optimised SSA — not from the AST.
- Existing `--O0` byte-compatible (golden artifacts unchanged).
- **Witness**: `tests/runtime/ssa_emit_drives_asm.aether` — `--emit=mir --O1` shows pre-optimised SSA matches post-optimised SSA up to the dropped statements.
- Tag: `P15.1`.

## 15.2 Real linear-scan in `emit_expr_value` (L)
- Replace today's stack-slot-on-every-load with the v3 `regalloc_drive::Allocator` plan — hot locals stay in r10..r15 across loop bodies, spills only when the pool is exhausted.
- Update peephole pass 1 + 2 to recognise reg-resident values (no fake `movq slot, %rax` on already-resident).
- **Witness**: `cuda_train_transformer_block.aether` .obj shrinks ≥30%. Tagged `P15.2`.
- **Bench**: `bench/matmul_micro` regresses ≤1%; `bench/optfx/scalar_inner` improves ≥10%.

## 15.3 Loop vectorizer emits AVX2/AVX-512 (L)
- The v3 `vectorize_drive` reports loop counts; v4 emits the vectorised body.
- AVX2 default; AVX-512 behind `--target-cpu=skylake-avx512` / `--target-cpu=znver4`.
- Scalar remainder loop tail.
- New asm encoder ops: `Vmovups`, `Vaddps`, `Vmulps`, `Vfmadd231ps`, `Vbroadcastss`, plus 256-bit + 512-bit `vmovdqu` for int.
- **Witness**: `vec_dot_real.aether` — 1024-elem f32 dot at `--O1` runs ≥4× faster than `--O0` on 11900K.
- Tag: `P15.3`.

## 15.4 Cross-fn inlining (M)
- Heuristic: inline fns ≤20 instructions OR fns with single call-site.
- Inlining happens at MIR level after monomorphisation, before emit.
- **Witness**: `inline_smoke.aether` — fn `add_one(x) { x + 1 }` called 5× in `main` produces 0 `call aether_add_one` lines in the asm at `--O1`.
- Tag: `P15.4`.

## 15.5 Profile-guided optimisation (M)
- `--profile-gen` instruments fn entries + branch directions; counters spilled to a `.aetherprof` file at `aether_pgo_dump_atexit`.
- `--profile-use=path.aetherprof` reads counters; biases inlining + register allocation + branch layout.
- **Witness**: `pgo_record.aether` exercises a hot/cold split; `--profile-use` shrinks .obj of the hot path ≥10%.
- Tag: `P15.5`.

## 15.6 Auto-tuning matmul tile/unroll search (M)
- `--auto-tune=matmul_micro` runs 30+ tile/unroll/blocking variants in a sandbox; picks the winner per `(M, K, N)` shape; persists to `tune_cache.aether-toml`.
- Resolves at codegen for known-shape matmul calls.
- **Witness**: 4096³ f32 sgemm on the 11900K within 5% of OpenBLAS sgemm in `bench/matmul_micro/`. Tagged `P15.6`.

## 15.7 Software pipelining for inner loops (M)
- For-loops with no carried deps + ≥4-cycle FU latency get prologue/kernel/epilogue split with the body interleaved across iterations.
- Targeted at the matmul micro-kernel and SDPA inner loop.
- **Witness**: `swp_matmul_inner.aether` measured at ≥1.3× the unpipelined baseline at the same f32 throughput.
- Tag: `P15.7`.

## 15.8 Auto-prefetch insertion (S)
- Walk strided memory accesses; emit `prefetcht0` 4 cache lines ahead.
- New asm encoder op: `PrefetchT0RbpDisp { disp: i32 }`.
- **Witness**: `prefetch_stream.aether` — 64 MiB strided sum gets ≥10% bandwidth lift on the 11900K.
- Tag: `P15.8`.

## 15.9 Real LTO drops dead pub fns (S)
- Gate the asm emitter on the v3 `lto_drive` reachability set: skip fns the set doesn't contain.
- **Witness**: `examples/aether_lm.exe` shrinks ≥15% with `--lto`. Tagged `P15.9`.

## 15.10 The 1%-of-asm pact (M, the witness for the whole phase)
- For matmul / softmax / layer_norm / SDPA / cross-entropy: hand-written reference asm sits in `bench/handasm/`. Aether-emitted code at `--O2` runs within 1% wall on the 11900K + the 3070 Ti.
- **Witness**: `BENCH_LEDGER.md` has rows showing ≤1% gap on each of the five named kernels. Tagged `P15.10`.

---

# Phase 16 — Rust language parity (the rest)

v3 added parser surface; v4 makes the semantics real.

## 16.1 HM type inference fully wired (L)
- Algorithm W with a unification table; remove the `: type` requirement from every `let` whose rhs has a derivable type.
- Generic param inference at call sites for type generics (today's const-shape inference is the model).
- **Witness**: ≥80% of `: type` annotations in the existing `tests/runtime/*` suite become removable (tracked by an `audit --only inference-coverage` dimension). Tagged `P16.1`.

## 16.2 Trait system end-game (XL)
- Associated types (`type Output;`) — name-resolution + monomorphisation.
- `dyn Trait` via vtable layout (fat pointer = data + vtable).
- Supertraits (`trait B: A`).
- Where clauses (`fn f<T>() where T: Foo + Bar`).
- Blanket impls (`impl<T: Foo> Bar for T`).
- Default-method inheritance + override.
- **Witness**: re-implement the GPU op surface as `trait Tensor` with f32/f16/bf16 backends; same training loop drives every backend.
- Tag: `P16.2`.

## 16.3 Lifetimes enforced as diagnostics (M)
- Convert `mir::lifetimes::Checker` errors into `AE0200`/`AE0201`/`AE0202` `Diag` entries.
- `--check --strict-borrow` fails the build on any violation.
- **Witness**: `tests/aether/negative/expect_AE0200_mut_alias.aether` produces exactly that code; existing positive witnesses stay green.
- Tag: `P16.3`.

## 16.4 Closures with captures (L)
- Capture analysis: free-var → Fn / FnMut / FnOnce.
- Synthesised env struct + `Fn{Mut,Once}` impl.
- Indirect call ABI: env ptr in rcx, args shift right.
- **Witness**: `let mut acc = 0; let inc = || { acc += 1; acc };` — `inc()` returns 1 then 2 then 3.
- Tag: `P16.4`.

## 16.5 Heap-allocated stdlib types (L)
- `Box<T>`, `Vec<T>` (capacity-doubling growth), `String` (UTF-8 + `&str` view), `HashMap<K, V>` (open-addr, FxHash), `BTreeMap<K, V>`, `Rc<T>` / `Arc<T>` (atomic for Arc), `RefCell<T>` / `Cell<T>`, `Mutex<T>`, `RwLock<T>`, `mpsc::channel<T>`, `VecDeque<T>`.
- Allocator: `aether_alloc_bytes` exists; add `aether_realloc_bytes`, `aether_dealloc_aligned`.
- **Witness**: each type has a `tests/runtime/heap_<name>.aether` exercising basic API + drop semantics.
- Tag: `P16.5`.

## 16.6 Iterator trait + 25+ adapters (M, depends 16.2 + 16.4)
- `trait Iterator { type Item; fn next(&mut self) -> Option<Self::Item>; }`.
- Adapters: `map`, `filter`, `fold`, `take`, `skip`, `chain`, `zip`, `enumerate`, `collect`, `sum`, `product`, `count`, `max`, `min`, `any`, `all`, `position`, `find`, `flat_map`, `flatten`, `rev`, `cycle`, `step_by`, `windows`, `chunks`, `peekable`.
- `for x in iter` desugar.
- **Witness**: `(0..100).filter(|i| i % 3 == 0).sum::<i64>() == 1683`. Tagged `P16.6`.

## 16.7 Pattern matching full (M)
- Range (`1..=5`), slice (`[a, b, ..rest]`), guards (`Some(x) if x > 0`), or-patterns (`A | B`), struct destructuring (`Point { x, y }`), reference patterns (`&Some(ref x)`), `@` bindings (`n @ 1..=5`).
- **Witness**: `pattern_full.aether` exhaustive-match witness covering each new shape. Tagged `P16.7`.

## 16.8 Real `macro_rules!` expansion (L)
- v3 skips the body. v4 captures pattern + body as token vectors, hands to `mir::macros::expand` at the call site.
- Fragment kinds: `expr`, `ident`, `tt`, `pat`, `ty`, `block`, `stmt`, `path`, `lit`, `meta`.
- Repetitions (`$($x:expr),*`) + nested repetitions.
- Hygiene (token-level identifier renaming).
- **Witness**: user-defined `vec![1, 2, 3]` lowers to `Vec::from_iter([1, 2, 3])`. Tagged `P16.8`.

## 16.9 Proc macros (XL, depends 16.8)
- Three flavours: `#[derive(...)]`, `#[attribute]`, `name!(...)` function-like.
- Compile-as-aether-fn that consumes a `TokenStream` and produces a `TokenStream` — fully self-hosted, no `proc_macro2`-style external dep.
- **Witness**: `#[derive(Debug, Clone, PartialEq)]` on a struct emits the three impls automatically, exercised by `derive_smoke.aether`.
- Tag: `P16.9`.

## 16.10 Cargo equivalent (`aether-pkg`) (L)
- Manifest format `Aether.toml` modelled on `Cargo.toml`.
- Resolver: registry + git + path; semver intersection.
- Workspaces, features (with feature unification), build scripts (`build.aether`).
- Build cache: per-crate fingerprint + incremental compilation.
- `aetherc build` / `aetherc test` / `aetherc run` / `aetherc publish`.
- **Witness**: a 5-crate workspace; touching one crate triggers only that crate's rebuild. Tagged `P16.10`.

## 16.11 Module system + visibility (M)
- `mod foo` / `mod foo { ... }` declarations.
- `pub`, `pub(crate)`, `pub(super)`, `pub(in path)`.
- Re-exports (`pub use`).
- Submodule trees with file-system mapping (`foo/bar.aether` ↔ `mod foo { mod bar; }`).
- **Witness**: a 4-module crate with private impls + re-exports compiles + the asm contains exactly the public surface. Tagged `P16.11`.

## 16.12 Standard I/O (M)
- **Filesystem**: `File`, `BufReader`, `BufWriter`, `fs::{read, write, read_dir, metadata, copy, rename, remove_file, create_dir_all}`.
- **Network**: `TcpStream`, `TcpListener`, `UdpSocket` with non-blocking + read/write timeouts.
- **Process**: `Command::new(...).spawn()`, pipes, exit code, env vars.
- **Env**: `env::var`, `env::args`, `env::current_dir`.
- **Witness**: 50-line HTTP/1.1 echo server in Aether handles 10k req/s. Tagged `P16.12`.

## 16.13 Operator overloading (S, depends 16.2)
- `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Neg`, `Not`, `BitAnd`, `BitOr`, `BitXor`, `Shl`, `Shr`, `Index`, `IndexMut`, `Deref`, `DerefMut`, `PartialEq`, `PartialOrd`, `Eq`, `Ord`, `Hash`.
- **Witness**: a custom `Vec3` struct with `+`, `*`, `-` operators behaves like `f32` arithmetic; `vec3_ops.aether` exit=42.
- Tag: `P16.13`.

## 16.14 Display/Debug + format!/println! (M, depends 16.8 + 16.13)
- Replace today's `println(LITERAL)` with full `println!("{} {}", a, b)`.
- `Debug` and `Display` traits + `{:?}` and `{}` format specs.
- Format specs: width, precision, fill, alignment, padding (`{:>10.2}`).
- **Witness**: `println!("hello {} {:.3}", name, pi);` emits the right string. Tagged `P16.14`.

## 16.15 Drop trait + RAII (M, depends 16.5)
- Drop glue inserted at scope end.
- Drop order: lexical reverse (last defined dropped first).
- `mem::drop(x)` early-drop intrinsic.
- **Witness**: a `LeakyVec` whose `Drop::drop` increments a static counter; counter matches `n` after `n` scoped allocations. Tagged `P16.15`.

## 16.16 Send/Sync auto traits (S, depends 16.2)
- Automatically derived from struct field traits.
- `thread::spawn` requires `T: Send + 'static`.
- `Arc<T>` requires `T: Send + Sync`.
- **Witness**: a struct with a `Cell<i64>` field fails to send across threads with `AE0210`. Tagged `P16.16`.

## 16.17 Test framework (`#[test]` runner) (S, depends 16.8)
- `#[test]` attribute; aetherc emits a `__aether_test_<n>` shim per test fn.
- `#[should_panic]`, `#[ignore]`.
- `assert!`, `assert_eq!`, `assert_ne!` macros.
- `aetherc test` walks every `#[test]` in the crate, runs each, summarises pass/fail.
- **Witness**: existing `tests/runtime/*.aether` migrate to `#[test]` form; audit unchanged. Tagged `P16.17`.

## 16.18 const fn evaluation (M)
- `const fn` evaluated at compile time when called with literal args.
- Used for shape arithmetic, lookup tables, embedded checksums.
- **Witness**: `const FACTORIAL_10: i64 = factorial(10);` compiles to `movq $3628800` immediate. Tagged `P16.18`.

## 16.19 Slice + str + char primitives (M)
- `[T]` unsized slice; `&[T]` fat pointer.
- `str` UTF-8 slice; `&str` fat pointer.
- `char` 32-bit Unicode scalar.
- Methods: `len`, `is_empty`, `iter`, `iter_mut`, slicing syntax `s[a..b]`.
- **Witness**: `let v: Vec<i64> = vec![1, 2, 3]; let s: &[i64] = &v[..]; assert_eq!(s.len(), 3);`. Tagged `P16.19`.

## 16.20 Unsafe + raw pointers (M)
- `unsafe { ... }` blocks.
- `*const T` / `*mut T` types.
- Pointer arithmetic, dereferencing, transmute.
- `std::ptr::{read, write, copy_nonoverlapping}`.
- **Witness**: a hand-rolled `Vec` impl in unsafe Aether returns 42. Tagged `P16.20`.

## 16.21 `repr` attributes (S, depends 16.20)
- `#[repr(C)]`, `#[repr(packed)]`, `#[repr(transparent)]`, `#[repr(u8)]` (for enums).
- Drives layout decisions in the asm backend.
- **Witness**: `#[repr(C)] struct Foo { ... }` round-trips through a C FFI call layout-equivalent to the C struct. Tagged `P16.21`.

## 16.22 Real async state machine + executor (XL, depends 16.4 + 16.5 + 16.8)
- `async fn` body lowers to a state-machine struct (continuations as enum variants).
- Real executor: thread pool over `aether_thread_spawn` + work-stealing deque.
- `tokio::spawn` / `tokio::join!` equivalents.
- IO drivers: `aether_async::fs`, `aether_async::net` (epoll on Linux, IOCP on Windows).
- **Witness**: 1000 concurrent `aether_async::fs::read` tasks complete in <1 ms wall. Tagged `P16.22`.

## 16.23 Concurrency primitives complete (M, depends 16.5)
- v3 has atomics + thread spawn/join. Add `Mutex<T>`, `RwLock<T>`, `Condvar`, `Barrier`, `Once`, `LazyLock`, `mpsc::channel`, `mpmc::channel` (crossbeam-style).
- **Witness**: 8-thread parallel matmul reaches >6× single-thread on the 11900K. Tagged `P16.23`.

## 16.24 Error model + `?` + `From` (S, depends 16.2)
- v3 has the `?` operator. v4 wires `From::from` on the err arm.
- `anyhow::Error`-style stdlib type with backtrace.
- `thiserror::Error` derive proc macro (depends 16.9).
- `main() -> Result<(), Error>` allowed.
- **Witness**: a fn parses 5 numbers from a string with `?` chains; first error propagates with the right code + backtrace. Tagged `P16.24`.

## 16.25 impl Trait return / arg-position (S, depends 16.2)
- `fn it() -> impl Iterator<Item = i64>` opaque return type.
- `fn f(x: impl Read)` argument-position generic.
- **Witness**: an iterator chain returned via `impl Iterator` consumes correctly in main. Tagged `P16.25`.

---

# Phase 17 — Tensor stdlib parity (Candle ∪ PyTorch)

The op-surface lift. Each row is a CUDA kernel + CPU body + dispatch entry + numerical-parity test.

## 17.1 Full dtype matrix (M, depends 16.2)
- Add: `f16`, `bf16`, `i8`, `u8`, `i16`, `u16`, `u32`, `u64`, `bool` (packed u8).
- Have: `f32`, `f64`, `i32`, `i64`.
- AVX-512 + `_Float16` intrinsics on Sapphire Rapids; `vcvtph2ps`/`vcvtps2ph` on AVX2.
- CUDA: native f16/bf16 via PTX `cvt.f16.f32` + tensor cores.
- **Witness**: `cuda_train_transformer_block_bf16.aether` trains with bf16 weights + fp32 master, loss within 5% of f32 baseline. Tagged `P17.1`.

## 17.2 N-D tensor + strided views (M)
- Layout = `(shape, strides, offset)`.
- Zero-copy: `transpose(d1, d2)`, `narrow(d, start, len)`, `slice([..2, 1.., ::2])`, `permute`.
- Broadcasting: align trailing dims, expand size-1 via stride=0.
- `reshape`: contiguous fast-path; copy fallback otherwise.
- `contiguous()` explicit copy.
- **Witness**: ResNet-50's first conv (with channel-last `permute` of input) compiles + matches PyTorch within 1e-5 rel. Tagged `P17.2`.

## 17.3 Convolutions (L)
- `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d` (im2col + sgemm OR direct cuDNN).
- Depthwise + dilated variants.
- Padding modes: zero, reflect, replicate, circular.
- **Witness**: `bench/conv2d/run_all.ps1` shows Aether within 5% of cuDNN sgemm-conv on 64×3×224×224 ResNet first conv. Tagged `P17.3`.

## 17.4 Pooling (S)
- `max_pool{1,2,3}d`, `avg_pool{1,2,3}d`, adaptive variants (`adaptive_avg_pool2d`).
- Ceil mode, count_include_pad, dilation.
- **Witness**: ResNet-50 stem (conv → bn → relu → maxpool) matches PyTorch within 1e-5 rel. Tagged `P17.4`.

## 17.5 Norm family (M)
- `batch_norm`, `instance_norm`, `group_norm`, `rms_norm` (have `layer_norm`).
- Backward for each.
- **Witness**: BatchNorm running-stat tracking matches PyTorch over a 100-step training loop. Tagged `P17.5`.

## 17.6 Activation family + backward (S)
- `silu`/`swish`, `tanh`, `sigmoid`, `leaky_relu`, `elu`, `mish`, `glu`, `swiglu`, `geglu`. Have: `gelu`, `relu`, `softmax`.
- Backwards finite-diff verified.
- **Witness**: each activation has a `tests/runtime/act_<name>_grad.aether` matching finite-diff to 1e-4. Tagged `P17.6`.

## 17.7 Math primitives (S)
- `log`, `exp`, `sin`, `cos`, `tan`, `atan2`, `pow`, `sqrt` (have sqrtss), `recip`, `abs`, `sign`, `clamp`.
- **Witness**: each routes through libm-equivalent or hand-tuned poly approximations within 1 ULP. Tagged `P17.7`.

## 17.8 Reductions (M)
- `sum`, `mean`, `var`, `std`, `min`, `max`, `argmax`, `argmin`, `prod` per-dim or full.
- Welford's algorithm for `var`/`std` (numerically stable).
- **Witness**: `reduce_full.aether` exercises every reduction shape; results within 1e-5 of PyTorch. Tagged `P17.8`.

## 17.9 Selection (S)
- `topk`, `sort` (radix or merge), `where`, `masked_fill`, `gather`, `scatter`.
- **Witness**: `topk_5.aether` selects top-5 from 1M f32 values; result matches PyTorch. Tagged `P17.9`.

## 17.10 Combine (S)
- `cat`, `stack`, `split`, `chunk`, `repeat`, `repeat_interleave`.
- **Witness**: `combine_smoke.aether` round-trips every combine through its inverse. Tagged `P17.10`.

## 17.11 Mask helpers (S)
- `tril`, `triu`, `eye`, `arange`, `zeros`, `ones`, `full`, `randn` (PCG64 by default).
- **Witness**: `causal_mask.aether` builds a 1024×1024 causal mask; pattern matches `tril(ones(1024, 1024))`. Tagged `P17.11`.

## 17.12 Embedding extras (S)
- `embedding_bag` (sum/mean of multiple indices into a single output).
- Sparse embedding for huge vocab.
- **Witness**: `embedding_bag.aether` — 4-token bag for 50k-vocab embedding. Tagged `P17.12`.

## 17.13 Attention specials (L)
- RoPE (rotary positional encoding), ALiBi (linear bias), FlashAttention v2 (memory-efficient causal), PagedAttention.
- **Witness**: 8k-context Llama forward matches HF transformers within 1e-3 rel. Tagged `P17.13`.

## 17.14 Quantization (L)
- GGUF reader/writer.
- Schemes: `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` (5+ HF GGUF scheme set).
- Fused dequant matmul (single-pass tile dequant).
- AWQ + GPTQ inference paths.
- INT8 QAT (training with quant simulation).
- **Witness**: Llama-2-7B Q4_K_M loads from HF GGUF + inferences on the 3070 Ti at >40 tok/s. Tagged `P17.14`.

## 17.15 SafeTensors (S)
- Reader/writer (small open spec).
- Memory-mapped load (zero-copy weight init).
- HF Hub layout-compatible.
- **Witness**: `safetensors_roundtrip.aether` writes 3 tensors → reads back → bytes-equal. Tagged `P17.15`.

## 17.16 Loss functions full (S)
- MSE, MAE, BCE, BCEWithLogits, KL divergence, Triplet, Contrastive, Huber, Smooth-L1. Have: cross-entropy.
- Each gradient-checked vs finite-diff.
- **Witness**: `loss_<name>.aether` per loss; finite-diff matches 1e-4. Tagged `P17.16`.

## 17.17 Optimizers + schedulers full (M)
- Optimizers: SGD-momentum, RMSprop, Adagrad, Adamax, Lion, Lamb, Adafactor (have AdamW).
- Schedulers: StepLR, CosineAnnealingLR, OneCycleLR, ReduceOnPlateau, warmup wrappers.
- **Witness**: `cuda_train_transformer_block.aether` with cosine + warmup beats constant-LR loss at step 100. Tagged `P17.17`.

## 17.18 Layer modules + initializers full (M, depends 16.5)
- Layers: `Conv{1,2,3}d`, `ConvTranspose2d`, `BatchNorm{1,2,3}d`, `GroupNorm`, `RMSNorm`, `Embedding`, `Dropout`, `MultiheadAttention`, `TransformerEncoder/Decoder`, `LSTM`, `GRU`, `RNN`. Have: `Linear`, `LayerNorm`.
- Init: Kaiming-{normal,uniform}, Xavier-{normal,uniform}, Orthogonal, Truncated-normal.
- **Witness**: 12-layer transformer encoder defined as `let layers: Vec<Block>;` trains synthetic data through one .aether file. Tagged `P17.18`.

## 17.19 Reference architectures ported (XL, depends 17.18 + 17.15)
- ResNet (CV).
- Vision Transformer (ViT).
- Llama-class transformer (decoder-only).
- BERT-class transformer (encoder).
- Diffusion U-Net (Stable Diffusion-class).
- Mamba (selective state-space).
- MoE routing (Switch Transformer).
- CLIP (text + image dual encoder).
- **Witness**: each model has `examples/<model>.aether` that loads weights from SafeTensors + matches HF reference within 1e-3. Tagged `P17.19`.

## 17.20 Numerical parity bench (S)
- `bench/parity/run_all.ps1` runs every op against PyTorch + Candle reference, measures rel + abs error.
- Per-op pass/fail at 1e-5 rel.
- **Witness**: 100% of `runtime/src/lib.rs` ops pass parity. Tagged `P17.20`.

---

# Phase 18 — Distributed training

## 18.1 Own NCCL bindings (M)
- Drop `cudarc-nccl`; bind directly to libnccl.so / nccl.dll.
- Surface mirrors NCCL API exactly.
- **Witness**: 2-rank single-host all-reduce sums correctly across both GPUs (3070 Ti + a borrowed second card). Tagged `P18.1`.

## 18.2 Collectives full (M, depends 18.1)
- `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast`, `send`, `recv`, `all_to_all`.
- Ring + tree + double-binary algorithms.
- **Witness**: `tests/distributed/coll_smoke.sh` runs each collective on 2 ranks, validates output. Tagged `P18.2`.

## 18.3 DDP (data parallel) (M, depends 18.2)
- Bucketed gradient all-reduce overlapped with backward.
- Hook insertion at backward pass.
- **Witness**: 2-GPU DDP training scales >1.7× single-GPU on `cuda_train_transformer_block.aether`. Tagged `P18.3`.

## 18.4 FSDP (fully-sharded data parallel) (L, depends 18.2)
- Shard params + grads + optim state across ranks.
- Forward all-gather of params per layer, backward reduce-scatter of grads.
- **Witness**: 2-GPU FSDP trains a 2× larger model than DDP could fit; loss matches DDP at half precision. Tagged `P18.4`.

## 18.5 TP (tensor parallel, Megatron-style) (L, depends 18.2)
- Column-parallel + row-parallel `Linear`.
- Sequence parallel residual + layer-norm.
- **Witness**: 2-GPU TP-shard a Llama block; output matches single-GPU. Tagged `P18.5`.

## 18.6 PP (pipeline parallel) (L, depends 18.2)
- Micro-batch interleaving (1F1B or interleaved 1F1B).
- Activation checkpointing per stage.
- **Witness**: 2-stage PP trains a 4-block transformer; throughput within 70% of single-GPU. Tagged `P18.6`.

## 18.7 ZeRO-1/2/3 staged sharding (L, depends 18.4)
- Z1: shard optim state.
- Z2: + grad shard.
- Z3: + param shard (= FSDP).
- Toggle via flag.
- **Witness**: each stage shows the documented memory savings on `cuda_train_transformer_block.aether`. Tagged `P18.7`.

## 18.8 Compute/comm overlap (M)
- CUDA streams for collectives concurrent with kernel launches.
- Pipelined H2D/D2H copies.
- **Witness**: nsight trace shows ≥80% comm hidden behind compute. Tagged `P18.8`.

## 18.9 Gradient compression (M, depends 18.3)
- PowerSGD-class low-rank approx.
- Per-bucket compression toggle.
- **Witness**: 8-bit gradient compression reduces all-reduce bytes by 4× with ≤5% loss-curve degradation. Tagged `P18.9`.

## 18.10 Multi-host RDMA (XL, depends 18.2)
- InfiniBand verbs + RoCE Ethernet support.
- GPU-Direct RDMA for nvlink-less hosts.
- **Witness**: 2-host 4-GPU training over 100Gbps Ethernet; throughput scales >3× single-host. Tagged `P18.10` (deferred until hardware available).

## 18.11 8-GPU Llama-7B training (the witness for the phase) (XL)
- Combines DDP + ZeRO-2 + activation checkpointing.
- **Witness**: 8-GPU cluster trains Llama-7B with measured throughput; throughput scales >7× single-GPU. Tagged `P18.11` (deferred until hardware available).

---

# Phase 19 — Serving stack (vLLM-class)

## 19.1 Own TLS 1.3 stack (XL)
- Pure-Aether implementation OR thin BoringSSL wrapper. Pure path preferred for the zero-deps mandate.
- ChaCha20-Poly1305 + AES-GCM + Ed25519 + X25519 + HMAC-SHA256.
- `aether::tls::Connector` + `aether::tls::Acceptor`.
- **Witness**: `tests/runtime/tls_handshake.aether` connects to `https://example.com` and reads the index page. Tagged `P19.1`.

## 19.2 HTTP/1.1 + HTTP/2 + HTTPS server (L, depends 19.1 + 16.12)
- `aether::http::Server::bind(":8080").serve(handler)`.
- Headers, body streaming, chunked transfer, keep-alive.
- HTTP/2 via h2o-style state machine.
- **Witness**: `bench/http_echo/` shows ≥10k req/s on the 11900K. Tagged `P19.2`.

## 19.3 OpenAI-compatible /v1/chat/completions (M, depends 19.2)
- Path: `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/models`.
- Streaming SSE responses.
- Max-tokens, temperature, top-p, frequency penalty.
- **Witness**: `curl http://localhost:8080/v1/chat/completions ...` matches OpenAI's API surface byte-for-byte. Tagged `P19.3`.

## 19.4 KV cache (paged, à la vLLM) (L)
- Block-allocated GPU memory with virtual-page mapping.
- Block size = 16 tokens (configurable).
- Eviction: LRU on full cache.
- **Witness**: 32-batch concurrent inference reuses overlapping prefixes; cache hit rate ≥80% on benchmark prompts. Tagged `P19.4`.

## 19.5 Continuous batching scheduler (L, depends 19.4)
- New requests enter the batch mid-decode (no padding waste).
- Eviction policy (preempt longest-running on full).
- **Witness**: 64 concurrent requests achieve ≥3× single-stream throughput. Tagged `P19.5`.

## 19.6 Speculative decoding (M, depends 17.19)
- Draft + verify model pair.
- Draft model 1B params, verify model 7B+.
- Acceptance rate target ≥40%.
- **Witness**: speculative-on Llama-7B outputs identical to greedy decode at 1.5-2× faster. Tagged `P19.6`.

## 19.7 Multi-model concurrent hosting (M)
- Single process hosts ≥3 models, each with its own KV cache pool.
- Per-model GPU memory budget.
- **Witness**: simultaneously serve Llama-1B + a 250M-class model + an embedding model on the 3070 Ti. Tagged `P19.7`.

## 19.8 gRPC + WebSocket (M, depends 19.2)
- Tonic-style codegen from `.proto` files.
- WebSocket upgrade + framing.
- **Witness**: a gRPC inference call streams 100 tokens with the right framing. Tagged `P19.8`.

## 19.9 Tokenizer parity with HF (M)
- BPE (GPT-2/Llama-style), sentencepiece (Llama 1/2-style), tiktoken (cl100k).
- Loadable from `tokenizer.json`.
- **Witness**: `tokenizer_parity.aether` round-trips 1 MB of WikiText through both Aether + HF tokenizers; bytes-equal. Tagged `P19.9`.

## 19.10 Prompt template engine (S, depends 16.14)
- Jinja-equivalent: `{{ }}`, `{% for %}`, `{% if %}`.
- Loads from `chat_template.jinja` in the model dir.
- **Witness**: chat template renders Llama-3 / GPT-4 / Mistral with correct turn boundaries. Tagged `P19.10`.

## 19.11 Tool / function calling (M, depends 19.3)
- OpenAI-spec tool message shape.
- Constrained decoding to JSON schema.
- **Witness**: a tool call to `get_weather(city)` round-trips with the right JSON args. Tagged `P19.11`.

## 19.12 Vision input (L, depends 17.19)
- Image preproc (resize, normalize, patchify).
- ViT encoder hookup.
- Multi-image batching.
- **Witness**: serve a vision-language model that captions an image at >5 caps/s. Tagged `P19.12`.

## 19.13 Speech input (Whisper-class) (L, depends 17.19)
- 16 kHz PCM → mel spectrogram.
- Whisper encoder.
- **Witness**: 30-second clip transcribes within 1 word of the reference. Tagged `P19.13`.

## 19.14 Auth + rate limiting (S, depends 19.2)
- API key, JWT.
- Token-bucket per user.
- **Witness**: rate-limit kicks in at 10 req/s for a single key; 429 response. Tagged `P19.14`.

## 19.15 Observability (M, depends 19.2)
- Prometheus `/metrics` endpoint.
- OTLP traces (gRPC export).
- Structured JSON logs.
- **Witness**: a 1-minute load test produces meaningful spans + metrics in Grafana. Tagged `P19.15`.

## 19.16 Llama-3-1B at >100 tok/s aggregate (the phase witness) (M)
- Real serving on the 3070 Ti.
- **Witness**: `BENCH_LEDGER.md` row showing ≥100 tok/s sustained over 1000 batched requests. Tagged `P19.16`.

---

# Phase 20 — Self-host the toolchain (drop Rust completely)

## 20.1 Self-hosted lexer (M)
- Extends v3 deposit 10 (`aetherc_self_emit_asm.aether`) into a full lexer with every token shape the Rust-aetherc lexer recognises.
- Output format matches `compiler/src/lexer/mod.rs::Token` byte-for-byte.
- **Witness**: lex `examples/aether_lm.aether` and `compiler/src/main.rs.aether` (translated); tokens exactly match the Rust-aetherc dump. Tagged `P20.1`.

## 20.2 Self-hosted parser (L, depends 20.1)
- Recursive-descent builder of `ast::Program` shape.
- Handles every item / expr / pattern from the Rust-aetherc parser.
- **Witness**: parse + re-emit the AST text for the same files; match the Rust-aetherc dump. Tagged `P20.2`.

## 20.3 Self-hosted MIR + autodiff pass (L, depends 20.2)
- Tape-based reverse mode.
- Symbolic partials for every primitive op.
- **Witness**: MIR text-emit for `aether_lm.aether` matches Rust-aetherc byte-for-byte. Tagged `P20.3`.

## 20.4 Self-hosted asm emitter (XL, depends 20.3)
- AT&T x86-64 emit per current `compiler/src/codegen/asm/mod.rs`.
- Scaffold modules wired (SSA + opt + regalloc + vectorize).
- **Witness**: asm emit for the entire `tests/runtime/*.aether` set matches Rust-aetherc byte-for-byte. Tagged `P20.4`.

## 20.5 Self-hosted runtime (CPU bodies) (L, depends 20.4)
- Every `aether_op_*` reimplemented in Aether.
- Outputs match Rust runtime within 1 ULP.
- **Witness**: `aether_lm.aether` trains identically through Aether-only runtime. Tagged `P20.5`.

## 20.6 Self-hosted trainer (M, depends 20.5)
- Loop driver + checkpointing + sampling.
- **Witness**: end-to-end training run identical curve to Rust-trainer. Tagged `P20.6`.

## 20.7 Self-hosted assembler (L, depends 20.4)
- x86-64 encoder + COFF + PE32+ + ELF writers.
- **Witness**: `aether_asm.aether` produces byte-identical .obj + .exe to Rust `aether_asm`. Tagged `P20.7`.

## 20.8 3-stage bootstrap (S, depends 20.4 + 20.7)
- Stage 0: Rust-aetherc.
- Stage 1: Stage 0 compiles Aether-aetherc → A1.
- Stage 2: A1 compiles Aether-aetherc → A2.
- Stage 3: A2 compiles Aether-aetherc → A3.
- A2 == A3 byte-for-byte (fix-point of the bootstrap).
- **Witness**: `scripts/bootstrap.ps1` produces A2 == A3. Tagged `P20.8`.

## 20.9 Drop Rust dep from CLAUDE.md / SPEC.md (S, depends 20.8)
- Update language: "Aether is implemented in Aether." Phase 5 is closed.
- **Witness**: `git grep "Rust"` in the canonical docs returns only historical context. Tagged `P20.9`.

## 20.10 Bootstrap CI (S, depends 20.8)
- Local CI script (no GitHub Actions per Matt's policy) runs `bootstrap.ps1` on every commit touching `compiler/`, `aether_asm/`, or `runtime/`.
- **Witness**: 50 consecutive commits keep A2 == A3. Tagged `P20.10`.

---

# Phase 21 — Multi-platform

## 21.1 ELF writer (Linux) (M)
- ELF64 header + sections + relocations + dynamic linker support.
- Symbol resolution via `.dynsym` + `.dynstr`.
- **Witness**: `examples/00_hello.aether` compiled on Linux runs + exits 0. Tagged `P21.1`.

## 21.2 Mach-O writer (macOS) (M)
- Mach-O 64 header + load commands + LC_DYLD_INFO.
- Apple Silicon page-protection quirks.
- **Witness**: same on macOS ARM64. Tagged `P21.2`.

## 21.3 ARM64 instruction encoder (L)
- Encoding + COFF/Mach-O/ELF rerouting.
- ARM64 NEON vector ops for vectorize phase.
- **Witness**: `aether_lm.aether` compiles + trains on Apple M2. Tagged `P21.3`.

## 21.4 ROCm runtime (AMD GPUs) (XL, depends 18.2)
- Replace cuBLAS calls with rocBLAS, cuDNN with MIOpen.
- HIP shim layer.
- **Witness**: Llama inference at >50% of CUDA throughput on a borrowed Radeon. Tagged `P21.4` (deferred until hardware).

## 21.5 Metal Performance Shaders (Apple GPU) (L, depends 21.3)
- MPS bindings for matmul/conv/attention.
- **Witness**: Llama-1B inference at >100 tok/s on M2 Pro. Tagged `P21.5` (deferred until hardware).

## 21.6 WebAssembly target (L)
- WASM core + SIMD128 backend for vectorize.
- Bundle as `.wasm` + JS glue.
- **Witness**: `examples/aether_lm.aether` runs in a browser at ≥10% native speed. Tagged `P21.6`.

## 21.7 no_std + embedded (M)
- `panic=abort`, custom alloc trait, no `std::sync`.
- `runtime_pe`-style trimmed surface.
- **Witness**: matmul micro-kernel runs on a Raspberry Pi 4 with `aetherc --target=arm64-linux-musl`. Tagged `P21.7`.

## 21.8 Mobile export (CoreML / NNAPI) (XL)
- Lower MIR to CoreML protobuf / NNAPI HAL.
- **Witness**: Stable Diffusion 1.5 runs on iPhone 15 in <30 s/image. Tagged `P21.8` (deferred).

## 21.9 RISC-V instruction encoder (L)
- RV64GC + V extension for SIMD.
- **Witness**: `examples/00_hello.aether` runs on a SiFive HiFive Unmatched. Tagged `P21.9` (deferred until hardware).

## 21.10 Cross-compile matrix (S, depends 21.1 + 21.2 + 21.3)
- `aetherc --target=<triple>` produces a runnable artefact for every supported triple.
- **Witness**: `scripts/cross.ps1` builds + tests on Linux x86_64, Linux ARM64, macOS ARM64, Windows x86_64 (the four core triples). Tagged `P21.10`.

---

# Phase 22 — Compiler tooling (developer experience parity)

## 22.1 LSP server (`aether-lsp`) (L)
- Completion (context-aware), hover (type), goto-def, find-references, rename, signature help, diagnostics.
- VS Code + Helix + Neovim clients.
- **Witness**: `editor_smoke.test` clicks through goto-def + hover on `examples/aether_lm.aether`. Tagged `P22.1`.

## 22.2 DAP server (`aether-dap`) (M)
- Breakpoints (line + conditional), step over/in/out, eval expression, variable inspection.
- Source maps from the asm backend.
- **Witness**: stepping through `train_step` shows variable values matching expectation. Tagged `P22.2`.

## 22.3 `aetherfmt` (S)
- Deterministic formatter (rustfmt-eq).
- Single round-trip stable.
- **Witness**: `aetherfmt --check tests/runtime/*.aether` returns no diffs. Tagged `P22.3`.

## 22.4 `aetherclippy` (M)
- Style + correctness lints.
- ~50 lints to start (unused let, `let _ = …;` for must-use, redundant clone, etc.).
- **Witness**: `aetherclippy tests/runtime/*.aether` produces zero false positives. Tagged `P22.4`.

## 22.5 `aetherdoc` (M)
- Generate HTML docs from doc comments.
- Cross-link types, fns, traits.
- Search index.
- **Witness**: `aetherdoc --output target/doc` produces a navigable site mirroring rustdoc layout. Tagged `P22.5`.

## 22.6 Coverage (line + branch) (M)
- Instrument every basic block; emit per-bb counters at exit.
- HTML report.
- **Witness**: `aether-cov tests/runtime` produces an HTML report identifying covered + uncovered regions. Tagged `P22.6`.

## 22.7 Fuzzing (libafl-eq) (L)
- Grammar-aware fuzzer for the parser.
- Coverage-guided exploration.
- **Witness**: 1 hour of fuzzing produces zero panics; corpus reaches ≥80% line coverage. Tagged `P22.7`.

## 22.8 Property-based testing (`#[quickcheck]`) (S, depends 16.9)
- Generators for primitive + struct types.
- Shrinking on failure.
- **Witness**: a `prop_sort_idempotent` quickcheck on `Vec::sort` runs 1000 cases and finds nothing. Tagged `P22.8`.

## 22.9 Differential testing vs Candle/PyTorch (M)
- Same input → same output ± 1e-5 on every op.
- Drives the `bench/parity` numerical-parity bench.
- **Witness**: 200+ ops covered; 0 fails at the 1e-5 threshold. Tagged `P22.9`.

## 22.10 Incremental compilation (M, depends 16.10)
- Per-fn + per-crate fingerprints.
- Touched-fn-only recompile.
- **Witness**: editing one fn in `examples/aether_lm.aether` rebuilds in <300 ms. Tagged `P22.10`.

---

# Phase 23 — AI-assisted synthesis

## 23.1 `#[spec(intent="…")]` LLM synthesis (M)
- v3 ships a file-based gate (`<fn>.spec.aether`). v4 wires an LLM call at compile time when the file is missing.
- Gate: human review on first synthesis; cache after approval.
- **Witness**: synthesis of an `add_two` body matches the hand-written reference; subsequent compiles use the cache. Tagged `P23.1`.

## 23.2 Auto-property generation (M, depends 23.1)
- For each synthesised fn, generate ≥3 property tests (idempotence, totality, preservation).
- **Witness**: each `#[spec]` fn ships with auto-generated quickcheck tests; all pass. Tagged `P23.2`.

## 23.3 Auto-test generation (M, depends 23.1)
- Round-trip + edge-case tests synthesised from the fn signature.
- **Witness**: `#[spec] fn parse(s: &str) -> Result<Foo, Error>` synthesis ships with 5 round-trip tests covering empty, malformed, oversized, valid, and unicode inputs. Tagged `P23.3`.

## 23.4 `#[infer]` compile-time numerical inference (M)
- For const-shape fns over const inputs, evaluate at compile time.
- Bake result as `.rdata` constant.
- **Witness**: a 256-element lookup table for `gelu` gets baked into `.rdata`; runtime `gelu_lut` uses it. Tagged `P23.4`.

## 23.5 Differential synthesis (L, depends 22.9)
- Find inputs where Aether vs PyTorch differ by >1 ULP.
- Drive towards 1-ULP parity via small kernel adjustments.
- **Witness**: differential-synth run on `softmax` discovers + closes a 2-ULP gap. Tagged `P23.5`.

## 23.6 Synthesis demo (the witness for the phase) (S, depends 23.1)
- A 5-fn synthesised module passes its auto-tests + shadows a hand-written reference within 1e-5 rel.
- **Witness**: `examples/synth_demo.aether`. Tagged `P23.6`.

---

# Phase 24 — Production hardening

## 24.1 Sanitizers (ASan / MSan / UBSan / TSan) (M)
- Instrumentation pass + runtime check fns.
- `--sanitize=address|memory|undefined|thread`.
- **Witness**: a known double-free triggers ASan with the right backtrace. Tagged `P24.1`.

## 24.2 Reproducible builds (S)
- Deterministic timestamps, no path leakage in .obj/.exe.
- `aetherc --reproducible`.
- **Witness**: same source + same flags → byte-identical .exe across two machines. Tagged `P24.2`.

## 24.3 Supply-chain (signed packages, SBOM) (M)
- Sigstore-shaped signing for `aether-pkg` registry uploads.
- SBOM in CycloneDX format.
- **Witness**: `aether-pkg verify` accepts signed packages, rejects unsigned. Tagged `P24.3`.

## 24.4 Cross-compilation (S, depends 21.10)
- See P21.10 — same goal.

## 24.5 Embedded runtime (M, depends 21.7)
- Trimmed `runtime_pe`-style cdylib.
- Custom alloc trait.
- **Witness**: `aether_lm.aether` minimal subset on an STM32-class MCU (deferred). Tagged `P24.5`.

## 24.6 Hot-reload (M)
- Edit-and-continue for serving processes.
- New code links into the running process; in-flight requests finish on old code, new ones use new code.
- **Witness**: edit a chat template, recompile, no dropped connections. Tagged `P24.6`.

## 24.7 Crash dumps + telemetry (own) (M)
- On panic, dump core + register state + backtrace.
- Optional remote telemetry (own server, no Sentry per Matt's policy).
- **Witness**: a forced panic produces a readable dump file. Tagged `P24.7`.

## 24.8 Real autoscaler for serving fleet (M, depends 19.16)
- Watches QPS; spins up additional replicas; load-balances.
- **Witness**: synthetic load doubles QPS; replica count doubles within 30 s. Tagged `P24.8`.

## 24.9 GPU memory leak detection (S)
- Per-allocation tracking; report unfreed allocations at exit.
- **Witness**: synthetic leak fires the warning with the alloc backtrace. Tagged `P24.9`.

## 24.10 OOM killer + graceful degradation (S, depends 19.4)
- Serving process under memory pressure: shrink KV cache pool, reject new requests with 503.
- **Witness**: under simulated OOM, no crash + 503 returned promptly. Tagged `P24.10`.

---

## Done criteria for v4

1. **Audit ≥150/150** roadmap-tagged witnesses pass.
2. **Honesty scan green** — no `todo!()`, `unimplemented!()`, or stub returns added by v4.
3. **Workspace tests** pass (cargo test or self-hosted equivalent).
4. **`docs/COVERAGE_MATRIX.md`** — every (op, dtype, device) cell needed for the reference models is `✓`.
5. **`docs/BENCH_LEDGER.md` ≥30 rows** — five hand-asm-vs-Aether comparisons within 1%; per-arch profiles documented.
6. **Self-host bootstrap** — A2 == A3 fix-point.
7. **Reference models** — ResNet, ViT, Llama, BERT, SD, Mamba, MoE, CLIP each have a passing `examples/<model>.aether` against the HF reference.
8. **Distributed** — DDP + FSDP + TP + PP all witness 2-rank training (single-host on the 3070 Ti or borrowed second card).
9. **Serving** — Llama-3-1B sustained ≥100 tok/s aggregate; OpenAI-compatible endpoint.
10. **Multi-platform** — at minimum: Linux x86_64, Linux ARM64 (where possible), macOS ARM64, Windows x86_64.

## Suggested execution order

The order keeps the audit monotone + matches dependency edges:

1. **P15.1 → P15.2 → P15.3 → P15.10** — make the asm fast first; the language/stdlib lifts that follow can lean on it.
2. **P16.1 → P16.2 → P16.4 → P16.5** — type system + closures + heap stdlib unlock everything in P17/P18/P19.
3. **P16.8 + P16.9 + P16.17** — macros + proc macros + tests in one push (they're symbiotic).
4. **P16.22 + P16.23 + P16.12** — async + concurrency + I/O for the serving stack.
5. **P17.1 → P17.2 → P17.3 → P17.6 → P17.13 → P17.14 → P17.19** — tensor stack lift, ending in reference models loaded from SafeTensors.
6. **P19.1 → P19.2 → P19.3 → P19.4 → P19.5 → P19.16** — serving stack culminating in the Llama-1B witness.
7. **P18.1 → P18.2 → P18.3 → P18.4** — distributed; depends on serving stack stability for testing rigs.
8. **P20.1 → … → P20.8** — self-host, AFTER P15-19 freezes the language/runtime semantics.
9. **P21.x + P22.x + P23.x + P24.x** — multi-platform, tooling, synthesis, hardening interleaved through everything.

History calibration: v3 priced 18 L/XL items, landed in one session. v4 prices ≈90 items at S/M/L/XL upper bounds; honest ≈3-5× faster. The ML-stack rows (P17.x) are the densest; the language rows (P16.x) compress hard on a focused weekend; the perf claims (P15.x, P25 implicit in P15.10) need real bench discipline.

## How v4 closes

When `target/debug/aether-audit.exe --only roadmap` reports ≥150/150 across phases 6-24, AND `bench/handasm/run_all.ps1` shows ≤1% gap on all five named kernels, AND `scripts/bootstrap.ps1` produces A2 == A3 — Aether ships. The next roadmap (v5) is the post-launch quality + ecosystem arc; v4 is the line that closes the engineering goal stated at the top of CLAUDE.md.
