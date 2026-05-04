# Aether Roadmap v2 — to Candle/PyTorch/Rust parity, zero deps, 1%-of-asm

**Provenance**: written 2026-05-03, after the original 28-item critical path was completed in CLAUDE.md (single-block transformer trains end-to-end on 3070 Ti, self-hosted toolchain, kernel fusion, baby self-hosted lex+parse+JIT in .aether). This doc lays out the 5 mega-phases (6 through 10) needed to reach the goal.

> **Goal**: Aether matches every shipping feature of (Candle ∪ PyTorch ∪ Rust), depends on **only** the OS kernel + the GPU driver, and the compiled native code lands within 1% of hand-tuned x86-64 / PTX assembly on the kernels that matter.

## Effort scale

- **S** = 1 session (≤1 day of focused push)
- **M** = several sessions (≤1 week)
- **L** = multi-week (1–4 weeks)
- **XL** = month-plus arc

History from this repo: I systematically underestimated. "Months" arcs (#27 self-host) shipped 10 bootstrap deposits in one evening. Treat S/M/L/XL as upper bounds with honest median 3-5× faster than priced.

## Cross-cutting rules

1. **Audit first**. Every item ships with a test under `tests/runtime/` or `tests/aether/positive` that exits 0. Audit count must monotonically grow; never go red. Currently **74/74 clean**.
2. **Honesty scan green**. No new `todo!()`/`unimplemented!()`/`unreachable!()` allowed. Stubs use spec-mode (#28) with a recorded request file.
3. **Bench every item that has a perf claim**. Reuse `bench/matmul_micro/run_all.ps1` infra; add a sibling `bench/<feature>/` with the same Aether-vs-Candle-vs-PyTorch discipline.
4. **Self-host bootstrap path is the primary client**. Each new feature SHOULD have a witness test that's compilable by the eventual self-hosted aetherc — not just by the Rust-aetherc.

---

# Phase 6 — Rust language parity

The compiler today is "Rust-subset + ML niceties". To match Rust we need the type system, error model, capturing closures, stdlib, concurrency, async, macros, and a Cargo equivalent. **Order matters here** because the rest of the roadmap (especially Phase 7's tensor stdlib) needs heap collections + traits + Result.

## 6.1 Hindley-Milner type inference (L)
- `let x = foo();` infers from the rhs's type without annotation.
- Generic param inference at call sites (today: const generics infer from tensor shapes; this extends to type generics).
- Implementation: Algorithm W, unification table, occurs check.
- Witness: every existing test in `tests/runtime/` re-compiles after stripping every `: type` annotation that's redundant.
- **Done criterion**: ≥80% of `let` annotations in the existing test suite become removable without code change.

## 6.2 Trait system (XL)
- Trait declarations (`trait Foo { fn bar(&self) -> i32; }`).
- Default methods + supertraits (`trait B: A { ... }`).
- Associated types (`type Output;`).
- Trait impls (`impl Foo for T { ... }`).
- Static dispatch via monomorphization (extend the const-generic worklist machinery).
- Trait objects (`dyn Foo`) via vtable layout — fat pointers (data, vtable).
- Lookup table at codegen: trait-method-resolution.
- Witness: re-implement the GPU op surface as `trait Tensor { fn matmul(&self, …) -> Self; ... }` impl'd by f32/f16 backends.

## 6.3 Lifetimes + borrow checker (XL)
- Region inference for `&` / `&mut` references.
- Outlives constraints.
- Drop checker.
- NLL-style flow-sensitive analysis (not the older lexical scope rules).
- Witness: `cuda_train_transformer_block.aether` recompiles unchanged; intentional `&mut` aliasing test fails with a clear error code (AE0200 family).

## 6.4 Data-carrying enum variants (M)
- `enum Option<T> { None, Some(T) }`.
- Tagged-union value layout: `[i32 tag | payload bytes]`. Niche optimization (`None` for `Option<&T>` = null) is later.
- Pattern bindings: `match opt { Some(x) => use(x), None => default }`.
- Generic enums: `Result<T, E>`.
- Witness: stdlib ships `Option`/`Result`; `tests/runtime/option_some.aether` exercises both branches.

## 6.5 `?` operator + early return (S, depends 6.4)
- Desugar `expr?` to `match expr { Ok(v) => v, Err(e) => return Err(e.into()) }`.
- `From` trait for the `.into()` (depends 6.2).
- Witness: a fn returning `Result<i64, ParseError>` with `?` chains parses 5 numbers from a string and propagates the first error.

## 6.6 Closures with captures (L)
- Capture analysis: classify each free var as `Fn`/`FnMut`/`FnOnce`.
- Synthesize anonymous struct holding captured values + impl `FnXxx`.
- Indirect call ABI: env ptr in rcx, args shift right.
- Witness: `let acc = 0; let inc = |x| { acc + x }; inc(5)` returns 5; mutable capture via FnMut returns 6 on second call.

## 6.7 Heap-allocated stdlib types (L)
- **Allocator**: `aether_alloc_bytes` already exists; add `aether_realloc_bytes` for Vec growth.
- **`Box<T>`**: `Box::new(x)` allocates + moves into heap; auto-drop on scope end.
- **`Vec<T>`**: dynamic array with capacity-doubling growth.
- **`String`**: UTF-8 owned; `&str` view.
- **`HashMap<K, V>`**: open-addressing or chaining; FxHash by default.
- **`Rc<T>`** + **`Arc<T>`**: refcounted (atomic for Arc).
- **`RefCell<T>`**: runtime borrow check.
- Witness: `let mut v = Vec::new(); for i in 0..1000 { v.push(i); } v[42]` returns 42.

## 6.8 Iterator trait + adapters (M, depends 6.2 + 6.6)
- `trait Iterator { type Item; fn next(&mut self) -> Option<Self::Item>; }`.
- Adapters: `map`, `filter`, `fold`, `take`, `skip`, `chain`, `zip`, `enumerate`, `collect`.
- `for x in iter` desugar.
- Witness: `(0..100).filter(|i| i % 3 == 0).sum::<i64>()` returns 1683.

## 6.9 Concurrency (L)
- **Atomics**: `AtomicI64`, `AtomicBool`, `compare_exchange`, `fetch_add` via `lock cmpxchg`/`lock xadd`.
- **Mutex / RwLock**: spinlock primitives + `parking_lot`-style fast path.
- **Threads**: `thread::spawn(closure)` via Win32 `CreateThread` / Linux `clone(SIGCHLD)`.
- Witness: 8-thread parallel matmul reaches >6× single-thread on the 11900K.

## 6.10 async/await + Future trait (XL, depends 6.2 + 6.6 + 6.9)
- `trait Future { type Output; fn poll(...) -> Poll<Self::Output>; }`.
- `async fn` → state-machine struct transform (continuations as enum variants).
- Executor: thread pool + epoll/IOCP.
- `await` desugar.
- Witness: 1000 concurrent `aether_read_file` async tasks complete in <1 ms wall.

## 6.11 Macros (L)
- **`macro_rules!`**: token-tree pattern matching + substitution. Reuse the self-hosted lexer's token model (#27 step 5).
- **Attribute macros**: extension hook for the existing `#[autodiff]` / `#[server]` / `#[spec]` attrs.
- **Derive macros**: `#[derive(Debug, Clone, PartialEq)]`.
- Witness: a user-defined `vec![1, 2, 3]` macro lowers to `Vec::from_iter([1, 2, 3])`.

## 6.12 Cargo equivalent (M)
- Manifest format (`Aether.toml`, modeled on Cargo.toml).
- Dependency resolver (registry + git + path).
- Workspaces, features, build scripts (`build.rs` equivalent in Aether).
- Build cache: per-crate fingerprinting + incremental compilation.
- Witness: `aetherc build` compiles a 5-crate workspace incrementally; touching one crate triggers only that crate's rebuild.

## 6.13 Standard I/O (M)
- **Filesystem**: `File`, `BufReader`, `fs::read`, `fs::write`, `fs::read_dir`, glob, mtime.
- **Network**: `TcpStream`, `TcpListener`, `UdpSocket`. Rustls-shaped TLS later.
- **Process**: spawn, pipe, exit code.
- Witness: a 50-line HTTP/1.1 echo server in Aether handles 10k req/s.

## 6.14 Test framework (S)
- `#[test]` attribute; aetherc emits a small runner harness.
- `assert_eq!`, `assert_ne!`, `assert!` macros (depends 6.11).
- `cargo test`-equivalent runner.
- Witness: existing `tests/runtime/*.aether` migrate to `#[test]` form; audit script unchanged.

**Phase 6 done**: a non-trivial Rust crate (e.g. `serde_json`'s pure-Rust subset) ports to Aether with mechanical translation only.

---

# Phase 7 — Candle parity (numerical operations)

Candle is ~200 ops + dtype matrix + format support + layer modules + distributed. Today Aether has ~25 GPU ops + f32-only + no quant/SafeTensors/distributed.

## 7.1 Full dtype matrix (M)
- **Add**: `f16` (FP16, IEEE half), `bf16` (Brain Float 16, wider exponent).
- **Add**: `i8`, `u8`, `i16`, `u16`, `u32`, `u64`, `bool` (packed u8).
- Have: `f32`, `f64`, `i32`, `i64`.
- Codegen: SSE2 has no native `f16` math; widen-to-f32 for ops, narrow on store. AVX-512 + `_Float16` intrinsics (Sapphire Rapids+) when available. Use `vcvtph2ps` / `vcvtps2ph`.
- CUDA: native f16 / bf16 via PTX `cvt.f16.f32` + tensor cores.
- Witness: `cuda_train_transformer_block_bf16.aether` trains the same block with bf16 weights and an fp32 master copy, loss within 5% of f32 baseline.

## 7.2 N-dimensional Tensor + strided views (M)
- Arbitrary rank (1D-7D).
- Layout = `(shape, strides, offset)` triple.
- `.transpose(d1, d2)`: swap stride entries (zero-copy).
- `.narrow(d, start, len)`: bump offset, shrink shape entry.
- `.slice([..2, 1.., ::2])`: stride-aware indexing.
- Broadcasting: align trailing dims, expand size-1 dims via stride=0.
- `.reshape(...)`: contiguous-aware fast path; copy fallback otherwise.
- `.contiguous()`: explicit copy to dense layout.
- Witness: ResNet-50's first conv (with `permute` of input channel-last) compiles + matches PyTorch numerically.

## 7.3 The full op surface (L)
Need to add (each is a CUDA kernel + dispatch entry + test):
- **Convolutions**: `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d` (im2col + sgemm OR direct).
- **Pooling**: `max_pool2d`, `avg_pool2d`, adaptive variants.
- **Norms**: `batch_norm`, `instance_norm`, `group_norm`, `rms_norm` (have layer_norm).
- **Activations**: `silu/swish`, `tanh`, `sigmoid`, `leaky_relu`, `elu`, `mish`. Each w/ backward.
- **Math**: `log`, `exp`, `sin`, `cos`, `pow`, `sqrt` (have sqrtss), `recip`, `abs`.
- **Reductions**: `sum`, `mean`, `var`, `std`, `min`, `max`, `argmax`, `argmin`, `prod` (per-dim or full).
- **Selection**: `topk`, `sort`, `where`, `masked_fill`, `gather`, `scatter`.
- **Combine**: `cat`, `stack`, `split`, `chunk`, `repeat`, `repeat_interleave`.
- **Mask helpers**: `tril`, `triu`, `eye`, `arange`.
- **Embedding**: `embedding` (gather rows by integer index).
- **Attention specials**: RoPE (rotary positional), ALiBi, FlashAttention v2 (memory-efficient causal attention), PagedAttention.
- Witness: a Llama-3-class 1B-param inference forward compiles + outputs match HF transformers within 1e-3 relative.

## 7.4 Quantization (L)
- **GGUF format reader/writer** (the file structure is open + small).
- Quant schemes: `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0` — at least these 5 cover most ggml-compatible models.
- Dequant fused into matmul kernels (single-pass tile dequant).
- AWQ + GPTQ inference paths.
- INT8 QAT (training with quantization simulation).
- Witness: a quantized Llama-2-7B (Q4_K_M) loads from a HF GGUF file and inferences on the 3070 Ti at >40 tok/s.

## 7.5 SafeTensors (S)
- File format reader/writer (the spec is small).
- Memory-mapped load (zero-copy weight init from disk).
- Compatible with Hugging Face Hub layout.
- Witness: `tests/runtime/safetensors_roundtrip.aether` writes 3 tensors → reads back → bytes-equal.

## 7.6 Loss functions (S)
- MSE, MAE, BCE, BCEWithLogits, KL divergence, Triplet, Contrastive, Huber, Smooth-L1.
- Have: cross-entropy.
- Witness: each loss is a `tests/runtime/loss_<name>.aether` that gradient-checks vs. a finite-difference reference.

## 7.7 Optimizers + LR schedulers (M)
- **Optimizers**: SGD-momentum, RMSprop, Adagrad, Adamax, Lion, Lamb, Adafactor. Have AdamW.
- **Schedulers**: StepLR, CosineAnnealingLR, OneCycleLR, ReduceOnPlateau, warmup wrappers.
- Witness: `cuda_train_transformer_block.aether` with a cosine schedule + warmup beats the constant-LR loss curve at step 100.

## 7.8 Layer modules + initializers (M)
- **Layers**: `Conv1d/2d/3d`, `ConvTranspose2d`, `BatchNorm1d/2d/3d`, `GroupNorm`, `RMSNorm`, `Embedding`, `Dropout`, `MultiheadAttention`, `TransformerEncoder/Decoder`, `LSTM`, `GRU`, `RNN`. Have `Linear`-equivalent + `LayerNorm`.
- **Init**: Kaiming-normal/uniform, Xavier-normal/uniform, Orthogonal, Truncated-normal.
- Witness: a 12-layer transformer encoder defined as `let layers: Vec<Block>;` (depends 6.7) trains on synthetic data through one .aether file.

## 7.9 Distributed (XL)
- **NCCL bindings** (drop cudarc-nccl, write our own or wrap directly).
- Collectives: all-reduce, all-gather, reduce-scatter, broadcast, send/recv, all-to-all.
- **DDP** (data parallel) — bucketed gradient all-reduce overlapped with backward.
- **FSDP** (fully-sharded data parallel) — shard params + grads + optim state across ranks.
- **TP** (tensor parallel, Megatron-style) — column-parallel + row-parallel linear.
- **PP** (pipeline parallel) — micro-batch interleaving.
- Witness: 2-GPU DDP training via 2 processes communicating over NCCL on a single host; throughput scales >1.7×.

**Phase 7 done**: every Candle integration test in `J:\candle-src\candle-core\tests` translates to Aether with mechanical changes only.

---

# Phase 8 — PyTorch parity (training + serving infrastructure)

## 8.1 Eager + traced execution modes (L)
- Eager (have): ops run immediately at the runtime ABI boundary.
- **Trace mode**: capture an op graph during a recording fn-call; replay with optimization (kernel fusion, schedule, layout opt).
- **TorchScript-equivalent**: serializable IR for graphs; load/save.
- Witness: a fn decorated with `#[trace]` produces a `.aether-trace` file; a separate `aetherc trace-run <file>` reproduces the call sequence on stored inputs.

## 8.2 Autograd graph + transforms (L)
- Have: `#[autodiff]` does tape-based reverse mode.
- Add: higher-order gradients (gradient-of-gradient).
- `vmap`: auto-vectorize a fn over a batch dim.
- `jvp` / `vjp` primitives.
- Witness: a fn that computes its own Hessian via `vmap(jvp(f))` on a 100-dim quadratic returns the true Hessian.

## 8.3 DataLoader + Dataset (M)
- `trait Dataset { fn len(&self); fn get(&self, idx) -> Sample; }` (random access).
- `trait IterableDataset { fn iter(&self) -> impl Iterator<Item=Sample> }`.
- `DataLoader { batch_size, shuffle, num_workers, prefetch, pinned_memory }` — fixed-size ring of prefetch threads.
- Stock datasets: MNIST, CIFAR, ImageNet, WikiText, FineWeb (HTTP fetch + cache).
- Witness: MNIST training run from raw `train-images-idx3-ubyte` to >97% test accuracy in 1 epoch.

## 8.4 Reference architectures (L)
- ResNet (CV).
- Vision Transformer (ViT).
- Llama-class transformer (decoder-only).
- BERT-class transformer (encoder).
- Diffusion U-Net (Stable Diffusion-class).
- Mamba (selective state-space).
- MoE routing (Switch Transformer).
- CLIP (text + image dual encoder).
- Witness: each model has a `examples/<model>.aether` that loads weights from a SafeTensors file and produces output matching the HF reference within 1e-3.

## 8.5 Inference + serving (L)
- HTTP server (depends 6.13 network I/O).
- OpenAI-compatible chat completions endpoint.
- KV cache (paged, à la vLLM).
- Continuous batching.
- Speculative decoding (draft + verify model pair).
- Witness: serving Llama-3-1B at >100 tok/s aggregate batch throughput on the 3070 Ti.

## 8.6 Profiling + debugging (M)
- Per-op wall time + GPU time recorder.
- Memory allocator statistics (live alloc, fragmentation, peak).
- Chrome-trace-format export (compatible with Perfetto / chrome://tracing).
- TensorBoard event file writer.
- Witness: a profiler trace for the transformer block training shows >80% time in matmul kernels (i.e. minimal framework overhead).

## 8.7 Mixed precision (M, depends 7.1)
- `autocast` context manager: ops in lower precision (f16/bf16), reductions in f32.
- `GradScaler` for FP16 numerical stability (loss scaling + skip-on-NaN).
- Native bf16 path for Ampere+.
- Witness: `cuda_train_transformer_block_amp.aether` matches f32 baseline within 5% loss at half the memory.

## 8.8 Checkpoint + resume (S, depends 7.5)
- Atomic write (temp file + rename).
- Optimizer state (m, v, step) serialized.
- LR scheduler state.
- DataLoader RNG state.
- Compression option (zstd).
- Witness: kill training mid-epoch; resume produces bit-identical loss trajectory from the next step.

## 8.9 ONNX (M, depends 7.2 + 7.3)
- Op-level translation table (subset of ONNX ops we care about).
- Read-only initially: load an ONNX inference model + run it.
- Export later (more involved due to ONNX's constraints).
- Witness: an ONNX-exported MobileNetV3 from torchvision runs in Aether and matches the original outputs.

## 8.10 Multi-platform (XL)
- **CUDA** (have).
- **ROCm** (AMD GPUs) — replace cuBLAS calls with rocBLAS, reuse the rest.
- **Metal** (Apple Silicon) — MPS-equivalent kernels.
- **Vulkan compute** (cross-vendor GPU).
- **WebGPU** (in-browser inference) — emit WGSL.
- **CPU SIMD**: AVX-512 / AVX2 microkernels for matmul + conv.
- **ARM** (mobile + Apple Silicon CPU): NEON + SVE.
- Witness: `cuda_train_transformer_block.aether` retargets to `--device=metal` on macOS without source changes.

**Phase 8 done**: a real model training pipeline (Llama-3-class 1B-param, real dataset, multi-GPU) runs end-to-end in Aether with throughput within 20% of PyTorch on the same hardware.

---

# Phase 9 — Zero outside dependencies

## 9.1 Self-host the compiler (XL — bootstrap done in #27)
- Continue from the 10-step bootstrap. Expand the `.miniaether` mini-language up to full Aether feature set:
  - if/else, while, for (S each)
  - struct + impl + match (M each, depends Phase 6 type system)
  - Generics + traits (L, depends 6.2)
- Build out: parser → AST → MIR → codegen → asm bytes → COFF → PE32+ writer, all in .aether.
- Witness: `aetherc-self examples/aether_lm.aether` produces a binary equivalent (or better) than `aetherc-rust examples/aether_lm.aether`.

## 9.2 Self-host the assembler + linker (M)
- Currently `aether_asm` is the Rust crate. Reimplement in Aether.
- The PE32+ writer (`aether_asm/src/pe.rs`) translates straightforwardly — bit-twiddling + section-header writes.
- ELF writer for Linux output (new — currently Windows only).
- Mach-O writer for macOS.
- Witness: `aether-asm-self foo.s -o foo.obj` produces byte-identical output to the Rust assembler.

## 9.3 Self-host the runtime (L, depends 6.7 + 6.9)
- Currently `runtime/` is Rust. Move op kernels to .aether (CPU paths first; GPU stays in PTX).
- Stays compatible with both Rust-aetherc and self-host-aetherc.
- Witness: `aether_rt-self.dll` is build-able by self-hosted aetherc and passes all 74 audit tests.

## 9.4 Drop gcc + msvcrt + libc (M)
- The `--emit=pe-bin` path already has zero gcc dep for slim runtime programs.
- Extend pe-bin to support the FULL aether_rt API surface (currently slim subset).
- Drop libc: implement printf-equivalent in Aether (use kernel32 WriteFile / Linux write syscall).
- Witness: `aetherc examples/aether_lm.aether --emit=pe-bin -o aether_lm.exe` produces a binary whose only DLL imports are kernel32 + nvcuda.

## 9.5 Drop cudarc, cuBLAS, cuDNN (XL)
- Write our own CUDA driver-API bindings (`nvcuda.dll` direct calls).
- Hand-tuned matmul kernel (PTX or SASS) — competitive with cuBLAS sgemm.
- Hand-tuned conv kernel.
- Hand-tuned attention kernel (FlashAttention-derived).
- Witness: build with `--no-cudarc --no-cublas`; `cuda_train_transformer_block.aether` still trains, throughput within 5% of cuBLAS path.

## 9.6 Replace libm (M — partial in `runtime_pe`)
- Implement: `sin/cos/tan`, `log/log2/log10`, `exp/exp2`, `pow`, `sqrt` (have via `sqrtss`), `tanh`, `gamma`, `erf`.
- Use minimax polynomial approximations (Sollya / FunctionMaker references).
- Accuracy: within 1 ULP for common ranges, within 4 ULP worst-case.
- Witness: a 1M-element f32 sin sweep matches glibc to within 1 ULP everywhere.

## 9.7 Replace OS-shipped CRT (M)
- Win32: write our own `mainCRTStartup` (have for pe-bin).
- Linux: write `_start` calling `main`, set up argv + envp, call `exit` syscall.
- macOS: same shape via dyld entry.
- Witness: `--emit=pe-bin` + `--emit=elf-bin` + `--emit=macho-bin` all work; no system CRT linked.

**Phase 9 done**: `ldd` (or PE depends-walker) on an Aether binary shows ONLY the OS kernel + GPU driver as dynamic deps. No libc, no msvcrt, no Rust runtime, no cudart.

---

# Phase 10 — 1%-of-hand-written-asm performance

The key insight: today's Aether asm backend emits straightforward "load → spill → load → op → store" sequences. A real optimizing compiler stays in registers, schedules for port pressure, vectorizes loops, and inlines hot paths. Each pass below is a separable item — the existing `mir/fuse.rs` framework gives us the place to slot them in.

## 10.1 SSA-based MIR (L)
- Convert the AST-walking codegen to a real Static-Single-Assignment IR.
- Phi nodes at block joins.
- Dominance + post-dominance trees.
- Use-def chains.
- Witness: existing tests recompile through SSA path; binary size reduced ≥10% (dead-store elim from the conversion).

## 10.2 Optimization passes (L, depends 10.1)
- **Constant folding** — evaluate `2 + 3` at compile time.
- **CSE** (common subexpression elimination).
- **GVN** (global value numbering).
- **DCE** (dead code elimination).
- **DSE** (dead store elimination).
- **Strength reduction** — `x * 8` → `x << 3`.
- **LICM** (loop-invariant code motion).
- **Loop unrolling** (cost-based).
- **TCO** (tail call optimization).
- **Inlining** (cost-based, with inline hints).
- **Function specialization** (cloning + per-callsite constant prop).
- **Devirtualization** (resolve `dyn Trait` calls when concrete type is known).
- Witness: `bench/optfx/` shows each pass independently delivers ≥3% speedup on its target microbench.

## 10.3 Register allocation (L)
- Linear scan (simpler) OR graph coloring (Chaitin-Briggs).
- Live range analysis.
- Spill heuristics (frequency × usage cost).
- Move coalescing.
- Today: every local lives in a stack slot. After RA, hot locals stay in registers across many ops.
- Witness: `cuda_train_transformer_block.aether` recompiles; .obj size shrinks ≥30% (fewer spill loads/stores).

## 10.4 Instruction selection (M)
- Tree-pattern matching (BURG-style or maximal munch).
- Multi-instruction patterns: `lea` for `a + b*scale + disp`, `bts/btr` for bit ops.
- ISA-aware selection: prefer 3-operand AVX over 2-operand SSE.
- Witness: a `let r = a + b * 8 + 17;` source compiles to a single `lea` instead of mul + add + add.

## 10.5 Instruction scheduling (M)
- Port-pressure aware (Skylake / Zen 4 scheduling tables).
- Latency-aware list scheduling.
- Software pipelining for inner loops (mod scheduling).
- Witness: `bench/matmul_micro/` hot-loop achieves ≥85% of hand-tuned Goto/BLIS microkernel throughput.

## 10.6 Vectorization (L)
- Auto-vectorize loops to AVX2 → AVX-512 → NEON / SVE.
- Recognize reduction patterns (`sum`, `dot_product`).
- First-class SIMD intrinsics (`Simd<f32, 16>` lane type — partial in the spec already).
- Loop versioning (scalar fallback for short loops or alignment misses).
- Witness: a hand-written `for i in 0..N { c[i] = a[i] + b[i]; }` matches a hand-coded AVX-512 reference within 5%.

## 10.7 PGO + AutoFDO (M)
- Profile collection: instrumentation OR sampling (perf record-style).
- Feedback-directed inlining (always-inline hot edges, never-inline cold).
- Branch probability annotations.
- Witness: a PGO build of `cuda_train_transformer_block.aether` is ≥10% faster than the static build.

## 10.8 Block layout (S, depends 10.7)
- Hot/cold splitting (cold blocks moved to a separate section).
- Branch reordering for I-cache locality (hot fall-through paths).
- Witness: instruction-fetch perf-counters show ≥20% reduction in I-cache misses.

## 10.9 Whole-program LTO (M)
- Cross-module inlining.
- Specialization across compilation units.
- Whole-program DCE.
- Witness: a multi-crate workspace's final binary is ≥15% smaller than the sum of independent crate compiles.

## 10.10 Microbench-driven kernel selection (M)
- Per-op multiple kernel implementations (tile sizes, vectorization widths).
- First-call probe + pick best for the observed shapes.
- Cache the choice keyed by `(op, dtype, shape, device)`.
- Witness: matmul auto-selects a different inner-tile size for `[8, 8, 8]` vs. `[1024, 1024, 1024]` and beats single-tile by ≥20% across the matrix.

**Phase 10 done**: on a curated set of 20 hot kernels (matmul tiles of various sizes, conv2d, attention, RMSNorm, GELU, AdamW), Aether is within 1% of hand-tuned reference asm. Where it isn't, the gap is documented + a remediation pass is queued.

---

# Suggested ordering / parallelism

The graph below shows the dependency edges that matter. Items inside a phase that don't share an edge can run in parallel.

```
6.1 type inference  ─┐
                     ├─►  6.2 traits  ─┬─►  6.5 ?-op  ─►  6.13 stdio
6.4 data enums  ────┘                  │
                                       └─►  6.8 iterators  ─►  7.8 layer modules
6.6 closures  ─────────────────────────►  6.10 async
6.7 heap stdlib  ──────────────────────►  6.9 concurrency  ─►  6.10 async
                  └────────────────────►  7.2 N-d Tensor  ─┐
                                                           ├─►  7.3 op surface  ─►  7.4 quant + 7.5 SafeTensors
                                                           └─►  8.3 DataLoader

10.1 SSA  ─►  10.2 opt passes  ─►  10.7 PGO  ─►  10.8 block layout
              10.3 reg alloc  ─►  10.5 instr scheduling
              10.6 vectorization

9.1 self-host  ─►  9.2 self-host asm  ─►  9.3 self-host runtime
9.5 drop cudarc  (parallel — independent of self-host)
```

**Recommended attack order** (alternating between user-visible value and infrastructure):

1. **Sprint A (user-visible)**: 6.4 enums → 6.5 ?-op → 6.7 heap stdlib (Vec, String) → 7.5 SafeTensors. Delivers Result/Option ergonomics + checkpoint round-trip.
2. **Sprint B (perf)**: 10.1 SSA + 10.3 reg alloc. Immediate win on every existing test (smaller .obj, fewer spills).
3. **Sprint C (model breadth)**: 7.1 dtype matrix (bf16) + 7.2 strided views + 7.3 op surface (conv2d, RoPE, FlashAttention, embedding).
4. **Sprint D (training infra)**: 6.6 closures with captures → 8.3 DataLoader → 8.4 Llama reference.
5. **Sprint E (concurrency)**: 6.9 atomics/threads → 7.9 distributed (DDP first).
6. **Sprint F (final polish)**: 10.6 vectorization → 10.10 kernel auto-tuning → 9.5 drop cudarc.

Each sprint = 1-3 weeks of concentrated work given current pace. Total roadmap: **6–9 months of focused execution** to reach the goal — calibrated against the historical pattern that "months" estimates land in days when LLM-aided.

# Bench cadence

Standing benches that run on every milestone:

- `bench/matmul_micro/run_all.ps1` — 3-way sgemm (have).
- `bench/conv2d/run_all.ps1` — once 7.3 lands.
- `bench/attention/run_all.ps1` — once 7.3 FlashAttention lands.
- `bench/llama_inference/run_all.ps1` — once 7.4 quant + 8.5 serving land.
- `bench/training_throughput/run_all.ps1` — once 8.3 DataLoader lands.

Every bench produces a row in `docs/BENCH_LEDGER.md` (date / commit / config / numbers / verdict) so regressions are visible.

# Closing

This roadmap is the path to the goal. It's measurable, ordered, and bench-anchored. Each item has a witness test that exits 0 OR a number that beats the comparison.

The original 28-item critical path took ~5 sessions. This is the next 100 items. Keep the audit green and the ledger honest, and we land the goal.
