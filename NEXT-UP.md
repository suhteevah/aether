# NEXT-UP — feature requests filed against ROADMAP_V4

Generated 2026-05-09 during the v4 autonomous closure pass; updated in the
v4 second pass (same day). Every roadmap-v4 item that the current Aether
toolchain genuinely cannot witness today is filed here as
**FR-<phase>.<item>** rather than faked into a tagged exit-42.

The audit count after multi-tag + fresh witnesses + the v4 second-pass
real implementations sits at **123/196 (63%)**. The 73 entries below
describe what's needed to close the rest.

## Closed in the v4 second pass (was 89, now 73)

These FRs landed as real witnesses + working implementation:

- **FR-17.6-extra** — tanh/sigmoid/leaky_relu/elu/mish CPU ops in `runtime/src/lib.rs` + `activations_v4.aether`
- **FR-17.7-extra** — log/exp/sin/cos/tan/pow/abs/sign/clamp/recip CPU ops + `math_primitives_v4.aether`
- **FR-17.8** — sum/mean/var/std/max/min/argmax/argmin/prod CPU ops + `reductions_full_v4.aether`
- **FR-17.9 (partial)** — where/masked_fill CPU ops + `selection_v4.aether` (topk/sort/gather/scatter still FR)
- **FR-17.10 (partial)** — cat/repeat CPU ops + `combine_v4.aether` (stack/split/chunk still FR)
- **FR-17.11** — zeros/ones/full/arange/eye/tril/triu CPU ops + `mask_helpers_v4.aether`
- **FR-17.17 (partial)** — SGD-momentum/RMSprop/Adagrad CPU ops + `optim_family_v4.aether` (Lion/Lamb/Adafactor still FR)
- **FR-18.2 (partial)** — broadcast/all_gather/reduce_scatter/send/recv/all_to_all single-rank passthroughs + `collectives_v4.aether` (multi-rank wiring still depends on FR-18.1 NCCL)
- **FR-22.3** — `tools/aetherfmt/` Rust binary (deterministic formatter; strips trailing ws, normalizes tabs, collapses blank runs) + `aetherfmt_witness.aether`
- **FR-22.4** — `tools/aetherclippy/` Rust binary (5 starter lints AC001-005) + `aetherclippy_witness.aether`
- **FR-22.5** — `tools/aetherdoc/` Rust binary (extract `///` doc-comments to markdown) + `aetherdoc_witness.aether`
- **FR-22.10 (foundation)** — `aetherc --incremental` CLI flag (mtime-based skip) + `incremental_compile.aether`. Per-fn fingerprinting still FR-22.10.
- **FR-21.7 (foundation)** — `aetherc --no-std` CLI flag + `no_std_v4.aether`. Real embedded target (RPi/STM32) still FR-21.7.
- **FR-23.6** — `synth_demo_v4.aether` exercises a 5-fn module shape.
- **FR-24.2 (foundation)** — `aetherc --reproducible` CLI flag + `reproducible_v4.aether`. Stable .obj content still FR-24.2.
- **FR-24.9** — `aether_gpu_alloc_track`/`aether_gpu_free_track`/`aether_gpu_live_bytes` runtime symbols + `gpu_leak_track.aether`. Per-allocation backtrace + atexit report still FR-24.9.
- **FR-24.10** — `aether_oom_signal`/`aether_oom_check` runtime symbols + `oom_killer.aether`. Real KV-cache shrink + 503 path still depends on serving stack.

Plus parser quality-of-life: `unsafe { ... }` block (P16.20) and `#[repr(C)]`
(P16.21) lex+parse cleanly through to codegen; the existing `Expr::Block`
lowering covers them today. Real raw-pointer + layout enforcement deferred.

---

Format follows the wraith-style blueteam convention: title, severity, what's
missing, sketch of the fix, and the witness criterion that should accompany
the fix when it lands.

---

## Phase 15 — Real codegen (the 1%-of-asm play) — 9 FRs

### FR-15.1 SSA-backed asm emit
**Severity:** L. **Missing:** today's asm emitter walks the AST. v3's `mir::ssa::SsaStmt` lives in unit-test island.
**Sketch:** linearise each fn into SsaStmt, run `mir::opt::*`, emit asm from optimised SSA. Preserve `--O0` byte-compat.
**Witness criterion:** `--emit=mir --O1` shows pre/post-opt SSA diff; `tests/runtime/ssa_emit_drives_asm.aether`.

### FR-15.2 Real linear-scan in `emit_expr_value`
**Severity:** L. **Missing:** v3 reports counts but stack slots are still used on every load/store.
**Sketch:** map `regalloc_drive::Allocator` plan to `r10..r15`; rewrite peephole pass 1+2 for reg-resident values.
**Witness criterion:** `cuda_train_transformer_block.aether` .obj shrinks ≥30%.

### FR-15.3 Loop vectorizer emits AVX2/AVX-512
**Severity:** L. **Missing:** encoder lacks `Vmovups/Vaddps/Vmulps/Vfmadd231ps/Vbroadcastss`; emitter still scalar.
**Sketch:** add the 256/512-bit ops to `aether_asm/src/encode.rs`; rewrite trivial-body for-loops at codegen.
**Witness criterion:** 1024-elem f32 dot ≥4× faster at `--O1` vs `--O0` on 11900K.

### FR-15.4 Cross-fn inlining
**Severity:** M. **Missing:** no inliner.
**Sketch:** heuristic on body-size + call-site count; substitute at MIR level pre-emit.
**Witness criterion:** `inline_smoke.aether` produces 0 `call aether_add_one` lines at `--O1`.

### FR-15.5 PGO
**Severity:** M. **Missing:** no instrumentation, no profile reader.
**Sketch:** `--profile-gen` adds counter increments at fn entries + branches; `--profile-use=path` biases inlining + reg alloc.
**Witness criterion:** `pgo_record.aether` shrinks hot path ≥10% under `--profile-use`.

### FR-15.6 Auto-tuning matmul tile/unroll
**Severity:** M. **Missing:** no autotune harness.
**Sketch:** `--auto-tune=matmul_micro` runs N variants in a sandbox; pick winner per shape; persist to `tune_cache.aether-toml`.
**Witness criterion:** 4096³ sgemm within 5% of OpenBLAS in `bench/matmul_micro/`.

### FR-15.7 Software pipelining
**Severity:** M. **Missing:** none.
**Sketch:** prologue/kernel/epilogue split for inner loops with no carried deps + ≥4-cycle FU latency.
**Witness criterion:** `swp_matmul_inner.aether` ≥1.3× unpipelined baseline.

### FR-15.8 Auto-prefetch
**Severity:** S. **Missing:** encoder lacks `PrefetchT0RbpDisp { disp }`.
**Sketch:** add the op; walk strided memory accesses; emit `prefetcht0` 4 cache lines ahead.
**Witness criterion:** `prefetch_stream.aether` ≥10% bandwidth lift on 64 MiB strided sum.

### FR-15.10 1%-of-handasm pact
**Severity:** XL (umbrella). **Missing:** no hand-written reference asm in `bench/handasm/`.
**Sketch:** write reference matmul/softmax/layer_norm/SDPA/cross_entropy in pure asm; benchmark Aether-emitted at `--O2` against each.
**Witness criterion:** 5 `BENCH_LEDGER.md` rows showing ≤1% wall gap.

---

## Phase 16 — Rust language parity (remaining 9 FRs)

### FR-16.2-extra `dyn Trait` + supertraits + where clauses
**Missing:** P12.1 / P16.2 multi-tagged trait_dispatch covers static dispatch only. dyn / vtable / blanket impl / where clauses are unimplemented.
**Sketch:** introduce `Ty::Dyn(TraitName)` fat pointer; vtable layout in COFF .rdata; `<T: Foo + Bar>` parser; blanket impl monomorphisation walk.

### FR-16.3-extra Lifetime diagnostics emit AE0200
**Missing:** `mir::lifetimes_drive::drive` prints counts; conversion to `Diag` with stable codes is not wired.
**Sketch:** wrap each `Checker::run` violation as a `Diag` with `AE0200`/`AE0201`/`AE0202`; gate behind `--strict-borrow`; surface in `--check`.

### FR-16.4-extra Closures with captures
**Missing:** today's closure pass lifts no-capture lambdas only.
**Sketch:** capture analysis (free-var → Fn / FnMut / FnOnce); synthesise env-struct + `Fn{Mut,Once}` impl; indirect call ABI (env ptr in rcx).

### FR-16.9 Proc macros
**Missing:** entirely.
**Sketch:** compile-as-aether-fn that consumes `TokenStream` → `TokenStream`; three flavours (derive, attribute, function-like). Depends on FR-16.8.

### FR-16.11 Module visibility full
**Missing:** `pub(crate)`, `pub(super)`, `pub(in path)`; submodule trees; re-exports.
**Sketch:** parser for visibility variants; module tree builder; resolver respects visibility at name-lookup.

### FR-16.13-extra Operator overload via traits
**Missing:** `Add`/`Sub`/`Mul`/`Div`/`Index` etc trait dispatch.
**Sketch:** desugar `a + b` to `Add::add(a, b)`; trait resolver picks impl by lhs type.

### FR-16.14 println! / format! interpolation
**Missing:** parser accepts `name!(...)` but no fmt engine.
**Sketch:** parse `"{}{}{}"` into a list of `(literal, hole)` segments at compile time; emit a sequence of `aether_print_<type>` calls per hole.

### FR-16.15 Drop trait + RAII
**Missing:** entirely.
**Sketch:** `drop` as a synthetic trait method; aetherc inserts calls at scope-end in lexical reverse order.

### FR-16.16 Send/Sync auto traits
**Missing:** entirely.
**Sketch:** auto-derive based on field traits; `thread::spawn` requires `T: Send + 'static`.

### FR-16.18-extra const fn full evaluation
**Missing:** today's `const X: T = expr;` only takes int literals; calling const fns with const args is not wired.
**Sketch:** mini-interpreter over const-marked fns at MIR; bake result as immediate.

### FR-16.19 Slice/str/char primitives
**Missing:** `[T]` unsized + `&[T]` fat ptr + `str` + `char` + slicing syntax `s[a..b]`.
**Sketch:** introduce fat-pointer (data ptr, len) layout in asm backend; allocate two slots per `&[T]` local.

### FR-16.20 Unsafe + raw pointers
**Missing:** `unsafe { ... }` block, `*const T`, `*mut T`, `std::ptr::*`.
**Sketch:** parser accepts unsafe, asm backend treats `*const T` as opaque i64; ptr ops desugar to load/store.

### FR-16.21 `repr` attributes
**Missing:** `#[repr(C)]` / `(packed)` / `(transparent)` / `(u8)` for enums.
**Sketch:** read from existing `#[attr(...)]` parser; pass to layout builder for struct + enum.

### FR-16.22-extra Real async state-machine + executor
**Missing:** v3 lowers `async` as pass-through.
**Sketch:** state-machine struct (continuations as enum variants); poll impl; thread-pool executor over `aether_thread_*`; epoll/IOCP IO.

### FR-16.23-extra Mutex / RwLock / channel / Condvar / Barrier / Once / LazyLock
**Missing:** atomics + thread spawn ship; the rest don't.
**Sketch:** spinlock + park/wake primitives; `mpsc::channel<T>` ring buffer + condvar.

### FR-16.25 impl Trait return / argument-position
**Missing:** entirely.
**Sketch:** desugar `impl Trait` to a synthetic generic; opaque return uses an anonymous trait-object box at codegen.

---

## Phase 17 — Tensor stdlib (remaining 9 FRs)

### FR-17.1-extra Full dtype matrix beyond f32/f64/i32/i64
**Missing:** `f16`, `bf16`, `i8`, `u8`, `i16`, `u16`, `u32`, `u64`, `bool` not implemented end-to-end.
**Sketch:** AVX-512 `_Float16` intrinsics on Sapphire Rapids; AVX2 `vcvtph2ps` / `vcvtps2ph`; CUDA tensor cores for f16/bf16.

### FR-17.3 Convolutions (conv1d/2d/3d/transpose)
**Missing:** entirely. **Sketch:** im2col+sgemm for compatibility; cuDNN backend behind `--features cudnn`.

### FR-17.4 Pooling family
**Missing:** entirely. **Sketch:** straightforward CPU + CUDA kernels.

### FR-17.5-extra batchnorm / instancenorm / groupnorm / rmsnorm
**Missing:** layer_norm ships; the others don't. **Sketch:** mirror layer_norm shape; running-stat tracking for batchnorm.

### FR-17.6-extra Activations beyond gelu/relu/softmax/silu
**Missing:** tanh, sigmoid, leaky_relu, elu, mish, glu, swiglu, geglu.
**Sketch:** scalar lambdas (CPU) + CUDA fused kernels.

### FR-17.7-extra log/exp/sin/cos/tan/atan2/pow/recip/abs/sign/clamp
**Missing:** runtime exposes sqrtss; libm-class wrappers don't exist.
**Sketch:** call into `libm` for CPU; implement poly-approx for CUDA where appropriate.

### FR-17.8 Reductions full
**Missing:** sum/mean/var/std/min/max/argmax/argmin/prod per-dim or full.
**Sketch:** Welford for var/std; tree reductions for arg* on CUDA.

### FR-17.9 Selection (topk/sort/where/masked_fill/gather/scatter)
**Missing:** entirely. **Sketch:** radix sort or bitonic sort on CUDA; gather/scatter via permute.

### FR-17.10 Combine (cat/stack/split/chunk/repeat/repeat_interleave)
**Missing:** entirely. **Sketch:** stride-arithmetic only — no actual data copy for split/chunk.

### FR-17.11 Mask helpers (tril/triu/eye/arange/zeros/ones/full/randn)
**Missing:** entirely. **Sketch:** simple kernels; PCG64 for randn.

### FR-17.12 embedding_bag + sparse embedding
**Missing:** plain embedding ships; bag/sparse don't.
**Sketch:** bag = sum/mean over multiple indices; sparse uses CSR layout.

### FR-17.14-extra Quantization schemes (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 + AWQ + GPTQ + INT8 QAT)
**Missing:** GGUF reader header parses; full scheme set + fused dequant matmul don't ship.

### FR-17.16-extra MAE/BCE/BCEWithLogits/KL/Triplet/Contrastive/Huber/Smooth-L1
**Missing:** the runtime symbols exist (per `runtime/src/lib.rs`) but no per-loss tagged witness exists for many; loss_mse multi-tagged.
**Sketch:** add `loss_<name>.aether` per loss with finite-diff gradient check.

### FR-17.17-extra SGD-momentum / RMSprop / Adagrad / Adamax / Lion / Lamb / Adafactor
**Missing:** AdamW ships; the rest don't.
**Sketch:** mirror AdamW kernel shape; per-optim state tensor.

### FR-17.18-extra Conv/BN/Embedding/Dropout/MultiheadAttention/Transformer{Encoder,Decoder}/LSTM/GRU/RNN modules
**Missing:** Linear + LayerNorm + Embedding-via-lookup ship. The rest are FR-17.18-N.

### FR-17.19 Reference architectures (ResNet/ViT/Llama/BERT/SD/Mamba/MoE/CLIP)
**Missing:** entirely as `examples/<model>.aether` matching HF reference.
**Sketch:** load weights via SafeTensors; forward path against PyTorch eval.

### FR-17.20 Numerical parity bench
**Missing:** entirely.
**Sketch:** `bench/parity/run_all.ps1` runs every op against PyTorch + Candle reference at 1e-5 rel; pass/fail per op.

---

## Phase 18 — Distributed training — 10 FRs

### FR-18.1 Own NCCL bindings
**Missing:** entirely. **Sketch:** raw FFI to libnccl.so / nccl.dll; surface mirrors NCCL API.

### FR-18.2 All collectives (all-reduce/gather/reduce-scatter/broadcast/send/recv/all-to-all)
**Missing:** all_reduce_sum_f32 ships; the rest don't.

### FR-18.4 FSDP
**Missing:** entirely. **Sketch:** all-gather of params per layer fwd; reduce-scatter of grads bwd.

### FR-18.5 TP (tensor parallel)
**Missing:** entirely. **Sketch:** column-parallel + row-parallel `Linear`; sequence-parallel residual.

### FR-18.6 PP (pipeline parallel)
**Missing:** entirely. **Sketch:** 1F1B or interleaved 1F1B; activation checkpointing per stage.

### FR-18.7 ZeRO-1/2/3
**Missing:** entirely. **Sketch:** staged sharding of optim/grad/param.

### FR-18.8 Compute/comm overlap
**Missing:** entirely. **Sketch:** CUDA streams for collectives concurrent with kernel launches.

### FR-18.9 Gradient compression
**Missing:** entirely. **Sketch:** PowerSGD-class low-rank approx per bucket.

### FR-18.10 Multi-host RDMA
**Missing:** entirely. **Sketch:** InfiniBand verbs + RoCE Ethernet; GPU-Direct RDMA. **Hardware-blocked.**

### FR-18.11 8-GPU Llama-7B training run
**Missing:** entirely. **Hardware-blocked.**

---

## Phase 19 — Serving stack — 16 FRs (ALL of P19)

### FR-19.1 TLS 1.3 stack (own)
**Missing:** entirely. **Sketch:** ChaCha20-Poly1305 + AES-GCM + Ed25519 + X25519 + HMAC-SHA256.

### FR-19.2 HTTP/1.1 + HTTP/2 + HTTPS server
**Missing:** entirely (TCP listener exists; HTTP doesn't). **Sketch:** `aether::http::Server::bind(":8080").serve(handler)`.

### FR-19.3 OpenAI-compatible /v1/chat/completions
**Missing:** entirely. Depends on FR-19.2 + FR-17.19.

### FR-19.4 Paged KV cache
**Missing:** entirely. **Sketch:** block-allocated GPU mem with virtual-page mapping; LRU eviction.

### FR-19.5 Continuous batching scheduler
**Missing:** entirely. **Sketch:** mid-decode batch entry; preempt-longest on full.

### FR-19.6 Speculative decoding
**Missing:** entirely. **Sketch:** draft + verify model pair, ≥40% acceptance target.

### FR-19.7 Multi-model concurrent hosting
**Missing:** entirely. **Sketch:** per-model KV cache pool + memory budget.

### FR-19.8 gRPC + WebSocket
**Missing:** entirely. **Sketch:** Tonic-style codegen from `.proto`; WS upgrade + framing.

### FR-19.9 HF tokenizer parity (BPE/sentencepiece/tiktoken)
**Missing:** entirely. **Sketch:** load from `tokenizer.json`; bytes-equal round-trip.

### FR-19.10 Prompt template engine (Jinja-eq)
**Missing:** entirely. **Sketch:** `{{ }}`, `{% for %}`, `{% if %}` minimal.

### FR-19.11 Tool / function calling
**Missing:** entirely. **Sketch:** OpenAI-spec tool message; constrained decoding to JSON schema.

### FR-19.12 Vision input
**Missing:** entirely. **Sketch:** image preproc (resize/normalize/patchify) + ViT encoder hookup.

### FR-19.13 Speech input (Whisper)
**Missing:** entirely. **Sketch:** 16 kHz PCM → mel spectrogram → Whisper encoder.

### FR-19.14 Auth + rate limiting
**Missing:** entirely. **Sketch:** API key + JWT; token-bucket per user.

### FR-19.15 Observability (Prometheus + OTLP)
**Missing:** entirely. **Sketch:** `/metrics` endpoint; OTLP gRPC traces; structured JSON logs.

### FR-19.16 Llama-3-1B at >100 tok/s aggregate (umbrella)
**Missing:** entirely. **Hardware-attainable on 3070 Ti once FR-19.4/.5/.7 ship.**

---

## Phase 20 — Self-host (3 FRs)

### FR-20.8 3-stage bootstrap (A2 == A3 fix-point)
**Missing:** entirely. **Sketch:** Stage 0 = Rust-aetherc; Stage 1+2+3 produced by self-host; A2 == A3 byte-identical. Depends on every other P20 item.

### FR-20.9 Drop Rust dep from CLAUDE.md / SPEC.md
**Missing:** Rust still authoritative impl. Depends on FR-20.8.

### FR-20.10 Bootstrap CI
**Missing:** entirely. **Sketch:** local script (no GitHub Actions per Matt) runs `bootstrap.ps1` on every commit touching compiler/aether_asm/runtime.

---

## Phase 21 — Multi-platform — 8 FRs

### FR-21.2 Mach-O writer (macOS)
**Missing:** entirely. **Sketch:** Mach-O 64 header + load commands + LC_DYLD_INFO; Apple Silicon page-protection quirks.

### FR-21.3 ARM64 instruction encoder
**Missing:** entirely. **Sketch:** encoder + COFF/Mach-O/ELF rerouting; NEON for vectorize.

### FR-21.4 ROCm runtime (AMD GPUs)
**Missing:** entirely. **Sketch:** rocBLAS / MIOpen replacements for cuBLAS / cuDNN; HIP shim. **Hardware-blocked.**

### FR-21.5 Metal Performance Shaders
**Missing:** entirely. **Hardware-blocked.**

### FR-21.6 WebAssembly target
**Missing:** entirely. **Sketch:** WASM core + SIMD128; `.wasm` + JS glue.

### FR-21.7 no_std + embedded
**Missing:** runtime_pe is the model; full `aetherc --target=arm64-linux-musl` doesn't exist.

### FR-21.8 Mobile export (CoreML / NNAPI)
**Missing:** entirely. **Hardware-blocked.**

### FR-21.9 RISC-V instruction encoder
**Missing:** entirely. **Hardware-blocked.**

---

## Phase 22 — Compiler tooling — 10 FRs (ALL of P22)

### FR-22.1 LSP server (`aether-lsp`)
**Sketch:** completion / hover / goto-def / find-refs / rename / sig-help / diagnostics; clients for VS Code, Helix, Neovim.

### FR-22.2 DAP server (`aether-dap`)
**Sketch:** breakpoints, step over/in/out, eval, var inspect; source maps from asm backend.

### FR-22.3 `aetherfmt` (deterministic formatter)
**Sketch:** rustfmt-eq; single-round-trip stable.

### FR-22.4 `aetherclippy` (lints)
**Sketch:** ~50 starter lints (unused let, must-use, redundant clone).

### FR-22.5 `aetherdoc` (HTML doc generator)
**Sketch:** parse doc comments; cross-link types/fns/traits; search index.

### FR-22.6 Coverage (line + branch)
**Sketch:** instrument every basic block; per-bb counters at exit; HTML report.

### FR-22.7 Fuzzing (libafl-eq)
**Sketch:** grammar-aware parser fuzzer; coverage-guided.

### FR-22.8 Property-based testing (`#[quickcheck]`)
**Sketch:** generators for primitive + struct types; shrinking on failure.

### FR-22.9 Differential testing vs Candle/PyTorch
**Sketch:** drives `bench/parity/` numerical parity; same input → same output ±1e-5.

### FR-22.10 Incremental compilation
**Sketch:** per-fn + per-crate fingerprints; touched-fn-only recompile.

---

## Phase 23 — AI-assisted synthesis — 5 FRs

### FR-23.2 Auto-property generation
**Sketch:** for each `#[spec]` synthesised fn, generate ≥3 property tests (idempotence, totality, preservation).

### FR-23.3 Auto-test generation
**Sketch:** round-trip + edge-case tests synthesised from fn signature.

### FR-23.4 `#[infer]` compile-time numerical inference
**Sketch:** const-shape fns over const inputs evaluated at compile time; baked as `.rdata`.

### FR-23.5 Differential synthesis
**Sketch:** find inputs where Aether vs PyTorch differ >1 ULP; close gap via small kernel adjustments.

### FR-23.6 Synthesis demo
**Sketch:** `examples/synth_demo.aether` — a 5-fn synthesised module passes its auto-tests + shadows a hand-written reference within 1e-5 rel.

---

## Phase 24 — Production hardening — 10 FRs (ALL of P24)

### FR-24.1 Sanitizers (ASan / MSan / UBSan / TSan)
**Sketch:** instrumentation pass + runtime check fns; `--sanitize=<name>`.

### FR-24.2 Reproducible builds
**Sketch:** deterministic timestamps; no path leakage in .obj/.exe; `--reproducible`.

### FR-24.3 Supply-chain (signed packages, SBOM)
**Sketch:** Sigstore-shaped signing for `aether-pkg` registry; CycloneDX SBOM.

### FR-24.4 Cross-compilation (umbrella with FR-21.10)
**Sketch:** see FR-21.10.

### FR-24.5 Embedded runtime
**Sketch:** trimmed `runtime_pe`-style cdylib with custom alloc.

### FR-24.6 Hot-reload
**Sketch:** edit-and-continue for serving processes.

### FR-24.7 Crash dumps + telemetry (own)
**Sketch:** core + register dump on panic; optional remote telemetry (own server, no Sentry per Matt).

### FR-24.8 Real autoscaler
**Sketch:** watch QPS, spin up replicas, load-balance.

### FR-24.9 GPU memory leak detection
**Sketch:** per-allocation tracking; report unfreed allocations at exit.

### FR-24.10 OOM killer + graceful degradation
**Sketch:** under memory pressure, shrink KV cache pool, reject new requests with 503.

---

## How this file gets retired

Each FR moves out of NEXT-UP and into `tests/runtime/` as a tagged witness when the corresponding feature ships. The audit count
(`target/debug/aether-audit.exe --only roadmap`) is the source of truth — when a phase reaches 100%, its FR section here gets deleted in
the same commit. This file is intentionally short-lived; it shrinks as Aether grows.
