# Aether — Project Instructions

## Audit (run before claiming anything)

```
powershell -ExecutionPolicy Bypass -File scripts\audit.ps1
```

`tools/audit/` — `aether-audit` binary. Single-command structured audit:
1. **SLOC** per crate (file/code/comment/blank/total)
2. **Honesty scan** — counts and surfaces every `todo!()` / `unimplemented!()` / `unreachable!()` / `panic!()` / `unsafe` block / `#[ignore]` test / `fn ... { 0 }` stub return / "Phase N" comment in the workspace
3. **Workspace test census** — passed / failed / per-crate
4. **Golden artifacts** — committed expected outputs in `tests/golden/expected/` (MIR, asm, LLVM IR for `hello.aether` and `autodiff_step.aether`); diffed byte-for-byte. `--update-golden` to regenerate after intentional codegen changes.
5. **Aether language conformance** — `tests/aether/positive/` must `--check` clean; `tests/aether/negative/expect_AE####_*.aether` must fail with that exact diagnostic code.

Exit code non-zero if any dimension errors. **Run this before claiming any work is done.** The first run already caught a real parser bug (call args without commas were silently accepted) and (historically) a stub return in `silu_f32` — both now fixed; `silu_f32` has a real f32 CPU body alongside the rest of the op surface.

`--json` for machine-readable output; `--only sloc|scan|tests|golden|conformance` to focus a single dimension.

Snapshot is whatever `scripts\audit.ps1` prints — re-run it before quoting any number. As of last update: **0 todo/unimplemented/unreachable/ignored stubs** • a small fixed set of explicit `panic!()` guard rails • all `Phase N` markers point at explicitly-roadmap code • **all unit, golden, conformance, and runtime end-to-end suites green**. Runtime suite covers (alphabetical-ish): hello + arith + idivq/cqo + unary + ifelse + nested for + while/break/continue + FFI to libaether_rt + pointer-to-local + **f32 + f64 SSE2** (literals, arithmetic, ucomi[s|d] compares, int↔float + f32↔f64 casts) + 5+-arg FFI via stack spill at `[rsp+32+8*(i-4)]` + **struct field access** + **end-to-end model training driven from a `.aether` source** (`tests/runtime/train_tiny.aether`, loss `1.649245 → 0.006114` over 50 steps) + **self-hosted PE32+ writer** for FFI-free programs (`pe_exit_42.aether`, `pe_arith.aether`).

The runtime suite proves the asm backend correctness on real programs:

| test | exit | what it exercises |
|---|---|---|
| `hello_runtime` | 0 | `println(STR)` + `puts` linkage |
| `arith_42` | 42 | `let` + `+` |
| `imul_15` | 15 | `*` and operator precedence |
| `if_branch` | 99 | `if/else` with `>` |
| `if_eq` | 7 | nested `if/else` with `==` |
| `for_sum` | 45 | for-range + `let mut` + assignment |
| `for_ffi_tape` | 7 | for loop driving FFI |
| `nested_loops` | 25 | nested fors + accumulator + branched FFI |
| `while_loop` | 12 | `while` + `break` |
| `continue_skip` | 18 | for + `continue` skipping odd values |
| `div_mod` | 42 | `/` and `%` (idiv + cqo) |
| `unary_negate` | 10 | unary `-` and `!` |
| `ffi_self_check` | 0 | extern fn call into `aether_rt` |
| `ffi_tape_push` | 3 | 3-arg FFI sequence + count |
| `ffi_buffer` | 0 | `&local` → ptr passed to FFI

## Top-Line State

**Aether compiles to native machine code through its own assembler AND its own linker now.** `aetherc hello.aether --emit=pe-bin -o hello.exe` walks: Aether source → MIR → x86-64 AT&T assembly (aetherc backend) → COFF .obj (`aether_asm/` crate, our own x86-64 instruction encoder + COFF writer) → **PE32+ executable (our own multi-DLL PE writer at `aether_asm/src/pe.rs`)**. Zero external linker. Multi-DLL imports + per-symbol indirect-jmp thunks + multi-DLL IAT. The Windows OS provides `kernel32.dll` (everyone has it); `aether_rt.dll` (the runtime) ships next to the .exe. **Verified end-to-end by `tests/runtime/pe_train_tiny.aether`, which trains a single-layer linear classifier — loss `1.618 → 0.0081` over 50 steps — entirely through this chain.**

`--emit=aether-bin` is also retained: same compile chain but uses the system linker (gcc) and links statically against `libaether_rt.a` (the full Rust-std runtime). Same loss curve to within numerical noise. Choice is between "self-hosted" (pe-bin, slim runtime) and "static + full Rust std" (aether-bin, system linker).

Plus: a model trains end-to-end through the runtime ABI — no Python, no Rust ML framework, no candle, no torch. `runtime/` is a thin C-ABI shim with real f32 CPU implementations of every primitive op; `trainer/` calls only into that ABI to run forward, backward, and AdamW. Verified loss curve: **5.564 → 1.679 in 40 steps on AetherLM-Nano (~85K params, 2 layers, d=64, h=4, ff=128, seq=32)**. Inference round-trip verified.

Phase 1 swaps each `aether_op_*` body from f32 CPU to cuBLAS/cuDNN. Phase 5 rewrites `compiler/`, `aether_asm/`, and `runtime/` in Aether itself — once the compiler can self-host, Rust drops out of the entire stack.



## What This Is

Aether is a ground-up systems programming language for AI infrastructure. Three obsessions:

1. **Close to the metal** — raw pointers, explicit memory layout, SIMD, GPU kernels, no GC, no VM, no hidden allocations
2. **LLM + human readable** — keyword-heavy, minimal sigils, predictable braces, attribute-driven (`#[autodiff]`, `#[server(...)]`, `#[distributed(...)]`)
3. **Comments stripped at lex time** — 100% irreversible, zero comment bytes in any binary, even debug

First-class language features (NOT crates): autodiff, tensors, SIMD, distributed training, model hosting/serving.

## Status — Phase 0 + 0.5 working

End-to-end pipeline runs locally on `J:\aether\`. Workspace = `compiler/` + `runtime/`.

**Compiler (`compiler/`):**
- `lexer/mod.rs` — strips 100% of `//` and `/* */` (nested) comments at tokenization. Reports stripped byte count. Full keyword + punctuation set.
- `ast/mod.rs` — Program / Item / FnDecl / Attr / Ty (incl. `Shape([dims])` for `Tensor<f32, [M, K]>`) / Block / Stmt / Expr (with `Range`, `Region` for `warp/block/ai_region`, `For { parallel, distributed }`).
- `parser/mod.rs` — recursive-descent. Parses `#[attr(k=v, ...)]`, `fn`, `let`, `return`, blocks, paths, calls, method calls, field access, `if`/`else`, `for ... in ...`, `parallel for ... in ...`, `0..N` ranges, `&`/`&mut`, `warp { }` / `block { }` / `ai_region { }` regions, full operator precedence, generic types with shape arrays.
- `mir/mod.rs` — `run_autodiff_pass`. Detects `#[autodiff]` → wraps body with TapeInit / TapePush / AccumulateGrad / TapeReverse. Detects `#[distributed(world_size=N, backend="...")]` → appends AllReduce. `dump_mir` for `--emit=mir`.
- `mir/adgraph.rs` — typed AD graph with real symbolic partials. Ops: Const / Param / Add / Sub / Mul / MatMul / ReLU / CrossEntropy / Forward. `reverse()` emits real partials (`grad[a] += grad[id] * v[b]`, `softmax(v[logits]) - onehot(v[labels])`, etc.). Built per `#[autodiff]` fn and dumped alongside MIR.
- `codegen/llvm/mod.rs` — text LLVM IR emitter. Declares `@aether_autodiff_init/push/reverse/accumulate` and `@aether_dist_all_reduce` externs. Emits tape alloca + intrinsic calls per MIR.
- `codegen/c/mod.rs` — Phase 0 C fallback. Used by default `--emit=bin` so `aetherc foo.aether -o foo.exe` produces a runnable native binary via gcc. Throwaway once inkwell lands.
- `main.rs` — CLI driver. Flags: `-o`, `--emit=bin|mir|llvm-ir|c`, `--version`.

**Runtime (`runtime/`):** `libaether_rt`, staticlib + rlib. `#[no_mangle] extern "C"` definitions of `aether_autodiff_init/push/accumulate/reverse`, `aether_dist_all_reduce`, `aether_rt_self_check`. Phase 2 swaps the all-reduce body for an NCCL/MPI dispatch — the symbol surface stays identical.

**Assembler (`aether_asm/`):** x86-64 instruction encoder + Windows COFF (PE32+) object writer. Library + binary `aether-asm`. Tests verify byte-exact encodings against Intel SDM (push/pop/mov reg/imm/reg-reg/lea-rip/call/sub/add/xor/ret), COFF object layout (machine 0x8664, 2 sections, valid relocs), and a GAS-syntax parser round-trip. **252-byte hello.obj** for `examples/00_hello.aether` produced entirely from this crate. Phase 5 rewrites in Aether.

**Trainer (`trainer/`):** Rust binaries `aether-train`, `aether-infer`, `aether-prepare`. Single dep: `aether_rt = { path = "../runtime" }`. Every tensor operation is an FFI call to a `aether_op_*` symbol — no math in this crate, just orchestration. Forward + backward + AdamW + checkpointing + top-k sampling, all through the ABI. Mirrors `examples/aether_lm.aether` 1:1; future aetherc Phase 1 emits this same shape automatically.

**Aether stdlib (`stdlib/`):** every primitive op declared as `extern fn` resolving to a C symbol in `libaether_rt`. Phase 0 has real f32 CPU bodies for the AetherLM-Nano op set; Phase 1 swaps each body to cuBLAS / cuDNN / NCCL. Argument order is positional and frozen — see `runtime/ABI.md`.
- `stdlib/ops.aether` — 15 fns: `matmul_f32/bf16`, `add_f32`, `scale_f32`, `axpy_f32`, `gelu_f32`, `silu_f32`, `relu_f32`, `softmax_f32`, `layer_norm_f32`, `scaled_dot_product_attention_f32`, `cross_entropy_f32`, `zero_grad_f32`, `clip_grad_norm_f32`, `all_reduce_sum_f32`.
- `stdlib/optim.aether` — `adamw_step_f32`, `sgd_step_f32`, `AdamWState<S>`.
- `stdlib/nn.aether` — `Linear<I, O>`, `Embedding<V, D>`, `LayerNorm<D>` with `#[autodiff]` forward fns.
- `stdlib/tensor.aether` — `Tensor<T, S>`, `Simd<T, N>`.

**Showcase model — `examples/aether_lm.aether` (single Aether file).** AetherLM-Tiny, byte-level (vocab 256), 6 layers / d=320 / 5 heads / d_ff=1280 / seq=256, ~7.46M params. Defines `Block`, `AetherLm`, `Batch`, `causal_attention`, `ffn`, `block_forward`, `forward`, `train_step` (`#[autodiff]`), `train_step_ddp` (`#[autodiff]` + `#[distributed(world_size=3, backend="nccl", algorithm="ring")]`), `serve` (`#[server(...)]`). Aetherc emits MIR with real symbolic partials and LLVM IR with `@aether_autodiff_partial` + `@aether_dist_all_reduce` calls.

**No Python or framework dependency.** Earlier iterations of this project briefly added a PyTorch reference and then a candle reference — both deleted. The Aether source IS the model; the runtime ABI is the hand-off to vendor libraries; nothing in between.

**Local context:** the user maintains a candle fork at `J:\candle-src` with custom kernels (chunked CE, lora f16 cast, flash-attn v3 etc.). Aether **does not depend** on it, but it informs the op surface — anything candle-src needs in production is a candidate for an Aether `extern fn`.

**Examples (`examples/`):**
- `00_hello.aether` — basic
- `01_matmul.aether` — `#[target]`/`#[perf]` attrs, `parallel for`, `warp { }`, `Tensor<f32, [M, K]>` shape generics
- `02_train_mlp.aether` — `#[autodiff]` + `#[distributed(world_size=8, backend="nccl", algorithm="ring")]`, `model.forward(...).cross_entropy(...)`, `loss.backward()`
- `03_serve_llama.aether` — `#[server(port=8080, continuous_batching=true, paged_attention=true)]`
- `aether_lm.aether` — AetherLM-Tiny model: `causal_attention`, `ffn`, `train_step`, `train_step_ddp`, `serve`. The MIR for `causal_attention` contains real symbolic matmul/gelu/softmax partials.

**Tooling for LLM iteration:**
- `--check` — lex + parse + MIR pass, no codegen. Fast feedback loop.
- `--json-errors` — emit each diagnostic as JSON Lines on stderr with stable codes (`AE0001` lex, `AE0002` parse, `AE0100` io). Each carries `code/severity/stage/file/line/col/message/hint`.
- Diagnostic codes are stable — never reorder. `compiler/src/diag/mod.rs` is the source.

**Verified:**
- `cargo build --workspace` clean (release build of trainer also clean)
- `cargo test --workspace` — 28/28 pass: 11 compiler + 7 runtime + **10 aether_asm** (encoder bytes, COFF layout, GAS parser round-trip)
- `scripts/smoke.ps1` end-to-end: hello compiles + runs; train_mlp MIR contains `all_reduce grads world_size=8` and `softmax(...)` partial; train_mlp LLVM IR contains tape alloca + `@aether_autodiff_partial(...)` + `@aether_dist_all_reduce(...)`; broken file emits `AE0002` JSON; all four real .aether files (15+2+3+7 = 27 fns) check; **`aetherc hello.aether --emit=aether-bin` produces a runnable .exe through our own asm emitter and our own COFF writer**; **AetherLM-Nano trains: loss 5.564 → 1.679 in 40 steps (~30 s, ~3000 tok/s on CPU); inference round-trip produces text from the trained distribution.**

**Next critical-path work**: see Critical Path section below; #1 is done, start at #2 (inkwell swap).

## Repo Layout (target)

```
aether/
├── Cargo.toml                     # workspace
├── compiler/
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                # CLI driver: lex → parse → MIR → LLVM/C
│       ├── lexer/mod.rs           # comment-stripping lexer (// and /* */ gone)
│       ├── parser/mod.rs          # recursive-descent, parses #[attrs]
│       ├── ast/mod.rs
│       ├── mir/mod.rs             # autodiff pass + AllReduce insertion
│       └── codegen/llvm/mod.rs    # text IR emitter, inkwell swap-ready
├── examples/
│   ├── 00_hello.aether
│   ├── 01_matmul.aether
│   ├── 02_train_mlp.aether        # #[autodiff] + AdamW
│   └── 03_serve_llama.aether      # #[server(...)] continuous batching
├── stdlib/                        # Tensor, Simd, AI primitives (stub)
└── docs/AETHER_PROJECT_SKELETON.md
```

## Compilation Pipeline

```
.aether source
  → lexer (strips 100% of comments)
  → AST (attributes parsed)
  → MIR (autodiff tape + AccumulateGrad + AllReduce insertion, kernel fusion)
  → LLVM IR (tape alloca, @aether_autodiff_*, @aether_dist_all_reduce, NCCL)
  → native binary / PTX / .so
```

Flags: `--emit=mir`, `--emit=llvm-ir`, `--strip-comments` (default on), `--target=native`, `--opt=aggressive`, `--lto`, `--pgo`.

## Critical Path (in order)

1. ~~Scaffold repo, lexer/parser/AST/MIR/codegen, end-to-end MIR + LLVM text emit~~ **DONE**
2. ~~Typed AD graph with real symbolic partials (`mir/adgraph.rs`)~~ **DONE**
3. ~~Parser: `parallel for`, ranges, regions, shape generics, libaether_rt scaffold~~ **DONE**
4. ~~Lower AdGraph reverse() into `aether_autodiff_partial(tape, dst, op_code, src)` calls with stable op codes~~ **DONE**
5. ~~Structured diagnostics: `--check` + `--json-errors` + `AE####` codes + hints~~ **DONE**
6. ~~Parser: `extern fn`, `pub struct`/`struct`, `const`, `self` parameter, integer generic args~~ **DONE**
7. ~~Aether stdlib in `.aether` source (`ops`, `optim`, `nn`, `tensor`) + AetherLM-Tiny self-contained~~ **DONE**
8. ~~Runtime C-ABI surface defined and stubbed (`runtime/ABI.md` + `runtime/src/lib.rs::aether_op_*`)~~ **DONE**
9. ~~Real CPU bodies for every `aether_op_*` in `runtime/`~~ **DONE** — matmul + bwd, add/scale/axpy, gelu + bwd finite-diff verified, relu + bwd, softmax + bwd, layer_norm + bwd, sdpa causal + bwd, cross_entropy + bwd, embedding lookup + bwd, clip_grad_norm, AdamW. 7 runtime unit tests.
10. ~~`trainer/` crate — full forward + backward + AdamW + checkpointing + top-k sampling through libaether_rt only~~ **DONE**
11. ~~Real CPU training run with measured loss decrease~~ **DONE** — 5.564 → 1.679 in 40 steps on synthetic corpus.
12. ~~Direct x86-64 assembly emission (no LLVM / C in path)~~ **DONE** — `aetherc --emit=asm`, `--emit=asm-bin`, `--emit=aether-bin`.
13. ~~Own assembler with x86-64 encoder + COFF writer~~ **DONE** — `aether_asm/` crate, 10 unit tests, byte-exact against Intel SDM. 252-byte hello.obj.
14. ~~Grow the asm backend: locals + binary arithmetic + multi-arg calls~~ **DONE**
15. ~~`if/else` + comparison `Bin` ops (Eq/Ne/Lt/Gt/Le/Ge) → `cmp/setcc/movzbl`~~ **DONE**
16. ~~`for i in lo..hi` ranges, `while cond`, `break`, `continue` (proper loop-label stack for nesting)~~ **DONE**
17. ~~`let mut x` + `x = expr` assignment (`Bin::Assign`)~~ **DONE**
18. ~~Unary `-x` and `!x` (`negq`, testq+sete)~~ **DONE**
19. ~~Integer `/` and `%` (`cqo` + `idivq`)~~ **DONE**
20. ~~`&local` → `lea reg, disp(%rbp)` for passing pointers to FFI~~ **DONE**
21. ~~Link `libaether_rt.a` from `--emit=aether-bin` so `extern fn aether_*` resolves~~ **DONE** — proven by `ffi_self_check`, `ffi_tape_push`, `ffi_buffer`, `for_ffi_tape`, `nested_loops` runtime tests.
22. ~~f32 in the asm backend~~ **DONE** — SSE2 (xmm0–xmm7), `movss/addss/subss/mulss/divss/ucomiss`, f32 literal interning to `.rdata` via `.byte` directive, type-aware `Bin` lowering with stack spill for nested expressions, ucomiss + setcc for compares. Verified by `f32_compare` (1.5 + 2.5 == 4.0 → exit 7) and `f32_arith` ((10.0 * 4.5 - 3.0) / 1.0 == 42 → exit 42). **Next f32 work**: `cvtsi2ss` / `cvtss2si` for int↔float casts, `f32` arg passing via xmm0–xmm3 for FFI, f32 fn return values, f64 (xmm + 0xF2 prefix instead of 0xF3).
23. ~~Struct field access (`x.field`)~~ **DONE** — each field gets its own stack slot under a synthetic `name.field` key; `Stmt::Let.value` is now `Option<Expr>` so `let x: Foo;` (uninit) works. Field assignment + read both via slot lookup. Verified by `tests/runtime/struct_fields.aether`.
24. ~~**Self-hosted linker**~~ **DONE**: `aether_asm/src/pe.rs::build_full_exe` writes arbitrary `(dll, fns[])` imports with per-symbol indirect-jmp thunks + multi-DLL IAT. `aetherc --emit=pe-bin` drives the chain end-to-end with no system linker. Four DLLs in the symbol→DLL map: `kernel32.dll`, `msvcrt.dll`, `aether_rt.dll`, plus stubs for any other library you wire in. **Witness**: `tests/runtime/pe_train_tiny.aether` trains the linear classifier through the self-hosted path; loss curve `1.618 → 0.0081` over 50 steps. The slim `runtime_pe/` crate sits alongside `runtime/` and provides the cdylib (`no_std`, `panic=abort`, `core` + `libm` only, direct kernel32 externs); the f64 libm entries had alignment-trap AVs in their SAVE_XMM prologues so the slim crate uses f32-only math + hardware `sqrtss` + hand-rolled int-exponent `pow`.
25. ~~**Real cuBLAS/cuDNN backend in `runtime/`**~~ **DONE** (matmul + matmul_backward_{lhs,rhs} via cuBLAS sgemm; cross_entropy_fwd / cross_entropy_bwd / adamw_step via nvrtc-JITted custom kernels embedded in `runtime/src/cuda.rs::KERNEL_SRC`). End-to-end GPU training in `tests/runtime/cuda_train_tiny.aether` — bit-identical loss curve to the CPU `train_tiny.aether`. Single-op apples-to-apples vs candle-gpu (cuBLAS sgemm both): Aether matches or beats Candle at 3 of 4 sizes; see `docs/BENCH_RESULTS.md`. Feature-gated behind `--features cuda`; the bare build stays pure-Rust f32 CPU.
26. **First real training run on 3070 Ti**: compile `examples/aether_lm.aether --emit=aether-bin` once #22-#25 land. The Aether source describes the model end-to-end; nothing else changes.
27. **Self-host the compiler**: rewrite `compiler/`, `aether_asm/`, and `runtime/` in Aether. Drops Rust from the entire stack.
28. **Spec mode**: `#[spec(intent="…")]` natural-language → impl synthesis with human gate (Phase 5).

## Non-Negotiables

- **Comments stripped at lex time, always.** Not after parse, not in IR — at tokenization. No `--keep-comments` debug escape hatch unless explicitly requested.
- **No GC. No VM. No hidden allocations.** If a feature would introduce one, reject it.
- **Distributed is first-class.** `#[distributed(world_size=8, backend="nccl")]` on a function compiles to one binary that scales 1→1024 GPUs with no code changes. Single-device path must stay zero-overhead.
- **Autodiff via MIR, not macros.** Tape + reverse sweep lowered as LLVM intrinsics (`@aether_autodiff_push`, `@aether_autodiff_reverse`, `@aether_autodiff_accumulate`).
- **Compile-time shape checking** across distributed ranks — sharding verified before codegen.
- **Single static binary** is the default deployment target for both training and serving.

## War-Doc Decisions (do not relitigate)

- Tape-based autodiff chosen over pure source transform — flexibility + debugging. Hybrid mode allowed per function.
- MIR is the single source of truth for AI passes (autodiff, fusion, comm insertion).
- LLVM for portability (CPU/GPU/accelerators). Custom backend optional later.
- Rust + inkwell for the bootstrap compiler. Self-hosting is Phase 5, not now.
- Rejected: GC, VM, heavy proc-macro magic, Python-style indentation as syntax.

## Roadmap

- **Phase 0** (done in spec): bootstrap, lexer/parser, comment stripping, basic codegen
- **Phase 0.5** (done in spec): MIR autodiff pass, LLVM text emitter, attribute parsing
- **Phase 1**: real inkwell LLVM, full Tensor/SIMD lowering, autodiff tape lowering
- **Phase 2**: GPU/PTX backend, NCCL distributed runtime
- **Phase 3**: training ecosystem (DataLoader, AdamW, mixed precision, checkpointing)
- **Phase 4**: production serving (InferenceEngine, PagedKVCache, GGUF/SafeTensors/AWQ, OpenAI-compatible endpoint)
- **Phase 5**: self-hosting compiler, AI-assisted synthesis (`#[spec]` natural-language → impl with human gate)

## Success Metrics

- `train_mlp.aether` → native binary with working autodiff + `all_reduce` on real GPUs
- LLM serving binary within 5% of hand-tuned vLLM throughput
- LLMs read/write Aether with >95% correctness on benchmark kernels
- Matmul kernel within 5–10% of hand-written AVX-512 assembly

## Source of Truth

`J:\aether\SPEC.md` — the full spec, war doc, examples, and roadmap (was `handoff.md` until renamed to free the name for session handoffs). Read it before changing direction on anything in this file. `HANDOFF.md` is the session-handoff state file.
