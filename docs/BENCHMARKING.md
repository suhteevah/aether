# Aether Benchmarking Plan

> **Purpose**: prove without ambiguity that Aether is faster than PyTorch and faster than candle, on CPU *and* on GPU, on their kernels *and* on ours, on their reference models *and* on ours. If Aether is **not** faster on a given axis, we want to know that too — and exactly why.
>
> **Status**: not yet runnable end-to-end. Blocked on critical-path item #25 (real cuBLAS/cuDNN bodies in `runtime/`). CPU-only baselines can land sooner but they are misleading on their own — the runtime's CPU matmul is single-threaded scalar f32, not AVX-512 BLAS.

## Why this matters

Aether's whole pitch is "closer to the metal than Python ever can be, no GC, no VM, no autograd graph allocation, no per-call PyObject wrapping". Until we **measure** that against the things people actually use, the pitch is hand-waving. Two specific theses to test:

1. **Aether ≪ PyTorch on inference and small-batch training.** PyTorch pays for its tensor metadata, autograd graph, dispatcher, and Python on every op. We don't. On batch=1 small models (LLM serving, real-time RL) we should win by a clear margin. On batch=large kernel-bound workloads we should *tie* (both call the same cuBLAS sgemm) — and that's the honest result.
2. **Aether vs candle is the real fight.** Candle is also Rust → cuBLAS/cuDNN/cudarc. Same layer of the stack. Wins over candle have to come from places candle pays overhead and we don't:
   - Candle has `Tensor` struct, lazy ops, an autograd graph; we have raw pointers and tape-based AD lowered into MIR by the compiler.
   - Candle dispatches each op through a backend trait and a kernel registry; our compiler emits direct `callq aether_op_*`.
   - Candle has no compile-time op fusion (today); ours can plan it in the MIR pass (Phase 1.5+).
   - Candle's `Tensor::matmul` allocates a result tensor on each call; we pass `out` as an explicit pointer.
   On naive matmul-only the answer should be roughly tied; on multi-op chains and end-to-end loops we should win.

## What "winning" means (per axis)

For every cell in the matrix below, we measure:

| metric | unit | notes |
|---|---|---|
| **wallclock per step** | µs/step | full forward+backward+optimizer for training; full forward for inference |
| **steady-state throughput** | tokens/sec or steps/sec | discard first 5 steps as warmup |
| **peak VRAM / RSS** | MiB | nvidia-smi for VRAM; getrusage(RUSAGE_SELF).ru_maxrss for RSS |
| **time-to-first-token** | ms | inference only |
| **lines of source code** | LOC | each runner side-by-side; soft signal but the user cares |
| **binary size** | bytes | static-linked binary or full deployment artifact (Python: env tarball) |

For correctness we also need:

| check | tolerance |
|---|---|
| **loss curve agreement** | within ±5% of reference at every recorded step on the same fixed-seed inputs |
| **logit max-abs delta** | ≤ 1e-4 on f32 / ≤ 1e-2 on bf16 vs reference |
| **gradient max-abs delta** | ≤ 1e-3 on f32 vs reference |

A bench result without these correctness gates is meaningless — fast wrong code is wrong code.

## The matrix

Three runtimes × two devices × two model sources × two op sources = 24 cells, but most are redundant. The minimum **interesting** cuts:

### A. Single-op micro-bench (cuBLAS surface area)
Compares pure dispatch overhead. Same device, same kernel, three frontends.
```
              CPU (single thread)          CPU (parallel)            GPU (3070 Ti)
matmul        Aether vs candle vs torch   Aether vs candle vs torch  Aether vs candle vs torch
attention     Aether vs candle vs torch   Aether vs candle vs torch  Aether vs candle vs torch
softmax       Aether vs candle vs torch   Aether vs candle vs torch  Aether vs candle vs torch
layernorm     Aether vs candle vs torch   Aether vs candle vs torch  Aether vs candle vs torch
```
Expected result: roughly tied on GPU (everyone calls cuBLAS sgemm); on CPU we're **slower** until we add a real BLAS path because our matmul is naive.

### B. End-to-end training, our reference model
`train_tiny.aether` (current bench: B=8, K=16, N=4 linear classifier, 50 steps, AdamW). Re-implement in candle and PyTorch with bit-identical inputs (same seed, same init, same labels). Compare loss at each step **and** wallclock.

Then scale up:
- **Tiny**: linear classifier, B=8, K=16, N=4 (have it now)
- **Small**: 2-layer MLP, B=128, hidden=256
- **Medium**: AetherLM-Nano (2-layer transformer, d=64, h=4, ff=128, seq=32, ~85K params)
- **Large**: AetherLM-Tiny (6-layer, d=320, h=5, ff=1280, seq=256, ~7.46M params)

We expect Aether to widen the lead as the model gets bigger because Python/Rust orchestration overhead becomes a smaller fraction.

### C. End-to-end training, *their* reference model
Take a published candle example (e.g. `candle-examples/examples/mnist`) and a PyTorch tutorial (the canonical `torch.nn.Linear` MNIST). Re-implement in Aether. Run all three with the exact same data + seed. Same metrics. This is the "no, you didn't cherry-pick the model" answer.

### D. Inference latency, identical weights
Train one model in PyTorch, save weights, load them into Aether and candle. Run inference at batch=1, batch=32, batch=512 on the same hardware. This is where Aether should look best — no Python tax per call.

### E. End-to-end serving (Phase 4)
Long-running OpenAI-compatible endpoint. Measure tokens/sec, p50/p99 latency, max-batch-stable throughput vs `vllm`, `tgi`, `llama.cpp` server. Out of scope until #4 lands but worth listing.

## Reference implementations

For each end-to-end bench we keep three siblings checked into `bench/`:

```
bench/
├── train_tiny/
│   ├── aether/   train_tiny.aether                    (already exists, in tests/runtime/)
│   ├── candle/   src/main.rs                          (Cargo project, depends on candle-core)
│   ├── torch/    train_tiny.py                        (single file, requirements.txt pinned)
│   └── README.md  shared-seed inputs, pinned versions, run command, expected metrics
├── matmul_micro/
│   ├── aether/   bench_matmul.aether
│   ├── candle/   src/main.rs
│   ├── torch/    bench_matmul.py
│   └── README.md
└── ...
```

Each `README.md` contains the exact command to run, the expected loss-curve / output, and the comparison table. The harness lives in `tools/bench/` (Rust binary, walks `bench/*/`, runs each runner, parses output, emits a single comparison table per bench dir + a top-level summary).

## Pinned versions

Reproducibility requires version pins:

| dep | version | reason |
|---|---|---|
| candle | 0.7.x (Hugging Face; pin commit hash, not just minor) | API drifts between minors |
| torch | 2.4.x CPU and 2.4.x+cu121 GPU | cu121 matches CUDA 12.1 on Matt's 3070 Ti |
| python | 3.11 | torch 2.4 supports it; 3.12 has stragglers |
| cuda toolkit | 12.1 | matches torch wheel |
| cuDNN | 9.x bundled with cuda 12.1 | |

Pin in `bench/<name>/{Cargo.toml, requirements.txt}`. Capture `git rev-parse HEAD` of candle and the torch wheel build hash in the README so future re-runs are exact.

## Hardware

- CPU bench: i9-11900K, 8 cores / 16 threads, single CCD. Lock CPU governor to performance via `powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c` before running.
- GPU bench: RTX 3070 Ti, 8 GiB VRAM. Lock clock with `nvidia-smi -lgc <base>,<base>` to remove boost noise.
- Disable any background Wraith / candle-src builds while benching.

## What blocks us today

Item-by-item, in dependency order:

1. **Real cuBLAS/cuDNN backend in `runtime/`** (#25 on the critical path).
   Without this, every "GPU" cell in the matrix is a no-op (the runtime's `aether_op_matmul_f32` is a single-threaded scalar Rust loop). CPU cells are runnable but uninteresting until we also add a parallel/AVX matmul path or accept that "we lose CPU until BLAS lands" is the real story we tell.
2. **Saved weights interop** for axis D (inference with shared weights). Need a deterministic layout — pick safetensors v1, write a loader in libaether_rt and matching savers in the candle/torch siblings.
3. **A bench harness** at `tools/bench/`. Subprocess each runner under controlled CPU/GPU state, parse output, emit a markdown table. ~1 day of work; do it after #25.
4. **Bigger Aether language surface** for axes B-medium and B-large: structs (#23), arrays / pointer arithmetic, fn calls with f32 returns inside more complex expressions. Most of these are queued.

## Honest expectations

Setting expectations before we run anything, so the results are read against a prior:

- **CPU matmul**: Aether **loses** to both, badly, until we add MKL/OpenBLAS or a tuned AVX kernel. This is correct: we deliberately stayed scalar so the runtime is auditable. Phase-1 work tells us where this lands.
- **GPU matmul (single op)**: roughly **tied** with both. Everyone is bound by the same cuBLAS sgemm. Wins of more than ~2% here would suggest one of the others has a real bug; wins of 5-10% would suggest layout/transpose differences.
- **GPU end-to-end training**: Aether **wins** by 1.2-2× over PyTorch (Python orchestration tax) and **wins** by 1.05-1.15× over candle (no Tensor metadata, no graph dispatch). If we don't see that, something in our code is leaving perf on the table.
- **Inference latency batch=1**: Aether **wins big** over PyTorch (3-10×) and modestly over candle (1.2-1.5×). This is the strongest pitch.
- **LOC / binary size**: Aether **wins** trivially: a `.aether` source + `libaether_rt.a` is smaller than candle + tokenizers + serde + the world, much smaller than a Python virtualenv.

If reality doesn't match these priors, the priors are wrong and the result is the news.

## Run order, when ready

1. Land #25 (cuBLAS bodies). Loss curve of `train_tiny.aether` should be unchanged; only wallclock changes. This is the regression gate for the swap.
2. Write the bench harness shell + add `bench/train_tiny/` siblings (candle, torch).
3. Run axis A (single-op) on CPU and GPU. Publish first table.
4. Run axis B (end-to-end, our model) on CPU and GPU. Scale through tiny → small → medium → large.
5. Run axis C (their model). This is the hardest sell to write but the easiest to defend — port a published reference, show numbers.
6. Run axis D (shared-weight inference). Phase-4 PagedKV-cache work feeds into this.
7. Phase-5 run axis E (serving).
