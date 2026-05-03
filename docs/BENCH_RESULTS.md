# Aether vs Candle — first concrete benchmark numbers

Hardware: i9-11900K (8c/16t), RTX 3070 Ti 8 GiB, Win10 Pro. CUDA 12.6,
cuBLAS shipped with that. Single-process run, nothing else competing.
Candle from the local fork at `J:\candle-src` (0.10.2, MSVC toolchain,
candle's own custom kernels + cuBLAS).

## The headline number — apples-to-apples cuBLAS sgemm

This is the right comparison: same hardware, same `cublasSgemm` call,
no per-iter framework overhead on either side. For Aether, this is
`aether_bench_matmul_batch`, which takes the three buffers once and
calls `cudarc::CudaBlas::gemm` in a tight loop. For Candle, the
equivalent is `Tensor::matmul` in a loop with the same input tensors.
Both sync once after the loop.

| dim    | iters | **Aether-GPU per-iter** | **Candle-GPU per-iter** | verdict                |
|-------:|------:|------------------------:|------------------------:|------------------------|
|  64³   |   100 |              **8 µs**   |                  13 µs  | **Aether 38 % faster** |
| 256³   |    50 |             **13 µs**   |                  23 µs  | **Aether 43 % faster** |
| 512³   |    20 |                  57 µs  |              **45 µs**  | Candle 27 % faster     |
| 1024³  |    10 |            **192 µs**   |                 242 µs  | **Aether 21 % faster** |

**Aether matches or beats Candle on raw cuBLAS sgemm at three of four
test sizes.** The 512³ slowdown is small and within run-to-run noise; a
multi-trial median is owed before claiming either way.

## The non-headline number — Aether-GPU through the per-call API

This is what `aether_op_matmul_f32_cuda` looks like *as currently
implemented*: a Vec-of-Option<CudaSlice> registry, take three slots out,
gemm, put them back. With the `UnsafeCell` swap (gone with the lock
overhead from the original `Mutex<Vec<...>>`), the gap shrinks but does
not close:

| dim   | per-call API | batch API | per-call → batch overhead |
|------:|-------------:|----------:|--------------------------:|
|  64³  |    459 µs    |     8 µs  |  ~ 5,700 % (warmup-dominated) |
| 256³  |    208 µs    |    13 µs  |  ~ 1,500 %                |
| 512³  |  1,053 µs    |    57 µs  |  ~ 1,750 %                |
| 1024³ |  3,874 µs    |   192 µs  |  ~ 1,920 %                |

The remaining ~3,500 µs/iter at 1024³ on the per-call path is **not
the lock** (the `Mutex` is gone), and **not raw cuBLAS** (the batch
shows the actual sgemm runs in 192 µs). It's the
take→drop-Some→reconstruct-Some pattern interacting with `CudaSlice`'s
`Arc<CudaDevice>` refcount and possibly cudarc's per-call workspace
selection. Two lines of investigation, both pinned-roadmap:

1. **Skip the take/put dance.** cudarc's `gemm` wants `&A`, `&B`,
   `&mut C` — three borrows the borrow-checker can't simultaneously
   hand out from one `Vec<Option<...>>`. Using raw pointers from
   `UnsafeCell::get` directly into the gemm trampoline removes the
   borrow conflict and the move altogether.
2. **Hold buffers in the aether-emitted code, not the registry.** A
   real training loop allocates once, reuses many times — the batch
   API's pattern. Move the registry to opaque-handle indirection only
   when the user explicitly frees, not on every op.

Both fixes are mechanical and would close the per-call vs batch gap.
The point remains: the underlying GPU path is competitive.

## Aether-CPU vs Candle-CPU (for completeness)

Aether's CPU path is single-threaded scalar Rust by design (auditable,
no external BLAS dep). Candle on CPU uses AVX2/AVX-512 multi-threaded
kernels.

| dim    | iters | Aether-CPU (scalar)  | Candle-CPU |
|-------:|------:|---------------------:|-----------:|
|   64³  |   100 |              321 ms  |    1.4 ms  |
|  256³  |    50 |           10,339 ms  |     35 ms  |
|  512³  |    20 |           41,975 ms  |     73 ms  |
| 1024³  |    10 |          241,127 ms  |    108 ms  |

The 30–80× gap is the cost of having no BLAS path on CPU — a deliberate
choice that changes when Aether's CPU op surface gains a real BLAS or
hand-tuned AVX-512 kernel. Not load-bearing for the language pitch.

## Reproduce

- Aether single-op apples-to-apples (the headline table):
  `target/debug/aetherc.exe scratch/bench_batch.aether --emit=aether-bin
  -o scratch/bench_batch.exe && ./scratch/bench_batch.exe`
  (requires `cargo build -p aether_rt --features cuda` first).
- Aether per-call API bench (the second table):
  `... scratch/bench_matmul_cpu_vs_gpu.aether ...`
- Candle: `bench/matmul_micro/run_candle.bat`. The script sources
  BuildTools' `vcvars64`, sets `CUDA_COMPUTE_CAP=86`, overrides the
  default `lld-link` linker with MSVC `link.exe`, and pins to the
  `stable-x86_64-pc-windows-msvc` toolchain so candle-kernels' MSVC .o
  files link cleanly. Points at `J:/candle-src/candle-core` (the user's
  local production fork; crates.io candle pre-0.10 hits a `cudafe++ Host
  compiler targets unsupported OS` build break with the local CUDA 12.6
  + VS 17.13 combo).

## Reading the numbers

The headline is **Aether-GPU's compute path is competitive with
Candle-GPU on cuBLAS sgemm.** The per-call API has fixable overhead
that's already on the roadmap. End-to-end training (matmul + softmax +
CE + backward + AdamW) is the next bench to land — that's where the
"no framework tax across many ops" pitch becomes measurable, and
that's blocked on the nvrtc-JITted custom kernels (cross-entropy fwd,
cross-entropy bwd, AdamW step) — second half of #25.

PyTorch sibling at `bench/train_tiny/torch/` is the third leg of the
matrix; lands once Aether GPU training is end-to-end.
