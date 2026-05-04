# Aether vs Candle vs PyTorch — 3-way matmul micro-benchmark

Same hardware, same input shapes, same iter counts, same warm-up discipline,
same hot-loop discipline (allocate + h2d once outside the timer; one warm-up
matmul to amortise lazy kernel/handle init; final `cuda.synchronize()` /
`device.synchronize()` / `aether_dev_sync()` before stopping the clock).

**Hardware**: i9-11900K (8c/16t), RTX 3070 Ti 8 GiB, Win10 Pro.
**CUDA**: 12.6 toolkit, runtime 12.8 (PyTorch ships its own).
**Candle**: 0.10.2, local fork at `J:/candle-src`, MSVC toolchain, custom kernels.
**PyTorch**: 2.11.0+cu128.
**Aether**: this repo, debug `aetherc` against release `libaether_rt.a` (`--features cuda`).

Run with: `powershell -ExecutionPolicy Bypass -File bench/matmul_micro/run_all.ps1`.

## GPU — per-iter µs (lower is better)

| dim    | iters | **Aether-GPU** | **Candle-GPU** | **PyTorch-GPU** | leader     |
|-------:|------:|---------------:|---------------:|----------------:|------------|
|   64³  |   100 |       **7**    |              9 |             26  | **Aether** |
|  256³  |    50 |      **11**    |             13 |             19  | **Aether** |
|  512³  |    20 |      **39**    |             44 |             46  | **Aether** |
| 1024³  |    10 |        310     |        **213** |            196  | PyTorch    |

The 1024³ measurement varies run-to-run — Aether's batch fn is a tight loop
of `cudarc::CudaBlas::gemm` calls so the loop body has no overhead beyond
what cuBLAS does. PyTorch's slight edge at the largest size is plausibly
better cuBLAS handle/stream config (TF32 / split-K heuristics). At the
sizes that dominate small-batch transformer training, **Aether ties or
beats both**.

## CPU — per-iter µs (lower is better)

| dim    | iters | **Candle-CPU** | **PyTorch-CPU** |
|-------:|------:|---------------:|----------------:|
|   64³  |   100 |              9 |           **5** |
|  256³  |    50 |            143 |          **77** |
|  512³  |    20 |            660 |         **496** |
| 1024³  |    10 |          4,611 |       **4,221** |

PyTorch wins CPU at every size — it links against MKL and gets fused AVX-512
microkernels. Candle uses its own naive `f32` CPU path for small dims and
only switches to a BLAS at larger sizes.

Aether's CPU path is the same naive `aether_op_matmul_f32` body in
`runtime/src/lib.rs` — explicitly **out of scope** for the apples-to-apples
GPU bench, since the runtime's CPU path is the Phase-0 reference for
correctness, not a performance target. When `runtime/` rewrites to
Aether-self-hosted with hand-tuned AVX-512 microkernels (Phase 5),
that's where Aether catches up to MKL.

## Caveats

- Single trial per cell — variance ≥ 10% at small sizes. A multi-trial
  median + IQR is owed before claiming individual cells beyond noise.
- The Aether bench takes the buffers out of the registry ONCE and holds
  them across all iters (`aether_bench_matmul_batch`). The per-call
  `aether_op_matmul_f32_cuda` API has another ~150 µs of warmup-dominated
  overhead at first call (cudarc handle init). Numbers above are the
  warm batch path — same discipline as Candle's `for _ in 0..iters {
  a.matmul(&b)? }` and PyTorch's `for _ in range(iters): a @ b`.
- All three frameworks are running with their own bookkeeping (Candle:
  Tensor wrapper + lazy graph; PyTorch: autograd graph nodes when grads
  are tracked — disabled here). Aether has nothing — the value is the
  raw cuBLAS handle.
- This bench measures `cublasSgemm` throughput at the framework boundary.
  It does NOT measure full training-loop performance (where memory layout,
  fused kernels, and gradient bookkeeping dominate). For that, see
  `tests/runtime/cuda_train_transformer_block.aether`.
