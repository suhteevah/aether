"""Direct counterpart to the Candle + Aether matmul bench.

Same sizes / iter counts / warm-up discipline as `../candle/src/main.rs`
and `../../scratch/bench_batch.aether`. Tensors are allocated + h2d'd
once outside the timed loop; one warm-up matmul triggers any lazy
kernel/cublas-handle init; final `cuda.synchronize()` drains the queue
before stopping the clock. CPU column is the same call on `device='cpu'`
for the matching apples-to-apples baseline.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch

def run(device, m, k, n, iters):
    a = torch.randn(m, k, device=device, dtype=torch.float32)
    b = torch.randn(k, n, device=device, dtype=torch.float32)
    # Warm-up.
    _ = a @ b
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _c = a @ b
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return int((time.perf_counter() - t0) * 1_000_000)

def main():
    print(f"# pytorch {torch.__version__}  cuda={torch.version.cuda}  RTX 3070 Ti")
    cuda = torch.device("cuda:0")
    cpu  = torch.device("cpu")
    cfgs = [(64, 64, 64, 100), (256, 256, 256, 50),
            (512, 512, 512, 20), (1024, 1024, 1024, 10)]
    for (m, n, k, iters) in cfgs:
        us = run(cuda, m, k, n, iters)
        print(f"pytorch-gpu  M={m:>4}  N={n:>4}  K={k:>4}  iters={iters:>4}  us={us:>10}")
        us = run(cpu, m, k, n, iters)
        print(f"pytorch-cpu  M={m:>4}  N={n:>4}  K={k:>4}  iters={iters:>4}  us={us:>10}")

if __name__ == "__main__":
    main()
