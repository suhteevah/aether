// Direct counterpart to ../../scratch/bench_matmul_cpu_vs_gpu.aether's GPU
// half. Uses candle-core's CUDA backend so the comparison is apples-to-apples
// at the cuBLAS-sgemm layer; the difference between this and Aether's number
// for the same size is the per-op overhead candle adds (Tensor wrapper,
// device dispatch, autograd-graph bookkeeping when present, lazy eval) vs
// our raw `aether_op_matmul_f32_cuda` thunk + cublasSgemm.

use std::time::Instant;
use candle_core::{Device, DType, Tensor};

fn run(device: &Device, m: usize, k: usize, n: usize, iters: usize) -> u128 {
    // Allocate + h2d once; matches Aether's bench (transfers excluded from
    // the timed loop).
    let a = Tensor::randn(0f32, 1.0, (m, k), device).unwrap();
    let b = Tensor::randn(0f32, 1.0, (k, n), device).unwrap();
    // Warm-up to trigger any lazy kernel compilation / cuBLAS handle init.
    let _ = a.matmul(&b).unwrap();
    device.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _c = a.matmul(&b).unwrap();
    }
    device.synchronize().unwrap();
    t0.elapsed().as_micros()
}

fn main() {
    let cuda = Device::new_cuda(0).expect("cuda 0");
    let cpu = Device::Cpu;
    println!("# candle 0.10.2 (J:/candle-src local fork)  RTX 3070 Ti");
    let cfgs = [(64,64,64,100), (256,256,256,50), (512,512,512,20), (1024,1024,1024,10)];
    for (m, n, k, iters) in cfgs {
        let us_gpu = run(&cuda, m, k, n, iters);
        println!("candle-gpu  M={:>4}  N={:>4}  K={:>4}  iters={:>4}  us={:>10}", m, n, k, iters, us_gpu);
        let us_cpu = run(&cpu, m, k, n, iters);
        println!("candle-cpu  M={:>4}  N={:>4}  K={:>4}  iters={:>4}  us={:>10}", m, n, k, iters, us_cpu);
    }
}
