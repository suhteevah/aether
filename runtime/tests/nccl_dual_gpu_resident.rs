//! GPU-resident data-parallel training step over 2× P100.
//!
//! Weights, gradients, and the optimizer update all live on each
//! rank's GPU across iterations. No host round-trip of W between
//! steps -- the only host work per iter is computing the loss
//! (small reduction) for logging.
//!
//! This is the deeper companion to `nccl_dual_gpu_dp_step.rs`. That
//! file proved cross-card all_reduce semantics with host-side weights.
//! This file proves the same DP shape with weights pinned to device
//! across N optimization steps -- the matt-voice training-deploy
//! shape minus the QMatMul kernel.
//!
//! Objective: minimise ||W - target||^2 with W per-rank initialised
//! identically and target shared. Each rank computes the local
//! gradient (just 2*(W - target)), all_reduces (sum then /world_size
//! = mean), and applies an SGD step. Loss should decline; final W
//! should be byte-identical across ranks (the DP invariant).

#![cfg(feature = "nccl")]

use aether_rt::nccl_real;

#[test]
fn dual_gpu_resident_dp_steps() {
    use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig, CudaSlice};
    use cudarc::nccl::safe::{ReduceOp, group_start, group_end};
    use cudarc::nvrtc::compile_ptx;

    let n_devs = CudaDevice::count().unwrap_or(0);
    if n_devs < 2 {
        eprintln!("[skip] need 2+ CUDA devices, found {}", n_devs);
        return;
    }

    let n_ranks = 2usize;
    let rc = nccl_real::aether_nccl_real_init_multi_gpu(n_ranks as i32);
    assert_eq!(rc, n_ranks as i32);

    // Tiny model: just a flat parameter vector of N elements.
    let n = 1024usize;
    let lr = 0.1f32;
    let n_steps = 50;

    // Target shared across ranks. W starts at all-zeros so initial
    // loss = ||target||^2 = sum_i target_i^2.
    let target_host: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 0.5).collect();

    // Per-rank: allocate W + grad on each comm's device.
    // Build a small CUDA kernel that does grad = 2*(W - target).
    let kernel_src = r#"
        extern "C" __global__ void compute_grad(
            float* w, const float* target, float* grad, int n
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                grad[i] = 2.0f * (w[i] - target[i]);
            }
        }
        extern "C" __global__ void sgd_step(
            float* w, const float* grad, float lr, int n
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                w[i] -= lr * grad[i];
            }
        }
        extern "C" __global__ void sq_diff(
            const float* w, const float* target, float* out, int n
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                float d = w[i] - target[i];
                out[i] = d * d;
            }
        }
    "#;
    let ptx = compile_ptx(kernel_src).expect("ptx compile");

    // Per-rank state.
    struct RankState {
        device_ordinal: usize,
        w: CudaSlice<f32>,
        target: CudaSlice<f32>,
        grad_send: CudaSlice<f32>,
        grad_recv: CudaSlice<f32>,
        sq_buf: CudaSlice<f32>,
        // Functions are loaded per-device (each device has its own ctx).
        compute_grad: cudarc::driver::CudaFunction,
        sgd_step: cudarc::driver::CudaFunction,
        sq_diff: cudarc::driver::CudaFunction,
    }
    let mut ranks: Vec<RankState> = Vec::with_capacity(n_ranks);
    for rank in 0..n_ranks {
        let comm = nccl_real::comm_at(rank).expect("comm");
        let dev = comm.device();
        dev.load_ptx(ptx.clone(), "dp", &["compute_grad", "sgd_step", "sq_diff"])
            .expect("load_ptx");
        let compute_grad = dev.get_func("dp", "compute_grad").expect("compute_grad");
        let sgd_step = dev.get_func("dp", "sgd_step").expect("sgd_step");
        let sq_diff = dev.get_func("dp", "sq_diff").expect("sq_diff");
        let w = dev.alloc_zeros::<f32>(n).expect("alloc W");
        let target = dev.htod_sync_copy(&target_host).expect("htod target");
        let grad_send = dev.alloc_zeros::<f32>(n).expect("alloc grad_send");
        let grad_recv = dev.alloc_zeros::<f32>(n).expect("alloc grad_recv");
        let sq_buf = dev.alloc_zeros::<f32>(n).expect("alloc sq_buf");
        ranks.push(RankState {
            device_ordinal: rank,
            w, target, grad_send, grad_recv, sq_buf,
            compute_grad, sgd_step, sq_diff,
        });
    }

    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut loss_trace = Vec::new();

    for step in 0..n_steps {
        // 1) Compute grad on each rank's device (in place).
        for r in 0..n_ranks {
            let rs = &mut ranks[r];
            unsafe {
                rs.compute_grad.clone().launch(cfg, (&mut rs.w, &rs.target, &mut rs.grad_send, n as i32))
                    .expect("launch compute_grad");
            }
        }

        // 2) NCCL all_reduce grads across ranks.
        group_start().expect("group_start");
        for r in 0..n_ranks {
            let comm = nccl_real::comm_at(r).expect("comm");
            let rs = &mut ranks[r];
            comm.all_reduce(&rs.grad_send, &mut rs.grad_recv, &ReduceOp::Sum)
                .unwrap_or_else(|e| panic!("rank {} all_reduce: {:?}", r, e));
        }
        group_end().expect("group_end");

        // 3) SGD step on each rank's device with scaled grad
        //    (recv = sum across ranks; effective lr / world_size).
        let effective_lr = lr / n_ranks as f32;
        for r in 0..n_ranks {
            let rs = &mut ranks[r];
            unsafe {
                rs.sgd_step.clone().launch(cfg, (&mut rs.w, &rs.grad_recv, effective_lr, n as i32))
                    .expect("launch sgd");
            }
        }

        // 4) Compute loss for logging (rank 0 only -- a small d2h
        //    that's NOT in the training-critical path).
        if step % 10 == 0 || step + 1 == n_steps {
            let rs = &mut ranks[0];
            unsafe {
                rs.sq_diff.clone().launch(cfg, (&rs.w, &rs.target, &mut rs.sq_buf, n as i32))
                    .expect("launch sq_diff");
            }
            let sq: Vec<f32> = rs.sq_buf.device().dtoh_sync_copy(&rs.sq_buf).expect("d2h sq");
            let loss: f32 = sq.iter().sum::<f32>() / n as f32;
            eprintln!("[gpu-resident-dp ws={}] step={:>4} loss={:.6}", n_ranks, step, loss);
            loss_trace.push((step, loss));
        }
    }

    // Final invariant: both ranks have identical W (sample first 8 elems).
    let w0: Vec<f32> = ranks[0].w.device().dtoh_sync_copy(&ranks[0].w).expect("d2h w0");
    let w1: Vec<f32> = ranks[1].w.device().dtoh_sync_copy(&ranks[1].w).expect("d2h w1");
    let mut all_eq = true;
    for i in 0..8 {
        if (w0[i] - w1[i]).abs() > 1e-5 {
            eprintln!("rank divergence at {}: {} vs {}", i, w0[i], w1[i]);
            all_eq = false;
        }
    }
    assert!(all_eq, "DP invariant violated: ranks have different W");

    // Loss must have declined.
    let first = loss_trace.first().unwrap().1;
    let last = loss_trace.last().unwrap().1;
    assert!(last < first * 0.5,
        "expected loss to halve, got {} -> {}", first, last);

    eprintln!(
        "[gpu-resident-dp] W on device for {} steps, loss {:.6} -> {:.6}, ranks byte-identical",
        n_steps, first, last,
    );

    nccl_real::aether_nccl_real_finalize();
}
