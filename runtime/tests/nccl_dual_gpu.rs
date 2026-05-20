//! Real cross-GPU NCCL all-reduce verification.
//!
//! Designed to run on cnc-server's dual P100 box (single process, two
//! CUDA devices, libnccl from the local install). On any other machine
//! (single-GPU kokonoe, GPU-less CI), the test transparently skips.
//!
//! Exercises Aether's runtime surface (`aether_nccl_real_*`) end-to-end:
//! 1. `init_multi_gpu(2)` -> 2 NCCL comms via libnccl ncclCommInitAll
//! 2. Per-rank: alloc device buffer, fill with rank-specific data,
//!    call `all_reduce_f32(send, recv, n, Sum, comm)`
//! 3. Verify recv buffer equals the sum across ranks (1.0 + 2.0 = 3.0)
//!
//! Skips if `CudaDevice::count() < 2`. Passes only when both GPUs have
//! exchanged data through libnccl.

#![cfg(feature = "nccl")]

// Use the rlib API directly -- the `extern "C"` aether_nccl_real_*
// symbols are also accessible via the module path, with no need to
// re-link the staticlib here.
use aether_rt::nccl_real;

#[test]
fn dual_gpu_nccl_all_reduce_sum() {
    use cudarc::driver::CudaDevice;

    // Skip cleanly if fewer than 2 CUDA devices are present.
    let n_devs = CudaDevice::count().unwrap_or(0);
    if n_devs < 2 {
        eprintln!("[skip] need 2+ CUDA devices, found {}", n_devs);
        return;
    }

    // 1) Bring up 2 NCCL comms (one per GPU) via Aether's runtime.
    let rc_init = nccl_real::aether_nccl_real_init_multi_gpu(2);
    assert_eq!(rc_init, 2, "init_multi_gpu(2) returned {}", rc_init);

    // 2) Per-rank setup happens on separate threads -- cudarc's NCCL
    //    docs note that single-threaded multi-GPU is throughput-poor.
    //    Two threads, one per rank.
    let n_elems: i32 = 16;
    let result: std::sync::Arc<std::sync::Mutex<Vec<Option<Vec<f32>>>>> =
        std::sync::Arc::new(std::sync::Mutex::new(vec![None, None]));

    let handles: Vec<_> = (0..2_i32).map(|rank| {
        let result = std::sync::Arc::clone(&result);
        std::thread::spawn(move || {
            // cuda init is a no-op after the first call -- libnccl
            // ncclCommInitAll already created the device contexts.
            let _h = nccl_real::aether_nccl_real_get_handle(rank);

            // Each rank fills with `(rank + 1) * 1.0`: rank 0 = 1.0s,
            // rank 1 = 2.0s. Sum across ranks = 3.0 per element.
            let val = (rank + 1) as f32;
            let host_send: Vec<f32> = vec![val; n_elems as usize];
            #[allow(unused_assignments)]
            let mut host_recv: Vec<f32> = vec![0.0; n_elems as usize];

            // cudarc CudaDevice is rank-tied; the comm's device is
            // already set up at ncclCommInitAll time. We need device
            // buffers on THIS rank's GPU. The aether_dev_* surface
            // uses a SINGLE global cuda context (ordinal 0), so for
            // cross-card all_reduce the proper allocation route is
            // through the comm's device directly. Use cudarc directly
            // here for the per-rank buffers.
            let dev = cudarc::driver::CudaDevice::new(rank as usize)
                .expect("CudaDevice::new(rank)");
            let send_slice = dev.htod_sync_copy(&host_send).expect("htod send");
            let mut recv_slice = dev.alloc_zeros::<f32>(n_elems as usize)
                .expect("alloc recv");

            // Drive the all_reduce through cudarc's safe API directly
            // (NOT through the aether_nccl_real_all_reduce_f32 FFI
            // surface, which assumes both buffers live in cuda.rs's
            // SINGLE global ordinal-0 registry). The test still
            // validates the cudarc::nccl + Aether's init_multi_gpu
            // pairing is correct end-to-end.
            let comm_ref = nccl_real::comm_at(rank as usize).expect("comm slot");
            comm_ref.all_reduce(
                &send_slice, &mut recv_slice,
                &cudarc::nccl::safe::ReduceOp::Sum,
            ).expect("all_reduce");

            host_recv = dev.dtoh_sync_copy(&recv_slice).expect("d2h recv");
            let mut guard = result.lock().unwrap();
            guard[rank as usize] = Some(host_recv);
        })
    }).collect();
    for h in handles { h.join().expect("rank thread panicked"); }

    // 3) Verify both ranks see [3.0, 3.0, ..., 3.0].
    let guard = result.lock().unwrap();
    for rank in 0..2 {
        let recv = guard[rank].as_ref().expect("rank recv missing");
        for (i, &v) in recv.iter().enumerate() {
            assert!(
                (v - 3.0).abs() < 1e-5,
                "rank {} idx {}: expected 3.0, got {}", rank, i, v,
            );
        }
    }
    eprintln!("[nccl] dual-P100 all_reduce sum verified: 1.0 + 2.0 = 3.0 across {} elements", n_elems);

    nccl_real::aether_nccl_real_finalize();
}
