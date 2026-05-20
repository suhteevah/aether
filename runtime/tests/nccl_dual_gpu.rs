//! Real cross-GPU NCCL all-reduce verification.
//!
//! Designed to run on cnc-server's dual P100 box (single process, two
//! CUDA devices, libnccl from the local install). On any other machine
//! (single-GPU kokonoe, GPU-less CI), the test transparently skips.
//!
//! Exercises Aether's runtime surface (`aether_nccl_real_*`) end-to-end:
//! 1. `init_multi_gpu(2)` -> 2 NCCL comms via libnccl ncclCommInitAll
//! 2. group_start / per-rank all_reduce / group_end
//! 3. Verify recv buffer equals the sum across ranks (1.0 + 2.0 = 3.0)

#![cfg(feature = "nccl")]

use aether_rt::nccl_real;

#[test]
fn dual_gpu_nccl_all_reduce_sum() {
    use cudarc::driver::CudaDevice;
    use cudarc::nccl::safe::{ReduceOp, group_start, group_end};

    let n_devs = CudaDevice::count().unwrap_or(0);
    if n_devs < 2 {
        eprintln!("[skip] need 2+ CUDA devices, found {}", n_devs);
        return;
    }

    let rc_init = nccl_real::aether_nccl_real_init_multi_gpu(2);
    assert_eq!(rc_init, 2, "init_multi_gpu(2) returned {}", rc_init);

    let n_elems: usize = 16;
    let n_ranks: usize = 2;

    // Per-rank send/recv device buffers, allocated on each comm's device.
    // Single-thread iteration; NCCL group mode lets all_reduce calls
    // queue together and dispatch as one collective.
    let mut send_bufs: Vec<cudarc::driver::CudaSlice<f32>> = Vec::with_capacity(n_ranks);
    let mut recv_bufs: Vec<cudarc::driver::CudaSlice<f32>> = Vec::with_capacity(n_ranks);
    for rank in 0..n_ranks {
        let comm = nccl_real::comm_at(rank).expect("comm slot");
        let dev = comm.device();
        let val = (rank + 1) as f32;
        let host_send: Vec<f32> = vec![val; n_elems];
        let send = dev.htod_sync_copy(&host_send).expect("htod send");
        let recv = dev.alloc_zeros::<f32>(n_elems).expect("alloc recv");
        send_bufs.push(send);
        recv_bufs.push(recv);
    }

    // Issue the collectives in group mode.
    group_start().expect("group_start");
    for rank in 0..n_ranks {
        let comm = nccl_real::comm_at(rank).expect("comm slot");
        let send = &send_bufs[rank];
        let recv = &mut recv_bufs[rank];
        comm.all_reduce(send, recv, &ReduceOp::Sum)
            .unwrap_or_else(|e| panic!("rank {} all_reduce: {:?}", rank, e));
    }
    group_end().expect("group_end");

    // Copy back and verify.
    let mut all_match = true;
    for rank in 0..n_ranks {
        let comm = nccl_real::comm_at(rank).expect("comm slot");
        let dev = comm.device();
        let recv_host = dev.dtoh_sync_copy(&recv_bufs[rank]).expect("d2h recv");
        for (i, &v) in recv_host.iter().enumerate() {
            // Expected: sum_{r=0..n_ranks} (r+1) = 1 + 2 + ... + n_ranks = n*(n+1)/2
            let expected: f32 = ((n_ranks * (n_ranks + 1)) / 2) as f32;
            if (v - expected).abs() > 1e-5 {
                eprintln!("rank {} idx {}: expected {}, got {}", rank, i, expected, v);
                all_match = false;
            }
        }
    }
    assert!(all_match, "all_reduce result mismatch -- see stderr above");
    eprintln!(
        "[nccl] dual-P100 all_reduce sum verified: 1.0 + 2.0 = 3.0 across {} elements x {} ranks",
        n_elems, n_ranks,
    );

    nccl_real::aether_nccl_real_finalize();
}
