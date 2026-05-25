//! Data-parallel wrapper for LoRA adapters.
//!
//! Mirrors `dp.rs`'s gradient-exchange pattern, but instead of the whole model
//! arena it exchanges only the (tiny) concatenated adapter gradients. Per step,
//! after each rank has run forward + backward and accumulated `grad_a`/`grad_b`
//! on its own adapter set, call `all_reduce_and_step`:
//!
//!   1. Flatten every adapter's [dA; dB] into one contiguous host buffer.
//!   2. All-reduce SUM across ranks:
//!        - `nccl` feature: NCCL group_start / all_reduce(Sum) / group_end on
//!          the device buffer for each rank (same as dp.rs).
//!        - no `nccl`: world_size==1 fast path is a no-op (single rank already
//!          holds the full gradient); world_size>1 without NCCL is unsupported.
//!   3. Scale by 1/world_size to get the mean.
//!   4. Scatter the reduced gradients back into each adapter.
//!   5. AdamW step per adapter.
//!
//! Because adapters are small, the entire exchange is one flat all-reduce — no
//! per-adapter latency tax.

use crate::lora::LoraAdapter;

/// Flatten all adapter gradients into one contiguous buffer ([dA;dB] per
/// adapter, in order).
pub fn flatten_grads(adapters: &[LoraAdapter]) -> Vec<f32> {
    let total: usize = adapters.iter().map(|a| a.n_params()).sum();
    let mut flat = Vec::with_capacity(total);
    for a in adapters {
        flat.extend_from_slice(&a.grad_a);
        flat.extend_from_slice(&a.grad_b);
    }
    flat
}

/// Scatter a flat gradient buffer back into each adapter (inverse of
/// `flatten_grads`).
pub fn scatter_grads(adapters: &mut [LoraAdapter], flat: &[f32]) {
    let mut off = 0usize;
    for a in adapters.iter_mut() {
        let n = a.n_params();
        a.grads_flat_mut(&flat[off..off + n]);
        off += n;
    }
    debug_assert_eq!(off, flat.len(), "scatter_grads: length mismatch");
}

/// CPU all-reduce fallback. For `world_size == 1` this is a no-op (the single
/// rank already holds the full gradient). `world_size > 1` without the `nccl`
/// feature is unsupported — the multi-rank path needs a real collective.
#[cfg(not(feature = "nccl"))]
pub fn all_reduce_sum_inplace(buf: &mut [f32], world_size: usize) {
    if world_size <= 1 {
        eprintln!("[lora-dp] all-reduce: world_size=1 no-op ({} elems)", buf.len());
        return;
    }
    panic!(
        "[lora-dp] world_size={} requested but built without the `nccl` feature; \
         rebuild with --features nccl for multi-rank all-reduce",
        world_size,
    );
}

/// NCCL all-reduce SUM across `world_size` in-process ranks. Each rank's comm
/// must already be initialised via `aether_nccl_real_init_multi_gpu`. The same
/// `buf` is reduced for every rank (single-process DP: one host buffer, summed
/// across all device comms). Mirrors the dp.rs group_start/all_reduce/group_end.
#[cfg(feature = "nccl")]
pub fn all_reduce_sum_inplace(buf: &mut [f32], world_size: usize) {
    use cudarc::nccl::safe::{ReduceOp, group_start, group_end};

    if world_size <= 1 {
        eprintln!("[lora-dp] all-reduce: world_size=1 no-op ({} elems)", buf.len());
        return;
    }

    let n = buf.len();

    // h2d the same host buffer to each rank's device send buffer.
    let mut sends = Vec::with_capacity(world_size);
    let mut recvs = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let comm = aether_rt::nccl_real::comm_at(rank)
            .unwrap_or_else(|| panic!("[lora-dp] no comm at rank {}", rank));
        let dev = comm.device();
        let send = dev.htod_sync_copy(buf).expect("[lora-dp] h2d send");
        let recv = dev.alloc_zeros::<f32>(n).expect("[lora-dp] alloc recv");
        sends.push(send);
        recvs.push(recv);
    }

    group_start().expect("[lora-dp] group_start");
    for rank in 0..world_size {
        let comm = aether_rt::nccl_real::comm_at(rank).expect("comm");
        comm.all_reduce(&sends[rank], &mut recvs[rank], &ReduceOp::Sum)
            .unwrap_or_else(|e| panic!("[lora-dp] rank {} all_reduce: {:?}", rank, e));
    }
    group_end().expect("[lora-dp] group_end");

    // d2h rank 0's reduced buffer back into the host buffer (all ranks hold the
    // same summed result post-all-reduce).
    let comm0 = aether_rt::nccl_real::comm_at(0).expect("comm0");
    let dev0 = comm0.device();
    let reduced: Vec<f32> = dev0.dtoh_sync_copy(&recvs[0]).expect("[lora-dp] d2h reduced");
    buf.copy_from_slice(&reduced);
}

/// Full DP step: flatten adapter grads, all-reduce SUM, scale by 1/world_size,
/// scatter back, then AdamW per adapter. After this call each adapter's params
/// reflect the mean gradient across all ranks.
pub fn all_reduce_and_step(
    adapters: &mut [LoraAdapter],
    world_size: usize,
    lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: i64,
) {
    let mut flat = flatten_grads(adapters);
    eprintln!(
        "[lora-dp] step={} world_size={} reducing {} adapter grad elems across {} adapters",
        step, world_size, flat.len(), adapters.len(),
    );

    all_reduce_sum_inplace(&mut flat, world_size);

    // Mean.
    if world_size > 1 {
        let inv = 1.0f32 / world_size as f32;
        for g in flat.iter_mut() { *g *= inv; }
    }

    scatter_grads(adapters, &flat);

    for a in adapters.iter_mut() {
        a.adamw_step(lr, b1, b2, eps, wd, step);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// world_size=1 (no-op all-reduce). Run one full DP step on a tiny adapter
    /// set and assert params actually changed.
    #[test]
    fn lora_dp_world1_step_changes_params() {
        let mut rng = Rng::new(1234);
        let mut adapters = vec![
            LoraAdapter::new("attn.q", 8, 8, 4, 2.0, &mut rng),
            LoraAdapter::new("attn.v", 8, 8, 4, 2.0, &mut rng),
        ];
        // Make B non-zero so a fabricated grad actually moves params through the
        // optimiser (init B=0 is fine — AdamW moves any param with nonzero grad).
        for a in adapters.iter_mut() {
            for x in a.b.iter_mut() { *x = rng.next_normal() * 0.1; }
        }

        // Snapshot params.
        let a0_before = adapters[0].a.clone();
        let b0_before = adapters[0].b.clone();

        // Fabricate gradients (as if backward had run).
        for a in adapters.iter_mut() {
            a.zero_grad();
            for (i, g) in a.grad_a.iter_mut().enumerate() { *g = 0.01 * (i as f32 + 1.0); }
            for (i, g) in a.grad_b.iter_mut().enumerate() { *g = 0.01 * (i as f32 + 1.0); }
        }

        all_reduce_and_step(&mut adapters, 1, 1e-2, 0.9, 0.95, 1e-8, 0.0, 1);

        let mut changed_a = false;
        for (x, y) in adapters[0].a.iter().zip(a0_before.iter()) {
            if (x - y).abs() > 1e-9 { changed_a = true; break; }
        }
        let mut changed_b = false;
        for (x, y) in adapters[0].b.iter().zip(b0_before.iter()) {
            if (x - y).abs() > 1e-9 { changed_b = true; break; }
        }
        assert!(changed_a, "adapter A params did not change after DP step");
        assert!(changed_b, "adapter B params did not change after DP step");
    }

    #[test]
    fn lora_dp_flatten_scatter_roundtrip() {
        let mut rng = Rng::new(99);
        let mut adapters = vec![
            LoraAdapter::new("l0", 5, 3, 2, 1.0, &mut rng),
            LoraAdapter::new("l1", 4, 6, 3, 1.0, &mut rng),
        ];
        for (k, a) in adapters.iter_mut().enumerate() {
            for (i, g) in a.grad_a.iter_mut().enumerate() { *g = (k * 1000 + i) as f32; }
            for (i, g) in a.grad_b.iter_mut().enumerate() { *g = (k * 1000 + 500 + i) as f32; }
        }
        let flat = flatten_grads(&adapters);
        let total: usize = adapters.iter().map(|a| a.n_params()).sum();
        assert_eq!(flat.len(), total);

        // Zero then scatter back; must recover.
        let mut adapters2 = adapters.clone();
        for a in adapters2.iter_mut() { a.zero_grad(); }
        scatter_grads(&mut adapters2, &flat);
        for (a, b) in adapters.iter().zip(adapters2.iter()) {
            assert_eq!(a.grad_a, b.grad_a);
            assert_eq!(a.grad_b, b.grad_b);
        }
    }
}
