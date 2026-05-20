//! Data-parallel training loop across N GPUs in one process via NCCL.
//!
//! Each rank holds its own `Model` (host-side `Vec<f32>` for params +
//! grads + Adam state) AND a CudaSlice on its assigned GPU for the
//! gradient exchange. Per step:
//! 1. Sample batch shard (rank r sees offset `(step * world_size + r)`).
//! 2. CPU forward + backward (same as the single-rank loop).
//! 3. h2d each rank's host grads to its GPU's device buffer.
//! 4. group_start / per-rank all_reduce(Sum) / group_end -- both ranks
//!    end up with the SAME summed gradients on their device.
//! 5. d2h reduced grads back to host.
//! 6. Scale by 1/world_size to get the mean.
//! 7. Each rank applies the same AdamW step.
//!
//! Loss is averaged across ranks for logging.

use crate::config::{ModelConfig, TrainConfig};
use crate::data::ByteDataset;
use crate::model::{adamw_step, backward, clip_grads, forward, Model};
use crate::rng::Rng;

use cudarc::driver::CudaDevice;
use cudarc::nccl::safe::{ReduceOp, group_start, group_end};

pub struct DpHandle {
    pub rank: usize,
    pub world_size: usize,
    pub device_ordinal: usize,
}

/// Run data-parallel training. Returns the final per-step loss-trace
/// (one entry per logged step) on rank 0. Other ranks return empty.
pub fn train_dp(
    cfg: ModelConfig,
    train: TrainConfig,
    world_size: usize,
    dataset: ByteDataset,
) -> std::io::Result<Vec<(usize, f32)>> {
    assert!(world_size >= 2, "train_dp wants world_size >= 2");

    // 1) Bring up NCCL.
    let rc = aether_rt::nccl_real::aether_nccl_real_init_multi_gpu(world_size as i32);
    if rc != world_size as i32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("nccl_real_init_multi_gpu({}) returned {}", world_size, rc),
        ));
    }

    // 2) Create one Model per rank. Same seed -> identical initial weights.
    let mut models: Vec<Model> = (0..world_size).map(|_| Model::new(cfg.clone(), train.seed)).collect();

    // 3) Per-rank RNG for batch sampling (distinct so ranks see different shards).
    let mut rngs: Vec<Rng> = (0..world_size)
        .map(|r| Rng::new(train.seed.wrapping_add(0xA17C ^ (r as u64) * 0x9E37_79B9_7F4A_7C15)))
        .collect();

    // 4) Per-rank device gradient buffers (allocated on each comm's device).
    let n_grads = models[0].grads.len();
    let mut dev_grad_send: Vec<cudarc::driver::CudaSlice<f32>> = Vec::with_capacity(world_size);
    let mut dev_grad_recv: Vec<cudarc::driver::CudaSlice<f32>> = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let comm = aether_rt::nccl_real::comm_at(rank)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other,
                format!("no comm at rank {}", rank)))?;
        let dev = comm.device();
        let send = dev.alloc_zeros::<f32>(n_grads).expect("alloc send");
        let recv = dev.alloc_zeros::<f32>(n_grads).expect("alloc recv");
        dev_grad_send.push(send);
        dev_grad_recv.push(recv);
    }

    let t0 = std::time::Instant::now();
    let mut loss_trace: Vec<(usize, f32)> = Vec::new();
    let mut running = 0.0f64;
    let mut running_count = 0usize;

    for step in 0..train.steps {
        let lr = cosine_lr(step, train.steps, train.lr, train.warmup);

        // Per-rank forward + backward.
        let mut per_rank_loss = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let (ids, labels) = dataset.sample_batch(train.batch_size, &mut rngs[rank]);
            let (act, loss) = forward(&models[rank], &ids, &labels, train.batch_size);
            backward(&mut models[rank], &act, &ids, &labels);
            per_rank_loss.push(loss);
        }

        // Per-rank: h2d host grads to device send buffer.
        for rank in 0..world_size {
            let comm = aether_rt::nccl_real::comm_at(rank).expect("comm");
            let dev = comm.device();
            dev.htod_sync_copy_into(&models[rank].grads, &mut dev_grad_send[rank])
                .expect("h2d grads");
        }

        // All-reduce SUM across ranks via NCCL group mode.
        group_start().expect("group_start");
        for rank in 0..world_size {
            let comm = aether_rt::nccl_real::comm_at(rank).expect("comm");
            comm.all_reduce(&dev_grad_send[rank], &mut dev_grad_recv[rank], &ReduceOp::Sum)
                .unwrap_or_else(|e| panic!("rank {} all_reduce: {:?}", rank, e));
        }
        group_end().expect("group_end");

        // Per-rank: d2h reduced grads, scale by 1/world_size, AdamW.
        let inv_ws = 1.0f32 / world_size as f32;
        for rank in 0..world_size {
            let comm = aether_rt::nccl_real::comm_at(rank).expect("comm");
            let dev = comm.device();
            let reduced: Vec<f32> = dev.dtoh_sync_copy(&dev_grad_recv[rank]).expect("d2h grads");
            // Scale + write back into the model's host grads.
            for (g, r) in models[rank].grads.iter_mut().zip(reduced.iter()) {
                *g = r * inv_ws;
            }
            let _norm = clip_grads(&mut models[rank], train.grad_clip);
            adamw_step(&mut models[rank], lr, 0.9, 0.95, 1e-8, train.weight_decay, (step + 1) as i64);
        }

        // Loss averaged across ranks.
        let mean_loss: f32 = per_rank_loss.iter().sum::<f32>() / world_size as f32;
        running += mean_loss as f64;
        running_count += 1;

        if step % train.log_every == 0 || step + 1 == train.steps {
            let avg = running / running_count.max(1) as f64;
            let elapsed = t0.elapsed().as_secs_f32();
            eprintln!(
                "[aether-train-dp ws={}] step={:>5} loss={:.4} lr={:.2e} elapsed={:.1}s",
                world_size, step, avg, lr, elapsed,
            );
            loss_trace.push((step, avg as f32));
            running = 0.0;
            running_count = 0;
        }
    }

    // Teardown NCCL.
    aether_rt::nccl_real::aether_nccl_real_finalize();

    // Optional final-state check: all ranks must hold identical weights
    // (this is the data-parallel invariant -- AdamW + same grads + same
    // init = same params). Sample the first few params for a sanity log.
    if world_size >= 2 {
        let sample = std::cmp::min(8, models[0].params.len());
        let mut all_eq = true;
        for r in 1..world_size {
            for i in 0..sample {
                if (models[r].params[i] - models[0].params[i]).abs() > 1e-5 {
                    all_eq = false;
                    break;
                }
            }
        }
        eprintln!(
            "[aether-train-dp] final params identical across ranks: {} (sampled first {} of {})",
            all_eq, sample, models[0].params.len(),
        );
    }

    Ok(loss_trace)
}

fn cosine_lr(step: usize, max_steps: usize, lr_max: f32, warmup: usize) -> f32 {
    if step < warmup { return lr_max * (step + 1) as f32 / warmup.max(1) as f32; }
    let p = (step - warmup) as f32 / (max_steps - warmup).max(1) as f32;
    lr_max * 0.5 * (1.0 + (std::f32::consts::PI * p).cos())
}
