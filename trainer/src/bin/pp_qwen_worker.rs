//! matt-voice FR-18.6-real leg 3 — multi-process pipeline-parallel worker.
//!
//! roadmap: P18
//!
//! One rank of a pipeline-parallel qwen3 training run. Spawned once per rank;
//! ranks rendezvous over localhost (or cross-host) TCP via connect_pipeline and
//! train a layer-split qwen3 block stack through run_1f1b. The GPU analog of the
//! leg-1 LinearReluStack thread-based witness: because the runtime CudaCtx is a
//! process-global singleton, each rank must be its OWN PROCESS (threads would
//! share one context + buffer pool). This is exactly the deployment shape for
//! the Qwen3-32B 2×P100 run on cnc — only `--host`, `--layers`, the dims, and
//! the GPU each rank binds change.
//!
//! Usage:
//!   pp_qwen_worker --rank R --world-size W --layers L [--base-port P]
//!                  [--host H] [--epochs N] [--seed S] [--lr LR] [--microbatch M]
//!
//! The last rank owns the loss head (final RMSNorm + lm_head + cross-entropy)
//! and prints, per epoch, `EPOCH <e> <mean_loss>`, then a final
//! `RESULT rank=<R> world=<W> final_loss=<x>` line the driver parses. A
//! world_size=1 invocation runs all layers in one process — the single-process
//! reference the pipelined run is compared against.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_alloc_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda, aether_op_rms_norm_backward_dx_f32_cuda,
    aether_op_matmul_f32_cuda, aether_op_matmul_backward_lhs_f32_cuda,
    aether_op_cross_entropy_f32_cuda, aether_op_cross_entropy_backward_f32_cuda,
};
use trainer::pipeline::{connect_pipeline, PipeLinks, run_1f1b};
use trainer::qwen_stage::{BlockDims, QwenBlockStage};

// Fixed model dims (small, runs in <1s; the cnc 32B run scales these up).
const T: usize = 8;
const H: usize = 4;
const HD: usize = 8;
const D: usize = H * HD;   // 32
const DFF: usize = 64;
const V: usize = 64;
const EPS: f32 = 1e-5;

fn ci(n: usize) -> c_int { n as c_int }

fn argval(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}
fn arg_usize(args: &[String], key: &str, def: usize) -> usize {
    argval(args, key).and_then(|s| s.parse().ok()).unwrap_or(def)
}
fn arg_f32(args: &[String], key: &str, def: f32) -> f32 {
    argval(args, key).and_then(|s| s.parse().ok()).unwrap_or(def)
}

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

/// Fixed loss head (deterministic, identical for ws=1 and ws=2 so the loss
/// trajectories are directly comparable). Only the last rank instantiates it.
struct LossHead { gf: i64, wlm: i64, targets: Vec<Vec<i32>> }
impl LossHead {
    fn new(microbatch: usize) -> Self {
        let g = vec![1.0f32; D];
        let gf = aether_dev_alloc_f32(ci(D));
        unsafe { aether_dev_h2d_f32(g.as_ptr() as i64, gf, ci(D)); }
        let w = fill(0x10551, D * V, 0.3);
        let wlm = aether_dev_alloc_f32(ci(D * V));
        unsafe { aether_dev_h2d_f32(w.as_ptr() as i64, wlm, ci(D * V)); }
        // Deterministic per-microbatch targets.
        let targets = (0..microbatch).map(|mb| {
            (0..T).map(|t| (((mb * 31 + t * 7 + 3) % V) as i32)).collect()
        }).collect();
        LossHead { gf, wlm, targets }
    }
    fn loss_and_grad(&self, mb: usize, hidden: &[f32]) -> (f32, Vec<f32>) {
        let td = T * D;
        let xb = aether_dev_alloc_f32(ci(td));
        unsafe { aether_dev_h2d_f32(hidden.as_ptr() as i64, xb, ci(td)); }
        let xf = aether_dev_alloc_f32(ci(td));
        aether_op_rms_norm_f32_cuda(xb, self.gf, xf, EPS, ci(T), ci(D));
        let logits = aether_dev_alloc_f32(ci(T * V));
        aether_op_matmul_f32_cuda(xf, self.wlm, logits, ci(T), ci(D), ci(V));
        let tgt = aether_dev_alloc_i32(ci(T));
        unsafe { aether_dev_h2d_i32(self.targets[mb].as_ptr() as i64, tgt, ci(T)); }
        let probs = aether_dev_alloc_f32(ci(T * V));
        let loss = aether_op_cross_entropy_f32_cuda(logits, tgt, probs, ci(T), ci(V));
        let d_logits = aether_dev_alloc_f32(ci(T * V));
        aether_op_cross_entropy_backward_f32_cuda(probs, tgt, d_logits, ci(T), ci(V));
        let d_xf = aether_dev_alloc_f32(ci(td));
        aether_op_matmul_backward_lhs_f32_cuda(d_logits, self.wlm, d_xf, ci(T), ci(D), ci(V));
        let d_xb = aether_dev_alloc_f32(ci(td));
        let inv = aether_dev_alloc_f32(ci(T));
        aether_op_rms_norm_backward_dx_f32_cuda(xb, self.gf, d_xf, d_xb, inv, EPS, ci(T), ci(D));
        let mut d_hidden = vec![0.0f32; td];
        unsafe { aether_dev_d2h_f32(d_xb, d_hidden.as_mut_ptr() as i64, ci(td)); aether_dev_sync(); }
        for h in [xb, xf, logits, probs, d_logits, d_xf, d_xb, inv] { aether_dev_free_f32(h); }
        (loss, d_hidden)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rank = arg_usize(&args, "--rank", 0);
    let world = arg_usize(&args, "--world-size", 1);
    let total_layers = arg_usize(&args, "--layers", 4);
    let base_port = arg_usize(&args, "--base-port", 29600) as i64;
    let host = argval(&args, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
    let epochs = arg_usize(&args, "--epochs", 40);
    let seed = arg_usize(&args, "--seed", 0xC0FFEE) as u64;
    let lr = arg_f32(&args, "--lr", 5e-3);
    let microbatch = arg_usize(&args, "--microbatch", 2);

    assert!(rank < world);
    assert!(total_layers % world == 0, "layers {} must split evenly across world {}", total_layers, world);
    let per = total_layers / world;
    let range = (rank * per)..((rank + 1) * per);
    let is_last = rank + 1 == world;

    aether_dev_init();
    eprintln!("[pp rank {}/{}] layers {:?} of {} (dims T={} D={} H={} DFF={} V={})",
        rank, world, range, total_layers, T, D, H, DFF, V);

    let dims = BlockDims { t: T, h: H, hd: HD, dff: DFF, base: 10000.0, eps: EPS };
    let mut stage = QwenBlockStage::build(dims, total_layers, range, seed);

    let links: PipeLinks = if world == 1 {
        PipeLinks::local_single()
    } else {
        connect_pipeline(rank, world, base_port, &host)
    };

    // Deterministic per-microbatch inputs (rank 0 only). Same for ws=1/ws=2.
    let inputs: Vec<Vec<f32>> = (0..microbatch)
        .map(|mb| fill(0x1000 + mb as u64, T * D, 1.0))
        .collect();
    let head = if is_last { Some(LossHead::new(microbatch)) } else { None };

    let mut first = 0.0f32;
    let mut last = 0.0f32;
    for ep in 0..epochs {
        let inp = inputs.clone();
        let losses = run_1f1b(
            &mut stage, rank, world, &links, microbatch, lr, (ep + 1) as i64,
            |mb| inp[mb].clone(),
            |mb, out| head.as_ref().unwrap().loss_and_grad(mb, out),
        );
        if is_last {
            let mean = losses.iter().sum::<f32>() / losses.len() as f32;
            if ep == 0 { first = mean; }
            last = mean;
            if ep % 10 == 0 || ep + 1 == epochs {
                println!("EPOCH {} {:.6}", ep, mean);
            }
        }
    }

    if is_last {
        println!("RESULT rank={} world={} first_loss={:.6} final_loss={:.6}", rank, world, first, last);
    }
    // Param checksum (sum of abs) lets the driver confirm the split trained.
    let params = stage.flat_params();
    let csum: f64 = params.iter().map(|v| v.abs() as f64).sum();
    println!("PARAMSUM rank={} world={} n={} sum_abs={:.6}", rank, world, params.len(), csum);
}
