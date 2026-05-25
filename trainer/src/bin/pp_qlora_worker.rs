//! matt-voice FR-18.6-real leg 3 — Qwen3-32B QLoRA pipeline-parallel worker.
//!
//! roadmap: P18
//!
//! One rank of a 2-way (or N-way) pipeline-parallel QLoRA training run on a real
//! Qwen3 GGUF. Each rank opens the GGUF, loads its contiguous layer slice
//! (frozen quant base) via QwenQLoraStage, and trains the LoRA adapters through
//! run_1f1b; activations/grads cross ranks over connect_pipeline TCP. The last
//! rank owns a synthetic loss head (final RMSNorm + small random lm_head + CE) —
//! enough to drive a real fwd/bwd/step over the actual 32B quant weights split
//! across two P100s and show a finite, decreasing loss. (A real fine-tune swaps
//! the synthetic head for the GGUF lm_head + real token data; the PP + QLoRA
//! machinery exercised here is identical.)
//!
//! Pin each rank to its GPU with CUDA_VISIBLE_DEVICES (the runtime CudaCtx uses
//! device 0 of whatever it sees). For the cnc 2xP100 run:
//!   CUDA_VISIBLE_DEVICES=0 pp-qlora-worker --rank 0 --world-size 2 --gguf ... &
//!   CUDA_VISIBLE_DEVICES=1 pp-qlora-worker --rank 1 --world-size 2 --gguf ...

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_sync, aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_alloc_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda, aether_op_rms_norm_backward_dx_f32_cuda,
    aether_op_matmul_f32_cuda, aether_op_matmul_backward_lhs_f32_cuda,
    aether_op_cross_entropy_f32_cuda, aether_op_cross_entropy_backward_f32_cuda,
};
use trainer::pipeline::{connect_pipeline, run_1f1b};
use trainer::qwen_qlora_stage::QwenQLoraStage;

const VS: usize = 256; // synthetic loss-head vocab
const EPS: f32 = 1e-5;

fn ci(n: usize) -> c_int { n as c_int }
fn argval(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned()
}
fn au(a: &[String], k: &str, d: usize) -> usize { argval(a, k).and_then(|s| s.parse().ok()).unwrap_or(d) }
fn af(a: &[String], k: &str, d: f32) -> f32 { argval(a, k).and_then(|s| s.parse().ok()).unwrap_or(d) }

fn fill(seed: u64, n: usize, s: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        (((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * s }).collect()
}

struct LossHead { gf: i64, wlm: i64, tgts: Vec<i64>, t: usize, d: usize }
impl LossHead {
    fn new(t: usize, d: usize, microbatch: usize) -> Self {
        let gf = { let g = vec![1.0f32; d]; let h = aether_dev_alloc_f32(ci(d));
            unsafe { aether_dev_h2d_f32(g.as_ptr() as i64, h, ci(d)); } h };
        let wlm = { let w = fill(0x105, d * VS, 0.02); let h = aether_dev_alloc_f32(ci(d * VS));
            unsafe { aether_dev_h2d_f32(w.as_ptr() as i64, h, ci(d * VS)); } h };
        let tgts = (0..microbatch).map(|mb| {
            let tg: Vec<i32> = (0..t).map(|i| ((mb * 29 + i * 37 + 11) % VS) as i32).collect();
            let h = aether_dev_alloc_i32(ci(t));
            unsafe { aether_dev_h2d_i32(tg.as_ptr() as i64, h, ci(t)); }
            h
        }).collect();
        LossHead { gf, wlm, tgts, t, d }
    }
    fn loss_and_grad(&self, mb: usize, hidden: &[f32]) -> (f32, Vec<f32>) {
        let (t, d) = (self.t, self.d); let td = t * d;
        let xb = aether_dev_alloc_f32(ci(td));
        unsafe { aether_dev_h2d_f32(hidden.as_ptr() as i64, xb, ci(td)); }
        let xf = aether_dev_alloc_f32(ci(td));
        aether_op_rms_norm_f32_cuda(xb, self.gf, xf, EPS, ci(t), ci(d));
        let logits = aether_dev_alloc_f32(ci(t * VS));
        aether_op_matmul_f32_cuda(xf, self.wlm, logits, ci(t), ci(d), ci(VS));
        let probs = aether_dev_alloc_f32(ci(t * VS));
        let loss = aether_op_cross_entropy_f32_cuda(logits, self.tgts[mb], probs, ci(t), ci(VS));
        let d_logits = aether_dev_alloc_f32(ci(t * VS));
        aether_op_cross_entropy_backward_f32_cuda(probs, self.tgts[mb], d_logits, ci(t), ci(VS));
        let d_xf = aether_dev_alloc_f32(ci(td));
        aether_op_matmul_backward_lhs_f32_cuda(d_logits, self.wlm, d_xf, ci(t), ci(d), ci(VS));
        let d_xb = aether_dev_alloc_f32(ci(td));
        let inv = aether_dev_alloc_f32(ci(t));
        aether_op_rms_norm_backward_dx_f32_cuda(xb, self.gf, d_xf, d_xb, inv, EPS, ci(t), ci(d));
        let mut dh = vec![0.0f32; td];
        unsafe { aether_dev_d2h_f32(d_xb, dh.as_mut_ptr() as i64, ci(td)); aether_dev_sync(); }
        for h in [xb, xf, logits, probs, d_logits, d_xf, d_xb, inv] { aether_dev_free_f32(h); }
        (loss, dh)
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let rank = au(&a, "--rank", 0);
    let world = au(&a, "--world-size", 2);
    let gguf = argval(&a, "--gguf").expect("--gguf PATH required");
    let base_port = au(&a, "--base-port", 29700) as i64;
    let host = argval(&a, "--host").unwrap_or_else(|| "127.0.0.1".to_string());
    let steps = au(&a, "--steps", 20);
    let t = au(&a, "--t", 8);
    let lora_rank = au(&a, "--lora-rank", 8);
    let alpha = af(&a, "--alpha", 16.0);
    let lr = af(&a, "--lr", 1e-3);
    let microbatch = au(&a, "--microbatch", 2);
    let seed = au(&a, "--seed", 0xC0FFEE) as u64;

    assert!(rank < world);
    // Build the stage for this rank's layer slice. The stage reads cfg from the
    // GGUF; we split cfg.n_layers evenly across world.
    // Peek layer count first via a throwaway full-config open is avoided — build
    // with a provisional range then re-split: instead, open once for cfg.
    let (_, cfg_peek) = unsafe { aether_rt::serving::open_gguf_config(&gguf).expect("open gguf") };
    let total = cfg_peek.n_layers;
    // --splits "28,36" gives per-rank layer counts (must sum to total + len==world);
    // de-risks an uneven-VRAM split (fewer layers on the smaller card). Absent =
    // even split.
    let range = if let Some(splits_s) = argval(&a, "--splits") {
        let counts: Vec<usize> = splits_s.split(',').map(|s| s.trim().parse().expect("--splits int")).collect();
        assert_eq!(counts.len(), world, "--splits len {} != world {}", counts.len(), world);
        assert_eq!(counts.iter().sum::<usize>(), total, "--splits sum {} != n_layers {}", counts.iter().sum::<usize>(), total);
        let lo: usize = counts[..rank].iter().sum();
        lo..(lo + counts[rank])
    } else {
        assert!(total % world == 0, "n_layers {} not divisible by world {}", total, world);
        let per = total / world;
        (rank * per)..((rank + 1) * per)
    };
    let is_last = rank + 1 == world;
    eprintln!("[pp-qlora rank {}/{}] layers {:?} of {} (gguf={})", rank, world, range, total, gguf);

    let mut stage = QwenQLoraStage::build(&gguf, range, t, lora_rank, alpha, seed + rank as u64)
        .expect("build qlora stage");
    let d = stage.cfg.d_model;
    eprintln!("[pp-qlora rank {}] d_model={} adapter_params={} t={} lora_rank={} alpha={}",
        rank, d, stage.total_adapter_params(), t, lora_rank, alpha);

    let links = connect_pipeline(rank, world, base_port, &host);
    let inputs: Vec<Vec<f32>> = (0..microbatch).map(|mb| fill(0x2000 + mb as u64, t * d, 1.0)).collect();
    let head = if is_last { Some(LossHead::new(t, d, microbatch)) } else { None };

    let b0 = stage.adapter_b_abs_sum();
    let mut first = 0.0f32; let mut last = 0.0f32;
    for s in 0..steps {
        let inp = inputs.clone();
        let losses = run_1f1b(&mut stage, rank, world, &links, microbatch, lr, (s + 1) as i64,
            |mb| inp[mb].clone(),
            |mb, out| head.as_ref().unwrap().loss_and_grad(mb, out));
        if is_last {
            let mean = losses.iter().sum::<f32>() / losses.len() as f32;
            if s == 0 { first = mean; }
            last = mean;
            println!("STEP {} {:.6}", s, mean);
        }
    }
    let b1 = stage.adapter_b_abs_sum();
    if is_last {
        println!("RESULT rank={} world={} first={:.6} final={:.6}", rank, world, first, last);
    }
    println!("BSUM rank={} world={} b0={:.4} b1={:.4} layers={}", rank, world, b0, b1, stage.n_layers());
}
