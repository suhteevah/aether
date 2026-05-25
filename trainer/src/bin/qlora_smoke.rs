//! matt-voice FR-18.6-real leg 3 — single-process QLoRA stage smoke.
//!
//! roadmap: P18
//!
//! Loads a layer range of a real Qwen GGUF through QwenQLoraStage (frozen quant
//! base + trainable LoRA adapters) and trains the adapters against a synthetic
//! loss head (final RMSNorm + random lm_head + cross-entropy over a small
//! synthetic vocab — the head is just a fixed scalar-loss generator). Asserts
//! the loss is finite and DECREASES and that adapter |B| grows from 0 (proving
//! gradients flow into the adapters). Validates the quant-load + dequant +
//! base-proj + GQA + adapter fwd/bwd mechanics before the 32B 2xP100 cnc run.
//!
//! Usage: qlora-smoke --gguf PATH [--lo 0 --hi 2 --t 8 --steps 30 --rank 8 --lr 5e-3]

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_sync, aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_alloc_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda, aether_op_rms_norm_backward_dx_f32_cuda,
    aether_op_matmul_f32_cuda, aether_op_matmul_backward_lhs_f32_cuda,
    aether_op_cross_entropy_f32_cuda, aether_op_cross_entropy_backward_f32_cuda,
};
use trainer::pipeline::Stage;
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

struct LossHead { gf: i64, wlm: i64, tgt: i64, t: usize, d: usize }
impl LossHead {
    fn new(t: usize, d: usize) -> Self {
        let gf = { let g = vec![1.0f32; d]; let h = aether_dev_alloc_f32(ci(d));
            unsafe { aether_dev_h2d_f32(g.as_ptr() as i64, h, ci(d)); } h };
        let wlm = { let w = fill(0x105, d * VS, 0.02); let h = aether_dev_alloc_f32(ci(d * VS));
            unsafe { aether_dev_h2d_f32(w.as_ptr() as i64, h, ci(d * VS)); } h };
        let tg: Vec<i32> = (0..t).map(|i| ((i * 37 + 11) % VS) as i32).collect();
        let tgt = aether_dev_alloc_i32(ci(t));
        unsafe { aether_dev_h2d_i32(tg.as_ptr() as i64, tgt, ci(t)); }
        LossHead { gf, wlm, tgt, t, d }
    }
    fn loss_and_grad(&self, hidden: &[f32]) -> (f32, Vec<f32>) {
        let (t, d) = (self.t, self.d); let td = t * d;
        let xb = aether_dev_alloc_f32(ci(td));
        unsafe { aether_dev_h2d_f32(hidden.as_ptr() as i64, xb, ci(td)); }
        let xf = aether_dev_alloc_f32(ci(td));
        aether_op_rms_norm_f32_cuda(xb, self.gf, xf, EPS, ci(t), ci(d));
        let logits = aether_dev_alloc_f32(ci(t * VS));
        aether_op_matmul_f32_cuda(xf, self.wlm, logits, ci(t), ci(d), ci(VS));
        let probs = aether_dev_alloc_f32(ci(t * VS));
        let loss = aether_op_cross_entropy_f32_cuda(logits, self.tgt, probs, ci(t), ci(VS));
        let d_logits = aether_dev_alloc_f32(ci(t * VS));
        aether_op_cross_entropy_backward_f32_cuda(probs, self.tgt, d_logits, ci(t), ci(VS));
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
    let gguf = argval(&a, "--gguf").expect("--gguf PATH required");
    let lo = au(&a, "--lo", 0); let hi = au(&a, "--hi", 2);
    let t = au(&a, "--t", 8); let steps = au(&a, "--steps", 30);
    let rank = au(&a, "--rank", 8); let alpha = af(&a, "--alpha", 16.0);
    let lr = af(&a, "--lr", 5e-3);

    let mut stage = QwenQLoraStage::build(&gguf, lo..hi, t, rank, alpha, 0xA1).expect("build stage");
    let d = stage.cfg.d_model;
    eprintln!("[qlora-smoke] {} layers, {} adapter params, d={} t={} rank={} alpha={}",
        stage.n_layers(), stage.total_adapter_params(), d, t, rank, alpha);
    let head = LossHead::new(t, d);
    let input = fill(0x1234, t * d, 1.0);

    let b0 = stage.adapter_b_abs_sum();
    let mut first = 0.0f32; let mut last = 0.0f32;
    for s in 0..steps {
        let out = stage.forward(&input);
        let (loss, dh) = head.loss_and_grad(&out);
        let _d_in = stage.backward(&dh);
        stage.step(lr, (s + 1) as i64);
        if s == 0 { first = loss; }
        last = loss;
        if s % 5 == 0 || s + 1 == steps {
            eprintln!("[qlora-smoke] step {:>3} loss {:.5}", s, loss);
        }
    }
    let b1 = stage.adapter_b_abs_sum();
    eprintln!("[qlora-smoke] loss {:.5} -> {:.5} over {} steps; adapter |B| {:.4} -> {:.4}",
        first, last, steps, b0, b1);
    assert!(first.is_finite() && last.is_finite(), "non-finite loss");
    assert!(b0 == 0.0, "adapter B should start at 0, was {}", b0);
    assert!(b1 > 0.0, "adapter B did not move — no grad flow");
    assert!(last < first, "loss did not decrease: {:.5} -> {:.5}", first, last);
    println!("QLORA_SMOKE_OK first={:.5} final={:.5} bsum={:.4}", first, last, b1);
}
