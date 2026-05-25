//! matt-voice FR-18.6-real leg 2 finisher #4 — GPU qwen3-block Stage trains
//! through the 1F1B pipeline driver.
//!
//! roadmap: P18
//!
//! Drives a real GPU QwenBlockStage (a 2-layer qwen3 block stack) through
//! `run_1f1b` and asserts the next-token cross-entropy loss decreases over
//! optimizer steps. world_size=1 (single process, single GPU — kokonoe has one
//! 3070 Ti); the actual 2-rank layer split is leg 3 on cnc's 2×P100. The loss
//! head (final RMSNorm + lm_head + cross-entropy) lives in the loss_and_grad
//! closure, keeping the Stage a pure block stack. num_microbatches=2 exercises
//! the stage's cross-microbatch grad ACCUMULATION (kernels overwrite their dst,
//! so the stage adds each microbatch's grad into a persistent accumulator before
//! the single optimizer step).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_alloc_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rms_norm_backward_dx_f32_cuda,
    aether_op_matmul_f32_cuda, aether_op_matmul_backward_lhs_f32_cuda,
    aether_op_cross_entropy_f32_cuda, aether_op_cross_entropy_backward_f32_cuda,
};
use trainer::pipeline::{PipeLinks, run_1f1b};
use trainer::qwen_stage::{BlockDims, QwenBlockStage};

const T: usize = 4;
const H: usize = 2;
const HD: usize = 4;
const D: usize = H * HD; // 8
const DFF: usize = 16;
const V: usize = 16;
const EPS: f32 = 1e-5;

fn ci(n: usize) -> c_int { n as c_int }

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

/// Fixed loss head: final RMSNorm (gamma=1) + lm_head -> logits -> CE. Weights
/// are held constant; only the block stack trains. Returns (loss, d_hidden).
struct LossHead {
    gf: i64,
    wlm: i64,
    targets: Vec<Vec<i32>>, // per microbatch
}
impl LossHead {
    fn new(targets: Vec<Vec<i32>>) -> Self {
        let gf = {
            let g = vec![1.0f32; D];
            let h = aether_dev_alloc_f32(ci(D));
            unsafe { aether_dev_h2d_f32(g.as_ptr() as i64, h, ci(D)); }
            h
        };
        let wlm = {
            let w = fill(17, D * V, 0.3);
            let h = aether_dev_alloc_f32(ci(D * V));
            unsafe { aether_dev_h2d_f32(w.as_ptr() as i64, h, ci(D * V)); }
            h
        };
        LossHead { gf, wlm, targets }
    }

    fn loss_and_grad(&self, mb: usize, hidden: &[f32]) -> (f32, Vec<f32>) {
        let td = T * D;
        // upload hidden
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
        // backward
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

#[test]
fn qwen_stage_trains_through_1f1b() {
    aether_dev_init();
    let dims = BlockDims { t: T, h: H, hd: HD, dff: DFF, base: 10000.0, eps: EPS };
    let mut stage = QwenBlockStage::build(dims, 2, 0..2, 0xA17E);

    // Two microbatches: fixed pseudo-embedded inputs + distinct target token ids.
    let inputs = [fill(1, T * D, 1.0), fill(2, T * D, 1.0)];
    let targets = vec![vec![2i32, 0, 3, 1], vec![5i32, 9, 1, 14]];
    let head = LossHead::new(targets);

    let links = PipeLinks::local_single();
    let lr = 5e-3f32;
    let epochs = 80usize;
    let mut first = 0.0f32;
    let mut last = 0.0f32;

    for ep in 0..epochs {
        let inp = inputs.clone();
        let losses = run_1f1b(
            &mut stage, 0, 1, &links, 2, lr, (ep + 1) as i64,
            |mb| inp[mb].clone(),
            |mb, out| head.loss_and_grad(mb, out),
        );
        let mean = losses.iter().sum::<f32>() / losses.len() as f32;
        if ep == 0 { first = mean; }
        if ep + 1 == epochs { last = mean; }
        if ep % 10 == 0 || ep + 1 == epochs {
            eprintln!("[qwen-stage PP train] epoch {:>3} mean loss = {:.5}", ep, mean);
        }
    }

    eprintln!("[qwen-stage PP train] loss {:.5} -> {:.5} over {} steps (2 layers, 2 microbatches/step)",
        first, last, epochs);
    assert!(first.is_finite() && last.is_finite(), "non-finite loss");
    assert!(last < first * 0.6,
        "loss did not decrease enough: {:.5} -> {:.5} (want < {:.5})", first, last, first * 0.6);
}
