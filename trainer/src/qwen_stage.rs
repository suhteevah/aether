//! GPU qwen3-block pipeline stage — FR-18.6-real leg 2 finisher #4.
//!
//! roadmap: P18
//!
//! Implements the `Stage` trait (pipeline.rs) with a stack of real qwen3-style
//! transformer blocks running on the GPU through the leg-2 kernels (the same
//! ones finite-diff verified in runtime/tests/cuda_qwen3_block_grad_check.rs).
//! This is the graft the pipeline module's docstring anticipates: the 1F1B
//! scheduler is unchanged; only the `Stage` impl swaps from the CPU
//! `LinearReluStack` to this GPU block stack.
//!
//! Each stage owns a contiguous range of layers. Weights + AdamW moment buffers
//! live persistently on the device; per-microbatch forward activations are
//! stashed in a FIFO (1F1B keeps `warmup+1` microbatches in flight, so backward
//! pops them in forward order). The loss head (final norm + lm_head + CE) is NOT
//! part of the stage — it lives in run_1f1b's `loss_and_grad` closure on the
//! last rank, keeping the stage a pure block stack (the PP-parallel unit).
//!
//! MHA (n_kv == n_q); the GQA repeat/reduce pair (gqa_repeat_kv /
//! gqa_reduce_kv_grad) is wired separately and witnessed in
//! cuda_qwen3_gqa_block_grad_check.rs.

#![cfg(feature = "cuda")]

use std::collections::VecDeque;
use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_dev_sync,
    aether_op_rms_norm_f32_cuda,
    aether_op_rms_norm_backward_dx_f32_cuda, aether_op_rms_norm_backward_gamma_f32_cuda,
    aether_op_matmul_f32_cuda,
    aether_op_matmul_backward_lhs_f32_cuda, aether_op_matmul_backward_rhs_f32_cuda,
    aether_op_rope_apply_f32_cuda, aether_op_rope_apply_backward_f32_cuda,
    aether_op_sdpa_causal_forward_f32_cuda, aether_op_sdpa_causal_backward_f32_cuda,
    aether_op_transpose_021_f32_cuda,
    aether_op_silu_f32_cuda, aether_op_silu_backward_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_adamw_step_f32_cuda,
};

use crate::pipeline::Stage;
use crate::rng::Rng;

/// Block dimensions. T = sequence length, D = model dim = H*HD.
#[derive(Clone, Copy)]
pub struct BlockDims {
    pub t: usize,
    pub h: usize,
    pub hd: usize,
    pub dff: usize,
    pub base: f32,
    pub eps: f32,
}
impl BlockDims {
    pub fn d(&self) -> usize { self.h * self.hd }
}

fn ci(n: usize) -> c_int { n as c_int }

/// One device buffer with a checked alloc; freed explicitly via `free`.
fn alloc(n: usize) -> i64 {
    let h = aether_dev_alloc_f32(ci(n));
    assert!(h >= 0, "[qwen_stage] alloc {} failed", n);
    h
}
fn upload(host: &[f32]) -> i64 {
    let h = alloc(host.len());
    unsafe { aether_dev_h2d_f32(host.as_ptr() as i64, h, ci(host.len())); }
    h
}
fn download(h: i64, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    unsafe { aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, ci(n)); }
    v
}

/// Persistent per-layer weights + AdamW state, all on device. Names mirror the
/// grad-check tests. Weights stored [in, out] so matmul_f32 applies directly.
struct LayerWeights {
    // params
    g1: i64, wq: i64, wk: i64, wv: i64, wo: i64,
    g2: i64, wgate: i64, wup: i64, wdown: i64,
    // grads (accumulated each backward, zeroed by realloc at step)
    d_g1: i64, d_wq: i64, d_wk: i64, d_wv: i64, d_wo: i64,
    d_g2: i64, d_wgate: i64, d_wup: i64, d_wdown: i64,
    // adamw moments (m, v) per param, persistent
    m: AdamMoments, v: AdamMoments,
}

struct AdamMoments {
    g1: i64, wq: i64, wk: i64, wv: i64, wo: i64,
    g2: i64, wgate: i64, wup: i64, wdown: i64,
}
impl AdamMoments {
    fn zeros(d: usize, dd: usize, ddff: usize, dffd: usize) -> Self {
        AdamMoments {
            g1: alloc(d), wq: alloc(dd), wk: alloc(dd), wv: alloc(dd), wo: alloc(dd),
            g2: alloc(d), wgate: alloc(ddff), wup: alloc(ddff), wdown: alloc(dffd),
        }
    }
}

/// Forward activations saved per microbatch per layer for the backward pass.
struct LayerAct {
    xd: i64,                     // block input
    xn: i64,                     // rmsnorm1 output
    qt: i64, kt: i64, vt: i64,   // [h,s,hd] q/k/v
    attn: i64,                   // softmax probs [h,s,s]
    o: i64,                      // attn out [s,d] (post transpose-back)
    x1: i64,                     // attn residual
    xn2: i64,                    // rmsnorm2 output
    gate_pre: i64, gate_act: i64, up: i64, h_ffn: i64,
}
impl LayerAct {
    fn free(&self) {
        for &h in &[self.xd, self.xn, self.qt, self.kt, self.vt, self.attn,
                    self.o, self.x1, self.xn2, self.gate_pre, self.gate_act,
                    self.up, self.h_ffn] {
            aether_dev_free_f32(h);
        }
    }
}

/// A stack of qwen3 blocks for a contiguous layer range, as a pipeline `Stage`.
pub struct QwenBlockStage {
    dims: BlockDims,
    layers: Vec<LayerWeights>,
    /// FIFO of saved forward contexts, one Vec<LayerAct> (per layer) per
    /// in-flight microbatch.
    fifo: VecDeque<Vec<LayerAct>>,
}

impl QwenBlockStage {
    /// Build the stage holding global layers in `range`, drawing weights from a
    /// single deterministic RNG over ALL `total_layers` so any slice matches the
    /// full-stack reference (each rank advances the RNG identically, keeps its
    /// slice). Mirrors LinearReluStack::build.
    pub fn build(dims: BlockDims, total_layers: usize,
                 range: std::ops::Range<usize>, seed: u64) -> Self {
        let d = dims.d();
        let dd = d * d;
        let ddff = d * dims.dff;
        let dffd = dims.dff * d;
        let mut rng = Rng::new(seed);
        let mut layers = Vec::new();
        for layer in 0..total_layers {
            // Draw every layer's weights so the RNG stream is range-independent.
            let g1 = vec![1.0f32; d];
            let wq = draw(&mut rng, dd, 0.3); let wk = draw(&mut rng, dd, 0.3);
            let wv = draw(&mut rng, dd, 0.3); let wo = draw(&mut rng, dd, 0.3);
            let g2 = vec![1.0f32; d];
            let wgate = draw(&mut rng, ddff, 0.2); let wup = draw(&mut rng, ddff, 0.2);
            let wdown = draw(&mut rng, dffd, 0.2);
            if range.contains(&layer) {
                layers.push(LayerWeights {
                    g1: upload(&g1), wq: upload(&wq), wk: upload(&wk),
                    wv: upload(&wv), wo: upload(&wo), g2: upload(&g2),
                    wgate: upload(&wgate), wup: upload(&wup), wdown: upload(&wdown),
                    d_g1: alloc(d), d_wq: alloc(dd), d_wk: alloc(dd), d_wv: alloc(dd),
                    d_wo: alloc(dd), d_g2: alloc(d), d_wgate: alloc(ddff),
                    d_wup: alloc(ddff), d_wdown: alloc(dffd),
                    m: AdamMoments::zeros(d, dd, ddff, dffd),
                    v: AdamMoments::zeros(d, dd, ddff, dffd),
                });
            }
        }
        QwenBlockStage { dims, layers, fifo: VecDeque::new() }
    }

    pub fn n_layers(&self) -> usize { self.layers.len() }

    /// Flat snapshot of all params (for parity checks). Order: per layer
    /// g1,wq,wk,wv,wo,g2,wgate,wup,wdown.
    pub fn flat_params(&self) -> Vec<f32> {
        let d = self.dims.d();
        let dd = d * d; let ddff = d * self.dims.dff; let dffd = self.dims.dff * d;
        let mut out = Vec::new();
        for lw in &self.layers {
            out.extend(download(lw.g1, d));
            out.extend(download(lw.wq, dd)); out.extend(download(lw.wk, dd));
            out.extend(download(lw.wv, dd)); out.extend(download(lw.wo, dd));
            out.extend(download(lw.g2, d));
            out.extend(download(lw.wgate, ddff)); out.extend(download(lw.wup, ddff));
            out.extend(download(lw.wdown, dffd));
        }
        unsafe { aether_dev_sync(); }
        out
    }

    /// Forward one layer; returns (output_handle, saved_activations). Caller owns
    /// both handles. `xin` is consumed-by-reference (not freed here — it is saved
    /// as `act.xd` for the backward, except the very first layer's input which is
    /// the stage input and freed by the caller after all layers run... handled by
    /// keeping each layer's xd in its LayerAct).
    fn layer_forward(&self, lw: &LayerWeights, xin: i64) -> (i64, LayerAct) {
        let dm = &self.dims;
        let t = dm.t; let h = dm.h; let hd = dm.hd; let d = dm.d(); let dff = dm.dff;
        let td = t * d;
        let xn = alloc(td);
        aether_op_rms_norm_f32_cuda(xin, lw.g1, xn, dm.eps, ci(t), ci(d));
        let q = alloc(td); let k = alloc(td); let v = alloc(td);
        aether_op_matmul_f32_cuda(xn, lw.wq, q, ci(t), ci(d), ci(d));
        aether_op_matmul_f32_cuda(xn, lw.wk, k, ci(t), ci(d), ci(d));
        aether_op_matmul_f32_cuda(xn, lw.wv, v, ci(t), ci(d), ci(d));
        aether_op_rope_apply_f32_cuda(q, ci(t), ci(h), ci(hd), dm.base, 0);
        aether_op_rope_apply_f32_cuda(k, ci(t), ci(h), ci(hd), dm.base, 0);
        let qt = alloc(td); let kt = alloc(td); let vt = alloc(td);
        aether_op_transpose_021_f32_cuda(q, qt, ci(t), ci(h), ci(hd));
        aether_op_transpose_021_f32_cuda(k, kt, ci(t), ci(h), ci(hd));
        aether_op_transpose_021_f32_cuda(v, vt, ci(t), ci(h), ci(hd));
        aether_dev_free_f32(q); aether_dev_free_f32(k); aether_dev_free_f32(v);
        let ot = alloc(td); let attn = alloc(h * t * t);
        aether_op_sdpa_causal_forward_f32_cuda(qt, kt, vt, ot, attn, ci(h), ci(t), ci(hd));
        let o = alloc(td);
        aether_op_transpose_021_f32_cuda(ot, o, ci(h), ci(t), ci(hd));
        aether_dev_free_f32(ot);
        let proj = alloc(td);
        aether_op_matmul_f32_cuda(o, lw.wo, proj, ci(t), ci(d), ci(d));
        let x1 = alloc(td);
        aether_op_add_inplace_f32_cuda(x1, xin, ci(td));
        aether_op_add_inplace_f32_cuda(x1, proj, ci(td));
        aether_dev_free_f32(proj);
        let xn2 = alloc(td);
        aether_op_rms_norm_f32_cuda(x1, lw.g2, xn2, dm.eps, ci(t), ci(d));
        let gate = alloc(t * dff); let up = alloc(t * dff);
        aether_op_matmul_f32_cuda(xn2, lw.wgate, gate, ci(t), ci(d), ci(dff));
        aether_op_matmul_f32_cuda(xn2, lw.wup, up, ci(t), ci(d), ci(dff));
        let gate_pre = alloc(t * dff);
        aether_op_add_inplace_f32_cuda(gate_pre, gate, ci(t * dff));
        aether_op_silu_f32_cuda(gate, ci(t * dff));
        let gate_act = alloc(t * dff);
        aether_op_add_inplace_f32_cuda(gate_act, gate, ci(t * dff));
        aether_op_mul_inplace_f32_cuda(gate, up, ci(t * dff));
        let h_ffn = alloc(t * dff);
        aether_op_add_inplace_f32_cuda(h_ffn, gate, ci(t * dff));
        let down = alloc(td);
        aether_op_matmul_f32_cuda(gate, lw.wdown, down, ci(t), ci(dff), ci(d));
        aether_dev_free_f32(gate);
        let xout = alloc(td);
        aether_op_add_inplace_f32_cuda(xout, x1, ci(td));
        aether_op_add_inplace_f32_cuda(xout, down, ci(td));
        aether_dev_free_f32(down);
        let act = LayerAct {
            xd: xin, xn, qt, kt, vt, attn, o, x1, xn2, gate_pre, gate_act, up, h_ffn,
        };
        (xout, act)
    }

    /// Backward one layer. `d_xout` is the grad of this layer's output (consumed,
    /// freed here). Accumulates weight grads into `lw.d_*`. Returns d_xin handle.
    fn layer_backward(&self, lw: &LayerWeights, act: &LayerAct, d_xout: i64) -> i64 {
        let dm = &self.dims;
        let t = dm.t; let h = dm.h; let hd = dm.hd; let d = dm.d(); let dff = dm.dff;
        let td = t * d; let dd = d * d; let ddff = d * dff; let dffd = dff * d;
        // Grad accumulation: kernels OVERWRITE their dst, but 1F1B runs several
        // microbatch backwards before one step(), so each weight grad is computed
        // into a temp and added into the persistent accumulator lw.d_*.
        let accum = |dst: i64, tmp: i64, n: usize| {
            aether_op_add_inplace_f32_cuda(dst, tmp, ci(n));
            aether_dev_free_f32(tmp);
        };
        // xout = x1 + down
        let d_x1 = alloc(td); let d_down = alloc(td);
        aether_op_add_inplace_f32_cuda(d_x1, d_xout, ci(td));
        aether_op_add_inplace_f32_cuda(d_down, d_xout, ci(td));
        aether_dev_free_f32(d_xout);
        let t_wdown = alloc(dffd);
        aether_op_matmul_backward_rhs_f32_cuda(act.h_ffn, d_down, t_wdown, ci(t), ci(dff), ci(d));
        accum(lw.d_wdown, t_wdown, dffd);
        let d_h = alloc(t * dff);
        aether_op_matmul_backward_lhs_f32_cuda(d_down, lw.wdown, d_h, ci(t), ci(dff), ci(d));
        aether_dev_free_f32(d_down);
        let d_gate_act = alloc(t * dff);
        aether_op_add_inplace_f32_cuda(d_gate_act, d_h, ci(t * dff));
        aether_op_mul_inplace_f32_cuda(d_gate_act, act.up, ci(t * dff));
        let d_up = alloc(t * dff);
        aether_op_add_inplace_f32_cuda(d_up, d_h, ci(t * dff));
        aether_op_mul_inplace_f32_cuda(d_up, act.gate_act, ci(t * dff));
        aether_dev_free_f32(d_h);
        let d_gate = alloc(t * dff);
        aether_op_silu_backward_f32_cuda(act.gate_pre, d_gate_act, d_gate, ci(t * dff));
        aether_dev_free_f32(d_gate_act);
        let t_wgate = alloc(ddff); let t_wup = alloc(ddff);
        aether_op_matmul_backward_rhs_f32_cuda(act.xn2, d_gate, t_wgate, ci(t), ci(d), ci(dff));
        aether_op_matmul_backward_rhs_f32_cuda(act.xn2, d_up, t_wup, ci(t), ci(d), ci(dff));
        accum(lw.d_wgate, t_wgate, ddff); accum(lw.d_wup, t_wup, ddff);
        let d_xn2 = alloc(td); let d_xn2_b = alloc(td);
        aether_op_matmul_backward_lhs_f32_cuda(d_gate, lw.wgate, d_xn2, ci(t), ci(d), ci(dff));
        aether_op_matmul_backward_lhs_f32_cuda(d_up, lw.wup, d_xn2_b, ci(t), ci(d), ci(dff));
        aether_op_add_inplace_f32_cuda(d_xn2, d_xn2_b, ci(td));
        aether_dev_free_f32(d_xn2_b); aether_dev_free_f32(d_gate); aether_dev_free_f32(d_up);
        // rmsnorm2: x1 -> xn2
        let d_x1_ffn = alloc(td); let inv2 = alloc(t);
        aether_op_rms_norm_backward_dx_f32_cuda(act.x1, lw.g2, d_xn2, d_x1_ffn, inv2, dm.eps, ci(t), ci(d));
        let t_g2 = alloc(d);
        aether_op_rms_norm_backward_gamma_f32_cuda(act.x1, d_xn2, inv2, t_g2, ci(t), ci(d));
        accum(lw.d_g2, t_g2, d);
        aether_op_add_inplace_f32_cuda(d_x1, d_x1_ffn, ci(td));
        aether_dev_free_f32(d_xn2); aether_dev_free_f32(d_x1_ffn); aether_dev_free_f32(inv2);
        // x1 = xd + proj : d_proj = d_x1
        let d_proj = alloc(td);
        aether_op_add_inplace_f32_cuda(d_proj, d_x1, ci(td));
        let t_wo = alloc(dd);
        aether_op_matmul_backward_rhs_f32_cuda(act.o, d_proj, t_wo, ci(t), ci(d), ci(d));
        accum(lw.d_wo, t_wo, dd);
        let d_o = alloc(td);
        aether_op_matmul_backward_lhs_f32_cuda(d_proj, lw.wo, d_o, ci(t), ci(d), ci(d));
        aether_dev_free_f32(d_proj);
        let d_ot = alloc(td);
        aether_op_transpose_021_f32_cuda(d_o, d_ot, ci(t), ci(h), ci(hd));
        aether_dev_free_f32(d_o);
        let d_qt = alloc(td); let d_kt = alloc(td); let d_vt = alloc(td);
        let dscores = alloc(h * t * t);
        aether_op_sdpa_causal_backward_f32_cuda(act.qt, act.kt, act.vt, act.attn, d_ot,
            d_qt, d_kt, d_vt, dscores, ci(h), ci(t), ci(hd));
        aether_dev_free_f32(d_ot); aether_dev_free_f32(dscores);
        let d_q = alloc(td); let d_k = alloc(td); let d_v = alloc(td);
        aether_op_transpose_021_f32_cuda(d_qt, d_q, ci(h), ci(t), ci(hd));
        aether_op_transpose_021_f32_cuda(d_kt, d_k, ci(h), ci(t), ci(hd));
        aether_op_transpose_021_f32_cuda(d_vt, d_v, ci(h), ci(t), ci(hd));
        aether_dev_free_f32(d_qt); aether_dev_free_f32(d_kt); aether_dev_free_f32(d_vt);
        aether_op_rope_apply_backward_f32_cuda(d_q, ci(t), ci(h), ci(hd), dm.base, 0);
        aether_op_rope_apply_backward_f32_cuda(d_k, ci(t), ci(h), ci(hd), dm.base, 0);
        let t_wq = alloc(dd); let t_wk = alloc(dd); let t_wv = alloc(dd);
        aether_op_matmul_backward_rhs_f32_cuda(act.xn, d_q, t_wq, ci(t), ci(d), ci(d));
        aether_op_matmul_backward_rhs_f32_cuda(act.xn, d_k, t_wk, ci(t), ci(d), ci(d));
        aether_op_matmul_backward_rhs_f32_cuda(act.xn, d_v, t_wv, ci(t), ci(d), ci(d));
        accum(lw.d_wq, t_wq, dd); accum(lw.d_wk, t_wk, dd); accum(lw.d_wv, t_wv, dd);
        let d_xn = alloc(td); let d_xn_b = alloc(td); let d_xn_c = alloc(td);
        aether_op_matmul_backward_lhs_f32_cuda(d_q, lw.wq, d_xn, ci(t), ci(d), ci(d));
        aether_op_matmul_backward_lhs_f32_cuda(d_k, lw.wk, d_xn_b, ci(t), ci(d), ci(d));
        aether_op_matmul_backward_lhs_f32_cuda(d_v, lw.wv, d_xn_c, ci(t), ci(d), ci(d));
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_b, ci(td));
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_c, ci(td));
        aether_dev_free_f32(d_q); aether_dev_free_f32(d_k); aether_dev_free_f32(d_v);
        aether_dev_free_f32(d_xn_b); aether_dev_free_f32(d_xn_c);
        // rmsnorm1: xd -> xn
        let d_x_attn = alloc(td); let inv1 = alloc(t);
        aether_op_rms_norm_backward_dx_f32_cuda(act.xd, lw.g1, d_xn, d_x_attn, inv1, dm.eps, ci(t), ci(d));
        let t_g1 = alloc(d);
        aether_op_rms_norm_backward_gamma_f32_cuda(act.xd, d_xn, inv1, t_g1, ci(t), ci(d));
        accum(lw.d_g1, t_g1, d);
        aether_dev_free_f32(d_xn); aether_dev_free_f32(inv1);
        // total d_xin = d_x1 (skip) + d_x_attn (through norm1)
        let d_xin = alloc(td);
        aether_op_add_inplace_f32_cuda(d_xin, d_x1, ci(td));
        aether_op_add_inplace_f32_cuda(d_xin, d_x_attn, ci(td));
        aether_dev_free_f32(d_x1); aether_dev_free_f32(d_x_attn);
        d_xin
    }
}

fn draw(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| rng.next_normal() * scale).collect()
}

impl Stage for QwenBlockStage {
    fn input_dim(&self) -> usize { self.dims.t * self.dims.d() }
    fn output_dim(&self) -> usize { self.dims.t * self.dims.d() }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let td = self.input_dim();
        assert_eq!(input.len(), td, "[qwen_stage] input dim mismatch");
        let mut cur = upload(input);
        let mut acts = Vec::with_capacity(self.layers.len());
        // NB: index by position; borrow checker — collect handles first.
        let n = self.layers.len();
        for i in 0..n {
            let lw = unsafe { &*(&self.layers[i] as *const LayerWeights) };
            let (xout, act) = self.layer_forward(lw, cur);
            acts.push(act);
            cur = xout;
        }
        let out = download(cur, td);
        aether_dev_free_f32(cur);
        self.fifo.push_back(acts);
        unsafe { aether_dev_sync(); }
        out
    }

    fn backward(&mut self, grad: &[f32]) -> Vec<f32> {
        let td = self.output_dim();
        assert_eq!(grad.len(), td, "[qwen_stage] grad dim mismatch");
        let acts = self.fifo.pop_front().expect("[qwen_stage] backward with empty FIFO");
        let mut d_cur = upload(grad);
        let n = self.layers.len();
        for i in (0..n).rev() {
            let lw = unsafe { &*(&self.layers[i] as *const LayerWeights) };
            d_cur = self.layer_backward(lw, &acts[i], d_cur);
        }
        let d_in = download(d_cur, td);
        aether_dev_free_f32(d_cur);
        for a in &acts { a.free(); }
        unsafe { aether_dev_sync(); }
        d_in
    }

    fn step(&mut self, lr: f32, opt_step: i64) {
        let dm = self.dims;
        let d = dm.d(); let dd = d * d; let ddff = d * dm.dff; let dffd = dm.dff * d;
        let (b1, b2, eps, wd) = (0.9f32, 0.999f32, 1e-8f32, 0.0f32);
        for lw in &self.layers {
            let mut go = |p: i64, g: i64, m: i64, v: i64, n: usize| {
                aether_op_adamw_step_f32_cuda(p, g, m, v, lr, b1, b2, eps, wd, opt_step, ci(n));
            };
            go(lw.g1, lw.d_g1, lw.m.g1, lw.v.g1, d);
            go(lw.wq, lw.d_wq, lw.m.wq, lw.v.wq, dd);
            go(lw.wk, lw.d_wk, lw.m.wk, lw.v.wk, dd);
            go(lw.wv, lw.d_wv, lw.m.wv, lw.v.wv, dd);
            go(lw.wo, lw.d_wo, lw.m.wo, lw.v.wo, dd);
            go(lw.g2, lw.d_g2, lw.m.g2, lw.v.g2, d);
            go(lw.wgate, lw.d_wgate, lw.m.wgate, lw.v.wgate, ddff);
            go(lw.wup, lw.d_wup, lw.m.wup, lw.v.wup, ddff);
            go(lw.wdown, lw.d_wdown, lw.m.wdown, lw.v.wdown, dffd);
        }
        // Zero grad accumulators for the next batch (re-zero in place via h2d 0s).
        self.zero_grads();
        unsafe { aether_dev_sync(); }
    }
}

impl QwenBlockStage {
    fn zero_grads(&self) {
        let d = self.dims.d(); let dd = d * d;
        let ddff = d * self.dims.dff; let dffd = self.dims.dff * d;
        let z_d = vec![0.0f32; d];
        let z_dd = vec![0.0f32; dd];
        let z_ddff = vec![0.0f32; ddff];
        let z_dffd = vec![0.0f32; dffd];
        for lw in &self.layers {
            unsafe {
                aether_dev_h2d_f32(z_d.as_ptr() as i64, lw.d_g1, ci(d));
                aether_dev_h2d_f32(z_dd.as_ptr() as i64, lw.d_wq, ci(dd));
                aether_dev_h2d_f32(z_dd.as_ptr() as i64, lw.d_wk, ci(dd));
                aether_dev_h2d_f32(z_dd.as_ptr() as i64, lw.d_wv, ci(dd));
                aether_dev_h2d_f32(z_dd.as_ptr() as i64, lw.d_wo, ci(dd));
                aether_dev_h2d_f32(z_d.as_ptr() as i64, lw.d_g2, ci(d));
                aether_dev_h2d_f32(z_ddff.as_ptr() as i64, lw.d_wgate, ci(ddff));
                aether_dev_h2d_f32(z_ddff.as_ptr() as i64, lw.d_wup, ci(ddff));
                aether_dev_h2d_f32(z_dffd.as_ptr() as i64, lw.d_wdown, ci(dffd));
            }
        }
    }
}
