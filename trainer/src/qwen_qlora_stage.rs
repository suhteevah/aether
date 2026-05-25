//! GPU Qwen3 QLoRA pipeline stage — FR-18.6-real leg 3 (matt-voice 32B).
//!
//! roadmap: P18
//!
//! A pipeline `Stage` that loads a contiguous LAYER RANGE of a real Qwen3 GGUF's
//! quantized weights (frozen base, kept quantized in VRAM) and trains f32 LoRA
//! adapters on top via the leg-2 fwd/bwd kernels. This is the matt-voice
//! fine-tune shape: the 19 GB Qwen3-32B base is frozen + quantized and split
//! 32/32 across two P100s by `pp-qwen-worker`; only the low-rank adapters train.
//!
//! Per proj: y = base(Wq, x) + s·(x·A)·B. The frozen base is dequantized to a
//! TRANSIENT f32 buffer per matmul (base stays quantized at rest), then the
//! existing multi-row f32 matmul ops run the forward (y = x·Wqᵀ via
//! matmul_backward_lhs) and the dx backward (dx = dy·Wq via matmul_f32); the
//! base gets NO gradient. Adapter fwd/bwd is the math finite-diff-verified in
//! runtime/tests/cuda_qlora_proj_grad_check.rs. Attention is full-sequence causal
//! SDPA (leg-2), with Qwen3 per-head Q/K RMSNorm before RoPE and GQA repeat/reduce
//! (n_kv < n_q). The loss head (final norm + lm_head + CE) lives in run_1f1b's
//! loss closure, so the stage stays a pure block stack.

#![cfg(feature = "cuda")]

use std::collections::VecDeque;
use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rms_norm_backward_dx_f32_cuda, aether_op_rms_norm_backward_gamma_f32_cuda,
    aether_op_matmul_f32_cuda,
    aether_op_matmul_backward_lhs_f32_cuda, aether_op_matmul_backward_rhs_f32_cuda,
    aether_op_rope_apply_f32_cuda, aether_op_rope_apply_backward_f32_cuda,
    aether_op_sdpa_causal_forward_f32_cuda, aether_op_sdpa_causal_backward_f32_cuda,
    aether_op_transpose_021_f32_cuda,
    aether_op_gqa_repeat_kv_f32_cuda, aether_op_gqa_reduce_kv_grad_f32_cuda,
    aether_op_silu_f32_cuda, aether_op_silu_backward_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_adamw_step_f32_cuda,
    aether_op_dequant_q4_k_m_f32_cuda, aether_op_dequant_q6_k_f32_cuda,
    aether_op_dequant_iq3_xxs_f32_cuda,
    aether_dev_alloc_u8, aether_dev_h2d_u8,
    aether_op_cross_entropy_f32_cuda, aether_op_cross_entropy_backward_f32_cuda,
    aether_dev_alloc_i32, aether_dev_h2d_i32,
};
use aether_rt::{
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_n_elems, aether_gguf_get_tensor_data_ptr,
    aether_dequant_q4_k_m,
};
use aether_rt::serving::{ModelConfig, QwenLayerWeights, open_gguf_config, load_qwen_layer};
use std::ffi::c_void;

use crate::pipeline::Stage;
use crate::rng::Rng;

fn ci(n: usize) -> c_int { n as c_int }

fn alloc(n: usize) -> i64 {
    let h = aether_dev_alloc_f32(ci(n));
    assert!(h >= 0, "[qlora] alloc {} failed", n);
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

/// Dequant a frozen quant weight [n_out, n_in] → a fresh transient f32 device
/// buffer. Caller frees it. Panics (naming the dtype) on an unsupported code so
/// extending coverage is mechanical.
fn dequant(w: i64, dt: i32, n_out: usize, n_in: usize) -> i64 {
    assert!((n_out * n_in) % 256 == 0, "[qlora] dequant n_out*n_in {} not %256", n_out * n_in);
    let nb = (n_out * n_in / 256) as c_int;
    let out = alloc(n_out * n_in);
    let rc = match dt {
        12 => aether_op_dequant_q4_k_m_f32_cuda(w, out, nb),
        14 => aether_op_dequant_q6_k_f32_cuda(w, out, nb),
        18 => aether_op_dequant_iq3_xxs_f32_cuda(w, out, nb),
        _  => panic!("[qlora] dequant: unsupported dtype {} (have 12=Q4_K,14=Q6_K,18=IQ3_XXS); \
                      add aether_op_dequant_<dt>_f32_cuda + an arm here", dt),
    };
    assert_eq!(rc, 0, "[qlora] dequant dt={} rc={}", dt, rc);
    out
}

/// base forward: y[rows, n_out] = x[rows, n_in] @ Wq^T  (Wq dequanted [n_out,n_in]).
fn base_fwd(x: i64, w: i64, dt: i32, y: i64, rows: usize, n_out: usize, n_in: usize) {
    let wf = dequant(w, dt, n_out, n_in);
    aether_op_matmul_backward_lhs_f32_cuda(x, wf, y, ci(rows), ci(n_out), ci(n_in));
    aether_dev_free_f32(wf);
}
/// base backward dx: dx[rows, n_in] += dy[rows, n_out] @ Wq  (frozen → no dW).
/// Writes into a fresh buffer and returns it (caller accumulates).
fn base_dx(dy: i64, w: i64, dt: i32, rows: usize, n_out: usize, n_in: usize) -> i64 {
    let wf = dequant(w, dt, n_out, n_in);
    let dx = alloc(rows * n_in);
    aether_op_matmul_f32_cuda(dy, wf, dx, ci(rows), ci(n_out), ci(n_in));
    aether_dev_free_f32(wf);
    dx
}

/// A trainable LoRA adapter on one projection. A[n_in,rank], B[rank,n_out] on
/// device; grads + AdamW moments persistent. B=0 at init (delta=0 → adapters
/// start as a no-op, standard LoRA).
struct Adapter {
    n_in: usize, n_out: usize, rank: usize, scale: f32,
    a: i64, b: i64,
    da: i64, db: i64,
    ma: i64, va: i64, mb: i64, vb: i64,
}
impl Adapter {
    fn new(n_in: usize, n_out: usize, rank: usize, alpha: f32, rng: &mut Rng) -> Self {
        let std = 1.0 / (rank as f32).sqrt();
        let a_host: Vec<f32> = (0..n_in * rank).map(|_| rng.next_normal() * std).collect();
        let a = upload(&a_host);
        let b = alloc(rank * n_out);   // zeros
        Adapter {
            n_in, n_out, rank, scale: alpha / rank as f32,
            a, b,
            da: alloc(n_in * rank), db: alloc(rank * n_out),
            ma: alloc(n_in * rank), va: alloc(n_in * rank),
            mb: alloc(rank * n_out), vb: alloc(rank * n_out),
        }
    }
    /// delta accumulated into `y`: y += s·(x·A)·B. Returns saved a_x[rows,rank].
    fn forward(&self, x: i64, y: i64, rows: usize) -> i64 {
        let a_x = alloc(rows * self.rank);
        aether_op_matmul_f32_cuda(x, self.a, a_x, ci(rows), ci(self.n_in), ci(self.rank));
        let delta = alloc(rows * self.n_out);
        aether_op_matmul_f32_cuda(a_x, self.b, delta, ci(rows), ci(self.rank), ci(self.n_out));
        // scale delta by s, then add to y
        let s = self.scale;
        // reuse mul_inplace? simplest: scale via a host-free path — fold s into add.
        // We scale delta in place by s using axpy-style: delta *= s.
        scale_inplace(delta, s, rows * self.n_out);
        aether_op_add_inplace_f32_cuda(y, delta, ci(rows * self.n_out));
        aether_dev_free_f32(delta);
        a_x
    }
    /// Accumulate dA, dB; return dx_lora[rows,n_in]. `dy` is grad of the proj
    /// output; `x`, `a_x` are the saved forward intermediates.
    fn backward(&self, x: i64, a_x: i64, dy: i64, rows: usize) -> i64 {
        let s = self.scale;
        // d_delta = s · dy
        let d_delta = alloc(rows * self.n_out);
        aether_op_add_inplace_f32_cuda(d_delta, dy, ci(rows * self.n_out));
        scale_inplace(d_delta, s, rows * self.n_out);
        // dB += a_x^T @ d_delta
        let t_db = alloc(self.rank * self.n_out);
        aether_op_matmul_backward_rhs_f32_cuda(a_x, d_delta, t_db, ci(rows), ci(self.rank), ci(self.n_out));
        aether_op_add_inplace_f32_cuda(self.db, t_db, ci(self.rank * self.n_out));
        aether_dev_free_f32(t_db);
        // d_a_x = d_delta @ B^T
        let d_a_x = alloc(rows * self.rank);
        aether_op_matmul_backward_lhs_f32_cuda(d_delta, self.b, d_a_x, ci(rows), ci(self.rank), ci(self.n_out));
        aether_dev_free_f32(d_delta);
        // dA += x^T @ d_a_x
        let t_da = alloc(self.n_in * self.rank);
        aether_op_matmul_backward_rhs_f32_cuda(x, d_a_x, t_da, ci(rows), ci(self.n_in), ci(self.rank));
        aether_op_add_inplace_f32_cuda(self.da, t_da, ci(self.n_in * self.rank));
        aether_dev_free_f32(t_da);
        // dx_lora = d_a_x @ A^T
        let dx = alloc(rows * self.n_in);
        aether_op_matmul_backward_lhs_f32_cuda(d_a_x, self.a, dx, ci(rows), ci(self.n_in), ci(self.rank));
        aether_dev_free_f32(d_a_x);
        dx
    }
    fn step(&self, lr: f32, opt_step: i64) {
        let (b1, b2, eps, wd) = (0.9f32, 0.999f32, 1e-8f32, 0.0f32);
        aether_op_adamw_step_f32_cuda(self.a, self.da, self.ma, self.va, lr, b1, b2, eps, wd, opt_step, ci(self.n_in * self.rank));
        aether_op_adamw_step_f32_cuda(self.b, self.db, self.mb, self.vb, lr, b1, b2, eps, wd, opt_step, ci(self.rank * self.n_out));
    }
    fn zero_grad(&self) {
        zero(self.da, self.n_in * self.rank);
        zero(self.db, self.rank * self.n_out);
    }
    fn n_params(&self) -> usize { self.n_in * self.rank + self.rank * self.n_out }
}

fn zero(h: i64, n: usize) {
    let z = vec![0.0f32; n];
    unsafe { aether_dev_h2d_f32(z.as_ptr() as i64, h, ci(n)); }
}
fn scale_inplace(h: i64, s: f32, n: usize) {
    // x *= s  via the runtime scale op.
    aether_rt::cuda::aether_op_scale_f32_cuda(h, s, ci(n));
}

/// Frozen quant weights for one layer + its 7 adapters.
struct LayerQ {
    w: QwenLayerWeights,
    ad_q: Adapter, ad_k: Adapter, ad_v: Adapter, ad_o: Adapter,
    ad_gate: Adapter, ad_up: Adapter, ad_down: Adapter,
}

/// Saved forward activations for one layer per microbatch.
struct ActQ {
    x: i64,                       // block input
    xn: i64,                      // attn rms output
    ax_q: i64, ax_k: i64, ax_v: i64,   // adapter a_x for q/k/v
    qn: i64, kn: i64,             // pre-rope, post-qknorm q/k (saved for qknorm bwd) [for q/k norm path]
    q_pre_qn: i64, k_pre_qn: i64, // q/k BEFORE per-head norm (rms bwd needs input)
    inv_qn: i64, inv_kn: i64,     // rms inv for q/k norm (T*HQ / T*HKV)
    qt: i64, kt: i64, vt: i64,    // [HQ,T,HD] (k/v repeated)
    attn: i64,
    o: i64, ax_o: i64,
    x1: i64,
    xn2: i64, ax_gate: i64, ax_up: i64,
    gate_pre: i64, gate_act: i64, up: i64, h_ffn: i64, ax_down: i64,
    has_qknorm: bool,
}
impl ActQ {
    fn free(&self) {
        let mut hs = vec![self.x, self.xn, self.ax_q, self.ax_k, self.ax_v,
            self.qt, self.kt, self.vt, self.attn, self.o, self.ax_o, self.x1,
            self.xn2, self.ax_gate, self.ax_up, self.gate_pre, self.gate_act,
            self.up, self.h_ffn, self.ax_down];
        if self.has_qknorm {
            hs.extend_from_slice(&[self.qn, self.kn, self.q_pre_qn, self.k_pre_qn, self.inv_qn, self.inv_kn]);
        }
        for h in hs { aether_dev_free_f32(h); }
    }
}

/// Pipeline stage: a layer range of a Qwen3 GGUF, frozen-quant base + LoRA.
pub struct QwenQLoraStage {
    pub cfg: ModelConfig,
    pub gguf_handle: i64,
    layers: Vec<LayerQ>,
    fifo: VecDeque<Vec<ActQ>>,
    rank_lo: usize,
    // cached dims
    t: usize,
    // last-rank real LM head (loaded on demand): output_norm (f32) + output.weight
    // (Q6_K) split into vocab-row CHUNKS so the f32 dequant transient stays small
    // (~190 MB/chunk vs ~3.1 GB for the whole weight) — fits the 16 GB P100.
    lm_norm: i64,
    lm_chunks: Vec<i64>,     // per-chunk Q6_K u8 device handles
    lm_chunk_rows: Vec<usize>, // vocab rows in each chunk
    lm_nb_row: usize,        // super-blocks per vocab row (d/256)
}

impl QwenQLoraStage {
    /// Open the GGUF, read config, and load layers in `range` (frozen quant) +
    /// init adapters. `t` = sequence length. Returns the stage; each rank loads
    /// only its slice so two ranks together hold all layers.
    pub fn build(gguf_path: &str, range: std::ops::Range<usize>, t: usize,
                 lora_rank: usize, lora_alpha: f32, seed: u64) -> Result<Self, String> {
        let (h, cfg) = unsafe { open_gguf_config(gguf_path)? };
        let d = cfg.d_model;
        let q_dim = cfg.n_q_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let dff = cfg.d_ff;
        let mut rng = Rng::new(seed);
        let mut layers = Vec::new();
        let rank_lo = range.start;
        eprintln!("[qlora] loading layers {:?} of {} (d={} q_dim={} kv_dim={} dff={} n_q={} n_kv={} hd={})",
            range, cfg.n_layers, d, q_dim, kv_dim, dff, cfg.n_q_heads, cfg.n_kv_heads, cfg.head_dim);
        for b in range.clone() {
            let w = unsafe { load_qwen_layer(h, b) };
            let mk = |ni, no, rng: &mut Rng| Adapter::new(ni, no, lora_rank, lora_alpha, rng);
            layers.push(LayerQ {
                ad_q: mk(d, q_dim, &mut rng), ad_k: mk(d, kv_dim, &mut rng),
                ad_v: mk(d, kv_dim, &mut rng), ad_o: mk(q_dim, d, &mut rng),
                ad_gate: mk(d, dff, &mut rng), ad_up: mk(d, dff, &mut rng),
                ad_down: mk(dff, d, &mut rng),
                w,
            });
        }
        eprintln!("[qlora] loaded {} layers, {} adapter params/layer",
            layers.len(),
            layers.first().map(|l| l.ad_q.n_params() + l.ad_k.n_params() + l.ad_v.n_params()
                + l.ad_o.n_params() + l.ad_gate.n_params() + l.ad_up.n_params() + l.ad_down.n_params()).unwrap_or(0));
        Ok(QwenQLoraStage { cfg, gguf_handle: h, layers, fifo: VecDeque::new(), rank_lo, t,
                            lm_norm: 0, lm_chunks: Vec::new(), lm_chunk_rows: Vec::new(), lm_nb_row: 0 })
    }

    pub fn n_layers(&self) -> usize { self.layers.len() }
    pub fn total_adapter_params(&self) -> usize {
        self.layers.iter().map(|l| l.ad_q.n_params() + l.ad_k.n_params() + l.ad_v.n_params()
            + l.ad_o.n_params() + l.ad_gate.n_params() + l.ad_up.n_params() + l.ad_down.n_params()).sum()
    }
    /// Sum of |adapter B| across all layers — 0 at init (B=0), grows as training
    /// moves the adapters. A nonzero value after a step proves grads flowed.
    pub fn adapter_b_abs_sum(&self) -> f64 {
        let mut s = 0.0f64;
        for l in &self.layers {
            for ad in [&l.ad_q, &l.ad_k, &l.ad_v, &l.ad_o, &l.ad_gate, &l.ad_up, &l.ad_down] {
                s += download(ad.b, ad.rank * ad.n_out).iter().map(|v| v.abs() as f64).sum::<f64>();
            }
        }
        unsafe { aether_dev_sync(); }
        s
    }

    fn layer_forward(&self, lq: &LayerQ, x: i64) -> (i64, ActQ) {
        let cfg = &self.cfg;
        let t = self.t; let d = cfg.d_model;
        let hq = cfg.n_q_heads; let hkv = cfg.n_kv_heads; let hd = cfg.head_dim;
        let q_dim = hq * hd; let kv_dim = hkv * hd; let dff = cfg.d_ff;
        let eps = cfg.norm_eps; let base = cfg.rope_base;
        let w = &lq.w;

        let xn = alloc(t * d);
        aether_op_rms_norm_f32_cuda(x, w.attn_norm_g, xn, eps, ci(t), ci(d));
        // q/k/v = base(xn) + adapter
        let q = alloc(t * q_dim); base_fwd(xn, w.w_q, w.dt_q, q, t, q_dim, d);
        let ax_q = lq.ad_q.forward(xn, q, t);
        let k = alloc(t * kv_dim); base_fwd(xn, w.w_k, w.dt_k, k, t, kv_dim, d);
        let ax_k = lq.ad_k.forward(xn, k, t);
        let v = alloc(t * kv_dim); base_fwd(xn, w.w_v, w.dt_v, v, t, kv_dim, d);
        let ax_v = lq.ad_v.forward(xn, v, t);

        // Qwen3 per-head Q/K RMSNorm (over head_dim), before RoPE. Skipped if absent.
        let has_qknorm = w.attn_q_norm_g != 0 && w.attn_k_norm_g != 0;
        let (qn, kn, q_pre_qn, k_pre_qn, inv_qn, inv_kn);
        if has_qknorm {
            q_pre_qn = alloc(t * q_dim); aether_op_add_inplace_f32_cuda(q_pre_qn, q, ci(t * q_dim));
            k_pre_qn = alloc(t * kv_dim); aether_op_add_inplace_f32_cuda(k_pre_qn, k, ci(t * kv_dim));
            qn = alloc(t * q_dim); inv_qn = alloc(t * hq);
            aether_op_rms_norm_f32_cuda(q, w.attn_q_norm_g, qn, eps, ci(t * hq), ci(hd));
            kn = alloc(t * kv_dim); inv_kn = alloc(t * hkv);
            aether_op_rms_norm_f32_cuda(k, w.attn_k_norm_g, kn, eps, ci(t * hkv), ci(hd));
            // overwrite q/k with normed for the rest of the path
            aether_dev_free_f32(q); aether_dev_free_f32(k);
        } else {
            qn = q; kn = k; q_pre_qn = 0; k_pre_qn = 0; inv_qn = 0; inv_kn = 0;
        }
        // rope
        aether_op_rope_apply_f32_cuda(qn, ci(t), ci(hq), ci(hd), base, 0);
        aether_op_rope_apply_f32_cuda(kn, ci(t), ci(hkv), ci(hd), base, 0);
        // GQA repeat k/v to hq heads
        let kr = alloc(t * q_dim); let vr = alloc(t * q_dim);
        aether_op_gqa_repeat_kv_f32_cuda(kn, kr, ci(t), ci(hkv), ci(hd), ci(hq));
        aether_op_gqa_repeat_kv_f32_cuda(v, vr, ci(t), ci(hkv), ci(hd), ci(hq));
        aether_dev_free_f32(v);
        // transpose to [hq,t,hd]
        let qt = alloc(t * q_dim); let kt = alloc(t * q_dim); let vt = alloc(t * q_dim);
        aether_op_transpose_021_f32_cuda(qn, qt, ci(t), ci(hq), ci(hd));
        aether_op_transpose_021_f32_cuda(kr, kt, ci(t), ci(hq), ci(hd));
        aether_op_transpose_021_f32_cuda(vr, vt, ci(t), ci(hq), ci(hd));
        aether_dev_free_f32(qn); aether_dev_free_f32(kn); aether_dev_free_f32(kr); aether_dev_free_f32(vr);
        let ot = alloc(t * q_dim); let attn = alloc(hq * t * t);
        aether_op_sdpa_causal_forward_f32_cuda(qt, kt, vt, ot, attn, ci(hq), ci(t), ci(hd));
        let o = alloc(t * q_dim);
        aether_op_transpose_021_f32_cuda(ot, o, ci(hq), ci(t), ci(hd));
        aether_dev_free_f32(ot);
        // o proj (in=q_dim, out=d)
        let proj = alloc(t * d); base_fwd(o, w.w_o, w.dt_o, proj, t, d, q_dim);
        let ax_o = lq.ad_o.forward(o, proj, t);
        let x1 = alloc(t * d);
        aether_op_add_inplace_f32_cuda(x1, x, ci(t * d));
        aether_op_add_inplace_f32_cuda(x1, proj, ci(t * d));
        aether_dev_free_f32(proj);
        // FFN
        let xn2 = alloc(t * d);
        aether_op_rms_norm_f32_cuda(x1, w.ffn_norm_g, xn2, eps, ci(t), ci(d));
        let gate = alloc(t * dff); base_fwd(xn2, w.w_gate, w.dt_gate, gate, t, dff, d);
        let ax_gate = lq.ad_gate.forward(xn2, gate, t);
        let up = alloc(t * dff); base_fwd(xn2, w.w_up, w.dt_up, up, t, dff, d);
        let ax_up = lq.ad_up.forward(xn2, up, t);
        let gate_pre = alloc(t * dff); aether_op_add_inplace_f32_cuda(gate_pre, gate, ci(t * dff));
        aether_op_silu_f32_cuda(gate, ci(t * dff));
        let gate_act = alloc(t * dff); aether_op_add_inplace_f32_cuda(gate_act, gate, ci(t * dff));
        aether_op_mul_inplace_f32_cuda(gate, up, ci(t * dff));
        let h_ffn = alloc(t * dff); aether_op_add_inplace_f32_cuda(h_ffn, gate, ci(t * dff));
        let down = alloc(t * d); base_fwd(gate, w.w_down, w.dt_down, down, t, d, dff);
        let ax_down = lq.ad_down.forward(gate, down, t);
        aether_dev_free_f32(gate);
        let xout = alloc(t * d);
        aether_op_add_inplace_f32_cuda(xout, x1, ci(t * d));
        aether_op_add_inplace_f32_cuda(xout, down, ci(t * d));
        aether_dev_free_f32(down);

        let act = ActQ {
            x, xn, ax_q, ax_k, ax_v, qn, kn, q_pre_qn, k_pre_qn, inv_qn, inv_kn,
            qt, kt, vt, attn, o, ax_o, x1, xn2, ax_gate, ax_up,
            gate_pre, gate_act, up, h_ffn, ax_down, has_qknorm,
        };
        (xout, act)
    }

    fn layer_backward(&self, lq: &LayerQ, a: &ActQ, d_xout: i64) -> i64 {
        let cfg = &self.cfg;
        let t = self.t; let d = cfg.d_model;
        let hq = cfg.n_q_heads; let hkv = cfg.n_kv_heads; let hd = cfg.head_dim;
        let q_dim = hq * hd; let kv_dim = hkv * hd; let dff = cfg.d_ff;
        let eps = cfg.norm_eps; let base = cfg.rope_base;
        let w = &lq.w;

        // xout = x1 + down
        let d_x1 = alloc(t * d); aether_op_add_inplace_f32_cuda(d_x1, d_xout, ci(t * d));
        let d_down = alloc(t * d); aether_op_add_inplace_f32_cuda(d_down, d_xout, ci(t * d));
        aether_dev_free_f32(d_xout);
        // down = base(h_ffn) + adapter(h_ffn)
        let d_h = base_dx(d_down, w.w_down, w.dt_down, t, d, dff);
        let d_h_lora = lq.ad_down.backward(a.h_ffn, a.ax_down, d_down, t);
        aether_op_add_inplace_f32_cuda(d_h, d_h_lora, ci(t * dff));
        aether_dev_free_f32(d_h_lora); aether_dev_free_f32(d_down);
        // h_ffn = silu(gate)*up
        let d_gate_act = alloc(t * dff); aether_op_add_inplace_f32_cuda(d_gate_act, d_h, ci(t * dff));
        aether_op_mul_inplace_f32_cuda(d_gate_act, a.up, ci(t * dff));
        let d_up = alloc(t * dff); aether_op_add_inplace_f32_cuda(d_up, d_h, ci(t * dff));
        aether_op_mul_inplace_f32_cuda(d_up, a.gate_act, ci(t * dff));
        aether_dev_free_f32(d_h);
        let d_gate = alloc(t * dff);
        aether_op_silu_backward_f32_cuda(a.gate_pre, d_gate_act, d_gate, ci(t * dff));
        aether_dev_free_f32(d_gate_act);
        // gate/up = base(xn2)+adapter(xn2)
        let d_xn2 = base_dx(d_gate, w.w_gate, w.dt_gate, t, dff, d);
        let d_xn2_lg = lq.ad_gate.backward(a.xn2, a.ax_gate, d_gate, t);
        aether_op_add_inplace_f32_cuda(d_xn2, d_xn2_lg, ci(t * d)); aether_dev_free_f32(d_xn2_lg);
        let d_xn2_u = base_dx(d_up, w.w_up, w.dt_up, t, dff, d);
        aether_op_add_inplace_f32_cuda(d_xn2, d_xn2_u, ci(t * d)); aether_dev_free_f32(d_xn2_u);
        let d_xn2_lu = lq.ad_up.backward(a.xn2, a.ax_up, d_up, t);
        aether_op_add_inplace_f32_cuda(d_xn2, d_xn2_lu, ci(t * d)); aether_dev_free_f32(d_xn2_lu);
        aether_dev_free_f32(d_gate); aether_dev_free_f32(d_up);
        // ffn rmsnorm: x1 -> xn2
        let d_x1_ffn = alloc(t * d); let inv2 = alloc(t);
        aether_op_rms_norm_backward_dx_f32_cuda(a.x1, w.ffn_norm_g, d_xn2, d_x1_ffn, inv2, eps, ci(t), ci(d));
        // ffn_norm_g grad is a base param (frozen) — skip gamma grad.
        aether_op_add_inplace_f32_cuda(d_x1, d_x1_ffn, ci(t * d));
        aether_dev_free_f32(d_xn2); aether_dev_free_f32(d_x1_ffn); aether_dev_free_f32(inv2);
        // x1 = x + proj : d_proj = d_x1
        let d_proj = alloc(t * d); aether_op_add_inplace_f32_cuda(d_proj, d_x1, ci(t * d));
        // o proj
        let d_o = base_dx(d_proj, w.w_o, w.dt_o, t, d, q_dim);
        let d_o_lora = lq.ad_o.backward(a.o, a.ax_o, d_proj, t);
        aether_op_add_inplace_f32_cuda(d_o, d_o_lora, ci(t * q_dim));
        aether_dev_free_f32(d_o_lora); aether_dev_free_f32(d_proj);
        // transpose o-grad to [hq,t,hd]
        let d_ot = alloc(t * q_dim);
        aether_op_transpose_021_f32_cuda(d_o, d_ot, ci(t), ci(hq), ci(hd));
        aether_dev_free_f32(d_o);
        let d_qt = alloc(t * q_dim); let d_kt = alloc(t * q_dim); let d_vt = alloc(t * q_dim);
        let dscores = alloc(hq * t * t);
        aether_op_sdpa_causal_backward_f32_cuda(a.qt, a.kt, a.vt, a.attn, d_ot,
            d_qt, d_kt, d_vt, dscores, ci(hq), ci(t), ci(hd));
        aether_dev_free_f32(d_ot); aether_dev_free_f32(dscores);
        // transpose back to [t,hq,hd]
        let d_q = alloc(t * q_dim); let d_kr = alloc(t * q_dim); let d_vr = alloc(t * q_dim);
        aether_op_transpose_021_f32_cuda(d_qt, d_q, ci(hq), ci(t), ci(hd));
        aether_op_transpose_021_f32_cuda(d_kt, d_kr, ci(hq), ci(t), ci(hd));
        aether_op_transpose_021_f32_cuda(d_vt, d_vr, ci(hq), ci(t), ci(hd));
        aether_dev_free_f32(d_qt); aether_dev_free_f32(d_kt); aether_dev_free_f32(d_vt);
        // GQA reduce k/v grads [t,hq,hd] -> [t,hkv,hd]
        let d_k = alloc(t * kv_dim); let d_v = alloc(t * kv_dim);
        aether_op_gqa_reduce_kv_grad_f32_cuda(d_kr, d_k, ci(t), ci(hkv), ci(hd), ci(hq));
        aether_op_gqa_reduce_kv_grad_f32_cuda(d_vr, d_v, ci(t), ci(hkv), ci(hd), ci(hq));
        aether_dev_free_f32(d_kr); aether_dev_free_f32(d_vr);
        // rope backward (q at hq, k at hkv)
        aether_op_rope_apply_backward_f32_cuda(d_q, ci(t), ci(hq), ci(hd), base, 0);
        aether_op_rope_apply_backward_f32_cuda(d_k, ci(t), ci(hkv), ci(hd), base, 0);
        // Qwen3 per-head Q/K RMSNorm backward (over head_dim) if present.
        let (d_q_pre, d_k_pre);
        if a.has_qknorm {
            // bwd_dx recomputes inv internally into the passed buffer; reuse the
            // saved inv_* scratch (its forward contents are unused).
            let dq = alloc(t * q_dim);
            aether_op_rms_norm_backward_dx_f32_cuda(a.q_pre_qn, w.attn_q_norm_g, d_q, dq, a.inv_qn, eps, ci(t * hq), ci(hd));
            let dk = alloc(t * kv_dim);
            aether_op_rms_norm_backward_dx_f32_cuda(a.k_pre_qn, w.attn_k_norm_g, d_k, dk, a.inv_kn, eps, ci(t * hkv), ci(hd));
            aether_dev_free_f32(d_q); aether_dev_free_f32(d_k);
            d_q_pre = dq; d_k_pre = dk;
        } else {
            d_q_pre = d_q; d_k_pre = d_k;
        }
        // q/k/v = base(xn)+adapter(xn) : accumulate d_xn
        let d_xn = base_dx(d_q_pre, w.w_q, w.dt_q, t, q_dim, d);
        let d_xn_lq = lq.ad_q.backward(a.xn, a.ax_q, d_q_pre, t);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_lq, ci(t * d)); aether_dev_free_f32(d_xn_lq);
        let d_xn_k = base_dx(d_k_pre, w.w_k, w.dt_k, t, kv_dim, d);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_k, ci(t * d)); aether_dev_free_f32(d_xn_k);
        let d_xn_lk = lq.ad_k.backward(a.xn, a.ax_k, d_k_pre, t);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_lk, ci(t * d)); aether_dev_free_f32(d_xn_lk);
        let d_xn_v = base_dx(d_v, w.w_v, w.dt_v, t, kv_dim, d);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_v, ci(t * d)); aether_dev_free_f32(d_xn_v);
        let d_xn_lv = lq.ad_v.backward(a.xn, a.ax_v, d_v, t);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_lv, ci(t * d)); aether_dev_free_f32(d_xn_lv);
        aether_dev_free_f32(d_q_pre); aether_dev_free_f32(d_k_pre); aether_dev_free_f32(d_v);
        // attn rmsnorm: x -> xn
        let d_x_attn = alloc(t * d); let inv1 = alloc(t);
        aether_op_rms_norm_backward_dx_f32_cuda(a.x, w.attn_norm_g, d_xn, d_x_attn, inv1, eps, ci(t), ci(d));
        aether_dev_free_f32(d_xn); aether_dev_free_f32(inv1);
        // total d_x = d_x1 (residual skip) + d_x_attn
        let d_x = alloc(t * d);
        aether_op_add_inplace_f32_cuda(d_x, d_x1, ci(t * d));
        aether_op_add_inplace_f32_cuda(d_x, d_x_attn, ci(t * d));
        aether_dev_free_f32(d_x1); aether_dev_free_f32(d_x_attn);
        d_x
    }
}

// matt-voice REAL training: embed-in (rank 0) + LM-head loss (last rank) + adapter save.
impl QwenQLoraStage {
    /// Embed `ids` (len T) into [T, D] host floats via the GGUF token_embd.weight
    /// (Q4_K rows, host dequant). Used by rank 0 to build the real model input.
    pub fn embed_tokens(&self, ids: &[usize]) -> Vec<f32> {
        let d = self.cfg.d_model;
        unsafe {
            let needle = b"token_embd.weight";
            let idx = aether_gguf_find_tensor_by_name(self.gguf_handle, needle.as_ptr() as i64, needle.len() as c_int);
            assert!(idx >= 0, "[qlora] token_embd.weight not found");
            let dt = aether_gguf_get_tensor_dtype(self.gguf_handle, idx);
            assert_eq!(dt, 12, "[qlora] token_embd dtype {} != Q4_K(12); add dispatch", dt);
            let n_elems = aether_gguf_get_tensor_n_elems(self.gguf_handle, idx) as usize;
            let total_rows = n_elems / d;
            let dptr = aether_gguf_get_tensor_data_ptr(self.gguf_handle, idx) as *const u8;
            let bpr = (d / 256) * 144; // Q4_K bytes per row
            let mut out = vec![0.0f32; ids.len() * d];
            for (oi, &id) in ids.iter().enumerate() {
                assert!(id < total_rows, "[qlora] token id {} >= vocab {}", id, total_rows);
                let row = std::slice::from_raw_parts(dptr.add(id * bpr), bpr);
                let mut rf = vec![0.0f32; d];
                aether_dequant_q4_k_m(row.as_ptr() as *const c_void, rf.as_mut_ptr() as *mut c_void, (d / 256) as c_int);
                out[oi * d..(oi + 1) * d].copy_from_slice(&rf);
            }
            out
        }
    }

    /// Load output_norm.weight (F32) + output.weight (lm head, Q6_K) to device.
    /// Last rank only; call once.
    pub fn load_lm_head(&mut self) {
        let d = self.cfg.d_model; let vocab = self.cfg.vocab;
        unsafe {
            // output_norm (f32)
            let nn = b"output_norm.weight";
            let ni = aether_gguf_find_tensor_by_name(self.gguf_handle, nn.as_ptr() as i64, nn.len() as c_int);
            assert!(ni >= 0, "[qlora] output_norm.weight missing");
            let ne = aether_gguf_get_tensor_n_elems(self.gguf_handle, ni) as usize;
            let np = aether_gguf_get_tensor_data_ptr(self.gguf_handle, ni) as *const f32;
            let nh = std::slice::from_raw_parts(np, ne).to_vec();
            self.lm_norm = alloc(ne);
            aether_dev_h2d_f32(nh.as_ptr() as i64, self.lm_norm, ci(ne));
            // output.weight (Q6_K)
            let wn = b"output.weight";
            let wi = aether_gguf_find_tensor_by_name(self.gguf_handle, wn.as_ptr() as i64, wn.len() as c_int);
            assert!(wi >= 0, "[qlora] output.weight missing");
            let dt = aether_gguf_get_tensor_dtype(self.gguf_handle, wi);
            assert_eq!(dt, 14, "[qlora] lm_head dtype {} != Q6_K(14); add dispatch", dt);
            let nb_row = d / 256;             // super-blocks per vocab row
            let bytes_row = nb_row * 210;     // Q6_K: 210 bytes / 256-elem block
            let wp = aether_gguf_get_tensor_data_ptr(self.gguf_handle, wi) as *const u8;
            self.lm_nb_row = nb_row;
            // Split the vocab rows into ~16 chunks; upload each chunk's bytes as a
            // separate Q6_K device buffer so the dequant transient is per-chunk.
            let n_chunks = 16usize;
            let rows_per = (vocab + n_chunks - 1) / n_chunks;
            let mut r0 = 0usize;
            while r0 < vocab {
                let r1 = (r0 + rows_per).min(vocab);
                let rows = r1 - r0;
                let chunk_bytes = rows * bytes_row;
                let h = aether_dev_alloc_u8(chunk_bytes as c_int);
                aether_dev_h2d_u8(wp.add(r0 * bytes_row) as i64, h, chunk_bytes as c_int);
                self.lm_chunks.push(h);
                self.lm_chunk_rows.push(rows);
                r0 = r1;
            }
            eprintln!("[qlora] lm_head loaded: output.weight Q6_K {}x{} in {} chunks (~{} rows ea), output_norm f32[{}]",
                vocab, d, self.lm_chunks.len(), rows_per, ne);
        }
    }

    /// Handles the free-fn loss needs (run_1f1b's loss closure can't borrow the
    /// &mut stage). Returns (lm_norm, chunk handles, chunk row counts, nb_per_row).
    pub fn lm_handles(&self) -> (i64, Vec<i64>, Vec<usize>, usize) {
        (self.lm_norm, self.lm_chunks.clone(), self.lm_chunk_rows.clone(), self.lm_nb_row)
    }
}

/// Real next-token LM loss for hidden [T, D] (last rank). `targets[t]` is the
/// gold token at position t (already shifted by the caller). Returns
/// (mean CE, d_hidden[T*D]). Base lm_head is FROZEN (no weight grad). Free fn so
/// run_1f1b's loss closure can call it without borrowing the &mut stage.
pub fn lm_head_loss(lm_norm: i64, chunks: &[i64], chunk_rows: &[usize], nb_row: usize,
                    vocab: usize, d: usize, t: usize, eps: f32,
                    hidden: &[f32], targets: &[i32]) -> (f32, Vec<f32>) {
    assert_eq!(hidden.len(), t * d);
    let xb = upload(hidden);
    let xf = alloc(t * d);
    aether_op_rms_norm_f32_cuda(xb, lm_norm, xf, eps, ci(t), ci(d));
    // FORWARD: build logits in [vocab, T] (logits_vt) so each chunk's rows are a
    // CONTIGUOUS slice. Per chunk: dequant Wc[rows,d] (~190 MB transient) ->
    // logits_vt[r0:r1,:] = Wc @ xf^T = matmul_backward_lhs(Wc, xf, scratch, rows, T, d)
    // -> d2d-copy scratch into logits_vt at the row offset (opaque handles can't be
    // written at an offset directly).
    let max_rows = *chunk_rows.iter().max().unwrap();
    let logits_vt = alloc(vocab * t);
    let scratch = alloc(max_rows * t);
    let mut r0 = 0usize;
    for (i, &rows) in chunk_rows.iter().enumerate() {
        let wc = alloc(rows * d);
        aether_op_dequant_q6_k_f32_cuda(chunks[i], wc, ci(nb_row * rows));
        aether_op_matmul_backward_lhs_f32_cuda(wc, xf, scratch, ci(rows), ci(t), ci(d));
        aether_rt::cuda::aether_dev_d2d_f32_offset(scratch, 0, logits_vt, (r0 * t) as c_int, (rows * t) as c_int);
        aether_dev_free_f32(wc); r0 += rows;
    }
    // transpose [vocab, T] -> [T, vocab] for CE (treat as [vocab, T, 1]).
    let logits = alloc(t * vocab);
    aether_op_transpose_021_f32_cuda(logits_vt, logits, ci(vocab), ci(t), 1);
    let tgt = aether_dev_alloc_i32(ci(t));
    unsafe { aether_dev_h2d_i32(targets.as_ptr() as i64, tgt, ci(t)); }
    let probs = alloc(t * vocab);
    let loss = aether_op_cross_entropy_f32_cuda(logits, tgt, probs, ci(t), ci(vocab));
    let d_logits = alloc(t * vocab);
    aether_op_cross_entropy_backward_f32_cuda(probs, tgt, d_logits, ci(t), ci(vocab));
    // transpose dlogits [T,vocab] -> [vocab,T] for chunked backward.
    let d_logits_vt = alloc(vocab * t);
    aether_op_transpose_021_f32_cuda(d_logits, d_logits_vt, ci(t), ci(vocab), 1);
    // BACKWARD: dxf[T,d] = sum_chunks d_logits_vt[r0:r1,:]^T @ Wc[rows,d]
    //   = matmul_backward_rhs(dl_vt_chunk[rows,T], Wc[rows,d], dxf_part[T,d], rows, T, d)
    let dxf = alloc(t * d);
    let dl_scratch = alloc(max_rows * t);
    let dxf_part = alloc(t * d);
    r0 = 0;
    for (i, &rows) in chunk_rows.iter().enumerate() {
        let wc = alloc(rows * d);
        aether_op_dequant_q6_k_f32_cuda(chunks[i], wc, ci(nb_row * rows));
        aether_rt::cuda::aether_dev_d2d_f32_offset(d_logits_vt, (r0 * t) as c_int, dl_scratch, 0, (rows * t) as c_int);
        aether_op_matmul_backward_rhs_f32_cuda(dl_scratch, wc, dxf_part, ci(rows), ci(t), ci(d));
        aether_op_add_inplace_f32_cuda(dxf, dxf_part, ci(t * d));
        aether_dev_free_f32(wc); r0 += rows;
    }
    let d_xb = alloc(t * d); let inv = alloc(t);
    aether_op_rms_norm_backward_dx_f32_cuda(xb, lm_norm, dxf, d_xb, inv, eps, ci(t), ci(d));
    let dh = download(d_xb, t * d);
    unsafe { aether_dev_sync(); }
    for h in [xb, xf, logits_vt, scratch, logits, probs, d_logits, d_logits_vt, dxf, dl_scratch, dxf_part, d_xb, inv] {
        aether_dev_free_f32(h);
    }
    aether_rt::cuda::aether_dev_free_i32(tgt);
    (loss, dh)
}

impl QwenQLoraStage {
    /// Write all adapter A/B (per layer) to `path` as a flat f32 dump with a tiny
    /// header. Order: per layer, for ad in [q,k,v,o,gate,up,down]: A then B.
    pub fn save_adapters(&self, path: &str) {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(self.rank_lo as u32).to_le_bytes());
        for l in &self.layers {
            for ad in [&l.ad_q, &l.ad_k, &l.ad_v, &l.ad_o, &l.ad_gate, &l.ad_up, &l.ad_down] {
                let a = download(ad.a, ad.n_in * ad.rank);
                let b = download(ad.b, ad.rank * ad.n_out);
                for v in a.iter().chain(b.iter()) { buf.extend_from_slice(&v.to_le_bytes()); }
            }
        }
        unsafe { aether_dev_sync(); }
        std::fs::write(path, &buf).expect("[qlora] save_adapters write failed");
        eprintln!("[qlora] saved adapters -> {} ({} bytes)", path, buf.len());
    }
}

impl Stage for QwenQLoraStage {
    fn input_dim(&self) -> usize { self.t * self.cfg.d_model }
    fn output_dim(&self) -> usize { self.t * self.cfg.d_model }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let td = self.input_dim();
        assert_eq!(input.len(), td, "[qlora] input dim mismatch");
        let mut cur = upload(input);
        let mut acts = Vec::with_capacity(self.layers.len());
        for i in 0..self.layers.len() {
            let lq = unsafe { &*(&self.layers[i] as *const LayerQ) };
            let (xout, act) = self.layer_forward(lq, cur);
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
        assert_eq!(grad.len(), td, "[qlora] grad dim mismatch");
        let acts = self.fifo.pop_front().expect("[qlora] backward with empty FIFO");
        let mut d_cur = upload(grad);
        for i in (0..self.layers.len()).rev() {
            let lq = unsafe { &*(&self.layers[i] as *const LayerQ) };
            d_cur = self.layer_backward(lq, &acts[i], d_cur);
        }
        let d_in = download(d_cur, td);
        aether_dev_free_f32(d_cur);
        for a in &acts { a.free(); }
        unsafe { aether_dev_sync(); }
        d_in
    }

    fn step(&mut self, lr: f32, opt_step: i64) {
        for l in &self.layers {
            for ad in [&l.ad_q, &l.ad_k, &l.ad_v, &l.ad_o, &l.ad_gate, &l.ad_up, &l.ad_down] {
                ad.step(lr, opt_step);
                ad.zero_grad();
            }
        }
        let _ = self.rank_lo;
        unsafe { aether_dev_sync(); }
    }
}
