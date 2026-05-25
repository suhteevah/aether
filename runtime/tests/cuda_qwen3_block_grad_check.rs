//! matt-voice FR-18.6-real leg 2 — qwen3 transformer block fwd+bwd composition.
//!
//! roadmap: P18
//!
//! Assembles a full qwen3-style block on the GPU from the leg-2 kernels:
//!   x -> RMSNorm(g1) -> Q/K/V proj -> RoPE(Q,K) -> [s,h,hd]->[h,s,hd] ->
//!   causal SDPA -> ->[s,h,hd] -> O proj -> +x (residual) = x1 ->
//!   RMSNorm(g2) -> SwiGLU(gate,up,down) -> +x1 = out
//! loss = 0.5 * sum(out^2). The backward chains rms_norm_bwd, matmul_bwd,
//! rope_bwd, sdpa_bwd, silu_bwd, transpose_021 in reverse.
//!
//! This is the composition witness: each kernel is parity-tested in isolation
//! elsewhere; here we prove they assemble into a block whose analytic
//! gradients match central finite differences. MHA (n_kv == n_q) — the GQA
//! repeat/reduce is a separate follow-on. Weights are f32, stored [in,out] so
//! matmul_f32 (out[m,n]=a[m,k]@b[k,n]) applies directly.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rms_norm_backward_dx_f32_cuda, aether_op_rms_norm_backward_gamma_f32_cuda,
    aether_op_matmul_f32_cuda,
    aether_op_matmul_backward_lhs_f32_cuda, aether_op_matmul_backward_rhs_f32_cuda,
    aether_op_rope_apply_f32_cuda, aether_op_rope_apply_backward_f32_cuda,
    aether_op_sdpa_causal_forward_f32_cuda, aether_op_sdpa_causal_backward_f32_cuda,
    aether_op_transpose_021_f32_cuda,
    aether_op_silu_f32_cuda, aether_op_silu_backward_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_add_inplace_f32_cuda,
};

const S: usize = 4;
const H: usize = 2;
const HD: usize = 4;
const D: usize = H * HD; // 8
const DFF: usize = 16;
const BASE: f32 = 10000.0;
const EPS: f32 = 1e-5;

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

#[derive(Clone)]
struct Weights {
    g1: Vec<f32>,                 // [D]
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>, // [D*D] each ([in,out])
    g2: Vec<f32>,                 // [D]
    wgate: Vec<f32>, wup: Vec<f32>, // [D*DFF]
    wdown: Vec<f32>,              // [DFF*D]
}

impl Weights {
    fn init() -> Self {
        Weights {
            g1: vec![1.0; D],
            wq: fill(10, D * D, 0.3), wk: fill(11, D * D, 0.3),
            wv: fill(12, D * D, 0.3), wo: fill(13, D * D, 0.3),
            g2: vec![1.0; D],
            wgate: fill(14, D * DFF, 0.2), wup: fill(15, D * DFF, 0.2),
            wdown: fill(16, DFF * D, 0.2),
        }
    }
}

struct Grads {
    g1: Vec<f32>, wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
    g2: Vec<f32>, wgate: Vec<f32>, wup: Vec<f32>, wdown: Vec<f32>,
}

/// Buffer pool: alloc tracked f32 device buffers, free all at drop.
struct Pool { handles: Vec<i64> }
impl Pool {
    fn new() -> Self { Pool { handles: Vec::new() } }
    fn zeros(&mut self, n: usize) -> i64 {
        let h = aether_dev_alloc_f32(n as c_int);
        assert!(h >= 0, "alloc {} failed", n);
        self.handles.push(h);
        h
    }
    fn up(&mut self, host: &[f32]) -> i64 {
        let h = self.zeros(host.len());
        unsafe { aether_dev_h2d_f32(host.as_ptr() as i64, h, host.len() as c_int); }
        h
    }
}
impl Drop for Pool {
    fn drop(&mut self) { for &h in &self.handles { unsafe { aether_dev_free_f32(h); } } }
}

fn dl(h: i64, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    unsafe { aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, n as c_int); aether_dev_sync(); }
    v
}

/// Forward through the block. Returns scalar loss. If `grads` is Some, also
/// runs the backward and fills it.
fn run(x: &[f32], w: &Weights, grads: Option<&mut Grads>) -> f32 {
    let mut p = Pool::new();
    unsafe {
        // ---- upload weights + input ----
        let xd = p.up(x);
        let g1 = p.up(&w.g1); let wq = p.up(&w.wq); let wk = p.up(&w.wk);
        let wv = p.up(&w.wv); let wo = p.up(&w.wo); let g2 = p.up(&w.g2);
        let wgate = p.up(&w.wgate); let wup = p.up(&w.wup); let wdown = p.up(&w.wdown);

        // ---- forward ----
        let xn = p.zeros(S * D);
        aether_op_rms_norm_f32_cuda(xd, g1, xn, EPS, S as c_int, D as c_int);
        let q = p.zeros(S * D); let k = p.zeros(S * D); let v = p.zeros(S * D);
        aether_op_matmul_f32_cuda(xn, wq, q, S as c_int, D as c_int, D as c_int);
        aether_op_matmul_f32_cuda(xn, wk, k, S as c_int, D as c_int, D as c_int);
        aether_op_matmul_f32_cuda(xn, wv, v, S as c_int, D as c_int, D as c_int);
        aether_op_rope_apply_f32_cuda(q, S as c_int, H as c_int, HD as c_int, BASE, 0);
        aether_op_rope_apply_f32_cuda(k, S as c_int, H as c_int, HD as c_int, BASE, 0);
        // [s,h,hd] -> [h,s,hd]
        let qt = p.zeros(S * D); let kt = p.zeros(S * D); let vt = p.zeros(S * D);
        aether_op_transpose_021_f32_cuda(q, qt, S as c_int, H as c_int, HD as c_int);
        aether_op_transpose_021_f32_cuda(k, kt, S as c_int, H as c_int, HD as c_int);
        aether_op_transpose_021_f32_cuda(v, vt, S as c_int, H as c_int, HD as c_int);
        let ot = p.zeros(S * D); let attn = p.zeros(H * S * S);
        aether_op_sdpa_causal_forward_f32_cuda(qt, kt, vt, ot, attn, H as c_int, S as c_int, HD as c_int);
        // [h,s,hd] -> [s,h,hd]
        let o = p.zeros(S * D);
        aether_op_transpose_021_f32_cuda(ot, o, H as c_int, S as c_int, HD as c_int);
        let proj = p.zeros(S * D);
        aether_op_matmul_f32_cuda(o, wo, proj, S as c_int, D as c_int, D as c_int);
        // x1 = x + proj
        let x1 = p.up(x);
        aether_op_add_inplace_f32_cuda(x1, proj, (S * D) as c_int);
        // FFN
        let xn2 = p.zeros(S * D);
        aether_op_rms_norm_f32_cuda(x1, g2, xn2, EPS, S as c_int, D as c_int);
        let gate = p.zeros(S * DFF); let up = p.zeros(S * DFF);
        aether_op_matmul_f32_cuda(xn2, wgate, gate, S as c_int, D as c_int, DFF as c_int);
        aether_op_matmul_f32_cuda(xn2, wup, up, S as c_int, D as c_int, DFF as c_int);
        let gate_pre = p.zeros(S * DFF); // save pre-silu for backward
        aether_op_add_inplace_f32_cuda(gate_pre, gate, (S * DFF) as c_int); // gate_pre = gate (copy via +0)
        aether_op_silu_f32_cuda(gate, (S * DFF) as c_int);  // gate := silu(gate_pre)
        let gate_act = p.zeros(S * DFF);
        aether_op_add_inplace_f32_cuda(gate_act, gate, (S * DFF) as c_int); // save silu(gate)
        aether_op_mul_inplace_f32_cuda(gate, up, (S * DFF) as c_int); // h_ffn = silu(gate)*up
        let h_ffn = p.zeros(S * DFF);
        aether_op_add_inplace_f32_cuda(h_ffn, gate, (S * DFF) as c_int); // save h_ffn
        let down = p.zeros(S * D);
        aether_op_matmul_f32_cuda(gate, wdown, down, S as c_int, DFF as c_int, D as c_int);
        // out = x1 + down
        let out = p.zeros(S * D);
        aether_op_add_inplace_f32_cuda(out, x1, (S * D) as c_int);
        aether_op_add_inplace_f32_cuda(out, down, (S * D) as c_int);
        aether_dev_sync();

        let out_h = dl(out, S * D);
        let loss: f32 = 0.5 * out_h.iter().map(|v| v * v).sum::<f32>();

        let grads = match grads { Some(g) => g, None => return loss };

        // ---- backward ----  dL/dout = out
        // out = x1 + down: d_x1 (init) = dout ; d_down = dout
        let d_x1 = p.up(&out_h);
        let d_down = p.up(&out_h);
        // down = h_ffn @ wdown : dwdown = h_ffn^T @ d_down ; d_h = d_down @ wdown^T
        let dwdown = p.zeros(DFF * D);
        aether_op_matmul_backward_rhs_f32_cuda(h_ffn, d_down, dwdown, S as c_int, DFF as c_int, D as c_int);
        let d_h = p.zeros(S * DFF);
        aether_op_matmul_backward_lhs_f32_cuda(d_down, wdown, d_h, S as c_int, DFF as c_int, D as c_int);
        // h_ffn = silu(gate)*up : d_gate_act = d_h*up ; d_up = d_h*gate_act
        let d_gate_act = p.zeros(S * DFF);
        aether_op_add_inplace_f32_cuda(d_gate_act, d_h, (S * DFF) as c_int);
        aether_op_mul_inplace_f32_cuda(d_gate_act, up, (S * DFF) as c_int);
        let d_up = p.zeros(S * DFF);
        aether_op_add_inplace_f32_cuda(d_up, d_h, (S * DFF) as c_int);
        aether_op_mul_inplace_f32_cuda(d_up, gate_act, (S * DFF) as c_int);
        // d_gate = silu_backward(gate_pre, d_gate_act)
        let d_gate = p.zeros(S * DFF);
        aether_op_silu_backward_f32_cuda(gate_pre, d_gate_act, d_gate, (S * DFF) as c_int);
        // gate = xn2@wgate, up = xn2@wup
        let dwgate = p.zeros(D * DFF); let dwup = p.zeros(D * DFF);
        aether_op_matmul_backward_rhs_f32_cuda(xn2, d_gate, dwgate, S as c_int, D as c_int, DFF as c_int);
        aether_op_matmul_backward_rhs_f32_cuda(xn2, d_up, dwup, S as c_int, D as c_int, DFF as c_int);
        let d_xn2 = p.zeros(S * D); let d_xn2_b = p.zeros(S * D);
        aether_op_matmul_backward_lhs_f32_cuda(d_gate, wgate, d_xn2, S as c_int, D as c_int, DFF as c_int);
        aether_op_matmul_backward_lhs_f32_cuda(d_up, wup, d_xn2_b, S as c_int, D as c_int, DFF as c_int);
        aether_op_add_inplace_f32_cuda(d_xn2, d_xn2_b, (S * D) as c_int);
        // rmsnorm2: x1 -> xn2. d_x1 += dx ; dg2 = gamma-grad
        let d_x1_ffn = p.zeros(S * D); let inv2 = p.zeros(S); let dg2 = p.zeros(D);
        aether_op_rms_norm_backward_dx_f32_cuda(x1, g2, d_xn2, d_x1_ffn, inv2, EPS, S as c_int, D as c_int);
        aether_op_rms_norm_backward_gamma_f32_cuda(x1, d_xn2, inv2, dg2, S as c_int, D as c_int);
        aether_op_add_inplace_f32_cuda(d_x1, d_x1_ffn, (S * D) as c_int);
        // attention residual: x1 = x + proj -> d_proj = d_x1
        let d_proj = d_x1;
        // proj = o@wo : dwo = o^T@d_proj ; d_o = d_proj@wo^T
        let dwo = p.zeros(D * D);
        aether_op_matmul_backward_rhs_f32_cuda(o, d_proj, dwo, S as c_int, D as c_int, D as c_int);
        let d_o = p.zeros(S * D);
        aether_op_matmul_backward_lhs_f32_cuda(d_proj, wo, d_o, S as c_int, D as c_int, D as c_int);
        // [s,h,hd] -> [h,s,hd]
        let d_ot = p.zeros(S * D);
        aether_op_transpose_021_f32_cuda(d_o, d_ot, S as c_int, H as c_int, HD as c_int);
        // sdpa backward
        let d_qt = p.zeros(S * D); let d_kt = p.zeros(S * D); let d_vt = p.zeros(S * D);
        let dscores = p.zeros(H * S * S);
        aether_op_sdpa_causal_backward_f32_cuda(qt, kt, vt, attn, d_ot,
            d_qt, d_kt, d_vt, dscores, H as c_int, S as c_int, HD as c_int);
        // [h,s,hd] -> [s,h,hd]
        let d_q = p.zeros(S * D); let d_k = p.zeros(S * D); let d_v = p.zeros(S * D);
        aether_op_transpose_021_f32_cuda(d_qt, d_q, H as c_int, S as c_int, HD as c_int);
        aether_op_transpose_021_f32_cuda(d_kt, d_k, H as c_int, S as c_int, HD as c_int);
        aether_op_transpose_021_f32_cuda(d_vt, d_v, H as c_int, S as c_int, HD as c_int);
        // rope backward on d_q, d_k (in place)
        aether_op_rope_apply_backward_f32_cuda(d_q, S as c_int, H as c_int, HD as c_int, BASE, 0);
        aether_op_rope_apply_backward_f32_cuda(d_k, S as c_int, H as c_int, HD as c_int, BASE, 0);
        // q/k/v = xn@w{q,k,v} : dw = xn^T@d_; d_xn += d_@w^T
        let dwq = p.zeros(D * D); let dwk = p.zeros(D * D); let dwv = p.zeros(D * D);
        aether_op_matmul_backward_rhs_f32_cuda(xn, d_q, dwq, S as c_int, D as c_int, D as c_int);
        aether_op_matmul_backward_rhs_f32_cuda(xn, d_k, dwk, S as c_int, D as c_int, D as c_int);
        aether_op_matmul_backward_rhs_f32_cuda(xn, d_v, dwv, S as c_int, D as c_int, D as c_int);
        let d_xn = p.zeros(S * D); let d_xn_b = p.zeros(S * D); let d_xn_c = p.zeros(S * D);
        aether_op_matmul_backward_lhs_f32_cuda(d_q, wq, d_xn, S as c_int, D as c_int, D as c_int);
        aether_op_matmul_backward_lhs_f32_cuda(d_k, wk, d_xn_b, S as c_int, D as c_int, D as c_int);
        aether_op_matmul_backward_lhs_f32_cuda(d_v, wv, d_xn_c, S as c_int, D as c_int, D as c_int);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_b, (S * D) as c_int);
        aether_op_add_inplace_f32_cuda(d_xn, d_xn_c, (S * D) as c_int);
        // rmsnorm1: dg1
        let d_x_attn = p.zeros(S * D); let inv1 = p.zeros(S); let dg1 = p.zeros(D);
        aether_op_rms_norm_backward_dx_f32_cuda(xd, g1, d_xn, d_x_attn, inv1, EPS, S as c_int, D as c_int);
        aether_op_rms_norm_backward_gamma_f32_cuda(xd, d_xn, inv1, dg1, S as c_int, D as c_int);
        aether_dev_sync();

        grads.g1 = dl(dg1, D);   grads.g2 = dl(dg2, D);
        grads.wq = dl(dwq, D * D); grads.wk = dl(dwk, D * D);
        grads.wv = dl(dwv, D * D); grads.wo = dl(dwo, D * D);
        grads.wgate = dl(dwgate, D * DFF); grads.wup = dl(dwup, D * DFF);
        grads.wdown = dl(dwdown, DFF * D);
        loss
    }
}

#[test]
fn qwen3_block_gradients_match_finite_diff() {
    aether_dev_init();
    let x = fill(99, S * D, 1.0);
    let w = Weights::init();

    let mut g = Grads {
        g1: vec![], wq: vec![], wk: vec![], wv: vec![], wo: vec![],
        g2: vec![], wgate: vec![], wup: vec![], wdown: vec![],
    };
    let _loss = run(&x, &w, Some(&mut g));

    // Central finite difference on a sample of entries across each weight.
    let eps = 1e-3f32;
    let mut max_rel = 0.0f32;
    let mut checked = 0;

    // (name, analytic grad, accessor to mutate a clone)
    macro_rules! check {
        ($field:ident, $len:expr, $stride:expr) => {{
            let analytic = &g.$field;
            let mut idx = 0usize;
            while idx < $len {
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp.$field[idx] += eps;
                wm.$field[idx] -= eps;
                let lp = run(&x, &wp, None);
                let lm = run(&x, &wm, None);
                let fd = (lp - lm) / (2.0 * eps);
                let a = analytic[idx];
                let rel = (fd - a).abs() / (a.abs() + 1e-2);
                if rel > max_rel { max_rel = rel; }
                checked += 1;
                idx += $stride;
            }
        }};
    }
    check!(g1, D, 1);
    check!(g2, D, 1);
    check!(wq, D * D, 5);
    check!(wk, D * D, 7);
    check!(wv, D * D, 5);
    check!(wo, D * D, 5);
    check!(wgate, D * DFF, 11);
    check!(wup, D * DFF, 13);
    check!(wdown, DFF * D, 11);

    eprintln!("[qwen3-block grad-check] checked {} entries across 9 tensors, max rel err = {:.3e}",
        checked, max_rel);
    assert!(max_rel < 5e-2, "gradient check failed: max rel err {:.3e} >= 5e-2", max_rel);
}
