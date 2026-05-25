//! matt-voice FR-18.6-real leg 2 — full LM-loss wrapper fwd+bwd grad-check.
//!
//! roadmap: P18
//!
//! Wraps the (already-proven) qwen3 block in the pieces that make a real scalar
//! language-model loss: token embedding lookup at the front, a final RMSNorm +
//! lm_head projection to vocab logits, and cross-entropy against target ids at
//! the back. This is the leg-2 finisher #2 — it proves embed_lookup /
//! embed_scatter_add + lm_head matmul + cross_entropy_fwd/bwd close the gradient
//! end-to-end, so a stack of blocks can be trained against next-token loss.
//!
//!   ids -> embed_lookup -> [MHA block] -> RMSNorm(gf) -> @w_lm -> logits ->
//!   cross_entropy(logits, targets) = scalar loss
//!
//! All learnable tensors (embed table, block weights/norms, final norm, lm_head)
//! are finite-diff verified. The embed table is checked only at rows referenced
//! by `ids` (unreferenced rows have exactly-zero gradient).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_alloc_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rms_norm_backward_dx_f32_cuda, aether_op_rms_norm_backward_gamma_f32_cuda,
    aether_op_matmul_f32_cuda,
    aether_op_matmul_backward_lhs_f32_cuda, aether_op_matmul_backward_rhs_f32_cuda,
    aether_op_rope_apply_f32_cuda, aether_op_rope_apply_backward_f32_cuda,
    aether_op_sdpa_causal_forward_f32_cuda, aether_op_sdpa_causal_backward_f32_cuda,
    aether_op_transpose_021_f32_cuda,
    aether_op_silu_f32_cuda, aether_op_silu_backward_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_embed_lookup_f32_cuda, aether_op_embed_scatter_add_f32_cuda,
    aether_op_cross_entropy_f32_cuda, aether_op_cross_entropy_backward_f32_cuda,
};

const T: usize = 4;        // sequence length / num tokens
const V: usize = 16;       // vocab
const H: usize = 2;        // heads
const HD: usize = 4;
const D: usize = H * HD;   // 8
const DFF: usize = 16;
const BASE: f32 = 10000.0;
const EPS: f32 = 1e-5;

const IDS: [i32; T] = [0, 1, 2, 3];      // token ids fed to the embedding
const TGT: [i32; T] = [2, 0, 3, 1];      // next-token targets for the CE loss

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

#[derive(Clone)]
struct Weights {
    emb: Vec<f32>,                // [V*D]
    g1: Vec<f32>,
    wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
    g2: Vec<f32>,
    wgate: Vec<f32>, wup: Vec<f32>, wdown: Vec<f32>,
    gf: Vec<f32>,                 // final norm [D]
    wlm: Vec<f32>,                // lm_head [D*V]
}

impl Weights {
    fn init() -> Self {
        Weights {
            emb: fill(1, V * D, 0.5),
            g1: vec![1.0; D],
            wq: fill(10, D * D, 0.3), wk: fill(11, D * D, 0.3),
            wv: fill(12, D * D, 0.3), wo: fill(13, D * D, 0.3),
            g2: vec![1.0; D],
            wgate: fill(14, D * DFF, 0.2), wup: fill(15, D * DFF, 0.2),
            wdown: fill(16, DFF * D, 0.2),
            gf: vec![1.0; D],
            wlm: fill(17, D * V, 0.3),
        }
    }
}

#[derive(Default)]
struct Grads {
    emb: Vec<f32>, g1: Vec<f32>, wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
    g2: Vec<f32>, wgate: Vec<f32>, wup: Vec<f32>, wdown: Vec<f32>,
    gf: Vec<f32>, wlm: Vec<f32>,
}

struct Pool { f: Vec<i64> }
impl Pool {
    fn new() -> Self { Pool { f: Vec::new() } }
    fn zeros(&mut self, n: usize) -> i64 {
        let h = aether_dev_alloc_f32(n as c_int);
        assert!(h >= 0, "alloc {} failed", n);
        self.f.push(h);
        h
    }
    fn up(&mut self, host: &[f32]) -> i64 {
        let h = self.zeros(host.len());
        unsafe { aether_dev_h2d_f32(host.as_ptr() as i64, h, host.len() as c_int); }
        h
    }
    fn up_i32(&mut self, host: &[i32]) -> i64 {
        let h = aether_dev_alloc_i32(host.len() as c_int);
        unsafe { aether_dev_h2d_i32(host.as_ptr() as i64, h, host.len() as c_int); }
        h
    }
}
impl Drop for Pool {
    fn drop(&mut self) { for &h in &self.f { aether_dev_free_f32(h); } }
}

fn dl(h: i64, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    unsafe { aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, n as c_int); aether_dev_sync(); }
    v
}

fn run(w: &Weights, grads: Option<&mut Grads>) -> f32 {
    let mut p = Pool::new();
    // upload
    let emb = p.up(&w.emb);
    let g1 = p.up(&w.g1); let wq = p.up(&w.wq); let wk = p.up(&w.wk);
    let wv = p.up(&w.wv); let wo = p.up(&w.wo); let g2 = p.up(&w.g2);
    let wgate = p.up(&w.wgate); let wup = p.up(&w.wup); let wdown = p.up(&w.wdown);
    let gf = p.up(&w.gf); let wlm = p.up(&w.wlm);
    let ids = p.up_i32(&IDS); let tgt = p.up_i32(&TGT);

    // ---- forward ----
    // embed: x = emb[ids]
    let xd = p.zeros(T * D);
    aether_op_embed_lookup_f32_cuda(emb, ids, xd, T as c_int, D as c_int);

    // ---- block (MHA, identical structure to cuda_qwen3_block_grad_check) ----
    let xn = p.zeros(T * D);
    aether_op_rms_norm_f32_cuda(xd, g1, xn, EPS, T as c_int, D as c_int);
    let q = p.zeros(T * D); let k = p.zeros(T * D); let v = p.zeros(T * D);
    aether_op_matmul_f32_cuda(xn, wq, q, T as c_int, D as c_int, D as c_int);
    aether_op_matmul_f32_cuda(xn, wk, k, T as c_int, D as c_int, D as c_int);
    aether_op_matmul_f32_cuda(xn, wv, v, T as c_int, D as c_int, D as c_int);
    aether_op_rope_apply_f32_cuda(q, T as c_int, H as c_int, HD as c_int, BASE, 0);
    aether_op_rope_apply_f32_cuda(k, T as c_int, H as c_int, HD as c_int, BASE, 0);
    let qt = p.zeros(T * D); let kt = p.zeros(T * D); let vt = p.zeros(T * D);
    aether_op_transpose_021_f32_cuda(q, qt, T as c_int, H as c_int, HD as c_int);
    aether_op_transpose_021_f32_cuda(k, kt, T as c_int, H as c_int, HD as c_int);
    aether_op_transpose_021_f32_cuda(v, vt, T as c_int, H as c_int, HD as c_int);
    let ot = p.zeros(T * D); let attn = p.zeros(H * T * T);
    aether_op_sdpa_causal_forward_f32_cuda(qt, kt, vt, ot, attn, H as c_int, T as c_int, HD as c_int);
    let o = p.zeros(T * D);
    aether_op_transpose_021_f32_cuda(ot, o, H as c_int, T as c_int, HD as c_int);
    let proj = p.zeros(T * D);
    aether_op_matmul_f32_cuda(o, wo, proj, T as c_int, D as c_int, D as c_int);
    let x1 = p.zeros(T * D);
    aether_op_add_inplace_f32_cuda(x1, xd, (T * D) as c_int);
    aether_op_add_inplace_f32_cuda(x1, proj, (T * D) as c_int);
    let xn2 = p.zeros(T * D);
    aether_op_rms_norm_f32_cuda(x1, g2, xn2, EPS, T as c_int, D as c_int);
    let gate = p.zeros(T * DFF); let up = p.zeros(T * DFF);
    aether_op_matmul_f32_cuda(xn2, wgate, gate, T as c_int, D as c_int, DFF as c_int);
    aether_op_matmul_f32_cuda(xn2, wup, up, T as c_int, D as c_int, DFF as c_int);
    let gate_pre = p.zeros(T * DFF);
    aether_op_add_inplace_f32_cuda(gate_pre, gate, (T * DFF) as c_int);
    aether_op_silu_f32_cuda(gate, (T * DFF) as c_int);
    let gate_act = p.zeros(T * DFF);
    aether_op_add_inplace_f32_cuda(gate_act, gate, (T * DFF) as c_int);
    aether_op_mul_inplace_f32_cuda(gate, up, (T * DFF) as c_int);
    let h_ffn = p.zeros(T * DFF);
    aether_op_add_inplace_f32_cuda(h_ffn, gate, (T * DFF) as c_int);
    let down = p.zeros(T * D);
    aether_op_matmul_f32_cuda(gate, wdown, down, T as c_int, DFF as c_int, D as c_int);
    let xb = p.zeros(T * D);   // block output
    aether_op_add_inplace_f32_cuda(xb, x1, (T * D) as c_int);
    aether_op_add_inplace_f32_cuda(xb, down, (T * D) as c_int);

    // ---- final norm + lm_head + CE ----
    let xf = p.zeros(T * D);
    aether_op_rms_norm_f32_cuda(xb, gf, xf, EPS, T as c_int, D as c_int);
    let logits = p.zeros(T * V);
    aether_op_matmul_f32_cuda(xf, wlm, logits, T as c_int, D as c_int, V as c_int);
    let probs = p.zeros(T * V);
    let loss = aether_op_cross_entropy_f32_cuda(logits, tgt, probs, T as c_int, V as c_int);
    aether_dev_sync();

    let grads = match grads { Some(g) => g, None => return loss };

    // ---- backward ----
    // CE: d_logits (already 1/T scaled)
    let d_logits = p.zeros(T * V);
    aether_op_cross_entropy_backward_f32_cuda(probs, tgt, d_logits, T as c_int, V as c_int);
    // lm_head: logits = xf @ wlm
    let dwlm = p.zeros(D * V);
    aether_op_matmul_backward_rhs_f32_cuda(xf, d_logits, dwlm, T as c_int, D as c_int, V as c_int);
    let d_xf = p.zeros(T * D);
    aether_op_matmul_backward_lhs_f32_cuda(d_logits, wlm, d_xf, T as c_int, D as c_int, V as c_int);
    // final rms_norm: xb -> xf
    let d_xb = p.zeros(T * D); let invf = p.zeros(T); let dgf = p.zeros(D);
    aether_op_rms_norm_backward_dx_f32_cuda(xb, gf, d_xf, d_xb, invf, EPS, T as c_int, D as c_int);
    aether_op_rms_norm_backward_gamma_f32_cuda(xb, d_xf, invf, dgf, T as c_int, D as c_int);

    // ---- block backward (mirror of capstone) ----
    // xb = x1 + down
    let d_x1 = p.zeros(T * D); let d_down = p.zeros(T * D);
    aether_op_add_inplace_f32_cuda(d_x1, d_xb, (T * D) as c_int);
    aether_op_add_inplace_f32_cuda(d_down, d_xb, (T * D) as c_int);
    let dwdown = p.zeros(DFF * D);
    aether_op_matmul_backward_rhs_f32_cuda(h_ffn, d_down, dwdown, T as c_int, DFF as c_int, D as c_int);
    let d_h = p.zeros(T * DFF);
    aether_op_matmul_backward_lhs_f32_cuda(d_down, wdown, d_h, T as c_int, DFF as c_int, D as c_int);
    let d_gate_act = p.zeros(T * DFF);
    aether_op_add_inplace_f32_cuda(d_gate_act, d_h, (T * DFF) as c_int);
    aether_op_mul_inplace_f32_cuda(d_gate_act, up, (T * DFF) as c_int);
    let d_up = p.zeros(T * DFF);
    aether_op_add_inplace_f32_cuda(d_up, d_h, (T * DFF) as c_int);
    aether_op_mul_inplace_f32_cuda(d_up, gate_act, (T * DFF) as c_int);
    let d_gate = p.zeros(T * DFF);
    aether_op_silu_backward_f32_cuda(gate_pre, d_gate_act, d_gate, (T * DFF) as c_int);
    let dwgate = p.zeros(D * DFF); let dwup = p.zeros(D * DFF);
    aether_op_matmul_backward_rhs_f32_cuda(xn2, d_gate, dwgate, T as c_int, D as c_int, DFF as c_int);
    aether_op_matmul_backward_rhs_f32_cuda(xn2, d_up, dwup, T as c_int, D as c_int, DFF as c_int);
    let d_xn2 = p.zeros(T * D); let d_xn2_b = p.zeros(T * D);
    aether_op_matmul_backward_lhs_f32_cuda(d_gate, wgate, d_xn2, T as c_int, D as c_int, DFF as c_int);
    aether_op_matmul_backward_lhs_f32_cuda(d_up, wup, d_xn2_b, T as c_int, D as c_int, DFF as c_int);
    aether_op_add_inplace_f32_cuda(d_xn2, d_xn2_b, (T * D) as c_int);
    let d_x1_ffn = p.zeros(T * D); let inv2 = p.zeros(T); let dg2 = p.zeros(D);
    aether_op_rms_norm_backward_dx_f32_cuda(x1, g2, d_xn2, d_x1_ffn, inv2, EPS, T as c_int, D as c_int);
    aether_op_rms_norm_backward_gamma_f32_cuda(x1, d_xn2, inv2, dg2, T as c_int, D as c_int);
    aether_op_add_inplace_f32_cuda(d_x1, d_x1_ffn, (T * D) as c_int);
    // x1 = xd + proj : d_proj = d_x1 ; d_xd starts as d_x1
    let d_proj = p.zeros(T * D);
    aether_op_add_inplace_f32_cuda(d_proj, d_x1, (T * D) as c_int);
    let dwo = p.zeros(D * D);
    aether_op_matmul_backward_rhs_f32_cuda(o, d_proj, dwo, T as c_int, D as c_int, D as c_int);
    let d_o = p.zeros(T * D);
    aether_op_matmul_backward_lhs_f32_cuda(d_proj, wo, d_o, T as c_int, D as c_int, D as c_int);
    let d_ot = p.zeros(T * D);
    aether_op_transpose_021_f32_cuda(d_o, d_ot, T as c_int, H as c_int, HD as c_int);
    let d_qt = p.zeros(T * D); let d_kt = p.zeros(T * D); let d_vt = p.zeros(T * D);
    let dscores = p.zeros(H * T * T);
    aether_op_sdpa_causal_backward_f32_cuda(qt, kt, vt, attn, d_ot,
        d_qt, d_kt, d_vt, dscores, H as c_int, T as c_int, HD as c_int);
    let d_q = p.zeros(T * D); let d_k = p.zeros(T * D); let d_v = p.zeros(T * D);
    aether_op_transpose_021_f32_cuda(d_qt, d_q, H as c_int, T as c_int, HD as c_int);
    aether_op_transpose_021_f32_cuda(d_kt, d_k, H as c_int, T as c_int, HD as c_int);
    aether_op_transpose_021_f32_cuda(d_vt, d_v, H as c_int, T as c_int, HD as c_int);
    aether_op_rope_apply_backward_f32_cuda(d_q, T as c_int, H as c_int, HD as c_int, BASE, 0);
    aether_op_rope_apply_backward_f32_cuda(d_k, T as c_int, H as c_int, HD as c_int, BASE, 0);
    let dwq = p.zeros(D * D); let dwk = p.zeros(D * D); let dwv = p.zeros(D * D);
    aether_op_matmul_backward_rhs_f32_cuda(xn, d_q, dwq, T as c_int, D as c_int, D as c_int);
    aether_op_matmul_backward_rhs_f32_cuda(xn, d_k, dwk, T as c_int, D as c_int, D as c_int);
    aether_op_matmul_backward_rhs_f32_cuda(xn, d_v, dwv, T as c_int, D as c_int, D as c_int);
    let d_xn = p.zeros(T * D); let d_xn_b = p.zeros(T * D); let d_xn_c = p.zeros(T * D);
    aether_op_matmul_backward_lhs_f32_cuda(d_q, wq, d_xn, T as c_int, D as c_int, D as c_int);
    aether_op_matmul_backward_lhs_f32_cuda(d_k, wk, d_xn_b, T as c_int, D as c_int, D as c_int);
    aether_op_matmul_backward_lhs_f32_cuda(d_v, wv, d_xn_c, T as c_int, D as c_int, D as c_int);
    aether_op_add_inplace_f32_cuda(d_xn, d_xn_b, (T * D) as c_int);
    aether_op_add_inplace_f32_cuda(d_xn, d_xn_c, (T * D) as c_int);
    // rmsnorm1: xd -> xn. d_xd += dx (and dg1)
    let d_x_attn = p.zeros(T * D); let inv1 = p.zeros(T); let dg1 = p.zeros(D);
    aether_op_rms_norm_backward_dx_f32_cuda(xd, g1, d_xn, d_x_attn, inv1, EPS, T as c_int, D as c_int);
    aether_op_rms_norm_backward_gamma_f32_cuda(xd, d_xn, inv1, dg1, T as c_int, D as c_int);
    // total d_xd = d_x1 (residual skip) + d_x_attn (through norm1)
    let d_xd = p.zeros(T * D);
    aether_op_add_inplace_f32_cuda(d_xd, d_x1, (T * D) as c_int);
    aether_op_add_inplace_f32_cuda(d_xd, d_x_attn, (T * D) as c_int);
    // embedding backward (scatter-add into zeroed d_emb)
    let d_emb = p.zeros(V * D);
    aether_op_embed_scatter_add_f32_cuda(d_xd, ids, d_emb, T as c_int, D as c_int);
    aether_dev_sync();

    grads.emb = dl(d_emb, V * D);
    grads.g1 = dl(dg1, D); grads.g2 = dl(dg2, D); grads.gf = dl(dgf, D);
    grads.wq = dl(dwq, D * D); grads.wk = dl(dwk, D * D);
    grads.wv = dl(dwv, D * D); grads.wo = dl(dwo, D * D);
    grads.wgate = dl(dwgate, D * DFF); grads.wup = dl(dwup, D * DFF);
    grads.wdown = dl(dwdown, DFF * D);
    grads.wlm = dl(dwlm, D * V);
    loss
}

#[test]
fn qwen3_lm_loss_gradients_match_finite_diff() {
    aether_dev_init();
    let w = Weights::init();
    let mut g = Grads::default();
    let loss = run(&w, Some(&mut g));
    assert!(loss.is_finite() && loss > 0.0, "loss not a finite positive scalar: {}", loss);

    let eps = 6e-3f32;
    let mut max_rel = 0.0f32;
    let mut checked = 0;

    macro_rules! check {
        ($field:ident, $len:expr, $stride:expr) => {{
            let analytic = &g.$field;
            let mut idx = 0usize;
            let mut field_max = 0.0f32;
            let mut worst = (0usize, 0.0f32, 0.0f32);
            while idx < $len {
                let mut wp = w.clone();
                let mut wm = w.clone();
                wp.$field[idx] += eps;
                wm.$field[idx] -= eps;
                let lp = run(&wp, None);
                let lm = run(&wm, None);
                let fd = (lp - lm) / (2.0 * eps);
                let a = analytic[idx];
                let rel = (fd - a).abs() / (a.abs() + 1e-2);
                if rel > max_rel { max_rel = rel; }
                if rel > field_max { field_max = rel; worst = (idx, a, fd); }
                checked += 1;
                idx += $stride;
            }
            eprintln!("  {:<6} max_rel={:.3e}  worst idx={} analytic={:.4e} fd={:.4e}",
                stringify!($field), field_max, worst.0, worst.1, worst.2);
        }};
    }
    check!(emb, T * D, 1);   // only rows referenced by IDS=[0,1,2,3] (= first T*D entries)
    check!(g1, D, 1);
    check!(g2, D, 1);
    check!(gf, D, 1);
    check!(wq, D * D, 5);
    check!(wk, D * D, 7);
    check!(wv, D * D, 5);
    check!(wo, D * D, 5);
    check!(wgate, D * DFF, 11);
    check!(wup, D * DFF, 13);
    check!(wdown, DFF * D, 11);
    check!(wlm, D * V, 7);

    eprintln!("[qwen3-LM-loss grad-check] loss={:.4} — checked {} entries across 12 tensors, max rel err = {:.3e}",
        loss, checked, max_rel);
    assert!(max_rel < 5e-2, "gradient check failed: max rel err {:.3e} >= 5e-2", max_rel);
}
