//! matt-voice FR-18.6-real leg 2 finisher #3 — QLoRA proj fwd+bwd grad-check.
//!
//! roadmap: P18
//!
//! Proves the QLoRA *training math* on the GPU: a linear projection whose base
//! weight W is FROZEN (receives no gradient) plus a trainable low-rank adapter
//!     y = base(W, x) + s · (x·A)·B,   s = alpha / rank
//! Only A and B train. The backward flows dx through BOTH the frozen base and
//! the adapter, and accumulates dA, dB. This is exactly the matt-voice fine-tune
//! shape (frozen quantized base + LoRA adapters); here the base is f32 so the
//! whole proj is finite-diff verifiable in one test. Swapping the base forward
//! /backward for the quant kernels (aether_op_fused_*_matmul_seq1 +
//! aether_op_quant_matmul_backward_lhs_f32_cuda, both parity-verified earlier)
//! is a mechanical substitution — the adapter math witnessed here is unchanged.
//!
//! A stored [in, rank], B stored [rank, out] (matmul-friendly layouts) so the
//! forward is two plain matmul_f32 calls. Mirrors the CPU LoraAdapter convention
//! (out += s·B·(A·x)) in trainer/src/lora.rs, just batched over T rows and with
//! A/B transposed for the row-major matmul.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_matmul_f32_cuda,
    aether_op_matmul_backward_lhs_f32_cuda, aether_op_matmul_backward_rhs_f32_cuda,
    aether_op_scale_f32_cuda, aether_op_add_inplace_f32_cuda,
};

const T: usize = 4;
const IN: usize = 8;
const OUT: usize = 8;
const RANK: usize = 2;
const ALPHA: f32 = 4.0;
const SCALE: f32 = ALPHA / RANK as f32; // 2.0

fn ci(n: usize) -> c_int { n as c_int }

fn fill(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        (((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
    }).collect()
}

#[derive(Clone)]
struct Adapter { a: Vec<f32>, b: Vec<f32> } // a:[IN*RANK], b:[RANK*OUT]

struct Pool { h: Vec<i64> }
impl Pool {
    fn new() -> Self { Pool { h: Vec::new() } }
    fn zeros(&mut self, n: usize) -> i64 {
        let h = aether_dev_alloc_f32(ci(n)); assert!(h >= 0); self.h.push(h); h
    }
    fn up(&mut self, host: &[f32]) -> i64 {
        let h = self.zeros(host.len());
        unsafe { aether_dev_h2d_f32(host.as_ptr() as i64, h, ci(host.len())); }
        h
    }
}
impl Drop for Pool { fn drop(&mut self) { for &h in &self.h { aether_dev_free_f32(h); } } }

fn dl(h: i64, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    unsafe { aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, ci(n)); aether_dev_sync(); }
    v
}

/// Frozen base W (constant across the whole test) + trainable adapter.
/// loss = 0.5 * sum(y^2). Returns loss; if grads, fills (dA, dB).
fn run(x: &[f32], w_base: &[f32], ad: &Adapter, grads: Option<(&mut Vec<f32>, &mut Vec<f32>)>) -> f32 {
    let mut p = Pool::new();
    let xd = p.up(x);
    let wb = p.up(w_base);          // [IN, OUT]
    let a = p.up(&ad.a);            // [IN, RANK]
    let b = p.up(&ad.b);            // [RANK, OUT]

    // forward
    let base = p.zeros(T * OUT);
    aether_op_matmul_f32_cuda(xd, wb, base, ci(T), ci(IN), ci(OUT));
    let a_x = p.zeros(T * RANK);
    aether_op_matmul_f32_cuda(xd, a, a_x, ci(T), ci(IN), ci(RANK));
    let delta = p.zeros(T * OUT);
    aether_op_matmul_f32_cuda(a_x, b, delta, ci(T), ci(RANK), ci(OUT));
    aether_op_scale_f32_cuda(delta, SCALE, ci(T * OUT));
    let y = p.zeros(T * OUT);
    aether_op_add_inplace_f32_cuda(y, base, ci(T * OUT));
    aether_op_add_inplace_f32_cuda(y, delta, ci(T * OUT));
    aether_dev_sync();

    let y_h = dl(y, T * OUT);
    let loss = 0.5 * y_h.iter().map(|v| v * v).sum::<f32>();

    let (g_a, g_b) = match grads { Some(g) => g, None => return loss };

    // backward: dL/dy = y
    let dy = p.up(&y_h);
    // d_delta = s * dy ; base path gets dy (W frozen — no dW computed)
    let d_delta = p.up(&y_h);
    aether_op_scale_f32_cuda(d_delta, SCALE, ci(T * OUT));
    // delta = a_x @ b : dB = a_x^T @ d_delta ; d_a_x = d_delta @ b^T
    let dB = p.zeros(RANK * OUT);
    aether_op_matmul_backward_rhs_f32_cuda(a_x, d_delta, dB, ci(T), ci(RANK), ci(OUT));
    let d_a_x = p.zeros(T * RANK);
    aether_op_matmul_backward_lhs_f32_cuda(d_delta, b, d_a_x, ci(T), ci(RANK), ci(OUT));
    // a_x = x @ a : dA = x^T @ d_a_x ; (d_x via base + adapter, not needed for the check)
    let dA = p.zeros(IN * RANK);
    aether_op_matmul_backward_rhs_f32_cuda(xd, d_a_x, dA, ci(T), ci(IN), ci(RANK));
    // dx would be: base_bwd_lhs(dy, W) [frozen base] + d_a_x @ a^T. We compute it
    // to exercise the full path but it is not part of the (param) grad check.
    let dx = p.zeros(T * IN);
    aether_op_matmul_backward_lhs_f32_cuda(dy, wb, dx, ci(T), ci(IN), ci(OUT));
    let dx_lora = p.zeros(T * IN);
    aether_op_matmul_backward_lhs_f32_cuda(d_a_x, a, dx_lora, ci(T), ci(IN), ci(RANK));
    aether_op_add_inplace_f32_cuda(dx, dx_lora, ci(T * IN));
    aether_dev_sync();

    *g_a = dl(dA, IN * RANK);
    *g_b = dl(dB, RANK * OUT);
    loss
}

#[test]
fn qlora_proj_adapter_gradients_match_finite_diff() {
    aether_dev_init();
    let x = fill(99, T * IN, 1.0);
    let w_base = fill(7, IN * OUT, 0.4);            // FROZEN base
    // A ~ small, B nonzero (so the adapter path is live — real init starts B=0).
    let ad = Adapter { a: fill(10, IN * RANK, 0.5), b: fill(11, RANK * OUT, 0.5) };

    let (mut g_a, mut g_b) = (Vec::new(), Vec::new());
    let _loss = run(&x, &w_base, &ad, Some((&mut g_a, &mut g_b)));

    let eps = 2e-3f32;
    let mut max_rel = 0.0f32;
    let mut checked = 0;

    // dA
    for idx in 0..(IN * RANK) {
        let mut ap = ad.clone(); let mut am = ad.clone();
        ap.a[idx] += eps; am.a[idx] -= eps;
        let lp = run(&x, &w_base, &ap, None);
        let lm = run(&x, &w_base, &am, None);
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - g_a[idx]).abs() / (g_a[idx].abs() + 1e-2);
        if rel > max_rel { max_rel = rel; }
        checked += 1;
    }
    // dB
    for idx in 0..(RANK * OUT) {
        let mut bp = ad.clone(); let mut bm = ad.clone();
        bp.b[idx] += eps; bm.b[idx] -= eps;
        let lp = run(&x, &w_base, &bp, None);
        let lm = run(&x, &w_base, &bm, None);
        let fd = (lp - lm) / (2.0 * eps);
        let rel = (fd - g_b[idx]).abs() / (g_b[idx].abs() + 1e-2);
        if rel > max_rel { max_rel = rel; }
        checked += 1;
    }

    eprintln!("[QLoRA proj grad-check] frozen base [{},{}] + adapter rank={} scale={} — checked {} adapter params, max rel err = {:.3e}",
        IN, OUT, RANK, SCALE, checked, max_rel);
    assert!(max_rel < 2e-2, "adapter gradient check failed: max rel err {:.3e}", max_rel);
}
