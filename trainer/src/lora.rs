//! LoRA (Low-Rank Adaptation) adapter — the foundation for parameter-efficient
//! fine-tuning of a frozen base model. This file is pure CPU f32 orchestration;
//! the only math call out is `aether_rt::ops::adamw_step_f32` for the optimiser
//! step (same primitive `model.rs` uses on the full param arena).
//!
//! Math (with scale `s = alpha / rank`):
//!   forward:   out += s * B @ (A @ x)
//!     A : [rank, in_dim]   (row-major, row r = A[r*in .. r*in+in])
//!     B : [out_dim, rank]  (row-major, row o = B[o*rank .. o*rank+rank])
//!     a_x = A @ x          [rank]    (saved for backward)
//!     delta = s * (B @ a_x) [out_dim]
//!   backward (loss gradient `grad_out` w.r.t. `out`):
//!     dB += s * grad_out (outer) a_x^T       -> dB[o,r] += s * grad_out[o] * a_x[r]
//!     dA += s * (B^T grad_out) (outer) x^T   -> dA[r,i] += s * (sum_o B[o,r]*grad_out[o]) * x[i]
//!     dx += s * A^T B^T grad_out             -> dx[i]  += s * sum_r A[r,i] * (sum_o B[o,r]*grad_out[o])
//!
//! Standard LoRA init: A ~ N(0, 1/rank) (kaiming-ish on the rank fan-in), B = 0,
//! so the initial delta is exactly zero and the frozen base is unperturbed at
//! step 0.
//!
//! The A and B parameters are stored in their own arrays; their gradients and
//! Adam moments mirror them. `grads_flat*` exposes the concatenated [dA; dB]
//! vector so a data-parallel wrapper can all-reduce every adapter's gradients
//! as one contiguous buffer.

use aether_rt::ops;
use crate::rng::Rng;

/// A single LoRA adapter attached to one frozen linear layer (in_dim -> out_dim).
#[derive(Clone)]
pub struct LoraAdapter {
    pub name: String,
    pub in_dim: usize,
    pub out_dim: usize,
    pub rank: usize,
    pub alpha: f32,

    // Learnable params.
    pub a: Vec<f32>, // [rank * in_dim]
    pub b: Vec<f32>, // [out_dim * rank]

    // Gradients (same shapes as a / b).
    pub grad_a: Vec<f32>,
    pub grad_b: Vec<f32>,

    // AdamW first/second moment estimates.
    pub m_a: Vec<f32>,
    pub v_a: Vec<f32>,
    pub m_b: Vec<f32>,
    pub v_b: Vec<f32>,
}

impl LoraAdapter {
    /// `s = alpha / rank` — the LoRA scaling factor applied to the delta.
    #[inline]
    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    #[inline]
    pub fn a_len(&self) -> usize { self.rank * self.in_dim }
    #[inline]
    pub fn b_len(&self) -> usize { self.out_dim * self.rank }
    /// Total learnable element count ([A] + [B]).
    #[inline]
    pub fn n_params(&self) -> usize { self.a_len() + self.b_len() }

    /// Allocate + init an adapter. A is filled with N(0, 1/sqrt(rank)) (kaiming
    /// on the rank fan-in), B is all zeros (so delta = 0 at init — standard LoRA).
    pub fn new(name: impl Into<String>, in_dim: usize, out_dim: usize, rank: usize, alpha: f32, rng: &mut Rng) -> Self {
        let name = name.into();
        assert!(rank > 0, "lora rank must be > 0");
        assert!(in_dim > 0 && out_dim > 0, "lora dims must be > 0");

        let a_len = rank * in_dim;
        let b_len = out_dim * rank;
        let std = 1.0f32 / (rank as f32).sqrt();

        let mut a = vec![0.0f32; a_len];
        for x in a.iter_mut() { *x = rng.next_normal() * std; }
        let b = vec![0.0f32; b_len]; // zeros: initial delta is exactly 0.

        eprintln!(
            "[lora] init adapter '{}' in={} out={} rank={} alpha={} scale={:.4} (A~N(0,{:.4}^2), B=0) params={}",
            name, in_dim, out_dim, rank, alpha, alpha / rank as f32, std, a_len + b_len,
        );

        Self {
            name,
            in_dim, out_dim, rank, alpha,
            a, b,
            grad_a: vec![0.0f32; a_len],
            grad_b: vec![0.0f32; b_len],
            m_a: vec![0.0f32; a_len],
            v_a: vec![0.0f32; a_len],
            m_b: vec![0.0f32; b_len],
            v_b: vec![0.0f32; b_len],
        }
    }

    /// out += s * B @ (A @ x). Returns the saved intermediate `a_x = A @ x`
    /// ([rank]) which `backward` needs.
    ///
    /// `x` is [in_dim]; `base_out` is [out_dim] and is ACCUMULATED into (the
    /// caller has already written the frozen base linear's output there).
    pub fn forward(&self, x: &[f32], base_out: &mut [f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.in_dim, "lora '{}' forward: x len {} != in_dim {}", self.name, x.len(), self.in_dim);
        assert_eq!(base_out.len(), self.out_dim, "lora '{}' forward: base_out len {} != out_dim {}", self.name, base_out.len(), self.out_dim);

        let s = self.scale();

        // a_x = A @ x   -> [rank]
        let mut a_x = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            let row = &self.a[r * self.in_dim..(r + 1) * self.in_dim];
            let mut acc = 0.0f32;
            for i in 0..self.in_dim { acc += row[i] * x[i]; }
            a_x[r] = acc;
        }

        // base_out += s * (B @ a_x)
        for o in 0..self.out_dim {
            let brow = &self.b[o * self.rank..(o + 1) * self.rank];
            let mut acc = 0.0f32;
            for r in 0..self.rank { acc += brow[r] * a_x[r]; }
            base_out[o] += s * acc;
        }

        a_x
    }

    /// Accumulate grad_a / grad_b and add the adapter's contribution to `grad_in`
    /// (d-loss / d-x). `a_x` must be the intermediate returned by the matching
    /// `forward`. `grad_out` is d-loss / d-out ([out_dim]); `grad_in` is
    /// ACCUMULATED into ([in_dim]).
    pub fn backward(&mut self, x: &[f32], a_x: &[f32], grad_out: &[f32], grad_in: &mut [f32]) {
        assert_eq!(x.len(), self.in_dim, "lora '{}' backward: x len mismatch", self.name);
        assert_eq!(a_x.len(), self.rank, "lora '{}' backward: a_x len mismatch", self.name);
        assert_eq!(grad_out.len(), self.out_dim, "lora '{}' backward: grad_out len mismatch", self.name);
        assert_eq!(grad_in.len(), self.in_dim, "lora '{}' backward: grad_in len mismatch", self.name);

        let s = self.scale();

        // bt_go[r] = sum_o B[o,r] * grad_out[o]   (B^T grad_out)  -> [rank]
        let mut bt_go = vec![0.0f32; self.rank];
        for o in 0..self.out_dim {
            let go = grad_out[o];
            let brow = &self.b[o * self.rank..(o + 1) * self.rank];
            for r in 0..self.rank { bt_go[r] += brow[r] * go; }
        }

        // dB[o,r] += s * grad_out[o] * a_x[r]
        for o in 0..self.out_dim {
            let go = grad_out[o];
            let off = o * self.rank;
            for r in 0..self.rank {
                self.grad_b[off + r] += s * go * a_x[r];
            }
        }

        // dA[r,i] += s * bt_go[r] * x[i]
        for r in 0..self.rank {
            let g = s * bt_go[r];
            let off = r * self.in_dim;
            for i in 0..self.in_dim {
                self.grad_a[off + i] += g * x[i];
            }
        }

        // dx[i] += s * sum_r A[r,i] * bt_go[r]
        for r in 0..self.rank {
            let g = s * bt_go[r];
            let off = r * self.in_dim;
            for i in 0..self.in_dim {
                grad_in[i] += g * self.a[off + i];
            }
        }
    }

    /// AdamW update on A and B from their accumulated gradients. Reuses the
    /// runtime's `adamw_step_f32` primitive (same one `model.rs` uses).
    pub fn adamw_step(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: i64) {
        let a_len = self.a_len();
        let b_len = self.b_len();
        unsafe {
            ops::adamw_step_f32(
                self.a.as_mut_ptr(), self.grad_a.as_ptr(),
                self.m_a.as_mut_ptr(), self.v_a.as_mut_ptr(),
                lr, b1, b2, eps, wd, step, a_len,
            );
            ops::adamw_step_f32(
                self.b.as_mut_ptr(), self.grad_b.as_ptr(),
                self.m_b.as_mut_ptr(), self.v_b.as_mut_ptr(),
                lr, b1, b2, eps, wd, step, b_len,
            );
        }
    }

    /// Concatenated [dA; dB] gradient vector (read-only). NOTE: this is a fresh
    /// allocation since grad_a / grad_b are separate Vecs; use this to snapshot
    /// before all-reduce. For the in-place scatter path see `grads_flat_mut`.
    pub fn grads_flat(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_params());
        out.extend_from_slice(&self.grad_a);
        out.extend_from_slice(&self.grad_b);
        out
    }

    /// Write a flat [dA; dB] vector back into grad_a / grad_b (the inverse of
    /// `grads_flat`). Used by the DP wrapper to scatter the reduced gradients.
    pub fn grads_flat_mut(&mut self, flat: &[f32]) {
        let a_len = self.a_len();
        let b_len = self.b_len();
        assert_eq!(flat.len(), a_len + b_len, "lora '{}' grads_flat_mut: len {} != {}", self.name, flat.len(), a_len + b_len);
        self.grad_a.copy_from_slice(&flat[..a_len]);
        self.grad_b.copy_from_slice(&flat[a_len..]);
    }

    /// Zero both gradient buffers (call before each backward accumulation).
    pub fn zero_grad(&mut self) {
        for g in self.grad_a.iter_mut() { *g = 0.0; }
        for g in self.grad_b.iter_mut() { *g = 0.0; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Finite-difference check of the analytic backward against numerical
    /// gradients. Loss = sum(base_out) where base_out starts at zero and the
    /// adapter delta is added in forward. With loss = sum(out), grad_out is all
    /// ones, so dA/dB from `backward` must match the numerical derivative of
    /// the scalar loss w.r.t. each A/B element.
    #[test]
    fn lora_backward_finite_diff() {
        let in_dim = 8usize;
        let out_dim = 6usize;
        let rank = 4usize;
        let alpha = 2.0f32;

        let mut rng = Rng::new(0x10A4_u64 ^ 0xDEAD_BEEF);
        let mut ad = LoraAdapter::new("test", in_dim, out_dim, rank, alpha, &mut rng);

        // Force B to be non-zero so dA is exercised (B starts at zero, which
        // would zero the dx and dA paths and make the test vacuous).
        for x in ad.b.iter_mut() { *x = rng.next_normal() * 0.3; }

        // Random input.
        let x: Vec<f32> = (0..in_dim).map(|_| rng.next_normal() * 0.5).collect();

        // Scalar loss = sum(base_out). base_out is zeroed before each eval.
        let loss_of = |ad: &LoraAdapter| -> f32 {
            let mut out = vec![0.0f32; out_dim];
            let _ = ad.forward(&x, &mut out);
            out.iter().sum::<f32>()
        };

        // Analytic gradients: grad_out = ones (d sum / d out_o = 1).
        let mut out = vec![0.0f32; out_dim];
        let a_x = ad.forward(&x, &mut out);
        let grad_out = vec![1.0f32; out_dim];
        let mut grad_in = vec![0.0f32; in_dim];
        ad.zero_grad();
        ad.backward(&x, &a_x, &grad_out, &mut grad_in);

        let eps = 1e-3f32;
        let mut max_diff = 0.0f32;

        // Check dA.
        for i in 0..ad.a_len() {
            let orig = ad.a[i];
            ad.a[i] = orig + eps;
            let lp = loss_of(&ad);
            ad.a[i] = orig - eps;
            let lm = loss_of(&ad);
            ad.a[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            let diff = (num - ad.grad_a[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }

        // Check dB.
        for i in 0..ad.b_len() {
            let orig = ad.b[i];
            ad.b[i] = orig + eps;
            let lp = loss_of(&ad);
            ad.b[i] = orig - eps;
            let lm = loss_of(&ad);
            ad.b[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            let diff = (num - ad.grad_b[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }

        // Check dx (gradient w.r.t. input).
        for i in 0..in_dim {
            let mut xp = x.clone(); xp[i] += eps;
            let mut xm = x.clone(); xm[i] -= eps;
            let mut op = vec![0.0f32; out_dim];
            let mut om = vec![0.0f32; out_dim];
            let _ = ad.forward(&xp, &mut op);
            let _ = ad.forward(&xm, &mut om);
            let lp: f32 = op.iter().sum();
            let lm: f32 = om.iter().sum();
            let num = (lp - lm) / (2.0 * eps);
            let diff = (num - grad_in[i]).abs();
            if diff > max_diff { max_diff = diff; }
        }

        eprintln!("[lora test] backward finite-diff max abs diff = {:.3e}", max_diff);
        assert!(max_diff < 1e-2, "lora backward finite-diff max diff {:.3e} >= 1e-2", max_diff);
    }

    #[test]
    fn lora_init_zero_delta() {
        // B = 0 at init => delta is exactly zero, base_out unchanged.
        let mut rng = Rng::new(7);
        let ad = LoraAdapter::new("zero", 8, 6, 4, 2.0, &mut rng);
        let x: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.5f32; 6];
        let before = out.clone();
        let _ = ad.forward(&x, &mut out);
        for (a, b) in out.iter().zip(before.iter()) {
            assert!((a - b).abs() < 1e-7, "delta not zero at init: {} vs {}", a, b);
        }
    }

    #[test]
    fn lora_grads_flat_roundtrip() {
        let mut rng = Rng::new(11);
        let mut ad = LoraAdapter::new("rt", 5, 3, 2, 1.0, &mut rng);
        for (i, g) in ad.grad_a.iter_mut().enumerate() { *g = i as f32; }
        for (i, g) in ad.grad_b.iter_mut().enumerate() { *g = 100.0 + i as f32; }
        let flat = ad.grads_flat();
        assert_eq!(flat.len(), ad.n_params());
        // Scatter a scaled copy back and verify.
        let scaled: Vec<f32> = flat.iter().map(|v| v * 0.5).collect();
        ad.grads_flat_mut(&scaled);
        assert_eq!(ad.grad_a[1], 0.5);
        assert_eq!(ad.grad_b[0], 50.0);
    }
}
