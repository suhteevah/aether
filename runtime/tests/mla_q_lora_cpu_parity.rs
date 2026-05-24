//! Synthetic CPU-reference parity test for the MLA Q-LoRA chain.
//!
//! Mirrors the Q-LoRA segment of `serving.rs::mla_attention_forward_absorbed`
//! (commit 7d3879a, GLM-4.7-flash gate-close):
//!
//!     q_a    = w_q_a       @ x_norm    // shape [q_lora_rank]
//!     q_a_n  = rms_norm(q_a, q_a_norm_g)
//!     q_proj = w_q_b       @ q_a_n     // shape [n_heads * key_mla]
//!     q_view = reshape(q_proj, [n_tokens, n_heads, head_dim])
//!
//! The GPU dispatch in serving.rs computes
//!     out[n_out] = W[n_out, n_in] @ x[n_in]
//! with W stored row-major `[n_out, n_in]`.  This test mirrors that
//! contract using the host-side `aether_op_matmul_f32`, which expects
//! row-major `A[m, k] @ B[k, n]`.  We feed `A = x[1, n_in]` and
//! `B = W^T[n_in, n_out]` so the output `[1, n_out]` matches what the
//! GPU path would emit numerically.
//!
//! `rms_norm` matches `aether_op_rms_norm_f32` — the same fn the absorbed
//! forward calls (`aether_op_rms_norm_f32_cuda` is the GPU equivalent).
//!
//! Two computations are produced:
//!   1. A hand-rolled triple-loop CPU reference written inside this file.
//!   2. The Aether primitive op chain (`aether_op_matmul_f32` +
//!      `aether_op_rms_norm_f32`).
//! The test asserts `max |a - b| < 1e-4` between them.
//!
//! Pure CPU.  No `--features cuda`.  No external file IO.  No randomness.
//! Self-contained, runs anywhere `cargo test -p aether_rt` runs.
//!
//! roadmap: P17.5

use std::ffi::c_void;
use std::os::raw::c_int;

use aether_rt::{aether_op_matmul_f32, aether_op_rms_norm_f32};

// Small but representative MLA Q-LoRA shape.
const N_HEADS_Q: usize = 4;
const HEAD_DIM: usize = 128;          // == key_mla in the absorbed code
const Q_LORA_RANK: usize = 64;
const D_MODEL: usize = 256;
const N_TOKENS: usize = 1;
const NORM_EPS: f32 = 1e-6;

// Deterministic seeded fill in [-1, 1].
fn seeded_fill(buf: &mut [f32], salt: u64) {
    for (i, v) in buf.iter_mut().enumerate() {
        // ((i * 13 + salt * 17) mod 23) - 11) / 11.0 keeps values in a clean
        // band and is integer-derived so it round-trips identically on every
        // platform.
        let z = ((i as u64).wrapping_mul(13).wrapping_add(salt.wrapping_mul(17))) % 23;
        *v = (z as f32 - 11.0) / 11.0;
    }
}

/// CPU reference matmul mirroring the GPU dispatch contract:
///     out[r, o] = sum_i x[r, i] * w[o, i]
/// Shapes: x [rows, n_in],  w [n_out, n_in],  out [rows, n_out].
fn ref_matmul_w_rowmajor(x: &[f32], w: &[f32], out: &mut [f32],
                         rows: usize, n_in: usize, n_out: usize) {
    assert_eq!(x.len(), rows * n_in);
    assert_eq!(w.len(), n_out * n_in);
    assert_eq!(out.len(), rows * n_out);
    for r in 0..rows {
        for o in 0..n_out {
            let mut acc = 0.0f64;
            for i in 0..n_in {
                acc += (x[r * n_in + i] as f64) * (w[o * n_in + i] as f64);
            }
            out[r * n_out + o] = acc as f32;
        }
    }
}

/// CPU reference RMSNorm matching `ops::rms_norm_f32`:
///     y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps)
fn ref_rms_norm(x: &[f32], gamma: &[f32], eps: f32,
                rows: usize, d: usize) -> Vec<f32> {
    assert_eq!(x.len(), rows * d);
    assert_eq!(gamma.len(), d);
    let mut out = vec![0f32; rows * d];
    for r in 0..rows {
        let off = r * d;
        let mut sumsq = 0.0f64;
        for i in 0..d {
            let v = x[off + i] as f64;
            sumsq += v * v;
        }
        let inv = 1.0 / ((sumsq / d as f64) + eps as f64).sqrt();
        for i in 0..d {
            out[off + i] = ((x[off + i] as f64) * inv * (gamma[i] as f64)) as f32;
        }
    }
    out
}

/// Transpose a row-major `[rows, cols]` buffer to row-major `[cols, rows]`.
fn transpose(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(src.len(), rows * cols);
    let mut dst = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn mla_q_lora_chain_cpu_parity() {
    // ---- Synthetic inputs ----
    // x_norm shape: [n_tokens, d_model] -- row-major.
    let mut x_norm = vec![0f32; N_TOKENS * D_MODEL];
    seeded_fill(&mut x_norm, 1);

    // w_q_a stored as GPU-row-major [q_lora_rank, d_model] (== [n_out, n_in]).
    let mut w_q_a = vec![0f32; Q_LORA_RANK * D_MODEL];
    seeded_fill(&mut w_q_a, 2);

    // q_a RMS-norm gain, one weight per q_lora_rank dim.
    let mut q_a_norm_g = vec![0f32; Q_LORA_RANK];
    seeded_fill(&mut q_a_norm_g, 3);
    // Keep gamma strictly positive-ish so normed q_a stays well-conditioned.
    for v in q_a_norm_g.iter_mut() { *v = 0.5 + (*v + 1.0) * 0.25; }

    // w_q_b stored row-major [n_heads * head_dim, q_lora_rank].
    let mut w_q_b = vec![0f32; N_HEADS_Q * HEAD_DIM * Q_LORA_RANK];
    seeded_fill(&mut w_q_b, 4);

    // ---- Reference computation (hand rolled) ----
    let mut ref_q_a = vec![0f32; N_TOKENS * Q_LORA_RANK];
    ref_matmul_w_rowmajor(&x_norm, &w_q_a, &mut ref_q_a,
                          N_TOKENS, D_MODEL, Q_LORA_RANK);
    let ref_q_a_n = ref_rms_norm(&ref_q_a, &q_a_norm_g, NORM_EPS,
                                 N_TOKENS, Q_LORA_RANK);
    let mut ref_q_proj = vec![0f32; N_TOKENS * N_HEADS_Q * HEAD_DIM];
    ref_matmul_w_rowmajor(&ref_q_a_n, &w_q_b, &mut ref_q_proj,
                          N_TOKENS, Q_LORA_RANK, N_HEADS_Q * HEAD_DIM);

    // ---- Aether primitive op chain ----
    // aether_op_matmul_f32 contract: out[m, n] = a[m, k] @ b[k, n].
    // To emulate GPU `out = W[n_out, n_in] @ x[n_in]` for x rows, we feed
    //   a = x [rows, n_in],  b = W^T [n_in, n_out]  →  out [rows, n_out].
    let w_q_a_t = transpose(&w_q_a, Q_LORA_RANK, D_MODEL);
    let mut got_q_a = vec![0f32; N_TOKENS * Q_LORA_RANK];
    unsafe {
        aether_op_matmul_f32(
            x_norm.as_ptr() as *const c_void,
            w_q_a_t.as_ptr() as *const c_void,
            got_q_a.as_mut_ptr() as *mut c_void,
            N_TOKENS as c_int,
            D_MODEL as c_int,
            Q_LORA_RANK as c_int,
        );
    }

    // RMSNorm (matches ops::rms_norm_f32 exactly).
    let mut got_q_a_n = vec![0f32; N_TOKENS * Q_LORA_RANK];
    unsafe {
        aether_op_rms_norm_f32(
            got_q_a.as_ptr() as *const c_void,
            q_a_norm_g.as_ptr() as *const c_void,
            NORM_EPS,
            got_q_a_n.as_mut_ptr() as *mut c_void,
            N_TOKENS as c_int,
            Q_LORA_RANK as c_int,
        );
    }

    let w_q_b_t = transpose(&w_q_b, N_HEADS_Q * HEAD_DIM, Q_LORA_RANK);
    let mut got_q_proj = vec![0f32; N_TOKENS * N_HEADS_Q * HEAD_DIM];
    unsafe {
        aether_op_matmul_f32(
            got_q_a_n.as_ptr() as *const c_void,
            w_q_b_t.as_ptr() as *const c_void,
            got_q_proj.as_mut_ptr() as *mut c_void,
            N_TOKENS as c_int,
            Q_LORA_RANK as c_int,
            (N_HEADS_Q * HEAD_DIM) as c_int,
        );
    }

    // ---- Reshape sanity: q_proj viewed as [n_tokens, n_heads, head_dim].
    // Logically a no-op for row-major contiguous storage; we still witness
    // the per-head slices match the flat view so a future reshape kernel
    // can be diffed against this test.
    for t in 0..N_TOKENS {
        for h in 0..N_HEADS_Q {
            let base = (t * N_HEADS_Q + h) * HEAD_DIM;
            let ref_slice = &ref_q_proj[base .. base + HEAD_DIM];
            let got_slice = &got_q_proj[base .. base + HEAD_DIM];
            let diff = max_abs_diff(ref_slice, got_slice);
            assert!(
                diff < 1e-4,
                "per-head diff at (t={}, h={}) = {:.3e} exceeds 1e-4",
                t, h, diff,
            );
        }
    }

    // ---- Stage-by-stage parity ----
    let d_q_a    = max_abs_diff(&ref_q_a,    &got_q_a);
    let d_q_a_n  = max_abs_diff(&ref_q_a_n,  &got_q_a_n);
    let d_q_proj = max_abs_diff(&ref_q_proj, &got_q_proj);

    eprintln!(
        "[mla_q_lora_cpu_parity] shapes: n_tokens={} d_model={} q_lora_rank={} \
         n_heads={} head_dim={}",
        N_TOKENS, D_MODEL, Q_LORA_RANK, N_HEADS_Q, HEAD_DIM,
    );
    eprintln!(
        "[mla_q_lora_cpu_parity] max-abs-diff: q_a={:.3e} q_a_n={:.3e} q_proj={:.3e}",
        d_q_a, d_q_a_n, d_q_proj,
    );

    // Loose enough to absorb f32 summation-order drift in the 256-/64-wide
    // accumulators; tight enough to catch a real algorithmic divergence.
    assert!(d_q_a    < 1e-4, "q_a diff = {:.3e}",    d_q_a);
    assert!(d_q_a_n  < 1e-4, "q_a_n diff = {:.3e}",  d_q_a_n);
    assert!(d_q_proj < 1e-4, "q_proj diff = {:.3e}", d_q_proj);

    // Non-degeneracy: at least some output must be non-zero (catches the
    // "everything is identical because everything is zero" failure mode).
    let abs_max_out = got_q_proj.iter().cloned()
        .map(f32::abs).fold(0.0f32, f32::max);
    assert!(abs_max_out > 0.01,
        "q_proj appears degenerate (max |x| = {:.3e})", abs_max_out);
}
