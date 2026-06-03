//! CPU implementations of every primitive op declared in `runtime/ABI.md`.
//!
//! Pure Rust + std. No framework dependency. Single-threaded; Phase 1
//! replaces these bodies with cuBLAS / cuDNN calls, and the FFI surface
//! stays identical because aetherc-emitted IR doesn't care whether the
//! body runs on CPU or GPU.
//!
//! Notation: f32 throughout. All shape arguments are positional — see
//! `runtime/ABI.md` for the contract. Caller owns every output buffer;
//! the runtime writes into them.

use std::slice;

unsafe fn s<'a>(p: *const f32, n: usize) -> &'a [f32] { slice::from_raw_parts(p, n) }
unsafe fn sm<'a>(p: *mut f32, n: usize) -> &'a mut [f32] { slice::from_raw_parts_mut(p, n) }

// ---------------------------------------------------------------- matmul

/// Blocked matmul (cache-tile-friendly). Same shape contract as `matmul_f32`
/// but uses 32x32 register-resident tiles to improve L1 reuse on bigger
/// shapes. Identical numerical output to the naive version (modulo
/// fp-summation order differences, which are below the audit's float
/// tolerance for the witness shape).
pub unsafe fn matmul_blocked_f32(
    a: *const f32, b: *const f32, out: *mut f32,
    m: usize, k: usize, n: usize,
) {
    const BS: usize = 32;
    let a = s(a, m * k);
    let b = s(b, k * n);
    let out = sm(out, m * n);
    for v in out.iter_mut() { *v = 0.0; }
    let mut i0 = 0;
    while i0 < m {
        let i_end = (i0 + BS).min(m);
        let mut j0 = 0;
        while j0 < n {
            let j_end = (j0 + BS).min(n);
            let mut k0 = 0;
            while k0 < k {
                let k_end = (k0 + BS).min(k);
                for i in i0..i_end {
                    for kk in k0..k_end {
                        let av = a[i * k + kk];
                        for j in j0..j_end {
                            out[i * n + j] += av * b[kk * n + j];
                        }
                    }
                }
                k0 += BS;
            }
            j0 += BS;
        }
        i0 += BS;
    }
}

// ---------------------------------------------------------------- matmul

/// `out[i, j] = sum_k a[i, k] * b[k, j]` — row-major.
pub unsafe fn matmul_f32(
    a: *const f32, b: *const f32, out: *mut f32,
    m: usize, k: usize, n: usize,
) {
    let a = s(a, m * k);
    let b = s(b, k * n);
    let out = sm(out, m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

/// `dA = dY @ B^T`  (shape `[M, K]`)
pub unsafe fn matmul_backward_lhs_f32(
    dy: *const f32, b: *const f32, da: *mut f32,
    m: usize, k: usize, n: usize,
) {
    let dy = s(dy, m * n);
    let b = s(b, k * n);
    let da = sm(da, m * k);
    for da_v in da.iter_mut() { *da_v = 0.0; }
    for i in 0..m {
        for kk in 0..k {
            let mut acc = 0.0f32;
            for j in 0..n {
                acc += dy[i * n + j] * b[kk * n + j];
            }
            da[i * k + kk] = acc;
        }
    }
}

/// `dB = A^T @ dY`  (shape `[K, N]`)
pub unsafe fn matmul_backward_rhs_f32(
    a: *const f32, dy: *const f32, db: *mut f32,
    m: usize, k: usize, n: usize,
) {
    let a = s(a, m * k);
    let dy = s(dy, m * n);
    let db = sm(db, k * n);
    for db_v in db.iter_mut() { *db_v = 0.0; }
    for kk in 0..k {
        for j in 0..n {
            let mut acc = 0.0f32;
            for i in 0..m {
                acc += a[i * k + kk] * dy[i * n + j];
            }
            db[kk * n + j] = acc;
        }
    }
}

// ---------------------------------------------------------------- elementwise

/// In-place: x += b broadcast over the trailing `cols` dim.
/// Shapes: x: [rows, cols], b: [cols].
pub unsafe fn add_bias_f32(x: *mut f32, b: *const f32, rows: usize, cols: usize) {
    let x = sm(x, rows * cols);
    let b = s(b, cols);
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] += b[c];
        }
    }
}

pub unsafe fn add_f32(a: *const f32, b: *const f32, out: *mut f32, n: usize) {
    let a = s(a, n); let b = s(b, n); let out = sm(out, n);
    for i in 0..n { out[i] = a[i] + b[i]; }
}

pub unsafe fn add_inplace_f32(x: *mut f32, y: *const f32, n: usize) {
    let x = sm(x, n); let y = s(y, n);
    for i in 0..n { x[i] += y[i]; }
}

pub unsafe fn scale_f32(x: *mut f32, s_v: f32, n: usize) {
    let x = sm(x, n);
    for v in x.iter_mut() { *v *= s_v; }
}

pub unsafe fn axpy_f32(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
    let x = s(x, n); let y = sm(y, n);
    for i in 0..n { y[i] += alpha * x[i]; }
}

// ---------------------------------------------------------------- nonlinearities

/// GELU (tanh approximation). In-place.
pub unsafe fn gelu_f32(x: *mut f32, n: usize) {
    let x = sm(x, n);
    let c0 = (2.0_f32 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        let t = c0 * (*v + 0.044715 * *v * *v * *v);
        *v = 0.5 * *v * (1.0 + t.tanh());
    }
}

/// erf(x) via Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7), in f64.
#[inline]
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Exact (erf) GELU, in place. `gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))`.
/// This is the variant PyTorch / HF `hidden_act="gelu"` use — distinct from the
/// tanh approximation in `gelu_f32`. DINOv3 / ViT need this for bit-faithful
/// reproduction (cosine >= 0.999 vs the reference). Computed in f64 internally
/// for accuracy, narrowed to f32 on store.
pub unsafe fn gelu_erf_f32(x: *mut f32, n: usize) {
    let x = sm(x, n);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    for v in x.iter_mut() {
        let xf = *v as f64;
        *v = (0.5 * xf * (1.0 + erf_f64(xf * inv_sqrt2))) as f32;
    }
}

/// d/dx GELU (tanh approx). dx[i] = dy[i] * gelu'(x[i]).
pub unsafe fn gelu_backward_f32(x: *const f32, dy: *const f32, dx: *mut f32, n: usize) {
    let x = s(x, n); let dy = s(dy, n); let dx = sm(dx, n);
    let c0 = (2.0_f32 / std::f32::consts::PI).sqrt();
    for i in 0..n {
        let xi = x[i];
        let inner = c0 * (xi + 0.044715 * xi * xi * xi);
        let tanh = inner.tanh();
        let sech2 = 1.0 - tanh * tanh;
        let dinner = c0 * (1.0 + 3.0 * 0.044715 * xi * xi);
        let g_prime = 0.5 * (1.0 + tanh) + 0.5 * xi * sech2 * dinner;
        dx[i] = dy[i] * g_prime;
    }
}

/// SiLU (a.k.a. swish): silu(x) = x * sigmoid(x). In-place.
pub unsafe fn silu_f32(x: *mut f32, n: usize) {
    let x = sm(x, n);
    for v in x.iter_mut() {
        let s = 1.0 / (1.0 + (-*v).exp());
        *v *= s;
    }
}

/// d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
///              = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
pub unsafe fn silu_backward_f32(x: *const f32, dy: *const f32, dx: *mut f32, n: usize) {
    let x = s(x, n); let dy = s(dy, n); let dx = sm(dx, n);
    for i in 0..n {
        let sig = 1.0 / (1.0 + (-x[i]).exp());
        dx[i] = dy[i] * sig * (1.0 + x[i] * (1.0 - sig));
    }
}

pub unsafe fn relu_f32(x: *mut f32, n: usize) {
    let x = sm(x, n);
    for v in x.iter_mut() { if *v < 0.0 { *v = 0.0; } }
}

pub unsafe fn relu_backward_f32(x: *const f32, dy: *const f32, dx: *mut f32, n: usize) {
    let x = s(x, n); let dy = s(dy, n); let dx = sm(dx, n);
    for i in 0..n { dx[i] = if x[i] > 0.0 { dy[i] } else { 0.0 }; }
}

// ---------------------------------------------------------------- softmax

/// Softmax along the last axis. Shape: [rows, cols]. In-place.
pub unsafe fn softmax_last_f32(x: *mut f32, rows: usize, cols: usize) {
    let x = sm(x, rows * cols);
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let mut mx = row[0];
        for &v in row.iter().skip(1) { if v > mx { mx = v; } }
        let mut sum = 0.0f32;
        for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}

/// Backward of softmax along the last axis.
/// dx[r, i] = sum_j (delta_ij - y[r, j]) * y[r, i] * dy[r, j]
///         = y[r, i] * (dy[r, i] - sum_j y[r, j] * dy[r, j])
pub unsafe fn softmax_backward_last_f32(
    y: *const f32, dy: *const f32, dx: *mut f32, rows: usize, cols: usize,
) {
    let y = s(y, rows * cols); let dy = s(dy, rows * cols); let dx = sm(dx, rows * cols);
    for r in 0..rows {
        let off = r * cols;
        let mut dot = 0.0f32;
        for j in 0..cols { dot += y[off + j] * dy[off + j]; }
        for i in 0..cols { dx[off + i] = y[off + i] * (dy[off + i] - dot); }
    }
}

// ---------------------------------------------------------------- layer norm

/// y = (x - mean) / sqrt(var + eps) * gamma + beta
/// Shape: x: [rows, d], gamma/beta: [d]. Saves mean & inv_std for backward.
pub unsafe fn layer_norm_f32(
    x: *const f32, gamma: *const f32, beta: *const f32, eps: f32,
    out: *mut f32, mean_out: *mut f32, inv_std_out: *mut f32,
    rows: usize, d: usize,
) {
    let x = s(x, rows * d);
    let gamma = s(gamma, d);
    let beta = s(beta, d);
    let out = sm(out, rows * d);
    let mean_out = sm(mean_out, rows);
    let inv_std_out = sm(inv_std_out, rows);
    for r in 0..rows {
        let off = r * d;
        let mut mean = 0.0f32;
        for i in 0..d { mean += x[off + i]; }
        mean /= d as f32;
        let mut var = 0.0f32;
        for i in 0..d {
            let dv = x[off + i] - mean;
            var += dv * dv;
        }
        var /= d as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        mean_out[r] = mean;
        inv_std_out[r] = inv_std;
        for i in 0..d {
            out[off + i] = (x[off + i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }
}

/// LayerNorm backward.  Inputs: x, gamma, dy, mean, inv_std.
/// Outputs: dx, dgamma (accumulated), dbeta (accumulated).
pub unsafe fn layer_norm_backward_f32(
    x: *const f32, gamma: *const f32, dy: *const f32,
    mean: *const f32, inv_std: *const f32,
    dx: *mut f32, dgamma: *mut f32, dbeta: *mut f32,
    rows: usize, d: usize,
) {
    let x = s(x, rows * d);
    let gamma = s(gamma, d);
    let dy = s(dy, rows * d);
    let mean = s(mean, rows);
    let inv_std = s(inv_std, rows);
    let dx = sm(dx, rows * d);
    let dgamma = sm(dgamma, d);
    let dbeta = sm(dbeta, d);
    let dn = d as f32;
    for r in 0..rows {
        let off = r * d;
        let m = mean[r];
        let inv = inv_std[r];

        let mut sum_dy = 0.0f32;
        let mut sum_dy_xhat = 0.0f32;
        for i in 0..d {
            let xhat = (x[off + i] - m) * inv;
            let g = gamma[i] * dy[off + i];
            sum_dy += g;
            sum_dy_xhat += g * xhat;
        }
        for i in 0..d {
            let xhat = (x[off + i] - m) * inv;
            let g = gamma[i] * dy[off + i];
            dx[off + i] = inv * (g - sum_dy / dn - xhat * sum_dy_xhat / dn);
            dgamma[i] += dy[off + i] * xhat;
            dbeta[i] += dy[off + i];
        }
    }
}

/// RMSNorm forward. `y = x * gamma / sqrt(mean(x^2) + eps)`. No beta
/// (Qwen / Llama don't use one). `rows × d` activations, `gamma` is
/// `[d]`. In-place by passing `x == out`.
pub unsafe fn rms_norm_f32(
    x: *const f32, gamma: *const f32, eps: f32,
    out: *mut f32, rows: usize, d: usize,
) {
    let x = s(x, rows * d);
    let gamma = s(gamma, d);
    let out = sm(out, rows * d);
    for r in 0..rows {
        let off = r * d;
        let mut sumsq = 0.0f32;
        for i in 0..d { sumsq += x[off + i] * x[off + i]; }
        let inv_rms = 1.0 / (sumsq / d as f32 + eps).sqrt();
        for i in 0..d { out[off + i] = x[off + i] * inv_rms * gamma[i]; }
    }
}

/// RoPE (rotary positional embeddings) applied in place on a contiguous
/// `[seq, n_heads, head_dim]` buffer at the given starting position. For
/// each pair of dims `(2i, 2i+1)` within a head, multiplies by the
/// rotation matrix at angle `theta = pos * base^(-2i/head_dim)`.
///
/// `head_dim` must be even. `base` is typically `10000.0` (Llama) or
/// `1000000.0` (Qwen2.5). `pos_start` is the position of the first
/// token in this batch (for inference batches that aren't prefilling
/// from scratch). For a one-shot forward pass starting at the BOS pad,
/// `pos_start = 0`.
pub unsafe fn rope_apply_f32(
    x: *mut f32, seq: usize, n_heads: usize, head_dim: usize,
    base: f32, pos_start: usize,
) {
    let total = seq * n_heads * head_dim;
    let x = sm(x, total);
    assert!(head_dim % 2 == 0, "RoPE needs even head_dim");
    let hd_half = head_dim / 2;
    for t in 0..seq {
        let pos = (pos_start + t) as f32;
        for h in 0..n_heads {
            let base_off = (t * n_heads + h) * head_dim;
            for i in 0..hd_half {
                // theta_i = pos * base^(-2i/head_dim)
                let exp = -2.0 * i as f32 / head_dim as f32;
                let theta = pos * base.powf(exp);
                let (sin, cos) = theta.sin_cos();
                let i0 = base_off + i;
                let i1 = base_off + i + hd_half;
                // Llama-style "half-half" interleave: pair (i, i+hd/2)
                // is the rotation pair, NOT (2i, 2i+1).
                let x0 = x[i0];
                let x1 = x[i1];
                x[i0] = x0 * cos - x1 * sin;
                x[i1] = x0 * sin + x1 * cos;
            }
        }
    }
}

/// Grouped-query attention helper: broadcast a key/value tensor of
/// shape `[seq, n_kv_heads, head_dim]` to `[seq, n_q_heads, head_dim]`
/// by repeating each KV head `n_q_heads / n_kv_heads` times.
///
/// Used between K/V projection (which produces `n_kv_heads`-wide output
/// in GQA) and the SDPA kernel (which expects K and V at full
/// `n_q_heads` width).
pub unsafe fn gqa_repeat_kv_f32(
    kv_in: *const f32, kv_out: *mut f32,
    seq: usize, n_kv_heads: usize, head_dim: usize, n_q_heads: usize,
) {
    assert!(n_q_heads % n_kv_heads == 0, "n_q_heads must be a multiple of n_kv_heads");
    let g = n_q_heads / n_kv_heads;  // group size
    let kv_in = s(kv_in, seq * n_kv_heads * head_dim);
    let kv_out = sm(kv_out, seq * n_q_heads * head_dim);
    for t in 0..seq {
        for kh in 0..n_kv_heads {
            let src_off = (t * n_kv_heads + kh) * head_dim;
            for repeat in 0..g {
                let dst_h = kh * g + repeat;
                let dst_off = (t * n_q_heads + dst_h) * head_dim;
                kv_out[dst_off..dst_off + head_dim]
                    .copy_from_slice(&kv_in[src_off..src_off + head_dim]);
            }
        }
    }
}

/// FR-17.17-extra / matt-voice — apply a LoRA update in place to a
/// weight matrix in Aether matmul layout.
///
/// Conventions:
/// - `w` is the base weight stored as `[d_in, d_out]` (Aether matmul
///   layout, where `out = X @ W` reads W as `[d_in_k, d_out_n]`).
/// - `lora_a` is PEFT-style `[rank, d_in]` (rank rows, d_in cols).
/// - `lora_b` is PEFT-style `[d_out, rank]`.
/// - `scale` is `alpha / rank`.
///
/// In PEFT math the effective forward weight is `W_eff = W_math +
/// scale * B @ A`, where `W_math` is `[d_out, d_in]`. Our matmul
/// layout stores `W = W_math^T`, so we add `scale * (B @ A)^T =
/// scale * A^T @ B^T = scale * sum_r A[r, i_in] * B[i_out, r]`
/// into `w[i_in, i_out]` for each (i_in, i_out).
///
/// Total work: O(d_in * d_out * rank). Typical matt-voice LoRA
/// rank is 8-32; on Qwen2.5-7B's d=3584 that's ~100M-400M ops --
/// fast enough to apply at load time, not in the hot loop.
pub unsafe fn apply_lora_f32(
    w: *mut f32, lora_a: *const f32, lora_b: *const f32,
    scale: f32, d_in: usize, d_out: usize, rank: usize,
) {
    let w = sm(w, d_in * d_out);
    let a = s(lora_a, rank * d_in);
    let b = s(lora_b, d_out * rank);
    for i_in in 0..d_in {
        for i_out in 0..d_out {
            let mut delta = 0.0f32;
            for r in 0..rank {
                delta += a[r * d_in + i_in] * b[i_out * rank + r];
            }
            w[i_in * d_out + i_out] += scale * delta;
        }
    }
}

// ---------------------------------------------------------------- attention

/// Causal scaled-dot-product attention.
/// Shapes: q/k/v: [B*H, S, D] flattened, out: same. Saves attn weights.
pub unsafe fn sdpa_causal_f32(
    q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, attn_out: *mut f32,
    bh: usize, s_len: usize, d: usize,
) {
    let q = s(q, bh * s_len * d);
    let k = s(k, bh * s_len * d);
    let v = s(v, bh * s_len * d);
    let out = sm(out, bh * s_len * d);
    let attn_out = sm(attn_out, bh * s_len * s_len);
    let scale = 1.0 / (d as f32).sqrt();
    let neg_inf = f32::NEG_INFINITY;

    for h in 0..bh {
        let qh = &q[h * s_len * d..(h + 1) * s_len * d];
        let kh = &k[h * s_len * d..(h + 1) * s_len * d];
        let vh = &v[h * s_len * d..(h + 1) * s_len * d];
        let oh = &mut out[h * s_len * d..(h + 1) * s_len * d];
        let ah = &mut attn_out[h * s_len * s_len..(h + 1) * s_len * s_len];

        // scores[i, j] = (q[i] dot k[j]) * scale
        for i in 0..s_len {
            for j in 0..s_len {
                if j > i {
                    ah[i * s_len + j] = neg_inf;
                } else {
                    let mut acc = 0.0f32;
                    for dd in 0..d { acc += qh[i * d + dd] * kh[j * d + dd]; }
                    ah[i * s_len + j] = acc * scale;
                }
            }
        }
        // softmax along j
        for i in 0..s_len {
            let row = &mut ah[i * s_len..(i + 1) * s_len];
            let mut mx = row[0];
            for &v in row.iter().skip(1) { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }
        // out[i] = sum_j attn[i, j] * v[j]
        for i in 0..s_len {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..s_len {
                    acc += ah[i * s_len + j] * vh[j * d + dd];
                }
                oh[i * d + dd] = acc;
            }
        }
    }
}

/// Non-causal (bidirectional) scaled dot-product attention. Same layout +
/// math as `sdpa_causal_f32` but every query attends to every key (no mask).
/// Used by encoder / ViT models (BERT, DINOv3). q/k/v are `[bh, s_len, d]`
/// head-major; `out` is `[bh, s_len, d]`; `attn_out` is `[bh, s_len, s_len]`
/// (post-softmax weights, kept for debugging/backward). scale = 1/sqrt(d).
pub unsafe fn sdpa_full_f32(
    q: *const f32, k: *const f32, v: *const f32,
    out: *mut f32, attn_out: *mut f32,
    bh: usize, s_len: usize, d: usize,
) {
    let q = s(q, bh * s_len * d);
    let k = s(k, bh * s_len * d);
    let v = s(v, bh * s_len * d);
    let out = sm(out, bh * s_len * d);
    let attn_out = sm(attn_out, bh * s_len * s_len);
    let scale = 1.0 / (d as f32).sqrt();

    for h in 0..bh {
        let qh = &q[h * s_len * d..(h + 1) * s_len * d];
        let kh = &k[h * s_len * d..(h + 1) * s_len * d];
        let vh = &v[h * s_len * d..(h + 1) * s_len * d];
        let oh = &mut out[h * s_len * d..(h + 1) * s_len * d];
        let ah = &mut attn_out[h * s_len * s_len..(h + 1) * s_len * s_len];

        for i in 0..s_len {
            for j in 0..s_len {
                let mut acc = 0.0f32;
                for dd in 0..d { acc += qh[i * d + dd] * kh[j * d + dd]; }
                ah[i * s_len + j] = acc * scale;
            }
        }
        for i in 0..s_len {
            let row = &mut ah[i * s_len..(i + 1) * s_len];
            let mut mx = row[0];
            for &v in row.iter().skip(1) { if v > mx { mx = v; } }
            let mut sum = 0.0f32;
            for v in row.iter_mut() { *v = (*v - mx).exp(); sum += *v; }
            let inv = 1.0 / sum;
            for v in row.iter_mut() { *v *= inv; }
        }
        for i in 0..s_len {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..s_len {
                    acc += ah[i * s_len + j] * vh[j * d + dd];
                }
                oh[i * d + dd] = acc;
            }
        }
    }
}

/// Backward of `sdpa_causal_f32`. Inputs: q, k, v, attn (saved fwd), dout.
/// Outputs: dq, dk, dv.
pub unsafe fn sdpa_causal_backward_f32(
    q: *const f32, k: *const f32, v: *const f32, attn: *const f32, dout: *const f32,
    dq: *mut f32, dk: *mut f32, dv: *mut f32,
    bh: usize, s_len: usize, d: usize,
) {
    let q = s(q, bh * s_len * d);
    let k = s(k, bh * s_len * d);
    let v = s(v, bh * s_len * d);
    let attn = s(attn, bh * s_len * s_len);
    let dout = s(dout, bh * s_len * d);
    let dq = sm(dq, bh * s_len * d);
    let dk = sm(dk, bh * s_len * d);
    let dv = sm(dv, bh * s_len * d);
    for v in dq.iter_mut() { *v = 0.0; }
    for v in dk.iter_mut() { *v = 0.0; }
    for v in dv.iter_mut() { *v = 0.0; }
    let scale = 1.0 / (d as f32).sqrt();

    for h in 0..bh {
        let qh = &q[h * s_len * d..(h + 1) * s_len * d];
        let kh = &k[h * s_len * d..(h + 1) * s_len * d];
        let vh = &v[h * s_len * d..(h + 1) * s_len * d];
        let ah = &attn[h * s_len * s_len..(h + 1) * s_len * s_len];
        let doh = &dout[h * s_len * d..(h + 1) * s_len * d];
        let dqh = &mut dq[h * s_len * d..(h + 1) * s_len * d];
        let dkh = &mut dk[h * s_len * d..(h + 1) * s_len * d];
        let dvh = &mut dv[h * s_len * d..(h + 1) * s_len * d];

        // d_attn[i, j] = sum_dd dout[i, dd] * v[j, dd]
        // dv[j, dd]   += sum_i attn[i, j] * dout[i, dd]
        let mut d_attn = vec![0.0f32; s_len * s_len];
        for i in 0..s_len {
            for j in 0..=i {
                let mut acc = 0.0f32;
                for dd in 0..d { acc += doh[i * d + dd] * vh[j * d + dd]; }
                d_attn[i * s_len + j] = acc;
            }
        }
        for j in 0..s_len {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for i in j..s_len { acc += ah[i * s_len + j] * doh[i * d + dd]; }
                dvh[j * d + dd] += acc;
            }
        }

        // softmax backward → d_scores
        let mut d_scores = vec![0.0f32; s_len * s_len];
        for i in 0..s_len {
            let mut dot = 0.0f32;
            for j in 0..=i { dot += ah[i * s_len + j] * d_attn[i * s_len + j]; }
            for j in 0..=i {
                d_scores[i * s_len + j] = ah[i * s_len + j] * (d_attn[i * s_len + j] - dot);
            }
        }

        // scores = (q @ k^T) * scale
        // dq[i, dd] += scale * sum_j d_scores[i, j] * k[j, dd]
        // dk[j, dd] += scale * sum_i d_scores[i, j] * q[i, dd]
        for i in 0..s_len {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for j in 0..=i { acc += d_scores[i * s_len + j] * kh[j * d + dd]; }
                dqh[i * d + dd] += scale * acc;
            }
        }
        for j in 0..s_len {
            for dd in 0..d {
                let mut acc = 0.0f32;
                for i in j..s_len { acc += d_scores[i * s_len + j] * qh[i * d + dd]; }
                dkh[j * d + dd] += scale * acc;
            }
        }
    }
}

// ---------------------------------------------------------------- cross entropy

/// Returns mean loss across batch. Saves softmax probs for backward.
/// logits: [B, V], labels: [B] (i32), probs_out: [B, V].
pub unsafe fn cross_entropy_f32(
    logits: *const f32, labels: *const i32,
    probs_out: *mut f32, b: usize, v: usize,
) -> f32 {
    let logits = s(logits, b * v);
    let labels = slice::from_raw_parts(labels, b);
    let probs = sm(probs_out, b * v);
    let mut total = 0.0f64;
    for i in 0..b {
        let off = i * v;
        let mut mx = logits[off];
        for &x in logits.iter().skip(off + 1).take(v - 1) { if x > mx { mx = x; } }
        let mut sum = 0.0f32;
        for j in 0..v { probs[off + j] = (logits[off + j] - mx).exp(); sum += probs[off + j]; }
        let inv = 1.0 / sum;
        for j in 0..v { probs[off + j] *= inv; }
        let lab = labels[i] as usize;
        let p = probs[off + lab].max(1e-12);
        total += -(p.ln() as f64);
    }
    (total / b as f64) as f32
}

/// Backward: dlogits = (probs - one_hot(labels)) / B. probs already saved.
pub unsafe fn cross_entropy_backward_f32(
    probs: *const f32, labels: *const i32,
    dlogits: *mut f32, b: usize, v: usize,
) {
    let probs = s(probs, b * v);
    let labels = slice::from_raw_parts(labels, b);
    let dlogits = sm(dlogits, b * v);
    let inv_b = 1.0 / (b as f32);
    for i in 0..b {
        let off = i * v;
        for j in 0..v { dlogits[off + j] = probs[off + j] * inv_b; }
        let lab = labels[i] as usize;
        dlogits[off + lab] -= inv_b;
    }
}

// ---------------------------------------------------------------- losses (P7.6)

/// Mean-squared error: scalar = mean_i (pred[i] - target[i])^2.
pub unsafe fn mse_f32(pred: *const f32, target: *const f32, n: usize) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let d = pred[i] - target[i];
        acc += (d * d) as f64;
    }
    (acc / n as f64) as f32
}

/// dpred[i] = 2 * (pred[i] - target[i]) / n. Matches mean reduction.
pub unsafe fn mse_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let scale = 2.0 / n as f32;
    for i in 0..n {
        dpred[i] = (pred[i] - target[i]) * scale;
    }
}

/// Mean absolute error: scalar = mean_i |pred[i] - target[i]|.
pub unsafe fn mae_f32(pred: *const f32, target: *const f32, n: usize) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n { acc += (pred[i] - target[i]).abs() as f64; }
    (acc / n as f64) as f32
}

/// dpred[i] = sign(pred[i] - target[i]) / n. Subgradient at zero is 0.
pub unsafe fn mae_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let d = pred[i] - target[i];
        dpred[i] = if d > 0.0 { inv_n } else if d < 0.0 { -inv_n } else { 0.0 };
    }
}

/// Binary cross-entropy with logits (numerically stable). Per-elem:
///   loss[i] = max(z, 0) - z*t + log(1 + exp(-|z|))
/// where z = pred[i] (logit), t = target[i] (in [0,1]). Reduce by mean.
pub unsafe fn bce_with_logits_f32(
    pred: *const f32, target: *const f32, n: usize,
) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let z = pred[i];
        let t = target[i];
        let max_zero = if z > 0.0 { z } else { 0.0 };
        let l = max_zero - z * t + (1.0 + (-z.abs()).exp()).ln();
        acc += l as f64;
    }
    (acc / n as f64) as f32
}

/// dpred[i] = (sigmoid(pred[i]) - target[i]) / n.
pub unsafe fn bce_with_logits_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let z = pred[i];
        let sigmoid = 1.0 / (1.0 + (-z).exp());
        dpred[i] = (sigmoid - target[i]) * inv_n;
    }
}

/// Binary cross-entropy on probabilities (clamps for numerical safety):
///   loss[i] = -(t*log(p) + (1-t)*log(1-p)). Reduce by mean.
pub unsafe fn bce_f32(pred: *const f32, target: *const f32, n: usize) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let p = pred[i].clamp(1e-7, 1.0 - 1e-7);
        let t = target[i];
        acc += -((t * p.ln() + (1.0 - t) * (1.0 - p).ln()) as f64);
    }
    (acc / n as f64) as f32
}

/// dpred[i] = (p - t) / (p*(1-p)) / n. Clamps mirror the forward.
pub unsafe fn bce_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let p = pred[i].clamp(1e-7, 1.0 - 1e-7);
        dpred[i] = (p - target[i]) / (p * (1.0 - p)) * inv_n;
    }
}

/// KL divergence KL(target || pred) on probability distributions:
///   loss = sum_i target[i] * (log(target[i]) - log(pred[i])) / n.
/// `pred` and `target` must be valid probability vectors (>0 entries).
pub unsafe fn kl_div_f32(pred: *const f32, target: *const f32, n: usize) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let p = pred[i].max(1e-12);
        let t = target[i].max(1e-12);
        acc += (target[i] * (t.ln() - p.ln())) as f64;
    }
    (acc / n as f64) as f32
}

/// dpred[i] = -target[i] / pred[i] / n. (gradient wrt pred only)
pub unsafe fn kl_div_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let p = pred[i].max(1e-12);
        dpred[i] = -target[i] / p * inv_n;
    }
}

/// Huber loss with parameter `delta`:
///   loss[i] = 0.5 * d^2                  if |d| <= delta
///           = delta * (|d| - 0.5*delta)  otherwise
/// where d = pred[i] - target[i]. Reduce by mean.
pub unsafe fn huber_f32(
    pred: *const f32, target: *const f32, n: usize, delta: f32,
) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let d = pred[i] - target[i];
        let ad = d.abs();
        let l = if ad <= delta { 0.5 * d * d } else { delta * (ad - 0.5 * delta) };
        acc += l as f64;
    }
    (acc / n as f64) as f32
}

/// dpred[i] = d if |d|<=delta else delta*sign(d), divided by n.
pub unsafe fn huber_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize, delta: f32,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let d = pred[i] - target[i];
        let g = if d.abs() <= delta { d } else if d > 0.0 { delta } else { -delta };
        dpred[i] = g * inv_n;
    }
}

/// Smooth-L1 (Huber with delta=1, beta-form):
///   loss[i] = 0.5 * d^2 / beta  if |d| < beta
///           = |d| - 0.5 * beta  otherwise
/// where d = pred - target. Reduce by mean.
pub unsafe fn smooth_l1_f32(
    pred: *const f32, target: *const f32, n: usize, beta: f32,
) -> f32 {
    if n == 0 { return 0.0; }
    let pred = s(pred, n);
    let target = s(target, n);
    let mut acc = 0.0f64;
    for i in 0..n {
        let d = pred[i] - target[i];
        let ad = d.abs();
        let l = if ad < beta { 0.5 * d * d / beta } else { ad - 0.5 * beta };
        acc += l as f64;
    }
    (acc / n as f64) as f32
}

/// dpred[i] = d/beta if |d|<beta else sign(d), divided by n.
pub unsafe fn smooth_l1_backward_f32(
    pred: *const f32, target: *const f32, dpred: *mut f32, n: usize, beta: f32,
) {
    if n == 0 { return; }
    let pred = s(pred, n);
    let target = s(target, n);
    let dpred = sm(dpred, n);
    let inv_n = 1.0 / n as f32;
    for i in 0..n {
        let d = pred[i] - target[i];
        let g = if d.abs() < beta { d / beta } else if d > 0.0 { 1.0 } else { -1.0 };
        dpred[i] = g * inv_n;
    }
}

/// Triplet margin loss (single triplet, dim-d vectors):
///   loss = max(0, ||a-p||^2 - ||a-n||^2 + margin)
pub unsafe fn triplet_f32(
    anchor: *const f32, positive: *const f32, negative: *const f32,
    d: usize, margin: f32,
) -> f32 {
    let a = s(anchor, d);
    let p = s(positive, d);
    let nn = s(negative, d);
    let mut dap = 0.0f32;
    let mut dan = 0.0f32;
    for i in 0..d {
        let u = a[i] - p[i]; dap += u * u;
        let v = a[i] - nn[i]; dan += v * v;
    }
    let raw = dap - dan + margin;
    if raw > 0.0 { raw } else { 0.0 }
}

/// Backward wrt anchor: 2*(a-p) - 2*(a-n) when active, else 0. Caller can
/// derive d_positive / d_negative from -2*(a-p) and 2*(a-n) symmetrically.
pub unsafe fn triplet_backward_f32(
    anchor: *const f32, positive: *const f32, negative: *const f32,
    d_anchor: *mut f32, d: usize, margin: f32,
) {
    let a = s(anchor, d);
    let p = s(positive, d);
    let nn = s(negative, d);
    let da = sm(d_anchor, d);
    let mut dap = 0.0f32;
    let mut dan = 0.0f32;
    for i in 0..d {
        let u = a[i] - p[i]; dap += u * u;
        let v = a[i] - nn[i]; dan += v * v;
    }
    let active = dap - dan + margin > 0.0;
    if active {
        for i in 0..d { da[i] = 2.0 * (nn[i] - p[i]); }
    } else {
        for i in 0..d { da[i] = 0.0; }
    }
}

/// Contrastive loss (Hadsell et al.) on a single pair, dim-d vectors:
///   d2 = ||x1 - x2||^2
///   loss = y * d2 + (1-y) * max(0, margin - sqrt(d2))^2
/// y == 1 means "similar"; y == 0 means "dissimilar".
pub unsafe fn contrastive_f32(
    x1: *const f32, x2: *const f32, y: f32, d: usize, margin: f32,
) -> f32 {
    let a = s(x1, d);
    let b = s(x2, d);
    let mut d2 = 0.0f32;
    for i in 0..d { let u = a[i] - b[i]; d2 += u * u; }
    let dist = d2.sqrt();
    let neg_part = (margin - dist).max(0.0);
    y * d2 + (1.0 - y) * neg_part * neg_part
}

/// dx1[i]: 2*y*(x1-x2)[i] for similar pair; for dissimilar, when margin>dist,
/// dx1[i] = -2*(margin - dist)*(x1-x2)[i] / dist (else 0).
pub unsafe fn contrastive_backward_f32(
    x1: *const f32, x2: *const f32, dx1: *mut f32,
    y: f32, d: usize, margin: f32,
) {
    let a = s(x1, d);
    let b = s(x2, d);
    let dx = sm(dx1, d);
    let mut d2 = 0.0f32;
    for i in 0..d { let u = a[i] - b[i]; d2 += u * u; }
    let dist = d2.sqrt().max(1e-12);
    let neg_active = margin - dist > 0.0;
    for i in 0..d {
        let diff = a[i] - b[i];
        let g_pos = 2.0 * y * diff;
        let g_neg = if neg_active {
            -2.0 * (margin - dist) * diff / dist * (1.0 - y)
        } else { 0.0 };
        dx[i] = g_pos + g_neg;
    }
}

// ---------------------------------------------------------------- embedding

/// out[b, s, d] = w[ids[b, s], d]. Shapes: w [V, D], ids [B, S] i32, out [B, S, D].
pub unsafe fn embedding_lookup_f32(
    w: *const f32, ids: *const i32, out: *mut f32,
    b: usize, s_len: usize, v_size: usize, d: usize,
) {
    let w = s(w, v_size * d);
    let ids = slice::from_raw_parts(ids, b * s_len);
    let out = sm(out, b * s_len * d);
    for i in 0..b * s_len {
        let id = ids[i] as usize;
        let src = &w[id * d..(id + 1) * d];
        let dst = &mut out[i * d..(i + 1) * d];
        dst.copy_from_slice(src);
    }
}

/// Backward: dW[id, :] += sum_{b,s : ids[b,s]==id} dY[b, s, :].
/// `dW` must be zero-initialised by the caller.
pub unsafe fn embedding_backward_f32(
    ids: *const i32, dy: *const f32, dw: *mut f32,
    b: usize, s_len: usize, v_size: usize, d: usize,
) {
    let ids = slice::from_raw_parts(ids, b * s_len);
    let dy = s(dy, b * s_len * d);
    let dw = sm(dw, v_size * d);
    for i in 0..b * s_len {
        let id = ids[i] as usize;
        let src = &dy[i * d..(i + 1) * d];
        let dst = &mut dw[id * d..(id + 1) * d];
        for j in 0..d { dst[j] += src[j]; }
    }
}

// ---------------------------------------------------------------- gradient utils

pub unsafe fn zero_grad_f32(g: *mut f32, n: usize) {
    let g = sm(g, n);
    for v in g.iter_mut() { *v = 0.0; }
}

pub unsafe fn clip_grad_norm_f32(g: *mut f32, max_norm: f32, n: usize) -> f32 {
    let g = sm(g, n);
    let mut sq = 0.0f64;
    for &v in g.iter() { sq += (v as f64) * (v as f64); }
    let norm = sq.sqrt() as f32;
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for v in g.iter_mut() { *v *= scale; }
    }
    norm
}

// ---------------------------------------------------------------- optim

/// Fused AdamW step. Mutates param, m, v in place.
pub unsafe fn adamw_step_f32(
    param: *mut f32, grad: *const f32,
    m: *mut f32, v_buf: *mut f32,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    step: i64, n: usize,
) {
    let param = sm(param, n);
    let grad = s(grad, n);
    let m = sm(m, n);
    let v_buf = sm(v_buf, n);
    let bias1 = 1.0 - beta1.powi(step as i32);
    let bias2 = 1.0 - beta2.powi(step as i32);
    for i in 0..n {
        let g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v_buf[i] = beta2 * v_buf[i] + (1.0 - beta2) * g * g;
        let m_hat = m[i] / bias1;
        let v_hat = v_buf[i] / bias2;
        // decoupled weight decay
        param[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + wd * param[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_close(a: f32, b: f32, eps: f32) -> bool { (a - b).abs() < eps }

    #[test]
    fn matmul_identity() {
        // [2,2] @ I[2,2] = [2,2]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut out = vec![0.0; 4];
        unsafe { matmul_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 2, 2, 2); }
        assert_eq!(out, a);
    }

    #[test]
    fn gelu_then_backward_finite_diff() {
        let x = vec![0.5, -0.3, 1.2];
        let mut y = x.clone();
        unsafe { gelu_f32(y.as_mut_ptr(), 3); }
        // numeric gradient
        let h = 1e-3f32;
        let mut numeric = vec![0.0; 3];
        for i in 0..3 {
            let mut xp = x.clone(); xp[i] += h;
            let mut xm = x.clone(); xm[i] -= h;
            unsafe {
                gelu_f32(xp.as_mut_ptr(), 3);
                gelu_f32(xm.as_mut_ptr(), 3);
            }
            numeric[i] = (xp[i] - xm[i]) / (2.0 * h);
        }
        let dy = vec![1.0, 1.0, 1.0];
        let mut analytic = vec![0.0; 3];
        unsafe { gelu_backward_f32(x.as_ptr(), dy.as_ptr(), analytic.as_mut_ptr(), 3); }
        for i in 0..3 {
            assert!(f32_close(analytic[i], numeric[i], 1e-2),
                "mismatch i={i}: analytic={} numeric={}", analytic[i], numeric[i]);
        }
    }

    #[test]
    fn silu_then_backward_finite_diff() {
        let x = vec![0.7f32, -0.4, 1.5];
        // numeric gradient of sum(silu(x)) wrt x is silu'(x).
        let h = 1e-3f32;
        let mut numeric = vec![0.0f32; 3];
        for i in 0..3 {
            let mut xp = x.clone(); xp[i] += h;
            let mut xm = x.clone(); xm[i] -= h;
            unsafe { silu_f32(xp.as_mut_ptr(), 3); silu_f32(xm.as_mut_ptr(), 3); }
            numeric[i] = (xp[i] - xm[i]) / (2.0 * h);
        }
        let dy = vec![1.0f32; 3];
        let mut analytic = vec![0.0f32; 3];
        unsafe { silu_backward_f32(x.as_ptr(), dy.as_ptr(), analytic.as_mut_ptr(), 3); }
        for i in 0..3 {
            assert!(f32_close(analytic[i], numeric[i], 1e-2),
                "silu mismatch i={i}: {} vs {}", analytic[i], numeric[i]);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        unsafe { softmax_last_f32(x.as_mut_ptr(), 2, 3); }
        let s1: f32 = x[0..3].iter().sum();
        let s2: f32 = x[3..6].iter().sum();
        assert!(f32_close(s1, 1.0, 1e-5));
        assert!(f32_close(s2, 1.0, 1e-5));
    }

    #[test]
    fn cross_entropy_correct_label_low_loss() {
        // logits strongly favouring index 1; labels = [1].
        let logits = vec![0.0, 5.0, 0.0];
        let labels = vec![1i32];
        let mut probs = vec![0.0; 3];
        let loss = unsafe { cross_entropy_f32(logits.as_ptr(), labels.as_ptr(), probs.as_mut_ptr(), 1, 3) };
        assert!(loss < 0.05, "loss too high: {}", loss);
    }

    #[test]
    fn adamw_decreases_param_when_grad_positive() {
        let mut p = vec![1.0f32, 2.0, 3.0];
        let g = vec![1.0f32, 1.0, 1.0];
        let mut m = vec![0.0f32; 3];
        let mut v = vec![0.0f32; 3];
        unsafe {
            adamw_step_f32(p.as_mut_ptr(), g.as_ptr(), m.as_mut_ptr(), v.as_mut_ptr(),
                0.1, 0.9, 0.999, 1e-8, 0.0, 1, 3);
        }
        assert!(p[0] < 1.0); assert!(p[1] < 2.0); assert!(p[2] < 3.0);
    }

    #[test]
    fn rms_norm_identity_gamma_unit_input() {
        // x = [1, 1, 1, 1], gamma = [1, 1, 1, 1]:
        //   mean(x^2) = 1.0, inv_rms = 1/sqrt(1+eps) ~ 1.0
        //   out_i = x_i * 1.0 * gamma_i ~ x_i
        let x = vec![1.0f32; 4];
        let g = vec![1.0f32; 4];
        let mut out = vec![0.0f32; 4];
        unsafe { rms_norm_f32(x.as_ptr(), g.as_ptr(), 1e-5, out.as_mut_ptr(), 1, 4); }
        for v in &out { assert!(f32_close(*v, 1.0, 1e-3)); }
    }

    #[test]
    fn rms_norm_normalizes_magnitude() {
        // x = [2, 0, 0, 0]: mean(x^2) = 1.0, inv_rms ~ 1.0
        //   out = [2, 0, 0, 0]
        // x = [4, 0, 0, 0]: mean(x^2) = 4.0, inv_rms ~ 0.5
        //   out = [2, 0, 0, 0]
        // So out is the same regardless of input magnitude (scale-invariant).
        let mut o1 = vec![0.0f32; 4];
        let mut o2 = vec![0.0f32; 4];
        let g = vec![1.0f32; 4];
        unsafe {
            rms_norm_f32([2.0f32, 0.0, 0.0, 0.0].as_ptr(), g.as_ptr(),
                1e-5, o1.as_mut_ptr(), 1, 4);
            rms_norm_f32([4.0f32, 0.0, 0.0, 0.0].as_ptr(), g.as_ptr(),
                1e-5, o2.as_mut_ptr(), 1, 4);
        }
        for i in 0..4 { assert!(f32_close(o1[i], o2[i], 1e-3),
            "RMSNorm should be scale-invariant: o1[{}]={} vs o2[{}]={}", i, o1[i], i, o2[i]); }
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        // At position 0, theta = 0 so sin = 0, cos = 1 -- identity rotation.
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x.clone();
        unsafe { rope_apply_f32(x.as_mut_ptr(), 1, 1, 8, 10000.0, 0); }
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!(f32_close(*a, *b, 1e-5), "RoPE at pos 0 should be identity: {} != {}", a, b);
        }
    }

    #[test]
    fn rope_norm_preserved() {
        // RoPE rotates pairs; ||x|| must be preserved.
        let mut x = vec![1.0f32, 0.5, -0.3, 0.8];
        let norm_in: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        unsafe { rope_apply_f32(x.as_mut_ptr(), 1, 1, 4, 10000.0, 3); }
        let norm_out: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_in - norm_out).abs() < 1e-5,
            "RoPE must preserve L2 norm: {} -> {}", norm_in, norm_out);
    }

    #[test]
    fn apply_lora_zero_is_noop() {
        // LoRA with all-zero A and B should leave W unchanged.
        let mut w = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];  // [d_in=2, d_out=3]
        let w_orig = w.clone();
        let a = vec![0.0f32; 2 * 2];  // [rank=2, d_in=2]
        let b = vec![0.0f32; 3 * 2];  // [d_out=3, rank=2]
        unsafe { apply_lora_f32(w.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 1.0, 2, 3, 2); }
        assert_eq!(w, w_orig);
    }

    #[test]
    fn apply_lora_matches_direct_math() {
        // d_in=2, d_out=3, rank=2, scale=0.5
        // A = [[1, 2], [3, 4]]   (rank x d_in)
        // B = [[1, 0], [0, 1], [1, 1]]   (d_out x rank)
        // BA = B @ A in math notation = [d_out, d_in]:
        //   BA[0] = 1*A[0] + 0*A[1] = [1, 2]
        //   BA[1] = 0*A[0] + 1*A[1] = [3, 4]
        //   BA[2] = 1*A[0] + 1*A[1] = [4, 6]
        // BA = [[1, 2], [3, 4], [4, 6]]
        // In matmul layout we add scale * BA^T to W [d_in, d_out]:
        //   BA^T[0] = [1, 3, 4]      <- d_in=0 row, d_out 0..3
        //   BA^T[1] = [2, 4, 6]      <- d_in=1 row, d_out 0..3
        // Starting W = [[0, 0, 0], [0, 0, 0]] (in matmul layout):
        // Result: W = scale * [[1, 3, 4], [2, 4, 6]] = [[0.5, 1.5, 2.0], [1.0, 2.0, 3.0]]
        let mut w = vec![0.0f32; 2 * 3];
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        unsafe { apply_lora_f32(w.as_mut_ptr(), a.as_ptr(), b.as_ptr(), 0.5, 2, 3, 2); }
        let expected = vec![0.5f32, 1.5, 2.0, 1.0, 2.0, 3.0];
        for (i, (a, e)) in w.iter().zip(expected.iter()).enumerate() {
            assert!(f32_close(*a, *e, 1e-5), "[{}] {} != {}", i, a, e);
        }
    }

    #[test]
    fn gqa_repeat_broadcasts() {
        // seq=1, n_kv_heads=2, head_dim=3, n_q_heads=4 -> g=2:
        //   kv_in = [[1,2,3], [4,5,6]]
        //   kv_out = [[1,2,3], [1,2,3], [4,5,6], [4,5,6]]
        let kv_in = vec![1.0f32, 2.0, 3.0,  4.0, 5.0, 6.0];
        let mut kv_out = vec![0.0f32; 12];
        unsafe { gqa_repeat_kv_f32(kv_in.as_ptr(), kv_out.as_mut_ptr(), 1, 2, 3, 4); }
        let expected = vec![1.0, 2.0, 3.0,  1.0, 2.0, 3.0,  4.0, 5.0, 6.0,  4.0, 5.0, 6.0];
        assert_eq!(kv_out, expected);
    }
}
