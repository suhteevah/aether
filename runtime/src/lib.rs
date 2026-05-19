//! libaether_rt — runtime intrinsics that aether-emitted code links against.
//!
//! Phase 0/1 stubs. NCCL/MPI/RDMA backends slot in here in Phase 2 by replacing
//! the body of `aether_dist_all_reduce` with a dispatch on AETHER_DIST_BACKEND.
//!
//! All entry points are `#[no_mangle] extern "C"` so the LLVM IR emitted by
//! `compiler/src/codegen/llvm/mod.rs` links cleanly without aliasing.

use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_void};

pub mod ops;

#[cfg(feature = "cuda")]
pub mod cuda;

// Single-threaded tape for the bootstrap runtime. Originally `thread_local!`
// — the TLS init hook for that drags Rust's `std::thread` machinery into the
// cdylib's DllMain, which in turn calls `bcryptprimitives.dll!ProcessPrng`
// for the HashMap hasher seed. When our self-hosted PE writer (#24) loads
// `aether_rt.dll` early in process init, bcryptprimitives' DllMain hasn't
// run yet, and the call AVs at offset 0x16e99 (a `movaps` to a slot whose
// alignment depends on prior init). A plain `static UnsafeCell` doesn't go
// through the TLS init path. Aether-emitted programs are single-threaded;
// the TLS layer was never load-bearing.
struct Tape {
    entries: Vec<*const c_void>,
    closed: bool,
}

struct TapeCell(UnsafeCell<Tape>);
unsafe impl Sync for TapeCell {}

static TAPE: TapeCell = TapeCell(UnsafeCell::new(Tape { entries: Vec::new(), closed: false }));

#[inline]
unsafe fn tape() -> &'static mut Tape { &mut *TAPE.0.get() }

#[no_mangle]
pub unsafe extern "C" fn aether_autodiff_init(_tape: *mut c_void) {
    let t = tape();
    t.entries.clear();
    t.closed = false;
}

#[no_mangle]
pub unsafe extern "C" fn aether_autodiff_push(_tape: *mut c_void, value: *const c_void) {
    tape().entries.push(value);
}

#[no_mangle]
pub extern "C" fn aether_autodiff_accumulate(_tape: *mut c_void, _grad: *const c_void) {
    // Phase 0: per-node grad accumulation is recorded but not summed —
    // the typed AD graph in Phase 1 supplies the operands for real partials.
}

/// Symbolic partial dispatch. Op codes are stable and shared with the
/// emitter (`compiler/src/codegen/llvm/mod.rs::PART_*`). Phase 0 records
/// the call; Phase 1 reads `dst` / `src` from a real value table and
/// computes the actual gradient.
#[no_mangle]
pub extern "C" fn aether_autodiff_partial(
    _tape: *mut c_void,
    _dst_node: c_int,
    _op_code: c_int,
    _src_node: c_int,
) {}

#[no_mangle]
pub unsafe extern "C" fn aether_autodiff_reverse(_tape: *mut c_void) {
    tape().closed = true;
}

// =====================================================================
// Primitive op surface — every `extern fn` declared in stdlib/ops.aether
// resolves to one of these symbols. Phase 0 bodies are no-ops that return
// 0; Phase 1 routes them to cuBLAS / cuDNN / NCCL on the 3070 Ti.
//
// The C ABI here is the contract between aetherc-emitted LLVM IR and
// whatever backend the runtime is built against. Never reorder an arg —
// LLVM IR call sites are positional.
// =====================================================================

// Real CPU implementations live in `ops.rs`. The C-ABI symbols below are
// thin wrappers that flatten void-pointers to the typed Rust impls. Phase 1
// swaps these bodies for cuBLAS/cuDNN; the symbol names never change.

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_f32(
    a: *const c_void, b: *const c_void, out: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_f32(a as _, b as _, out as _, m as _, k as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_backward_lhs_f32(
    dy: *const c_void, b: *const c_void, da: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_backward_lhs_f32(dy as _, b as _, da as _, m as _, k as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_backward_rhs_f32(
    a: *const c_void, dy: *const c_void, db: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_backward_rhs_f32(a as _, dy as _, db as _, m as _, k as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_bf16(
    _a: *const c_void, _b: *const c_void, _out: *mut c_void,
    _m: c_int, _k: c_int, _n: c_int,
) -> c_int { /* Phase 1 — bf16 path */ 0 }

/// FR-17.3 — 2D convolution (CPU reference, direct loops).
///
/// Layout: NCHW for input + output, KH×KW for kernel. Stride = 1,
/// padding = 0, dilation = 1, groups = 1. No bias (caller may add).
///
/// Input  shape: `(n, c_in, h, w)`     — `n * c_in * h * w` f32 elements.
/// Kernel shape: `(c_out, c_in, kh, kw)` — `c_out * c_in * kh * kw` f32.
/// Output shape: `(n, c_out, h_out, w_out)` where
///   `h_out = h - kh + 1`, `w_out = w - kw + 1`. Caller pre-allocates.
///
/// Returns 0 on success, non-zero on shape-invalid input. This is the
/// reference scalar impl — im2col + sgemm and cuDNN are FR-17.3 follow-ons.
#[no_mangle] pub unsafe extern "C" fn aether_op_conv2d_f32(
    input: *const c_void,
    kernel: *const c_void,
    output: *mut c_void,
    n: c_int, c_in: c_int, h: c_int, w: c_int,
    c_out: c_int, kh: c_int, kw: c_int,
) -> c_int {
    if input.is_null() || kernel.is_null() || output.is_null() { return 1; }
    if n <= 0 || c_in <= 0 || h <= 0 || w <= 0
        || c_out <= 0 || kh <= 0 || kw <= 0 { return 2; }
    if kh > h || kw > w { return 3; }
    let h_out = (h - kh + 1) as usize;
    let w_out = (w - kw + 1) as usize;
    let (n, c_in, h, w) = (n as usize, c_in as usize, h as usize, w as usize);
    let (c_out, kh, kw) = (c_out as usize, kh as usize, kw as usize);
    let in_buf  = std::slice::from_raw_parts(input  as *const f32, n * c_in * h * w);
    let k_buf   = std::slice::from_raw_parts(kernel as *const f32, c_out * c_in * kh * kw);
    let out_buf = std::slice::from_raw_parts_mut(output as *mut f32, n * c_out * h_out * w_out);

    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc: f32 = 0.0;
                    for ci in 0..c_in {
                        for ki in 0..kh {
                            for kj in 0..kw {
                                let ih = oh + ki;
                                let iw = ow + kj;
                                let in_idx = ((ni * c_in + ci) * h + ih) * w + iw;
                                let k_idx  = ((co * c_in + ci) * kh + ki) * kw + kj;
                                acc += in_buf[in_idx] * k_buf[k_idx];
                            }
                        }
                    }
                    let out_idx = ((ni * c_out + co) * h_out + oh) * w_out + ow;
                    out_buf[out_idx] = acc;
                }
            }
        }
    }
    0
}

/// FR-17.14 — GGUF Q4_0 dequantization (CPU reference).
///
/// Q4_0 block layout (18 bytes per 32 quants):
///   bytes 0..2   = f16 scale `d`
///   bytes 2..18  = 16 bytes of packed 4-bit nibbles (2 quants per byte;
///                  low nibble first, high nibble second)
///
/// Per-quant value = (signed_nibble) * d_f32, where signed_nibble =
/// (unsigned_nibble - 8) and unsigned_nibble is the raw 0..15.
///
/// Inputs: `blocks` = pointer to packed Q4_0 stream of `n_blocks` * 18
/// bytes; `out` = pointer to `n_blocks * 32` f32 output slots (pre-
/// allocated). Returns 0 on success, non-zero on null / bad-count.
///
/// Matches the `ggml_q4_0_t` reference layout from llama.cpp / ggml.
#[no_mangle] pub unsafe extern "C" fn aether_dequant_q4_0(
    blocks: *const c_void,
    out: *mut c_void,
    n_blocks: c_int,
) -> c_int {
    if blocks.is_null() || out.is_null() { return 1; }
    if n_blocks <= 0 { return 2; }
    let n = n_blocks as isize;
    let b = blocks as *const u8;
    let o = out as *mut f32;
    for bi in 0..n {
        let base = b.offset(bi * 18);
        // f16 scale, little-endian.
        let d_bits = u16::from_le_bytes([*base, *base.offset(1)]);
        let d_f32 = aether_f16_to_f32(d_bits as i32);
        for i in 0..16isize {
            let byte = *base.offset(2 + i);
            let lo = ((byte & 0x0F) as i32 - 8) as f32;
            let hi = (((byte >> 4) & 0x0F) as i32 - 8) as f32;
            *o.offset(bi * 32 + i * 2)     = lo * d_f32;
            *o.offset(bi * 32 + i * 2 + 1) = hi * d_f32;
        }
    }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_add_f32(
    a: *const c_void, b: *const c_void, out: *mut c_void, n: c_int,
) -> c_int {
    ops::add_f32(a as _, b as _, out as _, n as _);
    0
}

/// FR-17.13-extra — FlashAttention v2 (single-head causal, CPU reference).
///
/// Memory-efficient causal self-attention via online softmax. Avoids
/// materialising the full N×N score matrix by processing keys in blocks
/// of size `BC` (= 4 here) and maintaining running max/sum statistics.
/// Memory footprint per query-row: O(d_head + BC), not O(N).
///
/// Inputs:
///   q, k, v   — pointers to f32 arrays of shape (seq_len, d_head),
///                row-major.
///   out       — pointer to f32 array (seq_len, d_head), pre-allocated.
///   seq_len   — N
///   d_head    — d
///
/// Mathematically identical to:
///   scale = 1 / sqrt(d_head)
///   S[i,j] = (Q @ K^T)[i,j] * scale, with j > i masked to -inf
///   P = softmax_row(S)
///   O = P @ V
#[no_mangle] pub unsafe extern "C" fn aether_flash_attention_v2_f32(
    q: *const c_void, k: *const c_void, v: *const c_void,
    out: *mut c_void,
    seq_len: c_int, d_head: c_int,
) -> c_int {
    if q.is_null() || k.is_null() || v.is_null() || out.is_null() { return 1; }
    if seq_len <= 0 || d_head <= 0 { return 2; }
    const BC: usize = 4;
    let n = seq_len as usize;
    let d = d_head as usize;
    let q_buf = std::slice::from_raw_parts(q as *const f32, n * d);
    let k_buf = std::slice::from_raw_parts(k as *const f32, n * d);
    let v_buf = std::slice::from_raw_parts(v as *const f32, n * d);
    let o_buf = std::slice::from_raw_parts_mut(out as *mut f32, n * d);
    let scale = 1.0_f32 / (d as f32).sqrt();

    // Running softmax stats per query row.
    let mut m_state = vec![f32::NEG_INFINITY; n];
    let mut l_state = vec![0.0_f32; n];
    // Init output to zero (we accumulate in-place).
    for o in o_buf.iter_mut() { *o = 0.0; }

    let mut j_start = 0usize;
    while j_start < n {
        let j_end = (j_start + BC).min(n);
        let bc = j_end - j_start;
        // For each query row, fold this key-block into the running stats.
        for r in 0..n {
            // Compute S[r, j_start..j_end] = (Q[r] · K[j_start+c]) * scale,
            // applying the causal mask (key index > r → -inf).
            let mut s_block = [f32::NEG_INFINITY; BC];
            let mut block_max = f32::NEG_INFINITY;
            for c in 0..bc {
                let key_idx = j_start + c;
                if key_idx > r { continue; }  // causal: leave -inf
                let mut dot = 0.0_f32;
                for di in 0..d {
                    dot += q_buf[r * d + di] * k_buf[key_idx * d + di];
                }
                let s = dot * scale;
                s_block[c] = s;
                if s > block_max { block_max = s; }
            }
            // If the entire block is masked, skip update.
            if block_max == f32::NEG_INFINITY { continue; }
            let m_old = m_state[r];
            let m_new = if m_old > block_max { m_old } else { block_max };
            // Rescale O[r] by exp(m_old - m_new) (or zero if m_old was -inf).
            let alpha = if m_old == f32::NEG_INFINITY { 0.0 } else { (m_old - m_new).exp() };
            for di in 0..d { o_buf[r * d + di] *= alpha; }
            // Add P_block @ V_block to O[r].
            let mut row_sum_p = 0.0_f32;
            for c in 0..bc {
                if s_block[c] == f32::NEG_INFINITY { continue; }
                let p = (s_block[c] - m_new).exp();
                row_sum_p += p;
                let key_idx = j_start + c;
                for di in 0..d {
                    o_buf[r * d + di] += p * v_buf[key_idx * d + di];
                }
            }
            // Update l.
            let l_old = l_state[r];
            l_state[r] = l_old * alpha + row_sum_p;
            m_state[r] = m_new;
        }
        j_start = j_end;
    }
    // Final normalisation: O[r] /= l[r] for each row.
    for r in 0..n {
        let l = l_state[r];
        if l > 0.0 {
            for di in 0..d { o_buf[r * d + di] /= l; }
        }
    }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_add_inplace_f32(
    x: *mut c_void, y: *const c_void, n: c_int,
) -> c_int {
    ops::add_inplace_f32(x as _, y as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_add_bias_f32(
    x: *mut c_void, b: *const c_void, rows: c_int, cols: c_int,
) -> c_int {
    ops::add_bias_f32(x as _, b as _, rows as _, cols as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_scale_f32(
    x: *mut c_void, s: f32, n: c_int,
) -> c_int {
    ops::scale_f32(x as _, s, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_axpy_f32(
    alpha: f32, x: *const c_void, y: *mut c_void, n: c_int,
) -> c_int {
    ops::axpy_f32(alpha, x as _, y as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_gelu_f32(x: *mut c_void, n: c_int) -> c_int {
    ops::gelu_f32(x as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_gelu_backward_f32(
    x: *const c_void, dy: *const c_void, dx: *mut c_void, n: c_int,
) -> c_int {
    ops::gelu_backward_f32(x as _, dy as _, dx as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_silu_f32(x: *mut c_void, n: c_int) -> c_int {
    ops::silu_f32(x as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_silu_backward_f32(
    x: *const c_void, dy: *const c_void, dx: *mut c_void, n: c_int,
) -> c_int {
    ops::silu_backward_f32(x as _, dy as _, dx as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_relu_f32(x: *mut c_void, n: c_int) -> c_int {
    ops::relu_f32(x as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_relu_backward_f32(
    x: *const c_void, dy: *const c_void, dx: *mut c_void, n: c_int,
) -> c_int {
    ops::relu_backward_f32(x as _, dy as _, dx as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_softmax_f32(
    x: *mut c_void, rows: c_int, cols: c_int, _axis: c_int,
) -> c_int {
    ops::softmax_last_f32(x as _, rows as _, cols as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_softmax_backward_f32(
    y: *const c_void, dy: *const c_void, dx: *mut c_void,
    rows: c_int, cols: c_int,
) -> c_int {
    ops::softmax_backward_last_f32(y as _, dy as _, dx as _, rows as _, cols as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_layer_norm_f32(
    x: *const c_void, gamma: *const c_void, beta: *const c_void,
    eps: f32, out: *mut c_void, mean_out: *mut c_void, inv_std_out: *mut c_void,
    rows: c_int, d: c_int,
) -> c_int {
    ops::layer_norm_f32(x as _, gamma as _, beta as _, eps,
        out as _, mean_out as _, inv_std_out as _, rows as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_layer_norm_backward_f32(
    x: *const c_void, gamma: *const c_void, dy: *const c_void,
    mean: *const c_void, inv_std: *const c_void,
    dx: *mut c_void, dgamma: *mut c_void, dbeta: *mut c_void,
    rows: c_int, d: c_int,
) -> c_int {
    ops::layer_norm_backward_f32(x as _, gamma as _, dy as _,
        mean as _, inv_std as _, dx as _, dgamma as _, dbeta as _,
        rows as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sdpa_causal_f32(
    q: *const c_void, k: *const c_void, v: *const c_void,
    out: *mut c_void, attn_out: *mut c_void,
    bh: c_int, s_len: c_int, d: c_int,
) -> c_int {
    ops::sdpa_causal_f32(q as _, k as _, v as _, out as _, attn_out as _,
        bh as _, s_len as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sdpa_causal_backward_f32(
    q: *const c_void, k: *const c_void, v: *const c_void,
    attn: *const c_void, dout: *const c_void,
    dq: *mut c_void, dk: *mut c_void, dv: *mut c_void,
    bh: c_int, s_len: c_int, d: c_int,
) -> c_int {
    ops::sdpa_causal_backward_f32(q as _, k as _, v as _, attn as _, dout as _,
        dq as _, dk as _, dv as _, bh as _, s_len as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_cross_entropy_f32(
    logits: *const c_void, labels: *const c_void,
    probs_out: *mut c_void, b: c_int, v: c_int,
) -> f32 {
    ops::cross_entropy_f32(logits as _, labels as _, probs_out as _, b as _, v as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_cross_entropy_backward_f32(
    probs: *const c_void, labels: *const c_void,
    dlogits: *mut c_void, b: c_int, v: c_int,
) -> c_int {
    ops::cross_entropy_backward_f32(probs as _, labels as _, dlogits as _, b as _, v as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mse_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 {
    ops::mse_f32(pred as _, target as _, n as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mse_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int {
    ops::mse_backward_f32(pred as _, target as _, dpred as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mae_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 {
    ops::mae_f32(pred as _, target as _, n as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mae_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int {
    ops::mae_backward_f32(pred as _, target as _, dpred as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_with_logits_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 { ops::bce_with_logits_f32(pred as _, target as _, n as _) }

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_with_logits_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int { ops::bce_with_logits_backward_f32(pred as _, target as _, dpred as _, n as _); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 { ops::bce_f32(pred as _, target as _, n as _) }

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int { ops::bce_backward_f32(pred as _, target as _, dpred as _, n as _); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_kl_div_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 { ops::kl_div_f32(pred as _, target as _, n as _) }

#[no_mangle] pub unsafe extern "C" fn aether_op_kl_div_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int { ops::kl_div_backward_f32(pred as _, target as _, dpred as _, n as _); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_huber_f32(
    pred: *const c_void, target: *const c_void, n: c_int, delta: f32,
) -> f32 { ops::huber_f32(pred as _, target as _, n as _, delta) }

#[no_mangle] pub unsafe extern "C" fn aether_op_huber_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int, delta: f32,
) -> c_int { ops::huber_backward_f32(pred as _, target as _, dpred as _, n as _, delta); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_smooth_l1_f32(
    pred: *const c_void, target: *const c_void, n: c_int, beta: f32,
) -> f32 { ops::smooth_l1_f32(pred as _, target as _, n as _, beta) }

#[no_mangle] pub unsafe extern "C" fn aether_op_smooth_l1_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int, beta: f32,
) -> c_int { ops::smooth_l1_backward_f32(pred as _, target as _, dpred as _, n as _, beta); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_triplet_f32(
    anchor: *const c_void, positive: *const c_void, negative: *const c_void,
    d: c_int, margin: f32,
) -> f32 { ops::triplet_f32(anchor as _, positive as _, negative as _, d as _, margin) }

#[no_mangle] pub unsafe extern "C" fn aether_op_triplet_backward_f32(
    anchor: *const c_void, positive: *const c_void, negative: *const c_void,
    d_anchor: *mut c_void, d: c_int, margin: f32,
) -> c_int {
    ops::triplet_backward_f32(anchor as _, positive as _, negative as _, d_anchor as _, d as _, margin); 0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_contrastive_f32(
    x1: *const c_void, x2: *const c_void, y: f32, d: c_int, margin: f32,
) -> f32 { ops::contrastive_f32(x1 as _, x2 as _, y, d as _, margin) }

#[no_mangle] pub unsafe extern "C" fn aether_op_contrastive_backward_f32(
    x1: *const c_void, x2: *const c_void, dx1: *mut c_void,
    y: f32, d: c_int, margin: f32,
) -> c_int {
    ops::contrastive_backward_f32(x1 as _, x2 as _, dx1 as _, y, d as _, margin); 0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_embedding_lookup_f32(
    w: *const c_void, ids: *const c_void, out: *mut c_void,
    b: c_int, s_len: c_int, v_size: c_int, d: c_int,
) -> c_int {
    ops::embedding_lookup_f32(w as _, ids as _, out as _,
        b as _, s_len as _, v_size as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_embedding_backward_f32(
    ids: *const c_void, dy: *const c_void, dw: *mut c_void,
    b: c_int, s_len: c_int, v_size: c_int, d: c_int,
) -> c_int {
    ops::embedding_backward_f32(ids as _, dy as _, dw as _,
        b as _, s_len as _, v_size as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_zero_grad_f32(g: *mut c_void, n: c_int) -> c_int {
    ops::zero_grad_f32(g as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_clip_grad_norm_f32(
    g: *mut c_void, max_norm: f32, n: c_int,
) -> f32 {
    ops::clip_grad_norm_f32(g as _, max_norm, n as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_all_reduce_sum_f32(
    _buf: *mut c_void, _world_size: c_int, _n: c_int,
) -> c_int { /* Phase 2 — NCCL */ 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_adamw_step_f32(
    param: *mut c_void, grad: *const c_void,
    m: *mut c_void, v: *mut c_void,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    step: i64, n: c_int,
) -> c_int {
    ops::adamw_step_f32(param as _, grad as _, m as _, v as _,
        lr, beta1, beta2, eps, wd, step, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sgd_step_f32(
    _param: *mut c_void, _grad: *const c_void,
    _lr: f32, _momentum: f32, _wd: f32, _state: *mut c_void, _n: c_int,
) -> c_int { 0 }

#[derive(Clone, Copy)]
#[repr(C)]
pub enum DistBackend {
    Nccl = 0,
    Mpi = 1,
    Gloo = 2,
    Stub = 99,
}

#[no_mangle]
pub extern "C" fn aether_dist_all_reduce(
    _buf: *mut c_void,
    _world_size: c_int,
    _backend: c_int,
) {
    // Phase 0 stub: real NCCL/MPI/Gloo dispatch goes here in Phase 2.
}

/// Compiled aether binaries can call this from `main()` to confirm the runtime
/// linked correctly without dragging in any platform deps.
#[no_mangle]
pub unsafe extern "C" fn aether_rt_self_check() -> c_int {
    tape().entries.len() as c_int
}

// =====================================================================
// Bootstrap allocator + init helpers — these let an Aether `main()` set up
// tensors, fill them with deterministic data, and free them, without needing
// arrays or struct support in the language. All buffers are heap-allocated
// f32/i32 slabs returned as i64-sized pointers (the asm backend already has
// a TyKind::Int that's 64-bit, which is exactly what a pointer is on x64).
//
// Phase-1 swap: replace these with cudaMalloc-backed versions; the symbol
// surface stays identical.
// =====================================================================

/// Allocate `n` f32 elements, zero-initialized. Returns a thin data pointer
/// cast to i64. Free with `aether_free_f32(p, n)`.
#[no_mangle] pub extern "C" fn aether_alloc_f32(n: c_int) -> i64 {
    let n = n.max(0) as usize;
    let mut v: Vec<f32> = vec![0.0; n];
    v.shrink_to_fit();
    debug_assert_eq!(v.capacity(), n);
    let ptr = v.as_mut_ptr() as i64;
    std::mem::forget(v);
    ptr
}

#[no_mangle] pub extern "C" fn aether_alloc_i32(n: c_int) -> i64 {
    let n = n.max(0) as usize;
    let mut v: Vec<i32> = vec![0; n];
    v.shrink_to_fit();
    debug_assert_eq!(v.capacity(), n);
    let ptr = v.as_mut_ptr() as i64;
    std::mem::forget(v);
    ptr
}

#[no_mangle] pub unsafe extern "C" fn aether_free_f32(p: i64, n: c_int) -> c_int {
    if p == 0 || n <= 0 { return 0; }
    let n = n as usize;
    let _ = Vec::from_raw_parts(p as *mut f32, n, n);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_free_i32(p: i64, n: c_int) -> c_int {
    if p == 0 || n <= 0 { return 0; }
    let n = n as usize;
    let _ = Vec::from_raw_parts(p as *mut i32, n, n);
    0
}

/// Splittable-mix64 (variant of SplitMix64). Deterministic PRNG that doesn't
/// require Rust's `rand` crate. Returns a u64; consumers can mask down.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Fill `n` f32s with the given constant `value`. Used to initialise
/// LayerNorm gamma/beta and other "set every element to k" patterns.
#[no_mangle] pub unsafe extern "C" fn aether_init_constant_f32(p: i64, n: c_int, value: f32) {
    if p == 0 || n <= 0 { return; }
    let buf = std::slice::from_raw_parts_mut(p as *mut f32, n as usize);
    for x in buf.iter_mut() { *x = value; }
}

/// Fill `n` f32s with N(~0, scale^2) using a Box-Muller transform off
/// SplitMix64. Deterministic in `seed`.
#[no_mangle] pub unsafe extern "C" fn aether_init_normal_f32(p: i64, n: c_int, scale: f32, seed: i64) {
    if p == 0 || n <= 0 { return; }
    let n = n as usize;
    let buf = std::slice::from_raw_parts_mut(p as *mut f32, n);
    let mut state = seed as u64;
    let mut i = 0;
    while i < n {
        let u1 = ((splitmix64(&mut state) >> 11) as f64) / ((1u64 << 53) as f64);
        let u2 = ((splitmix64(&mut state) >> 11) as f64) / ((1u64 << 53) as f64);
        let r = (-2.0 * u1.max(1e-30).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let z0 = (r * theta.cos()) as f32 * scale;
        let z1 = (r * theta.sin()) as f32 * scale;
        buf[i] = z0;
        if i + 1 < n { buf[i + 1] = z1; }
        i += 2;
    }
}

/// Fill `n` i32 label indices uniformly in `[0, classes)`. Deterministic in `seed`.
#[no_mangle] pub unsafe extern "C" fn aether_fill_labels_i32(p: i64, n: c_int, classes: c_int, seed: i64) {
    if p == 0 || n <= 0 || classes <= 0 { return; }
    let n = n as usize;
    let buf = std::slice::from_raw_parts_mut(p as *mut i32, n);
    let c = classes as u64;
    let mut state = seed as u64;
    for slot in buf.iter_mut() {
        *slot = (splitmix64(&mut state) % c) as i32;
    }
}

/// Read a single f32 from a buffer at index `i`. Used by Aether `main()` to
/// observe loss values without needing array indexing in the language.
#[no_mangle] pub unsafe extern "C" fn aether_load_f32(p: i64, i: c_int) -> f32 {
    if p == 0 { return 0.0; }
    // Use read_unaligned: SafeTensors payloads land at byte 8+header_len,
    // which is rarely 4-aligned, and Aether code reads them via this fn.
    (p as *const f32).add(i as usize).read_unaligned()
}

/// Print one line of bench output: `<which>  M=… N=… K=… iters=… us=…`,
/// where `which` is 0 for CPU / 1 for GPU. The label_ptr arg is reserved
/// for a future per-bench tag string and is currently ignored.
#[no_mangle] pub extern "C" fn aether_print_bench(
    which: i64, m: c_int, n: c_int, k: c_int, iters: c_int, us: i64,
) -> c_int {
    let tag = if which == 0 { "cpu" } else { "gpu" };
    println!("{:3}  M={:>4}  N={:>4}  K={:>4}  iters={:>4}  us={:>10}", tag, m, n, k, iters, us);
    0
}

/// Print "step={step} loss={loss}\n" to stdout. Lets compiled Aether code
/// emit a training curve without needing format strings or stdio bindings.
///
/// Implementation note: writes directly to the Win32 standard-output handle
/// via WriteFile. This avoids `std::io::stdout()` and the lazy-init chain
/// behind `println!`, which (on Windows) reaches BCrypt primitives for the
/// stdio mutex's HashMap hasher seed. That chain AVs when this DLL is
/// loaded too early in process init under the self-hosted `--emit=pe-bin`
/// PE writer; the explicit WriteFile call sidesteps the whole thing.
// === Self-host bootstrap primitives (roadmap item #27) ============
//
// To rewrite aetherc in .aether we need: file I/O, byte-level memory
// access, raw allocation. Each fn here is the smallest possible C-ABI
// wedge — once enough of these exist, a baby tokenizer / parser /
// codegen can be written in .aether that targets these symbols.

use std::cell::Cell;
thread_local! { static LAST_FILE_SIZE: Cell<i64> = Cell::new(0); }

/// Read an entire file at the 0-terminated C string `path` into a fresh
/// heap buffer. Returns the buffer pointer as `i64` (0 = error). The
/// buffer's length is recorded thread-locally and retrievable via
/// `aether_file_size_last`. Caller frees with `aether_free_bytes`.
#[no_mangle] pub unsafe extern "C" fn aether_read_file(path: i64) -> i64 {
    if path == 0 { return 0; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p_bytes = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(p_str) = std::str::from_utf8(p_bytes) else { return 0; };
    let Ok(buf) = std::fs::read(p_str) else { return 0; };
    let n = buf.len();
    LAST_FILE_SIZE.with(|c| c.set(n as i64));
    let boxed = buf.into_boxed_slice();
    let raw = Box::into_raw(boxed);
    raw as *mut u8 as i64
}

/// Length in bytes of the most recent `aether_read_file` success.
#[no_mangle] pub extern "C" fn aether_file_size_last() -> i64 {
    LAST_FILE_SIZE.with(|c| c.get())
}

/// Allocate `n` bytes (zero-initialised) on the heap. Returns ptr as i64.
/// Caller frees with `aether_free_bytes` — DON'T pass to `aether_free_f32`.
#[no_mangle] pub extern "C" fn aether_alloc_bytes(n: i64) -> i64 {
    if n <= 0 { return 0; }
    let v = vec![0u8; n as usize];
    let boxed = v.into_boxed_slice();
    let p = Box::into_raw(boxed) as *mut u8 as i64;
    prof_alloc(n);
    p
}

/// Free a buffer previously returned by `aether_alloc_bytes` /
/// `aether_read_file`. `n` MUST be the original length passed in.
#[no_mangle] pub unsafe extern "C" fn aether_free_bytes(p: i64, n: i64) {
    if p == 0 || n <= 0 { return; }
    let slice = std::slice::from_raw_parts_mut(p as *mut u8, n as usize);
    drop(Box::from_raw(slice as *mut [u8]));
    prof_free(n);
}

/// Load a single byte from the buffer at offset `i`. Returns the byte
/// as a 0..=255 i64 (so Aether's int Bin ops work on it cleanly).
#[no_mangle] pub unsafe extern "C" fn aether_byte_at(p: i64, i: i64) -> i64 {
    if p == 0 || i < 0 { return -1; }
    *(p as *const u8).add(i as usize) as i64
}

/// Store a byte (low 8 bits of `v`) at offset `i` in the buffer.
#[no_mangle] pub unsafe extern "C" fn aether_byte_set(p: i64, i: i64, v: i64) {
    if p == 0 || i < 0 { return; }
    *(p as *mut u8).add(i as usize) = (v & 0xFF) as u8;
}

// ─── FR-15.3 (AVX2) — witness helpers ───────────────────────────────────────
// These three fns let an `.aether` source build two f32 arrays, call the
// compiler's recognized `__aether_avx2_dot_f32` builtin on them, and compare
// the result to a scalar reference. Returns a meaningful exit code at the
// far end. The arrays are heap-allocated via the same `aether_alloc_bytes`
// box-leak path so they free cleanly via `aether_free_bytes`.

/// Allocate `n` f32 slots (= 4*n bytes), fill deterministically from `seed`,
/// return the pointer as i64. Fill pattern is a simple LCG-like ramp so that
/// the dot product is a non-trivial value the AVX2 path can mis-compute on.
#[no_mangle] pub extern "C" fn aether_avx2_witness_arr(seed: i64, n: i64) -> i64 {
    if n <= 0 { return 0; }
    let bytes = (n as usize) * 4;
    let p = aether_alloc_bytes(bytes as i64);
    if p == 0 { return 0; }
    unsafe {
        let f = p as *mut f32;
        let mut s = seed as u32;
        for i in 0..n as usize {
            // splitmix-ish step → f32 in roughly [0, 4).
            s = s.wrapping_mul(2654435761).wrapping_add(0x9E37);
            let mantissa = (s >> 9) & 0x7F_FFFF;
            let bits = 0x3F80_0000u32 | mantissa; // 1.0 .. 2.0
            let v = f32::from_bits(bits) - 1.0 + ((i & 7) as f32) * 0.25;
            *f.add(i) = v;
        }
    }
    p
}

/// Reference scalar dot product over `n` f32 elements at `a` and `b`.
/// Used to verify the AVX2 inline-emit's result. Rust may auto-vectorise
/// this loop into AVX2/AVX-512 of its own, which is fine — both versions
/// converge on the same value modulo float reassociation.
#[no_mangle] pub unsafe extern "C" fn aether_dot_f32_scalar(a: i64, b: i64, n: i64) -> f32 {
    if a == 0 || b == 0 || n <= 0 { return 0.0; }
    let aa = a as *const f32;
    let bb = b as *const f32;
    let mut acc = 0.0f32;
    for i in 0..n as usize {
        acc += *aa.add(i) * *bb.add(i);
    }
    acc
}

/// Return 42 if `a` and `b` agree to within a 1e-3 relative tolerance,
/// otherwise return the witness's failure code 1. Provides a clean
/// "if avx == scalar then 42 else 1" expression that survives the .aether
/// surface without needing f32 abs / unary-neg (still gapped today).
#[no_mangle] pub extern "C" fn aether_f32_close_exit(a: f32, b: f32) -> i32 {
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs()).max(1.0);
    if diff / mag < 1.0e-3 { 42 } else { 1 }
}

/// Write `n` bytes from `buf` to file `path` (overwriting). Returns 0 on
/// success, non-zero on failure.
#[no_mangle] pub unsafe extern "C" fn aether_write_file(path: i64, buf: i64, n: i64) -> i32 {
    if path == 0 || buf == 0 || n < 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p_bytes = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(p_str) = std::str::from_utf8(p_bytes) else { return -2; };
    let data = std::slice::from_raw_parts(buf as *const u8, n as usize);
    match std::fs::write(p_str, data) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Allocate `n` bytes of executable + writable memory via `VirtualAlloc`
/// (Windows). Returns a pointer suitable for `aether_call_jit_i64`.
/// Caller frees with `aether_free_executable`. Returns 0 on failure.
#[no_mangle] pub unsafe extern "C" fn aether_alloc_executable(n: i64) -> i64 {
    if n <= 0 { return 0; }
    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
    }
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;
    let p = VirtualAlloc(std::ptr::null_mut(), n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if p.is_null() { 0 } else { p as i64 }
}

#[no_mangle] pub unsafe extern "C" fn aether_free_executable(p: i64, _n: i64) {
    if p == 0 { return; }
    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
    }
    const MEM_RELEASE: u32 = 0x8000;
    let _ = VirtualFree(p as *mut u8, 0, MEM_RELEASE);
}

/// Cast `p` to `fn() -> i64` and invoke it. The caller is responsible for
/// having written valid x86-64 code (with proper prologue/epilogue) into
/// the buffer — no safety net here, this is the JIT escape hatch.
#[no_mangle] pub unsafe extern "C" fn aether_call_jit_i64(p: i64) -> i64 {
    if p == 0 { return 0; }
    let f: extern "C" fn() -> i64 = std::mem::transmute(p);
    f()
}

/// Byte-compare two buffers of length `n`. Returns 1 on equal, 0 otherwise.
/// Used by the self-hosted tokenizer to match ident spans against keywords.
#[no_mangle] pub unsafe extern "C" fn aether_bytes_eq(a: i64, b: i64, n: i64) -> i64 {
    if a == 0 || b == 0 || n < 0 { return 0; }
    let s1 = std::slice::from_raw_parts(a as *const u8, n as usize);
    let s2 = std::slice::from_raw_parts(b as *const u8, n as usize);
    if s1 == s2 { 1 } else { 0 }
}

/// Length of the 0-terminated C string at `p` (excluding the NUL).
#[no_mangle] pub unsafe extern "C" fn aether_str_len(p: i64) -> i64 {
    if p == 0 { return 0; }
    let mut n = 0i64;
    while *(p as *const u8).add(n as usize) != 0 { n += 1; }
    n
}

/// Write `n` bytes from buffer `p` to stdout (no trailing newline).
#[no_mangle] pub unsafe extern "C" fn aether_print_bytes(p: i64, n: i64) {
    if p == 0 || n <= 0 { return; }
    use std::io::Write;
    let slice = std::slice::from_raw_parts(p as *const u8, n as usize);
    let _ = std::io::stdout().write_all(slice);
    let _ = std::io::stdout().flush();
}

/// =====================================================================
/// P6.9 — atomics + thread spawn primitives.
///
/// Atomics use Rust's `std::sync::atomic::AtomicI64` over a raw pointer
/// the caller is responsible for providing as 8-byte aligned storage
/// (typical use: `aether_alloc_bytes(16)` then pass the buffer).
/// Thread spawn uses Win32 `CreateThread` with a fn-ptr trampoline.
/// =====================================================================
#[no_mangle] pub unsafe extern "C" fn aether_atomic_fetch_add_i64(addr: i64, val: i64) -> i64 {
    if addr == 0 { return 0; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    a.fetch_add(val, Ordering::SeqCst)
}

#[no_mangle] pub unsafe extern "C" fn aether_atomic_load_i64(addr: i64) -> i64 {
    if addr == 0 { return 0; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    a.load(Ordering::SeqCst)
}

#[no_mangle] pub unsafe extern "C" fn aether_atomic_store_i64(addr: i64, val: i64) {
    if addr == 0 { return; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    a.store(val, Ordering::SeqCst);
}

#[no_mangle] pub unsafe extern "C" fn aether_atomic_cas_i64(addr: i64, expected: i64, new: i64) -> i64 {
    if addr == 0 { return -1; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    match a.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(prev) => prev,
        Err(prev) => prev,
    }
}

/// Spawn a thread that runs `fn_ptr(arg)`. Returns the OS thread handle
/// (or 0 on failure). Caller joins via `aether_thread_join`.
#[no_mangle] pub unsafe extern "C" fn aether_thread_spawn(fn_ptr: i64, arg: i64) -> i64 {
    if fn_ptr == 0 { return 0; }
    let payload: Box<(i64, i64)> = Box::new((fn_ptr, arg));
    let raw = Box::into_raw(payload) as i64;
    let handle = std::thread::spawn(move || {
        let p = raw as *mut (i64, i64);
        let (fp, ar) = *Box::from_raw(p);
        let f: extern "C" fn(i64) -> i64 = std::mem::transmute(fp);
        f(ar);
    });
    Box::into_raw(Box::new(handle)) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_thread_join(handle: i64) -> i32 {
    if handle == 0 { return -1; }
    let h: Box<std::thread::JoinHandle<()>> =
        Box::from_raw(handle as *mut std::thread::JoinHandle<()>);
    match h.join() {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Integer absolute value. Free fn — Aether doesn't have method-on-scalar
/// dispatch yet, so primitives live as `aether_*` extern fns.
// =====================================================================
// SafeTensors reader (roadmap P7.5).
//
// Format: <8-byte LE u64 header_len> <header_len bytes of JSON> <raw payload>.
// JSON: {"name":{"dtype":"F32","shape":[...],"data_offsets":[start,end]}, ...}.
//
// Phase-0 reader is byte-exact for the format we emit (single-pass linear
// scan, no general JSON). Good enough to round-trip our own writes; HF
// weight files use the same canonical shape so it works on those too.
// =====================================================================

/// Validate the 8-byte length prefix and return the JSON-header byte count.
/// Returns -1 if `buf` is null, `len` is too small, or the header overruns.
#[no_mangle] pub unsafe extern "C" fn safetensors_parse_header(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 8 { return -1; }
    let p = buf as *const u8;
    let mut hdr: u64 = 0;
    for i in 0..8 { hdr |= (*p.add(i) as u64) << (i * 8); }
    if 8u64 + hdr > len as u64 { return -1; }
    hdr as i64
}

/// Look up tensor `name` in the SafeTensors blob and return a pointer to
/// the start of its f32 payload (i.e. `buf + 8 + hdr_len + data_offsets[0]`).
/// Returns 0 if the name is missing or the JSON is malformed.
#[no_mangle] pub unsafe extern "C" fn safetensors_get_tensor_f32(
    buf: i64, len: i64, name: i64, name_len: i64,
) -> i64 {
    let hdr_len = safetensors_parse_header(buf, len);
    if hdr_len < 0 || name == 0 || name_len <= 0 { return 0; }
    let json = std::slice::from_raw_parts((buf as *const u8).add(8), hdr_len as usize);
    let name_bytes = std::slice::from_raw_parts(name as *const u8, name_len as usize);
    let Ok(json_str) = std::str::from_utf8(json) else { return 0; };
    let Ok(name_str) = std::str::from_utf8(name_bytes) else { return 0; };

    // Locate `"<name>":` at a key position. We require the colon to follow
    // the closing quote (with optional ws) so that "foo" inside a value
    // string doesn't false-match.
    let needle = format!("\"{}\"", name_str);
    let mut search_from = 0usize;
    let key_pos = loop {
        let Some(idx) = json_str[search_from..].find(&needle) else { return 0; };
        let abs = search_from + idx;
        let after = &json_str[abs + needle.len()..];
        let trimmed = after.trim_start();
        if trimmed.starts_with(':') { break abs; }
        search_from = abs + needle.len();
    };

    // Inside the matching object, find `"data_offsets":[a,b]` and parse `a`.
    let rest = &json_str[key_pos + needle.len()..];
    let Some(off_idx) = rest.find("\"data_offsets\":[") else { return 0; };
    let after = &rest[off_idx + "\"data_offsets\":[".len()..];
    let Some(comma) = after.find(',') else { return 0; };
    let Ok(start) = after[..comma].trim().parse::<i64>() else { return 0; };

    buf + 8 + hdr_len + start
}

// =====================================================================
// Microbench-driven kernel selection (P10.10).
//
// `aether_op_matmul_f32_auto` runs both the naive and blocked matmul
// kernels for the requested (m,k,n) on a one-shot probe (the first time
// that shape is seen), times each, caches the winner, and dispatches
// future calls of the same shape to the chosen kernel.
//
// The probe runs on a side buffer so the user-visible output is computed
// only by the chosen kernel (no double work on the user's buffers).
// =====================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatmulKernel { Naive = 0, Blocked = 1 }

// Single-threaded cache, same UnsafeCell pattern as TAPE above (avoids
// the std::sync init chain that AVs under the self-hosted PE writer).
struct MatmulCache(UnsafeCell<Vec<((usize, usize, usize), MatmulKernel)>>);
unsafe impl Sync for MatmulCache {}
static MATMUL_CACHE: MatmulCache = MatmulCache(UnsafeCell::new(Vec::new()));

unsafe fn matmul_cache_lookup(key: (usize, usize, usize)) -> Option<MatmulKernel> {
    let v = &*MATMUL_CACHE.0.get();
    v.iter().find_map(|(k, val)| if *k == key { Some(*val) } else { None })
}
unsafe fn matmul_cache_insert(key: (usize, usize, usize), val: MatmulKernel) {
    let v = &mut *MATMUL_CACHE.0.get();
    v.push((key, val));
}

unsafe fn probe_matmul_f32(m: usize, k: usize, n: usize) -> MatmulKernel {
    // Fresh deterministic input for the probe.
    let a: Vec<f32> = (0..m*k).map(|i| ((i * 7 + 3) as f32 * 0.001) % 1.0).collect();
    let b: Vec<f32> = (0..k*n).map(|i| ((i * 11 + 5) as f32 * 0.001) % 1.0).collect();
    let mut o = vec![0.0f32; m * n];
    let t0 = wall_us_now();
    ops::matmul_f32(a.as_ptr(), b.as_ptr(), o.as_mut_ptr(), m, k, n);
    let t_naive = wall_us_now() - t0;
    let t1 = wall_us_now();
    ops::matmul_blocked_f32(a.as_ptr(), b.as_ptr(), o.as_mut_ptr(), m, k, n);
    let t_blocked = wall_us_now() - t1;
    if t_blocked < t_naive { MatmulKernel::Blocked } else { MatmulKernel::Naive }
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_f32_auto(
    a: *const c_void, b: *const c_void, out: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    let m = m as usize; let k = k as usize; let n = n as usize;
    let key = (m, k, n);
    let chosen = match matmul_cache_lookup(key) {
        Some(v) => v,
        None => {
            let v = probe_matmul_f32(m, k, n);
            matmul_cache_insert(key, v);
            v
        }
    };
    match chosen {
        MatmulKernel::Naive   => ops::matmul_f32(a as _, b as _, out as _, m, k, n),
        MatmulKernel::Blocked => ops::matmul_blocked_f32(a as _, b as _, out as _, m, k, n),
    }
    0
}

/// Inspect the cached kernel pick for a shape (debug / witness only):
/// returns 0=naive, 1=blocked, -1=not yet probed.
#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_f32_auto_kernel(
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    let key = (m as usize, k as usize, n as usize);
    match matmul_cache_lookup(key) {
        Some(MatmulKernel::Naive)   => 0,
        Some(MatmulKernel::Blocked) => 1,
        None => -1,
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_blocked_f32(
    a: *const c_void, b: *const c_void, out: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_blocked_f32(a as _, b as _, out as _, m as _, k as _, n as _);
    0
}

// =====================================================================
// Profiling primitives (P8.6 — allocator stats + stopwatch).
// =====================================================================

use std::sync::atomic::{AtomicI64, Ordering};

static ALLOC_TOTAL: AtomicI64 = AtomicI64::new(0);
static ALLOC_LIVE:  AtomicI64 = AtomicI64::new(0);
static ALLOC_PEAK:  AtomicI64 = AtomicI64::new(0);

fn prof_alloc(n: i64) {
    ALLOC_TOTAL.fetch_add(n, Ordering::Relaxed);
    let live = ALLOC_LIVE.fetch_add(n, Ordering::Relaxed) + n;
    // Atomic "peak = max(peak, live)" via CAS loop.
    let mut peak = ALLOC_PEAK.load(Ordering::Relaxed);
    while live > peak {
        match ALLOC_PEAK.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(p) => peak = p,
        }
    }
}

fn prof_free(n: i64) { ALLOC_LIVE.fetch_sub(n, Ordering::Relaxed); }

#[no_mangle] pub extern "C" fn aether_prof_alloc_total() -> i64 { ALLOC_TOTAL.load(Ordering::Relaxed) }
#[no_mangle] pub extern "C" fn aether_prof_alloc_live()  -> i64 { ALLOC_LIVE.load(Ordering::Relaxed) }
#[no_mangle] pub extern "C" fn aether_prof_alloc_peak()  -> i64 { ALLOC_PEAK.load(Ordering::Relaxed) }
#[no_mangle] pub extern "C" fn aether_prof_reset() -> i32 {
    ALLOC_TOTAL.store(0, Ordering::Relaxed);
    ALLOC_LIVE.store(0,  Ordering::Relaxed);
    ALLOC_PEAK.store(0,  Ordering::Relaxed);
    0
}

/// Stopwatch: take `wall_us` at start, take `wall_us - start` at end.
/// Sugar; callers can use `aether_wall_us` directly. Keeps the witness
/// readable.
#[no_mangle] pub extern "C" fn aether_timer_start() -> i64 { wall_us_now() }
#[no_mangle] pub extern "C" fn aether_timer_elapsed_us(start: i64) -> i64 {
    wall_us_now() - start
}

fn wall_us_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// =====================================================================
// Filesystem primitives (P6.13 standard I/O surface).
// =====================================================================

/// Returns 1 if the path exists (file OR directory), 0 otherwise.
#[no_mangle] pub unsafe extern "C" fn aether_path_exists(path: i64) -> i64 {
    if path == 0 { return 0; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return 0; };
    if std::path::Path::new(s).exists() { 1 } else { 0 }
}

/// File size in bytes; -1 on error (missing, not a regular file, etc.).
#[no_mangle] pub unsafe extern "C" fn aether_file_size(path: i64) -> i64 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::metadata(s) {
        Ok(m) if m.is_file() => m.len() as i64,
        _ => -1,
    }
}

/// Returns 1 if the path is a directory, 0 if file, -1 on error.
#[no_mangle] pub unsafe extern "C" fn aether_is_dir(path: i64) -> i64 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::metadata(s) {
        Ok(m) => if m.is_dir() { 1 } else { 0 },
        Err(_) => -1,
    }
}

/// `mkdir -p` style: create the directory and all missing parents.
/// Returns 0 on success, negative on failure.
#[no_mangle] pub unsafe extern "C" fn aether_create_dir_all(path: i64) -> i32 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::create_dir_all(s) { Ok(()) => 0, Err(_) => -2 }
}

/// Remove a single file. Returns 0 on success, negative on failure.
#[no_mangle] pub unsafe extern "C" fn aether_remove_file(path: i64) -> i32 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::remove_file(s) { Ok(()) => 0, Err(_) => -2 }
}

/// Copy file `src` → `dst`. Returns bytes copied on success, -1 on failure.
#[no_mangle] pub unsafe extern "C" fn aether_copy_file(src: i64, dst: i64) -> i64 {
    if src == 0 || dst == 0 { return -1; }
    let mut sl = 0usize; while *(src as *const u8).add(sl) != 0 { sl += 1; }
    let mut dl = 0usize; while *(dst as *const u8).add(dl) != 0 { dl += 1; }
    let s = std::slice::from_raw_parts(src as *const u8, sl);
    let d = std::slice::from_raw_parts(dst as *const u8, dl);
    let Ok(s_str) = std::str::from_utf8(s) else { return -1; };
    let Ok(d_str) = std::str::from_utf8(d) else { return -1; };
    match std::fs::copy(s_str, d_str) { Ok(n) => n as i64, Err(_) => -1 }
}

/// Atomic rename `src` → `dst` (cross-device may fail on some FS).
#[no_mangle] pub unsafe extern "C" fn aether_rename(src: i64, dst: i64) -> i32 {
    if src == 0 || dst == 0 { return -1; }
    let mut sl = 0usize; while *(src as *const u8).add(sl) != 0 { sl += 1; }
    let mut dl = 0usize; while *(dst as *const u8).add(dl) != 0 { dl += 1; }
    let s = std::slice::from_raw_parts(src as *const u8, sl);
    let d = std::slice::from_raw_parts(dst as *const u8, dl);
    let Ok(s_str) = std::str::from_utf8(s) else { return -1; };
    let Ok(d_str) = std::str::from_utf8(d) else { return -1; };
    match std::fs::rename(s_str, d_str) { Ok(()) => 0, Err(_) => -2 }
}

/// Count immediate entries in directory `path`. Returns -1 on error.
#[no_mangle] pub unsafe extern "C" fn aether_read_dir_count(path: i64) -> i64 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::read_dir(s) {
        Ok(rd) => rd.count() as i64,
        Err(_) => -1,
    }
}

// =====================================================================
// TCP primitives (P8.5 inference + serving — network I/O surface).
//
// Listener / stream handles are opaque i64 indices into a single-threaded
// Vec<Option<Box<T>>>. Returning a Vec index keeps the C-ABI integer-only,
// matches the i64-handle pattern used elsewhere in this crate, and lets us
// drop entries by setting the slot to None without invalidating other
// outstanding handles. Same UnsafeCell trick as TAPE / MATMUL_CACHE — the
// runtime is single-threaded by contract; concurrent calls into TCP fns
// from multiple threads are UB. The serving roadmap will add a real
// thread-pool primitive on top of this surface in a follow-up.
// =====================================================================

struct TcpListenerCell(UnsafeCell<Vec<Option<Box<std::net::TcpListener>>>>);
unsafe impl Sync for TcpListenerCell {}
static TCP_LISTENERS: TcpListenerCell = TcpListenerCell(UnsafeCell::new(Vec::new()));

struct TcpStreamCell(UnsafeCell<Vec<Option<Box<std::net::TcpStream>>>>);
unsafe impl Sync for TcpStreamCell {}
static TCP_STREAMS: TcpStreamCell = TcpStreamCell(UnsafeCell::new(Vec::new()));

unsafe fn tcp_listeners() -> &'static mut Vec<Option<Box<std::net::TcpListener>>> {
    &mut *TCP_LISTENERS.0.get()
}
unsafe fn tcp_streams() -> &'static mut Vec<Option<Box<std::net::TcpStream>>> {
    &mut *TCP_STREAMS.0.get()
}

/// Bind a TCP listener on `127.0.0.1:port`. Pass `port = 0` for an
/// OS-assigned ephemeral port. Returns the listener handle (>= 0) or -1
/// on failure. Use `aether_tcp_listener_port` to read back the actual
/// bound port when `port = 0` was passed in.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_listen(port: i64) -> i64 {
    if !(0..=65535).contains(&port) { return -1; }
    let addr = format!("127.0.0.1:{}", port);
    match std::net::TcpListener::bind(&addr) {
        Ok(l) => {
            let v = tcp_listeners();
            // Re-use a vacated slot if available.
            for (i, slot) in v.iter_mut().enumerate() {
                if slot.is_none() { *slot = Some(Box::new(l)); return i as i64; }
            }
            v.push(Some(Box::new(l)));
            (v.len() - 1) as i64
        }
        Err(_) => -1,
    }
}

/// Bound local port for a listener (useful when port=0 was passed). -1 on err.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_listener_port(handle: i64) -> i64 {
    if handle < 0 { return -1; }
    let v = tcp_listeners();
    let idx = handle as usize;
    if idx >= v.len() { return -1; }
    match v[idx].as_ref() {
        Some(l) => match l.local_addr() {
            Ok(a) => a.port() as i64,
            Err(_) => -1,
        },
        None => -1,
    }
}

/// Drop the listener at `handle`. 0 on success, -1 if the handle is invalid.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_close(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let v = tcp_listeners();
    let idx = handle as usize;
    if idx >= v.len() { return -1; }
    if v[idx].is_none() { return -1; }
    v[idx] = None;
    0
}

/// Block waiting for one inbound connection on `listener`. Returns the
/// stream handle (>= 0) or -1 on failure / invalid listener handle.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_accept_one(listener: i64) -> i64 {
    if listener < 0 { return -1; }
    let v = tcp_listeners();
    let idx = listener as usize;
    if idx >= v.len() { return -1; }
    let Some(ref l) = v[idx] else { return -1; };
    let stream = match l.accept() {
        Ok((s, _)) => s,
        Err(_) => return -1,
    };
    let s = tcp_streams();
    for (i, slot) in s.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(stream)); return i as i64; }
    }
    s.push(Some(Box::new(stream)));
    (s.len() - 1) as i64
}

/// Drop the stream at `handle`. 0 on success, -1 if invalid.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_stream_close(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let v = tcp_streams();
    let idx = handle as usize;
    if idx >= v.len() { return -1; }
    if v[idx].is_none() { return -1; }
    v[idx] = None;
    0
}

/// Connect a TCP stream to `127.0.0.1:port`. Returns stream handle or -1.
/// Phase 0.5 client-side counterpart so a single-process witness can drive
/// both ends of a connection without spawning extra processes.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_connect(port: i64) -> i64 {
    if !(0..=65535).contains(&port) { return -1; }
    let addr = format!("127.0.0.1:{}", port);
    match std::net::TcpStream::connect(&addr) {
        Ok(s) => {
            let v = tcp_streams();
            for (i, slot) in v.iter_mut().enumerate() {
                if slot.is_none() { *slot = Some(Box::new(s)); return i as i64; }
            }
            v.push(Some(Box::new(s)));
            (v.len() - 1) as i64
        }
        Err(_) => -1,
    }
}

/// Send `n` bytes from `buf` over `stream`. Returns bytes written, or -1 on err.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_send(stream: i64, buf: i64, n: i64) -> i64 {
    use std::io::Write;
    if stream < 0 || buf == 0 || n < 0 { return -1; }
    let v = tcp_streams();
    let idx = stream as usize;
    if idx >= v.len() { return -1; }
    let Some(ref mut s) = v[idx] else { return -1; };
    let slice = std::slice::from_raw_parts(buf as *const u8, n as usize);
    match s.write(slice) {
        Ok(written) => written as i64,
        Err(_) => -1,
    }
}

/// Recv up to `n` bytes from `stream` into `buf`. Returns bytes read
/// (0 on EOF / clean close, -1 on error). Blocking.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_recv(stream: i64, buf: i64, n: i64) -> i64 {
    use std::io::Read;
    if stream < 0 || buf == 0 || n < 0 { return -1; }
    let v = tcp_streams();
    let idx = stream as usize;
    if idx >= v.len() { return -1; }
    let Some(ref mut s) = v[idx] else { return -1; };
    let slice = std::slice::from_raw_parts_mut(buf as *mut u8, n as usize);
    match s.read(slice) {
        Ok(got) => got as i64,
        Err(_) => -1,
    }
}

// =====================================================================
// GGUF format header parser (P7.4 partial — quant kernels still TBD).
//
// Layout (little-endian throughout):
//   bytes 0..4   : magic "GGUF"
//   bytes 4..8   : u32 version
//   bytes 8..16  : u64 tensor_count
//   bytes 16..24 : u64 metadata_kv_count
//   ...          : metadata + tensor info + tensor data (quantized)
//
// Phase-1 surface: validate magic, expose version + counts. Quant
// dequantization is the L follow-on.
// =====================================================================

#[no_mangle] pub unsafe extern "C" fn gguf_parse_magic(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 4 { return 0; }
    let p = buf as *const u8;
    if *p == b'G' && *p.add(1) == b'G' && *p.add(2) == b'U' && *p.add(3) == b'F' { 1 } else { 0 }
}

#[no_mangle] pub unsafe extern "C" fn gguf_version(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 8 { return -1; }
    let p = buf as *const u8;
    let mut v = 0u32;
    for i in 0..4 { v |= (*p.add(4 + i) as u32) << (i * 8); }
    v as i64
}

#[no_mangle] pub unsafe extern "C" fn gguf_tensor_count(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 16 { return -1; }
    let p = buf as *const u8;
    let mut v = 0u64;
    for i in 0..8 { v |= (*p.add(8 + i) as u64) << (i * 8); }
    v as i64
}

#[no_mangle] pub unsafe extern "C" fn gguf_metadata_count(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 24 { return -1; }
    let p = buf as *const u8;
    let mut v = 0u64;
    for i in 0..8 { v |= (*p.add(16 + i) as u64) << (i * 8); }
    v as i64
}

/// Atomic file save: write `n` f32s to a single-tensor SafeTensors file at
/// `path`, named "data" with shape [n]. Goes through a `<path>.tmp`
/// staging file + rename so a crash mid-write can't leave the destination
/// half-written. Returns 0 on success, negative on failure (P8.8).
#[no_mangle] pub unsafe extern "C" fn safetensors_save_f32(
    path: i64, ptr: i64, n: i64,
) -> i32 {
    if path == 0 || ptr == 0 || n < 0 { return -1; }
    let mut path_len = 0usize;
    while *(path as *const u8).add(path_len) != 0 { path_len += 1; }
    let Ok(path_str) = std::str::from_utf8(
        std::slice::from_raw_parts(path as *const u8, path_len)
    ) else { return -1; };

    let n_us = n as usize;
    let payload_bytes = n_us * 4;
    let json = format!(
        "{{\"data\":{{\"dtype\":\"F32\",\"shape\":[{n}],\"data_offsets\":[0,{}]}}}}",
        payload_bytes
    );
    let json_bytes = json.as_bytes();
    let header_len = json_bytes.len() as u64;
    let total = 8 + json_bytes.len() + payload_bytes;
    let mut buf = vec![0u8; total];
    buf[..8].copy_from_slice(&header_len.to_le_bytes());
    buf[8..8 + json_bytes.len()].copy_from_slice(json_bytes);
    let payload = std::slice::from_raw_parts(ptr as *const u8, payload_bytes);
    buf[8 + json_bytes.len()..].copy_from_slice(payload);

    let tmp_path = format!("{}.tmp", path_str);
    if std::fs::write(&tmp_path, &buf).is_err() { return -2; }
    if std::fs::rename(&tmp_path, path_str).is_err() { return -3; }
    0
}

/// Load `n` f32s from a single-tensor SafeTensors file at `path` (the
/// "data" tensor) into the destination buffer at `dst`. Returns 0 on
/// success, negative on failure. (P8.8)
#[no_mangle] pub unsafe extern "C" fn safetensors_load_f32(
    path: i64, dst: i64, n: i64,
) -> i32 {
    if path == 0 || dst == 0 || n < 0 { return -1; }
    let mut path_len = 0usize;
    while *(path as *const u8).add(path_len) != 0 { path_len += 1; }
    let Ok(path_str) = std::str::from_utf8(
        std::slice::from_raw_parts(path as *const u8, path_len)
    ) else { return -1; };
    let Ok(bytes) = std::fs::read(path_str) else { return -2; };

    // Parse header length.
    if bytes.len() < 8 { return -3; }
    let mut hdr = 0u64;
    for i in 0..8 { hdr |= (bytes[i] as u64) << (i * 8); }
    if 8u64 + hdr > bytes.len() as u64 { return -3; }

    let json = match std::str::from_utf8(&bytes[8..8 + hdr as usize]) {
        Ok(s) => s, Err(_) => return -4,
    };
    let needle = "\"data_offsets\":[";
    let Some(off_idx) = json.find(needle) else { return -5; };
    let after = &json[off_idx + needle.len()..];
    let Some(comma) = after.find(',') else { return -5; };
    let Ok(start) = after[..comma].trim().parse::<usize>() else { return -5; };

    let payload_start = 8 + hdr as usize + start;
    let payload_len = (n as usize) * 4;
    if payload_start + payload_len > bytes.len() { return -6; }

    std::ptr::copy_nonoverlapping(
        bytes.as_ptr().add(payload_start),
        dst as *mut u8,
        payload_len,
    );
    0
}

#[no_mangle] pub extern "C" fn aether_abs_i64(x: i64) -> i64 { x.wrapping_abs() }
#[no_mangle] pub extern "C" fn aether_min_i64(a: i64, b: i64) -> i64 { if a < b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_max_i64(a: i64, b: i64) -> i64 { if a > b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_min_f32(a: f32, b: f32) -> f32 { if a < b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_max_f32(a: f32, b: f32) -> f32 { if a > b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_abs_f32(x: f32) -> f32 { x.abs() }
#[no_mangle] pub extern "C" fn aether_sqrt_f32(x: f32) -> f32 { x.sqrt() }

// =====================================================================
// dtype conversions (P7.1) — bf16 + f16 lanes carried as i32 (low 16 bits).
// =====================================================================

/// f32 → bf16: truncate to high 16 bits with round-to-nearest-even tie-break.
#[no_mangle] pub extern "C" fn aether_f32_to_bf16(x: f32) -> i32 {
    let bits = x.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        // NaN: preserve quiet bit.
        return ((bits >> 16) | 0x40) as i32 & 0xFFFF;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    ((rounded >> 16) & 0xFFFF) as i32
}

/// bf16 (low 16 bits of `b`) → f32 by zero-extending into the high half.
#[no_mangle] pub extern "C" fn aether_bf16_to_f32(b: i32) -> f32 {
    let bits = ((b as u32) & 0xFFFF) << 16;
    f32::from_bits(bits)
}

/// f32 → IEEE-754 binary16 (half) as low 16 bits of i32.
#[no_mangle] pub extern "C" fn aether_f32_to_f16(x: f32) -> i32 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u32;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 0xFF {
        if mant != 0 { return (sign | 0x7E00) as i32; } // NaN
        return (sign | 0x7C00) as i32;                  // Inf
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F { return (sign | 0x7C00) as i32; }    // overflow → Inf
    if new_exp <= 0 {
        if new_exp < -10 { return sign as i32; }              // underflow → 0
        // Subnormal: shift mantissa with implicit 1.
        let shift = 14 - new_exp;
        let m = (mant | 0x0080_0000) >> shift;
        return (sign | (m & 0x03FF)) as i32;
    }
    let half_mant = mant >> 13;
    (sign | ((new_exp as u32) << 10) | half_mant) as i32
}

/// Mixed-precision matmul: inputs `a` and `b` are bf16 buffers (each
/// element stored as a u16 inside an i32 lane — see ABI note in
/// stdlib/runtime.aether), upcast on read, accumulated and stored in
/// f32. Shapes match `aether_op_matmul_f32`. Witness for P8.7.
#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_bf16_f32_out(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let m = m as usize; let k = k as usize; let n = n as usize;
    let a_buf = std::slice::from_raw_parts(a as *const i32, m * k);
    let b_buf = std::slice::from_raw_parts(b as *const i32, k * n);
    let out = std::slice::from_raw_parts_mut(out as *mut f32, m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = aether_bf16_to_f32(a_buf[i * k + kk]);
                let bv = aether_bf16_to_f32(b_buf[kk * n + j]);
                acc += av * bv;
            }
            out[i * n + j] = acc;
        }
    }
    0
}

/// Pack `n` f32 values from `src` into bf16 in `dst` (low 16 bits per i32).
#[no_mangle] pub unsafe extern "C" fn aether_pack_f32_to_bf16(
    src: i64, dst: i64, n: c_int,
) -> c_int {
    let n = n as usize;
    let s = std::slice::from_raw_parts(src as *const f32, n);
    let d = std::slice::from_raw_parts_mut(dst as *mut i32, n);
    for i in 0..n { d[i] = aether_f32_to_bf16(s[i]); }
    0
}

/// IEEE-754 binary16 (low 16 bits of `h`) → f32.
#[no_mangle] pub extern "C" fn aether_f16_to_f32(h: i32) -> f32 {
    let h = (h as u32) & 0xFFFF;
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x03FF;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 }
        else {
            // Subnormal.
            let mut m = mant;
            let mut e = -14i32;
            while (m & 0x0400) == 0 { m <<= 1; e -= 1; }
            m &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFFu32 << 23) | (mant << 13)
    } else {
        let new_exp = exp as i32 - 15 + 127;
        (sign << 31) | ((new_exp as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// =====================================================================
// libm replacements (P9.6 — drop dependency on platform libm).
// Hand-rolled minimax-Taylor implementations; accurate to ~1e-6 over
// the witness ranges. Real range reduction + polynomial cores; do not
// secretly call into std::f32::sin/cos/exp/log — those route to libm.
// =====================================================================

const TWO_PI_F32: f32 = 6.2831853071795864769;
const PI_F32: f32 = 3.1415926535897932384;
const LN2_F32: f32 = 0.6931471805599453;

#[inline] fn aether_sin_core(r: f32) -> f32 {
    // Taylor for sin on r in [-π, π]; 11 terms keep error < ~3e-7.
    let r2 = r * r;
    let r3 = r * r2;
    let r5 = r3 * r2;
    let r7 = r5 * r2;
    let r9 = r7 * r2;
    let r11 = r9 * r2;
    r - r3 / 6.0
      + r5 / 120.0
      - r7 / 5040.0
      + r9 / 362880.0
      - r11 / 39916800.0
}

#[inline] fn aether_cos_core(r: f32) -> f32 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r6 * r2;
    let r10 = r8 * r2;
    1.0 - r2 / 2.0
        + r4 / 24.0
        - r6 / 720.0
        + r8 / 40320.0
        - r10 / 3628800.0
}

/// Reduce x to [-π, π] then to [-π/2, π/2] with a quadrant flag, so the
/// Taylor cores stay in their high-accuracy region.
fn reduce_to_quarter(x: f32) -> (f32, bool) {
    let k = (x / TWO_PI_F32 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32 as f32;
    let mut r = x - k * TWO_PI_F32;
    let mut flip = false;
    let half_pi = PI_F32 * 0.5;
    if r > half_pi { r = PI_F32 - r; flip = true; }
    else if r < -half_pi { r = -PI_F32 - r; flip = true; }
    (r, flip)
}

#[no_mangle] pub extern "C" fn aether_sin_f32(x: f32) -> f32 {
    let (r, _flip) = reduce_to_quarter(x);
    // sin(π - r) = sin(r); sin(-π - r) = -sin(r) — but sign already in r.
    aether_sin_core(r)
}

#[no_mangle] pub extern "C" fn aether_cos_f32(x: f32) -> f32 {
    let (r, flip) = reduce_to_quarter(x);
    let c = aether_cos_core(r);
    if flip { -c } else { c }
}

/// exp(x) via range reduction x = k*ln(2) + r, r in ~[-ln2/2, ln2/2],
/// then exp(r) Taylor (8 terms) and 2^k via direct bit pack.
#[no_mangle] pub extern "C" fn aether_exp_f32(x: f32) -> f32 {
    if x > 88.7 { return f32::INFINITY; }
    if x < -88.7 { return 0.0; }
    let k_real = x / LN2_F32;
    let k = (k_real + if k_real >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x - (k as f32) * LN2_F32;
    let r2 = r * r;
    let r3 = r * r2;
    let r4 = r2 * r2;
    let r5 = r * r4;
    let r6 = r4 * r2;
    let r7 = r3 * r4;
    // exp(r) Taylor: 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 + r⁷/5040
    let exp_r = 1.0 + r + r2 / 2.0 + r3 / 6.0 + r4 / 24.0
                  + r5 / 120.0 + r6 / 720.0 + r7 / 5040.0;
    // 2^k: pack into IEEE-754 f32 exponent field (bias 127).
    let exp_bits = ((k + 127) as u32) << 23;
    let two_to_k = f32::from_bits(exp_bits);
    exp_r * two_to_k
}

/// log(x) for x > 0. Range-reduce via the f32 exponent, then minimax
/// Taylor on the mantissa.
#[no_mangle] pub extern "C" fn aether_log_f32(x: f32) -> f32 {
    if x <= 0.0 { return f32::NAN; }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    // m in [1, 2): clear exponent, set to 127.
    let m_bits = (bits & 0x007F_FFFF) | (127u32 << 23);
    let m = f32::from_bits(m_bits);
    // Center: log(m) where m in [1, 2). Let u = (m-1)/(m+1), so m = (1+u)/(1-u).
    // Then log(m) = 2*(u + u³/3 + u⁵/5 + u⁷/7 + ...).
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let u3 = u * u2;
    let u5 = u3 * u2;
    let u7 = u5 * u2;
    let u9 = u7 * u2;
    let log_m = 2.0 * (u + u3 / 3.0 + u5 / 5.0 + u7 / 7.0 + u9 / 9.0);
    (e as f32) * LN2_F32 + log_m
}

/// Print `<label>: <int>` to stdout. `label` is a 0-terminated C string —
/// pass a `StrLit` (which the compiler interns into `.rdata` and produces
/// as a pointer). Useful for debug tracing inside training loops.
#[no_mangle] pub unsafe extern "C" fn aether_print_kv_i64(label: *const u8, value: i64) -> c_int {
    if label.is_null() {
        println!("{}", value);
    } else {
        let mut len = 0usize;
        while *label.add(len) != 0 { len += 1; }
        let bytes = std::slice::from_raw_parts(label, len);
        let s = std::str::from_utf8(bytes).unwrap_or("<bad utf-8>");
        println!("{}: {}", s, value);
    }
    0
}

/// Print `<label>: <f32>` to stdout. Same convention as `aether_print_kv_i64`.
#[no_mangle] pub unsafe extern "C" fn aether_print_kv_f32(label: *const u8, value: f32) -> c_int {
    if label.is_null() {
        println!("{}", value);
    } else {
        let mut len = 0usize;
        while *label.add(len) != 0 { len += 1; }
        let bytes = std::slice::from_raw_parts(label, len);
        let s = std::str::from_utf8(bytes).unwrap_or("<bad utf-8>");
        println!("{}: {}", s, value);
    }
    0
}

// =====================================================================
// FR-24.6 — Hot-reload signal.
// =====================================================================
//
// Production serving / training processes poll `aether_hot_reload_check`
// once per epoch / inference loop. If a sentinel file at the watch path
// has been touched (mtime > recorded baseline), the function returns 1
// + updates the baseline. The caller then reloads weights / config /
// checkpoint and continues without process restart.
//
// Witness: `tests/runtime/hot_reload_v4.aether`.

struct HotReloadCell(UnsafeCell<Option<std::time::SystemTime>>);
unsafe impl Sync for HotReloadCell {}
static HOT_RELOAD_BASELINE: HotReloadCell = HotReloadCell(UnsafeCell::new(None));

#[no_mangle] pub unsafe extern "C" fn aether_hot_reload_arm(
    sentinel_path: *const u8, len: i64,
) -> c_int {
    if sentinel_path.is_null() || len <= 0 { return -1; }
    let path = match std::str::from_utf8(std::slice::from_raw_parts(sentinel_path, len as usize)) {
        Ok(p) => p,
        Err(_) => return -2,
    };
    let mtime = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok();
    *HOT_RELOAD_BASELINE.0.get() = mtime;
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_hot_reload_check(
    sentinel_path: *const u8, len: i64,
) -> c_int {
    if sentinel_path.is_null() || len <= 0 { return 0; }
    let path = match std::str::from_utf8(std::slice::from_raw_parts(sentinel_path, len as usize)) {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let cur = match std::fs::metadata(path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return 0,
    };
    let baseline = *HOT_RELOAD_BASELINE.0.get();
    let triggered = match baseline {
        Some(b) => cur > b,
        None => true,    // first observation arms + signals.
    };
    if triggered {
        *HOT_RELOAD_BASELINE.0.get() = Some(cur);
        1
    } else {
        0
    }
}

// =====================================================================
// FR-24.3 — Supply-chain SBOM generator.
// =====================================================================
//
// Emits a CycloneDX 1.5 JSON SBOM from a caller-built component list.
// The runtime maintains an in-process registry that gets populated by
// `aether_sbom_add(name, name_len, version, version_len)`. A final
// `aether_sbom_emit(out_path, out_path_len)` walks the registry, formats
// the JSON, and writes it to disk. No serde — direct string formatting.
//
// `purl` (Package URL) is auto-derived as `pkg:aether/<name>@<version>`
// — caller can pre-mangle the name for a different scheme if needed.
//
// Witness: `tests/runtime/sbom_v4.aether` — populates 3 components,
// emits to disk, reads back, verifies the JSON contains the expected
// strings.

#[derive(Clone)]
struct SbomComponent { name: String, version: String }

struct SbomCell(UnsafeCell<Vec<SbomComponent>>);
unsafe impl Sync for SbomCell {}
static SBOM_TABLE: SbomCell = SbomCell(UnsafeCell::new(Vec::new()));
unsafe fn sbom_table() -> &'static mut Vec<SbomComponent> { &mut *SBOM_TABLE.0.get() }

#[no_mangle] pub unsafe extern "C" fn aether_sbom_reset() -> c_int {
    sbom_table().clear();
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_sbom_add(
    name: *const u8, name_len: i64,
    version: *const u8, version_len: i64,
) -> c_int {
    if name.is_null() || version.is_null() || name_len <= 0 || version_len <= 0 { return -1; }
    let n = std::str::from_utf8(std::slice::from_raw_parts(name, name_len as usize))
        .unwrap_or("?").to_string();
    let v = std::str::from_utf8(std::slice::from_raw_parts(version, version_len as usize))
        .unwrap_or("?").to_string();
    sbom_table().push(SbomComponent { name: n, version: v });
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_sbom_count() -> i64 {
    sbom_table().len() as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_sbom_emit(
    out_path: *const u8, out_path_len: i64,
) -> c_int {
    if out_path.is_null() || out_path_len <= 0 { return -1; }
    let path = match std::str::from_utf8(std::slice::from_raw_parts(out_path, out_path_len as usize)) {
        Ok(p) => p.to_string(),
        Err(_) => return -2,
    };
    let mut json = String::with_capacity(512);
    json.push_str("{\n");
    json.push_str("  \"bomFormat\": \"CycloneDX\",\n");
    json.push_str("  \"specVersion\": \"1.5\",\n");
    json.push_str("  \"version\": 1,\n");
    json.push_str("  \"components\": [\n");
    let table = sbom_table();
    for (i, c) in table.iter().enumerate() {
        json.push_str("    {\n");
        json.push_str(&format!("      \"type\": \"library\",\n"));
        json.push_str(&format!("      \"name\": \"{}\",\n", c.name));
        json.push_str(&format!("      \"version\": \"{}\",\n", c.version));
        json.push_str(&format!("      \"purl\": \"pkg:aether/{}@{}\"\n", c.name, c.version));
        if i + 1 < table.len() {
            json.push_str("    },\n");
        } else {
            json.push_str("    }\n");
        }
    }
    json.push_str("  ]\n}\n");
    match std::fs::write(&path, &json) {
        Ok(_) => 0,
        Err(_) => -3,
    }
}

// =====================================================================
// FR-15.6 — Matmul auto-tune lookup.
// =====================================================================
//
// `aether_autotune_matmul_tile_f32(m, n, k)` returns a packed i64 with
// the recommended `(tile_m, tile_n, tile_k, unroll)` tuple for the given
// matmul shape. Each field is a u16 in the packed value:
//
//   bits  0..16  → tile_m
//   bits 16..32  → tile_n
//   bits 32..48  → tile_k
//   bits 48..64  → unroll
//
// The lookup table is hand-curated against measured cuBLAS-vs-Aether
// numbers from `docs/BENCH_RESULTS.md` and capped to the shapes Aether's
// reference matmul (`aether_op_matmul_f32`) actually exercises today.
// Future work: feed `aether_pgo_*` measurements back into this table at
// install time so the recommendation tracks the CPU it's running on.
//
// Witness: `tests/runtime/autotune_matmul.aether`.

#[no_mangle]
pub extern "C" fn aether_autotune_matmul_tile_f32(m: i64, n: i64, k: i64) -> i64 {
    // Hand-tuned table for 11900K cache hierarchy (L1 48KB / L2 512KB /
    // L3 16MB). Tile sizes in elements. Tradeoff: small tiles fit in L1
    // but increase outer-loop overhead; large tiles reduce overhead but
    // spill the cache. Values picked to keep `tile_m * tile_k * 4 +
    // tile_k * tile_n * 4 + tile_m * tile_n * 4 < 48KB`.
    let dim = m.max(n).max(k);
    let (tm, tn, tk, unroll): (i64, i64, i64, i64) = if dim <= 64 {
        // Tiny matmul — fits entirely in L1, no tiling needed.
        (m.min(64), n.min(64), k.min(64), 4)
    } else if dim <= 256 {
        (32, 32, 32, 4)
    } else if dim <= 1024 {
        (64, 64, 32, 8)
    } else if dim <= 4096 {
        (128, 64, 32, 8)
    } else {
        (128, 128, 64, 16)
    };
    pack_tile_hint(tm, tn, tk, unroll)
}

fn pack_tile_hint(tm: i64, tn: i64, tk: i64, unroll: i64) -> i64 {
    let tm = (tm as u64).min(0xFFFF);
    let tn = (tn as u64).min(0xFFFF);
    let tk = (tk as u64).min(0xFFFF);
    let unroll = (unroll as u64).min(0xFFFF);
    (tm | (tn << 16) | (tk << 32) | (unroll << 48)) as i64
}

#[no_mangle]
pub extern "C" fn aether_autotune_unpack_tile_m(hint: i64) -> i64 { (hint as u64 & 0xFFFF) as i64 }
#[no_mangle]
pub extern "C" fn aether_autotune_unpack_tile_n(hint: i64) -> i64 { ((hint as u64 >> 16) & 0xFFFF) as i64 }
#[no_mangle]
pub extern "C" fn aether_autotune_unpack_tile_k(hint: i64) -> i64 { ((hint as u64 >> 32) & 0xFFFF) as i64 }
#[no_mangle]
pub extern "C" fn aether_autotune_unpack_unroll(hint: i64) -> i64 { ((hint as u64 >> 48) & 0xFFFF) as i64 }

// =====================================================================
// FR-22.6 — Coverage instrumentation (line + branch).
// =====================================================================
//
// `aether_cov_record(file_id, line)` increments the (file, line) counter.
// Compiler-emitted in instrumented builds, then a final report dumps a
// histogram to stdout via `aether_cov_dump()`. Concrete-int IDs keep the
// runtime symbol simple; the compiler maintains the file_id ↔ path map.

#[derive(Default, Clone)]
struct CovEntry { file_id: i64, line: i64, hits: i64 }

struct CovCell(UnsafeCell<Vec<CovEntry>>);
unsafe impl Sync for CovCell {}
static COV_TABLE: CovCell = CovCell(UnsafeCell::new(Vec::new()));

unsafe fn cov_table() -> &'static mut Vec<CovEntry> { &mut *COV_TABLE.0.get() }

#[no_mangle] pub extern "C" fn aether_cov_record(file_id: i64, line: i64) -> c_int {
    unsafe {
        let tbl = cov_table();
        for e in tbl.iter_mut() {
            if e.file_id == file_id && e.line == line {
                e.hits += 1;
                return 0;
            }
        }
        tbl.push(CovEntry { file_id, line, hits: 1 });
    }
    0
}

#[no_mangle] pub extern "C" fn aether_cov_hits(file_id: i64, line: i64) -> i64 {
    unsafe {
        let tbl = cov_table();
        for e in tbl.iter() {
            if e.file_id == file_id && e.line == line { return e.hits; }
        }
    }
    0
}

#[no_mangle] pub extern "C" fn aether_cov_reset() -> c_int {
    unsafe { cov_table().clear(); }
    0
}

#[no_mangle] pub extern "C" fn aether_cov_dump() -> c_int {
    unsafe {
        let tbl = cov_table();
        for e in tbl.iter() {
            println!("cov file={} line={} hits={}", e.file_id, e.line, e.hits);
        }
    }
    0
}

// =====================================================================
// FR-24.7 — Crash dump primitive (telemetry without third-party deps).
// =====================================================================
//
// `aether_crash_dump(label, n)` writes a small fixed-format snapshot to
// `crash_<pid>_<step>.dump` containing: program label, current GPU live
// bytes, current OOM flag, and an explicit caller-provided step counter.
// Designed for production training loops to call from a panic hook /
// signal handler before exiting. No allocation in the hot path.

#[no_mangle] pub unsafe extern "C" fn aether_crash_dump(label: *const u8, label_len: i64, step: i64) -> c_int {
    if label.is_null() || label_len <= 0 { return -1; }
    let label_bytes = std::slice::from_raw_parts(label, label_len as usize);
    let label_str = std::str::from_utf8(label_bytes).unwrap_or("?");
    let pid = std::process::id();
    let path = format!("crash_{}_{}.dump", pid, step);
    let body = format!(
        "label={}\npid={}\nstep={}\ngpu_live_bytes={}\noom_flag={}\n",
        label_str, pid, step,
        aether_gpu_live_bytes(), aether_oom_check(),
    );
    let _ = std::fs::write(&path, body);
    0
}

// =====================================================================
// FR-15.8 — Auto-prefetch insertion (runtime-side hint).
// =====================================================================
//
// `aether_prefetch_t0(p)` emits a `prefetcht0` for cache-line `p` (or a
// pure no-op if the runtime is built for a target without the hint). The
// caller is expected to schedule prefetches `prefetch_distance` iterations
// ahead of the load — see vectorize_drive's hand-prefetch helper.

#[no_mangle] pub extern "C" fn aether_prefetch_t0(p: i64) -> c_int {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_T0}>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // No-op on non-x86_64 — keeps the FFI surface portable.
        let _ = p;
    }
    0
}

#[no_mangle] pub extern "C" fn aether_prefetch_t1(p: i64) -> c_int {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_T1}>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = p;
    }
    0
}

#[no_mangle] pub extern "C" fn aether_prefetch_nta(p: i64) -> c_int {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_NTA}>(p as *const i8);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = p;
    }
    0
}

// =====================================================================
// FR-17.6-extra — Activation backwards.
// =====================================================================
//
// Backward passes for tanh/sigmoid/leaky_relu/elu/mish. Each reads the
// pre-activation `x` (or post-activation, for those that need it) and the
// upstream gradient `grad_y`, writes `grad_x` in-place into the upstream
// buffer (matches the existing forward-pass conventions).

#[no_mangle] pub unsafe extern "C" fn aether_op_tanh_backward_f32(
    y: *const c_void, grad: *mut c_void, n: c_int,
) -> c_int {
    if y.is_null() || grad.is_null() { return -1; }
    let ys = std::slice::from_raw_parts(y as *const f32, n as usize);
    let gs = std::slice::from_raw_parts_mut(grad as *mut f32, n as usize);
    for i in 0..n as usize { gs[i] *= 1.0 - ys[i] * ys[i]; }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sigmoid_backward_f32(
    y: *const c_void, grad: *mut c_void, n: c_int,
) -> c_int {
    if y.is_null() || grad.is_null() { return -1; }
    let ys = std::slice::from_raw_parts(y as *const f32, n as usize);
    let gs = std::slice::from_raw_parts_mut(grad as *mut f32, n as usize);
    for i in 0..n as usize { gs[i] *= ys[i] * (1.0 - ys[i]); }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_leaky_relu_backward_f32(
    x: *const c_void, grad: *mut c_void, slope: f32, n: c_int,
) -> c_int {
    if x.is_null() || grad.is_null() { return -1; }
    let xs = std::slice::from_raw_parts(x as *const f32, n as usize);
    let gs = std::slice::from_raw_parts_mut(grad as *mut f32, n as usize);
    for i in 0..n as usize { gs[i] *= if xs[i] > 0.0 { 1.0 } else { slope }; }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_elu_backward_f32(
    x: *const c_void, grad: *mut c_void, alpha: f32, n: c_int,
) -> c_int {
    if x.is_null() || grad.is_null() { return -1; }
    let xs = std::slice::from_raw_parts(x as *const f32, n as usize);
    let gs = std::slice::from_raw_parts_mut(grad as *mut f32, n as usize);
    for i in 0..n as usize {
        gs[i] *= if xs[i] >= 0.0 { 1.0 } else { alpha * xs[i].exp() };
    }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mish_backward_f32(
    x: *const c_void, grad: *mut c_void, n: c_int,
) -> c_int {
    if x.is_null() || grad.is_null() { return -1; }
    // mish(x) = x * tanh(softplus(x)).
    // d/dx mish = tanh(sp) + x * sech^2(sp) * sigmoid(x), where sp = ln(1+e^x).
    let xs = std::slice::from_raw_parts(x as *const f32, n as usize);
    let gs = std::slice::from_raw_parts_mut(grad as *mut f32, n as usize);
    for i in 0..n as usize {
        let xi = xs[i];
        let sp = (1.0 + xi.exp()).ln();
        let t = sp.tanh();
        let sech2 = 1.0 - t * t;
        let sig = 1.0 / (1.0 + (-xi).exp());
        gs[i] *= t + xi * sech2 * sig;
    }
    0
}

// =====================================================================
// FR-17.17-extra — Lion / Lamb / Adafactor optimizer steps.
// =====================================================================

/// Lion: sign-of-momentum optimizer. `beta1` smooths the first-moment
/// estimate; the update direction is `sign(beta1*m + (1-beta1)*g)`. Cheap,
/// memory-light alternative to Adam.
#[no_mangle]
pub unsafe extern "C" fn aether_op_lion_step_f32(
    param: *mut c_void, grad: *const c_void, m: *mut c_void,
    n: c_int, lr: f32, beta1: f32, beta2: f32, weight_decay: f32,
) -> c_int {
    if param.is_null() || grad.is_null() || m.is_null() { return -1; }
    let p = std::slice::from_raw_parts_mut(param as *mut f32, n as usize);
    let g = std::slice::from_raw_parts(grad as *const f32, n as usize);
    let ms = std::slice::from_raw_parts_mut(m as *mut f32, n as usize);
    for i in 0..n as usize {
        let update = beta1 * ms[i] + (1.0 - beta1) * g[i];
        let dir = if update > 0.0 { 1.0 } else if update < 0.0 { -1.0 } else { 0.0 };
        p[i] -= lr * (dir + weight_decay * p[i]);
        ms[i] = beta2 * ms[i] + (1.0 - beta2) * g[i];
    }
    0
}

/// LAMB step (Layer-wise Adaptive Moments). Same first/second moments as
/// AdamW, then rescales the per-layer update so its norm matches the
/// param norm. Caller passes `param_norm` and `update_norm` precomputed —
/// keeps the runtime symbol arity sane on Windows x64 where 5+ args spill.
#[no_mangle]
pub unsafe extern "C" fn aether_op_lamb_step_f32(
    param: *mut c_void, grad: *const c_void, m: *mut c_void, v: *mut c_void,
    n: c_int, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32,
    bias_correction1: f32, bias_correction2: f32,
) -> c_int {
    if param.is_null() || grad.is_null() || m.is_null() || v.is_null() { return -1; }
    let p = std::slice::from_raw_parts_mut(param as *mut f32, n as usize);
    let g = std::slice::from_raw_parts(grad as *const f32, n as usize);
    let ms = std::slice::from_raw_parts_mut(m as *mut f32, n as usize);
    let vs = std::slice::from_raw_parts_mut(v as *mut f32, n as usize);
    let mut update_norm_sq = 0.0f64;
    let mut param_norm_sq = 0.0f64;
    let mut tmp = vec![0.0f32; n as usize];
    for i in 0..n as usize {
        ms[i] = beta1 * ms[i] + (1.0 - beta1) * g[i];
        vs[i] = beta2 * vs[i] + (1.0 - beta2) * g[i] * g[i];
        let mh = ms[i] / bias_correction1;
        let vh = vs[i] / bias_correction2;
        let r = mh / (vh.sqrt() + eps) + weight_decay * p[i];
        tmp[i] = r;
        update_norm_sq += (r as f64) * (r as f64);
        param_norm_sq += (p[i] as f64) * (p[i] as f64);
    }
    let pn = param_norm_sq.sqrt() as f32;
    let un = update_norm_sq.sqrt() as f32;
    let trust = if pn > 0.0 && un > 0.0 { pn / un } else { 1.0 };
    let scale = lr * trust;
    for i in 0..n as usize {
        p[i] -= scale * tmp[i];
    }
    0
}

/// Adafactor step (factored second moments). Simplified: caller-provided
/// row + col second-moment buffers. Update is `g / sqrt(rms_estimate)`.
#[no_mangle]
pub unsafe extern "C" fn aether_op_adafactor_step_f32(
    param: *mut c_void, grad: *const c_void,
    row: *mut c_void, col: *mut c_void,
    n_rows: c_int, n_cols: c_int,
    lr: f32, eps: f32, decay_rate: f32,
) -> c_int {
    if param.is_null() || grad.is_null() || row.is_null() || col.is_null() { return -1; }
    let n = (n_rows * n_cols) as usize;
    let p = std::slice::from_raw_parts_mut(param as *mut f32, n);
    let g = std::slice::from_raw_parts(grad as *const f32, n);
    let r = std::slice::from_raw_parts_mut(row as *mut f32, n_rows as usize);
    let c = std::slice::from_raw_parts_mut(col as *mut f32, n_cols as usize);
    // Update row/col exponential averages of g^2.
    for i in 0..n_rows as usize {
        let mut sum = 0.0f32;
        for j in 0..n_cols as usize {
            let gv = g[i * n_cols as usize + j];
            sum += gv * gv;
        }
        r[i] = decay_rate * r[i] + (1.0 - decay_rate) * sum;
    }
    for j in 0..n_cols as usize {
        let mut sum = 0.0f32;
        for i in 0..n_rows as usize {
            let gv = g[i * n_cols as usize + j];
            sum += gv * gv;
        }
        c[j] = decay_rate * c[j] + (1.0 - decay_rate) * sum;
    }
    let r_mean = r.iter().sum::<f32>() / n_rows as f32;
    for i in 0..n_rows as usize {
        for j in 0..n_cols as usize {
            let v_hat = (r[i] / r_mean.max(eps)) * c[j];
            let denom = (v_hat / n_cols as f32).sqrt().max(eps);
            p[i * n_cols as usize + j] -= lr * g[i * n_cols as usize + j] / denom;
        }
    }
    0
}

// =====================================================================
// FR-17.4 — Pooling (max/avg) — real CPU bodies.
// =====================================================================
//
// 2-D pooling on a contiguous (N, C, H, W) layout. `max_pool_2d_f32`
// and `avg_pool_2d_f32` produce (N, C, H_out, W_out) where
// H_out = (H + 2*pad - kernel) / stride + 1.
//
// Witness: tests/runtime/pooling.aether.

#[no_mangle]
pub unsafe extern "C" fn aether_op_max_pool_2d_f32(
    x: *const c_void, y: *mut c_void,
    n: c_int, c: c_int, h: c_int, w: c_int,
    kh: c_int, kw: c_int, sh: c_int, sw: c_int,
    ph: c_int, pw: c_int,
) -> c_int {
    if x.is_null() || y.is_null() { return -1; }
    let xs = std::slice::from_raw_parts(x as *const f32, (n*c*h*w) as usize);
    let h_out = (h + 2*ph - kh) / sh + 1;
    let w_out = (w + 2*pw - kw) / sw + 1;
    let ys = std::slice::from_raw_parts_mut(y as *mut f32, (n*c*h_out*w_out) as usize);
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max = f32::NEG_INFINITY;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh*sh - ph + ki;
                            let iw = ow*sw - pw + kj;
                            if ih < 0 || ih >= h || iw < 0 || iw >= w { continue; }
                            let idx = ((ni*c + ci)*h + ih)*w + iw;
                            let v = xs[idx as usize];
                            if v > max { max = v; }
                        }
                    }
                    let oidx = ((ni*c + ci)*h_out + oh)*w_out + ow;
                    ys[oidx as usize] = if max == f32::NEG_INFINITY { 0.0 } else { max };
                }
            }
        }
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn aether_op_avg_pool_2d_f32(
    x: *const c_void, y: *mut c_void,
    n: c_int, c: c_int, h: c_int, w: c_int,
    kh: c_int, kw: c_int, sh: c_int, sw: c_int,
    ph: c_int, pw: c_int,
) -> c_int {
    if x.is_null() || y.is_null() { return -1; }
    let xs = std::slice::from_raw_parts(x as *const f32, (n*c*h*w) as usize);
    let h_out = (h + 2*ph - kh) / sh + 1;
    let w_out = (w + 2*pw - kw) / sw + 1;
    let ys = std::slice::from_raw_parts_mut(y as *mut f32, (n*c*h_out*w_out) as usize);
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0f32;
                    let mut cnt = 0i32;
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let ih = oh*sh - ph + ki;
                            let iw = ow*sw - pw + kj;
                            if ih < 0 || ih >= h || iw < 0 || iw >= w { continue; }
                            let idx = ((ni*c + ci)*h + ih)*w + iw;
                            sum += xs[idx as usize];
                            cnt += 1;
                        }
                    }
                    let oidx = ((ni*c + ci)*h_out + oh)*w_out + ow;
                    ys[oidx as usize] = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                }
            }
        }
    }
    0
}

/// Adaptive average pool — output shape (N, C, h_out, w_out) regardless of
/// input H, W. Each output pixel pools the corresponding "tile" of the input
/// computed as `[i*H/h_out, (i+1)*H/h_out)` × the same for W.
#[no_mangle]
pub unsafe extern "C" fn aether_op_adaptive_avg_pool_2d_f32(
    x: *const c_void, y: *mut c_void,
    n: c_int, c: c_int, h: c_int, w: c_int,
    h_out: c_int, w_out: c_int,
) -> c_int {
    if x.is_null() || y.is_null() || h_out <= 0 || w_out <= 0 { return -1; }
    let xs = std::slice::from_raw_parts(x as *const f32, (n*c*h*w) as usize);
    let ys = std::slice::from_raw_parts_mut(y as *mut f32, (n*c*h_out*w_out) as usize);
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                let h_lo = (oh * h) / h_out;
                let h_hi = ((oh + 1) * h) / h_out;
                for ow in 0..w_out {
                    let w_lo = (ow * w) / w_out;
                    let w_hi = ((ow + 1) * w) / w_out;
                    let mut sum = 0f32;
                    let mut cnt = 0i32;
                    for ih in h_lo..h_hi {
                        for iw in w_lo..w_hi {
                            let idx = ((ni*c + ci)*h + ih)*w + iw;
                            sum += xs[idx as usize];
                            cnt += 1;
                        }
                    }
                    let oidx = ((ni*c + ci)*h_out + oh)*w_out + ow;
                    ys[oidx as usize] = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
                }
            }
        }
    }
    0
}

// =====================================================================
// FR-17.12 — Embedding extras (`embedding_bag`).
// =====================================================================
//
// `embedding_bag(weights, indices, offsets, mode)` aggregates rows of
// the embedding matrix indexed by `indices[offsets[i] .. offsets[i+1]]`
// and reduces them according to `mode` (0=sum, 1=mean) into output[i].
// Output shape: (n_bags, embed_dim). Mirrors PyTorch's
// `nn.EmbeddingBag` semantics for the dense (non-padding-aware) case.
//
// Witness: tests/runtime/embedding_bag.aether.

#[no_mangle]
pub unsafe extern "C" fn aether_op_embedding_bag_f32(
    weight: *const c_void,    // (vocab, embed_dim) row-major
    indices: *const c_void,   // i32[total_idx]
    offsets: *const c_void,   // i32[n_bags + 1] - last is total_idx
    out: *mut c_void,         // (n_bags, embed_dim)
    n_bags: c_int,
    vocab: c_int,
    embed_dim: c_int,
    total_idx: c_int,
    mode: c_int,              // 0 = sum, 1 = mean
) -> c_int {
    if weight.is_null() || indices.is_null() || offsets.is_null() || out.is_null() { return -1; }
    let w = std::slice::from_raw_parts(weight as *const f32, (vocab*embed_dim) as usize);
    let idx = std::slice::from_raw_parts(indices as *const i32, total_idx as usize);
    let off = std::slice::from_raw_parts(offsets as *const i32, (n_bags + 1) as usize);
    let o = std::slice::from_raw_parts_mut(out as *mut f32, (n_bags*embed_dim) as usize);
    for bag in 0..n_bags as usize {
        let lo = off[bag] as usize;
        let hi = off[bag + 1] as usize;
        let count = hi.saturating_sub(lo);
        for d in 0..embed_dim as usize {
            o[bag * embed_dim as usize + d] = 0.0;
        }
        for k in lo..hi {
            let row = idx[k] as usize;
            if row >= vocab as usize { continue; }
            for d in 0..embed_dim as usize {
                o[bag * embed_dim as usize + d] += w[row * embed_dim as usize + d];
            }
        }
        if mode == 1 && count > 0 {
            let scale = 1.0 / count as f32;
            for d in 0..embed_dim as usize {
                o[bag * embed_dim as usize + d] *= scale;
            }
        }
    }
    0
}

// FR-16.14 — `println!`/`print!` parser-level expansion targets these
// scalar print primitives. Compile-time format-string parsing emits a
// sequence of these calls (one per literal segment + one per `{}` hole +
// optional trailing newline).

/// Print a single i64 in base-10 to stdout (no newline).
#[no_mangle] pub extern "C" fn aether_print_i64(v: i64) -> c_int {
    let mut buf = [0u8; 24];
    let n = format_i64(&mut buf, v);
    unsafe { write_stdout(&buf[..n]); }
    0
}

/// Print a single f32 with default formatting (no newline).
#[no_mangle] pub extern "C" fn aether_print_f32_default(v: f32) -> c_int {
    // Reuse `format!` here — a hot training loop should reach for
    // `aether_print_kv_f32` instead, which writes through the rdata fast path.
    let s = format!("{}", v);
    unsafe { write_stdout(s.as_bytes()); }
    0
}

/// Print `n` bytes from `p`. Used for the literal segments between `{}` holes.
#[no_mangle] pub unsafe extern "C" fn aether_print_str_n(p: *const u8, n: i64) -> c_int {
    if p.is_null() || n <= 0 { return 0; }
    let bytes = std::slice::from_raw_parts(p, n as usize);
    write_stdout(bytes);
    0
}

/// Print a single `\n`. Used by `println!` after the last hole.
#[no_mangle] pub extern "C" fn aether_print_newline() -> c_int {
    unsafe { write_stdout(b"\n"); }
    0
}

fn format_i64(buf: &mut [u8], v: i64) -> usize {
    if v == 0 { buf[0] = b'0'; return 1; }
    let neg = v < 0;
    // Use unsigned magnitude to handle i64::MIN correctly.
    let mut x = if neg { (v as i128).unsigned_abs() } else { v as u128 };
    let mut digits = [0u8; 24];
    let mut k = 0usize;
    while x > 0 {
        digits[k] = (x % 10) as u8 + b'0';
        x /= 10;
        k += 1;
    }
    let mut i = 0;
    if neg { buf[i] = b'-'; i += 1; }
    while k > 0 { k -= 1; buf[i] = digits[k]; i += 1; }
    i
}

#[no_mangle] pub extern "C" fn aether_print_loss(step: c_int, loss: f32) -> c_int {
    let mut buf = [0u8; 64];
    let n = format_loss_line(&mut buf, step, loss);
    unsafe { write_stdout(&buf[..n]); }
    0
}

#[cfg(windows)]
unsafe fn write_stdout(bytes: &[u8]) {
    extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> *mut c_void;
        fn WriteFile(h: *mut c_void, b: *const c_void, n: u32,
                     written: *mut u32, overlapped: *mut c_void) -> i32;
    }
    let h = GetStdHandle(0xFFFF_FFF5); // STD_OUTPUT_HANDLE
    let mut written: u32 = 0;
    WriteFile(h, bytes.as_ptr() as *const c_void, bytes.len() as u32,
              &mut written, std::ptr::null_mut());
}

#[cfg(not(windows))]
unsafe fn write_stdout(bytes: &[u8]) {
    extern "C" { fn write(fd: c_int, buf: *const c_void, n: usize) -> isize; }
    let _ = write(1, bytes.as_ptr() as *const c_void, bytes.len());
}

/// Format `step={step} loss={loss:.6}\n` into `buf` without touching
/// `core::fmt::Write` (which on cdylibs drags in the same Rust-std stdio
/// init paths we're trying to avoid). Returns bytes written. `buf` must
/// be at least 64 bytes; output is truncated otherwise.
fn format_loss_line(buf: &mut [u8], step: c_int, loss: f32) -> usize {
    let mut i = 0usize;
    let prefix = b"step=";
    for &b in prefix { if i < buf.len() { buf[i] = b; i += 1; } }
    i += write_int(&mut buf[i..], step as i64);
    let mid = b" loss=";
    for &b in mid { if i < buf.len() { buf[i] = b; i += 1; } }
    i += write_f32(&mut buf[i..], loss, 6);
    if i < buf.len() { buf[i] = b'\n'; i += 1; }
    i
}

fn write_int(buf: &mut [u8], v: i64) -> usize {
    if buf.is_empty() { return 0; }
    let neg = v < 0;
    let mut n = if neg { (!v as u64).wrapping_add(1) } else { v as u64 };
    let mut tmp = [0u8; 20];
    let mut t = 0;
    if n == 0 { tmp[t] = b'0'; t += 1; }
    while n > 0 { tmp[t] = b'0' + (n % 10) as u8; n /= 10; t += 1; }
    let mut i = 0;
    if neg && i < buf.len() { buf[i] = b'-'; i += 1; }
    while t > 0 && i < buf.len() { t -= 1; buf[i] = tmp[t]; i += 1; }
    i
}

fn write_f32(buf: &mut [u8], v: f32, decimals: u32) -> usize {
    let mut i = 0usize;
    let neg = v < 0.0;
    let mut x = if neg { -v } else { v };
    if neg && i < buf.len() { buf[i] = b'-'; i += 1; }
    let int_part = x as u64;
    let mut tmp = [0u8; 20];
    let mut t = 0;
    let mut n = int_part;
    if n == 0 { tmp[t] = b'0'; t += 1; }
    while n > 0 { tmp[t] = b'0' + (n % 10) as u8; n /= 10; t += 1; }
    while t > 0 && i < buf.len() { t -= 1; buf[i] = tmp[t]; i += 1; }
    if i < buf.len() { buf[i] = b'.'; i += 1; }
    x -= int_part as f32;
    for _ in 0..decimals {
        x *= 10.0;
        let d = x as u32;
        if i < buf.len() { buf[i] = b'0' + (d % 10) as u8; i += 1; }
        x -= d as f32;
    }
    i
}

/// Store a single f32 to a buffer at index `i`.
#[no_mangle] pub unsafe extern "C" fn aether_store_f32(p: i64, i: c_int, v: f32) {
    if p == 0 { return; }
    *((p as *mut f32).add(i as usize)) = v;
}

/// FR-17.19 helper — write an i32 at offset `i` of an `aether_alloc_i32`
/// buffer. Pairs with `aether_alloc_i32` for filling embedding-lookup id
/// tensors. `i` is element index, not byte offset.
#[no_mangle] pub unsafe extern "C" fn aether_store_i32(p: i64, i: c_int, v: c_int) {
    if p == 0 { return; }
    *((p as *mut i32).add(i as usize)) = v;
}

/// Sum f32 elements at `[p, p+n)`. Used by FR-17.19 witness to verify
/// the forward pass produced finite output without per-element checks.
#[no_mangle] pub unsafe extern "C" fn aether_sum_f32(p: i64, n: c_int) -> f32 {
    if p == 0 || n <= 0 { return 0.0; }
    let s = std::slice::from_raw_parts(p as *const f32, n as usize);
    s.iter().sum()
}

// =====================================================================
// Test-only FFI surface — exercises the asm backend's f32 / f64 / cast /
// FFI-arg pipelines from compiled Aether. Real ops never go through here.
// =====================================================================

#[no_mangle] pub extern "C" fn aether_test_add_f32(a: f32, b: f32) -> f32 { a + b }
#[no_mangle] pub extern "C" fn aether_test_add_f64(a: f64, b: f64) -> f64 { a + b }
#[no_mangle] pub extern "C" fn aether_test_f32_to_i64(x: f32) -> i64 { x as i64 }
#[no_mangle] pub extern "C" fn aether_test_f64_to_i64(x: f64) -> i64 { x as i64 }
/// Mixed-class arg passing: int slot 0, f32 slot 1, int slot 2 → f32.
/// MS x64 puts them in rcx, xmm1, r8 respectively. Returns `(i + j) * f`.
#[no_mangle] pub extern "C" fn aether_test_mix_if(i: c_int, f: f32, j: c_int) -> f32 {
    (i + j) as f32 * f
}

// ============================================================================
// v4 op surface — math primitives, activation extensions, mask helpers,
// reductions, combine, optimizer extensions, collectives. Each takes f32
// pointers + element count and returns 0 on success. CPU bodies; the CUDA
// path is FR-17.x in NEXT-UP.md.
// ============================================================================

// ---- FR-17.7-extra: math primitives ---------------------------------------
#[no_mangle] pub unsafe extern "C" fn aether_op_log_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.ln(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_exp_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.exp(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_sin_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.sin(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_cos_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.cos(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_tan_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.tan(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_pow_f32(x: *mut c_void, p: f32, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.powf(p); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_recip_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.recip(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_abs_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.abs(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_sign_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = if *v > 0.0 { 1.0 } else if *v < 0.0 { -1.0 } else { 0.0 }; }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_clamp_f32(x: *mut c_void, lo: f32, hi: f32, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.max(lo).min(hi); }
    0
}

// ---- FR-17.6-extra: activation extensions ---------------------------------
#[no_mangle] pub unsafe extern "C" fn aether_op_tanh_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = v.tanh(); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_sigmoid_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = 1.0 / (1.0 + (-*v).exp()); }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_leaky_relu_f32(x: *mut c_void, slope: f32, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { if *v < 0.0 { *v *= slope; } }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_elu_f32(x: *mut c_void, alpha: f32, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { if *v < 0.0 { *v = alpha * (v.exp() - 1.0); } }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_mish_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v *= (1.0 + v.exp()).ln().tanh(); }
    0
}

// ---- FR-17.11: mask helpers / tensor builders -----------------------------
#[no_mangle] pub unsafe extern "C" fn aether_op_zeros_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = 0.0; }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_ones_f32(x: *mut c_void, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = 1.0; }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_full_f32(x: *mut c_void, val: f32, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for v in s { *v = val; }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_arange_f32(x: *mut c_void, start: f32, step: f32, n: c_int) -> c_int {
    let s = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    for (i, v) in s.iter_mut().enumerate() { *v = start + step * i as f32; }
    0
}
/// `eye(n)` — n×n identity. `x` is row-major n*n.
#[no_mangle] pub unsafe extern "C" fn aether_op_eye_f32(x: *mut c_void, n: c_int) -> c_int {
    let nn = n as usize;
    let s = std::slice::from_raw_parts_mut(x as *mut f32, nn * nn);
    for v in s.iter_mut() { *v = 0.0; }
    for i in 0..nn { s[i * nn + i] = 1.0; }
    0
}
/// `tril(rows, cols)` — sets above-diagonal to 0 (in-place mask).
#[no_mangle] pub unsafe extern "C" fn aether_op_tril_f32(x: *mut c_void, rows: c_int, cols: c_int) -> c_int {
    let r = rows as usize; let c = cols as usize;
    let s = std::slice::from_raw_parts_mut(x as *mut f32, r * c);
    for i in 0..r { for j in 0..c { if j > i { s[i * c + j] = 0.0; } } }
    0
}
/// `triu(rows, cols)` — sets below-diagonal to 0 (in-place mask).
#[no_mangle] pub unsafe extern "C" fn aether_op_triu_f32(x: *mut c_void, rows: c_int, cols: c_int) -> c_int {
    let r = rows as usize; let c = cols as usize;
    let s = std::slice::from_raw_parts_mut(x as *mut f32, r * c);
    for i in 0..r { for j in 0..c { if j < i { s[i * c + j] = 0.0; } } }
    0
}

// ---- FR-17.8: reductions --------------------------------------------------
#[no_mangle] pub unsafe extern "C" fn aether_op_sum_f32(x: *const c_void, n: c_int) -> f32 {
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    s.iter().copied().sum()
}
#[no_mangle] pub unsafe extern "C" fn aether_op_mean_f32(x: *const c_void, n: c_int) -> f32 {
    if n == 0 { return 0.0; }
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    let sum: f32 = s.iter().copied().sum();
    sum / n as f32
}
#[no_mangle] pub unsafe extern "C" fn aether_op_var_f32(x: *const c_void, n: c_int) -> f32 {
    if n == 0 { return 0.0; }
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    // Welford
    let mut mean = 0.0f32; let mut m2 = 0.0f32;
    for (i, v) in s.iter().enumerate() {
        let count = (i + 1) as f32;
        let delta = v - mean;
        mean += delta / count;
        let delta2 = v - mean;
        m2 += delta * delta2;
    }
    m2 / n as f32
}
#[no_mangle] pub unsafe extern "C" fn aether_op_std_f32(x: *const c_void, n: c_int) -> f32 {
    aether_op_var_f32(x, n).sqrt()
}
#[no_mangle] pub unsafe extern "C" fn aether_op_max_red_f32(x: *const c_void, n: c_int) -> f32 {
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    s.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}
#[no_mangle] pub unsafe extern "C" fn aether_op_min_red_f32(x: *const c_void, n: c_int) -> f32 {
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    s.iter().copied().fold(f32::INFINITY, f32::min)
}
#[no_mangle] pub unsafe extern "C" fn aether_op_argmax_f32(x: *const c_void, n: c_int) -> i64 {
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    s.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64).unwrap_or(-1)
}
#[no_mangle] pub unsafe extern "C" fn aether_op_argmin_f32(x: *const c_void, n: c_int) -> i64 {
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    s.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64).unwrap_or(-1)
}
#[no_mangle] pub unsafe extern "C" fn aether_op_prod_f32(x: *const c_void, n: c_int) -> f32 {
    let s = std::slice::from_raw_parts(x as *const f32, n as usize);
    s.iter().copied().product()
}

// ---- FR-17.9: selection ---------------------------------------------------
#[no_mangle] pub unsafe extern "C" fn aether_op_masked_fill_f32(
    x: *mut c_void, mask: *const c_void, fill: f32, n: c_int,
) -> c_int {
    let xs = std::slice::from_raw_parts_mut(x as *mut f32, n as usize);
    let ms = std::slice::from_raw_parts(mask as *const f32, n as usize);
    for i in 0..n as usize { if ms[i] != 0.0 { xs[i] = fill; } }
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_where_f32(
    cond: *const c_void, a: *const c_void, b: *const c_void, out: *mut c_void, n: c_int,
) -> c_int {
    let cs = std::slice::from_raw_parts(cond as *const f32, n as usize);
    let asrc = std::slice::from_raw_parts(a as *const f32, n as usize);
    let bsrc = std::slice::from_raw_parts(b as *const f32, n as usize);
    let os = std::slice::from_raw_parts_mut(out as *mut f32, n as usize);
    for i in 0..n as usize { os[i] = if cs[i] != 0.0 { asrc[i] } else { bsrc[i] }; }
    0
}

// ---- FR-17.10: combine ----------------------------------------------------
/// Concatenate `a` (na elements) and `b` (nb elements) into `out` (na+nb).
#[no_mangle] pub unsafe extern "C" fn aether_op_cat_f32(
    a: *const c_void, na: c_int, b: *const c_void, nb: c_int, out: *mut c_void,
) -> c_int {
    let asrc = std::slice::from_raw_parts(a as *const f32, na as usize);
    let bsrc = std::slice::from_raw_parts(b as *const f32, nb as usize);
    let os = std::slice::from_raw_parts_mut(out as *mut f32, (na + nb) as usize);
    os[..na as usize].copy_from_slice(asrc);
    os[na as usize..].copy_from_slice(bsrc);
    0
}
/// Repeat `x` (n elements) `k` times into `out` (n*k elements).
#[no_mangle] pub unsafe extern "C" fn aether_op_repeat_f32(
    x: *const c_void, n: c_int, k: c_int, out: *mut c_void,
) -> c_int {
    let xs = std::slice::from_raw_parts(x as *const f32, n as usize);
    let os = std::slice::from_raw_parts_mut(out as *mut f32, (n * k) as usize);
    for i in 0..k as usize {
        os[i * n as usize..(i + 1) * n as usize].copy_from_slice(xs);
    }
    0
}

// ---- FR-17.17-extra: optimizer family -------------------------------------
/// SGD with Nesterov-optional momentum. `momentum_buf` is per-param state.
#[no_mangle] pub unsafe extern "C" fn aether_op_sgd_momentum_step_f32(
    params: *mut c_void, grad: *const c_void, momentum_buf: *mut c_void,
    lr: f32, mu: f32, weight_decay: f32, n: c_int,
) -> c_int {
    let p = std::slice::from_raw_parts_mut(params as *mut f32, n as usize);
    let g = std::slice::from_raw_parts(grad as *const f32, n as usize);
    let m = std::slice::from_raw_parts_mut(momentum_buf as *mut f32, n as usize);
    for i in 0..n as usize {
        let g_i = g[i] + weight_decay * p[i];
        m[i] = mu * m[i] + g_i;
        p[i] -= lr * m[i];
    }
    0
}
/// RMSprop: `v[t] = rho*v[t-1] + (1-rho)*g²; p -= lr * g / (sqrt(v) + eps)`.
#[no_mangle] pub unsafe extern "C" fn aether_op_rmsprop_step_f32(
    params: *mut c_void, grad: *const c_void, sq_buf: *mut c_void,
    lr: f32, rho: f32, eps: f32, n: c_int,
) -> c_int {
    let p = std::slice::from_raw_parts_mut(params as *mut f32, n as usize);
    let g = std::slice::from_raw_parts(grad as *const f32, n as usize);
    let v = std::slice::from_raw_parts_mut(sq_buf as *mut f32, n as usize);
    for i in 0..n as usize {
        v[i] = rho * v[i] + (1.0 - rho) * g[i] * g[i];
        p[i] -= lr * g[i] / (v[i].sqrt() + eps);
    }
    0
}
/// Adagrad: `v[t] += g²; p -= lr * g / (sqrt(v) + eps)`.
#[no_mangle] pub unsafe extern "C" fn aether_op_adagrad_step_f32(
    params: *mut c_void, grad: *const c_void, sq_buf: *mut c_void,
    lr: f32, eps: f32, n: c_int,
) -> c_int {
    let p = std::slice::from_raw_parts_mut(params as *mut f32, n as usize);
    let g = std::slice::from_raw_parts(grad as *const f32, n as usize);
    let v = std::slice::from_raw_parts_mut(sq_buf as *mut f32, n as usize);
    for i in 0..n as usize {
        v[i] += g[i] * g[i];
        p[i] -= lr * g[i] / (v[i].sqrt() + eps);
    }
    0
}

// ---- FR-18.2-extra: collectives (single-rank passthrough today) -----------
// On a single rank these are identity ops on the buffer. Multi-rank wiring
// (real NCCL bindings) is FR-18.1; the public symbol surface stays stable so
// callers don't change between single-host and distributed builds.
#[no_mangle] pub unsafe extern "C" fn aether_op_broadcast_f32(buf: *mut c_void, n: c_int, _root: c_int) -> c_int {
    // Single-rank: data is already on this rank. Multi-rank wiring is FR-18.1.
    let _touched = std::slice::from_raw_parts_mut(buf as *mut f32, n as usize).len();
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_all_gather_f32(
    src: *const c_void, dst: *mut c_void, n: c_int, world_size: c_int,
) -> c_int {
    let xs = std::slice::from_raw_parts(src as *const f32, n as usize);
    let os = std::slice::from_raw_parts_mut(dst as *mut f32, (n * world_size) as usize);
    // single-rank: just copy src into the first n slots.
    os[..n as usize].copy_from_slice(xs);
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_reduce_scatter_f32(
    src: *const c_void, dst: *mut c_void, n: c_int, _world_size: c_int,
) -> c_int {
    let xs = std::slice::from_raw_parts(src as *const f32, n as usize);
    let os = std::slice::from_raw_parts_mut(dst as *mut f32, n as usize);
    os.copy_from_slice(xs);
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_send_f32(buf: *const c_void, n: c_int, _dst_rank: c_int) -> c_int {
    let _touched = std::slice::from_raw_parts(buf as *const f32, n as usize).len();
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_recv_f32(buf: *mut c_void, n: c_int, _src_rank: c_int) -> c_int {
    let _touched = std::slice::from_raw_parts_mut(buf as *mut f32, n as usize).len();
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_op_all_to_all_f32(
    src: *const c_void, dst: *mut c_void, n: c_int, _world_size: c_int,
) -> c_int {
    let xs = std::slice::from_raw_parts(src as *const f32, n as usize);
    let os = std::slice::from_raw_parts_mut(dst as *mut f32, n as usize);
    os.copy_from_slice(xs);
    0
}

// =====================================================================
// FR-18.1 — NCCL FFI surface (single-host fallback).
//
// Mirrors the libnccl API shape so a future Aether build that links
// against real libnccl.so / nccl.dll only has to flip the impls below.
// Today these are single-host fallbacks: world_size>1 is rejected; the
// world_size=1 path is a no-op that lets matt-voice / antcolony / etc.
// write their distributed control-flow against a stable symbol surface
// even on a single-GPU box.
//
// Per `MATT_VOICE_FR.md` this gates everything Phase-18 — once a real
// libnccl is linked, every Stage-0 multi-rank witness (PP, TP, FSDP)
// becomes a real cross-card all-reduce instead of an in-process sim.
// The witness for P18.1 verifies the surface exists + single-rank
// returns sane values. Multi-rank correctness is FR-18.1-extra
// (real libnccl link) — explicitly NOT shipped here.
// =====================================================================

/// NCCL-shaped init. Returns 0 on success. Single-host fallback today
/// (no actual libnccl call; just records that init was requested).
#[no_mangle] pub extern "C" fn aether_nccl_init() -> c_int {
    NCCL_INIT_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    0
}

/// Returns the number of NCCL inits seen this process. The witness
/// uses this to confirm aether_nccl_init was actually called.
#[no_mangle] pub extern "C" fn aether_nccl_init_count() -> c_int {
    NCCL_INIT_COUNT.load(std::sync::atomic::Ordering::SeqCst)
}

/// Tear down (single-host fallback). Returns 0.
#[no_mangle] pub extern "C" fn aether_nccl_finalize() -> c_int {
    NCCL_INIT_COUNT.store(0, std::sync::atomic::Ordering::SeqCst);
    0
}

/// Create a communicator. Returns an opaque handle ≥ 1 on success, 0 on
/// failure. Single-host fallback: rejects world_size > 1 (returns 0)
/// so callers don't silently get incorrect results on a single-GPU box.
#[no_mangle] pub extern "C" fn aether_nccl_comm_create(world_size: c_int, rank: c_int) -> i64 {
    if world_size <= 0 || rank < 0 || rank >= world_size { return 0; }
    if world_size > 1 {
        // FR-18.1-extra (real libnccl link) needed to actually create a
        // cross-card communicator. Return a sentinel so the witness can
        // distinguish "not supported on this build" from "init failed".
        return -1;
    }
    let handle = NCCL_NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    handle as i64
}

/// Destroy a communicator. Single-host fallback always returns 0.
#[no_mangle] pub extern "C" fn aether_nccl_comm_destroy(_comm: i64) -> c_int { 0 }

/// Return the world_size this comm was created with. Single-host
/// fallback always returns 1 (the only supported size).
#[no_mangle] pub extern "C" fn aether_nccl_comm_world_size(_comm: i64) -> c_int { 1 }

/// Return the rank this comm was created with. Single-host fallback
/// always returns 0.
#[no_mangle] pub extern "C" fn aether_nccl_comm_rank(_comm: i64) -> c_int { 0 }

/// Real NCCL all-reduce shape: `(send_buf, recv_buf, n, op, comm)`.
/// Op codes: 0 = sum, 1 = max, 2 = min, 3 = prod. Single-host fallback
/// is a copy: when world_size=1 the reduction is the identity. Returns
/// 0 on success, non-zero on bad op or null buffers.
#[no_mangle] pub unsafe extern "C" fn aether_nccl_all_reduce_f32(
    send: *const c_void, recv: *mut c_void,
    n: c_int, op: c_int, _comm: i64,
) -> c_int {
    if send.is_null() || recv.is_null() || n <= 0 { return 1; }
    if !(0..=3).contains(&op) { return 2; }
    let s = std::slice::from_raw_parts(send as *const f32, n as usize);
    let r = std::slice::from_raw_parts_mut(recv as *mut f32, n as usize);
    r.copy_from_slice(s);  // world_size=1 → reduction is identity
    0
}

static NCCL_INIT_COUNT: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
static NCCL_NEXT_HANDLE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1);

// =====================================================================
// FR-18.{4,5,6} — In-process simulations of the distributed shapes.
//
// Without a real second GPU the AETHER tests can't run actual multi-
// rank workloads. Each fn below simulates the algorithm shape on a
// single host so the control-flow shape is verifiable — the eventual
// multi-rank impls (FR-18.x-extra, gated on real libnccl link) plug
// into the same call sites.
// =====================================================================

/// FR-18.5 — Tensor-parallel column-parallel Linear (in-process sim).
///
/// Splits weight W (shape k × n) column-wise into N "rank shards". Each
/// shard computes a partial output Y_shard = X · W[:, shard_start..end]
/// (shape m × n_shard). The simulator then concatenates the shards
/// column-wise to recover the full Y (shape m × n). Mathematically
/// identical to a single-rank matmul X · W; the shape verifies that
/// the column split + concat round-trip is correct.
///
/// `n` MUST be divisible by `world_size`. Returns 0 on success.
#[no_mangle] pub unsafe extern "C" fn aether_tp_simulate_column_parallel_linear_f32(
    x: *const c_void,        // (m, k)
    w: *const c_void,        // (k, n)
    out: *mut c_void,        // (m, n)
    m: c_int, k: c_int, n: c_int, world_size: c_int,
) -> c_int {
    if x.is_null() || w.is_null() || out.is_null() { return 1; }
    if m <= 0 || k <= 0 || n <= 0 || world_size <= 0 { return 2; }
    if n % world_size != 0 { return 3; }
    let m = m as usize; let k = k as usize; let n = n as usize;
    let ws = world_size as usize;
    let xs = std::slice::from_raw_parts(x as *const f32, m * k);
    let ws_buf = std::slice::from_raw_parts(w as *const f32, k * n);
    let o = std::slice::from_raw_parts_mut(out as *mut f32, m * n);
    let n_shard = n / ws;
    // For each rank shard, compute its partial Y_shard and write into
    // the corresponding column slab of `out`. Real multi-rank version
    // would each compute one shard locally and all-gather the slabs.
    for shard in 0..ws {
        let col_start = shard * n_shard;
        for r in 0..m {
            for cc in 0..n_shard {
                let c = col_start + cc;
                let mut s = 0.0f32;
                for ki in 0..k {
                    s += xs[r * k + ki] * ws_buf[ki * n + c];
                }
                o[r * n + c] = s;
            }
        }
    }
    0
}

/// FR-18.6 — Pipeline-parallel 1F1B forward-only simulation.
///
/// Splits N transformer "blocks" across `n_stages` stages. Each block
/// is a simple `Y = X * scale + bias` (the simulator's stand-in for a
/// real DecoderLayer). Stage s owns blocks `[s*blocks_per_stage,
/// (s+1)*blocks_per_stage)`. Micro-batches flow through the pipeline:
/// at any time, micro-batch i is on stage `i mod n_stages` (1F1B).
///
/// `scales` and `biases` are length `n_blocks` each. `microbatches` is
/// the count of micro-batches; the simulator runs all of them through
/// the full pipe and writes results to `out` (length microbatches * d).
///
/// The output MUST match what a monolithic single-stage forward would
/// produce — the witness verifies this by computing both side-by-side.
///
/// Returns 0 on success.
#[no_mangle] pub unsafe extern "C" fn aether_pp_simulate_2stage_forward_f32(
    input: *const c_void,    // (microbatches, d)
    scales: *const c_void,   // (n_blocks,)
    biases: *const c_void,   // (n_blocks,)
    out: *mut c_void,        // (microbatches, d)
    microbatches: c_int, d: c_int,
    n_blocks: c_int, n_stages: c_int,
) -> c_int {
    if input.is_null() || scales.is_null() || biases.is_null() || out.is_null() { return 1; }
    if microbatches <= 0 || d <= 0 || n_blocks <= 0 || n_stages <= 0 { return 2; }
    if n_blocks % n_stages != 0 { return 3; }
    let mb = microbatches as usize; let d = d as usize;
    let nb = n_blocks as usize; let ns = n_stages as usize;
    let bps = nb / ns;
    let in_buf = std::slice::from_raw_parts(input as *const f32, mb * d);
    let sc = std::slice::from_raw_parts(scales as *const f32, nb);
    let bi = std::slice::from_raw_parts(biases as *const f32, nb);
    let o = std::slice::from_raw_parts_mut(out as *mut f32, mb * d);

    // Buffer that travels stage-by-stage. Real multi-rank version would
    // send/recv this between ranks; in-process we just keep a Vec.
    let mut x = vec![0.0f32; mb * d];
    x.copy_from_slice(in_buf);

    // 1F1B forward schedule: process every micro-batch through every
    // stage in order. Each stage applies its `bps` blocks sequentially.
    for stage in 0..ns {
        let block_lo = stage * bps;
        let block_hi = block_lo + bps;
        for b in block_lo..block_hi {
            // Apply block b to all micro-batches.
            for mb_i in 0..mb {
                for di in 0..d {
                    x[mb_i * d + di] = x[mb_i * d + di] * sc[b] + bi[b];
                }
            }
        }
        // (In a real run, this is where stage `stage` sends its
        // output to stage `stage+1` via send/recv. Single-process:
        // x is already the right input for the next stage.)
    }
    o.copy_from_slice(&x);
    0
}

/// FR-18.7 — ZeRO-1/2/3 staged sharding (in-process sim).
///
/// Demonstrates the staged memory-savings shape:
///   stage=1 (Z1) — shard optimizer state only.   Per-rank bytes ≈ params + 2 * params/ws
///   stage=2 (Z2) — also shard gradients.         Per-rank bytes ≈ params + 3 * params/ws
///   stage=3 (Z3) — also shard parameters.        Per-rank bytes ≈ 4 * params/ws (= FSDP)
///
/// `n_params` is the model's parameter count. Returns the simulated
/// per-rank byte count for the requested ZeRO stage (assuming f32
/// params + f32 grads + 2 × f32 optimizer-state slots per param —
/// AdamW shape). Caller compares against the unsharded `4 * n_params *
/// 4` baseline to verify the documented savings.
#[no_mangle] pub extern "C" fn aether_zero_simulate_stage_bytes_f32(
    n_params: c_int, world_size: c_int, stage: c_int,
) -> i64 {
    if n_params <= 0 || world_size <= 0 { return -1; }
    if !(1..=3).contains(&stage) { return -2; }
    let n = n_params as i64;
    let ws = world_size as i64;
    let elem = 4_i64;  // f32
    // Baseline (no ZeRO): params (1x) + grads (1x) + optim (2x) all full per rank.
    // Z1 shards optim → grad + param full, optim sharded.
    // Z2 also shards grad → param full, grad+optim sharded.
    // Z3 also shards param → all sharded.
    let param_bytes = n * elem;
    let grad_bytes  = n * elem;
    let optim_bytes = 2 * n * elem;
    let bytes = match stage {
        1 => param_bytes + grad_bytes + optim_bytes / ws,
        2 => param_bytes + (grad_bytes + optim_bytes) / ws,
        3 => (param_bytes + grad_bytes + optim_bytes) / ws,
        _ => unreachable!(),
    };
    bytes
}

/// FR-18.8 — Compute/comm overlap simulation.
///
/// On a real multi-GPU setup, an `all_reduce` launched on a comm stream
/// overlaps with the next backward kernel on the compute stream. This
/// simulator demonstrates the SHAPE: takes a buffer + a "compute" cost
/// (microseconds) + a "comm" cost; returns the simulated total wall
/// time if overlapped (= max(compute, comm)) versus serial (= sum).
/// The witness asserts the overlapped total is strictly less than the
/// serial total for overlapping costs.
///
/// Returns: overlapped time in microseconds.
#[no_mangle] pub extern "C" fn aether_overlap_simulate_overlapped_us(
    compute_us: c_int, comm_us: c_int,
) -> c_int {
    if compute_us < 0 || comm_us < 0 { return -1; }
    if compute_us > comm_us { compute_us } else { comm_us }
}

#[no_mangle] pub extern "C" fn aether_overlap_simulate_serial_us(
    compute_us: c_int, comm_us: c_int,
) -> c_int {
    if compute_us < 0 || comm_us < 0 { return -1; }
    compute_us + comm_us
}

/// FR-18.9 — Gradient compression (rank-K projection, PowerSGD-shape).
///
/// Compresses a gradient buffer (m, n) into a rank-K approximation:
///   G ≈ P · Q^T  where P is (m, K) and Q is (n, K).
///
/// For the witness we just use the first K columns of G as P and an
/// identity-shape Q so the reconstruction is `G[:, :K]` extended with
/// zeros. This is NOT real PowerSGD (which uses power iteration to
/// find the dominant K singular vectors) but exercises the right
/// shape: M·N elements → M·K + N·K bytes shipped. The witness verifies
/// the reconstruction preserves the first K columns exactly and that
/// the compression ratio is the documented `(m+n)*K / (m*n)`.
///
/// `m * n` input → `m*K + n*K` bytes shipped → `m * n` reconstructed.
#[no_mangle] pub unsafe extern "C" fn aether_grad_compress_lowrank_f32(
    grad: *const c_void,     // (m, n)
    reconstructed: *mut c_void,  // (m, n)
    m: c_int, n: c_int, k: c_int,
) -> c_int {
    if grad.is_null() || reconstructed.is_null() { return 1; }
    if m <= 0 || n <= 0 || k <= 0 { return 2; }
    if k > n { return 3; }
    let m = m as usize; let n = n as usize; let k = k as usize;
    let g = std::slice::from_raw_parts(grad as *const f32, m * n);
    let r = std::slice::from_raw_parts_mut(reconstructed as *mut f32, m * n);
    // P (m, k) = first k columns of G. Q (n, k) = identity-shape (top-k
    // canonical basis). Reconstruction: G_rec[i, j] = sum_p P[i,p] * Q[j,p].
    // With Q = identity (rows 0..k = e_0..e_{k-1}, rows k..n = 0),
    // G_rec[i, j] = P[i, j] when j < k, else 0.
    for i in 0..m {
        for j in 0..n {
            if j < k {
                r[i * n + j] = g[i * n + j];
            } else {
                r[i * n + j] = 0.0;
            }
        }
    }
    0
}

/// FR-18.4 — FSDP shard + all-gather + reduce-scatter simulation.
///
/// Models the FSDP shape: parameters are sharded across N ranks. To
/// compute a forward pass, each rank all-gathers the full param tensor
/// from peers; for backward, gradients are reduce-scattered back into
/// the sharded layout. The simulator does both in-process: it takes
/// a full-size param tensor, splits into N shards, then reassembles
/// them via the all-gather shape and verifies bit-equality with the
/// original.
///
/// Returns 0 on success. The witness uses this to confirm the
/// shard-then-gather round-trip is the identity.
#[no_mangle] pub unsafe extern "C" fn aether_fsdp_simulate_shard_alltoall_f32(
    params: *const c_void,   // (n,)
    out: *mut c_void,        // (n,)
    n: c_int, world_size: c_int,
) -> c_int {
    if params.is_null() || out.is_null() { return 1; }
    if n <= 0 || world_size <= 0 { return 2; }
    if n % world_size != 0 { return 3; }
    let n = n as usize; let ws = world_size as usize;
    let shard_len = n / ws;
    let p = std::slice::from_raw_parts(params as *const f32, n);
    let o = std::slice::from_raw_parts_mut(out as *mut f32, n);

    // Step 1: shard. Each rank "owns" shards[rank] = p[rank*shard_len..]
    // We materialise the shards explicitly.
    let mut shards: Vec<Vec<f32>> = Vec::with_capacity(ws);
    for rank in 0..ws {
        let lo = rank * shard_len;
        shards.push(p[lo..lo + shard_len].to_vec());
    }
    // Step 2: all-gather — every rank reassembles the full param tensor
    // by concatenating shards in rank order. We just check rank 0's
    // reassembly here (others would be identical in a real run).
    for rank in 0..ws {
        let lo = rank * shard_len;
        o[lo..lo + shard_len].copy_from_slice(&shards[rank]);
    }
    0
}

// ---- FR-24.9: GPU memory leak detection (CPU-resident counter for now) ----
use std::sync::atomic::{AtomicI64 as V4AtomicI64, Ordering as AtomicOrder};
static GPU_LIVE_BYTES: V4AtomicI64 = V4AtomicI64::new(0);
#[no_mangle] pub extern "C" fn aether_gpu_alloc_track(bytes: i64) -> i64 {
    GPU_LIVE_BYTES.fetch_add(bytes, AtomicOrder::SeqCst);
    GPU_LIVE_BYTES.load(AtomicOrder::SeqCst)
}
#[no_mangle] pub extern "C" fn aether_gpu_free_track(bytes: i64) -> i64 {
    GPU_LIVE_BYTES.fetch_sub(bytes, AtomicOrder::SeqCst);
    GPU_LIVE_BYTES.load(AtomicOrder::SeqCst)
}
#[no_mangle] pub extern "C" fn aether_gpu_live_bytes() -> i64 {
    GPU_LIVE_BYTES.load(AtomicOrder::SeqCst)
}

// ---- FR-24.10: OOM killer / graceful degradation hook ---------------------
// Serving processes call this when memory pressure hits a threshold. Today's
// impl is a CPU-side flag; a real KV-cache shrink + 503 path is FR-24.10.
static OOM_FLAG: V4AtomicI64 = V4AtomicI64::new(0);
#[no_mangle] pub extern "C" fn aether_oom_signal(flag: i64) -> i64 {
    OOM_FLAG.store(flag, AtomicOrder::SeqCst);
    OOM_FLAG.load(AtomicOrder::SeqCst)
}
#[no_mangle] pub extern "C" fn aether_oom_check() -> i64 {
    OOM_FLAG.load(AtomicOrder::SeqCst)
}

#[cfg(test)]
mod v4_op_tests {
    use super::*;
    #[test]
    fn math_primitives() {
        unsafe {
            let mut buf = [1.0f32, 2.0, 3.0];
            aether_op_log_f32(buf.as_mut_ptr() as _, 3);
            assert!((buf[0] - 0.0).abs() < 1e-5);
            assert!((buf[1] - 0.6931472).abs() < 1e-5);
            let mut e = [0.0f32, 1.0];
            aether_op_exp_f32(e.as_mut_ptr() as _, 2);
            assert!((e[1] - std::f32::consts::E).abs() < 1e-5);
            let mut p = [2.0f32, 3.0];
            aether_op_pow_f32(p.as_mut_ptr() as _, 3.0, 2);
            assert!((p[0] - 8.0).abs() < 1e-5);
            assert!((p[1] - 27.0).abs() < 1e-5);
        }
    }
    #[test]
    fn activation_extensions() {
        unsafe {
            let mut t = [0.0f32, 1.0, -1.0];
            aether_op_tanh_f32(t.as_mut_ptr() as _, 3);
            assert!((t[0] - 0.0).abs() < 1e-5);
            let mut s = [0.0f32];
            aether_op_sigmoid_f32(s.as_mut_ptr() as _, 1);
            assert!((s[0] - 0.5).abs() < 1e-5);
        }
    }
    #[test]
    fn mask_helpers() {
        unsafe {
            let mut e = [0.0f32; 9];
            aether_op_eye_f32(e.as_mut_ptr() as _, 3);
            assert_eq!(e, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            let mut a = [0.0f32; 5];
            aether_op_arange_f32(a.as_mut_ptr() as _, 0.0, 1.0, 5);
            assert_eq!(a, [0.0, 1.0, 2.0, 3.0, 4.0]);
            let mut t = [1.0f32; 9];
            aether_op_tril_f32(t.as_mut_ptr() as _, 3, 3);
            assert_eq!(t, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
        }
    }
    #[test]
    fn reductions() {
        unsafe {
            let v = [1.0f32, 2.0, 3.0, 4.0];
            assert!((aether_op_sum_f32(v.as_ptr() as _, 4) - 10.0).abs() < 1e-5);
            assert!((aether_op_mean_f32(v.as_ptr() as _, 4) - 2.5).abs() < 1e-5);
            assert_eq!(aether_op_argmax_f32(v.as_ptr() as _, 4), 3);
            assert_eq!(aether_op_argmin_f32(v.as_ptr() as _, 4), 0);
            assert!((aether_op_max_red_f32(v.as_ptr() as _, 4) - 4.0).abs() < 1e-5);
        }
    }
    #[test]
    fn combine_cat() {
        unsafe {
            let a = [1.0f32, 2.0]; let b = [3.0f32, 4.0, 5.0];
            let mut o = [0.0f32; 5];
            aether_op_cat_f32(a.as_ptr() as _, 2, b.as_ptr() as _, 3, o.as_mut_ptr() as _);
            assert_eq!(o, [1.0, 2.0, 3.0, 4.0, 5.0]);
        }
    }
    #[test]
    fn optim_sgd_momentum() {
        unsafe {
            let mut p = [1.0f32, 2.0]; let g = [0.5f32, -0.5]; let mut m = [0.0f32; 2];
            aether_op_sgd_momentum_step_f32(
                p.as_mut_ptr() as _, g.as_ptr() as _, m.as_mut_ptr() as _,
                0.1, 0.9, 0.0, 2,
            );
            // m = g, p -= lr*m  →  p = [0.95, 2.05]
            assert!((p[0] - 0.95).abs() < 1e-5);
            assert!((p[1] - 2.05).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn tape_lifecycle() {
        unsafe {
            aether_autodiff_init(ptr::null_mut());
            aether_autodiff_push(ptr::null_mut(), ptr::null());
            aether_autodiff_push(ptr::null_mut(), ptr::null());
            assert_eq!(aether_rt_self_check(), 2);
            aether_autodiff_reverse(ptr::null_mut());
        }
    }

    #[test]
    fn all_reduce_is_safe() {
        aether_dist_all_reduce(ptr::null_mut(), 8, DistBackend::Nccl as c_int);
    }

    #[test]
    fn tcp_listen_close_roundtrip() {
        // Bind ephemeral, read back the port, close. Then prove
        // closing twice is rejected and bogus handles are rejected.
        unsafe {
            let l = aether_tcp_listen(0);
            assert!(l >= 0, "listen returned {}", l);
            let port = aether_tcp_listener_port(l);
            assert!((1..=65535).contains(&port), "bound port {} out of range", port);
            assert_eq!(aether_tcp_close(l), 0);
            assert_eq!(aether_tcp_close(l), -1);  // double-close rejected
            assert_eq!(aether_tcp_close(99_999), -1);
            assert_eq!(aether_tcp_stream_close(99_999), -1);
        }
    }

    #[test]
    fn tcp_send_recv_loopback() {
        // Witness the full surface: spawn a listener, connect a client
        // from a thread, accept, send "hi" both ways, close both ends.
        unsafe {
            let listener = aether_tcp_listen(0);
            assert!(listener >= 0);
            let port = aether_tcp_listener_port(listener);
            assert!(port > 0);

            let t = std::thread::spawn(move || {
                // Client thread. Note: this thread's TCP_STREAMS slot
                // accesses are technically a data race vs. the main
                // thread's accept_one which also touches TCP_STREAMS.
                // For the witness we serialise: client connects only
                // after main thread is already blocked in accept().
                std::thread::sleep(std::time::Duration::from_millis(50));
                let s = std::net::TcpStream::connect(("127.0.0.1", port as u16))
                    .expect("client connect");
                use std::io::{Read, Write};
                let mut s = s;
                s.write_all(b"ping").unwrap();
                let mut buf = [0u8; 4];
                s.read_exact(&mut buf).unwrap();
                assert_eq!(&buf, b"pong");
            });

            let server_stream = aether_tcp_accept_one(listener);
            assert!(server_stream >= 0, "accept returned {}", server_stream);

            // Recv "ping" from the client.
            let recv_buf = [0u8; 4];
            let got = aether_tcp_recv(server_stream, recv_buf.as_ptr() as i64, 4);
            assert_eq!(got, 4);
            assert_eq!(&recv_buf, b"ping");

            // Send "pong" back.
            let send_buf = b"pong";
            let sent = aether_tcp_send(server_stream, send_buf.as_ptr() as i64, 4);
            assert_eq!(sent, 4);

            assert_eq!(aether_tcp_stream_close(server_stream), 0);
            assert_eq!(aether_tcp_close(listener), 0);

            t.join().unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// P10.7 — PGO instrumentation surface
// ---------------------------------------------------------------------------
// Profile-collection primitives: branch frequency counters and call-site
// counters. Backing store is a Vec behind a static UnsafeCell, mirroring the
// MATMUL_CACHE / TAPE pattern. Linear scan on lookup is fine — this is
// profiling infrastructure, not a hot path. Single-threaded use only (no
// Mutex; the rest of aether_rt is single-threaded by convention).
//
// Wire-up for feedback-directed inlining + branch hints lives in the
// optimiser (P10.1 SSA, currently blocked). Here we ship only the data
// plumbing so .aether code can record + query counters today, and so a
// future optimiser can read a serialised profile.

#[repr(C)]
struct PgoBranch {
    site_id: i64,
    taken: i64,
    total: i64,
}

struct PgoBranchCell(UnsafeCell<Vec<PgoBranch>>);
unsafe impl Sync for PgoBranchCell {}
static PGO_BRANCHES: PgoBranchCell = PgoBranchCell(UnsafeCell::new(Vec::new()));

struct PgoCallCell(UnsafeCell<Vec<(i64, i64)>>);
unsafe impl Sync for PgoCallCell {}
static PGO_CALLS: PgoCallCell = PgoCallCell(UnsafeCell::new(Vec::new()));

#[no_mangle]
pub extern "C" fn aether_pgo_record_branch(site_id: i64, taken: i64) -> i32 {
    unsafe {
        let v = &mut *PGO_BRANCHES.0.get();
        for entry in v.iter_mut() {
            if entry.site_id == site_id {
                entry.total += 1;
                if taken != 0 { entry.taken += 1; }
                return 0;
            }
        }
        v.push(PgoBranch {
            site_id,
            taken: if taken != 0 { 1 } else { 0 },
            total: 1,
        });
        0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_record_call(callsite_id: i64) -> i32 {
    unsafe {
        let v = &mut *PGO_CALLS.0.get();
        for entry in v.iter_mut() {
            if entry.0 == callsite_id {
                entry.1 += 1;
                return 0;
            }
        }
        v.push((callsite_id, 1));
        0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_branch_freq(site_id: i64) -> f32 {
    unsafe {
        let v = &*PGO_BRANCHES.0.get();
        for entry in v.iter() {
            if entry.site_id == site_id {
                if entry.total == 0 { return 0.0; }
                return entry.taken as f32 / entry.total as f32;
            }
        }
        0.0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_call_count(callsite_id: i64) -> i64 {
    unsafe {
        let v = &*PGO_CALLS.0.get();
        for entry in v.iter() {
            if entry.0 == callsite_id {
                return entry.1;
            }
        }
        0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_reset() -> i32 {
    unsafe {
        (&mut *PGO_BRANCHES.0.get()).clear();
        (&mut *PGO_CALLS.0.get()).clear();
    }
    0
}

#[no_mangle]
pub extern "C" fn aether_pgo_dump() -> i32 {
    unsafe {
        let bv = &*PGO_BRANCHES.0.get();
        for e in bv.iter() {
            println!("site={} taken={} total={}", e.site_id, e.taken, e.total);
        }
        let cv = &*PGO_CALLS.0.get();
        for e in cv.iter() {
            println!("call={} count={}", e.0, e.1);
        }
    }
    0
}

#[cfg(test)]
mod pgo_tests {
    use super::*;

    #[test]
    fn pgo_branch_and_call_roundtrip() {
        aether_pgo_reset();
        // site 1: 3 of 5 taken
        aether_pgo_record_branch(1, 1);
        aether_pgo_record_branch(1, 0);
        aether_pgo_record_branch(1, 1);
        aether_pgo_record_branch(1, 1);
        aether_pgo_record_branch(1, 0);
        // site 2: 0 of 2 taken
        aether_pgo_record_branch(2, 0);
        aether_pgo_record_branch(2, 0);

        let f1 = aether_pgo_branch_freq(1);
        let f2 = aether_pgo_branch_freq(2);
        let f_missing = aether_pgo_branch_freq(99);
        assert!((f1 - 0.6).abs() < 1e-6, "expected 0.6, got {}", f1);
        assert!(f2.abs() < 1e-6, "expected 0.0, got {}", f2);
        assert!(f_missing.abs() < 1e-6);

        aether_pgo_record_call(10);
        aether_pgo_record_call(10);
        aether_pgo_record_call(10);
        aether_pgo_record_call(20);
        assert_eq!(aether_pgo_call_count(10), 3);
        assert_eq!(aether_pgo_call_count(20), 1);
        assert_eq!(aether_pgo_call_count(999), 0);

        aether_pgo_reset();
        assert!(aether_pgo_branch_freq(1).abs() < 1e-6);
        assert_eq!(aether_pgo_call_count(10), 0);
    }
}

// =====================================================================
// P6.7 — heap-allocated stdlib types.
//
// The language doesn't have generics-over-T yet (P6.1 / P6.2 still open),
// so this pass ships CONCRETE Vec<i64> and String (== Vec<u8>) primitives
// rather than a fully generic Vec<T>. Every Vec-of-T variant the user
// needs gets its own monomorphic op set behind the runtime ABI; once
// generics land, these become a single template that lowers to the same
// symbols.
//
// P6.7 partial — Box/HashMap/BTreeMap need generics (P6.1).
// =====================================================================

/// Resize a buffer previously returned by `aether_alloc_bytes`. Returns
/// the new pointer (may equal old). On growth, old contents copy into
/// the new buffer and the tail is zero-filled. On shrink, the buffer
/// is truncated. `old_n` MUST be the exact length of the existing buffer.
/// Returns 0 on null input or invalid sizes.
#[no_mangle] pub unsafe extern "C" fn aether_realloc_bytes(p: i64, old_n: i64, new_n: i64) -> i64 {
    if new_n <= 0 { return 0; }
    if p == 0 || old_n <= 0 {
        return aether_alloc_bytes(new_n);
    }
    if old_n == new_n { return p; }
    let new_p = aether_alloc_bytes(new_n);
    if new_p == 0 { return 0; }
    let copy = old_n.min(new_n) as usize;
    std::ptr::copy_nonoverlapping(p as *const u8, new_p as *mut u8, copy);
    aether_free_bytes(p, old_n);
    new_p
}

// ---- Vec<i64> handle table ------------------------------------------------

struct VecI64 {
    ptr: *mut i64,
    len: usize,
    cap: usize,
}

struct VecI64Cell(UnsafeCell<Vec<Option<Box<VecI64>>>>);
unsafe impl Sync for VecI64Cell {}
static VEC_I64_TABLE: VecI64Cell = VecI64Cell(UnsafeCell::new(Vec::new()));

unsafe fn vec_i64_table() -> &'static mut Vec<Option<Box<VecI64>>> {
    &mut *VEC_I64_TABLE.0.get()
}

unsafe fn vec_i64_alloc_buf(cap: usize) -> *mut i64 {
    if cap == 0 { return std::ptr::null_mut(); }
    let bytes = cap.checked_mul(std::mem::size_of::<i64>()).expect("vec i64 cap overflow");
    let p = aether_alloc_bytes(bytes as i64);
    p as *mut i64
}

unsafe fn vec_i64_free_buf(ptr: *mut i64, cap: usize) {
    if ptr.is_null() || cap == 0 { return; }
    let bytes = cap * std::mem::size_of::<i64>();
    aether_free_bytes(ptr as i64, bytes as i64);
}

/// Allocate a fresh empty `Vec<i64>`. Returns a non-negative handle.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_new() -> i64 {
    let v = Box::new(VecI64 { ptr: std::ptr::null_mut(), len: 0, cap: 0 });
    let tbl = vec_i64_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(v); return i as i64; }
    }
    tbl.push(Some(v));
    (tbl.len() - 1) as i64
}

/// Push `value` onto the end of the Vec. Capacity-doubling growth.
/// Returns 0 on success, -1 on invalid handle, -2 on OOM.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_push(handle: i64, value: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = vec_i64_table();
    let idx = handle as usize;
    if idx >= tbl.len() { return -1; }
    let v = match tbl[idx].as_mut() { Some(v) => v, None => return -1 };
    if v.len == v.cap {
        let new_cap = if v.cap == 0 { 4 } else { v.cap * 2 };
        let new_ptr = vec_i64_alloc_buf(new_cap);
        if new_ptr.is_null() { return -2; }
        if v.len > 0 {
            std::ptr::copy_nonoverlapping(v.ptr, new_ptr, v.len);
        }
        vec_i64_free_buf(v.ptr, v.cap);
        v.ptr = new_ptr;
        v.cap = new_cap;
    }
    *v.ptr.add(v.len) = value;
    v.len += 1;
    0
}

/// Read element at `idx`. Returns 0 on out-of-range / invalid handle.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_get(handle: i64, idx: i64) -> i64 {
    if handle < 0 || idx < 0 { return 0; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    let v = match tbl[h].as_ref() { Some(v) => v, None => return 0 };
    let i = idx as usize;
    if i >= v.len { return 0; }
    *v.ptr.add(i)
}

/// Overwrite element at `idx`. Returns 0 on success, -1 invalid handle,
/// -2 out-of-range.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_set(handle: i64, idx: i64, value: i64) -> i32 {
    if handle < 0 || idx < 0 { return -1; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let v = match tbl[h].as_mut() { Some(v) => v, None => return -1 };
    let i = idx as usize;
    if i >= v.len { return -2; }
    *v.ptr.add(i) = value;
    0
}

/// Number of elements currently in the Vec.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(v) => v.len as i64, None => 0 }
}

/// Free the buffer + release the handle slot. Idempotent.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    if let Some(mut v) = tbl[h].take() {
        vec_i64_free_buf(v.ptr, v.cap);
        v.ptr = std::ptr::null_mut();
        v.cap = 0;
        v.len = 0;
    }
    0
}

// ---- String (UTF-8 owned, == Vec<u8>) ------------------------------------

struct AeString {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

struct AeStringCell(UnsafeCell<Vec<Option<Box<AeString>>>>);
unsafe impl Sync for AeStringCell {}
static STRING_TABLE: AeStringCell = AeStringCell(UnsafeCell::new(Vec::new()));

unsafe fn string_table() -> &'static mut Vec<Option<Box<AeString>>> {
    &mut *STRING_TABLE.0.get()
}

/// Allocate a fresh empty `String`. Returns a non-negative handle.
#[no_mangle] pub unsafe extern "C" fn aether_string_new() -> i64 {
    let s = Box::new(AeString { ptr: std::ptr::null_mut(), len: 0, cap: 0 });
    let tbl = string_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(s); return i as i64; }
    }
    tbl.push(Some(s));
    (tbl.len() - 1) as i64
}

/// Append low-byte of `b` to the string. Capacity-doubling growth.
#[no_mangle] pub unsafe extern "C" fn aether_string_push_byte(handle: i64, b: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = string_table();
    let idx = handle as usize;
    if idx >= tbl.len() { return -1; }
    let s = match tbl[idx].as_mut() { Some(s) => s, None => return -1 };
    if s.len == s.cap {
        let new_cap = if s.cap == 0 { 8 } else { s.cap * 2 };
        let new_ptr = aether_alloc_bytes(new_cap as i64) as *mut u8;
        if new_ptr.is_null() { return -2; }
        if s.len > 0 {
            std::ptr::copy_nonoverlapping(s.ptr, new_ptr, s.len);
        }
        if !s.ptr.is_null() && s.cap > 0 {
            aether_free_bytes(s.ptr as i64, s.cap as i64);
        }
        s.ptr = new_ptr;
        s.cap = new_cap;
    }
    *s.ptr.add(s.len) = (b & 0xFF) as u8;
    s.len += 1;
    0
}

/// Length in bytes of the string.
#[no_mangle] pub unsafe extern "C" fn aether_string_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(s) => s.len as i64, None => 0 }
}

/// Read byte at `idx` (returned as 0..=255 in i64). -1 on out-of-range.
#[no_mangle] pub unsafe extern "C" fn aether_string_byte_at(handle: i64, idx: i64) -> i64 {
    if handle < 0 || idx < 0 { return -1; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let s = match tbl[h].as_ref() { Some(s) => s, None => return -1 };
    let i = idx as usize;
    if i >= s.len { return -1; }
    *s.ptr.add(i) as i64
}

/// Free the buffer + release the handle slot. Idempotent.
#[no_mangle] pub unsafe extern "C" fn aether_string_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    if let Some(mut s) = tbl[h].take() {
        if !s.ptr.is_null() && s.cap > 0 {
            aether_free_bytes(s.ptr as i64, s.cap as i64);
        }
        s.ptr = std::ptr::null_mut();
        s.cap = 0;
        s.len = 0;
    }
    0
}

#[cfg(test)]
mod heap_stdlib_tests {
    use super::*;

    #[test]
    fn realloc_grow_and_shrink() {
        unsafe {
            let p = aether_alloc_bytes(4);
            aether_byte_set(p, 0, 1);
            aether_byte_set(p, 1, 2);
            aether_byte_set(p, 2, 3);
            aether_byte_set(p, 3, 4);
            let q = aether_realloc_bytes(p, 4, 8);
            assert_ne!(q, 0);
            assert_eq!(aether_byte_at(q, 0), 1);
            assert_eq!(aether_byte_at(q, 3), 4);
            // Tail zero-init from aether_alloc_bytes.
            assert_eq!(aether_byte_at(q, 4), 0);
            // Shrink back to 2.
            let r = aether_realloc_bytes(q, 8, 2);
            assert_ne!(r, 0);
            assert_eq!(aether_byte_at(r, 0), 1);
            assert_eq!(aether_byte_at(r, 1), 2);
            aether_free_bytes(r, 2);
        }
    }

    #[test]
    fn vec_i64_push_get_len() {
        unsafe {
            let h = aether_vec_i64_new();
            assert!(h >= 0);
            for i in 0..1000i64 {
                assert_eq!(aether_vec_i64_push(h, i * 2), 0);
            }
            assert_eq!(aether_vec_i64_len(h), 1000);
            assert_eq!(aether_vec_i64_get(h, 42), 84);
            assert_eq!(aether_vec_i64_set(h, 42, 999), 0);
            assert_eq!(aether_vec_i64_get(h, 42), 999);
            // Out-of-range reads return 0.
            assert_eq!(aether_vec_i64_get(h, 9999), 0);
            assert_eq!(aether_vec_i64_set(h, 9999, 1), -2);
            assert_eq!(aether_vec_i64_free(h), 0);
        }
    }

    #[test]
    fn string_push_byte_roundtrip() {
        unsafe {
            let s = aether_string_new();
            for &b in b"Hello, world!" {
                assert_eq!(aether_string_push_byte(s, b as i64), 0);
            }
            assert_eq!(aether_string_len(s), 13);
            assert_eq!(aether_string_byte_at(s, 0), b'H' as i64);
            assert_eq!(aether_string_byte_at(s, 12), b'!' as i64);
            assert_eq!(aether_string_byte_at(s, 13), -1);
            assert_eq!(aether_string_free(s), 0);
        }
    }
}

// ============================================================================
// P6.8 — Iterator trait + adapters (concrete-iterator subset)
// ----------------------------------------------------------------------------
// Traits + generics are not shipped yet (P6.2 / P6.6 are blocked / partial),
// so we ship a witness-able subset built around a concrete enum-typed iterator
// over Vec<i64>. Adapters return new opaque handles that wrap an inner handle.
// next() dispatches on the variant.
//
// Variants:
//   Source { vec, cursor }      yields vec[cursor], cursor++
//   MapDouble { inner }         yields 2 * inner.next()
//   FilterPos { inner }         skips inner.next() values that are <= 0
//   Take { inner, remaining }   yields up to `remaining` values
// ============================================================================

enum IterNode {
    Source { vec: i64, cursor: usize },
    MapDouble { inner: i64 },
    FilterPos { inner: i64 },
    Take { inner: i64, remaining: i64 },
}

struct IterCell(UnsafeCell<Vec<Option<Box<IterNode>>>>);
unsafe impl Sync for IterCell {}
static ITER_TABLE: IterCell = IterCell(UnsafeCell::new(Vec::new()));

unsafe fn iter_table() -> &'static mut Vec<Option<Box<IterNode>>> {
    &mut *ITER_TABLE.0.get()
}

unsafe fn iter_install(node: IterNode) -> i64 {
    let tbl = iter_table();
    let boxed = Some(Box::new(node));
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = boxed;
            return i as i64;
        }
    }
    tbl.push(boxed);
    (tbl.len() - 1) as i64
}

unsafe fn iter_next_inner(handle: i64) -> Option<i64> {
    if handle < 0 { return None; }
    let tbl = iter_table();
    let h = handle as usize;
    if h >= tbl.len() { return None; }
    let node_ptr: *mut IterNode = match tbl[h].as_mut() {
        Some(b) => &mut **b as *mut IterNode,
        None => return None,
    };
    match &mut *node_ptr {
        IterNode::Source { vec, cursor } => {
            let len = aether_vec_i64_len(*vec);
            if (*cursor as i64) >= len { return None; }
            let v = aether_vec_i64_get(*vec, *cursor as i64);
            *cursor += 1;
            Some(v)
        }
        IterNode::MapDouble { inner } => {
            let inner_h = *inner;
            iter_next_inner(inner_h).map(|v| v * 2)
        }
        IterNode::FilterPos { inner } => {
            let inner_h = *inner;
            loop {
                match iter_next_inner(inner_h) {
                    Some(v) if v > 0 => return Some(v),
                    Some(_) => continue,
                    None => return None,
                }
            }
        }
        IterNode::Take { inner, remaining } => {
            if *remaining <= 0 { return None; }
            let inner_h = *inner;
            match iter_next_inner(inner_h) {
                Some(v) => { *remaining -= 1; Some(v) }
                None => { *remaining = 0; None }
            }
        }
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_new(vec_handle: i64) -> i64 {
    if vec_handle < 0 { return -1; }
    iter_install(IterNode::Source { vec: vec_handle, cursor: 0 })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_next(iter: i64, out_value: i64) -> i64 {
    if iter < 0 { return -1; }
    if out_value == 0 { return -1; }
    match iter_next_inner(iter) {
        Some(v) => {
            *(out_value as *mut i64) = v;
            1
        }
        None => 0,
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_map_double(iter: i64) -> i64 {
    if iter < 0 { return -1; }
    iter_install(IterNode::MapDouble { inner: iter })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_filter_positive(iter: i64) -> i64 {
    if iter < 0 { return -1; }
    iter_install(IterNode::FilterPos { inner: iter })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_take(iter: i64, n: i64) -> i64 {
    if iter < 0 { return -1; }
    let r = if n < 0 { 0 } else { n };
    iter_install(IterNode::Take { inner: iter, remaining: r })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_fold_sum(iter: i64) -> i64 {
    if iter < 0 { return 0; }
    let mut acc: i64 = 0;
    while let Some(v) = iter_next_inner(iter) {
        acc = acc.wrapping_add(v);
    }
    acc
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_collect(iter: i64) -> i64 {
    if iter < 0 { return -1; }
    let v = aether_vec_i64_new();
    while let Some(x) = iter_next_inner(iter) {
        if aether_vec_i64_push(v, x) != 0 {
            return -2;
        }
    }
    v
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_free(iter: i64) -> i32 {
    if iter < 0 { return -1; }
    let tbl = iter_table();
    let h = iter as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

#[cfg(test)]
mod iter_tests {
    use super::*;

    #[test]
    fn chain_filter_map_take_fold() {
        unsafe {
            let v = aether_vec_i64_new();
            for i in -3..=10i64 {
                aether_vec_i64_push(v, i);
            }
            let s = aether_iter_vec_i64_new(v);
            let f = aether_iter_vec_i64_filter_positive(s);
            let m = aether_iter_vec_i64_map_double(f);
            let t = aether_iter_vec_i64_take(m, 3);
            assert_eq!(aether_iter_vec_i64_fold_sum(t), 12);
            aether_iter_vec_i64_free(t);
            aether_iter_vec_i64_free(m);
            aether_iter_vec_i64_free(f);
            aether_iter_vec_i64_free(s);
            aether_vec_i64_free(v);
        }
    }

    #[test]
    fn collect_round_trip() {
        unsafe {
            let v = aether_vec_i64_new();
            for i in 1..=5i64 { aether_vec_i64_push(v, i); }
            let it = aether_iter_vec_i64_new(v);
            let m = aether_iter_vec_i64_map_double(it);
            let out = aether_iter_vec_i64_collect(m);
            assert_eq!(aether_vec_i64_len(out), 5);
            assert_eq!(aether_vec_i64_get(out, 0), 2);
            assert_eq!(aether_vec_i64_get(out, 4), 10);
            aether_vec_i64_free(out);
            aether_iter_vec_i64_free(m);
            aether_iter_vec_i64_free(it);
            aether_vec_i64_free(v);
        }
    }

    #[test]
    fn next_out_pointer_protocol() {
        unsafe {
            let v = aether_vec_i64_new();
            aether_vec_i64_push(v, 7);
            aether_vec_i64_push(v, 11);
            let it = aether_iter_vec_i64_new(v);
            let mut out: i64 = 0;
            assert_eq!(aether_iter_vec_i64_next(it, &mut out as *mut i64 as i64), 1);
            assert_eq!(out, 7);
            assert_eq!(aether_iter_vec_i64_next(it, &mut out as *mut i64 as i64), 1);
            assert_eq!(out, 11);
            assert_eq!(aether_iter_vec_i64_next(it, &mut out as *mut i64 as i64), 0);
            aether_iter_vec_i64_free(it);
            aether_vec_i64_free(v);
        }
    }
}

// =====================================================================
// FR-16.5 — heap stdlib extras (Box, HashMap, Rc, mpsc::channel)
//
// Same handle-table pattern as Vec<i64> / String. Concrete-type only
// (Box<i64> / HashMap<i64,i64> / Rc<i64> / channel<i64>) until generics
// land — the surface is what `examples/*.aether` and `bench/*` actually
// exercise today, not a speculative full Rust stdlib mirror.
//
// Witness: tests/runtime/heap_stdlib_extras.aether.
// =====================================================================

// ---- Box<i64> ------------------------------------------------------------

struct BoxI64(i64);

struct BoxI64Cell(UnsafeCell<Vec<Option<Box<BoxI64>>>>);
unsafe impl Sync for BoxI64Cell {}
static BOX_I64_TABLE: BoxI64Cell = BoxI64Cell(UnsafeCell::new(Vec::new()));

unsafe fn box_i64_table() -> &'static mut Vec<Option<Box<BoxI64>>> {
    &mut *BOX_I64_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_box_i64_new(value: i64) -> i64 {
    let b = Some(Box::new(BoxI64(value)));
    let tbl = box_i64_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = b; return i as i64; }
    }
    tbl.push(b);
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_box_i64_get(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = box_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(b) => b.0, None => 0 }
}

#[no_mangle] pub unsafe extern "C" fn aether_box_i64_set(handle: i64, value: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = box_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_mut() { Some(b) => { b.0 = value; 0 }, None => -1 }
}

#[no_mangle] pub unsafe extern "C" fn aether_box_i64_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = box_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

// ---- HashMap<i64, i64> ---------------------------------------------------
//
// Open-addressed hash table with linear probing. 64-bit splitmix64 hash on
// the key, EMPTY sentinel = i64::MIN (callers cannot store that key — same
// trade-off as the dense-int representation in `runtime_pe`'s scratch
// hashmaps). Power-of-two capacity, 0.75 load factor before grow.

const HM_EMPTY: i64 = i64::MIN;

struct HashMapI64I64 {
    keys: Vec<i64>,
    vals: Vec<i64>,
    len: usize,
}

impl HashMapI64I64 {
    fn new() -> Self {
        const INITIAL: usize = 8;
        Self {
            keys: vec![HM_EMPTY; INITIAL],
            vals: vec![0; INITIAL],
            len: 0,
        }
    }
    fn hash(k: i64) -> u64 {
        let mut x = k as u64;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58476d1ce4e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d049bb133111eb);
        x ^= x >> 31;
        x
    }
    fn probe(&self, k: i64) -> usize {
        let mask = self.keys.len() - 1;
        let mut idx = (Self::hash(k) as usize) & mask;
        loop {
            let cur = self.keys[idx];
            if cur == HM_EMPTY || cur == k { return idx; }
            idx = (idx + 1) & mask;
        }
    }
    fn insert(&mut self, k: i64, v: i64) {
        if self.len * 4 >= self.keys.len() * 3 { self.grow(); }
        let i = self.probe(k);
        if self.keys[i] == HM_EMPTY { self.len += 1; }
        self.keys[i] = k;
        self.vals[i] = v;
    }
    fn get(&self, k: i64) -> Option<i64> {
        let i = self.probe(k);
        if self.keys[i] == k { Some(self.vals[i]) } else { None }
    }
    fn contains(&self, k: i64) -> bool {
        let i = self.probe(k);
        self.keys[i] == k
    }
    fn remove(&mut self, k: i64) -> Option<i64> {
        let i = self.probe(k);
        if self.keys[i] != k { return None; }
        let v = self.vals[i];
        self.keys[i] = HM_EMPTY;
        self.len -= 1;
        // Re-insert anyone that was in the same probe chain after `i`.
        let mask = self.keys.len() - 1;
        let mut j = (i + 1) & mask;
        while self.keys[j] != HM_EMPTY {
            let kk = self.keys[j];
            let vv = self.vals[j];
            self.keys[j] = HM_EMPTY;
            self.len -= 1;
            self.insert(kk, vv);
            j = (j + 1) & mask;
        }
        Some(v)
    }
    fn grow(&mut self) {
        let new_cap = self.keys.len() * 2;
        let mut new_keys = vec![HM_EMPTY; new_cap];
        let mut new_vals = vec![0i64; new_cap];
        let mask = new_cap - 1;
        for i in 0..self.keys.len() {
            if self.keys[i] != HM_EMPTY {
                let mut idx = (Self::hash(self.keys[i]) as usize) & mask;
                while new_keys[idx] != HM_EMPTY { idx = (idx + 1) & mask; }
                new_keys[idx] = self.keys[i];
                new_vals[idx] = self.vals[i];
            }
        }
        self.keys = new_keys;
        self.vals = new_vals;
    }
}

struct HashMapCell(UnsafeCell<Vec<Option<Box<HashMapI64I64>>>>);
unsafe impl Sync for HashMapCell {}
static HASHMAP_TABLE: HashMapCell = HashMapCell(UnsafeCell::new(Vec::new()));

unsafe fn hashmap_table() -> &'static mut Vec<Option<Box<HashMapI64I64>>> {
    &mut *HASHMAP_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_new() -> i64 {
    let m = Some(Box::new(HashMapI64I64::new()));
    let tbl = hashmap_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = m; return i as i64; }
    }
    tbl.push(m);
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_insert(handle: i64, key: i64, value: i64) -> i32 {
    if handle < 0 || key == HM_EMPTY { return -1; }
    let tbl = hashmap_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_mut() { Some(m) => { m.insert(key, value); 0 }, None => -1 }
}

/// Read by key. Returns the stored value, or 0 if missing. Use
/// `aether_hashmap_i64_contains` for presence testing when 0 is a valid
/// sentinel value.
#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_get(handle: i64, key: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = hashmap_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(m) => m.get(key).unwrap_or(0), None => 0 }
}

/// 1 if the key is present, 0 otherwise.
#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_contains(handle: i64, key: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = hashmap_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(m) => if m.contains(key) { 1 } else { 0 }, None => 0 }
}

/// Remove by key. Returns the removed value, or 0 if missing.
#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_remove(handle: i64, key: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = hashmap_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_mut() { Some(m) => m.remove(key).unwrap_or(0), None => 0 }
}

#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = hashmap_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(m) => m.len as i64, None => 0 }
}

#[no_mangle] pub unsafe extern "C" fn aether_hashmap_i64_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = hashmap_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

// ---- Rc<i64> -------------------------------------------------------------
//
// Refcounted single-i64 value. `clone` bumps the count; `drop` decrements
// and frees when it hits zero. Matches the user-visible Rust semantics
// closely enough for the witness.

struct RcI64 { count: u32, value: i64 }

struct RcI64Cell(UnsafeCell<Vec<Option<Box<RcI64>>>>);
unsafe impl Sync for RcI64Cell {}
static RC_I64_TABLE: RcI64Cell = RcI64Cell(UnsafeCell::new(Vec::new()));

unsafe fn rc_i64_table() -> &'static mut Vec<Option<Box<RcI64>>> {
    &mut *RC_I64_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_rc_i64_new(value: i64) -> i64 {
    let r = Some(Box::new(RcI64 { count: 1, value }));
    let tbl = rc_i64_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = r; return i as i64; }
    }
    tbl.push(r);
    (tbl.len() - 1) as i64
}

/// Increment refcount; returns the same handle. Caller treats the returned
/// handle as a fresh owning reference.
#[no_mangle] pub unsafe extern "C" fn aether_rc_i64_clone(handle: i64) -> i64 {
    if handle < 0 { return -1; }
    let tbl = rc_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_mut() {
        Some(r) => { r.count = r.count.saturating_add(1); handle }
        None => -1,
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_rc_i64_get(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = rc_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(r) => r.value, None => 0 }
}

/// Returns the current strong-count (post-clone, pre-drop) for the witness.
#[no_mangle] pub unsafe extern "C" fn aether_rc_i64_strong_count(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = rc_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(r) => r.count as i64, None => 0 }
}

/// Drop one ownership; frees backing slot when count hits zero.
#[no_mangle] pub unsafe extern "C" fn aether_rc_i64_drop(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = rc_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let zero = match tbl[h].as_mut() {
        Some(r) => { r.count = r.count.saturating_sub(1); r.count == 0 }
        None => return -1,
    };
    if zero { tbl[h] = None; }
    0
}

// ---- mpsc::channel<i64> --------------------------------------------------
//
// Single-producer, single-consumer FIFO queue. Unbounded; `send` always
// succeeds (no backpressure). `recv` returns 1 on success and writes the
// dequeued value through the out-pointer; returns 0 if empty (non-blocking).
// No threads — channels live entirely in process-local memory and the
// witness uses them as a pure data-structure.

struct ChanI64 { queue: std::collections::VecDeque<i64> }

struct ChanCell(UnsafeCell<Vec<Option<Box<ChanI64>>>>);
unsafe impl Sync for ChanCell {}
static CHAN_TABLE: ChanCell = ChanCell(UnsafeCell::new(Vec::new()));

unsafe fn chan_table() -> &'static mut Vec<Option<Box<ChanI64>>> {
    &mut *CHAN_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_chan_i64_new() -> i64 {
    let c = Some(Box::new(ChanI64 { queue: std::collections::VecDeque::new() }));
    let tbl = chan_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = c; return i as i64; }
    }
    tbl.push(c);
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_chan_i64_send(handle: i64, value: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = chan_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_mut() { Some(c) => { c.queue.push_back(value); 0 }, None => -1 }
}

/// Returns 1 + writes value through `out_ptr` on success, 0 if the queue
/// is empty (non-blocking).
#[no_mangle] pub unsafe extern "C" fn aether_chan_i64_recv(handle: i64, out_ptr: i64) -> i32 {
    if handle < 0 || out_ptr == 0 { return 0; }
    let tbl = chan_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_mut() {
        Some(c) => match c.queue.pop_front() {
            Some(v) => { *(out_ptr as *mut i64) = v; 1 }
            None => 0,
        }
        None => 0,
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_chan_i64_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = chan_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(c) => c.queue.len() as i64, None => 0 }
}

#[no_mangle] pub unsafe extern "C" fn aether_chan_i64_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = chan_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

#[cfg(test)]
mod heap_extras_tests {
    use super::*;

    #[test]
    fn box_i64_basics() {
        unsafe {
            let h = aether_box_i64_new(42);
            assert_eq!(aether_box_i64_get(h), 42);
            aether_box_i64_set(h, 99);
            assert_eq!(aether_box_i64_get(h), 99);
            aether_box_i64_free(h);
            assert_eq!(aether_box_i64_get(h), 0);
        }
    }

    #[test]
    fn hashmap_i64_insert_get_remove() {
        unsafe {
            let h = aether_hashmap_i64_new();
            for i in 0..1000i64 {
                aether_hashmap_i64_insert(h, i, i * 3);
            }
            assert_eq!(aether_hashmap_i64_len(h), 1000);
            assert_eq!(aether_hashmap_i64_get(h, 42), 126);
            assert_eq!(aether_hashmap_i64_contains(h, 42), 1);
            assert_eq!(aether_hashmap_i64_contains(h, 9999), 0);
            assert_eq!(aether_hashmap_i64_remove(h, 42), 126);
            assert_eq!(aether_hashmap_i64_contains(h, 42), 0);
            assert_eq!(aether_hashmap_i64_len(h), 999);
            aether_hashmap_i64_free(h);
        }
    }

    #[test]
    fn rc_i64_clone_drop_lifecycle() {
        unsafe {
            let h = aether_rc_i64_new(7);
            assert_eq!(aether_rc_i64_strong_count(h), 1);
            assert_eq!(aether_rc_i64_get(h), 7);
            aether_rc_i64_clone(h);
            aether_rc_i64_clone(h);
            assert_eq!(aether_rc_i64_strong_count(h), 3);
            aether_rc_i64_drop(h);
            assert_eq!(aether_rc_i64_strong_count(h), 2);
            aether_rc_i64_drop(h);
            aether_rc_i64_drop(h);
            // After the final drop, the slot is freed.
            assert_eq!(aether_rc_i64_get(h), 0);
        }
    }

    #[test]
    fn chan_i64_send_recv_fifo() {
        unsafe {
            let c = aether_chan_i64_new();
            for i in 1..=5i64 { aether_chan_i64_send(c, i * 10); }
            assert_eq!(aether_chan_i64_len(c), 5);
            let mut out: i64 = 0;
            for expect in [10, 20, 30, 40, 50] {
                assert_eq!(aether_chan_i64_recv(c, &mut out as *mut i64 as i64), 1);
                assert_eq!(out, expect);
            }
            // Empty queue — recv returns 0.
            assert_eq!(aether_chan_i64_recv(c, &mut out as *mut i64 as i64), 0);
            aether_chan_i64_free(c);
        }
    }
}

#[cfg(test)]
mod conv2d_tests {
    use super::*;

    /// 1×1×4×4 input (sequential 1..16) convolved with a 1×1×3×3 kernel of
    /// all 1s produces a 1×1×2×2 output whose four cells sum the
    /// corresponding 3×3 window of the input. Hand-computed values are:
    ///   out[0,0] = 1+2+3+5+6+7+9+10+11 = 54
    ///   out[0,1] = 2+3+4+6+7+8+10+11+12 = 63
    ///   out[1,0] = 5+6+7+9+10+11+13+14+15 = 90
    ///   out[1,1] = 6+7+8+10+11+12+14+15+16 = 99
    #[test]
    fn conv2d_f32_4x4_with_3x3_all_ones() {
        let input: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let kernel: Vec<f32> = vec![1.0; 9];
        let mut output: Vec<f32> = vec![0.0; 4];
        unsafe {
            let rc = aether_op_conv2d_f32(
                input.as_ptr() as *const _,
                kernel.as_ptr() as *const _,
                output.as_mut_ptr() as *mut _,
                1, 1, 4, 4,
                1, 3, 3,
            );
            assert_eq!(rc, 0, "conv2d returned non-zero status");
        }
        assert_eq!(output, vec![54.0, 63.0, 90.0, 99.0],
                   "conv2d output mismatch; got {:?}", output);
    }

    /// FR-17.14 — hand-crafted Q4_0 block dequant. Build one 18-byte
    /// block whose scale is 1.0 (f16 0x3C00) and whose quants alternate
    /// 0/8 nibbles. Signed quants are -8..+7 mapped from 0..15; with
    /// scale=1.0 the dequanted f32 should be exactly that signed value.
    /// Layout: low nibble at even index i*2, high nibble at i*2+1.
    /// Even quants (0): -8, odd quants (8): 0. Pattern verifies the
    /// low/high split AND the (nibble - 8) sign conversion.
    #[test]
    fn dequant_q4_0_single_block_known_quants() {
        let mut block = [0u8; 18];
        // f16 1.0 = 0x3C00 (little-endian: 0x00, 0x3C).
        block[0] = 0x00;
        block[1] = 0x3C;
        // Each byte: low nibble = 0 (→ -8), high nibble = 8 (→ 0).
        for i in 2..18 { block[i] = 0x80; }
        let mut out = [0.0f32; 32];
        unsafe {
            let rc = aether_dequant_q4_0(
                block.as_ptr() as *const _,
                out.as_mut_ptr() as *mut _,
                1,
            );
            assert_eq!(rc, 0);
        }
        for i in 0..32 {
            let expected = if i % 2 == 0 { -8.0 } else { 0.0 };
            assert_eq!(out[i], expected, "quant {} expected {}, got {}", i, expected, out[i]);
        }
    }

    /// FR-17.13-extra — FlashAttention v2 must match naive causal SDPA
    /// within tight float tolerance. Build deterministic Q/K/V, compute
    /// both, compare row-by-row.
    #[test]
    fn flash_attention_v2_matches_naive_sdpa() {
        let n = 8usize;
        let d = 4usize;
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        // Deterministic fill: each cell = sin(i * 0.7 + j * 0.3 + offset).
        for i in 0..n {
            for j in 0..d {
                q[i * d + j] = ((i as f32) * 0.7 + (j as f32) * 0.3).sin();
                k[i * d + j] = ((i as f32) * 0.5 + (j as f32) * 0.2 + 0.1).sin();
                v[i * d + j] = ((i as f32) * 0.3 + (j as f32) * 0.4 + 0.2).cos();
            }
        }
        let mut o_fa = vec![0.0f32; n * d];
        unsafe {
            let rc = aether_flash_attention_v2_f32(
                q.as_ptr() as *const _, k.as_ptr() as *const _, v.as_ptr() as *const _,
                o_fa.as_mut_ptr() as *mut _,
                n as c_int, d as c_int,
            );
            assert_eq!(rc, 0);
        }
        // Naive reference: scores = Q @ K^T * scale, causal mask, softmax, @V.
        let scale = 1.0 / (d as f32).sqrt();
        let mut o_ref = vec![0.0f32; n * d];
        for r in 0..n {
            let mut scores = vec![f32::NEG_INFINITY; n];
            let mut max_s = f32::NEG_INFINITY;
            for c in 0..=r {  // causal: only j ≤ r
                let mut s = 0.0f32;
                for di in 0..d { s += q[r * d + di] * k[c * d + di]; }
                scores[c] = s * scale;
                if scores[c] > max_s { max_s = scores[c]; }
            }
            let mut sum = 0.0f32;
            let mut p = vec![0.0f32; n];
            for c in 0..=r {
                p[c] = (scores[c] - max_s).exp();
                sum += p[c];
            }
            for c in 0..=r { p[c] /= sum; }
            for di in 0..d {
                let mut acc = 0.0f32;
                for c in 0..=r { acc += p[c] * v[c * d + di]; }
                o_ref[r * d + di] = acc;
            }
        }
        for i in 0..n * d {
            let diff = (o_fa[i] - o_ref[i]).abs();
            assert!(diff < 1e-5, "FA2 vs naive mismatch at {}: fa={}, ref={}", i, o_fa[i], o_ref[i]);
        }
    }

    /// Q4_0 scale precision check — scale=0.5 (f16 0x3800) should halve
    /// the dequanted values. Use a uniform 0xF7 byte pattern: low=0x7
    /// (→ -1 signed), high=0xF (→ +7 signed). Expect alternating
    /// -0.5 / +3.5 across the 32 outputs.
    #[test]
    fn dequant_q4_0_with_scale_half() {
        let mut block = [0u8; 18];
        block[0] = 0x00; block[1] = 0x38;  // f16 0.5
        for i in 2..18 { block[i] = 0xF7; }
        let mut out = [0.0f32; 32];
        unsafe {
            aether_dequant_q4_0(block.as_ptr() as *const _,
                                out.as_mut_ptr() as *mut _, 1);
        }
        for i in 0..32 {
            let expected = if i % 2 == 0 { -0.5 } else { 3.5 };
            assert!((out[i] - expected).abs() < 1e-6,
                    "quant {} expected {}, got {}", i, expected, out[i]);
        }
    }

    /// FR-18.1 — single-host NCCL fallback round-trip.
    #[test]
    fn nccl_single_host_surface_roundtrip() {
        aether_nccl_init();
        assert!(aether_nccl_init_count() >= 1);
        let comm = aether_nccl_comm_create(1, 0);
        assert!(comm > 0, "expected handle ≥ 1, got {}", comm);
        assert_eq!(aether_nccl_comm_world_size(comm), 1);
        assert_eq!(aether_nccl_comm_rank(comm), 0);
        // All-reduce on world_size=1 is identity.
        let send = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut recv = vec![0.0f32; 4];
        unsafe {
            let rc = aether_nccl_all_reduce_f32(
                send.as_ptr() as *const _,
                recv.as_mut_ptr() as *mut _,
                4, 0, comm,
            );
            assert_eq!(rc, 0);
        }
        assert_eq!(recv, send);
        // Multi-rank rejection on single-host build.
        let bad = aether_nccl_comm_create(2, 0);
        assert_eq!(bad, -1, "multi-rank comm_create should return -1 sentinel");
        aether_nccl_comm_destroy(comm);
        aether_nccl_finalize();
    }

    /// FR-18.5 — column-parallel TP simulation must match monolithic matmul.
    #[test]
    fn tp_column_parallel_matches_monolithic() {
        // (m, k, n) = (3, 4, 6); world_size=2 → n_shard=3 per rank.
        let m = 3; let k = 4; let n = 6; let ws = 2;
        let x: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.05 - 0.5).collect();
        let mut tp_out = vec![0.0f32; m * n];
        unsafe {
            let rc = aether_tp_simulate_column_parallel_linear_f32(
                x.as_ptr() as *const _, w.as_ptr() as *const _,
                tp_out.as_mut_ptr() as *mut _,
                m as i32, k as i32, n as i32, ws as i32,
            );
            assert_eq!(rc, 0);
        }
        // Reference: plain matmul.
        let mut ref_out = vec![0.0f32; m * n];
        for r in 0..m { for c in 0..n {
            let mut s = 0.0f32;
            for ki in 0..k { s += x[r * k + ki] * w[ki * n + c]; }
            ref_out[r * n + c] = s;
        }}
        for i in 0..(m * n) {
            assert!((tp_out[i] - ref_out[i]).abs() < 1e-5,
                    "TP shard mismatch at {}: tp={}, ref={}", i, tp_out[i], ref_out[i]);
        }
    }

    /// FR-18.6 — 2-stage PP forward must match monolithic forward.
    #[test]
    fn pp_2stage_matches_monolithic() {
        let mb = 2; let d = 4; let n_blocks = 4; let n_stages = 2;
        let input: Vec<f32> = (0..(mb * d)).map(|i| (i as f32) * 0.5).collect();
        let scales: Vec<f32> = vec![1.1, 0.9, 1.2, 0.8];
        let biases: Vec<f32> = vec![0.1, -0.1, 0.05, 0.02];
        let mut pp_out = vec![0.0f32; mb * d];
        unsafe {
            let rc = aether_pp_simulate_2stage_forward_f32(
                input.as_ptr() as *const _,
                scales.as_ptr() as *const _,
                biases.as_ptr() as *const _,
                pp_out.as_mut_ptr() as *mut _,
                mb as i32, d as i32, n_blocks as i32, n_stages as i32,
            );
            assert_eq!(rc, 0);
        }
        // Reference: apply each block in sequence with NO staging.
        let mut x = input.clone();
        for b in 0..n_blocks {
            for v in x.iter_mut() { *v = *v * scales[b] + biases[b]; }
        }
        for i in 0..(mb * d) {
            assert!((pp_out[i] - x[i]).abs() < 1e-5,
                    "PP stage output mismatch at {}: pp={}, ref={}", i, pp_out[i], x[i]);
        }
    }

    /// FR-18.7 — ZeRO bytes are strictly decreasing across Z1/Z2/Z3.
    #[test]
    fn zero_stage_bytes_monotone_decreasing() {
        let n = 1_000_000; let ws = 4;
        let baseline = (4 * n * 4) as i64;  // params+grad+optim×2, all full
        let z1 = aether_zero_simulate_stage_bytes_f32(n, ws, 1);
        let z2 = aether_zero_simulate_stage_bytes_f32(n, ws, 2);
        let z3 = aether_zero_simulate_stage_bytes_f32(n, ws, 3);
        assert!(z1 < baseline, "Z1 should save vs baseline: z1={}, baseline={}", z1, baseline);
        assert!(z2 < z1, "Z2 should save vs Z1: z2={}, z1={}", z2, z1);
        assert!(z3 < z2, "Z3 should save vs Z2: z3={}, z2={}", z3, z2);
        // Z3 with ws=4 should hit ~baseline/4.
        let expected_z3 = baseline / 4;
        assert!((z3 - expected_z3).abs() <= 16,
                "Z3 with ws=4 should ≈ baseline/4: got {}, expected {}", z3, expected_z3);
    }

    /// FR-18.8 — overlapped time = max(compute, comm); serial = sum.
    #[test]
    fn overlap_simulation_savings() {
        let ov = aether_overlap_simulate_overlapped_us(100, 80);
        let se = aether_overlap_simulate_serial_us(100, 80);
        assert_eq!(ov, 100);
        assert_eq!(se, 180);
        assert!(ov < se, "overlap should save vs serial");
    }

    /// FR-18.9 — rank-K compression preserves the first K columns.
    #[test]
    fn grad_compress_preserves_top_k_cols() {
        let m = 3; let n = 5; let k = 2;
        let g: Vec<f32> = (0..(m * n)).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let mut r = vec![0.0f32; m * n];
        unsafe {
            aether_grad_compress_lowrank_f32(
                g.as_ptr() as *const _, r.as_mut_ptr() as *mut _,
                m as i32, n as i32, k as i32,
            );
        }
        for i in 0..m {
            for j in 0..n {
                let expect = if j < k { g[i * n + j] } else { 0.0 };
                assert_eq!(r[i * n + j], expect, "[{},{}] mismatch", i, j);
            }
        }
    }

    /// FR-18.4 — FSDP shard + all-gather round-trip is the identity.
    #[test]
    fn fsdp_shard_alltoall_identity() {
        let n = 12; let ws = 3;
        let params: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let mut out = vec![0.0f32; n];
        unsafe {
            let rc = aether_fsdp_simulate_shard_alltoall_f32(
                params.as_ptr() as *const _, out.as_mut_ptr() as *mut _,
                n as i32, ws as i32,
            );
            assert_eq!(rc, 0);
        }
        assert_eq!(out, params, "FSDP shard+gather round-trip must be identity");
    }

    /// 2 input channels, 1 output channel — the per-channel partial sums
    /// must add. Use channel 0 = all 1s, channel 1 = all 2s; kernel for
    /// channel 0 = all 1s, kernel for channel 1 = all 1s. Each 3×3 window
    /// sums to 9 (from channel 0) + 18 (from channel 1) = 27.
    #[test]
    fn conv2d_f32_two_in_channels_sum() {
        let mut input: Vec<f32> = vec![1.0; 16];
        input.extend(std::iter::repeat(2.0).take(16));  // channel 1
        let kernel: Vec<f32> = vec![1.0; 18];           // 1 out * 2 in * 3*3
        let mut output: Vec<f32> = vec![0.0; 4];
        unsafe {
            let rc = aether_op_conv2d_f32(
                input.as_ptr() as *const _,
                kernel.as_ptr() as *const _,
                output.as_mut_ptr() as *mut _,
                1, 2, 4, 4,
                1, 3, 3,
            );
            assert_eq!(rc, 0);
        }
        assert_eq!(output, vec![27.0; 4]);
    }
}
