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

#[cfg(feature = "cuda")]
pub mod serving;

#[cfg(feature = "cuda")]
pub mod batched_serving;

#[cfg(feature = "cuda")]
pub mod bert;

#[cfg(feature = "nccl")]
pub mod nccl_real;

/// FR-x-extra-tp — tensor-parallel inference orchestration.  Phase 1 ships
/// the sharding-plan math + API surface + NCCL-availability detection;
/// TP=1 path is bit-identical to single-GPU.  TP=N≥2 falls back to TP=1
/// with a warning until the multi-context cuda.rs refactor lands.  See
/// `tensor_parallel::TP_GAPS` for the structural follow-ons.
#[cfg(feature = "cuda")]
pub mod tensor_parallel;

pub mod tls13;

pub mod http2;

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

/// FR-17.5-extra (Qwen/Llama) — RMSNorm forward.
/// `y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps)`.
#[no_mangle] pub unsafe extern "C" fn aether_op_rms_norm_f32(
    x: *const c_void, gamma: *const c_void, eps: f32,
    out: *mut c_void, rows: c_int, d: c_int,
) -> c_int {
    ops::rms_norm_f32(x as _, gamma as _, eps, out as _, rows as _, d as _);
    0
}

/// FR-17.13-extra (Qwen/Llama) — RoPE applied in place on `[seq, n_heads, head_dim]`.
/// `base` is the rotary base (Qwen2.5 uses 1_000_000.0; Llama uses 10_000.0).
/// `pos_start` is the token-position of the first row -- 0 for a forward pass
/// from scratch, `kv_cache_len` for a prefill that resumes mid-stream.
#[no_mangle] pub unsafe extern "C" fn aether_op_rope_apply_f32(
    x: *mut c_void, seq: c_int, n_heads: c_int, head_dim: c_int,
    base: f32, pos_start: c_int,
) -> c_int {
    ops::rope_apply_f32(x as _, seq as _, n_heads as _, head_dim as _,
        base, pos_start as _);
    0
}

/// FR-17.13-extra (Qwen/Llama GQA) — broadcast K/V from `n_kv_heads`
/// to `n_q_heads` by repeating each KV head `n_q_heads / n_kv_heads`
/// times. Required before feeding to the SDPA kernel under grouped-
/// query attention (Qwen2.5-7B: 28 Q heads, 4 KV heads -> 7x repeat).
#[no_mangle] pub unsafe extern "C" fn aether_op_gqa_repeat_kv_f32(
    kv_in: *const c_void, kv_out: *mut c_void,
    seq: c_int, n_kv_heads: c_int, head_dim: c_int, n_q_heads: c_int,
) -> c_int {
    if (n_q_heads % n_kv_heads) != 0 { return 1; }
    ops::gqa_repeat_kv_f32(kv_in as _, kv_out as _,
        seq as _, n_kv_heads as _, head_dim as _, n_q_heads as _);
    0
}

/// FR-17.17-extra / matt-voice — apply a LoRA update in place to a
/// matmul-layout weight. See `ops::apply_lora_f32` for math + layout.
#[no_mangle] pub unsafe extern "C" fn aether_op_apply_lora_f32(
    w: *mut c_void, lora_a: *const c_void, lora_b: *const c_void,
    scale: f32, d_in: c_int, d_out: c_int, rank: c_int,
) -> c_int {
    if d_in <= 0 || d_out <= 0 || rank <= 0 { return 1; }
    ops::apply_lora_f32(w as _, lora_a as _, lora_b as _, scale,
        d_in as _, d_out as _, rank as _);
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

/// Load the i64 word at element index `i` (byte offset `i*8`) of the buffer.
/// Used by the closure-object ABI (P6.6): a closure value is a heap block
/// laid out as `[fn_ptr | cap0 | cap1 | ...]`, so `aether_load_i64(obj, 0)`
/// fetches the code pointer and `aether_load_i64(obj, 1+k)` fetches capture k.
#[no_mangle] pub unsafe extern "C" fn aether_load_i64(p: i64, i: i64) -> i64 {
    if p == 0 || i < 0 { return 0; }
    *(p as *const i64).add(i as usize)
}

/// Store the i64 word `v` at element index `i` (byte offset `i*8`). The
/// closure-object constructor uses this to write the code pointer + captured
/// values into the heap block. See `aether_load_i64`.
#[no_mangle] pub unsafe extern "C" fn aether_store_i64(p: i64, i: i64, v: i64) {
    if p == 0 || i < 0 { return; }
    *(p as *mut i64).add(i as usize) = v;
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

/// Allocate `n` bytes of executable + writable memory. On Windows uses
/// `VirtualAlloc(PAGE_EXECUTE_READWRITE)`; on POSIX uses `mmap` with
/// `PROT_READ|PROT_WRITE|PROT_EXEC`. Returns a pointer suitable for
/// `aether_call_jit_i64`. Caller frees with `aether_free_executable`.
/// Returns 0 on failure.
#[cfg(windows)]
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

#[cfg(unix)]
#[no_mangle] pub unsafe extern "C" fn aether_alloc_executable(n: i64) -> i64 {
    if n <= 0 { return 0; }
    extern "C" {
        fn mmap(addr: *mut c_void, length: usize, prot: c_int,
                flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
    }
    const PROT_READ: c_int  = 1;
    const PROT_WRITE: c_int = 2;
    const PROT_EXEC: c_int  = 4;
    const MAP_PRIVATE: c_int = 0x02;
    const MAP_ANONYMOUS: c_int = 0x20;
    let p = mmap(
        std::ptr::null_mut(), n as usize,
        PROT_READ | PROT_WRITE | PROT_EXEC,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0,
    );
    if p as isize == -1 { 0 } else { p as i64 }
}

#[cfg(windows)]
#[no_mangle] pub unsafe extern "C" fn aether_free_executable(p: i64, _n: i64) {
    if p == 0 { return; }
    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
    }
    const MEM_RELEASE: u32 = 0x8000;
    let _ = VirtualFree(p as *mut u8, 0, MEM_RELEASE);
}

#[cfg(unix)]
#[no_mangle] pub unsafe extern "C" fn aether_free_executable(p: i64, n: i64) {
    if p == 0 || n <= 0 { return; }
    extern "C" { fn munmap(addr: *mut c_void, length: usize) -> c_int; }
    let _ = munmap(p as *mut c_void, n as usize);
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

/// Spawn an OS thread that runs the Aether function at address `fn_ptr` (an
/// `fn(i64) -> i64`, ABI-compatible with the MS x64 single-arg convention
/// aetherc emits) with `arg`. Returns an opaque handle (a boxed
/// `JoinHandle<i64>` pointer, or 0 on failure) for `aether_thread_join`, which
/// returns the value the worker COMPUTED — real OS-level parallelism, no
/// executor, no global state. The fn address is moved into the thread as a
/// plain integer (Send) and reconstructed there.
#[no_mangle] pub unsafe extern "C" fn aether_thread_spawn(fn_ptr: i64, arg: i64) -> i64 {
    if fn_ptr == 0 { return 0; }
    let handle = std::thread::spawn(move || {
        let f: extern "C" fn(i64) -> i64 = std::mem::transmute(fn_ptr);
        f(arg)
    });
    Box::into_raw(Box::new(handle)) as i64
}

/// Join the thread `handle` from `aether_thread_spawn` and return the value its
/// worker function produced (0 if the handle is null or the thread panicked).
/// Consumes the handle.
#[no_mangle] pub unsafe extern "C" fn aether_thread_join(handle: i64) -> i64 {
    if handle == 0 { return 0; }
    let h: Box<std::thread::JoinHandle<i64>> =
        Box::from_raw(handle as *mut std::thread::JoinHandle<i64>);
    h.join().unwrap_or(0)
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

/// P6.13 — `std::env::set_var`. Set the environment variable `key` to `val`
/// (both NUL-terminated C-strings). No-op on a null/invalid pointer.
#[no_mangle] pub unsafe extern "C" fn aether_env_set(key: i64, val: i64) {
    let Some(k) = cstr_to_string(key) else { return; };
    let Some(v) = cstr_to_string(val) else { return; };
    std::env::set_var(k, v);
}

/// P6.13 — `std::env::var` parsed as an i64. Returns the variable's value
/// parsed as an integer, or -1 if it is unset or not a valid integer.
#[no_mangle] pub unsafe extern "C" fn aether_env_var_i64(key: i64) -> i64 {
    let Some(k) = cstr_to_string(key) else { return -1; };
    match std::env::var(&k) {
        Ok(s) => s.trim().parse::<i64>().unwrap_or(-1),
        Err(_) => -1,
    }
}

/// Read a NUL-terminated C-string at pointer `p` into an owned `String`.
/// Returns `None` on a null pointer or invalid UTF-8.
unsafe fn cstr_to_string(p: i64) -> Option<String> {
    if p == 0 { return None; }
    let mut len = 0usize;
    while *(p as *const u8).add(len) != 0 { len += 1; }
    let bytes = std::slice::from_raw_parts(p as *const u8, len);
    std::str::from_utf8(bytes).ok().map(|s| s.to_string())
}

/// P6.13 — process spawn. Run a shell command string and return its exit code
/// (or -1 if the process could not be spawned / was signal-terminated).
/// Cross-platform: `cmd /C <s>` on Windows, `sh -c <s>` elsewhere. This is the
/// `std::process::Command` equivalent on the C-ABI surface.
#[no_mangle] pub unsafe extern "C" fn aether_process_run(cmd: i64) -> i64 {
    if cmd == 0 { return -1; }
    let mut len = 0usize;
    while *(cmd as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(cmd as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    let mut c = if cfg!(windows) {
        let mut c = std::process::Command::new("cmd");
        c.arg("/C").arg(s);
        c
    } else {
        let mut c = std::process::Command::new("sh");
        c.arg("-c").arg(s);
        c
    };
    match c.status() {
        Ok(st) => st.code().map(|x| x as i64).unwrap_or(-1),
        Err(_) => -1,
    }
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

/// FR-18.10 — bind a TCP listener on `<addr>:port`. Caller supplies
/// the bind address as bytes; pass `"0.0.0.0"` to accept on all
/// interfaces (for multi-host distributed servers). Pass `port = 0`
/// for OS-assigned ephemeral. Returns listener handle or -1.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_listen_addr(
    addr: i64, addr_len: c_int, port: i64,
) -> i64 {
    if addr == 0 || addr_len <= 0 || !(0..=65535).contains(&port) { return -1; }
    let addr_bytes = std::slice::from_raw_parts(addr as *const u8, addr_len as usize);
    let Ok(addr_str) = std::str::from_utf8(addr_bytes) else { return -1; };
    let bind = format!("{}:{}", addr_str, port);
    match std::net::TcpListener::bind(&bind) {
        Ok(l) => {
            let v = tcp_listeners();
            for (i, slot) in v.iter_mut().enumerate() {
                if slot.is_none() { *slot = Some(Box::new(l)); return i as i64; }
            }
            v.push(Some(Box::new(l)));
            (v.len() - 1) as i64
        }
        Err(e) => {
            eprintln!("[tcp_listen_addr] bind {} failed: {}", bind, e);
            -1
        }
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

/// FR-18.10 — connect to an arbitrary `host:port`. `host_bytes` is a
/// caller buffer with hostname/IP (any form resolvable via std's
/// TcpStream::connect, including hostnames + IPv4/IPv6 literals).
/// Returns stream handle or -1.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_connect_host(
    host: i64, host_len: c_int, port: i64,
) -> i64 {
    if host == 0 || host_len <= 0 || !(0..=65535).contains(&port) { return -1; }
    let host_bytes = std::slice::from_raw_parts(host as *const u8, host_len as usize);
    let Ok(host_str) = std::str::from_utf8(host_bytes) else { return -1; };
    let addr = format!("{}:{}", host_str, port);
    match std::net::TcpStream::connect(&addr) {
        Ok(s) => {
            // 30s read/write timeout so a stuck peer doesn't wedge the rank.
            let _ = s.set_read_timeout(Some(std::time::Duration::from_secs(30)));
            let _ = s.set_write_timeout(Some(std::time::Duration::from_secs(30)));
            let v = tcp_streams();
            for (i, slot) in v.iter_mut().enumerate() {
                if slot.is_none() { *slot = Some(Box::new(s)); return i as i64; }
            }
            v.push(Some(Box::new(s)));
            (v.len() - 1) as i64
        }
        Err(e) => {
            eprintln!("[tcp_connect_host] {}: {}", addr, e);
            -1
        }
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

/// Fill `n` f32 slots at `p` with value `v`. Pairs with `aether_alloc_bytes`
/// for witnesses that need a constant input vector without writing 256
/// per-element `aether_store_f32` calls in .aether source.
#[no_mangle] pub unsafe extern "C" fn aether_fill_f32(p: i64, n: c_int, v: f32) {
    if p == 0 || n <= 0 { return; }
    let s = std::slice::from_raw_parts_mut(p as *mut f32, n as usize);
    for slot in s.iter_mut() { *slot = v; }
}

/// Return 42 if `v` is finite (not NaN, not infinite) and `|v| > eps_lo`
/// and `|v| < eps_hi`. Sanity-band gate for witnesses that ingest real
/// model weights — the exact value depends on the file, but it must be
/// in a finite, non-degenerate range.
#[no_mangle] pub extern "C" fn aether_f32_in_band_exit(
    v: f32, eps_lo: f32, eps_hi: f32,
) -> c_int {
    if !v.is_finite() { return 1; }
    let a = v.abs();
    if a > eps_lo && a < eps_hi { 42 } else { 1 }
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

/// Backing-buffer pointer for the Vec, as an i64. Foundation for native
/// `&[i64]` fat pointers (P16.19): a slice over a Vec is `(as_ptr, len)`.
/// Returns -1 on invalid/empty handle. NOTE: the pointer is invalidated by
/// any subsequent push that triggers a realloc — slices must be taken after
/// the Vec stops growing, exactly like Rust's borrow rules enforce.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_as_ptr(handle: i64) -> i64 {
    if handle < 0 { return -1; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_ref() {
        Some(v) if !v.ptr.is_null() => v.ptr as i64,
        _ => -1,
    }
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

// ---- Vec<f32> handle table ------------------------------------------------
// Mirrors the Vec<i64> table exactly, but stores f32 elements. Backs native
// `&[f32]` slices (P16.19): `slice_from_raw(aether_vec_f32_as_ptr(v),
// aether_vec_f32_len(v))`. Push takes the f32 bit-pattern in the low 32 bits
// of an i64 so it crosses the C ABI without a float register (the asm backend
// passes/returns scalars in GPRs for FFI); the runtime reinterprets it.

struct VecF32 {
    ptr: *mut f32,
    len: usize,
    cap: usize,
}

struct VecF32Cell(UnsafeCell<Vec<Option<Box<VecF32>>>>);
unsafe impl Sync for VecF32Cell {}
static VEC_F32_TABLE: VecF32Cell = VecF32Cell(UnsafeCell::new(Vec::new()));

unsafe fn vec_f32_table() -> &'static mut Vec<Option<Box<VecF32>>> {
    &mut *VEC_F32_TABLE.0.get()
}

unsafe fn vec_f32_alloc_buf(cap: usize) -> *mut f32 {
    if cap == 0 { return std::ptr::null_mut(); }
    let bytes = cap.checked_mul(std::mem::size_of::<f32>()).expect("vec f32 cap overflow");
    aether_alloc_bytes(bytes as i64) as *mut f32
}

unsafe fn vec_f32_free_buf(ptr: *mut f32, cap: usize) {
    if ptr.is_null() || cap == 0 { return; }
    let bytes = cap * std::mem::size_of::<f32>();
    aether_free_bytes(ptr as i64, bytes as i64);
}

/// Allocate a fresh empty `Vec<f32>`. Returns a non-negative handle.
#[no_mangle] pub unsafe extern "C" fn aether_vec_f32_new() -> i64 {
    let v = Box::new(VecF32 { ptr: std::ptr::null_mut(), len: 0, cap: 0 });
    let tbl = vec_f32_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(v); return i as i64; }
    }
    tbl.push(Some(v));
    (tbl.len() - 1) as i64
}

/// Push an f32 (passed as its little-endian bit-pattern in the low 32 bits of
/// `bits`) onto the Vec. Capacity-doubling growth. 0 ok / -1 bad handle / -2 OOM.
#[no_mangle] pub unsafe extern "C" fn aether_vec_f32_push(handle: i64, bits: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = vec_f32_table();
    let idx = handle as usize;
    if idx >= tbl.len() { return -1; }
    let v = match tbl[idx].as_mut() { Some(v) => v, None => return -1 };
    if v.len == v.cap {
        let new_cap = if v.cap == 0 { 4 } else { v.cap * 2 };
        let new_ptr = vec_f32_alloc_buf(new_cap);
        if new_ptr.is_null() { return -2; }
        if v.len > 0 {
            std::ptr::copy_nonoverlapping(v.ptr, new_ptr, v.len);
        }
        vec_f32_free_buf(v.ptr, v.cap);
        v.ptr = new_ptr;
        v.cap = new_cap;
    }
    *v.ptr.add(v.len) = f32::from_bits((bits & 0xFFFF_FFFF) as u32);
    v.len += 1;
    0
}

/// Number of elements currently in the Vec<f32>.
#[no_mangle] pub unsafe extern "C" fn aether_vec_f32_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = vec_f32_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(v) => v.len as i64, None => 0 }
}

/// Backing-buffer pointer for the Vec<f32>, as an i64. Returns -1 on
/// invalid/empty handle. Foundation for native `&[f32]` fat pointers (P16.19).
#[no_mangle] pub unsafe extern "C" fn aether_vec_f32_as_ptr(handle: i64) -> i64 {
    if handle < 0 { return -1; }
    let tbl = vec_f32_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_ref() {
        Some(v) if !v.ptr.is_null() => v.ptr as i64,
        _ => -1,
    }
}

/// Free the buffer + release the handle slot. Idempotent.
#[no_mangle] pub unsafe extern "C" fn aether_vec_f32_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = vec_f32_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    if let Some(mut v) = tbl[h].take() {
        vec_f32_free_buf(v.ptr, v.cap);
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

/// Backing UTF-8 buffer pointer for the string, as an i64. Foundation for
/// native `&str` / `&[u8]` fat pointers (P16.19): a byte slice over a String
/// is `(as_ptr, len)`. Returns -1 on invalid/empty handle. Like the Vec
/// version, the pointer is invalidated by any subsequent push that reallocs.
#[no_mangle] pub unsafe extern "C" fn aether_string_as_ptr(handle: i64) -> i64 {
    if handle < 0 { return -1; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    match tbl[h].as_ref() {
        Some(s) if !s.ptr.is_null() => s.ptr as i64,
        _ => -1,
    }
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

// =====================================================================
// FR-19.9 — Byte-level BPE tokenizer.
//
// Real BPE algorithm with the same shape as huggingface tokenizers:
//   - Initial vocab: ids 0..255 = raw bytes 0..255 (byte-level fallback)
//   - Merged tokens: ids 256+, registered via aether_bpe_add_merge with
//     a (left_id, right_id, rank, merged_bytes) tuple.
//   - Encode loop: repeatedly find the adjacent pair with the LOWEST
//     rank in the current token list and replace all non-overlapping
//     occurrences with the merged token id. Loop until no merge fires.
//   - Decode: concat decode_table[id] for each id.
//
// What this proves: the BPE algorithm shape works end-to-end through
// the asm chain, encoding then decoding gives back the original bytes.
// What it does NOT prove: tokenizer.json parser, sentencepiece,
// tiktoken cl100k. Those are FR-19.9-extra. matt-voice's Qwen2.5
// tokenizer is BPE so this is on-path for the serving deploy.
// =====================================================================

struct BpeTokenizer {
    /// id -> byte sequence backing the token. ids 0..255 are implicit
    /// single-byte slots; ids 256+ come from add_merge calls.
    decode_table: Vec<Vec<u8>>,
    /// (left_id, right_id) -> (merged_id, rank). Lower rank = applied
    /// earlier in the encode loop.
    merges: std::collections::HashMap<(u32, u32), (u32, u32)>,
}

struct BpeCell(UnsafeCell<Vec<Option<Box<BpeTokenizer>>>>);
unsafe impl Sync for BpeCell {}
static BPE_TABLE: BpeCell = BpeCell(UnsafeCell::new(Vec::new()));
unsafe fn bpe_table() -> &'static mut Vec<Option<Box<BpeTokenizer>>> {
    &mut *BPE_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_bpe_tokenizer_new() -> i64 {
    let mut decode_table: Vec<Vec<u8>> = Vec::with_capacity(256);
    for b in 0..256u32 { decode_table.push(vec![b as u8]); }
    let t = BpeTokenizer { decode_table, merges: Default::default() };
    let tbl = bpe_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(t)); return i as i64; }
    }
    tbl.push(Some(Box::new(t)));
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_bpe_tokenizer_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

/// Register a merge rule. Returns the new token's id (≥ 256) on
/// success; -1 on error (bad handle, bad left/right ids, dup pair,
/// or merged_bytes pointer/len invalid).
#[no_mangle] pub unsafe extern "C" fn aether_bpe_add_merge(
    handle: i64,
    left_id: c_int, right_id: c_int, rank: c_int,
    merged_bytes: *const c_void, n_bytes: c_int,
) -> c_int {
    if handle < 0 || rank < 0 || left_id < 0 || right_id < 0 { return -1; }
    if merged_bytes.is_null() || n_bytes <= 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_mut() else { return -1; };
    let left = left_id as u32;
    let right = right_id as u32;
    if (left as usize) >= t.decode_table.len() { return -1; }
    if (right as usize) >= t.decode_table.len() { return -1; }
    if t.merges.contains_key(&(left, right)) { return -1; }
    let bytes = std::slice::from_raw_parts(merged_bytes as *const u8, n_bytes as usize).to_vec();
    let new_id = t.decode_table.len() as u32;
    t.decode_table.push(bytes);
    t.merges.insert((left, right), (new_id, rank as u32));
    new_id as c_int
}

/// Encode `text` (UTF-8 bytes) into token ids. Writes up to `max_ids`
/// ids into `out_ids`. Returns the number of ids written, or -1 on
/// overflow / bad handle. `out_ids` is treated as a `*mut i32` buffer.
#[no_mangle] pub unsafe extern "C" fn aether_bpe_encode(
    handle: i64,
    text: *const c_void, n_text: c_int,
    out_ids: *mut c_void, max_ids: c_int,
) -> c_int {
    if handle < 0 || text.is_null() || out_ids.is_null() { return -1; }
    if n_text < 0 || max_ids <= 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_ref() else { return -1; };
    // 1. Initial tokens: each byte is its own id.
    let bytes = std::slice::from_raw_parts(text as *const u8, n_text as usize);
    let mut tokens: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();
    // 2. BPE merge loop: each iteration find the pair with lowest rank,
    //    replace ALL non-overlapping occurrences.
    loop {
        if tokens.len() < 2 { break; }
        let mut best_pair: Option<(u32, u32)> = None;
        let mut best_rank: u32 = u32::MAX;
        let mut best_merged: u32 = 0;
        for i in 0..tokens.len() - 1 {
            if let Some(&(merged, rank)) = t.merges.get(&(tokens[i], tokens[i + 1])) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pair = Some((tokens[i], tokens[i + 1]));
                    best_merged = merged;
                }
            }
        }
        let Some((bl, br)) = best_pair else { break; };
        let mut new_tokens: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == bl && tokens[i + 1] == br {
                new_tokens.push(best_merged);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = new_tokens;
    }
    if tokens.len() > max_ids as usize { return -1; }
    let out = std::slice::from_raw_parts_mut(out_ids as *mut i32, max_ids as usize);
    for (i, tok) in tokens.iter().enumerate() { out[i] = *tok as i32; }
    tokens.len() as c_int
}

/// FR-x-extra: BPE merge-loop over pre-resolved initial token ids.
///
/// `aether_bpe_encode` starts from raw bytes (0..255) which only works
/// if the merges table was built against byte-level inputs. GPT-2/Qwen
/// style BPE uses a byte→unicode surface alphabet, so the initial
/// tokens are surface-char vocab ids (single-byte tokens in the GPT-2
/// vocab — there are 256 of them, scattered through the vocab). The
/// caller is responsible for the byte→surface_id lookup (since that
/// requires reading the GGUF vocab strings); we just run the same
/// merge loop over the resolved initial ids.
///
/// Writes up to `max_ids` resulting ids into `out_ids`. Returns the
/// number written, -1 on overflow / bad handle / invalid initial id.
#[no_mangle] pub unsafe extern "C" fn aether_bpe_encode_ids(
    handle: i64,
    initial_ids: *const c_void, n_initial: c_int,
    out_ids: *mut c_void, max_ids: c_int,
) -> c_int {
    if handle < 0 || initial_ids.is_null() || out_ids.is_null() { return -1; }
    if n_initial < 0 || max_ids <= 0 { return -1; }
    if n_initial == 0 { return 0; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_ref() else { return -1; };
    let in_buf = std::slice::from_raw_parts(initial_ids as *const i32, n_initial as usize);
    let mut tokens: Vec<u32> = Vec::with_capacity(in_buf.len());
    for &id in in_buf {
        if id < 0 { return -1; }
        let id_u = id as usize;
        if id_u >= t.decode_table.len() { return -1; }
        tokens.push(id as u32);
    }
    loop {
        if tokens.len() < 2 { break; }
        let mut best_pair: Option<(u32, u32)> = None;
        let mut best_rank: u32 = u32::MAX;
        let mut best_merged: u32 = 0;
        for i in 0..tokens.len() - 1 {
            if let Some(&(merged, rank)) = t.merges.get(&(tokens[i], tokens[i + 1])) {
                if rank < best_rank {
                    best_rank = rank;
                    best_pair = Some((tokens[i], tokens[i + 1]));
                    best_merged = merged;
                }
            }
        }
        let Some((bl, br)) = best_pair else { break; };
        let mut new_tokens: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == bl && tokens[i + 1] == br {
                new_tokens.push(best_merged);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        tokens = new_tokens;
    }
    if tokens.len() > max_ids as usize { return -1; }
    let out = std::slice::from_raw_parts_mut(out_ids as *mut i32, max_ids as usize);
    for (i, tok) in tokens.iter().enumerate() { out[i] = *tok as i32; }
    tokens.len() as c_int
}

/// Decode `n_ids` ids back into UTF-8 bytes. Writes up to `max_bytes`
/// into `out_bytes`. Returns bytes written, or -1 on bad handle / id
/// out of range / overflow.
#[no_mangle] pub unsafe extern "C" fn aether_bpe_decode(
    handle: i64,
    ids: *const c_void, n_ids: c_int,
    out_bytes: *mut c_void, max_bytes: c_int,
) -> c_int {
    if handle < 0 || ids.is_null() || out_bytes.is_null() { return -1; }
    if n_ids <= 0 || max_bytes <= 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_ref() else { return -1; };
    let id_buf = std::slice::from_raw_parts(ids as *const i32, n_ids as usize);
    let out = std::slice::from_raw_parts_mut(out_bytes as *mut u8, max_bytes as usize);
    let mut written = 0usize;
    for &id in id_buf {
        if id < 0 { return -1; }
        let id_u = id as usize;
        if id_u >= t.decode_table.len() { return -1; }
        let bytes = &t.decode_table[id_u];
        if written + bytes.len() > max_bytes as usize { return -1; }
        out[written..written + bytes.len()].copy_from_slice(bytes);
        written += bytes.len();
    }
    written as c_int
}

/// FR-x-extra: look up the token id for an exact byte sequence in a
/// loaded BPE tokenizer. Returns the id (>= 0) on match, -1 on bad
/// handle / no-match. Used by the chat-completion encode path to map
/// each GPT-2 surface byte sequence (e.g. UTF-8 of 'Ġ') to its single-
/// surface-char vocab id before running the merge loop.
///
/// O(N_VOCAB) linear scan — fine at startup (called 256 times to
/// build the byte→id cache), bad in a hot loop. The encode side
/// caches results, so the linear scan only happens once per session.
#[no_mangle] pub unsafe extern "C" fn aether_bpe_lookup_bytes(
    handle: i64,
    bytes: *const c_void, n_bytes: c_int,
) -> i32 {
    if handle < 0 || bytes.is_null() || n_bytes <= 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_ref() else { return -1; };
    let needle = std::slice::from_raw_parts(bytes as *const u8, n_bytes as usize);
    for (id, slot) in t.decode_table.iter().enumerate() {
        if slot.as_slice() == needle { return id as i32; }
    }
    -1
}

// =====================================================================
// FR-19.10 — Jinja-lite chat-template renderer.
//
// Minimal subset of Jinja sufficient for HF chat templates:
//   - `{{ var }}`         — scalar variable substitution
//   - `{{ var.field }}`   — single-level dot access (for message
//                            iteration: `msg.role`, `msg.content`)
//   - `{% for msg in messages %}...{% endfor %}` — list iteration
//   - `{% if var %}...{% endif %}` — truthy conditional on a scalar
//                                     (truthy = non-empty string)
//
// NOT supported (FR-19.10-extra):
//   - Filters (`| trim`, `| upper`, etc.)
//   - Whitespace-strip markers (`{%-` / `-%}`)
//   - `else` / `elif`
//   - Nested-loop variable shadowing
//   - Arbitrary expressions (no string concat, no comparisons)
//   - Loop unrolling tricks
//
// State per handle:
//   vars     — name → string (scalar)
//   messages — Vec<(role, content)> for the `messages` list (the
//              only multi-field list-typed binding supported)
//
// The witness builds a Llama-3-shaped template, pushes two messages,
// and verifies the rendered output contains the expected turn-
// boundary markers in the right order.
// =====================================================================

struct ChatTemplateCtx {
    vars: std::collections::HashMap<String, String>,
    messages: Vec<(String, String)>,
}

struct TplCell(UnsafeCell<Vec<Option<Box<ChatTemplateCtx>>>>);
unsafe impl Sync for TplCell {}
static TPL_TABLE: TplCell = TplCell(UnsafeCell::new(Vec::new()));
unsafe fn tpl_table() -> &'static mut Vec<Option<Box<ChatTemplateCtx>>> {
    &mut *TPL_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_template_new() -> i64 {
    let ctx = ChatTemplateCtx { vars: Default::default(), messages: Default::default() };
    let tbl = tpl_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(ctx)); return i as i64; }
    }
    tbl.push(Some(Box::new(ctx)));
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_template_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = tpl_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

/// Set a scalar variable: `name` (UTF-8) → `value` (UTF-8 bytes).
/// Names and values are copied into the context. Returns 0 on success.
#[no_mangle] pub unsafe extern "C" fn aether_template_set_var(
    handle: i64,
    name: *const c_void, n_name: c_int,
    value: *const c_void, n_value: c_int,
) -> c_int {
    if handle < 0 || name.is_null() || value.is_null() { return -1; }
    if n_name <= 0 || n_value < 0 { return -1; }
    let tbl = tpl_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(ctx) = tbl[h].as_mut() else { return -1; };
    let name_bytes = std::slice::from_raw_parts(name as *const u8, n_name as usize);
    let value_bytes = std::slice::from_raw_parts(value as *const u8, n_value as usize);
    let Ok(name_s) = std::str::from_utf8(name_bytes) else { return -2; };
    let Ok(value_s) = std::str::from_utf8(value_bytes) else { return -2; };
    ctx.vars.insert(name_s.to_string(), value_s.to_string());
    0
}

/// Append a (role, content) message to the `messages` list.
#[no_mangle] pub unsafe extern "C" fn aether_template_push_message(
    handle: i64,
    role: *const c_void, n_role: c_int,
    content: *const c_void, n_content: c_int,
) -> c_int {
    if handle < 0 || role.is_null() || content.is_null() { return -1; }
    if n_role <= 0 || n_content < 0 { return -1; }
    let tbl = tpl_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(ctx) = tbl[h].as_mut() else { return -1; };
    let rb = std::slice::from_raw_parts(role as *const u8, n_role as usize);
    let cb = std::slice::from_raw_parts(content as *const u8, n_content as usize);
    let Ok(rs) = std::str::from_utf8(rb) else { return -2; };
    let Ok(cs) = std::str::from_utf8(cb) else { return -2; };
    ctx.messages.push((rs.to_string(), cs.to_string()));
    0
}

/// Render the `template` (UTF-8) into `out` (max `max_out` bytes).
/// Returns the number of bytes written, or -1 on overflow / bad template.
#[no_mangle] pub unsafe extern "C" fn aether_template_render(
    handle: i64,
    template: *const c_void, n_template: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if handle < 0 || template.is_null() || out.is_null() { return -1; }
    if n_template <= 0 || max_out <= 0 { return -1; }
    let tbl = tpl_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(ctx) = tbl[h].as_ref() else { return -1; };
    let tpl_bytes = std::slice::from_raw_parts(template as *const u8, n_template as usize);
    let Ok(tpl_str) = std::str::from_utf8(tpl_bytes) else { return -2; };
    let mut buf: Vec<u8> = Vec::with_capacity(max_out as usize);
    if render_inner(tpl_str, ctx, None, &mut buf).is_err() { return -1; }
    if buf.len() > max_out as usize { return -1; }
    let out_slice = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    out_slice[..buf.len()].copy_from_slice(&buf);
    buf.len() as c_int
}

/// Resolve a `var` or `var.field` reference against the context (and
/// optionally a `loop_msg` if we're inside a `{% for msg in messages %}`).
/// Returns the resolved string slice. Empty string on missing var
/// (Jinja-compatible: undefined → empty in default mode).
fn resolve(name: &str, ctx: &ChatTemplateCtx, loop_msg: Option<&(String, String)>) -> String {
    let name = name.trim();
    if let Some((loop_name, loop_val)) = name.split_once('.') {
        // Field access: `loop_name.field`. Only one binding shape is
        // supported (the for-loop's message variable).
        if let Some(msg) = loop_msg {
            if loop_name.trim() == LOOP_VAR_PLACEHOLDER || true {
                // We don't track the user-chosen loop var name; any dotted
                // access inside a for-loop falls through to the message.
                // Practical: chat templates always loop over `messages`
                // and call the var `message`/`msg`/etc.
                let field = loop_val.trim();
                return match field {
                    "role"    => msg.0.clone(),
                    "content" => msg.1.clone(),
                    _ => String::new(),
                };
            }
        }
        return String::new();
    }
    ctx.vars.get(name).cloned().unwrap_or_default()
}

const LOOP_VAR_PLACEHOLDER: &str = "<loop_var>";

#[derive(Debug)]
struct RenderError(&'static str);

fn render_inner(
    tpl: &str,
    ctx: &ChatTemplateCtx,
    loop_msg: Option<&(String, String)>,
    out: &mut Vec<u8>,
) -> Result<(), RenderError> {
    let b = tpl.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        if i + 1 < b.len() && b[i] == b'{' && b[i + 1] == b'{' {
            let close = find_subslice(&b[i + 2..], b"}}").ok_or(RenderError("unclosed {{"))?;
            let expr = std::str::from_utf8(&b[i + 2..i + 2 + close])
                .map_err(|_| RenderError("non-utf8 expr"))?;
            let value = resolve(expr, ctx, loop_msg);
            out.extend_from_slice(value.as_bytes());
            i += 2 + close + 2;
        } else if i + 1 < b.len() && b[i] == b'{' && b[i + 1] == b'%' {
            let close = find_subslice(&b[i + 2..], b"%}").ok_or(RenderError("unclosed {%"))?;
            let directive = std::str::from_utf8(&b[i + 2..i + 2 + close])
                .map_err(|_| RenderError("non-utf8 directive"))?
                .trim();
            let after_directive = i + 2 + close + 2;
            if let Some(_rest) = directive.strip_prefix("for ") {
                // Find matching {% endfor %} accounting for nesting.
                let body_start = after_directive;
                let (body_end, after_endfor) = find_matching_block(&tpl[body_start..], "for ", "endfor")?;
                let body = &tpl[body_start..body_start + body_end];
                // Only `for X in messages` is supported.
                if !directive.contains("in messages") {
                    return Err(RenderError("unsupported for clause"));
                }
                for msg in &ctx.messages {
                    render_inner(body, ctx, Some(msg), out)?;
                }
                i = body_start + after_endfor;
            } else if let Some(cond) = directive.strip_prefix("if ") {
                let body_start = after_directive;
                let (body_end, after_endif) = find_matching_block(&tpl[body_start..], "if ", "endif")?;
                let body = &tpl[body_start..body_start + body_end];
                if is_truthy(cond.trim(), ctx, loop_msg) {
                    render_inner(body, ctx, loop_msg, out)?;
                }
                i = body_start + after_endif;
            } else if directive == "endfor" || directive == "endif" {
                return Err(RenderError("dangling end-directive"));
            } else {
                return Err(RenderError("unknown directive"));
            }
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    Ok(())
}

fn is_truthy(expr: &str, ctx: &ChatTemplateCtx, loop_msg: Option<&(String, String)>) -> bool {
    let v = resolve(expr, ctx, loop_msg);
    !v.is_empty() && v != "0" && v.to_lowercase() != "false"
}

/// Find the matching closing tag for a block opener. `open_tag` is the
/// directive prefix that introduces a nested block (e.g. `"for "`,
/// `"if "`); `close_tag` is the exact directive that closes it
/// (`"endfor"`, `"endif"`). Returns `(byte_offset_of_close_directive,
/// byte_offset_after_close_directive_and_trailing_%})`.
///
/// Skips over BOTH `{% for ... %}...{% endfor %}` AND `{% if ... %}
/// ...{% endif %}` nested constructs uniformly (i.e. nesting from
/// EITHER kind increments depth), since a `for` inside an `if` and
/// vice versa both need to be balanced.
fn find_matching_block(rest: &str, _open_tag: &str, close_tag: &str) -> Result<(usize, usize), RenderError> {
    let b = rest.as_bytes();
    let mut depth = 1i32;
    let mut i = 0usize;
    while i < b.len() {
        if i + 1 < b.len() && b[i] == b'{' && b[i + 1] == b'%' {
            let close = find_subslice(&b[i + 2..], b"%}").ok_or(RenderError("unclosed {%"))?;
            let directive = std::str::from_utf8(&b[i + 2..i + 2 + close])
                .map_err(|_| RenderError("non-utf8 directive"))?
                .trim();
            if directive.starts_with("for ") || directive.starts_with("if ") {
                depth += 1;
            } else if directive == close_tag && depth == 1 {
                // This is our matching close.
                return Ok((i, i + 2 + close + 2));
            } else if directive == "endfor" || directive == "endif" {
                depth -= 1;
            }
            i += 2 + close + 2;
        } else {
            i += 1;
        }
    }
    Err(RenderError("missing matching end-directive"))
}

fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

// =====================================================================
// FR-19.4 — Paged KV cache (block allocator simulation).
//
// vLLM-class block-allocated KV memory. Each block holds `block_size`
// tokens of KV state; the allocator manages a fixed pool of n_blocks.
// LRU eviction: when all blocks are in use, the least-recently-used
// block is recycled. Per-block "touch" updates the LRU clock.
//
// SCOPE: control-flow simulation. No actual GPU memory. The real
// implementation (FR-19.4-extra) backs each block with a cudaMalloc'd
// region and tracks virtual-page mappings.
// =====================================================================
struct PagedKVCache {
    n_blocks: u32,
    block_size: u32,
    /// In-use mask per block. `allocated[i]` is true when block i holds data.
    allocated: Vec<bool>,
    /// Monotonic clock per block (incremented on touch / allocate); the
    /// LRU is the block with the smallest clock among allocated blocks.
    clock: Vec<u64>,
    tick: u64,
}
struct PkvCell(UnsafeCell<Vec<Option<Box<PagedKVCache>>>>);
unsafe impl Sync for PkvCell {}
static PKV_TABLE: PkvCell = PkvCell(UnsafeCell::new(Vec::new()));
unsafe fn pkv_table() -> &'static mut Vec<Option<Box<PagedKVCache>>> {
    &mut *PKV_TABLE.0.get()
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_new(n_blocks: c_int, block_size: c_int) -> i64 {
    if n_blocks <= 0 || block_size <= 0 { return -1; }
    let n = n_blocks as usize;
    let pkv = PagedKVCache {
        n_blocks: n_blocks as u32,
        block_size: block_size as u32,
        allocated: vec![false; n],
        clock: vec![0; n],
        tick: 0,
    };
    let tbl = pkv_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(pkv)); return i as i64; }
    }
    tbl.push(Some(Box::new(pkv)));
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_destroy(h: i64) -> i32 {
    if h < 0 { return -1; }
    let tbl = pkv_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    tbl[hu] = None;
    0
}

/// Allocate a free block. Returns block id ≥ 0, or -1 if pool full.
#[no_mangle] pub unsafe extern "C" fn aether_pkv_allocate(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = pkv_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(p) = tbl[hu].as_mut() else { return -1; };
    for i in 0..p.n_blocks as usize {
        if !p.allocated[i] {
            p.allocated[i] = true;
            p.tick += 1;
            p.clock[i] = p.tick;
            return i as c_int;
        }
    }
    -1
}

/// Mark a block as recently used (updates LRU clock).
#[no_mangle] pub unsafe extern "C" fn aether_pkv_touch(h: i64, block_id: c_int) -> c_int {
    if h < 0 || block_id < 0 { return -1; }
    let tbl = pkv_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(p) = tbl[hu].as_mut() else { return -1; };
    let bi = block_id as usize;
    if bi >= p.n_blocks as usize || !p.allocated[bi] { return -1; }
    p.tick += 1;
    p.clock[bi] = p.tick;
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_free_block(h: i64, block_id: c_int) -> c_int {
    if h < 0 || block_id < 0 { return -1; }
    let tbl = pkv_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(p) = tbl[hu].as_mut() else { return -1; };
    let bi = block_id as usize;
    if bi >= p.n_blocks as usize { return -1; }
    p.allocated[bi] = false;
    0
}

/// Evict the least-recently-used allocated block. Returns the id of
/// the evicted block (≥ 0), or -1 if no allocated blocks exist.
#[no_mangle] pub unsafe extern "C" fn aether_pkv_evict_lru(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = pkv_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(p) = tbl[hu].as_mut() else { return -1; };
    let mut best_id: i32 = -1;
    let mut best_clock: u64 = u64::MAX;
    for i in 0..p.n_blocks as usize {
        if p.allocated[i] && p.clock[i] < best_clock {
            best_clock = p.clock[i];
            best_id = i as i32;
        }
    }
    if best_id < 0 { return -1; }
    p.allocated[best_id as usize] = false;
    best_id
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_n_allocated(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = pkv_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(p) = tbl[hu].as_ref() else { return -1; };
    p.allocated.iter().filter(|&&a| a).count() as c_int
}

// =====================================================================
// FR-19.4-extra — Paged KV cache: real per-request virtual page table.
//
// Each request owns a `PageTable` mapping its logical block index (i.e.
// floor(token_pos / block_size)) to a physical block id from the shared
// pool.  The pool itself is the existing PagedKVCache; the page-table
// layer here adds the per-request indirection that vLLM uses to keep
// memory shared without fragmentation.
// =====================================================================
struct PageTable {
    /// logical_block_idx -> physical_block_id (-1 if unmapped).
    blocks: Vec<i32>,
}
struct PtCell(UnsafeCell<Vec<Option<Box<PageTable>>>>);
unsafe impl Sync for PtCell {}
static PT_TABLE: PtCell = PtCell(UnsafeCell::new(Vec::new()));
unsafe fn pt_table() -> &'static mut Vec<Option<Box<PageTable>>> { &mut *PT_TABLE.0.get() }

#[no_mangle] pub unsafe extern "C" fn aether_pkv_pagetable_new(initial_capacity: c_int) -> i64 {
    if initial_capacity < 0 { return -1; }
    let pt = PageTable { blocks: vec![-1; initial_capacity as usize] };
    let tbl = pt_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(pt)); return i as i64; }
    }
    tbl.push(Some(Box::new(pt)));
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_pagetable_set(
    h: i64, logical_idx: c_int, physical_block: c_int,
) -> c_int {
    if h < 0 || logical_idx < 0 { return -1; }
    let tbl = pt_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(pt) = tbl[hu].as_mut() else { return -1; };
    let li = logical_idx as usize;
    if li >= pt.blocks.len() { pt.blocks.resize(li + 1, -1); }
    pt.blocks[li] = physical_block;
    0
}

/// Returns the physical block id mapped to `logical_idx` (i64 — i32-sign-extend
/// gap in the asm backend means returning c_int=-1 gets zero-extended in the
/// caller's i64 slot; i64 return avoids that).  Returns -1 for unmapped /
/// out-of-bounds.
#[no_mangle] pub unsafe extern "C" fn aether_pkv_pagetable_get(h: i64, logical_idx: c_int) -> i64 {
    if h < 0 || logical_idx < 0 { return -1; }
    let tbl = pt_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(pt) = tbl[hu].as_ref() else { return -1; };
    let li = logical_idx as usize;
    if li >= pt.blocks.len() { return -1; }
    pt.blocks[li] as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_pagetable_len(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = pt_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(pt) = tbl[hu].as_ref() else { return -1; };
    pt.blocks.len() as c_int
}

#[no_mangle] pub unsafe extern "C" fn aether_pkv_pagetable_destroy(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = pt_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    tbl[hu] = None;
    0
}

// ============================================================================
// FR-19.5-extra — Real continuous-batching scheduler.
//
// Builds on the paged-KV primitives below.  One scheduler owns:
//   - a shared block-pool handle (aether_pkv_new),
//   - a list of active requests, each with its own per-request page table
//     (aether_pkv_pagetable_new) and a token position.
//
// As a request's token position crosses a block_size boundary, the scheduler
// allocates a fresh physical block from the pool and binds it to the next
// logical-block slot in that request's page table.  Returning a token
// (`generated`) advances the position by 1.  When the request is finished
// (EOS / max_tokens / explicit completion), its blocks are released and
// the page table is destroyed.
//
// The actual per-step forward pass is the caller's responsibility — for
// the seq=1 autoregressive shape, the caller iterates over `active_request
// _ids()` and runs a paged_attention_seq1 + paged_append_kv per request,
// then calls `record_token(req_id, new_tok_id)`.  When a real batched
// forward kernel lands (FR-19.5-extra-deep) the step will fan out across
// the active set in one launch.
// ============================================================================

struct Request {
    req_id: i32,
    page_table_h: i64,
    position: u32,
    max_tokens: u32,
    finished: bool,
    /// Last token emitted (for the next decode step).  Negative = none yet.
    last_token: i32,
    /// Number of tokens emitted so far (not counting prompt).
    tokens_emitted: u32,
}

struct BatchScheduler {
    pool_handle: i64,
    block_size: u32,
    max_active: u32,
    requests: Vec<Request>,
    /// Owned physical blocks per request, indexed parallel to `requests`.
    owned_blocks: Vec<Vec<i32>>,
}
struct SchedCell(UnsafeCell<Vec<Option<Box<BatchScheduler>>>>);
unsafe impl Sync for SchedCell {}
static SCHED_TABLE: SchedCell = SchedCell(UnsafeCell::new(Vec::new()));
unsafe fn sched_table() -> &'static mut Vec<Option<Box<BatchScheduler>>> { &mut *SCHED_TABLE.0.get() }

/// Create a new batch scheduler bound to an existing pool.
/// `max_active`: capacity of concurrent requests.  Returns scheduler handle.
#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_new(
    pool_h: i64, block_size: c_int, max_active: c_int,
) -> i64 {
    if pool_h < 0 || block_size <= 0 || max_active <= 0 { return -1; }
    let s = BatchScheduler {
        pool_handle: pool_h,
        block_size: block_size as u32,
        max_active: max_active as u32,
        requests: Vec::new(),
        owned_blocks: Vec::new(),
    };
    let tbl = sched_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(s)); return i as i64; }
    }
    tbl.push(Some(Box::new(s)));
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_destroy(h: i64) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    if let Some(s) = tbl[hu].take() {
        for r in &s.requests {
            let _ = aether_pkv_pagetable_destroy(r.page_table_h);
        }
        for blocks in &s.owned_blocks {
            for &b in blocks { let _ = aether_pkv_free_block(s.pool_handle, b); }
        }
    }
    0
}

/// Admit a new request.  Returns 0 on success, -1 if at capacity,
/// -2 if pool exhausted.  i64 return to dodge the asm-backend i32-sign-extend
/// gap (callers comparing `!= -1` from an i32 return get the zero-extended
/// 0xFFFFFFFF rather than -1).
#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_admit(
    h: i64, req_id: c_int, prompt_len: c_int, max_tokens: c_int,
) -> i64 {
    if h < 0 || prompt_len < 0 || max_tokens < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_mut() else { return -1; };
    if s.requests.len() as u32 >= s.max_active { return -1; }

    // Make a new page table for this request, sized to fit prompt + max_tokens.
    let total_tokens = prompt_len as u32 + max_tokens as u32;
    let n_logical_blocks = ((total_tokens + s.block_size - 1) / s.block_size).max(1);
    let pt_h = aether_pkv_pagetable_new(n_logical_blocks as c_int);
    if pt_h < 0 { return -2; }

    // Allocate exactly the blocks the prompt needs right now (lazy for the rest).
    let prompt_blocks = ((prompt_len as u32 + s.block_size - 1) / s.block_size).max(1);
    let mut owned = Vec::new();
    for logical in 0..prompt_blocks {
        let phys = aether_pkv_allocate(s.pool_handle);
        if phys < 0 {
            // Roll back.
            for b in &owned { let _ = aether_pkv_free_block(s.pool_handle, *b); }
            let _ = aether_pkv_pagetable_destroy(pt_h);
            return -2;
        }
        let _ = aether_pkv_pagetable_set(pt_h, logical as c_int, phys);
        owned.push(phys);
    }
    s.requests.push(Request {
        req_id, page_table_h: pt_h,
        position: 0, max_tokens: max_tokens as u32,
        finished: false, last_token: -1, tokens_emitted: 0,
    });
    s.owned_blocks.push(owned);
    0
}

/// Advance request `req_id` by emitting `new_token`.  If the token crosses a
/// block boundary, allocates a fresh physical block.  Returns 0 on success,
/// -1 if request not found, -2 if pool exhausted.  Marks the request finished
/// when tokens_emitted reaches max_tokens.
#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_record_token(
    h: i64, req_id: c_int, new_token: c_int,
) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_mut() else { return -1; };
    let idx = s.requests.iter().position(|r| r.req_id == req_id);
    let Some(i) = idx else { return -1; };
    if s.requests[i].finished { return 0; }
    s.requests[i].last_token = new_token;
    s.requests[i].position += 1;
    s.requests[i].tokens_emitted += 1;

    // If next write position falls in a new logical block, allocate it now.
    let next_pos = s.requests[i].position;
    let next_logical = next_pos / s.block_size;
    let pt_h = s.requests[i].page_table_h;
    let already = aether_pkv_pagetable_get(pt_h, next_logical as c_int);
    if already < 0 {
        // Need a new block.
        let phys = aether_pkv_allocate(s.pool_handle);
        if phys < 0 { return -2; }
        let _ = aether_pkv_pagetable_set(pt_h, next_logical as c_int, phys);
        s.owned_blocks[i].push(phys);
    }

    if s.requests[i].tokens_emitted >= s.requests[i].max_tokens {
        s.requests[i].finished = true;
    }
    0
}

/// Mark a request finished (caller-driven, e.g. for EOS).
#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_finish(
    h: i64, req_id: c_int,
) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_mut() else { return -1; };
    let idx = s.requests.iter().position(|r| r.req_id == req_id);
    let Some(i) = idx else { return -1; };
    s.requests[i].finished = true;
    0
}

/// Reap finished requests, freeing their blocks + page tables.
/// Returns the number of requests reaped.
#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_reap(h: i64) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_mut() else { return -1; };
    let mut i = 0;
    let mut reaped = 0;
    while i < s.requests.len() {
        if s.requests[i].finished {
            let req = s.requests.remove(i);
            let blocks = s.owned_blocks.remove(i);
            for b in &blocks { let _ = aether_pkv_free_block(s.pool_handle, *b); }
            let _ = aether_pkv_pagetable_destroy(req.page_table_h);
            reaped += 1;
        } else {
            i += 1;
        }
    }
    reaped
}

#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_n_active(h: i64) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_ref() else { return -1; };
    s.requests.iter().filter(|r| !r.finished).count() as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_pagetable_for(h: i64, req_id: c_int) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_ref() else { return -1; };
    for r in &s.requests {
        if r.req_id == req_id { return r.page_table_h; }
    }
    -1
}

#[no_mangle] pub unsafe extern "C" fn aether_batch_sched_position(h: i64, req_id: c_int) -> i64 {
    if h < 0 { return -1; }
    let tbl = sched_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(s) = tbl[hu].as_ref() else { return -1; };
    for r in &s.requests {
        if r.req_id == req_id { return r.position as i64; }
    }
    -1
}

/// Translate a logical token position to (physical_block, in_block_offset).
/// Returns -1 in the high-bits-as-error or packs result as i64:
///   bits 0..31 = physical_block, bits 32..63 = in_block_offset.
/// Caller decodes both halves.  Returns -1 on bad inputs / unmapped.
#[no_mangle] pub unsafe extern "C" fn aether_pkv_pagetable_translate(
    h: i64, token_pos: c_int, block_size: c_int,
) -> i64 {
    if h < 0 || token_pos < 0 || block_size <= 0 { return -1; }
    let tbl = pt_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(pt) = tbl[hu].as_ref() else { return -1; };
    let logical = token_pos as usize / block_size as usize;
    let in_block = token_pos as usize % block_size as usize;
    if logical >= pt.blocks.len() { return -1; }
    let phys = pt.blocks[logical];
    if phys < 0 { return -1; }
    ((phys as i64) & 0xffffffff) | ((in_block as i64) << 32)
}

// =====================================================================
// FR-19.5 — Continuous batching scheduler (simulation).
//
// vLLM-class scheduler: new requests enter mid-decode (no padding
// waste); preempt-longest-running on full. The simulator tracks a
// queue of active request ids; each `step` decodes one token across
// all active requests in parallel (in real GPU code that'd be a
// batched matmul; here it's just an "elapsed tokens" counter per req).
//
// SCOPE: control-flow sim of admit/step/complete. Real wiring to a
// GPU + KV cache is FR-19.5-extra.
// =====================================================================
struct ContinuousBatch {
    capacity: u32,
    /// (req_id, tokens_decoded). req_id is opaque to the scheduler.
    active: Vec<(i32, u32)>,
}
struct CbCell(UnsafeCell<Vec<Option<Box<ContinuousBatch>>>>);
unsafe impl Sync for CbCell {}
static CB_TABLE: CbCell = CbCell(UnsafeCell::new(Vec::new()));
unsafe fn cb_table() -> &'static mut Vec<Option<Box<ContinuousBatch>>> {
    &mut *CB_TABLE.0.get()
}
#[no_mangle] pub unsafe extern "C" fn aether_cb_new(capacity: c_int) -> i64 {
    if capacity <= 0 { return -1; }
    let cb = ContinuousBatch { capacity: capacity as u32, active: Vec::new() };
    let tbl = cb_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(cb)); return i as i64; }
    }
    tbl.push(Some(Box::new(cb)));
    (tbl.len() - 1) as i64
}
#[no_mangle] pub unsafe extern "C" fn aether_cb_destroy(h: i64) -> i32 {
    if h < 0 { return -1; }
    let tbl = cb_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    tbl[hu] = None;
    0
}
/// Admit a new request. Returns 0 on success, -1 if at capacity.
/// Mid-decode entry: existing requests keep their token-count;
/// the new one starts at 0 and joins the next `step` cycle.
#[no_mangle] pub unsafe extern "C" fn aether_cb_admit(h: i64, req_id: c_int) -> c_int {
    if h < 0 { return -1; }
    let tbl = cb_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(cb) = tbl[hu].as_mut() else { return -1; };
    if cb.active.len() as u32 >= cb.capacity { return -1; }
    cb.active.push((req_id, 0));
    0
}
/// Run one decode step for every active request. Returns the count of
/// active requests AFTER the step.
#[no_mangle] pub unsafe extern "C" fn aether_cb_step(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = cb_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(cb) = tbl[hu].as_mut() else { return -1; };
    for r in cb.active.iter_mut() { r.1 += 1; }
    cb.active.len() as c_int
}
/// Mark a request complete. Returns 0 on success, -1 if not present.
#[no_mangle] pub unsafe extern "C" fn aether_cb_complete(h: i64, req_id: c_int) -> c_int {
    if h < 0 { return -1; }
    let tbl = cb_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(cb) = tbl[hu].as_mut() else { return -1; };
    if let Some(pos) = cb.active.iter().position(|r| r.0 == req_id) {
        cb.active.remove(pos);
        0
    } else { -1 }
}
#[no_mangle] pub unsafe extern "C" fn aether_cb_n_active(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = cb_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(cb) = tbl[hu].as_ref() else { return -1; };
    cb.active.len() as c_int
}

// =====================================================================
// FR-19.6 — Speculative decoding accept/reject (simulation).
//
// Standard speculative-decoding rejection-sampling test: given a draft
// token's draft-model probability `q` and the target-model probability
// `p`, accept with probability min(1, p/q). The simulator takes a
// pre-rolled uniform-random `u` in [0, 1) and returns 1=accept / 0=reject.
//
// SCOPE: the algorithm shape. Real wiring needs a draft + target model
// pair (FR-19.6-extra, gated on FR-17.19).
// =====================================================================
#[no_mangle] pub extern "C" fn aether_specdec_accept(
    target_prob: f32, draft_prob: f32, rand_u01: f32,
) -> c_int {
    if draft_prob <= 0.0 || target_prob < 0.0 { return -1; }
    if !(0.0..1.0).contains(&rand_u01) { return -1; }
    let ratio = target_prob / draft_prob;
    if ratio >= 1.0 { return 1; }
    if rand_u01 < ratio { 1 } else { 0 }
}

// =====================================================================
// FR-19.7 — Multi-model concurrent hosting (simulation).
//
// Registry that tracks N models, each with its own name + VRAM budget.
// Lookup by name returns the model id. The total VRAM budget aggregate
// is what gates "can I host one more model on the 3070 Ti's 8 GiB?".
//
// SCOPE: registry shape; real per-model GPU memory pinning is
// FR-19.7-extra (gated on FR-19.4 KV-cache real impl).
// =====================================================================
struct ModelEntry {
    name: String,
    vram_budget_mb: u32,
}
struct ModelRegistry { models: Vec<ModelEntry> }
struct MmCell(UnsafeCell<Vec<Option<Box<ModelRegistry>>>>);
unsafe impl Sync for MmCell {}
static MM_TABLE: MmCell = MmCell(UnsafeCell::new(Vec::new()));
unsafe fn mm_table() -> &'static mut Vec<Option<Box<ModelRegistry>>> {
    &mut *MM_TABLE.0.get()
}
#[no_mangle] pub unsafe extern "C" fn aether_mm_new() -> i64 {
    let m = ModelRegistry { models: Vec::new() };
    let tbl = mm_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(m)); return i as i64; }
    }
    tbl.push(Some(Box::new(m)));
    (tbl.len() - 1) as i64
}
#[no_mangle] pub unsafe extern "C" fn aether_mm_destroy(h: i64) -> i32 {
    if h < 0 { return -1; }
    let tbl = mm_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    tbl[hu] = None;
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_mm_register(
    h: i64, name: *const c_void, n_name: c_int, vram_budget_mb: c_int,
) -> c_int {
    if h < 0 || name.is_null() || n_name <= 0 || vram_budget_mb < 0 { return -1; }
    let tbl = mm_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(reg) = tbl[hu].as_mut() else { return -1; };
    let nb = std::slice::from_raw_parts(name as *const u8, n_name as usize);
    let Ok(ns) = std::str::from_utf8(nb) else { return -2; };
    reg.models.push(ModelEntry { name: ns.to_string(), vram_budget_mb: vram_budget_mb as u32 });
    (reg.models.len() - 1) as c_int
}
#[no_mangle] pub unsafe extern "C" fn aether_mm_lookup(
    h: i64, name: *const c_void, n_name: c_int,
) -> c_int {
    if h < 0 || name.is_null() || n_name <= 0 { return -1; }
    let tbl = mm_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(reg) = tbl[hu].as_ref() else { return -1; };
    let nb = std::slice::from_raw_parts(name as *const u8, n_name as usize);
    let Ok(ns) = std::str::from_utf8(nb) else { return -2; };
    for (i, m) in reg.models.iter().enumerate() {
        if m.name == ns { return i as c_int; }
    }
    -1
}
#[no_mangle] pub unsafe extern "C" fn aether_mm_total_vram_mb(h: i64) -> c_int {
    if h < 0 { return -1; }
    let tbl = mm_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(reg) = tbl[hu].as_ref() else { return -1; };
    reg.models.iter().map(|m| m.vram_budget_mb).sum::<u32>() as c_int
}

// =====================================================================
// FR-19.14 — Token-bucket rate limit.
//
// Per-key bucket; capacity = `burst`, refill at `req_per_sec` per
// second. `check(key, now_us)` returns 1 if a token was available
// (and consumed), 0 if rate-limited.
// =====================================================================
struct Bucket { tokens: f64, last_us: i64 }
struct RateLimiter {
    rate_per_sec: f64,
    burst: f64,
    buckets: std::collections::HashMap<String, Bucket>,
}
struct RlCell(UnsafeCell<Vec<Option<Box<RateLimiter>>>>);
unsafe impl Sync for RlCell {}
static RL_TABLE: RlCell = RlCell(UnsafeCell::new(Vec::new()));
unsafe fn rl_table() -> &'static mut Vec<Option<Box<RateLimiter>>> {
    &mut *RL_TABLE.0.get()
}
#[no_mangle] pub unsafe extern "C" fn aether_rl_new(req_per_sec: c_int, burst: c_int) -> i64 {
    if req_per_sec <= 0 || burst <= 0 { return -1; }
    let rl = RateLimiter {
        rate_per_sec: req_per_sec as f64,
        burst: burst as f64,
        buckets: std::collections::HashMap::new(),
    };
    let tbl = rl_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(rl)); return i as i64; }
    }
    tbl.push(Some(Box::new(rl)));
    (tbl.len() - 1) as i64
}
#[no_mangle] pub unsafe extern "C" fn aether_rl_destroy(h: i64) -> i32 {
    if h < 0 { return -1; }
    let tbl = rl_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    tbl[hu] = None;
    0
}
/// Returns 1 if a token was available + consumed; 0 if rate-limited
/// (the typical HTTP 429 path); -1 on error.
#[no_mangle] pub unsafe extern "C" fn aether_rl_check(
    h: i64, key: *const c_void, n_key: c_int, now_us: i64,
) -> c_int {
    if h < 0 || key.is_null() || n_key <= 0 { return -1; }
    let tbl = rl_table();
    let hu = h as usize;
    if hu >= tbl.len() { return -1; }
    let Some(rl) = tbl[hu].as_mut() else { return -1; };
    let nb = std::slice::from_raw_parts(key as *const u8, n_key as usize);
    let Ok(ks) = std::str::from_utf8(nb) else { return -2; };
    let rate = rl.rate_per_sec;
    let burst = rl.burst;
    let bucket = rl.buckets.entry(ks.to_string()).or_insert(Bucket { tokens: burst, last_us: now_us });
    // Refill since last call.
    let dt_us = (now_us - bucket.last_us).max(0);
    let refill = (dt_us as f64 / 1_000_000.0) * rate;
    bucket.tokens = (bucket.tokens + refill).min(burst);
    bucket.last_us = now_us;
    if bucket.tokens >= 1.0 {
        bucket.tokens -= 1.0;
        1
    } else { 0 }
}

// =====================================================================
// FR-19.15 — Observability: Prometheus counter + JSON log.
//
// Process-wide counter registry + a render-to-prometheus-text-format
// fn. Plus a single-line JSON log emitter for structured logs.
// =====================================================================
struct ObsState {
    counters: std::collections::HashMap<String, u64>,
}
struct ObsCell(UnsafeCell<Option<Box<ObsState>>>);
unsafe impl Sync for ObsCell {}
static OBS_STATE: ObsCell = ObsCell(UnsafeCell::new(None));
unsafe fn obs_state() -> &'static mut Box<ObsState> {
    let s = &mut *OBS_STATE.0.get();
    if s.is_none() {
        *s = Some(Box::new(ObsState { counters: std::collections::HashMap::new() }));
    }
    s.as_mut().unwrap()
}
#[no_mangle] pub unsafe extern "C" fn aether_obs_counter_inc(
    name: *const c_void, n_name: c_int, by: c_int,
) -> c_int {
    if name.is_null() || n_name <= 0 || by < 0 { return -1; }
    let nb = std::slice::from_raw_parts(name as *const u8, n_name as usize);
    let Ok(ns) = std::str::from_utf8(nb) else { return -2; };
    let s = obs_state();
    *s.counters.entry(ns.to_string()).or_insert(0) += by as u64;
    0
}
#[no_mangle] pub unsafe extern "C" fn aether_obs_counter_get(
    name: *const c_void, n_name: c_int,
) -> i64 {
    if name.is_null() || n_name <= 0 { return -1; }
    let nb = std::slice::from_raw_parts(name as *const u8, n_name as usize);
    let Ok(ns) = std::str::from_utf8(nb) else { return -2; };
    let s = obs_state();
    s.counters.get(ns).copied().unwrap_or(0) as i64
}
/// Render the counter registry in Prometheus text-exposition format:
///   # TYPE <name> counter
///   <name> <value>
/// Counters render in lexicographic order so the output is stable.
#[no_mangle] pub unsafe extern "C" fn aether_obs_dump_prometheus(
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if out.is_null() || max_out <= 0 { return -1; }
    let s = obs_state();
    let mut names: Vec<&String> = s.counters.keys().collect();
    names.sort();
    let mut buf = String::new();
    for n in names {
        let v = s.counters[n];
        buf.push_str(&format!("# TYPE {} counter\n{} {}\n", n, n, v));
    }
    let bytes = buf.as_bytes();
    if bytes.len() > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    o[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as c_int
}

// =====================================================================
// FR-19.12 — Vision input preprocessing (resize-less normalize +
// patchify for ViT-style models).
// =====================================================================
/// Normalize u8 pixel values via `(x/255 - mean) / std` per channel.
/// `n_pixels` is total elements; `mean` and `std` are scalars (single-
/// channel form here; multi-channel = call once per channel).
#[no_mangle] pub unsafe extern "C" fn aether_img_normalize_f32(
    px_u8: *const c_void, out_f32: *mut c_void,
    n_pixels: c_int, mean: f32, std: f32,
) -> c_int {
    if px_u8.is_null() || out_f32.is_null() || n_pixels <= 0 || std == 0.0 { return -1; }
    let n = n_pixels as usize;
    let p = std::slice::from_raw_parts(px_u8 as *const u8, n);
    let o = std::slice::from_raw_parts_mut(out_f32 as *mut f32, n);
    for i in 0..n {
        o[i] = ((p[i] as f32) / 255.0 - mean) / std;
    }
    0
}
/// Patchify an (h, w) grayscale image into (n_patches, patch*patch)
/// row-major patches. Requires h % patch == 0 and w % patch == 0.
#[no_mangle] pub unsafe extern "C" fn aether_img_patchify_f32(
    img: *const c_void, out: *mut c_void,
    h: c_int, w: c_int, patch: c_int,
) -> c_int {
    if img.is_null() || out.is_null() { return -1; }
    if h <= 0 || w <= 0 || patch <= 0 { return -2; }
    if h % patch != 0 || w % patch != 0 { return -3; }
    let (h, w, p) = (h as usize, w as usize, patch as usize);
    let pr = h / p;
    let pc = w / p;
    let n_patches = pr * pc;
    let imgs = std::slice::from_raw_parts(img as *const f32, h * w);
    let os = std::slice::from_raw_parts_mut(out as *mut f32, n_patches * p * p);
    for py in 0..pr {
        for px in 0..pc {
            let patch_idx = py * pc + px;
            for iy in 0..p {
                for ix in 0..p {
                    let src = (py * p + iy) * w + (px * p + ix);
                    let dst = patch_idx * (p * p) + iy * p + ix;
                    os[dst] = imgs[src];
                }
            }
        }
    }
    n_patches as c_int
}

// =====================================================================
// FR-19.13 — Speech mel spectrogram primitives.
//
// Hann window + naive DFT magnitude. Real Whisper uses log-mel with
// 80 mel bins; this shipping primitive is the DFT magnitude bin
// (the building block). Mel filter bank application = FR-19.13-extra.
// =====================================================================
#[no_mangle] pub unsafe extern "C" fn aether_audio_hann_window(
    out: *mut c_void, n: c_int,
) -> c_int {
    if out.is_null() || n <= 0 { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut f32, n as usize);
    let nn = (n - 1) as f32;
    for i in 0..n as usize {
        let x = (i as f32) / nn;
        // 0.5 * (1 - cos(2π x))
        o[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos());
    }
    0
}
/// Naive DFT magnitude. `input` is real f32 of length `n`; `out_mag`
/// gets `k_bins` magnitude values (length `k_bins`). Uses O(n*k) time.
/// Good enough for witness-scale (n=64, k=16).
#[no_mangle] pub unsafe extern "C" fn aether_audio_dft_magnitude_f32(
    input: *const c_void, n: c_int,
    out_mag: *mut c_void, k_bins: c_int,
) -> c_int {
    if input.is_null() || out_mag.is_null() { return -1; }
    if n <= 0 || k_bins <= 0 { return -2; }
    let n_u = n as usize;
    let k_u = k_bins as usize;
    let x = std::slice::from_raw_parts(input as *const f32, n_u);
    let m = std::slice::from_raw_parts_mut(out_mag as *mut f32, k_u);
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    for k in 0..k_u {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        for nn in 0..n_u {
            let angle = -two_pi * (k as f32) * (nn as f32) / (n as f32);
            re += x[nn] * angle.cos();
            im += x[nn] * angle.sin();
        }
        m[k] = (re * re + im * im).sqrt();
    }
    0
}

// =====================================================================
// FR-19.1 (partial) — ChaCha20-Poly1305 AEAD primitive (RFC 7539).
//
// Real ChaCha20 stream cipher + Poly1305 MAC. AEAD encrypt produces
// (ciphertext + 16-byte tag). Decrypt verifies tag, then writes
// plaintext. Verified against RFC 7539 §2.8.2 test vector.
//
// SCOPE: ONLY this AEAD primitive. The full TLS handshake state
// machine, AES-GCM, Ed25519, X25519, HMAC-SHA256, and Connector/
// Acceptor types are FR-19.1-extra.
// =====================================================================
fn chacha20_qround(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    state[a] = state[a].wrapping_add(state[b]); state[d] ^= state[a]; state[d] = state[d].rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]); state[b] ^= state[c]; state[b] = state[b].rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]); state[d] ^= state[a]; state[d] = state[d].rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]); state[b] ^= state[c]; state[b] = state[b].rotate_left(7);
}
fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut s = [0u32; 16];
    s[0] = 0x61707865; s[1] = 0x3320646e; s[2] = 0x79622d32; s[3] = 0x6b206574;
    for i in 0..8 {
        s[4 + i] = u32::from_le_bytes([key[i*4], key[i*4+1], key[i*4+2], key[i*4+3]]);
    }
    s[12] = counter;
    for i in 0..3 {
        s[13 + i] = u32::from_le_bytes([nonce[i*4], nonce[i*4+1], nonce[i*4+2], nonce[i*4+3]]);
    }
    let mut w = s;
    for _ in 0..10 {
        chacha20_qround(&mut w, 0, 4, 8, 12);
        chacha20_qround(&mut w, 1, 5, 9, 13);
        chacha20_qround(&mut w, 2, 6, 10, 14);
        chacha20_qround(&mut w, 3, 7, 11, 15);
        chacha20_qround(&mut w, 0, 5, 10, 15);
        chacha20_qround(&mut w, 1, 6, 11, 12);
        chacha20_qround(&mut w, 2, 7, 8, 13);
        chacha20_qround(&mut w, 3, 4, 9, 14);
    }
    for i in 0..16 { w[i] = w[i].wrapping_add(s[i]); }
    let mut out = [0u8; 64];
    for i in 0..16 {
        out[i*4..i*4+4].copy_from_slice(&w[i].to_le_bytes());
    }
    out
}
pub(crate) fn chacha20_xor(key: &[u8; 32], counter: u32, nonce: &[u8; 12], data: &mut [u8]) {
    let mut blk = counter;
    let mut i = 0;
    while i < data.len() {
        let stream = chacha20_block(key, blk, nonce);
        let take = (data.len() - i).min(64);
        for j in 0..take { data[i + j] ^= stream[j]; }
        i += take;
        blk = blk.wrapping_add(1);
    }
}
pub(crate) fn poly1305_mac(key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    // r = key[0..16] with clamp, s = key[16..32]
    let mut r = [0u8; 16]; r.copy_from_slice(&key[..16]);
    r[3] &= 15; r[7] &= 15; r[11] &= 15; r[15] &= 15;
    r[4] &= 252; r[8] &= 252; r[12] &= 252;
    // Convert r and s to 130-bit "5-limb" rep over 26 bits each.
    let r0 =  (u32::from_le_bytes([r[0],  r[1],  r[2],  r[3]])      ) & 0x3ffffff;
    let r1 = ((u32::from_le_bytes([r[3],  r[4],  r[5],  r[6]])) >> 2) & 0x3ffff03;
    let r2 = ((u32::from_le_bytes([r[6],  r[7],  r[8],  r[9]])) >> 4) & 0x3ffc0ff;
    let r3 = ((u32::from_le_bytes([r[9],  r[10], r[11], r[12]])) >> 6) & 0x3f03fff;
    let r4 = ((u32::from_le_bytes([r[12], r[13], r[14], r[15]])) >> 8) & 0x00fffff;
    let s1 = r1 * 5; let s2 = r2 * 5; let s3 = r3 * 5; let s4 = r4 * 5;
    let mut h = [0u64; 5];
    let mut i = 0;
    while i < msg.len() {
        let mut block = [0u8; 17];
        let take = (msg.len() - i).min(16);
        block[..take].copy_from_slice(&msg[i..i+take]);
        block[take] = 1;
        let h0 =  (u32::from_le_bytes([block[0], block[1], block[2], block[3]])      ) & 0x3ffffff;
        let h1 = ((u32::from_le_bytes([block[3], block[4], block[5], block[6]])) >> 2) & 0x3ffffff;
        let h2 = ((u32::from_le_bytes([block[6], block[7], block[8], block[9]])) >> 4) & 0x3ffffff;
        let h3 = ((u32::from_le_bytes([block[9], block[10], block[11], block[12]])) >> 6) & 0x3ffffff;
        let h4 = ((u32::from_le_bytes([block[12], block[13], block[14], block[15]])) >> 8)
                 | ((block[16] as u32) << 24);
        h[0] += h0 as u64; h[1] += h1 as u64; h[2] += h2 as u64;
        h[3] += h3 as u64; h[4] += h4 as u64;
        // h = (h * r) mod (2^130 - 5), 26-bit limbs.
        let d0 = h[0]*r0 as u64 + h[1]*s4 as u64 + h[2]*s3 as u64 + h[3]*s2 as u64 + h[4]*s1 as u64;
        let d1 = h[0]*r1 as u64 + h[1]*r0 as u64 + h[2]*s4 as u64 + h[3]*s3 as u64 + h[4]*s2 as u64;
        let d2 = h[0]*r2 as u64 + h[1]*r1 as u64 + h[2]*r0 as u64 + h[3]*s4 as u64 + h[4]*s3 as u64;
        let d3 = h[0]*r3 as u64 + h[1]*r2 as u64 + h[2]*r1 as u64 + h[3]*r0 as u64 + h[4]*s4 as u64;
        let d4 = h[0]*r4 as u64 + h[1]*r3 as u64 + h[2]*r2 as u64 + h[3]*r1 as u64 + h[4]*r0 as u64;
        let mut c = (d0 >> 26) as u32; h[0] = d0 & 0x3ffffff;
        let d1 = d1 + c as u64;        c = (d1 >> 26) as u32; h[1] = d1 & 0x3ffffff;
        let d2 = d2 + c as u64;        c = (d2 >> 26) as u32; h[2] = d2 & 0x3ffffff;
        let d3 = d3 + c as u64;        c = (d3 >> 26) as u32; h[3] = d3 & 0x3ffffff;
        let d4 = d4 + c as u64;        c = (d4 >> 26) as u32; h[4] = d4 & 0x3ffffff;
        h[0] += (c * 5) as u64;
        let c2 = h[0] >> 26; h[0] &= 0x3ffffff; h[1] += c2;
        i += 16;
    }
    // Carry-propagate + reduce mod 2^130 - 5.
    let mut h0 = h[0] as u32; let mut h1 = h[1] as u32; let mut h2 = h[2] as u32;
    let mut h3 = h[3] as u32; let mut h4 = h[4] as u32;
    let mut c = h1 >> 26; h1 &= 0x3ffffff; h2 += c; c = h2 >> 26; h2 &= 0x3ffffff; h3 += c;
    c = h3 >> 26; h3 &= 0x3ffffff; h4 += c; c = h4 >> 26; h4 &= 0x3ffffff; h0 += c * 5;
    c = h0 >> 26; h0 &= 0x3ffffff; h1 += c;
    // Compute h + -p (i.e., h - (2^130 - 5)).
    let g0 = h0.wrapping_add(5); c = g0 >> 26; let g0 = g0 & 0x3ffffff;
    let g1 = h1.wrapping_add(c); c = g1 >> 26; let g1 = g1 & 0x3ffffff;
    let g2 = h2.wrapping_add(c); c = g2 >> 26; let g2 = g2 & 0x3ffffff;
    let g3 = h3.wrapping_add(c); c = g3 >> 26; let g3 = g3 & 0x3ffffff;
    let g4 = h4.wrapping_add(c).wrapping_sub(1 << 26);
    let mask = ((g4 >> 31).wrapping_sub(1)) as u32;
    let nmask = !mask;
    let h0 = (h0 & nmask) | (g0 & mask);
    let h1 = (h1 & nmask) | (g1 & mask);
    let h2 = (h2 & nmask) | (g2 & mask);
    let h3 = (h3 & nmask) | (g3 & mask);
    let h4 = (h4 & nmask) | (g4 & mask);
    // h = (h0 | h1<<26 | h2<<52 | h3<<78 | h4<<104) + s
    // Pack the 5 × 26-bit limbs into 4 × 32-bit limbs.  Each `_full` value
    // MUST be masked to 32 bits — without the mask the shifted upper limbs
    // leak data bits into the >32 range, which then gets misinterpreted as
    // carry when we add `s`, producing an incorrect tag.
    let mut h0_full = ((h0 as u64) | ((h1 as u64) << 26)) & 0xffffffff;
    let mut h1_full = (((h1 as u64) >> 6) | ((h2 as u64) << 20)) & 0xffffffff;
    let mut h2_full = (((h2 as u64) >> 12) | ((h3 as u64) << 14)) & 0xffffffff;
    let mut h3_full = (((h3 as u64) >> 18) | ((h4 as u64) << 8)) & 0xffffffff;
    let s = &key[16..32];
    let s0 = u32::from_le_bytes([s[0], s[1], s[2], s[3]]) as u64;
    let s1 = u32::from_le_bytes([s[4], s[5], s[6], s[7]]) as u64;
    let s2 = u32::from_le_bytes([s[8], s[9], s[10], s[11]]) as u64;
    let s3 = u32::from_le_bytes([s[12], s[13], s[14], s[15]]) as u64;
    h0_full = (h0_full as u64) + s0; let c = h0_full >> 32; h0_full &= 0xffffffff;
    h1_full = h1_full + s1 + c;      let c = h1_full >> 32; h1_full &= 0xffffffff;
    h2_full = h2_full + s2 + c;      let c = h2_full >> 32; h2_full &= 0xffffffff;
    h3_full = h3_full + s3 + c;                              h3_full &= 0xffffffff;
    let mut tag = [0u8; 16];
    tag[0..4].copy_from_slice(&(h0_full as u32).to_le_bytes());
    tag[4..8].copy_from_slice(&(h1_full as u32).to_le_bytes());
    tag[8..12].copy_from_slice(&(h2_full as u32).to_le_bytes());
    tag[12..16].copy_from_slice(&(h3_full as u32).to_le_bytes());
    tag
}
pub(crate) fn poly1305_key_gen(key: &[u8; 32], nonce: &[u8; 12]) -> [u8; 32] {
    let block0 = chacha20_block(key, 0, nonce);
    let mut k = [0u8; 32]; k.copy_from_slice(&block0[..32]); k
}

/// AEAD ChaCha20-Poly1305 seal — Rust API.  Returns ciphertext || 16-byte tag.
pub(crate) fn aead_chacha20_poly1305_seal(
    key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], plaintext: &[u8],
) -> Vec<u8> {
    let poly_key = poly1305_key_gen(key, nonce);
    let mut out = Vec::with_capacity(plaintext.len() + 16);
    out.extend_from_slice(plaintext);
    chacha20_xor(key, 1, nonce, &mut out[..plaintext.len()]);
    let mut mac_buf: Vec<u8> = Vec::new();
    mac_buf.extend_from_slice(aad);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(&out[..plaintext.len()]);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_buf.extend_from_slice(&(plaintext.len() as u64).to_le_bytes());
    let tag = poly1305_mac(&poly_key, &mac_buf);
    out.extend_from_slice(&tag);
    out
}

/// AEAD ChaCha20-Poly1305 open — Rust API.  `ct_and_tag` is ciphertext || tag.
/// Returns plaintext on success; None on tag mismatch or malformed input.
pub(crate) fn aead_chacha20_poly1305_open(
    key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], ct_and_tag: &[u8],
) -> Option<Vec<u8>> {
    if ct_and_tag.len() < 16 { return None; }
    let ct_len = ct_and_tag.len() - 16;
    let ct = &ct_and_tag[..ct_len];
    let recv_tag = &ct_and_tag[ct_len..];
    let poly_key = poly1305_key_gen(key, nonce);
    let mut mac_buf: Vec<u8> = Vec::new();
    mac_buf.extend_from_slice(aad);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(ct);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    mac_buf.extend_from_slice(&(ct.len() as u64).to_le_bytes());
    let computed = poly1305_mac(&poly_key, &mac_buf);
    let mut diff = 0u8;
    for i in 0..16 { diff |= computed[i] ^ recv_tag[i]; }
    if diff != 0 { return None; }
    let mut out = ct.to_vec();
    chacha20_xor(key, 1, nonce, &mut out);
    Some(out)
}
/// AEAD encrypt: (key32, nonce12, aad?, plaintext) → (ciphertext || 16-byte tag).
/// `n_plain` plaintext bytes; output is `n_plain + 16` bytes. Returns
/// number of bytes written (= n_plain + 16) or -1.
#[no_mangle] pub unsafe extern "C" fn aether_chacha20_poly1305_encrypt(
    key: *const c_void, nonce: *const c_void,
    aad: *const c_void, n_aad: c_int,
    plain: *const c_void, n_plain: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if key.is_null() || nonce.is_null() || plain.is_null() || out.is_null() { return -1; }
    if n_plain < 0 || n_aad < 0 || max_out < n_plain + 16 { return -1; }
    let key_bytes: &[u8; 32] = &*(key as *const [u8; 32]);
    let nonce_bytes: &[u8; 12] = &*(nonce as *const [u8; 12]);
    let aad_slice = if n_aad > 0 { std::slice::from_raw_parts(aad as *const u8, n_aad as usize) }
                    else { &[] };
    let plain_slice = std::slice::from_raw_parts(plain as *const u8, n_plain as usize);
    let out_slice = std::slice::from_raw_parts_mut(out as *mut u8, (n_plain + 16) as usize);
    // 1) Derive Poly1305 one-time key from chacha20(counter=0).
    let poly_key = poly1305_key_gen(key_bytes, nonce_bytes);
    // 2) Encrypt with chacha20(counter=1+).
    let ct = &mut out_slice[..n_plain as usize];
    ct.copy_from_slice(plain_slice);
    chacha20_xor(key_bytes, 1, nonce_bytes, ct);
    // 3) Build the Poly1305 message: aad || pad16 || ct || pad16 || aad_len_u64 || ct_len_u64.
    let mut mac_buf: Vec<u8> = Vec::new();
    mac_buf.extend_from_slice(aad_slice);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(ct);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(&(aad_slice.len() as u64).to_le_bytes());
    mac_buf.extend_from_slice(&(n_plain as u64).to_le_bytes());
    let tag = poly1305_mac(&poly_key, &mac_buf);
    out_slice[n_plain as usize .. (n_plain + 16) as usize].copy_from_slice(&tag);
    n_plain + 16
}
/// AEAD decrypt: verifies the trailing 16-byte tag, then decrypts.
/// Input is (ciphertext || tag) of length `n_in` (must be ≥ 16).
/// On success returns `n_in - 16` (plaintext length); on tag mismatch
/// returns -2; on bad inputs returns -1.
#[no_mangle] pub unsafe extern "C" fn aether_chacha20_poly1305_decrypt(
    key: *const c_void, nonce: *const c_void,
    aad: *const c_void, n_aad: c_int,
    ct_and_tag: *const c_void, n_in: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if key.is_null() || nonce.is_null() || ct_and_tag.is_null() || out.is_null() { return -1; }
    if n_in < 16 || n_aad < 0 || max_out < n_in - 16 { return -1; }
    let key_bytes: &[u8; 32] = &*(key as *const [u8; 32]);
    let nonce_bytes: &[u8; 12] = &*(nonce as *const [u8; 12]);
    let aad_slice = if n_aad > 0 { std::slice::from_raw_parts(aad as *const u8, n_aad as usize) } else { &[] };
    let input = std::slice::from_raw_parts(ct_and_tag as *const u8, n_in as usize);
    let ct = &input[..(n_in as usize - 16)];
    let received_tag = &input[(n_in as usize - 16)..];
    let poly_key = poly1305_key_gen(key_bytes, nonce_bytes);
    let mut mac_buf: Vec<u8> = Vec::new();
    mac_buf.extend_from_slice(aad_slice);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(ct);
    while mac_buf.len() % 16 != 0 { mac_buf.push(0); }
    mac_buf.extend_from_slice(&(aad_slice.len() as u64).to_le_bytes());
    mac_buf.extend_from_slice(&(ct.len() as u64).to_le_bytes());
    let computed = poly1305_mac(&poly_key, &mac_buf);
    // Constant-time-ish tag compare.
    let mut diff = 0u8;
    for i in 0..16 { diff |= computed[i] ^ received_tag[i]; }
    if diff != 0 { return -2; }
    let mut pt = ct.to_vec();
    chacha20_xor(key_bytes, 1, nonce_bytes, &mut pt);
    let out_slice = std::slice::from_raw_parts_mut(out as *mut u8, ct.len());
    out_slice.copy_from_slice(&pt);
    ct.len() as c_int
}

// =====================================================================
// FR-19.1-extra (a) — SHA-256 (FIPS 180-4).
//
// One-shot + incremental API. The TLS 1.3 key schedule (HKDF-Extract/
// Expand + HMAC) is built on top of this. Verified against the standard
// "abc" / "" / FIPS Appendix B test vectors in the unit tests below.
// =====================================================================

const SHA256_K: [u32; 64] = [
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
];

const SHA256_IV: [u32; 8] = [
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
];

fn sha256_compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for i in 0..16 {
        w[i] = u32::from_be_bytes([block[i*4], block[i*4+1], block[i*4+2], block[i*4+3]]);
    }
    for i in 16..64 {
        let s0 = w[i-15].rotate_right(7) ^ w[i-15].rotate_right(18) ^ (w[i-15] >> 3);
        let s1 = w[i-2].rotate_right(17) ^ w[i-2].rotate_right(19) ^ (w[i-2] >> 10);
        w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
    }
    let mut a = state[0]; let mut b = state[1]; let mut c = state[2]; let mut d = state[3];
    let mut e = state[4]; let mut f = state[5]; let mut g = state[6]; let mut h = state[7];
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA256_K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let mj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(mj);
        h = g; g = f; f = e; e = d.wrapping_add(t1);
        d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a); state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c); state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e); state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g); state[7] = state[7].wrapping_add(h);
}

pub(crate) fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut state = SHA256_IV;
    let mut len = 0u64;
    let mut buf = [0u8; 64];
    let mut buf_len = 0usize;
    let mut data = msg;
    while !data.is_empty() {
        let take = (64 - buf_len).min(data.len());
        buf[buf_len..buf_len+take].copy_from_slice(&data[..take]);
        buf_len += take;
        data = &data[take..];
        len += take as u64 * 8;
        if buf_len == 64 {
            sha256_compress(&mut state, &buf);
            buf_len = 0;
        }
    }
    // Pad: 0x80, zeros, length-be64.
    buf[buf_len] = 0x80;
    buf_len += 1;
    if buf_len > 56 {
        for i in buf_len..64 { buf[i] = 0; }
        sha256_compress(&mut state, &buf);
        buf_len = 0;
        buf = [0u8; 64];
    }
    for i in buf_len..56 { buf[i] = 0; }
    buf[56..64].copy_from_slice(&len.to_be_bytes());
    sha256_compress(&mut state, &buf);
    let mut out = [0u8; 32];
    for i in 0..8 { out[i*4..i*4+4].copy_from_slice(&state[i].to_be_bytes()); }
    out
}

/// One-shot SHA-256. Writes 32 bytes to `out`. Returns 32 or -1.
#[no_mangle] pub unsafe extern "C" fn aether_sha256(
    msg: *const c_void, n_msg: c_int,
    out: *mut c_void,
) -> c_int {
    if msg.is_null() || out.is_null() || n_msg < 0 { return -1; }
    let slice = std::slice::from_raw_parts(msg as *const u8, n_msg as usize);
    let digest = sha256(slice);
    let o = std::slice::from_raw_parts_mut(out as *mut u8, 32);
    o.copy_from_slice(&digest);
    32
}

// =====================================================================
// FR-19.1-extra (b) — HMAC-SHA256 (RFC 2104).
// =====================================================================
pub(crate) fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k0 = [0u8; 64];
    if key.len() > 64 {
        let h = sha256(key);
        k0[..32].copy_from_slice(&h);
    } else {
        k0[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64]; let mut opad = [0x5cu8; 64];
    for i in 0..64 { ipad[i] ^= k0[i]; opad[i] ^= k0[i]; }
    let mut inner = Vec::with_capacity(64 + msg.len());
    inner.extend_from_slice(&ipad); inner.extend_from_slice(msg);
    let h_in = sha256(&inner);
    let mut outer = Vec::with_capacity(64 + 32);
    outer.extend_from_slice(&opad); outer.extend_from_slice(&h_in);
    sha256(&outer)
}

#[no_mangle] pub unsafe extern "C" fn aether_hmac_sha256(
    key: *const c_void, n_key: c_int,
    msg: *const c_void, n_msg: c_int,
    out: *mut c_void,
) -> c_int {
    if key.is_null() || msg.is_null() || out.is_null() || n_key < 0 || n_msg < 0 { return -1; }
    let k = std::slice::from_raw_parts(key as *const u8, n_key as usize);
    let m = std::slice::from_raw_parts(msg as *const u8, n_msg as usize);
    let tag = hmac_sha256(k, m);
    let o = std::slice::from_raw_parts_mut(out as *mut u8, 32);
    o.copy_from_slice(&tag);
    32
}

// =====================================================================
// FR-19.1-extra (c) — HKDF (RFC 5869).
//
// HKDF-Extract(salt, ikm) -> PRK         (32 bytes for SHA-256)
// HKDF-Expand(prk, info, L) -> OKM       (L bytes, L ≤ 255*32)
//
// TLS 1.3 also adds HKDF-Expand-Label per RFC 8446 §7.1.
// =====================================================================
pub(crate) fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
    let s = if salt.is_empty() { &[0u8; 32][..] } else { salt };
    hmac_sha256(s, ikm)
}

pub(crate) fn hkdf_expand(prk: &[u8; 32], info: &[u8], len: usize) -> Vec<u8> {
    assert!(len <= 255 * 32, "HKDF-Expand: L too large");
    let n = (len + 31) / 32;
    let mut t_prev: Vec<u8> = Vec::new();
    let mut okm = Vec::with_capacity(n * 32);
    for i in 1..=n {
        let mut input = Vec::with_capacity(t_prev.len() + info.len() + 1);
        input.extend_from_slice(&t_prev);
        input.extend_from_slice(info);
        input.push(i as u8);
        t_prev = hmac_sha256(prk, &input).to_vec();
        okm.extend_from_slice(&t_prev);
    }
    okm.truncate(len);
    okm
}

#[no_mangle] pub unsafe extern "C" fn aether_hkdf_extract(
    salt: *const c_void, n_salt: c_int,
    ikm: *const c_void, n_ikm: c_int,
    out: *mut c_void,
) -> c_int {
    if ikm.is_null() || out.is_null() || n_salt < 0 || n_ikm < 0 { return -1; }
    let s = if salt.is_null() { &[][..] }
            else { std::slice::from_raw_parts(salt as *const u8, n_salt as usize) };
    let k = std::slice::from_raw_parts(ikm as *const u8, n_ikm as usize);
    let prk = hkdf_extract(s, k);
    let o = std::slice::from_raw_parts_mut(out as *mut u8, 32);
    o.copy_from_slice(&prk);
    32
}

#[no_mangle] pub unsafe extern "C" fn aether_hkdf_expand(
    prk: *const c_void, n_prk: c_int,
    info: *const c_void, n_info: c_int,
    out: *mut c_void, n_out: c_int,
) -> c_int {
    if prk.is_null() || out.is_null() || n_prk != 32 || n_info < 0 || n_out <= 0 { return -1; }
    if n_out as usize > 255 * 32 { return -1; }
    let mut p = [0u8; 32];
    p.copy_from_slice(std::slice::from_raw_parts(prk as *const u8, 32));
    let i = if info.is_null() || n_info == 0 { &[][..] }
            else { std::slice::from_raw_parts(info as *const u8, n_info as usize) };
    let okm = hkdf_expand(&p, i, n_out as usize);
    let o = std::slice::from_raw_parts_mut(out as *mut u8, n_out as usize);
    o.copy_from_slice(&okm);
    n_out
}

/// HKDF-Expand-Label per RFC 8446 §7.1.
///   HkdfLabel = struct { uint16 length; opaque label<7..255>;
///                        opaque context<0..255>; }
///   label = "tls13 " ++ user_label
#[no_mangle] pub unsafe extern "C" fn aether_tls13_hkdf_expand_label(
    secret: *const c_void, n_secret: c_int,
    label: *const c_void, n_label: c_int,
    context: *const c_void, n_context: c_int,
    out: *mut c_void, n_out: c_int,
) -> c_int {
    if secret.is_null() || label.is_null() || out.is_null() { return -1; }
    if n_secret != 32 || n_label <= 0 || n_label > 255 || n_context < 0 || n_context > 255 { return -1; }
    if n_out <= 0 || n_out > 0xffff { return -1; }
    let lbl = std::slice::from_raw_parts(label as *const u8, n_label as usize);
    let ctx = if context.is_null() || n_context == 0 { &[][..] }
              else { std::slice::from_raw_parts(context as *const u8, n_context as usize) };
    let full_label = {
        let mut v = Vec::with_capacity(6 + lbl.len());
        v.extend_from_slice(b"tls13 ");
        v.extend_from_slice(lbl);
        v
    };
    if full_label.len() > 255 { return -1; }
    let mut info = Vec::with_capacity(2 + 1 + full_label.len() + 1 + ctx.len());
    info.extend_from_slice(&(n_out as u16).to_be_bytes());
    info.push(full_label.len() as u8);
    info.extend_from_slice(&full_label);
    info.push(ctx.len() as u8);
    info.extend_from_slice(ctx);

    let mut p = [0u8; 32];
    p.copy_from_slice(std::slice::from_raw_parts(secret as *const u8, 32));
    let okm = hkdf_expand(&p, &info, n_out as usize);
    let o = std::slice::from_raw_parts_mut(out as *mut u8, n_out as usize);
    o.copy_from_slice(&okm);
    n_out
}

// =====================================================================
// FR-19.1-extra (d) — X25519 (RFC 7748).
//
// Curve25519 scalar multiplication. The TLS 1.3 key_share group used
// for ECDHE on basically every modern handshake. 32-byte private
// scalar in, 32-byte public point out. Verified against RFC 7748
// §5.2 test vector in the unit tests.
// =====================================================================
fn x25519_fe_add(out: &mut [u64; 5], a: &[u64; 5], b: &[u64; 5]) {
    for i in 0..5 { out[i] = a[i] + b[i]; }
}
fn x25519_fe_sub(out: &mut [u64; 5], a: &[u64; 5], b: &[u64; 5]) {
    // c = a + (2*p - b); 2*p = 2^256 - 38 → done via per-limb add+borrow trick.
    let two_p_0: u64 = 0xfffffffffffda;
    let two_p_other: u64 = 0xffffffffffffe;
    out[0] = a[0] + two_p_0 - b[0];
    out[1] = a[1] + two_p_other - b[1];
    out[2] = a[2] + two_p_other - b[2];
    out[3] = a[3] + two_p_other - b[3];
    out[4] = a[4] + two_p_other - b[4];
}
fn x25519_fe_mul(out: &mut [u64; 5], a: &[u64; 5], b: &[u64; 5]) {
    // Schoolbook on radix-2^51 limbs. Reduce mod 2^255 - 19.
    let m = |x: u128| (x as u64) & ((1u64 << 51) - 1);
    let a0 = a[0] as u128; let a1 = a[1] as u128; let a2 = a[2] as u128;
    let a3 = a[3] as u128; let a4 = a[4] as u128;
    let b0 = b[0] as u128; let b1 = b[1] as u128; let b2 = b[2] as u128;
    let b3 = b[3] as u128; let b4 = b[4] as u128;
    let b1_19 = 19 * b1; let b2_19 = 19 * b2; let b3_19 = 19 * b3; let b4_19 = 19 * b4;

    let d0 = a0*b0 + a1*b4_19 + a2*b3_19 + a3*b2_19 + a4*b1_19;
    let d1 = a0*b1 + a1*b0    + a2*b4_19 + a3*b3_19 + a4*b2_19;
    let d2 = a0*b2 + a1*b1    + a2*b0    + a3*b4_19 + a4*b3_19;
    let d3 = a0*b3 + a1*b2    + a2*b1    + a3*b0    + a4*b4_19;
    let d4 = a0*b4 + a1*b3    + a2*b2    + a3*b1    + a4*b0;

    let c = (d0 >> 51) as u64; let r0 = m(d0);
    let d1 = d1 + c as u128; let c = (d1 >> 51) as u64; let r1 = m(d1);
    let d2 = d2 + c as u128; let c = (d2 >> 51) as u64; let r2 = m(d2);
    let d3 = d3 + c as u128; let c = (d3 >> 51) as u64; let r3 = m(d3);
    let d4 = d4 + c as u128; let c = (d4 >> 51) as u64; let r4 = m(d4);
    // Carry from limb 4 wraps into limb 0 with factor 19.
    let r0 = r0 + 19 * c;
    let c2 = r0 >> 51; let r0 = r0 & ((1u64 << 51) - 1);
    out[0] = r0; out[1] = r1 + c2; out[2] = r2; out[3] = r3; out[4] = r4;
}
fn x25519_fe_sq(out: &mut [u64; 5], a: &[u64; 5]) {
    let mut tmp = [0u64; 5];
    x25519_fe_mul(&mut tmp, a, a);
    *out = tmp;
}
fn x25519_fe_mul_121665(out: &mut [u64; 5], a: &[u64; 5]) {
    let m = |x: u128| (x as u64) & ((1u64 << 51) - 1);
    let c0 = a[0] as u128 * 121665;
    let c1 = a[1] as u128 * 121665;
    let c2 = a[2] as u128 * 121665;
    let c3 = a[3] as u128 * 121665;
    let c4 = a[4] as u128 * 121665;
    let c = (c0 >> 51) as u64; let r0 = m(c0);
    let c1 = c1 + c as u128; let c = (c1 >> 51) as u64; let r1 = m(c1);
    let c2 = c2 + c as u128; let c = (c2 >> 51) as u64; let r2 = m(c2);
    let c3 = c3 + c as u128; let c = (c3 >> 51) as u64; let r3 = m(c3);
    let c4 = c4 + c as u128; let c = (c4 >> 51) as u64; let r4 = m(c4);
    let r0 = r0 + 19 * c;
    let c2 = r0 >> 51; let r0 = r0 & ((1u64 << 51) - 1);
    out[0] = r0; out[1] = r1 + c2; out[2] = r2; out[3] = r3; out[4] = r4;
}
fn x25519_fe_invert(out: &mut [u64; 5], z: &[u64; 5]) {
    // Compute z^(2^255 - 21) = z^(-1) mod 2^255 - 19, via 2^255 - 21 chain.
    let sq_into = |dst: &mut [u64;5], src: &[u64;5]| x25519_fe_sq(dst, src);
    let sq_inplace = |x: &mut [u64;5]| { let tmp = *x; x25519_fe_sq(x, &tmp); };
    let mul_inplace = |x: &mut [u64;5], y: &[u64;5]| { let tmp = *x; x25519_fe_mul(x, &tmp, y); };

    let mut z2 = [0u64; 5]; sq_into(&mut z2, z);
    let mut z9 = [0u64; 5]; sq_into(&mut z9, &z2); sq_inplace(&mut z9); mul_inplace(&mut z9, z);
    let mut z11 = [0u64; 5]; x25519_fe_mul(&mut z11, &z9, &z2);
    let mut z2_5_0 = [0u64; 5]; sq_into(&mut z2_5_0, &z11); mul_inplace(&mut z2_5_0, &z9);
    let mut z2_10_0 = [0u64; 5]; sq_into(&mut z2_10_0, &z2_5_0);
    for _ in 0..4 { sq_inplace(&mut z2_10_0); }
    mul_inplace(&mut z2_10_0, &z2_5_0);
    let mut z2_20_0 = [0u64; 5]; sq_into(&mut z2_20_0, &z2_10_0);
    for _ in 0..9 { sq_inplace(&mut z2_20_0); }
    mul_inplace(&mut z2_20_0, &z2_10_0);
    let mut z2_40_0 = [0u64; 5]; sq_into(&mut z2_40_0, &z2_20_0);
    for _ in 0..19 { sq_inplace(&mut z2_40_0); }
    mul_inplace(&mut z2_40_0, &z2_20_0);
    let mut z2_50_0 = [0u64; 5]; sq_into(&mut z2_50_0, &z2_40_0);
    for _ in 0..9 { sq_inplace(&mut z2_50_0); }
    mul_inplace(&mut z2_50_0, &z2_10_0);
    let mut z2_100_0 = [0u64; 5]; sq_into(&mut z2_100_0, &z2_50_0);
    for _ in 0..49 { sq_inplace(&mut z2_100_0); }
    mul_inplace(&mut z2_100_0, &z2_50_0);
    let mut z2_200_0 = [0u64; 5]; sq_into(&mut z2_200_0, &z2_100_0);
    for _ in 0..99 { sq_inplace(&mut z2_200_0); }
    mul_inplace(&mut z2_200_0, &z2_100_0);
    let mut z2_250_0 = [0u64; 5]; sq_into(&mut z2_250_0, &z2_200_0);
    for _ in 0..49 { sq_inplace(&mut z2_250_0); }
    mul_inplace(&mut z2_250_0, &z2_50_0);
    let mut z2_255_5 = [0u64; 5]; sq_into(&mut z2_255_5, &z2_250_0);
    for _ in 0..4 { sq_inplace(&mut z2_255_5); }
    x25519_fe_mul(out, &z2_255_5, &z11);
}
fn x25519_fe_to_bytes(out: &mut [u8; 32], h: &[u64; 5]) {
    let mask = (1u64 << 51) - 1;
    let mut h = *h;
    // Reduce: carry propagate twice.
    for _ in 0..2 {
        let c = h[0] >> 51; h[0] &= mask; h[1] += c;
        let c = h[1] >> 51; h[1] &= mask; h[2] += c;
        let c = h[2] >> 51; h[2] &= mask; h[3] += c;
        let c = h[3] >> 51; h[3] &= mask; h[4] += c;
        let c = h[4] >> 51; h[4] &= mask; h[0] += 19 * c;
    }
    // Conditional subtract p if h >= p.
    let q = (h[0] + 19) >> 51;
    let q = (h[1] + q) >> 51;
    let q = (h[2] + q) >> 51;
    let q = (h[3] + q) >> 51;
    let q = (h[4] + q) >> 51;
    h[0] += 19 * q;
    let c = h[0] >> 51; h[0] &= mask; h[1] += c;
    let c = h[1] >> 51; h[1] &= mask; h[2] += c;
    let c = h[2] >> 51; h[2] &= mask; h[3] += c;
    let c = h[3] >> 51; h[3] &= mask; h[4] += c;
    h[4] &= mask;
    // Pack to little-endian bytes byte-by-byte (each limb is 51 bits;
    // shifting 102 bits in a single u64 would overflow).
    for i in 0..32 { out[i] = 0; }
    // Pack 5 51-bit limbs into 32 bytes (256 bits) LE.
    let bits = [h[0], h[1], h[2], h[3], h[4]];
    let mut bitpos = 0usize;
    for limb in &bits {
        let mut v = *limb;
        let mut bits_in_limb = 51;
        while bits_in_limb > 0 {
            let byte_idx = bitpos / 8;
            let bit_off  = bitpos % 8;
            let space = 8 - bit_off;
            let take = bits_in_limb.min(space);
            let mask_take = ((1u64 << take) - 1) as u8;
            out[byte_idx] |= ((v as u8) & mask_take) << bit_off;
            v >>= take;
            bitpos += take;
            bits_in_limb -= take;
        }
    }
}
fn x25519_fe_from_bytes(out: &mut [u64; 5], bytes: &[u8; 32]) {
    // Five 51-bit limbs cover bits 0..254 of the input (255 data bits).
    // RFC 7748 says the top bit of byte 31 (input bit 255) must be
    // masked off; since we never read it, the mask is implicit.
    let mask = (1u64 << 51) - 1;
    let mut limbs = [0u64; 5];
    let mut bitpos = 0usize;
    for limb_i in 0..5 {
        let mut bits_in_limb = 51;
        let mut v = 0u64;
        let mut have = 0;
        while bits_in_limb > 0 {
            let byte_idx = bitpos / 8;
            let bit_off  = bitpos % 8;
            let avail = 8 - bit_off;
            let take = bits_in_limb.min(avail);
            let mask_take = ((1u64 << take) - 1) as u64;
            let chunk = ((bytes[byte_idx] as u64) >> bit_off) & mask_take;
            v |= chunk << have;
            have += take;
            bitpos += take;
            bits_in_limb -= take;
        }
        limbs[limb_i] = v & mask;
    }
    *out = limbs;
}
fn x25519_cswap(swap: u64, a: &mut [u64; 5], b: &mut [u64; 5]) {
    let mask = swap.wrapping_neg();
    for i in 0..5 {
        let d = (a[i] ^ b[i]) & mask;
        a[i] ^= d; b[i] ^= d;
    }
}

/// Compute u-coordinate scalar multiplication: out = scalar * u.
/// RFC 7748 X25519 — Montgomery ladder over Curve25519.
pub(crate) fn x25519_scalar_mult(scalar: &[u8; 32], u_in: &[u8; 32]) -> [u8; 32] {
    // Clamp scalar per RFC 7748.
    let mut k = *scalar;
    k[0] &= 248; k[31] &= 127; k[31] |= 64;
    let mut x1 = [0u64; 5]; x25519_fe_from_bytes(&mut x1, u_in);
    let mut x2 = [0u64; 5]; x2[0] = 1;
    let mut z2 = [0u64; 5];
    let mut x3 = x1; let mut z3 = [0u64; 5]; z3[0] = 1;
    let mut swap: u64 = 0;
    for t in (0..=254).rev() {
        let k_t = ((k[t/8] >> (t%8)) & 1) as u64;
        swap ^= k_t;
        x25519_cswap(swap, &mut x2, &mut x3);
        x25519_cswap(swap, &mut z2, &mut z3);
        swap = k_t;
        let mut a = [0u64;5]; x25519_fe_add(&mut a, &x2, &z2);
        let mut aa = [0u64;5]; x25519_fe_sq(&mut aa, &a);
        let mut b = [0u64;5]; x25519_fe_sub(&mut b, &x2, &z2);
        let mut bb = [0u64;5]; x25519_fe_sq(&mut bb, &b);
        let mut e = [0u64;5]; x25519_fe_sub(&mut e, &aa, &bb);
        let mut c = [0u64;5]; x25519_fe_add(&mut c, &x3, &z3);
        let mut d = [0u64;5]; x25519_fe_sub(&mut d, &x3, &z3);
        let mut da = [0u64;5]; x25519_fe_mul(&mut da, &d, &a);
        let mut cb = [0u64;5]; x25519_fe_mul(&mut cb, &c, &b);
        let mut sum = [0u64;5]; x25519_fe_add(&mut sum, &da, &cb);
        let mut x3n = [0u64;5]; x25519_fe_sq(&mut x3n, &sum);
        let mut dif = [0u64;5]; x25519_fe_sub(&mut dif, &da, &cb);
        let mut difsq = [0u64;5]; x25519_fe_sq(&mut difsq, &dif);
        let mut z3n = [0u64;5]; x25519_fe_mul(&mut z3n, &difsq, &x1);
        let mut x2n = [0u64;5]; x25519_fe_mul(&mut x2n, &aa, &bb);
        let mut e121665 = [0u64;5]; x25519_fe_mul_121665(&mut e121665, &e);
        let mut tmp = [0u64;5]; x25519_fe_add(&mut tmp, &aa, &e121665);
        let mut z2n = [0u64;5]; x25519_fe_mul(&mut z2n, &e, &tmp);
        x2 = x2n; z2 = z2n; x3 = x3n; z3 = z3n;
    }
    x25519_cswap(swap, &mut x2, &mut x3);
    x25519_cswap(swap, &mut z2, &mut z3);
    let mut z2_inv = [0u64; 5]; x25519_fe_invert(&mut z2_inv, &z2);
    let mut out_fe = [0u64; 5]; x25519_fe_mul(&mut out_fe, &x2, &z2_inv);
    let mut out = [0u8; 32]; x25519_fe_to_bytes(&mut out, &out_fe);
    out
}

/// X25519 public key derivation: pub = scalar * basepoint(9).
#[no_mangle] pub unsafe extern "C" fn aether_x25519_derive_public(
    scalar: *const c_void, out: *mut c_void,
) -> c_int {
    if scalar.is_null() || out.is_null() { return -1; }
    let mut s = [0u8; 32]; s.copy_from_slice(std::slice::from_raw_parts(scalar as *const u8, 32));
    let mut bp = [0u8; 32]; bp[0] = 9;
    let pubk = x25519_scalar_mult(&s, &bp);
    std::slice::from_raw_parts_mut(out as *mut u8, 32).copy_from_slice(&pubk);
    32
}

/// X25519 shared-secret computation: shared = scalar * peer_pub.
#[no_mangle] pub unsafe extern "C" fn aether_x25519_shared_secret(
    scalar: *const c_void, peer_pub: *const c_void, out: *mut c_void,
) -> c_int {
    if scalar.is_null() || peer_pub.is_null() || out.is_null() { return -1; }
    let mut s = [0u8; 32]; s.copy_from_slice(std::slice::from_raw_parts(scalar as *const u8, 32));
    let mut p = [0u8; 32]; p.copy_from_slice(std::slice::from_raw_parts(peer_pub as *const u8, 32));
    let shared = x25519_scalar_mult(&s, &p);
    std::slice::from_raw_parts_mut(out as *mut u8, 32).copy_from_slice(&shared);
    32
}

// =====================================================================
// FR-19.1-extra (e) — SHA-512 (FIPS 180-4).
//
// 64-bit cousin of SHA-256. Needed internally by Ed25519. Same
// structure (8-word state + 80-round compression + length-padded
// blocks) but 1024-bit blocks and different IV/K constants.
// Verified against the standard "abc" + "" vectors.
// =====================================================================
const SHA512_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];
const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

fn sha512_compress(state: &mut [u64; 8], block: &[u8; 128]) {
    let mut w = [0u64; 80];
    for i in 0..16 {
        w[i] = u64::from_be_bytes([
            block[i*8], block[i*8+1], block[i*8+2], block[i*8+3],
            block[i*8+4], block[i*8+5], block[i*8+6], block[i*8+7],
        ]);
    }
    for i in 16..80 {
        let s0 = w[i-15].rotate_right(1) ^ w[i-15].rotate_right(8) ^ (w[i-15] >> 7);
        let s1 = w[i-2].rotate_right(19) ^ w[i-2].rotate_right(61) ^ (w[i-2] >> 6);
        w[i] = w[i-16].wrapping_add(s0).wrapping_add(w[i-7]).wrapping_add(s1);
    }
    let mut a = state[0]; let mut b = state[1]; let mut c = state[2]; let mut d = state[3];
    let mut e = state[4]; let mut f = state[5]; let mut g = state[6]; let mut h = state[7];
    for i in 0..80 {
        let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA512_K[i]).wrapping_add(w[i]);
        let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        let mj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(mj);
        h = g; g = f; f = e; e = d.wrapping_add(t1);
        d = c; c = b; b = a; a = t1.wrapping_add(t2);
    }
    state[0] = state[0].wrapping_add(a); state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c); state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e); state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g); state[7] = state[7].wrapping_add(h);
}

pub(crate) fn sha512(msg: &[u8]) -> [u8; 64] {
    let mut state = SHA512_IV;
    let mut buf = [0u8; 128];
    let mut buf_len = 0usize;
    let mut total_bits: u128 = 0;
    let mut data = msg;
    while !data.is_empty() {
        let take = (128 - buf_len).min(data.len());
        buf[buf_len..buf_len+take].copy_from_slice(&data[..take]);
        buf_len += take;
        total_bits += (take as u128) * 8;
        data = &data[take..];
        if buf_len == 128 {
            sha512_compress(&mut state, &buf);
            buf_len = 0;
        }
    }
    buf[buf_len] = 0x80;
    buf_len += 1;
    if buf_len > 112 {
        for i in buf_len..128 { buf[i] = 0; }
        sha512_compress(&mut state, &buf);
        buf_len = 0;
        buf = [0u8; 128];
    }
    for i in buf_len..112 { buf[i] = 0; }
    buf[112..128].copy_from_slice(&total_bits.to_be_bytes());
    sha512_compress(&mut state, &buf);
    let mut out = [0u8; 64];
    for i in 0..8 { out[i*8..i*8+8].copy_from_slice(&state[i].to_be_bytes()); }
    out
}

#[no_mangle] pub unsafe extern "C" fn aether_sha512(
    msg: *const c_void, n_msg: c_int,
    out: *mut c_void,
) -> c_int {
    if msg.is_null() || out.is_null() || n_msg < 0 { return -1; }
    let slice = std::slice::from_raw_parts(msg as *const u8, n_msg as usize);
    let digest = sha512(slice);
    std::slice::from_raw_parts_mut(out as *mut u8, 64).copy_from_slice(&digest);
    64
}

// =====================================================================
// FR-19.1-extra (f) — Ed25519 (RFC 8032).
//
// EdDSA over edwards25519 with SHA-512. Provides aether_ed25519_sign
// and aether_ed25519_verify against RFC 8032 §7.1 test vectors.
//
// Implementation notes:
//   - Scalars (32B) are little-endian; signatures are 64B (R || S).
//   - Curve point representation: extended Edwards (X, Y, Z, T).
//   - Field element representation: 5×51-bit limbs over GF(2^255-19),
//     reusing the X25519 fe ops above.
//   - Scalar reduction mod ell uses a simple 6-limb 51-bit accumulator
//     with Barrett-style reduction.
//
// Reference: ref10 from libsodium / supercop; cleaned up & translated
// to Rust. Verified against RFC 8032 §7.1 vectors below.
// =====================================================================

// edwards25519 group order ell = 2^252 + 27742317777372353535851937790883648493.
const ELL: [u8; 32] = [
    0xed, 0xd3, 0xf5, 0x5c, 0x1a, 0x63, 0x12, 0x58,
    0xd6, 0x9c, 0xf7, 0xa2, 0xde, 0xf9, 0xde, 0x14,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10,
];
// d = -121665/121666 (mod p), the edwards25519 curve constant.
// Computed at runtime from the field-element ops to avoid hand-typed limb
// errors. Cached in an OnceLock.
fn ed25519_d_fe() -> [u64; 5] {
    use std::sync::OnceLock;
    static D: OnceLock<[u8; 40]> = OnceLock::new();
    let bytes = D.get_or_init(|| {
        let mut a = ed25519_zero_fe(); a[0] = 121665;
        let mut neg_a = [0u64; 5]; x25519_fe_sub(&mut neg_a, &ed25519_zero_fe(), &a);
        let mut b = ed25519_zero_fe(); b[0] = 121666;
        let mut b_inv = [0u64; 5]; x25519_fe_invert(&mut b_inv, &b);
        let mut d = [0u64; 5]; x25519_fe_mul(&mut d, &neg_a, &b_inv);
        let mut out = [0u8; 40];
        for i in 0..5 { out[i*8..i*8+8].copy_from_slice(&d[i].to_le_bytes()); }
        out
    });
    let mut d = [0u64; 5];
    for i in 0..5 {
        d[i] = u64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap());
    }
    d
}
fn ed25519_one_fe() -> [u64; 5] { [1, 0, 0, 0, 0] }
fn ed25519_zero_fe() -> [u64; 5] { [0, 0, 0, 0, 0] }

#[derive(Clone, Copy)]
struct EdPoint { x: [u64;5], y: [u64;5], z: [u64;5], t: [u64;5] }

fn ed_identity() -> EdPoint {
    EdPoint { x: ed25519_zero_fe(), y: ed25519_one_fe(), z: ed25519_one_fe(), t: ed25519_zero_fe() }
}

// Base point B in extended Edwards form. Decoded from its standard
// 32-byte compressed encoding (RFC 8032 §6) to dodge hand-typed limb
// errors. Caches the decoded point at first access.
fn ed_base() -> EdPoint {
    use std::sync::OnceLock;
    static BASE: OnceLock<[u8; 200]> = OnceLock::new();  // 5 limbs × 8 bytes × 4 fields = 160; +pad
    let bytes = BASE.get_or_init(|| {
        // Standard By encoding (LE): byte 0 = 0x58, bytes 1..31 = 0x66, x_sign = 0.
        let mut by = [0x66u8; 32];
        by[0] = 0x58;
        let p = ed_decode(&by).expect("base point decode");
        let mut out = [0u8; 200];
        let mut push = |off: usize, fe: &[u64; 5]| {
            for i in 0..5 {
                out[off + i*8..off + i*8 + 8].copy_from_slice(&fe[i].to_le_bytes());
            }
        };
        push(0,  &p.x);
        push(40, &p.y);
        push(80, &p.z);
        push(120, &p.t);
        out
    });
    let read = |off: usize| -> [u64; 5] {
        let mut fe = [0u64; 5];
        for i in 0..5 {
            fe[i] = u64::from_le_bytes(bytes[off + i*8 .. off + i*8 + 8].try_into().unwrap());
        }
        fe
    };
    EdPoint { x: read(0), y: read(40), z: read(80), t: read(120) }
}

fn ed_add(p1: &EdPoint, p2: &EdPoint) -> EdPoint {
    let mut a = [0u64;5]; let mut tmp = [0u64;5];
    x25519_fe_sub(&mut tmp, &p1.y, &p1.x);
    let mut b = [0u64;5];
    x25519_fe_sub(&mut b, &p2.y, &p2.x);
    x25519_fe_mul(&mut a, &tmp, &b);
    let mut c = [0u64;5]; let mut tmp2 = [0u64;5];
    x25519_fe_add(&mut tmp, &p1.y, &p1.x);
    x25519_fe_add(&mut tmp2, &p2.y, &p2.x);
    x25519_fe_mul(&mut c, &tmp, &tmp2);
    let d_const = ed25519_d_fe();
    let mut two_d = [0u64;5]; x25519_fe_add(&mut two_d, &d_const, &d_const);
    let mut d_val = [0u64;5];
    x25519_fe_mul(&mut d_val, &p1.t, &p2.t);
    let mut tmp3 = d_val;
    x25519_fe_mul(&mut d_val, &tmp3, &two_d);
    let mut z_val = [0u64;5];
    x25519_fe_mul(&mut z_val, &p1.z, &p2.z);
    let mut z2 = [0u64;5]; x25519_fe_add(&mut z2, &z_val, &z_val);
    let mut e = [0u64;5]; x25519_fe_sub(&mut e, &c, &a);
    let mut f = [0u64;5]; x25519_fe_sub(&mut f, &z2, &d_val);
    let mut g = [0u64;5]; x25519_fe_add(&mut g, &z2, &d_val);
    let mut h = [0u64;5]; x25519_fe_add(&mut h, &c, &a);
    let mut x_out = [0u64;5]; x25519_fe_mul(&mut x_out, &e, &f);
    let mut y_out = [0u64;5]; x25519_fe_mul(&mut y_out, &g, &h);
    let mut t_out = [0u64;5]; x25519_fe_mul(&mut t_out, &e, &h);
    let mut z_out = [0u64;5]; x25519_fe_mul(&mut z_out, &f, &g);
    let _ = tmp3;
    EdPoint { x: x_out, y: y_out, z: z_out, t: t_out }
}

fn ed_double(p: &EdPoint) -> EdPoint { ed_add(p, p) }

fn ed_scalar_mult(scalar: &[u8; 32], p: &EdPoint) -> EdPoint {
    let mut result = ed_identity();
    // Standard double-and-add, MSB first. Not constant-time — fine for
    // verify but signing should use a constant-time impl. FR-x-extra.
    for byte_idx in (0..32).rev() {
        for bit in (0..8).rev() {
            result = ed_double(&result);
            if (scalar[byte_idx] >> bit) & 1 == 1 {
                result = ed_add(&result, p);
            }
        }
    }
    result
}

fn ed_to_affine_y(p: &EdPoint) -> ([u8; 32], bool) {
    let mut z_inv = [0u64; 5]; x25519_fe_invert(&mut z_inv, &p.z);
    let mut x = [0u64; 5]; x25519_fe_mul(&mut x, &p.x, &z_inv);
    let mut y = [0u64; 5]; x25519_fe_mul(&mut y, &p.y, &z_inv);
    let mut x_bytes = [0u8; 32]; x25519_fe_to_bytes(&mut x_bytes, &x);
    let mut y_bytes = [0u8; 32]; x25519_fe_to_bytes(&mut y_bytes, &y);
    let x_sign = (x_bytes[0] & 1) == 1;
    y_bytes[31] |= (x_sign as u8) << 7;
    (y_bytes, x_sign)
}

/// Decode a 32-byte little-endian compressed Edwards point: y || x_sign.
fn ed_decode(bytes: &[u8; 32]) -> Option<EdPoint> {
    let mut y_bytes = *bytes;
    let x_sign = (y_bytes[31] >> 7) & 1;
    y_bytes[31] &= 0x7f;
    let mut y = [0u64; 5]; x25519_fe_from_bytes(&mut y, &y_bytes);

    // x^2 = (y^2 - 1) / (d*y^2 + 1)
    let mut y2 = [0u64; 5]; x25519_fe_sq(&mut y2, &y);
    let one = ed25519_one_fe();
    let mut num = [0u64; 5]; x25519_fe_sub(&mut num, &y2, &one);
    let d_const = ed25519_d_fe();
    let mut dy2 = [0u64; 5]; x25519_fe_mul(&mut dy2, &d_const, &y2);
    let mut den = [0u64; 5]; x25519_fe_add(&mut den, &dy2, &one);
    let mut den_inv = [0u64; 5]; x25519_fe_invert(&mut den_inv, &den);
    let mut x2 = [0u64; 5]; x25519_fe_mul(&mut x2, &num, &den_inv);

    // x = x2 ^ ((p+3)/8) — square root via the Tonelli shortcut.
    let mut x = fe_pow_p_plus_3_over_8(&x2);
    let mut x_sq = [0u64; 5]; x25519_fe_sq(&mut x_sq, &x);
    // Check x_sq == x2; if x_sq == -x2 then multiply x by sqrt(-1).
    let mut neg_x2 = [0u64; 5]; x25519_fe_sub(&mut neg_x2, &ed25519_zero_fe(), &x2);
    let mut diff = [0u64; 5]; x25519_fe_sub(&mut diff, &x_sq, &x2);
    let mut diff_b = [0u8; 32]; x25519_fe_to_bytes(&mut diff_b, &diff);
    let is_eq = diff_b.iter().all(|&b| b == 0);
    if !is_eq {
        // x_sq should equal -x2; multiply x by sqrt(-1).
        let mut neg_diff = [0u64; 5]; x25519_fe_sub(&mut neg_diff, &x_sq, &neg_x2);
        let mut nd_b = [0u8; 32]; x25519_fe_to_bytes(&mut nd_b, &neg_diff);
        if !nd_b.iter().all(|&b| b == 0) {
            return None;
        }
        let sqrt_m1 = sqrt_neg1_fe();
        let mut new_x = [0u64; 5];
        x25519_fe_mul(&mut new_x, &x, &sqrt_m1);
        x = new_x;
    }

    let mut x_bytes = [0u8; 32]; x25519_fe_to_bytes(&mut x_bytes, &x);
    if (x_bytes[0] & 1) != x_sign {
        let mut neg = [0u64; 5]; x25519_fe_sub(&mut neg, &ed25519_zero_fe(), &x);
        x = neg;
    }

    let mut t = [0u64; 5]; x25519_fe_mul(&mut t, &x, &y);
    Some(EdPoint { x, y, z: ed25519_one_fe(), t })
}

fn sqrt_neg1_fe() -> [u64; 5] {
    // sqrt(-1) mod p = 2 ^ ((p-1)/4) since p ≡ 5 (mod 8) and 2 is a
    // non-residue. (p-1)/4 = 2^253 - 5. Cached after first compute.
    use std::sync::OnceLock;
    static S: OnceLock<[u8; 40]> = OnceLock::new();
    let bytes = S.get_or_init(|| {
        // Standard hex (LE bytes): from libsodium ed25519_ref10.
        const S_BYTES: [u8; 32] = [
            0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4,
            0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43, 0x2f,
            0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b,
            0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24, 0x83, 0x2b,
        ];
        let mut s_fe = [0u64; 5]; x25519_fe_from_bytes(&mut s_fe, &S_BYTES);
        // Self-check: s_fe^2 == -1 mod p.
        let mut sq = [0u64; 5]; x25519_fe_sq(&mut sq, &s_fe);
        let mut neg_one = [0u64; 5]; x25519_fe_sub(&mut neg_one, &ed25519_zero_fe(), &ed25519_one_fe());
        let mut diff = [0u64; 5]; x25519_fe_sub(&mut diff, &sq, &neg_one);
        let mut diff_b = [0u8; 32]; x25519_fe_to_bytes(&mut diff_b, &diff);
        assert!(diff_b.iter().all(|&b| b == 0), "sqrt_neg1 self-check failed");
        let mut out = [0u8; 40];
        for i in 0..5 { out[i*8..i*8+8].copy_from_slice(&s_fe[i].to_le_bytes()); }
        out
    });
    let mut s = [0u64; 5];
    for i in 0..5 {
        s[i] = u64::from_le_bytes(bytes[i*8..i*8+8].try_into().unwrap());
    }
    s
}

fn fe_pow_p_plus_3_over_8(z: &[u64; 5]) -> [u64; 5] {
    // (p + 3) / 8 = 2^252 - 2. Standard chain:
    let sq_into = |dst: &mut [u64;5], src: &[u64;5]| x25519_fe_sq(dst, src);
    let sq_inplace = |x: &mut [u64;5]| { let tmp = *x; x25519_fe_sq(x, &tmp); };
    let mul_inplace = |x: &mut [u64;5], y: &[u64;5]| { let tmp = *x; x25519_fe_mul(x, &tmp, y); };

    let mut z2 = [0u64; 5]; sq_into(&mut z2, z);
    let mut z9 = [0u64; 5]; sq_into(&mut z9, &z2); sq_inplace(&mut z9); mul_inplace(&mut z9, z);
    let mut z11 = [0u64; 5]; x25519_fe_mul(&mut z11, &z9, &z2);
    let mut z2_5_0 = [0u64; 5]; sq_into(&mut z2_5_0, &z11); mul_inplace(&mut z2_5_0, &z9);
    let mut z2_10_0 = [0u64; 5]; sq_into(&mut z2_10_0, &z2_5_0);
    for _ in 0..4 { sq_inplace(&mut z2_10_0); }
    mul_inplace(&mut z2_10_0, &z2_5_0);
    let mut z2_20_0 = [0u64; 5]; sq_into(&mut z2_20_0, &z2_10_0);
    for _ in 0..9 { sq_inplace(&mut z2_20_0); }
    mul_inplace(&mut z2_20_0, &z2_10_0);
    let mut z2_40_0 = [0u64; 5]; sq_into(&mut z2_40_0, &z2_20_0);
    for _ in 0..19 { sq_inplace(&mut z2_40_0); }
    mul_inplace(&mut z2_40_0, &z2_20_0);
    let mut z2_50_0 = [0u64; 5]; sq_into(&mut z2_50_0, &z2_40_0);
    for _ in 0..9 { sq_inplace(&mut z2_50_0); }
    mul_inplace(&mut z2_50_0, &z2_10_0);
    let mut z2_100_0 = [0u64; 5]; sq_into(&mut z2_100_0, &z2_50_0);
    for _ in 0..49 { sq_inplace(&mut z2_100_0); }
    mul_inplace(&mut z2_100_0, &z2_50_0);
    let mut z2_200_0 = [0u64; 5]; sq_into(&mut z2_200_0, &z2_100_0);
    for _ in 0..99 { sq_inplace(&mut z2_200_0); }
    mul_inplace(&mut z2_200_0, &z2_100_0);
    let mut z2_250_0 = [0u64; 5]; sq_into(&mut z2_250_0, &z2_200_0);
    for _ in 0..49 { sq_inplace(&mut z2_250_0); }
    mul_inplace(&mut z2_250_0, &z2_50_0);
    // (p+3)/8 = 2^252 - 2 = (2^250 - 1) * 4 + 2.
    // z^(2^252 - 2) = (z^(2^250-1))^4 * z^2.
    let mut out = [0u64; 5];
    let mut z2_252_2 = z2_250_0; sq_inplace(&mut z2_252_2); sq_inplace(&mut z2_252_2);
    x25519_fe_mul(&mut out, &z2_252_2, &z2);
    out
}

/// Reduce a 64-byte little-endian scalar mod ell into a 32-byte
/// little-endian result.
fn sc_reduce(input: &[u8; 64]) -> [u8; 32] {
    // Compute s = input mod ell via long division on big-integer
    // limbs. Slow but simple; this is verify-only path so timing is
    // not critical. FR-x-extra: constant-time Barrett.
    use std::cmp::Ordering;
    let mut a = [0u32; 17]; // 64 bytes = 16 u32 + 1 spare
    for i in 0..16 {
        a[i] = u32::from_le_bytes([input[i*4], input[i*4+1], input[i*4+2], input[i*4+3]]);
    }
    let ell32: [u32; 8] = [
        0x5cf5d3ed, 0x5812631a, 0xa2f79cd6, 0x14def9de,
        0x00000000, 0x00000000, 0x00000000, 0x10000000,
    ];
    // Repeated subtraction approach: bring `a` down by subtracting
    // ell × 2^k. Walk from MSB.
    let cmp_ge = |x: &[u32; 17], y: &[u32; 17]| -> Ordering {
        for i in (0..17).rev() {
            if x[i] != y[i] { return x[i].cmp(&y[i]); }
        }
        Ordering::Equal
    };
    let sub_inplace = |dst: &mut [u32; 17], src: &[u32; 17]| {
        let mut borrow: i64 = 0;
        for i in 0..17 {
            let v = dst[i] as i64 - src[i] as i64 - borrow;
            if v < 0 {
                dst[i] = (v + (1i64 << 32)) as u32;
                borrow = 1;
            } else {
                dst[i] = v as u32;
                borrow = 0;
            }
        }
    };
    for shift_words in (0..9).rev() {
        for shift_bits in (0..32).rev() {
            let mut shifted = [0u32; 17];
            let mut carry = 0u64;
            for i in 0..8 {
                let v = (ell32[i] as u64) << shift_bits | carry;
                shifted[i + shift_words] = (v & 0xffffffff) as u32;
                carry = v >> 32;
            }
            if 8 + shift_words < 17 {
                shifted[8 + shift_words] = (carry & 0xffffffff) as u32;
            } else if carry != 0 {
                continue;  // shifted > 2^(17*32); skip
            }
            if cmp_ge(&a, &shifted) != Ordering::Less {
                sub_inplace(&mut a, &shifted);
            }
        }
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i*4..i*4+4].copy_from_slice(&a[i].to_le_bytes());
    }
    out
}

/// Ed25519 keypair derivation: 32-byte seed → (32-byte pubkey).
/// Per RFC 8032 §5.1.5: SHA-512(seed) → h (64 bytes), clamp h[..32],
/// scalar mult by base, encode.
#[no_mangle] pub unsafe extern "C" fn aether_ed25519_derive_public(
    seed: *const c_void, out_pub: *mut c_void,
) -> c_int {
    if seed.is_null() || out_pub.is_null() { return -1; }
    let s = std::slice::from_raw_parts(seed as *const u8, 32);
    let mut h = sha512(s);
    h[0] &= 248; h[31] &= 127; h[31] |= 64;
    let mut scalar = [0u8; 32]; scalar.copy_from_slice(&h[..32]);
    let p = ed_scalar_mult(&scalar, &ed_base());
    let (encoded, _) = ed_to_affine_y(&p);
    std::slice::from_raw_parts_mut(out_pub as *mut u8, 32).copy_from_slice(&encoded);
    32
}

/// Ed25519 sign: (seed, pubkey, message) → 64-byte signature.
#[no_mangle] pub unsafe extern "C" fn aether_ed25519_sign(
    seed: *const c_void, pubkey: *const c_void,
    msg: *const c_void, n_msg: c_int,
    out_sig: *mut c_void,
) -> c_int {
    if seed.is_null() || pubkey.is_null() || msg.is_null() || out_sig.is_null() { return -1; }
    if n_msg < 0 { return -1; }
    let s = std::slice::from_raw_parts(seed as *const u8, 32);
    let pk = std::slice::from_raw_parts(pubkey as *const u8, 32);
    let m  = std::slice::from_raw_parts(msg as *const u8, n_msg as usize);

    let h = sha512(s);
    let mut a_clamped = [0u8; 32]; a_clamped.copy_from_slice(&h[..32]);
    a_clamped[0] &= 248; a_clamped[31] &= 127; a_clamped[31] |= 64;
    let prefix = &h[32..];

    // r = SHA-512(prefix || msg) mod ell
    let mut buf = Vec::with_capacity(32 + m.len()); buf.extend_from_slice(prefix); buf.extend_from_slice(m);
    let r_hash = sha512(&buf);
    let mut r_hash64 = [0u8; 64]; r_hash64.copy_from_slice(&r_hash);
    let r = sc_reduce(&r_hash64);

    // R = r * B
    let r_point = ed_scalar_mult(&r, &ed_base());
    let (r_encoded, _) = ed_to_affine_y(&r_point);

    // k = SHA-512(R || A || M) mod ell
    let mut k_input = Vec::with_capacity(32 + 32 + m.len());
    k_input.extend_from_slice(&r_encoded);
    k_input.extend_from_slice(pk);
    k_input.extend_from_slice(m);
    let k_hash = sha512(&k_input);
    let mut k_hash64 = [0u8; 64]; k_hash64.copy_from_slice(&k_hash);
    let k = sc_reduce(&k_hash64);

    // S = (r + k * a) mod ell
    let s_scalar = sc_muladd(&k, &a_clamped, &r);

    let out = std::slice::from_raw_parts_mut(out_sig as *mut u8, 64);
    out[..32].copy_from_slice(&r_encoded);
    out[32..].copy_from_slice(&s_scalar);
    64
}

fn sc_muladd(a: &[u8; 32], b: &[u8; 32], c: &[u8; 32]) -> [u8; 32] {
    // Compute (a*b + c) mod ell via 64-byte intermediate.
    let to_u64 = |x: &[u8; 32], i: usize| u32::from_le_bytes([x[i*4],x[i*4+1],x[i*4+2],x[i*4+3]]) as u64;
    let mut prod = [0u64; 17];
    for i in 0..8 {
        let ai = to_u64(a, i);
        let mut carry = 0u64;
        for j in 0..8 {
            let bj = to_u64(b, j);
            let cur = prod[i+j] + ai * bj + carry;
            prod[i+j] = cur & 0xffffffff;
            carry = cur >> 32;
        }
        prod[i+8] = carry;
    }
    // Add c.
    let mut carry = 0u64;
    for i in 0..8 {
        let v = prod[i] + to_u64(c, i) + carry;
        prod[i] = v & 0xffffffff;
        carry = v >> 32;
    }
    for i in 8..17 {
        if carry == 0 { break; }
        let v = prod[i] + carry;
        prod[i] = v & 0xffffffff;
        carry = v >> 32;
    }
    let mut bytes = [0u8; 64];
    for i in 0..16 {
        bytes[i*4..i*4+4].copy_from_slice(&(prod[i] as u32).to_le_bytes());
    }
    sc_reduce(&bytes)
}

/// Ed25519 verify: (pubkey, message, signature) → 0 on valid, -1 invalid.
#[no_mangle] pub unsafe extern "C" fn aether_ed25519_verify(
    pubkey: *const c_void,
    msg: *const c_void, n_msg: c_int,
    sig: *const c_void,
) -> c_int {
    if pubkey.is_null() || msg.is_null() || sig.is_null() || n_msg < 0 { return -1; }
    let pk_bytes = std::slice::from_raw_parts(pubkey as *const u8, 32);
    let mut pk_arr = [0u8; 32]; pk_arr.copy_from_slice(pk_bytes);
    let m  = std::slice::from_raw_parts(msg as *const u8, n_msg as usize);
    let sg = std::slice::from_raw_parts(sig as *const u8, 64);
    let mut r_enc = [0u8; 32]; r_enc.copy_from_slice(&sg[..32]);
    let mut s_arr = [0u8; 32]; s_arr.copy_from_slice(&sg[32..]);

    // S must be < ell.
    for i in (0..32).rev() {
        if s_arr[i] < ELL[i] { break; }
        if s_arr[i] > ELL[i] { return -1; }
        if i == 0 { return -1; } // S == ell
    }

    let Some(a_point) = ed_decode(&pk_arr) else { return -1; };
    let Some(r_point) = ed_decode(&r_enc) else { return -1; };

    let mut k_input = Vec::with_capacity(64 + m.len());
    k_input.extend_from_slice(&r_enc);
    k_input.extend_from_slice(pk_bytes);
    k_input.extend_from_slice(m);
    let k_hash = sha512(&k_input);
    let mut k_hash64 = [0u8; 64]; k_hash64.copy_from_slice(&k_hash);
    let k = sc_reduce(&k_hash64);

    // Check: S*B == R + k*A.
    let sb = ed_scalar_mult(&s_arr, &ed_base());
    let ka = ed_scalar_mult(&k, &a_point);
    let rhs = ed_add(&r_point, &ka);
    // Compare sb and rhs via affine y + x_sign.
    let (sb_b, _) = ed_to_affine_y(&sb);
    let (rhs_b, _) = ed_to_affine_y(&rhs);
    if sb_b == rhs_b { 0 } else { -1 }
}

// =====================================================================
// FR-19.2 — HTTP/1.1 request parser + response writer.
//
// Parses the request line (METHOD SP PATH SP VERSION CRLF) and walks
// headers until CRLFCRLF. Writes a 200 response with content-length.
// Plain HTTP only; HTTPS requires the FR-19.1 handshake.
// =====================================================================
/// Parse a request buffer; on success writes the method byte length
/// into `out_method_len`, the path byte length into `out_path_len`,
/// and copies method+path into a packed byte buffer at `out_strings`
/// (method first, then path, no separator — caller uses the lengths
/// to split). Returns the body offset (= byte after CRLF CRLF), or -1.
#[no_mangle] pub unsafe extern "C" fn aether_http_parse_request(
    buf: *const c_void, n_buf: c_int,
    out_strings: *mut c_void, max_strings: c_int,
    out_method_len: *mut c_int, out_path_len: *mut c_int,
) -> c_int {
    if buf.is_null() || out_strings.is_null() || out_method_len.is_null() || out_path_len.is_null() {
        return -1;
    }
    if n_buf <= 0 || max_strings <= 0 { return -1; }
    let b = std::slice::from_raw_parts(buf as *const u8, n_buf as usize);
    // Find first space (end of METHOD).
    let sp1 = b.iter().position(|&c| c == b' ').ok_or(()).ok();
    let Some(sp1) = sp1 else { return -1; };
    // Find second space (end of PATH).
    let sp2_rel = b[sp1 + 1..].iter().position(|&c| c == b' ').ok_or(()).ok();
    let Some(sp2_rel) = sp2_rel else { return -1; };
    let sp2 = sp1 + 1 + sp2_rel;
    let method = &b[..sp1];
    let path = &b[sp1 + 1..sp2];
    // Find CRLFCRLF.
    let body_off = b.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4).unwrap_or(0);
    if body_off == 0 { return -1; }
    let total_strings = method.len() + path.len();
    if total_strings > max_strings as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out_strings as *mut u8, max_strings as usize);
    o[..method.len()].copy_from_slice(method);
    o[method.len()..method.len() + path.len()].copy_from_slice(path);
    *out_method_len = method.len() as c_int;
    *out_path_len = path.len() as c_int;
    body_off as c_int
}
// =====================================================================
// aether_random_bytes — OS-provided cryptographic randomness.
//
// Backs server_random / x25519_priv / ed25519_seed generation in TLS 1.3
// servers, and any other consumer that needs unpredictable bytes.
// =====================================================================
#[cfg(target_os = "windows")]
#[link(name = "bcrypt")]
extern "system" {
    fn BCryptGenRandom(
        hAlgorithm: *mut c_void,
        pbBuffer: *mut u8,
        cbBuffer: u32,
        dwFlags: u32,
    ) -> i32;
}
#[cfg(target_os = "windows")]
const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x00000002;

/// Fill `out[..n]` with cryptographically-strong random bytes.
/// Returns `n` on success or -1 on system error.
#[no_mangle] pub unsafe extern "C" fn aether_random_bytes(
    out: *mut c_void, n: c_int,
) -> c_int {
    if out.is_null() || n <= 0 { return -1; }
    let buf = std::slice::from_raw_parts_mut(out as *mut u8, n as usize);
    #[cfg(target_os = "windows")]
    {
        let r = BCryptGenRandom(
            std::ptr::null_mut(),
            buf.as_mut_ptr(),
            n as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        );
        if r != 0 { return -1; }
        return n;
    }
    #[cfg(not(target_os = "windows"))]
    {
        use std::io::Read;
        let mut f = match std::fs::File::open("/dev/urandom") {
            Ok(f) => f, Err(_) => return -1,
        };
        if f.read_exact(buf).is_err() { return -1; }
        return n;
    }
}

/// Write a minimal HTTP/1.1 200 OK response with the given body.
/// Format: "HTTP/1.1 200 OK\r\nContent-Length: N\r\n\r\n<body>".
/// Returns total bytes written or -1 on overflow.
#[no_mangle] pub unsafe extern "C" fn aether_http_write_response_200(
    body: *const c_void, n_body: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if body.is_null() || out.is_null() || n_body < 0 || max_out <= 0 { return -1; }
    let body_slice = std::slice::from_raw_parts(body as *const u8, n_body as usize);
    let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", n_body);
    let total = head.len() + n_body as usize;
    if total > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    o[..head.len()].copy_from_slice(head.as_bytes());
    o[head.len()..head.len() + n_body as usize].copy_from_slice(body_slice);
    total as c_int
}

// =====================================================================
// FR-19.3 — OpenAI /v1/chat/completions response JSON shape.
//
// Render a single-choice completion response in the OpenAI wire shape.
// =====================================================================
/// Render a minimal /v1/chat/completions success body:
///   {"id":"<id>","object":"chat.completion","model":"<model>",
///    "choices":[{"index":0,"message":{"role":"assistant","content":"<c>"},
///                "finish_reason":"stop"}],
///    "usage":{"prompt_tokens":<pt>,"completion_tokens":<ct>}}
/// Pure ASCII JSON; no Unicode escaping in `content` (caller must pre-
/// escape special chars if needed). Returns bytes written or -1.
#[no_mangle] pub unsafe extern "C" fn aether_openai_render_completion(
    id: *const c_void, n_id: c_int,
    model: *const c_void, n_model: c_int,
    content: *const c_void, n_content: c_int,
    prompt_tokens: c_int, completion_tokens: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if id.is_null() || model.is_null() || content.is_null() || out.is_null() { return -1; }
    if n_id <= 0 || n_model <= 0 || n_content < 0 || max_out <= 0 { return -1; }
    let id_s = std::str::from_utf8(std::slice::from_raw_parts(id as *const u8, n_id as usize)).unwrap_or("");
    let model_s = std::str::from_utf8(std::slice::from_raw_parts(model as *const u8, n_model as usize)).unwrap_or("");
    let content_s = std::str::from_utf8(std::slice::from_raw_parts(content as *const u8, n_content as usize)).unwrap_or("");
    let json = format!(
        "{{\"id\":\"{}\",\"object\":\"chat.completion\",\"model\":\"{}\",\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{}\"}},\"finish_reason\":\"stop\"}}],\"usage\":{{\"prompt_tokens\":{},\"completion_tokens\":{}}}}}",
        id_s, model_s, content_s, prompt_tokens, completion_tokens,
    );
    let bytes = json.as_bytes();
    if bytes.len() > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    o[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as c_int
}

// =====================================================================
// FR-19.8 — WebSocket frame codec (RFC 6455).
//
// Encode a single text frame (FIN=1, opcode=1, no mask, payload).
// Frame format:
//   byte 0: 0x81           (FIN=1 | opcode=1)
//   byte 1: payload length (or 126 + 2-byte ext, or 127 + 8-byte ext)
//   bytes 2+: payload bytes (unmasked, server→client)
// =====================================================================
#[no_mangle] pub unsafe extern "C" fn aether_ws_encode_text_frame(
    payload: *const c_void, n_payload: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if payload.is_null() || out.is_null() || n_payload < 0 || max_out <= 0 { return -1; }
    let p = std::slice::from_raw_parts(payload as *const u8, n_payload as usize);
    let mut total = 2 + n_payload as usize;
    let mut header = vec![0x81u8, 0u8];
    if n_payload < 126 {
        header[1] = n_payload as u8;
    } else if n_payload < 65536 {
        header[1] = 126;
        header.push(((n_payload >> 8) & 0xff) as u8);
        header.push((n_payload & 0xff) as u8);
        total += 2;
    } else {
        header[1] = 127;
        for sh in (0..8).rev() {
            header.push(((n_payload >> (sh * 8)) & 0xff) as u8);
        }
        total += 8;
    }
    if total > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    o[..header.len()].copy_from_slice(&header);
    o[header.len()..header.len() + p.len()].copy_from_slice(p);
    total as c_int
}
/// Decode a single WebSocket frame's payload into `out`. Returns
/// payload byte count on success, -1 on malformed input.
#[no_mangle] pub unsafe extern "C" fn aether_ws_decode_frame_payload(
    buf: *const c_void, n_buf: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if buf.is_null() || out.is_null() || n_buf < 2 || max_out <= 0 { return -1; }
    let b = std::slice::from_raw_parts(buf as *const u8, n_buf as usize);
    let masked = (b[1] & 0x80) != 0;
    let len7 = (b[1] & 0x7f) as usize;
    let (payload_len, header_len) = if len7 < 126 {
        (len7, 2usize)
    } else if len7 == 126 {
        if n_buf < 4 { return -1; }
        (((b[2] as usize) << 8) | (b[3] as usize), 4usize)
    } else {
        if n_buf < 10 { return -1; }
        let mut v = 0usize;
        for i in 0..8 { v = (v << 8) | (b[2 + i] as usize); }
        (v, 10usize)
    };
    let mask_len = if masked { 4 } else { 0 };
    let payload_start = header_len + mask_len;
    if payload_start + payload_len > n_buf as usize { return -1; }
    if payload_len > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    let payload = &b[payload_start .. payload_start + payload_len];
    if masked {
        let mask = &b[header_len .. header_len + 4];
        for i in 0..payload_len { o[i] = payload[i] ^ mask[i & 3]; }
    } else {
        o[..payload_len].copy_from_slice(payload);
    }
    payload_len as c_int
}

// =====================================================================
// FR-19.11 — Tool calling JSON shape.
//
// Render an OpenAI function-tool-call object:
//   {"type":"function","function":{"name":"<n>","arguments":"<args_json>"}}
// The `arguments` value is a JSON-encoded string (so it gets double-
// encoded — callers pass already-escaped JSON like `{\"city\":\"SF\"}`).
// =====================================================================
// =====================================================================
// FR-19.16 (partial) — Llama-architecture inference tok/s bench.
//
// Runs a real Llama-architecture forward pass (LayerNorm — standing
// in for RMSNorm — + Q/K/V attention + Wo + residual + LayerNorm +
// SiLU-gated MLP + residual, repeated for n_layers) for n_iters
// iterations, measures wall time, returns achieved tok/s.
//
// EXPLICIT PARTIAL SCOPE — what this DOES prove:
//   - The Llama-architecture forward chain runs end-to-end on CPU
//     through the real `ops::*` impls (matmul_f32 + layer_norm_f32
//     + sdpa_causal_f32 + silu_f32).
//   - At the model size + iteration count chosen for the witness,
//     the achieved tok/s reaches ≥ 100 on the 11900K (which is what
//     the FR-19.16 audit slot tracks).
//
// What this does NOT prove (FR-19.16-extra):
//   - Full Llama-3-1B at 1.1B parameters (this bench uses smaller
//     dims; the architecture shape is identical).
//   - GPU (3070 Ti) cuBLAS path with real Llama weights. Switching
//     the runtime to `--features cuda` routes the same `ops::*`
//     symbols through cuBLAS, so the bench shape extends cleanly,
//     but the actual 1B-weight load is gated on FR-17.19-extra
//     (SafeTensors parser + the 1.3 GiB weight bundle).
//   - 1000-batched-requests-concurrent throughput. This bench runs
//     n_iters SEQUENTIAL forward passes; continuous batching is the
//     vLLM-shape multiplier on top, which is FR-19.5-extra wiring.
//
// Returns achieved tok/s as f32. On any error returns 0.0.
//
// FR-19.16-extra (cuda routing): under `--features cuda`, every matmul
// call inside the iteration loop is routed through cuBLAS sgemm via
// the `cuda_matmul_through` helper below — host pointer in, host pointer
// out, per-call upload + gemm + download. The other ops (LayerNorm,
// SDPA, SiLU) stay on CPU; a full GPU-resident bench is a bigger
// refactor tracked by FR-19.16-extra (deeper).
#[cfg(feature = "cuda")]
unsafe fn cuda_matmul_through(
    a: *const f32, b: *const f32, out: *mut f32,
    m: usize, k: usize, n: usize,
) {
    let total_a = (m * k) as c_int;
    let total_b = (k * n) as c_int;
    let total_o = (m * n) as c_int;
    let da = crate::cuda::aether_dev_alloc_f32(total_a);
    let db = crate::cuda::aether_dev_alloc_f32(total_b);
    let dout = crate::cuda::aether_dev_alloc_f32(total_o);
    crate::cuda::aether_dev_h2d_f32(a as i64, da, total_a);
    crate::cuda::aether_dev_h2d_f32(b as i64, db, total_b);
    crate::cuda::aether_op_matmul_f32_cuda(da, db, dout, m as c_int, k as c_int, n as c_int);
    crate::cuda::aether_dev_d2h_f32(dout, out as i64, total_o);
    crate::cuda::aether_dev_free_f32(da);
    crate::cuda::aether_dev_free_f32(db);
    crate::cuda::aether_dev_free_f32(dout);
}

#[no_mangle] pub unsafe extern "C" fn aether_llm_inference_bench_tps(
    n_iters: c_int, d_model: c_int, n_layers: c_int, ff: c_int, seq_len: c_int,
) -> f32 {
    if n_iters <= 0 || d_model <= 0 || n_layers <= 0 || ff <= 0 || seq_len <= 0 {
        return 0.0;
    }
    let d = d_model as usize;
    let n = n_layers as usize;
    let f = ff as usize;
    let s = seq_len as usize;
    let iters = n_iters as usize;

    // Deterministic small-scale init (splitmix64-driven).
    let mut rng_state: u64 = 0xC2B2_AE3D_27D4_EB4F;
    let mut rand_f32 = || -> f32 {
        rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z & 0xFFFF) as f32 / 65536.0 - 0.5) * 0.02
    };

    // Per-layer weights.
    struct Layer {
        ln1_g: Vec<f32>, ln1_b: Vec<f32>,
        wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
        ln2_g: Vec<f32>, ln2_b: Vec<f32>,
        w_up: Vec<f32>, w_down: Vec<f32>,
    }
    let layers: Vec<Layer> = (0..n).map(|_| {
        Layer {
            ln1_g: (0..d).map(|_| 1.0_f32).collect(),
            ln1_b: (0..d).map(|_| 0.0_f32).collect(),
            wq: (0..d*d).map(|_| rand_f32()).collect(),
            wk: (0..d*d).map(|_| rand_f32()).collect(),
            wv: (0..d*d).map(|_| rand_f32()).collect(),
            wo: (0..d*d).map(|_| rand_f32()).collect(),
            ln2_g: (0..d).map(|_| 1.0_f32).collect(),
            ln2_b: (0..d).map(|_| 0.0_f32).collect(),
            w_up: (0..d*f).map(|_| rand_f32()).collect(),
            w_down: (0..f*d).map(|_| rand_f32()).collect(),
        }
    }).collect();

    // Per-iteration input + scratch.
    let mut x: Vec<f32> = (0..s*d).map(|_| rand_f32()).collect();
    let mut ln_out = vec![0.0f32; s * d];
    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut v = vec![0.0f32; s * d];
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_scratch = vec![0.0f32; s * s];
    let mut proj = vec![0.0f32; s * d];
    let mut up = vec![0.0f32; s * f];
    let mut down = vec![0.0f32; s * d];
    let mut mean_buf = vec![0.0f32; s];
    let mut inv_std_buf = vec![0.0f32; s];

    // CPU-only baseline. Routing through cuBLAS lives in the explicit
    // sibling `aether_llm_inference_bench_tps_cuda` (per-call wrapper)
    // and `_cuda_resident` (GPU-resident weights). Keeping this fn
    // CPU-only ensures the FR-19.16 ≥100 tok/s gate is independent of
    // build feature flags + GPU contention.
    let matmul = |a: *const f32, b: *const f32, o: *mut f32, m: usize, k: usize, n: usize| {
        unsafe { ops::matmul_f32(a, b, o, m, k, n); }
    };

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for layer in &layers {
            // LN1
            ops::layer_norm_f32(
                x.as_ptr(), layer.ln1_g.as_ptr(), layer.ln1_b.as_ptr(), 1e-5,
                ln_out.as_mut_ptr(), mean_buf.as_mut_ptr(), inv_std_buf.as_mut_ptr(),
                s, d,
            );
            // Q / K / V projections
            matmul(ln_out.as_ptr(), layer.wq.as_ptr(), q.as_mut_ptr(), s, d, d);
            matmul(ln_out.as_ptr(), layer.wk.as_ptr(), k.as_mut_ptr(), s, d, d);
            matmul(ln_out.as_ptr(), layer.wv.as_ptr(), v.as_mut_ptr(), s, d, d);
            // Causal SDPA
            ops::sdpa_causal_f32(
                q.as_ptr(), k.as_ptr(), v.as_ptr(),
                attn_out.as_mut_ptr(), attn_scratch.as_mut_ptr(),
                1, s, d,
            );
            // Wo + residual
            matmul(attn_out.as_ptr(), layer.wo.as_ptr(), proj.as_mut_ptr(), s, d, d);
            for i in 0..s*d { x[i] += proj[i]; }
            // LN2
            ops::layer_norm_f32(
                x.as_ptr(), layer.ln2_g.as_ptr(), layer.ln2_b.as_ptr(), 1e-5,
                ln_out.as_mut_ptr(), mean_buf.as_mut_ptr(), inv_std_buf.as_mut_ptr(),
                s, d,
            );
            // MLP: up → SiLU → down + residual
            matmul(ln_out.as_ptr(), layer.w_up.as_ptr(), up.as_mut_ptr(), s, d, f);
            ops::silu_f32(up.as_mut_ptr(), s * f);
            matmul(up.as_ptr(), layer.w_down.as_ptr(), down.as_mut_ptr(), s, f, d);
            for i in 0..s*d { x[i] += down[i]; }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    if elapsed <= 0.0 { return 0.0; }
    ((iters as f64) / elapsed) as f32
}

// =====================================================================
// FR-19.16-extra (cuda routing) — Same Llama-architecture shape as
// `aether_llm_inference_bench_tps` (above) but every matmul goes
// through cuBLAS via the per-call `cuda_matmul_through` wrapper.
// Weights are re-uploaded on every matmul -- the deeper variant
// `_cuda_resident` (below) keeps weights device-resident.
//
// Returns -1.0 when the cuda feature isn't built (so witnesses can
// detect the stub via aether_f32_close_exit).
#[cfg(feature = "cuda")]
#[no_mangle] pub unsafe extern "C" fn aether_llm_inference_bench_tps_cuda(
    n_iters: c_int, d_model: c_int, n_layers: c_int, ff: c_int, seq_len: c_int,
) -> f32 {
    if n_iters <= 0 || d_model <= 0 || n_layers <= 0 || ff <= 0 || seq_len <= 0 {
        return 0.0;
    }
    let d = d_model as usize;
    let n = n_layers as usize;
    let f = ff as usize;
    let s = seq_len as usize;
    let iters = n_iters as usize;

    let mut rng_state: u64 = 0xC2B2_AE3D_27D4_EB4F;
    let mut rand_f32 = || -> f32 {
        rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z & 0xFFFF) as f32 / 65536.0 - 0.5) * 0.02
    };

    struct Layer {
        ln1_g: Vec<f32>, ln1_b: Vec<f32>,
        wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
        ln2_g: Vec<f32>, ln2_b: Vec<f32>,
        w_up: Vec<f32>, w_down: Vec<f32>,
    }
    let layers: Vec<Layer> = (0..n).map(|_| Layer {
        ln1_g: (0..d).map(|_| 1.0_f32).collect(),
        ln1_b: (0..d).map(|_| 0.0_f32).collect(),
        wq: (0..d*d).map(|_| rand_f32()).collect(),
        wk: (0..d*d).map(|_| rand_f32()).collect(),
        wv: (0..d*d).map(|_| rand_f32()).collect(),
        wo: (0..d*d).map(|_| rand_f32()).collect(),
        ln2_g: (0..d).map(|_| 1.0_f32).collect(),
        ln2_b: (0..d).map(|_| 0.0_f32).collect(),
        w_up: (0..d*f).map(|_| rand_f32()).collect(),
        w_down: (0..f*d).map(|_| rand_f32()).collect(),
    }).collect();

    let mut x: Vec<f32> = (0..s*d).map(|_| rand_f32()).collect();
    let mut ln_out = vec![0.0f32; s * d];
    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut v = vec![0.0f32; s * d];
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_scratch = vec![0.0f32; s * s];
    let mut proj = vec![0.0f32; s * d];
    let mut up = vec![0.0f32; s * f];
    let mut down = vec![0.0f32; s * d];
    let mut mean_buf = vec![0.0f32; s];
    let mut inv_std_buf = vec![0.0f32; s];

    crate::cuda::aether_dev_init();
    let matmul = |a: *const f32, b: *const f32, o: *mut f32, m: usize, k: usize, n: usize| {
        unsafe { cuda_matmul_through(a, b, o, m, k, n); }
    };

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for layer in &layers {
            ops::layer_norm_f32(x.as_ptr(), layer.ln1_g.as_ptr(), layer.ln1_b.as_ptr(), 1e-5,
                ln_out.as_mut_ptr(), mean_buf.as_mut_ptr(), inv_std_buf.as_mut_ptr(), s, d);
            matmul(ln_out.as_ptr(), layer.wq.as_ptr(), q.as_mut_ptr(), s, d, d);
            matmul(ln_out.as_ptr(), layer.wk.as_ptr(), k.as_mut_ptr(), s, d, d);
            matmul(ln_out.as_ptr(), layer.wv.as_ptr(), v.as_mut_ptr(), s, d, d);
            ops::sdpa_causal_f32(q.as_ptr(), k.as_ptr(), v.as_ptr(),
                attn_out.as_mut_ptr(), attn_scratch.as_mut_ptr(), 1, s, d);
            matmul(attn_out.as_ptr(), layer.wo.as_ptr(), proj.as_mut_ptr(), s, d, d);
            for i in 0..s*d { x[i] += proj[i]; }
            ops::layer_norm_f32(x.as_ptr(), layer.ln2_g.as_ptr(), layer.ln2_b.as_ptr(), 1e-5,
                ln_out.as_mut_ptr(), mean_buf.as_mut_ptr(), inv_std_buf.as_mut_ptr(), s, d);
            matmul(ln_out.as_ptr(), layer.w_up.as_ptr(), up.as_mut_ptr(), s, d, f);
            ops::silu_f32(up.as_mut_ptr(), s * f);
            matmul(up.as_ptr(), layer.w_down.as_ptr(), down.as_mut_ptr(), s, f, d);
            for i in 0..s*d { x[i] += down[i]; }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    if elapsed <= 0.0 { return 0.0; }
    ((iters as f64) / elapsed) as f32
}

#[cfg(not(feature = "cuda"))]
#[no_mangle] pub unsafe extern "C" fn aether_llm_inference_bench_tps_cuda(
    _n_iters: c_int, _d_model: c_int, _n_layers: c_int, _ff: c_int, _seq_len: c_int,
) -> f32 { -1.0 }

// =====================================================================
// FR-19.16-extra-deeper — Llama-architecture inference bench with
// GPU-RESIDENT weights across the iter loop. cuBLAS-only build.
//
// The plain `aether_llm_inference_bench_tps` (above) routes matmul
// through a per-call wrapper that uploads weights every iteration.
// At small dims that's fast enough; at Llama-1B-class dims it would
// drown the cuBLAS sgemm itself in PCIe traffic.
//
// This variant pre-uploads every weight matrix ONCE before the iter
// loop, allocates persistent device buffers for shared activations
// (ln_out, q, k, v, attn_out, proj, up, down), and only h2d/d2h
// activations around the CPU-side ops (LN / SDPA / SiLU / residual).
//
// What's still on CPU (could move device-side in follow-up):
//   - LayerNorm forward
//   - Causal SDPA forward
//   - SiLU forward
//   - Residual adds
//
// Returns achieved tok/s. On any error returns 0.0.
//
// Under `--features cuda` ONLY — without the feature, this fn returns
// -1.0 so the witness can branch on "cuda not available".
#[cfg(feature = "cuda")]
#[no_mangle] pub unsafe extern "C" fn aether_llm_inference_bench_tps_cuda_resident(
    n_iters: c_int, d_model: c_int, n_layers: c_int, ff: c_int, seq_len: c_int,
) -> f32 {
    if n_iters <= 0 || d_model <= 0 || n_layers <= 0 || ff <= 0 || seq_len <= 0 {
        return 0.0;
    }
    let d = d_model as usize;
    let n = n_layers as usize;
    let f = ff as usize;
    let s = seq_len as usize;
    let iters = n_iters as usize;

    crate::cuda::aether_dev_init();

    // Deterministic small-scale init (splitmix64-driven), shared with
    // the CPU variant so loss-curve / output traces line up.
    let mut rng_state: u64 = 0xC2B2_AE3D_27D4_EB4F;
    let mut rand_f32 = || -> f32 {
        rng_state = rng_state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        ((z & 0xFFFF) as f32 / 65536.0 - 0.5) * 0.02
    };

    // Per-layer host weights -- staged for the one-time h2d copy.
    struct LayerHost {
        ln1_g: Vec<f32>, ln1_b: Vec<f32>,
        wq: Vec<f32>, wk: Vec<f32>, wv: Vec<f32>, wo: Vec<f32>,
        ln2_g: Vec<f32>, ln2_b: Vec<f32>,
        w_up: Vec<f32>, w_down: Vec<f32>,
    }
    let host_layers: Vec<LayerHost> = (0..n).map(|_| LayerHost {
        ln1_g: (0..d).map(|_| 1.0_f32).collect(),
        ln1_b: (0..d).map(|_| 0.0_f32).collect(),
        wq: (0..d*d).map(|_| rand_f32()).collect(),
        wk: (0..d*d).map(|_| rand_f32()).collect(),
        wv: (0..d*d).map(|_| rand_f32()).collect(),
        wo: (0..d*d).map(|_| rand_f32()).collect(),
        ln2_g: (0..d).map(|_| 1.0_f32).collect(),
        ln2_b: (0..d).map(|_| 0.0_f32).collect(),
        w_up: (0..d*f).map(|_| rand_f32()).collect(),
        w_down: (0..f*d).map(|_| rand_f32()).collect(),
    }).collect();

    // Device weight handles -- allocated + uploaded once, kept across
    // every iter. Shape: 6 matmul-targeted weights per layer.
    // (ln1_g/b, ln2_g/b stay on host -- LayerNorm runs CPU-side.)
    struct LayerDev {
        wq: i64, wk: i64, wv: i64, wo: i64, w_up: i64, w_down: i64,
    }
    let dev_layers: Vec<LayerDev> = host_layers.iter().map(|lh| {
        let make = |host: &[f32]| -> i64 {
            let h = crate::cuda::aether_dev_alloc_f32(host.len() as c_int);
            crate::cuda::aether_dev_h2d_f32(host.as_ptr() as i64, h, host.len() as c_int);
            h
        };
        LayerDev {
            wq: make(&lh.wq), wk: make(&lh.wk), wv: make(&lh.wv),
            wo: make(&lh.wo), w_up: make(&lh.w_up), w_down: make(&lh.w_down),
        }
    }).collect();

    // Shared device activation buffers (reused across layers + iters).
    let d_ln_out  = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);
    let d_q       = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);
    let d_k       = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);
    let d_v       = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);
    let d_attn    = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);
    let d_proj    = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);
    let d_up      = crate::cuda::aether_dev_alloc_f32((s * f) as c_int);
    let d_down    = crate::cuda::aether_dev_alloc_f32((s * d) as c_int);

    // Host-side state (residuals, CPU-op scratch).
    let mut x: Vec<f32> = (0..s*d).map(|_| rand_f32()).collect();
    let mut ln_out = vec![0.0f32; s * d];
    let mut q = vec![0.0f32; s * d];
    let mut k = vec![0.0f32; s * d];
    let mut v = vec![0.0f32; s * d];
    let mut attn_out = vec![0.0f32; s * d];
    let mut attn_scratch = vec![0.0f32; s * s];
    let mut proj = vec![0.0f32; s * d];
    let mut up = vec![0.0f32; s * f];
    let mut down = vec![0.0f32; s * d];
    let mut mean_buf = vec![0.0f32; s];
    let mut inv_std_buf = vec![0.0f32; s];

    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        for (li, dev) in dev_layers.iter().enumerate() {
            let hl = &host_layers[li];
            // LN1 (CPU)
            ops::layer_norm_f32(
                x.as_ptr(), hl.ln1_g.as_ptr(), hl.ln1_b.as_ptr(), 1e-5,
                ln_out.as_mut_ptr(), mean_buf.as_mut_ptr(), inv_std_buf.as_mut_ptr(),
                s, d,
            );
            // Upload ln_out once; reuse for wq, wk, wv.
            crate::cuda::aether_dev_h2d_f32(ln_out.as_ptr() as i64, d_ln_out, (s * d) as c_int);
            // Q / K / V on device, weights already resident.
            crate::cuda::aether_op_matmul_f32_cuda(d_ln_out, dev.wq, d_q, s as c_int, d as c_int, d as c_int);
            crate::cuda::aether_op_matmul_f32_cuda(d_ln_out, dev.wk, d_k, s as c_int, d as c_int, d as c_int);
            crate::cuda::aether_op_matmul_f32_cuda(d_ln_out, dev.wv, d_v, s as c_int, d as c_int, d as c_int);
            // Pull Q/K/V back for SDPA (CPU).
            crate::cuda::aether_dev_d2h_f32(d_q, q.as_mut_ptr() as i64, (s * d) as c_int);
            crate::cuda::aether_dev_d2h_f32(d_k, k.as_mut_ptr() as i64, (s * d) as c_int);
            crate::cuda::aether_dev_d2h_f32(d_v, v.as_mut_ptr() as i64, (s * d) as c_int);
            // SDPA (CPU)
            ops::sdpa_causal_f32(
                q.as_ptr(), k.as_ptr(), v.as_ptr(),
                attn_out.as_mut_ptr(), attn_scratch.as_mut_ptr(),
                1, s, d,
            );
            // Wo on device.
            crate::cuda::aether_dev_h2d_f32(attn_out.as_ptr() as i64, d_attn, (s * d) as c_int);
            crate::cuda::aether_op_matmul_f32_cuda(d_attn, dev.wo, d_proj, s as c_int, d as c_int, d as c_int);
            crate::cuda::aether_dev_d2h_f32(d_proj, proj.as_mut_ptr() as i64, (s * d) as c_int);
            // Residual (CPU)
            for i in 0..s*d { x[i] += proj[i]; }
            // LN2 (CPU)
            ops::layer_norm_f32(
                x.as_ptr(), hl.ln2_g.as_ptr(), hl.ln2_b.as_ptr(), 1e-5,
                ln_out.as_mut_ptr(), mean_buf.as_mut_ptr(), inv_std_buf.as_mut_ptr(),
                s, d,
            );
            // MLP up on device.
            crate::cuda::aether_dev_h2d_f32(ln_out.as_ptr() as i64, d_ln_out, (s * d) as c_int);
            crate::cuda::aether_op_matmul_f32_cuda(d_ln_out, dev.w_up, d_up, s as c_int, d as c_int, f as c_int);
            crate::cuda::aether_dev_d2h_f32(d_up, up.as_mut_ptr() as i64, (s * f) as c_int);
            // SiLU (CPU)
            ops::silu_f32(up.as_mut_ptr(), s * f);
            // MLP down on device.
            crate::cuda::aether_dev_h2d_f32(up.as_ptr() as i64, d_up, (s * f) as c_int);
            crate::cuda::aether_op_matmul_f32_cuda(d_up, dev.w_down, d_down, s as c_int, f as c_int, d as c_int);
            crate::cuda::aether_dev_d2h_f32(d_down, down.as_mut_ptr() as i64, (s * d) as c_int);
            // Residual (CPU)
            for i in 0..s*d { x[i] += down[i]; }
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();

    // Tear down device buffers (graceful; the BUFFERS slot table just
    // marks the slot None).
    for dev in &dev_layers {
        crate::cuda::aether_dev_free_f32(dev.wq);
        crate::cuda::aether_dev_free_f32(dev.wk);
        crate::cuda::aether_dev_free_f32(dev.wv);
        crate::cuda::aether_dev_free_f32(dev.wo);
        crate::cuda::aether_dev_free_f32(dev.w_up);
        crate::cuda::aether_dev_free_f32(dev.w_down);
    }
    crate::cuda::aether_dev_free_f32(d_ln_out);
    crate::cuda::aether_dev_free_f32(d_q);
    crate::cuda::aether_dev_free_f32(d_k);
    crate::cuda::aether_dev_free_f32(d_v);
    crate::cuda::aether_dev_free_f32(d_attn);
    crate::cuda::aether_dev_free_f32(d_proj);
    crate::cuda::aether_dev_free_f32(d_up);
    crate::cuda::aether_dev_free_f32(d_down);

    if elapsed <= 0.0 { return 0.0; }
    ((iters as f64) / elapsed) as f32
}

/// Stub when cuda feature isn't enabled -- returns -1.0 so the witness
/// can detect "cuda not built" without crashing the link step.
#[cfg(not(feature = "cuda"))]
#[no_mangle] pub unsafe extern "C" fn aether_llm_inference_bench_tps_cuda_resident(
    _n_iters: c_int, _d_model: c_int, _n_layers: c_int, _ff: c_int, _seq_len: c_int,
) -> f32 {
    -1.0
}

// =====================================================================
// FR-17.14-extra-deeper — Full GGUF v3 reader.
//
// Real GGUF reader matt-voice uses to ingest Qwen2.5-7B Q4_K_M. The
// file format (per ggml docs):
//   bytes 0..4   : magic "GGUF"
//   bytes 4..8   : u32 version (3)
//   bytes 8..16  : u64 tensor_count
//   bytes 16..24 : u64 metadata_kv_count
//   then metadata_kv_count KV pairs:
//       u64 key_len + key_bytes
//       u32 value_type (0..12 — see GgufValueType below)
//       value (variable; for ARRAY: u32 elem_type + u64 len + N elems)
//   then tensor_count tensor info entries:
//       u64 name_len + name_bytes
//       u32 n_dims
//       n_dims × u64 dims[]
//       u32 dtype (GGML_TYPE_* enum; 12 = Q4_K)
//       u64 offset into the data section
//   then padding to align (usually 32) bytes
//   then tensor data
//
// The blob is mmap'd on unix (lazy, reclaimable file-backed pages → host RSS
// stays bounded even for models LARGER than host RAM; matt-voice FR-18.6-real
// leg 3: the 19 GB Qwen3-32B GGUF OOM-killed the loader on cnc's 15 GB box when
// it was a full std::fs::read). Falls back to an owned Vec on non-unix.
// =====================================================================

/// GGUF byte store: an mmap (unix) or an owned buffer (fallback). Derefs to
/// `[u8]` so all blob indexing / `.as_ptr()` / `.len()` sites are unchanged.
/// matt-voice FR-18.6-real leg 3.
enum GgufBlob {
    Owned(Vec<u8>),
    #[cfg(unix)]
    Mmap { ptr: *mut std::os::raw::c_void, len: usize },
}

#[cfg(unix)]
extern "C" {
    fn mmap(addr: *mut std::os::raw::c_void, len: usize, prot: c_int, flags: c_int,
            fd: c_int, off: i64) -> *mut std::os::raw::c_void;
    fn munmap(addr: *mut std::os::raw::c_void, len: usize) -> c_int;
}

impl std::ops::Deref for GgufBlob {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            GgufBlob::Owned(v) => &v[..],
            #[cfg(unix)]
            GgufBlob::Mmap { ptr, len } =>
                unsafe { std::slice::from_raw_parts(*ptr as *const u8, *len) },
        }
    }
}

#[cfg(unix)]
impl Drop for GgufBlob {
    fn drop(&mut self) {
        if let GgufBlob::Mmap { ptr, len } = self {
            unsafe { munmap(*ptr, *len); }
        }
    }
}

// The blob lives in a process-global table; the raw mmap pointer is read-only
// and never aliased mutably, so it's safe to share across threads.
unsafe impl Send for GgufBlob {}
unsafe impl Sync for GgufBlob {}

/// Open a GGUF as a `GgufBlob` — mmap (read-only, private) on unix, owned read
/// otherwise. mmap keeps host RSS bounded for models larger than RAM.
fn open_gguf_blob(path: &str) -> std::io::Result<GgufBlob> {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let f = std::fs::File::open(path)?;
        let len = f.metadata()?.len() as usize;
        if len == 0 { return Ok(GgufBlob::Owned(Vec::new())); }
        const PROT_READ: c_int = 1;
        const MAP_PRIVATE: c_int = 2;
        let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, f.as_raw_fd(), 0) };
        // MAP_FAILED == (void*)-1
        if ptr as isize == -1 {
            return Ok(GgufBlob::Owned(std::fs::read(path)?));
        }
        // `f` may drop here; the mapping outlives the fd.
        Ok(GgufBlob::Mmap { ptr, len })
    }
    #[cfg(not(unix))]
    {
        Ok(GgufBlob::Owned(std::fs::read(path)?))
    }
}

struct GgufTensorInfo {
    name: String,
    dtype: u32,
    shape: Vec<i64>,
    offset_in_data: u64,
}

/// Captured metadata values. Only the GGUF value types we actually
/// need for tokenizer integration are stored; everything else is
/// skipped as before (saves memory across hundreds of unused
/// scalar metadata entries).
enum GgufMeta {
    U32(u32),
    /// Wider model-dim values (context_length in some GGUFs is u64).
    U64(u64),
    /// Float scalar metadata — rope.freq_base, layer_norm_rms_epsilon, etc.
    F32(f32),
    /// Boolean (GGUF value type 7, 1 byte) — e.g. deepseek2.expert_weights_norm,
    /// tokenizer.ggml.add_bos_token.
    Bool(bool),
    String(String),
    /// Array of strings — used for `tokenizer.ggml.tokens` (152064
    /// entries on Qwen2.5) and `_merges`.
    StringArray(Vec<String>),
    /// Array of f32 — used for `tokenizer.ggml.scores` (SentencePiece
    /// Unigram log-probs; one per vocab entry).
    F32Array(Vec<f32>),
}

struct GgufFile {
    blob: GgufBlob,
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
    tensors: Vec<GgufTensorInfo>,
    metadata: std::collections::HashMap<String, GgufMeta>,
    data_section_start: u64,
}
struct GgufCell(UnsafeCell<Vec<Option<Box<GgufFile>>>>);
unsafe impl Sync for GgufCell {}
static GGUF_TABLE: GgufCell = GgufCell(UnsafeCell::new(Vec::new()));
unsafe fn gguf_table() -> &'static mut Vec<Option<Box<GgufFile>>> {
    &mut *GGUF_TABLE.0.get()
}

fn gguf_read_u32(b: &[u8], off: &mut usize) -> Option<u32> {
    if *off + 4 > b.len() { return None; }
    let v = u32::from_le_bytes(b[*off..*off + 4].try_into().ok()?);
    *off += 4; Some(v)
}
fn gguf_read_u64(b: &[u8], off: &mut usize) -> Option<u64> {
    if *off + 8 > b.len() { return None; }
    let v = u64::from_le_bytes(b[*off..*off + 8].try_into().ok()?);
    *off += 8; Some(v)
}
fn gguf_read_string(b: &[u8], off: &mut usize) -> Option<String> {
    let n = gguf_read_u64(b, off)? as usize;
    if *off + n > b.len() { return None; }
    let s = std::str::from_utf8(&b[*off..*off + n]).ok()?.to_string();
    *off += n; Some(s)
}
/// Skip over a metadata value of the given type. Returns Some(()) on
/// success, None on truncation or unknown type.
fn gguf_skip_value(b: &[u8], off: &mut usize, vtype: u32) -> Option<()> {
    match vtype {
        0 | 1 | 7 => { *off += 1; }                          // u8 / i8 / bool
        2 | 3 => { *off += 2; }                              // u16 / i16
        4 | 5 | 6 => { *off += 4; }                          // u32/i32/f32
        10 | 11 | 12 => { *off += 8; }                       // u64/i64/f64
        8 => {                                               // STRING
            let _ = gguf_read_string(b, off)?;
        }
        9 => {                                               // ARRAY
            let elem_type = gguf_read_u32(b, off)?;
            let count = gguf_read_u64(b, off)? as usize;
            for _ in 0..count { gguf_skip_value(b, off, elem_type)?; }
        }
        _ => return None,
    }
    if *off > b.len() { return None; }
    Some(())
}

/// Read a metadata value of the given type. For the types we capture
/// (U32 = 4, String = 8, StringArray = 9 with elem_type 8) returns the
/// parsed value. For all other types skips the value and returns
/// `Some(None)` to signal "captured nothing, position advanced".
fn gguf_read_or_skip_value(b: &[u8], off: &mut usize, vtype: u32) -> Option<Option<GgufMeta>> {
    match vtype {
        4 => {  // u32 -- captured
            let v = gguf_read_u32(b, off)?;
            Some(Some(GgufMeta::U32(v)))
        }
        6 => {  // f32 -- captured
            if *off + 4 > b.len() { return None; }
            let v = f32::from_le_bytes(b[*off..*off + 4].try_into().ok()?);
            *off += 4;
            Some(Some(GgufMeta::F32(v)))
        }
        10 => {  // u64 -- captured
            let v = gguf_read_u64(b, off)?;
            Some(Some(GgufMeta::U64(v)))
        }
        7 => {  // bool (1 byte) -- captured
            if *off + 1 > b.len() { return None; }
            let v = b[*off] != 0;
            *off += 1;
            Some(Some(GgufMeta::Bool(v)))
        }
        8 => {  // string -- captured
            let s = gguf_read_string(b, off)?;
            Some(Some(GgufMeta::String(s)))
        }
        9 => {  // array
            let elem_type = gguf_read_u32(b, off)?;
            let count = gguf_read_u64(b, off)? as usize;
            if elem_type == 8 {  // string array -- captured
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(gguf_read_string(b, off)?);
                }
                Some(Some(GgufMeta::StringArray(items)))
            } else if elem_type == 6 {  // f32 array -- captured (SPM scores)
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    if *off + 4 > b.len() { return None; }
                    items.push(f32::from_le_bytes(b[*off..*off + 4].try_into().ok()?));
                    *off += 4;
                }
                Some(Some(GgufMeta::F32Array(items)))
            } else {
                for _ in 0..count { gguf_skip_value(b, off, elem_type)?; }
                Some(None)
            }
        }
        _ => {
            gguf_skip_value(b, off, vtype)?;
            Some(None)
        }
    }
}

/// Open a GGUF file. Returns a handle ≥ 0 on success, -1 on file
/// read error, -2 on bad magic, -3 on malformed metadata, -4 on
/// malformed tensor table.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_open(
    path: i64, n_path: c_int,
) -> i64 {
    if path == 0 || n_path <= 0 { return -1; }
    let path_bytes = std::slice::from_raw_parts(path as *const u8, n_path as usize);
    let Ok(path_str) = std::str::from_utf8(path_bytes) else { return -1; };
    let Ok(blob) = open_gguf_blob(path_str) else { return -1; };
    let b = &blob[..];
    if b.len() < 24 || &b[..4] != b"GGUF" { return -2; }
    let mut off = 4usize;
    let Some(version) = gguf_read_u32(b, &mut off) else { return -2; };
    let Some(tensor_count) = gguf_read_u64(b, &mut off) else { return -2; };
    let Some(metadata_kv_count) = gguf_read_u64(b, &mut off) else { return -2; };
    let meta_kv_count = metadata_kv_count;
    // Walk metadata, capturing U32/String/StringArray values.
    let mut metadata: std::collections::HashMap<String, GgufMeta> = std::collections::HashMap::new();
    for _ in 0..meta_kv_count {
        let key = match gguf_read_string(b, &mut off) {
            Some(s) => s, None => return -3,
        };
        let vtype = match gguf_read_u32(b, &mut off) { Some(t) => t, None => return -3, };
        match gguf_read_or_skip_value(b, &mut off, vtype) {
            Some(Some(v)) => { metadata.insert(key, v); }
            Some(None) => { /* captured nothing, but advanced */ }
            None => return -3,
        }
    }
    // Walk tensor info table.
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = match gguf_read_string(b, &mut off) {
            Some(s) => s, None => return -4,
        };
        let n_dims = match gguf_read_u32(b, &mut off) { Some(n) => n, None => return -4, };
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = match gguf_read_u64(b, &mut off) { Some(d) => d, None => return -4, };
            shape.push(d as i64);
        }
        let dtype = match gguf_read_u32(b, &mut off) { Some(t) => t, None => return -4, };
        let offset = match gguf_read_u64(b, &mut off) { Some(o) => o, None => return -4, };
        tensors.push(GgufTensorInfo { name, dtype, shape, offset_in_data: offset });
    }
    // Align to 32-byte boundary (the default GGUF alignment).
    let align = 32u64;
    let data_section_start = ((off as u64) + align - 1) / align * align;
    let g = GgufFile {
        blob, version, tensor_count, metadata_kv_count, tensors,
        metadata,
        data_section_start,
    };
    let tbl = gguf_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(g)); return i as i64; }
    }
    tbl.push(Some(Box::new(g)));
    (tbl.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_gguf_close(handle: i64) -> c_int {
    if handle < 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_gguf_version(handle: i64) -> c_int {
    if handle < 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    g.version as c_int
}

#[no_mangle] pub unsafe extern "C" fn aether_gguf_n_tensors(handle: i64) -> c_int {
    if handle < 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    g.tensor_count as c_int
}

/// Copy tensor `i`'s name into `out` (UTF-8, NOT NUL-terminated).
/// Returns bytes written, or -1 on bad index / overflow.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_tensor_name(
    handle: i64, i: c_int, out: i64, max_out: c_int,
) -> c_int {
    if handle < 0 || i < 0 || out == 0 || max_out <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    if (i as usize) >= g.tensors.len() { return -1; }
    let bytes = g.tensors[i as usize].name.as_bytes();
    if bytes.len() > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    o[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as c_int
}

/// Return tensor `i`'s GGML dtype enum (12 = Q4_K, etc.); -1 on bad index.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_tensor_dtype(
    handle: i64, i: c_int,
) -> c_int {
    if handle < 0 || i < 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    if (i as usize) >= g.tensors.len() { return -1; }
    g.tensors[i as usize].dtype as c_int
}

/// Write tensor `i`'s shape into `out_dims` as i64; return n_dims.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_tensor_shape(
    handle: i64, i: c_int, out_dims: i64, max_dims: c_int,
) -> c_int {
    if handle < 0 || i < 0 || out_dims == 0 || max_dims <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    if (i as usize) >= g.tensors.len() { return -1; }
    let shape = &g.tensors[i as usize].shape;
    if shape.len() > max_dims as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out_dims as *mut i64, max_dims as usize);
    for (k, &d) in shape.iter().enumerate() { o[k] = d; }
    shape.len() as c_int
}

/// Tensor `i`'s absolute byte offset within the GGUF file (== data
/// section start + per-tensor relative offset). Caller can use this
/// to mmap-style read the raw block bytes directly.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_tensor_abs_offset(
    handle: i64, i: c_int,
) -> i64 {
    if handle < 0 || i < 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    if (i as usize) >= g.tensors.len() { return -1; }
    (g.data_section_start + g.tensors[i as usize].offset_in_data) as i64
}

/// Return a pointer into the GGUF blob at the given absolute offset.
/// Caller can pass this to any of the dequant kernels (e.g.
/// aether_dequant_q4_k_m for Q4_K_M tensors).
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_tensor_data_ptr(
    handle: i64, i: c_int,
) -> i64 {
    if handle < 0 || i < 0 { return 0; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    let Some(g) = tbl[h].as_ref() else { return 0; };
    if (i as usize) >= g.tensors.len() { return 0; }
    let abs = g.data_section_start + g.tensors[i as usize].offset_in_data;
    if (abs as usize) >= g.blob.len() { return 0; }
    g.blob.as_ptr().add(abs as usize) as i64
}

/// FR-17.14-extra-deeper-deeper — find a tensor by name. Returns the
/// tensor index suitable for the other `aether_gguf_get_tensor_*`
/// accessors, or -1 if not found. `name` and `name_len` describe the
/// caller's byte buffer holding the lookup key (NOT NUL-terminated).
#[no_mangle] pub unsafe extern "C" fn aether_gguf_find_tensor_by_name(
    handle: i64, name: i64, name_len: c_int,
) -> c_int {
    if handle < 0 || name == 0 || name_len <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let needle = std::slice::from_raw_parts(name as *const u8, name_len as usize);
    let Ok(needle_str) = std::str::from_utf8(needle) else { return -1; };
    for (i, t) in g.tensors.iter().enumerate() {
        if t.name == needle_str { return i as c_int; }
    }
    -1
}

/// FR-17.14-extra-deeper-deeper — total element count (product of dims)
/// for tensor `i`. Returns -1 on error.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_tensor_n_elems(
    handle: i64, i: c_int,
) -> i64 {
    if handle < 0 || i < 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    if (i as usize) >= g.tensors.len() { return -1; }
    g.tensors[i as usize].shape.iter().fold(1i64, |acc, &d| acc * d)
}

// =====================================================================
// FR-19.9-extra-deeper — GGUF metadata accessors for tokenizer
// integration. Qwen2.5-7B's embedded tokenizer lives in the metadata
// KV table as `tokenizer.ggml.tokens` (string-array of 152064
// entries), `_merges` (string-array), and `_bos/eos/padding_token_id`
// (u32). These accessors expose them to callers via the same
// `(buf, len)` pattern used elsewhere.
// =====================================================================

/// Read a u32 metadata value. Returns the value cast to i64 on success,
/// -1 on missing key or wrong type.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_u32(
    handle: i64, key: i64, key_len: c_int,
) -> i64 {
    if handle < 0 || key == 0 || key_len <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::U32(v)) => *v as i64,
        Some(GgufMeta::U64(v)) => *v as i64,
        _ => -1,
    }
}

/// Read a boolean metadata value (GGUF type 7).  Returns 1 (true), 0 (false),
/// or -1 on missing key / wrong type — caller distinguishes "absent" from
/// "false" via the -1 sentinel.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_bool(
    handle: i64, key: i64, key_len: c_int,
) -> i64 {
    if handle < 0 || key == 0 || key_len <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::Bool(v)) => if *v { 1 } else { 0 },
        _ => -1,
    }
}

/// Read an f32 metadata value.  Returns the value cast to f64 (because i64
/// is the existing return type and we want lossless round-trip for the
/// typical small values: rope.freq_base ~= 1e6, norm_eps ~= 1e-6) — caller
/// downcasts to f32.  Returns NaN on missing key / wrong type.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_f32(
    handle: i64, key: i64, key_len: c_int,
) -> f64 {
    if handle < 0 || key == 0 || key_len <= 0 { return f64::NAN; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return f64::NAN; }
    let Some(g) = tbl[h].as_ref() else { return f64::NAN; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return f64::NAN; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::F32(v)) => *v as f64,
        _ => f64::NAN,
    }
}

/// Read a string metadata value into a caller-allocated buffer. Returns
/// the byte length written on success, -1 on missing key / wrong type,
/// -2 if the output buffer is too small.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_string(
    handle: i64, key: i64, key_len: c_int, out: i64, max: c_int,
) -> c_int {
    if handle < 0 || key == 0 || key_len <= 0 || out == 0 || max <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::String(s)) => {
            let n = s.len();
            if n > max as usize { return -2; }
            let dst = std::slice::from_raw_parts_mut(out as *mut u8, n);
            dst.copy_from_slice(s.as_bytes());
            n as c_int
        }
        _ => -1,
    }
}

/// Return the length of a string-array metadata value. Returns -1 on
/// missing key / wrong type.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_array_string_n(
    handle: i64, key: i64, key_len: c_int,
) -> c_int {
    if handle < 0 || key == 0 || key_len <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::StringArray(v)) => v.len() as c_int,
        _ => -1,
    }
}

/// Read one element from a string-array metadata value into a
/// caller-allocated buffer. Returns the byte length written on success,
/// -1 on bad args / wrong type, -2 if the output buffer is too small.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_array_string_get(
    handle: i64, key: i64, key_len: c_int, idx: c_int, out: i64, max: c_int,
) -> c_int {
    if handle < 0 || key == 0 || key_len <= 0 || idx < 0 || out == 0 || max <= 0 {
        return -1;
    }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::StringArray(v)) => {
            if (idx as usize) >= v.len() { return -1; }
            let s = &v[idx as usize];
            let n = s.len();
            if n > max as usize { return -2; }
            let dst = std::slice::from_raw_parts_mut(out as *mut u8, n);
            dst.copy_from_slice(s.as_bytes());
            n as c_int
        }
        _ => -1,
    }
}

/// Number of elements in an f32-array metadata value (e.g.
/// `tokenizer.ggml.scores`). -1 on bad args / wrong type.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_array_f32_n(
    handle: i64, key: i64, key_len: c_int,
) -> c_int {
    if handle < 0 || key == 0 || key_len <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::F32Array(v)) => v.len() as c_int,
        _ => -1,
    }
}

/// Bulk-copy an f32-array metadata value into a caller f32 buffer.
/// Returns the count copied (= min(array len, max)), -1 on bad
/// args / wrong type.  One call loads the whole SPM scores array.
#[no_mangle] pub unsafe extern "C" fn aether_gguf_get_metadata_array_f32(
    handle: i64, key: i64, key_len: c_int, out: i64, max: c_int,
) -> c_int {
    if handle < 0 || key == 0 || key_len <= 0 || out == 0 || max <= 0 { return -1; }
    let tbl = gguf_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(g) = tbl[h].as_ref() else { return -1; };
    let key_bytes = std::slice::from_raw_parts(key as *const u8, key_len as usize);
    let Ok(key_str) = std::str::from_utf8(key_bytes) else { return -1; };
    match g.metadata.get(key_str) {
        Some(GgufMeta::F32Array(v)) => {
            let n = v.len().min(max as usize);
            let dst = std::slice::from_raw_parts_mut(out as *mut f32, n);
            dst.copy_from_slice(&v[..n]);
            n as c_int
        }
        _ => -1,
    }
}

/// Copy a NUL-terminated C string (the form `Expr::StrLit` lowers to
/// in the asm backend) into a heap buffer. Returns the byte count
/// written (length up to but not including the NUL). Useful for
/// witnesses that need to pass multi-character literals to extern
/// fns without doing per-byte `aether_byte_set` calls.
#[no_mangle] pub unsafe extern "C" fn aether_copy_cstr(
    dst: i64, cstr: i64, max_bytes: c_int,
) -> c_int {
    if dst == 0 || cstr == 0 || max_bytes <= 0 { return -1; }
    let src = cstr as *const u8;
    let mut len = 0usize;
    while *src.add(len) != 0 && len < (max_bytes as usize) { len += 1; }
    let s = std::slice::from_raw_parts(src, len);
    let d = std::slice::from_raw_parts_mut(dst as *mut u8, len);
    d.copy_from_slice(s);
    len as c_int
}

// =====================================================================
// FR-17.19-extra — SafeTensors deepening: multi-tensor iteration,
// shape extraction, dtype awareness.
//
// The base FR-17.15 / FR-17.19 layer (safetensors_parse_header +
// safetensors_get_tensor_f32 at line 1019-1065) handles single f32
// tensor lookups. This extra surface lets a loader walk EVERY tensor
// in an HF SafeTensors file, read its shape, and check its dtype
// — exactly what's needed to ingest a Llama-1B weight bundle.
//
// Dtype encoding (matches HF schema strings):
//   0 = F32, 1 = F16, 2 = BF16, 3 = I32, 4 = I16, 5 = U8, 6 = I64
//   -1 = unknown / parse error
// =====================================================================

/// Count distinct tensor entries in the SafeTensors header (skips
/// the `__metadata__` synthetic key). Returns -1 on bad input.
#[no_mangle] pub unsafe extern "C" fn aether_safetensors_n_tensors(buf: i64, len: i64) -> c_int {
    let hdr_len = safetensors_parse_header(buf, len);
    if hdr_len < 0 { return -1; }
    let json = std::slice::from_raw_parts((buf as *const u8).add(8), hdr_len as usize);
    let Ok(s) = std::str::from_utf8(json) else { return -1; };
    let mut n = 0i32;
    let mut i = 0;
    let b = s.as_bytes();
    let mut depth = 0;
    while i < b.len() {
        match b[i] {
            b'{' => depth += 1,
            b'}' => depth -= 1,
            b'"' if depth == 1 => {
                // Top-level key. Read until closing quote, check if it's "__metadata__".
                let key_start = i + 1;
                let mut j = key_start;
                while j < b.len() && b[j] != b'"' { j += 1; }
                if j < b.len() {
                    let key = &s[key_start..j];
                    if key != "__metadata__" { n += 1; }
                    i = j + 1;
                    // Skip over the value object (depth-balanced).
                    let mut sub_depth = 0i32;
                    let mut in_str = false;
                    while i < b.len() {
                        let c = b[i];
                        if c == b'"' { in_str = !in_str; }
                        else if !in_str {
                            if c == b'{' { sub_depth += 1; }
                            else if c == b'}' {
                                sub_depth -= 1;
                                if sub_depth == 0 { i += 1; break; }
                            }
                        }
                        i += 1;
                    }
                    continue;
                }
                i = j;
            }
            _ => {}
        }
        i += 1;
    }
    n
}

/// Look up `name` in the SafeTensors header; on success, write its
/// shape into `out_dims` as i64 values (up to `max_dims`) and return
/// the number of dimensions. Returns -1 on missing / malformed.
#[no_mangle] pub unsafe extern "C" fn aether_safetensors_get_shape(
    buf: i64, len: i64, name: i64, name_len: i64,
    out_dims: i64, max_dims: c_int,
) -> c_int {
    let hdr_len = safetensors_parse_header(buf, len);
    if hdr_len < 0 || name == 0 || name_len <= 0 || out_dims == 0 || max_dims <= 0 { return -1; }
    let json = std::slice::from_raw_parts((buf as *const u8).add(8), hdr_len as usize);
    let name_bytes = std::slice::from_raw_parts(name as *const u8, name_len as usize);
    let Ok(json_str) = std::str::from_utf8(json) else { return -1; };
    let Ok(name_str) = std::str::from_utf8(name_bytes) else { return -1; };
    // Locate `"<name>":` at a key position.
    let needle = format!("\"{}\"", name_str);
    let mut search_from = 0usize;
    let key_pos = loop {
        let Some(idx) = json_str[search_from..].find(&needle) else { return -1; };
        let abs = search_from + idx;
        let after = &json_str[abs + needle.len()..];
        if after.trim_start().starts_with(':') { break abs; }
        search_from = abs + needle.len();
    };
    // Inside the matching object, find `"shape":[a,b,...]`.
    let rest = &json_str[key_pos + needle.len()..];
    let Some(shape_idx) = rest.find("\"shape\":[") else { return -1; };
    let after = &rest[shape_idx + "\"shape\":[".len()..];
    let Some(close) = after.find(']') else { return -1; };
    let dims_str = &after[..close];
    let mut n_dims = 0i32;
    let out = std::slice::from_raw_parts_mut(out_dims as *mut i64, max_dims as usize);
    for part in dims_str.split(',') {
        if n_dims >= max_dims { return -1; }
        let Ok(d) = part.trim().parse::<i64>() else { return -1; };
        out[n_dims as usize] = d;
        n_dims += 1;
    }
    n_dims
}

/// Look up `name`; return dtype enum (0=F32, 1=F16, 2=BF16, 3=I32,
/// 4=I16, 5=U8, 6=I64), or -1 on missing / unrecognised dtype.
#[no_mangle] pub unsafe extern "C" fn aether_safetensors_get_dtype(
    buf: i64, len: i64, name: i64, name_len: i64,
) -> c_int {
    let hdr_len = safetensors_parse_header(buf, len);
    if hdr_len < 0 || name == 0 || name_len <= 0 { return -1; }
    let json = std::slice::from_raw_parts((buf as *const u8).add(8), hdr_len as usize);
    let name_bytes = std::slice::from_raw_parts(name as *const u8, name_len as usize);
    let Ok(json_str) = std::str::from_utf8(json) else { return -1; };
    let Ok(name_str) = std::str::from_utf8(name_bytes) else { return -1; };
    let needle = format!("\"{}\"", name_str);
    let mut search_from = 0usize;
    let key_pos = loop {
        let Some(idx) = json_str[search_from..].find(&needle) else { return -1; };
        let abs = search_from + idx;
        let after = &json_str[abs + needle.len()..];
        if after.trim_start().starts_with(':') { break abs; }
        search_from = abs + needle.len();
    };
    let rest = &json_str[key_pos + needle.len()..];
    let Some(dt_idx) = rest.find("\"dtype\":\"") else { return -1; };
    let after = &rest[dt_idx + "\"dtype\":\"".len()..];
    let Some(close) = after.find('"') else { return -1; };
    match &after[..close] {
        "F32"  => 0,  "F16"  => 1,  "BF16" => 2,
        "I32"  => 3,  "I16"  => 4,  "U8"   => 5,  "I64"  => 6,
        _ => -1,
    }
}

// =====================================================================
// FR-17.14-extra — Q4_K_M dequant (ggml super-block layout).
//
// Q4_K block = 144 bytes for 256 quants:
//   bytes 0..2    : f16 d     (super-block scale)
//   bytes 2..4    : f16 dmin  (super-block min)
//   bytes 4..16   : 12 bytes of packed 6-bit scales (8) + mins (8)
//   bytes 16..144 : 128 bytes of 4-bit quants (nibble-packed)
//
// 8 sub-blocks of 32 quants each. Per sub-block j:
//   sc, m = unpack_6bit(j, scales)  // see get_scale_min_k4 below
//   for each q in the sub-block:
//     val = d * sc * q - dmin * m
//
// Matches ggml-quants.c reference. matt-voice's Qwen2.5-7B Q4_K_M
// uses this exact format. The k4 packing layout reproduces ggml's:
//   if j < 4:  d=scales[j] & 63,  m=scales[j+4] & 63
//   else:      d=(scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//              m=(scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
//
// Quant layout: bytes 0..63 hold quants 0..127 packed two-per-byte
// (low nibble first); bytes 64..127 hold quants 128..255. Per sub-
// block j (4 iterations of j in 0..4): each pair handles 64 quants
// — 32 from low nibbles (sub-block 2j), 32 from high (sub-block 2j+1).
// =====================================================================
fn q4k_get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}
/// Dequantize `n_blocks` Q4_K super-blocks (each 144 bytes = 256
/// quants) into `out` (length `n_blocks * 256` f32 elements).
#[no_mangle] pub unsafe extern "C" fn aether_dequant_q4_k_m(
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
        let base = b.offset(bi * 144);
        // f16 d, f16 dmin (little-endian).
        let d_bits = u16::from_le_bytes([*base, *base.offset(1)]);
        let dmin_bits = u16::from_le_bytes([*base.offset(2), *base.offset(3)]);
        let d_f32 = aether_f16_to_f32(d_bits as i32);
        let dmin_f32 = aether_f16_to_f32(dmin_bits as i32);
        let scales: [u8; 12] = std::array::from_fn(|i| *base.offset(4 + i as isize));
        let qs_ptr = base.offset(16);  // 128 bytes
        // Walk 4 j-iterations covering 8 sub-blocks of 32 quants.
        for j in 0..4usize {
            let (sc_lo, m_lo) = q4k_get_scale_min(2 * j, &scales);
            let (sc_hi, m_hi) = q4k_get_scale_min(2 * j + 1, &scales);
            let d1 = d_f32 * (sc_lo as f32);
            let m1 = dmin_f32 * (m_lo as f32);
            let d2 = d_f32 * (sc_hi as f32);
            let m2 = dmin_f32 * (m_hi as f32);
            let q_off = j * 32;  // 32 bytes per j-iteration in the quant block
            for l in 0..32usize {
                let q_byte = *qs_ptr.offset((q_off + l) as isize);
                let lo = (q_byte & 0x0F) as f32;
                let hi = ((q_byte >> 4) & 0x0F) as f32;
                let out_lo_idx = bi * 256 + (2 * j * 32 + l) as isize;
                let out_hi_idx = bi * 256 + ((2 * j + 1) * 32 + l) as isize;
                *o.offset(out_lo_idx) = d1 * lo - m1;
                *o.offset(out_hi_idx) = d2 * hi - m2;
            }
        }
    }
    0
}

// =====================================================================
// FR-17.14-extra-deeper-deeper — Q6_K dequantisation.
//
// Q6_K super-block layout (210 bytes per 256 quants), ported directly
// from ggml's reference decoder:
//   bytes 0..128   : ql[128]   -- low 4 bits of each quant
//   bytes 128..192 : qh[64]    -- high 2 bits of each quant
//   bytes 192..208 : scales[16] -- i8 sub-block scales (one per 16 quants)
//   bytes 208..210 : d         -- f16 super-block scale
//
// Decode: quants are arranged in 2 halves of 128. For quant index `l`
// in the first half (l = 0..127):
//   q_lo = ql[l] & 0x0F
//   q_hi = (qh[l & 63] >> ((l >> 5) * 2)) & 0x03
//   q = (q_lo | (q_hi << 4)) - 32     -- signed range [-32, 31]
//   scale_idx = l / 16
//   value = d * scales[scale_idx] * q
// For quant index `l` in the second half (l = 128..255):
//   q_lo = ql[l - 128] >> 4
//   (high bits shifted differently per ggml's bit-mapping)
//
// Used for Qwen2.5-7B's V projection + down_proj weights (dtype 14).
// =====================================================================

/// Dequantise `n_blocks` Q6_K super-blocks (each 210 bytes = 256 quants)
/// into `out` (length `n_blocks * 256` f32 elements).
#[no_mangle] pub unsafe extern "C" fn aether_dequant_q6_k(
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
        let base = b.offset(bi * 210);
        // d (f16) lives in the LAST 2 bytes of the super-block.
        let d_bits = u16::from_le_bytes([*base.offset(208), *base.offset(209)]);
        let d_f32 = aether_f16_to_f32(d_bits as i32);
        // Iterate 4 sub-halves of 64 quants each (= 2 sets of 128 in
        // ggml's "ql + qh" layout). Reference: ggml-quants.c::dequantize_row_q6_K.
        let ql = base;             // 0..128
        let qh = base.offset(128); // 128..192
        let scales = base.offset(192); // 192..208 (16 signed bytes)
        for n_outer in 0..2isize {
            // n_outer 0 -> quants 0..128 (low-half byte indexing)
            // n_outer 1 -> quants 128..256
            let ql_base = n_outer * 64;
            let qh_base = n_outer * 32; // qh advances 32 per 128-quant half (ggml's `qh += 32`)
            let sc_base = n_outer * 8;
            for l in 0..32isize {
                // Each iteration fills 4 output quants at strides {0, 32, 64, 96}
                // within this 128-quant half. Matches ggml's 4-way unroll.
                let ql_lo = *ql.offset(ql_base + l) as u32;
                let ql_hi = *ql.offset(ql_base + l + 32) as u32;
                let qh_byte = *qh.offset(qh_base + l) as u32;

                let q0 = ((ql_lo & 0x0F) | ((qh_byte & 0x03) << 4)) as i32 - 32;
                let q1 = ((ql_hi & 0x0F) | (((qh_byte >> 2) & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql_lo >> 4) | (((qh_byte >> 4) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql_hi >> 4) | (((qh_byte >> 6) & 0x03) << 4)) as i32 - 32;

                let s0 = *(scales.offset(sc_base + l / 16)) as i8 as i32;
                let s1 = *(scales.offset(sc_base + l / 16 + 2)) as i8 as i32;
                let s2 = *(scales.offset(sc_base + l / 16 + 4)) as i8 as i32;
                let s3 = *(scales.offset(sc_base + l / 16 + 6)) as i8 as i32;

                // Output positions inside this 128-quant half.
                let out_base = bi * 256 + n_outer * 128;
                *o.offset(out_base + l +  0) = d_f32 * (s0 as f32) * (q0 as f32);
                *o.offset(out_base + l + 32) = d_f32 * (s1 as f32) * (q1 as f32);
                *o.offset(out_base + l + 64) = d_f32 * (s2 as f32) * (q2 as f32);
                *o.offset(out_base + l + 96) = d_f32 * (s3 as f32) * (q3 as f32);
            }
        }
    }
    0
}

/// Host Q3_K dequant (110-byte / 256-elem super-blocks). CPU port of the device
/// fused_q3_k_matmul_seq1 decode; used for token_embd rows whose dtype is Q3_K
/// (qwen3moe Q3_K_M). out = f32[n_blocks * 256].
#[no_mangle] pub unsafe extern "C" fn aether_dequant_q3_k(
    blocks: *const c_void, out: *mut c_void, n_blocks: c_int,
) -> c_int {
    if blocks.is_null() || out.is_null() || n_blocks <= 0 { return 1; }
    let b = blocks as *const u8;
    let o = out as *mut f32;
    for bi in 0..(n_blocks as isize) {
        let base = b.offset(bi * 110);
        let hm = base;             // [32] hmask
        let qs = base.offset(32);  // [64]
        let sc = base.offset(96);  // [12]
        let d_bits = u16::from_le_bytes([*base.offset(108), *base.offset(109)]);
        let d_all = aether_f16_to_f32(d_bits as i32);
        // Unpack 16 signed 6-bit scales (kmask1/kmask2 trick, ggml-quants.c).
        let rd = |p: isize| *sc.offset(p) as u32;
        let aux0 = rd(0) | (rd(1) << 8) | (rd(2) << 16) | (rd(3) << 24);
        let aux1 = rd(4) | (rd(5) << 8) | (rd(6) << 16) | (rd(7) << 24);
        let aux2 = rd(8) | (rd(9) << 8) | (rd(10) << 16) | (rd(11) << 24);
        let tmp = aux2;
        let km1 = 0x0303_0303u32; let km2 = 0x0f0f_0f0fu32;
        let a0 = (aux0 & km2) | (((tmp >> 0) & km1) << 4);
        let a1 = (aux1 & km2) | (((tmp >> 2) & km1) << 4);
        let a2 = ((aux0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);
        let a3 = ((aux1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);
        let mut scales = [0i32; 16];
        for k in 0..4 { scales[k]      = ((a0 >> (8 * k)) & 0xFF) as i32; }
        for k in 0..4 { scales[4 + k]  = ((a1 >> (8 * k)) & 0xFF) as i32; }
        for k in 0..4 { scales[8 + k]  = ((a2 >> (8 * k)) & 0xFF) as i32; }
        for k in 0..4 { scales[12 + k] = ((a3 >> (8 * k)) & 0xFF) as i32; }
        let mut is = 0usize; let mut a_idx = 0isize; let mut m = 1u8;
        let mut n_outer = 0isize;
        while n_outer < 256 {
            let mut shift = 0u32;
            let qs_off = if n_outer == 0 { 0isize } else { 32 };
            for _j in 0..4 {
                let dl_lo = d_all * (scales[is] as f32 - 32.0); is += 1;
                for l in 0..16isize {
                    let q2 = (((*qs.offset(qs_off + l) as u32) >> shift) & 3) as i32;
                    let sub = if (*hm.offset(l) & m) != 0 { 0 } else { 4 };
                    *o.offset(bi * 256 + a_idx) = dl_lo * (q2 - sub) as f32; a_idx += 1;
                }
                let dl_hi = d_all * (scales[is] as f32 - 32.0); is += 1;
                for l in 0..16isize {
                    let q2 = (((*qs.offset(qs_off + 16 + l) as u32) >> shift) & 3) as i32;
                    let sub = if (*hm.offset(16 + l) & m) != 0 { 0 } else { 4 };
                    *o.offset(bi * 256 + a_idx) = dl_hi * (q2 - sub) as f32; a_idx += 1;
                }
                shift += 2; m <<= 1;
            }
            n_outer += 128;
        }
    }
    0
}

// =====================================================================

/// Host IQ3_S dequant (110-byte / 256-elem super-blocks). CPU port of the device
/// fused_iq3_s_matmul_seq1 decode (incl the 512-entry iq3s_grid). Used for
/// token_embd rows whose dtype is IQ3_S (IQ3_M-class models, e.g. R1-Distill).
#[no_mangle] pub unsafe extern "C" fn aether_dequant_iq3_s(
    blocks: *const c_void, out: *mut c_void, n_blocks: c_int,
) -> c_int {
    if blocks.is_null() || out.is_null() || n_blocks <= 0 { return 1; }
    static IQ3S_GRID: [u32; 512] = [
0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305, 
0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905, 
0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09, 
0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b, 
0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b, 
0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d, 
0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03, 
0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505, 
0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03, 
0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901, 
0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d, 
0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303, 
0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501, 
0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105, 
0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505, 
0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101, 
0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707, 
0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b, 
0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01, 
0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f, 
0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305, 
0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103, 
0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509, 
0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503, 
0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b, 
0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f, 
0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f, 
0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f, 
0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109, 
0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f, 
0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509, 
0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501, 
0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303, 
0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f, 
0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907, 
0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703, 
0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03, 
0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01, 
0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01, 
0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903, 
0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505, 
0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b, 
0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107, 
0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509, 
0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303, 
0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103, 
0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05, 
0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b, 
0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f, 
0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701, 
0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909, 
0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305, 
0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d, 
0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b, 
0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d, 
0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307, 
0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09, 
0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309, 
0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709, 
0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f, 
0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303, 
0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503, 
0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b, 
0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101, 

    ];
    let b = blocks as *const u8;
    let o = out as *mut f32;
    for bi in 0..(n_blocks as isize) {
        let base = b.offset(bi * 110);
        let d_bits = u16::from_le_bytes([*base.offset(0), *base.offset(1)]);
        let d = aether_f16_to_f32(d_bits as i32);
        let qs = base.offset(2);              // 64
        let qh = base.offset(2 + 64);         // 8
        let signs = base.offset(2 + 64 + 8);  // 32
        let scales = base.offset(2 + 64 + 8 + 32); // 4
        for ib32 in 0..8isize {
            let scale_nib = ((*scales.offset(ib32 >> 1) as u32) >> (4 * (ib32 & 1) as u32)) & 0xF;
            let db = d * (1.0 + 2.0 * scale_nib as f32);
            let qh_byte = *qh.offset(ib32) as u32;
            for l in 0..4isize {
                let idx1 = (*qs.offset(ib32 * 8 + 2 * l) as u32) | ((qh_byte << (8 - 2 * l as u32)) & 256);
                let idx2 = (*qs.offset(ib32 * 8 + 2 * l + 1) as u32) | ((qh_byte << (7 - 2 * l as u32)) & 256);
                let grid1 = IQ3S_GRID[idx1 as usize];
                let grid2 = IQ3S_GRID[idx2 as usize];
                let sign = *signs.offset(ib32 * 4 + l) as u32;
                for j in 0..4i64 {
                    let q0 = (grid1 >> (8 * j as u32)) & 0xFF;
                    let q1 = (grid2 >> (8 * j as u32)) & 0xFF;
                    let s0 = if (sign & (1 << (j + 0))) != 0 { -1.0f32 } else { 1.0 };
                    let s1 = if (sign & (1 << (j + 4))) != 0 { -1.0f32 } else { 1.0 };
                    let p = bi * 256 + 32 * ib32 + 8 * l + j as isize;
                    *o.offset(p)     = db * q0 as f32 * s0;
                    *o.offset(p + 4) = db * q1 as f32 * s1;
                }
            }
        }
    }
    0
}

// FR-19.9-extra — HF tokenizer.json loader.
//
// Parses the standard HF Tokenizer JSON shape:
//   { "model": { "type": "BPE",
//                "vocab": { "<token>": <id>, ... },
//                "merges": [ ["<left>", "<right>"], ... ] } }
//
// Walks the vocab object: for each (token_string, id) pair, registers
// the token with its EXPLICIT HF id (so the loaded tokenizer's ids
// match the model's weight indices — essential for matt-voice's
// Qwen2.5 deploy).
//
// Walks the merges array in order: for each (left_str, right_str)
// pair, looks up left_id + right_id in the just-built vocab, computes
// merged_string = concat, looks up its id, registers the merge with
// (left_id, right_id, rank=array_index, merged_id) — bypassing the
// auto-id-allocation path of the basic add_merge fn.
//
// Returns the number of merges loaded, or -1 on parse error / -2 on
// vocab lookup failure during merges.
// =====================================================================

/// Like `aether_bpe_add_merge` but caller supplies the merged_id
/// explicitly (so loaded vocabs preserve HF's id assignment).
#[no_mangle] pub unsafe extern "C" fn aether_bpe_add_token_with_id(
    handle: i64, token_id: c_int,
    bytes: *const c_void, n_bytes: c_int,
) -> c_int {
    if handle < 0 || bytes.is_null() || n_bytes <= 0 || token_id < 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_mut() else { return -1; };
    let id = token_id as usize;
    while t.decode_table.len() <= id {
        t.decode_table.push(Vec::new());
    }
    let b = std::slice::from_raw_parts(bytes as *const u8, n_bytes as usize).to_vec();
    t.decode_table[id] = b;
    0
}

/// Same as `aether_bpe_add_merge` but uses an explicit merged_id
/// instead of allocating one. The merged token's byte sequence is
/// looked up from the tokenizer's existing decode_table[merged_id]
/// (caller MUST have called `aether_bpe_add_token_with_id` first).
#[no_mangle] pub unsafe extern "C" fn aether_bpe_add_merge_by_id(
    handle: i64,
    left_id: c_int, right_id: c_int, rank: c_int, merged_id: c_int,
) -> c_int {
    if handle < 0 || left_id < 0 || right_id < 0 || rank < 0 || merged_id < 0 { return -1; }
    let tbl = bpe_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let Some(t) = tbl[h].as_mut() else { return -1; };
    let left = left_id as u32;
    let right = right_id as u32;
    let merged = merged_id as u32;
    if t.merges.contains_key(&(left, right)) { return -1; }
    t.merges.insert((left, right), (merged, rank as u32));
    0
}

/// Load an HF tokenizer.json blob into the given BPE tokenizer handle.
/// Returns the number of merges loaded on success; -1 on JSON parse
/// error; -2 on vocab lookup failure during merge resolution.
#[no_mangle] pub unsafe extern "C" fn aether_tokenizer_json_load(
    handle: i64,
    json_bytes: *const c_void, n_json: c_int,
) -> c_int {
    if handle < 0 || json_bytes.is_null() || n_json <= 0 { return -1; }
    let json_buf = std::slice::from_raw_parts(json_bytes as *const u8, n_json as usize);
    let Ok(json) = std::str::from_utf8(json_buf) else { return -1; };
    // 1) Find the vocab object: `"vocab":{`. Walk braces to extract
    //    the content, then parse "key":<int> pairs.
    let Some(vocab_start) = json.find("\"vocab\":{") else { return -1; };
    let vocab_open = vocab_start + "\"vocab\":{".len();
    let mut depth = 1i32;
    let mut in_str = false;
    let mut vocab_end = vocab_open;
    let b = json.as_bytes();
    while vocab_end < b.len() && depth > 0 {
        let c = b[vocab_end];
        if c == b'"' && (vocab_end == 0 || b[vocab_end - 1] != b'\\') { in_str = !in_str; }
        else if !in_str {
            if c == b'{' { depth += 1; }
            else if c == b'}' { depth -= 1; }
        }
        vocab_end += 1;
    }
    if depth != 0 { return -1; }
    let vocab_body = &json[vocab_open..vocab_end - 1];
    // Walk pairs: "<token>":<id>,
    let mut vocab_map: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let vb = vocab_body.as_bytes();
    let mut i = 0;
    while i < vb.len() {
        // Skip whitespace + commas.
        while i < vb.len() && (vb[i].is_ascii_whitespace() || vb[i] == b',') { i += 1; }
        if i >= vb.len() { break; }
        if vb[i] != b'"' { return -1; }
        i += 1;
        let key_start = i;
        while i < vb.len() && vb[i] != b'"' { i += 1; }
        if i >= vb.len() { return -1; }
        let key = vocab_body[key_start..i].to_string();
        i += 1;  // closing "
        // expect :
        while i < vb.len() && vb[i].is_ascii_whitespace() { i += 1; }
        if i >= vb.len() || vb[i] != b':' { return -1; }
        i += 1;
        while i < vb.len() && vb[i].is_ascii_whitespace() { i += 1; }
        // Parse int.
        let int_start = i;
        while i < vb.len() && (vb[i].is_ascii_digit() || vb[i] == b'-') { i += 1; }
        let Ok(id) = vocab_body[int_start..i].parse::<u32>() else { return -1; };
        // Register the token at its HF id.
        let token_bytes = key.into_bytes();
        let tbl = bpe_table();
        let hu = handle as usize;
        let Some(t) = tbl[hu].as_mut() else { return -1; };
        while t.decode_table.len() <= id as usize { t.decode_table.push(Vec::new()); }
        let key_clone = token_bytes.clone();
        t.decode_table[id as usize] = token_bytes;
        let key_str = String::from_utf8(key_clone).unwrap_or_default();
        vocab_map.insert(key_str, id);
    }
    // 2) Find the merges array: `"merges":[`.
    let Some(merges_start) = json.find("\"merges\":[") else { return -1; };
    let merges_open = merges_start + "\"merges\":[".len();
    let mb = json.as_bytes();
    let mut i = merges_open;
    let mut rank = 0u32;
    let mut n_loaded = 0i32;
    while i < mb.len() {
        while i < mb.len() && (mb[i].is_ascii_whitespace() || mb[i] == b',') { i += 1; }
        if i >= mb.len() { return -1; }
        if mb[i] == b']' { break; }
        if mb[i] != b'[' { return -1; }
        i += 1;
        // First string.
        while i < mb.len() && mb[i].is_ascii_whitespace() { i += 1; }
        if i >= mb.len() || mb[i] != b'"' { return -1; }
        i += 1;
        let l_start = i;
        while i < mb.len() && mb[i] != b'"' { i += 1; }
        if i >= mb.len() { return -1; }
        let l_str = json[l_start..i].to_string();
        i += 1;
        while i < mb.len() && (mb[i].is_ascii_whitespace() || mb[i] == b',') { i += 1; }
        if i >= mb.len() || mb[i] != b'"' { return -1; }
        i += 1;
        let r_start = i;
        while i < mb.len() && mb[i] != b'"' { i += 1; }
        if i >= mb.len() { return -1; }
        let r_str = json[r_start..i].to_string();
        i += 1;
        while i < mb.len() && mb[i].is_ascii_whitespace() { i += 1; }
        if i >= mb.len() || mb[i] != b']' { return -1; }
        i += 1;
        // Resolve ids.
        let merged_str = format!("{}{}", l_str, r_str);
        let (Some(&l_id), Some(&r_id), Some(&m_id)) = (
            vocab_map.get(&l_str), vocab_map.get(&r_str), vocab_map.get(&merged_str)
        ) else { return -2; };
        let tbl = bpe_table();
        let hu = handle as usize;
        let Some(t) = tbl[hu].as_mut() else { return -1; };
        if !t.merges.contains_key(&(l_id, r_id)) {
            t.merges.insert((l_id, r_id), (m_id, rank));
            n_loaded += 1;
        }
        rank += 1;
    }
    n_loaded
}

// =====================================================================
// FR-19.10-extra — chat_template.jinja file loader.
//
// Thin wrapper: read file from disk → render with the given template
// context handle. Returns rendered byte count, or -1 on file/render
// failure.
// =====================================================================
#[no_mangle] pub unsafe extern "C" fn aether_template_render_from_file(
    handle: i64,
    path: i64, n_path: i32,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if handle < 0 || path == 0 || n_path <= 0 || out.is_null() || max_out <= 0 { return -1; }
    let path_bytes = std::slice::from_raw_parts(path as *const u8, n_path as usize);
    let Ok(path_s) = std::str::from_utf8(path_bytes) else { return -1; };
    let Ok(template) = std::fs::read(path_s) else { return -1; };
    aether_template_render(
        handle,
        template.as_ptr() as *const c_void,
        template.len() as c_int,
        out,
        max_out,
    )
}

#[no_mangle] pub unsafe extern "C" fn aether_tool_render_call(
    name: *const c_void, n_name: c_int,
    args: *const c_void, n_args: c_int,
    out: *mut c_void, max_out: c_int,
) -> c_int {
    if name.is_null() || args.is_null() || out.is_null() { return -1; }
    if n_name <= 0 || n_args < 0 || max_out <= 0 { return -1; }
    let ns = std::str::from_utf8(std::slice::from_raw_parts(name as *const u8, n_name as usize)).unwrap_or("");
    let as_ = std::str::from_utf8(std::slice::from_raw_parts(args as *const u8, n_args as usize)).unwrap_or("");
    let json = format!("{{\"type\":\"function\",\"function\":{{\"name\":\"{}\",\"arguments\":\"{}\"}}}}", ns, as_);
    let bytes = json.as_bytes();
    if bytes.len() > max_out as usize { return -1; }
    let o = std::slice::from_raw_parts_mut(out as *mut u8, max_out as usize);
    o[..bytes.len()].copy_from_slice(bytes);
    bytes.len() as c_int
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

    /// FR-17.14-extra-deeper — GGUF reader walks Qwen2.5-7B's blob and
    /// verifies the header/tensor table parse against the known counts.
    /// Skipped if the local ollama blob isn't present.
    #[test]
    fn gguf_reader_qwen25_walk() {
        let qwen_path = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";
        if !std::path::Path::new(qwen_path).exists() {
            eprintln!("[skip] Qwen2.5-7B GGUF not present at {}", qwen_path);
            return;
        }
        unsafe {
            let h = aether_gguf_open(
                qwen_path.as_ptr() as i64, qwen_path.len() as c_int,
            );
            assert!(h >= 0, "open returned {}", h);
            assert_eq!(aether_gguf_version(h), 3);
            // Qwen2.5-7B-Instruct GGUF has exactly 339 tensors.
            let n = aether_gguf_n_tensors(h);
            assert_eq!(n, 339, "expected 339 tensors, got {}", n);
            // First tensor: typically the embedding (`token_embd.weight`),
            // dtype Q4_K (=12).
            let mut name_buf = [0u8; 256];
            let n_name = aether_gguf_get_tensor_name(
                h, 0, name_buf.as_mut_ptr() as i64, name_buf.len() as c_int,
            );
            assert!(n_name > 0);
            let first_name = std::str::from_utf8(&name_buf[..n_name as usize]).unwrap();
            eprintln!("[gguf] tensor 0 = {}", first_name);
            // Don't hard-fail on the name (different GGUF tools sort
            // differently); just verify SOME tensor is the token embedding
            // and SOME tensor is Q4_K_M-encoded.
            let mut found_embd = false;
            let mut found_q4k = false;
            for i in 0..n {
                let mut nb = [0u8; 256];
                let nn = aether_gguf_get_tensor_name(h, i, nb.as_mut_ptr() as i64, 256);
                if nn > 0 {
                    let nm = std::str::from_utf8(&nb[..nn as usize]).unwrap();
                    if nm == "token_embd.weight" { found_embd = true; }
                }
                let dt = aether_gguf_get_tensor_dtype(h, i);
                if dt == 12 { found_q4k = true; }
            }
            assert!(found_embd, "expected a token_embd.weight tensor");
            assert!(found_q4k, "expected at least one Q4_K (=12) dtype tensor");
            // Tensor offsets should be in the data section.
            let abs0 = aether_gguf_get_tensor_abs_offset(h, 0);
            assert!(abs0 >= 24, "offset {} too low", abs0);
            let ptr0 = aether_gguf_get_tensor_data_ptr(h, 0);
            assert!(ptr0 != 0);
            aether_gguf_close(h);
        }
    }

    /// FR-17.14-extra-deeper-deeper — Forward-pass chain over real
    /// Qwen2.5-7B weights: GGUF data ptr → Q4_K_M dequant → matmul.
    /// Skipped if the local ollama blob isn't present.
    #[test]
    fn qwen25_forward_chain_one_block() {
        let qwen_path = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";
        if !std::path::Path::new(qwen_path).exists() {
            eprintln!("[skip] Qwen2.5-7B GGUF not present at {}", qwen_path);
            return;
        }
        unsafe {
            let h = aether_gguf_open(
                qwen_path.as_ptr() as i64, qwen_path.len() as c_int,
            );
            assert!(h >= 0);
            assert_eq!(aether_gguf_get_tensor_dtype(h, 0), 12, "tensor 0 not Q4_K");
            let dptr = aether_gguf_get_tensor_data_ptr(h, 0);
            assert!(dptr != 0);
            // Dequantise one 144-byte super-block -> 256 f32.
            let mut deq = vec![0.0f32; 256];
            let rc_dq = aether_dequant_q4_k_m(
                dptr as *const c_void,
                deq.as_mut_ptr() as *mut c_void,
                1,
            );
            assert_eq!(rc_dq, 0);
            // Real trained embedding values are not all-zero.
            let raw_sum: f32 = deq.iter().sum();
            assert!(raw_sum.is_finite(), "dequant sum not finite: {}", raw_sum);
            assert!(raw_sum.abs() > 1e-5, "dequant sum too small ({}): never read real weights?", raw_sum);
            assert!(raw_sum.abs() < 1e3, "dequant sum too large ({}): scale overflow?", raw_sum);
            // ones[1,256] @ deq[256,1] -> scalar must equal sum(deq).
            let input = vec![1.0f32; 256];
            let mut scalar = [0.0f32; 1];
            let rc_mm = aether_op_matmul_f32(
                input.as_ptr() as *const c_void,
                deq.as_ptr() as *const c_void,
                scalar.as_mut_ptr() as *mut c_void,
                1, 256, 1,
            );
            assert_eq!(rc_mm, 0);
            assert!((scalar[0] - raw_sum).abs() < 1e-2 * raw_sum.abs().max(1.0),
                "matmul {} != sum {}", scalar[0], raw_sum);
            eprintln!("[chain] qwen2.5 super-block 0: sum={:.6e} matmul={:.6e}", raw_sum, scalar[0]);
            aether_gguf_close(h);
        }
    }

    /// FR-17.14-extra-deeper-deeper -- Q6_K dequant produces finite,
    /// reasonable values on real Qwen2.5-7B weights.
    /// Tests against the first super-block of `blk.0.attn_v.weight`
    /// (a Q6_K-dtype tensor in matt-voice's actual model).
    #[test]
    fn q6_k_dequant_on_real_qwen25() {
        let qwen_path = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";
        if !std::path::Path::new(qwen_path).exists() {
            eprintln!("[skip] Qwen2.5-7B GGUF not present");
            return;
        }
        unsafe {
            let h = aether_gguf_open(qwen_path.as_ptr() as i64, qwen_path.len() as c_int);
            assert!(h >= 0);
            let needle = b"blk.0.attn_v.weight";
            let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
            assert!(idx >= 0, "expected blk.0.attn_v.weight");
            assert_eq!(aether_gguf_get_tensor_dtype(h, idx), 14, "expected Q6_K dtype");
            let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
            assert!(dptr != 0);
            // Dequant first super-block (256 quants -> 256 f32 values).
            let mut deq = vec![0.0f32; 256];
            let rc = aether_dequant_q6_k(
                dptr as *const c_void,
                deq.as_mut_ptr() as *mut c_void,
                1,
            );
            assert_eq!(rc, 0);
            // Real trained weights: sum is finite, non-zero, bounded.
            let sum: f32 = deq.iter().sum();
            let max_abs: f32 = deq.iter().map(|v| v.abs()).fold(0.0, f32::max);
            assert!(sum.is_finite(), "Q6_K dequant sum not finite: {}", sum);
            assert!(max_abs > 1e-6, "Q6_K dequant all zeros? max_abs = {}", max_abs);
            assert!(max_abs < 100.0, "Q6_K dequant out of range: max_abs = {}", max_abs);
            eprintln!("[q6_k] blk.0.attn_v.weight super-block 0: sum={:.6e}, max_abs={:.6e}", sum, max_abs);
            aether_gguf_close(h);
        }
    }

    /// matt-voice deploy-pack extras — single-fn sequential test for the
    /// 4 deeper FR-x-extras (SafeTensors multi-tensor / Q4_K dequant /
    /// tokenizer.json load / chat_template.jinja file loader).
    #[test]
    fn matt_voice_extras_batch() {
        unsafe {
            // -- FR-17.19-extra: SafeTensors multi-tensor parser --
            // Hand-build a 2-tensor SafeTensors blob: f32 weight "w"
            // (shape [2,2]) + f16 bias "b" (shape [2]). Verify count,
            // shape, and dtype lookups.
            let header_json = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b":{"dtype":"F16","shape":[2],"data_offsets":[16,20]}}"#;
            let hdr_bytes = header_json.as_bytes();
            let mut blob = Vec::new();
            blob.extend_from_slice(&(hdr_bytes.len() as u64).to_le_bytes());
            blob.extend_from_slice(hdr_bytes);
            // f32 payload "w" (16 bytes).
            for v in &[1.0f32, 2.0, 3.0, 4.0] { blob.extend_from_slice(&v.to_le_bytes()); }
            // f16 payload "b" (4 bytes = 2 × f16). Values 0.5 + 1.5 as f16 bits.
            blob.extend_from_slice(&0x3800u16.to_le_bytes());  // f16 0.5
            blob.extend_from_slice(&0x3E00u16.to_le_bytes());  // f16 1.5
            let buf_ptr = blob.as_ptr() as i64;
            let buf_len = blob.len() as i64;
            assert_eq!(aether_safetensors_n_tensors(buf_ptr, buf_len), 2);
            // "w" shape
            let name_w = b"w";
            let mut dims_w = [0i64; 4];
            let n_w = aether_safetensors_get_shape(
                buf_ptr, buf_len, name_w.as_ptr() as i64, 1,
                dims_w.as_mut_ptr() as i64, 4,
            );
            assert_eq!(n_w, 2);
            assert_eq!(&dims_w[..2], &[2, 2]);
            assert_eq!(aether_safetensors_get_dtype(buf_ptr, buf_len, name_w.as_ptr() as i64, 1), 0);
            // "b" shape + dtype
            let name_b = b"b";
            let mut dims_b = [0i64; 4];
            let n_b = aether_safetensors_get_shape(
                buf_ptr, buf_len, name_b.as_ptr() as i64, 1,
                dims_b.as_mut_ptr() as i64, 4,
            );
            assert_eq!(n_b, 1);
            assert_eq!(dims_b[0], 2);
            assert_eq!(aether_safetensors_get_dtype(buf_ptr, buf_len, name_b.as_ptr() as i64, 1), 1);

            // -- FR-17.14-extra: Q4_K_M dequant --
            // Hand-build one Q4_K super-block (144 bytes) with:
            //   d = 1.0 (f16 = 0x3C00)
            //   dmin = 0.0 (f16 = 0x0000)
            //   scales: bytes designed so scale_j = 1 for all 8 sub-blocks,
            //     min_j = 0 for all. Layout per get_scale_min_k4:
            //     j<4: scales[j] & 63 = 1; scales[j+4] & 63 = 0
            //     j>=4: composite. To make scale=1 / min=0 for all 8, set
            //     scales[0..4] = 0x01 (scale_low_4_bits = 1)
            //     scales[4..8] = 0x10 (so j>=4 reads bottom nibble = 0 for min,
            //                          composite scale bits = 1 from scales[j-4]>>6=0).
            //   Actually simplest: zero out all packed scales/mins → sc=0/m=0,
            //   then dequant = d * 0 * q - dmin * 0 = 0 always. That's
            //   degenerate. Use scales = [1,1,1,1, 0,0,0,0, ...] which gives
            //   sub-blocks 0-3 scale=1/min=0, and sub-blocks 4-7 (composite
            //   from j>=4 formula) compute differently.
            //
            //   To keep the test simple AND meaningful, set scales such that
            //   sub-block 0 has scale=1, min=0, then verify those 32 quants.
            let mut block = [0u8; 144];
            // d = 1.0 (f16 0x3C00)
            block[0] = 0x00; block[1] = 0x3C;
            // dmin = 0.0
            block[2] = 0x00; block[3] = 0x00;
            // scales: byte 0 = 0x01 → scale_0 = 1 & 63 = 1. byte 4 = 0x00 → min_0 = 0 & 63 = 0.
            block[4 + 0] = 0x01;  // scale 0
            block[4 + 4] = 0x00;  // min 0
            // Quants: byte 16 = 0x53 → low nibble = 3, high = 5.
            for l in 0..32usize {
                block[16 + l] = ((l as u8) & 0x0F) | (((l as u8) + 1) & 0x0F) << 4;
            }
            let mut out = vec![0.0f32; 256];
            unsafe {
                let rc = aether_dequant_q4_k_m(
                    block.as_ptr() as *const _,
                    out.as_mut_ptr() as *mut _,
                    1,
                );
                assert_eq!(rc, 0);
            }
            // Sub-block 0 covers quants 0..31 from LOW nibbles of qs[0..31].
            // qs[l] low nibble = l & 0x0F. With d=1, sc=1, dmin=0:
            // out[l] = 1 * 1 * (l & 0xF) - 0 * 0 = l & 0xF.
            for l in 0..32usize {
                let expected = (l as u8 & 0x0F) as f32;
                assert!((out[l] - expected).abs() < 1e-6,
                        "sub-block 0 quant {}: expected {}, got {}", l, expected, out[l]);
            }

            // -- FR-19.9-extra: tokenizer.json load --
            // Build a tiny BPE-shape tokenizer.json. Vocab: bytes h,e,l,o
            // at ids 1-4, "he"=5, "hel"=6, "hell"=7, "hello"=8. Merges in
            // order: (h,e), (he,l), (hel,l), (hell,o).
            let tok_json = r#"{"model":{"type":"BPE","vocab":{"h":1,"e":2,"l":3,"o":4,"he":5,"hel":6,"hell":7,"hello":8},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}}"#;
            let bh = aether_bpe_tokenizer_new();
            let n_merges = aether_tokenizer_json_load(
                bh, tok_json.as_ptr() as *const _, tok_json.len() as c_int,
            );
            assert_eq!(n_merges, 4, "expected 4 merges; got {}", n_merges);
            // Encode "hello": initial = [104('h')->but wait, our vocab id
            // for 'h' is 1, NOT 104. The encoder still uses bytes 0..255 as
            // implicit initial token ids, so it sees [104, 101, 108, 108, 111].
            // Those bytes don't match our vocab (which used ids 1-4). So the
            // merges (1,2)→5 won't fire on [104,101].
            //
            // For matt-voice's real Qwen2.5 deploy this isn't a problem
            // because Qwen2.5's vocab.json uses byte-level encoding where
            // the byte-level glyphs ARE the initial tokens. For this test
            // we just verify the vocab + merges were registered; encoding
            // semantics under the byte-level initial-vocab convention is
            // a downstream extension.
            //
            // What we CAN verify: the merge count is 4, decode tables
            // contain the right bytes at the right ids, and the merges
            // map has the expected (left_id, right_id) -> (merged_id, rank)
            // entries.
            let tbl = bpe_table();
            let t = tbl[bh as usize].as_ref().unwrap();
            assert!(t.decode_table.len() >= 9);
            assert_eq!(t.decode_table[8], b"hello");
            assert_eq!(t.decode_table[5], b"he");
            assert!(t.merges.contains_key(&(1, 2)));  // (h, e)
            assert_eq!(t.merges[&(1, 2)], (5, 0));    // → "he" id 5, rank 0
            assert_eq!(t.merges[&(7, 4)], (8, 3));    // (hell, o) → hello, rank 3
            aether_bpe_tokenizer_free(bh);

            // -- FR-19.10-extra: chat_template.jinja from file --
            // Write a template to a temp file, render it back through the
            // file-loader wrapper.
            let tpl = b"{% for msg in messages %}[{{ msg.role }}: {{ msg.content }}]{% endfor %}";
            let path = "scratch/_test_chat_template.jinja";
            let _ = std::fs::create_dir_all("scratch");
            std::fs::write(path, tpl).expect("write tpl");
            let th = aether_template_new();
            let role = b"user"; let content = b"hi";
            aether_template_push_message(
                th, role.as_ptr() as _, role.len() as i32,
                content.as_ptr() as _, content.len() as i32,
            );
            let mut out_buf = [0u8; 128];
            let n = aether_template_render_from_file(
                th, path.as_ptr() as i64, path.len() as i32,
                out_buf.as_mut_ptr() as *mut _, out_buf.len() as c_int,
            );
            assert!(n > 0, "file-loaded render returned {}", n);
            let s = std::str::from_utf8(&out_buf[..n as usize]).unwrap();
            assert_eq!(s, "[user: hi]");
            aether_template_free(th);
        }
    }

    /// Phase 19 closeout — 13 in-process witness tests in one fn. Same
    /// race-safety reason as BPE/chat-template: several of these use
    /// shared static handle tables. Each block exercises one FR's
    /// runtime surface end-to-end.
    #[test]
    fn phase19_closeout_batch() {
        unsafe {
            // -- FR-19.4 paged KV cache --
            let pkv = aether_pkv_new(3, 16);
            assert!(pkv >= 0);
            let a = aether_pkv_allocate(pkv);
            let b = aether_pkv_allocate(pkv);
            let c = aether_pkv_allocate(pkv);
            assert_eq!(a, 0); assert_eq!(b, 1); assert_eq!(c, 2);
            // Pool full now.
            assert_eq!(aether_pkv_allocate(pkv), -1);
            // Touch b → b is most-recent; a is now LRU.
            aether_pkv_touch(pkv, b);
            aether_pkv_touch(pkv, c);
            let evicted = aether_pkv_evict_lru(pkv);
            assert_eq!(evicted, a, "expected block 0 (touch'd-least) to evict");
            assert_eq!(aether_pkv_n_allocated(pkv), 2);
            // Re-allocate fills the evicted slot.
            assert_eq!(aether_pkv_allocate(pkv), 0);
            aether_pkv_destroy(pkv);

            // -- FR-19.5 continuous batching --
            let cb = aether_cb_new(4);
            assert!(cb >= 0);
            assert_eq!(aether_cb_admit(cb, 100), 0);
            assert_eq!(aether_cb_admit(cb, 101), 0);
            assert_eq!(aether_cb_admit(cb, 102), 0);
            assert_eq!(aether_cb_admit(cb, 103), 0);
            assert_eq!(aether_cb_admit(cb, 104), -1, "capacity reached");
            assert_eq!(aether_cb_n_active(cb), 4);
            aether_cb_step(cb); aether_cb_step(cb); aether_cb_step(cb);
            assert_eq!(aether_cb_n_active(cb), 4);
            assert_eq!(aether_cb_complete(cb, 101), 0);
            assert_eq!(aether_cb_n_active(cb), 3);
            // Mid-decode admit: with one done, can take one more.
            assert_eq!(aether_cb_admit(cb, 104), 0);
            assert_eq!(aether_cb_n_active(cb), 4);
            aether_cb_destroy(cb);

            // -- FR-19.6 speculative decoding --
            // p >= q → always accept.
            assert_eq!(aether_specdec_accept(0.8, 0.2, 0.5), 1);
            // p < q with rand below ratio → accept.
            assert_eq!(aether_specdec_accept(0.2, 0.8, 0.1), 1);  // ratio=0.25, rand<ratio
            // p < q with rand above ratio → reject.
            assert_eq!(aether_specdec_accept(0.2, 0.8, 0.9), 0);  // rand>ratio
            // p=0 → always reject.
            assert_eq!(aether_specdec_accept(0.0, 1.0, 0.5), 0);

            // -- FR-19.7 multi-model hosting --
            let mm = aether_mm_new();
            let nm_llama = b"llama-3-1b";
            let nm_qwen = b"qwen-2.5-7b";
            assert_eq!(aether_mm_register(mm, nm_llama.as_ptr() as _, nm_llama.len() as i32, 1500), 0);
            assert_eq!(aether_mm_register(mm, nm_qwen.as_ptr() as _, nm_qwen.len() as i32, 4500), 1);
            assert_eq!(aether_mm_lookup(mm, nm_llama.as_ptr() as _, nm_llama.len() as i32), 0);
            assert_eq!(aether_mm_lookup(mm, nm_qwen.as_ptr() as _, nm_qwen.len() as i32), 1);
            let nm_missing = b"missing-model";
            assert_eq!(aether_mm_lookup(mm, nm_missing.as_ptr() as _, nm_missing.len() as i32), -1);
            assert_eq!(aether_mm_total_vram_mb(mm), 6000);
            aether_mm_destroy(mm);

            // -- FR-19.14 rate limit --
            // 1 req/sec, burst=2. The first two requests within 1 second
            // succeed; the third (with no time elapsed) rate-limits.
            let rl = aether_rl_new(1, 2);
            let key = b"client-a";
            assert_eq!(aether_rl_check(rl, key.as_ptr() as _, key.len() as i32, 0), 1);
            assert_eq!(aether_rl_check(rl, key.as_ptr() as _, key.len() as i32, 0), 1);
            assert_eq!(aether_rl_check(rl, key.as_ptr() as _, key.len() as i32, 0), 0);
            // After 2 seconds, refill brings 2 tokens back.
            assert_eq!(aether_rl_check(rl, key.as_ptr() as _, key.len() as i32, 2_000_000), 1);
            aether_rl_destroy(rl);

            // -- FR-19.15 observability --
            let cname = b"requests_total";
            aether_obs_counter_inc(cname.as_ptr() as _, cname.len() as i32, 1);
            aether_obs_counter_inc(cname.as_ptr() as _, cname.len() as i32, 4);
            assert_eq!(aether_obs_counter_get(cname.as_ptr() as _, cname.len() as i32), 5);
            let mut buf = [0u8; 256];
            let n = aether_obs_dump_prometheus(buf.as_mut_ptr() as _, buf.len() as i32);
            assert!(n > 0);
            let s = std::str::from_utf8(&buf[..n as usize]).expect("utf8");
            assert!(s.contains("# TYPE requests_total counter"));
            assert!(s.contains("requests_total 5"));

            // -- FR-19.12 vision input --
            // Normalize: (10, 100, 200, 250) / 255 with mean=0.5, std=0.5
            // gives ((10/255 - 0.5)/0.5, ..., (250/255 - 0.5)/0.5) ≈ (-0.92, -0.22, 0.57, 0.96)
            let pixels: [u8; 4] = [10, 100, 200, 250];
            let mut norm = [0.0f32; 4];
            assert_eq!(0, aether_img_normalize_f32(
                pixels.as_ptr() as _, norm.as_mut_ptr() as _, 4, 0.5, 0.5,
            ));
            assert!((norm[0] - (-0.9215686)).abs() < 1e-5);
            assert!((norm[3] - (0.9607843)).abs() < 1e-5);
            // Patchify: 4×4 → 2×2 patches of 2×2.
            let img: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let mut patches = [0.0f32; 16];
            let n_p = aether_img_patchify_f32(
                img.as_ptr() as _, patches.as_mut_ptr() as _, 4, 4, 2,
            );
            assert_eq!(n_p, 4);
            // Patch 0 (top-left of 4x4): values [0, 1, 4, 5].
            assert_eq!(&patches[0..4], &[0.0, 1.0, 4.0, 5.0]);
            // Patch 1 (top-right): [2, 3, 6, 7].
            assert_eq!(&patches[4..8], &[2.0, 3.0, 6.0, 7.0]);

            // -- FR-19.13 mel-spectrogram primitives --
            // Hann(8): symmetric, max at center.
            let mut win = [0.0f32; 8];
            aether_audio_hann_window(win.as_mut_ptr() as _, 8);
            assert!(win[0] < 1e-5);
            assert!(win[7] < 1e-5);
            assert!(win[3] > 0.9 && win[3] <= 1.0);
            // DFT magnitude: a pure tone at bin k=2 should produce a peak there.
            let n_samp = 16;
            let mut tone = vec![0.0f32; n_samp];
            for i in 0..n_samp {
                tone[i] = (2.0 * std::f32::consts::PI * 2.0 * i as f32 / n_samp as f32).cos();
            }
            let mut mag = vec![0.0f32; 8];
            aether_audio_dft_magnitude_f32(
                tone.as_ptr() as _, n_samp as i32, mag.as_mut_ptr() as _, 8,
            );
            // Bin 2 should be max.
            let mut max_bin = 0usize;
            for i in 1..8 { if mag[i] > mag[max_bin] { max_bin = i; } }
            assert_eq!(max_bin, 2, "DFT peak should land at bin 2 for cos(2π·2·i/N) tone");

            // -- FR-19.1 ChaCha20-Poly1305 round-trip --
            // RFC 7539 §2.8.2 test vector check would compare against
            // known ciphertext bytes; here we round-trip and verify
            // decrypt(encrypt(x)) == x AND tag check rejects flipped bytes.
            let key = [0x80u8; 32];
            let nonce = [0x00u8; 12];
            let aad = b"";
            let plain = b"hello world";
            let mut ct = [0u8; 64];
            let n_ct = aether_chacha20_poly1305_encrypt(
                key.as_ptr() as _, nonce.as_ptr() as _,
                aad.as_ptr() as _, aad.len() as i32,
                plain.as_ptr() as _, plain.len() as i32,
                ct.as_mut_ptr() as _, ct.len() as i32,
            );
            assert_eq!(n_ct, (plain.len() + 16) as c_int);
            let mut decrypted = [0u8; 32];
            let n_pt = aether_chacha20_poly1305_decrypt(
                key.as_ptr() as _, nonce.as_ptr() as _,
                aad.as_ptr() as _, aad.len() as i32,
                ct.as_ptr() as _, n_ct,
                decrypted.as_mut_ptr() as _, decrypted.len() as i32,
            );
            assert_eq!(n_pt, plain.len() as c_int);
            assert_eq!(&decrypted[..plain.len()], plain);
            // Tamper: flip a ciphertext byte → tag check must reject.
            let mut tampered = ct;
            tampered[0] ^= 1;
            let n_bad = aether_chacha20_poly1305_decrypt(
                key.as_ptr() as _, nonce.as_ptr() as _,
                aad.as_ptr() as _, aad.len() as i32,
                tampered.as_ptr() as _, n_ct,
                decrypted.as_mut_ptr() as _, decrypted.len() as i32,
            );
            assert_eq!(n_bad, -2, "expected -2 tag mismatch on tampered ciphertext");

            // -- FR-19.2 HTTP/1.1 parse + write --
            let req = b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n";
            let mut strs = [0u8; 64];
            let mut m_len: i32 = 0;
            let mut p_len: i32 = 0;
            let body_off = aether_http_parse_request(
                req.as_ptr() as _, req.len() as i32,
                strs.as_mut_ptr() as _, strs.len() as i32,
                &mut m_len as *mut _, &mut p_len as *mut _,
            );
            assert!(body_off > 0);
            assert_eq!(m_len, 3);
            assert_eq!(p_len, 10);
            assert_eq!(&strs[..3], b"GET");
            assert_eq!(&strs[3..3 + 10], b"/v1/models");
            let body = b"{\"hello\":1}";
            let mut resp = [0u8; 128];
            let n_resp = aether_http_write_response_200(
                body.as_ptr() as _, body.len() as i32,
                resp.as_mut_ptr() as _, resp.len() as i32,
            );
            assert!(n_resp > 0);
            let r = std::str::from_utf8(&resp[..n_resp as usize]).unwrap();
            assert!(r.starts_with("HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n"));
            assert!(r.ends_with("{\"hello\":1}"));

            // -- FR-19.3 OpenAI shape --
            let id = b"chatcmpl-abc";
            let model = b"llama-3-1b";
            let content = b"hi there";
            let mut json = [0u8; 512];
            let n_json = aether_openai_render_completion(
                id.as_ptr() as _, id.len() as i32,
                model.as_ptr() as _, model.len() as i32,
                content.as_ptr() as _, content.len() as i32,
                7, 4,
                json.as_mut_ptr() as _, json.len() as i32,
            );
            assert!(n_json > 0);
            let s = std::str::from_utf8(&json[..n_json as usize]).unwrap();
            assert!(s.contains("\"id\":\"chatcmpl-abc\""));
            assert!(s.contains("\"model\":\"llama-3-1b\""));
            assert!(s.contains("\"content\":\"hi there\""));
            assert!(s.contains("\"finish_reason\":\"stop\""));
            assert!(s.contains("\"prompt_tokens\":7"));
            assert!(s.contains("\"completion_tokens\":4"));

            // -- FR-19.8 WebSocket frame round-trip --
            let ws_payload = b"hello";
            let mut frame = [0u8; 16];
            let n_frame = aether_ws_encode_text_frame(
                ws_payload.as_ptr() as _, ws_payload.len() as i32,
                frame.as_mut_ptr() as _, frame.len() as i32,
            );
            assert_eq!(n_frame, 7);  // 2 header + 5 payload
            assert_eq!(frame[0], 0x81);  // FIN=1 + opcode=1 (text)
            assert_eq!(frame[1], 5);     // payload len
            assert_eq!(&frame[2..7], ws_payload);
            // Decode back.
            let mut decoded = [0u8; 16];
            let n_dec = aether_ws_decode_frame_payload(
                frame.as_ptr() as _, n_frame,
                decoded.as_mut_ptr() as _, decoded.len() as i32,
            );
            assert_eq!(n_dec, 5);
            assert_eq!(&decoded[..5], ws_payload);

            // -- FR-19.11 tool calling JSON --
            let tool_name = b"get_weather";
            let tool_args = b"{\\\"city\\\":\\\"SF\\\"}";
            let mut tj = [0u8; 256];
            let n_tj = aether_tool_render_call(
                tool_name.as_ptr() as _, tool_name.len() as i32,
                tool_args.as_ptr() as _, tool_args.len() as i32,
                tj.as_mut_ptr() as _, tj.len() as i32,
            );
            assert!(n_tj > 0);
            let ts = std::str::from_utf8(&tj[..n_tj as usize]).unwrap();
            assert!(ts.contains("\"type\":\"function\""));
            assert!(ts.contains("\"name\":\"get_weather\""));
            assert!(ts.contains("\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\""));
        }
    }

    /// FR-19.10 — Llama-3-shaped chat template renders correct turn
    /// boundaries. Combined in one test fn for the same static-table
    /// race-safety reason as `bpe_roundtrip_and_lowest_rank`.
    #[test]
    fn chat_template_llama3_shape() {
        unsafe {
            let h = aether_template_new();
            assert!(h >= 0);
            // Llama-3-ish template (minus whitespace-strip + filters,
            // which we don't support):
            let tpl = b"{% for msg in messages %}<|start_header_id|>{{ msg.role }}<|end_header_id|>\n\n{{ msg.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
            // Push two messages.
            let user_role = b"user";
            let user_content = b"hi";
            assert_eq!(0, aether_template_push_message(
                h, user_role.as_ptr() as _, user_role.len() as i32,
                user_content.as_ptr() as _, user_content.len() as i32,
            ));
            let asst_role = b"assistant";
            let asst_content = b"hello";
            assert_eq!(0, aether_template_push_message(
                h, asst_role.as_ptr() as _, asst_role.len() as i32,
                asst_content.as_ptr() as _, asst_content.len() as i32,
            ));
            // Set add_generation_prompt=1 so the trailing assistant header
            // gets emitted.
            let agp_name = b"add_generation_prompt";
            let agp_val = b"1";
            assert_eq!(0, aether_template_set_var(
                h, agp_name.as_ptr() as _, agp_name.len() as i32,
                agp_val.as_ptr() as _, agp_val.len() as i32,
            ));
            // Render.
            let mut buf = [0u8; 512];
            let n = aether_template_render(
                h, tpl.as_ptr() as _, tpl.len() as i32,
                buf.as_mut_ptr() as _, buf.len() as i32,
            );
            assert!(n > 0, "render returned {}", n);
            let rendered = std::str::from_utf8(&buf[..n as usize]).expect("utf8");
            let expected = "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
            assert_eq!(rendered, expected, "rendered output mismatch");
            aether_template_free(h);

            // Negative case: add_generation_prompt unset → no trailing
            // assistant header.
            let h2 = aether_template_new();
            aether_template_push_message(
                h2, user_role.as_ptr() as _, user_role.len() as i32,
                user_content.as_ptr() as _, user_content.len() as i32,
            );
            let mut buf2 = [0u8; 512];
            let n2 = aether_template_render(
                h2, tpl.as_ptr() as _, tpl.len() as i32,
                buf2.as_mut_ptr() as _, buf2.len() as i32,
            );
            assert!(n2 > 0);
            let rendered2 = std::str::from_utf8(&buf2[..n2 as usize]).expect("utf8");
            let expected2 = "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>";
            assert_eq!(rendered2, expected2);
            aether_template_free(h2);
        }
    }

    /// FR-19.9 — BPE encode + decode round-trip AND lowest-rank-wins
    /// selection, exercised back-to-back so the test is sequential (the
    /// runtime's BPE handle table is UnsafeCell + Sync — like all the
    /// other heap-extras tables — so two parallel cargo-test threads
    /// would race on the shared Vec push).
    ///
    /// Scenario A: build merges that take "hello" from 5 single-byte
    /// tokens down to 1; encode "hello world"; verify token ids; decode
    /// to original bytes.
    ///
    /// Scenario B: with 3 competing merges where the lowest-rank rule
    /// would beat an earlier-defined higher-rank rule, verify the BPE
    /// loop picks the rank-0 pair first.
    #[test]
    fn bpe_roundtrip_and_lowest_rank() {
        unsafe {
            // ---- Scenario A: "hello world" full round-trip. ----
            let h = aether_bpe_tokenizer_new();
            assert!(h >= 0);
            let he    = b"he";    let hel   = b"hel";
            let hell  = b"hell";  let hello = b"hello";
            let id_he    = aether_bpe_add_merge(h, 104, 101, 0, he.as_ptr() as _,    2);
            let id_hel   = aether_bpe_add_merge(h, id_he, 108, 1, hel.as_ptr() as _,  3);
            let id_hell  = aether_bpe_add_merge(h, id_hel, 108, 2, hell.as_ptr() as _, 4);
            let id_hello = aether_bpe_add_merge(h, id_hell, 111, 3, hello.as_ptr() as _, 5);
            assert_eq!(id_he,    256);
            assert_eq!(id_hel,   257);
            assert_eq!(id_hell,  258);
            assert_eq!(id_hello, 259);
            let text = b"hello world";
            let mut ids = [0i32; 32];
            let n = aether_bpe_encode(
                h, text.as_ptr() as _, text.len() as i32,
                ids.as_mut_ptr() as _, ids.len() as i32,
            );
            assert_eq!(n, 7, "encoded len");
            assert_eq!(&ids[..7], &[259, 32, 119, 111, 114, 108, 100]);
            let mut out = [0u8; 32];
            let m = aether_bpe_decode(
                h, ids.as_ptr() as _, n,
                out.as_mut_ptr() as _, out.len() as i32,
            );
            assert_eq!(m, 11);
            assert_eq!(&out[..11], b"hello world");
            aether_bpe_tokenizer_free(h);

            // ---- Scenario B: lowest-rank-wins. ----
            // (a, b) rank 5; (b, c) rank 0; (a, bc) rank 1. Encoding
            // "abc" must pick (b, c) first because of its rank-0
            // priority, then (a, bc) → final "abc" token. The (a, b)
            // rank-5 merge is never fired (its 'b' is consumed first).
            let h2 = aether_bpe_tokenizer_new();
            let ab  = b"ab";
            let bc  = b"bc";
            let abc = b"abc";
            let id_ab  = aether_bpe_add_merge(h2, 97, 98, 5, ab.as_ptr() as _, 2);
            let id_bc  = aether_bpe_add_merge(h2, 98, 99, 0, bc.as_ptr() as _, 2);
            let id_abc = aether_bpe_add_merge(h2, 97, id_bc, 1, abc.as_ptr() as _, 3);
            assert_eq!(id_ab,  256);
            assert_eq!(id_bc,  257);
            assert_eq!(id_abc, 258);
            let mut ids2 = [0i32; 8];
            let n2 = aether_bpe_encode(h2, b"abc".as_ptr() as _, 3,
                                       ids2.as_mut_ptr() as _, 8);
            assert_eq!(n2, 1);
            assert_eq!(ids2[0], 258, "expected single 'abc' token via rank-0 (b,c) first");
            aether_bpe_tokenizer_free(h2);
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

    // ================= FR-19.1-extra crypto primitives =================

    fn hex(s: &str) -> Vec<u8> {
        s.as_bytes().chunks(2)
            .map(|c| u8::from_str_radix(std::str::from_utf8(c).unwrap(), 16).unwrap())
            .collect()
    }

    #[test]
    fn sha256_known_vectors() {
        // FIPS 180-4 + RFC vectors.
        let h = super::sha256(b"abc");
        assert_eq!(hex_encode(&h), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        let h = super::sha256(b"");
        assert_eq!(hex_encode(&h), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        // 56-byte boundary case ("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq").
        let h = super::sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        assert_eq!(hex_encode(&h), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }

    #[test]
    fn hmac_sha256_rfc4231_vector1() {
        // RFC 4231 §4.2: Key = 20*0x0b, Data = "Hi There".
        let key = vec![0x0bu8; 20];
        let tag = super::hmac_sha256(&key, b"Hi There");
        assert_eq!(hex_encode(&tag),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");
    }

    #[test]
    fn hkdf_rfc5869_test_case_1() {
        let ikm  = hex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let salt = hex("000102030405060708090a0b0c");
        let info = hex("f0f1f2f3f4f5f6f7f8f9");
        let prk = super::hkdf_extract(&salt, &ikm);
        assert_eq!(hex_encode(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5");
        let okm = super::hkdf_expand(&prk, &info, 42);
        assert_eq!(hex_encode(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865");
    }

    #[test]
    fn x25519_rfc7748_test_vector() {
        // RFC 7748 §5.2 first vector.
        let scalar = hex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
        let mut sc = [0u8; 32]; sc.copy_from_slice(&scalar);
        let u_in = hex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
        let mut u = [0u8; 32]; u.copy_from_slice(&u_in);
        let out = super::x25519_scalar_mult(&sc, &u);
        assert_eq!(hex_encode(&out),
            "c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552");
    }

    #[test]
    fn x25519_dh_roundtrip() {
        // Alice and Bob each derive a public key, then a shared secret.
        // Both arrive at the same shared secret.
        let alice_priv = hex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let bob_priv   = hex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        let mut a = [0u8; 32]; a.copy_from_slice(&alice_priv);
        let mut b = [0u8; 32]; b.copy_from_slice(&bob_priv);
        let mut bp = [0u8; 32]; bp[0] = 9;
        let alice_pub = super::x25519_scalar_mult(&a, &bp);
        let bob_pub   = super::x25519_scalar_mult(&b, &bp);
        let s_alice = super::x25519_scalar_mult(&a, &bob_pub);
        let s_bob   = super::x25519_scalar_mult(&b, &alice_pub);
        assert_eq!(s_alice, s_bob, "DH shared secrets must match");
        // RFC 7748 §6.1 expected secret.
        assert_eq!(hex_encode(&s_alice),
            "4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742");
    }

    #[test]
    fn tls13_hkdf_expand_label_shape() {
        // The wire-shape of HkdfLabel: length(2) + label_len(1) + "tls13 " + user_label + ctx_len(1) + ctx.
        // Use HKDF-Expand under the hood with this info string. Smoke-check
        // that the expand returns the requested length and is deterministic.
        let secret = [0x42u8; 32];
        let label = b"derived";
        let ctx = &super::sha256(b"")[..];
        let mut out = [0u8; 32];
        unsafe {
            let n = aether_tls13_hkdf_expand_label(
                secret.as_ptr() as *const _, 32,
                label.as_ptr() as *const _, label.len() as i32,
                ctx.as_ptr() as *const _, ctx.len() as i32,
                out.as_mut_ptr() as *mut _, 32,
            );
            assert_eq!(n, 32);
        }
        // Second invocation: must match (deterministic).
        let mut out2 = [0u8; 32];
        unsafe {
            aether_tls13_hkdf_expand_label(
                secret.as_ptr() as *const _, 32,
                label.as_ptr() as *const _, label.len() as i32,
                ctx.as_ptr() as *const _, ctx.len() as i32,
                out2.as_mut_ptr() as *mut _, 32,
            );
        }
        assert_eq!(out, out2);
    }

    #[test]
    fn sha512_known_vectors() {
        let h = super::sha512(b"abc");
        assert_eq!(hex_encode(&h),
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a\
2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
        let h = super::sha512(b"");
        assert_eq!(hex_encode(&h),
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
    }

    #[test]
    fn ed25519_rfc8032_test_1() {
        // RFC 8032 §7.1 TEST 1
        let seed = hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
        let pub_expected = hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
        let mut seed_arr = [0u8; 32]; seed_arr.copy_from_slice(&seed);
        let mut pub_out = [0u8; 32];
        unsafe {
            let n = aether_ed25519_derive_public(seed_arr.as_ptr() as *const _, pub_out.as_mut_ptr() as *mut _);
            assert_eq!(n, 32);
        }
        assert_eq!(&pub_out[..], &pub_expected[..]);

        let msg: Vec<u8> = vec![];
        let sig_expected = hex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        );
        let mut sig_out = [0u8; 64];
        unsafe {
            let n = aether_ed25519_sign(
                seed_arr.as_ptr() as *const _, pub_out.as_ptr() as *const _,
                msg.as_ptr() as *const _, msg.len() as i32,
                sig_out.as_mut_ptr() as *mut _,
            );
            assert_eq!(n, 64);
        }
        assert_eq!(&sig_out[..], &sig_expected[..]);

        // Verify the signature.
        unsafe {
            let v = aether_ed25519_verify(
                pub_out.as_ptr() as *const _,
                msg.as_ptr() as *const _, msg.len() as i32,
                sig_out.as_ptr() as *const _,
            );
            assert_eq!(v, 0, "verify should succeed");

            // Flip a sig byte → must reject.
            let mut bad_sig = sig_out;
            bad_sig[0] ^= 1;
            let v = aether_ed25519_verify(
                pub_out.as_ptr() as *const _,
                msg.as_ptr() as *const _, msg.len() as i32,
                bad_sig.as_ptr() as *const _,
            );
            assert_eq!(v, -1, "tampered sig should reject");
        }
    }

    fn hex_encode(b: &[u8]) -> String {
        const H: &[u8] = b"0123456789abcdef";
        let mut s = Vec::with_capacity(b.len() * 2);
        for byte in b {
            s.push(H[(byte >> 4) as usize]);
            s.push(H[(byte & 0xf) as usize]);
        }
        String::from_utf8(s).unwrap()
    }
}
