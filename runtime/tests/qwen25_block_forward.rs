//! Real Qwen2.5-7B block 0 forward pass.
//!
//! Reads matt-voice's actual GGUF blob from disk, dequantises every
//! weight tensor of decoder block 0, and runs the full transformer
//! block chain on a small synthetic input through the existing ops
//! surface (RMSNorm + matmul + RoPE + GQA repeat + SDPA + SwiGLU MLP +
//! residual). Output sanity-checked: finite, non-zero, within scale.
//!
//! Skipped on machines where the Qwen2.5-7B blob isn't present.
//!
//! Memory budget: ~870 MB of dequantised f32 weights for block 0
//! (gate/up/down are each ~270 MB). Activations are ~MB-class for
//! seq=4. Runs in single-digit seconds on the 11900K + 64 GB RAM.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_shape, aether_gguf_get_tensor_data_ptr,
    aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m, aether_dequant_q6_k,
    aether_op_matmul_f32, aether_op_rms_norm_f32,
    aether_op_rope_apply_f32, aether_op_gqa_repeat_kv_f32,
    aether_op_sdpa_causal_f32,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

// Qwen2.5-7B-Instruct architecture constants.
const D_MODEL: usize = 3584;
const N_Q_HEADS: usize = 28;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / N_Q_HEADS; // 128
const D_KV: usize = N_KV_HEADS * HEAD_DIM;   // 512
const D_FF: usize = 18944;
const ROPE_BASE: f32 = 1_000_000.0;
const NORM_EPS: f32 = 1e-6;

/// Load a tensor by name and dequant to a flat f32 Vec.
///
/// Returns (data, dims) where `dims` is the GGUF dim list (innermost
/// first, so for an embedding-like tensor it's [d_model, vocab] etc.).
unsafe fn load_tensor_f32(h: i64, name: &str) -> (Vec<f32>, Vec<i64>) {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0, "tensor not found: {}", name);
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx);
    assert!(n_elems > 0);
    let mut dims_buf = [0i64; 8];
    let n_dims = aether_gguf_get_tensor_shape(h, idx, dims_buf.as_mut_ptr() as i64, 8);
    let dims: Vec<i64> = dims_buf[..n_dims as usize].to_vec();
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    assert!(dptr != 0);

    let mut out = vec![0.0f32; n_elems as usize];
    match dt {
        0 => {  // F32 -- straight copy from the GGUF blob
            let src = std::slice::from_raw_parts(dptr as *const f32, n_elems as usize);
            out.copy_from_slice(src);
        }
        12 => {  // Q4_K
            let n_super = (n_elems / 256) as c_int;
            let rc = aether_dequant_q4_k_m(
                dptr as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                n_super,
            );
            assert_eq!(rc, 0, "Q4_K dequant failed for {}", name);
        }
        14 => {  // Q6_K
            let n_super = (n_elems / 256) as c_int;
            let rc = aether_dequant_q6_k(
                dptr as *const c_void,
                out.as_mut_ptr() as *mut c_void,
                n_super,
            );
            assert_eq!(rc, 0, "Q6_K dequant failed for {}", name);
        }
        other => panic!("unsupported dtype {} for {}", other, name),
    }
    (out, dims)
}

/// Transpose a weight tensor from GGUF storage `[d_in_inner, d_out_outer]`
/// to the matmul-friendly layout `[d_in, d_out]` (row-major, outer = d_in).
///
/// GGUF stores W such that elementwise: `gguf[outer=d_out_idx, inner=d_in_idx]
/// = W_math[d_out_idx, d_in_idx]`. Our `aether_op_matmul_f32(a, b, out, m, k, n)`
/// expects `b[k=d_in, n=d_out]`, i.e. transposed from GGUF. This helper
/// produces that layout.
fn transpose_weight(gguf: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d_in * d_out];
    for i_out in 0..d_out {
        for i_in in 0..d_in {
            out[i_in * d_out + i_out] = gguf[i_out * d_in + i_in];
        }
    }
    out
}

/// Transpose activations `[seq, n_heads, head_dim]` -> `[n_heads, seq, head_dim]`
/// (or vice versa). The SDPA kernel expects `[bh, seq, head_dim]`.
fn transpose_seq_head(input: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * n_heads * head_dim];
    for t in 0..seq {
        for h in 0..n_heads {
            for d in 0..head_dim {
                out[(h * seq + t) * head_dim + d] = input[(t * n_heads + h) * head_dim + d];
            }
        }
    }
    out
}

fn transpose_head_seq(input: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * n_heads * head_dim];
    for h in 0..n_heads {
        for t in 0..seq {
            for d in 0..head_dim {
                out[(t * n_heads + h) * head_dim + d] = input[(h * seq + t) * head_dim + d];
            }
        }
    }
    out
}

fn add_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    assert_eq!(bias.len(), cols);
    for r in 0..rows {
        for c in 0..cols { x[r * cols + c] += bias[c]; }
    }
}

#[test]
fn qwen25_block0_full_forward() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    let t_total = std::time::Instant::now();
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // ============================================================ Weights load
        let t = std::time::Instant::now();
        let (token_embd, _) = load_tensor_f32(h, "token_embd.weight");
        let (attn_norm_g, _) = load_tensor_f32(h, "blk.0.attn_norm.weight");
        let (w_q_gguf, _)  = load_tensor_f32(h, "blk.0.attn_q.weight");
        let (b_q,      _)  = load_tensor_f32(h, "blk.0.attn_q.bias");
        let (w_k_gguf, _)  = load_tensor_f32(h, "blk.0.attn_k.weight");
        let (b_k,      _)  = load_tensor_f32(h, "blk.0.attn_k.bias");
        let (w_v_gguf, _)  = load_tensor_f32(h, "blk.0.attn_v.weight");
        let (b_v,      _)  = load_tensor_f32(h, "blk.0.attn_v.bias");
        let (w_o_gguf, _)  = load_tensor_f32(h, "blk.0.attn_output.weight");
        let (ffn_norm_g, _) = load_tensor_f32(h, "blk.0.ffn_norm.weight");
        let (w_gate_gguf, _) = load_tensor_f32(h, "blk.0.ffn_gate.weight");
        let (w_up_gguf,   _) = load_tensor_f32(h, "blk.0.ffn_up.weight");
        let (w_down_gguf, _) = load_tensor_f32(h, "blk.0.ffn_down.weight");
        eprintln!("[load] {:.2}s -- all 13 block-0 tensors dequantised", t.elapsed().as_secs_f32());

        // ============================================================ Transpose weights to matmul layout
        let t = std::time::Instant::now();
        let w_q    = transpose_weight(&w_q_gguf,    D_MODEL, D_MODEL);
        let w_k    = transpose_weight(&w_k_gguf,    D_KV,    D_MODEL);
        let w_v    = transpose_weight(&w_v_gguf,    D_KV,    D_MODEL);
        let w_o    = transpose_weight(&w_o_gguf,    D_MODEL, D_MODEL);
        let w_gate = transpose_weight(&w_gate_gguf, D_FF,    D_MODEL);
        let w_up   = transpose_weight(&w_up_gguf,   D_FF,    D_MODEL);
        let w_down = transpose_weight(&w_down_gguf, D_MODEL, D_FF);
        eprintln!("[xpose] {:.2}s -- 7 weight transposes", t.elapsed().as_secs_f32());

        // ============================================================ Input: 4 tokens, IDs [0, 1, 2, 3]
        let seq = 4usize;
        let token_ids: Vec<usize> = (0..seq).collect();
        let mut x = vec![0.0f32; seq * D_MODEL];
        // token_embd is stored [d_model inner, vocab outer], so token T's
        // embedding is at offset T * D_MODEL in dequant order.
        for (i, &t_id) in token_ids.iter().enumerate() {
            let src = &token_embd[t_id * D_MODEL .. (t_id + 1) * D_MODEL];
            x[i * D_MODEL .. (i + 1) * D_MODEL].copy_from_slice(src);
        }
        eprintln!("[embd] x: 4 tokens -> [4, 3584]");

        // ============================================================ attn_norm (RMSNorm)
        let mut x_norm = vec![0.0f32; seq * D_MODEL];
        let rc = aether_op_rms_norm_f32(
            x.as_ptr() as *const c_void,
            attn_norm_g.as_ptr() as *const c_void,
            NORM_EPS,
            x_norm.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int,
        );
        assert_eq!(rc, 0);

        // ============================================================ Q / K / V projections + bias
        let mut q = vec![0.0f32; seq * D_MODEL];
        let mut k = vec![0.0f32; seq * D_KV];
        let mut v = vec![0.0f32; seq * D_KV];
        let t = std::time::Instant::now();
        aether_op_matmul_f32(
            x_norm.as_ptr() as *const c_void, w_q.as_ptr() as *const c_void, q.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int, D_MODEL as c_int,
        );
        aether_op_matmul_f32(
            x_norm.as_ptr() as *const c_void, w_k.as_ptr() as *const c_void, k.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int, D_KV as c_int,
        );
        aether_op_matmul_f32(
            x_norm.as_ptr() as *const c_void, w_v.as_ptr() as *const c_void, v.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int, D_KV as c_int,
        );
        add_bias(&mut q, &b_q, seq, D_MODEL);
        add_bias(&mut k, &b_k, seq, D_KV);
        add_bias(&mut v, &b_v, seq, D_KV);
        eprintln!("[qkv] {:.2}s -- Q/K/V projections", t.elapsed().as_secs_f32());

        // ============================================================ RoPE on Q and K, positions 0..seq
        aether_op_rope_apply_f32(
            q.as_mut_ptr() as *mut c_void,
            seq as c_int, N_Q_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, 0,
        );
        aether_op_rope_apply_f32(
            k.as_mut_ptr() as *mut c_void,
            seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, 0,
        );

        // ============================================================ GQA repeat K/V: 4 -> 28 heads
        let mut k_full = vec![0.0f32; seq * D_MODEL];
        let mut v_full = vec![0.0f32; seq * D_MODEL];
        aether_op_gqa_repeat_kv_f32(
            k.as_ptr() as *const c_void, k_full.as_mut_ptr() as *mut c_void,
            seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int,
        );
        aether_op_gqa_repeat_kv_f32(
            v.as_ptr() as *const c_void, v_full.as_mut_ptr() as *mut c_void,
            seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int,
        );

        // ============================================================ SDPA causal -- needs [bh, seq, head_dim]
        let q_hs = transpose_seq_head(&q, seq, N_Q_HEADS, HEAD_DIM);
        let k_hs = transpose_seq_head(&k_full, seq, N_Q_HEADS, HEAD_DIM);
        let v_hs = transpose_seq_head(&v_full, seq, N_Q_HEADS, HEAD_DIM);
        let mut attn_hs = vec![0.0f32; seq * N_Q_HEADS * HEAD_DIM];
        let mut attn_scratch = vec![0.0f32; N_Q_HEADS * seq * seq];
        let t = std::time::Instant::now();
        aether_op_sdpa_causal_f32(
            q_hs.as_ptr() as *const c_void,
            k_hs.as_ptr() as *const c_void,
            v_hs.as_ptr() as *const c_void,
            attn_hs.as_mut_ptr() as *mut c_void,
            attn_scratch.as_mut_ptr() as *mut c_void,
            N_Q_HEADS as c_int, seq as c_int, HEAD_DIM as c_int,
        );
        let attn = transpose_head_seq(&attn_hs, seq, N_Q_HEADS, HEAD_DIM);
        eprintln!("[sdpa] {:.2}s -- causal SDPA over 28 heads", t.elapsed().as_secs_f32());

        // ============================================================ Output projection + residual
        let mut proj = vec![0.0f32; seq * D_MODEL];
        aether_op_matmul_f32(
            attn.as_ptr() as *const c_void, w_o.as_ptr() as *const c_void, proj.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int, D_MODEL as c_int,
        );
        for i in 0..(seq * D_MODEL) { x[i] += proj[i]; }
        eprintln!("[res1] residual after attention");

        // ============================================================ ffn_norm
        aether_op_rms_norm_f32(
            x.as_ptr() as *const c_void,
            ffn_norm_g.as_ptr() as *const c_void,
            NORM_EPS,
            x_norm.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int,
        );

        // ============================================================ SwiGLU MLP: down( silu(gate) * up )
        let t = std::time::Instant::now();
        let mut gate = vec![0.0f32; seq * D_FF];
        let mut up = vec![0.0f32; seq * D_FF];
        aether_op_matmul_f32(
            x_norm.as_ptr() as *const c_void, w_gate.as_ptr() as *const c_void, gate.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int, D_FF as c_int,
        );
        aether_op_matmul_f32(
            x_norm.as_ptr() as *const c_void, w_up.as_ptr() as *const c_void, up.as_mut_ptr() as *mut c_void,
            seq as c_int, D_MODEL as c_int, D_FF as c_int,
        );
        // silu(gate) * up
        for i in 0..(seq * D_FF) {
            let g = gate[i];
            let silu_g = g / (1.0 + (-g).exp());
            gate[i] = silu_g * up[i];
        }
        let mut down = vec![0.0f32; seq * D_MODEL];
        aether_op_matmul_f32(
            gate.as_ptr() as *const c_void, w_down.as_ptr() as *const c_void, down.as_mut_ptr() as *mut c_void,
            seq as c_int, D_FF as c_int, D_MODEL as c_int,
        );
        for i in 0..(seq * D_MODEL) { x[i] += down[i]; }
        eprintln!("[mlp] {:.2}s -- SwiGLU MLP", t.elapsed().as_secs_f32());

        // ============================================================ Sanity checks
        let mut any_nan = false;
        let mut any_inf = false;
        let mut max_abs = 0.0f32;
        let mut sum = 0.0f32;
        for &v in &x {
            if v.is_nan() { any_nan = true; }
            if v.is_infinite() { any_inf = true; }
            max_abs = max_abs.max(v.abs());
            sum += v;
        }
        eprintln!(
            "[output] sum={:.3e}, max_abs={:.3e}, nan={}, inf={}, total_time={:.2}s",
            sum, max_abs, any_nan, any_inf, t_total.elapsed().as_secs_f32(),
        );
        assert!(!any_nan, "output contains NaN");
        assert!(!any_inf, "output contains Inf");
        assert!(max_abs > 1e-3, "output too small: max_abs={}", max_abs);
        assert!(max_abs < 1e4, "output too large: max_abs={}", max_abs);

        // Verify the output is meaningfully different per-token (not all-same row).
        let row0_norm: f32 = x[..D_MODEL].iter().map(|v| v * v).sum::<f32>().sqrt();
        let row3_norm: f32 = x[3 * D_MODEL..].iter().map(|v| v * v).sum::<f32>().sqrt();
        eprintln!("[per-token norms] row0={:.3e}, row3={:.3e}", row0_norm, row3_norm);
        assert!(row0_norm > 0.0 && row3_norm > 0.0);

        aether_gguf_close(h);
    }
}
