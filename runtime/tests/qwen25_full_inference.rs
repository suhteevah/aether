//! End-to-end Qwen2.5-7B inference: 28 decoder blocks + final RMSNorm
//! + lm_head, producing actual next-token predictions for a small
//! input sequence.
//!
//! Builds on `qwen25_block_forward.rs`. The single-block proof is
//! repeated 28× under a streaming-dequant loop (load + use + drop a
//! block at a time so peak f32 weight footprint stays around ~870 MB).
//!
//! Skipped if the local Qwen2.5-7B blob isn't present. Marked
//! #[ignore] by default because the full pass takes ~5 min in
//! release; run explicitly with
//!   cargo test -p aether_rt --release --test qwen25_full_inference \
//!     -- --ignored --nocapture

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m, aether_dequant_q6_k,
    aether_op_matmul_f32, aether_op_rms_norm_f32,
    aether_op_rope_apply_f32, aether_op_gqa_repeat_kv_f32,
    aether_op_sdpa_causal_f32,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

const D_MODEL: usize = 3584;
const N_LAYERS: usize = 28;
const N_Q_HEADS: usize = 28;
const N_KV_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / N_Q_HEADS;
const D_KV: usize = N_KV_HEADS * HEAD_DIM;
const D_FF: usize = 18944;
const VOCAB: usize = 152064;
const ROPE_BASE: f32 = 1_000_000.0;
const NORM_EPS: f32 = 1e-6;

unsafe fn load_tensor_f32(h: i64, name: &str) -> Vec<f32> {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0, "tensor not found: {}", name);
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx);
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let mut out = vec![0.0f32; n_elems as usize];
    match dt {
        0 => {
            let src = std::slice::from_raw_parts(dptr as *const f32, n_elems as usize);
            out.copy_from_slice(src);
        }
        12 => {
            let rc = aether_dequant_q4_k_m(dptr as *const c_void, out.as_mut_ptr() as *mut c_void,
                (n_elems / 256) as c_int);
            assert_eq!(rc, 0);
        }
        14 => {
            let rc = aether_dequant_q6_k(dptr as *const c_void, out.as_mut_ptr() as *mut c_void,
                (n_elems / 256) as c_int);
            assert_eq!(rc, 0);
        }
        other => panic!("unsupported dtype {} for {}", other, name),
    }
    out
}

/// Dequant only specific rows of a token-embedding-style tensor stored
/// as `[d_inner=D_MODEL, vocab_outer=VOCAB]` in GGUF. Returns flat
/// f32 of length `rows.len() * D_MODEL`.
unsafe fn dequant_embd_rows(h: i64, name: &str, rows: &[usize]) -> Vec<f32> {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0);
    let dtype = aether_gguf_get_tensor_dtype(h, idx);
    assert_eq!(dtype, 12, "expected token_embd as Q4_K");
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    assert_eq!(D_MODEL % 256, 0, "rows must align to super-block size");
    let blocks_per_row = D_MODEL / 256;  // 14 for Qwen2.5
    let mut out = vec![0.0f32; rows.len() * D_MODEL];
    for (i, &t_id) in rows.iter().enumerate() {
        let base_blocks = t_id * blocks_per_row;
        let block_ptr = (dptr as *const u8).add(base_blocks * 144);  // 144 bytes/Q4_K super-block
        let rc = aether_dequant_q4_k_m(
            block_ptr as *const c_void,
            out[i * D_MODEL..(i + 1) * D_MODEL].as_mut_ptr() as *mut c_void,
            blocks_per_row as c_int,
        );
        assert_eq!(rc, 0);
    }
    out
}

fn transpose_weight(gguf: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d_in * d_out];
    for i_out in 0..d_out {
        for i_in in 0..d_in {
            out[i_in * d_out + i_out] = gguf[i_out * d_in + i_in];
        }
    }
    out
}

fn transpose_seq_head(input: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * n_heads * head_dim];
    for t in 0..seq {
        for hh in 0..n_heads {
            for d in 0..head_dim {
                out[(hh * seq + t) * head_dim + d] = input[(t * n_heads + hh) * head_dim + d];
            }
        }
    }
    out
}

fn transpose_head_seq(input: &[f32], seq: usize, n_heads: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * n_heads * head_dim];
    for hh in 0..n_heads {
        for t in 0..seq {
            for d in 0..head_dim {
                out[(t * n_heads + hh) * head_dim + d] = input[(hh * seq + t) * head_dim + d];
            }
        }
    }
    out
}

fn add_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols { x[r * cols + c] += bias[c]; }
    }
}

/// Run one decoder block forward in place on `x`. Loads + drops all
/// block tensors so peak memory stays ~870 MB.
unsafe fn block_forward_in_place(h: i64, block_idx: usize, x: &mut [f32], seq: usize) {
    let prefix = format!("blk.{}.", block_idx);
    // Load + dequant.
    let attn_norm_g = load_tensor_f32(h, &format!("{}attn_norm.weight", prefix));
    let w_q_gguf  = load_tensor_f32(h, &format!("{}attn_q.weight", prefix));
    let b_q       = load_tensor_f32(h, &format!("{}attn_q.bias", prefix));
    let w_k_gguf  = load_tensor_f32(h, &format!("{}attn_k.weight", prefix));
    let b_k       = load_tensor_f32(h, &format!("{}attn_k.bias", prefix));
    let w_v_gguf  = load_tensor_f32(h, &format!("{}attn_v.weight", prefix));
    let b_v       = load_tensor_f32(h, &format!("{}attn_v.bias", prefix));
    let w_o_gguf  = load_tensor_f32(h, &format!("{}attn_output.weight", prefix));
    let ffn_norm_g = load_tensor_f32(h, &format!("{}ffn_norm.weight", prefix));
    let w_gate_gguf = load_tensor_f32(h, &format!("{}ffn_gate.weight", prefix));
    let w_up_gguf   = load_tensor_f32(h, &format!("{}ffn_up.weight", prefix));
    let w_down_gguf = load_tensor_f32(h, &format!("{}ffn_down.weight", prefix));

    // Transpose to matmul layout. Drop the GGUF-layout copies.
    let w_q = transpose_weight(&w_q_gguf, D_MODEL, D_MODEL); drop(w_q_gguf);
    let w_k = transpose_weight(&w_k_gguf, D_KV, D_MODEL); drop(w_k_gguf);
    let w_v = transpose_weight(&w_v_gguf, D_KV, D_MODEL); drop(w_v_gguf);
    let w_o = transpose_weight(&w_o_gguf, D_MODEL, D_MODEL); drop(w_o_gguf);
    let w_gate = transpose_weight(&w_gate_gguf, D_FF, D_MODEL); drop(w_gate_gguf);
    let w_up = transpose_weight(&w_up_gguf, D_FF, D_MODEL); drop(w_up_gguf);
    let w_down = transpose_weight(&w_down_gguf, D_MODEL, D_FF); drop(w_down_gguf);

    let mut x_norm = vec![0.0f32; seq * D_MODEL];
    aether_op_rms_norm_f32(
        x.as_ptr() as *const c_void,
        attn_norm_g.as_ptr() as *const c_void,
        NORM_EPS,
        x_norm.as_mut_ptr() as *mut c_void,
        seq as c_int, D_MODEL as c_int,
    );

    let mut q = vec![0.0f32; seq * D_MODEL];
    let mut k = vec![0.0f32; seq * D_KV];
    let mut v = vec![0.0f32; seq * D_KV];
    aether_op_matmul_f32(x_norm.as_ptr() as _, w_q.as_ptr() as _, q.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_MODEL as c_int);
    aether_op_matmul_f32(x_norm.as_ptr() as _, w_k.as_ptr() as _, k.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_KV as c_int);
    aether_op_matmul_f32(x_norm.as_ptr() as _, w_v.as_ptr() as _, v.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_KV as c_int);
    add_bias(&mut q, &b_q, seq, D_MODEL);
    add_bias(&mut k, &b_k, seq, D_KV);
    add_bias(&mut v, &b_v, seq, D_KV);

    aether_op_rope_apply_f32(q.as_mut_ptr() as _, seq as c_int, N_Q_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, 0);
    aether_op_rope_apply_f32(k.as_mut_ptr() as _, seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, 0);

    let mut k_full = vec![0.0f32; seq * D_MODEL];
    let mut v_full = vec![0.0f32; seq * D_MODEL];
    aether_op_gqa_repeat_kv_f32(k.as_ptr() as _, k_full.as_mut_ptr() as _,
        seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int);
    aether_op_gqa_repeat_kv_f32(v.as_ptr() as _, v_full.as_mut_ptr() as _,
        seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int);

    let q_hs = transpose_seq_head(&q, seq, N_Q_HEADS, HEAD_DIM);
    let k_hs = transpose_seq_head(&k_full, seq, N_Q_HEADS, HEAD_DIM);
    let v_hs = transpose_seq_head(&v_full, seq, N_Q_HEADS, HEAD_DIM);
    let mut attn_hs = vec![0.0f32; seq * N_Q_HEADS * HEAD_DIM];
    let mut scratch = vec![0.0f32; N_Q_HEADS * seq * seq];
    aether_op_sdpa_causal_f32(
        q_hs.as_ptr() as _, k_hs.as_ptr() as _, v_hs.as_ptr() as _,
        attn_hs.as_mut_ptr() as _, scratch.as_mut_ptr() as _,
        N_Q_HEADS as c_int, seq as c_int, HEAD_DIM as c_int,
    );
    let attn = transpose_head_seq(&attn_hs, seq, N_Q_HEADS, HEAD_DIM);

    let mut proj = vec![0.0f32; seq * D_MODEL];
    aether_op_matmul_f32(attn.as_ptr() as _, w_o.as_ptr() as _, proj.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_MODEL as c_int);
    for i in 0..(seq * D_MODEL) { x[i] += proj[i]; }

    aether_op_rms_norm_f32(x.as_ptr() as _, ffn_norm_g.as_ptr() as _, NORM_EPS,
        x_norm.as_mut_ptr() as _, seq as c_int, D_MODEL as c_int);

    let mut gate = vec![0.0f32; seq * D_FF];
    let mut up = vec![0.0f32; seq * D_FF];
    aether_op_matmul_f32(x_norm.as_ptr() as _, w_gate.as_ptr() as _, gate.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_FF as c_int);
    aether_op_matmul_f32(x_norm.as_ptr() as _, w_up.as_ptr() as _, up.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_FF as c_int);
    for i in 0..(seq * D_FF) {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        gate[i] = silu_g * up[i];
    }
    let mut down = vec![0.0f32; seq * D_MODEL];
    aether_op_matmul_f32(gate.as_ptr() as _, w_down.as_ptr() as _, down.as_mut_ptr() as _,
        seq as c_int, D_FF as c_int, D_MODEL as c_int);
    for i in 0..(seq * D_MODEL) { x[i] += down[i]; }
}

#[test]
#[ignore]  // run explicitly: ~5 min release build
fn qwen25_full_inference_28_blocks() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    let t_total = std::time::Instant::now();
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // Input: 4 token IDs. Pick low IDs that exist in Qwen's vocab.
        // Qwen2.5's BPE has IDs 0..151 as byte-level tokens; we use
        // [9707, 11, 1879, 0] which spells "Hello, world!" approximately.
        // (Exact IDs don't matter for this proof; we just need legitimate
        // tokens that produce valid forward activations.)
        let token_ids: Vec<usize> = vec![9707, 11, 1879, 0];
        let seq = token_ids.len();

        // Dequant only the 4 input embedding rows.
        let t = std::time::Instant::now();
        let mut x = dequant_embd_rows(h, "token_embd.weight", &token_ids);
        eprintln!("[embd] {:.2}s -- 4 token embedding rows", t.elapsed().as_secs_f32());

        // Forward through 28 decoder blocks.
        for b in 0..N_LAYERS {
            let t = std::time::Instant::now();
            block_forward_in_place(h, b, &mut x, seq);
            eprintln!("[blk {:>2}] {:.2}s -- forward pass", b, t.elapsed().as_secs_f32());
        }

        // Final RMSNorm.
        let t = std::time::Instant::now();
        let final_norm_g = load_tensor_f32(h, "output_norm.weight");
        let mut x_final = vec![0.0f32; seq * D_MODEL];
        aether_op_rms_norm_f32(
            x.as_ptr() as _, final_norm_g.as_ptr() as _, NORM_EPS,
            x_final.as_mut_ptr() as _, seq as c_int, D_MODEL as c_int,
        );
        eprintln!("[final_norm] {:.2}s", t.elapsed().as_secs_f32());

        // lm_head: matmul x_final @ output.weight^T to get logits over vocab.
        // Load + transpose output.weight (~2.18 GB f32 + transpose pass).
        let t = std::time::Instant::now();
        let lm_head_gguf = load_tensor_f32(h, "output.weight");
        eprintln!("[lm_head_load] {:.2}s -- {} elems Q6_K -> f32",
            t.elapsed().as_secs_f32(), lm_head_gguf.len());
        let t = std::time::Instant::now();
        let lm_head = transpose_weight(&lm_head_gguf, VOCAB, D_MODEL);
        drop(lm_head_gguf);
        eprintln!("[lm_head_xpose] {:.2}s -- [VOCAB={}, D_MODEL={}] -> [D_MODEL, VOCAB]",
            t.elapsed().as_secs_f32(), VOCAB, D_MODEL);

        let mut logits = vec![0.0f32; seq * VOCAB];
        let t = std::time::Instant::now();
        aether_op_matmul_f32(
            x_final.as_ptr() as _, lm_head.as_ptr() as _, logits.as_mut_ptr() as _,
            seq as c_int, D_MODEL as c_int, VOCAB as c_int,
        );
        eprintln!("[lm_head_mm] {:.2}s -- [{}, {}] @ [{}, {}] -> [{}, {}]",
            t.elapsed().as_secs_f32(), seq, D_MODEL, D_MODEL, VOCAB, seq, VOCAB);

        // Argmax over vocab for each position.
        for t_idx in 0..seq {
            let row = &logits[t_idx * VOCAB .. (t_idx + 1) * VOCAB];
            let (best_id, &best_val) = row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let min_val = row.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(best_val.is_finite(), "logit[{}] not finite", t_idx);
            assert!(min_val.is_finite(), "min logit[{}] not finite ({})", t_idx, min_val);
            assert!(max_val.is_finite(), "max logit[{}] not finite ({})", t_idx, max_val);
            assert!(best_id < VOCAB, "argmax {} out of vocab range", best_id);
            eprintln!("[pos {}] input token {} -> argmax next_id={} logit={:.3} (range [{:.3}, {:.3}])",
                t_idx, token_ids[t_idx], best_id, best_val, min_val, max_val);
        }

        eprintln!("[total] {:.2}s -- full 28-block Qwen2.5-7B forward through Aether",
            t_total.elapsed().as_secs_f32());
        aether_gguf_close(h);
    }
}
