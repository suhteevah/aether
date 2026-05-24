//! BERT / BGE encoder serving session (FR-17-extra-bert-fwd).
//!
//! End-to-end forward pass through a BERT-shape encoder, producing a sentence
//! embedding suitable for vector search.  Reference: HuggingFace
//! `sentence-transformers/bge-large-en-v1.5` — 24 layers, 1024-d, 16 heads,
//! 64-d head, GELU, CLS-pooling, L2-normalized output.
//!
//! Two construction paths:
//!   * `BertSession::new_synthetic(cfg, seed)` — builds the session with
//!     deterministic synthetic F32 weights.  Used by the parity test.
//!   * `BertSession::from_gguf(path)` — loads bge-style GGUF tensors.  All
//!     F16 weights are dequantized to F32 on the CPU at construction time
//!     and uploaded as F32 device buffers (≈ 1.3 GB resident for bge-large).
//!
//! Lifecycle:
//!   let mut s = BertSession::from_gguf(path)?;
//!   let emb = s.embed(&input_ids, &token_type_ids);  // Vec<f32>, length d_model

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use std::collections::HashMap;

use crate::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_gguf_get_metadata_u32, aether_gguf_get_metadata_f32,
    aether_gguf_get_metadata_array_string_n,
    aether_gguf_get_metadata_array_string_get,
    aether_f16_to_f32,
};
use crate::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_matmul_nt_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_layer_norm_f32_cuda, aether_op_gelu_f32_cuda,
    aether_op_bert_self_attention_fwd_f32_cuda,
    aether_op_bert_embed_sum_f32_cuda,
};

/// BERT-shape runtime config.  Populated from `bert.*` GGUF metadata or
/// manually for the synthetic constructor.
#[derive(Debug, Clone)]
pub struct BertConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub max_pos: usize,
    pub n_token_types: usize,
    pub norm_eps: f32,
    /// Pooling type per llama.cpp convention: 0=NONE, 1=MEAN, 2=CLS, 3=LAST.
    /// bge-large-en-v1.5 uses CLS (= 2).
    pub pooling_type: i32,
}

impl BertConfig {
    pub fn from_gguf(h: i64) -> Self {
        let d_model = read_u32(h, "bert.embedding_length").unwrap_or(1024) as usize;
        let n_layers = read_u32(h, "bert.block_count").unwrap_or(24) as usize;
        let n_heads = read_u32(h, "bert.attention.head_count").unwrap_or(16) as usize;
        let head_dim = if n_heads > 0 { d_model / n_heads } else { 64 };
        let d_ff = read_u32(h, "bert.feed_forward_length").unwrap_or(4096) as usize;
        let max_pos = read_u32(h, "bert.context_length").unwrap_or(512) as usize;
        // bge GGUFs typically store the F16 token_embd at [d_model, vocab] —
        // vocab comes from tokenizer.ggml.tokens length, or fall back to 30522
        // (bert-base-uncased default) if the array isn't there.
        let vocab = {
            let key = b"tokenizer.ggml.tokens";
            let n = unsafe { crate::aether_gguf_get_metadata_array_string_n(
                h, key.as_ptr() as i64, key.len() as c_int) };
            if n > 0 { n as usize } else { 30522 }
        };
        let norm_eps = read_f32(h, "bert.attention.layer_norm_epsilon").unwrap_or(1e-12);
        let pooling_type = read_u32(h, "bert.pooling_type").unwrap_or(2) as i32;
        Self {
            d_model, n_layers, n_heads, head_dim, d_ff, vocab,
            max_pos, n_token_types: 2,
            norm_eps, pooling_type,
        }
    }
}

fn read_u32(h: i64, key: &str) -> Option<u32> {
    let v = unsafe { crate::aether_gguf_get_metadata_u32(
        h, key.as_ptr() as i64, key.len() as c_int) };
    if v < 0 { None } else { Some(v as u32) }
}
fn read_f32(h: i64, key: &str) -> Option<f32> {
    let v = unsafe { crate::aether_gguf_get_metadata_f32(
        h, key.as_ptr() as i64, key.len() as c_int) };
    if v.is_nan() { None } else { Some(v as f32) }
}

struct BertBlock {
    // All weights F32 device handles.  W stored row-major as [n_out, n_in].
    w_q: i64, w_k: i64, w_v: i64, w_o: i64,
    b_q: i64, b_k: i64, b_v: i64, b_o: i64,
    // post-attention LayerNorm
    attn_out_norm_g: i64, attn_out_norm_b: i64,
    // FFN: ffn_up [d_ff, d_model], ffn_down [d_model, d_ff]
    w_ffn_up: i64, b_ffn_up: i64,
    w_ffn_down: i64, b_ffn_down: i64,
    // post-FFN LayerNorm
    layer_out_norm_g: i64, layer_out_norm_b: i64,
}

impl BertSession {
    /// Read back the first layer's W_Q weight tensor for debugging — caller
    /// passes a host buffer big enough for n_out*n_in floats.  No-op when the
    /// session has zero layers.
    pub fn debug_download_block0_wq(&self, out: &mut Vec<f32>) {
        if self.blocks.is_empty() { return; }
        let n = self.cfg.d_model * self.cfg.d_model;
        out.resize(n, 0.0);
        unsafe {
            aether_dev_d2h_f32(self.blocks[0].w_q, out.as_mut_ptr() as i64, n as c_int);
        }
    }

    /// Debug helper — extract ALL the f32 weight tensors for block 0 into a
    /// flat slice of host vectors so the parity test can drive its CPU
    /// reference with exactly the same weights the GPU is using.
    pub fn debug_download_block0_weights(&self) -> Vec<Vec<f32>> {
        if self.blocks.is_empty() { return Vec::new(); }
        let b = &self.blocks[0];
        let d = self.cfg.d_model;
        let d_ff = self.cfg.d_ff;
        let dump = |h: i64, n: usize| -> Vec<f32> {
            let mut v = vec![0f32; n];
            unsafe { aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, n as c_int); }
            v
        };
        vec![
            dump(b.w_q, d * d), dump(b.w_k, d * d), dump(b.w_v, d * d), dump(b.w_o, d * d),
            dump(b.b_q, d), dump(b.b_k, d), dump(b.b_v, d), dump(b.b_o, d),
            dump(b.attn_out_norm_g, d), dump(b.attn_out_norm_b, d),
            dump(b.w_ffn_up, d_ff * d), dump(b.b_ffn_up, d_ff),
            dump(b.w_ffn_down, d * d_ff), dump(b.b_ffn_down, d),
            dump(b.layer_out_norm_g, d), dump(b.layer_out_norm_b, d),
        ]
    }

    /// Debug helper — return Q after the first matmul (no bias since synthetic
    /// b=0) of block 0.  Bypasses attention; useful for isolating matmul drift.
    pub fn debug_block0_q(
        &mut self, input_ids: &[i32], token_type_ids: &[i32],
    ) -> Vec<f32> {
        let seq = input_ids.len();
        let d_model = self.cfg.d_model;
        let eps = self.cfg.norm_eps;
        unsafe {
            let act = (seq * d_model) as c_int;
            let input_ids_dev = aether_dev_alloc_i32(seq as c_int);
            let type_ids_dev  = aether_dev_alloc_i32(seq as c_int);
            let act_x = aether_dev_alloc_f32(act);
            let act_q = aether_dev_alloc_f32(act);
            let mean = aether_dev_alloc_f32(seq as c_int);
            let rstd = aether_dev_alloc_f32(seq as c_int);
            aether_dev_h2d_i32(input_ids.as_ptr() as i64, input_ids_dev, seq as c_int);
            aether_dev_h2d_i32(token_type_ids.as_ptr() as i64, type_ids_dev, seq as c_int);
            aether_op_bert_embed_sum_f32_cuda(input_ids_dev, type_ids_dev,
                self.word_embd, self.pos_embd, self.type_embd,
                act_x, seq as c_int, d_model as c_int);
            aether_op_layer_norm_f32_cuda(act_x, self.pre_norm_g, self.pre_norm_b,
                act_x, mean, rstd, eps, seq as c_int, d_model as c_int);
            aether_op_matmul_nt_f32_cuda(act_x, self.blocks[0].w_q, act_q,
                seq as c_int, d_model as c_int, d_model as c_int);
            aether_dev_sync();
            let mut out = vec![0f32; seq * d_model];
            aether_dev_d2h_f32(act_q, out.as_mut_ptr() as i64, (seq * d_model) as c_int);
            aether_dev_free_i32(input_ids_dev); aether_dev_free_i32(type_ids_dev);
            for h in [act_x, act_q, mean, rstd] { aether_dev_free_f32(h); }
            out
        }
    }

    /// Debug helper — return the post-`stop_after_blocks`-blocks hidden state
    /// (skips pooling/L2-norm).  `stop_after_blocks = 0` returns the
    /// pre-encoder LN'd embedding; `stop_after_blocks = N` returns the
    /// activation after N blocks have run.
    pub fn debug_intermediate(
        &mut self, input_ids: &[i32], token_type_ids: &[i32],
        stop_after_blocks: usize,
    ) -> Vec<f32> {
        let seq = input_ids.len();
        let d_model = self.cfg.d_model;
        let d_ff = self.cfg.d_ff;
        let n_heads = self.cfg.n_heads as c_int;
        let head_dim = self.cfg.head_dim as c_int;
        let eps = self.cfg.norm_eps;
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();
        unsafe {
            let act = (seq * d_model) as c_int;
            let ffn = (seq * d_ff) as c_int;
            let input_ids_dev = aether_dev_alloc_i32(seq as c_int);
            let type_ids_dev  = aether_dev_alloc_i32(seq as c_int);
            let act_x        = aether_dev_alloc_f32(act);
            let act_resid    = aether_dev_alloc_f32(act);
            let act_q        = aether_dev_alloc_f32(act);
            let act_k        = aether_dev_alloc_f32(act);
            let act_v        = aether_dev_alloc_f32(act);
            let act_attn_out = aether_dev_alloc_f32(act);
            let act_ffn_up   = aether_dev_alloc_f32(ffn);
            let act_ffn_down = aether_dev_alloc_f32(act);
            let mean = aether_dev_alloc_f32(seq as c_int);
            let rstd = aether_dev_alloc_f32(seq as c_int);
            aether_dev_h2d_i32(input_ids.as_ptr() as i64, input_ids_dev, seq as c_int);
            aether_dev_h2d_i32(token_type_ids.as_ptr() as i64, type_ids_dev, seq as c_int);
            aether_op_bert_embed_sum_f32_cuda(input_ids_dev, type_ids_dev,
                self.word_embd, self.pos_embd, self.type_embd,
                act_x, seq as c_int, d_model as c_int);
            aether_op_layer_norm_f32_cuda(act_x, self.pre_norm_g, self.pre_norm_b,
                act_x, mean, rstd, eps, seq as c_int, d_model as c_int);
            for (idx, blk) in self.blocks.iter().enumerate() {
                if idx >= stop_after_blocks { break; }
                copy_dev_f32(act_x, act_resid, (seq * d_model) as c_int);
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_q, act_q,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_q, blk.b_q, seq as c_int, d_model as c_int);
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_k, act_k,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_k, blk.b_k, seq as c_int, d_model as c_int);
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_v, act_v,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_v, blk.b_v, seq as c_int, d_model as c_int);
                let _rc = aether_op_bert_self_attention_fwd_f32_cuda(act_q, act_k, act_v,
                    act_attn_out, seq as c_int, n_heads, head_dim, scale);
                assert_eq!(_rc, 0, "bert_self_attention_fwd rc={}", _rc);
                aether_op_matmul_nt_f32_cuda(act_attn_out, blk.w_o, act_x,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_x, blk.b_o, seq as c_int, d_model as c_int);
                aether_op_add_inplace_f32_cuda(act_x, act_resid, (seq * d_model) as c_int);
                aether_op_layer_norm_f32_cuda(act_x, blk.attn_out_norm_g, blk.attn_out_norm_b,
                    act_x, mean, rstd, eps, seq as c_int, d_model as c_int);
                copy_dev_f32(act_x, act_resid, (seq * d_model) as c_int);
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_ffn_up, act_ffn_up,
                    seq as c_int, d_model as c_int, d_ff as c_int);
                aether_op_bias_add_f32_cuda(act_ffn_up, blk.b_ffn_up,
                    seq as c_int, d_ff as c_int);
                aether_op_gelu_f32_cuda(act_ffn_up, act_ffn_up, (seq * d_ff) as c_int);
                aether_op_matmul_nt_f32_cuda(act_ffn_up, blk.w_ffn_down, act_ffn_down,
                    seq as c_int, d_ff as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_ffn_down, blk.b_ffn_down,
                    seq as c_int, d_model as c_int);
                aether_op_add_inplace_f32_cuda(act_ffn_down, act_resid,
                    (seq * d_model) as c_int);
                aether_op_layer_norm_f32_cuda(act_ffn_down,
                    blk.layer_out_norm_g, blk.layer_out_norm_b,
                    act_x, mean, rstd, eps, seq as c_int, d_model as c_int);
            }
            aether_dev_sync();
            let mut out = vec![0f32; seq * d_model];
            aether_dev_d2h_f32(act_x, out.as_mut_ptr() as i64, (seq * d_model) as c_int);
            aether_dev_free_i32(input_ids_dev);
            aether_dev_free_i32(type_ids_dev);
            for h in [act_x, act_resid, act_q, act_k, act_v, act_attn_out,
                      act_ffn_up, act_ffn_down, mean, rstd] {
                aether_dev_free_f32(h);
            }
            out
        }
    }
}

pub struct BertSession {
    pub cfg: BertConfig,
    pub max_seq: usize,
    // Embedding tables (all F32).
    word_embd: i64,   // [vocab, d_model]
    pos_embd: i64,    // [max_pos, d_model]
    type_embd: i64,   // [n_types, d_model]
    // Pre-encoder LayerNorm (token_embd_norm.*)
    pre_norm_g: i64, pre_norm_b: i64,
    blocks: Vec<BertBlock>,
    // Stale handles to free in Drop.
    bufs_to_free: Vec<i64>,
}

impl BertSession {
    /// Build a session with deterministic synthetic F32 weights — used by the
    /// parity test.  Every weight tensor is seeded from a tiny PRNG so the
    /// CPU and GPU paths see exactly the same numbers.
    pub fn new_synthetic(cfg: BertConfig, max_seq: usize, seed: u64) -> Self {
        unsafe { aether_dev_init(); }
        let mut s = SyntheticGen { state: seed.wrapping_add(1) };
        let d_model = cfg.d_model;
        let d_ff = cfg.d_ff;

        let mut bufs_to_free = Vec::new();
        let alloc_and_upload = |data: &[f32], bufs_to_free: &mut Vec<i64>| -> i64 {
            unsafe {
                let h = aether_dev_alloc_f32(data.len() as c_int);
                aether_dev_h2d_f32(data.as_ptr() as i64, h, data.len() as c_int);
                bufs_to_free.push(h);
                h
            }
        };

        let word_embd_host = s.fill(cfg.vocab * d_model, 0.02);
        let pos_embd_host = s.fill(cfg.max_pos * d_model, 0.02);
        let type_embd_host = s.fill(cfg.n_token_types * d_model, 0.02);
        let pre_norm_g_host = s.fill_constant(d_model, 1.0);
        let pre_norm_b_host = s.fill_constant(d_model, 0.0);

        let word_embd = alloc_and_upload(&word_embd_host, &mut bufs_to_free);
        let pos_embd  = alloc_and_upload(&pos_embd_host,  &mut bufs_to_free);
        let type_embd = alloc_and_upload(&type_embd_host, &mut bufs_to_free);
        let pre_norm_g = alloc_and_upload(&pre_norm_g_host, &mut bufs_to_free);
        let pre_norm_b = alloc_and_upload(&pre_norm_b_host, &mut bufs_to_free);

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let init_w = |out_dim: usize, in_dim: usize, gen: &mut SyntheticGen| -> Vec<f32> {
                // Xavier-ish: variance 1/in_dim.
                let sc = (1.0 / in_dim as f32).sqrt();
                gen.fill(out_dim * in_dim, sc)
            };
            let wq = init_w(d_model, d_model, &mut s);
            let wk = init_w(d_model, d_model, &mut s);
            let wv = init_w(d_model, d_model, &mut s);
            let wo = init_w(d_model, d_model, &mut s);
            let bq = s.fill_constant(d_model, 0.0);
            let bk = s.fill_constant(d_model, 0.0);
            let bv = s.fill_constant(d_model, 0.0);
            let bo = s.fill_constant(d_model, 0.0);
            let aon_g = s.fill_constant(d_model, 1.0);
            let aon_b = s.fill_constant(d_model, 0.0);
            let w_up = init_w(d_ff, d_model, &mut s);
            let b_up = s.fill_constant(d_ff, 0.0);
            let w_dn = init_w(d_model, d_ff, &mut s);
            let b_dn = s.fill_constant(d_model, 0.0);
            let lon_g = s.fill_constant(d_model, 1.0);
            let lon_b = s.fill_constant(d_model, 0.0);

            let blk = BertBlock {
                w_q: alloc_and_upload(&wq, &mut bufs_to_free),
                w_k: alloc_and_upload(&wk, &mut bufs_to_free),
                w_v: alloc_and_upload(&wv, &mut bufs_to_free),
                w_o: alloc_and_upload(&wo, &mut bufs_to_free),
                b_q: alloc_and_upload(&bq, &mut bufs_to_free),
                b_k: alloc_and_upload(&bk, &mut bufs_to_free),
                b_v: alloc_and_upload(&bv, &mut bufs_to_free),
                b_o: alloc_and_upload(&bo, &mut bufs_to_free),
                attn_out_norm_g: alloc_and_upload(&aon_g, &mut bufs_to_free),
                attn_out_norm_b: alloc_and_upload(&aon_b, &mut bufs_to_free),
                w_ffn_up: alloc_and_upload(&w_up, &mut bufs_to_free),
                b_ffn_up: alloc_and_upload(&b_up, &mut bufs_to_free),
                w_ffn_down: alloc_and_upload(&w_dn, &mut bufs_to_free),
                b_ffn_down: alloc_and_upload(&b_dn, &mut bufs_to_free),
                layer_out_norm_g: alloc_and_upload(&lon_g, &mut bufs_to_free),
                layer_out_norm_b: alloc_and_upload(&lon_b, &mut bufs_to_free),
            };
            blocks.push(blk);
        }

        // Activations are allocated per-request inside embed() — cudarc's
        // sync-copy paths assert equal-length slices, so a max-seq-sized
        // buffer can't host a shorter seq.  See the helper `embed_alloc`.
        let _ = d_model; let _ = d_ff;
        Self {
            cfg, max_seq,
            word_embd, pos_embd, type_embd,
            pre_norm_g, pre_norm_b,
            blocks,
            bufs_to_free,
        }
    }

    /// Embed a sequence of token ids.  Returns a Vec<f32> of length d_model
    /// containing the sentence embedding (CLS-pooled for bge-large), L2-
    /// normalized when the config's pooling type is CLS or MEAN.
    ///
    /// `input_ids` and `token_type_ids` must have the same length and ≤ max_seq.
    pub fn embed(&mut self, input_ids: &[i32], token_type_ids: &[i32]) -> Vec<f32> {
        assert_eq!(input_ids.len(), token_type_ids.len(), "ids length mismatch");
        let seq = input_ids.len();
        assert!(seq > 0 && seq <= self.max_seq, "seq {} out of range [1, {}]", seq, self.max_seq);
        let d_model = self.cfg.d_model;
        let d_ff = self.cfg.d_ff;
        let n_heads = self.cfg.n_heads as c_int;
        let head_dim = self.cfg.head_dim as c_int;
        let eps = self.cfg.norm_eps;
        let scale = 1.0 / (self.cfg.head_dim as f32).sqrt();

        unsafe {
            // Per-request scratch buffers — all sized to the actual seq.
            let act = (seq * d_model) as c_int;
            let ffn = (seq * d_ff) as c_int;
            let input_ids_dev = aether_dev_alloc_i32(seq as c_int);
            let type_ids_dev  = aether_dev_alloc_i32(seq as c_int);
            let act_x        = aether_dev_alloc_f32(act);
            let act_resid    = aether_dev_alloc_f32(act);
            let act_q        = aether_dev_alloc_f32(act);
            let act_k        = aether_dev_alloc_f32(act);
            let act_v        = aether_dev_alloc_f32(act);
            let act_attn_out = aether_dev_alloc_f32(act);
            let act_ffn_up   = aether_dev_alloc_f32(ffn);
            let act_ffn_down = aether_dev_alloc_f32(act);
            let mean = aether_dev_alloc_f32(seq as c_int);
            let rstd = aether_dev_alloc_f32(seq as c_int);

            aether_dev_h2d_i32(input_ids.as_ptr() as i64, input_ids_dev, seq as c_int);
            aether_dev_h2d_i32(token_type_ids.as_ptr() as i64, type_ids_dev, seq as c_int);

            // 1. embed sum: act_x = word[input] + pos + type
            let rc = aether_op_bert_embed_sum_f32_cuda(
                input_ids_dev, type_ids_dev,
                self.word_embd, self.pos_embd, self.type_embd,
                act_x, seq as c_int, d_model as c_int);
            assert_eq!(rc, 0, "bert_embed_sum rc={}", rc);

            // 2. pre-encoder LN (token_embd_norm).  In-place via act_x → act_x.
            aether_op_layer_norm_f32_cuda(
                act_x, self.pre_norm_g, self.pre_norm_b,
                act_x, mean, rstd,
                eps, seq as c_int, d_model as c_int);

            // 3. Per-block forward.
            for blk in &self.blocks {
                // Save residual for post-attention add.
                copy_dev_f32(act_x, act_resid, (seq * d_model) as c_int);

                // Q/K/V matmuls.  Weight W is stored as [n_out, n_in] row-major
                // so matmul_nt does X[seq, d_in] @ W^T → [seq, d_out].
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_q, act_q,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_q, blk.b_q, seq as c_int, d_model as c_int);
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_k, act_k,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_k, blk.b_k, seq as c_int, d_model as c_int);
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_v, act_v,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_v, blk.b_v, seq as c_int, d_model as c_int);

                // Bidirectional self-attention.
                let _rc = aether_op_bert_self_attention_fwd_f32_cuda(
                    act_q, act_k, act_v, act_attn_out,
                    seq as c_int, n_heads, head_dim, scale);
                assert_eq!(_rc, 0, "bert_self_attention_fwd rc={}", _rc);

                // Output projection.
                aether_op_matmul_nt_f32_cuda(act_attn_out, blk.w_o, act_x,
                    seq as c_int, d_model as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_x, blk.b_o, seq as c_int, d_model as c_int);

                // Residual + post-attention LN.
                aether_op_add_inplace_f32_cuda(act_x, act_resid, (seq * d_model) as c_int);
                aether_op_layer_norm_f32_cuda(
                    act_x, blk.attn_out_norm_g, blk.attn_out_norm_b,
                    act_x, mean, rstd,
                    eps, seq as c_int, d_model as c_int);

                // Save residual for post-FFN add.
                copy_dev_f32(act_x, act_resid, (seq * d_model) as c_int);

                // FFN up.
                aether_op_matmul_nt_f32_cuda(act_x, blk.w_ffn_up, act_ffn_up,
                    seq as c_int, d_model as c_int, d_ff as c_int);
                aether_op_bias_add_f32_cuda(act_ffn_up, blk.b_ffn_up,
                    seq as c_int, d_ff as c_int);
                aether_op_gelu_f32_cuda(act_ffn_up, act_ffn_up,
                    (seq * d_ff) as c_int);
                // FFN down.
                aether_op_matmul_nt_f32_cuda(act_ffn_up, blk.w_ffn_down, act_ffn_down,
                    seq as c_int, d_ff as c_int, d_model as c_int);
                aether_op_bias_add_f32_cuda(act_ffn_down, blk.b_ffn_down,
                    seq as c_int, d_model as c_int);

                // Add residual + post-FFN LN.  Add resid into act_ffn_down,
                // then LN(act_ffn_down) into act_x.
                aether_op_add_inplace_f32_cuda(act_ffn_down, act_resid,
                    (seq * d_model) as c_int);
                aether_op_layer_norm_f32_cuda(
                    act_ffn_down, blk.layer_out_norm_g, blk.layer_out_norm_b,
                    act_x, mean, rstd,
                    eps, seq as c_int, d_model as c_int);
            }

            // 4. Pooling.
            aether_dev_sync();
            let mut full_host = vec![0f32; seq * d_model];
            aether_dev_d2h_f32(act_x, full_host.as_mut_ptr() as i64,
                (seq * d_model) as c_int);
            let mut emb = match self.cfg.pooling_type {
                2 => full_host[..d_model].to_vec(),                       // CLS
                1 => {
                    let mut m = vec![0f32; d_model];
                    for t in 0..seq {
                        for j in 0..d_model {
                            m[j] += full_host[t * d_model + j];
                        }
                    }
                    for j in 0..d_model { m[j] /= seq as f32; }
                    m
                }
                _ => full_host[..d_model].to_vec(),
            };
            // L2 normalize — bge-large emits unit-norm vectors so cosine sim
            // works as dot product downstream.
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for v in &mut emb { *v /= norm; }
            }
            // Free per-request scratch.
            aether_dev_free_i32(input_ids_dev);
            aether_dev_free_i32(type_ids_dev);
            for h in [act_x, act_resid, act_q, act_k, act_v, act_attn_out,
                      act_ffn_up, act_ffn_down, mean, rstd] {
                aether_dev_free_f32(h);
            }
            emb
        }
    }
}

impl Drop for BertSession {
    fn drop(&mut self) {
        unsafe {
            for h in self.bufs_to_free.drain(..) {
                aether_dev_free_f32(h);
            }
        }
    }
}

// ============================================================================
// WordPiece tokenizer — BERT-style (llama.cpp bge-large convention).
//
// llama.cpp's bge-large GGUF re-encodes the BERT WordPiece vocab with
// SentencePiece-style `▁` (U+2581) prefixes for word-initial pieces.  The
// `##` continuation prefix from HuggingFace's BertTokenizer is NOT in this
// vocab — continuation pieces are just bare strings.  Verified by probing
// the local bge-large GGUF: zero `##*` entries, but `▁the`, `▁quick`, etc.
// at IDs 1996, 4248, ... (matching HF bert-base-uncased's "the", "quick").
//
// Pipeline:
//   1. Lowercase ASCII (uncased BERT family).
//   2. Whitespace + punctuation tokenize.  Each punctuation char becomes its
//      own basic-token (matches HF's BertTokenizer).
//   3. For each basic-token, prepend `▁` and do greedy LONGEST-match against
//      the vocab.  If no prefix matches, emit a single [UNK] for the whole
//      basic-token.  Continuation pieces (after the first match) have no
//      prefix.
//   4. Prepend [CLS] (101), append [SEP] (102).
//
// Non-ASCII characters are passed through (lowercased only when in ASCII
// range).  Accent stripping + NFD normalization are NOT implemented yet —
// callers passing accented text get [UNK] for the accented words.  Fine for
// the OpenClaw English corpus; tracked as a follow-on.
// ============================================================================

/// Llama.cpp's BERT vocab uses U+2581 (▁) as the word-initial marker.
const WORD_PREFIX: &str = "\u{2581}";

pub struct WordPieceTokenizer {
    vocab: HashMap<String, i32>,
    pub cls_id: i32,
    pub sep_id: i32,
    pub unk_id: i32,
    pub pad_id: i32,
    pub mask_id: i32,
    /// Per-word character cap — words longer than this emit a single [UNK]
    /// without attempting WordPiece.  Matches bert-base default of 100.
    pub max_input_chars_per_word: usize,
}

impl WordPieceTokenizer {
    /// Build a tokenizer from a bert-arch GGUF.  Reads
    /// `tokenizer.ggml.tokens` (StringArray) + the cls / sep / unk / pad /
    /// mask token-id metadata keys.  Defaults match bert-base-uncased when
    /// any id key is missing.
    pub unsafe fn from_gguf(h: i64) -> Result<Self, String> {
        let key = b"tokenizer.ggml.tokens";
        let n = aether_gguf_get_metadata_array_string_n(
            h, key.as_ptr() as i64, key.len() as c_int);
        if n <= 0 {
            return Err("GGUF missing tokenizer.ggml.tokens array".to_string());
        }
        let mut vocab: HashMap<String, i32> = HashMap::with_capacity(n as usize);
        let mut buf = vec![0u8; 512];
        for i in 0..n {
            let got = aether_gguf_get_metadata_array_string_get(
                h, key.as_ptr() as i64, key.len() as c_int, i,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if got <= 0 { continue; }
            if let Ok(s) = std::str::from_utf8(&buf[..got as usize]) {
                vocab.insert(s.to_string(), i as i32);
            }
        }
        let read_id = |k: &str| -> Option<i32> {
            let v = aether_gguf_get_metadata_u32(
                h, k.as_ptr() as i64, k.len() as c_int);
            if v < 0 { None } else { Some(v as i32) }
        };
        Ok(Self {
            cls_id:  read_id("tokenizer.ggml.cls_token_id").unwrap_or(101),
            sep_id:  read_id("tokenizer.ggml.seperator_token_id")
                        .or_else(|| read_id("tokenizer.ggml.sep_token_id"))
                        .unwrap_or(102),
            unk_id:  read_id("tokenizer.ggml.unknown_token_id").unwrap_or(100),
            pad_id:  read_id("tokenizer.ggml.padding_token_id").unwrap_or(0),
            mask_id: read_id("tokenizer.ggml.mask_token_id").unwrap_or(103),
            max_input_chars_per_word: 100,
            vocab,
        })
    }

    /// Tokenize `text` into BERT-shape ids: [CLS] + WordPiece(text) + [SEP].
    /// Does NOT pad — the caller's max_seq cap is enforced by the embedding
    /// session's range check.  Returns an empty vec only when both CLS/SEP
    /// are absent and text is empty — never panics.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let basic = self.basic_tokenize(text);
        let mut ids = Vec::with_capacity(basic.len() + 2);
        if self.cls_id >= 0 { ids.push(self.cls_id); }
        for tok in &basic {
            self.wordpiece(tok, &mut ids);
        }
        if self.sep_id >= 0 { ids.push(self.sep_id); }
        ids
    }

    /// Basic split: lowercase ASCII letters in-place, treat every ASCII
    /// punctuation character as its own basic-token, split on whitespace.
    /// Non-ASCII characters flow through verbatim (no NFD / accent
    /// stripping yet).
    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let flush = |cur: &mut String, out: &mut Vec<String>| {
            if !cur.is_empty() {
                out.push(std::mem::take(cur));
            }
        };
        for ch in text.chars() {
            if ch.is_whitespace() {
                flush(&mut cur, &mut out);
            } else if is_bert_punct(ch) {
                flush(&mut cur, &mut out);
                out.push(ch.to_string());
            } else {
                // ASCII-lowercase; pass non-ASCII through (NFD / accent
                // strip is a follow-on).
                let lo = if ch.is_ascii_uppercase() {
                    ch.to_ascii_lowercase()
                } else { ch };
                cur.push(lo);
            }
        }
        flush(&mut cur, &mut out);
        out
    }

    /// Greedy longest-match on `word`.  First piece is looked up with the
    /// `▁` (U+2581) prefix that marks word-initial pieces in this vocab;
    /// continuation pieces have no prefix.  If no prefix matches at any
    /// step, the whole word emits a single [UNK].
    fn wordpiece(&self, word: &str, out: &mut Vec<i32>) {
        if word.is_empty() { return; }
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_input_chars_per_word {
            out.push(self.unk_id);
            return;
        }
        let mut sub_tokens = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut found: Option<(i32, usize)> = None;
            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let key = if start == 0 {
                    format!("{}{}", WORD_PREFIX, substr)
                } else {
                    substr
                };
                if let Some(&id) = self.vocab.get(&key) {
                    found = Some((id, end));
                    break;
                }
                end -= 1;
            }
            match found {
                Some((id, new_start)) => {
                    sub_tokens.push(id);
                    start = new_start;
                }
                None => {
                    // No match for ANY prefix → emit [UNK] for the whole word.
                    out.push(self.unk_id);
                    return;
                }
            }
        }
        out.extend(sub_tokens);
    }
}

fn is_bert_punct(ch: char) -> bool {
    // Match HuggingFace's BertTokenizer punctuation set: ASCII punct (codes
    // !".../09:?@[\\]^_` etc.) PLUS Unicode categories P*/S* — but we only
    // implement the ASCII slice here; the unicode-category check is the same
    // follow-on as accent stripping.
    if ch.is_ascii() && !ch.is_ascii_alphanumeric() && !ch.is_ascii_whitespace() {
        return true;
    }
    false
}

impl BertSession {
    /// Build a WordPiece tokenizer from this session's GGUF and use it to
    /// embed a raw text string.  Only available when the session was loaded
    /// from a GGUF (the synthetic constructor doesn't carry a tokenizer).
    /// Re-opens the GGUF — for repeat calls callers should keep a
    /// WordPieceTokenizer in hand and reuse it across embed() calls.
    pub fn embed_text(
        &mut self, gguf_path: &str, text: &str,
    ) -> Result<Vec<f32>, String> {
        unsafe {
            let h = aether_gguf_open(
                gguf_path.as_ptr() as i64, gguf_path.len() as c_int);
            if h < 0 { return Err(format!("aether_gguf_open: {}", h)); }
            let tok = WordPieceTokenizer::from_gguf(h)?;
            aether_gguf_close(h);
            let ids = tok.encode(text);
            let token_type_ids = vec![0i32; ids.len()];
            Ok(self.embed(&ids, &token_type_ids))
        }
    }
}

unsafe fn copy_dev_f32(src: i64, dst: i64, n: c_int) {
    // Round-trip via host — same shape as several other places in
    // serving.rs that do small in-graph copies.  For larger tensors a
    // dedicated device-to-device memcpy kernel would be the win.
    let mut tmp = vec![0f32; n as usize];
    aether_dev_d2h_f32(src, tmp.as_mut_ptr() as i64, n);
    aether_dev_h2d_f32(tmp.as_ptr() as i64, dst, n);
}

/// Deterministic splitmix64 PRNG for synthetic weight init.  `pub` so the
/// parity test can reproduce the exact weight sequence on the CPU side.
pub struct SyntheticGen { pub state: u64 }
impl SyntheticGen {
    pub fn next_u32(&mut self) -> u32 {
        let mut z = self.state.wrapping_add(0x9E3779B97F4A7C15);
        self.state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z >> 32) ^ z) as u32
    }
    pub fn next_f32(&mut self) -> f32 {
        // Uniform in [-1, 1].
        let u = self.next_u32();
        (u as f32 / 4_294_967_296.0) * 2.0 - 1.0
    }
    pub fn fill(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
    pub fn fill_constant(&mut self, n: usize, c: f32) -> Vec<f32> {
        vec![c; n]
    }
}

/// Load a bge-style BERT GGUF.  All F16 weights are dequantized to F32 on the
/// CPU and uploaded as F32 device buffers.  Resident memory is ~2× the GGUF
/// file size — bge-large at 670 MiB on disk lands at ~1.3 GiB in VRAM.
impl BertSession {
    pub fn from_gguf(gguf_path: &str) -> Result<Self, String> {
        if !std::path::Path::new(gguf_path).exists() {
            return Err(format!("GGUF not found: {}", gguf_path));
        }
        unsafe {
            aether_dev_init();
            let h = aether_gguf_open(gguf_path.as_ptr() as i64, gguf_path.len() as c_int);
            if h < 0 {
                return Err(format!("aether_gguf_open failed: {}", h));
            }
            let cfg = BertConfig::from_gguf(h);
            let max_seq = cfg.max_pos;
            eprintln!("[BertSession] arch=bert layers={} d_model={} heads={} head_dim={} d_ff={} vocab={} max_pos={} eps={:.2e} pool={}",
                cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.head_dim, cfg.d_ff,
                cfg.vocab, cfg.max_pos, cfg.norm_eps, cfg.pooling_type);

            let mut bufs_to_free = Vec::new();
            let upload_tensor = |name: &str, expected_elems: usize,
                                  bufs_to_free: &mut Vec<i64>| -> Result<i64, String> {
                let idx = aether_gguf_find_tensor_by_name(
                    h, name.as_ptr() as i64, name.len() as c_int);
                if idx < 0 {
                    aether_gguf_close(h);
                    return Err(format!("missing tensor: {}", name));
                }
                let dt = aether_gguf_get_tensor_dtype(h, idx);
                let n_elems = aether_gguf_get_tensor_n_elems(h, idx);
                if n_elems != expected_elems as i64 {
                    eprintln!("[bert] WARN tensor {} has {} elems, expected {}",
                        name, n_elems, expected_elems);
                }
                let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
                let n = n_elems as usize;
                let host: Vec<f32> = match dt {
                    0 => {
                        // F32 — direct cast.
                        let src = std::slice::from_raw_parts(dptr as *const f32, n);
                        src.to_vec()
                    }
                    1 => {
                        // F16 — dequant on CPU.
                        let src = std::slice::from_raw_parts(dptr as *const u16, n);
                        src.iter().map(|&h| aether_f16_to_f32(h as i32)).collect()
                    }
                    _ => {
                        aether_gguf_close(h);
                        return Err(format!("tensor {} has unsupported dtype {}", name, dt));
                    }
                };
                let h_buf = aether_dev_alloc_f32(n as c_int);
                aether_dev_h2d_f32(host.as_ptr() as i64, h_buf, n as c_int);
                bufs_to_free.push(h_buf);
                Ok(h_buf)
            };

            let d_model = cfg.d_model;
            let d_ff = cfg.d_ff;

            let word_embd = upload_tensor("token_embd.weight",
                cfg.vocab * d_model, &mut bufs_to_free)?;
            let pos_embd  = upload_tensor("position_embd.weight",
                cfg.max_pos * d_model, &mut bufs_to_free)?;
            let type_embd = upload_tensor("token_types.weight",
                cfg.n_token_types * d_model, &mut bufs_to_free)?;
            let pre_norm_g = upload_tensor("token_embd_norm.weight",
                d_model, &mut bufs_to_free)?;
            let pre_norm_b = upload_tensor("token_embd_norm.bias",
                d_model, &mut bufs_to_free)?;

            let mut blocks = Vec::with_capacity(cfg.n_layers);
            for b in 0..cfg.n_layers {
                let p = format!("blk.{}.", b);
                let blk = BertBlock {
                    w_q: upload_tensor(&format!("{}attn_q.weight", p),
                        d_model * d_model, &mut bufs_to_free)?,
                    w_k: upload_tensor(&format!("{}attn_k.weight", p),
                        d_model * d_model, &mut bufs_to_free)?,
                    w_v: upload_tensor(&format!("{}attn_v.weight", p),
                        d_model * d_model, &mut bufs_to_free)?,
                    w_o: upload_tensor(&format!("{}attn_output.weight", p),
                        d_model * d_model, &mut bufs_to_free)?,
                    b_q: upload_tensor(&format!("{}attn_q.bias", p),
                        d_model, &mut bufs_to_free)?,
                    b_k: upload_tensor(&format!("{}attn_k.bias", p),
                        d_model, &mut bufs_to_free)?,
                    b_v: upload_tensor(&format!("{}attn_v.bias", p),
                        d_model, &mut bufs_to_free)?,
                    b_o: upload_tensor(&format!("{}attn_output.bias", p),
                        d_model, &mut bufs_to_free)?,
                    attn_out_norm_g: upload_tensor(&format!("{}attn_output_norm.weight", p),
                        d_model, &mut bufs_to_free)?,
                    attn_out_norm_b: upload_tensor(&format!("{}attn_output_norm.bias", p),
                        d_model, &mut bufs_to_free)?,
                    w_ffn_up:   upload_tensor(&format!("{}ffn_up.weight", p),
                        d_model * d_ff, &mut bufs_to_free)?,
                    b_ffn_up:   upload_tensor(&format!("{}ffn_up.bias", p),
                        d_ff, &mut bufs_to_free)?,
                    w_ffn_down: upload_tensor(&format!("{}ffn_down.weight", p),
                        d_ff * d_model, &mut bufs_to_free)?,
                    b_ffn_down: upload_tensor(&format!("{}ffn_down.bias", p),
                        d_model, &mut bufs_to_free)?,
                    layer_out_norm_g: upload_tensor(&format!("{}layer_output_norm.weight", p),
                        d_model, &mut bufs_to_free)?,
                    layer_out_norm_b: upload_tensor(&format!("{}layer_output_norm.bias", p),
                        d_model, &mut bufs_to_free)?,
                };
                blocks.push(blk);
                if b % 6 == 5 || b == cfg.n_layers - 1 {
                    eprintln!("[BertSession] loaded {} / {} blocks", b + 1, cfg.n_layers);
                }
            }

            let _ = d_ff;  // silence unused on the from_gguf path
            aether_gguf_close(h);
            Ok(Self {
                cfg, max_seq,
                word_embd, pos_embd, type_embd,
                pre_norm_g, pre_norm_b,
                blocks,
                bufs_to_free,
            })
        }
    }
}
