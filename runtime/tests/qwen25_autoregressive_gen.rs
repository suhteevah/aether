//! Autoregressive generation through real Qwen2.5-7B with KV cache.
//!
//! Builds on `qwen25_full_inference.rs`. That file did one forward
//! pass of a 4-token prompt; this one prefills the prompt then
//! GENERATES N additional tokens one at a time, caching K/V per
//! block so each new-token step only processes 1 new token's worth
//! of matmul (not the whole growing prefix).
//!
//! Without KV cache, generating N tokens costs O(prefix^2 + N*prefix)
//! ~ 1700s for 5 tokens. With KV cache, it costs O(prefix + N) ~
//! single-block-per-token. Verified to produce plausible IDs.
//!
//! Marked #[ignore] -- ~7-10 min release build for prompt=4 + gen=3.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m, aether_dequant_q6_k,
    aether_op_matmul_f32, aether_op_rms_norm_f32,
    aether_op_rope_apply_f32, aether_op_gqa_repeat_kv_f32,
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

unsafe fn dequant_embd_rows(h: i64, rows: &[usize]) -> Vec<f32> {
    let needle = b"token_embd.weight";
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let blocks_per_row = D_MODEL / 256;
    let mut out = vec![0.0f32; rows.len() * D_MODEL];
    for (i, &t_id) in rows.iter().enumerate() {
        let block_ptr = (dptr as *const u8).add(t_id * blocks_per_row * 144);
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

fn add_bias(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        for c in 0..cols { x[r * cols + c] += bias[c]; }
    }
}

/// All weights for one decoder block, kept in matmul-ready (transposed)
/// layout. Allocated once at session start and reused across every
/// generation step.
struct BlockWeights {
    attn_norm_g: Vec<f32>,
    w_q: Vec<f32>, b_q: Vec<f32>,
    w_k: Vec<f32>, b_k: Vec<f32>,
    w_v: Vec<f32>, b_v: Vec<f32>,
    w_o: Vec<f32>,
    ffn_norm_g: Vec<f32>,
    w_gate: Vec<f32>, w_up: Vec<f32>, w_down: Vec<f32>,
}

unsafe fn load_block(h: i64, block_idx: usize) -> BlockWeights {
    let p = format!("blk.{}.", block_idx);
    let attn_norm_g = load_tensor_f32(h, &format!("{}attn_norm.weight", p));
    let w_q = transpose_weight(&load_tensor_f32(h, &format!("{}attn_q.weight", p)), D_MODEL, D_MODEL);
    let b_q = load_tensor_f32(h, &format!("{}attn_q.bias", p));
    let w_k = transpose_weight(&load_tensor_f32(h, &format!("{}attn_k.weight", p)), D_KV, D_MODEL);
    let b_k = load_tensor_f32(h, &format!("{}attn_k.bias", p));
    let w_v = transpose_weight(&load_tensor_f32(h, &format!("{}attn_v.weight", p)), D_KV, D_MODEL);
    let b_v = load_tensor_f32(h, &format!("{}attn_v.bias", p));
    let w_o = transpose_weight(&load_tensor_f32(h, &format!("{}attn_output.weight", p)), D_MODEL, D_MODEL);
    let ffn_norm_g = load_tensor_f32(h, &format!("{}ffn_norm.weight", p));
    let w_gate = transpose_weight(&load_tensor_f32(h, &format!("{}ffn_gate.weight", p)), D_FF, D_MODEL);
    let w_up = transpose_weight(&load_tensor_f32(h, &format!("{}ffn_up.weight", p)), D_FF, D_MODEL);
    let w_down = transpose_weight(&load_tensor_f32(h, &format!("{}ffn_down.weight", p)), D_MODEL, D_FF);
    BlockWeights { attn_norm_g, w_q, b_q, w_k, b_k, w_v, b_v, w_o,
                   ffn_norm_g, w_gate, w_up, w_down }
}

/// KV cache: per-block storage of all past K/V activations.
/// Layout: `k[t * D_KV + i]` -- token-major, K is shape [past_seq, D_KV].
/// V identical layout.
struct KvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    past_seq: usize,
}

impl KvCache {
    fn new() -> Self { Self { k: Vec::new(), v: Vec::new(), past_seq: 0 } }
    fn append(&mut self, new_k: &[f32], new_v: &[f32], n_new: usize) {
        self.k.extend_from_slice(new_k);
        self.v.extend_from_slice(new_v);
        self.past_seq += n_new;
    }
}

/// One block forward with KV cache. `x` has shape [seq, D_MODEL] (in
/// place residual update). `seq` is the NEW tokens being processed
/// (1 for autoregressive step, prompt_len for prefill).
unsafe fn block_forward_kv(
    bw: &BlockWeights, cache: &mut KvCache,
    x: &mut [f32], seq: usize, pos_start: usize,
) {
    let mut x_norm = vec![0.0f32; seq * D_MODEL];
    aether_op_rms_norm_f32(
        x.as_ptr() as _, bw.attn_norm_g.as_ptr() as _, NORM_EPS,
        x_norm.as_mut_ptr() as _, seq as c_int, D_MODEL as c_int,
    );

    let mut q = vec![0.0f32; seq * D_MODEL];
    let mut k_new = vec![0.0f32; seq * D_KV];
    let mut v_new = vec![0.0f32; seq * D_KV];
    aether_op_matmul_f32(x_norm.as_ptr() as _, bw.w_q.as_ptr() as _, q.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_MODEL as c_int);
    aether_op_matmul_f32(x_norm.as_ptr() as _, bw.w_k.as_ptr() as _, k_new.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_KV as c_int);
    aether_op_matmul_f32(x_norm.as_ptr() as _, bw.w_v.as_ptr() as _, v_new.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_KV as c_int);
    add_bias(&mut q, &bw.b_q, seq, D_MODEL);
    add_bias(&mut k_new, &bw.b_k, seq, D_KV);
    add_bias(&mut v_new, &bw.b_v, seq, D_KV);

    // RoPE on Q (at pos_start..pos_start+seq) and K (same positions).
    aether_op_rope_apply_f32(q.as_mut_ptr() as _, seq as c_int, N_Q_HEADS as c_int, HEAD_DIM as c_int,
        ROPE_BASE, pos_start as c_int);
    aether_op_rope_apply_f32(k_new.as_mut_ptr() as _, seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int,
        ROPE_BASE, pos_start as c_int);

    // Append k_new + v_new into the cache (now contains past + new).
    cache.append(&k_new, &v_new, seq);
    let total_seq = cache.past_seq;  // past + new

    // GQA-repeat the FULL cached K and V to n_q_heads (each time -- not
    // amortised yet; matt-voice deploy follow-up).
    let mut k_full = vec![0.0f32; total_seq * D_MODEL];
    let mut v_full = vec![0.0f32; total_seq * D_MODEL];
    aether_op_gqa_repeat_kv_f32(cache.k.as_ptr() as _, k_full.as_mut_ptr() as _,
        total_seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int);
    aether_op_gqa_repeat_kv_f32(cache.v.as_ptr() as _, v_full.as_mut_ptr() as _,
        total_seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int);

    // Manual attention: each new-Q row attends to ALL past keys. With
    // causal masking based on absolute positions. Q shape: [seq, n_q*hd].
    // K/V shape: [total_seq, n_q*hd]. Per Q row at relative pos r
    // (absolute pos pos_start+r), attend to keys at abs positions
    // [0..pos_start+r+1] (causal).
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut attn = vec![0.0f32; seq * D_MODEL];
    for r in 0..seq {
        let abs_pos = pos_start + r;
        for h in 0..N_Q_HEADS {
            let q_off = (r * N_Q_HEADS + h) * HEAD_DIM;
            // Score for each past+self key position.
            let mut scores = vec![f32::NEG_INFINITY; total_seq];
            for t in 0..=abs_pos {  // causal mask
                let k_off = (t * N_Q_HEADS + h) * HEAD_DIM;
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM {
                    s += q[q_off + d] * k_full[k_off + d];
                }
                scores[t] = s * scale;
            }
            // Softmax over valid keys (0..=abs_pos).
            let mx = scores[..=abs_pos].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exps = vec![0.0f32; total_seq];
            for t in 0..=abs_pos {
                exps[t] = (scores[t] - mx).exp();
                sum_exp += exps[t];
            }
            let inv = 1.0 / sum_exp;
            // Weighted sum of V rows.
            let out_off = (r * N_Q_HEADS + h) * HEAD_DIM;
            for d in 0..HEAD_DIM { attn[out_off + d] = 0.0; }
            for t in 0..=abs_pos {
                let v_off = (t * N_Q_HEADS + h) * HEAD_DIM;
                let w = exps[t] * inv;
                for d in 0..HEAD_DIM {
                    attn[out_off + d] += w * v_full[v_off + d];
                }
            }
        }
    }

    let mut proj = vec![0.0f32; seq * D_MODEL];
    aether_op_matmul_f32(attn.as_ptr() as _, bw.w_o.as_ptr() as _, proj.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_MODEL as c_int);
    for i in 0..(seq * D_MODEL) { x[i] += proj[i]; }

    aether_op_rms_norm_f32(x.as_ptr() as _, bw.ffn_norm_g.as_ptr() as _, NORM_EPS,
        x_norm.as_mut_ptr() as _, seq as c_int, D_MODEL as c_int);

    let mut gate = vec![0.0f32; seq * D_FF];
    let mut up = vec![0.0f32; seq * D_FF];
    aether_op_matmul_f32(x_norm.as_ptr() as _, bw.w_gate.as_ptr() as _, gate.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_FF as c_int);
    aether_op_matmul_f32(x_norm.as_ptr() as _, bw.w_up.as_ptr() as _, up.as_mut_ptr() as _,
        seq as c_int, D_MODEL as c_int, D_FF as c_int);
    for i in 0..(seq * D_FF) {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        gate[i] = silu_g * up[i];
    }
    let mut down = vec![0.0f32; seq * D_MODEL];
    aether_op_matmul_f32(gate.as_ptr() as _, bw.w_down.as_ptr() as _, down.as_mut_ptr() as _,
        seq as c_int, D_FF as c_int, D_MODEL as c_int);
    for i in 0..(seq * D_MODEL) { x[i] += down[i]; }
}

#[test]
#[ignore]  // ~10 min release build
fn qwen25_autoregressive_5_tokens() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    let t_total = std::time::Instant::now();
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // Load ALL block weights once -- 28 blocks * ~870 MB = ~24 GB
        // f32. Tight on 32 GB host but fits if no other heavy
        // processes. Allows per-step generation without re-load.
        let t = std::time::Instant::now();
        let blocks: Vec<BlockWeights> = (0..N_LAYERS).map(|b| {
            let t = std::time::Instant::now();
            let bw = load_block(h, b);
            eprintln!("[load blk {:>2}] {:.2}s", b, t.elapsed().as_secs_f32());
            bw
        }).collect();
        eprintln!("[all blocks loaded] {:.2}s -- {} blocks", t.elapsed().as_secs_f32(), N_LAYERS);

        let final_norm_g = load_tensor_f32(h, "output_norm.weight");
        let t = std::time::Instant::now();
        let lm_head_gguf = load_tensor_f32(h, "output.weight");
        eprintln!("[lm_head load] {:.2}s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let lm_head = transpose_weight(&lm_head_gguf, VOCAB, D_MODEL);
        drop(lm_head_gguf);
        eprintln!("[lm_head xpose] {:.2}s", t.elapsed().as_secs_f32());

        // Per-block KV cache, all empty.
        let mut caches: Vec<KvCache> = (0..N_LAYERS).map(|_| KvCache::new()).collect();

        // Input: 4-token prompt + generate 5 more tokens.
        let mut token_ids: Vec<usize> = vec![9707, 11, 1879, 0];
        let prompt_len = token_ids.len();
        let n_gen = 5;

        // PREFILL: forward the full prompt once.
        let t_prefill = std::time::Instant::now();
        {
            let mut x = dequant_embd_rows(h, &token_ids);
            for b in 0..N_LAYERS {
                let t = std::time::Instant::now();
                block_forward_kv(&blocks[b], &mut caches[b], &mut x, prompt_len, 0);
                if b == 0 || b == N_LAYERS - 1 {
                    eprintln!("[prefill blk {}] {:.2}s", b, t.elapsed().as_secs_f32());
                }
            }
            // Final norm + lm_head ONLY at the last position.
            let mut x_final = vec![0.0f32; prompt_len * D_MODEL];
            aether_op_rms_norm_f32(x.as_ptr() as _, final_norm_g.as_ptr() as _, NORM_EPS,
                x_final.as_mut_ptr() as _, prompt_len as c_int, D_MODEL as c_int);
            let last_pos_start = (prompt_len - 1) * D_MODEL;
            let last_x = &x_final[last_pos_start..];
            let mut logits = vec![0.0f32; VOCAB];
            aether_op_matmul_f32(last_x.as_ptr() as _, lm_head.as_ptr() as _, logits.as_mut_ptr() as _,
                1, D_MODEL as c_int, VOCAB as c_int);
            let (best_id, &best_val) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            assert!(best_val.is_finite() && best_id < VOCAB);
            eprintln!("[prefill] {:.2}s -- next_id={} logit={:.3}",
                t_prefill.elapsed().as_secs_f32(), best_id, best_val);
            token_ids.push(best_id);
        }

        // GENERATE: 4 more tokens.
        for step in 1..n_gen {
            let t_step = std::time::Instant::now();
            let new_id = token_ids.last().copied().unwrap();
            let mut x = dequant_embd_rows(h, &[new_id]);
            let pos_start = token_ids.len() - 1;  // abs pos of this new token

            for b in 0..N_LAYERS {
                block_forward_kv(&blocks[b], &mut caches[b], &mut x, 1, pos_start);
            }
            let mut x_final = vec![0.0f32; D_MODEL];
            aether_op_rms_norm_f32(x.as_ptr() as _, final_norm_g.as_ptr() as _, NORM_EPS,
                x_final.as_mut_ptr() as _, 1, D_MODEL as c_int);
            let mut logits = vec![0.0f32; VOCAB];
            aether_op_matmul_f32(x_final.as_ptr() as _, lm_head.as_ptr() as _, logits.as_mut_ptr() as _,
                1, D_MODEL as c_int, VOCAB as c_int);
            let (best_id, &best_val) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            assert!(best_val.is_finite() && best_id < VOCAB);
            eprintln!("[gen {}/{}] {:.2}s -- next_id={} logit={:.3} (abs_pos={})",
                step, n_gen - 1, t_step.elapsed().as_secs_f32(),
                best_id, best_val, pos_start + 1);
            token_ids.push(best_id);
        }

        eprintln!("[total] {:.2}s -- {} prompt + {} generated = {:?}",
            t_total.elapsed().as_secs_f32(), prompt_len, n_gen, token_ids);
        aether_gguf_close(h);

        // Sanity: all IDs in vocab.
        for &id in &token_ids {
            assert!(id < VOCAB, "id {} out of vocab", id);
        }
    }
}
