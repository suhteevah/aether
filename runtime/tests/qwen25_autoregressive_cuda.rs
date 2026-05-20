//! GPU-routed autoregressive generation through real Qwen2.5-7B.
//!
//! Same shape as `qwen25_autoregressive_gen.rs` but every matmul
//! is routed through cuBLAS via the per-call host-pointer wrapper:
//!   dev_alloc -> h2d(input) -> h2d(weight) -> sgemm -> d2h(out) -> free
//!
//! Other ops (RMSNorm/RoPE/GQA/SiLU/attention) stay on CPU. The
//! matmuls dominate cost so even with per-call h2d/d2h overhead the
//! GPU path should massively beat pure CPU:
//!   - lm_head: 545M ops, ~14s on CPU vs ~5ms on cuBLAS
//!   - per-block FFN matmuls: ~136-271M ops each × 3 per block ×
//!     28 blocks = the dominant cost, fully routed
//!
//! Expected: per-token cost ~5-10s (vs 53s on CPU), 5-10× speedup.
//! Marked #[ignore]; explicit invocation:
//!   cargo test -p aether_rt --release --features cuda \
//!     --test qwen25_autoregressive_cuda -- --ignored --nocapture

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m, aether_dequant_q6_k,
    aether_op_rms_norm_f32, aether_op_rope_apply_f32,
    aether_op_gqa_repeat_kv_f32,
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

/// Per-call cuBLAS matmul: alloc / h2d / sgemm / d2h / free.
/// Drop-in replacement for the CPU `aether_op_matmul_f32` shape.
unsafe fn matmul_via_cublas(
    a: *const f32, b: *const f32, out: *mut f32,
    m: usize, k: usize, n: usize,
) {
    use aether_rt::cuda::{
        aether_dev_alloc_f32, aether_dev_h2d_f32, aether_dev_d2h_f32,
        aether_dev_free_f32, aether_op_matmul_f32_cuda,
    };
    let total_a = (m * k) as c_int;
    let total_b = (k * n) as c_int;
    let total_o = (m * n) as c_int;
    let da = aether_dev_alloc_f32(total_a);
    let db = aether_dev_alloc_f32(total_b);
    let dout = aether_dev_alloc_f32(total_o);
    assert!(da > 0 && db > 0 && dout > 0, "cuda alloc failed");
    aether_dev_h2d_f32(a as i64, da, total_a);
    aether_dev_h2d_f32(b as i64, db, total_b);
    aether_op_matmul_f32_cuda(da, db, dout, m as c_int, k as c_int, n as c_int);
    aether_dev_d2h_f32(dout, out as i64, total_o);
    aether_dev_free_f32(da);
    aether_dev_free_f32(db);
    aether_dev_free_f32(dout);
}

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

struct KvCache {
    k: Vec<f32>, v: Vec<f32>, past_seq: usize,
}
impl KvCache {
    fn new() -> Self { Self { k: Vec::new(), v: Vec::new(), past_seq: 0 } }
    fn append(&mut self, new_k: &[f32], new_v: &[f32], n_new: usize) {
        self.k.extend_from_slice(new_k);
        self.v.extend_from_slice(new_v);
        self.past_seq += n_new;
    }
}

/// Same shape as block_forward_kv in the CPU test, but matmul calls
/// route through cuBLAS. Non-matmul ops stay on CPU.
unsafe fn block_forward_kv_cuda(
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
    matmul_via_cublas(x_norm.as_ptr(), bw.w_q.as_ptr(), q.as_mut_ptr(), seq, D_MODEL, D_MODEL);
    matmul_via_cublas(x_norm.as_ptr(), bw.w_k.as_ptr(), k_new.as_mut_ptr(), seq, D_MODEL, D_KV);
    matmul_via_cublas(x_norm.as_ptr(), bw.w_v.as_ptr(), v_new.as_mut_ptr(), seq, D_MODEL, D_KV);
    add_bias(&mut q, &bw.b_q, seq, D_MODEL);
    add_bias(&mut k_new, &bw.b_k, seq, D_KV);
    add_bias(&mut v_new, &bw.b_v, seq, D_KV);

    aether_op_rope_apply_f32(q.as_mut_ptr() as _, seq as c_int, N_Q_HEADS as c_int, HEAD_DIM as c_int,
        ROPE_BASE, pos_start as c_int);
    aether_op_rope_apply_f32(k_new.as_mut_ptr() as _, seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int,
        ROPE_BASE, pos_start as c_int);

    cache.append(&k_new, &v_new, seq);
    let total_seq = cache.past_seq;

    let mut k_full = vec![0.0f32; total_seq * D_MODEL];
    let mut v_full = vec![0.0f32; total_seq * D_MODEL];
    aether_op_gqa_repeat_kv_f32(cache.k.as_ptr() as _, k_full.as_mut_ptr() as _,
        total_seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int);
    aether_op_gqa_repeat_kv_f32(cache.v.as_ptr() as _, v_full.as_mut_ptr() as _,
        total_seq as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, N_Q_HEADS as c_int);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut attn = vec![0.0f32; seq * D_MODEL];
    for r in 0..seq {
        let abs_pos = pos_start + r;
        for hh in 0..N_Q_HEADS {
            let q_off = (r * N_Q_HEADS + hh) * HEAD_DIM;
            let mut scores = vec![f32::NEG_INFINITY; total_seq];
            for t in 0..=abs_pos {
                let k_off = (t * N_Q_HEADS + hh) * HEAD_DIM;
                let mut s = 0.0f32;
                for d in 0..HEAD_DIM { s += q[q_off + d] * k_full[k_off + d]; }
                scores[t] = s * scale;
            }
            let mx = scores[..=abs_pos].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            let mut exps = vec![0.0f32; total_seq];
            for t in 0..=abs_pos {
                exps[t] = (scores[t] - mx).exp();
                sum_exp += exps[t];
            }
            let inv = 1.0 / sum_exp;
            let out_off = (r * N_Q_HEADS + hh) * HEAD_DIM;
            for d in 0..HEAD_DIM { attn[out_off + d] = 0.0; }
            for t in 0..=abs_pos {
                let v_off = (t * N_Q_HEADS + hh) * HEAD_DIM;
                let w = exps[t] * inv;
                for d in 0..HEAD_DIM { attn[out_off + d] += w * v_full[v_off + d]; }
            }
        }
    }

    let mut proj = vec![0.0f32; seq * D_MODEL];
    matmul_via_cublas(attn.as_ptr(), bw.w_o.as_ptr(), proj.as_mut_ptr(), seq, D_MODEL, D_MODEL);
    for i in 0..(seq * D_MODEL) { x[i] += proj[i]; }

    aether_op_rms_norm_f32(x.as_ptr() as _, bw.ffn_norm_g.as_ptr() as _, NORM_EPS,
        x_norm.as_mut_ptr() as _, seq as c_int, D_MODEL as c_int);

    let mut gate = vec![0.0f32; seq * D_FF];
    let mut up = vec![0.0f32; seq * D_FF];
    matmul_via_cublas(x_norm.as_ptr(), bw.w_gate.as_ptr(), gate.as_mut_ptr(), seq, D_MODEL, D_FF);
    matmul_via_cublas(x_norm.as_ptr(), bw.w_up.as_ptr(), up.as_mut_ptr(), seq, D_MODEL, D_FF);
    for i in 0..(seq * D_FF) {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        gate[i] = silu_g * up[i];
    }
    let mut down = vec![0.0f32; seq * D_MODEL];
    matmul_via_cublas(gate.as_ptr(), bw.w_down.as_ptr(), down.as_mut_ptr(), seq, D_FF, D_MODEL);
    for i in 0..(seq * D_MODEL) { x[i] += down[i]; }
}

#[test]
#[ignore]
fn qwen25_autoregressive_cuda_5_tokens() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    let t_total = std::time::Instant::now();
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // Warm cuda context out of the timed region.
        aether_rt::cuda::aether_dev_init();

        let t = std::time::Instant::now();
        let blocks: Vec<BlockWeights> = (0..N_LAYERS).map(|b| load_block(h, b)).collect();
        eprintln!("[all blocks loaded] {:.2}s -- {} blocks", t.elapsed().as_secs_f32(), N_LAYERS);

        let final_norm_g = load_tensor_f32(h, "output_norm.weight");
        let t = std::time::Instant::now();
        let lm_head_gguf = load_tensor_f32(h, "output.weight");
        eprintln!("[lm_head load] {:.2}s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let lm_head = transpose_weight(&lm_head_gguf, VOCAB, D_MODEL);
        drop(lm_head_gguf);
        eprintln!("[lm_head xpose] {:.2}s", t.elapsed().as_secs_f32());

        let mut caches: Vec<KvCache> = (0..N_LAYERS).map(|_| KvCache::new()).collect();

        let mut token_ids: Vec<usize> = vec![9707, 11, 1879, 0];
        let prompt_len = token_ids.len();
        let n_gen = 5;

        let t_prefill = std::time::Instant::now();
        {
            let mut x = dequant_embd_rows(h, &token_ids);
            for b in 0..N_LAYERS {
                block_forward_kv_cuda(&blocks[b], &mut caches[b], &mut x, prompt_len, 0);
            }
            let mut x_final = vec![0.0f32; prompt_len * D_MODEL];
            aether_op_rms_norm_f32(x.as_ptr() as _, final_norm_g.as_ptr() as _, NORM_EPS,
                x_final.as_mut_ptr() as _, prompt_len as c_int, D_MODEL as c_int);
            let last_x = &x_final[(prompt_len - 1) * D_MODEL..];
            let mut logits = vec![0.0f32; VOCAB];
            matmul_via_cublas(last_x.as_ptr(), lm_head.as_ptr(), logits.as_mut_ptr(),
                1, D_MODEL, VOCAB);
            let (best_id, &best_val) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            assert!(best_val.is_finite() && best_id < VOCAB);
            eprintln!("[prefill] {:.2}s -- next_id={} logit={:.3}",
                t_prefill.elapsed().as_secs_f32(), best_id, best_val);
            token_ids.push(best_id);
        }

        for step in 1..n_gen {
            let t_step = std::time::Instant::now();
            let new_id = token_ids.last().copied().unwrap();
            let mut x = dequant_embd_rows(h, &[new_id]);
            let pos_start = token_ids.len() - 1;
            for b in 0..N_LAYERS {
                block_forward_kv_cuda(&blocks[b], &mut caches[b], &mut x, 1, pos_start);
            }
            let mut x_final = vec![0.0f32; D_MODEL];
            aether_op_rms_norm_f32(x.as_ptr() as _, final_norm_g.as_ptr() as _, NORM_EPS,
                x_final.as_mut_ptr() as _, 1, D_MODEL as c_int);
            let mut logits = vec![0.0f32; VOCAB];
            matmul_via_cublas(x_final.as_ptr(), lm_head.as_ptr(), logits.as_mut_ptr(),
                1, D_MODEL, VOCAB);
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

        for &id in &token_ids { assert!(id < VOCAB); }
    }
}
