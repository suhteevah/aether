//! Measures: if we DON'T write new multi-token kernels and instead
//! just verify N candidate tokens by running the existing seq=1 graph
//! N times, what's the cost? This is the "speculative decoding without
//! multi-token kernels" lower-bound.
//!
//! Result interpretation: if N forward passes take ~N×26.8 ms each,
//! then to win at any acceptance rate r, we need
//!     N × 26.8 ms / (r × N) < 26.8 ms
//!     <=>  1/r < 1  <=>  r > 1
//! which is impossible (acceptance rate is at most 1.0). So naive
//! repeat-launch loses; we must amortize weight bandwidth across N
//! candidate tokens in a single forward pass.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;
use std::time::Instant;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m,
};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_alloc_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8,
    aether_dev_alloc_i32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_rms_norm_f32_cuda, aether_op_bias_add_f32_cuda,
    aether_op_add_inplace_f32_cuda,
    aether_op_rope_apply_devarg_f32_cuda,
    aether_op_append_kv_devarg_f32_cuda,
    aether_op_attention_seq1_devarg_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_dev_graph_begin, aether_dev_graph_end,
    aether_dev_graph_launch, aether_dev_graph_destroy,
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
const MAX_SEQ: usize = 64;

struct BlockGpu {
    attn_norm_g: i64, ffn_norm_g: i64,
    w_q: i64, w_k: i64, w_o: i64, w_gate: i64, w_up: i64,
    w_v: i64, dt_v: i32,
    w_down: i64, dt_down: i32,
    b_q: i64, b_k: i64, b_v: i64,
    nb_qo: usize, nb_kv: usize, nb_gate_up: usize, nb_down: usize,
}

struct ActivationGpu {
    x: i64, x_norm: i64,
    q: i64, k_step: i64, v_step: i64,
    attn_out: i64, proj: i64,
    gate: i64, down: i64,
    logits: i64,
}

struct KvCacheGpu { k_cache: i64, v_cache: i64 }

unsafe fn upload_tensor_u8(h: i64, name: &str) -> (i64, usize, i32) {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0, "{} not found", name);
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let n_blocks = n_elems / 256;
    let bytes_per_block = if dt == 12 { 144 } else if dt == 14 { 210 } else { panic!("dtype {}", dt) };
    let n_bytes = n_blocks * bytes_per_block;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let d_handle = aether_dev_alloc_u8(n_bytes as c_int);
    aether_dev_h2d_u8(dptr, d_handle, n_bytes as c_int);
    (d_handle, n_blocks, dt)
}

unsafe fn upload_f32_tensor(h: i64, name: &str) -> i64 {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0, "{} not found", name);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let host: Vec<f32> = std::slice::from_raw_parts(dptr as *const f32, n_elems).to_vec();
    let d = aether_dev_alloc_f32(n_elems as c_int);
    aether_dev_h2d_f32(host.as_ptr() as i64, d, n_elems as c_int);
    d
}

unsafe fn load_block(h: i64, b: usize) -> BlockGpu {
    let p = format!("blk.{}.", b);
    let (w_q, nb_qo, _)         = upload_tensor_u8(h, &format!("{}attn_q.weight", p));
    let (w_k, nb_kv, _)         = upload_tensor_u8(h, &format!("{}attn_k.weight", p));
    let (w_v, _, dt_v)          = upload_tensor_u8(h, &format!("{}attn_v.weight", p));
    let (w_o, _, _)             = upload_tensor_u8(h, &format!("{}attn_output.weight", p));
    let (w_gate, nb_gate_up, _) = upload_tensor_u8(h, &format!("{}ffn_gate.weight", p));
    let (w_up, _, _)            = upload_tensor_u8(h, &format!("{}ffn_up.weight", p));
    let (w_down, nb_down, dt_down) = upload_tensor_u8(h, &format!("{}ffn_down.weight", p));
    BlockGpu {
        attn_norm_g: upload_f32_tensor(h, &format!("{}attn_norm.weight", p)),
        ffn_norm_g:  upload_f32_tensor(h, &format!("{}ffn_norm.weight", p)),
        w_q, w_k, w_o, w_gate, w_up,
        w_v, dt_v, w_down, dt_down,
        b_q: upload_f32_tensor(h, &format!("{}attn_q.bias", p)),
        b_k: upload_f32_tensor(h, &format!("{}attn_k.bias", p)),
        b_v: upload_f32_tensor(h, &format!("{}attn_v.bias", p)),
        nb_qo, nb_kv, nb_gate_up, nb_down,
    }
}

unsafe fn dequant_embd_rows(h: i64, ids: &[usize]) -> Vec<f32> {
    let needle = b"token_embd.weight";
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let total_rows = n_elems / D_MODEL;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;
    let blocks_per_row = D_MODEL / 256;
    let bytes_per_row = blocks_per_row * 144;
    let total_bytes = total_rows * bytes_per_row;
    let bytes = std::slice::from_raw_parts(dptr, total_bytes);
    let mut out = vec![0.0f32; ids.len() * D_MODEL];
    for (oi, &id) in ids.iter().enumerate() {
        assert!(id < total_rows);
        let row_bytes = &bytes[id * bytes_per_row..(id + 1) * bytes_per_row];
        let mut row_f32 = vec![0.0f32; D_MODEL];
        aether_dequant_q4_k_m(
            row_bytes.as_ptr() as *const c_void,
            row_f32.as_mut_ptr() as *mut c_void,
            blocks_per_row as c_int,
        );
        out[oi * D_MODEL..(oi + 1) * D_MODEL].copy_from_slice(&row_f32);
    }
    out
}

unsafe fn block_forward(bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, step_args: i64) {
    aether_op_rms_norm_f32_cuda(act.x, bw.attn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_q, act.q,
        D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
    aether_op_bias_add_f32_cuda(act.q, bw.b_q, 1, D_MODEL as c_int);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_k, act.k_step,
        D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    aether_op_bias_add_f32_cuda(act.k_step, bw.b_k, 1, D_KV as c_int);
    if bw.dt_v == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step,
            D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step,
            D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    }
    aether_op_bias_add_f32_cuda(act.v_step, bw.b_v, 1, D_KV as c_int);
    aether_op_rope_apply_devarg_f32_cuda(act.q, 1, N_Q_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, step_args);
    aether_op_rope_apply_devarg_f32_cuda(act.k_step, 1, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, step_args);
    aether_op_append_kv_devarg_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache, D_KV as c_int, step_args);
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    aether_op_attention_seq1_devarg_f32_cuda(
        act.q, kv.k_cache, kv.v_cache, act.attn_out,
        N_Q_HEADS as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, scale,
        MAX_SEQ as c_int, step_args);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.attn_out, bw.w_o, act.proj,
        D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
    aether_op_add_inplace_f32_cuda(act.x, act.proj, D_MODEL as c_int);
    aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
        act.x_norm, bw.w_gate, bw.w_up, act.gate,
        D_FF as c_int, (bw.nb_gate_up / D_FF) as c_int);
    if bw.dt_down == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
            D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
            D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
    }
    aether_op_add_inplace_f32_cuda(act.x, act.down, D_MODEL as c_int);
}

#[test]
#[ignore]
fn spec_dec_naive_verify_cost() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);

        eprintln!("[loading 28 blocks]");
        let blocks: Vec<BlockGpu> = (0..N_LAYERS).map(|b| load_block(h, b)).collect();
        let final_norm_g = upload_f32_tensor(h, "output_norm.weight");
        let (lm_head, lm_n_blocks, lm_dt) = upload_tensor_u8(h, "output.weight");

        let act = ActivationGpu {
            x: aether_dev_alloc_f32(D_MODEL as c_int),
            x_norm: aether_dev_alloc_f32(D_MODEL as c_int),
            q: aether_dev_alloc_f32(D_MODEL as c_int),
            k_step: aether_dev_alloc_f32(D_KV as c_int),
            v_step: aether_dev_alloc_f32(D_KV as c_int),
            attn_out: aether_dev_alloc_f32(D_MODEL as c_int),
            proj: aether_dev_alloc_f32(D_MODEL as c_int),
            gate: aether_dev_alloc_f32(D_FF as c_int),
            down: aether_dev_alloc_f32(D_MODEL as c_int),
            logits: aether_dev_alloc_f32(VOCAB as c_int),
        };
        let kvs: Vec<KvCacheGpu> = (0..N_LAYERS).map(|_| KvCacheGpu {
            k_cache: aether_dev_alloc_f32((MAX_SEQ * D_KV) as c_int),
            v_cache: aether_dev_alloc_f32((MAX_SEQ * D_KV) as c_int),
        }).collect();
        let step_args = aether_dev_alloc_i32(4);

        // Prefill 4 tokens so KV cache is populated
        let prompt = [9707usize, 11, 1879, 0];
        for (step, &t_id) in prompt.iter().enumerate() {
            let emb = dequant_embd_rows(h, &[t_id]);
            aether_dev_h2d_f32(emb.as_ptr() as i64, act.x, D_MODEL as c_int);
            let step_host = [step as i32, step as i32 + 1, 0, 0];
            aether_dev_h2d_i32(step_host.as_ptr() as i64, step_args, 4);
            for b in 0..N_LAYERS { block_forward(&blocks[b], &act, &kvs[b], step_args); }
        }
        aether_dev_sync();

        // Capture single-token forward into graph
        let pos = prompt.len() as i32 - 1;
        let step_host = [pos, pos + 1, 0, 0];
        aether_dev_h2d_i32(step_host.as_ptr() as i64, step_args, 4);
        let emb = dequant_embd_rows(h, &[prompt[3]]);
        aether_dev_h2d_f32(emb.as_ptr() as i64, act.x, D_MODEL as c_int);
        aether_dev_sync();

        assert_eq!(0, aether_dev_graph_begin());
        for b in 0..N_LAYERS { block_forward(&blocks[b], &act, &kvs[b], step_args); }
        aether_op_rms_norm_f32_cuda(act.x, final_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
        if lm_dt == 14 {
            aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
        } else {
            aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
        }
        assert_eq!(0, aether_dev_graph_end());

        // Warmup
        for _ in 0..3 {
            assert_eq!(0, aether_dev_graph_launch());
        }
        aether_dev_sync();

        // Bench 1: single seq=1 forward (graph launch only)
        const N_ITERS: usize = 30;
        let t = Instant::now();
        for _ in 0..N_ITERS {
            assert_eq!(0, aether_dev_graph_launch());
        }
        aether_dev_sync();
        let single_ms = t.elapsed().as_secs_f64() * 1000.0 / N_ITERS as f64;

        // Bench 2: 4-token "verify" via 4 repeated launches
        let t = Instant::now();
        for _ in 0..N_ITERS {
            for _ in 0..4 {
                assert_eq!(0, aether_dev_graph_launch());
            }
        }
        aether_dev_sync();
        let four_ms = t.elapsed().as_secs_f64() * 1000.0 / N_ITERS as f64;

        // Bench 3: 8-token "verify"
        let t = Instant::now();
        for _ in 0..N_ITERS {
            for _ in 0..8 {
                assert_eq!(0, aether_dev_graph_launch());
            }
        }
        aether_dev_sync();
        let eight_ms = t.elapsed().as_secs_f64() * 1000.0 / N_ITERS as f64;

        eprintln!("\nNaive speculative-decoding verify cost (no multi-token kernels):\n");
        eprintln!("  seq=1 forward:  {:6.2} ms = {:5.1} tok/s", single_ms, 1000.0 / single_ms);
        eprintln!("  seq=4 verify:   {:6.2} ms ({:.2}x single)", four_ms, four_ms / single_ms);
        eprintln!("  seq=8 verify:   {:6.2} ms ({:.2}x single)", eight_ms, eight_ms / single_ms);
        eprintln!("");
        eprintln!("Break-even acceptance rate (for naive repeat-launch verify to match baseline):");
        eprintln!("  N=4: need accept r > {:.2}% to match single-token (impossible, r <= 1)", 100.0 * four_ms / (4.0 * single_ms));
        eprintln!("  N=8: need accept r > {:.2}% to match single-token (impossible)", 100.0 * eight_ms / (8.0 * single_ms));
        eprintln!("");
        eprintln!("Implication: speculative decoding REQUIRES multi-token attention +");
        eprintln!("matmul kernels (seq>1) to be a win. Naive repeat-launch verifies");
        eprintln!("scale linearly in N, so any acceptance rate <= 1 loses.");

        aether_dev_graph_destroy();
        aether_gguf_close(h);
    }
}
