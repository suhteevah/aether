//! Qwen2.5-7B autoregressive decode with CUDA-graph-captured per-step
//! forward. Captures once at the first decode step, replays for all
//! subsequent steps with just an h2d update of the device-side
//! step_args buffer (pos + cur_seq).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m,
};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_dev_alloc_i32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rope_apply_devarg_f32_cuda,
    aether_op_append_kv_devarg_f32_cuda,
    aether_op_attention_seq1_devarg_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_matmul_seq1_smallN_cuda,
    aether_op_fused_q6k_matmul_seq1_smallN_cuda,
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
const MAX_SEQ: usize = 32;

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

unsafe fn block_forward_devarg(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, step_args: i64,
) {
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
    aether_op_rope_apply_devarg_f32_cuda(act.q,
        1, N_Q_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, step_args);
    aether_op_rope_apply_devarg_f32_cuda(act.k_step,
        1, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, step_args);
    aether_op_append_kv_devarg_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache,
        D_KV as c_int, step_args);
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
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
fn qwen25_graph_decode_tok_per_sec() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);

        eprintln!("[loading 28 blocks ~3 GB]");
        let blocks: Vec<BlockGpu> = (0..N_LAYERS).map(|b| load_block(h, b)).collect();
        let final_norm_g = upload_f32_tensor(h, "output_norm.weight");
        let (lm_head, lm_n_blocks, lm_dt) = upload_tensor_u8(h, "output.weight");
        eprintln!("[lm_head dtype] {}", lm_dt);

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
        let step_args = aether_dev_alloc_i32(4);  // [pos, cur_seq, 0, 0]

        let prompt = [9707usize, 11, 1879, 0];
        let mut token_ids: Vec<usize> = prompt.to_vec();

        // --- PREFILL using devarg variants (no graph) ---
        for (step, &t_id) in prompt.iter().enumerate() {
            let emb = dequant_embd_rows(h, &[t_id]);
            aether_dev_h2d_f32(emb.as_ptr() as i64, act.x, D_MODEL as c_int);
            let pos = step as i32;
            let cur_seq = pos + 1;
            let step_host = [pos, cur_seq, 0i32, 0i32];
            aether_dev_h2d_i32(step_host.as_ptr() as i64, step_args, 4);
            for b in 0..N_LAYERS {
                block_forward_devarg(&blocks[b], &act, &kvs[b], step_args);
            }
        }
        aether_dev_sync();

        // --- Capture the per-step decode into a graph ---
        // We use the LAST prompt token's state as the input; the first
        // captured step will use whatever step_args is currently set to.
        // After capture, we update step_args per actual decode step.
        let n_gen = 5usize;
        let last_id_for_capture = token_ids.last().copied().unwrap();
        let emb_cap = dequant_embd_rows(h, &[last_id_for_capture]);
        aether_dev_h2d_f32(emb_cap.as_ptr() as i64, act.x, D_MODEL as c_int);
        // For capture, set step_args to the FIRST decode step's values.
        let cap_pos = token_ids.len() as i32 - 1;
        let cap_cur = cap_pos + 1;
        let step_cap_host = [cap_pos, cap_cur, 0i32, 0i32];
        aether_dev_h2d_i32(step_cap_host.as_ptr() as i64, step_args, 4);
        aether_dev_sync();

        let t_capture = std::time::Instant::now();
        assert_eq!(0, aether_dev_graph_begin(), "graph_begin failed");
        for b in 0..N_LAYERS {
            block_forward_devarg(&blocks[b], &act, &kvs[b], step_args);
        }
        aether_op_rms_norm_f32_cuda(act.x, final_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
        if lm_dt == 14 {
            aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
        } else {
            aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
        }
        assert_eq!(0, aether_dev_graph_end(), "graph_end failed");
        let cap_secs = t_capture.elapsed().as_secs_f32();
        eprintln!("[graph capture+instantiate] {:.3}s (one-time)", cap_secs);

        // The CAPTURE itself recorded the side effects of running the
        // first decode step. So the first generation token's logits are
        // already on device after the capture. We can either:
        //  (a) consume them now as the first generated token, or
        //  (b) re-launch the graph to do it again.
        // For a clean tok/s measurement, do (b): time only the graph
        // launches, not the capture.

        let t_gen = std::time::Instant::now();
        for step in 0..n_gen {
            // Compute new pos/cur_seq, h2d the step_args.
            let pos = token_ids.len() as i32 - 1;
            let cur_seq = pos + 1;
            let step_host = [pos, cur_seq, 0i32, 0i32];
            aether_dev_h2d_i32(step_host.as_ptr() as i64, step_args, 4);

            // Update x to embedding of the last token (may be the prompt
            // last token for step=0, or the previous gen token for later).
            let last_id = token_ids.last().copied().unwrap();
            let emb = dequant_embd_rows(h, &[last_id]);
            aether_dev_h2d_f32(emb.as_ptr() as i64, act.x, D_MODEL as c_int);

            assert_eq!(0, aether_dev_graph_launch(), "graph_launch failed");
            aether_dev_sync();

            let mut logits = vec![0.0f32; VOCAB];
            aether_dev_d2h_f32(act.logits, logits.as_mut_ptr() as i64, VOCAB as c_int);
            let (best_id, _) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            token_ids.push(best_id);
            let _ = step;
        }
        let gen_secs = t_gen.elapsed().as_secs_f32();
        eprintln!("[graph decode {} tokens] {:.3}s = {:.1} ms/token = {:.2} tok/s",
            n_gen, gen_secs, gen_secs * 1000.0 / n_gen as f32, n_gen as f32 / gen_secs);
        eprintln!("[generated IDs] {:?}", &token_ids[prompt.len()..]);

        aether_dev_graph_destroy();
        aether_gguf_close(h);
    }
}
