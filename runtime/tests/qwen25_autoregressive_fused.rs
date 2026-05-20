//! Real tok/s measurement: full Qwen2.5-7B autoregressive with the
//! fused Q4_K v2 + Q6_K v2 kernels wired in.
//!
//! All weights resident on device (~3 GB Q4_K + Q6_K bytes, fits in
//! 8 GB VRAM with plenty of headroom). Per-token forward uses the
//! v2 fused matmul kernels + the on-device RMSNorm/RoPE/GQA/SiLU
//! shipped earlier this session.

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
    aether_op_rms_norm_f32_cuda, aether_op_rope_apply_f32_cuda,
    aether_op_gqa_repeat_kv_f32_cuda, aether_op_silu_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_matmul_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_op_append_kv_f32_cuda, aether_op_attention_seq1_f32_cuda,
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

/// Device handles for one decoder block's weights, all resident.
/// Each weight tensor carries its GGUF dtype (12 = Q4_K, 14 = Q6_K)
/// so block_forward can dispatch the right fused matmul kernel.
/// In Qwen2.5-7B Q4_K_M, V proj and ffn_down are mixed-precision:
/// some blocks use Q4_K, others Q6_K. The rest are uniformly Q4_K.
struct BlockGpu {
    // Norms (F32, small)
    attn_norm_g: i64,
    ffn_norm_g: i64,
    // Q4_K weights (always)
    w_q: i64, w_k: i64, w_o: i64, w_gate: i64, w_up: i64,
    // V and Down: dtype varies per block
    w_v: i64,    dt_v: i32,
    w_down: i64, dt_down: i32,
    // Biases (F32, small)
    b_q: i64, b_k: i64, b_v: i64,
    // n_blocks shapes
    nb_qo: usize,
    nb_kv: usize,
    nb_gate_up: usize,
    nb_down: usize,
}

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
    assert!(idx >= 0);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let d = aether_dev_alloc_f32(n_elems as c_int);
    aether_dev_h2d_f32(dptr, d, n_elems as c_int);
    d
}

unsafe fn load_block_to_device(h: i64, block_idx: usize) -> BlockGpu {
    let prefix = format!("blk.{}.", block_idx);
    let attn_norm_g = upload_f32_tensor(h, &format!("{}attn_norm.weight", prefix));
    let ffn_norm_g  = upload_f32_tensor(h, &format!("{}ffn_norm.weight", prefix));
    let (w_q,    nb_qo,      _)       = upload_tensor_u8(h, &format!("{}attn_q.weight", prefix));
    let (w_k,    nb_kv,      _)       = upload_tensor_u8(h, &format!("{}attn_k.weight", prefix));
    let (w_v,    _,          dt_v)    = upload_tensor_u8(h, &format!("{}attn_v.weight", prefix));
    let (w_o,    _,          _)       = upload_tensor_u8(h, &format!("{}attn_output.weight", prefix));
    let (w_gate, nb_gate_up, _)       = upload_tensor_u8(h, &format!("{}ffn_gate.weight", prefix));
    let (w_up,   _,          _)       = upload_tensor_u8(h, &format!("{}ffn_up.weight", prefix));
    let (w_down, nb_down,    dt_down) = upload_tensor_u8(h, &format!("{}ffn_down.weight", prefix));
    let b_q = upload_f32_tensor(h, &format!("{}attn_q.bias", prefix));
    let b_k = upload_f32_tensor(h, &format!("{}attn_k.bias", prefix));
    let b_v = upload_f32_tensor(h, &format!("{}attn_v.bias", prefix));
    BlockGpu { attn_norm_g, ffn_norm_g, w_q, w_k, w_o, w_gate, w_up,
               w_v, dt_v, w_down, dt_down,
               b_q, b_k, b_v, nb_qo, nb_kv, nb_gate_up, nb_down }
}

unsafe fn dequant_embd_rows(h: i64, rows: &[usize]) -> Vec<f32> {
    let needle = b"token_embd.weight";
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let blocks_per_row = D_MODEL / 256;
    let mut out = vec![0.0f32; rows.len() * D_MODEL];
    for (i, &t_id) in rows.iter().enumerate() {
        let block_ptr = (dptr as *const u8).add(t_id * blocks_per_row * 144);
        aether_dequant_q4_k_m(block_ptr as *const c_void,
            out[i * D_MODEL..(i + 1) * D_MODEL].as_mut_ptr() as *mut c_void,
            blocks_per_row as c_int);
    }
    out
}

/// Activation scratch buffers on device, reused across blocks + steps.
struct ActivationGpu {
    x: i64,            // D_MODEL
    x_norm: i64,       // D_MODEL
    q: i64,            // D_MODEL
    k_step: i64,       // D_KV (new K for this step)
    v_step: i64,       // D_KV (new V for this step)
    attn_out: i64,     // D_MODEL
    proj: i64,         // D_MODEL
    gate: i64,         // D_FF
    up: i64,           // D_FF
    down: i64,         // D_MODEL
    logits: i64,       // VOCAB (only for final lm_head step)
}

const MAX_SEQ: usize = 32;  // small for test; for full deploy bump to 4096

/// Per-block on-device KV cache. Allocated once before generation.
struct KvCacheGpu {
    k_cache: i64,  // [MAX_SEQ, D_KV]
    v_cache: i64,  // [MAX_SEQ, D_KV]
}

unsafe fn alloc_activations() -> ActivationGpu {
    ActivationGpu {
        x:          aether_dev_alloc_f32(D_MODEL as c_int),
        x_norm:     aether_dev_alloc_f32(D_MODEL as c_int),
        q:          aether_dev_alloc_f32(D_MODEL as c_int),
        k_step:     aether_dev_alloc_f32(D_KV as c_int),
        v_step:     aether_dev_alloc_f32(D_KV as c_int),
        attn_out:   aether_dev_alloc_f32(D_MODEL as c_int),
        proj:       aether_dev_alloc_f32(D_MODEL as c_int),
        gate:       aether_dev_alloc_f32(D_FF as c_int),
        up:         aether_dev_alloc_f32(D_FF as c_int),
        down:       aether_dev_alloc_f32(D_MODEL as c_int),
        logits:     aether_dev_alloc_f32(VOCAB as c_int),
    }
}

unsafe fn alloc_kv_caches() -> Vec<KvCacheGpu> {
    (0..N_LAYERS).map(|_| KvCacheGpu {
        k_cache: aether_dev_alloc_f32((MAX_SEQ * D_KV) as c_int),
        v_cache: aether_dev_alloc_f32((MAX_SEQ * D_KV) as c_int),
    }).collect()
}

/// One block forward for seq=1 (autoregressive step) with real
/// on-device KV cache attention. `pos` is the absolute token position
/// (0-indexed). `cur_seq = pos + 1` is the valid length in the cache
/// after appending this step's K/V.
/// Probe magnitude of a device buffer (debug helper).
unsafe fn probe(label: &str, handle: i64, n: usize, enabled: bool) {
    if !enabled { return; }
    aether_dev_sync();
    let mut host = vec![0.0f32; n];
    aether_dev_d2h_f32(handle, host.as_mut_ptr() as i64, n as c_int);
    let mut max_abs = 0.0f32;
    let mut any_nan = false;
    let mut any_inf = false;
    let mut sum = 0.0f32;
    let mut nonzero = 0;
    for &v in &host {
        if v.is_nan() { any_nan = true; continue; }
        if v.is_infinite() { any_inf = true; continue; }
        let a = v.abs();
        if a > max_abs { max_abs = a; }
        if a > 1e-10 { nonzero += 1; }
        sum += v;
    }
    eprintln!("    [{:<14}] max_abs={:.3e} sum={:.3e} nonzero={}/{} nan={} inf={}",
        label, max_abs, sum, nonzero, n, any_nan, any_inf);
}

unsafe fn block_forward_step(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, pos: i32,
) {
    block_forward_step_debug(bw, act, kv, pos, false);
}

unsafe fn block_forward_step_debug(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, pos: i32, dbg: bool,
) {
    if dbg { probe("x (block in)", act.x, D_MODEL, dbg); }
    // attn_norm
    aether_op_rms_norm_f32_cuda(act.x, bw.attn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
    if dbg { probe("attn_norm out", act.x_norm, D_MODEL, dbg); }
    // Q proj (Q4_K v2) + bias
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_q, act.q,
        D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
    if dbg { probe("Q proj", act.q, D_MODEL, dbg); }
    aether_op_bias_add_f32_cuda(act.q, bw.b_q, 1, D_MODEL as c_int);
    if dbg { probe("Q + bias", act.q, D_MODEL, dbg); }
    // K proj (Q4_K v2) + bias
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_k, act.k_step,
        D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    aether_op_bias_add_f32_cuda(act.k_step, bw.b_k, 1, D_KV as c_int);
    if dbg { probe("K + bias", act.k_step, D_KV, dbg); }
    // V proj (Q4_K or Q6_K depending on block) + bias
    if bw.dt_v == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step,
            D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step,
            D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    }
    aether_op_bias_add_f32_cuda(act.v_step, bw.b_v, 1, D_KV as c_int);
    if dbg { probe("V + bias", act.v_step, D_KV, dbg); }
    // RoPE on Q and K_step (K_step is the per-kv-head K we'll cache;
    // Q has full n_q_heads expanded already from the proj).
    aether_op_rope_apply_f32_cuda(act.q,      1, N_Q_HEADS  as c_int, HEAD_DIM as c_int, ROPE_BASE, pos);
    aether_op_rope_apply_f32_cuda(act.k_step, 1, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, pos);
    if dbg { probe("Q after RoPE", act.q, D_MODEL, dbg); }
    if dbg { probe("K after RoPE", act.k_step, D_KV, dbg); }
    // Append K_step / V_step to per-block KV cache at position `pos`.
    aether_op_append_kv_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache,
        pos, D_KV as c_int);
    // Real causal attention with on-device KV cache. cur_seq = pos + 1.
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    aether_op_attention_seq1_f32_cuda(
        act.q, kv.k_cache, kv.v_cache, act.attn_out,
        pos + 1, N_Q_HEADS as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int,
        scale,
    );
    if dbg { probe("attn_out", act.attn_out, D_MODEL, dbg); }
    // O proj (Q4_K v2)
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.attn_out, bw.w_o, act.proj,
        D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
    if dbg { probe("O proj", act.proj, D_MODEL, dbg); }
    // Residual
    aether_op_add_inplace_f32_cuda(act.x, act.proj, D_MODEL as c_int);
    if dbg { probe("x after res1", act.x, D_MODEL, dbg); }
    // ffn_norm
    aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
    if dbg { probe("ffn_norm out", act.x_norm, D_MODEL, dbg); }
    // FUSED gate+up+silu+mul (1 kernel replaces 4): writes silu(gate)*up
    // directly into act.gate.
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
        act.x_norm, bw.w_gate, bw.w_up, act.gate,
        D_FF as c_int, (bw.nb_gate_up / D_FF) as c_int);
    if dbg { probe("silu*up (fused)", act.gate, D_FF, dbg); }
    // down (Q4_K or Q6_K depending on block)
    if bw.dt_down == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
            D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
            D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
    }
    if dbg { probe("down", act.down, D_MODEL, dbg); }
    // Residual
    aether_op_add_inplace_f32_cuda(act.x, act.down, D_MODEL as c_int);
    if dbg { probe("x (block out)", act.x, D_MODEL, dbg); }
}

#[test]
#[ignore]  // ~70s load + ~3s autoregressive
fn qwen25_autoregressive_fused_tok_per_sec() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    let t_total = std::time::Instant::now();
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        let t = std::time::Instant::now();
        let blocks: Vec<BlockGpu> = (0..N_LAYERS).map(|b| load_block_to_device(h, b)).collect();
        eprintln!("[upload all 28 blocks] {:.2}s", t.elapsed().as_secs_f32());

        let final_norm_g = upload_f32_tensor(h, "output_norm.weight");
        let (lm_head, lm_n_blocks, lm_dt) = upload_tensor_u8(h, "output.weight");
        eprintln!("[lm_head dtype] {}", lm_dt);
        eprintln!("[upload output_norm + lm_head] {:.2}s",
            t.elapsed().as_secs_f32());

        let act = alloc_activations();
        let kvs = alloc_kv_caches();

        // Sample prompt: 4 tokens
        let prompt = [9707usize, 11, 1879, 0];
        let n_gen = 5;
        let mut token_ids: Vec<usize> = prompt.to_vec();

        // PREFILL stage: process the prompt tokens. For the bench
        // simplicity we just run each prompt token through as if it
        // were a generation step (no past-KV-cache attention -- the
        // full prefill with KV cache is the FR-x-extra-deeper follow-up
        // tagged in HANDOFF).
        let t_prefill = std::time::Instant::now();
        for (step, &t_id) in prompt.iter().enumerate() {
            // Set x to the embedding row.
            let emb_host = dequant_embd_rows(h, &[t_id]);
            aether_dev_h2d_f32(emb_host.as_ptr() as i64, act.x, D_MODEL as c_int);
            for b in 0..N_LAYERS {
                let dbg = step == 0 && b == 3;
                if dbg { eprintln!("\n  === DEBUG: block {} forward (step=0) ===", b); }
                block_forward_step_debug(&blocks[b], &act, &kvs[b], step as c_int, dbg);
            }
        }
        aether_dev_sync();
        eprintln!("[prefill 4 tok] {:.2}s = {:.1} ms/token",
            t_prefill.elapsed().as_secs_f32(),
            t_prefill.elapsed().as_secs_f32() * 1000.0 / 4.0);

        // GENERATE n_gen tokens
        let t_gen_start = std::time::Instant::now();
        for step in 0..n_gen {
            // Get last token's embedding
            let last_id = token_ids.last().copied().unwrap();
            let emb_host = dequant_embd_rows(h, &[last_id]);
            aether_dev_h2d_f32(emb_host.as_ptr() as i64, act.x, D_MODEL as c_int);

            let abs_pos = token_ids.len() as i32 - 1;  // 0-indexed; current step appends at abs_pos
            for b in 0..N_LAYERS {
                block_forward_step(&blocks[b], &act, &kvs[b], abs_pos);

                // [DEBUG] Print x magnitude per block 0..7
                if false && step == 0 && b < 8 {
                    aether_dev_sync();
                    let mut x_host = vec![0.0f32; D_MODEL];
                    aether_dev_d2h_f32(act.x, x_host.as_mut_ptr() as i64, D_MODEL as c_int);
                    let max_abs: f32 = x_host.iter().map(|v| v.abs()).fold(0.0, f32::max);
                    let sum: f32 = x_host.iter().sum();
                    let nonzero = x_host.iter().filter(|&&v| v.abs() > 1e-10).count();
                    eprintln!("  [debug step 0, after blk {}] max_abs={:.3e} sum={:.3e} nonzero={}/{}",
                        b, max_abs, sum, nonzero, D_MODEL);
                }
            }

            // Final RMSNorm
            aether_op_rms_norm_f32_cuda(act.x, final_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
            // lm_head (Q6_K v2): act.x_norm @ output.weight -> logits[VOCAB]
            if lm_dt == 14 {
                aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                    VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
            } else {
                aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                    VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
            }

            aether_dev_sync();
            // Argmax on host (small d2h)
            let mut logits = vec![0.0f32; VOCAB];
            aether_dev_d2h_f32(act.logits, logits.as_mut_ptr() as i64, VOCAB as c_int);
            let (best_id, &best_val) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            let min_val: f32 = logits.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val: f32 = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if false && step == 0 {
                eprintln!("  [debug step 0] logit range=[{:.3e}, {:.3e}], argmax={}, sample logits[0..4]={:?}",
                    min_val, max_val, best_id, &logits[..4]);
            }
            let _ = (best_val, min_val, max_val);
            token_ids.push(best_id);
        }
        aether_dev_sync();
        let gen_secs = t_gen_start.elapsed().as_secs_f32();
        let ms_per_token = gen_secs * 1000.0 / n_gen as f32;
        let tok_per_sec = n_gen as f32 / gen_secs;
        eprintln!("[generate {} tokens] {:.3}s = {:.1} ms/token = {:.2} tok/s",
            n_gen, gen_secs, ms_per_token, tok_per_sec);

        eprintln!("[generated IDs] {:?}", &token_ids[prompt.len()..]);
        eprintln!("[total] {:.2}s (incl 70s+ load)", t_total.elapsed().as_secs_f32());

        // Cleanup
        for b in &blocks {
            aether_dev_free_u8(b.w_q); aether_dev_free_u8(b.w_k); aether_dev_free_u8(b.w_v);
            aether_dev_free_u8(b.w_o); aether_dev_free_u8(b.w_gate); aether_dev_free_u8(b.w_up);
            aether_dev_free_u8(b.w_down);
            aether_dev_free_f32(b.attn_norm_g); aether_dev_free_f32(b.ffn_norm_g);
            aether_dev_free_f32(b.b_q); aether_dev_free_f32(b.b_k); aether_dev_free_f32(b.b_v);
        }
        for kv in &kvs {
            aether_dev_free_f32(kv.k_cache);
            aether_dev_free_f32(kv.v_cache);
        }
        aether_dev_free_f32(final_norm_g);
        aether_dev_free_u8(lm_head);
        for h in [act.x, act.x_norm, act.q, act.k_step, act.v_step,
                  act.attn_out, act.proj, act.gate, act.up, act.down, act.logits] {
            aether_dev_free_f32(h);
        }
        aether_gguf_close(h);
    }
}
