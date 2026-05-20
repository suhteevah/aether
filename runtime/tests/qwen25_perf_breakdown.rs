//! Per-op time breakdown for the autoregressive_fused chain.
//! Times major op categories across N decode steps, prints the
//! split so we can see where the ~6 ms/token gap to llama.cpp lives.
//!
//! Runs the SAME chain as qwen25_autoregressive_fused, just with
//! cudaDeviceSynchronize between groups so wall-clock measurements
//! attribute correctly.

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
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_rms_norm_f32_cuda, aether_op_rope_apply_f32_cuda,
    aether_op_silu_f32_cuda, aether_op_mul_inplace_f32_cuda,
    aether_op_add_inplace_f32_cuda, aether_op_bias_add_f32_cuda,
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
    gate: i64, up: i64, down: i64,
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

// Time accumulator: sync, then group, then sync, then add elapsed.
struct Timer { totals: [f64; 8], labels: [&'static str; 8] }
impl Timer {
    fn new() -> Self {
        Timer {
            totals: [0.0; 8],
            labels: [
                "attn_norm + Q/K/V proj + biases + RoPE",
                "append_kv + attention",
                "O proj + residual",
                "ffn_norm + gate/up + silu + mul",
                "down + residual",
                "final_norm + lm_head",
                "argmax d2h + host",
                "(total)",
            ],
        }
    }
    fn print(&self, n_steps: usize) {
        let per_step_total: f64 = self.totals.iter().take(7).sum();
        eprintln!("\nPer-token time breakdown ({} steps avg):", n_steps);
        for i in 0..7 {
            let avg_ms = self.totals[i] * 1000.0 / n_steps as f64;
            let pct = self.totals[i] / per_step_total * 100.0;
            eprintln!("  {:5.2} ms ({:5.1}%)  {}", avg_ms, pct, self.labels[i]);
        }
        eprintln!("  {:5.2} ms (100.0%)  TOTAL per token",
            per_step_total * 1000.0 / n_steps as f64);
        eprintln!("  {:5.2} tok/s",
            n_steps as f64 / per_step_total);
    }
}

#[test]
#[ignore]
fn qwen25_perf_breakdown() {
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

        let act = ActivationGpu {
            x: aether_dev_alloc_f32(D_MODEL as c_int),
            x_norm: aether_dev_alloc_f32(D_MODEL as c_int),
            q: aether_dev_alloc_f32(D_MODEL as c_int),
            k_step: aether_dev_alloc_f32(D_KV as c_int),
            v_step: aether_dev_alloc_f32(D_KV as c_int),
            attn_out: aether_dev_alloc_f32(D_MODEL as c_int),
            proj: aether_dev_alloc_f32(D_MODEL as c_int),
            gate: aether_dev_alloc_f32(D_FF as c_int),
            up: aether_dev_alloc_f32(D_FF as c_int),
            down: aether_dev_alloc_f32(D_MODEL as c_int),
            logits: aether_dev_alloc_f32(VOCAB as c_int),
        };
        let kvs: Vec<KvCacheGpu> = (0..N_LAYERS).map(|_| KvCacheGpu {
            k_cache: aether_dev_alloc_f32((MAX_SEQ * D_KV) as c_int),
            v_cache: aether_dev_alloc_f32((MAX_SEQ * D_KV) as c_int),
        }).collect();

        // Prefill 4 tokens (no timing, just to seed KV cache).
        let prompt = [9707usize, 11, 1879, 0];
        let mut token_ids: Vec<usize> = prompt.to_vec();
        for (step, &t_id) in prompt.iter().enumerate() {
            let emb = dequant_embd_rows(h, &[t_id]);
            aether_dev_h2d_f32(emb.as_ptr() as i64, act.x, D_MODEL as c_int);
            for b in 0..N_LAYERS {
                block_forward_step(&blocks[b], &act, &kvs[b], step as c_int);
            }
        }
        aether_dev_sync();

        // Now N timed decode steps.
        const N_STEPS: usize = 8;
        let mut timer = Timer::new();

        for step in 0..N_STEPS {
            let last_id = token_ids.last().copied().unwrap();
            let emb = dequant_embd_rows(h, &[last_id]);
            aether_dev_h2d_f32(emb.as_ptr() as i64, act.x, D_MODEL as c_int);
            let abs_pos = token_ids.len() as i32 - 1;
            // Make sure h2d is done before timing.
            aether_dev_sync();

            for b in 0..N_LAYERS {
                let bw = &blocks[b];
                let kv = &kvs[b];

                // Group 0: attn_norm + Q/K/V proj + biases + RoPE
                let t = Instant::now();
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
                aether_op_rope_apply_f32_cuda(act.q,      1, N_Q_HEADS  as c_int, HEAD_DIM as c_int, ROPE_BASE, abs_pos);
                aether_op_rope_apply_f32_cuda(act.k_step, 1, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, abs_pos);
                aether_dev_sync();
                timer.totals[0] += t.elapsed().as_secs_f64();

                // Group 1: append_kv + attention
                let t = Instant::now();
                aether_op_append_kv_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache,
                    abs_pos, D_KV as c_int);
                let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
                aether_op_attention_seq1_f32_cuda(
                    act.q, kv.k_cache, kv.v_cache, act.attn_out,
                    abs_pos + 1, N_Q_HEADS as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int,
                    scale,
                );
                aether_dev_sync();
                timer.totals[1] += t.elapsed().as_secs_f64();

                // Group 2: O proj + residual
                let t = Instant::now();
                aether_op_fused_q4k_matmul_seq1_v2_cuda(act.attn_out, bw.w_o, act.proj,
                    D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
                aether_op_add_inplace_f32_cuda(act.x, act.proj, D_MODEL as c_int);
                aether_dev_sync();
                timer.totals[2] += t.elapsed().as_secs_f64();

                // Group 3: ffn_norm + FUSED gate+up+silu+mul (1 kernel)
                let t = Instant::now();
                aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
                aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
                    act.x_norm, bw.w_gate, bw.w_up, act.gate,
                    D_FF as c_int, (bw.nb_gate_up / D_FF) as c_int);
                aether_dev_sync();
                timer.totals[3] += t.elapsed().as_secs_f64();

                // Group 4: down + residual
                let t = Instant::now();
                if bw.dt_down == 14 {
                    aether_op_fused_q6k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
                        D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
                } else {
                    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
                        D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
                }
                aether_op_add_inplace_f32_cuda(act.x, act.down, D_MODEL as c_int);
                aether_dev_sync();
                timer.totals[4] += t.elapsed().as_secs_f64();
            }

            // Group 5: final_norm + lm_head
            let t = Instant::now();
            aether_op_rms_norm_f32_cuda(act.x, final_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
            if lm_dt == 14 {
                aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                    VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
            } else {
                aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, lm_head, act.logits,
                    VOCAB as c_int, (lm_n_blocks / VOCAB) as c_int);
            }
            aether_dev_sync();
            timer.totals[5] += t.elapsed().as_secs_f64();

            // Group 6: argmax (d2h + host scan)
            let t = Instant::now();
            let mut logits = vec![0.0f32; VOCAB];
            aether_dev_d2h_f32(act.logits, logits.as_mut_ptr() as i64, VOCAB as c_int);
            let (best_id, _) = logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            timer.totals[6] += t.elapsed().as_secs_f64();
            token_ids.push(best_id);
            let _ = step;
        }

        timer.print(N_STEPS);

        // Cleanup omitted (test exits after).
        aether_gguf_close(h);
    }
}

unsafe fn block_forward_step(bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, pos: i32) {
    aether_op_rms_norm_f32_cuda(act.x, bw.attn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_q, act.q, D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
    aether_op_bias_add_f32_cuda(act.q, bw.b_q, 1, D_MODEL as c_int);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_k, act.k_step, D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    aether_op_bias_add_f32_cuda(act.k_step, bw.b_k, 1, D_KV as c_int);
    if bw.dt_v == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step, D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step, D_KV as c_int, (bw.nb_kv / D_KV) as c_int);
    }
    aether_op_bias_add_f32_cuda(act.v_step, bw.b_v, 1, D_KV as c_int);
    aether_op_rope_apply_f32_cuda(act.q, 1, N_Q_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, pos);
    aether_op_rope_apply_f32_cuda(act.k_step, 1, N_KV_HEADS as c_int, HEAD_DIM as c_int, ROPE_BASE, pos);
    aether_op_append_kv_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache, pos, D_KV as c_int);
    let scale: f32 = 1.0 / (HEAD_DIM as f32).sqrt();
    aether_op_attention_seq1_f32_cuda(act.q, kv.k_cache, kv.v_cache, act.attn_out,
        pos + 1, N_Q_HEADS as c_int, N_KV_HEADS as c_int, HEAD_DIM as c_int, scale);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.attn_out, bw.w_o, act.proj, D_MODEL as c_int, (bw.nb_qo / D_MODEL) as c_int);
    aether_op_add_inplace_f32_cuda(act.x, act.proj, D_MODEL as c_int);
    aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, NORM_EPS, 1, D_MODEL as c_int);
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
        act.x_norm, bw.w_gate, bw.w_up, act.gate,
        D_FF as c_int, (bw.nb_gate_up / D_FF) as c_int);
    if bw.dt_down == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down, D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down, D_MODEL as c_int, (bw.nb_down / D_MODEL) as c_int);
    }
    aether_op_add_inplace_f32_cuda(act.x, act.down, D_MODEL as c_int);
}
