//! Qwen2.5-7B autoregressive serving session.
//!
//! Owns model weights + KV cache + activation buffers + a captured CUDA
//! graph. Drives the per-token decode loop that `aether-serve` exposes
//! over HTTP. Reference impl lives in `runtime/tests/qwen25_graph_decode.rs`;
//! this module is the factored-out reusable version.
//!
//! Hardcoded to Qwen2.5-7B-Instruct (Q4_K_M) shape today. matt-voice's
//! finetune is the same shape. Other architectures (Llama-3, Qwen3,
//! Gemma3) will need either separate sessions or a runtime-shape variant.
//!
//! Lifecycle:
//!   let mut s = QwenSession::new(gguf_path)?;
//!   s.reset();
//!   s.prefill(&[9707, 11, 1879, 0]);       // BOS+prompt tokens
//!   for _ in 0..max_tokens {
//!       let id = s.decode_step();
//!       if id == eos { break; }
//!       generated.push(id);
//!   }
//!
//! All buffers freed in Drop.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use crate::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m,
    aether_gguf_get_metadata_u32, aether_gguf_get_metadata_array_string_n,
    aether_gguf_get_metadata_array_string_get,
    aether_bpe_tokenizer_new, aether_bpe_tokenizer_free,
    aether_bpe_add_token_with_id, aether_bpe_add_merge_by_id,
    aether_bpe_decode,
};
use crate::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_dev_alloc_i32, aether_dev_free_i32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rope_apply_devarg_f32_cuda,
    aether_op_append_kv_devarg_f32_cuda,
    aether_op_attention_seq1_devarg_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_dev_graph_begin, aether_dev_graph_end,
    aether_dev_graph_launch, aether_dev_graph_destroy,
};

pub const D_MODEL: usize = 3584;
pub const N_LAYERS: usize = 28;
pub const N_Q_HEADS: usize = 28;
pub const N_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = D_MODEL / N_Q_HEADS;
pub const D_KV: usize = N_KV_HEADS * HEAD_DIM;
pub const D_FF: usize = 18944;
pub const VOCAB: usize = 152064;
pub const ROPE_BASE: f32 = 1_000_000.0;
pub const NORM_EPS: f32 = 1e-6;
pub const MAX_SEQ: usize = 32;  // FIXME: bump after profiling per-MAX_SEQ cost

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
    assert!(idx >= 0, "{} not found in GGUF", name);
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let n_blocks = n_elems / 256;
    let bytes_per_block = match dt {
        12 => 144,
        14 => 210,
        _  => panic!("unsupported dtype {} for tensor {}", dt, name),
    };
    let n_bytes = n_blocks * bytes_per_block;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let d_handle = aether_dev_alloc_u8(n_bytes as c_int);
    aether_dev_h2d_u8(dptr, d_handle, n_bytes as c_int);
    (d_handle, n_blocks, dt)
}

unsafe fn upload_f32_tensor(h: i64, name: &str) -> i64 {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    assert!(idx >= 0, "{} not found in GGUF", name);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let host: Vec<f32> = std::slice::from_raw_parts(dptr as *const f32, n_elems).to_vec();
    let d = aether_dev_alloc_f32(n_elems as c_int);
    aether_dev_h2d_f32(host.as_ptr() as i64, d, n_elems as c_int);
    d
}

unsafe fn load_block(h: i64, b: usize) -> BlockGpu {
    let p = format!("blk.{}.", b);
    let (w_q, nb_qo, _)            = upload_tensor_u8(h, &format!("{}attn_q.weight", p));
    let (w_k, nb_kv, _)            = upload_tensor_u8(h, &format!("{}attn_k.weight", p));
    let (w_v, _, dt_v)             = upload_tensor_u8(h, &format!("{}attn_v.weight", p));
    let (w_o, _, _)                = upload_tensor_u8(h, &format!("{}attn_output.weight", p));
    let (w_gate, nb_gate_up, _)    = upload_tensor_u8(h, &format!("{}ffn_gate.weight", p));
    let (w_up, _, _)               = upload_tensor_u8(h, &format!("{}ffn_up.weight", p));
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

/// Owns the entire decode-ready GPU state for one Qwen2.5-7B model.
///
/// Construction is heavy (~5 GB GGUF read + ~3 GB H2D upload, ~1-2 s
/// on a 3070 Ti). `reset()` + `prefill()` + `decode_step()` are the
/// per-request hot path. The captured graph is reused across requests
/// because the kernels read `pos`/`cur_seq` from `step_args` device
/// memory rather than baked into the launch args.
pub struct QwenSession {
    gguf_handle: i64,
    blocks: Vec<BlockGpu>,
    final_norm_g: i64,
    lm_head: i64,
    lm_n_blocks: usize,
    lm_dt: i32,
    act: ActivationGpu,
    kvs: Vec<KvCacheGpu>,
    step_args: i64,
    /// Position to use in the NEXT decode step. Matches the test's
    /// `token_ids.len() - 1` convention:
    ///   - After prefill of N tokens: next_pos = N - 1 (the first decode
    ///     step re-runs the last prefill step idempotently and reads the
    ///     logits for "what comes after the prompt").
    ///   - After each decode_step: next_pos += 1.
    next_pos: i32,
    /// True after the per-step graph has been captured.
    graph_captured: bool,
    /// BPE tokenizer handle loaded from the GGUF's `tokenizer.ggml.*`
    /// metadata. Used for `decode_ids` → text. -1 if load failed (some
    /// non-Qwen GGUFs lack this metadata).
    bpe_handle: i64,
    /// GPT-2 byte-to-unicode lookup, inverted for surface-char →
    /// real-byte fixup after BPE decode.
    gpt2_u2b: std::collections::HashMap<char, u8>,
    /// Cached EOS token ID from `tokenizer.ggml.eos_token_id`. Used for
    /// auto-stop in `generate()` when caller doesn't pass an explicit
    /// stop_token. -1 if metadata absent.
    pub eos_token: i32,
}

impl QwenSession {
    /// Open a GGUF + upload all weights to GPU. Returns Err on missing
    /// file or shape mismatch with Qwen2.5-7B (28 layers, d=3584).
    pub fn new(gguf_path: &str) -> Result<Self, String> {
        if !std::path::Path::new(gguf_path).exists() {
            return Err(format!("GGUF not found: {}", gguf_path));
        }
        unsafe {
            aether_dev_init();
            let h = aether_gguf_open(gguf_path.as_ptr() as i64, gguf_path.len() as c_int);
            if h < 0 {
                return Err(format!("aether_gguf_open failed: {}", h));
            }

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
            let step_args = aether_dev_alloc_i32(4);  // [pos, cur_seq, 0, 0]

            let (bpe_handle, eos_token) = load_tokenizer_from_gguf(h);
            let gpt2_u2b = build_gpt2_unicode_to_byte();
            Ok(QwenSession {
                gguf_handle: h, blocks, final_norm_g,
                lm_head, lm_n_blocks, lm_dt,
                act, kvs, step_args,
                next_pos: 0,
                graph_captured: false,
                bpe_handle, gpt2_u2b, eos_token,
            })
        }
    }

    /// Reset the KV cache + position for a new request. Cheap (no GPU
    /// allocation; the cache pages stay resident and get overwritten
    /// at pos=0). The captured graph stays — it's stateless.
    pub fn reset(&mut self) {
        self.next_pos = 0;
    }

    /// Dequantize one embedding row by token id and return it on the
    /// host. (Q4_K_M token_embd.weight stays on host in this version;
    /// 152064 × 3584 f32 would be 2 GB and we don't pay that cost when
    /// most tokens never appear in any one request.)
    unsafe fn dequant_embd_row(&self, token_id: usize) -> Vec<f32> {
        let needle = b"token_embd.weight";
        let idx = aether_gguf_find_tensor_by_name(
            self.gguf_handle, needle.as_ptr() as i64, needle.len() as c_int);
        assert!(idx >= 0);
        let n_elems = aether_gguf_get_tensor_n_elems(self.gguf_handle, idx) as usize;
        let total_rows = n_elems / D_MODEL;
        let dptr = aether_gguf_get_tensor_data_ptr(self.gguf_handle, idx) as *const u8;
        let blocks_per_row = D_MODEL / 256;
        let bytes_per_row = blocks_per_row * 144;
        assert!(token_id < total_rows, "token_id {} out of vocab {}", token_id, total_rows);
        let row_bytes = std::slice::from_raw_parts(
            dptr.add(token_id * bytes_per_row), bytes_per_row);
        let mut row_f32 = vec![0.0f32; D_MODEL];
        aether_dequant_q4_k_m(
            row_bytes.as_ptr() as *const c_void,
            row_f32.as_mut_ptr() as *mut c_void,
            blocks_per_row as c_int,
        );
        row_f32
    }

    /// Prefill the KV cache by running the forward pass once per input
    /// token in `prompt_ids`. Uses the immediate (non-graph) devarg
    /// kernels — the graph capture happens after prefill.
    ///
    /// On return, `next_pos` is set to `prompt_ids.len() - 1` so that
    /// the first `decode_step` re-runs the last prefill step idempotently
    /// (matching the reference impl in `qwen25_graph_decode.rs`).
    pub fn prefill(&mut self, prompt_ids: &[usize]) {
        assert!(!prompt_ids.is_empty(), "prompt cannot be empty");
        unsafe {
            for (i, &t_id) in prompt_ids.iter().enumerate() {
                let emb = self.dequant_embd_row(t_id);
                aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, D_MODEL as c_int);
                let pos = i as i32;
                let cur_seq = pos + 1;
                let step_host = [pos, cur_seq, 0i32, 0i32];
                aether_dev_h2d_i32(step_host.as_ptr() as i64, self.step_args, 4);
                for b in 0..N_LAYERS {
                    block_forward_devarg(&self.blocks[b], &self.act, &self.kvs[b], self.step_args);
                }
            }
            aether_dev_sync();
            // The next decode iter re-runs the last prefill step
            // idempotently and reads its logits — matches the test.
            self.next_pos = (prompt_ids.len() as i32) - 1;
        }
    }

    /// Capture the per-step decode into a CUDA graph. Lazy: called on
    /// the first `decode_step` of the first request after the session
    /// is loaded. Subsequent decode steps reuse the graph.
    ///
    /// PRECONDITION: caller has already set `act.x` to the last-token
    /// embedding and h2d'd `step_args = [next_pos, next_pos+1, 0, 0]`
    /// — this lets the captured graph "see" valid inputs so the
    /// capture's ghost step doesn't produce garbage.
    unsafe fn capture_graph_now(&mut self) {
        let rc = aether_dev_graph_begin();
        assert_eq!(rc, 0, "aether_dev_graph_begin failed: {}", rc);
        for b in 0..N_LAYERS {
            block_forward_devarg(&self.blocks[b], &self.act, &self.kvs[b], self.step_args);
        }
        aether_op_rms_norm_f32_cuda(
            self.act.x, self.final_norm_g, self.act.x_norm,
            NORM_EPS, 1, D_MODEL as c_int);
        if self.lm_dt == 14 {
            aether_op_fused_q6k_matmul_seq1_v2_cuda(
                self.act.x_norm, self.lm_head, self.act.logits,
                VOCAB as c_int, (self.lm_n_blocks / VOCAB) as c_int);
        } else {
            aether_op_fused_q4k_matmul_seq1_v2_cuda(
                self.act.x_norm, self.lm_head, self.act.logits,
                VOCAB as c_int, (self.lm_n_blocks / VOCAB) as c_int);
        }
        let rc = aether_dev_graph_end();
        assert_eq!(rc, 0, "aether_dev_graph_end failed: {}", rc);
        self.graph_captured = true;
    }

    /// Run one decode step.
    ///
    /// Semantics: feeds the embedding of `last_id` into the model at
    /// position `next_pos`, writes K/V for `last_id` into the cache at
    /// that slot (overwriting if the slot was already used), reads
    /// logits, returns argmax. `next_pos` advances by 1.
    ///
    /// On the FIRST call after construction, the per-step forward is
    /// captured into a CUDA graph for replay on subsequent calls.
    pub fn decode_step(&mut self, last_id: usize) -> usize {
        unsafe {
            // Feed input embedding + step args.
            let emb = self.dequant_embd_row(last_id);
            aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, D_MODEL as c_int);
            let pos = self.next_pos;
            let cur_seq = pos + 1;
            let step_host = [pos, cur_seq, 0i32, 0i32];
            aether_dev_h2d_i32(step_host.as_ptr() as i64, self.step_args, 4);

            if !self.graph_captured {
                aether_dev_sync();
                self.capture_graph_now();
                // Capture only RECORDS — explicit launch needed to execute.
            }
            let rc = aether_dev_graph_launch();
            assert_eq!(rc, 0, "aether_dev_graph_launch failed: {}", rc);
            aether_dev_sync();

            let mut logits = vec![0.0f32; VOCAB];
            aether_dev_d2h_f32(self.act.logits, logits.as_mut_ptr() as i64, VOCAB as c_int);
            self.next_pos += 1;
            argmax(&logits)
        }
    }

    /// Warm the GPU by running a few decode iterations on a synthetic
    /// prompt. Drives the GPU into P0/P2 power state so the FIRST real
    /// request doesn't get stuck at idle clocks (210 MHz → ~100x slower).
    ///
    /// Also forces the lazy graph capture to happen on startup rather than
    /// inside the first user request.
    pub fn warmup(&mut self, n_steps: usize) {
        let synth_prompt: Vec<usize> = vec![1, 2, 3, 4];
        self.reset();
        self.prefill(&synth_prompt);
        let mut last = synth_prompt[synth_prompt.len() - 1];
        for _ in 0..n_steps {
            last = self.decode_step(last);
        }
    }

    /// Generate `max_tokens` token ids starting from `prompt_ids`.
    /// Stops early if `stop_token` is produced. Returns the generated
    /// suffix (does NOT include the prompt).
    pub fn generate(
        &mut self, prompt_ids: &[usize], max_tokens: usize, stop_token: Option<usize>,
    ) -> Vec<usize> {
        self.reset();
        self.prefill(prompt_ids);
        let mut generated = Vec::with_capacity(max_tokens);
        let mut last = *prompt_ids.last().expect("prompt cannot be empty");
        for _ in 0..max_tokens {
            let id = self.decode_step(last);
            if Some(id) == stop_token { break; }
            generated.push(id);
            last = id;
            if self.next_pos as usize >= MAX_SEQ - 1 { break; }
        }
        generated
    }

    /// Total VRAM footprint reported by the runtime (approximate).
    /// Useful for the /v1/models endpoint diagnostics.
    pub fn approx_vram_mb(&self) -> u64 {
        let weights = (N_LAYERS as u64) * 870 + 2200;  // ~25 GB est? no, q4_k_m
        // Q4_K_M packs to ~4.7 GB total for Qwen2.5-7B.
        weights.min(5_000)
    }
}

impl Drop for QwenSession {
    fn drop(&mut self) {
        unsafe {
            if self.graph_captured {
                aether_dev_graph_destroy();
            }
            // Per-block weights + biases + norms
            for b in self.blocks.drain(..) {
                let _ = aether_dev_free_f32(b.attn_norm_g);
                let _ = aether_dev_free_f32(b.ffn_norm_g);
                let _ = aether_dev_free_u8(b.w_q);
                let _ = aether_dev_free_u8(b.w_k);
                let _ = aether_dev_free_u8(b.w_v);
                let _ = aether_dev_free_u8(b.w_o);
                let _ = aether_dev_free_u8(b.w_gate);
                let _ = aether_dev_free_u8(b.w_up);
                let _ = aether_dev_free_u8(b.w_down);
                let _ = aether_dev_free_f32(b.b_q);
                let _ = aether_dev_free_f32(b.b_k);
                let _ = aether_dev_free_f32(b.b_v);
            }
            let _ = aether_dev_free_f32(self.final_norm_g);
            let _ = aether_dev_free_u8(self.lm_head);
            let _ = aether_dev_free_f32(self.act.x);
            let _ = aether_dev_free_f32(self.act.x_norm);
            let _ = aether_dev_free_f32(self.act.q);
            let _ = aether_dev_free_f32(self.act.k_step);
            let _ = aether_dev_free_f32(self.act.v_step);
            let _ = aether_dev_free_f32(self.act.attn_out);
            let _ = aether_dev_free_f32(self.act.proj);
            let _ = aether_dev_free_f32(self.act.gate);
            let _ = aether_dev_free_f32(self.act.down);
            let _ = aether_dev_free_f32(self.act.logits);
            for kv in self.kvs.drain(..) {
                let _ = aether_dev_free_f32(kv.k_cache);
                let _ = aether_dev_free_f32(kv.v_cache);
            }
            let _ = aether_dev_free_i32(self.step_args);
            if self.bpe_handle >= 0 {
                let _ = aether_bpe_tokenizer_free(self.bpe_handle);
            }
            aether_gguf_close(self.gguf_handle);
        }
    }
}

// ---------------------- tokenizer (decode side) ----------------------
//
// Loads Qwen2.5's tokenizer.ggml.tokens + tokenizer.ggml.merges into
// aether_bpe_tokenizer + the EOS token id. Decode-only: encode (text
// → ids) needs GPT-2 unicode-char-level BPE which is FR-19.9-extra-
// deeper. See `runtime/tests/qwen25_tokenizer_roundtrip.rs` for the
// reference impl this is factored from.

unsafe fn load_tokenizer_from_gguf(h: i64) -> (i64, i32) {
    let tok_key = b"tokenizer.ggml.tokens";
    let n = aether_gguf_get_metadata_array_string_n(
        h, tok_key.as_ptr() as i64, tok_key.len() as c_int);
    if n <= 0 {
        eprintln!("[QwenSession] no tokenizer.ggml.tokens — text decode disabled");
        return (-1, -1);
    }

    let bpe = aether_bpe_tokenizer_new();
    if bpe < 0 {
        eprintln!("[QwenSession] aether_bpe_tokenizer_new failed: {}", bpe);
        return (-1, -1);
    }

    let mut vocab_bytes: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    let mut buf = vec![0u8; 512];
    for i in 0..n {
        let nb = aether_gguf_get_metadata_array_string_get(
            h, tok_key.as_ptr() as i64, tok_key.len() as c_int, i,
            buf.as_mut_ptr() as i64, buf.len() as c_int);
        if nb < 0 {
            eprintln!("[QwenSession] vocab entry {} truncated (nb={})", i, nb);
            aether_bpe_tokenizer_free(bpe);
            return (-1, -1);
        }
        let bytes = buf[..nb as usize].to_vec();
        let rc = aether_bpe_add_token_with_id(
            bpe, i, bytes.as_ptr() as *const c_void, nb);
        if rc != 0 {
            eprintln!("[QwenSession] add_token({}) -> {}", i, rc);
            aether_bpe_tokenizer_free(bpe);
            return (-1, -1);
        }
        vocab_bytes.push(bytes);
    }

    let merges_key = b"tokenizer.ggml.merges";
    let m = aether_gguf_get_metadata_array_string_n(
        h, merges_key.as_ptr() as i64, merges_key.len() as c_int);
    if m > 0 {
        let mut lookup: std::collections::HashMap<Vec<u8>, u32> =
            std::collections::HashMap::with_capacity(vocab_bytes.len());
        for (i, b) in vocab_bytes.iter().enumerate() {
            lookup.insert(b.clone(), i as u32);
        }
        let mut loaded = 0;
        for i in 0..m {
            let nb = aether_gguf_get_metadata_array_string_get(
                h, merges_key.as_ptr() as i64, merges_key.len() as c_int, i,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if nb <= 0 { continue; }
            let s = &buf[..nb as usize];
            let Some(space_idx) = s.iter().position(|&b| b == b' ') else { continue; };
            let left = &s[..space_idx];
            let right = &s[space_idx + 1..];
            let Some(&left_id) = lookup.get(left) else { continue; };
            let Some(&right_id) = lookup.get(right) else { continue; };
            let mut merged = Vec::with_capacity(left.len() + right.len());
            merged.extend_from_slice(left);
            merged.extend_from_slice(right);
            let Some(&merged_id) = lookup.get(&merged) else { continue; };
            let rc = aether_bpe_add_merge_by_id(
                bpe, left_id as c_int, right_id as c_int, i, merged_id as c_int);
            if rc == 0 { loaded += 1; }
        }
        eprintln!("[QwenSession] tokenizer loaded — vocab={}, merges={}", n, loaded);
    } else {
        eprintln!("[QwenSession] tokenizer loaded — vocab={}, no merges in GGUF", n);
    }

    let eos_key = b"tokenizer.ggml.eos_token_id";
    let eos = aether_gguf_get_metadata_u32(
        h, eos_key.as_ptr() as i64, eos_key.len() as c_int);
    let eos_token: i32 = if eos < 0 { -1 } else { eos as i32 };
    eprintln!("[QwenSession] EOS token id: {}", eos_token);
    (bpe, eos_token)
}

/// Build the GPT-2 byte-to-unicode mapping and return its inverse. This
/// is the same table used by GPT-2/3, Llama-3, Qwen, etc. for surface-
/// level BPE tokenization. Every byte 0..255 maps to a printable
/// unicode char, and the inverse recovers raw bytes after decode.
fn build_gpt2_unicode_to_byte() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = Vec::new();
    for b in 33..=126_u32 { bs.push(b); }
    for b in 161..=172_u32 { bs.push(b); }
    for b in 174..=255_u32 { bs.push(b); }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256_u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut m = std::collections::HashMap::with_capacity(256);
    for (b, c) in bs.iter().zip(cs.iter()) {
        if let Some(ch) = char::from_u32(*c) {
            m.insert(ch, *b as u8);
        }
    }
    m
}

impl QwenSession {
    /// Decode a slice of token ids back to UTF-8 text. Uses the BPE
    /// surface-byte decoder + GPT-2 byte fixup. Returns an empty string
    /// if the tokenizer wasn't loaded.
    pub fn decode_ids(&self, ids: &[usize]) -> String {
        if self.bpe_handle < 0 || ids.is_empty() { return String::new(); }
        unsafe {
            let id_buf: Vec<i32> = ids.iter().map(|&i| i as i32).collect();
            let mut out_buf = vec![0u8; 8192];
            let nb = aether_bpe_decode(
                self.bpe_handle,
                id_buf.as_ptr() as *const c_void, id_buf.len() as c_int,
                out_buf.as_mut_ptr() as *mut c_void, out_buf.len() as c_int);
            if nb <= 0 { return String::new(); }
            let surface = match std::str::from_utf8(&out_buf[..nb as usize]) {
                Ok(s) => s.to_string(),
                Err(_) => return String::new(),
            };
            // GPT-2 inverse byte mapping: surface "Ġ" → byte 0x20, etc.
            let real_bytes: Vec<u8> = surface.chars()
                .filter_map(|c| self.gpt2_u2b.get(&c).copied())
                .collect();
            String::from_utf8_lossy(&real_bytes).into_owned()
        }
    }
}

fn argmax(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}
