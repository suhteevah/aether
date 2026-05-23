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
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_dev_graph_begin, aether_dev_graph_end,
    aether_dev_graph_launch, aether_dev_graph_destroy,
};

// Historical Qwen2.5-7B-specific constants — kept ONLY for tests/witnesses
// that need to reference the 7B shape explicitly.  Production paths use the
// runtime-loaded `ModelConfig` populated from GGUF metadata at session
// construction so 14B / 32B / other-arch models pick up their correct shapes.
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

/// FR-17-extra-runtime-shape — Runtime model configuration read from GGUF
/// metadata.  Replaces the historical `const`s for everything that's
/// actually shape-dependent.  Populated by `ModelConfig::from_gguf` at
/// session construction.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub vocab: usize,
    pub rope_base: f32,
    pub norm_eps: f32,
    pub arch: String, // "qwen2", "llama", etc.  Used for metadata-key namespacing.
}

impl ModelConfig {
    /// Hardcoded Qwen2.5-7B shape — fallback when GGUF metadata is missing or
    /// for tests.  Kept consistent with the const block above.
    pub fn qwen2_5_7b() -> Self {
        Self {
            d_model: D_MODEL, n_layers: N_LAYERS, n_q_heads: N_Q_HEADS,
            n_kv_heads: N_KV_HEADS, head_dim: HEAD_DIM, d_kv: D_KV,
            d_ff: D_FF, vocab: VOCAB,
            rope_base: ROPE_BASE, norm_eps: NORM_EPS,
            arch: "qwen2".to_string(),
        }
    }

    /// Read shape parameters from a GGUF metadata block.  Falls back to
    /// 7B defaults for any key that's missing or malformed.
    pub unsafe fn from_gguf(gguf_handle: i64) -> Self {
        let arch = read_meta_string(gguf_handle, "general.architecture")
            .unwrap_or_else(|| "qwen2".to_string());
        let prefix = arch.clone();

        let d_model = read_meta_u32(gguf_handle, &format!("{}.embedding_length", prefix))
            .map(|v| v as usize).unwrap_or(D_MODEL);
        let n_layers = read_meta_u32(gguf_handle, &format!("{}.block_count", prefix))
            .map(|v| v as usize).unwrap_or(N_LAYERS);
        let n_q_heads = read_meta_u32(gguf_handle, &format!("{}.attention.head_count", prefix))
            .map(|v| v as usize).unwrap_or(N_Q_HEADS);
        let n_kv_heads = read_meta_u32(gguf_handle, &format!("{}.attention.head_count_kv", prefix))
            .map(|v| v as usize).unwrap_or(N_KV_HEADS);
        let head_dim = if n_q_heads > 0 { d_model / n_q_heads } else { HEAD_DIM };
        let d_kv = n_kv_heads * head_dim;
        let d_ff = read_meta_u32(gguf_handle, &format!("{}.feed_forward_length", prefix))
            .map(|v| v as usize).unwrap_or(D_FF);
        // VOCAB usually comes from tokenizer.ggml.tokens length, not from a
        // model-shape key.  Use the tokenizer-array count when present.
        let vocab = {
            let key = b"tokenizer.ggml.tokens";
            let n = crate::aether_gguf_get_metadata_array_string_n(
                gguf_handle, key.as_ptr() as i64, key.len() as c_int);
            if n > 0 { n as usize } else { VOCAB }
        };
        let rope_base = read_meta_f32(gguf_handle, &format!("{}.rope.freq_base", prefix))
            .unwrap_or(ROPE_BASE);
        let norm_eps = read_meta_f32(gguf_handle,
            &format!("{}.attention.layer_norm_rms_epsilon", prefix))
            .unwrap_or(NORM_EPS);
        Self {
            d_model, n_layers, n_q_heads, n_kv_heads, head_dim, d_kv,
            d_ff, vocab, rope_base, norm_eps, arch,
        }
    }
}

unsafe fn read_meta_u32(h: i64, key: &str) -> Option<u32> {
    let v = crate::aether_gguf_get_metadata_u32(
        h, key.as_ptr() as i64, key.len() as c_int);
    if v < 0 { None } else { Some(v as u32) }
}
unsafe fn read_meta_f32(h: i64, key: &str) -> Option<f32> {
    let v = crate::aether_gguf_get_metadata_f32(
        h, key.as_ptr() as i64, key.len() as c_int);
    if v.is_nan() { None } else { Some(v as f32) }
}
unsafe fn read_meta_string(h: i64, key: &str) -> Option<String> {
    let mut buf = vec![0u8; 256];
    let n = crate::aether_gguf_get_metadata_string(
        h, key.as_ptr() as i64, key.len() as c_int,
        buf.as_mut_ptr() as i64, buf.len() as c_int);
    if n <= 0 { return None; }
    String::from_utf8(buf[..n as usize].to_vec()).ok()
}

struct BlockGpu {
    attn_norm_g: i64, ffn_norm_g: i64,
    w_q: i64, w_k: i64, w_o: i64, w_gate: i64, w_up: i64,
    w_v: i64, dt_v: i32,
    w_down: i64, dt_down: i32,
    /// Q/K/V biases — present in Qwen2.5 (qwen2 arch).  Qwen3 dropped these.
    /// 0 indicates "absent".
    b_q: i64, b_k: i64, b_v: i64,
    /// FR-17-extra-qwen3-fwd — per-head Q/K RMS norm (Qwen3-style).
    /// 0 indicates "absent" (qwen2 / older archs).
    attn_q_norm_g: i64, attn_k_norm_g: i64,
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

// =====================================================================
// SharedKvPool — FR-19.4-extra-tenant.
//
// One GPU-resident pool per (layer × {K, V}), shared across multiple
// PagedQwenSessions on the same model.  Blocks within the pool are
// handed out by a host-side free-list; sessions track their own page
// tables that map their logical block index -> a physical block id
// in this pool.
//
// Memory footprint: 2 × N_LAYERS × n_blocks × block_size × D_KV × 4 bytes.
// For Qwen2.5 (28 layers, D_KV=512), 32 blocks × 4 tokens/block = 128
// token slots ≈ 14.7 MiB total.  Larger pools just grow proportionally.
// =====================================================================
pub struct SharedKvPool {
    pub n_blocks: i32,
    pub block_size: i32,
    pub n_layers: usize,
    pub d_kv: usize,
    pool_k: Vec<i64>,   // per-layer device pointer (f32, size = n_blocks*block_size*d_kv)
    pool_v: Vec<i64>,
    free: std::sync::Mutex<Vec<bool>>,  // free[b] = block b is free
}

impl SharedKvPool {
    /// Allocate `n_blocks` blocks of `block_size` tokens each, sized for a
    /// model with `n_layers` × `d_kv` K/V dimensions.  Each block holds
    /// block_size × d_kv f32 K and V values per layer.
    pub fn new_for_shape(
        n_blocks: i32, block_size: i32, n_layers: usize, d_kv: usize,
    ) -> std::sync::Arc<Self> {
        unsafe { crate::cuda::aether_dev_init(); }
        let n_per_pool = (n_blocks * block_size) as usize * d_kv;
        let mut pool_k = Vec::with_capacity(n_layers);
        let mut pool_v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            unsafe {
                pool_k.push(aether_dev_alloc_f32(n_per_pool as c_int));
                pool_v.push(aether_dev_alloc_f32(n_per_pool as c_int));
            }
        }
        std::sync::Arc::new(Self {
            n_blocks, block_size, n_layers, d_kv, pool_k, pool_v,
            free: std::sync::Mutex::new(vec![true; n_blocks as usize]),
        })
    }

    /// Backwards-compatible shortcut for the Qwen2.5-7B shape.  Use
    /// `new_for_shape` for any other architecture.
    pub fn new(n_blocks: i32, block_size: i32) -> std::sync::Arc<Self> {
        Self::new_for_shape(n_blocks, block_size, N_LAYERS, D_KV)
    }

    /// Per-layer K pool device pointer.  Stable for the lifetime of the pool.
    pub fn pool_k(&self, layer: usize) -> i64 { self.pool_k[layer] }
    /// Per-layer V pool device pointer.
    pub fn pool_v(&self, layer: usize) -> i64 { self.pool_v[layer] }

    /// Allocate a free block; returns block_id or -1 if pool exhausted.
    pub fn allocate_block(&self) -> i32 {
        let mut g = self.free.lock().unwrap();
        for (i, slot) in g.iter_mut().enumerate() {
            if *slot { *slot = false; return i as i32; }
        }
        -1
    }
    /// Return a block to the free pool.
    pub fn free_block(&self, block_id: i32) {
        if block_id < 0 { return; }
        let mut g = self.free.lock().unwrap();
        if (block_id as usize) < g.len() { g[block_id as usize] = true; }
    }
    /// Count of currently-allocated blocks.
    pub fn n_allocated(&self) -> i32 {
        self.free.lock().unwrap().iter().filter(|&&b| !b).count() as i32
    }
}

impl Drop for SharedKvPool {
    fn drop(&mut self) {
        unsafe {
            for &p in &self.pool_k { let _ = aether_dev_free_f32(p); }
            for &p in &self.pool_v { let _ = aether_dev_free_f32(p); }
        }
    }
}

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

/// Non-panicking variant for tensors that exist on some archs but not others.
/// Returns 0 (treated as "absent" by callers) if the tensor isn't in the GGUF.
unsafe fn upload_f32_tensor_opt(h: i64, name: &str) -> i64 {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    if idx < 0 { return 0; }
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
        // Qwen2 has biases on Q/K/V; Qwen3 doesn't.  Load as optional.
        b_q: upload_f32_tensor_opt(h, &format!("{}attn_q.bias", p)),
        b_k: upload_f32_tensor_opt(h, &format!("{}attn_k.bias", p)),
        b_v: upload_f32_tensor_opt(h, &format!("{}attn_v.bias", p)),
        // Qwen3 has per-head Q/K RMS norm; Qwen2 doesn't.  Load as optional.
        attn_q_norm_g: upload_f32_tensor_opt(h, &format!("{}attn_q_norm.weight", p)),
        attn_k_norm_g: upload_f32_tensor_opt(h, &format!("{}attn_k_norm.weight", p)),
        nb_qo, nb_kv, nb_gate_up, nb_down,
    }
}

/// Forward one block.  Takes a `cfg: &ModelConfig` so the runtime dims
/// flow into every kernel launch.  Hardcoded Qwen2.5-7B shape removed
/// — same kernel code path works for any model whose ops are
/// shape-compatible (Qwen2.5-14B, 32B, future Qwen variants).
///
/// `paged_cfg = Some((page_table_dev, block_size))` routes append_kv +
/// attention_seq1 through the paged variants; None uses the contiguous
/// kernels.  With an identity-mapping page table both modes are
/// bit-identical (witnessed in `runtime/tests/cuda_paged_kv_parity.rs`).
unsafe fn block_forward_devarg(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, step_args: i64,
    paged_cfg: Option<(i64, i32)>,
    cfg: &ModelConfig,
    max_seq: usize,
) {
    let d_model = cfg.d_model as c_int;
    let d_kv = cfg.d_kv as c_int;
    let d_ff = cfg.d_ff as c_int;
    let n_q_heads = cfg.n_q_heads as c_int;
    let n_kv_heads = cfg.n_kv_heads as c_int;
    let head_dim = cfg.head_dim as c_int;
    let rope_base = cfg.rope_base;
    let norm_eps = cfg.norm_eps;
    aether_op_rms_norm_f32_cuda(act.x, bw.attn_norm_g, act.x_norm, norm_eps, 1, d_model);
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_q, act.q,
        d_model, (bw.nb_qo / cfg.d_model) as c_int);
    if bw.b_q != 0 {
        // Qwen2 has Q bias; Qwen3 doesn't (BlockGpu.b_q == 0 for qwen3).
        aether_op_bias_add_f32_cuda(act.q, bw.b_q, 1, d_model);
    }
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_k, act.k_step,
        d_kv, (bw.nb_kv / cfg.d_kv) as c_int);
    if bw.b_k != 0 {
        aether_op_bias_add_f32_cuda(act.k_step, bw.b_k, 1, d_kv);
    }
    if bw.dt_v == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step,
            d_kv, (bw.nb_kv / cfg.d_kv) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.x_norm, bw.w_v, act.v_step,
            d_kv, (bw.nb_kv / cfg.d_kv) as c_int);
    }
    if bw.b_v != 0 {
        aether_op_bias_add_f32_cuda(act.v_step, bw.b_v, 1, d_kv);
    }
    // FR-17-extra-qwen3-fwd — per-head Q/K RMS norm (Qwen3-style).
    // gamma shape [head_dim] is broadcast across heads; applied to each head's
    // head_dim-sized slice via rms_norm with rows=n_q_heads, d=head_dim.
    if bw.attn_q_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(
            act.q, bw.attn_q_norm_g, act.q,
            norm_eps, n_q_heads, head_dim);
    }
    if bw.attn_k_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(
            act.k_step, bw.attn_k_norm_g, act.k_step,
            norm_eps, n_kv_heads, head_dim);
    }
    aether_op_rope_apply_devarg_f32_cuda(act.q,
        1, n_q_heads, head_dim, rope_base, step_args);
    aether_op_rope_apply_devarg_f32_cuda(act.k_step,
        1, n_kv_heads, head_dim, rope_base, step_args);
    let scale: f32 = 1.0 / (cfg.head_dim as f32).sqrt();
    if let Some((page_table_dev, block_size)) = paged_cfg {
        aether_op_paged_append_kv_devarg_f32_cuda(
            act.k_step, act.v_step, kv.k_cache, kv.v_cache, page_table_dev,
            d_kv, block_size, step_args);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            act.q, kv.k_cache, kv.v_cache, page_table_dev, act.attn_out,
            n_q_heads, n_kv_heads, head_dim,
            block_size, scale, max_seq as c_int, step_args);
    } else {
        aether_op_append_kv_devarg_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache,
            d_kv, step_args);
        aether_op_attention_seq1_devarg_f32_cuda(
            act.q, kv.k_cache, kv.v_cache, act.attn_out,
            n_q_heads, n_kv_heads, head_dim, scale,
            max_seq as c_int, step_args);
    }
    aether_op_fused_q4k_matmul_seq1_v2_cuda(act.attn_out, bw.w_o, act.proj,
        d_model, (bw.nb_qo / cfg.d_model) as c_int);
    aether_op_add_inplace_f32_cuda(act.x, act.proj, d_model);
    aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, norm_eps, 1, d_model);
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
        act.x_norm, bw.w_gate, bw.w_up, act.gate,
        d_ff, (bw.nb_gate_up / cfg.d_ff) as c_int);
    if bw.dt_down == 14 {
        aether_op_fused_q6k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
            d_model, (bw.nb_down / cfg.d_model) as c_int);
    } else {
        aether_op_fused_q4k_matmul_seq1_v2_cuda(act.gate, bw.w_down, act.down,
            d_model, (bw.nb_down / cfg.d_model) as c_int);
    }
    aether_op_add_inplace_f32_cuda(act.x, act.down, d_model);
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
    /// FR-19.4-extra paged-KV mode.  When Some, kvs[i].k_cache / v_cache point
    /// at the per-layer KV POOL (size pool_blocks * block_size * d_kv f32);
    /// `page_table_dev` holds an identity mapping [0,1,..,pool_blocks-1] used
    /// by the paged kernels.  Bit-identical to contiguous mode at identity
    /// mapping (proven in cuda_paged_kv_parity.rs).
    paged_cfg: Option<PagedCfg>,
    /// FR-19.4-extra-tenant: when Some, this session shares the per-layer
    /// pools with other sessions; `owned_blocks` tracks the blocks this
    /// session currently holds (returned to the pool on Drop).
    pool: Option<std::sync::Arc<SharedKvPool>>,
    owned_blocks: Vec<i32>,
    page_table_host: Vec<i32>,
    /// FR-17-extra-runtime-shape — runtime shape from GGUF metadata.
    /// Falls back to Qwen2.5-7B if metadata absent.
    pub cfg: ModelConfig,
}

struct PagedCfg {
    page_table_dev: i64,
    block_size: i32,
}

impl QwenSession {
    /// Open a GGUF + upload all weights to GPU.  Default: contiguous KV.
    pub fn new(gguf_path: &str) -> Result<Self, String> {
        Self::new_with_mode(gguf_path, false)
    }
    /// Construct with explicit KV-cache mode.  `paged = true` routes K/V
    /// reads/writes through `paged_append_kv_devarg` + `paged_attention_seq1_devarg`
    /// against an identity page table.  Bit-identical to contiguous mode but
    /// exercises the FR-19.4-extra paged path end-to-end in the real decoder.
    pub fn new_paged(gguf_path: &str) -> Result<Self, String> {
        Self::new_with_mode(gguf_path, true)
    }

    /// Multi-tenant constructor.  Binds this session to a `SharedKvPool`
    /// (per-layer GPU pools shared across multiple sessions).  Allocates
    /// blocks from the pool dynamically as the session's position advances
    /// past block_size boundaries.  Returns the blocks to the pool on Drop.
    ///
    /// The kernels use `pool.pool_k(layer)` / `pool.pool_v(layer)` as the
    /// per-layer K/V base pointers; the per-session page_table_dev maps
    /// logical block index -> physical block id within the pool.  Multiple
    /// concurrent sessions running on the same model + the same pool are
    /// independent because each has its own page_table.
    pub fn new_paged_with_pool(
        gguf_path: &str, pool: std::sync::Arc<SharedKvPool>,
    ) -> Result<Self, String> {
        let mut s = Self::new_with_mode(gguf_path, true)?;
        unsafe { s.rebind_to_shared_pool(pool)?; }
        Ok(s)
    }

    fn new_with_mode(gguf_path: &str, paged: bool) -> Result<Self, String> {
        if !std::path::Path::new(gguf_path).exists() {
            return Err(format!("GGUF not found: {}", gguf_path));
        }
        unsafe {
            aether_dev_init();
            let h = aether_gguf_open(gguf_path.as_ptr() as i64, gguf_path.len() as c_int);
            if h < 0 {
                return Err(format!("aether_gguf_open failed: {}", h));
            }

            // FR-17-extra-runtime-shape: read shape from GGUF metadata.
            // Qwen2.5-7B reads back to the 7B defaults; Qwen2.5-14B picks up
            // 48 blocks / d=5120 / 40 heads / 8 KV heads / D_FF=13824.
            let cfg = ModelConfig::from_gguf(h);
            eprintln!("[QwenSession] arch={} layers={} d_model={} heads_q={} heads_kv={} head_dim={} d_ff={} vocab={} rope={} eps={:.2e}",
                cfg.arch, cfg.n_layers, cfg.d_model, cfg.n_q_heads, cfg.n_kv_heads,
                cfg.head_dim, cfg.d_ff, cfg.vocab, cfg.rope_base, cfg.norm_eps);
            // Kernel constraints (FR-17-extra-runtime-shape).  The fused
            // kernels work for any Qwen-style shape that satisfies these
            // bounds.  Everything else (n_layers, d_model, d_ff, vocab)
            // flows through as a runtime dim into the launch args.
            //   - head_dim must be a multiple of 32 and <= 256
            //     (attention_seq1 lays out per_lane = head_dim >> 5 with
            //      8 slots per lane).
            //   - n_q_heads must be divisible by n_kv_heads (GQA invariant).
            //   - d_model must be a multiple of 256 (Q4_K super-block size).
            //   - d_kv must be a multiple of 256.
            if cfg.head_dim == 0 || cfg.head_dim % 32 != 0 || cfg.head_dim > 256 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: unsupported head_dim={}.  \
                     Kernel supports head_dim ∈ {{32, 64, 96, 128, 160, 192, 224, 256}}.",
                    cfg.head_dim));
            }
            if cfg.n_kv_heads == 0 || cfg.n_q_heads % cfg.n_kv_heads != 0 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: n_q_heads({}) must be a multiple of n_kv_heads({}).",
                    cfg.n_q_heads, cfg.n_kv_heads));
            }
            if cfg.d_model == 0 || cfg.d_model % 256 != 0 || cfg.d_kv == 0 || cfg.d_kv % 256 != 0 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: d_model({}) and d_kv({}) must both be multiples of 256 (Q4_K super-block).",
                    cfg.d_model, cfg.d_kv));
            }

            let blocks: Vec<BlockGpu> = (0..cfg.n_layers).map(|b| load_block(h, b)).collect();
            let final_norm_g = upload_f32_tensor(h, "output_norm.weight");
            let (lm_head, lm_n_blocks, lm_dt) = upload_tensor_u8(h, "output.weight");

            let act = ActivationGpu {
                x: aether_dev_alloc_f32(cfg.d_model as c_int),
                x_norm: aether_dev_alloc_f32(cfg.d_model as c_int),
                q: aether_dev_alloc_f32(cfg.d_model as c_int),
                k_step: aether_dev_alloc_f32(cfg.d_kv as c_int),
                v_step: aether_dev_alloc_f32(cfg.d_kv as c_int),
                attn_out: aether_dev_alloc_f32(cfg.d_model as c_int),
                proj: aether_dev_alloc_f32(cfg.d_model as c_int),
                gate: aether_dev_alloc_f32(cfg.d_ff as c_int),
                down: aether_dev_alloc_f32(cfg.d_model as c_int),
                logits: aether_dev_alloc_f32(cfg.vocab as c_int),
            };
            let kvs: Vec<KvCacheGpu> = (0..cfg.n_layers).map(|_| KvCacheGpu {
                k_cache: aether_dev_alloc_f32((MAX_SEQ * cfg.d_kv) as c_int),
                v_cache: aether_dev_alloc_f32((MAX_SEQ * cfg.d_kv) as c_int),
            }).collect();
            let step_args = aether_dev_alloc_i32(4);  // [pos, cur_seq, 0, 0]

            // Paged-KV: identity mapping of block_size=4 logical blocks to
            // physical blocks [0, 1, ..., MAX_SEQ/block_size - 1].  Same
            // memory layout as contiguous; the kernel walks the page_table
            // for every K/V access.
            let paged_cfg = if paged {
                const BLOCK_SIZE: i32 = 4;
                let n_blocks = (MAX_SEQ as i32) / BLOCK_SIZE;
                let pt_dev = aether_dev_alloc_i32(n_blocks);
                let pt_host: Vec<i32> = (0..n_blocks).collect();
                aether_dev_h2d_i32(pt_host.as_ptr() as i64, pt_dev, n_blocks);
                Some(PagedCfg { page_table_dev: pt_dev, block_size: BLOCK_SIZE })
            } else {
                None
            };

            let (bpe_handle, eos_token) = load_tokenizer_from_gguf(h);
            let gpt2_u2b = build_gpt2_unicode_to_byte();
            Ok(QwenSession {
                gguf_handle: h, blocks, final_norm_g,
                lm_head, lm_n_blocks, lm_dt,
                act, kvs, step_args,
                next_pos: 0,
                graph_captured: false,
                bpe_handle, gpt2_u2b, eos_token,
                paged_cfg,
                pool: None,
                owned_blocks: Vec::new(),
                page_table_host: Vec::new(),
                cfg,
            })
        }
    }

    fn paged_arg(&self) -> Option<(i64, i32)> {
        self.paged_cfg.as_ref().map(|p| (p.page_table_dev, p.block_size))
    }

    /// Switch the session's per-layer KV pointers to a SharedKvPool's pool_k /
    /// pool_v.  Frees the per-session pool storage allocated by
    /// `new_with_mode(_, paged=true)` and replaces page_table_dev contents with
    /// a single initial block allocated from the pool.  The page_table grows
    /// dynamically in `ensure_block_for_position`.
    unsafe fn rebind_to_shared_pool(&mut self, pool: std::sync::Arc<SharedKvPool>) -> Result<(), String> {
        // Free the per-session pool buffers — replace with shared pool pointers.
        for kv in self.kvs.iter_mut() {
            let _ = aether_dev_free_f32(kv.k_cache);
            let _ = aether_dev_free_f32(kv.v_cache);
        }
        for (i, kv) in self.kvs.iter_mut().enumerate() {
            kv.k_cache = pool.pool_k(i);
            kv.v_cache = pool.pool_v(i);
        }
        // Resize page_table_dev — needs MAX_SEQ/block_size logical slots, but
        // we already allocated that in new_with_mode for the per-session case;
        // it's fine to reuse the same device alloc.  Init host-side mirror to
        // "all unmapped" and allocate the first block.
        let block_size = self.paged_cfg.as_ref().ok_or("paged_cfg required")?.block_size;
        let n_logical = (MAX_SEQ as i32 + block_size - 1) / block_size;
        self.page_table_host = vec![-1i32; n_logical as usize];
        let b0 = pool.allocate_block();
        if b0 < 0 { return Err("pool exhausted at first allocate".into()); }
        self.page_table_host[0] = b0;
        self.owned_blocks.push(b0);
        if let Some(p) = &self.paged_cfg {
            aether_dev_h2d_i32(self.page_table_host.as_ptr() as i64, p.page_table_dev, n_logical);
        }
        self.pool = Some(pool);
        Ok(())
    }

    /// If `pos` falls into a logical block that isn't yet mapped, allocate
    /// a new physical block from the pool and update page_table_dev.  No-op
    /// when not in shared-pool mode (the per-session pool is fully identity-
    /// mapped from new_with_mode).
    unsafe fn ensure_block_for_position(&mut self, pos: i32) -> Result<(), &'static str> {
        let Some(p) = &self.paged_cfg else { return Ok(()); };
        let Some(pool) = self.pool.clone() else { return Ok(()); };
        let logical = pos / p.block_size;
        if logical < 0 { return Err("negative position"); }
        let li = logical as usize;
        if li < self.page_table_host.len() && self.page_table_host[li] >= 0 {
            return Ok(()); // already mapped
        }
        if li >= self.page_table_host.len() {
            self.page_table_host.resize(li + 1, -1);
        }
        let b = pool.allocate_block();
        if b < 0 { return Err("pool exhausted"); }
        self.page_table_host[li] = b;
        self.owned_blocks.push(b);
        // H2D the updated page_table.
        aether_dev_h2d_i32(
            self.page_table_host.as_ptr() as i64,
            p.page_table_dev,
            self.page_table_host.len() as c_int,
        );
        Ok(())
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
        let total_rows = n_elems / self.cfg.d_model;
        let dptr = aether_gguf_get_tensor_data_ptr(self.gguf_handle, idx) as *const u8;
        let blocks_per_row = self.cfg.d_model / 256;
        let bytes_per_row = blocks_per_row * 144;
        assert!(token_id < total_rows, "token_id {} out of vocab {}", token_id, total_rows);
        let row_bytes = std::slice::from_raw_parts(
            dptr.add(token_id * bytes_per_row), bytes_per_row);
        let mut row_f32 = vec![0.0f32; self.cfg.d_model];
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
                let pos = i as i32;
                // Shared-pool mode: ensure the logical block for this pos is
                // mapped to a physical block.  No-op for per-session paged or
                // contiguous modes.
                if let Err(e) = self.ensure_block_for_position(pos) {
                    panic!("[QwenSession.prefill] pool allocation failed at pos {}: {}", pos, e);
                }
                let emb = self.dequant_embd_row(t_id);
                aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, self.cfg.d_model as c_int);
                let cur_seq = pos + 1;
                let step_host = [pos, cur_seq, 0i32, 0i32];
                aether_dev_h2d_i32(step_host.as_ptr() as i64, self.step_args, 4);
                for b in 0..self.cfg.n_layers {
                    block_forward_devarg(&self.blocks[b], &self.act, &self.kvs[b], self.step_args, self.paged_arg(), &self.cfg, MAX_SEQ);
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
        for b in 0..self.cfg.n_layers {
            block_forward_devarg(&self.blocks[b], &self.act, &self.kvs[b], self.step_args, self.paged_arg(), &self.cfg, MAX_SEQ);
        }
        aether_op_rms_norm_f32_cuda(
            self.act.x, self.final_norm_g, self.act.x_norm,
            self.cfg.norm_eps, 1, self.cfg.d_model as c_int);
        if self.lm_dt == 14 {
            aether_op_fused_q6k_matmul_seq1_v2_cuda(
                self.act.x_norm, self.lm_head, self.act.logits,
                self.cfg.vocab as c_int, (self.lm_n_blocks / self.cfg.vocab) as c_int);
        } else {
            aether_op_fused_q4k_matmul_seq1_v2_cuda(
                self.act.x_norm, self.lm_head, self.act.logits,
                self.cfg.vocab as c_int, (self.lm_n_blocks / self.cfg.vocab) as c_int);
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
            let pos = self.next_pos;
            // Shared-pool mode may need a fresh block when pos crosses a
            // block_size boundary; no-op otherwise.
            if let Err(e) = self.ensure_block_for_position(pos) {
                panic!("[QwenSession.decode_step] pool allocation failed at pos {}: {}", pos, e);
            }
            // Feed input embedding + step args.
            let emb = self.dequant_embd_row(last_id);
            aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, self.cfg.d_model as c_int);
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

            let mut logits = vec![0.0f32; self.cfg.vocab];
            aether_dev_d2h_f32(self.act.logits, logits.as_mut_ptr() as i64, self.cfg.vocab as c_int);
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
                // Optional tensors — only free if present.
                if b.b_q != 0 { let _ = aether_dev_free_f32(b.b_q); }
                if b.b_k != 0 { let _ = aether_dev_free_f32(b.b_k); }
                if b.b_v != 0 { let _ = aether_dev_free_f32(b.b_v); }
                if b.attn_q_norm_g != 0 { let _ = aether_dev_free_f32(b.attn_q_norm_g); }
                if b.attn_k_norm_g != 0 { let _ = aether_dev_free_f32(b.attn_k_norm_g); }
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
            let shared = self.pool.is_some();
            for kv in self.kvs.drain(..) {
                if !shared {
                    // Per-session pool buffer — owned by us, must be freed here.
                    // In shared-pool mode, k_cache/v_cache point at the pool's
                    // buffers; the SharedKvPool Drop frees them.
                    let _ = aether_dev_free_f32(kv.k_cache);
                    let _ = aether_dev_free_f32(kv.v_cache);
                }
            }
            if let Some(pool) = self.pool.take() {
                for b in self.owned_blocks.drain(..) {
                    pool.free_block(b);
                }
            }
            let _ = aether_dev_free_i32(self.step_args);
            if let Some(p) = self.paged_cfg.take() {
                let _ = aether_dev_free_i32(p.page_table_dev);
            }
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
