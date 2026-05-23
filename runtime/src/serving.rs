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
    aether_op_paged_attention_flex_devarg_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_silu_f32_cuda,
    aether_op_scale_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_op_fused_f16_matmul_seq1_cuda,
    aether_op_fused_q4k_expert_matmul_seq1_cuda,
    aether_op_matmul_f32_cuda,
    aether_dev_graph_begin, aether_dev_graph_end,
    aether_dev_graph_launch, aether_dev_graph_destroy,
};

/// Dispatch matmul kernel based on weight dtype.  Routes F16/Q4_K/Q6_K to
/// the appropriate fused kernel.  For Q4_K and Q6_K, `nb_units` = number of
/// 256-elem super-blocks; for F16, `nb_units` = number of elements.
/// `n_out` = output rows, `n_in` = input cols (= d_model or d_kv etc.).
unsafe fn dispatch_matmul(
    x_norm: i64, w: i64, dt: i32, y: i64, n_out: c_int, n_in: c_int,
) {
    match dt {
        12 => {
            // Q4_K: 256-elem super-blocks; blocks_per_row = n_in / 256.
            aether_op_fused_q4k_matmul_seq1_v2_cuda(x_norm, w, y, n_out, n_in / 256);
        }
        14 => {
            aether_op_fused_q6k_matmul_seq1_v2_cuda(x_norm, w, y, n_out, n_in / 256);
        }
        1 => {
            // F16 (FR-17-extra-f16-fwd).  Weights stored row-major
            // [n_out * n_in] as raw F16.
            aether_op_fused_f16_matmul_seq1_cuda(x_norm, w, y, n_in, n_out);
        }
        _ => panic!("dispatch_matmul: unsupported weight dtype {}", dt),
    }
}

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
    /// FR-17-extra-moe-fwd: Mixture-of-Experts.  0 = dense FFN; >0 = MoE
    /// with this many total experts.
    pub n_experts: usize,
    /// MoE top-k: number of experts routed per token.  Unused when n_experts=0.
    pub n_experts_used: usize,
    /// FR-17-extra-gemma-fwd: sliding-window attention scope.  0 = full
    /// attention (default).  > 0 = restrict attention to last N positions.
    /// Gemma3 specifically alternates sliding/full per layer (per-layer
    /// alternation is a future refinement; today this is a uniform setting).
    pub sliding_window: i32,
    /// FR-17-extra-mla-fwd — DeepSeek-V2 Multi-head Latent Attention.
    /// 0 = standard attention (default).  >0 = use MLA with this latent KV
    /// rank.  When > 0 the per-block tensor layout switches to (attn_kv_a_mqa,
    /// attn_kv_a_norm, attn_kv_b, attn_q [+optional q_a/q_b]) and the per-
    /// head K/V dims become (qk_head_dim, v_head_dim) rather than head_dim
    /// for both.
    pub kv_lora_rank: i32,
    /// MLA: low-rank Q projection rank.  0 = direct attn_q (no Q LoRA, used
    /// by DeepSeek-V2-Lite).  >0 = attn_q_a + attn_q_a_norm + attn_q_b path.
    pub q_lora_rank: i32,
    /// MLA: per-head Q/K dim = qk_nope_head_dim + qk_rope_head_dim
    /// (e.g. 128 + 64 = 192 for V2-Lite).  0 = N/A.
    pub qk_head_dim: i32,
    /// MLA: subset of qk_head_dim that gets rotary applied.  K_rope is
    /// SHARED across heads in MLA (a single qk_rope_head_dim vector per
    /// token), while Q_rope is per-head.  0 = N/A.
    pub qk_rope_head_dim: i32,
    /// MLA: per-head V dim (e.g. 128 for V2-Lite).  Different from qk_head_dim.
    /// 0 = N/A.
    pub v_head_dim: i32,
    /// MLA: number of leading blocks that use the DENSE FFN (instead of
    /// MoE).  DeepSeek-V2-Lite has 1 leading dense block (layer 0).
    /// Layers in [0, leading_dense_blocks) are dense; layers in
    /// [leading_dense_blocks, n_layers) are MoE.  0 = all blocks MoE
    /// (when n_experts > 0).  Unused when n_experts == 0.
    pub leading_dense_blocks: i32,
    /// MLA: number of always-on shared experts (in addition to top-k routed
    /// experts).  DeepSeek-V2-Lite uses 2 shared experts.  0 = no shared
    /// experts (Qwen3-MoE).  Unused when n_experts == 0.
    pub n_shared_experts: i32,
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
            n_experts: 0, n_experts_used: 0,
            sliding_window: 0,
            kv_lora_rank: 0, q_lora_rank: 0,
            qk_head_dim: 0, qk_rope_head_dim: 0, v_head_dim: 0,
            leading_dense_blocks: 0, n_shared_experts: 0,
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
        // MoE — present in qwen3moe / qwen3vlmoe / deepseek2 / mixtral / etc.
        // `expert_count` is the total expert pool; `expert_used_count` is top-k.
        let n_experts = read_meta_u32(gguf_handle, &format!("{}.expert_count", prefix))
            .map(|v| v as usize).unwrap_or(0);
        let n_experts_used = read_meta_u32(gguf_handle, &format!("{}.expert_used_count", prefix))
            .map(|v| v as usize).unwrap_or(0);
        let sliding_window = read_meta_u32(gguf_handle,
            &format!("{}.attention.sliding_window", prefix))
            .map(|v| v as i32).unwrap_or(0);
        // FR-17-extra-mla-fwd — DeepSeek-V2 MLA keys.
        //   deepseek2.attention.kv_lora_rank      (e.g. 512)
        //   deepseek2.attention.q_lora_rank       (optional; absent for V2-Lite)
        //   deepseek2.attention.key_length        (qk_nope + qk_rope, e.g. 192)
        //   deepseek2.attention.value_length      (v_head_dim, e.g. 128)
        //   deepseek2.rope.dimension_count        (qk_rope_head_dim, e.g. 64)
        //   deepseek2.leading_dense_block_count   (e.g. 1)
        //   deepseek2.expert_shared_count         (e.g. 2)
        let kv_lora_rank = read_meta_u32(gguf_handle,
            &format!("{}.attention.kv_lora_rank", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let q_lora_rank = read_meta_u32(gguf_handle,
            &format!("{}.attention.q_lora_rank", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let qk_head_dim = read_meta_u32(gguf_handle,
            &format!("{}.attention.key_length", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let v_head_dim = read_meta_u32(gguf_handle,
            &format!("{}.attention.value_length", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let qk_rope_head_dim = read_meta_u32(gguf_handle,
            &format!("{}.rope.dimension_count", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let leading_dense_blocks = read_meta_u32(gguf_handle,
            &format!("{}.leading_dense_block_count", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let n_shared_experts = read_meta_u32(gguf_handle,
            &format!("{}.expert_shared_count", prefix))
            .map(|v| v as i32).unwrap_or(0);
        Self {
            d_model, n_layers, n_q_heads, n_kv_heads, head_dim, d_kv,
            d_ff, vocab, rope_base, norm_eps, arch,
            n_experts, n_experts_used,
            sliding_window,
            kv_lora_rank, q_lora_rank,
            qk_head_dim, qk_rope_head_dim, v_head_dim,
            leading_dense_blocks, n_shared_experts,
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
    w_v: i64,
    w_down: i64,
    /// Per-tensor dtypes (12=Q4_K, 14=Q6_K, 1=F16).  Mixed-quant GGUFs
    /// (Q4_K_M, Qwen3-Q4_K_M-with-F16-V) need per-tensor dispatch.
    dt_q: i32, dt_k: i32, dt_v: i32, dt_o: i32,
    dt_gate: i32, dt_up: i32, dt_down: i32,
    /// Q/K/V biases — present in Qwen2.5 (qwen2 arch).  Qwen3 dropped these.
    /// 0 indicates "absent".
    b_q: i64, b_k: i64, b_v: i64,
    /// FR-17-extra-qwen3-fwd — per-head Q/K RMS norm (Qwen3-style).
    /// 0 indicates "absent" (qwen2 / older archs).
    attn_q_norm_g: i64, attn_k_norm_g: i64,
    /// FR-17-extra-gemma-fwd — post-attention + post-FFN RMSNorm.
    /// Gemma3 places extra RMS norms AFTER the attention output projection
    /// and AFTER the FFN down projection, BEFORE the residual add.  Qwen
    /// archs don't have these.  0 = absent.
    post_attn_norm_g: i64, post_ffn_norm_g: i64,
    /// `nb_*` semantics: for Q4_K/Q6_K, # of 256-elem super-blocks;
    /// for F16, # of elements.  See `upload_tensor_u8` for the contract.
    nb_qo: usize, nb_kv: usize, nb_gate_up: usize, nb_down: usize,
    /// FR-17-extra-moe-fwd — MoE expert weights.  All 0 when arch is dense.
    /// w_router: F32 device buffer [d_model × n_experts], stored as f32.
    /// w_*_exps: u8 device buffer holding n_experts concatenated expert
    /// weights in the underlying quant dtype (typically Q4_K).
    w_router: i64,
    w_gate_exps: i64, w_up_exps: i64, w_down_exps: i64,
    dt_gate_exps: i32, dt_up_exps: i32, dt_down_exps: i32,
    /// FR-17-extra-mla-fwd — DeepSeek-V2 MLA per-block tensors.
    /// All 0 when the arch isn't MLA.
    ///   w_kv_a_mqa: [d_model x (kv_lora_rank + qk_rope_head_dim)] (Q4_K)
    ///   attn_kv_a_norm_g: [kv_lora_rank] (F32) — RMS norm gain on c_kv latent
    ///   w_kv_b: [kv_lora_rank x (n_heads * (qk_nope_head_dim + v_head_dim))] (Q4_K)
    ///   w_q_a / attn_q_a_norm_g / w_q_b: present iff q_lora_rank > 0.
    ///     w_q_a:   [d_model x q_lora_rank]
    ///     attn_q_a_norm_g: [q_lora_rank]
    ///     w_q_b:   [q_lora_rank x (n_heads * qk_head_dim)]
    ///   When q_lora_rank == 0 the existing `w_q` field holds the direct
    ///   [d_model x (n_heads * qk_head_dim)] projection.
    w_kv_a_mqa: i64,
    attn_kv_a_norm_g: i64,
    w_kv_b: i64,
    w_q_a: i64,
    attn_q_a_norm_g: i64,
    w_q_b: i64,
    dt_kv_a_mqa: i32, dt_kv_b: i32, dt_q_a: i32, dt_q_b: i32,
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
    // For block-quantized tensors (Q4_K / Q6_K), "n_blocks" counts 256-elem
    // super-blocks.  For F16, we return n_elems as the second tuple element
    // so callers can do `nb / d_model` and get the row count regardless of
    // the underlying packing.
    let (n_units, bytes) = match dt {
        12 => { let nb = n_elems / 256; (nb, nb * 144) }     // Q4_K
        14 => { let nb = n_elems / 256; (nb, nb * 210) }     // Q6_K
        1  => { (n_elems, n_elems * 2) }                     // F16 (2 bytes/elem)
        _  => panic!("unsupported dtype {} for tensor {}", dt, name),
    };
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let d_handle = aether_dev_alloc_u8(bytes as c_int);
    aether_dev_h2d_u8(dptr, d_handle, bytes as c_int);
    (d_handle, n_units, dt)
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

/// Optional u8 tensor loader for MoE expert weights.  Returns (handle, n_units, dt)
/// where n_units is 256-elem blocks for Q4_K/Q6_K and elem count for F16.
/// Returns (0, 0, 0) if absent.
unsafe fn upload_tensor_u8_opt(h: i64, name: &str) -> (i64, usize, i32) {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    if idx < 0 { return (0, 0, 0); }
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let (n_units, bytes) = match dt {
        12 => { let nb = n_elems / 256; (nb, nb * 144) }
        14 => { let nb = n_elems / 256; (nb, nb * 210) }
        1  => { (n_elems, n_elems * 2) }
        _  => return (0, 0, 0),
    };
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let d_handle = aether_dev_alloc_u8(bytes as c_int);
    aether_dev_h2d_u8(dptr, d_handle, bytes as c_int);
    (d_handle, n_units, dt)
}

unsafe fn load_block(h: i64, b: usize) -> BlockGpu {
    let p = format!("blk.{}.", b);
    // FR-17-extra-mla-fwd — DeepSeek-V2 MLA blocks use a different K/V
    // layout: attn_kv_a_mqa + attn_kv_a_norm + attn_kv_b instead of
    // attn_k + attn_v.  Detected by the presence of attn_kv_a_mqa.weight.
    // Q can be either direct attn_q (q_lora_rank == 0, V2-Lite) or
    // attn_q_a + attn_q_a_norm + attn_q_b (q_lora_rank > 0, larger V2 vars).
    let (w_kv_a_mqa, _, dt_kv_a_mqa) =
        upload_tensor_u8_opt(h, &format!("{}attn_kv_a_mqa.weight", p));
    let is_mla = w_kv_a_mqa != 0;
    let (w_kv_b, _, dt_kv_b) = if is_mla {
        upload_tensor_u8_opt(h, &format!("{}attn_kv_b.weight", p))
    } else { (0, 0, 0) };
    let (w_q_a, _, dt_q_a) = if is_mla {
        upload_tensor_u8_opt(h, &format!("{}attn_q_a.weight", p))
    } else { (0, 0, 0) };
    let (w_q_b, _, dt_q_b) = if is_mla {
        upload_tensor_u8_opt(h, &format!("{}attn_q_b.weight", p))
    } else { (0, 0, 0) };
    // For MLA blocks attn_k/attn_v don't exist; for non-MLA blocks they do.
    let (w_q, nb_qo, dt_q)         = upload_tensor_u8(h, &format!("{}attn_q.weight", p));
    let (w_k, nb_kv, dt_k)         = if is_mla {
        (0, 0, 0)
    } else {
        upload_tensor_u8(h, &format!("{}attn_k.weight", p))
    };
    let (w_v, _, dt_v)             = if is_mla {
        (0, 0, 0)
    } else {
        upload_tensor_u8(h, &format!("{}attn_v.weight", p))
    };
    let (w_o, _, dt_o)             = upload_tensor_u8(h, &format!("{}attn_output.weight", p));
    // For DENSE FFN the three tensors live under ffn_gate/up/down.  For MoE
    // they live under ffn_gate_exps/up_exps/down_exps and there's a router
    // tensor ffn_gate_inp.  Try DENSE first; fall back to MoE.
    let (w_gate, nb_gate_up, dt_gate) = upload_tensor_u8_opt(h, &format!("{}ffn_gate.weight", p));
    let (w_up, _, dt_up)           = upload_tensor_u8_opt(h, &format!("{}ffn_up.weight", p));
    let (w_down, nb_down, dt_down) = upload_tensor_u8_opt(h, &format!("{}ffn_down.weight", p));
    let (w_gate_exps, _, dt_gate_exps) = upload_tensor_u8_opt(h, &format!("{}ffn_gate_exps.weight", p));
    let (w_up_exps,   _, dt_up_exps)   = upload_tensor_u8_opt(h, &format!("{}ffn_up_exps.weight", p));
    let (w_down_exps, _, dt_down_exps) = upload_tensor_u8_opt(h, &format!("{}ffn_down_exps.weight", p));
    let w_router = upload_f32_tensor_opt(h, &format!("{}ffn_gate_inp.weight", p));
    if w_gate == 0 && w_gate_exps == 0 {
        panic!("blk.{} has neither dense ffn_gate nor MoE ffn_gate_exps", b);
    }
    BlockGpu {
        attn_norm_g: upload_f32_tensor(h, &format!("{}attn_norm.weight", p)),
        ffn_norm_g:  upload_f32_tensor(h, &format!("{}ffn_norm.weight", p)),
        w_q, w_k, w_o, w_gate, w_up,
        w_v, w_down,
        dt_q, dt_k, dt_v, dt_o, dt_gate, dt_up, dt_down,
        b_q: upload_f32_tensor_opt(h, &format!("{}attn_q.bias", p)),
        b_k: upload_f32_tensor_opt(h, &format!("{}attn_k.bias", p)),
        b_v: upload_f32_tensor_opt(h, &format!("{}attn_v.bias", p)),
        attn_q_norm_g: upload_f32_tensor_opt(h, &format!("{}attn_q_norm.weight", p)),
        attn_k_norm_g: upload_f32_tensor_opt(h, &format!("{}attn_k_norm.weight", p)),
        // Gemma3 names these post_attention_norm.weight + post_ffw_norm.weight.
        // We accept either spelling for forward compatibility.
        post_attn_norm_g: upload_f32_tensor_opt(h, &format!("{}post_attention_norm.weight", p)),
        post_ffn_norm_g:  upload_f32_tensor_opt(h, &format!("{}post_ffw_norm.weight", p)),
        nb_qo, nb_kv, nb_gate_up, nb_down,
        w_router, w_gate_exps, w_up_exps, w_down_exps,
        dt_gate_exps, dt_up_exps, dt_down_exps,
        w_kv_a_mqa,
        attn_kv_a_norm_g: upload_f32_tensor_opt(h, &format!("{}attn_kv_a_norm.weight", p)),
        w_kv_b,
        w_q_a,
        attn_q_a_norm_g: upload_f32_tensor_opt(h, &format!("{}attn_q_a_norm.weight", p)),
        w_q_b,
        dt_kv_a_mqa, dt_kv_b, dt_q_a, dt_q_b,
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
    // FR-17-extra-mla-fwd — DeepSeek-V2 MLA branch.  The MLA attention kernel
    // + paged append kernel are landed and CPU↔GPU bit-witnessed in
    // runtime/tests/cuda_attention_mla_parity.rs.  The pre-attention plumbing
    // (compressed-KV projection + per-head decompression + RoPE on partial
    // dims + Q/K composition) needs a few small fan-out kernels that aren't
    // wired into this forward path yet.  Detect the MLA layout and stop with
    // a clear error message so a user loading DeepSeek-V2 doesn't get
    // silently-wrong activations.
    if bw.w_kv_a_mqa != 0 || cfg.kv_lora_rank > 0 {
        panic!("FR-17-extra-mla-fwd: MLA attention kernel + paged-append \
            kernel are landed (paged_attention_mla_devarg / \
            paged_append_kv_mla_devarg), but the full forward integration \
            (c_kv projection + decompression + per-head RoPE composition) \
            is not yet wired into block_forward_devarg.  Detected MLA at \
            cfg.kv_lora_rank={}, w_kv_a_mqa={}.",
            cfg.kv_lora_rank, bw.w_kv_a_mqa);
    }
    dispatch_matmul(act.x_norm, bw.w_q, bw.dt_q, act.q, d_model, d_model);
    if bw.b_q != 0 {
        // Qwen2 has Q bias; Qwen3 doesn't (BlockGpu.b_q == 0 for qwen3).
        aether_op_bias_add_f32_cuda(act.q, bw.b_q, 1, d_model);
    }
    dispatch_matmul(act.x_norm, bw.w_k, bw.dt_k, act.k_step, d_kv, d_model);
    if bw.b_k != 0 {
        aether_op_bias_add_f32_cuda(act.k_step, bw.b_k, 1, d_kv);
    }
    dispatch_matmul(act.x_norm, bw.w_v, bw.dt_v, act.v_step, d_kv, d_model);
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
    // FR-17-extra-gemma-fwd: route via the flex attention kernel when the
    // arch requires features the seq1 kernels don't support: head_dim
    // that isn't a multiple of 32, or sliding-window scope.
    let needs_flex = (cfg.head_dim % 32) != 0 || cfg.sliding_window > 0;
    if let Some((page_table_dev, block_size)) = paged_cfg {
        aether_op_paged_append_kv_devarg_f32_cuda(
            act.k_step, act.v_step, kv.k_cache, kv.v_cache, page_table_dev,
            d_kv, block_size, step_args);
        if needs_flex {
            aether_op_paged_attention_flex_devarg_f32_cuda(
                act.q, kv.k_cache, kv.v_cache, page_table_dev, act.attn_out,
                n_q_heads, n_kv_heads, head_dim,
                block_size, cfg.sliding_window, scale, max_seq as c_int, step_args);
        } else {
            aether_op_paged_attention_seq1_devarg_f32_cuda(
                act.q, kv.k_cache, kv.v_cache, page_table_dev, act.attn_out,
                n_q_heads, n_kv_heads, head_dim,
                block_size, scale, max_seq as c_int, step_args);
        }
    } else {
        if needs_flex {
            panic!("FR-17-extra-gemma-fwd: arches needing flex attention \
                (head_dim%32 != 0 or sliding_window>0) require --paged mode \
                today.  Contiguous-KV flex kernel is a follow-on.");
        }
        aether_op_append_kv_devarg_f32_cuda(act.k_step, act.v_step, kv.k_cache, kv.v_cache,
            d_kv, step_args);
        aether_op_attention_seq1_devarg_f32_cuda(
            act.q, kv.k_cache, kv.v_cache, act.attn_out,
            n_q_heads, n_kv_heads, head_dim, scale,
            max_seq as c_int, step_args);
    }
    dispatch_matmul(act.attn_out, bw.w_o, bw.dt_o, act.proj, d_model, d_model);
    // FR-17-extra-gemma-fwd: Gemma3 applies a POST-attention RMS norm to
    // the output projection BEFORE adding to the residual.  Qwen archs
    // skip this (post_attn_norm_g == 0).
    if bw.post_attn_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(act.proj, bw.post_attn_norm_g, act.proj,
            norm_eps, 1, d_model);
    }
    aether_op_add_inplace_f32_cuda(act.x, act.proj, d_model);
    aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, norm_eps, 1, d_model);

    // FFN dispatch: dense (Qwen2/3/Llama) vs MoE (Qwen3-MoE/DeepSeek-V2).
    if bw.w_router != 0 {
        moe_ffn_forward(bw, act, cfg);
    } else {
        // Dense FFN — gate+up fused (Q4_K only).  If either is F16/Q6_K
        // we'd need a non-fused fallback (FR-17-extra-non-fused-ffn).
        if bw.dt_gate == 12 && bw.dt_up == 12 {
            aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
                act.x_norm, bw.w_gate, bw.w_up, act.gate,
                d_ff, (bw.nb_gate_up / cfg.d_ff) as c_int);
        } else {
            panic!("FFN gate/up dtypes not both Q4_K (got gate={}, up={}); needs a non-fused fallback",
                bw.dt_gate, bw.dt_up);
        }
        dispatch_matmul(act.gate, bw.w_down, bw.dt_down, act.down, d_model, d_ff);
        // FR-17-extra-gemma-fwd: Gemma3 applies a POST-FFN RMS norm to
        // the down output BEFORE the residual add.  Qwen archs skip.
        if bw.post_ffn_norm_g != 0 {
            aether_op_rms_norm_f32_cuda(act.down, bw.post_ffn_norm_g, act.down,
                norm_eps, 1, d_model);
        }
        aether_op_add_inplace_f32_cuda(act.x, act.down, d_model);
    }
}

/// FR-17-extra-moe-fwd — Mixture-of-Experts FFN forward pass.
///
/// Per-token decode path:
///   1. router_logits = W_router @ x_norm                 [n_experts]
///   2. d2h → sort top-k experts on host → softmax routing weights
///   3. For each selected expert e_i with weight w_i:
///        gate_e = Q4K_expert_matmul(x_norm, W_gate_exps, e_i)  [d_ff]
///        up_e   = Q4K_expert_matmul(x_norm, W_up_exps, e_i)    [d_ff]
///        gate_e = silu(gate_e) * up_e                          [d_ff]
///        down_e = quant_matmul(gate_e, W_down_exps_slice_e)    [d_model]
///        out += w_i * down_e
///   4. x += out  (residual)
///
/// This is the SLOW PATH — per-expert dispatch via separate kernel launches
/// (1 router matmul + 2*n_experts_used expert gate/up + n_experts_used down
/// + n_experts_used silu/mul/scale/add per token).  CUDA graph capture
/// disabled while this is active (top-k selection happens on the host).
/// Future stage: fused MoE kernel that does router + top-k + per-expert
/// + combine in one launch with a router-aware dispatcher.
unsafe fn moe_ffn_forward(bw: &BlockGpu, act: &ActivationGpu, cfg: &ModelConfig) {
    let d_model = cfg.d_model;
    let d_ff = cfg.d_ff;
    let n_experts = cfg.n_experts;
    let n_used = cfg.n_experts_used.max(1);
    assert!(n_experts > 0, "moe_ffn_forward called on dense block");

    // Workspace allocations.  These are small and short-lived; we re-alloc
    // per call rather than thread per-session through every kernel signature.
    // TODO: cache them on QwenSession.
    let router_logits = aether_dev_alloc_f32(n_experts as c_int);
    let gate_e = aether_dev_alloc_f32(d_ff as c_int);
    let up_e = aether_dev_alloc_f32(d_ff as c_int);
    let down_e = aether_dev_alloc_f32(d_model as c_int);
    let out_acc = aether_dev_alloc_f32(d_model as c_int);
    // Zero out_acc by H2D of a zero vector (small, ~16 KB).
    let zero = vec![0f32; d_model];
    aether_dev_h2d_f32(zero.as_ptr() as i64, out_acc, d_model as c_int);

    // 1. router_logits = w_router @ x_norm.  w_router is row-major
    // [n_experts × d_model] f32.  aether_op_matmul_f32_cuda computes
    // out = a @ b with a: [m, k], b: [k, n], out: [m, n].
    // We want logits = w_router @ x (column vec).  Set m=n_experts, k=d_model,
    // n=1, a=w_router, b=x_norm.
    aether_op_matmul_f32_cuda(
        bw.w_router, act.x_norm, router_logits,
        n_experts as c_int, d_model as c_int, 1);

    // 2. D2H + top-k + softmax on host.
    let mut logits = vec![0f32; n_experts];
    aether_dev_sync();
    aether_dev_d2h_f32(router_logits, logits.as_mut_ptr() as i64, n_experts as c_int);
    let mut idx_sorted: Vec<usize> = (0..n_experts).collect();
    idx_sorted.sort_unstable_by(|a, b|
        logits[*b].partial_cmp(&logits[*a]).unwrap_or(std::cmp::Ordering::Equal));
    let selected = &idx_sorted[..n_used];
    let max_l = selected.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = selected.iter().map(|&i| (logits[i] - max_l).exp()).collect();
    let sum: f32 = weights.iter().sum();
    for w in &mut weights { *w /= sum; }

    // 3. Per-expert forward.  blocks_per_row_gate_up = d_model/256 (input dim).
    // blocks_per_row_down = d_ff/256.
    let bpr_in = (d_model / 256) as c_int;
    let bpr_ff = (d_ff / 256) as c_int;
    for (k, &expert_idx) in selected.iter().enumerate() {
        let w_i = weights[k];
        // gate_e = expert_matmul(x_norm, w_gate_exps, expert_idx)  [d_ff]
        aether_op_fused_q4k_expert_matmul_seq1_cuda(
            act.x_norm, bw.w_gate_exps, gate_e,
            d_ff as c_int, bpr_in, expert_idx as c_int);
        // up_e = expert_matmul(x_norm, w_up_exps, expert_idx)  [d_ff]
        aether_op_fused_q4k_expert_matmul_seq1_cuda(
            act.x_norm, bw.w_up_exps, up_e,
            d_ff as c_int, bpr_in, expert_idx as c_int);
        // silu(gate_e); gate_e *= up_e
        aether_op_silu_f32_cuda(gate_e, d_ff as c_int);
        aether_op_mul_inplace_f32_cuda(gate_e, up_e, d_ff as c_int);
        // down_e = expert_matmul(gate_e, w_down_exps, expert_idx)  [d_model]
        aether_op_fused_q4k_expert_matmul_seq1_cuda(
            gate_e, bw.w_down_exps, down_e,
            d_model as c_int, bpr_ff, expert_idx as c_int);
        // out_acc += w_i * down_e
        aether_op_scale_f32_cuda(down_e, w_i, d_model as c_int);
        aether_op_add_inplace_f32_cuda(out_acc, down_e, d_model as c_int);
    }

    // 4. x += out_acc (residual).
    aether_op_add_inplace_f32_cuda(act.x, out_acc, d_model as c_int);

    // Free workspaces.
    let _ = aether_dev_free_f32(router_logits);
    let _ = aether_dev_free_f32(gate_e);
    let _ = aether_dev_free_f32(up_e);
    let _ = aether_dev_free_f32(down_e);
    let _ = aether_dev_free_f32(out_acc);
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
            // The flex attention kernel handles any head_dim in [1, 256]
            // (FR-17-extra-gemma-fwd).  Non-multiples-of-32 trigger the
            // flex path in block_forward_devarg.
            if cfg.head_dim == 0 || cfg.head_dim > 256 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: head_dim={} out of range [1, 256] \
                     (attention kernel q_local[8] × per_lane=8 maxes out).",
                    cfg.head_dim));
            }
            if cfg.n_kv_heads == 0 || cfg.n_q_heads % cfg.n_kv_heads != 0 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: n_q_heads({}) must be a multiple of n_kv_heads({}).",
                    cfg.n_q_heads, cfg.n_kv_heads));
            }
            // Q4_K kernels iterate over n_in in 256-elem super-blocks; only the
            // shared (input) dimension needs to be a multiple of 256.  Both
            // d_model (Q/K/V/O/LM-head n_in) and d_ff (down n_in) feed this.
            // Output dims (d_kv, vocab) have no such constraint — the kernel
            // launches one CTA per output row.
            if cfg.d_model == 0 || cfg.d_model % 256 != 0 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: d_model({}) must be a multiple of 256 (Q4_K super-block).",
                    cfg.d_model));
            }
            if cfg.d_ff == 0 || cfg.d_ff % 256 != 0 {
                return Err(format!(
                    "FR-17-extra-runtime-shape: d_ff({}) must be a multiple of 256 (Q4_K super-block).",
                    cfg.d_ff));
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

    /// FR-17-extra-moe-fwd: imperative (non-graph-captured) forward pass.
    /// Used when host-side dispatch is required (MoE routing).  Runs all
    /// 28+ block forwards + final norm + LM head + argmax inputs in the
    /// current call.
    unsafe fn run_forward_imperative(&mut self) {
        for b in 0..self.cfg.n_layers {
            block_forward_devarg(&self.blocks[b], &self.act, &self.kvs[b],
                self.step_args, self.paged_arg(), &self.cfg, MAX_SEQ);
        }
        aether_op_rms_norm_f32_cuda(
            self.act.x, self.final_norm_g, self.act.x_norm,
            self.cfg.norm_eps, 1, self.cfg.d_model as c_int);
        dispatch_matmul(self.act.x_norm, self.lm_head, self.lm_dt, self.act.logits,
            self.cfg.vocab as c_int, self.cfg.d_model as c_int);
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
        dispatch_matmul(self.act.x_norm, self.lm_head, self.lm_dt, self.act.logits,
            self.cfg.vocab as c_int, self.cfg.d_model as c_int);
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
            if let Err(e) = self.ensure_block_for_position(pos) {
                panic!("[QwenSession.decode_step] pool allocation failed at pos {}: {}", pos, e);
            }
            let emb = self.dequant_embd_row(last_id);
            aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, self.cfg.d_model as c_int);
            let cur_seq = pos + 1;
            let step_host = [pos, cur_seq, 0i32, 0i32];
            aether_dev_h2d_i32(step_host.as_ptr() as i64, self.step_args, 4);

            if self.cfg.n_experts > 0 {
                // FR-17-extra-moe-fwd: MoE forward involves host-side top-k
                // routing per layer, which can't be captured into a CUDA
                // graph.  Run the forward imperatively each step.
                self.run_forward_imperative();
            } else {
                if !self.graph_captured {
                    aether_dev_sync();
                    self.capture_graph_now();
                }
                let rc = aether_dev_graph_launch();
                assert_eq!(rc, 0, "aether_dev_graph_launch failed: {}", rc);
            }
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
                // Dense ffn tensors — may be 0 on MoE archs (which use the _exps variants).
                if b.w_gate != 0 { let _ = aether_dev_free_u8(b.w_gate); }
                if b.w_up   != 0 { let _ = aether_dev_free_u8(b.w_up); }
                if b.w_down != 0 { let _ = aether_dev_free_u8(b.w_down); }
                // Optional tensors — only free if present.
                if b.b_q != 0 { let _ = aether_dev_free_f32(b.b_q); }
                if b.b_k != 0 { let _ = aether_dev_free_f32(b.b_k); }
                if b.b_v != 0 { let _ = aether_dev_free_f32(b.b_v); }
                if b.attn_q_norm_g != 0 { let _ = aether_dev_free_f32(b.attn_q_norm_g); }
                if b.attn_k_norm_g != 0 { let _ = aether_dev_free_f32(b.attn_k_norm_g); }
                if b.post_attn_norm_g != 0 { let _ = aether_dev_free_f32(b.post_attn_norm_g); }
                if b.post_ffn_norm_g  != 0 { let _ = aether_dev_free_f32(b.post_ffn_norm_g); }
                // MoE expert weights — only present on qwen3moe/deepseek2.
                if b.w_router != 0 { let _ = aether_dev_free_f32(b.w_router); }
                if b.w_gate_exps != 0 { let _ = aether_dev_free_u8(b.w_gate_exps); }
                if b.w_up_exps != 0 { let _ = aether_dev_free_u8(b.w_up_exps); }
                if b.w_down_exps != 0 { let _ = aether_dev_free_u8(b.w_down_exps); }
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
