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
    aether_dequant_q4_k_m, aether_dequant_q6_k, aether_dequant_q3_k, aether_dequant_iq3_s,
    aether_f16_to_f32,
    aether_gguf_get_metadata_u32, aether_gguf_get_metadata_string,
    aether_gguf_get_metadata_array_string_n,
    aether_gguf_get_metadata_array_string_get,
    aether_bpe_tokenizer_new, aether_bpe_tokenizer_free,
    aether_bpe_add_token_with_id, aether_bpe_add_merge_by_id,
    aether_bpe_decode, aether_bpe_encode_ids, aether_bpe_lookup_bytes,
    aether_template_new, aether_template_free, aether_template_push_message,
    aether_template_render, aether_template_set_var,
};
use crate::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_h2d_f32_n, aether_dev_d2h_f32_n, aether_dev_h2d_i32_n,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_dev_alloc_i32, aether_dev_free_i32, aether_dev_h2d_i32,
    aether_op_rms_norm_f32_cuda,
    aether_op_rope_apply_devarg_f32_cuda,
    aether_op_append_kv_devarg_f32_cuda,
    aether_op_attention_seq1_devarg_f32_cuda,
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
    aether_op_paged_attention_flex_devarg_f32_cuda,
    aether_op_paged_append_kv_mla_devarg_f32_cuda,
    aether_op_paged_attention_mla_devarg_f32_cuda,
    aether_op_mla_split_kv_a_f32_cuda,
    aether_op_mla_assemble_k_f32_cuda,
    aether_op_mla_extract_v_f32_cuda,
    aether_op_mla_rope_q_partial_f32_cuda,
    aether_op_mla_rope_k_shared_f32_cuda,
    aether_op_mla_rope_q_partial_yarn_f32_cuda,
    aether_op_mla_rope_k_shared_yarn_f32_cuda,
    aether_op_mla_absorb_q_q8_0_cuda,
    aether_op_mla_absorb_v_q8_0_cuda,
    aether_op_mla_absorb_q_f16_cuda,
    aether_op_mla_absorb_v_f16_cuda,
    aether_op_mla_absorb_q_q4_k_cuda,
    aether_op_mla_absorb_v_q4_k_cuda,
    aether_op_mla_absorb_q_q5_k_cuda,
    aether_op_mla_absorb_v_q5_k_cuda,
    aether_op_mla_absorb_q_q6_k_cuda,
    aether_op_mla_absorb_v_q6_k_cuda,
    aether_op_mla_absorb_q_iq4_nl_cuda,
    aether_op_mla_absorb_v_iq4_nl_cuda,
    aether_op_mla_broadcast_kv_for_mqa_f32_cuda,
    aether_op_matmul_nt_f32_cuda,
    aether_op_bias_add_f32_cuda, aether_op_add_inplace_f32_cuda,
    aether_op_mul_inplace_f32_cuda, aether_op_silu_f32_cuda,
    aether_op_scale_f32_cuda,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q6k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_op_fused_f16_matmul_seq1_cuda,
    aether_op_fused_f32_matmul_seq1_cuda,
    aether_op_fused_q4_0_matmul_seq1_cuda,
    aether_op_fused_q5_0_matmul_seq1_cuda,
    aether_op_fused_q8_0_matmul_seq1_cuda,
    aether_op_fused_q5_k_matmul_seq1_cuda,
    aether_op_fused_q3_k_matmul_seq1_cuda,
    aether_op_fused_iq4_nl_matmul_seq1_cuda,
    aether_op_fused_iq4_xs_matmul_seq1_cuda,
    aether_op_fused_iq3_xxs_matmul_seq1_cuda,
    aether_op_fused_iq3_s_matmul_seq1_cuda,
    aether_op_fused_q4k_expert_matmul_seq1_cuda,
    aether_op_fused_q8_0_expert_matmul_seq1_cuda,
    aether_op_fused_q5_0_expert_matmul_seq1_cuda,
    aether_op_fused_iq3_s_expert_matmul_seq1_cuda,
    aether_op_fused_q3_k_expert_matmul_seq1_cuda,
    aether_op_fused_q5_k_expert_matmul_seq1_cuda,
    aether_op_fused_iq4_xs_expert_matmul_seq1_cuda,
    aether_op_fused_iq3_xxs_expert_matmul_seq1_cuda,
    aether_op_matmul_f32_cuda,
    aether_op_fused_q4k_matmul_seqB_v3_cuda,
    aether_op_batched_rope_apply_devarg_f32_cuda,
    aether_op_batched_paged_append_kv_hetero_devarg_f32_cuda,
    aether_op_batched_paged_attention_hetero_devarg_f32_cuda,
    aether_dev_d2d_f32_offset,
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
        0 => {
            // F32 (FR-17-extra-mla-fwd).  Some GLM-4.7-flash tensors are
            // stored as raw float32 (notably the shared-expert MLPs and a
            // handful of head-adjacent tensors).  Layout is row-major
            // [n_out, n_in] in a u8-registered buffer (same upload path as
            // F16 / Q4_K / etc.).
            aether_op_fused_f32_matmul_seq1_cuda(x_norm, w, y, n_in, n_out);
        }
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
        2 => {
            // Q4_0 (FR-17-extra-q4_0-fwd).  32-elem blocks of 18 bytes each:
            // f16 scale + 16 nibble-packed bytes.  Used by older / local
            // DeepSeek-V2-Lite and a few other small models.
            aether_op_fused_q4_0_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 32);
        }
        6 => {
            // Q5_0 (FR-17-extra-q5_0-fwd).  22-byte blocks: f16 d + 4-byte
            // qh high-bits + 16 byte nibble-packed quants.  Used by cnc's
            // V2-Lite Q4_K_M for ~half of its ffn_down_exps tensors (the
            // ones whose d_in=1408 doesn't align to Q4_K's 256 super-block).
            aether_op_fused_q5_0_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 32);
        }
        8 => {
            // Q8_0 (FR-17-extra-q8_0-fwd).  34-byte blocks: f16 d + 32 i8
            // quants.  Used by cnc's V2-Lite Q4_K_M for the dense
            // ffn_down (d_in=10944) and the other half of ffn_down_exps.
            aether_op_fused_q8_0_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 32);
        }
        13 => {
            // Q5_K (FR-17-extra-q5_k-fwd).  176-byte 256-elem super-blocks:
            // f16 d + f16 dmin + 12-byte scales (Q4_K shape) + 32-byte qh
            // (high-bits) + 128-byte qs (nibbles).  Used by Qwen2.5-32B
            // Q5_K_M, Llama-3 Q5_K_M, GLM-4.7-flash (~51 tensors), and
            // most modern Q5_K_M GGUFs.
            aether_op_fused_q5_k_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 256);
        }
        11 => {
            // Q3_K (FR-17-extra-q3_k-fwd).  110-byte 256-elem super-blocks:
            // 32-byte hmask + 64-byte qs + 12-byte scales + f16 d.
            // Used by Qwen3-MoE-30B-Q3_K_M (198 tensors), DeepSeek-R1-
            // Distill Q3_K_M, Llama-3 Q3_K_M variants.
            aether_op_fused_q3_k_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 256);
        }
        18 => {
            // IQ3_XXS (FR-17-extra-iq3_xxs-fwd).  98-byte 256-elem blocks:
            // f16 d + 64-byte codebook indices + 32-byte scales_and_signs.
            // Used by cnc's glm-4.7-flash-UD-IQ3_XXS GGUF.
            aether_op_fused_iq3_xxs_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 256);
        }
        21 => {
            // IQ3_S (FR-17-extra-iq3_s-fwd).  110-byte 256-elem blocks:
            // f16 d + 64-byte qs + 8-byte qh + 32-byte signs + 4-byte scales.
            // Per-sub-block odd-integer scale (db = d * (1 + 2*nib)) × 512-entry
            // codebook lookup.  Used by GLM-4.7-flash-UD-IQ3_XXS for ~44 tensors.
            aether_op_fused_iq3_s_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 256);
        }
        20 => {
            // IQ4_NL (FR-17-extra-iq4_nl-fwd).  18-byte 32-elem blocks:
            // f16 d + 16-byte nibble-packed indices into a 16-entry
            // non-linear codebook of signed int8 values.  Used by
            // GLM-4.7-flash for ~72 tensors.
            aether_op_fused_iq4_nl_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 32);
        }
        23 => {
            // IQ4_XS (FR-17-extra-iq4_xs-fwd).  136-byte 256-elem blocks
            // with per-sub-block 6-bit signed scales + kvalues_iq4nl
            // codebook lookup.  Used by GLM-4.7-flash for ~55 tensors.
            aether_op_fused_iq4_xs_matmul_seq1_cuda(x_norm, w, y, n_out, n_in / 256);
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

/// FR-19.5-extra-deep Phase 2b-2b — max requests fused into one batched
/// decode tick.  Capped at 8 because `fused_q4k_matmul_seqB_v3` rejects
/// `batch > 8` (its weight-reuse register budget).  Matches the scheduler's
/// default `--max-concurrent`.
pub const MAX_BATCH: usize = 8;

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
    /// MLA-absorbed mode: per-head Q proj dim that w_q_b OUTPUTS.  For
    /// GLM-4.7-flash this is 256 (= q_nope_in 192 + q_pe 64), not the
    /// full 576 of qk_head_dim.  When > 0, GGUF metadata signals absorbed
    /// MLA — w_k_b / w_v_b absorb per-head decompression into the Q/V paths.
    /// Read from `<arch>.attention.key_length_mla`.
    pub key_length_mla: i32,
    /// MLA-absorbed mode: per-head V output dim from w_v_b.  For
    /// GLM-4.7-flash this is 256.  Read from `<arch>.attention.value_length_mla`.
    pub value_length_mla: i32,
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
    /// MLA / MoE: per-expert FFN hidden dim.  Routed experts use this dim;
    /// the n_shared experts are FUSED into a single MLP with hidden dim
    /// `n_shared_experts * expert_ff_dim`.  DeepSeek-V2-Lite: 1408.
    /// 0 = N/A.  Read from `<arch>.expert_feed_forward_length`.
    pub expert_ff_dim: usize,
    /// FR-17-extra-mla-fwd YaRN — RoPE scaling factor (s).  0 or 1 = no
    /// scaling (standard RoPE).  > 1 = YaRN-by-parts scaling with this
    /// factor.  DeepSeek-V2-Lite uses 40.
    pub yarn_factor: f32,
    /// YaRN attention temperature coefficient.  Final temperature mscale =
    /// 1 + yarn_log_multiplier * ln(yarn_factor).  Applied to attention
    /// scale: final_scale = (1/sqrt(qk_head_dim)) * mscale * mscale.
    /// DeepSeek-V2-Lite stores 0.0707; HF config calls this "mscale_all_dim"
    /// (sometimes pre-multiplied by 0.1).
    pub yarn_log_multiplier: f32,
    /// YaRN original context length (pre-extension).  DeepSeek-V2-Lite uses
    /// 4096.  Used in the per-frequency-dim correction-dim formula.
    pub yarn_orig_ctx: f32,
    /// YaRN ramp bounds in rotation counts.  Defaults from the paper:
    /// beta_fast=32 (high frequency cutoff), beta_slow=1 (low frequency).
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
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
            key_length_mla: 0, value_length_mla: 0,
            leading_dense_blocks: 0, n_shared_experts: 0,
            expert_ff_dim: 0,
            yarn_factor: 1.0, yarn_log_multiplier: 0.0,
            yarn_orig_ctx: 4096.0,
            yarn_beta_fast: 32.0, yarn_beta_slow: 1.0,
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
        // head_dim is usually d_model / n_q_heads, but several llama-family
        // models set an EXPLICIT head_dim via {arch}.attention.key_length that
        // does NOT divide d_model evenly: Mistral Small 24B (32 heads * 128 =
        // 4096 != 5120 hidden), Gemma3 (head_dim 256, not 240), Qwen3-MoE
        // (head_dim 128, not 64). Prefer the explicit value when present and
        // this is NOT an MLA arch (where key_length means the composite
        // qk_nope+qk_rope dim, handled separately below).
        let explicit_head_dim = read_meta_u32(gguf_handle,
            &format!("{}.attention.key_length", prefix)).map(|v| v as usize).unwrap_or(0);
        let is_mla_arch = read_meta_u32(gguf_handle,
            &format!("{}.attention.kv_lora_rank", prefix)).map(|v| v as usize).unwrap_or(0) > 0;
        let head_dim = if explicit_head_dim > 0 && !is_mla_arch {
            explicit_head_dim
        } else if n_q_heads > 0 {
            d_model / n_q_heads
        } else { HEAD_DIM };
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
        let key_length_mla = read_meta_u32(gguf_handle,
            &format!("{}.attention.key_length_mla", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let value_length_mla = read_meta_u32(gguf_handle,
            &format!("{}.attention.value_length_mla", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let leading_dense_blocks = read_meta_u32(gguf_handle,
            &format!("{}.leading_dense_block_count", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let n_shared_experts = read_meta_u32(gguf_handle,
            &format!("{}.expert_shared_count", prefix))
            .map(|v| v as i32).unwrap_or(0);
        let expert_ff_dim = read_meta_u32(gguf_handle,
            &format!("{}.expert_feed_forward_length", prefix))
            .map(|v| v as usize).unwrap_or(0);
        // FR-17-extra-mla-fwd YaRN — long-context RoPE scaling.  Only active
        // when `<arch>.rope.scaling.type == "yarn"`; for "linear" / absent
        // we keep the standard RoPE path.  Defaults match the YaRN paper +
        // DeepSeek-V2 conventions.
        let scaling_type = read_meta_string(gguf_handle,
            &format!("{}.rope.scaling.type", prefix)).unwrap_or_default();
        let (yarn_factor, yarn_log_multiplier, yarn_orig_ctx) =
            if scaling_type == "yarn" {
                let s = read_meta_f32(gguf_handle,
                    &format!("{}.rope.scaling.factor", prefix)).unwrap_or(1.0);
                let log_m = read_meta_f32(gguf_handle,
                    &format!("{}.rope.scaling.yarn_log_multiplier", prefix))
                    .unwrap_or(0.0);
                let orig = read_meta_u32(gguf_handle,
                    &format!("{}.rope.scaling.original_context_length", prefix))
                    .map(|v| v as f32)
                    // Fall back to context_length / factor when the original
                    // isn't stored explicitly.
                    .unwrap_or_else(|| {
                        let ctx = read_meta_u32(gguf_handle,
                            &format!("{}.context_length", prefix))
                            .map(|v| v as f32).unwrap_or(4096.0 * s);
                        ctx / s.max(1.0)
                    });
                (s, log_m, orig)
            } else { (1.0, 0.0, 4096.0) };
        let yarn_beta_fast = read_meta_f32(gguf_handle,
            &format!("{}.rope.scaling.beta_fast", prefix)).unwrap_or(32.0);
        let yarn_beta_slow = read_meta_f32(gguf_handle,
            &format!("{}.rope.scaling.beta_slow", prefix)).unwrap_or(1.0);
        Self {
            d_model, n_layers, n_q_heads, n_kv_heads, head_dim, d_kv,
            d_ff, vocab, rope_base, norm_eps, arch,
            n_experts, n_experts_used,
            sliding_window,
            kv_lora_rank, q_lora_rank,
            qk_head_dim, qk_rope_head_dim, v_head_dim,
            key_length_mla, value_length_mla,
            leading_dense_blocks, n_shared_experts, expert_ff_dim,
            yarn_factor, yarn_log_multiplier, yarn_orig_ctx,
            yarn_beta_fast, yarn_beta_slow,
        }
    }
    /// MLA absorbed-mode detector.  When both `key_length_mla` and
    /// `value_length_mla` GGUF fields are present (> 0), this arch stores
    /// the per-head K/V decompression matrices as separate `attn_k_b` /
    /// `attn_v_b` tensors and uses MQA-style shared compressed KV in
    /// attention.  GLM-4.7-flash sets this; V2-Lite does not.
    pub fn is_mla_absorbed(&self) -> bool {
        self.key_length_mla > 0 && self.value_length_mla > 0
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
    /// MLA-absorbed mode per-head decompression matrices (GLM-4.7-flash).
    /// w_k_b: GGUF [n_embd_head_qk_nope, kv_lora_rank, n_heads] — per head,
    ///   absorbs Q-side K decompression: q_nope_absorbed[h] = w_k_b[h] @ q_nope[h]
    ///   producing kv_lora_rank dims that match the compressed K cache.
    /// w_v_b: GGUF [kv_lora_rank, n_embd_head_v_mla, n_heads] — per head,
    ///   absorbs V decompression: attn_out_real[h] = w_v_b[h] @ attn_v[h]
    ///   reducing the kv_lora_rank-dim attention output back to n_embd_head_v_mla.
    /// Both are 0 in non-absorbed mode (V2-Lite, Qwen2.5, etc.).
    w_k_b: i64, dt_k_b: i32,
    w_v_b: i64, dt_v_b: i32,
    /// FR-17-extra-mla-fwd MoE shared experts — DeepSeek-V2 / GLM-4.7-flash
    /// have `expert_shared_count > 0` always-on experts that are FUSED into
    /// a single MLP with hidden dim = n_shared * expert_ff_dim
    /// (V2-Lite: 2 * 1408 = 2816).  Stored under
    /// `blk.N.ffn_{gate,up,down}_shexp.weight`.  All 0 when absent (no
    /// shared experts).
    w_gate_shexp: i64, w_up_shexp: i64, w_down_shexp: i64,
    dt_gate_shexp: i32, dt_up_shexp: i32, dt_down_shexp: i32,
}

/// matt-voice FR-18.6-real leg 3 — public per-layer quant weights for a dense
/// Qwen3 layer, consumed by the QLoRA pipeline trainer (trainer/qwen_qlora_stage).
/// Handles are device-buffer slots (frozen base, kept QUANTIZED in VRAM); dtypes
/// are GGUF quant codes (12=Q4_K, 14=Q6_K, 18=IQ3_XXS, ...). The trainer dequants
/// each proj to a transient f32 buffer per forward (base stays quantized) and runs
/// the f32 matmul ops; the adapter is the only trainable part. Norm gains are F32.
///
/// roadmap: P18
pub struct QwenLayerWeights {
    pub attn_norm_g: i64, pub ffn_norm_g: i64,
    pub attn_q_norm_g: i64, pub attn_k_norm_g: i64, // Qwen3 per-head Q/K RMSNorm (0=absent)
    pub w_q: i64, pub w_k: i64, pub w_v: i64, pub w_o: i64,
    pub w_gate: i64, pub w_up: i64, pub w_down: i64,
    pub dt_q: i32, pub dt_k: i32, pub dt_v: i32, pub dt_o: i32,
    pub dt_gate: i32, pub dt_up: i32, pub dt_down: i32,
}

/// Open a GGUF and read its ModelConfig. Returns (gguf_handle, cfg). The handle
/// stays valid for subsequent `load_qwen_layer` calls.
///
/// roadmap: P18
pub unsafe fn open_gguf_config(path: &str) -> Result<(i64, ModelConfig), String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("GGUF not found: {}", path));
    }
    aether_dev_init();
    let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
    if h < 0 { return Err(format!("aether_gguf_open failed: {}", h)); }
    let cfg = ModelConfig::from_gguf(h);
    Ok((h, cfg))
}

/// Load one dense-Qwen3 layer's quant weights to device (frozen base). Reuses the
/// internal `load_block`; any layer index works independently, so a pipeline rank
/// can load just its range (e.g. 32..64).
///
/// roadmap: P18
pub unsafe fn load_qwen_layer(h: i64, b: usize) -> QwenLayerWeights {
    let bw = load_block(h, b);
    QwenLayerWeights {
        attn_norm_g: bw.attn_norm_g, ffn_norm_g: bw.ffn_norm_g,
        attn_q_norm_g: bw.attn_q_norm_g, attn_k_norm_g: bw.attn_k_norm_g,
        w_q: bw.w_q, w_k: bw.w_k, w_v: bw.w_v, w_o: bw.w_o,
        w_gate: bw.w_gate, w_up: bw.w_up, w_down: bw.w_down,
        dt_q: bw.dt_q, dt_k: bw.dt_k, dt_v: bw.dt_v, dt_o: bw.dt_o,
        dt_gate: bw.dt_gate, dt_up: bw.dt_up, dt_down: bw.dt_down,
    }
}

struct ActivationGpu {
    x: i64, x_norm: i64,
    q: i64, k_step: i64, v_step: i64,
    attn_out: i64, proj: i64,
    gate: i64, down: i64,
    logits: i64,
    // FR-17-extra-mla-absorbed-persist — persistent workspace buffers for
    // `mla_attention_forward_absorbed`.  Allocated once at session
    // construction (only if `cfg.is_mla_absorbed()`); reused across every
    // layer × token instead of cuda_alloc/free per call.
    // Each is 0 for non-MLA-absorbed archs.
    mla_abs_kv_a: i64,         // [kv_lora_rank + qk_rope]
    mla_abs_c_kv: i64,         // [kv_lora_rank]
    mla_abs_c_kv_n: i64,       // [kv_lora_rank]
    mla_abs_k_pe: i64,         // [qk_rope]
    mla_abs_q_a: i64,          // [q_lora_rank]   (0 when q_lora_rank == 0)
    mla_abs_q_a_n: i64,        // [q_lora_rank]   (0 when q_lora_rank == 0)
    mla_abs_q_proj: i64,       // [n_heads * key_mla]
    mla_abs_q_full: i64,       // [n_heads * (kv_lora_rank + qk_rope)]
    mla_abs_k_row: i64,        // [n_heads * (kv_lora_rank + qk_rope)]
    mla_abs_v_row: i64,        // [n_heads * kv_lora_rank]
    mla_abs_attn_v_out: i64,   // [n_heads * kv_lora_rank]
}

/// FR-19.5-extra-deep Phase 2b-2b — batched-decode activation workspace.
/// Sized `MAX_BATCH` rows; the batched forward reuses the first `b` rows
/// each tick (b = active slot count ≤ MAX_BATCH).  Allocated lazily on the
/// first `step_logits_for_batch` call so single-stream decode pays nothing.
/// Every field is row-major `[MAX_BATCH * dim]`.
struct BatchActivationGpu {
    x: i64,            // [cap * d_model]
    x_norm: i64,       // [cap * d_model]
    q: i64,            // [cap * (n_q_heads * head_dim)]
    k_step: i64,       // [cap * d_kv]
    v_step: i64,       // [cap * d_kv]
    attn_out: i64,     // [cap * d_model]
    proj: i64,         // [cap * d_model]
    gate: i64,         // [cap * d_ff]
    up: i64,           // [cap * d_ff]   (separate buffer for the SwiGLU up branch)
    down: i64,         // [cap * d_model]
    logits: i64,       // [cap * vocab]
    // Per-request metadata, device-side i32 (length cap, or cap*n_logical).
    pos_batch: i64,        // i32 [cap]            decode position per request
    cur_seq_batch: i64,    // i32 [cap]            attention window length per request
    page_table_batch: i64, // i32 [cap * n_logical] each request's logical→physical map
    n_logical: i32,        // page_table stride
    // Scratch row buffers for the non-Q4_K offset-copy matmul fallback.
    scratch_in: i64,   // [max(d_model, d_ff)]
    scratch_out: i64,  // [vocab]
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
        0  => { (n_elems, n_elems * 4) }                     // F32 (4 bytes/elem)
        12 => { let nb = n_elems / 256; (nb, nb * 144) }     // Q4_K
        14 => { let nb = n_elems / 256; (nb, nb * 210) }     // Q6_K
        1  => { (n_elems, n_elems * 2) }                     // F16 (2 bytes/elem)
        2  => { let nb = n_elems / 32; (nb, nb * 18) }       // Q4_0 (FR-17-extra-q4_0-fwd)
        6  => { let nb = n_elems / 32; (nb, nb * 22) }       // Q5_0 (FR-17-extra-q5_0-fwd)
        8  => { let nb = n_elems / 32; (nb, nb * 34) }       // Q8_0 (FR-17-extra-q8_0-fwd)
        13 => { let nb = n_elems / 256; (nb, nb * 176) }     // Q5_K (FR-17-extra-q5_k-fwd)
        11 => { let nb = n_elems / 256; (nb, nb * 110) }     // Q3_K (matmul+dispatch already present; unblocks qwen3moe Q3_K_M)
        18 => { let nb = n_elems / 256; (nb, nb * 98) }      // IQ3_XXS (FR-17-extra-iq3_xxs-fwd)
        20 => { let nb = n_elems / 32; (nb, nb * 18) }       // IQ4_NL (FR-17-extra-iq4_nl-fwd)
        21 => { let nb = n_elems / 256; (nb, nb * 110) }     // IQ3_S (FR-17-extra-iq3_s-fwd)
        23 => { let nb = n_elems / 256; (nb, nb * 136) }     // IQ4_XS (FR-17-extra-iq4_xs-fwd)
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
///
/// Dtype-aware as of the GLM-4.7-flash debug (router NaN-cascade bisect).  If
/// the GGUF stores the tensor as F16 (dt=1), dequantises to F32 host-side
/// before the H2D copy.  Without this, the prior body read 4 bytes/elem from
/// 2-byte F16 storage → garbage F32 values → NaN router_logits in the very
/// first MoE block.
unsafe fn upload_f32_tensor_opt(h: i64, name: &str) -> i64 {
    let needle = name.as_bytes();
    let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
    if idx < 0 { return 0; }
    let dt = aether_gguf_get_tensor_dtype(h, idx);
    let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
    let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
    let host: Vec<f32> = match dt {
        0 => {
            // F32 — direct copy.
            std::slice::from_raw_parts(dptr as *const f32, n_elems).to_vec()
        }
        1 => {
            // F16 — dequantise per element.
            let raw = std::slice::from_raw_parts(dptr as *const u16, n_elems);
            raw.iter().map(|&h16| aether_f16_to_f32(h16 as i32)).collect()
        }
        _ => panic!("upload_f32_tensor_opt: tensor '{}' has unsupported dtype {} \
                    (only F32=0 and F16=1 dequant-on-load today)", name, dt),
    };
    if std::env::var("AETHER_DUMP_F32_LOADS").is_ok() {
        eprintln!("[upload_f32_tensor_opt] {} dt={} n_elems={}", name, dt, n_elems);
    }
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
        0  => { (n_elems, n_elems * 4) }
        12 => { let nb = n_elems / 256; (nb, nb * 144) }
        14 => { let nb = n_elems / 256; (nb, nb * 210) }
        1  => { (n_elems, n_elems * 2) }
        2  => { let nb = n_elems / 32; (nb, nb * 18) }
        6  => { let nb = n_elems / 32; (nb, nb * 22) }
        8  => { let nb = n_elems / 32; (nb, nb * 34) }
        13 => { let nb = n_elems / 256; (nb, nb * 176) }
        11 => { let nb = n_elems / 256; (nb, nb * 110) }     // Q3_K (unblocks qwen3moe Q3_K_M)
        18 => { let nb = n_elems / 256; (nb, nb * 98) }
        20 => { let nb = n_elems / 32; (nb, nb * 18) }
        21 => { let nb = n_elems / 256; (nb, nb * 110) }
        23 => { let nb = n_elems / 256; (nb, nb * 136) }
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
    // For MLA + q_lora_rank > 0 (GLM-4.7-flash, DeepSeek-V2 large) attn_q
    // doesn't exist either — Q comes from attn_q_a @ attn_q_a_norm @ attn_q_b.
    // Use the opt loader so MLA-Q-LoRA archs don't panic; non-MLA archs and
    // V2-Lite (MLA + q_lora_rank == 0) still produce a real w_q.  Forward
    // pass at serving.rs:978-992 picks the right branch based on
    // cfg.q_lora_rank + bw.w_q_a/w_q_b presence.
    let (w_q, nb_qo, dt_q) = if is_mla && w_q_a != 0 && w_q_b != 0 {
        upload_tensor_u8_opt(h, &format!("{}attn_q.weight", p))
    } else {
        upload_tensor_u8(h, &format!("{}attn_q.weight", p))
    };
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
    // MLA-absorbed variant (GLM-4.7-flash): per-head K/V decompression matrices.
    // Stored as 3D tensors [n_embd_head_qk_nope, kv_lora_rank, n_heads] and
    // [kv_lora_rank, n_embd_head_v_mla, n_heads] respectively.  0 for archs
    // that use the combined attn_kv_b (V2-Lite, etc.).
    let (w_k_b, _, dt_k_b) = if is_mla {
        upload_tensor_u8_opt(h, &format!("{}attn_k_b.weight", p))
    } else { (0, 0, 0) };
    let (w_v_b, _, dt_v_b) = if is_mla {
        upload_tensor_u8_opt(h, &format!("{}attn_v_b.weight", p))
    } else { (0, 0, 0) };
    if std::env::var("AETHER_DUMP_ATTN_DTYPES").is_ok() && b < 4 {
        eprintln!("[ATTN-DT b={}] kv_a_mqa={} kv_b={} q_a={} q_b={} q={} k={} v={} o={} k_b={} v_b={}",
            b, dt_kv_a_mqa, dt_kv_b, dt_q_a, dt_q_b, dt_q, dt_k, dt_v, dt_o, dt_k_b, dt_v_b);
    }
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
    // Shared experts (FR-17-extra-mla-fwd MoE).  Present on deepseek2 /
    // glm-4.7-flash MoE blocks; absent on Qwen3-MoE and on the leading
    // dense block.  Stored as a single FUSED MLP with hidden dim
    // n_shared * expert_ff_dim (already concatenated in the GGUF).
    let (w_gate_shexp, _, dt_gate_shexp) = upload_tensor_u8_opt(h, &format!("{}ffn_gate_shexp.weight", p));
    let (w_up_shexp,   _, dt_up_shexp)   = upload_tensor_u8_opt(h, &format!("{}ffn_up_shexp.weight", p));
    let (w_down_shexp, _, dt_down_shexp) = upload_tensor_u8_opt(h, &format!("{}ffn_down_shexp.weight", p));
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
        w_k_b, dt_k_b, w_v_b, dt_v_b,
        w_gate_shexp, w_up_shexp, w_down_shexp,
        dt_gate_shexp, dt_up_shexp, dt_down_shexp,
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
    // FR-17-extra-mla-fwd — pre-attention dispatch.  MLA path runs the
    // compressed-KV → decompression → partial-RoPE → MLA-attention chain;
    // non-MLA path runs the standard Q/K/V → RoPE → attention chain.  Both
    // paths write to act.attn_out so the common O-proj + residual + FFN
    // tail below works for either.
    let is_mla = cfg.kv_lora_rank > 0 || bw.w_kv_a_mqa != 0;
    let attn_out_n_in: c_int = if is_mla {
        mla_attention_forward(bw, act, kv, step_args, paged_cfg, cfg, max_seq);
        // Absorbed MLA reduces per-head attn_out to value_length_mla via wv_b;
        // non-absorbed leaves it at v_head_dim.
        if cfg.is_mla_absorbed() {
            (cfg.n_q_heads * cfg.value_length_mla as usize) as c_int
        } else {
            (cfg.n_q_heads * cfg.v_head_dim as usize) as c_int
        }
    } else {
        standard_attention_forward(bw, act, kv, step_args, paged_cfg, cfg, max_seq);
        // o-proj input = n_q_heads * head_dim (= d_model for Qwen/Llama; smaller
        // for Mistral Small 24B with explicit head_dim).
        n_q_heads * head_dim
    };

    // ---- Common post-attention tail: O proj + residual + LN + FFN ----
    dispatch_matmul(act.attn_out, bw.w_o, bw.dt_o, act.proj, d_model, attn_out_n_in);
    if bw.post_attn_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(act.proj, bw.post_attn_norm_g, act.proj,
            norm_eps, 1, d_model);
    }
    aether_op_add_inplace_f32_cuda(act.x, act.proj, d_model);
    // glm-debug: dump x AFTER attention residual, BEFORE second RMSnorm + FFN.
    // Pinpoints whether NaN enters via attn vs FFN at layer 1 (first MoE).
    // Gated on AETHER_DUMP_INTRA_BLOCK so it's free for prod paths.
    if std::env::var("AETHER_DUMP_INTRA_BLOCK").is_ok() {
        aether_dev_sync();
        let mut h = vec![0.0f32; cfg.d_model];
        aether_dev_d2h_f32(act.x, h.as_mut_ptr() as i64, d_model);
        let n_nan = h.iter().filter(|x| x.is_nan()).count();
        let n_inf = h.iter().filter(|x| x.is_infinite()).count();
        let finite: Vec<f32> = h.iter().cloned().filter(|x| x.is_finite()).collect();
        let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = if !finite.is_empty() { finite.iter().sum::<f32>() / finite.len() as f32 } else { 0.0 };
        eprintln!("[POST-ATTN x] nan={} inf={} min={:.4e} max={:.4e} mean={:.4e}",
            n_nan, n_inf, mn, mx, mean);
    }
    aether_op_rms_norm_f32_cuda(act.x, bw.ffn_norm_g, act.x_norm, norm_eps, 1, d_model);
    if bw.w_router != 0 {
        moe_ffn_forward(bw, act, cfg);
    } else {
        if bw.dt_gate == 12 && bw.dt_up == 12 {
            aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
                act.x_norm, bw.w_gate, bw.w_up, act.gate,
                d_ff, (bw.nb_gate_up / cfg.d_ff) as c_int);
        } else {
            // Non-fused SwiGLU fallback for non-Q4_K gate/up dtypes
            // (e.g. GLM-4.7-flash layer 0 stores both as IQ4_XS).  Three
            // separate kernel launches instead of the fused one — same
            // arithmetic result, slightly more overhead.
            let up_tmp = aether_dev_alloc_f32(d_ff);
            dispatch_matmul(act.x_norm, bw.w_gate, bw.dt_gate, act.gate, d_ff, d_model);
            dispatch_matmul(act.x_norm, bw.w_up,   bw.dt_up,   up_tmp,   d_ff, d_model);
            aether_op_silu_f32_cuda(act.gate, d_ff);
            aether_op_mul_inplace_f32_cuda(act.gate, up_tmp, d_ff);
            let _ = aether_dev_free_f32(up_tmp);
        }
        dispatch_matmul(act.gate, bw.w_down, bw.dt_down, act.down, d_model, d_ff);
        if bw.post_ffn_norm_g != 0 {
            aether_op_rms_norm_f32_cuda(act.down, bw.post_ffn_norm_g, act.down,
                norm_eps, 1, d_model);
        }
        aether_op_add_inplace_f32_cuda(act.x, act.down, d_model);
    }
}

/// Standard Qwen/Llama/Gemma3 attention path: Q/K/V matmul → optional bias →
/// optional Q/K RMSnorm (Qwen3) → RoPE → paged or contiguous attention,
/// writing the per-head attention output to `act.attn_out`.
unsafe fn standard_attention_forward(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, step_args: i64,
    paged_cfg: Option<(i64, i32)>,
    cfg: &ModelConfig,
    max_seq: usize,
) {
    let d_model = cfg.d_model as c_int;
    let d_kv = cfg.d_kv as c_int;
    let n_q_heads = cfg.n_q_heads as c_int;
    let n_kv_heads = cfg.n_kv_heads as c_int;
    let head_dim = cfg.head_dim as c_int;
    let rope_base = cfg.rope_base;
    let norm_eps = cfg.norm_eps;

    // Q projection output = n_q_heads * head_dim (= d_model for Qwen/Llama,
    // but 4096 != 5120 for Mistral Small 24B where head_dim is explicit).
    let q_dim = n_q_heads * head_dim;
    dispatch_matmul(act.x_norm, bw.w_q, bw.dt_q, act.q, q_dim, d_model);
    if bw.b_q != 0 {
        aether_op_bias_add_f32_cuda(act.q, bw.b_q, 1, q_dim);
    }
    dispatch_matmul(act.x_norm, bw.w_k, bw.dt_k, act.k_step, d_kv, d_model);
    if bw.b_k != 0 {
        aether_op_bias_add_f32_cuda(act.k_step, bw.b_k, 1, d_kv);
    }
    dispatch_matmul(act.x_norm, bw.w_v, bw.dt_v, act.v_step, d_kv, d_model);
    if bw.b_v != 0 {
        aether_op_bias_add_f32_cuda(act.v_step, bw.b_v, 1, d_kv);
    }
    if bw.attn_q_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(act.q, bw.attn_q_norm_g, act.q,
            norm_eps, n_q_heads, head_dim);
    }
    if bw.attn_k_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(act.k_step, bw.attn_k_norm_g, act.k_step,
            norm_eps, n_kv_heads, head_dim);
    }
    aether_op_rope_apply_devarg_f32_cuda(act.q,
        1, n_q_heads, head_dim, rope_base, step_args);
    aether_op_rope_apply_devarg_f32_cuda(act.k_step,
        1, n_kv_heads, head_dim, rope_base, step_args);
    let scale: f32 = 1.0 / (cfg.head_dim as f32).sqrt();
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
}

/// FR-19.5-extra-deep Phase 2b-2b — batched matmul over `b` rows.
///
/// `x` is `[b * n_in]`, `y` is `[b * n_out]`.  For Q4_K weights (dt==12) the
/// weight-reuse seqB kernel dequants each super-block once and applies it to
/// all `b` rows (the 1.9× win).  For every other dtype the handle API can't
/// sub-slice the weight, so we stage each row through `scratch_in`/`scratch_out`
/// (offset 0) and run the existing seq1 dispatch — correct for ALL dtypes,
/// no weight-reuse for those specific tensors (e.g. Qwen2.5-7B Q6_K v/down).
unsafe fn matmul_batched(
    x: i64, w: i64, dt: i32, y: i64,
    n_out: c_int, n_in: c_int, b: c_int,
    scratch_in: i64, scratch_out: i64,
) {
    if dt == 12 {
        let rc = aether_op_fused_q4k_matmul_seqB_v3_cuda(x, w, y, n_out, n_in / 256, b);
        assert_eq!(rc, 0, "fused_q4k_matmul_seqB_v3 failed (rc={}, b={})", rc, b);
    } else {
        for row in 0..b {
            aether_dev_d2d_f32_offset(x, row * n_in, scratch_in, 0, n_in);
            dispatch_matmul(scratch_in, w, dt, scratch_out, n_out, n_in);
            aether_dev_d2d_f32_offset(scratch_out, 0, y, row * n_out, n_out);
        }
    }
}

/// FR-19.5-extra-deep Phase 2b-2b — one transformer layer over `b` fused
/// requests at heterogeneous decode positions.  Mirrors
/// `standard_attention_forward` + the dense-FFN tail of `block_forward_devarg`,
/// but every op runs over `b` rows: norms/biases take `rows=b`, elementwise
/// ops take `n=b*dim`, matmuls go through `matmul_batched`, and RoPE / append /
/// attention use the per-request hetero kernels driven by `pos_batch` /
/// `cur_seq_batch` / `page_table_batch`.
///
/// Caller contract: this path covers the STANDARD (non-MLA) attention + DENSE
/// (non-MoE) FFN shape with `head_dim % 32 == 0` and `sliding_window == 0`
/// (`QwenSession::is_batchable`).  MLA / MoE / flex arches stay on the serial
/// per-slot path.
unsafe fn block_forward_batched(
    bw: &BlockGpu, ba: &BatchActivationGpu, kv: &KvCacheGpu,
    b: c_int, cfg: &ModelConfig, max_seq: usize, block_size: c_int,
) {
    let d_model = cfg.d_model as c_int;
    let d_kv = cfg.d_kv as c_int;
    let d_ff = cfg.d_ff as c_int;
    let n_q_heads = cfg.n_q_heads as c_int;
    let n_kv_heads = cfg.n_kv_heads as c_int;
    let head_dim = cfg.head_dim as c_int;
    let rope_base = cfg.rope_base;
    let eps = cfg.norm_eps;
    let stride = ba.n_logical;
    let (si, so) = (ba.scratch_in, ba.scratch_out);

    aether_op_rms_norm_f32_cuda(ba.x, bw.attn_norm_g, ba.x_norm, eps, b, d_model);

    matmul_batched(ba.x_norm, bw.w_q, bw.dt_q, ba.q, d_model, d_model, b, si, so);
    if bw.b_q != 0 { aether_op_bias_add_f32_cuda(ba.q, bw.b_q, b, d_model); }
    matmul_batched(ba.x_norm, bw.w_k, bw.dt_k, ba.k_step, d_kv, d_model, b, si, so);
    if bw.b_k != 0 { aether_op_bias_add_f32_cuda(ba.k_step, bw.b_k, b, d_kv); }
    matmul_batched(ba.x_norm, bw.w_v, bw.dt_v, ba.v_step, d_kv, d_model, b, si, so);
    if bw.b_v != 0 { aether_op_bias_add_f32_cuda(ba.v_step, bw.b_v, b, d_kv); }

    // Qwen3 per-head Q/K RMSNorm — each (request, head) row of head_dim is
    // normalized independently, so rows = b * n_{q,kv}_heads.
    if bw.attn_q_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(ba.q, bw.attn_q_norm_g, ba.q, eps, b * n_q_heads, head_dim);
    }
    if bw.attn_k_norm_g != 0 {
        aether_op_rms_norm_f32_cuda(ba.k_step, bw.attn_k_norm_g, ba.k_step, eps, b * n_kv_heads, head_dim);
    }

    aether_op_batched_rope_apply_devarg_f32_cuda(ba.q, b, n_q_heads, head_dim, rope_base, ba.pos_batch);
    aether_op_batched_rope_apply_devarg_f32_cuda(ba.k_step, b, n_kv_heads, head_dim, rope_base, ba.pos_batch);

    let scale: f32 = 1.0 / (cfg.head_dim as f32).sqrt();
    aether_op_batched_paged_append_kv_hetero_devarg_f32_cuda(
        ba.k_step, ba.v_step, kv.k_cache, kv.v_cache, ba.page_table_batch,
        b, d_kv, block_size, stride, ba.pos_batch);
    aether_op_batched_paged_attention_hetero_devarg_f32_cuda(
        ba.q, kv.k_cache, kv.v_cache, ba.page_table_batch, ba.attn_out,
        b, n_q_heads, n_kv_heads, head_dim, block_size, stride,
        scale, max_seq as c_int, ba.cur_seq_batch);

    matmul_batched(ba.attn_out, bw.w_o, bw.dt_o, ba.proj, d_model, d_model, b, si, so);
    aether_op_add_inplace_f32_cuda(ba.x, ba.proj, b * d_model);

    aether_op_rms_norm_f32_cuda(ba.x, bw.ffn_norm_g, ba.x_norm, eps, b, d_model);
    matmul_batched(ba.x_norm, bw.w_gate, bw.dt_gate, ba.gate, d_ff, d_model, b, si, so);
    matmul_batched(ba.x_norm, bw.w_up,   bw.dt_up,   ba.up,   d_ff, d_model, b, si, so);
    aether_op_silu_f32_cuda(ba.gate, b * d_ff);
    aether_op_mul_inplace_f32_cuda(ba.gate, ba.up, b * d_ff);
    matmul_batched(ba.gate, bw.w_down, bw.dt_down, ba.down, d_model, d_ff, b, si, so);
    aether_op_add_inplace_f32_cuda(ba.x, ba.down, b * d_model);
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
/// FR-17-extra-mla-fwd — DeepSeek-V2 / GLM-4.7-flash MLA forward.  Runs the
/// PRE-attention plumbing (compressed-KV projection + per-head decompression
/// + partial-dim RoPE composition + Q projection [+ optional Q LoRA]) then
/// dispatches the paged MLA attention kernel.  After the attention, control
/// returns to `block_forward_devarg`'s common O-proj / residual / LN / FFN
/// tail by writing `act.attn_out` and the rest matches the non-MLA path.
///
/// Caller contract:
///   - `kv.k_cache` must be sized `n_heads * qk_head_dim * pool_tokens` f32
///   - `kv.v_cache` must be sized `n_heads * v_head_dim * pool_tokens` f32
/// The existing Qwen-style QwenSession::new_with_mode sizes these for
/// `n_kv_heads * head_dim` which UNDER-ALLOCATES K (qk_head_dim > head_dim
/// for MLA's qk_nope + qk_rope vs head_dim).  A real DeepSeek-V2 / GLM-4.7
/// serving path needs a separate constructor that uses cfg.qk_head_dim /
/// cfg.v_head_dim for these allocations.  Until that constructor exists,
/// this path will write out-of-bounds.  Component kernels are all CPU↔GPU
/// witnessed in tests/cuda_mla_e2e_synthetic.rs (multi-step end-to-end).
unsafe fn mla_attention_forward(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, step_args: i64,
    paged_cfg: Option<(i64, i32)>,
    cfg: &ModelConfig,
    max_seq: usize,
) {
    if cfg.is_mla_absorbed() {
        mla_attention_forward_absorbed(bw, act, kv, step_args, paged_cfg, cfg, max_seq);
        return;
    }
    let d_model = cfg.d_model as c_int;
    let kv_lora_rank = cfg.kv_lora_rank as c_int;
    let qk_head_dim = cfg.qk_head_dim as c_int;
    let qk_rope_head_dim = cfg.qk_rope_head_dim as c_int;
    let qk_nope_head_dim = qk_head_dim - qk_rope_head_dim;
    let v_head_dim = cfg.v_head_dim as c_int;
    let n_heads = cfg.n_q_heads as c_int;
    let rope_base = cfg.rope_base;
    let norm_eps = cfg.norm_eps;

    // Per-call workspace allocs.  Perf is future work; correctness first.
    let kv_a       = aether_dev_alloc_f32(kv_lora_rank + qk_rope_head_dim);
    let c_kv       = aether_dev_alloc_f32(kv_lora_rank);
    let c_kv_n     = aether_dev_alloc_f32(kv_lora_rank);
    let k_rope     = aether_dev_alloc_f32(qk_rope_head_dim);
    let kv_b       = aether_dev_alloc_f32(n_heads * (qk_nope_head_dim + v_head_dim));
    let k_row      = aether_dev_alloc_f32(n_heads * qk_head_dim);
    let v_row      = aether_dev_alloc_f32(n_heads * v_head_dim);
    let q_full     = aether_dev_alloc_f32(n_heads * qk_head_dim);

    let dump_mla = std::env::var("AETHER_DUMP_MLA").is_ok();
    let probe = |label: &str, h: i64, n: usize| {
        if !dump_mla { return; }
        aether_dev_sync();
        let mut v = vec![0.0f32; n];
        aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, n as c_int);
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let fin: Vec<f32> = v.iter().cloned().filter(|x| x.is_finite()).collect();
        let mn = fin.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = fin.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[MLA {}] n={} nan={} inf={} min={:.4e} max={:.4e}",
            label, n, nan, inf, mn, mx);
    };
    probe("x_norm_IN", act.x_norm, d_model as usize);
    // 1. kv_a_mqa: x_norm @ W → [kv_lora_rank + qk_rope_head_dim]
    dispatch_matmul(act.x_norm, bw.w_kv_a_mqa, bw.dt_kv_a_mqa,
        kv_a, kv_lora_rank + qk_rope_head_dim, d_model);
    probe("kv_a", kv_a, (kv_lora_rank + qk_rope_head_dim) as usize);

    // 2. Split kv_a into c_kv + k_rope_shared.
    aether_op_mla_split_kv_a_f32_cuda(kv_a, c_kv, k_rope,
        kv_lora_rank, qk_rope_head_dim);

    // 3. RMSNorm on c_kv with kv_a_norm gain.
    aether_op_rms_norm_f32_cuda(c_kv, bw.attn_kv_a_norm_g, c_kv_n,
        norm_eps, 1, kv_lora_rank);

    probe("c_kv_n", c_kv_n, kv_lora_rank as usize);
    // 4. kv_b: c_kv_normed @ W → [n_heads * (qk_nope + v_head)]
    dispatch_matmul(c_kv_n, bw.w_kv_b, bw.dt_kv_b,
        kv_b, n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank);
    probe("kv_b", kv_b, (n_heads * (qk_nope_head_dim + v_head_dim)) as usize);

    // 5. Extract V from kv_b → v_row.
    aether_op_mla_extract_v_f32_cuda(kv_b, v_row,
        n_heads, qk_nope_head_dim, v_head_dim);

    // 6. Partial RoPE on shared k_rope.  Use YaRN-aware kernel when the
    // model's RoPE scaling type is YaRN (deepseek2 / glm-4.7-flash).
    let yarn_active = cfg.yarn_factor > 1.0;
    if yarn_active {
        aether_op_mla_rope_k_shared_yarn_f32_cuda(k_rope, qk_rope_head_dim,
            rope_base, cfg.yarn_factor, cfg.yarn_orig_ctx,
            cfg.yarn_beta_fast, cfg.yarn_beta_slow, step_args);
    } else {
        aether_op_mla_rope_k_shared_f32_cuda(k_rope, qk_rope_head_dim,
            rope_base, step_args);
    }

    // 7. Assemble per-head K row = [K_nope | k_rope_shared] per head.
    aether_op_mla_assemble_k_f32_cuda(kv_b, k_rope, k_row,
        n_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim);

    // 8. Q projection.  When q_lora_rank == 0 (V2-Lite) → direct attn_q.
    //    When q_lora_rank > 0 (larger V2 vars / GLM-4.7-flash) → low-rank
    //    attn_q_a → RMSNorm → attn_q_b.
    if cfg.q_lora_rank > 0 && bw.w_q_a != 0 && bw.w_q_b != 0 {
        let q_a_dim = cfg.q_lora_rank as c_int;
        let q_a = aether_dev_alloc_f32(q_a_dim);
        let q_a_n = aether_dev_alloc_f32(q_a_dim);
        dispatch_matmul(act.x_norm, bw.w_q_a, bw.dt_q_a, q_a, q_a_dim, d_model);
        probe("q_a", q_a, q_a_dim as usize);
        aether_op_rms_norm_f32_cuda(q_a, bw.attn_q_a_norm_g, q_a_n,
            norm_eps, 1, q_a_dim);
        probe("q_a_n", q_a_n, q_a_dim as usize);
        dispatch_matmul(q_a_n, bw.w_q_b, bw.dt_q_b,
            q_full, n_heads * qk_head_dim, q_a_dim);
        probe("q_full", q_full, (n_heads * qk_head_dim) as usize);
        let _ = aether_dev_free_f32(q_a);
        let _ = aether_dev_free_f32(q_a_n);
    } else {
        dispatch_matmul(act.x_norm, bw.w_q, bw.dt_q,
            q_full, n_heads * qk_head_dim, d_model);
    }

    // 9. Partial RoPE on Q's rope sub-region (YaRN-aware).
    if yarn_active {
        aether_op_mla_rope_q_partial_yarn_f32_cuda(q_full,
            n_heads, qk_head_dim, qk_nope_head_dim, rope_base,
            cfg.yarn_factor, cfg.yarn_orig_ctx,
            cfg.yarn_beta_fast, cfg.yarn_beta_slow, step_args);
    } else {
        aether_op_mla_rope_q_partial_f32_cuda(q_full,
            n_heads, qk_head_dim, qk_nope_head_dim, rope_base, step_args);
    }

    // 10. Paged append + paged MLA attention.  Caller's KV cache MUST be
    // sized to MLA dims — see top-of-fn contract.
    //
    // YaRN attention temperature: mscale = 1 + log_multiplier * ln(s),
    // applied to BOTH Q and K → mscale^2 multiplies the dot product.
    // Equivalently, fold mscale^2 into the standard 1/sqrt(qk_head_dim) scale.
    let d_k_row = n_heads * qk_head_dim;
    let d_v_row = n_heads * v_head_dim;
    let base_scale = 1.0f32 / (qk_head_dim as f32).sqrt();
    let scale = if yarn_active {
        let mscale = 1.0 + cfg.yarn_log_multiplier * cfg.yarn_factor.ln();
        base_scale * mscale * mscale
    } else { base_scale };
    let (page_table_dev, block_size) = paged_cfg.expect(
        "FR-17-extra-mla-fwd: MLA path requires --paged mode today \
         (contiguous-KV variant is follow-on).");
    aether_op_paged_append_kv_mla_devarg_f32_cuda(
        k_row, v_row, kv.k_cache, kv.v_cache, page_table_dev,
        d_k_row, d_v_row, block_size, step_args);
    aether_op_paged_attention_mla_devarg_f32_cuda(
        q_full, kv.k_cache, kv.v_cache, page_table_dev, act.attn_out,
        n_heads, qk_head_dim, v_head_dim, block_size,
        scale, max_seq as c_int, step_args);
    probe("attn_out", act.attn_out, (n_heads * v_head_dim) as usize);
    let _ = matmul_nt_f32_cuda_unused();  // silence dead-import warning

    // Free per-call workspace.  (Drop on QwenSession owns persistent bufs.)
    for h in [kv_a, c_kv, c_kv_n, k_rope, kv_b, k_row, v_row, q_full] {
        let _ = aether_dev_free_f32(h);
    }
}

/// FR-17-extra-mla-absorbed — GLM-4.7-flash absorbed-MLA forward.
///
/// Differs from the non-absorbed path (V2-Lite) in three places:
/// 1. w_q_b outputs per-head `key_length_mla` (e.g. 256 for GLM) — split into
///    q_nope (192) + q_pe (64).
/// 2. Per-head wk_b absorbs Q-side K decompression: q_nope_absorbed[h] =
///    wk_b[h] @ q_nope[h] producing kv_lora_rank (512) per head.  Combined with
///    q_pe → Qcur per head (kv_lora_rank + qk_rope = 576).
/// 3. K and V in cache are the SHARED compressed c_kv (+ k_pe for K) broadcast
///    across all heads (MQA semantics).  After attention, wv_b reduces the
///    kv_lora_rank-dim attn_v back to per-head value_length_mla (256).
///
/// The existing paged_attention_mla / paged_append_kv_mla kernels are reused
/// after the broadcast — they're shape-agnostic, just need consistent dims.
unsafe fn mla_attention_forward_absorbed(
    bw: &BlockGpu, act: &ActivationGpu, kv: &KvCacheGpu, step_args: i64,
    paged_cfg: Option<(i64, i32)>,
    cfg: &ModelConfig,
    max_seq: usize,
) {
    let d_model = cfg.d_model as c_int;
    let kv_lora_rank = cfg.kv_lora_rank as c_int;
    let qk_rope = cfg.qk_rope_head_dim as c_int;
    let key_mla = cfg.key_length_mla as c_int;
    let val_mla = cfg.value_length_mla as c_int;
    let q_nope_per_head = key_mla - qk_rope;
    let n_heads = cfg.n_q_heads as c_int;
    let rope_base = cfg.rope_base;
    let norm_eps = cfg.norm_eps;
    assert!(bw.w_q_a != 0 && bw.w_q_b != 0,
        "absorbed MLA requires Q-LoRA (q_lora_rank > 0)");
    assert!(bw.w_k_b != 0 && bw.w_v_b != 0,
        "absorbed MLA requires attn_k_b + attn_v_b tensors");

    // FR-17-extra-mla-absorbed-dtypes — per-dtype helper closures for the
    // per-head Q absorption + V reduction kernels.  Dispatch on bw.dt_k_b /
    // bw.dt_v_b independently (in theory an arch could quantize each side
    // differently, though in practice both sides match).
    //
    // Block-quant types (Q4_K/Q5_K/Q6_K) require their natural alignment:
    //     Q4_K/Q5_K/Q6_K: n_in % 256 == 0
    //     Q8_0 / IQ4_NL : n_in % 32  == 0
    //     F16: no alignment constraint (per-element)
    //
    // We check alignment per-side and only dispatch into the kernel matching
    // the actual dtype.  Anything not in the table panics with a clear
    // dtype + side identifier so a future arch surfaces the gap explicitly.

    // FR-17-extra-mla-absorbed-persist — reuse the ActivationGpu persistent
    // buffers instead of cuda_alloc_f32 / cuda_free_f32 per call.  This drops
    // the per-token alloc count from ~517 (47 layers × 11 buffers) to 0.
    assert!(act.mla_abs_kv_a != 0,
        "ActivationGpu MLA-absorbed workspace was not allocated; \
         was the session built with is_mla_absorbed() == true?");
    let kv_a = act.mla_abs_kv_a;
    let c_kv = act.mla_abs_c_kv;
    let c_kv_n = act.mla_abs_c_kv_n;
    let k_pe = act.mla_abs_k_pe;
    let q_a_dim = cfg.q_lora_rank as c_int;
    let q_a = act.mla_abs_q_a;
    let q_a_n = act.mla_abs_q_a_n;
    let q_proj = act.mla_abs_q_proj;
    let q_full = act.mla_abs_q_full;
    let k_row = act.mla_abs_k_row;
    let v_row = act.mla_abs_v_row;
    let attn_v_out = act.mla_abs_attn_v_out;

    let dump_mla = std::env::var("AETHER_DUMP_MLA").is_ok();
    let probe = |label: &str, h: i64, n: usize| {
        if !dump_mla { return; }
        aether_dev_sync();
        let mut v = vec![0.0f32; n];
        aether_dev_d2h_f32(h, v.as_mut_ptr() as i64, n as c_int);
        let nan = v.iter().filter(|x| x.is_nan()).count();
        let inf = v.iter().filter(|x| x.is_infinite()).count();
        let fin: Vec<f32> = v.iter().cloned().filter(|x| x.is_finite()).collect();
        let mn = fin.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = fin.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("[MLA-A {}] n={} nan={} inf={} min={:.4e} max={:.4e}",
            label, n, nan, inf, mn, mx);
    };

    probe("x_norm_IN", act.x_norm, d_model as usize);

    // Step 1: KV-A projection (shared latent + k_pe).  Same as non-absorbed.
    dispatch_matmul(act.x_norm, bw.w_kv_a_mqa, bw.dt_kv_a_mqa,
        kv_a, kv_lora_rank + qk_rope, d_model);
    aether_op_mla_split_kv_a_f32_cuda(kv_a, c_kv, k_pe, kv_lora_rank, qk_rope);
    aether_op_rms_norm_f32_cuda(c_kv, bw.attn_kv_a_norm_g, c_kv_n,
        norm_eps, 1, kv_lora_rank);
    probe("c_kv_n", c_kv_n, kv_lora_rank as usize);

    let yarn_active = cfg.yarn_factor > 1.0;
    if yarn_active {
        aether_op_mla_rope_k_shared_yarn_f32_cuda(k_pe, qk_rope,
            rope_base, cfg.yarn_factor, cfg.yarn_orig_ctx,
            cfg.yarn_beta_fast, cfg.yarn_beta_slow, step_args);
    } else {
        aether_op_mla_rope_k_shared_f32_cuda(k_pe, qk_rope,
            rope_base, step_args);
    }

    // Step 2: Q-LoRA chain.  Output q_proj has per-head key_mla dims.
    dispatch_matmul(act.x_norm, bw.w_q_a, bw.dt_q_a, q_a, q_a_dim, d_model);
    aether_op_rms_norm_f32_cuda(q_a, bw.attn_q_a_norm_g, q_a_n, norm_eps, 1, q_a_dim);
    dispatch_matmul(q_a_n, bw.w_q_b, bw.dt_q_b, q_proj, n_heads * key_mla, q_a_dim);
    probe("q_proj", q_proj, (n_heads * key_mla) as usize);

    // Step 3: Apply RoPE to q_proj's per-head rope sub-region (positions
    // q_nope_per_head .. key_mla).  Existing kernel takes (n_heads, qk_head_dim,
    // qk_nope_head_dim) so pass (n_heads, key_mla, q_nope_per_head).
    if yarn_active {
        aether_op_mla_rope_q_partial_yarn_f32_cuda(q_proj,
            n_heads, key_mla, q_nope_per_head, rope_base,
            cfg.yarn_factor, cfg.yarn_orig_ctx,
            cfg.yarn_beta_fast, cfg.yarn_beta_slow, step_args);
    } else {
        aether_op_mla_rope_q_partial_f32_cuda(q_proj,
            n_heads, key_mla, q_nope_per_head, rope_base, step_args);
    }

    // Step 4: Per-head Q absorption + q_pe concat via wk_b.  Output q_full
    // has per-head dim (kv_lora_rank + qk_rope) — matches Kcur per head in
    // the broadcast cache.  Dispatch on bw.dt_k_b for the dequant body;
    // alignment of q_nope_per_head determines which block-size is valid.
    mla_absorb_q_dispatch(
        bw.dt_k_b, q_proj, bw.w_k_b, q_full,
        n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head);
    probe("q_full_absorbed", q_full, (n_heads * (kv_lora_rank + qk_rope)) as usize);

    // Step 5: Broadcast c_kv (+ k_pe for K) to per-head MQA slots.
    aether_op_mla_broadcast_kv_for_mqa_f32_cuda(
        c_kv_n, k_pe, k_row, v_row, n_heads, kv_lora_rank, qk_rope);
    probe("k_row", k_row, (n_heads * (kv_lora_rank + qk_rope)) as usize);
    probe("v_row", v_row, (n_heads * kv_lora_rank) as usize);

    // Step 6: Paged append + attention.  Per-head K = kv_lora_rank + qk_rope,
    // per-head V = kv_lora_rank.  These are the SAME numbers as the
    // non-absorbed path's qk_head_dim and v_head_dim for GLM (576 and 512),
    // so the existing KV cache sizing works as-is.
    let d_k_row = n_heads * (kv_lora_rank + qk_rope);
    let d_v_row = n_heads * kv_lora_rank;
    let q_head_dim = kv_lora_rank + qk_rope;
    let v_head_dim_eff = kv_lora_rank;
    let base_scale = 1.0f32 / (q_head_dim as f32).sqrt();
    let scale = if yarn_active {
        let mscale = 1.0 + cfg.yarn_log_multiplier * cfg.yarn_factor.ln();
        base_scale * mscale * mscale
    } else { base_scale };
    let (page_table_dev, block_size) = paged_cfg.expect(
        "absorbed MLA requires --paged mode (contiguous KV not implemented).");
    aether_op_paged_append_kv_mla_devarg_f32_cuda(
        k_row, v_row, kv.k_cache, kv.v_cache, page_table_dev,
        d_k_row, d_v_row, block_size, step_args);
    aether_op_paged_attention_mla_devarg_f32_cuda(
        q_full, kv.k_cache, kv.v_cache, page_table_dev, attn_v_out,
        n_heads, q_head_dim, v_head_dim_eff, block_size,
        scale, max_seq as c_int, step_args);
    probe("attn_v_out", attn_v_out, (n_heads * kv_lora_rank) as usize);

    // Step 7: Per-head V absorption via wv_b — reduces kv_lora_rank to
    // value_mla per head.  Writes into act.attn_out (first n_heads*val_mla
    // floats).  Dispatch on bw.dt_v_b.
    mla_absorb_v_dispatch(
        bw.dt_v_b, attn_v_out, bw.w_v_b, act.attn_out,
        n_heads, kv_lora_rank, val_mla);
    probe("attn_out", act.attn_out, (n_heads * val_mla) as usize);

    // Per-call workspace is persistent on ActivationGpu — nothing to free.
    let _ = q_a_dim; // suppress unused-binding warning when q_lora_rank == 0
}

/// FR-17-extra-mla-absorbed-dtypes — Q absorption dispatch.  Maps the GGUF
/// dtype code of `attn_k_b` to the matching CUDA kernel.  Each kernel reads
/// `q_nope_per_head` input elements (the non-rope sub-region of q_proj per
/// head) and outputs to `q_full[h, oi]` for oi in [0, kv_lora_rank + qk_rope).
/// Asserts the alignment requirement of the chosen quant block size, and
/// panics with a clear dtype string for unsupported dtypes so future archs
/// surface the gap explicitly rather than producing silent wrong output.
unsafe fn mla_absorb_q_dispatch(
    dt: i32, q_proj: i64, w_k_b: i64, q_full: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, q_nope_per_head: c_int,
) {
    match dt {
        1 => {
            // F16 — no block alignment; n_in_per_row = q_nope_per_head.
            aether_op_mla_absorb_q_f16_cuda(
                q_proj, w_k_b, q_full,
                n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head);
        }
        8 => {
            // Q8_0 — 32-elem blocks.
            assert!(q_nope_per_head % 32 == 0,
                "q_nope_per_head ({}) must be a multiple of 32 for Q8_0 wk_b",
                q_nope_per_head);
            aether_op_mla_absorb_q_q8_0_cuda(
                q_proj, w_k_b, q_full,
                n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head / 32);
        }
        20 => {
            // IQ4_NL — 32-elem blocks (same shape as Q4_0, codebook lookup).
            assert!(q_nope_per_head % 32 == 0,
                "q_nope_per_head ({}) must be a multiple of 32 for IQ4_NL wk_b",
                q_nope_per_head);
            aether_op_mla_absorb_q_iq4_nl_cuda(
                q_proj, w_k_b, q_full,
                n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head / 32);
        }
        12 => {
            // Q4_K — 256-elem super-blocks.
            assert!(q_nope_per_head % 256 == 0,
                "q_nope_per_head ({}) must be a multiple of 256 for Q4_K wk_b",
                q_nope_per_head);
            aether_op_mla_absorb_q_q4_k_cuda(
                q_proj, w_k_b, q_full,
                n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head / 256);
        }
        13 => {
            // Q5_K — 256-elem super-blocks.
            assert!(q_nope_per_head % 256 == 0,
                "q_nope_per_head ({}) must be a multiple of 256 for Q5_K wk_b",
                q_nope_per_head);
            aether_op_mla_absorb_q_q5_k_cuda(
                q_proj, w_k_b, q_full,
                n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head / 256);
        }
        14 => {
            // Q6_K — 256-elem super-blocks.
            assert!(q_nope_per_head % 256 == 0,
                "q_nope_per_head ({}) must be a multiple of 256 for Q6_K wk_b",
                q_nope_per_head);
            aether_op_mla_absorb_q_q6_k_cuda(
                q_proj, w_k_b, q_full,
                n_heads, key_mla, qk_rope, kv_lora_rank, q_nope_per_head / 256);
        }
        _ => panic!(
            "mla_absorb_q_dispatch: unsupported attn_k_b dtype dt={} \
             (supported: 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K, 20=IQ4_NL; \
             TODO: 6=Q5_0, 18=IQ3_XXS, 21=IQ3_S, 23=IQ4_XS)", dt),
    }
}

/// FR-17-extra-mla-absorbed-dtypes — V reduction dispatch.  Mirror of the
/// Q-absorption dispatch but for `attn_v_b`.  Input per head is
/// `kv_lora_rank` floats (the paged_attention output); output per head is
/// `value_mla` floats written into `attn_out`.
unsafe fn mla_absorb_v_dispatch(
    dt: i32, attn_v_out: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, val_mla: c_int,
) {
    match dt {
        1 => {
            // F16 — no block alignment; n_in_per_row = kv_lora_rank.
            aether_op_mla_absorb_v_f16_cuda(
                attn_v_out, w_v_b, attn_out,
                n_heads, kv_lora_rank, val_mla, kv_lora_rank);
        }
        8 => {
            assert!(kv_lora_rank % 32 == 0,
                "kv_lora_rank ({}) must be a multiple of 32 for Q8_0 wv_b",
                kv_lora_rank);
            aether_op_mla_absorb_v_q8_0_cuda(
                attn_v_out, w_v_b, attn_out,
                n_heads, kv_lora_rank, val_mla, kv_lora_rank / 32);
        }
        20 => {
            assert!(kv_lora_rank % 32 == 0,
                "kv_lora_rank ({}) must be a multiple of 32 for IQ4_NL wv_b",
                kv_lora_rank);
            aether_op_mla_absorb_v_iq4_nl_cuda(
                attn_v_out, w_v_b, attn_out,
                n_heads, kv_lora_rank, val_mla, kv_lora_rank / 32);
        }
        12 => {
            assert!(kv_lora_rank % 256 == 0,
                "kv_lora_rank ({}) must be a multiple of 256 for Q4_K wv_b",
                kv_lora_rank);
            aether_op_mla_absorb_v_q4_k_cuda(
                attn_v_out, w_v_b, attn_out,
                n_heads, kv_lora_rank, val_mla, kv_lora_rank / 256);
        }
        13 => {
            assert!(kv_lora_rank % 256 == 0,
                "kv_lora_rank ({}) must be a multiple of 256 for Q5_K wv_b",
                kv_lora_rank);
            aether_op_mla_absorb_v_q5_k_cuda(
                attn_v_out, w_v_b, attn_out,
                n_heads, kv_lora_rank, val_mla, kv_lora_rank / 256);
        }
        14 => {
            assert!(kv_lora_rank % 256 == 0,
                "kv_lora_rank ({}) must be a multiple of 256 for Q6_K wv_b",
                kv_lora_rank);
            aether_op_mla_absorb_v_q6_k_cuda(
                attn_v_out, w_v_b, attn_out,
                n_heads, kv_lora_rank, val_mla, kv_lora_rank / 256);
        }
        _ => panic!(
            "mla_absorb_v_dispatch: unsupported attn_v_b dtype dt={} \
             (supported: 1=F16, 8=Q8_0, 12=Q4_K, 13=Q5_K, 14=Q6_K, 20=IQ4_NL; \
             TODO: 6=Q5_0, 18=IQ3_XXS, 21=IQ3_S, 23=IQ4_XS)", dt),
    }
}

/// Re-exported just to keep `aether_op_matmul_nt_f32_cuda` reachable as a
/// dependency of the bert module via the same `use crate::cuda::*` block.
/// Eliminating this trampoline is a future cleanup once we replace the
/// per-call workspace allocs with persistent buffers on ActivationGpu.
#[inline(always)]
fn matmul_nt_f32_cuda_unused() -> i32 {
    let _ = aether_op_matmul_nt_f32_cuda; 0
}

// ---------------------------------------------------------------------------
// FR-17-extra-moe-quant: MoE expert-matmul dispatch table.
//
// Adding a new dtype to the MoE expert dispatch is now a one-line table row:
//   ExpertDtype { dt: <ggml_type_int>, block_n_elems: <32 or 256>, kernel: <fn> },
//
// The kernel signature is the per-expert fused-matmul ABI:
//     fn(x: i64, w_base: i64, y: i64, n_out: c_int,
//        blocks_per_row: c_int, expert_idx: c_int) -> c_int
//
// `block_n_elems` is the number of input elements consumed per quant block
// (32 for Q8_0/Q5_0/Q4_0/IQ4_NL/Q5_1/Q8_1; 256 for Q4_K/Q5_K/Q6_K/IQ3_S/
// IQ3_XXS/IQ4_XS).  It determines `blocks_per_row = n_in / block_n_elems`.
//
// Coverage today (GLM-4.7-flash + DeepSeek-V2-Lite + Qwen3-MoE):
//   Q4_K(12), Q5_0(6), Q8_0(8), IQ3_XXS(18), IQ3_S(21), IQ4_XS(23)
//
// Cold-path dtypes that have a STANDALONE `dispatch_matmul` kernel but no
// expert-variant kernel yet (would require a new fused_<dt>_expert_matmul_seq1
// in cuda.rs to add — same pattern as the existing 6):
//   F32(0), F16(1), Q4_0(2), Q5_K(13), Q6_K(14), IQ4_NL(20)
// These will panic with a clear "add a table row + expert kernel" message.
//
// roadmap: P17.5

type ExpertKernel = extern "C" fn(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int;

struct ExpertDtype {
    dt: i32,
    block_n_elems: usize,
    kernel: ExpertKernel,
}

const MOE_EXPERT_DISPATCH: &[ExpertDtype] = &[
    ExpertDtype { dt: 12, block_n_elems: 256, kernel: aether_op_fused_q4k_expert_matmul_seq1_cuda },
    ExpertDtype { dt:  8, block_n_elems:  32, kernel: aether_op_fused_q8_0_expert_matmul_seq1_cuda },
    ExpertDtype { dt:  6, block_n_elems:  32, kernel: aether_op_fused_q5_0_expert_matmul_seq1_cuda },
    ExpertDtype { dt: 21, block_n_elems: 256, kernel: aether_op_fused_iq3_s_expert_matmul_seq1_cuda },
    ExpertDtype { dt: 11, block_n_elems: 256, kernel: aether_op_fused_q3_k_expert_matmul_seq1_cuda },
    ExpertDtype { dt: 13, block_n_elems: 256, kernel: aether_op_fused_q5_k_expert_matmul_seq1_cuda },
    ExpertDtype { dt: 23, block_n_elems: 256, kernel: aether_op_fused_iq4_xs_expert_matmul_seq1_cuda },
    ExpertDtype { dt: 18, block_n_elems: 256, kernel: aether_op_fused_iq3_xxs_expert_matmul_seq1_cuda },
    // <- add new dtypes here, one line each, plus the matching kernel
    //    `fused_<dt>_expert_matmul_seq1` in cuda.rs.
];

/// Look up the MoE expert-matmul kernel for weight dtype `dt`.  Panics with
/// a directive pointing at MOE_EXPERT_DISPATCH if the dtype isn't tabled.
#[inline]
fn lookup_expert_kernel(dt: i32) -> &'static ExpertDtype {
    for entry in MOE_EXPERT_DISPATCH {
        if entry.dt == dt { return entry; }
    }
    let supported: Vec<String> = MOE_EXPERT_DISPATCH.iter()
        .map(|e| e.dt.to_string()).collect();
    panic!(
        "moe expert matmul: unsupported dtype {} (supported: [{}]).  \
         To add: append `ExpertDtype {{ dt: {}, block_n_elems: <32|256>, \
         kernel: aether_op_fused_<DT>_expert_matmul_seq1_cuda }}` to \
         MOE_EXPERT_DISPATCH in serving.rs and ship the matching kernel \
         in cuda.rs.",
        dt, supported.join(","), dt);
}

unsafe fn moe_ffn_forward(bw: &BlockGpu, act: &ActivationGpu, cfg: &ModelConfig) {
    let d_model = cfg.d_model;
    // Per-expert FFN hidden dim.  DeepSeek-V2 / GLM-4.7-flash store this in
    // `<arch>.expert_feed_forward_length`; we cached it as cfg.expert_ff_dim.
    // For older Qwen3-MoE GGUFs where the metadata is absent (we default
    // expert_ff_dim to 0), fall back to cfg.d_ff to preserve legacy behavior.
    let expert_ff = if cfg.expert_ff_dim > 0 { cfg.expert_ff_dim } else { cfg.d_ff };
    let n_experts = cfg.n_experts;
    let n_used = cfg.n_experts_used.max(1);
    assert!(n_experts > 0, "moe_ffn_forward called on dense block");

    // Workspace allocations sized to PER-EXPERT FFN dim (not cfg.d_ff).
    // The previous bug: V2-Lite has d_ff=10944 (dense) and expert_ff=1408,
    // and treating d_ff as the per-expert n_out makes the kernel walk the
    // per-expert pointer 7.77× too far into adjacent memory → garbage →
    // NaN cascades through every MoE block.
    let router_logits = aether_dev_alloc_f32(n_experts as c_int);
    let gate_e = aether_dev_alloc_f32(expert_ff as c_int);
    let up_e = aether_dev_alloc_f32(expert_ff as c_int);
    let down_e = aether_dev_alloc_f32(d_model as c_int);
    let out_acc = aether_dev_alloc_f32(d_model as c_int);
    let zero = vec![0f32; d_model];
    aether_dev_h2d_f32(zero.as_ptr() as i64, out_acc, d_model as c_int);

    // 1. router_logits = w_router @ x_norm.
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

    // 3. Per-expert forward.  Dispatch on dtype-per-tensor via the
    // MOE_EXPERT_DISPATCH table above.  `blocks_per_row` is derived from the
    // per-dtype `block_n_elems`, so 32-elem block dtypes (Q8_0/Q5_0) and
    // 256-elem block dtypes (Q4_K/Q5_K/IQ3_S/IQ4_XS/IQ3_XXS) coexist.
    //
    // Common case: expert_ff isn't a multiple of 256 (e.g. V2-Lite
    // expert_ff=1408 / 32 = 44) so the `down` tensor MUST land on a
    // 32-elem-block dtype if any expert is non-Q4_K.  The dtype-per-tensor
    // routing handles this transparently.
    let exp_ff_c = expert_ff as c_int;
    let d_model_c = d_model as c_int;

    let dispatch_expert = |x_in: i64, w_base: i64, dt: i32, y: i64,
                            n_out: c_int, n_in_d_model: bool, expert_idx: c_int| {
        // n_in_d_model=true → n_in = d_model.  Else n_in = expert_ff.
        let entry = lookup_expert_kernel(dt);
        let n_in = if n_in_d_model { d_model } else { expert_ff };
        debug_assert!(n_in % entry.block_n_elems == 0,
            "moe expert dt={}: n_in={} not divisible by block_n_elems={}",
            dt, n_in, entry.block_n_elems);
        let bpr = (n_in / entry.block_n_elems) as c_int;
        (entry.kernel)(x_in, w_base, y, n_out, bpr, expert_idx);
    };

    for (k, &expert_idx) in selected.iter().enumerate() {
        let w_i = weights[k];
        // gate_e = expert_matmul(x_norm, w_gate_exps, expert_idx)  [expert_ff]
        dispatch_expert(act.x_norm, bw.w_gate_exps, bw.dt_gate_exps,
            gate_e, exp_ff_c, true, expert_idx as c_int);
        // up_e = expert_matmul(x_norm, w_up_exps, expert_idx)
        dispatch_expert(act.x_norm, bw.w_up_exps, bw.dt_up_exps,
            up_e, exp_ff_c, true, expert_idx as c_int);
        // silu(gate_e); gate_e *= up_e
        aether_op_silu_f32_cuda(gate_e, exp_ff_c);
        aether_op_mul_inplace_f32_cuda(gate_e, up_e, exp_ff_c);
        // down_e = expert_matmul(gate_e, w_down_exps, expert_idx)  [d_model]
        dispatch_expert(gate_e, bw.w_down_exps, bw.dt_down_exps,
            down_e, d_model_c, false, expert_idx as c_int);
        // out_acc += w_i * down_e
        aether_op_scale_f32_cuda(down_e, w_i, d_model_c);
        aether_op_add_inplace_f32_cuda(out_acc, down_e, d_model_c);
    }

    // 4. Shared experts (FR-17-extra-mla-fwd MoE shared).  DeepSeek-V2 /
    // GLM-4.7-flash have a small always-on FFN alongside the routed
    // experts.  The GGUF pre-concatenates the n_shared experts into a
    // single fused MLP with hidden dim d_ff_shared = n_shared *
    // expert_ff_dim — so this is just a regular gate/up/silu_mul/down
    // chain at that hidden dim.  Contribution is added at full weight 1.0.
    if cfg.n_shared_experts > 0 && bw.w_gate_shexp != 0 {
        let d_ff_shared = (cfg.n_shared_experts as usize) * cfg.expert_ff_dim;
        if d_ff_shared > 0 {
            let gate_sh = aether_dev_alloc_f32(d_ff_shared as c_int);
            let up_sh   = aether_dev_alloc_f32(d_ff_shared as c_int);
            let down_sh = aether_dev_alloc_f32(d_model as c_int);
            // gate = x_norm @ w_gate_shexp^T  [d_ff_shared]
            dispatch_matmul(act.x_norm, bw.w_gate_shexp, bw.dt_gate_shexp,
                gate_sh, d_ff_shared as c_int, d_model as c_int);
            dispatch_matmul(act.x_norm, bw.w_up_shexp, bw.dt_up_shexp,
                up_sh, d_ff_shared as c_int, d_model as c_int);
            aether_op_silu_f32_cuda(gate_sh, d_ff_shared as c_int);
            aether_op_mul_inplace_f32_cuda(gate_sh, up_sh, d_ff_shared as c_int);
            // down = gate @ w_down_shexp^T  [d_model]
            dispatch_matmul(gate_sh, bw.w_down_shexp, bw.dt_down_shexp,
                down_sh, d_model as c_int, d_ff_shared as c_int);
            // out_acc += shared_output (weight 1.0)
            aether_op_add_inplace_f32_cuda(out_acc, down_sh, d_model as c_int);
            let _ = aether_dev_free_f32(gate_sh);
            let _ = aether_dev_free_f32(up_sh);
            let _ = aether_dev_free_f32(down_sh);
        }
    }

    // 5. x += out_acc (residual).
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
    /// FR-x-extra-text-encode — lazy cache of byte → token-id for the
    /// 256 GPT-2 surface-char single-byte vocab entries.  Built on
    /// first `encode_text` call by mapping each byte 0..=255 through
    /// the GPT-2 byte→unicode alphabet and looking up the resulting
    /// surface char (UTF-8 encoded) in the BPE decode_table.  -1 in a
    /// slot means "no vocab entry for that surface char" — which
    /// shouldn't happen for any well-formed GPT-2-style vocab.
    byte_to_id_cache: std::sync::OnceLock<Box<[i32; 256]>>,
    /// FR-x-extra-chat-template — Jinja-style chat template loaded from
    /// `tokenizer.chat_template` GGUF metadata.  `None` if the model
    /// doesn't ship one (e.g. base / non-instruct models).  Used by
    /// `apply_chat_template` to render `messages: [(role, content)]`
    /// into the wire text the model was trained to expect.
    chat_template: Option<String>,
    /// FR-19.5-extra-deep Phase 2b-2b — lazily-allocated batched-decode
    /// workspace.  `None` until the first `step_logits_for_batch` call.
    /// Single-stream decode never touches it (zero cost).
    batch_state: Option<BatchActivationGpu>,
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
        if s.cfg.kv_lora_rank > 0 {
            return Err(format!(
                "FR-17-extra-mla-fwd: SharedKvPool doesn't support MLA archs yet \
                 (deepseek2 / glm-4.7-flash).  MLA needs asymmetric K-row \
                 (n_heads * qk_head_dim = {}) and V-row (n_heads * v_head_dim = {}) \
                 strides; SharedKvPool currently allocates both at the same d_kv. \
                 Either use --paged (no --pool-blocks) for single-tenant MLA, \
                 or wait for FR-17-extra-mla-pool.",
                s.cfg.n_q_heads * s.cfg.qk_head_dim as usize,
                s.cfg.n_q_heads * s.cfg.v_head_dim as usize));
        }
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
            // For MLA archs (kv_lora_rank > 0) the standard head_dim check
            // doesn't apply — Q/K/V have asymmetric per-head dims (qk_head_dim
            // vs v_head_dim) read from `attention.key_length` /
            // `attention.value_length`.  paged_attention_mla_devarg's register
            // arrays are sized for up to 640 elements per head (q_local[20] ×
            // per_lane=20).  GLM-4.7-flash (qk=576, v=512) fits; DeepSeek-V2
            // (qk=192, v=128) and V2-Lite (qk=192, v=128) also fit.
            if cfg.kv_lora_rank > 0 {
                if cfg.qk_head_dim <= 0 || cfg.qk_head_dim > 640 {
                    return Err(format!(
                        "FR-17-extra-mla-fwd: qk_head_dim={} out of range [1, 640] \
                         (MLA attention kernel q_local[20] × per_lane=20 maxes out).",
                        cfg.qk_head_dim));
                }
                if cfg.v_head_dim <= 0 || cfg.v_head_dim > 640 {
                    return Err(format!(
                        "FR-17-extra-mla-fwd: v_head_dim={} out of range [1, 640] \
                         (MLA attention kernel out_local[20] × per_lane=20 maxes out).",
                        cfg.v_head_dim));
                }
            } else if cfg.head_dim == 0 || cfg.head_dim > 256 {
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
                // FR-17-extra-mla-fwd: relax the d_ff alignment check for MLA
                // archs.  DeepSeek-V2-Lite's dense d_ff (10944) and per-expert
                // ffn (1408) are both non-multiples of 256; the dense layer-0
                // matmul will fail at runtime when it hits the unaligned Q4_K
                // kernel (a separate FR-17-extra-q4k-pad work item).  We let
                // construction proceed so the MLA attention path can still
                // be exercised on dense-FFN-skipping flows / tests.
                if cfg.kv_lora_rank == 0 {
                    return Err(format!(
                        "FR-17-extra-runtime-shape: d_ff({}) must be a multiple of 256 (Q4_K super-block).",
                        cfg.d_ff));
                } else {
                    eprintln!("[QwenSession] WARN d_ff({}) not multiple of 256; \
                        FFN will fail at runtime — MLA attention path is still constructible. \
                        FR-17-extra-q4k-pad tracks the unaligned-FFN fix.",
                        cfg.d_ff);
                }
            }

            let blocks: Vec<BlockGpu> = (0..cfg.n_layers).map(|b| load_block(h, b)).collect();
            let final_norm_g = upload_f32_tensor(h, "output_norm.weight");
            // Most archs ship an explicit `output.weight` lm head. Gemma3 (and other
            // tied-embedding archs) omit it and reuse `token_embd.weight` as the lm
            // head (logits = hidden @ token_embd^T). Fall back to token_embd when
            // output.weight is absent rather than panicking.
            let (lm_head, lm_n_blocks, lm_dt) = {
                let (oh, on, odt) = upload_tensor_u8_opt(h, "output.weight");
                if oh != 0 {
                    (oh, on, odt)
                } else {
                    eprintln!("[QwenSession] output.weight absent (tied embeddings) — using token_embd.weight as lm head");
                    upload_tensor_u8(h, "token_embd.weight")
                }
            };

            // FR-17-extra-mla-fwd: MLA archs (deepseek2, glm-4.7-flash) need
            // bigger Q (per-head qk_head_dim instead of head_dim) and an
            // asymmetric K-row / V-row stride.  Non-MLA archs collapse to
            // n_kv_heads * head_dim = cfg.d_kv for both.
            let is_mla = cfg.kv_lora_rank > 0;
            let q_total = if is_mla {
                cfg.n_q_heads * cfg.qk_head_dim as usize
            } else {
                // Q projection output = n_q_heads * head_dim. Equals d_model for
                // Qwen/Llama; differs for Mistral Small 24B / Gemma3 (explicit
                // head_dim). See ModelConfig::from_gguf head_dim derivation.
                cfg.n_q_heads * cfg.head_dim
            };
            let d_k_row = if is_mla {
                cfg.n_q_heads * cfg.qk_head_dim as usize
            } else {
                cfg.d_kv
            };
            let d_v_row = if is_mla {
                cfg.n_q_heads * cfg.v_head_dim as usize
            } else {
                cfg.d_kv
            };
            let attn_out_dim = if is_mla {
                cfg.n_q_heads * cfg.v_head_dim as usize
            } else {
                // = n_q_heads * head_dim (o-proj input). d_model for Qwen/Llama;
                // smaller for Mistral Small 24B (4096 < 5120 hidden).
                cfg.n_q_heads * cfg.head_dim
            };
            if is_mla {
                eprintln!("[QwenSession] MLA mode: q_total={} d_k_row={} d_v_row={} attn_out_dim={}",
                    q_total, d_k_row, d_v_row, attn_out_dim);
                let b0 = &blocks[0];
                let q_branch = if cfg.q_lora_rank > 0 && b0.w_q_a != 0 && b0.w_q_b != 0 {
                    format!("Q-LoRA (q_lora_rank={}, w_q_a=0x{:x}, w_q_b=0x{:x})",
                        cfg.q_lora_rank, b0.w_q_a, b0.w_q_b)
                } else {
                    format!("direct attn_q (q_lora_rank=0, w_q=0x{:x})", b0.w_q)
                };
                eprintln!("[QwenSession] MLA Q branch (layer 0): {}", q_branch);
            }
            // FR-17-extra-mla-absorbed-persist — persistent workspace buffers
            // for the absorbed-MLA forward path.  Allocated once here (per
            // session) and reused across every layer × token, replacing the
            // 11-buffer per-call alloc/free pattern in
            // `mla_attention_forward_absorbed`.  On a 47-layer GLM-4.7-flash
            // decode this drops ~517 device allocations per token to ~0.
            let mla_absorbed = cfg.is_mla_absorbed();
            let mla_abs_kv_a = if mla_absorbed {
                aether_dev_alloc_f32((cfg.kv_lora_rank + cfg.qk_rope_head_dim) as c_int) } else { 0 };
            let mla_abs_c_kv = if mla_absorbed {
                aether_dev_alloc_f32(cfg.kv_lora_rank as c_int) } else { 0 };
            let mla_abs_c_kv_n = if mla_absorbed {
                aether_dev_alloc_f32(cfg.kv_lora_rank as c_int) } else { 0 };
            let mla_abs_k_pe = if mla_absorbed {
                aether_dev_alloc_f32(cfg.qk_rope_head_dim as c_int) } else { 0 };
            let mla_abs_q_a = if mla_absorbed && cfg.q_lora_rank > 0 {
                aether_dev_alloc_f32(cfg.q_lora_rank as c_int) } else { 0 };
            let mla_abs_q_a_n = if mla_absorbed && cfg.q_lora_rank > 0 {
                aether_dev_alloc_f32(cfg.q_lora_rank as c_int) } else { 0 };
            let mla_abs_q_proj = if mla_absorbed {
                aether_dev_alloc_f32((cfg.n_q_heads as i32 * cfg.key_length_mla) as c_int) } else { 0 };
            let mla_abs_q_full = if mla_absorbed {
                aether_dev_alloc_f32(
                    (cfg.n_q_heads as i32 * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)) as c_int)
            } else { 0 };
            let mla_abs_k_row = if mla_absorbed {
                aether_dev_alloc_f32(
                    (cfg.n_q_heads as i32 * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)) as c_int)
            } else { 0 };
            let mla_abs_v_row = if mla_absorbed {
                aether_dev_alloc_f32((cfg.n_q_heads as i32 * cfg.kv_lora_rank) as c_int) } else { 0 };
            let mla_abs_attn_v_out = if mla_absorbed {
                aether_dev_alloc_f32((cfg.n_q_heads as i32 * cfg.kv_lora_rank) as c_int) } else { 0 };
            let act = ActivationGpu {
                x: aether_dev_alloc_f32(cfg.d_model as c_int),
                x_norm: aether_dev_alloc_f32(cfg.d_model as c_int),
                q: aether_dev_alloc_f32(q_total as c_int),
                k_step: aether_dev_alloc_f32(d_k_row as c_int),
                v_step: aether_dev_alloc_f32(d_v_row as c_int),
                attn_out: aether_dev_alloc_f32(attn_out_dim as c_int),
                proj: aether_dev_alloc_f32(cfg.d_model as c_int),
                gate: aether_dev_alloc_f32(cfg.d_ff as c_int),
                down: aether_dev_alloc_f32(cfg.d_model as c_int),
                logits: aether_dev_alloc_f32(cfg.vocab as c_int),
                mla_abs_kv_a, mla_abs_c_kv, mla_abs_c_kv_n, mla_abs_k_pe,
                mla_abs_q_a, mla_abs_q_a_n, mla_abs_q_proj, mla_abs_q_full,
                mla_abs_k_row, mla_abs_v_row, mla_abs_attn_v_out,
            };
            let kvs: Vec<KvCacheGpu> = (0..cfg.n_layers).map(|_| KvCacheGpu {
                k_cache: aether_dev_alloc_f32((MAX_SEQ * d_k_row) as c_int),
                v_cache: aether_dev_alloc_f32((MAX_SEQ * d_v_row) as c_int),
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
            // FR-x-extra-chat-template: best-effort read of the GGUF's
            // `tokenizer.chat_template` string metadata.  Buffer is 64 KiB
            // since some templates (DeepSeek / Yi family) approach 8 KiB,
            // and concatenated function-calling templates can grow.
            let chat_template = {
                let key = b"tokenizer.chat_template";
                let mut buf = vec![0u8; 65536];
                let nb = aether_gguf_get_metadata_string(
                    h, key.as_ptr() as i64, key.len() as c_int,
                    buf.as_mut_ptr() as i64, buf.len() as c_int);
                if nb > 0 {
                    match std::str::from_utf8(&buf[..nb as usize]) {
                        Ok(s) => {
                            eprintln!("[QwenSession] chat_template loaded ({} bytes)", nb);
                            Some(s.to_string())
                        }
                        Err(_) => None,
                    }
                } else {
                    eprintln!("[QwenSession] no chat_template in GGUF metadata");
                    None
                }
            };
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
                byte_to_id_cache: std::sync::OnceLock::new(),
                chat_template,
                batch_state: None,
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
        // token_embd dtype is NOT always Q4_K: gemma3=Q6_K, qwen3moe=Q3_K,
        // IQ3_M models=IQ3_S. Hardcoding Q4_K dequant here garbled the embeddings
        // for those → degenerate forward (vocab-1 pin). Dispatch on the real dtype.
        let dt = aether_gguf_get_tensor_dtype(self.gguf_handle, idx);
        let bytes_per_row = match dt {
            12 => blocks_per_row * 144,  // Q4_K
            14 => blocks_per_row * 210,  // Q6_K
            11 => blocks_per_row * 110,  // Q3_K
            21 => blocks_per_row * 110,  // IQ3_S
            _  => panic!("dequant_embd_row: unsupported token_embd dtype {} \
                          (have Q4_K/Q6_K/Q3_K/IQ3_S)", dt),
        };
        assert!(token_id < total_rows, "token_id {} out of vocab {}", token_id, total_rows);
        let row_bytes = std::slice::from_raw_parts(
            dptr.add(token_id * bytes_per_row), bytes_per_row);
        let mut row_f32 = vec![0.0f32; self.cfg.d_model];
        let bp = blocks_per_row as c_int;
        let src = row_bytes.as_ptr() as *const c_void;
        let dst = row_f32.as_mut_ptr() as *mut c_void;
        match dt {
            12 => { aether_dequant_q4_k_m(src, dst, bp); }
            14 => { aether_dequant_q6_k(src, dst, bp); }
            11 => { aether_dequant_q3_k(src, dst, bp); }
            21 => { aether_dequant_iq3_s(src, dst, bp); }
            _  => unreachable!(),
        }
        // Gemma scales input embeddings by sqrt(d_model) (llama.cpp `inp_scale`);
        // without it the residual stream is ~sqrt(d) too small and the forward
        // produces degenerate logits (vocab-1/pad). Other archs use no scale.
        if self.cfg.arch == "gemma3" {
            let s = (self.cfg.d_model as f32).sqrt();
            for v in row_f32.iter_mut() { *v *= s; }
        }
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
        let dump_blocks = std::env::var("AETHER_DUMP_BLOCKS").is_ok();
        for b in 0..self.cfg.n_layers {
            block_forward_devarg(&self.blocks[b], &self.act, &self.kvs[b],
                self.step_args, self.paged_arg(), &self.cfg, MAX_SEQ);
            // FR-17-extra-mla-fwd debug — dump `act.x` stats after each
            // block to bisect where NaN first appears in the V2-Lite
            // forward.  AETHER_DUMP_BLOCKS=1 to enable.  Adds one D2H
            // per layer; only set during diagnostic runs.
            if dump_blocks && b < 4 {
                aether_dev_sync();
                let mut x_host = vec![0.0f32; self.cfg.d_model];
                aether_dev_d2h_f32(self.act.x, x_host.as_mut_ptr() as i64,
                    self.cfg.d_model as c_int);
                let n_nan = x_host.iter().filter(|x| x.is_nan()).count();
                let n_inf = x_host.iter().filter(|x| x.is_infinite()).count();
                let finite: Vec<f32> = x_host.iter().cloned()
                    .filter(|x| x.is_finite()).collect();
                let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean = if !finite.is_empty() {
                    finite.iter().sum::<f32>() / finite.len() as f32
                } else { 0.0 };
                eprintln!("[BLOCK b={}] x: nan={} inf={} min={:.4e} max={:.4e} mean={:.4e}",
                    b, n_nan, n_inf, mn, mx, mean);
            }
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
            // FR-17-extra-mla-fwd debug: on the FIRST decode step, dump
            // logits statistics so we can diagnose the "all-tokens=vocab-1"
            // degenerate output we hit on the cnc V2-Lite Q4_K_M load.
            // Remove once decode is witnessed.
            if pos < 32 && std::env::var("AETHER_DUMP_LOGITS").is_ok() {
                let n_nan = logits.iter().filter(|x| x.is_nan()).count();
                let n_inf = logits.iter().filter(|x| x.is_infinite()).count();
                let n_zero = logits.iter().filter(|x| **x == 0.0).count();
                let finite: Vec<f32> = logits.iter().cloned()
                    .filter(|x| x.is_finite()).collect();
                let mn = finite.iter().cloned().fold(f32::INFINITY, f32::min);
                let mx = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = finite.iter().sum();
                let mean = if !finite.is_empty() { sum / finite.len() as f32 } else { 0.0 };
                let var = finite.iter().map(|x| (x - mean).powi(2))
                    .sum::<f32>() / finite.len().max(1) as f32;
                let std_ = var.sqrt();
                // top-5 argmax
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_unstable_by(|a, b|
                    logits[*b].partial_cmp(&logits[*a])
                        .unwrap_or(std::cmp::Ordering::Equal));
                let top: Vec<(usize, f32)> = idx[..5.min(idx.len())].iter()
                    .map(|&i| (i, logits[i])).collect();
                eprintln!(
                    "[DUMP pos={}] vocab={} nan={} inf={} zero={} min={:.4e} max={:.4e} mean={:.4e} std={:.4e} top5={:?}",
                    pos, self.cfg.vocab, n_nan, n_inf, n_zero, mn, mx, mean, std_, top);
            }
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
                if b.w_k_b != 0 { let _ = aether_dev_free_u8(b.w_k_b); }
                if b.w_v_b != 0 { let _ = aether_dev_free_u8(b.w_v_b); }
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
            // Persistent MLA-absorbed workspace buffers (0 if not allocated).
            for h in [self.act.mla_abs_kv_a, self.act.mla_abs_c_kv,
                      self.act.mla_abs_c_kv_n, self.act.mla_abs_k_pe,
                      self.act.mla_abs_q_a, self.act.mla_abs_q_a_n,
                      self.act.mla_abs_q_proj, self.act.mla_abs_q_full,
                      self.act.mla_abs_k_row, self.act.mla_abs_v_row,
                      self.act.mla_abs_attn_v_out] {
                if h != 0 { let _ = aether_dev_free_f32(h); }
            }
            // FR-19.5-extra-deep Phase 2b-2b — batched-decode workspace.
            if let Some(ba) = self.batch_state.take() {
                for h in [ba.x, ba.x_norm, ba.q, ba.k_step, ba.v_step,
                          ba.attn_out, ba.proj, ba.gate, ba.up, ba.down,
                          ba.logits, ba.scratch_in, ba.scratch_out] {
                    if h != 0 { let _ = aether_dev_free_f32(h); }
                }
                for h in [ba.pos_batch, ba.cur_seq_batch, ba.page_table_batch] {
                    if h != 0 { let _ = aether_dev_free_i32(h); }
                }
            }
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

    // GLM-4.7-flash has at least one vocab entry > 512 bytes (entry
    // 112972).  Bump the buffer + handle the "too small" return (-2)
    // by skipping the entry rather than bailing out of the whole
    // tokenizer load.  Each skipped entry inserts an empty Vec at its
    // index so merge-table lookups stay index-aligned (empty bytes
    // never matches a real GPT-2 surface char in encode-side lookup).
    let mut vocab_bytes: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    let mut buf = vec![0u8; 16384];
    let mut skipped_oversized = 0i64;
    for i in 0..n {
        let nb = aether_gguf_get_metadata_array_string_get(
            h, tok_key.as_ptr() as i64, tok_key.len() as c_int, i,
            buf.as_mut_ptr() as i64, buf.len() as c_int);
        if nb == -2 {
            // Entry larger than 16 KiB — skip but keep index alignment.
            skipped_oversized += 1;
            vocab_bytes.push(Vec::new());
            continue;
        }
        if nb < 0 {
            eprintln!("[QwenSession] vocab entry {} failed (nb={})", i, nb);
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
    if skipped_oversized > 0 {
        eprintln!("[QwenSession] {} vocab entries skipped as oversized (> 16 KiB)",
            skipped_oversized);
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

// =====================================================================
// FR-x-extra-text-encode — wire text → token ids for GPT-2-style BPE.
//
// aether-serve's chat-completion handler historically required
// `prompt_ids: [int]` because text encoding wasn't wired.  Every chat
// client (LiteLLM, OpenAI Python lib, raw curl) sends
// `messages: [{role, content: "..."}]` instead — and 501'd.
//
// This impl block bridges the gap: take raw text → GPT-2 byte→unicode
// surface chars → vocab-id lookup → BPE merge loop → token ids.
//
// Limitation vs reference tokenizers:
//   - No GPT-2 regex pre-tokenization (`'s|'t| ?\p{L}+|...`).  The merge
//     loop runs over the full input ids as one stream.  For typical
//     English prompts this still yields valid tokenization that the
//     model can consume — just may not be byte-exact with HF's
//     `AutoTokenizer.encode(text)`.  Pre-tokenization is filed as
//     FR-x-extra-text-encode-regex.
//   - First call rebuilds the byte→id cache via 256 linear scans of the
//     vocab (152K entries each).  Subsequent calls hit the cache.
// =====================================================================

/// FR-x-extra-chat-template: hand-rolled per-arch simplified templates.
///
/// The GGUF-embedded `tokenizer.chat_template` strings on modern
/// instruct models use heavy Jinja features (namespaces, macros,
/// filters, whitespace-strip markers, elif/loop.first, etc.) that the
/// jinja-lite renderer in lib.rs doesn't support.  Rather than expand
/// jinja-lite to full Jinja (a substantial chunk of work), we ship
/// known-good simplified templates per architecture that produce the
/// wire format the model was trained to expect.  apply_chat_template
/// uses these when the GGUF template fails to render.
///
/// Templates are formatted via direct string concat in
/// `apply_per_arch_template` below.  Each is a TINY shape — just the
/// role markers + content — sufficient to put the model in
/// "respond as assistant" mode.
fn apply_per_arch_template(arch: &str, messages: &[(String, String)]) -> Option<String> {
    if messages.is_empty() { return None; }
    let mut out = String::new();
    match arch {
        // GLM / DeepSeek-v2 / DeepSeek-v3 family (architecture = "deepseek2"
        // in the GGUF for GLM-4.7-flash + V2-Lite + V3).  Wire format:
        //   [gMASK]<sop><|user|>{content}<|assistant|>{content}...<|assistant|>
        "deepseek2" | "glm" | "glm4" => {
            out.push_str("[gMASK]<sop>");
            for (role, content) in messages {
                match role.as_str() {
                    "system"    => { out.push_str("<|system|>"); out.push_str(content); }
                    "user"      => { out.push_str("<|user|>");   out.push_str(content); }
                    "assistant" => { out.push_str("<|assistant|>"); out.push_str(content); }
                    _ => {}
                }
            }
            out.push_str("<|assistant|>");
        }
        // Qwen2 / Qwen2.5 / Qwen3:  <|im_start|>{role}\n{content}<|im_end|>\n
        "qwen2" | "qwen3" | "qwen3moe" | "qwen3vl" => {
            for (role, content) in messages {
                out.push_str("<|im_start|>");
                out.push_str(role);
                out.push('\n');
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            out.push_str("<|im_start|>assistant\n");
        }
        // Llama-3:  <|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
        "llama" => {
            out.push_str("<|begin_of_text|>");
            for (role, content) in messages {
                out.push_str("<|start_header_id|>");
                out.push_str(role);
                out.push_str("<|end_header_id|>\n\n");
                out.push_str(content);
                out.push_str("<|eot_id|>");
            }
            out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        // Gemma3:  <start_of_turn>{role}\n{content}<end_of_turn>\n
        "gemma3" => {
            for (role, content) in messages {
                let r = if role == "assistant" { "model" } else { role.as_str() };
                out.push_str("<start_of_turn>");
                out.push_str(r);
                out.push('\n');
                out.push_str(content);
                out.push_str("<end_of_turn>\n");
            }
            out.push_str("<start_of_turn>model\n");
        }
        // Unknown arch — caller falls back to plain-text encode.
        _ => return None,
    }
    Some(out)
}

/// FR-x-extra-chat-template: per-arch list of special-token surface
/// strings.  Each must be looked up in the GGUF vocab via
/// `aether_bpe_lookup_bytes` to find its single-id encoding.  encode_
/// text_with_specials uses this list to split the input around the
/// special tokens and emit each as its proper id rather than letting
/// BPE byte-encode the raw `<|...|>` markers (which the model has
/// never seen at training time).
fn arch_special_tokens(arch: &str) -> &'static [&'static str] {
    match arch {
        "deepseek2" | "glm" | "glm4" => &[
            "[gMASK]", "<sop>",
            "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
            "<|endoftext|>", "<|eot_id|>", "<|eom_id|>",
        ],
        "qwen2" | "qwen3" | "qwen3moe" | "qwen3vl" => &[
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>",
        ],
        "llama" => &[
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
        ],
        "gemma3" => &["<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>"],
        _ => &[],
    }
}

impl QwenSession {
    /// FR-x-extra-chat-template: encode text containing special-token
    /// surface markers (e.g. `<|user|>`, `[gMASK]`) into a mixed id
    /// stream where each known marker becomes its single vocab id and
    /// the gaps are BPE-encoded normally.  Falls back to plain
    /// `encode_text` when no specials match.
    pub fn encode_text_with_specials(&self, text: &str) -> Vec<usize> {
        if self.bpe_handle < 0 || text.is_empty() { return Vec::new(); }
        let specials = arch_special_tokens(&self.cfg.arch);
        if specials.is_empty() { return self.encode_text(text); }

        // Lookup each special surface → vocab id once.  Skip any that
        // don't actually appear in this GGUF's vocab.
        let mut special_ids: Vec<(&str, i32)> = Vec::with_capacity(specials.len());
        for s in specials {
            let id = unsafe {
                aether_bpe_lookup_bytes(
                    self.bpe_handle,
                    s.as_ptr() as *const c_void,
                    s.len() as c_int,
                )
            };
            if id >= 0 { special_ids.push((s, id)); }
        }
        if special_ids.is_empty() { return self.encode_text(text); }

        // Walk the text left-to-right, emitting BPE-encoded chunks
        // between special-marker matches.  Use longest-match-wins so
        // e.g. `<|eot_id|>` beats `<|`.
        let mut out: Vec<usize> = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0usize;
        let mut chunk_start = 0usize;
        while i < bytes.len() {
            let mut matched: Option<(usize, i32)> = None; // (len, id)
            for (s, id) in &special_ids {
                let sb = s.as_bytes();
                if i + sb.len() <= bytes.len() && &bytes[i..i + sb.len()] == sb {
                    if matched.map(|(l, _)| sb.len() > l).unwrap_or(true) {
                        matched = Some((sb.len(), *id));
                    }
                }
            }
            if let Some((slen, id)) = matched {
                if i > chunk_start {
                    let chunk = &text[chunk_start..i];
                    let mut ids = self.encode_text(chunk);
                    out.append(&mut ids);
                }
                out.push(id as usize);
                i += slen;
                chunk_start = i;
            } else {
                i += 1;
            }
        }
        if chunk_start < bytes.len() {
            let chunk = &text[chunk_start..];
            let mut ids = self.encode_text(chunk);
            out.append(&mut ids);
        }
        out
    }

    /// FR-x-extra-chat-template: render `messages` through the GGUF-
    /// embedded jinja-lite chat template.  Returns `None` if no
    /// template is loaded;  the caller should fall back to plain-text
    /// encode of joined contents in that case.
    ///
    /// Templates often reference `add_generation_prompt` and similar
    /// flags that signal "the next thing the model emits is the
    /// assistant turn".  We set this to `"true"` so the template
    /// closes with whatever role marker invites the assistant to
    /// respond.
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> Option<String> {
        if messages.is_empty() { return None; }
        // Always prefer the per-arch hand-rolled template.  It produces
        // exactly the wire format the model was trained on, without
        // depending on jinja-lite handling every feature the GGUF
        // template might use (macros, filters, namespaces, etc.).
        if let Some(s) = apply_per_arch_template(&self.cfg.arch, messages) {
            return Some(s);
        }
        let template = self.chat_template.as_ref()?;
        unsafe {
            let th = aether_template_new();
            if th < 0 { return None; }

            // Common template variables.  Jinja-lite resolves missing
            // ones to "" (empty), so over-setting is safe.
            let _ = aether_template_set_var(th,
                b"add_generation_prompt".as_ptr() as *const c_void, 20,
                b"true".as_ptr() as *const c_void, 4);
            let _ = aether_template_set_var(th,
                b"bos_token".as_ptr() as *const c_void, 9,
                b"".as_ptr() as *const c_void, 0);
            let _ = aether_template_set_var(th,
                b"eos_token".as_ptr() as *const c_void, 9,
                b"".as_ptr() as *const c_void, 0);

            for (role, content) in messages {
                let rc = aether_template_push_message(th,
                    role.as_ptr() as *const c_void, role.len() as c_int,
                    content.as_ptr() as *const c_void, content.len() as c_int);
                if rc != 0 {
                    aether_template_free(th);
                    return None;
                }
            }

            // 256 KiB output buffer — chat templates seldom exceed
            // even 16 KiB but Function-calling / RAG payloads can.
            let mut out = vec![0u8; 256 * 1024];
            let nb = aether_template_render(th,
                template.as_ptr() as *const c_void, template.len() as c_int,
                out.as_mut_ptr() as *mut c_void, out.len() as c_int);
            aether_template_free(th);
            if nb <= 0 { return None; }
            String::from_utf8(out[..nb as usize].to_vec()).ok()
        }
    }

    /// FR-x-extra-sampling: split out from `decode_step` so the caller
    /// can do something other than argmax on the result.  Advances
    /// `next_pos` and returns the raw logits vector (length =
    /// cfg.vocab).  Same forward path as `decode_step`.
    pub fn step_logits(&mut self, last_id: usize) -> Vec<f32> {
        unsafe {
            let pos = self.next_pos;
            if let Err(e) = self.ensure_block_for_position(pos) {
                panic!("[step_logits] pool allocation failed at pos {}: {}", pos, e);
            }
            let emb = self.dequant_embd_row(last_id);
            aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, self.cfg.d_model as c_int);
            let cur_seq = pos + 1;
            let step_host = [pos, cur_seq, 0i32, 0i32];
            aether_dev_h2d_i32(step_host.as_ptr() as i64, self.step_args, 4);

            if self.cfg.n_experts > 0 {
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
            logits
        }
    }

    /// FR-x-extra-sampling: temperature + top-p + top-k sampler over
    /// the generation loop with optional repetition penalties,
    /// logit_bias, and stop-strings.  Mirrors the OpenAI v1 chat-
    /// completions sampler shape so any OpenAI-compatible client
    /// works without translation.
    ///
    /// `params.temperature <= 0.0` falls back to greedy argmax (bit-
    /// identical to the legacy `generate` path).  `params.top_p >=
    /// 1.0` disables nucleus cutoff.  `params.top_k <= 0` disables
    /// top-k cutoff.  `params.seed` if `Some(s)` seeds the per-call
    /// RNG to `s` for deterministic output;  `None` falls back to OS
    /// nanos + pid.
    ///
    /// `stop_strings` is a list of raw text strings;  generation
    /// breaks as soon as the decoded running output ends with any of
    /// them (independent of the token-id stop).
    pub fn generate_sampled_v2(
        &mut self,
        prompt_ids: &[usize],
        max_tokens: usize,
        stop_token: Option<usize>,
        params: &SamplingParams,
        stop_strings: &[String],
    ) -> Vec<usize> {
        self.reset();
        self.prefill(prompt_ids);
        let mut generated = Vec::with_capacity(max_tokens);
        let mut last = *prompt_ids.last().expect("prompt cannot be empty");
        let mut rng = params.seed.unwrap_or_else(seed_rng);
        if rng == 0 { rng = seed_rng(); }
        let mut seen: std::collections::HashMap<usize, u32> =
            std::collections::HashMap::new();
        let mut running_text = String::new();
        for _ in 0..max_tokens {
            let mut logits = self.step_logits(last);
            if !params.logit_bias.is_empty() {
                apply_logit_bias(&mut logits, &params.logit_bias);
            }
            if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
                apply_repetition_penalty(&mut logits, &seen,
                    params.presence_penalty, params.frequency_penalty);
            }
            let id = if params.temperature <= 0.0 {
                argmax(&logits)
            } else {
                sample_from_logits_v2(&mut logits,
                    params.temperature, params.top_p, params.top_k, &mut rng)
            };
            if Some(id) == stop_token { break; }
            *seen.entry(id).or_insert(0) += 1;
            generated.push(id);
            // Stop-string check:  decode just the new piece and append.
            // Tail-match each configured stop string.  Truncation of
            // the matched stop string from the output is conventional.
            if !stop_strings.is_empty() {
                let piece = self.decode_ids(&[id]);
                running_text.push_str(&piece);
                let mut hit: Option<usize> = None;
                for s in stop_strings {
                    if running_text.ends_with(s) {
                        hit = Some(s.len());
                        break;
                    }
                }
                if let Some(slen) = hit {
                    // Strip the matched stop string out of the result so
                    // the client doesn't see it.  Re-encode the trimmed
                    // running_text → ids?  No — the client side decodes
                    // ids and sees the trimmed text;  drop trailing tokens
                    // whose decoded suffix forms the stop string.
                    let _ = slen;
                    // Walk back from the tail until we've removed at least
                    // `slen` characters of decoded text.
                    let target = running_text.len() - slen;
                    let mut cur_text = running_text.clone();
                    while generated.len() > 0 && cur_text.len() > target {
                        generated.pop();
                        // Re-decode from scratch — cheap, small.
                        cur_text = self.decode_ids(&generated);
                    }
                    break;
                }
            }
            last = id;
            if self.next_pos as usize >= MAX_SEQ - 1 { break; }
        }
        generated
    }

    /// FR-x-extra-sampling (legacy signature): forwards to v2 with
    /// stop_strings empty and seed pulled from OS.  Kept so older
    /// call sites in the trainer crate don't need to change at once.
    pub fn generate_sampled(
        &mut self,
        prompt_ids: &[usize],
        max_tokens: usize,
        stop_token: Option<usize>,
        temperature: f32,
        top_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
    ) -> Vec<usize> {
        let params = SamplingParams {
            temperature, top_p, top_k: 0,
            presence_penalty, frequency_penalty,
            seed: None,
            logit_bias: std::collections::HashMap::new(),
        };
        self.generate_sampled_v2(prompt_ids, max_tokens, stop_token, &params, &[])
    }

    /// Encode arbitrary UTF-8 text into token ids using the BPE
    /// tokenizer loaded from the GGUF.  Returns empty vec if the
    /// tokenizer wasn't loaded.
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        if self.bpe_handle < 0 || text.is_empty() { return Vec::new(); }

        // Build (or get cached) byte→token_id table.  This is the
        // GPT-2 byte alphabet: byte 0 → 'Ā' (U+0100), byte 32 → 'Ġ'
        // (U+0120), etc., looked up in the vocab.
        let cache = self.byte_to_id_cache.get_or_init(|| {
            let b2u = build_gpt2_byte_to_unicode_array();
            let mut table = Box::new([-1i32; 256]);
            let mut buf = [0u8; 4];
            for b in 0..256u32 {
                let ch = b2u[b as usize];
                let s = ch.encode_utf8(&mut buf);
                let id = unsafe {
                    aether_bpe_lookup_bytes(
                        self.bpe_handle,
                        s.as_ptr() as *const c_void,
                        s.len() as c_int,
                    )
                };
                table[b as usize] = id;
            }
            table
        });

        // Map each input byte → its surface-char vocab id.
        let bytes = text.as_bytes();
        let mut initial: Vec<i32> = Vec::with_capacity(bytes.len());
        for &b in bytes {
            let id = cache[b as usize];
            if id < 0 {
                // Vocab is missing the surface char for this byte; bail.
                // The caller can fall back to raw byte ids or refuse.
                eprintln!("[encode_text] vocab missing byte {} (surface char {})",
                    b, b);
                return Vec::new();
            }
            initial.push(id);
        }

        // Run the BPE merge loop on the initial surface-id stream.
        let mut out_ids = vec![0i32; initial.len()];
        let n = unsafe {
            aether_bpe_encode_ids(
                self.bpe_handle,
                initial.as_ptr() as *const c_void, initial.len() as c_int,
                out_ids.as_mut_ptr() as *mut c_void, out_ids.len() as c_int,
            )
        };
        if n < 0 { return Vec::new(); }
        out_ids[..n as usize].iter().map(|&i| i as usize).collect()
    }

    // ================================================================
    // FR-19.5-extra-deep — batched-serving slot helpers.
    //
    // These accessors let `crate::batched_serving::BatchScheduler` drive
    // N independent in-flight requests through ONE `QwenSession`.  Each
    // request has its own page_table_host + owned_blocks + next_pos that
    // live in a `SessionSlot` outside this struct.  Per decode tick the
    // scheduler:
    //   1. H2Ds the slot's page_table_host into `paged_cfg.page_table_dev`
    //      (single shared device buffer — the captured graph reads it at
    //      kernel-launch time so the same graph executes correctly with
    //      different slot KV mappings).
    //   2. Sets `self.next_pos` from the slot.
    //   3. Calls `step_logits` (existing single-session path).
    //   4. Reads the new `self.next_pos` back into the slot.
    //
    // No QwenSession internals leak out of the crate — the scheduler
    // works against these methods alone.  Legacy single-session callers
    // (`generate`, `generate_sampled_v2`, etc.) ignore everything below.
    // ================================================================

    /// Block size (tokens per logical block) for the session's paged-KV
    /// layout, or `None` if not in paged mode.  Slots use this to size
    /// their `page_table_host`.
    pub fn paged_block_size(&self) -> Option<i32> {
        self.paged_cfg.as_ref().map(|p| p.block_size)
    }

    /// Number of logical block slots the per-slot `page_table_host`
    /// vector should reserve (sized for `MAX_SEQ` tokens).  Returns 0
    /// when paged mode is off.
    pub fn paged_n_logical(&self) -> usize {
        match &self.paged_cfg {
            Some(p) => ((MAX_SEQ as i32 + p.block_size - 1) / p.block_size) as usize,
            None => 0,
        }
    }

    /// Reference to the model's shared KV pool (for slot block alloc),
    /// or `None` when this session isn't pool-backed.
    pub fn pool_arc(&self) -> Option<std::sync::Arc<SharedKvPool>> {
        self.pool.clone()
    }

    /// Allocate (if not already mapped) a physical block from the shared
    /// pool for logical position `pos`.  Updates the supplied
    /// `page_table_host` + `owned_blocks` in place; no session state
    /// touched.  Returns Err on pool exhaustion.
    pub fn slot_ensure_block(
        &self,
        pos: i32,
        page_table_host: &mut Vec<i32>,
        owned_blocks: &mut Vec<i32>,
    ) -> Result<(), &'static str> {
        let Some(p) = &self.paged_cfg else { return Ok(()); };
        let Some(pool) = &self.pool else { return Ok(()); };
        if pos < 0 { return Err("negative position"); }
        let logical = (pos / p.block_size) as usize;
        if logical < page_table_host.len() && page_table_host[logical] >= 0 {
            return Ok(());
        }
        if logical >= page_table_host.len() {
            page_table_host.resize(logical + 1, -1);
        }
        let b = pool.allocate_block();
        if b < 0 { return Err("pool exhausted"); }
        page_table_host[logical] = b;
        owned_blocks.push(b);
        Ok(())
    }

    /// Return all of `owned_blocks` to the shared pool's free list.
    /// Idempotent; safe to call multiple times.
    pub fn slot_release_blocks(&self, owned_blocks: &mut Vec<i32>) {
        if let Some(pool) = &self.pool {
            for b in owned_blocks.drain(..) {
                pool.free_block(b);
            }
        } else {
            owned_blocks.clear();
        }
    }

    /// Vocab size — exposed so the scheduler can validate prompt_ids and
    /// out-of-range logit_bias entries without locking the session.
    pub fn vocab(&self) -> usize { self.cfg.vocab }

    /// True if the session is configured for paged-KV with a shared
    /// `SharedKvPool` (required for the batched scheduler).
    pub fn is_pool_backed(&self) -> bool {
        self.pool.is_some() && self.paged_cfg.is_some()
    }

    /// MoE archs can't be CUDA-graph-captured (host-side router top-k each
    /// layer).  Scheduler uses this to size single-request critical
    /// sections — MoE batched workers still produce correct output but
    /// can't piggyback on the captured graph.
    pub fn is_moe(&self) -> bool { self.cfg.n_experts > 0 }

    /// Maximum decode position before the slot must stop (matches the
    /// MAX_SEQ - 1 guard inside `generate*`).
    pub fn max_pos(&self) -> i32 { (MAX_SEQ as i32) - 1 }

    /// Decode-step entry point used by the batched scheduler.  Binds the
    /// supplied per-slot state (page_table + position), runs ONE forward
    /// pass, reads `act.logits` back to host, advances `*next_pos` in
    /// the caller-owned variable, and returns the raw logits vector.
    ///
    /// PRECONDITION: `page_table_host[pos / block_size]` is already
    /// mapped (call `slot_ensure_block` first).  Caller holds the
    /// session lock; no other thread may run forward concurrently.
    pub fn step_logits_for_slot(
        &mut self,
        page_table_host: &[i32],
        next_pos: &mut i32,
        last_id: usize,
    ) -> Vec<f32> {
        unsafe {
            // 1. H2D the slot's page-table mapping into the session-wide
            //    device buffer.  The captured graph reads page_table_dev
            //    in-kernel (NOT baked in at capture), so the same graph
            //    correctly fetches this slot's K/V pool blocks.
            if let Some(p) = &self.paged_cfg {
                if !page_table_host.is_empty() {
                    aether_dev_h2d_i32(
                        page_table_host.as_ptr() as i64,
                        p.page_table_dev,
                        page_table_host.len() as c_int,
                    );
                }
            }
            // 2. Bind position state.
            self.next_pos = *next_pos;
            // 3. Run the standard forward (graph or imperative).
            let logits = self.step_logits(last_id);
            // 4. Read advanced position back to the caller's slot.
            *next_pos = self.next_pos;
            logits
        }
    }

    /// FR-19.5-extra-deep Phase 2b-2b — can this model use the batched
    /// decode path?  Covers the STANDARD attention + DENSE FFN shape only:
    /// non-MLA (kv_lora_rank == 0), non-MoE (n_experts == 0), non-flex
    /// (head_dim % 32 == 0 && sliding_window == 0), paged KV.  Everything
    /// else (deepseek2 / glm MLA, qwen3moe, gemma3 flex) stays on the
    /// serial per-slot path.
    pub fn is_batchable(&self) -> bool {
        self.paged_cfg.is_some()
            && self.cfg.kv_lora_rank == 0
            && self.cfg.n_experts == 0
            && (self.cfg.head_dim % 32) == 0
            && self.cfg.sliding_window == 0
    }

    /// Lazily allocate the batched-decode workspace (sized `MAX_BATCH` rows).
    /// Idempotent; only the first call allocates.
    unsafe fn ensure_batch_state(&mut self) {
        if self.batch_state.is_some() { return; }
        let cap = MAX_BATCH as c_int;
        let d_model = self.cfg.d_model as c_int;
        let d_kv = self.cfg.d_kv as c_int;
        let d_ff = self.cfg.d_ff as c_int;
        let q_total = (self.cfg.n_q_heads * self.cfg.head_dim) as c_int;
        let vocab = self.cfg.vocab as c_int;
        let n_logical = self.paged_n_logical() as i32;
        let scratch_in_n = d_model.max(d_ff);
        self.batch_state = Some(BatchActivationGpu {
            x:        aether_dev_alloc_f32(cap * d_model),
            x_norm:   aether_dev_alloc_f32(cap * d_model),
            q:        aether_dev_alloc_f32(cap * q_total),
            k_step:   aether_dev_alloc_f32(cap * d_kv),
            v_step:   aether_dev_alloc_f32(cap * d_kv),
            attn_out: aether_dev_alloc_f32(cap * d_model),
            proj:     aether_dev_alloc_f32(cap * d_model),
            gate:     aether_dev_alloc_f32(cap * d_ff),
            up:       aether_dev_alloc_f32(cap * d_ff),
            down:     aether_dev_alloc_f32(cap * d_model),
            logits:   aether_dev_alloc_f32(cap * vocab),
            pos_batch:        aether_dev_alloc_i32(cap),
            cur_seq_batch:    aether_dev_alloc_i32(cap),
            page_table_batch: aether_dev_alloc_i32(cap * n_logical),
            n_logical,
            scratch_in:  aether_dev_alloc_f32(scratch_in_n),
            scratch_out: aether_dev_alloc_f32(vocab),
        });
    }

    /// FR-19.5-extra-deep Phase 2b-2b — fused batched decode.  Runs ONE
    /// forward pass over `b = page_tables.len()` requests at heterogeneous
    /// decode positions and returns `b` raw logit vectors (each length
    /// `cfg.vocab`).  Advances every `next_positions[i]` by 1.
    ///
    /// This is the e2e throughput win: Q4_K weights are dequantized once per
    /// super-block and applied to all `b` rows (1.9× at b=4), and attention /
    /// RoPE / append run as single per-request hetero launches instead of `b`
    /// serial seq1 steps.
    ///
    /// PRECONDITION (mirrors `step_logits_for_slot`): the caller has already
    /// mapped each slot's block for its current position (`slot_ensure_block`)
    /// so every `page_tables[i]` is valid for `next_positions[i]`.  Caller
    /// holds the session lock.  `is_batchable()` must be true.
    pub fn step_logits_for_batch(
        &mut self,
        page_tables: &[Vec<i32>],
        last_ids: &[usize],
        next_positions: &mut [i32],
    ) -> Vec<Vec<f32>> {
        let b = page_tables.len();
        assert!(b >= 1 && b <= MAX_BATCH,
            "step_logits_for_batch: batch {} out of range 1..={}", b, MAX_BATCH);
        assert_eq!(b, last_ids.len(), "last_ids length mismatch");
        assert_eq!(b, next_positions.len(), "next_positions length mismatch");
        unsafe {
            self.ensure_batch_state();
            let d_model = self.cfg.d_model;
            let vocab = self.cfg.vocab;
            // Snapshot the (Copy) device handles so the dequant loop can take
            // an immutable borrow of self without conflicting.
            let (bx, bpos, bcur, bpt, blogits, b_si, b_so, n_logical) = {
                let ba = self.batch_state.as_ref().unwrap();
                (ba.x, ba.pos_batch, ba.cur_seq_batch, ba.page_table_batch,
                 ba.logits, ba.scratch_in, ba.scratch_out, ba.n_logical as usize)
            };

            // Assemble host-side batched inputs.
            let mut emb_host = vec![0.0f32; b * d_model];
            let mut pos_host = vec![0i32; b];
            let mut cur_seq_host = vec![0i32; b];
            let mut pt_host = vec![-1i32; b * n_logical];
            for i in 0..b {
                let row = self.dequant_embd_row(last_ids[i]);
                emb_host[i * d_model..(i + 1) * d_model].copy_from_slice(&row);
                let pos = next_positions[i];
                pos_host[i] = pos;
                cur_seq_host[i] = pos + 1;
                let pt = &page_tables[i];
                let n = pt.len().min(n_logical);
                pt_host[i * n_logical..i * n_logical + n].copy_from_slice(&pt[..n]);
            }
            aether_dev_h2d_f32_n(emb_host.as_ptr() as i64, bx, (b * d_model) as c_int);
            aether_dev_h2d_i32_n(pos_host.as_ptr() as i64, bpos, b as c_int);
            aether_dev_h2d_i32_n(cur_seq_host.as_ptr() as i64, bcur, b as c_int);
            aether_dev_h2d_i32_n(pt_host.as_ptr() as i64, bpt, (b * n_logical) as c_int);

            // Per-layer batched forward.
            let bc = b as c_int;
            let block_size = self.paged_cfg.as_ref().map(|p| p.block_size).unwrap_or(1);
            for layer in 0..self.cfg.n_layers {
                block_forward_batched(
                    &self.blocks[layer],
                    self.batch_state.as_ref().unwrap(),
                    &self.kvs[layer],
                    bc, &self.cfg, MAX_SEQ, block_size);
            }

            // Final RMSNorm + LM head over all b rows.
            {
                let ba = self.batch_state.as_ref().unwrap();
                aether_op_rms_norm_f32_cuda(
                    ba.x, self.final_norm_g, ba.x_norm,
                    self.cfg.norm_eps, bc, d_model as c_int);
                matmul_batched(
                    ba.x_norm, self.lm_head, self.lm_dt, ba.logits,
                    vocab as c_int, d_model as c_int, bc, b_si, b_so);
            }
            aether_dev_sync();

            // Read back b logit vectors.
            let mut logits_host = vec![0.0f32; b * vocab];
            aether_dev_d2h_f32_n(blogits, logits_host.as_mut_ptr() as i64, (b * vocab) as c_int);
            let mut out = Vec::with_capacity(b);
            for i in 0..b {
                out.push(logits_host[i * vocab..(i + 1) * vocab].to_vec());
            }
            for p in next_positions.iter_mut() { *p += 1; }
            out
        }
    }

    /// Prefill helper for the batched scheduler.  Mirrors `prefill` but
    /// against an external slot state.  Allocates pool blocks as needed
    /// and walks every prompt token through the forward pass.  After
    /// return, `*next_pos = prompt_ids.len() - 1` (matching the legacy
    /// single-session `prefill`).
    pub fn prefill_for_slot(
        &mut self,
        page_table_host: &mut Vec<i32>,
        owned_blocks: &mut Vec<i32>,
        next_pos: &mut i32,
        prompt_ids: &[usize],
    ) -> Result<(), &'static str> {
        if prompt_ids.is_empty() { return Err("prompt cannot be empty"); }
        unsafe {
            for (i, &t_id) in prompt_ids.iter().enumerate() {
                let pos = i as i32;
                self.slot_ensure_block(pos, page_table_host, owned_blocks)?;
                // H2D the (possibly-updated) page table.
                if let Some(p) = &self.paged_cfg {
                    if !page_table_host.is_empty() {
                        aether_dev_h2d_i32(
                            page_table_host.as_ptr() as i64,
                            p.page_table_dev,
                            page_table_host.len() as c_int,
                        );
                    }
                }
                let emb = self.dequant_embd_row(t_id);
                aether_dev_h2d_f32(emb.as_ptr() as i64, self.act.x, self.cfg.d_model as c_int);
                let cur_seq = pos + 1;
                let step_host = [pos, cur_seq, 0i32, 0i32];
                aether_dev_h2d_i32(step_host.as_ptr() as i64, self.step_args, 4);
                for b in 0..self.cfg.n_layers {
                    block_forward_devarg(
                        &self.blocks[b], &self.act, &self.kvs[b],
                        self.step_args, self.paged_arg(), &self.cfg, MAX_SEQ);
                }
            }
            aether_dev_sync();
            *next_pos = (prompt_ids.len() as i32) - 1;
        }
        Ok(())
    }
}

/// External-callable version of the internal greedy argmax — used by
/// the trainer-bin streaming path to mirror the non-streaming sampler
/// behaviour exactly.
pub fn argmax_external(logits: &[f32]) -> usize {
    argmax(logits)
}

/// OpenAI-compat repetition penalties.  Both apply to logits BEFORE
/// the temperature/softmax step (i.e. raw logits).  Semantics match
/// the OpenAI API spec:
///   - `presence_penalty p`:  `logits[t] -= p`  for any `t` in `seen`
///     (the indicator of having appeared, regardless of count).
///   - `frequency_penalty f`: `logits[t] -= f * count[t]`
///     (proportional to how many times `t` has appeared so far).
///
/// Both default to 0.0 (no penalty).  Typical chat values are 0.1..=1.5
/// for either field;  larger numbers more aggressively avoid repeats.
pub fn apply_repetition_penalty(
    logits: &mut [f32],
    seen: &std::collections::HashMap<usize, u32>,
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    for (&id, &count) in seen.iter() {
        if id >= logits.len() { continue; }
        let penalty = presence_penalty + frequency_penalty * (count as f32);
        logits[id] -= penalty;
    }
}

/// External-callable version of `seed_rng` for the trainer-bin
/// streaming path.
pub fn seed_rng_external() -> u64 {
    seed_rng()
}

/// FR-x-extra-sampling: seed an xorshift64 RNG from the OS / time.
/// Not crypto-grade — we just need uncorrelated draws across
/// sessions.  std::time::SystemTime nanos provides enough entropy for
/// next-token sampling.
fn seed_rng() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64).unwrap_or(0xdead_beef_cafe_babe);
    let pid = std::process::id() as u64;
    let mut s = nanos ^ pid.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    if s == 0 { s = 0xdead_beef_cafe_babe; }
    s
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// FR-x-extra-sampling: OpenAI-compat per-call sampling parameters.
/// Bundles the optional-set so callers don't carry 7 f32 args around.
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// `<= 0` disables top-k cutoff.  Typical values 1..=200.
    pub top_k: i32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    /// If `Some`, the per-call xorshift RNG is initialised to this
    /// state — produces a deterministic sample sequence.  `None`
    /// falls back to OS nanos + pid.
    pub seed: Option<u64>,
    /// OpenAI `logit_bias`: `{token_id → bias}`.  Added to raw logits
    /// before any penalty / temperature step.  Special values:
    /// `+100.0` effectively forces the token, `-100.0` effectively
    /// blocks it.
    pub logit_bias: std::collections::HashMap<usize, f32>,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0, top_p: 1.0, top_k: 0,
            presence_penalty: 0.0, frequency_penalty: 0.0,
            seed: None,
            logit_bias: std::collections::HashMap::new(),
        }
    }
}

/// Apply OpenAI-style `logit_bias` map to raw logits.  Out-of-range
/// ids are silently ignored.
pub fn apply_logit_bias(
    logits: &mut [f32],
    bias: &std::collections::HashMap<usize, f32>,
) {
    for (&id, &b) in bias.iter() {
        if id < logits.len() { logits[id] += b; }
    }
}

/// FR-x-extra-sampling v2: like `sample_from_logits` but also supports
/// `top_k` (filter to the K most-probable tokens before nucleus +
/// sample).  Order of operations matches HF's reference sampler:
/// temperature → softmax → top_k → top_p → renormalise → draw.
pub fn sample_from_logits_v2(
    logits: &mut [f32],
    temperature: f32, top_p: f32, top_k: i32,
    rng: &mut u64,
) -> usize {
    // Apply temperature.
    for x in logits.iter_mut() { *x /= temperature; }
    // Numerically-stable softmax.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 0.0 {
        for x in logits.iter_mut() { *x /= sum; }
    }
    // Sort once if we need top_k OR top_p.  Skip if both disabled.
    let need_sort = (top_k > 0 && (top_k as usize) < logits.len())
        || (top_p < 1.0 && top_p > 0.0);
    if need_sort {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|a, b|
            logits[*b].partial_cmp(&logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal));
        // top_k cutoff:  zero everything past rank K-1.
        let k_cutoff = if top_k > 0 { (top_k as usize).min(idx.len()) }
                       else { idx.len() };
        for &i in &idx[k_cutoff..] { logits[i] = 0.0; }
        // top_p cutoff over the surviving top-K window.
        if top_p < 1.0 && top_p > 0.0 {
            let mut cum = 0.0f32;
            let mut p_cutoff = k_cutoff;
            for (rank, &i) in idx[..k_cutoff].iter().enumerate() {
                cum += logits[i];
                if cum >= top_p { p_cutoff = rank + 1; break; }
            }
            for &i in &idx[p_cutoff..k_cutoff] { logits[i] = 0.0; }
        }
        let new_sum: f32 = logits.iter().sum();
        if new_sum > 0.0 {
            for x in logits.iter_mut() { *x /= new_sum; }
        }
    }
    // Draw uniform [0,1) and walk cumulative.
    let r = (xorshift64(rng) as f64 / u64::MAX as f64) as f32;
    let mut cum = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cum += p;
        if cum >= r { return i; }
    }
    logits.len() - 1
}

/// Multinomial sample over `logits` with temperature + top-p (nucleus)
/// cutoff.  Mutates `logits` in place (softmax + zeroing for cutoff).
/// Returns the chosen index.  Caller guarantees temperature > 0.0.
/// Kept for the legacy v1 streaming call site;  new code should use
/// `sample_from_logits_v2`.
pub fn sample_from_logits(
    logits: &mut [f32], temperature: f32, top_p: f32, rng: &mut u64,
) -> usize {
    // Apply temperature.
    for x in logits.iter_mut() { *x /= temperature; }
    // Numerically-stable softmax.
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in logits.iter_mut() { *x = (*x - max).exp(); sum += *x; }
    if sum > 0.0 {
        for x in logits.iter_mut() { *x /= sum; }
    }
    // Top-p nucleus filter.  Sort indices by descending prob, find the
    // cumulative cutoff, zero everything past it, renormalise.
    if top_p < 1.0 && top_p > 0.0 {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|a, b|
            logits[*b].partial_cmp(&logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0f32;
        let mut cutoff = idx.len();
        for (rank, &i) in idx.iter().enumerate() {
            cum += logits[i];
            if cum >= top_p { cutoff = rank + 1; break; }
        }
        // Zero past-cutoff entries by walking idx[cutoff..].
        for &i in &idx[cutoff..] { logits[i] = 0.0; }
        let new_sum: f32 = logits.iter().sum();
        if new_sum > 0.0 {
            for x in logits.iter_mut() { *x /= new_sum; }
        }
    }
    // Draw uniform [0,1) and walk cumulative.
    let r = (xorshift64(rng) as f64 / u64::MAX as f64) as f32;
    let mut cum = 0.0f32;
    for (i, &p) in logits.iter().enumerate() {
        cum += p;
        if cum >= r { return i; }
    }
    logits.len() - 1
}

/// GPT-2 byte→unicode alphabet (the 256-entry version that maps every
/// byte 0..=255 to a single printable unicode codepoint).  Returns a
/// 256-array indexed by byte value.  Identical to the one in
/// `runtime/tests/qwen25_tokenizer_roundtrip.rs`.
fn build_gpt2_byte_to_unicode_array() -> [char; 256] {
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
    let mut tbl = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        tbl[*b as usize] = char::from_u32(*c).unwrap_or('\0');
    }
    tbl
}
