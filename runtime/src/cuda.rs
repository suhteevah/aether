//! CUDA backend for Aether's runtime ABI — Phase 1 of critical-path #25.
//!
//! This file is feature-gated behind `cuda`. When enabled, three new C-ABI
//! symbol families show up in `libaether_rt`:
//!
//!   * **Device memory** — `aether_dev_alloc_f32`, `aether_dev_free_f32`,
//!     `aether_dev_h2d_f32`, `aether_dev_d2h_f32`. Returns / consumes an
//!     `i64` opaque handle (a small slot index into the `BUFFERS` registry,
//!     plus 1 so 0 stays a sentinel "null"). The handle plumbs through
//!     existing aether-emitted code that already passes `i64` pointers
//!     around for buffers — no asm-backend changes needed.
//!
//!   * **Device ops** — `aether_op_matmul_f32_cuda`. Same shape as the CPU
//!     `aether_op_matmul_f32` but its arguments are device handles. Calls
//!     cuBLAS sgemm. cuBLAS uses column-major; we adapt by computing
//!     `out^T = b^T · a^T`, which gives row-major `out = a · b` after the
//!     view transpose — same trick every BLAS-row-major shim uses.
//!
//!   * **Misc** — `aether_dev_init` initialises the global CUDA device +
//!     cuBLAS handle (lazy, fine to call multiple times). `aether_wall_us`
//!     returns a wallclock in microseconds for bench harnesses.
//!
//! The CPU ops in `lib.rs` are untouched; this is additive. A future cut
//! makes `aether_op_matmul_f32` itself dispatch on a backend selector.

use std::cell::UnsafeCell;
use std::os::raw::c_int;
use std::sync::OnceLock;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, LaunchAsync, LaunchConfig};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::nvrtc::compile_ptx;

struct CudaCtx {
    device: Arc<CudaDevice>,
    blas: CudaBlas,
    /// Per-kernel function handles, JIT-compiled once at first init.
    cross_entropy_fwd: CudaFunction,
    cross_entropy_bwd: CudaFunction,
    adamw_step:        CudaFunction,
    add_f32:           CudaFunction,
    gelu_fwd:          CudaFunction,
    gelu_bwd:          CudaFunction,
    layer_norm_fwd:    CudaFunction,
    layer_norm_bwd_dx: CudaFunction,
    layer_norm_bwd_params: CudaFunction,
    softmax_f32:       CudaFunction,
    softmax_bwd:       CudaFunction,
    softmax_bwd_scaled:CudaFunction,
    scale_f32:         CudaFunction,
    gelu_inplace:      CudaFunction,
    add_layer_norm_fwd:CudaFunction,
    // matt-voice deploy: keep entire Qwen forward on device.
    rms_norm_fwd:      CudaFunction,
    rope_apply:        CudaFunction,
    gqa_repeat_kv:     CudaFunction,
    silu_inplace:      CudaFunction,
    mul_inplace:       CudaFunction,
    add_inplace:       CudaFunction,
    bias_add:          CudaFunction,
    dequant_q4_k_m_gpu:CudaFunction,
    dequant_q6_k_gpu:  CudaFunction,
    fused_q4k_matmul_seq1: CudaFunction,
    fused_q4_0_matmul_seq1: CudaFunction,
    fused_q5_0_matmul_seq1: CudaFunction,
    fused_q8_0_matmul_seq1: CudaFunction,
    fused_q5_k_matmul_seq1: CudaFunction,
    fused_q3_k_matmul_seq1: CudaFunction,
    fused_iq4_nl_matmul_seq1: CudaFunction,
    fused_iq4_xs_matmul_seq1: CudaFunction,
    fused_iq3_xxs_matmul_seq1: CudaFunction,
    fused_iq3_s_matmul_seq1: CudaFunction,
    fused_q4k_matmul_seq1_v2: CudaFunction,
    fused_q6k_matmul_seq1_v2: CudaFunction,
    fused_q4k_ffn_gate_up_silu_mul: CudaFunction,
    fused_q4k_matmul_seq1_v3: CudaFunction,
    fused_q4k_ffn_gate_up_silu_mul_v2: CudaFunction,
    rope_apply_devarg: CudaFunction,
    append_kv_devarg: CudaFunction,
    attention_seq1_devarg: CudaFunction,
    append_kv: CudaFunction,
    attention_seq1: CudaFunction,
    fused_f16_matmul_seq1: CudaFunction,
    fused_f32_matmul_seq1: CudaFunction,
    fused_q4k_expert_matmul_seq1: CudaFunction,
    fused_q8_0_expert_matmul_seq1: CudaFunction,
    fused_q5_0_expert_matmul_seq1: CudaFunction,
    fused_iq3_s_expert_matmul_seq1: CudaFunction,
    fused_iq4_xs_expert_matmul_seq1: CudaFunction,
    fused_iq3_xxs_expert_matmul_seq1: CudaFunction,
    bert_self_attention_fwd: CudaFunction,
    bert_embed_sum: CudaFunction,
}

/// FR-19.4-extra paged-KV kernels — held in a separate `OnceCell` so the
/// PTX/load only happens when paged kernels are first invoked.  This keeps
/// the contiguous-decode hot path untouched: the qwen25_graph_decode bench
/// regressed from 37 → 31 tok/s when paged compilation ran inside the main
/// `ctx()` init, even though the paged kernels were never called by that
/// bench (cross-module device-side pressure per memory:
/// nvrtc_kernel_unit_pressure.md).  Lazy-init restores baseline.
struct PagedCtx {
    paged_append_kv_devarg: CudaFunction,
    paged_attention_seq1_devarg: CudaFunction,
    batched_paged_attention_seqB_devarg: CudaFunction,
    batched_paged_append_kv_seqB_devarg: CudaFunction,
    batched_paged_attention_hetero_devarg: CudaFunction,
    batched_paged_append_kv_hetero_devarg: CudaFunction,
    fused_q4k_matmul_seqB_v3: CudaFunction,
    paged_attention_flex_devarg: CudaFunction,
    paged_attention_mla_devarg: CudaFunction,
    paged_append_kv_mla_devarg: CudaFunction,
    mla_split_kv_a: CudaFunction,
    mla_assemble_k: CudaFunction,
    mla_extract_v: CudaFunction,
    mla_rope_q_partial: CudaFunction,
    mla_rope_k_shared: CudaFunction,
    mla_rope_q_partial_yarn: CudaFunction,
    mla_rope_k_shared_yarn: CudaFunction,
    mla_absorb_q_q8_0: CudaFunction,
    mla_absorb_v_q8_0: CudaFunction,
    mla_broadcast_kv_for_mqa: CudaFunction,
    // FR-17-extra-mla-absorbed-dtypes — extra dtype dispatch for w_k_b / w_v_b.
    mla_absorb_q_f16: CudaFunction,
    mla_absorb_v_f16: CudaFunction,
    mla_absorb_q_q4_k: CudaFunction,
    mla_absorb_v_q4_k: CudaFunction,
    mla_absorb_q_q5_k: CudaFunction,
    mla_absorb_v_q5_k: CudaFunction,
    mla_absorb_q_q6_k: CudaFunction,
    mla_absorb_v_q6_k: CudaFunction,
    mla_absorb_q_iq4_nl: CudaFunction,
    mla_absorb_v_iq4_nl: CudaFunction,
}

static PAGED_CTX: OnceLock<PagedCtx> = OnceLock::new();

fn paged_ctx() -> &'static PagedCtx {
    PAGED_CTX.get_or_init(|| {
        let device = &ctx().device;
        let paged_ptx = compile_ptx(PAGED_KERNEL_SRC).expect("compile_ptx paged");
        device.load_ptx(paged_ptx, "aether_paged_kernels",
            &["paged_append_kv_devarg", "paged_attention_seq1_devarg",
              "batched_paged_attention_seqB_devarg",
              "batched_paged_append_kv_seqB_devarg",
              "batched_paged_attention_hetero_devarg",
              "batched_paged_append_kv_hetero_devarg",
              "fused_q4k_matmul_seqB_v3",
              "paged_attention_flex_devarg",
              "paged_attention_mla_devarg",
              "paged_append_kv_mla_devarg",
              "mla_split_kv_a",
              "mla_assemble_k",
              "mla_extract_v",
              "mla_rope_q_partial",
              "mla_rope_k_shared",
              "mla_rope_q_partial_yarn",
              "mla_rope_k_shared_yarn",
              "mla_absorb_q_q8_0",
              "mla_absorb_v_q8_0",
              "mla_broadcast_kv_for_mqa",
              "mla_absorb_q_f16",
              "mla_absorb_v_f16",
              "mla_absorb_q_q4_k",
              "mla_absorb_v_q4_k",
              "mla_absorb_q_q5_k",
              "mla_absorb_v_q5_k",
              "mla_absorb_q_q6_k",
              "mla_absorb_v_q6_k",
              "mla_absorb_q_iq4_nl",
              "mla_absorb_v_iq4_nl"])
            .expect("load_ptx paged");
        PagedCtx {
            paged_append_kv_devarg:
                device.get_func("aether_paged_kernels", "paged_append_kv_devarg").unwrap(),
            paged_attention_seq1_devarg:
                device.get_func("aether_paged_kernels", "paged_attention_seq1_devarg").unwrap(),
            batched_paged_attention_seqB_devarg:
                device.get_func("aether_paged_kernels", "batched_paged_attention_seqB_devarg").unwrap(),
            batched_paged_append_kv_seqB_devarg:
                device.get_func("aether_paged_kernels", "batched_paged_append_kv_seqB_devarg").unwrap(),
            batched_paged_attention_hetero_devarg:
                device.get_func("aether_paged_kernels", "batched_paged_attention_hetero_devarg").unwrap(),
            batched_paged_append_kv_hetero_devarg:
                device.get_func("aether_paged_kernels", "batched_paged_append_kv_hetero_devarg").unwrap(),
            fused_q4k_matmul_seqB_v3:
                device.get_func("aether_paged_kernels", "fused_q4k_matmul_seqB_v3").unwrap(),
            paged_attention_flex_devarg:
                device.get_func("aether_paged_kernels", "paged_attention_flex_devarg").unwrap(),
            paged_attention_mla_devarg:
                device.get_func("aether_paged_kernels", "paged_attention_mla_devarg").unwrap(),
            paged_append_kv_mla_devarg:
                device.get_func("aether_paged_kernels", "paged_append_kv_mla_devarg").unwrap(),
            mla_split_kv_a:
                device.get_func("aether_paged_kernels", "mla_split_kv_a").unwrap(),
            mla_assemble_k:
                device.get_func("aether_paged_kernels", "mla_assemble_k").unwrap(),
            mla_extract_v:
                device.get_func("aether_paged_kernels", "mla_extract_v").unwrap(),
            mla_rope_q_partial:
                device.get_func("aether_paged_kernels", "mla_rope_q_partial").unwrap(),
            mla_rope_k_shared:
                device.get_func("aether_paged_kernels", "mla_rope_k_shared").unwrap(),
            mla_rope_q_partial_yarn:
                device.get_func("aether_paged_kernels", "mla_rope_q_partial_yarn").unwrap(),
            mla_rope_k_shared_yarn:
                device.get_func("aether_paged_kernels", "mla_rope_k_shared_yarn").unwrap(),
            mla_absorb_q_q8_0:
                device.get_func("aether_paged_kernels", "mla_absorb_q_q8_0").unwrap(),
            mla_absorb_v_q8_0:
                device.get_func("aether_paged_kernels", "mla_absorb_v_q8_0").unwrap(),
            mla_broadcast_kv_for_mqa:
                device.get_func("aether_paged_kernels", "mla_broadcast_kv_for_mqa").unwrap(),
            mla_absorb_q_f16:
                device.get_func("aether_paged_kernels", "mla_absorb_q_f16").unwrap(),
            mla_absorb_v_f16:
                device.get_func("aether_paged_kernels", "mla_absorb_v_f16").unwrap(),
            mla_absorb_q_q4_k:
                device.get_func("aether_paged_kernels", "mla_absorb_q_q4_k").unwrap(),
            mla_absorb_v_q4_k:
                device.get_func("aether_paged_kernels", "mla_absorb_v_q4_k").unwrap(),
            mla_absorb_q_q5_k:
                device.get_func("aether_paged_kernels", "mla_absorb_q_q5_k").unwrap(),
            mla_absorb_v_q5_k:
                device.get_func("aether_paged_kernels", "mla_absorb_v_q5_k").unwrap(),
            mla_absorb_q_q6_k:
                device.get_func("aether_paged_kernels", "mla_absorb_q_q6_k").unwrap(),
            mla_absorb_v_q6_k:
                device.get_func("aether_paged_kernels", "mla_absorb_v_q6_k").unwrap(),
            mla_absorb_q_iq4_nl:
                device.get_func("aether_paged_kernels", "mla_absorb_q_iq4_nl").unwrap(),
            mla_absorb_v_iq4_nl:
                device.get_func("aether_paged_kernels", "mla_absorb_v_iq4_nl").unwrap(),
        }
    })
}

/// FR-19.4-extra paged-KV kernels.  Separate nvrtc unit so its register
/// allocation doesn't perturb the main KERNEL_SRC kernels.
///
/// Pool memory layout (per layer × {K, V}): contiguous f32 of size
/// `pool_n_blocks * block_size * d_kv`.  Block id `b` lives at offset
/// `b * block_size * d_kv`.  Within a block, position `p` of `d_kv`
/// elements lives at `b * block_size * d_kv + p * d_kv`.
///
/// Per-request page_table: `int[]` where `page_table[L]` = physical block
/// id holding logical block `L` of the request.  Token at request-relative
/// position `pos` ↔ physical row `page_table[pos / block_size] * block_size
/// + (pos % block_size)`.
const PAGED_KERNEL_SRC: &str = r#"
// Device-side F16→F32 conversion (copied from KERNEL_SRC since this is a
// separate nvrtc compilation unit and cross-PTX symbol resolution isn't set
// up).  Used by the absorbed-MLA Q8_0 dequant kernels below.
extern "C" __device__ float aether_f16_to_f32_dev(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
        if (mant == 0u) { bits = sign << 31; return __int_as_float(bits); }
        unsigned int m = mant;
        int e = -14;
        while ((m & 0x0400u) == 0u) { m <<= 1; e -= 1; }
        m &= 0x03FFu;
        bits = (sign << 31) | ((unsigned int)(e + 127) << 23) | (m << 13);
        return __int_as_float(bits);
    }
    if (exp == 0x1Fu) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
        return __int_as_float(bits);
    }
    unsigned int f32_exp  = (exp - 15u + 127u) << 23;
    unsigned int f32_mant = mant << 13;
    bits = (sign << 31) | f32_exp | f32_mant;
    return __int_as_float(bits);
}

// FR-19.5-extra-deep — batched paged append_kv. B (k_new, v_new) pairs at
// position step_args[0] of B independent page tables, all writing into the
// same shared pool.  Grid (d_kv, B).
extern "C" __global__ void batched_paged_append_kv_seqB_devarg(
    const float* __restrict__ k_new_batch,    // [B * d_kv]
    const float* __restrict__ v_new_batch,    // [B * d_kv]
    float*       __restrict__ k_pool,
    float*       __restrict__ v_pool,
    const int*   __restrict__ page_table_batch,
    int d_kv, int block_size, int page_table_stride,
    const int* __restrict__ step_args)
{
    int pos = step_args[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int req = blockIdx.y;
    if (tid >= d_kv) return;
    int logical_blk = pos / block_size;
    int in_blk_pos  = pos - logical_blk * block_size;
    int phys_blk    = page_table_batch[req * page_table_stride + logical_blk];
    size_t row = (size_t)phys_blk * block_size + in_blk_pos;
    k_pool[row * d_kv + tid] = k_new_batch[req * d_kv + tid];
    v_pool[row * d_kv + tid] = v_new_batch[req * d_kv + tid];
}

// FR-17-extra-gemma-fwd — flexible attention: handles head_dim that's NOT a
// multiple of 32 (e.g. Gemma3's head_dim=168) AND optional sliding window
// scope.  `sliding_window`: when > 0, restrict t to
// [max(0, cur_seq - sliding_window), cur_seq).  When <= 0, full attention
// (same as paged_attention_seq1_devarg).
extern "C" __global__ void paged_attention_flex_devarg(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int*   __restrict__ page_table,
    float*       __restrict__ attn_out,
    int n_q_heads, int n_kv_heads, int head_dim, int block_size,
    int sliding_window,
    float scale, const int* __restrict__ step_args)
{
    int cur_seq = step_args[1];
    int t_lo = (sliding_window > 0 && cur_seq > sliding_window) ? cur_seq - sliding_window : 0;
    extern __shared__ float scores[];

    int head    = blockIdx.x;
    int lane    = threadIdx.x;
    int kv_per_q = n_q_heads / n_kv_heads;
    int kv_head = head / kv_per_q;
    int d_kv    = n_kv_heads * head_dim;
    // per_lane = ceil(head_dim / 32) so head_dim NOT a multiple of 32
    // (e.g. Gemma3 head_dim=168) still works.  Bounds-checked per element.
    int per_lane = (head_dim + 31) >> 5;

    const float* q_ptr = q + head * head_dim;
    float q_local[8];   // up to head_dim=256 with per_lane=8
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = lane * per_lane + i;
        q_local[i] = (i < per_lane && col < head_dim) ? q_ptr[col] : 0.0f;
    }

    // Pass 1: scores[t] = Q · K[t, kv_head] * scale
    int n_active = cur_seq - t_lo;
    for (int t = t_lo; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = page_table[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        const float* k_ptr = k_pool + row * d_kv + kv_head * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = lane * per_lane + i;
            if (i < per_lane && col < head_dim) acc += q_local[i] * k_ptr[col];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t - t_lo] = acc * scale;
    }
    __syncwarp();

    // Pass 2: softmax over the active window.
    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < n_active; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < n_active; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < n_active; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // Pass 3: aggregate V over the active window.
    float out_local[8] = {0.0f};
    for (int t = t_lo; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = page_table[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        float w = scores[t - t_lo];
        const float* v_ptr = v_pool + row * d_kv + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = lane * per_lane + i;
            if (i < per_lane && col < head_dim) out_local[i] += w * v_ptr[col];
        }
    }
    float* out_ptr = attn_out + head * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = lane * per_lane + i;
        if (i < per_lane && col < head_dim) out_ptr[col] = out_local[i];
    }
}

// FR-19.5-extra-deep — batched paged attention. B queries × B page tables
// in ONE launch.  Grid layout: blockIdx.x = head, blockIdx.y = request_idx.
// All B requests share k_pool / v_pool but each indexes via its own row
// of page_table_batch (row stride = page_table_stride int32 entries).
// Step args layout: [pos_global_unused, cur_seq_per_request, _, _] (we use
// cur_seq from step_args[1] for every request — they're all at the same
// decode position in a synchronous batched step).
extern "C" __global__ void batched_paged_attention_seqB_devarg(
    const float* __restrict__ q_batch,            // [B * n_q_heads * head_dim]
    const float* __restrict__ k_pool,             // shared pool
    const float* __restrict__ v_pool,             // shared pool
    const int*   __restrict__ page_table_batch,   // [B * page_table_stride]
    float*       __restrict__ attn_out_batch,     // [B * n_q_heads * head_dim]
    int n_q_heads, int n_kv_heads, int head_dim, int block_size,
    int page_table_stride,
    float scale, const int* __restrict__ step_args)
{
    int cur_seq = step_args[1];
    extern __shared__ float scores[];

    int head    = blockIdx.x;
    int req     = blockIdx.y;
    int lane    = threadIdx.x;
    int kv_per_q = n_q_heads / n_kv_heads;
    int kv_head = head / kv_per_q;
    int d_kv    = n_kv_heads * head_dim;
    int per_lane = head_dim >> 5;

    const float* q_ptr = q_batch + (req * n_q_heads + head) * head_dim;
    const int*   pt    = page_table_batch + req * page_table_stride;

    float q_local[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) q_local[i] = q_ptr[lane * per_lane + i];
    }

    // Pass 1: scores[t] = Q · K[t, kv_head] * scale
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = pt[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        const float* k_ptr = k_pool + row * d_kv + kv_head * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) acc += q_local[i] * k_ptr[lane * per_lane + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    // Pass 2: softmax
    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < cur_seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < cur_seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < cur_seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // Pass 3: aggregate V by softmax weights
    float out_local[8] = {0.0f};
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = pt[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        float w = scores[t];
        const float* v_ptr = v_pool + row * d_kv + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) out_local[i] += w * v_ptr[lane * per_lane + i];
        }
    }
    float* out_ptr = attn_out_batch + (req * n_q_heads + head) * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) out_ptr[lane * per_lane + i] = out_local[i];
    }
}

// FR-19.5-extra-deep Phase 2 — HETEROGENEOUS-position batched append_kv.
// Identical to batched_paged_append_kv_seqB_devarg except each request
// writes its new K/V at its OWN position `pos_batch[req]` (instead of a
// single shared step_args[0]).  This is what lets the continuous-
// batching scheduler fuse N slots that are at different decode positions
// into ONE launch.  Grid (ceil(d_kv/threads), B).
extern "C" __global__ void batched_paged_append_kv_hetero_devarg(
    const float* __restrict__ k_new_batch,    // [B * d_kv]
    const float* __restrict__ v_new_batch,    // [B * d_kv]
    float*       __restrict__ k_pool,
    float*       __restrict__ v_pool,
    const int*   __restrict__ page_table_batch,
    int d_kv, int block_size, int page_table_stride,
    const int* __restrict__ pos_batch)        // [B] — per-request position
{
    int req = blockIdx.y;
    int pos = pos_batch[req];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_kv) return;
    int logical_blk = pos / block_size;
    int in_blk_pos  = pos - logical_blk * block_size;
    int phys_blk    = page_table_batch[req * page_table_stride + logical_blk];
    size_t row = (size_t)phys_blk * block_size + in_blk_pos;
    k_pool[row * d_kv + tid] = k_new_batch[req * d_kv + tid];
    v_pool[row * d_kv + tid] = v_new_batch[req * d_kv + tid];
}

// FR-19.5-extra-deep Phase 2 — HETEROGENEOUS-position batched attention.
// Identical to batched_paged_attention_seqB_devarg except each request
// attends over its OWN window [0, cur_seq_batch[req]) rather than a
// single shared step_args[1].  Block (head, req); one warp per (head,
// req).  Shared `scores[]` is launch-sized for the MAX cur_seq across
// the batch; each block uses only its own request's prefix.
extern "C" __global__ void batched_paged_attention_hetero_devarg(
    const float* __restrict__ q_batch,            // [B * n_q_heads * head_dim]
    const float* __restrict__ k_pool,             // shared pool
    const float* __restrict__ v_pool,             // shared pool
    const int*   __restrict__ page_table_batch,   // [B * page_table_stride]
    float*       __restrict__ attn_out_batch,     // [B * n_q_heads * head_dim]
    int n_q_heads, int n_kv_heads, int head_dim, int block_size,
    int page_table_stride,
    float scale, const int* __restrict__ cur_seq_batch)  // [B]
{
    int req     = blockIdx.y;
    int cur_seq = cur_seq_batch[req];
    extern __shared__ float scores[];

    int head    = blockIdx.x;
    int lane    = threadIdx.x;
    int kv_per_q = n_q_heads / n_kv_heads;
    int kv_head = head / kv_per_q;
    int d_kv    = n_kv_heads * head_dim;
    int per_lane = head_dim >> 5;

    const float* q_ptr = q_batch + (req * n_q_heads + head) * head_dim;
    const int*   pt    = page_table_batch + req * page_table_stride;

    float q_local[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) q_local[i] = q_ptr[lane * per_lane + i];
    }

    // Pass 1: scores[t] = Q · K[t, kv_head] * scale
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = pt[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        const float* k_ptr = k_pool + row * d_kv + kv_head * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) acc += q_local[i] * k_ptr[lane * per_lane + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    // Pass 2: softmax over [0, cur_seq).
    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < cur_seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < cur_seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < cur_seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // Pass 3: aggregate V by softmax weights.
    float out_local[8] = {0.0f};
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = pt[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        float w = scores[t];
        const float* v_ptr = v_pool + row * d_kv + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) out_local[i] += w * v_ptr[lane * per_lane + i];
        }
    }
    float* out_ptr = attn_out_batch + (req * n_q_heads + head) * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) out_ptr[lane * per_lane + i] = out_local[i];
    }
}

// FR-19.5-extra-deep Phase 2 — WEIGHT-REUSE batched Q4_K matmul.
//
// The decode throughput lever for continuous batching: at batch=1 the
// seq1 matmul is DRAM-bandwidth-bound on the Q4_K weights (each output
// row reads its whole 144-byte-per-block weight strip once for only 256
// FMAs/block against it).  This kernel dequantizes each weight block
// EXACTLY ONCE (byte-once layout identical to KERNEL_SRC's seq1_v3) and
// applies it to `batch` activation rows held in shared memory — weight
// memory traffic unchanged while FMA throughput scales with batch.
//
// LIVES IN PAGED_KERNEL_SRC (a SEPARATE, lazily-loaded nvrtc module) on
// purpose: adding it to KERNEL_SRC regressed the single-stream seq1
// decode path ~7% via nvrtc register-allocation pressure (see
// BENCH_LEDGER 02fca19 → fix; the same class as f40d259).  Here it never
// touches the main decode ctx() and only compiles when paged/batched
// kernels are first used.  The q4k_get_scale/min helpers are duplicated
// below (same reason aether_f16_to_f32_dev is duplicated in this module).
//
// a    : [batch * n_blocks*256]   (row-major; row b at b*n_blocks*256)
// w    : [n * n_blocks * 144]     (same Q4_K weight layout as seq1)
// out  : [batch * n]              (row-major; row b at b*n)
// FMA order per (b, ni) is bit-identical to seq1_v3 → exact parity.
extern "C" __device__ unsigned int q4k_get_scale_paged(int sub, const unsigned char* sc) {
    if (sub < 4) return sc[sub] & 63u;
    return (sc[sub + 4] & 0xFu) | (((unsigned int)(sc[sub - 4] >> 6)) << 4);
}
extern "C" __device__ unsigned int q4k_get_min_paged(int sub, const unsigned char* sc) {
    if (sub < 4) return sc[sub + 4] & 63u;
    return (sc[sub + 4] >> 4) | (((unsigned int)(sc[sub] >> 6)) << 4);
}
extern "C" __global__ void fused_q4k_matmul_seqB_v3(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks, int batch)   // batch in [1, 8]
{
    __shared__ float a_tile[8 * 256];   // up to batch=8 rows × 256-elem K-tile

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int ni = blockIdx.x * 8 + warp;

    // Split lo/hi accumulators per row to expose ILP (matches seq1_v3).
    float acc_lo[8]; float acc_hi[8];
    #pragma unroll
    for (int b = 0; b < 8; b++) { acc_lo[b] = 0.0f; acc_hi[b] = 0.0f; }

    for (int bi = 0; bi < n_blocks; bi++) {
        // CTA-wide cooperative load of `batch` activation tiles.
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            if (b < batch) {
                a_tile[b * 256 + threadIdx.x] =
                    a[(size_t)b * n_blocks * 256 + bi * 256 + threadIdx.x];
            }
        }
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w
                + (size_t)ni * n_blocks * 144
                + (size_t)bi * 144;
            unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* scales = base + 4;
            const unsigned char* qs     = base + 16;

            // Dequant scale/min computed ONCE per (ni, bi) tile, reused
            // across all `batch` activation rows.
            float d_eff[8], m_eff[8];
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                unsigned int sc = q4k_get_scale_paged(s, scales);
                unsigned int mn = q4k_get_min_paged(s, scales);
                d_eff[s] = d * (float)sc;
                m_eff[s] = dmin * (float)mn;
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int sub_lo = i * 2;
                int sub_hi = i * 2 + 1;
                unsigned char byte = qs[i * 32 + lane];        // weight byte: read ONCE
                unsigned int nib_lo = ((unsigned int)byte) & 0xFu;
                unsigned int nib_hi = (((unsigned int)byte) >> 4) & 0xFu;
                float w_lo = d_eff[sub_lo] * (float)nib_lo - m_eff[sub_lo];
                float w_hi = d_eff[sub_hi] * (float)nib_hi - m_eff[sub_hi];
                int k_lo = sub_lo * 32 + lane;
                int k_hi = sub_hi * 32 + lane;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    if (b < batch) {
                        acc_lo[b] += a_tile[b * 256 + k_lo] * w_lo;
                        acc_hi[b] += a_tile[b * 256 + k_hi] * w_hi;
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int b = 0; b < 8; b++) {
        if (b < batch) {
            float acc = acc_lo[b] + acc_hi[b];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
            }
            if (lane == 0 && ni < n) {
                out[(size_t)b * n + ni] = acc;
            }
        }
    }
}

extern "C" __global__ void paged_append_kv_devarg(
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    float*       __restrict__ k_pool,
    float*       __restrict__ v_pool,
    const int*   __restrict__ page_table,
    int          d_kv,
    int          block_size,
    const int*   __restrict__ step_args)
{
    int pos = step_args[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_kv) return;
    int logical_blk = pos / block_size;
    int in_blk_pos  = pos - logical_blk * block_size;
    int phys_blk    = page_table[logical_blk];
    size_t row = (size_t)phys_blk * block_size + in_blk_pos;
    k_pool[row * d_kv + tid] = k_new[tid];
    v_pool[row * d_kv + tid] = v_new[tid];
}

// FR-17-extra-mla-fwd — paged append with INDEPENDENT K and V row strides.
// K row stride = d_k_row (= n_heads * qk_head_dim);
// V row stride = d_v_row (= n_heads * v_head_dim).
// Launch grid spans max(d_k_row, d_v_row); each thread writes one element of
// whichever buffer is in range.
extern "C" __global__ void paged_append_kv_mla_devarg(
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    float*       __restrict__ k_pool,
    float*       __restrict__ v_pool,
    const int*   __restrict__ page_table,
    int          d_k_row,
    int          d_v_row,
    int          block_size,
    const int*   __restrict__ step_args)
{
    int pos = step_args[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int logical_blk = pos / block_size;
    int in_blk_pos  = pos - logical_blk * block_size;
    int phys_blk    = page_table[logical_blk];
    size_t row = (size_t)phys_blk * block_size + in_blk_pos;
    if (tid < d_k_row) {
        k_pool[row * d_k_row + tid] = k_new[tid];
    }
    if (tid < d_v_row) {
        v_pool[row * d_v_row + tid] = v_new[tid];
    }
}

extern "C" __global__ void paged_attention_seq1_devarg(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int*   __restrict__ page_table,
    float*       __restrict__ attn_out,
    int n_q_heads, int n_kv_heads, int head_dim, int block_size,
    float scale, const int* __restrict__ step_args)
{
    int cur_seq = step_args[1];
    extern __shared__ float scores[];

    int head    = blockIdx.x;
    int lane    = threadIdx.x;
    int kv_per_q = n_q_heads / n_kv_heads;
    int kv_head = head / kv_per_q;
    int d_kv    = n_kv_heads * head_dim;
    int per_lane = head_dim >> 5;

    const float* q_ptr = q + head * head_dim;
    float q_local[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) q_local[i] = q_ptr[lane * per_lane + i];
    }

    // Pass 1: scores[t] = Q · K[t, kv_head] * scale, with K from paged pool
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = page_table[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        const float* k_ptr = k_pool + row * d_kv + kv_head * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) acc += q_local[i] * k_ptr[lane * per_lane + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    // Pass 2: softmax (max, exp+sum, normalize)
    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < cur_seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < cur_seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < cur_seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // Pass 3: aggregate V by softmax weights, V from paged pool
    float out_local[8] = {0.0f};
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = page_table[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        float w = scores[t];
        const float* v_ptr = v_pool + row * d_kv + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) out_local[i] += w * v_ptr[lane * per_lane + i];
        }
    }
    float* out_ptr = attn_out + head * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) out_ptr[lane * per_lane + i] = out_local[i];
    }
}

// FR-17-extra-mla-fwd — DeepSeek-V2 Multi-head Latent Attention.
//
// MLA differs from standard GQA/MQA in two ways relevant to the attention
// kernel itself:
//   (1) Q's per-head dim (`qk_head_dim` = qk_nope + qk_rope, e.g. 192) and
//       V's per-head dim (`v_head_dim`, e.g. 128) are DIFFERENT.
//   (2) Every head shares the same projected K and V (n_kv_heads == n_heads);
//       there is no GQA replication.  In practice the latent-decompression
//       path produces a per-head K and V from a small latent c_kv, but for
//       the attention kernel itself we treat n_heads = n_kv_heads.
//
// Cache layout (in the per-layer K / V pool):
//   K row per token = `n_heads * qk_head_dim` f32 (per-head K = [K_nope | K_rope])
//   V row per token = `n_heads * v_head_dim`  f32
// Both pools are paged with `block_size` tokens per physical block.
//
// Threading: 32-lane warp per head, max qk_head_dim = 256.  per_lane uses
// CEIL so qk_head_dim that's not a multiple of 32 (e.g. 192 ÷ 32 = 6) still
// works.  Output is v_head_dim wide; we use a SECOND per_lane index for V.
extern "C" __global__ void paged_attention_mla_devarg(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const int*   __restrict__ page_table,
    float*       __restrict__ attn_out,
    int n_heads, int qk_head_dim, int v_head_dim, int block_size,
    float scale, const int* __restrict__ step_args)
{
    int cur_seq = step_args[1];
    extern __shared__ float scores[];

    int head = blockIdx.x;
    int lane = threadIdx.x;
    int d_k_row = n_heads * qk_head_dim;   // K row stride per token
    int d_v_row = n_heads * v_head_dim;    // V row stride per token

    // MLA per-lane register-array bound: 20 covers GLM-4.7-flash's
    // qk_head_dim=576 (per_lane=18) and v_head_dim=512 (per_lane=16) with
    // margin.  Bumping from 8 lifts the 256-head-dim cap to 640.
    int per_lane_k = (qk_head_dim + 31) >> 5;
    int per_lane_v = (v_head_dim  + 31) >> 5;

    // Load Q for this head into thread-local registers.
    const float* q_ptr = q + head * qk_head_dim;
    float q_local[20];
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        int col = lane * per_lane_k + i;
        q_local[i] = (i < per_lane_k && col < qk_head_dim) ? q_ptr[col] : 0.0f;
    }

    // Pass 1: scores[t] = Q · K[t, head] * scale.
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = page_table[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        const float* k_ptr = k_pool + row * d_k_row + head * qk_head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 20; i++) {
            int col = lane * per_lane_k + i;
            if (i < per_lane_k && col < qk_head_dim) acc += q_local[i] * k_ptr[col];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    // Pass 2: softmax over [0, cur_seq).
    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < cur_seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < cur_seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < cur_seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // Pass 3: aggregate V over [0, cur_seq).  V has a DIFFERENT per-head dim
    // and a DIFFERENT row stride than K.
    float out_local[20] = {0.0f};
    for (int t = 0; t < cur_seq; t++) {
        int logical_blk = t / block_size;
        int in_blk_pos  = t - logical_blk * block_size;
        int phys_blk    = page_table[logical_blk];
        size_t row = (size_t)phys_blk * block_size + in_blk_pos;
        float w = scores[t];
        const float* v_ptr = v_pool + row * d_v_row + head * v_head_dim;
        #pragma unroll
        for (int i = 0; i < 20; i++) {
            int col = lane * per_lane_v + i;
            if (i < per_lane_v && col < v_head_dim) out_local[i] += w * v_ptr[col];
        }
    }
    float* out_ptr = attn_out + head * v_head_dim;
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        int col = lane * per_lane_v + i;
        if (i < per_lane_v && col < v_head_dim) out_ptr[col] = out_local[i];
    }
}

// FR-17-extra-mla-fwd — glue kernels for the MLA forward path.
//
// Step 1: split the kv_a_mqa output into the latent c_kv and the shared
// k_rope vector.  kv_a_mqa produces [kv_lora_rank + qk_rope_head_dim] per
// token; we just need contiguous slices, so this is a memcpy fan-out.
extern "C" __global__ void mla_split_kv_a(
    const float* __restrict__ kv_a,   // [kv_lora_rank + qk_rope_head_dim]
    float*       __restrict__ c_kv,   // [kv_lora_rank]
    float*       __restrict__ k_rope, // [qk_rope_head_dim]
    int kv_lora_rank, int qk_rope_head_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = kv_lora_rank + qk_rope_head_dim;
    if (i >= total) return;
    if (i < kv_lora_rank) c_kv[i] = kv_a[i];
    else k_rope[i - kv_lora_rank] = kv_a[i];
}

// Step 2: assemble the per-head K row from K_nope (per-head, taken from
// the first qk_nope_head_dim columns of each per-head kv_b slice) and the
// shared k_rope vector broadcast across all heads.
//   kv_b_out layout: [n_heads * (qk_nope_head_dim + v_head_dim)] where each
//   per-head chunk is [K_nope (qk_nope_head_dim) | V (v_head_dim)].
//   k_row layout:    [n_heads * qk_head_dim] where qk_head_dim =
//   qk_nope_head_dim + qk_rope_head_dim and each per-head chunk is
//   [K_nope | k_rope_shared].
// Grid spans (n_heads × qk_head_dim) so each thread writes one f32.
extern "C" __global__ void mla_assemble_k(
    const float* __restrict__ kv_b_out,    // [n_heads * (qk_nope+v_head)]
    const float* __restrict__ k_rope,      // [qk_rope_head_dim]
    float*       __restrict__ k_row,       // [n_heads * qk_head_dim]
    int n_heads, int qk_nope_head_dim,
    int qk_rope_head_dim, int v_head_dim)
{
    int qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    int kv_b_stride = qk_nope_head_dim + v_head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * qk_head_dim;
    if (idx >= total) return;
    int h = idx / qk_head_dim;
    int j = idx - h * qk_head_dim;
    if (j < qk_nope_head_dim) {
        // K_nope from kv_b's per-head [0, qk_nope_head_dim) slice.
        k_row[idx] = kv_b_out[h * kv_b_stride + j];
    } else {
        // K_rope shared across heads.
        k_row[idx] = k_rope[j - qk_nope_head_dim];
    }
}

// Step 3: extract V from kv_b_out — V is per-head [qk_nope_head_dim,
// qk_nope+v_head) of each per-head chunk.  Result is [n_heads * v_head_dim].
extern "C" __global__ void mla_extract_v(
    const float* __restrict__ kv_b_out,
    float*       __restrict__ v_row,
    int n_heads, int qk_nope_head_dim, int v_head_dim)
{
    int kv_b_stride = qk_nope_head_dim + v_head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * v_head_dim;
    if (idx >= total) return;
    int h = idx / v_head_dim;
    int j = idx - h * v_head_dim;
    v_row[idx] = kv_b_out[h * kv_b_stride + qk_nope_head_dim + j];
}

// Step 4: partial-dim RoPE.
//   Q layout: [n_heads * qk_head_dim] where qk_head_dim = nope + rope.
//   For each head, rotate the LAST qk_rope_head_dim elements as a pair-
//   wise (i, i + qk_rope_head_dim/2) rotation — exactly the same shape as
//   the standard rope_apply kernel, just scoped to the rope sub-region.
// Single-token (seq=1) only — same shape as the decode-step rope_apply.
// step_args[0] = pos.
extern "C" __global__ void mla_rope_q_partial(
    float*       __restrict__ q,            // [n_heads * qk_head_dim]
    int n_heads, int qk_head_dim, int qk_nope_head_dim,
    float base, const int* __restrict__ step_args)
{
    int qk_rope_head_dim = qk_head_dim - qk_nope_head_dim;
    int hd_half = qk_rope_head_dim / 2;
    int total = n_heads * hd_half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx / hd_half;
    int i = idx - h * hd_half;
    int base_off = h * qk_head_dim + qk_nope_head_dim;
    float pos = (float)step_args[0];
    float exp = -2.0f * (float)i / (float)qk_rope_head_dim;
    float theta = pos * powf(base, exp);
    float c = cosf(theta), s = sinf(theta);
    int i0 = base_off + i;
    int i1 = base_off + i + hd_half;
    float x0 = q[i0], x1 = q[i1];
    q[i0] = x0 * c - x1 * s;
    q[i1] = x0 * s + x1 * c;
}

// Same partial RoPE but on the SHARED k_rope vector ([qk_rope_head_dim],
// single instance — not per-head).  Used right before the kv_b path.
extern "C" __global__ void mla_rope_k_shared(
    float*       __restrict__ k_rope,    // [qk_rope_head_dim]
    int qk_rope_head_dim,
    float base, const int* __restrict__ step_args)
{
    int hd_half = qk_rope_head_dim / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hd_half) return;
    float pos = (float)step_args[0];
    float exp = -2.0f * (float)i / (float)qk_rope_head_dim;
    float theta = pos * powf(base, exp);
    float c = cosf(theta), s = sinf(theta);
    float x0 = k_rope[i], x1 = k_rope[i + hd_half];
    k_rope[i] = x0 * c - x1 * s;
    k_rope[i + hd_half] = x0 * s + x1 * c;
}

// FR-17-extra-mla-fwd YaRN: YaRN-by-parts RoPE scaling for DeepSeek-V2 /
// GLM-4.7-flash long-context.  For each frequency dim i:
//   wavelength_i = 2π * base^(2i/d)
//   - "low frequency" dims (long wavelength) get FULL linear interpolation
//     (pos / s)
//   - "high frequency" dims (short wavelength) get NO interpolation
//     (pos as-is, extrapolation)
//   - in between: smooth linear ramp
// The ramp bounds are expressed in `rotation counts` β (beta_fast=32,
// beta_slow=1 are the YaRN paper defaults) and converted to dim indices via:
//   find_correction_dim(β) = d * ln(orig_ctx / (β * 2π)) / (2 * ln(base))
//
// Result: theta_yarn = pos * scale_factor_i * base^(-2i/d)
// where scale_factor_i = (1 - ramp_i) + ramp_i / s.
extern "C" __device__ float yarn_correction_dim(
    float num_rotations, float head_dim, float base, float orig_ctx)
{
    return head_dim * logf(orig_ctx / (num_rotations * 6.283185307179586f))
           / (2.0f * logf(base));
}

extern "C" __device__ float yarn_scale_factor(
    int i, int head_dim, float base, float yarn_s,
    float yarn_orig_ctx, float yarn_beta_fast, float yarn_beta_slow)
{
    float i_high = yarn_correction_dim(yarn_beta_fast, (float)head_dim, base, yarn_orig_ctx);
    float i_low  = yarn_correction_dim(yarn_beta_slow, (float)head_dim, base, yarn_orig_ctx);
    // The ramp goes from "no interpolation" (small i, high freq, ramp=0) to
    // "full interpolation" (large i, low freq, ramp=1).  HF/llama.cpp use:
    //   ramp = clip((i - i_low) / (i_high - i_low), 0, 1)  — but with the
    //   roles flipped depending on which end is "high freq".  In this
    //   parametrization (i increases → lower freq → longer wavelength):
    //     ramp = clip((i - i_low) / (i_high - i_low), 0, 1)
    //     scale_factor = (1 - ramp) + ramp / s
    float denom = fmaxf(i_high - i_low, 1e-3f);
    float ramp = ((float)i - i_low) / denom;
    ramp = fmaxf(0.0f, fminf(1.0f, ramp));
    return (1.0f - ramp) + ramp / yarn_s;
}

extern "C" __global__ void mla_rope_q_partial_yarn(
    float*       __restrict__ q,
    int n_heads, int qk_head_dim, int qk_nope_head_dim,
    float base, float yarn_s, float yarn_orig_ctx,
    float yarn_beta_fast, float yarn_beta_slow,
    const int* __restrict__ step_args)
{
    int qk_rope_head_dim = qk_head_dim - qk_nope_head_dim;
    int hd_half = qk_rope_head_dim / 2;
    int total = n_heads * hd_half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = idx / hd_half;
    int i = idx - h * hd_half;
    int base_off = h * qk_head_dim + qk_nope_head_dim;
    float pos = (float)step_args[0];
    // For YaRN, the per-pair dim index used in correction_dim is the
    // half-index doubled (i.e. 2*i), which corresponds to the "true" rotary
    // frequency index in [0, qk_rope_head_dim).
    float scale_factor = yarn_scale_factor(2 * i, qk_rope_head_dim, base,
        yarn_s, yarn_orig_ctx, yarn_beta_fast, yarn_beta_slow);
    float exp = -2.0f * (float)i / (float)qk_rope_head_dim;
    float theta = pos * scale_factor * powf(base, exp);
    float c = cosf(theta), s = sinf(theta);
    int i0 = base_off + i;
    int i1 = base_off + i + hd_half;
    float x0 = q[i0], x1 = q[i1];
    q[i0] = x0 * c - x1 * s;
    q[i1] = x0 * s + x1 * c;
}

extern "C" __global__ void mla_rope_k_shared_yarn(
    float*       __restrict__ k_rope,
    int qk_rope_head_dim,
    float base, float yarn_s, float yarn_orig_ctx,
    float yarn_beta_fast, float yarn_beta_slow,
    const int* __restrict__ step_args)
{
    int hd_half = qk_rope_head_dim / 2;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hd_half) return;
    float pos = (float)step_args[0];
    float scale_factor = yarn_scale_factor(2 * i, qk_rope_head_dim, base,
        yarn_s, yarn_orig_ctx, yarn_beta_fast, yarn_beta_slow);
    float exp = -2.0f * (float)i / (float)qk_rope_head_dim;
    float theta = pos * scale_factor * powf(base, exp);
    float c = cosf(theta), s = sinf(theta);
    float x0 = k_rope[i], x1 = k_rope[i + hd_half];
    k_rope[i] = x0 * c - x1 * s;
    k_rope[i + hd_half] = x0 * s + x1 * c;
}

// FR-17-extra-mla-absorbed — GLM-4.7-flash absorbed-MLA Q-side absorption.
// Combines per-head Q-nope absorption via Q8_0 w_k_b + q_pe concat in one
// launch.  Output layout per head: [q_nope_absorbed (kv_lora_rank) || q_pe
// (qk_rope)] = qk_head_dim total.
//
// w_k_b GGUF shape: [q_nope_per_head, kv_lora_rank, n_heads] (Q8_0).
//   Per head matrix viewed as [kv_lora_rank rows × q_nope_per_head cols]:
//     row stride bytes = (q_nope_per_head/32) * 34
//     per-head bytes  = kv_lora_rank * row_stride
//
// Grid: (kv_lora_rank + qk_rope, n_heads).  One CTA per (output_col, head)
// pair, threadIdx.x lanes cooperatively reduce over the q_nope dim if
// needed.  For first witness we use 1 thread per CTA — adequate when
// q_nope_per_head ≤ 192 (4 Q8_0 blocks).
extern "C" __global__ void mla_absorb_q_q8_0(
    const float*         __restrict__ q_proj,     // [n_heads * key_mla]
    const unsigned char* __restrict__ w_k_b,      // [n_heads * kv_lora_rank * (q_nope/32) * 34]
    float*               __restrict__ q_out,      // [n_heads * (kv_lora_rank + qk_rope)]
    int n_heads, int key_mla, int qk_rope, int kv_lora_rank, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    int q_nope = key_mla - qk_rope;
    int out_per_head = kv_lora_rank + qk_rope;
    if (oi >= out_per_head || h >= n_heads) return;

    if (oi < kv_lora_rank) {
        // Per-head Q8_0 matmul: q_out[h, oi] = sum_j w_k_b[h, oi, j] * q_proj[h, j].
        size_t base_bytes = (size_t)h * kv_lora_rank * blocks_per_row * 34
                          + (size_t)oi * blocks_per_row * 34;
        const unsigned char* base = w_k_b + base_bytes;
        const float* qin = q_proj + (size_t)h * key_mla;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const unsigned char* blk = base + (size_t)b * 34;
            unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const signed char* qs = (const signed char*)(blk + 2);
            #pragma unroll
            for (int k = 0; k < 32; k++) {
                acc += qin[b * 32 + k] * (d * (float)(int)qs[k]);
            }
        }
        q_out[(size_t)h * out_per_head + oi] = acc;
    } else {
        // q_pe concat: q_out[h, kv_lora_rank + j] = q_proj[h, q_nope + j].
        int j = oi - kv_lora_rank;
        q_out[(size_t)h * out_per_head + oi] =
            q_proj[(size_t)h * key_mla + q_nope + j];
    }
}

// FR-17-extra-mla-absorbed — GLM-4.7-flash absorbed-MLA V-side reduction.
// Per-head attn_v (kv_lora_rank dim) → per-head attn_out (value_mla dim)
// via Q8_0 w_v_b per head.
//
// w_v_b GGUF shape: [kv_lora_rank, value_mla, n_heads] (Q8_0).
//   Per head matrix [value_mla rows × kv_lora_rank cols]:
//     row stride bytes = (kv_lora_rank/32) * 34
//     per-head bytes = value_mla * row_stride
extern "C" __global__ void mla_absorb_v_q8_0(
    const float*         __restrict__ attn_v,     // [n_heads * kv_lora_rank]
    const unsigned char* __restrict__ w_v_b,      // [n_heads * value_mla * (kv_lora_rank/32) * 34]
    float*               __restrict__ attn_out,   // [n_heads * value_mla]
    int n_heads, int kv_lora_rank, int value_mla, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    if (oi >= value_mla || h >= n_heads) return;

    size_t base_bytes = (size_t)h * value_mla * blocks_per_row * 34
                      + (size_t)oi * blocks_per_row * 34;
    const unsigned char* base = w_v_b + base_bytes;
    const float* ain = attn_v + (size_t)h * kv_lora_rank;
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* blk = base + (size_t)b * 34;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        const signed char* qs = (const signed char*)(blk + 2);
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            acc += ain[b * 32 + k] * (d * (float)(int)qs[k]);
        }
    }
    attn_out[(size_t)h * value_mla + oi] = acc;
}

// FR-17-extra-mla-absorbed — broadcast compressed c_kv (+ k_pe for K) to
// all n_q_heads slots.  Used to fit absorbed-MLA's MQA-style shared
// K/V cache into the per-head paged_attention_mla kernel that expects
// per-head K/V data already laid out per head.
//
// k_row[h, j in 0..kv_lora_rank]   = c_kv[j]
// k_row[h, j in kv_lora_rank..]    = k_pe[j - kv_lora_rank]
// v_row[h, j in 0..kv_lora_rank]   = c_kv[j]
extern "C" __global__ void mla_broadcast_kv_for_mqa(
    const float* __restrict__ c_kv,        // [kv_lora_rank]
    const float* __restrict__ k_pe,        // [qk_rope]
    float*       __restrict__ k_row,       // [n_heads * (kv_lora_rank + qk_rope)]
    float*       __restrict__ v_row,       // [n_heads * kv_lora_rank]
    int n_heads, int kv_lora_rank, int qk_rope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k_per_head = kv_lora_rank + qk_rope;
    int total_k = n_heads * k_per_head;
    int total_v = n_heads * kv_lora_rank;
    if (idx < total_k) {
        int j = idx % k_per_head;
        k_row[idx] = (j < kv_lora_rank) ? c_kv[j] : k_pe[j - kv_lora_rank];
    }
    if (idx < total_v) {
        int j = idx % kv_lora_rank;
        v_row[idx] = c_kv[j];
    }
}

// FR-17-extra-mla-absorbed-dtypes — multi-dtype Q absorption kernels.
//
// All mirror the Q8_0 variant's CTA / grid shape (one CTA per
// (out_col, head), one thread per CTA), so q_nope_per_head fits in a
// single thread loop.  Only the per-block dequant body differs.
//
// =====================
// F16 (dt = 1) — 2 bytes per element, no blocks.
// =====================
extern "C" __global__ void mla_absorb_q_f16(
    const float*          __restrict__ q_proj,
    const unsigned short* __restrict__ w_k_b,
    float*                __restrict__ q_out,
    int n_heads, int key_mla, int qk_rope, int kv_lora_rank, int n_in_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    int q_nope = key_mla - qk_rope;
    int out_per_head = kv_lora_rank + qk_rope;
    if (oi >= out_per_head || h >= n_heads) return;

    if (oi < kv_lora_rank) {
        size_t row_off = (size_t)h * kv_lora_rank * n_in_per_row
                       + (size_t)oi * n_in_per_row;
        const unsigned short* w_row = w_k_b + row_off;
        const float* qin = q_proj + (size_t)h * key_mla;
        float acc = 0.0f;
        for (int k = 0; k < n_in_per_row; k++) {
            acc += qin[k] * aether_f16_to_f32_dev(w_row[k]);
        }
        q_out[(size_t)h * out_per_head + oi] = acc;
    } else {
        int j = oi - kv_lora_rank;
        q_out[(size_t)h * out_per_head + oi] =
            q_proj[(size_t)h * key_mla + q_nope + j];
    }
}

extern "C" __global__ void mla_absorb_v_f16(
    const float*          __restrict__ attn_v,
    const unsigned short* __restrict__ w_v_b,
    float*                __restrict__ attn_out,
    int n_heads, int kv_lora_rank, int value_mla, int n_in_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    if (oi >= value_mla || h >= n_heads) return;

    size_t row_off = (size_t)h * value_mla * n_in_per_row
                   + (size_t)oi * n_in_per_row;
    const unsigned short* w_row = w_v_b + row_off;
    const float* ain = attn_v + (size_t)h * kv_lora_rank;
    float acc = 0.0f;
    for (int k = 0; k < n_in_per_row; k++) {
        acc += ain[k] * aether_f16_to_f32_dev(w_row[k]);
    }
    attn_out[(size_t)h * value_mla + oi] = acc;
}

// =====================
// Q4_K (dt = 12) — 144 bytes per 256-elem super-block, requires
// n_in_per_row % 256 == 0.  Scale/min decode mirrors Q4_K standalone
// matmul (q4k_get_scale / q4k_get_min inlined to avoid cross-unit deps).
// =====================
extern "C" __global__ void mla_absorb_q_q4_k(
    const float*         __restrict__ q_proj,
    const unsigned char* __restrict__ w_k_b,
    float*               __restrict__ q_out,
    int n_heads, int key_mla, int qk_rope, int kv_lora_rank, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    int q_nope = key_mla - qk_rope;
    int out_per_head = kv_lora_rank + qk_rope;
    if (oi >= out_per_head || h >= n_heads) return;

    if (oi < kv_lora_rank) {
        size_t base_bytes = (size_t)h * kv_lora_rank * blocks_per_row * 144
                          + (size_t)oi * blocks_per_row * 144;
        const unsigned char* base = w_k_b + base_bytes;
        const float* qin = q_proj + (size_t)h * key_mla;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const unsigned char* blk = base + (size_t)b * 144;
            unsigned short d_bits    = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
            unsigned short dmin_bits = ((unsigned short)blk[3] << 8) | (unsigned short)blk[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* sc = blk + 4;
            const unsigned char* qs = blk + 16;
            for (int sub = 0; sub < 8; sub++) {
                unsigned int sc6, mn6;
                if (sub < 4) {
                    sc6 = (unsigned int)sc[sub] & 63u;
                    mn6 = (unsigned int)sc[sub + 4] & 63u;
                } else {
                    sc6 = ((unsigned int)sc[sub + 4] & 0xFu)
                        | (((unsigned int)sc[sub - 4] >> 6) << 4);
                    mn6 = ((unsigned int)sc[sub + 4] >> 4)
                        | (((unsigned int)sc[sub] >> 6) << 4);
                }
                float d_eff = d * (float)sc6;
                float m_eff = dmin * (float)mn6;
                int j = sub >> 1;
                int is_hi = sub & 1;
                int qs_off = j * 32;
                for (int l = 0; l < 32; l++) {
                    unsigned char byte = qs[qs_off + l];
                    unsigned int nibble = is_hi
                        ? (((unsigned int)byte >> 4) & 0xFu)
                        : ((unsigned int)byte & 0xFu);
                    float w_val = d_eff * (float)nibble - m_eff;
                    acc += qin[b * 256 + sub * 32 + l] * w_val;
                }
            }
        }
        q_out[(size_t)h * out_per_head + oi] = acc;
    } else {
        int j = oi - kv_lora_rank;
        q_out[(size_t)h * out_per_head + oi] =
            q_proj[(size_t)h * key_mla + q_nope + j];
    }
}

extern "C" __global__ void mla_absorb_v_q4_k(
    const float*         __restrict__ attn_v,
    const unsigned char* __restrict__ w_v_b,
    float*               __restrict__ attn_out,
    int n_heads, int kv_lora_rank, int value_mla, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    if (oi >= value_mla || h >= n_heads) return;

    size_t base_bytes = (size_t)h * value_mla * blocks_per_row * 144
                      + (size_t)oi * blocks_per_row * 144;
    const unsigned char* base = w_v_b + base_bytes;
    const float* ain = attn_v + (size_t)h * kv_lora_rank;
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* blk = base + (size_t)b * 144;
        unsigned short d_bits    = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        unsigned short dmin_bits = ((unsigned short)blk[3] << 8) | (unsigned short)blk[2];
        float d    = aether_f16_to_f32_dev(d_bits);
        float dmin = aether_f16_to_f32_dev(dmin_bits);
        const unsigned char* sc = blk + 4;
        const unsigned char* qs = blk + 16;
        for (int sub = 0; sub < 8; sub++) {
            unsigned int sc6, mn6;
            if (sub < 4) {
                sc6 = (unsigned int)sc[sub] & 63u;
                mn6 = (unsigned int)sc[sub + 4] & 63u;
            } else {
                sc6 = ((unsigned int)sc[sub + 4] & 0xFu)
                    | (((unsigned int)sc[sub - 4] >> 6) << 4);
                mn6 = ((unsigned int)sc[sub + 4] >> 4)
                    | (((unsigned int)sc[sub] >> 6) << 4);
            }
            float d_eff = d * (float)sc6;
            float m_eff = dmin * (float)mn6;
            int j = sub >> 1;
            int is_hi = sub & 1;
            int qs_off = j * 32;
            for (int l = 0; l < 32; l++) {
                unsigned char byte = qs[qs_off + l];
                unsigned int nibble = is_hi
                    ? (((unsigned int)byte >> 4) & 0xFu)
                    : ((unsigned int)byte & 0xFu);
                float w_val = d_eff * (float)nibble - m_eff;
                acc += ain[b * 256 + sub * 32 + l] * w_val;
            }
        }
    }
    attn_out[(size_t)h * value_mla + oi] = acc;
}

// =====================
// Q5_K (dt = 13) — 176 bytes per 256-elem super-block.  Same Q4_K-style
// scales + 32-byte qh high-bits + 128-byte qs nibbles.
// =====================
extern "C" __global__ void mla_absorb_q_q5_k(
    const float*         __restrict__ q_proj,
    const unsigned char* __restrict__ w_k_b,
    float*               __restrict__ q_out,
    int n_heads, int key_mla, int qk_rope, int kv_lora_rank, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    int q_nope = key_mla - qk_rope;
    int out_per_head = kv_lora_rank + qk_rope;
    if (oi >= out_per_head || h >= n_heads) return;

    if (oi < kv_lora_rank) {
        size_t base_bytes = (size_t)h * kv_lora_rank * blocks_per_row * 176
                          + (size_t)oi * blocks_per_row * 176;
        const unsigned char* base = w_k_b + base_bytes;
        const float* qin = q_proj + (size_t)h * key_mla;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const unsigned char* blk = base + (size_t)b * 176;
            unsigned short d_bits    = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
            unsigned short dmin_bits = ((unsigned short)blk[3] << 8) | (unsigned short)blk[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* sc = blk + 4;
            const unsigned char* qh = blk + 16;
            const unsigned char* qs = blk + 48;
            for (int sub = 0; sub < 8; sub++) {
                unsigned int sc6, mn6;
                if (sub < 4) {
                    sc6 = (unsigned int)sc[sub] & 63u;
                    mn6 = (unsigned int)sc[sub + 4] & 63u;
                } else {
                    sc6 = ((unsigned int)sc[sub + 4] & 0xFu)
                        | (((unsigned int)sc[sub - 4] >> 6) << 4);
                    mn6 = ((unsigned int)sc[sub + 4] >> 4)
                        | (((unsigned int)sc[sub] >> 6) << 4);
                }
                float d_eff = d * (float)sc6;
                float m_eff = dmin * (float)mn6;
                int j = sub >> 1;
                int is_hi = sub & 1;
                int qs_off = j * 32;
                for (int l = 0; l < 32; l++) {
                    unsigned char byte = qs[qs_off + l];
                    unsigned int nibble = is_hi
                        ? (((unsigned int)byte >> 4) & 0xFu)
                        : ((unsigned int)byte & 0xFu);
                    unsigned int hi_bit = ((unsigned int)qh[l] >> sub) & 1u;
                    unsigned int quant  = nibble | (hi_bit << 4);
                    float w_val = d_eff * (float)quant - m_eff;
                    acc += qin[b * 256 + sub * 32 + l] * w_val;
                }
            }
        }
        q_out[(size_t)h * out_per_head + oi] = acc;
    } else {
        int j = oi - kv_lora_rank;
        q_out[(size_t)h * out_per_head + oi] =
            q_proj[(size_t)h * key_mla + q_nope + j];
    }
}

extern "C" __global__ void mla_absorb_v_q5_k(
    const float*         __restrict__ attn_v,
    const unsigned char* __restrict__ w_v_b,
    float*               __restrict__ attn_out,
    int n_heads, int kv_lora_rank, int value_mla, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    if (oi >= value_mla || h >= n_heads) return;

    size_t base_bytes = (size_t)h * value_mla * blocks_per_row * 176
                      + (size_t)oi * blocks_per_row * 176;
    const unsigned char* base = w_v_b + base_bytes;
    const float* ain = attn_v + (size_t)h * kv_lora_rank;
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* blk = base + (size_t)b * 176;
        unsigned short d_bits    = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        unsigned short dmin_bits = ((unsigned short)blk[3] << 8) | (unsigned short)blk[2];
        float d    = aether_f16_to_f32_dev(d_bits);
        float dmin = aether_f16_to_f32_dev(dmin_bits);
        const unsigned char* sc = blk + 4;
        const unsigned char* qh = blk + 16;
        const unsigned char* qs = blk + 48;
        for (int sub = 0; sub < 8; sub++) {
            unsigned int sc6, mn6;
            if (sub < 4) {
                sc6 = (unsigned int)sc[sub] & 63u;
                mn6 = (unsigned int)sc[sub + 4] & 63u;
            } else {
                sc6 = ((unsigned int)sc[sub + 4] & 0xFu)
                    | (((unsigned int)sc[sub - 4] >> 6) << 4);
                mn6 = ((unsigned int)sc[sub + 4] >> 4)
                    | (((unsigned int)sc[sub] >> 6) << 4);
            }
            float d_eff = d * (float)sc6;
            float m_eff = dmin * (float)mn6;
            int j = sub >> 1;
            int is_hi = sub & 1;
            int qs_off = j * 32;
            for (int l = 0; l < 32; l++) {
                unsigned char byte = qs[qs_off + l];
                unsigned int nibble = is_hi
                    ? (((unsigned int)byte >> 4) & 0xFu)
                    : ((unsigned int)byte & 0xFu);
                unsigned int hi_bit = ((unsigned int)qh[l] >> sub) & 1u;
                unsigned int quant  = nibble | (hi_bit << 4);
                float w_val = d_eff * (float)quant - m_eff;
                acc += ain[b * 256 + sub * 32 + l] * w_val;
            }
        }
    }
    attn_out[(size_t)h * value_mla + oi] = acc;
}

// =====================
// Q6_K (dt = 14) — 210 bytes per 256-elem super-block.  Layout:
//   ql[0..128]   = low 4 bits
//   qh[128..192] = high 2 bits
//   sc[192..208] = i8 sub-block scales (16 sub-blocks of 16 elems each)
//   d  [208..210] = f16 super-block scale
// =====================
extern "C" __global__ void mla_absorb_q_q6_k(
    const float*         __restrict__ q_proj,
    const unsigned char* __restrict__ w_k_b,
    float*               __restrict__ q_out,
    int n_heads, int key_mla, int qk_rope, int kv_lora_rank, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    int q_nope = key_mla - qk_rope;
    int out_per_head = kv_lora_rank + qk_rope;
    if (oi >= out_per_head || h >= n_heads) return;

    if (oi < kv_lora_rank) {
        size_t base_bytes = (size_t)h * kv_lora_rank * blocks_per_row * 210
                          + (size_t)oi * blocks_per_row * 210;
        const unsigned char* base = w_k_b + base_bytes;
        const float* qin = q_proj + (size_t)h * key_mla;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const unsigned char* blk = base + (size_t)b * 210;
            const unsigned char* ql = blk;
            const unsigned char* qh = blk + 128;
            const signed char*   sc = (const signed char*)(blk + 192);
            unsigned short d_bits = ((unsigned short)blk[209] << 8) | (unsigned short)blk[208];
            float d = aether_f16_to_f32_dev(d_bits);
            for (int n_outer = 0; n_outer < 2; n_outer++) {
                int ql_off = n_outer * 64;
                int qh_off = n_outer * 32;
                int sc_off = n_outer * 8;
                for (int l = 0; l < 32; l++) {
                    int is = l >> 4;
                    unsigned char qhv = qh[qh_off + l];
                    int q1 = (int)((ql[ql_off + l] & 0xFu) | (((unsigned int)(qhv >> 0) & 3u) << 4)) - 32;
                    int q2 = (int)((ql[ql_off + l + 32] & 0xFu) | (((unsigned int)(qhv >> 2) & 3u) << 4)) - 32;
                    int q3 = (int)((((unsigned int)ql[ql_off + l] >> 4) & 0xFu) | (((unsigned int)(qhv >> 4) & 3u) << 4)) - 32;
                    int q4 = (int)((((unsigned int)ql[ql_off + l + 32] >> 4) & 0xFu) | (((unsigned int)(qhv >> 6) & 3u) << 4)) - 32;
                    float s1 = d * (float)sc[sc_off + is + 0];
                    float s2 = d * (float)sc[sc_off + is + 2];
                    float s3 = d * (float)sc[sc_off + is + 4];
                    float s4 = d * (float)sc[sc_off + is + 6];
                    int a_base = b * 256 + n_outer * 128;
                    acc += qin[a_base + l +  0] * (s1 * (float)q1);
                    acc += qin[a_base + l + 32] * (s2 * (float)q2);
                    acc += qin[a_base + l + 64] * (s3 * (float)q3);
                    acc += qin[a_base + l + 96] * (s4 * (float)q4);
                }
            }
        }
        q_out[(size_t)h * out_per_head + oi] = acc;
    } else {
        int j = oi - kv_lora_rank;
        q_out[(size_t)h * out_per_head + oi] =
            q_proj[(size_t)h * key_mla + q_nope + j];
    }
}

extern "C" __global__ void mla_absorb_v_q6_k(
    const float*         __restrict__ attn_v,
    const unsigned char* __restrict__ w_v_b,
    float*               __restrict__ attn_out,
    int n_heads, int kv_lora_rank, int value_mla, int blocks_per_row)
{
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    if (oi >= value_mla || h >= n_heads) return;

    size_t base_bytes = (size_t)h * value_mla * blocks_per_row * 210
                      + (size_t)oi * blocks_per_row * 210;
    const unsigned char* base = w_v_b + base_bytes;
    const float* ain = attn_v + (size_t)h * kv_lora_rank;
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* blk = base + (size_t)b * 210;
        const unsigned char* ql = blk;
        const unsigned char* qh = blk + 128;
        const signed char*   sc = (const signed char*)(blk + 192);
        unsigned short d_bits = ((unsigned short)blk[209] << 8) | (unsigned short)blk[208];
        float d = aether_f16_to_f32_dev(d_bits);
        for (int n_outer = 0; n_outer < 2; n_outer++) {
            int ql_off = n_outer * 64;
            int qh_off = n_outer * 32;
            int sc_off = n_outer * 8;
            for (int l = 0; l < 32; l++) {
                int is = l >> 4;
                unsigned char qhv = qh[qh_off + l];
                int q1 = (int)((ql[ql_off + l] & 0xFu) | (((unsigned int)(qhv >> 0) & 3u) << 4)) - 32;
                int q2 = (int)((ql[ql_off + l + 32] & 0xFu) | (((unsigned int)(qhv >> 2) & 3u) << 4)) - 32;
                int q3 = (int)((((unsigned int)ql[ql_off + l] >> 4) & 0xFu) | (((unsigned int)(qhv >> 4) & 3u) << 4)) - 32;
                int q4 = (int)((((unsigned int)ql[ql_off + l + 32] >> 4) & 0xFu) | (((unsigned int)(qhv >> 6) & 3u) << 4)) - 32;
                float s1 = d * (float)sc[sc_off + is + 0];
                float s2 = d * (float)sc[sc_off + is + 2];
                float s3 = d * (float)sc[sc_off + is + 4];
                float s4 = d * (float)sc[sc_off + is + 6];
                int a_base = b * 256 + n_outer * 128;
                acc += ain[a_base + l +  0] * (s1 * (float)q1);
                acc += ain[a_base + l + 32] * (s2 * (float)q2);
                acc += ain[a_base + l + 64] * (s3 * (float)q3);
                acc += ain[a_base + l + 96] * (s4 * (float)q4);
            }
        }
    }
    attn_out[(size_t)h * value_mla + oi] = acc;
}

// =====================
// IQ4_NL (dt = 20) — 18 bytes per 32-elem block.  f16 d + 16 nibble bytes
// indexing the kvalues_iq4nl signed-int8 codebook.
// =====================
extern "C" __global__ void mla_absorb_q_iq4_nl(
    const float*         __restrict__ q_proj,
    const unsigned char* __restrict__ w_k_b,
    float*               __restrict__ q_out,
    int n_heads, int key_mla, int qk_rope, int kv_lora_rank, int blocks_per_row)
{
    static const int kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    int q_nope = key_mla - qk_rope;
    int out_per_head = kv_lora_rank + qk_rope;
    if (oi >= out_per_head || h >= n_heads) return;

    if (oi < kv_lora_rank) {
        size_t base_bytes = (size_t)h * kv_lora_rank * blocks_per_row * 18
                          + (size_t)oi * blocks_per_row * 18;
        const unsigned char* base = w_k_b + base_bytes;
        const float* qin = q_proj + (size_t)h * key_mla;
        float acc = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const unsigned char* blk = base + (size_t)b * 18;
            unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const unsigned char* qs = blk + 2;
            for (int i = 0; i < 16; i++) {
                unsigned char byte = qs[i];
                int q_lo = kvalues[byte & 0xF];
                int q_hi = kvalues[(byte >> 4) & 0xF];
                acc += qin[b * 32 + i]      * (d * (float)q_lo);
                acc += qin[b * 32 + i + 16] * (d * (float)q_hi);
            }
        }
        q_out[(size_t)h * out_per_head + oi] = acc;
    } else {
        int j = oi - kv_lora_rank;
        q_out[(size_t)h * out_per_head + oi] =
            q_proj[(size_t)h * key_mla + q_nope + j];
    }
}

extern "C" __global__ void mla_absorb_v_iq4_nl(
    const float*         __restrict__ attn_v,
    const unsigned char* __restrict__ w_v_b,
    float*               __restrict__ attn_out,
    int n_heads, int kv_lora_rank, int value_mla, int blocks_per_row)
{
    static const int kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };
    int oi = blockIdx.x;
    int h  = blockIdx.y;
    if (oi >= value_mla || h >= n_heads) return;

    size_t base_bytes = (size_t)h * value_mla * blocks_per_row * 18
                      + (size_t)oi * blocks_per_row * 18;
    const unsigned char* base = w_v_b + base_bytes;
    const float* ain = attn_v + (size_t)h * kv_lora_rank;
    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; b++) {
        const unsigned char* blk = base + (size_t)b * 18;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        const unsigned char* qs = blk + 2;
        for (int i = 0; i < 16; i++) {
            unsigned char byte = qs[i];
            int q_lo = kvalues[byte & 0xF];
            int q_hi = kvalues[(byte >> 4) & 0xF];
            acc += ain[b * 32 + i]      * (d * (float)q_lo);
            acc += ain[b * 32 + i + 16] * (d * (float)q_hi);
        }
    }
    attn_out[(size_t)h * value_mla + oi] = acc;
}
"#;

/// Embedded CUDA C source for the small custom kernels cuBLAS doesn't
/// cover. JIT-compiled to PTX once at first `aether_dev_init` and loaded
/// into the context. Kept tiny — the heavy lifting is in cuBLAS sgemm.
const KERNEL_SRC: &str = r#"
extern "C" __global__ void cross_entropy_fwd(
    const float* __restrict__ logits,
    const int*   __restrict__ labels,
    float*       __restrict__ probs,
    float*       __restrict__ losses,
    int B, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    const float* row = logits + i * V;
    float* prow = probs + i * V;
    float mx = row[0];
    for (int j = 1; j < V; j++) if (row[j] > mx) mx = row[j];
    float sum = 0.0f;
    for (int j = 0; j < V; j++) { float e = expf(row[j] - mx); prow[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j < V; j++) prow[j] *= inv;
    int lab = labels[i];
    float p = prow[lab];
    if (p < 1e-12f) p = 1e-12f;
    losses[i] = -logf(p);
}

extern "C" __global__ void cross_entropy_bwd(
    const float* __restrict__ probs,
    const int*   __restrict__ labels,
    float*       __restrict__ dlogits,
    int B, int V)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * V) return;
    int row = i / V;
    int col = i % V;
    float inv_b = 1.0f / (float)B;
    float v = probs[i] * inv_b;
    if (col == labels[row]) v -= inv_b;
    dlogits[i] = v;
}

// Fused softmax-backward + in-place scale. Used by the attention-backward
// fusion pattern (`d_scores = softmax_bwd(attn, d_attn); d_scores.scale(s)`).
// Saves one extra kernel launch + one extra round-trip through the runtime
// ABI per attention layer's backward.
extern "C" __global__ void softmax_bwd_scaled(
    const float* __restrict__ y,
    const float* __restrict__ dy,
    float*       __restrict__ dx,
    float s,
    int B, int D)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const float* yr  = y  + row * D;
    const float* dyr = dy + row * D;
    float*       dxr = dx + row * D;
    float dot = 0.0f;
    for (int j = 0; j < D; j++) dot += yr[j] * dyr[j];
    for (int j = 0; j < D; j++) dxr[j] = (yr[j] * (dyr[j] - dot)) * s;
}

// Row-wise softmax backward. Given y = softmax(x), dy upstream:
//   dx[i,j] = y[i,j] * (dy[i,j] - sum_k y[i,k] * dy[i,k])
extern "C" __global__ void softmax_bwd(
    const float* __restrict__ y,
    const float* __restrict__ dy,
    float*       __restrict__ dx,
    int B, int D)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const float* yr  = y  + row * D;
    const float* dyr = dy + row * D;
    float*       dxr = dx + row * D;
    float dot = 0.0f;
    for (int j = 0; j < D; j++) dot += yr[j] * dyr[j];
    for (int j = 0; j < D; j++) dxr[j] = yr[j] * (dyr[j] - dot);
}

// Row-wise softmax across last dim D. y[i,j] = exp(x[i,j] - max_i) / sum_i.
extern "C" __global__ void softmax_f32(
    const float* __restrict__ x,
    float*       __restrict__ y,
    int B, int D)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const float* xr = x + row * D;
    float* yr = y + row * D;
    float mx = xr[0];
    for (int j = 1; j < D; j++) if (xr[j] > mx) mx = xr[j];
    float sum = 0.0f;
    for (int j = 0; j < D; j++) { float e = expf(xr[j] - mx); yr[j] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j < D; j++) yr[j] *= inv;
}

// Elementwise scale-in-place: x[i] *= s.
extern "C" __global__ void scale_f32(
    float* __restrict__ x,
    float s,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// Fused gelu-after-something: y[i] = gelu(x[i]) where x is the same buffer
// as y (in-place). Used by the explicit-fusion path for `matmul → gelu`
// chains — the matmul writes into `out`, this kernel rewrites it in place.
extern "C" __global__ void gelu_inplace(
    float* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = y[i];
    float c = 0.7978845608f;
    float t = c * (xi + 0.044715f * xi * xi * xi);
    y[i] = 0.5f * xi * (1.0f + tanhf(t));
}

// Elementwise add: out[i] = a[i] + b[i].
extern "C" __global__ void add_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ out,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

// GELU forward (tanh approximation, matches torch / candle defaults):
//   y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
extern "C" __global__ void gelu_fwd(
    const float* __restrict__ x,
    float*       __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float c = 0.7978845608f; // sqrt(2/pi)
    float t = c * (xi + 0.044715f * xi * xi * xi);
    y[i] = 0.5f * xi * (1.0f + tanhf(t));
}

// GELU backward (tanh approx): dx = dy * (0.5*(1+tanh(t)) + 0.5*x*sech^2(t)*c*(1+3*0.044715*x^2))
extern "C" __global__ void gelu_bwd(
    const float* __restrict__ x,
    const float* __restrict__ dy,
    float*       __restrict__ dx,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    float c = 0.7978845608f;
    float k = 0.044715f;
    float t = c * (xi + k * xi * xi * xi);
    float th = tanhf(t);
    float sech2 = 1.0f - th * th;
    float dt_dx = c * (1.0f + 3.0f * k * xi * xi);
    float dy_dxi = 0.5f * (1.0f + th) + 0.5f * xi * sech2 * dt_dx;
    dx[i] = dy[i] * dy_dxi;
}

// LayerNorm forward across last dim D for each of B rows.
//   mean = sum(x)/D ; var = sum((x-mean)^2)/D
//   y = (x - mean) / sqrt(var + eps) * gamma + beta
// Caches per-row mean & rstd for the backward pass.
extern "C" __global__ void layer_norm_fwd(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ y,
    float*       __restrict__ mean_out,
    float*       __restrict__ rstd_out,
    int B, int D, float eps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const float* xr = x + row * D;
    float* yr = y + row * D;
    float m = 0.0f;
    for (int j = 0; j < D; j++) m += xr[j];
    m /= (float)D;
    float v = 0.0f;
    for (int j = 0; j < D; j++) { float d = xr[j] - m; v += d * d; }
    v /= (float)D;
    float rstd = rsqrtf(v + eps);
    for (int j = 0; j < D; j++) yr[j] = (xr[j] - m) * rstd * gamma[j] + beta[j];
    mean_out[row] = m;
    rstd_out[row] = rstd;
}

// LayerNorm backward to dx (gamma/beta grads not produced — sufficient for
// "frozen-norm" experiments; full backward is on the roadmap).
extern "C" __global__ void layer_norm_bwd_dx(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ dy,
    float*       __restrict__ dx,
    int B, int D)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const float* xr = x + row * D;
    const float* dyr = dy + row * D;
    float* dxr = dx + row * D;
    float m = mean[row]; float r = rstd[row];
    // sum1 = sum(dy * gamma); sum2 = sum(dy * gamma * (x - m) * r)
    float s1 = 0.0f, s2 = 0.0f;
    for (int j = 0; j < D; j++) {
        float dyg = dyr[j] * gamma[j];
        s1 += dyg;
        s2 += dyg * (xr[j] - m) * r;
    }
    float invD = 1.0f / (float)D;
    for (int j = 0; j < D; j++) {
        float dyg = dyr[j] * gamma[j];
        dxr[j] = r * (dyg - invD * s1 - invD * (xr[j] - m) * r * s2);
    }
}

// Fused add+LayerNorm: y = LN((a + b) * gamma + beta) over the last dim.
// Equivalent to `add_f32(a, b, tmp); tmp.layer_norm(...)` but fuses the
// residual sum INTO the LN reduction's data load — no intermediate buffer
// needed and the residual passes through L1 only once. Pattern shows up
// once per transformer sublayer (post-attention residual+norm and post-MLP
// residual+norm), so this is one of the highest-frequency fusions.
extern "C" __global__ void add_layer_norm_fwd(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float*       __restrict__ y,
    float*       __restrict__ mean_out,
    float*       __restrict__ rstd_out,
    int B, int D, float eps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const float* ar = a + row * D;
    const float* br = b + row * D;
    float* yr = y + row * D;
    float m = 0.0f;
    for (int j = 0; j < D; j++) m += ar[j] + br[j];
    m /= (float)D;
    float v = 0.0f;
    for (int j = 0; j < D; j++) { float d = (ar[j] + br[j]) - m; v += d * d; }
    v /= (float)D;
    float rstd = rsqrtf(v + eps);
    for (int j = 0; j < D; j++) yr[j] = ((ar[j] + br[j]) - m) * rstd * gamma[j] + beta[j];
    mean_out[row] = m;
    rstd_out[row] = rstd;
}

// LayerNorm parameter backward: per-feature reductions across the batch.
//   dgamma[j] = sum_i dy[i,j] * (x[i,j] - mean[i]) * rstd[i]
//   dbeta[j]  = sum_i dy[i,j]
// Launch D threads — each accumulates across B rows.
extern "C" __global__ void layer_norm_bwd_params(
    const float* __restrict__ x,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ dy,
    float*       __restrict__ dgamma,
    float*       __restrict__ dbeta,
    int B, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= D) return;
    float dg = 0.0f, db = 0.0f;
    for (int i = 0; i < B; i++) {
        float dyi = dy[i * D + j];
        db += dyi;
        dg += dyi * (x[i * D + j] - mean[i]) * rstd[i];
    }
    dgamma[j] = dg;
    dbeta[j]  = db;
}

extern "C" __global__ void adamw_step(
    float*       __restrict__ param,
    const float* __restrict__ grad,
    float*       __restrict__ m,
    float*       __restrict__ v,
    float lr, float beta1, float beta2, float eps, float wd,
    float bc1_inv, float bc2_inv,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mi; v[i] = vi;
    float mh = mi * bc1_inv;
    float vh = vi * bc2_inv;
    param[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * param[i]);
}

// matt-voice / FR-17.5-extra — RMSNorm: y[r,i] = x[r,i] * gamma[i] / sqrt(mean(x[r,:]^2) + eps)
// One thread per row. d ≤ 4096 fits in a single block's worth of shared work.
extern "C" __global__ void rms_norm_fwd(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float*       __restrict__ y,
    float eps,
    int rows, int d)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + r * d;
    float*       yr = y + r * d;
    float sumsq = 0.0f;
    for (int i = 0; i < d; i++) sumsq += xr[i] * xr[i];
    float inv_rms = 1.0f / sqrtf(sumsq / (float)d + eps);
    for (int i = 0; i < d; i++) yr[i] = xr[i] * inv_rms * gamma[i];
}

// matt-voice / FR-17.13-extra — RoPE in-place. Llama-style "half-half"
// pair layout: pair (i, i + head_dim/2). One thread per (t, h, i) tuple
// where i in [0, head_dim/2). theta = (t + pos_start) * base^(-2i/head_dim).
extern "C" __global__ void rope_apply(
    float*       __restrict__ x,
    int seq, int n_heads, int head_dim,
    float base, int pos_start)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hd_half = head_dim / 2;
    int total = seq * n_heads * hd_half;
    if (idx >= total) return;
    int t = idx / (n_heads * hd_half);
    int rem = idx - t * (n_heads * hd_half);
    int h = rem / hd_half;
    int i = rem - h * hd_half;
    int base_off = (t * n_heads + h) * head_dim;
    float pos = (float)(t + pos_start);
    float exp = -2.0f * (float)i / (float)head_dim;
    float theta = pos * powf(base, exp);
    float c = cosf(theta), s = sinf(theta);
    int i0 = base_off + i;
    int i1 = base_off + i + hd_half;
    float x0 = x[i0], x1 = x[i1];
    x[i0] = x0 * c - x1 * s;
    x[i1] = x0 * s + x1 * c;
}

// FR-17.14-extra-deepest-graph -- rope_apply variant that reads pos
// from device memory instead of taking it as an immediate launch arg.
// Lets the autoregressive forward pass be captured into a single CUDA
// graph that's reused for every decode step (only the per-step args
// buffer needs to be updated by h2d each step). step_args[0] = pos.
extern "C" __global__ void rope_apply_devarg(
    float*       __restrict__ x,
    int seq, int n_heads, int head_dim,
    float base, const int* __restrict__ step_args)
{
    int pos_start = step_args[0];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int hd_half = head_dim / 2;
    int total = seq * n_heads * hd_half;
    if (idx >= total) return;
    int t = idx / (n_heads * hd_half);
    int rem = idx - t * (n_heads * hd_half);
    int h = rem / hd_half;
    int i = rem - h * hd_half;
    int base_off = (t * n_heads + h) * head_dim;
    float pos = (float)(t + pos_start);
    float exp = -2.0f * (float)i / (float)head_dim;
    float theta = pos * powf(base, exp);
    float c = cosf(theta), s = sinf(theta);
    int i0 = base_off + i;
    int i1 = base_off + i + hd_half;
    float x0 = x[i0], x1 = x[i1];
    x[i0] = x0 * c - x1 * s;
    x[i1] = x0 * s + x1 * c;
}

// matt-voice / FR-17.13-extra GQA — broadcast n_kv_heads -> n_q_heads
// by repeating each KV head g = n_q_heads / n_kv_heads times.
extern "C" __global__ void gqa_repeat_kv(
    const float* __restrict__ kv_in,
    float*       __restrict__ kv_out,
    int seq, int n_kv_heads, int head_dim, int n_q_heads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq * n_q_heads * head_dim;
    if (idx >= total) return;
    int t = idx / (n_q_heads * head_dim);
    int rem = idx - t * (n_q_heads * head_dim);
    int qh = rem / head_dim;
    int d  = rem - qh * head_dim;
    int g  = n_q_heads / n_kv_heads;
    int kh = qh / g;
    int src_off = (t * n_kv_heads + kh) * head_dim + d;
    kv_out[idx] = kv_in[src_off];
}

// matt-voice / FR-17.6-extra — SiLU in place: x[i] = x[i] * sigmoid(x[i])
// = x[i] / (1 + exp(-x[i])).
extern "C" __global__ void silu_inplace(
    float* __restrict__ x,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xi = x[i];
    x[i] = xi / (1.0f + expf(-xi));
}

// matt-voice — element-wise multiply in place: x[i] *= y[i]. Used by
// SwiGLU MLP after SiLU(gate) so we get silu(gate) * up.
extern "C" __global__ void mul_inplace(
    float*       __restrict__ x,
    const float* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= y[i];
}

// matt-voice — residual / in-place add: x[i] += y[i].
extern "C" __global__ void add_inplace(
    float*       __restrict__ x,
    const float* __restrict__ y,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// matt-voice — broadcast-add a bias vector across rows: x[r, c] += bias[c].
extern "C" __global__ void bias_add(
    float*       __restrict__ x,
    const float* __restrict__ bias,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int c = idx % cols;
    x[idx] += bias[c];
}

// matt-voice / FR-17.14-extra-deepest — Q4_K_M dequant on GPU.
// 144 bytes per 256-quant super-block: f16 d + f16 dmin + 12 packed
// scales/mins + 128 nibble-packed quants. Mirrors aether_dequant_q4_k_m
// in ops::lib.rs, parallelised one-thread-per-output.
//
// Per output qi in [0, 256):
//   sub = qi / 32              (sub-block 0..7)
//   l   = qi % 32              (offset within sub-block)
//   j   = sub / 2              (byte cluster, 0..4)
//   is_hi = sub & 1            (low or high nibble)
//   byte = qs[j*32 + l]
//   nibble = is_hi ? (byte >> 4) & 0xF : byte & 0xF
//   sc, mn = q4k_get_scale_min(sub) from the 12 scales bytes
//   value = d * sc * nibble - dmin * mn

extern "C" __device__ float aether_f16_to_f32_dev(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int bits;
    if (exp == 0u) {
        if (mant == 0u) { bits = sign << 31; return __int_as_float(bits); }
        // Subnormal: normalise the mantissa by shifting left until bit 10
        // is set, decrementing the exponent for each shift. Matches the
        // CPU `aether_f16_to_f32` reference (runtime/src/lib.rs).
        unsigned int m = mant;
        int e = -14;
        while ((m & 0x0400u) == 0u) { m <<= 1; e -= 1; }
        m &= 0x03FFu;
        bits = (sign << 31) | ((unsigned int)(e + 127) << 23) | (m << 13);
        return __int_as_float(bits);
    }
    if (exp == 0x1Fu) {
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
        return __int_as_float(bits);
    }
    unsigned int f32_exp  = (exp - 15u + 127u) << 23;
    unsigned int f32_mant = mant << 13;
    bits = (sign << 31) | f32_exp | f32_mant;
    return __int_as_float(bits);
}

// q4k_get_scale_min: decode sub-block sub's (scale_low6, min_low6)
// from the 12-byte scales array. See ggml-quants.c::get_scale_min_k4.
extern "C" __device__ unsigned int q4k_get_scale(int sub, const unsigned char* sc) {
    if (sub < 4) return sc[sub] & 63u;
    return (sc[sub + 4] & 0xFu) | (((unsigned int)(sc[sub - 4] >> 6)) << 4);
}
extern "C" __device__ unsigned int q4k_get_min(int sub, const unsigned char* sc) {
    if (sub < 4) return sc[sub + 4] & 63u;
    return (sc[sub + 4] >> 4) | (((unsigned int)(sc[sub] >> 6)) << 4);
}

extern "C" __global__ void dequant_q4_k_m(
    const unsigned char* __restrict__ blocks,
    float*               __restrict__ out,
    int n_blocks)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_blocks * 256;
    if (idx >= total) return;
    int bi = idx / 256;
    int qi = idx % 256;
    int sub = qi / 32;
    int l   = qi % 32;
    int j   = sub / 2;
    int is_hi = sub & 1;

    const unsigned char* base = blocks + bi * 144;
    unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
    unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
    float d    = aether_f16_to_f32_dev(d_bits);
    float dmin = aether_f16_to_f32_dev(dmin_bits);

    const unsigned char* scales = base + 4;
    unsigned int sc = q4k_get_scale(sub, scales);
    unsigned int mn = q4k_get_min(sub, scales);

    const unsigned char* qs = base + 16;
    unsigned char byte = qs[j * 32 + l];
    unsigned int nibble = is_hi ? (((unsigned int)byte >> 4) & 0xFu) : ((unsigned int)byte & 0xFu);

    out[idx] = d * (float)sc * (float)nibble - dmin * (float)mn;
}

// matt-voice / FR-17.14-extra-deepest — Q6_K dequant on GPU.
// 210 bytes per 256-quant super-block:
//   bytes 0..128   : ql[128]   -- low 4 bits of each quant
//   bytes 128..192 : qh[64]    -- high 2 bits of each quant
//   bytes 192..208 : scales[16] -- i8 sub-block scales
//   bytes 208..210 : d         -- f16 super-block scale
// Mirrors aether_dequant_q6_k in lib.rs. One thread per output f32.
//
// Layout (per ggml-quants.c::dequantize_row_q6_K):
//   For each l in 0..32, for each n_outer (0..2):
//     q1 = ((ql[l +  0 + 64*n_outer] & 0xF) | ((qh[l + 32*n_outer] >> 0) & 3) << 4) - 32
//     q2 = ((ql[l + 32 + 64*n_outer] & 0xF) | ((qh[l + 32*n_outer] >> 2) & 3) << 4) - 32
//     q3 = ((ql[l +  0 + 64*n_outer]  >> 4) | ((qh[l + 32*n_outer] >> 4) & 3) << 4) - 32
//     q4 = ((ql[l + 32 + 64*n_outer]  >> 4) | ((qh[l + 32*n_outer] >> 6) & 3) << 4) - 32
//     y[l +  0 + 128*n_outer] = d * sc[is +  0 + 8*n_outer] * q1
//     y[l + 32 + 128*n_outer] = d * sc[is +  2 + 8*n_outer] * q2
//     y[l + 64 + 128*n_outer] = d * sc[is +  4 + 8*n_outer] * q3
//     y[l + 96 + 128*n_outer] = d * sc[is +  6 + 8*n_outer] * q4
//   where is = l/16
// matt-voice / FR-17.14-extra-deepest — FUSED Q4_K dequant + matmul.
//
// Computes out[n] = sum_k a[k] * dequant(w_q4k)[n, k] for one row of A
// (seq=1, the autoregressive-generation case).
//
// W layout: GGUF natural order. Each output column ni corresponds to
// row ni of w_q4k (n_blocks super-blocks of 144 bytes each). Row stride
// in W is `n_blocks * 144` bytes.
//
// CTA design (one CTA per BLOCK_N output columns):
//   - threadIdx.x in [0, BLOCK_N)
//   - Per K-tile (256 quants = one super-block):
//     * All BLOCK_N threads cooperatively load 256 floats of A into
//       shared memory (8 loads per thread, fully coalesced).
//     * Each thread reads its OWN super-block of W from global memory,
//       dequants inline, and accumulates fma into a per-thread float.
//   - Each thread writes one output element at the end.
//
// Shared mem: 256 * 4 = 1 KB for A tile.
// Per-thread work: n_blocks * (144 bytes read from W + 256 fma).
extern "C" __global__ void fused_q4k_matmul_seq1(
    const float*         __restrict__ a,           // [k]
    const unsigned char* __restrict__ w,           // n rows of (n_blocks * 144) bytes
    float*               __restrict__ out,         // [n]
    int n, int n_blocks)                           // k = n_blocks * 256
{
    const int BLOCK_N = 32;
    __shared__ float a_tile[256];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        // Cooperatively load 256 floats of A: each of 32 threads loads 8.
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int kk = p * BLOCK_N + threadIdx.x;
            a_tile[kk] = a[bi * 256 + kk];
        }
        __syncthreads();

        if (ni < n) {
            // Dequant THIS thread's super-block of W and accumulate.
            const unsigned char* base = w + (size_t)ni * n_blocks * 144 + (size_t)bi * 144;
            unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* scales = base + 4;
            const unsigned char* qs     = base + 16;

            #pragma unroll
            for (int sub = 0; sub < 8; sub++) {
                int j = sub >> 1;
                int is_hi = sub & 1;
                unsigned int sc = q4k_get_scale(sub, scales);
                unsigned int mn = q4k_get_min(sub, scales);
                float d_eff = d * (float)sc;
                float m_eff = dmin * (float)mn;
                int qs_off = j * 32;
                #pragma unroll 8
                for (int l = 0; l < 32; l++) {
                    unsigned char byte = qs[qs_off + l];
                    unsigned int nibble = is_hi ? (((unsigned int)byte >> 4) & 0xFu) : ((unsigned int)byte & 0xFu);
                    float w_val = d_eff * (float)nibble - m_eff;
                    acc += a_tile[sub * 32 + l] * w_val;
                }
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-iq4_xs-fwd — FUSED IQ4_XS matmul.  Used by cnc's GLM-4.7-flash
// for ~55 tensors.  4-bit "extra small": same kvalues_iq4nl codebook lookup
// as IQ4_NL, but with PER-SUB-BLOCK 6-bit signed scales (vs IQ4_NL's single
// f16 per 32-elem block).  Block size 256, total 136 bytes:
//
//   bytes 0-1    : f16 super-block scale `d`
//   bytes 2-3    : u16 scales_h (16 bits — 2 bits per sub-block × 8 = 16)
//   bytes 4-7    : 4 bytes scales_l (8 nibbles — 4 bits per sub-block × 8 = 32)
//   bytes 8-135  : 128 bytes qs (nibble-packed indices into kvalues_iq4nl)
//
// Per sub-block ib in [0, 8):
//   ls = (scales_l[ib/2] >> (4*(ib%2)) & 0xF) | ((scales_h >> (2*ib)) & 3) << 4
//   dl = d * (ls - 32)                              — signed scale [-32, 31]
//   for j in [0, 16):
//     y[16*ib*2 + j + 0] = dl * kvalues_iq4nl[qs[16*ib + j] & 0xF]
//     y[16*ib*2 + j + 16] = dl * kvalues_iq4nl[qs[16*ib + j] >> 4]
extern "C" __global__ void fused_iq4_xs_matmul_seq1(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks)                           // k = n_blocks * 256
{
    static const int kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };
    const int BLOCK_N = 32;
    __shared__ float a_tile[256];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int kk = p * BLOCK_N + threadIdx.x;
            a_tile[kk] = a[bi * 256 + kk];
        }
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w + (size_t)ni * n_blocks * 136 + (size_t)bi * 136;
            unsigned short d_bits = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            unsigned int scales_h = ((unsigned int)base[3] << 8) | (unsigned int)base[2];
            const unsigned char* scales_l = base + 4;     // 4 bytes
            const unsigned char* qs       = base + 8;     // 128 bytes

            #pragma unroll
            for (int ib = 0; ib < 8; ib++) {
                unsigned int ls_lo = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xFu;
                unsigned int ls_hi = (scales_h >> (2 * ib)) & 3u;
                int ls = (int)(ls_lo | (ls_hi << 4));     // 6-bit unsigned [0, 63]
                float dl = d * (float)(ls - 32);

                int qs_off = ib * 16;
                #pragma unroll 8
                for (int j = 0; j < 16; j++) {
                    unsigned char byte = qs[qs_off + j];
                    int q_lo = kvalues[byte & 0xF];
                    int q_hi = kvalues[(byte >> 4) & 0xF];
                    acc += a_tile[ib * 32 + j]      * (dl * (float)q_lo);
                    acc += a_tile[ib * 32 + j + 16] * (dl * (float)q_hi);
                }
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-iq4_nl-fwd — FUSED IQ4_NL matmul.  Used by cnc's GLM-4.7-flash
// for ~72 tensors.  Same byte layout as Q4_0 (18-byte 32-elem blocks: f16 d
// + 16-byte nibble-packed indices) BUT the nibbles index a 16-entry
// non-linear lookup table of signed int8 values instead of being `(q - 8)`.
//
// Dequant: y[j]    = d * kvalues_iq4nl[qs[j] & 0xF]      for j in [0, 16)
//          y[j+16] = d * kvalues_iq4nl[qs[j] >> 4]
extern "C" __global__ void fused_iq4_nl_matmul_seq1(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks)                           // k = n_blocks * 32
{
    // kvalues_iq4nl from ggml-common.h — signed int8 lookup.
    static const int kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };

    const int BLOCK_N = 32;
    __shared__ float a_tile[32];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 32 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base =
                w + (size_t)ni * n_blocks * 18 + (size_t)bi * 18;
            unsigned short d_bits =
                ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const unsigned char* qs = base + 2;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned char byte = qs[i];
                int q_lo = kvalues[byte & 0xF];
                int q_hi = kvalues[(byte >> 4) & 0xF];
                acc += a_tile[i]      * (d * (float)q_lo);
                acc += a_tile[i + 16] * (d * (float)q_hi);
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-iq3_xxs-fwd — FUSED IQ3_XXS matmul.  Used by cnc's
// glm-4.7-flash-UD-IQ3_XXS GGUF for almost every weight tensor
// (IQ3_XXS is one of llama.cpp's i-quants; 3.0625 average bits/weight,
// codebook-indexed with sign-pattern lookup).
//
// IQ3_XXS block layout (98 bytes per 256-element block):
//   bytes 0-1   : f16 scale `d`
//   bytes 2-65  : 64 codebook indices (one byte per 4-quant lane,
//                 used in pairs → each pair encodes 8 weights)
//   bytes 66-97 : 32 bytes scales_and_signs (8 sub-blocks × 4-byte u32):
//                 low 28 bits = 4 × 7-bit sign indices (one per lane);
//                 high 4 bits = per-sub-block scale index (0..15).
//
// Dequant for sub-block ib32 (32 weights):
//   aux32 = u32 from scales_and_signs[4*ib32 .. 4*ib32+4]
//   db    = d * (0.5 + (aux32 >> 28)) * 0.5
//   For l in 0..4 (each lane = 8 weights):
//     signs = ksigns_iq2xs[(aux32 >> 7*l) & 127]
//     grid1 = iq3xxs_grid[qs[8*ib32 + 2*l + 0]]    // 4 packed uint8 quants
//     grid2 = iq3xxs_grid[qs[8*ib32 + 2*l + 1]]    // 4 packed uint8 quants
//     For j in 0..4:
//       y[8*l + j+0] = db * grid1.byte[j] * (signs & (1<<(j+0)) ? -1 : 1)
//       y[8*l + j+4] = db * grid2.byte[j] * (signs & (1<<(j+4)) ? -1 : 1)
//
// Codebook constants iq3xxs_grid (256 × u32) + ksigns_iq2xs (128 × u8)
// are embedded directly in the kernel source as `static const __device__`
// arrays — they're never going to change and inlining avoids needing to
// upload + pass them as separate device buffers.
extern "C" __global__ void fused_iq3_xxs_matmul_seq1(
    const float*         __restrict__ a,        // [k]
    const unsigned char* __restrict__ w,        // [n * n_blocks * 98]
    float*               __restrict__ out,      // [n]
    int n, int n_blocks)                        // k = n_blocks * 256
{
    // ksigns_iq2xs: 7-bit index -> 8-bit sign pattern
    static const unsigned char ksigns[128] = {
          0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
        144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
        160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
         48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
        192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
         80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
         96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
    };
    // iq3xxs_grid: 8-bit index -> 32-bit packed (4 uint8 quants).
    static const unsigned int iq3xxs_grid[256] = {
        0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
        0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
        0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
        0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
        0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
        0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
        0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
        0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
        0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
        0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
        0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
        0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
        0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
        0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
        0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
        0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
        0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
        0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
        0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
        0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
        0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
        0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
        0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
        0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
        0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
        0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
        0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
        0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
        0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
        0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
        0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
        0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u
    };

    const int BLOCK_N = 32;
    __shared__ float a_tile[256];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        // Cooperatively load 256 floats of A: 32 threads × 8 each.
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int kk = p * 32 + threadIdx.x;
            a_tile[kk] = a[bi * 256 + kk];
        }
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w + (size_t)ni * n_blocks * 98 + (size_t)bi * 98;
            unsigned short d_bits = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const unsigned char* qs = base + 2;
            const unsigned char* sas = base + 2 + 64;

            #pragma unroll
            for (int ib32 = 0; ib32 < 8; ib32++) {
                unsigned int aux32 =
                      (unsigned int)sas[4*ib32 + 0]
                    | ((unsigned int)sas[4*ib32 + 1] << 8)
                    | ((unsigned int)sas[4*ib32 + 2] << 16)
                    | ((unsigned int)sas[4*ib32 + 3] << 24);
                float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;

                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    unsigned int sign_idx = (aux32 >> (7 * l)) & 127u;
                    unsigned int signs    = (unsigned int)ksigns[sign_idx];
                    unsigned int grid1 = iq3xxs_grid[qs[8*ib32 + 2*l + 0]];
                    unsigned int grid2 = iq3xxs_grid[qs[8*ib32 + 2*l + 1]];

                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        unsigned int q0 = (grid1 >> (8 * j)) & 0xFFu;
                        unsigned int q1 = (grid2 >> (8 * j)) & 0xFFu;
                        float s0 = (signs & (1u << (j + 0))) ? -1.0f : 1.0f;
                        float s1 = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
                        acc += a_tile[32*ib32 + 8*l + j + 0] * (db * (float)q0 * s0);
                        acc += a_tile[32*ib32 + 8*l + j + 4] * (db * (float)q1 * s1);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-iq3_s-fwd — FUSED IQ3_S matmul.  Used by cnc's
// glm-4.7-flash-UD-IQ3_XXS GGUF for ~44 tensors (the "selected" 3-bit
// variant — slightly higher precision than IQ3_XXS via per-sub-block
// odd-integer scales and a 512-entry codebook).
//
// IQ3_S block layout (110 bytes per 256-element block):
//   bytes 0-1     : f16 super-block scale `d`
//   bytes 2-65    : 64 bytes qs   (low 8 bits of codebook index, one byte per
//                                   weight-pair-of-4; 64 entries × 4 weights = 256)
//   bytes 66-73   : 8 bytes  qh   (one byte per 32-elem sub-block — supplies
//                                   the high bit (bit 8) of the codebook index
//                                   for each of 4 lanes × 2 codebooks per lane)
//   bytes 74-105  : 32 bytes signs (8 bits per 8-weight lane × 4 lanes × 8 sub-blocks)
//   bytes 106-109 : 4 bytes  scales (4-bit unsigned per sub-block × 8)
//
// Dequant for sub-block ib32 in [0, 8):
//   scale_nib = (scales[ib32 / 2] >> (4 * (ib32 & 1))) & 0xF
//   db        = d * (1 + 2 * scale_nib)      // odd integers 1, 3, ..., 31
//   For l in 0..4:
//     sign = signs[ib32*4 + l]
//     grid1_idx = qs[ib32*8 + 2*l + 0] | ((qh[ib32] << (8 - 2*l)) & 256)
//     grid2_idx = qs[ib32*8 + 2*l + 1] | ((qh[ib32] << (7 - 2*l)) & 256)
//     grid1 = iq3s_grid[grid1_idx]    // 4 packed uint8 quants
//     grid2 = iq3s_grid[grid2_idx]
//     For j in 0..4:
//       y[ib32*32 + 8*l + j + 0] = db * grid1.byte[j] * (sign & (1 << (j+0)) ? -1 : 1)
//       y[ib32*32 + 8*l + j + 4] = db * grid2.byte[j] * (sign & (1 << (j+4)) ? -1 : 1)
//
// Unlike IQ3_XXS, signs are stored DIRECTLY as 8-bit patterns (no ksigns
// indirection) and there's no aux32 packing — the qs/qh/signs/scales arrays
// are each first-class.
extern "C" __global__ void fused_iq3_s_matmul_seq1(
    const float*         __restrict__ a,        // [k]
    const unsigned char* __restrict__ w,        // [n * n_blocks * 110]
    float*               __restrict__ out,      // [n]
    int n, int n_blocks)                        // k = n_blocks * 256
{
    // iq3s_grid: 9-bit index (0..511) -> 32-bit packed (4 uint8 quants).
    static const unsigned int iq3s_grid[512] = {
        0x01010101,0x01010103,0x01010105,0x0101010b,0x0101010f,0x01010301,0x01010303,0x01010305,
        0x01010309,0x0101030d,0x01010501,0x01010503,0x0101050b,0x01010707,0x01010901,0x01010905,
        0x0101090b,0x0101090f,0x01010b03,0x01010b07,0x01010d01,0x01010d05,0x01010f03,0x01010f09,
        0x01010f0f,0x01030101,0x01030103,0x01030105,0x01030109,0x01030301,0x01030303,0x0103030b,
        0x01030501,0x01030507,0x0103050f,0x01030703,0x0103070b,0x01030909,0x01030d03,0x01030d0b,
        0x01030f05,0x01050101,0x01050103,0x0105010b,0x0105010f,0x01050301,0x01050307,0x0105030d,
        0x01050503,0x0105050b,0x01050701,0x01050709,0x01050905,0x0105090b,0x0105090f,0x01050b03,
        0x01050b07,0x01050f01,0x01050f07,0x01070107,0x01070303,0x0107030b,0x01070501,0x01070505,
        0x01070703,0x01070707,0x0107070d,0x01070909,0x01070b01,0x01070b05,0x01070d0f,0x01070f03,
        0x01070f0b,0x01090101,0x01090307,0x0109030f,0x01090503,0x01090509,0x01090705,0x01090901,
        0x01090907,0x01090b03,0x01090f01,0x010b0105,0x010b0109,0x010b0501,0x010b0505,0x010b050d,
        0x010b0707,0x010b0903,0x010b090b,0x010b090f,0x010b0d0d,0x010b0f07,0x010d010d,0x010d0303,
        0x010d0307,0x010d0703,0x010d0b05,0x010d0f03,0x010f0101,0x010f0105,0x010f0109,0x010f0501,
        0x010f0505,0x010f050d,0x010f0707,0x010f0b01,0x010f0b09,0x03010101,0x03010103,0x03010105,
        0x03010109,0x03010301,0x03010303,0x03010307,0x0301030b,0x0301030f,0x03010501,0x03010505,
        0x03010703,0x03010709,0x0301070d,0x03010b09,0x03010b0d,0x03010d03,0x03010f05,0x03030101,
        0x03030103,0x03030107,0x0303010d,0x03030301,0x03030309,0x03030503,0x03030701,0x03030707,
        0x03030903,0x03030b01,0x03030b05,0x03030f01,0x03030f0d,0x03050101,0x03050305,0x0305030b,
        0x0305030f,0x03050501,0x03050509,0x03050705,0x03050901,0x03050907,0x03050b0b,0x03050d01,
        0x03050f05,0x03070103,0x03070109,0x0307010f,0x03070301,0x03070307,0x03070503,0x0307050f,
        0x03070701,0x03070709,0x03070903,0x03070d05,0x03070f01,0x03090107,0x0309010b,0x03090305,
        0x03090309,0x03090703,0x03090707,0x03090905,0x0309090d,0x03090b01,0x03090b09,0x030b0103,
        0x030b0301,0x030b0307,0x030b0503,0x030b0701,0x030b0705,0x030b0b03,0x030d0501,0x030d0509,
        0x030d050f,0x030d0909,0x030d090d,0x030f0103,0x030f0107,0x030f0301,0x030f0305,0x030f0503,
        0x030f070b,0x030f0903,0x030f0d05,0x030f0f01,0x05010101,0x05010103,0x05010107,0x0501010b,
        0x0501010f,0x05010301,0x05010305,0x05010309,0x0501030d,0x05010503,0x05010507,0x0501050f,
        0x05010701,0x05010705,0x05010903,0x05010907,0x0501090b,0x05010b01,0x05010b05,0x05010d0f,
        0x05010f01,0x05010f07,0x05010f0b,0x05030101,0x05030105,0x05030301,0x05030307,0x0503030f,
        0x05030505,0x0503050b,0x05030703,0x05030709,0x05030905,0x05030b03,0x05050103,0x05050109,
        0x0505010f,0x05050503,0x05050507,0x05050701,0x0505070f,0x05050903,0x05050b07,0x05050b0f,
        0x05050f03,0x05050f09,0x05070101,0x05070105,0x0507010b,0x05070303,0x05070505,0x05070509,
        0x05070703,0x05070707,0x05070905,0x05070b01,0x05070d0d,0x05090103,0x0509010f,0x05090501,
        0x05090507,0x05090705,0x0509070b,0x05090903,0x05090f05,0x05090f0b,0x050b0109,0x050b0303,
        0x050b0505,0x050b070f,0x050b0901,0x050b0b07,0x050b0f01,0x050d0101,0x050d0105,0x050d010f,
        0x050d0503,0x050d0b0b,0x050d0d03,0x050f010b,0x050f0303,0x050f050d,0x050f0701,0x050f0907,
        0x050f0b01,0x07010105,0x07010303,0x07010307,0x0701030b,0x0701030f,0x07010505,0x07010703,
        0x07010707,0x0701070b,0x07010905,0x07010909,0x0701090f,0x07010b03,0x07010d07,0x07010f03,
        0x07030103,0x07030107,0x0703010b,0x07030309,0x07030503,0x07030507,0x07030901,0x07030d01,
        0x07030f05,0x07030f0d,0x07050101,0x07050305,0x07050501,0x07050705,0x07050709,0x07050b01,
        0x07070103,0x07070301,0x07070309,0x07070503,0x07070507,0x0707050f,0x07070701,0x07070903,
        0x07070907,0x0707090f,0x07070b0b,0x07070f07,0x07090107,0x07090303,0x0709030d,0x07090505,
        0x07090703,0x07090b05,0x07090d01,0x07090d09,0x070b0103,0x070b0301,0x070b0305,0x070b050b,
        0x070b0705,0x070b0909,0x070b0b0d,0x070b0f07,0x070d030d,0x070d0903,0x070f0103,0x070f0107,
        0x070f0501,0x070f0505,0x070f070b,0x09010101,0x09010109,0x09010305,0x09010501,0x09010509,
        0x0901050f,0x09010705,0x09010903,0x09010b01,0x09010f01,0x09030105,0x0903010f,0x09030303,
        0x09030307,0x09030505,0x09030701,0x0903070b,0x09030907,0x09030b03,0x09030b0b,0x09050103,
        0x09050107,0x09050301,0x0905030b,0x09050503,0x09050707,0x09050901,0x09050b0f,0x09050d05,
        0x09050f01,0x09070109,0x09070303,0x09070307,0x09070501,0x09070505,0x09070703,0x0907070b,
        0x09090101,0x09090105,0x09090509,0x0909070f,0x09090901,0x09090f03,0x090b010b,0x090b010f,
        0x090b0503,0x090b0d05,0x090d0307,0x090d0709,0x090d0d01,0x090f0301,0x090f030b,0x090f0701,
        0x090f0907,0x090f0b03,0x0b010105,0x0b010301,0x0b010309,0x0b010505,0x0b010901,0x0b010909,
        0x0b01090f,0x0b010b05,0x0b010d0d,0x0b010f09,0x0b030103,0x0b030107,0x0b03010b,0x0b030305,
        0x0b030503,0x0b030705,0x0b030f05,0x0b050101,0x0b050303,0x0b050507,0x0b050701,0x0b05070d,
        0x0b050b07,0x0b070105,0x0b07010f,0x0b070301,0x0b07050f,0x0b070909,0x0b070b03,0x0b070d0b,
        0x0b070f07,0x0b090103,0x0b090109,0x0b090501,0x0b090705,0x0b09090d,0x0b0b0305,0x0b0b050d,
        0x0b0b0b03,0x0b0b0b07,0x0b0d0905,0x0b0f0105,0x0b0f0109,0x0b0f0505,0x0d010303,0x0d010307,
        0x0d01030b,0x0d010703,0x0d010707,0x0d010d01,0x0d030101,0x0d030501,0x0d03050f,0x0d030d09,
        0x0d050305,0x0d050709,0x0d050905,0x0d050b0b,0x0d050d05,0x0d050f01,0x0d070101,0x0d070309,
        0x0d070503,0x0d070901,0x0d09050b,0x0d090907,0x0d090d05,0x0d0b0101,0x0d0b0107,0x0d0b0709,
        0x0d0b0d01,0x0d0d010b,0x0d0d0901,0x0d0f0303,0x0d0f0307,0x0f010101,0x0f010109,0x0f01010f,
        0x0f010501,0x0f010505,0x0f01070d,0x0f010901,0x0f010b09,0x0f010d05,0x0f030105,0x0f030303,
        0x0f030509,0x0f030907,0x0f03090b,0x0f050103,0x0f050109,0x0f050301,0x0f05030d,0x0f050503,
        0x0f050701,0x0f050b03,0x0f070105,0x0f070705,0x0f07070b,0x0f070b07,0x0f090103,0x0f09010b,
        0x0f090307,0x0f090501,0x0f090b01,0x0f0b0505,0x0f0b0905,0x0f0d0105,0x0f0d0703,0x0f0f0101
    };

    const int BLOCK_N = 32;
    __shared__ float a_tile[256];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int kk = p * 32 + threadIdx.x;
            a_tile[kk] = a[bi * 256 + kk];
        }
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w + (size_t)ni * n_blocks * 110 + (size_t)bi * 110;
            unsigned short d_bits = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const unsigned char* qs     = base + 2;       // 64 bytes
            const unsigned char* qh     = base + 2 + 64;  // 8 bytes
            const unsigned char* signs  = base + 2 + 64 + 8;   // 32 bytes
            const unsigned char* scales = base + 2 + 64 + 8 + 32; // 4 bytes

            #pragma unroll
            for (int ib32 = 0; ib32 < 8; ib32++) {
                unsigned int scale_nib =
                    ((unsigned int)scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xFu;
                float db = d * (float)(1 + 2 * (int)scale_nib);
                unsigned int qh_byte = (unsigned int)qh[ib32];

                #pragma unroll
                for (int l = 0; l < 4; l++) {
                    unsigned int idx1 =
                          (unsigned int)qs[ib32 * 8 + 2 * l + 0]
                        | ((qh_byte << (8 - 2 * l)) & 256u);
                    unsigned int idx2 =
                          (unsigned int)qs[ib32 * 8 + 2 * l + 1]
                        | ((qh_byte << (7 - 2 * l)) & 256u);
                    unsigned int grid1 = iq3s_grid[idx1];
                    unsigned int grid2 = iq3s_grid[idx2];
                    unsigned int sign  = (unsigned int)signs[ib32 * 4 + l];

                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        unsigned int q0 = (grid1 >> (8 * j)) & 0xFFu;
                        unsigned int q1 = (grid2 >> (8 * j)) & 0xFFu;
                        float s0 = (sign & (1u << (j + 0))) ? -1.0f : 1.0f;
                        float s1 = (sign & (1u << (j + 4))) ? -1.0f : 1.0f;
                        acc += a_tile[32 * ib32 + 8 * l + j + 0] * (db * (float)q0 * s0);
                        acc += a_tile[32 * ib32 + 8 * l + j + 4] * (db * (float)q1 * s1);
                    }
                }
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-q8_0-fwd — FUSED Q8_0 matmul.  Used by cnc's V2-Lite Q4_K_M
// for its ffn_down tensors (the unaligned d_ff=10944 / expert_ff_dim=1408
// dimensions; llama.cpp's quantiser falls back to Q8_0 there because Q4_K's
// 256-element super-block doesn't align to those dims).
//
// Q8_0 block layout (34 bytes per 32-element block):
//   bytes 0-1   : f16 scale `d`
//   bytes 2-33  : 32 signed-int8 quants
//   dequant: y_i = d * (int8)qs[i]
extern "C" __global__ void fused_q8_0_matmul_seq1(
    const float*         __restrict__ a,           // [k]
    const unsigned char* __restrict__ w,           // [n * n_blocks * 34]
    float*               __restrict__ out,         // [n]
    int n, int n_blocks)                           // k = n_blocks * 32
{
    const int BLOCK_N = 32;
    __shared__ float a_tile[32];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 32 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base =
                w + (size_t)ni * n_blocks * 34 + (size_t)bi * 34;
            unsigned short d_bits =
                ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const signed char* qs = (const signed char*)(base + 2);

            #pragma unroll
            for (int i = 0; i < 32; i++) {
                float w_val = d * (float)(int)qs[i];
                acc += a_tile[i] * w_val;
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-q5_0-fwd — FUSED Q5_0 matmul.  Companion to Q8_0; cnc's
// V2-Lite Q4_K_M uses Q5_0 for ~half of its ffn_down_exps tensors.
//
// Q5_0 block layout (22 bytes per 32-element block):
//   bytes 0-1   : f16 scale `d`
//   bytes 2-5   : 4-byte qh (32 high-bits, one per element, little-endian u32)
//   bytes 6-21  : 16-byte ql (nibble-packed low 4 bits, byte i holds
//                 low-nibble for elem i AND high-nibble for elem i+16)
//   dequant: q_lo = (ql[i] & 0xF) | ((qh >> i)        & 1) << 4
//            q_hi = (ql[i] >> 4)  | ((qh >> (i + 16)) & 1) << 4
//            y_at_i      = d * (q_lo - 16)
//            y_at_i_p_16 = d * (q_hi - 16)
extern "C" __global__ void fused_q5_0_matmul_seq1(
    const float*         __restrict__ a,           // [k]
    const unsigned char* __restrict__ w,           // [n * n_blocks * 22]
    float*               __restrict__ out,         // [n]
    int n, int n_blocks)                           // k = n_blocks * 32
{
    const int BLOCK_N = 32;
    __shared__ float a_tile[32];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 32 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base =
                w + (size_t)ni * n_blocks * 22 + (size_t)bi * 22;
            unsigned short d_bits =
                ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            unsigned int qh =
                  (unsigned int)base[2]
                | ((unsigned int)base[3] << 8)
                | ((unsigned int)base[4] << 16)
                | ((unsigned int)base[5] << 24);
            const unsigned char* ql = base + 6;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned char byte = ql[i];
                int q_lo = (int)((byte & 0xFu) | (((qh >> i) & 1u) << 4)) - 16;
                int q_hi = (int)(((byte >> 4) & 0xFu)
                                  | (((qh >> (i + 16)) & 1u) << 4)) - 16;
                float w_lo = d * (float)q_lo;
                float w_hi = d * (float)q_hi;
                acc += a_tile[i]      * w_lo;
                acc += a_tile[i + 16] * w_hi;
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-q4_0-fwd — FUSED Q4_0 matmul for the local DeepSeek-V2-Lite
// GGUF (which ships in Q4_0 not Q4_K) and other Q4_0-quantised models.
//
// Q4_0 block layout (18 bytes per 32-element block):
//   bytes 0-1   : f16 scale `d`
//   bytes 2-17  : 16 bytes of nibble-packed quants
//     byte i (i in [0,16)) holds:
//       low  nibble → quant for position i       (range [-8, +7])
//       high nibble → quant for position i + 16  (range [-8, +7])
//   dequant: y_j = d * (q_j - 8) where q_j is the unsigned 4-bit value.
//
// Threading: BLOCK_N=32 threads per CTA, each thread owns ONE output row.
// Grid x = ceil(n / 32).  Each block iteration:
//   - cooperatively load 32 floats of A into shared mem (one per thread)
//   - each thread dequantises its row's Q4_0 block (16 nibble pairs) and
//     fma's into its accumulator.
extern "C" __global__ void fused_q4_0_matmul_seq1(
    const float*         __restrict__ a,           // [k]
    const unsigned char* __restrict__ w,           // [n * n_blocks * 18]
    float*               __restrict__ out,         // [n]
    int n, int n_blocks)                           // k = n_blocks * 32
{
    const int BLOCK_N = 32;
    __shared__ float a_tile[32];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        // Each thread loads ONE element of A's 32-elem chunk.
        a_tile[threadIdx.x] = a[bi * 32 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base =
                w + (size_t)ni * n_blocks * 18 + (size_t)bi * 18;
            unsigned short d_bits =
                ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            float d = aether_f16_to_f32_dev(d_bits);
            const unsigned char* qs = base + 2;

            #pragma unroll
            for (int i = 0; i < 16; i++) {
                unsigned char byte = qs[i];
                int q_lo = (int)(byte & 0xFu) - 8;
                int q_hi = (int)((byte >> 4) & 0xFu) - 8;
                float w_lo = d * (float)q_lo;
                float w_hi = d * (float)q_hi;
                acc += a_tile[i] * w_lo;
                acc += a_tile[i + 16] * w_hi;
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-q5_k-fwd — FUSED Q5_K matmul.  Same shape as fused_q4k_matmul_seq1
// but with 5-bit quants (4 nibble + 1 qh bit) instead of 4-bit.
//
// Q5_K block layout (176 bytes per 256-elem super-block):
//   bytes 0-1     : f16 super-block scale `d`
//   bytes 2-3     : f16 super-block min `dmin`
//   bytes 4-15    : 12 bytes scales (8 × {6-bit scale, 6-bit min}, packed
//                   identically to Q4_K — q4k_get_scale/min work here too)
//   bytes 16-47   : 32 bytes qh (high bits, one per element)
//   bytes 48-175  : 128 bytes qs (nibble-packed low 4 bits, 2 elems per byte)
//
// Dequant for sub-block sub in [0, 8), elem l in [0, 32):
//   scale_eff = d    * q4k_get_scale(sub, scales)
//   min_eff   = dmin * q4k_get_min  (sub, scales)
//   byte      = qs[(sub >> 1) * 32 + l]
//   nibble    = (sub & 1) ? (byte >> 4 & 0xF) : (byte & 0xF)
//   hi_bit    = (qh[l] >> sub) & 1
//   quant     = nibble | (hi_bit << 4)             ∈ [0, 31]
//   y         = scale_eff * quant - min_eff
extern "C" __global__ void fused_q5_k_matmul_seq1(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks)
{
    const int BLOCK_N = 32;
    __shared__ float a_tile[256];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int kk = p * BLOCK_N + threadIdx.x;
            a_tile[kk] = a[bi * 256 + kk];
        }
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w + (size_t)ni * n_blocks * 176 + (size_t)bi * 176;
            unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* scales = base + 4;     // 12 bytes
            const unsigned char* qh     = base + 16;    // 32 bytes
            const unsigned char* qs     = base + 48;    // 128 bytes

            #pragma unroll
            for (int sub = 0; sub < 8; sub++) {
                int j = sub >> 1;
                int is_hi = sub & 1;
                unsigned int sc = q4k_get_scale(sub, scales);
                unsigned int mn = q4k_get_min(sub, scales);
                float d_eff = d * (float)sc;
                float m_eff = dmin * (float)mn;
                int qs_off = j * 32;
                #pragma unroll 8
                for (int l = 0; l < 32; l++) {
                    unsigned char byte = qs[qs_off + l];
                    unsigned int nibble = is_hi
                        ? (((unsigned int)byte >> 4) & 0xFu)
                        : ((unsigned int)byte & 0xFu);
                    unsigned int hi_bit = ((unsigned int)qh[l] >> sub) & 1u;
                    unsigned int quant  = nibble | (hi_bit << 4);   // 5-bit in [0, 31]
                    float w_val = d_eff * (float)quant - m_eff;
                    acc += a_tile[sub * 32 + l] * w_val;
                }
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// FR-17-extra-q3_k-fwd — FUSED Q3_K matmul (3-bit quants).  Unblocks
// Qwen3-MoE Q3_K_M which has 198 Q3_K tensors out of 579.  Block layout
// matches ggml's `block_q3_K` (110 bytes per 256-elem super-block):
//
//   bytes 0-31    : hmask (32 bytes)  — 1 high bit per quant
//   bytes 32-95   : qs    (64 bytes)  — 2 low bits per quant (4 per byte)
//   bytes 96-107  : scales (12 bytes) — 16 packed 6-bit signed scales
//   bytes 108-109 : d      (f16)      — super-block scale
//
// Dequant (matches ggml-quants.c dequantize_row_q3_K exactly):
//   1. Unpack 16 signed 6-bit scales via the kmask1=0x03030303 /
//      kmask2=0x0f0f0f0f trick used by reference C.  Subtract 32 → range
//      [-32, +31].
//   2. Per element i in [0, 256):
//        sub = which scale (16 scales × 16 elems = 256)
//        low_2 = (qs[?] >> shift) & 3
//        high_bit = (hmask[?] & m) != 0
//        signed_3bit = high_bit ? low_2 : low_2 - 4   ∈ {-4..-1, 0..3}
//        y = d_all * (scale[sub] - 32) * signed_3bit
//
// Threading: BLOCK_N=32 threads per CTA, each thread owns one output row.
// Grid x = ceil(n / 32).  Mirrors fused_q4k_matmul_seq1's shape so the
// dispatch + perf characteristics are familiar.
extern "C" __global__ void fused_q3_k_matmul_seq1(
    const float*         __restrict__ a,           // [k]
    const unsigned char* __restrict__ w,           // n rows of (n_blocks * 110) bytes
    float*               __restrict__ out,         // [n]
    int n, int n_blocks)                           // k = n_blocks * 256
{
    const int BLOCK_N = 32;
    __shared__ float a_tile[256];

    int ni = blockIdx.x * BLOCK_N + threadIdx.x;
    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        // Cooperatively load 256 floats of A: each of 32 threads loads 8.
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            int kk = p * BLOCK_N + threadIdx.x;
            a_tile[kk] = a[bi * 256 + kk];
        }
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w + (size_t)ni * n_blocks * 110 + (size_t)bi * 110;
            const unsigned char* hm  = base;          // [32]
            const unsigned char* qs  = base + 32;     // [64]
            const unsigned char* sc  = base + 96;     // [12]
            unsigned short d_bits =
                ((unsigned short)base[109] << 8) | (unsigned short)base[108];
            float d_all = aether_f16_to_f32_dev(d_bits);

            // Unpack 16 signed 6-bit scales via kmask1/kmask2 trick.
            // Mirrors ggml-quants.c lines that build `scales[0..15]` as
            // signed 8-bit values (range will be 0..63, caller subtracts 32).
            int scales[16];
            {
                unsigned int aux0 = (unsigned int)sc[0]
                                  | ((unsigned int)sc[1] << 8)
                                  | ((unsigned int)sc[2] << 16)
                                  | ((unsigned int)sc[3] << 24);
                unsigned int aux1 = (unsigned int)sc[4]
                                  | ((unsigned int)sc[5] << 8)
                                  | ((unsigned int)sc[6] << 16)
                                  | ((unsigned int)sc[7] << 24);
                unsigned int aux2 = (unsigned int)sc[8]
                                  | ((unsigned int)sc[9] << 8)
                                  | ((unsigned int)sc[10] << 16)
                                  | ((unsigned int)sc[11] << 24);
                unsigned int tmp  = aux2;
                unsigned int km1  = 0x03030303u;
                unsigned int km2  = 0x0f0f0f0fu;
                unsigned int a0 = (aux0 & km2) | (((tmp >> 0) & km1) << 4);
                unsigned int a1 = (aux1 & km2) | (((tmp >> 2) & km1) << 4);
                unsigned int a2 = ((aux0 >> 4) & km2) | (((tmp >> 4) & km1) << 4);
                unsigned int a3 = ((aux1 >> 4) & km2) | (((tmp >> 6) & km1) << 4);
                scales[0]  = (int)( a0        & 0xFFu);
                scales[1]  = (int)((a0 >>  8) & 0xFFu);
                scales[2]  = (int)((a0 >> 16) & 0xFFu);
                scales[3]  = (int)((a0 >> 24) & 0xFFu);
                scales[4]  = (int)( a1        & 0xFFu);
                scales[5]  = (int)((a1 >>  8) & 0xFFu);
                scales[6]  = (int)((a1 >> 16) & 0xFFu);
                scales[7]  = (int)((a1 >> 24) & 0xFFu);
                scales[8]  = (int)( a2        & 0xFFu);
                scales[9]  = (int)((a2 >>  8) & 0xFFu);
                scales[10] = (int)((a2 >> 16) & 0xFFu);
                scales[11] = (int)((a2 >> 24) & 0xFFu);
                scales[12] = (int)( a3        & 0xFFu);
                scales[13] = (int)((a3 >>  8) & 0xFFu);
                scales[14] = (int)((a3 >> 16) & 0xFFu);
                scales[15] = (int)((a3 >> 24) & 0xFFu);
            }

            // ggml outer loop:  n_outer over {0, 128}.  Inside, j in [0,4)
            // with shift += 2 and m <<= 1 (m NOT reset between n_outer
            // iterations — it advances bit positions 0..7 of hmask).
            int is = 0;
            int a_idx = 0;
            unsigned char m = 1u;
            for (int n_outer = 0; n_outer < 256; n_outer += 128) {
                int shift = 0;
                int qs_off = (n_outer == 0) ? 0 : 32;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    float dl_lo = d_all * (float)(scales[is++] - 32);
                    #pragma unroll
                    for (int l = 0; l < 16; l++) {
                        int q2  = (int)((qs[qs_off + l] >> shift) & 3u);
                        int sub = (hm[l] & m) ? 0 : 4;
                        acc += a_tile[a_idx++] * (dl_lo * (float)(q2 - sub));
                    }
                    float dl_hi = d_all * (float)(scales[is++] - 32);
                    #pragma unroll
                    for (int l = 0; l < 16; l++) {
                        int q2  = (int)((qs[qs_off + 16 + l] >> shift) & 3u);
                        int sub = (hm[16 + l] & m) ? 0 : 4;
                        acc += a_tile[a_idx++] * (dl_hi * (float)(q2 - sub));
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
        }
        __syncthreads();
    }

    if (ni < n) out[ni] = acc;
}

// matt-voice / FR-17.14-extra-deepest — FUSED Q4_K matmul v2 (split-K).
//
// Each WARP owns one output column. Within a warp, the 32 lanes
// cooperatively process the K dim, then warp-reduce via __shfl_down_sync.
//
// CTA = 8 warps = 256 threads. Each CTA processes 8 output columns.
// At N=512 (Qwen W_k), that's 64 CTAs = 16K threads -- saturates the
// GPU. v1 was 16 CTAs * 32 threads = 512 threads at N=512 (under-uses).
//
// Per K-tile (256 quants, one super-block):
//   - CTA cooperatively loads 256 floats of A into shared mem
//     (1 element per thread, perfectly coalesced)
//   - Per warp: each of 32 lanes handles 8 quants of the 256
//     - lane l owns sub-block (l/4), sub_offset (l%4)*8
//     - reads 8 bytes of qs, dequants 8 nibbles, fmas with 8 floats of A
//   - At the END (after all K-tiles), warp-reduce the 32 partials.
//   - Lane 0 writes the output.
//
// Branch divergence per warp: only 2-way (lanes 0..3 + 8..11 + ... take
// is_hi=0; lanes 4..7 + 12..15 + ... take is_hi=1). NVCC predicates this
// efficiently. No __syncthreads inside the warp loop -- only one
// sync per K-tile to coordinate the shared A load.
extern "C" __global__ void fused_q4k_matmul_seq1_v2(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks)
{
    __shared__ float a_tile[256];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int ni = blockIdx.x * 8 + warp;

    float acc = 0.0f;

    // Per-lane sub-block assignment (constant across K-tiles)
    int sub      = lane >> 2;       // 0..7
    int sub_off  = (lane & 3) << 3; // 0, 8, 16, 24
    int j        = sub >> 1;
    int is_hi    = sub & 1;
    int qs_off   = j * 32 + sub_off;
    int a_off    = sub * 32 + sub_off;

    for (int bi = 0; bi < n_blocks; bi++) {
        // CTA-wide cooperative load of A tile (256 floats).
        a_tile[threadIdx.x] = a[bi * 256 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w
                + (size_t)ni * n_blocks * 144
                + (size_t)bi * 144;
            unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* scales = base + 4;
            const unsigned char* qs     = base + 16;
            unsigned int sc = q4k_get_scale(sub, scales);
            unsigned int mn = q4k_get_min(sub, scales);
            float d_eff = d * (float)sc;
            float m_eff = dmin * (float)mn;

            #pragma unroll
            for (int p = 0; p < 8; p++) {
                unsigned char byte = qs[qs_off + p];
                unsigned int nibble = is_hi ? (((unsigned int)byte >> 4) & 0xFu) : ((unsigned int)byte & 0xFu);
                float w_val = d_eff * (float)nibble - m_eff;
                acc += a_tile[a_off + p] * w_val;
            }
        }
        __syncthreads();
    }

    // Warp-reduce 32 partial sums into lane 0.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0 && ni < n) {
        out[ni] = acc;
    }
}

// matt-voice / FR-17.14-extra-deepest-v3 -- byte-once Q4_K matmul.
//
// The v2 kernel had 32 lanes × 8-byte reads = 256 byte-reads per warp
// per K-tile, but each qs byte was actually read TWICE: sub=2g and
// sub=2g+1 share the same 32 bytes of qs (low vs high nibble).
//
// v3 reads each byte ONCE per warp and uses BOTH nibbles within the
// same lane. Each lane handles 1 byte per inner iteration with both
// nibbles contributing to its accumulator. 4 inner iterations cover
// the full 128-byte qs (32 lanes × 4 = 128). Reads per warp per
// K-tile go from 256 to 128 → halves memory-instruction throughput,
// closer to DRAM-BW-bound.
//
// FMA count is unchanged (256 quants × n_blocks per output). Lane
// mapping is uniform within iter (sub_lo/sub_hi identical across all
// lanes for a given i), so zero warp divergence.
extern "C" __global__ void fused_q4k_matmul_seq1_v3(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks)
{
    __shared__ float a_tile[256];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int ni = blockIdx.x * 8 + warp;

    // Split accumulator to expose ILP: NVCC can interleave the two
    // independent FMA chains rather than serializing through one acc.
    float acc_lo = 0.0f;
    float acc_hi = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 256 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w
                + (size_t)ni * n_blocks * 144
                + (size_t)bi * 144;
            unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
            unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
            float d    = aether_f16_to_f32_dev(d_bits);
            float dmin = aether_f16_to_f32_dev(dmin_bits);
            const unsigned char* scales = base + 4;
            const unsigned char* qs     = base + 16;

            float d_eff[8], m_eff[8];
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                unsigned int sc = q4k_get_scale(s, scales);
                unsigned int mn = q4k_get_min(s, scales);
                d_eff[s] = d * (float)sc;
                m_eff[s] = dmin * (float)mn;
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int sub_lo = i * 2;
                int sub_hi = i * 2 + 1;
                unsigned char byte = qs[i * 32 + lane];
                unsigned int nib_lo = ((unsigned int)byte) & 0xFu;
                unsigned int nib_hi = (((unsigned int)byte) >> 4) & 0xFu;

                float w_lo = d_eff[sub_lo] * (float)nib_lo - m_eff[sub_lo];
                float w_hi = d_eff[sub_hi] * (float)nib_hi - m_eff[sub_hi];

                int k_lo = sub_lo * 32 + lane;
                int k_hi = sub_hi * 32 + lane;

                acc_lo += a_tile[k_lo] * w_lo;
                acc_hi += a_tile[k_hi] * w_hi;
            }
        }
        __syncthreads();
    }

    float acc = acc_lo + acc_hi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0 && ni < n) {
        out[ni] = acc;
    }
}

// matt-voice / FR-17.14-extra-deepest-v3 -- FUSED FFN gate+up+silu+mul.
//
// Replaces 4 separate kernels (gate matmul, up matmul, silu, mul_inplace)
// with one. For each output index ni:
//   gate[ni] = sum_k a[k] * W_gate[ni, k]   (Q4_K)
//   up[ni]   = sum_k a[k] * W_up[ni, k]     (Q4_K)
//   out[ni]  = silu(gate[ni]) * up[ni]
//
// Both gate and up share the same x_norm input -- loading it into shmem
// once and using it for both halves of the FMA cuts a_tile traffic 2x.
// Each warp computes BOTH gate[ni] and up[ni] in parallel by maintaining
// two accumulators and reading both weight rows per K-tile.
//
// Same CTA layout as v2: 256 threads / 8 warps / 1 output per warp.
extern "C" __global__ void fused_q4k_ffn_gate_up_silu_mul(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_up,
    float*               __restrict__ out,
    int n, int n_blocks)
{
    __shared__ float a_tile[256];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int ni = blockIdx.x * 8 + warp;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    int sub      = lane >> 2;
    int sub_off  = (lane & 3) << 3;
    int j        = sub >> 1;
    int is_hi    = sub & 1;
    int qs_off   = j * 32 + sub_off;
    int a_off    = sub * 32 + sub_off;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 256 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            // --- gate row ---
            {
                const unsigned char* base = w_gate
                    + (size_t)ni * n_blocks * 144
                    + (size_t)bi * 144;
                unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
                unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
                float d    = aether_f16_to_f32_dev(d_bits);
                float dmin = aether_f16_to_f32_dev(dmin_bits);
                const unsigned char* scales = base + 4;
                const unsigned char* qs     = base + 16;
                unsigned int sc = q4k_get_scale(sub, scales);
                unsigned int mn = q4k_get_min(sub, scales);
                float d_eff = d * (float)sc;
                float m_eff = dmin * (float)mn;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    unsigned char byte = qs[qs_off + p];
                    unsigned int nibble = is_hi ? (((unsigned int)byte >> 4) & 0xFu) : ((unsigned int)byte & 0xFu);
                    float w_val = d_eff * (float)nibble - m_eff;
                    acc_g += a_tile[a_off + p] * w_val;
                }
            }
            // --- up row ---
            {
                const unsigned char* base = w_up
                    + (size_t)ni * n_blocks * 144
                    + (size_t)bi * 144;
                unsigned short d_bits    = ((unsigned short)base[1] << 8) | (unsigned short)base[0];
                unsigned short dmin_bits = ((unsigned short)base[3] << 8) | (unsigned short)base[2];
                float d    = aether_f16_to_f32_dev(d_bits);
                float dmin = aether_f16_to_f32_dev(dmin_bits);
                const unsigned char* scales = base + 4;
                const unsigned char* qs     = base + 16;
                unsigned int sc = q4k_get_scale(sub, scales);
                unsigned int mn = q4k_get_min(sub, scales);
                float d_eff = d * (float)sc;
                float m_eff = dmin * (float)mn;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    unsigned char byte = qs[qs_off + p];
                    unsigned int nibble = is_hi ? (((unsigned int)byte >> 4) & 0xFu) : ((unsigned int)byte & 0xFu);
                    float w_val = d_eff * (float)nibble - m_eff;
                    acc_u += a_tile[a_off + p] * w_val;
                }
            }
        }
        __syncthreads();
    }

    // Warp-reduce both partials.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, offset);
        acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, offset);
    }
    if (lane == 0 && ni < n) {
        // silu(g) * u = (g / (1 + exp(-g))) * u
        float silu_g = acc_g / (1.0f + expf(-acc_g));
        out[ni] = silu_g * acc_u;
    }
}

// matt-voice / FR-17.14-extra-deepest-v3 -- FUSED FFN byte-once.
//
// Same byte-once layout as fused_q4k_matmul_seq1_v3 but computing
// both gate[ni] and up[ni] simultaneously, then applying silu*mul.
extern "C" __global__ void fused_q4k_ffn_gate_up_silu_mul_v2(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w_gate,
    const unsigned char* __restrict__ w_up,
    float*               __restrict__ out,
    int n, int n_blocks)
{
    __shared__ float a_tile[256];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int ni = blockIdx.x * 8 + warp;

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 256 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            // Two pointers, one per weight tensor.
            const unsigned char* base_g = w_gate + (size_t)ni * n_blocks * 144 + (size_t)bi * 144;
            const unsigned char* base_u = w_up   + (size_t)ni * n_blocks * 144 + (size_t)bi * 144;
            unsigned short dg    = ((unsigned short)base_g[1] << 8) | (unsigned short)base_g[0];
            unsigned short dmg   = ((unsigned short)base_g[3] << 8) | (unsigned short)base_g[2];
            unsigned short du    = ((unsigned short)base_u[1] << 8) | (unsigned short)base_u[0];
            unsigned short dmu   = ((unsigned short)base_u[3] << 8) | (unsigned short)base_u[2];
            float d_g    = aether_f16_to_f32_dev(dg);
            float dmin_g = aether_f16_to_f32_dev(dmg);
            float d_u    = aether_f16_to_f32_dev(du);
            float dmin_u = aether_f16_to_f32_dev(dmu);
            const unsigned char* scales_g = base_g + 4;
            const unsigned char* qs_g     = base_g + 16;
            const unsigned char* scales_u = base_u + 4;
            const unsigned char* qs_u     = base_u + 16;

            // Hoist 8 (d_eff, m_eff) pairs per weight tensor.
            float gd_eff[8], gm_eff[8], ud_eff[8], um_eff[8];
            #pragma unroll
            for (int s = 0; s < 8; s++) {
                unsigned int gsc = q4k_get_scale(s, scales_g);
                unsigned int gmn = q4k_get_min  (s, scales_g);
                unsigned int usc = q4k_get_scale(s, scales_u);
                unsigned int umn = q4k_get_min  (s, scales_u);
                gd_eff[s] = d_g    * (float)gsc;
                gm_eff[s] = dmin_g * (float)gmn;
                ud_eff[s] = d_u    * (float)usc;
                um_eff[s] = dmin_u * (float)umn;
            }

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int sub_lo = i * 2;
                int sub_hi = i * 2 + 1;
                unsigned char byte_g = qs_g[i * 32 + lane];
                unsigned char byte_u = qs_u[i * 32 + lane];
                unsigned int g_lo = ((unsigned int)byte_g) & 0xFu;
                unsigned int g_hi = (((unsigned int)byte_g) >> 4) & 0xFu;
                unsigned int u_lo = ((unsigned int)byte_u) & 0xFu;
                unsigned int u_hi = (((unsigned int)byte_u) >> 4) & 0xFu;

                int k_lo = sub_lo * 32 + lane;
                int k_hi = sub_hi * 32 + lane;
                float a_lo = a_tile[k_lo];
                float a_hi = a_tile[k_hi];

                acc_g += a_lo * (gd_eff[sub_lo] * (float)g_lo - gm_eff[sub_lo]);
                acc_g += a_hi * (gd_eff[sub_hi] * (float)g_hi - gm_eff[sub_hi]);
                acc_u += a_lo * (ud_eff[sub_lo] * (float)u_lo - um_eff[sub_lo]);
                acc_u += a_hi * (ud_eff[sub_hi] * (float)u_hi - um_eff[sub_hi]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_g += __shfl_down_sync(0xFFFFFFFFu, acc_g, offset);
        acc_u += __shfl_down_sync(0xFFFFFFFFu, acc_u, offset);
    }
    if (lane == 0 && ni < n) {
        float silu_g = acc_g / (1.0f + expf(-acc_g));
        out[ni] = silu_g * acc_u;
    }
}

// matt-voice / FR-17.13-extra — append new K/V step to the on-device
// KV cache at position `pos`. Simple memcpy-shaped kernel.
extern "C" __global__ void append_kv(
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    float*       __restrict__ k_cache,
    float*       __restrict__ v_cache,
    int pos, int d_kv)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_kv) return;
    k_cache[(size_t)pos * d_kv + tid] = k_new[tid];
    v_cache[(size_t)pos * d_kv + tid] = v_new[tid];
}

// FR-17.14-extra-deepest-graph -- append_kv variant reading pos from
// device memory (step_args[0]).
extern "C" __global__ void append_kv_devarg(
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    float*       __restrict__ k_cache,
    float*       __restrict__ v_cache,
    int d_kv, const int* __restrict__ step_args)
{
    int pos = step_args[0];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= d_kv) return;
    k_cache[(size_t)pos * d_kv + tid] = k_new[tid];
    v_cache[(size_t)pos * d_kv + tid] = v_new[tid];
}

// matt-voice / FR-17.13-extra — single-step attention for seq=1
// autoregressive generation with on-device KV cache.
//
// One warp per Q head. CTA = 32 threads (one warp). Each of the 32
// lanes handles head_dim/32 = 4 elements (for Qwen2.5 head_dim=128).
//
// Math per Q head h (kv_head = h / (n_q_heads/n_kv_heads)):
//   scores[t] = (Q_h · K_cache[t, kv_head]) * scale     for t in 0..cur_seq
//   softmax over scores[0..cur_seq]
//   attn_out[h, d] = sum_t softmax[t] * V_cache[t, kv_head, d]
//
// Shared memory: cur_seq * 4 bytes per CTA. Sized at launch time via
// dynamic shared mem.
extern "C" __global__ void attention_seq1(
    const float* __restrict__ q,         // [n_q_heads * head_dim]
    const float* __restrict__ k_cache,   // [max_seq, n_kv_heads * head_dim]
    const float* __restrict__ v_cache,   // [max_seq, n_kv_heads * head_dim]
    float*       __restrict__ attn_out,  // [n_q_heads * head_dim]
    int cur_seq, int n_q_heads, int n_kv_heads, int head_dim,
    float scale)
{
    extern __shared__ float scores[];

    int head    = blockIdx.x;
    int lane    = threadIdx.x;
    int kv_per_q = n_q_heads / n_kv_heads;
    int kv_head = head / kv_per_q;
    int d_kv    = n_kv_heads * head_dim;
    int per_lane = head_dim >> 5;     // head_dim / 32 (Qwen2.5: 4)

    const float* q_ptr = q + head * head_dim;

    // Load this head's Q row into registers (4 elements per lane).
    float q_local[8];  // up to head_dim=256
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) q_local[i] = q_ptr[lane * per_lane + i];
    }

    // === Pass 1: compute scores[t] = Q · K_cache[t, kv_head] * scale ===
    for (int t = 0; t < cur_seq; t++) {
        const float* k_ptr = k_cache + (size_t)t * d_kv + kv_head * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) acc += q_local[i] * k_ptr[lane * per_lane + i];
        }
        // Warp-reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    // === Pass 2: softmax (max, exp+sum, normalize) ===
    float local_max = __int_as_float(0xFF800000u);  // -inf (nvrtc has no INFINITY macro)
    for (int t = lane; t < cur_seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < cur_seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < cur_seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // === Pass 3: aggregate V_cache by softmax weights ===
    float out_local[8] = {0.0f};
    for (int t = 0; t < cur_seq; t++) {
        float w = scores[t];
        const float* v_ptr = v_cache + (size_t)t * d_kv + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) out_local[i] += w * v_ptr[lane * per_lane + i];
        }
    }

    float* out_ptr = attn_out + head * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) out_ptr[lane * per_lane + i] = out_local[i];
    }
}

// FR-17.14-extra-deepest-graph -- attention_seq1 variant reading cur_seq
// from device memory (step_args[1]). Allows the autoregressive forward
// pass to be captured into ONE CUDA graph: per step we update step_args
// via h2d and replay the graph. dyn shmem is launched at max_seq * 4
// bytes (we just don't use the tail beyond cur_seq).
extern "C" __global__ void attention_seq1_devarg(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float*       __restrict__ attn_out,
    int n_q_heads, int n_kv_heads, int head_dim,
    float scale, const int* __restrict__ step_args)
{
    int cur_seq = step_args[1];
    extern __shared__ float scores[];

    int head    = blockIdx.x;
    int lane    = threadIdx.x;
    int kv_per_q = n_q_heads / n_kv_heads;
    int kv_head = head / kv_per_q;
    int d_kv    = n_kv_heads * head_dim;
    int per_lane = head_dim >> 5;

    const float* q_ptr = q + head * head_dim;

    float q_local[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) q_local[i] = q_ptr[lane * per_lane + i];
    }

    for (int t = 0; t < cur_seq; t++) {
        const float* k_ptr = k_cache + (size_t)t * d_kv + kv_head * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) acc += q_local[i] * k_ptr[lane * per_lane + i];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < cur_seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < cur_seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < cur_seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    float out_local[8] = {0.0f};
    for (int t = 0; t < cur_seq; t++) {
        float w = scores[t];
        const float* v_ptr = v_cache + (size_t)t * d_kv + kv_head * head_dim;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < per_lane) out_local[i] += w * v_ptr[lane * per_lane + i];
        }
    }

    float* out_ptr = attn_out + head * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (i < per_lane) out_ptr[lane * per_lane + i] = out_local[i];
    }
}

// matt-voice / FR-17.14-extra-deepest — FUSED Q6_K matmul v2 for seq=1.
//
// Same pattern as fused_q4k_matmul_seq1_v2 but reading 210-byte
// Q6_K super-blocks instead of 144-byte Q4_K.
//
// Per Q6_K super-block (256 outputs):
//   - 2 n_outer halves, 4 sub_pos each = 8 (n_outer, sub_pos) combos
//   - Each combo covers 32 contiguous output positions
//
// Lane mapping: all 32 lanes execute the SAME (n_outer, sub_pos) per
// inner iteration. Each lane handles one quant. 8 iterations
// (2 × 4) cover all 256 quants. No warp divergence.
//
// Per lane per super-block: 8 fma + 8 byte reads from W. No diverging
// switch -- each iteration is a specialized code path because the
// outer (n_outer, sub_pos) loops are compile-time constants under
// #pragma unroll.
extern "C" __global__ void fused_q6k_matmul_seq1_v2(
    const float*         __restrict__ a,
    const unsigned char* __restrict__ w,
    float*               __restrict__ out,
    int n, int n_blocks)
{
    __shared__ float a_tile[256];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int ni = blockIdx.x * 8 + warp;

    float acc = 0.0f;

    for (int bi = 0; bi < n_blocks; bi++) {
        a_tile[threadIdx.x] = a[bi * 256 + threadIdx.x];
        __syncthreads();

        if (ni < n) {
            const unsigned char* base = w + (size_t)ni * n_blocks * 210 + (size_t)bi * 210;
            const unsigned char* ql = base;
            const unsigned char* qh = base + 128;
            const signed char*   sc = (const signed char*)(base + 192);
            unsigned short d_bits = ((unsigned short)base[209] << 8) | (unsigned short)base[208];
            float d = aether_f16_to_f32_dev(d_bits);

            // 2 halves × 4 sub_pos. With unrolled iteration the (n_outer,
            // sub_pos) values are compile-time constants, so the inner
            // if-else cascade becomes 8 separate specialised code paths.
            // All 32 lanes execute the same path at the same time = no
            // intra-warp divergence.
            #pragma unroll
            for (int n_outer = 0; n_outer < 2; n_outer++) {
                int ql_off = n_outer * 64;
                int qh_off = n_outer * 32;
                int sc_off = n_outer * 8;

                #pragma unroll
                for (int sub_pos = 0; sub_pos < 4; sub_pos++) {
                    int l_iter = lane;  // 0..31
                    int scale_idx = sc_off + (l_iter >> 4) + 2 * sub_pos;
                    float sc_val = (float)sc[scale_idx];

                    int q;
                    if (sub_pos == 0) {
                        unsigned char ql_byte = ql[ql_off + l_iter];
                        unsigned char qh_byte = qh[qh_off + l_iter];
                        q = (int)((ql_byte & 0xFu) | ((qh_byte & 3u) << 4)) - 32;
                    } else if (sub_pos == 1) {
                        unsigned char ql_byte = ql[ql_off + l_iter + 32];
                        unsigned char qh_byte = qh[qh_off + l_iter];
                        q = (int)((ql_byte & 0xFu) | (((qh_byte >> 2) & 3u) << 4)) - 32;
                    } else if (sub_pos == 2) {
                        unsigned char ql_byte = ql[ql_off + l_iter];
                        unsigned char qh_byte = qh[qh_off + l_iter];
                        q = (int)(((ql_byte >> 4) & 0xFu) | (((qh_byte >> 4) & 3u) << 4)) - 32;
                    } else {
                        unsigned char ql_byte = ql[ql_off + l_iter + 32];
                        unsigned char qh_byte = qh[qh_off + l_iter];
                        q = (int)(((ql_byte >> 4) & 0xFu) | (((qh_byte >> 6) & 3u) << 4)) - 32;
                    }
                    float w_val = d * sc_val * (float)q;
                    int a_idx = (n_outer * 128) + (sub_pos * 32) + l_iter;
                    acc += a_tile[a_idx] * w_val;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0 && ni < n) {
        out[ni] = acc;
    }
}

// FR-17-extra-moe-fwd — Q4_K matmul against ONE expert's slice of a
// concatenated MoE expert weight buffer (gate_exps / up_exps / down_exps).
// `expert_offset_blocks` selects the per-expert slice: each expert occupies
// `n_out * (n_in/256)` consecutive 256-elem super-blocks.  Otherwise
// identical to fused_q4k_matmul_seq1 (single-warp split-K design).
// Used by qwen3moe + deepseek2 MoE FFN.
extern "C" __global__ void fused_q4k_expert_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ w_base,
    float*               __restrict__ y,
    int n_out, int blocks_per_row, int expert_offset_blocks)
{
    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    // Per-expert weight base = w_base + expert_offset * 144 bytes.
    // Per-output-row base = expert_base + o * blocks_per_row * 144.
    size_t exp_off_bytes = (size_t)expert_offset_blocks * 144;
    size_t row_off_bytes = (size_t)o * (size_t)blocks_per_row * 144;
    const unsigned char* w_row = w_base + exp_off_bytes + row_off_bytes;
    float acc = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const unsigned char* blk = w_row + (size_t)b * 144;
        unsigned short d_bits  = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        unsigned short dmin_bits = ((unsigned short)blk[3] << 8) | (unsigned short)blk[2];
        float d    = aether_f16_to_f32_dev(d_bits);
        float dmin = aether_f16_to_f32_dev(dmin_bits);
        const unsigned char* scales = blk + 4;     // 12 bytes
        const unsigned char* qs     = blk + 16;    // 128 bytes
        const float* x_blk = x + (size_t)b * 256;
        float blk_acc = 0.0f;
        for (int si = 0; si < 8; si++) {
            int sc6, m6;
            if (si < 4) {
                sc6 = scales[si] & 0x3F;
                m6  = scales[si + 4] & 0x3F;
            } else {
                int j = si - 4;
                sc6 = ((scales[8 + j] & 0xF) | ((scales[j] >> 6) << 4)) & 0x3F;
                m6  = ((scales[8 + j] >> 4) | ((scales[j + 4] >> 6) << 4)) & 0x3F;
            }
            float scale = d * (float)sc6;
            float bias  = dmin * (float)m6;
            for (int k = 0; k < 32; k++) {
                int byte_idx = si * 16 + (k >> 1);
                unsigned char by = qs[byte_idx];
                int q = (k & 1) == 0 ? (by & 0xF) : (by >> 4);
                float dq = scale * (float)q - bias;
                blk_acc += x_blk[si * 32 + k] * dq;
            }
        }
        acc += blk_acc;
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-mla-fwd MoE expert-variant of Q8_0 matmul.  Mirrors
// fused_q4k_expert_matmul_seq1 but for Q8_0 weights (34-byte 32-elem
// blocks).  Per-expert offset = `expert_offset_blocks * 34` bytes from
// w_base.  One CTA per output row; 256 threads stride-loop the blocks.
extern "C" __global__ void fused_q8_0_expert_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ w_base,
    float*               __restrict__ y,
    int n_out, int blocks_per_row, int expert_offset_blocks)
{
    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    size_t exp_off_bytes = (size_t)expert_offset_blocks * 34;
    size_t row_off_bytes = (size_t)o * (size_t)blocks_per_row * 34;
    const unsigned char* w_row = w_base + exp_off_bytes + row_off_bytes;
    float acc = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const unsigned char* blk = w_row + (size_t)b * 34;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        const signed char* qs = (const signed char*)(blk + 2);
        const float* x_blk = x + (size_t)b * 32;
        float blk_acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            blk_acc += x_blk[k] * (d * (float)(int)qs[k]);
        }
        acc += blk_acc;
    }
    // Same warp-reduce-then-cross-warp-reduce as Q4_K expert kernel.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-mla-fwd MoE expert-variant of Q5_0 matmul.  22-byte 32-elem
// blocks: f16 d + 4-byte qh + 16-byte ql.  Same per-expert offset shape.
extern "C" __global__ void fused_q5_0_expert_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ w_base,
    float*               __restrict__ y,
    int n_out, int blocks_per_row, int expert_offset_blocks)
{
    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    size_t exp_off_bytes = (size_t)expert_offset_blocks * 22;
    size_t row_off_bytes = (size_t)o * (size_t)blocks_per_row * 22;
    const unsigned char* w_row = w_base + exp_off_bytes + row_off_bytes;
    float acc = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const unsigned char* blk = w_row + (size_t)b * 22;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        unsigned int qh = (unsigned int)blk[2]
            | ((unsigned int)blk[3] << 8)
            | ((unsigned int)blk[4] << 16)
            | ((unsigned int)blk[5] << 24);
        const unsigned char* ql = blk + 6;
        const float* x_blk = x + (size_t)b * 32;
        float blk_acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte = ql[i];
            int q_lo = (int)((byte & 0xFu) | (((qh >> i) & 1u) << 4)) - 16;
            int q_hi = (int)(((byte >> 4) & 0xFu)
                              | (((qh >> (i + 16)) & 1u) << 4)) - 16;
            blk_acc += x_blk[i]      * (d * (float)q_lo);
            blk_acc += x_blk[i + 16] * (d * (float)q_hi);
        }
        acc += blk_acc;
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-moe-quant-dispatch — MoE expert-variant of IQ3_S matmul.
// 110-byte 256-elem blocks (f16 d + 64-byte qs + 8-byte qh + 32-byte signs +
// 4-byte scales).  Per-expert offset = `expert_offset_blocks * 110` bytes
// from w_base.  Used by GLM-4.7-flash MoE expert tensors that quantise to
// IQ3_S (dt=21) — the dominant non-Q4_K dtype in the IQ3_XXS GGUF mix.
// One CTA per output row; 256 threads stride-loop the blocks; warp-reduce
// then cross-warp-reduce.
extern "C" __global__ void fused_iq3_s_expert_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ w_base,
    float*               __restrict__ y,
    int n_out, int blocks_per_row, int expert_offset_blocks)
{
    // Duplicate of the iq3s_grid table from fused_iq3_s_matmul_seq1.
    // Kept as a per-kernel `static const` so each kernel stays
    // self-contained (avoids cross-kernel symbol coupling in nvrtc).
    static const unsigned int iq3s_grid[512] = {
        0x01010101,0x01010103,0x01010105,0x0101010b,0x0101010f,0x01010301,0x01010303,0x01010305,
        0x01010309,0x0101030d,0x01010501,0x01010503,0x0101050b,0x01010707,0x01010901,0x01010905,
        0x0101090b,0x0101090f,0x01010b03,0x01010b07,0x01010d01,0x01010d05,0x01010f03,0x01010f09,
        0x01010f0f,0x01030101,0x01030103,0x01030105,0x01030109,0x01030301,0x01030303,0x0103030b,
        0x01030501,0x01030507,0x0103050f,0x01030703,0x0103070b,0x01030909,0x01030d03,0x01030d0b,
        0x01030f05,0x01050101,0x01050103,0x0105010b,0x0105010f,0x01050301,0x01050307,0x0105030d,
        0x01050503,0x0105050b,0x01050701,0x01050709,0x01050905,0x0105090b,0x0105090f,0x01050b03,
        0x01050b07,0x01050f01,0x01050f07,0x01070107,0x01070303,0x0107030b,0x01070501,0x01070505,
        0x01070703,0x01070707,0x0107070d,0x01070909,0x01070b01,0x01070b05,0x01070d0f,0x01070f03,
        0x01070f0b,0x01090101,0x01090307,0x0109030f,0x01090503,0x01090509,0x01090705,0x01090901,
        0x01090907,0x01090b03,0x01090f01,0x010b0105,0x010b0109,0x010b0501,0x010b0505,0x010b050d,
        0x010b0707,0x010b0903,0x010b090b,0x010b090f,0x010b0d0d,0x010b0f07,0x010d010d,0x010d0303,
        0x010d0307,0x010d0703,0x010d0b05,0x010d0f03,0x010f0101,0x010f0105,0x010f0109,0x010f0501,
        0x010f0505,0x010f050d,0x010f0707,0x010f0b01,0x010f0b09,0x03010101,0x03010103,0x03010105,
        0x03010109,0x03010301,0x03010303,0x03010307,0x0301030b,0x0301030f,0x03010501,0x03010505,
        0x03010703,0x03010709,0x0301070d,0x03010b09,0x03010b0d,0x03010d03,0x03010f05,0x03030101,
        0x03030103,0x03030107,0x0303010d,0x03030301,0x03030309,0x03030503,0x03030701,0x03030707,
        0x03030903,0x03030b01,0x03030b05,0x03030f01,0x03030f0d,0x03050101,0x03050305,0x0305030b,
        0x0305030f,0x03050501,0x03050509,0x03050705,0x03050901,0x03050907,0x03050b0b,0x03050d01,
        0x03050f05,0x03070103,0x03070109,0x0307010f,0x03070301,0x03070307,0x03070503,0x0307050f,
        0x03070701,0x03070709,0x03070903,0x03070d05,0x03070f01,0x03090107,0x0309010b,0x03090305,
        0x03090309,0x03090703,0x03090707,0x03090905,0x0309090d,0x03090b01,0x03090b09,0x030b0103,
        0x030b0301,0x030b0307,0x030b0503,0x030b0701,0x030b0705,0x030b0b03,0x030d0501,0x030d0509,
        0x030d050f,0x030d0909,0x030d090d,0x030f0103,0x030f0107,0x030f0301,0x030f0305,0x030f0503,
        0x030f070b,0x030f0903,0x030f0d05,0x030f0f01,0x05010101,0x05010103,0x05010107,0x0501010b,
        0x0501010f,0x05010301,0x05010305,0x05010309,0x0501030d,0x05010503,0x05010507,0x0501050f,
        0x05010701,0x05010705,0x05010903,0x05010907,0x0501090b,0x05010b01,0x05010b05,0x05010d0f,
        0x05010f01,0x05010f07,0x05010f0b,0x05030101,0x05030105,0x05030301,0x05030307,0x0503030f,
        0x05030505,0x0503050b,0x05030703,0x05030709,0x05030905,0x05030b03,0x05050103,0x05050109,
        0x0505010f,0x05050503,0x05050507,0x05050701,0x0505070f,0x05050903,0x05050b07,0x05050b0f,
        0x05050f03,0x05050f09,0x05070101,0x05070105,0x0507010b,0x05070303,0x05070505,0x05070509,
        0x05070703,0x05070707,0x05070905,0x05070b01,0x05070d0d,0x05090103,0x0509010f,0x05090501,
        0x05090507,0x05090705,0x0509070b,0x05090903,0x05090f05,0x05090f0b,0x050b0109,0x050b0303,
        0x050b0505,0x050b070f,0x050b0901,0x050b0b07,0x050b0f01,0x050d0101,0x050d0105,0x050d010f,
        0x050d0503,0x050d0b0b,0x050d0d03,0x050f010b,0x050f0303,0x050f050d,0x050f0701,0x050f0907,
        0x050f0b01,0x07010105,0x07010303,0x07010307,0x0701030b,0x0701030f,0x07010505,0x07010703,
        0x07010707,0x0701070b,0x07010905,0x07010909,0x0701090f,0x07010b03,0x07010d07,0x07010f03,
        0x07030103,0x07030107,0x0703010b,0x07030309,0x07030503,0x07030507,0x07030901,0x07030d01,
        0x07030f05,0x07030f0d,0x07050101,0x07050305,0x07050501,0x07050705,0x07050709,0x07050b01,
        0x07070103,0x07070301,0x07070309,0x07070503,0x07070507,0x0707050f,0x07070701,0x07070903,
        0x07070907,0x0707090f,0x07070b0b,0x07070f07,0x07090107,0x07090303,0x0709030d,0x07090505,
        0x07090703,0x07090b05,0x07090d01,0x07090d09,0x070b0103,0x070b0301,0x070b0305,0x070b050b,
        0x070b0705,0x070b0909,0x070b0b0d,0x070b0f07,0x070d030d,0x070d0903,0x070f0103,0x070f0107,
        0x070f0501,0x070f0505,0x070f070b,0x09010101,0x09010109,0x09010305,0x09010501,0x09010509,
        0x0901050f,0x09010705,0x09010903,0x09010b01,0x09010f01,0x09030105,0x0903010f,0x09030303,
        0x09030307,0x09030505,0x09030701,0x0903070b,0x09030907,0x09030b03,0x09030b0b,0x09050103,
        0x09050107,0x09050301,0x0905030b,0x09050503,0x09050707,0x09050901,0x09050b0f,0x09050d05,
        0x09050f01,0x09070109,0x09070303,0x09070307,0x09070501,0x09070505,0x09070703,0x0907070b,
        0x09090101,0x09090105,0x09090509,0x0909070f,0x09090901,0x09090f03,0x090b010b,0x090b010f,
        0x090b0503,0x090b0d05,0x090d0307,0x090d0709,0x090d0d01,0x090f0301,0x090f030b,0x090f0701,
        0x090f0907,0x090f0b03,0x0b010105,0x0b010301,0x0b010309,0x0b010505,0x0b010901,0x0b010909,
        0x0b01090f,0x0b010b05,0x0b010d0d,0x0b010f09,0x0b030103,0x0b030107,0x0b03010b,0x0b030305,
        0x0b030503,0x0b030705,0x0b030f05,0x0b050101,0x0b050303,0x0b050507,0x0b050701,0x0b05070d,
        0x0b050b07,0x0b070105,0x0b07010f,0x0b070301,0x0b07050f,0x0b070909,0x0b070b03,0x0b070d0b,
        0x0b070f07,0x0b090103,0x0b090109,0x0b090501,0x0b090705,0x0b09090d,0x0b0b0305,0x0b0b050d,
        0x0b0b0b03,0x0b0b0b07,0x0b0d0905,0x0b0f0105,0x0b0f0109,0x0b0f0505,0x0d010303,0x0d010307,
        0x0d01030b,0x0d010703,0x0d010707,0x0d010d01,0x0d030101,0x0d030501,0x0d03050f,0x0d030d09,
        0x0d050305,0x0d050709,0x0d050905,0x0d050b0b,0x0d050d05,0x0d050f01,0x0d070101,0x0d070309,
        0x0d070503,0x0d070901,0x0d09050b,0x0d090907,0x0d090d05,0x0d0b0101,0x0d0b0107,0x0d0b0709,
        0x0d0b0d01,0x0d0d010b,0x0d0d0901,0x0d0f0303,0x0d0f0307,0x0f010101,0x0f010109,0x0f01010f,
        0x0f010501,0x0f010505,0x0f01070d,0x0f010901,0x0f010b09,0x0f010d05,0x0f030105,0x0f030303,
        0x0f030509,0x0f030907,0x0f03090b,0x0f050103,0x0f050109,0x0f050301,0x0f05030d,0x0f050503,
        0x0f050701,0x0f050b03,0x0f070105,0x0f070705,0x0f07070b,0x0f070b07,0x0f090103,0x0f09010b,
        0x0f090307,0x0f090501,0x0f090b01,0x0f0b0505,0x0f0b0905,0x0f0d0105,0x0f0d0703,0x0f0f0101
    };

    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    // Per-expert weight base = w_base + expert_offset_blocks * 110 bytes.
    // Per-output-row base = expert_base + o * blocks_per_row * 110.
    size_t exp_off_bytes = (size_t)expert_offset_blocks * 110;
    size_t row_off_bytes = (size_t)o * (size_t)blocks_per_row * 110;
    const unsigned char* w_row = w_base + exp_off_bytes + row_off_bytes;
    float acc = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const unsigned char* blk = w_row + (size_t)b * 110;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        const unsigned char* qs     = blk + 2;             // 64 bytes
        const unsigned char* qh     = blk + 2 + 64;        // 8 bytes
        const unsigned char* signs  = blk + 2 + 64 + 8;    // 32 bytes
        const unsigned char* scales = blk + 2 + 64 + 8 + 32; // 4 bytes
        const float* x_blk = x + (size_t)b * 256;
        float blk_acc = 0.0f;
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ib32++) {
            unsigned int scale_nib =
                ((unsigned int)scales[ib32 >> 1] >> (4 * (ib32 & 1))) & 0xFu;
            float db = d * (float)(1 + 2 * (int)scale_nib);
            unsigned int qh_byte = (unsigned int)qh[ib32];
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                unsigned int idx1 =
                      (unsigned int)qs[ib32 * 8 + 2 * l + 0]
                    | ((qh_byte << (8 - 2 * l)) & 256u);
                unsigned int idx2 =
                      (unsigned int)qs[ib32 * 8 + 2 * l + 1]
                    | ((qh_byte << (7 - 2 * l)) & 256u);
                unsigned int grid1 = iq3s_grid[idx1];
                unsigned int grid2 = iq3s_grid[idx2];
                unsigned int sign  = (unsigned int)signs[ib32 * 4 + l];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned int q0 = (grid1 >> (8 * j)) & 0xFFu;
                    unsigned int q1 = (grid2 >> (8 * j)) & 0xFFu;
                    float s0 = (sign & (1u << (j + 0))) ? -1.0f : 1.0f;
                    float s1 = (sign & (1u << (j + 4))) ? -1.0f : 1.0f;
                    blk_acc += x_blk[32 * ib32 + 8 * l + j + 0] * (db * (float)q0 * s0);
                    blk_acc += x_blk[32 * ib32 + 8 * l + j + 4] * (db * (float)q1 * s1);
                }
            }
        }
        acc += blk_acc;
    }
    // Same warp-reduce-then-cross-warp-reduce as Q4_K/Q8_0/Q5_0 expert kernels.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-moe-quant-dispatch-iq4xs — MoE expert-variant of IQ4_XS matmul.
// 136-byte 256-elem blocks (f16 d + 2-byte scales_h + 4-byte scales_l +
// 128-byte qs), kvalues_iq4nl codebook + per-sub-block 6-bit signed scales.
// Per-expert offset = `expert_offset_blocks * 136` bytes from w_base.  Used
// by GLM-4.7-flash MoE expert tensors quantised to IQ4_XS (dt=23) — the
// second non-Q4_K dtype to surface in the IQ3_XXS GGUF mix.  One CTA per
// output row; 256 threads stride-loop the blocks; warp-reduce then
// cross-warp-reduce.
extern "C" __global__ void fused_iq4_xs_expert_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ w_base,
    float*               __restrict__ y,
    int n_out, int blocks_per_row, int expert_offset_blocks)
{
    static const int kvalues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10,
           1,   13,  25,  38,  53,  69,  89, 113
    };

    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    size_t exp_off_bytes = (size_t)expert_offset_blocks * 136;
    size_t row_off_bytes = (size_t)o * (size_t)blocks_per_row * 136;
    const unsigned char* w_row = w_base + exp_off_bytes + row_off_bytes;
    float acc = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const unsigned char* blk = w_row + (size_t)b * 136;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        unsigned int scales_h = ((unsigned int)blk[3] << 8) | (unsigned int)blk[2];
        const unsigned char* scales_l = blk + 4;     // 4 bytes
        const unsigned char* qs       = blk + 8;     // 128 bytes
        const float* x_blk = x + (size_t)b * 256;
        float blk_acc = 0.0f;
        #pragma unroll
        for (int ib = 0; ib < 8; ib++) {
            unsigned int ls_lo = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xFu;
            unsigned int ls_hi = (scales_h >> (2 * ib)) & 3u;
            int ls = (int)(ls_lo | (ls_hi << 4));     // 6-bit unsigned [0, 63]
            float dl = d * (float)(ls - 32);
            int qs_off = ib * 16;
            #pragma unroll 8
            for (int j = 0; j < 16; j++) {
                unsigned char byte = qs[qs_off + j];
                int q_lo = kvalues[byte & 0xF];
                int q_hi = kvalues[(byte >> 4) & 0xF];
                blk_acc += x_blk[ib * 32 + j]      * (dl * (float)q_lo);
                blk_acc += x_blk[ib * 32 + j + 16] * (dl * (float)q_hi);
            }
        }
        acc += blk_acc;
    }
    // Same warp-reduce-then-cross-warp-reduce as Q4_K/Q8_0/Q5_0/IQ3_S expert.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-moe-quant-dispatch-iq3xxs — MoE expert-variant of IQ3_XXS matmul.
// 98-byte 256-elem blocks (f16 d + 64-byte qs + 32-byte scales_and_signs),
// 256-entry iq3xxs_grid lookup + 128-entry ksigns_iq2xs indirection +
// per-sub-block scale (0.5 + (aux32 >> 28)) * 0.5.  Per-expert offset =
// `expert_offset_blocks * 98` bytes from w_base.  Used by GLM-4.7-flash MoE
// expert tensors quantised to IQ3_XXS (dt=18) — the third non-Q4_K dtype
// the IQ3_XXS GGUF hits after IQ4_XS and IQ3_S.  One CTA per output row;
// 256 threads stride-loop the blocks; warp-reduce then cross-warp-reduce.
extern "C" __global__ void fused_iq3_xxs_expert_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned char* __restrict__ w_base,
    float*               __restrict__ y,
    int n_out, int blocks_per_row, int expert_offset_blocks)
{
    static const unsigned char ksigns[128] = {
          0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
        144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
        160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
         48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
        192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
         80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
         96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
        240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
    };
    static const unsigned int iq3xxs_grid[256] = {
        0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu, 0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
        0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu, 0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
        0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u, 0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
        0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u, 0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
        0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu, 0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
        0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u, 0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
        0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu, 0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
        0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu, 0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
        0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu, 0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
        0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu, 0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
        0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u, 0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
        0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u, 0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
        0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu, 0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
        0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu, 0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
        0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u, 0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
        0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u, 0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
        0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu, 0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
        0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu, 0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
        0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u, 0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
        0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu, 0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
        0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu, 0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
        0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu, 0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
        0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu, 0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
        0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu, 0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
        0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u, 0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
        0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u, 0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
        0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu, 0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
        0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u, 0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
        0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u, 0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
        0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu, 0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
        0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u, 0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
        0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu, 0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u
    };

    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    size_t exp_off_bytes = (size_t)expert_offset_blocks * 98;
    size_t row_off_bytes = (size_t)o * (size_t)blocks_per_row * 98;
    const unsigned char* w_row = w_base + exp_off_bytes + row_off_bytes;
    float acc = 0.0f;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const unsigned char* blk = w_row + (size_t)b * 98;
        unsigned short d_bits = ((unsigned short)blk[1] << 8) | (unsigned short)blk[0];
        float d = aether_f16_to_f32_dev(d_bits);
        const unsigned char* qs  = blk + 2;       // 64 bytes
        const unsigned char* sas = blk + 2 + 64;  // 32 bytes
        const float* x_blk = x + (size_t)b * 256;
        float blk_acc = 0.0f;
        #pragma unroll
        for (int ib32 = 0; ib32 < 8; ib32++) {
            unsigned int aux32 =
                  (unsigned int)sas[4 * ib32 + 0]
                | ((unsigned int)sas[4 * ib32 + 1] << 8)
                | ((unsigned int)sas[4 * ib32 + 2] << 16)
                | ((unsigned int)sas[4 * ib32 + 3] << 24);
            float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                unsigned int sign_idx = (aux32 >> (7 * l)) & 127u;
                unsigned int signs    = (unsigned int)ksigns[sign_idx];
                unsigned int grid1 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 0]];
                unsigned int grid2 = iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]];
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    unsigned int q0 = (grid1 >> (8 * j)) & 0xFFu;
                    unsigned int q1 = (grid2 >> (8 * j)) & 0xFFu;
                    float s0 = (signs & (1u << (j + 0))) ? -1.0f : 1.0f;
                    float s1 = (signs & (1u << (j + 4))) ? -1.0f : 1.0f;
                    blk_acc += x_blk[32 * ib32 + 8 * l + j + 0] * (db * (float)q0 * s0);
                    blk_acc += x_blk[32 * ib32 + 8 * l + j + 4] * (db * (float)q1 * s1);
                }
            }
        }
        acc += blk_acc;
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-f16-fwd — F16-weight matmul for seq=1 autoregressive decode.
// Layout: weights stored as raw F16 (2 bytes/elem) in row-major order
// [n_out * n_in].  Input x is F32 [n_in].  Output y is F32 [n_out].
// One CTA per output row; 256 threads/CTA stride-loop over n_in.
extern "C" __global__ void fused_f16_matmul_seq1(
    const float*         __restrict__ x,
    const unsigned short* __restrict__ w_f16,
    float*               __restrict__ y,
    int n_in, int n_out)
{
    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    const unsigned short* w_row = w_f16 + (size_t)o * n_in;
    float acc = 0.0f;
    for (int i = tid; i < n_in; i += blockDim.x) {
        float w = aether_f16_to_f32_dev(w_row[i]);
        acc += x[i] * w;
    }
    // Warp-reduce then single-warp reduce.
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

// FR-17-extra-mla-fwd companion — F32-weight matmul for seq=1 decode.
// GLM-4.7-flash stores some tensors (the MoE shared-expert MLPs and a few
// LM-head-adjacent tensors) as raw F32.  Layout is row-major [n_out, n_in].
// Same CTA shape as the F16 variant; just no f16->f32 conversion needed.
extern "C" __global__ void fused_f32_matmul_seq1(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float*       __restrict__ y,
    int n_in, int n_out)
{
    int o = blockIdx.x;
    if (o >= n_out) return;
    int tid = threadIdx.x;
    const float* w_row = w + (size_t)o * n_in;
    float acc = 0.0f;
    for (int i = tid; i < n_in; i += blockDim.x) {
        acc += x[i] * w_row[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
    }
    __shared__ float warp_sums[8];
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float w_acc = (tid < 8) ? warp_sums[tid] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            w_acc += __shfl_down_sync(0xFFFFFFFFu, w_acc, off);
        }
        if (lane == 0) y[o] = w_acc;
    }
}

extern "C" __global__ void dequant_q6_k(
    const unsigned char* __restrict__ blocks,
    float*               __restrict__ out,
    int n_blocks)
{
    // 256 threads per CTA, n_blocks CTAs. Each thread = one output f32.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_blocks * 256;
    if (idx >= total) return;
    int bi = idx / 256;
    int qi = idx % 256;
    // Which n_outer half (0 or 1) and which of the 4 sub-positions
    // (0..32, 32..64, 64..96, 96..128) within that half.
    int n_outer = qi / 128;            // 0 or 1
    int qi_local = qi % 128;
    int sub_pos = qi_local / 32;       // 0..3 within the half
    int l = qi_local % 32;

    const unsigned char* base = blocks + bi * 210;
    const unsigned char* ql = base;                      // 128 bytes
    const unsigned char* qh = base + 128;                // 64 bytes
    const signed char*   sc = (const signed char*)(base + 192);  // 16 bytes signed
    unsigned short d_bits = ((unsigned short)base[209] << 8) | (unsigned short)base[208];
    float d = aether_f16_to_f32_dev(d_bits);

    int ql_off  = n_outer * 64;
    int qh_off  = n_outer * 32;
    int sc_off  = n_outer * 8;

    unsigned char ql_lo = ql[ql_off + l];
    unsigned char ql_hi = ql[ql_off + l + 32];
    unsigned char qh_byte = qh[qh_off + l];

    int q;
    int sc_idx;
    switch (sub_pos) {
        case 0:
            q = (int)((ql_lo & 0xFu) | (((qh_byte >> 0) & 3u) << 4)) - 32;
            sc_idx = sc_off + (l / 16) + 0;
            break;
        case 1:
            q = (int)((ql_hi & 0xFu) | (((qh_byte >> 2) & 3u) << 4)) - 32;
            sc_idx = sc_off + (l / 16) + 2;
            break;
        case 2:
            q = (int)(((ql_lo >> 4) & 0xFu) | (((qh_byte >> 4) & 3u) << 4)) - 32;
            sc_idx = sc_off + (l / 16) + 4;
            break;
        default:
            q = (int)(((ql_hi >> 4) & 0xFu) | (((qh_byte >> 6) & 3u) << 4)) - 32;
            sc_idx = sc_off + (l / 16) + 6;
            break;
    }
    out[idx] = d * (float)sc[sc_idx] * (float)q;
}

// FR-17-extra-bert-fwd — BERT bidirectional (full) self-attention.
//
// Encoder-only models like BGE compute attention over the WHOLE sequence in
// one pass — every query position attends to every key position with no
// causal mask.  Grid is (n_heads, seq) so each block computes one
// (head, q_pos) row of the [seq, n_heads, head_dim] output tensor.
//
// Q / K / V layout: [seq, n_heads, head_dim] (token-major; same as the
// activations the caller passes through the Q/K/V matmuls).  head_dim must
// be a multiple of 32 and ≤ 256 (same limits as the decode kernels — per
// _lane[8] × per_lane=8 caps out at 256).
//
// Shared mem: scores[max_seq] f32, sized by the launch.  BERT-base uses
// seq=512 → 2 KiB.  BERT-large + bge-large stay well under 4 KiB at
// max_seq=512.
extern "C" __global__ void bert_self_attention_fwd(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float*       __restrict__ attn_out,
    int seq, int n_heads, int head_dim,
    float scale)
{
    extern __shared__ float scores[];
    int head  = blockIdx.x;
    int q_pos = blockIdx.y;
    int lane  = threadIdx.x;
    // CEIL so head_dim < 32 (e.g. 16) still loops once per warp; per-element
    // `col < head_dim` bounds check below filters the over-counted lanes.
    int per_lane = (head_dim + 31) >> 5;

    const float* q_ptr = q + (q_pos * n_heads + head) * head_dim;

    // Load Q[q_pos, head] into thread-local lanes.
    float q_local[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = lane * per_lane + i;
        q_local[i] = (i < per_lane && col < head_dim) ? q_ptr[col] : 0.0f;
    }

    // Pass 1: scores[t] = Q[q_pos] · K[t] * scale  for t in [0, seq).
    for (int t = 0; t < seq; t++) {
        const float* k_ptr = k + (t * n_heads + head) * head_dim;
        float acc = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = lane * per_lane + i;
            if (i < per_lane && col < head_dim) acc += q_local[i] * k_ptr[col];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) scores[t] = acc * scale;
    }
    __syncwarp();

    // Pass 2: softmax over [0, seq).  Bidirectional => no mask.
    float local_max = __int_as_float(0xFF800000u);
    for (int t = lane; t < seq; t += 32) {
        float s = scores[t];
        if (s > local_max) local_max = s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFFu, local_max, off);
        if (other > local_max) local_max = other;
    }
    float max_val = __shfl_sync(0xFFFFFFFFu, local_max, 0);

    float local_sum = 0.0f;
    for (int t = lane; t < seq; t += 32) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFFu, local_sum, off);
    }
    float sum_val = __shfl_sync(0xFFFFFFFFu, local_sum, 0);
    float inv_sum = 1.0f / sum_val;
    for (int t = lane; t < seq; t += 32) {
        scores[t] *= inv_sum;
    }
    __syncwarp();

    // Pass 3: out[q_pos, head] = sum_t scores[t] * V[t, head].
    float out_local[8] = {0.0f};
    for (int t = 0; t < seq; t++) {
        const float* v_ptr = v + (t * n_heads + head) * head_dim;
        float w = scores[t];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int col = lane * per_lane + i;
            if (i < per_lane && col < head_dim) out_local[i] += w * v_ptr[col];
        }
    }
    float* out_ptr = attn_out + (q_pos * n_heads + head) * head_dim;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int col = lane * per_lane + i;
        if (i < per_lane && col < head_dim) out_ptr[col] = out_local[i];
    }
}

// FR-17-extra-bert-fwd — BERT embedding sum.
//
// BERT inputs are constructed by summing three learned tables:
//   x[i] = word_embd[input_ids[i]]
//        + pos_embd[i]                       (position 0..seq)
//        + type_embd[token_type_ids[i]]      (segment id 0 or 1)
// Followed by LayerNorm-with-bias (which we already have).
//
// Grid spans seq tokens; each thread copies one element of d_model.
extern "C" __global__ void bert_embed_sum(
    const int*   __restrict__ input_ids,
    const int*   __restrict__ token_type_ids,
    const float* __restrict__ word_embd,        // [vocab x d_model]
    const float* __restrict__ pos_embd,         // [max_pos x d_model]
    const float* __restrict__ type_embd,        // [n_types x d_model]
    float*       __restrict__ out,              // [seq x d_model]
    int seq, int d_model)
{
    int t   = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= seq || tid >= d_model) return;
    int word = input_ids[t];
    int typ  = token_type_ids[t];
    float w = word_embd[word * d_model + tid];
    float p = pos_embd[t * d_model + tid];
    float s = type_embd[typ * d_model + tid];
    out[t * d_model + tid] = w + p + s;
}
"#;

static CTX: OnceLock<CudaCtx> = OnceLock::new();
// Single-threaded by construction: Aether-emitted programs run a single
// thread of execution. The earlier `Mutex<Vec<Option<...>>>` registry
// added per-call lock+take+put overhead that turned a 240µs cuBLAS sgemm
// into a 3,600µs end-to-end op (15× regression vs candle-gpu). With
// `UnsafeCell`, three per-call buffer fetches drop to a few ns of pointer
// arithmetic. If we ever multi-thread the GPU path, gate this behind a
// runtime flag and fall back to a Mutex.
struct BufferRegistry(UnsafeCell<Vec<Option<CudaSlice<f32>>>>);
unsafe impl Sync for BufferRegistry {}
static BUFFERS: BufferRegistry = BufferRegistry(UnsafeCell::new(Vec::new()));

#[inline]
pub(crate) unsafe fn bufs() -> &'static mut Vec<Option<CudaSlice<f32>>> { &mut *BUFFERS.0.get() }

/// Parallel registry for i32 device buffers — labels for cross-entropy.
/// Same single-threaded reasoning as `BUFFERS`.
struct I32Registry(UnsafeCell<Vec<Option<CudaSlice<i32>>>>);
unsafe impl Sync for I32Registry {}
static I32_BUFFERS: I32Registry = I32Registry(UnsafeCell::new(Vec::new()));

#[inline]
unsafe fn i32_bufs() -> &'static mut Vec<Option<CudaSlice<i32>>> { &mut *I32_BUFFERS.0.get() }

/// Registry for u8 device buffers — used for quantised weight blocks
/// (Q4_K, Q6_K) that stay in their compact form on device. Avoids the
/// 4× host->device PCIe blowup of dequantising to f32 before upload.
struct U8Registry(UnsafeCell<Vec<Option<CudaSlice<u8>>>>);
unsafe impl Sync for U8Registry {}
static U8_BUFFERS: U8Registry = U8Registry(UnsafeCell::new(Vec::new()));

#[inline]
unsafe fn u8_bufs() -> &'static mut Vec<Option<CudaSlice<u8>>> { &mut *U8_BUFFERS.0.get() }

fn ctx() -> &'static CudaCtx {
    CTX.get_or_init(|| {
        // FR-17.14-extra-deepest-graph: use non-default stream so we can
        // CUDA-graph-capture launches. Default stream is the legacy null
        // stream which cuStreamBeginCapture_v2 rejects.
        let device = CudaDevice::new_with_stream(0).expect("CudaDevice::new_with_stream(0)");
        let blas = CudaBlas::new(device.clone()).expect("CudaBlas::new");
        // JIT-compile the small custom kernels via nvrtc.
        let ptx = compile_ptx(KERNEL_SRC).expect("compile_ptx");
        device.load_ptx(ptx, "aether_kernels",
            &["cross_entropy_fwd", "cross_entropy_bwd", "adamw_step",
              "add_f32", "gelu_fwd", "gelu_bwd",
              "layer_norm_fwd", "layer_norm_bwd_dx", "layer_norm_bwd_params",
              "softmax_f32", "softmax_bwd", "softmax_bwd_scaled", "scale_f32",
              "gelu_inplace", "add_layer_norm_fwd",
              // matt-voice deploy kernels
              "rms_norm_fwd", "rope_apply", "gqa_repeat_kv",
              "silu_inplace", "mul_inplace", "add_inplace", "bias_add",
              "dequant_q4_k_m", "dequant_q6_k", "fused_q4k_matmul_seq1",
              "fused_q4_0_matmul_seq1",
              "fused_q5_0_matmul_seq1",
              "fused_q8_0_matmul_seq1",
              "fused_q5_k_matmul_seq1",
              "fused_q3_k_matmul_seq1",
              "fused_iq4_nl_matmul_seq1",
              "fused_iq4_xs_matmul_seq1",
              "fused_iq3_xxs_matmul_seq1",
              "fused_iq3_s_matmul_seq1",
              "fused_q4k_matmul_seq1_v2", "fused_q6k_matmul_seq1_v2",
              "fused_q4k_ffn_gate_up_silu_mul",
              "fused_q4k_matmul_seq1_v3", "fused_q4k_ffn_gate_up_silu_mul_v2",
              "rope_apply_devarg", "append_kv_devarg", "attention_seq1_devarg",
              "append_kv", "attention_seq1",
              "fused_f16_matmul_seq1",
              "fused_f32_matmul_seq1",
              "fused_q4k_expert_matmul_seq1",
              "fused_q8_0_expert_matmul_seq1",
              "fused_q5_0_expert_matmul_seq1",
              "fused_iq3_s_expert_matmul_seq1",
              "fused_iq4_xs_expert_matmul_seq1",
              "fused_iq3_xxs_expert_matmul_seq1",
              "bert_self_attention_fwd",
              "bert_embed_sum"])
            .expect("load_ptx");
        let cross_entropy_fwd = device.get_func("aether_kernels", "cross_entropy_fwd").unwrap();
        let cross_entropy_bwd = device.get_func("aether_kernels", "cross_entropy_bwd").unwrap();
        let adamw_step        = device.get_func("aether_kernels", "adamw_step").unwrap();
        let add_f32           = device.get_func("aether_kernels", "add_f32").unwrap();
        let gelu_fwd          = device.get_func("aether_kernels", "gelu_fwd").unwrap();
        let gelu_bwd          = device.get_func("aether_kernels", "gelu_bwd").unwrap();
        let layer_norm_fwd    = device.get_func("aether_kernels", "layer_norm_fwd").unwrap();
        let layer_norm_bwd_dx     = device.get_func("aether_kernels", "layer_norm_bwd_dx").unwrap();
        let layer_norm_bwd_params = device.get_func("aether_kernels", "layer_norm_bwd_params").unwrap();
        let softmax_f32           = device.get_func("aether_kernels", "softmax_f32").unwrap();
        let softmax_bwd           = device.get_func("aether_kernels", "softmax_bwd").unwrap();
        let softmax_bwd_scaled    = device.get_func("aether_kernels", "softmax_bwd_scaled").unwrap();
        let scale_f32             = device.get_func("aether_kernels", "scale_f32").unwrap();
        let gelu_inplace          = device.get_func("aether_kernels", "gelu_inplace").unwrap();
        let add_layer_norm_fwd    = device.get_func("aether_kernels", "add_layer_norm_fwd").unwrap();
        let rms_norm_fwd          = device.get_func("aether_kernels", "rms_norm_fwd").unwrap();
        let rope_apply            = device.get_func("aether_kernels", "rope_apply").unwrap();
        let gqa_repeat_kv         = device.get_func("aether_kernels", "gqa_repeat_kv").unwrap();
        let silu_inplace          = device.get_func("aether_kernels", "silu_inplace").unwrap();
        let mul_inplace           = device.get_func("aether_kernels", "mul_inplace").unwrap();
        let add_inplace           = device.get_func("aether_kernels", "add_inplace").unwrap();
        let bias_add              = device.get_func("aether_kernels", "bias_add").unwrap();
        let dequant_q4_k_m_gpu    = device.get_func("aether_kernels", "dequant_q4_k_m").unwrap();
        let dequant_q6_k_gpu      = device.get_func("aether_kernels", "dequant_q6_k").unwrap();
        let fused_q4k_matmul_seq1 = device.get_func("aether_kernels", "fused_q4k_matmul_seq1").unwrap();
        let fused_q4_0_matmul_seq1 = device.get_func("aether_kernels", "fused_q4_0_matmul_seq1").unwrap();
        let fused_q5_0_matmul_seq1 = device.get_func("aether_kernels", "fused_q5_0_matmul_seq1").unwrap();
        let fused_q8_0_matmul_seq1 = device.get_func("aether_kernels", "fused_q8_0_matmul_seq1").unwrap();
        let fused_q5_k_matmul_seq1 = device.get_func("aether_kernels", "fused_q5_k_matmul_seq1").unwrap();
        let fused_q3_k_matmul_seq1 = device.get_func("aether_kernels", "fused_q3_k_matmul_seq1").unwrap();
        let fused_iq4_nl_matmul_seq1 = device.get_func("aether_kernels", "fused_iq4_nl_matmul_seq1").unwrap();
        let fused_iq4_xs_matmul_seq1 = device.get_func("aether_kernels", "fused_iq4_xs_matmul_seq1").unwrap();
        let fused_iq3_xxs_matmul_seq1 = device.get_func("aether_kernels", "fused_iq3_xxs_matmul_seq1").unwrap();
        let fused_iq3_s_matmul_seq1 = device.get_func("aether_kernels", "fused_iq3_s_matmul_seq1").unwrap();
        let fused_q4k_matmul_seq1_v2 = device.get_func("aether_kernels", "fused_q4k_matmul_seq1_v2").unwrap();
        let fused_q6k_matmul_seq1_v2 = device.get_func("aether_kernels", "fused_q6k_matmul_seq1_v2").unwrap();
        let fused_q4k_ffn_gate_up_silu_mul = device.get_func("aether_kernels", "fused_q4k_ffn_gate_up_silu_mul").unwrap();
        let fused_q4k_matmul_seq1_v3 = device.get_func("aether_kernels", "fused_q4k_matmul_seq1_v3").unwrap();
        let fused_q4k_ffn_gate_up_silu_mul_v2 = device.get_func("aether_kernels", "fused_q4k_ffn_gate_up_silu_mul_v2").unwrap();
        let rope_apply_devarg = device.get_func("aether_kernels", "rope_apply_devarg").unwrap();
        let append_kv_devarg = device.get_func("aether_kernels", "append_kv_devarg").unwrap();
        let attention_seq1_devarg = device.get_func("aether_kernels", "attention_seq1_devarg").unwrap();
        let append_kv = device.get_func("aether_kernels", "append_kv").unwrap();
        let attention_seq1 = device.get_func("aether_kernels", "attention_seq1").unwrap();
        let fused_f16_matmul_seq1 = device.get_func("aether_kernels", "fused_f16_matmul_seq1").unwrap();
        let fused_f32_matmul_seq1 = device.get_func("aether_kernels", "fused_f32_matmul_seq1").unwrap();
        let fused_q4k_expert_matmul_seq1 = device.get_func("aether_kernels", "fused_q4k_expert_matmul_seq1").unwrap();
        let fused_q8_0_expert_matmul_seq1 = device.get_func("aether_kernels", "fused_q8_0_expert_matmul_seq1").unwrap();
        let fused_q5_0_expert_matmul_seq1 = device.get_func("aether_kernels", "fused_q5_0_expert_matmul_seq1").unwrap();
        let fused_iq3_s_expert_matmul_seq1 = device.get_func("aether_kernels", "fused_iq3_s_expert_matmul_seq1").unwrap();
        let fused_iq4_xs_expert_matmul_seq1 = device.get_func("aether_kernels", "fused_iq4_xs_expert_matmul_seq1").unwrap();
        let fused_iq3_xxs_expert_matmul_seq1 = device.get_func("aether_kernels", "fused_iq3_xxs_expert_matmul_seq1").unwrap();
        let bert_self_attention_fwd = device.get_func("aether_kernels", "bert_self_attention_fwd").unwrap();
        let bert_embed_sum = device.get_func("aether_kernels", "bert_embed_sum").unwrap();

        CudaCtx { device, blas, cross_entropy_fwd, cross_entropy_bwd, adamw_step,
                  add_f32, gelu_fwd, gelu_bwd,
                  layer_norm_fwd, layer_norm_bwd_dx, layer_norm_bwd_params,
                  softmax_f32, softmax_bwd, softmax_bwd_scaled, scale_f32, gelu_inplace,
                  add_layer_norm_fwd,
                  rms_norm_fwd, rope_apply, gqa_repeat_kv,
                  silu_inplace, mul_inplace, add_inplace, bias_add,
                  dequant_q4_k_m_gpu, dequant_q6_k_gpu,
                  fused_q4k_matmul_seq1, fused_q4_0_matmul_seq1,
                  fused_q5_0_matmul_seq1, fused_q8_0_matmul_seq1,
                  fused_q5_k_matmul_seq1,
                  fused_q3_k_matmul_seq1,
                  fused_iq4_nl_matmul_seq1,
                  fused_iq4_xs_matmul_seq1,
                  fused_iq3_xxs_matmul_seq1,
                  fused_iq3_s_matmul_seq1,
                  fused_q4k_matmul_seq1_v2,
                  fused_q6k_matmul_seq1_v2, fused_q4k_ffn_gate_up_silu_mul,
                  fused_q4k_matmul_seq1_v3,
                  fused_q4k_ffn_gate_up_silu_mul_v2,
                  rope_apply_devarg, append_kv_devarg, attention_seq1_devarg,
                  append_kv, attention_seq1, fused_f16_matmul_seq1,
                  fused_f32_matmul_seq1,
                  fused_q4k_expert_matmul_seq1,
                  fused_q8_0_expert_matmul_seq1,
                  fused_q5_0_expert_matmul_seq1,
                  fused_iq3_s_expert_matmul_seq1,
                  fused_iq4_xs_expert_matmul_seq1,
                  fused_iq3_xxs_expert_matmul_seq1,
                  bert_self_attention_fwd, bert_embed_sum }
    })
}

fn handle_to_i32_idx(h: i64) -> Option<usize> {
    if h <= 0 { None } else { Some((h - 1) as usize) }
}

/// Allocate `n` i32s on the device, zero-initialised. Separate registry from
/// the f32 one. Returns an opaque i64 handle.
#[no_mangle] pub extern "C" fn aether_dev_alloc_i32(n: c_int) -> i64 {
    if n <= 0 { return 0; }
    let buf = ctx().device.alloc_zeros::<i32>(n as usize).expect("cudaMalloc i32");
    let bs = unsafe { i32_bufs() };
    bs.push(Some(buf));
    bs.len() as i64
}

#[no_mangle] pub extern "C" fn aether_dev_free_i32(handle: i64) -> c_int {
    if let Some(i) = handle_to_i32_idx(handle) {
        let bs = unsafe { i32_bufs() };
        if i < bs.len() { bs[i] = None; }
    }
    0
}

/// Host → device copy of `n` i32s.
#[no_mangle] pub unsafe extern "C" fn aether_dev_h2d_i32(host: i64, dev: i64, n: c_int) -> c_int {
    let Some(i) = handle_to_i32_idx(dev) else { return -1; };
    if host == 0 || n <= 0 { return -1; }
    let host_slice = std::slice::from_raw_parts(host as *const i32, n as usize);
    let bs = i32_bufs();
    let buf = bs[i].as_mut().expect("freed i32 buf");
    ctx().device.htod_sync_copy_into(host_slice, buf).expect("h2d i32");
    0
}

// === u8 device buffers (FR-17.14-extra-deepest: Q4_K-on-GPU) ===

fn handle_to_u8_idx(h: i64) -> Option<usize> {
    if h <= 0 { None } else { Some((h - 1) as usize) }
}

/// Allocate `n` bytes on the device, zero-initialised. Returns an
/// opaque i64 handle (1-based, separate from f32 / i32 registries).
#[no_mangle] pub extern "C" fn aether_dev_alloc_u8(n: c_int) -> i64 {
    if n <= 0 { return 0; }
    let buf = ctx().device.alloc_zeros::<u8>(n as usize).expect("cudaMalloc u8");
    let bs = unsafe { u8_bufs() };
    bs.push(Some(buf));
    bs.len() as i64
}

#[no_mangle] pub extern "C" fn aether_dev_free_u8(handle: i64) -> c_int {
    if let Some(i) = handle_to_u8_idx(handle) {
        let bs = unsafe { u8_bufs() };
        if i < bs.len() { bs[i] = None; }
    }
    0
}

/// Host → device copy of `n` bytes. Used to upload Q4_K / Q6_K block
/// data without the f32 dequant blowup.
#[no_mangle] pub unsafe extern "C" fn aether_dev_h2d_u8(host: i64, dev: i64, n: c_int) -> c_int {
    let Some(i) = handle_to_u8_idx(dev) else { return -1; };
    if host == 0 || n <= 0 { return -1; }
    let host_slice = std::slice::from_raw_parts(host as *const u8, n as usize);
    let bs = u8_bufs();
    let buf = bs[i].as_mut().expect("freed u8 buf");
    ctx().device.htod_sync_copy_into(host_slice, buf).expect("h2d u8");
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_dev_d2h_u8(dev: i64, host: i64, n: c_int) -> c_int {
    let Some(i) = handle_to_u8_idx(dev) else { return -1; };
    if host == 0 || n <= 0 { return -1; }
    let host_slice = std::slice::from_raw_parts_mut(host as *mut u8, n as usize);
    let bs = u8_bufs();
    let buf = bs[i].as_ref().expect("freed u8 buf");
    ctx().device.dtoh_sync_copy_into(buf, host_slice).expect("d2h u8");
    0
}

fn handle_to_idx(h: i64) -> Option<usize> {
    if h <= 0 { None } else { Some((h - 1) as usize) }
}

/// Initialise the global CUDA context. Idempotent. Returns 0 on success.
#[no_mangle] pub extern "C" fn aether_dev_init() -> c_int {
    let _ = ctx(); 0
}

/// Allocate `n` f32s on the device, zero-initialised. Returns an opaque
/// `i64` handle (1-based slot index) — 0 is the null sentinel.
#[no_mangle] pub extern "C" fn aether_dev_alloc_f32(n: c_int) -> i64 {
    if n <= 0 { return 0; }
    let buf = ctx().device.alloc_zeros::<f32>(n as usize).expect("cudaMalloc");
    let bs = unsafe { bufs() };
    bs.push(Some(buf));
    bs.len() as i64
}

/// Free a device buffer. Safe on `0` / already-freed handles.
#[no_mangle] pub extern "C" fn aether_dev_free_f32(handle: i64) -> c_int {
    if let Some(i) = handle_to_idx(handle) {
        let bs = unsafe { bufs() };
        if i < bs.len() { bs[i] = None; }
    }
    0
}

/// Host → device copy of `n` f32s. `host` is a raw f32 pointer (from
/// `aether_alloc_f32` or any caller-owned buffer); `dev` is a device
/// handle.
#[no_mangle] pub unsafe extern "C" fn aether_dev_h2d_f32(host: i64, dev: i64, n: c_int) -> c_int {
    let Some(i) = handle_to_idx(dev) else { return -1; };
    if host == 0 || n <= 0 { return -1; }
    let host_slice = std::slice::from_raw_parts(host as *const f32, n as usize);
    let bs = unsafe { bufs() };
    let buf = bs[i].as_mut().expect("freed buffer");
    ctx().device.htod_sync_copy_into(host_slice, buf).expect("h2d");
    0
}

/// Device → host copy of `n` f32s.
#[no_mangle] pub unsafe extern "C" fn aether_dev_d2h_f32(dev: i64, host: i64, n: c_int) -> c_int {
    let Some(i) = handle_to_idx(dev) else { return -1; };
    if host == 0 || n <= 0 { return -1; }
    let host_slice = std::slice::from_raw_parts_mut(host as *mut f32, n as usize);
    let bs = unsafe { bufs() };
    let buf = bs[i].as_ref().expect("freed buffer");
    ctx().device.dtoh_sync_copy_into(buf, host_slice).expect("d2h");
    0
}

/// `out[m,n] = a[m,k] · b[k,n]` on the device via cuBLAS sgemm.
///
/// cuBLAS is column-major; our buffers are row-major. We compute
/// `out^T = b^T · a^T` in column-major land, which is identical to
/// `out = a · b` in row-major land — no actual transpose, just a view
/// reinterpretation.
#[no_mangle] pub extern "C" fn aether_op_matmul_f32_cuda(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let (Some(ia), Some(ib), Some(io)) = (handle_to_idx(a), handle_to_idx(b), handle_to_idx(out))
        else { return -1; };
    // Take all three slots out of the Vec so we can hold three independent
    // borrows (two & + one &mut). Put them back after the gemm.
    let (a_buf, b_buf, mut out_buf) = {
        let bs = unsafe { bufs() };
        (bs[ia].take().expect("freed a"),
         bs[ib].take().expect("freed b"),
         bs[io].take().expect("freed out"))
    };
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,         // swapped row/col for column-major view
        n: m as i32,
        k: k as i32,
        alpha: 1.0,
        beta: 0.0,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
    };
    unsafe { ctx().blas.gemm(cfg, &b_buf, &a_buf, &mut out_buf).expect("sgemm"); }
    let bs = unsafe { bufs() };
    bs[ia] = Some(a_buf);
    bs[ib] = Some(b_buf);
    bs[io] = Some(out_buf);
    0
}

/// `out[m, n] = a[k, m]^T · b[k, n]` — T transA × N transB. Used for the dK
/// path of attention backward: dK = scores^T @ Q.
/// Inputs:
///   a:  [K, M]  (transposed view → [M, K])
///   b:  [K, N]
///   out:[M, N]
#[no_mangle] pub extern "C" fn aether_op_matmul_tn_f32_cuda(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let (Some(ia), Some(ib), Some(io)) = (handle_to_idx(a), handle_to_idx(b), handle_to_idx(out))
        else { return -1; };
    let (a_buf, b_buf, mut out_buf) = {
        let bs = unsafe { bufs() };
        (bs[ia].take().unwrap(), bs[ib].take().unwrap(), bs[io].take().unwrap())
    };
    // Row-major C[M,N] = A^T[M,K] @ B[K,N], with A row-major [K,M], B row-major
    // [K,N]. Column-major view: C^T[N,M] = B^T[N,K] @ A[K,M] →
    // sgemm(B, A) with transa=T (transposing B in col-view), transb=N.
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0,
        beta: 0.0,
        lda: n as i32,
        ldb: m as i32,
        ldc: n as i32,
    };
    unsafe { ctx().blas.gemm(cfg, &b_buf, &a_buf, &mut out_buf).expect("sgemm tn"); }
    let bs = unsafe { bufs() };
    bs[ia] = Some(a_buf);
    bs[ib] = Some(b_buf);
    bs[io] = Some(out_buf);
    0
}

/// `out[m, n] = a[m, k] · b[n, k]^T` — N transA × T transB sgemm. Output shape
/// [M, N], with `b` laid out as `[N, K]` so we transpose it on the fly.
/// This is the "scores = Q @ K^T" path in single-head attention.
#[no_mangle] pub extern "C" fn aether_op_matmul_nt_f32_cuda(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let (Some(ia), Some(ib), Some(io)) = (handle_to_idx(a), handle_to_idx(b), handle_to_idx(out))
        else { return -1; };
    let (a_buf, b_buf, mut out_buf) = {
        let bs = unsafe { bufs() };
        (bs[ia].take().unwrap(), bs[ib].take().unwrap(), bs[io].take().unwrap())
    };
    // Row-major C = A @ B^T with A=[M,K], B=[N,K], C=[M,N] is column-major
    // C^T = (A @ B^T)^T = B @ A^T. So feed (B, A) with transb=T:
    //   col-major view: m_col=N, n_col=M, k_col=K
    //   B is [N,K] row-major → leading dim K, no transpose in col view
    //   A is [M,K] row-major → leading dim K, "T" in col view to get A^T
    //   C is [M,N] row-major → leading dim N, written as col-major C^T [N,M]
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        alpha: 1.0,
        beta: 0.0,
        lda: k as i32,
        ldb: k as i32,
        ldc: n as i32,
    };
    unsafe { ctx().blas.gemm(cfg, &b_buf, &a_buf, &mut out_buf).expect("sgemm nt"); }
    let bs = unsafe { bufs() };
    bs[ia] = Some(a_buf);
    bs[ib] = Some(b_buf);
    bs[io] = Some(out_buf);
    0
}

/// Fused matmul + GELU. Single user-visible op replacing the
/// `x.matmul(&w, &mut out); out.gelu(&mut out);` two-call sequence.
/// Performs cuBLAS sgemm into `out`, then in-place GELU on `out`. Saves
/// one round-trip through the runtime ABI + one buffer registry hit;
/// the GELU launch immediately follows the sgemm so they queue back-to-
/// back with no explicit sync between.
///
/// This is the kernel side of the MIR-level fusion pass — exposed today
/// as an explicit method `x.matmul_gelu(&w, &mut out)` while the pass
/// matures; once the pass lands, the unfused source-level pattern
/// gets rewritten to call this automatically.
#[no_mangle] pub extern "C" fn aether_op_matmul_gelu_f32_cuda(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    // Reuse the matmul implementation verbatim.
    let rc = aether_op_matmul_f32_cuda(a, b, out, m, k, n);
    if rc != 0 { return rc; }
    // In-place GELU on `out`.
    let Some(io) = handle_to_idx(out) else { return -1; };
    let bs = unsafe { bufs() };
    let o_p = bs[io].as_mut().unwrap() as *mut CudaSlice<f32>;
    let n_total = (m * n) as u32;
    let cfg = LaunchConfig::for_num_elems(n_total);
    unsafe {
        let ov = &mut *o_p;
        ctx().gelu_inplace.clone().launch(cfg, (ov, n_total as i32))
            .expect("launch gelu_inplace");
    }
    0
}

/// Diagnostic: split a single matmul into its three measurable phases —
/// (1) registry lock + take, (2) raw `cublasSgemm` enqueue, (3) device
/// synchronize for that one call. Returns nothing; emits a single
/// `prof  ...` line to stdout. Used to pin down where the 15× gap vs
/// candle-gpu lives. Don't ship in a hot loop.
#[no_mangle] pub extern "C" fn aether_op_matmul_f32_cuda_profile(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    use std::time::Instant;
    let (Some(ia), Some(ib), Some(io)) = (handle_to_idx(a), handle_to_idx(b), handle_to_idx(out))
        else { return -1; };
    let t0 = Instant::now();
    let (a_buf, b_buf, mut out_buf) = {
        let bs = unsafe { bufs() };
        (bs[ia].take().unwrap(), bs[ib].take().unwrap(), bs[io].take().unwrap())
    };
    let lock_take_us = t0.elapsed().as_micros();
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N, transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: k as i32,
        alpha: 1.0, beta: 0.0, lda: n as i32, ldb: k as i32, ldc: n as i32,
    };
    let t1 = Instant::now();
    unsafe { ctx().blas.gemm(cfg, &b_buf, &a_buf, &mut out_buf).expect("sgemm"); }
    let enqueue_us = t1.elapsed().as_micros();
    let t2 = Instant::now();
    let _ = ctx().device.synchronize();
    let sync_us = t2.elapsed().as_micros();
    let t3 = Instant::now();
    let bs = unsafe { bufs() };
    bs[ia] = Some(a_buf); bs[ib] = Some(b_buf); bs[io] = Some(out_buf);
    let putback_us = t3.elapsed().as_micros();
    println!("prof  M={m} N={n} K={k}  lock_take={lock_take_us}µs  enqueue={enqueue_us}µs  sync={sync_us}µs  putback={putback_us}µs  total={}µs",
             lock_take_us + enqueue_us + sync_us + putback_us);
    0
}

/// `db[k,n] = a[m,k]^T · dy[m,n]` — same shape as the CPU op. Single sgemm
/// with `transA = T` in the column-major view of the row-major arrays.
#[no_mangle] pub extern "C" fn aether_op_matmul_backward_rhs_f32_cuda(
    a: i64, dy: i64, db: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let (Some(ia), Some(idy), Some(idb)) = (handle_to_idx(a), handle_to_idx(dy), handle_to_idx(db))
        else { return -1; };
    let (a_buf, dy_buf, mut db_buf) = {
        let bs = unsafe { bufs() };
        (bs[ia].take().unwrap(), bs[idy].take().unwrap(), bs[idb].take().unwrap())
    };
    // Row-major:  db[k,n] = a[m,k]^T · dy[m,n]
    // Column-major view (swap rows ↔ cols of every matrix):
    //   a_cm  = (k,m), dy_cm = (n,m), db_cm = (n,k)
    // We want (n,k) = (n,m) · (m,k), so:
    //   sgemm(transA=N, transB=T, m=n, n=k, k=m, A=dy_cm[n×m], B=a_cm[k×m])
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: n as i32,
        n: k as i32,
        k: m as i32,
        alpha: 1.0,
        beta: 0.0,
        lda: n as i32,
        ldb: k as i32,
        ldc: n as i32,
    };
    unsafe { ctx().blas.gemm(cfg, &dy_buf, &a_buf, &mut db_buf).expect("sgemm bwd_rhs"); }
    let bs = unsafe { bufs() };
    bs[ia] = Some(a_buf);
    bs[idy] = Some(dy_buf);
    bs[idb] = Some(db_buf);
    0
}

/// `da[m,k] = dy[m,n] · b[k,n]^T` — single sgemm with `transB = T`.
#[no_mangle] pub extern "C" fn aether_op_matmul_backward_lhs_f32_cuda(
    dy: i64, b: i64, da: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let (Some(idy), Some(ib), Some(ida)) = (handle_to_idx(dy), handle_to_idx(b), handle_to_idx(da))
        else { return -1; };
    let (dy_buf, b_buf, mut da_buf) = {
        let bs = unsafe { bufs() };
        (bs[idy].take().unwrap(), bs[ib].take().unwrap(), bs[ida].take().unwrap())
    };
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: k as i32,
        n: m as i32,
        k: n as i32,
        alpha: 1.0,
        beta: 0.0,
        lda: n as i32,
        ldb: n as i32,
        ldc: k as i32,
    };
    unsafe { ctx().blas.gemm(cfg, &b_buf, &dy_buf, &mut da_buf).expect("sgemm bwd_lhs"); }
    let bs = unsafe { bufs() };
    bs[idy] = Some(dy_buf);
    bs[ib] = Some(b_buf);
    bs[ida] = Some(da_buf);
    0
}

/// Diagnostic batch profiler: enqueue `iters` matmuls back-to-back without
/// any per-call sync, then a single final sync, and report enqueue total
/// vs sync total. Pinpoints whether the bench gap is per-call cudarc
/// overhead or queue-drain time.
#[no_mangle] pub extern "C" fn aether_bench_matmul_batch(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int, iters: c_int,
) -> c_int {
    use std::time::Instant;
    let (Some(ia), Some(ib), Some(io)) = (handle_to_idx(a), handle_to_idx(b), handle_to_idx(out))
        else { return -1; };
    let cfg = GemmConfig::<f32> {
        transa: cublasOperation_t::CUBLAS_OP_N, transb: cublasOperation_t::CUBLAS_OP_N,
        m: n as i32, n: m as i32, k: k as i32,
        alpha: 1.0, beta: 0.0, lda: n as i32, ldb: k as i32, ldc: n as i32,
    };
    // Take buffers ONCE, hold them across all iters, put back after.
    let (a_buf, b_buf, mut out_buf) = {
        let bs = unsafe { bufs() };
        (bs[ia].take().unwrap(), bs[ib].take().unwrap(), bs[io].take().unwrap())
    };
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { ctx().blas.gemm(cfg, &b_buf, &a_buf, &mut out_buf).expect("sgemm"); }
    }
    let enqueue_us = t0.elapsed().as_micros();
    let t1 = Instant::now();
    let _ = ctx().device.synchronize();
    let sync_us = t1.elapsed().as_micros();
    let bs = unsafe { bufs() };
    bs[ia] = Some(a_buf); bs[ib] = Some(b_buf); bs[io] = Some(out_buf);
    println!("batch  M={m} N={n} K={k}  iters={iters}  enqueue={enqueue_us}µs  sync={sync_us}µs  total={}µs  per_iter={}µs",
             enqueue_us + sync_us,
             (enqueue_us + sync_us) / (iters.max(1) as u128));
    0
}

/// Cross-entropy forward on device. Same shape + return as the CPU op
/// (`aether_op_cross_entropy_f32`): mean loss across the batch, with the
/// per-row softmax probabilities written to `probs_out`. Loss reduction is
/// done host-side after a tiny d2h copy of the per-row losses; the kernel
/// stays per-row and avoids cross-block reductions.
#[no_mangle] pub extern "C" fn aether_op_cross_entropy_f32_cuda(
    logits: i64, labels_i32: i64, probs_out: i64, b: c_int, v: c_int,
) -> f32 {
    let (Some(il), Some(ip)) = (handle_to_idx(logits), handle_to_idx(probs_out))
        else { return 0.0; };
    let Some(ilab) = handle_to_i32_idx(labels_i32) else { return 0.0; };
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let mut losses = ctx().device.alloc_zeros::<f32>(b as usize).expect("alloc losses");
    let logits_buf_p = bs[il].as_ref().unwrap() as *const CudaSlice<f32>;
    let probs_buf_p  = bs[ip].as_mut().unwrap() as *mut CudaSlice<f32>;
    let labels_buf_p = ibs[ilab].as_ref().unwrap() as *const CudaSlice<i32>;
    let cfg = LaunchConfig::for_num_elems(b as u32);
    unsafe {
        let logits_buf = &*logits_buf_p;
        let probs_buf  = &mut *probs_buf_p;
        let labels_buf = &*labels_buf_p;
        ctx().cross_entropy_fwd.clone().launch(cfg, (logits_buf, labels_buf, probs_buf, &mut losses, b, v))
            .expect("launch ce_fwd");
    }
    let host = ctx().device.dtoh_sync_copy(&losses).expect("d2h losses");
    let mut sum = 0.0f64;
    for x in &host { sum += *x as f64; }
    (sum / b as f64) as f32
}

#[no_mangle] pub extern "C" fn aether_op_cross_entropy_backward_f32_cuda(
    probs: i64, labels_i32: i64, dlogits: i64, b: c_int, v: c_int,
) -> c_int {
    let (Some(ip), Some(idl)) = (handle_to_idx(probs), handle_to_idx(dlogits))
        else { return -1; };
    let Some(ilab) = handle_to_i32_idx(labels_i32) else { return -1; };
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let probs_buf_p   = bs[ip].as_ref().unwrap() as *const CudaSlice<f32>;
    let dlogits_buf_p = bs[idl].as_mut().unwrap() as *mut CudaSlice<f32>;
    let labels_buf_p  = ibs[ilab].as_ref().unwrap() as *const CudaSlice<i32>;
    let n = (b * v) as u32;
    let cfg = LaunchConfig::for_num_elems(n);
    unsafe {
        let probs_buf   = &*probs_buf_p;
        let dlogits_buf = &mut *dlogits_buf_p;
        let labels_buf  = &*labels_buf_p;
        ctx().cross_entropy_bwd.clone().launch(cfg, (probs_buf, labels_buf, dlogits_buf, b, v))
            .expect("launch ce_bwd");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_adamw_step_f32_cuda(
    param: i64, grad: i64, m: i64, v: i64,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    step: i64, n: c_int,
) -> c_int {
    let (Some(ip), Some(ig), Some(im), Some(iv)) = (
        handle_to_idx(param), handle_to_idx(grad), handle_to_idx(m), handle_to_idx(v))
        else { return -1; };
    let bs = unsafe { bufs() };
    let bc1_inv = 1.0 / (1.0 - libm_powf(beta1, step as f32));
    let bc2_inv = 1.0 / (1.0 - libm_powf(beta2, step as f32));
    let p_buf = bs[ip].as_mut().unwrap() as *mut CudaSlice<f32>;
    let g_buf = bs[ig].as_ref().unwrap() as *const CudaSlice<f32>;
    let m_buf = bs[im].as_mut().unwrap() as *mut CudaSlice<f32>;
    let v_buf = bs[iv].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    // Multiple borrows from the same Vec — go through raw pointers.
    unsafe {
        let p = &mut *p_buf;
        let g = &*g_buf;
        let m = &mut *m_buf;
        let v = &mut *v_buf;
        ctx().adamw_step.clone().launch(cfg, (p, g, m, v, lr, beta1, beta2, eps, wd, bc1_inv, bc2_inv, n))
            .expect("launch adamw");
    }
    0
}

/// `out[i] = a[i] + b[i]`, length `n`. Both `a` and `b` are device handles.
#[no_mangle] pub extern "C" fn aether_op_add_f32_cuda(
    a: i64, b: i64, out: i64, n: c_int,
) -> c_int {
    let (Some(ia), Some(ib), Some(io)) = (handle_to_idx(a), handle_to_idx(b), handle_to_idx(out))
        else { return -1; };
    let bs = unsafe { bufs() };
    let a_p = bs[ia].as_ref().unwrap() as *const CudaSlice<f32>;
    let b_p = bs[ib].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[io].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let av = &*a_p; let bv = &*b_p; let ov = &mut *o_p;
        ctx().add_f32.clone().launch(cfg, (av, bv, ov, n)).expect("launch add_f32");
    }
    0
}

/// GELU forward (tanh approx). `out[i] = gelu(in[i])`, length `n`.
#[no_mangle] pub extern "C" fn aether_op_gelu_f32_cuda(
    x: i64, y: i64, n: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p;
        ctx().gelu_fwd.clone().launch(cfg, (xv, yv, n)).expect("launch gelu_fwd");
    }
    0
}

/// GELU backward (tanh approx). `dx[i] = dy[i] * gelu'(x[i])`.
#[no_mangle] pub extern "C" fn aether_op_gelu_backward_f32_cuda(
    x: i64, dy: i64, dx: i64, n: c_int,
) -> c_int {
    let (Some(ix), Some(idy), Some(idx_)) = (handle_to_idx(x), handle_to_idx(dy), handle_to_idx(dx))
        else { return -1; };
    let bs = unsafe { bufs() };
    let x_p  = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let dy_p = bs[idy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dx_p = bs[idx_].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let xv = &*x_p; let dyv = &*dy_p; let dxv = &mut *dx_p;
        ctx().gelu_bwd.clone().launch(cfg, (xv, dyv, dxv, n)).expect("launch gelu_bwd");
    }
    0
}

/// LayerNorm forward: `y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`,
/// last-dim reduction. `mean_out` and `rstd_out` are per-row caches for the
/// backward pass (length B each).
#[no_mangle] pub extern "C" fn aether_op_layer_norm_f32_cuda(
    x: i64, gamma: i64, beta: i64, y: i64,
    mean_out: i64, rstd_out: i64,
    eps: f32, b: c_int, d: c_int,
) -> c_int {
    let (Some(ix), Some(igamma), Some(ibeta), Some(iy), Some(im), Some(ir)) = (
        handle_to_idx(x), handle_to_idx(gamma), handle_to_idx(beta),
        handle_to_idx(y), handle_to_idx(mean_out), handle_to_idx(rstd_out))
        else { return -1; };
    let bs = unsafe { bufs() };
    let x_p     = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let g_p     = bs[igamma].as_ref().unwrap() as *const CudaSlice<f32>;
    let beta_p  = bs[ibeta].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p     = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let mean_p  = bs[im].as_mut().unwrap() as *mut CudaSlice<f32>;
    let rstd_p  = bs[ir].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(b as u32);
    unsafe {
        let xv = &*x_p; let gv = &*g_p; let bv = &*beta_p;
        let yv = &mut *y_p; let mv = &mut *mean_p; let rv = &mut *rstd_p;
        ctx().layer_norm_fwd.clone()
            .launch(cfg, (xv, gv, bv, yv, mv, rv, b, d, eps))
            .expect("launch layer_norm_fwd");
    }
    0
}

/// LayerNorm backward to `dx`. Gamma/beta grads are NOT produced — sufficient
/// for frozen-norm experiments. Inputs are the cached `mean` + `rstd` from
/// the forward pass plus the upstream `dy`.
#[no_mangle] pub extern "C" fn aether_op_layer_norm_backward_dx_f32_cuda(
    x: i64, gamma: i64, mean: i64, rstd: i64, dy: i64, dx: i64,
    b: c_int, d: c_int,
) -> c_int {
    let (Some(ix), Some(ig), Some(im), Some(ir), Some(idy), Some(idx_)) = (
        handle_to_idx(x), handle_to_idx(gamma), handle_to_idx(mean),
        handle_to_idx(rstd), handle_to_idx(dy), handle_to_idx(dx))
        else { return -1; };
    let bs = unsafe { bufs() };
    let x_p  = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let g_p  = bs[ig].as_ref().unwrap() as *const CudaSlice<f32>;
    let m_p  = bs[im].as_ref().unwrap() as *const CudaSlice<f32>;
    let r_p  = bs[ir].as_ref().unwrap() as *const CudaSlice<f32>;
    let dy_p = bs[idy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dx_p = bs[idx_].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(b as u32);
    unsafe {
        let xv = &*x_p; let gv = &*g_p; let mv = &*m_p; let rv = &*r_p;
        let dyv = &*dy_p; let dxv = &mut *dx_p;
        ctx().layer_norm_bwd_dx.clone()
            .launch(cfg, (xv, gv, mv, rv, dyv, dxv, b, d))
            .expect("launch layer_norm_bwd_dx");
    }
    0
}

/// LayerNorm parameter backward: produces dgamma + dbeta of shape [D] each
/// from cached forward-pass mean/rstd plus upstream dy.
#[no_mangle] pub extern "C" fn aether_op_layer_norm_backward_params_f32_cuda(
    x: i64, mean: i64, rstd: i64, dy: i64, dgamma: i64, dbeta: i64,
    b: c_int, d: c_int,
) -> c_int {
    let (Some(ix), Some(im), Some(ir), Some(idy), Some(idg), Some(idb)) = (
        handle_to_idx(x), handle_to_idx(mean), handle_to_idx(rstd),
        handle_to_idx(dy), handle_to_idx(dgamma), handle_to_idx(dbeta))
        else { return -1; };
    let bs = unsafe { bufs() };
    let x_p  = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let m_p  = bs[im].as_ref().unwrap() as *const CudaSlice<f32>;
    let r_p  = bs[ir].as_ref().unwrap() as *const CudaSlice<f32>;
    let dy_p = bs[idy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dg_p = bs[idg].as_mut().unwrap() as *mut CudaSlice<f32>;
    let db_p = bs[idb].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(d as u32);
    unsafe {
        let xv = &*x_p; let mv = &*m_p; let rv = &*r_p; let dyv = &*dy_p;
        let dgv = &mut *dg_p; let dbv = &mut *db_p;
        ctx().layer_norm_bwd_params.clone()
            .launch(cfg, (xv, mv, rv, dyv, dgv, dbv, b, d))
            .expect("launch layer_norm_bwd_params");
    }
    0
}

/// Fused add+layer_norm: `y = LN(a + b; gamma, beta)` over the last dim,
/// with `mean_out` + `rstd_out` cached for backward.
#[no_mangle] pub extern "C" fn aether_op_add_layer_norm_f32_cuda(
    a: i64, b: i64, gamma: i64, beta: i64, y: i64,
    mean_out: i64, rstd_out: i64,
    eps: f32, bsz: c_int, d: c_int,
) -> c_int {
    let (Some(ia), Some(ib), Some(igamma), Some(ibeta), Some(iy), Some(im), Some(ir)) = (
        handle_to_idx(a), handle_to_idx(b), handle_to_idx(gamma), handle_to_idx(beta),
        handle_to_idx(y), handle_to_idx(mean_out), handle_to_idx(rstd_out))
        else { return -1; };
    let bs = unsafe { bufs() };
    let a_p     = bs[ia].as_ref().unwrap() as *const CudaSlice<f32>;
    let b_p     = bs[ib].as_ref().unwrap() as *const CudaSlice<f32>;
    let g_p     = bs[igamma].as_ref().unwrap() as *const CudaSlice<f32>;
    let beta_p  = bs[ibeta].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p     = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let mean_p  = bs[im].as_mut().unwrap() as *mut CudaSlice<f32>;
    let rstd_p  = bs[ir].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(bsz as u32);
    unsafe {
        let av = &*a_p; let bv = &*b_p; let gv = &*g_p; let betav = &*beta_p;
        let yv = &mut *y_p; let mv = &mut *mean_p; let rv = &mut *rstd_p;
        ctx().add_layer_norm_fwd.clone()
            .launch(cfg, (av, bv, gv, betav, yv, mv, rv, bsz, d, eps))
            .expect("launch add_layer_norm_fwd");
    }
    0
}

/// Row-wise softmax across last dim. `x` and `y` are [B, D] device handles.
#[no_mangle] pub extern "C" fn aether_op_softmax_f32_cuda(
    x: i64, y: i64, b: c_int, d: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(b as u32);
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p;
        ctx().softmax_f32.clone().launch(cfg, (xv, yv, b, d)).expect("launch softmax");
    }
    0
}

/// Row-wise softmax backward. `y` and `dy` are [B, D] forward output / upstream
/// gradient; `dx` is the produced [B, D] downstream gradient.
#[no_mangle] pub extern "C" fn aether_op_softmax_backward_f32_cuda(
    y: i64, dy: i64, dx: i64, b: c_int, d: c_int,
) -> c_int {
    let (Some(iy), Some(idy), Some(idx_)) = (handle_to_idx(y), handle_to_idx(dy), handle_to_idx(dx))
        else { return -1; };
    let bs = unsafe { bufs() };
    let y_p  = bs[iy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dy_p = bs[idy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dx_p = bs[idx_].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(b as u32);
    unsafe {
        let yv = &*y_p; let dyv = &*dy_p; let dxv = &mut *dx_p;
        ctx().softmax_bwd.clone().launch(cfg, (yv, dyv, dxv, b, d)).expect("launch softmax_bwd");
    }
    0
}

/// Fused softmax-backward + in-place scale. Combines `softmax_backward(...)`
/// followed by `dx.scale(s)` — emitted by the MIR fusion pass when it
/// detects that exact two-statement pattern in attention backward.
#[no_mangle] pub extern "C" fn aether_op_softmax_backward_scaled_f32_cuda(
    y: i64, dy: i64, dx: i64, s: f32, b: c_int, d: c_int,
) -> c_int {
    let (Some(iy), Some(idy), Some(idx_)) = (handle_to_idx(y), handle_to_idx(dy), handle_to_idx(dx))
        else { return -1; };
    let bs = unsafe { bufs() };
    let y_p  = bs[iy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dy_p = bs[idy].as_ref().unwrap() as *const CudaSlice<f32>;
    let dx_p = bs[idx_].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(b as u32);
    unsafe {
        let yv = &*y_p; let dyv = &*dy_p; let dxv = &mut *dx_p;
        ctx().softmax_bwd_scaled.clone()
            .launch(cfg, (yv, dyv, dxv, s, b, d))
            .expect("launch softmax_bwd_scaled");
    }
    0
}

/// Elementwise in-place scale: `x[i] *= s`. Useful for the Q@K^T / sqrt(d_k)
/// step in attention.
#[no_mangle] pub extern "C" fn aether_op_scale_f32_cuda(
    x: i64, s: f32, n: c_int,
) -> c_int {
    let Some(ix) = handle_to_idx(x) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let xv = &mut *x_p;
        ctx().scale_f32.clone().launch(cfg, (xv, s, n)).expect("launch scale");
    }
    0
}

/// Pure-Rust replacement for libm::powf that doesn't link in libm itself
/// (avoids extra deps in the CUDA-feature build).
fn libm_powf(base: f32, n: f32) -> f32 {
    // Integer fast path — adamw bias correction always passes integer step.
    let ni = n as i64;
    if ni as f32 == n && ni >= 0 {
        let mut r = 1.0f32; let mut b = base; let mut k = ni as u64;
        while k > 0 { if k & 1 == 1 { r *= b; } b *= b; k >>= 1; }
        return r;
    }
    base.powf(n)
}

/// Wallclock in microseconds since some monotonic epoch. For bench timers.
#[no_mangle] pub extern "C" fn aether_wall_us() -> i64 {
    use std::time::Instant;
    static T0: OnceLock<Instant> = OnceLock::new();
    let t0 = T0.get_or_init(Instant::now);
    Instant::now().duration_since(*t0).as_micros() as i64
}

/// Block until all queued device work has completed. Required before
/// timing measurements that span GPU kernel launches.
#[no_mangle] pub extern "C" fn aether_dev_sync() -> c_int {
    let _ = ctx().device.synchronize();
    0
}

// =====================================================================
// FR-17.14-extra-deepest-graph -- CUDA graph capture / instantiate / launch.
//
// Builds on cudarc 0.13's raw driver sys bindings to record the
// per-token autoregressive forward into a graph that's reused for
// every decode step. Only the device-side step_args (pos, cur_seq)
// needs to change between launches; the graph itself is fixed.
//
// Trades:
//  - one-time capture cost (~few ms)
// for
//  - per-step ~3 ms of host-side launch overhead saved (370 kernels x ~8 us each)
//
// Process model:
//  - One graph + one exec per process.
//  - aether_dev_graph_begin() puts the device stream into capture mode.
//  - aether_dev_graph_end() ends capture and instantiates the graph.
//  - aether_dev_graph_launch() replays it.
//  - aether_dev_graph_destroy() releases both handles.
// =====================================================================

struct GraphHandles {
    graph: Option<cudarc::driver::sys::CUgraph>,
    exec:  Option<cudarc::driver::sys::CUgraphExec>,
}
struct GraphState(UnsafeCell<GraphHandles>);
unsafe impl Sync for GraphState {}
static GRAPH_STATE: GraphState = GraphState(UnsafeCell::new(GraphHandles { graph: None, exec: None }));
unsafe fn graph_state() -> &'static mut GraphHandles { &mut *GRAPH_STATE.0.get() }

/// Put the device stream into thread-local capture mode. All subsequent
/// kernel launches up to `aether_dev_graph_end` are recorded into a
/// CUDA graph. Returns 0 on success.
#[no_mangle] pub extern "C" fn aether_dev_graph_begin() -> c_int {
    let stream = *ctx().device.cu_stream();
    unsafe {
        let lib = cudarc::driver::sys::lib();
        let rc = lib.cuStreamBeginCapture_v2(
            stream,
            cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
        );
        if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            eprintln!("[aether_dev_graph_begin] cuStreamBeginCapture_v2 -> {:?}", rc);
            return -1;
        }
    }
    0
}

/// End capture, instantiate the graph into an executable graph, and
/// store the handles globally. Returns 0 on success.
#[no_mangle] pub extern "C" fn aether_dev_graph_end() -> c_int {
    let stream = *ctx().device.cu_stream();
    unsafe {
        let lib = cudarc::driver::sys::lib();
        let mut g: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
        let rc = lib.cuStreamEndCapture(stream, &mut g);
        if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS || g.is_null() { return -1; }

        let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        let rc = lib.cuGraphInstantiateWithFlags(&mut exec, g, 0);
        if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS || exec.is_null() {
            let _ = lib.cuGraphDestroy(g);
            return -2;
        }
        let st = graph_state();
        // Destroy any previously held handles.
        if let Some(old_exec) = st.exec.take() { let _ = lib.cuGraphExecDestroy(old_exec); }
        if let Some(old_graph) = st.graph.take() { let _ = lib.cuGraphDestroy(old_graph); }
        st.graph = Some(g);
        st.exec  = Some(exec);
    }
    0
}

/// Replay the captured graph on the device stream. Async w.r.t. the host;
/// caller is responsible for sync if a result is needed before next launch.
#[no_mangle] pub extern "C" fn aether_dev_graph_launch() -> c_int {
    let stream = *ctx().device.cu_stream();
    unsafe {
        let lib = cudarc::driver::sys::lib();
        let st = graph_state();
        let Some(exec) = st.exec else { return -1; };
        let rc = lib.cuGraphLaunch(exec, stream);
        if rc != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS { return -2; }
    }
    0
}

/// Free the captured graph + exec. Safe to call multiple times.
#[no_mangle] pub extern "C" fn aether_dev_graph_destroy() -> c_int {
    unsafe {
        let lib = cudarc::driver::sys::lib();
        let st = graph_state();
        if let Some(exec) = st.exec.take() { let _ = lib.cuGraphExecDestroy(exec); }
        if let Some(g) = st.graph.take() { let _ = lib.cuGraphDestroy(g); }
    }
    0
}

// =====================================================================
// matt-voice deploy — Qwen forward kernels on device. The non-matmul
// ops (RMSNorm / RoPE / GQA / SiLU / element-wise mul + add / bias_add)
// run entirely on the GPU, so the per-block forward only crosses PCIe
// at block boundaries (or never, with a full GPU-resident weight cache).
// =====================================================================

/// FR-17.5-extra — RMSNorm forward on device.
/// y[r, i] = x[r, i] * gamma[i] / sqrt(mean(x[r, :]^2) + eps)
#[no_mangle] pub extern "C" fn aether_op_rms_norm_f32_cuda(
    x: i64, gamma: i64, out: i64, eps: f32, rows: c_int, d: c_int,
) -> c_int {
    let (Some(ix), Some(ig), Some(io)) = (handle_to_idx(x), handle_to_idx(gamma), handle_to_idx(out))
        else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let g_p = bs[ig].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[io].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(rows as u32);
    unsafe {
        let xv = &*x_p; let gv = &*g_p; let ov = &mut *o_p;
        ctx().rms_norm_fwd.clone()
            .launch(cfg, (xv, gv, ov, eps, rows, d))
            .expect("launch rms_norm_fwd");
    }
    0
}

/// FR-17.13-extra — RoPE applied in place to `[seq, n_heads, head_dim]`.
/// Llama-style half-half pair layout. `base` is the rotary base (Qwen2.5: 1e6).
/// `pos_start` is the absolute position of the first row.
#[no_mangle] pub extern "C" fn aether_op_rope_apply_f32_cuda(
    x: i64, seq: c_int, n_heads: c_int, head_dim: c_int,
    base: f32, pos_start: c_int,
) -> c_int {
    let Some(ix) = handle_to_idx(x) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (seq * n_heads * (head_dim / 2)) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let xv = &mut *x_p;
        ctx().rope_apply.clone()
            .launch(cfg, (xv, seq, n_heads, head_dim, base, pos_start))
            .expect("launch rope_apply");
    }
    0
}

/// FR-17.14-extra-deepest-graph — rope_apply variant reading pos from a
/// device-side step_args[0]. Used inside the captured CUDA graph for
/// autoregressive decoding -- only the step_args buffer needs to be
/// h2d-updated per step; the graph itself is reused.
#[no_mangle] pub extern "C" fn aether_op_rope_apply_devarg_f32_cuda(
    x: i64, seq: c_int, n_heads: c_int, head_dim: c_int,
    base: f32, step_args_i32: i64,
) -> c_int {
    let Some(ix) = handle_to_idx(x) else { return -1; };
    let Some(is) = handle_to_i32_idx(step_args_i32) else { return -1; };
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let s_p = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let total = (seq * n_heads * (head_dim / 2)) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let xv = &mut *x_p; let sv = &*s_p;
        ctx().rope_apply_devarg.clone()
            .launch(cfg, (xv, seq, n_heads, head_dim, base, sv))
            .expect("launch rope_apply_devarg");
    }
    0
}

/// FR-17.14-extra-deepest-graph — append_kv variant reading pos from step_args[0].
#[no_mangle] pub extern "C" fn aether_op_append_kv_devarg_f32_cuda(
    k_new_dev: i64, v_new_dev: i64,
    k_cache_dev: i64, v_cache_dev: i64,
    d_kv: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_kn) = handle_to_idx(k_new_dev) else { return -1; };
    let Some(i_vn) = handle_to_idx(v_new_dev) else { return -1; };
    let Some(i_kc) = handle_to_idx(k_cache_dev) else { return -1; };
    let Some(i_vc) = handle_to_idx(v_cache_dev) else { return -1; };
    let Some(is)   = handle_to_i32_idx(step_args_i32) else { return -1; };
    if d_kv <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let kn = bs[i_kn].as_ref().unwrap() as *const CudaSlice<f32>;
    let vn = bs[i_vn].as_ref().unwrap() as *const CudaSlice<f32>;
    let kc = bs[i_kc].as_mut().unwrap() as *mut CudaSlice<f32>;
    let vc = bs[i_vc].as_mut().unwrap() as *mut CudaSlice<f32>;
    let sp = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let cfg = LaunchConfig::for_num_elems(d_kv as u32);
    unsafe {
        let knr = &*kn; let vnr = &*vn;
        let kcm = &mut *kc; let vcm = &mut *vc;
        let sv = &*sp;
        ctx().append_kv_devarg.clone()
            .launch(cfg, (knr, vnr, kcm, vcm, d_kv, sv))
            .expect("launch append_kv_devarg");
    }
    0
}

/// FR-17.14-extra-deepest-graph — attention_seq1 variant reading cur_seq
/// from step_args[1]. Launched with max_shmem bytes (max_seq * 4); the
/// kernel only uses cur_seq * 4 of them. Graph-safe.
#[no_mangle] pub extern "C" fn aether_op_attention_seq1_devarg_f32_cuda(
    q_dev: i64, k_cache: i64, v_cache: i64, attn_out: i64,
    n_q_heads: c_int, n_kv_heads: c_int, head_dim: c_int,
    scale: f32, max_seq: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_dev) else { return -1; };
    let Some(i_kc) = handle_to_idx(k_cache) else { return -1; };
    let Some(i_vc) = handle_to_idx(v_cache) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out) else { return -1; };
    let Some(is)   = handle_to_i32_idx(step_args_i32) else { return -1; };
    if n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0 { return -1; }
    if (n_q_heads % n_kv_heads) != 0 { return -2; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kc_p = bs[i_kc].as_ref().unwrap() as *const CudaSlice<f32>;
    let vc_p = bs[i_vc].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let sp  = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let shmem = (max_seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_q_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kcv = &*kc_p; let vcv = &*vc_p; let ov = &mut *o_p;
        let sv = &*sp;
        ctx().attention_seq1_devarg.clone()
            .launch(cfg, (qv, kcv, vcv, ov, n_q_heads, n_kv_heads, head_dim, scale, sv))
            .expect("launch attention_seq1_devarg");
    }
    0
}

/// FR-19.4-extra — paged append_kv. Writes K/V at the physical row located via
/// `page_table[pos / block_size] * block_size + (pos % block_size)`.
#[no_mangle] pub extern "C" fn aether_op_paged_append_kv_devarg_f32_cuda(
    k_new_dev: i64, v_new_dev: i64,
    k_pool_dev: i64, v_pool_dev: i64,
    page_table_dev: i64,
    d_kv: c_int, block_size: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_kn) = handle_to_idx(k_new_dev) else { return -1; };
    let Some(i_vn) = handle_to_idx(v_new_dev) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool_dev) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool_dev) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_dev) else { return -1; };
    let Some(is)   = handle_to_i32_idx(step_args_i32) else { return -1; };
    if d_kv <= 0 || block_size <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let kn = bs[i_kn].as_ref().unwrap() as *const CudaSlice<f32>;
    let vn = bs[i_vn].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp = bs[i_kp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let vp = bs[i_vp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let pt = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let sp = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let cfg = LaunchConfig::for_num_elems(d_kv as u32);
    unsafe {
        let knr = &*kn; let vnr = &*vn;
        let kpm = &mut *kp; let vpm = &mut *vp;
        let ptv = &*pt; let sv = &*sp;
        paged_ctx().paged_append_kv_devarg.clone()
            .launch(cfg, (knr, vnr, kpm, vpm, ptv, d_kv, block_size, sv))
            .expect("launch paged_append_kv_devarg");
    }
    0
}

/// FR-19.4-extra — paged attention_seq1. Reads K[t], V[t] from
/// `pool + (page_table[t / block_size] * block_size + (t % block_size)) * d_kv`.
#[no_mangle] pub extern "C" fn aether_op_paged_attention_seq1_devarg_f32_cuda(
    q_dev: i64, k_pool: i64, v_pool: i64,
    page_table_dev: i64,
    attn_out: i64,
    n_q_heads: c_int, n_kv_heads: c_int, head_dim: c_int, block_size: c_int,
    scale: f32, max_seq: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_dev) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_dev) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out) else { return -1; };
    let Some(is)   = handle_to_i32_idx(step_args_i32) else { return -1; };
    if n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0 || block_size <= 0 { return -1; }
    if (n_q_heads % n_kv_heads) != 0 { return -2; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp_p = bs[i_kp].as_ref().unwrap() as *const CudaSlice<f32>;
    let vp_p = bs[i_vp].as_ref().unwrap() as *const CudaSlice<f32>;
    let pt_p = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let sp  = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let shmem = (max_seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_q_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kpv = &*kp_p; let vpv = &*vp_p; let ptv = &*pt_p;
        let ov = &mut *o_p; let sv = &*sp;
        paged_ctx().paged_attention_seq1_devarg.clone()
            .launch(cfg, (qv, kpv, vpv, ptv, ov, n_q_heads, n_kv_heads, head_dim, block_size, scale, sv))
            .expect("launch paged_attention_seq1_devarg");
    }
    0
}

/// FR-17-extra-moe-fwd — Q4_K matmul against ONE expert's slice of a
/// concatenated MoE expert weight buffer.  `expert_idx ∈ [0, n_experts)`.
/// Per-expert slice byte offset is computed internally as
/// `expert_idx * n_out * (n_in/256) * 144`.
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_expert_matmul_seq1_cuda(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_base) else { return -1; };
    if n_out <= 0 || blocks_per_row <= 0 || expert_idx < 0 { return -1; }
    let expert_offset_blocks = expert_idx * n_out * blocks_per_row;
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_q4k_expert_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_out, blocks_per_row, expert_offset_blocks))
            .expect("launch fused_q4k_expert_matmul_seq1");
    }
    0
}

/// FR-17-extra-mla-fwd MoE — Q8_0 expert-variant matmul.  Shape mirrors
/// the Q4_K expert kernel; `blocks_per_row` counts 32-elem Q8_0 blocks.
#[no_mangle] pub extern "C" fn aether_op_fused_q8_0_expert_matmul_seq1_cuda(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_base) else { return -1; };
    if n_out <= 0 || blocks_per_row <= 0 || expert_idx < 0 { return -1; }
    let expert_offset_blocks = expert_idx * n_out * blocks_per_row;
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_q8_0_expert_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_out, blocks_per_row, expert_offset_blocks))
            .expect("launch fused_q8_0_expert_matmul_seq1");
    }
    0
}

/// FR-17-extra-mla-fwd MoE — Q5_0 expert-variant matmul.
#[no_mangle] pub extern "C" fn aether_op_fused_q5_0_expert_matmul_seq1_cuda(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_base) else { return -1; };
    if n_out <= 0 || blocks_per_row <= 0 || expert_idx < 0 { return -1; }
    let expert_offset_blocks = expert_idx * n_out * blocks_per_row;
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_q5_0_expert_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_out, blocks_per_row, expert_offset_blocks))
            .expect("launch fused_q5_0_expert_matmul_seq1");
    }
    0
}

/// FR-17-extra-moe-quant-dispatch MoE — IQ3_S expert-variant matmul.
/// 110-byte 256-elem blocks.  Per-expert offset = `expert_idx * n_out *
/// blocks_per_row * 110` bytes.  Used by GLM-4.7-flash MoE expert tensors
/// quantised to IQ3_S (dt=21).
#[no_mangle] pub extern "C" fn aether_op_fused_iq3_s_expert_matmul_seq1_cuda(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_base) else { return -1; };
    if n_out <= 0 || blocks_per_row <= 0 || expert_idx < 0 { return -1; }
    let expert_offset_blocks = expert_idx * n_out * blocks_per_row;
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_iq3_s_expert_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_out, blocks_per_row, expert_offset_blocks))
            .expect("launch fused_iq3_s_expert_matmul_seq1");
    }
    0
}

/// FR-17-extra-moe-quant-dispatch-iq4xs MoE — IQ4_XS expert-variant matmul.
/// 136-byte 256-elem blocks.  Per-expert offset = `expert_idx * n_out *
/// blocks_per_row * 136` bytes.  Used by GLM-4.7-flash MoE expert tensors
/// quantised to IQ4_XS (dt=23) — second non-Q4_K dtype the IQ3_XXS GGUF hits.
#[no_mangle] pub extern "C" fn aether_op_fused_iq4_xs_expert_matmul_seq1_cuda(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_base) else { return -1; };
    if n_out <= 0 || blocks_per_row <= 0 || expert_idx < 0 { return -1; }
    let expert_offset_blocks = expert_idx * n_out * blocks_per_row;
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_iq4_xs_expert_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_out, blocks_per_row, expert_offset_blocks))
            .expect("launch fused_iq4_xs_expert_matmul_seq1");
    }
    0
}

/// FR-17-extra-moe-quant-dispatch-iq3xxs MoE — IQ3_XXS expert-variant matmul.
/// 98-byte 256-elem blocks.  Per-expert offset = `expert_idx * n_out *
/// blocks_per_row * 98` bytes.  Third non-Q4_K dtype to surface in the GLM
/// IQ3_XXS GGUF — namesake quant, used for many gate/up/down expert tensors.
#[no_mangle] pub extern "C" fn aether_op_fused_iq3_xxs_expert_matmul_seq1_cuda(
    x: i64, w_base: i64, y: i64,
    n_out: c_int, blocks_per_row: c_int, expert_idx: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_base) else { return -1; };
    if n_out <= 0 || blocks_per_row <= 0 || expert_idx < 0 { return -1; }
    let expert_offset_blocks = expert_idx * n_out * blocks_per_row;
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_iq3_xxs_expert_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_out, blocks_per_row, expert_offset_blocks))
            .expect("launch fused_iq3_xxs_expert_matmul_seq1");
    }
    0
}

/// FR-17-extra-f16-fwd — matmul against F16-stored weights.
/// y[o] = sum_i x[i] * f16_to_f32(w[o, i])  for o in 0..n_out, x in F32.
#[no_mangle] pub extern "C" fn aether_op_fused_f16_matmul_seq1_cuda(
    x: i64, w_f16: i64, y: i64, n_in: c_int, n_out: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_f16) else { return -1; };
    if n_in <= 0 || n_out <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_f16_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_in, n_out))
            .expect("launch fused_f16_matmul_seq1");
    }
    0
}

/// FR-17-extra-mla-fwd companion — F32-weight matmul for seq=1 decode.
/// Weight is stored as raw float32 in a u8-registered buffer (same upload
/// path as F16/Q4_K/etc.).  Used by GLM-4.7-flash which keeps some shared-
/// expert and head-adjacent tensors as F32.
#[no_mangle] pub extern "C" fn aether_op_fused_f32_matmul_seq1_cuda(
    x: i64, w_f32: i64, y: i64, n_in: c_int, n_out: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let Some(iw) = handle_to_u8_idx(w_f32) else { return -1; };
    if n_in <= 0 || n_out <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let x_p = bs[ix].as_ref().unwrap() as *const CudaSlice<f32>;
    let y_p = bs[iy].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[iw].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (n_out as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let xv = &*x_p; let yv = &mut *y_p; let wv = &*w_p;
        ctx().fused_f32_matmul_seq1.clone()
            .launch(cfg, (xv, wv, yv, n_in, n_out))
            .expect("launch fused_f32_matmul_seq1");
    }
    0
}

/// FR-19.5-extra-deep — batched paged append_kv: B (k_new, v_new) pairs
/// written at `step_args[0]` against B page tables in one launch.
#[no_mangle] pub extern "C" fn aether_op_batched_paged_append_kv_seqB_devarg_f32_cuda(
    k_new_batch: i64, v_new_batch: i64,
    k_pool: i64, v_pool: i64,
    page_table_batch_dev: i64,
    batch: c_int, d_kv: c_int, block_size: c_int, page_table_stride: c_int,
    step_args_i32: i64,
) -> c_int {
    let Some(i_kn) = handle_to_idx(k_new_batch) else { return -1; };
    let Some(i_vn) = handle_to_idx(v_new_batch) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_batch_dev) else { return -1; };
    let Some(is)   = handle_to_i32_idx(step_args_i32) else { return -1; };
    if batch <= 0 || d_kv <= 0 || block_size <= 0 || page_table_stride <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let kn = bs[i_kn].as_ref().unwrap() as *const CudaSlice<f32>;
    let vn = bs[i_vn].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp = bs[i_kp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let vp = bs[i_vp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let pt = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let sp = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let threads_per_block: u32 = 256;
    let blocks_per_req: u32 = ((d_kv as u32) + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_req, batch as u32, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let knr = &*kn; let vnr = &*vn;
        let kpm = &mut *kp; let vpm = &mut *vp;
        let ptv = &*pt; let sv = &*sp;
        paged_ctx().batched_paged_append_kv_seqB_devarg.clone()
            .launch(cfg, (knr, vnr, kpm, vpm, ptv,
                          d_kv, block_size, page_table_stride, sv))
            .expect("launch batched_paged_append_kv_seqB_devarg");
    }
    0
}

/// FR-17-extra-gemma-fwd — flexible paged attention.  Accepts head_dim
/// values that aren't multiples of 32 (e.g. Gemma3's head_dim=168) and
/// optional sliding-window scope (sliding_window > 0 restricts the t-range).
/// Strictly a superset of paged_attention_seq1_devarg's behavior at the
/// cost of two extra bounds checks per element in the inner loops.
#[no_mangle] pub extern "C" fn aether_op_paged_attention_flex_devarg_f32_cuda(
    q_dev: i64, k_pool: i64, v_pool: i64,
    page_table_dev: i64, attn_out: i64,
    n_q_heads: c_int, n_kv_heads: c_int, head_dim: c_int, block_size: c_int,
    sliding_window: c_int,
    scale: f32, max_seq: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_dev) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_dev) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out) else { return -1; };
    let Some(is)  = handle_to_i32_idx(step_args_i32) else { return -1; };
    if n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0 || block_size <= 0 { return -1; }
    if (n_q_heads % n_kv_heads) != 0 { return -2; }
    if head_dim > 256 { return -3; }  // q_local[8] × per_lane=8 maxes out
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp_p = bs[i_kp].as_ref().unwrap() as *const CudaSlice<f32>;
    let vp_p = bs[i_vp].as_ref().unwrap() as *const CudaSlice<f32>;
    let pt_p = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let sp  = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let shmem = (max_seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_q_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kpv = &*kp_p; let vpv = &*vp_p; let ptv = &*pt_p;
        let ov = &mut *o_p; let sv = &*sp;
        paged_ctx().paged_attention_flex_devarg.clone()
            .launch(cfg, (qv, kpv, vpv, ptv, ov,
                          n_q_heads, n_kv_heads, head_dim, block_size,
                          sliding_window, scale, sv))
            .expect("launch paged_attention_flex_devarg");
    }
    0
}

/// FR-17-extra-mla-fwd — DeepSeek-V2 Multi-head Latent Attention kernel.
/// Differs from `paged_attention_seq1_devarg` in that Q/K share one per-head
/// dim (`qk_head_dim`, e.g. 192) while V uses a different per-head dim
/// (`v_head_dim`, e.g. 128).  Caller is responsible for projecting the
/// per-token latent c_kv up to per-head K (`n_heads * qk_head_dim`) and per-
/// head V (`n_heads * v_head_dim`) before each step's `paged_append_kv` call,
/// and for laying out the K / V pools with the matching row strides.
///
/// Grid: (n_heads, 1, 1).  Block: (32, 1, 1).  Shared mem: max_seq * 4 bytes.
#[no_mangle] pub extern "C" fn aether_op_paged_attention_mla_devarg_f32_cuda(
    q_dev: i64, k_pool: i64, v_pool: i64,
    page_table_dev: i64, attn_out: i64,
    n_heads: c_int, qk_head_dim: c_int, v_head_dim: c_int, block_size: c_int,
    scale: f32, max_seq: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_dev) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_dev) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out) else { return -1; };
    let Some(is)  = handle_to_i32_idx(step_args_i32) else { return -1; };
    if n_heads <= 0 || qk_head_dim <= 0 || v_head_dim <= 0
        || max_seq <= 0 || block_size <= 0 { return -1; }
    // q_local[20] / out_local[20] in paged_attention_mla_devarg → 20 × 32 = 640
    // max per-head dim.  Bumped from the original 256 cap in commit b6bfacf
    // (the kernel arrays were bumped; this host-side guard was missed).
    if qk_head_dim > 640 || v_head_dim > 640 { return -3; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp_p = bs[i_kp].as_ref().unwrap() as *const CudaSlice<f32>;
    let vp_p = bs[i_vp].as_ref().unwrap() as *const CudaSlice<f32>;
    let pt_p = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let sp  = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let shmem = (max_seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_heads as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kpv = &*kp_p; let vpv = &*vp_p; let ptv = &*pt_p;
        let ov = &mut *o_p; let sv = &*sp;
        paged_ctx().paged_attention_mla_devarg.clone()
            .launch(cfg, (qv, kpv, vpv, ptv, ov,
                          n_heads, qk_head_dim, v_head_dim, block_size,
                          scale, sv))
            .expect("launch paged_attention_mla_devarg");
    }
    0
}

/// FR-17-extra-mla-fwd — paged append with independent K / V row strides.
/// MLA's K-per-token is `n_heads * qk_head_dim` (qk_nope + qk_rope) and V-
/// per-token is `n_heads * v_head_dim`.  Step args layout matches the rest
/// of the paged kernels: `step_args[0] = pos`, `step_args[1] = cur_seq`.
#[no_mangle] pub extern "C" fn aether_op_paged_append_kv_mla_devarg_f32_cuda(
    k_new: i64, v_new: i64, k_pool: i64, v_pool: i64,
    page_table_dev: i64,
    d_k_row: c_int, d_v_row: c_int, block_size: c_int,
    step_args_i32: i64,
) -> c_int {
    let Some(i_kn) = handle_to_idx(k_new) else { return -1; };
    let Some(i_vn) = handle_to_idx(v_new) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_dev) else { return -1; };
    let Some(is)  = handle_to_i32_idx(step_args_i32) else { return -1; };
    if d_k_row <= 0 || d_v_row <= 0 || block_size <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let kn_p = bs[i_kn].as_ref().unwrap() as *const CudaSlice<f32>;
    let vn_p = bs[i_vn].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp_p = bs[i_kp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let vp_p = bs[i_vp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let pt_p = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let sp   = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let span = std::cmp::max(d_k_row, d_v_row) as u32;
    let block = 128u32;
    let cfg = LaunchConfig {
        grid_dim:  ((span + block - 1) / block, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let knv = &*kn_p; let vnv = &*vn_p;
        let kpv = &mut *kp_p; let vpv = &mut *vp_p;
        let ptv = &*pt_p; let sv = &*sp;
        paged_ctx().paged_append_kv_mla_devarg.clone()
            .launch(cfg, (knv, vnv, kpv, vpv, ptv,
                          d_k_row, d_v_row, block_size, sv))
            .expect("launch paged_append_kv_mla_devarg");
    }
    0
}

/// FR-17-extra-mla-fwd — split the kv_a_mqa output [kv_lora_rank +
/// qk_rope_head_dim] into the latent c_kv slice and the shared k_rope slice.
#[no_mangle] pub extern "C" fn aether_op_mla_split_kv_a_f32_cuda(
    kv_a: i64, c_kv: i64, k_rope: i64,
    kv_lora_rank: c_int, qk_rope_head_dim: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(kv_a) else { return -1; };
    let Some(i_c) = handle_to_idx(c_kv) else { return -1; };
    let Some(i_k) = handle_to_idx(k_rope) else { return -1; };
    if kv_lora_rank <= 0 || qk_rope_head_dim <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let c_p = bs[i_c].as_mut().unwrap() as *mut CudaSlice<f32>;
    let k_p = bs[i_k].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (kv_lora_rank + qk_rope_head_dim) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let av = &*a_p; let cv = &mut *c_p; let kv = &mut *k_p;
        paged_ctx().mla_split_kv_a.clone()
            .launch(cfg, (av, cv, kv, kv_lora_rank, qk_rope_head_dim))
            .expect("launch mla_split_kv_a");
    }
    0
}

/// FR-17-extra-mla-fwd — assemble per-head K = [K_nope | k_rope_shared].
#[no_mangle] pub extern "C" fn aether_op_mla_assemble_k_f32_cuda(
    kv_b_out: i64, k_rope: i64, k_row: i64,
    n_heads: c_int, qk_nope_head_dim: c_int,
    qk_rope_head_dim: c_int, v_head_dim: c_int,
) -> c_int {
    let Some(i_b) = handle_to_idx(kv_b_out) else { return -1; };
    let Some(i_kr) = handle_to_idx(k_rope) else { return -1; };
    let Some(i_o) = handle_to_idx(k_row) else { return -1; };
    if n_heads <= 0 || qk_nope_head_dim <= 0 || qk_rope_head_dim <= 0
        || v_head_dim <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let b_p = bs[i_b].as_ref().unwrap() as *const CudaSlice<f32>;
    let r_p = bs[i_kr].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (n_heads * (qk_nope_head_dim + qk_rope_head_dim)) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let bv = &*b_p; let rv = &*r_p; let ov = &mut *o_p;
        paged_ctx().mla_assemble_k.clone()
            .launch(cfg, (bv, rv, ov, n_heads, qk_nope_head_dim,
                          qk_rope_head_dim, v_head_dim))
            .expect("launch mla_assemble_k");
    }
    0
}

/// FR-17-extra-mla-fwd — extract V from kv_b_out [n_heads * (qk_nope +
/// v_head)].
#[no_mangle] pub extern "C" fn aether_op_mla_extract_v_f32_cuda(
    kv_b_out: i64, v_row: i64,
    n_heads: c_int, qk_nope_head_dim: c_int, v_head_dim: c_int,
) -> c_int {
    let Some(i_b) = handle_to_idx(kv_b_out) else { return -1; };
    let Some(i_v) = handle_to_idx(v_row) else { return -1; };
    if n_heads <= 0 || qk_nope_head_dim <= 0 || v_head_dim <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let b_p = bs[i_b].as_ref().unwrap() as *const CudaSlice<f32>;
    let v_p = bs[i_v].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (n_heads * v_head_dim) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let bv = &*b_p; let vv = &mut *v_p;
        paged_ctx().mla_extract_v.clone()
            .launch(cfg, (bv, vv, n_heads, qk_nope_head_dim, v_head_dim))
            .expect("launch mla_extract_v");
    }
    0
}

/// FR-17-extra-mla-fwd — partial-dim RoPE on Q's rope sub-region
/// (the last qk_rope_head_dim columns of each per-head qk_head_dim slice).
#[no_mangle] pub extern "C" fn aether_op_mla_rope_q_partial_f32_cuda(
    q: i64,
    n_heads: c_int, qk_head_dim: c_int, qk_nope_head_dim: c_int,
    base: f32, step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q) else { return -1; };
    let Some(is) = handle_to_i32_idx(step_args_i32) else { return -1; };
    if n_heads <= 0 || qk_head_dim <= 0 || qk_nope_head_dim < 0
        || qk_nope_head_dim >= qk_head_dim { return -1; }
    let qk_rope_head_dim = qk_head_dim - qk_nope_head_dim;
    if (qk_rope_head_dim & 1) != 0 { return -2; }  // half-pair requires even
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_mut().unwrap() as *mut CudaSlice<f32>;
    let s_p = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let total = (n_heads * (qk_rope_head_dim / 2)) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let qv = &mut *q_p; let sv = &*s_p;
        paged_ctx().mla_rope_q_partial.clone()
            .launch(cfg, (qv, n_heads, qk_head_dim, qk_nope_head_dim, base, sv))
            .expect("launch mla_rope_q_partial");
    }
    0
}

/// FR-17-extra-mla-fwd — partial-dim RoPE on the shared k_rope vector.
#[no_mangle] pub extern "C" fn aether_op_mla_rope_k_shared_f32_cuda(
    k_rope: i64, qk_rope_head_dim: c_int,
    base: f32, step_args_i32: i64,
) -> c_int {
    let Some(i_k) = handle_to_idx(k_rope) else { return -1; };
    let Some(is) = handle_to_i32_idx(step_args_i32) else { return -1; };
    if qk_rope_head_dim <= 0 || (qk_rope_head_dim & 1) != 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let k_p = bs[i_k].as_mut().unwrap() as *mut CudaSlice<f32>;
    let s_p = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let total = (qk_rope_head_dim / 2) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let kv = &mut *k_p; let sv = &*s_p;
        paged_ctx().mla_rope_k_shared.clone()
            .launch(cfg, (kv, qk_rope_head_dim, base, sv))
            .expect("launch mla_rope_k_shared");
    }
    0
}

/// FR-17-extra-mla-fwd YaRN — partial-dim RoPE on Q with per-frequency-dim
/// scale factor for YaRN-style long-context extension.
#[no_mangle] pub extern "C" fn aether_op_mla_rope_q_partial_yarn_f32_cuda(
    q: i64,
    n_heads: c_int, qk_head_dim: c_int, qk_nope_head_dim: c_int,
    base: f32,
    yarn_s: f32, yarn_orig_ctx: f32,
    yarn_beta_fast: f32, yarn_beta_slow: f32,
    step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q) else { return -1; };
    let Some(is) = handle_to_i32_idx(step_args_i32) else { return -1; };
    if n_heads <= 0 || qk_head_dim <= 0 || qk_nope_head_dim < 0
        || qk_nope_head_dim >= qk_head_dim { return -1; }
    let qk_rope_head_dim = qk_head_dim - qk_nope_head_dim;
    if (qk_rope_head_dim & 1) != 0 { return -2; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_mut().unwrap() as *mut CudaSlice<f32>;
    let s_p = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let total = (n_heads * (qk_rope_head_dim / 2)) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let qv = &mut *q_p; let sv = &*s_p;
        paged_ctx().mla_rope_q_partial_yarn.clone()
            .launch(cfg, (qv, n_heads, qk_head_dim, qk_nope_head_dim, base,
                          yarn_s, yarn_orig_ctx, yarn_beta_fast, yarn_beta_slow, sv))
            .expect("launch mla_rope_q_partial_yarn");
    }
    0
}

/// FR-17-extra-mla-fwd YaRN — partial-dim RoPE on the shared K vector with
/// per-frequency-dim YaRN scale factor.
#[no_mangle] pub extern "C" fn aether_op_mla_rope_k_shared_yarn_f32_cuda(
    k_rope: i64, qk_rope_head_dim: c_int,
    base: f32,
    yarn_s: f32, yarn_orig_ctx: f32,
    yarn_beta_fast: f32, yarn_beta_slow: f32,
    step_args_i32: i64,
) -> c_int {
    let Some(i_k) = handle_to_idx(k_rope) else { return -1; };
    let Some(is) = handle_to_i32_idx(step_args_i32) else { return -1; };
    if qk_rope_head_dim <= 0 || (qk_rope_head_dim & 1) != 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let k_p = bs[i_k].as_mut().unwrap() as *mut CudaSlice<f32>;
    let s_p = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let total = (qk_rope_head_dim / 2) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let kv = &mut *k_p; let sv = &*s_p;
        paged_ctx().mla_rope_k_shared_yarn.clone()
            .launch(cfg, (kv, qk_rope_head_dim, base,
                          yarn_s, yarn_orig_ctx, yarn_beta_fast, yarn_beta_slow, sv))
            .expect("launch mla_rope_k_shared_yarn");
    }
    0
}

/// FR-17-extra-mla-absorbed — GLM-4.7-flash absorbed-MLA Q absorption +
/// q_pe concat in one launch.  q_proj is the w_q_b output [n_heads * key_mla];
/// w_k_b is the Q8_0-packed per-head [q_nope_per_head × kv_lora_rank × n_heads]
/// weight tensor.  Output q_out has shape [n_heads * (kv_lora_rank + qk_rope)],
/// matching the existing paged_attention_mla kernel's Q layout.
#[no_mangle] pub extern "C" fn aether_op_mla_absorb_q_q8_0_cuda(
    q_proj: i64, w_k_b: i64, q_out: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_q), Some(i_o)) = (handle_to_idx(q_proj), handle_to_idx(q_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_k_b) else { return -1; };
    if n_heads <= 0 || key_mla <= 0 || qk_rope <= 0 || kv_lora_rank <= 0
        || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: ((kv_lora_rank + qk_rope) as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let qv = &*q_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_q_q8_0.clone()
            .launch(cfg, (qv, wv, ov, n_heads, key_mla, qk_rope,
                          kv_lora_rank, blocks_per_row))
            .expect("launch mla_absorb_q_q8_0");
    }
    0
}

/// FR-17-extra-mla-absorbed — GLM-4.7-flash absorbed-MLA V absorption.
/// attn_v is the paged-attention output [n_heads * kv_lora_rank]; w_v_b is
/// the Q8_0-packed per-head [kv_lora_rank × value_mla × n_heads] weight.
/// Output attn_out has shape [n_heads * value_mla] = post-w_o input.
#[no_mangle] pub extern "C" fn aether_op_mla_absorb_v_q8_0_cuda(
    attn_v: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, value_mla: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_a), Some(i_o)) = (handle_to_idx(attn_v), handle_to_idx(attn_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_v_b) else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || value_mla <= 0 || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (value_mla as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_v_q8_0.clone()
            .launch(cfg, (av, wv, ov, n_heads, kv_lora_rank, value_mla, blocks_per_row))
            .expect("launch mla_absorb_v_q8_0");
    }
    0
}

/// FR-17-extra-mla-absorbed — broadcast compressed c_kv (and k_pe for K) to
/// per-head MQA slots in k_row / v_row, so existing paged_attention_mla
/// kernel (which assumes per-head K/V) can run unchanged.
#[no_mangle] pub extern "C" fn aether_op_mla_broadcast_kv_for_mqa_f32_cuda(
    c_kv: i64, k_pe: i64, k_row: i64, v_row: i64,
    n_heads: c_int, kv_lora_rank: c_int, qk_rope: c_int,
) -> c_int {
    let (Some(i_c), Some(i_p), Some(i_k), Some(i_v)) =
        (handle_to_idx(c_kv), handle_to_idx(k_pe),
         handle_to_idx(k_row), handle_to_idx(v_row))
        else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || qk_rope <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let c_p = bs[i_c].as_ref().unwrap() as *const CudaSlice<f32>;
    let p_p = bs[i_p].as_ref().unwrap() as *const CudaSlice<f32>;
    let k_p = bs[i_k].as_mut().unwrap() as *mut CudaSlice<f32>;
    let v_p = bs[i_v].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (n_heads * (kv_lora_rank + qk_rope)) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let cv = &*c_p; let pv = &*p_p; let kv = &mut *k_p; let vv = &mut *v_p;
        paged_ctx().mla_broadcast_kv_for_mqa.clone()
            .launch(cfg, (cv, pv, kv, vv, n_heads, kv_lora_rank, qk_rope))
            .expect("launch mla_broadcast_kv_for_mqa");
    }
    0
}

// =====================================================================
// FR-17-extra-mla-absorbed-dtypes — Q / V absorb wrappers for additional
// w_k_b / w_v_b dtypes (F16, Q4_K, Q5_K, Q6_K, IQ4_NL).  All mirror the
// Q8_0 versions' calling convention; only the device kernel chosen
// differs.  `n_in_per_row` for F16 = element count per output row;
// `blocks_per_row` for block-quants = number of super-blocks per row.
// =====================================================================

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_q_f16_cuda(
    q_proj: i64, w_k_b: i64, q_out: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, n_in_per_row: c_int,
) -> c_int {
    let (Some(i_q), Some(i_o)) = (handle_to_idx(q_proj), handle_to_idx(q_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_k_b) else { return -1; };
    if n_heads <= 0 || key_mla <= 0 || qk_rope <= 0 || kv_lora_rank <= 0
        || n_in_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: ((kv_lora_rank + qk_rope) as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let qv = &*q_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_q_f16.clone()
            .launch(cfg, (qv, wv, ov, n_heads, key_mla, qk_rope,
                          kv_lora_rank, n_in_per_row))
            .expect("launch mla_absorb_q_f16");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_v_f16_cuda(
    attn_v: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, value_mla: c_int, n_in_per_row: c_int,
) -> c_int {
    let (Some(i_a), Some(i_o)) = (handle_to_idx(attn_v), handle_to_idx(attn_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_v_b) else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || value_mla <= 0 || n_in_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (value_mla as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_v_f16.clone()
            .launch(cfg, (av, wv, ov, n_heads, kv_lora_rank, value_mla, n_in_per_row))
            .expect("launch mla_absorb_v_f16");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_q_q4_k_cuda(
    q_proj: i64, w_k_b: i64, q_out: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_q), Some(i_o)) = (handle_to_idx(q_proj), handle_to_idx(q_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_k_b) else { return -1; };
    if n_heads <= 0 || key_mla <= 0 || qk_rope <= 0 || kv_lora_rank <= 0
        || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: ((kv_lora_rank + qk_rope) as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let qv = &*q_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_q_q4_k.clone()
            .launch(cfg, (qv, wv, ov, n_heads, key_mla, qk_rope,
                          kv_lora_rank, blocks_per_row))
            .expect("launch mla_absorb_q_q4_k");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_v_q4_k_cuda(
    attn_v: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, value_mla: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_a), Some(i_o)) = (handle_to_idx(attn_v), handle_to_idx(attn_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_v_b) else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || value_mla <= 0 || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (value_mla as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_v_q4_k.clone()
            .launch(cfg, (av, wv, ov, n_heads, kv_lora_rank, value_mla, blocks_per_row))
            .expect("launch mla_absorb_v_q4_k");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_q_q5_k_cuda(
    q_proj: i64, w_k_b: i64, q_out: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_q), Some(i_o)) = (handle_to_idx(q_proj), handle_to_idx(q_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_k_b) else { return -1; };
    if n_heads <= 0 || key_mla <= 0 || qk_rope <= 0 || kv_lora_rank <= 0
        || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: ((kv_lora_rank + qk_rope) as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let qv = &*q_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_q_q5_k.clone()
            .launch(cfg, (qv, wv, ov, n_heads, key_mla, qk_rope,
                          kv_lora_rank, blocks_per_row))
            .expect("launch mla_absorb_q_q5_k");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_v_q5_k_cuda(
    attn_v: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, value_mla: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_a), Some(i_o)) = (handle_to_idx(attn_v), handle_to_idx(attn_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_v_b) else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || value_mla <= 0 || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (value_mla as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_v_q5_k.clone()
            .launch(cfg, (av, wv, ov, n_heads, kv_lora_rank, value_mla, blocks_per_row))
            .expect("launch mla_absorb_v_q5_k");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_q_q6_k_cuda(
    q_proj: i64, w_k_b: i64, q_out: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_q), Some(i_o)) = (handle_to_idx(q_proj), handle_to_idx(q_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_k_b) else { return -1; };
    if n_heads <= 0 || key_mla <= 0 || qk_rope <= 0 || kv_lora_rank <= 0
        || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: ((kv_lora_rank + qk_rope) as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let qv = &*q_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_q_q6_k.clone()
            .launch(cfg, (qv, wv, ov, n_heads, key_mla, qk_rope,
                          kv_lora_rank, blocks_per_row))
            .expect("launch mla_absorb_q_q6_k");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_v_q6_k_cuda(
    attn_v: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, value_mla: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_a), Some(i_o)) = (handle_to_idx(attn_v), handle_to_idx(attn_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_v_b) else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || value_mla <= 0 || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (value_mla as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_v_q6_k.clone()
            .launch(cfg, (av, wv, ov, n_heads, kv_lora_rank, value_mla, blocks_per_row))
            .expect("launch mla_absorb_v_q6_k");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_q_iq4_nl_cuda(
    q_proj: i64, w_k_b: i64, q_out: i64,
    n_heads: c_int, key_mla: c_int, qk_rope: c_int,
    kv_lora_rank: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_q), Some(i_o)) = (handle_to_idx(q_proj), handle_to_idx(q_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_k_b) else { return -1; };
    if n_heads <= 0 || key_mla <= 0 || qk_rope <= 0 || kv_lora_rank <= 0
        || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: ((kv_lora_rank + qk_rope) as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let qv = &*q_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_q_iq4_nl.clone()
            .launch(cfg, (qv, wv, ov, n_heads, key_mla, qk_rope,
                          kv_lora_rank, blocks_per_row))
            .expect("launch mla_absorb_q_iq4_nl");
    }
    0
}

#[no_mangle] pub extern "C" fn aether_op_mla_absorb_v_iq4_nl_cuda(
    attn_v: i64, w_v_b: i64, attn_out: i64,
    n_heads: c_int, kv_lora_rank: c_int, value_mla: c_int, blocks_per_row: c_int,
) -> c_int {
    let (Some(i_a), Some(i_o)) = (handle_to_idx(attn_v), handle_to_idx(attn_out))
        else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_v_b) else { return -1; };
    if n_heads <= 0 || kv_lora_rank <= 0 || value_mla <= 0 || blocks_per_row <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let us = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let w_p = us[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let cfg = LaunchConfig {
        grid_dim: (value_mla as u32, n_heads as u32, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let ov = &mut *o_p; let wv = &*w_p;
        paged_ctx().mla_absorb_v_iq4_nl.clone()
            .launch(cfg, (av, wv, ov, n_heads, kv_lora_rank, value_mla, blocks_per_row))
            .expect("launch mla_absorb_v_iq4_nl");
    }
    0
}

/// FR-17-extra-bert-fwd — full bidirectional self-attention.  Encoder-only
/// models (BERT, BGE, etc.) process the entire sequence in one pass, every
/// query attending to every key with no causal mask.
///
/// Q / K / V are token-major: [seq, n_heads, head_dim].  Output has the same
/// shape.  `head_dim` must be a multiple of 32 and ≤ 256.  Shared mem sized
/// to `seq * 4` bytes (BERT-base seq=512 → 2 KiB, well within the per-block
/// 48 KiB limit on every supported GPU).
#[no_mangle] pub extern "C" fn aether_op_bert_self_attention_fwd_f32_cuda(
    q_dev: i64, k_dev: i64, v_dev: i64, attn_out: i64,
    seq: c_int, n_heads: c_int, head_dim: c_int, scale: f32,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_dev) else { return -1; };
    let Some(i_k) = handle_to_idx(k_dev) else { return -1; };
    let Some(i_v) = handle_to_idx(v_dev) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out) else { return -1; };
    if seq <= 0 || n_heads <= 0 || head_dim <= 0 { return -1; }
    // head_dim ≤ 256 caps q_local[8] × per_lane=8 lanes worth of storage.
    // No multiple-of-32 constraint — the kernel uses CEIL per_lane + a
    // per-element bounds check, same shape as the flex paged kernel.
    if head_dim > 256 { return -2; }
    let bs = unsafe { bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let k_p = bs[i_k].as_ref().unwrap() as *const CudaSlice<f32>;
    let v_p = bs[i_v].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let shmem = (seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_heads as u32, seq as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kv = &*k_p; let vv = &*v_p;
        let ov = &mut *o_p;
        ctx().bert_self_attention_fwd.clone()
            .launch(cfg, (qv, kv, vv, ov, seq, n_heads, head_dim, scale))
            .expect("launch bert_self_attention_fwd");
    }
    0
}

/// FR-17-extra-bert-fwd — sum word + position + token-type embeddings into
/// the input activation.  All three embedding tables are F32; output is
/// [seq, d_model].  Apply BERT's LayerNorm-with-bias afterward (use the
/// existing `aether_op_layer_norm_fwd_f32_cuda`).
#[no_mangle] pub extern "C" fn aether_op_bert_embed_sum_f32_cuda(
    input_ids_dev: i64, token_type_ids_dev: i64,
    word_embd: i64, pos_embd: i64, type_embd: i64,
    out: i64,
    seq: c_int, d_model: c_int,
) -> c_int {
    let Some(i_ii) = handle_to_i32_idx(input_ids_dev) else { return -1; };
    let Some(i_ti) = handle_to_i32_idx(token_type_ids_dev) else { return -1; };
    let Some(i_we) = handle_to_idx(word_embd) else { return -1; };
    let Some(i_pe) = handle_to_idx(pos_embd) else { return -1; };
    let Some(i_te) = handle_to_idx(type_embd) else { return -1; };
    let Some(i_o) = handle_to_idx(out) else { return -1; };
    if seq <= 0 || d_model <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let ii_p = ibs[i_ii].as_ref().unwrap() as *const CudaSlice<i32>;
    let ti_p = ibs[i_ti].as_ref().unwrap() as *const CudaSlice<i32>;
    let we_p = bs[i_we].as_ref().unwrap() as *const CudaSlice<f32>;
    let pe_p = bs[i_pe].as_ref().unwrap() as *const CudaSlice<f32>;
    let te_p = bs[i_te].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block = 128u32;
    let cfg = LaunchConfig {
        grid_dim:  (seq as u32, ((d_model as u32) + block - 1) / block, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let iiv = &*ii_p; let tiv = &*ti_p;
        let wev = &*we_p; let pev = &*pe_p; let tev = &*te_p;
        let ov = &mut *o_p;
        ctx().bert_embed_sum.clone()
            .launch(cfg, (iiv, tiv, wev, pev, tev, ov, seq, d_model))
            .expect("launch bert_embed_sum");
    }
    0
}

/// FR-19.5-extra-deep — batched paged attention: B queries × B page tables
/// in one launch.  All B requests at the same `cur_seq` (synchronous batched
/// step).  Each request indexes its own row of `page_table_batch` (row
/// stride = `page_table_stride` int32 entries).
#[no_mangle] pub extern "C" fn aether_op_batched_paged_attention_seqB_devarg_f32_cuda(
    q_batch: i64, k_pool: i64, v_pool: i64,
    page_table_batch_dev: i64, attn_out_batch: i64,
    batch: c_int,
    n_q_heads: c_int, n_kv_heads: c_int, head_dim: c_int, block_size: c_int,
    page_table_stride: c_int,
    scale: f32, max_seq: c_int, step_args_i32: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_batch) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_batch_dev) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out_batch) else { return -1; };
    let Some(is)  = handle_to_i32_idx(step_args_i32) else { return -1; };
    if batch <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0
        || max_seq <= 0 || block_size <= 0 || page_table_stride <= 0 { return -1; }
    if (n_q_heads % n_kv_heads) != 0 { return -2; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp_p = bs[i_kp].as_ref().unwrap() as *const CudaSlice<f32>;
    let vp_p = bs[i_vp].as_ref().unwrap() as *const CudaSlice<f32>;
    let pt_p = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let sp  = ibs[is].as_ref().unwrap() as *const CudaSlice<i32>;
    let shmem = (max_seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_q_heads as u32, batch as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kpv = &*kp_p; let vpv = &*vp_p; let ptv = &*pt_p;
        let ov = &mut *o_p; let sv = &*sp;
        paged_ctx().batched_paged_attention_seqB_devarg.clone()
            .launch(cfg, (qv, kpv, vpv, ptv, ov,
                          n_q_heads, n_kv_heads, head_dim, block_size,
                          page_table_stride, scale, sv))
            .expect("launch batched_paged_attention_seqB_devarg");
    }
    0
}

/// FR-19.5-extra-deep Phase 2 — HETEROGENEOUS-position batched append_kv.
/// Like the seqB variant but each request writes at its OWN position read
/// from `pos_batch_dev` (an i32 device buffer of length `batch`) instead
/// of a single shared step_args[0].  Lets the scheduler fuse N slots at
/// different decode positions into one launch.
#[no_mangle] pub extern "C" fn aether_op_batched_paged_append_kv_hetero_devarg_f32_cuda(
    k_new_batch: i64, v_new_batch: i64,
    k_pool: i64, v_pool: i64,
    page_table_batch_dev: i64,
    batch: c_int, d_kv: c_int, block_size: c_int, page_table_stride: c_int,
    pos_batch_dev: i64,
) -> c_int {
    let Some(i_kn) = handle_to_idx(k_new_batch) else { return -1; };
    let Some(i_vn) = handle_to_idx(v_new_batch) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_batch_dev) else { return -1; };
    let Some(i_pos) = handle_to_i32_idx(pos_batch_dev) else { return -1; };
    if batch <= 0 || d_kv <= 0 || block_size <= 0 || page_table_stride <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let kn = bs[i_kn].as_ref().unwrap() as *const CudaSlice<f32>;
    let vn = bs[i_vn].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp = bs[i_kp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let vp = bs[i_vp].as_mut().unwrap() as *mut CudaSlice<f32>;
    let pt = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let pos = ibs[i_pos].as_ref().unwrap() as *const CudaSlice<i32>;
    let threads_per_block: u32 = 256;
    let blocks_per_req: u32 = ((d_kv as u32) + threads_per_block - 1) / threads_per_block;
    let cfg = LaunchConfig {
        grid_dim: (blocks_per_req, batch as u32, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let knr = &*kn; let vnr = &*vn;
        let kpm = &mut *kp; let vpm = &mut *vp;
        let ptv = &*pt; let posv = &*pos;
        paged_ctx().batched_paged_append_kv_hetero_devarg.clone()
            .launch(cfg, (knr, vnr, kpm, vpm, ptv,
                          d_kv, block_size, page_table_stride, posv))
            .expect("launch batched_paged_append_kv_hetero_devarg");
    }
    0
}

/// FR-19.5-extra-deep Phase 2 — HETEROGENEOUS-position batched attention.
/// Like the seqB variant but each request attends over its OWN window
/// [0, cur_seq_batch[req]) read from `cur_seq_batch_dev` (an i32 device
/// buffer of length `batch`) instead of a single shared step_args[1].
/// Shared scores[] is launch-sized for `max_seq` (the upper bound on any
/// request's cur_seq); each block uses only its own prefix.
#[no_mangle] pub extern "C" fn aether_op_batched_paged_attention_hetero_devarg_f32_cuda(
    q_batch: i64, k_pool: i64, v_pool: i64,
    page_table_batch_dev: i64, attn_out_batch: i64,
    batch: c_int,
    n_q_heads: c_int, n_kv_heads: c_int, head_dim: c_int, block_size: c_int,
    page_table_stride: c_int,
    scale: f32, max_seq: c_int, cur_seq_batch_dev: i64,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_batch) else { return -1; };
    let Some(i_kp) = handle_to_idx(k_pool) else { return -1; };
    let Some(i_vp) = handle_to_idx(v_pool) else { return -1; };
    let Some(i_pt) = handle_to_i32_idx(page_table_batch_dev) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out_batch) else { return -1; };
    let Some(i_cs) = handle_to_i32_idx(cur_seq_batch_dev) else { return -1; };
    if batch <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0
        || max_seq <= 0 || block_size <= 0 || page_table_stride <= 0 { return -1; }
    if (n_q_heads % n_kv_heads) != 0 { return -2; }
    let bs = unsafe { bufs() };
    let ibs = unsafe { i32_bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kp_p = bs[i_kp].as_ref().unwrap() as *const CudaSlice<f32>;
    let vp_p = bs[i_vp].as_ref().unwrap() as *const CudaSlice<f32>;
    let pt_p = ibs[i_pt].as_ref().unwrap() as *const CudaSlice<i32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cs  = ibs[i_cs].as_ref().unwrap() as *const CudaSlice<i32>;
    let shmem = (max_seq as u32) * 4;
    let cfg = LaunchConfig {
        grid_dim:  (n_q_heads as u32, batch as u32, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kpv = &*kp_p; let vpv = &*vp_p; let ptv = &*pt_p;
        let ov = &mut *o_p; let csv = &*cs;
        paged_ctx().batched_paged_attention_hetero_devarg.clone()
            .launch(cfg, (qv, kpv, vpv, ptv, ov,
                          n_q_heads, n_kv_heads, head_dim, block_size,
                          page_table_stride, scale, csv))
            .expect("launch batched_paged_attention_hetero_devarg");
    }
    0
}

/// FR-17.13-extra (GQA) — broadcast K/V from `n_kv_heads` to `n_q_heads`.
#[no_mangle] pub extern "C" fn aether_op_gqa_repeat_kv_f32_cuda(
    kv_in: i64, kv_out: i64,
    seq: c_int, n_kv_heads: c_int, head_dim: c_int, n_q_heads: c_int,
) -> c_int {
    let (Some(ii), Some(io)) = (handle_to_idx(kv_in), handle_to_idx(kv_out))
        else { return -1; };
    if (n_q_heads % n_kv_heads) != 0 { return 1; }
    let bs = unsafe { bufs() };
    let i_p = bs[ii].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[io].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (seq * n_q_heads * head_dim) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let iv = &*i_p; let ov = &mut *o_p;
        ctx().gqa_repeat_kv.clone()
            .launch(cfg, (iv, ov, seq, n_kv_heads, head_dim, n_q_heads))
            .expect("launch gqa_repeat_kv");
    }
    0
}

/// matt-voice — SiLU forward in place: x[i] = x[i] / (1 + exp(-x[i])).
#[no_mangle] pub extern "C" fn aether_op_silu_f32_cuda(x: i64, n: c_int) -> c_int {
    let Some(ix) = handle_to_idx(x) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let xv = &mut *x_p;
        ctx().silu_inplace.clone().launch(cfg, (xv, n)).expect("launch silu_inplace");
    }
    0
}

/// matt-voice — element-wise multiply in place: x[i] *= y[i]. Used for
/// SwiGLU's `silu(gate) * up` step.
#[no_mangle] pub extern "C" fn aether_op_mul_inplace_f32_cuda(
    x: i64, y: i64, n: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let y_p = bs[iy].as_ref().unwrap() as *const CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let xv = &mut *x_p; let yv = &*y_p;
        ctx().mul_inplace.clone().launch(cfg, (xv, yv, n)).expect("launch mul_inplace");
    }
    0
}

/// matt-voice — residual in place: x[i] += y[i].
#[no_mangle] pub extern "C" fn aether_op_add_inplace_f32_cuda(
    x: i64, y: i64, n: c_int,
) -> c_int {
    let (Some(ix), Some(iy)) = (handle_to_idx(x), handle_to_idx(y)) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let y_p = bs[iy].as_ref().unwrap() as *const CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    unsafe {
        let xv = &mut *x_p; let yv = &*y_p;
        ctx().add_inplace.clone().launch(cfg, (xv, yv, n)).expect("launch add_inplace");
    }
    0
}

/// matt-voice — broadcast-add a bias vector along the last dim:
/// x[r, c] += bias[c].
#[no_mangle] pub extern "C" fn aether_op_bias_add_f32_cuda(
    x: i64, bias: i64, rows: c_int, cols: c_int,
) -> c_int {
    let (Some(ix), Some(ib)) = (handle_to_idx(x), handle_to_idx(bias)) else { return -1; };
    let bs = unsafe { bufs() };
    let x_p = bs[ix].as_mut().unwrap() as *mut CudaSlice<f32>;
    let b_p = bs[ib].as_ref().unwrap() as *const CudaSlice<f32>;
    let total = (rows * cols) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let xv = &mut *x_p; let bv = &*b_p;
        ctx().bias_add.clone().launch(cfg, (xv, bv, rows, cols)).expect("launch bias_add");
    }
    0
}

/// FR-17.14-extra-deepest — dequant `n_blocks` Q4_K_M super-blocks
/// on device. `blocks_u8` is a u8 device handle pointing to
/// `n_blocks * 144` raw bytes; `out_f32` is an f32 device handle of
/// length `n_blocks * 256`.
///
/// Threading: 256 threads per block; n_blocks total CTAs. Each thread
/// produces ONE dequantised f32. Mirrors the CPU
/// `aether_dequant_q4_k_m` exactly byte-for-byte (verified by the
/// parity test).
#[no_mangle] pub extern "C" fn aether_op_dequant_q4_k_m_f32_cuda(
    blocks_u8: i64, out_f32: i64, n_blocks: c_int,
) -> c_int {
    let Some(i_blk) = handle_to_u8_idx(blocks_u8) else { return -1; };
    let Some(i_out) = handle_to_idx(out_f32) else { return -1; };
    if n_blocks <= 0 { return -1; }
    let bs_u8 = unsafe { u8_bufs() };
    let bs_f32 = unsafe { bufs() };
    let b_p = bs_u8[i_blk].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs_f32[i_out].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (n_blocks * 256) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let bv = &*b_p; let ov = &mut *o_p;
        ctx().dequant_q4_k_m_gpu.clone()
            .launch(cfg, (bv, ov, n_blocks))
            .expect("launch dequant_q4_k_m");
    }
    0
}

/// FR-17.14-extra-deepest -- FUSED Q4_K matmul for seq=1.
///
/// out[n] = a[k] @ dequant(w_q4k)[n, k]
///
/// Args:
///   a_dev_f32 : f32 device handle, length k = n_blocks * 256
///   w_dev_u8  : u8 device handle, length n * n_blocks * 144 (GGUF
///               natural order: each row is one output column's
///               worth of n_blocks super-blocks)
///   out_dev_f32: f32 device handle, length n
///   n, n_blocks: matmul dims (k = n_blocks * 256)
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    // CTA size = 32 (BLOCK_N matches the kernel constant). Grid covers
    // n output columns in chunks of 32.
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,  // a_tile is __shared__ static
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q4k_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q4k_matmul_seq1");
    }
    0
}

/// FR-17-extra-q4_0-fwd — Fused Q4_0 matmul for seq=1.  Same shape as the
/// Q4_K variant but with smaller blocks (32 elems / 18 bytes) and a simpler
/// dequant (single scale per block, no per-sub-block min/scale tables).
///
/// `n_blocks` here counts 32-element blocks (k = n_blocks * 32), NOT
/// super-blocks like Q4_K's 256-element blocks.  Weight buffer is
/// `n * n_blocks * 18` bytes.
#[no_mangle] pub extern "C" fn aether_op_fused_q4_0_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q4_0_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q4_0_matmul_seq1");
    }
    0
}

/// FR-17-extra-q5_0-fwd — Fused Q5_0 matmul (22-byte 32-elem blocks).
/// `n_blocks` counts 32-element blocks (k = n_blocks * 32).
#[no_mangle] pub extern "C" fn aether_op_fused_q5_0_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q5_0_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q5_0_matmul_seq1");
    }
    0
}

/// FR-17-extra-q8_0-fwd — Fused Q8_0 matmul (34-byte 32-elem blocks).
#[no_mangle] pub extern "C" fn aether_op_fused_q8_0_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q8_0_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q8_0_matmul_seq1");
    }
    0
}

/// FR-17-extra-q5_k-fwd — Fused Q5_K matmul (176-byte 256-elem super-blocks).
/// Same shape as the Q4_K kernel (same n_blocks = n_in/256, same scale
/// layout) but with 5-bit quants per element (qs nibble + qh high-bit).
#[no_mangle] pub extern "C" fn aether_op_fused_q5_k_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q5_k_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q5_k_matmul_seq1");
    }
    0
}

/// FR-17-extra-q3_k-fwd — Fused Q3_K matmul (110-byte 256-elem blocks).
/// Mirror of `aether_op_fused_q4k_matmul_seq1_cuda` with 3-bit unpacking
/// and the kmask1/kmask2 scale-decode trick (see kernel for layout).
/// Unblocks Qwen3-MoE Q3_K_M end-to-end (198 Q3_K tensors).
#[no_mangle] pub extern "C" fn aether_op_fused_q3_k_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q3_k_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q3_k_matmul_seq1");
    }
    0
}

/// FR-17-extra-iq4_nl-fwd — Fused IQ4_NL matmul (18-byte 32-elem blocks).
/// Same byte layout as Q4_0 but with 16-entry non-linear codebook lookup
/// instead of `(q - 8)`.
#[no_mangle] pub extern "C" fn aether_op_fused_iq4_nl_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_iq4_nl_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_iq4_nl_matmul_seq1");
    }
    0
}

/// FR-17-extra-iq4_xs-fwd — Fused IQ4_XS matmul (136-byte 256-elem blocks).
/// Per-sub-block 6-bit signed scale (4 nibble bits from scales_l + 2 high
/// bits from scales_h) × kvalues_iq4nl codebook lookup per elem.
#[no_mangle] pub extern "C" fn aether_op_fused_iq4_xs_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_iq4_xs_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_iq4_xs_matmul_seq1");
    }
    0
}

/// FR-17-extra-iq3_xxs-fwd — Fused IQ3_XXS matmul (98-byte 256-elem blocks).
/// `n_blocks` counts 256-elem blocks (k = n_blocks * 256).
#[no_mangle] pub extern "C" fn aether_op_fused_iq3_xxs_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_iq3_xxs_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_iq3_xxs_matmul_seq1");
    }
    0
}

/// FR-17-extra-iq3_s-fwd — Fused IQ3_S matmul (110-byte 256-elem blocks).
/// Per-sub-block odd-integer scale (db = d * (1 + 2 * scale_nib)) ×
/// 512-entry codebook lookup with direct 8-bit sign patterns.
#[no_mangle] pub extern "C" fn aether_op_fused_iq3_s_matmul_seq1_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let block_n = 32u32;
    let grid_x = ((n as u32) + block_n - 1) / block_n;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (block_n, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_iq3_s_matmul_seq1.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_iq3_s_matmul_seq1");
    }
    0
}

/// FR-17.14-extra-deepest v2 -- split-K fused Q4_K matmul for seq=1.
///
/// Same interface as `aether_op_fused_q4k_matmul_seq1_cuda` but with
/// 32 threads per output (warp-per-output split-K). Closes the
/// small-N under-utilization gap of v1.
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_matmul_seq1_v2_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    // 256 threads per CTA (8 warps * 32 threads). 8 outputs per CTA.
    let cta_threads = 256u32;
    let outputs_per_cta = 8u32;
    let grid_x = ((n as u32) + outputs_per_cta - 1) / outputs_per_cta;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (cta_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q4k_matmul_seq1_v2.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q4k_matmul_seq1_v2");
    }
    0
}

/// FR-17.14-extra-deepest-v3 -- fused FFN: gate matmul + up matmul + silu + mul.
///
/// Replaces 4 kernel launches (gate, up, silu, mul_inplace) with 1.
/// Reads x_norm into shmem once and uses it for both halves of the FMA.
/// Each warp produces silu(gate[ni]) * up[ni] for its assigned output.
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
    a_dev_f32: i64, w_gate_dev_u8: i64, w_up_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_wg) = handle_to_u8_idx(w_gate_dev_u8) else { return -1; };
    let Some(i_wu) = handle_to_u8_idx(w_up_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p  = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let wg_p = bs_u8[i_wg].as_ref().unwrap() as *const CudaSlice<u8>;
    let wu_p = bs_u8[i_wu].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p  = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cta_threads = 256u32;
    let outputs_per_cta = 8u32;
    let grid_x = ((n as u32) + outputs_per_cta - 1) / outputs_per_cta;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (cta_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wgv = &*wg_p; let wuv = &*wu_p; let ov = &mut *o_p;
        ctx().fused_q4k_ffn_gate_up_silu_mul.clone()
            .launch(cfg, (av, wgv, wuv, ov, n, n_blocks))
            .expect("launch fused_q4k_ffn_gate_up_silu_mul");
    }
    0
}

/// FR-17.14-extra-deepest-v3 -- byte-once Q4_K matmul.
///
/// Same interface as v2 but reads each qs byte once per warp (uses both
/// low and high nibbles within one lane) -> half the memory instructions.
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_matmul_seq1_v3_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cta_threads = 256u32;
    let outputs_per_cta = 8u32;
    let grid_x = ((n as u32) + outputs_per_cta - 1) / outputs_per_cta;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (cta_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q4k_matmul_seq1_v3.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q4k_matmul_seq1_v3");
    }
    0
}

/// FR-19.5-extra-deep Phase 2 — weight-reuse batched Q4_K matmul.
/// Computes `batch` independent output rows (a:[batch*n_blocks*256],
/// out:[batch*n]) against shared Q4_K weights, dequantizing each weight
/// block once and reusing it across all rows.  `batch` must be in [1, 8]
/// (the kernel holds batch activation tiles + 2·batch accumulators per
/// lane).  FMA order per (row, output) is bit-identical to seq1_v3.
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_matmul_seqB_v3_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int, batch: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 || batch <= 0 || batch > 8 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cta_threads = 256u32;
    let outputs_per_cta = 8u32;
    let grid_x = ((n as u32) + outputs_per_cta - 1) / outputs_per_cta;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (cta_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        paged_ctx().fused_q4k_matmul_seqB_v3.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks, batch))
            .expect("launch fused_q4k_matmul_seqB_v3");
    }
    0
}

/// FR-17.14-extra-deepest-v3 -- fused FFN with byte-once matmul.
#[no_mangle] pub extern "C" fn aether_op_fused_q4k_ffn_gate_up_silu_mul_v2_cuda(
    a_dev_f32: i64, w_gate_dev_u8: i64, w_up_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_wg) = handle_to_u8_idx(w_gate_dev_u8) else { return -1; };
    let Some(i_wu) = handle_to_u8_idx(w_up_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p  = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let wg_p = bs_u8[i_wg].as_ref().unwrap() as *const CudaSlice<u8>;
    let wu_p = bs_u8[i_wu].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p  = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cta_threads = 256u32;
    let outputs_per_cta = 8u32;
    let grid_x = ((n as u32) + outputs_per_cta - 1) / outputs_per_cta;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (cta_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wgv = &*wg_p; let wuv = &*wu_p; let ov = &mut *o_p;
        ctx().fused_q4k_ffn_gate_up_silu_mul_v2.clone()
            .launch(cfg, (av, wgv, wuv, ov, n, n_blocks))
            .expect("launch fused_q4k_ffn_gate_up_silu_mul_v2");
    }
    0
}

/// FR-17.13-extra — append new K/V step into the on-device KV cache
/// at position `pos`. `k_new_dev` / `v_new_dev` are f32 device handles
/// of length `d_kv`. `k_cache_dev` / `v_cache_dev` are f32 device
/// handles allocated for `max_seq * d_kv` floats.
#[no_mangle] pub extern "C" fn aether_op_append_kv_f32_cuda(
    k_new_dev: i64, v_new_dev: i64,
    k_cache_dev: i64, v_cache_dev: i64,
    pos: c_int, d_kv: c_int,
) -> c_int {
    let Some(i_kn) = handle_to_idx(k_new_dev) else { return -1; };
    let Some(i_vn) = handle_to_idx(v_new_dev) else { return -1; };
    let Some(i_kc) = handle_to_idx(k_cache_dev) else { return -1; };
    let Some(i_vc) = handle_to_idx(v_cache_dev) else { return -1; };
    if pos < 0 || d_kv <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let kn = bs[i_kn].as_ref().unwrap() as *const CudaSlice<f32>;
    let vn = bs[i_vn].as_ref().unwrap() as *const CudaSlice<f32>;
    let kc = bs[i_kc].as_mut().unwrap() as *mut CudaSlice<f32>;
    let vc = bs[i_vc].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cfg = LaunchConfig::for_num_elems(d_kv as u32);
    unsafe {
        let knr = &*kn; let vnr = &*vn;
        let kcm = &mut *kc; let vcm = &mut *vc;
        ctx().append_kv.clone()
            .launch(cfg, (knr, vnr, kcm, vcm, pos, d_kv))
            .expect("launch append_kv");
    }
    0
}

/// FR-17.13-extra — single-step causal attention with on-device KV cache.
///
/// Args:
///   q_dev      : f32 [n_q_heads * head_dim]
///   k_cache    : f32 [max_seq, n_kv_heads * head_dim]
///   v_cache    : f32 [max_seq, n_kv_heads * head_dim]
///   attn_out   : f32 [n_q_heads * head_dim]
///   cur_seq    : current valid length in cache (incl. just-appended)
///   n_q_heads, n_kv_heads, head_dim: GQA / shape config
///   scale      : 1/sqrt(head_dim) typically
///
/// Launches one warp per Q head. Shared mem sized for cur_seq * 4 bytes.
#[no_mangle] pub extern "C" fn aether_op_attention_seq1_f32_cuda(
    q_dev: i64, k_cache: i64, v_cache: i64, attn_out: i64,
    cur_seq: c_int,
    n_q_heads: c_int, n_kv_heads: c_int, head_dim: c_int,
    scale: f32,
) -> c_int {
    let Some(i_q) = handle_to_idx(q_dev) else { return -1; };
    let Some(i_kc) = handle_to_idx(k_cache) else { return -1; };
    let Some(i_vc) = handle_to_idx(v_cache) else { return -1; };
    let Some(i_o) = handle_to_idx(attn_out) else { return -1; };
    if cur_seq <= 0 || n_q_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 { return -1; }
    if (n_q_heads % n_kv_heads) != 0 { return -2; }
    let bs = unsafe { bufs() };
    let q_p = bs[i_q].as_ref().unwrap() as *const CudaSlice<f32>;
    let kc_p = bs[i_kc].as_ref().unwrap() as *const CudaSlice<f32>;
    let vc_p = bs[i_vc].as_ref().unwrap() as *const CudaSlice<f32>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let shmem = (cur_seq as u32) * 4;  // bytes for scores[cur_seq]
    let cfg = LaunchConfig {
        grid_dim:  (n_q_heads as u32, 1, 1),
        block_dim: (32, 1, 1),  // one warp per head
        shared_mem_bytes: shmem,
    };
    unsafe {
        let qv = &*q_p; let kcv = &*kc_p; let vcv = &*vc_p; let ov = &mut *o_p;
        ctx().attention_seq1.clone()
            .launch(cfg, (qv, kcv, vcv, ov, cur_seq, n_q_heads, n_kv_heads, head_dim, scale))
            .expect("launch attention_seq1");
    }
    0
}

/// FR-17.14-extra-deepest v2 -- fused Q6_K matmul for seq=1.
/// Same interface as the Q4_K v2 fused matmul, reads Q6_K bytes
/// directly. Used for V proj + ffn_down + lm_head in Qwen2.5.
#[no_mangle] pub extern "C" fn aether_op_fused_q6k_matmul_seq1_v2_cuda(
    a_dev_f32: i64, w_dev_u8: i64, out_dev_f32: i64,
    n: c_int, n_blocks: c_int,
) -> c_int {
    let Some(i_a) = handle_to_idx(a_dev_f32) else { return -1; };
    let Some(i_w) = handle_to_u8_idx(w_dev_u8) else { return -1; };
    let Some(i_o) = handle_to_idx(out_dev_f32) else { return -1; };
    if n <= 0 || n_blocks <= 0 { return -1; }
    let bs = unsafe { bufs() };
    let bs_u8 = unsafe { u8_bufs() };
    let a_p = bs[i_a].as_ref().unwrap() as *const CudaSlice<f32>;
    let w_p = bs_u8[i_w].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs[i_o].as_mut().unwrap() as *mut CudaSlice<f32>;
    let cta_threads = 256u32;
    let outputs_per_cta = 8u32;
    let grid_x = ((n as u32) + outputs_per_cta - 1) / outputs_per_cta;
    let cfg = LaunchConfig {
        grid_dim:  (grid_x, 1, 1),
        block_dim: (cta_threads, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        let av = &*a_p; let wv = &*w_p; let ov = &mut *o_p;
        ctx().fused_q6k_matmul_seq1_v2.clone()
            .launch(cfg, (av, wv, ov, n, n_blocks))
            .expect("launch fused_q6k_matmul_seq1_v2");
    }
    0
}

/// FR-17.14-extra-deepest (Q6_K) — dequant `n_blocks` Q6_K super-blocks
/// on device. `blocks_u8` is `n_blocks * 210` raw bytes; `out_f32` is
/// `n_blocks * 256` f32. Mirrors the CPU `aether_dequant_q6_k` and
/// matches it byte-for-byte (verified by the parity test).
#[no_mangle] pub extern "C" fn aether_op_dequant_q6_k_f32_cuda(
    blocks_u8: i64, out_f32: i64, n_blocks: c_int,
) -> c_int {
    let Some(i_blk) = handle_to_u8_idx(blocks_u8) else { return -1; };
    let Some(i_out) = handle_to_idx(out_f32) else { return -1; };
    if n_blocks <= 0 { return -1; }
    let bs_u8 = unsafe { u8_bufs() };
    let bs_f32 = unsafe { bufs() };
    let b_p = bs_u8[i_blk].as_ref().unwrap() as *const CudaSlice<u8>;
    let o_p = bs_f32[i_out].as_mut().unwrap() as *mut CudaSlice<f32>;
    let total = (n_blocks * 256) as u32;
    let cfg = LaunchConfig::for_num_elems(total);
    unsafe {
        let bv = &*b_p; let ov = &mut *o_p;
        ctx().dequant_q6_k_gpu.clone()
            .launch(cfg, (bv, ov, n_blocks))
            .expect("launch dequant_q6_k");
    }
    0
}

