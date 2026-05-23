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
}

static PAGED_CTX: OnceLock<PagedCtx> = OnceLock::new();

fn paged_ctx() -> &'static PagedCtx {
    PAGED_CTX.get_or_init(|| {
        let device = &ctx().device;
        let paged_ptx = compile_ptx(PAGED_KERNEL_SRC).expect("compile_ptx paged");
        device.load_ptx(paged_ptx, "aether_paged_kernels",
            &["paged_append_kv_devarg", "paged_attention_seq1_devarg",
              "batched_paged_attention_seqB_devarg",
              "batched_paged_append_kv_seqB_devarg"])
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
              "fused_q4k_matmul_seq1_v2", "fused_q6k_matmul_seq1_v2",
              "fused_q4k_ffn_gate_up_silu_mul",
              "fused_q4k_matmul_seq1_v3", "fused_q4k_ffn_gate_up_silu_mul_v2",
              "rope_apply_devarg", "append_kv_devarg", "attention_seq1_devarg",
              "append_kv", "attention_seq1",
              "fused_f16_matmul_seq1"])
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

        CudaCtx { device, blas, cross_entropy_fwd, cross_entropy_bwd, adamw_step,
                  add_f32, gelu_fwd, gelu_bwd,
                  layer_norm_fwd, layer_norm_bwd_dx, layer_norm_bwd_params,
                  softmax_f32, softmax_bwd, softmax_bwd_scaled, scale_f32, gelu_inplace,
                  add_layer_norm_fwd,
                  rms_norm_fwd, rope_apply, gqa_repeat_kv,
                  silu_inplace, mul_inplace, add_inplace, bias_add,
                  dequant_q4_k_m_gpu, dequant_q6_k_gpu,
                  fused_q4k_matmul_seq1, fused_q4k_matmul_seq1_v2,
                  fused_q6k_matmul_seq1_v2, fused_q4k_ffn_gate_up_silu_mul,
                  fused_q4k_matmul_seq1_v3, fused_q4k_ffn_gate_up_silu_mul_v2,
                  rope_apply_devarg, append_kv_devarg, attention_seq1_devarg,
                  append_kv, attention_seq1, fused_f16_matmul_seq1 }
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

