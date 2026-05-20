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
}

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

fn ctx() -> &'static CudaCtx {
    CTX.get_or_init(|| {
        let device = CudaDevice::new(0).expect("CudaDevice::new(0)");
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
              "silu_inplace", "mul_inplace", "add_inplace", "bias_add"])
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
        CudaCtx { device, blas, cross_entropy_fwd, cross_entropy_bwd, adamw_step,
                  add_f32, gelu_fwd, gelu_bwd,
                  layer_norm_fwd, layer_norm_bwd_dx, layer_norm_bwd_params,
                  softmax_f32, softmax_bwd, softmax_bwd_scaled, scale_f32, gelu_inplace,
                  add_layer_norm_fwd,
                  rms_norm_fwd, rope_apply, gqa_repeat_kv,
                  silu_inplace, mul_inplace, add_inplace, bias_add }
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

