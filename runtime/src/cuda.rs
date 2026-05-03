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
unsafe fn bufs() -> &'static mut Vec<Option<CudaSlice<f32>>> { &mut *BUFFERS.0.get() }

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
            &["cross_entropy_fwd", "cross_entropy_bwd", "adamw_step"])
            .expect("load_ptx");
        let cross_entropy_fwd = device.get_func("aether_kernels", "cross_entropy_fwd").unwrap();
        let cross_entropy_bwd = device.get_func("aether_kernels", "cross_entropy_bwd").unwrap();
        let adamw_step        = device.get_func("aether_kernels", "adamw_step").unwrap();
        CudaCtx { device, blas, cross_entropy_fwd, cross_entropy_bwd, adamw_step }
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

