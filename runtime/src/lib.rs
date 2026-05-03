//! libaether_rt — runtime intrinsics that aether-emitted code links against.
//!
//! Phase 0/1 stubs. NCCL/MPI/RDMA backends slot in here in Phase 2 by replacing
//! the body of `aether_dist_all_reduce` with a dispatch on AETHER_DIST_BACKEND.
//!
//! All entry points are `#[no_mangle] extern "C"` so the LLVM IR emitted by
//! `compiler/src/codegen/llvm/mod.rs` links cleanly without aliasing.

use std::cell::UnsafeCell;
use std::os::raw::{c_int, c_void};

pub mod ops;

#[cfg(feature = "cuda")]
pub mod cuda;

// Single-threaded tape for the bootstrap runtime. Originally `thread_local!`
// — the TLS init hook for that drags Rust's `std::thread` machinery into the
// cdylib's DllMain, which in turn calls `bcryptprimitives.dll!ProcessPrng`
// for the HashMap hasher seed. When our self-hosted PE writer (#24) loads
// `aether_rt.dll` early in process init, bcryptprimitives' DllMain hasn't
// run yet, and the call AVs at offset 0x16e99 (a `movaps` to a slot whose
// alignment depends on prior init). A plain `static UnsafeCell` doesn't go
// through the TLS init path. Aether-emitted programs are single-threaded;
// the TLS layer was never load-bearing.
struct Tape {
    entries: Vec<*const c_void>,
    closed: bool,
}

struct TapeCell(UnsafeCell<Tape>);
unsafe impl Sync for TapeCell {}

static TAPE: TapeCell = TapeCell(UnsafeCell::new(Tape { entries: Vec::new(), closed: false }));

#[inline]
unsafe fn tape() -> &'static mut Tape { &mut *TAPE.0.get() }

#[no_mangle]
pub unsafe extern "C" fn aether_autodiff_init(_tape: *mut c_void) {
    let t = tape();
    t.entries.clear();
    t.closed = false;
}

#[no_mangle]
pub unsafe extern "C" fn aether_autodiff_push(_tape: *mut c_void, value: *const c_void) {
    tape().entries.push(value);
}

#[no_mangle]
pub extern "C" fn aether_autodiff_accumulate(_tape: *mut c_void, _grad: *const c_void) {
    // Phase 0: per-node grad accumulation is recorded but not summed —
    // the typed AD graph in Phase 1 supplies the operands for real partials.
}

/// Symbolic partial dispatch. Op codes are stable and shared with the
/// emitter (`compiler/src/codegen/llvm/mod.rs::PART_*`). Phase 0 records
/// the call; Phase 1 reads `dst` / `src` from a real value table and
/// computes the actual gradient.
#[no_mangle]
pub extern "C" fn aether_autodiff_partial(
    _tape: *mut c_void,
    _dst_node: c_int,
    _op_code: c_int,
    _src_node: c_int,
) {}

#[no_mangle]
pub unsafe extern "C" fn aether_autodiff_reverse(_tape: *mut c_void) {
    tape().closed = true;
}

// =====================================================================
// Primitive op surface — every `extern fn` declared in stdlib/ops.aether
// resolves to one of these symbols. Phase 0 bodies are no-ops that return
// 0; Phase 1 routes them to cuBLAS / cuDNN / NCCL on the 3070 Ti.
//
// The C ABI here is the contract between aetherc-emitted LLVM IR and
// whatever backend the runtime is built against. Never reorder an arg —
// LLVM IR call sites are positional.
// =====================================================================

// Real CPU implementations live in `ops.rs`. The C-ABI symbols below are
// thin wrappers that flatten void-pointers to the typed Rust impls. Phase 1
// swaps these bodies for cuBLAS/cuDNN; the symbol names never change.

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_f32(
    a: *const c_void, b: *const c_void, out: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_f32(a as _, b as _, out as _, m as _, k as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_backward_lhs_f32(
    dy: *const c_void, b: *const c_void, da: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_backward_lhs_f32(dy as _, b as _, da as _, m as _, k as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_backward_rhs_f32(
    a: *const c_void, dy: *const c_void, db: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_backward_rhs_f32(a as _, dy as _, db as _, m as _, k as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_bf16(
    _a: *const c_void, _b: *const c_void, _out: *mut c_void,
    _m: c_int, _k: c_int, _n: c_int,
) -> c_int { /* Phase 1 — bf16 path */ 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_add_f32(
    a: *const c_void, b: *const c_void, out: *mut c_void, n: c_int,
) -> c_int {
    ops::add_f32(a as _, b as _, out as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_add_inplace_f32(
    x: *mut c_void, y: *const c_void, n: c_int,
) -> c_int {
    ops::add_inplace_f32(x as _, y as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_add_bias_f32(
    x: *mut c_void, b: *const c_void, rows: c_int, cols: c_int,
) -> c_int {
    ops::add_bias_f32(x as _, b as _, rows as _, cols as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_scale_f32(
    x: *mut c_void, s: f32, n: c_int,
) -> c_int {
    ops::scale_f32(x as _, s, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_axpy_f32(
    alpha: f32, x: *const c_void, y: *mut c_void, n: c_int,
) -> c_int {
    ops::axpy_f32(alpha, x as _, y as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_gelu_f32(x: *mut c_void, n: c_int) -> c_int {
    ops::gelu_f32(x as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_gelu_backward_f32(
    x: *const c_void, dy: *const c_void, dx: *mut c_void, n: c_int,
) -> c_int {
    ops::gelu_backward_f32(x as _, dy as _, dx as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_silu_f32(x: *mut c_void, n: c_int) -> c_int {
    ops::silu_f32(x as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_silu_backward_f32(
    x: *const c_void, dy: *const c_void, dx: *mut c_void, n: c_int,
) -> c_int {
    ops::silu_backward_f32(x as _, dy as _, dx as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_relu_f32(x: *mut c_void, n: c_int) -> c_int {
    ops::relu_f32(x as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_relu_backward_f32(
    x: *const c_void, dy: *const c_void, dx: *mut c_void, n: c_int,
) -> c_int {
    ops::relu_backward_f32(x as _, dy as _, dx as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_softmax_f32(
    x: *mut c_void, rows: c_int, cols: c_int, _axis: c_int,
) -> c_int {
    ops::softmax_last_f32(x as _, rows as _, cols as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_softmax_backward_f32(
    y: *const c_void, dy: *const c_void, dx: *mut c_void,
    rows: c_int, cols: c_int,
) -> c_int {
    ops::softmax_backward_last_f32(y as _, dy as _, dx as _, rows as _, cols as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_layer_norm_f32(
    x: *const c_void, gamma: *const c_void, beta: *const c_void,
    eps: f32, out: *mut c_void, mean_out: *mut c_void, inv_std_out: *mut c_void,
    rows: c_int, d: c_int,
) -> c_int {
    ops::layer_norm_f32(x as _, gamma as _, beta as _, eps,
        out as _, mean_out as _, inv_std_out as _, rows as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_layer_norm_backward_f32(
    x: *const c_void, gamma: *const c_void, dy: *const c_void,
    mean: *const c_void, inv_std: *const c_void,
    dx: *mut c_void, dgamma: *mut c_void, dbeta: *mut c_void,
    rows: c_int, d: c_int,
) -> c_int {
    ops::layer_norm_backward_f32(x as _, gamma as _, dy as _,
        mean as _, inv_std as _, dx as _, dgamma as _, dbeta as _,
        rows as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sdpa_causal_f32(
    q: *const c_void, k: *const c_void, v: *const c_void,
    out: *mut c_void, attn_out: *mut c_void,
    bh: c_int, s_len: c_int, d: c_int,
) -> c_int {
    ops::sdpa_causal_f32(q as _, k as _, v as _, out as _, attn_out as _,
        bh as _, s_len as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sdpa_causal_backward_f32(
    q: *const c_void, k: *const c_void, v: *const c_void,
    attn: *const c_void, dout: *const c_void,
    dq: *mut c_void, dk: *mut c_void, dv: *mut c_void,
    bh: c_int, s_len: c_int, d: c_int,
) -> c_int {
    ops::sdpa_causal_backward_f32(q as _, k as _, v as _, attn as _, dout as _,
        dq as _, dk as _, dv as _, bh as _, s_len as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_cross_entropy_f32(
    logits: *const c_void, labels: *const c_void,
    probs_out: *mut c_void, b: c_int, v: c_int,
) -> f32 {
    ops::cross_entropy_f32(logits as _, labels as _, probs_out as _, b as _, v as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_cross_entropy_backward_f32(
    probs: *const c_void, labels: *const c_void,
    dlogits: *mut c_void, b: c_int, v: c_int,
) -> c_int {
    ops::cross_entropy_backward_f32(probs as _, labels as _, dlogits as _, b as _, v as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_embedding_lookup_f32(
    w: *const c_void, ids: *const c_void, out: *mut c_void,
    b: c_int, s_len: c_int, v_size: c_int, d: c_int,
) -> c_int {
    ops::embedding_lookup_f32(w as _, ids as _, out as _,
        b as _, s_len as _, v_size as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_embedding_backward_f32(
    ids: *const c_void, dy: *const c_void, dw: *mut c_void,
    b: c_int, s_len: c_int, v_size: c_int, d: c_int,
) -> c_int {
    ops::embedding_backward_f32(ids as _, dy as _, dw as _,
        b as _, s_len as _, v_size as _, d as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_zero_grad_f32(g: *mut c_void, n: c_int) -> c_int {
    ops::zero_grad_f32(g as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_clip_grad_norm_f32(
    g: *mut c_void, max_norm: f32, n: c_int,
) -> f32 {
    ops::clip_grad_norm_f32(g as _, max_norm, n as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_all_reduce_sum_f32(
    _buf: *mut c_void, _world_size: c_int, _n: c_int,
) -> c_int { /* Phase 2 — NCCL */ 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_adamw_step_f32(
    param: *mut c_void, grad: *const c_void,
    m: *mut c_void, v: *mut c_void,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    step: i64, n: c_int,
) -> c_int {
    ops::adamw_step_f32(param as _, grad as _, m as _, v as _,
        lr, beta1, beta2, eps, wd, step, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_sgd_step_f32(
    _param: *mut c_void, _grad: *const c_void,
    _lr: f32, _momentum: f32, _wd: f32, _state: *mut c_void, _n: c_int,
) -> c_int { 0 }

#[derive(Clone, Copy)]
#[repr(C)]
pub enum DistBackend {
    Nccl = 0,
    Mpi = 1,
    Gloo = 2,
    Stub = 99,
}

#[no_mangle]
pub extern "C" fn aether_dist_all_reduce(
    _buf: *mut c_void,
    _world_size: c_int,
    _backend: c_int,
) {
    // Phase 0 stub: real NCCL/MPI/Gloo dispatch goes here in Phase 2.
}

/// Compiled aether binaries can call this from `main()` to confirm the runtime
/// linked correctly without dragging in any platform deps.
#[no_mangle]
pub unsafe extern "C" fn aether_rt_self_check() -> c_int {
    tape().entries.len() as c_int
}

// =====================================================================
// Bootstrap allocator + init helpers — these let an Aether `main()` set up
// tensors, fill them with deterministic data, and free them, without needing
// arrays or struct support in the language. All buffers are heap-allocated
// f32/i32 slabs returned as i64-sized pointers (the asm backend already has
// a TyKind::Int that's 64-bit, which is exactly what a pointer is on x64).
//
// Phase-1 swap: replace these with cudaMalloc-backed versions; the symbol
// surface stays identical.
// =====================================================================

/// Allocate `n` f32 elements, zero-initialized. Returns a thin data pointer
/// cast to i64. Free with `aether_free_f32(p, n)`.
#[no_mangle] pub extern "C" fn aether_alloc_f32(n: c_int) -> i64 {
    let n = n.max(0) as usize;
    let mut v: Vec<f32> = vec![0.0; n];
    v.shrink_to_fit();
    debug_assert_eq!(v.capacity(), n);
    let ptr = v.as_mut_ptr() as i64;
    std::mem::forget(v);
    ptr
}

#[no_mangle] pub extern "C" fn aether_alloc_i32(n: c_int) -> i64 {
    let n = n.max(0) as usize;
    let mut v: Vec<i32> = vec![0; n];
    v.shrink_to_fit();
    debug_assert_eq!(v.capacity(), n);
    let ptr = v.as_mut_ptr() as i64;
    std::mem::forget(v);
    ptr
}

#[no_mangle] pub unsafe extern "C" fn aether_free_f32(p: i64, n: c_int) -> c_int {
    if p == 0 || n <= 0 { return 0; }
    let n = n as usize;
    let _ = Vec::from_raw_parts(p as *mut f32, n, n);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_free_i32(p: i64, n: c_int) -> c_int {
    if p == 0 || n <= 0 { return 0; }
    let n = n as usize;
    let _ = Vec::from_raw_parts(p as *mut i32, n, n);
    0
}

/// Splittable-mix64 (variant of SplitMix64). Deterministic PRNG that doesn't
/// require Rust's `rand` crate. Returns a u64; consumers can mask down.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Fill `n` f32s with N(~0, scale^2) using a Box-Muller transform off
/// SplitMix64. Deterministic in `seed`.
#[no_mangle] pub unsafe extern "C" fn aether_init_normal_f32(p: i64, n: c_int, scale: f32, seed: i64) {
    if p == 0 || n <= 0 { return; }
    let n = n as usize;
    let buf = std::slice::from_raw_parts_mut(p as *mut f32, n);
    let mut state = seed as u64;
    let mut i = 0;
    while i < n {
        let u1 = ((splitmix64(&mut state) >> 11) as f64) / ((1u64 << 53) as f64);
        let u2 = ((splitmix64(&mut state) >> 11) as f64) / ((1u64 << 53) as f64);
        let r = (-2.0 * u1.max(1e-30).ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let z0 = (r * theta.cos()) as f32 * scale;
        let z1 = (r * theta.sin()) as f32 * scale;
        buf[i] = z0;
        if i + 1 < n { buf[i + 1] = z1; }
        i += 2;
    }
}

/// Fill `n` i32 label indices uniformly in `[0, classes)`. Deterministic in `seed`.
#[no_mangle] pub unsafe extern "C" fn aether_fill_labels_i32(p: i64, n: c_int, classes: c_int, seed: i64) {
    if p == 0 || n <= 0 || classes <= 0 { return; }
    let n = n as usize;
    let buf = std::slice::from_raw_parts_mut(p as *mut i32, n);
    let c = classes as u64;
    let mut state = seed as u64;
    for slot in buf.iter_mut() {
        *slot = (splitmix64(&mut state) % c) as i32;
    }
}

/// Read a single f32 from a buffer at index `i`. Used by Aether `main()` to
/// observe loss values without needing array indexing in the language.
#[no_mangle] pub unsafe extern "C" fn aether_load_f32(p: i64, i: c_int) -> f32 {
    if p == 0 { return 0.0; }
    *((p as *const f32).add(i as usize))
}

/// Print one line of bench output: `<which>  M=… N=… K=… iters=… us=…`,
/// where `which` is 0 for CPU / 1 for GPU. The label_ptr arg is reserved
/// for a future per-bench tag string and is currently ignored.
#[no_mangle] pub extern "C" fn aether_print_bench(
    which: i64, m: c_int, n: c_int, k: c_int, iters: c_int, us: i64,
) -> c_int {
    let tag = if which == 0 { "cpu" } else { "gpu" };
    println!("{:3}  M={:>4}  N={:>4}  K={:>4}  iters={:>4}  us={:>10}", tag, m, n, k, iters, us);
    0
}

/// Print "step={step} loss={loss}\n" to stdout. Lets compiled Aether code
/// emit a training curve without needing format strings or stdio bindings.
///
/// Implementation note: writes directly to the Win32 standard-output handle
/// via WriteFile. This avoids `std::io::stdout()` and the lazy-init chain
/// behind `println!`, which (on Windows) reaches BCrypt primitives for the
/// stdio mutex's HashMap hasher seed. That chain AVs when this DLL is
/// loaded too early in process init under the self-hosted `--emit=pe-bin`
/// PE writer; the explicit WriteFile call sidesteps the whole thing.
#[no_mangle] pub extern "C" fn aether_print_loss(step: c_int, loss: f32) -> c_int {
    let mut buf = [0u8; 64];
    let n = format_loss_line(&mut buf, step, loss);
    unsafe { write_stdout(&buf[..n]); }
    0
}

#[cfg(windows)]
unsafe fn write_stdout(bytes: &[u8]) {
    extern "system" {
        fn GetStdHandle(n_std_handle: u32) -> *mut c_void;
        fn WriteFile(h: *mut c_void, b: *const c_void, n: u32,
                     written: *mut u32, overlapped: *mut c_void) -> i32;
    }
    let h = GetStdHandle(0xFFFF_FFF5); // STD_OUTPUT_HANDLE
    let mut written: u32 = 0;
    WriteFile(h, bytes.as_ptr() as *const c_void, bytes.len() as u32,
              &mut written, std::ptr::null_mut());
}

#[cfg(not(windows))]
unsafe fn write_stdout(bytes: &[u8]) {
    extern "C" { fn write(fd: c_int, buf: *const c_void, n: usize) -> isize; }
    let _ = write(1, bytes.as_ptr() as *const c_void, bytes.len());
}

/// Format `step={step} loss={loss:.6}\n` into `buf` without touching
/// `core::fmt::Write` (which on cdylibs drags in the same Rust-std stdio
/// init paths we're trying to avoid). Returns bytes written. `buf` must
/// be at least 64 bytes; output is truncated otherwise.
fn format_loss_line(buf: &mut [u8], step: c_int, loss: f32) -> usize {
    let mut i = 0usize;
    let prefix = b"step=";
    for &b in prefix { if i < buf.len() { buf[i] = b; i += 1; } }
    i += write_int(&mut buf[i..], step as i64);
    let mid = b" loss=";
    for &b in mid { if i < buf.len() { buf[i] = b; i += 1; } }
    i += write_f32(&mut buf[i..], loss, 6);
    if i < buf.len() { buf[i] = b'\n'; i += 1; }
    i
}

fn write_int(buf: &mut [u8], v: i64) -> usize {
    if buf.is_empty() { return 0; }
    let neg = v < 0;
    let mut n = if neg { (!v as u64).wrapping_add(1) } else { v as u64 };
    let mut tmp = [0u8; 20];
    let mut t = 0;
    if n == 0 { tmp[t] = b'0'; t += 1; }
    while n > 0 { tmp[t] = b'0' + (n % 10) as u8; n /= 10; t += 1; }
    let mut i = 0;
    if neg && i < buf.len() { buf[i] = b'-'; i += 1; }
    while t > 0 && i < buf.len() { t -= 1; buf[i] = tmp[t]; i += 1; }
    i
}

fn write_f32(buf: &mut [u8], v: f32, decimals: u32) -> usize {
    let mut i = 0usize;
    let neg = v < 0.0;
    let mut x = if neg { -v } else { v };
    if neg && i < buf.len() { buf[i] = b'-'; i += 1; }
    let int_part = x as u64;
    let mut tmp = [0u8; 20];
    let mut t = 0;
    let mut n = int_part;
    if n == 0 { tmp[t] = b'0'; t += 1; }
    while n > 0 { tmp[t] = b'0' + (n % 10) as u8; n /= 10; t += 1; }
    while t > 0 && i < buf.len() { t -= 1; buf[i] = tmp[t]; i += 1; }
    if i < buf.len() { buf[i] = b'.'; i += 1; }
    x -= int_part as f32;
    for _ in 0..decimals {
        x *= 10.0;
        let d = x as u32;
        if i < buf.len() { buf[i] = b'0' + (d % 10) as u8; i += 1; }
        x -= d as f32;
    }
    i
}

/// Store a single f32 to a buffer at index `i`.
#[no_mangle] pub unsafe extern "C" fn aether_store_f32(p: i64, i: c_int, v: f32) {
    if p == 0 { return; }
    *((p as *mut f32).add(i as usize)) = v;
}

// =====================================================================
// Test-only FFI surface — exercises the asm backend's f32 / f64 / cast /
// FFI-arg pipelines from compiled Aether. Real ops never go through here.
// =====================================================================

#[no_mangle] pub extern "C" fn aether_test_add_f32(a: f32, b: f32) -> f32 { a + b }
#[no_mangle] pub extern "C" fn aether_test_add_f64(a: f64, b: f64) -> f64 { a + b }
#[no_mangle] pub extern "C" fn aether_test_f32_to_i64(x: f32) -> i64 { x as i64 }
#[no_mangle] pub extern "C" fn aether_test_f64_to_i64(x: f64) -> i64 { x as i64 }
/// Mixed-class arg passing: int slot 0, f32 slot 1, int slot 2 → f32.
/// MS x64 puts them in rcx, xmm1, r8 respectively. Returns `(i + j) * f`.
#[no_mangle] pub extern "C" fn aether_test_mix_if(i: c_int, f: f32, j: c_int) -> f32 {
    (i + j) as f32 * f
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn tape_lifecycle() {
        unsafe {
            aether_autodiff_init(ptr::null_mut());
            aether_autodiff_push(ptr::null_mut(), ptr::null());
            aether_autodiff_push(ptr::null_mut(), ptr::null());
            assert_eq!(aether_rt_self_check(), 2);
            aether_autodiff_reverse(ptr::null_mut());
        }
    }

    #[test]
    fn all_reduce_is_safe() {
        aether_dist_all_reduce(ptr::null_mut(), 8, DistBackend::Nccl as c_int);
    }
}
