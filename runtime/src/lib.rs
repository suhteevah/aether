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

#[no_mangle] pub unsafe extern "C" fn aether_op_mse_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 {
    ops::mse_f32(pred as _, target as _, n as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mse_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int {
    ops::mse_backward_f32(pred as _, target as _, dpred as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mae_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 {
    ops::mae_f32(pred as _, target as _, n as _)
}

#[no_mangle] pub unsafe extern "C" fn aether_op_mae_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int {
    ops::mae_backward_f32(pred as _, target as _, dpred as _, n as _);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_with_logits_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 { ops::bce_with_logits_f32(pred as _, target as _, n as _) }

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_with_logits_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int { ops::bce_with_logits_backward_f32(pred as _, target as _, dpred as _, n as _); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 { ops::bce_f32(pred as _, target as _, n as _) }

#[no_mangle] pub unsafe extern "C" fn aether_op_bce_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int { ops::bce_backward_f32(pred as _, target as _, dpred as _, n as _); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_kl_div_f32(
    pred: *const c_void, target: *const c_void, n: c_int,
) -> f32 { ops::kl_div_f32(pred as _, target as _, n as _) }

#[no_mangle] pub unsafe extern "C" fn aether_op_kl_div_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int,
) -> c_int { ops::kl_div_backward_f32(pred as _, target as _, dpred as _, n as _); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_huber_f32(
    pred: *const c_void, target: *const c_void, n: c_int, delta: f32,
) -> f32 { ops::huber_f32(pred as _, target as _, n as _, delta) }

#[no_mangle] pub unsafe extern "C" fn aether_op_huber_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int, delta: f32,
) -> c_int { ops::huber_backward_f32(pred as _, target as _, dpred as _, n as _, delta); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_smooth_l1_f32(
    pred: *const c_void, target: *const c_void, n: c_int, beta: f32,
) -> f32 { ops::smooth_l1_f32(pred as _, target as _, n as _, beta) }

#[no_mangle] pub unsafe extern "C" fn aether_op_smooth_l1_backward_f32(
    pred: *const c_void, target: *const c_void, dpred: *mut c_void, n: c_int, beta: f32,
) -> c_int { ops::smooth_l1_backward_f32(pred as _, target as _, dpred as _, n as _, beta); 0 }

#[no_mangle] pub unsafe extern "C" fn aether_op_triplet_f32(
    anchor: *const c_void, positive: *const c_void, negative: *const c_void,
    d: c_int, margin: f32,
) -> f32 { ops::triplet_f32(anchor as _, positive as _, negative as _, d as _, margin) }

#[no_mangle] pub unsafe extern "C" fn aether_op_triplet_backward_f32(
    anchor: *const c_void, positive: *const c_void, negative: *const c_void,
    d_anchor: *mut c_void, d: c_int, margin: f32,
) -> c_int {
    ops::triplet_backward_f32(anchor as _, positive as _, negative as _, d_anchor as _, d as _, margin); 0
}

#[no_mangle] pub unsafe extern "C" fn aether_op_contrastive_f32(
    x1: *const c_void, x2: *const c_void, y: f32, d: c_int, margin: f32,
) -> f32 { ops::contrastive_f32(x1 as _, x2 as _, y, d as _, margin) }

#[no_mangle] pub unsafe extern "C" fn aether_op_contrastive_backward_f32(
    x1: *const c_void, x2: *const c_void, dx1: *mut c_void,
    y: f32, d: c_int, margin: f32,
) -> c_int {
    ops::contrastive_backward_f32(x1 as _, x2 as _, dx1 as _, y, d as _, margin); 0
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

/// Fill `n` f32s with the given constant `value`. Used to initialise
/// LayerNorm gamma/beta and other "set every element to k" patterns.
#[no_mangle] pub unsafe extern "C" fn aether_init_constant_f32(p: i64, n: c_int, value: f32) {
    if p == 0 || n <= 0 { return; }
    let buf = std::slice::from_raw_parts_mut(p as *mut f32, n as usize);
    for x in buf.iter_mut() { *x = value; }
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
    // Use read_unaligned: SafeTensors payloads land at byte 8+header_len,
    // which is rarely 4-aligned, and Aether code reads them via this fn.
    (p as *const f32).add(i as usize).read_unaligned()
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
// === Self-host bootstrap primitives (roadmap item #27) ============
//
// To rewrite aetherc in .aether we need: file I/O, byte-level memory
// access, raw allocation. Each fn here is the smallest possible C-ABI
// wedge — once enough of these exist, a baby tokenizer / parser /
// codegen can be written in .aether that targets these symbols.

use std::cell::Cell;
thread_local! { static LAST_FILE_SIZE: Cell<i64> = Cell::new(0); }

/// Read an entire file at the 0-terminated C string `path` into a fresh
/// heap buffer. Returns the buffer pointer as `i64` (0 = error). The
/// buffer's length is recorded thread-locally and retrievable via
/// `aether_file_size_last`. Caller frees with `aether_free_bytes`.
#[no_mangle] pub unsafe extern "C" fn aether_read_file(path: i64) -> i64 {
    if path == 0 { return 0; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p_bytes = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(p_str) = std::str::from_utf8(p_bytes) else { return 0; };
    let Ok(buf) = std::fs::read(p_str) else { return 0; };
    let n = buf.len();
    LAST_FILE_SIZE.with(|c| c.set(n as i64));
    let boxed = buf.into_boxed_slice();
    let raw = Box::into_raw(boxed);
    raw as *mut u8 as i64
}

/// Length in bytes of the most recent `aether_read_file` success.
#[no_mangle] pub extern "C" fn aether_file_size_last() -> i64 {
    LAST_FILE_SIZE.with(|c| c.get())
}

/// Allocate `n` bytes (zero-initialised) on the heap. Returns ptr as i64.
/// Caller frees with `aether_free_bytes` — DON'T pass to `aether_free_f32`.
#[no_mangle] pub extern "C" fn aether_alloc_bytes(n: i64) -> i64 {
    if n <= 0 { return 0; }
    let v = vec![0u8; n as usize];
    let boxed = v.into_boxed_slice();
    let p = Box::into_raw(boxed) as *mut u8 as i64;
    prof_alloc(n);
    p
}

/// Free a buffer previously returned by `aether_alloc_bytes` /
/// `aether_read_file`. `n` MUST be the original length passed in.
#[no_mangle] pub unsafe extern "C" fn aether_free_bytes(p: i64, n: i64) {
    if p == 0 || n <= 0 { return; }
    let slice = std::slice::from_raw_parts_mut(p as *mut u8, n as usize);
    drop(Box::from_raw(slice as *mut [u8]));
    prof_free(n);
}

/// Load a single byte from the buffer at offset `i`. Returns the byte
/// as a 0..=255 i64 (so Aether's int Bin ops work on it cleanly).
#[no_mangle] pub unsafe extern "C" fn aether_byte_at(p: i64, i: i64) -> i64 {
    if p == 0 || i < 0 { return -1; }
    *(p as *const u8).add(i as usize) as i64
}

/// Store a byte (low 8 bits of `v`) at offset `i` in the buffer.
#[no_mangle] pub unsafe extern "C" fn aether_byte_set(p: i64, i: i64, v: i64) {
    if p == 0 || i < 0 { return; }
    *(p as *mut u8).add(i as usize) = (v & 0xFF) as u8;
}

/// Write `n` bytes from `buf` to file `path` (overwriting). Returns 0 on
/// success, non-zero on failure.
#[no_mangle] pub unsafe extern "C" fn aether_write_file(path: i64, buf: i64, n: i64) -> i32 {
    if path == 0 || buf == 0 || n < 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p_bytes = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(p_str) = std::str::from_utf8(p_bytes) else { return -2; };
    let data = std::slice::from_raw_parts(buf as *const u8, n as usize);
    match std::fs::write(p_str, data) {
        Ok(()) => 0,
        Err(_) => -3,
    }
}

/// Allocate `n` bytes of executable + writable memory via `VirtualAlloc`
/// (Windows). Returns a pointer suitable for `aether_call_jit_i64`.
/// Caller frees with `aether_free_executable`. Returns 0 on failure.
#[no_mangle] pub unsafe extern "C" fn aether_alloc_executable(n: i64) -> i64 {
    if n <= 0 { return 0; }
    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualAlloc(addr: *mut u8, size: usize, alloc_type: u32, protect: u32) -> *mut u8;
    }
    const MEM_COMMIT: u32 = 0x1000;
    const MEM_RESERVE: u32 = 0x2000;
    const PAGE_EXECUTE_READWRITE: u32 = 0x40;
    let p = VirtualAlloc(std::ptr::null_mut(), n as usize, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE);
    if p.is_null() { 0 } else { p as i64 }
}

#[no_mangle] pub unsafe extern "C" fn aether_free_executable(p: i64, _n: i64) {
    if p == 0 { return; }
    #[link(name = "kernel32")]
    extern "system" {
        fn VirtualFree(addr: *mut u8, size: usize, free_type: u32) -> i32;
    }
    const MEM_RELEASE: u32 = 0x8000;
    let _ = VirtualFree(p as *mut u8, 0, MEM_RELEASE);
}

/// Cast `p` to `fn() -> i64` and invoke it. The caller is responsible for
/// having written valid x86-64 code (with proper prologue/epilogue) into
/// the buffer — no safety net here, this is the JIT escape hatch.
#[no_mangle] pub unsafe extern "C" fn aether_call_jit_i64(p: i64) -> i64 {
    if p == 0 { return 0; }
    let f: extern "C" fn() -> i64 = std::mem::transmute(p);
    f()
}

/// Byte-compare two buffers of length `n`. Returns 1 on equal, 0 otherwise.
/// Used by the self-hosted tokenizer to match ident spans against keywords.
#[no_mangle] pub unsafe extern "C" fn aether_bytes_eq(a: i64, b: i64, n: i64) -> i64 {
    if a == 0 || b == 0 || n < 0 { return 0; }
    let s1 = std::slice::from_raw_parts(a as *const u8, n as usize);
    let s2 = std::slice::from_raw_parts(b as *const u8, n as usize);
    if s1 == s2 { 1 } else { 0 }
}

/// Length of the 0-terminated C string at `p` (excluding the NUL).
#[no_mangle] pub unsafe extern "C" fn aether_str_len(p: i64) -> i64 {
    if p == 0 { return 0; }
    let mut n = 0i64;
    while *(p as *const u8).add(n as usize) != 0 { n += 1; }
    n
}

/// Write `n` bytes from buffer `p` to stdout (no trailing newline).
#[no_mangle] pub unsafe extern "C" fn aether_print_bytes(p: i64, n: i64) {
    if p == 0 || n <= 0 { return; }
    use std::io::Write;
    let slice = std::slice::from_raw_parts(p as *const u8, n as usize);
    let _ = std::io::stdout().write_all(slice);
    let _ = std::io::stdout().flush();
}

/// =====================================================================
/// P6.9 — atomics + thread spawn primitives.
///
/// Atomics use Rust's `std::sync::atomic::AtomicI64` over a raw pointer
/// the caller is responsible for providing as 8-byte aligned storage
/// (typical use: `aether_alloc_bytes(16)` then pass the buffer).
/// Thread spawn uses Win32 `CreateThread` with a fn-ptr trampoline.
/// =====================================================================
#[no_mangle] pub unsafe extern "C" fn aether_atomic_fetch_add_i64(addr: i64, val: i64) -> i64 {
    if addr == 0 { return 0; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    a.fetch_add(val, Ordering::SeqCst)
}

#[no_mangle] pub unsafe extern "C" fn aether_atomic_load_i64(addr: i64) -> i64 {
    if addr == 0 { return 0; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    a.load(Ordering::SeqCst)
}

#[no_mangle] pub unsafe extern "C" fn aether_atomic_store_i64(addr: i64, val: i64) {
    if addr == 0 { return; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    a.store(val, Ordering::SeqCst);
}

#[no_mangle] pub unsafe extern "C" fn aether_atomic_cas_i64(addr: i64, expected: i64, new: i64) -> i64 {
    if addr == 0 { return -1; }
    use std::sync::atomic::{AtomicI64, Ordering};
    let a = &*(addr as *const AtomicI64);
    match a.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(prev) => prev,
        Err(prev) => prev,
    }
}

/// Spawn a thread that runs `fn_ptr(arg)`. Returns the OS thread handle
/// (or 0 on failure). Caller joins via `aether_thread_join`.
#[no_mangle] pub unsafe extern "C" fn aether_thread_spawn(fn_ptr: i64, arg: i64) -> i64 {
    if fn_ptr == 0 { return 0; }
    let payload: Box<(i64, i64)> = Box::new((fn_ptr, arg));
    let raw = Box::into_raw(payload) as i64;
    let handle = std::thread::spawn(move || {
        let p = raw as *mut (i64, i64);
        let (fp, ar) = *Box::from_raw(p);
        let f: extern "C" fn(i64) -> i64 = std::mem::transmute(fp);
        f(ar);
    });
    Box::into_raw(Box::new(handle)) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_thread_join(handle: i64) -> i32 {
    if handle == 0 { return -1; }
    let h: Box<std::thread::JoinHandle<()>> =
        Box::from_raw(handle as *mut std::thread::JoinHandle<()>);
    match h.join() {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Integer absolute value. Free fn — Aether doesn't have method-on-scalar
/// dispatch yet, so primitives live as `aether_*` extern fns.
// =====================================================================
// SafeTensors reader (roadmap P7.5).
//
// Format: <8-byte LE u64 header_len> <header_len bytes of JSON> <raw payload>.
// JSON: {"name":{"dtype":"F32","shape":[...],"data_offsets":[start,end]}, ...}.
//
// Phase-0 reader is byte-exact for the format we emit (single-pass linear
// scan, no general JSON). Good enough to round-trip our own writes; HF
// weight files use the same canonical shape so it works on those too.
// =====================================================================

/// Validate the 8-byte length prefix and return the JSON-header byte count.
/// Returns -1 if `buf` is null, `len` is too small, or the header overruns.
#[no_mangle] pub unsafe extern "C" fn safetensors_parse_header(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 8 { return -1; }
    let p = buf as *const u8;
    let mut hdr: u64 = 0;
    for i in 0..8 { hdr |= (*p.add(i) as u64) << (i * 8); }
    if 8u64 + hdr > len as u64 { return -1; }
    hdr as i64
}

/// Look up tensor `name` in the SafeTensors blob and return a pointer to
/// the start of its f32 payload (i.e. `buf + 8 + hdr_len + data_offsets[0]`).
/// Returns 0 if the name is missing or the JSON is malformed.
#[no_mangle] pub unsafe extern "C" fn safetensors_get_tensor_f32(
    buf: i64, len: i64, name: i64, name_len: i64,
) -> i64 {
    let hdr_len = safetensors_parse_header(buf, len);
    if hdr_len < 0 || name == 0 || name_len <= 0 { return 0; }
    let json = std::slice::from_raw_parts((buf as *const u8).add(8), hdr_len as usize);
    let name_bytes = std::slice::from_raw_parts(name as *const u8, name_len as usize);
    let Ok(json_str) = std::str::from_utf8(json) else { return 0; };
    let Ok(name_str) = std::str::from_utf8(name_bytes) else { return 0; };

    // Locate `"<name>":` at a key position. We require the colon to follow
    // the closing quote (with optional ws) so that "foo" inside a value
    // string doesn't false-match.
    let needle = format!("\"{}\"", name_str);
    let mut search_from = 0usize;
    let key_pos = loop {
        let Some(idx) = json_str[search_from..].find(&needle) else { return 0; };
        let abs = search_from + idx;
        let after = &json_str[abs + needle.len()..];
        let trimmed = after.trim_start();
        if trimmed.starts_with(':') { break abs; }
        search_from = abs + needle.len();
    };

    // Inside the matching object, find `"data_offsets":[a,b]` and parse `a`.
    let rest = &json_str[key_pos + needle.len()..];
    let Some(off_idx) = rest.find("\"data_offsets\":[") else { return 0; };
    let after = &rest[off_idx + "\"data_offsets\":[".len()..];
    let Some(comma) = after.find(',') else { return 0; };
    let Ok(start) = after[..comma].trim().parse::<i64>() else { return 0; };

    buf + 8 + hdr_len + start
}

// =====================================================================
// Microbench-driven kernel selection (P10.10).
//
// `aether_op_matmul_f32_auto` runs both the naive and blocked matmul
// kernels for the requested (m,k,n) on a one-shot probe (the first time
// that shape is seen), times each, caches the winner, and dispatches
// future calls of the same shape to the chosen kernel.
//
// The probe runs on a side buffer so the user-visible output is computed
// only by the chosen kernel (no double work on the user's buffers).
// =====================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatmulKernel { Naive = 0, Blocked = 1 }

// Single-threaded cache, same UnsafeCell pattern as TAPE above (avoids
// the std::sync init chain that AVs under the self-hosted PE writer).
struct MatmulCache(UnsafeCell<Vec<((usize, usize, usize), MatmulKernel)>>);
unsafe impl Sync for MatmulCache {}
static MATMUL_CACHE: MatmulCache = MatmulCache(UnsafeCell::new(Vec::new()));

unsafe fn matmul_cache_lookup(key: (usize, usize, usize)) -> Option<MatmulKernel> {
    let v = &*MATMUL_CACHE.0.get();
    v.iter().find_map(|(k, val)| if *k == key { Some(*val) } else { None })
}
unsafe fn matmul_cache_insert(key: (usize, usize, usize), val: MatmulKernel) {
    let v = &mut *MATMUL_CACHE.0.get();
    v.push((key, val));
}

unsafe fn probe_matmul_f32(m: usize, k: usize, n: usize) -> MatmulKernel {
    // Fresh deterministic input for the probe.
    let a: Vec<f32> = (0..m*k).map(|i| ((i * 7 + 3) as f32 * 0.001) % 1.0).collect();
    let b: Vec<f32> = (0..k*n).map(|i| ((i * 11 + 5) as f32 * 0.001) % 1.0).collect();
    let mut o = vec![0.0f32; m * n];
    let t0 = wall_us_now();
    ops::matmul_f32(a.as_ptr(), b.as_ptr(), o.as_mut_ptr(), m, k, n);
    let t_naive = wall_us_now() - t0;
    let t1 = wall_us_now();
    ops::matmul_blocked_f32(a.as_ptr(), b.as_ptr(), o.as_mut_ptr(), m, k, n);
    let t_blocked = wall_us_now() - t1;
    if t_blocked < t_naive { MatmulKernel::Blocked } else { MatmulKernel::Naive }
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_f32_auto(
    a: *const c_void, b: *const c_void, out: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    let m = m as usize; let k = k as usize; let n = n as usize;
    let key = (m, k, n);
    let chosen = match matmul_cache_lookup(key) {
        Some(v) => v,
        None => {
            let v = probe_matmul_f32(m, k, n);
            matmul_cache_insert(key, v);
            v
        }
    };
    match chosen {
        MatmulKernel::Naive   => ops::matmul_f32(a as _, b as _, out as _, m, k, n),
        MatmulKernel::Blocked => ops::matmul_blocked_f32(a as _, b as _, out as _, m, k, n),
    }
    0
}

/// Inspect the cached kernel pick for a shape (debug / witness only):
/// returns 0=naive, 1=blocked, -1=not yet probed.
#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_f32_auto_kernel(
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    let key = (m as usize, k as usize, n as usize);
    match matmul_cache_lookup(key) {
        Some(MatmulKernel::Naive)   => 0,
        Some(MatmulKernel::Blocked) => 1,
        None => -1,
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_blocked_f32(
    a: *const c_void, b: *const c_void, out: *mut c_void,
    m: c_int, k: c_int, n: c_int,
) -> c_int {
    ops::matmul_blocked_f32(a as _, b as _, out as _, m as _, k as _, n as _);
    0
}

// =====================================================================
// Profiling primitives (P8.6 — allocator stats + stopwatch).
// =====================================================================

use std::sync::atomic::{AtomicI64, Ordering};

static ALLOC_TOTAL: AtomicI64 = AtomicI64::new(0);
static ALLOC_LIVE:  AtomicI64 = AtomicI64::new(0);
static ALLOC_PEAK:  AtomicI64 = AtomicI64::new(0);

fn prof_alloc(n: i64) {
    ALLOC_TOTAL.fetch_add(n, Ordering::Relaxed);
    let live = ALLOC_LIVE.fetch_add(n, Ordering::Relaxed) + n;
    // Atomic "peak = max(peak, live)" via CAS loop.
    let mut peak = ALLOC_PEAK.load(Ordering::Relaxed);
    while live > peak {
        match ALLOC_PEAK.compare_exchange_weak(peak, live, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(p) => peak = p,
        }
    }
}

fn prof_free(n: i64) { ALLOC_LIVE.fetch_sub(n, Ordering::Relaxed); }

#[no_mangle] pub extern "C" fn aether_prof_alloc_total() -> i64 { ALLOC_TOTAL.load(Ordering::Relaxed) }
#[no_mangle] pub extern "C" fn aether_prof_alloc_live()  -> i64 { ALLOC_LIVE.load(Ordering::Relaxed) }
#[no_mangle] pub extern "C" fn aether_prof_alloc_peak()  -> i64 { ALLOC_PEAK.load(Ordering::Relaxed) }
#[no_mangle] pub extern "C" fn aether_prof_reset() -> i32 {
    ALLOC_TOTAL.store(0, Ordering::Relaxed);
    ALLOC_LIVE.store(0,  Ordering::Relaxed);
    ALLOC_PEAK.store(0,  Ordering::Relaxed);
    0
}

/// Stopwatch: take `wall_us` at start, take `wall_us - start` at end.
/// Sugar; callers can use `aether_wall_us` directly. Keeps the witness
/// readable.
#[no_mangle] pub extern "C" fn aether_timer_start() -> i64 { wall_us_now() }
#[no_mangle] pub extern "C" fn aether_timer_elapsed_us(start: i64) -> i64 {
    wall_us_now() - start
}

fn wall_us_now() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// =====================================================================
// Filesystem primitives (P6.13 standard I/O surface).
// =====================================================================

/// Returns 1 if the path exists (file OR directory), 0 otherwise.
#[no_mangle] pub unsafe extern "C" fn aether_path_exists(path: i64) -> i64 {
    if path == 0 { return 0; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return 0; };
    if std::path::Path::new(s).exists() { 1 } else { 0 }
}

/// File size in bytes; -1 on error (missing, not a regular file, etc.).
#[no_mangle] pub unsafe extern "C" fn aether_file_size(path: i64) -> i64 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::metadata(s) {
        Ok(m) if m.is_file() => m.len() as i64,
        _ => -1,
    }
}

/// Returns 1 if the path is a directory, 0 if file, -1 on error.
#[no_mangle] pub unsafe extern "C" fn aether_is_dir(path: i64) -> i64 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::metadata(s) {
        Ok(m) => if m.is_dir() { 1 } else { 0 },
        Err(_) => -1,
    }
}

/// `mkdir -p` style: create the directory and all missing parents.
/// Returns 0 on success, negative on failure.
#[no_mangle] pub unsafe extern "C" fn aether_create_dir_all(path: i64) -> i32 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::create_dir_all(s) { Ok(()) => 0, Err(_) => -2 }
}

/// Remove a single file. Returns 0 on success, negative on failure.
#[no_mangle] pub unsafe extern "C" fn aether_remove_file(path: i64) -> i32 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::remove_file(s) { Ok(()) => 0, Err(_) => -2 }
}

/// Copy file `src` → `dst`. Returns bytes copied on success, -1 on failure.
#[no_mangle] pub unsafe extern "C" fn aether_copy_file(src: i64, dst: i64) -> i64 {
    if src == 0 || dst == 0 { return -1; }
    let mut sl = 0usize; while *(src as *const u8).add(sl) != 0 { sl += 1; }
    let mut dl = 0usize; while *(dst as *const u8).add(dl) != 0 { dl += 1; }
    let s = std::slice::from_raw_parts(src as *const u8, sl);
    let d = std::slice::from_raw_parts(dst as *const u8, dl);
    let Ok(s_str) = std::str::from_utf8(s) else { return -1; };
    let Ok(d_str) = std::str::from_utf8(d) else { return -1; };
    match std::fs::copy(s_str, d_str) { Ok(n) => n as i64, Err(_) => -1 }
}

/// Atomic rename `src` → `dst` (cross-device may fail on some FS).
#[no_mangle] pub unsafe extern "C" fn aether_rename(src: i64, dst: i64) -> i32 {
    if src == 0 || dst == 0 { return -1; }
    let mut sl = 0usize; while *(src as *const u8).add(sl) != 0 { sl += 1; }
    let mut dl = 0usize; while *(dst as *const u8).add(dl) != 0 { dl += 1; }
    let s = std::slice::from_raw_parts(src as *const u8, sl);
    let d = std::slice::from_raw_parts(dst as *const u8, dl);
    let Ok(s_str) = std::str::from_utf8(s) else { return -1; };
    let Ok(d_str) = std::str::from_utf8(d) else { return -1; };
    match std::fs::rename(s_str, d_str) { Ok(()) => 0, Err(_) => -2 }
}

/// Count immediate entries in directory `path`. Returns -1 on error.
#[no_mangle] pub unsafe extern "C" fn aether_read_dir_count(path: i64) -> i64 {
    if path == 0 { return -1; }
    let mut len = 0usize;
    while *(path as *const u8).add(len) != 0 { len += 1; }
    let p = std::slice::from_raw_parts(path as *const u8, len);
    let Ok(s) = std::str::from_utf8(p) else { return -1; };
    match std::fs::read_dir(s) {
        Ok(rd) => rd.count() as i64,
        Err(_) => -1,
    }
}

// =====================================================================
// TCP primitives (P8.5 inference + serving — network I/O surface).
//
// Listener / stream handles are opaque i64 indices into a single-threaded
// Vec<Option<Box<T>>>. Returning a Vec index keeps the C-ABI integer-only,
// matches the i64-handle pattern used elsewhere in this crate, and lets us
// drop entries by setting the slot to None without invalidating other
// outstanding handles. Same UnsafeCell trick as TAPE / MATMUL_CACHE — the
// runtime is single-threaded by contract; concurrent calls into TCP fns
// from multiple threads are UB. The serving roadmap will add a real
// thread-pool primitive on top of this surface in a follow-up.
// =====================================================================

struct TcpListenerCell(UnsafeCell<Vec<Option<Box<std::net::TcpListener>>>>);
unsafe impl Sync for TcpListenerCell {}
static TCP_LISTENERS: TcpListenerCell = TcpListenerCell(UnsafeCell::new(Vec::new()));

struct TcpStreamCell(UnsafeCell<Vec<Option<Box<std::net::TcpStream>>>>);
unsafe impl Sync for TcpStreamCell {}
static TCP_STREAMS: TcpStreamCell = TcpStreamCell(UnsafeCell::new(Vec::new()));

unsafe fn tcp_listeners() -> &'static mut Vec<Option<Box<std::net::TcpListener>>> {
    &mut *TCP_LISTENERS.0.get()
}
unsafe fn tcp_streams() -> &'static mut Vec<Option<Box<std::net::TcpStream>>> {
    &mut *TCP_STREAMS.0.get()
}

/// Bind a TCP listener on `127.0.0.1:port`. Pass `port = 0` for an
/// OS-assigned ephemeral port. Returns the listener handle (>= 0) or -1
/// on failure. Use `aether_tcp_listener_port` to read back the actual
/// bound port when `port = 0` was passed in.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_listen(port: i64) -> i64 {
    if !(0..=65535).contains(&port) { return -1; }
    let addr = format!("127.0.0.1:{}", port);
    match std::net::TcpListener::bind(&addr) {
        Ok(l) => {
            let v = tcp_listeners();
            // Re-use a vacated slot if available.
            for (i, slot) in v.iter_mut().enumerate() {
                if slot.is_none() { *slot = Some(Box::new(l)); return i as i64; }
            }
            v.push(Some(Box::new(l)));
            (v.len() - 1) as i64
        }
        Err(_) => -1,
    }
}

/// Bound local port for a listener (useful when port=0 was passed). -1 on err.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_listener_port(handle: i64) -> i64 {
    if handle < 0 { return -1; }
    let v = tcp_listeners();
    let idx = handle as usize;
    if idx >= v.len() { return -1; }
    match v[idx].as_ref() {
        Some(l) => match l.local_addr() {
            Ok(a) => a.port() as i64,
            Err(_) => -1,
        },
        None => -1,
    }
}

/// Drop the listener at `handle`. 0 on success, -1 if the handle is invalid.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_close(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let v = tcp_listeners();
    let idx = handle as usize;
    if idx >= v.len() { return -1; }
    if v[idx].is_none() { return -1; }
    v[idx] = None;
    0
}

/// Block waiting for one inbound connection on `listener`. Returns the
/// stream handle (>= 0) or -1 on failure / invalid listener handle.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_accept_one(listener: i64) -> i64 {
    if listener < 0 { return -1; }
    let v = tcp_listeners();
    let idx = listener as usize;
    if idx >= v.len() { return -1; }
    let Some(ref l) = v[idx] else { return -1; };
    let stream = match l.accept() {
        Ok((s, _)) => s,
        Err(_) => return -1,
    };
    let s = tcp_streams();
    for (i, slot) in s.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(stream)); return i as i64; }
    }
    s.push(Some(Box::new(stream)));
    (s.len() - 1) as i64
}

/// Drop the stream at `handle`. 0 on success, -1 if invalid.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_stream_close(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let v = tcp_streams();
    let idx = handle as usize;
    if idx >= v.len() { return -1; }
    if v[idx].is_none() { return -1; }
    v[idx] = None;
    0
}

/// Connect a TCP stream to `127.0.0.1:port`. Returns stream handle or -1.
/// Phase 0.5 client-side counterpart so a single-process witness can drive
/// both ends of a connection without spawning extra processes.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_connect(port: i64) -> i64 {
    if !(0..=65535).contains(&port) { return -1; }
    let addr = format!("127.0.0.1:{}", port);
    match std::net::TcpStream::connect(&addr) {
        Ok(s) => {
            let v = tcp_streams();
            for (i, slot) in v.iter_mut().enumerate() {
                if slot.is_none() { *slot = Some(Box::new(s)); return i as i64; }
            }
            v.push(Some(Box::new(s)));
            (v.len() - 1) as i64
        }
        Err(_) => -1,
    }
}

/// Send `n` bytes from `buf` over `stream`. Returns bytes written, or -1 on err.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_send(stream: i64, buf: i64, n: i64) -> i64 {
    use std::io::Write;
    if stream < 0 || buf == 0 || n < 0 { return -1; }
    let v = tcp_streams();
    let idx = stream as usize;
    if idx >= v.len() { return -1; }
    let Some(ref mut s) = v[idx] else { return -1; };
    let slice = std::slice::from_raw_parts(buf as *const u8, n as usize);
    match s.write(slice) {
        Ok(written) => written as i64,
        Err(_) => -1,
    }
}

/// Recv up to `n` bytes from `stream` into `buf`. Returns bytes read
/// (0 on EOF / clean close, -1 on error). Blocking.
#[no_mangle] pub unsafe extern "C" fn aether_tcp_recv(stream: i64, buf: i64, n: i64) -> i64 {
    use std::io::Read;
    if stream < 0 || buf == 0 || n < 0 { return -1; }
    let v = tcp_streams();
    let idx = stream as usize;
    if idx >= v.len() { return -1; }
    let Some(ref mut s) = v[idx] else { return -1; };
    let slice = std::slice::from_raw_parts_mut(buf as *mut u8, n as usize);
    match s.read(slice) {
        Ok(got) => got as i64,
        Err(_) => -1,
    }
}

// =====================================================================
// GGUF format header parser (P7.4 partial — quant kernels still TBD).
//
// Layout (little-endian throughout):
//   bytes 0..4   : magic "GGUF"
//   bytes 4..8   : u32 version
//   bytes 8..16  : u64 tensor_count
//   bytes 16..24 : u64 metadata_kv_count
//   ...          : metadata + tensor info + tensor data (quantized)
//
// Phase-1 surface: validate magic, expose version + counts. Quant
// dequantization is the L follow-on.
// =====================================================================

#[no_mangle] pub unsafe extern "C" fn gguf_parse_magic(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 4 { return 0; }
    let p = buf as *const u8;
    if *p == b'G' && *p.add(1) == b'G' && *p.add(2) == b'U' && *p.add(3) == b'F' { 1 } else { 0 }
}

#[no_mangle] pub unsafe extern "C" fn gguf_version(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 8 { return -1; }
    let p = buf as *const u8;
    let mut v = 0u32;
    for i in 0..4 { v |= (*p.add(4 + i) as u32) << (i * 8); }
    v as i64
}

#[no_mangle] pub unsafe extern "C" fn gguf_tensor_count(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 16 { return -1; }
    let p = buf as *const u8;
    let mut v = 0u64;
    for i in 0..8 { v |= (*p.add(8 + i) as u64) << (i * 8); }
    v as i64
}

#[no_mangle] pub unsafe extern "C" fn gguf_metadata_count(buf: i64, len: i64) -> i64 {
    if buf == 0 || len < 24 { return -1; }
    let p = buf as *const u8;
    let mut v = 0u64;
    for i in 0..8 { v |= (*p.add(16 + i) as u64) << (i * 8); }
    v as i64
}

/// Atomic file save: write `n` f32s to a single-tensor SafeTensors file at
/// `path`, named "data" with shape [n]. Goes through a `<path>.tmp`
/// staging file + rename so a crash mid-write can't leave the destination
/// half-written. Returns 0 on success, negative on failure (P8.8).
#[no_mangle] pub unsafe extern "C" fn safetensors_save_f32(
    path: i64, ptr: i64, n: i64,
) -> i32 {
    if path == 0 || ptr == 0 || n < 0 { return -1; }
    let mut path_len = 0usize;
    while *(path as *const u8).add(path_len) != 0 { path_len += 1; }
    let Ok(path_str) = std::str::from_utf8(
        std::slice::from_raw_parts(path as *const u8, path_len)
    ) else { return -1; };

    let n_us = n as usize;
    let payload_bytes = n_us * 4;
    let json = format!(
        "{{\"data\":{{\"dtype\":\"F32\",\"shape\":[{n}],\"data_offsets\":[0,{}]}}}}",
        payload_bytes
    );
    let json_bytes = json.as_bytes();
    let header_len = json_bytes.len() as u64;
    let total = 8 + json_bytes.len() + payload_bytes;
    let mut buf = vec![0u8; total];
    buf[..8].copy_from_slice(&header_len.to_le_bytes());
    buf[8..8 + json_bytes.len()].copy_from_slice(json_bytes);
    let payload = std::slice::from_raw_parts(ptr as *const u8, payload_bytes);
    buf[8 + json_bytes.len()..].copy_from_slice(payload);

    let tmp_path = format!("{}.tmp", path_str);
    if std::fs::write(&tmp_path, &buf).is_err() { return -2; }
    if std::fs::rename(&tmp_path, path_str).is_err() { return -3; }
    0
}

/// Load `n` f32s from a single-tensor SafeTensors file at `path` (the
/// "data" tensor) into the destination buffer at `dst`. Returns 0 on
/// success, negative on failure. (P8.8)
#[no_mangle] pub unsafe extern "C" fn safetensors_load_f32(
    path: i64, dst: i64, n: i64,
) -> i32 {
    if path == 0 || dst == 0 || n < 0 { return -1; }
    let mut path_len = 0usize;
    while *(path as *const u8).add(path_len) != 0 { path_len += 1; }
    let Ok(path_str) = std::str::from_utf8(
        std::slice::from_raw_parts(path as *const u8, path_len)
    ) else { return -1; };
    let Ok(bytes) = std::fs::read(path_str) else { return -2; };

    // Parse header length.
    if bytes.len() < 8 { return -3; }
    let mut hdr = 0u64;
    for i in 0..8 { hdr |= (bytes[i] as u64) << (i * 8); }
    if 8u64 + hdr > bytes.len() as u64 { return -3; }

    let json = match std::str::from_utf8(&bytes[8..8 + hdr as usize]) {
        Ok(s) => s, Err(_) => return -4,
    };
    let needle = "\"data_offsets\":[";
    let Some(off_idx) = json.find(needle) else { return -5; };
    let after = &json[off_idx + needle.len()..];
    let Some(comma) = after.find(',') else { return -5; };
    let Ok(start) = after[..comma].trim().parse::<usize>() else { return -5; };

    let payload_start = 8 + hdr as usize + start;
    let payload_len = (n as usize) * 4;
    if payload_start + payload_len > bytes.len() { return -6; }

    std::ptr::copy_nonoverlapping(
        bytes.as_ptr().add(payload_start),
        dst as *mut u8,
        payload_len,
    );
    0
}

#[no_mangle] pub extern "C" fn aether_abs_i64(x: i64) -> i64 { x.wrapping_abs() }
#[no_mangle] pub extern "C" fn aether_min_i64(a: i64, b: i64) -> i64 { if a < b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_max_i64(a: i64, b: i64) -> i64 { if a > b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_min_f32(a: f32, b: f32) -> f32 { if a < b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_max_f32(a: f32, b: f32) -> f32 { if a > b { a } else { b } }
#[no_mangle] pub extern "C" fn aether_abs_f32(x: f32) -> f32 { x.abs() }
#[no_mangle] pub extern "C" fn aether_sqrt_f32(x: f32) -> f32 { x.sqrt() }

// =====================================================================
// dtype conversions (P7.1) — bf16 + f16 lanes carried as i32 (low 16 bits).
// =====================================================================

/// f32 → bf16: truncate to high 16 bits with round-to-nearest-even tie-break.
#[no_mangle] pub extern "C" fn aether_f32_to_bf16(x: f32) -> i32 {
    let bits = x.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        // NaN: preserve quiet bit.
        return ((bits >> 16) | 0x40) as i32 & 0xFFFF;
    }
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1));
    ((rounded >> 16) & 0xFFFF) as i32
}

/// bf16 (low 16 bits of `b`) → f32 by zero-extending into the high half.
#[no_mangle] pub extern "C" fn aether_bf16_to_f32(b: i32) -> f32 {
    let bits = ((b as u32) & 0xFFFF) << 16;
    f32::from_bits(bits)
}

/// f32 → IEEE-754 binary16 (half) as low 16 bits of i32.
#[no_mangle] pub extern "C" fn aether_f32_to_f16(x: f32) -> i32 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u32;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp == 0xFF {
        if mant != 0 { return (sign | 0x7E00) as i32; } // NaN
        return (sign | 0x7C00) as i32;                  // Inf
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F { return (sign | 0x7C00) as i32; }    // overflow → Inf
    if new_exp <= 0 {
        if new_exp < -10 { return sign as i32; }              // underflow → 0
        // Subnormal: shift mantissa with implicit 1.
        let shift = 14 - new_exp;
        let m = (mant | 0x0080_0000) >> shift;
        return (sign | (m & 0x03FF)) as i32;
    }
    let half_mant = mant >> 13;
    (sign | ((new_exp as u32) << 10) | half_mant) as i32
}

/// Mixed-precision matmul: inputs `a` and `b` are bf16 buffers (each
/// element stored as a u16 inside an i32 lane — see ABI note in
/// stdlib/runtime.aether), upcast on read, accumulated and stored in
/// f32. Shapes match `aether_op_matmul_f32`. Witness for P8.7.
#[no_mangle] pub unsafe extern "C" fn aether_op_matmul_bf16_f32_out(
    a: i64, b: i64, out: i64, m: c_int, k: c_int, n: c_int,
) -> c_int {
    let m = m as usize; let k = k as usize; let n = n as usize;
    let a_buf = std::slice::from_raw_parts(a as *const i32, m * k);
    let b_buf = std::slice::from_raw_parts(b as *const i32, k * n);
    let out = std::slice::from_raw_parts_mut(out as *mut f32, m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                let av = aether_bf16_to_f32(a_buf[i * k + kk]);
                let bv = aether_bf16_to_f32(b_buf[kk * n + j]);
                acc += av * bv;
            }
            out[i * n + j] = acc;
        }
    }
    0
}

/// Pack `n` f32 values from `src` into bf16 in `dst` (low 16 bits per i32).
#[no_mangle] pub unsafe extern "C" fn aether_pack_f32_to_bf16(
    src: i64, dst: i64, n: c_int,
) -> c_int {
    let n = n as usize;
    let s = std::slice::from_raw_parts(src as *const f32, n);
    let d = std::slice::from_raw_parts_mut(dst as *mut i32, n);
    for i in 0..n { d[i] = aether_f32_to_bf16(s[i]); }
    0
}

/// IEEE-754 binary16 (low 16 bits of `h`) → f32.
#[no_mangle] pub extern "C" fn aether_f16_to_f32(h: i32) -> f32 {
    let h = (h as u32) & 0xFFFF;
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x03FF;
    let bits = if exp == 0 {
        if mant == 0 { sign << 31 }
        else {
            // Subnormal.
            let mut m = mant;
            let mut e = -14i32;
            while (m & 0x0400) == 0 { m <<= 1; e -= 1; }
            m &= 0x03FF;
            (sign << 31) | (((e + 127) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFFu32 << 23) | (mant << 13)
    } else {
        let new_exp = exp as i32 - 15 + 127;
        (sign << 31) | ((new_exp as u32) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// =====================================================================
// libm replacements (P9.6 — drop dependency on platform libm).
// Hand-rolled minimax-Taylor implementations; accurate to ~1e-6 over
// the witness ranges. Real range reduction + polynomial cores; do not
// secretly call into std::f32::sin/cos/exp/log — those route to libm.
// =====================================================================

const TWO_PI_F32: f32 = 6.2831853071795864769;
const PI_F32: f32 = 3.1415926535897932384;
const LN2_F32: f32 = 0.6931471805599453;

#[inline] fn aether_sin_core(r: f32) -> f32 {
    // Taylor for sin on r in [-π, π]; 11 terms keep error < ~3e-7.
    let r2 = r * r;
    let r3 = r * r2;
    let r5 = r3 * r2;
    let r7 = r5 * r2;
    let r9 = r7 * r2;
    let r11 = r9 * r2;
    r - r3 / 6.0
      + r5 / 120.0
      - r7 / 5040.0
      + r9 / 362880.0
      - r11 / 39916800.0
}

#[inline] fn aether_cos_core(r: f32) -> f32 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let r8 = r6 * r2;
    let r10 = r8 * r2;
    1.0 - r2 / 2.0
        + r4 / 24.0
        - r6 / 720.0
        + r8 / 40320.0
        - r10 / 3628800.0
}

/// Reduce x to [-π, π] then to [-π/2, π/2] with a quadrant flag, so the
/// Taylor cores stay in their high-accuracy region.
fn reduce_to_quarter(x: f32) -> (f32, bool) {
    let k = (x / TWO_PI_F32 + if x >= 0.0 { 0.5 } else { -0.5 }) as i32 as f32;
    let mut r = x - k * TWO_PI_F32;
    let mut flip = false;
    let half_pi = PI_F32 * 0.5;
    if r > half_pi { r = PI_F32 - r; flip = true; }
    else if r < -half_pi { r = -PI_F32 - r; flip = true; }
    (r, flip)
}

#[no_mangle] pub extern "C" fn aether_sin_f32(x: f32) -> f32 {
    let (r, _flip) = reduce_to_quarter(x);
    // sin(π - r) = sin(r); sin(-π - r) = -sin(r) — but sign already in r.
    aether_sin_core(r)
}

#[no_mangle] pub extern "C" fn aether_cos_f32(x: f32) -> f32 {
    let (r, flip) = reduce_to_quarter(x);
    let c = aether_cos_core(r);
    if flip { -c } else { c }
}

/// exp(x) via range reduction x = k*ln(2) + r, r in ~[-ln2/2, ln2/2],
/// then exp(r) Taylor (8 terms) and 2^k via direct bit pack.
#[no_mangle] pub extern "C" fn aether_exp_f32(x: f32) -> f32 {
    if x > 88.7 { return f32::INFINITY; }
    if x < -88.7 { return 0.0; }
    let k_real = x / LN2_F32;
    let k = (k_real + if k_real >= 0.0 { 0.5 } else { -0.5 }) as i32;
    let r = x - (k as f32) * LN2_F32;
    let r2 = r * r;
    let r3 = r * r2;
    let r4 = r2 * r2;
    let r5 = r * r4;
    let r6 = r4 * r2;
    let r7 = r3 * r4;
    // exp(r) Taylor: 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120 + r⁶/720 + r⁷/5040
    let exp_r = 1.0 + r + r2 / 2.0 + r3 / 6.0 + r4 / 24.0
                  + r5 / 120.0 + r6 / 720.0 + r7 / 5040.0;
    // 2^k: pack into IEEE-754 f32 exponent field (bias 127).
    let exp_bits = ((k + 127) as u32) << 23;
    let two_to_k = f32::from_bits(exp_bits);
    exp_r * two_to_k
}

/// log(x) for x > 0. Range-reduce via the f32 exponent, then minimax
/// Taylor on the mantissa.
#[no_mangle] pub extern "C" fn aether_log_f32(x: f32) -> f32 {
    if x <= 0.0 { return f32::NAN; }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127;
    // m in [1, 2): clear exponent, set to 127.
    let m_bits = (bits & 0x007F_FFFF) | (127u32 << 23);
    let m = f32::from_bits(m_bits);
    // Center: log(m) where m in [1, 2). Let u = (m-1)/(m+1), so m = (1+u)/(1-u).
    // Then log(m) = 2*(u + u³/3 + u⁵/5 + u⁷/7 + ...).
    let u = (m - 1.0) / (m + 1.0);
    let u2 = u * u;
    let u3 = u * u2;
    let u5 = u3 * u2;
    let u7 = u5 * u2;
    let u9 = u7 * u2;
    let log_m = 2.0 * (u + u3 / 3.0 + u5 / 5.0 + u7 / 7.0 + u9 / 9.0);
    (e as f32) * LN2_F32 + log_m
}

/// Print `<label>: <int>` to stdout. `label` is a 0-terminated C string —
/// pass a `StrLit` (which the compiler interns into `.rdata` and produces
/// as a pointer). Useful for debug tracing inside training loops.
#[no_mangle] pub unsafe extern "C" fn aether_print_kv_i64(label: *const u8, value: i64) -> c_int {
    if label.is_null() {
        println!("{}", value);
    } else {
        let mut len = 0usize;
        while *label.add(len) != 0 { len += 1; }
        let bytes = std::slice::from_raw_parts(label, len);
        let s = std::str::from_utf8(bytes).unwrap_or("<bad utf-8>");
        println!("{}: {}", s, value);
    }
    0
}

/// Print `<label>: <f32>` to stdout. Same convention as `aether_print_kv_i64`.
#[no_mangle] pub unsafe extern "C" fn aether_print_kv_f32(label: *const u8, value: f32) -> c_int {
    if label.is_null() {
        println!("{}", value);
    } else {
        let mut len = 0usize;
        while *label.add(len) != 0 { len += 1; }
        let bytes = std::slice::from_raw_parts(label, len);
        let s = std::str::from_utf8(bytes).unwrap_or("<bad utf-8>");
        println!("{}: {}", s, value);
    }
    0
}

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

    #[test]
    fn tcp_listen_close_roundtrip() {
        // Bind ephemeral, read back the port, close. Then prove
        // closing twice is rejected and bogus handles are rejected.
        unsafe {
            let l = aether_tcp_listen(0);
            assert!(l >= 0, "listen returned {}", l);
            let port = aether_tcp_listener_port(l);
            assert!((1..=65535).contains(&port), "bound port {} out of range", port);
            assert_eq!(aether_tcp_close(l), 0);
            assert_eq!(aether_tcp_close(l), -1);  // double-close rejected
            assert_eq!(aether_tcp_close(99_999), -1);
            assert_eq!(aether_tcp_stream_close(99_999), -1);
        }
    }

    #[test]
    fn tcp_send_recv_loopback() {
        // Witness the full surface: spawn a listener, connect a client
        // from a thread, accept, send "hi" both ways, close both ends.
        unsafe {
            let listener = aether_tcp_listen(0);
            assert!(listener >= 0);
            let port = aether_tcp_listener_port(listener);
            assert!(port > 0);

            let t = std::thread::spawn(move || {
                // Client thread. Note: this thread's TCP_STREAMS slot
                // accesses are technically a data race vs. the main
                // thread's accept_one which also touches TCP_STREAMS.
                // For the witness we serialise: client connects only
                // after main thread is already blocked in accept().
                std::thread::sleep(std::time::Duration::from_millis(50));
                let s = std::net::TcpStream::connect(("127.0.0.1", port as u16))
                    .expect("client connect");
                use std::io::{Read, Write};
                let mut s = s;
                s.write_all(b"ping").unwrap();
                let mut buf = [0u8; 4];
                s.read_exact(&mut buf).unwrap();
                assert_eq!(&buf, b"pong");
            });

            let server_stream = aether_tcp_accept_one(listener);
            assert!(server_stream >= 0, "accept returned {}", server_stream);

            // Recv "ping" from the client.
            let recv_buf = [0u8; 4];
            let got = aether_tcp_recv(server_stream, recv_buf.as_ptr() as i64, 4);
            assert_eq!(got, 4);
            assert_eq!(&recv_buf, b"ping");

            // Send "pong" back.
            let send_buf = b"pong";
            let sent = aether_tcp_send(server_stream, send_buf.as_ptr() as i64, 4);
            assert_eq!(sent, 4);

            assert_eq!(aether_tcp_stream_close(server_stream), 0);
            assert_eq!(aether_tcp_close(listener), 0);

            t.join().unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// P10.7 — PGO instrumentation surface
// ---------------------------------------------------------------------------
// Profile-collection primitives: branch frequency counters and call-site
// counters. Backing store is a Vec behind a static UnsafeCell, mirroring the
// MATMUL_CACHE / TAPE pattern. Linear scan on lookup is fine — this is
// profiling infrastructure, not a hot path. Single-threaded use only (no
// Mutex; the rest of aether_rt is single-threaded by convention).
//
// Wire-up for feedback-directed inlining + branch hints lives in the
// optimiser (P10.1 SSA, currently blocked). Here we ship only the data
// plumbing so .aether code can record + query counters today, and so a
// future optimiser can read a serialised profile.

#[repr(C)]
struct PgoBranch {
    site_id: i64,
    taken: i64,
    total: i64,
}

struct PgoBranchCell(UnsafeCell<Vec<PgoBranch>>);
unsafe impl Sync for PgoBranchCell {}
static PGO_BRANCHES: PgoBranchCell = PgoBranchCell(UnsafeCell::new(Vec::new()));

struct PgoCallCell(UnsafeCell<Vec<(i64, i64)>>);
unsafe impl Sync for PgoCallCell {}
static PGO_CALLS: PgoCallCell = PgoCallCell(UnsafeCell::new(Vec::new()));

#[no_mangle]
pub extern "C" fn aether_pgo_record_branch(site_id: i64, taken: i64) -> i32 {
    unsafe {
        let v = &mut *PGO_BRANCHES.0.get();
        for entry in v.iter_mut() {
            if entry.site_id == site_id {
                entry.total += 1;
                if taken != 0 { entry.taken += 1; }
                return 0;
            }
        }
        v.push(PgoBranch {
            site_id,
            taken: if taken != 0 { 1 } else { 0 },
            total: 1,
        });
        0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_record_call(callsite_id: i64) -> i32 {
    unsafe {
        let v = &mut *PGO_CALLS.0.get();
        for entry in v.iter_mut() {
            if entry.0 == callsite_id {
                entry.1 += 1;
                return 0;
            }
        }
        v.push((callsite_id, 1));
        0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_branch_freq(site_id: i64) -> f32 {
    unsafe {
        let v = &*PGO_BRANCHES.0.get();
        for entry in v.iter() {
            if entry.site_id == site_id {
                if entry.total == 0 { return 0.0; }
                return entry.taken as f32 / entry.total as f32;
            }
        }
        0.0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_call_count(callsite_id: i64) -> i64 {
    unsafe {
        let v = &*PGO_CALLS.0.get();
        for entry in v.iter() {
            if entry.0 == callsite_id {
                return entry.1;
            }
        }
        0
    }
}

#[no_mangle]
pub extern "C" fn aether_pgo_reset() -> i32 {
    unsafe {
        (&mut *PGO_BRANCHES.0.get()).clear();
        (&mut *PGO_CALLS.0.get()).clear();
    }
    0
}

#[no_mangle]
pub extern "C" fn aether_pgo_dump() -> i32 {
    unsafe {
        let bv = &*PGO_BRANCHES.0.get();
        for e in bv.iter() {
            println!("site={} taken={} total={}", e.site_id, e.taken, e.total);
        }
        let cv = &*PGO_CALLS.0.get();
        for e in cv.iter() {
            println!("call={} count={}", e.0, e.1);
        }
    }
    0
}

#[cfg(test)]
mod pgo_tests {
    use super::*;

    #[test]
    fn pgo_branch_and_call_roundtrip() {
        aether_pgo_reset();
        // site 1: 3 of 5 taken
        aether_pgo_record_branch(1, 1);
        aether_pgo_record_branch(1, 0);
        aether_pgo_record_branch(1, 1);
        aether_pgo_record_branch(1, 1);
        aether_pgo_record_branch(1, 0);
        // site 2: 0 of 2 taken
        aether_pgo_record_branch(2, 0);
        aether_pgo_record_branch(2, 0);

        let f1 = aether_pgo_branch_freq(1);
        let f2 = aether_pgo_branch_freq(2);
        let f_missing = aether_pgo_branch_freq(99);
        assert!((f1 - 0.6).abs() < 1e-6, "expected 0.6, got {}", f1);
        assert!(f2.abs() < 1e-6, "expected 0.0, got {}", f2);
        assert!(f_missing.abs() < 1e-6);

        aether_pgo_record_call(10);
        aether_pgo_record_call(10);
        aether_pgo_record_call(10);
        aether_pgo_record_call(20);
        assert_eq!(aether_pgo_call_count(10), 3);
        assert_eq!(aether_pgo_call_count(20), 1);
        assert_eq!(aether_pgo_call_count(999), 0);

        aether_pgo_reset();
        assert!(aether_pgo_branch_freq(1).abs() < 1e-6);
        assert_eq!(aether_pgo_call_count(10), 0);
    }
}

// =====================================================================
// P6.7 — heap-allocated stdlib types.
//
// The language doesn't have generics-over-T yet (P6.1 / P6.2 still open),
// so this pass ships CONCRETE Vec<i64> and String (== Vec<u8>) primitives
// rather than a fully generic Vec<T>. Every Vec-of-T variant the user
// needs gets its own monomorphic op set behind the runtime ABI; once
// generics land, these become a single template that lowers to the same
// symbols.
//
// P6.7 partial — Box/HashMap/BTreeMap need generics (P6.1).
// =====================================================================

/// Resize a buffer previously returned by `aether_alloc_bytes`. Returns
/// the new pointer (may equal old). On growth, old contents copy into
/// the new buffer and the tail is zero-filled. On shrink, the buffer
/// is truncated. `old_n` MUST be the exact length of the existing buffer.
/// Returns 0 on null input or invalid sizes.
#[no_mangle] pub unsafe extern "C" fn aether_realloc_bytes(p: i64, old_n: i64, new_n: i64) -> i64 {
    if new_n <= 0 { return 0; }
    if p == 0 || old_n <= 0 {
        return aether_alloc_bytes(new_n);
    }
    if old_n == new_n { return p; }
    let new_p = aether_alloc_bytes(new_n);
    if new_p == 0 { return 0; }
    let copy = old_n.min(new_n) as usize;
    std::ptr::copy_nonoverlapping(p as *const u8, new_p as *mut u8, copy);
    aether_free_bytes(p, old_n);
    new_p
}

// ---- Vec<i64> handle table ------------------------------------------------

struct VecI64 {
    ptr: *mut i64,
    len: usize,
    cap: usize,
}

struct VecI64Cell(UnsafeCell<Vec<Option<Box<VecI64>>>>);
unsafe impl Sync for VecI64Cell {}
static VEC_I64_TABLE: VecI64Cell = VecI64Cell(UnsafeCell::new(Vec::new()));

unsafe fn vec_i64_table() -> &'static mut Vec<Option<Box<VecI64>>> {
    &mut *VEC_I64_TABLE.0.get()
}

unsafe fn vec_i64_alloc_buf(cap: usize) -> *mut i64 {
    if cap == 0 { return std::ptr::null_mut(); }
    let bytes = cap.checked_mul(std::mem::size_of::<i64>()).expect("vec i64 cap overflow");
    let p = aether_alloc_bytes(bytes as i64);
    p as *mut i64
}

unsafe fn vec_i64_free_buf(ptr: *mut i64, cap: usize) {
    if ptr.is_null() || cap == 0 { return; }
    let bytes = cap * std::mem::size_of::<i64>();
    aether_free_bytes(ptr as i64, bytes as i64);
}

/// Allocate a fresh empty `Vec<i64>`. Returns a non-negative handle.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_new() -> i64 {
    let v = Box::new(VecI64 { ptr: std::ptr::null_mut(), len: 0, cap: 0 });
    let tbl = vec_i64_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(v); return i as i64; }
    }
    tbl.push(Some(v));
    (tbl.len() - 1) as i64
}

/// Push `value` onto the end of the Vec. Capacity-doubling growth.
/// Returns 0 on success, -1 on invalid handle, -2 on OOM.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_push(handle: i64, value: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = vec_i64_table();
    let idx = handle as usize;
    if idx >= tbl.len() { return -1; }
    let v = match tbl[idx].as_mut() { Some(v) => v, None => return -1 };
    if v.len == v.cap {
        let new_cap = if v.cap == 0 { 4 } else { v.cap * 2 };
        let new_ptr = vec_i64_alloc_buf(new_cap);
        if new_ptr.is_null() { return -2; }
        if v.len > 0 {
            std::ptr::copy_nonoverlapping(v.ptr, new_ptr, v.len);
        }
        vec_i64_free_buf(v.ptr, v.cap);
        v.ptr = new_ptr;
        v.cap = new_cap;
    }
    *v.ptr.add(v.len) = value;
    v.len += 1;
    0
}

/// Read element at `idx`. Returns 0 on out-of-range / invalid handle.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_get(handle: i64, idx: i64) -> i64 {
    if handle < 0 || idx < 0 { return 0; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    let v = match tbl[h].as_ref() { Some(v) => v, None => return 0 };
    let i = idx as usize;
    if i >= v.len { return 0; }
    *v.ptr.add(i)
}

/// Overwrite element at `idx`. Returns 0 on success, -1 invalid handle,
/// -2 out-of-range.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_set(handle: i64, idx: i64, value: i64) -> i32 {
    if handle < 0 || idx < 0 { return -1; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let v = match tbl[h].as_mut() { Some(v) => v, None => return -1 };
    let i = idx as usize;
    if i >= v.len { return -2; }
    *v.ptr.add(i) = value;
    0
}

/// Number of elements currently in the Vec.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(v) => v.len as i64, None => 0 }
}

/// Free the buffer + release the handle slot. Idempotent.
#[no_mangle] pub unsafe extern "C" fn aether_vec_i64_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = vec_i64_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    if let Some(mut v) = tbl[h].take() {
        vec_i64_free_buf(v.ptr, v.cap);
        v.ptr = std::ptr::null_mut();
        v.cap = 0;
        v.len = 0;
    }
    0
}

// ---- String (UTF-8 owned, == Vec<u8>) ------------------------------------

struct AeString {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}

struct AeStringCell(UnsafeCell<Vec<Option<Box<AeString>>>>);
unsafe impl Sync for AeStringCell {}
static STRING_TABLE: AeStringCell = AeStringCell(UnsafeCell::new(Vec::new()));

unsafe fn string_table() -> &'static mut Vec<Option<Box<AeString>>> {
    &mut *STRING_TABLE.0.get()
}

/// Allocate a fresh empty `String`. Returns a non-negative handle.
#[no_mangle] pub unsafe extern "C" fn aether_string_new() -> i64 {
    let s = Box::new(AeString { ptr: std::ptr::null_mut(), len: 0, cap: 0 });
    let tbl = string_table();
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(s); return i as i64; }
    }
    tbl.push(Some(s));
    (tbl.len() - 1) as i64
}

/// Append low-byte of `b` to the string. Capacity-doubling growth.
#[no_mangle] pub unsafe extern "C" fn aether_string_push_byte(handle: i64, b: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = string_table();
    let idx = handle as usize;
    if idx >= tbl.len() { return -1; }
    let s = match tbl[idx].as_mut() { Some(s) => s, None => return -1 };
    if s.len == s.cap {
        let new_cap = if s.cap == 0 { 8 } else { s.cap * 2 };
        let new_ptr = aether_alloc_bytes(new_cap as i64) as *mut u8;
        if new_ptr.is_null() { return -2; }
        if s.len > 0 {
            std::ptr::copy_nonoverlapping(s.ptr, new_ptr, s.len);
        }
        if !s.ptr.is_null() && s.cap > 0 {
            aether_free_bytes(s.ptr as i64, s.cap as i64);
        }
        s.ptr = new_ptr;
        s.cap = new_cap;
    }
    *s.ptr.add(s.len) = (b & 0xFF) as u8;
    s.len += 1;
    0
}

/// Length in bytes of the string.
#[no_mangle] pub unsafe extern "C" fn aether_string_len(handle: i64) -> i64 {
    if handle < 0 { return 0; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return 0; }
    match tbl[h].as_ref() { Some(s) => s.len as i64, None => 0 }
}

/// Read byte at `idx` (returned as 0..=255 in i64). -1 on out-of-range.
#[no_mangle] pub unsafe extern "C" fn aether_string_byte_at(handle: i64, idx: i64) -> i64 {
    if handle < 0 || idx < 0 { return -1; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    let s = match tbl[h].as_ref() { Some(s) => s, None => return -1 };
    let i = idx as usize;
    if i >= s.len { return -1; }
    *s.ptr.add(i) as i64
}

/// Free the buffer + release the handle slot. Idempotent.
#[no_mangle] pub unsafe extern "C" fn aether_string_free(handle: i64) -> i32 {
    if handle < 0 { return -1; }
    let tbl = string_table();
    let h = handle as usize;
    if h >= tbl.len() { return -1; }
    if let Some(mut s) = tbl[h].take() {
        if !s.ptr.is_null() && s.cap > 0 {
            aether_free_bytes(s.ptr as i64, s.cap as i64);
        }
        s.ptr = std::ptr::null_mut();
        s.cap = 0;
        s.len = 0;
    }
    0
}

#[cfg(test)]
mod heap_stdlib_tests {
    use super::*;

    #[test]
    fn realloc_grow_and_shrink() {
        unsafe {
            let p = aether_alloc_bytes(4);
            aether_byte_set(p, 0, 1);
            aether_byte_set(p, 1, 2);
            aether_byte_set(p, 2, 3);
            aether_byte_set(p, 3, 4);
            let q = aether_realloc_bytes(p, 4, 8);
            assert_ne!(q, 0);
            assert_eq!(aether_byte_at(q, 0), 1);
            assert_eq!(aether_byte_at(q, 3), 4);
            // Tail zero-init from aether_alloc_bytes.
            assert_eq!(aether_byte_at(q, 4), 0);
            // Shrink back to 2.
            let r = aether_realloc_bytes(q, 8, 2);
            assert_ne!(r, 0);
            assert_eq!(aether_byte_at(r, 0), 1);
            assert_eq!(aether_byte_at(r, 1), 2);
            aether_free_bytes(r, 2);
        }
    }

    #[test]
    fn vec_i64_push_get_len() {
        unsafe {
            let h = aether_vec_i64_new();
            assert!(h >= 0);
            for i in 0..1000i64 {
                assert_eq!(aether_vec_i64_push(h, i * 2), 0);
            }
            assert_eq!(aether_vec_i64_len(h), 1000);
            assert_eq!(aether_vec_i64_get(h, 42), 84);
            assert_eq!(aether_vec_i64_set(h, 42, 999), 0);
            assert_eq!(aether_vec_i64_get(h, 42), 999);
            // Out-of-range reads return 0.
            assert_eq!(aether_vec_i64_get(h, 9999), 0);
            assert_eq!(aether_vec_i64_set(h, 9999, 1), -2);
            assert_eq!(aether_vec_i64_free(h), 0);
        }
    }

    #[test]
    fn string_push_byte_roundtrip() {
        unsafe {
            let s = aether_string_new();
            for &b in b"Hello, world!" {
                assert_eq!(aether_string_push_byte(s, b as i64), 0);
            }
            assert_eq!(aether_string_len(s), 13);
            assert_eq!(aether_string_byte_at(s, 0), b'H' as i64);
            assert_eq!(aether_string_byte_at(s, 12), b'!' as i64);
            assert_eq!(aether_string_byte_at(s, 13), -1);
            assert_eq!(aether_string_free(s), 0);
        }
    }
}

// ============================================================================
// P6.8 — Iterator trait + adapters (concrete-iterator subset)
// ----------------------------------------------------------------------------
// Traits + generics are not shipped yet (P6.2 / P6.6 are blocked / partial),
// so we ship a witness-able subset built around a concrete enum-typed iterator
// over Vec<i64>. Adapters return new opaque handles that wrap an inner handle.
// next() dispatches on the variant.
//
// Variants:
//   Source { vec, cursor }      yields vec[cursor], cursor++
//   MapDouble { inner }         yields 2 * inner.next()
//   FilterPos { inner }         skips inner.next() values that are <= 0
//   Take { inner, remaining }   yields up to `remaining` values
// ============================================================================

enum IterNode {
    Source { vec: i64, cursor: usize },
    MapDouble { inner: i64 },
    FilterPos { inner: i64 },
    Take { inner: i64, remaining: i64 },
}

struct IterCell(UnsafeCell<Vec<Option<Box<IterNode>>>>);
unsafe impl Sync for IterCell {}
static ITER_TABLE: IterCell = IterCell(UnsafeCell::new(Vec::new()));

unsafe fn iter_table() -> &'static mut Vec<Option<Box<IterNode>>> {
    &mut *ITER_TABLE.0.get()
}

unsafe fn iter_install(node: IterNode) -> i64 {
    let tbl = iter_table();
    let boxed = Some(Box::new(node));
    for (i, slot) in tbl.iter_mut().enumerate() {
        if slot.is_none() {
            *slot = boxed;
            return i as i64;
        }
    }
    tbl.push(boxed);
    (tbl.len() - 1) as i64
}

unsafe fn iter_next_inner(handle: i64) -> Option<i64> {
    if handle < 0 { return None; }
    let tbl = iter_table();
    let h = handle as usize;
    if h >= tbl.len() { return None; }
    let node_ptr: *mut IterNode = match tbl[h].as_mut() {
        Some(b) => &mut **b as *mut IterNode,
        None => return None,
    };
    match &mut *node_ptr {
        IterNode::Source { vec, cursor } => {
            let len = aether_vec_i64_len(*vec);
            if (*cursor as i64) >= len { return None; }
            let v = aether_vec_i64_get(*vec, *cursor as i64);
            *cursor += 1;
            Some(v)
        }
        IterNode::MapDouble { inner } => {
            let inner_h = *inner;
            iter_next_inner(inner_h).map(|v| v * 2)
        }
        IterNode::FilterPos { inner } => {
            let inner_h = *inner;
            loop {
                match iter_next_inner(inner_h) {
                    Some(v) if v > 0 => return Some(v),
                    Some(_) => continue,
                    None => return None,
                }
            }
        }
        IterNode::Take { inner, remaining } => {
            if *remaining <= 0 { return None; }
            let inner_h = *inner;
            match iter_next_inner(inner_h) {
                Some(v) => { *remaining -= 1; Some(v) }
                None => { *remaining = 0; None }
            }
        }
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_new(vec_handle: i64) -> i64 {
    if vec_handle < 0 { return -1; }
    iter_install(IterNode::Source { vec: vec_handle, cursor: 0 })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_next(iter: i64, out_value: i64) -> i64 {
    if iter < 0 { return -1; }
    if out_value == 0 { return -1; }
    match iter_next_inner(iter) {
        Some(v) => {
            *(out_value as *mut i64) = v;
            1
        }
        None => 0,
    }
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_map_double(iter: i64) -> i64 {
    if iter < 0 { return -1; }
    iter_install(IterNode::MapDouble { inner: iter })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_filter_positive(iter: i64) -> i64 {
    if iter < 0 { return -1; }
    iter_install(IterNode::FilterPos { inner: iter })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_take(iter: i64, n: i64) -> i64 {
    if iter < 0 { return -1; }
    let r = if n < 0 { 0 } else { n };
    iter_install(IterNode::Take { inner: iter, remaining: r })
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_fold_sum(iter: i64) -> i64 {
    if iter < 0 { return 0; }
    let mut acc: i64 = 0;
    while let Some(v) = iter_next_inner(iter) {
        acc = acc.wrapping_add(v);
    }
    acc
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_collect(iter: i64) -> i64 {
    if iter < 0 { return -1; }
    let v = aether_vec_i64_new();
    while let Some(x) = iter_next_inner(iter) {
        if aether_vec_i64_push(v, x) != 0 {
            return -2;
        }
    }
    v
}

#[no_mangle] pub unsafe extern "C" fn aether_iter_vec_i64_free(iter: i64) -> i32 {
    if iter < 0 { return -1; }
    let tbl = iter_table();
    let h = iter as usize;
    if h >= tbl.len() { return -1; }
    tbl[h] = None;
    0
}

#[cfg(test)]
mod iter_tests {
    use super::*;

    #[test]
    fn chain_filter_map_take_fold() {
        unsafe {
            let v = aether_vec_i64_new();
            for i in -3..=10i64 {
                aether_vec_i64_push(v, i);
            }
            let s = aether_iter_vec_i64_new(v);
            let f = aether_iter_vec_i64_filter_positive(s);
            let m = aether_iter_vec_i64_map_double(f);
            let t = aether_iter_vec_i64_take(m, 3);
            assert_eq!(aether_iter_vec_i64_fold_sum(t), 12);
            aether_iter_vec_i64_free(t);
            aether_iter_vec_i64_free(m);
            aether_iter_vec_i64_free(f);
            aether_iter_vec_i64_free(s);
            aether_vec_i64_free(v);
        }
    }

    #[test]
    fn collect_round_trip() {
        unsafe {
            let v = aether_vec_i64_new();
            for i in 1..=5i64 { aether_vec_i64_push(v, i); }
            let it = aether_iter_vec_i64_new(v);
            let m = aether_iter_vec_i64_map_double(it);
            let out = aether_iter_vec_i64_collect(m);
            assert_eq!(aether_vec_i64_len(out), 5);
            assert_eq!(aether_vec_i64_get(out, 0), 2);
            assert_eq!(aether_vec_i64_get(out, 4), 10);
            aether_vec_i64_free(out);
            aether_iter_vec_i64_free(m);
            aether_iter_vec_i64_free(it);
            aether_vec_i64_free(v);
        }
    }

    #[test]
    fn next_out_pointer_protocol() {
        unsafe {
            let v = aether_vec_i64_new();
            aether_vec_i64_push(v, 7);
            aether_vec_i64_push(v, 11);
            let it = aether_iter_vec_i64_new(v);
            let mut out: i64 = 0;
            assert_eq!(aether_iter_vec_i64_next(it, &mut out as *mut i64 as i64), 1);
            assert_eq!(out, 7);
            assert_eq!(aether_iter_vec_i64_next(it, &mut out as *mut i64 as i64), 1);
            assert_eq!(out, 11);
            assert_eq!(aether_iter_vec_i64_next(it, &mut out as *mut i64 as i64), 0);
            aether_iter_vec_i64_free(it);
            aether_vec_i64_free(v);
        }
    }
}
