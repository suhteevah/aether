//! libaether_rt — runtime intrinsics that aether-emitted code links against.
//!
//! Phase 0/1 stubs. NCCL/MPI/RDMA backends slot in here in Phase 2 by replacing
//! the body of `aether_dist_all_reduce` with a dispatch on AETHER_DIST_BACKEND.
//!
//! All entry points are `#[no_mangle] extern "C"` so the LLVM IR emitted by
//! `compiler/src/codegen/llvm/mod.rs` links cleanly without aliasing.

use std::cell::RefCell;
use std::os::raw::{c_int, c_void};

pub mod ops;

#[derive(Default)]
struct Tape {
    entries: Vec<*const c_void>,
    closed: bool,
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::default());
}

#[no_mangle]
pub extern "C" fn aether_autodiff_init(_tape: *mut c_void) {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        t.entries.clear();
        t.closed = false;
    });
}

#[no_mangle]
pub extern "C" fn aether_autodiff_push(_tape: *mut c_void, value: *const c_void) {
    TAPE.with(|t| t.borrow_mut().entries.push(value));
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
pub extern "C" fn aether_autodiff_reverse(_tape: *mut c_void) {
    TAPE.with(|t| {
        let mut t = t.borrow_mut();
        t.closed = true;
    });
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
pub extern "C" fn aether_rt_self_check() -> c_int {
    let mut entries = 0i32;
    TAPE.with(|t| entries = t.borrow().entries.len() as i32);
    entries
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn tape_lifecycle() {
        aether_autodiff_init(ptr::null_mut());
        aether_autodiff_push(ptr::null_mut(), ptr::null());
        aether_autodiff_push(ptr::null_mut(), ptr::null());
        assert_eq!(aether_rt_self_check(), 2);
        aether_autodiff_reverse(ptr::null_mut());
    }

    #[test]
    fn all_reduce_is_safe() {
        aether_dist_all_reduce(ptr::null_mut(), 8, DistBackend::Nccl as c_int);
    }
}
