//! Slim no-std-flavoured runtime DLL for `aetherc --emit=pe-bin`.
//!
//! Same C-ABI symbols as `aether_rt`, but built so the cdylib's DllMain
//! doesn't need the Rust-std init chain that reaches `bcryptprimitives.dll`
//! for HashMap-hasher seeding. That chain AVs when the DLL loads early in
//! process init (bcryptprimitives' DllMain hasn't run yet), so this crate
//! avoids:
//!   * `std::collections::HashMap` (and its hasher init)
//!   * `thread_local!` (TLS callbacks)
//!   * `std::io::stdout()` (lazy stdio init)
//!   * `Vec` for the heap-alloc helpers (which would link in Rust's allocator
//!     init paths). We talk to `kernel32!HeapAlloc` directly instead.
//!   * `std::panic` formatting (we abort on errors via UD2-style traps)
//!
//! Only the symbols actually exercised by `tests/runtime/*.aether` are
//! implemented. Adding more is mechanical — most of the math is `core::f32`
//! / `core::f64` operations on raw slices.

#![no_std]

use core::ffi::c_void;

type CInt = i32;

// ----- Win32 imports we need -----
#[link(name = "kernel32")]
extern "system" {
    fn GetProcessHeap() -> *mut c_void;
    fn HeapAlloc(heap: *mut c_void, flags: u32, n: usize) -> *mut c_void;
    fn HeapFree(heap: *mut c_void, flags: u32, p: *mut c_void) -> i32;
    fn GetStdHandle(n: u32) -> *mut c_void;
    fn WriteFile(h: *mut c_void, b: *const c_void, n: u32,
                 written: *mut u32, overlapped: *mut c_void) -> i32;
}

const HEAP_ZERO_MEMORY: u32 = 0x8;
const STD_OUTPUT_HANDLE: u32 = 0xFFFF_FFF5; // (DWORD)-11

#[inline]
unsafe fn alloc_zeroed(n_bytes: usize) -> *mut u8 {
    HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, n_bytes) as *mut u8
}

#[inline]
unsafe fn free(p: *mut u8) {
    if !p.is_null() { HeapFree(GetProcessHeap(), 0, p as *mut c_void); }
}

// ----- autodiff tape (single-threaded, no TLS) -----
struct Tape { entries: usize, closed: bool }
static mut TAPE: Tape = Tape { entries: 0, closed: false };

#[no_mangle] pub unsafe extern "C" fn aether_autodiff_init(_tape: *mut c_void) {
    TAPE.entries = 0; TAPE.closed = false;
}
#[no_mangle] pub unsafe extern "C" fn aether_autodiff_push(_tape: *mut c_void, _value: *const c_void) {
    TAPE.entries += 1;
}
#[no_mangle] pub unsafe extern "C" fn aether_autodiff_accumulate(_tape: *mut c_void, _grad: *const c_void) {}
#[no_mangle] pub unsafe extern "C" fn aether_autodiff_partial(
    _tape: *mut c_void, _dst: CInt, _op: CInt, _src: CInt) {}
#[no_mangle] pub unsafe extern "C" fn aether_autodiff_reverse(_tape: *mut c_void) {
    TAPE.closed = true;
}
#[no_mangle] pub unsafe extern "C" fn aether_rt_self_check() -> CInt {
    TAPE.entries as CInt
}

// ----- allocator helpers -----
#[no_mangle] pub unsafe extern "C" fn aether_alloc_f32(n: CInt) -> i64 {
    if n <= 0 { return 0; }
    alloc_zeroed((n as usize) * 4) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_alloc_i32(n: CInt) -> i64 {
    if n <= 0 { return 0; }
    alloc_zeroed((n as usize) * 4) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_free_f32(p: i64, _n: CInt) -> CInt {
    free(p as *mut u8);
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_free_i32(p: i64, _n: CInt) -> CInt {
    free(p as *mut u8);
    0
}

// ----- deterministic RNG (SplitMix64) -----
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[no_mangle]
pub unsafe extern "C" fn aether_init_normal_f32(p: i64, n: CInt, scale: f32, seed: i64) -> CInt {
    if p == 0 || n <= 0 { return 0; }
    let n = n as usize;
    let buf = core::slice::from_raw_parts_mut(p as *mut f32, n);
    let mut state = seed as u64;
    // Uniform-in-[-1,1] init scaled to match a Gaussian's stddev. Skipping
    // Box-Muller because every f64 entry in `libm` (and a number of f32
    // ones — sinf/cosf/powf) has a SAVE_XMM6+ prologue that AVs when called
    // through our self-hosted PE writer's import resolution. Their
    // `movaps %xmm6, disp(%rsp)` lands on a misaligned address; the root
    // cause is in libm-on-windows-gnu prologue codegen, not in the PE
    // writer or Aether's emitted call sites (verified: sub-prologues that
    // only need 8-byte spills work fine). For init, scaled uniform is a
    // perfectly serviceable substitute and gets the loss curve moving.
    // sqrt(3) ≈ 1.732 is the variance match factor for U(-1,1) → N(0,1).
    for slot in buf.iter_mut() {
        let raw = splitmix64(&mut state);
        let u = ((raw as i64) as f32) / (i64::MAX as f32);
        *slot = u * scale * 1.732_050_8;
    }
    0
}

#[no_mangle]
pub unsafe extern "C" fn aether_fill_labels_i32(p: i64, n: CInt, classes: CInt, seed: i64) -> CInt {
    if p == 0 || n <= 0 || classes <= 0 { return 0; }
    let n = n as usize;
    let buf = core::slice::from_raw_parts_mut(p as *mut i32, n);
    let c = classes as u64;
    let mut state = seed as u64;
    for slot in buf.iter_mut() {
        *slot = (splitmix64(&mut state) % c) as i32;
    }
    0
}

#[no_mangle] pub unsafe extern "C" fn aether_load_f32(p: i64, i: CInt) -> f32 {
    if p == 0 { return 0.0; }
    *((p as *const f32).add(i as usize))
}

#[no_mangle] pub unsafe extern "C" fn aether_store_f32(p: i64, i: CInt, v: f32) {
    if p == 0 { return; }
    *((p as *mut f32).add(i as usize)) = v;
}

// ----- core ops -----
// y[m,n] = a[m,k] @ b[k,n]
#[no_mangle]
pub unsafe extern "C" fn aether_op_matmul_f32(
    a: i64, b: i64, out: i64, m: CInt, k: CInt, n: CInt,
) -> CInt {
    let m = m as usize; let k = k as usize; let n = n as usize;
    let a = core::slice::from_raw_parts(a as *const f32, m * k);
    let b = core::slice::from_raw_parts(b as *const f32, k * n);
    let o = core::slice::from_raw_parts_mut(out as *mut f32, m * n);
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for kk in 0..k { s += a[i * k + kk] * b[kk * n + j]; }
            o[i * n + j] = s;
        }
    }
    0
}

// db[k,n] = a[m,k]^T @ dy[m,n]
#[no_mangle]
pub unsafe extern "C" fn aether_op_matmul_backward_rhs_f32(
    a: i64, dy: i64, db: i64, m: CInt, k: CInt, n: CInt,
) -> CInt {
    let m = m as usize; let k = k as usize; let n = n as usize;
    let a  = core::slice::from_raw_parts(a as *const f32, m * k);
    let dy = core::slice::from_raw_parts(dy as *const f32, m * n);
    let db = core::slice::from_raw_parts_mut(db as *mut f32, k * n);
    for kk in 0..k {
        for j in 0..n {
            let mut s = 0.0f32;
            for i in 0..m { s += a[i * k + kk] * dy[i * n + j]; }
            db[kk * n + j] = s;
        }
    }
    0
}

// Mean cross-entropy. Saves softmax probs to probs_out.
#[no_mangle]
pub unsafe extern "C" fn aether_op_cross_entropy_f32(
    logits: i64, labels: i64, probs_out: i64, b: CInt, v: CInt,
) -> f32 {
    let b = b as usize; let v = v as usize;
    let logits = core::slice::from_raw_parts(logits as *const f32, b * v);
    let labels = core::slice::from_raw_parts(labels as *const i32, b);
    let probs  = core::slice::from_raw_parts_mut(probs_out as *mut f32, b * v);
    let mut total = 0.0f64;
    for i in 0..b {
        let off = i * v;
        let mut mx = logits[off];
        for j in 1..v { if logits[off + j] > mx { mx = logits[off + j]; } }
        let mut sum = 0.0f32;
        for j in 0..v {
            let e = libm_expf(logits[off + j] - mx);
            probs[off + j] = e; sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..v { probs[off + j] *= inv; }
        let lab = labels[i] as usize;
        let p = probs[off + lab].max(1e-12);
        total += -(libm_lnf(p) as f64);
    }
    (total / b as f64) as f32
}

// dlogits = (probs - one_hot(labels)) / B
#[no_mangle]
pub unsafe extern "C" fn aether_op_cross_entropy_backward_f32(
    probs: i64, labels: i64, dlogits: i64, b: CInt, v: CInt,
) -> CInt {
    let b = b as usize; let v = v as usize;
    let probs   = core::slice::from_raw_parts(probs as *const f32, b * v);
    let labels  = core::slice::from_raw_parts(labels as *const i32, b);
    let dlogits = core::slice::from_raw_parts_mut(dlogits as *mut f32, b * v);
    let inv_b = 1.0 / (b as f32);
    for i in 0..b {
        let off = i * v;
        for j in 0..v { dlogits[off + j] = probs[off + j] * inv_b; }
        let lab = labels[i] as usize;
        dlogits[off + lab] -= inv_b;
    }
    0
}

// AdamW with bias correction.
#[no_mangle]
pub unsafe extern "C" fn aether_op_adamw_step_f32(
    param: i64, grad: i64, m_state: i64, v_state: i64,
    lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32,
    step: i64, n: CInt,
) -> CInt {
    let n = n as usize;
    let p = core::slice::from_raw_parts_mut(param as *mut f32, n);
    let g = core::slice::from_raw_parts(grad as *const f32, n);
    let m = core::slice::from_raw_parts_mut(m_state as *mut f32, n);
    let v = core::slice::from_raw_parts_mut(v_state as *mut f32, n);
    // Replace `libm::powf` (AVs in its SAVE_XMM prologue under our PE) with
    // exponentiation-by-squaring on integer exponents — `step` is always a
    // positive int. Replace `libm::sqrtf` with the hardware SQRTSS via
    // inline asm — single instruction, no prologue, no alignment trap.
    let bc1 = 1.0 - pow_int(beta1, step as u64);
    let bc2 = 1.0 - pow_int(beta2, step as u64);
    for i in 0..n {
        m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
        let mh = m[i] / bc1;
        let vh = v[i] / bc2;
        p[i] -= lr * (mh / (sqrtss(vh) + eps) + wd * p[i]);
    }
    0
}

/// `base ** n` for non-negative integer `n` via exponentiation-by-squaring.
fn pow_int(base: f32, mut n: u64) -> f32 {
    let mut r = 1.0f32;
    let mut b = base;
    while n > 0 {
        if n & 1 == 1 { r *= b; }
        b *= b;
        n >>= 1;
    }
    r
}

/// f32 sqrt via the SSE2 `sqrtss` instruction directly — bypasses libm's
/// AV-prone software-fallback trampoline.
#[inline]
fn sqrtss(x: f32) -> f32 {
    let r: f32;
    unsafe {
        core::arch::asm!(
            "sqrtss {tmp}, {tmp}",
            tmp = inout(xmm_reg) x => r,
            options(pure, nomem, nostack),
        );
    }
    r
}

// ----- print loss curve via direct WriteFile -----
#[no_mangle]
pub extern "C" fn aether_print_loss(step: CInt, loss: f32) -> CInt {
    let mut buf = [0u8; 64];
    let n = format_loss_line(&mut buf, step, loss);
    unsafe {
        let h = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut written: u32 = 0;
        WriteFile(h, buf.as_ptr() as *const c_void, n as u32,
                  &mut written, core::ptr::null_mut());
    }
    0
}

fn format_loss_line(buf: &mut [u8], step: CInt, loss: f32) -> usize {
    let mut i = 0usize;
    for &b in b"step=" { if i < buf.len() { buf[i] = b; i += 1; } }
    i += write_int(&mut buf[i..], step as i64);
    for &b in b" loss=" { if i < buf.len() { buf[i] = b; i += 1; } }
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

// ----- test exports -----
#[no_mangle] pub extern "C" fn aether_test_log_f32(x: f32) -> f32 { libm::logf(x.max(1e-30)) }

// ----- math built-ins via the libm crate (f32 only — see `init_normal`
// for why the f64 helpers are off-limits in pe-bin DLLs). -----
#[inline] fn libm_expf(x: f32) -> f32 { libm::expf(x) }
#[inline] fn libm_lnf(x: f32)  -> f32 { libm::logf(x) }

// ----- panic + lang items required by no_std -----
// In `cargo test` mode the test harness pulls in `std` (and its own
// `panic_impl`), so our handler must only register for non-test builds.
#[cfg(not(test))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // Trip a UD2 so the OS terminates us cleanly. Aether-emitted code
    // should never trigger this.
    unsafe { core::arch::asm!("ud2", options(noreturn)) }
}

// `core`'s pre-built rlib carries unwind references to `rust_eh_personality`.
// Define a no-op so the link succeeds; we never unwind (panic = abort).
#[cfg(not(test))]
#[no_mangle] pub extern "C" fn rust_eh_personality() {}
