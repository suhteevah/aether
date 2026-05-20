//! Real libnccl backend for Aether's distributed surface.
//!
//! Compiled only under `--features nccl`. Wraps `cudarc::nccl::safe::Comm`
//! to provide a thin C-ABI on top of single-process multi-GPU NCCL.
//!
//! Design choices:
//! - **Single process, N comms**. `cudarc::nccl::safe::Comm::from_devices`
//!   creates one `ncclComm_t` per GPU within a single process. This matches
//!   matt-voice's deploy shape on cnc's 2× P100 box: one Aether binary
//!   spawning N threads/streams, one per card.
//! - **i64 opaque handles**. We index into a `Vec<Option<Comm>>` registry
//!   (handle = slot index + 1; 0 is null sentinel). Matches the existing
//!   `aether_dev_*` device-buffer convention.
//! - **Device-buffer arg**. NCCL only operates on device pointers, so the
//!   real `aether_nccl_all_reduce_f32_dev` takes the device-handle (from
//!   `aether_dev_alloc_f32` / `aether_dev_h2d_f32`) rather than host
//!   pointers — different from the fallback's host-pointer ABI.

use std::os::raw::c_int;
use std::sync::Arc;
use std::cell::UnsafeCell;

use cudarc::driver::CudaDevice;
use cudarc::nccl::safe::{Comm, ReduceOp};

struct NcclRegistry {
    comms: Vec<Option<Comm>>,
}

// Same UnsafeCell-static pattern as cuda.rs's BUFFERS to avoid the
// std::sync init AVs that struck early PE-bin loads. Multi-thread access
// is OK in practice because witness threads only register comms upfront
// then operate on their own comm handle in a fixed assignment.
struct NcclRegStatic(UnsafeCell<Option<NcclRegistry>>);
unsafe impl Sync for NcclRegStatic {}
static NCCL_REG: NcclRegStatic = NcclRegStatic(UnsafeCell::new(None));

#[allow(clippy::mut_from_ref)]
unsafe fn reg() -> &'static mut NcclRegistry {
    let r = NCCL_REG.0.get();
    if (*r).is_none() {
        *r = Some(NcclRegistry { comms: Vec::new() });
    }
    (*r).as_mut().unwrap()
}

fn handle_to_idx(h: i64) -> Option<usize> {
    if h <= 0 { None } else { Some((h - 1) as usize) }
}

/// Initialise N comms via libnccl's `ncclCommInitAll` (cudarc's
/// `Comm::from_devices`). Returns the number of comms created on success,
/// negative on error. Stores each comm in the registry; their handles are
/// retrievable in slot-order via `aether_nccl_real_get_handle`.
///
/// On a 2× P100 box, calling this with `n=2` lights up both cards.
#[no_mangle] pub extern "C" fn aether_nccl_real_init_multi_gpu(n: c_int) -> c_int {
    if n <= 0 { return -1; }
    let count = match CudaDevice::count() {
        Ok(c) => c as usize,
        Err(_) => return -2,
    };
    if (n as usize) > count { return -3; }
    let devices: Vec<Arc<CudaDevice>> = (0..n as usize)
        .filter_map(|i| CudaDevice::new(i).ok())
        .collect();
    if devices.len() != n as usize { return -4; }
    let comms = match Comm::from_devices(devices) {
        Ok(c) => c,
        Err(_) => return -5,
    };
    unsafe {
        let r = reg();
        // Reset existing comms so re-init starts fresh.
        r.comms.clear();
        for c in comms { r.comms.push(Some(c)); }
    }
    n
}

/// Get the opaque comm handle for rank `i` (0-based). Returns 0 on error.
#[no_mangle] pub extern "C" fn aether_nccl_real_get_handle(i: c_int) -> i64 {
    if i < 0 { return 0; }
    let r = unsafe { reg() };
    let idx = i as usize;
    if idx >= r.comms.len() || r.comms[idx].is_none() { return 0; }
    (idx + 1) as i64
}

/// Real all_reduce on device buffers. Op codes: 0=sum, 1=max, 2=min, 3=prod.
/// `send_dev` and `recv_dev` are handles from `aether_dev_alloc_f32`.
/// The element count must match what the source data was allocated with.
///
/// Returns 0 on success, non-zero on error.
#[no_mangle] pub extern "C" fn aether_nccl_real_all_reduce_f32(
    send_dev: i64, recv_dev: i64, n: c_int, op: c_int, comm: i64,
) -> c_int {
    let Some(c_idx) = handle_to_idx(comm) else { return 1; };
    let Some(s_idx) = handle_to_idx(send_dev) else { return 2; };
    let Some(r_idx) = handle_to_idx(recv_dev) else { return 3; };
    if n <= 0 { return 4; }
    let red_op = match op {
        0 => ReduceOp::Sum,
        1 => ReduceOp::Max,
        2 => ReduceOp::Min,
        3 => ReduceOp::Prod,
        _ => return 5,
    };
    // Take the send and recv buffers out of the cuda.rs BUFFERS table,
    // run all_reduce, put them back. Same dance as
    // aether_op_matmul_f32_cuda.
    let (send_buf, recv_buf_taken) = unsafe {
        let bs = crate::cuda::bufs();
        if s_idx >= bs.len() || r_idx >= bs.len() { return 6; }
        let s = bs[s_idx].take();
        let r = bs[r_idx].take();
        (s, r)
    };
    let (Some(send_buf), Some(recv_buf_unwrapped)) = (send_buf, recv_buf_taken) else {
        return 7;
    };
    let mut recv_buf = Some(recv_buf_unwrapped);
    let comm_ref = unsafe {
        let r = reg();
        r.comms[c_idx].as_ref().expect("comm slot empty")
    };
    let rc = comm_ref.all_reduce(&send_buf, recv_buf.as_mut().unwrap(), &red_op);
    // Put buffers back regardless of NCCL result.
    unsafe {
        let bs = crate::cuda::bufs();
        bs[s_idx] = Some(send_buf);
        bs[r_idx] = recv_buf;
    }
    match rc {
        Ok(_) => 0,
        Err(_) => 8,
    }
}

/// Return the world_size for `comm` (number of ranks the comm sees).
#[no_mangle] pub extern "C" fn aether_nccl_real_comm_world_size(comm: i64) -> c_int {
    let Some(idx) = handle_to_idx(comm) else { return -1; };
    let r = unsafe { reg() };
    if idx >= r.comms.len() { return -1; }
    r.comms[idx].as_ref().map(|c| c.world_size() as c_int).unwrap_or(-1)
}

/// Return the rank assigned to `comm`.
#[no_mangle] pub extern "C" fn aether_nccl_real_comm_rank(comm: i64) -> c_int {
    let Some(idx) = handle_to_idx(comm) else { return -1; };
    let r = unsafe { reg() };
    if idx >= r.comms.len() { return -1; }
    r.comms[idx].as_ref().map(|c| c.rank() as c_int).unwrap_or(-1)
}

/// Tear down all comms in the registry.
#[no_mangle] pub extern "C" fn aether_nccl_real_finalize() -> c_int {
    unsafe {
        let r = reg();
        r.comms.clear();
    }
    0
}

/// Rust-side accessor for the integration test. Returns a reference to
/// the comm at index `i` (NOT a handle). Used by the dual-GPU test to
/// drive `Comm::all_reduce` directly with cudarc-allocated device
/// buffers (one per rank/GPU), since the `aether_dev_*` registry is
/// tied to a single global cuda context.
pub fn comm_at(i: usize) -> Option<&'static Comm> {
    unsafe {
        let r = reg();
        if i < r.comms.len() {
            r.comms[i].as_ref().map(|c| &*(c as *const Comm))
        } else {
            None
        }
    }
}
