//! matt-voice FR-18.6-real leg 2 — transpose_021 GPU parity + round-trip.
//!
//! roadmap: P18
//!
//! Swaps [s,h,hd] <-> [h,s,hd] for training attention. Checks GPU vs CPU and
//! that transpose-then-inverse recovers the input.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_transpose_021_f32_cuda,
};

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }).collect()
}

#[test]
fn transpose021_matches_cpu_and_round_trips() {
    aether_dev_init();
    let (s, h, hd) = (3usize, 2usize, 4usize);
    let n = s * h * hd;
    let x = fill(1, n);

    // CPU [s,h,hd] -> [h,s,hd].
    let mut cpu = vec![0.0f32; n];
    for a in 0..s { for b in 0..h { for c in 0..hd {
        cpu[(b * s + a) * hd + c] = x[(a * h + b) * hd + c];
    }}}

    let xd = aether_dev_alloc_f32(n as c_int);
    let td = aether_dev_alloc_f32(n as c_int);
    let rd = aether_dev_alloc_f32(n as c_int);
    let mut gpu_t = vec![0.0f32; n];
    let mut gpu_r = vec![0.0f32; n];
    unsafe {
        aether_dev_h2d_f32(x.as_ptr() as i64, xd, n as c_int);
        // [s,h,hd] -> [h,s,hd]
        assert_eq!(aether_op_transpose_021_f32_cuda(xd, td, s as c_int, h as c_int, hd as c_int), 0);
        // inverse [h,s,hd] -> [s,h,hd]
        assert_eq!(aether_op_transpose_021_f32_cuda(td, rd, h as c_int, s as c_int, hd as c_int), 0);
        aether_dev_sync();
        aether_dev_d2h_f32(td, gpu_t.as_mut_ptr() as i64, n as c_int);
        aether_dev_d2h_f32(rd, gpu_r.as_mut_ptr() as i64, n as c_int);
        aether_dev_sync();
        aether_dev_free_f32(xd); aether_dev_free_f32(td); aether_dev_free_f32(rd);
    }
    let md = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    eprintln!("[transpose021] max|gpu-cpu|={:.3e} round-trip max|diff|={:.3e}",
        md(&gpu_t, &cpu), md(&gpu_r, &x));
    assert!(md(&gpu_t, &cpu) < 1e-9, "transpose mismatch");
    assert!(md(&gpu_r, &x) < 1e-9, "round-trip mismatch");
}
