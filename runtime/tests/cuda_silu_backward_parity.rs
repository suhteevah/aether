//! matt-voice FR-18.6-real leg 2 — SiLU backward GPU parity.
//!
//! roadmap: P18
//!
//! qwen3 FFN is SwiGLU (silu, not gelu); training backward needs silu'(x).
//! Verifies the new silu_bwd kernel against a from-scratch CPU reference
//! silu'(x) = sigmoid(x)*(1 + x*(1 - sigmoid(x))).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_silu_backward_f32_cuda,
};

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }).collect()
}

#[test]
fn silu_backward_matches_cpu() {
    aether_dev_init();
    let n = 4096usize;
    let x = fill(1, n);
    let dy = fill(2, n);

    let mut cpu = vec![0.0f32; n];
    for i in 0..n {
        let sig = 1.0 / (1.0 + (-x[i]).exp());
        let gp = sig * (1.0 + x[i] * (1.0 - sig));
        cpu[i] = dy[i] * gp;
    }

    let xd = aether_dev_alloc_f32(n as c_int);
    let dyd = aether_dev_alloc_f32(n as c_int);
    let dxd = aether_dev_alloc_f32(n as c_int);
    assert!(xd >= 0 && dyd >= 0 && dxd >= 0);
    let mut gpu = vec![0.0f32; n];
    unsafe {
        aether_dev_h2d_f32(x.as_ptr() as i64, xd, n as c_int);
        aether_dev_h2d_f32(dy.as_ptr() as i64, dyd, n as c_int);
        let rc = aether_op_silu_backward_f32_cuda(xd, dyd, dxd, n as c_int);
        assert_eq!(rc, 0, "silu_backward returned {}", rc);
        aether_dev_sync();
        aether_dev_d2h_f32(dxd, gpu.as_mut_ptr() as i64, n as c_int);
        aether_dev_sync();
        aether_dev_free_f32(xd); aether_dev_free_f32(dyd); aether_dev_free_f32(dxd);
    }
    let md = gpu.iter().zip(&cpu).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("[silu_bwd parity] n={} max|diff|={:.3e}", n, md);
    assert!(md < 1e-6, "silu_bwd parity {:.3e}", md);
}
