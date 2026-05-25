//! matt-voice FR-18.6-real leg 2 — causal SDPA backward GPU parity.
//!
//! roadmap: P18
//!
//! Verifies the GPU sdpa_causal_bwd_{dq,dkv} kernels against the trusted CPU
//! reference ops::sdpa_causal_backward_f32 (already used by the trainer's
//! model.rs). Synthetic q/k/v: run CPU forward to get attn+out, build a
//! synthetic dout, take CPU backward as ground truth, then run the GPU op on
//! the same inputs and assert max-abs-diff on dq/dk/dv.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::ops::{sdpa_causal_f32, sdpa_causal_backward_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_sdpa_causal_backward_f32_cuda,
};

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }).collect()
}

#[test]
fn sdpa_causal_backward_matches_cpu() {
    aether_dev_init();
    let bh = 6usize;   // batch*heads
    let s_len = 12usize;
    let d = 16usize;   // head_dim
    let n = bh * s_len * d;
    let nss = bh * s_len * s_len;

    let q = fill(1, n);
    let k = fill(2, n);
    let v = fill(3, n);
    let dout = fill(4, n);

    // CPU forward → attn probs + out (out unused, attn fed to backward).
    let mut out = vec![0.0f32; n];
    let mut attn = vec![0.0f32; nss];
    unsafe { sdpa_causal_f32(q.as_ptr(), k.as_ptr(), v.as_ptr(), out.as_mut_ptr(), attn.as_mut_ptr(), bh, s_len, d); }

    // CPU backward = ground truth.
    let mut cdq = vec![0.0f32; n];
    let mut cdk = vec![0.0f32; n];
    let mut cdv = vec![0.0f32; n];
    unsafe {
        sdpa_causal_backward_f32(
            q.as_ptr(), k.as_ptr(), v.as_ptr(), attn.as_ptr(), dout.as_ptr(),
            cdq.as_mut_ptr(), cdk.as_mut_ptr(), cdv.as_mut_ptr(), bh, s_len, d);
    }

    // GPU backward.
    let qd = aether_dev_alloc_f32(n as c_int);
    let kd = aether_dev_alloc_f32(n as c_int);
    let vd = aether_dev_alloc_f32(n as c_int);
    let ad = aether_dev_alloc_f32(nss as c_int);
    let dod = aether_dev_alloc_f32(n as c_int);
    let dqd = aether_dev_alloc_f32(n as c_int);
    let dkd = aether_dev_alloc_f32(n as c_int);
    let dvd = aether_dev_alloc_f32(n as c_int);
    let dsd = aether_dev_alloc_f32(nss as c_int);
    assert!([qd, kd, vd, ad, dod, dqd, dkd, dvd, dsd].iter().all(|&h| h >= 0));

    let mut gdq = vec![0.0f32; n];
    let mut gdk = vec![0.0f32; n];
    let mut gdv = vec![0.0f32; n];
    unsafe {
        aether_dev_h2d_f32(q.as_ptr() as i64, qd, n as c_int);
        aether_dev_h2d_f32(k.as_ptr() as i64, kd, n as c_int);
        aether_dev_h2d_f32(v.as_ptr() as i64, vd, n as c_int);
        aether_dev_h2d_f32(attn.as_ptr() as i64, ad, nss as c_int);
        aether_dev_h2d_f32(dout.as_ptr() as i64, dod, n as c_int);

        let rc = aether_op_sdpa_causal_backward_f32_cuda(
            qd, kd, vd, ad, dod, dqd, dkd, dvd, dsd,
            bh as c_int, s_len as c_int, d as c_int);
        assert_eq!(rc, 0, "sdpa_causal_backward returned {}", rc);
        aether_dev_sync();

        aether_dev_d2h_f32(dqd, gdq.as_mut_ptr() as i64, n as c_int);
        aether_dev_d2h_f32(dkd, gdk.as_mut_ptr() as i64, n as c_int);
        aether_dev_d2h_f32(dvd, gdv.as_mut_ptr() as i64, n as c_int);
        aether_dev_sync();
        for h in [qd, kd, vd, ad, dod, dqd, dkd, dvd, dsd] { aether_dev_free_f32(h); }
    }

    let md = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (mq, mk, mv) = (md(&gdq, &cdq), md(&gdk, &cdk), md(&gdv, &cdv));
    eprintln!("[sdpa_bwd parity] bh={} s={} d={} max|dq|={:.3e} max|dk|={:.3e} max|dv|={:.3e}",
        bh, s_len, d, mq, mk, mv);
    assert!(mq < 1e-4, "dq parity {:.3e}", mq);
    assert!(mk < 1e-4, "dk parity {:.3e}", mk);
    assert!(mv < 1e-4, "dv parity {:.3e}", mv);
}
