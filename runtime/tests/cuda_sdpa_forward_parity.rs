//! matt-voice FR-18.6-real leg 2 — full-sequence causal SDPA forward GPU parity.
//!
//! roadmap: P18
//!
//! The training forward must materialise the [bh,s,s] attention probs that the
//! SDPA backward consumes (the decode path is seq1/paged and never does). This
//! verifies the new sdpa_causal_fwd kernel against the trusted CPU reference
//! ops::sdpa_causal_f32 — both `out` AND `attn` (the saved softmax probs).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::ops::sdpa_causal_f32;
use aether_rt::cuda::{
    aether_dev_init, aether_dev_sync,
    aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_op_sdpa_causal_forward_f32_cuda,
};

fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(|_| {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }).collect()
}

#[test]
fn sdpa_causal_forward_matches_cpu() {
    aether_dev_init();
    let bh = 6usize;
    let s_len = 12usize;
    let d = 16usize;
    let n = bh * s_len * d;
    let nss = bh * s_len * s_len;

    let q = fill(1, n);
    let k = fill(2, n);
    let v = fill(3, n);

    // CPU reference.
    let mut cout = vec![0.0f32; n];
    let mut cattn = vec![0.0f32; nss];
    unsafe { sdpa_causal_f32(q.as_ptr(), k.as_ptr(), v.as_ptr(), cout.as_mut_ptr(), cattn.as_mut_ptr(), bh, s_len, d); }

    // GPU.
    let qd = aether_dev_alloc_f32(n as c_int);
    let kd = aether_dev_alloc_f32(n as c_int);
    let vd = aether_dev_alloc_f32(n as c_int);
    let od = aether_dev_alloc_f32(n as c_int);
    let ad = aether_dev_alloc_f32(nss as c_int);
    assert!([qd, kd, vd, od, ad].iter().all(|&h| h >= 0));

    let mut gout = vec![0.0f32; n];
    let mut gattn = vec![0.0f32; nss];
    unsafe {
        aether_dev_h2d_f32(q.as_ptr() as i64, qd, n as c_int);
        aether_dev_h2d_f32(k.as_ptr() as i64, kd, n as c_int);
        aether_dev_h2d_f32(v.as_ptr() as i64, vd, n as c_int);
        let rc = aether_op_sdpa_causal_forward_f32_cuda(qd, kd, vd, od, ad, bh as c_int, s_len as c_int, d as c_int);
        assert_eq!(rc, 0, "sdpa_causal_forward returned {}", rc);
        aether_dev_sync();
        aether_dev_d2h_f32(od, gout.as_mut_ptr() as i64, n as c_int);
        aether_dev_d2h_f32(ad, gattn.as_mut_ptr() as i64, nss as c_int);
        aether_dev_sync();
        for h in [qd, kd, vd, od, ad] { aether_dev_free_f32(h); }
    }

    let md = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let (mo, ma) = (md(&gout, &cout), md(&gattn, &cattn));
    eprintln!("[sdpa_fwd parity] bh={} s={} d={} max|out|={:.3e} max|attn|={:.3e}", bh, s_len, d, mo, ma);
    assert!(mo < 1e-4, "out parity {:.3e}", mo);
    assert!(ma < 1e-5, "attn parity {:.3e}", ma);
}
