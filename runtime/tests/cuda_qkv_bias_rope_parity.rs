//! Parity test for the fused `qkv_bias_rope` kernel (whole-layer-fusion Slice A).
//!
//! The fused kernel collapses bias_add(q)+bias_add(k)+bias_add(v)+rope(q)+rope(k)
//! — 5 seq1-decode kernels — into one launch.  This guards that the fused result
//! is bit-identical to running those 5 ops in sequence (same FP ops, same order:
//! bias THEN rope), across the Qwen2.5-7B GQA shape + a few others.
//!
//! roadmap: P19.5 (perf — whole-layer kernel fusion)
#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_i32, aether_dev_free_i32, aether_dev_h2d_i32,
    aether_op_bias_add_f32_cuda, aether_op_rope_apply_devarg_f32_cuda,
    aether_op_qkv_bias_rope_devarg_f32_cuda,
};

struct Gen { s: u64 }
impl Gen {
    fn next(&mut self) -> f32 {
        self.s ^= self.s << 13; self.s ^= self.s >> 7; self.s ^= self.s << 17;
        ((self.s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.next()).collect() }
}

unsafe fn upload(v: &[f32]) -> i64 {
    let d = aether_dev_alloc_f32(v.len() as c_int);
    aether_dev_h2d_f32(v.as_ptr() as i64, d, v.len() as c_int);
    d
}
unsafe fn download(d: i64, n: usize) -> Vec<f32> {
    let mut h = vec![0f32; n];
    aether_dev_d2h_f32(d, h.as_mut_ptr() as i64, n as c_int);
    h
}

fn run_case(n_q_heads: usize, n_kv_heads: usize, head_dim: usize, pos: i32, base: f32, seed: u64) {
    let q_dim = n_q_heads * head_dim;
    let d_kv = n_kv_heads * head_dim;
    let mut g = Gen { s: seed };
    let q = g.fill(q_dim);
    let k = g.fill(d_kv);
    let v = g.fill(d_kv);
    let bq = g.fill(q_dim);
    let bk = g.fill(d_kv);
    let bv = g.fill(d_kv);

    unsafe {
        assert_eq!(aether_dev_init(), 0);
        let sa = aether_dev_alloc_i32(4);
        let sa_host = [pos, pos + 1, 0i32, 0i32];
        aether_dev_h2d_i32(sa_host.as_ptr() as i64, sa, 4);

        // --- reference: bias_add x3 then rope x2 (the per-op path) ---
        let (rq, rk, rv) = (upload(&q), upload(&k), upload(&v));
        let (rbq, rbk, rbv) = (upload(&bq), upload(&bk), upload(&bv));
        aether_op_bias_add_f32_cuda(rq, rbq, 1, q_dim as c_int);
        aether_op_bias_add_f32_cuda(rk, rbk, 1, d_kv as c_int);
        aether_op_bias_add_f32_cuda(rv, rbv, 1, d_kv as c_int);
        aether_op_rope_apply_devarg_f32_cuda(rq, 1, n_q_heads as c_int, head_dim as c_int, base, sa);
        aether_op_rope_apply_devarg_f32_cuda(rk, 1, n_kv_heads as c_int, head_dim as c_int, base, sa);
        aether_dev_sync();
        let ref_q = download(rq, q_dim);
        let ref_k = download(rk, d_kv);
        let ref_v = download(rv, d_kv);

        // --- fused: one launch ---
        let (fq, fk, fv) = (upload(&q), upload(&k), upload(&v));
        let (fbq, fbk, fbv) = (upload(&bq), upload(&bk), upload(&bv));
        let rc = aether_op_qkv_bias_rope_devarg_f32_cuda(
            fq, fk, fv, fbq, fbk, fbv,
            n_q_heads as c_int, n_kv_heads as c_int, head_dim as c_int, base, sa);
        assert_eq!(rc, 0, "fused launch rc");
        aether_dev_sync();
        let fus_q = download(fq, q_dim);
        let fus_k = download(fk, d_kv);
        let fus_v = download(fv, d_kv);

        let maxd = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let dq = maxd(&ref_q, &fus_q);
        let dk = maxd(&ref_k, &fus_k);
        let dv = maxd(&ref_v, &fus_v);
        println!("[qkv_bias_rope] nq={} nkv={} hd={} pos={}: max_diff q={:.2e} k={:.2e} v={:.2e}",
            n_q_heads, n_kv_heads, head_dim, pos, dq, dk, dv);
        // Identical FP ops in identical order → expect bit-exact (allow tiny slack).
        assert!(dq < 1e-5 && dk < 1e-5 && dv < 1e-5,
            "fused != sequential (q={:.2e} k={:.2e} v={:.2e})", dq, dk, dv);

        for h in [sa, rq, rk, rv, rbq, rbk, rbv, fq, fk, fv, fbq, fbk, fbv] {
            // i32 vs f32 free: sa is i32, rest f32.
            if h == sa { aether_dev_free_i32(h); } else { aether_dev_free_f32(h); }
        }
    }
}

#[test]
fn qkv_bias_rope_matches_sequential() {
    // Qwen2.5-7B: 28 q heads, 4 kv heads (GQA), head_dim 128.
    run_case(28, 4, 128, 0, 1_000_000.0, 0x1234);
    run_case(28, 4, 128, 37, 1_000_000.0, 0xBEEF);
    run_case(28, 4, 128, 511, 1_000_000.0, 0xC0DE);
    // MHA (no GQA) + smaller head_dim + different base.
    run_case(16, 16, 64, 13, 10_000.0, 0xFACE);
    run_case(8, 2, 128, 200, 500_000.0, 0xD00D);
}
