//! Verify the `_devarg` kernel variants (rope_apply, append_kv,
//! attention_seq1) produce bit-identical output to their immediate-arg
//! counterparts. Required before we wire them into the CUDA-graph-
//! captured autoregressive forward pass.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_i32, aether_dev_h2d_i32,
    aether_op_rope_apply_f32_cuda,
    aether_op_rope_apply_devarg_f32_cuda,
    aether_op_append_kv_f32_cuda,
    aether_op_append_kv_devarg_f32_cuda,
    aether_op_attention_seq1_f32_cuda,
    aether_op_attention_seq1_devarg_f32_cuda,
};

fn close_enough(a: &[f32], b: &[f32], tol: f32) -> (f32, usize) {
    let mut max_diff = 0.0f32;
    let mut bad = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_diff { max_diff = d; }
        if d > tol { bad += 1; }
    }
    (max_diff, bad)
}

#[test]
#[ignore]
fn rope_devarg_matches_immediate() {
    unsafe {
        aether_dev_init();
        const SEQ: c_int = 1;
        const N_HEADS: c_int = 28;
        const HEAD_DIM: c_int = 128;
        const BASE: f32 = 1_000_000.0;
        const POS: c_int = 7;
        let n = (SEQ * N_HEADS * HEAD_DIM) as usize;

        let x_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let d_x_a = aether_dev_alloc_f32(n as c_int);
        let d_x_b = aether_dev_alloc_f32(n as c_int);
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x_a, n as c_int);
        aether_dev_h2d_f32(x_host.as_ptr() as i64, d_x_b, n as c_int);

        // Path A: immediate arg
        assert_eq!(0, aether_op_rope_apply_f32_cuda(d_x_a, SEQ, N_HEADS, HEAD_DIM, BASE, POS));

        // Path B: devarg
        let d_step = aether_dev_alloc_i32(4);
        let step_host = [POS, 0i32, 0i32, 0i32];
        aether_dev_h2d_i32(step_host.as_ptr() as i64, d_step, 4);
        assert_eq!(0, aether_op_rope_apply_devarg_f32_cuda(
            d_x_b, SEQ, N_HEADS, HEAD_DIM, BASE, d_step));
        aether_dev_sync();

        let mut a = vec![0.0f32; n];
        let mut b = vec![0.0f32; n];
        aether_dev_d2h_f32(d_x_a, a.as_mut_ptr() as i64, n as c_int);
        aether_dev_d2h_f32(d_x_b, b.as_mut_ptr() as i64, n as c_int);

        let (max_diff, bad) = close_enough(&a, &b, 1e-6);
        eprintln!("[rope devarg] max_diff={:.3e} bad={}/{}", max_diff, bad, n);
        assert_eq!(bad, 0, "rope_apply_devarg diverges");

        aether_dev_free_f32(d_x_a); aether_dev_free_f32(d_x_b);
    }
}

#[test]
#[ignore]
fn append_kv_devarg_matches_immediate() {
    unsafe {
        aether_dev_init();
        const D_KV: c_int = 512;
        const MAX_SEQ: c_int = 32;
        const POS: c_int = 5;
        let k_new: Vec<f32> = (0..D_KV as usize).map(|i| i as f32 * 0.01).collect();
        let v_new: Vec<f32> = (0..D_KV as usize).map(|i| -(i as f32) * 0.02).collect();

        let d_kn  = aether_dev_alloc_f32(D_KV);
        let d_vn  = aether_dev_alloc_f32(D_KV);
        let d_kca = aether_dev_alloc_f32(MAX_SEQ * D_KV);
        let d_vca = aether_dev_alloc_f32(MAX_SEQ * D_KV);
        let d_kcb = aether_dev_alloc_f32(MAX_SEQ * D_KV);
        let d_vcb = aether_dev_alloc_f32(MAX_SEQ * D_KV);
        aether_dev_h2d_f32(k_new.as_ptr() as i64, d_kn, D_KV);
        aether_dev_h2d_f32(v_new.as_ptr() as i64, d_vn, D_KV);

        assert_eq!(0, aether_op_append_kv_f32_cuda(d_kn, d_vn, d_kca, d_vca, POS, D_KV));

        let d_step = aether_dev_alloc_i32(4);
        let step_host = [POS, 0i32, 0i32, 0i32];
        aether_dev_h2d_i32(step_host.as_ptr() as i64, d_step, 4);
        assert_eq!(0, aether_op_append_kv_devarg_f32_cuda(d_kn, d_vn, d_kcb, d_vcb, D_KV, d_step));
        aether_dev_sync();

        let total = (MAX_SEQ * D_KV) as usize;
        let mut ka = vec![0.0f32; total]; let mut va = vec![0.0f32; total];
        let mut kb = vec![0.0f32; total]; let mut vb = vec![0.0f32; total];
        aether_dev_d2h_f32(d_kca, ka.as_mut_ptr() as i64, MAX_SEQ * D_KV);
        aether_dev_d2h_f32(d_vca, va.as_mut_ptr() as i64, MAX_SEQ * D_KV);
        aether_dev_d2h_f32(d_kcb, kb.as_mut_ptr() as i64, MAX_SEQ * D_KV);
        aether_dev_d2h_f32(d_vcb, vb.as_mut_ptr() as i64, MAX_SEQ * D_KV);

        let (md_k, bad_k) = close_enough(&ka, &kb, 0.0);
        let (md_v, bad_v) = close_enough(&va, &vb, 0.0);
        eprintln!("[append_kv devarg] k max_diff={:.3e} bad={}, v max_diff={:.3e} bad={}",
            md_k, bad_k, md_v, bad_v);
        assert_eq!(bad_k + bad_v, 0, "append_kv_devarg diverges");
    }
}

#[test]
#[ignore]
fn attention_devarg_matches_immediate() {
    unsafe {
        aether_dev_init();
        const N_Q: c_int = 28;
        const N_KV: c_int = 4;
        const HEAD_DIM: c_int = 128;
        const D_KV: c_int = N_KV * HEAD_DIM;
        const MAX_SEQ: c_int = 32;
        const CUR_SEQ: c_int = 7;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();

        let q: Vec<f32> = (0..(N_Q * HEAD_DIM) as usize).map(|i| ((i as f32) * 1e-3) - 0.5).collect();
        let k_cache: Vec<f32> = (0..(MAX_SEQ * D_KV) as usize).map(|i| ((i as f32) * 2e-4) - 1.0).collect();
        let v_cache: Vec<f32> = (0..(MAX_SEQ * D_KV) as usize).map(|i| -((i as f32) * 1.5e-4) + 0.7).collect();

        let d_q = aether_dev_alloc_f32(N_Q * HEAD_DIM);
        let d_kc = aether_dev_alloc_f32(MAX_SEQ * D_KV);
        let d_vc = aether_dev_alloc_f32(MAX_SEQ * D_KV);
        let d_out_a = aether_dev_alloc_f32(N_Q * HEAD_DIM);
        let d_out_b = aether_dev_alloc_f32(N_Q * HEAD_DIM);
        aether_dev_h2d_f32(q.as_ptr() as i64, d_q, N_Q * HEAD_DIM);
        aether_dev_h2d_f32(k_cache.as_ptr() as i64, d_kc, MAX_SEQ * D_KV);
        aether_dev_h2d_f32(v_cache.as_ptr() as i64, d_vc, MAX_SEQ * D_KV);

        assert_eq!(0, aether_op_attention_seq1_f32_cuda(
            d_q, d_kc, d_vc, d_out_a, CUR_SEQ, N_Q, N_KV, HEAD_DIM, scale));

        let d_step = aether_dev_alloc_i32(4);
        let step_host = [0i32, CUR_SEQ, 0i32, 0i32];
        aether_dev_h2d_i32(step_host.as_ptr() as i64, d_step, 4);
        assert_eq!(0, aether_op_attention_seq1_devarg_f32_cuda(
            d_q, d_kc, d_vc, d_out_b, N_Q, N_KV, HEAD_DIM, scale, MAX_SEQ, d_step));
        aether_dev_sync();

        let n = (N_Q * HEAD_DIM) as usize;
        let mut a = vec![0.0f32; n]; let mut b = vec![0.0f32; n];
        aether_dev_d2h_f32(d_out_a, a.as_mut_ptr() as i64, N_Q * HEAD_DIM);
        aether_dev_d2h_f32(d_out_b, b.as_mut_ptr() as i64, N_Q * HEAD_DIM);

        let (md, bad) = close_enough(&a, &b, 1e-5);
        eprintln!("[attention devarg] max_diff={:.3e} bad={}/{}", md, bad, n);
        eprintln!("  a[0..4] = {:?}", &a[..4]);
        eprintln!("  b[0..4] = {:?}", &b[..4]);
        assert_eq!(bad, 0, "attention_seq1_devarg diverges");
    }
}
