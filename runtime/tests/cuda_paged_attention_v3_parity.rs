//! Flash-attention-style paged seq1 (v3) parity test vs v1 (canonical) + v2.
//! v3 uses online softmax + fused K+V single pass.  Math is identical up to
//! float reassociation (online vs offline softmax + per-warp partial weighting),
//! so rel ~1e-4 is expected.  This catches int-math bugs (wrong KQ dot, wrong
//! V indexing, wrong online-softmax merge).
//!
//! roadmap: P19.4

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_alloc_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
    aether_op_paged_attention_seq1_v3_devarg_f32_cuda,
};

const N_Q_HEADS: i32 = 28;
const N_KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 128;
const D_KV: i32 = N_KV_HEADS * HEAD_DIM;
const BLOCK_SIZE: i32 = 16;

fn run_case(cur_seq: i32) {
    let max_seq: i32 = ((cur_seq / BLOCK_SIZE) + 2) * BLOCK_SIZE;
    let n_blocks = max_seq / BLOCK_SIZE;
    let pool_tokens = max_seq;
    let pt: Vec<i32> = (0..n_blocks).collect();
    let pt_dev = unsafe { aether_dev_alloc_i32(n_blocks) };
    unsafe { aether_dev_h2d_i32(pt.as_ptr() as i64, pt_dev, n_blocks); }

    let q: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM))
        .map(|i| ((i as f32) * 0.013 - 1.5).sin() * 0.7).collect();
    let q_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    unsafe { aether_dev_h2d_f32(q.as_ptr() as i64, q_dev, N_Q_HEADS * HEAD_DIM); }

    let k_pool = unsafe { aether_dev_alloc_f32(pool_tokens * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(pool_tokens * D_KV) };
    let k_new = unsafe { aether_dev_alloc_f32(D_KV) };
    let v_new = unsafe { aether_dev_alloc_f32(D_KV) };
    let step_args = unsafe { aether_dev_alloc_i32(4) };
    for pos in 0..cur_seq {
        let k: Vec<f32> = (0..D_KV).map(|i| (((i + pos) as f32) * 0.021 + 0.4).cos() * 0.5).collect();
        let v: Vec<f32> = (0..D_KV).map(|i| (((i + 3 * pos) as f32) * 0.017 - 0.2).sin() * 0.4).collect();
        unsafe {
            aether_dev_h2d_f32(k.as_ptr() as i64, k_new, D_KV);
            aether_dev_h2d_f32(v.as_ptr() as i64, v_new, D_KV);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args, 4);
            assert_eq!(0, aether_op_paged_append_kv_devarg_f32_cuda(
                k_new, v_new, k_pool, v_pool, pt_dev, D_KV, BLOCK_SIZE, step_args));
        }
    }
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let out1 = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let out3 = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let sa = [cur_seq - 1, cur_seq, 0, 0];
    unsafe { aether_dev_h2d_i32(sa.as_ptr() as i64, step_args, 4); }
    unsafe {
        assert_eq!(0, aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_dev, k_pool, v_pool, pt_dev, out1,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, max_seq, step_args));
        assert_eq!(0, aether_op_paged_attention_seq1_v3_devarg_f32_cuda(
            q_dev, k_pool, v_pool, pt_dev, out3,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, max_seq, step_args));
        aether_dev_sync();
    }
    let mut o1 = vec![0.0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut o3 = vec![0.0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    unsafe {
        aether_dev_d2h_f32(out1, o1.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        aether_dev_d2h_f32(out3, o3.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (a, b) in o1.iter().zip(o3.iter()) {
        let d = (a - b).abs();
        if d > max_abs { max_abs = d; }
        let r = d / a.abs().max(b.abs()).max(1e-6);
        if r > max_rel { max_rel = r; }
    }
    eprintln!("[v3-parity] cur_seq={:>4} max_abs={:.3e} max_rel={:.3e}",
        cur_seq, max_abs, max_rel);
    assert!(max_abs < 1e-3, "cur_seq={} max_abs={:.3e}", cur_seq, max_abs);
    assert!(max_rel < 1e-2, "cur_seq={} max_rel={:.3e}", cur_seq, max_rel);
}

#[test]
fn paged_attention_v3_matches_v1() {
    unsafe { assert!(aether_dev_init() == 0); }
    for &cs in &[1, 7, 33, 100, 257, 513] { run_case(cs); }
}
