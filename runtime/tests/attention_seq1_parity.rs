//! Verify the GPU `attention_seq1` kernel produces the same output as
//! a CPU reference for known Q, K_cache, V_cache.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_op_attention_seq1_f32_cuda,
};

fn cpu_attention_seq1(
    q: &[f32], k_cache: &[f32], v_cache: &[f32],
    cur_seq: usize, n_q: usize, n_kv: usize, head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let d_kv = n_kv * head_dim;
    let mut out = vec![0.0f32; n_q * head_dim];
    let kv_per_q = n_q / n_kv;
    for h in 0..n_q {
        let kv_h = h / kv_per_q;
        // scores[t] = (Q[h] · K_cache[t, kv_h]) * scale
        let mut scores = vec![0.0f32; cur_seq];
        for t in 0..cur_seq {
            let mut acc = 0.0f32;
            for d in 0..head_dim {
                acc += q[h * head_dim + d] * k_cache[t * d_kv + kv_h * head_dim + d];
            }
            scores[t] = acc * scale;
        }
        // softmax
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - mx).exp();
            sum += *s;
        }
        let inv = 1.0 / sum;
        for s in scores.iter_mut() { *s *= inv; }
        // out[h] = sum_t softmax[t] * V_cache[t, kv_h]
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for t in 0..cur_seq {
                acc += scores[t] * v_cache[t * d_kv + kv_h * head_dim + d];
            }
            out[h * head_dim + d] = acc;
        }
    }
    out
}

#[test]
fn attention_seq1_matches_cpu() {
    unsafe {
        aether_dev_init();
        // Qwen2.5-7B shape
        let n_q = 28;
        let n_kv = 4;
        let head_dim = 128;
        let cur_seq = 7;  // arbitrary
        let d_q = n_q * head_dim;
        let d_kv = n_kv * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..d_q).map(|i| ((i as f32) * 0.0013 - 0.5)).collect();
        let k_cache: Vec<f32> = (0..cur_seq * d_kv).map(|i| ((i as f32) * 0.0007 - 0.3)).collect();
        let v_cache: Vec<f32> = (0..cur_seq * d_kv).map(|i| ((i as f32) * 0.0011 - 0.4)).collect();

        let cpu = cpu_attention_seq1(&q, &k_cache, &v_cache, cur_seq, n_q, n_kv, head_dim, scale);

        let d_q_buf  = aether_dev_alloc_f32(d_q as c_int);
        let d_kc     = aether_dev_alloc_f32((cur_seq * d_kv) as c_int);
        let d_vc     = aether_dev_alloc_f32((cur_seq * d_kv) as c_int);
        let d_out    = aether_dev_alloc_f32(d_q as c_int);
        aether_dev_h2d_f32(q.as_ptr() as i64, d_q_buf, d_q as c_int);
        aether_dev_h2d_f32(k_cache.as_ptr() as i64, d_kc, (cur_seq * d_kv) as c_int);
        aether_dev_h2d_f32(v_cache.as_ptr() as i64, d_vc, (cur_seq * d_kv) as c_int);

        let rc = aether_op_attention_seq1_f32_cuda(
            d_q_buf, d_kc, d_vc, d_out, cur_seq as c_int,
            n_q as c_int, n_kv as c_int, head_dim as c_int, scale,
        );
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu = vec![0.0f32; d_q];
        aether_dev_d2h_f32(d_out, gpu.as_mut_ptr() as i64, d_q as c_int);

        let mut max_diff = 0.0f32;
        let mut worst = 0;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_diff { max_diff = d; worst = i; }
        }
        eprintln!("[attn] cur_seq={}, n_q={}, max_diff = {:.3e} at i={}", cur_seq, n_q, max_diff, worst);
        eprintln!("  cpu[0..4] = {:?}", &cpu[..4]);
        eprintln!("  gpu[0..4] = {:?}", &gpu[..4]);
        eprintln!("  cpu[worst] = {}, gpu[worst] = {}", cpu[worst], gpu[worst]);
        assert!(max_diff < 1e-3, "attention GPU/CPU mismatch beyond tol");

        aether_dev_free_f32(d_q_buf);
        aether_dev_free_f32(d_kc);
        aether_dev_free_f32(d_vc);
        aether_dev_free_f32(d_out);
    }
}
