//! MLA attention kernel parity test (FR-17-extra-mla-fwd).
//!
//! DeepSeek-V2's Multi-head Latent Attention has two structural deviations
//! from standard GQA that the existing paged_attention kernels can't
//! accommodate:
//!
//!   (1) Q/K share one per-head dim `qk_head_dim` (e.g. 192 = qk_nope 128 +
//!       qk_rope 64), while V uses a DIFFERENT per-head dim `v_head_dim`
//!       (e.g. 128).
//!   (2) Per-token K row stride = `n_heads * qk_head_dim`; per-token V row
//!       stride = `n_heads * v_head_dim`.  The two pools are sized
//!       independently.
//!
//! Three GPU assertions in this file:
//!
//!   1. `mla_matches_cpu_reference_mla_shape`: with the V2-Lite shape
//!      (n_heads=16, qk=192, v=128, cur_seq=5) the GPU kernel matches a
//!      naive CPU reference to ≤ 1e-4 max abs diff.
//!
//!   2. `mla_matches_paged_seq1_when_qk_equals_v`: when qk_head_dim ==
//!      v_head_dim == 128, the MLA kernel is bit-identical to
//!      `paged_attention_seq1_devarg` (a degenerate-MLA check that locks in
//!      no regressions on the standard path).
//!
//!   3. `mla_append_kv_writes_with_independent_strides`: the paged MLA
//!      append kernel writes K row at qk_head_dim*n_heads stride and V row
//!      at v_head_dim*n_heads stride into the per-token slots of two pools
//!      sized independently.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
    aether_op_paged_attention_mla_devarg_f32_cuda,
    aether_op_paged_append_kv_mla_devarg_f32_cuda,
};

const BLOCK_SIZE: i32 = 4;
const N_BLOCKS:   i32 = 4;
const POOL_TOKENS: i32 = BLOCK_SIZE * N_BLOCKS;
const MAX_SEQ:    i32 = 16;

/// CPU reference for MLA attention.  Mirrors the kernel exactly:
///   per-head, scores[t] = (Q[head] · K[t, head]) * scale
///   probs   = softmax(scores)
///   out[head, :v_head_dim] = sum_t probs[t] * V[t, head]
fn cpu_mla_reference(
    q: &[f32],
    k_steps: &[Vec<f32>],
    v_steps: &[Vec<f32>],
    n_heads: usize, qk_head_dim: usize, v_head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let cur_seq = k_steps.len();
    let mut out = vec![0f32; n_heads * v_head_dim];
    for h in 0..n_heads {
        let q_h = &q[h * qk_head_dim .. (h + 1) * qk_head_dim];
        // scores
        let mut scores = vec![0f32; cur_seq];
        for t in 0..cur_seq {
            let k_t_h = &k_steps[t][h * qk_head_dim .. (h + 1) * qk_head_dim];
            let dot: f32 = q_h.iter().zip(k_t_h.iter()).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        // softmax
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
        for s in &mut scores { *s /= sum; }
        // weighted V
        for t in 0..cur_seq {
            let v_t_h = &v_steps[t][h * v_head_dim .. (h + 1) * v_head_dim];
            for i in 0..v_head_dim {
                out[h * v_head_dim + i] += scores[t] * v_t_h[i];
            }
        }
    }
    out
}

fn run_mla_pipeline(
    n_heads: i32, qk_head_dim: i32, v_head_dim: i32, cur_seq: i32,
) -> (Vec<f32>, Vec<f32>) {
    let d_k_row = n_heads * qk_head_dim;
    let d_v_row = n_heads * v_head_dim;

    // Deterministic synthetic inputs.
    let q: Vec<f32> = (0..(n_heads * qk_head_dim))
        .map(|i| ((i as f32) * 0.011 - 0.7).sin() * 0.6).collect();
    let k_steps: Vec<Vec<f32>> = (0..cur_seq).map(|t| {
        (0..d_k_row).map(|i| ((i + t) as f32 * 0.019 + 0.3).cos() * 0.5).collect()
    }).collect();
    let v_steps: Vec<Vec<f32>> = (0..cur_seq).map(|t| {
        (0..d_v_row).map(|i| ((i + 3 * t) as f32 * 0.013 - 0.1).sin() * 0.4).collect()
    }).collect();

    let n_logical = ((cur_seq + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
    let pt_host: Vec<i32> = (0..n_logical).collect();
    let pt_dev = unsafe { aether_dev_alloc_i32(n_logical) };
    unsafe { aether_dev_h2d_i32(pt_host.as_ptr() as i64, pt_dev, n_logical); }

    let q_dev    = unsafe { aether_dev_alloc_f32(n_heads * qk_head_dim) };
    let k_pool   = unsafe { aether_dev_alloc_f32(POOL_TOKENS * d_k_row) };
    let v_pool   = unsafe { aether_dev_alloc_f32(POOL_TOKENS * d_v_row) };
    let attn_out = unsafe { aether_dev_alloc_f32(n_heads * v_head_dim) };
    let k_new    = unsafe { aether_dev_alloc_f32(d_k_row) };
    let v_new    = unsafe { aether_dev_alloc_f32(d_v_row) };
    let step_args_dev = unsafe { aether_dev_alloc_i32(4) };

    unsafe { aether_dev_h2d_f32(q.as_ptr() as i64, q_dev, n_heads * qk_head_dim); }

    for pos in 0..cur_seq {
        unsafe {
            aether_dev_h2d_f32(k_steps[pos as usize].as_ptr() as i64, k_new, d_k_row);
            aether_dev_h2d_f32(v_steps[pos as usize].as_ptr() as i64, v_new, d_v_row);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
            let rc = aether_op_paged_append_kv_mla_devarg_f32_cuda(
                k_new, v_new, k_pool, v_pool, pt_dev,
                d_k_row, d_v_row, BLOCK_SIZE, step_args_dev);
            assert_eq!(rc, 0, "paged_append_kv_mla rc={}", rc);
        }
    }
    let scale = 1.0f32 / (qk_head_dim as f32).sqrt();
    let final_sa = [cur_seq - 1, cur_seq, 0, 0];
    unsafe { aether_dev_h2d_i32(final_sa.as_ptr() as i64, step_args_dev, 4); }
    unsafe {
        let rc = aether_op_paged_attention_mla_devarg_f32_cuda(
            q_dev, k_pool, v_pool, pt_dev, attn_out,
            n_heads, qk_head_dim, v_head_dim, BLOCK_SIZE,
            scale, MAX_SEQ, step_args_dev);
        assert_eq!(rc, 0, "paged_attention_mla rc={}", rc);
        aether_dev_sync();
    }
    let mut gpu_out = vec![0f32; (n_heads * v_head_dim) as usize];
    unsafe { aether_dev_d2h_f32(attn_out, gpu_out.as_mut_ptr() as i64, n_heads * v_head_dim); }
    unsafe {
        aether_dev_free_f32(q_dev); aether_dev_free_f32(k_pool);
        aether_dev_free_f32(v_pool); aether_dev_free_f32(attn_out);
        aether_dev_free_f32(k_new); aether_dev_free_f32(v_new);
        aether_dev_free_i32(pt_dev); aether_dev_free_i32(step_args_dev);
    }

    let cpu_out = cpu_mla_reference(
        &q, &k_steps, &v_steps,
        n_heads as usize, qk_head_dim as usize, v_head_dim as usize,
        scale);
    (cpu_out, gpu_out)
}

#[test]
fn mla_matches_cpu_reference_mla_shape() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    // DeepSeek-V2-Lite per-block shape.
    let n_heads = 16;
    let qk_head_dim = 192;   // qk_nope (128) + qk_rope (64)
    let v_head_dim = 128;
    let cur_seq = 5;
    let (cpu, gpu) = run_mla_pipeline(n_heads, qk_head_dim, v_head_dim, cur_seq);
    assert_eq!(cpu.len(), gpu.len());
    let max_diff = cpu.iter().zip(gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let n_finite = gpu.iter().filter(|x| x.is_finite()).count();
    println!("[mla] V2-Lite shape n_heads={} qk={} v={} cur_seq={} -> max_diff={:.3e} finite={}/{}",
        n_heads, qk_head_dim, v_head_dim, cur_seq, max_diff, n_finite, gpu.len());
    assert_eq!(n_finite, gpu.len(), "non-finite values in MLA output");
    assert!(max_diff < 1e-4, "MLA kernel diverged from CPU reference ({:.3e})", max_diff);
}

#[test]
fn mla_handles_small_shape() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    // Stress the per_lane=2 branch (head_dim < 32) and an asymmetric small
    // shape to exercise the bounds checks.
    let n_heads = 4;
    let qk_head_dim = 48;
    let v_head_dim = 32;
    let cur_seq = 3;
    let (cpu, gpu) = run_mla_pipeline(n_heads, qk_head_dim, v_head_dim, cur_seq);
    let max_diff = cpu.iter().zip(gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("[mla] small shape n_heads={} qk={} v={} cur_seq={} -> max_diff={:.3e}",
        n_heads, qk_head_dim, v_head_dim, cur_seq, max_diff);
    assert!(gpu.iter().all(|x| x.is_finite()),
        "non-finite values in small-shape MLA output");
    assert!(max_diff < 1e-4,
        "small-shape MLA kernel diverged from CPU reference ({:.3e})", max_diff);
}

#[test]
fn mla_matches_paged_seq1_when_qk_equals_v() {
    // When qk_head_dim == v_head_dim AND n_heads == n_kv_heads, MLA reduces
    // to standard MHA — so the MLA kernel must be bit-identical (modulo
    // floating-point order) to paged_attention_seq1_devarg.
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n_heads = 8;
    let head_dim = 128;
    let cur_seq = 6;
    let d_kv = n_heads * head_dim;

    // Inputs.
    let q: Vec<f32> = (0..(n_heads * head_dim))
        .map(|i| ((i as f32) * 0.011 - 0.7).sin() * 0.6).collect();
    let k_steps: Vec<Vec<f32>> = (0..cur_seq).map(|t| {
        (0..d_kv).map(|i| ((i + t) as f32 * 0.019 + 0.3).cos() * 0.5).collect()
    }).collect();
    let v_steps: Vec<Vec<f32>> = (0..cur_seq).map(|t| {
        (0..d_kv).map(|i| ((i + 3 * t) as f32 * 0.013 - 0.1).sin() * 0.4).collect()
    }).collect();

    let n_logical = ((cur_seq + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
    let pt_host: Vec<i32> = (0..n_logical).collect();

    // --- Standard kernel via paged_append_kv + paged_attention_seq1_devarg.
    let pt_dev_s = unsafe { aether_dev_alloc_i32(n_logical) };
    unsafe { aether_dev_h2d_i32(pt_host.as_ptr() as i64, pt_dev_s, n_logical); }
    let q_dev_s = unsafe { aether_dev_alloc_f32(n_heads * head_dim) };
    let k_pool_s = unsafe { aether_dev_alloc_f32(POOL_TOKENS * d_kv) };
    let v_pool_s = unsafe { aether_dev_alloc_f32(POOL_TOKENS * d_kv) };
    let out_s    = unsafe { aether_dev_alloc_f32(n_heads * head_dim) };
    let k_new_s  = unsafe { aether_dev_alloc_f32(d_kv) };
    let v_new_s  = unsafe { aether_dev_alloc_f32(d_kv) };
    let sa_dev_s = unsafe { aether_dev_alloc_i32(4) };
    unsafe { aether_dev_h2d_f32(q.as_ptr() as i64, q_dev_s, n_heads * head_dim); }
    for pos in 0..cur_seq {
        unsafe {
            aether_dev_h2d_f32(k_steps[pos as usize].as_ptr() as i64, k_new_s, d_kv);
            aether_dev_h2d_f32(v_steps[pos as usize].as_ptr() as i64, v_new_s, d_kv);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, sa_dev_s, 4);
            aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_s, v_new_s, k_pool_s, v_pool_s, pt_dev_s,
                d_kv, BLOCK_SIZE, sa_dev_s);
        }
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let final_sa = [cur_seq - 1, cur_seq, 0, 0];
    unsafe {
        aether_dev_h2d_i32(final_sa.as_ptr() as i64, sa_dev_s, 4);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_dev_s, k_pool_s, v_pool_s, pt_dev_s, out_s,
            n_heads, n_heads, head_dim, BLOCK_SIZE,
            scale, MAX_SEQ, sa_dev_s);
        aether_dev_sync();
    }
    let mut seq1_out = vec![0f32; (n_heads * head_dim) as usize];
    unsafe { aether_dev_d2h_f32(out_s, seq1_out.as_mut_ptr() as i64, n_heads * head_dim); }
    unsafe {
        aether_dev_free_f32(q_dev_s); aether_dev_free_f32(k_pool_s);
        aether_dev_free_f32(v_pool_s); aether_dev_free_f32(out_s);
        aether_dev_free_f32(k_new_s); aether_dev_free_f32(v_new_s);
        aether_dev_free_i32(pt_dev_s); aether_dev_free_i32(sa_dev_s);
    }

    // --- MLA kernel through the run_mla_pipeline helper.
    let (_cpu, mla_out) = run_mla_pipeline(n_heads, head_dim, head_dim, cur_seq);

    let max_diff = seq1_out.iter().zip(mla_out.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("[mla] degenerate (qk==v=={}) vs paged_seq1 max_diff={:.3e}", head_dim, max_diff);
    assert!(max_diff < 1e-5,
        "MLA degenerate path differs from paged_seq1 ({:.3e})", max_diff);
}

#[test]
fn mla_append_kv_writes_with_independent_strides() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n_heads = 4;
    let qk_head_dim = 96;
    let v_head_dim = 64;
    let d_k_row = n_heads * qk_head_dim;
    let d_v_row = n_heads * v_head_dim;

    let n_logical = 2i32;
    let pt_host: Vec<i32> = vec![0, 1];
    let pt_dev = unsafe { aether_dev_alloc_i32(n_logical) };
    unsafe { aether_dev_h2d_i32(pt_host.as_ptr() as i64, pt_dev, n_logical); }
    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * d_k_row) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * d_v_row) };
    let k_new  = unsafe { aether_dev_alloc_f32(d_k_row) };
    let v_new  = unsafe { aether_dev_alloc_f32(d_v_row) };
    let sa_dev = unsafe { aether_dev_alloc_i32(4) };

    // Zero out both pools.
    let zero_k = vec![0f32; (POOL_TOKENS * d_k_row) as usize];
    let zero_v = vec![0f32; (POOL_TOKENS * d_v_row) as usize];
    unsafe {
        aether_dev_h2d_f32(zero_k.as_ptr() as i64, k_pool, POOL_TOKENS * d_k_row);
        aether_dev_h2d_f32(zero_v.as_ptr() as i64, v_pool, POOL_TOKENS * d_v_row);
    }

    // Append at pos 5 (block 1, in-block 1).
    let k_payload: Vec<f32> = (0..d_k_row).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let v_payload: Vec<f32> = (0..d_v_row).map(|i| -2.0 - (i as f32) * 0.002).collect();
    unsafe {
        aether_dev_h2d_f32(k_payload.as_ptr() as i64, k_new, d_k_row);
        aether_dev_h2d_f32(v_payload.as_ptr() as i64, v_new, d_v_row);
        let sa = [5i32, 6, 0, 0];
        aether_dev_h2d_i32(sa.as_ptr() as i64, sa_dev, 4);
        let rc = aether_op_paged_append_kv_mla_devarg_f32_cuda(
            k_new, v_new, k_pool, v_pool, pt_dev,
            d_k_row, d_v_row, BLOCK_SIZE, sa_dev);
        assert_eq!(rc, 0);
        aether_dev_sync();
    }

    // Read whole pools back.
    let mut k_back = vec![0f32; (POOL_TOKENS * d_k_row) as usize];
    let mut v_back = vec![0f32; (POOL_TOKENS * d_v_row) as usize];
    unsafe {
        aether_dev_d2h_f32(k_pool, k_back.as_mut_ptr() as i64, POOL_TOKENS * d_k_row);
        aether_dev_d2h_f32(v_pool, v_back.as_mut_ptr() as i64, POOL_TOKENS * d_v_row);
    }

    // Pos 5 lives at physical row `page_table[1] * block_size + 1` = 1*4 + 1 = 5.
    let k_row_start = 5 * d_k_row as usize;
    let v_row_start = 5 * d_v_row as usize;
    let k_written = &k_back[k_row_start .. k_row_start + d_k_row as usize];
    let v_written = &v_back[v_row_start .. v_row_start + d_v_row as usize];
    assert_eq!(k_written, &k_payload[..],
        "K row at pos 5 didn't match payload");
    assert_eq!(v_written, &v_payload[..],
        "V row at pos 5 didn't match payload");

    // Adjacent rows (pos 4 and pos 6) should still be zero in both pools.
    for &other_pos in &[4usize, 6, 7] {
        let k_off = other_pos * d_k_row as usize;
        let v_off = other_pos * d_v_row as usize;
        assert!(k_back[k_off .. k_off + d_k_row as usize].iter().all(|x| *x == 0.0),
            "K row at pos {} was unexpectedly modified", other_pos);
        assert!(v_back[v_off .. v_off + d_v_row as usize].iter().all(|x| *x == 0.0),
            "V row at pos {} was unexpectedly modified", other_pos);
    }

    unsafe {
        aether_dev_free_f32(k_pool); aether_dev_free_f32(v_pool);
        aether_dev_free_f32(k_new); aether_dev_free_f32(v_new);
        aether_dev_free_i32(pt_dev); aether_dev_free_i32(sa_dev);
    }
}
