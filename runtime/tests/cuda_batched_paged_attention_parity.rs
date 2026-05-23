//! Batched paged attention parity test (FR-19.5-extra-deep stage 1).
//!
//! Verifies that `batched_paged_attention_seqB_devarg` (B queries × B page
//! tables in one launch) produces bit-identical output to running
//! `paged_attention_seq1_devarg` B times sequentially.
//!
//! With B=2 and DIFFERENT page tables per request (different physical block
//! mappings inside the same shared pool), the batched kernel must:
//!   1. Read each request's K/V from the correct physical blocks (via its
//!      row of `page_table_batch`).
//!   2. Compute Q · K independently per request.
//!   3. Write its result to the right slot of `attn_out_batch`.
//!
//! Passing this test proves the foundation for BatchedQwenSession: the
//! attention kernel can fuse B concurrent requests into one launch.
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
    aether_op_batched_paged_attention_seqB_devarg_f32_cuda,
    aether_op_batched_paged_append_kv_seqB_devarg_f32_cuda,
};

const N_Q_HEADS: i32 = 28;
const N_KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 128;
const D_KV: i32 = N_KV_HEADS * HEAD_DIM;
const CUR_SEQ: i32 = 7;
const MAX_SEQ: i32 = 16;
const BLOCK_SIZE: i32 = 4;
const N_BLOCKS: i32 = 8;
const POOL_TOKENS: i32 = BLOCK_SIZE * N_BLOCKS;
const BATCH: i32 = 2;
const PAGE_TABLE_STRIDE: i32 = 4; // logical blocks per request

#[test]
fn batched_paged_attention_matches_b_sequential() {
    unsafe { assert!(aether_dev_init() == 0, "CUDA init"); }

    // Two distinct (Q, K, V) input sets — one per request.  Different RNG
    // seeds so any mistaken cross-request mixing in the batched kernel
    // would diverge the output.
    let q_a: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| ((i as f32) * 0.013 - 1.5).sin()).collect();
    let q_b: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| ((i as f32) * 0.029 + 0.7).cos()).collect();
    let k_steps_a: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + t) as f32 * 0.021 + 0.4).cos() * 0.5).collect()
    }).collect();
    let v_steps_a: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 3 * t) as f32 * 0.017 - 0.2).sin() * 0.4).collect()
    }).collect();
    let k_steps_b: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 2 * t) as f32 * 0.031 - 0.3).cos() * 0.45).collect()
    }).collect();
    let v_steps_b: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 5 * t) as f32 * 0.019 + 0.1).sin() * 0.35).collect()
    }).collect();

    // ONE shared K/V pool sized for both requests' blocks.
    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };

    // Request A uses physical blocks [0, 1]; request B uses [3, 5].
    // Distinct mappings stress the batched kernel's per-request indexing.
    let pt_a: Vec<i32> = vec![0, 1, -1, -1];
    let pt_b: Vec<i32> = vec![3, 5, -1, -1];
    let pt_a_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    let pt_b_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    unsafe {
        aether_dev_h2d_i32(pt_a.as_ptr() as i64, pt_a_dev, PAGE_TABLE_STRIDE);
        aether_dev_h2d_i32(pt_b.as_ptr() as i64, pt_b_dev, PAGE_TABLE_STRIDE);
    }

    // Batched page table: row-major concatenation of A's then B's tables.
    let pt_batch: Vec<i32> = pt_a.iter().chain(pt_b.iter()).copied().collect();
    let pt_batch_dev = unsafe { aether_dev_alloc_i32(BATCH * PAGE_TABLE_STRIDE) };
    unsafe { aether_dev_h2d_i32(pt_batch.as_ptr() as i64, pt_batch_dev, BATCH * PAGE_TABLE_STRIDE); }

    // Device staging.
    let q_a_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let q_b_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let q_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * N_Q_HEADS * HEAD_DIM) };
    let attn_a_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let attn_b_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let attn_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * N_Q_HEADS * HEAD_DIM) };
    let step_args_dev = unsafe { aether_dev_alloc_i32(4) };
    let k_new_dev = unsafe { aether_dev_alloc_f32(D_KV) };
    let v_new_dev = unsafe { aether_dev_alloc_f32(D_KV) };

    unsafe {
        aether_dev_h2d_f32(q_a.as_ptr() as i64, q_a_dev, N_Q_HEADS * HEAD_DIM);
        aether_dev_h2d_f32(q_b.as_ptr() as i64, q_b_dev, N_Q_HEADS * HEAD_DIM);
        // Q_batch row 0 = q_a, row 1 = q_b
        let mut q_batch_host = Vec::with_capacity((BATCH * N_Q_HEADS * HEAD_DIM) as usize);
        q_batch_host.extend_from_slice(&q_a);
        q_batch_host.extend_from_slice(&q_b);
        aether_dev_h2d_f32(q_batch_host.as_ptr() as i64, q_batch_dev, BATCH * N_Q_HEADS * HEAD_DIM);
    }

    // Populate pool: write A's K/V into blocks [0, 1], B's into [3, 5].
    // Use paged_append_kv to do this (its `page_table` arg is the per-request table).
    for pos in 0..CUR_SEQ {
        unsafe {
            aether_dev_h2d_f32(k_steps_a[pos as usize].as_ptr() as i64, k_new_dev, D_KV);
            aether_dev_h2d_f32(v_steps_a[pos as usize].as_ptr() as i64, v_new_dev, D_KV);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
            assert_eq!(0, aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_dev, v_new_dev, k_pool, v_pool, pt_a_dev,
                D_KV, BLOCK_SIZE, step_args_dev), "A append pos={}", pos);

            aether_dev_h2d_f32(k_steps_b[pos as usize].as_ptr() as i64, k_new_dev, D_KV);
            aether_dev_h2d_f32(v_steps_b[pos as usize].as_ptr() as i64, v_new_dev, D_KV);
            assert_eq!(0, aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_dev, v_new_dev, k_pool, v_pool, pt_b_dev,
                D_KV, BLOCK_SIZE, step_args_dev), "B append pos={}", pos);
        }
    }
    unsafe { aether_dev_sync(); }

    // Reference: 2 sequential paged_attention_seq1 calls.
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let final_sa = [CUR_SEQ - 1, CUR_SEQ, 0, 0];
    unsafe { aether_dev_h2d_i32(final_sa.as_ptr() as i64, step_args_dev, 4); }
    unsafe {
        assert_eq!(0, aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_a_dev, k_pool, v_pool, pt_a_dev, attn_a_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev));
        assert_eq!(0, aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_b_dev, k_pool, v_pool, pt_b_dev, attn_b_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev));
        aether_dev_sync();
    }

    // Candidate: one batched call.
    unsafe {
        assert_eq!(0, aether_op_batched_paged_attention_seqB_devarg_f32_cuda(
            q_batch_dev, k_pool, v_pool, pt_batch_dev, attn_batch_dev,
            BATCH,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
            PAGE_TABLE_STRIDE,
            scale, MAX_SEQ, step_args_dev));
        aether_dev_sync();
    }

    let mut ref_a = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut ref_b = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut batch_out = vec![0f32; (BATCH * N_Q_HEADS * HEAD_DIM) as usize];
    unsafe {
        aether_dev_d2h_f32(attn_a_dev, ref_a.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        aether_dev_d2h_f32(attn_b_dev, ref_b.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        aether_dev_d2h_f32(attn_batch_dev, batch_out.as_mut_ptr() as i64, BATCH * N_Q_HEADS * HEAD_DIM);
    }
    let cand_a = &batch_out[..(N_Q_HEADS * HEAD_DIM) as usize];
    let cand_b = &batch_out[(N_Q_HEADS * HEAD_DIM) as usize..];

    let mut max_abs_a = 0.0f32; let mut max_abs_b = 0.0f32;
    for i in 0..ref_a.len() {
        let d = (cand_a[i] - ref_a[i]).abs(); if d > max_abs_a { max_abs_a = d; }
        let d = (cand_b[i] - ref_b[i]).abs(); if d > max_abs_b { max_abs_b = d; }
    }
    println!("[batched-attn] req A max_abs_diff = {:.3e}", max_abs_a);
    println!("[batched-attn] req B max_abs_diff = {:.3e}", max_abs_b);
    assert!(max_abs_a < 1e-5, "request A diverged from sequential paged_attention_seq1 (max_abs={})", max_abs_a);
    assert!(max_abs_b < 1e-5, "request B diverged from sequential paged_attention_seq1 (max_abs={})", max_abs_b);

    unsafe {
        aether_dev_free_f32(k_pool); aether_dev_free_f32(v_pool);
        aether_dev_free_f32(q_a_dev); aether_dev_free_f32(q_b_dev); aether_dev_free_f32(q_batch_dev);
        aether_dev_free_f32(attn_a_dev); aether_dev_free_f32(attn_b_dev); aether_dev_free_f32(attn_batch_dev);
        aether_dev_free_f32(k_new_dev); aether_dev_free_f32(v_new_dev);
        aether_dev_free_i32(pt_a_dev); aether_dev_free_i32(pt_b_dev); aether_dev_free_i32(pt_batch_dev);
        aether_dev_free_i32(step_args_dev);
    }
}

/// End-to-end batched chain: batched append_kv → batched attention.
/// Populates the pool via ONE batched_paged_append_kv_seqB call (B requests
/// writing at the same pos to their respective page tables in the shared
/// pool), then reads via batched_paged_attention_seqB.  Result must match
/// the sequential reference (paged_append_kv_devarg × B + paged_attention_seq1_devarg × B).
#[test]
fn batched_paged_append_and_attention_full_chain() {
    unsafe { assert!(aether_dev_init() == 0, "CUDA init"); }

    let q_a: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| ((i as f32) * 0.011 - 1.2).sin()).collect();
    let q_b: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| ((i as f32) * 0.023 + 0.9).cos()).collect();
    let k_steps_a: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + t) as f32 * 0.019 + 0.5).cos() * 0.4).collect()
    }).collect();
    let v_steps_a: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 4 * t) as f32 * 0.015 - 0.1).sin() * 0.35).collect()
    }).collect();
    let k_steps_b: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 2 * t) as f32 * 0.033 - 0.4).cos() * 0.4).collect()
    }).collect();
    let v_steps_b: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 6 * t) as f32 * 0.022 + 0.2).sin() * 0.3).collect()
    }).collect();

    // Reference: 2 separate sequential append+attention pipelines.
    let k_pool_ref = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool_ref = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let pt_a: Vec<i32> = vec![0, 1, -1, -1];
    let pt_b: Vec<i32> = vec![3, 5, -1, -1];
    let pt_a_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    let pt_b_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    unsafe {
        aether_dev_h2d_i32(pt_a.as_ptr() as i64, pt_a_dev, PAGE_TABLE_STRIDE);
        aether_dev_h2d_i32(pt_b.as_ptr() as i64, pt_b_dev, PAGE_TABLE_STRIDE);
    }
    let step_args_dev = unsafe { aether_dev_alloc_i32(4) };
    let k_new_dev = unsafe { aether_dev_alloc_f32(D_KV) };
    let v_new_dev = unsafe { aether_dev_alloc_f32(D_KV) };
    let q_a_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let q_b_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let attn_a_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let attn_b_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    unsafe {
        aether_dev_h2d_f32(q_a.as_ptr() as i64, q_a_dev, N_Q_HEADS * HEAD_DIM);
        aether_dev_h2d_f32(q_b.as_ptr() as i64, q_b_dev, N_Q_HEADS * HEAD_DIM);
    }

    for pos in 0..CUR_SEQ {
        unsafe {
            aether_dev_h2d_f32(k_steps_a[pos as usize].as_ptr() as i64, k_new_dev, D_KV);
            aether_dev_h2d_f32(v_steps_a[pos as usize].as_ptr() as i64, v_new_dev, D_KV);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
            aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_dev, v_new_dev, k_pool_ref, v_pool_ref, pt_a_dev,
                D_KV, BLOCK_SIZE, step_args_dev);
            aether_dev_h2d_f32(k_steps_b[pos as usize].as_ptr() as i64, k_new_dev, D_KV);
            aether_dev_h2d_f32(v_steps_b[pos as usize].as_ptr() as i64, v_new_dev, D_KV);
            aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_dev, v_new_dev, k_pool_ref, v_pool_ref, pt_b_dev,
                D_KV, BLOCK_SIZE, step_args_dev);
        }
    }
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    unsafe {
        let final_sa = [CUR_SEQ - 1, CUR_SEQ, 0, 0];
        aether_dev_h2d_i32(final_sa.as_ptr() as i64, step_args_dev, 4);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_a_dev, k_pool_ref, v_pool_ref, pt_a_dev, attn_a_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_b_dev, k_pool_ref, v_pool_ref, pt_b_dev, attn_b_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev);
        aether_dev_sync();
    }

    // Candidate: ONE shared pool, batched append + batched attention.
    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let pt_batch: Vec<i32> = pt_a.iter().chain(pt_b.iter()).copied().collect();
    let pt_batch_dev = unsafe { aether_dev_alloc_i32(BATCH * PAGE_TABLE_STRIDE) };
    unsafe { aether_dev_h2d_i32(pt_batch.as_ptr() as i64, pt_batch_dev, BATCH * PAGE_TABLE_STRIDE); }
    let k_new_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * D_KV) };
    let v_new_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * D_KV) };
    let q_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * N_Q_HEADS * HEAD_DIM) };
    let attn_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * N_Q_HEADS * HEAD_DIM) };
    unsafe {
        let mut q_batch_host = Vec::with_capacity((BATCH * N_Q_HEADS * HEAD_DIM) as usize);
        q_batch_host.extend_from_slice(&q_a);
        q_batch_host.extend_from_slice(&q_b);
        aether_dev_h2d_f32(q_batch_host.as_ptr() as i64, q_batch_dev, BATCH * N_Q_HEADS * HEAD_DIM);
    }

    for pos in 0..CUR_SEQ {
        unsafe {
            let mut k_host = Vec::with_capacity((BATCH * D_KV) as usize);
            k_host.extend_from_slice(&k_steps_a[pos as usize]);
            k_host.extend_from_slice(&k_steps_b[pos as usize]);
            let mut v_host = Vec::with_capacity((BATCH * D_KV) as usize);
            v_host.extend_from_slice(&v_steps_a[pos as usize]);
            v_host.extend_from_slice(&v_steps_b[pos as usize]);
            aether_dev_h2d_f32(k_host.as_ptr() as i64, k_new_batch_dev, BATCH * D_KV);
            aether_dev_h2d_f32(v_host.as_ptr() as i64, v_new_batch_dev, BATCH * D_KV);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
            assert_eq!(0, aether_op_batched_paged_append_kv_seqB_devarg_f32_cuda(
                k_new_batch_dev, v_new_batch_dev, k_pool, v_pool, pt_batch_dev,
                BATCH, D_KV, BLOCK_SIZE, PAGE_TABLE_STRIDE, step_args_dev));
        }
    }
    unsafe {
        let final_sa = [CUR_SEQ - 1, CUR_SEQ, 0, 0];
        aether_dev_h2d_i32(final_sa.as_ptr() as i64, step_args_dev, 4);
        assert_eq!(0, aether_op_batched_paged_attention_seqB_devarg_f32_cuda(
            q_batch_dev, k_pool, v_pool, pt_batch_dev, attn_batch_dev,
            BATCH, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
            PAGE_TABLE_STRIDE, scale, MAX_SEQ, step_args_dev));
        aether_dev_sync();
    }

    let mut ref_a = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut ref_b = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut cand = vec![0f32; (BATCH * N_Q_HEADS * HEAD_DIM) as usize];
    unsafe {
        aether_dev_d2h_f32(attn_a_dev, ref_a.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        aether_dev_d2h_f32(attn_b_dev, ref_b.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        aether_dev_d2h_f32(attn_batch_dev, cand.as_mut_ptr() as i64, BATCH * N_Q_HEADS * HEAD_DIM);
    }
    let cand_a = &cand[..(N_Q_HEADS * HEAD_DIM) as usize];
    let cand_b = &cand[(N_Q_HEADS * HEAD_DIM) as usize..];
    let max_a = ref_a.iter().zip(cand_a.iter()).map(|(r, c)| (r - c).abs()).fold(0f32, f32::max);
    let max_b = ref_b.iter().zip(cand_b.iter()).map(|(r, c)| (r - c).abs()).fold(0f32, f32::max);
    println!("[batched-chain] req A max_abs_diff = {:.3e}", max_a);
    println!("[batched-chain] req B max_abs_diff = {:.3e}", max_b);
    assert!(max_a < 1e-5, "req A diverged ({})", max_a);
    assert!(max_b < 1e-5, "req B diverged ({})", max_b);

    unsafe {
        aether_dev_free_f32(k_pool_ref); aether_dev_free_f32(v_pool_ref);
        aether_dev_free_f32(k_pool); aether_dev_free_f32(v_pool);
        aether_dev_free_f32(q_a_dev); aether_dev_free_f32(q_b_dev); aether_dev_free_f32(q_batch_dev);
        aether_dev_free_f32(attn_a_dev); aether_dev_free_f32(attn_b_dev); aether_dev_free_f32(attn_batch_dev);
        aether_dev_free_f32(k_new_dev); aether_dev_free_f32(v_new_dev);
        aether_dev_free_f32(k_new_batch_dev); aether_dev_free_f32(v_new_batch_dev);
        aether_dev_free_i32(pt_a_dev); aether_dev_free_i32(pt_b_dev); aether_dev_free_i32(pt_batch_dev);
        aether_dev_free_i32(step_args_dev);
    }
}
