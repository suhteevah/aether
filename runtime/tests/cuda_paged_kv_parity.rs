//! Paged KV cache parity test (FR-19.4-extra).
//!
//! Verifies that the paged kernels (paged_append_kv_devarg +
//! paged_attention_seq1_devarg) produce bit-identical output to the
//! contiguous kernels (append_kv_devarg + attention_seq1_devarg) when
//! the page table is the identity mapping (i.e. logical block i ↔
//! physical block i, both pools have the same byte layout).
//!
//! Shape: Qwen2.5-7B (n_q_heads=28, n_kv_heads=4, head_dim=128) per layer.
//! cur_seq = 7, block_size = 4, so 2 logical blocks are used (positions 0..3
//! + 4..6).  page_table = [0, 1] (identity).
//!
//! Skipped without `--features cuda`.
//!
//! roadmap: P19.4

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_append_kv_devarg_f32_cuda,
    aether_op_attention_seq1_devarg_f32_cuda,
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
};

const N_Q_HEADS: i32 = 28;
const N_KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 128;
const D_KV: i32 = N_KV_HEADS * HEAD_DIM;
const CUR_SEQ: i32 = 7;
const MAX_SEQ: i32 = 16;
const BLOCK_SIZE: i32 = 4;
const N_BLOCKS: i32 = 4;
const POOL_TOKENS: i32 = BLOCK_SIZE * N_BLOCKS;

#[test]
fn paged_kv_matches_contiguous() {
    unsafe { assert!(aether_dev_init() == 0, "CUDA init"); }

    // ---- Build reproducible Q + K[t] + V[t] arrays on the host. ----
    let q: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| {
        ((i as f32) * 0.013 - 1.5).sin()
    }).collect();
    let mut k_steps: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + t) as f32 * 0.021 + 0.4).cos() * 0.5).collect()
    }).collect();
    let mut v_steps: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 3 * t) as f32 * 0.017 - 0.2).sin() * 0.4).collect()
    }).collect();

    // ---- Allocate device buffers. ----
    // Contiguous path: k_cache, v_cache sized for MAX_SEQ tokens.
    let q_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let k_cache = unsafe { aether_dev_alloc_f32(MAX_SEQ * D_KV) };
    let v_cache = unsafe { aether_dev_alloc_f32(MAX_SEQ * D_KV) };
    let attn_out_c = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };

    // Paged path: pool sized for N_BLOCKS * BLOCK_SIZE tokens.
    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let attn_out_p = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };

    // Page table: 2 logical blocks (4 + 3 tokens fits in 2 blocks of size 4),
    // identity mapping (logical i -> physical i).
    let n_logical_blocks = (CUR_SEQ + BLOCK_SIZE - 1) / BLOCK_SIZE; // 2
    let page_table_host: Vec<i32> = (0..n_logical_blocks).collect();
    let page_table_dev = unsafe { aether_dev_alloc_i32(n_logical_blocks) };
    unsafe {
        aether_dev_h2d_i32(page_table_host.as_ptr() as i64, page_table_dev, n_logical_blocks);
    }

    // step_args buffer: [pos, cur_seq, _, _].
    let step_args_dev = unsafe { aether_dev_alloc_i32(4) };

    // K_new / V_new staging buffers (one step at a time).
    let k_new = unsafe { aether_dev_alloc_f32(D_KV) };
    let v_new = unsafe { aether_dev_alloc_f32(D_KV) };

    // Upload Q.
    unsafe { aether_dev_h2d_f32(q.as_ptr() as i64, q_dev, N_Q_HEADS * HEAD_DIM); }

    // Append CUR_SEQ tokens to both caches.
    for pos in 0..CUR_SEQ {
        unsafe {
            aether_dev_h2d_f32(k_steps[pos as usize].as_ptr() as i64, k_new, D_KV);
            aether_dev_h2d_f32(v_steps[pos as usize].as_ptr() as i64, v_new, D_KV);
            let step_args_host = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(step_args_host.as_ptr() as i64, step_args_dev, 4);

            assert_eq!(aether_op_append_kv_devarg_f32_cuda(
                k_new, v_new, k_cache, v_cache, D_KV, step_args_dev), 0,
                "append_kv_devarg pos={}", pos);

            assert_eq!(aether_op_paged_append_kv_devarg_f32_cuda(
                k_new, v_new, k_pool, v_pool, page_table_dev,
                D_KV, BLOCK_SIZE, step_args_dev), 0,
                "paged_append_kv_devarg pos={}", pos);
        }
    }
    unsafe { aether_dev_sync(); }

    // Run attention with cur_seq = CUR_SEQ on both.
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let final_step_args = [CUR_SEQ - 1, CUR_SEQ, 0, 0];
    unsafe { aether_dev_h2d_i32(final_step_args.as_ptr() as i64, step_args_dev, 4); }

    unsafe {
        assert_eq!(aether_op_attention_seq1_devarg_f32_cuda(
            q_dev, k_cache, v_cache, attn_out_c,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, scale, MAX_SEQ, step_args_dev), 0,
            "attention_seq1_devarg (contiguous)");

        assert_eq!(aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_dev, k_pool, v_pool, page_table_dev, attn_out_p,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev), 0,
            "paged_attention_seq1_devarg");

        aether_dev_sync();
    }

    // D2H both outputs.
    let mut out_c = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut out_p = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    unsafe {
        aether_dev_d2h_f32(attn_out_c, out_c.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        aether_dev_d2h_f32(attn_out_p, out_p.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
    }

    // Compare.  Should be bit-identical (same memory access pattern under
    // identity mapping) or at worst very close (single-thread accumulation
    // order is identical).
    let mut max_abs_diff = 0.0f32;
    let mut max_rel_diff = 0.0f32;
    for i in 0..out_c.len() {
        let d = (out_c[i] - out_p[i]).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        let mag = out_c[i].abs().max(out_p[i].abs()).max(1e-6);
        let r = d / mag;
        if r > max_rel_diff { max_rel_diff = r; }
    }
    println!("[paged-kv parity] max_abs_diff = {:.3e}, max_rel_diff = {:.3e}",
        max_abs_diff, max_rel_diff);
    assert!(max_abs_diff < 1e-5, "paged kernels diverged from contiguous (max_abs_diff = {})", max_abs_diff);

    // Cleanup.
    unsafe {
        aether_dev_free_f32(q_dev);
        aether_dev_free_f32(k_cache);
        aether_dev_free_f32(v_cache);
        aether_dev_free_f32(attn_out_c);
        aether_dev_free_f32(k_pool);
        aether_dev_free_f32(v_pool);
        aether_dev_free_f32(attn_out_p);
        aether_dev_free_f32(k_new);
        aether_dev_free_f32(v_new);
        aether_dev_free_i32(page_table_dev);
        aether_dev_free_i32(step_args_dev);
    }
}

/// Sanity test: with a NON-identity page table that permutes logical → physical,
/// the paged path should produce the same result IF we write to the permuted
/// physical layout (i.e. the page table is the only level of indirection).
#[test]
fn paged_kv_permuted_page_table() {
    unsafe { assert!(aether_dev_init() == 0, "CUDA init"); }

    // Use a permuted page table: logical block 0 -> physical 2, logical 1 -> physical 0.
    // Tokens still appended at logical positions 0..6; should land at physical
    // rows: pos 0..3 in physical block 2 (= pool rows 8..11),
    //       pos 4..6 in physical block 0 (= pool rows 0..2).
    // Attention should still find them via the page table and produce the
    // same Q·K output.
    let q: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| {
        ((i as f32) * 0.013 - 1.5).sin()
    }).collect();
    let mut k_steps: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + t) as f32 * 0.021 + 0.4).cos() * 0.5).collect()
    }).collect();
    let mut v_steps: Vec<Vec<f32>> = (0..CUR_SEQ).map(|t| {
        (0..D_KV).map(|i| ((i + 3 * t) as f32 * 0.017 - 0.2).sin() * 0.4).collect()
    }).collect();

    let q_dev = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };
    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let attn_out_a = unsafe { aether_dev_alloc_f32(N_Q_HEADS * HEAD_DIM) };

    // Two page tables: identity and permuted [2, 0].
    let page_table_id: Vec<i32> = vec![0, 1];
    let page_table_perm: Vec<i32> = vec![2, 0];

    // Run with identity.
    let pt_id_dev = unsafe { aether_dev_alloc_i32(2) };
    let pt_perm_dev = unsafe { aether_dev_alloc_i32(2) };
    unsafe {
        aether_dev_h2d_i32(page_table_id.as_ptr() as i64, pt_id_dev, 2);
        aether_dev_h2d_i32(page_table_perm.as_ptr() as i64, pt_perm_dev, 2);
    }
    let step_args_dev = unsafe { aether_dev_alloc_i32(4) };
    let k_new = unsafe { aether_dev_alloc_f32(D_KV) };
    let v_new = unsafe { aether_dev_alloc_f32(D_KV) };
    unsafe { aether_dev_h2d_f32(q.as_ptr() as i64, q_dev, N_Q_HEADS * HEAD_DIM); }
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();

    // Run twice — once with identity table (writes 0..6 contiguously),
    // once with permuted table (writes 0..3 -> phys block 2, 4..6 -> phys block 0).
    let mut out_id = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    let mut out_perm = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];

    for (label, pt_dev, out_buf) in [
        ("identity", pt_id_dev, &mut out_id),
        ("permuted", pt_perm_dev, &mut out_perm),
    ] {
        // Zero the pool first.
        let zero = vec![0f32; (POOL_TOKENS * D_KV) as usize];
        unsafe {
            aether_dev_h2d_f32(zero.as_ptr() as i64, k_pool, POOL_TOKENS * D_KV);
            aether_dev_h2d_f32(zero.as_ptr() as i64, v_pool, POOL_TOKENS * D_KV);
        }
        for pos in 0..CUR_SEQ {
            unsafe {
                aether_dev_h2d_f32(k_steps[pos as usize].as_ptr() as i64, k_new, D_KV);
                aether_dev_h2d_f32(v_steps[pos as usize].as_ptr() as i64, v_new, D_KV);
                let sa = [pos, pos + 1, 0, 0];
                aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
                assert_eq!(aether_op_paged_append_kv_devarg_f32_cuda(
                    k_new, v_new, k_pool, v_pool, pt_dev,
                    D_KV, BLOCK_SIZE, step_args_dev), 0, "{} append pos={}", label, pos);
            }
        }
        unsafe {
            let sa = [CUR_SEQ - 1, CUR_SEQ, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
            assert_eq!(aether_op_paged_attention_seq1_devarg_f32_cuda(
                q_dev, k_pool, v_pool, pt_dev, attn_out_a,
                N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev),
                0, "{} attention", label);
            aether_dev_sync();
            aether_dev_d2h_f32(attn_out_a, out_buf.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
        }
    }

    // The two outputs must match: the page-table indirection is the only
    // structural difference and the kernel walks it correctly.
    let mut max_abs = 0.0f32;
    for i in 0..out_id.len() {
        let d = (out_id[i] - out_perm[i]).abs();
        if d > max_abs { max_abs = d; }
    }
    println!("[paged-kv permuted] max_abs_diff(id vs perm) = {:.3e}", max_abs);
    assert!(max_abs < 1e-5,
        "permuted page table changed attention output (max_abs={}) — kernel ignoring the page table?",
        max_abs);

    unsafe {
        aether_dev_free_f32(q_dev);
        aether_dev_free_f32(k_pool);
        aether_dev_free_f32(v_pool);
        aether_dev_free_f32(attn_out_a);
        aether_dev_free_f32(k_new);
        aether_dev_free_f32(v_new);
        aether_dev_free_i32(pt_id_dev);
        aether_dev_free_i32(pt_perm_dev);
        aether_dev_free_i32(step_args_dev);
    }
}
