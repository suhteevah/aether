//! Heterogeneous-position batched paged attention parity (FR-19.5-extra-deep
//! Phase 2).
//!
//! The seqB batched kernels require ALL requests to share one `cur_seq`
//! (step_args[1]) / `pos` (step_args[0]), so the continuous-batching
//! scheduler could not fuse slots that were at DIFFERENT decode positions.
//! The hetero kernels lift that: each request reads its own position from a
//! per-request i32 array.
//!
//! These tests prove the hetero kernels are correct against the already-
//! trusted single-request `paged_attention_seq1_devarg` / `paged_append_kv
//! _devarg` path:
//!
//!   1. `hetero_attention_matches_staggered_seq1` — request A at cur_seq=7
//!      and request B at cur_seq=5 (a SHORTER window) in ONE launch; each
//!      output must match its own seq1 reference with its own cur_seq.
//!   2. `hetero_append_writes_per_request_positions` — one hetero append
//!      launch writes A at pos=6 and B at pos=4 (DIFFERENT positions) into
//!      their page tables; the resulting pool rows must equal what two
//!      per-request seq1 appends at those positions produce.
//!
//! roadmap: P19.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_paged_append_kv_devarg_f32_cuda,
    aether_op_paged_attention_seq1_devarg_f32_cuda,
    aether_op_batched_paged_attention_hetero_devarg_f32_cuda,
    aether_op_batched_paged_append_kv_hetero_devarg_f32_cuda,
};

const N_Q_HEADS: i32 = 28;
const N_KV_HEADS: i32 = 4;
const HEAD_DIM: i32 = 128;
const D_KV: i32 = N_KV_HEADS * HEAD_DIM;
const CUR_SEQ_A: i32 = 7;     // request A window
const CUR_SEQ_B: i32 = 5;     // request B window — SHORTER (staggered)
const MAX_SEQ: i32 = 16;
const BLOCK_SIZE: i32 = 4;
const N_BLOCKS: i32 = 8;
const POOL_TOKENS: i32 = BLOCK_SIZE * N_BLOCKS;
const BATCH: i32 = 2;
const PAGE_TABLE_STRIDE: i32 = 4; // logical blocks per request

#[test]
fn hetero_attention_matches_staggered_seq1() {
    unsafe { assert!(aether_dev_init() == 0, "CUDA init"); }

    // Distinct Q per request.
    let q_a: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| ((i as f32) * 0.013 - 1.5).sin()).collect();
    let q_b: Vec<f32> = (0..(N_Q_HEADS * HEAD_DIM)).map(|i| ((i as f32) * 0.029 + 0.7).cos()).collect();
    // Fill both requests' K/V over the FULL CUR_SEQ_A range so the pool
    // contains data past B's window — this is exactly the case the hetero
    // kernel must get right: B must NOT attend over positions ≥ CUR_SEQ_B
    // even though valid K/V data exists at those rows.
    let k_steps_a: Vec<Vec<f32>> = (0..CUR_SEQ_A).map(|t| {
        (0..D_KV).map(|i| ((i + t) as f32 * 0.021 + 0.4).cos() * 0.5).collect()
    }).collect();
    let v_steps_a: Vec<Vec<f32>> = (0..CUR_SEQ_A).map(|t| {
        (0..D_KV).map(|i| ((i + 3 * t) as f32 * 0.017 - 0.2).sin() * 0.4).collect()
    }).collect();
    let k_steps_b: Vec<Vec<f32>> = (0..CUR_SEQ_A).map(|t| {
        (0..D_KV).map(|i| ((i + 2 * t) as f32 * 0.031 - 0.3).cos() * 0.45).collect()
    }).collect();
    let v_steps_b: Vec<Vec<f32>> = (0..CUR_SEQ_A).map(|t| {
        (0..D_KV).map(|i| ((i + 5 * t) as f32 * 0.019 + 0.1).sin() * 0.35).collect()
    }).collect();

    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };

    let pt_a: Vec<i32> = vec![0, 1, -1, -1];
    let pt_b: Vec<i32> = vec![3, 5, -1, -1];
    let pt_a_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    let pt_b_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    unsafe {
        aether_dev_h2d_i32(pt_a.as_ptr() as i64, pt_a_dev, PAGE_TABLE_STRIDE);
        aether_dev_h2d_i32(pt_b.as_ptr() as i64, pt_b_dev, PAGE_TABLE_STRIDE);
    }
    let pt_batch: Vec<i32> = pt_a.iter().chain(pt_b.iter()).copied().collect();
    let pt_batch_dev = unsafe { aether_dev_alloc_i32(BATCH * PAGE_TABLE_STRIDE) };
    unsafe { aether_dev_h2d_i32(pt_batch.as_ptr() as i64, pt_batch_dev, BATCH * PAGE_TABLE_STRIDE); }

    // Per-request cur_seq array — THE hetero input.
    let cur_seq_batch: Vec<i32> = vec![CUR_SEQ_A, CUR_SEQ_B];
    let cur_seq_dev = unsafe { aether_dev_alloc_i32(BATCH) };
    unsafe { aether_dev_h2d_i32(cur_seq_batch.as_ptr() as i64, cur_seq_dev, BATCH); }

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
        let mut q_batch_host = Vec::with_capacity((BATCH * N_Q_HEADS * HEAD_DIM) as usize);
        q_batch_host.extend_from_slice(&q_a);
        q_batch_host.extend_from_slice(&q_b);
        aether_dev_h2d_f32(q_batch_host.as_ptr() as i64, q_batch_dev, BATCH * N_Q_HEADS * HEAD_DIM);
    }

    // Fill the pool over the full CUR_SEQ_A range for BOTH requests.
    for pos in 0..CUR_SEQ_A {
        unsafe {
            aether_dev_h2d_f32(k_steps_a[pos as usize].as_ptr() as i64, k_new_dev, D_KV);
            aether_dev_h2d_f32(v_steps_a[pos as usize].as_ptr() as i64, v_new_dev, D_KV);
            let sa = [pos, pos + 1, 0, 0];
            aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
            aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_dev, v_new_dev, k_pool, v_pool, pt_a_dev,
                D_KV, BLOCK_SIZE, step_args_dev);
            aether_dev_h2d_f32(k_steps_b[pos as usize].as_ptr() as i64, k_new_dev, D_KV);
            aether_dev_h2d_f32(v_steps_b[pos as usize].as_ptr() as i64, v_new_dev, D_KV);
            aether_op_paged_append_kv_devarg_f32_cuda(
                k_new_dev, v_new_dev, k_pool, v_pool, pt_b_dev,
                D_KV, BLOCK_SIZE, step_args_dev);
        }
    }
    unsafe { aether_dev_sync(); }

    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    // Reference: seq1 for A over cur_seq=7, seq1 for B over cur_seq=5.
    unsafe {
        let sa = [CUR_SEQ_A - 1, CUR_SEQ_A, 0, 0];
        aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_a_dev, k_pool, v_pool, pt_a_dev, attn_a_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev);
        let sb = [CUR_SEQ_B - 1, CUR_SEQ_B, 0, 0];
        aether_dev_h2d_i32(sb.as_ptr() as i64, step_args_dev, 4);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_b_dev, k_pool, v_pool, pt_b_dev, attn_b_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev);
        aether_dev_sync();
    }

    // Candidate: ONE hetero launch with cur_seq_batch=[7, 5].
    unsafe {
        assert_eq!(0, aether_op_batched_paged_attention_hetero_devarg_f32_cuda(
            q_batch_dev, k_pool, v_pool, pt_batch_dev, attn_batch_dev,
            BATCH, N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE,
            PAGE_TABLE_STRIDE, scale, MAX_SEQ, cur_seq_dev));
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

    let max_a = ref_a.iter().zip(cand_a).map(|(r, c)| (r - c).abs()).fold(0f32, f32::max);
    let max_b = ref_b.iter().zip(cand_b).map(|(r, c)| (r - c).abs()).fold(0f32, f32::max);
    println!("[hetero-attn] req A (cur_seq=7) max_abs_diff = {:.3e}", max_a);
    println!("[hetero-attn] req B (cur_seq=5) max_abs_diff = {:.3e}", max_b);
    assert!(max_a < 1e-5, "req A diverged from seq1@cur_seq=7 ({})", max_a);
    assert!(max_b < 1e-5, "req B diverged from seq1@cur_seq=5 ({})", max_b);

    // Sanity: B's output must DIFFER from a full-window (cur_seq=7) seq1 —
    // otherwise the per-request cur_seq isn't actually restricting scope.
    let mut ref_b_full = vec![0f32; (N_Q_HEADS * HEAD_DIM) as usize];
    unsafe {
        let sb = [CUR_SEQ_A - 1, CUR_SEQ_A, 0, 0];
        aether_dev_h2d_i32(sb.as_ptr() as i64, step_args_dev, 4);
        aether_op_paged_attention_seq1_devarg_f32_cuda(
            q_b_dev, k_pool, v_pool, pt_b_dev, attn_b_dev,
            N_Q_HEADS, N_KV_HEADS, HEAD_DIM, BLOCK_SIZE, scale, MAX_SEQ, step_args_dev);
        aether_dev_sync();
        aether_dev_d2h_f32(attn_b_dev, ref_b_full.as_mut_ptr() as i64, N_Q_HEADS * HEAD_DIM);
    }
    let diff_full = ref_b.iter().zip(&ref_b_full).map(|(s, f)| (s - f).abs()).fold(0f32, f32::max);
    println!("[hetero-attn] req B cur_seq=5 vs cur_seq=7 diff = {:.3e} (must be > 0)", diff_full);
    assert!(diff_full > 1e-4,
        "B's cur_seq=5 output equals cur_seq=7 — scope restriction not effective");

    unsafe {
        aether_dev_free_f32(k_pool); aether_dev_free_f32(v_pool);
        aether_dev_free_f32(q_a_dev); aether_dev_free_f32(q_b_dev); aether_dev_free_f32(q_batch_dev);
        aether_dev_free_f32(attn_a_dev); aether_dev_free_f32(attn_b_dev); aether_dev_free_f32(attn_batch_dev);
        aether_dev_free_f32(k_new_dev); aether_dev_free_f32(v_new_dev);
        aether_dev_free_i32(pt_a_dev); aether_dev_free_i32(pt_b_dev); aether_dev_free_i32(pt_batch_dev);
        aether_dev_free_i32(cur_seq_dev); aether_dev_free_i32(step_args_dev);
    }
}

/// One hetero append launch writes request A at pos=6 and request B at
/// pos=4 (DIFFERENT positions) into their respective page tables in a
/// shared pool.  The resulting pool rows must equal what two per-request
/// `paged_append_kv_devarg` calls at those positions produce.
#[test]
fn hetero_append_writes_per_request_positions() {
    unsafe { assert!(aether_dev_init() == 0, "CUDA init"); }

    let pos_a: i32 = 6;
    let pos_b: i32 = 4;
    let k_a: Vec<f32> = (0..D_KV).map(|i| (i as f32 * 0.013 + 0.2).sin()).collect();
    let v_a: Vec<f32> = (0..D_KV).map(|i| (i as f32 * 0.017 - 0.4).cos()).collect();
    let k_b: Vec<f32> = (0..D_KV).map(|i| (i as f32 * 0.029 - 0.1).cos()).collect();
    let v_b: Vec<f32> = (0..D_KV).map(|i| (i as f32 * 0.023 + 0.5).sin()).collect();

    let pt_a: Vec<i32> = vec![0, 1, -1, -1];
    let pt_b: Vec<i32> = vec![3, 5, -1, -1];
    // pos=6 → logical blk 1 → A: phys 1; pos=4 → logical blk 1 → B: phys 5.
    let row_a = (pt_a[(pos_a / BLOCK_SIZE) as usize] * BLOCK_SIZE + pos_a % BLOCK_SIZE) as usize;
    let row_b = (pt_b[(pos_b / BLOCK_SIZE) as usize] * BLOCK_SIZE + pos_b % BLOCK_SIZE) as usize;

    let pt_a_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    let pt_b_dev = unsafe { aether_dev_alloc_i32(PAGE_TABLE_STRIDE) };
    let pt_batch_dev = unsafe { aether_dev_alloc_i32(BATCH * PAGE_TABLE_STRIDE) };
    let pt_batch: Vec<i32> = pt_a.iter().chain(pt_b.iter()).copied().collect();
    unsafe {
        aether_dev_h2d_i32(pt_a.as_ptr() as i64, pt_a_dev, PAGE_TABLE_STRIDE);
        aether_dev_h2d_i32(pt_b.as_ptr() as i64, pt_b_dev, PAGE_TABLE_STRIDE);
        aether_dev_h2d_i32(pt_batch.as_ptr() as i64, pt_batch_dev, BATCH * PAGE_TABLE_STRIDE);
    }

    // Reference pool via two per-request seq1 appends.
    let k_pool_ref = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool_ref = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let k_new_dev = unsafe { aether_dev_alloc_f32(D_KV) };
    let v_new_dev = unsafe { aether_dev_alloc_f32(D_KV) };
    let step_args_dev = unsafe { aether_dev_alloc_i32(4) };
    unsafe {
        aether_dev_h2d_f32(k_a.as_ptr() as i64, k_new_dev, D_KV);
        aether_dev_h2d_f32(v_a.as_ptr() as i64, v_new_dev, D_KV);
        let sa = [pos_a, pos_a + 1, 0, 0];
        aether_dev_h2d_i32(sa.as_ptr() as i64, step_args_dev, 4);
        aether_op_paged_append_kv_devarg_f32_cuda(
            k_new_dev, v_new_dev, k_pool_ref, v_pool_ref, pt_a_dev,
            D_KV, BLOCK_SIZE, step_args_dev);
        aether_dev_h2d_f32(k_b.as_ptr() as i64, k_new_dev, D_KV);
        aether_dev_h2d_f32(v_b.as_ptr() as i64, v_new_dev, D_KV);
        let sb = [pos_b, pos_b + 1, 0, 0];
        aether_dev_h2d_i32(sb.as_ptr() as i64, step_args_dev, 4);
        aether_op_paged_append_kv_devarg_f32_cuda(
            k_new_dev, v_new_dev, k_pool_ref, v_pool_ref, pt_b_dev,
            D_KV, BLOCK_SIZE, step_args_dev);
        aether_dev_sync();
    }

    // Candidate: ONE hetero append with pos_batch=[6, 4].
    let k_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let v_pool = unsafe { aether_dev_alloc_f32(POOL_TOKENS * D_KV) };
    let k_new_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * D_KV) };
    let v_new_batch_dev = unsafe { aether_dev_alloc_f32(BATCH * D_KV) };
    let pos_batch_dev = unsafe { aether_dev_alloc_i32(BATCH) };
    unsafe {
        let mut kh = Vec::with_capacity((BATCH * D_KV) as usize);
        kh.extend_from_slice(&k_a); kh.extend_from_slice(&k_b);
        let mut vh = Vec::with_capacity((BATCH * D_KV) as usize);
        vh.extend_from_slice(&v_a); vh.extend_from_slice(&v_b);
        aether_dev_h2d_f32(kh.as_ptr() as i64, k_new_batch_dev, BATCH * D_KV);
        aether_dev_h2d_f32(vh.as_ptr() as i64, v_new_batch_dev, BATCH * D_KV);
        let pos_batch: Vec<i32> = vec![pos_a, pos_b];
        aether_dev_h2d_i32(pos_batch.as_ptr() as i64, pos_batch_dev, BATCH);
        assert_eq!(0, aether_op_batched_paged_append_kv_hetero_devarg_f32_cuda(
            k_new_batch_dev, v_new_batch_dev, k_pool, v_pool, pt_batch_dev,
            BATCH, D_KV, BLOCK_SIZE, PAGE_TABLE_STRIDE, pos_batch_dev));
        aether_dev_sync();
    }

    // Compare the two written rows (A's row and B's row) between ref and cand.
    let read_row = |pool: i64, row: usize| -> Vec<f32> {
        let mut whole = vec![0f32; (POOL_TOKENS * D_KV) as usize];
        unsafe { aether_dev_d2h_f32(pool, whole.as_mut_ptr() as i64, POOL_TOKENS * D_KV); }
        whole[row * D_KV as usize..(row + 1) * D_KV as usize].to_vec()
    };
    let ref_ka = read_row(k_pool_ref, row_a);
    let ref_kb = read_row(k_pool_ref, row_b);
    let ref_va = read_row(v_pool_ref, row_a);
    let ref_vb = read_row(v_pool_ref, row_b);
    let cand_ka = read_row(k_pool, row_a);
    let cand_kb = read_row(k_pool, row_b);
    let cand_va = read_row(v_pool, row_a);
    let cand_vb = read_row(v_pool, row_b);

    let maxd = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let dka = maxd(&ref_ka, &cand_ka);
    let dkb = maxd(&ref_kb, &cand_kb);
    let dva = maxd(&ref_va, &cand_va);
    let dvb = maxd(&ref_vb, &cand_vb);
    println!("[hetero-append] K row A diff={:.3e} K row B diff={:.3e}", dka, dkb);
    println!("[hetero-append] V row A diff={:.3e} V row B diff={:.3e}", dva, dvb);
    assert_eq!(dka, 0.0, "A's K row mismatch (pos=6)");
    assert_eq!(dkb, 0.0, "B's K row mismatch (pos=4)");
    assert_eq!(dva, 0.0, "A's V row mismatch (pos=6)");
    assert_eq!(dvb, 0.0, "B's V row mismatch (pos=4)");
    // The two requests wrote to DIFFERENT physical rows.
    assert_ne!(row_a, row_b, "test setup: rows must differ to be meaningful");

    unsafe {
        aether_dev_free_f32(k_pool_ref); aether_dev_free_f32(v_pool_ref);
        aether_dev_free_f32(k_pool); aether_dev_free_f32(v_pool);
        aether_dev_free_f32(k_new_dev); aether_dev_free_f32(v_new_dev);
        aether_dev_free_f32(k_new_batch_dev); aether_dev_free_f32(v_new_batch_dev);
        aether_dev_free_i32(pt_a_dev); aether_dev_free_i32(pt_b_dev); aether_dev_free_i32(pt_batch_dev);
        aether_dev_free_i32(pos_batch_dev); aether_dev_free_i32(step_args_dev);
    }
}
