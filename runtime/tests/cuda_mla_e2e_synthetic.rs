//! End-to-end MLA forward parity (FR-17-extra-mla-fwd, finishing).
//!
//! Chains all five MLA glue kernels (split_kv_a, assemble_k, extract_v,
//! rope_q_partial, rope_k_shared) plus the previously-witnessed
//! paged_attention_mla / paged_append_kv_mla / matmul to run a complete
//! DeepSeek-V2-style MLA attention step on synthetic F32 weights and assert
//! parity to a naive CPU reference.
//!
//! Shape mirrors DeepSeek-V2-Lite per-block:
//!   n_heads = 16, d_model = 2048, kv_lora_rank = 512,
//!   qk_nope_head_dim = 128, qk_rope_head_dim = 64 (qk_head_dim = 192),
//!   v_head_dim = 128
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_matmul_nt_f32_cuda,
    aether_op_mla_split_kv_a_f32_cuda,
    aether_op_mla_assemble_k_f32_cuda,
    aether_op_mla_extract_v_f32_cuda,
    aether_op_mla_rope_q_partial_f32_cuda,
    aether_op_mla_rope_k_shared_f32_cuda,
    aether_op_paged_append_kv_mla_devarg_f32_cuda,
    aether_op_paged_attention_mla_devarg_f32_cuda,
};

// ---------- CPU reference ----------

fn rms_norm_cpu(x: &[f32], gamma: &[f32], eps: f32, d: usize) -> Vec<f32> {
    // RMS norm: y[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
    assert_eq!(x.len() % d, 0);
    let rows = x.len() / d;
    let mut out = vec![0f32; x.len()];
    for r in 0..rows {
        let row = &x[r * d .. (r + 1) * d];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for j in 0..d {
            out[r * d + j] = row[j] * inv * gamma[j];
        }
    }
    out
}

/// `out[m,n] = a[m,k] @ b[n,k]^T` with row-major buffers, B laid out [n,k].
fn matmul_nt_cpu(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += a[i * k + kk] * b[j * k + kk];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn partial_rope_cpu(
    x: &mut [f32],
    n_heads: usize, head_dim: usize, nope_dim: usize,
    base: f32, pos: f32,
) {
    let rope_dim = head_dim - nope_dim;
    let hd_half = rope_dim / 2;
    for h in 0..n_heads {
        let off = h * head_dim + nope_dim;
        for i in 0..hd_half {
            let exp = -2.0f32 * (i as f32) / (rope_dim as f32);
            let theta = pos * base.powf(exp);
            let c = theta.cos();
            let s = theta.sin();
            let x0 = x[off + i];
            let x1 = x[off + i + hd_half];
            x[off + i] = x0 * c - x1 * s;
            x[off + i + hd_half] = x0 * s + x1 * c;
        }
    }
}

fn partial_rope_shared_cpu(k_rope: &mut [f32], base: f32, pos: f32) {
    let d = k_rope.len();
    let hd_half = d / 2;
    for i in 0..hd_half {
        let exp = -2.0f32 * (i as f32) / (d as f32);
        let theta = pos * base.powf(exp);
        let c = theta.cos();
        let s = theta.sin();
        let x0 = k_rope[i];
        let x1 = k_rope[i + hd_half];
        k_rope[i] = x0 * c - x1 * s;
        k_rope[i + hd_half] = x0 * s + x1 * c;
    }
}

fn mla_attention_cpu(
    q: &[f32], k_hist: &[Vec<f32>], v_hist: &[Vec<f32>],
    n_heads: usize, qk_head_dim: usize, v_head_dim: usize, scale: f32,
) -> Vec<f32> {
    let seq = k_hist.len();
    let mut out = vec![0f32; n_heads * v_head_dim];
    for h in 0..n_heads {
        let q_off = h * qk_head_dim;
        let q_h = &q[q_off .. q_off + qk_head_dim];
        let mut scores = vec![0f32; seq];
        for t in 0..seq {
            let k_off = h * qk_head_dim;
            let k_th = &k_hist[t][k_off .. k_off + qk_head_dim];
            let dot: f32 = q_h.iter().zip(k_th.iter()).map(|(a, b)| a * b).sum();
            scores[t] = dot * scale;
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
        for s in &mut scores { *s /= sum; }
        for t in 0..seq {
            let v_off = h * v_head_dim;
            let v_th = &v_hist[t][v_off .. v_off + v_head_dim];
            for i in 0..v_head_dim {
                out[h * v_head_dim + i] += scores[t] * v_th[i];
            }
        }
    }
    out
}

fn cpu_mla_forward_step(
    x: &[f32], step_pos: i32,
    // Weights
    attn_norm: &[f32],
    w_kv_a: &[f32],            // row-major [kv_lora_rank+qk_rope, d_model]
    kv_a_norm: &[f32],         // RMSnorm gamma on c_kv [kv_lora_rank]
    w_kv_b: &[f32],            // row-major [n_heads*(qk_nope+v_head), kv_lora_rank]
    w_q: &[f32],               // row-major [n_heads*qk_head_dim, d_model]
    // History (K/V for previous steps)
    k_hist: &[Vec<f32>],       // each [n_heads * qk_head_dim]
    v_hist: &[Vec<f32>],       // each [n_heads * v_head_dim]
    // Shape
    d_model: usize, n_heads: usize, kv_lora_rank: usize,
    qk_nope_head_dim: usize, qk_rope_head_dim: usize, v_head_dim: usize,
    rope_base: f32, norm_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Returns (attn_out, k_row_this_step, v_row_this_step).
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let pos = step_pos as f32;

    // 1. RMSNorm on x → x_norm
    let x_norm = rms_norm_cpu(x, attn_norm, norm_eps, d_model);

    // 2. kv_a = x_norm @ w_kv_a^T  → [kv_lora_rank + qk_rope]
    let kv_a = matmul_nt_cpu(&x_norm, w_kv_a, 1, d_model, kv_lora_rank + qk_rope_head_dim);

    // 3. Split kv_a → c_kv, k_rope_shared
    let c_kv: Vec<f32> = kv_a[..kv_lora_rank].to_vec();
    let mut k_rope: Vec<f32> = kv_a[kv_lora_rank..].to_vec();

    // 4. RMSNorm on c_kv with kv_a_norm gamma
    let c_kv_normed = rms_norm_cpu(&c_kv, kv_a_norm, norm_eps, kv_lora_rank);

    // 5. kv_b = c_kv_normed @ w_kv_b^T  → [n_heads * (qk_nope + v_head)]
    let kv_b = matmul_nt_cpu(&c_kv_normed, w_kv_b, 1, kv_lora_rank,
        n_heads * (qk_nope_head_dim + v_head_dim));

    // 6. Extract V from kv_b → [n_heads * v_head_dim]
    let mut v_row = vec![0f32; n_heads * v_head_dim];
    let kv_b_stride = qk_nope_head_dim + v_head_dim;
    for h in 0..n_heads {
        for j in 0..v_head_dim {
            v_row[h * v_head_dim + j] = kv_b[h * kv_b_stride + qk_nope_head_dim + j];
        }
    }

    // 7. RoPE on shared k_rope
    partial_rope_shared_cpu(&mut k_rope, rope_base, pos);

    // 8. Assemble K row [n_heads * qk_head_dim]
    let mut k_row = vec![0f32; n_heads * qk_head_dim];
    for h in 0..n_heads {
        for j in 0..qk_nope_head_dim {
            k_row[h * qk_head_dim + j] = kv_b[h * kv_b_stride + j];
        }
        for j in 0..qk_rope_head_dim {
            k_row[h * qk_head_dim + qk_nope_head_dim + j] = k_rope[j];
        }
    }

    // 9. Q = x_norm @ w_q^T  → [n_heads * qk_head_dim]  (direct, no q_lora here)
    let mut q = matmul_nt_cpu(&x_norm, w_q, 1, d_model, n_heads * qk_head_dim);

    // 10. RoPE on Q's partial rope sub-region
    partial_rope_cpu(&mut q, n_heads, qk_head_dim, qk_nope_head_dim, rope_base, pos);

    // 11. Compose K/V history including this step.
    let mut k_full = k_hist.to_vec();
    k_full.push(k_row.clone());
    let mut v_full = v_hist.to_vec();
    v_full.push(v_row.clone());

    // 12. Attention.
    let scale = 1.0 / (qk_head_dim as f32).sqrt();
    let attn_out = mla_attention_cpu(&q, &k_full, &v_full,
        n_heads, qk_head_dim, v_head_dim, scale);

    (attn_out, k_row, v_row)
}

// ---------- GPU pipeline ----------

fn gpu_mla_forward_step(
    x: &[f32], step_pos: i32,
    attn_norm_dev: i64,
    w_kv_a_dev: i64,
    kv_a_norm_dev: i64,
    w_kv_b_dev: i64,
    w_q_dev: i64,
    k_hist: &[Vec<f32>], v_hist: &[Vec<f32>],
    d_model: usize, n_heads: usize, kv_lora_rank: usize,
    qk_nope_head_dim: usize, qk_rope_head_dim: usize, v_head_dim: usize,
    rope_base: f32, norm_eps: f32,
    block_size: i32,
) -> Vec<f32> {
    use std::os::raw::c_int;
    use aether_rt::cuda::aether_op_rms_norm_f32_cuda;
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let d_k_row = (n_heads * qk_head_dim) as c_int;
    let d_v_row = (n_heads * v_head_dim) as c_int;

    unsafe {
        // Workspace device buffers.
        let x_dev = aether_dev_alloc_f32(d_model as c_int);
        let x_norm = aether_dev_alloc_f32(d_model as c_int);
        let kv_a = aether_dev_alloc_f32((kv_lora_rank + qk_rope_head_dim) as c_int);
        let c_kv = aether_dev_alloc_f32(kv_lora_rank as c_int);
        let c_kv_normed = aether_dev_alloc_f32(kv_lora_rank as c_int);
        let k_rope = aether_dev_alloc_f32(qk_rope_head_dim as c_int);
        let kv_b = aether_dev_alloc_f32((n_heads * (qk_nope_head_dim + v_head_dim)) as c_int);
        let k_row = aether_dev_alloc_f32(d_k_row);
        let v_row = aether_dev_alloc_f32(d_v_row);
        let q = aether_dev_alloc_f32((n_heads * qk_head_dim) as c_int);
        let attn_out = aether_dev_alloc_f32((n_heads * v_head_dim) as c_int);

        // KV pool sized for cur_seq + 1 tokens, in `block_size` chunks.
        let total_tokens = (k_hist.len() + 1) as i32;
        let n_logical_blocks = ((total_tokens + block_size - 1) / block_size).max(1);
        let pool_tokens = n_logical_blocks * block_size;
        let k_pool = aether_dev_alloc_f32(pool_tokens * d_k_row);
        let v_pool = aether_dev_alloc_f32(pool_tokens * d_v_row);
        let pt_host: Vec<i32> = (0..n_logical_blocks).collect();
        let pt_dev = aether_dev_alloc_i32(n_logical_blocks);
        aether_dev_h2d_i32(pt_host.as_ptr() as i64, pt_dev, n_logical_blocks);
        let step_args = aether_dev_alloc_i32(4);

        // Upload x.
        aether_dev_h2d_f32(x.as_ptr() as i64, x_dev, d_model as c_int);

        // 1. RMSNorm
        aether_op_rms_norm_f32_cuda(x_dev, attn_norm_dev, x_norm,
            norm_eps, 1, d_model as c_int);

        // 2. kv_a = x_norm @ w_kv_a^T
        aether_op_matmul_nt_f32_cuda(x_norm, w_kv_a_dev, kv_a,
            1, d_model as c_int, (kv_lora_rank + qk_rope_head_dim) as c_int);

        // 3. Split kv_a
        aether_op_mla_split_kv_a_f32_cuda(kv_a, c_kv, k_rope,
            kv_lora_rank as c_int, qk_rope_head_dim as c_int);

        // 4. RMSNorm c_kv
        aether_op_rms_norm_f32_cuda(c_kv, kv_a_norm_dev, c_kv_normed,
            norm_eps, 1, kv_lora_rank as c_int);

        // 5. kv_b = c_kv_normed @ w_kv_b^T
        aether_op_matmul_nt_f32_cuda(c_kv_normed, w_kv_b_dev, kv_b,
            1, kv_lora_rank as c_int,
            (n_heads * (qk_nope_head_dim + v_head_dim)) as c_int);

        // 6. Extract V
        aether_op_mla_extract_v_f32_cuda(kv_b, v_row,
            n_heads as c_int, qk_nope_head_dim as c_int, v_head_dim as c_int);

        // 7. RoPE shared k_rope (uses step_args[0] = pos).
        let sa = [step_pos, step_pos + 1, 0, 0];
        aether_dev_h2d_i32(sa.as_ptr() as i64, step_args, 4);
        aether_op_mla_rope_k_shared_f32_cuda(k_rope,
            qk_rope_head_dim as c_int, rope_base, step_args);

        // 8. Assemble K row
        aether_op_mla_assemble_k_f32_cuda(kv_b, k_rope, k_row,
            n_heads as c_int, qk_nope_head_dim as c_int,
            qk_rope_head_dim as c_int, v_head_dim as c_int);

        // 9. Q = x_norm @ w_q^T
        aether_op_matmul_nt_f32_cuda(x_norm, w_q_dev, q,
            1, d_model as c_int, (n_heads * qk_head_dim) as c_int);

        // 10. Partial RoPE on Q
        aether_op_mla_rope_q_partial_f32_cuda(q,
            n_heads as c_int, qk_head_dim as c_int,
            qk_nope_head_dim as c_int, rope_base, step_args);

        // 11. Backfill history into the pool (positions 0..k_hist.len()),
        // then append the current step's k_row + v_row at step_pos.
        for (t, (kh, vh)) in k_hist.iter().zip(v_hist.iter()).enumerate() {
            let k_tmp = aether_dev_alloc_f32(d_k_row);
            let v_tmp = aether_dev_alloc_f32(d_v_row);
            aether_dev_h2d_f32(kh.as_ptr() as i64, k_tmp, d_k_row);
            aether_dev_h2d_f32(vh.as_ptr() as i64, v_tmp, d_v_row);
            let sa_t = [t as i32, (t + 1) as i32, 0, 0];
            aether_dev_h2d_i32(sa_t.as_ptr() as i64, step_args, 4);
            aether_op_paged_append_kv_mla_devarg_f32_cuda(
                k_tmp, v_tmp, k_pool, v_pool, pt_dev,
                d_k_row, d_v_row, block_size, step_args);
            aether_dev_free_f32(k_tmp);
            aether_dev_free_f32(v_tmp);
        }
        // Append THIS step's k_row / v_row at position step_pos.
        let sa_step = [step_pos, step_pos + 1, 0, 0];
        aether_dev_h2d_i32(sa_step.as_ptr() as i64, step_args, 4);
        aether_op_paged_append_kv_mla_devarg_f32_cuda(
            k_row, v_row, k_pool, v_pool, pt_dev,
            d_k_row, d_v_row, block_size, step_args);

        // 12. Attention.  cur_seq = step_pos + 1.
        let scale = 1.0 / (qk_head_dim as f32).sqrt();
        aether_op_paged_attention_mla_devarg_f32_cuda(
            q, k_pool, v_pool, pt_dev, attn_out,
            n_heads as c_int, qk_head_dim as c_int, v_head_dim as c_int,
            block_size, scale, pool_tokens as c_int, step_args);

        aether_dev_sync();
        let mut out = vec![0f32; n_heads * v_head_dim];
        aether_dev_d2h_f32(attn_out, out.as_mut_ptr() as i64,
            (n_heads * v_head_dim) as c_int);

        for h in [x_dev, x_norm, kv_a, c_kv, c_kv_normed, k_rope, kv_b,
                  k_row, v_row, q, attn_out, k_pool, v_pool] {
            aether_dev_free_f32(h);
        }
        aether_dev_free_i32(pt_dev);
        aether_dev_free_i32(step_args);
        out
    }
}

// ---------- splitmix64 for reproducible weights ----------

struct G { s: u64 }
impl G {
    fn next_u32(&mut self) -> u32 {
        let mut z = self.s.wrapping_add(0x9E3779B97F4A7C15);
        self.s = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z >> 32) ^ z) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 / 4_294_967_296.0) * 2.0 - 1.0
    }
    fn fill(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() * scale).collect()
    }
}

#[test]
fn mla_forward_step_matches_cpu_v2_lite_shape() {
    unsafe { assert_eq!(aether_dev_init(), 0); }

    // DeepSeek-V2-Lite per-block shape.
    let d_model = 256;             // shrunken from 2048 to keep test fast
    let n_heads = 8;
    let kv_lora_rank = 64;
    let qk_nope_head_dim = 32;
    let qk_rope_head_dim = 16;
    let v_head_dim = 32;
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let rope_base = 10000.0f32;
    let norm_eps = 1e-6f32;
    let block_size = 4;

    let mut g = G { s: 7 };
    let xs0 = g.fill(d_model, 0.5);
    let xs1 = g.fill(d_model, 0.5);
    let xs2 = g.fill(d_model, 0.5);

    let attn_norm: Vec<f32> = vec![1.0; d_model];
    let sc_in = (1.0 / d_model as f32).sqrt();
    let w_kv_a = g.fill((kv_lora_rank + qk_rope_head_dim) * d_model, sc_in);
    let kv_a_norm: Vec<f32> = vec![1.0; kv_lora_rank];
    let sc_b = (1.0 / kv_lora_rank as f32).sqrt();
    let w_kv_b = g.fill(n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank, sc_b);
    let w_q = g.fill(n_heads * qk_head_dim * d_model, sc_in);

    // Upload all weights once.
    let attn_norm_dev;
    let w_kv_a_dev;
    let kv_a_norm_dev;
    let w_kv_b_dev;
    let w_q_dev;
    unsafe {
        attn_norm_dev = aether_dev_alloc_f32(d_model as i32);
        aether_dev_h2d_f32(attn_norm.as_ptr() as i64, attn_norm_dev, d_model as i32);
        w_kv_a_dev = aether_dev_alloc_f32(w_kv_a.len() as i32);
        aether_dev_h2d_f32(w_kv_a.as_ptr() as i64, w_kv_a_dev, w_kv_a.len() as i32);
        kv_a_norm_dev = aether_dev_alloc_f32(kv_lora_rank as i32);
        aether_dev_h2d_f32(kv_a_norm.as_ptr() as i64, kv_a_norm_dev, kv_lora_rank as i32);
        w_kv_b_dev = aether_dev_alloc_f32(w_kv_b.len() as i32);
        aether_dev_h2d_f32(w_kv_b.as_ptr() as i64, w_kv_b_dev, w_kv_b.len() as i32);
        w_q_dev = aether_dev_alloc_f32(w_q.len() as i32);
        aether_dev_h2d_f32(w_q.as_ptr() as i64, w_q_dev, w_q.len() as i32);
    }

    // Run three forward steps, accumulating history.
    let mut k_hist: Vec<Vec<f32>> = Vec::new();
    let mut v_hist: Vec<Vec<f32>> = Vec::new();
    let inputs = [&xs0[..], &xs1[..], &xs2[..]];

    for (step, x) in inputs.iter().enumerate() {
        let (cpu_attn, cpu_k, cpu_v) = cpu_mla_forward_step(
            x, step as i32,
            &attn_norm, &w_kv_a, &kv_a_norm, &w_kv_b, &w_q,
            &k_hist, &v_hist,
            d_model, n_heads, kv_lora_rank,
            qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            rope_base, norm_eps);

        let gpu_attn = gpu_mla_forward_step(
            x, step as i32,
            attn_norm_dev, w_kv_a_dev, kv_a_norm_dev, w_kv_b_dev, w_q_dev,
            &k_hist, &v_hist,
            d_model, n_heads, kv_lora_rank,
            qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
            rope_base, norm_eps, block_size);

        assert_eq!(cpu_attn.len(), gpu_attn.len());
        let max_diff = cpu_attn.iter().zip(gpu_attn.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_attn.iter().filter(|x| x.is_finite()).count();
        println!("[mla-e2e] step {} max_diff = {:.3e}  finite = {}/{}",
            step, max_diff, n_finite, gpu_attn.len());
        assert_eq!(n_finite, gpu_attn.len(), "non-finite at step {}", step);
        assert!(max_diff < 1e-3,
            "MLA forward step {} diverged ({:.3e})", step, max_diff);
        k_hist.push(cpu_k);
        v_hist.push(cpu_v);
    }

    unsafe {
        for h in [attn_norm_dev, w_kv_a_dev, kv_a_norm_dev, w_kv_b_dev, w_q_dev] {
            aether_dev_free_f32(h);
        }
    }
}
