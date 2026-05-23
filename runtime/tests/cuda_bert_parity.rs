//! BERT primitive parity tests (FR-17-extra-bert-fwd).
//!
//! BERT-shape encoder-only models (the bge embedding family that OpenClaw
//! needs to swap in for Google embeddings) differ from the existing
//! Qwen-style decode path in two structural ways that the existing kernels
//! can't cover:
//!
//!   (1) **Bidirectional (non-causal) self-attention** — every query
//!       position attends to every key position with no mask.  The existing
//!       paged_attention kernels are decode-only (output for one query at
//!       cur_seq-1).
//!   (2) **Sum of three embedding tables** (word + position + token-type)
//!       as the BERT input — Qwen uses a single token-embedding lookup
//!       plus rotary later.
//!
//! Three GPU assertions:
//!
//!   1. `bert_attention_matches_cpu_bge_large_shape` — bge-large-en-v1.5
//!      shape (n_heads=16, head_dim=64, seq=32) matches a naive CPU
//!      reference to ≤ 1e-4 max abs diff.
//!   2. `bert_attention_handles_long_sequence` — seq=512 (BERT's max
//!      position) produces all-finite output of the right shape.
//!   3. `bert_embed_sum_matches_cpu` — embedding-sum kernel matches a CPU
//!      reference that performs the same three lookups + add.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32,
    aether_dev_h2d_i32, aether_dev_sync,
    aether_op_bert_self_attention_fwd_f32_cuda,
    aether_op_bert_embed_sum_f32_cuda,
};

fn cpu_bert_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq: usize, n_heads: usize, head_dim: usize, scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; seq * n_heads * head_dim];
    for h in 0..n_heads {
        for qp in 0..seq {
            let q_off = (qp * n_heads + h) * head_dim;
            // scores
            let mut scores = vec![0f32; seq];
            for t in 0..seq {
                let k_off = (t * n_heads + h) * head_dim;
                let dot: f32 = q[q_off..q_off + head_dim].iter()
                    .zip(k[k_off..k_off + head_dim].iter())
                    .map(|(a, b)| a * b).sum();
                scores[t] = dot * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32;
            for s in &mut scores { *s = (*s - mx).exp(); sum += *s; }
            for s in &mut scores { *s /= sum; }
            for t in 0..seq {
                let v_off = (t * n_heads + h) * head_dim;
                let o_off = (qp * n_heads + h) * head_dim;
                for i in 0..head_dim {
                    out[o_off + i] += scores[t] * v[v_off + i];
                }
            }
        }
    }
    out
}

fn run_bert_attention(
    seq: i32, n_heads: i32, head_dim: i32,
) -> (Vec<f32>, Vec<f32>) {
    let n_elem = (seq * n_heads * head_dim) as usize;
    let q: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.011 - 0.5).sin() * 0.4).collect();
    let k: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.013 + 0.3).cos() * 0.4).collect();
    let v: Vec<f32> = (0..n_elem).map(|i| ((i as f32) * 0.017 - 0.1).sin() * 0.4).collect();

    let q_dev = unsafe { aether_dev_alloc_f32(n_elem as i32) };
    let k_dev = unsafe { aether_dev_alloc_f32(n_elem as i32) };
    let v_dev = unsafe { aether_dev_alloc_f32(n_elem as i32) };
    let out_dev = unsafe { aether_dev_alloc_f32(n_elem as i32) };
    unsafe {
        aether_dev_h2d_f32(q.as_ptr() as i64, q_dev, n_elem as i32);
        aether_dev_h2d_f32(k.as_ptr() as i64, k_dev, n_elem as i32);
        aether_dev_h2d_f32(v.as_ptr() as i64, v_dev, n_elem as i32);
    }
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    unsafe {
        let rc = aether_op_bert_self_attention_fwd_f32_cuda(
            q_dev, k_dev, v_dev, out_dev,
            seq, n_heads, head_dim, scale);
        assert_eq!(rc, 0, "bert_self_attention_fwd rc={}", rc);
        aether_dev_sync();
    }
    let mut gpu_out = vec![0f32; n_elem];
    unsafe { aether_dev_d2h_f32(out_dev, gpu_out.as_mut_ptr() as i64, n_elem as i32); }
    unsafe {
        aether_dev_free_f32(q_dev); aether_dev_free_f32(k_dev);
        aether_dev_free_f32(v_dev); aether_dev_free_f32(out_dev);
    }

    let cpu_out = cpu_bert_attention(&q, &k, &v,
        seq as usize, n_heads as usize, head_dim as usize, scale);
    (cpu_out, gpu_out)
}

#[test]
fn bert_attention_matches_cpu_bge_large_shape() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    // bge-large-en-v1.5: hidden=1024, n_heads=16, head_dim=64.
    let n_heads = 16;
    let head_dim = 64;
    let seq = 32;
    let (cpu, gpu) = run_bert_attention(seq, n_heads, head_dim);
    assert_eq!(cpu.len(), gpu.len());
    let max_diff = cpu.iter().zip(gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let n_finite = gpu.iter().filter(|x| x.is_finite()).count();
    println!("[bert] bge-large shape n_heads={} head_dim={} seq={} -> max_diff={:.3e} finite={}/{}",
        n_heads, head_dim, seq, max_diff, n_finite, gpu.len());
    assert_eq!(n_finite, gpu.len(), "non-finite values in BERT attention");
    assert!(max_diff < 1e-4,
        "bert_self_attention diverged from CPU reference ({:.3e})", max_diff);
}

#[test]
fn bert_attention_handles_long_sequence() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    // BERT max_position_embeddings = 512.
    let n_heads = 16;
    let head_dim = 64;
    let seq = 512;
    let (cpu, gpu) = run_bert_attention(seq, n_heads, head_dim);
    let max_diff = cpu.iter().zip(gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let n_finite = gpu.iter().filter(|x| x.is_finite()).count();
    println!("[bert] seq=512 max_diff={:.3e} finite={}/{}",
        max_diff, n_finite, gpu.len());
    assert_eq!(n_finite, gpu.len(), "non-finite values at seq=512");
    // Allow slightly looser tolerance at long sequence due to fp accumulation.
    assert!(max_diff < 1e-3,
        "bert_self_attention diverged at seq=512 ({:.3e})", max_diff);
}

#[test]
fn bert_embed_sum_matches_cpu() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let vocab = 1024;
    let max_pos = 64;
    let n_types = 2;
    let d_model = 256;
    let seq = 16;

    let word_embd: Vec<f32> = (0..vocab * d_model)
        .map(|i| ((i as f32) * 0.001 - 0.5).sin() * 0.2).collect();
    let pos_embd: Vec<f32> = (0..max_pos * d_model)
        .map(|i| ((i as f32) * 0.002 + 0.1).cos() * 0.2).collect();
    let type_embd: Vec<f32> = (0..n_types * d_model)
        .map(|i| ((i as f32) * 0.003 - 0.2).sin() * 0.2).collect();
    let input_ids: Vec<i32> = (0..seq).map(|i| (i * 13 + 7) % vocab as i32).collect();
    let token_type_ids: Vec<i32> = (0..seq).map(|i| (i % n_types as i32)).collect();

    // CPU reference.
    let mut cpu_out = vec![0f32; (seq * d_model) as usize];
    for t in 0..seq as usize {
        let word = input_ids[t] as usize;
        let typ  = token_type_ids[t] as usize;
        for j in 0..d_model as usize {
            cpu_out[t * d_model as usize + j] =
                word_embd[word * d_model as usize + j]
                + pos_embd[t * d_model as usize + j]
                + type_embd[typ * d_model as usize + j];
        }
    }

    // GPU.
    let we_dev = unsafe { aether_dev_alloc_f32(vocab * d_model) };
    let pe_dev = unsafe { aether_dev_alloc_f32(max_pos * d_model) };
    let te_dev = unsafe { aether_dev_alloc_f32(n_types * d_model) };
    let ii_dev = unsafe { aether_dev_alloc_i32(seq) };
    let ti_dev = unsafe { aether_dev_alloc_i32(seq) };
    let out_dev = unsafe { aether_dev_alloc_f32(seq * d_model) };
    unsafe {
        aether_dev_h2d_f32(word_embd.as_ptr() as i64, we_dev, vocab * d_model);
        aether_dev_h2d_f32(pos_embd.as_ptr() as i64, pe_dev, max_pos * d_model);
        aether_dev_h2d_f32(type_embd.as_ptr() as i64, te_dev, n_types * d_model);
        aether_dev_h2d_i32(input_ids.as_ptr() as i64, ii_dev, seq);
        aether_dev_h2d_i32(token_type_ids.as_ptr() as i64, ti_dev, seq);
        let rc = aether_op_bert_embed_sum_f32_cuda(
            ii_dev, ti_dev, we_dev, pe_dev, te_dev, out_dev,
            seq, d_model);
        assert_eq!(rc, 0, "bert_embed_sum rc={}", rc);
        aether_dev_sync();
    }
    let mut gpu_out = vec![0f32; (seq * d_model) as usize];
    unsafe { aether_dev_d2h_f32(out_dev, gpu_out.as_mut_ptr() as i64, seq * d_model); }
    unsafe {
        aether_dev_free_f32(we_dev); aether_dev_free_f32(pe_dev);
        aether_dev_free_f32(te_dev);
        aether_dev_free_i32(ii_dev); aether_dev_free_i32(ti_dev);
        aether_dev_free_f32(out_dev);
    }

    let max_diff = cpu_out.iter().zip(gpu_out.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("[bert] embed_sum max_diff={:.3e}", max_diff);
    assert!(max_diff < 1e-6,
        "bert_embed_sum diverged from CPU reference ({:.3e})", max_diff);
}
