//! YaRN-aware partial RoPE parity (FR-17-extra-mla-fwd YaRN).
//!
//! Verifies the GPU `mla_rope_q_partial_yarn` and `mla_rope_k_shared_yarn`
//! kernels match a naive CPU implementation of YaRN-by-parts RoPE for
//! DeepSeek-V2-Lite parameters (s=40, log_mult=0.0707, orig_ctx=4096,
//! beta_fast=32, beta_slow=1).  Three positions exercised: 0, 1024, 16384.
//!
//! When yarn_factor == 1 (no scaling) the YaRN kernel must reduce to plain
//! RoPE — checked against the non-yarn kernel.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_i32, aether_dev_free_i32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_i32, aether_dev_sync,
    aether_op_mla_rope_q_partial_f32_cuda,
    aether_op_mla_rope_k_shared_f32_cuda,
    aether_op_mla_rope_q_partial_yarn_f32_cuda,
    aether_op_mla_rope_k_shared_yarn_f32_cuda,
};

// ----- CPU reference matching the kernel exactly -----

// Independent reference — derived from the HF DeepSeek-V2 `yarn` /
// llama.cpp `rope_yarn_corr_dims` + `rope_yarn_ramp` formulas, NOT copied from
// the kernel.  (The previous version of this file copied the kernel's buggy
// `(i_high - i_low).max(1e-3)` step verbatim, so it validated the bug against
// itself.  See `yarn_scale_factor_reference_anchors` for the hand-computed
// values this reference must reproduce.)
fn yarn_correction_dim(num_rotations: f32, head_dim: f32, base: f32, orig_ctx: f32) -> f32 {
    let two_pi = 2.0 * std::f32::consts::PI;
    head_dim * (orig_ctx / (num_rotations * two_pi)).ln() / (2.0 * base.ln())
}

/// `i` is the rotary PAIR index in `[0, head_dim/2)`.
fn yarn_scale_factor(
    i: i32, head_dim: i32, base: f32, yarn_s: f32,
    orig_ctx: f32, beta_fast: f32, beta_slow: f32,
) -> f32 {
    // corr_dim is decreasing in num_rotations → beta_fast gives the LOW end.
    let low  = yarn_correction_dim(beta_fast, head_dim as f32, base, orig_ctx)
        .floor().max(0.0);
    let high = yarn_correction_dim(beta_slow, head_dim as f32, base, orig_ctx)
        .ceil().min((head_dim - 1) as f32);
    let denom = (high - low).max(1e-3);
    let ramp = ((i as f32 - low) / denom).clamp(0.0, 1.0);
    (1.0 - ramp) + ramp / yarn_s
}

fn rope_k_shared_yarn_cpu(
    k_rope: &mut [f32],
    pos: f32, base: f32,
    yarn_s: f32, orig_ctx: f32, beta_fast: f32, beta_slow: f32,
) {
    let d = k_rope.len() as i32;
    let hd_half = d / 2;
    // llama.cpp rope_yarn cos/sin mscale (ext_factor != 0): attn_factor=1.0,
    // 1/freq_scale = yarn_s → mscale = 1 + 0.1*ln(yarn_s).
    let rope_mscale = 1.0 + 0.1 * yarn_s.ln();
    for i in 0..hd_half {
        let scale = yarn_scale_factor(i, d, base, yarn_s, orig_ctx,
            beta_fast, beta_slow);
        let exp = -2.0 * (i as f32) / (d as f32);
        let theta = pos * scale * base.powf(exp);
        let c = theta.cos() * rope_mscale;
        let s = theta.sin() * rope_mscale;
        let x0 = k_rope[i as usize];
        let x1 = k_rope[(i + hd_half) as usize];
        k_rope[i as usize] = x0 * c - x1 * s;
        k_rope[(i + hd_half) as usize] = x0 * s + x1 * c;
    }
}

fn rope_q_partial_yarn_cpu(
    q: &mut [f32], n_heads: i32, qk_head_dim: i32, nope_dim: i32,
    pos: f32, base: f32,
    yarn_s: f32, orig_ctx: f32, beta_fast: f32, beta_slow: f32,
) {
    let rope_dim = qk_head_dim - nope_dim;
    let hd_half = rope_dim / 2;
    let rope_mscale = 1.0 + 0.1 * yarn_s.ln();
    for h in 0..n_heads {
        let base_off = (h * qk_head_dim + nope_dim) as usize;
        for i in 0..hd_half {
            let scale = yarn_scale_factor(i, rope_dim, base, yarn_s,
                orig_ctx, beta_fast, beta_slow);
            let exp = -2.0 * (i as f32) / (rope_dim as f32);
            let theta = pos * scale * base.powf(exp);
            let c = theta.cos() * rope_mscale;
            let s = theta.sin() * rope_mscale;
            let x0 = q[base_off + i as usize];
            let x1 = q[base_off + (i + hd_half) as usize];
            q[base_off + i as usize] = x0 * c - x1 * s;
            q[base_off + (i + hd_half) as usize] = x0 * s + x1 * c;
        }
    }
}

fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n).map(|_| {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let u = (((z >> 32) ^ z) as u32) as f32 / 4_294_967_296.0;
        u * 2.0 - 1.0
    }).collect()
}

fn run_k_shared_gpu(
    k_in: &[f32], pos: i32, base: f32,
    yarn_s: f32, orig_ctx: f32, beta_fast: f32, beta_slow: f32,
    use_yarn: bool,
) -> Vec<f32> {
    let n = k_in.len() as i32;
    unsafe {
        let k_dev = aether_dev_alloc_f32(n);
        let sa_dev = aether_dev_alloc_i32(4);
        aether_dev_h2d_f32(k_in.as_ptr() as i64, k_dev, n);
        let sa = [pos, pos + 1, 0, 0];
        aether_dev_h2d_i32(sa.as_ptr() as i64, sa_dev, 4);
        if use_yarn {
            aether_op_mla_rope_k_shared_yarn_f32_cuda(
                k_dev, n, base, yarn_s, orig_ctx, beta_fast, beta_slow, sa_dev);
        } else {
            aether_op_mla_rope_k_shared_f32_cuda(k_dev, n, base, sa_dev);
        }
        aether_dev_sync();
        let mut out = vec![0f32; n as usize];
        aether_dev_d2h_f32(k_dev, out.as_mut_ptr() as i64, n);
        aether_dev_free_f32(k_dev);
        aether_dev_free_i32(sa_dev);
        out
    }
}

fn run_q_partial_gpu(
    q_in: &[f32], n_heads: i32, qk_head_dim: i32, nope_dim: i32, pos: i32, base: f32,
    yarn_s: f32, orig_ctx: f32, beta_fast: f32, beta_slow: f32,
    use_yarn: bool,
) -> Vec<f32> {
    let n = q_in.len() as i32;
    unsafe {
        let q_dev = aether_dev_alloc_f32(n);
        let sa_dev = aether_dev_alloc_i32(4);
        aether_dev_h2d_f32(q_in.as_ptr() as i64, q_dev, n);
        let sa = [pos, pos + 1, 0, 0];
        aether_dev_h2d_i32(sa.as_ptr() as i64, sa_dev, 4);
        if use_yarn {
            aether_op_mla_rope_q_partial_yarn_f32_cuda(q_dev,
                n_heads, qk_head_dim, nope_dim, base,
                yarn_s, orig_ctx, beta_fast, beta_slow, sa_dev);
        } else {
            aether_op_mla_rope_q_partial_f32_cuda(q_dev,
                n_heads, qk_head_dim, nope_dim, base, sa_dev);
        }
        aether_dev_sync();
        let mut out = vec![0f32; n as usize];
        aether_dev_d2h_f32(q_dev, out.as_mut_ptr() as i64, n);
        aether_dev_free_f32(q_dev);
        aether_dev_free_i32(sa_dev);
        out
    }
}

#[test]
fn yarn_scale_factor_reference_anchors() {
    // Hand-computed against the HF/llama.cpp YaRN formula for DeepSeek-V2-Lite
    // (rope_dim=64, base=1e4, s=40, orig_ctx=4096, beta_fast=32, beta_slow=1):
    //   corr_dim(32) = 10.472 -> floor -> low  = 10
    //   corr_dim(1)  = 22.514 -> ceil  -> high = 23   (denom = 13)
    //   scale(i) = (1 - ramp) + ramp/40,  ramp = clip((i - 10) / 13, 0, 1)
    // These values are independent of the kernel; if a future refactor
    // re-breaks the ramp (e.g. swaps the bounds or doubles the index), this
    // test fails before any GPU work runs.
    let (base, s, oc, bf, bs) = (10_000.0f32, 40.0f32, 4096.0f32, 32.0f32, 1.0f32);
    let sf = |i: i32| yarn_scale_factor(i, 64, base, s, oc, bf, bs);
    let cases = [
        (0i32,  1.0f32),     // high freq -> pure extrapolation
        (5,     1.0),        // still below the ramp
        (10,    1.0),        // ramp start (ramp = 0)
        (16,    0.550_0),    // mid ramp: 0.53846 + 0.46154/40
        (23,    0.025),      // ramp end (ramp = 1) -> 1/s
        (31,    0.025),      // clamped past the end
    ];
    for (i, want) in cases {
        let got = sf(i);
        assert!((got - want).abs() < 1e-3,
            "yarn scale_factor(i={i}) = {got:.6}, want {want:.6}");
    }
    // Monotone non-increasing across the band (high freq -> low freq).
    let mut prev = f32::INFINITY;
    for i in 0..32 {
        let v = sf(i);
        assert!(v <= prev + 1e-6, "scale_factor not monotone at i={i}: {v} > {prev}");
        prev = v;
    }
}

#[test]
fn yarn_k_shared_matches_cpu_v2_lite_params() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let qk_rope = 64;
    let base = 10_000.0f32;
    // DeepSeek-V2-Lite YaRN params
    let yarn_s = 40.0f32;
    let orig_ctx = 4096.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;

    for &pos in &[0i32, 1024, 16384] {
        let k_in = deterministic(qk_rope as usize, 7);
        let gpu = run_k_shared_gpu(&k_in, pos, base,
            yarn_s, orig_ctx, beta_fast, beta_slow, true);
        let mut cpu = k_in.clone();
        rope_k_shared_yarn_cpu(&mut cpu, pos as f32, base,
            yarn_s, orig_ctx, beta_fast, beta_slow);
        let max_diff = cpu.iter().zip(gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu.iter().filter(|x| x.is_finite()).count();
        println!("[yarn-k] pos={} max_diff={:.3e} finite={}/{}",
            pos, max_diff, n_finite, gpu.len());
        assert_eq!(n_finite, gpu.len(), "non-finite at pos {}", pos);
        assert!(max_diff < 1e-5,
            "YaRN k_shared diverged at pos {} ({:.3e})", pos, max_diff);
    }
}

#[test]
fn yarn_q_partial_matches_cpu_v2_lite_params() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n_heads = 16;
    let qk_head_dim = 192;
    let nope_dim = 128;
    let base = 10_000.0f32;
    let yarn_s = 40.0f32;
    let orig_ctx = 4096.0f32;
    let beta_fast = 32.0f32;
    let beta_slow = 1.0f32;

    for &pos in &[0i32, 1024, 16384] {
        let q_in = deterministic((n_heads * qk_head_dim) as usize, 11);
        let gpu = run_q_partial_gpu(&q_in,
            n_heads, qk_head_dim, nope_dim, pos, base,
            yarn_s, orig_ctx, beta_fast, beta_slow, true);
        let mut cpu = q_in.clone();
        rope_q_partial_yarn_cpu(&mut cpu,
            n_heads, qk_head_dim, nope_dim, pos as f32, base,
            yarn_s, orig_ctx, beta_fast, beta_slow);
        let max_diff = cpu.iter().zip(gpu.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu.iter().filter(|x| x.is_finite()).count();
        println!("[yarn-q] pos={} max_diff={:.3e} finite={}/{}",
            pos, max_diff, n_finite, gpu.len());
        assert_eq!(n_finite, gpu.len(), "non-finite at pos {}", pos);
        assert!(max_diff < 1e-4,
            "YaRN q_partial diverged at pos {} ({:.3e})", pos, max_diff);
    }
}

#[test]
fn yarn_reduces_to_plain_rope_at_factor_1() {
    // When yarn_factor == 1, the scale factor must be 1.0 for every i, so
    // the yarn-aware kernel must produce the SAME output as the non-yarn
    // kernel — confirming the fast-path identity for non-YaRN models.
    //
    // NB: There's a subtle numerical subtlety — the YaRN kernel runs
    // logf/expf/divisions to compute scale=1 even at factor=1, while the
    // plain kernel skips them entirely.  Tolerance accordingly.
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let qk_rope = 64;
    let base = 10_000.0f32;
    let k_in = deterministic(qk_rope as usize, 13);
    for &pos in &[0i32, 100, 1000] {
        let gpu_plain = run_k_shared_gpu(&k_in, pos, base,
            1.0, 4096.0, 32.0, 1.0, false);
        let gpu_yarn = run_k_shared_gpu(&k_in, pos, base,
            1.0, 4096.0, 32.0, 1.0, true);
        let max_diff = gpu_plain.iter().zip(gpu_yarn.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[yarn-identity] pos={} plain vs yarn(s=1) max_diff={:.3e}",
            pos, max_diff);
        assert!(max_diff < 1e-4,
            "YaRN at factor=1 differs from plain RoPE at pos {} ({:.3e})", pos, max_diff);
    }
}
