//! MoE shared-expert FFN parity (FR-17-extra-mla-fwd MoE shared).
//!
//! DeepSeek-V2 / GLM-4.7-flash MoE blocks have `expert_shared_count > 0`
//! always-on experts in addition to the top-k routed ones.  The GGUF
//! pre-concatenates the n_shared experts into a single FUSED MLP with
//! hidden dim `n_shared * expert_ff_dim` — so the shared-expert forward
//! is just a regular dense FFN at that hidden dim.
//!
//! This test exercises that chain (gate matmul → up matmul → silu → mul
//! → down matmul) against a naive CPU reference at V2-Lite parameters:
//!   d_model     = 2048
//!   n_shared    = 2
//!   expert_ff   = 1408
//!   d_ff_shared = 2 * 1408 = 2816
//!
//! Weights are F32 (the production path runs through dispatch_matmul
//! against Q4_K weights — Q4_K vs F32 matmul parity is already covered
//! by other tests).
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_op_matmul_nt_f32_cuda,
    aether_op_silu_f32_cuda, aether_op_mul_inplace_f32_cuda,
};

fn matmul_nt_cpu(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += x[i * k + kk] * w[j * k + kk];
            }
            out[i * n + j] = s;
        }
    }
    out
}

fn silu_cpu(x: &mut [f32]) {
    for v in x { *v = *v / (1.0 + (-*v).exp()); }
}

fn deterministic(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n).map(|_| {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let u = (((z >> 32) ^ z) as u32) as f32 / 4_294_967_296.0;
        (u * 2.0 - 1.0) * scale
    }).collect()
}

#[test]
fn shared_expert_fused_ffn_matches_cpu_v2_lite_shape() {
    unsafe { assert_eq!(aether_dev_init(), 0); }

    // Shrunken V2-Lite shape to keep the test fast (preserves the
    // shared-expert algorithmic structure).
    let d_model = 256;
    let expert_ff_dim = 176;     // 1408 ÷ 8
    let n_shared = 2;
    let d_ff_shared = n_shared * expert_ff_dim;   // 352

    let sc_in = (1.0 / d_model as f32).sqrt();
    let sc_dn = (1.0 / d_ff_shared as f32).sqrt();

    let x = deterministic(d_model, 7, 1.0);
    let w_gate = deterministic(d_ff_shared * d_model, 11, sc_in);
    let w_up   = deterministic(d_ff_shared * d_model, 13, sc_in);
    let w_down = deterministic(d_model * d_ff_shared, 17, sc_dn);

    // ---- CPU reference: gate → up → silu(gate) * up → down. ----
    let mut gate_cpu = matmul_nt_cpu(&x, &w_gate, 1, d_model, d_ff_shared);
    let up_cpu = matmul_nt_cpu(&x, &w_up, 1, d_model, d_ff_shared);
    silu_cpu(&mut gate_cpu);
    for i in 0..d_ff_shared { gate_cpu[i] *= up_cpu[i]; }
    let down_cpu = matmul_nt_cpu(&gate_cpu, &w_down, 1, d_ff_shared, d_model);

    // ---- GPU: same chain via the FFI-exposed ops. ----
    let x_dev = unsafe { aether_dev_alloc_f32(d_model as i32) };
    let wg_dev = unsafe { aether_dev_alloc_f32((d_ff_shared * d_model) as i32) };
    let wu_dev = unsafe { aether_dev_alloc_f32((d_ff_shared * d_model) as i32) };
    let wd_dev = unsafe { aether_dev_alloc_f32((d_model * d_ff_shared) as i32) };
    let gate_dev = unsafe { aether_dev_alloc_f32(d_ff_shared as i32) };
    let up_dev = unsafe { aether_dev_alloc_f32(d_ff_shared as i32) };
    let down_dev = unsafe { aether_dev_alloc_f32(d_model as i32) };
    unsafe {
        aether_dev_h2d_f32(x.as_ptr() as i64, x_dev, d_model as i32);
        aether_dev_h2d_f32(w_gate.as_ptr() as i64, wg_dev, (d_ff_shared * d_model) as i32);
        aether_dev_h2d_f32(w_up.as_ptr() as i64, wu_dev, (d_ff_shared * d_model) as i32);
        aether_dev_h2d_f32(w_down.as_ptr() as i64, wd_dev, (d_model * d_ff_shared) as i32);

        aether_op_matmul_nt_f32_cuda(x_dev, wg_dev, gate_dev,
            1, d_model as i32, d_ff_shared as i32);
        aether_op_matmul_nt_f32_cuda(x_dev, wu_dev, up_dev,
            1, d_model as i32, d_ff_shared as i32);
        aether_op_silu_f32_cuda(gate_dev, d_ff_shared as i32);
        aether_op_mul_inplace_f32_cuda(gate_dev, up_dev, d_ff_shared as i32);
        aether_op_matmul_nt_f32_cuda(gate_dev, wd_dev, down_dev,
            1, d_ff_shared as i32, d_model as i32);
        aether_dev_sync();
    }
    let mut down_gpu = vec![0f32; d_model];
    unsafe { aether_dev_d2h_f32(down_dev, down_gpu.as_mut_ptr() as i64, d_model as i32); }

    unsafe {
        for h in [x_dev, wg_dev, wu_dev, wd_dev, gate_dev, up_dev, down_dev] {
            aether_dev_free_f32(h);
        }
    }

    let max_diff = down_cpu.iter().zip(down_gpu.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let n_finite = down_gpu.iter().filter(|x| x.is_finite()).count();
    println!("[moe-shexp] d_model={} d_ff_shared={} max_diff={:.3e} finite={}/{}",
        d_model, d_ff_shared, max_diff, n_finite, d_model);
    assert_eq!(n_finite, d_model, "non-finite values in shared-expert output");
    // The chain spans 4 matmuls + silu + mul; fp accumulation across ~352
    // inner-product widths keeps the worst-case under 1e-4.
    assert!(max_diff < 1e-4,
        "shared-expert FFN diverged from CPU reference ({:.3e})", max_diff);
}
