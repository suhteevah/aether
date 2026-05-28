//! FFN-section perf — faithful llama-MMVQ Q4_K+Q8_1 + SwiGLU port correctness.
//!
//! The MMVQ kernel quantizes the activation to Q8_1 (lossy by design), so it
//! can't be bit-identical to the float-activation base.  Tight gate that
//! catches int-math bugs (wrong nibble extraction, wrong bq8_offset, wrong
//! scale/min unpacking, wrong SwiGLU order) while keeping the only legit
//! difference small: well-conditioned weights + smooth activation → ~1-2% rel.
//! Bugs in the indexing typically blow this to >20%.
//! Real coherence is the model smoke (AETHER_FFN_LLAMA=1).
//!
//! roadmap: P10

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_op_quantize_q8_1_llama_cuda,
    aether_op_mmvq_q4k_q8_1_swiglu_cuda,
};

fn cond_q4k_bytes(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = vec![0u8; n_outputs * blocks_per_row * 144];
    let mut s = Wrapping(seed);
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 144;
        out[off]=0x47; out[off+1]=0x21; out[off+2]=0x47; out[off+3]=0x19;
        for b in 4..16 { out[off+b] = 0x22; }   // moderate fixed scales
        for b in 16..144 {
            s ^= s<<13; s ^= s>>7; s ^= s<<17; out[off+b] = (s.0&0xFF) as u8;
        }
    }
    out
}

#[test]
fn mmvq_llama_q4k_swiglu_matches_base() {
    unsafe {
        assert_eq!(0, aether_dev_init());
        const N_FF: usize = 4096;
        const N_BLOCKS: c_int = 14;       // d_model/256 = 3584/256
        const K: usize = (N_BLOCKS as usize) * 256;

        let a: Vec<f32> = (0..K).map(|i|
            (((i as f32)*0.011).sin() + 0.3*((i as f32)*0.003).cos()) * 0.5
        ).collect();
        let wg = cond_q4k_bytes(N_FF, N_BLOCKS as usize, 0xCAFE);
        let wu = cond_q4k_bytes(N_FF, N_BLOCKS as usize, 0xBEEF);

        let d_a = aether_dev_alloc_f32(K as c_int);
        let d_wg = aether_dev_alloc_u8(wg.len() as c_int);
        let d_wu = aether_dev_alloc_u8(wu.len() as c_int);
        let d_base = aether_dev_alloc_f32(N_FF as c_int);
        let d_mmvq = aether_dev_alloc_f32(N_FF as c_int);
        let d_aq = aether_dev_alloc_u8(K as c_int);
        let d_ad = aether_dev_alloc_f32((K/32) as c_int);
        let d_as = aether_dev_alloc_f32((K/32) as c_int);

        aether_dev_h2d_f32(a.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(wg.as_ptr() as i64, d_wg, wg.len() as c_int);
        aether_dev_h2d_u8(wu.as_ptr() as i64, d_wu, wu.len() as c_int);

        assert_eq!(0, aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
            d_a, d_wg, d_wu, d_base, N_FF as c_int, N_BLOCKS));
        assert_eq!(0, aether_op_quantize_q8_1_llama_cuda(d_a, d_aq, d_ad, d_as, K as c_int));
        assert_eq!(0, aether_op_mmvq_q4k_q8_1_swiglu_cuda(
            d_wg, d_wu, d_aq, d_ad, d_mmvq, N_FF as c_int, N_BLOCKS));
        aether_dev_sync();

        let mut b = vec![0.0f32; N_FF]; let mut m = vec![0.0f32; N_FF];
        aether_dev_d2h_f32(d_base, b.as_mut_ptr() as i64, N_FF as c_int);
        aether_dev_d2h_f32(d_mmvq, m.as_mut_ptr() as i64, N_FF as c_int);

        let mut max_rel = 0.0f32;
        let mut sum_sq_err = 0.0f64;
        let mut sum_sq = 0.0f64;
        for i in 0..N_FF {
            let d = (b[i] - m[i]).abs();
            let r = d / b[i].abs().max(m[i].abs()).max(1e-6);
            if r > max_rel { max_rel = r; }
            sum_sq_err += (d as f64).powi(2);
            sum_sq += (b[i] as f64).powi(2);
        }
        let rms_rel = (sum_sq_err / sum_sq.max(1e-12)).sqrt();
        eprintln!("[mmvq-llama-parity] N_FF={} max_rel={:.3e} rms_rel={:.3e}  base[0..3]={:?} mmvq[0..3]={:?}",
            N_FF, max_rel, rms_rel, &b[..3], &m[..3]);
        assert!(rms_rel < 3e-2, "rms_rel {:.3e} exceeds 3% — likely an int-math bug", rms_rel);

        aether_dev_free_f32(d_a);
        aether_dev_free_u8(d_wg); aether_dev_free_u8(d_wu);
        aether_dev_free_f32(d_base); aether_dev_free_f32(d_mmvq);
        aether_dev_free_u8(d_aq); aether_dev_free_f32(d_ad); aether_dev_free_f32(d_as);
    }
}
