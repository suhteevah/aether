//! Verify fused FFN kernel (gate+up+silu+mul) produces the same
//! output as running the 4 separate kernels.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{aether_dequant_q4_k_m};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_op_silu_f32_cuda, aether_op_mul_inplace_f32_cuda,
};

// Build a small synthetic Q4_K weight by quantising a known dequant.
// For parity testing we don't actually need real Q4_K bytes -- we just
// need the SAME weight bytes fed to both paths.
//
// We'll use the existing real Qwen2.5 model if present (a real test);
// otherwise synthetic random Q4_K bytes (the kernel doesn't care what
// the bytes mean as long as they're a consistent 144-byte super-block).
fn random_q4k_bytes(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let n_bytes = n_outputs * blocks_per_row * 144;
    let mut out = vec![0u8; n_bytes];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        // xorshift-ish
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    // Fix d/dmin (first 4 bytes of each block) to sane f16 values so we
    // don't accidentally get NaN/inf during dequant -> matmul.
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 144;
        // d = 0.01 in f16 -> 0x2147 (approx 0.00999...)
        out[off + 0] = 0x47; out[off + 1] = 0x21;
        // dmin = 0.005 in f16 -> 0x1947
        out[off + 2] = 0x47; out[off + 3] = 0x19;
    }
    out
}

#[test]
#[ignore]
fn fused_ffn_matches_separate_kernels() {
    unsafe {
        aether_dev_init();
        const N_FF: usize = 1024;
        const N_BLOCKS: c_int = 14;  // D_MODEL/256 = 3584/256
        const K: usize = (N_BLOCKS as usize) * 256;

        // Random input.
        let a_host: Vec<f32> = (0..K).map(|i| ((i * 1103515245).wrapping_add(12345)) as f32 * 1e-9).collect();
        let w_gate_host = random_q4k_bytes(N_FF, N_BLOCKS as usize, 0xCAFEu64);
        let w_up_host   = random_q4k_bytes(N_FF, N_BLOCKS as usize, 0xBEEFu64);

        // Device buffers.
        let d_a    = aether_dev_alloc_f32(K as c_int);
        let d_wg   = aether_dev_alloc_u8(w_gate_host.len() as c_int);
        let d_wu   = aether_dev_alloc_u8(w_up_host.len() as c_int);
        let d_gate = aether_dev_alloc_f32(N_FF as c_int);
        let d_up   = aether_dev_alloc_f32(N_FF as c_int);
        let d_fused= aether_dev_alloc_f32(N_FF as c_int);

        aether_dev_h2d_f32(a_host.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(w_gate_host.as_ptr() as i64, d_wg, w_gate_host.len() as c_int);
        aether_dev_h2d_u8(w_up_host.as_ptr()   as i64, d_wu, w_up_host.len()   as c_int);

        // --- Path 1: separate kernels ---
        assert_eq!(0, aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_wg, d_gate, N_FF as c_int, N_BLOCKS));
        assert_eq!(0, aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_wu, d_up,   N_FF as c_int, N_BLOCKS));
        aether_op_silu_f32_cuda(d_gate, N_FF as c_int);
        aether_op_mul_inplace_f32_cuda(d_gate, d_up, N_FF as c_int);
        aether_dev_sync();
        let mut ref_out = vec![0.0f32; N_FF];
        aether_dev_d2h_f32(d_gate, ref_out.as_mut_ptr() as i64, N_FF as c_int);

        // --- Path 2: fused kernel ---
        assert_eq!(0, aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(
            d_a, d_wg, d_wu, d_fused, N_FF as c_int, N_BLOCKS));
        aether_dev_sync();
        let mut fused_out = vec![0.0f32; N_FF];
        aether_dev_d2h_f32(d_fused, fused_out.as_mut_ptr() as i64, N_FF as c_int);

        // Compare.
        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut bad = 0usize;
        for i in 0..N_FF {
            let r = ref_out[i];
            let f = fused_out[i];
            let d = (r - f).abs();
            if d > max_diff { max_diff = d; }
            let rel = if r.abs() > 1e-6 { d / r.abs() } else { 0.0 };
            if rel > max_rel { max_rel = rel; }
            if d > 1e-3 { bad += 1; }
        }
        eprintln!("[fused FFN parity] max_diff={:.3e} max_rel={:.3e} bad={}/{}",
            max_diff, max_rel, bad, N_FF);
        eprintln!("  ref[0..4]   = {:?}", &ref_out[..4]);
        eprintln!("  fused[0..4] = {:?}", &fused_out[..4]);

        // Tolerance: both paths read the same Q4_K bytes the same way,
        // do the same FMAs, just at different launches -- expected
        // identical up to last-bit FP fluctuation in the silu math.
        assert!(max_diff < 1e-3, "fused FFN diverges from separate kernels");

        aether_dev_free_f32(d_a);
        aether_dev_free_u8(d_wg); aether_dev_free_u8(d_wu);
        aether_dev_free_f32(d_gate); aether_dev_free_f32(d_up);
        aether_dev_free_f32(d_fused);
    }
}
