//! Verify v3 (byte-once) Q4_K matmul + fused FFN produce the same
//! outputs as v2.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_fused_q4k_matmul_seq1_v3_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_v2_cuda,
};

fn random_q4k_bytes(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let n_bytes = n_outputs * blocks_per_row * 144;
    let mut out = vec![0u8; n_bytes];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    // Fix d/dmin per block so we don't get NaN/inf.
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 144;
        out[off + 0] = 0x47; out[off + 1] = 0x21;  // d = 0.01
        out[off + 2] = 0x47; out[off + 3] = 0x19;  // dmin = 0.005
    }
    out
}

#[test]
#[ignore]
fn q4k_matmul_v3_matches_v2() {
    unsafe {
        aether_dev_init();
        const N: usize = 1024;
        const N_BLOCKS: c_int = 14;
        const K: usize = (N_BLOCKS as usize) * 256;

        let a_host: Vec<f32> = (0..K).map(|i| ((i * 1103515245).wrapping_add(12345)) as f32 * 1e-9).collect();
        let w_host = random_q4k_bytes(N, N_BLOCKS as usize, 0xDEAD_BEEFu64);

        let d_a   = aether_dev_alloc_f32(K as c_int);
        let d_w   = aether_dev_alloc_u8(w_host.len() as c_int);
        let d_out_v2 = aether_dev_alloc_f32(N as c_int);
        let d_out_v3 = aether_dev_alloc_f32(N as c_int);

        aether_dev_h2d_f32(a_host.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(w_host.as_ptr() as i64, d_w, w_host.len() as c_int);

        assert_eq!(0, aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_w, d_out_v2, N as c_int, N_BLOCKS));
        assert_eq!(0, aether_op_fused_q4k_matmul_seq1_v3_cuda(d_a, d_w, d_out_v3, N as c_int, N_BLOCKS));
        aether_dev_sync();

        let mut o_v2 = vec![0.0f32; N];
        let mut o_v3 = vec![0.0f32; N];
        aether_dev_d2h_f32(d_out_v2, o_v2.as_mut_ptr() as i64, N as c_int);
        aether_dev_d2h_f32(d_out_v3, o_v3.as_mut_ptr() as i64, N as c_int);

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..N {
            let d = (o_v2[i] - o_v3[i]).abs();
            if d > max_diff { max_diff = d; }
            let rel = if o_v2[i].abs() > 1e-6 { d / o_v2[i].abs() } else { 0.0 };
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("[q4k matmul v2 vs v3] max_diff={:.3e} max_rel={:.3e}", max_diff, max_rel);
        eprintln!("  v2[0..4] = {:?}", &o_v2[..4]);
        eprintln!("  v3[0..4] = {:?}", &o_v3[..4]);
        // The two kernels do the FMAs in a different order so trace-level
        // floating-point differences are expected; bound by ~1e-3 relative.
        assert!(max_rel < 1e-3 || max_diff < 1e-3, "v3 diverges from v2");

        aether_dev_free_f32(d_a); aether_dev_free_u8(d_w);
        aether_dev_free_f32(d_out_v2); aether_dev_free_f32(d_out_v3);
    }
}

#[test]
#[ignore]
fn ffn_v2_matches_v1() {
    unsafe {
        aether_dev_init();
        const N: usize = 1024;
        const N_BLOCKS: c_int = 14;
        const K: usize = (N_BLOCKS as usize) * 256;

        let a_host: Vec<f32> = (0..K).map(|i| ((i * 1103515245).wrapping_add(12345)) as f32 * 1e-9).collect();
        let w_gate = random_q4k_bytes(N, N_BLOCKS as usize, 0xCAFEu64);
        let w_up   = random_q4k_bytes(N, N_BLOCKS as usize, 0xBEEFu64);

        let d_a    = aether_dev_alloc_f32(K as c_int);
        let d_wg   = aether_dev_alloc_u8(w_gate.len() as c_int);
        let d_wu   = aether_dev_alloc_u8(w_up.len() as c_int);
        let d_out_v1 = aether_dev_alloc_f32(N as c_int);
        let d_out_v2 = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_f32(a_host.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(w_gate.as_ptr() as i64, d_wg, w_gate.len() as c_int);
        aether_dev_h2d_u8(w_up.as_ptr()   as i64, d_wu, w_up.len()   as c_int);

        assert_eq!(0, aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(d_a, d_wg, d_wu, d_out_v1, N as c_int, N_BLOCKS));
        assert_eq!(0, aether_op_fused_q4k_ffn_gate_up_silu_mul_v2_cuda(d_a, d_wg, d_wu, d_out_v2, N as c_int, N_BLOCKS));
        aether_dev_sync();

        let mut o1 = vec![0.0f32; N];
        let mut o2 = vec![0.0f32; N];
        aether_dev_d2h_f32(d_out_v1, o1.as_mut_ptr() as i64, N as c_int);
        aether_dev_d2h_f32(d_out_v2, o2.as_mut_ptr() as i64, N as c_int);

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for i in 0..N {
            let d = (o1[i] - o2[i]).abs();
            if d > max_diff { max_diff = d; }
            let rel = if o1[i].abs() > 1e-6 { d / o1[i].abs() } else { 0.0 };
            if rel > max_rel { max_rel = rel; }
        }
        eprintln!("[ffn v1 vs v2] max_diff={:.3e} max_rel={:.3e}", max_diff, max_rel);
        eprintln!("  v1[0..4] = {:?}", &o1[..4]);
        eprintln!("  v2[0..4] = {:?}", &o2[..4]);
        assert!(max_rel < 1e-3 || max_diff < 1e-3, "FFN v2 diverges from v1");
    }
}
