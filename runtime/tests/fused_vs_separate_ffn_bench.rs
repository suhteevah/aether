//! After CUDA graphs land, kernel launch overhead is gone. Re-bench
//! fused FFN vs. 4 separate kernels (gate matmul, up matmul, silu,
//! mul_inplace) -- if separate is faster, the fused kernel's higher
//! register pressure was a worse tradeoff than the saved launches.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::time::Instant;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_alloc_u8,
    aether_dev_h2d_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q4k_matmul_seq1_v2_cuda,
    aether_op_silu_f32_cuda, aether_op_mul_inplace_f32_cuda,
    aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda,
};

fn random_q4k(n_outputs: usize, blocks_per_row: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let n_bytes = n_outputs * blocks_per_row * 144;
    let mut out = vec![0u8; n_bytes];
    let mut s = Wrapping(seed);
    for byte in out.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *byte = (s.0 & 0xFF) as u8;
    }
    for i in 0..n_outputs * blocks_per_row {
        let off = i * 144;
        out[off + 0] = 0x47; out[off + 1] = 0x21;
        out[off + 2] = 0x47; out[off + 3] = 0x19;
    }
    out
}

#[test]
#[ignore]
fn fused_vs_separate_ffn() {
    unsafe {
        aether_dev_init();
        const N: usize = 18944;
        const BLOCKS: c_int = 14;
        const K: usize = (BLOCKS as usize) * 256;
        const ITERS: usize = 100;

        let a: Vec<f32> = (0..K).map(|i| (i as f32) * 1e-3 - 1.0).collect();
        let wg = random_q4k(N, BLOCKS as usize, 0xAAAA);
        let wu = random_q4k(N, BLOCKS as usize, 0xBBBB);
        let d_a = aether_dev_alloc_f32(K as c_int);
        let d_wg = aether_dev_alloc_u8(wg.len() as c_int);
        let d_wu = aether_dev_alloc_u8(wu.len() as c_int);
        let d_gate = aether_dev_alloc_f32(N as c_int);
        let d_up = aether_dev_alloc_f32(N as c_int);
        let d_out_fused = aether_dev_alloc_f32(N as c_int);
        aether_dev_h2d_f32(a.as_ptr() as i64, d_a, K as c_int);
        aether_dev_h2d_u8(wg.as_ptr() as i64, d_wg, wg.len() as c_int);
        aether_dev_h2d_u8(wu.as_ptr() as i64, d_wu, wu.len() as c_int);

        // SEPARATE: 4 kernels
        for _ in 0..3 {
            aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_wg, d_gate, N as c_int, BLOCKS);
            aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_wu, d_up, N as c_int, BLOCKS);
            aether_op_silu_f32_cuda(d_gate, N as c_int);
            aether_op_mul_inplace_f32_cuda(d_gate, d_up, N as c_int);
        }
        aether_dev_sync();
        let t = Instant::now();
        for _ in 0..ITERS {
            aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_wg, d_gate, N as c_int, BLOCKS);
            aether_op_fused_q4k_matmul_seq1_v2_cuda(d_a, d_wu, d_up, N as c_int, BLOCKS);
            aether_op_silu_f32_cuda(d_gate, N as c_int);
            aether_op_mul_inplace_f32_cuda(d_gate, d_up, N as c_int);
        }
        aether_dev_sync();
        let sep_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        // FUSED: 1 kernel
        for _ in 0..3 {
            aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(d_a, d_wg, d_wu, d_out_fused, N as c_int, BLOCKS);
        }
        aether_dev_sync();
        let t = Instant::now();
        for _ in 0..ITERS {
            aether_op_fused_q4k_ffn_gate_up_silu_mul_cuda(d_a, d_wg, d_wu, d_out_fused, N as c_int, BLOCKS);
        }
        aether_dev_sync();
        let fus_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

        let bytes = (N * BLOCKS as usize * 144 * 2) as f64;
        eprintln!("[FFN 18944x3584 Q4_K (gate+up+silu+mul)]");
        eprintln!("  Separate (4 launches): {:6.2} us = {:5.1} GB/s",
            sep_us, bytes / 1e9 / (sep_us / 1e6));
        eprintln!("  Fused    (1 launch)  : {:6.2} us = {:5.1} GB/s  ({:.2}x vs separate)",
            fus_us, bytes / 1e9 / (fus_us / 1e6), sep_us / fus_us);
    }
}
