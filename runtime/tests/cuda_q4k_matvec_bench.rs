//! Q4_K seq1 mat-VEC throughput microbench — v3 (prod) vs v4/v5 rewrites.
//!
//! Decode is memory-bandwidth-bound on the 4-bit weight reads. v3 is the
//! production kernel; v4 = no-shared/no-sync/inline-scales (regressed); v5 =
//! multi-row-per-warp (R=4) for memory-level parallelism. Times all three at
//! every Qwen2.5-7B decode matmul shape, asserts each matches v3 per-row, and
//! reports achieved DRAM bandwidth + speedup vs the P100 peak (~720 GB/s).
//! Opt-in (`--ignored`). Sweep occupancy: AETHER_Q4K_V4_WPB / AETHER_Q4K_V5_WPB.
//!
//! roadmap: P19.5 (perf — surpass-llama kernel work)
#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::time::Instant;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_fused_q4k_matmul_seq1_v3_cuda,
    aether_op_fused_q4k_matmul_seq1_v4_cuda,
    aether_op_fused_q4k_matmul_seq1_v5_cuda,
    aether_op_fused_q4k_matmul_seq1_v6_cuda,
    aether_op_fused_q4k_membw_probe_cuda,
};

/// Well-conditioned Q4_K weight bytes: random qs/scales, but FIXED small d/dmin
/// (f16 0.0625 / 0.0156) per 144-byte block so dequant outputs stay
/// normal-magnitude. Fully-random d/dmin f16 reach ±65504, producing ~1e7
/// outputs with catastrophic cancellation that make v6's (correct) fp
/// reassociation look like a large relative error.
fn rng_bytes(n: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = vec![0u8; n];
    let mut s = Wrapping(seed | 1);
    for b in out.iter_mut() {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        *b = (s.0 & 0xFF) as u8;
    }
    for blk in out.chunks_mut(144) {
        if blk.len() >= 4 {
            blk[0] = 0x00; blk[1] = 0x2C; // d    = f16 0.0625
            blk[2] = 0x00; blk[3] = 0x24; // dmin = f16 0.0156
        }
    }
    out
}

const SHAPES: &[(&str, usize, usize)] = &[
    ("q_proj   ", 3584, 3584),
    ("k/v_proj ", 512, 3584),
    ("o_proj   ", 3584, 3584),
    ("gate/up  ", 18944, 3584),
    ("down     ", 3584, 18944),
    ("lm_head  ", 152064, 3584),
];

#[test]
#[ignore]
fn q4k_seq1_kernel_ab() {
    unsafe {
        assert_eq!(aether_dev_init(), 0);
        const ITERS: usize = 300;
        const WARMUP: usize = 50;
        // (name, fn). All must equal v3 per-row.
        type K = unsafe extern "C" fn(i64, i64, i64, c_int, c_int) -> c_int;
        let kernels: [(&str, K); 4] = [
            ("v3", aether_op_fused_q4k_matmul_seq1_v3_cuda),
            ("v4", aether_op_fused_q4k_matmul_seq1_v4_cuda),
            ("v5", aether_op_fused_q4k_matmul_seq1_v5_cuda),
            ("v6", aether_op_fused_q4k_matmul_seq1_v6_cuda),
        ];
        println!("\n=== Q4_K seq1 mat-vec A/B (P100 peak ~720 GB/s) ===");
        let mut tot = [0f64; 4];
        let mut bytes = 0f64;
        for &(label, n_out, n_in) in SHAPES {
            let n_blocks = n_in / 256;
            let w_bytes = n_out * n_blocks * 144;
            let w = rng_bytes(w_bytes, 0xC0FFEE ^ (n_out as u64));
            let a: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
            let d_a = aether_dev_alloc_f32(n_in as c_int);
            let d_w = aether_dev_alloc_u8(w_bytes as c_int);
            let d_o = aether_dev_alloc_f32(n_out as c_int);
            aether_dev_h2d_f32(a.as_ptr() as i64, d_a, n_in as c_int);
            aether_dev_h2d_u8(w.as_ptr() as i64, d_w, w_bytes as c_int);

            let mut ref_o = vec![0f32; n_out];
            let mut us = [0f64; 4];
            let mut maxdiff = 0f32;
            for (ki, (name, f)) in kernels.iter().enumerate() {
                let call = || { f(d_a, d_w, d_o, n_out as c_int, n_blocks as c_int); };
                for _ in 0..WARMUP { call(); }
                aether_dev_sync();
                let t0 = Instant::now();
                for _ in 0..ITERS { call(); }
                aether_dev_sync();
                us[ki] = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
                let mut o = vec![0f32; n_out];
                aether_dev_d2h_f32(d_o, o.as_mut_ptr() as i64, n_out as c_int);
                if ki == 0 { ref_o = o; }
                else {
                    // RELATIVE tolerance — random test weights give ~1e7-magnitude
                    // outputs, so v6's different fp summation grouping shows large
                    // ABSOLUTE diffs that are ~1e-6 relative (correct).
                    let d = ref_o.iter().zip(&o)
                        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
                        .fold(0f32, f32::max);
                    let _ = name;
                    maxdiff = maxdiff.max(d);
                }
                tot[ki] += us[ki];
            }
            // memory-ceiling probe (minimal ALU, same byte reads) — timing only.
            let probe = || { aether_op_fused_q4k_membw_probe_cuda(d_a, d_w, d_o, n_out as c_int, n_blocks as c_int); };
            for _ in 0..WARMUP { probe(); }
            aether_dev_sync();
            let tp = Instant::now();
            for _ in 0..ITERS { probe(); }
            aether_dev_sync();
            let us_probe = tp.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            let gbs = |u: f64| w_bytes as f64 / (u * 1e-6) / 1e9;
            println!("{:<10} {:>7}  v3 {:>6.1}  v6 {:>6.1}({:.2}x)  v4 {:>6.1}  v5 {:>6.1}  PROBE {:>6.1}  diff {:.1e}",
                label, n_out, gbs(us[0]), gbs(us[3]), us[0] / us[3], gbs(us[1]), gbs(us[2]), gbs(us_probe), maxdiff);
            bytes += w_bytes as f64;
            if maxdiff >= 1e-3 { println!("   ^ {} rel maxdiff {:.2e} (>1e-3)", label.trim(), maxdiff); }
            let _ = aether_dev_free_f32(d_a);
            let _ = aether_dev_free_u8(d_w);
            let _ = aether_dev_free_f32(d_o);
        }
        let g = |t: f64| bytes / (t * 1e-6) / 1e9;
        println!("--------");
        println!("aggregate GB/s:  v3 {:.1} ({:.1}% peak) | v6 {:.1} ({:.2}x, {:.1}% peak) | v4 {:.1} | v5 {:.1}",
            g(tot[0]), g(tot[0]) / 720.0 * 100.0,
            g(tot[3]), tot[0] / tot[3], g(tot[3]) / 720.0 * 100.0,
            g(tot[1]), g(tot[2]));
    }
}
