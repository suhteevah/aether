//! Attention-vs-FFN mat-vec throughput microbench — the NEW llama-MMVQ singles.
//!
//! The FFN-fusion sprint profile showed FFN is bandwidth-saturated (gate/up
//! ~205 GB/s, down ~172 GB/s, both >= llama) while the attention section runs
//! at only ~37 GB/s aggregate.  This bench isolates each decode matmul shape on
//! the SHIPPED kernels (`aether_op_mmvq_q4k_q8_1_single_cuda`,
//! `..._q6k_...`, `..._swiglu_...`) to localize the attention deficit: are the
//! q/k/v/o mat-vecs occupancy-bound (k/v = 512 rows = ~57% of a 56-SM P100), or
//! already at FFN bandwidth (→ deficit is the paged-attn kernel, not the matmul)?
//!
//! Opt-in (`--ignored`).  P100 peak ~720 GB/s (16GB) / ~549 (12GB).
//!
//! roadmap: P19.5 (perf — surpass-llama kernel work)
#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::time::Instant;

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_sync,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_op_quantize_q8_1_llama_cuda,
    aether_op_mmvq_q4k_q8_1_single_cuda,
    aether_op_mmvq_q6k_q8_1_single_cuda,
    aether_op_mmvq_q4k_q8_1_swiglu_cuda,
};

// Random weight bytes with FIXED small f16 d/dmin (or d for Q6_K) per block so
// dequant stays normal-magnitude.  Q4_K block = 144 B (d@0, dmin@2); Q6_K block
// = 210 B (d (half) @ 208).
fn rng_q4k(n_blocks: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = vec![0u8; n_blocks * 144];
    let mut s = Wrapping(seed | 1);
    for b in out.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; *b = (s.0 & 0xFF) as u8; }
    for blk in out.chunks_mut(144) {
        blk[0] = 0x00; blk[1] = 0x2C; // d    = f16 0.0625
        blk[2] = 0x00; blk[3] = 0x24; // dmin = f16 0.0156
    }
    out
}
fn rng_q6k(n_blocks: usize, seed: u64) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = vec![0u8; n_blocks * 210];
    let mut s = Wrapping(seed | 1);
    for b in out.iter_mut() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; *b = (s.0 & 0xFF) as u8; }
    for blk in out.chunks_mut(210) {
        blk[208] = 0x00; blk[209] = 0x2C; // d = f16 0.0625
    }
    out
}

const ITERS: usize = 400;
const WARMUP: usize = 80;

// (label, n_out, n_in) — every Qwen2.5-7B decode matmul shape.
const SHAPES: &[(&str, usize, usize)] = &[
    ("q_proj  ", 3584, 3584),
    ("k_proj  ", 512, 3584),
    ("v_proj  ", 512, 3584),
    ("o_proj  ", 3584, 3584),
    ("down    ", 3584, 18944),
    ("lm_head ", 152064, 3584),
];

unsafe fn quantize(n_in: usize) -> (i64, i64) {
    let a: Vec<f32> = (0..n_in).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let d_a = aether_dev_alloc_f32(n_in as c_int);
    aether_dev_h2d_f32(a.as_ptr() as i64, d_a, n_in as c_int);
    let aq = aether_dev_alloc_u8(n_in as c_int);
    let ad = aether_dev_alloc_f32((n_in / 32) as c_int);
    let asm = aether_dev_alloc_f32((n_in / 32) as c_int);
    assert_eq!(aether_op_quantize_q8_1_llama_cuda(d_a, aq, ad, asm, n_in as c_int), 0);
    aether_dev_sync();
    let _ = aether_dev_free_f32(d_a);
    let _ = aether_dev_free_f32(asm);
    (aq, ad)
}

#[test]
#[ignore]
fn mmvq_attn_single_bandwidth() {
    unsafe {
        assert_eq!(aether_dev_init(), 0);
        println!("\n=== llama-MMVQ single mat-vec bandwidth (P100 peak ~720 GB/s) ===");
        println!("{:<9} {:>6} {:>5} {:>7}  {:>10}  {:>10}", "shape", "n_out", "bpr", "grid", "Q4_K GB/s", "Q6_K GB/s");
        for &(label, n_out, n_in) in SHAPES {
            let bpr = n_in / 256;
            let (aq, ad) = quantize(n_in);
            let d_o = aether_dev_alloc_f32(n_out as c_int);

            // Q4_K
            let w4 = rng_q4k(n_out * bpr, 0xC0FFEE ^ n_out as u64);
            let dw4 = aether_dev_alloc_u8((n_out * bpr * 144) as c_int);
            aether_dev_h2d_u8(w4.as_ptr() as i64, dw4, (n_out * bpr * 144) as c_int);
            let call4 = || { aether_op_mmvq_q4k_q8_1_single_cuda(dw4, aq, ad, d_o, n_out as c_int, bpr as c_int); };
            for _ in 0..WARMUP { call4(); }
            aether_dev_sync();
            let t = Instant::now();
            for _ in 0..ITERS { call4(); }
            aether_dev_sync();
            let us4 = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            let gbs4 = (n_out * bpr * 144) as f64 / (us4 * 1e-6) / 1e9;
            let _ = aether_dev_free_u8(dw4);

            // Q6_K
            let w6 = rng_q6k(n_out * bpr, 0xBEEF ^ n_out as u64);
            let dw6 = aether_dev_alloc_u8((n_out * bpr * 210) as c_int);
            aether_dev_h2d_u8(w6.as_ptr() as i64, dw6, (n_out * bpr * 210) as c_int);
            let call6 = || { aether_op_mmvq_q6k_q8_1_single_cuda(dw6, aq, ad, d_o, n_out as c_int, bpr as c_int); };
            for _ in 0..WARMUP { call6(); }
            aether_dev_sync();
            let t = Instant::now();
            for _ in 0..ITERS { call6(); }
            aether_dev_sync();
            let us6 = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            let gbs6 = (n_out * bpr * 210) as f64 / (us6 * 1e-6) / 1e9;
            let _ = aether_dev_free_u8(dw6);

            println!("{:<9} {:>6} {:>5} {:>7}  {:>6.1} ({:>5.1}us)  {:>6.1} ({:>5.1}us)",
                label, n_out, bpr, n_out, gbs4, us4, gbs6, us6);
            let _ = aether_dev_free_u8(aq);
            let _ = aether_dev_free_f32(ad);
            let _ = aether_dev_free_f32(d_o);
        }

        // gate/up SwiGLU (2 tensors, n_out=18944, bpr=14)
        let (n_out, n_in) = (18944usize, 3584usize);
        let bpr = n_in / 256;
        let (aq, ad) = quantize(n_in);
        let d_o = aether_dev_alloc_f32(n_out as c_int);
        let wg = rng_q4k(n_out * bpr, 0x1111);
        let wu = rng_q4k(n_out * bpr, 0x2222);
        let dwg = aether_dev_alloc_u8((n_out * bpr * 144) as c_int);
        let dwu = aether_dev_alloc_u8((n_out * bpr * 144) as c_int);
        aether_dev_h2d_u8(wg.as_ptr() as i64, dwg, (n_out * bpr * 144) as c_int);
        aether_dev_h2d_u8(wu.as_ptr() as i64, dwu, (n_out * bpr * 144) as c_int);
        let callgu = || { aether_op_mmvq_q4k_q8_1_swiglu_cuda(dwg, dwu, aq, ad, d_o, n_out as c_int, bpr as c_int); };
        for _ in 0..WARMUP { callgu(); }
        aether_dev_sync();
        let t = Instant::now();
        for _ in 0..ITERS { callgu(); }
        aether_dev_sync();
        let usgu = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        let gbsgu = (2 * n_out * bpr * 144) as f64 / (usgu * 1e-6) / 1e9;
        println!("gate/up   {:>6} {:>5} {:>7}  {:>6.1} GB/s ({:.1}us, 2 tensors)", n_out, bpr, n_out, gbsgu, usgu);
        let _ = aether_dev_free_u8(dwg); let _ = aether_dev_free_u8(dwu);
        let _ = aether_dev_free_u8(aq); let _ = aether_dev_free_f32(ad); let _ = aether_dev_free_f32(d_o);
    }
}
