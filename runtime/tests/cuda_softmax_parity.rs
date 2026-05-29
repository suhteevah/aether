//! Parity test for the parallel `softmax_f32` CUDA kernel.
//!
//! After parallelizing `softmax_f32` (grid=rows, block=256 cooperative reduce —
//! same fix as `rms_norm_fwd`/`layer_norm_fwd`), this guards that the cooperative
//! max-reduce + sum-reduce still produce row-wise softmax bit-faithful (within
//! fp32 noise) to a sequential CPU reference. Covers the decode trap case
//! (B=1, wide D), non-power-of-2 D, D < blockDim, and multi-row batches.
//!
//! roadmap: P7
#![cfg(feature = "cuda")]

use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_sync,
    aether_op_softmax_f32_cuda,
};

fn cpu_softmax(x: &[f32], b: usize, d: usize) -> Vec<f32> {
    let mut y = vec![0f32; b * d];
    for r in 0..b {
        let xr = &x[r * d..(r + 1) * d];
        let mx = xr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..d { let e = (xr[j] - mx).exp(); y[r * d + j] = e; sum += e; }
        let inv = 1.0 / sum;
        for j in 0..d { y[r * d + j] *= inv; }
    }
    y
}

// Deterministic pseudo-random fill (no external rng dep).
struct Gen { s: u64 }
impl Gen {
    fn next(&mut self) -> f32 {
        self.s ^= self.s << 13; self.s ^= self.s >> 7; self.s ^= self.s << 17;
        // map to [-4, 4) so exp() spans a meaningful dynamic range
        ((self.s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    }
    fn fill(&mut self, n: usize) -> Vec<f32> { (0..n).map(|_| self.next()).collect() }
}

fn run_case(b: usize, d: usize, seed: u64) {
    let mut g = Gen { s: seed };
    let x = g.fill(b * d);
    let cpu = cpu_softmax(&x, b, d);

    unsafe {
        assert_eq!(aether_dev_init(), 0);
        let xd = aether_dev_alloc_f32((b * d) as i32);
        let yd = aether_dev_alloc_f32((b * d) as i32);
        aether_dev_h2d_f32(x.as_ptr() as i64, xd, (b * d) as i32);
        let rc = aether_op_softmax_f32_cuda(xd, yd, b as i32, d as i32);
        assert_eq!(rc, 0, "softmax launch rc");
        aether_dev_sync();
        let mut gpu = vec![0f32; b * d];
        aether_dev_d2h_f32(yd, gpu.as_mut_ptr() as i64, (b * d) as i32);
        aether_dev_free_f32(xd);
        aether_dev_free_f32(yd);

        let mut max_diff = 0.0f32;
        let mut all_finite = true;
        for i in 0..b * d {
            let dd = (gpu[i] - cpu[i]).abs();
            if dd > max_diff { max_diff = dd; }
            if !gpu[i].is_finite() { all_finite = false; }
        }
        // each row must sum to 1
        for r in 0..b {
            let s: f32 = gpu[r * d..(r + 1) * d].iter().sum();
            assert!((s - 1.0).abs() < 1e-4, "row {} sum={} (B={} D={})", r, s, b, d);
        }
        println!("[softmax] B={} D={} max_diff={:.3e} finite={}", b, d, max_diff, all_finite);
        assert!(all_finite, "non-finite output (B={} D={})", b, d);
        assert!(max_diff < 1e-5, "max_diff {:.3e} too large (B={} D={})", max_diff, b, d);
    }
}

#[test]
fn softmax_decode_trap_b1_wide() {
    // The exact trap: one row, vocab-wide. Old kernel = one thread, O(D) serial.
    run_case(1, 152064, 0x1234_5678);
}

#[test]
fn softmax_small_and_odd_shapes() {
    run_case(1, 7, 0xABCD);          // D < blockDim, non-power-of-2
    run_case(1, 256, 0xBEEF);        // D == blockDim
    run_case(1, 257, 0xC0DE);        // D just over blockDim
    run_case(4, 1000, 0xFACE);       // multi-row, non-power-of-2
    run_case(32, 4096, 0xD00D);      // batched, attention-scores-shaped
}
