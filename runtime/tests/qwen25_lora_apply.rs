//! Apply a matt-voice-shape LoRA adapter to a real Qwen2.5-7B
//! weight matrix and verify the apply math.
//!
//! Production matt-voice trains rank-8 LoRA on (q_proj, v_proj) of
//! every block per the matt-voice candle config. We use the same
//! rank here; the A and B matrices are synthetic so the test runs
//! deterministically and can independently verify the math.
//!
//! Verifies:
//!   1. Zero LoRA leaves W unchanged (no-op identity).
//!   2. Non-zero LoRA updates W by exactly the expected delta
//!      (scale * A^T @ B^T per element).
//!   3. The apply is in place on a real Q4_K dequantised weight,
//!      i.e. the math composes with our existing GGUF + dequant chain.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_data_ptr,
    aether_gguf_get_tensor_n_elems,
    aether_dequant_q4_k_m,
    aether_op_apply_lora_f32,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

const D_MODEL: usize = 3584;

fn transpose_weight(gguf: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; d_in * d_out];
    for i_out in 0..d_out {
        for i_in in 0..d_in {
            out[i_in * d_out + i_out] = gguf[i_out * d_in + i_in];
        }
    }
    out
}

#[test]
fn lora_apply_on_real_qwen25_wq() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // Load blk.0.attn_q.weight (real Q4_K dequantised, then transposed).
        let needle = b"blk.0.attn_q.weight";
        let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
        assert!(idx >= 0);
        let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
        assert_eq!(n_elems, D_MODEL * D_MODEL);
        let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
        let mut w_gguf = vec![0.0f32; n_elems];
        aether_dequant_q4_k_m(dptr as *const c_void, w_gguf.as_mut_ptr() as *mut c_void,
            (n_elems / 256) as c_int);
        let mut w = transpose_weight(&w_gguf, D_MODEL, D_MODEL);
        drop(w_gguf);

        // Baseline checksum on the pristine W.
        let baseline_sum: f64 = w.iter().map(|&v| v as f64).sum();
        let baseline_sq: f64 = w.iter().map(|&v| (v as f64) * (v as f64)).sum();
        eprintln!("[baseline W_q] sum={:.6e}, ||W||_F^2={:.6e}", baseline_sum, baseline_sq);

        // === Step 1: zero LoRA -> W unchanged ===
        let rank = 8usize;
        let a_zero = vec![0.0f32; rank * D_MODEL];
        let b_zero = vec![0.0f32; D_MODEL * rank];
        aether_op_apply_lora_f32(
            w.as_mut_ptr() as *mut c_void,
            a_zero.as_ptr() as *const c_void,
            b_zero.as_ptr() as *const c_void,
            1.0, D_MODEL as c_int, D_MODEL as c_int, rank as c_int,
        );
        let after_zero_sum: f64 = w.iter().map(|&v| v as f64).sum();
        assert!((after_zero_sum - baseline_sum).abs() < 1e-3,
            "zero LoRA shifted sum: {} -> {}", baseline_sum, after_zero_sum);
        eprintln!("[zero LoRA] sum unchanged: {:.6e}", after_zero_sum);

        // === Step 2: synthetic rank-8 LoRA shaped like matt-voice's adapter ===
        // A: rank=8 rows of d_in=3584 cols. Initialise with a tiny but
        // distinct pattern so the apply has a measurable effect.
        let scale = 0.05_f32;  // typical alpha/rank in PEFT (alpha=4, rank=8 ish)
        let mut a = vec![0.0f32; rank * D_MODEL];
        for r in 0..rank {
            for i_in in 0..D_MODEL {
                a[r * D_MODEL + i_in] = ((r * 31 + i_in * 7) % 17) as f32 * 0.01;
            }
        }
        let mut b = vec![0.0f32; D_MODEL * rank];
        for i_out in 0..D_MODEL {
            for r in 0..rank {
                b[i_out * rank + r] = ((i_out * 13 + r * 5) % 11) as f32 * 0.01;
            }
        }

        // Compute the expected delta at a few sample (i_in, i_out) cells
        // by direct formula, BEFORE applying.
        let probe_cells = [(0, 0), (100, 200), (3583, 3583), (1234, 5678 % D_MODEL)];
        let expected_deltas: Vec<f32> = probe_cells.iter().map(|&(i_in, i_out)| {
            let mut delta = 0.0f32;
            for r in 0..rank {
                delta += a[r * D_MODEL + i_in] * b[i_out * rank + r];
            }
            scale * delta
        }).collect();

        let before_at_probes: Vec<f32> = probe_cells.iter().map(|&(i_in, i_out)| {
            w[i_in * D_MODEL + i_out]
        }).collect();

        let t = std::time::Instant::now();
        let rc = aether_op_apply_lora_f32(
            w.as_mut_ptr() as *mut c_void,
            a.as_ptr() as *const c_void,
            b.as_ptr() as *const c_void,
            scale, D_MODEL as c_int, D_MODEL as c_int, rank as c_int,
        );
        assert_eq!(rc, 0);
        eprintln!("[apply LoRA rank={}] {:.2}s -- d_in=d_out={}",
            rank, t.elapsed().as_secs_f32(), D_MODEL);

        // Verify each probe cell changed by exactly the expected delta.
        for (i, &(i_in, i_out)) in probe_cells.iter().enumerate() {
            let after = w[i_in * D_MODEL + i_out];
            let actual_delta = after - before_at_probes[i];
            let expected = expected_deltas[i];
            assert!(
                (actual_delta - expected).abs() < 1e-4,
                "probe ({}, {}): expected delta {:.6e}, actual {:.6e} (before {:.6e}, after {:.6e})",
                i_in, i_out, expected, actual_delta, before_at_probes[i], after,
            );
            eprintln!("  probe ({}, {}): before={:.6e} after={:.6e} delta={:.6e} ✓",
                i_in, i_out, before_at_probes[i], after, actual_delta);
        }

        // Verify the global Frobenius norm changed by a meaningful amount.
        let after_sq: f64 = w.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let ratio = after_sq / baseline_sq;
        eprintln!("[apply LoRA] ||W||_F^2: {:.6e} -> {:.6e} (ratio {:.4})",
            baseline_sq, after_sq, ratio);
        assert!(ratio != 1.0, "LoRA apply had no effect");

        aether_gguf_close(h);
    }
}
