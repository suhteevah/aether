//! Q4_K_M dequant on GPU matches the CPU `aether_dequant_q4_k_m`
//! reference byte-for-byte (to f32 tolerance).
//!
//! Tests:
//!  1. Synthetic super-block with known scales / nibbles -> known output
//!  2. Real Qwen2.5 blk.0.attn_q.weight first super-block -> compare
//!     GPU vs CPU element-wise

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_dequant_q4_k_m, aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_data_ptr,
};
use aether_rt::cuda::{
    aether_dev_init,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_d2h_f32,
    aether_dev_sync,
    aether_op_dequant_q4_k_m_f32_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

/// Build a synthetic Q4_K_M super-block (144 bytes) with known values.
fn make_synth_block() -> Vec<u8> {
    let mut b = vec![0u8; 144];
    // d = 1.0 f16 = 0x3C00, dmin = 0.5 f16 = 0x3800.
    b[0] = 0x00; b[1] = 0x3C;
    b[2] = 0x00; b[3] = 0x38;
    // scales: 12 bytes. For sub < 4: sc = scales[sub] & 63, mn = scales[sub+4] & 63
    // For sub >= 4: scales[sub-4][6:7] -> sc high bits, scales[sub][6:7] -> mn high bits
    // Set scales[0..4] = sc for sub 0..3 = [3, 5, 7, 11]
    // Set scales[4..8] = mn for sub 0..3 = [2, 4, 6, 10]
    // Set scales[8..12] = packed sc/mn for sub 4..7 (we use 0 for the high bits => sc/mn 4..7 read from scales[8..12])
    b[4] = 3;  b[5] = 5;  b[6] = 7;  b[7] = 11;
    b[8] = 2;  b[9] = 4;  b[10] = 6; b[11] = 10;
    // Quants: 128 bytes, 2 quants per byte. We use byte 0x12 throughout:
    // low nibble = 2, high nibble = 1.
    for i in 16..144 { b[i] = 0x12; }
    b
}

fn cpu_dequant(blocks: &[u8], n_blocks: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_blocks * 256];
    unsafe {
        aether_dequant_q4_k_m(blocks.as_ptr() as *const c_void, out.as_mut_ptr() as *mut c_void, n_blocks as c_int);
    }
    out
}

fn gpu_dequant(blocks: &[u8], n_blocks: usize) -> Vec<f32> {
    unsafe {
        aether_dev_init();
        let d_blocks = aether_dev_alloc_u8(blocks.len() as c_int);
        let d_out    = aether_dev_alloc_f32((n_blocks * 256) as c_int);
        aether_dev_h2d_u8(blocks.as_ptr() as i64, d_blocks, blocks.len() as c_int);
        let rc = aether_op_dequant_q4_k_m_f32_cuda(d_blocks, d_out, n_blocks as c_int);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut host = vec![0.0f32; n_blocks * 256];
        aether_dev_d2h_f32(d_out, host.as_mut_ptr() as i64, (n_blocks * 256) as c_int);
        aether_dev_free_u8(d_blocks);
        aether_dev_free_f32(d_out);
        host
    }
}

#[test]
fn q4_k_dequant_synth_block() {
    let block = make_synth_block();
    let cpu = cpu_dequant(&block, 1);
    let gpu = gpu_dequant(&block, 1);
    let mut max_diff = 0.0f32;
    for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_diff { max_diff = d; }
        if d > 1e-4 {
            eprintln!("[mismatch] {}: cpu={} gpu={} diff={}", i, c, g, d);
        }
    }
    eprintln!("[synth] max_diff CPU vs GPU = {:.3e} (first 4: cpu={:?} gpu={:?})",
        max_diff, &cpu[..4], &gpu[..4]);
    assert!(max_diff < 1e-4, "max diff {} too large", max_diff);
}

#[test]
fn q4_k_dequant_real_qwen25() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);
        let needle = b"blk.0.attn_q.weight";
        let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
        assert!(idx >= 0);
        let dptr = aether_gguf_get_tensor_data_ptr(h, idx);
        // Take the first 4 super-blocks worth (576 bytes -> 1024 f32 elements).
        let n_blocks = 4usize;
        let blocks_slice = std::slice::from_raw_parts(dptr as *const u8, n_blocks * 144);
        let blocks_vec = blocks_slice.to_vec();
        let cpu = cpu_dequant(&blocks_vec, n_blocks);
        let gpu = gpu_dequant(&blocks_vec, n_blocks);
        let mut max_diff = 0.0f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let d = (g - c).abs();
            if d > max_diff { max_diff = d; }
        }
        eprintln!("[real qwen2.5 W_q first {} super-blocks] max diff CPU/GPU = {:.3e}",
            n_blocks, max_diff);
        eprintln!("  cpu first 4: {:?}", &cpu[..4]);
        eprintln!("  gpu first 4: {:?}", &gpu[..4]);
        assert!(max_diff < 1e-4, "GPU/CPU mismatch beyond tolerance");
        aether_gguf_close(h);
    }
}
