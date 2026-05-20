//! Q6_K dequant on GPU matches the CPU `aether_dequant_q6_k`
//! reference byte-for-byte. Tested against real Qwen2.5
//! `blk.0.attn_v.weight` (Q6_K dtype, ~7 MB).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_dequant_q6_k, aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
    aether_gguf_get_tensor_data_ptr, aether_gguf_get_tensor_n_elems,
};
use aether_rt::cuda::{
    aether_dev_init,
    aether_dev_alloc_u8, aether_dev_free_u8, aether_dev_h2d_u8,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_d2h_f32,
    aether_dev_sync,
    aether_op_dequant_q6_k_f32_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

#[test]
fn q6_k_dequant_real_qwen25() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B not present");
        return;
    }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);
        let needle = b"blk.0.attn_v.weight";
        let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
        assert!(idx >= 0);
        assert_eq!(aether_gguf_get_tensor_dtype(h, idx), 14, "expected Q6_K");
        let n_elems = aether_gguf_get_tensor_n_elems(h, idx) as usize;
        let n_blocks = n_elems / 256;
        let n_bytes = n_blocks * 210;
        let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;
        let bytes: Vec<u8> = std::slice::from_raw_parts(dptr, n_bytes).to_vec();

        // CPU dequant
        let mut cpu = vec![0.0f32; n_elems];
        aether_dequant_q6_k(bytes.as_ptr() as *const c_void,
            cpu.as_mut_ptr() as *mut c_void, n_blocks as c_int);

        // GPU dequant
        let d_blocks = aether_dev_alloc_u8(n_bytes as c_int);
        let d_out    = aether_dev_alloc_f32(n_elems as c_int);
        aether_dev_h2d_u8(bytes.as_ptr() as i64, d_blocks, n_bytes as c_int);
        let rc = aether_op_dequant_q6_k_f32_cuda(d_blocks, d_out, n_blocks as c_int);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu = vec![0.0f32; n_elems];
        aether_dev_d2h_f32(d_out, gpu.as_mut_ptr() as i64, n_elems as c_int);
        aether_dev_free_u8(d_blocks);
        aether_dev_free_f32(d_out);

        let mut max_diff = 0.0f32;
        let mut worst_i = 0usize;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let d = (g - c).abs();
            if d > max_diff { max_diff = d; worst_i = i; }
        }
        eprintln!("[Q6_K real qwen2.5 W_v] {} blocks * 210B = {} MB Q6_K (vs {} MB f32, 4.0x less)",
            n_blocks, n_bytes / (1024*1024), n_elems * 4 / (1024*1024));
        eprintln!("[Q6_K real qwen2.5 W_v] max diff GPU vs CPU = {:.3e} at i={}", max_diff, worst_i);
        eprintln!("  cpu first 4: {:?}", &cpu[..4]);
        eprintln!("  gpu first 4: {:?}", &gpu[..4]);
        assert!(max_diff == 0.0, "Q6_K GPU/CPU mismatch: {}", max_diff);
        aether_gguf_close(h);
    }
}
