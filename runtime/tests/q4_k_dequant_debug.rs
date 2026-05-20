//! Debug Q4_K dequant: dump the raw f16 values + scales + first
//! quants of a real Qwen super-block, computed both CPU and GPU side.

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_data_ptr,
    aether_dequant_q4_k_m,
};
use aether_rt::cuda::{
    aether_dev_init,
    aether_dev_alloc_u8, aether_dev_h2d_u8, aether_dev_free_u8,
    aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_d2h_f32,
    aether_dev_sync,
    aether_op_dequant_q4_k_m_f32_cuda,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

#[test]
fn q4_k_debug_real_qwen25() {
    if !std::path::Path::new(QWEN_BLOB).exists() { return; }
    unsafe {
        aether_dev_init();
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        let needle = b"blk.0.attn_q.weight";
        let idx = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
        let dptr = aether_gguf_get_tensor_data_ptr(h, idx) as *const u8;

        // First super-block: dump raw bytes
        let block_bytes: Vec<u8> = std::slice::from_raw_parts(dptr, 144).to_vec();
        let d_bits    = u16::from_le_bytes([block_bytes[0], block_bytes[1]]);
        let dmin_bits = u16::from_le_bytes([block_bytes[2], block_bytes[3]]);
        eprintln!("raw d_bits={:#x} dmin_bits={:#x}", d_bits, dmin_bits);
        eprintln!("raw scales[0..12]={:?}", &block_bytes[4..16]);
        eprintln!("raw qs[0..8]={:?}", &block_bytes[16..24]);

        // CPU dequant single block
        let mut cpu = vec![0.0f32; 256];
        aether_dequant_q4_k_m(block_bytes.as_ptr() as *const c_void,
            cpu.as_mut_ptr() as *mut c_void, 1);

        // GPU dequant
        let d_blocks = aether_dev_alloc_u8(144);
        let d_out    = aether_dev_alloc_f32(256);
        aether_dev_h2d_u8(block_bytes.as_ptr() as i64, d_blocks, 144);
        aether_op_dequant_q4_k_m_f32_cuda(d_blocks, d_out, 1);
        aether_dev_sync();
        let mut gpu = vec![0.0f32; 256];
        aether_dev_d2h_f32(d_out, gpu.as_mut_ptr() as i64, 256);
        aether_dev_free_u8(d_blocks);
        aether_dev_free_f32(d_out);

        eprintln!("\nFirst 8 outputs (sub-block 0):");
        for i in 0..8 {
            eprintln!("  i={}: cpu={:>14.6e}  gpu={:>14.6e}  diff={:.3e}",
                i, cpu[i], gpu[i], (cpu[i] - gpu[i]).abs());
        }
        eprintln!("\nIndex 32 (sub-block 1):");
        for i in 32..40 {
            eprintln!("  i={}: cpu={:>14.6e}  gpu={:>14.6e}  diff={:.3e}",
                i, cpu[i], gpu[i], (cpu[i] - gpu[i]).abs());
        }
        eprintln!("\nIndex 128 (sub-block 4):");
        for i in 128..136 {
            eprintln!("  i={}: cpu={:>14.6e}  gpu={:>14.6e}  diff={:.3e}",
                i, cpu[i], gpu[i], (cpu[i] - gpu[i]).abs());
        }
        aether_gguf_close(h);
    }
}
