//! Probe: list block-0 + global tensors in matt-voice's Qwen2.5-7B GGUF
//! to verify the llama.cpp-convention names we'll need for the block
//! forward witness.

use std::os::raw::c_int;

// Call FFI symbols directly through the rlib module path -- aether_rt's
// #[no_mangle] extern fns are also accessible as Rust items.
use aether_rt::{
    aether_gguf_open, aether_gguf_close, aether_gguf_n_tensors,
    aether_gguf_get_tensor_name, aether_gguf_get_tensor_dtype,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_shape,
};

#[test]
fn list_qwen25_block0_tensors() {
    let qwen = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";
    if !std::path::Path::new(qwen).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    unsafe {
        let _ = c_int::default();  // silence c_int unused-warn if any
        let h = aether_gguf_open(qwen.as_ptr() as i64, qwen.len() as c_int);
        assert!(h >= 0);
        let n = aether_gguf_n_tensors(h);
        let mut blk0_names = Vec::new();
        let mut global_names = Vec::new();
        for i in 0..n {
            let mut buf = [0u8; 256];
            let nn = aether_gguf_get_tensor_name(h, i, buf.as_mut_ptr() as i64, buf.len() as c_int);
            if nn <= 0 { continue; }
            let name = std::str::from_utf8(&buf[..nn as usize]).unwrap_or("");
            let dt = aether_gguf_get_tensor_dtype(h, i);
            if name.starts_with("blk.0.") {
                blk0_names.push((name.to_string(), dt));
            }
            if !name.contains("blk.") {
                global_names.push((name.to_string(), dt));
            }
        }
        blk0_names.sort();
        global_names.sort();
        eprintln!("=== Qwen2.5-7B block 0 tensors ===");
        for (name, dt) in &blk0_names {
            let needle = name.as_bytes();
            let i = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
            let mut dims_buf = [0i64; 8];
            let nd = aether_gguf_get_tensor_shape(h, i, dims_buf.as_mut_ptr() as i64, 8);
            let shape: Vec<i64> = dims_buf[..nd as usize].to_vec();
            eprintln!("  {} (dtype={}, shape={:?})", name, dt, shape);
        }
        eprintln!("=== Qwen2.5-7B global tensors ===");
        for (name, dt) in &global_names {
            let needle = name.as_bytes();
            let i = aether_gguf_find_tensor_by_name(h, needle.as_ptr() as i64, needle.len() as c_int);
            let mut dims_buf = [0i64; 8];
            let nd = aether_gguf_get_tensor_shape(h, i, dims_buf.as_mut_ptr() as i64, 8);
            let shape: Vec<i64> = dims_buf[..nd as usize].to_vec();
            eprintln!("  {} (dtype={}, shape={:?})", name, dt, shape);
        }

        // Verify the name-lookup helper finds a known tensor.
        let needle = b"blk.0.attn_q.weight";
        let idx = aether_gguf_find_tensor_by_name(
            h, needle.as_ptr() as i64, needle.len() as c_int,
        );
        eprintln!("find_tensor_by_name(blk.0.attn_q.weight) -> {}", idx);
        assert!(idx >= 0, "expected blk.0.attn_q.weight to exist in Qwen2.5-7B");

        aether_gguf_close(h);
    }
}
