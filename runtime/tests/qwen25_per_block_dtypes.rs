//! Probe Qwen2.5-7B per-block tensor dtypes. We've been hardcoding
//! V proj and ffn_down as Q6_K based on block 0, but the dtype
//! varies by block (mixed-precision quantisation).

#![cfg(feature = "cuda")]

use std::os::raw::c_int;
use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_find_tensor_by_name, aether_gguf_get_tensor_dtype,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

fn dtype_name(dt: c_int) -> &'static str {
    match dt {
        0 => "F32", 1 => "F16", 2 => "BF16",
        12 => "Q4_K", 14 => "Q6_K",
        _ => "???",
    }
}

#[test]
fn list_per_block_dtypes() {
    if !std::path::Path::new(QWEN_BLOB).exists() { return; }
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        let tensors = ["attn_q.weight", "attn_k.weight", "attn_v.weight",
                       "attn_output.weight", "ffn_gate.weight", "ffn_up.weight",
                       "ffn_down.weight"];

        eprintln!("\nQwen2.5-7B per-block weight dtypes:\n");
        eprintln!("{:>3}  {}", "blk", tensors.iter().map(|s| format!("{:>14}", s)).collect::<Vec<_>>().join(""));

        for b in 0..28 {
            let mut row = vec![];
            for t in &tensors {
                let name = format!("blk.{}.{}", b, t);
                let idx = aether_gguf_find_tensor_by_name(
                    h, name.as_ptr() as i64, name.len() as c_int,
                );
                let dt = aether_gguf_get_tensor_dtype(h, idx);
                row.push(format!("{:>14}", dtype_name(dt)));
            }
            eprintln!("{:>3}  {}", b, row.join(""));
        }
        aether_gguf_close(h);
    }
}
