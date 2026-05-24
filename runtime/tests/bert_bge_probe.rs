//! Probe a bge-large-en GGUF — enumerate metadata keys + blk.0.* + token-
//! level tensors so we know the exact tensor naming convention to load.
//!
//! Run with:
//!   cargo test --release -p aether_rt --test bert_bge_probe \
//!     -- --ignored --nocapture
//!
//! Path read from AETHER_TEST_BGE_GGUF env var, defaults to the ollama blob.

use aether_rt::*;
use std::os::raw::c_int;

#[test]
#[ignore]
fn probe_bge_bert_surface() {
    let path = std::env::var("AETHER_TEST_BGE_GGUF")
        .unwrap_or_else(|_|
            "C:/Users/Matt/.ollama/models/blobs/sha256-92b37e50807d951e27ead73c059cf9c3b14941498e37dfde57271e19e6d411df"
                .to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[probe] skipping — {} not present", path);
        return;
    }
    unsafe {
        let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
        assert!(h >= 0, "open failed: {}", h);

        // Candidate BERT metadata keys (per llama.cpp conventions).
        let candidates = [
            "general.architecture",
            "general.name",
            "bert.embedding_length",
            "bert.block_count",
            "bert.feed_forward_length",
            "bert.attention.head_count",
            "bert.attention.head_count_kv",
            "bert.attention.key_length",
            "bert.attention.value_length",
            "bert.attention.layer_norm_epsilon",
            "bert.attention.causal",
            "bert.context_length",
            "bert.pooling_type",
            "bert.token_types_count",
            "tokenizer.ggml.model",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.cls_token_id",
            "tokenizer.ggml.sep_token_id",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.padding_token_id",
            "tokenizer.ggml.mask_token_id",
        ];
        eprintln!("\n[probe] bert metadata keys:");
        for k in &candidates {
            let u = aether_gguf_get_metadata_u32(h, k.as_ptr() as i64, k.len() as c_int);
            let f = aether_gguf_get_metadata_f32(h, k.as_ptr() as i64, k.len() as c_int);
            let mut buf = vec![0u8; 256];
            let nstr = aether_gguf_get_metadata_string(
                h, k.as_ptr() as i64, k.len() as c_int,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if u >= 0 {
                eprintln!("  u32  {:<48} = {}", k, u);
            } else if !f.is_nan() {
                eprintln!("  f32  {:<48} = {}", k, f);
            } else if nstr > 0 {
                let s = std::str::from_utf8(&buf[..nstr as usize]).unwrap_or("<utf8?>");
                eprintln!("  str  {:<48} = {:?}", k, s);
            } else {
                eprintln!("  ABS  {:<48}", k);
            }
        }

        // Enumerate every blk.0.* + token-level tensor.
        let n = aether_gguf_n_tensors(h);
        eprintln!("\n[probe] {} total tensors; selected names:", n);
        let mut name_buf = vec![0u8; 256];
        let mut shape_buf = vec![0i64; 8];
        for i in 0..n {
            let nb = aether_gguf_get_tensor_name(h, i,
                name_buf.as_mut_ptr() as i64, name_buf.len() as c_int);
            if nb <= 0 { continue; }
            let name = match std::str::from_utf8(&name_buf[..nb as usize]) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if !name.starts_with("blk.0.") && !name.starts_with("token_embd")
                && !name.starts_with("position_embd")
                && !name.starts_with("token_types")
                && !name.starts_with("output_norm")
                && !name.starts_with("output")
                && !name.starts_with("pooler") {
                continue;
            }
            let dt = aether_gguf_get_tensor_dtype(h, i);
            let nd = aether_gguf_get_tensor_shape(h, i,
                shape_buf.as_mut_ptr() as i64, shape_buf.len() as c_int);
            let dims: Vec<i64> = shape_buf[..nd as usize].to_vec();
            eprintln!("  [{:3}] dt={:>2}  shape={:?}  {}", i, dt, dims, name);
        }
    }
}
