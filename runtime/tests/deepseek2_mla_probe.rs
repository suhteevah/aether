//! Probe a DeepSeek-V2 GGUF — enumerate all metadata keys whose name starts
//! with `deepseek2.` and every tensor name in block 0.  Used to confirm the
//! MLA tensor surface for FR-17-extra-mla-fwd.
//!
//! Run with:
//!   cargo test --release -p aether_rt --test deepseek2_mla_probe \
//!     -- --ignored --nocapture
//!
//! Path is read from env var `AETHER_TEST_DEEPSEEK2_GGUF`.

use aether_rt::*;
use std::os::raw::c_int;

#[test]
#[ignore]
fn probe_deepseek2_mla_surface() {
    let path = std::env::var("AETHER_TEST_DEEPSEEK2_GGUF")
        .unwrap_or_else(|_|
            "C:/Users/Matt/.ollama/models/blobs/sha256-5ff0abeeac1d2dbdd5455c0b49ba3b29a9ce3c1fb181b2eef2e948689d55d046"
                .to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[probe] skipping — {} not present", path);
        return;
    }
    unsafe {
        let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
        assert!(h >= 0, "open failed: {}", h);

        // Probe a battery of candidate MLA-related metadata keys.
        let candidates = [
            "general.architecture",
            "deepseek2.embedding_length",
            "deepseek2.block_count",
            "deepseek2.feed_forward_length",
            "deepseek2.expert_feed_forward_length",
            "deepseek2.attention.head_count",
            "deepseek2.attention.head_count_kv",
            "deepseek2.attention.key_length",
            "deepseek2.attention.value_length",
            "deepseek2.attention.q_lora_rank",
            "deepseek2.attention.kv_lora_rank",
            "deepseek2.expert_count",
            "deepseek2.expert_used_count",
            "deepseek2.expert_shared_count",
            "deepseek2.expert_weights_scale",
            "deepseek2.expert_gating_func",
            "deepseek2.leading_dense_block_count",
            "deepseek2.rope.dimension_count",
            "deepseek2.rope.freq_base",
            "deepseek2.rope.scaling.factor",
            "deepseek2.rope.scaling.type",
            "deepseek2.rope.scaling.yarn_log_multiplier",
            "deepseek2.attention.layer_norm_rms_epsilon",
        ];
        eprintln!("\n[probe] deepseek2 metadata keys:");
        for k in &candidates {
            let u = aether_gguf_get_metadata_u32(h, k.as_ptr() as i64, k.len() as c_int);
            let f = aether_gguf_get_metadata_f32(h, k.as_ptr() as i64, k.len() as c_int);
            let mut buf = vec![0u8; 256];
            let nstr = aether_gguf_get_metadata_string(
                h, k.as_ptr() as i64, k.len() as c_int,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if u >= 0 {
                eprintln!("  u32  {:<55} = {}", k, u);
            } else if !f.is_nan() {
                eprintln!("  f32  {:<55} = {}", k, f);
            } else if nstr > 0 {
                let s = std::str::from_utf8(&buf[..nstr as usize]).unwrap_or("<utf8?>");
                eprintln!("  str  {:<55} = {:?}", k, s);
            } else {
                eprintln!("  ABS  {:<55}", k);
            }
        }

        // Enumerate every blk.0.* tensor.
        let n = aether_gguf_n_tensors(h);
        eprintln!("\n[probe] {} total tensors; blk.0.* tensors:", n);
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
            if !name.starts_with("blk.0.") && !name.starts_with("blk.1.")
                && !name.starts_with("token_embd")
                && !name.starts_with("output") {
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
