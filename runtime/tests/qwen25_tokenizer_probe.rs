//! Probe Qwen2.5-7B GGUF metadata to verify the tokenizer keys we
//! need are present and shape as expected.

use std::os::raw::c_int;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_get_metadata_u32, aether_gguf_get_metadata_string,
    aether_gguf_get_metadata_array_string_n,
    aether_gguf_get_metadata_array_string_get,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

#[test]
fn list_qwen25_tokenizer_metadata() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // String-valued metadata we expect to find.
        for key in ["general.architecture", "general.name", "tokenizer.ggml.model"] {
            let mut buf = [0u8; 256];
            let n = aether_gguf_get_metadata_string(
                h, key.as_ptr() as i64, key.len() as c_int,
                buf.as_mut_ptr() as i64, buf.len() as c_int,
            );
            if n > 0 {
                let s = std::str::from_utf8(&buf[..n as usize]).unwrap();
                eprintln!("[meta-string] {} = {:?}", key, s);
            } else {
                eprintln!("[meta-string] {} -- not found (n={})", key, n);
            }
        }

        // U32-valued metadata.
        for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                    "tokenizer.ggml.padding_token_id", "qwen2.context_length",
                    "qwen2.block_count", "qwen2.attention.head_count"] {
            let v = aether_gguf_get_metadata_u32(h, key.as_ptr() as i64, key.len() as c_int);
            eprintln!("[meta-u32] {} = {}", key, v);
        }

        // String-array metadata (tokens + merges).
        for key in ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"] {
            let n = aether_gguf_get_metadata_array_string_n(
                h, key.as_ptr() as i64, key.len() as c_int,
            );
            eprintln!("[meta-strarray] {}: n = {}", key, n);
            if n > 0 {
                // Print first 5 and last 1 elements.
                for i in 0..n.min(5) {
                    let mut buf = [0u8; 256];
                    let nb = aether_gguf_get_metadata_array_string_get(
                        h, key.as_ptr() as i64, key.len() as c_int, i,
                        buf.as_mut_ptr() as i64, buf.len() as c_int,
                    );
                    if nb > 0 {
                        let s = std::str::from_utf8(&buf[..nb as usize]).unwrap_or("<not-utf8>");
                        eprintln!("  [{}] {:?}", i, s);
                    }
                }
                if n > 5 {
                    let mut buf = [0u8; 256];
                    let nb = aether_gguf_get_metadata_array_string_get(
                        h, key.as_ptr() as i64, key.len() as c_int, n - 1,
                        buf.as_mut_ptr() as i64, buf.len() as c_int,
                    );
                    if nb > 0 {
                        let s = std::str::from_utf8(&buf[..nb as usize]).unwrap_or("<not-utf8>");
                        eprintln!("  [{}] (last) {:?}", n - 1, s);
                    }
                }
            }
        }

        aether_gguf_close(h);
    }
}
