//! Paged QwenSession parity test (FR-19.4-extra-deep).
//!
//! Constructs TWO QwenSession instances against the SAME Qwen2.5-7B GGUF —
//! one with contiguous KV (the prod path), one with paged KV (block_size=4,
//! identity page table).  Runs the same prompt through both and verifies
//! identical generated token IDs.
//!
//! With the identity-mapping page table the paged path's K/V access pattern
//! is byte-identical to the contiguous path; this test promotes the kernel-
//! level parity proven in `cuda_paged_kv_parity.rs` to the full Qwen forward
//! chain (RMS norm → Q/K/V matmul → RoPE → append_kv → attention → O matmul
//! → residual → FFN → LM head → argmax).
//!
//! Skipped without `--features cuda` AND without the Qwen2.5-7B GGUF.
//! AETHER_QWEN25_GGUF env var sets the path; default is matt-voice's
//! ollama blob location.
//!
//! roadmap: P19.4

#![cfg(feature = "cuda")]

use aether_rt::serving::QwenSession;

const DEFAULT_GGUF: &str = "C:/Users/Matt/.ollama/models/blobs/sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

fn gguf_path() -> Option<String> {
    let p = std::env::var("AETHER_QWEN25_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    if std::path::Path::new(&p).exists() {
        Some(p)
    } else {
        eprintln!("[paged-parity] GGUF not at {} — skipping", p);
        None
    }
}

#[test]
fn paged_vs_contiguous_qwen25_decode_parity() {
    let Some(path) = gguf_path() else { return; };

    let prompt = [9707usize, 11, 1879, 0];  // "Hello, world!" tokenized
    let max_tokens = 8usize;

    eprintln!("[paged-parity] loading contiguous session...");
    let t0 = std::time::Instant::now();
    let mut c_sess = QwenSession::new(&path).expect("contiguous session");
    eprintln!("[paged-parity] contiguous loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let t1 = std::time::Instant::now();
    let c_ids = c_sess.generate(&prompt, max_tokens, None);
    eprintln!("[paged-parity] contiguous decoded {} tokens in {:.3}s -> {:?}",
        c_ids.len(), t1.elapsed().as_secs_f32(), c_ids);
    drop(c_sess);  // free GPU memory before loading the paged session

    eprintln!("[paged-parity] loading paged session...");
    let t2 = std::time::Instant::now();
    let mut p_sess = QwenSession::new_paged(&path).expect("paged session");
    eprintln!("[paged-parity] paged loaded in {:.2}s", t2.elapsed().as_secs_f32());

    let t3 = std::time::Instant::now();
    let p_ids = p_sess.generate(&prompt, max_tokens, None);
    eprintln!("[paged-parity] paged decoded {} tokens in {:.3}s -> {:?}",
        p_ids.len(), t3.elapsed().as_secs_f32(), p_ids);

    assert_eq!(c_ids, p_ids,
        "paged and contiguous produced different token IDs!\n  contiguous: {:?}\n  paged: {:?}",
        c_ids, p_ids);
    eprintln!("[paged-parity] PASS — {} tokens identical in both modes", c_ids.len());
}
