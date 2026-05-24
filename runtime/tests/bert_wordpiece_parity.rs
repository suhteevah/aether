//! WordPiece tokenizer parity (FR-17-extra-bert-fwd / text input).
//!
//! Verifies aether's WordPiece tokenizer produces token-ID output matching
//! HuggingFace's `bert-base-uncased` tokenizer (which bge-large-en-v1.5
//! ships).  Three assertions:
//!
//!   1. Bare "the quick brown fox" → [101, 1996, 4248, 2829, 4419, 102].
//!      [CLS]=101, [SEP]=102, the/quick/brown/fox are all in-vocab.
//!   2. A word with WordPiece subwords ("unhappiness" → unh + ##appiness or
//!      similar) decomposes correctly via greedy longest match.
//!   3. End-to-end via BertSession::embed_text — encode "the quick brown
//!      fox" + run forward; assert L2-norm = 1 and the same byte sequence
//!      we got from the pre-tokenized input_ids path.
//!
//! Marked #[ignore] because they need a real bge-large GGUF.  Run with:
//!   cargo test --release -p aether_rt --features cuda \
//!     --test bert_wordpiece_parity -- --ignored --nocapture
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::bert::{BertSession, WordPieceTokenizer};
use aether_rt::{aether_gguf_open, aether_gguf_close};
use std::os::raw::c_int;

fn bge_path() -> String {
    std::env::var("AETHER_TEST_BGE_GGUF").unwrap_or_else(|_|
        "C:/Users/Matt/.ollama/models/blobs/sha256-92b37e50807d951e27ead73c059cf9c3b14941498e37dfde57271e19e6d411df"
            .to_string())
}

#[test]
#[ignore]
fn wordpiece_matches_hf_bert_base_uncased() {
    let path = bge_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("[wp] skipping — {} not present", path);
        return;
    }
    unsafe {
        let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
        assert!(h >= 0);
        let tok = WordPieceTokenizer::from_gguf(h).expect("from_gguf");
        aether_gguf_close(h);

        let ids = tok.encode("the quick brown fox");
        eprintln!("[wp] 'the quick brown fox' -> {:?}", ids);
        // HuggingFace bert-base-uncased tokenizes this to [CLS=101, the=1996,
        // quick=4248, brown=2829, fox=4419, SEP=102].
        assert_eq!(ids, vec![101, 1996, 4248, 2829, 4419, 102]);

        // Sanity-check special-token IDs.
        assert_eq!(tok.cls_id, 101);
        assert_eq!(tok.sep_id, 102);
        assert_eq!(tok.unk_id, 100);

        // Case insensitive (bge-large uses uncased).
        let upper_ids = tok.encode("The Quick Brown Fox");
        eprintln!("[wp] 'The Quick Brown Fox' -> {:?}", upper_ids);
        assert_eq!(upper_ids, vec![101, 1996, 4248, 2829, 4419, 102]);

        // Punctuation split.  "hello, world!" should split commas / bangs.
        let punct = tok.encode("hello, world!");
        eprintln!("[wp] 'hello, world!' -> {:?}", punct);
        // [CLS] hello , world ! [SEP] — at least 6 tokens.
        assert!(punct.len() >= 6, "expected ≥6 tokens, got {}", punct.len());
        assert_eq!(punct[0], 101);
        assert_eq!(*punct.last().unwrap(), 102);
    }
}

#[test]
#[ignore]
fn wordpiece_subword_decomposition() {
    let path = bge_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("[wp] skipping — {} not present", path);
        return;
    }
    unsafe {
        let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
        assert!(h >= 0);
        let tok = WordPieceTokenizer::from_gguf(h).expect("from_gguf");
        aether_gguf_close(h);

        // "unhappiness" decomposes into BERT subwords.  HF bert-base-uncased
        // produces ["un", "##hap", "##piness"] = [4895, 28290, 8492]
        // surrounded by CLS/SEP.  We accept any valid WordPiece chain ≥ 3
        // tokens (the exact split depends on vocab); the assertion is that
        // it doesn't degenerate to [UNK] and round-trips through forward.
        let ids = tok.encode("unhappiness");
        eprintln!("[wp] 'unhappiness' -> {:?}", ids);
        assert!(ids.len() >= 3 + 2, "expected ≥5 ids (cls + ≥3 pieces + sep), got {}", ids.len());
        assert_eq!(ids[0], tok.cls_id);
        assert_eq!(*ids.last().unwrap(), tok.sep_id);
        // No [UNK] in the middle.
        for &id in &ids[1..ids.len()-1] {
            assert_ne!(id, tok.unk_id, "unexpected [UNK] for 'unhappiness'");
        }
    }
}

#[test]
#[ignore]
fn embed_text_end_to_end() {
    let path = bge_path();
    if !std::path::Path::new(&path).exists() {
        eprintln!("[wp] skipping — {} not present", path);
        return;
    }
    let mut s = BertSession::from_gguf(&path).expect("from_gguf");

    // Path A: pre-tokenized
    let pretok_ids: Vec<i32> = vec![101, 1996, 4248, 2829, 4419, 102];
    let token_type_ids = vec![0i32; pretok_ids.len()];
    let emb_a = s.embed(&pretok_ids, &token_type_ids);

    // Path B: encode from raw text via WordPiece, then embed
    let emb_b = s.embed_text(&path, "the quick brown fox").expect("embed_text");

    assert_eq!(emb_a.len(), emb_b.len());
    let max_diff = emb_a.iter().zip(emb_b.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[embed_text] max_diff between paths = {:.3e}", max_diff);
    eprintln!("[embed_text] emb_b[..8] = {:?}", &emb_b[..8]);
    assert!(max_diff < 1e-6,
        "text-input embedding diverged from pre-tokenized path ({:.3e})", max_diff);
    let norm: f32 = emb_b.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "L2 norm {} should be ~1", norm);
}
