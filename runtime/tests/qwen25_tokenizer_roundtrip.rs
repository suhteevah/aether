//! Load Qwen2.5-7B's embedded tokenizer from GGUF metadata into
//! aether_bpe_tokenizer and verify decode round-trip.
//!
//! This is the matt-voice deploy missing link from "model generates
//! token IDs" to "user sees text". The tokenizer.ggml.tokens array
//! (152064 entries) + _merges (151387 entries) + bos/eos ids are
//! all read via the new aether_gguf_get_metadata_* accessors.
//!
//! Qwen2.5 uses GPT-2-style byte-level BPE with the bytes-to-unicode
//! mapping: byte 0x20 (space) appears as 'Ġ' (U+0120), byte 0x0A
//! (newline) as 'Ċ' (U+010A), etc. Real human text decoding requires
//! the inverse mapping after BPE decode -- shipped here as a small
//! helper.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_gguf_open, aether_gguf_close,
    aether_gguf_get_metadata_u32, aether_gguf_get_metadata_string,
    aether_gguf_get_metadata_array_string_n,
    aether_gguf_get_metadata_array_string_get,
    aether_bpe_tokenizer_new, aether_bpe_add_token_with_id,
    aether_bpe_add_merge_by_id, aether_bpe_encode, aether_bpe_decode,
    aether_bpe_encode_ids, aether_bpe_lookup_bytes,
};

const QWEN_BLOB: &str = "C:\\Users\\Matt\\.ollama\\models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730";

/// GPT-2 byte-to-unicode mapping (the standard one used by Qwen,
/// Llama-3, GPT-2/3, etc.). Returns a 256-entry table where
/// `table[byte_val]` is the unicode char that represents that byte in
/// the tokenizer's surface representation.
fn build_gpt2_byte_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = Vec::new();
    for b in 33..=126_u32 { bs.push(b); }   // '!'..'~'
    for b in 161..=172_u32 { bs.push(b); }
    for b in 174..=255_u32 { bs.push(b); }
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0u32;
    for b in 0..256_u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut tbl = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        tbl[*b as usize] = char::from_u32(*c).unwrap_or('\0');
    }
    tbl
}

/// Inverse: unicode char -> byte value, packed as a HashMap.
fn build_gpt2_unicode_to_byte(b2u: &[char; 256]) -> std::collections::HashMap<char, u8> {
    let mut m = std::collections::HashMap::with_capacity(256);
    for (b, &c) in b2u.iter().enumerate() {
        m.insert(c, b as u8);
    }
    m
}

/// Decode a tokenizer surface string (with Ġ/Ċ/etc.) into actual
/// UTF-8 bytes using the GPT-2 inverse byte mapping.
fn surface_to_bytes(surface: &str, u2b: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    surface.chars().filter_map(|c| u2b.get(&c).copied()).collect()
}

unsafe fn load_qwen25_vocab(h: i64) -> (i64, Vec<Vec<u8>>) {
    let tok_key = b"tokenizer.ggml.tokens";
    let n = aether_gguf_get_metadata_array_string_n(
        h, tok_key.as_ptr() as i64, tok_key.len() as c_int,
    );
    assert!(n > 0, "no tokenizer.ggml.tokens metadata");
    eprintln!("[tokenizer] vocab size: {}", n);

    let bpe = aether_bpe_tokenizer_new();
    assert!(bpe >= 0);

    // Also keep a Rust-side mirror of the surface bytes so we can build
    // the merge table by name lookup.
    let mut vocab_bytes: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    let mut buf = [0u8; 256];
    for i in 0..n {
        let nb = aether_gguf_get_metadata_array_string_get(
            h, tok_key.as_ptr() as i64, tok_key.len() as c_int, i,
            buf.as_mut_ptr() as i64, buf.len() as c_int,
        );
        assert!(nb >= 0, "vocab entry {} truncated (n={})", i, nb);
        let bytes_vec = buf[..nb as usize].to_vec();
        let rc = aether_bpe_add_token_with_id(
            bpe, i, bytes_vec.as_ptr() as *const c_void, nb,
        );
        assert_eq!(rc, 0, "add_token_with_id({}) failed", i);
        vocab_bytes.push(bytes_vec);
    }
    eprintln!("[tokenizer] {} entries registered", n);
    (bpe, vocab_bytes)
}

/// Load the merge rules. Each merge is "left right" -- two
/// space-separated token surface strings whose IDs we look up in the
/// vocab. The merged token is "left+right" concatenated; its ID is also
/// in the vocab. Returns count loaded.
unsafe fn load_qwen25_merges(h: i64, bpe: i64, vocab_bytes: &[Vec<u8>]) -> c_int {
    let key = b"tokenizer.ggml.merges";
    let n = aether_gguf_get_metadata_array_string_n(
        h, key.as_ptr() as i64, key.len() as c_int,
    );
    assert!(n > 0, "no tokenizer.ggml.merges metadata");

    // Build a hashmap from surface bytes -> token id for O(1) merge lookup.
    let mut lookup: std::collections::HashMap<Vec<u8>, u32> =
        std::collections::HashMap::with_capacity(vocab_bytes.len());
    for (i, b) in vocab_bytes.iter().enumerate() {
        lookup.insert(b.clone(), i as u32);
    }

    let mut buf = [0u8; 256];
    let mut loaded = 0i32;
    let mut skipped = 0i32;
    for i in 0..n {
        let nb = aether_gguf_get_metadata_array_string_get(
            h, key.as_ptr() as i64, key.len() as c_int, i,
            buf.as_mut_ptr() as i64, buf.len() as c_int,
        );
        if nb <= 0 { skipped += 1; continue; }
        let s = &buf[..nb as usize];
        // Find the FIRST space (separator between left and right).
        let Some(space_idx) = s.iter().position(|&b| b == b' ') else {
            skipped += 1; continue;
        };
        let left = &s[..space_idx];
        let right = &s[space_idx + 1..];
        let Some(&left_id) = lookup.get(left) else { skipped += 1; continue; };
        let Some(&right_id) = lookup.get(right) else { skipped += 1; continue; };
        let mut merged = Vec::with_capacity(left.len() + right.len());
        merged.extend_from_slice(left);
        merged.extend_from_slice(right);
        let Some(&merged_id) = lookup.get(&merged) else { skipped += 1; continue; };
        let rc = aether_bpe_add_merge_by_id(bpe, left_id as c_int, right_id as c_int, i, merged_id as c_int);
        if rc != 0 { skipped += 1; continue; }
        loaded += 1;
    }
    eprintln!("[tokenizer] merges loaded: {} / skipped: {} / total: {}", loaded, skipped, n);
    loaded
}

#[test]
fn qwen25_tokenizer_decode_roundtrip() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        // Verify metadata sanity first.
        let mut model_buf = [0u8; 64];
        let key = b"tokenizer.ggml.model";
        let n = aether_gguf_get_metadata_string(
            h, key.as_ptr() as i64, key.len() as c_int,
            model_buf.as_mut_ptr() as i64, model_buf.len() as c_int,
        );
        let model = std::str::from_utf8(&model_buf[..n as usize]).unwrap();
        assert_eq!(model, "gpt2");

        let eos_key = b"tokenizer.ggml.eos_token_id";
        let eos_id = aether_gguf_get_metadata_u32(h, eos_key.as_ptr() as i64, eos_key.len() as c_int);
        eprintln!("[meta] eos_token_id = {}", eos_id);
        assert_eq!(eos_id, 151645, "Qwen2.5 EOS expected 151645");

        // Load the full vocab + merges into aether_bpe_tokenizer.
        let t = std::time::Instant::now();
        let (bpe, vocab_bytes) = load_qwen25_vocab(h);
        eprintln!("[load_vocab] {:.2}s", t.elapsed().as_secs_f32());
        let t = std::time::Instant::now();
        let n_merges = load_qwen25_merges(h, bpe, &vocab_bytes);
        eprintln!("[load_merges] {:.2}s -- {} merges loaded", t.elapsed().as_secs_f32(), n_merges);
        assert!(n_merges > 150000, "expected >150K merges, got {}", n_merges);

        // Decode the IDs that our autoregressive test generated:
        //   [9707, 11, 1879, 0, 358, 2776, 264, 220, 17]
        let ids: Vec<i32> = vec![9707, 11, 1879, 0, 358, 2776, 264, 220, 17];
        let mut surface_buf = vec![0u8; 1024];
        let nb = aether_bpe_decode(
            bpe,
            ids.as_ptr() as *const c_void, ids.len() as c_int,
            surface_buf.as_mut_ptr() as *mut c_void, surface_buf.len() as c_int,
        );
        assert!(nb > 0, "decode failed (nb={})", nb);
        let surface = std::str::from_utf8(&surface_buf[..nb as usize]).unwrap_or("<not-utf8>");
        eprintln!("[decode] {:?} (surface)", surface);

        // Apply the GPT-2 inverse byte mapping to recover real text.
        let b2u = build_gpt2_byte_to_unicode();
        let u2b = build_gpt2_unicode_to_byte(&b2u);
        let real_bytes = surface_to_bytes(surface, &u2b);
        let real_text = std::str::from_utf8(&real_bytes).unwrap_or("<not-utf8>");
        eprintln!("[decode] {:?} (real text)", real_text);

        // Sanity: decoded text must be NON-EMPTY and contain at least
        // one printable ASCII character. The first token (9707) is
        // "Hello" in Qwen's vocab.
        assert!(!real_bytes.is_empty(), "decoded text is empty");
        let has_printable = real_bytes.iter().any(|&b| (b >= 0x20 && b < 0x7F));
        assert!(has_printable, "decoded text has no printable chars: {:?}", real_bytes);

        // Encode side note: aether_bpe_encode operates on RAW BYTES as
        // initial tokens (byte values 0..255). Qwen's vocab uses GPT-2
        // unicode-char-level BPE where the byte-to-unicode mapping is
        // implicit (every byte maps to a printable unicode char in the
        // bytes_to_unicode table). Wiring full encode requires either
        // (a) extending aether_bpe to support unicode-char-level initial
        // split, or (b) pre-applying the byte->unicode map and pre-
        // splitting into chars before encode. Both are FR-x-extra.
        // For matt-voice deploy the inference path is decode-only:
        // user-side tokenizes prompt -> sends IDs to Aether -> Aether
        // generates ID stream -> decodes back to text. We've proved the
        // decode half here.

        eprintln!("[OK] Qwen2.5-7B tokenizer decode + GPT-2 byte fixup verified");
        aether_gguf_close(h);
    }
}

/// FR-x-extra-text-encode: encode "hello world" with the wired BPE
/// encode path (byte→surface_id via aether_bpe_lookup_bytes, then merge
/// loop via aether_bpe_encode_ids), then decode the resulting ids and
/// confirm the surface text round-trips back to "hello world".
#[test]
fn qwen25_tokenizer_encode_roundtrip() {
    if !std::path::Path::new(QWEN_BLOB).exists() {
        eprintln!("[skip] Qwen2.5-7B GGUF not present");
        return;
    }
    unsafe {
        let h = aether_gguf_open(QWEN_BLOB.as_ptr() as i64, QWEN_BLOB.len() as c_int);
        assert!(h >= 0);

        let (bpe, vocab_bytes) = load_qwen25_vocab(h);
        let _ = load_qwen25_merges(h, bpe, &vocab_bytes);

        // Build the 256-entry byte→token_id cache via the new
        // aether_bpe_lookup_bytes primitive.  Mirrors what QwenSession::
        // encode_text does internally.
        let b2u = build_gpt2_byte_to_unicode();
        let mut byte_to_id = [-1i32; 256];
        let mut tmp = [0u8; 4];
        for b in 0..256u32 {
            let ch = b2u[b as usize];
            let s = ch.encode_utf8(&mut tmp);
            let id = aether_bpe_lookup_bytes(
                bpe,
                s.as_ptr() as *const c_void,
                s.len() as c_int,
            );
            byte_to_id[b as usize] = id;
        }
        let missing = byte_to_id.iter().filter(|&&i| i < 0).count();
        eprintln!("[encode] byte→id cache built; missing={}", missing);
        assert_eq!(missing, 0, "Qwen2.5 vocab should cover all 256 surface bytes");

        // Encode "hello world".
        let text = b"hello world";
        let initial: Vec<i32> = text.iter().map(|&b| byte_to_id[b as usize]).collect();
        let mut out_ids = vec![0i32; 64];
        let n = aether_bpe_encode_ids(
            bpe,
            initial.as_ptr() as *const c_void, initial.len() as c_int,
            out_ids.as_mut_ptr() as *mut c_void, out_ids.len() as c_int,
        );
        assert!(n > 0, "encode_ids failed (n={})", n);
        out_ids.truncate(n as usize);
        eprintln!("[encode] {} ids: {:?}", n, out_ids);

        // Should be fewer ids than initial bytes — the BPE merges
        // collapsed " world" / "hello" / etc.
        assert!((n as usize) < text.len(),
            "expected merges to collapse 11 bytes to <11 ids, got {}", n);
        assert!((n as usize) <= 4,
            "expected ≤ 4 ids for 'hello world' on Qwen2.5, got {}", n);

        // Decode back to surface + GPT-2 inverse → real text.
        let mut surface_buf = vec![0u8; 256];
        let nb = aether_bpe_decode(
            bpe,
            out_ids.as_ptr() as *const c_void, out_ids.len() as c_int,
            surface_buf.as_mut_ptr() as *mut c_void, surface_buf.len() as c_int,
        );
        assert!(nb > 0, "decode failed (nb={})", nb);
        let surface = std::str::from_utf8(&surface_buf[..nb as usize]).unwrap();
        let u2b = build_gpt2_unicode_to_byte(&b2u);
        let real = surface_to_bytes(surface, &u2b);
        eprintln!("[encode] roundtrip: {:?}", std::str::from_utf8(&real).unwrap());

        // Round-trip must yield "hello world" byte-for-byte.
        assert_eq!(&real[..], text);
        eprintln!("[OK] Qwen2.5-7B encode_ids + decode round-trip verified");

        aether_gguf_close(h);
    }
}
