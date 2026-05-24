//! Quick probe — dump bge-large's vocab entries at IDs we expect to be
//! "the", "quick", "brown", "fox", and around 10760 (where our broken
//! tokenizer landed for "the").

use aether_rt::*;
use std::os::raw::c_int;

#[test]
#[ignore]
fn dump_bge_vocab_samples() {
    let path = std::env::var("AETHER_TEST_BGE_GGUF").unwrap_or_else(|_|
        "C:/Users/Matt/.ollama/models/blobs/sha256-92b37e50807d951e27ead73c059cf9c3b14941498e37dfde57271e19e6d411df"
            .to_string());
    if !std::path::Path::new(&path).exists() {
        eprintln!("[vocab] skipping — {} not present", path);
        return;
    }
    unsafe {
        let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
        assert!(h >= 0);
        let key = b"tokenizer.ggml.tokens";
        let n = aether_gguf_get_metadata_array_string_n(
            h, key.as_ptr() as i64, key.len() as c_int);
        eprintln!("[vocab] total tokens = {}", n);
        let mut buf = vec![0u8; 512];
        let print_at = |idx: c_int, buf: &mut [u8]| {
            let got = aether_gguf_get_metadata_array_string_get(
                h, key.as_ptr() as i64, key.len() as c_int, idx,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if got <= 0 { eprintln!("  [{}] <err {}>", idx, got); return; }
            let s = std::str::from_utf8(&buf[..got as usize]).unwrap_or("<utf8?>");
            eprintln!("  [{}] {:?} (bytes={:?})", idx, s, &buf[..got as usize]);
        };
        // Known good IDs from HF bert-base-uncased.
        for &id in &[0, 100, 101, 102, 103, 1996, 4248, 2829, 4419, 10760] {
            print_at(id, &mut buf);
        }
        // Print a slice around the start to see ## entries.
        eprintln!("[vocab] first 30:");
        for i in 0..30 {
            print_at(i, &mut buf);
        }
        // Print a few entries that should be ## continuations.
        eprintln!("[vocab] 1000..1010:");
        for i in 1000..1010 {
            print_at(i, &mut buf);
        }
        // Search for any "##" prefixed entry to confirm WordPiece vs SP.
        eprintln!("[vocab] scanning for ## entries:");
        let mut ct = 0;
        for i in 0..n {
            let got = aether_gguf_get_metadata_array_string_get(
                h, key.as_ptr() as i64, key.len() as c_int, i,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if got >= 2 && buf[0] == b'#' && buf[1] == b'#' {
                if ct < 6 {
                    let s = std::str::from_utf8(&buf[..got as usize]).unwrap_or("?");
                    eprintln!("  [{}] {:?}", i, s);
                }
                ct += 1;
            }
        }
        eprintln!("[vocab] ## entries total = {}", ct);

        // Also check what 'unhappiness' would match against — search for
        // a few entries starting with 'un'.
        eprintln!("[vocab] entries starting with 'un' (first 10):");
        let mut un_ct = 0;
        for i in 0..n {
            if un_ct >= 10 { break; }
            let got = aether_gguf_get_metadata_array_string_get(
                h, key.as_ptr() as i64, key.len() as c_int, i,
                buf.as_mut_ptr() as i64, buf.len() as c_int);
            if got >= 2 && buf[0] == b'u' && buf[1] == b'n' {
                let s = std::str::from_utf8(&buf[..got as usize]).unwrap_or("?");
                eprintln!("  [{}] {:?}", i, s);
                un_ct += 1;
            }
        }
    }
}
