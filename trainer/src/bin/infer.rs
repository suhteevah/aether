//! aether-infer — load an AetherLM checkpoint and generate bytes from a prompt.

use std::io::Read;
use std::path::{Component, Path, PathBuf};

use trainer::config::ModelConfig;
use trainer::model::{forward, Model};
use trainer::rng::Rng;
use trainer::sample::sample_topk;

fn parse_meta(s: &str) -> ModelConfig {
    fn pick(s: &str, key: &str) -> Option<i64> {
        let needle = format!("\"{}\":", key);
        let i = s.find(&needle)?;
        let rest = &s[i + needle.len()..];
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        rest[..end].trim().parse().ok()
    }
    ModelConfig {
        vocab:    pick(s, "vocab").unwrap_or(256) as usize,
        d_model:  pick(s, "d_model").unwrap_or(64) as usize,
        n_layers: pick(s, "n_layers").unwrap_or(2) as usize,
        n_heads:  pick(s, "n_heads").unwrap_or(4) as usize,
        d_ff:     pick(s, "d_ff").unwrap_or(128) as usize,
        seq_len:  pick(s, "seq_len").unwrap_or(32) as usize,
    }
}

fn safe_path_under_cwd(p: &Path) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(Error::new(ErrorKind::InvalidInput,
            "path may not contain `..`"));
    }
    let canon = p.canonicalize()?;
    let cwd = std::env::current_dir()?.canonicalize()?;
    if !canon.starts_with(&cwd) {
        return Err(Error::new(ErrorKind::InvalidInput,
            format!("path {:?} escapes cwd", canon)));
    }
    Ok(canon)
}

fn main() {
    let mut stem: Option<PathBuf> = None;
    let mut prompt = String::from("the quick brown");
    let mut max_new: usize = 64;
    let mut temperature: f32 = 0.9;
    let mut top_k: usize = 40;
    let mut seed: u64 = 7;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ckpt" => stem = Some(PathBuf::from(it.next().unwrap())),
            "--prompt" => prompt = it.next().unwrap(),
            "--max-new" => max_new = it.next().unwrap().parse().unwrap(),
            "--temperature" => temperature = it.next().unwrap().parse().unwrap(),
            "--top-k" => top_k = it.next().unwrap().parse().unwrap(),
            "--seed" => seed = it.next().unwrap().parse().unwrap(),
            "-h" | "--help" => {
                eprintln!("aether-infer --ckpt PATH [--prompt STR] [--max-new N] [--temperature F] [--top-k N]");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }
    let stem = stem.expect("--ckpt required");

    let weights_path = safe_path_under_cwd(&stem.with_extension("weights")).expect("weights path");
    let meta_path    = safe_path_under_cwd(&stem.with_extension("meta")).expect("meta path");

    let meta = std::fs::read_to_string(&meta_path).expect("read meta");
    let cfg = parse_meta(&meta);
    eprintln!("[aether-infer] cfg: {:?}", cfg);

    let mut wf = std::fs::File::open(&weights_path).expect("open weights");
    let mut header = [0u8; 9];
    wf.read_exact(&mut header).expect("read header");
    assert_eq!(&header, b"AETHER01\n", "bad weights header");
    let mut count_buf = [0u8; 8];
    wf.read_exact(&mut count_buf).expect("read count");
    let n = u64::from_le_bytes(count_buf) as usize;
    let mut params = vec![0.0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(params.as_mut_ptr() as *mut u8, n * 4) };
    wf.read_exact(bytes).expect("read weights");

    let mut model = Model::new(cfg.clone(), 0);
    assert_eq!(model.params.len(), params.len(),
        "weight count mismatch: ckpt={} model={}", params.len(), model.params.len());
    model.params = params;

    let mut rng = Rng::new(seed);
    let mut buf: Vec<i32> = prompt.bytes().map(|b| b as i32).collect();
    print!("{}", prompt);
    let _ = std::io::Write::flush(&mut std::io::stdout());

    for _ in 0..max_new {
        // Take the trailing window of size seq_len.
        let win_start = buf.len().saturating_sub(cfg.seq_len);
        let window = &buf[win_start..];
        let mut padded = vec![0i32; cfg.seq_len];
        padded[..window.len()].copy_from_slice(window);
        let labels = padded.clone(); // unused for inference; cross_entropy still runs
        let (act, _loss) = forward(&model, &padded, &labels, 1);

        let last = window.len().saturating_sub(1);
        let v = cfg.vocab;
        let logits_row = &act.logits[last * v..(last + 1) * v];
        let next = sample_topk(logits_row, temperature, top_k, &mut rng);
        buf.push(next);
        let byte = next as u8;
        print!("{}", byte as char);
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    println!();
}
