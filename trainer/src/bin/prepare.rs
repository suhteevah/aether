//! aether-prepare — copy a UTF-8 text file to a flat byte stream the trainer
//! can mmap. Today this is a verbatim copy with size + path validation. Future
//! versions can add normalisation / whitespace-collapsing options.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

fn safe(p: &Path) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(Error::new(ErrorKind::InvalidInput, "path may not contain `..`"));
    }
    let cwd = std::env::current_dir()?.canonicalize()?;
    let parent = p.parent().filter(|q| !q.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let parent_canon = parent.canonicalize()?;
    if !parent_canon.starts_with(&cwd) {
        return Err(Error::new(ErrorKind::InvalidInput, "path escapes cwd"));
    }
    let name = p.file_name().ok_or_else(|| Error::new(ErrorKind::InvalidInput, "missing file name"))?;
    Ok(parent_canon.join(name))
}

fn main() {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--in" => input = Some(PathBuf::from(it.next().unwrap())),
            "--out" => output = Some(PathBuf::from(it.next().unwrap())),
            "-h" | "--help" => {
                eprintln!("aether-prepare --in PATH --out PATH");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }
    let input = input.expect("--in required");
    let output = output.expect("--out required");

    let in_canon = input.canonicalize().expect("input not found");
    let out_canon = safe(&output).expect("output path invalid");

    const MAX: u64 = 4 * 1024 * 1024 * 1024;
    let meta = std::fs::metadata(&in_canon).expect("stat input");
    if !meta.is_file() { panic!("--in must be a regular file"); }
    if meta.len() > MAX { panic!("input too large ({} bytes)", meta.len()); }

    let mut buf = Vec::with_capacity(meta.len() as usize);
    std::fs::File::open(&in_canon).expect("open").read_to_end(&mut buf).expect("read");
    let mut wf = std::fs::File::create(&out_canon).expect("create output");
    wf.write_all(&buf).expect("write output");

    eprintln!("[aether-prepare] {} bytes -> {:?}", buf.len(), out_canon);
}
