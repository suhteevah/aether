//! aether-asm — assemble GAS-syntax x86-64 .s into a Windows COFF .obj.
//!
//! Bootstrap implementation in Rust; Phase 5 rewrites in Aether.

use std::path::{Component, Path, PathBuf};

fn safe_under_cwd(p: &Path) -> std::io::Result<PathBuf> {
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
            "-o" => output = Some(PathBuf::from(it.next().expect("-o needs path"))),
            "-h" | "--help" => {
                eprintln!("aether-asm <input.s> -o <output.obj>");
                std::process::exit(0);
            }
            other if !other.starts_with('-') => input = Some(PathBuf::from(other)),
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
    }
    let input = input.expect("input .s file required");
    let in_canon = input.canonicalize().expect("input not found");
    let output = output.unwrap_or_else(|| input.with_extension("obj"));
    let out_canon = safe_under_cwd(&output).expect("output path invalid");

    let src = std::fs::read_to_string(&in_canon).expect("read input");
    let obj = match aether_asm::parse_gas(&src) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("aether-asm: parse error: {:?}", e);
            std::process::exit(1);
        }
    };
    let bytes = obj.write();
    std::fs::write(&out_canon, &bytes).expect("write output");
    eprintln!("[aether-asm] wrote {:?} ({} bytes)", out_canon, bytes.len());
}
