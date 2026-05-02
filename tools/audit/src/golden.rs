//! Golden artifact verification.
//!
//! For each input under `tests/golden/inputs/`, run aetherc and compare the
//! emitted artifact byte-for-byte against the committed `tests/golden/expected/`
//! file. Mismatches are diffed and reported. To regenerate the expected
//! files (when an intentional codegen change lands), run with `--update-golden`.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub struct GoldenCase {
    pub input: PathBuf,
    pub emit: &'static str,
    pub expected_suffix: &'static str,
}

#[derive(Debug)]
pub enum GoldenResult {
    Match,
    Mismatch { expected_path: PathBuf, got: String, expected: String },
    AetherError { stderr: String },
    Missing { expected_path: PathBuf },
}

pub fn cases(root: &Path) -> Vec<GoldenCase> {
    let inputs = root.join("tests").join("golden").join("inputs");
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&inputs) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("aether") {
                for (emit, suf) in [("mir", "mir"), ("asm", "s"), ("llvm-ir", "ll")] {
                    out.push(GoldenCase {
                        input: p.clone(),
                        emit,
                        expected_suffix: suf,
                    });
                }
            }
        }
    }
    out
}

pub fn run_case(root: &Path, case: &GoldenCase, update: bool) -> GoldenResult {
    let aetherc = root.join("target").join("debug").join(if cfg!(windows) { "aetherc.exe" } else { "aetherc" });
    let out_dir = root.join("target").join("audit-tmp");
    let _ = std::fs::create_dir_all(&out_dir);
    let stem = case.input.file_stem().and_then(|s| s.to_str()).unwrap_or("case");
    let out_path = out_dir.join(format!("{stem}.{}", case.expected_suffix));

    let result = Command::new(&aetherc)
        .arg(&case.input)
        .arg(format!("--emit={}", case.emit))
        .arg("-o").arg(&out_path)
        .output();

    let output = match result {
        Ok(o) => o,
        Err(e) => return GoldenResult::AetherError { stderr: format!("spawn: {}", e) },
    };
    if !output.status.success() {
        return GoldenResult::AetherError {
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        };
    }
    let got = match std::fs::read_to_string(&out_path) {
        Ok(t) => t,
        Err(e) => return GoldenResult::AetherError { stderr: format!("read out: {}", e) },
    };

    let expected_dir = root.join("tests").join("golden").join("expected");
    let expected_path = expected_dir.join(format!("{}.{}.expected", stem, case.expected_suffix));

    if update {
        let _ = std::fs::create_dir_all(&expected_dir);
        let _ = std::fs::write(&expected_path, &got);
        return GoldenResult::Match;
    }

    let expected = match std::fs::read_to_string(&expected_path) {
        Ok(t) => t,
        Err(_) => return GoldenResult::Missing { expected_path },
    };
    if normalize(&got) == normalize(&expected) {
        GoldenResult::Match
    } else {
        GoldenResult::Mismatch { expected_path, got, expected }
    }
}

/// Normalise line endings before comparison so Windows checkouts don't show
/// false diffs against CRLF expected files.
fn normalize(s: &str) -> String {
    s.replace("\r\n", "\n")
}
