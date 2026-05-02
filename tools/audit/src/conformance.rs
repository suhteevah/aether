//! Aether language conformance suite.
//!
//! Two directories:
//! * `tests/aether/positive/` — files that must `--check` clean
//! * `tests/aether/negative/` — files that must fail with a specific
//!   `AE####` code; the expected code is encoded in the filename:
//!   `expect_AE0002_missing_brace.aether`.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub struct ConformanceCase {
    pub input: PathBuf,
    pub expectation: Expectation,
}

#[derive(Debug)]
pub enum Expectation { CheckOk, ExpectCode(String) }

#[derive(Debug)]
pub enum ConformanceResult {
    Pass,
    UnexpectedFailure { stderr: String },
    UnexpectedSuccess,
    WrongCode { expected: String, got_stderr: String },
    SpawnError(String),
}

pub fn cases(root: &Path) -> Vec<ConformanceCase> {
    let mut out = Vec::new();
    let pos = root.join("tests").join("aether").join("positive");
    if let Ok(entries) = std::fs::read_dir(&pos) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("aether") {
                out.push(ConformanceCase { input: p, expectation: Expectation::CheckOk });
            }
        }
    }
    let neg = root.join("tests").join("aether").join("negative");
    if let Ok(entries) = std::fs::read_dir(&neg) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("aether") { continue; }
            // Parse the AE#### code out of the filename.
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            let code = stem.split('_').find(|t| t.starts_with("AE") && t.len() >= 6
                && t[2..].chars().all(|c| c.is_ascii_digit()))
                .map(|s| s.to_string());
            if let Some(c) = code {
                out.push(ConformanceCase {
                    input: p,
                    expectation: Expectation::ExpectCode(c),
                });
            }
        }
    }
    out
}

pub fn run_case(root: &Path, case: &ConformanceCase) -> ConformanceResult {
    let aetherc = root.join("target").join("debug").join(if cfg!(windows) { "aetherc.exe" } else { "aetherc" });
    let out = match Command::new(&aetherc)
        .arg(&case.input).arg("--check").arg("--json-errors")
        .output() {
        Ok(o) => o,
        Err(e) => return ConformanceResult::SpawnError(format!("{}", e)),
    };
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    match &case.expectation {
        Expectation::CheckOk => {
            if out.status.success() {
                ConformanceResult::Pass
            } else {
                ConformanceResult::UnexpectedFailure { stderr }
            }
        }
        Expectation::ExpectCode(code) => {
            if out.status.success() {
                ConformanceResult::UnexpectedSuccess
            } else if stderr.contains(&format!("\"code\":\"{}\"", code)) {
                ConformanceResult::Pass
            } else {
                ConformanceResult::WrongCode { expected: code.clone(), got_stderr: stderr }
            }
        }
    }
}
