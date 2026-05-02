//! Runtime smoke for the Aether-only compile chain.
//!
//! For each `tests/runtime/*.aether` file, run `aetherc --emit=aether-bin`,
//! invoke the produced .exe, and assert the exit code matches the
//! `// expect: exit=N` annotation at the top of the file. This is the
//! strongest end-to-end check we can run on the asm backend without a GPU.

use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug)]
pub struct RuntimeCase {
    pub input: PathBuf,
    pub expected_exit: i32,
    pub expected_stdout_contains: Option<String>,
}

#[derive(Debug)]
pub enum RuntimeResult {
    Pass,
    BuildFailed(String),
    WrongExit { expected: i32, got: i32 },
    StdoutMissing { expected: String, got: String },
    SpawnError(String),
}

pub fn cases(root: &Path) -> Vec<RuntimeCase> {
    let dir = root.join("tests").join("runtime");
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&dir) else { return out; };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("aether") { continue; }
        let Ok(src) = std::fs::read_to_string(&p) else { continue; };
        let mut expected_exit: i32 = 0;
        let mut expected_stdout: Option<String> = None;
        for line in src.lines().take(10) {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("// expect: exit=") {
                if let Ok(n) = rest.trim().parse::<i32>() { expected_exit = n; }
            }
            if let Some(rest) = l.strip_prefix("// expect: stdout contains ") {
                expected_stdout = Some(rest.trim().to_string());
            }
        }
        out.push(RuntimeCase {
            input: p, expected_exit,
            expected_stdout_contains: expected_stdout,
        });
    }
    out
}

pub fn run_case(root: &Path, case: &RuntimeCase) -> RuntimeResult {
    let aetherc = root.join("target").join("debug")
        .join(if cfg!(windows) { "aetherc.exe" } else { "aetherc" });
    let out_dir = root.join("target").join("audit-tmp");
    let _ = std::fs::create_dir_all(&out_dir);
    let stem = case.input.file_stem().and_then(|s| s.to_str()).unwrap_or("case");
    let exe_path = out_dir.join(format!("{stem}.exe"));

    let build = Command::new(&aetherc)
        .arg(&case.input).arg("--emit=aether-bin")
        .arg("-o").arg(&exe_path).output();
    let build = match build {
        Ok(o) => o,
        Err(e) => return RuntimeResult::SpawnError(format!("aetherc spawn: {}", e)),
    };
    if !build.status.success() {
        return RuntimeResult::BuildFailed(
            String::from_utf8_lossy(&build.stderr).into_owned());
    }
    let run = match Command::new(&exe_path).output() {
        Ok(o) => o,
        Err(e) => return RuntimeResult::SpawnError(format!("exe spawn: {}", e)),
    };
    let got_exit = run.status.code().unwrap_or(-1);
    if got_exit != case.expected_exit {
        return RuntimeResult::WrongExit { expected: case.expected_exit, got: got_exit };
    }
    if let Some(needle) = &case.expected_stdout_contains {
        let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
        if !stdout.contains(needle) {
            return RuntimeResult::StdoutMissing {
                expected: needle.clone(),
                got: stdout,
            };
        }
    }
    RuntimeResult::Pass
}
