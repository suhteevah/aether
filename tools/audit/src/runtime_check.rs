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
    /// Build-mode flag passed to aetherc. `aether-bin` (default) goes
    /// asm → COFF → system-linker → libaether_rt-linked .exe. `pe-bin`
    /// goes asm → COFF → self-hosted PE32+ writer → kernel32-only .exe
    /// (no FFI). A test opts into the PE path with `// build-mode: pe-bin`.
    pub build_mode: String,
    /// Build-time precondition: if set, the audit will skip the case unless
    /// the named runtime feature is detected. Use `// requires: cuda`. The
    /// detection just probes for the cudart import in libaether_rt.a.
    pub requires: Option<String>,
}

#[derive(Debug)]
pub enum SkipReason { MissingFeature(String) }

#[derive(Debug)]
pub enum RuntimeResult {
    Pass,
    Skipped(String),
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
        let mut build_mode = "aether-bin".to_string();
        let mut requires: Option<String> = None;
        for line in src.lines().take(10) {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("// expect: exit=") {
                if let Ok(n) = rest.trim().parse::<i32>() { expected_exit = n; }
            }
            if let Some(rest) = l.strip_prefix("// expect: stdout contains ") {
                expected_stdout = Some(rest.trim().to_string());
            }
            if let Some(rest) = l.strip_prefix("// build-mode:") {
                build_mode = rest.trim().to_string();
            }
            if let Some(rest) = l.strip_prefix("// requires:") {
                requires = Some(rest.trim().to_string());
            }
        }
        out.push(RuntimeCase {
            input: p, expected_exit,
            expected_stdout_contains: expected_stdout,
            build_mode,
            requires,
        });
    }
    out
}

pub fn run_case(root: &Path, case: &RuntimeCase) -> RuntimeResult {
    // Skip if a feature precondition isn't satisfied. For `cuda` we look
    // for the `cudart64_*.dll` reference in libaether_rt.a (only present
    // when the runtime crate was built with `--features cuda`). Cheap to
    // check and avoids a confusing link-error report.
    if let Some(req) = &case.requires {
        if req == "cuda" {
            let lib = root.join("target").join("debug").join("libaether_rt.a");
            let has_cuda = std::fs::read(&lib).map(|bytes|
                bytes.windows(7).any(|w| w == b"cudart6") ||
                bytes.windows(6).any(|w| w == b"cublas")
            ).unwrap_or(false);
            if !has_cuda {
                return RuntimeResult::Skipped(
                    "libaether_rt.a not built with --features cuda".into());
            }
        }
    }

    let aetherc = root.join("target").join("debug")
        .join(if cfg!(windows) { "aetherc.exe" } else { "aetherc" });
    let out_dir = root.join("target").join("audit-tmp");
    let _ = std::fs::create_dir_all(&out_dir);
    let stem = case.input.file_stem().and_then(|s| s.to_str()).unwrap_or("case");
    let exe_path = out_dir.join(format!("{stem}.exe"));

    let emit_flag = format!("--emit={}", case.build_mode);
    let build = Command::new(&aetherc)
        .arg(&case.input).arg(&emit_flag)
        .arg("-o").arg(&exe_path).output();
    let build = match build {
        Ok(o) => o,
        Err(e) => return RuntimeResult::SpawnError(format!("aetherc spawn: {}", e)),
    };
    if !build.status.success() {
        return RuntimeResult::BuildFailed(
            String::from_utf8_lossy(&build.stderr).into_owned());
    }
    // For the pe-bin path the .exe imports `aether_rt.dll` by name. Make
    // sure the slim runtime DLL is sitting next to the .exe so the loader
    // finds it via the standard SafeDllSearchMode lookup.
    if case.build_mode == "pe-bin" {
        let dll_src = root.join("target").join("debug").join("aether_rt.dll");
        if dll_src.exists() {
            let dll_dst = out_dir.join("aether_rt.dll");
            let _ = std::fs::copy(&dll_src, &dll_dst);
        }
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
