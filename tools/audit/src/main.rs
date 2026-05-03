//! aether-audit — single-command, structured codebase audit.
//!
//! Outputs both a human-readable report on stderr and a structured JSON
//! report on stdout (`--json`). Exit code: 0 on clean, 1 if any audit
//! dimension reports an error.
//!
//! Honesty contract: this tool is the source of truth for "what's actually
//! built vs stubbed". The numbers it prints are the numbers we're allowed
//! to claim. Don't add a stub here without flagging it via Self.

use std::io::Write;
use std::path::PathBuf;

mod conformance;
mod golden;
mod runtime_check;
mod scan;
mod sloc;

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn find_workspace_root() -> PathBuf {
    // Walk up from the executable's location looking for a Cargo.toml that
    // has a [workspace] table.
    let mut p = std::env::current_dir().unwrap();
    loop {
        let candidate = p.join("Cargo.toml");
        if let Ok(s) = std::fs::read_to_string(&candidate) {
            if s.contains("[workspace]") { return p; }
        }
        if !p.pop() { break; }
    }
    std::env::current_dir().unwrap()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let update_golden = args.iter().any(|a| a == "--update-golden");
    let only: Option<&str> = args.iter()
        .position(|a| a == "--only")
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()));

    let root = find_workspace_root();
    let mut errors: Vec<String> = Vec::new();
    let mut report = String::new();

    macro_rules! section {
        ($name:expr, $body:block) => {{
            if only.map_or(true, |o| o == $name) { $body }
        }};
    }

    let mut writeln_human = |s: String| {
        let _ = writeln!(std::io::stderr(), "{}", s);
        report.push_str(&s);
        report.push('\n');
    };

    writeln_human(format!("aether-audit  workspace={}", root.display()));
    writeln_human(format!("{:=<72}", ""));

    // ---- 1. SLOC ----
    let sloc_report = sloc::count_workspace(&root);
    let mut sloc_json = String::from("[");
    section!("sloc", {
        writeln_human("\n[1/5] SLOC".into());
        let mut crates: Vec<_> = sloc_report.by_crate.iter().collect();
        crates.sort_by_key(|(k, _)| (*k).clone());
        for (k, fs) in &crates {
            writeln_human(format!("  {:<14} files? code={:>5} comment={:>5} blank={:>4} total={:>5}",
                k, fs.lines_code, fs.lines_comment, fs.lines_blank, fs.lines_total));
            if !sloc_json.ends_with('[') { sloc_json.push(','); }
            sloc_json.push_str(&format!(
                "{{\"crate\":{},\"code\":{},\"comment\":{},\"blank\":{},\"total\":{}}}",
                json_str(k), fs.lines_code, fs.lines_comment, fs.lines_blank, fs.lines_total));
        }
        let total = sloc_report.total();
        writeln_human(format!("  {:<14}        code={:>5} comment={:>5} blank={:>4} total={:>5}",
            "(total)", total.lines_code, total.lines_comment, total.lines_blank, total.lines_total));
    });
    sloc_json.push(']');

    // ---- 2. Honesty scan ----
    let scan_report = scan::scan_workspace(&root);
    let mut scan_json = String::from("{");
    section!("scan", {
        writeln_human("\n[2/5] Honesty scan".into());
        for kind in [
            scan::FindingKind::Todo,
            scan::FindingKind::Unimplemented,
            scan::FindingKind::Unreachable,
            scan::FindingKind::Panic,
            scan::FindingKind::Unsafe,
            scan::FindingKind::IgnoredTest,
            scan::FindingKind::StubReturn,
            scan::FindingKind::PhaseMarker,
        ] {
            let n = scan_report.count(kind);
            writeln_human(format!("  {:<16} {:>4}", kind.as_str(), n));
            if !scan_json.ends_with('{') { scan_json.push(','); }
            scan_json.push_str(&format!("{}:{}", json_str(kind.as_str()), n));
        }
        // Always surface the high-signal kinds in full so a reader can audit
        // the claim "X is implemented" against the actual code paths.
        for kind in [scan::FindingKind::Todo, scan::FindingKind::Unimplemented,
                     scan::FindingKind::Unreachable, scan::FindingKind::IgnoredTest,
                     scan::FindingKind::Panic] {
            for f in scan_report.by_kind(kind).take(20) {
                writeln_human(format!("    {} {}:{}  {}",
                    kind.as_str(), f.path.display(), f.line, f.text));
            }
        }
        // Stub returns are noisy (every Phase-1 extern hits this) — surface
        // first 8 plus a count tail so the inventory is honest without
        // drowning the report.
        let stubs: Vec<_> = scan_report.by_kind(scan::FindingKind::StubReturn).collect();
        for f in stubs.iter().take(8) {
            writeln_human(format!("    stub_return {}:{}  {}", f.path.display(), f.line, f.text));
        }
        if stubs.len() > 8 {
            writeln_human(format!("    ... and {} more stub_return findings",
                stubs.len() - 8));
        }
        writeln_human(format!("  files_scanned={} bytes_scanned={}",
            scan_report.files_scanned, scan_report.bytes_scanned));
    });
    scan_json.push('}');

    // ---- 3. Test census + run ----
    let mut test_json = String::from("[");
    section!("tests", {
        writeln_human("\n[3/5] Workspace tests".into());
        let test_out = std::process::Command::new("cargo")
            .arg("test").arg("--workspace").arg("--quiet").arg("--no-fail-fast")
            .current_dir(&root).output();
        match test_out {
            Ok(o) => {
                let combined = String::from_utf8_lossy(&o.stderr).into_owned()
                    + &String::from_utf8_lossy(&o.stdout);
                let mut total_passed = 0u32; let mut total_failed = 0u32;
                for line in combined.lines() {
                    if let Some(rest) = line.strip_prefix("test result: ok. ") {
                        if let Some((n, _)) = rest.split_once(" passed") {
                            if let Ok(p) = n.trim().parse::<u32>() { total_passed += p; }
                        }
                    }
                    if let Some(rest) = line.strip_prefix("test result: FAILED. ") {
                        // count reported failures
                        for tok in rest.split_whitespace() {
                            if let Some(stripped) = tok.strip_suffix(';') {
                                if let Ok(n) = stripped.parse::<u32>() { total_failed += n; }
                            }
                        }
                    }
                }
                writeln_human(format!("  passed={}  failed={}  status={}",
                    total_passed, total_failed,
                    if o.status.success() { "OK" } else { "FAIL" }));
                test_json.push_str(&format!(
                    "{{\"passed\":{},\"failed\":{},\"ok\":{}}}",
                    total_passed, total_failed, o.status.success()));
                if !o.status.success() {
                    errors.push("workspace tests failed".into());
                }
            }
            Err(e) => {
                writeln_human(format!("  cargo test spawn failed: {}", e));
                errors.push(format!("cargo test spawn: {}", e));
            }
        }
    });
    test_json.push(']');

    // ---- 4. Golden artifacts ----
    let mut golden_json = String::from("[");
    section!("golden", {
        writeln_human("\n[4/5] Golden artifacts".into());
        let cases = golden::cases(&root);
        if cases.is_empty() {
            writeln_human("  (no golden cases — populate tests/golden/inputs/ to enable)".into());
        }
        for c in cases {
            let r = golden::run_case(&root, &c, update_golden);
            let stem = c.input.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let (status, detail, ok) = match &r {
                golden::GoldenResult::Match => ("OK", String::new(), true),
                golden::GoldenResult::Mismatch { expected_path, .. } =>
                    ("MISMATCH", format!(" expected={}", expected_path.display()), false),
                golden::GoldenResult::AetherError { stderr } =>
                    ("ERR", format!(" stderr={}", stderr.lines().next().unwrap_or("")), false),
                golden::GoldenResult::Missing { expected_path } =>
                    ("MISSING", format!(" {} (run with --update-golden to create)", expected_path.display()), false),
            };
            writeln_human(format!("  {} --emit={:<7}  {}{}", stem, c.emit, status, detail));
            if !ok { errors.push(format!("golden {} {}", stem, status)); }
            if !golden_json.ends_with('[') { golden_json.push(','); }
            golden_json.push_str(&format!(
                "{{\"case\":{},\"emit\":{},\"status\":{},\"ok\":{}}}",
                json_str(stem), json_str(c.emit), json_str(status), ok));
        }
    });
    golden_json.push(']');

    // ---- 5. Aether language conformance ----
    let mut conf_json = String::from("[");
    section!("conformance", {
        writeln_human("\n[5/5] Aether language conformance".into());
        let cases = conformance::cases(&root);
        if cases.is_empty() {
            writeln_human("  (no conformance cases — populate tests/aether/{positive,negative}/)".into());
        }
        for c in cases {
            let r = conformance::run_case(&root, &c);
            let stem = c.input.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let (status, ok) = match &r {
                conformance::ConformanceResult::Pass => ("OK", true),
                conformance::ConformanceResult::UnexpectedFailure { .. } => ("FAIL_UNEXPECTED", false),
                conformance::ConformanceResult::UnexpectedSuccess => ("PASSED_BUT_SHOULD_FAIL", false),
                conformance::ConformanceResult::WrongCode { expected, .. } => {
                    let _ = expected; ("WRONG_CODE", false)
                }
                conformance::ConformanceResult::SpawnError(_) => ("SPAWN_ERR", false),
            };
            writeln_human(format!("  {:<40}  {}", stem, status));
            if !ok { errors.push(format!("conformance {} {}", stem, status)); }
            if !conf_json.ends_with('[') { conf_json.push(','); }
            conf_json.push_str(&format!(
                "{{\"case\":{},\"status\":{},\"ok\":{}}}",
                json_str(stem), json_str(status), ok));
        }
    });
    conf_json.push(']');

    // ---- 6. Runtime smoke through the Aether-only compile chain ----
    let mut runtime_json = String::from("[");
    section!("runtime", {
        writeln_human("\n[6/6] Runtime: aether-bin chain end-to-end".into());
        let cases = runtime_check::cases(&root);
        if cases.is_empty() {
            writeln_human("  (no runtime cases — populate tests/runtime/*.aether)".into());
        }
        for c in cases {
            let r = runtime_check::run_case(&root, &c);
            let stem = c.input.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
            let (status, detail, ok) = match &r {
                runtime_check::RuntimeResult::Pass =>
                    ("OK", format!(" exit={}", c.expected_exit), true),
                runtime_check::RuntimeResult::Skipped(why) =>
                    ("SKIP", format!(" {}", why), true),
                runtime_check::RuntimeResult::BuildFailed(_) =>
                    ("BUILD_FAIL", String::new(), false),
                runtime_check::RuntimeResult::WrongExit { expected, got } =>
                    ("WRONG_EXIT", format!(" expected={} got={}", expected, got), false),
                runtime_check::RuntimeResult::StdoutMissing { expected, .. } =>
                    ("STDOUT_MISS", format!(" needle=\"{}\"", expected), false),
                runtime_check::RuntimeResult::SpawnError(e) =>
                    ("SPAWN_ERR", format!(" {}", e), false),
            };
            writeln_human(format!("  {:<32}  {}{}", stem, status, detail));
            if !ok { errors.push(format!("runtime {} {}", stem, status)); }
            if !runtime_json.ends_with('[') { runtime_json.push(','); }
            runtime_json.push_str(&format!(
                "{{\"case\":{},\"status\":{},\"ok\":{}}}",
                json_str(stem), json_str(status), ok));
        }
    });
    runtime_json.push(']');

    // ---- summary ----
    writeln_human("\n=== summary ===".into());
    writeln_human(format!("  errors: {}", errors.len()));
    for e in &errors { writeln_human(format!("    - {}", e)); }

    if json {
        let body = format!(
            "{{\"sloc\":{},\"scan\":{},\"tests\":{},\"golden\":{},\"conformance\":{},\"runtime\":{},\"errors\":{}}}",
            sloc_json, scan_json,
            test_json.trim_start_matches('[').trim_end_matches(']'),
            golden_json, conf_json, runtime_json,
            errors.len(),
        );
        let _ = std::io::stdout().write_all(body.as_bytes());
        let _ = std::io::stdout().write_all(b"\n");
    }

    if !errors.is_empty() {
        std::process::exit(1);
    }
}
