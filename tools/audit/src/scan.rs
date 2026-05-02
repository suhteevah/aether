//! Source-tree scanner. Finds patterns the audit cares about:
//! stubs, todos, panics, unsafe blocks, no-op extern functions,
//! ignored tests, "Phase N" forward-looking comments.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Finding {
    pub path: PathBuf,
    pub line: u32,
    pub kind: FindingKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindingKind {
    Todo,
    Unimplemented,
    Panic,
    Unreachable,
    Unsafe,
    IgnoredTest,
    PhaseMarker,
    StubReturn,
    UnusedField,
}

impl FindingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FindingKind::Todo => "todo",
            FindingKind::Unimplemented => "unimplemented",
            FindingKind::Panic => "panic",
            FindingKind::Unreachable => "unreachable",
            FindingKind::Unsafe => "unsafe",
            FindingKind::IgnoredTest => "ignored_test",
            FindingKind::PhaseMarker => "phase_marker",
            FindingKind::StubReturn => "stub_return",
            FindingKind::UnusedField => "unused_field",
        }
    }
}

#[derive(Default)]
pub struct ScanReport {
    pub findings: Vec<Finding>,
    pub files_scanned: u32,
    pub bytes_scanned: u64,
}

impl ScanReport {
    pub fn count(&self, kind: FindingKind) -> usize {
        self.findings.iter().filter(|f| f.kind == kind).count()
    }

    pub fn by_kind(&self, kind: FindingKind) -> impl Iterator<Item = &Finding> {
        self.findings.iter().filter(move |f| f.kind == kind)
    }
}

pub fn scan_workspace(root: &Path) -> ScanReport {
    let mut report = ScanReport::default();
    walk(root, root, &mut report);
    report
}

fn walk(root: &Path, dir: &Path, report: &mut ScanReport) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Skip build artifacts, vendored deps, hidden dirs, and our own
        // checkpoints/golden output dirs (which are inputs to other checks).
        if matches!(name.as_ref(), "target" | ".git" | "checkpoints" | "node_modules") {
            continue;
        }
        if name.starts_with('.') { continue; }
        if path.is_dir() {
            walk(root, &path, report);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "rs" | "aether") {
                scan_file(root, &path, report);
            }
        }
    }
}

fn scan_file(root: &Path, path: &Path, report: &mut ScanReport) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return,
    };
    report.files_scanned += 1;
    report.bytes_scanned += text.len() as u64;
    let rel = path.strip_prefix(root).unwrap_or(path).to_path_buf();
    // Don't audit the audit tool itself for findings — we'd find ourselves.
    if rel.starts_with("tools") && rel.to_string_lossy().contains("audit") {
        return;
    }
    let is_rust = path.extension().and_then(|e| e.to_str()) == Some("rs");

    // Track whether we're inside a comment so we don't flag literal mentions
    // of `todo!()` inside doc comments etc. We do still flag PhaseMarker in
    // comments deliberately — the whole point is to surface roadmap items.
    for (idx, raw) in text.lines().enumerate() {
        let line = idx as u32 + 1;
        let trimmed = raw.trim();
        let in_comment = trimmed.starts_with("//") || trimmed.starts_with("///")
            || trimmed.starts_with("//!") || trimmed.starts_with('*');

        // PhaseMarker — looks for "Phase 1", "Phase 2" etc. anywhere.
        if has_phase_marker(raw) {
            report.findings.push(Finding {
                path: rel.clone(), line, kind: FindingKind::PhaseMarker,
                text: raw.trim().to_string(),
            });
        }

        if !is_rust { continue; }
        if in_comment { continue; }

        if let Some(kind) = classify_rust_line(raw) {
            report.findings.push(Finding {
                path: rel.clone(), line, kind, text: raw.trim().to_string(),
            });
        }
    }
}

fn has_phase_marker(s: &str) -> bool {
    // Match "Phase 1", "Phase 2", ..., case-sensitive to avoid false hits.
    let bytes = s.as_bytes();
    let mut i = 0;
    while i + 7 <= bytes.len() {
        if &bytes[i..i + 6] == b"Phase " {
            let next = bytes[i + 6];
            if next.is_ascii_digit() { return true; }
        }
        i += 1;
    }
    false
}

fn classify_rust_line(raw: &str) -> Option<FindingKind> {
    let s = raw.trim();
    if s.starts_with("//") { return None; }
    // Raw strings are almost always test fixtures embedding fake fn syntax.
    if s.contains("r#\"") || s.contains("r\"") { return None; }

    // Honest stubs we want surfaced:
    if s.contains("todo!(") { return Some(FindingKind::Todo); }
    if s.contains("unimplemented!(") { return Some(FindingKind::Unimplemented); }
    if s.contains("unreachable!(") { return Some(FindingKind::Unreachable); }

    // panic! is interesting but noisy — only flag explicit ones, not .unwrap().
    if s.contains("panic!(") { return Some(FindingKind::Panic); }

    // Unsafe blocks (not `unsafe fn` declarations — those are FFI surface).
    if s.contains("unsafe {") || s == "unsafe" { return Some(FindingKind::Unsafe); }
    // `unsafe extern "C" fn` is a no-op extern symbol — flag explicitly only
    // when the body is `{ 0 }` etc., which we catch via StubReturn below.

    // Ignored tests
    if s.starts_with("#[ignore]") || s.contains("#[ignore =") {
        return Some(FindingKind::IgnoredTest);
    }

    // No-op stub return: a complete `fn ... { 0 }` one-liner. Restricted to
    // lines that actually declare a function so we don't flag every block
    // expression with a literal body.
    if is_fn_declaration(s) && has_trivial_oneliner_body(s) {
        return Some(FindingKind::StubReturn);
    }

    None
}

fn is_fn_declaration(s: &str) -> bool {
    // Match either a top-level `fn name(` or `pub fn name(` or `extern "C" fn`,
    // optionally with `unsafe` / `pub unsafe` etc. We just look for the `fn `
    // keyword followed by a name and an open paren on the same line.
    let mut idx = 0;
    while let Some(off) = s[idx..].find("fn ") {
        let pos = idx + off;
        // Must be at start, or preceded by space/keyword, never inside an ident.
        let prev_ok = pos == 0
            || s.as_bytes()[pos - 1] == b' '
            || s.as_bytes()[pos - 1] == b'\t';
        if prev_ok && s[pos..].contains('(') {
            return true;
        }
        idx = pos + 3;
    }
    false
}

/// Recognise function bodies of the form `... { 0 }`, `... { /* x */ 0 }`,
/// `... {}`. Requires both `{` and the matching `}` on the same line.
fn has_trivial_oneliner_body(s: &str) -> bool {
    let Some(open) = s.find('{') else { return false; };
    let Some(close) = s.rfind('}') else { return false; };
    if close <= open { return false; }
    let mut inner = s[open + 1..close].trim().to_string();
    // Strip /* ... */ block comment if present.
    if let Some(start) = inner.find("/*") {
        if let Some(end_off) = inner[start..].find("*/") {
            let end = start + end_off + 2;
            inner = format!("{}{}", &inner[..start], &inner[end..]).trim().to_string();
        }
    }
    matches!(inner.as_str(), "" | "0" | "0.0" | "0i32" | "0u32" | "0.0_f32" | "0.0f32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_todo_and_unimpl() {
        let mut r = ScanReport::default();
        let dir = std::env::temp_dir().join("aether_audit_test");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("x.rs");
        std::fs::write(&p, "fn a() { todo!(); }\nfn b() { unimplemented!(); }\n").unwrap();
        scan_file(&dir, &p, &mut r);
        assert!(r.count(FindingKind::Todo) >= 1);
        assert!(r.count(FindingKind::Unimplemented) >= 1);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn ignores_comments() {
        let mut r = ScanReport::default();
        let dir = std::env::temp_dir().join("aether_audit_test2");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("y.rs");
        std::fs::write(&p, "// todo!() this is in a comment\n").unwrap();
        scan_file(&dir, &p, &mut r);
        assert_eq!(r.count(FindingKind::Todo), 0);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn flags_phase_markers_in_comments() {
        let mut r = ScanReport::default();
        let dir = std::env::temp_dir().join("aether_audit_test3");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("z.rs");
        std::fs::write(&p, "// Phase 1 will replace this with cuBLAS\n").unwrap();
        scan_file(&dir, &p, &mut r);
        assert_eq!(r.count(FindingKind::PhaseMarker), 1);
        std::fs::remove_file(&p).ok();
    }
}
