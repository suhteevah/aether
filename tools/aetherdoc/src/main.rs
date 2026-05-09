//! aetherdoc — extract `///` doc-comments from .aether sources.
//!
//! Phase 0 surface: parse line-grep style. For each fn/struct/impl/trait
//! declaration, capture the run of `///` lines immediately above it, write
//! a markdown index to stdout.
//!
//! Full HTML output, cross-linking, search index — FR-22.5 in NEXT-UP.md.

use std::env;
use std::fs;
use std::process::ExitCode;

#[derive(Debug)]
struct DocItem {
    file: String,
    line: usize,
    kind: &'static str,    // "fn" / "struct" / "impl" / "trait" / "enum" / "const"
    name: String,
    doc: String,
}

fn extract_docs(file: &str, src: &str) -> Vec<DocItem> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out = Vec::new();
    let mut pending_doc = String::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("/// ") {
            if !pending_doc.is_empty() { pending_doc.push('\n'); }
            pending_doc.push_str(rest);
            continue;
        } else if let Some(rest) = t.strip_prefix("///") {
            if !pending_doc.is_empty() { pending_doc.push('\n'); }
            pending_doc.push_str(rest);
            continue;
        }
        if t.starts_with("//") { continue; }
        if t.is_empty() { pending_doc.clear(); continue; }
        // Item signatures.
        let detect: Option<(&'static str, &str)> = if let Some(r) = t.strip_prefix("pub fn ").or(t.strip_prefix("fn ")) {
            Some(("fn", r))
        } else if let Some(r) = t.strip_prefix("pub struct ").or(t.strip_prefix("struct ")) {
            Some(("struct", r))
        } else if let Some(r) = t.strip_prefix("pub trait ").or(t.strip_prefix("trait ")) {
            Some(("trait", r))
        } else if let Some(r) = t.strip_prefix("pub enum ").or(t.strip_prefix("enum ")) {
            Some(("enum", r))
        } else if let Some(r) = t.strip_prefix("impl ") {
            Some(("impl", r))
        } else if let Some(r) = t.strip_prefix("pub const ").or(t.strip_prefix("const ")) {
            Some(("const", r))
        } else { None };
        if let Some((kind, rest)) = detect {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '<' || *c == ':')
                .collect();
            if !pending_doc.is_empty() {
                out.push(DocItem {
                    file: file.into(), line: i + 1, kind,
                    name: name.trim_end_matches('<').to_string(),
                    doc: std::mem::take(&mut pending_doc),
                });
            } else {
                pending_doc.clear();
            }
        } else {
            // Any other non-comment line breaks the doc-comment streak.
            pending_doc.clear();
        }
    }
    out
}

fn render_markdown(items: &[DocItem]) -> String {
    let mut out = String::from("# aetherdoc — auto-generated API index\n\n");
    let mut by_file: std::collections::BTreeMap<&str, Vec<&DocItem>> =
        std::collections::BTreeMap::new();
    for it in items { by_file.entry(it.file.as_str()).or_default().push(it); }
    for (f, list) in &by_file {
        out.push_str(&format!("## `{}`\n\n", f));
        for it in list {
            out.push_str(&format!("### `{} {}`  _(line {})_\n\n{}\n\n", it.kind, it.name, it.line, it.doc));
        }
    }
    out
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: aetherdoc <file.aether>...");
        return ExitCode::from(2);
    }
    let mut all = Vec::new();
    for p in &args {
        let src = match fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => { eprintln!("aetherdoc: {p}: {e}"); return ExitCode::from(2); }
        };
        all.extend(extract_docs(p, &src));
    }
    print!("{}", render_markdown(&all));
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn extracts_fn_doc() {
        let src = "/// Adds two numbers.\nfn add(a: i64, b: i64) -> i64 { a + b }\n";
        let d = extract_docs("t", src);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].kind, "fn");
        assert_eq!(d[0].name, "add");
        assert!(d[0].doc.contains("Adds two numbers"));
    }
    #[test]
    fn extracts_struct_doc() {
        let src = "/// A 3D vector.\npub struct V3 { x: i64 }\n";
        let d = extract_docs("t", src);
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].kind, "struct");
        assert_eq!(d[0].name, "V3");
    }
    #[test]
    fn no_doc_no_item() {
        let src = "fn no_docs() {}\n";
        assert!(extract_docs("t", src).is_empty());
    }
    #[test]
    fn breaks_on_blank() {
        let src = "/// floating doc\n\nfn unrelated() {}\n";
        // blank line breaks the doc streak — fn is undocumented.
        assert!(extract_docs("t", src).is_empty());
    }
}
