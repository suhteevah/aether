//! aetherclippy — small starter linter for .aether source.
//!
//! Phase 0 surface: 5 lints, line-grep-based (no AST yet). Real AST-based
//! lints with auto-fix, suppress attributes, and config files are FR-22.4.
//!
//!   1. trailing_ws   — line ends with whitespace.
//!   2. tab_indent    — line starts with a tab.
//!   3. let_underscore — `let _ = expr;` form (use the discard expression).
//!   4. magic_number  — int literal > 4096 not in a `const` decl.
//!   5. todo_marker   — `TODO`/`FIXME`/`XXX` in source comments.

use std::env;
use std::fs;
use std::process::ExitCode;

#[derive(Debug)]
struct Lint {
    file: String,
    line: usize,
    code: &'static str,
    msg: String,
}

fn lint_source(file: &str, src: &str) -> Vec<Lint> {
    let mut out = Vec::new();
    for (i, raw) in src.lines().enumerate() {
        let lineno = i + 1;
        if !raw.is_empty() && raw.ends_with(|c: char| c == ' ' || c == '\t') {
            out.push(Lint { file: file.into(), line: lineno, code: "AC001",
                msg: "trailing whitespace".into() });
        }
        if raw.starts_with('\t') {
            out.push(Lint { file: file.into(), line: lineno, code: "AC002",
                msg: "tab indent (use 4 spaces)".into() });
        }
        let trimmed = raw.trim_start();
        if trimmed.starts_with("let _ =") {
            out.push(Lint { file: file.into(), line: lineno, code: "AC003",
                msg: "`let _ = expr;` discards the value — drop the binding instead".into() });
        }
        // magic_number: look for >4096 outside of `const` declarations.
        if !trimmed.starts_with("const ") {
            let mut chars = raw.chars().peekable();
            while let Some(c) = chars.next() {
                if c.is_ascii_digit() {
                    let mut s = String::from(c);
                    while let Some(&cc) = chars.peek() {
                        if cc.is_ascii_digit() || cc == '_' { s.push(cc); chars.next(); }
                        else { break; }
                    }
                    let v: String = s.chars().filter(|c| *c != '_').collect();
                    if let Ok(n) = v.parse::<u64>() {
                        if n > 4096 && n < 1_000_000_000_000 {
                            out.push(Lint { file: file.into(), line: lineno, code: "AC004",
                                msg: format!("magic literal {n} — bind it to a `const`") });
                            break; // one report per line
                        }
                    }
                }
            }
        }
        for marker in ["TODO", "FIXME", "XXX"] {
            if raw.contains(marker) {
                out.push(Lint { file: file.into(), line: lineno, code: "AC005",
                    msg: format!("{marker} marker in source") });
                break;
            }
        }
    }
    out
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: aetherclippy <file.aether>...");
        return ExitCode::from(2);
    }
    let mut total = 0usize;
    for p in &args {
        let src = match fs::read_to_string(p) {
            Ok(s) => s,
            Err(e) => { eprintln!("aetherclippy: {p}: {e}"); return ExitCode::from(2); }
        };
        for l in lint_source(p, &src) {
            println!("{}:{}: [{}] {}", l.file, l.line, l.code, l.msg);
            total += 1;
        }
    }
    if total > 0 {
        eprintln!("aetherclippy: {} finding(s)", total);
        ExitCode::from(1)
    } else { ExitCode::SUCCESS }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flags_trailing_ws() {
        let l = lint_source("t", "fn main() {  \n");
        assert!(l.iter().any(|x| x.code == "AC001"));
    }
    #[test]
    fn flags_tab_indent() {
        let l = lint_source("t", "\tfn main()\n");
        assert!(l.iter().any(|x| x.code == "AC002"));
    }
    #[test]
    fn flags_let_underscore() {
        let l = lint_source("t", "let _ = foo();\n");
        assert!(l.iter().any(|x| x.code == "AC003"));
    }
    #[test]
    fn flags_magic_number() {
        let l = lint_source("t", "let x: i64 = 99999;\n");
        assert!(l.iter().any(|x| x.code == "AC004"));
    }
    #[test]
    fn flags_todo() {
        let l = lint_source("t", "// TODO: fix this\n");
        assert!(l.iter().any(|x| x.code == "AC005"));
    }
    #[test]
    fn skips_const_decls() {
        let l = lint_source("t", "const N: i64 = 99999;\n");
        assert!(!l.iter().any(|x| x.code == "AC004"));
    }
}
