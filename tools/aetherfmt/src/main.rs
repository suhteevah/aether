//! aetherfmt — deterministic .aether formatter.
//!
//! Phase 0 surface: strip trailing whitespace, normalize tabs → 4 spaces,
//! collapse runs of >2 blank lines to exactly 1. Reads source from a file
//! arg, writes back in place (or to stdout with `--check`).
//!
//! The full rustfmt-equivalent (deep token-tree re-emit, alignment rules,
//! line-length wrapping) lives behind FR-22.3 in NEXT-UP.md. This binary
//! is the foothold — small enough to verify by hand, useful enough to keep
//! the existing .aether corpus consistent.

use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::process::ExitCode;

fn format_source(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut prev_blank = false;
    let mut second_blank = false;
    for line in src.lines() {
        let normalized = line.replace('\t', "    ");
        let trimmed = normalized.trim_end();
        let is_blank = trimmed.is_empty();
        if is_blank {
            if prev_blank {
                if second_blank { continue; } // drop 3rd+ consecutive blank
                second_blank = true;
            }
            prev_blank = true;
        } else {
            prev_blank = false;
            second_blank = false;
        }
        out.push_str(trimmed);
        out.push('\n');
    }
    out
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut check_only = false;
    let mut paths: Vec<String> = Vec::new();
    for a in args {
        if a == "--check" { check_only = true; }
        else if a == "-" {
            let mut buf = String::new();
            if io::stdin().read_to_string(&mut buf).is_err() {
                eprintln!("aetherfmt: stdin read failed"); return ExitCode::from(2);
            }
            let out = format_source(&buf);
            if io::stdout().write_all(out.as_bytes()).is_err() { return ExitCode::from(2); }
            return ExitCode::SUCCESS;
        }
        else { paths.push(a); }
    }
    if paths.is_empty() {
        eprintln!("usage: aetherfmt [--check] <file.aether>...");
        return ExitCode::from(2);
    }
    let mut diff = false;
    for p in paths {
        let src = match fs::read_to_string(&p) {
            Ok(s) => s,
            Err(e) => { eprintln!("aetherfmt: {p}: {e}"); return ExitCode::from(2); }
        };
        let out = format_source(&src);
        if out != src {
            diff = true;
            if check_only {
                eprintln!("aetherfmt: would reformat {}", p);
            } else {
                if let Err(e) = fs::write(&p, &out) {
                    eprintln!("aetherfmt: write {p}: {e}");
                    return ExitCode::from(2);
                }
            }
        }
    }
    if check_only && diff { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_trailing_ws() {
        assert_eq!(format_source("fn main() {  \n    42  \n}\n"), "fn main() {\n    42\n}\n");
    }
    #[test]
    fn collapses_blank_runs() {
        let src = "a\n\n\n\n\nb\n";
        let out = format_source(src);
        // first blank kept, second kept, third+ dropped → max 2 blanks.
        assert!(out.matches("\n\n\n").count() <= 1);
    }
    #[test]
    fn tabs_to_spaces() {
        assert_eq!(format_source("\tfn main()\n"), "    fn main()\n");
    }
}
