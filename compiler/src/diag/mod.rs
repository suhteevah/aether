//! Diagnostics — structured, code-tagged, JSON-serialisable.
//!
//! Why it matters: this language is meant to be written by LLMs, and an LLM
//! that gets `AE0042: expected ',' or ')' at 14:7 — try inserting ','` can
//! fix the file in one shot. A bare `parse error` cannot. Every diagnostic
//! carries a stable code, a span, a message, and an optional fix hint. The
//! whole set is serialisable as JSON Lines via `--json-errors`.

use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity { Error, Warning, Note }

impl Severity {
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

impl Span {
    pub fn at(line: u32, col: u32) -> Self { Self { line, col } }
}

#[derive(Debug, Clone)]
pub struct Diag {
    pub code: &'static str,
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
    pub hint: Option<String>,
    pub stage: &'static str,
}

impl Diag {
    pub fn error(code: &'static str, stage: &'static str, message: impl Into<String>) -> Self {
        Diag { code, severity: Severity::Error, message: message.into(), span: None, hint: None, stage }
    }
    pub fn at(mut self, line: u32, col: u32) -> Self { self.span = Some(Span::at(line, col)); self }
    pub fn with_hint(mut self, h: impl Into<String>) -> Self { self.hint = Some(h.into()); self }

    pub fn render_human(&self, file: &str) -> String {
        let mut s = String::new();
        let loc = match self.span {
            Some(sp) => format!("{}:{}:{}", file, sp.line, sp.col),
            None => file.to_string(),
        };
        let _ = write!(s, "{} [{}/{}]: {} ({})",
            self.severity.as_str(), self.code, self.stage, self.message, loc);
        if let Some(h) = &self.hint {
            let _ = write!(s, "\n  hint: {}", h);
        }
        s
    }

    pub fn render_json(&self, file: &str) -> String {
        let mut s = String::from("{");
        let _ = write!(s, "\"code\":\"{}\"", self.code);
        let _ = write!(s, ",\"severity\":\"{}\"", self.severity.as_str());
        let _ = write!(s, ",\"stage\":\"{}\"", self.stage);
        let _ = write!(s, ",\"file\":{}", json_str(file));
        if let Some(sp) = self.span {
            let _ = write!(s, ",\"line\":{},\"col\":{}", sp.line, sp.col);
        }
        let _ = write!(s, ",\"message\":{}", json_str(&self.message));
        if let Some(h) = &self.hint {
            let _ = write!(s, ",\"hint\":{}", json_str(h));
        }
        s.push('}');
        s
    }
}

fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => { let _ = write!(out, "\\u{:04x}", c as u32); }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parse a legacy `"line:col: msg"` string into a Diag. Used as a bridge
/// while the lexer/parser still return Strings — the diagnostics surface
/// stays uniform from the driver's perspective.
pub fn from_legacy(code: &'static str, stage: &'static str, raw: &str) -> Diag {
    if let Some((loc, msg)) = raw.split_once(": ") {
        let mut parts = loc.splitn(2, ':');
        if let (Some(l), Some(c)) = (parts.next(), parts.next()) {
            if let (Ok(line), Ok(col)) = (l.parse::<u32>(), c.parse::<u32>()) {
                return Diag::error(code, stage, msg).at(line, col);
            }
        }
    }
    Diag::error(code, stage, raw)
}

#[derive(Default)]
pub struct DiagSink {
    pub diags: Vec<Diag>,
}

impl DiagSink {
    pub fn push(&mut self, d: Diag) { self.diags.push(d); }
    pub fn has_errors(&self) -> bool {
        self.diags.iter().any(|d| d.severity == Severity::Error)
    }
    pub fn render_human(&self, file: &str) -> String {
        self.diags.iter().map(|d| d.render_human(file)).collect::<Vec<_>>().join("\n")
    }
    pub fn render_json(&self, file: &str) -> String {
        self.diags.iter().map(|d| d.render_json(file)).collect::<Vec<_>>().join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_shape() {
        let d = Diag::error("AE0001", "parse", "expected `,`")
            .at(3, 14)
            .with_hint("insert `,` after the previous argument");
        let j = d.render_json("foo.aether");
        assert!(j.contains("\"code\":\"AE0001\""));
        assert!(j.contains("\"line\":3"));
        assert!(j.contains("\"col\":14"));
        assert!(j.contains("\"hint\":"));
    }

    #[test]
    fn json_escapes_quotes_and_newlines() {
        let d = Diag::error("AE0002", "lex", "got \"x\nbad\"");
        let j = d.render_json("a.aether");
        assert!(j.contains("\\\""));
        assert!(j.contains("\\n"));
    }
}
