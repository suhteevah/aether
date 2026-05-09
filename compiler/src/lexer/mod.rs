//! Lexer for Aether.
//!
//! Strips 100% of comments at tokenization. `//` line comments and `/* */`
//! block comments are consumed and discarded — no token is emitted, no span
//! is preserved. By the time anything past this module sees the source,
//! comments do not exist.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    // Keywords
    Fn,
    Let,
    Mut,
    Return,
    If,
    Else,
    For,
    While,
    Break,
    Continue,
    As,
    In,
    Pub,
    Module,
    Use,
    Const,
    Struct,
    Impl,
    Trait,
    Async,
    Await,
    MacroRules,
    Unsafe,
    Enum,
    Match,
    SelfLower,
    True,
    False,

    // Punctuation
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Semi,
    Colon,
    ColonColon,
    Arrow,        // ->
    FatArrow,     // =>
    Eq,
    EqEq,
    Bang,
    BangEq,
    Lt,
    Gt,
    LtEq,
    GtEq,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    Caret,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    Dot,
    DotDot,
    Hash,         // #
    Question,

    // Literals / identifiers
    Ident(String),
    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    /// `'a`, `'static`, etc. Carries the lifetime name without the leading `'`.
    Lifetime(String),

    Eof,
}

impl fmt::Display for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tok::Ident(s) => write!(f, "ident({})", s),
            Tok::IntLit(n) => write!(f, "int({})", n),
            Tok::FloatLit(n) => write!(f, "float({})", n),
            Tok::StrLit(s) => write!(f, "str({:?})", s),
            other => write!(f, "{:?}", other),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub line: u32,
    pub col: u32,
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    col: u32,
    pub stripped_comment_bytes: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
            stripped_comment_bytes: 0,
        }
    }

    pub fn tokenize(mut self) -> Result<(Vec<Token>, usize), String> {
        let mut out = Vec::new();
        loop {
            self.skip_ws_and_comments();
            if self.pos >= self.src.len() {
                out.push(Token { tok: Tok::Eof, line: self.line, col: self.col });
                return Ok((out, self.stripped_comment_bytes));
            }
            let line = self.line;
            let col = self.col;
            let tok = self.next_tok()?;
            out.push(Token { tok, line, col });
        }
    }

    fn peek(&self, off: usize) -> Option<u8> {
        self.src.get(self.pos + off).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.src.get(self.pos).copied()?;
        self.pos += 1;
        if b == b'\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(b)
    }

    fn skip_ws_and_comments(&mut self) {
        loop {
            match self.peek(0) {
                Some(b) if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' => {
                    self.bump();
                }
                Some(b'/') if self.peek(1) == Some(b'/') => {
                    let start = self.pos;
                    self.bump(); self.bump(); // consume //
                    while let Some(b) = self.peek(0) {
                        if b == b'\n' { break; }
                        self.bump();
                    }
                    self.stripped_comment_bytes += self.pos - start;
                }
                Some(b'/') if self.peek(1) == Some(b'*') => {
                    let start = self.pos;
                    self.bump(); self.bump(); // consume /*
                    let mut depth = 1usize;
                    while depth > 0 {
                        match (self.peek(0), self.peek(1)) {
                            (Some(b'/'), Some(b'*')) => { self.bump(); self.bump(); depth += 1; }
                            (Some(b'*'), Some(b'/')) => { self.bump(); self.bump(); depth -= 1; }
                            (Some(_), _) => { self.bump(); }
                            (None, _) => break,
                        }
                    }
                    self.stripped_comment_bytes += self.pos - start;
                }
                _ => break,
            }
        }
    }

    fn next_tok(&mut self) -> Result<Tok, String> {
        let b = self.peek(0).ok_or("unexpected EOF")?;

        // Identifier / keyword
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = self.pos;
            while let Some(c) = self.peek(0) {
                if c.is_ascii_alphanumeric() || c == b'_' { self.bump(); } else { break; }
            }
            let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap();
            return Ok(match s {
                "fn" => Tok::Fn,
                "let" => Tok::Let,
                "mut" => Tok::Mut,
                "return" => Tok::Return,
                "if" => Tok::If,
                "else" => Tok::Else,
                "for" => Tok::For,
                "while" => Tok::While,
                "break" => Tok::Break,
                "continue" => Tok::Continue,
                "in" => Tok::In,
                "pub" => Tok::Pub,
                "module" => Tok::Module,
                "use" => Tok::Use,
                "const" => Tok::Const,
                "struct" => Tok::Struct,
                "impl" => Tok::Impl,
                "trait" => Tok::Trait,
                "async" => Tok::Async,
                "await" => Tok::Await,
                "unsafe" => Tok::Unsafe,
                "enum" => Tok::Enum,
                "match" => Tok::Match,
                "as" => Tok::As,
                "self" => Tok::SelfLower,
                "true" => Tok::True,
                "false" => Tok::False,
                _ => Tok::Ident(s.to_string()),
            });
        }

        // Number — decimal, hex (`0x...`), binary (`0b...`), octal (`0o...`).
        if b.is_ascii_digit() {
            // Hex/bin/octal prefix? Bare `0` followed by x/b/o.
            if b == b'0' && matches!(self.peek(1), Some(b'x') | Some(b'b') | Some(b'o')) {
                self.bump(); // 0
                let radix_byte = self.bump().unwrap();
                let radix = match radix_byte { b'x' => 16, b'b' => 2, b'o' => 8, _ => unreachable!() };
                let start = self.pos;
                while let Some(c) = self.peek(0) {
                    let ok = match radix {
                        16 => c.is_ascii_hexdigit(),
                        2  => c == b'0' || c == b'1',
                        8  => (b'0'..=b'7').contains(&c),
                        _  => false,
                    } || c == b'_';
                    if ok { self.bump(); } else { break; }
                }
                let raw: String = std::str::from_utf8(&self.src[start..self.pos])
                    .unwrap()
                    .chars()
                    .filter(|c| *c != '_')
                    .collect();
                let n = i64::from_str_radix(&raw, radix)
                    .map_err(|e| format!("bad int (radix {}): {e}", radix))?;
                return Ok(Tok::IntLit(n));
            }
            let start = self.pos;
            let mut is_float = false;
            while let Some(c) = self.peek(0) {
                if c.is_ascii_digit() { self.bump(); }
                else if c == b'.' && self.peek(1).map_or(false, |d| d.is_ascii_digit()) {
                    is_float = true; self.bump();
                } else if c == b'_' { self.bump(); }
                else { break; }
            }
            let raw: String = std::str::from_utf8(&self.src[start..self.pos])
                .unwrap()
                .chars()
                .filter(|c| *c != '_')
                .collect();
            return Ok(if is_float {
                Tok::FloatLit(raw.parse().map_err(|e| format!("bad float: {e}"))?)
            } else {
                Tok::IntLit(raw.parse().map_err(|e| format!("bad int: {e}"))?)
            });
        }

        // Lifetime / label: `'a`, `'_lt`. Lexer-side we just emit a Lifetime
        // token; the parser silently consumes them after `&` or `&mut` in
        // type positions (P12.2).
        if b == b'\'' && self.peek(1).map_or(false, |c| c.is_ascii_alphabetic() || c == b'_') {
            self.bump(); // '
            let start = self.pos;
            while let Some(c) = self.peek(0) {
                if c.is_ascii_alphanumeric() || c == b'_' { self.bump(); } else { break; }
            }
            let s = std::str::from_utf8(&self.src[start..self.pos]).unwrap().to_string();
            return Ok(Tok::Lifetime(s));
        }

        // String literal
        if b == b'"' {
            self.bump();
            let mut s = String::new();
            loop {
                match self.bump() {
                    Some(b'"') => return Ok(Tok::StrLit(s)),
                    Some(b'\\') => match self.bump() {
                        Some(b'n') => s.push('\n'),
                        Some(b't') => s.push('\t'),
                        Some(b'r') => s.push('\r'),
                        Some(b'\\') => s.push('\\'),
                        Some(b'"') => s.push('"'),
                        Some(b'0') => s.push('\0'),
                        Some(c) => return Err(format!("bad escape \\{}", c as char)),
                        None => return Err("unterminated string".into()),
                    },
                    Some(c) => s.push(c as char),
                    None => return Err("unterminated string".into()),
                }
            }
        }

        // Punctuation
        self.bump();
        let two = |a: u8, b: u8, this: &mut Lexer| -> bool {
            if this.peek(0) == Some(b) {
                this.bump();
                let _ = a;
                true
            } else { false }
        };

        let tok = match b {
            b'{' => Tok::LBrace,
            b'}' => Tok::RBrace,
            b'(' => Tok::LParen,
            b')' => Tok::RParen,
            b'[' => Tok::LBracket,
            b']' => Tok::RBracket,
            b',' => Tok::Comma,
            b';' => Tok::Semi,
            b':' => if two(b':', b':', self) { Tok::ColonColon } else { Tok::Colon },
            b'-' => if two(b'-', b'>', self) { Tok::Arrow }
                    else if two(b'-', b'=', self) { Tok::MinusEq }
                    else { Tok::Minus },
            b'=' => if two(b'=', b'>', self) { Tok::FatArrow }
                    else if two(b'=', b'=', self) { Tok::EqEq }
                    else { Tok::Eq },
            b'!' => if two(b'!', b'=', self) { Tok::BangEq } else { Tok::Bang },
            b'<' => if two(b'<', b'=', self) { Tok::LtEq } else { Tok::Lt },
            b'>' => if two(b'>', b'=', self) { Tok::GtEq } else { Tok::Gt },
            b'+' => if two(b'+', b'=', self) { Tok::PlusEq } else { Tok::Plus },
            b'*' => if two(b'*', b'=', self) { Tok::StarEq } else { Tok::Star },
            b'/' => if two(b'/', b'=', self) { Tok::SlashEq } else { Tok::Slash }, // comments handled above
            b'%' => Tok::Percent,
            b'&' => if two(b'&', b'&', self) { Tok::AmpAmp } else { Tok::Amp },
            b'|' => if two(b'|', b'|', self) { Tok::PipePipe } else { Tok::Pipe },
            b'^' => Tok::Caret,
            b'.' => if two(b'.', b'.', self) { Tok::DotDot } else { Tok::Dot },
            b'#' => Tok::Hash,
            b'?' => Tok::Question,
            other => return Err(format!("unexpected byte {:?}", other as char)),
        };
        Ok(tok)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_are_stripped() {
        let src = "// gone\nfn main() {} /* also gone */ ";
        let (toks, stripped) = Lexer::new(src).tokenize().unwrap();
        assert!(stripped > 0, "lexer must strip comment bytes");
        for t in &toks {
            match &t.tok {
                Tok::StrLit(s) => assert!(!s.contains("gone")),
                Tok::Ident(s) => assert!(!s.contains("gone")),
                _ => {}
            }
        }
    }

    #[test]
    fn nested_block_comments() {
        let src = "/* a /* b */ c */ fn";
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        assert!(matches!(toks[0].tok, Tok::Fn));
    }

    #[test]
    fn attribute_tokens() {
        let src = "#[autodiff]";
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        assert!(matches!(toks[0].tok, Tok::Hash));
        assert!(matches!(toks[1].tok, Tok::LBracket));
    }
}
