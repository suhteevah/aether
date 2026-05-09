//! `macro_rules!` token-tree expansion (subset).
//!
//! Phase 6.11 — pattern + body model and a tiny matcher/expander. Each
//! macro_rules! invocation has a list of `(pattern, body)` arms; the
//! matcher picks the first pattern that fits the call's token stream
//! and substitutes captured fragments into the body.
//!
//! Captured fragments are typed (`expr`, `ident`, `tt`); this module
//! stays at the token level (Vec<String>) for Phase 0.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum PatternToken {
    Lit(String),
    Capture { name: String, kind: String },
}

#[derive(Debug, Clone)]
pub enum BodyToken {
    Lit(String),
    Insert(String),
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pattern: Vec<PatternToken>,
    pub body: Vec<BodyToken>,
}

#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    pub arms: Vec<Arm>,
}

pub fn expand(def: &MacroDef, call_tokens: &[String]) -> Option<Vec<String>> {
    'outer: for arm in &def.arms {
        let mut bindings: HashMap<String, String> = HashMap::new();
        let mut ci = 0usize;
        for pt in &arm.pattern {
            match pt {
                PatternToken::Lit(s) => {
                    if ci >= call_tokens.len() || &call_tokens[ci] != s { continue 'outer; }
                    ci += 1;
                }
                PatternToken::Capture { name, .. } => {
                    if ci >= call_tokens.len() { continue 'outer; }
                    bindings.insert(name.clone(), call_tokens[ci].clone());
                    ci += 1;
                }
            }
        }
        if ci != call_tokens.len() { continue 'outer; }
        let mut out = Vec::with_capacity(arm.body.len());
        for bt in &arm.body {
            match bt {
                BodyToken::Lit(s) => out.push(s.clone()),
                BodyToken::Insert(name) => {
                    out.push(bindings.get(name).cloned().unwrap_or_default());
                }
            }
        }
        return Some(out);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> String { x.into() }

    #[test]
    fn vec_macro_expansion() {
        // Mock `vec![1, 2, 3]` → `Vec::from_iter([1, 2, 3])`.
        let def = MacroDef {
            name: s("vec"),
            arms: vec![Arm {
                pattern: vec![
                    PatternToken::Lit(s("[")),
                    PatternToken::Capture { name: s("a"), kind: s("expr") },
                    PatternToken::Lit(s(",")),
                    PatternToken::Capture { name: s("b"), kind: s("expr") },
                    PatternToken::Lit(s(",")),
                    PatternToken::Capture { name: s("c"), kind: s("expr") },
                    PatternToken::Lit(s("]")),
                ],
                body: vec![
                    BodyToken::Lit(s("Vec::from_iter([")),
                    BodyToken::Insert(s("a")), BodyToken::Lit(s(",")),
                    BodyToken::Insert(s("b")), BodyToken::Lit(s(",")),
                    BodyToken::Insert(s("c")),
                    BodyToken::Lit(s("])")),
                ],
            }],
        };
        let call = vec![s("["), s("1"), s(","), s("2"), s(","), s("3"), s("]")];
        let out = expand(&def, &call).unwrap();
        assert_eq!(out.join(""), "Vec::from_iter([1,2,3])");
    }

    #[test]
    fn no_matching_arm_returns_none() {
        let def = MacroDef {
            name: s("only_zero"),
            arms: vec![Arm {
                pattern: vec![PatternToken::Lit(s("0"))],
                body: vec![BodyToken::Lit(s("ZERO"))],
            }],
        };
        assert!(expand(&def, &[s("1")]).is_none());
    }

    #[test]
    fn assert_eq_macro_expansion() {
        // assert_eq!(a, b) → if a != b { panic!() }
        let def = MacroDef {
            name: s("assert_eq"),
            arms: vec![Arm {
                pattern: vec![
                    PatternToken::Capture { name: s("l"), kind: s("expr") },
                    PatternToken::Lit(s(",")),
                    PatternToken::Capture { name: s("r"), kind: s("expr") },
                ],
                body: vec![
                    BodyToken::Lit(s("if ")), BodyToken::Insert(s("l")),
                    BodyToken::Lit(s(" != ")), BodyToken::Insert(s("r")),
                    BodyToken::Lit(s(" { panic!() }")),
                ],
            }],
        };
        let call = vec![s("x"), s(","), s("42")];
        let out = expand(&def, &call).unwrap();
        assert_eq!(out.join(""), "if x != 42 { panic!() }");
    }
}
