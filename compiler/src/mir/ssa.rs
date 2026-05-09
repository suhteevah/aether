//! SSA (Static Single Assignment) baby pass.
//!
//! Phase 10.1 — converts a flat sequence of (lhs := op(rhs...)) statements
//! into SSA form by suffixing every variable with a generation counter.
//! The result is a parallel statement list where every lhs is unique;
//! every rhs reference resolves to the most-recent prior generation.
//!
//! This is the smallest non-trivial SSA transform — block-local renaming
//! with no phi-node insertion. Phi insertion across joins is downstream
//! (depends on dominance computation in the asm backend's CFG, which
//! today is implicit in the AST walker). The renaming pass alone is
//! enough to enable: dead-store elim, constant-folding of single-use
//! defs, common-subexpression elimination on the renamed RHS.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SsaStmt {
    /// Renamed lhs, e.g. "x_2".
    pub lhs: String,
    /// Original op string, e.g. "add".
    pub op: String,
    /// Renamed rhs operands, e.g. ["x_1", "5"].
    pub rhs: Vec<String>,
}

/// One linear block of `(lhs, op, rhs...)` triples in original program
/// order. Returns the SSA-renamed equivalent.
pub fn rename_block(block: &[(String, String, Vec<String>)]) -> Vec<SsaStmt> {
    let mut counters: HashMap<String, u32> = HashMap::new();
    let mut current: HashMap<String, String> = HashMap::new();
    let mut out = Vec::with_capacity(block.len());
    for (lhs, op, rhs) in block {
        let renamed_rhs: Vec<String> = rhs
            .iter()
            .map(|r| current.get(r).cloned().unwrap_or_else(|| r.clone()))
            .collect();
        let n = counters.entry(lhs.clone()).or_insert(0);
        *n += 1;
        let new_lhs = format!("{}_{}", lhs, n);
        current.insert(lhs.clone(), new_lhs.clone());
        out.push(SsaStmt { lhs: new_lhs, op: op.clone(), rhs: renamed_rhs });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> String { x.to_string() }

    #[test]
    fn single_assignment_unchanged() {
        let block = vec![
            (s("x"), s("const"), vec![s("5")]),
        ];
        let out = rename_block(&block);
        assert_eq!(out[0].lhs, "x_1");
        assert_eq!(out[0].rhs, vec!["5".to_string()]);
    }

    #[test]
    fn shadowed_var_gets_new_generation() {
        // x = 5; x = x + 1; y = x * 2
        let block = vec![
            (s("x"), s("const"), vec![s("5")]),
            (s("x"), s("add"),   vec![s("x"), s("1")]),
            (s("y"), s("mul"),   vec![s("x"), s("2")]),
        ];
        let out = rename_block(&block);
        assert_eq!(out[0].lhs, "x_1");
        assert_eq!(out[1].lhs, "x_2");
        // The add's first rhs must reference the previous gen, x_1.
        assert_eq!(out[1].rhs[0], "x_1");
        // The mul reads the latest x, x_2.
        assert_eq!(out[2].rhs[0], "x_2");
        assert_eq!(out[2].lhs, "y_1");
    }

    #[test]
    fn unknown_rhs_passes_through() {
        // y = z + 1   (z is a parameter, never assigned in this block)
        let block = vec![
            (s("y"), s("add"), vec![s("z"), s("1")]),
        ];
        let out = rename_block(&block);
        assert_eq!(out[0].rhs[0], "z");
    }
}
