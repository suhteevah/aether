//! Optimization passes operating over the SSA-renamed statement list.
//!
//! Phase 10.2 — small but real implementations of the passes most likely
//! to deliver a measurable win on Aether's existing kernels:
//!
//!   * Constant folding   — eval pure ops over literal operands
//!   * Strength reduction — `x * 2^n` → `x << n`
//!   * Dead code elim     — drop defs whose lhs is never read
//!   * Common subexpr elim — collapse identical (op, rhs...) tuples
//!
//! Each pass works over the `SsaStmt` shape from `mir::ssa`. Operands are
//! string-typed for Phase 0 simplicity; the underlying transform is the
//! same once the IR migrates to typed values.

use super::ssa::SsaStmt;
use std::collections::HashMap;

fn parse_int(s: &str) -> Option<i64> { s.parse::<i64>().ok() }

/// Constant folding for binary integer ops.
pub fn const_fold(stmts: Vec<SsaStmt>) -> Vec<SsaStmt> {
    stmts
        .into_iter()
        .map(|s| {
            if s.rhs.len() == 2 {
                if let (Some(a), Some(b)) = (parse_int(&s.rhs[0]), parse_int(&s.rhs[1])) {
                    let folded = match s.op.as_str() {
                        "add" => Some(a + b),
                        "sub" => Some(a - b),
                        "mul" => Some(a.wrapping_mul(b)),
                        "shl" => Some(a << (b & 63)),
                        _ => None,
                    };
                    if let Some(v) = folded {
                        return SsaStmt {
                            lhs: s.lhs,
                            op: "const".to_string(),
                            rhs: vec![v.to_string()],
                        };
                    }
                }
            }
            s
        })
        .collect()
}

/// Strength reduction: `mul x, pow_of_two` → `shl x, log2(pow)`.
pub fn strength_reduce(stmts: Vec<SsaStmt>) -> Vec<SsaStmt> {
    stmts
        .into_iter()
        .map(|s| {
            if s.op == "mul" && s.rhs.len() == 2 {
                if let Some(b) = parse_int(&s.rhs[1]) {
                    if b > 0 && (b & (b - 1)) == 0 {
                        let log = b.trailing_zeros() as i64;
                        return SsaStmt {
                            lhs: s.lhs,
                            op: "shl".to_string(),
                            rhs: vec![s.rhs[0].clone(), log.to_string()],
                        };
                    }
                }
            }
            s
        })
        .collect()
}

/// Dead code elimination: drop `(lhs, op, rhs)` triples whose lhs is
/// never used by any later statement and whose op is pure (no side
/// effects). For Phase 0 every op is pure.
pub fn dce(stmts: Vec<SsaStmt>) -> Vec<SsaStmt> {
    let mut used = std::collections::HashSet::new();
    for s in &stmts { for r in &s.rhs { used.insert(r.clone()); } }
    stmts.into_iter().filter(|s| used.contains(&s.lhs)).collect()
}

/// Common subexpression elimination: if two SSA statements compute the
/// same `(op, rhs...)`, rewrite later uses to point at the first def.
pub fn cse(stmts: Vec<SsaStmt>) -> Vec<SsaStmt> {
    let mut seen: HashMap<(String, Vec<String>), String> = HashMap::new();
    let mut alias: HashMap<String, String> = HashMap::new();
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        let renamed_rhs: Vec<String> = s.rhs.iter()
            .map(|r| alias.get(r).cloned().unwrap_or_else(|| r.clone()))
            .collect();
        let key = (s.op.clone(), renamed_rhs.clone());
        if let Some(prev) = seen.get(&key) {
            alias.insert(s.lhs, prev.clone());
        } else {
            seen.insert(key, s.lhs.clone());
            out.push(SsaStmt { lhs: s.lhs, op: s.op, rhs: renamed_rhs });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(lhs: &str, op: &str, rhs: &[&str]) -> SsaStmt {
        SsaStmt {
            lhs: lhs.into(),
            op: op.into(),
            rhs: rhs.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn fold_add_to_const() {
        let r = const_fold(vec![st("x_1", "add", &["2", "3"])]);
        assert_eq!(r[0].op, "const");
        assert_eq!(r[0].rhs[0], "5");
    }

    #[test]
    fn strength_reduce_mul_8_to_shl_3() {
        let r = strength_reduce(vec![st("y_1", "mul", &["x", "8"])]);
        assert_eq!(r[0].op, "shl");
        assert_eq!(r[0].rhs[1], "3");
    }

    #[test]
    fn dce_drops_unused_def() {
        let in_ = vec![
            st("a_1", "add", &["x", "y"]),
            st("b_1", "add", &["a_1", "1"]),
        ];
        // a_1 is used by b_1, but b_1 is unused. After DCE only a_1 remains.
        let r = dce(in_);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].lhs, "a_1");
    }

    #[test]
    fn cse_collapses_duplicate_compute() {
        // a = x + y; b = x + y; c = b + 1 → after CSE, b is aliased to a.
        let in_ = vec![
            st("a_1", "add", &["x", "y"]),
            st("b_1", "add", &["x", "y"]),
            st("c_1", "add", &["b_1", "1"]),
        ];
        let r = cse(in_);
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].lhs, "a_1");
        assert_eq!(r[1].lhs, "c_1");
        // c_1's first rhs should now reference a_1, not b_1.
        assert_eq!(r[1].rhs[0], "a_1");
    }
}
