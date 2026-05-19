//! Drive the `mir::ssa` + `mir::opt` pipeline over each fn body.
//!
//! Phase 15.1 (FR-15.1) — the SSA scaffold (`mir::ssa::rename_block`) and the
//! opt passes (`mir::opt::const_fold` / `strength_reduce` / `cse` / `dce`)
//! exist independently of the asm backend. This module is the bridge: it
//! linearises the *leading run of pure arithmetic let-bindings* of each fn
//! body into the SSA shape, runs the opt pipeline, and materialises the
//! optimised stmt list back into the AST in-place. The asm backend then
//! emits asm from the rewritten AST.
//!
//! Scope is intentionally narrow: only let-statements binding an `Ident` or
//! integer literal or `IntLit (op) IntLit` / `Ident (op) IntLit` /
//! `Ident (op) Ident` expression for `op in {Add, Sub, Mul, Shl}`, plus an
//! optional tail of the same shape. Anything outside that subset terminates
//! the linearisation; later stmts pass through unchanged. This keeps the
//! transform value-preserving on Aether's full surface — calls, ifs, loops,
//! field access, method calls etc. are simply left alone — while still
//! visibly driving the SSA pipeline on the simple-arithmetic prefix that
//! shows up at the top of essentially every fn body.
//!
//! The opt pipeline runs in this order:
//!
//!   1. `const_fold`        — `IntLit op IntLit` collapses to a const.
//!   2. `strength_reduce`   — `Ident * pow2` rewrites to `Ident shl log2`.
//!   3. `cse`               — duplicate `(op, [rhs...])` tuples collapse;
//!                            later uses redirect to the earlier defn.
//!   4. `dce`               — defs whose lhs is never read drop out.
//!
//! `--O0` byte-compat: this driver only runs when `opt_level >= 1`. At
//! `--O0` it is never invoked, so the AST handed to the asm backend is
//! byte-identical to today's behaviour.

use crate::ast::{BinOp, Block, Expr, FnDecl, Item, Program, Stmt};
use super::ssa::{rename_block, SsaStmt};
use super::opt::{const_fold, strength_reduce, cse, dce};

#[derive(Debug, Default, Clone, Copy)]
pub struct Report {
    pub fns_processed: usize,
    pub stmts_in: usize,
    pub stmts_out: usize,
}

/// Run the SSA-driven pipeline on every fn body. Mutates `prog` in place.
pub fn drive(prog: &mut Program) -> Report {
    let mut r = Report::default();
    for it in &mut prog.items {
        if let Item::Fn(f) = it {
            drive_fn(f, &mut r);
        }
    }
    r
}

fn drive_fn(f: &mut FnDecl, r: &mut Report) {
    let Some(body) = f.body.as_mut() else { return; };
    let (linearised, prefix_len, tail_kind) = linearise(body);
    if linearised.is_empty() { return; }
    // FR-15.1 safety: if the linearised prefix is followed by ANY non-
    // linearisable statement, that suffix may reference one of the prefix's
    // lhs names. Our SSA opt pipeline (specifically DCE + CSE) only sees
    // the linearised set, so it would happily drop or rename a let that the
    // suffix still depends on. The honest fix is to keep the linearisation
    // restricted to bodies whose ONLY content is the let-prefix + (optional)
    // absorbed tail. Wider applicability requires a true SSA-aware
    // suffix-analysis — deferred to a future iteration.
    if prefix_len < body.stmts.len() { return; }

    // Build the (lhs, op, rhs) triples that `ssa::rename_block` expects.
    let triples: Vec<(String, String, Vec<String>)> = linearised
        .iter().map(|l| (l.lhs.clone(), l.op.clone(), l.rhs.clone()))
        .collect();
    let renamed = rename_block(&triples);

    // Run the opt pipeline. const_fold (cheap, no rename) → strength_reduce
    // (Mul pow2 → Shl) → cse (collapse duplicates; aliases applied to later
    // rhs uses) → dce (drop unused lhs except the synthetic tail marker).
    let folded   = const_fold(renamed);
    let strength = strength_reduce(folded);
    let csed     = cse(strength);
    let pruned   = dce_preserve_tail(csed, &tail_kind);

    // Materialise back. The renamed lhs (`x_1`) → original name (`x`) by
    // stripping the trailing `_<digits>` suffix. Tail (synthetic) is emitted
    // as the block's `tail` expression. Stmts that were dropped by CSE/DCE
    // simply do not produce an output let.
    let mut new_stmts: Vec<Stmt> = materialise(&pruned, &tail_kind);

    // Splice into the body. We replace the consumed prefix + tail with the
    // materialised output; statements after the linearised prefix stay put.
    let suffix: Vec<Stmt> = body.stmts.drain(prefix_len..).collect();
    body.stmts.clear();
    let new_tail = match &tail_kind {
        TailKind::TakeFromMaterialised => {
            // The materialiser puts the tail expression as the last entry
            // tagged via a sentinel name in the LinearStmt; here we lift the
            // final `Stmt::Expr` out as the block's tail.
            if let Some(Stmt::Expr(e)) = new_stmts.last().cloned() {
                new_stmts.pop();
                Some(Box::new(e))
            } else { None }
        }
        TailKind::None => body.tail.take(),
    };
    body.stmts.extend(new_stmts);
    body.stmts.extend(suffix);
    if matches!(tail_kind, TailKind::TakeFromMaterialised) {
        body.tail = new_tail;
    } else if body.tail.is_none() {
        body.tail = new_tail;
    }

    r.fns_processed += 1;
    r.stmts_in  += linearised.len();
    r.stmts_out += pruned.len();
}

#[derive(Debug, Clone)]
struct LinearStmt { lhs: String, op: String, rhs: Vec<String> }

#[derive(Debug, Clone)]
enum TailKind {
    /// No tail in the original Block; nothing to put back.
    None,
    /// Original block had a linearisable tail; the materialiser should pop
    /// the final stmt and assign it to `body.tail`.
    TakeFromMaterialised,
}

const TAIL_SENTINEL: &str = "__ssa_tail__";

/// Walk the body's leading run of pure let-stmts + optional pure tail. Stops
/// at the first non-linearisable stmt. Returns:
///   - the linearised stmts (one per `let`, plus optional sentinel for tail)
///   - the number of *original* Stmt entries consumed from `body.stmts`
///   - tail kind (whether to put the materialised tail back as body.tail)
fn linearise(body: &Block) -> (Vec<LinearStmt>, usize, TailKind) {
    let mut out: Vec<LinearStmt> = Vec::new();
    let mut idx = 0usize;
    let mut seen_names: std::collections::HashSet<String> = Default::default();
    for s in &body.stmts {
        // Stop the linearised run at any non-trivial statement.
        let Stmt::Let { name, value: Some(value), mutable: false, ty: None } = s else { break; };
        // Bail out of the run if the same name shadows itself — SSA renaming
        // handles this, but cse/dce hooked into our materialiser would lose
        // the original-name mapping. Cheap to forbid in v1.
        if !seen_names.insert(name.clone()) { break; }
        let Some((op, rhs)) = expr_to_op_rhs(value) else { break; };
        out.push(LinearStmt { lhs: name.clone(), op, rhs });
        idx += 1;
    }
    // Try to absorb the tail expression too, so cse/dce see the full live
    // range of names. Synthetic lhs = TAIL_SENTINEL.
    let tail_kind = if let Some(t) = body.tail.as_ref() {
        if !out.is_empty() {
            if let Some((op, rhs)) = expr_to_op_rhs(t) {
                out.push(LinearStmt { lhs: TAIL_SENTINEL.to_string(), op, rhs });
                TailKind::TakeFromMaterialised
            } else { TailKind::None }
        } else { TailKind::None }
    } else { TailKind::None };
    (out, idx, tail_kind)
}

/// Map an `Expr` to the (op, rhs) shape SSA expects, but only for the
/// linearisable subset: a leaf (`Ident` / `IntLit`) or a flat binary op
/// over leaves with op in {Add, Sub, Mul, Shl}. Anything else is None,
/// terminating the linearised run.
fn expr_to_op_rhs(e: &Expr) -> Option<(String, Vec<String>)> {
    match e {
        Expr::IntLit(n) => Some(("const".into(), vec![n.to_string()])),
        Expr::Ident(s)  => Some(("copy".into(),  vec![s.clone()])),
        Expr::Bin { op, lhs, rhs } => {
            let op_str = match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                BinOp::Shl => "shl",
                _ => return None,
            };
            let l = leaf_to_str(lhs)?;
            let r = leaf_to_str(rhs)?;
            Some((op_str.into(), vec![l, r]))
        }
        _ => None,
    }
}

fn leaf_to_str(e: &Expr) -> Option<String> {
    match e {
        Expr::IntLit(n) => Some(n.to_string()),
        Expr::Ident(s)  => Some(s.clone()),
        _ => None,
    }
}

/// Like `opt::dce` but keeps the synthetic tail stmt live unconditionally.
/// The base `dce` would drop the tail because no later stmt references its
/// lhs (the tail IS the return value, not consumed by any successor).
fn dce_preserve_tail(stmts: Vec<SsaStmt>, tail_kind: &TailKind) -> Vec<SsaStmt> {
    let preserve_tail = matches!(tail_kind, TailKind::TakeFromMaterialised);
    let mut used = std::collections::HashSet::new();
    for s in &stmts { for r in &s.rhs { used.insert(r.clone()); } }
    stmts.into_iter().filter(|s| {
        if preserve_tail && s.lhs.starts_with(TAIL_SENTINEL) { return true; }
        used.contains(&s.lhs)
    }).collect()
}

/// Strip the `_<digits>` SSA-rename suffix from a renamed lhs. `x_1` → `x`,
/// `__ssa_tail___1` → `__ssa_tail__`. Names that don't carry the suffix
/// (e.g. parameters referenced as rhs) pass through unchanged.
fn strip_ssa_suffix(name: &str) -> &str {
    if let Some(p) = name.rfind('_') {
        if name[p+1..].chars().all(|c| c.is_ascii_digit()) && p > 0 {
            return &name[..p];
        }
    }
    name
}

/// Build an Expr from an SsaStmt's rhs operand string. A leaf that parses as
/// i64 becomes `IntLit`; otherwise it's an `Ident` referring to a previously
/// emitted let (whose original name we recover via `strip_ssa_suffix`).
fn rhs_to_expr(s: &str) -> Expr {
    if let Ok(n) = s.parse::<i64>() {
        Expr::IntLit(n)
    } else {
        Expr::Ident(strip_ssa_suffix(s).to_string())
    }
}

/// Materialise the optimised SsaStmt list back into AST `Stmt`s. The
/// synthetic tail (TAIL_SENTINEL) is emitted as a `Stmt::Expr` and lifted
/// to `body.tail` by the caller.
fn materialise(stmts: &[SsaStmt], _tail_kind: &TailKind) -> Vec<Stmt> {
    let mut out = Vec::with_capacity(stmts.len());
    for s in stmts {
        let expr = build_expr(&s.op, &s.rhs);
        let is_tail = strip_ssa_suffix(&s.lhs) == TAIL_SENTINEL;
        if is_tail {
            out.push(Stmt::Expr(expr));
        } else {
            let orig = strip_ssa_suffix(&s.lhs).to_string();
            out.push(Stmt::Let { name: orig, mutable: false, ty: None, value: Some(expr) });
        }
    }
    out
}

fn build_expr(op: &str, rhs: &[String]) -> Expr {
    match op {
        "const" | "copy" => rhs_to_expr(&rhs[0]),
        "add" | "sub" | "mul" | "shl" => {
            let bin = match op {
                "add" => BinOp::Add,
                "sub" => BinOp::Sub,
                "mul" => BinOp::Mul,
                "shl" => BinOp::Shl,
                _ => unreachable!(),
            };
            Expr::Bin {
                op: bin,
                lhs: Box::new(rhs_to_expr(&rhs[0])),
                rhs: Box::new(rhs_to_expr(&rhs[1])),
            }
        }
        // Any op we didn't introduce in linearise() should be unreachable
        // here; if it shows up, fall through as a copy of the first operand
        // so we never panic in codegen.
        _ => rhs_to_expr(&rhs[0]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(src: &str) -> Program {
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        Parser::new(toks).parse_program().unwrap()
    }

    fn fn_body(prog: &Program, name: &str) -> Block {
        for it in &prog.items {
            if let Item::Fn(f) = it {
                if f.name == name { return f.body.clone().unwrap(); }
            }
        }
        panic!("no fn {}", name);
    }

    #[test]
    fn const_fold_visible_in_materialised_ast() {
        let mut prog = parse("fn f() -> i64 { let a = 1 + 1; a }");
        let r = drive(&mut prog);
        assert_eq!(r.fns_processed, 1);
        let body = fn_body(&prog, "f");
        // a was 1+1 → after fold, the materialised let binds an IntLit(2).
        if let Stmt::Let { value: Some(Expr::IntLit(2)), name, .. } = &body.stmts[0] {
            assert_eq!(name, "a");
        } else { panic!("expected let a = IntLit(2); got {:?}", body.stmts[0]); }
    }

    #[test]
    fn strength_reduce_mul_pow2_to_shl() {
        let mut prog = parse("fn f(x: i64) -> i64 { let y = x * 8; y }");
        drive(&mut prog);
        let body = fn_body(&prog, "f");
        // y was `x * 8` → strength_reduce rewrites to `x << 3`.
        if let Stmt::Let { value: Some(Expr::Bin { op: BinOp::Shl, lhs, rhs }), .. } = &body.stmts[0] {
            assert!(matches!(lhs.as_ref(), Expr::Ident(s) if s == "x"));
            assert!(matches!(rhs.as_ref(), Expr::IntLit(3)));
        } else { panic!("expected let y = x << 3; got {:?}", body.stmts[0]); }
    }

    #[test]
    fn cse_collapses_duplicate_compute_and_dce_drops_unused() {
        let src = r#"
            fn f(x: i64) -> i64 {
                let b = x * 8;
                let c = x * 8;
                let _u = 99;
                let e = c + 2;
                e + 24
            }
        "#;
        let mut prog = parse(src);
        let r = drive(&mut prog);
        assert_eq!(r.fns_processed, 1);
        let body = fn_body(&prog, "f");
        // After cse: `c = x*8` aliases to `b`; `e` references `b` not `c`.
        // After dce: `c` and `_u` lets are dropped.
        // Surviving lets: b, e. Tail: e + 24.
        let let_names: Vec<String> = body.stmts.iter().filter_map(|s| match s {
            Stmt::Let { name, .. } => Some(name.clone()),
            _ => None,
        }).collect();
        assert_eq!(let_names, vec!["b".to_string(), "e".to_string()],
                   "expected lets b + e after cse/dce; got {:?}", let_names);
        // Tail is `e + 24`.
        if let Some(t) = &body.tail {
            if let Expr::Bin { op: BinOp::Add, lhs, rhs } = t.as_ref() {
                assert!(matches!(lhs.as_ref(), Expr::Ident(s) if s == "e"));
                assert!(matches!(rhs.as_ref(), Expr::IntLit(24)));
            } else { panic!("tail not add; got {:?}", t); }
        } else { panic!("tail missing"); }
    }
}
