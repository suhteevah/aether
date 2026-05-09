//! Drive `mir::lto::LtoGraph` over the program at `--lto`. Builds a
//! single-crate unit from the AST: each fn is a node, callees come from
//! `Expr::Call(Ident(...))` walks. Exported = `pub` or `extern` or named
//! `main`. Returns the count of reachable + dead fns. The asm backend
//! still emits every fn — the integration step is reachability *known*,
//! not yet acted on. Wiring drop-on-emit is straightforward once this
//! count is trusted.

use crate::ast::{Expr, Item, Program, Stmt, Block};
use super::lto::{CrateUnit, FnSummary, LtoGraph};

pub fn drive(prog: &Program, crate_name: &str) -> (usize, usize) {
    let (live, dead, _) = drive_with_live(prog, crate_name);
    (live, dead)
}

/// Same as `drive` but also returns the set of live unqualified fn names so
/// the caller can filter `prog.items` before codegen (P15.9).
pub fn drive_with_live(prog: &Program, crate_name: &str)
    -> (usize, usize, std::collections::HashSet<String>)
{
    let mut fns = Vec::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            let mut callees = Vec::new();
            if let Some(body) = &f.body { collect_callees(body, &mut callees); }
            let exported = f.is_pub || f.is_extern || f.name == "main";
            let callees: Vec<String> = callees.into_iter()
                .map(|c| if c.contains("::") { c } else { format!("{}::{}", crate_name, c) })
                .collect();
            fns.push(FnSummary { name: f.name.clone(), callees, exported });
        }
    }
    let total = fns.len();
    let mut g = LtoGraph::default();
    g.add(CrateUnit { name: crate_name.into(), fns });
    let reachable = g.reachable();
    let dead = total.saturating_sub(reachable.len());
    let prefix = format!("{}::", crate_name);
    let live_unqualified: std::collections::HashSet<String> = reachable.iter()
        .filter_map(|fqn| fqn.strip_prefix(&prefix).map(String::from))
        .collect();
    (reachable.len(), dead, live_unqualified)
}

fn collect_callees(b: &Block, out: &mut Vec<String>) {
    for s in &b.stmts { collect_in_stmt(s, out); }
    if let Some(t) = &b.tail { collect_in_expr(t, out); }
}
fn collect_in_stmt(s: &Stmt, out: &mut Vec<String>) {
    match s {
        Stmt::Let { value: Some(v), .. } => collect_in_expr(v, out),
        Stmt::Let { .. } => {}
        Stmt::LetTuple { value, .. } => collect_in_expr(value, out),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => collect_in_expr(e, out),
        Stmt::Return(None) => {}
    }
}
fn collect_in_expr(e: &Expr, out: &mut Vec<String>) {
    match e {
        Expr::Call { callee, args } => {
            if let Expr::Ident(n) = callee.as_ref() { out.push(n.clone()); }
            for a in args { collect_in_expr(a, out); }
        }
        Expr::MethodCall { recv, args, .. } => { collect_in_expr(recv, out); for a in args { collect_in_expr(a, out); } }
        Expr::Bin { lhs, rhs, .. } => { collect_in_expr(lhs, out); collect_in_expr(rhs, out); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => collect_in_expr(expr, out),
        Expr::If { cond, then, else_ } => {
            collect_in_expr(cond, out);
            collect_callees(then, out);
            if let Some(e) = else_ { collect_callees(e, out); }
        }
        Expr::While { cond, body } | Expr::For { iter: cond, body, .. } => {
            collect_in_expr(cond, out);
            collect_callees(body, out);
        }
        Expr::Block(b) => collect_callees(b, out),
        Expr::Range { lo, hi, step } => {
            collect_in_expr(lo, out); collect_in_expr(hi, out);
            if let Some(s) = step { collect_in_expr(s, out); }
        }
        Expr::Index { recv, idx } => { collect_in_expr(recv, out); collect_in_expr(idx, out); }
        Expr::Field { recv, .. } => collect_in_expr(recv, out),
        Expr::Tuple(es) => for e in es { collect_in_expr(e, out); },
        Expr::StructLit { fields, .. } => for (_, e) in fields { collect_in_expr(e, out); },
        Expr::Match { scrutinee, arms } => { collect_in_expr(scrutinee, out); for (_, e) in arms { collect_in_expr(e, out); } }
        _ => {}
    }
}

