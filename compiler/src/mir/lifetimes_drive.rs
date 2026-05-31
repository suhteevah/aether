//! Drive `mir::lifetimes::Checker` over each fn body during `--check`.
//! Walks AST stmts, synthesizes `BorrowEvent`s for every `let r = &x;` /
//! `let r = &mut x;` (a *held* borrow that lives to end-of-fn, matching
//! today's lexical-scope simplification) and every plain identifier use,
//! then runs the NLL borrow checker. Returns the coded violations so the
//! driver can surface them as `AE0200`-family diagnostics.
//!
//! Refs in *argument* position (`foo(&mut x)`) do NOT create a held
//! borrow — they're consumed by the call, so back-to-back `foo(&mut x);
//! bar(&mut x);` stays legal. Only a `let`-bound reference is tracked as
//! live, which is the conservative half of the eventual NLL upgrade.

use crate::ast::{Block, Expr, Item, Program, Stmt};
use super::lifetimes::{BorrowEvent, BorrowKind, Checker, Violation};

pub fn drive(prog: &Program) -> Vec<Violation> {
    let mut all = Vec::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            if let Some(body) = &f.body {
                let mut events = Vec::new();
                let mut next_id = 0u32;
                walk_block(body, &mut events, &mut next_id);
                all.extend(Checker::run_coded(&events));
            }
        }
    }
    all
}

fn walk_block(b: &Block, ev: &mut Vec<BorrowEvent>, id: &mut u32) {
    for s in &b.stmts { walk_stmt(s, ev, id); }
    if let Some(t) = &b.tail { walk_expr(t, ev, id); }
}

fn walk_stmt(s: &Stmt, ev: &mut Vec<BorrowEvent>, id: &mut u32) {
    match s {
        Stmt::Let { name, value, .. } => {
            if let Some(v) = value {
                match v {
                    Expr::Ref { mutable, expr } => {
                        if let Expr::Ident(place) = expr.as_ref() {
                            let kind = if *mutable { BorrowKind::Mut } else { BorrowKind::Shared };
                            ev.push(BorrowEvent::Borrow { place: place.clone(), kind, id: *id });
                            *id += 1;
                            // We deliberately don't EndBorrow here — the
                            // ref `name` is alive for the rest of the fn,
                            // matching today's lexical-scope simplification.
                            let _ = name;
                        } else {
                            walk_expr(v, ev, id);
                        }
                    }
                    _ => walk_expr(v, ev, id),
                }
            }
        }
        Stmt::LetTuple { value, .. } => walk_expr(value, ev, id),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => walk_expr(e, ev, id),
        Stmt::Return(None) => {}
    }
}

fn walk_expr(e: &Expr, ev: &mut Vec<BorrowEvent>, id: &mut u32) {
    match e {
        Expr::Ident(n) => ev.push(BorrowEvent::Use { place: n.clone() }),
        Expr::Bin { lhs, rhs, .. } => { walk_expr(lhs, ev, id); walk_expr(rhs, ev, id); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => walk_expr(expr, ev, id),
        // A ref in expression/argument position is consumed in place — recurse
        // for nested uses but do NOT register a held borrow (that's the
        // `let`-binding case in walk_stmt).
        Expr::Ref { expr, .. } => walk_expr(expr, ev, id),
        Expr::Call { args, .. } => for a in args { walk_expr(a, ev, id); },
        Expr::MethodCall { recv, args, .. } => { walk_expr(recv, ev, id); for a in args { walk_expr(a, ev, id); } }
        Expr::Field { recv, .. } => walk_expr(recv, ev, id),
        Expr::Index { recv, idx } => { walk_expr(recv, ev, id); walk_expr(idx, ev, id); }
        // Recurse into nested control-flow blocks so a held borrow inside an
        // `if`/`for`/`while`/region body is tracked too.
        Expr::Block(b) => walk_block(b, ev, id),
        Expr::If { cond, then, else_ } => {
            walk_expr(cond, ev, id);
            walk_block(then, ev, id);
            if let Some(b) = else_ { walk_block(b, ev, id); }
        }
        Expr::For { iter, body, .. } => { walk_expr(iter, ev, id); walk_block(body, ev, id); }
        Expr::While { cond, body } => { walk_expr(cond, ev, id); walk_block(body, ev, id); }
        Expr::Region { body, .. } => walk_block(body, ev, id),
        _ => {}
    }
}
