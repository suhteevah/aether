//! Drive `mir::lifetimes::Checker` over each fn body during `--check`.
//! Walks AST stmts, synthesizes `BorrowEvent`s for every `let r = &x;` /
//! `let r = &mut x;` and every plain identifier use, then runs the NLL
//! borrow checker. Reports the count of detected violations.

use crate::ast::{Block, Expr, Item, Program, Stmt};
use super::lifetimes::{BorrowEvent, BorrowKind, Checker};

pub fn drive(prog: &Program) -> usize {
    let mut total_errors = 0usize;
    for it in &prog.items {
        if let Item::Fn(f) = it {
            if let Some(body) = &f.body {
                let mut events = Vec::new();
                let mut next_id = 0u32;
                walk_block(body, &mut events, &mut next_id);
                let errs = Checker::run(&events);
                total_errors += errs.len();
            }
        }
    }
    total_errors
}

fn walk_block(b: &Block, ev: &mut Vec<BorrowEvent>, id: &mut u32) {
    for s in &b.stmts { walk_stmt(s, ev, id); }
    if let Some(t) = &b.tail { walk_expr_use(t, ev); }
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
                            walk_expr_use(v, ev);
                        }
                    }
                    _ => walk_expr_use(v, ev),
                }
            }
        }
        Stmt::LetTuple { value, .. } => walk_expr_use(value, ev),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => walk_expr_use(e, ev),
        Stmt::Return(None) => {}
    }
}
fn walk_expr_use(e: &Expr, ev: &mut Vec<BorrowEvent>) {
    match e {
        Expr::Ident(n) => ev.push(BorrowEvent::Use { place: n.clone() }),
        Expr::Bin { lhs, rhs, .. } => { walk_expr_use(lhs, ev); walk_expr_use(rhs, ev); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => walk_expr_use(expr, ev),
        Expr::Call { args, .. } => for a in args { walk_expr_use(a, ev); },
        Expr::MethodCall { recv, args, .. } => { walk_expr_use(recv, ev); for a in args { walk_expr_use(a, ev); } }
        Expr::Field { recv, .. } => walk_expr_use(recv, ev),
        _ => {}
    }
}
