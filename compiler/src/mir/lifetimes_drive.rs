//! Drive `mir::lifetimes::Checker` over each fn body (at `--check` AND on the
//! compile path). Walks AST stmts, synthesizes `BorrowEvent`s for every
//! `let r = &x;` / `let r = &mut x;` and every identifier use, then runs the
//! borrow checker. Returns the coded violations as `AE0200`-family diagnostics.
//!
//! NON-LEXICAL lifetimes: each `let`-bound borrow is ENDED at the LAST use of
//! its binding (an `EndBorrow` event inserted right after that use), instead of
//! living to end-of-fn. So `let a = &mut v; *a = 1; let b = &mut v; *b = 2;` —
//! valid Rust under NLL — now checks clean, while genuinely overlapping borrows
//! (both bindings still used afterwards) are still rejected. A borrow whose
//! binding is never used ends immediately after the borrow.
//!
//! Refs in *argument* position (`foo(&mut x)`) do NOT create a held borrow —
//! they're consumed by the call, so back-to-back `foo(&mut x); bar(&mut x);`
//! stays legal. Only a `let`-bound reference is tracked as live.

use crate::ast::{Block, Expr, Item, Program, Stmt};
use super::lifetimes::{BorrowEvent, BorrowKind, Checker, Violation};

pub fn drive(prog: &Program) -> Vec<Violation> {
    let mut all = Vec::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            if let Some(body) = &f.body {
                let mut events = Vec::new();
                let mut next_id = 0u32;
                // (binding name, borrow id, index of its Borrow event).
                let mut binds: Vec<(String, u32, usize)> = Vec::new();
                walk_block(body, &mut events, &mut next_id, &mut binds);

                // NLL: schedule an EndBorrow right after each binding's LAST use
                // (or right after its Borrow if it's never used).
                let mut after: Vec<(usize, u32)> = Vec::new();
                for (name, id, borrow_idx) in &binds {
                    let last_use = events.iter().enumerate().rev().find_map(|(i, e)| {
                        match e { BorrowEvent::Use { place } if place == name => Some(i), _ => None }
                    });
                    after.push((last_use.unwrap_or(*borrow_idx), *id));
                }
                // Rebuild the stream inserting each EndBorrow after its index.
                let mut out: Vec<BorrowEvent> = Vec::with_capacity(events.len() + after.len());
                for (i, e) in events.into_iter().enumerate() {
                    out.push(e);
                    for (idx, id) in after.iter().filter(|(idx, _)| *idx == i) {
                        let _ = idx;
                        out.push(BorrowEvent::EndBorrow { id: *id });
                    }
                }
                all.extend(Checker::run_coded(&out));
            }
        }
    }
    all
}

fn walk_block(b: &Block, ev: &mut Vec<BorrowEvent>, id: &mut u32, binds: &mut Vec<(String, u32, usize)>) {
    for s in &b.stmts { walk_stmt(s, ev, id, binds); }
    if let Some(t) = &b.tail { walk_expr(t, ev, id); }
}

fn walk_stmt(s: &Stmt, ev: &mut Vec<BorrowEvent>, id: &mut u32, binds: &mut Vec<(String, u32, usize)>) {
    match s {
        Stmt::Let { name, value, .. } => {
            if let Some(v) = value {
                match v {
                    Expr::Ref { mutable, expr } => {
                        if let Expr::Ident(place) = expr.as_ref() {
                            let kind = if *mutable { BorrowKind::Mut } else { BorrowKind::Shared };
                            let this_id = *id;
                            let borrow_idx = ev.len();
                            ev.push(BorrowEvent::Borrow { place: place.clone(), kind, id: this_id });
                            *id += 1;
                            // Record the binding so drive() can end the borrow at
                            // its last use (the NLL upgrade).
                            binds.push((name.clone(), this_id, borrow_idx));
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

// Plain expression walk (no new `let`-bound borrows are created here, so it
// doesn't thread `binds`); nested blocks recurse via `walk_block_e` which does.
fn walk_expr(e: &Expr, ev: &mut Vec<BorrowEvent>, id: &mut u32) {
    let mut sink: Vec<(String, u32, usize)> = Vec::new();
    walk_expr_b(e, ev, id, &mut sink);
}

fn walk_expr_b(e: &Expr, ev: &mut Vec<BorrowEvent>, id: &mut u32, binds: &mut Vec<(String, u32, usize)>) {
    match e {
        Expr::Ident(n) => ev.push(BorrowEvent::Use { place: n.clone() }),
        Expr::Bin { lhs, rhs, .. } => { walk_expr_b(lhs, ev, id, binds); walk_expr_b(rhs, ev, id, binds); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => walk_expr_b(expr, ev, id, binds),
        // A ref in expression/argument position is consumed in place — recurse
        // for nested uses but do NOT register a held borrow (that's the
        // `let`-binding case in walk_stmt).
        Expr::Ref { expr, .. } => walk_expr_b(expr, ev, id, binds),
        Expr::Call { args, .. } => for a in args { walk_expr_b(a, ev, id, binds); },
        Expr::MethodCall { recv, args, .. } => { walk_expr_b(recv, ev, id, binds); for a in args { walk_expr_b(a, ev, id, binds); } }
        Expr::Field { recv, .. } => walk_expr_b(recv, ev, id, binds),
        Expr::Index { recv, idx } => { walk_expr_b(recv, ev, id, binds); walk_expr_b(idx, ev, id, binds); }
        // Recurse into nested control-flow blocks so a held borrow inside an
        // `if`/`for`/`while`/region body is tracked too.
        Expr::Block(b) => walk_block(b, ev, id, binds),
        Expr::If { cond, then, else_ } => {
            walk_expr_b(cond, ev, id, binds);
            walk_block(then, ev, id, binds);
            if let Some(b) = else_ { walk_block(b, ev, id, binds); }
        }
        Expr::For { iter, body, .. } => { walk_expr_b(iter, ev, id, binds); walk_block(body, ev, id, binds); }
        Expr::While { cond, body } => { walk_expr_b(cond, ev, id, binds); walk_block(body, ev, id, binds); }
        Expr::Region { body, .. } => walk_block(body, ev, id, binds),
        _ => {}
    }
}
