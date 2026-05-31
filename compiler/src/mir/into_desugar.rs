//! Phase 6.5 — `.into()` desugaring, backed by `From`.
//!
//! Rewrites `let x: T = e.into();` to `let x: T = T::from(e);` (the flattened
//! `T__from(e)`) for every type `T` that has an `impl From<…> for T` in the
//! program. The conversion fn returns `T` by value, which the P6.5 struct-return
//! ABI handles. Precise: only fires for types that actually have a `From` impl,
//! so unrelated `.into()` calls are left alone and the rewritten call always
//! resolves to a real fn.
//!
//! Scope (v1): `.into()` in `let`-with-annotation position (the target type is
//! read from the annotation; full context-driven `.into()` inference is a
//! follow-up). One `From` impl per target type (multiple sources would collide
//! on the single `T__from` mangled name).

use crate::ast::{Block, Expr, Item, Program, Stmt, Ty};
use std::collections::HashSet;

pub fn run(prog: &mut Program) -> usize {
    let from_types: HashSet<String> = prog.items.iter().filter_map(|it| {
        if let Item::ImplTrait { trait_name, type_name, .. } = it {
            if trait_name == "From" { return Some(type_name.clone()); }
        }
        None
    }).collect();
    if from_types.is_empty() { return 0; }

    let mut count = 0;
    for item in prog.items.iter_mut() {
        match item {
            Item::Fn(f) => { if let Some(b) = f.body.as_mut() { desugar_block(b, &from_types, &mut count); } }
            Item::Impl { methods, .. } | Item::ImplTrait { methods, .. } => {
                for m in methods.iter_mut() {
                    if let Some(b) = m.body.as_mut() { desugar_block(b, &from_types, &mut count); }
                }
            }
            _ => {}
        }
    }
    count
}

fn desugar_block(b: &mut Block, from: &HashSet<String>, count: &mut usize) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { ty: Some(Ty::Named(t)), value: Some(v), .. } if from.contains(t) => {
                if let Some(recv) = into_receiver(v) {
                    *v = Expr::Call {
                        callee: Box::new(Expr::Ident(format!("{}__from", t))),
                        args: vec![recv],
                    };
                    *count += 1;
                } else {
                    desugar_expr(v, from, count);
                }
            }
            Stmt::Let { value: Some(v), .. } => desugar_expr(v, from, count),
            Stmt::LetTuple { value, .. } => desugar_expr(value, from, count),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => desugar_expr(e, from, count),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { desugar_expr(t, from, count); }
}

/// If `e` is `<recv>.into()` (no args), return the receiver to convert.
fn into_receiver(e: &Expr) -> Option<Expr> {
    if let Expr::MethodCall { recv, name, args } = e {
        if name == "into" && args.is_empty() {
            return Some((**recv).clone());
        }
    }
    None
}

fn desugar_expr(e: &mut Expr, from: &HashSet<String>, count: &mut usize) {
    match e {
        Expr::Block(b) => desugar_block(b, from, count),
        Expr::If { cond, then, else_ } => {
            desugar_expr(cond, from, count);
            desugar_block(then, from, count);
            if let Some(b) = else_ { desugar_block(b, from, count); }
        }
        Expr::For { iter, body, .. } => { desugar_expr(iter, from, count); desugar_block(body, from, count); }
        Expr::While { cond, body } => { desugar_expr(cond, from, count); desugar_block(body, from, count); }
        Expr::Region { body, .. } => desugar_block(body, from, count),
        Expr::Call { callee, args } => { desugar_expr(callee, from, count); for a in args { desugar_expr(a, from, count); } }
        Expr::MethodCall { recv, args, .. } => { desugar_expr(recv, from, count); for a in args { desugar_expr(a, from, count); } }
        Expr::Bin { lhs, rhs, .. } => { desugar_expr(lhs, from, count); desugar_expr(rhs, from, count); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => desugar_expr(expr, from, count),
        Expr::Field { recv, .. } => desugar_expr(recv, from, count),
        Expr::Index { recv, idx } => { desugar_expr(recv, from, count); desugar_expr(idx, from, count); }
        _ => {}
    }
}
