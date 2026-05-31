//! Phase 6 — associated-function (UFCS static) calls.
//!
//! Rewrites `Type::method(args)` (a `Call` with a `Path([Type, method])`
//! callee) to the flattened `Type__method(args)` when `Type__method` is a
//! real inherent/trait impl method. This is how Rust's associated functions /
//! constructors are spelled — `Counter::new()`, `Celsius::from(40)`,
//! `T::default()` — and the flattener already emits the `Type__method` fn.
//!
//! Precise by construction: only Path-calls that resolve to a known impl
//! method are rewritten, so enum constructors (`Color::Red(x)`, handled by the
//! backend's `resolve_enum_ctor`) and any other 2-segment path are left alone.

use crate::ast::{Block, Expr, Item, Program, Stmt};
use std::collections::HashSet;

pub fn run(prog: &mut Program) -> usize {
    let mut methods: HashSet<String> = HashSet::new();
    for it in &prog.items {
        if let Item::Impl { type_name, methods: ms } | Item::ImplTrait { type_name, methods: ms, .. } = it {
            for m in ms { methods.insert(format!("{}__{}", type_name, m.name)); }
        }
    }
    if methods.is_empty() { return 0; }

    let mut count = 0;
    for item in prog.items.iter_mut() {
        match item {
            Item::Fn(f) => { if let Some(b) = f.body.as_mut() { walk_block(b, &methods, &mut count); } }
            Item::Impl { methods: ms, .. } | Item::ImplTrait { methods: ms, .. } => {
                for m in ms.iter_mut() { if let Some(b) = m.body.as_mut() { walk_block(b, &methods, &mut count); } }
            }
            _ => {}
        }
    }
    count
}

fn walk_block(b: &mut Block, methods: &HashSet<String>, count: &mut usize) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { value: Some(e), .. } => walk_expr(e, methods, count),
            Stmt::LetTuple { value, .. } => walk_expr(value, methods, count),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => walk_expr(e, methods, count),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { walk_expr(t, methods, count); }
}

fn walk_expr(e: &mut Expr, methods: &HashSet<String>, count: &mut usize) {
    // Rewrite `Type::method(args)` -> `Type__method(args)` when known.
    if let Expr::Call { callee, args } = e {
        if let Expr::Path(p) = callee.as_ref() {
            if p.len() == 2 {
                let mangled = format!("{}__{}", p[0], p[1]);
                if methods.contains(&mangled) {
                    let new_args = std::mem::take(args);
                    let mut rewritten_args = new_args;
                    for a in rewritten_args.iter_mut() { walk_expr(a, methods, count); }
                    *e = Expr::Call {
                        callee: Box::new(Expr::Ident(mangled)),
                        args: rewritten_args,
                    };
                    *count += 1;
                    return;
                }
            }
        }
    }
    match e {
        Expr::Call { callee, args } => { walk_expr(callee, methods, count); for a in args { walk_expr(a, methods, count); } }
        Expr::MethodCall { recv, args, .. } => { walk_expr(recv, methods, count); for a in args { walk_expr(a, methods, count); } }
        Expr::Bin { lhs, rhs, .. } => { walk_expr(lhs, methods, count); walk_expr(rhs, methods, count); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => walk_expr(expr, methods, count),
        Expr::Field { recv, .. } => walk_expr(recv, methods, count),
        Expr::Index { recv, idx } => { walk_expr(recv, methods, count); walk_expr(idx, methods, count); }
        Expr::Block(b) => walk_block(b, methods, count),
        Expr::If { cond, then, else_ } => {
            walk_expr(cond, methods, count);
            walk_block(then, methods, count);
            if let Some(b) = else_ { walk_block(b, methods, count); }
        }
        Expr::For { iter, body, .. } => { walk_expr(iter, methods, count); walk_block(body, methods, count); }
        Expr::While { cond, body } => { walk_expr(cond, methods, count); walk_block(body, methods, count); }
        Expr::Region { body, .. } => walk_block(body, methods, count),
        Expr::StructLit { fields, .. } => for (_, fv) in fields { walk_expr(fv, methods, count); },
        Expr::Match { scrutinee, arms } => {
            walk_expr(scrutinee, methods, count);
            for (_, a) in arms { walk_expr(a, methods, count); }
        }
        Expr::Tuple(elems) => for e in elems { walk_expr(e, methods, count); },
        Expr::Range { lo, hi, step } => {
            walk_expr(lo, methods, count); walk_expr(hi, methods, count);
            if let Some(s) = step { walk_expr(s, methods, count); }
        }
        _ => {}
    }
}
