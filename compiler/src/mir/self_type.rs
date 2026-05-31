//! Phase 6 — `Self` type resolution.
//!
//! Within `impl T { … }` / `impl Tr for T { … }`, replaces every `Self` with
//! the concrete type `T`: return/param/let-annotation types, `Self { … }`
//! struct literals, `Self::method` paths, and `expr as Self` casts. This makes
//! the ubiquitous Rust constructor idiom work:
//!     impl Counter { fn new() -> Self { Self { n: 0 } } }
//!
//! Runs before the associated-fn (`path_call`) + struct-literal codegen so they
//! see the concrete type. Pure AST rewrite.

use crate::ast::{Block, Expr, FnDecl, Item, Program, Stmt, Ty};

pub fn run(prog: &mut Program) -> usize {
    let mut count = 0;
    for item in prog.items.iter_mut() {
        if let Item::Impl { type_name, methods } | Item::ImplTrait { type_name, methods, .. } = item {
            let t = type_name.clone();
            for m in methods.iter_mut() { fix_fn(m, &t, &mut count); }
        }
    }
    count
}

fn fix_fn(f: &mut FnDecl, t: &str, count: &mut usize) {
    for p in f.params.iter_mut() { fix_ty(&mut p.ty, t, count); }
    if let Some(ret) = f.ret.as_mut() { fix_ty(ret, t, count); }
    if let Some(b) = f.body.as_mut() { fix_block(b, t, count); }
}

fn fix_ty(ty: &mut Ty, t: &str, count: &mut usize) {
    match ty {
        Ty::Named(n) if n == "Self" => { *n = t.to_string(); *count += 1; }
        Ty::Ref { inner, .. } => fix_ty(inner, t, count),
        Ty::Generic { args, .. } => for a in args { fix_ty(a, t, count); },
        Ty::Slice { elem, .. } => fix_ty(elem, t, count),
        Ty::Array { elem, .. } => fix_ty(elem, t, count),
        Ty::Tuple(elems) => for e in elems { fix_ty(e, t, count); },
        _ => {}
    }
}

fn fix_block(b: &mut Block, t: &str, count: &mut usize) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { ty, value, .. } => {
                if let Some(a) = ty.as_mut() { fix_ty(a, t, count); }
                if let Some(e) = value.as_mut() { fix_expr(e, t, count); }
            }
            Stmt::LetTuple { value, .. } => fix_expr(value, t, count),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => fix_expr(e, t, count),
            _ => {}
        }
    }
    if let Some(tl) = b.tail.as_mut() { fix_expr(tl, t, count); }
}

fn fix_expr(e: &mut Expr, t: &str, count: &mut usize) {
    match e {
        Expr::StructLit { name, fields } => {
            if name == "Self" { *name = t.to_string(); *count += 1; }
            for (_, fv) in fields { fix_expr(fv, t, count); }
        }
        Expr::Path(p) => {
            if p.first().map(|s| s.as_str()) == Some("Self") { p[0] = t.to_string(); *count += 1; }
        }
        Expr::Cast { expr, ty } => {
            if ty == "Self" { *ty = t.to_string(); *count += 1; }
            fix_expr(expr, t, count);
        }
        Expr::Call { callee, args } => { fix_expr(callee, t, count); for a in args { fix_expr(a, t, count); } }
        Expr::MethodCall { recv, args, .. } => { fix_expr(recv, t, count); for a in args { fix_expr(a, t, count); } }
        Expr::Bin { lhs, rhs, .. } => { fix_expr(lhs, t, count); fix_expr(rhs, t, count); }
        Expr::Unary { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => fix_expr(expr, t, count),
        Expr::Field { recv, .. } => fix_expr(recv, t, count),
        Expr::Index { recv, idx } => { fix_expr(recv, t, count); fix_expr(idx, t, count); }
        Expr::Block(b) => fix_block(b, t, count),
        Expr::If { cond, then, else_ } => {
            fix_expr(cond, t, count);
            fix_block(then, t, count);
            if let Some(b) = else_ { fix_block(b, t, count); }
        }
        Expr::For { iter, body, .. } => { fix_expr(iter, t, count); fix_block(body, t, count); }
        Expr::While { cond, body } => { fix_expr(cond, t, count); fix_block(body, t, count); }
        Expr::Region { body, .. } => fix_block(body, t, count),
        Expr::Match { scrutinee, arms } => {
            fix_expr(scrutinee, t, count);
            for (_, arm) in arms { fix_expr(arm, t, count); }
        }
        _ => {}
    }
}
