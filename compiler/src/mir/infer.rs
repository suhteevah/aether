//! Phase 6.1 — Hindley-Milner-style type inference (monomorphic core).
//!
//! The asm backend already *tolerates* a missing `let` annotation by defaulting
//! the storage class from the rhs, so `let x = 5;` compiles. What was missing
//! is an actual inference engine — and the type *checking* that comes with it.
//! Today `let x: i64 = 3.5;` silently compiles wrong. This module is the engine
//! that catches it.
//!
//! It implements the Algorithm-W kernel: type variables in a union-find
//! substitution table, `unify` with an occurs check, and a bottom-up inference
//! walk that assigns a `Type` to every binding (from the rhs, reconciled with
//! any explicit annotation).
//!
//! Diagnostics are emitted conservatively — only when an explicit annotation
//! and the inferred rhs are BOTH concrete scalars (`Int` / `Float` / `Bool`)
//! that genuinely conflict. Aggregate / pointer / generic types are inferred as
//! opaque vars and never flagged, so the engine adds a real check without
//! false-positiving on the rich existing suite. Integer widths are bucketed
//! (Aether casts freely between them), so `let x: i32 = 5i64` is fine; the
//! catch is the cross-bucket case `Int` vs `Float`.

use crate::ast::{BinOp, Block, Expr, Item, Program, Stmt, Ty};
use crate::diag::Diag;
use std::collections::HashMap;

/// Inferred type. Integer widths collapse to `Int`; `f32`/`f64` to `Float`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Str,
    Named(String),
    Unit,
    Var(u32),
}

impl Type {
    fn is_scalar(&self) -> bool {
        matches!(self, Type::Int | Type::Float | Type::Bool)
    }
}

#[derive(Default)]
pub struct InferCtx {
    /// Union-find store: `subst[v]` is the binding of var `v` (another type,
    /// possibly a var) or `None` if still free.
    subst: Vec<Option<Type>>,
}

impl InferCtx {
    fn fresh(&mut self) -> Type {
        let id = self.subst.len() as u32;
        self.subst.push(None);
        Type::Var(id)
    }

    /// Follow var bindings to a representative (path-compressed conceptually).
    fn resolve(&self, t: &Type) -> Type {
        let mut cur = t.clone();
        while let Type::Var(v) = cur {
            match self.subst.get(v as usize).and_then(|o| o.clone()) {
                Some(next) => cur = next,
                None => return Type::Var(v),
            }
        }
        cur
    }

    /// Occurs check: does var `v` appear in (the resolution of) `t`? Prevents
    /// building an infinite type when unifying `v` with something mentioning it.
    fn occurs(&self, v: u32, t: &Type) -> bool {
        match self.resolve(t) {
            Type::Var(w) => v == w,
            _ => false, // our Types have no nested vars (monomorphic core)
        }
    }

    /// Unify two types. Binds free vars; returns the conflicting pair on a
    /// concrete clash. The let-checker decides whether a given clash is worth a
    /// diagnostic (only scalar-vs-scalar today).
    pub fn unify(&mut self, a: &Type, b: &Type) -> Result<(), (Type, Type)> {
        let ra = self.resolve(a);
        let rb = self.resolve(b);
        match (&ra, &rb) {
            (Type::Var(x), Type::Var(y)) if x == y => Ok(()),
            (Type::Var(x), _) => {
                if self.occurs(*x, &rb) { return Err((ra, rb)); }
                self.subst[*x as usize] = Some(rb);
                Ok(())
            }
            (_, Type::Var(y)) => {
                if self.occurs(*y, &ra) { return Err((ra, rb)); }
                self.subst[*y as usize] = Some(ra);
                Ok(())
            }
            (Type::Int, Type::Int)
            | (Type::Float, Type::Float)
            | (Type::Bool, Type::Bool)
            | (Type::Str, Type::Str)
            | (Type::Unit, Type::Unit) => Ok(()),
            (Type::Named(x), Type::Named(y)) if x == y => Ok(()),
            _ => Err((ra, rb)),
        }
    }
}

/// Map an AST annotation to an inferred `Type`. Scalars become concrete
/// buckets; everything aggregate/reference becomes a fresh opaque var so it
/// unifies with anything (no false positives on rich types).
fn ann_to_type(ctx: &mut InferCtx, ty: &Ty) -> Type {
    match ty {
        Ty::Named(n) => match n.as_str() {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize"
                => Type::Int,
            "f32" | "f64" => Type::Float,
            "bool" => Type::Bool,
            "str" | "String" => Type::Str,
            _ => Type::Named(n.clone()),
        },
        Ty::Unit => Type::Unit,
        // References / slices / arrays / tuples / shapes / generics: opaque.
        _ => ctx.fresh(),
    }
}

fn ret_type(ctx: &mut InferCtx, ret: &Option<Ty>) -> Type {
    match ret {
        Some(t) => ann_to_type(ctx, t),
        None => Type::Unit,
    }
}

pub fn run(prog: &Program) -> Vec<Diag> {
    let mut ctx = InferCtx::default();
    let mut diags = Vec::new();

    // Pre-collect fn return types so call expressions infer their result.
    let mut fn_ret: HashMap<String, Type> = HashMap::new();
    for it in &prog.items {
        match it {
            Item::Fn(f) => { let t = ret_type(&mut ctx, &f.ret); fn_ret.insert(f.name.clone(), t); }
            Item::Impl { type_name, methods } | Item::ImplTrait { type_name, methods, .. } => {
                for m in methods {
                    let t = ret_type(&mut ctx, &m.ret);
                    fn_ret.insert(format!("{}__{}", type_name, m.name), t.clone());
                    fn_ret.insert(m.name.clone(), t);
                }
            }
            _ => {}
        }
    }

    for it in &prog.items {
        if let Item::Fn(f) = it {
            let Some(body) = &f.body else { continue; };
            let mut env: HashMap<String, Type> = HashMap::new();
            for p in &f.params {
                let t = ann_to_type(&mut ctx, &p.ty);
                env.insert(p.name.clone(), t);
            }
            infer_block(&mut ctx, &fn_ret, &mut env, body, &mut diags);
        }
    }
    diags
}

fn infer_block(
    ctx: &mut InferCtx,
    fn_ret: &HashMap<String, Type>,
    env: &mut HashMap<String, Type>,
    b: &Block,
    diags: &mut Vec<Diag>,
) -> Type {
    for s in &b.stmts {
        infer_stmt(ctx, fn_ret, env, s, diags);
    }
    match &b.tail {
        Some(t) => infer_expr(ctx, fn_ret, env, t, diags),
        None => Type::Unit,
    }
}

fn infer_stmt(
    ctx: &mut InferCtx,
    fn_ret: &HashMap<String, Type>,
    env: &mut HashMap<String, Type>,
    s: &Stmt,
    diags: &mut Vec<Diag>,
) {
    match s {
        Stmt::Let { name, ty, value, .. } => {
            let rhs_t = match value {
                Some(v) => infer_expr(ctx, fn_ret, env, v, diags),
                None => ctx.fresh(),
            };
            let bind_t = match ty {
                Some(ann) => {
                    let ann_t = ann_to_type(ctx, ann);
                    if let Err((a, b)) = ctx.unify(&ann_t, &rhs_t) {
                        // Only a concrete scalar-vs-scalar clash is reported.
                        if a.is_scalar() && b.is_scalar() {
                            diags.push(Diag::error("AE0220", "type",
                                format!("type mismatch in `let {}`: annotation is {}, but the value is {}",
                                    name, bucket_name(&a), bucket_name(&b)))
                                .with_hint("make the value match the annotation, drop the annotation to \
                                    infer it, or insert an explicit `as` cast"));
                        }
                    }
                    ann_t
                }
                None => rhs_t,
            };
            env.insert(name.clone(), ctx.resolve(&bind_t));
        }
        Stmt::LetTuple { names, value } => {
            let _ = infer_expr(ctx, fn_ret, env, value, diags);
            for n in names { let v = ctx.fresh(); env.insert(n.clone(), v); }
        }
        Stmt::Expr(e) | Stmt::Return(Some(e)) => { let _ = infer_expr(ctx, fn_ret, env, e, diags); }
        Stmt::Return(None) => {}
    }
}

fn infer_expr(
    ctx: &mut InferCtx,
    fn_ret: &HashMap<String, Type>,
    env: &mut HashMap<String, Type>,
    e: &Expr,
    diags: &mut Vec<Diag>,
) -> Type {
    match e {
        Expr::IntLit(_) => Type::Int,
        Expr::FloatLit(_) => Type::Float,
        Expr::BoolLit(_) => Type::Bool,
        Expr::StrLit(_) => Type::Str,
        Expr::Ident(n) => env.get(n).cloned().unwrap_or_else(|| ctx.fresh()),
        Expr::Bin { op, lhs, rhs } => {
            let lt = infer_expr(ctx, fn_ret, env, lhs, diags);
            let rt = infer_expr(ctx, fn_ret, env, rhs, diags);
            match op {
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                | BinOp::And | BinOp::Or => Type::Bool,
                BinOp::Assign => Type::Unit,
                // Arithmetic / bitwise: result follows a concrete operand.
                _ => {
                    let lr = ctx.resolve(&lt);
                    if lr.is_scalar() { lr } else { ctx.resolve(&rt) }
                }
            }
        }
        Expr::Unary { op, expr } => {
            let t = infer_expr(ctx, fn_ret, env, expr, diags);
            match op {
                crate::ast::UnOp::Not => Type::Bool,
                crate::ast::UnOp::Neg => ctx.resolve(&t),
            }
        }
        Expr::Call { callee, args } => {
            for a in args { let _ = infer_expr(ctx, fn_ret, env, a, diags); }
            if let Expr::Ident(n) = callee.as_ref() {
                fn_ret.get(n).cloned().unwrap_or_else(|| ctx.fresh())
            } else {
                let _ = infer_expr(ctx, fn_ret, env, callee, diags);
                ctx.fresh()
            }
        }
        Expr::Cast { expr, ty } => {
            let _ = infer_expr(ctx, fn_ret, env, expr, diags);
            // The cast target dictates the result bucket.
            ann_to_type(ctx, &Ty::Named(ty.clone()))
        }
        Expr::Block(b) => infer_block(ctx, fn_ret, env, b, diags),
        Expr::If { cond, then, else_ } => {
            let _ = infer_expr(ctx, fn_ret, env, cond, diags);
            let tt = infer_block(ctx, fn_ret, env, then, diags);
            if let Some(eb) = else_ {
                let _ = infer_block(ctx, fn_ret, env, eb, diags);
            }
            tt
        }
        Expr::While { cond, body } => {
            let _ = infer_expr(ctx, fn_ret, env, cond, diags);
            let _ = infer_block(ctx, fn_ret, env, body, diags);
            Type::Unit
        }
        Expr::For { iter, body, .. } => {
            let _ = infer_expr(ctx, fn_ret, env, iter, diags);
            let _ = infer_block(ctx, fn_ret, env, body, diags);
            Type::Unit
        }
        Expr::Region { body, .. } => infer_block(ctx, fn_ret, env, body, diags),
        Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => {
            let _ = infer_expr(ctx, fn_ret, env, expr, diags);
            ctx.fresh()
        }
        Expr::MethodCall { recv, args, .. } => {
            let _ = infer_expr(ctx, fn_ret, env, recv, diags);
            for a in args { let _ = infer_expr(ctx, fn_ret, env, a, diags); }
            ctx.fresh()
        }
        Expr::Field { recv, .. } => { let _ = infer_expr(ctx, fn_ret, env, recv, diags); ctx.fresh() }
        Expr::Index { recv, idx } => {
            let _ = infer_expr(ctx, fn_ret, env, recv, diags);
            let _ = infer_expr(ctx, fn_ret, env, idx, diags);
            ctx.fresh()
        }
        Expr::Range { lo, hi, step } => {
            let _ = infer_expr(ctx, fn_ret, env, lo, diags);
            let _ = infer_expr(ctx, fn_ret, env, hi, diags);
            if let Some(s) = step { let _ = infer_expr(ctx, fn_ret, env, s, diags); }
            ctx.fresh()
        }
        // Aggregate / control literals: opaque for now.
        _ => ctx.fresh(),
    }
}

fn bucket_name(t: &Type) -> &'static str {
    match t {
        Type::Int => "an integer",
        Type::Float => "a float",
        Type::Bool => "a bool",
        Type::Str => "a string",
        _ => "a different type",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unify_binds_var_to_concrete() {
        let mut c = InferCtx::default();
        let v = c.fresh();
        assert!(c.unify(&v, &Type::Int).is_ok());
        assert_eq!(c.resolve(&v), Type::Int);
    }

    #[test]
    fn unify_concrete_conflict_errs() {
        let mut c = InferCtx::default();
        assert!(c.unify(&Type::Int, &Type::Float).is_err());
        assert!(c.unify(&Type::Bool, &Type::Int).is_err());
        assert!(c.unify(&Type::Named("Foo".into()), &Type::Named("Bar".into())).is_err());
    }

    #[test]
    fn unify_same_concrete_ok() {
        let mut c = InferCtx::default();
        assert!(c.unify(&Type::Int, &Type::Int).is_ok());
        assert!(c.unify(&Type::Named("Foo".into()), &Type::Named("Foo".into())).is_ok());
    }

    #[test]
    fn var_chain_resolves() {
        let mut c = InferCtx::default();
        let a = c.fresh();
        let b = c.fresh();
        assert!(c.unify(&a, &b).is_ok());
        assert!(c.unify(&b, &Type::Float).is_ok());
        assert_eq!(c.resolve(&a), Type::Float);
    }

    #[test]
    fn mismatch_let_flagged() {
        // fn main() { let x: i64 = 3.5; }
        use crate::ast::*;
        let prog = Program { items: vec![Item::Fn(FnDecl {
            attrs: vec![], is_pub: false, is_extern: false, name: "main".into(),
            const_params: vec![], params: vec![], ret: None,
            body: Some(Block {
                stmts: vec![Stmt::Let {
                    name: "x".into(), mutable: false,
                    ty: Some(Ty::Named("i64".into())),
                    value: Some(Expr::FloatLit(3.5)),
                }],
                tail: None,
            }),
        })] };
        let diags = run(&prog);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].code, "AE0220");
    }

    #[test]
    fn matching_let_clean() {
        use crate::ast::*;
        let prog = Program { items: vec![Item::Fn(FnDecl {
            attrs: vec![], is_pub: false, is_extern: false, name: "main".into(),
            const_params: vec![], params: vec![], ret: None,
            body: Some(Block {
                stmts: vec![
                    Stmt::Let { name: "x".into(), mutable: false,
                        ty: Some(Ty::Named("i64".into())), value: Some(Expr::IntLit(5)) },
                    // f32 annotation + float literal: same bucket, no error.
                    Stmt::Let { name: "y".into(), mutable: false,
                        ty: Some(Ty::Named("f32".into())), value: Some(Expr::FloatLit(2.5)) },
                ],
                tail: None,
            }),
        })] };
        assert!(run(&prog).is_empty());
    }
}
