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
//! any explicit annotation) and to every call argument (vs. the declared param
//! type).
//!
//! Diagnostics are emitted conservatively — only when both sides of a clash are
//! concrete scalars (`Int` / `Float` / `Bool`). Aggregate / pointer / generic
//! types are inferred as opaque vars and never flagged, so the engine adds a
//! real check without false-positiving on the rich existing suite. Integer
//! widths are bucketed (Aether casts freely between them); the catch is the
//! cross-bucket case `Int` vs `Float`.
//!
//!   * `AE0220` — a `let` annotation conflicts with the inferred value.
//!   * `AE0221` — a call argument conflicts with the declared parameter type.

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

/// Declared signatures: per fn-name return type + parameter types.
#[derive(Default)]
struct Sigs {
    ret: HashMap<String, Type>,
    params: HashMap<String, Vec<Type>>,
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

    /// Follow var bindings to a representative.
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

    /// Occurs check: does var `v` appear in (the resolution of) `t`?
    fn occurs(&self, v: u32, t: &Type) -> bool {
        match self.resolve(t) {
            Type::Var(w) => v == w,
            _ => false, // monomorphic core has no nested vars
        }
    }

    /// Unify two types. Binds free vars; returns the conflicting pair on a
    /// concrete clash. The caller decides whether a clash warrants a diagnostic.
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
    let mut sigs = Sigs::default();

    // Pre-collect fn return + parameter types so calls can be checked.
    let mut record = |ctx: &mut InferCtx, sigs: &mut Sigs, name: String, f: &crate::ast::FnDecl| {
        sigs.ret.insert(name.clone(), ret_type(ctx, &f.ret));
        let ptys: Vec<Type> = f.params.iter().map(|p| ann_to_type(ctx, &p.ty)).collect();
        sigs.params.insert(name, ptys);
    };
    for it in &prog.items {
        match it {
            Item::Fn(f) => record(&mut ctx, &mut sigs, f.name.clone(), f),
            Item::Impl { type_name, methods } | Item::ImplTrait { type_name, methods, .. } => {
                for m in methods {
                    record(&mut ctx, &mut sigs, format!("{}__{}", type_name, m.name), m);
                    record(&mut ctx, &mut sigs, m.name.clone(), m);
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
            let body_t = infer_block(&mut ctx, &sigs, &mut env, body, &mut diags);
            // P6.1 — the implicit-return (tail) expression must match the
            // declared return type. Explicit `return e;` checking is a
            // follow-up (needs the expected type threaded into the walk).
            if let Some(ret_ann) = &f.ret {
                let rt = ann_to_type(&mut ctx, ret_ann);
                if let Err((a, b)) = ctx.unify(&body_t, &rt) {
                    if a.is_scalar() && b.is_scalar() {
                        diags.push(Diag::error("AE0222", "type",
                            format!("`{}` returns {}, but its body yields {}",
                                f.name, bucket_name(&b), bucket_name(&a)))
                            .with_hint("make the final expression match the declared return type, \
                                or insert an explicit `as` cast"));
                    }
                }
            }
        }
    }
    diags
}

fn infer_block(
    ctx: &mut InferCtx,
    sigs: &Sigs,
    env: &mut HashMap<String, Type>,
    b: &Block,
    diags: &mut Vec<Diag>,
) -> Type {
    for s in &b.stmts {
        infer_stmt(ctx, sigs, env, s, diags);
    }
    match &b.tail {
        Some(t) => infer_expr(ctx, sigs, env, t, diags),
        None => Type::Unit,
    }
}

fn infer_stmt(
    ctx: &mut InferCtx,
    sigs: &Sigs,
    env: &mut HashMap<String, Type>,
    s: &Stmt,
    diags: &mut Vec<Diag>,
) {
    match s {
        Stmt::Let { name, ty, value, .. } => {
            let rhs_t = match value {
                Some(v) => infer_expr(ctx, sigs, env, v, diags),
                None => ctx.fresh(),
            };
            let bind_t = match ty {
                Some(ann) => {
                    let ann_t = ann_to_type(ctx, ann);
                    if let Err((a, b)) = ctx.unify(&ann_t, &rhs_t) {
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
            let _ = infer_expr(ctx, sigs, env, value, diags);
            for n in names { let v = ctx.fresh(); env.insert(n.clone(), v); }
        }
        Stmt::Expr(e) | Stmt::Return(Some(e)) => { let _ = infer_expr(ctx, sigs, env, e, diags); }
        Stmt::Return(None) => {}
    }
}

fn infer_expr(
    ctx: &mut InferCtx,
    sigs: &Sigs,
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
            let lt = infer_expr(ctx, sigs, env, lhs, diags);
            let rt = infer_expr(ctx, sigs, env, rhs, diags);
            match op {
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge
                | BinOp::And | BinOp::Or => Type::Bool,
                BinOp::Assign => Type::Unit,
                _ => {
                    let lr = ctx.resolve(&lt);
                    if lr.is_scalar() { lr } else { ctx.resolve(&rt) }
                }
            }
        }
        Expr::Unary { op, expr } => {
            let t = infer_expr(ctx, sigs, env, expr, diags);
            match op {
                crate::ast::UnOp::Not => Type::Bool,
                crate::ast::UnOp::Neg => ctx.resolve(&t),
            }
        }
        Expr::Call { callee, args } => {
            let arg_types: Vec<Type> = args.iter()
                .map(|a| infer_expr(ctx, sigs, env, a, diags))
                .collect();
            if let Expr::Ident(n) = callee.as_ref() {
                // Check each arg against the declared parameter type.
                if let Some(ptys) = sigs.params.get(n) {
                    for (i, (at, pt)) in arg_types.iter().zip(ptys.iter()).enumerate() {
                        if let Err((a, b)) = ctx.unify(at, pt) {
                            if a.is_scalar() && b.is_scalar() {
                                diags.push(Diag::error("AE0221", "type",
                                    format!("argument {} to `{}` is {}, but the parameter is {}",
                                        i + 1, n, bucket_name(&a), bucket_name(&b)))
                                    .with_hint("pass a value of the parameter's type or insert an \
                                        explicit `as` cast at the call site"));
                            }
                        }
                    }
                }
                sigs.ret.get(n).cloned().unwrap_or_else(|| ctx.fresh())
            } else {
                let _ = infer_expr(ctx, sigs, env, callee, diags);
                ctx.fresh()
            }
        }
        Expr::Cast { expr, ty } => {
            let _ = infer_expr(ctx, sigs, env, expr, diags);
            ann_to_type(ctx, &Ty::Named(ty.clone()))
        }
        Expr::Block(b) => infer_block(ctx, sigs, env, b, diags),
        Expr::If { cond, then, else_ } => {
            let _ = infer_expr(ctx, sigs, env, cond, diags);
            let tt = infer_block(ctx, sigs, env, then, diags);
            if let Some(eb) = else_ {
                let _ = infer_block(ctx, sigs, env, eb, diags);
            }
            tt
        }
        Expr::While { cond, body } => {
            let _ = infer_expr(ctx, sigs, env, cond, diags);
            let _ = infer_block(ctx, sigs, env, body, diags);
            Type::Unit
        }
        Expr::For { iter, body, .. } => {
            let _ = infer_expr(ctx, sigs, env, iter, diags);
            let _ = infer_block(ctx, sigs, env, body, diags);
            Type::Unit
        }
        Expr::Region { body, .. } => infer_block(ctx, sigs, env, body, diags),
        Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => {
            let _ = infer_expr(ctx, sigs, env, expr, diags);
            ctx.fresh()
        }
        Expr::MethodCall { recv, args, .. } => {
            let _ = infer_expr(ctx, sigs, env, recv, diags);
            for a in args { let _ = infer_expr(ctx, sigs, env, a, diags); }
            ctx.fresh()
        }
        Expr::Field { recv, .. } => { let _ = infer_expr(ctx, sigs, env, recv, diags); ctx.fresh() }
        Expr::Index { recv, idx } => {
            let _ = infer_expr(ctx, sigs, env, recv, diags);
            let _ = infer_expr(ctx, sigs, env, idx, diags);
            ctx.fresh()
        }
        Expr::Range { lo, hi, step } => {
            let _ = infer_expr(ctx, sigs, env, lo, diags);
            let _ = infer_expr(ctx, sigs, env, hi, diags);
            if let Some(s) = step { let _ = infer_expr(ctx, sigs, env, s, diags); }
            ctx.fresh()
        }
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
                    Stmt::Let { name: "y".into(), mutable: false,
                        ty: Some(Ty::Named("f32".into())), value: Some(Expr::FloatLit(2.5)) },
                ],
                tail: None,
            }),
        })] };
        assert!(run(&prog).is_empty());
    }

    #[test]
    fn arg_type_mismatch_flagged() {
        // fn f(a: i64) -> i64 { a }  fn main() { f(3.5); }
        use crate::ast::*;
        let f = FnDecl {
            attrs: vec![], is_pub: false, is_extern: false, name: "f".into(),
            const_params: vec![], params: vec![Param { name: "a".into(), ty: Ty::Named("i64".into()) }],
            ret: Some(Ty::Named("i64".into())),
            body: Some(Block { stmts: vec![], tail: Some(Box::new(Expr::Ident("a".into()))) }),
        };
        let main = FnDecl {
            attrs: vec![], is_pub: false, is_extern: false, name: "main".into(),
            const_params: vec![], params: vec![], ret: None,
            body: Some(Block {
                stmts: vec![Stmt::Expr(Expr::Call {
                    callee: Box::new(Expr::Ident("f".into())),
                    args: vec![Expr::FloatLit(3.5)],
                })],
                tail: None,
            }),
        };
        let prog = Program { items: vec![Item::Fn(f), Item::Fn(main)] };
        let diags = run(&prog);
        assert!(diags.iter().any(|d| d.code == "AE0221"));
    }
}
