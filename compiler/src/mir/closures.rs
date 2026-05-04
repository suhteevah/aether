//! Closure-lifting pass.
//!
//! Walks every `Expr::Closure { params, body }` in the program, generates a
//! synthetic top-level fn `__closure_<n>` containing the body verbatim, and
//! rewrites the closure expression in-place to `Expr::Ident("__closure_<n>")`.
//! That ident loads as a function pointer (the asm backend's
//! `Expr::Ident` arm emits `leaq aether_<name>(%rip), %rax` when `<name>`
//! is a known fn).
//!
//! Today this is **closures-without-captures only** — closure bodies that
//! reference outer-scope locals will fail at codegen with `UnknownIdent`
//! since the lifted fn doesn't see the caller's frame. Capturing closures
//! need an env-struct + an indirect call ABI; both are well-scoped future
//! work but not in this pass yet.

use crate::ast::{Block, Expr, FnDecl, Item, Param, Program, Stmt, Ty};

pub fn run(prog: &mut Program) -> usize {
    let mut ctx = LiftCtx { lifted: Vec::new(), counter: 0 };
    for item in prog.items.iter_mut() {
        match item {
            Item::Fn(f) => lift_fn(f, &mut ctx),
            Item::Impl { methods, .. } => {
                for m in methods.iter_mut() { lift_fn(m, &mut ctx); }
            }
            _ => {}
        }
    }
    let n = ctx.lifted.len();
    for f in ctx.lifted { prog.items.push(Item::Fn(f)); }
    n
}

struct LiftCtx {
    lifted: Vec<FnDecl>,
    counter: usize,
}

fn lift_fn(f: &mut FnDecl, ctx: &mut LiftCtx) {
    if let Some(b) = f.body.as_mut() { lift_block(b, ctx); }
}

fn lift_block(b: &mut Block, ctx: &mut LiftCtx) {
    for s in b.stmts.iter_mut() { lift_stmt(s, ctx); }
    if let Some(t) = b.tail.as_mut() { lift_expr(t, ctx); }
}

fn lift_stmt(s: &mut Stmt, ctx: &mut LiftCtx) {
    match s {
        Stmt::Let { value: Some(e), .. } => lift_expr(e, ctx),
        Stmt::LetTuple { value, .. } => lift_expr(value, ctx),
        Stmt::Expr(e) => lift_expr(e, ctx),
        Stmt::Return(Some(e)) => lift_expr(e, ctx),
        _ => {}
    }
}

fn lift_expr(e: &mut Expr, ctx: &mut LiftCtx) {
    // Recurse into children FIRST so nested closures get lifted bottom-up.
    match e {
        Expr::Closure { .. } => {} // handled below
        Expr::Call { callee, args } => {
            lift_expr(callee, ctx);
            for a in args.iter_mut() { lift_expr(a, ctx); }
        }
        Expr::MethodCall { recv, args, .. } => {
            lift_expr(recv, ctx);
            for a in args.iter_mut() { lift_expr(a, ctx); }
        }
        Expr::Field { recv, .. } => lift_expr(recv, ctx),
        Expr::Bin { lhs, rhs, .. } => { lift_expr(lhs, ctx); lift_expr(rhs, ctx); }
        Expr::Unary { expr, .. } => lift_expr(expr, ctx),
        Expr::Block(b) => lift_block(b, ctx),
        Expr::If { cond, then, else_ } => {
            lift_expr(cond, ctx);
            lift_block(then, ctx);
            if let Some(b) = else_ { lift_block(b, ctx); }
        }
        Expr::For { iter, body, .. } => { lift_expr(iter, ctx); lift_block(body, ctx); }
        Expr::While { cond, body } => { lift_expr(cond, ctx); lift_block(body, ctx); }
        Expr::Range { lo, hi, step } => {
            lift_expr(lo, ctx); lift_expr(hi, ctx);
            if let Some(s) = step { lift_expr(s, ctx); }
        }
        Expr::Ref { expr, .. } => lift_expr(expr, ctx),
        Expr::Region { body, .. } => lift_block(body, ctx),
        Expr::StructLit { fields, .. } => for (_, fv) in fields { lift_expr(fv, ctx); },
        Expr::Match { scrutinee, arms } => {
            lift_expr(scrutinee, ctx);
            for (_, arm) in arms { lift_expr(arm, ctx); }
        }
        Expr::Cast { expr, .. } => lift_expr(expr, ctx),
        Expr::Try(inner) => lift_expr(inner, ctx),
        Expr::Index { recv, idx } => { lift_expr(recv, ctx); lift_expr(idx, ctx); }
        Expr::Tuple(elems) => for el in elems { lift_expr(el, ctx); },
        _ => {}
    }
    // Now lift THIS node if it's a Closure.
    if let Expr::Closure { params, body } = e {
        let name = format!("__closure_{}", ctx.counter);
        ctx.counter += 1;
        let fn_params: Vec<Param> = params.iter().map(|(n, ty)| Param {
            name: n.clone(),
            ty: ty.clone().unwrap_or(Ty::Named("i64".into())),
        }).collect();
        // Wrap body in a return-tail Block so it lowers as `fn { tail }`.
        let body_block = Block { stmts: Vec::new(), tail: Some(Box::new((**body).clone())) };
        let lifted = FnDecl {
            attrs: Vec::new(),
            is_pub: false,
            is_extern: false,
            name: name.clone(),
            const_params: Vec::new(),
            params: fn_params,
            ret: Some(Ty::Named("i64".into())),
            body: Some(body_block),
        };
        ctx.lifted.push(lifted);
        *e = Expr::Ident(name);
    }
}
