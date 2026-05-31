//! Phase 6.6 — closure objects for capturing closures passed *as values*.
//!
//! The closure-lifting pass (`mir::closures`) handles a capturing closure that
//! is only ever called directly by name (it prepends the captures at the call
//! site). The case it explicitly punts on is a capturing closure passed to
//! another fn — `let inc = |x| acc + x; apply(inc, 5)` — because `apply` gets
//! a bare code pointer with the captures lost.
//!
//! This pass closes that gap with a heap **closure object**: a block laid out
//! as `[fn_ptr | cap0 | cap1 | ...]` (one i64 word each). It runs BEFORE
//! `mir::closures` and fully lowers the closures it touches, so the lifting
//! pass never sees them.
//!
//! Two rewrites, both at the AST level (zero asm-backend changes):
//!
//! 1. **Construction.** A `let NAME = |params| body` whose body captures ≥1
//!    enclosing local AND whose NAME appears in call-argument position becomes:
//!    ```text
//!    let NAME = aether_alloc_bytes(8 * (1 + ncaps));
//!    aether_store_i64(NAME, 0, __cloobj_K);   // code pointer (Ident -> leaq)
//!    aether_store_i64(NAME, 1, cap0);         // captured by value at creation
//!    ...
//!    ```
//!    The lifted fn `__cloobj_K(__cloenv, <params>)` reads each capture back
//!    from the env via `aether_load_i64(__cloenv, 1 + idx)`.
//!
//! 2. **Invocation.** A call `T(args)` where `T` is a closure-object local or a
//!    parameter declared `Closure` becomes:
//!    ```text
//!    { let __clofp_K = aether_load_i64(T, 0); __clofp_K(T, args...) }
//!    ```
//!    `__clofp_K` is a plain i64 local holding the code pointer, so the call
//!    goes through the asm backend's existing indirect-call path (`callq
//!    *%r10`) with the env pointer `T` prepended in the first arg register.
//!
//! Scope (honest): by-value captures of a let-bound closure passed as a value;
//! the receiver advertises the param type `Closure`. Mutable-capture-as-value
//! and inline `apply(|x| .., 5)` closures are follow-ups.

use crate::ast::{Block, Expr, FnDecl, Item, Param, Program, Stmt, Ty};
use std::collections::HashSet;

pub fn run(prog: &mut Program) -> usize {
    let mut globals: HashSet<String> = HashSet::new();
    for item in &prog.items {
        match item {
            Item::Fn(f) => { globals.insert(f.name.clone()); }
            Item::Impl { methods, .. } | Item::ImplTrait { methods, .. } => {
                for m in methods { globals.insert(m.name.clone()); }
            }
            _ => {}
        }
    }

    let mut ctx = Ctx { lifted: Vec::new(), counter: 0, globals };
    // Process top-level fns + impl methods. `ctx` is disjoint from `prog`, so
    // a plain `iter_mut` is sound; the lifted fns are appended after the loop.
    for item in prog.items.iter_mut() {
        match item {
            Item::Fn(f) => process_fn(f, &mut ctx),
            Item::Impl { methods, .. } | Item::ImplTrait { methods, .. } => {
                for m in methods.iter_mut() { process_fn(m, &mut ctx); }
            }
            _ => {}
        }
    }

    let n = ctx.lifted.len();
    for f in ctx.lifted { prog.items.push(Item::Fn(f)); }
    n
}

struct Ctx {
    lifted: Vec<FnDecl>,
    counter: usize,
    globals: HashSet<String>,
}

const ENV_PARAM: &str = "__cloenv";

fn is_closure_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Named(n) if n == "Closure")
}

fn process_fn(f: &mut FnDecl, ctx: &mut Ctx) {
    let Some(body) = f.body.as_mut() else { return; };

    // Targets whose calls use the closure-object ABI: params typed `Closure`,
    // plus any closure-object local we create below.
    let mut targets: HashSet<String> = f.params.iter()
        .filter(|p| is_closure_ty(&p.ty))
        .map(|p| p.name.clone())
        .collect();

    // Pre-scan the ORIGINAL body for names used in call-argument position —
    // that's what makes a closure "passed as a value" rather than only
    // direct-called. Done before construction lowering, which would otherwise
    // add synthetic `aether_store_i64(NAME, ...)` arg uses.
    let mut arg_used: HashSet<String> = HashSet::new();
    collect_arg_idents_block(body, &mut arg_used);

    // Phase A — lower capturing closures bound to a value-used name into
    // heap-object construction; record each as a call target.
    lower_block(body, ctx, &arg_used, &mut targets);

    // Phase B — rewrite calls through any target (closure params + objects).
    if !targets.is_empty() {
        rewrite_calls_block(body, &targets, &mut ctx.counter);
    }
}

// ─── Phase A: construction lowering ─────────────────────────────────────────

fn lower_block(b: &mut Block, ctx: &mut Ctx, arg_used: &HashSet<String>, targets: &mut HashSet<String>) {
    let mut out: Vec<Stmt> = Vec::with_capacity(b.stmts.len());
    for stmt in std::mem::take(&mut b.stmts) {
        match stmt {
            Stmt::Let { name, mutable, ty, value: Some(Expr::Closure { params, body }) }
                if arg_used.contains(&name) =>
            {
                let param_names: HashSet<String> =
                    params.iter().map(|(n, _)| n.clone()).collect();
                let mut caps: Vec<String> = Vec::new();
                let mut seen: HashSet<String> = HashSet::new();
                let mut bound = param_names.clone();
                collect_free_vars(&body, &mut bound, &ctx.globals, &mut caps, &mut seen);

                if caps.is_empty() {
                    // Non-capturing closure used as a value: leave it for
                    // mir::closures (lifts to a bare fn pointer, which the
                    // i64 indirect-call path already handles).
                    out.push(Stmt::Let { name, mutable, ty, value: Some(Expr::Closure { params, body }) });
                    continue;
                }

                // Build the lifted fn: env param first, captures read from env.
                let lifted_name = format!("__cloobj_{}", ctx.counter);
                ctx.counter += 1;
                let mut lifted_body = (*body).clone();
                let cap_idx: std::collections::HashMap<String, usize> =
                    caps.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();
                rewrite_captures(&mut lifted_body, &cap_idx);
                let mut fn_params: Vec<Param> = Vec::with_capacity(1 + params.len());
                fn_params.push(Param { name: ENV_PARAM.into(), ty: Ty::Named("i64".into()) });
                for (n, t) in &params {
                    fn_params.push(Param { name: n.clone(), ty: t.clone().unwrap_or(Ty::Named("i64".into())) });
                }
                ctx.lifted.push(FnDecl {
                    attrs: Vec::new(),
                    is_pub: false,
                    is_extern: false,
                    name: lifted_name.clone(),
                    const_params: Vec::new(),
                    params: fn_params,
                    ret: Some(Ty::Named("i64".into())),
                    body: Some(Block { stmts: Vec::new(), tail: Some(Box::new(lifted_body)) }),
                });

                // Emit object construction: alloc, store fn ptr, store captures.
                let words = 1 + caps.len();
                out.push(Stmt::Let {
                    name: name.clone(),
                    mutable: false,
                    ty: None,
                    value: Some(call("aether_alloc_bytes", vec![Expr::IntLit((words * 8) as i64)])),
                });
                out.push(Stmt::Expr(call("aether_store_i64", vec![
                    Expr::Ident(name.clone()),
                    Expr::IntLit(0),
                    Expr::Ident(lifted_name.clone()),
                ])));
                for (i, cap) in caps.iter().enumerate() {
                    out.push(Stmt::Expr(call("aether_store_i64", vec![
                        Expr::Ident(name.clone()),
                        Expr::IntLit((1 + i) as i64),
                        Expr::Ident(cap.clone()),
                    ])));
                }
                targets.insert(name);
            }
            mut other => {
                // Recurse into nested blocks so closures bound inside
                // if/for/while/block bodies are lowered too.
                descend_stmt(&mut other, ctx, arg_used, targets);
                out.push(other);
            }
        }
    }
    b.stmts = out;
    if let Some(t) = b.tail.as_mut() { descend_expr(t, ctx, arg_used, targets); }
}

fn descend_stmt(s: &mut Stmt, ctx: &mut Ctx, arg_used: &HashSet<String>, targets: &mut HashSet<String>) {
    match s {
        Stmt::Let { value: Some(e), .. } => descend_expr(e, ctx, arg_used, targets),
        Stmt::LetTuple { value, .. } => descend_expr(value, ctx, arg_used, targets),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => descend_expr(e, ctx, arg_used, targets),
        _ => {}
    }
}

fn descend_expr(e: &mut Expr, ctx: &mut Ctx, arg_used: &HashSet<String>, targets: &mut HashSet<String>) {
    match e {
        Expr::Block(b) => lower_block(b, ctx, arg_used, targets),
        Expr::If { cond, then, else_ } => {
            descend_expr(cond, ctx, arg_used, targets);
            lower_block(then, ctx, arg_used, targets);
            if let Some(b) = else_ { lower_block(b, ctx, arg_used, targets); }
        }
        Expr::For { iter, body, .. } => { descend_expr(iter, ctx, arg_used, targets); lower_block(body, ctx, arg_used, targets); }
        Expr::While { cond, body } => { descend_expr(cond, ctx, arg_used, targets); lower_block(body, ctx, arg_used, targets); }
        Expr::Region { body, .. } => lower_block(body, ctx, arg_used, targets),
        Expr::Call { callee, args } => { descend_expr(callee, ctx, arg_used, targets); for a in args { descend_expr(a, ctx, arg_used, targets); } }
        Expr::MethodCall { recv, args, .. } => { descend_expr(recv, ctx, arg_used, targets); for a in args { descend_expr(a, ctx, arg_used, targets); } }
        Expr::Bin { lhs, rhs, .. } => { descend_expr(lhs, ctx, arg_used, targets); descend_expr(rhs, ctx, arg_used, targets); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => descend_expr(expr, ctx, arg_used, targets),
        Expr::Field { recv, .. } => descend_expr(recv, ctx, arg_used, targets),
        Expr::Index { recv, idx } => { descend_expr(recv, ctx, arg_used, targets); descend_expr(idx, ctx, arg_used, targets); }
        _ => {}
    }
}

// ─── Phase B: call rewriting ────────────────────────────────────────────────

fn rewrite_calls_block(b: &mut Block, targets: &HashSet<String>, ctr: &mut usize) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { value: Some(e), .. } => rewrite_calls(e, targets, ctr),
            Stmt::LetTuple { value, .. } => rewrite_calls(value, targets, ctr),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => rewrite_calls(e, targets, ctr),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { rewrite_calls(t, targets, ctr); }
}

fn rewrite_calls(e: &mut Expr, targets: &HashSet<String>, ctr: &mut usize) {
    // A call whose callee is a target name → closure-object invocation.
    if let Expr::Call { callee, args } = e {
        if let Expr::Ident(t) = callee.as_ref() {
            if targets.contains(t) {
                let t = t.clone();
                // Rewrite nested calls in the args first.
                for a in args.iter_mut() { rewrite_calls(a, targets, ctr); }
                let fp = format!("__clofp_{}", *ctr);
                *ctr += 1;
                let mut new_args: Vec<Expr> = Vec::with_capacity(1 + args.len());
                new_args.push(Expr::Ident(t.clone()));      // env pointer
                new_args.extend(args.drain(..));
                let block = Block {
                    stmts: vec![Stmt::Let {
                        name: fp.clone(),
                        mutable: false,
                        ty: None,
                        value: Some(call("aether_load_i64", vec![Expr::Ident(t), Expr::IntLit(0)])),
                    }],
                    tail: Some(Box::new(Expr::Call {
                        callee: Box::new(Expr::Ident(fp)),
                        args: new_args,
                    })),
                };
                *e = Expr::Block(block);
                return;
            }
        }
    }
    // Otherwise recurse.
    match e {
        Expr::Call { callee, args } => { rewrite_calls(callee, targets, ctr); for a in args { rewrite_calls(a, targets, ctr); } }
        Expr::MethodCall { recv, args, .. } => { rewrite_calls(recv, targets, ctr); for a in args { rewrite_calls(a, targets, ctr); } }
        Expr::Bin { lhs, rhs, .. } => { rewrite_calls(lhs, targets, ctr); rewrite_calls(rhs, targets, ctr); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => rewrite_calls(expr, targets, ctr),
        Expr::Field { recv, .. } => rewrite_calls(recv, targets, ctr),
        Expr::Index { recv, idx } => { rewrite_calls(recv, targets, ctr); rewrite_calls(idx, targets, ctr); }
        Expr::Block(b) => rewrite_calls_block(b, targets, ctr),
        Expr::If { cond, then, else_ } => {
            rewrite_calls(cond, targets, ctr);
            rewrite_calls_block(then, targets, ctr);
            if let Some(b) = else_ { rewrite_calls_block(b, targets, ctr); }
        }
        Expr::For { iter, body, .. } => { rewrite_calls(iter, targets, ctr); rewrite_calls_block(body, targets, ctr); }
        Expr::While { cond, body } => { rewrite_calls(cond, targets, ctr); rewrite_calls_block(body, targets, ctr); }
        Expr::Region { body, .. } => rewrite_calls_block(body, targets, ctr),
        _ => {}
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call { callee: Box::new(Expr::Ident(name.into())), args }
}

/// Collect idents that appear in call-argument position anywhere in the block.
fn collect_arg_idents_block(b: &Block, out: &mut HashSet<String>) {
    for s in &b.stmts { collect_arg_idents_stmt(s, out); }
    if let Some(t) = &b.tail { collect_arg_idents_expr(t, out); }
}
fn collect_arg_idents_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Let { value: Some(e), .. } => collect_arg_idents_expr(e, out),
        Stmt::LetTuple { value, .. } => collect_arg_idents_expr(value, out),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => collect_arg_idents_expr(e, out),
        _ => {}
    }
}
fn collect_arg_idents_expr(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Call { callee, args } => {
            collect_arg_idents_expr(callee, out);
            for a in args {
                if let Expr::Ident(n) = a { out.insert(n.clone()); }
                collect_arg_idents_expr(a, out);
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_arg_idents_expr(recv, out);
            for a in args {
                if let Expr::Ident(n) = a { out.insert(n.clone()); }
                collect_arg_idents_expr(a, out);
            }
        }
        Expr::Bin { lhs, rhs, .. } => { collect_arg_idents_expr(lhs, out); collect_arg_idents_expr(rhs, out); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => collect_arg_idents_expr(expr, out),
        Expr::Field { recv, .. } => collect_arg_idents_expr(recv, out),
        Expr::Index { recv, idx } => { collect_arg_idents_expr(recv, out); collect_arg_idents_expr(idx, out); }
        Expr::Block(b) => collect_arg_idents_block(b, out),
        Expr::If { cond, then, else_ } => {
            collect_arg_idents_expr(cond, out);
            collect_arg_idents_block(then, out);
            if let Some(b) = else_ { collect_arg_idents_block(b, out); }
        }
        Expr::For { iter, body, .. } => { collect_arg_idents_expr(iter, out); collect_arg_idents_block(body, out); }
        Expr::While { cond, body } => { collect_arg_idents_expr(cond, out); collect_arg_idents_block(body, out); }
        Expr::Region { body, .. } => collect_arg_idents_block(body, out),
        _ => {}
    }
}

/// Collect free variables of a closure body (idents not bound by the closure's
/// params / inner lets / for-vars, and not global fn names), in first-seen
/// order. These become the captures, stored by value at creation time.
fn collect_free_vars(e: &Expr, bound: &mut HashSet<String>, globals: &HashSet<String>, caps: &mut Vec<String>, seen: &mut HashSet<String>) {
    match e {
        Expr::Ident(n) => {
            if !bound.contains(n) && !globals.contains(n) && seen.insert(n.clone()) {
                caps.push(n.clone());
            }
        }
        Expr::Call { callee, args } => {
            // Don't capture the callee name (it resolves as a fn/symbol).
            if !matches!(callee.as_ref(), Expr::Ident(_)) {
                collect_free_vars(callee, bound, globals, caps, seen);
            }
            for a in args { collect_free_vars(a, bound, globals, caps, seen); }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_free_vars(recv, bound, globals, caps, seen);
            for a in args { collect_free_vars(a, bound, globals, caps, seen); }
        }
        Expr::Bin { lhs, rhs, .. } => { collect_free_vars(lhs, bound, globals, caps, seen); collect_free_vars(rhs, bound, globals, caps, seen); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => collect_free_vars(expr, bound, globals, caps, seen),
        Expr::Field { recv, .. } => collect_free_vars(recv, bound, globals, caps, seen),
        Expr::Index { recv, idx } => { collect_free_vars(recv, bound, globals, caps, seen); collect_free_vars(idx, bound, globals, caps, seen); }
        Expr::Block(b) => collect_free_vars_block(b, bound, globals, caps, seen),
        Expr::If { cond, then, else_ } => {
            collect_free_vars(cond, bound, globals, caps, seen);
            collect_free_vars_block(then, bound, globals, caps, seen);
            if let Some(b) = else_ { collect_free_vars_block(b, bound, globals, caps, seen); }
        }
        Expr::For { var, iter, body, .. } => {
            collect_free_vars(iter, bound, globals, caps, seen);
            let mut inner = bound.clone();
            inner.insert(var.clone());
            collect_free_vars_block(body, &mut inner, globals, caps, seen);
        }
        Expr::While { cond, body } => { collect_free_vars(cond, bound, globals, caps, seen); collect_free_vars_block(body, bound, globals, caps, seen); }
        Expr::Range { lo, hi, step } => {
            collect_free_vars(lo, bound, globals, caps, seen); collect_free_vars(hi, bound, globals, caps, seen);
            if let Some(s) = step { collect_free_vars(s, bound, globals, caps, seen); }
        }
        _ => {}
    }
}
fn collect_free_vars_block(b: &Block, bound: &mut HashSet<String>, globals: &HashSet<String>, caps: &mut Vec<String>, seen: &mut HashSet<String>) {
    let mut inner = bound.clone();
    for s in &b.stmts {
        match s {
            Stmt::Let { name, value, .. } => {
                if let Some(e) = value { collect_free_vars(e, &mut inner, globals, caps, seen); }
                inner.insert(name.clone());
            }
            Stmt::LetTuple { names, value } => {
                collect_free_vars(value, &mut inner, globals, caps, seen);
                for n in names { inner.insert(n.clone()); }
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => collect_free_vars(e, &mut inner, globals, caps, seen),
            _ => {}
        }
    }
    if let Some(t) = &b.tail { collect_free_vars(t, &mut inner, globals, caps, seen); }
}

/// Rewrite each capture read in the lifted body to `aether_load_i64(env, 1+idx)`.
fn rewrite_captures(e: &mut Expr, cap_idx: &std::collections::HashMap<String, usize>) {
    if let Expr::Ident(n) = e {
        if let Some(&i) = cap_idx.get(n) {
            *e = call("aether_load_i64", vec![Expr::Ident(ENV_PARAM.into()), Expr::IntLit((1 + i) as i64)]);
            return;
        }
    }
    match e {
        Expr::Call { callee, args } => { rewrite_captures(callee, cap_idx); for a in args { rewrite_captures(a, cap_idx); } }
        Expr::MethodCall { recv, args, .. } => { rewrite_captures(recv, cap_idx); for a in args { rewrite_captures(a, cap_idx); } }
        Expr::Bin { lhs, rhs, .. } => { rewrite_captures(lhs, cap_idx); rewrite_captures(rhs, cap_idx); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => rewrite_captures(expr, cap_idx),
        Expr::Field { recv, .. } => rewrite_captures(recv, cap_idx),
        Expr::Index { recv, idx } => { rewrite_captures(recv, cap_idx); rewrite_captures(idx, cap_idx); }
        Expr::Block(b) => rewrite_captures_block(b, cap_idx),
        Expr::If { cond, then, else_ } => {
            rewrite_captures(cond, cap_idx);
            rewrite_captures_block(then, cap_idx);
            if let Some(b) = else_ { rewrite_captures_block(b, cap_idx); }
        }
        Expr::For { iter, body, .. } => { rewrite_captures(iter, cap_idx); rewrite_captures_block(body, cap_idx); }
        Expr::While { cond, body } => { rewrite_captures(cond, cap_idx); rewrite_captures_block(body, cap_idx); }
        Expr::Range { lo, hi, step } => { rewrite_captures(lo, cap_idx); rewrite_captures(hi, cap_idx); if let Some(s) = step { rewrite_captures(s, cap_idx); } }
        Expr::Region { body, .. } => rewrite_captures_block(body, cap_idx),
        _ => {}
    }
}
fn rewrite_captures_block(b: &mut Block, cap_idx: &std::collections::HashMap<String, usize>) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { value: Some(e), .. } => rewrite_captures(e, cap_idx),
            Stmt::LetTuple { value, .. } => rewrite_captures(value, cap_idx),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => rewrite_captures(e, cap_idx),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { rewrite_captures(t, cap_idx); }
}
