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
use std::collections::{HashMap, HashSet};

pub fn run(prog: &mut Program) -> usize {
    let mut globals: HashSet<String> = HashSet::new();
    // name -> which param positions are declared `Closure` (object ABI). A
    // closure value must become a heap object ONLY when it flows into one of
    // these positions; a closure passed to an `i64` param stays a bare fn
    // pointer (the direct `op(x, y)` convention, no env arg).
    let mut sigs: HashMap<String, Vec<bool>> = HashMap::new();
    let mut note_sig = |f: &FnDecl| {
        sigs.insert(f.name.clone(), f.params.iter().map(|p| is_closure_ty(&p.ty)).collect());
    };
    for item in &prog.items {
        match item {
            Item::Fn(f) => { globals.insert(f.name.clone()); note_sig(f); }
            Item::Impl { methods, .. } | Item::ImplTrait { methods, .. } => {
                for m in methods { globals.insert(m.name.clone()); note_sig(m); }
            }
            _ => {}
        }
    }

    let mut ctx = Ctx { lifted: Vec::new(), counter: 0, globals, sigs };
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
    sigs: HashMap<String, Vec<bool>>,
}

const ENV_PARAM: &str = "__cloenv";

fn is_closure_ty(ty: &Ty) -> bool {
    matches!(ty, Ty::Named(n) if n == "Closure")
}

fn process_fn(f: &mut FnDecl, ctx: &mut Ctx) {
    let Some(body) = f.body.as_mut() else { return; };

    // Phase 0 — hoist an INLINE closure that sits in a `Closure`-typed arg
    // position (`apply(|x| .., 5)`) into a synthetic `let __inlineclo_K = |..|`
    // immediately before the enclosing statement. From there it is an ordinary
    // value-used named closure, so Phases A/B lower it like any other.
    hoist_inline_block(body, &ctx.sigs, &mut ctx.counter);

    // Phase 0.5 — a closure that ESCAPES upward as the fn's return value
    // (`fn make_adder(n) -> Closure { |x| x + n }`) is lowered in place to a
    // heap-object pointer, so the caller (which holds a `Closure`) can invoke it
    // through the object ABI with the captures intact.
    if f.ret.as_ref().map_or(false, is_closure_ty) {
        lower_escaping_in_fn(body, ctx);
    }

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
    collect_arg_idents_block(body, &ctx.sigs, &mut arg_used);

    // Phase A — lower capturing closures bound to a value-used name into
    // heap-object construction; record each as a call target.
    lower_block(body, ctx, &arg_used, &mut targets);

    // Phase B — rewrite calls through any target (closure params + objects).
    if !targets.is_empty() {
        rewrite_calls_block(body, &targets, &mut ctx.counter);
    }
}

// ─── Phase 0: inline-closure hoisting ───────────────────────────────────────

/// Replace `f(.., |p| body, ..)` (closure in a `Closure`-typed arg slot) with a
/// hoisted `let __inlineclo_K = |p| body;` before the statement and the ident
/// `__inlineclo_K` in the call. Nested blocks (if/for/while/region bodies) and
/// match arms are handled in their own scope so a hoisted binding never escapes
/// past a payload pattern bind.
fn hoist_inline_block(b: &mut Block, sigs: &HashMap<String, Vec<bool>>, ctr: &mut usize) {
    let mut out: Vec<Stmt> = Vec::with_capacity(b.stmts.len());
    for mut s in std::mem::take(&mut b.stmts) {
        let mut hoisted: Vec<Stmt> = Vec::new();
        match &mut s {
            Stmt::Let { value: Some(e), .. } => hoist_inline_expr(e, sigs, &mut hoisted, ctr),
            Stmt::LetTuple { value, .. } => hoist_inline_expr(value, sigs, &mut hoisted, ctr),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => hoist_inline_expr(e, sigs, &mut hoisted, ctr),
            _ => {}
        }
        out.extend(hoisted);
        out.push(s);
    }
    if let Some(t) = b.tail.as_mut() {
        let mut hoisted: Vec<Stmt> = Vec::new();
        hoist_inline_expr(t, sigs, &mut hoisted, ctr);
        out.extend(hoisted);
    }
    b.stmts = out;
}

fn make_inline_let(clo: Expr, ctr: &mut usize) -> (String, Stmt) {
    let name = format!("__inlineclo_{}", *ctr);
    *ctr += 1;
    let st = Stmt::Let {
        name: name.clone(),
        mutable: false,
        ty: Some(Ty::Named("Closure".into())),
        value: Some(clo),
    };
    (name, st)
}

fn hoist_inline_expr(e: &mut Expr, sigs: &HashMap<String, Vec<bool>>, hoisted: &mut Vec<Stmt>, ctr: &mut usize) {
    match e {
        Expr::Call { callee, args } => {
            hoist_inline_expr(callee, sigs, hoisted, ctr);
            let clo_pos: Option<Vec<bool>> = match callee.as_ref() {
                Expr::Ident(f) => sigs.get(f).cloned(),
                _ => None,
            };
            for (i, a) in args.iter_mut().enumerate() {
                let is_clo_slot = clo_pos.as_ref().map_or(false, |v| v.get(i).copied().unwrap_or(false));
                if is_clo_slot && matches!(a, Expr::Closure { .. }) {
                    let clo = std::mem::replace(a, Expr::IntLit(0));
                    let (name, st) = make_inline_let(clo, ctr);
                    hoisted.push(st);
                    *a = Expr::Ident(name);
                } else {
                    hoist_inline_expr(a, sigs, hoisted, ctr);
                }
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            hoist_inline_expr(recv, sigs, hoisted, ctr);
            for a in args { hoist_inline_expr(a, sigs, hoisted, ctr); }
        }
        Expr::Bin { lhs, rhs, .. } => { hoist_inline_expr(lhs, sigs, hoisted, ctr); hoist_inline_expr(rhs, sigs, hoisted, ctr); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => hoist_inline_expr(expr, sigs, hoisted, ctr),
        Expr::Field { recv, .. } => hoist_inline_expr(recv, sigs, hoisted, ctr),
        Expr::Index { recv, idx } => { hoist_inline_expr(recv, sigs, hoisted, ctr); hoist_inline_expr(idx, sigs, hoisted, ctr); }
        Expr::StructLit { fields, .. } => for (_, fv) in fields { hoist_inline_expr(fv, sigs, hoisted, ctr); },
        Expr::Tuple(elems) => for el in elems { hoist_inline_expr(el, sigs, hoisted, ctr); },
        Expr::Range { lo, hi, step } => {
            hoist_inline_expr(lo, sigs, hoisted, ctr); hoist_inline_expr(hi, sigs, hoisted, ctr);
            if let Some(s) = step { hoist_inline_expr(s, sigs, hoisted, ctr); }
        }
        // Block-introducing sub-exprs recurse in their OWN scope.
        Expr::Block(b) => hoist_inline_block(b, sigs, ctr),
        Expr::If { cond, then, else_ } => {
            hoist_inline_expr(cond, sigs, hoisted, ctr);
            hoist_inline_block(then, sigs, ctr);
            if let Some(b) = else_ { hoist_inline_block(b, sigs, ctr); }
        }
        Expr::For { iter, body, .. } => { hoist_inline_expr(iter, sigs, hoisted, ctr); hoist_inline_block(body, sigs, ctr); }
        Expr::While { cond, body } => { hoist_inline_expr(cond, sigs, hoisted, ctr); hoist_inline_block(body, sigs, ctr); }
        Expr::Region { body, .. } => hoist_inline_block(body, sigs, ctr),
        Expr::Match { scrutinee, arms } => {
            hoist_inline_expr(scrutinee, sigs, hoisted, ctr);
            for (_, arm) in arms {
                // Each arm is its own scope (may bind a payload). Hoist within
                // the arm, wrapping it in a block if anything was hoisted.
                let mut arm_hoist: Vec<Stmt> = Vec::new();
                hoist_inline_expr(arm, sigs, &mut arm_hoist, ctr);
                if !arm_hoist.is_empty() {
                    let inner = std::mem::replace(arm, Expr::IntLit(0));
                    *arm = Expr::Block(Block { stmts: arm_hoist, tail: Some(Box::new(inner)) });
                }
            }
        }
        _ => {}
    }
}

// ─── shared closure-object construction ─────────────────────────────────────

/// Build the lifted `__cloobj_K(__cloenv, <params>)` fn (captures read back from
/// the env) and return the statements that allocate the heap object bound to
/// `name` and store `[fn_ptr | cap0 | cap1 | ...]`. NON-capturing closures get a
/// 1-word `[fn_ptr]` object so the object call ABI stays uniform.
fn construct_object(name: &str, params: &[(String, Option<Ty>)], body: &Expr, ctx: &mut Ctx) -> Vec<Stmt> {
    let param_names: HashSet<String> = params.iter().map(|(n, _)| n.clone()).collect();
    let mut caps: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut bound = param_names.clone();
    collect_free_vars(body, &mut bound, &ctx.globals, &mut caps, &mut seen);

    let lifted_name = format!("__cloobj_{}", ctx.counter);
    ctx.counter += 1;
    let mut lifted_body = body.clone();
    let cap_idx: HashMap<String, usize> =
        caps.iter().cloned().enumerate().map(|(i, n)| (n, i)).collect();
    rewrite_captures(&mut lifted_body, &cap_idx);
    let mut fn_params: Vec<Param> = Vec::with_capacity(1 + params.len());
    fn_params.push(Param { name: ENV_PARAM.into(), ty: Ty::Named("i64".into()) });
    for (n, t) in params {
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

    let words = 1 + caps.len();
    let mut out: Vec<Stmt> = Vec::with_capacity(2 + caps.len());
    out.push(Stmt::Let {
        name: name.to_string(),
        mutable: false,
        ty: None,
        value: Some(call("aether_alloc_bytes", vec![Expr::IntLit((words * 8) as i64)])),
    });
    out.push(Stmt::Expr(call("aether_store_i64", vec![
        Expr::Ident(name.to_string()),
        Expr::IntLit(0),
        Expr::Ident(lifted_name),
    ])));
    for (i, cap) in caps.iter().enumerate() {
        out.push(Stmt::Expr(call("aether_store_i64", vec![
            Expr::Ident(name.to_string()),
            Expr::IntLit((1 + i) as i64),
            Expr::Ident(cap.clone()),
        ])));
    }
    out
}

/// Lower an ESCAPING closure expression (return / tail / Closure-typed binding)
/// in place: replace it with `{ <construction>; __escclo_K }`, yielding the
/// heap-object pointer. Used where the closure is not bound to a user name.
fn lower_escaping_closure(e: &mut Expr, ctx: &mut Ctx) {
    let Expr::Closure { params, body } = e else { return; };
    let params = std::mem::take(params);
    let body = std::mem::replace(body.as_mut(), Expr::IntLit(0));
    let objname = format!("__escclo_{}", ctx.counter);
    ctx.counter += 1;
    let stmts = construct_object(&objname, &params, &body, ctx);
    *e = Expr::Block(Block { stmts, tail: Some(Box::new(Expr::Ident(objname))) });
}

/// Lower closures that escape upward out of `fn` (return type `Closure`): the
/// body tail and every `return <closure>`. Tail recursion descends through
/// `if`/`match`/`block` so each branch tail that is a closure is lowered.
fn lower_escaping_in_fn(body: &mut Block, ctx: &mut Ctx) {
    lower_escaping_in_block_tail(body, ctx);
    lower_escaping_returns_block(body, ctx);
}

fn lower_escaping_in_block_tail(b: &mut Block, ctx: &mut Ctx) {
    if let Some(t) = b.tail.as_mut() {
        lower_escaping_tail_expr(t, ctx);
    }
}

fn lower_escaping_tail_expr(e: &mut Expr, ctx: &mut Ctx) {
    match e {
        Expr::Closure { .. } => lower_escaping_closure(e, ctx),
        Expr::Block(b) => lower_escaping_in_block_tail(b, ctx),
        Expr::If { then, else_, .. } => {
            lower_escaping_in_block_tail(then, ctx);
            if let Some(b) = else_ { lower_escaping_in_block_tail(b, ctx); }
        }
        Expr::Match { arms, .. } => {
            for (_, arm) in arms { lower_escaping_tail_expr(arm, ctx); }
        }
        _ => {}
    }
}

fn lower_escaping_returns_block(b: &mut Block, ctx: &mut Ctx) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Return(Some(e)) => lower_escaping_tail_expr(e, ctx),
            Stmt::Let { value: Some(e), .. } => lower_escaping_returns_in_expr(e, ctx),
            Stmt::LetTuple { value, .. } => lower_escaping_returns_in_expr(value, ctx),
            Stmt::Expr(e) => lower_escaping_returns_in_expr(e, ctx),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { lower_escaping_returns_in_expr(t, ctx); }
}

// Walk into nested blocks (if/for/while/region/match/block bodies) to lower any
// `return <closure>` they contain (a return exits the whole fn).
fn lower_escaping_returns_in_expr(e: &mut Expr, ctx: &mut Ctx) {
    match e {
        Expr::Block(b) => lower_escaping_returns_block(b, ctx),
        Expr::If { cond, then, else_ } => {
            lower_escaping_returns_in_expr(cond, ctx);
            lower_escaping_returns_block(then, ctx);
            if let Some(b) = else_ { lower_escaping_returns_block(b, ctx); }
        }
        Expr::For { body, .. } => lower_escaping_returns_block(body, ctx),
        Expr::While { body, .. } => lower_escaping_returns_block(body, ctx),
        Expr::Region { body, .. } => lower_escaping_returns_block(body, ctx),
        Expr::Match { arms, .. } => for (_, arm) in arms { lower_escaping_returns_in_expr(arm, ctx); },
        _ => {}
    }
}

// ─── Phase A: construction lowering ─────────────────────────────────────────

fn lower_block(b: &mut Block, ctx: &mut Ctx, arg_used: &HashSet<String>, targets: &mut HashSet<String>) {
    let mut out: Vec<Stmt> = Vec::with_capacity(b.stmts.len());
    for stmt in std::mem::take(&mut b.stmts) {
        match stmt {
            Stmt::Let { name, ty, value: Some(Expr::Closure { params, body }), .. }
                // A closure value: either passed as an arg (`arg_used`) or bound
                // to an explicitly `Closure`-typed local. Both need the object
                // representation + object call ABI through `name`.
                if arg_used.contains(&name) || ty.as_ref().map_or(false, is_closure_ty) =>
            {
                out.extend(construct_object(&name, &params, &body, ctx));
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
        Expr::Match { scrutinee, arms } => {
            descend_expr(scrutinee, ctx, arg_used, targets);
            for (_, arm) in arms { descend_expr(arm, ctx, arg_used, targets); }
        }
        Expr::StructLit { fields, .. } => for (_, fv) in fields { descend_expr(fv, ctx, arg_used, targets); },
        Expr::Tuple(elems) => for el in elems { descend_expr(el, ctx, arg_used, targets); },
        Expr::Range { lo, hi, step } => {
            descend_expr(lo, ctx, arg_used, targets); descend_expr(hi, ctx, arg_used, targets);
            if let Some(s) = step { descend_expr(s, ctx, arg_used, targets); }
        }
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
        Expr::Match { scrutinee, arms } => {
            rewrite_calls(scrutinee, targets, ctr);
            for (_, arm) in arms { rewrite_calls(arm, targets, ctr); }
        }
        Expr::StructLit { fields, .. } => for (_, fv) in fields { rewrite_calls(fv, targets, ctr); },
        Expr::Tuple(elems) => for e in elems { rewrite_calls(e, targets, ctr); },
        Expr::Range { lo, hi, step } => {
            rewrite_calls(lo, targets, ctr); rewrite_calls(hi, targets, ctr);
            if let Some(s) = step { rewrite_calls(s, targets, ctr); }
        }
        _ => {}
    }
}

// ─── helpers ────────────────────────────────────────────────────────────────

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call { callee: Box::new(Expr::Ident(name.into())), args }
}

/// Collect idents passed into a `Closure`-typed parameter position anywhere in
/// the block. ONLY such idents need the heap-object representation — a closure
/// passed to an `i64` param keeps the bare-fn-pointer convention. `sigs` maps a
/// fn name to which of its param positions are declared `Closure`.
fn collect_arg_idents_block(b: &Block, sigs: &HashMap<String, Vec<bool>>, out: &mut HashSet<String>) {
    for s in &b.stmts { collect_arg_idents_stmt(s, sigs, out); }
    if let Some(t) = &b.tail { collect_arg_idents_expr(t, sigs, out); }
}
fn collect_arg_idents_stmt(s: &Stmt, sigs: &HashMap<String, Vec<bool>>, out: &mut HashSet<String>) {
    match s {
        Stmt::Let { value: Some(e), .. } => collect_arg_idents_expr(e, sigs, out),
        Stmt::LetTuple { value, .. } => collect_arg_idents_expr(value, sigs, out),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => collect_arg_idents_expr(e, sigs, out),
        _ => {}
    }
}
fn collect_arg_idents_expr(e: &Expr, sigs: &HashMap<String, Vec<bool>>, out: &mut HashSet<String>) {
    match e {
        Expr::Call { callee, args } => {
            collect_arg_idents_expr(callee, sigs, out);
            // Resolve the callee's Closure-typed positions (if known).
            let clo_pos: Option<&Vec<bool>> = match callee.as_ref() {
                Expr::Ident(fname) => sigs.get(fname),
                _ => None,
            };
            for (i, a) in args.iter().enumerate() {
                if let Expr::Ident(n) = a {
                    let is_clo = clo_pos.map_or(false, |v| v.get(i).copied().unwrap_or(false));
                    if is_clo { out.insert(n.clone()); }
                }
                collect_arg_idents_expr(a, sigs, out);
            }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_arg_idents_expr(recv, sigs, out);
            // Method signatures aren't position-resolved here; recurse only.
            for a in args { collect_arg_idents_expr(a, sigs, out); }
        }
        Expr::Bin { lhs, rhs, .. } => { collect_arg_idents_expr(lhs, sigs, out); collect_arg_idents_expr(rhs, sigs, out); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => collect_arg_idents_expr(expr, sigs, out),
        Expr::Field { recv, .. } => collect_arg_idents_expr(recv, sigs, out),
        Expr::Index { recv, idx } => { collect_arg_idents_expr(recv, sigs, out); collect_arg_idents_expr(idx, sigs, out); }
        Expr::Block(b) => collect_arg_idents_block(b, sigs, out),
        Expr::If { cond, then, else_ } => {
            collect_arg_idents_expr(cond, sigs, out);
            collect_arg_idents_block(then, sigs, out);
            if let Some(b) = else_ { collect_arg_idents_block(b, sigs, out); }
        }
        Expr::For { iter, body, .. } => { collect_arg_idents_expr(iter, sigs, out); collect_arg_idents_block(body, sigs, out); }
        Expr::While { cond, body } => { collect_arg_idents_expr(cond, sigs, out); collect_arg_idents_block(body, sigs, out); }
        Expr::Region { body, .. } => collect_arg_idents_block(body, sigs, out),
        Expr::Match { scrutinee, arms } => {
            collect_arg_idents_expr(scrutinee, sigs, out);
            for (_, arm) in arms { collect_arg_idents_expr(arm, sigs, out); }
        }
        Expr::StructLit { fields, .. } => for (_, fv) in fields { collect_arg_idents_expr(fv, sigs, out); },
        Expr::Tuple(elems) => for el in elems { collect_arg_idents_expr(el, sigs, out); },
        Expr::Range { lo, hi, step } => {
            collect_arg_idents_expr(lo, sigs, out); collect_arg_idents_expr(hi, sigs, out);
            if let Some(s) = step { collect_arg_idents_expr(s, sigs, out); }
        }
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
        Expr::Region { body, .. } => collect_free_vars_block(body, bound, globals, caps, seen),
        Expr::Range { lo, hi, step } => {
            collect_free_vars(lo, bound, globals, caps, seen); collect_free_vars(hi, bound, globals, caps, seen);
            if let Some(s) = step { collect_free_vars(s, bound, globals, caps, seen); }
        }
        Expr::Match { scrutinee, arms } => {
            collect_free_vars(scrutinee, bound, globals, caps, seen);
            for (pat, arm) in arms {
                // A payload-binding arm introduces a fresh local that shadows
                // any outer name, so it is NOT a free var inside that arm.
                let mut inner = bound.clone();
                if let crate::ast::MatchPat::EnumVariantBind(_, binds) = pat {
                    for b in binds { inner.insert(b.clone()); }
                }
                collect_free_vars(arm, &mut inner, globals, caps, seen);
            }
        }
        Expr::StructLit { fields, .. } => for (_, fv) in fields { collect_free_vars(fv, bound, globals, caps, seen); },
        Expr::Tuple(elems) => for el in elems { collect_free_vars(el, bound, globals, caps, seen); },
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
