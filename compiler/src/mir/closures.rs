//! Closure-lifting pass.
//!
//! Walks every `Expr::Closure { params, body }` in the program, generates a
//! synthetic top-level fn `__closure_<n>` containing the body verbatim, and
//! rewrites the closure expression in-place to `Expr::Ident("__closure_<n>")`.
//!
//! ## Captures (FR-16.4-extra)
//!
//! Two cases:
//! - **No captures** — the lifted fn has the closure's own params; the
//!   closure expression becomes `Expr::Ident("__closure_<n>")` which loads
//!   as a function pointer (the asm backend's `Expr::Ident` arm emits
//!   `leaq aether_<name>(%rip), %rax` when `<name>` is a known fn). Same
//!   as the original Phase-6 closures.
//! - **With captures** — for each free var `c` in the body that lives in
//!   the enclosing scope:
//!     - if the body assigns to `c`, the lifted fn takes `c: &mut i64` as
//!       an extra param; reads of `c` in the body are rewritten to `*c`
//!       and writes to `*c = …`. By-mut-ref capture.
//!     - else, the lifted fn takes `c: i64` as an extra param; reads of
//!       `c` stay as `c`. By-value capture.
//!   The closure expression still becomes `Expr::Ident("__closure_<n>")`
//!   (so it lives somewhere as a value), but a per-fn binding map records
//!   the closure name and capture list. When subsequent code calls the
//!   binding by name (`f(args)`), the call site is rewritten to call the
//!   lifted fn directly with captures prepended:
//!   `f(args)` → `__closure_<n>(&mut cap1, cap2, args...)`.
//!
//! Limitation: capturing closures used in pass-as-value position (e.g.
//! `apply(f, ...)`) are NOT supported — the env-struct + indirect-call ABI
//! that handles that case is the L-effort sequel to this work. The pass
//! detects pass-as-value and leaves the call site untouched, which means
//! the resulting code is broken at runtime; we don't try to mask that.

use crate::ast::{BinOp, Block, Expr, FnDecl, Item, Param, Program, Stmt, Ty};
use std::collections::{HashMap, HashSet};

pub fn run(prog: &mut Program) -> usize {
    // Collect all top-level fn names so we don't classify them as captures.
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

    let mut ctx = LiftCtx { lifted: Vec::new(), counter: 0, globals, capture_table: HashMap::new() };
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

#[derive(Debug, Clone)]
struct CapInfo {
    name: String,
    by_mut_ref: bool,
}

struct LiftCtx {
    lifted: Vec<FnDecl>,
    counter: usize,
    globals: HashSet<String>,
    /// Map from a lifted fn name (`__closure_<N>`) to its capture list,
    /// so the call-site rewrite knows what to prepend.
    capture_table: HashMap<String, Vec<CapInfo>>,
}

/// Per-fn state: the scope stack of locally-defined idents, plus the
/// closure-binding map (`bind_name → (lifted_fn_name, captures)`).
#[derive(Default)]
struct FnState {
    scope_stack: Vec<HashSet<String>>,
    bindings: HashMap<String, (String, Vec<CapInfo>)>,
}

impl FnState {
    fn push_scope(&mut self) { self.scope_stack.push(HashSet::new()); }
    fn pop_scope(&mut self)  { self.scope_stack.pop(); }
    fn add_local(&mut self, name: &str) {
        if let Some(top) = self.scope_stack.last_mut() {
            top.insert(name.to_string());
        }
    }
    fn outer_contains(&self, name: &str) -> bool {
        self.scope_stack.iter().any(|s| s.contains(name))
    }
}

fn process_fn(f: &mut FnDecl, ctx: &mut LiftCtx) {
    let body = match f.body.as_mut() {
        Some(b) => b,
        None => return,
    };
    let mut st = FnState::default();
    st.push_scope();
    for p in &f.params { st.add_local(&p.name); }
    process_block(body, ctx, &mut st);
    st.pop_scope();
}

fn process_block(b: &mut Block, ctx: &mut LiftCtx, st: &mut FnState) {
    st.push_scope();
    for s in b.stmts.iter_mut() { process_stmt(s, ctx, st); }
    if let Some(t) = b.tail.as_mut() { process_expr(t, ctx, st); }
    st.pop_scope();
}

fn process_stmt(s: &mut Stmt, ctx: &mut LiftCtx, st: &mut FnState) {
    match s {
        Stmt::Let { name, value, .. } => {
            if let Some(e) = value.as_mut() { process_expr(e, ctx, st); }
            st.add_local(name);
            // If the rhs lifted to an `Ident("__closure_N")` AND that closure
            // had captures, record the binding so subsequent `Call { Ident(name) }`
            // gets the captures prepended.
            if let Some(Expr::Ident(rhs_name)) = value.as_ref() {
                // Pull the capture list out of the lifted-fns roster.
                if let Some(caps) = lookup_captures(ctx, rhs_name) {
                    st.bindings.insert(name.clone(), (rhs_name.clone(), caps));
                }
            }
        }
        Stmt::LetTuple { names, value } => {
            process_expr(value, ctx, st);
            for n in names { st.add_local(n); }
        }
        Stmt::Expr(e) | Stmt::Return(Some(e)) => process_expr(e, ctx, st),
        _ => {}
    }
}

fn lookup_captures(ctx: &LiftCtx, name: &str) -> Option<Vec<CapInfo>> {
    ctx.capture_table.get(name).cloned()
}

fn process_expr(e: &mut Expr, ctx: &mut LiftCtx, st: &mut FnState) {
    // Direct-call rewrite: `Call { callee: Ident(bind_name), args }` where
    // `bind_name` is in the closure-binding map → call lifted fn with
    // captures prepended.
    if let Expr::Call { callee, args } = e {
        if let Expr::Ident(name) = callee.as_ref() {
            if let Some((closure_fn, caps)) = st.bindings.get(name).cloned() {
                let mut new_args: Vec<Expr> = Vec::with_capacity(caps.len() + args.len());
                for c in &caps {
                    if c.by_mut_ref {
                        new_args.push(Expr::Ref {
                            mutable: true,
                            expr: Box::new(Expr::Ident(c.name.clone())),
                        });
                    } else {
                        new_args.push(Expr::Ident(c.name.clone()));
                    }
                }
                // Recurse into existing args FIRST (they may contain other
                // closures or call rewrites).
                for a in args.iter_mut() { process_expr(a, ctx, st); }
                new_args.extend(args.drain(..));
                *callee = Box::new(Expr::Ident(closure_fn));
                *args = new_args;
                return;
            }
        }
    }

    // Recurse into sub-exprs FIRST so nested closures get lifted bottom-up.
    match e {
        Expr::Closure { .. } => {} // handled below
        Expr::Call { callee, args } => {
            process_expr(callee, ctx, st);
            for a in args.iter_mut() { process_expr(a, ctx, st); }
        }
        Expr::MethodCall { recv, args, .. } => {
            process_expr(recv, ctx, st);
            for a in args.iter_mut() { process_expr(a, ctx, st); }
        }
        Expr::Field { recv, .. } => process_expr(recv, ctx, st),
        Expr::Bin { lhs, rhs, .. } => { process_expr(lhs, ctx, st); process_expr(rhs, ctx, st); }
        Expr::Unary { expr, .. } => process_expr(expr, ctx, st),
        Expr::Block(b) => process_block(b, ctx, st),
        Expr::If { cond, then, else_ } => {
            process_expr(cond, ctx, st);
            process_block(then, ctx, st);
            if let Some(b) = else_ { process_block(b, ctx, st); }
        }
        Expr::For { var, iter, body, .. } => {
            process_expr(iter, ctx, st);
            st.push_scope();
            st.add_local(var);
            process_block(body, ctx, st);
            st.pop_scope();
        }
        Expr::While { cond, body } => { process_expr(cond, ctx, st); process_block(body, ctx, st); }
        Expr::Range { lo, hi, step } => {
            process_expr(lo, ctx, st); process_expr(hi, ctx, st);
            if let Some(s) = step { process_expr(s, ctx, st); }
        }
        Expr::Ref { expr, .. } => process_expr(expr, ctx, st),
        Expr::Region { body, .. } => process_block(body, ctx, st),
        Expr::StructLit { fields, .. } => for (_, fv) in fields { process_expr(fv, ctx, st); },
        Expr::Match { scrutinee, arms } => {
            process_expr(scrutinee, ctx, st);
            for (_, arm) in arms { process_expr(arm, ctx, st); }
        }
        Expr::Cast { expr, .. } => process_expr(expr, ctx, st),
        Expr::Try(inner) => process_expr(inner, ctx, st),
        Expr::Index { recv, idx } => { process_expr(recv, ctx, st); process_expr(idx, ctx, st); }
        Expr::Tuple(elems) => for el in elems { process_expr(el, ctx, st); },
        Expr::Deref(inner) => process_expr(inner, ctx, st),
        _ => {}
    }
    // Now lift THIS node if it's a Closure.
    if let Expr::Closure { params, body } = e {
        let name = format!("__closure_{}", ctx.counter);
        ctx.counter += 1;
        // Capture analysis: find free vars in the body that exist in the
        // enclosing scope (st.scope_stack). Closure params shadow.
        let param_set: HashSet<String> = params.iter().map(|(n, _)| n.clone()).collect();
        let mut caps: Vec<CapInfo> = Vec::new();
        let mut cap_index: HashMap<String, usize> = HashMap::new();
        collect_captures(body, &param_set, &st.scope_stack, &ctx.globals,
                         &mut caps, &mut cap_index);
        // Rewrite body to deref mut captures on read and write.
        let mut_caps: HashSet<String> = caps.iter().filter(|c| c.by_mut_ref)
            .map(|c| c.name.clone()).collect();
        if !mut_caps.is_empty() {
            rewrite_mut_capture_uses(body, &mut_caps);
        }
        // Synthesize the lifted fn. Captures come first as params (mut → &mut i64,
        // by-val → i64), then the user's params.
        let mut fn_params: Vec<Param> = Vec::new();
        for c in &caps {
            let ty = if c.by_mut_ref {
                Ty::Ref { mutable: true, inner: Box::new(Ty::Named("i64".into())) }
            } else {
                Ty::Named("i64".into())
            };
            fn_params.push(Param { name: c.name.clone(), ty });
        }
        for (n, t) in params.iter() {
            fn_params.push(Param {
                name: n.clone(),
                ty: t.clone().unwrap_or(Ty::Named("i64".into())),
            });
        }
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
        ctx.capture_table.insert(name.clone(), caps);
        ctx.lifted.push(lifted);
        *e = Expr::Ident(name);
    }
}

/// Walk the closure body, collecting free vars (idents not in the closure's
/// own params, but present in the enclosing scope stack). For each capture,
/// classify by whether the body writes to it (Bin::Assign with Ident lhs).
fn collect_captures(
    body: &Expr,
    params: &HashSet<String>,
    outer_scope: &[HashSet<String>],
    globals: &HashSet<String>,
    caps: &mut Vec<CapInfo>,
    cap_index: &mut HashMap<String, usize>,
) {
    let in_outer = |n: &str| outer_scope.iter().any(|s| s.contains(n));
    let mut record = |n: &str, is_mut: bool, caps: &mut Vec<CapInfo>, cap_index: &mut HashMap<String, usize>| {
        if let Some(&i) = cap_index.get(n) {
            if is_mut { caps[i].by_mut_ref = true; }
        } else {
            cap_index.insert(n.to_string(), caps.len());
            caps.push(CapInfo { name: n.to_string(), by_mut_ref: is_mut });
        }
    };
    let consider = |n: &str| -> bool {
        !params.contains(n) && in_outer(n) && !globals.contains(n)
    };
    walk_expr_collect(body, params, &consider, &mut record, caps, cap_index);
}

fn walk_expr_collect<F, G>(
    e: &Expr,
    params: &HashSet<String>,
    consider: &F,
    record: &mut G,
    caps: &mut Vec<CapInfo>,
    cap_index: &mut HashMap<String, usize>,
) where F: Fn(&str) -> bool, G: FnMut(&str, bool, &mut Vec<CapInfo>, &mut HashMap<String, usize>) {
    match e {
        Expr::Ident(n) => {
            if consider(n) { record(n, false, caps, cap_index); }
        }
        Expr::Bin { op: BinOp::Assign, lhs, rhs } => {
            // If lhs is a bare ident matching an outer-scope name, this
            // is a write to a capture → mark mut. Recurse into rhs as
            // a normal read context.
            if let Expr::Ident(n) = lhs.as_ref() {
                if consider(n) { record(n, true, caps, cap_index); }
            } else {
                walk_expr_collect(lhs, params, consider, record, caps, cap_index);
            }
            walk_expr_collect(rhs, params, consider, record, caps, cap_index);
        }
        Expr::Call { callee, args } => {
            walk_expr_collect(callee, params, consider, record, caps, cap_index);
            for a in args { walk_expr_collect(a, params, consider, record, caps, cap_index); }
        }
        Expr::MethodCall { recv, args, .. } => {
            walk_expr_collect(recv, params, consider, record, caps, cap_index);
            for a in args { walk_expr_collect(a, params, consider, record, caps, cap_index); }
        }
        Expr::Field { recv, .. } => walk_expr_collect(recv, params, consider, record, caps, cap_index),
        Expr::Bin { lhs, rhs, .. } => {
            walk_expr_collect(lhs, params, consider, record, caps, cap_index);
            walk_expr_collect(rhs, params, consider, record, caps, cap_index);
        }
        Expr::Unary { expr, .. } => walk_expr_collect(expr, params, consider, record, caps, cap_index),
        Expr::Block(b) => walk_block_collect(b, params, consider, record, caps, cap_index),
        Expr::If { cond, then, else_ } => {
            walk_expr_collect(cond, params, consider, record, caps, cap_index);
            walk_block_collect(then, params, consider, record, caps, cap_index);
            if let Some(b) = else_ { walk_block_collect(b, params, consider, record, caps, cap_index); }
        }
        Expr::For { iter, body, .. } => {
            walk_expr_collect(iter, params, consider, record, caps, cap_index);
            walk_block_collect(body, params, consider, record, caps, cap_index);
        }
        Expr::While { cond, body } => {
            walk_expr_collect(cond, params, consider, record, caps, cap_index);
            walk_block_collect(body, params, consider, record, caps, cap_index);
        }
        Expr::Range { lo, hi, step } => {
            walk_expr_collect(lo, params, consider, record, caps, cap_index);
            walk_expr_collect(hi, params, consider, record, caps, cap_index);
            if let Some(s) = step { walk_expr_collect(s, params, consider, record, caps, cap_index); }
        }
        Expr::Ref { mutable, expr } => {
            // `&mut x` where x is an outer-scope local → capture as mut.
            if let Expr::Ident(n) = expr.as_ref() {
                if consider(n) { record(n, *mutable, caps, cap_index); }
            } else {
                walk_expr_collect(expr, params, consider, record, caps, cap_index);
            }
        }
        Expr::Region { body, .. } => walk_block_collect(body, params, consider, record, caps, cap_index),
        Expr::StructLit { fields, .. } => for (_, fv) in fields {
            walk_expr_collect(fv, params, consider, record, caps, cap_index);
        },
        Expr::Match { scrutinee, arms } => {
            walk_expr_collect(scrutinee, params, consider, record, caps, cap_index);
            for (_, arm) in arms { walk_expr_collect(arm, params, consider, record, caps, cap_index); }
        }
        Expr::Cast { expr, .. } => walk_expr_collect(expr, params, consider, record, caps, cap_index),
        Expr::Try(inner) => walk_expr_collect(inner, params, consider, record, caps, cap_index),
        Expr::Index { recv, idx } => {
            walk_expr_collect(recv, params, consider, record, caps, cap_index);
            walk_expr_collect(idx, params, consider, record, caps, cap_index);
        }
        Expr::Tuple(elems) => for el in elems {
            walk_expr_collect(el, params, consider, record, caps, cap_index);
        },
        Expr::Deref(inner) => walk_expr_collect(inner, params, consider, record, caps, cap_index),
        Expr::Closure { .. } => {} // nested closures handled by their own pass
        _ => {}
    }
}

fn walk_block_collect<F, G>(
    b: &Block,
    params: &HashSet<String>,
    consider: &F,
    record: &mut G,
    caps: &mut Vec<CapInfo>,
    cap_index: &mut HashMap<String, usize>,
) where F: Fn(&str) -> bool, G: FnMut(&str, bool, &mut Vec<CapInfo>, &mut HashMap<String, usize>) {
    for s in &b.stmts {
        match s {
            Stmt::Let { value: Some(e), .. } => walk_expr_collect(e, params, consider, record, caps, cap_index),
            Stmt::LetTuple { value, .. } => walk_expr_collect(value, params, consider, record, caps, cap_index),
            Stmt::Expr(e) => walk_expr_collect(e, params, consider, record, caps, cap_index),
            Stmt::Return(Some(e)) => walk_expr_collect(e, params, consider, record, caps, cap_index),
            _ => {}
        }
    }
    if let Some(t) = &b.tail { walk_expr_collect(t, params, consider, record, caps, cap_index); }
}

/// Rewrite reads and writes of captured-mut-ref vars to go through Deref.
/// `acc` (read) → `*acc`. `acc = rhs` → `*acc = rhs`. The lifted fn's param
/// `acc` has type `&mut i64`, which the asm backend treats as an i64 holding
/// a pointer, so `Deref(Ident("acc"))` lowers to `movq (%rax), %rax` after
/// loading `acc`'s slot.
fn rewrite_mut_capture_uses(e: &mut Expr, mut_caps: &HashSet<String>) {
    match e {
        Expr::Ident(n) if mut_caps.contains(n) => {
            let inner = Expr::Ident(n.clone());
            *e = Expr::Deref(Box::new(inner));
        }
        Expr::Bin { op: BinOp::Assign, lhs, rhs } => {
            // Rewrite the LHS specially: don't recurse into a bare ident
            // that's a mut capture (we want `*ident = rhs`, not
            // `*(*ident) = rhs`).
            match lhs.as_mut() {
                Expr::Ident(n) if mut_caps.contains(n) => {
                    let inner = Expr::Ident(n.clone());
                    **lhs = Expr::Deref(Box::new(inner));
                }
                other => rewrite_mut_capture_uses(other, mut_caps),
            }
            rewrite_mut_capture_uses(rhs, mut_caps);
        }
        Expr::Call { callee, args } => {
            rewrite_mut_capture_uses(callee, mut_caps);
            for a in args { rewrite_mut_capture_uses(a, mut_caps); }
        }
        Expr::MethodCall { recv, args, .. } => {
            rewrite_mut_capture_uses(recv, mut_caps);
            for a in args { rewrite_mut_capture_uses(a, mut_caps); }
        }
        Expr::Field { recv, .. } => rewrite_mut_capture_uses(recv, mut_caps),
        Expr::Bin { lhs, rhs, .. } => {
            rewrite_mut_capture_uses(lhs, mut_caps);
            rewrite_mut_capture_uses(rhs, mut_caps);
        }
        Expr::Unary { expr, .. } => rewrite_mut_capture_uses(expr, mut_caps),
        Expr::Block(b) => rewrite_block_mut_capture(b, mut_caps),
        Expr::If { cond, then, else_ } => {
            rewrite_mut_capture_uses(cond, mut_caps);
            rewrite_block_mut_capture(then, mut_caps);
            if let Some(b) = else_ { rewrite_block_mut_capture(b, mut_caps); }
        }
        Expr::For { iter, body, .. } => {
            rewrite_mut_capture_uses(iter, mut_caps);
            rewrite_block_mut_capture(body, mut_caps);
        }
        Expr::While { cond, body } => {
            rewrite_mut_capture_uses(cond, mut_caps);
            rewrite_block_mut_capture(body, mut_caps);
        }
        Expr::Range { lo, hi, step } => {
            rewrite_mut_capture_uses(lo, mut_caps);
            rewrite_mut_capture_uses(hi, mut_caps);
            if let Some(s) = step { rewrite_mut_capture_uses(s, mut_caps); }
        }
        Expr::Ref { expr, .. } => rewrite_mut_capture_uses(expr, mut_caps),
        Expr::Region { body, .. } => rewrite_block_mut_capture(body, mut_caps),
        Expr::StructLit { fields, .. } => for (_, fv) in fields {
            rewrite_mut_capture_uses(fv, mut_caps);
        },
        Expr::Match { scrutinee, arms } => {
            rewrite_mut_capture_uses(scrutinee, mut_caps);
            for (_, arm) in arms { rewrite_mut_capture_uses(arm, mut_caps); }
        }
        Expr::Cast { expr, .. } => rewrite_mut_capture_uses(expr, mut_caps),
        Expr::Try(inner) => rewrite_mut_capture_uses(inner, mut_caps),
        Expr::Index { recv, idx } => {
            rewrite_mut_capture_uses(recv, mut_caps);
            rewrite_mut_capture_uses(idx, mut_caps);
        }
        Expr::Tuple(elems) => for el in elems { rewrite_mut_capture_uses(el, mut_caps); },
        Expr::Deref(inner) => rewrite_mut_capture_uses(inner, mut_caps),
        _ => {}
    }
}

fn rewrite_block_mut_capture(b: &mut Block, mut_caps: &HashSet<String>) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { value: Some(e), .. } => rewrite_mut_capture_uses(e, mut_caps),
            Stmt::LetTuple { value, .. } => rewrite_mut_capture_uses(value, mut_caps),
            Stmt::Expr(e) => rewrite_mut_capture_uses(e, mut_caps),
            Stmt::Return(Some(e)) => rewrite_mut_capture_uses(e, mut_caps),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { rewrite_mut_capture_uses(t, mut_caps); }
}
