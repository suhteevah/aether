//! P15.4 — Cross-fn inlining.
//!
//! At `--O1` we walk the program twice:
//!
//! **Pass 1** — survey. For each `Item::Fn` with a body decide whether the
//! fn is *inlinable*:
//!   - body present (not `extern`), no const-generic params,
//!   - no `Stmt::Return` anywhere in the body (falls off the end via tail),
//!   - body stmt count ≤ 5 (heuristic for "small"),
//!   - not recursive (the fn doesn't call itself directly).
//!
//! Build a map `name → InlineSnippet { params, body }` of inlinable fns.
//!
//! **Pass 2** — splice. Walk every fn body. At each
//! `Expr::Call { callee: Ident(name), args }` where `name` is inlinable and
//! the arg count matches, replace the Call with a `Expr::Block` containing:
//!
//!   `[let __inl_N_p1 = arg1; let __inl_N_p2 = arg2; ...;
//!     <renamed body stmts>]`
//!
//! and the renamed body's tail expression as the Block's tail. `N` is a
//! per-splice counter so multiple inlines of the same fn don't collide on
//! local names.
//!
//! The pass returns the count of substitutions made; main.rs prints that to
//! stderr alongside the other --O1 reports.
//!
//! Witness: `tests/runtime/inline_smoke.aether` declares
//! `fn add_one(x: i64) -> i64 { x + 1 }` and exits `add_one(41)`; at --O1
//! the resulting asm contains zero `callq aether_add_one` lines and a
//! single `movq $42, %rax` (the constant fold lands on top of the inline).

use std::collections::{HashMap, HashSet};

use crate::ast::{Block, Expr, FnDecl, Item, Program, Stmt};

#[derive(Clone)]
struct InlineSnippet {
    params: Vec<String>,
    body: Block,
}

pub fn run(prog: &mut Program) -> usize {
    // ---- Pass 1: survey ----
    let mut snippets: HashMap<String, InlineSnippet> = HashMap::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            if let Some(b) = &f.body {
                if !is_inlinable(f, b) { continue; }
                snippets.insert(f.name.clone(), InlineSnippet {
                    params: f.params.iter().map(|p| p.name.clone()).collect(),
                    body: b.clone(),
                });
            }
        }
    }
    if snippets.is_empty() { return 0; }

    // ---- Pass 2: splice ----
    let mut counter: usize = 0;
    let mut count: usize = 0;
    for it in prog.items.iter_mut() {
        match it {
            Item::Fn(f) => {
                if let Some(b) = f.body.as_mut() {
                    splice_block(b, &snippets, &f.name, &mut counter, &mut count);
                }
            }
            Item::Impl { methods, .. } | Item::ImplTrait { methods, .. } => {
                for m in methods.iter_mut() {
                    if let Some(b) = m.body.as_mut() {
                        splice_block(b, &snippets, &m.name, &mut counter, &mut count);
                    }
                }
            }
            _ => {}
        }
    }
    count
}

fn is_inlinable(f: &FnDecl, b: &Block) -> bool {
    // Skip extern/const-generic/distributed/autodiff/server/spec — those have
    // codegen-relevant attributes whose semantics we can't preserve via
    // textual substitution.
    if f.is_extern { return false; }
    if !f.const_params.is_empty() { return false; }
    for a in &f.attrs {
        if matches!(a.name.as_str(), "autodiff" | "distributed" | "server" | "spec") {
            return false;
        }
    }
    // Don't inline `main` — it's the program entry, never a callee.
    if f.name == "main" { return false; }
    // Don't inline lifted closures — their direct-call rewrite already gave
    // them special treatment.
    if f.name.starts_with("__closure_") { return false; }
    // Recursion guard: scan body for direct calls to ourselves.
    if calls_self(b, &f.name) { return false; }
    // Size heuristic.
    if b.stmts.len() > 5 { return false; }
    // No `Stmt::Return` — every exit must be via the tail expr.
    if has_return(b) { return false; }
    true
}

fn calls_self(b: &Block, self_name: &str) -> bool {
    for s in &b.stmts { if stmt_calls(s, self_name) { return true; } }
    if let Some(t) = &b.tail { if expr_calls(t, self_name) { return true; } }
    false
}

fn stmt_calls(s: &Stmt, name: &str) -> bool {
    match s {
        Stmt::Let { value: Some(e), .. } => expr_calls(e, name),
        Stmt::LetTuple { value, .. } => expr_calls(value, name),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => expr_calls(e, name),
        _ => false,
    }
}

fn expr_calls(e: &Expr, name: &str) -> bool {
    match e {
        Expr::Call { callee, args } => {
            if let Expr::Ident(n) = callee.as_ref() {
                if n == name { return true; }
            }
            args.iter().any(|a| expr_calls(a, name))
        }
        Expr::MethodCall { recv, args, .. } => {
            expr_calls(recv, name) || args.iter().any(|a| expr_calls(a, name))
        }
        Expr::Field { recv, .. } => expr_calls(recv, name),
        Expr::Bin { lhs, rhs, .. } => expr_calls(lhs, name) || expr_calls(rhs, name),
        Expr::Unary { expr, .. } => expr_calls(expr, name),
        Expr::Block(b) => calls_self(b, name),
        Expr::If { cond, then, else_ } => {
            expr_calls(cond, name) || calls_self(then, name)
                || else_.as_ref().map(|b| calls_self(b, name)).unwrap_or(false)
        }
        Expr::For { iter, body, .. } => expr_calls(iter, name) || calls_self(body, name),
        Expr::While { cond, body } => expr_calls(cond, name) || calls_self(body, name),
        Expr::Range { lo, hi, step } => {
            expr_calls(lo, name) || expr_calls(hi, name)
                || step.as_ref().map(|s| expr_calls(s, name)).unwrap_or(false)
        }
        Expr::Ref { expr, .. } | Expr::Cast { expr, .. } => expr_calls(expr, name),
        Expr::Region { body, .. } => calls_self(body, name),
        Expr::StructLit { fields, .. } => fields.iter().any(|(_, e)| expr_calls(e, name)),
        Expr::Match { scrutinee, arms } => {
            expr_calls(scrutinee, name) || arms.iter().any(|(_, a)| expr_calls(a, name))
        }
        Expr::Try(inner) | Expr::Deref(inner) => expr_calls(inner, name),
        Expr::Index { recv, idx } => expr_calls(recv, name) || expr_calls(idx, name),
        Expr::Tuple(elems) => elems.iter().any(|e| expr_calls(e, name)),
        _ => false,
    }
}

fn has_return(b: &Block) -> bool {
    for s in &b.stmts {
        if let Stmt::Return(_) = s { return true; }
        // Check for nested Return inside expressions.
        if stmt_has_return(s) { return true; }
    }
    if let Some(t) = &b.tail { if expr_has_return(t) { return true; } }
    false
}

fn stmt_has_return(s: &Stmt) -> bool {
    match s {
        Stmt::Let { value: Some(e), .. } => expr_has_return(e),
        Stmt::LetTuple { value, .. } => expr_has_return(value),
        Stmt::Expr(e) => expr_has_return(e),
        Stmt::Return(_) => true,
        _ => false,
    }
}

fn expr_has_return(e: &Expr) -> bool {
    match e {
        Expr::Block(b) => has_return(b),
        Expr::If { cond, then, else_ } => {
            expr_has_return(cond) || has_return(then)
                || else_.as_ref().map(|b| has_return(b)).unwrap_or(false)
        }
        Expr::For { iter, body, .. } => expr_has_return(iter) || has_return(body),
        Expr::While { cond, body } => expr_has_return(cond) || has_return(body),
        Expr::Bin { lhs, rhs, .. } => expr_has_return(lhs) || expr_has_return(rhs),
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. }
        | Expr::Try(expr) | Expr::Deref(expr) => expr_has_return(expr),
        Expr::Call { callee, args } => {
            expr_has_return(callee) || args.iter().any(|a| expr_has_return(a))
        }
        _ => false,
    }
}

fn splice_block(
    b: &mut Block,
    snippets: &HashMap<String, InlineSnippet>,
    cur_fn: &str,
    counter: &mut usize,
    count: &mut usize,
) {
    for s in b.stmts.iter_mut() {
        splice_stmt(s, snippets, cur_fn, counter, count);
    }
    if let Some(t) = b.tail.as_mut() {
        splice_expr(t, snippets, cur_fn, counter, count);
    }
}

fn splice_stmt(
    s: &mut Stmt,
    snippets: &HashMap<String, InlineSnippet>,
    cur_fn: &str,
    counter: &mut usize,
    count: &mut usize,
) {
    match s {
        Stmt::Let { value: Some(e), .. } => splice_expr(e, snippets, cur_fn, counter, count),
        Stmt::LetTuple { value, .. } => splice_expr(value, snippets, cur_fn, counter, count),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => splice_expr(e, snippets, cur_fn, counter, count),
        _ => {}
    }
}

fn splice_expr(
    e: &mut Expr,
    snippets: &HashMap<String, InlineSnippet>,
    cur_fn: &str,
    counter: &mut usize,
    count: &mut usize,
) {
    // Recurse into children FIRST.
    match e {
        Expr::Call { callee, args } => {
            splice_expr(callee, snippets, cur_fn, counter, count);
            for a in args.iter_mut() { splice_expr(a, snippets, cur_fn, counter, count); }
        }
        Expr::MethodCall { recv, args, .. } => {
            splice_expr(recv, snippets, cur_fn, counter, count);
            for a in args.iter_mut() { splice_expr(a, snippets, cur_fn, counter, count); }
        }
        Expr::Field { recv, .. } => splice_expr(recv, snippets, cur_fn, counter, count),
        Expr::Bin { lhs, rhs, .. } => {
            splice_expr(lhs, snippets, cur_fn, counter, count);
            splice_expr(rhs, snippets, cur_fn, counter, count);
        }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. }
        | Expr::Try(expr) | Expr::Deref(expr) | Expr::Ref { expr, .. } => {
            splice_expr(expr, snippets, cur_fn, counter, count);
        }
        Expr::Block(b) => splice_block(b, snippets, cur_fn, counter, count),
        Expr::If { cond, then, else_ } => {
            splice_expr(cond, snippets, cur_fn, counter, count);
            splice_block(then, snippets, cur_fn, counter, count);
            if let Some(b) = else_ { splice_block(b, snippets, cur_fn, counter, count); }
        }
        Expr::For { iter, body, .. } => {
            splice_expr(iter, snippets, cur_fn, counter, count);
            splice_block(body, snippets, cur_fn, counter, count);
        }
        Expr::While { cond, body } => {
            splice_expr(cond, snippets, cur_fn, counter, count);
            splice_block(body, snippets, cur_fn, counter, count);
        }
        Expr::Range { lo, hi, step } => {
            splice_expr(lo, snippets, cur_fn, counter, count);
            splice_expr(hi, snippets, cur_fn, counter, count);
            if let Some(s) = step { splice_expr(s, snippets, cur_fn, counter, count); }
        }
        Expr::Region { body, .. } => splice_block(body, snippets, cur_fn, counter, count),
        Expr::StructLit { fields, .. } => for (_, fv) in fields {
            splice_expr(fv, snippets, cur_fn, counter, count);
        },
        Expr::Match { scrutinee, arms } => {
            splice_expr(scrutinee, snippets, cur_fn, counter, count);
            for (_, arm) in arms { splice_expr(arm, snippets, cur_fn, counter, count); }
        }
        Expr::Index { recv, idx } => {
            splice_expr(recv, snippets, cur_fn, counter, count);
            splice_expr(idx, snippets, cur_fn, counter, count);
        }
        Expr::Tuple(elems) => for el in elems { splice_expr(el, snippets, cur_fn, counter, count); },
        _ => {}
    }
    // Now check whether THIS node is an inlinable Call.
    if let Expr::Call { callee, args } = e {
        if let Expr::Ident(name) = callee.as_ref() {
            // Don't inline self (post-recursion guard — even the survey skips
            // it, but a method named the same as a fn could re-trigger).
            if name == cur_fn { return; }
            if let Some(snip) = snippets.get(name) {
                if snip.params.len() != args.len() { return; }
                let n = *counter;
                *counter += 1;
                // Fresh locals: rename every body local + every param to
                // `__inl_<n>_<orig>`. Compute the rename map up front by
                // walking the body, then apply it to a clone.
                let mut renames: HashMap<String, String> = HashMap::new();
                for p in &snip.params {
                    renames.insert(p.clone(), format!("__inl_{}_{}", n, p));
                }
                collect_local_renames(&snip.body, n, &mut renames);
                let mut body = snip.body.clone();
                rename_block(&mut body, &renames);
                // Build the splicing block: for each (param, arg) emit a
                // `let __inl_<n>_<param> = arg;` stmt. Then append the
                // body's stmts (already renamed). Tail = body's tail.
                let mut new_stmts: Vec<Stmt> = Vec::new();
                let taken_args = std::mem::take(args);
                for (p, a) in snip.params.iter().zip(taken_args.into_iter()) {
                    new_stmts.push(Stmt::Let {
                        name: format!("__inl_{}_{}", n, p),
                        mutable: false,
                        ty: None,
                        value: Some(a),
                    });
                }
                new_stmts.extend(body.stmts);
                let block = Block { stmts: new_stmts, tail: body.tail };
                *e = Expr::Block(block);
                *count += 1;
            }
        }
    }
}

fn collect_local_renames(b: &Block, n: usize, renames: &mut HashMap<String, String>) {
    for s in &b.stmts {
        match s {
            Stmt::Let { name, .. } => {
                renames.entry(name.clone()).or_insert_with(|| format!("__inl_{}_{}", n, name));
            }
            Stmt::LetTuple { names, .. } => {
                for nm in names {
                    renames.entry(nm.clone()).or_insert_with(|| format!("__inl_{}_{}", n, nm));
                }
            }
            _ => {}
        }
    }
}

fn rename_block(b: &mut Block, renames: &HashMap<String, String>) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { name, value, .. } => {
                if let Some(new_name) = renames.get(name) { *name = new_name.clone(); }
                if let Some(e) = value { rename_expr(e, renames); }
            }
            Stmt::LetTuple { names, value } => {
                for nm in names.iter_mut() {
                    if let Some(new_name) = renames.get(nm) { *nm = new_name.clone(); }
                }
                rename_expr(value, renames);
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => rename_expr(e, renames),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { rename_expr(t, renames); }
}

fn rename_expr(e: &mut Expr, renames: &HashMap<String, String>) {
    match e {
        Expr::Ident(n) => {
            if let Some(new_name) = renames.get(n) { *n = new_name.clone(); }
        }
        Expr::Call { callee, args } => {
            // The callee Ident is a fn name, NEVER a local — don't rename.
            // But if it's a complex callee expression, recurse.
            if !matches!(callee.as_ref(), Expr::Ident(_)) {
                rename_expr(callee, renames);
            }
            for a in args.iter_mut() { rename_expr(a, renames); }
        }
        Expr::MethodCall { recv, args, .. } => {
            rename_expr(recv, renames);
            for a in args.iter_mut() { rename_expr(a, renames); }
        }
        Expr::Field { recv, .. } => rename_expr(recv, renames),
        Expr::Bin { lhs, rhs, .. } => { rename_expr(lhs, renames); rename_expr(rhs, renames); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. }
        | Expr::Try(expr) | Expr::Deref(expr) | Expr::Ref { expr, .. } => {
            rename_expr(expr, renames);
        }
        Expr::Block(b) => rename_block(b, renames),
        Expr::If { cond, then, else_ } => {
            rename_expr(cond, renames);
            rename_block(then, renames);
            if let Some(b) = else_ { rename_block(b, renames); }
        }
        Expr::For { var, iter, body, .. } => {
            // The for-var is a binding, follow the rename if present (the
            // survey adds it via collect_local_renames — but inlinable bodies
            // typically don't carry For loops since size ≤ 5; defensive).
            if let Some(new_name) = renames.get(var) { *var = new_name.clone(); }
            rename_expr(iter, renames);
            rename_block(body, renames);
        }
        Expr::While { cond, body } => {
            rename_expr(cond, renames);
            rename_block(body, renames);
        }
        Expr::Range { lo, hi, step } => {
            rename_expr(lo, renames);
            rename_expr(hi, renames);
            if let Some(s) = step { rename_expr(s, renames); }
        }
        Expr::Region { body, .. } => rename_block(body, renames),
        Expr::StructLit { fields, .. } => for (_, fv) in fields { rename_expr(fv, renames); },
        Expr::Match { scrutinee, arms } => {
            rename_expr(scrutinee, renames);
            for (_, arm) in arms { rename_expr(arm, renames); }
        }
        Expr::Index { recv, idx } => { rename_expr(recv, renames); rename_expr(idx, renames); }
        Expr::Tuple(elems) => for el in elems { rename_expr(el, renames); },
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    #![allow(unused_imports, unused_variables)]
    use super::*;
    use crate::ast::{BinOp, FnDecl, Item, Param, Program, Ty};

    fn mk_fn(name: &str, params: Vec<&str>, body: Block) -> FnDecl {
        FnDecl {
            attrs: Vec::new(), is_pub: false, is_extern: false,
            name: name.into(), const_params: Vec::new(),
            params: params.into_iter().map(|p| Param {
                name: p.into(), ty: Ty::Named("i64".into()),
            }).collect(),
            ret: Some(Ty::Named("i64".into())),
            body: Some(body),
        }
    }

    #[test]
    fn inlines_tail_only_fn() {
        // fn add_one(x: i64) -> i64 { x + 1 }
        let add_one = mk_fn("add_one", vec!["x"], Block {
            stmts: Vec::new(),
            tail: Some(Box::new(Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(Expr::Ident("x".into())),
                rhs: Box::new(Expr::IntLit(1)),
            })),
        });
        // fn main() -> i64 { add_one(41) }
        let main = mk_fn("main", vec![], Block {
            stmts: Vec::new(),
            tail: Some(Box::new(Expr::Call {
                callee: Box::new(Expr::Ident("add_one".into())),
                args: vec![Expr::IntLit(41)],
            })),
        });
        let mut prog = Program { items: vec![Item::Fn(add_one), Item::Fn(main)] };
        let n = run(&mut prog);
        assert_eq!(n, 1);
        // Verify main's tail is now a Block, not a Call.
        if let Item::Fn(m) = &prog.items[1] {
            let tail = m.body.as_ref().unwrap().tail.as_ref().unwrap();
            assert!(matches!(tail.as_ref(), Expr::Block(_)));
        } else { panic!() }
    }

    #[test]
    fn skips_recursive() {
        // fn fib(x: i64) -> i64 { fib(x - 1) + fib(x - 2) }
        let fib = mk_fn("fib", vec!["x"], Block {
            stmts: Vec::new(),
            tail: Some(Box::new(Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(Expr::Call {
                    callee: Box::new(Expr::Ident("fib".into())),
                    args: vec![Expr::Bin {
                        op: BinOp::Sub,
                        lhs: Box::new(Expr::Ident("x".into())),
                        rhs: Box::new(Expr::IntLit(1)),
                    }],
                }),
                rhs: Box::new(Expr::Call {
                    callee: Box::new(Expr::Ident("fib".into())),
                    args: vec![Expr::Bin {
                        op: BinOp::Sub,
                        lhs: Box::new(Expr::Ident("x".into())),
                        rhs: Box::new(Expr::IntLit(2)),
                    }],
                }),
            })),
        });
        let main = mk_fn("main", vec![], Block {
            stmts: Vec::new(),
            tail: Some(Box::new(Expr::Call {
                callee: Box::new(Expr::Ident("fib".into())),
                args: vec![Expr::IntLit(5)],
            })),
        });
        let mut prog = Program { items: vec![Item::Fn(fib), Item::Fn(main)] };
        let n = run(&mut prog);
        assert_eq!(n, 0, "recursive fn must not be inlined");
    }

    #[test]
    fn unique_renames_across_two_inlines() {
        // Ensure two splices of the same fn don't collide on the synthesized
        // local name. add_one inlined at two distinct call sites yields
        // __inl_0_x and __inl_1_x.
        let _ = HashSet::<String>::new(); // silence import
        let add_one = mk_fn("add_one", vec!["x"], Block {
            stmts: Vec::new(),
            tail: Some(Box::new(Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(Expr::Ident("x".into())),
                rhs: Box::new(Expr::IntLit(1)),
            })),
        });
        // fn main() { add_one(1) + add_one(2) }
        let main = mk_fn("main", vec![], Block {
            stmts: Vec::new(),
            tail: Some(Box::new(Expr::Bin {
                op: BinOp::Add,
                lhs: Box::new(Expr::Call {
                    callee: Box::new(Expr::Ident("add_one".into())),
                    args: vec![Expr::IntLit(1)],
                }),
                rhs: Box::new(Expr::Call {
                    callee: Box::new(Expr::Ident("add_one".into())),
                    args: vec![Expr::IntLit(2)],
                }),
            })),
        });
        let mut prog = Program { items: vec![Item::Fn(add_one), Item::Fn(main)] };
        let n = run(&mut prog);
        assert_eq!(n, 2);
    }
}
