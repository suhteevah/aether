//! Async lowering — transform `async fn`s into poll-based state machines driven
//! by the real cooperative executor (`aether_executor_run` / `aether_block_on`).
//! This is the compiler half of async/await (the runtime half lives in
//! `runtime/src/lib.rs`).
//!
//! An `async fn f(p0, …) -> i64 { body }` (marked `#[__async]` by the parser)
//! becomes a FUTURE laid out on the heap as
//!   `[ poll_fn(0) | pc(1) | result(2) | p0(3) | p1(4) | … | local0 | local1 | … ]`
//! and is split into two ordinary fns:
//!   * a CONSTRUCTOR (keeps the name `f`): allocates the future, stores the poll
//!     fn ptr + pc=0 + each param, returns the future pointer;
//!   * a POLL fn `__f_poll(state) -> i64`: a `pc`-dispatched state machine. Each
//!     `yield_now().await;` suspension statement splits the body into a segment;
//!     a segment advances `pc` and returns 0 (Pending); the final segment stores
//!     the tail value into `result` and returns 1 (Ready).
//! Every param/local lives in the future struct (accessed via `aether_load_i64`
//! / `aether_store_i64`), so values persist across suspensions automatically —
//! no separate liveness pass.
//!
//! A `.await` (parsed as `MethodCall{name:"__await"}`): a bare `expr.await`
//! statement inside an async fn is a SUSPENSION POINT; everywhere else
//! `expr.await` lowers to `aether_block_on(expr)` (drive the future to
//! completion and yield its result). v1 scope: straight-line async-fn bodies.

use crate::ast::{Block, Expr, FnDecl, Item, Program, Stmt, Ty};
use std::collections::HashMap;

const SLOT_POLL: i64 = 0;
const SLOT_PC: i64 = 1;
const SLOT_RESULT: i64 = 2;
const SLOT_PARAM0: i64 = 3;

fn is_async(f: &FnDecl) -> bool {
    f.attrs.iter().any(|a| a.name == "__async")
}

fn is_await(e: &Expr) -> bool {
    matches!(e, Expr::MethodCall { name, .. } if name == "__await")
}

fn call(name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call { callee: Box::new(Expr::Ident(name.into())), args }
}
fn store(slot: i64, val: Expr) -> Stmt {
    Stmt::Expr(call("aether_store_i64", vec![Expr::Ident("__st".into()), Expr::IntLit(slot), val]))
}
fn load(slot: i64) -> Expr {
    call("aether_load_i64", vec![Expr::Ident("__st".into()), Expr::IntLit(slot)])
}

pub fn run(prog: &mut Program) -> usize {
    let mut generated: Vec<FnDecl> = Vec::new();
    let mut n = 0;
    for it in prog.items.iter_mut() {
        if let Item::Fn(f) = it {
            if is_async(f) {
                if let Some(poll) = lower_async_fn(f) {
                    generated.push(poll);
                    n += 1;
                }
            } else if let Some(b) = f.body.as_mut() {
                // Non-async fn: a `.await` here drives the future synchronously.
                rewrite_block_awaits(b);
            }
        }
    }
    for f in generated { prog.items.push(Item::Fn(f)); }
    n
}

fn collect_lets(b: &Block, m: &mut HashMap<String, i64>, slot: &mut i64) {
    for s in &b.stmts {
        if let Stmt::Let { name, .. } = s {
            if !m.contains_key(name) { m.insert(name.clone(), *slot); *slot += 1; }
        }
    }
}

/// Transform an async fn in place into its constructor, returning the poll fn.
fn lower_async_fn(f: &mut FnDecl) -> Option<FnDecl> {
    let body = f.body.take()?;
    // Slot map computed from the TAKEN body (params first, then locals).
    let mut slots: HashMap<String, i64> = HashMap::new();
    let mut slot = SLOT_PARAM0;
    for p in &f.params { slots.insert(p.name.clone(), slot); slot += 1; }
    collect_lets(&body, &mut slots, &mut slot);
    // 3 reserved (poll/pc/result) + one slot per param + one per local.
    let total_slots = SLOT_PARAM0 + slots.len() as i64;
    let poll_name = format!("__{}_poll", f.name);

    // Split body statements into segments at bare `.await` suspension stmts.
    let mut segments: Vec<Vec<Stmt>> = vec![Vec::new()];
    for s in &body.stmts {
        if let Stmt::Expr(e) = s {
            if is_await(e) { segments.push(Vec::new()); continue; }
        }
        segments.last_mut().unwrap().push(s.clone());
    }
    let tail = body.tail.clone();

    // Build the poll fn body: an `if pc == k { … }` per segment.
    let mut poll_stmts: Vec<Stmt> = Vec::new();
    let nseg = segments.len();
    for (k, seg) in segments.iter().enumerate() {
        let mut blk: Vec<Stmt> = Vec::new();
        for s in seg {
            blk.push(rewrite_stmt(s, &slots));
        }
        let is_last = k + 1 == nseg;
        if is_last {
            if let Some(t) = &tail {
                blk.push(store(SLOT_RESULT, rewrite_expr(t, &slots)));
            }
            blk.push(store(SLOT_PC, Expr::IntLit(k as i64 + 1)));
            blk.push(Stmt::Return(Some(Expr::IntLit(1)))); // Ready
        } else {
            blk.push(store(SLOT_PC, Expr::IntLit(k as i64 + 1)));
            blk.push(Stmt::Return(Some(Expr::IntLit(0)))); // Pending
        }
        // `if pc == k { blk }`
        let cond = Expr::Bin {
            op: crate::ast::BinOp::Eq,
            lhs: Box::new(load(SLOT_PC)),
            rhs: Box::new(Expr::IntLit(k as i64)),
        };
        poll_stmts.push(Stmt::Expr(Expr::If {
            cond: Box::new(cond),
            then: Block { stmts: blk, tail: None },
            else_: None,
        }));
    }
    // The poll fn takes `__st` (the future) directly, so segment code that
    // references `__st` resolves to the param. Fallthrough tail `1` = Ready.
    let poll_fn = FnDecl {
        attrs: Vec::new(), is_pub: false, is_extern: false,
        name: poll_name.clone(), const_params: Vec::new(),
        params: vec![crate::ast::Param { name: "__st".into(), ty: Ty::Named("i64".into()) }],
        ret: Some(Ty::Named("i64".into())),
        body: Some(Block { stmts: poll_stmts, tail: Some(Box::new(Expr::IntLit(1))) }),
    };

    // Rewrite the async fn `f` into the CONSTRUCTOR.
    let mut ctor: Vec<Stmt> = Vec::new();
    ctor.push(Stmt::Let {
        name: "__st".into(), mutable: false, ty: Some(Ty::Named("i64".into())),
        value: Some(call("aether_alloc_bytes", vec![Expr::IntLit(total_slots * 8)])),
    });
    ctor.push(store(SLOT_POLL, Expr::Ident(poll_name)));
    ctor.push(store(SLOT_PC, Expr::IntLit(0)));
    for p in &f.params {
        let slot = slots[&p.name];
        ctor.push(store(slot, Expr::Ident(p.name.clone())));
    }
    f.attrs.retain(|a| a.name != "__async");
    f.body = Some(Block { stmts: ctor, tail: Some(Box::new(Expr::Ident("__st".into()))) });

    Some(poll_fn)
}

fn rewrite_stmt(s: &Stmt, slots: &HashMap<String, i64>) -> Stmt {
    match s {
        Stmt::Let { name, value: Some(v), .. } => {
            if let Some(&slot) = slots.get(name) {
                store(slot, rewrite_expr(v, slots))
            } else {
                Stmt::Expr(rewrite_expr(v, slots))
            }
        }
        Stmt::Expr(e) => {
            // Assignment `x = v` -> store to x's slot.
            if let Expr::Bin { op: crate::ast::BinOp::Assign, lhs, rhs } = e {
                if let Expr::Ident(n) = lhs.as_ref() {
                    if let Some(&slot) = slots.get(n) {
                        return store(slot, rewrite_expr(rhs, slots));
                    }
                }
            }
            Stmt::Expr(rewrite_expr(e, slots))
        }
        Stmt::Return(Some(e)) => Stmt::Return(Some(rewrite_expr(e, slots))),
        other => other.clone(),
    }
}

/// Replace each param/local `Ident` with a load from its future slot, and any
/// `.await` with a synchronous `aether_block_on`.
fn rewrite_expr(e: &Expr, slots: &HashMap<String, i64>) -> Expr {
    match e {
        Expr::Ident(n) => {
            if let Some(&slot) = slots.get(n) { load(slot) } else { e.clone() }
        }
        Expr::MethodCall { recv, name, args } if name == "__await" && args.is_empty() => {
            call("aether_block_on", vec![rewrite_expr(recv, slots)])
        }
        Expr::Bin { op, lhs, rhs } => Expr::Bin {
            op: *op,
            lhs: Box::new(rewrite_expr(lhs, slots)),
            rhs: Box::new(rewrite_expr(rhs, slots)),
        },
        Expr::Unary { op, expr } => Expr::Unary { op: *op, expr: Box::new(rewrite_expr(expr, slots)) },
        Expr::Cast { expr, ty } => Expr::Cast { expr: Box::new(rewrite_expr(expr, slots)), ty: ty.clone() },
        Expr::Call { callee, args } => Expr::Call {
            callee: callee.clone(),
            args: args.iter().map(|a| rewrite_expr(a, slots)).collect(),
        },
        Expr::MethodCall { recv, name, args } => Expr::MethodCall {
            recv: Box::new(rewrite_expr(recv, slots)),
            name: name.clone(),
            args: args.iter().map(|a| rewrite_expr(a, slots)).collect(),
        },
        Expr::If { cond, then, else_ } => Expr::If {
            cond: Box::new(rewrite_expr(cond, slots)),
            then: rewrite_block(then, slots),
            else_: else_.as_ref().map(|b| rewrite_block(b, slots)),
        },
        _ => e.clone(),
    }
}

fn rewrite_block(b: &Block, slots: &HashMap<String, i64>) -> Block {
    Block {
        stmts: b.stmts.iter().map(|s| rewrite_stmt(s, slots)).collect(),
        tail: b.tail.as_ref().map(|t| Box::new(rewrite_expr(t, slots))),
    }
}

/// In a non-async fn, rewrite every `.await` to `aether_block_on(recv)`.
fn rewrite_block_awaits(b: &mut Block) {
    let empty: HashMap<String, i64> = HashMap::new();
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { value: Some(v), .. } => *v = rewrite_awaits_only(v, &empty),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => *e = rewrite_awaits_only(e, &empty),
            Stmt::LetTuple { value, .. } => *value = rewrite_awaits_only(value, &empty),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { **t = rewrite_awaits_only(t, &empty); }
}

/// Like rewrite_expr but only touches `.await` (no slot substitution) — for
/// non-async fns where idents are real locals.
fn rewrite_awaits_only(e: &Expr, _s: &HashMap<String, i64>) -> Expr {
    match e {
        Expr::MethodCall { recv, name, args } if name == "__await" && args.is_empty() => {
            call("aether_block_on", vec![rewrite_awaits_only(recv, _s)])
        }
        Expr::Bin { op, lhs, rhs } => Expr::Bin {
            op: *op,
            lhs: Box::new(rewrite_awaits_only(lhs, _s)),
            rhs: Box::new(rewrite_awaits_only(rhs, _s)),
        },
        Expr::Cast { expr, ty } => Expr::Cast { expr: Box::new(rewrite_awaits_only(expr, _s)), ty: ty.clone() },
        Expr::Call { callee, args } => Expr::Call {
            callee: callee.clone(),
            args: args.iter().map(|a| rewrite_awaits_only(a, _s)).collect(),
        },
        _ => e.clone(),
    }
}
