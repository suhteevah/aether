//! MIR-level kernel fusion pass.
//!
//! Today this is a thin AST rewrite — the MIR proper isn't on the asm-codegen
//! path yet, so "fusion" runs as a peephole transform over `Block::stmts` just
//! before asm codegen. The pass is structured so each pattern is a small,
//! independently-testable rewriter; adding a new fusion is just adding another
//! match arm in `try_fuse_pair`.
//!
//! Patterns supported (Phase-0 set, all on the cuBLAS GPU path):
//!
//! 1. `x.matmul(&w, &mut out); out.gelu(&mut out);`
//!     →  `x.matmul_gelu(&w, &mut out);`
//!     Saves one ABI round-trip + one buffer-registry hit. The fused
//!     runtime fn issues sgemm + a single in-place gelu launch back to back.
//!
//! Future patterns (sketched in comments at each match site):
//! - `x.layer_norm(...) → x.matmul(...) → out.gelu(...)` triple fusion.
//! - `dscores.softmax_backward(&dy, &mut dx); dx.scale(s);` fused into
//!   `softmax_bwd_scaled`.
//! - `x.add(&residual, &mut out); out.layer_norm(...)` (norm-after-residual).
//!
//! The pass MUST be conservative: a fused method's output buffer is only safe
//! to substitute when the intermediate's only-use is the immediately following
//! op AND every subsequent reference is to the FINAL buffer (not the
//! intermediate). Since today's source pattern is `let h: Tensor; x.matmul(...,
//! &mut h); h.gelu(&mut h);` (in-place), this trivially holds.

use crate::ast::{Block, Expr, FnDecl, Item, Program, Stmt};

pub fn run(prog: &mut Program) -> usize {
    let mut total = 0;
    for item in &mut prog.items {
        if let Item::Fn(f) = item {
            if let Some(b) = f.body.as_mut() {
                total += fuse_block(b);
            }
        }
        // Methods inside `impl` blocks are flattened to `Item::Fn` AFTER this
        // pass by the asm backend (try_emit time), so we'd miss them here.
        // Walk them too.
        if let Item::Impl { methods, .. } | Item::ImplTrait { methods, .. } = item {
            for m in methods.iter_mut() {
                if let Some(b) = m.body.as_mut() {
                    total += fuse_block(b);
                }
            }
        }
        let _ = fn_marker; // silence unused-fn lint if no fns visited
    }
    total
}

fn fn_marker(_: &FnDecl) {}

fn fuse_block(b: &mut Block) -> usize {
    let mut total = 0;
    // Recurse into nested blocks first so inner fuses count + outer can see
    // a stable form.
    for s in b.stmts.iter_mut() {
        total += recurse_stmt(s);
    }
    if let Some(t) = b.tail.as_mut() {
        total += recurse_expr(t);
    }
    // Now scan the stmt sequence for adjacent fusable pairs, in order.
    let mut i = 0;
    while i + 1 < b.stmts.len() {
        if let Some(fused) = try_fuse_pair(&b.stmts[i], &b.stmts[i + 1]) {
            b.stmts[i] = fused;
            b.stmts.remove(i + 1);
            total += 1;
            // Don't advance i — the new stmt at i might itself fuse with i+1.
        } else {
            i += 1;
        }
    }
    total
}

fn recurse_stmt(s: &mut Stmt) -> usize {
    match s {
        Stmt::Expr(e) => recurse_expr(e),
        Stmt::Let { value: Some(e), .. } => recurse_expr(e),
        Stmt::LetTuple { value, .. } => recurse_expr(value),
        Stmt::Return(Some(e)) => recurse_expr(e),
        _ => 0,
    }
}

fn recurse_expr(e: &mut Expr) -> usize {
    match e {
        Expr::If { then, else_, .. } => {
            let mut t = fuse_block(then);
            if let Some(b) = else_ { t += fuse_block(b); }
            t
        }
        Expr::For { body, .. } | Expr::While { body, .. } => fuse_block(body),
        Expr::Block(b) => fuse_block(b),
        _ => 0,
    }
}

/// Try to fuse two adjacent statements. Returns `Some(replacement_stmt)` on
/// match; `None` to keep the pair as-is.
fn try_fuse_pair(a: &Stmt, b: &Stmt) -> Option<Stmt> {
    // Pattern: matmul → gelu in-place.
    //   a: Stmt::Expr(MethodCall { method: "matmul", recv, args: [w, &mut out] })
    //   b: Stmt::Expr(MethodCall { method: "gelu",   recv: out, args: [&mut out2] })
    //   require: name(out) == name(recv_b) == name(out2)
    if let (Stmt::Expr(ea), Stmt::Expr(eb)) = (a, b) {
        if let (Expr::MethodCall { name: name_a, recv: recv_a, args: args_a },
                Expr::MethodCall { name: name_b, recv: recv_b, args: args_b }) = (ea, eb) {
            if name_a == "matmul" && name_b == "gelu"
                && args_a.len() == 2 && args_b.len() == 1
            {
                let out_ident_a = ref_target_ident(&args_a[1]);
                let recv_ident_b = match recv_b.as_ref() {
                    Expr::Ident(n) => Some(n.clone()),
                    _ => None,
                };
                let out_ident_b = ref_target_ident(&args_b[0]);
                if out_ident_a.is_some() && out_ident_a == recv_ident_b && out_ident_a == out_ident_b {
                    let fused = Expr::MethodCall {
                        name: "matmul_gelu".to_string(),
                        recv: recv_a.clone(),
                        args: vec![args_a[0].clone(), args_a[1].clone()],
                    };
                    return Some(Stmt::Expr(fused));
                }
            }
            // Pattern: add → layer_norm (residual+norm, every transformer
            // sublayer has this).
            //   a: x.add(&y, &mut sum)
            //   b: sum.layer_norm(&gamma, &beta, &mut out, &mut mean, &mut rstd, eps)
            //   require: name(sum) == name(recv_b)
            // Combined into x.add_layer_norm(&y, &gamma, &beta, &mut out,
            // &mut mean, &mut rstd, eps).
            if name_a == "add" && name_b == "layer_norm"
                && args_a.len() == 2 && args_b.len() == 6
            {
                let sum_ident_a = ref_target_ident(&args_a[1]);
                let recv_ident_b = match recv_b.as_ref() {
                    Expr::Ident(n) => Some(n.clone()),
                    _ => None,
                };
                if sum_ident_a.is_some() && sum_ident_a == recv_ident_b {
                    // Fused: recv = x; args = [&y, gamma, beta, out, mean, rstd, eps].
                    let mut new_args = Vec::with_capacity(7);
                    new_args.push(args_a[0].clone()); // &y
                    for arg in args_b.iter() { new_args.push(arg.clone()); }
                    let fused = Expr::MethodCall {
                        name: "add_layer_norm".to_string(),
                        recv: recv_a.clone(),
                        args: new_args,
                    };
                    return Some(Stmt::Expr(fused));
                }
            }
            // Pattern: softmax_backward → in-place scale.
            //   a: y.softmax_backward(&dy, &mut dx)
            //   b: dx.scale(s)            (scale is in-place on the receiver)
            // Combined into y.softmax_backward_scaled(&dy, &mut dx, s).
            // Used heavily in attention backward (dscores = softmax_bwd; dscores.scale(1/sqrt(d_k))).
            if name_a == "softmax_backward" && name_b == "scale"
                && args_a.len() == 2 && args_b.len() == 1
            {
                let dx_ident_a = ref_target_ident(&args_a[1]);
                let recv_ident_b = match recv_b.as_ref() {
                    Expr::Ident(n) => Some(n.clone()),
                    _ => None,
                };
                if dx_ident_a.is_some() && dx_ident_a == recv_ident_b {
                    let mut new_args = args_a.clone();   // [&dy, &mut dx]
                    new_args.push(args_b[0].clone());    // append scalar s
                    let fused = Expr::MethodCall {
                        name: "softmax_backward_scaled".to_string(),
                        recv: recv_a.clone(),
                        args: new_args,
                    };
                    return Some(Stmt::Expr(fused));
                }
            }
        }
    }
    None
}

/// Pull the identifier name out of `&x` or `&mut x` or bare `x`. None for
/// anything else.
fn ref_target_ident(e: &Expr) -> Option<String> {
    match e {
        Expr::Ref { expr, .. } => match expr.as_ref() {
            Expr::Ident(n) => Some(n.clone()),
            _ => None,
        },
        Expr::Ident(n) => Some(n.clone()),
        _ => None,
    }
}
