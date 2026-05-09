//! AST-level optimization pass driven by `--O1`. Walks each fn body and
//! folds binary-op expressions whose operands are both integer literals.
//!
//! This is the bridge between the SSA/opt scaffold (`mir::ssa`, `mir::opt`,
//! which operate on a string-keyed parallel IR) and the AST-walking asm
//! emitter that actually drives codegen. Rather than route every fn through
//! the SSA builder + back, we apply the same algebraic identities directly
//! to the AST so the asm emitter sees pre-folded literals — the `--O1`
//! witness compiles `let x = 2 * 3 * 7;` to a single `movq $42` immediate.
//!
//! Scope intentionally narrow: integer constfold + a tiny `0 + x` / `x * 1`
//! identity collapse. Larger transforms (DCE, CSE) belong on the typed IR.

use crate::ast::{BinOp, Expr, FnDecl, Item, Program, Stmt, UnOp};

pub fn optimize_program(p: &mut Program, opt_level: u8) {
    if opt_level == 0 { return; }
    for it in &mut p.items {
        if let Item::Fn(f) = it { optimize_fn(f); }
    }
}

fn optimize_fn(f: &mut FnDecl) {
    if let Some(body) = &mut f.body {
        for s in &mut body.stmts {
            optimize_stmt(s);
        }
        if let Some(tail) = &mut body.tail {
            **tail = fold_expr(std::mem::replace(&mut **tail, Expr::IntLit(0)));
        }
    }
}

fn optimize_stmt(s: &mut Stmt) {
    match s {
        Stmt::Let { value, .. } => {
            if let Some(v) = value {
                *v = fold_expr(std::mem::replace(v, Expr::IntLit(0)));
            }
        }
        Stmt::LetTuple { value, .. } => {
            *value = fold_expr(std::mem::replace(value, Expr::IntLit(0)));
        }
        Stmt::Return(opt) => {
            if let Some(v) = opt {
                *v = fold_expr(std::mem::replace(v, Expr::IntLit(0)));
            }
        }
        Stmt::Expr(e) => { *e = fold_expr(std::mem::replace(e, Expr::IntLit(0))); }
    }
}

/// Fold integer constant arithmetic. Recurses bottom-up so chains like
/// `2 * 3 * 7` collapse step-by-step.
pub fn fold_expr(e: Expr) -> Expr {
    match e {
        Expr::Bin { op, lhs, rhs } => {
            let lhs = fold_expr(*lhs);
            let rhs = fold_expr(*rhs);
            if let (Expr::IntLit(a), Expr::IntLit(b)) = (&lhs, &rhs) {
                let folded = match op {
                    BinOp::Add => Some(a.wrapping_add(*b)),
                    BinOp::Sub => Some(a.wrapping_sub(*b)),
                    BinOp::Mul => Some(a.wrapping_mul(*b)),
                    BinOp::Div if *b != 0 => Some(a.wrapping_div(*b)),
                    BinOp::Mod if *b != 0 => Some(a.wrapping_rem(*b)),
                    BinOp::Shl => Some(a.wrapping_shl((*b as u32) & 63)),
                    BinOp::Shr => Some(a.wrapping_shr((*b as u32) & 63)),
                    BinOp::BitAnd => Some(a & b),
                    BinOp::BitOr  => Some(a | b),
                    BinOp::BitXor => Some(a ^ b),
                    _ => None,
                };
                if let Some(v) = folded { return Expr::IntLit(v); }
            }
            // Identity collapses.
            if let (BinOp::Add, Expr::IntLit(0)) = (op, &rhs) { return lhs; }
            if let (BinOp::Add, Expr::IntLit(0)) = (op, &lhs) { return rhs; }
            if let (BinOp::Mul, Expr::IntLit(1)) = (op, &rhs) { return lhs; }
            if let (BinOp::Mul, Expr::IntLit(1)) = (op, &lhs) { return rhs; }
            if let (BinOp::Sub, Expr::IntLit(0)) = (op, &rhs) { return lhs; }
            Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs) }
        }
        Expr::Unary { op, expr } => {
            let inner = fold_expr(*expr);
            if let Expr::IntLit(v) = &inner {
                return match op {
                    UnOp::Neg => Expr::IntLit(v.wrapping_neg()),
                    UnOp::Not => Expr::IntLit(if *v == 0 { 1 } else { 0 }),
                };
            }
            Expr::Unary { op, expr: Box::new(inner) }
        }
        Expr::Call { callee, args } => Expr::Call {
            callee,
            args: args.into_iter().map(fold_expr).collect(),
        },
        Expr::MethodCall { recv, name, args } => Expr::MethodCall {
            recv: Box::new(fold_expr(*recv)),
            name,
            args: args.into_iter().map(fold_expr).collect(),
        },
        Expr::If { cond, then, else_ } => Expr::If {
            cond: Box::new(fold_expr(*cond)),
            then: fold_block(then),
            else_: else_.map(fold_block),
        },
        Expr::While { cond, body } => Expr::While {
            cond: Box::new(fold_expr(*cond)),
            body: fold_block(body),
        },
        Expr::For { var, iter, body, parallel, distributed } => Expr::For {
            var, iter: Box::new(fold_expr(*iter)),
            body: fold_block(body),
            parallel, distributed,
        },
        Expr::Range { lo, hi, step } => Expr::Range {
            lo: Box::new(fold_expr(*lo)),
            hi: Box::new(fold_expr(*hi)),
            step: step.map(|s| Box::new(fold_expr(*s))),
        },
        Expr::Block(b) => Expr::Block(fold_block(b)),
        Expr::Cast { expr, ty } => Expr::Cast { expr: Box::new(fold_expr(*expr)), ty },
        Expr::Index { recv, idx } => Expr::Index {
            recv: Box::new(fold_expr(*recv)),
            idx: Box::new(fold_expr(*idx)),
        },
        Expr::Field { recv, name } => Expr::Field { recv: Box::new(fold_expr(*recv)), name },
        Expr::Ref { mutable, expr } => Expr::Ref { mutable, expr: Box::new(fold_expr(*expr)) },
        Expr::Deref(e) => Expr::Deref(Box::new(fold_expr(*e))),
        Expr::Try(e) => Expr::Try(Box::new(fold_expr(*e))),
        // Pass-through leaves.
        other => other,
    }
}

fn fold_block(mut b: crate::ast::Block) -> crate::ast::Block {
    for s in &mut b.stmts { optimize_stmt(s); }
    if let Some(tail) = &mut b.tail {
        **tail = fold_expr(std::mem::replace(&mut **tail, Expr::IntLit(0)));
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fold_chain_237() {
        // 2 * 3 * 7 = 42 — left-assoc evaluation.
        let e = Expr::Bin {
            op: BinOp::Mul,
            lhs: Box::new(Expr::Bin {
                op: BinOp::Mul,
                lhs: Box::new(Expr::IntLit(2)),
                rhs: Box::new(Expr::IntLit(3)),
            }),
            rhs: Box::new(Expr::IntLit(7)),
        };
        assert!(matches!(fold_expr(e), Expr::IntLit(42)));
    }
    #[test]
    fn fold_add_zero_keeps_other() {
        let e = Expr::Bin {
            op: BinOp::Add,
            lhs: Box::new(Expr::Ident("x".into())),
            rhs: Box::new(Expr::IntLit(0)),
        };
        match fold_expr(e) {
            Expr::Ident(n) => assert_eq!(n, "x"),
            _ => panic!(),
        }
    }
}
