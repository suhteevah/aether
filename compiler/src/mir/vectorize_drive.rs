//! Drive `mir::vectorize::plan` over each fn's for-loops at `--O1`. Reports
//! the count of vectorizable loops; the asm backend stays scalar today
//! (vectorization-aware lowering is a deeper rewrite). Integration step:
//! the module sees real loops, not synthetic test inputs.

use crate::ast::{Block, Expr, Item, Program, Stmt};
use super::vectorize::{plan, LoopShape, SimdWidth};

pub fn drive(prog: &Program) -> usize {
    let mut count = 0usize;
    for it in &prog.items {
        if let Item::Fn(f) = it {
            if let Some(body) = &f.body {
                count += count_vectorizable(body);
            }
        }
    }
    count
}

fn count_vectorizable(b: &Block) -> usize {
    let mut n = 0usize;
    for s in &b.stmts {
        n += count_in_stmt(s);
    }
    if let Some(t) = &b.tail { n += count_in_expr(t); }
    n
}

fn count_in_stmt(s: &Stmt) -> usize {
    match s {
        Stmt::Let { value: Some(v), .. } => count_in_expr(v),
        Stmt::Let { .. } => 0,
        Stmt::LetTuple { value, .. } => count_in_expr(value),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => count_in_expr(e),
        Stmt::Return(None) => 0,
    }
}

fn count_in_expr(e: &Expr) -> usize {
    match e {
        Expr::For { iter, body, .. } => {
            // Range with both literal endpoints → known trip count.
            let trip = match iter.as_ref() {
                Expr::Range { lo, hi, .. } => {
                    if let (Expr::IntLit(a), Expr::IntLit(b)) = (lo.as_ref(), hi.as_ref()) {
                        ((*b - *a).max(0)) as u32
                    } else { 0 }
                }
                _ => 0,
            };
            let ops = body.stmts.len() as u32;
            let mut count = if trip > 0 {
                let shape = LoopShape { trip_count: trip, has_loop_carried_dep: false, body_op_count: ops };
                if plan(&shape, SimdWidth::Avx256).is_some() { 1 } else { 0 }
            } else { 0 };
            count += count_vectorizable(body);
            count
        }
        Expr::While { body, .. } => count_vectorizable(body),
        Expr::If { then, else_, .. } => {
            let mut n = count_vectorizable(then);
            if let Some(e) = else_ { n += count_vectorizable(e); }
            n
        }
        Expr::Block(b) => count_vectorizable(b),
        _ => 0,
    }
}
