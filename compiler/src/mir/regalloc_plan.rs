//! Per-fn register-assignment plan for the asm backend.
//!
//! Phase 15.2 (FR-15.2) — `mir::regalloc_drive::drive` already computes a
//! linear-scan register assignment over each fn body and reports
//! `(reg_count, spill_count)`. The asm backend ignored the result and
//! stored every local on the stack. This module is the bridge: it returns
//! a `HashMap<String /*fn*/, HashMap<String /*local*/, u8 /*reg*/>>` that
//! the asm backend consults at each Ident load and Let/Assign store.
//!
//! Hot locals stay in callee-saved registers (r12..r15). The asm backend's
//! prologue saves each used reg, the epilogue restores it; in between the
//! local lives in BOTH the stack slot (write-through) and the reg. Reads
//! are always from the reg; writes go to both so anything that depended on
//! the stack slot (FFI by &local — explicitly excluded from reg promotion
//! below — or post-mortem stack inspection) still sees fresh values.
//!
//! Exclusions: a local is NOT promoted if any of these hold —
//!   * its address is taken (`&local` or `&mut local`)
//!   * its value-expression is a struct/tuple/enum/array literal
//!   * its declared type is a struct/tuple/array/Tensor
//!   * it shadows an earlier let in the same body (the SSA scaffold can't
//!     map shadowed names to a single physical reg without confusing the
//!     CSE step downstream — easier to leave these on the stack)
//!
//! Pool is `[12, 13, 14, 15]` (callee-saved per the Windows x64 ABI). The
//! asm backend handles push/pop in the prologue/epilogue.

use std::collections::{HashMap, HashSet};

use crate::ast::{Block, Expr, FnDecl, Item, Program, Stmt, Ty};
use super::regalloc::{Allocator, LiveRange, Loc};

/// Plan map: fn name → (local name → physical reg id).
pub type PlanMap = HashMap<String, HashMap<String, u8>>;

/// Build the per-fn assignment plan. Pool defaults to the callee-saved set
/// {r12, r13, r14, r15}; pass a custom pool for tests.
pub fn plan_program(prog: &Program) -> PlanMap {
    plan_program_with_pool(prog, vec![12, 13, 14, 15])
}

pub fn plan_program_with_pool(prog: &Program, pool: Vec<u8>) -> PlanMap {
    let mut out = PlanMap::new();
    for it in &prog.items {
        if let Item::Fn(f) = it {
            if f.is_extern { continue; }
            let Some(body) = &f.body else { continue; };
            let plan = plan_fn(f, body, &pool);
            if !plan.is_empty() { out.insert(f.name.clone(), plan); }
        }
    }
    out
}

fn plan_fn(_f: &FnDecl, body: &Block, pool: &[u8]) -> HashMap<String, u8> {
    // 1. Walk body to collect (name, decl_idx, last_use_idx) for every
    //    Stmt::Let. Skips shadowed re-declarations.
    let mut decls: Vec<(String, u32)> = Vec::new();
    let mut last_use: HashMap<String, u32> = HashMap::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut shadowed: HashSet<String> = HashSet::new();
    let mut idx: u32 = 0;
    walk_block(body, &mut decls, &mut last_use, &mut seen, &mut shadowed, &mut idx);

    // 2. Find names whose address is taken — they MUST stay in their stack
    //    slot so `&x` returns a meaningful pointer.
    let mut addr_taken: HashSet<String> = HashSet::new();
    collect_addr_taken_block(body, &mut addr_taken);

    // 3. Find names whose let-value or declared-type makes them ineligible
    //    (struct/tuple/array/Tensor — these are multi-slot or handle-bound).
    let mut ineligible_kind: HashSet<String> = HashSet::new();
    collect_ineligible_kind(body, &mut ineligible_kind);

    // 4. Build LiveRanges for eligible names.
    let eligible: Vec<&(String, u32)> = decls.iter()
        .filter(|(n, _)| !addr_taken.contains(n))
        .filter(|(n, _)| !ineligible_kind.contains(n))
        .filter(|(n, _)| !shadowed.contains(n))
        .collect();
    if eligible.is_empty() { return HashMap::new(); }

    let mut ranges: Vec<LiveRange> = eligible.iter().enumerate().map(|(i, (n, start))| {
        let end = *last_use.get(n).unwrap_or(start);
        LiveRange { vreg: i as u32, start: *start, end }
    }).collect();

    let allocator = Allocator::new(pool.to_vec());
    let result = allocator.allocate(&mut ranges);

    // 5. Materialise (name → reg) for the Loc::Reg outputs. The allocator
    //    returns (vreg, Loc) pairs; map vreg back to the original name via
    //    the `eligible` vector's position.
    let mut plan: HashMap<String, u8> = HashMap::new();
    for (vreg, loc) in result {
        if let Loc::Reg(r) = loc {
            let (name, _) = &eligible[vreg as usize];
            plan.insert(name.clone(), r);
        }
    }
    plan
}

// ── walkers ──────────────────────────────────────────────────────────────

fn walk_block(
    b: &Block,
    decls: &mut Vec<(String, u32)>,
    last_use: &mut HashMap<String, u32>,
    seen: &mut HashSet<String>,
    shadowed: &mut HashSet<String>,
    idx: &mut u32,
) {
    for s in &b.stmts {
        match s {
            Stmt::Let { name, value, .. } => {
                if !seen.insert(name.clone()) { shadowed.insert(name.clone()); }
                if let Some(v) = value { collect_uses(v, last_use, *idx); }
                decls.push((name.clone(), *idx));
                *idx += 1;
            }
            Stmt::LetTuple { names, value } => {
                for n in names {
                    if !seen.insert(n.clone()) { shadowed.insert(n.clone()); }
                }
                collect_uses(value, last_use, *idx);
                // LetTuple is multi-slot; not reg-promotable. Still record
                // decls so walk indexing stays consistent with body order.
                for n in names { decls.push((n.clone(), *idx)); shadowed.insert(n.clone()); }
                *idx += 1;
            }
            Stmt::Expr(e) | Stmt::Return(Some(e)) => {
                collect_uses(e, last_use, *idx);
                *idx += 1;
            }
            Stmt::Return(None) => { *idx += 1; }
        }
    }
    if let Some(t) = &b.tail { collect_uses(t, last_use, *idx); }
}

fn collect_uses(e: &Expr, out: &mut HashMap<String, u32>, idx: u32) {
    match e {
        Expr::Ident(n) => { out.insert(n.clone(), idx); }
        Expr::Bin { lhs, rhs, .. } => { collect_uses(lhs, out, idx); collect_uses(rhs, out, idx); }
        Expr::Unary { expr, .. } => collect_uses(expr, out, idx),
        Expr::Call { callee, args } => {
            collect_uses(callee, out, idx);
            for a in args { collect_uses(a, out, idx); }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_uses(recv, out, idx);
            for a in args { collect_uses(a, out, idx); }
        }
        Expr::Field { recv, .. } => collect_uses(recv, out, idx),
        Expr::If { cond, then, else_ } => {
            collect_uses(cond, out, idx);
            for s in &then.stmts { collect_stmt_uses(s, out, idx); }
            if let Some(t) = &then.tail { collect_uses(t, out, idx); }
            if let Some(eb) = else_ {
                for s in &eb.stmts { collect_stmt_uses(s, out, idx); }
                if let Some(t) = &eb.tail { collect_uses(t, out, idx); }
            }
        }
        Expr::While { cond, body } | Expr::For { iter: cond, body, .. } => {
            collect_uses(cond, out, idx);
            for s in &body.stmts { collect_stmt_uses(s, out, idx); }
            if let Some(t) = &body.tail { collect_uses(t, out, idx); }
        }
        Expr::Block(b) => {
            for s in &b.stmts { collect_stmt_uses(s, out, idx); }
            if let Some(t) = &b.tail { collect_uses(t, out, idx); }
        }
        Expr::Range { lo, hi, step } => {
            collect_uses(lo, out, idx); collect_uses(hi, out, idx);
            if let Some(s) = step { collect_uses(s, out, idx); }
        }
        Expr::Cast { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) =>
            collect_uses(expr, out, idx),
        Expr::Index { recv, idx: i } => { collect_uses(recv, out, idx); collect_uses(i, out, idx); }
        Expr::Tuple(es) => for e in es { collect_uses(e, out, idx); }
        Expr::StructLit { fields, .. } => for (_, e) in fields { collect_uses(e, out, idx); }
        Expr::Match { scrutinee, arms } =>
            { collect_uses(scrutinee, out, idx); for (_, e) in arms { collect_uses(e, out, idx); } }
        _ => {}
    }
}

fn collect_stmt_uses(s: &Stmt, out: &mut HashMap<String, u32>, idx: u32) {
    match s {
        Stmt::Let { value: Some(v), .. } => collect_uses(v, out, idx),
        Stmt::Let { .. } => {}
        Stmt::LetTuple { value, .. } => collect_uses(value, out, idx),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => collect_uses(e, out, idx),
        Stmt::Return(None) => {}
    }
}

fn collect_addr_taken_block(b: &Block, out: &mut HashSet<String>) {
    for s in &b.stmts { collect_addr_taken_stmt(s, out); }
    if let Some(t) = &b.tail { collect_addr_taken_expr(t, out); }
}

fn collect_addr_taken_stmt(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Let { value: Some(v), .. } => collect_addr_taken_expr(v, out),
        Stmt::LetTuple { value, .. } => collect_addr_taken_expr(value, out),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => collect_addr_taken_expr(e, out),
        _ => {}
    }
}

fn collect_addr_taken_expr(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::Ref { expr, .. } => {
            if let Expr::Ident(n) = expr.as_ref() { out.insert(n.clone()); }
            collect_addr_taken_expr(expr, out);
        }
        Expr::Bin { lhs, rhs, .. } => { collect_addr_taken_expr(lhs, out); collect_addr_taken_expr(rhs, out); }
        Expr::Unary { expr, .. } | Expr::Cast { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) =>
            collect_addr_taken_expr(expr, out),
        Expr::Call { callee, args } => {
            collect_addr_taken_expr(callee, out);
            for a in args { collect_addr_taken_expr(a, out); }
        }
        Expr::MethodCall { recv, args, .. } => {
            collect_addr_taken_expr(recv, out);
            for a in args { collect_addr_taken_expr(a, out); }
        }
        Expr::Field { recv, .. } => collect_addr_taken_expr(recv, out),
        Expr::Index { recv, idx } => { collect_addr_taken_expr(recv, out); collect_addr_taken_expr(idx, out); }
        Expr::If { cond, then, else_ } => {
            collect_addr_taken_expr(cond, out);
            collect_addr_taken_block(then, out);
            if let Some(eb) = else_ { collect_addr_taken_block(eb, out); }
        }
        Expr::While { cond, body } | Expr::For { iter: cond, body, .. } => {
            collect_addr_taken_expr(cond, out);
            collect_addr_taken_block(body, out);
        }
        Expr::Block(b) => collect_addr_taken_block(b, out),
        Expr::Range { lo, hi, step } => {
            collect_addr_taken_expr(lo, out); collect_addr_taken_expr(hi, out);
            if let Some(s) = step { collect_addr_taken_expr(s, out); }
        }
        Expr::Tuple(es) => for e in es { collect_addr_taken_expr(e, out); }
        Expr::StructLit { fields, .. } => for (_, e) in fields { collect_addr_taken_expr(e, out); }
        Expr::Match { scrutinee, arms } =>
            { collect_addr_taken_expr(scrutinee, out); for (_, e) in arms { collect_addr_taken_expr(e, out); } }
        _ => {}
    }
}

fn collect_ineligible_kind(b: &Block, out: &mut HashSet<String>) {
    for s in &b.stmts {
        if let Stmt::Let { name, value, ty, .. } = s {
            if let Some(t) = ty {
                if ty_is_composite(t) { out.insert(name.clone()); continue; }
            }
            if let Some(v) = value {
                if matches!(v, Expr::StructLit { .. } | Expr::Tuple(_)) {
                    out.insert(name.clone());
                }
            }
            // Uninit `let x;` with composite type also excluded above; uninit
            // with no type is already filtered by the eligibility check (no
            // value to read into a reg — would still work technically, but
            // skip for simplicity).
            if value.is_none() { out.insert(name.clone()); }
        }
    }
    // Recurse into nested blocks so inner lets are also classified.
    for s in &b.stmts { recurse_ineligible(s, out); }
    if let Some(t) = &b.tail { recurse_ineligible_expr(t, out); }
}

fn recurse_ineligible(s: &Stmt, out: &mut HashSet<String>) {
    match s {
        Stmt::Let { value: Some(v), .. } => recurse_ineligible_expr(v, out),
        Stmt::Expr(e) | Stmt::Return(Some(e)) => recurse_ineligible_expr(e, out),
        _ => {}
    }
}

fn recurse_ineligible_expr(e: &Expr, out: &mut HashSet<String>) {
    match e {
        Expr::If { cond, then, else_ } => {
            recurse_ineligible_expr(cond, out);
            collect_ineligible_kind(then, out);
            if let Some(eb) = else_ { collect_ineligible_kind(eb, out); }
        }
        Expr::While { cond, body } | Expr::For { iter: cond, body, .. } => {
            recurse_ineligible_expr(cond, out);
            collect_ineligible_kind(body, out);
        }
        Expr::Block(b) => collect_ineligible_kind(b, out),
        _ => {}
    }
}

fn ty_is_composite(t: &Ty) -> bool {
    match t {
        Ty::Array { .. } | Ty::Tuple(_) | Ty::Shape(_) => true,
        Ty::Ref { inner, .. } => ty_is_composite(inner),
        Ty::Generic { name, .. } => matches!(name.as_str(), "Tensor"),
        Ty::Named(n) => matches!(n.as_str(),
            "Tensor" | "TensorDev" | "TensorDevI32"
            // Be conservative: any capitalised name that's not a primitive
            // is likely a struct/enum. Primitives are i32/i64/f32/f64/bool.
            ) || !is_primitive_name(n),
        Ty::Unit => false,
    }
}

fn is_primitive_name(n: &str) -> bool {
    matches!(n, "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
                | "isize" | "usize" | "f32" | "f64" | "bool")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn parse(src: &str) -> Program {
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        Parser::new(toks).parse_program().unwrap()
    }

    #[test]
    fn assigns_callee_saved_regs_to_simple_locals() {
        let prog = parse(r#"
            fn f() -> i64 {
                let a = 1;
                let b = 2;
                let c = a + b;
                c
            }
        "#);
        let plan = plan_program(&prog);
        let f_plan = plan.get("f").expect("f planned");
        assert!(f_plan.contains_key("a"));
        assert!(f_plan.contains_key("b"));
        assert!(f_plan.contains_key("c"));
        for &r in f_plan.values() {
            assert!((12..=15).contains(&r), "expected r12..r15, got r{}", r);
        }
    }

    #[test]
    fn addr_taken_excluded() {
        let prog = parse(r#"
            extern fn ext_use(p: i64) -> i64;
            fn f() -> i64 {
                let pinned = 42;
                let q = ext_use(&pinned);
                q
            }
        "#);
        let plan = plan_program(&prog);
        let f_plan = plan.get("f").expect("f planned");
        // `pinned` must stay on the stack so its address means something.
        assert!(!f_plan.contains_key("pinned"));
        // `q` is fine to promote.
        assert!(f_plan.contains_key("q"));
    }

    #[test]
    fn extern_fn_not_planned() {
        let prog = parse(r#"
            extern fn ext(p: i64) -> i64;
        "#);
        let plan = plan_program(&prog);
        assert!(plan.is_empty());
    }
}
