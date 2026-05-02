//! x86-64 assembly emitter (AT&T syntax, GAS-compatible) — Phase 0.5+.
//!
//! Microsoft x64 ABI: rcx, rdx, r8, r9 + 32-byte shadow space; rsp 16-aligned
//! at every CALL.
//!
//! Frame layout per function:
//!
//!     rbp + 0 ......... saved rbp
//!     rbp - 8 ......... local slot 1
//!     rbp - 16 ........ local slot 2
//!     ...
//!     rbp - 8*N ....... local slot N
//!     rbp - 8*N - 32 .. shadow space (callee scratch)
//!     rsp ............. 16-aligned at every CALL
//!
//! Supported expressions:
//! * `IntLit(n)`        → `movq $n, %rax`
//! * `StrLit(s)`        → `leaq .LC{i}(%rip), %rax` (interned)
//! * `Ident(name)`      → `movq -8*slot(%rbp), %rax`
//! * `Bin Add/Sub/Mul`  → push lhs, eval rhs, pop r10, op
//! * `Call(f, args)`    → up to 4 args, each evaluated and moved into the
//!                       Microsoft x64 arg register; nested calls in args
//!                       are not yet supported (the asm path returns an error).
//!
//! Statements:
//! * `let x = expr;` allocates the next slot
//! * `expr;` evaluated for side-effects
//! * `return expr;` evaluates into rax then runs the epilogue
//! * tail expression is the function's return value (rax)

use std::collections::HashMap;

use crate::ast::{BinOp, Block, Expr, FnDecl, Item, Program, Stmt, Ty, UnOp};

/// Where the value of an expression lives after evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TyKind { Int, F32 }

impl TyKind {
    fn from_ty(t: &Ty) -> Option<TyKind> {
        match t {
            Ty::Named(n) if n == "f32" => Some(TyKind::F32),
            Ty::Named(n) if matches!(n.as_str(), "i32" | "i64" | "u32" | "u64" | "bool") => Some(TyKind::Int),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum AsmError {
    NestedCallInArg,
    TooManyArgs,
    UnsupportedExpr(&'static str),
    UnsupportedBinOp(BinOp),
    UnknownIdent(String),
}

pub fn emit(p: &Program) -> String {
    match try_emit(p) {
        Ok(s) => s,
        Err(e) => format!("# asm backend error: {:?}\n", e),
    }
}

pub fn try_emit(p: &Program) -> Result<String, AsmError> {
    let mut s = String::new();
    s.push_str("# AETHER x86-64 assembly (Microsoft x64 ABI)\n");
    s.push_str("# Emitted by aetherc; comments here are debug-only and do not\n");
    s.push_str("# come from any .aether source — those were stripped at lex time.\n\n");

    let mut data = StringTable::default();
    let mut text = String::new();
    let mut all_floats: Vec<(String, f32)> = Vec::new();

    for item in &p.items {
        if let Item::Fn(f) = item {
            if f.body.is_some() {
                let floats = emit_fn(f, &mut text, &mut data)?;
                all_floats.extend(floats);
            }
        }
    }

    if !data.entries.is_empty() || !all_floats.is_empty() {
        s.push_str(".section .rdata,\"dr\"\n");
        for (label, bytes) in &data.entries {
            s.push_str(&format!("{}:\n", label));
            s.push_str(&format!("    .asciz \"{}\"\n", escape(bytes)));
        }
        for (label, v) in &all_floats {
            s.push_str(&format!("{}:\n", label));
            // Emit raw f32 bytes via .long with the bit pattern; gas accepts
            // .long for 4 bytes which is what aether-asm .asciz path doesn't
            // cover.  We hand-emit hex bytes via .byte to stay within our
            // assembler's parser surface.
            let bits = v.to_bits();
            for i in 0..4 {
                s.push_str(&format!("    .byte 0x{:02x}\n", (bits >> (i * 8)) & 0xff));
            }
        }
        s.push('\n');
    }

    s.push_str(".section .text\n");
    s.push_str(".globl main\n\n");
    s.push_str(&text);
    Ok(s)
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            b'\t' => out.push_str("\\t"),
            0x20..=0x7e => out.push(b as char),
            _ => out.push_str(&format!("\\{:03o}", b)),
        }
    }
    out
}

#[derive(Default)]
struct StringTable {
    entries: Vec<(String, String)>,
    counter: usize,
}

impl StringTable {
    fn intern(&mut self, s: &str) -> String {
        for (label, val) in &self.entries {
            if val == s { return label.clone(); }
        }
        let label = format!(".LC{}", self.counter);
        self.counter += 1;
        self.entries.push((label.clone(), s.to_string()));
        label
    }
}

#[derive(Default)]
struct Locals {
    /// name → 1-based slot index (rbp - 8*slot)
    slots: HashMap<String, usize>,
    /// name → kind. Defaults to Int when a let has no annotation.
    types: HashMap<String, TyKind>,
    next_slot: usize,
    /// Counter for generating unique label names per function.
    label_counter: u32,
    /// Function name for label prefixing (so `.Lif_0_0` is unique across fns).
    fn_label_prefix: String,
    /// Stack of (continue_target, break_target) labels for nested loops.
    loop_labels: Vec<(String, String)>,
    /// Number of f32 constants emitted, for naming `.LF<n>` labels.
    float_consts: Vec<f32>,
}

impl Locals {
    fn alloc(&mut self, name: &str) -> usize {
        self.next_slot += 1;
        let s = self.next_slot;
        self.slots.insert(name.to_string(), s);
        s
    }
    fn get(&self, name: &str) -> Option<usize> { self.slots.get(name).copied() }
    fn frame_bytes(&self) -> usize {
        // 8 bytes per slot + 32 bytes shadow space, rounded up to 16.
        let raw = self.next_slot * 8 + 32;
        (raw + 15) & !15
    }
    fn fresh_label(&mut self, hint: &str) -> String {
        let n = self.label_counter; self.label_counter += 1;
        format!(".L_{}_{}_{}", self.fn_label_prefix, hint, n)
    }
    /// Intern an f32 constant; return its label. Per-fn unique via prefix.
    fn intern_f32(&mut self, v: f32) -> String {
        for (i, &existing) in self.float_consts.iter().enumerate() {
            if existing.to_bits() == v.to_bits() {
                return format!(".LF_{}_{}", self.fn_label_prefix, i);
            }
        }
        let label = format!(".LF_{}_{}", self.fn_label_prefix, self.float_consts.len());
        self.float_consts.push(v);
        label
    }
}

/// Emit a cmp + setcc + zero-extend sequence. Operands: rax = lhs, r10 = rhs.
fn emit_cmp(out: &mut String, setcc_mnem: &str) {
    out.push_str("    cmpq %r10, %rax\n");
    out.push_str(&format!("    {} %al\n", setcc_mnem));
    out.push_str("    movzbl %al, %eax\n");
}

/// Emit just the setcc + zero-extend (caller already issued the compare).
fn emit_setcc_int(out: &mut String, setcc_mnem: &str) {
    out.push_str(&format!("    {} %al\n", setcc_mnem));
    out.push_str("    movzbl %al, %eax\n");
}

fn emit_fn(f: &FnDecl, out: &mut String, data: &mut StringTable)
    -> Result<Vec<(String, f32)>, AsmError>
{
    let name = if f.name == "main" { "main".to_string() } else { format!("aether_{}", f.name) };

    // Pre-pass: count locals so the prologue reserves the right amount.
    let mut locals = Locals::default();
    locals.fn_label_prefix = f.name.clone();
    let body = f.body.as_ref().unwrap();
    count_locals(body, &mut locals);
    // After counting, reset name→slot mapping; we'll re-allocate in emit order.
    let frame = locals.frame_bytes();
    locals.slots.clear();
    locals.next_slot = 0;

    out.push_str(&format!("{name}:\n"));
    out.push_str("    pushq %rbp\n");
    out.push_str("    movq %rsp, %rbp\n");
    out.push_str(&format!("    subq ${}, %rsp\n", frame));

    emit_block(body, out, data, &mut locals)?;

    // If the body had a tail expression, its value is already in %rax — leave
    // it alone. Otherwise zero %rax so the function returns 0 by default.
    if body.tail.is_none() {
        out.push_str("    xorl %eax, %eax\n");
    }
    out.push_str(&format!("    addq ${}, %rsp\n", frame));
    out.push_str("    popq %rbp\n");
    out.push_str("    ret\n\n");
    let mut floats = Vec::with_capacity(locals.float_consts.len());
    for (i, v) in locals.float_consts.iter().enumerate() {
        floats.push((format!(".LF_{}_{}", locals.fn_label_prefix, i), *v));
    }
    Ok(floats)
}

fn count_locals(b: &Block, locals: &mut Locals) {
    for s in &b.stmts {
        match s {
            Stmt::Let { name, value, .. } => {
                count_locals_in_expr(value, locals);
                locals.alloc(name);
            }
            Stmt::Expr(e) => count_locals_in_expr(e, locals),
            Stmt::Return(Some(e)) => count_locals_in_expr(e, locals),
            Stmt::Return(None) => {}
        }
    }
    if let Some(t) = &b.tail { count_locals_in_expr(t, locals); }
}

fn count_locals_in_expr(e: &Expr, locals: &mut Locals) {
    match e {
        Expr::If { cond, then, else_ } => {
            count_locals_in_expr(cond, locals);
            count_locals(then, locals);
            if let Some(b) = else_ { count_locals(b, locals); }
        }
        Expr::For { var, iter, body, .. } => {
            // The iteration variable lives in a slot; the upper bound also
            // gets a slot so we don't re-evaluate it each loop.
            count_locals_in_expr(iter, locals);
            locals.alloc(var);
            locals.alloc("_for_end_");
            count_locals(body, locals);
        }
        Expr::While { cond, body } => {
            count_locals_in_expr(cond, locals);
            count_locals(body, locals);
        }
        Expr::Block(b) => count_locals(b, locals),
        Expr::Bin { lhs, rhs, .. } => {
            count_locals_in_expr(lhs, locals);
            count_locals_in_expr(rhs, locals);
        }
        Expr::Unary { expr, .. } => count_locals_in_expr(expr, locals),
        Expr::Call { args, .. } => for a in args { count_locals_in_expr(a, locals); },
        Expr::Range { lo, hi, .. } => {
            count_locals_in_expr(lo, locals);
            count_locals_in_expr(hi, locals);
        }
        _ => {}
    }
}

fn emit_block(b: &Block, out: &mut String, data: &mut StringTable, locals: &mut Locals)
    -> Result<TyKind, AsmError>
{
    let mut last = TyKind::Int;
    for s in &b.stmts { emit_stmt(s, out, data, locals)?; }
    if let Some(tail) = &b.tail {
        last = emit_expr_value(tail, out, data, locals)?;
    }
    Ok(last)
}

fn emit_stmt(s: &Stmt, out: &mut String, data: &mut StringTable, locals: &mut Locals)
    -> Result<(), AsmError>
{
    match s {
        Stmt::Expr(e) => { emit_expr_value(e, out, data, locals)?; Ok(()) }
        Stmt::Return(Some(e)) => {
            emit_expr_value(e, out, data, locals)?;
            let frame = locals.frame_bytes();
            out.push_str(&format!("    addq ${}, %rsp\n", frame));
            out.push_str("    popq %rbp\n");
            out.push_str("    ret\n");
            Ok(())
        }
        Stmt::Return(None) => {
            out.push_str("    xorl %eax, %eax\n");
            let frame = locals.frame_bytes();
            out.push_str(&format!("    addq ${}, %rsp\n", frame));
            out.push_str("    popq %rbp\n");
            out.push_str("    ret\n");
            Ok(())
        }
        Stmt::Let { name, value, ty, .. } => {
            // Decide the local's TyKind: explicit annotation wins, else infer
            // from the value's runtime type.
            let val_ty = emit_expr_value(value, out, data, locals)?;
            let kind = ty.as_ref().and_then(TyKind::from_ty).unwrap_or(val_ty);
            let slot = locals.alloc(name);
            locals.types.insert(name.clone(), kind);
            match kind {
                TyKind::Int => out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
            }
            Ok(())
        }
    }
}

/// Evaluate `e` and leave its result in %rax (Int) or %xmm0 (F32).
/// Returns the TyKind so callers know which register to read.
fn emit_expr_value(e: &Expr, out: &mut String, data: &mut StringTable, locals: &mut Locals)
    -> Result<TyKind, AsmError>
{
    match e {
        Expr::IntLit(n) => {
            out.push_str(&format!("    movq ${}, %rax\n", n));
            Ok(TyKind::Int)
        }
        Expr::FloatLit(f) => {
            let label = locals.intern_f32(*f as f32);
            out.push_str(&format!("    movss {}(%rip), %xmm0\n", label));
            Ok(TyKind::F32)
        }
        Expr::StrLit(s) => {
            let label = data.intern(s);
            out.push_str(&format!("    leaq {}(%rip), %rax\n", label));
            Ok(TyKind::Int)
        }
        Expr::Ident(name) => {
            let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            let kind = locals.types.get(name).copied().unwrap_or(TyKind::Int);
            match kind {
                TyKind::Int => out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss -{}(%rbp), %xmm0\n", slot * 8)),
            }
            Ok(kind)
        }
        Expr::Bin { op: BinOp::Assign, lhs, rhs } => {
            let name = match lhs.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("LHS of assignment must be an ident")),
            };
            let slot = locals.get(&name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            let kind = locals.types.get(&name).copied().unwrap_or(TyKind::Int);
            let val_ty = emit_expr_value(rhs, out, data, locals)?;
            if val_ty != kind {
                return Err(AsmError::UnsupportedExpr(
                    "assignment type mismatch (Int/F32 must match the local's declared type)"));
            }
            match kind {
                TyKind::Int => out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
            }
            Ok(kind)
        }
        Expr::Bin { op, lhs, rhs } => {
            // Eval lhs first; pick the integer or float pipeline based on its type.
            let lhs_ty = emit_expr_value(lhs, out, data, locals)?;
            // Spill lhs to free up the result register for the rhs.
            match lhs_ty {
                TyKind::Int => out.push_str("    pushq %rax\n"),
                TyKind::F32 => {
                    // Reserve 16 bytes on the stack to keep alignment, store xmm0.
                    out.push_str("    subq $16, %rsp\n");
                    out.push_str("    movss %xmm0, (%rsp)\n");
                }
            }
            let rhs_ty = emit_expr_value(rhs, out, data, locals)?;
            if rhs_ty != lhs_ty {
                return Err(AsmError::UnsupportedExpr("Bin operands must be same type"));
            }
            match lhs_ty {
                TyKind::Int => {
                    out.push_str("    popq %r10\n");
                    out.push_str("    xchgq %rax, %r10\n");
                    match op {
                        BinOp::Add => { out.push_str("    addq %r10, %rax\n"); Ok(TyKind::Int) }
                        BinOp::Sub => { out.push_str("    subq %r10, %rax\n"); Ok(TyKind::Int) }
                        BinOp::Mul => { out.push_str("    imulq %r10, %rax\n"); Ok(TyKind::Int) }
                        BinOp::Div => { out.push_str("    cqo\n    idivq %r10\n"); Ok(TyKind::Int) }
                        BinOp::Mod => { out.push_str("    cqo\n    idivq %r10\n    movq %rdx, %rax\n"); Ok(TyKind::Int) }
                        BinOp::Eq => { emit_cmp(out, "sete");  Ok(TyKind::Int) }
                        BinOp::Ne => { emit_cmp(out, "setne"); Ok(TyKind::Int) }
                        BinOp::Lt => { emit_cmp(out, "setl");  Ok(TyKind::Int) }
                        BinOp::Gt => { emit_cmp(out, "setg");  Ok(TyKind::Int) }
                        BinOp::Le => { emit_cmp(out, "setle"); Ok(TyKind::Int) }
                        BinOp::Ge => { emit_cmp(out, "setge"); Ok(TyKind::Int) }
                        other => Err(AsmError::UnsupportedBinOp(*other)),
                    }
                }
                TyKind::F32 => {
                    // After rhs, xmm0 holds rhs and the spill slot holds lhs.
                    // Move rhs aside, reload lhs, then run the op.
                    out.push_str("    movss %xmm0, %xmm1\n");          // xmm1 = rhs
                    out.push_str("    movss (%rsp), %xmm0\n");          // xmm0 = lhs
                    out.push_str("    addq $16, %rsp\n");
                    match op {
                        // xmm0 op= xmm1
                        BinOp::Add => { out.push_str("    addss %xmm1, %xmm0\n"); Ok(TyKind::F32) }
                        BinOp::Sub => { out.push_str("    subss %xmm1, %xmm0\n"); Ok(TyKind::F32) }
                        BinOp::Mul => { out.push_str("    mulss %xmm1, %xmm0\n"); Ok(TyKind::F32) }
                        BinOp::Div => { out.push_str("    divss %xmm1, %xmm0\n"); Ok(TyKind::F32) }
                        // ucomiss sets ZF/CF based on xmm0 <=> xmm1; result is int bool.
                        BinOp::Eq => { out.push_str("    ucomiss %xmm1, %xmm0\n"); emit_setcc_int(out, "sete"); Ok(TyKind::Int) }
                        BinOp::Ne => { out.push_str("    ucomiss %xmm1, %xmm0\n"); emit_setcc_int(out, "setne"); Ok(TyKind::Int) }
                        BinOp::Lt => { out.push_str("    ucomiss %xmm1, %xmm0\n"); emit_setcc_int(out, "setb");  Ok(TyKind::Int) }
                        BinOp::Gt => { out.push_str("    ucomiss %xmm1, %xmm0\n"); emit_setcc_int(out, "seta");  Ok(TyKind::Int) }
                        BinOp::Le => { out.push_str("    ucomiss %xmm1, %xmm0\n"); emit_setcc_int(out, "setbe"); Ok(TyKind::Int) }
                        BinOp::Ge => { out.push_str("    ucomiss %xmm1, %xmm0\n"); emit_setcc_int(out, "setae"); Ok(TyKind::Int) }
                        other => Err(AsmError::UnsupportedBinOp(*other)),
                    }
                }
            }
        }
        Expr::Unary { op, expr } => {
            let kind = emit_expr_value(expr, out, data, locals)?;
            if kind != TyKind::Int {
                return Err(AsmError::UnsupportedExpr("unary op on non-int (f32 unary not yet wired)"));
            }
            match op {
                UnOp::Neg => out.push_str("    negq %rax\n"),
                UnOp::Not => {
                    out.push_str("    testq %rax, %rax\n");
                    out.push_str("    sete %al\n");
                    out.push_str("    movzbl %al, %eax\n");
                }
            }
            Ok(TyKind::Int)
        }
        Expr::Ref { expr, .. } => {
            match expr.as_ref() {
                Expr::Ident(name) => {
                    let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
                    out.push_str(&format!("    leaq -{}(%rbp), %rax\n", slot * 8));
                    Ok(TyKind::Int)
                }
                _ => Err(AsmError::UnsupportedExpr("`&` only supports a bare local")),
            }
        }
        Expr::If { cond, then, else_ } => {
            let else_label = locals.fresh_label("else");
            let end_label = locals.fresh_label("endif");
            let cond_ty = emit_expr_value(cond, out, data, locals)?;
            if cond_ty != TyKind::Int {
                return Err(AsmError::UnsupportedExpr("if condition must be int/bool"));
            }
            out.push_str("    testq %rax, %rax\n");
            out.push_str(&format!("    je {}\n", else_label));
            let then_ty = emit_block(then, out, data, locals)?;
            out.push_str(&format!("    jmp {}\n", end_label));
            out.push_str(&format!("{}:\n", else_label));
            let else_ty = if let Some(b) = else_ {
                emit_block(b, out, data, locals)?
            } else { TyKind::Int };
            out.push_str(&format!("{}:\n", end_label));
            if then_ty != else_ty {
                return Err(AsmError::UnsupportedExpr(
                    "if/else branches must have same type"));
            }
            Ok(then_ty)
        }
        Expr::Block(b) => emit_block(b, out, data, locals),
        Expr::While { cond, body } => {
            let top = locals.fresh_label("while_top");
            let end = locals.fresh_label("while_end");
            out.push_str(&format!("{}:\n", top));
            emit_expr_value(cond, out, data, locals)?;
            out.push_str("    testq %rax, %rax\n");
            out.push_str(&format!("    je {}\n", end));
            locals.loop_labels.push((top.clone(), end.clone()));
            emit_block(body, out, data, locals)?;
            locals.loop_labels.pop();
            out.push_str(&format!("    jmp {}\n", top));
            out.push_str(&format!("{}:\n", end));
            out.push_str("    xorl %eax, %eax\n");
            Ok(TyKind::Int)
        }
        Expr::Break => {
            let (_, end) = locals.loop_labels.last()
                .ok_or(AsmError::UnsupportedExpr("`break` outside a loop"))?
                .clone();
            out.push_str(&format!("    jmp {}\n", end));
            Ok(TyKind::Int)
        }
        Expr::Continue => {
            let (top, _) = locals.loop_labels.last()
                .ok_or(AsmError::UnsupportedExpr("`continue` outside a loop"))?
                .clone();
            out.push_str(&format!("    jmp {}\n", top));
            Ok(TyKind::Int)
        }
        Expr::For { var, iter, body, .. } => {
            // Only `lo..hi` is supported in the asm backend today.
            let (lo, hi) = match iter.as_ref() {
                Expr::Range { lo, hi, .. } => (lo.as_ref(), hi.as_ref()),
                _ => return Err(AsmError::UnsupportedExpr("for over non-range")),
            };
            // Evaluate hi first so its result lives in `_for_end_`.
            emit_expr_value(hi, out, data, locals)?;
            let end_slot = locals.alloc("_for_end_");
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", end_slot * 8));

            // i = lo
            emit_expr_value(lo, out, data, locals)?;
            let i_slot = locals.alloc(var);
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", i_slot * 8));

            let top = locals.fresh_label("for_top");
            let cont = locals.fresh_label("for_cont");
            let end = locals.fresh_label("for_end");
            out.push_str(&format!("{}:\n", top));
            // if i >= end goto end
            out.push_str(&format!("    movq -{}(%rbp), %rax\n", i_slot * 8));
            out.push_str(&format!("    movq -{}(%rbp), %r10\n", end_slot * 8));
            out.push_str("    cmpq %r10, %rax\n");
            out.push_str(&format!("    jge {}\n", end));

            locals.loop_labels.push((cont.clone(), end.clone()));
            emit_block(body, out, data, locals)?;
            locals.loop_labels.pop();
            out.push_str(&format!("{}:\n", cont));

            // i = i + 1
            out.push_str(&format!("    movq -{}(%rbp), %rax\n", i_slot * 8));
            out.push_str("    addq $1, %rax\n");
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", i_slot * 8));
            out.push_str(&format!("    jmp {}\n", top));
            out.push_str(&format!("{}:\n", end));
            out.push_str("    xorl %eax, %eax\n");
            Ok(TyKind::Int)
        }
        Expr::Call { callee, args } => {
            let name = match callee.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("non-ident callee")),
            };
            // Special-case: println(STR) → puts(STR). Phase 0 stays here so
            // the simplest hello-world keeps working.
            if name == "println" && args.len() == 1 {
                if let Expr::StrLit(s) = &args[0] {
                    let label = data.intern(s);
                    out.push_str(&format!("    leaq {}(%rip), %rcx\n", label));
                    out.push_str("    callq puts\n");
                    return Ok(TyKind::Int);
                }
            }
            if args.len() > 4 {
                return Err(AsmError::TooManyArgs);
            }
            for a in args {
                if matches!(a, Expr::Call { .. }) {
                    return Err(AsmError::NestedCallInArg);
                }
            }
            // Reverse-order eval. For now this is int-only — float arg passing
            // (xmm0..3) lands in a future bump.
            let arg_regs = ["%rcx", "%rdx", "%r8", "%r9"];
            for i in (0..args.len()).rev() {
                let kind = emit_expr_value(&args[i], out, data, locals)?;
                if kind != TyKind::Int {
                    return Err(AsmError::UnsupportedExpr("float args to FFI not yet supported"));
                }
                if arg_regs[i] != "%rax" {
                    out.push_str(&format!("    movq %rax, {}\n", arg_regs[i]));
                }
            }
            out.push_str(&format!("    callq {}\n", name));
            Ok(TyKind::Int)
        }
        _ => Err(AsmError::UnsupportedExpr("unhandled expr in asm backend")),
    }
}
