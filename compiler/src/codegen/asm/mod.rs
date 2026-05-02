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

use crate::ast::{BinOp, Block, Expr, FnDecl, Item, Program, Stmt, UnOp};

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

    for item in &p.items {
        if let Item::Fn(f) = item {
            if f.body.is_some() {
                emit_fn(f, &mut text, &mut data)?;
            }
        }
    }

    if !data.entries.is_empty() {
        s.push_str(".section .rdata,\"dr\"\n");
        for (label, bytes) in &data.entries {
            s.push_str(&format!("{}:\n", label));
            s.push_str(&format!("    .asciz \"{}\"\n", escape(bytes)));
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
    next_slot: usize,
    /// Counter for generating unique label names per function.
    label_counter: u32,
    /// Function name for label prefixing (so `.Lif_0_0` is unique across fns).
    fn_label_prefix: String,
    /// Stack of (continue_target, break_target) labels for nested loops.
    loop_labels: Vec<(String, String)>,
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
}

/// Emit a cmp + setcc + zero-extend sequence. Operands: rax = lhs, r10 = rhs;
/// flags after `cmpq %r10, %rax` reflect `lhs - rhs`.
fn emit_cmp(out: &mut String, setcc_mnem: &str) {
    out.push_str("    cmpq %r10, %rax\n");
    out.push_str(&format!("    {} %al\n", setcc_mnem));
    out.push_str("    movzbl %al, %eax\n");
}

fn emit_fn(f: &FnDecl, out: &mut String, data: &mut StringTable) -> Result<(), AsmError> {
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
    Ok(())
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
    -> Result<(), AsmError>
{
    for s in &b.stmts { emit_stmt(s, out, data, locals)?; }
    if let Some(tail) = &b.tail { emit_expr_value(tail, out, data, locals)?; }
    Ok(())
}

fn emit_stmt(s: &Stmt, out: &mut String, data: &mut StringTable, locals: &mut Locals)
    -> Result<(), AsmError>
{
    match s {
        Stmt::Expr(e) => emit_expr_value(e, out, data, locals),
        Stmt::Return(Some(e)) => {
            emit_expr_value(e, out, data, locals)?;
            // Use the same epilogue offset as the function header — the frame
            // size is captured implicitly via locals.frame_bytes() which the
            // caller wrote into the prologue. Re-emit it here.
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
        Stmt::Let { name, value, .. } => {
            emit_expr_value(value, out, data, locals)?;
            let slot = locals.alloc(name);
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8));
            Ok(())
        }
    }
}

/// Evaluate `e` and leave its result in %rax.
fn emit_expr_value(e: &Expr, out: &mut String, data: &mut StringTable, locals: &mut Locals)
    -> Result<(), AsmError>
{
    match e {
        Expr::IntLit(n) => {
            out.push_str(&format!("    movq ${}, %rax\n", n));
            Ok(())
        }
        Expr::StrLit(s) => {
            let label = data.intern(s);
            out.push_str(&format!("    leaq {}(%rip), %rax\n", label));
            Ok(())
        }
        Expr::Ident(name) => {
            let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8));
            Ok(())
        }
        Expr::Bin { op: BinOp::Assign, lhs, rhs } => {
            // Currently only `Ident = expr` is supported.
            let name = match lhs.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("LHS of assignment must be an ident")),
            };
            let slot = locals.get(&name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            emit_expr_value(rhs, out, data, locals)?;
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8));
            Ok(())
        }
        Expr::Bin { op, lhs, rhs } => {
            // eval lhs → rax, push; eval rhs → rax, pop r10, xchg → rax=lhs, r10=rhs.
            emit_expr_value(lhs, out, data, locals)?;
            out.push_str("    pushq %rax\n");
            emit_expr_value(rhs, out, data, locals)?;
            out.push_str("    popq %r10\n");
            out.push_str("    xchgq %rax, %r10\n");
            match op {
                BinOp::Add => out.push_str("    addq %r10, %rax\n"),
                BinOp::Sub => out.push_str("    subq %r10, %rax\n"),
                BinOp::Mul => out.push_str("    imulq %r10, %rax\n"),
                BinOp::Div => {
                    // rax = lhs, r10 = rhs.  cqo sign-extends rax → rdx:rax,
                    // idivq %r10 → quotient in rax, remainder in rdx.
                    out.push_str("    cqo\n");
                    out.push_str("    idivq %r10\n");
                }
                BinOp::Mod => {
                    out.push_str("    cqo\n");
                    out.push_str("    idivq %r10\n");
                    out.push_str("    movq %rdx, %rax\n");
                }
                BinOp::Eq => emit_cmp(out, "sete"),
                BinOp::Ne => emit_cmp(out, "setne"),
                BinOp::Lt => emit_cmp(out, "setl"),
                BinOp::Gt => emit_cmp(out, "setg"),
                BinOp::Le => emit_cmp(out, "setle"),
                BinOp::Ge => emit_cmp(out, "setge"),
                other => return Err(AsmError::UnsupportedBinOp(*other)),
            }
            Ok(())
        }
        Expr::Unary { op, expr } => {
            emit_expr_value(expr, out, data, locals)?;
            match op {
                UnOp::Neg => out.push_str("    negq %rax\n"),
                UnOp::Not => {
                    // Logical not: rax = (rax == 0) ? 1 : 0
                    out.push_str("    testq %rax, %rax\n");
                    out.push_str("    sete %al\n");
                    out.push_str("    movzbl %al, %eax\n");
                }
            }
            Ok(())
        }
        Expr::Ref { expr, .. } => {
            // `&local` — lea rax, slot(%rbp). Other forms aren't yet supported.
            match expr.as_ref() {
                Expr::Ident(name) => {
                    let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
                    out.push_str(&format!("    leaq -{}(%rbp), %rax\n", slot * 8));
                    Ok(())
                }
                _ => Err(AsmError::UnsupportedExpr("`&` only supports a bare local")),
            }
        }
        Expr::If { cond, then, else_ } => {
            let else_label = locals.fresh_label("else");
            let end_label = locals.fresh_label("endif");
            emit_expr_value(cond, out, data, locals)?;
            out.push_str("    testq %rax, %rax\n");
            // jz else_label  (i.e. if cond is false, jump to else)
            out.push_str(&format!("    je {}\n", else_label));
            emit_block(then, out, data, locals)?;
            out.push_str(&format!("    jmp {}\n", end_label));
            out.push_str(&format!("{}:\n", else_label));
            if let Some(b) = else_ {
                emit_block(b, out, data, locals)?;
            }
            out.push_str(&format!("{}:\n", end_label));
            Ok(())
        }
        Expr::Block(b) => {
            emit_block(b, out, data, locals)
        }
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
            Ok(())
        }
        Expr::Break => {
            let (_, end) = locals.loop_labels.last()
                .ok_or(AsmError::UnsupportedExpr("`break` outside a loop"))?
                .clone();
            out.push_str(&format!("    jmp {}\n", end));
            Ok(())
        }
        Expr::Continue => {
            let (top, _) = locals.loop_labels.last()
                .ok_or(AsmError::UnsupportedExpr("`continue` outside a loop"))?
                .clone();
            out.push_str(&format!("    jmp {}\n", top));
            Ok(())
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
            // For loops produce 0 as their value (to keep emit_expr_value happy).
            out.push_str("    xorl %eax, %eax\n");
            Ok(())
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
                    return Ok(());
                }
            }
            if args.len() > 4 {
                return Err(AsmError::TooManyArgs);
            }
            // Reject nested calls in args to keep clobbering manageable.
            for a in args {
                if matches!(a, Expr::Call { .. }) {
                    return Err(AsmError::NestedCallInArg);
                }
            }
            // Eval args in reverse so loading an earlier arg into rcx doesn't
            // get clobbered by evaluating a later one (later args use higher
            // arg regs r9/r8/rdx, which the simpler eval paths don't touch).
            let arg_regs = ["%rcx", "%rdx", "%r8", "%r9"];
            for i in (0..args.len()).rev() {
                emit_expr_value(&args[i], out, data, locals)?;
                if arg_regs[i] != "%rax" {
                    out.push_str(&format!("    movq %rax, {}\n", arg_regs[i]));
                }
            }
            out.push_str(&format!("    callq {}\n", name));
            Ok(())
        }
        _ => Err(AsmError::UnsupportedExpr("unhandled expr in asm backend")),
    }
}
