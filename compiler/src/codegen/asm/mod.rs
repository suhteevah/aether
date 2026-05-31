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

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::ast::{BinOp, Block, Expr, FnDecl, Item, MatchPat, Program, ShapeDim, Stmt, StructDecl, Ty, UnOp};

/// Shared state for const-generic monomorphization. Filled in `try_emit` and
/// shared across all per-fn `Locals` via `Rc<RefCell<…>>`.
///
/// * `templates`  — name → FnDecl for every fn with `const_params.len() > 0`.
///                  These are NOT emitted directly; each call site triggers a
///                  specialization.
/// * `pending`    — worklist of (template_name, sorted bindings, mangled_name)
///                  awaiting emission. Drained by `try_emit` after the initial
///                  fn loop and after each spec emit.
/// * `seen`       — mangled names already emitted (or queued) so we don't
///                  duplicate work or re-mangle.
#[derive(Default)]
struct GenericState {
    templates: HashMap<String, FnDecl>,
    /// (template, const(shape) bindings, TYPE bindings, mangled name). Type
    /// bindings map a type param `T` → a concrete type name ("i64"/"f32"/…)
    /// for type-generic fns (`fn id<T>(x: T) -> T`).
    pending: Vec<(String, Vec<(String, i64)>, Vec<(String, String)>, String)>,
    seen: HashSet<String>,
}

/// Concrete type name of a call argument, for type-generic inference.
fn arg_concrete_type_name(arg: &Expr, locals: &Locals) -> Option<String> {
    match arg {
        Expr::IntLit(_) | Expr::BoolLit(_) => Some("i64".into()),
        Expr::FloatLit(_) => Some(match locals.default_float {
            Some(TyKind::F64) => "f64", _ => "f32" }.into()),
        Expr::Ident(n) => locals.types.get(n).map(|k| match k {
            TyKind::F32 => "f32".to_string(),
            TyKind::F64 => "f64".to_string(),
            _ => "i64".to_string(),
        }),
        Expr::Cast { ty, .. } => Some(ty.clone()),
        _ => None,
    }
}

/// Substitute a type param `param` → concrete type name throughout a fn's
/// signature + body (let annotations + cast targets). Used to build a
/// type-generic specialization.
fn subst_type_param_fn(f: &mut FnDecl, param: &str, concrete: &str) {
    for p in f.params.iter_mut() { subst_type_in_ty(&mut p.ty, param, concrete); }
    if let Some(r) = f.ret.as_mut() { subst_type_in_ty(r, param, concrete); }
    if let Some(b) = f.body.as_mut() { subst_type_in_block(b, param, concrete); }
}
fn subst_type_in_ty(ty: &mut Ty, param: &str, concrete: &str) {
    match ty {
        Ty::Named(n) if n == param => *n = concrete.to_string(),
        Ty::Ref { inner, .. } => subst_type_in_ty(inner, param, concrete),
        Ty::Slice { elem, .. } => subst_type_in_ty(elem, param, concrete),
        Ty::Array { elem, .. } => subst_type_in_ty(elem, param, concrete),
        Ty::Tuple(es) => for e in es { subst_type_in_ty(e, param, concrete); },
        Ty::Generic { args, .. } => for a in args { subst_type_in_ty(a, param, concrete); },
        _ => {}
    }
}
fn subst_type_in_block(b: &mut Block, param: &str, concrete: &str) {
    for s in b.stmts.iter_mut() {
        match s {
            Stmt::Let { ty, value, .. } => {
                if let Some(t) = ty.as_mut() { subst_type_in_ty(t, param, concrete); }
                if let Some(e) = value.as_mut() { subst_type_in_expr(e, param, concrete); }
            }
            Stmt::LetTuple { value, .. } => subst_type_in_expr(value, param, concrete),
            Stmt::Expr(e) | Stmt::Return(Some(e)) => subst_type_in_expr(e, param, concrete),
            _ => {}
        }
    }
    if let Some(t) = b.tail.as_mut() { subst_type_in_expr(t, param, concrete); }
}
fn subst_type_in_expr(e: &mut Expr, param: &str, concrete: &str) {
    match e {
        Expr::Cast { ty, expr } => {
            if ty == param { *ty = concrete.to_string(); }
            subst_type_in_expr(expr, param, concrete);
        }
        Expr::Block(b) => subst_type_in_block(b, param, concrete),
        Expr::If { cond, then, else_ } => {
            subst_type_in_expr(cond, param, concrete);
            subst_type_in_block(then, param, concrete);
            if let Some(eb) = else_ { subst_type_in_block(eb, param, concrete); }
        }
        Expr::For { iter, body, .. } => { subst_type_in_expr(iter, param, concrete); subst_type_in_block(body, param, concrete); }
        Expr::While { cond, body } => { subst_type_in_expr(cond, param, concrete); subst_type_in_block(body, param, concrete); }
        Expr::Region { body, .. } => subst_type_in_block(body, param, concrete),
        Expr::Call { callee, args } => { subst_type_in_expr(callee, param, concrete); for a in args { subst_type_in_expr(a, param, concrete); } }
        Expr::MethodCall { recv, args, .. } => { subst_type_in_expr(recv, param, concrete); for a in args { subst_type_in_expr(a, param, concrete); } }
        Expr::Bin { lhs, rhs, .. } => { subst_type_in_expr(lhs, param, concrete); subst_type_in_expr(rhs, param, concrete); }
        Expr::Unary { expr, .. } | Expr::Ref { expr, .. } | Expr::Deref(expr) | Expr::Try(expr) => subst_type_in_expr(expr, param, concrete),
        Expr::Field { recv, .. } => subst_type_in_expr(recv, param, concrete),
        Expr::Index { recv, idx } => { subst_type_in_expr(recv, param, concrete); subst_type_in_expr(idx, param, concrete); }
        _ => {}
    }
}

/// Where the value of an expression lives after evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TyKind {
    Int, F32, F64,
    /// `Tensor<f32, [N]>`. Stored at an `i64` handle (returned by
    /// `aether_dev_alloc_f32(N)`); auto-freed at fn natural end.
    /// Behaves like `Int` everywhere a value is read, plus carries the
    /// element count so the prologue allocates and the epilogue frees.
    TensorDev(usize),
    /// `Tensor<i32, [N]>` — same shape, i32 elements (labels).
    TensorDevI32(usize),
}

impl TyKind {
    fn from_ty(t: &Ty) -> Option<TyKind> {
        match t {
            Ty::Named(n) if n == "f32" => Some(TyKind::F32),
            Ty::Named(n) if n == "f64" => Some(TyKind::F64),
            Ty::Named(n) if matches!(n.as_str(), "i32" | "i64" | "u32" | "u64" | "bool") => Some(TyKind::Int),
            // Tensor with all-Const shape dims. Symbolic dims (`[BSZ, KK]`)
            // require a const env which `from_ty` doesn't have access to;
            // the Stmt::Let path uses `from_ty_with_env` for that case.
            Ty::Generic { name, args } if name == "Tensor" && args.len() == 2 => {
                let count = tensor_shape_const(&args[1], None)?.iter().product::<usize>();
                match &args[0] {
                    Ty::Named(e) if e == "f32" => Some(TyKind::TensorDev(count)),
                    Ty::Named(e) if e == "i32" => Some(TyKind::TensorDevI32(count)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
    fn is_float(self) -> bool { matches!(self, TyKind::F32 | TyKind::F64) }
    fn is_handle(self) -> bool { matches!(self, TyKind::TensorDev(_) | TyKind::TensorDevI32(_)) }

    /// Same as `from_ty` but resolves symbolic shape dims through `const_env`.
    /// Used by the `Stmt::Let` Tensor path so `let x: Tensor<f32, [BSZ, KK]>;`
    /// works with file-level `const BSZ: i32 = 8;` decls.
    fn from_ty_with_env(t: &Ty, const_env: &HashMap<String, i64>) -> Option<TyKind> {
        if let Ty::Generic { name, args } = t {
            if name == "Tensor" && args.len() == 2 {
                let count = tensor_shape_const(&args[1], Some(const_env))?.iter().product::<usize>();
                return match &args[0] {
                    Ty::Named(e) if e == "f32" => Some(TyKind::TensorDev(count)),
                    Ty::Named(e) if e == "i32" => Some(TyKind::TensorDevI32(count)),
                    _ => None,
                };
            }
        }
        TyKind::from_ty(t)
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
    emit_with_plan(p, &Default::default())
}

/// Like `emit` but consumes a per-fn register-assignment plan produced by
/// `mir::regalloc_plan::plan_program`. At `--O0` callers pass an empty map
/// (the default) and the asm output is byte-identical to today. At `--O1+`
/// callers pass the planner's output; hot locals are promoted into callee-
/// saved r12..r15 across loop bodies / repeated reads.
pub fn emit_with_plan(p: &Program, plan: &crate::mir::regalloc_plan::PlanMap) -> String {
    let s = match try_emit(p, plan) {
        Ok(s) => s,
        Err(e) => format!("# asm backend error: {:?}\n", e),
    };
    let s = peephole(&s);
    // P10.5 — instruction scheduling (load-store reorder). Compose after the
    // peephole pass so we reorder the leaner survivors, not the round-trip
    // forms peephole would otherwise collapse.
    let s = schedule(&s);
    // P10.8 — block layout: hot/cold splitting. Functions carrying `#[cold]`
    // are moved into a `.text.cold` section so the loader keeps them off the
    // hot I-cache.
    let mut cold_fns: HashSet<String> = HashSet::new();
    for item in &p.items {
        if let Item::Fn(f) = item {
            if f.body.is_some() && f.attrs.iter().any(|a| a.name == "cold") {
                let label = if f.name == "main" {
                    "main".to_string()
                } else {
                    format!("aether_{}", f.name)
                };
                cold_fns.insert(label);
            }
        }
    }
    block_layout(&s, &cold_fns)
}

/// Roadmap P10.8 — block layout / hot-cold splitting.
///
/// Splits the post-peephole assembly into per-fn blocks (boundaries are top-
/// level labels matching `main:` or `aether_<name>:` at column 0) and wraps
/// each block whose label is in `cold_fns` with section directives that move
/// it into `.text.cold`. The default `.text` section is restored after each
/// cold block so subsequent fns continue to land in the hot section.
fn block_layout(asm: &str, cold_fns: &HashSet<String>) -> String {
    if cold_fns.is_empty() {
        return asm.to_string();
    }
    let mut out = String::with_capacity(asm.len() + cold_fns.len() * 64);
    let lines: Vec<&str> = asm.split('\n').collect();
    // Find the start of every fn block (label at column 0 ending with ':',
    // not a directive, not indented).
    let is_fn_label = |s: &str| -> Option<String> {
        if s.starts_with(' ') || s.starts_with('\t') { return None; }
        if s.starts_with('.') || s.starts_with('#') { return None; }
        let trimmed = s.trim_end();
        let label = trimmed.strip_suffix(':')?;
        // No spaces, no tabs — labels are single tokens.
        if label.is_empty() || label.contains(|c: char| c.is_whitespace()) {
            return None;
        }
        Some(label.to_string())
    };

    // Walk lines, tracking whether we are currently inside a cold block. When
    // we hit a new fn label we (a) close the prior cold section if needed,
    // and (b) open a new cold section if the new label is cold.
    let mut in_cold = false;
    for (i, line) in lines.iter().enumerate() {
        if let Some(label) = is_fn_label(line) {
            // Closing the previous cold block, if any.
            if in_cold {
                out.push_str(".section .text,\"x\"\n");
                in_cold = false;
            }
            if cold_fns.contains(&label) {
                out.push_str(".section .text.cold,\"x\"\n");
                in_cold = true;
            }
        }
        out.push_str(line);
        // Re-add newline except after the final element produced by split.
        if i + 1 < lines.len() {
            out.push('\n');
        }
    }
    if in_cold {
        out.push_str("\n.section .text,\"x\"\n");
    }
    out
}

/// Roadmap P10.4 — asm-level peephole optimizer.
///
/// Line-based pattern match over the AT&T assembly the backend just produced.
/// Conservative: only collapses windows where every line exactly matches the
/// expected pattern, and never touches labels, directives (.global/.section/
/// .quad/.asciz/etc), or comments. Two patterns today:
///
/// 1. `movq $imm, %rax` + `movq %rax, -N(%rbp)`
///    → `movq $imm, -N(%rbp)`
///    Eliminates the rax round-trip when the immediate fits in 32 bits
///    (i.e. when the original `movq $imm, %rax` was already sign-extended
///    imm32, which is what aetherc emits).
///
/// 2. `movq %rax, -N(%rbp)` + `movq -N(%rbp), %rax`
///    → `movq %rax, -N(%rbp)`
///    Eliminates the redundant reload — rax already holds the value just
///    written, so re-loading it is a no-op.
fn peephole(asm: &str) -> String {
    // Split preserving exact line content; we'll re-join with '\n'. The input
    // ends with '\n' on every emitter path so we drop the trailing empty
    // sentinel produced by split and re-add a final newline at the end.
    let lines: Vec<&str> = asm.split('\n').collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());

    let mut i = 0usize;
    while i < lines.len() {
        let cur = lines[i];
        // Try 2-line windows. Both lookups guarded.
        if i + 1 < lines.len() {
            let next = lines[i + 1];

            // Pattern 1: movq $imm, %rax  /  movq %rax, -N(%rbp)
            if let Some(imm) = parse_mov_imm_to_rax(cur) {
                if let Some(disp) = parse_mov_rax_to_rbp_disp(next) {
                    // Preserve indentation from the first line so the file
                    // visually lines up.
                    let indent = leading_ws(cur);
                    out.push(format!("{}movq ${}, {}(%rbp)", indent, imm, disp));
                    i += 2;
                    continue;
                }
            }

            // Pattern 2: movq %rax, -N(%rbp)  /  movq -N(%rbp), %rax
            if let Some(d1) = parse_mov_rax_to_rbp_disp(cur) {
                if let Some(d2) = parse_mov_rbp_disp_to_rax(next) {
                    if d1 == d2 {
                        // Keep the store, drop the redundant reload.
                        out.push(cur.to_string());
                        i += 2;
                        continue;
                    }
                }
            }
        }

        out.push(cur.to_string());
        i += 1;
    }

    out.join("\n")
}

fn leading_ws(s: &str) -> &str {
    let end = s.bytes().position(|b| b != b' ' && b != b'\t').unwrap_or(s.len());
    &s[..end]
}

/// Parse `    movq $<imm>, %rax`. Returns the immediate as a signed i64 if the
/// line matches exactly that shape (plus optional whitespace). Anything else
/// (different reg, different op, label, directive, comment, trailing noise)
/// returns None.
fn parse_mov_imm_to_rax(line: &str) -> Option<i64> {
    let t = line.trim();
    let rest = t.strip_prefix("movq ")?;
    let (a, b) = split_two(rest)?;
    if b.trim() != "%rax" { return None; }
    let a = a.trim();
    let imm_str = a.strip_prefix('$')?;
    imm_str.parse::<i64>().ok()
}

/// Parse `    movq %rax, -<N>(%rbp)`. Returns the disp (as a signed i32) when
/// matched. Both negative and positive disps accepted.
fn parse_mov_rax_to_rbp_disp(line: &str) -> Option<i32> {
    let t = line.trim();
    let rest = t.strip_prefix("movq ")?;
    let (a, b) = split_two(rest)?;
    if a.trim() != "%rax" { return None; }
    parse_rbp_disp(b.trim())
}

/// Parse `    movq -<N>(%rbp), %rax`.
fn parse_mov_rbp_disp_to_rax(line: &str) -> Option<i32> {
    let t = line.trim();
    let rest = t.strip_prefix("movq ")?;
    let (a, b) = split_two(rest)?;
    if b.trim() != "%rax" { return None; }
    parse_rbp_disp(a.trim())
}

/// Parse `<disp>(%rbp)`. disp may be negative or zero.
fn parse_rbp_disp(s: &str) -> Option<i32> {
    let s = s.strip_suffix("(%rbp)")?;
    s.parse::<i32>().ok()
}

/// Split a single-comma operand list. None if there isn't exactly one comma at
/// top-level (we never encounter parens-wrapped commas in these instr forms).
fn split_two(s: &str) -> Option<(&str, &str)> {
    let idx = s.find(',')?;
    let (a, b) = s.split_at(idx);
    Some((a, &b[1..]))
}

/// Roadmap P10.5 — instruction scheduling (load-store reorder).
///
/// Conservative list-scheduler that runs after peephole. Within a maximal run
/// of plain rbp-relative load/store moves (no labels, branches, calls, or any
/// other instruction kind), it interleaves paired loads and stores:
///
///   load A→r1; store r1→B; load C→r2; store r2→D
/// becomes
///   load A→r1; load C→r2; store r1→B; store r2→D
///
/// Hides load latency (a 4-5-cycle L1 hit on Skylake/Zen 4) behind the
/// independent second load instead of stalling on the dependent store.
///
/// Safety: the runs contain ONLY rbp-disp loads and stores; reordering is
/// only emitted when the two pairs use distinct slots A,B,C,D and distinct
/// destination registers, so neither store can affect the other load and
/// neither load can clobber the other's value.
fn schedule(asm: &str) -> String {
    let lines: Vec<&str> = asm.split('\n').collect();
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut i = 0usize;
    while i < lines.len() {
        let run_start = i;
        while i < lines.len() && is_schedulable(lines[i]) {
            i += 1;
        }
        if i >= run_start + 4 {
            let run: Vec<&str> = lines[run_start..i].to_vec();
            let scheduled = reorder_run(&run);
            out.extend(scheduled);
        } else {
            for j in run_start..i {
                out.push(lines[j].to_string());
            }
        }
        if i < lines.len() {
            out.push(lines[i].to_string());
            i += 1;
        }
    }
    out.join("\n")
}

fn is_schedulable(line: &str) -> bool {
    parse_load_rbp(line).is_some() || parse_store_rbp(line).is_some()
}

fn parse_load_rbp(line: &str) -> Option<(i32, &str)> {
    let t = line.trim();
    let rest = t.strip_prefix("movq ")?;
    let (a, b) = split_two(rest)?;
    let a = a.trim();
    let b = b.trim();
    let reg = b.strip_prefix('%')?;
    if !is_plain_gpr(reg) { return None; }
    let disp = parse_rbp_disp(a)?;
    Some((disp, reg))
}

fn parse_store_rbp(line: &str) -> Option<(&str, i32)> {
    let t = line.trim();
    let rest = t.strip_prefix("movq ")?;
    let (a, b) = split_two(rest)?;
    let a = a.trim();
    let b = b.trim();
    let reg = a.strip_prefix('%')?;
    if !is_plain_gpr(reg) { return None; }
    let disp = parse_rbp_disp(b)?;
    Some((reg, disp))
}

fn is_plain_gpr(r: &str) -> bool {
    matches!(r, "rax" | "rbx" | "rcx" | "rdx" | "rsi" | "rdi"
        | "r8" | "r9" | "r10" | "r11" | "r12" | "r13" | "r14" | "r15")
}

fn reorder_run(run: &[&str]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(run.len());
    let mut i = 0usize;
    while i < run.len() {
        if i + 3 < run.len() {
            let p1l = parse_load_rbp(run[i]);
            let p1s = parse_store_rbp(run[i + 1]);
            let p2l = parse_load_rbp(run[i + 2]);
            let p2s = parse_store_rbp(run[i + 3]);
            if let (Some((da, ra)), Some((rs1, db)), Some((dc, rc)), Some((rs2, dd))) =
                (p1l, p1s, p2l, p2s)
            {
                let pair1_ok = ra == rs1;
                let pair2_ok = rc == rs2;
                let regs_distinct = ra != rc;
                let slots_distinct = da != db && da != dc && da != dd
                    && db != dc && db != dd && dc != dd;
                if pair1_ok && pair2_ok && regs_distinct && slots_distinct {
                    out.push(run[i].to_string());
                    out.push(run[i + 2].to_string());
                    out.push(run[i + 1].to_string());
                    out.push(run[i + 3].to_string());
                    i += 4;
                    continue;
                }
            }
        }
        out.push(run[i].to_string());
        i += 1;
    }
    out
}

pub fn try_emit(p: &Program, plan: &crate::mir::regalloc_plan::PlanMap) -> Result<String, AsmError> {
    let mut s = String::new();
    s.push_str("# AETHER x86-64 assembly (Microsoft x64 ABI)\n");
    s.push_str("# Emitted by aetherc; comments here are debug-only and do not\n");
    s.push_str("# come from any .aether source — those were stripped at lex time.\n\n");

    let mut data = StringTable::default();
    let mut text = String::new();
    let mut all_floats: Vec<(String, f32)> = Vec::new();
    let mut all_f64s: Vec<(String, f64)> = Vec::new();

    // Build a fn-name → return-TyKind map so call sites know which register
    // the result lives in (rax for Int, xmm0 for F32/F64). Both extern and
    // local fn decls go in. For local fns the linker name is `aether_<name>`;
    // for externs it's the bare name.
    // Expand `impl Foo { fn bar(...) ... }` into top-level
    // `fn Foo__bar(...)` entries before further codegen processing. The
    // dispatcher in `Expr::MethodCall` looks up `<TypeName>__<method>`
    // when the receiver is a local of struct type Foo. This is a
    // name-mangling lowering; no machinery for self-deref is needed —
    // `self` becomes a regular first param and field access uses the
    // existing struct-field machinery (slot-flat layout) which only works
    // when the caller passes the struct's flat-slot tail rather than an
    // address. For first cut, methods that take `self` by value (no `&`)
    // copy field-by-field into the callee's frame; methods that take
    // `&self` get a pointer (Phase-2 needs deref+offset addressing in
    // the asm backend, deferred).
    let mut p = p.clone();
    {
        use crate::ast::FnDecl;
        let mut new_items = Vec::with_capacity(p.items.len());
        for item in p.items {
            match item {
                Item::Impl { type_name, methods } => {
                    for mut m in methods {
                        m.name = format!("{}__{}", type_name, m.name);
                        new_items.push(Item::Fn(m));
                    }
                    let _: Option<FnDecl> = None;
                }
                // P12.1 — `impl Trait for Type` flattens the same way as inherent
                // impl. The trait_name is currently informational; mir::traits
                // resolves dispatch.
                Item::ImplTrait { type_name, methods, .. } => {
                    for mut m in methods {
                        m.name = format!("{}__{}", type_name, m.name);
                        new_items.push(Item::Fn(m));
                    }
                }
                // Trait declarations don't emit code — they declare an
                // interface. Bodies (default methods) live in the
                // `ImplTrait` blocks that satisfy them.
                Item::Trait { .. } => {}
                other => new_items.push(other),
            }
        }
        p.items = new_items;
    }
    let p = &p;

    let mut sigs: HashMap<String, TyKind> = HashMap::new();
    let mut local_fns: HashSet<String> = HashSet::new();
    /// Fn name → payload-enum name, for fns whose declared return type is a
    /// payload-carrying enum (Result-shaped).  Used to drive the 2-register
    /// return ABI: tag in %rax, value in %rdx. Indexed by both the bare fn
    /// name and the linker name (`aether_<n>`) for caller-side lookup
    /// symmetry.
    let mut fn_returns_enum: HashMap<String, String> = HashMap::new();
    // P6.5 — fns whose declared return is a small (≤2 i64-field) struct use
    // the 2-register return ABI (field0 → %rax, field1 → %rdx). Populated after
    // struct_decls is fully built (a fn may precede its return struct's decl).
    let mut fn_returns_struct: HashMap<String, String> = HashMap::new();
    let mut struct_decls: HashMap<String, StructDecl> = HashMap::new();
    let mut enum_decls: HashMap<String, EnumDecl> = HashMap::new();
    let mut const_env: HashMap<String, i64> = HashMap::new();
    let generics: Rc<RefCell<GenericState>> = Rc::new(RefCell::new(GenericState::default()));
    for item in &p.items {
        if let Item::Const(cd) = item {
            if let Expr::IntLit(n) = &cd.value {
                const_env.insert(cd.name.clone(), *n);
            }
        }
        // Enum variants enter the const env as `<EnumName>::<Variant>`
        // → tag-as-i64. So `Color::Red` lowered as `Expr::Path(["Color",
        // "Red"])` resolves at the Path-codegen site to the same int
        // any other const lookup gives.
        if let Item::Enum { name, variants, payloads } = item {
            for (i, v) in variants.iter().enumerate() {
                const_env.insert(format!("{}::{}", name, v), i as i64);
            }
            // Register payload-aware variants for the codegen rewrites below.
            let has_any_payload = payloads.iter().any(|p| p.is_some());
            if has_any_payload {
                enum_decls.insert(name.clone(), EnumDecl {
                    variants: variants.clone(),
                    payloads: payloads.clone(),
                });
            }
        }
    }
    for item in &p.items {
        if let Item::Struct(sd) = item {
            struct_decls.insert(sd.name.clone(), sd.clone());
        }
        if let Item::Fn(f) = item {
            // Templates (const-generic fns) are NOT registered as local_fns
            // by their bare name — call sites resolve to mangled specializations
            // instead. They go into `generics.templates` and are emitted lazily.
            if f.body.is_some() && !f.const_params.is_empty() {
                generics.borrow_mut().templates.insert(f.name.clone(), f.clone());
                continue;
            }
            if f.body.is_some() && f.name != "main" {
                local_fns.insert(f.name.clone());
            }
            if let Some(rk) = f.ret.as_ref().and_then(TyKind::from_ty) {
                let linker_name = if f.is_extern || f.body.is_none() {
                    f.name.clone()
                } else {
                    format!("aether_{}", f.name)
                };
                sigs.insert(linker_name, rk);
                // Also let the bare name resolve so source-level call sites
                // (which use the unmangled name) work.
                sigs.insert(f.name.clone(), rk);
            }
            // Detect fns whose declared return type is a payload-enum.
            // These use the 2-register return ABI (tag in %rax, value in
            // %rdx) and drive the `?`-operator early-return propagation.
            if let Some(Ty::Named(rname)) = f.ret.as_ref() {
                if enum_decls.contains_key(rname) {
                    fn_returns_enum.insert(f.name.clone(), rname.clone());
                    fn_returns_enum.insert(format!("aether_{}", f.name), rname.clone());
                }
            }
        }
    }

    // P6.5 — now that struct_decls is complete, detect fns returning a small
    // struct (1 or 2 i64/Int fields → rax:rdx). Larger structs / float fields
    // need an sret hidden-pointer ABI (follow-up) and are intentionally left
    // out so they still hit the clear "struct literal must appear…" error.
    for item in &p.items {
        if let Item::Fn(f) = item {
            if let Some(Ty::Named(rname)) = f.ret.as_ref() {
                if let Some(sd) = struct_decls.get(rname) {
                    let small = (1..=2).contains(&sd.fields.len())
                        && sd.fields.iter().all(|fld|
                            matches!(TyKind::from_ty(&fld.ty), Some(TyKind::Int)));
                    if small {
                        fn_returns_struct.insert(f.name.clone(), rname.clone());
                        fn_returns_struct.insert(format!("aether_{}", f.name), rname.clone());
                    }
                }
            }
        }
    }

    // Pre-register every template's return TyKind under both its bare name and
    // every future mangled name's bare prefix — call sites read sigs by name
    // before the spec exists, so without this the call would default to Int.
    for (tname, tdecl) in generics.borrow().templates.iter() {
        if let Some(rk) = tdecl.ret.as_ref().and_then(TyKind::from_ty) {
            sigs.insert(tname.clone(), rk);
        }
    }

    for item in &p.items {
        if let Item::Fn(f) = item {
            if f.body.is_some() && f.const_params.is_empty() {
                let fn_plan = plan.get(&f.name);
                let (floats, f64s) = emit_fn(
                    f, &mut text, &mut data, &sigs, &local_fns,
                    &struct_decls, &enum_decls, &fn_returns_enum, &fn_returns_struct,
                    &const_env, Some(generics.clone()), fn_plan)?;
                all_floats.extend(floats);
                all_f64s.extend(f64s);
            }
        }
    }

    // Drain the spec worklist. Each spec emit may queue more specs (cascading
    // through other templates), so we loop until stable. Bound the work at a
    // generous ceiling so a pathological recursive template doesn't spin.
    let mut iters = 0usize;
    loop {
        iters += 1;
        if iters > 10_000 {
            return Err(AsmError::UnsupportedExpr("monomorphization runaway"));
        }
        let next = generics.borrow_mut().pending.pop();
        let Some((tname, bindings, type_bindings, mangled)) = next else { break; };
        let tdecl = generics.borrow().templates.get(&tname).cloned()
            .ok_or(AsmError::UnsupportedExpr("monomorphization: unknown template"))?;
        // Build the specialized FnDecl: rename, drop const_params (it's now
        // concrete), keep everything else intact. The shape Sym dims in
        // its param/return types still reference the const-param names; the
        // emit_fn we call below resolves them through the extended const_env.
        let mut spec = tdecl.clone();
        spec.name = mangled.clone();
        spec.const_params.clear();
        // Type-generic substitution: replace each type param `T` → its bound
        // concrete type throughout the spec's signature + body, so the spec
        // codegens with concrete types (correct storage class per instantiation).
        for (param, concrete) in &type_bindings {
            subst_type_param_fn(&mut spec, param, concrete);
        }
        let mut spec_env = const_env.clone();
        for (k, v) in &bindings { spec_env.insert(k.clone(), *v); }
        // Register the specialization so cascading calls resolve.
        local_fns.insert(mangled.clone());
        if let Some(rk) = spec.ret.as_ref().and_then(|t| TyKind::from_ty_with_env(t, &spec_env)) {
            sigs.insert(mangled.clone(), rk);
            sigs.insert(format!("aether_{}", mangled), rk);
        }
        // Const-generic specializations don't get a planned reg map (the
        // planner runs on the source program; specs are synthesised post-plan).
        // Future work: re-plan each spec on the fly. For now they stay on the
        // stack path, which is the same conservative default as --O0.
        let (floats, f64s) = emit_fn(
            &spec, &mut text, &mut data, &sigs, &local_fns,
            &struct_decls, &enum_decls, &fn_returns_enum, &fn_returns_struct,
            &spec_env, Some(generics.clone()), None)?;
        all_floats.extend(floats);
        all_f64s.extend(f64s);
    }

    if !data.entries.is_empty() || !all_floats.is_empty() || !all_f64s.is_empty() {
        s.push_str(".section .rdata,\"dr\"\n");
        for (label, bytes) in &data.entries {
            s.push_str(&format!("{}:\n", label));
            s.push_str(&format!("    .asciz \"{}\"\n", escape(bytes)));
        }
        for (label, v) in &all_floats {
            s.push_str(&format!("{}:\n", label));
            // Emit raw f32 bytes via .byte to stay within our assembler's parser surface.
            let bits = v.to_bits();
            for i in 0..4 {
                s.push_str(&format!("    .byte 0x{:02x}\n", (bits >> (i * 8)) & 0xff));
            }
        }
        for (label, v) in &all_f64s {
            s.push_str(&format!("{}:\n", label));
            // Emit raw f64 bit pattern via .quad — our assembler recognises it.
            s.push_str(&format!("    .quad 0x{:016x}\n", v.to_bits()));
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

#[derive(Clone)]
struct EnumDecl {
    variants: Vec<String>,
    payloads: Vec<Option<Ty>>,
}

/// If `e` is `EnumName::Variant(arg)` (`Call(Path([..2]), [arg])`) or the
/// no-arg form `EnumName::Variant` (`Path([..2])`), and the path resolves
/// to a known payload-enum, return (enum_name, variant_idx, payload_expr).
fn resolve_enum_ctor(
    e: &Expr,
    enum_decls: &HashMap<String, EnumDecl>,
) -> Option<(String, usize, Option<Expr>)> {
    let (path, arg) = match e {
        Expr::Call { callee, args } => {
            if let Expr::Path(p) = callee.as_ref() {
                if p.len() != 2 || args.len() != 1 { return None; }
                (p.clone(), Some(args[0].clone()))
            } else { return None; }
        }
        Expr::Path(p) if p.len() == 2 => (p.clone(), None),
        _ => return None,
    };
    let decl = enum_decls.get(&path[0])?;
    let idx = decl.variants.iter().position(|v| *v == path[1])?;
    Some((path[0].clone(), idx, arg))
}

/// If `e` is a `Call { callee: Ident(fn_name), .. }` where `fn_name` is a
/// fn that returns a payload-enum (registered in `fn_returns_enum`), return
/// the enum's name. Drives the 2-register return ABI on the caller side.
fn call_returns_enum(e: &Expr, fn_returns_enum: &HashMap<String, String>) -> Option<String> {
    if let Expr::Call { callee, .. } = e {
        if let Expr::Ident(n) = callee.as_ref() {
            return fn_returns_enum.get(n).cloned();
        }
    }
    None
}

/// If `e` is a call to a fn that returns a small struct (registered in
/// `fn_returns_struct`), return the struct's name. Drives the 2-register
/// struct-return ABI on the caller side. P6.5.
fn call_returns_struct(e: &Expr, fn_returns_struct: &HashMap<String, String>) -> Option<String> {
    if let Expr::Call { callee, .. } = e {
        if let Expr::Ident(n) = callee.as_ref() {
            return fn_returns_struct.get(n).cloned();
        }
    }
    None
}

/// Emit a small struct's fields into (%rax = field0, %rdx = field1) for the
/// struct-return ABI. `fields` are the literal's (name, expr) pairs; `sd`
/// gives the declared field order. field1 is evaluated and parked in rdx
/// first (via a push) so evaluating field0 can't clobber it. P6.5.
fn emit_struct_return_value(
    lit_fields: &[(String, Expr)],
    sd: &StructDecl,
    out: &mut String,
    data: &mut StringTable,
    locals: &mut Locals,
) -> Result<(), AsmError> {
    let find = |fname: &str| lit_fields.iter().find(|(n, _)| n == fname).map(|(_, e)| e);
    // field0 -> rax (saved on the stack while field1 is computed)
    let f0 = &sd.fields[0];
    let e0 = find(&f0.name).ok_or(AsmError::UnsupportedExpr(
        "struct-return literal missing a declared field"))?;
    let k0 = emit_expr_value(e0, out, data, locals)?;
    if !matches!(k0, TyKind::Int) {
        return Err(AsmError::UnsupportedExpr("struct-return field must be i64-shaped"));
    }
    out.push_str("    pushq %rax\n");
    if sd.fields.len() >= 2 {
        let f1 = &sd.fields[1];
        let e1 = find(&f1.name).ok_or(AsmError::UnsupportedExpr(
            "struct-return literal missing a declared field"))?;
        let k1 = emit_expr_value(e1, out, data, locals)?;
        if !matches!(k1, TyKind::Int) {
            return Err(AsmError::UnsupportedExpr("struct-return field must be i64-shaped"));
        }
        out.push_str("    movq %rax, %rdx\n");
    } else {
        out.push_str("    xorl %edx, %edx\n");
    }
    out.push_str("    popq %rax\n");
    Ok(())
}

/// Emit code that produces a payload-enum value in (%rax = tag, %rdx = val).
/// Used by fn-tail and `Stmt::Return` paths when the enclosing fn returns a
/// payload-enum, and by `Expr::Try` to propagate the Err variant unchanged.
///
/// Three accepted shapes for `e`:
///   * `EnumName::Variant[(payload)]` literal  — tag from variant index, val
///     from the payload expr (or 0 if no payload).
///   * Bare ident referring to an existing payload-enum local — load both
///     `.tag` and `.val` slots.
///   * `Call(...)` to a fn that itself returns a payload-enum — the call
///     already leaves (rax, rdx) populated; nothing more to do.
fn emit_enum_return_value(
    e: &Expr,
    out: &mut String,
    data: &mut StringTable,
    locals: &mut Locals,
) -> Result<(), AsmError> {
    // Case 1: enum constructor literal.
    if let Some((_enum_name, variant_idx, payload_expr)) =
        resolve_enum_ctor(e, &locals.enum_decls)
    {
        // Evaluate the payload first (clobbers rax) then set both regs.
        // Keep the tag immediate small and dependency-free so it doesn't
        // get reordered with the payload eval.
        if let Some(pe) = payload_expr {
            let pty = emit_expr_value(&pe, out, data, locals)?;
            if !matches!(pty, TyKind::Int) {
                return Err(AsmError::UnsupportedExpr(
                    "enum payload return currently restricted to i64-shaped types"));
            }
            out.push_str("    movq %rax, %rdx\n");
        } else {
            out.push_str("    xorl %edx, %edx\n");
        }
        out.push_str(&format!("    movq ${}, %rax\n", variant_idx as i64));
        return Ok(());
    }
    // Case 2: bare ident → existing payload-enum local.
    if let Expr::Ident(n) = e {
        if locals.enum_locals.contains_key(n) {
            let tag_key = format!("{}.tag", n);
            let val_key = format!("{}.val", n);
            let tag_slot = locals.get(&tag_key).ok_or(AsmError::UnsupportedExpr(
                "enum return: ident missing .tag slot"))?;
            let val_slot = locals.get(&val_key).ok_or(AsmError::UnsupportedExpr(
                "enum return: ident missing .val slot"))?;
            out.push_str(&format!("    movq -{}(%rbp), %rax\n", tag_slot * 8));
            out.push_str(&format!("    movq -{}(%rbp), %rdx\n", val_slot * 8));
            return Ok(());
        }
    }
    // Case 3: call returning a payload-enum — falls through (rax, rdx)
    // already set by the CALL.
    if call_returns_enum(e, &locals.fn_returns_enum).is_some() {
        let _ = emit_expr_value(e, out, data, locals)?;
        return Ok(());
    }
    // Case 4: `if cond { then_tail } else { else_tail }` where each branch
    // independently produces an enum value. Recurse into both arms.
    if let Expr::If { cond, then, else_ } = e {
        // Evaluate cond into rax.
        let cond_kind = emit_expr_value(cond, out, data, locals)?;
        if !matches!(cond_kind, TyKind::Int) {
            return Err(AsmError::UnsupportedExpr("enum-return if-cond must be int"));
        }
        let else_label = locals.fresh_label("enumret_else");
        let end_label = locals.fresh_label("enumret_end");
        out.push_str("    testq %rax, %rax\n");
        out.push_str(&format!("    je {}\n", else_label));
        // then-arm: emit stmts then dispatch tail through this same helper.
        for s in &then.stmts { emit_stmt(s, out, data, locals)?; }
        if let Some(t) = &then.tail {
            emit_enum_return_value(t, out, data, locals)?;
        } else {
            return Err(AsmError::UnsupportedExpr(
                "enum-return if-then arm must end in an enum value"));
        }
        out.push_str(&format!("    jmp {}\n", end_label));
        out.push_str(&format!("{}:\n", else_label));
        let else_block = else_.as_ref().ok_or(AsmError::UnsupportedExpr(
            "enum-return if must have an else"))?;
        for s in &else_block.stmts { emit_stmt(s, out, data, locals)?; }
        if let Some(t) = &else_block.tail {
            emit_enum_return_value(t, out, data, locals)?;
        } else {
            return Err(AsmError::UnsupportedExpr(
                "enum-return if-else arm must end in an enum value"));
        }
        out.push_str(&format!("{}:\n", end_label));
        return Ok(());
    }
    // Case 5: `Block { ... }` — recurse into tail.
    if let Expr::Block(b) = e {
        for s in &b.stmts { emit_stmt(s, out, data, locals)?; }
        if let Some(t) = &b.tail {
            return emit_enum_return_value(t, out, data, locals);
        }
        return Err(AsmError::UnsupportedExpr(
            "enum-return block must have a tail expression"));
    }
    Err(AsmError::UnsupportedExpr(
        "fn returning a payload-enum: tail/return must be an enum ctor, an enum local, a call to a payload-enum-returning fn, or an if/block whose arms produce enum values"))
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
    /// f64 constants per fn; labelled `.LD_<fnname>_<n>`.
    f64_consts: Vec<f64>,
    /// Default float width for bare `FloatLit` when no surrounding annotation
    /// disambiguates. F32 by default; `let x: f64 = ...` flips to F64 for the
    /// duration of the value expression.
    default_float: Option<TyKind>,
    /// Program-wide fn name → return TyKind. Lets call sites know whether the
    /// result lives in rax (Int) or xmm0 (F32/F64). Cloned in per-fn.
    sigs: HashMap<String, TyKind>,
    /// Set of locally-defined fn names (bodies present, not main). Call sites
    /// to these get the `aether_` prefix; everything else (extern fns, libc)
    /// is called by its bare name.
    local_fns: HashSet<String>,
    /// Max `args.len() - 4` seen across every call site in this fn. Drives
    /// extra outgoing-arg stack reservation in the prologue.
    max_call_extras: usize,
    /// Tensor locals to auto-free at fn natural end. `(slot, free_fn_name)`
    /// pairs in declaration order. Free order is reverse-declaration so
    /// resources are torn down in stack discipline.
    tensor_handles: Vec<(usize, &'static str)>,
    /// Per-Tensor-local shape, captured at the `let x: Tensor<…, [M, K]>;`
    /// site. Method-call dispatch (`x.matmul(&w, &mut y)`) reads dims back
    /// from here to synthesize the runtime call's M/K/N int args.
    tensor_shapes: HashMap<String, Vec<usize>>,
    /// File-level integer constants. `const BSZ: i32 = 8;` populates this
    /// at try_emit time; Tensor shape dims like `[BSZ, KK]` resolve
    /// symbolic names against it. Lets shape parameters live in one place.
    const_env: HashMap<String, i64>,
    /// Per-local element type for Tensor lets ("f32" or "i32"). Lets the
    /// method dispatcher pick `aether_op_*_f32` vs `…_i32` variants.
    tensor_elem: HashMap<String, &'static str>,
    /// Per-local struct type name. Set when `let x: Foo;` or `let x: Foo
    /// = Foo { ... };`. The MethodCall dispatcher looks here to pick a
    /// `Foo__bar`-style mangled callee.
    struct_locals: HashMap<String, String>,
    /// Slot reserved for spilling `%rax` across the auto-free callq sequence
    /// in the epilogue (frees clobber rax with their `0` return). Only
    /// allocated if the fn has at least one Tensor local.
    ret_save_slot: Option<usize>,
    /// Struct decls keyed by struct name. Drives struct-typed `let` layout —
    /// each field gets its own slot, accessed as a synthetic `name.field` key.
    struct_decls: HashMap<String, StructDecl>,
    /// Payload-carrying enum decls. Drives 2-slot (`name.tag` + `name.val`)
    /// layout for enum locals where any variant carries data.
    enum_decls: HashMap<String, EnumDecl>,
    /// Per-local payload-enum type name → enum-decl name. Lets `match local`
    /// detect that scrutinee is a 2-slot enum and dispatch on `.tag`.
    enum_locals: HashMap<String, String>,
    /// Caller-side: fn name → enum-decl name for fns whose return is a
    /// payload-enum. Drives the 2-register return ABI at call sites.
    fn_returns_enum: HashMap<String, String>,
    /// Callee-side: enum-decl name iff the *current* fn returns a payload-enum.
    /// `Stmt::Return` and the block tail consult this to emit the 2-register
    /// return (tag → %rax, val → %rdx) instead of single-rax. Drives
    /// `Expr::Try` early-return: the handler runs when the inner call's tag
    /// is non-zero (Err), copying both registers to the caller as-is.
    current_fn_returns_enum: Option<String>,
    /// Caller-side: fn name → struct-decl name for fns whose return is a
    /// small (≤2 i64-field) struct. Drives the same 2-register return ABI
    /// (field0 → %rax, field1 → %rdx) at call sites. P6.5 struct-return.
    fn_returns_struct: HashMap<String, String>,
    /// Callee-side: struct-decl name iff the *current* fn returns a small
    /// struct. The block tail (a struct literal) is marshalled into
    /// (%rax = field0, %rdx = field1) instead of single-rax.
    current_fn_returns_struct: Option<String>,
    /// Cached fn-frame size in bytes — used by `Expr::Try` and `Stmt::Return`
    /// to emit `addq $frame, %rsp` epilogues without re-running the
    /// frame-sizing pass.
    frame_bytes_cache: usize,
    /// Stack arrays: name → (base_slot, n, elem_kind). `base_slot` is the
    /// slot of element 0 (closest to rbp); element k is at addr
    /// `-8*base_slot(%rbp) - 8*k`. Per-element kind only int/handle for
    /// now (8-byte slots) — float arrays would need a 4-byte stride.
    arrays: HashMap<String, (usize, usize, TyKind)>,
    /// Native slices (P16.19): name → (elem_kind, elem_size_bytes). The
    /// (ptr, len) pair lives in the synthetic `<name>.ptr` / `<name>.len`
    /// slots (both i64); this sidecar just records what the elements are so
    /// `s[i]` can scale the index and pick the right load width. Only i64
    /// elements (8-byte) are exercised today; the size is carried explicitly
    /// so widening to f32/u8 slices is a one-line change.
    slices: HashMap<String, (TyKind, usize)>,
    /// Const-generic specialization state, shared across all fns in the program.
    /// At each call site we check `templates` for the callee; if hit, we infer
    /// concrete dim bindings from the caller's tensor_shapes, mangle, queue.
    /// `None` only in unit tests that build a Locals by hand.
    generics: Option<Rc<RefCell<GenericState>>>,
    /// P15.2 — per-local physical register assignment computed by
    /// `mir::regalloc_plan`. Empty at --O0; populated at --O1 for the
    /// callee-saved subset (r12..r15). When a local maps here, its value
    /// lives in BOTH the stack slot (write-through) AND the assigned reg.
    /// Reads prefer the reg; writes hit both. Float / Tensor / composite
    /// locals are excluded by the planner so this map is always Int-only.
    reg_map: HashMap<String, u8>,
    /// Callee-saved regs actually pushed by this fn's prologue, in push
    /// order. Epilogue (and any Stmt::Return early-return path) pops them
    /// in reverse. Always a subset of `reg_map`'s values (deduplicated).
    saved_regs: Vec<u8>,
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
        // 8 bytes per slot + 32 bytes shadow space + 8 bytes per outgoing arg
        // beyond the first 4 (caller-allocated stack args), rounded up to 16.
        // P15.2 — when the fn pushes an odd count of callee-saved regs in the
        // prologue, rsp arrives at the `subq` 8-byte-misaligned. Add 8 to the
        // frame to restore 16-byte alignment post-subq (still rounds up).
        let raw = self.next_slot * 8 + 32 + self.max_call_extras * 8;
        let base = (raw + 15) & !15;
        if self.saved_regs.len() % 2 == 1 { base + 8 } else { base }
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
    /// Intern an f64 constant; return its label. Per-fn unique via prefix.
    fn intern_f64(&mut self, v: f64) -> String {
        for (i, &existing) in self.f64_consts.iter().enumerate() {
            if existing.to_bits() == v.to_bits() {
                return format!(".LD_{}_{}", self.fn_label_prefix, i);
            }
        }
        let label = format!(".LD_{}_{}", self.fn_label_prefix, self.f64_consts.len());
        self.f64_consts.push(v);
        label
    }
}

/// Lower a builtin numeric cast. `inner` is the TyKind already in rax/xmm0.
/// `to` is the target name: "f32", "f64", or "i64". Returns the resulting
/// TyKind. Same-type casts are no-ops (still valid).
fn emit_cast(out: &mut String, inner: TyKind, to: &str) -> Result<TyKind, AsmError> {
    match (inner, to) {
        (TyKind::Int, "f32") => { out.push_str("    cvtsi2ssq %rax, %xmm0\n"); Ok(TyKind::F32) }
        (TyKind::Int, "f64") => { out.push_str("    cvtsi2sdq %rax, %xmm0\n"); Ok(TyKind::F64) }
        (TyKind::F32, "i64") => { out.push_str("    cvtss2siq %xmm0, %rax\n"); Ok(TyKind::Int) }
        (TyKind::F64, "i64") => { out.push_str("    cvtsd2siq %xmm0, %rax\n"); Ok(TyKind::Int) }
        // Identity casts (we only model one int width internally — i32/i64
        // are both `TyKind::Int` for now, so widening/narrowing is a no-op).
        (TyKind::F32, "f32") | (TyKind::F64, "f64") => Ok(inner),
        (TyKind::Int, "i64") | (TyKind::Int, "i32") => Ok(inner),
        (TyKind::Int, "u64") | (TyKind::Int, "usize") | (TyKind::Int, "isize") => Ok(inner),
        // Narrowing int casts: keep the low N bits, matching Rust `as`
        // truncation. `u*` zero-extend, `i8`/`i16` sign-extend. All encodings
        // now supported by aether-asm (movzbl/movzwl/movsbq/movswq/movl reg-form).
        (TyKind::Int, "u8")  => { out.push_str("    movzbl %al, %eax\n"); Ok(TyKind::Int) }
        (TyKind::Int, "u16") => { out.push_str("    movzwl %ax, %eax\n"); Ok(TyKind::Int) }
        (TyKind::Int, "u32") => { out.push_str("    movl %eax, %eax\n");  Ok(TyKind::Int) }
        (TyKind::Int, "i8")  => { out.push_str("    movsbq %al, %rax\n"); Ok(TyKind::Int) }
        (TyKind::Int, "i16") => { out.push_str("    movswq %ax, %rax\n"); Ok(TyKind::Int) }
        // f32→i32 via the same cvtss2siq we use for i64.
        (TyKind::F32, "i32") => { out.push_str("    cvtss2siq %xmm0, %rax\n"); Ok(TyKind::Int) }
        (TyKind::F64, "i32") => { out.push_str("    cvtsd2siq %xmm0, %rax\n"); Ok(TyKind::Int) }
        // Narrow/widen between f32 and f64 via SSE2 cvt instructions.
        (TyKind::F32, "f64") => { out.push_str("    cvtss2sd %xmm0, %xmm0\n"); Ok(TyKind::F64) }
        (TyKind::F64, "f32") => { out.push_str("    cvtsd2ss %xmm0, %xmm0\n"); Ok(TyKind::F32) }
        _ => Err(AsmError::UnsupportedExpr("unsupported cast combination")),
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

fn emit_fn(f: &FnDecl, out: &mut String, data: &mut StringTable,
           sigs: &HashMap<String, TyKind>,
           local_fns: &HashSet<String>,
           struct_decls: &HashMap<String, StructDecl>,
           enum_decls: &HashMap<String, EnumDecl>,
           fn_returns_enum: &HashMap<String, String>,
           fn_returns_struct: &HashMap<String, String>,
           const_env: &HashMap<String, i64>,
           generics: Option<Rc<RefCell<GenericState>>>,
           fn_plan: Option<&HashMap<String, u8>>)
    -> Result<(Vec<(String, f32)>, Vec<(String, f64)>), AsmError>
{
    let name = if f.name == "main" { "main".to_string() } else { format!("aether_{}", f.name) };

    // Pre-pass: count locals so the prologue reserves the right amount.
    let mut locals = Locals::default();
    locals.fn_label_prefix = f.name.clone();
    locals.sigs = sigs.clone();
    locals.local_fns = local_fns.clone();
    locals.struct_decls = struct_decls.clone();
    locals.enum_decls = enum_decls.clone();
    locals.fn_returns_enum = fn_returns_enum.clone();
    // Callee-side: is THIS fn's declared return type a payload-enum?
    // (Both the bare fn name and the linker-mangled name are registered in
    // the caller-side map; either lookup hits.)
    locals.current_fn_returns_enum = fn_returns_enum.get(&f.name).cloned();
    locals.fn_returns_struct = fn_returns_struct.clone();
    locals.current_fn_returns_struct = fn_returns_struct.get(&f.name).cloned();
    locals.const_env = const_env.clone();
    locals.generics = generics;
    // P15.2 — install per-fn reg plan (callee-saved r12..r15). Empty at --O0.
    if let Some(plan) = fn_plan {
        locals.reg_map = plan.clone();
        let mut seen: HashSet<u8> = HashSet::new();
        // Sort by reg id for stable push order across rebuilds; reverse pop
        // order in the epilogue happens by iterating saved_regs in reverse.
        let mut regs: Vec<u8> = plan.values().copied().collect();
        regs.sort();
        for r in regs {
            if seen.insert(r) { locals.saved_regs.push(r); }
        }
    }
    let body = f.body.as_ref().unwrap();
    // Reserve slots for incoming params so the frame includes them.
    for p in &f.params { locals.alloc(&p.name); }
    count_locals(body, &mut locals);
    // One extra slot to spill `%rax` across the auto-free callq sequence
    // in the epilogue. Wastes 8 bytes if the fn has no Tensor locals but
    // keeps the frame-sizing pass symmetric across the two count passes.
    locals.alloc("_ret_save_");
    let frame = locals.frame_bytes();
    locals.frame_bytes_cache = frame;
    locals.slots.clear();
    locals.next_slot = 0;
    locals.types.clear();

    let ret_kind = f.ret.as_ref().and_then(TyKind::from_ty);

    out.push_str(&format!("{name}:\n"));
    out.push_str("    pushq %rbp\n");
    out.push_str("    movq %rsp, %rbp\n");
    // P15.2 — save the callee-saved regs this fn promotes locals into. The
    // ABI requires r12..r15 be preserved across calls; pushing in the
    // prologue is the cheapest way to honour that. Frame-bytes calc below
    // already adds 8 when the push count is odd to keep rsp 16-aligned
    // after the subq.
    for &r in &locals.saved_regs {
        out.push_str(&format!("    pushq %r{}\n", r));
    }
    // P13.4: when the frame would skip past a stack guard page (>4 KiB on
    // Windows), probe each page as we decrement rsp. Inline equivalent of the
    // MSVC `__chkstk` helper — keeps us free of an external linkage.
    if frame > 4096 {
        out.push_str(&format!("    movq ${}, %r11\n", frame));   // remaining
        out.push_str("    movq $4096, %r10\n");                  // page size
        out.push_str(&format!(".Lchkstk_loop_{}:\n", name));
        out.push_str("    cmpq %r10, %r11\n");                   // r11 - 4096
        out.push_str(&format!("    jbe .Lchkstk_done_{}\n", name));
        out.push_str("    subq $4096, %rsp\n");
        out.push_str("    movq %r10, (%rsp)\n");                 // touch page
        out.push_str("    subq $4096, %r11\n");
        out.push_str(&format!("    jmp .Lchkstk_loop_{}\n", name));
        out.push_str(&format!(".Lchkstk_done_{}:\n", name));
        out.push_str("    subq %r11, %rsp\n");
    } else {
        out.push_str(&format!("    subq ${}, %rsp\n", frame));
    }

    // Spill incoming param regs into their stack slots and record type info.
    // MS x64: positional. Slot i picks {rcx,rdx,r8,r9} (int) or xmm{i} (float).
    let int_arg_regs = ["%rcx", "%rdx", "%r8", "%r9"];
    let mut arg_idx = 0usize; // ABI arg slot index (rcx/rdx/r8/r9 / xmm0..3)
    for p in f.params.iter() {
        // Resolve Tensor params through the const env so `fn forward(x:
        // &Tensor<f32, [B, K]>, ...)` works when B / K are file-level
        // consts. Refs of Tensor types collapse to the Tensor itself —
        // the value passed at runtime is the i64 handle either way.
        let p_ty = match &p.ty {
            Ty::Ref { inner, .. } => inner.as_ref(),
            other => other,
        };
        // Struct-by-value: the param's struct fields each occupy one ABI
        // arg slot. `let f: Foo = Foo { x: 1, y: 2.0 }; foo_method(f)`
        // passes (1, 2.0) in (rcx, xmm1). Inside `foo_method(self: Foo)`
        // we allocate `self.x` + `self.y` slots and spill from the
        // corresponding arg regs. `&self` / `&mut self` follow the same
        // path — Aether has no borrow semantics yet so a ref is just the
        // same multi-slot copy. Limited to <=4 fields total.
        // (`p_ty` already has any outer `Ref` stripped above.)
        let struct_name = struct_name_of(p_ty);
        if let Some(sname) = struct_name {
            if let Some(sd) = locals.struct_decls.get(&sname).cloned() {
                locals.struct_locals.insert(p.name.clone(), sname.clone());
                for field in &sd.fields {
                    if arg_idx >= 4 { return Err(AsmError::TooManyArgs); }
                    let field_kind = TyKind::from_ty(&field.ty).unwrap_or(TyKind::Int);
                    let key = format!("{}.{}", p.name, field.name);
                    let slot = locals.alloc(&key);
                    locals.types.insert(key, field_kind);
                    match field_kind {
                        TyKind::Int => out.push_str(&format!("    movq {}, -{}(%rbp)\n", int_arg_regs[arg_idx], slot * 8)),
                        TyKind::F32 => out.push_str(&format!("    movss %xmm{}, -{}(%rbp)\n", arg_idx, slot * 8)),
                        TyKind::F64 => out.push_str(&format!("    movsd %xmm{}, -{}(%rbp)\n", arg_idx, slot * 8)),
                        TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                            out.push_str(&format!("    movq {}, -{}(%rbp)\n", int_arg_regs[arg_idx], slot * 8)),
                    }
                    arg_idx += 1;
                }
                continue;
            }
        }

        let i = arg_idx;
        let kind = TyKind::from_ty_with_env(p_ty, &locals.const_env).unwrap_or(TyKind::Int);
        let slot = locals.alloc(&p.name);
        locals.types.insert(p.name.clone(), kind);
        if matches!(kind, TyKind::TensorDev(_) | TyKind::TensorDevI32(_)) {
            if let Some(shape) = tensor_type_shape(p_ty, Some(&locals.const_env)) {
                locals.tensor_shapes.insert(p.name.clone(), shape);
            }
            let elem = match kind { TyKind::TensorDevI32(_) => "i32", _ => "f32" };
            locals.tensor_elem.insert(p.name.clone(), elem);
        }
        if i < 4 {
            // Register-passed arg.
            match kind {
                TyKind::Int => out.push_str(&format!("    movq {}, -{}(%rbp)\n", int_arg_regs[i], slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss %xmm{}, -{}(%rbp)\n", i, slot * 8)),
                TyKind::F64 => out.push_str(&format!("    movsd %xmm{}, -{}(%rbp)\n", i, slot * 8)),
                TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                    out.push_str(&format!("    movq {}, -{}(%rbp)\n", int_arg_regs[i], slot * 8)),
            }
        } else {
            // Stack-passed arg (i >= 4): MS x64 ABI puts arg-i at
            // [rbp + 48 + (i-4)*8] — past the saved rbp (8), saved rip (8),
            // and the 32-byte shadow space the caller reserved. Float args
            // are also passed on the stack in this slot (caller stored
            // %xmm0/etc into that 8-byte slot).
            let stk_off = 48 + (i - 4) * 8;
            match kind {
                TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) => {
                    out.push_str(&format!("    movq {}(%rbp), %rax\n", stk_off));
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8));
                }
                TyKind::F32 => {
                    out.push_str(&format!("    movss {}(%rbp), %xmm0\n", stk_off));
                    out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8));
                }
                TyKind::F64 => {
                    out.push_str(&format!("    movsd {}(%rbp), %xmm0\n", stk_off));
                    out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", slot * 8));
                }
            }
        }
        arg_idx += 1;
    }
    // Mirror the count-pass `_ret_save_` alloc so its slot offset matches.
    let ret_save = locals.alloc("_ret_save_");
    locals.ret_save_slot = Some(ret_save);

    // For float-returning fns, set the default float width so a bare literal
    // tail (e.g. `0.0` in `fn f() -> f64 { ...; 0.0 }`) is interned at the
    // matching width. Restored after the block.
    let saved = locals.default_float;
    if matches!(ret_kind, Some(TyKind::F32) | Some(TyKind::F64)) {
        locals.default_float = ret_kind;
    }
    if locals.current_fn_returns_enum.is_some() {
        // For payload-enum-returning fns the tail expression must produce
        // a (tag, val) pair in (rax, rdx) — route it through the
        // enum-aware helper instead of the standard emit_expr_value.
        for s in &body.stmts {
            emit_stmt(s, out, data, &mut locals)?;
        }
        if let Some(tail) = &body.tail {
            emit_enum_return_value(tail, out, data, &mut locals)?;
        } else {
            // Fall-through with no tail expression isn't meaningful for an
            // enum-returning fn, but keep behaviour permissive: leave (0, 0)
            // i.e. the first variant with a zeroed payload.
            out.push_str("    xorl %eax, %eax\n");
            out.push_str("    xorl %edx, %edx\n");
        }
    } else if locals.current_fn_returns_struct.is_some() {
        // P6.5 — struct-returning fn: the tail must be a struct literal, which
        // we marshal into (field0 → %rax, field1 → %rdx).
        for s in &body.stmts {
            emit_stmt(s, out, data, &mut locals)?;
        }
        match body.tail.as_deref() {
            Some(Expr::StructLit { name: lit_name, fields }) => {
                let sd = locals.struct_decls.get(lit_name).cloned()
                    .ok_or(AsmError::UnsupportedExpr(
                        "struct-return: unknown struct in tail literal"))?;
                emit_struct_return_value(fields, &sd, out, data, &mut locals)?;
            }
            _ => return Err(AsmError::UnsupportedExpr(
                "struct-returning fn must end with a struct-literal tail \
                 (explicit `return T{..}` is a follow-up)")),
        }
    } else {
        emit_block(body, out, data, &mut locals)?;
    }
    locals.default_float = saved;

    // Default-zero %rax only if the fn returns an int (or has no declared ret)
    // *and* the body has no tail expression. For float returns, the tail value
    // is already in xmm0 and we leave it. For payload-enum returns rax+rdx
    // were set by `emit_enum_return_value` above; don't clobber rax.
    if body.tail.is_none()
        && !matches!(ret_kind, Some(TyKind::F32) | Some(TyKind::F64))
        && locals.current_fn_returns_enum.is_none()
        && locals.current_fn_returns_struct.is_none()
    {
        out.push_str("    xorl %eax, %eax\n");
    }
    // Auto-free Tensor locals in reverse declaration order. Each free clobbers
    // %rax with its int return; spill rax to `_ret_save_` first then restore.
    // For float-returning fns, %xmm0 already holds the tail value and our
    // free calls don't touch xmm regs.
    if !locals.tensor_handles.is_empty() {
        let save = locals.ret_save_slot.expect("ret_save_slot must be allocated");
        let returns_int_in_rax = !matches!(ret_kind, Some(TyKind::F32) | Some(TyKind::F64));
        if returns_int_in_rax {
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", save * 8));
        }
        // Take to drop the borrow on `locals` while we iterate.
        let handles = std::mem::take(&mut locals.tensor_handles);
        for (slot, free_name) in handles.iter().rev() {
            out.push_str(&format!("    movq -{}(%rbp), %rcx\n", slot * 8));
            out.push_str(&format!("    callq {}\n", free_name));
        }
        if returns_int_in_rax {
            out.push_str(&format!("    movq -{}(%rbp), %rax\n", save * 8));
        }
    }
    out.push_str(&format!("    addq ${}, %rsp\n", frame));
    // P15.2 — restore callee-saved regs in reverse push order. No-op when
    // the planner assigned none (--O0 stays byte-identical).
    for &r in locals.saved_regs.iter().rev() {
        out.push_str(&format!("    popq %r{}\n", r));
    }
    out.push_str("    popq %rbp\n");
    out.push_str("    ret\n\n");
    let mut floats = Vec::with_capacity(locals.float_consts.len());
    for (i, v) in locals.float_consts.iter().enumerate() {
        floats.push((format!(".LF_{}_{}", locals.fn_label_prefix, i), *v));
    }
    let mut f64s = Vec::with_capacity(locals.f64_consts.len());
    for (i, v) in locals.f64_consts.iter().enumerate() {
        f64s.push((format!(".LD_{}_{}", locals.fn_label_prefix, i), *v));
    }
    Ok((floats, f64s))
}

fn count_locals(b: &Block, locals: &mut Locals) {
    for s in &b.stmts {
        match s {
            Stmt::Let { name, value, ty, .. } => {
                if let Some(v) = value { count_locals_in_expr(v, locals); }
                // Payload-enum constructor rhs: reserve `<name>.tag` and
                // `<name>.val` slots (mirrors emit_stmt's dual-slot layout).
                if let Some(v) = value {
                    if resolve_enum_ctor(v, &locals.enum_decls).is_some() {
                        locals.alloc(&format!("{}.tag", name));
                        locals.alloc(&format!("{}.val", name));
                        continue;
                    }
                }
                // Call to a fn returning a payload-enum: same 2-slot layout,
                // populated from (rax, rdx) instead of from the literal.
                if let Some(v) = value {
                    if call_returns_enum(v, &locals.fn_returns_enum).is_some() {
                        locals.alloc(&format!("{}.tag", name));
                        locals.alloc(&format!("{}.val", name));
                        continue;
                    }
                }
                // Struct literal rhs: reserve slot per field, same as the
                // uninit-struct branch below. Skips the trailing single-slot
                // alloc since each field gets its own slot.
                // Tuple literal rhs: reserve N slots `<name>.0` .. `<name>.<N-1>`.
                if let Some(Expr::Tuple(elems)) = value {
                    for (i, _) in elems.iter().enumerate() {
                        locals.alloc(&format!("{}.{}", name, i));
                    }
                    continue;
                }
                if let Some(Expr::StructLit { name: lit_name, .. }) = value {
                    if let Some(sd) = locals.struct_decls.get(lit_name).cloned() {
                        for f in &sd.fields {
                            locals.alloc(&format!("{}.{}", name, f.name));
                        }
                        continue;
                    }
                }
                // Tensor-typed uninit lets get one slot for the i64 handle.
                // Resolve symbolic shape dims through the const env so
                // `Tensor<f32, [BSZ, KK]>` counts as a tensor at this stage.
                if let Some(annot) = ty.as_ref().and_then(|t| TyKind::from_ty_with_env(t, &locals.const_env)) {
                    if matches!(annot, TyKind::TensorDev(_) | TyKind::TensorDevI32(_)) {
                        locals.alloc(name);
                        continue;
                    }
                }
                // Struct-typed lets allocate one slot per declared field.
                if let Some(struct_name) = ty.as_ref().and_then(struct_name_of) {
                    if let Some(sd) = locals.struct_decls.get(&struct_name).cloned() {
                        for f in &sd.fields {
                            locals.alloc(&format!("{}.{}", name, f.name));
                        }
                        continue;
                    }
                }
                // Tuple-typed let — reserves N positional slots.
                if let Some(Ty::Tuple(elem_tys)) = ty {
                    for i in 0..elem_tys.len() {
                        locals.alloc(&format!("{}.{}", name, i));
                    }
                    continue;
                }
                // Stack array `let buf: [T; N];` — reserves N consecutive slots.
                // Element 0 lives in the slot allocated FIRST (closest to rbp);
                // element k is at addr `&buf[0] - 8*k` so the index codegen can
                // do `negq %rax; movq (base, %rax, 8), …`.
                if let Some(Ty::Array { n, .. }) = ty {
                    for k in 0..*n {
                        locals.alloc(&format!("{}.{}", name, k));
                    }
                    continue;
                }
                // Native slice `let s: &[T] = …;` — a (ptr, len) fat pointer.
                // Two slots, `<name>.ptr` and `<name>.len`, both i64. P16.19.
                if let Some(Ty::Slice { .. }) = ty {
                    locals.alloc(&format!("{}.ptr", name));
                    locals.alloc(&format!("{}.len", name));
                    continue;
                }
                locals.alloc(name);
            }
            Stmt::LetTuple { names, value } => {
                count_locals_in_expr(value, locals);
                for n in names { locals.alloc(n); }
            }
            Stmt::Expr(e) => count_locals_in_expr(e, locals),
            Stmt::Return(Some(e)) => count_locals_in_expr(e, locals),
            Stmt::Return(None) => {}
        }
    }
    if let Some(t) = &b.tail { count_locals_in_expr(t, locals); }
}

/// If `t` is a Named type pointing to a registered struct, return its name.
fn struct_name_of(t: &Ty) -> Option<String> {
    if let Ty::Named(n) = t { Some(n.clone()) } else { None }
}

/// For a `Ty::Slice`, return `(loaded-value TyKind, element size in bytes)`.
/// P16.19 — the element size drives the index-load width and the sub-slice
/// stride; the kind is what `s[i]` evaluates to (Int for all int widths,
/// F32/F64 for float slices). Mapping is by the element type *name* so a
/// `&[u8]`/`&str` reads exactly one byte, `&[f32]` reads 4 via `movss`, etc.
///   * u8 / i8           → (Int, 1)
///   * u16 / i16         → (Int, 2)
///   * u32 / i32         → (Int, 4)
///   * f32               → (F32, 4)
///   * i64 / u64 / isize → (Int, 8)   (handles too — Tensor<…> handle slices)
///   * f64               → (F64, 8)
/// Returns None only for genuinely unsupported element types.
fn slice_elem_info(t: &Ty) -> Option<(TyKind, usize)> {
    let elem = if let Ty::Slice { elem, .. } = t { elem.as_ref() } else { return None; };
    // A `&str` parses to a slice whose element is the synthetic `u8` name.
    if let Ty::Named(name) = elem {
        return match name.as_str() {
            "u8" | "i8"            => Some((TyKind::Int, 1)),
            "u16" | "i16"          => Some((TyKind::Int, 2)),
            "u32" | "i32"          => Some((TyKind::Int, 4)),
            "f32"                  => Some((TyKind::F32, 4)),
            "i64" | "u64" | "isize" | "usize" => Some((TyKind::Int, 8)),
            "f64"                  => Some((TyKind::F64, 8)),
            _ => None,
        };
    }
    // Tensor<…> handle element (8-byte i64 handle).
    match TyKind::from_ty(elem) {
        Some(TyKind::TensorDev(_)) | Some(TyKind::TensorDevI32(_)) => Some((TyKind::Int, 8)),
        _ => None,
    }
}

/// Scale `%rax` by a power-of-two `factor` using repeated self-adds. Used
/// for slice/array element-offset math where the only available multiply in
/// the asm assembler is the 2-operand register `imulq`. `factor` MUST be a
/// power of two (asserted at the slice-construct site via elem_size).
fn emit_scale_pow2(out: &mut String, factor: usize) {
    let mut f = factor;
    while f > 1 {
        out.push_str("    addq %rax, %rax\n");
        f >>= 1;
    }
}

/// Emit code that writes a slice's (ptr, len) into the given stack slots.
/// `value` is the construction expression on the rhs of `let s: &[T] = …`.
/// Recognised forms (P16.19):
///   * `slice_from_raw(ptr_expr, len_expr)` — evaluate both args; ptr→.ptr,
///     len→.len. The general "make a fat pointer from a raw base + count".
///   * `&s[lo..hi]` — sub-slice of an existing slice local `s`. New ptr is
///     `s.ptr + lo*elem_size`; new len is `hi - lo`.
fn emit_slice_construct(
    value: &Expr,
    ptr_slot: usize,
    len_slot: usize,
    elem_size: usize,
    out: &mut String,
    data: &mut StringTable,
    locals: &mut Locals,
) -> Result<(), AsmError> {
    // Form 1 — `slice_from_raw(ptr, len)`.
    if let Expr::Call { callee, args } = value {
        if let Expr::Ident(n) = callee.as_ref() {
            if n == "slice_from_raw" {
                if args.len() != 2 {
                    return Err(AsmError::UnsupportedExpr("slice_from_raw takes (ptr, len)"));
                }
                // ptr → .ptr
                let pk = emit_expr_value(&args[0], out, data, locals)?;
                if !matches!(pk, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("slice_from_raw ptr must be i64"));
                }
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", ptr_slot * 8));
                // len → .len
                let lk = emit_expr_value(&args[1], out, data, locals)?;
                if !matches!(lk, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("slice_from_raw len must be i64"));
                }
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", len_slot * 8));
                return Ok(());
            }
        }
    }
    // Form 3 — `&v[..]` full slice over a CONTAINER handle (Vec/String).
    // Disambiguator vs Form 2: the receiver is NOT a slice local — it's an i64
    // container handle. The accessor FFI is chosen by element size:
    //   8 → Vec<i64>, 4 → Vec<f32>, 1 → String/&str. This is exactly what
    //   `slice_from_raw(<container>_as_ptr(v), <container>_len(v))` lowers to;
    //   the calls are synthesized so the call ABI is handled by the Call path.
    if let Expr::Ref { expr, .. } = value {
        if let Expr::Index { recv, idx } = expr.as_ref() {
            if let (Expr::Ident(src), Expr::Range { .. }) = (recv.as_ref(), idx.as_ref()) {
                if !locals.slices.contains_key(src) {
                    let (as_ptr_fn, len_fn) = match elem_size {
                        8 => ("aether_vec_i64_as_ptr", "aether_vec_i64_len"),
                        4 => ("aether_vec_f32_as_ptr", "aether_vec_f32_len"),
                        1 => ("aether_string_as_ptr", "aether_string_len"),
                        _ => return Err(AsmError::UnsupportedExpr(
                            "&container[..] supports &[i64] / &[f32] / &str only")),
                    };
                    let ptr_call = Expr::Call {
                        callee: Box::new(Expr::Ident(as_ptr_fn.to_string())),
                        args: vec![Expr::Ident(src.clone())],
                    };
                    emit_expr_value(&ptr_call, out, data, locals)?;
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", ptr_slot * 8));
                    let len_call = Expr::Call {
                        callee: Box::new(Expr::Ident(len_fn.to_string())),
                        args: vec![Expr::Ident(src.clone())],
                    };
                    emit_expr_value(&len_call, out, data, locals)?;
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", len_slot * 8));
                    return Ok(());
                }
            }
        }
    }
    // Form 2 — `&s[lo..hi]` sub-slice.
    if let Expr::Ref { expr, .. } = value {
        if let Expr::Index { recv, idx } = expr.as_ref() {
            if let (Expr::Ident(src), Expr::Range { lo, hi, .. }) = (recv.as_ref(), idx.as_ref()) {
                if !locals.slices.contains_key(src) {
                    return Err(AsmError::UnsupportedExpr(
                        "sub-slice receiver must be a slice local"));
                }
                let src_ptr = locals.get(&format!("{}.ptr", src))
                    .ok_or_else(|| AsmError::UnknownIdent(format!("{}.ptr", src)))?;
                // new ptr = src.ptr + lo*elem_size
                let lo_kind = emit_expr_value(lo, out, data, locals)?;
                if !matches!(lo_kind, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("sub-slice lo bound must be int"));
                }
                // rax *= elem_size. elem_size is a power of two (8 today) so
                // scale via repeated `addq %rax,%rax` doublings — the asm
                // assembler only encodes the 2-operand register `imulq`, so
                // we avoid the immediate form entirely.
                emit_scale_pow2(out, elem_size);
                out.push_str(&format!("    movq -{}(%rbp), %r10\n", src_ptr * 8));
                out.push_str("    addq %r10, %rax\n");
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", ptr_slot * 8));
                // new len = hi - lo. Evaluate hi → rax, push; eval lo → r10; sub.
                let hi_kind = emit_expr_value(hi, out, data, locals)?;
                if !matches!(hi_kind, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("sub-slice hi bound must be int"));
                }
                out.push_str("    pushq %rax\n");
                let lo2 = emit_expr_value(lo, out, data, locals)?;
                if !matches!(lo2, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("sub-slice lo bound must be int"));
                }
                out.push_str("    movq %rax, %r10\n");
                out.push_str("    popq %rax\n");      // rax = hi
                out.push_str("    subq %r10, %rax\n"); // rax = hi - lo
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", len_slot * 8));
                return Ok(());
            }
        }
    }
    Err(AsmError::UnsupportedExpr(
        "slice construction must be `slice_from_raw(ptr, len)` or `&s[lo..hi]`"))
}

/// Extract the integer shape vector from a `Ty::Shape([Const(d0), Const(d1), …])`.
/// Symbolic dims (`ShapeDim::Sym(name)`) resolve via `const_env` when one is
/// supplied; otherwise an unresolved sym is `None` (caller decides whether
/// that's an error). Lets `Tensor<f32, [BSZ, KK]>` work as long as `BSZ` and
/// `KK` are file-level integer consts.
fn tensor_shape_const(t: &Ty, const_env: Option<&HashMap<String, i64>>) -> Option<Vec<usize>> {
    use crate::ast::ShapeDim;
    if let Ty::Shape(dims) = t {
        let mut out = Vec::with_capacity(dims.len());
        for d in dims {
            match d {
                ShapeDim::Const(n) if *n >= 0 => out.push(*n as usize),
                ShapeDim::Sym(name) => {
                    let env = const_env?;
                    let v = *env.get(name)?;
                    if v < 0 { return None; }
                    out.push(v as usize);
                }
                _ => return None,
            }
        }
        Some(out)
    } else {
        None
    }
}

/// Pull the shape out of a Tensor type annotation. `Tensor<f32, [M, K]>` →
/// `Some([M, K])`. Symbolic dims resolve through `const_env`.
fn tensor_type_shape(t: &Ty, const_env: Option<&HashMap<String, i64>>) -> Option<Vec<usize>> {
    if let Ty::Generic { name, args } = t {
        if name == "Tensor" && args.len() == 2 {
            return tensor_shape_const(&args[1], const_env);
        }
    }
    None
}

/// Like `tensor_type_shape` but returns the SYMBOLIC dim list (`Sym(M)` /
/// `Const(8)`) without resolving — used by the const-generic call-site
/// inference to pair template Sym names against caller-side concrete shapes.
fn tensor_type_dims(t: &Ty) -> Option<&[ShapeDim]> {
    if let Ty::Generic { name, args } = t {
        if name == "Tensor" && args.len() == 2 {
            if let Ty::Shape(dims) = &args[1] {
                return Some(dims.as_slice());
            }
        }
    }
    None
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
            // gets a slot so we don't re-evaluate it each loop.  Both the range
            // path (var + _for_end_) and the P16.19 slice-iter path (_for_sidx_
            // + var) allocate exactly two slots in emit, so two here keeps the
            // prologue frame and the return epilogues in lock-step.
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
        Expr::Call { args, .. } => {
            for a in args { count_locals_in_expr(a, locals); }
            let extras = args.len().saturating_sub(4);
            if extras > locals.max_call_extras {
                locals.max_call_extras = extras;
            }
        }
        // `recv.method(args...)` desugars (in `emit_expr_value`) to a Call
        // with `1 + args.len() + extra_int_args.len()` real arguments.
        // The exact extra-int count depends on `method_dispatch`'s shape
        // recipe, which we don't have visibility into during count_locals.
        // Use a conservative upper bound of 3 extra ints (matmul's M/K/N
        // is the worst seen so far). Over-reserving is harmless; under-
        // reserving corrupts the outgoing-args region during the call.
        Expr::MethodCall { recv, args, .. } => {
            count_locals_in_expr(recv, locals);
            for a in args { count_locals_in_expr(a, locals); }
            let desugared = 1 + args.len() + 3;
            let extras = desugared.saturating_sub(4);
            if extras > locals.max_call_extras {
                locals.max_call_extras = extras;
            }
        }
        Expr::Field { recv, .. } => count_locals_in_expr(recv, locals),
        Expr::Range { lo, hi, .. } => {
            count_locals_in_expr(lo, locals);
            count_locals_in_expr(hi, locals);
        }
        Expr::Match { scrutinee, arms } => {
            count_locals_in_expr(scrutinee, locals);
            // Reserve a slot for the scrutinee save (saves a clobbered
            // %rax across per-arm cmp+jmp). Allocated once per match;
            // subsequent matches reuse via name (alloc returns a fresh
            // slot but that's fine).
            locals.alloc("_match_scrut_");
            for (pat, body) in arms {
                if let MatchPat::EnumVariantBind(_, bind) = pat {
                    locals.alloc(bind);
                }
                count_locals_in_expr(body, locals);
            }
        }
        Expr::Cast { expr, .. } => count_locals_in_expr(expr, locals),
        Expr::Try(inner) => count_locals_in_expr(inner, locals),
        Expr::Index { recv, idx } => {
            count_locals_in_expr(recv, locals);
            count_locals_in_expr(idx, locals);
        }
        Expr::Tuple(elems) => {
            for e in elems { count_locals_in_expr(e, locals); }
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
        Stmt::LetTuple { names, value } => {
            // `let (a, b, ...) = (e1, e2, ...);` — must be a tuple literal of
            // matching arity. Each name binds to its own top-level slot.
            let elems = match value {
                Expr::Tuple(es) => es,
                _ => return Err(AsmError::UnsupportedExpr(
                    "let-tuple rhs must be a tuple literal (fn-tuple-returns require sret)")),
            };
            if elems.len() != names.len() {
                return Err(AsmError::UnsupportedExpr("let-tuple arity mismatch"));
            }
            for (n, e) in names.iter().zip(elems.iter()) {
                let val_ty = emit_expr_value(e, out, data, locals)?;
                let slot = locals.alloc(n);
                locals.types.insert(n.clone(), val_ty);
                match val_ty {
                    TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                    TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
                    TyKind::F64 => out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", slot * 8)),
                }
            }
            Ok(())
        }
        Stmt::Expr(e) => { emit_expr_value(e, out, data, locals)?; Ok(()) }
        Stmt::Return(Some(e)) => {
            if locals.current_fn_returns_enum.is_some() {
                emit_enum_return_value(e, out, data, locals)?;
            } else {
                emit_expr_value(e, out, data, locals)?;
            }
            // Use the cached FINAL frame size, not the live mid-emission
            // `frame_bytes()`. An early `return` inside an `if` is emitted
            // before later `let`s have bumped `next_slot`, so `frame_bytes()`
            // here under-counts slots and the `addq` would restore `%rsp`
            // short of the prologue's `subq` — corrupting the saved %rbp /
            // return address (manifested as a SIGSEGV in non-tail tree
            // recursion). `frame_bytes_cache` is computed by the count_locals
            // pre-pass and matches the prologue exactly.
            let frame = locals.frame_bytes_cache;
            out.push_str(&format!("    addq ${}, %rsp\n", frame));
            // P15.2 — Stmt::Return is an early-exit; it must run the same
            // pop sequence the natural epilogue does, or the caller's
            // callee-saved regs leak out clobbered.
            for &r in locals.saved_regs.clone().iter().rev() {
                out.push_str(&format!("    popq %r{}\n", r));
            }
            out.push_str("    popq %rbp\n");
            out.push_str("    ret\n");
            Ok(())
        }
        Stmt::Return(None) => {
            out.push_str("    xorl %eax, %eax\n");
            // Cached final frame size — see the Return(Some) note above.
            let frame = locals.frame_bytes_cache;
            out.push_str(&format!("    addq ${}, %rsp\n", frame));
            for &r in locals.saved_regs.clone().iter().rev() {
                out.push_str(&format!("    popq %r{}\n", r));
            }
            out.push_str("    popq %rbp\n");
            out.push_str("    ret\n");
            Ok(())
        }
        Stmt::Let { name, value: None, ty, .. } => {
            // Uninit declaration. Two forms supported:
            //   (a) struct types — slots reserved by `count_locals`, body
            //       initialises each field via `name.field = expr` before read.
            //   (b) `Tensor<T, [N]>` — auto-call `aether_dev_alloc_*` here,
            //       store the i64 handle in the local slot, and queue the
            //       matching free for the fn epilogue.
            if let Some(annot) = ty.as_ref().and_then(|t| TyKind::from_ty_with_env(t, &locals.const_env)) {
                match annot {
                    TyKind::TensorDev(count) => {
                        out.push_str(&format!("    movq ${}, %rax\n", count));
                        out.push_str("    movq %rax, %rcx\n");
                        out.push_str("    callq aether_dev_alloc_f32\n");
                        let slot = locals.alloc(name);
                        locals.types.insert(name.clone(), TyKind::Int);
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8));
                        locals.tensor_handles.push((slot, "aether_dev_free_f32"));
                        if let Some(shape) = ty.as_ref().and_then(|t| tensor_type_shape(t, Some(&locals.const_env))) {
                            locals.tensor_shapes.insert(name.clone(), shape);
                        }
                        locals.tensor_elem.insert(name.clone(), "f32");
                        return Ok(());
                    }
                    TyKind::TensorDevI32(count) => {
                        out.push_str(&format!("    movq ${}, %rax\n", count));
                        out.push_str("    movq %rax, %rcx\n");
                        out.push_str("    callq aether_dev_alloc_i32\n");
                        let slot = locals.alloc(name);
                        locals.types.insert(name.clone(), TyKind::Int);
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8));
                        locals.tensor_handles.push((slot, "aether_dev_free_i32"));
                        if let Some(shape) = ty.as_ref().and_then(|t| tensor_type_shape(t, Some(&locals.const_env))) {
                            locals.tensor_shapes.insert(name.clone(), shape);
                        }
                        locals.tensor_elem.insert(name.clone(), "i32");
                        return Ok(());
                    }
                    _ => {}
                }
            }
            // Tuple `let pair: (i32, f32);` — reserve N positional slots.
            if let Some(Ty::Tuple(elem_tys)) = ty {
                for (i, t) in elem_tys.iter().enumerate() {
                    let key = format!("{}.{}", name, i);
                    let slot = locals.alloc(&key);
                    let kind = TyKind::from_ty(t).unwrap_or(TyKind::Int);
                    locals.types.insert(key, kind);
                    let _ = slot;
                }
                return Ok(());
            }
            // Stack array `let buf: [T; N];` — N slots already reserved by
            // count_locals (named "<buf>.0" .. "<buf>.<N-1>"). Allocate them
            // here in the same order to fix the base_slot, and record the
            // metadata sidecar so Index codegen can compute the address.
            if let Some(Ty::Array { elem, n }) = ty {
                let elem_kind = TyKind::from_ty(elem).unwrap_or(TyKind::Int);
                if !matches!(elem_kind, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("stack arrays currently support int/handle elements only"));
                }
                let mut base_slot: Option<usize> = None;
                for k in 0..*n {
                    let s = locals.alloc(&format!("{}.{}", name, k));
                    if k == 0 { base_slot = Some(s); }
                    locals.types.insert(format!("{}.{}", name, k), elem_kind);
                }
                if let Some(bs) = base_slot {
                    locals.arrays.insert(name.clone(), (bs, *n, elem_kind));
                }
                return Ok(());
            }
            let struct_name = ty.as_ref().and_then(struct_name_of)
                .ok_or(AsmError::UnsupportedExpr("uninit `let` requires a struct, Tensor, or array type"))?;
            let sd = locals.struct_decls.get(&struct_name).cloned()
                .ok_or(AsmError::UnsupportedExpr("unknown struct in uninit `let`"))?;
            locals.struct_locals.insert(name.clone(), struct_name.clone());
            for field in &sd.fields {
                let key = format!("{}.{}", name, field.name);
                let slot = locals.alloc(&key);
                let kind = TyKind::from_ty(&field.ty).unwrap_or(TyKind::Int);
                locals.types.insert(key, kind);
                let _ = slot;
            }
            Ok(())
        }
        Stmt::Let { name, value: Some(value), ty, .. } => {
            // Payload-enum constructor: `let b = Box::Full(42);` →
            // allocate `name.tag` (i64) + `name.val` (i64), write tag from
            // the variant index, write the payload arg into `.val`.
            if let Some((enum_name, variant_idx, payload_expr)) =
                resolve_enum_ctor(value, &locals.enum_decls)
            {
                let tag_key = format!("{}.tag", name);
                let val_key = format!("{}.val", name);
                let tag_slot = locals.alloc(&tag_key);
                locals.types.insert(tag_key, TyKind::Int);
                let val_slot = locals.alloc(&val_key);
                locals.types.insert(val_key, TyKind::Int);
                locals.enum_locals.insert(name.clone(), enum_name);
                out.push_str(&format!("    movq ${}, %rax\n", variant_idx as i64));
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", tag_slot * 8));
                if let Some(pe) = payload_expr {
                    let pty = emit_expr_value(&pe, out, data, locals)?;
                    if !matches!(pty, TyKind::Int) {
                        return Err(AsmError::UnsupportedExpr(
                            "enum payload currently restricted to i64-shaped types"));
                    }
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", val_slot * 8));
                } else {
                    out.push_str("    xorl %eax, %eax\n");
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", val_slot * 8));
                }
                let _ = ty;
                return Ok(());
            }
            // Call to a fn returning a payload-enum: same 2-slot layout
            // populated from the 2-register return ABI (rax=tag, rdx=val).
            // `let r = parse_one(x);` materialises `r.tag` + `r.val` from
            // the call result so subsequent `match r { ... }` can decode it.
            if let Some(enum_name) = call_returns_enum(value, &locals.fn_returns_enum) {
                let tag_key = format!("{}.tag", name);
                let val_key = format!("{}.val", name);
                let tag_slot = locals.alloc(&tag_key);
                locals.types.insert(tag_key, TyKind::Int);
                let val_slot = locals.alloc(&val_key);
                locals.types.insert(val_key, TyKind::Int);
                locals.enum_locals.insert(name.clone(), enum_name);
                // Evaluate the call. The 2-register return convention leaves
                // tag in rax and val in rdx — emit_expr_value reports Int
                // (rax) and we capture rdx ourselves before any subsequent
                // instruction can clobber it.
                let _ = emit_expr_value(value, out, data, locals)?;
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", tag_slot * 8));
                out.push_str(&format!("    movq %rdx, -{}(%rbp)\n", val_slot * 8));
                let _ = ty;
                return Ok(());
            }
            // Native slice `let s: &[T] = <slice-expr>;` (P16.19). Builds a
            // (ptr, len) fat pointer into the synthetic `<name>.ptr` /
            // `<name>.len` slots (allocated by count_locals). Two construction
            // forms are recognised:
            //   * `slice_from_raw(ptr_i64, len_i64)` — a builtin that takes a
            //     raw backing pointer + element count (the witness feeds it
            //     `aether_vec_i64_as_ptr(v)` + `aether_vec_i64_len(v)`).
            //   * `&s[a..b]` — sub-slice of an existing slice local: the new
            //     ptr is `s.ptr + a*elem_size`, the new len is `b - a`.
            if let Some(slice_ty) = ty.as_ref().filter(|t| matches!(t, Ty::Slice { .. })) {
                let (elem_kind, elem_size) = slice_elem_info(slice_ty)
                    .ok_or(AsmError::UnsupportedExpr(
                        "slice element type unsupported (u8/i8/u16/i16/u32/i32/i64/u64/f32/f64 + Tensor handles)"))?;
                let ptr_key = format!("{}.ptr", name);
                let len_key = format!("{}.len", name);
                let ptr_slot = locals.alloc(&ptr_key);
                let len_slot = locals.alloc(&len_key);
                locals.types.insert(ptr_key, TyKind::Int);
                locals.types.insert(len_key, TyKind::Int);
                locals.slices.insert(name.clone(), (elem_kind, elem_size));
                emit_slice_construct(value, ptr_slot, len_slot, elem_size, out, data, locals)?;
                return Ok(());
            }
            // Struct literal as let rhs — desugars to "uninit struct let,
            // then per-field assignment." The struct decl's field list
            // gives types; lit's `(field_name, expr)` pairs give values.
            // Order doesn't matter — fields are matched by name.
            // Tuple literal `let pair = (1, 2.0);` — same machinery as a struct
            // lit but elements are positional `<name>.0`, `<name>.1`, etc. Type
            // is inferred elementwise from the rhs values; an explicit annotation
            // (`let pair: (i32, f32) = ...`) is allowed but informational.
            if let Expr::Tuple(elems) = value {
                for (i, elem) in elems.iter().enumerate() {
                    let key = format!("{}.{}", name, i);
                    let slot = locals.alloc(&key);
                    let saved = locals.default_float;
                    let annot_kind = if let Some(Ty::Tuple(tys)) = ty {
                        tys.get(i).and_then(TyKind::from_ty)
                    } else { None };
                    if matches!(annot_kind, Some(TyKind::F32) | Some(TyKind::F64)) {
                        locals.default_float = annot_kind;
                    }
                    let val_ty = emit_expr_value(elem, out, data, locals)?;
                    locals.default_float = saved;
                    let kind = annot_kind.unwrap_or(val_ty);
                    locals.types.insert(key, kind);
                    match kind {
                        TyKind::Int => out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                        TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
                        TyKind::F64 => out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", slot * 8)),
                        TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                    }
                }
                return Ok(());
            }
            if let Expr::StructLit { name: lit_name, fields } = value {
                let sd = locals.struct_decls.get(lit_name).cloned()
                    .ok_or(AsmError::UnsupportedExpr("struct literal: unknown type"))?;
                locals.struct_locals.insert(name.clone(), lit_name.clone());
                // Allocate one slot per declared field under `name.field` keys.
                for field in &sd.fields {
                    let key = format!("{}.{}", name, field.name);
                    let slot = locals.alloc(&key);
                    let kind = TyKind::from_ty(&field.ty).unwrap_or(TyKind::Int);
                    locals.types.insert(key, kind);
                    let _ = slot;
                }
                // Now emit each provided field's initialiser into the slot.
                for (fname, fvalue) in fields {
                    let key = format!("{}.{}", name, fname);
                    let slot = locals.get(&key)
                        .ok_or_else(|| AsmError::UnknownIdent(fname.clone()))?;
                    let kind = locals.types.get(&key).copied().unwrap_or(TyKind::Int);
                    let saved = locals.default_float;
                    if matches!(kind, TyKind::F32 | TyKind::F64) {
                        locals.default_float = Some(kind);
                    }
                    let val_ty = emit_expr_value(fvalue, out, data, locals)?;
                    locals.default_float = saved;
                    if val_ty != kind {
                        return Err(AsmError::UnsupportedExpr(
                            "struct literal field type mismatch"));
                    }
                    match kind {
                        TyKind::Int => out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                        TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
                        TyKind::F64 => out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", slot * 8)),
                        TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                    }
                }
                return Ok(());
            }
            // P6.5 — `let p: T = small_struct_call();`. count_locals already
            // reserved `p.<field>` slots (the struct-typed-let branch keys off
            // the `: T` annotation), so here we mirror that allocation, run the
            // call (which leaves field0 in %rax, field1 in %rdx via the
            // struct-return ABI), and store each register into its field slot.
            if call_returns_struct(value, &locals.fn_returns_struct).is_some() {
                if let Some(struct_name) = ty.as_ref().and_then(struct_name_of) {
                    if let Some(sd) = locals.struct_decls.get(&struct_name).cloned() {
                        locals.struct_locals.insert(name.clone(), struct_name.clone());
                        for field in &sd.fields {
                            let key = format!("{}.{}", name, field.name);
                            let slot = locals.alloc(&key);
                            let kind = TyKind::from_ty(&field.ty).unwrap_or(TyKind::Int);
                            locals.types.insert(key, kind);
                            let _ = slot;
                        }
                        // Run the call: field0 → rax, field1 → rdx.
                        let _ = emit_expr_value(value, out, data, locals)?;
                        let s0 = locals.get(&format!("{}.{}", name, sd.fields[0].name))
                            .ok_or(AsmError::UnsupportedExpr("struct-return: field0 slot"))?;
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", s0 * 8));
                        if sd.fields.len() >= 2 {
                            let s1 = locals.get(&format!("{}.{}", name, sd.fields[1].name))
                                .ok_or(AsmError::UnsupportedExpr("struct-return: field1 slot"))?;
                            out.push_str(&format!("    movq %rdx, -{}(%rbp)\n", s1 * 8));
                        }
                        return Ok(());
                    }
                }
            }
            // Decide the local's TyKind: explicit annotation wins, else infer
            // from the value's runtime type. If the annotation is a float type,
            // bias bare FloatLits in the rhs to that width.
            let annot = ty.as_ref().and_then(TyKind::from_ty);
            let saved = locals.default_float;
            if matches!(annot, Some(TyKind::F32) | Some(TyKind::F64)) {
                locals.default_float = annot;
            }
            let val_ty = emit_expr_value(value, out, data, locals)?;
            locals.default_float = saved;
            let kind = annot.unwrap_or(val_ty);
            let slot = locals.alloc(name);
            locals.types.insert(name.clone(), kind);
            match kind {
                TyKind::Int => out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
                TyKind::F64 => out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", slot * 8)),
                TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
            }
            // P15.2 — write-through: hot Int local also lives in its
            // assigned callee-saved reg. We load from the just-written
            // stack slot rather than copying %rax → %rN, because the
            // peephole at lines 233-243 can collapse `movq $imm, %rax;
            // movq %rax, slot` into `movq $imm, slot`, leaving %rax with
            // a stale value. Loading from the slot is correct regardless
            // of whether the peephole fired, and is invisible to it
            // (peephole only matches %rax-destinated loads).
            if matches!(kind, TyKind::Int) {
                if let Some(&r) = locals.reg_map.get(name) {
                    out.push_str(&format!("    movq -{}(%rbp), %r{}\n", slot * 8, r));
                }
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
        Expr::BoolLit(b) => {
            // `true` / `false` lower to 1 / 0 in %rax (Int-class). Branch and
            // arithmetic codegen treat them as ordinary integers.
            out.push_str(&format!("    movq ${}, %rax\n", if *b { 1 } else { 0 }));
            Ok(TyKind::Int)
        }
        Expr::FloatLit(f) => {
            // Width selected by the surrounding annotation (set by `Stmt::Let`,
            // assignment, or float-returning fn). Defaults to F32.
            match locals.default_float {
                Some(TyKind::F64) => {
                    let label = locals.intern_f64(*f);
                    out.push_str(&format!("    movsd {}(%rip), %xmm0\n", label));
                    Ok(TyKind::F64)
                }
                _ => {
                    let label = locals.intern_f32(*f as f32);
                    out.push_str(&format!("    movss {}(%rip), %xmm0\n", label));
                    Ok(TyKind::F32)
                }
            }
        }
        Expr::StrLit(s) => {
            let label = data.intern(s);
            out.push_str(&format!("    leaq {}(%rip), %rax\n", label));
            Ok(TyKind::Int)
        }
        Expr::Ident(name) => {
            // Ident name might be a local OR a registered fn — function pointer
            // case takes precedence ONLY when no local of that name exists,
            // so shadowing of a fn name by a local works as expected.
            if locals.get(name).is_none() && locals.local_fns.contains(name) {
                // Load the address of `aether_<name>` into rax — works for any
                // local fn since we ALWAYS prefix at codegen time. Acts as a
                // value of TyKind::Int (function pointer is just a 64-bit int).
                out.push_str(&format!("    leaq aether_{}(%rip), %rax\n", name));
                return Ok(TyKind::Int);
            }
            // File-level int const? Inline the literal value.
            if locals.get(name).is_none() {
                if let Some(v) = locals.const_env.get(name).copied() {
                    out.push_str(&format!("    movq ${}, %rax\n", v));
                    return Ok(TyKind::Int);
                }
            }
            let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            let kind = locals.types.get(name).copied().unwrap_or(TyKind::Int);
            // P15.2 — if this Int local was promoted to a callee-saved reg,
            // read from the reg instead of touching the stack. Float / Tensor
            // / composite locals are never in the plan (planner excludes them).
            if matches!(kind, TyKind::Int) {
                if let Some(&r) = locals.reg_map.get(name) {
                    out.push_str(&format!("    movq %r{}, %rax\n", r));
                    return Ok(kind);
                }
            }
            match kind {
                TyKind::Int => out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss -{}(%rbp), %xmm0\n", slot * 8)),
                TyKind::F64 => out.push_str(&format!("    movsd -{}(%rbp), %xmm0\n", slot * 8)),
                TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                    out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8)),
            }
            Ok(kind)
        }
        Expr::Field { recv, name: field } => {
            // Only `ident.field` for a struct-typed local — nested paths await a
            // future bump.
            let base = match recv.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("nested field access not yet supported")),
            };
            let key = format!("{}.{}", base, field);
            let slot = locals.get(&key).ok_or_else(|| AsmError::UnknownIdent(key.clone()))?;
            let kind = locals.types.get(&key).copied().unwrap_or(TyKind::Int);
            match kind {
                TyKind::Int => out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss -{}(%rbp), %xmm0\n", slot * 8)),
                TyKind::F64 => out.push_str(&format!("    movsd -{}(%rbp), %xmm0\n", slot * 8)),
                TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                    out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8)),
            }
            Ok(kind)
        }
        Expr::Bin { op: BinOp::Assign, lhs, rhs } => {
            // `*ptr = rhs` — store-through-pointer. Used by the closure-with-
            // captures lowering: a `&mut`-captured local becomes a `*mut i64`
            // param the body dereferences on read AND write. Eval ptr → rax
            // (push), eval rhs → rax, pop ptr into rdi, `movq %rax, (%rdi)`.
            if let Expr::Deref(inner) = lhs.as_ref() {
                let _ = emit_expr_value(inner, out, data, locals)?;
                out.push_str("    pushq %rax\n");
                let val_ty = emit_expr_value(rhs, out, data, locals)?;
                if !matches!(val_ty, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("store-through-pointer: only int/handle supported"));
                }
                out.push_str("    popq %rdi\n");
                out.push_str("    movq %rax, 0(%rdi)\n");
                return Ok(TyKind::Int);
            }
            // Indexed assignment `buf[i] = expr`. The buf must be a known
            // stack-array local; idx is computed at runtime, then we store
            // into &buf[0] - 8*idx.
            if let Expr::Index { recv, idx } = lhs.as_ref() {
                let arr_name = match recv.as_ref() {
                    Expr::Ident(n) => n.clone(),
                    _ => return Err(AsmError::UnsupportedExpr("array index assign: receiver must be an ident")),
                };
                let (base_slot, _, _) = locals.arrays.get(&arr_name).copied()
                    .ok_or(AsmError::UnsupportedExpr("array index assign: receiver is not a stack array"))?;
                // Evaluate index → rax, spill, then evaluate rhs.
                let idx_kind = emit_expr_value(idx, out, data, locals)?;
                if !matches!(idx_kind, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("array index must be int"));
                }
                out.push_str("    pushq %rax\n");
                let val_ty = emit_expr_value(rhs, out, data, locals)?;
                if !matches!(val_ty, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("array element write: only int/handle supported"));
                }
                out.push_str("    popq %r10\n");          // r10 = idx
                out.push_str("    negq %r10\n");          // r10 = -idx
                // Multiply r10 by 8 via 3 adds (our asm has no SIB scale form
                // and no shlq encoding yet — kept minimal so the encoder stays
                // small. SIB-with-scale is on the asm extension list.)
                out.push_str("    addq %r10, %r10\n");
                out.push_str("    addq %r10, %r10\n");
                out.push_str("    addq %r10, %r10\n");
                out.push_str(&format!("    leaq -{}(%rbp), %rdi\n", base_slot * 8));
                out.push_str("    addq %r10, %rdi\n");
                out.push_str("    movq %rax, 0(%rdi)\n");
                return Ok(TyKind::Int);
            }
            // LHS may be a bare ident (`x = ...`) or a struct field path
            // (`x.field = ...`). Build a synthetic key in either case.
            let name = match lhs.as_ref() {
                Expr::Ident(n) => n.clone(),
                Expr::Field { recv, name: field } => match recv.as_ref() {
                    Expr::Ident(base) => format!("{}.{}", base, field),
                    _ => return Err(AsmError::UnsupportedExpr("LHS of assignment: nested field access not yet supported")),
                },
                _ => return Err(AsmError::UnsupportedExpr("LHS of assignment must be an ident, field, or array index")),
            };
            let slot = locals.get(&name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            let kind = locals.types.get(&name).copied().unwrap_or(TyKind::Int);
            let saved = locals.default_float;
            if matches!(kind, TyKind::F32 | TyKind::F64) {
                locals.default_float = Some(kind);
            }
            let val_ty = emit_expr_value(rhs, out, data, locals)?;
            locals.default_float = saved;
            if val_ty != kind {
                return Err(AsmError::UnsupportedExpr(
                    "assignment type mismatch (Int/F32/F64 must match the local's declared type)"));
            }
            match kind {
                TyKind::Int => out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
                TyKind::F32 => out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", slot * 8)),
                TyKind::F64 => out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", slot * 8)),
                TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", slot * 8)),
            }
            // P15.2 — assignment to a reg-promoted Int local also updates
            // its callee-saved reg. Same load-from-slot pattern as
            // Stmt::Let so the peephole's `imm→rax + rax→slot` collapse
            // stays correct.
            if matches!(kind, TyKind::Int) {
                if let Some(&r) = locals.reg_map.get(&name) {
                    out.push_str(&format!("    movq -{}(%rbp), %r{}\n", slot * 8, r));
                }
            }
            Ok(kind)
        }
        Expr::Bin { op, lhs, rhs } => {
            // Short-circuit `&&` / `||`: never evaluate rhs when lhs decides
            // the result. Both operands treated as int booleans (0 = false,
            // anything else = true). Output is 0 or 1 in %rax.
            if matches!(op, BinOp::And | BinOp::Or) {
                let _ = emit_expr_value(lhs, out, data, locals)?;
                let short_lab = locals.fresh_label("scshort");
                let end_lab = locals.fresh_label("scend");
                out.push_str("    testq %rax, %rax\n");
                match op {
                    BinOp::And => out.push_str(&format!("    je {}\n", short_lab)),
                    BinOp::Or  => out.push_str(&format!("    jne {}\n", short_lab)),
                    _ => unreachable!(),
                }
                let _ = emit_expr_value(rhs, out, data, locals)?;
                out.push_str("    testq %rax, %rax\n");
                out.push_str("    setne %al\n");
                out.push_str("    movzbl %al, %eax\n");
                out.push_str(&format!("    jmp {}\n", end_lab));
                out.push_str(&format!("{}:\n", short_lab));
                // Short-circuit value: 0 for &&, 1 for ||.
                let val = if matches!(op, BinOp::And) { 0 } else { 1 };
                out.push_str(&format!("    movq ${}, %rax\n", val));
                out.push_str(&format!("{}:\n", end_lab));
                return Ok(TyKind::Int);
            }
            // Eval lhs first; pick the integer or float pipeline based on its type.
            let lhs_ty = emit_expr_value(lhs, out, data, locals)?;
            // Spill lhs to free up the result register for the rhs.
            match lhs_ty {
                TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                    out.push_str("    pushq %rax\n"),
                TyKind::F32 => {
                    out.push_str("    subq $16, %rsp\n");
                    out.push_str("    movss %xmm0, (%rsp)\n");
                }
                TyKind::F64 => {
                    out.push_str("    subq $16, %rsp\n");
                    out.push_str("    movsd %xmm0, (%rsp)\n");
                }
            }
            // Bias the rhs's bare-FloatLit width to match the lhs type.
            let saved = locals.default_float;
            if matches!(lhs_ty, TyKind::F32 | TyKind::F64) {
                locals.default_float = Some(lhs_ty);
            }
            let rhs_ty = emit_expr_value(rhs, out, data, locals)?;
            locals.default_float = saved;
            if rhs_ty != lhs_ty {
                return Err(AsmError::UnsupportedExpr("Bin operands must be same type"));
            }
            match lhs_ty {
                TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) => {
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
                        BinOp::BitAnd => { out.push_str("    andq %r10, %rax\n"); Ok(TyKind::Int) }
                        BinOp::BitOr  => { out.push_str("    orq %r10, %rax\n");  Ok(TyKind::Int) }
                        BinOp::BitXor => { out.push_str("    xorq %r10, %rax\n"); Ok(TyKind::Int) }
                        // Shifts use the CL form so any RHS value works; we
                        // already have the count in r10, so move r10 into rcx
                        // (low 8 bits = cl). `sarq` for `>>` matches Rust
                        // signed-shift semantics.
                        BinOp::Shl => {
                            out.push_str("    movq %r10, %rcx\n");
                            out.push_str("    shlq %cl, %rax\n");
                            Ok(TyKind::Int)
                        }
                        BinOp::Shr => {
                            out.push_str("    movq %r10, %rcx\n");
                            out.push_str("    sarq %cl, %rax\n");
                            Ok(TyKind::Int)
                        }
                        other => Err(AsmError::UnsupportedBinOp(*other)),
                    }
                }
                TyKind::F32 | TyKind::F64 => {
                    // Mnemonic prefix: "ss" for f32, "sd" for f64. Same opcodes.
                    let (mov, add, sub, mul, div, ucomi) = if matches!(lhs_ty, TyKind::F32) {
                        ("movss", "addss", "subss", "mulss", "divss", "ucomiss")
                    } else {
                        ("movsd", "addsd", "subsd", "mulsd", "divsd", "ucomisd")
                    };
                    out.push_str(&format!("    {} %xmm0, %xmm1\n", mov)); // xmm1 = rhs
                    out.push_str(&format!("    {} (%rsp), %xmm0\n", mov)); // xmm0 = lhs
                    out.push_str("    addq $16, %rsp\n");
                    match op {
                        BinOp::Add => { out.push_str(&format!("    {} %xmm1, %xmm0\n", add)); Ok(lhs_ty) }
                        BinOp::Sub => { out.push_str(&format!("    {} %xmm1, %xmm0\n", sub)); Ok(lhs_ty) }
                        BinOp::Mul => { out.push_str(&format!("    {} %xmm1, %xmm0\n", mul)); Ok(lhs_ty) }
                        BinOp::Div => { out.push_str(&format!("    {} %xmm1, %xmm0\n", div)); Ok(lhs_ty) }
                        BinOp::Eq => { out.push_str(&format!("    {} %xmm1, %xmm0\n", ucomi)); emit_setcc_int(out, "sete"); Ok(TyKind::Int) }
                        BinOp::Ne => { out.push_str(&format!("    {} %xmm1, %xmm0\n", ucomi)); emit_setcc_int(out, "setne"); Ok(TyKind::Int) }
                        BinOp::Lt => { out.push_str(&format!("    {} %xmm1, %xmm0\n", ucomi)); emit_setcc_int(out, "setb");  Ok(TyKind::Int) }
                        BinOp::Gt => { out.push_str(&format!("    {} %xmm1, %xmm0\n", ucomi)); emit_setcc_int(out, "seta");  Ok(TyKind::Int) }
                        BinOp::Le => { out.push_str(&format!("    {} %xmm1, %xmm0\n", ucomi)); emit_setcc_int(out, "setbe"); Ok(TyKind::Int) }
                        BinOp::Ge => { out.push_str(&format!("    {} %xmm1, %xmm0\n", ucomi)); emit_setcc_int(out, "setae"); Ok(TyKind::Int) }
                        other => Err(AsmError::UnsupportedBinOp(*other)),
                    }
                }
            }
        }
        Expr::Unary { op, expr } => {
            let kind = emit_expr_value(expr, out, data, locals)?;
            match (kind, op) {
                (TyKind::Int, UnOp::Neg) => { out.push_str("    negq %rax\n"); Ok(TyKind::Int) }
                (TyKind::Int, UnOp::Not) => {
                    out.push_str("    testq %rax, %rax\n");
                    out.push_str("    sete %al\n");
                    out.push_str("    movzbl %al, %eax\n");
                    Ok(TyKind::Int)
                }
                // P13.3: f32/f64 unary negate via `0 - x`. Loads 0.0 into xmm1
                // (sub-from), subtracts xmm0, copies result back. Uses only
                // existing encoder ops (movss/subss/{movsd/subsd}).
                (TyKind::F32, UnOp::Neg) => {
                    let zero = locals.intern_f32(0.0);
                    out.push_str(&format!("    movss {}(%rip), %xmm1\n", zero));
                    out.push_str("    subss %xmm0, %xmm1\n");
                    out.push_str("    movss %xmm1, %xmm0\n");
                    Ok(TyKind::F32)
                }
                (TyKind::F64, UnOp::Neg) => {
                    let zero = locals.intern_f64(0.0);
                    out.push_str(&format!("    movsd {}(%rip), %xmm1\n", zero));
                    out.push_str("    subsd %xmm0, %xmm1\n");
                    out.push_str("    movsd %xmm1, %xmm0\n");
                    Ok(TyKind::F64)
                }
                _ => Err(AsmError::UnsupportedExpr("unary op on this type")),
            }
        }
        // P12.5: `*expr` — load through a reference. The inner expression
        // should evaluate to an address in rax; we follow it.
        Expr::Deref(inner) => {
            let _ = emit_expr_value(inner, out, data, locals)?;
            out.push_str("    movq (%rax), %rax\n");
            Ok(TyKind::Int)
        }
        Expr::Ref { expr, .. } => {
            match expr.as_ref() {
                Expr::Ident(name) => {
                    let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
                    // For Tensor locals the value IS the device pointer
                    // (i64 handle). `&x` and `x` mean the same thing in
                    // call-site terms — load the handle, not its slot
                    // address. Same for `&mut x`. Avoids the user having
                    // to know whether they're passing a "pointer to a
                    // pointer" or just "the pointer".
                    if locals.tensor_shapes.contains_key(name) {
                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8));
                    } else {
                        out.push_str(&format!("    leaq -{}(%rbp), %rax\n", slot * 8));
                    }
                    Ok(TyKind::Int)
                }
                // `&self.w` for a Tensor field: load the field's stored
                // i64 handle (same path the bare `Field` read takes).
                // `&self.scalar` for a non-Tensor field would want the
                // address; not currently supported.
                Expr::Field { recv, name: field } => {
                    let recv_name = match recv.as_ref() {
                        Expr::Ident(n) => n.clone(),
                        _ => return Err(AsmError::UnsupportedExpr("`&` of nested field not supported")),
                    };
                    let key = format!("{}.{}", recv_name, field);
                    let slot = locals.get(&key).ok_or_else(|| AsmError::UnknownIdent(key.clone()))?;
                    out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8));
                    Ok(TyKind::Int)
                }
                _ => Err(AsmError::UnsupportedExpr("`&` only supports a bare local or struct field")),
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
            // P16.19 — slice iteration `for x in s` / `for x in s.iter()`:
            // walk `0..s.len()` binding `x = s[i]` (width-correct load) each
            // step. Recognised when the iterable is a slice local, or `.iter()`
            // / `.iter_mut()` on one.
            let slice_src = match iter.as_ref() {
                Expr::Ident(n) if locals.slices.contains_key(n) => Some(n.clone()),
                Expr::MethodCall { recv, name, .. }
                    if name == "iter" || name == "iter_mut" => match recv.as_ref() {
                        Expr::Ident(n) if locals.slices.contains_key(n) => Some(n.clone()),
                        _ => None,
                    },
                _ => None,
            };
            if let Some(s) = slice_src {
                let (elem_kind, elem_size) = locals.slices.get(&s).copied().unwrap();
                let ptr_slot = locals.get(&format!("{}.ptr", s))
                    .ok_or_else(|| AsmError::UnknownIdent(format!("{}.ptr", s)))?;
                let len_slot = locals.get(&format!("{}.len", s))
                    .ok_or_else(|| AsmError::UnknownIdent(format!("{}.len", s)))?;
                // counter i = 0
                let idx_slot = locals.alloc("_for_sidx_");
                out.push_str(&format!("    movq $0, -{}(%rbp)\n", idx_slot * 8));
                // element binding `var` (its own slot + elem type)
                let x_slot = locals.alloc(var);
                locals.types.insert(var.clone(), elem_kind);

                let top = locals.fresh_label("foriter_top");
                let cont = locals.fresh_label("foriter_cont");
                let end = locals.fresh_label("foriter_end");
                out.push_str(&format!("{}:\n", top));
                // if i >= len goto end
                out.push_str(&format!("    movq -{}(%rbp), %rax\n", idx_slot * 8));
                out.push_str(&format!("    movq -{}(%rbp), %r10\n", len_slot * 8));
                out.push_str("    cmpq %r10, %rax\n");
                out.push_str(&format!("    jge {}\n", end));
                // addr = s.ptr + i*elem_size  →  %rdi
                out.push_str(&format!("    movq -{}(%rbp), %rax\n", idx_slot * 8));
                emit_scale_pow2(out, elem_size);
                out.push_str(&format!("    movq -{}(%rbp), %r10\n", ptr_slot * 8));
                out.push_str("    addq %r10, %rax\n");
                out.push_str("    movq %rax, %rdi\n");
                // x = *(addr) — width-correct load + store to x's slot
                match (elem_kind, elem_size) {
                    (TyKind::Int, 1) => {
                        out.push_str("    movzbl 0(%rdi), %eax\n");
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", x_slot * 8));
                    }
                    (TyKind::Int, 2) => {
                        out.push_str("    movzwl 0(%rdi), %eax\n");
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", x_slot * 8));
                    }
                    (TyKind::Int, 4) => {
                        out.push_str("    movl 0(%rdi), %eax\n");
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", x_slot * 8));
                    }
                    (TyKind::F32, _) => {
                        out.push_str("    movss 0(%rdi), %xmm0\n");
                        out.push_str(&format!("    movss %xmm0, -{}(%rbp)\n", x_slot * 8));
                    }
                    (TyKind::F64, _) => {
                        out.push_str("    movsd 0(%rdi), %xmm0\n");
                        out.push_str(&format!("    movsd %xmm0, -{}(%rbp)\n", x_slot * 8));
                    }
                    _ => {
                        out.push_str("    movq 0(%rdi), %rax\n");
                        out.push_str(&format!("    movq %rax, -{}(%rbp)\n", x_slot * 8));
                    }
                }
                locals.loop_labels.push((cont.clone(), end.clone()));
                emit_block(body, out, data, locals)?;
                locals.loop_labels.pop();
                out.push_str(&format!("{}:\n", cont));
                // i++
                out.push_str(&format!("    movq -{}(%rbp), %rax\n", idx_slot * 8));
                out.push_str("    addq $1, %rax\n");
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", idx_slot * 8));
                out.push_str(&format!("    jmp {}\n", top));
                out.push_str(&format!("{}:\n", end));
                out.push_str("    xorl %eax, %eax\n");
                return Ok(TyKind::Int);
            }
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
            let mut name = match callee.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("non-ident callee")),
            };
            // Const-generic monomorphization. If `name` is a template, infer
            // concrete dim bindings from the caller's tensor_shapes by aligning
            // each Tensor param's symbolic dims with the matching arg's recorded
            // shape. Mangle, queue (idempotent), and rewrite `name` to the
            // specialization. The rest of the Call branch then sees a plain fn.
            if let Some(g) = locals.generics.clone() {
                let tdecl_opt = g.borrow().templates.get(&name).cloned();
                if let Some(tdecl) = tdecl_opt {
                    let mut bindings: HashMap<String, i64> = HashMap::new();
                    let mut type_bindings: HashMap<String, String> = HashMap::new();
                    for (i, tp) in tdecl.params.iter().enumerate() {
                        let p_ty = match &tp.ty {
                            Ty::Ref { inner, .. } => inner.as_ref(),
                            other => other,
                        };
                        // TYPE-generic param: the type is a bare `Named(T)` whose
                        // name is a generic param → infer T from the arg's type.
                        if let Ty::Named(tn) = p_ty {
                            if tdecl.const_params.iter().any(|cp| cp == tn) {
                                if let Some(arg) = args.get(i) {
                                    if let Some(concrete) = arg_concrete_type_name(arg, locals) {
                                        type_bindings.insert(tn.clone(), concrete);
                                    }
                                }
                                continue;
                            }
                        }
                        let Some(sym_dims) = tensor_type_dims(p_ty) else { continue; };
                        // Get the caller arg's concrete shape. Args are positional
                        // 1:1 with template params (no struct expansion at template
                        // call sites yet — templates take Tensors, not user structs).
                        let arg = args.get(i).ok_or(AsmError::UnsupportedExpr(
                            "template call: arg count mismatch"))?;
                        let arg_shape: Option<Vec<usize>> = match arg {
                            Expr::Ident(n) => locals.tensor_shapes.get(n).cloned(),
                            Expr::Ref { expr, .. } => {
                                if let Expr::Ident(n) = expr.as_ref() {
                                    locals.tensor_shapes.get(n).cloned()
                                } else { None }
                            }
                            _ => None,
                        };
                        let Some(shape) = arg_shape else { continue; };
                        if shape.len() != sym_dims.len() { continue; }
                        for (sym, &concrete) in sym_dims.iter().zip(shape.iter()) {
                            if let ShapeDim::Sym(s) = sym {
                                if tdecl.const_params.iter().any(|cp| cp == s) {
                                    bindings.insert(s.clone(), concrete as i64);
                                }
                            }
                        }
                    }
                    // Every generic param must be bound — by a shape (i64) or a
                    // type binding.
                    for cp in &tdecl.const_params {
                        if !bindings.contains_key(cp) && !type_bindings.contains_key(cp) {
                            return Err(AsmError::UnsupportedExpr(string_to_static(
                                format!("template '{}': could not infer generic param '{}'", name, cp))));
                        }
                    }
                    // Stable mangling: const_params order from the template; each
                    // is either a shape (i64) or a type binding.
                    let mut sorted: Vec<(String, i64)> = Vec::new();
                    let mut sorted_types: Vec<(String, String)> = Vec::new();
                    let mut suffix = String::new();
                    for cp in &tdecl.const_params {
                        if let Some(v) = bindings.get(cp) {
                            sorted.push((cp.clone(), *v));
                            suffix.push_str(&format!("__{}{}", cp, v));
                        } else if let Some(t) = type_bindings.get(cp) {
                            sorted_types.push((cp.clone(), t.clone()));
                            suffix.push_str(&format!("__{}_{}", cp, t));
                        }
                    }
                    let mangled = format!("{}{}", name, suffix);
                    {
                        let mut gm = g.borrow_mut();
                        if gm.seen.insert(mangled.clone()) {
                            sorted.sort_by(|a, b| a.0.cmp(&b.0));
                            sorted_types.sort_by(|a, b| a.0.cmp(&b.0));
                            gm.pending.push((name.clone(), sorted, sorted_types, mangled.clone()));
                        }
                    }
                    // Make the call resolve through the spec name. Register the
                    // spec's return TyKind so the caller reads the right register:
                    // from the template's sig if concrete, OR — for a type-generic
                    // `-> T` return — from the bound concrete type (so an `f32`
                    // instantiation is read from xmm0, not rax).
                    locals.local_fns.insert(mangled.clone());
                    let spec_ret_kind = locals.sigs.get(&name).copied().or_else(|| {
                        if let Some(Ty::Named(tn)) = tdecl.ret.as_ref() {
                            type_bindings.get(tn).and_then(|c| TyKind::from_ty(&Ty::Named(c.clone())))
                        } else { None }
                    });
                    if let Some(rk) = spec_ret_kind {
                        locals.sigs.insert(mangled.clone(), rk);
                    }
                    name = mangled;
                }
            }
            // Special-case: println(STR) → puts(STR).
            if name == "println" && args.len() == 1 {
                if let Expr::StrLit(s) = &args[0] {
                    let label = data.intern(s);
                    out.push_str(&format!("    leaq {}(%rip), %rcx\n", label));
                    out.push_str("    callq puts\n");
                    return Ok(TyKind::Int);
                }
            }
            // FR-15.3 — recognized AVX2 builtin: `__aether_avx2_dot_f32(a_ptr,
            // b_ptr, n) -> f32`. Inlines a 256-bit AVX2 dot product loop using
            // aether_asm's new VEX-encoded ops (vxorps/vmovups/vmulps/vaddps/
            // vzeroupper). `n` MUST be > 0 and a multiple of 8 at runtime;
            // the loop assumes that. The args are evaluated into rcx/rdx/r8
            // by simple inline movq — this is safe when the args are simple
            // pointer-yielding exprs (Ref / Ident / Field) and IntLits, which
            // is the witness's case. Pure i64 throughout; returns f32.
            if name == "__aether_avx2_dot_f32" && args.len() == 3 {
                let ka = emit_expr_value(&args[0], out, data, locals)?;
                if !matches!(ka, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr(
                        "__aether_avx2_dot_f32 arg 0 must be i64 / pointer"));
                }
                out.push_str("    movq %rax, %rcx\n");
                let kb = emit_expr_value(&args[1], out, data, locals)?;
                if !matches!(kb, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr(
                        "__aether_avx2_dot_f32 arg 1 must be i64 / pointer"));
                }
                out.push_str("    movq %rax, %rdx\n");
                let kn = emit_expr_value(&args[2], out, data, locals)?;
                if !matches!(kn, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr(
                        "__aether_avx2_dot_f32 arg 2 (n) must be i64"));
                }
                out.push_str("    movq %rax, %r8\n");
                // Acc = ymm0; index = rax. Step: 8 f32 per iter (32 bytes).
                let loop_lbl = locals.fresh_label("avx2dot");
                out.push_str("    vxorps %ymm0, %ymm0, %ymm0\n");
                out.push_str("    xorq %rax, %rax\n");
                out.push_str(&format!("{}:\n", loop_lbl));
                out.push_str("    vmovups 0(%rcx), %ymm1\n");
                out.push_str("    vmovups 0(%rdx), %ymm2\n");
                out.push_str("    vmulps %ymm2, %ymm1, %ymm1\n");
                out.push_str("    vaddps %ymm1, %ymm0, %ymm0\n");
                out.push_str("    addq $32, %rcx\n");
                out.push_str("    addq $32, %rdx\n");
                out.push_str("    addq $8, %rax\n");
                out.push_str("    cmpq %rax, %r8\n");
                out.push_str(&format!("    jne {}\n", loop_lbl));
                // Horizontal sum: store ymm0 to a 32-byte stack scratch,
                // vzeroupper to release AVX state, sum 8 f32s scalar-style.
                out.push_str("    subq $32, %rsp\n");
                out.push_str("    vmovups %ymm0, (%rsp)\n");
                out.push_str("    vzeroupper\n");
                out.push_str("    movss (%rsp), %xmm0\n");
                for off in [4, 8, 12, 16, 20, 24, 28] {
                    out.push_str(&format!("    movss {}(%rsp), %xmm1\n", off));
                    out.push_str("    addss %xmm1, %xmm0\n");
                }
                out.push_str("    addq $32, %rsp\n");
                return Ok(TyKind::F32);
            }
            // Builtin numeric casts: f32(x) / f64(x) / i64(x).
            // Keep the surrounding default_float intact so that bare literal
            // arguments don't get widened/narrowed by accident — the cast
            // value is whatever the inner expression naturally produces.
            if args.len() == 1 && matches!(name.as_str(), "f32" | "f64" | "i64") {
                let saved = locals.default_float;
                // For `i64(x)` we don't bias; for `f32(x)` / `f64(x)` we tell
                // bare literals which width they should take.
                match name.as_str() {
                    "f32" => locals.default_float = Some(TyKind::F32),
                    "f64" => locals.default_float = Some(TyKind::F64),
                    _ => {}
                }
                let inner = emit_expr_value(&args[0], out, data, locals)?;
                locals.default_float = saved;
                return emit_cast(out, inner, &name);
            }
            // MS x64 arg slots are positional. Slot i (0-indexed) picks:
            //   i < 4  → int → {rcx, rdx, r8, r9}[i]; float → xmm{i}
            //   i ≥ 4  → 8-byte stack slot at [rsp + 32 + (i-4)*8]
            //                                  ^^ above the 32-byte shadow
            //
            // Two-phase strategy so nested calls work correctly:
            //
            //   PHASE 1 — evaluate every arg in source order, spill each
            //     result onto the stack (16 bytes per arg to keep rsp
            //     16-aligned across f32/f64 spills). After N pushes, rsp
            //     sits at `base - N*16`; arg i's value is at offset
            //     `(N-1-i)*16` from the new rsp. Nested calls in args run
            //     between phase-1 pushes; they unwind their own stack
            //     before returning, so the outer push/pop discipline holds.
            //
            //   PHASE 2 — load each arg from its known stack offset and
            //     route to the right register or outgoing-args slot. The
            //     outgoing-args region was reserved in the prologue at
            //     [base+32, base+32+max_call_extras*8); from the lowered
            //     rsp it lives at offset `N*16 + 32 + (i-4)*8`.
            //
            //   PHASE 3 — `addq $(N*16), %rsp` to drop the spill region.
            //     rsp is back to `base`, satisfying the 16-byte alignment
            //     invariant required at the CALL.
            let mut arg_kinds: Vec<TyKind> = Vec::with_capacity(args.len());
            for arg in args {
                // Struct-by-value: a struct ident expands to one push per
                // declared field (in declaration order), each treated as
                // an independent ABI arg slot. Mirrors the param-spill
                // side in `emit_fn`.
                if let Expr::Ident(arg_name) = arg {
                    if let Some(struct_ty) = locals.struct_locals.get(arg_name).cloned() {
                        if let Some(sd) = locals.struct_decls.get(&struct_ty).cloned() {
                            for field in &sd.fields {
                                let key = format!("{}.{}", arg_name, field.name);
                                let slot = locals.get(&key)
                                    .ok_or_else(|| AsmError::UnknownIdent(key.clone()))?;
                                let field_kind = locals.types.get(&key).copied().unwrap_or(TyKind::Int);
                                match field_kind {
                                    TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) => {
                                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", slot * 8));
                                        out.push_str("    subq $16, %rsp\n");
                                        out.push_str("    movq %rax, (%rsp)\n");
                                    }
                                    TyKind::F32 => {
                                        out.push_str(&format!("    movss -{}(%rbp), %xmm0\n", slot * 8));
                                        out.push_str("    subq $16, %rsp\n");
                                        out.push_str("    movss %xmm0, (%rsp)\n");
                                    }
                                    TyKind::F64 => {
                                        out.push_str(&format!("    movsd -{}(%rbp), %xmm0\n", slot * 8));
                                        out.push_str("    subq $16, %rsp\n");
                                        out.push_str("    movsd %xmm0, (%rsp)\n");
                                    }
                                }
                                arg_kinds.push(field_kind);
                            }
                            continue;
                        }
                    }
                }
                let kind = emit_expr_value(arg, out, data, locals)?;
                out.push_str("    subq $16, %rsp\n");
                match kind {
                    TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                        out.push_str("    movq %rax, (%rsp)\n"),
                    TyKind::F32 => out.push_str("    movss %xmm0, (%rsp)\n"),
                    TyKind::F64 => out.push_str("    movsd %xmm0, (%rsp)\n"),
                }
                arg_kinds.push(kind);
            }
            // `n` is the count of ABI arg slots after struct expansion,
            // not the source-level arg count — a struct literal arg may
            // expand to multiple slots.
            let n = arg_kinds.len();
            let int_regs = ["%rcx", "%rdx", "%r8", "%r9"];
            // Iterate reverse so that when an in-register arg (i<4) lands
            // in rax/xmm0 then moves to its target reg, we don't clobber
            // it with the next iteration's load (which always targets
            // rax/xmm0 first).
            for i in (0..n).rev() {
                let kind = arg_kinds[i];
                let arg_off = ((n - 1 - i) * 16) as i64; // bytes from current rsp
                let (load, src_reg) = match kind {
                    TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) => ("movq",  "%rax"),
                    TyKind::F32 => ("movss", "%xmm0"),
                    TyKind::F64 => ("movsd", "%xmm0"),
                };
                out.push_str(&format!("    {} {}(%rsp), {}\n", load, arg_off, src_reg));
                if i < 4 {
                    match kind {
                        TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) => {
                            if int_regs[i] != "%rax" {
                                out.push_str(&format!("    movq %rax, {}\n", int_regs[i]));
                            }
                        }
                        TyKind::F32 | TyKind::F64 => {
                            if i != 0 {
                                out.push_str(&format!("    {} %xmm0, %xmm{}\n", load, i));
                            }
                        }
                    }
                } else {
                    let disp = (n * 16 + 32 + (i - 4) * 8) as i64;
                    let store = match kind {
                        TyKind::Int | TyKind::TensorDev(_) | TyKind::TensorDevI32(_) => "movq %rax",
                        TyKind::F32 => "movss %xmm0",
                        TyKind::F64 => "movsd %xmm0",
                    };
                    out.push_str(&format!("    {}, {}(%rsp)\n", store, disp));
                }
            }
            out.push_str(&format!("    addq ${}, %rsp\n", n * 16));
            // Indirect call through a local fn-pointer: `let f = my_fn; f(x)`.
            // `name` resolves to a local slot whose value is a code address
            // (loaded from a fn-name Ident or any other 64-bit-int source).
            if locals.get(&name).is_some() && !locals.local_fns.contains(&name) {
                let slot = locals.get(&name).unwrap();
                out.push_str(&format!("    movq -{}(%rbp), %r10\n", slot * 8));
                out.push_str("    callq *%r10\n");
                return Ok(locals.sigs.get(&name).copied().unwrap_or(TyKind::Int));
            }
            let linker_name = if locals.local_fns.contains(&name) {
                format!("aether_{}", name)
            } else {
                name.clone()
            };
            out.push_str(&format!("    callq {}\n", linker_name));
            Ok(locals.sigs.get(&name).copied().unwrap_or(TyKind::Int))
        }
        Expr::Path(parts) => {
            // Two-part `EnumName::Variant` lookup. Resolved through the
            // const env (populated at try_emit time from `Item::Enum`).
            // Returns the variant's i32 tag as an Int.
            let key = parts.join("::");
            let v = locals.const_env.get(&key).copied()
                .ok_or_else(|| AsmError::UnsupportedExpr(string_to_static(format!("unknown path: {}", key))))?;
            out.push_str(&format!("    movq ${}, %rax\n", v));
            Ok(TyKind::Int)
        }
        Expr::Index { recv, idx } => {
            // Stack-array read: load `*(&buf[0] - 8*idx)` into rax.
            let arr_name = match recv.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("array index read: receiver must be an ident")),
            };
            // P16.19 — slice index `s[i]`: load `*(s.ptr + i*elem_size)`.
            // Slices store their base as a heap pointer in `<name>.ptr` and
            // grow upward (unlike stack arrays, which grow toward lower
            // addresses), so the addressing is `ptr + i*size`, no negation.
            if let Some((elem_kind, elem_size)) = locals.slices.get(&arr_name).copied() {
                let ptr_slot = locals.get(&format!("{}.ptr", arr_name))
                    .ok_or_else(|| AsmError::UnknownIdent(format!("{}.ptr", arr_name)))?;
                let idx_kind = emit_expr_value(idx, out, data, locals)?;
                if !matches!(idx_kind, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("slice index must be int"));
                }
                // rax = i; scale to byte offset, then add the base pointer so
                // %rdi holds the EXACT element address.
                emit_scale_pow2(out, elem_size);
                out.push_str(&format!("    movq -{}(%rbp), %r10\n", ptr_slot * 8));
                out.push_str("    addq %r10, %rax\n");
                out.push_str("    movq %rax, %rdi\n");
                // P16.19 — width-correct load. Each element is read at EXACTLY
                // its own size so a 1-byte `&[u8]` element never over-reads the
                // 8 bytes the old i64-only path assumed (which could fault at
                // the tail of a heap allocation).
                match (elem_kind, elem_size) {
                    (TyKind::Int, 1) => out.push_str("    movzbl 0(%rdi), %eax\n"),
                    (TyKind::Int, 2) => out.push_str("    movzwl 0(%rdi), %eax\n"),
                    (TyKind::Int, 4) => out.push_str("    movl 0(%rdi), %eax\n"),
                    (TyKind::Int, _) => out.push_str("    movq 0(%rdi), %rax\n"),
                    (TyKind::F32, _) => out.push_str("    movss 0(%rdi), %xmm0\n"),
                    (TyKind::F64, _) => out.push_str("    movsd 0(%rdi), %xmm0\n"),
                    // Handle-typed (Tensor) elements behave like 8-byte ints.
                    (TyKind::TensorDev(_), _) | (TyKind::TensorDevI32(_), _) =>
                        out.push_str("    movq 0(%rdi), %rax\n"),
                }
                return Ok(elem_kind);
            }
            let (base_slot, _, elem_kind) = locals.arrays.get(&arr_name).copied()
                .ok_or(AsmError::UnsupportedExpr("array index read: receiver is not a stack array"))?;
            let idx_kind = emit_expr_value(idx, out, data, locals)?;
            if !matches!(idx_kind, TyKind::Int) {
                return Err(AsmError::UnsupportedExpr("array index must be int"));
            }
            out.push_str("    negq %rax\n");
            // *8 via three adds; see write-side note above.
            out.push_str("    addq %rax, %rax\n");
            out.push_str("    addq %rax, %rax\n");
            out.push_str("    addq %rax, %rax\n");
            out.push_str(&format!("    leaq -{}(%rbp), %rdi\n", base_slot * 8));
            out.push_str("    addq %rax, %rdi\n");
            out.push_str("    movq 0(%rdi), %rax\n");
            Ok(elem_kind)
        }
        Expr::Cast { expr, ty } => {
            // `expr as Ty` numeric coercion (i32/i64/f32/f64). Reuses the
            // same emit_cast as the f32(x)/f64(x)/i64(x) builtin form.
            let saved = locals.default_float;
            match ty.as_str() {
                "f32" => locals.default_float = Some(TyKind::F32),
                "f64" => locals.default_float = Some(TyKind::F64),
                _ => {}
            }
            let inner = emit_expr_value(expr, out, data, locals)?;
            locals.default_float = saved;
            emit_cast(out, inner, ty.as_str())
        }
        Expr::Match { scrutinee, arms } => {
            // Linear cmp-and-branch dispatch. Evaluate the scrutinee to
            // %rax; for each arm, compare against the pattern's int value
            // (Wildcard always matches); on mismatch fall through to the
            // next test, on match jump to the arm's body. After the body
            // jump to a shared end label. The result type is taken from
            // the first arm (other arms must agree — we don't enforce
            // beyond what runtime tests catch).
            //
            // Special case: payload-enum scrutinees. If the scrutinee is
            // an Ident resolving to a known payload-enum local, the "value"
            // we compare against is `<name>.tag`, and binding patterns
            // copy `<name>.val` into a fresh local for the arm body.
            let payload_enum_scrut: Option<String> = match scrutinee.as_ref() {
                Expr::Ident(n) if locals.enum_locals.contains_key(n) => Some(n.clone()),
                _ => None,
            };
            let save_slot = locals.alloc("_match_scrut_");
            let payload_val_slot = if let Some(enum_local) = &payload_enum_scrut {
                let tag_key = format!("{}.tag", enum_local);
                let tag_slot = locals.get(&tag_key)
                    .ok_or(AsmError::UnsupportedExpr("payload-enum match: missing .tag slot"))?;
                out.push_str(&format!("    movq -{}(%rbp), %rax\n", tag_slot * 8));
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", save_slot * 8));
                let val_key = format!("{}.val", enum_local);
                let val_slot = locals.get(&val_key)
                    .ok_or(AsmError::UnsupportedExpr("payload-enum match: missing .val slot"))?;
                Some(val_slot)
            } else {
                let scrut_kind = emit_expr_value(scrutinee, out, data, locals)?;
                if !matches!(scrut_kind, TyKind::Int) {
                    return Err(AsmError::UnsupportedExpr("match scrutinee must be int (or enum variant)"));
                }
                out.push_str(&format!("    movq %rax, -{}(%rbp)\n", save_slot * 8));
                None
            };
            let end_label = locals.fresh_label("match_end");
            let mut arm_kind: Option<TyKind> = None;
            for (i, (pat, body)) in arms.iter().enumerate() {
                let next_label = if i + 1 < arms.len() {
                    Some(locals.fresh_label("match_next"))
                } else {
                    None
                };
                match pat {
                    MatchPat::Wildcard => { /* fall through to body */ }
                    MatchPat::Int(n) => {
                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", save_slot * 8));
                        out.push_str(&format!("    movq ${}, %r10\n", n));
                        out.push_str("    cmpq %r10, %rax\n");
                        if let Some(nl) = &next_label {
                            out.push_str(&format!("    jne {}\n", nl));
                        } else {
                            // Last arm without wildcard — fall through if no match
                            out.push_str(&format!("    jne {}\n", end_label));
                        }
                    }
                    MatchPat::EnumVariant(parts) => {
                        let key = parts.join("::");
                        let v = locals.const_env.get(&key).copied()
                            .ok_or_else(|| AsmError::UnsupportedExpr(string_to_static(format!("match: unknown enum variant {}", key))))?;
                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", save_slot * 8));
                        out.push_str(&format!("    movq ${}, %r10\n", v));
                        out.push_str("    cmpq %r10, %rax\n");
                        if let Some(nl) = &next_label {
                            out.push_str(&format!("    jne {}\n", nl));
                        } else {
                            out.push_str(&format!("    jne {}\n", end_label));
                        }
                    }
                    MatchPat::EnumVariantBind(parts, _bind) => {
                        let key = parts.join("::");
                        let v = locals.const_env.get(&key).copied()
                            .ok_or_else(|| AsmError::UnsupportedExpr(string_to_static(format!("match: unknown enum variant {}", key))))?;
                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", save_slot * 8));
                        out.push_str(&format!("    movq ${}, %r10\n", v));
                        out.push_str("    cmpq %r10, %rax\n");
                        if let Some(nl) = &next_label {
                            out.push_str(&format!("    jne {}\n", nl));
                        } else {
                            out.push_str(&format!("    jne {}\n", end_label));
                        }
                    }
                }
                // For binding patterns, copy the payload into the bound local
                // BEFORE emitting the arm body so the body sees `bind` as a local.
                if let MatchPat::EnumVariantBind(_, bind) = pat {
                    let val_slot = payload_val_slot
                        .ok_or(AsmError::UnsupportedExpr(
                            "binding pattern requires a payload-enum scrutinee"))?;
                    let bind_slot = locals.alloc(bind);
                    locals.types.insert(bind.clone(), TyKind::Int);
                    out.push_str(&format!("    movq -{}(%rbp), %rax\n", val_slot * 8));
                    out.push_str(&format!("    movq %rax, -{}(%rbp)\n", bind_slot * 8));
                }
                let body_kind = emit_expr_value(body, out, data, locals)?;
                arm_kind.get_or_insert(body_kind);
                out.push_str(&format!("    jmp {}\n", end_label));
                if let Some(nl) = next_label {
                    out.push_str(&format!("{}:\n", nl));
                }
            }
            out.push_str(&format!("{}:\n", end_label));
            Ok(arm_kind.unwrap_or(TyKind::Int))
        }
        Expr::StructLit { name: struct_name, fields } => {
            // `Foo { a: 1, b: 2.0 }` allocates one slot per declared field
            // under synthetic `_anon.<n>.<field>` keys, evaluates each
            // initialiser into the slot, and yields the base "anon" handle
            // (currently unused — struct lits live their whole life as a
            // build-up of named fields). The intended idiom is to use the
            // result of a struct literal directly via field access in the
            // same expression, which the asm backend doesn't support yet;
            // for now this serves as a sugar over the `let x: Foo;
            // x.a = …; x.b = …;` pattern when the lit is the rhs of a let.
            // The `Stmt::Let` arm below picks it up specially.
            let _ = struct_name;
            let _ = fields;
            Err(AsmError::UnsupportedExpr(
                "struct literal must appear directly as the rhs of `let x: T = T { … };`"))
        }
        Expr::MethodCall { recv, name: method, args } => {
            // `recv.method(args...)` desugars to a call into the runtime
            // C-ABI surface. The dispatch table below maps each known
            // method to a runtime symbol and a recipe for synthesising
            // shape-derived integer args from `recv`'s + the args' Tensor
            // shapes (recorded in `Locals.tensor_shapes` at let time).
            //
            // No method body is generated; this is purely sugar over the
            // existing `aether_op_*` symbols. When MIR-level kernel
            // fusion lands, this lowering is the place that picks up
            // fusion-aware dispatch.
            let recv_name = match recv.as_ref() {
                Expr::Ident(n) => n.clone(),
                _ => return Err(AsmError::UnsupportedExpr("method receiver must be a bare local")),
            };
            // P16.19 — slice methods. `s.len()` reads the `<name>.len` slot;
            // `s.is_empty()` is `len == 0`. Checked before struct/Tensor
            // dispatch since a slice local carries neither a struct_locals nor
            // a tensor_shapes entry.
            if locals.slices.contains_key(&recv_name) {
                let len_slot = locals.get(&format!("{}.len", recv_name))
                    .ok_or_else(|| AsmError::UnknownIdent(format!("{}.len", recv_name)))?;
                match method.as_str() {
                    "len" => {
                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", len_slot * 8));
                        return Ok(TyKind::Int);
                    }
                    "is_empty" => {
                        out.push_str(&format!("    movq -{}(%rbp), %rax\n", len_slot * 8));
                        out.push_str("    testq %rax, %rax\n");
                        emit_setcc_int(out, "sete");
                        return Ok(TyKind::Int);
                    }
                    other => return Err(AsmError::UnsupportedExpr(
                        string_to_static(format!("unsupported slice method `.{}()`", other)))),
                }
            }
            // Fast path: receiver is a struct local with a corresponding
            // `Foo__method` mangled fn. UFCS lowering — `obj.bar(x)` →
            // `Foo__bar(obj, x)`. Receiver passes by-value via existing
            // arg-spill machinery; for plain ints/floats this is the
            // value, for struct types it's currently the FIRST FIELD only
            // (proper struct-pass-by-value awaits Phase-2 deref work).
            // For user-defined methods on Tensors (`impl Tensor { ... }`),
            // the same path applies — the user's mangled fn wins over the
            // built-in dispatch table because the lookup happens first.
            if let Some(struct_ty) = locals.struct_locals.get(&recv_name).cloned() {
                let mangled = format!("{}__{}", struct_ty, method);
                if locals.local_fns.contains(&mangled) {
                    let mut desugared_args: Vec<Expr> = Vec::with_capacity(1 + args.len());
                    desugared_args.push(Expr::Ident(recv_name.clone()));
                    for a in args { desugared_args.push(a.clone()); }
                    let desugared = Expr::Call {
                        callee: Box::new(Expr::Ident(mangled)),
                        args: desugared_args,
                    };
                    return emit_expr_value(&desugared, out, data, locals);
                }
            }
            let recv_shape = locals.tensor_shapes.get(&recv_name).cloned()
                .ok_or(AsmError::UnsupportedExpr("method receiver must be a Tensor local or struct local with matching impl method"))?;
            // Map method → (runtime_fn, shape-recipe).
            // The recipe is a closure taking (recv_shape, arg_shapes) and
            // returning the int args to append after the i64 handle args.
            // Implemented inline per method; centralising into a table is
            // future work.
            let arg_shapes: Vec<Option<Vec<usize>>> = args.iter().map(|a| {
                // Drill through Ref wrappers — `&x` and `x` are interchangeable
                // for shape lookup since Tensor handles flow as i64 either way.
                let inner = match a {
                    Expr::Ref { expr, .. } => expr.as_ref(),
                    other => other,
                };
                match inner {
                    // Bare Tensor local: shape comes from the per-fn sidecar.
                    Expr::Ident(n) => locals.tensor_shapes.get(n).cloned(),
                    // `self.w` style: receiver must be a struct local, the
                    // field's declared Ty supplies the Tensor shape.
                    Expr::Field { recv, name } => {
                        if let Expr::Ident(recv_name) = recv.as_ref() {
                            let stype = locals.struct_locals.get(recv_name).cloned()?;
                            let sd = locals.struct_decls.get(&stype)?;
                            let f = sd.fields.iter().find(|f| f.name == *name)?;
                            tensor_type_shape(&f.ty, Some(&locals.const_env))
                        } else { None }
                    }
                    _ => None,
                }
            }).collect();

            // Synthesise the desugared Call expression. We re-use the
            // existing `Expr::Call` codegen path so push/pop arg discipline,
            // nested-call handling, and stack-arg spill all stay in one
            // place. The desugar materialises:
            //   recv, args[0], args[1], …, M, K, N, …
            let (runtime_fn, extra_int_args) = method_dispatch(method, &recv_shape, &arg_shapes)?;
            let mut desugared_args: Vec<Expr> = Vec::with_capacity(1 + args.len() + extra_int_args.len());
            desugared_args.push(Expr::Ident(recv_name));
            for a in args {
                // `&tensor_ident` collapses to the bare ident.
                // `&self.tensor_field` keeps the Field — the Call-arg path
                // reads it correctly via the existing Expr::Field handler.
                let collapsed = match a {
                    Expr::Ref { expr, .. } => match expr.as_ref() {
                        Expr::Ident(n) if locals.tensor_shapes.contains_key(n) =>
                            Expr::Ident(n.clone()),
                        Expr::Field { .. } => (**expr).clone(),
                        _ => a.clone(),
                    },
                    other => other.clone(),
                };
                desugared_args.push(collapsed);
            }
            for ix in &extra_int_args {
                desugared_args.push(Expr::IntLit(*ix as i64));
            }
            let desugared = Expr::Call {
                callee: Box::new(Expr::Ident(runtime_fn.to_string())),
                args: desugared_args,
            };
            emit_expr_value(&desugared, out, data, locals)
        }
        Expr::Try(inner) => {
            // `expr?` — early-return propagation for payload-enum returns.
            // Desugars to:
            //   match expr {
            //     Ok(v)  => v,
            //     Err(e) => return Err(e),
            //   }
            //
            // Implementation:
            //   * The enclosing fn MUST itself return a payload-enum (the same
            //     2-register ABI is reused for the propagation path).
            //   * `inner` MUST be a `Call` to a fn that returns a payload-enum.
            //     The call leaves (rax = tag, rdx = val) on return.
            //   * If tag != 0 (Err variant), run the fn epilogue with rax/rdx
            //     unchanged so the caller observes the same Err verbatim.
            //   * Else (Ok variant), the result of `expr?` is the payload —
            //     move rdx into rax and continue.
            if locals.current_fn_returns_enum.is_none() {
                return Err(AsmError::UnsupportedExpr(
                    "`?` operator is only valid inside a fn that returns a payload-enum"));
            }
            if call_returns_enum(inner.as_ref(), &locals.fn_returns_enum).is_none() {
                return Err(AsmError::UnsupportedExpr(
                    "`?` operand must be a call to a fn returning a payload-enum"));
            }
            // Evaluate the inner call. (rax, rdx) are now (tag, val).
            let _ = emit_expr_value(inner.as_ref(), out, data, locals)?;
            // If tag != 0, branch to the fn epilogue.
            let ok_label = locals.fresh_label("try_ok");
            out.push_str("    testq %rax, %rax\n");
            out.push_str(&format!("    je {}\n", ok_label));
            // Err path: rax/rdx already hold the propagated Err variant.
            // Run the same epilogue shape as `Stmt::Return`.
            let frame = locals.frame_bytes_cache;
            out.push_str(&format!("    addq ${}, %rsp\n", frame));
            for &r in locals.saved_regs.clone().iter().rev() {
                out.push_str(&format!("    popq %r{}\n", r));
            }
            out.push_str("    popq %rbp\n");
            out.push_str("    ret\n");
            // Ok path: result of `?` is the payload (rdx) — move to rax.
            out.push_str(&format!("{}:\n", ok_label));
            out.push_str("    movq %rdx, %rax\n");
            Ok(TyKind::Int)
        }
        _ => Err(AsmError::UnsupportedExpr("unhandled expr in asm backend")),
    }
}

/// Look up a Tensor method name and return `(runtime_symbol, extra_int_args)`.
/// `extra_int_args` are appended after the handle args in the desugared call.
/// Each method's shape recipe is hard-coded; this is the place that grows
/// when we add ops or want to swap GPU vs CPU dispatch.
fn method_dispatch(
    method: &str,
    recv_shape: &[usize],
    arg_shapes: &[Option<Vec<usize>>],
) -> Result<(&'static str, Vec<usize>), AsmError> {
    match method {
        // q.matmul_t(&k, &mut scores) → matmul_nt(q, k, scores, M, K, N)
        //   q: [M, K], k: [N, K]   (k is transposed on the fly), scores: [M, N]
        "matmul_t" => {
            let s = recv_shape;
            let kk = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul_t: k must be a Tensor with shape"))?;
            if s.len() != 2 || kk.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul_t: shapes must be 2-dim"));
            }
            // M=s[0], K=s[1], N=kk[0]. (k.cols must == K.)
            Ok(("aether_op_matmul_nt_f32_cuda", vec![s[0], s[1], kk[0]]))
        }
        // x.matmul(&w, &mut y) → matmul_f32_cuda(x, w, y, M, K, N)
        //   2D × 2D: x:[M, K], w:[K, N], y:[M, N]
        //   3D × 2D: x:[B, S, K], w:[K, N], y:[B, S, N]   (batch flattened)
        //
        // The 3D case lets transformer code write `x.matmul(&proj, &mut y)`
        // where x/y are [batch, seq, hidden] and the projection is
        // [hidden, hidden_out]. cuBLAS sees a single sgemm of (B*S) × K @
        // K × N → (B*S) × N — same byte layout, no reshape allocation needed.
        "matmul" => {
            let s = recv_shape;
            let w = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul: w must be a Tensor with shape"))?;
            if w.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul: w must be 2-dim"));
            }
            let (m, k) = match s.len() {
                2 => (s[0], s[1]),
                3 => (s[0] * s[1], s[2]),
                _ => return Err(AsmError::UnsupportedExpr("matmul: receiver must be 2- or 3-dim")),
            };
            Ok(("aether_op_matmul_f32_cuda", vec![m, k, w[1]]))
        }
        // x.matmul_gelu(&w, &mut y) → matmul + in-place gelu fused.
        // Same shape recipe as matmul; the only difference is the runtime fn.
        "matmul_gelu" => {
            let s = recv_shape;
            let w = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul_gelu: w must be a Tensor with shape"))?;
            if s.len() != 2 || w.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul_gelu: shapes must be 2-dim"));
            }
            Ok(("aether_op_matmul_gelu_f32_cuda", vec![s[0], s[1], w[1]]))
        }
        // x.matmul_backward_rhs(&dy, &mut dw) → mm_bwd_rhs(x, dy, dw, M, K, N)
        //   x: [M, K], dy: [M, N], dw: [K, N]
        "matmul_backward_rhs" => {
            let s = recv_shape;
            let dy = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul_backward_rhs: dy must be Tensor with shape"))?;
            if s.len() != 2 || dy.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul_backward_rhs: shapes must be 2-dim"));
            }
            Ok(("aether_op_matmul_backward_rhs_f32_cuda", vec![s[0], s[1], dy[1]]))
        }
        // y.cross_entropy(&labels, &mut probs) → ce_fwd(y, labels, probs, B, V)
        //   y: [B, V]
        "cross_entropy" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("cross_entropy: receiver must be 2-dim"));
            }
            Ok(("aether_op_cross_entropy_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // probs.cross_entropy_backward(&labels, &mut dy) → ce_bwd(probs, labels, dy, B, V)
        "cross_entropy_backward" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("cross_entropy_backward: receiver must be 2-dim"));
            }
            Ok(("aether_op_cross_entropy_backward_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // w.adamw_step(&dw, &mut m, &mut v, lr, beta1, beta2, eps, wd, step)
        //   → adamw(w, dw, m, v, lr, b1, b2, eps, wd, step, N)
        // The non-shape hyperparam args (lr, b1, b2, eps, wd, step) are
        // user-supplied as call args; only N (= flat element count of w)
        // gets synthesized.
        "adamw_step" => {
            let n = recv_shape.iter().product();
            Ok(("aether_op_adamw_step_f32_cuda", vec![n]))
        }
        // a.add(&b, &mut out) → add_f32(a, b, out, n=numel(a))
        "add" => {
            let n = recv_shape.iter().product();
            Ok(("aether_op_add_f32_cuda", vec![n]))
        }
        // x.gelu(&mut y) → gelu_fwd(x, y, n=numel(x))
        "gelu" => {
            let n = recv_shape.iter().product();
            Ok(("aether_op_gelu_f32_cuda", vec![n]))
        }
        // x.softmax(&mut y) → softmax(x, y, B, D); receiver must be 2-dim.
        "softmax" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("softmax: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_softmax_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // y.softmax_backward(&dy, &mut dx) → softmax_bwd(y, dy, dx, B, D)
        "softmax_backward" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("softmax_backward: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_softmax_backward_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // y.softmax_backward_scaled(&dy, &mut dx, s) — fused softmax_bwd + scale.
        // Emitted by the MIR fusion pass; user-callable too.
        "softmax_backward_scaled" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("softmax_backward_scaled: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_softmax_backward_scaled_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // a.matmul_tn(&b, &mut out): out[m,n] = a[k,m]^T @ b[k,n]
        "matmul_tn" => {
            let s = recv_shape;
            let bb = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul_tn: b must be a Tensor with shape"))?;
            if s.len() != 2 || bb.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul_tn: shapes must be 2-dim"));
            }
            // a is [K, M] (interpreted T → [M, K]); b is [K, N]; out is [M, N].
            // → M = s[1], K = s[0], N = bb[1].
            Ok(("aether_op_matmul_tn_f32_cuda", vec![s[1], s[0], bb[1]]))
        }
        // x.scale(s) → scale(x, s, n=numel(x)). In-place. `s` is a user-supplied
        // f32 arg; we synthesize n only.
        "scale" => {
            let n = recv_shape.iter().product();
            Ok(("aether_op_scale_f32_cuda", vec![n]))
        }
        // x.gelu_backward(&dy, &mut dx) → gelu_bwd(x, dy, dx, n=numel(x))
        "gelu_backward" => {
            let n = recv_shape.iter().product();
            Ok(("aether_op_gelu_backward_f32_cuda", vec![n]))
        }
        // a.add_layer_norm(&b, &gamma, &beta, &mut y, &mut mean, &mut rstd, eps)
        //   → add_layer_norm_fwd(a, b, gamma, beta, y, mean, rstd, eps, B, D)
        // Receiver is `a` ([B, D]); first arg is `b` (other addend, [B, D]).
        "add_layer_norm" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("add_layer_norm: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_add_layer_norm_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // x.layer_norm(&gamma, &beta, &mut y, &mut mean, &mut rstd, eps)
        //   → layer_norm_fwd(x, gamma, beta, y, mean, rstd, B, D, eps)
        // x is [B, D]; gamma/beta are [D]; mean/rstd are [B].
        "layer_norm" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("layer_norm: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_layer_norm_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // x.layer_norm_backward_dx(&gamma, &mean, &rstd, &dy, &mut dx)
        //   → layer_norm_bwd_dx(x, gamma, mean, rstd, dy, dx, B, D)
        "layer_norm_backward_dx" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("layer_norm_backward_dx: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_layer_norm_backward_dx_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // x.layer_norm_backward_params(&mean, &rstd, &dy, &mut dgamma, &mut dbeta)
        //   → layer_norm_bwd_params(x, mean, rstd, dy, dgamma, dbeta, B, D)
        "layer_norm_backward_params" => {
            if recv_shape.len() != 2 {
                return Err(AsmError::UnsupportedExpr("layer_norm_backward_params: receiver must be 2-dim [B, D]"));
            }
            Ok(("aether_op_layer_norm_backward_params_f32_cuda", vec![recv_shape[0], recv_shape[1]]))
        }
        // dy.matmul_backward_lhs(&w, &mut dx) → mm_bwd_lhs(dy, w, dx, M, K, N)
        //   dy: [M, N], w: [K, N], dx: [M, K]
        "matmul_backward_lhs" => {
            let s = recv_shape;
            let w = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul_backward_lhs: w must be Tensor with shape"))?;
            if s.len() != 2 || w.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul_backward_lhs: shapes must be 2-dim"));
            }
            // M = s[0] (dy.rows), N = s[1] (dy.cols), K = w[0] (w.rows)
            Ok(("aether_op_matmul_backward_lhs_f32_cuda", vec![s[0], w[0], s[1]]))
        }
        // (h2d / d2h would want a receiver-as-second-arg form; skipping
        // until we have a more flexible dispatch table or a small Aether
        // adapter fn for them.)
        other => Err(AsmError::UnsupportedExpr(string_to_static(format!("unknown method: {}", other)))),
    }
}

fn string_to_static(s: String) -> &'static str { Box::leak(s.into_boxed_str()) }
