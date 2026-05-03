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
    pending: Vec<(String, Vec<(String, i64)>, String)>,
    seen: HashSet<String>,
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
                other => new_items.push(other),
            }
        }
        p.items = new_items;
    }
    let p = &p;

    let mut sigs: HashMap<String, TyKind> = HashMap::new();
    let mut local_fns: HashSet<String> = HashSet::new();
    let mut struct_decls: HashMap<String, StructDecl> = HashMap::new();
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
        if let Item::Enum { name, variants } = item {
            for (i, v) in variants.iter().enumerate() {
                const_env.insert(format!("{}::{}", name, v), i as i64);
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
                let (floats, f64s) = emit_fn(
                    f, &mut text, &mut data, &sigs, &local_fns,
                    &struct_decls, &const_env, Some(generics.clone()))?;
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
        let Some((tname, bindings, mangled)) = next else { break; };
        let tdecl = generics.borrow().templates.get(&tname).cloned()
            .ok_or(AsmError::UnsupportedExpr("monomorphization: unknown template"))?;
        // Build the specialized FnDecl: rename, drop const_params (it's now
        // concrete), keep everything else intact. The shape Sym dims in
        // its param/return types still reference the const-param names; the
        // emit_fn we call below resolves them through the extended const_env.
        let mut spec = tdecl.clone();
        spec.name = mangled.clone();
        spec.const_params.clear();
        let mut spec_env = const_env.clone();
        for (k, v) in &bindings { spec_env.insert(k.clone(), *v); }
        // Register the specialization so cascading calls resolve.
        local_fns.insert(mangled.clone());
        if let Some(rk) = spec.ret.as_ref().and_then(|t| TyKind::from_ty_with_env(t, &spec_env)) {
            sigs.insert(mangled.clone(), rk);
            sigs.insert(format!("aether_{}", mangled), rk);
        }
        let (floats, f64s) = emit_fn(
            &spec, &mut text, &mut data, &sigs, &local_fns,
            &struct_decls, &spec_env, Some(generics.clone()))?;
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
    /// Stack arrays: name → (base_slot, n, elem_kind). `base_slot` is the
    /// slot of element 0 (closest to rbp); element k is at addr
    /// `-8*base_slot(%rbp) - 8*k`. Per-element kind only int/handle for
    /// now (8-byte slots) — float arrays would need a 4-byte stride.
    arrays: HashMap<String, (usize, usize, TyKind)>,
    /// Const-generic specialization state, shared across all fns in the program.
    /// At each call site we check `templates` for the callee; if hit, we infer
    /// concrete dim bindings from the caller's tensor_shapes, mangle, queue.
    /// `None` only in unit tests that build a Locals by hand.
    generics: Option<Rc<RefCell<GenericState>>>,
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
        let raw = self.next_slot * 8 + 32 + self.max_call_extras * 8;
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
           const_env: &HashMap<String, i64>,
           generics: Option<Rc<RefCell<GenericState>>>)
    -> Result<(Vec<(String, f32)>, Vec<(String, f64)>), AsmError>
{
    let name = if f.name == "main" { "main".to_string() } else { format!("aether_{}", f.name) };

    // Pre-pass: count locals so the prologue reserves the right amount.
    let mut locals = Locals::default();
    locals.fn_label_prefix = f.name.clone();
    locals.sigs = sigs.clone();
    locals.local_fns = local_fns.clone();
    locals.struct_decls = struct_decls.clone();
    locals.const_env = const_env.clone();
    locals.generics = generics;
    let body = f.body.as_ref().unwrap();
    // Reserve slots for incoming params so the frame includes them.
    for p in &f.params { locals.alloc(&p.name); }
    count_locals(body, &mut locals);
    // One extra slot to spill `%rax` across the auto-free callq sequence
    // in the epilogue. Wastes 8 bytes if the fn has no Tensor locals but
    // keeps the frame-sizing pass symmetric across the two count passes.
    locals.alloc("_ret_save_");
    let frame = locals.frame_bytes();
    locals.slots.clear();
    locals.next_slot = 0;
    locals.types.clear();

    let ret_kind = f.ret.as_ref().and_then(TyKind::from_ty);

    out.push_str(&format!("{name}:\n"));
    out.push_str("    pushq %rbp\n");
    out.push_str("    movq %rsp, %rbp\n");
    out.push_str(&format!("    subq ${}, %rsp\n", frame));

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

        if arg_idx >= 4 { return Err(AsmError::TooManyArgs); }
        let i = arg_idx;
        let kind = TyKind::from_ty_with_env(p_ty, &locals.const_env).unwrap_or(TyKind::Int);
        let slot = locals.alloc(&p.name);
        locals.types.insert(p.name.clone(), kind);
        // Populate the shape sidecar so method-call dispatch in the fn
        // body (`x.matmul(...)`) can read M/K/N back.
        if matches!(kind, TyKind::TensorDev(_) | TyKind::TensorDevI32(_)) {
            if let Some(shape) = tensor_type_shape(p_ty, Some(&locals.const_env)) {
                locals.tensor_shapes.insert(p.name.clone(), shape);
            }
            let elem = match kind { TyKind::TensorDevI32(_) => "i32", _ => "f32" };
            locals.tensor_elem.insert(p.name.clone(), elem);
        }
        match kind {
            TyKind::Int => out.push_str(&format!("    movq {}, -{}(%rbp)\n", int_arg_regs[i], slot * 8)),
            TyKind::F32 => out.push_str(&format!("    movss %xmm{}, -{}(%rbp)\n", i, slot * 8)),
            TyKind::F64 => out.push_str(&format!("    movsd %xmm{}, -{}(%rbp)\n", i, slot * 8)),
            // Tensor handles arrive as i64 in the integer arg reg.
            TyKind::TensorDev(_) | TyKind::TensorDevI32(_) =>
                out.push_str(&format!("    movq {}, -{}(%rbp)\n", int_arg_regs[i], slot * 8)),
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
    emit_block(body, out, data, &mut locals)?;
    locals.default_float = saved;

    // Default-zero %rax only if the fn returns an int (or has no declared ret)
    // *and* the body has no tail expression. For float returns, the tail value
    // is already in xmm0 and we leave it.
    if body.tail.is_none() && !matches!(ret_kind, Some(TyKind::F32) | Some(TyKind::F64)) {
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
                // Struct literal rhs: reserve slot per field, same as the
                // uninit-struct branch below. Skips the trailing single-slot
                // alloc since each field gets its own slot.
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
                locals.alloc(name);
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
            for (_pat, body) in arms { count_locals_in_expr(body, locals); }
        }
        Expr::Cast { expr, .. } => count_locals_in_expr(expr, locals),
        Expr::Index { recv, idx } => {
            count_locals_in_expr(recv, locals);
            count_locals_in_expr(idx, locals);
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
            // Struct literal as let rhs — desugars to "uninit struct let,
            // then per-field assignment." The struct decl's field list
            // gives types; lit's `(field_name, expr)` pairs give values.
            // Order doesn't matter — fields are matched by name.
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
            let slot = locals.get(name).ok_or_else(|| AsmError::UnknownIdent(name.clone()))?;
            let kind = locals.types.get(name).copied().unwrap_or(TyKind::Int);
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
                    for (i, tp) in tdecl.params.iter().enumerate() {
                        let p_ty = match &tp.ty {
                            Ty::Ref { inner, .. } => inner.as_ref(),
                            other => other,
                        };
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
                    // Every const_param must have a binding.
                    for cp in &tdecl.const_params {
                        if !bindings.contains_key(cp) {
                            return Err(AsmError::UnsupportedExpr(string_to_static(
                                format!("template '{}': could not infer const param '{}'", name, cp))));
                        }
                    }
                    // Stable mangling: const_params order from the template.
                    let mut sorted: Vec<(String, i64)> = tdecl.const_params.iter()
                        .map(|cp| (cp.clone(), *bindings.get(cp).unwrap()))
                        .collect();
                    let suffix: String = sorted.iter()
                        .map(|(k, v)| format!("__{}{}", k, v))
                        .collect();
                    let mangled = format!("{}{}", name, suffix);
                    {
                        let mut gm = g.borrow_mut();
                        if gm.seen.insert(mangled.clone()) {
                            sorted.sort_by(|a, b| a.0.cmp(&b.0));
                            gm.pending.push((name.clone(), sorted, mangled.clone()));
                        }
                    }
                    // Make the call resolve through the spec name.
                    locals.local_fns.insert(mangled.clone());
                    if let Some(rk) = locals.sigs.get(&name).copied() {
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
            let scrut_kind = emit_expr_value(scrutinee, out, data, locals)?;
            if !matches!(scrut_kind, TyKind::Int) {
                return Err(AsmError::UnsupportedExpr("match scrutinee must be int (or enum variant)"));
            }
            // Save scrutinee to a local for repeat compares.
            let save_slot = locals.alloc("_match_scrut_");
            out.push_str(&format!("    movq %rax, -{}(%rbp)\n", save_slot * 8));
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
        // x.matmul(&w, &mut y) → matmul_f32_cuda(x, w, y, M, K, N)
        //   x: [M, K], w: [K, N], y: [M, N]
        "matmul" => {
            let s = recv_shape;
            let w = arg_shapes.get(0).and_then(|x| x.as_ref())
                .ok_or(AsmError::UnsupportedExpr("matmul: w must be a Tensor with shape"))?;
            if s.len() != 2 || w.len() != 2 {
                return Err(AsmError::UnsupportedExpr("matmul: shapes must be 2-dim"));
            }
            Ok(("aether_op_matmul_f32_cuda", vec![s[0], s[1], w[1]]))
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
        // (h2d / d2h would want a receiver-as-second-arg form; skipping
        // until we have a more flexible dispatch table or a small Aether
        // adapter fn for them.)
        other => Err(AsmError::UnsupportedExpr(string_to_static(format!("unknown method: {}", other)))),
    }
}

fn string_to_static(s: String) -> &'static str { Box::leak(s.into_boxed_str()) }
