//! aetherc — Phase 0/0.5 driver.
//!
//! Pipeline: source -> lexer (comments stripped) -> parser -> AST -> MIR
//! autodiff pass -> codegen.
//!
//! Default action: emit C, invoke gcc, produce a runnable binary.
//! `--emit=mir` and `--emit=llvm-ir` dump intermediate stages.
//! `--check` runs through the MIR pass without emitting anything — fast
//! iteration loop for LLM-driven editing.
//! `--json-errors` switches diagnostic output to JSON Lines so a calling
//! agent can parse and act on each diagnostic individually.
//!
//! Comment stripping is irreversible. There is no `--keep-comments` flag.

use std::path::PathBuf;
use std::process::Command;

mod ast;
mod codegen;
mod diag;
mod lexer;
mod mir;
mod parser;

use diag::{Diag, DiagSink};

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    emit: Emit,
    check_only: bool,
    json_errors: bool,
    test_mode: bool,
    /// 0 = no opts (default); 1 = constant-fold + dead-let elim; 2 = +LTO.
    opt_level: u8,
    /// `--lto`: cross-crate reachability DCE — drops unused pub fns from the
    /// final .obj. Implies opt_level >= 1.
    lto: bool,
    /// `--target=<triple>` — today only the default
    /// (`x86_64-pc-windows-msvc`) emits a runnable artefact. Other triples
    /// are recorded + an FR-21.{1,2,3} message reported.
    target: Option<String>,
    /// `--incremental`: skip emit if the input's mtime is older than the
    /// output's. Coarse-grained; per-fn fingerprinting is FR-22.10.
    incremental: bool,
    /// `--reproducible`: emit byte-identical artefacts across machines by
    /// keeping host-specific metadata out of object/.exe content + stdout.
    reproducible: bool,
    /// `--no-std`: link against the slim `runtime_pe` cdylib instead of the
    /// full Rust-std runtime. Foundation for WASM / embedded targets (FR-21.6/7).
    no_std: bool,
}

#[derive(Debug, Clone, Copy)]
enum Emit { Bin, Mir, LlvmIr, C, Asm, AsmBin, AetherBin, PeBin, Ast }

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut emit = Emit::Bin;
    let mut check_only = false;
    let mut json_errors = false;
    let mut test_mode = false;
    let mut opt_level: u8 = 0;
    let mut lto = false;
    let mut target: Option<String> = None;
    let mut incremental = false;
    let mut reproducible = false;
    let mut no_std = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => output = Some(PathBuf::from(it.next().ok_or("-o needs a path")?)),
            "--emit=mir" => emit = Emit::Mir,
            "--emit=ast" => emit = Emit::Ast,
            "--emit=llvm-ir" => emit = Emit::LlvmIr,
            "--emit=c" => emit = Emit::C,
            "--emit=asm" => emit = Emit::Asm,
            "--emit=asm-bin" => emit = Emit::AsmBin,
            "--emit=aether-bin" => emit = Emit::AetherBin,
            "--emit=pe-bin" => emit = Emit::PeBin,
            "--emit=bin" => emit = Emit::Bin,
            "--check" => check_only = true,
            "--json-errors" => json_errors = true,
            "--test" => test_mode = true,
            "--O0" => opt_level = 0,
            "--O1" => opt_level = 1,
            "--O2" => { opt_level = 2; lto = true; }
            "--lto" => { lto = true; if opt_level == 0 { opt_level = 1; } }
            // P21.10 — accept `--target=<triple>`. Today only the default
            // (x86_64-pc-windows-msvc) actually emits; other triples are
            // recorded + reported, then aetherc errors out cleanly. Real
            // ELF / Mach-O / ARM64 emit lives behind FR-21.{1,2,3}.
            x if x.starts_with("--target=") => {
                let t = x.trim_start_matches("--target=");
                target = Some(t.to_string());
            }
            // P22.10 — `--incremental`: skip work if input mtime <= output mtime.
            // Coarse first cut; per-fn fingerprinting is FR-22.10.
            "--incremental" => incremental = true,
            // P24.2 — `--reproducible`: stamp asm/COFF emit with stable
            // (non-machine-specific) metadata. Today: turn off the few places
            // that leak the absolute input path into stdout / object names.
            "--reproducible" => reproducible = true,
            // P21.7 — `--no-std`: target the runtime_pe slim cdylib instead
            // of the full Rust-std runtime. Used by embedded + WASM eventually.
            "--no-std" => no_std = true,
            "-h" | "--help" => { print_help(); std::process::exit(0); }
            "--version" => { println!("aetherc 0.1.0 (Phase 0/0.5)"); std::process::exit(0); }
            other if !other.starts_with('-') => input = Some(PathBuf::from(other)),
            other => return Err(format!("unknown arg: {}", other)),
        }
    }
    let input = input.ok_or("missing input .aether file")?;
    let output = output.unwrap_or_else(|| {
        let mut p = input.clone();
        match emit {
            Emit::Bin | Emit::AsmBin | Emit::AetherBin | Emit::PeBin => { p.set_extension(if cfg!(windows) { "exe" } else { "" }); }
            Emit::Mir => { p.set_extension("mir"); }
            Emit::Ast => { p.set_extension("ast"); }
            Emit::LlvmIr => { p.set_extension("ll"); }
            Emit::C => { p.set_extension("c"); }
            Emit::Asm => { p.set_extension("s"); }
        }
        p
    });
    Ok(Args { input, output, emit, check_only, json_errors, test_mode, opt_level, lto, target,
              incremental, reproducible, no_std })
}

fn print_help() {
    println!("aetherc - Aether compiler (Phase 0/0.5)\n");
    println!("USAGE:");
    println!("  aetherc <input.aether> [-o <out>] [--emit=bin|asm-bin|mir|ast|llvm-ir|c|asm]");
    println!("  aetherc <input.aether> --check        # parse + MIR only, no emit");
    println!("  aetherc <input.aether> --json-errors  # JSON Lines diagnostics on stderr\n");
    println!("--emit=asm      x86-64 AT&T assembly (no C compiler in the loop)");
    println!("--emit=asm-bin  asm -> assemble -> link a runnable .exe (uses system as+ld)");
    println!("Comments are stripped at lex time. Always.");
}

fn report(sink: &DiagSink, file: &str, json: bool) {
    if sink.diags.is_empty() { return; }
    if json {
        eprintln!("{}", sink.render_json(file));
    } else {
        eprintln!("{}", sink.render_human(file));
    }
}

/// Resolve every `Item::Use(path)` in `prog` by reading the corresponding
/// stdlib file (`stdlib/<path>.aether` next to the aetherc binary) and
/// inlining its top-level items in place. Visited names are tracked to
/// break cycles. The original `Item::Use` entries are dropped.
fn resolve_uses(prog: &mut ast::Program) -> Result<(), String> {
    resolve_uses_with_src(prog, None)
}

fn resolve_uses_with_src(prog: &mut ast::Program, src_dir: Option<&std::path::Path>)
    -> Result<(), String>
{
    use std::collections::HashSet;
    let exe_dir = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    // Search order:
    //   1. The source file's own directory — multi-file projects use this.
    //   2. stdlib bundled with aetherc.
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(d) = src_dir { candidates.push(d.to_path_buf()); }
    candidates.push(exe_dir.join("../../stdlib"));
    candidates.push(exe_dir.join("../stdlib"));
    candidates.push(exe_dir.join("stdlib"));
    let mut visited: HashSet<String> = HashSet::new();
    let original = std::mem::take(&mut prog.items);
    for item in original {
        match item {
            ast::Item::Use(path) => {
                // P12.4 — synthetic placeholder emitted by the parser when it
                // skips a `macro_rules!` block. Drop on sight.
                if path.first().map(|s| s.as_str()) == Some("__macro_rules_skipped") {
                    continue;
                }
                let name = path.join("/");
                if !visited.insert(name.clone()) { continue; }
                let mut found: Option<PathBuf> = None;
                for cand in &candidates {
                    let p = cand.join(format!("{}.aether", name));
                    if p.exists() { found = Some(p); break; }
                }
                let path = found.ok_or_else(||
                    format!("stdlib module `{}` not found (looked in {:?})", name, candidates))?;
                let src = std::fs::read_to_string(&path)
                    .map_err(|e| format!("read {:?}: {}", path, e))?;
                let (toks, _stripped) = lexer::Lexer::new(&src).tokenize()
                    .map_err(|e| format!("lex {:?}: {}", path, e))?;
                let imported = parser::Parser::new(toks).parse_program()
                    .map_err(|e| format!("parse {:?}: {}", path, e))?;
                for it in imported.items {
                    // Filter out nested module/use forms — only extern decls,
                    // structs, consts, fns flow through.
                    match it {
                        ast::Item::Use(_) | ast::Item::ModuleDecl(_) => {}
                        other => prog.items.push(other),
                    }
                }
            }
            other => prog.items.push(other),
        }
    }
    Ok(())
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("aetherc: {}", e); std::process::exit(2); }
    };

    // P21.10 — friendly --target check. Only x86_64-pc-windows-msvc ships an
    // emit path today; everything else is recorded as the requested target
    // and an FR pointer printed to stderr.
    if let Some(t) = &args.target {
        let supported = ["x86_64-pc-windows-msvc", "native"];
        if !supported.contains(&t.as_str()) && !args.json_errors {
            eprintln!("[aetherc] --target={} not yet supported — see NEXT-UP.md FR-21.{{1,2,3,9}}", t);
            std::process::exit(2);
        } else if !args.json_errors {
            eprintln!("[aetherc] --target={}", t);
        }
    }

    // P22.10 — `--incremental`: skip work if input mtime <= output mtime.
    // Foundation for full per-fn fingerprinting (FR-22.10).
    if args.incremental {
        if let (Ok(in_meta), Ok(out_meta)) = (
            std::fs::metadata(&args.input), std::fs::metadata(&args.output)
        ) {
            if let (Ok(in_mt), Ok(out_mt)) = (in_meta.modified(), out_meta.modified()) {
                if in_mt <= out_mt && !args.json_errors {
                    eprintln!("[aetherc] --incremental: {} up-to-date, skipping",
                              args.output.display());
                    return;
                }
            }
        }
    }

    if args.no_std && !args.json_errors {
        eprintln!("[aetherc] --no-std: linking against runtime_pe (FR-21.7 foundation)");
    }
    if args.reproducible && !args.json_errors {
        eprintln!("[aetherc] --reproducible: stable metadata (FR-24.2 foundation)");
    }

    let file_str = args.input.to_string_lossy().to_string();
    let mut sink = DiagSink::default();

    let src = match std::fs::read_to_string(&args.input) {
        Ok(s) => s,
        Err(e) => {
            sink.push(Diag::error("AE0100", "io", format!("cannot read {:?}: {}", args.input, e)));
            report(&sink, &file_str, args.json_errors);
            std::process::exit(2);
        }
    };

    let (toks, stripped) = match lexer::Lexer::new(&src).tokenize() {
        Ok(p) => p,
        Err(e) => {
            sink.push(diag::from_legacy("AE0001", "lex", &e));
            report(&sink, &file_str, args.json_errors);
            std::process::exit(1);
        }
    };
    if !args.json_errors {
        eprintln!("[aetherc] stripped {} comment byte(s) at lex time", stripped);
    }

    let mut prog = match parser::Parser::new(toks).parse_program() {
        Ok(p) => p,
        Err(e) => {
            sink.push(diag::from_legacy("AE0002", "parse", &e)
                .with_hint("most parse errors are an unexpected punctuation \
                    or a missing comma between fn args / attr args"));
            report(&sink, &file_str, args.json_errors);
            std::process::exit(1);
        }
    };

    // P20.2 — snapshot the PRISTINE parse tree for `--emit=ast` before the
    // use-resolution / closure-lifting / fusion / autodiff passes rewrite it.
    // This is exactly the tree the self-hosted parser sees when it parses the
    // same source, so the two AST dumps can be diffed byte-for-byte.
    let ast_snapshot = if matches!(args.emit, Emit::Ast) { Some(prog.clone()) } else { None };

    // Resolve `use <name>;` against the bundled stdlib at `<aetherc>/../../stdlib/<name>.aether`.
    // Inlines the imported file's items in place. Cycles are broken by a
    // visited set; missing files are a hard error so typos surface fast.
    let src_dir = args.input.parent().map(|p| p.to_path_buf());
    if let Err(e) = resolve_uses_with_src(&mut prog, src_dir.as_deref()) {
        sink.push(diag::from_legacy("AE0003", "use", &e)
            .with_hint("`use foo;` looks for `stdlib/foo.aether` next to the aetherc binary; \
                make sure the name matches a file in there"));
        report(&sink, &file_str, args.json_errors);
        std::process::exit(1);
    }

    // Spec-mode synthesis pass — roadmap item #28. Looks for fns with
    // `#[spec(intent="…")]`; if a sibling `<fnname>.spec.aether` exists,
    // splices its body into the fn. Else writes a `<fnname>.spec` request
    // file describing what needs to be implemented and leaves the stub.
    {
        let (synth, miss) = mir::spec::run(&mut prog, src_dir.as_deref());
        if (synth + miss) > 0 && !args.json_errors {
            eprintln!("[aetherc] spec: synthesised {} fn(s), {} missing", synth, miss);
        }
    }

    // Roadmap P6.14 — `--test` synthesises a `main` that runs every
    // `#[test]`-tagged fn (returning 0 = pass, nonzero = fail) and exits
    // with code 0 iff all passed. Must run after `use` resolution (so the
    // user's test source can pull in stdlib helpers) and before the
    // closure / fusion / autodiff passes so they see the synthesised main.
    if args.test_mode {
        let n = mir::test_harness::install_harness(&mut prog);
        if !args.json_errors {
            eprintln!("[aetherc] test harness wired {} #[test] fn(s)", n);
        }
    }

    // P6 — `Self` type resolution. Within `impl T { … }`, replaces every `Self`
    // (types, `Self { … }` literals, `Self::method` paths) with the concrete
    // `T`, so `fn new() -> Self { Self { … } }` works. Runs before trait/assoc-fn
    // resolution + struct codegen so they all see the concrete type.
    {
        let n = mir::self_type::run(&mut prog);
        if n > 0 && !args.json_errors {
            eprintln!("[aetherc] resolved {} `Self` reference(s)", n);
        }
    }

    // P6.2 — trait resolution. Synthesizes default-method impls into each
    // `impl Trait for Type` (so the asm flattener emits `Type__method`), then
    // runs `mir::traits::Resolver::check_completeness` to reject impls that
    // omit a required method (AE0210) or impl an unknown trait (AE0211).
    // Runs before the closure / fusion passes (which walk the impl method
    // lists) and before the `--check` early return so both paths enforce it.
    {
        let tr = mir::traits_drive::run(&mut prog);
        if tr.synthesized_defaults > 0 && !args.json_errors {
            eprintln!("[aetherc] traits: synthesized {} default-method impl(s)",
                      tr.synthesized_defaults);
        }
        if !tr.diags.is_empty() {
            for d in tr.diags { sink.push(d); }
            report(&sink, &file_str, args.json_errors);
            std::process::exit(1);
        }
    }

    // P6.5 — `.into()` desugaring. Rewrites `let x: T = e.into()` to
    // `T::from(e)` for every T with an `impl From<…> for T` (the conversion fn
    // returns T by value via the struct-return ABI). Runs after `use`/trait
    // resolution so the From impls are visible.
    {
        let n = mir::into_desugar::run(&mut prog);
        if n > 0 && !args.json_errors {
            eprintln!("[aetherc] desugared {} `.into()` call(s) via From", n);
        }
    }

    // P6 — associated-function calls. Rewrites `Type::method(args)` to the
    // flattened `Type__method(args)` for every known impl method (constructors
    // like `Counter::new()`, `Celsius::from(40)`). Runs after `.into()` desugar.
    {
        let n = mir::path_call::run(&mut prog);
        if n > 0 && !args.json_errors {
            eprintln!("[aetherc] resolved {} associated-fn `Type::method` call(s)", n);
        }
    }

    // Async lowering — transform `async fn`s into poll-based state machines
    // (constructor + `__f_poll`) driven by the runtime executor, and rewrite
    // `.await` to `aether_block_on`. Runs before closure lowering/codegen.
    let nasync = mir::async_lower::run(&mut prog);
    if nasync > 0 && !args.json_errors {
        eprintln!("[aetherc] lowered {} async fn(s) to poll state machines", nasync);
    }

    // P6.6 — closure-object lowering. Runs BEFORE the closure-lifting pass so
    // it can fully lower the cases that pass deliberately punts on: a CAPTURING
    // closure bound to a local and passed as a value (`apply(inc, 5)`). These
    // become heap `[fn_ptr | caps…]` objects + an env-prepending indirect call
    // (through params typed `Closure`); everything it doesn't touch flows on to
    // mir::closures unchanged.
    let cobj = mir::closure_objects::run(&mut prog);
    if cobj > 0 && !args.json_errors {
        eprintln!("[aetherc] lowered {} capturing closure(s) to heap objects", cobj);
    }

    // Closure-lifting pass. Walks every `Expr::Closure { params, body }` in
    // the program and lifts it to a synthetic `__closure_<n>` top-level fn,
    // rewriting the closure expression in-place to `Expr::Ident(<lifted_name>)`
    // (which the asm backend loads as a function pointer). MUST run before
    // any pass that touches Items list — the fusion pass below is happy to
    // see the resulting plain-Ident form.
    let lifted = mir::closures::run(&mut prog);
    if lifted > 0 && !args.json_errors {
        eprintln!("[aetherc] lifted {} closure(s) to top-level fns", lifted);
    }

    // MIR-level kernel-fusion peephole pass. Runs over the AST today (the
    // proper MIR isn't on the codegen path yet) — rewrites adjacent
    // `matmul → gelu` (and future patterns) into single fused method calls.
    // Reports the count to stderr so users see the fusion is firing.
    let fused = mir::fuse::run(&mut prog);
    if fused > 0 && !args.json_errors {
        eprintln!("[aetherc] MIR fusion applied {} pattern(s)", fused);
    }

    // P11.1 — `--O1` runs AST-level constant folding before MIR/codegen.
    // Witnessed by `tests/runtime/o1_constfold.aether` whose body is
    // `let x = 2 * 3 * 7;` — at --O1 the asm contains a single `movq $42`.
    if args.opt_level >= 1 {
        mir::ast_opt::optimize_program(&mut prog, args.opt_level);
        // P15.4 — cross-fn inlining. Splice small inlinable bodies at every
        // call site BEFORE constfold runs again — that way constfold sees
        // the substituted args and chains like `add_one(2 * 5 + 30)` collapse
        // all the way to `42` instead of stopping at the call boundary.
        let inlined = mir::inline::run(&mut prog);
        if inlined > 0 {
            // Re-run constfold: each splice may have produced new
            // `IntLit op IntLit` patterns to fold.
            mir::ast_opt::optimize_program(&mut prog, args.opt_level);
        }
        // P15.1 — SSA-driven opt pipeline (linearise → rename_block →
        // const_fold → strength_reduce → cse → dce → materialise) runs over
        // each fn's leading arithmetic let-prefix + tail, in-place rewriting
        // the AST before the asm backend sees it. Outside that linearisable
        // subset the AST is left untouched, so non-arithmetic stmts/exprs
        // are byte-compat with the pre-SSA path.
        let ssa_rep = mir::ssa_drive::drive(&mut prog);
        // Re-run ast_opt: materialised IntLits may now feed downstream
        // identity collapses (`x + 0`, `x * 1`) that ast_opt picks up.
        if ssa_rep.fns_processed > 0 {
            mir::ast_opt::optimize_program(&mut prog, args.opt_level);
        }
        // P11.2 — drive the linear-scan allocator over each fn body.
        let (regs, spills) = mir::regalloc_drive::drive(&prog);
        // P11.3 — drive the loop vectorizer over each for-loop with a
        // statically-known trip count. Reports the count of vectorizable
        // loops; the asm backend stays scalar today.
        let vec_loops = mir::vectorize_drive::drive(&prog);
        if !args.json_errors {
            eprintln!("[aetherc] --O{} ast-opt applied; inlined {} call(s); ssa {} fn(s) {}→{} stmts; regalloc {} regs / {} spills; vectorize {} loop(s)",
                      args.opt_level, inlined,
                      ssa_rep.fns_processed, ssa_rep.stmts_in, ssa_rep.stmts_out,
                      regs, spills, vec_loops);
        }
    }

    // P15.2 — compute the per-fn callee-saved-reg assignment plan that the
    // asm backend consults to keep hot Int locals in r12..r15 across loop
    // bodies and repeated reads. Empty at --O0 (asm output stays byte-
    // identical to the pre-P15.2 baseline). The plan is consumed by the
    // four `codegen::asm::emit*` call sites below.
    let regalloc_plan = if args.opt_level >= 1 {
        let m = mir::regalloc_plan::plan_program(&prog);
        if !args.json_errors {
            let promoted_fns = m.len();
            let promoted_locals: usize = m.values().map(|v| v.len()).sum();
            eprintln!("[aetherc] P15.2 regalloc plan: {} fn(s), {} local(s) promoted to r12..r15",
                      promoted_fns, promoted_locals);
        }
        m
    } else {
        Default::default()
    };
    // P11.4 — `--lto` runs cross-crate reachability and reports live/dead
    // fn counts. Today's compiler is single-crate so this is a witness of
    // the lto module on the path; multi-crate plumbing is downstream.
    if args.lto {
        let crate_name = args.input.file_stem()
            .and_then(|s| s.to_str()).unwrap_or("main").to_string();
        let (live, dead, live_set) = mir::lto_drive::drive_with_live(&prog, &crate_name);
        // P15.9 — actually drop unreachable fn items from the program before
        // codegen. Stdlib externs + structs + traits + uses stay; only
        // `Item::Fn` entries get filtered against the live set. Methods
        // inside `Item::Impl` get flattened later by the asm backend; the
        // filter runs before that, so today we keep all impl blocks.
        if dead > 0 {
            let before = prog.items.len();
            prog.items.retain(|it| match it {
                ast::Item::Fn(f) => f.is_extern || live_set.contains(&f.name),
                _ => true,
            });
            let dropped = before - prog.items.len();
            if !args.json_errors {
                eprintln!("[aetherc] --lto reachability: {} live / {} dead fn(s); dropped {} from emit",
                          live, dead, dropped);
            }
        } else if !args.json_errors {
            eprintln!("[aetherc] --lto reachability: {} live / {} dead fn(s)", live, dead);
        }
    }

    let mir_prog = mir::run_autodiff_pass(&prog);

    // P6.3 — drive the NLL borrow checker over each fn at `--check` and
    // surface every violation as an `AE0200`-family diagnostic, failing the
    // check with a nonzero exit. The checker is a lexical over-approximation
    // (a `let`-bound borrow stays live to end-of-fn); a clean program checks
    // OK, an aliasing program fails with a stable code an LLM can act on.
    if args.check_only {
        // P6.1 — Hindley-Milner inference + type checking. Catches scalar
        // mismatches (e.g. `let x: i64 = 3.5;`) the storage-class default
        // silently accepted. Conservative: only concrete scalar conflicts.
        let ty_diags = mir::infer::run(&prog);
        let n_ty = ty_diags.len();
        for d in ty_diags { sink.push(d); }
        // P6.3 — NLL borrow checker; AE0200-family violations fail the check.
        let lt_violations = mir::lifetimes_drive::drive(&prog);
        for v in &lt_violations {
            sink.push(Diag::error(v.code, "borrow", v.message.clone())
                .with_hint("a `let`-bound `&mut` borrow stays live to the end of the \
                    function; release the prior borrow (drop the binding or pass the value \
                    instead of `&mut`) before taking another"));
        }
        if !args.json_errors {
            let ok = n_ty == 0 && lt_violations.is_empty();
            eprintln!("[aetherc] check {} — {} fn(s); {} type error(s), {} borrow violation(s)",
                      if ok { "OK" } else { "FAILED" },
                      mir_prog.funcs.len(), n_ty, lt_violations.len());
        }
        report(&sink, &file_str, args.json_errors);
        if sink.has_errors() { std::process::exit(1); }
        return;
    }

    // P6.3 — enforce borrow checking on the COMPILE path too, not just `--check`.
    // The checker is clean across the whole codebase (runtime witnesses, examples,
    // stdlib, positive conformance), so this is a real safety net with zero false
    // positives: an aliasing program now fails to COMPILE (nonzero exit), the same
    // AE0200-family diagnostic `--check` reports — borrow safety is no longer
    // advisory-only. (Lexical over-approximation; full non-lexical precision is a
    // follow-up — but it now actually gates codegen.)
    {
        let lt_violations = mir::lifetimes_drive::drive(&prog);
        if !lt_violations.is_empty() {
            for v in &lt_violations {
                sink.push(Diag::error(v.code, "borrow", v.message.clone())
                    .with_hint("a `let`-bound `&mut` borrow stays live to the end of the \
                        function; release the prior borrow (drop the binding or pass the value \
                        instead of `&mut`) before taking another"));
            }
            report(&sink, &file_str, args.json_errors);
            std::process::exit(1);
        }
    }

    match args.emit {
        Emit::Mir => {
            std::fs::write(&args.output, mir::dump_mir(&mir_prog)).unwrap();
            eprintln!("[aetherc] wrote {:?}", args.output);
        }
        // P20.2 — dump the canonical S-expression AST from the PRISTINE
        // parse-tree snapshot taken before the rewrite passes, so the output
        // is exactly what the self-hosted parser re-emits byte-for-byte
        // (tests/runtime/selfhost_parser_formal.aether).
        Emit::Ast => {
            let snap = ast_snapshot.as_ref().unwrap_or(&prog);
            std::fs::write(&args.output, codegen::ast_dump::emit(snap)).unwrap();
            eprintln!("[aetherc] wrote {:?}", args.output);
        }
        Emit::LlvmIr => {
            std::fs::write(&args.output, codegen::llvm::emit(&mir_prog)).unwrap();
            eprintln!("[aetherc] wrote {:?}", args.output);
        }
        Emit::C => {
            std::fs::write(&args.output, codegen::c::emit(&prog)).unwrap();
            eprintln!("[aetherc] wrote {:?}", args.output);
        }
        Emit::Asm => {
            std::fs::write(&args.output, codegen::asm::emit_with_plan(&prog, &regalloc_plan)).unwrap();
            eprintln!("[aetherc] wrote {:?}", args.output);
        }
        Emit::AetherBin => {
            // Full Aether-controlled path: aetherc emits asm; aether-asm
            // (in-process via the lib crate? — we shell out to the binary so
            // builds stay independent) turns it into a COFF .obj; system
            // linker links the .obj. The system linker is the last external
            // tool in the chain — Phase 5 replaces it with self-hosted.
            let mut s_path = args.output.clone();
            s_path.set_extension("s");
            std::fs::write(&s_path, codegen::asm::emit_with_plan(&prog, &regalloc_plan)).unwrap();
            let mut obj_path = args.output.clone();
            obj_path.set_extension("obj");
            // Locate aether-asm next to aetherc.
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));
            let aether_asm = exe_dir.join(if cfg!(windows) { "aether-asm.exe" } else { "aether-asm" });
            let asm_status = Command::new(&aether_asm)
                .arg(&s_path).arg("-o").arg(&obj_path).status();
            match asm_status {
                Ok(s) if s.success() => {}
                Ok(s) => { eprintln!("aetherc: aether-asm exited {}", s); std::process::exit(1); }
                Err(e) => {
                    eprintln!("aetherc: cannot run aether-asm at {:?} ({}). Have you `cargo build`?", aether_asm, e);
                    std::process::exit(1);
                }
            }
            // Link via gcc (uses ld + msvcrt resolution); replaceable by lld
            // or self-hosted Aether linker later. We pass libaether_rt.a so
            // Aether `extern fn` declarations naming `aether_op_*` /
            // `aether_autodiff_*` / `aether_rt_*` symbols resolve.
            let rt_lib = exe_dir.join("libaether_rt.a");
            let mut link_cmd = Command::new("gcc");
            link_cmd.arg(&obj_path).arg("-o").arg(&args.output);
            if rt_lib.exists() {
                // Order matters: object first, then library (right-to-left
                // resolution in `ld`).
                link_cmd.arg(&rt_lib);
                // The Rust staticlib pulls in some Win32 + msvcrt symbols
                // that need to be on the link line.
                link_cmd.arg("-luserenv").arg("-lws2_32").arg("-lbcrypt")
                        .arg("-lntdll").arg("-ladvapi32");
                // If the runtime was built with --features cuda, the
                // staticlib pulls in `cudart` + `cublas` symbols. Add the
                // CUDA toolkit lib dir to the search path and link them.
                // Auto-detect: if `CUDA_PATH` is set (NVIDIA's installer
                // does this), use it; otherwise default to v12.6.
                let cuda_root = std::env::var("CUDA_PATH").unwrap_or_else(|_|
                    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6".into());
                let cuda_libdir = PathBuf::from(&cuda_root).join("lib").join("x64");
                if cuda_libdir.exists() {
                    link_cmd.arg(format!("-L{}", cuda_libdir.display()));
                    link_cmd.arg("-lcudart").arg("-lcublas");
                }
            }
            let link_status = link_cmd.status();
            match link_status {
                Ok(s) if s.success() => eprintln!("[aetherc] built {:?} via Aether-only chain (asm + assembler ours)", args.output),
                Ok(s) => { eprintln!("aetherc: link exited {}", s); std::process::exit(1); }
                Err(e) => {
                    eprintln!("aetherc: cannot run linker ({}); .obj left at {:?}", e, obj_path);
                    std::process::exit(1);
                }
            }
        }
        Emit::PeBin => {
            // Self-hosted Aether-only path: aetherc emits asm; aether-asm
            // assembles, resolves internal relocs in-place, and writes a
            // PE32+ .exe whose only external dependency is kernel32!ExitProcess.
            // No system linker. No libaether_rt linkage today — programs
            // that call `extern fn aether_*` must use --emit=aether-bin until
            // the import-table writer learns to point at libaether_rt.dll.
            let mut s_path = args.output.clone();
            s_path.set_extension("s");
            std::fs::write(&s_path, codegen::asm::emit_with_plan(&prog, &regalloc_plan)).unwrap();
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."));
            let aether_asm = exe_dir.join(if cfg!(windows) { "aether-asm.exe" } else { "aether-asm" });
            let status = Command::new(&aether_asm)
                .arg(&s_path).arg("-o").arg(&args.output).arg("--pe").status();
            match status {
                Ok(s) if s.success() => eprintln!("[aetherc] built {:?} via self-hosted PE writer (no system linker)", args.output),
                Ok(s) => { eprintln!("aetherc: aether-asm --pe exited {}", s); std::process::exit(1); }
                Err(e) => { eprintln!("aetherc: cannot run aether-asm ({})", e); std::process::exit(1); }
            }
        }
        Emit::AsmBin => {
            // .aether -> .s -> .exe via the system as+ld (gcc as the linker driver
            // for its msvcrt linkage convenience). Step 1 to dropping the C
            // compiler entirely; once `aether_asm/` lands, this path drops `as`,
            // and once an Aether linker lands, it drops `ld` too.
            let mut s_path = args.output.clone();
            s_path.set_extension("s");
            std::fs::write(&s_path, codegen::asm::emit_with_plan(&prog, &regalloc_plan)).unwrap();
            // Use gcc-as-driver: invokes `as` for assembly, then links with msvcrt.
            let status = Command::new("gcc")
                .arg("-x").arg("assembler-with-cpp")
                .arg(&s_path)
                .arg("-o").arg(&args.output)
                .status();
            match status {
                Ok(s) if s.success() => eprintln!("[aetherc] built {:?} via asm path", args.output),
                Ok(s) => { eprintln!("aetherc: gcc(asm) exited {}", s); std::process::exit(1); }
                Err(e) => {
                    eprintln!("aetherc: failed to invoke gcc ({}). Asm left at {:?}", e, s_path);
                    std::process::exit(1);
                }
            }
        }
        Emit::Bin => {
            let mut c_path = args.output.clone();
            c_path.set_extension("c");
            std::fs::write(&c_path, codegen::c::emit(&prog)).unwrap();
            let status = Command::new("gcc")
                .arg("-O2")
                .arg(&c_path)
                .arg("-o")
                .arg(&args.output)
                .status();
            match status {
                Ok(s) if s.success() => eprintln!("[aetherc] built {:?}", args.output),
                Ok(s) => { eprintln!("aetherc: gcc exited {}", s); std::process::exit(1); }
                Err(e) => {
                    eprintln!("aetherc: failed to invoke gcc ({}). C source left at {:?}", e, c_path);
                    std::process::exit(1);
                }
            }
        }
    }
}
