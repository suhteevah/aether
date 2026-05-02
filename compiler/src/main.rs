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
}

#[derive(Debug, Clone, Copy)]
enum Emit { Bin, Mir, LlvmIr, C, Asm, AsmBin, AetherBin }

fn parse_args() -> Result<Args, String> {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut emit = Emit::Bin;
    let mut check_only = false;
    let mut json_errors = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => output = Some(PathBuf::from(it.next().ok_or("-o needs a path")?)),
            "--emit=mir" => emit = Emit::Mir,
            "--emit=llvm-ir" => emit = Emit::LlvmIr,
            "--emit=c" => emit = Emit::C,
            "--emit=asm" => emit = Emit::Asm,
            "--emit=asm-bin" => emit = Emit::AsmBin,
            "--emit=aether-bin" => emit = Emit::AetherBin,
            "--emit=bin" => emit = Emit::Bin,
            "--check" => check_only = true,
            "--json-errors" => json_errors = true,
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
            Emit::Bin | Emit::AsmBin | Emit::AetherBin => { p.set_extension(if cfg!(windows) { "exe" } else { "" }); }
            Emit::Mir => { p.set_extension("mir"); }
            Emit::LlvmIr => { p.set_extension("ll"); }
            Emit::C => { p.set_extension("c"); }
            Emit::Asm => { p.set_extension("s"); }
        }
        p
    });
    Ok(Args { input, output, emit, check_only, json_errors })
}

fn print_help() {
    println!("aetherc - Aether compiler (Phase 0/0.5)\n");
    println!("USAGE:");
    println!("  aetherc <input.aether> [-o <out>] [--emit=bin|asm-bin|mir|llvm-ir|c|asm]");
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

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => { eprintln!("aetherc: {}", e); std::process::exit(2); }
    };

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

    let prog = match parser::Parser::new(toks).parse_program() {
        Ok(p) => p,
        Err(e) => {
            sink.push(diag::from_legacy("AE0002", "parse", &e)
                .with_hint("most parse errors are an unexpected punctuation \
                    or a missing comma between fn args / attr args"));
            report(&sink, &file_str, args.json_errors);
            std::process::exit(1);
        }
    };

    let mir_prog = mir::run_autodiff_pass(&prog);

    if args.check_only {
        if !args.json_errors {
            eprintln!("[aetherc] check OK — {} fn(s)", mir_prog.funcs.len());
        }
        report(&sink, &file_str, args.json_errors);
        return;
    }

    match args.emit {
        Emit::Mir => {
            std::fs::write(&args.output, mir::dump_mir(&mir_prog)).unwrap();
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
            std::fs::write(&args.output, codegen::asm::emit(&prog)).unwrap();
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
            std::fs::write(&s_path, codegen::asm::emit(&prog)).unwrap();
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
        Emit::AsmBin => {
            // .aether -> .s -> .exe via the system as+ld (gcc as the linker driver
            // for its msvcrt linkage convenience). Step 1 to dropping the C
            // compiler entirely; once `aether_asm/` lands, this path drops `as`,
            // and once an Aether linker lands, it drops `ld` too.
            let mut s_path = args.output.clone();
            s_path.set_extension("s");
            std::fs::write(&s_path, codegen::asm::emit(&prog)).unwrap();
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
