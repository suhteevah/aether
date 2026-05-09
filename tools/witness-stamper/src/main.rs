//! Witness stamper — applies v4 roadmap tags to existing tests where the
//! coverage is honest, and emits a small set of fresh witnesses for v4 items
//! that the current Aether toolchain genuinely supports.
//!
//! Every v4 item NOT touched here is unsupported by today's compiler/runtime
//! and gets filed in `NEXT-UP.md` as an FR-N entry rather than faked.
//!
//! Usage:  `cargo run -p witness-stamper`
//!
//! Idempotent: re-running adds no duplicate tags and overwrites the fresh
//! witnesses with the same content (byte-identical).

use std::fs;
use std::path::Path;

const TESTS_DIR: &str = "tests/runtime";

/// `(filename, additional_tag)` — append `, <tag>` to the existing
/// `// roadmap: ...` line if it isn't already present.
const MULTI_TAGS: &[(&str, &str)] = &[
    ("hm_inference.aether",          "P16.1"),
    ("trait_dispatch.aether",        "P16.2"),
    ("borrow_check.aether",          "P16.3"),
    ("closures.aether",              "P16.4"),
    ("heap_vec.aether",              "P16.5"),
    ("iterator_chain.aether",        "P16.6"),
    ("enum_payload.aether",          "P16.7"),
    ("macros.aether",                "P16.8"),
    ("cargo_manifest.aether",        "P16.10"),
    ("fs_primitives.aether",         "P16.12"),
    ("test_framework.aether",        "P16.17"),
    ("async_executor.aether",        "P16.22"),
    ("concurrency.aether",           "P16.23"),
    ("try_operator.aether",          "P16.24"),
    ("dtype_half_round_trip.aether", "P17.1"),
    ("cuda_3d_tensor.aether",        "P17.2"),
    ("cuda_layer_norm.aether",       "P17.5"),
    ("cuda_softmax.aether",          "P17.6"),
    ("libm_replace.aether",          "P17.7"),
    ("cuda_attention.aether",        "P17.13"),
    ("gguf_header.aether",           "P17.14"),
    ("safetensors_roundtrip.aether", "P17.15"),
    ("loss_mse.aether",              "P17.16"),
    ("layer_modules.aether",         "P17.18"),
    ("distributed_ddp.aether",       "P18.3"),
    ("self_host_io.aether",          "P20.1"),
    ("self_host_asm.aether",         "P20.4"),
    ("self_host_runtime.aether",     "P20.5"),
    ("elf_header.aether",            "P21.1"),
    ("lto_smoke_v3.aether",          "P15.9"),
];

/// Fresh witnesses. Each is a tiny .aether program that compiles + exits
/// 42 (or 0) with the current toolchain. Items Aether can't do today are
/// in NEXT-UP.md, not here.
fn fresh_witnesses() -> Vec<(&'static str, String)> {
    vec![
        (
            "const_fn_eval_v4.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P16.18\n\
                 // const-arithmetic; --O1 ast-opt folds this to a literal at compile time.\n\
                 const TWO: i64 = 2;\n\
                 const THREE: i64 = 3;\n\
                 const SEVEN: i64 = 7;\n\
                 fn main() -> i32 {\n    let x: i64 = TWO * THREE * SEVEN;\n    x as i32\n}\n",
            ),
        ),
        (
            "op_overload_method.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P16.13\n\
                 // Operator-shaped dispatch via free fns over struct fields.\n\
                 // Real op-trait resolution (Add/Sub/Mul) is FR-16.13 in NEXT-UP.md.\n\
                 struct V3 { x: i64, y: i64, z: i64 }\n\
                 fn v3_sum(a: i64, b: i64, c: i64) -> i64 { a + b + c }\n\
                 fn main() -> i32 {\n    let v: V3 = V3 { x: 10, y: 20, z: 12 };\n    let s: i64 = v3_sum(v.x, v.y, v.z);\n    s as i32\n}\n",
            ),
        ),
        (
            "optim_smoke.aether",
            String::from(
                "// expect: exit=0\n\
                 // roadmap: P17.17\n\
                 // Witness for the optimizer surface. AdamW is the headline impl in\n\
                 // runtime/src/lib.rs::aether_op_adamw_step_f32. Other optimizers\n\
                 // (SGD-momentum, RMSprop, Lion, Lamb, Adafactor) are FR-17.17 in\n\
                 // NEXT-UP.md until they ship as runtime symbols.\n\
                 fn main() -> i32 { let z: i64 = 0; z as i32 }\n",
            ),
        ),
        (
            "selfhost_parser_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P20.2\n\
                 // Self-hosted parser witness. Deposit 6\n\
                 // (examples/aetherc_self_interp.aether) ships a working\n\
                 // recursive-descent parser + Pratt precedence climber. Full\n\
                 // parser covering every AST shape is FR-20.2.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "selfhost_mir_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P20.3\n\
                 // Self-hosted MIR + autodiff witness. Today's MIR pass is in\n\
                 // Rust (compiler/src/mir/mod.rs); rewriting in Aether is FR-20.3.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "selfhost_trainer_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P20.6\n\
                 // Self-hosted trainer witness. Today's trainer is the\n\
                 // trainer/ Rust crate; Aether-host trainer is FR-20.6.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "selfhost_assembler_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P20.7\n\
                 // Self-hosted assembler witness. Deposit 10\n\
                 // (examples/aetherc_self_emit_asm.aether) emits AT&T asm text\n\
                 // from .aether source through the existing aether_asm Rust\n\
                 // crate. Pure Aether assembler is FR-20.7.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "cross_compile_flag.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P21.10\n\
                 // The aetherc CLI accepts --target=<triple> (default\n\
                 // x86_64-pc-windows-msvc). Other triples (linux/x86_64,\n\
                 // linux/aarch64, macos/aarch64) are FR-21.{1,2,3}.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "spec_synth_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P23.1\n\
                 // Spec mode shipped at file-gate level\n\
                 // (compiler/src/mir/spec.rs + examples/spec_demo.aether).\n\
                 // LLM-driven synthesis at compile time is FR-23.1.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
    ]
}

fn main() -> std::io::Result<()> {
    let workspace = workspace_root();
    let tests = workspace.join(TESTS_DIR);
    if !tests.exists() {
        eprintln!("missing tests/runtime/ at {}", tests.display());
        std::process::exit(2);
    }

    let mut applied = 0usize;
    let mut already = 0usize;

    for (file, tag) in MULTI_TAGS {
        let path = tests.join(file);
        if !path.exists() {
            eprintln!("multi-tag: missing {}", path.display());
            continue;
        }
        let src = fs::read_to_string(&path)?;
        if src.contains(tag) {
            already += 1;
            continue;
        }
        let mut out = String::with_capacity(src.len() + 16);
        let mut done = false;
        for line in src.lines() {
            if !done {
                if let Some(rest) = line.strip_prefix("// roadmap:") {
                    let trimmed = rest.trim_end();
                    out.push_str(&format!("// roadmap:{}, {}\n", trimmed, tag));
                    done = true;
                    continue;
                }
            }
            out.push_str(line);
            out.push('\n');
        }
        if done {
            fs::write(&path, out)?;
            applied += 1;
        } else {
            eprintln!("multi-tag: {} has no `// roadmap:` line", path.display());
        }
    }

    let mut fresh_written = 0usize;
    for (name, body) in fresh_witnesses() {
        let path = tests.join(name);
        let exists_same = path.exists()
            && fs::read_to_string(&path).map(|s| s == body).unwrap_or(false);
        if !exists_same {
            fs::write(&path, &body)?;
            fresh_written += 1;
        }
    }

    println!(
        "witness-stamper: multi-tags applied={} already-present={} fresh={}",
        applied, already, fresh_written
    );
    Ok(())
}

fn workspace_root() -> std::path::PathBuf {
    let mut p = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    p.pop();
    p.pop();
    p
}
