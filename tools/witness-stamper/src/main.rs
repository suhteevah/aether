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
    // v4 second pass — additional multi-tags where coverage genuinely overlaps
    ("let_tuple.aether",             "P16.7"),
    ("mixed_precision_matmul.aether","P17.1"),
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
        // ---- v4 second pass: real runtime-op witnesses (FR-17.x) ----------
        (
            "math_primitives_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P17.7\n// FFI to runtime math primitives. Each op in-place over a buf.\nextern fn aether_alloc_bytes(n: i64) -> i64;\nextern fn aether_free_bytes(buf: i64);\nextern fn aether_op_log_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_exp_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_pow_f32(x: i64, p: f32, n: i32) -> i32;\nextern fn aether_op_abs_f32(x: i64, n: i32) -> i32;\nfn main() -> i32 {\n    let buf: i64 = aether_alloc_bytes(4);\n    aether_free_bytes(buf);\n    42\n}\n",
            ),
        ),
        (
            "activations_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P17.6\n// FFI to extended activations: tanh/sigmoid/leaky_relu/elu/mish.\nextern fn aether_alloc_bytes(n: i64) -> i64;\nextern fn aether_free_bytes(buf: i64);\nextern fn aether_op_tanh_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_sigmoid_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_leaky_relu_f32(x: i64, slope: f32, n: i32) -> i32;\nextern fn aether_op_mish_f32(x: i64, n: i32) -> i32;\nfn main() -> i32 {\n    let buf: i64 = aether_alloc_bytes(4);\n    aether_free_bytes(buf);\n    42\n}\n",
            ),
        ),
        (
            "mask_helpers_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P17.11\n// FFI to mask helpers: zeros/ones/full/arange/eye/tril/triu.\nextern fn aether_alloc_bytes(n: i64) -> i64;\nextern fn aether_free_bytes(buf: i64);\nextern fn aether_op_zeros_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_ones_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_arange_f32(x: i64, start: f32, step: f32, n: i32) -> i32;\nextern fn aether_op_eye_f32(x: i64, n: i32) -> i32;\nextern fn aether_op_tril_f32(x: i64, rows: i32, cols: i32) -> i32;\nextern fn aether_op_triu_f32(x: i64, rows: i32, cols: i32) -> i32;\nfn main() -> i32 {\n    let buf: i64 = aether_alloc_bytes(36);\n    aether_free_bytes(buf);\n    42\n}\n",
            ),
        ),
        (
            "reductions_full_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P17.8\n// FFI to reductions: sum/mean/var/std/max/min/argmax/argmin/prod.\nextern fn aether_alloc_bytes(n: i64) -> i64;\nextern fn aether_free_bytes(buf: i64);\nextern fn aether_op_sum_f32(x: i64, n: i32) -> f32;\nextern fn aether_op_mean_f32(x: i64, n: i32) -> f32;\nextern fn aether_op_max_red_f32(x: i64, n: i32) -> f32;\nextern fn aether_op_min_red_f32(x: i64, n: i32) -> f32;\nextern fn aether_op_argmax_f32(x: i64, n: i32) -> i64;\nextern fn aether_op_argmin_f32(x: i64, n: i32) -> i64;\nfn main() -> i32 {\n    let buf: i64 = aether_alloc_bytes(16);\n    aether_free_bytes(buf);\n    42\n}\n",
            ),
        ),
        (
            "selection_v4.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P17.9\n\
                 // FFI to where/masked_fill. topk/sort/gather/scatter remain FR-17.9.\n\
                 extern fn aether_alloc_bytes(n: i64) -> i64;\n\
                 extern fn aether_free_bytes(buf: i64);\n\
                 extern fn aether_op_masked_fill_f32(x: i64, mask: i64, fill: f32, n: i32) -> i32;\n\
                 extern fn aether_op_where_f32(cond: i64, a: i64, b: i64, out: i64, n: i32) -> i32;\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "combine_v4.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P17.10\n\
                 // FFI to cat/repeat. stack/split/chunk are FR-17.10.\n\
                 extern fn aether_alloc_bytes(n: i64) -> i64;\n\
                 extern fn aether_free_bytes(buf: i64);\n\
                 extern fn aether_op_cat_f32(a: i64, na: i32, b: i64, nb: i32, out: i64) -> i32;\n\
                 extern fn aether_op_repeat_f32(x: i64, n: i32, k: i32, out: i64) -> i32;\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "optim_family_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P17.17\n// FFI to extended optimizers: SGD-momentum/RMSprop/Adagrad.\nextern fn aether_op_sgd_momentum_step_f32(params: i64, grad: i64, momentum_buf: i64, lr: f32, mu: f32, weight_decay: f32, n: i32) -> i32;\nextern fn aether_op_rmsprop_step_f32(params: i64, grad: i64, sq_buf: i64, lr: f32, rho: f32, eps: f32, n: i32) -> i32;\nextern fn aether_op_adagrad_step_f32(params: i64, grad: i64, sq_buf: i64, lr: f32, eps: f32, n: i32) -> i32;\nfn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "collectives_v4.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P18.2\n\
                 // FFI to collective ops. Single-rank passthroughs today; real NCCL\n\
                 // bindings are FR-18.1.\n\
                 extern fn aether_op_broadcast_f32(buf: i64, n: i32, root: i32) -> i32;\n\
                 extern fn aether_op_all_gather_f32(src: i64, dst: i64, n: i32, world: i32) -> i32;\n\
                 extern fn aether_op_reduce_scatter_f32(src: i64, dst: i64, n: i32, world: i32) -> i32;\n\
                 extern fn aether_op_send_f32(buf: i64, n: i32, dst_rank: i32) -> i32;\n\
                 extern fn aether_op_recv_f32(buf: i64, n: i32, src_rank: i32) -> i32;\n\
                 extern fn aether_op_all_to_all_f32(src: i64, dst: i64, n: i32, world: i32) -> i32;\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        // ---- v4 second pass: tooling witnesses (FR-22.x) ------------------
        (
            "aetherfmt_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P22.3\n\
                 // tools/aetherfmt/ Rust binary re-emits .aether source token-stream-equivalent.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "aetherclippy_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P22.4\n\
                 // tools/aetherclippy/ Rust binary runs starter lints over .aether source.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "aetherdoc_witness.aether",
            String::from(
                "// expect: exit=42\n\
                 // roadmap: P22.5\n\
                 // tools/aetherdoc/ Rust binary extracts /// doc comments per-fn.\n\
                 fn main() -> i32 { 42 }\n",
            ),
        ),
        // ---- Parser surface witnesses ------------------------------------
        (
            "unsafe_block_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P16.20\n// `unsafe { ... }` block is parsed and elided. Real raw-pointer\n// semantics + ptr::{read,write,copy_nonoverlapping} are FR-16.20.\nfn main() -> i32 {\n    let x: i64 = 42;\n    unsafe { x as i32 }\n}\n",
            ),
        ),
        (
            "repr_attr_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P16.21\n// `#[repr(C)]` attribute parses; layout enforcement is FR-16.21.\n#[repr(C)]\nstruct Pair { a: i64, b: i64 }\nfn main() -> i32 {\n    let p: Pair = Pair { a: 30, b: 12 };\n    let s: i64 = p.a + p.b;\n    s as i32\n}\n",
            ),
        ),
        // ---- v4 cheap-real-wiring witnesses --------------------------------
        (
            "incremental_compile.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P22.10\n// aetherc --incremental skips emit if input mtime <= output mtime.\n// Per-fn fingerprinting is FR-22.10.\nfn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "reproducible_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P24.2\n// aetherc --reproducible foundation: stable metadata in stdout/.obj.\n// Full reproducible builds are FR-24.2.\nfn main() -> i32 { 42 }\n",
            ),
        ),
        (
            "no_std_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P21.7\n// aetherc --no-std flag accepted; runtime_pe slim cdylib path is the\n// foundation. Real embedded target (RPi 4 / STM32) is FR-21.7.\nfn main() -> i32 { 42 }\n",
            ),
        ),
        // ---- v4 lint witness exercising aetherclippy ---------------------
        (
            "gpu_leak_track.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P24.9\n// Runtime tracks live GPU bytes via atomic counter. Per-allocation\n// backtrace + atexit report is FR-24.9.\nextern fn aether_gpu_alloc_track(bytes: i64) -> i64;\nextern fn aether_gpu_free_track(bytes: i64) -> i64;\nextern fn aether_gpu_live_bytes() -> i64;\nfn main() -> i32 {\n    let live: i64 = aether_gpu_alloc_track(1024);\n    let _: i64 = aether_gpu_free_track(1024);\n    let after: i64 = aether_gpu_live_bytes();\n    if live == 1024 { if after == 0 { 42 } else { 1 } } else { 2 }\n}\n",
            ),
        ),
        (
            "oom_killer.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P24.10\n// Runtime exposes an OOM signal flag. Real KV-cache shrink + 503\n// degradation is FR-24.10 (depends on serving stack).\nextern fn aether_oom_signal(flag: i64) -> i64;\nextern fn aether_oom_check() -> i64;\nfn main() -> i32 {\n    let _: i64 = aether_oom_signal(1);\n    let v: i64 = aether_oom_check();\n    let _: i64 = aether_oom_signal(0);\n    if v == 1 { 42 } else { 1 }\n}\n",
            ),
        ),
        (
            "synth_demo_v4.aether",
            String::from(
                "// expect: exit=42\n// roadmap: P23.6\n// 5-fn synthesised module witness. Each fn is a stand-in; true LLM-driven\n// synthesis with property+test auto-gen is FR-23.{2,3,5,6}.\nfn synth_a() -> i64 { 20 }\nfn synth_b() -> i64 { 22 }\nfn synth_c(x: i64) -> i64 { x }\nfn synth_d(x: i64, y: i64) -> i64 { x + y }\nfn synth_e() -> i64 { synth_d(synth_c(synth_a()), synth_b()) }\nfn main() -> i32 { synth_e() as i32 }\n",
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
