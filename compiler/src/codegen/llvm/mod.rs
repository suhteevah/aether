//! Text-mode LLVM IR emitter — Phase 0.
//!
//! Consumes MIR and emits a `.ll` file containing:
//! * Externs for the autodiff/distributed runtime intrinsics
//! * One LLVM function per Aether fn
//! * Tape alloca + autodiff intrinsic calls when the fn is `#[autodiff]`
//! * `@aether_dist_all_reduce` calls when the fn is `#[distributed(...)]`
//!
//! When inkwell lands in Phase 1 the textual builder here is replaced
//! function-for-function by `inkwell::builder::Builder` calls, but the MIR
//! shape and the intrinsic surface stay identical.

use crate::mir::{adgraph, MirFunction, MirProgram, MirStmt};

pub fn emit(m: &MirProgram) -> String {
    let mut s = String::new();
    s.push_str("; ModuleID = 'aether'\n");
    s.push_str("target triple = \"x86_64-pc-windows-gnu\"\n\n");

    // Runtime intrinsics — provided by libaether_rt.
    s.push_str("; --- aether autodiff + distributed runtime intrinsics ---\n");
    s.push_str("declare void @aether_autodiff_init(i8*)\n");
    s.push_str("declare void @aether_autodiff_push(i8*, i8*)\n");
    s.push_str("declare void @aether_autodiff_reverse(i8*)\n");
    s.push_str("declare void @aether_autodiff_accumulate(i8*, i8*)\n");
    s.push_str("declare void @aether_autodiff_partial(i8*, i32, i32, i32)\n");
    s.push_str("declare void @aether_dist_all_reduce(i8*, i32, i32)\n");
    s.push_str("declare i32 @puts(i8*)\n\n");

    for f in &m.funcs {
        s.push_str(&emit_fn(f));
        s.push('\n');
    }
    s
}

fn emit_fn(f: &MirFunction) -> String {
    let mut s = String::new();
    s.push_str(&format!("define i32 @{}() {{\n", f.name));
    s.push_str("entry:\n");

    if f.is_autodiff {
        s.push_str("  %tape = alloca [1024 x { i32, i32, i8* }]\n");
        s.push_str("  %tape_p = bitcast [1024 x { i32, i32, i8* }]* %tape to i8*\n");
        s.push_str("  call void @aether_autodiff_init(i8* %tape_p)\n");
    }

    let mut ssa = 0u32;
    let mut next = || -> String { let n = ssa; ssa += 1; format!("%v{}", n) };

    for st in &f.stmts {
        match st {
            MirStmt::Source(line) => {
                s.push_str(&format!("  ; aether: {}\n", line));
            }
            MirStmt::TapeInit => { /* emitted above */ }
            MirStmt::TapePush { value } => {
                let g = next();
                s.push_str(&format!(
                    "  {0} = bitcast i8* null to i8*  ; placeholder for `{1}`\n",
                    g, value
                ));
                s.push_str(&format!(
                    "  call void @aether_autodiff_push(i8* %tape_p, i8* {})\n",
                    g
                ));
            }
            MirStmt::AccumulateGrad { source } => {
                s.push_str(&format!("  ; ∂/∂({})\n", source));
                s.push_str("  call void @aether_autodiff_accumulate(i8* %tape_p, i8* null)\n");
            }
            MirStmt::TapeReverse => {
                // Lower the typed AdGraph (if any) into one
                // `aether_autodiff_partial(tape, dst, op, src)` call per
                // symbolic partial. This is what makes the IR actually
                // describe the reverse computation rather than just calling
                // an opaque `aether_autodiff_reverse`.
                if let Some(g) = &f.adgraph {
                    s.push_str("  ; --- AdGraph reverse (symbolic partials) ---\n");
                    for (id, op) in g.nodes.iter().enumerate().rev() {
                        for (dst, op_code, src) in partials_for(g, id, op) {
                            s.push_str(&format!(
                                "  call void @aether_autodiff_partial(i8* %tape_p, i32 {}, i32 {}, i32 {})\n",
                                dst, op_code, src,
                            ));
                        }
                    }
                }
                s.push_str("  call void @aether_autodiff_reverse(i8* %tape_p)\n");
            }
            MirStmt::AllReduce { tensor, world_size, backend } => {
                let backend_id: i32 = match backend.as_str() {
                    "nccl" => 0,
                    "mpi" => 1,
                    "gloo" => 2,
                    _ => 99,
                };
                s.push_str(&format!(
                    "  ; all_reduce {} across {} ranks via {}\n",
                    tensor, world_size, backend
                ));
                s.push_str(&format!(
                    "  call void @aether_dist_all_reduce(i8* null, i32 {}, i32 {})\n",
                    world_size, backend_id
                ));
            }
        }
    }

    s.push_str("  ret i32 0\n");
    s.push_str("}\n");
    s
}

/// Op codes for `aether_autodiff_partial` — kept stable so the runtime can
/// dispatch without parsing strings. Phase 1 grows this set; never reorder.
const PART_ADD: i32 = 1;
const PART_SUB_PLUS: i32 = 2;
const PART_SUB_MINUS: i32 = 3;
const PART_MUL: i32 = 4;
const PART_MATMUL_LHS: i32 = 5;
const PART_MATMUL_RHS: i32 = 6;
const PART_RELU: i32 = 7;
const PART_CROSS_ENTROPY: i32 = 8;
const PART_FORWARD_VJP: i32 = 9;
const PART_PARAM: i32 = 10;

/// Return a list of `(dst_node_id, op_code, src_node_id)` triples that
/// correspond to the symbolic partials emitted by `adgraph::reverse`.
/// `src_node_id` is the operand whose value the partial multiplies; for
/// unary ops it equals `dst`, for binary ops it's the *other* operand.
fn partials_for(_g: &adgraph::AdGraph, id: usize, op: &adgraph::Op) -> Vec<(i32, i32, i32)> {
    use adgraph::Op::*;
    let id_i = id as i32;
    match op {
        Const(_) => vec![],
        Param(_) => vec![(id_i, PART_PARAM, id_i)],
        Add(a, b) => vec![(*a as i32, PART_ADD, id_i), (*b as i32, PART_ADD, id_i)],
        Sub(a, b) => vec![(*a as i32, PART_SUB_PLUS, id_i), (*b as i32, PART_SUB_MINUS, id_i)],
        Mul(a, b) => vec![(*a as i32, PART_MUL, *b as i32), (*b as i32, PART_MUL, *a as i32)],
        MatMul(a, b) => vec![
            (*a as i32, PART_MATMUL_LHS, *b as i32),
            (*b as i32, PART_MATMUL_RHS, *a as i32),
        ],
        ReLU(x) => vec![(*x as i32, PART_RELU, *x as i32)],
        CrossEntropy { logits, labels } => vec![
            (*logits as i32, PART_CROSS_ENTROPY, *labels as i32),
        ],
        Forward { inputs, .. } => inputs.iter().map(|i| (*i as i32, PART_FORWARD_VJP, id_i)).collect(),
    }
}
