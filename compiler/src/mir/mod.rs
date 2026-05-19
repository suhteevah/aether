//! MIR — mid-level IR for AI passes.
//!
//! Phase 0 implements just enough to demonstrate the autodiff + distributed
//! lowering shape from the handoff spec. The pass:
//!
//! 1. Walks each AST `FnDecl`.
//! 2. If the fn carries `#[autodiff]`, wraps its body with tape init / push /
//!    reverse calls, and rewrites every `expr.backward()` method call into an
//!    explicit `AccumulateGrad` MIR statement.
//! 3. If the fn carries `#[distributed(world_size=N, backend="…")]`,
//!    appends an `AllReduce` after the reverse sweep — this is what the
//!    backend will lower to `ncclAllReduce` / `MPI_Allreduce`.
//!
//! The result is intentionally a *shape*, not a full graph IR. The MIR is the
//! single source of truth for AI transforms once Phase 1 lands a real graph.

use crate::ast::*;

pub mod adgraph;
pub mod fuse;
pub mod closures;
pub mod spec;
pub mod ssa;
pub mod opt;
pub mod regalloc;
pub mod vectorize;
pub mod lto;
pub mod traits;
pub mod lifetimes;
pub mod async_exec;
pub mod ast_opt;
pub mod inline;
pub mod regalloc_drive;
pub mod vectorize_drive;
pub mod lto_drive;
pub mod lifetimes_drive;
pub mod ssa_drive;
pub mod regalloc_plan;
pub mod macros;
pub mod test_harness;

#[derive(Debug, Clone)]
pub struct MirProgram {
    pub funcs: Vec<MirFunction>,
}

#[derive(Debug, Clone)]
pub struct MirFunction {
    pub name: String,
    pub is_autodiff: bool,
    pub distributed: Option<DistributedSpec>,
    pub stmts: Vec<MirStmt>,
    pub adgraph: Option<adgraph::AdGraph>,
}

#[derive(Debug, Clone)]
pub struct DistributedSpec {
    pub world_size: i64,
    pub backend: String,
    pub algorithm: String,
}

#[derive(Debug, Clone)]
pub enum MirStmt {
    /// Lowered/passthrough source statement, kept as a string for Phase 0
    /// readability when dumping `--emit=mir`.
    Source(String),
    TapeInit,
    TapePush { value: String },
    /// `name.backward()` → reverse sweep + per-grad accumulation.
    AccumulateGrad { source: String },
    TapeReverse,
    AllReduce { tensor: String, world_size: i64, backend: String },
}

#[derive(Debug, Clone)]
pub struct TapeEntry {
    pub op: String,
    pub inputs: Vec<String>,
}

pub fn run_autodiff_pass(prog: &Program) -> MirProgram {
    let mut funcs = Vec::new();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            funcs.push(lower_fn(f));
        }
    }
    MirProgram { funcs }
}

fn lower_fn(f: &FnDecl) -> MirFunction {
    let is_autodiff = f.attrs.iter().any(|a| a.name == "autodiff");
    let distributed = f.attrs.iter().find(|a| a.name == "distributed").map(|a| {
        DistributedSpec {
            world_size: a.arg_int("world_size").unwrap_or(1),
            backend: a.arg_str("backend").unwrap_or("nccl").to_string(),
            algorithm: a.arg_str("algorithm").unwrap_or("ring").to_string(),
        }
    });

    let mut stmts = Vec::new();

    if is_autodiff { stmts.push(MirStmt::TapeInit); }

    if let Some(body) = &f.body {
        for s in &body.stmts {
            lower_stmt(s, is_autodiff, &mut stmts);
        }
        if let Some(tail) = &body.tail {
            lower_expr_stmt(tail, is_autodiff, &mut stmts);
        }
    } else {
        // extern fn: no body. Still emit a marker so MIR dumps remain readable.
        stmts.push(MirStmt::Source(format!("extern (forward decl)")));
    }

    if is_autodiff { stmts.push(MirStmt::TapeReverse); }

    if let Some(d) = &distributed {
        // Insert an AllReduce on the conceptual gradient tensor "grads".
        // Phase 1 will discover the real names from the autodiff graph.
        stmts.push(MirStmt::AllReduce {
            tensor: "grads".into(),
            world_size: d.world_size,
            backend: d.backend.clone(),
        });
    }

    let adgraph = if is_autodiff { Some(adgraph::build(f)) } else { None };

    MirFunction {
        name: f.name.clone(),
        is_autodiff,
        distributed,
        stmts,
        adgraph,
    }
}

fn lower_stmt(s: &Stmt, autodiff: bool, out: &mut Vec<MirStmt>) {
    match s {
        Stmt::Let { name, value: Some(value), .. } => {
            out.push(MirStmt::Source(format!("let {} = {}", name, render_expr(value))));
            if autodiff && expr_is_diff_relevant(value) {
                out.push(MirStmt::TapePush { value: name.clone() });
            }
        }
        Stmt::Let { name, value: None, .. } => {
            out.push(MirStmt::Source(format!("let {}: <uninit>", name)));
        }
        Stmt::LetTuple { names, value } => {
            out.push(MirStmt::Source(format!("let ({}) = {}", names.join(", "), render_expr(value))));
        }
        Stmt::Expr(e) => lower_expr_stmt(e, autodiff, out),
        Stmt::Return(Some(e)) => out.push(MirStmt::Source(format!("return {}", render_expr(e)))),
        Stmt::Return(None) => out.push(MirStmt::Source("return".into())),
    }
}

fn lower_expr_stmt(e: &Expr, autodiff: bool, out: &mut Vec<MirStmt>) {
    if autodiff {
        if let Expr::MethodCall { recv, name, .. } = e {
            if name == "backward" {
                out.push(MirStmt::AccumulateGrad { source: render_expr(recv) });
                return;
            }
        }
        if let Expr::Bin { op: BinOp::Assign, lhs, rhs } = e {
            if let Expr::MethodCall { recv, name, .. } = rhs.as_ref() {
                if name == "backward" {
                    out.push(MirStmt::Source(format!("let {} = ", render_expr(lhs))));
                    out.push(MirStmt::AccumulateGrad { source: render_expr(recv) });
                    return;
                }
            }
        }
    }
    out.push(MirStmt::Source(render_expr(e)));
}

fn expr_is_diff_relevant(e: &Expr) -> bool {
    // Phase 0 heuristic: any call/method-call may participate in the tape.
    matches!(e, Expr::Call { .. } | Expr::MethodCall { .. } | Expr::Bin { .. })
}

fn render_expr(e: &Expr) -> String {
    match e {
        Expr::IntLit(n) => n.to_string(),
        Expr::FloatLit(f) => format!("{f}"),
        Expr::StrLit(s) => format!("{:?}", s),
        Expr::BoolLit(b) => b.to_string(),
        Expr::Ident(s) => s.clone(),
        Expr::Path(p) => p.join("::"),
        Expr::Call { callee, args } => format!(
            "{}({})", render_expr(callee), args.iter().map(render_expr).collect::<Vec<_>>().join(", ")
        ),
        Expr::MethodCall { recv, name, args } => format!(
            "{}.{}({})", render_expr(recv), name,
            args.iter().map(render_expr).collect::<Vec<_>>().join(", ")
        ),
        Expr::Field { recv, name } => format!("{}.{}", render_expr(recv), name),
        Expr::Bin { op, lhs, rhs } => format!("{} {} {}", render_expr(lhs), bin_op_str(*op), render_expr(rhs)),
        Expr::Unary { op, expr } => format!("{}{}", un_op_str(*op), render_expr(expr)),
        Expr::Block(_) => "<block>".into(),
        Expr::If { .. } => "<if>".into(),
        Expr::For { parallel, distributed, .. } => {
            let mut tag = String::from("<for");
            if *parallel { tag.push_str(" parallel"); }
            if *distributed { tag.push_str(" distributed"); }
            tag.push('>');
            tag
        }
        Expr::While { .. } => "<while>".into(),
        Expr::Break => "break".into(),
        Expr::Continue => "continue".into(),
        Expr::Range { lo, hi, .. } => format!("{}..{}", render_expr(lo), render_expr(hi)),
        Expr::Region { kind, .. } => format!("<region:{:?}>", kind),
        Expr::Ref { mutable, expr } => format!("&{}{}", if *mutable { "mut " } else { "" }, render_expr(expr)),
        Expr::StructLit { name, fields } => {
            let body: Vec<String> = fields.iter()
                .map(|(f, v)| format!("{}: {}", f, render_expr(v))).collect();
            format!("{} {{ {} }}", name, body.join(", "))
        }
        Expr::Match { scrutinee, arms } => {
            format!("match {} {{ {} arms }}", render_expr(scrutinee), arms.len())
        }
        Expr::Cast { expr, ty } => format!("({} as {})", render_expr(expr), ty),
        Expr::Index { recv, idx } => format!("{}[{}]", render_expr(recv), render_expr(idx)),
        Expr::Tuple(elems) => {
            let parts: Vec<String> = elems.iter().map(render_expr).collect();
            format!("({})", parts.join(", "))
        }
        Expr::Closure { params, body } => {
            let plist: Vec<String> = params.iter().map(|(n, _)| n.clone()).collect();
            format!("|{}| {}", plist.join(", "), render_expr(body))
        }
        Expr::Try(inner) => format!("{}?", render_expr(inner)),
        Expr::Deref(inner) => format!("*{}", render_expr(inner)),
    }
}

fn bin_op_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
        BinOp::Mod => "%", BinOp::Eq => "==", BinOp::Ne => "!=", BinOp::Lt => "<",
        BinOp::Gt => ">", BinOp::Le => "<=", BinOp::Ge => ">=", BinOp::And => "&&",
        BinOp::Or => "||", BinOp::Assign => "=",
        BinOp::BitAnd => "&", BinOp::BitOr => "|", BinOp::BitXor => "^",
        BinOp::Shl => "<<", BinOp::Shr => ">>",
    }
}

fn un_op_str(op: UnOp) -> &'static str {
    match op { UnOp::Neg => "-", UnOp::Not => "!" }
}

pub fn dump_mir(m: &MirProgram) -> String {
    let mut s = String::new();
    s.push_str("// AETHER MIR — comments above this line do not exist in any binary\n");
    for f in &m.funcs {
        s.push_str(&format!("\nfn {}", f.name));
        if f.is_autodiff { s.push_str(" [autodiff]"); }
        if let Some(d) = &f.distributed {
            s.push_str(&format!(" [distributed world_size={} backend={} algo={}]",
                d.world_size, d.backend, d.algorithm));
        }
        s.push_str(" {\n");
        for st in &f.stmts {
            match st {
                MirStmt::Source(line) => s.push_str(&format!("    {}\n", line)),
                MirStmt::TapeInit => s.push_str("    tape_init\n"),
                MirStmt::TapePush { value } => s.push_str(&format!("    tape_push {}\n", value)),
                MirStmt::AccumulateGrad { source } => s.push_str(&format!("    accumulate_grad {}\n", source)),
                MirStmt::TapeReverse => s.push_str("    tape_reverse\n"),
                MirStmt::AllReduce { tensor, world_size, backend } => s.push_str(
                    &format!("    all_reduce {} world_size={} backend={}\n", tensor, world_size, backend)
                ),
            }
        }
        s.push_str("}\n");
        if let Some(g) = &f.adgraph {
            s.push_str(&adgraph::dump(g));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn mir(src: &str) -> MirProgram {
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        let prog = Parser::new(toks).parse_program().unwrap();
        run_autodiff_pass(&prog)
    }

    #[test]
    fn autodiff_inserts_tape() {
        let m = mir(r#"
            #[autodiff]
            fn step() {
                let loss = forward();
                loss.backward();
            }
        "#);
        let f = &m.funcs[0];
        assert!(f.is_autodiff);
        assert!(matches!(f.stmts.first(), Some(MirStmt::TapeInit)));
        assert!(matches!(f.stmts.last(), Some(MirStmt::TapeReverse)));
        assert!(f.stmts.iter().any(|s| matches!(s, MirStmt::AccumulateGrad { .. })));
    }

    #[test]
    fn distributed_inserts_all_reduce() {
        let m = mir(r#"
            #[autodiff]
            #[distributed(world_size=8, backend="nccl")]
            fn step() { let l = forward(); l.backward(); }
        "#);
        let f = &m.funcs[0];
        assert_eq!(f.distributed.as_ref().unwrap().world_size, 8);
        assert!(matches!(f.stmts.last(), Some(MirStmt::AllReduce { world_size: 8, .. })));
    }
}
