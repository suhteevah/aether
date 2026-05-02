//! Phase 0 C codegen — used as a fallback so `aetherc input.aether -o out`
//! actually produces a runnable binary via `gcc`. This is throwaway: Phase 1
//! drops it once the real LLVM/inkwell path emits native objects directly.

use crate::ast::{BinOp, Block, Expr, FnDecl, Item, Program, Stmt, UnOp};

pub fn emit(p: &Program) -> String {
    let mut s = String::new();
    s.push_str("#include <stdio.h>\n");
    s.push_str("#include <stdint.h>\n\n");

    // No-op runtime stubs so #[autodiff] / #[distributed] code links.
    s.push_str("static void aether_autodiff_init(void* t) { (void)t; }\n");
    s.push_str("static void aether_autodiff_push(void* t, void* v) { (void)t; (void)v; }\n");
    s.push_str("static void aether_autodiff_reverse(void* t) { (void)t; }\n");
    s.push_str("static void aether_autodiff_accumulate(void* t, void* g) { (void)t; (void)g; }\n");
    s.push_str("static void aether_dist_all_reduce(void* p, int n, int b) { (void)p; (void)n; (void)b; }\n\n");

    for item in &p.items {
        if let Item::Fn(f) = item {
            s.push_str(&emit_fn(f));
            s.push('\n');
        }
    }
    s
}

fn emit_fn(f: &FnDecl) -> String {
    let ret = match f.ret.as_ref().and_then(|t| match t {
        crate::ast::Ty::Named(n) => Some(n.as_str()),
        _ => None,
    }) {
        Some("i32") | Some("i64") | Some("u32") | Some("u64") => "int",
        Some("f32") | Some("f64") => "double",
        Some("bool") => "int",
        _ => "int",
    };
    if f.body.is_none() {
        // extern fn — emit a forward declaration only.
        return format!("extern int {}(/* extern */);\n", f.name);
    }
    let body = f.body.as_ref().unwrap();
    let mut s = format!("{} {}(", ret, f.name);
    if f.params.is_empty() { s.push_str("void"); }
    s.push_str(") {\n");
    s.push_str(&emit_block(body, 1));
    if !ends_with_return(body) {
        s.push_str("    return 0;\n");
    }
    s.push_str("}\n");
    s
}

fn ends_with_return(b: &Block) -> bool {
    matches!(b.stmts.last(), Some(Stmt::Return(_)))
}

fn indent(n: usize) -> String { "    ".repeat(n) }

fn emit_block(b: &Block, lvl: usize) -> String {
    let mut s = String::new();
    for st in &b.stmts {
        s.push_str(&emit_stmt(st, lvl));
    }
    if let Some(e) = &b.tail {
        s.push_str(&format!("{}{};\n", indent(lvl), emit_expr(e)));
    }
    s
}

fn emit_stmt(s: &Stmt, lvl: usize) -> String {
    match s {
        Stmt::Let { name, value, .. } => {
            let init = emit_expr(value);
            let ty = guess_c_ty(value);
            format!("{}{} {} = {};\n", indent(lvl), ty, name, init)
        }
        Stmt::Expr(e) => format!("{}{};\n", indent(lvl), emit_expr(e)),
        Stmt::Return(Some(e)) => format!("{}return {};\n", indent(lvl), emit_expr(e)),
        Stmt::Return(None) => format!("{}return 0;\n", indent(lvl)),
    }
}

fn guess_c_ty(e: &Expr) -> &'static str {
    match e {
        Expr::FloatLit(_) => "double",
        Expr::StrLit(_) => "const char*",
        Expr::BoolLit(_) => "int",
        _ => "long",
    }
}

fn emit_expr(e: &Expr) -> String {
    match e {
        Expr::IntLit(n) => n.to_string(),
        Expr::FloatLit(f) => format!("{f}"),
        Expr::StrLit(s) => format!("{:?}", s),
        Expr::BoolLit(b) => if *b { "1".into() } else { "0".into() },
        Expr::Ident(s) => s.clone(),
        Expr::Path(p) => p.join("_"),
        Expr::Call { callee, args } => {
            // Map `println(...)` → `printf(...); puts("")`-ish: keep it simple.
            if let Expr::Ident(n) = callee.as_ref() {
                if n == "println" {
                    let inner = args.iter().map(emit_expr).collect::<Vec<_>>().join(", ");
                    return format!("puts({})", if args.is_empty() { "\"\"".into() } else { inner });
                }
                return format!("{}({})", n,
                    args.iter().map(emit_expr).collect::<Vec<_>>().join(", "));
            }
            format!("{}({})", emit_expr(callee),
                args.iter().map(emit_expr).collect::<Vec<_>>().join(", "))
        }
        Expr::MethodCall { recv, name, .. } => {
            // Phase 0: model methods like `loss.backward()` as runtime calls.
            if name == "backward" {
                return "aether_autodiff_reverse(0)".into();
            }
            format!("/* method {}.{} */ 0", emit_expr(recv), name)
        }
        Expr::Field { recv, name } => format!("{}.{}", emit_expr(recv), name),
        Expr::Bin { op, lhs, rhs } => {
            let o = match op {
                BinOp::Add => "+", BinOp::Sub => "-", BinOp::Mul => "*", BinOp::Div => "/",
                BinOp::Mod => "%", BinOp::Eq => "==", BinOp::Ne => "!=", BinOp::Lt => "<",
                BinOp::Gt => ">", BinOp::Le => "<=", BinOp::Ge => ">=", BinOp::And => "&&",
                BinOp::Or => "||", BinOp::Assign => "=",
            };
            format!("({} {} {})", emit_expr(lhs), o, emit_expr(rhs))
        }
        Expr::Unary { op, expr } => {
            let o = match op { UnOp::Neg => "-", UnOp::Not => "!" };
            format!("({}{})", o, emit_expr(expr))
        }
        Expr::Block(b) => format!("({{ {} }})", b.stmts.iter().map(|s| emit_stmt(s, 0)).collect::<String>()),
        Expr::If { cond, then, else_ } => {
            let mut s = format!("if ({}) {{\n{}", emit_expr(cond), emit_block(then, 1));
            s.push_str("    }");
            if let Some(e) = else_ {
                s.push_str(&format!(" else {{\n{}    }}", emit_block(e, 1)));
            }
            s
        }
        Expr::For { var, iter, body, .. } => {
            if let Expr::Range { lo, hi, .. } = iter.as_ref() {
                let mut s = format!("for (long {var} = {}; {var} < {}; {var}++) {{\n",
                    emit_expr(lo), emit_expr(hi), var = var);
                s.push_str(&emit_block(body, 1));
                s.push('}');
                return s;
            }
            format!("/* for {} in {} */ 0", var, emit_expr(iter))
        }
        Expr::While { cond, body } => {
            let mut s = format!("while ({}) {{\n", emit_expr(cond));
            s.push_str(&emit_block(body, 1));
            s.push('}');
            s
        }
        Expr::Break => "break".into(),
        Expr::Continue => "continue".into(),
        Expr::Range { lo, hi, .. } => format!("/* {}..{} */ 0", emit_expr(lo), emit_expr(hi)),
        Expr::Region { body, .. } => {
            let mut s = String::from("({ ");
            for st in &body.stmts { s.push_str(&emit_stmt(st, 0)); }
            s.push_str("0; })");
            s
        }
        Expr::Ref { expr, .. } => format!("&({})", emit_expr(expr)),
    }
}
