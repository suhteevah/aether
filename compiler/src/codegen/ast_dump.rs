//! AST dump (`--emit=ast`) — a deterministic S-expression rendering of the
//! PARSED AST.
//!
//! P20.2: this is the *canonical* format that the self-hosted parser
//! (`tests/runtime/selfhost_parser_*.aether`) re-emits byte-for-byte. The
//! formal P20.2 witness compiles a shared `.aether` file with BOTH parsers
//! — Rust-aetherc via `--emit=ast`, and the Aether self-hosted parser — and
//! asserts the two dumps are byte-identical.
//!
//! The format is intentionally *structural*: function/parameter/identifier
//! NAMES, literal values, operators, and tree shape — but types are elided.
//! That keeps it reproducible by an independent parser that does not carry a
//! full type model yet. One top-level item per line, LF-terminated.
//!
//! Grammar of the emitted S-expressions (the subset both parsers share is the
//! load-bearing part; richer items are dumped too so `--emit=ast` is a general
//! tool, but only the shared subset is witnessed byte-for-byte):
//!
//!   item   := (fn NAME (params NAME*) BODY)
//!           | (use PATH) | (mod PATH) | (const NAME EXPR)
//!           | (struct NAME FIELD*) | (enum NAME VARIANT*)
//!   BODY   := (block STMT* TAIL?)            ; extern fns -> (extern)
//!   stmt   := (let NAME EXPR) | (return EXPR) | (assign NAME EXPR) | EXPR
//!   expr   := INT | NAME | (OP EXPR EXPR) | (neg EXPR) | (not EXPR)
//!           | (call NAME EXPR*) | (if EXPR BLOCK BLOCK?) | (while EXPR BLOCK)
//!           | ... (see dump_expr for the full surface)

use crate::ast::*;

/// Render the whole program. One top-level item per line, LF-terminated.
pub fn emit(p: &Program) -> String {
    let mut out = String::new();
    for item in &p.items {
        dump_item(item, &mut out);
    }
    out
}

fn dump_item(item: &Item, out: &mut String) {
    match item {
        Item::Fn(f) => {
            dump_fn(f, out);
            out.push('\n');
        }
        Item::Use(path) => {
            out.push_str("(use ");
            out.push_str(&path.join("::"));
            out.push_str(")\n");
        }
        Item::ModuleDecl(path) => {
            out.push_str("(mod ");
            out.push_str(&path.join("::"));
            out.push_str(")\n");
        }
        Item::Const(c) => {
            out.push_str("(const ");
            out.push_str(&c.name);
            out.push(' ');
            dump_expr(&c.value, out);
            out.push_str(")\n");
        }
        Item::Struct(s) => {
            out.push_str("(struct ");
            out.push_str(&s.name);
            for fld in &s.fields {
                out.push(' ');
                out.push_str(&fld.name);
            }
            out.push_str(")\n");
        }
        Item::Enum { name, variants, .. } => {
            out.push_str("(enum ");
            out.push_str(name);
            for v in variants {
                out.push(' ');
                out.push_str(v);
            }
            out.push_str(")\n");
        }
        // Methods flatten to `Type__method` fns, mirroring the asm backend's
        // mangling, so the dump reflects what actually reaches codegen.
        Item::Impl { type_name, methods } => {
            for m in methods {
                out.push_str("(fn ");
                out.push_str(&format!("{}__{}", type_name, m.name));
                out.push(' ');
                dump_fn_rest(m, out);
                out.push('\n');
            }
        }
        Item::ImplTrait { type_name, methods, .. } => {
            for m in methods {
                out.push_str("(fn ");
                out.push_str(&format!("{}__{}", type_name, m.name));
                out.push(' ');
                dump_fn_rest(m, out);
                out.push('\n');
            }
        }
        // Trait declarations carry signatures only (no bodies); dump the shape.
        Item::Trait { name, methods } => {
            out.push_str("(trait ");
            out.push_str(name);
            for m in methods {
                out.push(' ');
                out.push_str(&m.name);
            }
            out.push_str(")\n");
        }
    }
}

fn dump_fn(f: &FnDecl, out: &mut String) {
    out.push_str("(fn ");
    out.push_str(&f.name);
    out.push(' ');
    dump_fn_rest(f, out);
}

/// Everything after the name: `(params NAME*) BODY)`. The leading `(fn NAME `
/// is emitted by the caller so methods can substitute a mangled name.
fn dump_fn_rest(f: &FnDecl, out: &mut String) {
    out.push_str("(params");
    for p in &f.params {
        out.push(' ');
        out.push_str(&p.name);
    }
    out.push_str(") ");
    match &f.body {
        Some(b) => dump_block(b, out),
        None => out.push_str("(extern)"),
    }
    out.push(')');
}

fn dump_block(b: &Block, out: &mut String) {
    out.push_str("(block");
    for s in &b.stmts {
        out.push(' ');
        dump_stmt(s, out);
    }
    if let Some(t) = &b.tail {
        out.push(' ');
        dump_expr(t, out);
    }
    out.push(')');
}

fn dump_stmt(s: &Stmt, out: &mut String) {
    match s {
        Stmt::Let { name, value, .. } => {
            out.push_str("(let ");
            out.push_str(name);
            match value {
                Some(v) => {
                    out.push(' ');
                    dump_expr(v, out);
                }
                None => out.push_str(" ()"), // uninitialised stack decl
            }
            out.push(')');
        }
        Stmt::LetTuple { names, value } => {
            out.push_str("(let-tuple (");
            for (i, n) in names.iter().enumerate() {
                if i > 0 {
                    out.push(' ');
                }
                out.push_str(n);
            }
            out.push_str(") ");
            dump_expr(value, out);
            out.push(')');
        }
        Stmt::Return(Some(e)) => {
            out.push_str("(return ");
            dump_expr(e, out);
            out.push(')');
        }
        Stmt::Return(None) => out.push_str("(return)"),
        Stmt::Expr(e) => dump_expr(e, out),
    }
}

fn dump_expr(e: &Expr, out: &mut String) {
    match e {
        Expr::IntLit(n) => out.push_str(&n.to_string()),
        Expr::FloatLit(f) => out.push_str(&format!("{}", f)),
        Expr::BoolLit(b) => out.push_str(if *b { "true" } else { "false" }),
        Expr::StrLit(s) => {
            out.push('"');
            out.push_str(s);
            out.push('"');
        }
        Expr::Ident(n) => out.push_str(n),
        Expr::Path(p) => out.push_str(&p.join("::")),
        Expr::Bin { op, lhs, rhs } => {
            // `x = e` renders as `(assign NAME e)` to mirror the self-hosted
            // parser, which has a dedicated NODE_ASSIGN for IDENT `=` EXPR.
            if *op == BinOp::Assign {
                if let Expr::Ident(name) = lhs.as_ref() {
                    out.push_str("(assign ");
                    out.push_str(name);
                    out.push(' ');
                    dump_expr(rhs, out);
                    out.push(')');
                    return;
                }
            }
            out.push('(');
            out.push_str(binop_str(*op));
            out.push(' ');
            dump_expr(lhs, out);
            out.push(' ');
            dump_expr(rhs, out);
            out.push(')');
        }
        Expr::Unary { op, expr } => {
            out.push('(');
            out.push_str(match op {
                UnOp::Neg => "neg",
                UnOp::Not => "not",
            });
            out.push(' ');
            dump_expr(expr, out);
            out.push(')');
        }
        Expr::Call { callee, args } => {
            out.push_str("(call ");
            dump_expr(callee, out);
            for a in args {
                out.push(' ');
                dump_expr(a, out);
            }
            out.push(')');
        }
        Expr::MethodCall { recv, name, args } => {
            out.push_str("(mcall ");
            dump_expr(recv, out);
            out.push(' ');
            out.push_str(name);
            for a in args {
                out.push(' ');
                dump_expr(a, out);
            }
            out.push(')');
        }
        Expr::Field { recv, name } => {
            out.push_str("(field ");
            dump_expr(recv, out);
            out.push(' ');
            out.push_str(name);
            out.push(')');
        }
        Expr::If { cond, then, else_ } => {
            out.push_str("(if ");
            dump_expr(cond, out);
            out.push(' ');
            dump_block(then, out);
            if let Some(eb) = else_ {
                out.push(' ');
                dump_block(eb, out);
            }
            out.push(')');
        }
        Expr::While { cond, body } => {
            out.push_str("(while ");
            dump_expr(cond, out);
            out.push(' ');
            dump_block(body, out);
            out.push(')');
        }
        Expr::For { var, iter, body, .. } => {
            out.push_str("(for ");
            out.push_str(var);
            out.push(' ');
            dump_expr(iter, out);
            out.push(' ');
            dump_block(body, out);
            out.push(')');
        }
        Expr::Block(b) => dump_block(b, out),
        Expr::Range { lo, hi, .. } => {
            out.push_str("(range ");
            dump_expr(lo, out);
            out.push(' ');
            dump_expr(hi, out);
            out.push(')');
        }
        Expr::Break => out.push_str("(break)"),
        Expr::Continue => out.push_str("(continue)"),
        Expr::Ref { expr, .. } => {
            out.push_str("(ref ");
            dump_expr(expr, out);
            out.push(')');
        }
        Expr::Deref(e) => {
            out.push_str("(deref ");
            dump_expr(e, out);
            out.push(')');
        }
        Expr::Index { recv, idx } => {
            out.push_str("(index ");
            dump_expr(recv, out);
            out.push(' ');
            dump_expr(idx, out);
            out.push(')');
        }
        Expr::Cast { expr, ty } => {
            out.push_str("(cast ");
            dump_expr(expr, out);
            out.push(' ');
            out.push_str(ty);
            out.push(')');
        }
        Expr::StructLit { name, fields } => {
            out.push_str("(struct-lit ");
            out.push_str(name);
            for (fname, fexpr) in fields {
                out.push_str(" (");
                out.push_str(fname);
                out.push(' ');
                dump_expr(fexpr, out);
                out.push(')');
            }
            out.push(')');
        }
        Expr::Tuple(items) => {
            out.push_str("(tuple");
            for it in items {
                out.push(' ');
                dump_expr(it, out);
            }
            out.push(')');
        }
        Expr::Region { body, .. } => {
            out.push_str("(region ");
            dump_block(body, out);
            out.push(')');
        }
        Expr::Match { scrutinee, arms } => {
            out.push_str("(match ");
            dump_expr(scrutinee, out);
            for (_pat, arm) in arms {
                out.push_str(" (arm ");
                dump_expr(arm, out);
                out.push(')');
            }
            out.push(')');
        }
        Expr::Closure { body, .. } => {
            out.push_str("(closure ");
            dump_expr(body, out);
            out.push(')');
        }
        Expr::Try(e) => {
            out.push_str("(try ");
            dump_expr(e, out);
            out.push(')');
        }
    }
}

fn binop_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Mod => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::Le => "<=",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::Shl => "<<",
        BinOp::Shr => ">>",
        BinOp::Assign => "=",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> Program {
        let (toks, _stripped) = crate::lexer::Lexer::new(src).tokenize().expect("lex");
        crate::parser::Parser::new(toks).parse_program().expect("parse")
    }

    #[test]
    fn dumps_fn_with_while_and_call() {
        // Params carry types in real Aether; the canonical dump ELIDES them
        // (the self-hosted parser does too), so `(params n)` not `(params n: i64)`.
        let src = "fn sum_to(n: i64) -> i64 { let s = 0; let i = 1; while i <= n { s = s + i; i = i + 1; } return s; }\nfn main() -> i64 { sum_to(8) + 6 }\n";
        let prog = parse(src);
        let got = emit(&prog);
        let want = "(fn sum_to (params n) (block (let s 0) (let i 1) (while (<= i n) (block (assign s (+ s i)) (assign i (+ i 1)))) (return s)))\n(fn main (params) (block (+ (call sum_to 8) 6)))\n";
        assert_eq!(got, want);
    }
}
