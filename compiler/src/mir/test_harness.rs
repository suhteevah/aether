//! Roadmap P6.14 — `#[test]` attribute + minimal runner harness.
//!
//! When aetherc is invoked with `--test`, this pass:
//!
//! 1. Walks `prog.items` for every `Item::Fn` whose attrs include
//!    `#[test]` (no args expected; attr is otherwise generic).
//! 2. Drops any user-defined `fn main(...)` from the program — the
//!    harness owns the entry point in `--test` mode.
//! 3. Synthesises a fresh `fn main() -> i32` that calls each tagged
//!    test fn in source order, prints PASS/FAIL lines + a summary, and
//!    exits 0 if all tests returned 0, else 1.
//!
//! Convention (because P6.14's `assert_eq!`/`assert!` macros are blocked
//! on P6.11):
//!   - `#[test] fn foo() -> i32 { ... }` returns 0 for pass, nonzero for fail.
//!   - The harness sums non-zero results into a `fail` counter.
//!
//! Output (stdout):
//!   [test] running <test_name>
//!   [test] PASS <test_name>     OR    [test] FAIL <test_name>
//!   ...
//!   [test] N passed, M failed
//!
//! The synthesized main uses only `println(<str literal>)` — no
//! runtime crate dependency, so the harness works under both the
//! pe-bin and aether-bin paths.

use crate::ast::*;

/// Returns the names of every fn in `prog` carrying `#[test]`.
pub fn collect_tests(prog: &Program) -> Vec<String> {
    let mut out = Vec::new();
    for item in &prog.items {
        if let Item::Fn(f) = item {
            if f.attrs.iter().any(|a| a.name == "test") {
                out.push(f.name.clone());
            }
        }
    }
    out
}

/// Rewrite `prog` in place: drop any user `main`, append a synthesized
/// `fn main() -> i32` that runs each `#[test]` fn. Returns the number
/// of tests wired into the harness.
pub fn install_harness(prog: &mut Program) -> usize {
    let tests = collect_tests(prog);

    // Drop existing user main — the harness takes over the entry point.
    prog.items.retain(|it| match it {
        Item::Fn(f) => f.name != "main",
        _ => true,
    });

    let main_fn = build_main(&tests);
    prog.items.push(Item::Fn(main_fn));
    tests.len()
}

fn build_main(tests: &[String]) -> FnDecl {
    let mut stmts: Vec<Stmt> = Vec::new();

    // let mut fail: i32 = 0;
    stmts.push(Stmt::Let {
        name: "__test_fail".into(),
        mutable: true,
        ty: Some(Ty::Named("i32".into())),
        value: Some(Expr::IntLit(0)),
    });
    // let mut pass: i32 = 0;
    stmts.push(Stmt::Let {
        name: "__test_pass".into(),
        mutable: true,
        ty: Some(Ty::Named("i32".into())),
        value: Some(Expr::IntLit(0)),
    });

    for (idx, name) in tests.iter().enumerate() {
        // println("[test] running <name>");
        stmts.push(Stmt::Expr(call_println(format!("[test] running {}", name))));

        // let __r_<idx>: i32 = <name>();
        let result_name = format!("__test_r{}", idx);
        stmts.push(Stmt::Let {
            name: result_name.clone(),
            mutable: false,
            ty: Some(Ty::Named("i32".into())),
            value: Some(Expr::Call {
                callee: Box::new(Expr::Ident(name.clone())),
                args: vec![],
            }),
        });

        // if __r_<idx> != 0 { fail = fail + 1; println("[test] FAIL <name>"); }
        // else              { pass = pass + 1; println("[test] PASS <name>"); }
        let then_block = Block {
            stmts: vec![
                Stmt::Expr(Expr::Bin {
                    op: BinOp::Assign,
                    lhs: Box::new(Expr::Ident("__test_fail".into())),
                    rhs: Box::new(Expr::Bin {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::Ident("__test_fail".into())),
                        rhs: Box::new(Expr::IntLit(1)),
                    }),
                }),
                Stmt::Expr(call_println(format!("[test] FAIL {}", name))),
            ],
            tail: None,
        };
        let else_block = Block {
            stmts: vec![
                Stmt::Expr(Expr::Bin {
                    op: BinOp::Assign,
                    lhs: Box::new(Expr::Ident("__test_pass".into())),
                    rhs: Box::new(Expr::Bin {
                        op: BinOp::Add,
                        lhs: Box::new(Expr::Ident("__test_pass".into())),
                        rhs: Box::new(Expr::IntLit(1)),
                    }),
                }),
                Stmt::Expr(call_println(format!("[test] PASS {}", name))),
            ],
            tail: None,
        };
        stmts.push(Stmt::Expr(Expr::If {
            cond: Box::new(Expr::Bin {
                op: BinOp::Ne,
                lhs: Box::new(Expr::Ident(result_name)),
                rhs: Box::new(Expr::IntLit(0)),
            }),
            then: then_block,
            else_: Some(else_block),
        }));
    }

    // Final summary line — keep it human-readable. Aether's println today
    // only takes a string literal, so we emit a fixed line and let the
    // pass/fail counts carry the numeric truth via the exit code.
    stmts.push(Stmt::Expr(call_println(format!(
        "[test] {} test(s) total",
        tests.len()
    ))));

    // tail: if fail != 0 { 1 } else { 0 }
    let tail = Expr::If {
        cond: Box::new(Expr::Bin {
            op: BinOp::Ne,
            lhs: Box::new(Expr::Ident("__test_fail".into())),
            rhs: Box::new(Expr::IntLit(0)),
        }),
        then: Block {
            stmts: vec![Stmt::Expr(call_println("[test] SOME TESTS FAILED".into()))],
            tail: Some(Box::new(Expr::IntLit(1))),
        },
        else_: Some(Block {
            stmts: vec![Stmt::Expr(call_println("[test] ALL PASSED".into()))],
            tail: Some(Box::new(Expr::IntLit(0))),
        }),
    };

    FnDecl {
        attrs: vec![],
        is_pub: false,
        is_extern: false,
        name: "main".into(),
        const_params: vec![],
        params: vec![],
        ret: Some(Ty::Named("i32".into())),
        body: Some(Block { stmts, tail: Some(Box::new(tail)) }),
    }
}

fn call_println(msg: String) -> Expr {
    Expr::Call {
        callee: Box::new(Expr::Ident("println".into())),
        args: vec![Expr::StrLit(msg)],
    }
}
