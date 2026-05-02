//! Typed autodiff graph — Phase 1 prototype.
//!
//! The Phase 0 MIR pass only inserts opaque `TapePush` / `AccumulateGrad`
//! markers. This module builds a real DAG of operations from the forward
//! body of an `#[autodiff]` fn so the reverse sweep can emit *symbolic*
//! partials (e.g. `dz/dx = dz/dw * w` for `z = x * w`).
//!
//! Supported ops are deliberately small: `Add`, `Sub`, `Mul`, `MatMul`,
//! `ReLU`, `CrossEntropy`, `Forward(name)`. The graph is what later phases
//! widen — every new fusable op gets a node here, a primal lowering in
//! `codegen/llvm`, and a partial in `reverse()` below.

use crate::ast::{BinOp, Expr, FnDecl, Stmt};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum Op {
    Const(f64),
    Param(String),
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    MatMul(NodeId, NodeId),
    ReLU(NodeId),
    CrossEntropy { logits: NodeId, labels: NodeId },
    Forward { callee: String, inputs: Vec<NodeId> },
}

pub type NodeId = usize;

#[derive(Debug, Default, Clone)]
pub struct AdGraph {
    pub nodes: Vec<Op>,
    pub bindings: HashMap<String, NodeId>,
    pub loss: Option<NodeId>,
}

impl AdGraph {
    fn push(&mut self, op: Op) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(op);
        id
    }
}

pub fn build(f: &FnDecl) -> AdGraph {
    let mut g = AdGraph::default();
    let Some(body) = &f.body else { return g; };
    for s in &body.stmts {
        if let Stmt::Let { name, value, .. } = s {
            let id = lower(&mut g, value);
            g.bindings.insert(name.clone(), id);
            // First `let loss = ...` becomes the graph root.
            if g.loss.is_none() && name == "loss" { g.loss = Some(id); }
        }
    }
    // Tail expression like `loss.backward()` doesn't add a new op — it just
    // tags the root. If we never saw a `let loss`, treat the last bound name
    // as the loss.
    if g.loss.is_none() {
        if let Some(last) = body.stmts.iter().rev().find_map(|s| {
            if let Stmt::Let { name, .. } = s { Some(name.clone()) } else { None }
        }) {
            g.loss = g.bindings.get(&last).copied();
        }
    }
    g
}

fn lower(g: &mut AdGraph, e: &Expr) -> NodeId {
    match e {
        Expr::IntLit(n) => g.push(Op::Const(*n as f64)),
        Expr::FloatLit(f) => g.push(Op::Const(*f)),
        Expr::Ident(name) => {
            if let Some(&id) = g.bindings.get(name) { id }
            else { g.push(Op::Param(name.clone())) }
        }
        Expr::Bin { op, lhs, rhs } => {
            let l = lower(g, lhs);
            let r = lower(g, rhs);
            let n = match op {
                BinOp::Add => Op::Add(l, r),
                BinOp::Sub => Op::Sub(l, r),
                BinOp::Mul => Op::Mul(l, r),
                _ => Op::Forward { callee: format!("bin_{:?}", op), inputs: vec![l, r] },
            };
            g.push(n)
        }
        Expr::Call { callee, args } => {
            let inputs: Vec<NodeId> = args.iter().map(|a| lower(g, a)).collect();
            let name = match callee.as_ref() {
                Expr::Ident(s) => s.clone(),
                Expr::Path(p) => p.join("::"),
                _ => "<call>".into(),
            };
            match name.as_str() {
                "matmul" if inputs.len() == 2 => g.push(Op::MatMul(inputs[0], inputs[1])),
                "relu" if inputs.len() == 1 => g.push(Op::ReLU(inputs[0])),
                _ => g.push(Op::Forward { callee: name, inputs }),
            }
        }
        Expr::MethodCall { recv, name, args } => {
            let r = lower(g, recv);
            let arg_ids: Vec<NodeId> = args.iter().map(|a| lower(g, a)).collect();
            match name.as_str() {
                "matmul" if arg_ids.len() == 1 => g.push(Op::MatMul(r, arg_ids[0])),
                "relu" => g.push(Op::ReLU(r)),
                "cross_entropy" if arg_ids.len() == 1 => {
                    g.push(Op::CrossEntropy { logits: r, labels: arg_ids[0] })
                }
                "forward" => {
                    let inputs: Vec<NodeId> = std::iter::once(r).chain(arg_ids.into_iter()).collect();
                    g.push(Op::Forward { callee: "forward".into(), inputs })
                }
                _ => {
                    let inputs: Vec<NodeId> = std::iter::once(r).chain(arg_ids.into_iter()).collect();
                    g.push(Op::Forward { callee: name.clone(), inputs })
                }
            }
        }
        Expr::Ref { expr, .. } => lower(g, expr),
        // Anything else collapses to a parameter — Phase 1 widens this.
        _ => g.push(Op::Param("<expr>".into())),
    }
}

/// Reverse-mode partials. For each node we emit one line of pseudo-IR
/// describing how its grad propagates to its inputs. This is what the LLVM
/// backend lowers to `@aether_autodiff_accumulate` calls in Phase 1+.
pub fn reverse(g: &AdGraph) -> Vec<String> {
    let mut out = Vec::new();
    let Some(root) = g.loss else { return out; };
    out.push(format!("seed grad[{}] = 1", root));
    // Iterate in reverse construction order so children are visited before parents.
    for (id, op) in g.nodes.iter().enumerate().rev() {
        match op {
            Op::Const(_) => {}
            Op::Param(name) => out.push(format!("∂L/∂{} = grad[{}]", name, id)),
            Op::Add(a, b) => {
                out.push(format!("grad[{}] += grad[{}]", a, id));
                out.push(format!("grad[{}] += grad[{}]", b, id));
            }
            Op::Sub(a, b) => {
                out.push(format!("grad[{}] += grad[{}]", a, id));
                out.push(format!("grad[{}] -= grad[{}]", b, id));
            }
            Op::Mul(a, b) => {
                out.push(format!("grad[{}] += grad[{}] * v[{}]", a, id, b));
                out.push(format!("grad[{}] += grad[{}] * v[{}]", b, id, a));
            }
            Op::MatMul(a, b) => {
                out.push(format!("grad[{}] += grad[{}] @ v[{}].T", a, id, b));
                out.push(format!("grad[{}] += v[{}].T @ grad[{}]", b, a, id));
            }
            Op::ReLU(x) => out.push(format!("grad[{}] += grad[{}] * (v[{}] > 0)", x, id, x)),
            Op::CrossEntropy { logits, labels } => {
                out.push(format!("grad[{}] += softmax(v[{}]) - onehot(v[{}])", logits, logits, labels));
            }
            Op::Forward { inputs, callee } => {
                for inp in inputs {
                    out.push(format!("grad[{}] += vjp[{}](grad[{}])", inp, callee, id));
                }
            }
        }
    }
    out
}

pub fn dump(g: &AdGraph) -> String {
    let mut s = String::new();
    s.push_str("# AdGraph forward\n");
    for (i, op) in g.nodes.iter().enumerate() { s.push_str(&format!("  v{} = {:?}\n", i, op)); }
    if let Some(r) = g.loss { s.push_str(&format!("# loss = v{}\n", r)); }
    s.push_str("# AdGraph reverse\n");
    for line in reverse(g) { s.push_str(&format!("  {}\n", line)); }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::ast::Item;

    fn first_fn(src: &str) -> FnDecl {
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        let p = Parser::new(toks).parse_program().unwrap();
        match p.items.into_iter().next().unwrap() {
            Item::Fn(f) => f,
            _ => panic!(),
        }
    }

    #[test]
    fn mul_partials() {
        let f = first_fn("#[autodiff] fn s() { let z = x * w; let loss = z; }");
        let g = build(&f);
        let r = reverse(&g);
        assert!(r.iter().any(|l| l.contains("grad[") && l.contains("* v[")),
            "expected mul vjp pattern, got {:?}", r);
    }

    #[test]
    fn cross_entropy_uses_softmax() {
        let f = first_fn(r#"
            #[autodiff] fn step() {
                let logits = forward();
                let loss = logits.cross_entropy(labels);
            }
        "#);
        let g = build(&f);
        let r = reverse(&g);
        assert!(r.iter().any(|l| l.contains("softmax")), "got {:?}", r);
    }
}
