//! Recursive-descent parser for Aether — Phase 0.
//!
//! Handles `module`, `use`, `fn`, attributes (`#[name(k=v, ...)]`),
//! `let`, `return`, blocks, calls, method calls, field access, paths,
//! `if`/`else`, `for ... in ...`, basic expressions, and references.

use crate::ast::*;
use crate::lexer::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

type PResult<T> = Result<T, String>;

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self { Self { toks, pos: 0 } }

    fn peek(&self, off: usize) -> &Tok { &self.toks[self.pos + off].tok }
    fn at(&self, t: &Tok) -> bool { std::mem::discriminant(self.peek(0)) == std::mem::discriminant(t) }
    fn bump(&mut self) -> Tok { let t = self.toks[self.pos].tok.clone(); self.pos += 1; t }
    fn loc(&self) -> (u32, u32) { let t = &self.toks[self.pos]; (t.line, t.col) }

    fn expect(&mut self, want: Tok) -> PResult<()> {
        if std::mem::discriminant(self.peek(0)) == std::mem::discriminant(&want) {
            self.bump();
            Ok(())
        } else {
            let (l, c) = self.loc();
            Err(format!("{}:{}: expected {:?}, got {:?}", l, c, want, self.peek(0)))
        }
    }

    pub fn parse_program(mut self) -> PResult<Program> {
        let mut items = Vec::new();
        while !matches!(self.peek(0), Tok::Eof) {
            items.push(self.parse_item()?);
        }
        Ok(Program { items })
    }

    fn parse_item(&mut self) -> PResult<Item> {
        // Eat attributes — they belong to the next fn.
        let attrs = self.parse_attrs()?;

        // `extern` is a soft keyword (lexed as Ident). Consume it eagerly so
        // the rest of this function can pretend it doesn't exist.
        let is_extern = if matches!(self.peek(0), Tok::Ident(s) if s == "extern") {
            self.bump();
            true
        } else { false };

        match self.peek(0) {
            Tok::Module => {
                self.bump();
                let path = self.parse_path()?;
                self.expect(Tok::Semi)?;
                Ok(Item::ModuleDecl(path))
            }
            Tok::Use => {
                self.bump();
                let path = self.parse_path()?;
                self.expect(Tok::Semi)?;
                Ok(Item::Use(path))
            }
            Tok::Const => self.parse_const_item(false),
            Tok::Struct => self.parse_struct_item(false),
            Tok::Pub if matches!(self.peek(1), Tok::Const) => {
                self.bump(); self.parse_const_item(true)
            }
            Tok::Pub if matches!(self.peek(1), Tok::Struct) => {
                self.bump(); self.parse_struct_item(true)
            }
            Tok::Pub | Tok::Fn => {
                let is_pub = if matches!(self.peek(0), Tok::Pub) { self.bump(); true } else { false };
                self.expect(Tok::Fn)?;
                let name = self.expect_ident()?;
                self.expect(Tok::LParen)?;
                let mut params = Vec::new();
                while !matches!(self.peek(0), Tok::RParen) {
                    let pname = if matches!(self.peek(0), Tok::SelfLower) {
                        self.bump(); "self".to_string()
                    } else {
                        self.expect_ident()?
                    };
                    self.expect(Tok::Colon)?;
                    let ty = self.parse_ty()?;
                    params.push(Param { name: pname, ty });
                    if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                }
                self.expect(Tok::RParen)?;
                let ret = if matches!(self.peek(0), Tok::Arrow) {
                    self.bump();
                    Some(self.parse_ty()?)
                } else { None };
                let body = if matches!(self.peek(0), Tok::Semi) {
                    self.bump();
                    None
                } else {
                    Some(self.parse_block()?)
                };
                if is_extern && body.is_some() {
                    return Err(format!("extern fn {} must not have a body", name));
                }
                Ok(Item::Fn(FnDecl { attrs, is_pub, is_extern, name, params, ret, body }))
            }
            other => {
                let (l, c) = self.loc();
                Err(format!("{}:{}: expected item, got {:?}", l, c, other))
            }
        }
    }

    fn parse_const_item(&mut self, is_pub: bool) -> PResult<Item> {
        self.expect(Tok::Const)?;
        let name = self.expect_ident()?;
        self.expect(Tok::Colon)?;
        let ty = self.parse_ty()?;
        self.expect(Tok::Eq)?;
        let value = self.parse_expr()?;
        self.expect(Tok::Semi)?;
        Ok(Item::Const(ConstDecl { is_pub, name, ty, value }))
    }

    fn parse_struct_item(&mut self, is_pub: bool) -> PResult<Item> {
        self.expect(Tok::Struct)?;
        let name = self.expect_ident()?;
        let mut generics = Vec::new();
        if matches!(self.peek(0), Tok::Lt) {
            self.bump();
            while !matches!(self.peek(0), Tok::Gt) {
                generics.push(self.expect_ident()?);
                if matches!(self.peek(0), Tok::Comma) { self.bump(); }
            }
            self.expect(Tok::Gt)?;
        }
        self.expect(Tok::LBrace)?;
        let mut fields = Vec::new();
        while !matches!(self.peek(0), Tok::RBrace) {
            let fname = self.expect_ident()?;
            self.expect(Tok::Colon)?;
            let ty = self.parse_ty()?;
            fields.push(StructField { name: fname, ty });
            if matches!(self.peek(0), Tok::Comma) { self.bump(); }
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Struct(StructDecl { is_pub, name, generics, fields }))
    }

    fn parse_attrs(&mut self) -> PResult<Vec<Attr>> {
        let mut attrs = Vec::new();
        while matches!(self.peek(0), Tok::Hash) {
            self.bump();
            self.expect(Tok::LBracket)?;
            let name = self.expect_ident()?;
            let mut args = Vec::new();
            if matches!(self.peek(0), Tok::LParen) {
                self.bump();
                while !matches!(self.peek(0), Tok::RParen) {
                    let arg = self.parse_attr_arg()?;
                    args.push(arg);
                    if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                }
                self.expect(Tok::RParen)?;
            }
            self.expect(Tok::RBracket)?;
            attrs.push(Attr { name, args });
        }
        Ok(attrs)
    }

    fn parse_attr_arg(&mut self) -> PResult<AttrArg> {
        // [key =] value  where value is ident / int / str / bool
        let key = if matches!(self.peek(0), Tok::Ident(_)) && matches!(self.peek(1), Tok::Eq) {
            let k = self.expect_ident()?;
            self.expect(Tok::Eq)?;
            Some(k)
        } else { None };
        let value = match self.bump() {
            Tok::IntLit(n) => AttrVal::Int(n),
            Tok::StrLit(s) => AttrVal::Str(s),
            Tok::Ident(s) => AttrVal::Ident(s),
            Tok::True => AttrVal::Bool(true),
            Tok::False => AttrVal::Bool(false),
            other => return Err(format!("bad attr arg: {:?}", other)),
        };
        Ok(AttrArg { key, value })
    }

    fn parse_path(&mut self) -> PResult<Vec<String>> {
        let mut p = vec![self.expect_ident()?];
        while matches!(self.peek(0), Tok::ColonColon) {
            self.bump();
            p.push(self.expect_ident()?);
        }
        Ok(p)
    }

    fn expect_ident(&mut self) -> PResult<String> {
        match self.bump() {
            Tok::Ident(s) => Ok(s),
            other => {
                let (l, c) = self.loc();
                Err(format!("{}:{}: expected ident, got {:?}", l, c, other))
            }
        }
    }

    fn parse_ty(&mut self) -> PResult<Ty> {
        if matches!(self.peek(0), Tok::Amp) {
            self.bump();
            let mutable = if matches!(self.peek(0), Tok::Mut) { self.bump(); true } else { false };
            return Ok(Ty::Ref { mutable, inner: Box::new(self.parse_ty()?) });
        }
        if matches!(self.peek(0), Tok::LParen) {
            self.bump();
            self.expect(Tok::RParen)?;
            return Ok(Ty::Unit);
        }
        if matches!(self.peek(0), Tok::LBracket) {
            return self.parse_shape();
        }
        let name = self.expect_ident()?;
        if matches!(self.peek(0), Tok::Lt) {
            self.bump();
            let mut args = Vec::new();
            while !matches!(self.peek(0), Tok::Gt) {
                // Generic args can be types OR integer/symbolic shape values.
                // `Linear<D_MODEL, 960>` → both elements collapse to Ty::Named/Shape.
                if let Tok::IntLit(n) = self.peek(0).clone() {
                    self.bump();
                    args.push(Ty::Shape(vec![ShapeDim::Const(n)]));
                } else {
                    args.push(self.parse_ty()?);
                }
                if matches!(self.peek(0), Tok::Comma) { self.bump(); }
            }
            self.expect(Tok::Gt)?;
            return Ok(Ty::Generic { name, args });
        }
        Ok(Ty::Named(name))
    }

    fn parse_shape(&mut self) -> PResult<Ty> {
        self.expect(Tok::LBracket)?;
        let mut dims = Vec::new();
        while !matches!(self.peek(0), Tok::RBracket) {
            let dim = match self.bump() {
                Tok::IntLit(n) => ShapeDim::Const(n),
                Tok::Ident(s) => ShapeDim::Sym(s),
                other => return Err(format!("bad shape dim: {:?}", other)),
            };
            dims.push(dim);
            if matches!(self.peek(0), Tok::Comma) { self.bump(); }
        }
        self.expect(Tok::RBracket)?;
        Ok(Ty::Shape(dims))
    }

    fn parse_block(&mut self) -> PResult<Block> {
        self.expect(Tok::LBrace)?;
        let mut stmts = Vec::new();
        let mut tail: Option<Box<Expr>> = None;
        while !matches!(self.peek(0), Tok::RBrace) {
            if matches!(self.peek(0), Tok::Let) {
                self.bump();
                let mutable = if matches!(self.peek(0), Tok::Mut) { self.bump(); true } else { false };
                let name = self.expect_ident()?;
                let ty = if matches!(self.peek(0), Tok::Colon) {
                    self.bump();
                    Some(self.parse_ty()?)
                } else { None };
                self.expect(Tok::Eq)?;
                let value = self.parse_expr()?;
                self.expect(Tok::Semi)?;
                stmts.push(Stmt::Let { name, mutable, ty, value });
                continue;
            }
            if matches!(self.peek(0), Tok::Return) {
                self.bump();
                let val = if matches!(self.peek(0), Tok::Semi) { None } else { Some(self.parse_expr()?) };
                self.expect(Tok::Semi)?;
                stmts.push(Stmt::Return(val));
                continue;
            }
            // Expr statement or trailing expression
            let expr = self.parse_expr()?;
            if matches!(self.peek(0), Tok::Semi) {
                self.bump();
                stmts.push(Stmt::Expr(expr));
            } else if matches!(self.peek(0), Tok::RBrace) {
                tail = Some(Box::new(expr));
            } else {
                // implicit-semi after blocks (if / for)
                stmts.push(Stmt::Expr(expr));
            }
        }
        self.expect(Tok::RBrace)?;
        Ok(Block { stmts, tail })
    }

    fn parse_expr(&mut self) -> PResult<Expr> { self.parse_assign() }

    fn parse_assign(&mut self) -> PResult<Expr> {
        let lhs = self.parse_or()?;
        if matches!(self.peek(0), Tok::Eq) {
            self.bump();
            let rhs = self.parse_assign()?;
            return Ok(Expr::Bin { op: BinOp::Assign, lhs: Box::new(lhs), rhs: Box::new(rhs) });
        }
        Ok(lhs)
    }

    fn parse_or(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_and()?;
        while matches!(self.peek(0), Tok::PipePipe) {
            self.bump();
            let rhs = self.parse_and()?;
            lhs = Expr::Bin { op: BinOp::Or, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_and(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_cmp()?;
        while matches!(self.peek(0), Tok::AmpAmp) {
            self.bump();
            let rhs = self.parse_cmp()?;
            lhs = Expr::Bin { op: BinOp::And, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_cmp(&mut self) -> PResult<Expr> {
        let lhs = self.parse_add()?;
        let op = match self.peek(0) {
            Tok::EqEq => BinOp::Eq,
            Tok::BangEq => BinOp::Ne,
            Tok::Lt => BinOp::Lt,
            Tok::Gt => BinOp::Gt,
            Tok::LtEq => BinOp::Le,
            Tok::GtEq => BinOp::Ge,
            _ => return Ok(lhs),
        };
        self.bump();
        let rhs = self.parse_add()?;
        Ok(Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
    }

    fn parse_add(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_range()?;
        loop {
            let op = match self.peek(0) {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.parse_range()?;
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
    }

    fn parse_range(&mut self) -> PResult<Expr> {
        let lo = self.parse_mul()?;
        if matches!(self.peek(0), Tok::DotDot) {
            self.bump();
            let hi = self.parse_mul()?;
            // optional `.step_by(N)` is handled by the postfix method-call parser.
            return Ok(Expr::Range { lo: Box::new(lo), hi: Box::new(hi), step: None });
        }
        Ok(lo)
    }

    fn parse_mul(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let op = match self.peek(0) {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                Tok::Percent => BinOp::Mod,
                _ => return Ok(lhs),
            };
            self.bump();
            let rhs = self.parse_unary()?;
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        match self.peek(0) {
            Tok::Minus => { self.bump(); Ok(Expr::Unary { op: UnOp::Neg, expr: Box::new(self.parse_unary()?) }) }
            Tok::Bang => { self.bump(); Ok(Expr::Unary { op: UnOp::Not, expr: Box::new(self.parse_unary()?) }) }
            Tok::Amp => {
                self.bump();
                let mutable = if matches!(self.peek(0), Tok::Mut) { self.bump(); true } else { false };
                Ok(Expr::Ref { mutable, expr: Box::new(self.parse_unary()?) })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let mut e = self.parse_atom()?;
        loop {
            match self.peek(0) {
                Tok::LParen => {
                    self.bump();
                    let args = self.parse_call_args()?;
                    e = Expr::Call { callee: Box::new(e), args };
                }
                Tok::Dot => {
                    self.bump();
                    let name = self.expect_ident()?;
                    if matches!(self.peek(0), Tok::LParen) {
                        self.bump();
                        let args = self.parse_call_args()?;
                        e = Expr::MethodCall { recv: Box::new(e), name, args };
                    } else {
                        e = Expr::Field { recv: Box::new(e), name };
                    }
                }
                _ => return Ok(e),
            }
        }
    }

    fn parse_call_args(&mut self) -> PResult<Vec<Expr>> {
        let mut args = Vec::new();
        let mut first = true;
        while !matches!(self.peek(0), Tok::RParen) {
            if !first {
                let (l, c) = self.loc();
                return Err(format!("{}:{}: expected `,` or `)` between call arguments, got {:?}",
                    l, c, self.peek(0)));
            }
            args.push(self.parse_expr()?);
            if matches!(self.peek(0), Tok::Comma) {
                self.bump();
                // After a comma, another arg is required (or `)`).
            } else {
                first = false;
            }
        }
        self.expect(Tok::RParen)?;
        Ok(args)
    }

    fn parse_atom(&mut self) -> PResult<Expr> {
        match self.peek(0).clone() {
            Tok::IntLit(n) => { self.bump(); Ok(Expr::IntLit(n)) }
            Tok::FloatLit(f) => { self.bump(); Ok(Expr::FloatLit(f)) }
            Tok::StrLit(s) => { self.bump(); Ok(Expr::StrLit(s)) }
            Tok::True => { self.bump(); Ok(Expr::BoolLit(true)) }
            Tok::False => { self.bump(); Ok(Expr::BoolLit(false)) }
            Tok::LParen => {
                self.bump();
                let e = self.parse_expr()?;
                self.expect(Tok::RParen)?;
                Ok(e)
            }
            Tok::LBrace => Ok(Expr::Block(self.parse_block()?)),
            Tok::If => {
                self.bump();
                let cond = self.parse_expr()?;
                let then = self.parse_block()?;
                let else_ = if matches!(self.peek(0), Tok::Else) {
                    self.bump();
                    Some(self.parse_block()?)
                } else { None };
                Ok(Expr::If { cond: Box::new(cond), then, else_ })
            }
            Tok::For => {
                self.bump();
                let var = self.expect_ident()?;
                self.expect(Tok::In)?;
                let iter = self.parse_expr()?;
                let distributed = matches!(&iter, Expr::MethodCall { name, .. } if name == "distributed");
                let body = self.parse_block()?;
                Ok(Expr::For { var, iter: Box::new(iter), body, parallel: false, distributed })
            }
            Tok::While => {
                self.bump();
                let cond = self.parse_expr()?;
                let body = self.parse_block()?;
                Ok(Expr::While { cond: Box::new(cond), body })
            }
            Tok::Break => { self.bump(); Ok(Expr::Break) }
            Tok::Continue => { self.bump(); Ok(Expr::Continue) }
            Tok::Ident(s) => {
                // Soft keywords: parallel for / warp { } / block { } / ai_region { }
                match s.as_str() {
                    "parallel" if matches!(self.peek(1), Tok::For) => {
                        self.bump(); // parallel
                        self.bump(); // for
                        let var = self.expect_ident()?;
                        self.expect(Tok::In)?;
                        let iter = self.parse_expr()?;
                        let distributed = matches!(&iter, Expr::MethodCall { name, .. } if name == "distributed");
                        let body = self.parse_block()?;
                        return Ok(Expr::For {
                            var, iter: Box::new(iter), body, parallel: true, distributed,
                        });
                    }
                    "warp" if matches!(self.peek(1), Tok::LBrace) => {
                        self.bump();
                        let body = self.parse_block()?;
                        return Ok(Expr::Region { kind: RegionKind::Warp, body });
                    }
                    "block" if matches!(self.peek(1), Tok::LBrace) => {
                        self.bump();
                        let body = self.parse_block()?;
                        return Ok(Expr::Region { kind: RegionKind::Block, body });
                    }
                    "ai_region" if matches!(self.peek(1), Tok::LBrace) => {
                        self.bump();
                        let body = self.parse_block()?;
                        return Ok(Expr::Region { kind: RegionKind::AiRegion, body });
                    }
                    _ => {}
                }
                let mut path = vec![self.expect_ident()?];
                while matches!(self.peek(0), Tok::ColonColon) {
                    self.bump();
                    path.push(self.expect_ident()?);
                }
                if path.len() == 1 {
                    Ok(Expr::Ident(path.into_iter().next().unwrap()))
                } else {
                    Ok(Expr::Path(path))
                }
            }
            Tok::SelfLower => { self.bump(); Ok(Expr::Ident("self".into())) }
            other => {
                let (l, c) = self.loc();
                Err(format!("{}:{}: expected expression, got {:?}", l, c, other))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(src: &str) -> Program {
        let (toks, _) = Lexer::new(src).tokenize().unwrap();
        Parser::new(toks).parse_program().unwrap()
    }

    #[test]
    fn empty_main() {
        let p = parse("fn main() -> i32 { return 0; }");
        assert_eq!(p.items.len(), 1);
    }

    #[test]
    fn attrs_parsed() {
        let p = parse(r#"#[autodiff] #[distributed(world_size=8, backend="nccl")] fn step() {}"#);
        let Item::Fn(f) = &p.items[0] else { panic!() };
        assert_eq!(f.attrs.len(), 2);
        assert_eq!(f.attrs[1].arg_int("world_size"), Some(8));
        assert_eq!(f.attrs[1].arg_str("backend"), Some("nccl"));
    }
}
