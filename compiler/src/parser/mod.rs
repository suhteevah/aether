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
    /// When `false`, `Ident { ... }` does not parse as a struct literal — it
    /// stays an Ident expression and the `{` belongs to the surrounding
    /// construct (the body of an `if`/`while`/`for` cond, typically).
    /// Mirrors Rust's `no_struct_literal` flag. Restored by callers via
    /// `with_struct_lit_disabled`.
    struct_lit_allowed: bool,
}

type PResult<T> = Result<T, String>;

impl Parser {
    pub fn new(toks: Vec<Token>) -> Self {
        Self { toks, pos: 0, struct_lit_allowed: true }
    }

    /// Run `f` with struct literal parsing disabled (for `if`/`while`/`for`
    /// cond positions). Restores the previous flag value on exit.
    fn with_struct_lit_disabled<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = self.struct_lit_allowed;
        self.struct_lit_allowed = false;
        let r = f(self);
        self.struct_lit_allowed = saved;
        r
    }

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

    /// P16.11 — consume an optional `(crate)` / `(super)` / `(self)` /
    /// `(in path::to)` visibility specifier sitting after a `pub` token.
    /// Returns silently with the cursor advanced past the closing `)` when
    /// one is present; no-op when the next token isn't `(`. Multi-crate
    /// enforcement is FR-16.11-extra; today's single-crate compiler treats
    /// every visibility as crate-public.
    fn eat_visibility_paren(&mut self) {
        if !matches!(self.peek(0), Tok::LParen) { return; }
        self.bump(); // (
        // Burn tokens until balanced ).
        let mut depth = 1u32;
        while depth > 0 {
            match self.bump() {
                Tok::LParen => depth += 1,
                Tok::RParen => depth -= 1,
                Tok::Eof => return,
                _ => {}
            }
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

        // P12.3 — `async fn …` parses; today the body runs synchronously
        // (state-machine lowering is a deeper rewrite). The marker is
        // consumed so the rest of the parser sees a normal fn decl.
        if matches!(self.peek(0), Tok::Async) { self.bump(); }

        // P16.16 — `unsafe impl Trait for Type {}` is the canonical way to
        // declare a marker trait (Send, Sync) for a user type. We accept
        // the `unsafe` prefix before `impl` and delegate to the regular
        // parse_impl_item; the impl's body is typically empty (auto trait).
        if matches!(self.peek(0), Tok::Unsafe) && matches!(self.peek(1), Tok::Impl) {
            self.bump();
        }

        // P12.4 — `macro_rules! name { … }` item. We parse the shape (name
        // + balanced braces) and silently drop it. Call-site expansion is
        // handled in `parse_postfix` by treating `name!(...)` as `name(...)`.
        if matches!(self.peek(0), Tok::Ident(s) if s == "macro_rules")
            && matches!(self.peek(1), Tok::Bang)
        {
            self.bump(); // macro_rules
            self.bump(); // !
            let _name = self.expect_ident()?;
            self.expect(Tok::LBrace)?;
            let mut depth = 1u32;
            while depth > 0 {
                match self.peek(0).clone() {
                    Tok::LBrace => { depth += 1; self.bump(); }
                    Tok::RBrace => { depth -= 1; self.bump(); }
                    Tok::Eof => return Err("unterminated macro_rules! body".into()),
                    _ => { self.bump(); }
                }
            }
            // Synthesize a no-op item the rest of the pipeline ignores.
            return Ok(Item::Use(vec!["__macro_rules_skipped".to_string()]));
        }

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
            Tok::Impl => self.parse_impl_item(),
            Tok::Trait => self.parse_trait_item(),
            Tok::Enum => self.parse_enum_item(),
            Tok::Pub => {
                self.bump();
                // P16.11 — accept `pub(crate)` / `pub(super)` / `pub(self)` /
                // `pub(in path::to)` visibility specifiers. The single-crate
                // compiler treats them all as public-within-this-crate, which
                // matches single-crate semantics 1:1. Multi-crate enforcement
                // is FR-16.11-extra (depends real module system).
                self.eat_visibility_paren();
                match self.peek(0) {
                    Tok::Const => self.parse_const_item(true),
                    Tok::Struct => self.parse_struct_item(true),
                    Tok::Fn => {
                        let f = self.parse_fn_decl(attrs, true, is_extern, None)?;
                        Ok(Item::Fn(f))
                    }
                    other => {
                        let (l, c) = self.loc();
                        Err(format!("{}:{}: expected item after `pub`, got {:?}", l, c, other))
                    }
                }
            }
            Tok::Fn => {
                let f = self.parse_fn_decl(attrs, false, is_extern, None)?;
                Ok(Item::Fn(f))
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

    /// Parse a fn decl. `impl_type`, when `Some`, supplies the receiver type
    /// for `&self` / `&mut self` / `self` forms in impl blocks.
    fn parse_fn_decl(&mut self, attrs: Vec<Attr>, is_pub: bool, is_extern: bool,
                     impl_type: Option<&str>) -> PResult<FnDecl> {
        self.expect(Tok::Fn)?;
        let name = self.expect_ident()?;
        // Optional const-generic param list: `fn forward<M, K>(...)`.
        // Each name binds an i32 dim that resolves at each call site.
        let mut const_params: Vec<String> = Vec::new();
        if matches!(self.peek(0), Tok::Lt) {
            self.bump();
            while !matches!(self.peek(0), Tok::Gt) {
                // P12.2 — accept lifetime params (`<'a, T>`); we don't record
                // them in the AST yet, just consume to keep parsing intact.
                if matches!(self.peek(0), Tok::Lifetime(_)) { self.bump(); }
                else {
                    let p = self.expect_ident()?;
                    const_params.push(p);
                }
                if matches!(self.peek(0), Tok::Comma) { self.bump(); }
            }
            self.expect(Tok::Gt)?;
        }
        self.expect(Tok::LParen)?;
        let mut params = Vec::new();
        let mut first_param = true;
        while !matches!(self.peek(0), Tok::RParen) {
            // Receiver-shorthand forms inside impl blocks. Only the first
            // param can be `[&[mut]] self`; subsequent params must be the
            // standard `name: type`.
            if first_param && impl_type.is_some() {
                let recv_ty: Option<Ty> = if matches!(self.peek(0), Tok::SelfLower) {
                    self.bump();
                    Some(Ty::Named(impl_type.unwrap().to_string()))
                } else if matches!(self.peek(0), Tok::Amp) {
                    self.bump();
                    let mutable = if matches!(self.peek(0), Tok::Mut) { self.bump(); true } else { false };
                    if !matches!(self.peek(0), Tok::SelfLower) {
                        return Err(format!("after `&[mut]` expected `self` in impl method"));
                    }
                    self.bump();
                    Some(Ty::Ref { mutable, inner: Box::new(Ty::Named(impl_type.unwrap().to_string())) })
                } else { None };
                if let Some(ty) = recv_ty {
                    params.push(Param { name: "self".into(), ty });
                    if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                    first_param = false;
                    continue;
                }
            }
            first_param = false;
            // Allow `mut` on params (P13.1) — informational only; codegen treats every param slot writable.
            if matches!(self.peek(0), Tok::Mut) { self.bump(); }
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
        Ok(FnDecl { attrs, is_pub, is_extern, name, const_params, params, ret, body })
    }

    fn parse_impl_item(&mut self) -> PResult<Item> {
        self.expect(Tok::Impl)?;
        let first = self.expect_ident()?;
        // Optional generic args on the trait/type ref: `impl From<i64> for T`.
        // The source type in `From<S>` isn't needed for dispatch (the method
        // flattens to `T__from`), so the args are parsed + discarded. v1
        // supports a single non-nested arg list (`From<i64>`, not `From<Vec<i64>>`).
        if matches!(self.peek(0), Tok::Lt) {
            self.bump();
            loop {
                let _ = self.parse_ty()?;
                if matches!(self.peek(0), Tok::Comma) { self.bump(); continue; }
                break;
            }
            self.expect(Tok::Gt)?;
        }
        // `impl Foo for Bar { ... }` → ImplTrait. `impl Bar { ... }` → Impl.
        let (trait_name, type_name) = if matches!(self.peek(0), Tok::For) {
            self.bump();
            let bar = self.expect_ident()?;
            (Some(first), bar)
        } else { (None, first) };
        self.expect(Tok::LBrace)?;
        let mut methods = Vec::new();
        while !matches!(self.peek(0), Tok::RBrace) {
            let attrs = self.parse_attrs()?;
            let is_pub = if matches!(self.peek(0), Tok::Pub) { self.bump(); true } else { false };
            let m = self.parse_fn_decl(attrs, is_pub, false, Some(&type_name))?;
            methods.push(m);
        }
        self.expect(Tok::RBrace)?;
        if let Some(tr) = trait_name {
            Ok(Item::ImplTrait { trait_name: tr, type_name, methods })
        } else {
            Ok(Item::Impl { type_name, methods })
        }
    }

    /// `trait Foo { fn bar(&self) -> i32; fn baz() -> i64 { 0 } }`.
    /// Method bodies are optional (signature-only is fine).
    fn parse_trait_item(&mut self) -> PResult<Item> {
        self.expect(Tok::Trait)?;
        let name = self.expect_ident()?;
        // Optional supertrait bounds: `trait Pet: Animal + Named { ... }`.
        let mut supertraits = Vec::new();
        if matches!(self.peek(0), Tok::Colon) {
            self.bump();
            supertraits.push(self.expect_ident()?);
            while matches!(self.peek(0), Tok::Plus) {
                self.bump();
                supertraits.push(self.expect_ident()?);
            }
        }
        self.expect(Tok::LBrace)?;
        let mut methods = Vec::new();
        while !matches!(self.peek(0), Tok::RBrace) {
            let attrs = self.parse_attrs()?;
            let is_pub = if matches!(self.peek(0), Tok::Pub) { self.bump(); true } else { false };
            let m = self.parse_fn_decl(attrs, is_pub, false, Some(&name))?;
            methods.push(m);
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Trait { name, supertraits, methods })
    }

    fn parse_match_pat(&mut self) -> PResult<MatchPat> {
        match self.peek(0).clone() {
            Tok::IntLit(n) => { self.bump(); Ok(MatchPat::Int(n)) }
            Tok::Ident(s) if s == "_" => { self.bump(); Ok(MatchPat::Wildcard) }
            Tok::Ident(_) => {
                let mut path = vec![self.expect_ident()?];
                while matches!(self.peek(0), Tok::ColonColon) {
                    self.bump();
                    path.push(self.expect_ident()?);
                }
                // `Box::Full(x)` → bind payload into local `x`.
                if matches!(self.peek(0), Tok::LParen) {
                    self.bump();
                    let bind = self.expect_ident()?;
                    self.expect(Tok::RParen)?;
                    Ok(MatchPat::EnumVariantBind(path, bind))
                } else {
                    Ok(MatchPat::EnumVariant(path))
                }
            }
            other => {
                let (l, c) = self.loc();
                Err(format!("{}:{}: expected match pattern, got {:?}", l, c, other))
            }
        }
    }

    fn parse_enum_item(&mut self) -> PResult<Item> {
        self.expect(Tok::Enum)?;
        let name = self.expect_ident()?;
        self.expect(Tok::LBrace)?;
        let mut variants = Vec::new();
        let mut payloads = Vec::new();
        while !matches!(self.peek(0), Tok::RBrace) {
            let v = self.expect_ident()?;
            // Optional `( ty )` payload. Single-element only for now.
            let payload = if matches!(self.peek(0), Tok::LParen) {
                self.bump();
                let ty = self.parse_ty()?;
                self.expect(Tok::RParen)?;
                Some(ty)
            } else { None };
            variants.push(v);
            payloads.push(payload);
            if matches!(self.peek(0), Tok::Comma) { self.bump(); }
        }
        self.expect(Tok::RBrace)?;
        Ok(Item::Enum { name, variants, payloads })
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
        // P16.25 — `impl Trait` in argument or return position. Treated as
        // a placeholder that lowers to its underlying boxed/concrete type
        // once trait dispatch is real (FR-16.25-extra). Today we accept the
        // syntax and represent it as `Ty::Named("__impl_<trait>")` so the
        // signature parses and methods can be called on it via direct fn
        // dispatch wherever the user's actual concrete type matches.
        if matches!(self.peek(0), Tok::Impl) {
            self.bump();
            let trait_name = self.expect_ident()?;
            // Accept optional `+ Send + Sync` etc — discard.
            while matches!(self.peek(0), Tok::Plus) {
                self.bump();
                let _bound = self.expect_ident()?;
            }
            return Ok(Ty::Named(format!("__impl_{}", trait_name)));
        }
        if matches!(self.peek(0), Tok::Amp) {
            self.bump();
            // P12.2 — accept (and silently elide) explicit lifetime annotations:
            // `&'a T` / `&'a mut T`. Today's borrow checker (mir::lifetimes)
            // works on inferred regions; the lifetime name is recorded only in
            // diagnostics, not in the Ty AST.
            if matches!(self.peek(0), Tok::Lifetime(_)) { self.bump(); }
            let mutable = if matches!(self.peek(0), Tok::Mut) { self.bump(); true } else { false };
            // P16.19 — native slice `&[T]` / `&mut [T]`. The `[T]` here is an
            // unsized slice element list, NOT a tensor `Shape` (which never
            // appears behind a `&`). Parse `[ T ]` directly so we don't fall
            // into the `Tok::LBracket` tensor-shape branch in the recursive
            // `parse_ty`. `&str` is sugar for `&[u8]`.
            if matches!(self.peek(0), Tok::LBracket) {
                self.bump(); // [
                let elem = self.parse_ty()?;
                self.expect(Tok::RBracket)?;
                return Ok(Ty::Slice { mutable, elem: Box::new(elem) });
            }
            // `&str` → `&[u8]` slice (bytes). Distinguishing string-ness is not
            // needed for the i64 witness; we model it as a u8 slice.
            if matches!(self.peek(0), Tok::Ident(ref n) if n == "str") {
                self.bump();
                return Ok(Ty::Slice { mutable, elem: Box::new(Ty::Named("u8".into())) });
            }
            return Ok(Ty::Ref { mutable, inner: Box::new(self.parse_ty()?) });
        }
        if matches!(self.peek(0), Tok::LParen) {
            self.bump();
            // `()` = unit; `(T1, T2, ...)` = tuple type.
            if matches!(self.peek(0), Tok::RParen) {
                self.bump();
                return Ok(Ty::Unit);
            }
            let first = self.parse_ty()?;
            if matches!(self.peek(0), Tok::RParen) {
                // Single-elem parens — treat as the inner type itself, NOT a
                // 1-tuple (Rust requires `(T,)` for that — we don't bother).
                self.bump();
                return Ok(first);
            }
            // Comma — tuple.
            self.expect(Tok::Comma)?;
            let mut elems = vec![first];
            while !matches!(self.peek(0), Tok::RParen) {
                elems.push(self.parse_ty()?);
                if matches!(self.peek(0), Tok::Comma) { self.bump(); }
            }
            self.expect(Tok::RParen)?;
            return Ok(Ty::Tuple(elems));
        }
        if matches!(self.peek(0), Tok::LBracket) {
            // Disambiguate `[T; N]` (stack array) from `[d1, d2, ...]` (Tensor
            // shape). A shape's first element is always an int literal or a
            // bare ident followed by `,` or `]`. An array starts with a Type
            // followed by `;`. We look one past the first content token: if
            // it's `;`, parse as array; otherwise fall through to parse_shape.
            // (Handles single-token Type prefixes; complex generic Types in
            // arrays would need lookahead — punt until needed.)
            let is_array = matches!(self.peek(1), Tok::Semi)
                || matches!(self.peek(2), Tok::Semi);
            if is_array {
                self.bump(); // [
                let elem = self.parse_ty()?;
                self.expect(Tok::Semi)?;
                let n_raw = self.bump();
                let n = match n_raw {
                    Tok::IntLit(v) if v >= 0 => v as usize,
                    other => return Err(format!("array length must be a non-negative int literal, got {:?}", other)),
                };
                self.expect(Tok::RBracket)?;
                return Ok(Ty::Array { elem: Box::new(elem), n });
            }
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
                // Tuple destructuring: `let (a, b, ...) = (x, y, ...);`. Each
                // name becomes its own top-level local; the rhs MUST be a
                // tuple literal of matching arity (more general fn-tuple-
                // returns awaits sret ABI).
                if matches!(self.peek(0), Tok::LParen) {
                    self.bump();
                    let mut names = Vec::new();
                    while !matches!(self.peek(0), Tok::RParen) {
                        names.push(self.expect_ident()?);
                        if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                    }
                    self.expect(Tok::RParen)?;
                    self.expect(Tok::Eq)?;
                    let value = self.parse_expr()?;
                    self.expect(Tok::Semi)?;
                    let _ = mutable;
                    stmts.push(Stmt::LetTuple { names, value });
                    continue;
                }
                let name = self.expect_ident()?;
                let ty = if matches!(self.peek(0), Tok::Colon) {
                    self.bump();
                    Some(self.parse_ty()?)
                } else { None };
                let value = if matches!(self.peek(0), Tok::Semi) {
                    // `let x: Ty;` — uninit declaration (struct locals).
                    None
                } else {
                    self.expect(Tok::Eq)?;
                    Some(self.parse_expr()?)
                };
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
        // Compound assignments desugar to `lhs = lhs <op> rhs`. The lhs has
        // to be evaluated TWICE textually but that's fine because lvalues
        // here are bare idents / field paths / array indices — pure address
        // computations with no side effects. (When method-call lvalues
        // appear someday, this will need a temporary.)
        let compound_op = match self.peek(0) {
            Tok::Eq      => return self.finish_assign(lhs),
            Tok::PlusEq  => Some(BinOp::Add),
            Tok::MinusEq => Some(BinOp::Sub),
            Tok::StarEq  => Some(BinOp::Mul),
            Tok::SlashEq => Some(BinOp::Div),
            _ => None,
        };
        if let Some(op) = compound_op {
            self.bump();
            let rhs = self.parse_assign()?;
            let new_rhs = Expr::Bin { op, lhs: Box::new(lhs.clone()), rhs: Box::new(rhs) };
            return Ok(Expr::Bin {
                op: BinOp::Assign,
                lhs: Box::new(lhs),
                rhs: Box::new(new_rhs),
            });
        }
        Ok(lhs)
    }

    fn finish_assign(&mut self, lhs: Expr) -> PResult<Expr> {
        self.bump(); // =
        let rhs = self.parse_assign()?;
        Ok(Expr::Bin { op: BinOp::Assign, lhs: Box::new(lhs), rhs: Box::new(rhs) })
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
        let lhs = self.parse_bitor()?;
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
        let rhs = self.parse_bitor()?;
        Ok(Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
    }

    fn parse_bitor(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_bitxor()?;
        while matches!(self.peek(0), Tok::Pipe) {
            self.bump();
            let rhs = self.parse_bitxor()?;
            lhs = Expr::Bin { op: BinOp::BitOr, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_bitxor(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_bitand()?;
        while matches!(self.peek(0), Tok::Caret) {
            self.bump();
            let rhs = self.parse_bitand()?;
            lhs = Expr::Bin { op: BinOp::BitXor, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_bitand(&mut self) -> PResult<Expr> {
        let mut lhs = self.parse_shift()?;
        // Single `&` is overloaded with the address-of-prefix-form. Postfix
        // here can never be confused: `&` between two complete expressions
        // is bitwise AND. Address-of only appears at the START of an expr
        // and is parsed in parse_unary.
        while matches!(self.peek(0), Tok::Amp) {
            self.bump();
            let rhs = self.parse_shift()?;
            lhs = Expr::Bin { op: BinOp::BitAnd, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> PResult<Expr> {
        // Bitwise shifts. `<<` and `>>` lex as TWO tokens (Lt+Lt / Gt+Gt) so
        // they don't conflict with generic-args `Vec<Vec<T>>`. Here we peek
        // two-deep to recognise the shift pair.
        let mut lhs = self.parse_add()?;
        loop {
            let op = match (self.peek(0), self.peek(1)) {
                (Tok::Lt, Tok::Lt) => BinOp::Shl,
                (Tok::Gt, Tok::Gt) => BinOp::Shr,
                _ => break,
            };
            self.bump(); self.bump();
            let rhs = self.parse_add()?;
            lhs = Expr::Bin { op, lhs: Box::new(lhs), rhs: Box::new(rhs) };
        }
        Ok(lhs)
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
            // `*expr` deref (P12.5).
            Tok::Star => { self.bump(); Ok(Expr::Deref(Box::new(self.parse_unary()?))) }
            Tok::Amp => {
                self.bump();
                let mutable = if matches!(self.peek(0), Tok::Mut) { self.bump(); true } else { false };
                Ok(Expr::Ref { mutable, expr: Box::new(self.parse_unary()?) })
            }
            // `|| expr` — empty-param closure. The lexer fuses `||` to a
            // single PipePipe token (logical-or); we re-split it here when
            // we know we're at a unary slot (PipePipe at the start of an
            // expression can't be the binary operator).
            Tok::PipePipe => {
                self.bump();
                let body = self.parse_expr()?;
                Ok(Expr::Closure { params: Vec::new(), body: Box::new(body) })
            }
            // `|x| expr` or `|x: T, y: T| expr` — closure literal. Position
            // disambiguates against bitwise `|` (which only appears between
            // two complete expressions, never at the start of a unary slot).
            Tok::Pipe => {
                self.bump();
                let mut params = Vec::new();
                while !matches!(self.peek(0), Tok::Pipe) {
                    let name = self.expect_ident()?;
                    let ty = if matches!(self.peek(0), Tok::Colon) {
                        self.bump();
                        Some(self.parse_ty()?)
                    } else { None };
                    params.push((name, ty));
                    if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                }
                self.expect(Tok::Pipe)?;
                let body = self.parse_expr()?;
                Ok(Expr::Closure { params, body: Box::new(body) })
            }
            _ => self.parse_postfix(),
        }
    }

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let mut e = self.parse_atom()?;
        loop {
            // P12.4 — `name!(...)` macro invocation. Today's expansion is a
            // pass-through to a fn call: `name!(args)` desugars to `name(args)`.
            // FR-16.14 special-cases `println!` / `print!` / `format!` to
            // expand the format string into a Block of print-primitive calls
            // — one `aether_print_str_n(seg_ptr, seg_len)` per literal
            // segment, one `aether_print_<type>(arg)` per `{}` / `{:f}` hole,
            // and a trailing `aether_print_newline()` for `println!`.
            if matches!(self.peek(0), Tok::Bang) && matches!(self.peek(1), Tok::LParen) {
                if let Expr::Ident(name) = &e {
                    let nm = name.clone();
                    if nm == "println" || nm == "print" {
                        self.bump(); // !
                        self.bump(); // (
                        let args = self.parse_call_args()?;
                        e = expand_print_macro(&nm, args)?;
                        continue;
                    }
                    self.bump(); // !
                    self.bump(); // (
                    let args = self.parse_call_args()?;
                    e = Expr::Call { callee: Box::new(e), args };
                    continue;
                }
            }
            match self.peek(0) {
                Tok::LParen => {
                    self.bump();
                    let args = self.parse_call_args()?;
                    e = Expr::Call { callee: Box::new(e), args };
                }
                Tok::Dot => {
                    self.bump();
                    // P12.3 — `.await` postfix. Today's lowering is a
                    // pass-through (the value is the future itself); a real
                    // executor + state-machine transform is the deeper rewrite.
                    if matches!(self.peek(0), Tok::Await) {
                        self.bump();
                        continue;
                    }
                    // Tuple field syntax: `.0`, `.1`, etc. Numeric index is
                    // converted to the synthetic `<base>.<n>` slot key by the
                    // asm backend.
                    if let Tok::IntLit(n) = self.peek(0).clone() {
                        self.bump();
                        e = Expr::Field { recv: Box::new(e), name: n.to_string() };
                        continue;
                    }
                    let name = self.expect_ident()?;
                    if matches!(self.peek(0), Tok::LParen) {
                        self.bump();
                        let args = self.parse_call_args()?;
                        e = Expr::MethodCall { recv: Box::new(e), name, args };
                    } else {
                        e = Expr::Field { recv: Box::new(e), name };
                    }
                }
                Tok::As => {
                    self.bump();
                    let ty = self.parse_ty()?;
                    let ty_name = match ty {
                        Ty::Named(n) => n,
                        _ => return Err("`as` target must be a primitive type name".into()),
                    };
                    e = Expr::Cast { expr: Box::new(e), ty: ty_name };
                }
                Tok::LBracket => {
                    self.bump();
                    // P16.19 — `v[..]` full-range slice sugar. The bounds are
                    // placeholders (the container full-slice in emit_slice_construct
                    // ignores them, using the container's len). `v[i]` and
                    // `v[lo..hi]` still parse through parse_expr/parse_range.
                    let idx = if matches!(self.peek(0), Tok::DotDot) {
                        self.bump();
                        Expr::Range {
                            lo: Box::new(Expr::IntLit(0)),
                            hi: Box::new(Expr::IntLit(0)),
                            step: None,
                        }
                    } else {
                        self.parse_expr()?
                    };
                    self.expect(Tok::RBracket)?;
                    e = Expr::Index { recv: Box::new(e), idx: Box::new(idx) };
                }
                Tok::Question => {
                    // `expr?` — postfix try-operator.  Wrap the lhs in
                    // `Expr::Try(...)`; the asm backend desugars this to
                    // tag-check + early-return propagation.
                    self.bump();
                    e = Expr::Try(Box::new(e));
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
                // `()` parses as Unit-typed something (currently unused as an
                // expression — but keeps the parser symmetric).
                if matches!(self.peek(0), Tok::RParen) {
                    self.bump();
                    return Ok(Expr::Tuple(Vec::new()));
                }
                let first = self.parse_expr()?;
                if matches!(self.peek(0), Tok::Comma) {
                    self.bump();
                    let mut elems = vec![first];
                    while !matches!(self.peek(0), Tok::RParen) {
                        elems.push(self.parse_expr()?);
                        if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                    }
                    self.expect(Tok::RParen)?;
                    return Ok(Expr::Tuple(elems));
                }
                self.expect(Tok::RParen)?;
                Ok(first)
            }
            Tok::LBrace => Ok(Expr::Block(self.parse_block()?)),
            // P16.20 — `unsafe { ... }` is parsed and elided. The block runs
            // exactly as a normal block today; real raw-pointer semantics
            // (`*const T`, `*mut T`, `std::ptr::{read,write,copy}`) are FR-16.20.
            Tok::Unsafe => {
                self.bump();
                Ok(Expr::Block(self.parse_block()?))
            }
            Tok::If => {
                self.bump();
                // `if let PAT = SCRUT { THEN } else { ELSE }` desugars to
                // `match SCRUT { PAT => { THEN }, _ => { ELSE } }`. The match
                // codegen already handles enum-variant payload binding, so this
                // is a pure parser desugar onto existing machinery.
                if matches!(self.peek(0), Tok::Let) {
                    self.bump(); // `let`
                    let pat = self.parse_match_pat()?;
                    self.expect(Tok::Eq)?;
                    let scrut = self.with_struct_lit_disabled(|p| p.parse_expr())?;
                    let then = self.parse_block()?;
                    let else_arm = if matches!(self.peek(0), Tok::Else) {
                        self.bump();
                        if matches!(self.peek(0), Tok::If) {
                            let nested = self.parse_atom()?; // `else if` chains
                            Expr::Block(Block { stmts: Vec::new(), tail: Some(Box::new(nested)) })
                        } else {
                            Expr::Block(self.parse_block()?)
                        }
                    } else {
                        // No `else`: the non-matching arm yields unit (empty block).
                        Expr::Block(Block { stmts: Vec::new(), tail: None })
                    };
                    return Ok(Expr::Match {
                        scrutinee: Box::new(scrut),
                        arms: vec![
                            (pat, Expr::Block(then)),
                            (MatchPat::Wildcard, else_arm),
                        ],
                    });
                }
                let cond = self.with_struct_lit_disabled(|p| p.parse_expr())?;
                let then = self.parse_block()?;
                let else_ = if matches!(self.peek(0), Tok::Else) {
                    self.bump();
                    // `else if cond { ... }` desugars to `else { if cond { ... } }`
                    // so the chain composes the same way as Rust without nested
                    // braces in source.
                    if matches!(self.peek(0), Tok::If) {
                        let nested = self.parse_atom()?; // recurses into the If arm
                        Some(Block { stmts: Vec::new(), tail: Some(Box::new(nested)) })
                    } else {
                        Some(self.parse_block()?)
                    }
                } else { None };
                Ok(Expr::If { cond: Box::new(cond), then, else_ })
            }
            Tok::For => {
                self.bump();
                let var = self.expect_ident()?;
                self.expect(Tok::In)?;
                let iter = self.with_struct_lit_disabled(|p| p.parse_expr())?;
                let distributed = matches!(&iter, Expr::MethodCall { name, .. } if name == "distributed");
                let body = self.parse_block()?;
                Ok(Expr::For { var, iter: Box::new(iter), body, parallel: false, distributed })
            }
            Tok::While => {
                self.bump();
                let cond = self.with_struct_lit_disabled(|p| p.parse_expr())?;
                let body = self.parse_block()?;
                Ok(Expr::While { cond: Box::new(cond), body })
            }
            Tok::Match => {
                self.bump();
                let scrut = self.with_struct_lit_disabled(|p| p.parse_expr())?;
                self.expect(Tok::LBrace)?;
                let mut arms = Vec::new();
                while !matches!(self.peek(0), Tok::RBrace) {
                    let pat = self.parse_match_pat()?;
                    self.expect(Tok::FatArrow)?;
                    let arm_expr = self.parse_expr()?;
                    arms.push((pat, arm_expr));
                    if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                }
                self.expect(Tok::RBrace)?;
                Ok(Expr::Match { scrutinee: Box::new(scrut), arms })
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
                        let iter = self.with_struct_lit_disabled(|p| p.parse_expr())?;
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
                // Struct literal disambiguation: `Foo { ident : …` is a
                // struct literal in expression contexts where the
                // surrounding construct didn't disable it (i.e. not in
                // an if/while/for cond). `Foo { }` (empty) also counts.
                if path.len() == 1
                    && self.struct_lit_allowed
                    && matches!(self.peek(0), Tok::LBrace)
                    && (matches!(self.peek(1), Tok::RBrace) ||
                        (matches!(self.peek(1), Tok::Ident(_))
                         && matches!(self.peek(2), Tok::Colon)))
                {
                    self.bump(); // {
                    let mut fields = Vec::new();
                    while !matches!(self.peek(0), Tok::RBrace) {
                        let fname = self.expect_ident()?;
                        self.expect(Tok::Colon)?;
                        let value = self.parse_expr()?;
                        fields.push((fname, value));
                        if matches!(self.peek(0), Tok::Comma) { self.bump(); }
                    }
                    self.expect(Tok::RBrace)?;
                    return Ok(Expr::StructLit { name: path.into_iter().next().unwrap(), fields });
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

/// FR-16.14 — `println!("hello {} pi={:f}", name, pi)` style expansion.
///
/// At parse time we already see the format-string literal as `args[0]`. We
/// scan it for `{}` (i64 hole) and `{:f}` (f32 hole) placeholders, splitting
/// into literal segments and hole specifiers. Each literal segment lowers to
/// `aether_print_str_n(seg_ptr, seg_len)`; each hole lowers to one of the
/// scalar print primitives based on the hole spec; trailing newline iff the
/// macro is `println!`.
///
/// Limitations:
///   - format string MUST be a literal `StrLit` first arg.
///   - hole specifiers limited to `{}` (i64), `{:f}` (f32).
///   - escape `{{` / `}}` for literal braces.
///   - `{:.N}` precision, named args, positional args — NOT supported (file
///     as FR-16.14-extra when needed).
/// Falls back to a normal `name!(args)` call expression if the first arg is
/// not a string literal — the user can still write `println!(...)` against
/// a custom helper that has the pass-through signature.
fn expand_print_macro(name: &str, args: Vec<Expr>) -> PResult<Expr> {
    if args.is_empty() {
        // `println!()` with no args → just newline (println) or no-op (print).
        if name == "println" {
            return Ok(Expr::Call {
                callee: Box::new(Expr::Ident("aether_print_newline".into())),
                args: Vec::new(),
            });
        }
        return Ok(Expr::Block(Block { stmts: Vec::new(), tail: None }));
    }
    let fmt = match &args[0] {
        Expr::StrLit(s) => s.clone(),
        _ => {
            // Fallback: pass through as a call (current macro behavior).
            return Ok(Expr::Call {
                callee: Box::new(Expr::Ident(name.into())),
                args,
            });
        }
    };
    let value_args: Vec<Expr> = args.into_iter().skip(1).collect();
    // Walk the format string. For each segment, collect (Literal | Hole).
    // Holes alternate with literal segments. We materialize literal segments
    // as their own `StrLit` expressions; the asm backend interns them in
    // `.rdata` and produces a pointer + we know the byte length at parse time.
    let mut segments: Vec<Expr> = Vec::new();
    let mut hole_idx = 0usize;
    let mut current_lit = String::new();
    let mut chars = fmt.chars().peekable();
    let push_literal_segment = |seg: &str, segments: &mut Vec<Expr>| {
        if seg.is_empty() { return; }
        // Lower to: `aether_print_str_n(<str_lit>, <len>)`.
        // The asm backend's `Expr::Call` arg path doesn't auto-take the
        // address of a StrLit — but `Expr::StrLit` ALREADY lowers to a
        // pointer (it loads `lea .LC<n>(%rip), %rax`), so passing it
        // directly works.
        let n = seg.len() as i64;
        segments.push(Expr::Call {
            callee: Box::new(Expr::Ident("aether_print_str_n".into())),
            args: vec![Expr::StrLit(seg.to_string()), Expr::IntLit(n)],
        });
    };
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if matches!(chars.peek(), Some('{')) {
                    chars.next();
                    current_lit.push('{');
                    continue;
                }
                // Flush current literal.
                push_literal_segment(&current_lit, &mut segments);
                current_lit.clear();
                // Parse hole spec: read until '}'.
                let mut spec = String::new();
                while let Some(&n) = chars.peek() {
                    if n == '}' { chars.next(); break; }
                    spec.push(n);
                    chars.next();
                }
                if hole_idx >= value_args.len() {
                    return Err(format!("println!: not enough arguments for hole #{}", hole_idx));
                }
                let arg = value_args[hole_idx].clone();
                hole_idx += 1;
                // Dispatch: `{}` → i64, `{:f}` / `{:.<N>}` → f32, `{:s}` → AeString.
                let print_fn = match spec.as_str() {
                    "" => "aether_print_i64",
                    ":f" => "aether_print_f32_default",
                    s if s.starts_with(":.") => "aether_print_f32_default",
                    _ => "aether_print_i64",
                };
                segments.push(Expr::Call {
                    callee: Box::new(Expr::Ident(print_fn.into())),
                    args: vec![arg],
                });
            }
            '}' => {
                if matches!(chars.peek(), Some('}')) {
                    chars.next();
                    current_lit.push('}');
                } else {
                    return Err("println!: stray `}` in format string".into());
                }
            }
            other => current_lit.push(other),
        }
    }
    push_literal_segment(&current_lit, &mut segments);
    if name == "println" {
        segments.push(Expr::Call {
            callee: Box::new(Expr::Ident("aether_print_newline".into())),
            args: Vec::new(),
        });
    }
    // Lower to a Block: every print call as Stmt::Expr, no tail. The whole
    // expression's value is unit. Callers of `println!` use it as a
    // statement, never expect a return value.
    let stmts = segments.into_iter().map(Stmt::Expr).collect();
    Ok(Expr::Block(Block { stmts, tail: None }))
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
