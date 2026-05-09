//! AST for Aether — Phase 0 surface.

#[derive(Debug, Clone)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    Use(Vec<String>),
    ModuleDecl(Vec<String>),
    Struct(StructDecl),
    Const(ConstDecl),
    /// `impl Foo { fn name(…) … }` — at try_emit time each method gets
    /// flattened into a top-level `Item::Fn` with the name mangled to
    /// `<TypeName>__<method>`. Method-call dispatch (`obj.bar(x)`)
    /// then desugars to `Foo__bar(obj, x)` when `obj` is of struct
    /// type `Foo`.
    Impl { type_name: String, methods: Vec<FnDecl> },
    /// `trait Foo { fn bar(&self) -> i32; }` — declares a trait with a list
    /// of method signatures. Today only used by `mir::traits::Resolver` for
    /// completeness checks; static dispatch still goes through `Item::Impl`'s
    /// per-type method tables.
    Trait { name: String, methods: Vec<FnDecl> },
    /// `impl Foo for Bar { fn bar(&self) -> i32 { ... } }`. Lowered to the
    /// same `<Bar>__bar` mangling as inherent `impl`. The `trait_name` is
    /// recorded so the trait resolver can verify completeness.
    ImplTrait { trait_name: String, type_name: String, methods: Vec<FnDecl> },
    /// `enum Color { Red, Green, Blue }` — discriminant tags. Each variant
    /// gets a sequential i32 tag (Red=0, Green=1, ...). For variants with
    /// a payload (`Box::Full(i64)`), `payloads[i]` is `Some(Ty)`. Enums
    /// where any variant has a payload use a 2-slot layout (tag + val);
    /// payload-less enums stay as bare i64 tag values.
    Enum { name: String, variants: Vec<String>, payloads: Vec<Option<Ty>> },
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub is_pub: bool,
    pub name: String,
    pub generics: Vec<String>,
    pub fields: Vec<StructField>,
}

#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub is_pub: bool,
    pub name: String,
    pub ty: Ty,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub is_extern: bool,
    pub name: String,
    /// Const-generic param names (`fn forward<M, K>(...)` → `["M", "K"]`).
    /// Each is a placeholder symbolic dim that resolves at each call site
    /// to a concrete `i32`. Templates with non-empty `const_params` are
    /// emitted lazily, one specialization per unique inferred binding set.
    pub const_params: Vec<String>,
    pub params: Vec<Param>,
    pub ret: Option<Ty>,
    /// `None` for `extern fn name(...) -> T;` (forward decl into the runtime).
    pub body: Option<Block>,
}

#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub args: Vec<AttrArg>,
}

#[derive(Debug, Clone)]
pub struct AttrArg {
    pub key: Option<String>,
    pub value: AttrVal,
}

#[derive(Debug, Clone)]
pub enum AttrVal {
    Ident(String),
    Int(i64),
    Str(String),
    Bool(bool),
}

impl Attr {
    pub fn arg_int(&self, key: &str) -> Option<i64> {
        self.args.iter().find_map(|a| {
            if a.key.as_deref() == Some(key) {
                if let AttrVal::Int(n) = a.value { Some(n) } else { None }
            } else { None }
        })
    }

    pub fn arg_str(&self, key: &str) -> Option<&str> {
        self.args.iter().find_map(|a| {
            if a.key.as_deref() == Some(key) {
                if let AttrVal::Str(s) = &a.value { Some(s.as_str()) } else { None }
            } else { None }
        })
    }
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Ty,
}

#[derive(Debug, Clone)]
pub enum Ty {
    Named(String),
    Ref { mutable: bool, inner: Box<Ty> },
    Generic { name: String, args: Vec<Ty> },
    /// Const-shape array used for tensor shapes: `[M, K]`, `[batch, seq_len]`.
    /// Elements are either named symbols or int literals.
    Shape(Vec<ShapeDim>),
    /// Fixed-size stack array `[T; N]`. Length is a const-resolved usize.
    /// N consecutive slots reserved on the local frame; `buf[i]` indexes
    /// via `(base + i*8)` addressing.
    Array { elem: Box<Ty>, n: usize },
    /// Tuple type `(T1, T2, ...)`. Zero-cost — lowered to N synthetic field
    /// slots `<name>.0`, `<name>.1`, etc., and accessed via `.0`/`.1` field
    /// syntax. Reuses the struct machinery wholesale.
    Tuple(Vec<Ty>),
    Unit,
}

#[derive(Debug, Clone)]
pub enum ShapeDim {
    Sym(String),
    Const(i64),
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    /// `let name [: Ty] [= expr];`. `value` is `None` for an uninitialised
    /// stack-allocated declaration — currently only used for struct locals,
    /// which want per-field assignment after declaration.
    Let { name: String, mutable: bool, ty: Option<Ty>, value: Option<Expr> },
    /// `let (a, b, ...) = expr;` — tuple destructuring binding. Each name
    /// becomes its own top-level local; the rhs MUST be a tuple literal of
    /// matching arity (more general fn-returning-tuple awaits sret ABI).
    LetTuple { names: Vec<String>, value: Expr },
    Expr(Expr),
    Return(Option<Expr>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    BoolLit(bool),
    Ident(String),
    Call { callee: Box<Expr>, args: Vec<Expr> },
    MethodCall { recv: Box<Expr>, name: String, args: Vec<Expr> },
    Field { recv: Box<Expr>, name: String },
    Bin { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Unary { op: UnOp, expr: Box<Expr> },
    Block(Block),
    If { cond: Box<Expr>, then: Block, else_: Option<Block> },
    For { var: String, iter: Box<Expr>, body: Block, parallel: bool, distributed: bool },
    While { cond: Box<Expr>, body: Block },
    Break,
    Continue,
    Range { lo: Box<Expr>, hi: Box<Expr>, step: Option<Box<Expr>> },
    Path(Vec<String>),
    Ref { mutable: bool, expr: Box<Expr> },
    /// `*expr` — load through a reference. Codegen evaluates `expr` to rax
    /// (which holds the address) then issues `movq (%rax), %rax`.
    Deref(Box<Expr>),
    /// `warp { ... }` and `block { ... }` — GPU-shaped lexical scopes.
    Region { kind: RegionKind, body: Block },
    /// Struct literal: `Foo { a: 1, b: 2.0 }`. The parser disambiguates
    /// against `if cond { ... }` style blocks via a `no_struct_literal`
    /// flag — struct literals are forbidden in cond-of-if/while/for.
    StructLit { name: String, fields: Vec<(String, Expr)> },
    /// `match scrut { p1 => arm1, p2 => arm2, _ => default }`. Patterns
    /// are limited to `IntLit`, `Wildcard`, and `EnumVariant` (e.g.
    /// `Color::Red`). Discriminant-only — no value bindings yet. Each
    /// arm is a single expr; for blocks use `{ … }`.
    Match { scrutinee: Box<Expr>, arms: Vec<(MatchPat, Expr)> },
    /// `expr as Type` — numeric coercion. Lowered through `emit_cast` in the
    /// asm backend; supports int↔f32, int↔f64, f32↔f64.
    Cast { expr: Box<Expr>, ty: String },
    /// `recv[idx]` — indexed access into a stack array `[T; N]` local. Both
    /// read (`x = buf[i]`) and write (`buf[i] = x`) lower through this; the
    /// asm backend disambiguates by where the Index sits in a Bin::Assign.
    Index { recv: Box<Expr>, idx: Box<Expr> },
    /// `(e1, e2, ...)` tuple literal. Used exclusively as the rhs of a
    /// `let pair = (...)` style binding; the asm backend lowers it the same
    /// way as a struct literal — N synthetic per-element slots.
    Tuple(Vec<Expr>),
    /// `|x, y| expr` closure expression. Today this is a NO-CAPTURE
    /// closure — a `mir/closures.rs` pre-codegen pass lifts every Closure
    /// into a synthetic top-level `__closure_<n>` fn and rewrites the
    /// expression to `Expr::Ident("__closure_<n>")` (which the asm backend
    /// loads as a function pointer via `leaq aether_<name>(%rip), %rax`).
    /// Closures-with-captures need an env-struct allocation + indirect-call
    /// ABI — future work; the lifted fn becomes a method on the env then.
    Closure { params: Vec<(String, Option<Ty>)>, body: Box<Expr> },
    /// `expr?` — try-operator. Lowered by the asm backend to:
    ///   match expr { Ok(v) => v, Err(e) => return Err(e) }
    /// `expr` MUST be a `Call` to a fn whose return type is a payload-enum
    /// declared in the current program (concretely `Result`-shaped: variant 0
    /// is "Ok" carrying a payload, variant 1 is "Err" carrying a payload).
    /// The enclosing fn's return type must be the same payload-enum so the
    /// early-return path can propagate the error variant unchanged.
    Try(Box<Expr>),
}

#[derive(Debug, Clone)]
pub enum MatchPat {
    Int(i64),
    /// `Color::Red` — a path of length 2; the asm backend resolves it
    /// to the variant's i32 tag at codegen time.
    EnumVariant(Vec<String>),
    /// `Box::Full(x)` — payload-carrying variant pattern. After tag-cmp
    /// matches, the payload slot is copied into a freshly-introduced
    /// local named `bind` for use in the arm's body.
    EnumVariantBind(Vec<String>, String),
    Wildcard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind { Warp, Block, AiRegion }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Gt, Le, Ge, And, Or, BitAnd, BitOr, BitXor, Shl, Shr, Assign }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp { Neg, Not }
