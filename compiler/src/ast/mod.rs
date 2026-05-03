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
    /// `enum Color { Red, Green, Blue }` — discriminant-only enums for
    /// now. Each variant gets a sequential i32 tag (Red=0, Green=1, ...).
    /// `Color::Red` desugars to `IntLit(0)`. `match` dispatches on the
    /// scrutinee's tag value via cmp+jmp. Data-carrying variants are
    /// future work.
    Enum { name: String, variants: Vec<String> },
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
}

#[derive(Debug, Clone)]
pub enum MatchPat {
    Int(i64),
    /// `Color::Red` — a path of length 2; the asm backend resolves it
    /// to the variant's i32 tag at codegen time.
    EnumVariant(Vec<String>),
    Wildcard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind { Warp, Block, AiRegion }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Gt, Le, Ge, And, Or, BitAnd, BitOr, BitXor, Assign }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp { Neg, Not }
