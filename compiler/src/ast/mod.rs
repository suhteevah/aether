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
    Let { name: String, mutable: bool, ty: Option<Ty>, value: Expr },
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind { Warp, Block, AiRegion }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp { Add, Sub, Mul, Div, Mod, Eq, Ne, Lt, Gt, Le, Ge, And, Or, Assign }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp { Neg, Not }
