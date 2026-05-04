//! x86-64 instruction encoder for the aetherc-emitted subset.
//!
//! Instruction encoding refs: Intel SDM Vol. 2.  Each variant below documents
//! the exact byte pattern.  REX.W is always 0x48 here because aetherc emits
//! 64-bit forms; smaller widths are added on demand.
//!
//! "Rip-relative LEA / CALL / MOV" emit 4 zero bytes for the displacement and
//! return a relocation site; the COFF builder rewrites them after symbol
//! addresses are known.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XmmReg {
    Xmm0 = 0, Xmm1 = 1, Xmm2 = 2, Xmm3 = 3,
    Xmm4 = 4, Xmm5 = 5, Xmm6 = 6, Xmm7 = 7,
}
impl XmmReg {
    pub fn lo3(self) -> u8 { (self as u8) & 0b111 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CondCode {
    /// equal (ZF=1)
    E  = 0x4,
    /// not equal (ZF=0)
    Ne = 0x5,
    /// signed less (SF != OF)
    L  = 0xC,
    /// signed less-or-equal (ZF=1 or SF != OF)
    Le = 0xE,
    /// signed greater (ZF=0 and SF=OF)
    G  = 0xF,
    /// signed greater-or-equal (SF=OF)
    Ge = 0xD,
    /// unsigned/ucomiss above (CF=0, ZF=0)
    A  = 0x7,
    /// unsigned/ucomiss below (CF=1)
    B  = 0x2,
    /// unsigned above-or-equal (CF=0)
    Ae = 0x3,
    /// unsigned below-or-equal (CF=1 or ZF=1)
    Be = 0x6,
}

impl CondCode {
    pub fn opcode_byte(self) -> u8 { self as u8 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reg {
    Rax = 0, Rcx = 1, Rdx = 2, Rbx = 3,
    Rsp = 4, Rbp = 5, Rsi = 6, Rdi = 7,
    R8 = 8, R9 = 9, R10 = 10, R11 = 11,
    R12 = 12, R13 = 13, R14 = 14, R15 = 15,
}

impl Reg {
    pub fn lo3(self) -> u8 { (self as u8) & 0b111 }
    pub fn extension(self) -> u8 { ((self as u8) >> 3) & 1 }
}

/// Instruction enum for the subset aetherc emits today.
#[derive(Debug, Clone)]
pub enum Instr {
    PushReg(Reg),
    PopReg(Reg),
    MovRegReg { dst: Reg, src: Reg },
    /// `mov r/m64, imm32` — sign-extended.
    MovRegImm32 { dst: Reg, imm: i32 },
    /// `mov r64, [rbp + disp32]` — load from stack slot.
    MovRegFromRbpDisp { dst: Reg, disp: i32 },
    /// `movq disp(%base), %dst` for a non-rbp/rsp 64-bit base register.
    /// Used by stack-array index reads (`movq 0(%rdi), %rax`) where the
    /// base address has already been computed into `%rdi` by `lea + add`.
    MovRegFromBaseDisp { dst: Reg, base: Reg, disp: i32 },
    /// `movq %src, disp(%base)` for a non-rbp/rsp 64-bit base register.
    /// Used by stack-array index writes.
    MovBaseDispFromReg { src: Reg, base: Reg, disp: i32 },
    /// `mov [rbp + disp32], r64` — store to stack slot.
    MovRbpDispFromReg { src: Reg, disp: i32 },
    /// `movq $imm32, disp32(%rbp)` — store sign-extended imm32 directly to a
    /// stack slot. Used by the asm peephole pass to collapse the
    /// `movq $imm, %rax / movq %rax, -N(%rbp)` rax round-trip.
    /// Encoding: REX.W + C7 /0 ; ModR/M = 10_000_101 ; disp32 ; imm32 (11 bytes).
    MovRbpDispImm32 { disp: i32, imm: i32 },
    AddRegImm8 { dst: Reg, imm: i8 },
    SubRegImm8 { dst: Reg, imm: i8 },
    /// `add r/m64, imm32` — REX.W + 81 /0 imm32. 7 bytes. Use when imm > 127.
    AddRegImm32 { dst: Reg, imm: i32 },
    /// `sub r/m64, imm32` — REX.W + 81 /5 imm32. 7 bytes. Use when imm > 127.
    SubRegImm32 { dst: Reg, imm: i32 },
    /// `xor r32, r32` — clears upper 32 bits as well.
    XorRegReg32 { dst: Reg, src: Reg },
    /// `andq %src, %dst` → dst = dst & src. REX.W + 21 /r.
    AndRegRegQ { dst: Reg, src: Reg },
    /// `orq %src, %dst`  → dst = dst | src. REX.W + 09 /r.
    OrRegRegQ  { dst: Reg, src: Reg },
    /// `xorq %src, %dst` → dst = dst ^ src. REX.W + 31 /r.
    XorRegRegQ { dst: Reg, src: Reg },
    /// `shlq %cl, %dst` — left shift dst by count in CL. REX.W + D3 /4.
    ShlRegByCl { dst: Reg },
    /// `sarq %cl, %dst` — arithmetic right shift (signed) by count in CL.
    /// Matches Rust's `>>` semantics for signed integers. REX.W + D3 /7.
    SarRegByCl { dst: Reg },
    /// `add r/m64, r64` — REX.W + 01 /r
    AddRegRegQ { dst: Reg, src: Reg },
    /// `sub r/m64, r64` — REX.W + 29 /r
    SubRegRegQ { dst: Reg, src: Reg },
    /// `imul r64, r/m64` — REX.W + 0F AF /r
    ImulRegRegQ { dst: Reg, src: Reg },
    /// `xchg r64, r/m64` — REX.W + 87 /r (swap two registers)
    XchgRegRegQ { dst: Reg, src: Reg },
    /// `neg r/m64` — REX.W + F7 /3
    NegRegQ { dst: Reg },
    /// `cqo` — sign-extend RAX into RDX:RAX. REX.W + 99.
    CqoSignExt,
    /// `idiv r/m64` — RDX:RAX / src; quotient → RAX, remainder → RDX.
    /// REX.W + F7 /7
    IdivRegQ { src: Reg },

    // -------- SSE2 single-precision float (f32) -----------------------------
    /// `movss disp32(%rbp), %xmm` — load f32 from stack slot.
    MovssRbpDispToXmm { dst: XmmReg, disp: i32 },
    /// `movss %xmm, disp32(%rbp)` — store f32 to stack slot.
    MovssXmmToRbpDisp { src: XmmReg, disp: i32 },
    /// `movss sym(%rip), %xmm` — load f32 constant from .rdata.
    MovssRipSymToXmm { dst: XmmReg, sym: String },
    /// `movss %src, %dst` — register-register move.
    MovssXmmXmm { dst: XmmReg, src: XmmReg },
    /// `addss %src, %dst` — dst += src (AT&T order).
    AddssXmmXmm { dst: XmmReg, src: XmmReg },
    SubssXmmXmm { dst: XmmReg, src: XmmReg },
    MulssXmmXmm { dst: XmmReg, src: XmmReg },
    DivssXmmXmm { dst: XmmReg, src: XmmReg },
    /// `ucomiss %src, %dst` — sets flags based on `dst <=> src`.
    UcomissXmmXmm { dst: XmmReg, src: XmmReg },
    /// `movss (%rsp), %xmm` — load from top of stack.
    MovssRspToXmm { dst: XmmReg },
    /// `movss %xmm, (%rsp)` — store to top of stack.
    MovssXmmToRsp { src: XmmReg },

    // -------- SSE2 double-precision float (f64) -----------------------------
    // Same op shapes as the ss family but with the `F2` SSE prefix instead of
    // `F3` (and `66` for ucomisd, mirroring how ucomiss has no prefix).
    MovsdRbpDispToXmm { dst: XmmReg, disp: i32 },
    MovsdXmmToRbpDisp { src: XmmReg, disp: i32 },
    MovsdRipSymToXmm { dst: XmmReg, sym: String },
    MovsdXmmXmm { dst: XmmReg, src: XmmReg },
    AddsdXmmXmm { dst: XmmReg, src: XmmReg },
    SubsdXmmXmm { dst: XmmReg, src: XmmReg },
    MulsdXmmXmm { dst: XmmReg, src: XmmReg },
    DivsdXmmXmm { dst: XmmReg, src: XmmReg },
    /// `ucomisd %src, %dst` — `66 0F 2E /r`.
    UcomisdXmmXmm { dst: XmmReg, src: XmmReg },
    MovsdRspToXmm { dst: XmmReg },
    MovsdXmmToRsp { src: XmmReg },

    // -------- int ↔ float conversions -------------------------------------
    /// `cvtsi2ss %src(r64), %dst(xmm)` — `F3 REX.W 0F 2A /r`.
    Cvtsi2ssRegToXmm { dst: XmmReg, src: Reg },
    /// `cvtss2si %src(xmm), %dst(r64)` — `F3 REX.W 0F 2D /r`.
    Cvtss2siXmmToReg { dst: Reg, src: XmmReg },
    /// `cvtsi2sd %src(r64), %dst(xmm)` — `F2 REX.W 0F 2A /r`.
    Cvtsi2sdRegToXmm { dst: XmmReg, src: Reg },
    /// `cvtsd2si %src(xmm), %dst(r64)` — `F2 REX.W 0F 2D /r`.
    Cvtsd2siXmmToReg { dst: Reg, src: XmmReg },
    /// `cvtss2sd %src, %dst` — f32 → f64. `F3 0F 5A /r`.
    Cvtss2sdXmmXmm { dst: XmmReg, src: XmmReg },
    /// `cvtsd2ss %src, %dst` — f64 → f32. `F2 0F 5A /r`.
    Cvtsd2ssXmmXmm { dst: XmmReg, src: XmmReg },

    // -------- stack-relative stores (for FFI args 5+) ----------------------
    /// `movq %src, disp(%rsp)` — REX.W + 89 /r + SIB(rsp) + disp32. 8 bytes.
    MovRspDispFromReg { src: Reg, disp: i32 },
    /// `movq disp(%rsp), %dst` — REX.W + 8B /r + SIB(rsp) + disp32. 8 bytes.
    MovRegFromRspDisp { dst: Reg, disp: i32 },
    /// `movss %xmm, disp(%rsp)` — F3 0F 11 /r + SIB(rsp) + disp32. 9 bytes.
    MovssXmmToRspDisp { src: XmmReg, disp: i32 },
    /// `movsd %xmm, disp(%rsp)` — F2 0F 11 /r + SIB(rsp) + disp32. 9 bytes.
    MovsdXmmToRspDisp { src: XmmReg, disp: i32 },
    /// `movss disp(%rsp), %xmm` — F3 0F 10 /r + SIB(rsp) + disp32. 9 bytes.
    MovssRspDispToXmm { dst: XmmReg, disp: i32 },
    /// `movsd disp(%rsp), %xmm` — F2 0F 10 /r + SIB(rsp) + disp32. 9 bytes.
    MovsdRspDispToXmm { dst: XmmReg, disp: i32 },
    /// `cmpq %src, %dst` (AT&T): flags = dst - src.   REX.W + 39 /r
    CmpRegRegQ { dst: Reg, src: Reg },
    /// `testq %a, %b` — flags = a & b. REX.W + 85 /r.
    TestRegRegQ { a: Reg, b: Reg },
    /// `setcc %al` — write 1 / 0 to AL based on flags. 0F 9X C0.
    SetccAl { cc: CondCode },
    /// `movzbl %al, %eax` — zero-extend AL to EAX. 0F B6 C0.
    MovzblAlEax,
    /// `jcc rel32` — 0F 8X cd. Resolved as a Rel32Pc relocation against `sym`.
    JccRel32 { cc: CondCode, sym: String },
    /// `jmp rel32` — E9 cd. Same relocation kind.
    JmpRel32 { sym: String },
    /// `lea r64, [rip + symbol]` — emits a placeholder rel32 patched by the linker.
    LeaRipSym { dst: Reg, sym: String },
    /// `lea r64, disp32(%rbp)` — address of a stack slot.
    LeaRegFromRbpDisp { dst: Reg, disp: i32 },
    /// `call rel32` — symbol resolved via PLT-style external relocation.
    CallSym { sym: String },
    /// `callq *%reg` — indirect call through a 64-bit register holding a
    /// function pointer. Used for invoking fn-typed locals (closures-lite).
    /// FF /2 with REX.B for r8..r15.
    CallRegIndirect { reg: Reg },
    Ret,
}

/// One byte fragment + (optionally) a pending symbol relocation.
pub struct Encoded {
    pub bytes: Vec<u8>,
    pub reloc: Option<PendingReloc>,
}

pub struct PendingReloc {
    pub sym: String,
    /// Offset in `bytes` where the rel32 starts.
    pub offset_in_instr: usize,
    pub kind: PendingRelocKind,
}

#[derive(Clone, Copy)]
pub enum PendingRelocKind {
    /// CALL or LEA-RIP — addend is `-4` (rel32 from end of instruction).
    Rel32Pc,
}

pub fn encode_instruction(i: &Instr) -> Encoded {
    use Instr::*;
    match i {
        PushReg(r) => {
            // 50+rd  — 4x prefix needed if r ≥ R8.
            let mut b = Vec::new();
            if r.extension() != 0 { b.push(0x41); }
            b.push(0x50 | r.lo3());
            Encoded { bytes: b, reloc: None }
        }
        PopReg(r) => {
            let mut b = Vec::new();
            if r.extension() != 0 { b.push(0x41); }
            b.push(0x58 | r.lo3());
            Encoded { bytes: b, reloc: None }
        }
        MovRegReg { dst, src } => {
            // REX.W + 89 /r   (mov r/m64, r64)
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x89, modrm], reloc: None }
        }
        MovRegImm32 { dst, imm } => {
            // REX.W + C7 /0 imm32   (mov r/m64, imm32 sign-ext)
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_000_000 | dst.lo3();
            let mut b = vec![rex, 0xC7, modrm];
            b.extend_from_slice(&imm.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        AddRegImm8 { dst, imm } => {
            // REX.W + 83 /0 ib  (add r/m64, imm8)
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_000_000 | dst.lo3();
            Encoded { bytes: vec![rex, 0x83, modrm, *imm as u8], reloc: None }
        }
        SubRegImm8 { dst, imm } => {
            // REX.W + 83 /5 ib  (sub r/m64, imm8 sign-ext)
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_101_000 | dst.lo3();
            Encoded { bytes: vec![rex, 0x83, modrm, *imm as u8], reloc: None }
        }
        AddRegImm32 { dst, imm } => {
            // REX.W + 81 /0 imm32   (add r/m64, imm32 sign-ext)
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_000_000 | dst.lo3();
            let mut b = vec![rex, 0x81, modrm];
            b.extend_from_slice(&imm.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        SubRegImm32 { dst, imm } => {
            // REX.W + 81 /5 imm32   (sub r/m64, imm32 sign-ext)
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_101_000 | dst.lo3();
            let mut b = vec![rex, 0x81, modrm];
            b.extend_from_slice(&imm.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        AndRegRegQ { dst, src } => {
            // REX.W + 21 /r — and r/m64, r64.
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x21, modrm], reloc: None }
        }
        OrRegRegQ { dst, src } => {
            // REX.W + 09 /r — or r/m64, r64.
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x09, modrm], reloc: None }
        }
        XorRegRegQ { dst, src } => {
            // REX.W + 31 /r — xor r/m64, r64.
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x31, modrm], reloc: None }
        }
        ShlRegByCl { dst } => {
            // REX.W + D3 /4 — shl r/m64, CL. ModRM = 11 100 dst.lo3.
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_100_000 | dst.lo3();
            Encoded { bytes: vec![rex, 0xD3, modrm], reloc: None }
        }
        SarRegByCl { dst } => {
            // REX.W + D3 /7 — sar r/m64, CL.
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_111_000 | dst.lo3();
            Encoded { bytes: vec![rex, 0xD3, modrm], reloc: None }
        }
        XorRegReg32 { dst, src } => {
            // 31 /r   — REX optional. If both regs are RAX..RDI, no REX.
            let need_rex = dst.extension() != 0 || src.extension() != 0;
            let mut b = Vec::new();
            if need_rex {
                b.push(0x40 | (src.extension() << 2) | dst.extension());
            }
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            b.extend_from_slice(&[0x31, modrm]);
            Encoded { bytes: b, reloc: None }
        }
        LeaRipSym { dst, sym } => {
            // REX.W + 8D /r ; ModRM = 00 reg 101 (RIP+disp32)
            let rex = 0x48 | (dst.extension() << 2);
            let modrm = 0b00_000_101 | (dst.lo3() << 3);
            let mut b = vec![rex, 0x8D, modrm, 0, 0, 0, 0];
            // Reloc starts at offset 3 (inside `b`).
            Encoded {
                bytes: b,
                reloc: Some(PendingReloc {
                    sym: sym.clone(),
                    offset_in_instr: 3,
                    kind: PendingRelocKind::Rel32Pc,
                }),
            }
        }
        CallSym { sym } => {
            // E8 rel32
            let mut b = vec![0xE8, 0, 0, 0, 0];
            Encoded {
                bytes: b,
                reloc: Some(PendingReloc {
                    sym: sym.clone(),
                    offset_in_instr: 1,
                    kind: PendingRelocKind::Rel32Pc,
                }),
            }
        }
        CallRegIndirect { reg } => {
            // FF /2 — call r/m64. ModRM = 11 010 reg.lo3.
            // REX.B for r8..r15.
            let mut bytes = Vec::new();
            if reg.extension() != 0 { bytes.push(0x40 | reg.extension()); }
            bytes.push(0xFF);
            bytes.push(0b11_010_000 | reg.lo3());
            Encoded { bytes, reloc: None }
        }
        Ret => Encoded { bytes: vec![0xC3], reloc: None },
        AddRegRegQ { dst, src } => {
            // REX.W + 01 /r   (add r/m64, r64). REX.R for src ext, REX.B for dst.
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x01, modrm], reloc: None }
        }
        SubRegRegQ { dst, src } => {
            // REX.W + 29 /r
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x29, modrm], reloc: None }
        }
        ImulRegRegQ { dst, src } => {
            // REX.W + 0F AF /r — note the operand encoding: ModRM.reg = dst,
            // ModRM.r/m = src. REX.R is dst extension, REX.B is src extension.
            let rex = 0x48 | (dst.extension() << 2) | src.extension();
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![rex, 0x0F, 0xAF, modrm], reloc: None }
        }
        LeaRegFromRbpDisp { dst, disp } => {
            // REX.W + 8D /r ; ModR/M = 10 reg 101 (rbp + disp32).
            let rex = 0x48 | (dst.extension() << 2);
            let modrm = 0b10_000_101 | (dst.lo3() << 3);
            let mut b = vec![rex, 0x8D, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovRegFromRbpDisp { dst, disp } => {
            // REX.W + 8B /r ; ModR/M = 10 reg 101 (rbp + disp32). Note: rbp
            // as base requires explicit disp8/disp32 (mod != 00).
            let rex = 0x48 | (dst.extension() << 2);
            let modrm = 0b10_000_101 | (dst.lo3() << 3);
            let mut b = vec![rex, 0x8B, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovRbpDispFromReg { src, disp } => {
            // REX.W + 89 /r ; ModR/M = 10 reg 101 (rbp + disp32). reg field is src.
            let rex = 0x48 | (src.extension() << 2);
            let modrm = 0b10_000_101 | (src.lo3() << 3);
            let mut b = vec![rex, 0x89, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovRbpDispImm32 { disp, imm } => {
            // REX.W + C7 /0 ; ModR/M = 10_000_101 (rbp + disp32) ; disp32 ; imm32.
            // 11 bytes total. Direct memory immediate store — no reg round-trip.
            let mut b = vec![0x48u8, 0xC7, 0b10_000_101u8];
            b.extend_from_slice(&disp.to_le_bytes());
            b.extend_from_slice(&imm.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovRegFromBaseDisp { dst, base, disp } => {
            // REX.W + 8B /r. Mod=10 (disp32) when disp != 0 OR base needs explicit
            // disp (rbp/r13); else Mod=00. base goes in r/m. SIB needed when
            // base is rsp/r12 (rm=4 means SIB) — caller shouldn't hit those here.
            assert!(!matches!(base, Reg::Rsp), "MovRegFromBaseDisp with rsp base needs SIB");
            let rex = 0x48 | (dst.extension() << 2) | base.extension();
            let modrm = 0b10_000_000 | (dst.lo3() << 3) | base.lo3();
            let mut b = vec![rex, 0x8B, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovBaseDispFromReg { src, base, disp } => {
            // REX.W + 89 /r. Mod=10 disp32. base in r/m, src in reg field.
            assert!(!matches!(base, Reg::Rsp), "MovBaseDispFromReg with rsp base needs SIB");
            let rex = 0x48 | (src.extension() << 2) | base.extension();
            let modrm = 0b10_000_000 | (src.lo3() << 3) | base.lo3();
            let mut b = vec![rex, 0x89, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        XchgRegRegQ { dst, src } => {
            // REX.W + 87 /r   (xchg r/m64, r64)
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x87, modrm], reloc: None }
        }
        NegRegQ { dst } => {
            // REX.W + F7 /3 — ModRM = 11 011 dst
            let rex = 0x48 | dst.extension();
            let modrm = 0b11_011_000 | dst.lo3();
            Encoded { bytes: vec![rex, 0xF7, modrm], reloc: None }
        }
        CqoSignExt => {
            // 48 99
            Encoded { bytes: vec![0x48, 0x99], reloc: None }
        }
        IdivRegQ { src } => {
            // REX.W + F7 /7 — ModRM = 11 111 src
            let rex = 0x48 | src.extension();
            let modrm = 0b11_111_000 | src.lo3();
            Encoded { bytes: vec![rex, 0xF7, modrm], reloc: None }
        }

        // ------------------ SSE2 f32 -------------------------------------
        MovssRbpDispToXmm { dst, disp } => {
            // F3 0F 10 /r ; ModR/M = 10 reg 101 (rbp + disp32)
            let modrm = 0b10_000_101 | (dst.lo3() << 3);
            let mut b = vec![0xF3, 0x0F, 0x10, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovssXmmToRbpDisp { src, disp } => {
            // F3 0F 11 /r ; ModR/M = 10 reg 101
            let modrm = 0b10_000_101 | (src.lo3() << 3);
            let mut b = vec![0xF3, 0x0F, 0x11, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovssRipSymToXmm { dst, sym } => {
            // F3 0F 10 /r ; ModR/M = 00 reg 101 (rip + disp32). Reloc at offset 4.
            let modrm = 0b00_000_101 | (dst.lo3() << 3);
            let bytes = vec![0xF3, 0x0F, 0x10, modrm, 0, 0, 0, 0];
            Encoded {
                bytes,
                reloc: Some(PendingReloc {
                    sym: sym.clone(),
                    offset_in_instr: 4,
                    kind: PendingRelocKind::Rel32Pc,
                }),
            }
        }
        MovssXmmXmm { dst, src } => {
            // F3 0F 10 /r ; ModR/M = 11 reg(dst) rm(src)
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, 0x0F, 0x10, modrm], reloc: None }
        }
        AddssXmmXmm { dst, src } => {
            // F3 0F 58 /r
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, 0x0F, 0x58, modrm], reloc: None }
        }
        SubssXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, 0x0F, 0x5C, modrm], reloc: None }
        }
        MulssXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, 0x0F, 0x59, modrm], reloc: None }
        }
        DivssXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, 0x0F, 0x5E, modrm], reloc: None }
        }
        UcomissXmmXmm { dst, src } => {
            // 0F 2E /r ; ModR/M = 11 reg(dst) rm(src) — flags = dst <=> src
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0x0F, 0x2E, modrm], reloc: None }
        }
        MovssRspToXmm { dst } => {
            // F3 0F 10 ModRM SIB ; ModRM = 00 reg(dst) 100 (SIB follows)
            // SIB = 00 100 100 (scale=0, index=none, base=rsp)
            let modrm = 0b00_000_100 | (dst.lo3() << 3);
            Encoded { bytes: vec![0xF3, 0x0F, 0x10, modrm, 0x24], reloc: None }
        }
        MovssXmmToRsp { src } => {
            let modrm = 0b00_000_100 | (src.lo3() << 3);
            Encoded { bytes: vec![0xF3, 0x0F, 0x11, modrm, 0x24], reloc: None }
        }

        // ------------------ SSE2 f64 -------------------------------------
        MovsdRbpDispToXmm { dst, disp } => {
            let modrm = 0b10_000_101 | (dst.lo3() << 3);
            let mut b = vec![0xF2, 0x0F, 0x10, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovsdXmmToRbpDisp { src, disp } => {
            let modrm = 0b10_000_101 | (src.lo3() << 3);
            let mut b = vec![0xF2, 0x0F, 0x11, modrm];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovsdRipSymToXmm { dst, sym } => {
            let modrm = 0b00_000_101 | (dst.lo3() << 3);
            let bytes = vec![0xF2, 0x0F, 0x10, modrm, 0, 0, 0, 0];
            Encoded {
                bytes,
                reloc: Some(PendingReloc {
                    sym: sym.clone(),
                    offset_in_instr: 4,
                    kind: PendingRelocKind::Rel32Pc,
                }),
            }
        }
        MovsdXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, 0x0F, 0x10, modrm], reloc: None }
        }
        AddsdXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, 0x0F, 0x58, modrm], reloc: None }
        }
        SubsdXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, 0x0F, 0x5C, modrm], reloc: None }
        }
        MulsdXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, 0x0F, 0x59, modrm], reloc: None }
        }
        DivsdXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, 0x0F, 0x5E, modrm], reloc: None }
        }
        UcomisdXmmXmm { dst, src } => {
            // 66 0F 2E /r
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0x66, 0x0F, 0x2E, modrm], reloc: None }
        }
        MovsdRspToXmm { dst } => {
            let modrm = 0b00_000_100 | (dst.lo3() << 3);
            Encoded { bytes: vec![0xF2, 0x0F, 0x10, modrm, 0x24], reloc: None }
        }
        MovsdXmmToRsp { src } => {
            let modrm = 0b00_000_100 | (src.lo3() << 3);
            Encoded { bytes: vec![0xF2, 0x0F, 0x11, modrm, 0x24], reloc: None }
        }

        // ------------------ int ↔ float conversions ----------------------
        // ModRM.reg = xmm, ModRM.r/m = gpr (or vice versa for cvtss2si).
        // REX.W is set so the integer side is 64-bit.
        Cvtsi2ssRegToXmm { dst, src } => {
            // F3 REX.W 0F 2A /r ; ModRM = 11 reg(dst-xmm) rm(src-gpr)
            let rex = 0x48 | src.extension();
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, rex, 0x0F, 0x2A, modrm], reloc: None }
        }
        Cvtss2siXmmToReg { dst, src } => {
            // F3 REX.W 0F 2D /r ; ModRM = 11 reg(dst-gpr) rm(src-xmm)
            let rex = 0x48 | (dst.extension() << 2);
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, rex, 0x0F, 0x2D, modrm], reloc: None }
        }
        Cvtsi2sdRegToXmm { dst, src } => {
            let rex = 0x48 | src.extension();
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, rex, 0x0F, 0x2A, modrm], reloc: None }
        }
        Cvtsd2siXmmToReg { dst, src } => {
            let rex = 0x48 | (dst.extension() << 2);
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, rex, 0x0F, 0x2D, modrm], reloc: None }
        }
        Cvtss2sdXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF3, 0x0F, 0x5A, modrm], reloc: None }
        }
        Cvtsd2ssXmmXmm { dst, src } => {
            let modrm = 0b11_000_000 | (dst.lo3() << 3) | src.lo3();
            Encoded { bytes: vec![0xF2, 0x0F, 0x5A, modrm], reloc: None }
        }

        // ------------------ stack-relative stores ------------------------
        // `[rsp+disp32]` requires SIB byte: SIB = 00 100 100 (no scale, no
        // index, base=rsp). ModRM uses mod=10 with rm=100 (SIB follows).
        MovRspDispFromReg { src, disp } => {
            let rex = 0x48 | (src.extension() << 2);
            let modrm = 0b10_000_100 | (src.lo3() << 3);
            let sib = 0x24;
            let mut b = vec![rex, 0x89, modrm, sib];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovRegFromRspDisp { dst, disp } => {
            // REX.W + 8B /r + ModRM(mod=10, reg=dst, rm=100) + SIB(0x24) + disp32
            let rex = 0x48 | (dst.extension() << 2);
            let modrm = 0b10_000_100 | (dst.lo3() << 3);
            let sib = 0x24;
            let mut b = vec![rex, 0x8B, modrm, sib];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovssXmmToRspDisp { src, disp } => {
            let modrm = 0b10_000_100 | (src.lo3() << 3);
            let mut b = vec![0xF3, 0x0F, 0x11, modrm, 0x24];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovsdXmmToRspDisp { src, disp } => {
            let modrm = 0b10_000_100 | (src.lo3() << 3);
            let mut b = vec![0xF2, 0x0F, 0x11, modrm, 0x24];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovssRspDispToXmm { dst, disp } => {
            let modrm = 0b10_000_100 | (dst.lo3() << 3);
            let mut b = vec![0xF3, 0x0F, 0x10, modrm, 0x24];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        MovsdRspDispToXmm { dst, disp } => {
            let modrm = 0b10_000_100 | (dst.lo3() << 3);
            let mut b = vec![0xF2, 0x0F, 0x10, modrm, 0x24];
            b.extend_from_slice(&disp.to_le_bytes());
            Encoded { bytes: b, reloc: None }
        }
        CmpRegRegQ { dst, src } => {
            // REX.W + 39 /r — `cmp r/m64, r64`. AT&T syntax `cmpq %src, %dst`
            // sets flags as if dst - src. ModRM.reg = src, ModRM.rm = dst.
            let rex = 0x48 | (src.extension() << 2) | dst.extension();
            let modrm = 0b11_000_000 | (src.lo3() << 3) | dst.lo3();
            Encoded { bytes: vec![rex, 0x39, modrm], reloc: None }
        }
        TestRegRegQ { a, b } => {
            // REX.W + 85 /r
            let rex = 0x48 | (b.extension() << 2) | a.extension();
            let modrm = 0b11_000_000 | (b.lo3() << 3) | a.lo3();
            Encoded { bytes: vec![rex, 0x85, modrm], reloc: None }
        }
        SetccAl { cc } => {
            // 0F 9X C0 — sets AL = 1 if condition else 0. No REX needed (AL).
            let opcode = 0x90 | cc.opcode_byte();
            Encoded { bytes: vec![0x0F, opcode, 0xC0], reloc: None }
        }
        MovzblAlEax => {
            // 0F B6 C0 — movzbl %al, %eax. Clears upper 32 bits of rax too.
            Encoded { bytes: vec![0x0F, 0xB6, 0xC0], reloc: None }
        }
        JccRel32 { cc, sym } => {
            // 0F 8X cd — 6 bytes total. rel32 starts at offset 2.
            let opcode = 0x80 | cc.opcode_byte();
            let bytes = vec![0x0F, opcode, 0, 0, 0, 0];
            Encoded {
                bytes,
                reloc: Some(PendingReloc {
                    sym: sym.clone(),
                    offset_in_instr: 2,
                    kind: PendingRelocKind::Rel32Pc,
                }),
            }
        }
        JmpRel32 { sym } => {
            // E9 cd — 5 bytes total. rel32 at offset 1.
            let bytes = vec![0xE9, 0, 0, 0, 0];
            Encoded {
                bytes,
                reloc: Some(PendingReloc {
                    sym: sym.clone(),
                    offset_in_instr: 1,
                    kind: PendingRelocKind::Rel32Pc,
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(i: Instr) -> Vec<u8> { encode_instruction(&i).bytes }

    #[test]
    fn push_pop_rbp() {
        // `pushq %rbp` -> 55,  `popq %rbp` -> 5D
        assert_eq!(enc(Instr::PushReg(Reg::Rbp)), vec![0x55]);
        assert_eq!(enc(Instr::PopReg(Reg::Rbp)), vec![0x5D]);
    }

    #[test]
    fn mov_rsp_rbp() {
        // `movq %rsp, %rbp` -> 48 89 E5
        let bytes = enc(Instr::MovRegReg { dst: Reg::Rbp, src: Reg::Rsp });
        assert_eq!(bytes, vec![0x48, 0x89, 0xE5]);
    }

    #[test]
    fn sub_rsp_32() {
        // `subq $32, %rsp` -> 48 83 EC 20
        assert_eq!(enc(Instr::SubRegImm8 { dst: Reg::Rsp, imm: 32 }),
                   vec![0x48, 0x83, 0xEC, 0x20]);
    }

    #[test]
    fn add_rsp_32() {
        // `addq $32, %rsp` -> 48 83 C4 20
        assert_eq!(enc(Instr::AddRegImm8 { dst: Reg::Rsp, imm: 32 }),
                   vec![0x48, 0x83, 0xC4, 0x20]);
    }

    #[test]
    fn xor_eax_eax() {
        // `xorl %eax, %eax` -> 31 C0
        assert_eq!(enc(Instr::XorRegReg32 { dst: Reg::Rax, src: Reg::Rax }),
                   vec![0x31, 0xC0]);
    }

    #[test]
    fn ret() {
        assert_eq!(enc(Instr::Ret), vec![0xC3]);
    }

    #[test]
    fn lea_rip_rcx() {
        // `leaq sym(%rip), %rcx` -> 48 8D 0D 00 00 00 00
        let e = encode_instruction(&Instr::LeaRipSym { dst: Reg::Rcx, sym: "sym".into() });
        assert_eq!(e.bytes, vec![0x48, 0x8D, 0x0D, 0, 0, 0, 0]);
        assert_eq!(e.reloc.as_ref().unwrap().offset_in_instr, 3);
    }

    #[test]
    fn add_sub_imul_reg_reg() {
        // `addq %r10, %rax` -> 4C 01 D0
        assert_eq!(enc(Instr::AddRegRegQ { dst: Reg::Rax, src: Reg::R10 }), vec![0x4C, 0x01, 0xD0]);
        // `subq %r10, %rax` -> 4C 29 D0
        assert_eq!(enc(Instr::SubRegRegQ { dst: Reg::Rax, src: Reg::R10 }), vec![0x4C, 0x29, 0xD0]);
        // `imulq %r10, %rax` -> 49 0F AF C2  (REX.W|R, 0F AF, ModRM dst=rax/reg, src=r10/rm)
        assert_eq!(enc(Instr::ImulRegRegQ { dst: Reg::Rax, src: Reg::R10 }), vec![0x49, 0x0F, 0xAF, 0xC2]);
    }

    #[test]
    fn xchg_rax_r10() {
        // `xchgq %r10, %rax` -> 4C 87 D0
        assert_eq!(enc(Instr::XchgRegRegQ { dst: Reg::Rax, src: Reg::R10 }), vec![0x4C, 0x87, 0xD0]);
    }

    #[test]
    fn mov_from_rbp_slot() {
        // `movq -8(%rbp), %rax` -> 48 8B 45 F8
        let bytes = enc(Instr::MovRegFromRbpDisp { dst: Reg::Rax, disp: -8 });
        assert_eq!(bytes, vec![0x48, 0x8B, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
        // Note: we always use disp32 form (mod=10) to keep encoding uniform.
    }

    #[test]
    fn mov_to_rbp_slot() {
        // `movq %rax, -8(%rbp)` -> 48 89 85 F8 FF FF FF
        let bytes = enc(Instr::MovRbpDispFromReg { src: Reg::Rax, disp: -8 });
        assert_eq!(bytes, vec![0x48, 0x89, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn cmp_test_set() {
        // `cmpq %r10, %rax` -> 4C 39 D0
        assert_eq!(enc(Instr::CmpRegRegQ { dst: Reg::Rax, src: Reg::R10 }), vec![0x4C, 0x39, 0xD0]);
        // `testq %rax, %rax` -> 48 85 C0
        assert_eq!(enc(Instr::TestRegRegQ { a: Reg::Rax, b: Reg::Rax }), vec![0x48, 0x85, 0xC0]);
        // `sete %al` -> 0F 94 C0
        assert_eq!(enc(Instr::SetccAl { cc: CondCode::E }), vec![0x0F, 0x94, 0xC0]);
        // `setne %al` -> 0F 95 C0
        assert_eq!(enc(Instr::SetccAl { cc: CondCode::Ne }), vec![0x0F, 0x95, 0xC0]);
        // `movzbl %al, %eax` -> 0F B6 C0
        assert_eq!(enc(Instr::MovzblAlEax), vec![0x0F, 0xB6, 0xC0]);
    }

    #[test]
    fn jumps_have_relocations() {
        let e = encode_instruction(&Instr::JccRel32 { cc: CondCode::E, sym: "L".into() });
        assert_eq!(e.bytes, vec![0x0F, 0x84, 0, 0, 0, 0]);
        assert_eq!(e.reloc.as_ref().unwrap().offset_in_instr, 2);
        let e = encode_instruction(&Instr::JmpRel32 { sym: "L".into() });
        assert_eq!(e.bytes, vec![0xE9, 0, 0, 0, 0]);
        assert_eq!(e.reloc.as_ref().unwrap().offset_in_instr, 1);
    }

    #[test]
    fn sse_arithmetic_encodings() {
        // `addss %xmm1, %xmm0` -> F3 0F 58 C1
        assert_eq!(enc(Instr::AddssXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF3, 0x0F, 0x58, 0xC1]);
        // `subss %xmm1, %xmm0` -> F3 0F 5C C1
        assert_eq!(enc(Instr::SubssXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF3, 0x0F, 0x5C, 0xC1]);
        // `mulss %xmm1, %xmm0` -> F3 0F 59 C1
        assert_eq!(enc(Instr::MulssXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF3, 0x0F, 0x59, 0xC1]);
        // `divss %xmm1, %xmm0` -> F3 0F 5E C1
        assert_eq!(enc(Instr::DivssXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF3, 0x0F, 0x5E, 0xC1]);
        // `ucomiss %xmm1, %xmm0` -> 0F 2E C1
        assert_eq!(enc(Instr::UcomissXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0x0F, 0x2E, 0xC1]);
        // `movss disp(%rbp), %xmm0` -> F3 0F 10 85 disp32
        assert_eq!(enc(Instr::MovssRbpDispToXmm { dst: XmmReg::Xmm0, disp: -8 }),
            vec![0xF3, 0x0F, 0x10, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
        // `movss %xmm0, disp(%rbp)` -> F3 0F 11 85 disp32
        assert_eq!(enc(Instr::MovssXmmToRbpDisp { src: XmmReg::Xmm0, disp: -8 }),
            vec![0xF3, 0x0F, 0x11, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn sse_double_encodings() {
        // `addsd %xmm1, %xmm0` -> F2 0F 58 C1
        assert_eq!(enc(Instr::AddsdXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF2, 0x0F, 0x58, 0xC1]);
        // `subsd %xmm1, %xmm0` -> F2 0F 5C C1
        assert_eq!(enc(Instr::SubsdXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF2, 0x0F, 0x5C, 0xC1]);
        // `mulsd %xmm1, %xmm0` -> F2 0F 59 C1
        assert_eq!(enc(Instr::MulsdXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF2, 0x0F, 0x59, 0xC1]);
        // `divsd %xmm1, %xmm0` -> F2 0F 5E C1
        assert_eq!(enc(Instr::DivsdXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF2, 0x0F, 0x5E, 0xC1]);
        // `ucomisd %xmm1, %xmm0` -> 66 0F 2E C1
        assert_eq!(enc(Instr::UcomisdXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0x66, 0x0F, 0x2E, 0xC1]);
        // `movsd disp(%rbp), %xmm0` -> F2 0F 10 85 disp32
        assert_eq!(enc(Instr::MovsdRbpDispToXmm { dst: XmmReg::Xmm0, disp: -8 }),
            vec![0xF2, 0x0F, 0x10, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
        // `movsd %xmm0, disp(%rbp)` -> F2 0F 11 85 disp32
        assert_eq!(enc(Instr::MovsdXmmToRbpDisp { src: XmmReg::Xmm0, disp: -8 }),
            vec![0xF2, 0x0F, 0x11, 0x85, 0xF8, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn cvt_float_float_encodings() {
        // `cvtss2sd %xmm1, %xmm0` -> F3 0F 5A C1
        assert_eq!(enc(Instr::Cvtss2sdXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF3, 0x0F, 0x5A, 0xC1]);
        // `cvtsd2ss %xmm1, %xmm0` -> F2 0F 5A C1
        assert_eq!(enc(Instr::Cvtsd2ssXmmXmm { dst: XmmReg::Xmm0, src: XmmReg::Xmm1 }),
            vec![0xF2, 0x0F, 0x5A, 0xC1]);
    }

    #[test]
    fn cvt_int_float_encodings() {
        // `cvtsi2ssq %rcx, %xmm0` -> F3 48 0F 2A C1
        assert_eq!(enc(Instr::Cvtsi2ssRegToXmm { dst: XmmReg::Xmm0, src: Reg::Rcx }),
            vec![0xF3, 0x48, 0x0F, 0x2A, 0xC1]);
        // `cvtss2siq %xmm1, %rax` -> F3 48 0F 2D C1
        assert_eq!(enc(Instr::Cvtss2siXmmToReg { dst: Reg::Rax, src: XmmReg::Xmm1 }),
            vec![0xF3, 0x48, 0x0F, 0x2D, 0xC1]);
        // `cvtsi2sdq %rcx, %xmm0` -> F2 48 0F 2A C1
        assert_eq!(enc(Instr::Cvtsi2sdRegToXmm { dst: XmmReg::Xmm0, src: Reg::Rcx }),
            vec![0xF2, 0x48, 0x0F, 0x2A, 0xC1]);
        // `cvtsd2siq %xmm1, %rax` -> F2 48 0F 2D C1
        assert_eq!(enc(Instr::Cvtsd2siXmmToReg { dst: Reg::Rax, src: XmmReg::Xmm1 }),
            vec![0xF2, 0x48, 0x0F, 0x2D, 0xC1]);
    }

    #[test]
    fn sub_add_imm32() {
        // `subq $128, %rsp` -> 48 81 EC 80 00 00 00
        assert_eq!(enc(Instr::SubRegImm32 { dst: Reg::Rsp, imm: 128 }),
            vec![0x48, 0x81, 0xEC, 0x80, 0x00, 0x00, 0x00]);
        // `addq $256, %rsp` -> 48 81 C4 00 01 00 00
        assert_eq!(enc(Instr::AddRegImm32 { dst: Reg::Rsp, imm: 256 }),
            vec![0x48, 0x81, 0xC4, 0x00, 0x01, 0x00, 0x00]);
    }

    #[test]
    fn rsp_disp_stores() {
        // `movq %rax, 32(%rsp)` -> 48 89 44 24 20 00 00 00
        assert_eq!(enc(Instr::MovRspDispFromReg { src: Reg::Rax, disp: 32 }),
            vec![0x48, 0x89, 0x84, 0x24, 0x20, 0x00, 0x00, 0x00]);
        // `movss %xmm0, 32(%rsp)` -> F3 0F 11 84 24 20 00 00 00
        assert_eq!(enc(Instr::MovssXmmToRspDisp { src: XmmReg::Xmm0, disp: 32 }),
            vec![0xF3, 0x0F, 0x11, 0x84, 0x24, 0x20, 0x00, 0x00, 0x00]);
        // `movsd %xmm0, 32(%rsp)` -> F2 0F 11 84 24 20 00 00 00
        assert_eq!(enc(Instr::MovsdXmmToRspDisp { src: XmmReg::Xmm0, disp: 32 }),
            vec![0xF2, 0x0F, 0x11, 0x84, 0x24, 0x20, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn movss_rip_has_relocation() {
        let e = encode_instruction(&Instr::MovssRipSymToXmm { dst: XmmReg::Xmm0, sym: "C".into() });
        assert_eq!(e.bytes, vec![0xF3, 0x0F, 0x10, 0x05, 0, 0, 0, 0]);
        assert_eq!(e.reloc.as_ref().unwrap().offset_in_instr, 4);
    }

    #[test]
    fn callq_sym() {
        // `callq sym` -> E8 00 00 00 00
        let e = encode_instruction(&Instr::CallSym { sym: "puts".into() });
        assert_eq!(e.bytes, vec![0xE8, 0, 0, 0, 0]);
        assert_eq!(e.reloc.as_ref().unwrap().offset_in_instr, 1);
    }
}
