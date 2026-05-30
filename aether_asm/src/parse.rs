//! Tiny GAS-syntax parser for the subset aetherc emits.
//!
//! Recognised forms:
//!   `# comment` / `// comment`    — ignored
//!   `.section .text`              — switch to text
//!   `.section .rdata,"dr"`        — switch to rdata
//!   `.globl name`                 — mark symbol external
//!   `.asciz "..."`                — emit a NUL-terminated string in current section
//!   `LABEL:`                      — define a label
//!   `pushq %REG` / `popq %REG`
//!   `movq %REG, %REG`
//!   `subq $IMM, %REG` / `addq $IMM, %REG`
//!   `xorl %REG, %REG`
//!   `leaq SYM(%rip), %REG`
//!   `callq SYM`
//!   `ret`
//!
//! Anything else is rejected with a line number. This deliberately rejects
//! features aetherc doesn't yet emit, so the assembler stays small and any
//! drift gets caught immediately.

use crate::coff::{ObjectBuilder, Symbol, SymbolStorage};
use crate::encode::{CondCode, Instr, Reg, XmmReg};

#[derive(Debug)]
pub enum AsmError {
    Syntax { line: u32, msg: String },
}

pub fn parse_gas(src: &str) -> Result<ObjectBuilder, AsmError> {
    let mut ob = ObjectBuilder::new();
    let mut current = SectionId::Text;
    let mut text_instrs: Vec<Instr> = Vec::new();
    let mut text_labels: Vec<(String, u32)> = Vec::new();
    let mut rdata_labels: Vec<(String, u32)> = Vec::new();
    let mut globals: Vec<String> = Vec::new();

    for (lineno_zero, raw) in src.lines().enumerate() {
        let lineno = lineno_zero as u32 + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() { continue; }

        // Directives
        if let Some(rest) = line.strip_prefix(".section") {
            let r = rest.trim();
            current = if r.starts_with(".text") { SectionId::Text }
                      else if r.starts_with(".rdata") { SectionId::Rdata }
                      else { return Err(syn(lineno, format!("unknown section: {r}"))); };
            continue;
        }
        if let Some(rest) = line.strip_prefix(".globl") {
            globals.push(rest.trim().to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix(".asciz") {
            let s = parse_str_literal(rest.trim())
                .ok_or_else(|| syn(lineno, format!("bad .asciz: {rest}")))?;
            if !matches!(current, SectionId::Rdata) {
                return Err(syn(lineno, ".asciz outside .rdata".into()));
            }
            ob.rdata.data.extend_from_slice(s.as_bytes());
            ob.rdata.data.push(0);
            continue;
        }
        if let Some(rest) = line.strip_prefix(".quad") {
            if !matches!(current, SectionId::Rdata) {
                return Err(syn(lineno, ".quad outside .rdata".into()));
            }
            let v = rest.trim();
            let n: u64 = if let Some(hex) = v.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).map_err(|e| syn(lineno, format!("bad .quad: {e}")))?
            } else {
                v.parse().map_err(|e| syn(lineno, format!("bad .quad: {e}")))?
            };
            ob.rdata.data.extend_from_slice(&n.to_le_bytes());
            continue;
        }
        if let Some(rest) = line.strip_prefix(".byte") {
            if !matches!(current, SectionId::Rdata) {
                return Err(syn(lineno, ".byte outside .rdata".into()));
            }
            let v = rest.trim();
            let n: u64 = if let Some(hex) = v.strip_prefix("0x") {
                u64::from_str_radix(hex, 16).map_err(|e| syn(lineno, format!("bad .byte: {e}")))?
            } else {
                v.parse().map_err(|e| syn(lineno, format!("bad .byte: {e}")))?
            };
            ob.rdata.data.push(n as u8);
            continue;
        }
        // Label: `name:`
        if let Some(name) = line.strip_suffix(':') {
            let name = name.trim().to_string();
            match current {
                SectionId::Text => {
                    let off = synthetic_text_size(&text_instrs);
                    text_labels.push((name, off));
                }
                SectionId::Rdata => {
                    rdata_labels.push((name, ob.rdata.data.len() as u32));
                }
            }
            continue;
        }
        // Instruction
        let instr = parse_instr(line, lineno)?;
        text_instrs.push(instr);
    }

    // Materialize symbol table:
    //   index 0..N = labels in order encountered, but we need each symbol
    //   before it's referenced by a relocation. Collect all label names and
    //   the externals (anything in text_instrs that calls or LEAs a symbol
    //   we haven't defined locally) up front.
    let mut local_names: Vec<String> = Vec::new();
    for (n, _) in &text_labels { local_names.push(n.clone()); }
    for (n, _) in &rdata_labels { local_names.push(n.clone()); }

    let mut external_names: Vec<String> = Vec::new();
    for i in &text_instrs {
        let s = match i {
            Instr::LeaRipSym { sym, .. } => Some(sym),
            Instr::CallSym { sym } => Some(sym),
            _ => None,
        };
        if let Some(sym) = s {
            if !local_names.contains(sym) && !external_names.contains(sym) {
                external_names.push(sym.clone());
            }
        }
    }

    // Add label symbols (text first, then rdata) to the COFF builder.
    for (name, off) in &text_labels {
        let storage = if globals.iter().any(|g| g == name) {
            SymbolStorage::External
        } else { SymbolStorage::Static };
        ob.add_symbol(Symbol {
            name: name.clone(), section: 1, value: *off, storage,
        });
    }
    for (name, off) in &rdata_labels {
        ob.add_symbol(Symbol {
            name: name.clone(), section: 2, value: *off, storage: SymbolStorage::Static,
        });
    }
    for ext in &external_names {
        ob.add_symbol(Symbol {
            name: ext.clone(), section: 0, value: 0, storage: SymbolStorage::External,
        });
    }

    ob.assemble_text(&text_instrs).map_err(|m| syn(0, m))?;
    Ok(ob)
}

#[derive(Clone, Copy)]
enum SectionId { Text, Rdata }

fn syn(line: u32, msg: String) -> AsmError { AsmError::Syntax { line, msg } }

fn strip_comment(s: &str) -> &str {
    // String-literal-aware: `#` and `//` inside `"..."` (e.g.
    // `.asciz "# header\n"`) MUST NOT be treated as comment starts.
    let bytes = s.as_bytes();
    let mut in_str = false;
    let mut esc = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if in_str {
            if esc { esc = false; }
            else if c == b'\\' { esc = true; }
            else if c == b'"' { in_str = false; }
        } else {
            if c == b'"' { in_str = true; }
            else if c == b'#' { return &s[..i]; }
            else if c == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                return &s[..i];
            }
        }
        i += 1;
    }
    s
}

fn parse_str_literal(s: &str) -> Option<String> {
    let s = s.trim();
    if s.len() < 2 || !s.starts_with('"') || !s.ends_with('"') { return None; }
    let inner = &s[1..s.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                _ => return None,
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// Estimate byte size of an instruction *without* materialising bytes,
/// for resolving label offsets while we accumulate the instruction list.
/// Must agree with `encode_instruction(...)` byte counts exactly.
fn synthetic_text_size(instrs: &[Instr]) -> u32 {
    instrs.iter().map(|i| match i {
        Instr::PushReg(r) | Instr::PopReg(r) => if r.extension() != 0 { 2 } else { 1 },
        Instr::MovRegReg { .. } => 3,
        Instr::MovRegImm32 { .. } => 7,
        Instr::MovRegFromRbpDisp { .. } => 7,
        Instr::MovRbpDispFromReg { .. } => 7,
        Instr::MovRbpDispImm32 { .. } => 11, // REX + C7 + ModRM + disp32 + imm32

        Instr::MovRegFromBaseDisp { .. } => 7, // REX + 8B + ModRM + disp32
        Instr::MovBaseDispFromReg { .. } => 7, // REX + 89 + ModRM + disp32
        // Sized slice-element loads (P16.19). REX byte present only when an
        // r8..r15 dst or base is in play; size mirrors that exactly.
        Instr::MovzblBaseDispToReg { dst, base, disp: _ } => // [REX] 0F B6 ModRM disp32
            if dst.extension() != 0 || base.extension() != 0 { 8 } else { 7 },
        Instr::MovzwlBaseDispToReg { dst, base, disp: _ } => // [REX] 0F B7 ModRM disp32
            if dst.extension() != 0 || base.extension() != 0 { 8 } else { 7 },
        Instr::MovlBaseDispToReg { dst, base, disp: _ } => // [REX] 8B ModRM disp32
            if dst.extension() != 0 || base.extension() != 0 { 7 } else { 6 },
        Instr::AddRegImm8 { .. } | Instr::SubRegImm8 { .. } => 4,
        Instr::AddRegImm32 { .. } | Instr::SubRegImm32 { .. } => 7,
        Instr::AddRegRegQ { .. } | Instr::SubRegRegQ { .. } => 3,
        Instr::ImulRegRegQ { .. } => 4,
        Instr::XchgRegRegQ { .. } => 3,
        Instr::XorRegReg32 { dst, src } =>
            if dst.extension() != 0 || src.extension() != 0 { 3 } else { 2 },
        Instr::AndRegRegQ { .. } | Instr::OrRegRegQ { .. } | Instr::XorRegRegQ { .. } => 3,
        Instr::ShlRegByCl { .. } | Instr::SarRegByCl { .. } => 3,
        Instr::LeaRipSym { .. } => 7,
        Instr::LeaRegFromRbpDisp { .. } => 7,
        Instr::CallSym { .. } => 5,
        Instr::CallRegIndirect { reg } => if reg.extension() != 0 { 3 } else { 2 },
        Instr::Ret => 1,
        Instr::CmpRegRegQ { .. } | Instr::TestRegRegQ { .. } => 3,
        Instr::SetccAl { .. } | Instr::MovzblAlEax => 3,
        Instr::JccRel32 { .. } => 6,
        Instr::JmpRel32 { .. } => 5,
        Instr::NegRegQ { .. } => 3,
        Instr::CqoSignExt => 2,
        Instr::IdivRegQ { .. } => 3,
        Instr::MovssRbpDispToXmm { .. } | Instr::MovssXmmToRbpDisp { .. } => 8,
        Instr::MovssRipSymToXmm { .. } => 8,
        Instr::MovssXmmXmm { .. } => 4,
        Instr::AddssXmmXmm { .. } | Instr::SubssXmmXmm { .. }
            | Instr::MulssXmmXmm { .. } | Instr::DivssXmmXmm { .. } => 4,
        Instr::UcomissXmmXmm { .. } => 3,
        Instr::MovssRspToXmm { .. } | Instr::MovssXmmToRsp { .. } => 5,
        // F3 [REX] 0F 10 ModRM disp32 (P16.19 `&[f32]` load). REX only for r8+.
        Instr::MovssBaseDispToXmm { base, .. } =>
            if base.extension() != 0 { 9 } else { 8 },
        Instr::MovsdRbpDispToXmm { .. } | Instr::MovsdXmmToRbpDisp { .. } => 8,
        Instr::MovsdRipSymToXmm { .. } => 8,
        Instr::MovsdXmmXmm { .. } => 4,
        Instr::AddsdXmmXmm { .. } | Instr::SubsdXmmXmm { .. }
            | Instr::MulsdXmmXmm { .. } | Instr::DivsdXmmXmm { .. } => 4,
        Instr::UcomisdXmmXmm { .. } => 4,
        Instr::MovsdRspToXmm { .. } | Instr::MovsdXmmToRsp { .. } => 5,
        // F2 [REX] 0F 10 ModRM disp32 (P16.19 `&[f64]` load). REX only for r8+.
        Instr::MovsdBaseDispToXmm { base, .. } =>
            if base.extension() != 0 { 9 } else { 8 },
        Instr::Cvtsi2ssRegToXmm { .. } | Instr::Cvtss2siXmmToReg { .. } => 5,
        Instr::Cvtsi2sdRegToXmm { .. } | Instr::Cvtsd2siXmmToReg { .. } => 5,
        Instr::Cvtss2sdXmmXmm { .. } | Instr::Cvtsd2ssXmmXmm { .. } => 4,
        Instr::MovRspDispFromReg { .. } | Instr::MovRegFromRspDisp { .. } => 8,
        Instr::MovssXmmToRspDisp { .. } | Instr::MovsdXmmToRspDisp { .. } => 9,
        Instr::MovssRspDispToXmm { .. } | Instr::MovsdRspDispToXmm { .. } => 9,
        // AVX2 — FR-15.3. Sizes verified against `encode_instruction` bytes.
        Instr::VxorpsYmmYmmYmm { .. } => 4,   // C5 + vex1 + 57 + ModRM
        Instr::VaddpsYmmYmmYmm { .. } => 4,   // C5 + vex1 + 58 + ModRM
        Instr::VmulpsYmmYmmYmm { .. } => 4,   // C5 + vex1 + 59 + ModRM
        Instr::VmovupsMemToYmm { .. } => 8,   // C5 + vex1 + 10 + ModRM + disp32
        Instr::VmovupsYmmToMem { .. } => 8,   // C5 + vex1 + 11 + ModRM + disp32
        Instr::VmovupsYmmToRspNoDisp { .. } => 5,  // C5 FC 11 ModRM SIB
        Instr::Vzeroupper            => 3,   // C5 F8 77
    }).sum()
}

fn parse_reg(s: &str, line: u32) -> Result<Reg, AsmError> {
    let s = s.trim().trim_start_matches('%');
    Ok(match s {
        "rax" | "eax" => Reg::Rax,
        "rcx" | "ecx" | "cl" => Reg::Rcx,
        "rdx" | "edx" => Reg::Rdx,
        "rbx" | "ebx" => Reg::Rbx,
        "rsp" | "esp" => Reg::Rsp,
        "rbp" | "ebp" => Reg::Rbp,
        "rsi" | "esi" => Reg::Rsi,
        "rdi" | "edi" => Reg::Rdi,
        "r8"  | "r8d"  => Reg::R8,
        "r9"  | "r9d"  => Reg::R9,
        "r10" | "r10d" => Reg::R10,
        "r11" | "r11d" => Reg::R11,
        "r12" | "r12d" => Reg::R12,
        "r13" | "r13d" => Reg::R13,
        "r14" | "r14d" => Reg::R14,
        "r15" | "r15d" => Reg::R15,
        _ => return Err(syn(line, format!("unknown register: %{s}"))),
    })
}

fn parse_imm(s: &str, line: u32) -> Result<i64, AsmError> {
    let s = s.trim().trim_start_matches('$');
    s.parse::<i64>().map_err(|e| syn(line, format!("bad immediate {s}: {e}")))
}

fn parse_xmm(s: &str, line: u32) -> Result<XmmReg, AsmError> {
    let s = s.trim().trim_start_matches('%');
    Ok(match s {
        "xmm0" => XmmReg::Xmm0, "xmm1" => XmmReg::Xmm1,
        "xmm2" => XmmReg::Xmm2, "xmm3" => XmmReg::Xmm3,
        "xmm4" => XmmReg::Xmm4, "xmm5" => XmmReg::Xmm5,
        "xmm6" => XmmReg::Xmm6, "xmm7" => XmmReg::Xmm7,
        _ => return Err(syn(line, format!("unknown xmm register: %{s}"))),
    })
}

/// Parse a 256-bit ymm register (`%ymm0` .. `%ymm7`). FR-15.3 — only
/// ymm0..7 supported until aether_asm gains 3-byte VEX (C4) encoding
/// for the upper bank.
fn parse_ymm(s: &str, line: u32) -> Result<crate::encode::YmmReg, AsmError> {
    use crate::encode::YmmReg;
    let s = s.trim().trim_start_matches('%');
    Ok(match s {
        "ymm0" => YmmReg::Ymm0, "ymm1" => YmmReg::Ymm1,
        "ymm2" => YmmReg::Ymm2, "ymm3" => YmmReg::Ymm3,
        "ymm4" => YmmReg::Ymm4, "ymm5" => YmmReg::Ymm5,
        "ymm6" => YmmReg::Ymm6, "ymm7" => YmmReg::Ymm7,
        _ => return Err(syn(line, format!("unknown ymm register: %{s}"))),
    })
}

/// Parse `SYM(%rip)` form. Returns the symbol name on a match.
fn parse_rip_mem(s: &str) -> Option<String> {
    let s = s.trim();
    let open = s.find('(')?;
    let close = s.rfind(')')?;
    if close <= open { return None; }
    if s[open + 1..close].trim() != "%rip" { return None; }
    Some(s[..open].trim().to_string())
}

/// Parse a memory operand of the form `disp(%rbp)` or `-disp(%rbp)` or
/// `(%rbp)`. Returns the signed displacement on a match. AT&T syntax.
fn parse_rbp_mem(s: &str) -> Option<i32> {
    let s = s.trim();
    let open = s.find('(')?;
    let close = s.rfind(')')?;
    if close <= open { return None; }
    let inside = &s[open + 1..close];
    if inside.trim() != "%rbp" { return None; }
    let disp_str = s[..open].trim();
    let disp: i32 = if disp_str.is_empty() { 0 } else { disp_str.parse().ok()? };
    Some(disp)
}

/// Parse a memory operand `disp(%rsp)`. Returns the displacement.
fn parse_rsp_mem(s: &str) -> Option<i32> {
    let s = s.trim();
    let open = s.find('(')?;
    let close = s.rfind(')')?;
    if close <= open { return None; }
    let inside = &s[open + 1..close];
    if inside.trim() != "%rsp" { return None; }
    let disp_str = s[..open].trim();
    let disp: i32 = if disp_str.is_empty() { 0 } else { disp_str.parse().ok()? };
    Some(disp)
}

/// Parse `disp(%rXX)` for any 64-bit base reg EXCEPT rbp/rsp (those have
/// dedicated parsers + dedicated Instr variants because their encodings
/// require special-cased ModRM/SIB bytes). Returns `(disp, base_reg)`.
fn parse_base_mem(s: &str, lineno: u32) -> Option<(i32, Reg)> {
    let s = s.trim();
    let open = s.find('(')?;
    let close = s.rfind(')')?;
    if close <= open { return None; }
    let inside = s[open + 1..close].trim();
    if !inside.starts_with('%') { return None; }
    let base = parse_reg(inside, lineno).ok()?;
    if matches!(base, Reg::Rbp | Reg::Rsp) { return None; }
    let disp_str = s[..open].trim();
    let disp: i32 = if disp_str.is_empty() { 0 } else { disp_str.parse().ok()? };
    Some((disp, base))
}

fn parse_instr(line: &str, lineno: u32) -> Result<Instr, AsmError> {
    let mut parts = line.splitn(2, char::is_whitespace);
    let mnem = parts.next().unwrap();
    let rest = parts.next().unwrap_or("").trim();

    fn split_comma(s: &str) -> (&str, &str) {
        match s.split_once(',') { Some((a, b)) => (a.trim(), b.trim()), None => (s, "") }
    }

    Ok(match mnem {
        "ret"   => Instr::Ret,
        "retq"  => Instr::Ret,
        "pushq" => Instr::PushReg(parse_reg(rest, lineno)?),
        "popq"  => Instr::PopReg(parse_reg(rest, lineno)?),
        "movq"  => {
            let (a, b) = split_comma(rest);
            // Forms (left-to-right operand order, AT&T):
            //   movq $imm, %reg
            //   movq disp(%rbp), %reg     (load from rbp slot)
            //   movq %reg, disp(%rbp)     (store to rbp slot)
            //   movq %reg, disp(%rsp)     (store to outgoing arg slot)
            //   movq %reg, %reg
            if a.trim_start().starts_with('$') {
                // movq $imm, disp(%rbp) — peephole-emitted direct imm-to-mem store.
                if let Some(disp) = parse_rbp_mem(b) {
                    Instr::MovRbpDispImm32 { disp, imm: parse_imm(a, lineno)? as i32 }
                } else {
                    Instr::MovRegImm32 { dst: parse_reg(b, lineno)?, imm: parse_imm(a, lineno)? as i32 }
                }
            } else if let Some(disp) = parse_rbp_mem(a) {
                Instr::MovRegFromRbpDisp { dst: parse_reg(b, lineno)?, disp }
            } else if let Some(disp) = parse_rbp_mem(b) {
                Instr::MovRbpDispFromReg { src: parse_reg(a, lineno)?, disp }
            } else if let Some(disp) = parse_rsp_mem(b) {
                Instr::MovRspDispFromReg { src: parse_reg(a, lineno)?, disp }
            } else if let Some(disp) = parse_rsp_mem(a) {
                Instr::MovRegFromRspDisp { dst: parse_reg(b, lineno)?, disp }
            } else if let Some((disp, base)) = parse_base_mem(b, lineno) {
                // movq %src, disp(%base) — generic base reg form.
                Instr::MovBaseDispFromReg { src: parse_reg(a, lineno)?, base, disp }
            } else if let Some((disp, base)) = parse_base_mem(a, lineno) {
                // movq disp(%base), %dst — generic base reg form.
                Instr::MovRegFromBaseDisp { dst: parse_reg(b, lineno)?, base, disp }
            } else {
                Instr::MovRegReg { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
            }
        }
        "subq"  => {
            let (a, b) = split_comma(rest);
            if a.trim_start().starts_with('$') {
                let imm = parse_imm(a, lineno)?;
                let dst = parse_reg(b, lineno)?;
                if (-128..=127).contains(&imm) {
                    Instr::SubRegImm8 { dst, imm: imm as i8 }
                } else {
                    Instr::SubRegImm32 { dst, imm: imm as i32 }
                }
            } else {
                Instr::SubRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
            }
        }
        "addq"  => {
            let (a, b) = split_comma(rest);
            if a.trim_start().starts_with('$') {
                let imm = parse_imm(a, lineno)?;
                let dst = parse_reg(b, lineno)?;
                if (-128..=127).contains(&imm) {
                    Instr::AddRegImm8 { dst, imm: imm as i8 }
                } else {
                    Instr::AddRegImm32 { dst, imm: imm as i32 }
                }
            } else {
                Instr::AddRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
            }
        }
        "imulq" => {
            let (a, b) = split_comma(rest);
            // AT&T `imulq %src, %dst` — dst is second operand.
            Instr::ImulRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "xchgq" => {
            let (a, b) = split_comma(rest);
            Instr::XchgRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "negq"  => Instr::NegRegQ { dst: parse_reg(rest, lineno)? },
        "cqo"   => Instr::CqoSignExt,
        "cqto"  => Instr::CqoSignExt,
        "idivq" => Instr::IdivRegQ { src: parse_reg(rest, lineno)? },

        // SSE2 single-precision float instructions.
        "movss" => {
            let (a, b) = split_comma(rest);
            if let Some(disp) = parse_rbp_mem(a) {
                Instr::MovssRbpDispToXmm { dst: parse_xmm(b, lineno)?, disp }
            } else if let Some(disp) = parse_rbp_mem(b) {
                Instr::MovssXmmToRbpDisp { src: parse_xmm(a, lineno)?, disp }
            } else if let Some(sym) = parse_rip_mem(a) {
                Instr::MovssRipSymToXmm { dst: parse_xmm(b, lineno)?, sym }
            } else if a.trim() == "(%rsp)" {
                Instr::MovssRspToXmm { dst: parse_xmm(b, lineno)? }
            } else if b.trim() == "(%rsp)" {
                Instr::MovssXmmToRsp { src: parse_xmm(a, lineno)? }
            } else if let Some(disp) = parse_rsp_mem(b) {
                Instr::MovssXmmToRspDisp { src: parse_xmm(a, lineno)?, disp }
            } else if let Some(disp) = parse_rsp_mem(a) {
                Instr::MovssRspDispToXmm { dst: parse_xmm(b, lineno)?, disp }
            } else if let Some((disp, base)) = parse_base_mem(a, lineno) {
                // `movss disp(%base), %xmm` — P16.19 `&[f32]` element load.
                Instr::MovssBaseDispToXmm { dst: parse_xmm(b, lineno)?, base, disp }
            } else {
                Instr::MovssXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
            }
        }
        "addss" => {
            let (a, b) = split_comma(rest);
            Instr::AddssXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "subss" => {
            let (a, b) = split_comma(rest);
            Instr::SubssXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "mulss" => {
            let (a, b) = split_comma(rest);
            Instr::MulssXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "divss" => {
            let (a, b) = split_comma(rest);
            Instr::DivssXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "ucomiss" => {
            let (a, b) = split_comma(rest);
            Instr::UcomissXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "movsd" => {
            let (a, b) = split_comma(rest);
            if let Some(disp) = parse_rbp_mem(a) {
                Instr::MovsdRbpDispToXmm { dst: parse_xmm(b, lineno)?, disp }
            } else if let Some(disp) = parse_rbp_mem(b) {
                Instr::MovsdXmmToRbpDisp { src: parse_xmm(a, lineno)?, disp }
            } else if let Some(sym) = parse_rip_mem(a) {
                Instr::MovsdRipSymToXmm { dst: parse_xmm(b, lineno)?, sym }
            } else if a.trim() == "(%rsp)" {
                Instr::MovsdRspToXmm { dst: parse_xmm(b, lineno)? }
            } else if b.trim() == "(%rsp)" {
                Instr::MovsdXmmToRsp { src: parse_xmm(a, lineno)? }
            } else if let Some(disp) = parse_rsp_mem(b) {
                Instr::MovsdXmmToRspDisp { src: parse_xmm(a, lineno)?, disp }
            } else if let Some(disp) = parse_rsp_mem(a) {
                Instr::MovsdRspDispToXmm { dst: parse_xmm(b, lineno)?, disp }
            } else if let Some((disp, base)) = parse_base_mem(a, lineno) {
                // `movsd disp(%base), %xmm` — P16.19 `&[f64]` element load.
                Instr::MovsdBaseDispToXmm { dst: parse_xmm(b, lineno)?, base, disp }
            } else {
                Instr::MovsdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
            }
        }
        "addsd" => {
            let (a, b) = split_comma(rest);
            Instr::AddsdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "subsd" => {
            let (a, b) = split_comma(rest);
            Instr::SubsdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "mulsd" => {
            let (a, b) = split_comma(rest);
            Instr::MulsdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "divsd" => {
            let (a, b) = split_comma(rest);
            Instr::DivsdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "ucomisd" => {
            let (a, b) = split_comma(rest);
            Instr::UcomisdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        // Cvt forms — AT&T syntax: source first, dst second.
        "cvtsi2ssq" => {
            let (a, b) = split_comma(rest);
            Instr::Cvtsi2ssRegToXmm { dst: parse_xmm(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "cvtss2siq" => {
            let (a, b) = split_comma(rest);
            Instr::Cvtss2siXmmToReg { dst: parse_reg(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "cvtsi2sdq" => {
            let (a, b) = split_comma(rest);
            Instr::Cvtsi2sdRegToXmm { dst: parse_xmm(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "cvtsd2siq" => {
            let (a, b) = split_comma(rest);
            Instr::Cvtsd2siXmmToReg { dst: parse_reg(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "cvtss2sd" => {
            let (a, b) = split_comma(rest);
            Instr::Cvtss2sdXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "cvtsd2ss" => {
            let (a, b) = split_comma(rest);
            Instr::Cvtsd2ssXmmXmm { dst: parse_xmm(b, lineno)?, src: parse_xmm(a, lineno)? }
        }
        "seta"  => Instr::SetccAl { cc: CondCode::A  },
        "setb"  => Instr::SetccAl { cc: CondCode::B  },
        "setae" => Instr::SetccAl { cc: CondCode::Ae },
        "setbe" => Instr::SetccAl { cc: CondCode::Be },
        "xorl"  => {
            let (a, b) = split_comma(rest);
            Instr::XorRegReg32 { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "andq" => {
            let (a, b) = split_comma(rest);
            Instr::AndRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "orq" => {
            let (a, b) = split_comma(rest);
            Instr::OrRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "xorq" => {
            let (a, b) = split_comma(rest);
            Instr::XorRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "shlq" => {
            // `shlq %cl, %reg` — only the CL form is emitted by the compiler.
            let (a, b) = split_comma(rest);
            let _ = a; // must be %cl
            Instr::ShlRegByCl { dst: parse_reg(b, lineno)? }
        }
        "sarq" => {
            let (a, b) = split_comma(rest);
            let _ = a;
            Instr::SarRegByCl { dst: parse_reg(b, lineno)? }
        }
        "leaq"  => {
            // Two forms today:
            //   `leaq SYM(%rip), %REG`     — RIP-relative symbol load
            //   `leaq disp(%rbp), %REG`    — address of a stack slot
            let (mem, reg) = split_comma(rest);
            let mem = mem.trim();
            let dst = parse_reg(reg, lineno)?;
            if let Some(disp) = parse_rbp_mem(mem) {
                return Ok(Instr::LeaRegFromRbpDisp { dst, disp });
            }
            let open = mem.find('(').ok_or_else(|| syn(lineno, format!("leaq missing `(`: {mem}")))?;
            let sym = mem[..open].trim().to_string();
            let inner = &mem[open + 1..mem.rfind(')').unwrap_or(mem.len())];
            if inner.trim() != "%rip" {
                return Err(syn(lineno, format!("leaq base must be %rip, got {inner}")));
            }
            Instr::LeaRipSym { dst, sym }
        }
        "callq" | "call" => {
            // `callq *%reg` — indirect through reg (closures-lite); else direct.
            if rest.starts_with('*') {
                let reg_str = rest[1..].trim();
                Instr::CallRegIndirect { reg: parse_reg(reg_str, lineno)? }
            } else {
                Instr::CallSym { sym: rest.to_string() }
            }
        }
        "cmpq" => {
            let (a, b) = split_comma(rest);
            Instr::CmpRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
        }
        "testq" => {
            let (a, b) = split_comma(rest);
            Instr::TestRegRegQ { a: parse_reg(a, lineno)?, b: parse_reg(b, lineno)? }
        }
        "sete"  => Instr::SetccAl { cc: CondCode::E },
        "setne" => Instr::SetccAl { cc: CondCode::Ne },
        "setl"  => Instr::SetccAl { cc: CondCode::L  },
        "setg"  => Instr::SetccAl { cc: CondCode::G  },
        "setle" => Instr::SetccAl { cc: CondCode::Le },
        "setge" => Instr::SetccAl { cc: CondCode::Ge },
        "movzbl" => {
            let (a, b) = split_comma(rest);
            // Memory form: `movzbl disp(%base), %dst` — P16.19 `&[u8]` load.
            if let Some((disp, base)) = parse_base_mem(a, lineno) {
                Instr::MovzblBaseDispToReg { dst: parse_reg(b, lineno)?, base, disp }
            } else if a.trim() == "%al" && b.trim() == "%eax" {
                Instr::MovzblAlEax
            } else {
                return Err(syn(lineno, format!("movzbl: only `%al, %eax` or `disp(%base), %reg`, got {a}, {b}")));
            }
        }
        "movzwl" => {
            // Memory form only: `movzwl disp(%base), %dst` — P16.19 `&[u16]`.
            let (a, b) = split_comma(rest);
            if let Some((disp, base)) = parse_base_mem(a, lineno) {
                Instr::MovzwlBaseDispToReg { dst: parse_reg(b, lineno)?, base, disp }
            } else {
                return Err(syn(lineno, format!("movzwl: only `disp(%base), %reg`, got {a}, {b}")));
            }
        }
        "je"  | "jeq" => Instr::JccRel32 { cc: CondCode::E,  sym: rest.to_string() },
        "jne"        => Instr::JccRel32 { cc: CondCode::Ne, sym: rest.to_string() },
        "jl"         => Instr::JccRel32 { cc: CondCode::L,  sym: rest.to_string() },
        "jg"         => Instr::JccRel32 { cc: CondCode::G,  sym: rest.to_string() },
        "jle"        => Instr::JccRel32 { cc: CondCode::Le, sym: rest.to_string() },
        "jge"        => Instr::JccRel32 { cc: CondCode::Ge, sym: rest.to_string() },
        "jbe"        => Instr::JccRel32 { cc: CondCode::Be, sym: rest.to_string() },
        "ja"         => Instr::JccRel32 { cc: CondCode::A,  sym: rest.to_string() },
        "jb"         => Instr::JccRel32 { cc: CondCode::B,  sym: rest.to_string() },
        "jae"        => Instr::JccRel32 { cc: CondCode::Ae, sym: rest.to_string() },
        "jmp"        => Instr::JmpRel32 { sym: rest.to_string() },
        // Some forms emitted by aetherc — fold into known mnems.
        "movl"  if rest.starts_with('$') => {
            let (imm, reg) = split_comma(rest);
            Instr::MovRegImm32 { dst: parse_reg(reg, lineno)?, imm: parse_imm(imm, lineno)? as i32 }
        }
        // `movl disp(%base), %dst` — 32-bit load (P16.19 `&[u32]`/`&[i32]`).
        // Zero-extends into the full 64-bit dst.
        "movl" => {
            let (a, b) = split_comma(rest);
            if let Some((disp, base)) = parse_base_mem(a, lineno) {
                Instr::MovlBaseDispToReg { dst: parse_reg(b, lineno)?, base, disp }
            } else {
                return Err(syn(lineno, format!("movl: only `$imm, %reg` or `disp(%base), %reg`, got {a}, {b}")));
            }
        }

        // ── AVX2 (FR-15.3) ──────────────────────────────────────────────────
        // AT&T 3-operand form: `vXXXps %src2, %src1, %dst`. Split on the
        // first two commas to extract each operand.
        "vxorps" | "vaddps" | "vmulps" => {
            let parts: Vec<&str> = rest.splitn(3, ',').map(|s| s.trim()).collect();
            if parts.len() != 3 {
                return Err(syn(lineno, format!("{mnem} needs 3 ymm operands")));
            }
            let src2 = parse_ymm(parts[0], lineno)?;
            let src1 = parse_ymm(parts[1], lineno)?;
            let dst  = parse_ymm(parts[2], lineno)?;
            match mnem {
                "vxorps" => Instr::VxorpsYmmYmmYmm { dst, src1, src2 },
                "vaddps" => Instr::VaddpsYmmYmmYmm { dst, src1, src2 },
                "vmulps" => Instr::VmulpsYmmYmmYmm { dst, src1, src2 },
                _ => unreachable!(),
            }
        }
        "vmovups" => {
            let (a, b) = split_comma(rest);
            // Forms: `vmovups disp(%base), %ymm` (load),
            //        `vmovups %ymm, disp(%base)` (store), or
            //        `vmovups %ymm, (%rsp)` (no-disp store; SIB-encoded).
            if let Some((disp, base)) = parse_base_mem(a, lineno) {
                Instr::VmovupsMemToYmm { dst: parse_ymm(b, lineno)?, base, disp }
            } else if let Some((disp, base)) = parse_base_mem(b, lineno) {
                Instr::VmovupsYmmToMem { src: parse_ymm(a, lineno)?, base, disp }
            } else if b.trim() == "(%rsp)" {
                Instr::VmovupsYmmToRspNoDisp { src: parse_ymm(a, lineno)? }
            } else {
                return Err(syn(lineno, format!("vmovups: unsupported operands `{a}`, `{b}` — need disp(%base) on one side or (%rsp)")));
            }
        }
        "vzeroupper" => Instr::Vzeroupper,

        other   => return Err(syn(lineno, format!("unsupported instruction: {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hello_world() {
        let src = r#"
.section .rdata,"dr"
.LC0:
    .asciz "Hello"

.section .text
.globl main
main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    leaq .LC0(%rip), %rcx
    callq puts
    xorl %eax, %eax
    addq $32, %rsp
    popq %rbp
    ret
"#;
        let ob = parse_gas(src).unwrap();
        assert_eq!(ob.symbols.iter().filter(|s| s.name == "puts").count(), 1);
        assert_eq!(ob.symbols.iter().filter(|s| s.name == "main").count(), 1);
        assert!(ob.text.data.len() > 8);
        assert!(ob.text.relocs.len() == 2);
    }
}
