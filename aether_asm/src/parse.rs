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
use crate::encode::{CondCode, Instr, Reg};

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
            // Append at the end of rdata; label was already recorded.
            ob.rdata.data.extend_from_slice(s.as_bytes());
            ob.rdata.data.push(0);
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
    let s = s.split_once('#').map(|p| p.0).unwrap_or(s);
    s.split_once("//").map(|p| p.0).unwrap_or(s)
}

fn parse_str_literal(s: &str) -> Option<String> {
    let s = s.trim();
    if !s.starts_with('"') || !s.ends_with('"') { return None; }
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
        Instr::AddRegImm8 { .. } | Instr::SubRegImm8 { .. } => 4,
        Instr::AddRegRegQ { .. } | Instr::SubRegRegQ { .. } => 3,
        Instr::ImulRegRegQ { .. } => 4,
        Instr::XchgRegRegQ { .. } => 3,
        Instr::XorRegReg32 { dst, src } =>
            if dst.extension() != 0 || src.extension() != 0 { 3 } else { 2 },
        Instr::LeaRipSym { .. } => 7,
        Instr::LeaRegFromRbpDisp { .. } => 7,
        Instr::CallSym { .. } => 5,
        Instr::Ret => 1,
        Instr::CmpRegRegQ { .. } | Instr::TestRegRegQ { .. } => 3,
        Instr::SetccAl { .. } | Instr::MovzblAlEax => 3,
        Instr::JccRel32 { .. } => 6,
        Instr::JmpRel32 { .. } => 5,
        Instr::NegRegQ { .. } => 3,
        Instr::CqoSignExt => 2,
        Instr::IdivRegQ { .. } => 3,
    }).sum()
}

fn parse_reg(s: &str, line: u32) -> Result<Reg, AsmError> {
    let s = s.trim().trim_start_matches('%');
    Ok(match s {
        "rax" | "eax" => Reg::Rax,
        "rcx" | "ecx" => Reg::Rcx,
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
            //   movq %reg, %reg
            if a.trim_start().starts_with('$') {
                Instr::MovRegImm32 { dst: parse_reg(b, lineno)?, imm: parse_imm(a, lineno)? as i32 }
            } else if let Some(disp) = parse_rbp_mem(a) {
                Instr::MovRegFromRbpDisp { dst: parse_reg(b, lineno)?, disp }
            } else if let Some(disp) = parse_rbp_mem(b) {
                Instr::MovRbpDispFromReg { src: parse_reg(a, lineno)?, disp }
            } else {
                Instr::MovRegReg { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
            }
        }
        "subq"  => {
            let (a, b) = split_comma(rest);
            if a.trim_start().starts_with('$') {
                Instr::SubRegImm8 { dst: parse_reg(b, lineno)?, imm: parse_imm(a, lineno)? as i8 }
            } else {
                Instr::SubRegRegQ { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
            }
        }
        "addq"  => {
            let (a, b) = split_comma(rest);
            if a.trim_start().starts_with('$') {
                Instr::AddRegImm8 { dst: parse_reg(b, lineno)?, imm: parse_imm(a, lineno)? as i8 }
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
        "xorl"  => {
            let (a, b) = split_comma(rest);
            Instr::XorRegReg32 { dst: parse_reg(b, lineno)?, src: parse_reg(a, lineno)? }
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
        "callq" | "call" => Instr::CallSym { sym: rest.to_string() },
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
            // Only the form we emit: `movzbl %al, %eax`. Anything else fails.
            let (a, b) = split_comma(rest);
            if a.trim() != "%al" || b.trim() != "%eax" {
                return Err(syn(lineno, format!("only `movzbl %al, %eax` supported, got {a}, {b}")));
            }
            Instr::MovzblAlEax
        }
        "je"  | "jeq" => Instr::JccRel32 { cc: CondCode::E,  sym: rest.to_string() },
        "jne"        => Instr::JccRel32 { cc: CondCode::Ne, sym: rest.to_string() },
        "jl"         => Instr::JccRel32 { cc: CondCode::L,  sym: rest.to_string() },
        "jg"         => Instr::JccRel32 { cc: CondCode::G,  sym: rest.to_string() },
        "jle"        => Instr::JccRel32 { cc: CondCode::Le, sym: rest.to_string() },
        "jge"        => Instr::JccRel32 { cc: CondCode::Ge, sym: rest.to_string() },
        "jmp"        => Instr::JmpRel32 { sym: rest.to_string() },
        // Some forms emitted by aetherc — fold into known mnems.
        "movl"  if rest.starts_with('$') => {
            let (imm, reg) = split_comma(rest);
            Instr::MovRegImm32 { dst: parse_reg(reg, lineno)?, imm: parse_imm(imm, lineno)? as i32 }
        }
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
