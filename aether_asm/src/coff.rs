//! Windows COFF (PE32+, x86-64) object file writer.
//!
//! Produces an object file that the system linker (`ld` from MinGW or
//! `link.exe` from MSVC) consumes alongside libmsvcrt to produce a working
//! .exe. PE COFF reference: Microsoft "PE Format" spec.
//!
//! Supported today:
//! * One `.text` section with code bytes
//! * One `.rdata` section with read-only constants
//! * External symbol references (e.g. `puts` from msvcrt)
//! * `IMAGE_REL_AMD64_REL32` relocations (PC-relative 32-bit)
//!
//! Symbol storage classes:
//! * `IMAGE_SYM_CLASS_EXTERNAL` for `main` and any `puts`
//! * `IMAGE_SYM_CLASS_STATIC` for local labels (e.g. `.LC0`)

use crate::encode::{encode_instruction, Instr, PendingRelocKind};

const IMAGE_FILE_MACHINE_AMD64: u16 = 0x8664;
const IMAGE_REL_AMD64_REL32: u16 = 0x0004;
const IMAGE_SCN_CNT_CODE: u32 = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED_DATA: u32 = 0x0000_0040;
const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;
const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const IMAGE_SCN_ALIGN_16BYTES: u32 = 0x0050_0000;

const IMAGE_SYM_CLASS_EXTERNAL: u8 = 2;
const IMAGE_SYM_CLASS_STATIC: u8 = 3;

#[derive(Clone, Copy, Debug)]
pub enum SymbolStorage { External, Static }

impl SymbolStorage {
    fn class(self) -> u8 {
        match self {
            SymbolStorage::External => IMAGE_SYM_CLASS_EXTERNAL,
            SymbolStorage::Static => IMAGE_SYM_CLASS_STATIC,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RelocKind { Rel32Pc }

impl RelocKind {
    fn coff_type(self) -> u16 {
        match self { RelocKind::Rel32Pc => IMAGE_REL_AMD64_REL32 }
    }
}

#[derive(Debug)]
pub struct Symbol {
    pub name: String,
    /// Section number: 0 = undefined (external), 1 = .text, 2 = .rdata.
    pub section: i16,
    /// Offset within the section (0 for externals).
    pub value: u32,
    pub storage: SymbolStorage,
}

#[derive(Debug)]
pub struct Reloc {
    /// Offset of the rel32 site within its section.
    pub site: u32,
    /// Symbol-table index of the target.
    pub sym_index: u32,
    pub kind: RelocKind,
}

pub struct Section {
    pub name: [u8; 8],
    pub data: Vec<u8>,
    pub characteristics: u32,
    pub relocs: Vec<Reloc>,
}

pub struct ObjectBuilder {
    pub text: Section,
    pub rdata: Section,
    pub symbols: Vec<Symbol>,
}

impl ObjectBuilder {
    pub fn new() -> Self {
        let mut text_name = [0u8; 8]; text_name[..5].copy_from_slice(b".text");
        let mut rdata_name = [0u8; 8]; rdata_name[..6].copy_from_slice(b".rdata");
        Self {
            text: Section {
                name: text_name,
                data: Vec::new(),
                characteristics: IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_READ
                    | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_ALIGN_16BYTES,
                relocs: Vec::new(),
            },
            rdata: Section {
                name: rdata_name,
                data: Vec::new(),
                characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
                    | IMAGE_SCN_ALIGN_16BYTES,
                relocs: Vec::new(),
            },
            symbols: Vec::new(),
        }
    }

    /// Append a NUL-terminated string to .rdata, return its offset.
    pub fn intern_cstr(&mut self, s: &str) -> u32 {
        let off = self.rdata.data.len() as u32;
        self.rdata.data.extend_from_slice(s.as_bytes());
        self.rdata.data.push(0);
        off
    }

    pub fn add_symbol(&mut self, sym: Symbol) -> u32 {
        let i = self.symbols.len() as u32;
        self.symbols.push(sym);
        i
    }

    /// Resolve every Rel32Pc relocation that targets an internal symbol
    /// (one defined in this object's .text or .rdata) by writing the rel32
    /// disp at the relocation site. After this, the bytes can run as a flat
    /// blob without a linker — provided no `External` references remain.
    /// Returns `Err` listing any externals it had to skip.
    ///
    /// Internal layout assumption: .text begins at virtual address `text_rva`
    /// and .rdata begins at `rdata_rva` (caller-supplied PE-image RVAs).
    pub fn resolve_internal_relocs(&mut self, text_rva: u32, rdata_rva: u32)
        -> Result<(), Vec<String>>
    {
        let mut externals = Vec::new();
        // Take ownership of relocs so we can mutate .text.data while iterating.
        let relocs = std::mem::take(&mut self.text.relocs);
        let kept: Vec<Reloc> = relocs.into_iter().filter_map(|r| {
            let sym = &self.symbols[r.sym_index as usize];
            // External targets can't be resolved here; surface them.
            if matches!(sym.storage, SymbolStorage::External) && sym.section == 0 {
                externals.push(sym.name.clone());
                return Some(r);
            }
            // Compute target VA. section==1 → .text, section==2 → .rdata.
            let target_va = match sym.section {
                1 => text_rva + sym.value,
                2 => rdata_rva + sym.value,
                _ => { externals.push(sym.name.clone()); return Some(r); }
            };
            // Site VA is text_rva + r.site. The rel32 is from the byte right
            // after the disp32, i.e. site+4.
            let site_va = text_rva + r.site;
            let rip_after = site_va + 4;
            let rel32: i32 = (target_va as i64 - rip_after as i64) as i32;
            let bytes = rel32.to_le_bytes();
            let off = r.site as usize;
            self.text.data[off..off + 4].copy_from_slice(&bytes);
            None
        }).collect();
        self.text.relocs = kept;
        if externals.is_empty() { Ok(()) } else { Err(externals) }
    }

    /// Encode `instrs` into .text and resolve every pending symbol reloc.
    pub fn assemble_text(&mut self, instrs: &[Instr]) -> Result<(), String> {
        for i in instrs {
            let enc = encode_instruction(i);
            let base = self.text.data.len() as u32;
            self.text.data.extend_from_slice(&enc.bytes);
            if let Some(pr) = enc.reloc {
                let sym_index = self.symbols.iter().position(|s| s.name == pr.sym)
                    .ok_or_else(|| format!("unknown symbol: {}", pr.sym))? as u32;
                let kind = match pr.kind { PendingRelocKind::Rel32Pc => RelocKind::Rel32Pc };
                self.text.relocs.push(Reloc {
                    site: base + pr.offset_in_instr as u32,
                    sym_index,
                    kind,
                });
            }
        }
        Ok(())
    }

    pub fn write(&self) -> Vec<u8> {
        // Layout:
        //   [file header]
        //   [section table]   (2 entries)
        //   [section data .text]
        //   [section data .rdata]
        //   [.text relocs]
        //   [.rdata relocs] (none)
        //   [symbol table]
        //   [string table]
        const FILE_HEADER_SIZE: u32 = 20;
        const SECTION_HEADER_SIZE: u32 = 40;
        const SYMBOL_SIZE: u32 = 18;
        const RELOC_SIZE: u32 = 10;

        let n_sections = 2u16;
        let mut cursor = FILE_HEADER_SIZE + SECTION_HEADER_SIZE * n_sections as u32;

        let text_data_off = cursor;
        cursor += self.text.data.len() as u32;
        let rdata_data_off = cursor;
        cursor += self.rdata.data.len() as u32;
        let text_relocs_off = if self.text.relocs.is_empty() { 0 } else {
            let o = cursor; cursor += self.text.relocs.len() as u32 * RELOC_SIZE; o
        };
        let rdata_relocs_off = if self.rdata.relocs.is_empty() { 0 } else {
            let o = cursor; cursor += self.rdata.relocs.len() as u32 * RELOC_SIZE; o
        };
        let sym_table_off = cursor;
        let n_symbols = self.symbols.len() as u32;

        let mut buf = Vec::new();
        let mut put_u16 = |b: &mut Vec<u8>, v: u16| b.extend_from_slice(&v.to_le_bytes());
        let mut put_u32 = |b: &mut Vec<u8>, v: u32| b.extend_from_slice(&v.to_le_bytes());
        let mut put_i16 = |b: &mut Vec<u8>, v: i16| b.extend_from_slice(&v.to_le_bytes());

        // ---- file header ----
        put_u16(&mut buf, IMAGE_FILE_MACHINE_AMD64);
        put_u16(&mut buf, n_sections);
        put_u32(&mut buf, 0);                     // TimeDateStamp
        put_u32(&mut buf, sym_table_off);         // PointerToSymbolTable
        put_u32(&mut buf, n_symbols);             // NumberOfSymbols
        put_u16(&mut buf, 0);                     // SizeOfOptionalHeader (0 for object)
        put_u16(&mut buf, 0);                     // Characteristics

        // ---- section headers ----
        let write_section = |buf: &mut Vec<u8>, sec: &Section, raw_off: u32, reloc_off: u32| {
            buf.extend_from_slice(&sec.name);
            buf.extend_from_slice(&0u32.to_le_bytes());                      // VirtualSize
            buf.extend_from_slice(&0u32.to_le_bytes());                      // VirtualAddress
            buf.extend_from_slice(&(sec.data.len() as u32).to_le_bytes());   // SizeOfRawData
            buf.extend_from_slice(&raw_off.to_le_bytes());                   // PointerToRawData
            buf.extend_from_slice(&reloc_off.to_le_bytes());                 // PointerToRelocations
            buf.extend_from_slice(&0u32.to_le_bytes());                      // PointerToLinenumbers
            buf.extend_from_slice(&(sec.relocs.len() as u16).to_le_bytes()); // NumberOfRelocations
            buf.extend_from_slice(&0u16.to_le_bytes());                      // NumberOfLinenumbers
            buf.extend_from_slice(&sec.characteristics.to_le_bytes());       // Characteristics
        };
        write_section(&mut buf, &self.text, text_data_off, text_relocs_off);
        write_section(&mut buf, &self.rdata, rdata_data_off, rdata_relocs_off);

        // ---- raw section data ----
        buf.extend_from_slice(&self.text.data);
        buf.extend_from_slice(&self.rdata.data);

        // ---- relocations ----
        for r in &self.text.relocs {
            put_u32(&mut buf, r.site);
            put_u32(&mut buf, r.sym_index);
            put_u16(&mut buf, r.kind.coff_type());
        }

        // ---- symbol table ----
        // COFF stores names ≤ 8 bytes inline; longer names live in the string
        // table at offset 4 (after the 4-byte length prefix).
        let mut string_table: Vec<u8> = vec![0, 0, 0, 0]; // length placeholder
        for sym in &self.symbols {
            let mut name_buf = [0u8; 8];
            let bytes = sym.name.as_bytes();
            if bytes.len() <= 8 {
                name_buf[..bytes.len()].copy_from_slice(bytes);
            } else {
                let off = string_table.len() as u32;
                string_table.extend_from_slice(bytes);
                string_table.push(0);
                // First 4 bytes = 0, next 4 = string-table offset.
                name_buf[..4].copy_from_slice(&0u32.to_le_bytes());
                name_buf[4..].copy_from_slice(&off.to_le_bytes());
            }
            buf.extend_from_slice(&name_buf);
            put_u32(&mut buf, sym.value);
            put_i16(&mut buf, sym.section);
            put_u16(&mut buf, 0);                  // Type
            buf.push(sym.storage.class());
            buf.push(0);                           // NumberOfAuxSymbols
        }

        // ---- string table ----
        let st_len = string_table.len() as u32;
        string_table[..4].copy_from_slice(&st_len.to_le_bytes());
        buf.extend_from_slice(&string_table);

        // Suppress the "unused variable" warning when there are no rdata relocs.
        let _ = rdata_relocs_off;
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{Instr, Reg};

    #[test]
    fn hello_world_object_layout() {
        let mut o = ObjectBuilder::new();
        let str_off = o.intern_cstr("Hello");

        // Symbols: main (section 1), .LC0 (section 2), puts (external)
        o.add_symbol(Symbol { name: "main".into(),  section: 1, value: 0, storage: SymbolStorage::External });
        o.add_symbol(Symbol { name: ".LC0".into(), section: 2, value: str_off, storage: SymbolStorage::Static });
        o.add_symbol(Symbol { name: "puts".into(), section: 0, value: 0, storage: SymbolStorage::External });

        o.assemble_text(&[
            Instr::PushReg(Reg::Rbp),
            Instr::MovRegReg { dst: Reg::Rbp, src: Reg::Rsp },
            Instr::SubRegImm8 { dst: Reg::Rsp, imm: 32 },
            Instr::LeaRipSym { dst: Reg::Rcx, sym: ".LC0".into() },
            Instr::CallSym { sym: "puts".into() },
            Instr::XorRegReg32 { dst: Reg::Rax, src: Reg::Rax },
            Instr::AddRegImm8 { dst: Reg::Rsp, imm: 32 },
            Instr::PopReg(Reg::Rbp),
            Instr::Ret,
        ]).unwrap();

        let bytes = o.write();
        // Smoke checks: starts with machine=0x8664, has 2 sections.
        assert_eq!(&bytes[0..2], &[0x64, 0x86]);
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 2);
        // Two relocations on .text: LEA + CALL
        assert_eq!(o.text.relocs.len(), 2);
    }
}
