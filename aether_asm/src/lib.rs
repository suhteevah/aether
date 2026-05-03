//! aether_asm — x86-64 instruction encoder + Windows COFF (PE32+) object writer.
//!
//! Today: a hand-built encoder for the small instruction subset aetherc emits
//! for hello-world. Tomorrow: full x86-64 ISA tables. Phase 5: rewritten in
//! Aether once the language can self-host.
//!
//! The split:
//! * `encode` — instruction → byte sequence.
//! * `coff`   — symbol/relocation tables + COFF object file layout.
//! * `parse`  — tiny GAS-syntax parser for the subset aetherc emits.

pub mod encode;
pub mod coff;
pub mod parse;
pub mod pe;

pub use encode::{encode_instruction, Instr, Reg};
pub use coff::{ObjectBuilder, Reloc, RelocKind, Section, Symbol, SymbolStorage};
pub use parse::parse_gas;
