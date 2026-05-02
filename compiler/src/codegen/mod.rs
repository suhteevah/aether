//! Codegen — Phase 0.
//!
//! Three backends:
//! * `llvm` — text-mode LLVM IR emitter (drop-in replacement target for inkwell)
//! * `c`    — minimal C stub used to actually produce a runnable binary via gcc
//! * `asm`  — direct x86-64 AT&T assembly, no C compiler in the loop. The
//!   long-term destination once `aether_asm/` and a self-hosted linker land.

pub mod llvm;
pub mod c;
pub mod asm;
