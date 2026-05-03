//! aether-asm — assemble GAS-syntax x86-64 .s into a Windows COFF .obj,
//! or with `--pe`, into a runnable PE32+ console .exe via the self-hosted
//! PE writer. With imports of `aether_*` symbols this links against
//! `aether_rt.dll` at load time; `ExitProcess` always comes from kernel32.
//!
//! Bootstrap implementation in Rust; Phase 5 rewrites in Aether.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use aether_asm::pe::{PeImport, ExternalCallSite};

fn safe_under_cwd(p: &Path) -> std::io::Result<PathBuf> {
    use std::io::{Error, ErrorKind};
    if p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(Error::new(ErrorKind::InvalidInput, "path may not contain `..`"));
    }
    let cwd = std::env::current_dir()?.canonicalize()?;
    let parent = p.parent().filter(|q| !q.as_os_str().is_empty()).unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let parent_canon = parent.canonicalize()?;
    if !parent_canon.starts_with(&cwd) {
        return Err(Error::new(ErrorKind::InvalidInput, "path escapes cwd"));
    }
    let name = p.file_name().ok_or_else(|| Error::new(ErrorKind::InvalidInput, "missing file name"))?;
    Ok(parent_canon.join(name))
}

/// Pick a DLL for an external symbol name. The mapping is intentionally
/// hard-coded to the small set of DLLs aetherc-emitted code is allowed to
/// touch in `--pe` mode today; any other external is a compile error
/// because we'd be guessing.
fn dll_for_symbol(name: &str) -> Result<&'static str, String> {
    if name.starts_with("aether_") { Ok("aether_rt.dll") }
    else if name == "puts" || name == "printf" || name == "fwrite" { Ok("msvcrt.dll") }
    else if name == "ExitProcess" { Ok("kernel32.dll") }
    else if name == "_set_app_type" { Ok("ucrtbase.dll") }
    else { Err(format!("--pe doesn't know which DLL exports `{name}`")) }
}

fn main() {
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut emit_pe = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" => output = Some(PathBuf::from(it.next().expect("-o needs path"))),
            "--pe" => emit_pe = true,
            "-h" | "--help" => {
                eprintln!("aether-asm <input.s> -o <output.obj|out.exe> [--pe]");
                eprintln!("  --pe : self-hosted PE32+ link. Resolves internal labels");
                eprintln!("         in-place; for external symbols, generates per-symbol");
                eprintln!("         indirect-jump thunks and writes a multi-DLL IAT.");
                eprintln!("         Recognised DLLs: kernel32.dll (ExitProcess),");
                eprintln!("         aether_rt.dll (every aether_* symbol),");
                eprintln!("         msvcrt.dll (puts/printf/fwrite).");
                std::process::exit(0);
            }
            other if !other.starts_with('-') => input = Some(PathBuf::from(other)),
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
    }
    let input = input.expect("input .s file required");
    let in_canon = input.canonicalize().expect("input not found");
    let default_ext = if emit_pe { "exe" } else { "obj" };
    let output = output.unwrap_or_else(|| input.with_extension(default_ext));
    let out_canon = safe_under_cwd(&output).expect("output path invalid");

    let src = std::fs::read_to_string(&in_canon).expect("read input");
    let mut obj = match aether_asm::parse_gas(&src) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("aether-asm: parse error: {:?}", e);
            std::process::exit(1);
        }
    };

    let bytes = if emit_pe {
        let text_rva = 0x1000u32;
        let stub_size = aether_asm::pe::ENTRY_STUB_SIZE;
        let user_code_rva = text_rva + stub_size;

        // Step 1: figure out which external symbols are actually referenced
        // in .text. Each gets its own 6-byte thunk at the tail of .text.
        // Use BTreeMap so iteration order is stable.
        let mut externals: BTreeMap<String, ()> = BTreeMap::new();
        for r in &obj.text.relocs {
            let sym = &obj.symbols[r.sym_index as usize];
            if matches!(sym.storage, aether_asm::SymbolStorage::External) && sym.section == 0 {
                externals.insert(sym.name.clone(), ());
            }
        }
        let n_thunks = externals.len() as u32;
        // ExitProcess is forced into the IAT by build_full_exe even if the
        // user didn't reference it — which means it always gets a thunk too.
        // Account for that here so the rdata RVA is right.
        let n_thunks_inc_exit = if externals.contains_key("ExitProcess") { n_thunks } else { n_thunks + 1 };
        let thunks_size = n_thunks_inc_exit * 6;
        let text_size_unaligned = stub_size + obj.text.data.len() as u32 + thunks_size;
        let rdata_rva = text_rva + aether_asm::pe::section_align_up(text_size_unaligned);

        // Step 2: resolve every internal reloc (text-internal labels, rdata
        // string entries) with the now-known layout. Externals stay
        // unresolved — we patch them ourselves below.
        let _ = obj.resolve_internal_relocs(user_code_rva, rdata_rva);

        // Step 3: classify externals into per-DLL groups.
        let mut by_dll: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for name in externals.keys() {
            let dll = match dll_for_symbol(name) {
                Ok(d) => d,
                Err(e) => { eprintln!("aether-asm: {e}"); std::process::exit(1); }
            };
            by_dll.entry(dll.to_string()).or_default().push(name.clone());
        }
        // Always make sure kernel32!ExitProcess is in the import list — the
        // entry stub depends on it. build_full_exe also enforces this, but
        // adding it here keeps the user's view of the import list honest.
        let kernel = by_dll.entry("kernel32.dll".to_string()).or_default();
        if !kernel.iter().any(|s| s == "ExitProcess") {
            kernel.insert(0, "ExitProcess".into());
        }

        // (The eager bcrypt import experiment was dead-code that didn't move
        // the needle on the cdylib AV — see the Phase-5 follow-up notes for
        // the actual fix path.)
        let imports: Vec<PeImport> = by_dll.into_iter()
            .map(|(dll, names)| PeImport { dll, names }).collect();

        // Step 4: collect call sites — one entry per external reloc that
        // remains in obj.text.relocs (resolve_internal_relocs filtered the
        // internal ones out). Each entry tells build_full_exe: "patch the
        // 4 bytes at this offset to a rel32 pointing at the thunk for X".
        let mut call_sites: Vec<ExternalCallSite> = Vec::new();
        for r in &obj.text.relocs {
            let sym = &obj.symbols[r.sym_index as usize];
            call_sites.push(ExternalCallSite {
                offset_in_user_text: r.site,
                symbol: sym.name.clone(),
            });
        }

        let user_text = std::mem::take(&mut obj.text.data);
        let rdata = std::mem::take(&mut obj.rdata.data);
        aether_asm::pe::build_full_exe(user_text, rdata, imports, &call_sites)
    } else {
        obj.write()
    };
    std::fs::write(&out_canon, &bytes).expect("write output");
    let kind = if emit_pe { "PE32+ .exe" } else { ".obj" };
    eprintln!("[aether-asm] wrote {:?} ({} bytes) — {}", out_canon, bytes.len(), kind);
}
