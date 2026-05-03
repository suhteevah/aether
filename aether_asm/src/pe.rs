//! Minimal PE32+ (Windows 64-bit) executable writer.
//!
//! This is the Phase-5 self-hosted-linker's first cut. It can produce a
//! standalone .exe whose only external dependency is `kernel32!ExitProcess`
//! (so the program can return an exit code). No support yet for additional
//! DLL imports or for resolving libaether_rt symbols — that lands when we
//! either (a) build libaether_rt as a DLL and add it as a second import
//! descriptor, or (b) write a static linker that pulls .obj members from
//! libaether_rt.a and merges them into our text section.
//!
//! Layout (all offsets file-relative; image RVAs computed at write time):
//!   0x0000  DOS header + stub
//!   0x0040  PE signature ("PE\0\0")
//!   0x0044  COFF File Header (20 bytes)
//!   0x0058  Optional Header (240 bytes, PE32+)
//!   0x0148  Section table (3 sections × 40 bytes = 120)
//!   0x01C0  padding to file alignment
//!   0x0200  .text  — entry stub + caller-provided code
//!   0x0400  .rdata — imported function name + DLL name
//!   0x0600  .idata — import directory + ILT + IAT
//!
//! The entry stub at the top of .text:
//!   sub rsp, 40                     ; 32-byte shadow + 8 bytes alignment
//!   call    rel32 -> user_main      ; runs the user's `main`
//!   mov  ecx, eax                   ; exit code from main → ExitProcess arg0
//!   call qword ptr [rip+IAT slot]   ; kernel32!ExitProcess(ecx)
//!   ud2                             ; should never return
//!
//! Refs:
//! - "PE Format" Microsoft Learn (windows/win32/debug/pe-format)
//! - LIEF source for cross-checking offsets
//! - cargo's bootstrap-pe-min crate for IAT assembly inspiration


const FILE_ALIGN: u32 = 0x200;
const SECTION_ALIGN: u32 = 0x1000;
const IMAGE_BASE: u64 = 0x140000000;
const HEADERS_SIZE: u32 = 0x200; // After alignment.

const IMAGE_FILE_RELOCS_STRIPPED: u16     = 0x0001;
const IMAGE_FILE_EXECUTABLE_IMAGE: u16    = 0x0002;
const IMAGE_FILE_LARGE_ADDRESS_AWARE: u16 = 0x0020;
const IMAGE_FILE_MACHINE_AMD64: u16       = 0x8664;

const IMAGE_SUBSYSTEM_WINDOWS_CUI: u16 = 3;

const IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE: u16 = 0x0040;
const IMAGE_DLLCHARACTERISTICS_NX_COMPAT: u16    = 0x0100;
const IMAGE_DLLCHARACTERISTICS_TERMINAL_SERVER_AWARE: u16 = 0x8000;

const IMAGE_SCN_CNT_CODE: u32         = 0x0000_0020;
const IMAGE_SCN_CNT_INITIALIZED: u32  = 0x0000_0040;
const IMAGE_SCN_MEM_EXECUTE: u32      = 0x2000_0000;
const IMAGE_SCN_MEM_READ: u32         = 0x4000_0000;
const IMAGE_SCN_MEM_WRITE: u32        = 0x8000_0000;

/// Entry-stub size in bytes. Computed at write time but exposed here so
/// callers know the offset where their code begins inside .text.
pub const ENTRY_STUB_SIZE: u32 = 20;

/// Build a minimal PE32+ console .exe and return its bytes. Callers (the
/// `aether-asm` binary, today) are responsible for writing the bytes to the
/// destination chosen on the command line — same split as `coff::ObjectBuilder`.
///
/// `user_code`: the user's machine code. Will be placed immediately after a
/// small entry stub. The first byte of `user_code` is treated as the user
/// `main` symbol's start; the entry stub CALLs into it.
///
/// The user's `main` is expected to follow the MS x64 ABI: it must return its
/// exit code in `eax` (matching aetherc's existing `fn main() -> i32`
/// convention). The stub copies eax → ecx and tail-calls ExitProcess.
pub fn build_minimal_exe(user_code: &[u8]) -> Vec<u8> {
    // ---------- compute layout ----------
    let text_size_unaligned = (ENTRY_STUB_SIZE as usize) + user_code.len();
    let text_size = align_up(text_size_unaligned as u32, FILE_ALIGN);

    // .rdata holds the DLL name "kernel32.dll\0" and the Hint/Name entry
    // for "ExitProcess".
    let dll_name = b"KERNEL32.dll\0";
    let func_name = b"ExitProcess\0";
    // Hint(2) + name + pad to even.
    let mut rdata = Vec::new();
    let func_hint_off = rdata.len() as u32;
    rdata.extend_from_slice(&[0, 0]); // Hint
    rdata.extend_from_slice(func_name);
    if rdata.len() % 2 != 0 { rdata.push(0); }
    let dll_name_off = rdata.len() as u32;
    rdata.extend_from_slice(dll_name);
    while rdata.len() % 2 != 0 { rdata.push(0); }
    let rdata_size = align_up(rdata.len() as u32, FILE_ALIGN);

    // .idata: one IMAGE_IMPORT_DESCRIPTOR (20 bytes) + null terminator (20)
    //   + ILT (one 8-byte entry + 8-byte null)
    //   + IAT (one 8-byte entry + 8-byte null)
    let idata_descriptors = 2 * 20; // active + null terminator
    let idata_ilt = 2 * 8;
    let idata_iat = 2 * 8;
    let idata_size_unaligned = idata_descriptors + idata_ilt + idata_iat;
    let idata_size = align_up(idata_size_unaligned as u32, FILE_ALIGN);

    // RVAs (virtual addresses relative to ImageBase).
    let text_rva   = SECTION_ALIGN;                              // 0x1000
    let rdata_rva  = align_up(text_rva  + text_size,  SECTION_ALIGN);
    let idata_rva  = align_up(rdata_rva + rdata_size, SECTION_ALIGN);
    let image_size = align_up(idata_rva + idata_size, SECTION_ALIGN);

    // File offsets.
    let text_file_off  = HEADERS_SIZE;
    let rdata_file_off = text_file_off + text_size;
    let idata_file_off = rdata_file_off + rdata_size;
    let total_file_size = idata_file_off + idata_size;

    // .idata internals.
    let descriptors_rva = idata_rva;
    let ilt_rva = descriptors_rva + idata_descriptors as u32;
    let iat_rva = ilt_rva + idata_ilt as u32;
    let func_hint_rva = rdata_rva + func_hint_off;
    let dll_name_rva  = rdata_rva + dll_name_off;

    // Entry-stub layout (20 bytes total):
    //   48 83 EC 20           ; sub rsp, 32            (4 bytes)
    //   E8 ?? ?? ?? ??        ; call rel32 user_main   (5 bytes, total=9)
    //   89 C1                 ; mov ecx, eax           (2 bytes, total=11)
    //   FF 15 ?? ?? ?? ??     ; call qword ptr [rip+d] (6 bytes, total=17)
    //   0F 0B                 ; ud2                    (2 bytes, total=19)
    //   90                    ; nop pad                (1 byte,  total=20)
    //
    // Stack-alignment note: the Windows loader enters at AddressOfEntryPoint
    // with rsp 16-byte aligned (no return address pushed — loader jumps,
    // it doesn't call). The MS x64 ABI requires rsp ≡ 0 mod 16 just before
    // any CALL instruction. `sub rsp, 32` keeps rsp ≡ 0 mod 16; `sub rsp, 40`
    // would leave it at 8 mod 16 and silently violate the ABI for any callee
    // that uses xmm-aligned spills (which Rust's startup code in ExitProcess
    // does — symptom is STATUS_ACCESS_VIOLATION).
    let entry_rva = text_rva;
    let stub_call_iat_offset_in_text = 11u32; // after sub(4) + call_user(5) + mov ecx,eax(2)
    let user_main_rva = text_rva + ENTRY_STUB_SIZE;

    let mut stub = Vec::with_capacity(20);
    stub.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]); // sub rsp, 32
    // call rel32 to user_main: rel32 = user_main_rva - (entry_rva + 4 + 5)
    let call_user_rip_end = entry_rva + 4 + 5;
    let call_user_rel: i32 = (user_main_rva as i64 - call_user_rip_end as i64) as i32;
    stub.push(0xE8);
    stub.extend_from_slice(&call_user_rel.to_le_bytes());
    stub.extend_from_slice(&[0x89, 0xC1]); // mov ecx, eax
    // call qword ptr [rip+disp32] → [iat_rva]
    let call_iat_rip_end = entry_rva + stub_call_iat_offset_in_text + 6;
    let iat_rel: i32 = (iat_rva as i64 - call_iat_rip_end as i64) as i32;
    stub.extend_from_slice(&[0xFF, 0x15]);
    stub.extend_from_slice(&iat_rel.to_le_bytes());
    stub.extend_from_slice(&[0x0F, 0x0B]); // ud2
    stub.push(0x90);                       // nop pad
    debug_assert_eq!(stub.len() as u32, ENTRY_STUB_SIZE);

    // Rebuild text now that we know stub bytes.
    let mut text = Vec::with_capacity(text_size as usize);
    text.extend_from_slice(&stub);
    text.extend_from_slice(user_code);
    while text.len() < text_size as usize { text.push(0); }

    // ---------- assemble ----------
    let mut buf = Vec::with_capacity(total_file_size as usize);

    // DOS header (64 bytes). Mostly zero except magic and e_lfanew.
    let mut dos = vec![0u8; 64];
    dos[0] = b'M'; dos[1] = b'Z';
    let pe_offset: u32 = 64;
    dos[60..64].copy_from_slice(&pe_offset.to_le_bytes());
    buf.extend_from_slice(&dos);

    // PE signature.
    buf.extend_from_slice(b"PE\0\0");

    // COFF File Header (20 bytes).
    let coff_header_off = buf.len();
    buf.extend_from_slice(&IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes());                    // NumberOfSections
    buf.extend_from_slice(&0u32.to_le_bytes());                    // TimeDateStamp
    buf.extend_from_slice(&0u32.to_le_bytes());                    // PointerToSymbolTable
    buf.extend_from_slice(&0u32.to_le_bytes());                    // NumberOfSymbols
    buf.extend_from_slice(&240u16.to_le_bytes());                  // SizeOfOptionalHeader (PE32+)
    // RELOCS_STRIPPED + no DYNAMIC_BASE = "this image is fixed at ImageBase".
    // Required because we don't write a .reloc section; without RELOCS_STRIPPED
    // the loader thinks it can rebase us, then panics when it can't find the
    // .reloc table and the inner DLL load chain partially completes.
    let characteristics = IMAGE_FILE_RELOCS_STRIPPED
        | IMAGE_FILE_EXECUTABLE_IMAGE
        | IMAGE_FILE_LARGE_ADDRESS_AWARE;
    buf.extend_from_slice(&characteristics.to_le_bytes());
    let _ = coff_header_off;

    // Optional Header (PE32+, 240 bytes).
    buf.extend_from_slice(&0x20Bu16.to_le_bytes());                // Magic = PE32+
    buf.extend_from_slice(&[14, 0]);                                // LinkerVersion 14.0
    buf.extend_from_slice(&text_size.to_le_bytes());                // SizeOfCode
    buf.extend_from_slice(&(rdata_size + idata_size).to_le_bytes());// SizeOfInitializedData
    buf.extend_from_slice(&0u32.to_le_bytes());                     // SizeOfUninitializedData
    buf.extend_from_slice(&entry_rva.to_le_bytes());                // AddressOfEntryPoint
    buf.extend_from_slice(&text_rva.to_le_bytes());                 // BaseOfCode
    buf.extend_from_slice(&IMAGE_BASE.to_le_bytes());               // ImageBase (8 bytes for PE32+)
    buf.extend_from_slice(&SECTION_ALIGN.to_le_bytes());            // SectionAlignment
    buf.extend_from_slice(&FILE_ALIGN.to_le_bytes());               // FileAlignment
    buf.extend_from_slice(&[6, 0]);                                  // MajorOperatingSystemVersion
    buf.extend_from_slice(&[0, 0]);                                  // MinorOperatingSystemVersion
    buf.extend_from_slice(&[0, 0]);                                  // MajorImageVersion
    buf.extend_from_slice(&[0, 0]);                                  // MinorImageVersion
    buf.extend_from_slice(&[6, 0]);                                  // MajorSubsystemVersion
    buf.extend_from_slice(&[0, 0]);                                  // MinorSubsystemVersion
    buf.extend_from_slice(&0u32.to_le_bytes());                      // Win32VersionValue
    buf.extend_from_slice(&image_size.to_le_bytes());                // SizeOfImage
    buf.extend_from_slice(&HEADERS_SIZE.to_le_bytes());              // SizeOfHeaders
    buf.extend_from_slice(&0u32.to_le_bytes());                      // CheckSum (0 = ignore)
    buf.extend_from_slice(&IMAGE_SUBSYSTEM_WINDOWS_CUI.to_le_bytes());
    // Drop DYNAMIC_BASE: pairs with RELOCS_STRIPPED above (image is fixed
    // at ImageBase = 0x140000000, which is canonical for x64 .exes and
    // historically free of conflicts).
    let dll_chars = IMAGE_DLLCHARACTERISTICS_NX_COMPAT
        | IMAGE_DLLCHARACTERISTICS_TERMINAL_SERVER_AWARE;
    buf.extend_from_slice(&dll_chars.to_le_bytes());
    // Stack/heap reserves: 1 MiB stack reserve, 4 KiB stack commit, same for heap.
    buf.extend_from_slice(&(0x100000u64).to_le_bytes());             // SizeOfStackReserve
    buf.extend_from_slice(&(0x1000u64).to_le_bytes());               // SizeOfStackCommit
    buf.extend_from_slice(&(0x100000u64).to_le_bytes());             // SizeOfHeapReserve
    buf.extend_from_slice(&(0x1000u64).to_le_bytes());               // SizeOfHeapCommit
    buf.extend_from_slice(&0u32.to_le_bytes());                      // LoaderFlags
    buf.extend_from_slice(&16u32.to_le_bytes());                     // NumberOfRvaAndSizes
    // Data Directories (16 × 8 bytes = 128). Indices we care about:
    //   1: Import Directory   → idata
    //   12: Import Address Table → IAT region inside idata
    let mut dirs = vec![[0u32; 2]; 16];
    dirs[1]  = [descriptors_rva, idata_descriptors as u32];
    dirs[12] = [iat_rva, idata_iat as u32];
    for d in &dirs {
        buf.extend_from_slice(&d[0].to_le_bytes());
        buf.extend_from_slice(&d[1].to_le_bytes());
    }

    // Section table (3 sections × 40 bytes).
    write_section(&mut buf, b".text",
        text_size_unaligned as u32, text_rva, text_size, text_file_off,
        IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ);
    write_section(&mut buf, b".rdata",
        rdata.len() as u32, rdata_rva, rdata_size, rdata_file_off,
        IMAGE_SCN_CNT_INITIALIZED | IMAGE_SCN_MEM_READ);
    write_section(&mut buf, b".idata",
        idata_size_unaligned as u32, idata_rva, idata_size, idata_file_off,
        IMAGE_SCN_CNT_INITIALIZED | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE);

    // Pad headers to file alignment.
    while buf.len() < HEADERS_SIZE as usize { buf.push(0); }

    // .text
    buf.extend_from_slice(&text);

    // .rdata
    buf.extend_from_slice(&rdata);
    while buf.len() < (rdata_file_off + rdata_size) as usize { buf.push(0); }

    // .idata
    // IMAGE_IMPORT_DESCRIPTOR for kernel32:
    //   OriginalFirstThunk = ilt_rva
    //   TimeDateStamp = 0
    //   ForwarderChain = 0
    //   Name = dll_name_rva
    //   FirstThunk = iat_rva
    buf.extend_from_slice(&ilt_rva.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&dll_name_rva.to_le_bytes());
    buf.extend_from_slice(&iat_rva.to_le_bytes());
    // Null terminator descriptor.
    buf.extend_from_slice(&[0u8; 20]);

    // ILT — one entry pointing at the Hint/Name table entry.
    buf.extend_from_slice(&(func_hint_rva as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // null

    // IAT — initially the same as ILT; OS overwrites with the resolved fn ptr.
    buf.extend_from_slice(&(func_hint_rva as u64).to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes()); // null

    while buf.len() < total_file_size as usize { buf.push(0); }

    buf
}

fn write_section(buf: &mut Vec<u8>, name: &[u8],
    virtual_size: u32, virtual_address: u32,
    raw_size: u32, raw_pointer: u32,
    characteristics: u32)
{
    let mut name8 = [0u8; 8];
    let len = name.len().min(8);
    name8[..len].copy_from_slice(&name[..len]);
    buf.extend_from_slice(&name8);
    buf.extend_from_slice(&virtual_size.to_le_bytes());
    buf.extend_from_slice(&virtual_address.to_le_bytes());
    buf.extend_from_slice(&raw_size.to_le_bytes());
    buf.extend_from_slice(&raw_pointer.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // PointerToRelocations
    buf.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
    buf.extend_from_slice(&0u16.to_le_bytes()); // NumberOfRelocations
    buf.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
    buf.extend_from_slice(&characteristics.to_le_bytes());
}

fn align_up(v: u32, align: u32) -> u32 {
    (v + align - 1) & !(align - 1)
}

/// Round `v` up to the PE section alignment (`SECTION_ALIGN`). Exposed so
/// the aether-asm driver can compute matching RVAs when resolving internal
/// relocs against the layout used by `build_minimal_exe`.
pub fn section_align_up(v: u32) -> u32 { align_up(v, SECTION_ALIGN) }

/// One imported DLL with its requested symbols.
#[derive(Clone, Debug)]
pub struct PeImport {
    pub dll: String,
    pub names: Vec<String>,
}

/// One call site in the user .text that should resolve to a thunk for an
/// external symbol. `offset` is the start of the rel32 disp field within
/// the user text (i.e. for a `call rel32` it points one byte past the `E8`
/// opcode); the writer overwrites those 4 bytes with the rel32 to the thunk.
#[derive(Clone, Debug)]
pub struct ExternalCallSite {
    pub offset_in_user_text: u32,
    pub symbol: String,
}

/// Full PE32+ image build. Compared with `build_minimal_exe` this:
///   * Accepts arbitrary `.rdata` (string-literal pool from the user's asm).
///   * Accepts a list of imported DLLs and their functions; **kernel32.dll
///     with `ExitProcess` is added implicitly** if not already present, so
///     the entry stub can always tail-call it.
///   * Generates one 6-byte indirect-jmp thunk per imported function and
///     patches `external_call_sites` to point at those thunks.
///
/// The user is responsible for resolving internal-text and rdata-text
/// relocations before passing `user_text` in. See
/// `coff::ObjectBuilder::resolve_internal_relocs` (skip externals).
pub fn build_full_exe(
    mut user_text: Vec<u8>,
    rdata: Vec<u8>,
    mut imports: Vec<PeImport>,
    external_call_sites: &[ExternalCallSite],
) -> Vec<u8> {
    // ---- normalise the import order ----
    // The Windows loader walks the import descriptors in order, calling
    // DllMain on each as it loads. Many DLLs' DllMain (Rust cdylibs in
    // particular) call into kernel32 routines (TLS, sync primitives) — so
    // **kernel32 must be initialised first**. We always pull it to the head
    // of the descriptor list, and ensure ExitProcess is one of its imports
    // so the entry stub's tail-call always resolves.
    let kernel_idx = imports.iter().position(|i| i.dll.eq_ignore_ascii_case("kernel32.dll"));
    let mut kernel = match kernel_idx {
        Some(i) => imports.remove(i),
        None => PeImport { dll: "kernel32.dll".into(), names: vec![] },
    };
    if !kernel.names.iter().any(|n| n == "ExitProcess") {
        kernel.names.insert(0, "ExitProcess".into());
    }
    imports.insert(0, kernel);

    // ---- compute layout ----
    let total_fns: usize = imports.iter().map(|i| i.names.len()).sum();
    let thunks_size = (total_fns as u32) * 6;
    let user_text_len = user_text.len() as u32;
    let text_size_unaligned = ENTRY_STUB_SIZE + user_text_len + thunks_size;
    let text_size = align_up(text_size_unaligned, FILE_ALIGN);

    // .rdata: caller-provided pool + all import-name strings (Hint/Name table)
    // + DLL name strings. We assemble a fresh rdata buffer that starts with
    // the user's rdata (so the caller's rdata RVAs match `rdata_rva`) and
    // appends import-side strings after.
    let user_rdata_len = rdata.len();
    let mut full_rdata = rdata;
    // For each fn: 2-byte hint + name + null + pad-to-even.
    // Capture hint/name RVAs (offset within full_rdata) per (dll_idx, fn_idx).
    let mut hint_name_offsets: Vec<Vec<u32>> = Vec::with_capacity(imports.len());
    for imp in &imports {
        let mut for_dll = Vec::with_capacity(imp.names.len());
        for n in &imp.names {
            let off = full_rdata.len() as u32;
            for_dll.push(off);
            full_rdata.extend_from_slice(&[0, 0]); // hint
            full_rdata.extend_from_slice(n.as_bytes());
            full_rdata.push(0);
            if full_rdata.len() % 2 != 0 { full_rdata.push(0); }
        }
        hint_name_offsets.push(for_dll);
    }
    // DLL name strings.
    let mut dll_name_offsets: Vec<u32> = Vec::with_capacity(imports.len());
    for imp in &imports {
        dll_name_offsets.push(full_rdata.len() as u32);
        full_rdata.extend_from_slice(imp.dll.as_bytes());
        full_rdata.push(0);
        while full_rdata.len() % 2 != 0 { full_rdata.push(0); }
    }
    let _ = user_rdata_len;
    let rdata_size_unaligned = full_rdata.len() as u32;
    let rdata_size = align_up(rdata_size_unaligned, FILE_ALIGN);

    // .idata: per-DLL import descriptor (20 bytes) + null terminator (20)
    //         + per-DLL ILT: (n_fns + 1) × 8
    //         + per-DLL IAT: (n_fns + 1) × 8
    let n_dlls = imports.len();
    let descriptors_size = ((n_dlls + 1) * 20) as u32;
    let ilt_total: u32 = imports.iter().map(|i| ((i.names.len() + 1) * 8) as u32).sum();
    let iat_total: u32 = ilt_total;
    let idata_size_unaligned = descriptors_size + ilt_total + iat_total;
    let idata_size = align_up(idata_size_unaligned, FILE_ALIGN);

    let text_rva  = SECTION_ALIGN;
    let rdata_rva = align_up(text_rva  + text_size,  SECTION_ALIGN);
    let idata_rva = align_up(rdata_rva + rdata_size, SECTION_ALIGN);
    let image_size = align_up(idata_rva + idata_size, SECTION_ALIGN);

    let text_file_off  = HEADERS_SIZE;
    let rdata_file_off = text_file_off + text_size;
    let idata_file_off = rdata_file_off + rdata_size;
    let total_file_size = idata_file_off + idata_size;

    // RVAs inside .idata.
    let descriptors_rva = idata_rva;
    let mut ilts_rva: Vec<u32> = Vec::with_capacity(n_dlls);
    let mut iats_rva: Vec<u32> = Vec::with_capacity(n_dlls);
    {
        let mut cur_ilt = descriptors_rva + descriptors_size;
        let mut cur_iat = cur_ilt + ilt_total;
        for imp in &imports {
            ilts_rva.push(cur_ilt);
            iats_rva.push(cur_iat);
            let stride = ((imp.names.len() + 1) * 8) as u32;
            cur_ilt += stride;
            cur_iat += stride;
        }
    }
    let first_iat_rva = iats_rva.first().copied().unwrap_or(idata_rva);

    // Per-symbol IAT slot RVA.
    let mut iat_rva: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    for (di, imp) in imports.iter().enumerate() {
        for (fi, name) in imp.names.iter().enumerate() {
            iat_rva.insert(name.clone(), iats_rva[di] + (fi as u32) * 8);
        }
    }

    // Thunks live at the end of .text. Per-symbol RVA, in declaration order.
    let thunks_base_rva = text_rva + ENTRY_STUB_SIZE + user_text_len;
    let mut thunk_rva: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    {
        let mut idx = 0u32;
        for imp in &imports {
            for n in &imp.names {
                thunk_rva.insert(n.clone(), thunks_base_rva + idx * 6);
                idx += 1;
            }
        }
    }

    // ---- patch external call sites in user_text to point at thunks ----
    // Each site: 4 bytes of rel32 starting at `offset_in_user_text`. The
    // CALL instruction's RIP-after-rel32 = text_rva + ENTRY_STUB_SIZE +
    // offset + 4. Target = thunk_rva[symbol].
    for site in external_call_sites {
        let target = *thunk_rva.get(&site.symbol).expect("thunk for missing symbol");
        let site_va = text_rva + ENTRY_STUB_SIZE + site.offset_in_user_text;
        let rip_after = site_va + 4;
        let rel32: i32 = (target as i64 - rip_after as i64) as i32;
        let off = site.offset_in_user_text as usize;
        user_text[off..off + 4].copy_from_slice(&rel32.to_le_bytes());
    }

    // ---- entry stub: same shape as `build_minimal_exe`, but the IAT slot
    // it tail-calls is the explicit ExitProcess slot (which we forced to
    // exist above).
    let exitproc_iat = *iat_rva.get("ExitProcess").expect("ExitProcess in IAT");
    let entry_rva = text_rva;
    let user_main_rva = text_rva + ENTRY_STUB_SIZE;
    let mut stub = Vec::with_capacity(20);
    stub.extend_from_slice(&[0x48, 0x83, 0xEC, 0x20]);   // sub rsp, 32
    let call_user_rip_end = entry_rva + 4 + 5;
    let call_user_rel: i32 = (user_main_rva as i64 - call_user_rip_end as i64) as i32;
    stub.push(0xE8);
    stub.extend_from_slice(&call_user_rel.to_le_bytes());
    stub.extend_from_slice(&[0x89, 0xC1]);               // mov ecx, eax
    let call_iat_rip_end = entry_rva + 11 + 6;
    let iat_rel: i32 = (exitproc_iat as i64 - call_iat_rip_end as i64) as i32;
    stub.extend_from_slice(&[0xFF, 0x15]);               // call qword ptr [rip+disp32]
    stub.extend_from_slice(&iat_rel.to_le_bytes());
    stub.extend_from_slice(&[0x0F, 0x0B]);               // ud2
    stub.push(0x90);                                     // nop pad
    debug_assert_eq!(stub.len() as u32, ENTRY_STUB_SIZE);

    // ---- build thunks: jmp qword ptr [rip+disp32] ----
    let mut thunks = Vec::with_capacity(thunks_size as usize);
    let mut idx = 0u32;
    for imp in &imports {
        for n in &imp.names {
            let thunk_va = thunks_base_rva + idx * 6;
            let target_iat = iat_rva[n];
            let rip_after = thunk_va + 6;
            let disp: i32 = (target_iat as i64 - rip_after as i64) as i32;
            thunks.extend_from_slice(&[0xFF, 0x25]);
            thunks.extend_from_slice(&disp.to_le_bytes());
            idx += 1;
        }
    }
    debug_assert_eq!(thunks.len() as u32, thunks_size);

    // Final .text bytes: stub + user_text (with relocs already patched) + thunks.
    let mut text = Vec::with_capacity(text_size as usize);
    text.extend_from_slice(&stub);
    text.append(&mut user_text);
    text.extend_from_slice(&thunks);
    while text.len() < text_size as usize { text.push(0); }

    // ---- write file ----
    let mut buf = Vec::with_capacity(total_file_size as usize);

    // DOS header.
    let mut dos = vec![0u8; 64];
    dos[0] = b'M'; dos[1] = b'Z';
    dos[60..64].copy_from_slice(&64u32.to_le_bytes());
    buf.extend_from_slice(&dos);

    buf.extend_from_slice(b"PE\0\0");

    // COFF File Header.
    buf.extend_from_slice(&IMAGE_FILE_MACHINE_AMD64.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes());                    // NumberOfSections
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&240u16.to_le_bytes());
    // RELOCS_STRIPPED + no DYNAMIC_BASE = "this image is fixed at ImageBase".
    // Required because we don't write a .reloc section; without RELOCS_STRIPPED
    // the loader thinks it can rebase us, then panics when it can't find the
    // .reloc table and the inner DLL load chain partially completes.
    let characteristics = IMAGE_FILE_RELOCS_STRIPPED
        | IMAGE_FILE_EXECUTABLE_IMAGE
        | IMAGE_FILE_LARGE_ADDRESS_AWARE;
    buf.extend_from_slice(&characteristics.to_le_bytes());

    // Optional Header (PE32+).
    buf.extend_from_slice(&0x20Bu16.to_le_bytes());
    buf.extend_from_slice(&[14, 0]);
    buf.extend_from_slice(&text_size.to_le_bytes());
    buf.extend_from_slice(&(rdata_size + idata_size).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&entry_rva.to_le_bytes());
    buf.extend_from_slice(&text_rva.to_le_bytes());
    buf.extend_from_slice(&IMAGE_BASE.to_le_bytes());
    buf.extend_from_slice(&SECTION_ALIGN.to_le_bytes());
    buf.extend_from_slice(&FILE_ALIGN.to_le_bytes());
    buf.extend_from_slice(&[6, 0]);
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&[6, 0]);
    buf.extend_from_slice(&[0, 0]);
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&image_size.to_le_bytes());
    buf.extend_from_slice(&HEADERS_SIZE.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&IMAGE_SUBSYSTEM_WINDOWS_CUI.to_le_bytes());
    // Drop DYNAMIC_BASE: pairs with RELOCS_STRIPPED above (image is fixed
    // at ImageBase = 0x140000000, which is canonical for x64 .exes and
    // historically free of conflicts).
    let dll_chars = IMAGE_DLLCHARACTERISTICS_NX_COMPAT
        | IMAGE_DLLCHARACTERISTICS_TERMINAL_SERVER_AWARE;
    buf.extend_from_slice(&dll_chars.to_le_bytes());
    buf.extend_from_slice(&(0x100000u64).to_le_bytes());
    buf.extend_from_slice(&(0x1000u64).to_le_bytes());
    buf.extend_from_slice(&(0x100000u64).to_le_bytes());
    buf.extend_from_slice(&(0x1000u64).to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes());
    buf.extend_from_slice(&16u32.to_le_bytes());
    let mut dirs = vec![[0u32; 2]; 16];
    dirs[1]  = [descriptors_rva, descriptors_size];
    dirs[12] = [first_iat_rva, iat_total];
    for d in &dirs {
        buf.extend_from_slice(&d[0].to_le_bytes());
        buf.extend_from_slice(&d[1].to_le_bytes());
    }

    // Section table.
    write_section(&mut buf, b".text",
        text_size_unaligned, text_rva, text_size, text_file_off,
        IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ);
    write_section(&mut buf, b".rdata",
        rdata_size_unaligned, rdata_rva, rdata_size, rdata_file_off,
        IMAGE_SCN_CNT_INITIALIZED | IMAGE_SCN_MEM_READ);
    write_section(&mut buf, b".idata",
        idata_size_unaligned, idata_rva, idata_size, idata_file_off,
        IMAGE_SCN_CNT_INITIALIZED | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE);

    while buf.len() < HEADERS_SIZE as usize { buf.push(0); }

    // .text
    buf.extend_from_slice(&text);

    // .rdata (user pool then import-name strings).
    buf.extend_from_slice(&full_rdata);
    while buf.len() < (rdata_file_off + rdata_size) as usize { buf.push(0); }

    // .idata: import descriptors, ILTs, IATs.
    for (i, imp) in imports.iter().enumerate() {
        buf.extend_from_slice(&ilts_rva[i].to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&(rdata_rva + dll_name_offsets[i]).to_le_bytes());
        buf.extend_from_slice(&iats_rva[i].to_le_bytes());
        let _ = imp;
    }
    buf.extend_from_slice(&[0u8; 20]); // null descriptor

    for (di, imp) in imports.iter().enumerate() {
        for fi in 0..imp.names.len() {
            let hint_name_rva = (rdata_rva + hint_name_offsets[di][fi]) as u64;
            buf.extend_from_slice(&hint_name_rva.to_le_bytes());
        }
        buf.extend_from_slice(&0u64.to_le_bytes());
    }
    for (di, imp) in imports.iter().enumerate() {
        for fi in 0..imp.names.len() {
            let hint_name_rva = (rdata_rva + hint_name_offsets[di][fi]) as u64;
            buf.extend_from_slice(&hint_name_rva.to_le_bytes());
        }
        buf.extend_from_slice(&0u64.to_le_bytes());
    }

    while buf.len() < total_file_size as usize { buf.push(0); }

    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the structural invariants of the bytes returned by
    /// `build_minimal_exe`: DOS magic, PE signature at the right offset,
    /// PE32+ optional-header magic, three sections, expected total size.
    /// End-to-end "does it actually run on Windows and exit 42" coverage
    /// lives in the runtime audit dimension once aether-asm grows the
    /// `--pe` flag — keeping that out of unit tests means the unit tests
    /// stay portable.
    #[test]
    fn minimal_exe_structural_invariants() {
        let user_code: Vec<u8> = vec![
            0xB8, 0x2A, 0x00, 0x00, 0x00,   // mov eax, 42
            0xC3,                            // ret
        ];
        let bytes = build_minimal_exe(&user_code);
        assert!(bytes.len() >= 0x800, "exe bytes implausibly small: {}", bytes.len());
        assert_eq!(&bytes[0..2], b"MZ");
        let pe_off = u32::from_le_bytes([bytes[60], bytes[61], bytes[62], bytes[63]]) as usize;
        assert_eq!(&bytes[pe_off..pe_off + 4], b"PE\0\0");
        // PE32+ optional header magic is at PE+24.
        let opt_magic = u16::from_le_bytes([bytes[pe_off + 24], bytes[pe_off + 25]]);
        assert_eq!(opt_magic, 0x20B, "expected PE32+ magic");
        // NumberOfSections lives at PE+6.
        let nsec = u16::from_le_bytes([bytes[pe_off + 6], bytes[pe_off + 7]]);
        assert_eq!(nsec, 3);
    }
}
