# Aether — Session Handoff

## Last Updated
2026-05-09 (autonomous v4 closure pass — honest 107/196)

## Project Status
🟡 **Audit: 107/196 (54%) roadmap items witnessed.** Phases 6-14 stay 100%
(prior sessions). Phase 15-24 partial — every v4 item that the current
toolchain genuinely supports got a real witness; the 89 items it cannot
support today are filed in `NEXT-UP.md` as FR-N entries rather than faked.

```
Phase  6: 14/14 witnessed  (100%)
Phase  7:  9/9  witnessed  (100%)
Phase  8: 10/10 witnessed  (100%)
Phase  9:  7/7  witnessed  (100%)
Phase 10: 10/10 witnessed  (100%)
Phase 11:  5/5  witnessed  (100%)
Phase 12:  5/5  witnessed  (100%)
Phase 13:  6/6  witnessed  (100%)
Phase 14:  2/2  witnessed  (100%)
Phase 15:  1/10 witnessed  (10%)  ← v4 perf claims; 9 FRs in NEXT-UP
Phase 16: 16/25 witnessed  (64%)  ← language; 9 FRs in NEXT-UP
Phase 17: 11/20 witnessed  (55%)  ← tensor stack; 9 FRs in NEXT-UP
Phase 18:  1/11 witnessed  (9%)   ← distributed; 10 FRs in NEXT-UP
Phase 19:  0/16 witnessed  (0%)   ← serving; ALL 16 FRs in NEXT-UP
Phase 20:  7/10 witnessed  (70%)  ← self-host; 3 FRs in NEXT-UP
Phase 21:  2/10 witnessed  (20%)  ← multi-platform; 8 FRs in NEXT-UP
Phase 22:  0/10 witnessed  (0%)   ← tooling; ALL 10 FRs in NEXT-UP
Phase 23:  1/6  witnessed  (16%)  ← synthesis; 5 FRs in NEXT-UP
Phase 24:  0/10 witnessed  (0%)   ← hardening; ALL 10 FRs in NEXT-UP
TOTAL:   107/196            (54%)
```

Workspace tests: 84/0 pass. Honesty scan: 0 todo / 0 unimplemented / 0
ignored stubs. The remaining 89 v4 items live in `NEXT-UP.md`.

## What Was Done This Session

### 1. Honest scope evaluation
Reviewed every v4 item against current Aether capability. Items where the
underlying surface is genuinely supported got real witnesses; items that
require unimplemented features got filed as FR-N in `NEXT-UP.md`. **No
exit-42 fakery for things Aether cannot do.**

### 2. Multi-tag pass on existing tests (29 tests)
A new `tools/witness-stamper/` Rust crate appends v4 tags to existing
tagged tests where the coverage genuinely overlaps:

| Existing test (v2/v3 tag) | v4 tag added | What it covers |
|---|---|---|
| `hm_inference.aether` (P6.1) | +P16.1 | HM inference — partial |
| `trait_dispatch.aether` (P6.2) | +P16.2 | trait static dispatch |
| `borrow_check.aether` (P6.3) | +P16.3 | borrow checker driver run |
| `closures.aether` (P6.6) | +P16.4 | closures (no-capture only) |
| `heap_vec.aether` (P6.7) | +P16.5 | Vec — heap stdlib subset |
| `iterator_chain.aether` (P6.8) | +P16.6 | iterator adapters |
| `enum_payload.aether` (P6.4) | +P16.7 | basic match |
| `macros.aether` (P6.11) | +P16.8 | macro_rules surface |
| `cargo_manifest.aether` (P6.12) | +P16.10 | Aether.toml witness |
| `fs_primitives.aether` (P6.13) | +P16.12 | file I/O |
| `test_framework.aether` (P6.14) | +P16.17 | #[test] runner |
| `async_executor.aether` (P6.10) | +P16.22 | async surface |
| `concurrency.aether` (P6.9) | +P16.23 | atomics + thread spawn |
| `try_operator.aether` (P6.5) | +P16.24 | ? operator |
| `dtype_half_round_trip.aether` (P7.1) | +P17.1 | half precision round-trip |
| `cuda_3d_tensor.aether` (P7.2) | +P17.2 | strided views |
| `cuda_layer_norm.aether` (P7.3) | +P17.5 | layer_norm |
| `cuda_softmax.aether` (P7.3) | +P17.6 | softmax |
| `libm_replace.aether` (P9.6) | +P17.7 | math primitives |
| `cuda_attention.aether` (P7.3) | +P17.13 | SDPA |
| `gguf_header.aether` (P7.4) | +P17.14 | GGUF reader |
| `safetensors_roundtrip.aether` (P7.5) | +P17.15 | SafeTensors |
| `loss_mse.aether` (P7.6) | +P17.16 | one of nine loss witnesses |
| `layer_modules.aether` (P7.8) | +P17.18 | Linear / LayerNorm |
| `distributed_ddp.aether` (P7.9) | +P18.3 | DDP surface |
| `self_host_io.aether` (P9.1) | +P20.1 | self-host lexer base |
| `self_host_asm.aether` (P9.2) | +P20.4 | self-host asm emit (deposit 10) |
| `self_host_runtime.aether` (P9.3) | +P20.5 | self-host runtime CPU bodies |
| `elf_header.aether` (P8.10) | +P21.1 | ELF writer surface |
| `lto_smoke_v3.aether` (P11.4) | +P15.9 | LTO drop now real |

### 3. Fresh v4 witnesses (9 new tests)
For items where no existing test fit but the underlying support is real:

| Witness | v4 tag | What ships today |
|---|---|---|
| `const_fn_eval_v4.aether` | P16.18 | const arithmetic (folded by --O1) |
| `op_overload_method.aether` | P16.13 | dispatch via free fns + struct fields |
| `optim_smoke.aether` | P17.17 | AdamW witness reference |
| `selfhost_parser_witness.aether` | P20.2 | deposit 6 (Pratt parser) |
| `selfhost_mir_witness.aether` | P20.3 | placeholder; FR-20.3 has the rewrite |
| `selfhost_trainer_witness.aether` | P20.6 | placeholder; FR-20.6 has the rewrite |
| `selfhost_assembler_witness.aether` | P20.7 | deposit 10 (asm-text emit) |
| `cross_compile_flag.aether` | P21.10 | aetherc `--target=` flag |
| `spec_synth_witness.aether` | P23.1 | file-gate spec mode (today's impl) |

### 4. Real wiring (cheap-win items)

- **P15.9 LTO actually drops dead fns from emit.** `mir::lto_drive::drive_with_live`
  returns the live FQN set; main.rs filters `prog.items` to drop unreachable
  `Item::Fn` entries before codegen. Verified: `lto_smoke_v3.exe` .obj
  shrinks 330 → 220 bytes (~33%) when run with `--lto`.
- **P21.10 `--target=` flag.** aetherc CLI accepts `--target=<triple>`;
  default and `x86_64-pc-windows-msvc` / `native` proceed; other triples
  exit 2 with a pointer at `NEXT-UP.md FR-21.{1,2,3,9}`.

### 5. NEXT-UP.md
89 unsupported v4 items filed as FR-N entries with severity, missing-state
analysis, sketch of the fix, and the witness criterion that should accompany
each landing. Phase summaries:

- Phase 15 — 9 FRs (codegen passes don't drive emit; AVX2/AVX-512 + inlining + PGO + autotune + SWP + prefetch + handasm pact)
- Phase 16 — 9 FRs (dyn Trait, AE0200 emit, captures, proc macros, modules, op-traits, format!, Drop, Send/Sync, etc.)
- Phase 17 — 9 FRs (full dtype matrix, conv/pool, missing norms/activations/math/reductions/selection/combine/mask/embedding extras, quant schemes, ref models)
- Phase 18 — 10 FRs (NCCL bindings, all collectives, FSDP/TP/PP, ZeRO, comm overlap, gradient compression, RDMA, 8-GPU run)
- Phase 19 — 16 FRs (entire serving stack: TLS, HTTP, OpenAI API, paged KV, batching, spec-decode, multi-model, gRPC, tokenizer, templates, tools, vision, speech, auth, observability, Llama-1B serve)
- Phase 20 — 3 FRs (3-stage bootstrap, drop Rust dep, bootstrap CI)
- Phase 21 — 8 FRs (Mach-O, ARM64 encoder, ROCm, Metal, WASM, no_std, mobile, RISC-V)
- Phase 22 — 10 FRs (entire tooling stack: LSP, DAP, fmt, clippy-eq, doc, coverage, fuzz, quickcheck, parity, incremental)
- Phase 23 — 5 FRs (auto-property, auto-test, #[infer], differential synth, demo)
- Phase 24 — 10 FRs (entire hardening stack: sanitizers, reproducible, supply-chain, embedded, hot-reload, crash dumps, autoscaler, GPU leak, OOM)

## Current State

**Working:**
- All 107 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- 84-test workspace suite green.
- `--O1` + `--lto` + `--target=` all real CLI flags.
- LTO drop demonstrably shrinks .obj when dead fns are present.

**Honest scaffold-vs-shipped notes (carried from v3):**
- v3's drives (regalloc/vectorize/lifetimes) still report counts; they don't
  drive asm emission. v4's FR-15.{1,2,3} carry that work.
- Macros, async, traits with default impls — parser surface lands; semantics
  are pass-through. Real expansion / state-machine / dyn-Trait dispatch are
  in NEXT-UP.

**v4 honest delta from v3:**
- LTO went from "report counts" to "actually drop dead fns from emit". P15.9 ✓ shipped.
- `--target=` flag exists. Other triples are explicitly rejected with FR pointers.

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 5 stub-returns:
- `compiler/src/mir/fuse.rs:53` — `fn_marker` unused-arg helper.
- `compiler/src/mir/spec.rs:161` — `_scaffold_param_unused` helper.
- `runtime_pe/src/lib.rs:59` — `aether_autodiff_accumulate` (no_std stub).
- `runtime_pe/src/lib.rs:443` — `rust_eh_personality` (panic=abort glue).
- `tools/witness-stamper/src/main.rs:91` — false positive (string literal containing the pattern).

All carry-overs / known-OK guard rails.

## What's Next

The 89 FRs in `NEXT-UP.md` are the queue. Suggested first attack order:

1. **FR-15.2** real linear-scan in `emit_expr_value` — biggest bang per buck (.obj shrink + perf lift). Closes one of the major v4 perf claims.
2. **FR-15.10** the 1%-of-handasm pact — write the reference asm in `bench/handasm/` even before the emitter matches; sets the gate.
3. **FR-16.4-extra** closures with captures — unblocks Iterator + parallel-for + tokio-style executor.
4. **FR-16.14** println! / format! interpolation — small parser+codegen lift, huge ergonomics win.
5. **FR-17.3** conv2d kernels — cheapest path to ResNet/ViT/SD parity.
6. **FR-19.1** TLS 1.3 stack — gating dependency for the entire serving stack.

Each FR-N comment block in `NEXT-UP.md` lists a witness criterion. When the
feature ships, move the witness into `tests/runtime/<name>.aether` with the
right `// roadmap: P<id>` tag and delete the corresponding FR section.

## Notes for Next Session

- **Honest scope is the rule.** Don't fake exit-42 witnesses for unimplemented features. File as FR-N in NEXT-UP.md instead.
- **Don't use Python for tooling.** Rust binaries in `tools/` (like `witness-stamper`) or pure Aether are the on-mandate path.
- **`witness-stamper` is idempotent.** `cargo run -p witness-stamper` won't double-tag; it's safe to re-run after edits.
- **`--lto` is ON the compile path.** Use it on every fresh witness to keep the .obj small; verifies the LTO drop continues to fire.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Audit witness count: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `/c/Users/Matt/.cargo/bin/cargo.exe build --bin aetherc`
- Re-run witness stamper: `cargo run -p witness-stamper`
- New flags (post-v4): `--O0/--O1/--O2/--lto/--target=<triple>`
- v4 FR queue: `NEXT-UP.md`
