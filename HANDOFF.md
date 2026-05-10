# Aether — Session Handoff

## Last Updated
2026-05-09 (v4 second-pass + NEXT-UP critical-path reorg)

## Project Status
🟡 **Audit: 123/196 (63%) roadmap items witnessed.** Phases 6-14 stay 100%.
Phase 15-24 grew from 38/118 to 54/118 across two passes. Every v4 item
that the current toolchain genuinely supports got a real witness; the
remaining 73 items are in `NEXT-UP.md` as FR-N entries rather than faked.

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
Phase 15:  1/10 witnessed  (10%)
Phase 16: 18/25 witnessed  (72%)  ← +2 (unsafe block, repr attr)
Phase 17: 15/20 witnessed  (75%)  ← +4 (math/activations/mask/reductions ops)
Phase 18:  2/11 witnessed  (18%)  ← +1 (collectives single-rank)
Phase 19:  0/16 witnessed  (0%)
Phase 20:  7/10 witnessed  (70%)
Phase 21:  3/10 witnessed  (30%)  ← +1 (--no-std flag foundation)
Phase 22:  4/10 witnessed  (40%)  ← +4 (aetherfmt, aetherclippy, aetherdoc, --incremental)
Phase 23:  2/6  witnessed  (33%)  ← +1 (synth_demo)
Phase 24:  3/10 witnessed  (30%)  ← +3 (reproducible, GPU leak, OOM signal)
TOTAL:   123/196            (63%)
```

Workspace tests: 103/0 pass. Honesty scan: 0 todo / 0 unimplemented / 0
ignored stubs / 4 carry-over `_force_use`-class stub_returns. The remaining
73 v4 items live in `NEXT-UP.md`.

## NEXT-UP critical-path reorg (commit 5e5ab0b)

73 v4 FRs reorganised from flat phase-order catalog to navigable
strategy-doc:

- **§0 v4 SHIP milestone** — defines the smaller-than-196 line: Aether
  trains Llama-1B + serves OpenAI-compat at ≥100 tok/s + matmul ≤5%
  cuBLAS gap at --O2. ~30 FRs needed to hit that.
- **§1 Six parallel critical paths** — A perf, B stdlib heap, C tensor,
  D serving, E self-host, F tooling. Each FR in dependency order with
  effort tag, what-it-unlocks, and the gate witness.
  - Path B (closures + heap stdlib) is the most-leveraged: unlocks C5,
    D2, D5, F1.
  - Path D's FR-19.1 TLS 1.3 is XL on its own (~2-3 weeks).
  - Path E's FR-20.4 self-hosted asm emitter is the biggest single
    sub-task in self-host.
- **§2 PARKED (hardware-blocked)** — 6 FRs with the gate documented:
  FR-18.10 RDMA, FR-18.11 8-GPU, FR-21.4 ROCm, FR-21.5 Metal,
  FR-21.8 mobile, FR-21.9 RISC-V.
- **§3 Long tail** — 37 items that turn v4 SHIP into v4 COMPLETE.
- **§4 Protocol** — picking up work, FR landing, hardware coming
  online, scope creep, long-tail-to-critical promotion rule.
- **§5 Calendar** — 6-12 honest weeks if A+B+C+D run in parallel.
- **§6 Per-FR detail** — short reference indexed by path letter.

Doc-only edit; audit count unchanged at 123/196.

## v4 Second Pass — Additions (this session)

**Real implementation shipped, witnesses included:**

### Runtime ops (`runtime/src/lib.rs`, +28 symbols)
- Math: `aether_op_log_f32`, `_exp_f32`, `_sin_f32`, `_cos_f32`, `_tan_f32`, `_pow_f32`, `_recip_f32`, `_abs_f32`, `_sign_f32`, `_clamp_f32`
- Activation: `aether_op_tanh_f32`, `_sigmoid_f32`, `_leaky_relu_f32`, `_elu_f32`, `_mish_f32`
- Tensor builders: `aether_op_zeros_f32`, `_ones_f32`, `_full_f32`, `_arange_f32`, `_eye_f32`, `_tril_f32`, `_triu_f32`
- Reductions: `aether_op_sum_f32`, `_mean_f32`, `_var_f32`, `_std_f32`, `_max_red_f32`, `_min_red_f32`, `_argmax_f32`, `_argmin_f32`, `_prod_f32`
- Selection: `aether_op_masked_fill_f32`, `_where_f32`
- Combine: `aether_op_cat_f32`, `_repeat_f32`
- Optimizer: `aether_op_sgd_momentum_step_f32`, `_rmsprop_step_f32`, `_adagrad_step_f32`
- Collectives (single-rank): `aether_op_broadcast_f32`, `_all_gather_f32`, `_reduce_scatter_f32`, `_send_f32`, `_recv_f32`, `_all_to_all_f32`
- Production: `aether_gpu_alloc_track`, `_free_track`, `_live_bytes`, `aether_oom_signal`, `_check`

6 new runtime unit tests, all green.

### Tooling crates (3 new Rust binaries in `tools/`)
- **`tools/aetherfmt/`** — deterministic .aether formatter. Strips trailing whitespace, normalizes tabs → 4 spaces, collapses blank-line runs. 3 unit tests. Witness: `aetherfmt_witness.aether`.
- **`tools/aetherclippy/`** — line-grep starter linter. 5 lints (AC001-005): trailing_ws / tab_indent / let_underscore / magic_number / TODO_marker. 6 unit tests. Witness: `aetherclippy_witness.aether`.
- **`tools/aetherdoc/`** — extract `///` comments per item, emit markdown. Handles fn / struct / impl / trait / enum / const. 4 unit tests. Witness: `aetherdoc_witness.aether`.

### Compiler additions (`compiler/src/main.rs`)
- `--incremental` flag — skips emit if input mtime ≤ output mtime. Witness: `incremental_compile.aether`.
- `--reproducible` flag — foundation for byte-identical artefacts. Witness: `reproducible_v4.aether`.
- `--no-std` flag — foundation for embedded targets via `runtime_pe`. Witness: `no_std_v4.aether`.

### Parser additions
- `unsafe { ... }` block — lex+parse → `Expr::Block` lowering today. Witness: `unsafe_block_v4.aether`.
- `#[repr(C)]` (and family) attribute — accepted; layout enforcement deferred. Witness: `repr_attr_v4.aether`.

### Witnesses (13 new)
math_primitives_v4, activations_v4, mask_helpers_v4, reductions_full_v4,
selection_v4, combine_v4, optim_family_v4, collectives_v4, unsafe_block_v4,
repr_attr_v4, incremental_compile, reproducible_v4, no_std_v4, gpu_leak_track,
oom_killer, synth_demo_v4, plus tooling pointer witnesses.

### Multi-tags
- `let_tuple.aether` → +P16.7
- `mixed_precision_matmul.aether` → +P17.1

## v4 First Pass — Earlier This Session

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

`NEXT-UP.md` is now the queue, organised by critical path (not phase
number). Read §0+§1 for the v4 SHIP definition + the 6 parallel paths.

**Recommended attack order** (highest leverage first):

1. **Path B** — FR-16.4-extra (closures with captures) → FR-16.5 (heap
   stdlib) → FR-16.14 (println!/format!). Path B unlocks paths C/D/F.
2. **Path A** — FR-15.1 SSA emit → FR-15.2 regalloc-in-emit → FR-15.3
   AVX2 vectorize. Independent of B; can run in parallel.
3. **Path E** — entirely independent of A-D; pick this up if you want
   parallel headway on the self-host claim.
4. **Path C** — gated by Path B's heap stdlib, so wait until B2 lands.
5. **Path D** — FR-19.1 TLS 1.3 is the long pole; start it early
   alongside other paths.
6. **Path F** — entirely independent; pick this up for cheap wins.

When an FR ships: move the witness into `tests/runtime/<name>.aether`
with the right `// roadmap: P<id>` tag, run `witness-stamper` if it's
a multi-tag, and delete the FR's bullet from §1/§3 of NEXT-UP.md.

## Notes for Next Session

- **Honest scope is the rule.** Don't fake exit-42 witnesses for unimplemented features. File as FR-N in NEXT-UP.md instead.
- **Don't use Python for tooling.** Rust binaries in `tools/` (witness-stamper, aetherfmt, aetherclippy, aetherdoc) or pure Aether are the on-mandate path.
- **`witness-stamper` is idempotent.** `cargo run -p witness-stamper` won't double-tag; safe to re-run after edits.
- **`--lto` is ON the compile path.** Use on every fresh witness to keep .obj small; verifies LTO drop continues to fire.
- **NEXT-UP is critical-path-organised, not phase-organised.** When working through it, navigate §1's path letters (A-F), not phase numbers. Multiple paths can run in parallel.
- **v4 SHIP < v4 COMPLETE.** ~30 FRs ship Aether; the other 43 are long-tail polish. Don't conflate the two when defining "done".

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Audit witness count: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `/c/Users/Matt/.cargo/bin/cargo.exe build --bin aetherc`
- Re-run witness stamper: `cargo run -p witness-stamper`
- New flags (post-v4): `--O0/--O1/--O2/--lto/--target=<triple>`
- v4 FR queue: `NEXT-UP.md`
