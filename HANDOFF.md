# Aether — Session Handoff

## Last Updated
2026-05-02 (afternoon)

## This-Session Adds (50/50 audit clean)

13. **Const generics with monomorphization** (`tests/runtime/generic_matmul.aether`). `fn forward<M, K, N>(x: Tensor<f32, [M, K]>, w: Tensor<f32, [K, N]>, y: Tensor<f32, [M, N]>) -> i32` — one template, infers concrete dim bindings at each call site from the caller's `tensor_shapes`, emits one mangled specialization per unique binding set (`forward__M8__K16__N4`, `forward__M4__K32__N2`). Templates filtered out of normal emission; pending-spec worklist drained after the initial fn loop, with cascading-call support. `GenericState` shared across per-fn `Locals` via `Rc<RefCell<...>>`. **Closes the biggest single ergonomic gap: layer fns are now genuinely reusable across model sizes.**
14. **`as` cast** (`tests/runtime/as_cast.aether`). `expr as Type` postfix-parses with same precedence as method-call/field, lowers through the existing `emit_cast` (i32/i64/f32/f64 round-trip). Identity casts and f→i32 added.
15. **Short-circuit `&&` / `||`** (`tests/runtime/short_circuit.aether`). Proper non-evaluation of rhs when lhs decides; proven by `(z != 0 && (10 / z) > 0)` not div-by-zeroing.
16. **Stack arrays `[T; N]`** (`tests/runtime/stack_array.aether`). `let buf: [i64; 8];` + `buf[i] = v` + `acc + buf[k]`. N consecutive slots reserved on the frame; index codegen does `negq` + `*8` (via 3 adds) + `addq base` + load/store. Required new `Instr` variants `MovRegFromBaseDisp` / `MovBaseDispFromReg` with REX.W + 8B/89 + ModRM-disp32 encoding for generic non-rbp/rsp base regs (`disp(%rdi)`); encoder + parser + size table updated.
17. **Bitwise `& | ^`** (`tests/runtime/bitwise.aether`). New `BinOp::BitAnd/BitOr/BitXor`, parser tier between cmp and add (correct Rust precedence), asm `andq/orq/xorq` reg-reg encodings (REX.W + 21/09/31 /r). Single `&` between two complete expressions parses as bitwise — prefix `&` is still address-of (parse_unary handles that).
18. **Hex / binary / octal int literals** (`tests/runtime/hex_lits.aether`). `0xFF`, `0b1111`, `0o17` lex via prefix peek + `i64::from_str_radix`. Underscore separators allowed.
19. **Shadowing** (`tests/runtime/shadowing.aether`). Verified to work for free under the let-allocates-fresh-slot model (no parser/codegen changes needed).

## Project Status
🟢 **A `.aether` source program now trains a model end-to-end with ZERO external linker.** `tests/runtime/pe_train_tiny.aether` compiles via `aetherc → x86-64 asm (ours) → COFF (ours) → PE32+ writer (ours)` and trains the same single-layer linear classifier as `train_tiny.aether`, but the .exe is produced **without invoking gcc/link/lld at any stage**. Multi-DLL IAT writes `kernel32.dll` + `aether_rt.dll` (slim `runtime_pe` cdylib) imports; per-symbol indirect-jmp thunks rewrite each `callq aether_*` site to point at the IAT slot. Loss curve via this path: `1.618 → 0.0081` over 50 steps. The system-linker path (`--emit=aether-bin`) still exists for programs that want the full Rust-std runtime; it loses `1.649 → 0.006` on the same model.

#24 (self-hosted linker) is fully complete. The toolchain has zero external deps for static binaries — only the Windows OS provides `kernel32.dll`.

**Pivot point**: roadmap items 1-25 proved the language CAN. The next phase is making it **easy + 1%-of-hand-written-asm fast**. Three ergonomic wins shipped this session toward the "easy" half — each one is a foundational primitive future work builds on (no rebuild risk):

1. **Nested calls in args** (`tests/runtime/nested_calls.aether`). Asm backend now spills each arg through a 3-phase push/load/unwind sequence — inner calls run between phase-1 pushes without disturbing the outer stack discipline. Removes the "hoist every subexpression to a let" boilerplate.

2. **`use runtime;`** (`tests/runtime/use_runtime.aether`, `stdlib/runtime.aether`). 60 lines of `extern fn aether_*` decls collapse to one line. Aetherc's `Item::Use` resolver loads `stdlib/<name>.aether` next to the binary and inlines items. Cycle-safe.

3. **`Tensor<T, [N]>` with auto-alloc + auto-free** (`tests/runtime/cuda_train_tiny_tensor.aether`). `let x: Tensor<f32, [128]>;` (no `=`) auto-emits `aether_dev_alloc_f32(128)` at the let position; the fn epilogue auto-emits the matching `aether_dev_free_f32(handle)` for every Tensor local in reverse declaration order. `%rax` is preserved across the free sequence via a `_ret_save_` slot. Tensor handles round-trip as `i64` everywhere a value is read, so existing FFI call sites take them unchanged. Two element types today: `f32` and `i32`. The 9 manual alloc + 9 manual free lines in `cuda_train_tiny.aether` collapse to 9 type annotations and zero free lines.

Combined effect: a meaningful end-to-end GPU training program (`cuda_train_tiny_tensor.aether`) is now ~60 lines vs the original ~80 — and the 20 lines saved are the most error-prone (paired alloc/free). Next ergonomic primitives queued (still no rebuild risk for what's there): struct literals, fixed-size arrays, method-call dispatch onto Tensor (so `x.matmul(&w)` becomes a thing), shape inference (so the `bsz, kk, nn` triplet at every op call site goes away).

5. **Const-resolved shape dims** (`tests/runtime/const_shape.aether`, `cuda_train_tiny_clean.aether`). `const BSZ: i32 = 8;` at file scope, then `Tensor<f32, [BSZ, KK]>` resolves the symbolic dim at codegen time through a per-fn `const_env` cloned in from `try_emit`. Change one number at the top, the whole graph reshapes. Same trick numpy/candle users get from passing dims to a constructor — but compile-time, no runtime dispatch.

6. **`adamw_step` method dispatch + frame-size fix** (`cuda_train_tiny_clean.aether`). `w.adamw_step(&dw, &mut m, &mut v, lr, b1, b2, eps, wd, step)` lowers to the 11-arg runtime call with `n` synthesized from `w`'s shape product. Caught a real bug along the way: `count_locals_in_expr` had no `MethodCall` arm so the desugared call's larger arg count wasn't accounted for in frame sizing — outgoing-args region was 7 slots short and adamw scribbled over the loop counter, exiting after one iteration. Fixed by counting `1 + args.len() + 3` (recv + args + worst-case shape int args) for every MethodCall. Audit caught it, the bisect found it.

The cleanest end-to-end GPU training program Aether expresses today (`tests/runtime/cuda_train_tiny_clean.aether`):

```
use runtime;
const BSZ: i32 = 8; const KK: i32 = 16; const NN: i32 = 4; const KN: i32 = 64;
fn main() -> i32 {
    aether_dev_init();
    // ... host buffers + h2d (still manual)
    let x:      Tensor<f32, [BSZ, KK]>;
    let w:      Tensor<f32, [KK, NN]>;
    let y:      Tensor<f32, [BSZ, NN]>;
    let probs:  Tensor<f32, [BSZ, NN]>;
    let dy:     Tensor<f32, [BSZ, NN]>;
    let dw:     Tensor<f32, [KK, NN]>;
    let mom1:   Tensor<f32, [KN]>;
    let mom2:   Tensor<f32, [KN]>;
    let labels: Tensor<i32, [BSZ]>;
    // ... h2d
    while s <= 50 {
        x.matmul(&w, &mut y);
        let loss: f32 = y.cross_entropy(&labels, &mut probs);
        if s == 1 { initial = loss; }
        final_loss = loss;
        probs.cross_entropy_backward(&labels, &mut dy);
        x.matmul_backward_rhs(&dy, &mut dw);
        w.adamw_step(&dw, &mut mom1, &mut mom2, 0.05, 0.9, 0.999, 0.00000001, 0.0, s);
        s = s + 1;
    }
    // ... cleanup
    if final_loss < initial { 0 } else { 1 }
}
```

Reads like idiomatic Rust+PyTorch. **39/39 runtime tests, audit clean.**

7. **Struct literals** (`tests/runtime/struct_lit.aether`). `Point { x: 5, y: 6, scale: 2.0 }` parses as `Expr::StructLit` with parser disambiguation: a `no_struct_literal` flag (mirroring Rust's) is threaded through `parse_expr`, disabled by `with_struct_lit_disabled` for the cond positions of `if`/`while`/`for` so `if cond { body }` doesn't get misread as `Ident { ... }`. Codegen handles `let x: Foo = Foo { ... };` by reusing the uninit-struct slot machinery and emitting per-field assignments from the lit's `(field_name, expr)` pairs. Field types come from the struct decl; assignment values come from the lit. Eliminates the 4-line `let x: Foo; x.a = 1; x.b = 2;` dance.

8. **Free fns with Tensor params** (`tests/runtime/fn_tensor_params.aether`). `fn matmul_into(x: Tensor<f32, [BSZ, KK]>, w: …, y: …)` works; the param-spill phase now resolves Tensor types through the const env and populates `tensor_shapes` from param types, so `x.matmul(&w, &mut y)` inside the fn body dispatches correctly. Refs of Tensor types collapse to the bare Tensor at param-typing time — the runtime value is the i64 handle either way. Closes the loop: training loops can be extracted into reusable `forward(x, w, …)` style fns.

9. **Struct literals** (`tests/runtime/struct_lit.aether`). `Foo { a: 1, b: 2.0 }` parses with `no_struct_literal` flag disambiguating against if/while/for cond blocks. Lowers to per-field assignment.

10. **impl blocks with `&self`/`self`** (`tests/runtime/impl_block.aether`). `impl Foo { fn bar(&self, ...) -> ... }` — methods name-mangled to `Foo__bar`; method-call dispatch routes `obj.bar(x)` to `Foo__bar(obj, x)`. Struct-by-value param/arg passing: each field occupies one MS x64 ABI slot. No borrow semantics yet (`&self` and `self` both copy). Caught a real bug along the way: arg-count loop was using `args.len()` instead of `arg_kinds.len()` so struct-expanded args were truncated.

11. **Enums + match** (`tests/runtime/enum_match.aether`). `enum Color { Red, Green, Blue }` lowers to int tags via const env (Red=0, Green=1, …). `Color::Green` path expr resolves at codegen. `match scrut { Color::Red => 1, 7 => 22, _ => 0 }` via cmp+jmp dispatch. Patterns: IntLit, EnumVariant (path), Wildcard. Discriminant-only — payload-carrying variants (`Some(i32)`, `Err(String)`) need the asm backend to grow tagged-union value layout, future work.

12. **Multi-Tensor model struct + impl forward** (`tests/runtime/model_struct.aether`, `tests/runtime/showcase.aether`). The real ML pattern. `struct Model { w: Tensor<f32, [KK, NN]>, eps: f32 }` + `impl Model { fn forward(&self, x: Tensor<…>, y: Tensor<…>) }`. Required: `&self.w` as Tensor field access (loads the i64 handle, not the slot address), Field-aware arg-shape extraction in method dispatch (looks up the field's Ty in the struct decl, extracts Tensor shape via const env), and struct-by-value passing for the `self` param (each field maps to its own ABI arg slot).

**`tests/runtime/showcase.aether`** exercises the full feature set in one program — `use`, `const`, `enum`, `match`, `struct` with mixed Tensor + scalar fields, `impl` method, struct literal init, multi-dim Tensors with auto-alloc/free, method-call dispatch on both Tensors and user types, `&self.w` field access, multi-arg FFI through the stack-spill ABI, and `if`/`else` based on a Phase tag. All on the cuBLAS GPU path. **43/43 audit clean.**

## Honest gap-to-Rust

What's still missing for "real Rust feature parity":
- ~~**True const generics**~~ ✅ shipped — `fn name<M, K>(...)` infers + monomorphizes per call site.
- ~~**Stack arrays `[T; N]`**~~ ✅ shipped (int/handle elements; float arrays would need 4-byte stride work).
- **Data-carrying enum variants** + pattern bindings (`Some(x)`, `Err(msg)`). Today's enums are discriminant-only — needs tagged-union value layout.
- **Closures + iterators** — none. Closures need environment capture + indirect-call ABI; iterators need traits.
- **Result<T, E> error propagation** (`?` operator). Depends on data-carrying enums.
- **Tuples `(a, b)`** + multi-return — needs sret-pointer ABI work.
- **Tensor-returning methods + chaining** — `let y = x.matmul(&w);` then `y.add(&b).relu()`. Largest remaining ML ergonomic. Needs allocating method dispatch + receiver-as-expression handling + shape propagation through call results.
- **Borrow checker** — none. `&` and `&mut` are syntactic decoration only.
- **Real type inference** — almost every `let` needs an explicit annotation today.
- **String type beyond literals** — no `String`, no concat, no formatting.
- **Vec / HashMap / Iterator stdlib**.
- **Cargo / dependency management**.
- **Bitwise shifts `<<` / `>>`** — skipped because lex-time `<<` conflicts with nested generics; needs context-sensitive split during parse.

We have a **focused ML DSL** that compiles to native + GPU and writes-like-Rust+PyTorch in the narrow slice the audit exercises. The next big arc is true const generics — once that lands, layer fns become genuinely reusable across model sizes, and the language clears the "could you write a transformer in this" bar.

4. **Multi-dim Tensor types + method-call dispatch** (`tests/runtime/cuda_train_tiny_methods.aether`). `Tensor<f32, [M, K]>` parses, allocates `M*K` floats, and remembers `[M, K]` in a per-fn shape sidecar. `x.matmul(&w, &mut y)` desugars in the asm backend to `aether_op_matmul_f32_cuda(x, w, y, M, K, N)` — M, K from the receiver's shape, N from the first arg's shape. `&x` on a Tensor-typed Ident yields the i64 handle (not the slot address) so the source code reads the same regardless of whether the ABI underneath wants ownership or a borrow. Dispatch table covers `matmul`, `matmul_backward_rhs`, `cross_entropy`, `cross_entropy_backward`. AdamW stays as a direct FFI call (12 hyperparam args; method dispatch isn't the win here). The training-loop body in `cuda_train_tiny_methods.aether`:

```
x.matmul(&w, &mut y);
let loss: f32 = y.cross_entropy(&labels, &mut probs);
probs.cross_entropy_backward(&labels, &mut dy);
x.matmul_backward_rhs(&dy, &mut dw);
aether_op_adamw_step_f32_cuda(w, dw, mom1, mom2, 0.05, 0.9, 0.999, 1e-8, 0.0, s, 64);
```

vs the old aether-direct version's:

```
aether_op_matmul_f32_cuda(x, w, y, bsz, kk, nn);
let loss: f32 = aether_op_cross_entropy_f32_cuda(y, labels, probs, bsz, nn);
aether_op_cross_entropy_backward_f32_cuda(probs, labels, dy, bsz, nn);
aether_op_matmul_backward_rhs_f32_cuda(x, dy, dw, bsz, kk, nn);
aether_op_adamw_step_f32_cuda(w, dw, mom1, mom2, 0.05, 0.9, 0.999, 1e-8, 0.0, s, kn);
```

Same loss curve, same exit code. The `bsz, kk, nn` triplet at every op call site is gone — pulled from compile-time Tensor shape. **35/35 runtime tests, audit clean.**

The "1% asm-fast" half stays open — the right next move is **kernel fusion**: instead of one cuBLAS launch per op (matmul → softmax → CE backward → matmul-backward = 4 launches × ~50 µs dispatch tax), aetherc-side MIR fusion emits a **single fused PTX kernel per training step**, JITted via nvrtc. Removes ~200 µs/step of dispatch overhead at our current model size; closes most of the remaining gap to hand-written CUDA. Skeleton next session.

**#25 FULLY COMPLETE**: `runtime/src/cuda.rs` (feature-gated behind `--features cuda`) is the real cuBLAS-backed GPU path. **End-to-end GPU training works** — `tests/runtime/cuda_train_tiny.aether` mirrors the CPU `train_tiny.aether` and runs every op on the GPU through cuBLAS sgemm + nvrtc-JITted custom kernels. Loss curve is **bit-identical** to the CPU baseline (`1.649244 → 0.006113`). 50-step total wallclock: ~51 ms.

Op coverage:
- **cuBLAS sgemm**: matmul, matmul_backward_lhs, matmul_backward_rhs.
- **nvrtc-JITted (compiled at first `aether_dev_init`)**: cross_entropy_fwd, cross_entropy_bwd, adamw_step. Source embedded in `KERNEL_SRC` constant in `cuda.rs`; cudarc `compile_ptx` + `load_ptx` + `get_func` + `launch_async`.
- **Memory**: aether_dev_alloc_f32, aether_dev_alloc_i32, free, h2d_f32, h2d_i32, d2h_f32. Two parallel `UnsafeCell<Vec<Option<CudaSlice<T>>>>` registries (single-threaded by construction; the `Mutex` was pure overhead and is gone).
- **Bench helpers**: aether_wall_us, aether_dev_sync, aether_op_matmul_f32_cuda_profile, aether_bench_matmul_batch.

**Bench numbers vs candle (local fork at `J:\candle-src`, 0.10.2, MSVC toolchain) on RTX 3070 Ti** — full writeup in `docs/BENCH_RESULTS.md`. Apples-to-apples cuBLAS sgemm (buffers held across the loop, no per-iter framework overhead either side):

| dim    | Aether-GPU per-iter | Candle-GPU per-iter | verdict                  |
|-------:|--------------------:|--------------------:|--------------------------|
|  64³   |             8 µs    |              13 µs  | **Aether 38 % faster**   |
| 256³   |            13 µs    |              23 µs  | **Aether 43 % faster**   |
| 512³   |            57 µs    |              45 µs  | candle 27 % faster       |
| 1024³  |           192 µs    |             242 µs  | **Aether 21 % faster**   |

**Aether matches or beats Candle on raw cuBLAS sgemm at 3 of 4 sizes.** The earlier "15× slower" reading came from `aether_op_matmul_f32_cuda`'s per-call buffer-registry take/put pattern — the headline result was hidden behind the registry's overhead. The lock half is gone (`Mutex<Vec<Option<...>>>` → `UnsafeCell<Vec<Option<...>>>` — Aether-emitted code is single-threaded by construction; the lock served no purpose). The remaining ~3,500 µs/iter at 1024³ on the per-call path is the take→drop-Some→reconstruct-Some pattern interacting with `CudaSlice`'s `Arc<CudaDevice>` refcount and possibly cudarc's per-call workspace selection. Two pinned-roadmap fixes: (a) skip the take/put using raw pointers from `UnsafeCell::get` directly into the gemm trampoline, (b) hold buffers in aether-emitted code and reuse — which is the natural training-loop pattern anyway. Both close the per-call gap mechanically; the underlying compute path is already competitive.

**Bench infra learnings** (worth keeping for future bench work):
- The local candle fork at `J:\candle-src` is the right reference, not crates.io candle. Crates.io 0.7/0.9 hit `cudafe++ Host compiler targets unsupported OS` against the local CUDA 12.6 + VS 17.13 combo; the fork's 0.10.2 is past that cutover. The user's `J:\candle-src\build-cuda.bat` is the canonical build recipe — `bench/matmul_micro/run_candle.bat` mirrors it.
- Aether's default Rust toolchain is GNU; candle-kernels' MSVC-compiled .o files reference `__security_check_cookie` / `__GSHandlerCheck` which mingw ld can't resolve. Pin the bench cargo project to `+stable-x86_64-pc-windows-msvc --target x86_64-pc-windows-msvc`.

Custom kernels for `cross_entropy` / `cross_entropy_backward` / `adamw_step` are the remaining cuBLAS-doesn't-do-that work — JIT them via `cudarc::nvrtc::compile_ptx` and load with `device.load_ptx`. Required for end-to-end GPU training and for the more-honest "framework overhead amortised across many ops" bench. PyTorch sibling at `bench/train_tiny/torch/` is the third leg.

Audit clean. **28/28** runtime end-to-end tests pass. **#23 (struct field access)** + **#24 (self-hosted PE32+ writer, first cut)** + **f32↔f64 conversions** landed this session.

`tests/runtime/struct_fields.aether` declares a mixed-type struct, assigns each field, reads them back, exits 42. Each field gets its own stack slot under a synthetic `name.field` key; no special-case struct value class.

`--emit=pe-bin` produces a runnable Windows .exe via `aether_asm/src/pe.rs`. **Multi-DLL imports + per-symbol indirect-jmp thunks + multi-import IAT all work.** New `pe::build_full_exe` accepts `(user_text, rdata, imports, external_call_sites)`, generates one 6-byte `jmp qword ptr [rip+disp32]` thunk per external symbol, lays out a multi-DLL `.idata` (descriptors + per-DLL ILT + per-DLL IAT), and patches each external CALL site to point at its thunk. Three DLLs supported in the symbol→DLL map today (`aether-asm` source): `kernel32.dll` (ExitProcess implicit), `msvcrt.dll` (puts/printf/fwrite), `aether_rt.dll` (every `aether_*` symbol). Three new runtime tests opt in via `// build-mode: pe-bin`: `pe_exit_42.aether`, `pe_arith.aether`, `pe_hello_msvcrt.aether` (the last one proves multi-DLL: kernel32 + msvcrt).

Stack-alignment trap encountered + fixed: `sub rsp, 40` in the entry stub left rsp at `8 mod 16`, violating the MS x64 ABI for the inner CALL into `kernel32!ExitProcess` (STATUS_ACCESS_VIOLATION); now `sub rsp, 32`.

**`aether_rt.dll` integration RESOLVED via `runtime_pe/`.** The full `runtime/` cdylib's DllMain pulled Rust std init paths (HashMap hasher seed → bcryptprimitives.dll!ProcessPrng) which AVed when loaded into our minimal PE process — the loader hadn't run bcryptprimitives' DllMain yet at that point. **Fix**: a slim sibling crate `runtime_pe/` (same C ABI) built `no_std` + `panic=abort` against just `core` + `libm` + direct `kernel32` extern decls (`HeapAlloc/HeapFree/GetStdHandle/WriteFile`). Output: `aether_rt.dll` (replaces the full crate's old cdylib output, which is now disabled in `runtime/Cargo.toml`). Imports: only `KERNEL32.dll` and `msvcrt.dll`. DllMain is the libgcc default (a clean no-op); no Rust runtime init.

Two remaining libm-on-windows-gnu traps the slim crate sidesteps:
1. **f64 libm entries (`log/sqrt/cos/sin/powf` 64-bit) AV in their SAVE_XMM6+ prologues** when called from a freshly-loaded DLL — `movaps %xmm6, disp(%rsp)` lands on a misaligned slot. The f32 variants of `logf`/`expf` work; `sinf`/`cosf` we don't actually need any more (uniform-init substitute for Box-Muller). Caused by libm-on-windows-gnu, not our PE writer or aether-emitted call sites.
2. **`libm::powf` AVs the same way.** Replaced with hand-rolled exponentiation-by-squaring — `step` in AdamW's bias correction is always a small integer.
3. **`libm::sqrtf` likewise.** Replaced with hardware SSE2 `sqrtss` via inline asm — single instruction, no prologue, no alignment trap.

`tests/runtime/pe_train_tiny.aether` is the audit witness — same model + hyperparameters as `train_tiny.aether`, exits 0 iff loss decreased. The `// build-mode: pe-bin` directive picks the self-hosted linker path; the audit harness now copies `target/debug/aether_rt.dll` next to the .exe so the Windows loader's SafeDllSearchMode finds it.

f32↔f64 conversions wired via `cvtss2sd` (`F3 0F 5A /r`) and `cvtsd2ss` (`F2 0F 5A /r`); `f64(f32_val)` and `f32(f64_val)` builtin casts now succeed instead of erroring out. Test: `tests/runtime/f32_f64_cast.aether`.

**`docs/BENCHMARKING.md`** captures the full plan for proving Aether wins on CPU/GPU/inference vs PyTorch and candle — three-runtime sibling layout under `bench/`, axis matrix, pinned versions, honest expectations, what's blocked on #25.

**Wiki docs**: a subagent was dispatched to write per-item writeups for items 1-22 of the critical path at `J:\llm-wiki\projects\Aether\`. First run got 1-10 before rate-limiting; second run was dispatched for items 11-22 + the top-level `Aether.md` index page (status: in flight at end of session).

Also written this session: `docs/BENCHMARKING.md` — the full plan for proving Aether wins on CPU/GPU/inference vs PyTorch and candle. Blocked on #25 (cuBLAS bodies) for any meaningful GPU number; CPU baseline is honest only after a real BLAS / AVX matmul lands.

A subagent was dispatched to write per-item wiki docs at `J:\llm-wiki\projects\Aether\` for items 1-22 of the critical path. It got through items **01-10 before rate-limiting** (`01-Compiler Scaffold.md` through `10-Trainer Crate.md`). Items 11-22 + the top-level `Aether.md` index page are still TODO; resume with another subagent dispatch once the limit resets, briefing it on what's already done and what's missing.

The canonical numbers are whatever `scripts\audit.ps1` prints — that's the source of truth, not this file. Re-run it before claiming anything.

## Headline this session

End-to-end training **from a `.aether` source** through our own compile chain. Single-layer linear classifier; 50 steps of forward (matmul → softmax → CE) + backward (CE-backward → matmul-backward-rhs) + AdamW. Loss `1.649245 → 0.006114` deterministically. The .aether file at `tests/runtime/train_tiny.aether` is now part of the audit's runtime suite.

To get there this session: 5+-arg FFI passing (MS x64 stack-spill at `[rsp+32+8*(i-4)]`), a runtime allocator + Box-Muller normal init + label fill + per-step loss printer (`aether_alloc_f32`, `aether_init_normal_f32`, `aether_fill_labels_i32`, `aether_print_loss`, plus matching free), a critical bug fix where prologue `subq $imm` overflowed `i8` for any frame ≥128 bytes (now auto-selects between `SubRegImm8` and a new `SubRegImm32`), and three encoder additions (`MovRspDispFromReg`, `MovssXmmToRspDisp`, `MovsdXmmToRspDisp`).

## What Was Done This Session

### Core deliverable: #22 expansion + f64 (item #22 on the critical path is now fully complete)

- `aether_asm/src/encode.rs`
  - 11 new f64 SSE2 instruction variants (Movsd*/Add/Sub/Mul/Divsd, Ucomisd with `66` prefix).
  - 4 int↔float cast variants (`Cvtsi2ss/sd RegToXmm`, `Cvtss/sd2si XmmToReg`) using `F3`/`F2` REX.W prefixes.
  - 2 new encoder unit tests (`sse_double_encodings`, `cvt_int_float_encodings`) byte-exact vs Intel SDM.
- `aether_asm/src/parse.rs`
  - `.quad` directive (8-byte little-endian for f64 constant tables).
  - `movsd`/`addsd`/`subsd`/`mulsd`/`divsd`/`ucomisd` mnemonics.
  - `cvtsi2ssq`/`cvtss2siq`/`cvtsi2sdq`/`cvtsd2siq` mnemonics.
  - `synthetic_text_size` updated for every new variant (the silent-rel32-corruption gotcha).
- `compiler/src/codegen/asm/mod.rs`
  - `TyKind::F64` variant; `from_ty` recognises `"f64"`; `is_float()` helper.
  - `Locals` gained `f64_consts: Vec<f64>`, `default_float: Option<TyKind>`, `sigs: HashMap<String, TyKind>`, `local_fns: HashSet<String>` (for fn-name → `aether_<name>` mangling at call sites).
  - `intern_f64` for per-fn-unique `.LD_<fnname>_<n>` labels; `.rdata` emits f64s via `.quad 0x{:016x}`.
  - `emit_fn` now plumbs declared return type and skips the `xorl %eax, %eax` default for f32/f64-returning fns; param prologue spills incoming `{rcx,rdx,r8,r9}` (int) or `xmm0..3` (float) into stack slots and records type info.
  - `emit_expr_value`'s float pipeline factored to share between F32 and F64 (mnemonic-table dispatch).
  - `Expr::Call` builtin casts: `f32(x)`, `f64(x)`, `i64(x)` lower to `cvtsi2ss/sd` or `cvtss/sd2si`. Identity casts are no-ops; f32↔f64 explicitly rejected (would need `cvtss2sd`/`cvtsd2ss`).
  - `Expr::Call` MS x64 arg dispatch by TyKind: int → `{rcx,rdx,r8,r9}[i]`, f32/f64 → `xmm{i}`. Returns the callee's actual TyKind from the program-wide sig map (was hard-coded to Int).
  - User-fn call sites mangle to `aether_<name>` so source-level `add(2.5, 4.5)` resolves to the correct symbol.
- `runtime/src/lib.rs`
  - Test-only FFI surface: `aether_test_add_f32`, `aether_test_add_f64`, `aether_test_f32_to_i64`, `aether_test_f64_to_i64`, `aether_test_mix_if(i32, f32, i32) -> f32`. Real ops never go through these.

### Test coverage (6 new runtime cases)

- `f32_return.aether` (exit=7) — user fn `add(a: f32, b: f32) -> f32` called from main; cast result back to int as exit. Exercises f32 params + f32 return.
- `f32_cast_round_trip.aether` (exit=42) — `i64(f32(42))` round trip; covers `cvtsi2ssq` + `cvtss2siq`.
- `f64_arith.aether` (exit=42) — `((10.0 * 4.5) - 3.0) / 1.0 > 41.5`, mirrors `f32_arith` with f64 ops.
- `f64_cast.aether` (exit=42) — `i64(f64(42))` round trip; covers `cvtsi2sdq` + `cvtsd2siq`.
- `ffi_add_f32.aether` (exit=4) — calls `aether_test_add_f32(1.5, 2.5)` through libaether_rt. Exercises f32 args in xmm0/xmm1 + f32 return in xmm0.
- `ffi_mix_args.aether` (exit=21) — calls `aether_test_mix_if(2, 3.0, 5)`. Exercises mixed-class arg slots: int (rcx), f32 (xmm1), int (r8).

## Current State

### Working (verified by `scripts\audit.ps1` this session)

- Full Aether-only compile chain: `aetherc --emit=aether-bin` → x86-64 asm (ours) → COFF .obj (`aether_asm`, ours) → linked .exe.
- Aether language surface compiled by the asm backend: `let` (with optional type annotation), `let mut`, `Bin::Assign`, ints + arithmetic (Add/Sub/Mul/Div/Mod with idivq+cqo), comparisons (Eq/Ne/Lt/Gt/Le/Ge → bool), unary `-x` and `!x`, `if/else`, `for i in lo..hi`, `while cond`, `break`, `continue`, `&local`, multi-arg FFI calls, `println(STR)`, **f32 + f64 literals + arithmetic + ucomi[s|d] compares + builtin casts + fn params + float return values**.
- libaether_rt linkage from `--emit=aether-bin`: extern fns named `aether_*` resolve. Verified by `ffi_self_check`, `ffi_tape_push`, `ffi_buffer`, `for_ffi_tape`, `nested_loops`, `ffi_add_f32`, `ffi_mix_args`.
- User-defined fns can be called from other fns (linker mangling to `aether_<name>`).
- AetherLM-Nano trains on CPU through libaether_rt: 5.564 → 1.679 in 40 steps.
- 42/42 unit tests pass; 9/9 golden artifacts match; 8/8 conformance cases pass; **23/23 runtime cases pass**.
- SLOC ~8614 total / ~7022 code.

### Stubbed / explicitly Phase-N

- `aether_op_*` runtime bodies are real f32 CPU implementations; cuBLAS/cuDNN swap is Phase 1.
- `aether_op_all_reduce_sum_f32` and the higher-level `aether_dist_all_reduce` still no-op (`/* Phase 2 — NCCL */`).
- The `trainer/` Rust crate is bootstrap; it's what aetherc Phase 1 emits from `examples/aether_lm.aether`.
- `aether_asm` only encodes the instruction subset aetherc emits today (~45 mnemonics). Missing: most general addressing modes, all xmm8–xmm15, all int-vector / AVX. Not blocking; widen on demand.
- The system linker is the last external tool in `--emit=aether-bin`. Phase-5 self-hosted PE32+ writer drops it.

### Not yet wired

- f32 ↔ f64 conversions (`cvtss2sd` / `cvtsd2ss`) — explicitly rejected by `emit_cast` with a clear error.
- Struct field access (`x.field`).
- Arrays (`[T; N]`) and `lhs[i]`.
- Nested calls in args (the asm backend rejects).
- xmm8–xmm15 (only xmm0–7 in the encoder today).
- 5+ FFI args (no spill to stack yet).

## Blocking Issues

None.

## What's Next (priority order)

The CLAUDE.md "Critical Path" section lists 27 numbered steps; items 1–22 are done (#22 fully expanded this session: f32 returns, FFI float args, casts, f64). Live items:

1. **#23 — struct field access** (`x.field`). Layout: each struct is a contiguous block of f32/i64 slots in the same arena as the struct local. `count_locals` needs to recurse into struct types and sum their slot counts. `Expr::Field { recv, name }` looks up the field's offset and emits a slot read at the right disp.

2. **Arrays** (`[T; N]`) and `lhs[i]`. Stack-allocated, fixed size known at lex. Arms the runtime test `examples/aether_lm.aether` to pass real f32 buffers to `aether_op_*`.

3. **f32 ↔ f64 conversions**. `cvtss2sd` (`F3 0F 5A /r`) and `cvtsd2ss` (`F2 0F 5A /r`). Drop the rejection in `emit_cast`. Useful once the trainer wants mixed-precision.

4. **Nested calls in args**. Currently the asm backend rejects them; would need to spill outer arg-regs around the inner call. Lifts a real ergonomic limitation.

5. **#24 — self-hosted linker.** PE32+ writer in `aether_asm/`: DOS stub, PE/COFF headers, `.idata` import table for msvcrt's `puts` + libaether_rt's `aether_*` symbols, base-relocations, IAT. After this lands the toolchain has zero external deps for static binaries.

6. **#25 — real cuBLAS/cuDNN backend in `runtime/`.** Replace each `aether_op_*` body with a CUDA implementation. The Rust crate stays a thin shim. Use `cudarc` or hand-rolled FFI; differential-test against the (now-deleted) PyTorch oracle before swapping.

7. **#26 — first real training run on 3070 Ti**, once #23 + arrays + #25 land. Compile `examples/aether_lm.aether --emit=aether-bin`, link with the cuBLAS-backed runtime, run on the 3070 Ti, assert loss curve matches expectations.

8. **#27 — self-host the compiler.** Rewrite `compiler/`, `aether_asm/`, `runtime/` in Aether. Drops Rust from the entire stack.

## Notes for Next Session

- **Run `scripts\audit.ps1` first.** Don't take any number in this file at face value; the audit prints the live numbers.
- **The audit is the truth.** Every claim must be backed by an audit dimension. The audit caught a leftover `unreachable!()` in the codegen this session — that's exactly its job.
- **Rebuild golden expected files only when codegen changes are intentional.** Run `aether-audit --update-golden` to re-prime — the next audit will diff against the new files. (None needed this session: my changes only added new code paths; existing examples produce the same asm.)
- **`aether_asm` instruction-size tables in `parse.rs::synthetic_text_size` MUST stay in sync with the encoder.** A mismatch silently corrupts forward-jump rel32 displacements. Every new `Instr` variant needs an entry there. (15 new variants this session — all wired.)
- **Comments are stripped at lex time, irreversibly.** No `--keep-comments` flag exists. Reaffirmed multiple times — do not add a debug escape hatch.
- **Bootstrap reality:** Rust is the implementation language for the compiler, the assembler, the runtime, and the trainer. Aether self-hosts in Phase 5 (#27). Until then, "no language deps" means "no Python, no candle, no torch, no JAX, no XLA" — Rust-as-bootstrap stays.
- **The user's candle fork lives at `J:\candle-src`.** Production-grade work. Aether **does not depend on it**; Aether's runtime calls cuBLAS/cuDNN directly via C ABI. The candle fork is informational — it tells us which ops the runtime needs to expose.
- **Smoke runs the AetherLM-Nano training** end-to-end through libaether_rt: `scripts\smoke.ps1`. If you change the runtime ops, watch the loss curve there.
- **Don't run subagents to "explore" before reading this file + CLAUDE.md.** The prior-session knowledge is dense. Read both, then act.
