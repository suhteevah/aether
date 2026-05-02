# Aether

AI-native systems language. Close to the metal, LLM-readable, comments stripped at lex time.

See `SPEC.md` for the full spec, war doc, and roadmap. See `CLAUDE.md` for coder/agent instructions, and `HANDOFF.md` for the latest session state.

## Build

```
cargo build --workspace --release
cargo test  --workspace
```

## Audit

```
powershell -ExecutionPolicy Bypass -File scripts\audit.ps1
```

Single-command honesty audit: SLOC per crate, every stub / panic / unsafe / `Phase N` marker surfaced with file:line, golden-artifact diffs, language-conformance suite (positive samples must check clean, negative samples must fail with a specific `AE####` code). Run it before claiming any work is done.

## Train a model

AetherLM-Nano (~85K params) on a synthetic corpus, CPU only, no framework deps:

```
.\target\release\aether-train.exe --config nano --steps 60 --batch 8 --seq 32 --lr 3e-3
.\target\release\aether-infer.exe --ckpt checkpoints\aether_lm --prompt "the quick" --max-new 60
```

Loss drops from ~5.5 to ~0.8 in under 10 seconds. Sampled text reproduces fragments of the training corpus. **Every tensor operation goes through `runtime/src/lib.rs`'s C-ABI surface — no candle, no torch, no framework.** When aetherc Phase 1 lands, the same binary will be emitted directly from `examples/aether_lm.aether`.

## Compile to a native binary through Aether's own assembler

```
target\release\aetherc.exe examples\00_hello.aether --emit=aether-bin -o hello.exe
.\hello.exe
```

This walks: `.aether` source → x86-64 AT&T assembly (aetherc backend) → 252-byte COFF .obj (`aether_asm/`, our own x86-64 instruction encoder + PE32+ writer) → linked .exe. **No LLVM, no C compiler, no GAS.** The system linker is the last external tool — replaced in Phase 5.

You can also stop at intermediate stages:

```
target\release\aetherc.exe examples\00_hello.aether --emit=asm
target\release\aether-asm.exe hello.s -o hello.obj
```

## Inspect the language

```
target\release\aetherc.exe examples\aether_lm.aether --check
target\release\aetherc.exe examples\aether_lm.aether --emit=mir
target\release\aetherc.exe examples\aether_lm.aether --emit=llvm-ir
```

## Layout

* `compiler/` — `aetherc` (Rust, bootstrap). Lexer (comment-stripping), parser, AST, MIR + autodiff graph, LLVM text emitter, C fallback, **and direct x86-64 assembly backend**.
* `aether_asm/` — our own x86-64 instruction encoder and Windows PE32+ COFF writer. Phase 5 rewritten in Aether.
* `runtime/` — `libaether_rt`. Thin C-ABI shim with **real f32 CPU implementations** of matmul, gelu, softmax, layer_norm, scaled-dot-product-attention, cross_entropy, AdamW, and all their backwards. Phase 1 swaps each body to cuBLAS / cuDNN / NCCL — symbol surface stays identical. `runtime/ABI.md` is the contract.
* `trainer/` — Rust binaries (`aether-train`, `aether-infer`, `aether-prepare`) that call **only** runtime symbols. What aetherc Phase 1 will emit from the Aether source.
* `stdlib/` — `.aether` source for the language stdlib (`ops`, `optim`, `nn`, `tensor`). Every primitive op is an `extern fn` resolving to a runtime symbol.
* `examples/` — `.aether` programs: hello, matmul, distributed training, LLM serving, and `aether_lm.aether` (AetherLM-Tiny in pure Aether).
* `docs/`, `SPEC.md`, `CLAUDE.md`, `HANDOFF.md` — design + session state.

## Philosophy

Aether is the language; cuBLAS/cuDNN/NCCL are the bare-metal targets. Nothing in between. No PyTorch, no candle, no JAX, no XLA. The Rust runtime is a bootstrap concession — Aether self-hosts in Phase 5.

---

## Support This Project

If you find this project useful, consider buying me a coffee! Your support helps me keep building and sharing open-source tools.

[![Donate via PayPal](https://img.shields.io/badge/Donate-PayPal-blue.svg?logo=paypal)](https://www.paypal.me/baal_hosting)

**PayPal:** [baal_hosting@live.com](https://paypal.me/baal_hosting)

Every donation, no matter how small, is greatly appreciated and motivates continued development. Thank you!
