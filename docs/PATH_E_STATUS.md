# Path E — Self-host compiler — status

**Date:** 2026-05-20
**Current state:** Bootstrap step 10 shipped; step 11 attempt blocked on Aether-compiler stack-frame bug surfaced when functions take 8 args.

## What's already shipped (bootstrap steps 1-10)

A "baby aetherc" written in Aether-source that reads a `.miniaether` file from disk and emits AT&T x86-64 assembly text to a `.s` file. The existing `aether-asm` crate then assembles the .s to a COFF .obj, and the system linker links it.

Concretely, `examples/aetherc_self_emit_asm.aether`:
- Lexer in Aether: handles `let` / idents / numbers / `=` / `;` / `+` / `-` / `*` / EOF
- Pratt parser in Aether: factor → term (×) → expr (+/-)
- Asm emitter in Aether: per-let frame slot, push/pop, movq/imulq/addq/subq
- Input: `let a=7; let b=6; let c=100; c-a*b+a-b-17` → 752 bytes of asm → exit 42

The whole middle stage (lex + parse + asm emit) runs entirely in compiled Aether source.

## What step 11 should add

Conditional control flow: `if cond { ... } else { ... }` expressions with comparison operators (`<`, `>`, `<=`, `>=`, `==`, `!=`). The output asm needs `cmpq`/`setcc`/`movzbq` for comparisons and `cmpq $0/je/jmp` for the if-control-flow plus emitted labels (`.L_else_N`, `.L_endif_N`).

## Blocker discovered

When emit_factor's signature grows from 7 args (step 10) to 8 args (step 11 needs an extra `label_p: i64` so all recursive calls share the label counter), the **compiled program access-violations on return from emit_factor** even when the new code paths (TOK_IF) are not exercised. The crash reproduces on the original step-10 input (`let a=7; ...`) once the function signatures get an 8th arg.

Bisection findings:
- 8th-arg added but no other change → crash
- Debug prints inside emit_factor execute correctly through the entire NUM branch (read tokens, compute acc, emit asm, return)
- Crash occurs at the `popq %rbp; ret` epilogue, suggesting the saved %rbp or return address on the stack was clobbered earlier in the function

Likely cause: the asm backend's call-prep for 8 args writes outgoing-arg slots at offsets that fall inside the local-variable region of the caller's frame, OR the frame allocation `subq $N, %rsp` is sized too small to cover both locals AND the 8-arg outgoing slot area. Either way it's a real bug in `compiler/src/codegen/asm/mod.rs` around how MS x64 outgoing-arg space is reserved.

Memory `asm_backend_known_gaps.md` lists several known asm-backend gaps; this one is new and should be added.

## Workarounds

1. **Static-state approach**: Reserve the first 8 bytes of an existing buffer (e.g., the `toks` buffer) for the label counter. All recursive functions get it implicitly via the toks ptr they already pass. Cost: shift token indexing by 8. Risk: token-buffer reads (`read_tok`) may break if not carefully audited.
2. **Two-pass approach**: First pass counts how many `if`s the program has; allocate that many fixed labels up front and number them in source-order. Second pass emits using pre-computed labels. Cleaner but more code.
3. **Fix the compiler bug first**: investigate `compiler/src/codegen/asm/mod.rs` to find where outgoing-arg space is reserved and adjust the frame layout. ~M-L effort.

## Recommendation

Step 11 is unblocked once one of the workarounds is in place. The cleanest fix is option 3 (fix the underlying bug so future bootstrap steps don't need awkward workarounds), but it requires diving into Aether's own codegen which is multi-day work itself.

For this session: investigation done, blocker captured, no production changes. Resume when:
- The asm-backend bug is fixed (long-term clean), OR
- A small-scope alternate next-step for Path E is identified (e.g., step 11 = add `==/!=` operators only, no if/else, since comparisons don't need 8 args)

## Next-action options for whoever picks this up

| Move | Effort | Outcome |
|---|---|---|
| Fix the 8-arg frame-layout bug in `compiler/src/codegen/asm/mod.rs` | M-L | Unblocks step 11 cleanly; benefits any future fn with 5+ args |
| Step 11-lite: comparison ops only, no if/else | S | Lower-effort progress; doesn't need 8 args |
| Skip step 11; jump to a different Path E item (E1: self-hosted parser) | XL | Bigger ambition; might surface more compiler bugs |
| Move to a different Path entirely (D matt-voice deploy, C tensor stack) | — | Park Path E |
