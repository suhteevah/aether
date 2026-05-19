# Aether — Session Handoff

## Last Updated
2026-05-19 (Phase 19 advance — FR-19.10 chat template renderer)

## Project Status
🟢 **Audit: 155/196 (79%) roadmap items witnessed.** Phase 19
(serving stack): 1/16 → 2/16 (12%). 0 errors, all workspace tests
green (41 runtime tests now). matt-voice's serving-deploy critical
path now has both the BPE tokenizer + the chat template renderer in
place — only the tokenizer.json + chat_template.jinja LOADERS
(FR-19.{9,10}-extra) and the HTTP/TLS stack (FR-19.{1,2,3}) remain.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  — unchanged
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   20/20 witnessed (100%) — unchanged
Phase 18:    9/11 witnessed (81%)  — unchanged
Phase 19:    2/16 witnessed (12%)  ← +1 (FR-19.10 chat template)
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    155/196 (79%)
```

Workspace tests: 134 pass (+1 `chat_template_llama3_shape`).
Honesty scan: 0 todo / 0 unimplemented / 4 known carry-over stubs.

## What Was Done This Session

### FR-19.10 — Jinja-lite chat template renderer

**Why now**: matt-voice's Qwen2.5 chat template uses the same
shape as Llama-3 (for-loop over messages + dot access + if-guard
on add_generation_prompt), so this rounds out the
matt-voice-relevant Phase 19 surface — together with FR-19.9
(BPE), the deploy now has the two algorithmic pieces it needs.

**`runtime/src/lib.rs` additions:**
- `struct ChatTemplateCtx { vars: HashMap<String, String>,
  messages: Vec<(String, String)> }` — same UnsafeCell+Sync handle
  table pattern as the other heap extras.
- 5 extern "C" fns: `aether_template_new` / `_free` / `_set_var`
  (scalar lookup) / `_push_message(role, content)` / `_render`.
- Render engine: state-machine byte-walker supporting
  `{{ var }}`, `{{ msg.field }}` (only `.role` and `.content`
  resolve, against the current for-loop message),
  `{% for msg in messages %} ... {% endfor %}` (only `in messages`
  accepted), and `{% if var %} ... {% endif %}` (truthy = non-empty
  string, not "0", not "false").
- `find_matching_block` balances nested for/if via a single depth
  counter so loops + conditionals can interleave cleanly.

**Unit test** `chat_template_llama3_shape` exercises both:
- 2 messages + `add_generation_prompt="1"` → byte-exact match
  against `<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n\nhello<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>\n\n`.
- 1 message + `add_generation_prompt` unset → trailing assistant
  header omitted; output is just the user turn.

Combined into one fn for the same race-safety reason as the BPE
test (shared static handle table; parallel `_new` calls would
race on `Vec::push`).

**Witness** `tests/runtime/chat_template_render.aether`:
- Hand-builds the template bytes via repeated `aether_byte_set`
  (no string-literal → heap-bytes coercion at the FFI boundary
  in Aether today, so each opener / variable / closer goes in
  byte-by-byte).
- Pushes 2 messages, renders, asserts:
  - returned byte count == 116
  - spot-check bytes at offsets 0/19/22/42/43/54/73 (turn-boundary
    starts + content bytes)
- 23388-byte .obj through the full aetherc → aether-asm →
  aether-bin chain. **Exit=42 first run.**

honesty-auditor verified all 6 claims (signatures, grammar
subset, unit test scenarios + byte-exact assertions, witness exit
+ audit advance, non-claims carve-out in BOTH runtime + witness
headers). Zero false claims.

### Bench

Bench-runner skip note appended (template engine; no matmul / SDPA
/ LN path touched). The right place to bench is a future
`bench/chat_template_throughput/` fixture once matt-voice deploys.

## Current State

**Working:**
- 155/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 134 passing.
- Audit: `errors: 0` clean.
- Phase 19 critical path for matt-voice deploy:
  - ✅ FR-19.9 BPE algorithm
  - ✅ FR-19.10 chat template engine
  - ⏳ FR-19.9-extra tokenizer.json loader (small)
  - ⏳ FR-19.1 TLS 1.3 (XL, long pole)
  - ⏳ FR-19.2 HTTP/HTTPS server (L, depends 19.1)
  - ⏳ FR-19.3 /v1/chat/completions (M, depends 19.2)

**Honest scaffold-vs-shipped notes:**
- FR-19.10 ships ONLY the minimal Jinja subset needed for HF chat
  templates. Filters, whitespace-strip markers, else/elif,
  arbitrary expressions, multi-template files are all
  FR-19.10-extra. Both the runtime source AND the witness header
  document this explicitly.
- The Aether-side witness hand-builds template bytes because there's
  no string-literal-to-heap-bytes shorthand. matt-voice's real
  deploy would read `chat_template.jinja` via `aether_read_file`
  and pass the bytes directly — no per-byte assembly needed.
- Two template tests share the static handle table; combined into
  one test fn for race-safety. Same pattern as BPE.

## Blocking Issues

None. Audit reports `errors: 0`. Honesty scan flags 4 known-OK stubs
(unchanged).

## What's Next

`NEXT-UP.md` is the queue. Phase 19 has 14 more items:

1. **FR-19.9-extra tokenizer.json parser (S)** — uses the BPE
   runtime API shipped today. Pure JSON + add_merge loop. Probably
   half a session.
2. **FR-19.10-extra chat_template.jinja loader (S)** — read file +
   pass bytes through the renderer. Trivial.
3. **FR-19.14 auth + rate limit (S, depends FR-19.2)** — gated on
   HTTP server.
4. **FR-19.1 TLS 1.3 (XL)** — the long pole. Multi-session work.
5. **FR-19.2 HTTP/HTTPS server (L, depends 19.1)**.
6. **FR-19.16 (M, gate)** — Llama-3-1B at ≥100 tok/s. Depends on
   17.19 + 19.4 + 19.5. The phase witness.

For matt-voice specifically, the next 2 small unblockers
(tokenizer.json + chat_template.jinja loaders) get the
local-inference path basically ready. After that the work shifts
to HTTP / TLS to make it network-serveable.

## Notes for Next Session

- **Template tests share the static handle table.** When adding
  more chat-template unit tests, EITHER add to the same single test
  fn OR use a Mutex. Same constraint as BPE.
- **Aether's `aether_byte_set` is the FFI workaround** for
  passing arbitrary byte strings to the runtime. When/if Aether
  gains real bytes-literal support, witnesses like
  `chat_template_render.aether` get much shorter.
- **matt-voice / ant-brain artefacts** (`MATT_VOICE_FR.md`,
  `ANTCOLONY_FR.md`) are still in the aether root. Check those
  files before choosing Phase 19 work order.
- **Phase 19 critical path for serving any quantized model**:
  - Tokenizer (19.9 ✅ + 19.9-extra)
  - Template (19.10 ✅ + 19.10-extra)
  - Quant load (17.14 — already shipped Q4_0)
  - HTTP+TLS (19.1 + 19.2) ← the real work remaining
  - OpenAI endpoint (19.3) ← gates on HTTP
- **No Python for tooling.** Same as always.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime: `cargo build -p aether_rt`
- BPE witness: `cargo run --bin aetherc -- tests/runtime/bpe_tokenizer_roundtrip.aether --emit=aether-bin -o scratch/bpe.exe`
- Chat template witness: `cargo run --bin aetherc -- tests/runtime/chat_template_render.aether --emit=aether-bin -o scratch/tpl.exe`
- matt-voice FR list: `MATT_VOICE_FR.md` (root)
- ant-brain FR list: `ANTCOLONY_FR.md` (root)
- v4 FR queue: `NEXT-UP.md`

## Commits this session

```
e5fa443 Path C FR-17.3: conv2d CPU direct-loop reference
976dbce Phase 17 closeout: Q4_0 + FA2 + layer modules f32 + Llama-shaped partial
a8214f6 Phase 18 closeout: NCCL surface + PP/TP/FSDP/ZeRO/overlap/grad_compress sims
499c49e Phase 19 kickoff: FR-19.9 byte-level BPE tokenizer
(pending) Phase 19 advance: FR-19.10 Jinja-lite chat template renderer
```
