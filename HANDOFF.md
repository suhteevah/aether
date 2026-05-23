# Aether — Session Handoff

## Last Updated
2026-05-23 late evening (**Real continuous-batching scheduler — FR-19.5-extra.**
Pool-aware `BatchScheduler` in `runtime/src/lib.rs` that owns per-request page
tables on top of the FR-19.4-extra paged KV primitives.  FFI:
`aether_batch_sched_{new, destroy, admit, record_token, finish, reap,
n_active, pagetable_for, position}` — all returning i64 to dodge the
asm-backend i32 sign-extend gap.  Admit allocates the initial physical
blocks the prompt needs + a page table; record_token advances the position
and lazily allocates a new block when the next write crosses a block
boundary; reap collects finished requests and returns their blocks to the
pool.  Witness `tests/runtime/batch_scheduler.aether` exits 42 — admits
2 requests, exhausts capacity (3rd rejected), records 4 tokens through
req=10, reaps it back to the pool, ends with 1 active and 1 allocated
block (matching req=11's remaining state).

### 2026-05-23 evening (Real paged KV cache landing — FR-19.4-extra)
**Real paged KV cache landing — FR-19.4-extra.** Built on top of
the existing CPU-side block allocator (`aether_pkv_*` from FR-19.4): added per-request
**virtual page table** primitive (`aether_pkv_pagetable_*`) plus two new **GPU
kernels** (`paged_append_kv_devarg`, `paged_attention_seq1_devarg`) that walk the
page table on every K/V read and write.  Kernels compiled in a SEPARATE nvrtc
unit (`aether_paged_kernels`) so they don't apply register-allocator pressure
on the main `KERNEL_SRC` kernels (per the `nvrtc_kernel_unit_pressure.md`
memory).  Parity test (`runtime/tests/cuda_paged_kv_parity.rs`) runs on the
RTX 3070 Ti and proves:
1. With **identity page-table mapping** the paged kernels produce
   bit-identical output (`max_abs_diff < 1e-5`) to the contiguous
   `append_kv_devarg` + `attention_seq1_devarg` reference, across the full
   Qwen2.5 shape (n_q_heads=28, n_kv_heads=4, head_dim=128, cur_seq=7).
2. With a **permuted page table** (`logical [0,1] → physical [2,0]`) the
   attention output is unchanged — proving the kernel correctly walks the
   indirection rather than reading the raw pool layout.
`.aether` audit witness `tests/runtime/paged_kv_pagetable.aether` exercises
the page-table FFI surface (new/set/get/len/destroy + i64-returning get to
sidestep the asm-backend i32-sign-extend gap per
`asm_backend_known_gaps.md`).  Audit clean.  Integration into `QwenSession`
deferred — single-request serving doesn't benefit from page tables, the
payoff is when paged KV combines with the continuous-batching scheduler
(real implementation is FR-19.5-extra).

### 2026-05-23 morning (Path D landing — TLS 1.3 + HTTP/2 + aether-serve HTTPS)
**Path D landing — full TLS 1.3 + HTTP/2 + aether-serve HTTPS interop.**
Pickup from "go on 1" → completed every tractable Path D item.  Highlights:
(1) **aether-serve has a working `--tls` mode**: generates fresh Ed25519 + X25519
on startup via `aether_random_bytes` (BCryptGenRandom on Windows), self-signs
an X.509 cert, drives a `TlsStream` adapter over each accepted TCP socket,
runs HTTP/1.1 on the decrypted bytes.  GET /health, GET /v1/models, and POST
/v1/chat/completions all round-trip through TLS with `openssl s_client`
(RFC-conformant interop verified end-to-end — `Cipher is TLS_CHACHA20_POLY1305_SHA256`
+ `Peer Temp Key: X25519` + cert chain validated + real HTTP/1.1 response body
decrypted).  Schannel-backed curl on Windows refuses Ed25519 sigs
(`SEC_E_ALGORITHM_MISMATCH`) — a known Windows Schannel limitation, not a
server bug.
(2) **TlsClientSession** is now a public streaming state machine (mirrors
TlsServerSession), plus FFI (`aether_tls13_client_*`).  Loopback test
`tls13_client_session_loopback` drives a full handshake + 2 app-data records
through both `TlsServerSession` and `TlsClientSession` end-to-end.
(3) **HTTP/2 + HPACK shipped** (`runtime/src/http2.rs`, ~600 LOC).  Frame
codec (DATA, HEADERS, SETTINGS, WINDOW_UPDATE, PING, GOAWAY, RST_STREAM),
HPACK static table (61 entries), HPACK integer codec, Huffman DECODER
(RFC 7541 Appendix B — full 257-symbol table; decodes
`www.example.com` per §C.4.1).  6 unit tests pass; `.aether` witness
`tests/runtime/http2_hpack.aether` exits 42 via
`aether_http2_self_loopback_smoke`.
(4) **TLS 1.3 record layer made streaming-tolerant** — `TlsServerSession.feed`
now buffers partial records across calls (essential for real TCP delivery
which doesn't honor record boundaries).
(5) **Caught a real Poly1305 bug en route** — RFC 7539 §2.8.2 vector test
exposed unmasked spurious bits in the 5×26→4×32 limb conversion at MAC
finalization (round-trip tests pass because the bug was self-consistent).
Fix: mask each `h_full_i` to 32 bits before the final s-addition.  Without
this fix, openssl rejects with `bad_record_mac` on the first encrypted
record.
(6) **Real RNG primitive** `aether_random_bytes` shipped (BCryptGenRandom
on Windows, `/dev/urandom` on Unix).  No external deps.
Items DEFERRED (multi-session): real GPU-backed paged KV cache + continuous
batching require kernel changes to attention to walk page tables at seq>1
reads; the existing FR-19.4/FR-19.5 simulations stay closed.  Audit 169/196
unchanged (all this is FR-x-extra per the parent-tag convention).)

### 2026-05-22 evening (prior — TLS 1.3 handshake landing)
**TLS 1.3 server-side handshake state machine SHIPPED — FR-19.1-extra.**
`runtime/src/tls13.rs` implements RFC 8446 server end-to-end on top of Aether's
from-scratch crypto: TLSPlaintext/TLSCiphertext record layer with per-record
nonce derivation (IV XOR seq big-endian) + AAD construction, key schedule
(Early/Handshake/Master + 4 traffic secrets per §7.1), handshake message
codecs (parse ClientHello, encode ServerHello/EncryptedExtensions/Certificate/
CertificateVerify/Finished, parse client Finished), minimal X.509 self-signed
Ed25519 cert generator (OID 1.3.101.112, full DER tree), and a
`TlsServerSession` state-machine driver. Profile: TLS_CHACHA20_POLY1305_SHA256
+ X25519 + Ed25519 only; no PSK / 0-RTT / resumption / ALPN. Witness:
`runtime/tests/tls13_loopback.rs` (in-process server + matching client; full
handshake + 2 app-data records round-trip; tamper test rejects flipped server
flight). `.aether` audit witness: `tests/runtime/tls13_handshake.aether`
exits 42 via `aether_tls13_self_loopback_smoke`. FFI surface:
`aether_tls13_server_new/feed/take_outbound/send/is_done/free`. Audit clean.

### 2026-05-22 morning (prior in same day)
**aether-serve generates real Qwen text via HTTP**. End-to-end Qwen2.5-7B hosting on localhost: `aether-serve --gguf <path> --warmup 6` → curl POST `/v1/chat/completions` returns OpenAI-shaped JSON with real generated text at 24 tok/s warm (capped by GPU power state on bursty single-request workloads; same kernels reach 37 tok/s warm in the sustained `qwen25_graph_decode` bench). SSE streaming (`"stream":true`) emits per-token delta chunks. Plus FR-19.1-extra crypto primitives (SHA-256/512, HMAC, HKDF, X25519, Ed25519) — the foundation underneath the TLS 1.3 handshake shipped this evening.

### 2026-05-21 (prior session)
**37.22 tok/s warm** = 124% of llama.cpp on RTX 3070 Ti / Qwen2.5-7B. Standing wins from that session: per-block Q4_K/Q6_K dtype dispatch (NaN fix), fused FFN kernel (gate+up+silu+mul in 1 launch), CUDA graph capture for autoregressive decode (+37% throughput). Investigated speculative decoding (theoretical 1.6-2.6x but needs ~7-8 days of seq>1 kernel work; deferred). Path E step 11 attempt (self-host compiler if/else) blocked on Aether-compiler 8-arg fn bug — documented in docs/PATH_E_STATUS.md.

## Project Status
🟢 **Audit: 169/196 (86%) — 10 of 19 phases at 100%**. matt-voice's
serving-deploy critical path within Aether's language + runtime is
materially complete: GGUF reader + Q4_K_M dequant + cuda routing
live, tokenizer.json + chat_template loaders, SafeTensors multi-
tensor parser, **TLS 1.3 server-side handshake state machine LANDED
this evening (FR-19.1-extra, on top of the crypto primitives from
this morning's session)**. Remaining gates: HTTP/2 + HPACK, real-
world TLS interop (vs curl/openssl), real RNG, cert-chain verifier
(for client-side), and hardware-binding NCCL on cnc 2× P100.

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  ← +3 (FR-15.{1,2,3} earlier today)
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   20/20 witnessed (100%) ← closed today
Phase 18:    9/11 witnessed (81%)  ← +7 (only hardware-blocked remain)
Phase 19:   16/16 witnessed (100%) ← closed today
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    169/196 (86%)
```

Workspace tests: **61 lib tests pass** (up from 59 — +2 crypto: ed25519_rfc8032_test_1, sha512_known_vectors), 134+ workspace integration tests still pass.
Honesty scan: 0 todo / 0 unimplemented / 4 known-OK stubs (unchanged).

## What Was Done This Session (2026-05-22 evening, TLS 1.3 handshake)

User direction: "go on 1" — Path D, full TLS 1.3 handshake state machine
on top of the prior session's RFC-verified crypto primitives.

### `runtime/src/tls13.rs` — full TLS 1.3 server, no external deps

Single-file module (~1100 LOC) implementing RFC 8446 server profile:

| Layer | What it does |
|---|---|
| `record::*` | TLSPlaintext / TLSCiphertext codec; `build_nonce(iv, seq)` = IV XOR right-aligned seq big-endian; `build_aad` = type(0x17) ‖ 0x0303 ‖ u16 len; `seal_record` / `open_record` thin wrappers over `aead_chacha20_poly1305_{seal,open}` (newly exposed Rust API in lib.rs); `parse_plaintext` for cleartext records (ClientHello, ServerHello). |
| `key_schedule::Schedule` | Early Secret = HKDF-Extract(0, 0); Handshake Secret = HKDF-Extract(Derive-Secret(es, "derived", ""), DHE); Master Secret = HKDF-Extract(Derive-Secret(hs, "derived", ""), 0). `derive_secret(secret, label, transcript)` = HKDF-Expand-Label(secret, label, SHA-256(transcript), 32). Per-direction handshake + app traffic secrets all derived per §7.1. `traffic_keys(secret)` = HKDF-Expand-Label(., "key", 32) + (., "iv", 12). `finished_mac(base_key, transcript)` = HMAC-SHA256(HKDF-Expand-Label(base_key, "finished", 32), SHA-256(transcript)). |
| `handshake::*` | `wrap(msg_type, body)` adds the 4-byte (type ‖ u24 len) header. `parse_client_hello` walks `legacy_version=0x0303 ‖ random[32] ‖ sid<u8> ‖ cipher_suites<u16> ‖ comp<u8> ‖ extensions<u16>`; extracts supported_versions (0x002b), supported_groups (0x000a), signature_algorithms (0x000d), key_share (0x0033 — X25519 entry). `encode_server_hello` emits the same with supported_versions(=tls 1.3) + key_share(x25519, server_pub). `encode_encrypted_extensions` = `[0, 0]` (empty list). `encode_certificate` = `u8(0) ‖ u24(list_len) ‖ u24(cert_len) ‖ cert_der ‖ u16(0)`. `encode_certificate_verify` = `u16(sig_alg) ‖ u16(64) ‖ ed25519_sig`. `encode_finished(verify_data)` = the 32-byte VD. `server_cert_verify_signed_content(transcript)` = `0x20*64 ‖ "TLS 1.3, server CertificateVerify" ‖ 0x00 ‖ SHA-256(transcript)` per §4.4.3. Internal `Reader` for zero-copy parsing. |
| `x509::self_sign_ed25519(seed, cn, serial) -> (cert_der, pub_key)` | Builds TBSCertificate as SEQUENCE { version=v3, serial INTEGER, sigAlg, issuer (Name with CN), validity (UTCTime 240101000000Z..490101000000Z), subject (Name with CN), SPKI (Ed25519 OID 1.3.101.112 ‖ BIT STRING(0x00 ‖ pub_key)) } then signs TBS via `aether_ed25519_sign`, wraps Certificate = SEQUENCE { tbs, sigAlg, signature BIT STRING }. Returns DER bytes + the 32-byte pub key for the matching session. |
| `TlsServerSession` | State machine: `ExpectClientHello -> SentServerFlight -> Connected -> Closed`. `feed(input)` parses incoming records by current state; on CH it emits ServerHello (TLSPlaintext) + one encrypted record containing {EE, Cert, CV, Finished}; on encrypted Finished it verifies the MAC, transitions to Connected. `send_app_data(plaintext)` seals into a TLSCiphertext app-data record and appends to outbound. `take_outbound()` drains the response queue. CCS records tolerated and skipped (RFC 8446 §5). |
| `client_for_test::TestClient` | Matching client harness in the same file (used by tests, kept compiled because it's small and gated only by being inside `pub mod`). `build_client_hello_record` crafts a wire-correct ClientHello with our cipher suite/group/sig-alg/key_share. `process_server_flight_and_build_client_finished` parses ServerHello, derives DHE shared, decrypts the encrypted flight, extracts the Ed25519 SPKI from the Certificate, verifies the CertificateVerify signature against the transcript, recomputes server Finished MAC, and constructs the client's own Finished record. |
| FFI surface | `aether_tls13_server_new`, `_feed`, `_take_outbound`, `_send`, `_is_done`, `_free`, and `_self_loopback_smoke` (one-shot in-process loopback returning 42 on success). All `#[no_mangle] extern "C"`, callable from `.aether`. |

### Witnesses

`runtime/tests/tls13_loopback.rs` — two integration tests:
- `tls13_full_loopback` — CH (149 B) → server flight (503 B) → client Finished (58 B) → server `Connected` → 1 server→client + 1 client→server app-data record (37 B plaintext, 59 B on-the-wire each direction). All bytes verified.
- `tls13_tampered_server_flight_rejected` — bit-flip one byte inside the encrypted second record; client must reject (CV signature fail or MAC mismatch).

`tests/runtime/tls13_handshake.aether` — `// roadmap: P19.1`, calls `aether_tls13_self_loopback_smoke()` and forwards its exit code. Exits 42.

Audit: still 169/196 (FR-19.1-extra reuses the parent tag per the
`fr_x_extra_tag_convention.md` convention — depth-over-breadth).

### lib.rs surgery

- Made `chacha20_xor`, `poly1305_mac`, `poly1305_key_gen` `pub(crate)`.
- Added `aead_chacha20_poly1305_seal` and `aead_chacha20_poly1305_open` Rust
  helpers (mirror of the FFI functions but with Vec / Option return).
- Added `pub mod tls13;`.

### Architectural choices worth remembering

- **One encrypted record for the whole server flight**. EE‖Cert‖CV‖Finished
  concatenated as the inner plaintext of a single TLSCiphertext. RFC permits
  fragmentation; we don't, because everything fits within the 16 KB record limit.
- **No ChangeCipherSpec emit**. We tolerate-and-skip incoming CCS (some
  clients send one for middlebox compat) but don't emit one ourselves. RFC
  §5: "MAY send", we choose MAY-NOT.
- **No HelloRetryRequest**. ClientHello MUST come with an X25519 key_share in
  the first message. Real clients all do this for X25519 because the group
  is universally supported.
- **No padding** in TLSCiphertext.  RFC §5.4 makes padding optional.
- **DER builder is direct write** — no full ASN.1 lib. ~100 LOC.

### What's still missing on the D-path

In dependency order from "first to tackle next session":
1. **HTTP/2 framing + HPACK** (M-L). Independent of TLS. SETTINGS / HEADERS / DATA / WINDOW_UPDATE / PING + static HPACK table. h2c over plaintext TCP works as a first slice; h2-over-TLS slots on top of `TlsServerSession`.
2. **Real-world interop** (M). Drive a TLS connection from `curl --tlsv1.3` against `aether-serve` — currently the test loopback proves the bytes are correct against our own client implementation. A round-trip against curl/openssl/rustls would catch any wire-format drift.
3. **Wire `TlsServerSession` into `aether-serve`** (M). Today the binary uses plain HTTP/1.1. The clean integration: per-accept-on-listener spawn → wrap socket → drive TLS handshake → on Connected, plumb decrypted bytes into the existing HTTP/1.1 parser. Pure additive.
4. **TlsClientSession** (M). The client half (which we have inline in `client_for_test`) deserves a clean public API for outbound calls.
5. **Real RNG** (S). Today `server_random` / `x25519_priv` come from the caller; production needs `getrandom`-equivalent. Single-platform Win32 `BCryptGenRandom` works.
6. **Cert chain / root store** (L). Today: self-signed only. Real internet needs a trusted-root verifier (or a CA-signed leaf, with a manual root pre-loaded).
7. **Resumption / 0-RTT / PSK** (XL). Not blocking the D-path proof; multi-session.

The 1100 LOC of `tls13.rs` is the load-bearing core. Items 1-5 are each
similar weight or smaller. Path D is roughly half-done by code volume.

---

## What Was Done Prior in 2026-05-22 (morning — real-LLM hosting)

User direction: "lets try to closeout the rest of the hosting for real llama in aether"
+ "models downloaded now for smoke testing" + "in theory llama.cpp and aether are a switch flip away"
+ explicit scope: "Full Path D (TLS 1.3 + HTTP/2)".

### aether-serve now generates real Qwen2.5-7B text via OpenAI-compat HTTP

`trainer/src/bin/serve.rs` was previously a STUB that returned
`[aether-serve stub: integrate qwen25_autoregressive_cuda forward here]`.
This session wired the real forward chain + tokenizer + JSON + SSE
streaming. End state:

```
target/release/aether-serve.exe \
  --port 8181 \
  --model qwen2.5-7b-instruct \
  --gguf "C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7..." \
  --warmup 6

# In another shell:
curl -X POST http://localhost:8181/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_ids":[9707,11,1879,0],"max_tokens":32}'

→ {"id":"chatcmpl-aether-serve-1","object":"chat.completion",
   "model":"qwen2.5-7b-instruct",
   "choices":[{"index":0,"message":{"role":"assistant",
     "content":" I'm a 25-year-old software developer from the UK,
                and I'm currently working on a project that involves
                creating a web application"},
     "finish_reason":"stop"}],
   "usage":{"prompt_tokens":4,"completion_tokens":28}}
```

Generated IDs `[358, 2776, 264, 220, 17, ...]` are BIT-IDENTICAL to
the prior session's reference `qwen25_graph_decode.rs` test.

Pieces shipped:
1. **`runtime/src/serving.rs`** — `QwenSession` struct factored from
   the reference impl. Owns 28 BlockGpu + KvCache + activation
   buffers + step_args + a captured CUDA graph + the loaded BPE
   tokenizer (vocab 152064, merges 151387) + EOS token id. Methods:
   `new(gguf_path)`, `prefill(&[ids])`, `decode_step(last_id)`,
   `generate(&prompt, n, stop)`, `decode_ids(&ids) → String`,
   `warmup(N)`. Drop frees all GPU handles.
2. **`runtime/src/serving.rs` decode_ids** — uses GPT-2 byte fixup
   (RFC convention `Ġ` ↔ 0x20 etc.) so generated bytes round-trip
   to readable text. Returns "" if tokenizer absent.
3. **`trainer/src/bin/serve.rs`** — JSON body parser (`prompt_ids`,
   `max_tokens`, `stream`), endpoint dispatch (`GET /health`,
   `GET /v1/models`, `POST /v1/chat/completions`,
   `POST /v1/completions`), JSON-escape for content, EOS auto-stop,
   `--warmup N` flag.
4. **SSE streaming** — when `stream:true`, response uses
   `Content-Type: text/event-stream` + chunked transfer encoding,
   emits one `data: {choices:[{delta:{content:"..."}}]}\n\n` per
   generated token, terminates with `data: [DONE]\n\n`. Verified
   via `curl -N`.
5. **`trainer` cuda feature** — new `cuda = ["aether_rt/cuda"]`
   passthrough; `cargo build --release -p trainer --features cuda`
   produces a release binary with cuBLAS + nvrtc routing.

### Performance reality check

The 37.22 tok/s figure from prior session was a SUSTAINED bench
that holds the GPU at P2/1950MHz across N decode steps. The serve
binary's bursty workload pattern (request → idle → request) lets
the GPU drop into P5/P8 power states between requests (210 MHz)
and the FIRST request after idle takes ~200 ms per token while
the GPU climbs back to P2.

Mitigation: `--warmup N` runs N synthetic decode steps on startup
to drive the GPU into P2 and pre-capture the graph. Subsequent
requests within a few seconds of the warmup (or each other) sustain
24-37 tok/s. Long-idle gaps still trigger the cold-clock penalty.

The right production fix is `nvidia-smi -pm 1` (persistence mode)
+ `nvidia-smi -ac <mem>,<gfx>` (lock clocks) — admin commands
outside Aether's scope. Documented but not enforced.

### FR-19.1-extra: TLS 1.3 crypto primitives (foundation)

Path D requires a full TLS 1.3 handshake stack from scratch
(per Aether's no-external-deps rule). This session shipped the
crypto foundation, all RFC-verified against test vectors:

| Primitive | Source | Test vector |
|---|---|---|
| SHA-256 | FIPS 180-4 | `abc`, `""`, 56-byte boundary |
| SHA-512 | FIPS 180-4 | `abc`, `""` |
| HMAC-SHA256 | RFC 2104 | RFC 4231 §4.2 test case 1 |
| HKDF-Extract | RFC 5869 | RFC 5869 §A.1 test case 1 |
| HKDF-Expand | RFC 5869 | RFC 5869 §A.1 test case 1 |
| HKDF-Expand-Label | RFC 8446 §7.1 | Shape + determinism smoke |
| X25519 (scalar_mult / derive_public / shared_secret) | RFC 7748 | §5.2 first vector + DH roundtrip |
| Ed25519 (derive_public / sign / verify) | RFC 8032 | §7.1 TEST 1 |

ABI surface (all `#[no_mangle] extern "C"`):
- `aether_sha256`, `aether_sha512`
- `aether_hmac_sha256`
- `aether_hkdf_extract`, `aether_hkdf_expand`, `aether_tls13_hkdf_expand_label`
- `aether_x25519_derive_public`, `aether_x25519_shared_secret`
- `aether_ed25519_derive_public`, `aether_ed25519_sign`, `aether_ed25519_verify`

Internal implementations: radix-2^51 field arithmetic over GF(2^255-19),
twisted-Edwards point ops (X, Y, Z, T extended coords),
double-and-add scalar mult (non-constant-time; FR-x-extra for ct),
sc_reduce via long-division (verify-only path; Barrett-form
constant-time is FR-x-extra). Curve constants (d, sqrt(-1), base
point) computed/derived at runtime from byte forms or self-checked
against `x^2 == -1`.

### Bugs caught in-flight

1. **X25519 fe_from_bytes over-masked limb 4 to 50 bits** — total
   read was 254 bits (off by 1 from input bit 254). Fix: read full
   51 bits for all 5 limbs (255 total; input bit 255 ignored implicitly).
2. **fe_pow_p_plus_3_over_8 final mul was by z instead of z^2** —
   gave z^(2^252 - 3) instead of z^(2^252 - 2). Fix: multiply by z2.
3. **Initial "compute sqrt(-1) via fe_pow_p_plus_3_over_8(-1)" returns 1**
   (since `(-1)^(even exponent) == 1`). Fix: use `2^((p-1)/4)` with
   a hardcoded constant + self-check at first access.
4. **Ed25519 verify subtracted k*A instead of adding** — sign was
   inverted in the negation logic. Fix: drop the negate step, just
   compute `R + k*A` directly.

### Decode-side text round-trip caveat

`aether_bpe_encode` (text → ids) is byte-level and DOES NOT match
HF's GPT-2 unicode-char-level BPE for Qwen. Decode round-trips
fine (proven). For now, clients must send `prompt_ids` (token ids);
the `messages[].content` text path returns 501 with a clear error.
Encode parity is FR-19.9-extra-deeper (multi-session).

## What Was Done Prior Session (2026-05-21)

Twelve commits, pushed to `origin/main`. The arc is matt-voice perf
optimization from the broken NaN starting state to 124% of llama.cpp
on Qwen2.5-7B, plus two investigations (speculative decoding,
self-host step 11) that didn't ship code but produced actionable
plans + memory.

```
399718e fix(matt-voice): per-block Q4_K/Q6_K dtype dispatch
9b5a21e perf(matt-voice): fused FFN kernel (gate+up+silu+mul in 1 launch)
859745d perf(matt-voice): byte-once Q4_K matmul v3 (kept alt, NOT on hot path)
1682bfe docs: HANDOFF + NEXT-UP + BENCH_LEDGER for 27.22 tok/s baseline
7e1804f perf(matt-voice): CUDA graph capture for autoregressive decode -> 37.35 tok/s
5aaf3a4 docs: HANDOFF + NEXT-UP + BENCH_LEDGER for 37.35 tok/s graph baseline
f40d259 perf(matt-voice): small-N matmul kernel explored, not promoted
add5216 perf(matt-voice): revert smallN matmul kernels (regress FFN via nvrtc unit pressure)
a3aa6ef docs: kernel-asm exploration learnings + 37.22 tok/s warm baseline
ef94fa3 docs: speculative decoding investigation -- analysis, not implementation
b23d661 NEXT-UP: speculative decoding investigated, deferred per user direction
62e18aa docs: Path E (self-host) status -- step 11 blocked on 8-arg fn bug
```

### The big win: CUDA graphs (commit 7e1804f)
After the prior session shipped fused matmul kernels and on-device
KV cache + attention to reach 25 tok/s, this session pushed end-to-end
through:
1. **NaN bisect (399718e)**: per-block Q4_K/Q6_K dtype dispatch.
   Qwen2.5-7B Q4_K_M is mixed-precision; V proj and ffn_down switch
   between Q4_K (144 B blocks) and Q6_K (210 B blocks) per layer.
   Hardcoding from block 0's dtype made block 3's V proj read garbage.
   Fix: store dt_v + dt_down per BlockGpu, dispatch matmul kernel by
   stored dtype. Generated IDs `[358, 2776, 264, 220, 17]` match
   cuBLAS reference exactly. **25.5 tok/s.**

2. **Fused FFN kernel (9b5a21e)**: replaces 4 kernel launches per
   layer (gate matmul + up matmul + silu + mul_inplace) with 1.
   Gate and up share x_norm; one kernel computes both, applies
   silu(gate)*up, writes one output. Parity bit-identical
   (`max_diff = 0`). **+7% throughput: 25.5 -> 27.2 tok/s warm mean.**

3. **CUDA graph capture (7e1804f)**: per-token forward gets recorded
   into one CUDA graph at first decode step; subsequent steps replay
   the graph with just a 4-int step_args h2d update. Compresses
   ~370 kernel launches per token into one `cuGraphLaunch`. Three
   pieces:
   - **Device-arg kernel variants** of rope_apply, append_kv,
     attention_seq1 that read pos/cur_seq from device memory
   - **Raw cudarc::driver::sys** bindings to cuStreamBeginCapture_v2,
     cuStreamEndCapture, cuGraphInstantiateWithFlags, cuGraphLaunch
   - **CudaDevice::new_with_stream()** instead of new() — the legacy
     default stream cannot be captured (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED)
   **+37% throughput: 27.2 -> 37.35 tok/s = 124% of llama.cpp's ~30.**

### Kernel-asm exploration (a3aa6ef arc)
User asked to push further on the "assembly aspect" for more tok/s.
Per-shape matmul bench (matmul_per_shape_bench.rs) found:
- K/V proj at 512-out shape runs at 4.3% of peak BW (worst offender)
- FFN at 18944-out runs at 39% (dominant cost, ~9 ms/token)
- Q/O at 21%, down at 29%, lm_head at 38-46%

Tried 3 kernel-level wins:
1. **smallN matmul** (32-thread CTAs for K/V): 1.32x in isolation,
   end-to-end REGRESSION (~5%). SM scheduling fragments when mixing
   CTA sizes. Reverted.
2. **Interleaved FFN gate+up FMA**: 1.02x in isolation, end-to-end
   noise/regression. Reverted.
3. **Byte-once v3 matmul** (deeper from prior session): also slower
   end-to-end. Kept as alternate, not on hot path.

The unifying lesson: adding `__global__` kernels to KERNEL_SRC
regresses the existing actively-used ones by 5-7% via nvrtc unit
pressure (shared register allocation analysis). Removing the smallN
kernels restored the 37.35 baseline. **Treat KERNEL_SRC as load-
bearing — additions are not free.**

Plus discovered GPU boost-clock cold-start phantom: first run after
idle is ~5% slow while clocks ramp 210 -> 1950 MHz. Don't include
the cold run in N-run means.

### Speculative decoding investigation (ef94fa3, b23d661)
User: "Investigate speculative decoding". Empirical bench
(spec_dec_naive_verify_bench.rs) proved: naive verify by re-launching
the seq=1 graph N times scales **linearly** in N (4.00x at N=4).
**Break-even acceptance rate for N=4 is 99.96% — mathematically
impossible.** Speculative decoding requires real seq>1 kernels.

Full architecture analysis in docs/SPECULATIVE_DECODING_INVESTIGATION.md:
6 kernels need seq>1 variants + draft model integration + verification
orchestration. ~7-8 days for production quality. Expected speedup
1.6-2.6x (55-90 tok/s).

User decision: defer. The 37.22 tok/s baseline is already strong;
ship the matt-voice critical path instead.

### Path E step 11 attempt (62e18aa)
User: "go on 5" => Path E self-host compiler. Bootstrap step 10 (a
baby aetherc in Aether-source that emits real x86-64 .s files) is
shipped; step 11 was meant to add `if/else` + comparison operators.

Blocked on a real **Aether asm-backend bug**: 8-arg recursive fn
signatures access-violate at `popq %rbp; ret` epilogue, even on
step-10's working input the moment any fn gets an 8th arg. Bisection
ruled out the new code; the bug is purely about 8-arg signatures and
outgoing-arg space interfering with caller-frame locals.

Investigation captured in docs/PATH_E_STATUS.md with three concrete
next-step options. Memory updated at memory/asm_backend_known_gaps.md.

### New memories captured
- `qwen25_q4km_mixed_precision_per_block_dtype.md` — V/ffn_down dtype varies per layer
- `cuda_graphs_pattern.md` — non-default stream + devarg kernels + raw cudarc sys
- `nvrtc_kernel_unit_pressure.md` — KERNEL_SRC is load-bearing; unused kernels hurt active ones
- `gpu_boost_clock_warmup.md` — 5% cold-start phantom; discard run 1 from N-run means

### Stale-but-still-relevant prior session details

### Path A complete (FR-15.{1,2,3}) — earlier in session
- SSA-driven opt pipeline rewrites AST at --O1
- Regalloc plan drives callee-saved r12..r15 promotion
- AVX2 emit via aether_asm + recognised `__aether_avx2_dot_f32` builtin
- 23/23 honesty-auditor claims verified across the three commits

### Phase 17 closed to 100%
- conv2d CPU direct-loop reference (P17.3)
- Q4_0 GGUF dequant
- FlashAttention v2 (blocked online-softmax, matches naive SDPA)
- Real f32 Linear + LayerNorm witness
- Partial Llama-shape architecture forward (explicit partial scope)

### Phase 18 closed to 81% (only hardware-blocked items remain)
- NCCL FFI surface (single-host fallback; -1 sentinel on ws>1)
- Pipeline-parallel 1F1B sim (matt-voice's 2×P100 unlock per
  MATT_VOICE_FR.md)
- Tensor-parallel column-parallel Linear sim
- FSDP shard+alltoall sim
- ZeRO-1/2/3 staged sharding sim
- Compute/comm overlap sim
- Gradient compression shape

Only FR-18.10 (multi-host RDMA) + FR-18.11 (8-GPU) remain; both
hardware-binding per NEXT-UP §2 PARKED.

### Phase 19 closed to 100%
**13 items in one batch** (paged KV / continuous batching /
speculative decoding / multi-model / tool calling / rate-limit /
observability / vision / speech / ChaCha20-Poly1305 / HTTP/1.1 /
OpenAI shape / WS frame), plus **FR-19.9 BPE tokenizer** + **FR-
19.10 chat template renderer** + the **FR-19.16 partial tok/s bench
at 177 tok/s on Llama-shape** (well over the ≥100 threshold).

### matt-voice deploy pack — 5 FR-x-extras (commit 3283015)
- `cargo build -p aether_rt --features cuda` succeeds on kokonoe;
  libaether_rt.a has 39507 cuBLAS-symbol matches; cuda_train_tiny
  goes from skipped → real GPU train exit=0.
- FR-17.19-extra: SafeTensors multi-tensor parser (n_tensors /
  get_shape / get_dtype with the F32/F16/BF16/I32/I16/U8/I64 enum).
- FR-17.14-extra: Q4_K_M dequant kernel (real ggml 144-byte
  super-block layout; the format matt-voice's Qwen2.5-7B uses).
- FR-19.9-extra: HF tokenizer.json loader (hand-walks vocab + merges
  with explicit HF id preservation — essential for model weight
  indexing).
- FR-19.10-extra: chat_template.jinja file loader (wraps std::fs::read
  → render).
- Plus `aether_copy_cstr` helper: copies NUL-terminated `Expr::StrLit`
  literals from .rdata into heap buffers. Major witness-readability
  win — see `memory/aether_copy_cstr_pattern.md`.

### Real GGUF reader for Qwen2.5-7B (commit 172f423)
**The story**: user said Llama-1B weights were downloaded but
couldn't recall location. Searched extensively — no Llama-3-1B on
this machine. What IS local: matt-voice's actual base model,
Qwen2.5-7B-Instruct Q4_K_M, in ollama's blob store as 4.7 GB at
`C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`.

User picked: use the local Qwen2.5-7B. Built a real GGUF v3 reader
that walks all 339 tensors. Witness verifies tensor 0 is
`token_embd.weight` at dtype 12 (= Q4_K = matt-voice's quant
format). Together with the FR-17.14-extra Q4_K_M dequant kernel
already shipped, Aether can now READ real Qwen2.5-7B weight bytes.

Three primitive layers shipped together:
1. `aether_gguf_open` + 8 walker fns
2. `aether_dequant_q4_k_m` (Qwen2.5's exact format)
3. cuBLAS routing via `--features cuda`

honesty-auditor verdicts across the session: **51/51 claims
verified, zero false** across 9 commits.

## Current State

**Working (deploy — the new headline):**
- **`aether-serve` produces real Qwen2.5-7B text via OpenAI HTTP**
  - `POST /v1/chat/completions` (single-shot JSON OR `stream:true` SSE)
  - `GET /v1/models`, `GET /health`
  - JSON-escape, EOS auto-stop, warmup on startup
  - 24-37 tok/s warm (dependent on GPU power state — see HANDOFF
    "Performance reality check" subsection)
- Generated IDs bit-identical to `qwen25_graph_decode.rs` reference

**Working (matt-voice perf — prior session headline):**
- **37.22 tok/s warm mean on Qwen2.5-7B Q4_K_M / RTX 3070 Ti = 124% of llama.cpp**
- Generated IDs bit-identical to cuBLAS reference: `[358, 2776, 264, 220, 17]`
- Full forward pass via on-device fused matmul + on-device KV cache
  + GPU attention kernel, all wrapped in one CUDA graph that replays
  per decode step
- `runtime/tests/qwen25_graph_decode.rs` is the standing benchmark
- Per-shape diagnostic: `runtime/tests/matmul_per_shape_bench.rs`

**Working (TLS 1.3 crypto foundation — this session):**
- 7 RFC-verified primitives: SHA-256, SHA-512, HMAC-SHA256, HKDF
  (Extract+Expand+Expand-Label), X25519 (derive+shared), Ed25519
  (derive+sign+verify), ChaCha20-Poly1305 (prior).
- ABI: `aether_sha256/512`, `aether_hmac_sha256`, `aether_hkdf_*`,
  `aether_tls13_hkdf_expand_label`, `aether_x25519_*`, `aether_ed25519_*`
- Full TLS 1.3 cipher suite TLS_CHACHA20_POLY1305_SHA256 is now
  buildable from these primitives. Remaining: handshake state
  machine, record layer, X.509 cert handling.

**Working (broader project, unchanged from prior session):**
- 169/196 audit-tagged witnesses pass (no change — crypto adds are
  library funcs not roadmap line items).
- 61 lib tests pass (+2 from prior session: ed25519, sha512).
- 10 phases at 100% (6-14 + 17 + 19).
- `cargo build -p aether_rt --features cuda` succeeds; cuBLAS path live.
- Real Qwen2.5-7B Q4_K_M GGUF readable via `aether_gguf_*` extern surface.
- matt-voice's serving-deploy critical path's LANGUAGE work:
  - ✅ BPE algorithm + chat template engine
  - ✅ tokenizer.json + chat_template.jinja file loaders
  - ✅ Q4_K_M dequant + GGUF reader (and now mixed-precision-aware)
  - ✅ SafeTensors multi-tensor parser
  - ✅ cuda runtime path live (cuBLAS sgemm + nvrtc kernels + CUDA graphs)
  - ✅ Full forward pass through real Qwen2.5 weights at scale
  - ✅ **OpenAI-compat HTTP server returns real text (this session)**
  - ✅ **TLS 1.3 crypto primitives RFC-verified (this session)**
  - ⏳ FR-19.1-extra TLS 1.3 handshake state machine (XL, partial — primitives done)
  - ⏳ FR-19.2-extra HTTP/2 framing + HPACK
  - ⏳ FR-19.9-extra-deeper text→ids encode with GPT-2 unicode BPE
  - ⏳ FR-19.4-extra real paged KV cache (block allocator)
  - ⏳ FR-19.5-extra continuous batching scheduler

**Honest scaffold-vs-shipped notes** (updated):
- Phase 19's FR-19.16 ships a PARTIAL tok/s bench (177 tok/s on
  Llama-shape, NOT real Llama-1B). The witness header carves out
  the partial scope explicitly. The full Llama-1B target is
  FR-19.16-extra.
- The GGUF reader reads bytes only — no forward pass through real
  weights yet. The next gate (weight → dequant → matmul chain at
  every transformer layer) is multi-session.
- The "simulations" in Phase 18 are correctly named `*_simulate_*`
  so the simulation status is visible at the call site. Real
  multi-rank needs FR-18.1-extra libnccl + a second GPU.
- 4 known-OK stubs unchanged (mir/fuse.rs:53, mir/spec.rs:161,
  runtime_pe/src/lib.rs:59 + :443).
- Audit count for FR-x-extra tags doesn't advance (parent tag
  reuses) — design intent, not regression.

## Blocking Issues

None on the matt-voice perf critical path — the 37.22 tok/s baseline
is solid and tested. Remaining deploy items + active blockers:

- **Path E step 11 (self-host if/else)** — blocked on Aether asm
  backend bug: 8-arg fn signatures crash on `popq %rbp; ret`
  epilogue. See `docs/PATH_E_STATUS.md` for the investigation and
  three workaround options. Workaround needed before continuing
  the self-host bootstrap chain.
- **FR-19.1-extra full TLS 1.3** — XL multi-session work for real
  HTTPS serving. Not blocking the local 37 tok/s state.
- **FR-19.16-extra deploy as HTTP server** — composite of TLS +
  HTTP + serving. Multi-session.
- **FR-18.1-extra real libnccl** — needs the cnc 2× P100 box
  (kokonoe is single-GPU).

## What's Next

For "switch flip from llama.cpp": the **basic** hosting path is now
done (single-shot + SSE). The remaining Path D pieces, in dependency
order:

1. **TLS 1.3 handshake state machine** (XL, ~1-2 sessions). With the
   crypto primitives in place, this is mostly:
   - Record layer (TLSPlaintext + TLSCiphertext wrappers — ChaCha20-
     Poly1305 already encrypts the records)
   - Handshake messages: ClientHello, ServerHello, EncryptedExtensions,
     Certificate, CertificateVerify, Finished
   - Key schedule: early_secret, handshake_secret, master_secret +
     derive_secret + 8 per-direction traffic keys (HKDF-Expand-Label
     already shipped — just call it 8 times with the right labels)
   - Self-signed cert generation (Ed25519 keypair + minimal X.509 DER)
   - State machine driver: drive a TLS connection from accept to
     handshake_finished
2. **HTTP/2 framing + HPACK** (M-L). Independent of TLS — h2c
   over plain TCP is fine for the in-flight increment. HEADERS /
   DATA / SETTINGS / WINDOW_UPDATE / PING + static HPACK table.
3. **Real paged KV cache** (M) — block allocator over a fixed GPU
   pool, virtual-page mapping per request. Replaces today's
   single-resident KV per QwenSession. Unlocks multi-request
   serving without growing per-request VRAM.
4. **Continuous batching scheduler** (M) — decode N requests per
   step rather than one. Sits on top of paged KV cache.

Three alternative directions (orthogonal to D-path):

1. **D-path** (TLS / HTTP/2 / batching) — see above.
2. **E-path (self-host)**: fix the 8-arg fn bug in the asm backend,
   then continue bootstrap steps. M-L for the fix, then incremental.
3. **Spec-decode build-out** (deferred prior session): seq>1 kernel
   suite + draft model. ~7-8 days. 60-90 tok/s upside.

### Operational note for next-session smoke testing

When measuring `aether-serve` perf or running benches:
- Run `nvidia-smi -pm 1` and `nvidia-smi --query-gpu=pstate` to
  verify GPU is in persistence mode at P0/P2 before measuring.
- Use `--warmup 6` (or higher) on `aether-serve` to drive GPU
  into P2 before serving real requests.
- Discard the FIRST run after spawn from any N-run mean
  (cold-clock penalty per `memory/gpu_boost_clock_warmup.md`).
- Reference standing bench is `cargo test --release --features cuda
  -p aether_rt --test qwen25_graph_decode -- --ignored --nocapture`
  — this should report 27-37 tok/s on a hot GPU. If it reports
  <10 tok/s, the GPU is in P5/P8 and serve will be slow too.

(Continuing from prior session's option list — unchanged.)

2. **If extending CUDA perf work on matt-voice**: the FFN at
   39% peak BW is the dominant cost. Further gains require either
   PTX/SASS-level kernel work (high risk per this session's
   lessons) or moving to tensor-cores (XL, requires F16 reorg).
   Diminishing returns; ~37 tok/s is already past llama.cpp.

3. **If picking up Path E step 11**: easiest path is the
   step-11-lite variant (comparison ops only, no labels, doesn't
   need 8 args). Real fix is auditing
   `compiler/src/codegen/asm/mod.rs` outgoing-arg-area layout.

## Notes for Next Session

- **The 37.22 tok/s number is WARM mean.** First run after GPU idle
  shows ~35 tok/s while clocks ramp 210 → 1950 MHz. Always
  pre-warm before measuring. `nvidia-smi --query-gpu=clocks.current.graphics`
  confirms boost state per iteration.
- **KERNEL_SRC is load-bearing.** Adding new `__global__` kernels
  to `runtime/src/cuda.rs::KERNEL_SRC` can regress existing active
  kernels by 5-7% via nvrtc shared register-allocation analysis.
  Always re-run `matmul_per_shape_bench.rs` + the full graph
  decode after KERNEL_SRC edits.
- **In-isolation kernel benchmarks lie.** smallN was 1.32x in
  isolation but -5% end-to-end. Always validate via end-to-end
  `qwen25_graph_decode.rs` before promoting.
- **CUDA graph capture requires non-default stream.** `CudaDevice::new()`
  uses the legacy null stream; `cuStreamBeginCapture_v2` rejects
  it. Already switched to `CudaDevice::new_with_stream()` in cuda.rs.
- **Per-block dtype dispatch matters for Q4_K_M**. Qwen2.5-7B has
  V proj and ffn_down switching between Q4_K and Q6_K per layer.
  Hardcoding from block 0 → NaN at first mismatched layer. See
  `memory/qwen25_q4km_mixed_precision_per_block_dtype.md`.
- **bench/qwen25_7b_autoregressive section in docs/BENCH_LEDGER.md**
  is the canonical perf history for this model. Append rows when
  this number moves.

## FR-18.1-extra — Real libnccl cross-card (LANDED)

Aether's runtime now supports REAL cross-GPU NCCL collectives,
verified end-to-end on cnc-server's 2× P100 box.

New surface (gated `--features nccl`):
- `runtime/src/nccl_real.rs` — `aether_nccl_real_init_multi_gpu(n)`
  wraps `ncclCommInitAll`; `_get_handle(rank)` / `_all_reduce_f32` /
  `_comm_world_size` / `_comm_rank` / `_finalize` round out the
  surface. Plus a Rust-side `comm_at(i)` accessor for integration
  tests that need to drive cudarc's typed API directly.
- `runtime/tests/nccl_dual_gpu.rs` — `Comm::from_devices(2)`,
  group_start/end, all_reduce sum: rank 0 sends 1.0s, rank 1 sends
  2.0s, both ranks see 3.0s. Verified on 2× P100.
- `runtime/tests/nccl_dual_gpu_dp_step.rs` — data-parallel
  training step: each rank computes gradient on its own shard
  (rank 0=1.0, rank 1=3.0), all_reduce sum, divide by ws → mean=2.0,
  identical SGD update on both ranks. The matt-voice unlock shape.

nvidia-smi confirmation: a single test process appears on BOTH GPU
UUIDs simultaneously (`bb77bda0...` first P100, `17bd0d20...` second
P100) with ~260+318 MiB allocations — physically proving cross-card
data exchange.

NCCL compatibility note: ollama's bundled libnccl 2.29 dropped sm_60
(Pascal) kernels and fails with "named symbol not found" on the
P100s. Aether links against libnccl 2.21.5+cuda12.4 from the local
fish-speech venv via `/usr/local/lib/libnccl.so.2` symlink. Documented
in NEXT-UP.

## On-device KV cache + attention kernel LANDED (correctness still WIP)

User: "Go on attention correctness". Shipped:

### Kernels
- `append_kv` -- writes new K/V step into the per-block KV cache
  at position `pos`.
- `attention_seq1` -- one warp per Q head, lanes cooperatively compute
  scores via warp-reduce, softmax (max + exp+sum + normalise), then
  aggregate V_cache by softmax weights. Dynamic shared mem sized
  for `cur_seq * 4` bytes.
- Wrappers `aether_op_append_kv_f32_cuda` +
  `aether_op_attention_seq1_f32_cuda`.

### Verification (in isolation)
`runtime/tests/attention_seq1_parity.rs` — Q/K_cache/V_cache with
known values, n_q=28 / n_kv=4 / head_dim=128 / cur_seq=7. GPU
matches CPU reference within **max_diff = 7.15e-7**.

### Wiring into autoregressive_fused
`runtime/tests/qwen25_autoregressive_fused.rs` now uses the real
GPU attention kernel + per-block KV cache (MAX_SEQ=32 device
buffers per block, 28 caches). Speed: **25 tok/s**.

### NaN at block 3: ROOT-CAUSED + FIXED

Bisecting per-op magnitudes inside block 3 revealed
`[V + bias] max_abs=2.192e9 nan=true` — V proj's output was garbage,
not "FP drift accumulating across blocks".

`q6k_blk3_diagnose.rs` failed with `assertion left==right: 12 vs 14`:
**`blk.3.attn_v.weight` is Q4_K (12), not Q6_K (14)**. Qwen2.5-7B
Q4_K_M is **mixed-precision** — V proj and ffn_down switch between
Q4_K and Q6_K per layer:

- V proj Q6_K on blocks [0,1,2,5,9,11,14,17,20,23,24,25,26,27],
  Q4_K on the rest.
- ffn_down Q6_K on [0,1,2,5,7,12,14,17,20,23,24,25,26,27],
  Q4_K on the rest.
- Q/K/O/gate/up always Q4_K; lm_head Q6_K.

`qwen25_per_block_dtypes.rs` enumerates the full table.

The fused Q4_K and Q6_K kernels have **different super-block
layouts** (144 B vs 210 B per 256-elem block) — dispatching to the
wrong one reads completely garbled weights, hence the 2.19e9 / NaN.

**Fix** (commit pending): `BlockGpu` now stores `dt_v: i32` and
`dt_down: i32` captured from `aether_gguf_get_tensor_dtype` at
upload time; `upload_tensor_u8` returns the dtype as a 3rd element.
The forward pass dispatches V proj and ffn_down matmul on the
stored dtype:

```rust
if bw.dt_v == 14 {
    aether_op_fused_q6k_matmul_seq1_v2_cuda(...);
} else {
    aether_op_fused_q4k_matmul_seq1_v2_cuda(...);
}
```

Same dispatch for ffn_down on `bw.dt_down` and for lm_head on `lm_dt`.

**Result** (verified locally just now):
- Block 3 V + bias now `max_abs=1.688e0` (sane, was 2.19e9).
- All 28 blocks produce finite activations end-to-end.
- Generated IDs `[358, 2776, 264, 220, 17]` — matches the
  cuBLAS-routed reference exactly.
- Speed sustained at **25.53 tok/s** (39.2 ms/token, 5 decoded
  tokens after a 4-token prefill at 37.9 ms/token).

llama.cpp's Q4_K_M dispatch table for the same model reports
~30 tok/s on a 3070 Ti; Aether is at **85% of llama.cpp throughput
with sane outputs**. Correctness gap is closed.

Pitfall captured in memory at
`memory/qwen25_q4km_mixed_precision_per_block_dtype.md`.

## End-to-end fused autoregressive: 24 tok/s MEASURED on RTX 3070 Ti

User: "Go till the cuda tuning is complete". Continued the v2 line
through Q6_K fused matmul + end-to-end measurement.

### Shipped in this batch
- **Q6_K fused matmul v2** (`fused_q6k_matmul_seq1_v2` kernel +
  `aether_op_fused_q6k_matmul_seq1_v2_cuda` wrapper). Same warp-
  per-output split-K design as Q4_K v2 but for 210-byte Q6_K
  super-blocks. Used for V proj + ffn_down + lm_head.
- **End-to-end test** `qwen25_autoregressive_fused.rs` wires v2
  kernels into the full Qwen forward + lm_head + sampling chain.

### Q6_K v2 measured perf
```
tensor                       n      k    cuBLAS    v2  speedup  max_diff
blk.0.attn_v.weight        512  3584      26us  55us  0.47x   9.5e-7
blk.0.ffn_down.weight     3584 18944     475us 262us  1.81x   9.3e-5
output.weight (lm_head) 152064  3584    3692us 1580us 2.34x   2.9e-6
```

The **lm_head matmul (largest single matmul) saves 2.1 ms per token**.

### End-to-end measurement (RTX 3070 Ti, release build)

| Stage | Time |
|---|---|
| Upload 28 blocks Q4_K+Q6_K bytes | 0.68 s |
| Upload output_norm + lm_head | 0.75 s |
| Prefill 4 tokens | 158 ms (39.6 ms/token) |
| Generate 5 tokens | 207 ms |
| **Per-token cost** | **41.4 ms** |
| **tok/s** | **24.17** |
| llama.cpp reference (same hw) | ~30 tok/s |
| **Speedup vs prior CPU baseline** | **96× from 0.25 tok/s** |

### Honest caveat: attention stub

The current end-to-end uses a SHORTCUT for attention: at seq=1 the
attn_out is approximated rather than computing real
softmax(Q·K^T) over a growing KV cache. The 24 tok/s figure
reflects the real cost of all matmuls + non-matmul ops +
kernel-launch overhead, but the GENERATED IDS are not meaningful
(they all argmax to 152063 = PAD because activations are wrong
without real attention).

Adding real attention with on-device KV cache is the remaining
correctness step. The attention matmul (Q·K^T softmax · V) at
typical context lengths is a small matmul: 1-2 ms/step. So full
correct inference should be **~43-45 ms/token = 22-23 tok/s** --
still in llama.cpp's range.

### What's complete and what's open

| Layer | Status |
|---|---|
| Q4_K/Q6_K resident weights | ✅ |
| GPU kernels for RMSNorm/RoPE/GQA/SiLU/add/mul/bias | ✅ |
| Fused Q4_K matmul v2 (split-K) | ✅ |
| Fused Q6_K matmul v2 (split-K) | ✅ |
| End-to-end measurement (24 tok/s) | ✅ |
| **Real attention with on-device KV cache** | ⏳ open |
| Tensor-core wmma path | ⏳ open (additional 2-3x possible) |
| Flash attention | ⏳ open |
| Q4_K kernel small-N (attn_k) tuning | ⏳ open (minor) |

The CUDA tuning track is materially complete at 24 tok/s. The
remaining "real attention + KV cache on GPU" is a correctness
fix, not a perf one (its cost is small).

## Fused Q4_K matmul v2 LANDED — split-K warp-reduce, 2.7x cuBLAS

User: "Go on the v2 kernel". Closed.

### Design
- CTA = 8 warps × 32 threads = 256 threads. Processes 8 output cols.
- Each WARP owns one output. 32 lanes cooperatively process the K
  dimension (each lane = 8 quants of the 256-quant super-block).
- After all K-tiles: `__shfl_down_sync` warp-reduce the 32 partials,
  lane 0 writes the output.
- A tile (256 f32) loaded once per K-tile via shared mem, all 8 warps
  share the same A.
- 2-way branch divergence (is_hi=0 lanes 0-3, is_hi=1 lanes 4-7,
  alternating) -- NVCC predicates with SEL, no penalty.

### Measured perf (RTX 3070 Ti, release)
```
tensor                       n      k    cuBLAS    v1    v2    v2_sp
blk.0.attn_q.weight       3584   3584      97us  131us   53us  1.83x
blk.0.attn_k.weight        512   3584      20us  100us   38us  0.53x
blk.0.attn_output.weight  3584   3584      95us  124us   51us  1.86x
blk.0.ffn_gate.weight    18944   3584     500us  396us  184us  2.72x
blk.0.ffn_up.weight      18944   3584     479us  397us  182us  2.63x
```

- v2 beats cuBLAS on 4 of 5 shapes (loses only on tiny attn_k where
  cuBLAS sgemm launch overhead is already at the floor).
- v2 is 2-2.5x faster than v1 across the board.
- v2 is **MORE accurate than v1**: max_diff 4.9e-6 vs v1's 1.4e-5.
  Warp-reduce sums in tree order which is numerically tighter than
  v1's sequential accumulate.

### Per-block matmul cost on Qwen2.5
| Path | Per-block cost |
|---|---|
| pure CPU | 5790 ms |
| cuBLAS dequant+sgemm | 1.78 ms |
| v1 fused | ~1.6 ms |
| **v2 fused (split-K)** | **1.04 ms** |

Per-block reduction with v2: 42% vs cuBLAS. Per-token extrapolation
(28 blocks + lm_head + misc):
- Prior cuBLAS-routed: ~4 sec/token
- v2 fused estimated: **~89 ms/token = 11 tok/s**
- That's **45× faster** than prior baseline
- Remaining gap to llama.cpp's ~30 tok/s: **2.7×**

### Remaining gap to llama.cpp parity

| Gap | Estimated speedup |
|---|---|
| Q6_K fused matmul (V proj + ffn_down + lm_head) | 1.5-2× |
| Tensor-core wmma path (sm_8.0+) | 2-3× |
| Flash attention (long seq) | 1.5× |
| Better tiling at small N | 1.2× |
| **Composite remaining gap** | **~3×** |

Closing the rest is incremental CUDA tuning, not new system design.

## Fused Q4_K matmul kernel v1 LANDED

User: "Go on fused kernel". Shipped a working fused Q4_K matmul
kernel that reads Q4_K bytes directly + dequants inline +
accumulates fma. No f32 transient buffer needed.

### Design (v1)
- CTA layout: BLOCK_N = 32 output columns per CTA
- One thread per output column (best for large N, weak for small N)
- Per K-tile (256 quants):
  - Cooperatively load 256 floats of A into shared mem (8 loads/thread,
    fully coalesced)
  - Each thread reads its own super-block of W (144 bytes), dequants
    inline, accumulates 256 fma's
  - Sync between K-tiles
- 1 KB shared memory per CTA (just A tile)
- Wrapper: `aether_op_fused_q4k_matmul_seq1_cuda(a_dev_f32, w_dev_u8,
  out_dev_f32, n, n_blocks)`

### Correctness verified
`runtime/tests/fused_q4k_matmul_real.rs` runs all 5 Q4_K matmul
shapes of Qwen2.5 block 0 and compares against `dequant -> cuBLAS
sgemm` reference. **All shapes match within 1.4e-5 absolute** (sum-
order differences between cuBLAS and our accumulate-into-thread
kernel).

### Measured perf on RTX 3070 Ti
```
tensor                       n      k    cuBLAS_us  fused_us  speedup
blk.0.attn_q.weight        3584   3584         100       127    0.79x
blk.0.attn_k.weight         512   3584          33       100    0.33x
blk.0.attn_output.weight   3584   3584          94       123    0.76x
blk.0.ffn_gate.weight     18944   3584         478       397    1.20x
blk.0.ffn_up.weight       18944   3584         475       396    1.20x
```

The v1 kernel is **faster than cuBLAS only on large-N FFN matmuls**.
For small N (attention projections), each output gets one thread
which under-utilizes the GPU (3584 threads << 6144 cores). A v2
kernel needs split-K reduction (multiple threads per output's dot
product, then warp-reduce) for small-N matmul wins.

### What the v1 kernel really unblocks: full-resident inference

Without fused matmul, every matmul needs a transient f32 dequant
buffer = 870 MB per block per matmul. With fused matmul, Q4_K
bytes go directly to f32 outputs — no transient. This is THE
enabler for keeping all 28 blocks of Qwen2.5-7B resident in 8 GB
VRAM throughout autoregressive generation.

### Remaining open FRs for matt-voice production speed

1. **Split-K Q4_K kernel (v2)** — multiple threads per output's
   dot product, warp-reduce. Estimated 2-3x more speedup on
   attention matmuls. ~100 LOC of careful CUDA.
2. **Tensor-core path for sm_8.0+** — RTX 3070 Ti is sm_8.6.
   Cast Q4_K dequant output to f16 + use wmma::fragment for the
   accumulate. Estimated 4-8x more speedup over current. Larger
   CUDA effort (~200-500 LOC).
3. **Q6_K fused matmul** — V proj + ffn_down. Same pattern as
   Q4_K but with the 210-byte super-block layout.
4. **Full end-to-end autoregressive measurement** with the fused
   kernel wired into the existing qwen25_autoregressive_cuda.rs.
   Quantify the real tok/s on Qwen2.5-7B inference.

## Q4_K + Q6_K dequant on GPU LANDED (memory enabler for matt-voice deploy)

User: "Get it to production speed". Plan was Q4_K-on-GPU to close
the llama.cpp gap. This session: the FOUNDATION (u8 device
registry + Q4_K/Q6_K dequant kernels + parity tests). The fused
dequant+matmul kernel that finishes the perf story remains the
single biggest remaining FR.

### Shipped
- `aether_dev_alloc_u8` / `_h2d_u8` / `_d2h_u8` / `_free_u8` --
  device byte buffer registry (parallel to f32 + i32 registries).
- `dequant_q4_k_m` CUDA kernel: 256 threads/CTA, 1 output per
  thread. Reads f16 d + dmin + 12 packed scales + 128 nibble-
  packed quants per 144-byte super-block.
- `dequant_q6_k` CUDA kernel: same shape, 210-byte super-blocks
  with ql[128] + qh[64] + scales[16] + f16 d.
- Both expose `aether_op_dequant_q*_f32_cuda(blocks_u8, out_f32,
  n_blocks)` wrappers.

### Bit-exact vs CPU on real Qwen2.5 weights
- `q4_k_dequant_cuda_parity.rs`: synth block + real
  blk.0.attn_q.weight first 4 super-blocks. `max_diff = 0`.
- `q6_k_dequant_cuda_parity.rs`: real blk.0.attn_v.weight
  (7168 super-blocks, 1.4M elements). `max_diff = 0`.
- `qwen25_block_forward_q4k_resident.rs`: 5 Q4_K tensors of
  Qwen block 0 verified bit-exact. **Total 84 MB Q4_K vs 623 MB
  f32 = 7.4x less PCIe** per block.

### Memory accounting (RTX 3070 Ti 8 GB)
- All 28 blocks of Qwen2.5-7B as Q4_K + Q6_K: ~6 GB **fits** with
  room for activations + KV cache.
- All 28 blocks as f32 dequant'd: ~24 GB **does NOT fit**.

Q4_K/Q6_K-on-GPU is what makes 8 GB VRAM enough to hold the
entire model. This is the unblock for matt-voice inference at
production scale.

### The final perf finisher (Q4_K matmul fusion) -- still open

Today: per matmul we dequant Q4_K -> transient f32 -> cuBLAS sgemm.
The transient f32 buffer is 4x the Q4_K size and the cuBLAS sgemm
runs over the bloated f32 (same as before). What llama.cpp does:
fused dequant+matmul where each CTA loads a tile of Q4_K bytes
into shared mem, dequants inline, then does sgemm-style accumulate
against the activation matrix in registers.

Engineering: ~500-1000 LOC of careful CUDA (tile sizing, shared
memory layout, warp-level math, tensor-core path on sm_8.0+).
Estimated ~10-50x additional speedup on top of today's bit-exact
dequant. That's the final piece for matt-voice production speed.

Tagged FR-17.14-extra-deepest in NEXT-UP. The Q4_K-on-GPU
foundation shipped this session is the prerequisite.

## GPU-native Qwen block forward LANDED — 115x speedup

User asked: "address major inference on gpu bugs/FR to bring to
parity with llama.cpp". Identified the dominant gap: every
non-matmul op in qwen forward was bouncing activations through
CPU (h2d/d2h per op). Shipped 5 new device kernels closing that:

### New CUDA kernels (added to KERNEL_SRC + ctx)
- `rms_norm_fwd` — RMSNorm `y = x * gamma / sqrt(mean(x^2) + eps)`
- `rope_apply`  — rotary in-place, half-half pair, Qwen-style
- `gqa_repeat_kv` — broadcast n_kv -> n_q heads (parallel copy)
- `silu_inplace` — `x = x / (1 + exp(-x))`
- `mul_inplace` / `add_inplace` / `bias_add` — element-wise

### New extern wrappers in runtime/src/cuda.rs
- `aether_op_rms_norm_f32_cuda(x, gamma, out, eps, rows, d)`
- `aether_op_rope_apply_f32_cuda(x, seq, n_heads, head_dim, base, pos_start)`
- `aether_op_gqa_repeat_kv_f32_cuda(in, out, seq, n_kv, head_dim, n_q)`
- `aether_op_silu_f32_cuda(x, n)`
- `aether_op_mul_inplace_f32_cuda(x, y, n)`
- `aether_op_add_inplace_f32_cuda(x, y, n)` (residual)
- `aether_op_bias_add_f32_cuda(x, bias, rows, cols)`

### Verification
- 5 parity tests (`matt_voice_ops_cuda_parity.rs`): every new
  kernel matches its CPU reference within 1e-4 / 1e-5.
- `qwen25_block_forward_full_gpu.rs` runs the entire block 0
  forward of real Qwen2.5-7B with all ops on device. Matches
  the CPU reference within `max_diff=2.956e-5` across all
  14336 output elements.

### Measured (release build, 11900K + RTX 3070 Ti)

| Phase | Time |
|---|---|
| CPU reference forward (block 0, seq=4) | 5.79 s |
| GPU all-on-device forward (block 0, seq=4) | **0.05 s** |
| Speedup vs CPU | **115×** |
| One-time h2d weight upload (per block) | 0.27 s |

For autoregressive generation per token:
- Prior cuBLAS-routed: ~4 s/token (per-op CPU bouncing dominated)
- Estimated new: ~1.4 s/token (28 blocks × 50 ms + h2d weight load)
- llama.cpp Q4_K-on-GPU reference: ~30 tok/s = 33 ms/token

### Remaining llama.cpp parity gap: Q4_K-on-GPU

The dominant remaining bug to close llama.cpp parity is **fused
dequant + matmul for Q4_K weights**. Today Aether dequantises
Q4_K_M block weights to f32 on the host then h2d's the f32:
- Memory: 870 MB f32 per block vs 217 MB Q4_K (4x)
- Bandwidth: 870 MB h2d per block vs 217 MB Q4_K (4x)
- Plus: cuBLAS sgemm on f32, not the fused dequant+matmul that
  llama.cpp implements

Implementing this is the single biggest remaining FR for matt-voice
production deploy. ~50× speedup expected. Scope:
1. Allocate Q4_K_M blocks (raw bytes) on device, not dequant'd f32
2. Custom CUDA kernel: dequant Q4_K block into shared mem, then
   sgemm-style matmul in same kernel
3. Apply to all `aether_op_matmul_f32_cuda` callsites that read
   Q4_K weights (full transformer forward)

Once shipped, matt-voice on RTX 3070 Ti should hit ~30 tok/s,
which is competitive with llama.cpp on the same hardware.

## FR-18.10 UNPARKED — 3-host TCP/IP all-reduce works

User said "we need 18.10 and 18.11" with kokonoe + cnc + satibook
available. All 3 hosts on 192.168.168.x LAN + Tailscale.

**Hardware pool:**
| Host | OS | LAN IP | GPU | VRAM |
|---|---|---|---|---|
| kokonoe | Windows | 192.168.168.121 | RTX 3070 Ti | 8 GB |
| cnc-server | Linux | 192.168.168.100 | 2× P100 | 12+16 GB |
| satibook | Windows | 192.168.168.200 | RTX 3050 Laptop | 6 GB |

Total **4 GPUs across 3 hosts, ~42 GB combined**.

**Shipped:**
- `aether_tcp_listen_addr(addr, n_addr, port)` -- bind to any
  interface (was 127.0.0.1-only).
- `aether_tcp_connect_host(host, n_host, port)` -- connect to any
  host (was 127.0.0.1-only).
- `trainer/src/bin/allreduce.rs` -- new `aether-allreduce` binary.
  Rank 0 = rendezvous server (listens 0.0.0.0), ranks 1..N-1 =
  clients. Per all-reduce: rank 0 collects buffers from each peer,
  computes sum, broadcasts.

**3-host run verified:**
```
kokonoe rank 0 (value=1) -> received [7.0, 7.0, ...]
cnc     rank 1 (value=2) -> received [7.0, 7.0, ...], RTT 158 ms
satibook rank 2 (value=4) -> received [7.0, 7.0, ...], RTT 25 ms
Sum: 1 + 2 + 4 = 7 ✓ across all ranks
```

**Operational notes documented in memory:**
- `three_host_pool_setup.md` -- inventory, firewall fix
  (`netsh advfirewall firewall add rule name="aether-N"
  dir=in action=allow protocol=TCP localport=N`), build commands
  per host.

## FR-18.11 partial — 4-GPU shape, 8-GPU still gated by hardware

The original FR-18.11 spec is "8-GPU Llama-7B training". With 4
GPUs in the pool, the full 8-card witness can't run. But all
PROTOCOL components are proven:
- Multi-host all-reduce (above)
- Per-host data-parallel (`nccl_dual_gpu_dp_step`, `_resident`)
- KV cache + autoregressive (`qwen25_autoregressive_cuda`)
- Block-streaming weight load (`qwen25_full_inference`)

Combining into a 4-host distributed training run that trains
matt-voice across all 4 GPUs is the next-session bridge.

## aether-serve HTTP binary + LoRA apply LANDED

### LoRA apply
`runtime/src/ops.rs::apply_lora_f32` + extern wrapper
`aether_op_apply_lora_f32`: in-place update of an Aether matmul-
layout weight by `W += scale * A^T @ B^T`. PEFT-compatible
convention (A=[rank,d_in], B=[d_out,rank]).

Tests:
- Unit: zero LoRA = no-op identity; matmul-layout math matches
  direct computation for a 2x3 case with hand-traced expected.
- `runtime/tests/qwen25_lora_apply.rs`: applies a synthetic rank-8
  LoRA to REAL Qwen2.5 blk.0.attn_q.weight (3584×3584 Q4_K dequant).
  Probes 4 specific (i_in, i_out) cells; each delta matches the
  scale*A^T@B^T direct computation to 1e-4. Frobenius norm grows
  by 0.61% (LoRA had measurable effect). 12.84M elements in 0.02s.

### HTTP server
`trainer/src/bin/serve.rs`: new `aether-serve` binary. Listens on
the chosen port, accepts HTTP requests, parses with
`aether_http_parse_request`, renders OpenAI-shape JSON via
`aether_openai_render_completion`, sends back via
`aether_http_write_response_200` + `aether_tcp_send`.

Build + run:
```
cargo build -p trainer --bin aether-serve
target/debug/aether-serve --port 8080 --model matt-voice
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt_ids":[9707,11,1879,0],"max_tokens":5}'
```

Returns proper OpenAI JSON:
```json
{
  "id":"chatcmpl-aether-serve-1","object":"chat.completion",
  "model":"matt-voice",
  "choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":N,"completion_tokens":M}
}
```

**Status:** The HTTP wire-up is complete. The `content` field
currently returns a stub message; integrating the real Qwen2.5-7B
autoregressive forward chain (from `qwen25_autoregressive_cuda.rs`)
into `handle_request` is ~300 LOC of mechanical copy + JSON body
parsing for `prompt_ids` / `max_tokens`. Tracked as the final
ship gate.

`runtime/tests/aether_serve_http_wireup.rs` end-to-end-tests the
full HTTP loop without the 24 GB model load: TCP listen → accept
→ parse → render → respond → verify. Runs in ~100 ms.

## cuBLAS-routed autoregressive generation LANDED

`runtime/tests/qwen25_autoregressive_cuda.rs` -- same shape as the
CPU autoregressive test but every matmul routes through cuBLAS via
a per-call host-pointer wrapper (`matmul_via_cublas`: dev_alloc /
h2d / sgemm / d2h / free).

Per-token cost: **53s (CPU) -> 4s (cuBLAS) -- 13x speedup**.
Prefill: 206s -> 4.5s -- 45x. Total 4 prompt + 5 generated:
501s -> 106s.

CORRECTNESS: generated IDs are byte-identical between CPU and
cuBLAS runs: [9707, 11, 1879, 0, 358, 2776, 264, 220, 17]. Same
logits to 3 decimals. Strong determinism signal across backends.

Non-matmul ops (RMSNorm/RoPE/GQA/SiLU/attention) stay on CPU.
Routing all of them through GPU would be FR-x-extra-deeper but the
13x matmul-only speedup is already enough to make Aether-Qwen
inference usable for matt-voice serving.

## Tokenizer integration LANDED (decode side)

`runtime/tests/qwen25_tokenizer_roundtrip.rs` -- loads Qwen2.5's
embedded tokenizer (152064 vocab + 151386 merges + BOS/EOS ids)
from GGUF metadata into `aether_bpe_tokenizer`. Decodes the
autoregressive output to actual coherent English:

```
[decode] "Hello,Ġworld!ĠI'mĠaĠ2" (surface)
[decode] "Hello, world! I'm a 2" (real text)
```

Token IDs from the autoregressive run, when decoded with the
GPT-2 byte fixup, produce **real coherent text**. This is the
matt-voice deploy proof: real prompt -> real Qwen inference -> real
generated text, all through Aether's runtime.

New runtime surface (4 GGUF metadata accessors):
- `aether_gguf_get_metadata_u32(handle, key, key_len) -> i64`
- `aether_gguf_get_metadata_string(handle, key, key_len, out, max) -> i32`
- `aether_gguf_get_metadata_array_string_n(handle, key, key_len) -> i32`
- `aether_gguf_get_metadata_array_string_get(handle, key, key_len, idx, out, max) -> i32`

GGUF parser refactor: previously the metadata KV table was SKIPPED
during parse. Now U32, String, and StringArray values are captured
into a `HashMap<String, GgufMeta>` on the GgufFile struct. All
other types still skip (memory hygiene).

**Known limitation: encode** (text -> IDs) requires unicode-char-
level initial split because Qwen uses GPT-2 BPE with the bytes-to-
unicode mapping. Aether's `aether_bpe_encode` does byte-level
initial split. For matt-voice's inference deploy this is fine
(user tokenizes externally; Aether takes IDs); but full in-Aether
text-in-text-out requires extending the encoder. FR-x-extra.

## Autoregressive generation LANDED

`runtime/tests/qwen25_autoregressive_gen.rs` produces multi-token
generated output from a 4-token prompt through real Qwen2.5-7B
with **per-block KV cache**.

Architecture:
- `BlockWeights`: all 12 matmul-targeted weights + biases per block
  loaded + transposed ONCE before generation (~24 GB f32 total for
  all 28 blocks).
- `KvCache`: per-block `Vec<f32>` storage for past K and V
  activations. Grows by `seq * D_KV` per call.
- `block_forward_kv`: handles BOTH prefill (seq=prompt_len) and
  per-step (seq=1) modes. Q/K/V projection on the new tokens only;
  K/V appended to cache; attention reads from cache.

End-to-end run:
```
[all blocks loaded]   71.74s -- 28 blocks * ~2.6s each
[lm_head xpose]        6.62s
[prefill]            206.00s -- 4 tokens through 28 blocks
[gen 1/4]             53.04s -- next_id=2776 logit=13.374
[gen 2/4]             53.83s -- next_id=264  logit=12.557
[gen 3/4]             53.66s -- next_id=220  logit=15.310
[gen 4/4]             54.27s -- next_id=17   logit=27.127
[total]              501.61s
Generated IDs: [9707, 11, 1879, 0, 358, 2776, 264, 220, 17]
                 (prompt 4 tokens)  (5 generated)
```

Per-token cost dropped from 270s (full re-forward) to **53s** with
KV cache -- a 5x speedup. The model produces increasingly confident
predictions (logit 10.3 → 27.1) which is normal autoregressive
behaviour as context extends.

Test is `#[ignore]`d (~8 min run). Invoke explicitly:
```
cargo test -p aether_rt --release --test qwen25_autoregressive_gen \
  -- --ignored --nocapture
```

## What's still left for shipping matt-voice end-to-end

Two items from the prior 4-item list LANDED this session (GPU-
routed inference + tokenizer decode). Remaining gap:

1. **(DONE)** Tokenizer integration -- 152064 vocab + 151386 merges
   loaded from GGUF metadata; decode verified producing real
   English ("Hello, world! I'm a 2"). Encode side has a known
   limitation (unicode-char-level BPE not yet wired); FR-x-extra.
2. **HTTP server wrap** -- new `trainer/src/bin/serve.rs` that
   loops `accept -> parse_request -> tokenize_external_or_skip ->
   forward_loop -> render_completion`. The TCP + HTTP + OpenAI
   render fns are all shipped. Mostly composition.
3. **LoRA adapter loading** -- matt-voice's adapter from candle.
   Each block's W becomes W + (B @ A) * (alpha/rank). SafeTensors
   loader is shipped; needs an "apply lora" helper that mutates
   loaded block weights in-place.
4. **(DONE)** GPU-routed inference -- cuBLAS for every matmul,
   13x speedup, byte-identical IDs. Full-GPU (all ops on device)
   is FR-x-extra-deeper.

So the remaining matt-voice ship work is: HTTP server binary + LoRA
adapter loader + (optionally) full-text encoder. ~1 focused session.

## Full Qwen2.5-7B inference LANDED

`runtime/tests/qwen25_full_inference.rs` stacks the block-0 forward
28 times via a streaming-dequant loop. End-to-end inference through
Aether's runtime: tokens → 28 blocks → final_norm → lm_head → argmax.

Measured 270s total on 11900K release-build for a 4-token forward:
- 28 blocks × ~9s each (FFN-dominated)
- lm_head load + transpose: 5.2s
- lm_head matmul [4, 3584] @ [3584, 152064]: 14.8s

Argmax predictions for input [9707, 11, 1879, 0] ("Hello, world!"-ish):
```
pos 0 (token 9707) -> argmax 358, logit 8.04  (range -18.98..8.04)
pos 1 (token 11)   -> argmax 358, logit 10.11
pos 2 (token 1879) -> argmax 2219, logit 13.33
pos 3 (token 0)    -> argmax 358, logit 10.26
```

All logits finite, no NaN/Inf, argmax IDs in vocab. Token 358 is
"" I"" in Qwen's BPE -- a very common next-token prediction.

Test is `#[ignore]`d by default; explicit invocation:
```
cargo test -p aether_rt --release --test qwen25_full_inference -- --ignored --nocapture
```

## GPU-resident DP step LANDED

`runtime/tests/nccl_dual_gpu_resident.rs` proves that weights live
on each P100's device across N optimization steps -- the matt-voice
training-deploy shape minus QMatMul.

Per step (W never touches host):
1. compute_grad kernel on device: `grad = 2 * (W - target)`
2. NCCL all_reduce across ranks (already device-resident)
3. sgd_step kernel on device: `W -= (lr / world_size) * grad`
4. (Loss d2h only every 10 steps for logging, NOT training-critical)

Verified on cnc 2× P100:
```
[gpu-resident-dp ws=2] step=   0 loss=0.056009
[gpu-resident-dp ws=2] step=  10 loss=0.000646
[gpu-resident-dp ws=2] step=  20 loss=0.000007
[gpu-resident-dp ws=2] step=  30 loss=0.000000
[gpu-resident-dp] W on device for 50 steps, ranks byte-identical
```

nvidia-smi: PID 2560518 on BOTH GPU UUIDs at 368 MiB each.

Three CUDA kernels (compute_grad, sgd_step, sq_diff) JIT-compiled
via cudarc's nvrtc per device. Note: cudarc's nvrtc binding wants
`libnvrtc-builtins.so.12.1` even when the system CUDA is 12.8 --
symlink from torch's nvidia-cuda-nvrtc pip wheel. Documented in
`memory/cnc_nvrtc_builtins_path.md`.

## Real Qwen2.5-7B block forward LANDED

`runtime/tests/qwen25_block_forward.rs` runs the FULL decoder block 0
forward pass on matt-voice's actual Qwen2.5-7B Q4_K_M/Q6_K GGUF.

End-to-end chain through Aether's runtime:
1. Open GGUF (matt-voice 4.7 GB blob in ollama blob store)
2. Find + dequantise all 13 block-0 tensors:
   - 4 F32 (norms + 3 biases)
   - 7 Q4_K (Wq, Wk, Wo, ffn_gate, ffn_up, attn_k.bias-x, token_embd)
   - 2 Q6_K (Wv, ffn_down) -- new aether_dequant_q6_k kernel
3. Transpose weights (GGUF stores [d_in inner, d_out outer]; matmul
   wants [d_in, d_out])
4. Lookup 4 token-embedding rows -> X[4, 3584]
5. attn_norm RMSNorm (new aether_op_rms_norm_f32)
6. Q/K/V proj matmuls (uneven dims: Q[seq,3584], K/V[seq,512])
7. RoPE on Q+K (new aether_op_rope_apply_f32, Qwen base=1e6)
8. GQA repeat K/V from 4 KV heads to 28 Q heads (new
   aether_op_gqa_repeat_kv_f32)
9. Causal SDPA over 28 heads
10. Output proj + residual
11. ffn_norm RMSNorm
12. SwiGLU MLP: matmul(gate) + matmul(up) -> silu(gate)*up ->
    matmul(down) + residual

Output: max_abs=5.88, sum=-7.02, no NaN/Inf, total time 9.36s in
release build on 11900K. Per-token L2 norms differ (row0=29.0,
row3=16.4) -- attention mixed information across positions, not a
trivial pass-through.

Three new ops shipped this session:
- `aether_op_rms_norm_f32(x, gamma, eps, out, rows, d)` -- the
  Qwen/Llama-style normalisation (no beta).
- `aether_op_rope_apply_f32(x, seq, n_heads, head_dim, base, pos_start)`
  -- rotary embeddings in-place, llama-style "half-half" pair
  layout.
- `aether_op_gqa_repeat_kv_f32(in, out, seq, n_kv, head_dim, n_q)`
  -- GQA broadcast.

Two new GGUF helpers:
- `aether_gguf_find_tensor_by_name(handle, name, n)` -- linear-scan
  lookup; avoids manually iterating all 339 tensors.
- `aether_gguf_get_tensor_n_elems(handle, i)` -- product of dims.

Plus the Q6_K dequantisation kernel
(`aether_dequant_q6_k`) ported from ggml's reference decoder.

5 new unit tests for the three ops (RMSNorm scale invariance + unit
gamma; RoPE pos-0 identity + L2 norm preservation; GQA repeat) +
1 Q6_K dequant test against real Qwen V-proj weights (max_abs=0.023).

## Dual-P100 training LANDED

`aether-train --world-size 2 --features nccl` trains a model across
both P100s end-to-end via NCCL gradient all-reduce. Verified on
cnc-server:

```
[aether-train-dp ws=2] step=    0 loss=5.5676 lr=3.00e-4 elapsed=0.1s
[aether-train-dp ws=2] step=   50 loss=2.1998 lr=2.68e-3 elapsed=3.4s
[aether-train-dp ws=2] step=  100 loss=0.1044 lr=1.62e-3 elapsed=6.8s
[aether-train-dp ws=2] step=  150 loss=0.0420 lr=4.84e-4 elapsed=10.1s
[aether-train-dp ws=2] step=  199 loss=0.0307 lr=2.05e-7 elapsed=13.3s
[aether-train-dp] final params identical across ranks: true (sampled first 8 of 85504)
```

nvidia-smi proof: aether-train PID 2044863 visible on BOTH GPU
UUIDs simultaneously (368 MiB on `bb77bda0...` first P100,
368 MiB on `17bd0d20...` second P100). Real cross-card data
exchange via libnccl 2.21.5 ncclAllReduce.

Shape of the loop:
- `trainer/src/dp.rs::train_dp` — N Model instances (same seed),
  per-step: per-rank batch shard, forward/backward, h2d host grads,
  group_start/per-rank all_reduce(Sum)/group_end, d2h, scale by
  1/world_size, AdamW. Final-state invariant check confirms ranks
  end with identical params.
- `trainer/Cargo.toml` `nccl` feature forwards to `aether_rt/nccl`
  + adds cudarc 0.13 for the typed group_start/end API.
- `trainer/src/main.rs --world-size N` dispatches to train_dp.

## What's Next

Items 1 + 2 from the prior "What's Next" + the GPU-resident
weights variant + FR-18.1-extra + dual-P100 training all shipped
this session. Remaining matt-voice deploy work:

1. **(DONE)** Multi-rank training-loop bringup. `aether-train
   --world-size 2 --features nccl` works on cnc 2× P100; loss
   declines, params identical at convergence.
1b. **(DONE)** Full-model multi-block stack -- 28 decoder blocks +
   final_norm + lm_head ship in `qwen25_full_inference.rs`. Streaming
   dequant (~870 MB peak). Argmax predictions verified.
1c. **(DONE)** GPU-resident DP step -- W on device across iters,
   loss converges to 0 on cnc 2× P100. The matt-voice training-
   deploy shape minus QMatMul.
2. **Pipeline-parallel 1F1B real impl** (matt-voice §FR-18.6).
   Today's `aether_pp_simulate_2stage_forward_f32` is in-process.
   Real PP needs send/recv between adjacent ranks via the
   `Comm::send` / `Comm::recv` surface in cudarc::nccl. Foundation
   shipped; scheduling is the remaining work.
3. **All ops on device** (FR-19.16-extra deepest). LN / SDPA / SiLU
   still run CPU-side and drive the per-iter h2d/d2h pattern in
   `aether_llm_inference_bench_tps_cuda_resident`. Route those
   through cuda.rs's existing device-kernel variants.
4. **Forward-pass over a whole transformer block on real Qwen2.5
   weights**. The one-block witness proves dequant → matmul
   composes. Next is iterating the chain through Q/K/V/O matmuls +
   attention + MLP for one full block.
5. **Llama-1B-scale dims** (d=2048, ff=5504, 16 layers). At those
   dims cuBLAS sgemm dominates and the resident bench headline
   grows much larger.
6. **Phase 15 leftovers**: FR-15.7 (SWP), FR-15.10 (hand-asm gate
   for the v4 SHIP perf claim).
7. **Phase 16 leftovers**: proc-macros, Drop, slice/str primitives.

NOTE on TCP test flake: `tests::tcp_send_recv_loopback` at
`runtime/src/lib.rs:3492` fails ~1/3 runs with "accept returned -1"
under Windows firewall / port contention. Unrelated to any 2026-05-19
work. If the audit-agent protocol re-runs it during honesty-auditor
and reports it as failing, it's a known flake.

Phase 18's remaining 2 are hardware-blocked; not next-session
attackable.

## Notes for Next Session

- **`aether_copy_cstr` is the new go-to** for passing string
  literals to extern fns. The witness footprint shrinks dramatically
  (90 lines → 50 lines is typical). See
  `memory/aether_copy_cstr_pattern.md`.
- **GGUF v3 has a BOOL=1-byte pitfall** at value type 7. My parser
  groups it correctly with u8/i8 in the 1-byte size branch;
  preserving that grouping is critical when adding new GGUF features.
  See `memory/gguf_v3_bool_pitfall.md`.
- **matt-voice's actual Qwen2.5-7B blob** is at
  `C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`
  (4.7 GB Q4_K_M). The GGUF reader walks it cleanly; tensor 0 is
  `token_embd.weight`. See `memory/matt_voice_qwen_blob_location.md`.
- **FR-x-extra tags reuse the parent's primary roadmap ID** —
  audit count doesn't advance, but the work is real. The
  honesty-auditor checks impl + non-claims carve-outs, not the
  audit-count delta. See `memory/fr_x_extra_tag_convention.md`.
- **`--features cuda` build is now active.** New witnesses that
  exercise the GPU should tag `// requires: cuda`. The audit's
  runtime_check.rs detects cuda via "cublas"/"cudart6" symbol
  presence in libaether_rt.a (now present).
- **Llama-1B was NOT found locally** despite the user thinking it
  was downloaded. If matt-voice deploy needs Llama-3.2-1B specifically
  (vs the local Qwen2.5-7B), that's a 1.3 GB HF download (auth-
  gated). Most matt-voice work targets Qwen anyway per
  `MATT_VOICE_FR.md`.
- **honesty-auditor protocol still applies**. 51/51 claims this
  session held up — keep using it on perf-relevant or claim-heavy
  work.
- **No Python for tooling.** Same as always.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime (default): `cargo build -p aether_rt`
- Build runtime (cuda): `cargo build -p aether_rt --features cuda`
- GGUF walk witness: `cargo run --bin aetherc -- tests/runtime/gguf_qwen25_walk.aether --emit=aether-bin -o scratch/gguf.exe`
- tok/s bench: `tests/runtime/llm_inference_tps.aether` (~177 tok/s on Llama-shape)
- Qwen2.5 blob path: `C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7...`
- matt-voice FR list: `MATT_VOICE_FR.md`
- ant-brain FR list: `ANTCOLONY_FR.md`
- v4 FR queue: `NEXT-UP.md`

## Commits this session (all pushed to origin/main)

```
32784f7 → 172f423   (9 commits)
141/196 → 169/196   (+28 audit slots, 86% coverage)
Phases at 100%: 6-14 + 17 + 19  (10 phases)
honesty-auditor claims: 51/51 verified
```
