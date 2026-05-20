# Aether Feature Requests — driven by OpenClaw local-inference

**Source project:** `J:\openclaw-vault\` (deployment); design spec at `J:\llm-wiki\projects\openclaw-local-inference.md`
**Started:** 2026-05-20
**Owner of this list:** maintained as Claude works on OpenClaw local-inference rollout; updated whenever an aether feature is needed that the runtime doesn't have yet (or only has on the 7B-Qwen happy path).
**Sibling lists:** `J:\aether\MATT_VOICE_FR.md` (LoRA training driven), `J:\aether\ANTCOLONY_FR.md` (RL trainer driven). Shared dependencies (GGUF, Q4_K, autoregressive, NCCL) are intentionally duplicated; cross-link as items land.

## How this list is used

The OpenClaw harness is migrating from llama.cpp to aether as the primary inference substrate. The design (Approach B — tiered worker pool with mesh burst) is substrate-agnostic; the migration cost is one `ExecStart=` edit per systemd unit + one LiteLLM `api_base` URL change, **provided aether covers OpenClaw's specific traffic shape**. The items below are the aether features OpenClaw needs that either don't exist yet or have only been validated on the Qwen2.5-7B happy path.

When aether ships one of these, mark it `[done]` with the commit hash + module that implements it. When OpenClaw rollout surfaces a NEW aether gap not yet listed, append it here with a citation (which phase / which agent traffic surfaced it).

The closure-of-the-list gate is **OpenClaw cnc-local fleet running 100% on aether for ≥7 days with no llama.cpp fallback firings**. Until then, llama.cpp stays warm as the bootstrap fallback.

---

## Critical path (gates phase 0 of OpenClaw rollout)

### Big-model kernel validation

- [ ] **FR-17-extra-14b-e2e** — Qwen2.5-14B-Q4_K_M end-to-end autoregressive with per-block dtype dispatch + generated-token parity vs cuBLAS reference. The Qwen-7B NaN-at-block-3 bug (per `NEXT-UP.md`, fixed via per-block dtype dispatch on 2026-05-20) demonstrated that mixed-precision GGUFs surface composition bugs only at depth. 14B has 48 decoder blocks (vs 7B's 28) — needs the same dtype-enumeration + parity test. **Witness:** `qwen25_14b_per_block_dtypes.rs` enumerates the dtype table; `qwen25_14b_autoregressive_e2e.rs` generates 16 tokens with all blocks finite + token-ID-identical to cuBLAS-routed reference.

- [ ] **FR-17-extra-32b-e2e** — Qwen2.5-32B-Q3_K_M end-to-end. 64 decoder blocks, Q3_K_M layout (not yet exercised at scale by the closed Q4_K/Q6_K kernels). Likely needs the Q3_K dequant kernel if not already shipping. Required for OpenClaw's main-fallback path (Approach B §4). **Witness:** generated-token parity vs cuBLAS at ≥10 tok/s on a single 16 GB P100 (cnc P100 #2).

- [ ] **FR-17-extra-sustained-load** — 10-minute sustained autoregressive load on 14B without NaN regression, monitored for FP-accumulation drift between blocks. **Witness:** `qwen25_14b_sustained.rs` runs 600 s, asserts per-block max_abs stays bounded across the run.

### Embedding architecture (BERT-shape forward)

OpenClaw's memory-core lancedb today calls Google embeddings; the design replaces this with bge-large-en-v1.5 on aether. bge-large is BERT-shape (encoder-only), not Llama-shape — different architecture family than the Qwen runtime aether currently hosts.

- [ ] **FR-17-extra-bert-fwd** — BERT-shape encoder forward pass. Bidirectional self-attention (no causal mask), token-type embeddings, [CLS] pooling. Q8_0 or F16 weights (no Q4_K — bge models aren't typically distributed quantized). **Witness:** `bert_bge_forward.rs` produces a 1024-dim embedding for "the quick brown fox" matching the HuggingFace `sentence-transformers/bge-large-en-v1.5` reference to ±1e-4 cosine.

- [ ] **FR-19-extra-embed-endpoint** — OpenAI-compatible `/v1/embeddings` endpoint on aether-serve. Accepts `{"input": "...", "model": "bge-large-en-v1.5"}`, returns `{"data": [{"embedding": [...]}]}`. **Witness:** `aether_serve_embeddings.rs` round-trips through HTTP and matches FR-17-extra-bert-fwd output.

### Multi-request throughput validation

Phase 19's P19.5 (paged-KV) + P19.7 (multi-model concurrent hosting) are landed but the 27.22 tok/s benchmark is **single-request**. OpenClaw subagent traffic is bursty: the 9 LiteLLM-routed agents can hit the workhorse concurrently (mailclaw + briefing + a manual chat all firing within seconds of each other is normal).

- [ ] **FR-19-extra-cb-prod** — Continuous-batching aggregate throughput on Qwen2.5-14B-Q4 at concurrency=8 (eight simultaneous 32-token generations). **Witness:** `aether_serve_cb_14b_c8.rs` sustains ≥ 60 tok/s aggregate (≥ 7.5 tok/s per stream) on a 16 GB P100. Numbers calibrated against llama.cpp on same hardware as floor.

- [ ] **FR-19-extra-queue-fairness** — Under sustained concurrency=8 load, no single stream is starved >2× the mean inter-token latency. **Witness:** percentile latency histogram per stream — p99/p50 ratio < 3.

### Multi-host distributed inference (validates FR-18.10 for inference path)

FR-18.10 (multi-host TCP/IP all-reduce 3-host) closed 2026-05-20 against a training-style collective workload. OpenClaw's main-fallback path needs the same primitive for **tensor-parallel inference** (split a 72B model's weights across cnc + kokonoe + satibook + run forward, not gradient sync). The op shape is similar but the call-site context is different (forward pass not backward, no optimizer state).

- [ ] **FR-18-extra-tp-infer-3host** — Tensor-parallel autoregressive inference across cnc + kokonoe + satibook using the FR-18.10 TCP all-reduce primitive. **Witness:** Qwen2.5-72B-Q3_K_M (or whatever fits across ~30 GB total) generates 16 tokens with token-ID parity vs single-host reference (where the single-host reference fits — likely needs to be against a model that fits both ways like 14B sharded vs 14B single).

- [ ] **FR-18-extra-tp-infer-bench** — Same as above, measure tok/s. **Witness:** distributed 72B at ≥ 3 tok/s aggregate. Slow is fine — this fires on main-fallback (cloud outage) not steady state.

## Quality-of-life (post-critical-path)

### Health + introspection (for outage-detector + telemetry)

OpenClaw's main-fallback engages on OAuth Max outage (per spec §4). Detection requires hitting aether-serve's health endpoint to know it's alive + ready before swap.

- [ ] **FR-19-extra-healthz** — Structured `/healthz` endpoint returning JSON: `{"status": "ready|loading|degraded", "model": "qwen2.5-14b", "vram_used_mb": ..., "queue_depth": ..., "uptime_s": ...}`. **Witness:** `aether_serve_healthz.rs` hits the endpoint mid-load + after model swap; status transitions correctly.

- [ ] **FR-19-extra-model-swap-api** — `POST /v1/admin/swap` endpoint to swap the loaded model (workhorse 14B → main-fallback 32B) without restarting the process. Cold-start is fine (~25 s); request just needs to return 202 + poll-via-healthz pattern. **Witness:** `aether_serve_swap.rs` swaps 14B → 32B in <30 s, no leaked GPU memory.

### Quality-gate hook for the scout

The scout (Qwen 2.5-7B-Q4) is supposed to evaluate workhorse output against per-agent rubrics + escalate to cloud on fail. The voice rubric for `lead-responder` / `proposal-gen` (per spec §Failure-mode-policy item 2) wants a structured evaluation primitive.

- [ ] **FR-19-extra-rubric-eval** — Structured evaluation endpoint: given `{output, rubric}`, return `{score: 0.0-1.0, reasons: [...]}`. Implementation = scout model with a fixed evaluation prompt template. Not strictly an aether feature — could live in OpenClaw — but a clean primitive in aether-serve makes the design cleaner. **Witness:** consistent score for a fixed (output, rubric) pair across 10 runs (variance < 0.05).

### Streaming for the dashboard

The OpenClaw gateway dashboard talks to `main` via streaming. When `main` is in local-fallback (Approach B §4), it'll be talking to aether-serve. Streaming UX should match the Anthropic API streaming shape so the dashboard doesn't notice.

- [ ] **FR-19-extra-anthropic-stream** — `text/event-stream` output matching Anthropic Messages API's `message_start` / `content_block_delta` / `message_stop` event shapes (instead of or in addition to OpenAI's `data: {...}` SSE). Lets the dashboard stream-render local-fallback responses without per-substrate logic. **Witness:** capture stream from real Anthropic + aether-serve in fallback mode; event shapes match field-for-field.

## Won't-do (out of scope for this list)

- **Training-related FRs** — covered by MATT_VOICE_FR.md (LoRA) and ANTCOLONY_FR.md (RL). OpenClaw is inference-only.
- **Cross-platform serving** — aether on Linux is enough. Windows / macOS serving is out of OpenClaw's scope.
- **Non-Qwen / non-BERT model families** — explicitly: no Llama 3 / Mixtral / DeepSeek family asks here. If OpenClaw ever needs them, append.
- **Replacing llama.cpp entirely** — the goal here is OpenClaw on aether. llama.cpp may stay for ad-hoc use (kokonoe llama-server for the Apple Shortcuts "Ask LLM" flow, per `Apple Ecosystem.md`).

## Status tracking

Mark `[done]` + commit hash when shipped. Aggregate count goes in spec §"Aether migration target".

| Section | Total | Done |
|---|---|---|
| Critical path | 9 | 0 |
| QoL | 4 | 0 |
| **Total** | **13** | **0** |

When `Critical path` hits 9/9, OpenClaw can flip phase 0 from llama.cpp to aether-serve as the sustained-load test target. When QoL hits 4/4, the design's nice-to-haves are all native.
