# Aether — Session Handoff

## Last Updated
2026-05-19 (Phase 18 closeout — matt-voice + ant-brain critical path)

## Project Status
🟢 **Audit: 153/196 (78%) roadmap items witnessed.** **Phase 18 now
9/11 = 81% witnessed.** 0 errors, 132 workspace tests pass (+7 new
distributed-sim units). Honesty scan unchanged. **Phase 17 was the
first phase from 15-24 to close to 100%; Phase 18 now reaches 81%
with only the 2 hardware-blocked items (18.10 multi-host RDMA, 18.11
8-GPU Llama-7B) remaining.**

The user pointed at `J:\aether\MATT_VOICE_FR.md` (matt-voice QLoRA
trainer, 2× P100 PP/1F1B) and `J:\aether\ANTCOLONY_FR.md` (antcolony
PPO RL trainer) as the Phase-18-critical projects in the aether
directory. **Verified both files exist; the matt-voice critical path
(FR-18.1 → 18.2 → 18.6 → 18.5) is shipped at simulation level.**

```
Phase 6-14: 78/78 witnessed (100%) — unchanged
Phase 15:    8/10 witnessed (80%)  — unchanged
Phase 16:   22/25 witnessed (88%)  — unchanged
Phase 17:   20/20 witnessed (100%) — unchanged
Phase 18:    9/11 witnessed (81%)  ← +7 (P18.{1,4,5,6,7,8,9})
Phase 19:    0/16 witnessed (0%)   — unchanged
Phase 20:    7/10 witnessed (70%)  — unchanged
Phase 21:    4/10 witnessed (40%)  — unchanged
Phase 22:    6/10 witnessed (60%)  — unchanged
Phase 23:    2/6  witnessed (33%)  — unchanged
Phase 24:    7/10 witnessed (70%)  — unchanged
TOTAL:    153/196 (78%)
```

Workspace tests: 132 pass (+7 vs prior session: NCCL, TP, PP, FSDP,
ZeRO, overlap, grad_compress sims). Honesty scan: 0 todo / 0
unimplemented / 4 known carry-over stubs.

## What Was Done This Session

### Phase 18 closeout — 7 new audit slots + 1 deepening

Cited verbatim from `MATT_VOICE_FR.md`: matt-voice critical path =
FR-18.1 NCCL → FR-18.2-extra multi-rank → FR-18.6 PP/1F1B → FR-18.5
TP. This batch ships in-process simulations of all of them, plus
the remainder of the non-hardware-blocked Phase 18 surface.

**FR-18.1 NCCL FFI surface** — 8 extern "C" symbols matching the
libnccl shape; single-host fallback. `comm_create` rejects ws>1
with a -1 sentinel so callers don't silently misbehave on single-
GPU boxes. `all_reduce_f32` on ws=1 is identity. Witness
`nccl_single_host.aether` exercises full lifecycle + identity +
multi-rank rejection.

**FR-18.2 collectives deepening** — `collectives_exercise.aether`
calls each of broadcast/all_gather/reduce_scatter/send/recv/all_to_all
with known data ([10,20,30,40]) and asserts the pass-through output.
The prior `collectives_v4.aether` was decl-only.

**FR-18.5 Tensor parallel** — `aether_tp_simulate_column_parallel_
linear_f32`. Splits W column-wise across ws shards, computes per-
shard partials, concats. Witness verifies the concatenated output
matches `aether_op_matmul_f32` reference within 1e-5.

**FR-18.6 Pipeline parallel 1F1B** — `aether_pp_simulate_2stage_
forward_f32`. Splits N transformer blocks across n_stages stages,
runs micro-batches through the pipe. Witness verifies output matches
monolithic sequential block-application within 1e-5. Witness header
cites the matt-voice "2 P100s, 14B unlock" framing.

**FR-18.4 FSDP** — `aether_fsdp_simulate_shard_alltoall_f32`. Shard
+ reassemble round-trip is the identity. Witness header notes the
"overkill for QLoRA" framing.

**FR-18.7 ZeRO-1/2/3** — `aether_zero_simulate_stage_bytes_f32`
returns per-rank byte count for stage in {1, 2, 3}. Witness asserts
z1 < baseline, z2 < z1, z3 < z2, z3 ≈ baseline/ws.

**FR-18.8 Compute/comm overlap** — `aether_overlap_simulate_*_us`
return max(compute, comm) (overlapped) and compute+comm (serial).
CPU stand-in for the CUDA-stream version (FR-18.8-extra).

**FR-18.9 Gradient compression** — `aether_grad_compress_lowrank_f32`
preserves first K cols, zeros rest. Demonstrates the m·n → m·K + n·K
bandwidth shape. NOT real PowerSGD (no SVD/power iteration; that's
FR-18.9-extra).

**8 Aether witnesses + 7 runtime symbols + 7 unit tests.**
honesty-auditor verified all 14 claims. Each witness header
explicitly carves out "single-process simulation only; real
multi-rank requires libnccl + second card". Every distributed-sim
runtime symbol is named `*_simulate_*` to make the simulation
status load-bearing in the symbol surface.

### Bench

Bench-runner skip note appended to BENCH_LEDGER. All new fns are
either single-host NCCL fallbacks or `*_simulate_*` shapes — no
real multi-rank wall-time can be measured on kokonoe's single-card
3070 Ti. The cnc 2×P100 + libnccl link is where the actual cross-
card numbers will show up (FR-18.x-extra).

## Current State

**Working:**
- 153/196 roadmap-tagged witnesses pass via `aetherc --emit=aether-bin`.
- Workspace tests: 132 passing.
- Audit: `errors: 0` clean.
- Phase 17 = 100%, Phase 18 = 81% (only hardware-blocked items remain).
- matt-voice's distributed control flow can be written against the
  NCCL surface today and gets sane single-host semantics; flipping
  to real cross-card requires only the libnccl link + second GPU.
- PP / TP / FSDP / ZeRO / overlap / grad-compression all have
  runtime simulators that the algorithm shape verifies against.

**Honest scaffold-vs-shipped notes:**
- Every Phase 18 simulator is single-process. The runtime symbol
  names use `*_simulate_*` so the simulation status is visible at
  the call site (e.g., `aether_pp_simulate_2stage_forward_f32`).
- FR-18.1-extra (real libnccl link + cross-card all-reduce) is what
  unlocks the matt-voice 2×P100 distributed training. That requires
  hardware Matt has on cnc (cnc-server has the 2 P100s; kokonoe
  doesn't).
- FR-18.9 is rank-K shape, not real PowerSGD. Real SVD-driven
  compression is FR-18.9-extra.
- FR-18.8 overlap is a CPU stand-in returning the algorithmic total
  (max(compute, comm)); the CUDA-stream-driven impl is FR-18.8-extra.

## Blocking Issues

None on kokonoe. Audit reports `errors: 0`. Honesty scan flags 4
known-OK stubs (unchanged): `mir/fuse.rs:53`, `mir/spec.rs:161`,
`runtime_pe/src/lib.rs:59`, `runtime_pe/src/lib.rs:443`.

**Hardware-blocked Phase 18 items remaining** (not shipped, NOT
fake-witnessed):
- FR-18.10 Multi-host RDMA — needs 2+ hosts + IB switch.
- FR-18.11 8-GPU Llama-7B training — needs 8× CUDA GPUs.
Both correctly parked under `NEXT-UP.md §2 PARKED`.

## What's Next

`NEXT-UP.md` is the queue. Two phases still under 100%:

1. **Phase 19 (Serving) = 0/16**. FR-19.1 TLS 1.3 is the XL long-pole.
   FR-19.2 HTTP/HTTPS server depends on Path B. FR-19.16 (Llama-1B at
   ≥100 tok/s) is the matt-voice serving deployment gate.
2. **Phase 15 (Perf) = 8/10**. Two L items remain: FR-15.7 (SWP),
   FR-15.10 (hand-asm reference gate). FR-15.10 is the "v4 SHIP perf
   gate" — needs a bench fixture, not new compiler engineering.
3. **Path E — FR-20.4 self-hosted asm emitter (XL)**. Still gates
   the "drop Rust from the stack" milestone.
4. **Path C — FR-17.19-extra Llama-1B real**. SafeTensors load + HF
   parity + multi-block + RMSNorm runtime fn. The actual v4-SHIP gate
   (P17.19's existing witness is the architecturally-honest partial).

For **matt-voice specifically**, the unlock now is FR-18.1-extra
(real libnccl link on cnc) + the existing P17/P18 simulators
becoming real multi-rank impls. That's hardware work + a libnccl
link, not new Aether language work.

## Notes for Next Session

- **Phase 18 closeout is honest because of the `*_simulate_*`
  naming convention.** Don't rename them to drop the `_simulate_`
  prefix until the real cross-card impl actually ships. Renaming
  without changing the impl would burn audit honesty (see
  [[witness_not_shipped]]).
- **MATT_VOICE_FR.md and ANTCOLONY_FR.md are first-class artifacts
  in the aether root.** When working on distributed/QLoRA/RL features,
  check those files first — they encode the actual user-facing
  feature requirements driven by real workloads.
- **The cnc box has 2 P100s** per MATT_VOICE_FR.md. That's where
  the real cross-card multi-rank testing will happen. kokonoe is
  single-card (RTX 3070 Ti).
- **NCCL surface is stable for callers.** matt-voice can call
  `aether_nccl_*` today and get correct single-rank semantics. When
  the libnccl link arrives, callers don't change — only the bodies
  flip from "single-host fallback" to "real cross-card".
- **PP/TP/FSDP simulators verify against MONOLITHIC references.** The
  algorithm shape is preserved bit-for-bit; the runtime symbol shape
  is what changes when multi-rank lands.
- **No Python for tooling.** Same as always.

## Quick Reference

- Audit: `target/debug/aether-audit.exe`
- Roadmap-only: `target/debug/aether-audit.exe --only roadmap`
- Build aetherc: `cargo build --bin aetherc`
- Build runtime: `cargo build -p aether_rt`
- Build assembler: `cargo build --bin aether-asm`
- NCCL witness: `cargo run --bin aetherc -- tests/runtime/nccl_single_host.aether --emit=aether-bin -o scratch/nccl.exe`
- PP witness: `cargo run --bin aetherc -- tests/runtime/pp_2stage.aether --emit=aether-bin -o scratch/pp.exe`
- TP witness: `cargo run --bin aetherc -- tests/runtime/tp_column_parallel.aether --emit=aether-bin -o scratch/tp.exe`
- matt-voice FR list: `MATT_VOICE_FR.md` (root)
- ant-brain FR list: `ANTCOLONY_FR.md` (root)
- v4 FR queue: `NEXT-UP.md`

## Commits this session

```
e5fa443 Path C FR-17.3: conv2d CPU direct-loop reference
976dbce Phase 17 closeout: Q4_0 + FA2 + layer modules f32 + Llama-shaped partial
(pending) Phase 18 closeout: NCCL surface + PP/TP/FSDP/ZeRO/overlap/grad_compress sims
```
