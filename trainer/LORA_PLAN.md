# LoRA Fine-Tuning Foundation — Design & Status

Target: parameter-efficient fine-tune of **Qwen2.5-7B** ("matt-voice") on the
3-host GPU pool. Full 7B fine-tune does not fit in memory and the trainer cannot
load GGUF today, so LoRA is the path: **freeze the base, train low-rank
A (rank×in) + B (out×rank) adapters**, with delta `= (alpha/rank) * (B @ (A @ x))`.

This doc is intentionally precise about what is DONE (this CPU-tested foundation)
versus what still needs GPU work. Do not read more into the "done" column than is
written there.

---

## 1. The LoRA math (DONE, finite-diff verified)

`trainer/src/lora.rs` — `LoraAdapter`.

- **Shapes**: `A : [rank, in_dim]` row-major, `B : [out_dim, rank]` row-major.
- **Init** (`new`): `A ~ N(0, 1/sqrt(rank))` (kaiming on the rank fan-in) via the
  existing `rng::Rng` SplitMix64 + Box-Muller normal; `B = 0`. So the initial
  delta is **exactly zero** and the frozen base is unperturbed at step 0
  (standard LoRA init). Verified by `lora_init_zero_delta`.
- **forward(x, base_out)**: computes `a_x = A @ x` ([rank]), then
  `base_out += (alpha/rank) * (B @ a_x)`. Returns `a_x` for backward to reuse.
  `base_out` is *accumulated into* — the caller writes the frozen base linear's
  output there first.
- **backward(x, a_x, grad_out, grad_in)** with `s = alpha/rank`:
  - `dB += s * grad_out ⊗ a_x`   →  `dB[o,r] += s * grad_out[o] * a_x[r]`
  - `dA += s * (Bᵀ grad_out) ⊗ x` →  `dA[r,i] += s * (Σ_o B[o,r]·grad_out[o]) * x[i]`
  - `dx += s * Aᵀ Bᵀ grad_out`     →  accumulated into `grad_in`
- **adamw_step**: reuses `aether_rt::ops::adamw_step_f32` (the same primitive
  `model.rs` uses) on the A and B param/grad/m/v arrays separately.
- **grads_flat / grads_flat_mut / zero_grad**: expose the concatenated `[dA; dB]`
  gradient vector for contiguous all-reduce, and the inverse scatter.

**Finite-difference witness** (`lora_backward_finite_diff`): in=8, out=6, rank=4,
random x and non-zero B, scalar loss = sum(base_out), analytic dA/dB/dx vs
central-difference numerical gradient (eps=1e-3). **Max abs diff = 1.834e-5**
(threshold 1e-2). The test asserts and passes.

---

## 2. Data-parallel adapter all-reduce (DONE for world_size=1, NCCL path written)

`trainer/src/lora_dp.rs`.

- `flatten_grads` / `scatter_grads`: pack every adapter's `[dA; dB]` into one
  contiguous host buffer and back. Because adapters are tiny, the whole DP
  exchange is **one flat all-reduce** — no per-adapter latency tax.
- `all_reduce_sum_inplace`:
  - **`nccl` feature**: NCCL `group_start` / per-rank `all_reduce(Sum)` /
    `group_end` on device buffers, mirroring `dp.rs` exactly. Single-process,
    one host buffer summed across all device comms; `comm_at(rank)` from
    `runtime/src/nccl_real.rs` (the real 2×P100-witnessed NCCL).
  - **no `nccl`**: `world_size==1` is a no-op (the single rank already holds the
    full gradient); `world_size>1` panics with a "rebuild with --features nccl"
    message rather than silently producing wrong gradients.
- `all_reduce_and_step`: flatten → all-reduce SUM → scale by `1/world_size`
  (mean) → scatter back → AdamW per adapter. Mirrors the dp.rs gradient pipeline.

**CPU witness** (`lora_dp_world1_step_changes_params`): two 8→8 rank-4 adapters,
fabricated gradients, one full `all_reduce_and_step` at world_size=1, asserts both
A and B params changed. Plus `lora_dp_flatten_scatter_roundtrip`. Both pass.

> The NCCL multi-rank path is **written and compiles under `--features nccl`** but
> is **not exercised by a test here** — it needs the GPU pool to run, same as the
> existing `dp.rs` (whose NCCL path is also only run live, not in `cargo test`).
> Treat the multi-GPU adapter all-reduce as "code-complete, unverified on hardware"
> until run on the P100 box.

---

## 3. How this plugs into the frozen quantized GPU forward (NOT DONE — main GPU work)

The serving forward lives in `runtime/src/serving.rs` (`QwenSession`), which I did
**not** touch. The integration shape is:

- **Attachment points** (per transformer layer): `q_proj`, `k_proj`, `v_proj`,
  `o_proj` in attention; `gate_proj`, `up_proj`, `down_proj` in the MLP. Each is a
  frozen quantized linear `y = W_q x`. A LoRA adapter attaches to each: the frozen
  base matmul stays quantized on GPU, and the adapter delta `(alpha/rank)·B(Ax)`
  is computed (f32, small) and **added to that linear's output** before the next op.
- **Forward integration**: after the quantized base matmul produces `y` on device,
  run the adapter forward (a `[rank×in]` then `[out×rank]` matmul — trivial GEMM,
  can be on GPU or even CPU for small ranks) and add into `y`. The adapter A/B are
  f32 device tensors; the base weights never dequantize for forward.

**This wiring does not exist yet.** `lora.rs` is host-side f32; nothing in it calls
into `serving.rs` or uploads A/B to device. That is deliberate — the prompt scoped
this as the foundation, and serving.rs is being edited concurrently.

### The backward-through-frozen-base problem (the real remaining work)

LoRA only trains A/B, but to backprop the loss *to* the adapters you need `dx`
flowing *through* each frozen linear: `dx = dyᵀ · W` (gradient of `y = W x` w.r.t.
`x`). With a **quantized** `W` on GPU there are two options:

1. **Dequant-then-matmul**: dequantize `W` to f32 on device, then use the existing
   GPU `matmul_backward_lhs` (the CPU version `matmul_backward_lhs_f32` already
   exists in `runtime/src/ops.rs`; the cuBLAS path exists per the runtime cuda
   work). Simple, correct, costs a transient f32 copy of each weight matrix.
2. **Quantized matmul-backward kernel**: write a kernel that does `dy @ W` reading
   `W` straight from its quantized blocks. No transient f32 copy, more kernel work.

Either way this is a **GPU kernel/runtime task in `runtime/`**, not in `trainer/`,
and it is the single largest remaining piece. The adapter's own `dA`/`dB`/`dx`
contributions (the cheap part) are done and tested in `lora.rs::backward`.

A common simplification for first-light: only train adapters on the layers where
you need `dx` through *prior* layers anyway, and accept dequant-then-matmul cost.

---

## 4. Honest DONE vs NOT-DONE

### DONE (this foundation — all CPU-tested, `cargo test -p trainer lora` = 5/5 green)
- LoRA adapter struct + standard zero-delta init (A gaussian, B zeros).
- `forward` (delta accumulate) + saved `a_x` intermediate.
- `backward` with correct dA / dB / dx math — **finite-diff verified, max diff 1.834e-5**.
- Per-adapter AdamW via the runtime `adamw_step_f32` primitive.
- Flat `[dA;dB]` gradient export/import for contiguous all-reduce.
- DP wrapper: flatten → all-reduce(sum) → mean → scatter → step, with a tested
  world_size=1 no-op path.

### NOT DONE (needs GPU + runtime work, explicitly out of this foundation's scope)
- **GPU forward integration**: wiring adapters into `QwenSession` in
  `runtime/src/serving.rs` (add delta to each q/k/v/o/gate/up/down proj output).
- **Backward through the frozen quantized base** (`dx = dy @ W`): dequant-to-f32 +
  matmul_backward_lhs, *or* a quantized matmul-backward kernel. **Main remaining work.**
- **GGUF loading in the trainer**: the trainer cannot load GGUF today; the frozen
  7B base weights would come from the GGUF path that currently lives only in the
  serving/runtime side.
- **NCCL multi-GPU adapter all-reduce on real hardware**: the path is written and
  compiles under `--features nccl` but is unverified on the P100 box (no test here
  runs it, same caveat as `dp.rs`).
- **Activation checkpointing**: not implemented; full activations would need to be
  retained or recomputed for a real 7B backward.
- No adapter (de)serialization / save-load of trained A/B yet.

### Files in this change
- `trainer/src/lora.rs` (new) — adapter + 3 unit tests.
- `trainer/src/lora_dp.rs` (new) — DP wrapper + 2 unit tests.
- `trainer/src/lib.rs` — `pub mod lora; pub mod lora_dp;`.
- `trainer/LORA_PLAN.md` (this doc).

No changes to `runtime/src/cuda.rs` or `runtime/src/serving.rs`.
