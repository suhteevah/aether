# Aether Feature Requests — driven by antcolony PPO/RL training

**Source project:** `J:\antcolony\` (ant colony simulation game with RL-trained AI brains)
**Started:** 2026-05-03
**Owner of this list:** maintained as Claude works on antcolony's RL trainer; updated whenever a Candle feature is used that Aether doesn't have yet.

## How this list is used

The antcolony project is building a Rust+Candle PPO trainer to train ant-colony AI brains. We're using `J:\candle-src\` because Aether doesn't yet support all the operations RL training requires. **Each Candle dependency below is a feature Aether should add so we can swap Candle out for Aether eventually.**

When Aether ships a feature listed here, mark it `[done]` and (if applicable) note the Aether commit / module that implements it.

When the antcolony trainer encounters a NEW Candle dependency not yet listed, **append it to this file with a citation** (which trainer module needed it and why).

---

## Current dependencies (as antcolony-trainer is built)

### Tensor + Autograd

- [ ] `Tensor::randn` / `Tensor::zeros` / `Tensor::ones` constructors with shape + dtype + device
- [ ] `Tensor::matmul` with broadcast over batch dim
- [ ] `Tensor::add`, `Tensor::sub`, `Tensor::mul`, `Tensor::div` (elementwise + broadcast)
- [ ] `Tensor::relu`, `Tensor::sigmoid`, `Tensor::tanh` activations
- [ ] `Tensor::log`, `Tensor::exp` (for log-prob math)
- [ ] `Tensor::sum`, `Tensor::mean`, `Tensor::var` reductions over axes
- [ ] `Tensor::squeeze`, `Tensor::unsqueeze`, `Tensor::reshape`, `Tensor::transpose`
- [ ] `Tensor::clamp` (for tanh-squash + reward clipping)
- [ ] **Autograd** — `var.backward()` + `param.grad()` + `param.set_grad(None)`
- [ ] Layer params: `Linear { weight, bias }` with proper init (Kaiming/Xavier)

### Distributions (PPO-specific)

- [ ] `Normal(mean, std)` distribution
- [ ] `.sample()` / `.rsample()` (reparameterized sample for Gaussian — needed for PPO continuous actions)
- [ ] `.log_prob(action)` returning per-dimension log-density
- [ ] `.entropy()` returning per-dimension entropy
- [ ] Tanh-squashing transform with log-det-Jacobian correction (the standard SAC/PPO trick)

### Optimizers

- [ ] Adam optimizer with `lr`, `beta1`, `beta2`, `epsilon`, `weight_decay`
- [ ] `optimizer.step()` + `optimizer.zero_grad()`
- [ ] Learning-rate scheduler (linear warmup, cosine decay)
- [ ] `clip_grad_norm_(params, max_norm)` — global gradient clipping (PPO needs 0.5)

### Persistence

- [ ] Save/load parameters as `safetensors` format (industry standard, what Candle uses)
- [ ] Or: own format with versioning + dtype + shape metadata

### CUDA backend

- [ ] CUDA device selection (`Device::cuda(0)`) on Windows
- [ ] Native CUDA kernels for matmul / relu / sigmoid / softmax
- [ ] `compute_cap=86` for RTX 3070 Ti (Ampere)
- [ ] Fallback to CPU when CUDA unavailable

### Misc utilities

- [ ] Variance reduction: GAE (Generalized Advantage Estimation) helper
- [ ] Normalization helpers (z-score, batch-norm, layer-norm)
- [ ] Bool masking on tensors
- [ ] `argmax` / `argmin` along axis
- [ ] Random seeding for reproducibility

---

## Aether-equivalent already in place (existing capability)

(Add as Aether features land that match items above — moving items from "to do" to "done with cross-ref to Aether code".)

- (nothing yet — antcolony trainer hasn't started yet, list will populate as we build)

---

## Notes on the bigger picture

Aether's pitch for ant-colony training is the SAME pitch as for any RL workload: a from-scratch language with its own compiler + assembler + self-hosted PE32+ linker that emits CUDA-via-cuBLAS code. If Aether can supply the operations above with **Candle-comparable performance** (e.g., the documented "wins 3 of 4 sgemm sizes vs Candle and PyTorch" claim from `J:\aether\BENCH_RESULTS.md`), the antcolony trainer can swap Candle out cleanly.

The trainer will be designed with a thin abstraction layer (`trait Backend` over Tensor/Optimizer/Device) so swapping Candle ↔ Aether is a single trait-impl change rather than a rewrite.

---

## Per-commit log of what was added to this file

(Append here as the antcolony trainer adds new requirements.)

- 2026-05-03 — Initial list seeded by Claude based on the planned RL trainer architecture (ActorCritic + PPO + CUDA + safetensors). Items will be checked off as Aether supports them.
- 2026-05-03 — **HARD BLOCKER for Candle on this box**: `candle-kernels` requires MSVC `cl.exe` to compile its CUDA kernels (nvcc on Windows mandates MSVC host compiler — there is no GCC/clang fallback). kokonoe is pinned to stable-gnu (no MSVC), so Candle CUDA is BLOCKED until either (a) MSVC Build Tools are installed, or (b) Aether ships a CUDA path that works with MinGW/clang/native-Rust kernels (no nvcc dependency). **Aether parity advantage** — if Aether's CUDA codegen is self-contained (own assembler emitting PTX or cuBLAS calls without nvcc), it sidesteps this entire blocker on Windows. That's an explicit win condition for Aether vs Candle.
