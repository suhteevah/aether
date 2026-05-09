# Aether (op, dtype, device) coverage matrix — snapshot 2026-05-09

Generated for P14.2 (roadmap v3). Source: `runtime/src/lib.rs` (CPU bodies)
+ `runtime/src/cuda.rs` (GPU bodies behind `--features cuda`).

Cell content:
- `✓` — implemented + a tagged runtime witness exercises it.
- `~` — implemented in runtime, no dedicated witness yet.
- `·` — not implemented; on the Phase 7 op-surface roadmap.

| op group / op             | f32 CPU | f32 CUDA | f64 CPU | bf16 CUDA | i32 CPU |
|---------------------------|:-------:|:--------:|:-------:|:---------:|:-------:|
| **matmul**                |    ✓    |    ✓     |    ·    |     ~     |    ·    |
| matmul_backward_lhs       |    ✓    |    ~     |    ·    |     ·     |    ·    |
| matmul_backward_rhs       |    ✓    |    ~     |    ·    |     ·     |    ·    |
| matmul_blocked            |    ✓    |    ·     |    ·    |     ·     |    ·    |
| matmul_auto / auto_kernel |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **add / add_bias**        |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| add_inplace               |    ✓    |    ·     |    ·    |     ·     |    ·    |
| add_layer_norm (fused)    |    ·    |    ✓     |    ·    |     ·     |    ·    |
| axpy                      |    ✓    |    ·     |    ·    |     ·     |    ·    |
| scale                     |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **gelu / gelu_backward**  |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| relu / relu_backward      |    ✓    |    ·     |    ·    |     ·     |    ·    |
| silu                      |    ✓    |    ·     |    ·    |     ·     |    ·    |
| softmax / softmax_backward|    ✓    |    ✓     |    ·    |     ·     |    ·    |
| **layer_norm**            |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| layer_norm_backward       |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| **scaled_dot_product_attn**|   ✓    |    ✓     |    ·    |     ·     |    ·    |
| sdpa_backward             |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **cross_entropy**         |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| cross_entropy_backward    |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| **mse / mse_backward**    |    ✓    |    ·     |    ·    |     ·     |    ·    |
| mae / huber / kl_div      |    ✓    |    ·     |    ·    |     ·     |    ·    |
| bce / bce_with_logits     |    ✓    |    ·     |    ·    |     ·     |    ·    |
| contrastive               |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **embedding_lookup**      |    ✓    |    ·     |    ·    |     ·     |    ·    |
| embedding_backward        |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **clip_grad_norm**        |    ✓    |    ·     |    ·    |     ·     |    ·    |
| zero_grad                 |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **adamw_step**            |    ✓    |    ✓     |    ·    |     ·     |    ·    |
| sgd_step                  |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **all_reduce_sum**        |    ✓    |    ·     |    ·    |     ·     |    ·    |
| **conv1d / conv2d / conv3d** | · | · | · | · | · |
| pool (max/avg)            |    ·    |    ·     |    ·    |     ·     |    ·    |
| reductions (sum/mean/max) |    ·    |    ·     |    ·    |     ·     |    ·    |
| topk / argmax / sort      |    ·    |    ·     |    ·    |     ·     |    ·    |
| cat / stack / split       |    ·    |    ·     |    ·    |     ·     |    ·    |
| RoPE / FlashAttention     |    ·    |    ·     |    ·    |     ·     |    ·    |

## Totals (this snapshot)

- Total runtime symbols: ~73 (lib.rs) + ~10 (cuda.rs) = ~83 distinct op×backend.
- f32 CPU: 100% of the AetherLM-Nano op set (matmul/add/gelu/silu/relu/softmax
  /layer_norm/sdpa/cross_entropy/embedding/clip/AdamW + all backwards).
- f32 CUDA: matmul/sgemm + add + gelu + softmax + layer_norm + cross_entropy +
  AdamW + fused add_layer_norm. Backwards present for the hot path
  (cross_entropy, gelu, layer_norm dx + params).
- f64 / i32 / u-* dtypes: not implemented. Phase 7.1 work.
- bf16: matmul has a bf16 → f32 accumulation path; no other ops yet.
- Conv / pool / reductions / selection: all `·`. Phase 7.3 work.

## How this file is regenerated

Run the `coverage-matrix` subagent (`agent-sdk:Agent { subagent_type:
"coverage-matrix", prompt: "regenerate docs/COVERAGE_MATRIX.md against the
current runtime/" }`). The subagent reads `runtime/src/lib.rs` +
`runtime/src/cuda.rs`, rebuilds the grid, and re-issues this file. Hand
edits are fine for narrative rows; the data rows should track the runtime.
