# libaether_rt — C-ABI contract

This is the surface that `aetherc`-emitted LLVM IR calls into. **No framework layer.** The runtime is a thin shim that either no-ops (Phase 0) or routes directly to vendor libraries (Phase 1: cuBLAS / cuDNN / NCCL on the 3070 Ti).

The Rust crate in `runtime/` is bootstrap. Long-term the runtime is rewritten in Aether itself once self-hosting lands.

## Tape + autodiff

| symbol | purpose |
| --- | --- |
| `aether_autodiff_init(tape)` | reset the per-step tape |
| `aether_autodiff_push(tape, value)` | record a forward value |
| `aether_autodiff_partial(tape, dst, op_code, src)` | one symbolic partial — see op codes below |
| `aether_autodiff_accumulate(tape, grad)` | (legacy) bulk accumulation |
| `aether_autodiff_reverse(tape)` | run the reverse sweep |

### Partial op codes (stable, never reorder)

`PART_ADD=1 SUB_PLUS=2 SUB_MINUS=3 MUL=4 MATMUL_LHS=5 MATMUL_RHS=6 RELU=7 CROSS_ENTROPY=8 FORWARD_VJP=9 PARAM=10`

Defined in both `compiler/src/codegen/llvm/mod.rs` and the runtime; keep them in sync.

## Distributed

| symbol | purpose |
| --- | --- |
| `aether_dist_all_reduce(buf, world_size, backend)` | inserted by MIR pass for `#[distributed(...)]` fns. Backend codes: `0=NCCL 1=MPI 2=Gloo 99=Stub` |

## Primitive ops

Every Aether `extern fn` in `stdlib/ops.aether` and `stdlib/optim.aether` resolves to one of these:

| Aether                                | C symbol                          | Phase 1 backend       |
| ---                                   | ---                               | ---                   |
| `matmul_f32`                          | `aether_op_matmul_f32`            | cuBLAS `cublasSgemm`  |
| `matmul_bf16`                         | `aether_op_matmul_bf16`           | cuBLAS `cublasGemmEx` |
| `add_f32`                             | `aether_op_add_f32`               | cuDNN add / hand kernel |
| `scale_f32` / `axpy_f32`              | `aether_op_scale_f32` / `aether_op_axpy_f32` | cuBLAS BLAS-1 |
| `gelu_f32` / `silu_f32` / `relu_f32`  | `aether_op_*_f32`                 | hand kernel           |
| `softmax_f32`                         | `aether_op_softmax_f32`           | cuDNN / fused kernel  |
| `layer_norm_f32`                      | `aether_op_layer_norm_f32`        | cuDNN layer-norm      |
| `scaled_dot_product_attention_f32`    | `aether_op_sdpa_f32`              | flash-attn 2/3        |
| `cross_entropy_f32`                   | `aether_op_cross_entropy_f32`     | fused softmax+xent    |
| `clip_grad_norm_f32`                  | `aether_op_clip_grad_norm_f32`    | cuBLAS norm + scale   |
| `all_reduce_sum_f32`                  | `aether_op_all_reduce_sum_f32`    | NCCL `ncclAllReduce`  |
| `adamw_step_f32` / `sgd_step_f32`     | `aether_op_adamw_step_f32` / `aether_op_sgd_step_f32` | fused optimiser kernel |

Argument order is positional. Never reorder. Adding a new arg = new symbol.

## Lifecycle

* `aether_rt_self_check() -> i32` — returns the current tape entry count. Used as a startup smoke from compiled aether binaries.

## Memory ownership

* `*mut c_void` outputs: caller-allocated, callee-written.
* `*const c_void` inputs: caller-owned, callee never frees.
* No heap allocation inside the runtime — every shape is callsite-known.
