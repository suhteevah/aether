# DINOv3 ViT-L/16 — Exact Inference Spec (FR-V1)

Reference for the Aether-hosted DINOv3 backbone. Match
`facebook/dinov3-vitl16-pretrain-lvd1689m` (HF `transformers` `DINOv3ViTModel`)
to **cosine ≥ 0.999** vs the `onnx-community/...-ONNX` reference. Sources: HF
transformers `modeling_dinov3_vit.py` + `convert_dinov3_vit_to_hf.py`,
onnx-community `config.json`/`preprocessor_config.json`, Meta
`facebookresearch/dinov3`. (Research 2026-06-03.)

## Config (vitl16)
- hidden_size **1024**, num_hidden_layers **24**, num_attention_heads **16**,
  head_dim **64**, intermediate_size (MLP hidden) **4096**, patch_size **16**,
  image_size **224**, num_channels 3, num_register_tokens **4**.
- hidden_act **gelu (exact erf, NOT tanh)**, layer_norm_eps **1e-5**,
  layerscale_value init **1.0**, use_gated_mlp **false** (plain MLP, not SwiGLU),
  rope_theta **100.0**, drop_path 0.0 (off at eval).
- Biases (asymmetric!): query_bias **true**, **key_bias FALSE**, value_bias
  **true**, proj_bias(attn out) **true**, mlp_bias(fc1+fc2) **true**.

## Forward graph (DINOv3ViTModel)
1. Patch embed: `Conv2d(3, 1024, k=16, s=16)` (bias) → `[B,1024,14,14]` →
   `flatten(2).transpose(1,2)` → `[B,196,1024]`. (14×14=196 patches, row-major:
   h outer, w inner.)
2. Token assembly: `cat([cls(1), register(4), patches(196)], dim=1)` →
   **seq=201**. Order is **CLS, then 4 registers, then patches**. No additive
   position-embedding table (all positional info is RoPE).
3. Per layer (×24), pre-norm + LayerScale:
   - `x = x + ls1 * attn(norm1(x))`
   - `x = x + ls2 * mlp(norm2(x))`
   - LayerScale = elementwise multiply by learned 1024-vec (`lambda1`), no bias.
   - MLP = `down_proj(gelu(up_proj(x)))` (fc1 1024→4096, gelu, fc2 4096→1024).
4. Final `LayerNorm(1024, eps=1e-5)` over **all 201 tokens**.
5. Image embedding = `last_hidden_state[:,0,:]` (**CLS token, post final-LN**) =
   `pooler_output` (plain slice, no dense/tanh). **Model does NOT L2-normalize —
   caller must.**

## Attention
- Bidirectional, **no causal mask**. Scale **1/sqrt(64)=0.125**. No extra
  positional/relative bias beyond RoPE. attention_dropout 0.

## 2D axial RoPE (highest-risk — applied to Q,K only, patches only)
```
head_dim = 64; base = 100.0
inv_freq = 1 / base**arange(0, 1, 4/head_dim)        # 16 freqs (head_dim//4)
# coords, normalized [-1,1], patch-center +0.5, meshgrid ij (h outer, w inner):
ch = (arange(0.5, 14)/14); cw = (arange(0.5, 14)/14)
coords = stack(meshgrid(ch, cw, indexing='ij'), -1).flatten(0,1)   # [196,2]
coords = 2*coords - 1                                              # [-1,1]
angles = 2*pi * coords[:,:,None] * inv_freq[None,None,:]           # [196,2,16]
angles = angles.flatten(1,2)                                       # [196,32] = [h16, w16]
angles = angles.tile(2)                                            # [196,64] = [h16,w16,h16,w16]
cos = cos(angles); sin = sin(angles)                               # [196,64]
# rotate_half (SPLIT-HALF, not interleaved):
rotate_half(x) = cat(-x[..., 32:], x[..., :32], -1)
q_patches = q_patches*cos + rotate_half(q_patches)*sin
k_patches = k_patches*cos + rotate_half(k_patches)*sin
```
- Same cos/sin for all 16 heads.
- **CLS + 4 register tokens (first 5) get NO rotation** — split prefix at
  index 5, rotate patches[5:201] only, concat prefix back unrotated.

## Weights — HF safetensors names (i = 0..23), Linear stored [out,in], apply W.T
- `embeddings.cls_token [1,1,1024]`, `embeddings.register_tokens [1,4,1024]`,
  `embeddings.mask_token` (unused), `embeddings.patch_embeddings.weight
  [1024,3,16,16]`, `embeddings.patch_embeddings.bias [1024]`.
- `layer.{i}.norm1.{weight,bias} [1024]`.
- `layer.{i}.attention.q_proj.{weight[1024,1024],bias[1024]}`,
  `...k_proj.weight[1024,1024]` (**NO bias**),
  `...v_proj.{weight,bias}`, `...o_proj.{weight,bias}`.
- `layer.{i}.layer_scale1.lambda1 [1024]`.
- `layer.{i}.norm2.{weight,bias} [1024]`.
- `layer.{i}.mlp.up_proj.{weight[4096,1024],bias[4096]}`,
  `layer.{i}.mlp.down_proj.{weight[1024,4096],bias[1024]}`.
- `layer.{i}.layer_scale2.lambda1 [1024]`.
- `norm.{weight,bias} [1024]` (final LN).

Meta original checkpoint differs: fused `blocks.{i}.attn.qkv.{weight[3072,1024],
bias[3072]}` (chunk dim0 →q,k,v; drop k bias slice), `blocks.{i}.attn.proj`,
`blocks.{i}.ls{1,2}.gamma`, `blocks.{i}.mlp.fc1/fc2`, `storage_tokens`,
`patch_embed.proj`.

## ONNX model I/O (onnx-community, UNGATED)
- `onnx/model.onnx` (graph) + `onnx/model.onnx_data` (~1.21 GB external f32).
- Input `pixel_values [B,3,224,224]` f32. Outputs `last_hidden_state [B,201,1024]`
  + `pooler_output [B,1024]`. (Verify exact output names against the local file.)
- facebook/* is **gated** (HF login + license); onnx-community is **ungated**.

## Preprocess (preprocessor_config.json)
1. Convert RGB, channels_first.
2. Resize to **224×224 square** (do_center_crop off). resample=2; the
   DINOv3 fast processor's filter is the one ambiguity — pin empirically by
   feeding the reference `pixel_values` (safest: dump it from the oracle).
3. Rescale 1/255.
4. Normalize ImageNet `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`.
→ `pixel_values [1,3,224,224]` f32 RGB.

## Verification oracle
- Gold: PyTorch + transformers (3.11/3.12 venv) — it *is* the spec. Dump
  `pixel_values.npy`, `ref_cls.npy` (pooler_output, NOT normalized),
  `ref_last_hidden.npy`. Feed our runtime the exact pixel_values to isolate model
  math from resize ambiguity; compare CLS cosine after L2-norm both sides.
- No-login fallback: Rust `ort` crate + ungated ONNX (onnxruntime 1.22.0 dll
  present at `J:\visionsystem\models\onnxruntime-win-x64-1.22.0.zip`).

## 0.999 checklist (things that blow the target)
1. CLS+4 registers get NO RoPE (prefix split at 5).
2. RoPE is split-half (`[-x2,x1]`), angle layout `[h16,w16,h16,w16]`.
3. k_proj has NO bias; q/v/o + both MLP do.
4. Token order CLS,registers,patches; patches row-major (h outer, w inner).
5. Final LN over all 201, then CLS=pooler; no built-in L2.
6. MLP plain erf-GELU (not SwiGLU, not tanh-GELU).
7. HF splits qkv, Linear `[out,in]` (use W.T); Meta keeps fused qkv.
8. scale 0.125, bidirectional, no extra pos bias.
9. rope_theta=100.0, 16 freqs, coords [-1,1] with +0.5 center.
