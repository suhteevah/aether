//! DINOv3 ViT-L/16 vision-backbone inference — pure-CPU f32 reference (FR-V1).
//!
//! Matches the `onnx-community/dinov3-vitl16-pretrain-lvd1689m-ONNX` graph (which
//! is the ORT acceptance reference) op-for-op, to cosine >= 0.999 on the CLS
//! embedding. See `docs/DINOV3_VITL16_SPEC.md` for the full architecture.
//!
//! Every dense op routes through the runtime ABI (`aether_op_matmul_f32`,
//! `aether_op_layer_norm_f32`, `aether_op_gelu_erf_f32`, `aether_op_sdpa_full_f32`).
//! Patch im2col, 2D axial RoPE, head transpose, bias-add, LayerScale and the
//! residual adds are local scalar code (cheap; the GPU path promotes them to
//! kernels). No causal mask, no KV-cache — a single bidirectional forward.
//!
//! Key facts the ONNX graph dictates (not the abstract HF config):
//!   * NO q/k/v bias (only o_proj + the two MLP layers carry bias).
//!   * token order = [CLS, 4 register, 196 patch] = seq 201.
//!   * RoPE rotates Q,K for PATCH tokens only (indices 5..201); the 5 prefix
//!     tokens pass through unrotated.
//!   * pre-norm + LayerScale: x += ls1*attn(norm1(x)); x += ls2*mlp(norm2(x)).
//!   * final LayerNorm over all 201 tokens; embedding = CLS (row 0); L2-normalize.

use std::os::raw::c_int;
use std::path::Path;

use crate::ops;

/// Static DINOv3 ViT-L/16 shape. `image_size` is fixed at construction so the
/// patch grid + RoPE table can be precomputed.
#[derive(Clone, Debug)]
pub struct VitConfig {
    pub d: usize,         // 1024
    pub n_layers: usize,  // 24
    pub n_heads: usize,   // 16
    pub head_dim: usize,  // 64
    pub d_ff: usize,      // 4096
    pub patch: usize,     // 16
    pub img: usize,       // 224
    pub grid: usize,      // img/patch = 14
    pub n_patches: usize, // grid*grid = 196
    pub n_reg: usize,     // 4
    pub n_prefix: usize,  // 1 (CLS) + n_reg = 5
    pub seq: usize,       // n_prefix + n_patches = 201
    pub eps: f32,         // 1e-5
    pub rope_theta: f32,  // 100.0
}

impl VitConfig {
    pub fn vitl16() -> Self {
        let patch = 16;
        let img = 224;
        let grid = img / patch;
        let n_patches = grid * grid;
        let n_reg = 4;
        let n_prefix = 1 + n_reg;
        Self {
            d: 1024, n_layers: 24, n_heads: 16, head_dim: 64, d_ff: 4096,
            patch, img, grid, n_patches, n_reg, n_prefix,
            seq: n_prefix + n_patches, eps: 1e-5, rope_theta: 100.0,
        }
    }

    /// Tiny config for the synthetic unit test (keeps the same structure, small
    /// dims so it runs in microseconds).
    pub fn tiny() -> Self {
        let patch = 4;
        let img = 8;
        let grid = img / patch; // 2
        let n_patches = grid * grid; // 4
        let n_reg = 2;
        let n_prefix = 1 + n_reg; // 3
        Self {
            d: 16, n_layers: 2, n_heads: 2, head_dim: 8, d_ff: 32,
            patch, img, grid, n_patches, n_reg, n_prefix,
            seq: n_prefix + n_patches, eps: 1e-5, rope_theta: 100.0,
        }
    }
}

struct Layer {
    norm1_w: Vec<f32>, norm1_b: Vec<f32>,
    qw: Vec<f32>, kw: Vec<f32>, vw: Vec<f32>, // each [d, d] = [in, out]
    ow: Vec<f32>, o_bias: Vec<f32>,           // [d, d], [d]
    ls1: Vec<f32>,                            // [d]
    norm2_w: Vec<f32>, norm2_b: Vec<f32>,
    up_w: Vec<f32>, up_b: Vec<f32>,           // [d, d_ff], [d_ff]
    down_w: Vec<f32>, down_b: Vec<f32>,       // [d_ff, d], [d]
    ls2: Vec<f32>,
}

pub struct Dinov3Session {
    pub cfg: VitConfig,
    cls: Vec<f32>,        // [d]
    reg: Vec<f32>,        // [n_reg*d]
    patch_w_t: Vec<f32>,  // [patch_in, d] = [(3*patch*patch), d], transposed for matmul
    patch_b: Vec<f32>,    // [d]
    layers: Vec<Layer>,
    final_w: Vec<f32>, final_b: Vec<f32>,
    // precomputed RoPE tables for the patch tokens: [n_patches * head_dim]
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
}

fn read_bin_f32(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{}: len {} not /4", path.display(), bytes.len()));
    }
    let mut v = Vec::with_capacity(bytes.len() / 4);
    for c in bytes.chunks_exact(4) {
        v.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    Ok(v)
}

impl Dinov3Session {
    /// Load DINOv3 ViT-L/16 from a directory of raw little-endian f32 `.bin`
    /// tensors (one per weight, named by the canonical key — produced by
    /// `scratch/dinov3/extract_weights.py`).
    pub fn load_dir(dir: &str) -> Result<Self, String> {
        let cfg = VitConfig::vitl16();
        let d = dir.to_string();
        let get = |name: &str, want: usize| -> Result<Vec<f32>, String> {
            let p = Path::new(&d).join(format!("{name}.bin"));
            let v = read_bin_f32(&p)?;
            if v.len() != want {
                return Err(format!("{name}: got {} want {}", v.len(), want));
            }
            Ok(v)
        };

        let cls = get("embeddings.cls_token", cfg.d)?;
        let reg = get("embeddings.register_tokens", cfg.n_reg * cfg.d)?;
        let patch_in = 3 * cfg.patch * cfg.patch; // 768
        // conv weight [d, 3, patch, patch] -> flatten [d, patch_in] -> transpose [patch_in, d]
        let pw = get("embeddings.patch_embeddings.weight", cfg.d * patch_in)?;
        let mut patch_w_t = vec![0.0f32; patch_in * cfg.d];
        for oc in 0..cfg.d {
            for pi in 0..patch_in {
                patch_w_t[pi * cfg.d + oc] = pw[oc * patch_in + pi];
            }
        }
        let patch_b = get("embeddings.patch_embeddings.bias", cfg.d)?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("layer.{i}.{s}");
            layers.push(Layer {
                norm1_w: get(&p("norm1.weight"), cfg.d)?,
                norm1_b: get(&p("norm1.bias"), cfg.d)?,
                qw: get(&p("attention.q_proj.weight"), cfg.d * cfg.d)?,
                kw: get(&p("attention.k_proj.weight"), cfg.d * cfg.d)?,
                vw: get(&p("attention.v_proj.weight"), cfg.d * cfg.d)?,
                ow: get(&p("attention.o_proj.weight"), cfg.d * cfg.d)?,
                o_bias: get(&p("attention.o_proj.bias"), cfg.d)?,
                ls1: get(&p("layer_scale1.lambda1"), cfg.d)?,
                norm2_w: get(&p("norm2.weight"), cfg.d)?,
                norm2_b: get(&p("norm2.bias"), cfg.d)?,
                up_w: get(&p("mlp.up_proj.weight"), cfg.d * cfg.d_ff)?,
                up_b: get(&p("mlp.up_proj.bias"), cfg.d_ff)?,
                down_w: get(&p("mlp.down_proj.weight"), cfg.d_ff * cfg.d)?,
                down_b: get(&p("mlp.down_proj.bias"), cfg.d)?,
                ls2: get(&p("layer_scale2.lambda1"), cfg.d)?,
            });
        }
        let final_w = get("norm.weight", cfg.d)?;
        let final_b = get("norm.bias", cfg.d)?;

        let (rope_cos, rope_sin) = build_rope(&cfg);
        Ok(Self {
            cfg, cls, reg, patch_w_t, patch_b, layers, final_w, final_b,
            rope_cos, rope_sin,
        })
    }

    /// Build a session from explicit weights (used by the synthetic test).
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        cfg: VitConfig, cls: Vec<f32>, reg: Vec<f32>, patch_w_t: Vec<f32>,
        patch_b: Vec<f32>, layers: Vec<Layer>, final_w: Vec<f32>, final_b: Vec<f32>,
    ) -> Self {
        let (rope_cos, rope_sin) = build_rope(&cfg);
        Self { cfg, cls, reg, patch_w_t, patch_b, layers, final_w, final_b, rope_cos, rope_sin }
    }

    /// Run the forward pass on a preprocessed image `pixel_values` laid out
    /// `[3, img, img]` (channels-first, RGB, already resized + normalized).
    /// Returns the L2-normalized 1024-d CLS embedding.
    pub fn embed(&self, pixel_values: &[f32]) -> Vec<f32> {
        let c = &self.cfg;
        assert_eq!(pixel_values.len(), 3 * c.img * c.img, "pixel_values shape");
        let d = c.d;
        let seq = c.seq;

        // ---- patch embedding: im2col [n_patches, patch_in] @ patch_w_t -> [n_patches, d] ----
        let patch_in = 3 * c.patch * c.patch;
        let mut cols = vec![0.0f32; c.n_patches * patch_in];
        for p in 0..c.n_patches {
            let prow = p / c.grid;
            let pcol = p % c.grid;
            for ic in 0..3 {
                for ki in 0..c.patch {
                    for kj in 0..c.patch {
                        let y = prow * c.patch + ki;
                        let x = pcol * c.patch + kj;
                        let px = pixel_values[(ic * c.img + y) * c.img + x];
                        cols[p * patch_in + (ic * c.patch + ki) * c.patch + kj] = px;
                    }
                }
            }
        }
        let mut patch_emb = vec![0.0f32; c.n_patches * d];
        unsafe {
            ops::matmul_f32(cols.as_ptr(), self.patch_w_t.as_ptr(),
                patch_emb.as_mut_ptr(), c.n_patches, patch_in, d);
        }
        for p in 0..c.n_patches {
            for j in 0..d { patch_emb[p * d + j] += self.patch_b[j]; }
        }

        // ---- assemble token buffer x[seq, d] = [CLS, reg.., patches..] ----
        let mut x = vec![0.0f32; seq * d];
        x[0..d].copy_from_slice(&self.cls);
        for r in 0..c.n_reg {
            x[(1 + r) * d..(2 + r) * d].copy_from_slice(&self.reg[r * d..(r + 1) * d]);
        }
        x[c.n_prefix * d..seq * d].copy_from_slice(&patch_emb);

        // scratch buffers reused across layers
        let mut xn = vec![0.0f32; seq * d];
        let mut q = vec![0.0f32; seq * d];
        let mut k = vec![0.0f32; seq * d];
        let mut v = vec![0.0f32; seq * d];
        let mut qh = vec![0.0f32; c.n_heads * seq * c.head_dim];
        let mut kh = vec![0.0f32; c.n_heads * seq * c.head_dim];
        let mut vh = vec![0.0f32; c.n_heads * seq * c.head_dim];
        let mut oh = vec![0.0f32; c.n_heads * seq * c.head_dim];
        let mut attn_scratch = vec![0.0f32; c.n_heads * seq * seq];
        let mut attn = vec![0.0f32; seq * d];
        let mut o = vec![0.0f32; seq * d];
        let mut mean = vec![0.0f32; seq];
        let mut inv_std = vec![0.0f32; seq];
        let mut ff = vec![0.0f32; seq * c.d_ff];
        let mut ff_out = vec![0.0f32; seq * d];

        for lyr in &self.layers {
            // ---- attention sub-block ----
            unsafe {
                ops::layer_norm_f32(x.as_ptr(), lyr.norm1_w.as_ptr(), lyr.norm1_b.as_ptr(),
                    c.eps, xn.as_mut_ptr(), mean.as_mut_ptr(), inv_std.as_mut_ptr(), seq, d);
                ops::matmul_f32(xn.as_ptr(), lyr.qw.as_ptr(), q.as_mut_ptr(), seq, d, d);
                ops::matmul_f32(xn.as_ptr(), lyr.kw.as_ptr(), k.as_mut_ptr(), seq, d, d);
                ops::matmul_f32(xn.as_ptr(), lyr.vw.as_ptr(), v.as_mut_ptr(), seq, d, d);
            }
            // RoPE on q,k patch tokens (indices n_prefix..seq), then transpose
            // [seq, head, dim] -> [head, seq, dim] for sdpa.
            self.apply_rope_inplace(&mut q);
            self.apply_rope_inplace(&mut k);
            transpose_to_heads(&q, &mut qh, seq, c.n_heads, c.head_dim);
            transpose_to_heads(&k, &mut kh, seq, c.n_heads, c.head_dim);
            transpose_to_heads(&v, &mut vh, seq, c.n_heads, c.head_dim);
            unsafe {
                ops::sdpa_full_f32(qh.as_ptr(), kh.as_ptr(), vh.as_ptr(),
                    oh.as_mut_ptr(), attn_scratch.as_mut_ptr(),
                    c.n_heads, seq, c.head_dim);
            }
            transpose_from_heads(&oh, &mut attn, seq, c.n_heads, c.head_dim);
            // o = attn @ Wo + o_bias
            unsafe {
                ops::matmul_f32(attn.as_ptr(), lyr.ow.as_ptr(), o.as_mut_ptr(), seq, d, d);
            }
            for t in 0..seq {
                for j in 0..d {
                    // o += bias, then LayerScale, then residual into x
                    let val = (o[t * d + j] + lyr.o_bias[j]) * lyr.ls1[j];
                    x[t * d + j] += val;
                }
            }

            // ---- MLP sub-block ----
            unsafe {
                ops::layer_norm_f32(x.as_ptr(), lyr.norm2_w.as_ptr(), lyr.norm2_b.as_ptr(),
                    c.eps, xn.as_mut_ptr(), mean.as_mut_ptr(), inv_std.as_mut_ptr(), seq, d);
                ops::matmul_f32(xn.as_ptr(), lyr.up_w.as_ptr(), ff.as_mut_ptr(), seq, d, c.d_ff);
            }
            for t in 0..seq {
                for j in 0..c.d_ff { ff[t * c.d_ff + j] += lyr.up_b[j]; }
            }
            unsafe { ops::gelu_erf_f32(ff.as_mut_ptr(), seq * c.d_ff); }
            unsafe {
                ops::matmul_f32(ff.as_ptr(), lyr.down_w.as_ptr(),
                    ff_out.as_mut_ptr(), seq, c.d_ff, d);
            }
            for t in 0..seq {
                for j in 0..d {
                    let val = (ff_out[t * d + j] + lyr.down_b[j]) * lyr.ls2[j];
                    x[t * d + j] += val;
                }
            }
        }

        // ---- final LayerNorm over all tokens, take CLS (row 0), L2-normalize ----
        unsafe {
            ops::layer_norm_f32(x.as_ptr(), self.final_w.as_ptr(), self.final_b.as_ptr(),
                c.eps, xn.as_mut_ptr(), mean.as_mut_ptr(), inv_std.as_mut_ptr(), seq, d);
        }
        let mut cls = xn[0..d].to_vec();
        let mut norm = 0.0f32;
        for &z in &cls { norm += z * z; }
        let inv = 1.0 / (norm.sqrt() + 1e-12);
        for z in &mut cls { *z *= inv; }
        cls
    }

    /// 2D axial RoPE applied in place to `t[seq, n_heads*head_dim]` for PATCH
    /// tokens only (indices `n_prefix..seq`). Split-half rotate; same cos/sin
    /// for every head. Prefix tokens (CLS + registers) are left unrotated.
    fn apply_rope_inplace(&self, t: &mut [f32]) {
        let c = &self.cfg;
        let hd = c.head_dim;
        let half = hd / 2;
        for ti in c.n_prefix..c.seq {
            let p = ti - c.n_prefix; // patch index
            let cos = &self.rope_cos[p * hd..(p + 1) * hd];
            let sin = &self.rope_sin[p * hd..(p + 1) * hd];
            for h in 0..c.n_heads {
                let base = ti * c.d + h * hd;
                let seg = &mut t[base..base + hd];
                // rotate_half: rot[d<half] = -x[d+half]; rot[d>=half] = x[d-half]
                let mut buf = [0.0f32; 128]; // head_dim <= 128
                for dd in 0..hd { buf[dd] = seg[dd]; }
                for dd in 0..hd {
                    let rot = if dd < half { -buf[dd + half] } else { buf[dd - half] };
                    seg[dd] = buf[dd] * cos[dd] + rot * sin[dd];
                }
            }
        }
    }
}

/// Precompute RoPE cos/sin tables `[n_patches * head_dim]` per the DINOv3
/// 2D-axial scheme: 16 freqs (head_dim/4), patch-center +0.5 coords normalized
/// to [-1,1], angle layout `[h16, w16, h16, w16]`.
fn build_rope(c: &VitConfig) -> (Vec<f32>, Vec<f32>) {
    let hd = c.head_dim;
    let n_freq = hd / 4; // 16
    let mut inv_freq = vec![0.0f64; n_freq];
    for i in 0..n_freq {
        let exponent = (i as f64) * (4.0 / hd as f64); // arange(0,1,4/hd)
        inv_freq[i] = 1.0 / (c.rope_theta as f64).powf(exponent);
    }
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut cos = vec![0.0f32; c.n_patches * hd];
    let mut sin = vec![0.0f32; c.n_patches * hd];
    for p in 0..c.n_patches {
        let prow = (p / c.grid) as f64;
        let pcol = (p % c.grid) as f64;
        let ch = (prow + 0.5) / c.grid as f64;
        let cw = (pcol + 0.5) / c.grid as f64;
        let coord_h = 2.0 * ch - 1.0;
        let coord_w = 2.0 * cw - 1.0;
        // angle64 = [h16, w16, h16, w16]
        for i in 0..n_freq {
            let ah = (two_pi * coord_h * inv_freq[i]) as f32;
            let aw = (two_pi * coord_w * inv_freq[i]) as f32;
            let row = p * hd;
            cos[row + i] = ah.cos();
            cos[row + n_freq + i] = aw.cos();
            cos[row + 2 * n_freq + i] = ah.cos();
            cos[row + 3 * n_freq + i] = aw.cos();
            sin[row + i] = ah.sin();
            sin[row + n_freq + i] = aw.sin();
            sin[row + 2 * n_freq + i] = ah.sin();
            sin[row + 3 * n_freq + i] = aw.sin();
        }
    }
    (cos, sin)
}

/// `[seq, n_heads, head_dim]` (head-minor) -> `[n_heads, seq, head_dim]`.
fn transpose_to_heads(src: &[f32], dst: &mut [f32], seq: usize, n_heads: usize, hd: usize) {
    for t in 0..seq {
        for h in 0..n_heads {
            let s = t * n_heads * hd + h * hd;
            let d = h * seq * hd + t * hd;
            dst[d..d + hd].copy_from_slice(&src[s..s + hd]);
        }
    }
}

/// `[n_heads, seq, head_dim]` -> `[seq, n_heads*head_dim]`.
fn transpose_from_heads(src: &[f32], dst: &mut [f32], seq: usize, n_heads: usize, hd: usize) {
    for h in 0..n_heads {
        for t in 0..seq {
            let s = h * seq * hd + t * hd;
            let d = t * n_heads * hd + h * hd;
            dst[d..d + hd].copy_from_slice(&src[s..s + hd]);
        }
    }
}

// ===========================================================================
// GPU path (FR-V1 deploy target — cnc P100). Mirrors the CPU forward: patch
// embed on device, assemble the 201-token buffer once on host, then the full
// 24-layer forward stays device-resident. Reuses cuBLAS matmul_nt + bias_add +
// layer_norm + bert_self_attention; adds erf-gelu + 2D-rope kernels. LayerScale
// is folded into the o_proj/down_proj weights+biases at upload.
// ===========================================================================
#[cfg(feature = "cuda")]
pub use gpu::Dinov3GpuSession;

#[cfg(feature = "cuda")]
mod gpu {
    use super::{VitConfig, read_bin_f32, build_rope};
    use std::os::raw::c_int;
    use std::path::Path;
    use crate::cuda::{
        aether_dev_alloc_f32, aether_dev_free_f32, aether_dev_h2d_f32, aether_dev_d2h_f32,
        aether_dev_sync, aether_op_matmul_nt_f32_cuda, aether_op_bias_add_f32_cuda,
        aether_op_add_inplace_f32_cuda, aether_op_layer_norm_f32_cuda,
        aether_op_bert_self_attention_fwd_f32_cuda, aether_op_gelu_erf_f32_cuda,
        aether_op_dinov3_rope2d_f32_cuda,
    };

    struct GpuLayer {
        n1w: i64, n1b: i64,
        qw: i64, kw: i64, vw: i64, ow: i64, ob: i64, // ow,ob layerscale-folded
        n2w: i64, n2b: i64,
        up_w: i64, up_b: i64, down_w: i64, down_b: i64, // down_* layerscale-folded
    }

    pub struct Dinov3GpuSession {
        cfg: VitConfig,
        cls: i64, reg: i64, patch_w: i64, patch_b: i64,
        layers: Vec<GpuLayer>,
        final_w: i64, final_b: i64,
        rope_cos: i64, rope_sin: i64,
        // activation scratch (reused across layers)
        cols: i64, patch_emb: i64, x: i64, xn: i64,
        q: i64, k: i64, v: i64, attn: i64, o: i64, ff: i64, ff_out: i64,
        mean: i64, rstd: i64,
        handles: Vec<i64>, // everything, for Drop
    }

    // Upload a host f32 slice to a fresh device buffer; returns the handle.
    fn up(v: &[f32]) -> i64 {
        let h = aether_dev_alloc_f32(v.len() as c_int);
        assert!(h != 0, "cudaMalloc failed for {} f32", v.len());
        unsafe { aether_dev_h2d_f32(v.as_ptr() as i64, h, v.len() as c_int); }
        h
    }

    // ONNX Linear weight is [in, out] (MatMul RHS). matmul_nt wants [out, in].
    fn transpose(w: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut t = vec![0.0f32; n_in * n_out];
        for i in 0..n_in {
            for o in 0..n_out {
                t[o * n_in + i] = w[i * n_out + o];
            }
        }
        t
    }

    impl Dinov3GpuSession {
        pub fn load_dir(dir: &str) -> Result<Self, String> {
            let cfg = VitConfig::vitl16();
            let d = cfg.d; let dff = cfg.d_ff;
            let dd = dir.to_string();
            let get = |name: &str, want: usize| -> Result<Vec<f32>, String> {
                let p = Path::new(&dd).join(format!("{name}.bin"));
                let v = read_bin_f32(&p)?;
                if v.len() != want { return Err(format!("{name}: got {} want {}", v.len(), want)); }
                Ok(v)
            };
            let mut handles = Vec::new();
            let mut reg_h = |h: i64, hs: &mut Vec<i64>| { hs.push(h); h };

            let patch_in = 3 * cfg.patch * cfg.patch;
            // conv weight [out, in] — native layout works directly with matmul_nt.
            let patch_w = reg_h(up(&get("embeddings.patch_embeddings.weight", d * patch_in)?), &mut handles);
            let patch_b = reg_h(up(&get("embeddings.patch_embeddings.bias", d)?), &mut handles);
            let cls = reg_h(up(&get("embeddings.cls_token", d)?), &mut handles);
            let reg = reg_h(up(&get("embeddings.register_tokens", cfg.n_reg * d)?), &mut handles);

            let mut layers = Vec::with_capacity(cfg.n_layers);
            for i in 0..cfg.n_layers {
                let p = |s: &str| format!("layer.{i}.{s}");
                let ls1 = get(&p("layer_scale1.lambda1"), d)?;
                let ls2 = get(&p("layer_scale2.lambda1"), d)?;
                // transpose projections [in,out]->[out,in]
                let qw = transpose(&get(&p("attention.q_proj.weight"), d * d)?, d, d);
                let kw = transpose(&get(&p("attention.k_proj.weight"), d * d)?, d, d);
                let vw = transpose(&get(&p("attention.v_proj.weight"), d * d)?, d, d);
                let mut ow = transpose(&get(&p("attention.o_proj.weight"), d * d)?, d, d);
                let mut ob = get(&p("attention.o_proj.bias"), d)?;
                // fold ls1 into o_proj (row o == output channel o)
                for o in 0..d { for k in 0..d { ow[o * d + k] *= ls1[o]; } ob[o] *= ls1[o]; }
                let up_w = transpose(&get(&p("mlp.up_proj.weight"), d * dff)?, d, dff);
                let up_b = get(&p("mlp.up_proj.bias"), dff)?;
                let mut dw = transpose(&get(&p("mlp.down_proj.weight"), dff * d)?, dff, d);
                let mut db = get(&p("mlp.down_proj.bias"), d)?;
                for o in 0..d { for k in 0..dff { dw[o * dff + k] *= ls2[o]; } db[o] *= ls2[o]; }
                layers.push(GpuLayer {
                    n1w: reg_h(up(&get(&p("norm1.weight"), d)?), &mut handles),
                    n1b: reg_h(up(&get(&p("norm1.bias"), d)?), &mut handles),
                    qw: reg_h(up(&qw), &mut handles),
                    kw: reg_h(up(&kw), &mut handles),
                    vw: reg_h(up(&vw), &mut handles),
                    ow: reg_h(up(&ow), &mut handles),
                    ob: reg_h(up(&ob), &mut handles),
                    n2w: reg_h(up(&get(&p("norm2.weight"), d)?), &mut handles),
                    n2b: reg_h(up(&get(&p("norm2.bias"), d)?), &mut handles),
                    up_w: reg_h(up(&up_w), &mut handles),
                    up_b: reg_h(up(&up_b), &mut handles),
                    down_w: reg_h(up(&dw), &mut handles),
                    down_b: reg_h(up(&db), &mut handles),
                });
            }
            let final_w = reg_h(up(&get("norm.weight", d)?), &mut handles);
            let final_b = reg_h(up(&get("norm.bias", d)?), &mut handles);

            let (cos, sin) = build_rope(&cfg);
            let rope_cos = reg_h(up(&cos), &mut handles);
            let rope_sin = reg_h(up(&sin), &mut handles);

            // activation scratch
            let seq = cfg.seq;
            let a = |n: usize, hs: &mut Vec<i64>| { let h = aether_dev_alloc_f32(n as c_int); hs.push(h); h };
            let cols = a(cfg.n_patches * patch_in, &mut handles);
            let patch_emb = a(cfg.n_patches * d, &mut handles);
            let x = a(seq * d, &mut handles);
            let xn = a(seq * d, &mut handles);
            let q = a(seq * d, &mut handles);
            let k = a(seq * d, &mut handles);
            let vv = a(seq * d, &mut handles);
            let attn = a(seq * d, &mut handles);
            let o = a(seq * d, &mut handles);
            let ff = a(seq * dff, &mut handles);
            let ff_out = a(seq * d, &mut handles);
            let mean = a(seq, &mut handles);
            let rstd = a(seq, &mut handles);

            Ok(Self {
                cfg, cls, reg, patch_w, patch_b, layers, final_w, final_b,
                rope_cos, rope_sin, cols, patch_emb, x, xn, q, k, v: vv, attn, o,
                ff, ff_out, mean, rstd, handles,
            })
        }

        pub fn embed(&self, pixel_values: &[f32]) -> Vec<f32> {
            let c = &self.cfg;
            let d = c.d; let seq = c.seq; let dff = c.d_ff;
            let patch_in = 3 * c.patch * c.patch;
            assert_eq!(pixel_values.len(), 3 * c.img * c.img);
            unsafe {
                // 1. im2col on host -> upload
                let mut cols = vec![0.0f32; c.n_patches * patch_in];
                for p in 0..c.n_patches {
                    let prow = p / c.grid; let pcol = p % c.grid;
                    for ic in 0..3 {
                        for ki in 0..c.patch { for kj in 0..c.patch {
                            let y = prow * c.patch + ki; let x = pcol * c.patch + kj;
                            cols[p * patch_in + (ic * c.patch + ki) * c.patch + kj] =
                                pixel_values[(ic * c.img + y) * c.img + x];
                        }}
                    }
                }
                aether_dev_h2d_f32(cols.as_ptr() as i64, self.cols, (c.n_patches * patch_in) as c_int);
                aether_op_matmul_nt_f32_cuda(self.cols, self.patch_w, self.patch_emb,
                    c.n_patches as c_int, patch_in as c_int, d as c_int);
                aether_op_bias_add_f32_cuda(self.patch_emb, self.patch_b,
                    c.n_patches as c_int, d as c_int);
                // download patch_emb, assemble token buffer, upload
                let mut pe = vec![0.0f32; c.n_patches * d];
                aether_dev_d2h_f32(self.patch_emb, pe.as_mut_ptr() as i64, (c.n_patches * d) as c_int);
                let mut cls_reg = vec![0.0f32; c.n_prefix * d];
                aether_dev_d2h_f32(self.cls, cls_reg.as_mut_ptr() as i64, d as c_int);
                // cls occupies row 0; registers rows 1..n_prefix
                let mut reg = vec![0.0f32; c.n_reg * d];
                aether_dev_d2h_f32(self.reg, reg.as_mut_ptr() as i64, (c.n_reg * d) as c_int);
                let mut xh = vec![0.0f32; seq * d];
                xh[0..d].copy_from_slice(&cls_reg[0..d]);
                xh[d..c.n_prefix * d].copy_from_slice(&reg);
                xh[c.n_prefix * d..seq * d].copy_from_slice(&pe);
                aether_dev_h2d_f32(xh.as_ptr() as i64, self.x, (seq * d) as c_int);

                let eps = c.eps;
                let (sc, np, nh, hd) = (seq as c_int, c.n_prefix as c_int,
                    c.n_heads as c_int, c.head_dim as c_int);
                let scale = 1.0 / (c.head_dim as f32).sqrt();
                for l in &self.layers {
                    aether_op_layer_norm_f32_cuda(self.x, l.n1w, l.n1b, self.xn,
                        self.mean, self.rstd, eps, sc, d as c_int);
                    aether_op_matmul_nt_f32_cuda(self.xn, l.qw, self.q, sc, d as c_int, d as c_int);
                    aether_op_matmul_nt_f32_cuda(self.xn, l.kw, self.k, sc, d as c_int, d as c_int);
                    aether_op_matmul_nt_f32_cuda(self.xn, l.vw, self.v, sc, d as c_int, d as c_int);
                    aether_op_dinov3_rope2d_f32_cuda(self.q, self.rope_cos, self.rope_sin,
                        sc, np, nh, hd, c.n_patches as c_int);
                    aether_op_dinov3_rope2d_f32_cuda(self.k, self.rope_cos, self.rope_sin,
                        sc, np, nh, hd, c.n_patches as c_int);
                    aether_op_bert_self_attention_fwd_f32_cuda(self.q, self.k, self.v,
                        self.attn, sc, nh, hd, scale);
                    aether_op_matmul_nt_f32_cuda(self.attn, l.ow, self.o, sc, d as c_int, d as c_int);
                    aether_op_bias_add_f32_cuda(self.o, l.ob, sc, d as c_int);
                    aether_op_add_inplace_f32_cuda(self.x, self.o, (seq * d) as c_int);

                    aether_op_layer_norm_f32_cuda(self.x, l.n2w, l.n2b, self.xn,
                        self.mean, self.rstd, eps, sc, d as c_int);
                    aether_op_matmul_nt_f32_cuda(self.xn, l.up_w, self.ff, sc, d as c_int, dff as c_int);
                    aether_op_bias_add_f32_cuda(self.ff, l.up_b, sc, dff as c_int);
                    aether_op_gelu_erf_f32_cuda(self.ff, self.ff, (seq * dff) as c_int);
                    aether_op_matmul_nt_f32_cuda(self.ff, l.down_w, self.ff_out, sc, dff as c_int, d as c_int);
                    aether_op_bias_add_f32_cuda(self.ff_out, l.down_b, sc, d as c_int);
                    aether_op_add_inplace_f32_cuda(self.x, self.ff_out, (seq * d) as c_int);
                }
                aether_op_layer_norm_f32_cuda(self.x, self.final_w, self.final_b, self.xn,
                    self.mean, self.rstd, eps, sc, d as c_int);
                aether_dev_sync();
                let mut out = vec![0.0f32; seq * d];
                aether_dev_d2h_f32(self.xn, out.as_mut_ptr() as i64, (seq * d) as c_int);
                let mut cls = out[0..d].to_vec();
                let mut norm = 0.0f32;
                for &z in &cls { norm += z * z; }
                let inv = 1.0 / (norm.sqrt() + 1e-12);
                for z in &mut cls { *z *= inv; }
                cls
            }
        }
    }

    impl Drop for Dinov3GpuSession {
        fn drop(&mut self) {
            for &h in &self.handles { if h != 0 { aether_dev_free_f32(h); } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
        for i in 0..a.len() { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    /// Synthetic tiny ViT: deterministic weights, checks the forward machinery
    /// runs end to end and returns a finite, unit-norm embedding of the right
    /// dim. Always on (no fixtures needed).
    #[test]
    fn dinov3_tiny_forward_smoke() {
        let cfg = VitConfig::tiny();
        let d = cfg.d;
        let patch_in = 3 * cfg.patch * cfg.patch;
        let mut seed: u64 = 0x1234_5678;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((seed >> 33) as f32) / (u32::MAX as f32) - 0.5) * 0.1
        };
        let vfill = |n: usize, r: &mut dyn FnMut() -> f32| (0..n).map(|_| r()).collect::<Vec<f32>>();

        let cls = vfill(d, &mut rnd);
        let reg = vfill(cfg.n_reg * d, &mut rnd);
        let patch_w_t = vfill(patch_in * d, &mut rnd);
        let patch_b = vfill(d, &mut rnd);
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                norm1_w: vec![1.0; d], norm1_b: vec![0.0; d],
                qw: vfill(d * d, &mut rnd), kw: vfill(d * d, &mut rnd),
                vw: vfill(d * d, &mut rnd), ow: vfill(d * d, &mut rnd),
                o_bias: vfill(d, &mut rnd), ls1: vec![1.0; d],
                norm2_w: vec![1.0; d], norm2_b: vec![0.0; d],
                up_w: vfill(d * cfg.d_ff, &mut rnd), up_b: vfill(cfg.d_ff, &mut rnd),
                down_w: vfill(cfg.d_ff * d, &mut rnd), down_b: vfill(d, &mut rnd),
                ls2: vec![1.0; d],
            });
        }
        let s = Dinov3Session::from_parts(cfg.clone(), cls, reg, patch_w_t, patch_b,
            layers, vec![1.0; d], vec![0.0; d]);
        let px = vfill(3 * cfg.img * cfg.img, &mut rnd);
        let emb = s.embed(&px);
        assert_eq!(emb.len(), d);
        assert!(emb.iter().all(|z| z.is_finite()), "non-finite embedding");
        let norm: f32 = emb.iter().map(|z| z * z).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "not unit norm: {norm}");
    }

    /// Real-weight acceptance: cosine >= 0.999 vs the GOLDEN reference
    /// (`visionsystem/scratch/dinov3_vitl16_ref.json`, produced by
    /// `dinov3_ref.rs` on ORT with the exact `resize224_triangle_imagenet_chw`
    /// preprocessing). `golden_pixels.bin` is that run's preprocessed input, so
    /// this is the true end-to-end FR-V1 acceptance (model-math + matching the
    /// golden's pixels). Gated on the extracted fixtures (not committed — 1.2 GB
    /// weights); passes trivially when absent.
    #[test]
    fn dinov3_cosine_vs_ort_reference() {
        let wdir = std::env::var("DINOV3_WEIGHTS")
            .unwrap_or_else(|_| "J:/aether/scratch/dinov3/wclean".into());
        let fdir = std::env::var("DINOV3_FIXTURES")
            .unwrap_or_else(|_| "J:/aether/scratch/dinov3/fixtures".into());
        let (wdir, fdir) = (wdir.as_str(), fdir.as_str());
        if !Path::new(wdir).join("norm.weight.bin").exists()
            || !Path::new(fdir).join("golden_pixels.bin").exists() {
            eprintln!("[dinov3] fixtures absent — skipping real-weight cosine check");
            return;
        }
        let s = Dinov3Session::load_dir(wdir).expect("load weights");
        let px = read_bin_f32(&Path::new(fdir).join("golden_pixels.bin")).unwrap();
        let golden = read_bin_f32(&Path::new(fdir).join("golden_emb.bin")).unwrap();
        let emb = s.embed(&px);
        let cos = cosine(&emb, &golden);
        eprintln!("[dinov3] CPU CLS cosine vs GOLDEN (triangle preprocess) = {cos:.6}");
        assert!(cos >= 0.999, "cosine {cos} < 0.999");
    }

    /// GPU acceptance + latency. Same golden check on the device path; reports
    /// per-image forward latency (the FR's latency metric — run on the cnc P100
    /// for the deploy number; kokonoe 3070 Ti for dev). Gated on fixtures.
    #[cfg(feature = "cuda")]
    #[test]
    fn dinov3_gpu_cosine_vs_golden() {
        let wdir = std::env::var("DINOV3_WEIGHTS")
            .unwrap_or_else(|_| "J:/aether/scratch/dinov3/wclean".into());
        let fdir = std::env::var("DINOV3_FIXTURES")
            .unwrap_or_else(|_| "J:/aether/scratch/dinov3/fixtures".into());
        let (wdir, fdir) = (wdir.as_str(), fdir.as_str());
        if !Path::new(wdir).join("norm.weight.bin").exists()
            || !Path::new(fdir).join("golden_pixels.bin").exists() {
            eprintln!("[dinov3-gpu] fixtures absent — skipping");
            return;
        }
        let s = super::Dinov3GpuSession::load_dir(wdir).expect("load gpu weights");
        let px = read_bin_f32(&Path::new(fdir).join("golden_pixels.bin")).unwrap();
        let golden = read_bin_f32(&Path::new(fdir).join("golden_emb.bin")).unwrap();
        let _warm = s.embed(&px); // warm clocks + nvrtc JIT
        let t = std::time::Instant::now();
        let emb = s.embed(&px);
        let ms = t.elapsed().as_secs_f32() * 1000.0;
        let cos = cosine(&emb, &golden);
        eprintln!("[dinov3-gpu] GPU CLS cosine vs GOLDEN = {cos:.6} | forward {ms:.1} ms");
        assert!(cos >= 0.999, "gpu cosine {cos} < 0.999");
    }
}
