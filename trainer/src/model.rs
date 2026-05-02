//! AetherLM-Nano forward + backward. Pure orchestration: every tensor
//! operation is a call into `aether_rt::ops::*` — no math here. This file
//! is what aetherc Phase 1 will emit from `examples/aether_lm.aether`.
//!
//! Layout: row-major, contiguous, f32 throughout. Tensor shapes are
//! tracked by the caller; the runtime calls take flat pointers + size args.

use aether_rt::ops;
use crate::config::ModelConfig;
use crate::rng::Rng;

/// All learnable parameters of the model, laid out as one big arena so the
/// optimiser can iterate uniformly. Each entry is (offset, size).
pub struct Model {
    pub cfg: ModelConfig,
    pub params: Vec<f32>,
    pub grads: Vec<f32>,
    pub adam_m: Vec<f32>,
    pub adam_v: Vec<f32>,
    pub layout: Layout,
}

#[derive(Clone, Debug)]
pub struct Layout {
    // Each (offset, len) pair.
    pub tok_emb: (usize, usize),     // [V, D]
    pub pos_emb: (usize, usize),     // [S_max, D]
    pub blocks: Vec<BlockLayout>,
    pub ln_f_gamma: (usize, usize),  // [D]
    pub ln_f_beta: (usize, usize),   // [D]
    pub total: usize,
}

#[derive(Clone, Debug)]
pub struct BlockLayout {
    pub ln1_gamma: (usize, usize),
    pub ln1_beta: (usize, usize),
    pub w_qkv: (usize, usize),       // [D, 3D]
    pub b_qkv: (usize, usize),       // [3D]
    pub w_o: (usize, usize),         // [D, D]
    pub b_o: (usize, usize),         // [D]
    pub ln2_gamma: (usize, usize),
    pub ln2_beta: (usize, usize),
    pub w_fc1: (usize, usize),       // [D, F]
    pub b_fc1: (usize, usize),       // [F]
    pub w_fc2: (usize, usize),       // [F, D]
    pub b_fc2: (usize, usize),       // [D]
}

impl Model {
    /// Get a `*mut f32` into the param arena at `(off, len)`.
    #[inline]
    pub fn pp(&mut self, slot: (usize, usize)) -> *mut f32 {
        debug_assert!(slot.0 + slot.1 <= self.params.len());
        unsafe { self.params.as_mut_ptr().add(slot.0) }
    }
    /// Get a `*const f32` into the param arena at `(off, len)`.
    #[inline]
    pub fn pp_const(&self, slot: (usize, usize)) -> *const f32 {
        debug_assert!(slot.0 + slot.1 <= self.params.len());
        unsafe { self.params.as_ptr().add(slot.0) }
    }
    /// Get a `*mut f32` into the grad arena at `(off, len)`.
    #[inline]
    pub fn gp(&mut self, slot: (usize, usize)) -> *mut f32 {
        debug_assert!(slot.0 + slot.1 <= self.grads.len());
        unsafe { self.grads.as_mut_ptr().add(slot.0) }
    }
    /// Get a slice (immutable) into params.
    #[inline]
    pub fn ps(&self, slot: (usize, usize)) -> &[f32] {
        &self.params[slot.0..slot.0 + slot.1]
    }

    pub fn new(cfg: ModelConfig, seed: u64) -> Self {
        let mut layout = Layout {
            tok_emb: (0, 0), pos_emb: (0, 0), blocks: Vec::new(),
            ln_f_gamma: (0, 0), ln_f_beta: (0, 0), total: 0,
        };
        let mut o = 0;
        let mut alloc = |size: usize, lay: &mut Layout| -> (usize, usize) {
            let r = (o, size);
            // We can't borrow `o` mutably twice — use a closure-free pattern.
            r
        };
        let _ = (&mut layout, &mut alloc);

        let mut off = 0usize;
        let mut take = |size: usize, off: &mut usize| -> (usize, usize) {
            let pair = (*off, size); *off += size; pair
        };

        layout.tok_emb = take(cfg.vocab * cfg.d_model, &mut off);
        layout.pos_emb = take(cfg.seq_len * cfg.d_model, &mut off);
        for _ in 0..cfg.n_layers {
            let bl = BlockLayout {
                ln1_gamma: take(cfg.d_model, &mut off),
                ln1_beta:  take(cfg.d_model, &mut off),
                w_qkv:     take(cfg.d_model * 3 * cfg.d_model, &mut off),
                b_qkv:     take(3 * cfg.d_model, &mut off),
                w_o:       take(cfg.d_model * cfg.d_model, &mut off),
                b_o:       take(cfg.d_model, &mut off),
                ln2_gamma: take(cfg.d_model, &mut off),
                ln2_beta:  take(cfg.d_model, &mut off),
                w_fc1:     take(cfg.d_model * cfg.d_ff, &mut off),
                b_fc1:     take(cfg.d_ff, &mut off),
                w_fc2:     take(cfg.d_ff * cfg.d_model, &mut off),
                b_fc2:     take(cfg.d_model, &mut off),
            };
            layout.blocks.push(bl);
        }
        layout.ln_f_gamma = take(cfg.d_model, &mut off);
        layout.ln_f_beta  = take(cfg.d_model, &mut off);
        layout.total = off;

        let mut params = vec![0.0f32; off];
        let mut rng = Rng::new(seed);
        init_weights(&mut params, &layout, &cfg, &mut rng);

        let grads = vec![0.0f32; off];
        let adam_m = vec![0.0f32; off];
        let adam_v = vec![0.0f32; off];

        Self { cfg, params, grads, adam_m, adam_v, layout }
    }

    pub fn n_params(&self) -> usize { self.layout.total }
}

fn init_weights(p: &mut [f32], lay: &Layout, cfg: &ModelConfig, rng: &mut Rng) {
    fn fill_normal(slot: (usize, usize), p: &mut [f32], std: f32, rng: &mut Rng) {
        for i in 0..slot.1 { p[slot.0 + i] = rng.next_normal() * std; }
    }
    fn fill_const(slot: (usize, usize), p: &mut [f32], v: f32) {
        for i in 0..slot.1 { p[slot.0 + i] = v; }
    }

    fill_normal(lay.tok_emb, p, 0.02, rng);
    fill_normal(lay.pos_emb, p, 0.02, rng);
    for bl in &lay.blocks {
        fill_const(bl.ln1_gamma, p, 1.0);
        fill_const(bl.ln1_beta,  p, 0.0);
        fill_normal(bl.w_qkv, p, 0.02, rng);
        fill_const(bl.b_qkv, p, 0.0);
        fill_normal(bl.w_o, p, 0.02 / (2.0 * cfg.n_layers as f32).sqrt(), rng);
        fill_const(bl.b_o, p, 0.0);
        fill_const(bl.ln2_gamma, p, 1.0);
        fill_const(bl.ln2_beta,  p, 0.0);
        fill_normal(bl.w_fc1, p, 0.02, rng);
        fill_const(bl.b_fc1, p, 0.0);
        fill_normal(bl.w_fc2, p, 0.02 / (2.0 * cfg.n_layers as f32).sqrt(), rng);
        fill_const(bl.b_fc2, p, 0.0);
    }
    fill_const(lay.ln_f_gamma, p, 1.0);
    fill_const(lay.ln_f_beta,  p, 0.0);
}

/// All intermediate tensors saved during forward, needed by backward.
pub struct Activations {
    pub b: usize,
    pub s: usize,
    pub d: usize,
    pub f: usize,
    pub h: usize,
    pub hd: usize,

    pub x_emb: Vec<f32>,           // [B, S, D]  after token+pos embedding
    pub blocks: Vec<BlockAct>,
    pub ln_f_in: Vec<f32>,         // [B, S, D]
    pub ln_f_mean: Vec<f32>,       // [B*S]
    pub ln_f_inv: Vec<f32>,        // [B*S]
    pub ln_f_out: Vec<f32>,        // [B, S, D]
    pub logits: Vec<f32>,          // [B*S, V]
    pub probs: Vec<f32>,           // [B*S, V]
}

pub struct BlockAct {
    pub ln1_in: Vec<f32>,    // [B, S, D]
    pub ln1_mean: Vec<f32>,  // [B*S]
    pub ln1_inv: Vec<f32>,   // [B*S]
    pub ln1_out: Vec<f32>,   // [B, S, D]
    pub qkv: Vec<f32>,       // [B*S, 3D]
    pub q: Vec<f32>,         // [B*H, S, hd]
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,      // [B*H, S, S]
    pub attn_out: Vec<f32>,  // [B*H, S, hd]  -> reshaped to [B, S, D]
    pub o: Vec<f32>,         // [B, S, D]  after out proj + residual
    pub ln2_in: Vec<f32>,
    pub ln2_mean: Vec<f32>,
    pub ln2_inv: Vec<f32>,
    pub ln2_out: Vec<f32>,
    pub fc1_pre: Vec<f32>,   // [B*S, F]  linear out before gelu
    pub fc1_post: Vec<f32>,  // [B*S, F]  after gelu
    pub fc2: Vec<f32>,       // [B*S, D]
    pub block_out: Vec<f32>, // [B, S, D]
}

/// Forward pass. Returns mean cross-entropy loss.
pub fn forward(
    model: &Model, ids: &[i32], labels: &[i32], b: usize,
) -> (Activations, f32) {
    let cfg = &model.cfg;
    let s = cfg.seq_len.min(ids.len() / b);
    let d = cfg.d_model; let f = cfg.d_ff;
    let h = cfg.n_heads; let hd = cfg.head_dim();
    let bsd = b * s * d;
    let v = cfg.vocab;

    // 1. Embedding lookup + positional add.
    let mut x_emb = vec![0.0f32; bsd];
    unsafe {
        ops::embedding_lookup_f32(
            model.pp_const(model.layout.tok_emb), ids.as_ptr(),
            x_emb.as_mut_ptr(), b, s, v, d,
        );
    }
    // pos_ids = [0..S], one row of pos_emb added to each batch row
    let pos_emb = &model.params[model.layout.pos_emb.0..model.layout.pos_emb.0 + s * d];
    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                x_emb[(bi * s + si) * d + di] += pos_emb[si * d + di];
            }
        }
    }

    let mut blocks = Vec::with_capacity(cfg.n_layers);
    let mut x = x_emb.clone();
    for bl in &model.layout.blocks {
        let ba = run_block_forward(model, bl, &x, b, s, d, f, h, hd);
        x = ba.block_out.clone();
        blocks.push(ba);
    }

    // Final layer norm.
    let ln_f_in = x.clone();
    let mut ln_f_out = vec![0.0f32; bsd];
    let mut ln_f_mean = vec![0.0f32; b * s];
    let mut ln_f_inv  = vec![0.0f32; b * s];
    unsafe {
        ops::layer_norm_f32(
            ln_f_in.as_ptr(),
            model.pp_const(model.layout.ln_f_gamma),
            model.pp_const(model.layout.ln_f_beta),
            1e-5,
            ln_f_out.as_mut_ptr(),
            ln_f_mean.as_mut_ptr(),
            ln_f_inv.as_mut_ptr(),
            b * s, d,
        );
    }

    // Tied lm_head: logits = ln_f_out @ tok_emb^T  →  shape [B*S, V]
    // tok_emb is [V, D], so out[i, j] = sum_d ln_f_out[i, d] * tok_emb[j, d].
    let mut logits = vec![0.0f32; b * s * v];
    {
        let we = &model.params[model.layout.tok_emb.0..model.layout.tok_emb.0 + v * d];
        for i in 0..b * s {
            for j in 0..v {
                let mut acc = 0.0f32;
                for di in 0..d { acc += ln_f_out[i * d + di] * we[j * d + di]; }
                logits[i * v + j] = acc;
            }
        }
    }

    // Cross-entropy.
    let mut probs = vec![0.0f32; b * s * v];
    let loss = unsafe {
        ops::cross_entropy_f32(logits.as_ptr(), labels.as_ptr(), probs.as_mut_ptr(), b * s, v)
    };

    (
        Activations {
            b, s, d, f, h, hd,
            x_emb, blocks, ln_f_in, ln_f_mean, ln_f_inv, ln_f_out, logits, probs,
        },
        loss,
    )
}

fn run_block_forward(
    model: &Model, bl: &BlockLayout, x_in: &[f32],
    b: usize, s: usize, d: usize, f: usize, h: usize, hd: usize,
) -> BlockAct {
    let bsd = b * s * d;
    let bs = b * s;

    // ln1
    let ln1_in = x_in.to_vec();
    let mut ln1_out = vec![0.0f32; bsd];
    let mut ln1_mean = vec![0.0f32; bs];
    let mut ln1_inv  = vec![0.0f32; bs];
    unsafe {
        ops::layer_norm_f32(
            ln1_in.as_ptr(),
            model.pp_const(bl.ln1_gamma), model.pp_const(bl.ln1_beta),
            1e-5,
            ln1_out.as_mut_ptr(), ln1_mean.as_mut_ptr(), ln1_inv.as_mut_ptr(),
            bs, d,
        );
    }

    // qkv = ln1_out @ W_qkv + b_qkv     [bs, d] @ [d, 3d] = [bs, 3d]
    let mut qkv = vec![0.0f32; bs * 3 * d];
    unsafe {
        ops::matmul_f32(
            ln1_out.as_ptr(), model.pp_const(bl.w_qkv),
            qkv.as_mut_ptr(), bs, d, 3 * d,
        );
        ops::add_bias_f32(qkv.as_mut_ptr(), model.pp_const(bl.b_qkv), bs, 3 * d);
    }

    // Reshape & split q/k/v into [B*H, S, hd] each.
    // qkv layout: row i (in b*s) has [q(d), k(d), v(d)] contiguously.
    // For each head we extract head_dim consecutive elements.
    let mut q = vec![0.0f32; b * h * s * hd];
    let mut k = vec![0.0f32; b * h * s * hd];
    let mut v = vec![0.0f32; b * h * s * hd];
    for bi in 0..b {
        for si in 0..s {
            let row_off = (bi * s + si) * 3 * d;
            for hi in 0..h {
                for di in 0..hd {
                    let head_off = hi * hd + di;
                    let dst_off = ((bi * h + hi) * s + si) * hd + di;
                    q[dst_off] = qkv[row_off + head_off];
                    k[dst_off] = qkv[row_off + d + head_off];
                    v[dst_off] = qkv[row_off + 2 * d + head_off];
                }
            }
        }
    }

    // Causal SDPA per (batch, head).
    let mut attn = vec![0.0f32; b * h * s * s];
    let mut attn_out = vec![0.0f32; b * h * s * hd];
    unsafe {
        ops::sdpa_causal_f32(
            q.as_ptr(), k.as_ptr(), v.as_ptr(),
            attn_out.as_mut_ptr(), attn.as_mut_ptr(),
            b * h, s, hd,
        );
    }

    // Reshape attn_out [B*H, S, hd] -> [B, S, D] (head-concat).
    let mut concat = vec![0.0f32; bsd];
    for bi in 0..b {
        for si in 0..s {
            for hi in 0..h {
                for di in 0..hd {
                    let src = ((bi * h + hi) * s + si) * hd + di;
                    let dst = (bi * s + si) * d + hi * hd + di;
                    concat[dst] = attn_out[src];
                }
            }
        }
    }

    // out_proj = concat @ W_o + b_o; residual + x_in
    let mut o_lin = vec![0.0f32; bsd];
    unsafe {
        ops::matmul_f32(concat.as_ptr(), model.pp_const(bl.w_o), o_lin.as_mut_ptr(), bs, d, d);
        ops::add_bias_f32(o_lin.as_mut_ptr(), model.pp_const(bl.b_o), bs, d);
        ops::add_inplace_f32(o_lin.as_mut_ptr(), x_in.as_ptr(), bsd);
    }
    let o = o_lin;

    // ln2
    let ln2_in = o.clone();
    let mut ln2_out = vec![0.0f32; bsd];
    let mut ln2_mean = vec![0.0f32; bs];
    let mut ln2_inv  = vec![0.0f32; bs];
    unsafe {
        ops::layer_norm_f32(
            ln2_in.as_ptr(),
            model.pp_const(bl.ln2_gamma), model.pp_const(bl.ln2_beta),
            1e-5,
            ln2_out.as_mut_ptr(), ln2_mean.as_mut_ptr(), ln2_inv.as_mut_ptr(),
            bs, d,
        );
    }

    // FFN: fc1 -> gelu -> fc2; residual + o
    let mut fc1_pre = vec![0.0f32; bs * f];
    unsafe {
        ops::matmul_f32(
            ln2_out.as_ptr(), model.pp_const(bl.w_fc1),
            fc1_pre.as_mut_ptr(), bs, d, f,
        );
        ops::add_bias_f32(fc1_pre.as_mut_ptr(), model.pp_const(bl.b_fc1), bs, f);
    }
    let mut fc1_post = fc1_pre.clone();
    unsafe { ops::gelu_f32(fc1_post.as_mut_ptr(), bs * f); }

    let mut fc2 = vec![0.0f32; bsd];
    unsafe {
        ops::matmul_f32(fc1_post.as_ptr(), model.pp_const(bl.w_fc2), fc2.as_mut_ptr(), bs, f, d);
        ops::add_bias_f32(fc2.as_mut_ptr(), model.pp_const(bl.b_fc2), bs, d);
    }
    let mut block_out = fc2.clone();
    unsafe { ops::add_inplace_f32(block_out.as_mut_ptr(), o.as_ptr(), bsd); }

    BlockAct {
        ln1_in, ln1_mean, ln1_inv, ln1_out,
        qkv, q, k, v, attn, attn_out,
        o,
        ln2_in, ln2_mean, ln2_inv, ln2_out,
        fc1_pre, fc1_post, fc2, block_out,
    }
}

/// Backward pass. Accumulates gradients into `model.grads`.
pub fn backward(model: &mut Model, act: &Activations, ids: &[i32], labels: &[i32]) {
    let cfg = &model.cfg.clone();
    let b = act.b; let s = act.s; let d = act.d; let f = act.f;
    let h = act.h; let hd = act.hd; let v = cfg.vocab;
    let bs = b * s;
    let bsd = bs * d;

    // 0. Zero grads.
    for g in model.grads.iter_mut() { *g = 0.0; }

    // 1. Cross-entropy backward → dlogits [B*S, V]
    let mut dlogits = vec![0.0f32; bs * v];
    unsafe {
        ops::cross_entropy_backward_f32(
            act.probs.as_ptr(), labels.as_ptr(), dlogits.as_mut_ptr(), bs, v,
        );
    }

    // 2. Tied lm_head backward.
    //    logits = ln_f_out @ tok_emb^T
    //    d(ln_f_out) = dlogits @ tok_emb        [bs, V] @ [V, D] = [bs, D]
    //    d(tok_emb)  += dlogits^T @ ln_f_out    -> [V, D]
    let mut d_ln_f_out = vec![0.0f32; bsd];
    {
        let we_off = model.layout.tok_emb.0;
        for i in 0..bs {
            for di in 0..d {
                let mut acc = 0.0f32;
                for j in 0..v { acc += dlogits[i * v + j] * model.params[we_off + j * d + di]; }
                d_ln_f_out[i * d + di] = acc;
            }
        }
        for j in 0..v {
            for di in 0..d {
                let mut acc = 0.0f32;
                for i in 0..bs { acc += dlogits[i * v + j] * act.ln_f_out[i * d + di]; }
                model.grads[we_off + j * d + di] += acc;
            }
        }
    }

    // 3. Final layer norm backward.
    let mut d_x_after_blocks = vec![0.0f32; bsd];
    unsafe {
        let gamma_p = model.pp_const(model.layout.ln_f_gamma);
        let dgamma_p = model.gp(model.layout.ln_f_gamma);
        let dbeta_p  = model.gp(model.layout.ln_f_beta);
        ops::layer_norm_backward_f32(
            act.ln_f_in.as_ptr(),
            gamma_p,
            d_ln_f_out.as_ptr(),
            act.ln_f_mean.as_ptr(), act.ln_f_inv.as_ptr(),
            d_x_after_blocks.as_mut_ptr(),
            dgamma_p, dbeta_p,
            bs, d,
        );
    }

    // 4. Walk blocks in reverse.
    let mut dx = d_x_after_blocks;
    for li in (0..cfg.n_layers).rev() {
        let bl = model.layout.blocks[li].clone();
        let ba = &act.blocks[li];

        // Block residual: block_out = ffn_out + o
        // dffn = dx, do += dx
        let dffn = dx.clone();
        let mut d_o_resid = dx.clone();

        // FFN backward.
        // fc2 = fc1_post @ W_fc2 + b_fc2
        let mut d_fc1_post = vec![0.0f32; bs * f];
        unsafe {
            let dwfc2_p = model.gp(bl.w_fc2);
            let dbfc2_p = model.gp(bl.b_fc2);
            let w_fc2_p = model.pp_const(bl.w_fc2);
            ops::matmul_backward_rhs_f32(
                ba.fc1_post.as_ptr(), dffn.as_ptr(), dwfc2_p, bs, f, d,
            );
            for i in 0..bs { for di in 0..d {
                *dbfc2_p.add(di) += dffn[i * d + di];
            }}
            ops::matmul_backward_lhs_f32(
                dffn.as_ptr(), w_fc2_p, d_fc1_post.as_mut_ptr(), bs, f, d,
            );
        }
        // GELU backward.
        let mut d_fc1_pre = vec![0.0f32; bs * f];
        unsafe {
            ops::gelu_backward_f32(
                ba.fc1_pre.as_ptr(), d_fc1_post.as_ptr(), d_fc1_pre.as_mut_ptr(), bs * f,
            );
        }
        // fc1_pre = ln2_out @ W_fc1 + b_fc1
        let mut d_ln2_out = vec![0.0f32; bsd];
        unsafe {
            let dwfc1_p = model.gp(bl.w_fc1);
            let dbfc1_p = model.gp(bl.b_fc1);
            let w_fc1_p = model.pp_const(bl.w_fc1);
            ops::matmul_backward_rhs_f32(
                ba.ln2_out.as_ptr(), d_fc1_pre.as_ptr(), dwfc1_p, bs, d, f,
            );
            for i in 0..bs { for fi in 0..f {
                *dbfc1_p.add(fi) += d_fc1_pre[i * f + fi];
            }}
            ops::matmul_backward_lhs_f32(
                d_fc1_pre.as_ptr(), w_fc1_p, d_ln2_out.as_mut_ptr(), bs, d, f,
            );
        }

        // ln2 backward.
        let mut d_o_pre_ln2 = vec![0.0f32; bsd];
        unsafe {
            let gamma_p  = model.pp_const(bl.ln2_gamma);
            let dgamma_p = model.gp(bl.ln2_gamma);
            let dbeta_p  = model.gp(bl.ln2_beta);
            ops::layer_norm_backward_f32(
                ba.ln2_in.as_ptr(), gamma_p, d_ln2_out.as_ptr(),
                ba.ln2_mean.as_ptr(), ba.ln2_inv.as_ptr(),
                d_o_pre_ln2.as_mut_ptr(),
                dgamma_p, dbeta_p,
                bs, d,
            );
        }
        // d_o = d_o_resid + d_o_pre_ln2
        for i in 0..bsd { d_o_resid[i] += d_o_pre_ln2[i]; }
        let d_o = d_o_resid;

        // Attention block backward.
        // o = concat @ W_o + b_o + x_in
        let dx_in_residual = d_o.clone();
        let mut d_concat = vec![0.0f32; bsd];
        // Re-derive concat from attn_out (we didn't save it explicitly).
        let mut concat = vec![0.0f32; bsd];
        for bi in 0..b {
            for si in 0..s {
                for hi in 0..h {
                    for di in 0..hd {
                        let src = ((bi * h + hi) * s + si) * hd + di;
                        let dst = (bi * s + si) * d + hi * hd + di;
                        concat[dst] = ba.attn_out[src];
                    }
                }
            }
        }
        unsafe {
            let dwo_p = model.gp(bl.w_o);
            let dbo_p = model.gp(bl.b_o);
            let w_o_p = model.pp_const(bl.w_o);
            ops::matmul_backward_rhs_f32(
                concat.as_ptr(), d_o.as_ptr(), dwo_p, bs, d, d,
            );
            for i in 0..bs { for di in 0..d {
                *dbo_p.add(di) += d_o[i * d + di];
            }}
            ops::matmul_backward_lhs_f32(
                d_o.as_ptr(), w_o_p, d_concat.as_mut_ptr(), bs, d, d,
            );
        }

        // Reshape d_concat [B, S, D] -> d_attn_out [B*H, S, hd]
        let mut d_attn_out = vec![0.0f32; b * h * s * hd];
        for bi in 0..b {
            for si in 0..s {
                for hi in 0..h {
                    for di in 0..hd {
                        let src = (bi * s + si) * d + hi * hd + di;
                        let dst = ((bi * h + hi) * s + si) * hd + di;
                        d_attn_out[dst] = d_concat[src];
                    }
                }
            }
        }

        // SDPA backward.
        let mut dq = vec![0.0f32; b * h * s * hd];
        let mut dk = vec![0.0f32; b * h * s * hd];
        let mut dv = vec![0.0f32; b * h * s * hd];
        unsafe {
            ops::sdpa_causal_backward_f32(
                ba.q.as_ptr(), ba.k.as_ptr(), ba.v.as_ptr(),
                ba.attn.as_ptr(), d_attn_out.as_ptr(),
                dq.as_mut_ptr(), dk.as_mut_ptr(), dv.as_mut_ptr(),
                b * h, s, hd,
            );
        }

        // Pack dq/dk/dv back into d_qkv [B*S, 3D].
        let mut d_qkv = vec![0.0f32; bs * 3 * d];
        for bi in 0..b {
            for si in 0..s {
                let row_off = (bi * s + si) * 3 * d;
                for hi in 0..h {
                    for di in 0..hd {
                        let head_off = hi * hd + di;
                        let src = ((bi * h + hi) * s + si) * hd + di;
                        d_qkv[row_off + head_off]            += dq[src];
                        d_qkv[row_off + d + head_off]        += dk[src];
                        d_qkv[row_off + 2 * d + head_off]    += dv[src];
                    }
                }
            }
        }

        // qkv = ln1_out @ W_qkv + b_qkv
        let mut d_ln1_out = vec![0.0f32; bsd];
        unsafe {
            let dwqkv_p = model.gp(bl.w_qkv);
            let dbqkv_p = model.gp(bl.b_qkv);
            let w_qkv_p = model.pp_const(bl.w_qkv);
            ops::matmul_backward_rhs_f32(
                ba.ln1_out.as_ptr(), d_qkv.as_ptr(), dwqkv_p, bs, d, 3 * d,
            );
            for i in 0..bs { for di in 0..3*d {
                *dbqkv_p.add(di) += d_qkv[i * 3 * d + di];
            }}
            ops::matmul_backward_lhs_f32(
                d_qkv.as_ptr(), w_qkv_p, d_ln1_out.as_mut_ptr(), bs, d, 3 * d,
            );
        }

        // ln1 backward.
        let mut d_x_pre_block = vec![0.0f32; bsd];
        unsafe {
            let gamma_p  = model.pp_const(bl.ln1_gamma);
            let dgamma_p = model.gp(bl.ln1_gamma);
            let dbeta_p  = model.gp(bl.ln1_beta);
            ops::layer_norm_backward_f32(
                ba.ln1_in.as_ptr(), gamma_p, d_ln1_out.as_ptr(),
                ba.ln1_mean.as_ptr(), ba.ln1_inv.as_ptr(),
                d_x_pre_block.as_mut_ptr(),
                dgamma_p, dbeta_p,
                bs, d,
            );
        }
        // d_x = d_x_pre_block + dx_in_residual
        for i in 0..bsd { d_x_pre_block[i] += dx_in_residual[i]; }
        dx = d_x_pre_block;
    }

    // 5. Embedding & positional grads.
    unsafe {
        let dwe_p = model.gp(model.layout.tok_emb);
        ops::embedding_backward_f32(ids.as_ptr(), dx.as_ptr(), dwe_p, b, s, cfg.vocab, d);
    }
    let pos_off = model.layout.pos_emb.0;
    for bi in 0..b {
        for si in 0..s {
            for di in 0..d {
                model.grads[pos_off + si * d + di] += dx[(bi * s + si) * d + di];
            }
        }
    }
}

/// Single AdamW step on the whole param arena.
pub fn adamw_step(
    model: &mut Model, lr: f32, beta1: f32, beta2: f32, eps: f32, wd: f32, step: i64,
) {
    let n = model.layout.total;
    unsafe {
        ops::adamw_step_f32(
            model.params.as_mut_ptr(),
            model.grads.as_ptr(),
            model.adam_m.as_mut_ptr(),
            model.adam_v.as_mut_ptr(),
            lr, beta1, beta2, eps, wd, step, n,
        );
    }
}

pub fn clip_grads(model: &mut Model, max_norm: f32) -> f32 {
    let n = model.layout.total;
    unsafe { ops::clip_grad_norm_f32(model.grads.as_mut_ptr(), max_norm, n) }
}
