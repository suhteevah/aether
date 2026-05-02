//! Model + training config. Matches the spec at the top of
//! `examples/aether_lm.aether` exactly.

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub vocab: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub seq_len: usize,
}

impl ModelConfig {
    pub fn nano_cpu() -> Self {
        // Tiny enough for a CPU smoke run that completes in seconds.
        Self { vocab: 256, d_model: 64, n_layers: 2, n_heads: 4, d_ff: 128, seq_len: 32 }
    }

    pub fn tiny_3070ti() -> Self {
        // The full AetherLM-Tiny target.
        Self { vocab: 256, d_model: 320, n_layers: 6, n_heads: 5, d_ff: 1280, seq_len: 256 }
    }

    pub fn head_dim(&self) -> usize {
        assert!(self.d_model % self.n_heads == 0, "d_model must divide n_heads");
        self.d_model / self.n_heads
    }

    pub fn num_params(&self) -> usize {
        let v = self.vocab; let d = self.d_model;
        let l = self.n_layers; let f = self.d_ff;
        let emb = v * d;
        let pos = self.seq_len * d;
        let per_layer_attn = 4 * d * d + 4 * d;             // qkv (3d), out, biases
        let per_layer_ff = 2 * d * f + d + f;
        let per_layer_norm = 2 * 2 * d;
        let final_norm = 2 * d;
        emb + pos + l * (per_layer_attn + per_layer_ff + per_layer_norm) + final_norm
    }
}

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub batch_size: usize,
    pub steps: usize,
    pub lr: f32,
    pub warmup: usize,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub log_every: usize,
    pub seed: u64,
}

impl TrainConfig {
    pub fn smoke() -> Self {
        Self {
            batch_size: 8, steps: 100, lr: 3e-3, warmup: 10,
            weight_decay: 0.0, grad_clip: 1.0, log_every: 10, seed: 42,
        }
    }
}
