//! GLM-4.7-flash probe smoke (FR-17-extra-mla-fwd head_dim cap validation).
//!
//! Opens a GGUF, dumps `ModelConfig` (so we can eyeball MLA geometry), and
//! replicates the QwenSession session-construction `head_dim` guard.  Exits
//! before any tensor upload — the goal is to validate that:
//!   1. GGUF metadata reads cleanly
//!   2. ModelConfig::from_gguf populates qk_head_dim/v_head_dim/kv_lora_rank
//!   3. The new MLA branch of the construction guard accepts the geometry
//!
//! Designed for the GPU 0 (12GB P100) cnc smoke — peak VRAM stays in
//! metadata-only territory, never approaching the full-load 13GB.
//!
//! Usage: glm-probe <path-to-gguf>

use std::env;
use std::process::ExitCode;
use aether_rt::serving::ModelConfig;
use aether_rt::{aether_gguf_open, aether_gguf_close};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: glm-probe <gguf>");
        return ExitCode::from(2);
    }
    let path = &args[1];

    unsafe {
        let h = aether_gguf_open(path.as_ptr() as i64, path.len() as std::os::raw::c_int);
        if h < 0 {
            eprintln!("[glm-probe] aether_gguf_open failed for {} (rc={})", path, h);
            return ExitCode::from(3);
        }
        println!("[glm-probe] opened: {}", path);

        let cfg = ModelConfig::from_gguf(h);
        println!("[glm-probe] arch              = {}", cfg.arch);
        println!("[glm-probe] n_layers          = {}", cfg.n_layers);
        println!("[glm-probe] d_model           = {}", cfg.d_model);
        println!("[glm-probe] n_q_heads         = {}", cfg.n_q_heads);
        println!("[glm-probe] n_kv_heads        = {}", cfg.n_kv_heads);
        println!("[glm-probe] head_dim          = {}", cfg.head_dim);
        println!("[glm-probe] d_kv              = {}", cfg.d_kv);
        println!("[glm-probe] d_ff              = {}", cfg.d_ff);
        println!("[glm-probe] vocab             = {}", cfg.vocab);
        println!("[glm-probe] rope_base         = {}", cfg.rope_base);
        println!("[glm-probe] norm_eps          = {:.2e}", cfg.norm_eps);
        println!("[glm-probe] n_experts         = {}", cfg.n_experts);
        println!("[glm-probe] n_experts_used    = {}", cfg.n_experts_used);
        println!("[glm-probe] n_shared_experts  = {}", cfg.n_shared_experts);
        println!("[glm-probe] expert_ff_dim     = {}", cfg.expert_ff_dim);
        println!("[glm-probe] leading_dense     = {}", cfg.leading_dense_blocks);
        println!("[glm-probe] sliding_window    = {}", cfg.sliding_window);
        println!("[glm-probe] -- MLA geometry --");
        println!("[glm-probe] kv_lora_rank      = {}", cfg.kv_lora_rank);
        println!("[glm-probe] q_lora_rank       = {}", cfg.q_lora_rank);
        println!("[glm-probe] qk_head_dim       = {}", cfg.qk_head_dim);
        println!("[glm-probe] qk_rope_head_dim  = {}", cfg.qk_rope_head_dim);
        println!("[glm-probe] v_head_dim        = {}", cfg.v_head_dim);
        println!("[glm-probe] -- YaRN --");
        println!("[glm-probe] yarn_factor       = {}", cfg.yarn_factor);
        println!("[glm-probe] yarn_log_mult     = {}", cfg.yarn_log_multiplier);
        println!("[glm-probe] yarn_orig_ctx     = {}", cfg.yarn_orig_ctx);

        // Mirror the QwenSession session-construction `head_dim` guard from
        // serving.rs.  This is the exact code path we need to prove accepts
        // GLM-4.7-flash geometry.
        let guard_result: Result<&'static str, String> = if cfg.kv_lora_rank > 0 {
            if cfg.qk_head_dim <= 0 || cfg.qk_head_dim > 640 {
                Err(format!(
                    "FR-17-extra-mla-fwd: qk_head_dim={} out of range [1, 640]",
                    cfg.qk_head_dim))
            } else if cfg.v_head_dim <= 0 || cfg.v_head_dim > 640 {
                Err(format!(
                    "FR-17-extra-mla-fwd: v_head_dim={} out of range [1, 640]",
                    cfg.v_head_dim))
            } else {
                Ok("MLA branch accepted (qk≤640, v≤640)")
            }
        } else if cfg.head_dim == 0 || cfg.head_dim > 256 {
            Err(format!(
                "FR-17-extra-runtime-shape: head_dim={} out of range [1, 256]",
                cfg.head_dim))
        } else {
            Ok("non-MLA branch accepted (head_dim≤256)")
        };

        println!("[glm-probe] -- head_dim guard --");
        match &guard_result {
            Ok(msg)  => println!("[glm-probe] guard PASS: {}", msg),
            Err(msg) => println!("[glm-probe] guard FAIL: {}", msg),
        }

        aether_gguf_close(h);
        if guard_result.is_ok() {
            println!("[glm-probe] OK — early exit before tensor upload.");
            ExitCode::from(0)
        } else {
            ExitCode::from(1)
        }
    }
}
