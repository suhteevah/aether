//! aether-serve — minimal OpenAI-compatible HTTP server for matt-voice.
//!
//! Composes the shipped Aether runtime pieces:
//!   - aether_tcp_listen / accept_one / recv / send
//!   - aether_http_parse_request / write_response_200
//!   - aether_openai_render_completion
//!   - (optional) aether_gguf_open + 28-block forward with KV cache +
//!     aether_op_apply_lora_f32 for matt-voice's adapter
//!
//! Usage:
//!   aether-serve --port 8080
//!     -> bind to localhost:8080, accept requests
//!     -> respond with OpenAI-compat JSON (currently a stub response;
//!        wiring of real Qwen forward + LoRA application is the next
//!        FR-x-extra increment).
//!
//! Test it:
//!   curl -X POST http://localhost:8080/v1/chat/completions \
//!     -H 'Content-Type: application/json' \
//!     -d '{"prompt_ids":[9707,11,1879,0],"max_tokens":5}'
//!
//! The HANDOFF.md tracks how to plug the Qwen-forward integration:
//! the qwen25_autoregressive_cuda test is the reference impl for
//! the per-request forward chain.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_tcp_listen, aether_tcp_listener_port, aether_tcp_accept_one,
    aether_tcp_send, aether_tcp_recv, aether_tcp_close, aether_tcp_stream_close,
    aether_http_parse_request, aether_http_write_response_200,
    aether_openai_render_completion,
};

#[derive(Debug)]
struct Cli {
    port: i64,
    model: String,
}

fn parse_cli() -> Cli {
    let mut cli = Cli { port: 8080, model: "matt-voice".into() };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port"  => cli.port  = it.next().unwrap().parse().unwrap(),
            "--model" => cli.model = it.next().unwrap(),
            "-h" | "--help" => {
                eprintln!("aether-serve [--port N] [--model NAME]");
                eprintln!("  Listens on 0.0.0.0:port for OpenAI-compat /v1/chat/completions");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }
    cli
}

unsafe fn handle_request(stream: i64, model: &str) {
    // 1. Read the request (one buffer; assumes Content-Length fits).
    let mut req_buf = vec![0u8; 16384];
    let got = aether_tcp_recv(stream, req_buf.as_mut_ptr() as i64, req_buf.len() as i64);
    if got <= 0 {
        eprintln!("[serve] recv returned {}", got);
        return;
    }
    let req_bytes = &req_buf[..got as usize];

    // 2. Parse method + path.
    let mut strings = vec![0u8; 512];
    let mut m_len: c_int = 0;
    let mut p_len: c_int = 0;
    let body_off = aether_http_parse_request(
        req_bytes.as_ptr() as *const c_void, got as c_int,
        strings.as_mut_ptr() as *mut c_void, strings.len() as c_int,
        &mut m_len, &mut p_len,
    );
    if body_off <= 0 {
        eprintln!("[serve] bad request");
        return;
    }
    let method = std::str::from_utf8(&strings[..m_len as usize]).unwrap_or("");
    let path = std::str::from_utf8(&strings[m_len as usize..(m_len + p_len) as usize]).unwrap_or("");
    let body = &req_bytes[body_off as usize..];
    eprintln!("[serve] {} {} body_len={}", method, path, body.len());

    // 3. Currently a STUB response. The integration of the real Qwen
    //    forward chain (qwen25_autoregressive_cuda.rs) lives here:
    //    - parse prompt_ids + max_tokens from the JSON body
    //    - run forward over 28 blocks + KV cache
    //    - decode generated IDs via aether_bpe_decode + GPT-2 byte fixup
    //    - return generated text below
    let generated_text = "[aether-serve stub: integrate qwen25_autoregressive_cuda forward here]";
    let prompt_tokens = body.len() as c_int / 4;  // rough
    let completion_tokens = 0;

    let resp_id = b"chatcmpl-aether-serve-1";
    let mut json_buf = vec![0u8; 4096];
    let n_json = aether_openai_render_completion(
        resp_id.as_ptr() as *const c_void, resp_id.len() as c_int,
        model.as_ptr() as *const c_void, model.len() as c_int,
        generated_text.as_ptr() as *const c_void, generated_text.len() as c_int,
        prompt_tokens, completion_tokens,
        json_buf.as_mut_ptr() as *mut c_void, json_buf.len() as c_int,
    );
    if n_json <= 0 {
        eprintln!("[serve] render failed: {}", n_json);
        return;
    }

    let mut http_buf = vec![0u8; 8192];
    let n_http = aether_http_write_response_200(
        json_buf.as_ptr() as *const c_void, n_json,
        http_buf.as_mut_ptr() as *mut c_void, http_buf.len() as c_int,
    );
    if n_http <= 0 { return; }

    let sent = aether_tcp_send(stream, http_buf.as_ptr() as i64, n_http as i64);
    if sent != n_http as i64 {
        eprintln!("[serve] send sent={}, expected {}", sent, n_http);
    }
}

fn main() {
    let cli = parse_cli();
    unsafe {
        let listener = aether_tcp_listen(cli.port);
        if listener < 0 {
            eprintln!("[serve] failed to bind port {}", cli.port);
            std::process::exit(1);
        }
        let bound_port = aether_tcp_listener_port(listener);
        eprintln!("[aether-serve] listening on port {} (model={})", bound_port, cli.model);
        eprintln!("[aether-serve] try: curl -X POST http://localhost:{}/v1/chat/completions \\", bound_port);
        eprintln!("                       -H 'Content-Type: application/json' \\");
        eprintln!("                       -d '{{\"prompt_ids\":[9707,11,1879,0],\"max_tokens\":5}}'");

        // Serve until Ctrl+C.
        loop {
            let stream = aether_tcp_accept_one(listener);
            if stream < 0 {
                eprintln!("[serve] accept returned {} (continuing)", stream);
                continue;
            }
            handle_request(stream, &cli.model);
            aether_tcp_stream_close(stream);
        }
        // unreachable, but keep the close for completeness
        #[allow(unreachable_code)]
        { aether_tcp_close(listener); }
    }
}
