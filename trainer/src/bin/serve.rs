//! aether-serve — OpenAI-compatible HTTP server backed by a real Qwen
//! forward pass on GPU.
//!
//! Composes the shipped Aether runtime pieces:
//!   - aether_tcp_listen / accept_one / recv / send  (TCP)
//!   - aether_http_parse_request / write_response_200  (HTTP/1.1)
//!   - aether_openai_render_completion  (response JSON shape)
//!   - aether_rt::serving::QwenSession  (real CUDA-graph-captured Qwen2.5
//!     forward; weights stay GPU-resident across requests)
//!
//! Build:
//!   cargo build -p trainer --bin aether-serve --features cuda --release
//!
//! Run:
//!   target/release/aether-serve.exe \
//!     --port 8080 \
//!     --gguf "C:\Users\Matt\.ollama\models\blobs\sha256-2bada8a7..."
//!
//! Smoke test (token-id input — no tokenizer encode round-trip needed):
//!   curl -X POST http://localhost:8080/v1/chat/completions \
//!     -H 'Content-Type: application/json' \
//!     -d '{"model":"qwen2.5","prompt_ids":[9707,11,1879,0],"max_tokens":16}'
//!
//! Without `--features cuda`, the binary builds but `--gguf` rejects with
//! a clear error and the request path returns the historic stub. This
//! keeps the HTTP/JSON pieces testable on machines without CUDA.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_tcp_listen, aether_tcp_listener_port, aether_tcp_accept_one,
    aether_tcp_send, aether_tcp_recv, aether_tcp_close, aether_tcp_stream_close,
    aether_http_parse_request, aether_http_write_response_200,
    aether_openai_render_completion,
};

#[cfg(feature = "cuda")]
use aether_rt::serving::QwenSession;

#[derive(Debug)]
struct Cli {
    port: i64,
    model: String,
    gguf: Option<String>,
    max_tokens_default: usize,
    stop_token: Option<usize>,
    warmup: usize,
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        port: 8080,
        model: "qwen2.5-7b-instruct".into(),
        gguf: None,
        max_tokens_default: 64,
        stop_token: None,  // QwenSession.eos_token used as default
        warmup: 4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--port"  => cli.port  = it.next().expect("--port N").parse().expect("port int"),
            "--model" => cli.model = it.next().expect("--model NAME"),
            "--gguf"  => cli.gguf  = Some(it.next().expect("--gguf PATH")),
            "--max-tokens" => cli.max_tokens_default =
                it.next().expect("--max-tokens N").parse().expect("max-tokens int"),
            "--stop-token" => {
                let s = it.next().expect("--stop-token ID");
                cli.stop_token = if s == "none" { None }
                                 else { Some(s.parse().expect("stop-token int")) };
            }
            "--warmup" => cli.warmup = it.next().expect("--warmup N").parse().expect("warmup int"),
            "-h" | "--help" => {
                eprintln!("aether-serve [--port N] [--model NAME] [--gguf PATH] [--max-tokens N] [--stop-token ID|none] [--warmup N]");
                eprintln!();
                eprintln!("  Listens on 0.0.0.0:port for OpenAI-compat /v1/chat/completions.");
                eprintln!("  --gguf points at any Qwen2.5-7B-Instruct Q4_K_M model file.");
                eprintln!("  --warmup N runs N synthetic decode steps on startup to drive");
                eprintln!("    the GPU into P0/P2 power state and pre-capture the graph.");
                eprintln!("  Without --gguf, returns a stub response (HTTP/JSON plumbing only).");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }
    cli
}

// ---- minimal JSON body parser (only the fields we care about) ----
//
// Supports: {"model":"...", "prompt_ids":[1,2,3], "max_tokens": 16,
//            "messages":[{"role":"user","content":"..."}],
//            "stream": false}
//
// We don't write a full JSON parser; we cherry-pick the keys with simple
// substring + integer/array scans. Robust enough for the OpenAI client
// shape; refused for anything weird.

struct JsonBody {
    prompt_ids: Vec<usize>,
    max_tokens: usize,
    stream: bool,
    /// Best-effort surface of `messages[*].content` joined with "\n".
    /// Used only when `prompt_ids` is absent (FR-x-extra: BPE encode).
    text_prompt: Option<String>,
}

fn parse_body(body: &[u8], default_max: usize) -> Result<JsonBody, &'static str> {
    let s = std::str::from_utf8(body).map_err(|_| "body not utf-8")?;

    let prompt_ids = match find_key_array(s, "prompt_ids") {
        Some(arr) => parse_int_array(arr)?,
        None => Vec::new(),
    };

    let max_tokens = find_key_int(s, "max_tokens")
        .unwrap_or(default_max as i64) as usize;

    let stream = find_key_bool(s, "stream").unwrap_or(false);

    let text_prompt = find_messages_content(s);

    if prompt_ids.is_empty() && text_prompt.is_none() {
        return Err("body has neither prompt_ids nor messages[].content");
    }

    Ok(JsonBody { prompt_ids, max_tokens, stream, text_prompt })
}

/// Find `"key": [ ... ]` and return the slice inside the brackets.
fn find_key_array<'a>(s: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\"", key);
    let i = s.find(&pat)?;
    let after_key = &s[i + pat.len()..];
    let colon = after_key.find(':')?;
    let after_colon = &after_key[colon + 1..];
    let lb = after_colon.find('[')?;
    let after_lb = &after_colon[lb + 1..];
    let rb = after_lb.find(']')?;
    Some(&after_lb[..rb])
}

/// Parse a flat array of integers (space/comma separated, no nesting).
fn parse_int_array(s: &str) -> Result<Vec<usize>, &'static str> {
    let mut out = Vec::new();
    for chunk in s.split(|c: char| c == ',' || c.is_whitespace()) {
        let t = chunk.trim();
        if t.is_empty() { continue; }
        let v: usize = t.parse().map_err(|_| "non-integer in prompt_ids")?;
        out.push(v);
    }
    Ok(out)
}

fn find_key_int(s: &str, key: &str) -> Option<i64> {
    let pat = format!("\"{}\"", key);
    let i = s.find(&pat)?;
    let after_key = &s[i + pat.len()..];
    let colon = after_key.find(':')?;
    let after_colon = &after_key[colon + 1..];
    // Skip leading whitespace.
    let trimmed = after_colon.trim_start();
    // Take while digit or sign.
    let end = trimmed.find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(trimmed.len());
    let num = &trimmed[..end];
    num.parse().ok()
}

fn find_key_bool(s: &str, key: &str) -> Option<bool> {
    let pat = format!("\"{}\"", key);
    let i = s.find(&pat)?;
    let after_key = &s[i + pat.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    if after_colon.starts_with("true")  { Some(true) }
    else if after_colon.starts_with("false") { Some(false) }
    else { None }
}

/// Hack: walk `messages` looking for `"content": "..."`, join with \n.
fn find_messages_content(s: &str) -> Option<String> {
    let key = "\"messages\"";
    let i = s.find(key)?;
    let mut cursor = i + key.len();
    let mut parts: Vec<String> = Vec::new();
    while let Some(rel) = s[cursor..].find("\"content\"") {
        let abs = cursor + rel + "\"content\"".len();
        let after = &s[abs..];
        if let Some(colon) = after.find(':') {
            let after_colon = after[colon + 1..].trim_start();
            if after_colon.starts_with('"') {
                let q_start = (after.as_ptr() as usize) - (s.as_ptr() as usize)
                              + colon + 1 + (after[colon + 1..].len() - after_colon.len()) + 1;
                // Find matching closing quote (no escape handling beyond \").
                let mut j = q_start;
                let bytes = s.as_bytes();
                while j < bytes.len() {
                    if bytes[j] == b'"' && bytes[j - 1] != b'\\' { break; }
                    j += 1;
                }
                if j < bytes.len() {
                    let raw = &s[q_start..j];
                    parts.push(unescape_json_string(raw));
                    cursor = j + 1;
                    continue;
                }
            }
        }
        cursor = abs;
    }
    if parts.is_empty() { None } else { Some(parts.join("\n")) }
}

fn unescape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('"') => out.push('"'),
                Some('\\') => out.push('\\'),
                Some(other) => { out.push('\\'); out.push(other); }
                None => out.push('\\'),
            }
        } else { out.push(c); }
    }
    out
}

// ---- request dispatch ----

struct ServerState {
    cli: Cli,
    #[cfg(feature = "cuda")]
    session: Option<std::sync::Mutex<QwenSession>>,
}

impl ServerState {
    fn new(cli: Cli) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            let session = match &cli.gguf {
                Some(path) => {
                    eprintln!("[aether-serve] loading GGUF: {}", path);
                    let t = std::time::Instant::now();
                    let mut s = QwenSession::new(path)?;
                    eprintln!("[aether-serve] model loaded in {:.2}s", t.elapsed().as_secs_f32());
                    if cli.warmup > 0 {
                        eprintln!("[aether-serve] warming GPU + capturing graph ({} steps)...", cli.warmup);
                        let t = std::time::Instant::now();
                        s.warmup(cli.warmup);
                        eprintln!("[aether-serve] warmup done in {:.2}s", t.elapsed().as_secs_f32());
                    }
                    Some(std::sync::Mutex::new(s))
                }
                None => {
                    eprintln!("[aether-serve] no --gguf supplied; requests will return STUB responses");
                    None
                }
            };
            return Ok(ServerState { cli, session });
        }
        #[cfg(not(feature = "cuda"))]
        {
            if cli.gguf.is_some() {
                return Err("--gguf requires building with --features cuda".into());
            }
            Ok(ServerState { cli })
        }
    }
}

unsafe fn handle_request(state: &ServerState, stream: i64) {
    // 1. Read request bytes (one buffer for now; Content-Length must fit).
    let mut req_buf = vec![0u8; 65536];
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
        send_text(stream, 400, "bad request");
        return;
    }
    let method = std::str::from_utf8(&strings[..m_len as usize]).unwrap_or("");
    let path = std::str::from_utf8(&strings[m_len as usize..(m_len + p_len) as usize]).unwrap_or("");
    let body = &req_bytes[body_off as usize..];
    eprintln!("[serve] {} {} body_len={}", method, path, body.len());

    match (method, path) {
        ("GET", "/health") => send_text(stream, 200, "ok"),
        ("GET", "/v1/models") => handle_list_models(state, stream),
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            handle_completion(state, stream, body)
        }
        _ => send_text(stream, 404, "not found"),
    }
}

unsafe fn handle_list_models(state: &ServerState, stream: i64) {
    let body = format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"aether\"}}]}}",
        state.cli.model);
    send_json(stream, 200, &body);
}

unsafe fn handle_completion(state: &ServerState, stream: i64, body: &[u8]) {
    let req = match parse_body(body, state.cli.max_tokens_default) {
        Ok(r) => r,
        Err(e) => { send_text(stream, 400, e); return; }
    };

    if req.text_prompt.is_some() && req.prompt_ids.is_empty() {
        // FR-x-extra: BPE encode pending. For now, reject with a useful
        // hint so the caller knows what to send.
        send_text(stream, 501,
            "text-prompt encode not wired yet; pass prompt_ids (token ids) for now");
        return;
    }

    if req.stream {
        handle_completion_streaming(state, stream, &req);
        return;
    }

    let generated_text: String;
    let prompt_tokens = req.prompt_ids.len() as c_int;
    let completion_tokens: c_int;

    #[cfg(feature = "cuda")]
    {
        match &state.session {
            Some(sess_mu) => {
                let mut sess = sess_mu.lock().unwrap();
                let stop = state.cli.stop_token.or_else(|| {
                    if sess.eos_token >= 0 { Some(sess.eos_token as usize) } else { None }
                });
                let t = std::time::Instant::now();
                let ids = sess.generate(&req.prompt_ids, req.max_tokens, stop);
                let secs = t.elapsed().as_secs_f32();
                eprintln!("[serve] generated {} tokens in {:.3}s = {:.2} tok/s",
                    ids.len(), secs, ids.len() as f32 / secs.max(1e-6));
                completion_tokens = ids.len() as c_int;
                // Decode IDs back to real text via the loaded BPE tokenizer
                // + GPT-2 byte fixup. Falls back to id list if tokenizer
                // wasn't loaded (non-Qwen GGUF without ggml.tokens).
                let text = sess.decode_ids(&ids);
                generated_text = if text.is_empty() {
                    format_id_list(&ids)
                } else {
                    text
                };
            }
            None => {
                generated_text = "[aether-serve stub: --gguf not supplied]".into();
                completion_tokens = 0;
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        generated_text = "[aether-serve stub: built without --features cuda]".into();
        completion_tokens = 0;
        let _ = state;  // silence unused
    }

    let resp_id = b"chatcmpl-aether-serve-1";
    // JSON-escape the content string before handing it to the renderer
    // (the renderer interpolates raw bytes; we own escaping per its doc).
    let escaped = json_escape(&generated_text);
    let mut json_buf = vec![0u8; 65536];
    let n_json = aether_openai_render_completion(
        resp_id.as_ptr() as *const c_void, resp_id.len() as c_int,
        state.cli.model.as_ptr() as *const c_void, state.cli.model.len() as c_int,
        escaped.as_ptr() as *const c_void, escaped.len() as c_int,
        prompt_tokens, completion_tokens,
        json_buf.as_mut_ptr() as *mut c_void, json_buf.len() as c_int,
    );
    if n_json <= 0 {
        send_text(stream, 500, "render failed");
        return;
    }

    let mut http_buf = vec![0u8; 131072];
    let n_http = aether_http_write_response_200(
        json_buf.as_ptr() as *const c_void, n_json,
        http_buf.as_mut_ptr() as *mut c_void, http_buf.len() as c_int,
    );
    if n_http <= 0 {
        send_text(stream, 500, "http write failed");
        return;
    }

    let _ = aether_tcp_send(stream, http_buf.as_ptr() as i64, n_http as i64);

    // SSE streaming knob is wired (req.stream) — Phase 3 task will emit
    // real `data: {...}\n\n` chunks per token. Single-shot today.
    let _ = req.stream;
}

/// Render a list of token ids as a comma-joined string so the response
/// is round-trippable (the client can BPE-decode them with their own
/// tokenizer). When BPE encode/decode round-trip lands in aether_rt
/// we'll switch this to actual text.
#[cfg(feature = "cuda")]
fn format_id_list(ids: &[usize]) -> String {
    let mut s = String::with_capacity(ids.len() * 6);
    s.push_str("[ids] ");
    for (i, id) in ids.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&id.to_string());
    }
    s
}

/// SSE streaming variant of /v1/chat/completions. Sends one `data:` chunk
/// per generated token, decoded to real text via the loaded BPE
/// tokenizer + GPT-2 byte fixup. Terminates with `data: [DONE]\n\n`.
///
/// Uses HTTP/1.1 chunked transfer encoding so the response can stream
/// without knowing the final length in advance.
unsafe fn handle_completion_streaming(state: &ServerState, stream: i64, req: &JsonBody) {
    // Send headers immediately.
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";
    let _ = aether_tcp_send(stream, headers.as_ptr() as i64, headers.len() as i64);

    let send_chunk = |s: &str| {
        let hex_len = format!("{:x}\r\n", s.len());
        let _ = aether_tcp_send(stream, hex_len.as_ptr() as i64, hex_len.len() as i64);
        let _ = aether_tcp_send(stream, s.as_ptr() as i64, s.len() as i64);
        let _ = aether_tcp_send(stream, "\r\n".as_ptr() as i64, 2);
    };

    #[cfg(feature = "cuda")]
    {
        if let Some(sess_mu) = &state.session {
            let mut sess = sess_mu.lock().unwrap();
            let stop = state.cli.stop_token.or_else(|| {
                if sess.eos_token >= 0 { Some(sess.eos_token as usize) } else { None }
            });
            sess.reset();
            sess.prefill(&req.prompt_ids);
            let mut last = *req.prompt_ids.last().unwrap();
            for _ in 0..req.max_tokens {
                let id = sess.decode_step(last);
                if Some(id) == stop { break; }
                let piece = sess.decode_ids(&[id]);
                let escaped = json_escape(&piece);
                let chunk = format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
                    escaped);
                send_chunk(&chunk);
                last = id;
            }
            send_chunk("data: [DONE]\n\n");
        } else {
            send_chunk("data: {\"error\":\"--gguf not supplied\"}\n\n");
            send_chunk("data: [DONE]\n\n");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = state; let _ = req;
        send_chunk("data: {\"error\":\"built without --features cuda\"}\n\n");
        send_chunk("data: [DONE]\n\n");
    }

    // Final zero-length chunk to terminate transfer-encoding.
    let _ = aether_tcp_send(stream, "0\r\n\r\n".as_ptr() as i64, 5);
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

unsafe fn send_text(stream: i64, code: i32, msg: &str) {
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code, http_status_text(code), msg.len(), msg);
    let _ = aether_tcp_send(stream, resp.as_ptr() as i64, resp.len() as i64);
}

unsafe fn send_json(stream: i64, code: i32, body: &str) {
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code, http_status_text(code), body.len(), body);
    let _ = aether_tcp_send(stream, resp.as_ptr() as i64, resp.len() as i64);
}

fn http_status_text(code: i32) -> &'static str {
    match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found",
        500 => "Internal Server Error", 501 => "Not Implemented",
        _ => "Unknown",
    }
}

fn main() {
    let cli = parse_cli();
    let state = match ServerState::new(cli) {
        Ok(s) => s,
        Err(e) => { eprintln!("[aether-serve] startup error: {}", e); std::process::exit(1); }
    };
    unsafe {
        let listener = aether_tcp_listen(state.cli.port);
        if listener < 0 {
            eprintln!("[aether-serve] failed to bind port {}", state.cli.port);
            std::process::exit(1);
        }
        let bound_port = aether_tcp_listener_port(listener);
        eprintln!("[aether-serve] listening on 0.0.0.0:{} (model={})", bound_port, state.cli.model);
        eprintln!("[aether-serve] try:");
        eprintln!("  curl http://localhost:{}/v1/models", bound_port);
        eprintln!("  curl http://localhost:{}/health", bound_port);
        eprintln!("  curl -X POST http://localhost:{}/v1/chat/completions \\", bound_port);
        eprintln!("       -H 'Content-Type: application/json' \\");
        eprintln!("       -d '{{\"prompt_ids\":[9707,11,1879,0],\"max_tokens\":16}}'");

        loop {
            let stream = aether_tcp_accept_one(listener);
            if stream < 0 {
                eprintln!("[serve] accept returned {} (continuing)", stream);
                continue;
            }
            handle_request(&state, stream);
            aether_tcp_stream_close(stream);
        }
        #[allow(unreachable_code)]
        { aether_tcp_close(listener); }
    }
}
