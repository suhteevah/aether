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
    aether_tcp_listen, aether_tcp_listen_addr,
    aether_tcp_listener_port, aether_tcp_accept_one,
    aether_tcp_send, aether_tcp_recv, aether_tcp_close, aether_tcp_stream_close,
    aether_http_parse_request, aether_http_write_response_200,
    aether_openai_render_completion,
    aether_random_bytes,
    tls13::TlsServerSession,
    http2,
};

#[cfg(feature = "cuda")]
use aether_rt::serving::{QwenSession, SharedKvPool};
#[cfg(feature = "cuda")]
use aether_rt::bert::{BertSession, WordPieceTokenizer};
#[cfg(feature = "cuda")]
use aether_rt::{aether_gguf_open, aether_gguf_close};

#[derive(Debug)]
struct Cli {
    port: i64,
    model: String,
    gguf: Option<String>,
    max_tokens_default: usize,
    stop_token: Option<usize>,
    warmup: usize,
    tls: bool,
    tls_cn: String,
    /// FR-19.4-extra: route K/V through the paged kernels with an identity
    /// page table (block_size=4).  Bit-identical token output to the
    /// contiguous path (witnessed in qwen25_paged_parity.rs).
    paged: bool,
    /// FR-19.4-extra-tenant: when >0, allocate a SharedKvPool of this many
    /// blocks (block_size=4 tokens) and route every aether-serve session
    /// through it.  Implies --paged.  Sessions return their blocks to the
    /// pool on Drop, so the pool effectively caps concurrent KV memory
    /// rather than letting per-request KV grow unbounded.
    pool_blocks: i32,
    /// FR-17-extra-runtime-shape: probe mode — open the GGUF, print the
    /// detected ModelConfig (architecture + shape + rope/eps), exit.  No
    /// weight upload, no listener.  Useful for confirming the runtime-shape
    /// detector picks up a new model correctly before trying to serve it.
    probe: bool,
    /// FR-17-extra-bert-fwd: BERT/BGE encoder model for /v1/embeddings.
    /// Loaded as a sibling to the chat-completions QwenSession; the same
    /// aether-serve process can host both endpoints if both flags are set.
    bge_gguf: Option<String>,
    /// Embedding-model name reported back in /v1/embeddings responses
    /// (separate from --model which names the chat-completions model).
    bge_model: String,
    /// FR-x-extra: bind address — default "0.0.0.0" so the server is
    /// reachable from podman bridges / other hosts / etc.  The prior
    /// behavior (via `aether_tcp_listen`) hardcoded `127.0.0.1`, which
    /// blocked LiteLLM-in-podman from reaching the server (per kokonoe
    /// substrate-swap finding #3).  Pass `--bind 127.0.0.1` to restore
    /// the loopback-only behavior.
    bind: String,
    /// FR-x-extra-sampling: default sampler parameters when the request
    /// body omits `temperature` / `top_p`.  0.0 → greedy argmax.
    default_temperature: f32,
    default_top_p: f32,
    /// OpenAI-compat repetition penalty defaults.  0.0 = off.
    /// `0.3 / 0.3` is a gentle starting point that breaks the
    /// degenerate "loop on prompt suffix" pattern without making
    /// outputs jittery.
    default_presence_penalty: f32,
    default_frequency_penalty: f32,
    /// FR-x-extra-tp — tensor-parallel world size.  Default 1 (single
    /// GPU, behaviour unchanged).  `--tp 2` requests 2-way tensor
    /// parallelism; runtime-detects NCCL + multi-GPU availability and
    /// falls back to TP=1 with a warning if unavailable.
    tp: usize,
    /// FR-19.5-extra-deep — continuous-batching slot count.  Default 1
    /// preserves the legacy single-session path bit-for-bit.  When > 1,
    /// the server constructs an `aether_rt::batched_serving::BatchScheduler`
    /// over a pool-backed `QwenSession` and routes non-streaming chat
    /// requests through it.  Streaming + legacy single-session callers
    /// stay on the original path.
    max_concurrent: usize,
    /// FR-19.5-extra-deep — paged-KV blocks per scheduler slot.  Multiplied
    /// by `max_concurrent` to size the SharedKvPool when `max_concurrent
    /// > 1`.  Default 8 = 32 tokens of KV per concurrent slot (block_size 4);
    /// raise via `--blocks-per-slot` for longer concurrent contexts (memory
    /// scales with `max_concurrent * blocks_per_slot`).  NOTE: the default
    /// single-session path (`max_concurrent == 1`) is NOT bounded by this —
    /// it uses the full `serving::MAX_SEQ` (2048) KV cache.
    blocks_per_slot: i32,
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        port: 8080,
        model: "qwen2.5-7b-instruct".into(),
        gguf: None,
        max_tokens_default: 64,
        stop_token: None,
        warmup: 4,
        tls: false,
        tls_cn: "aether-serve.local".into(),
        paged: false,
        pool_blocks: 0,
        probe: false,
        bge_gguf: None,
        bge_model: "bge-large-en-v1.5".into(),
        bind: "0.0.0.0".into(),
        // 0.8 / 0.9 — sane chat defaults that break greedy loops while
        // still keeping outputs focused.  Override per-request via the
        // OpenAI-standard `temperature` / `top_p` body fields.
        default_temperature: 0.8,
        default_top_p: 0.9,
        default_presence_penalty: 0.3,
        default_frequency_penalty: 0.3,
        tp: 1,
        max_concurrent: 1,
        blocks_per_slot: 8,
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
            "--tls" => { cli.tls = true; if cli.port == 8080 { cli.port = 8443; } }
            "--tls-cn" => cli.tls_cn = it.next().expect("--tls-cn NAME"),
            "--paged" => cli.paged = true,
            "--pool-blocks" => {
                cli.pool_blocks = it.next().expect("--pool-blocks N").parse().expect("pool-blocks int");
                cli.paged = true;
            }
            "--probe" => cli.probe = true,
            "--bge-gguf" => cli.bge_gguf = Some(it.next().expect("--bge-gguf PATH")),
            "--bge-model" => cli.bge_model = it.next().expect("--bge-model NAME"),
            "--bind" => cli.bind = it.next().expect("--bind ADDR"),
            "--temperature" => cli.default_temperature =
                it.next().expect("--temperature F").parse().expect("temperature float"),
            "--top-p" => cli.default_top_p =
                it.next().expect("--top-p F").parse().expect("top_p float"),
            "--presence-penalty" => cli.default_presence_penalty =
                it.next().expect("--presence-penalty F").parse().expect("presence_penalty float"),
            "--frequency-penalty" => cli.default_frequency_penalty =
                it.next().expect("--frequency-penalty F").parse().expect("frequency_penalty float"),
            "--tp" => cli.tp =
                it.next().expect("--tp N").parse().expect("tp int"),
            "--max-concurrent" => cli.max_concurrent =
                it.next().expect("--max-concurrent N").parse().expect("max-concurrent int"),
            "--blocks-per-slot" => cli.blocks_per_slot =
                it.next().expect("--blocks-per-slot N").parse().expect("blocks-per-slot int"),
            "-h" | "--help" => {
                eprintln!("aether-serve [--port N] [--bind ADDR] [--model NAME] [--gguf PATH] [--max-tokens N] [--stop-token ID|none] [--warmup N] [--tls] [--tls-cn NAME] [--paged]");
                eprintln!();
                eprintln!("  Listens on <bind>:port (default bind=0.0.0.0) for OpenAI-compat /v1/chat/completions.");
                eprintln!("  Pass --bind 127.0.0.1 to restrict to loopback only.");
                eprintln!("  --gguf points at any Qwen2.5-7B-Instruct Q4_K_M model file.");
                eprintln!("  --warmup N runs N synthetic decode steps on startup to drive");
                eprintln!("    the GPU into P0/P2 power state and pre-capture the graph.");
                eprintln!("  --tls enables TLS 1.3 (self-signed Ed25519 cert generated on startup");
                eprintln!("        with --tls-cn as the cert CN; default port becomes 8443).");
                eprintln!("  --paged routes K/V through paged_append_kv_devarg +");
                eprintln!("        paged_attention_seq1_devarg (FR-19.4-extra) — identity");
                eprintln!("        page table, bit-identical token output to contiguous mode.");
                eprintln!("  --tp N requests N-way tensor parallelism.  N=1 is the default");
                eprintln!("        single-GPU path.  N>1 runtime-detects NCCL + multi-GPU");
                eprintln!("        and falls back to N=1 with a warning if unavailable.");
                eprintln!("  --max-concurrent N enables FR-19.5-extra-deep continuous batching:");
                eprintln!("        up to N chat requests share one paged-KV session via");
                eprintln!("        BatchScheduler; non-streaming requests route through the");
                eprintln!("        scheduler, streaming + N=1 keep the legacy single-session");
                eprintln!("        path bit-for-bit.  Default 1.  Implies --paged + --pool-blocks.");
                eprintln!("  --blocks-per-slot K sizes the SharedKvPool: total blocks = N*K.");
                eprintln!("        Default 8 (32 tokens of KV per slot).");
                eprintln!("  Without --gguf, returns a stub response (HTTP/JSON plumbing only).");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(2); }
        }
    }
    cli
}

// ============================================================================
// TLS adapter: wraps a TCP stream + TlsServerSession.  Drives the handshake
// at open time, then exposes read_app / write_app over decrypted bytes.
// ============================================================================

unsafe fn rand32() -> [u8; 32] {
    let mut a = [0u8; 32];
    let n = aether_random_bytes(a.as_mut_ptr() as *mut c_void, 32);
    assert_eq!(n, 32, "BCryptGenRandom failed");
    a
}

struct TlsStream {
    fd: i64,
    sess: TlsServerSession,
    app_buf: Vec<u8>,
    eof: bool,
}

impl TlsStream {
    /// Build a TLS session bound to the given socket fd.  Generates fresh
    /// Ed25519 + X25519 keys + server_random + serial.  Does NOT run the
    /// handshake — call `handshake()` next.
    ///
    /// ALPN support: advertises ["h2", "http/1.1"] so HTTP/2 over TLS works
    /// with clients that need ALPN to commit to h2 (RFC 7540 §3.4).  Post-
    /// handshake routing still uses the peek-preface auto-detect, so even
    /// non-ALPN clients work over TLS.
    unsafe fn accept(fd: i64, cn: &str) -> Self {
        let ed_seed = rand32();
        let server_random = rand32();
        let x25519_priv = rand32();
        let mut serial = [0u8; 16];
        let _ = aether_random_bytes(serial.as_mut_ptr() as *mut c_void, 16);
        serial[0] &= 0x7f;
        if serial[0] == 0 { serial[0] = 1; }
        let sess = TlsServerSession::new_with_alpn(
            &ed_seed, &server_random, &x25519_priv, cn, &serial,
            vec![b"h2".to_vec(), b"http/1.1".to_vec()],
        );
        Self { fd, sess, app_buf: Vec::new(), eof: false }
    }

    /// What protocol was negotiated via ALPN (or None if no overlap).
    fn negotiated_alpn(&self) -> Option<&[u8]> { self.sess.negotiated_alpn() }

    /// Drive the TLS handshake by ping-ponging recv/send until session is Connected.
    unsafe fn handshake(&mut self) -> Result<(), &'static str> {
        let mut tmp = vec![0u8; 16 * 1024];
        loop {
            if self.sess.is_handshake_done() { return Ok(()); }
            let n = aether_tcp_recv(self.fd, tmp.as_mut_ptr() as i64, tmp.len() as i64);
            if n <= 0 { return Err("tcp recv during handshake failed"); }
            let plain = self.sess.feed(&tmp[..n as usize]).map_err(|_| "tls feed error")?;
            self.app_buf.extend_from_slice(&plain);
            let out = self.sess.take_outbound();
            if !out.is_empty() {
                let sent = aether_tcp_send(self.fd, out.as_ptr() as i64, out.len() as i64);
                if sent != out.len() as i64 { return Err("tcp send during handshake failed"); }
            }
        }
    }

    /// Read up to `dst.len()` decrypted application-data bytes.  Returns 0
    /// on clean close, byte count otherwise.
    unsafe fn read_app(&mut self, dst: &mut [u8]) -> Result<usize, &'static str> {
        loop {
            if !self.app_buf.is_empty() {
                let n = self.app_buf.len().min(dst.len());
                dst[..n].copy_from_slice(&self.app_buf[..n]);
                self.app_buf.drain(..n);
                return Ok(n);
            }
            if self.eof { return Ok(0); }
            let mut tmp = vec![0u8; 16 * 1024];
            let n = aether_tcp_recv(self.fd, tmp.as_mut_ptr() as i64, tmp.len() as i64);
            if n == 0 { self.eof = true; return Ok(0); }
            if n < 0 { return Err("tcp recv app failed"); }
            let plain = self.sess.feed(&tmp[..n as usize]).map_err(|_| "tls feed app error")?;
            self.app_buf.extend_from_slice(&plain);
            let out = self.sess.take_outbound();
            if !out.is_empty() {
                let _ = aether_tcp_send(self.fd, out.as_ptr() as i64, out.len() as i64);
            }
        }
    }

    /// Encrypt + send application-data bytes.  Records are 16 KiB max in
    /// TLS 1.3; we fragment if needed.
    unsafe fn write_app(&mut self, src: &[u8]) -> Result<usize, &'static str> {
        const CHUNK: usize = 16 * 1024 - 32;
        let mut i = 0;
        while i < src.len() {
            let take = (src.len() - i).min(CHUNK);
            self.sess.send_app_data(&src[i..i + take]).map_err(|_| "tls send_app_data failed")?;
            let out = self.sess.take_outbound();
            let sent = aether_tcp_send(self.fd, out.as_ptr() as i64, out.len() as i64);
            if sent != out.len() as i64 { return Err("tcp send app failed"); }
            i += take;
        }
        Ok(src.len())
    }
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
    /// FR-x-extra-sampling: temperature for the next-token sampler.
    /// `None` or `0.0` → greedy argmax (the legacy behaviour).  `> 0.0`
    /// scales logits before softmax → multinomial sample.
    temperature: Option<f32>,
    /// Top-p (nucleus) sampling cutoff.  `None` or `1.0` → no cutoff.
    top_p: Option<f32>,
    /// OpenAI-compat penalty: subtract this from any previously-seen
    /// token's logit.  `None` or `0.0` → no penalty.
    presence_penalty: Option<f32>,
    /// OpenAI-compat penalty: subtract `frequency_penalty * count[t]`
    /// from each previously-seen token's logit.  `None` or `0.0` → no
    /// penalty.
    frequency_penalty: Option<f32>,
    /// HF / llama.cpp-style top-k cutoff.  `<= 0` disables.
    top_k: Option<i32>,
    /// Deterministic generation seed.  `None` → OS-derived seed.
    seed: Option<u64>,
    /// OpenAI `logit_bias`: `{token_id_string → bias_float}`.
    logit_bias: std::collections::HashMap<usize, f32>,
    /// OpenAI `stop`: list of strings;  generation stops on the first
    /// suffix-match.  Either a JSON string or a JSON array of strings.
    stop_strings: Vec<String>,
    /// OpenAI legacy /v1/completions `prompt` field — a raw string the
    /// model should continue (no chat template applied).  Present only
    /// when the caller hit /v1/completions, not /v1/chat/completions.
    raw_prompt: Option<String>,
    /// FR-x-extra-chat-template: ordered (role, content) pairs from the
    /// request's `messages: [...]` array.  Empty when the client sent
    /// `prompt_ids` directly.  Used by the chat-template apply path.
    messages: Vec<(String, String)>,
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
    let messages = find_messages_pairs(s);
    let temperature = find_key_float(s, "temperature");
    let top_p = find_key_float(s, "top_p");
    let presence_penalty = find_key_float(s, "presence_penalty");
    let frequency_penalty = find_key_float(s, "frequency_penalty");
    let top_k = find_key_int(s, "top_k").map(|v| v as i32);
    let seed = find_key_int(s, "seed").map(|v| v as u64);
    let logit_bias = find_logit_bias(s);
    let stop_strings = find_stop_strings(s);
    // /v1/completions legacy: `prompt` is a raw string.  We also
    // accept an array of strings (concatenated newline-separated)
    // because some clients send `prompt: ["..."]`.
    let raw_prompt = find_key_string(s, "prompt");

    if prompt_ids.is_empty() && text_prompt.is_none() && raw_prompt.is_none() {
        return Err("body has neither prompt_ids nor messages[].content nor prompt");
    }

    Ok(JsonBody { prompt_ids, max_tokens, stream, text_prompt, temperature, top_p,
                   presence_penalty, frequency_penalty, top_k, seed,
                   logit_bias, stop_strings, messages, raw_prompt })
}

// ============================================================================
// FR-17-extra-bert-fwd — /v1/embeddings request shape.
//
// OpenAI's request body is `{"input": "...", "model": "..."}`.  Since BPE
// encoding from raw text isn't wired through aether-serve yet, callers can
// alternatively supply `"input_ids":[101, 2003, ...]` directly.  At least one
// of `input` / `input_ids` must be present.
// ============================================================================

struct EmbeddingsRequest {
    /// Token IDs.  Either passed directly via `"input_ids":[...]` or
    /// produced by WordPiece-tokenising `"input":"..."` at handle time.
    input_ids: Vec<i32>,
    /// Optional token-type IDs (defaults to all zeros).
    token_type_ids: Vec<i32>,
    /// Raw text input (when `input_ids` is empty and the body carried
    /// `"input":"text"`).  Tokenised inside `render_embeddings_json` using
    /// the WordPiece tokenizer loaded alongside the BGE GGUF.
    text: Option<String>,
    /// Echo of the request's `model` field, for the response.
    model: Option<String>,
}

fn parse_embeddings_body(body: &[u8]) -> Result<EmbeddingsRequest, &'static str> {
    let s = std::str::from_utf8(body).map_err(|_| "body not utf-8")?;
    let input_ids = match find_key_array(s, "input_ids") {
        Some(arr) => parse_int_array(arr)
            .map_err(|_| "input_ids must be integers")?
            .into_iter().map(|v| v as i32).collect(),
        None => Vec::new(),
    };
    let text = find_key_string(s, "input");
    if input_ids.is_empty() && text.is_none() {
        return Err("/v1/embeddings requires either \"input\":\"text\" or \"input_ids\":[...]");
    }
    // For input_ids path, accept optional explicit token_type_ids.  For text
    // input, token_type_ids defaults to all zeros after tokenization.
    let token_type_ids = match find_key_array(s, "token_type_ids") {
        Some(arr) => parse_int_array(arr)
            .map_err(|_| "token_type_ids must be integers")?
            .into_iter().map(|v| v as i32).collect(),
        None => Vec::new(),  // filled in by render_embeddings_json
    };
    if !input_ids.is_empty() && !token_type_ids.is_empty()
        && token_type_ids.len() != input_ids.len() {
        return Err("token_type_ids length must match input_ids");
    }
    let model = find_key_string(s, "model");
    Ok(EmbeddingsRequest { input_ids, token_type_ids, text, model })
}

fn find_key_string(s: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\"", key);
    let i = s.find(&pat)?;
    let after_key = &s[i + pat.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let after_q = after_colon.strip_prefix('"')?;
    let end = after_q.find('"')?;
    Some(after_q[..end].to_string())
}

#[cfg(feature = "cuda")]
fn render_embeddings_json(state: &ServerState, req: &EmbeddingsRequest) -> String {
    let model_name = req.model.as_deref().unwrap_or(&state.cli.bge_model);
    let bert_mu = match &state.bert {
        Some(b) => b,
        None => return "{\"error\":\"no BGE model loaded — pass --bge-gguf PATH on aether-serve startup\"}".to_string(),
    };
    // Resolve input_ids: prefer the explicit array; otherwise tokenize `text`
    // via the WordPiece tokenizer built at server startup.
    let input_ids: Vec<i32> = if !req.input_ids.is_empty() {
        req.input_ids.clone()
    } else if let Some(text) = req.text.as_deref() {
        match &state.bert_tokenizer {
            Some(tok) => tok.encode(text),
            None => return "{\"error\":\"BGE tokenizer not initialised\"}".to_string(),
        }
    } else {
        return "{\"error\":\"request had neither input nor input_ids\"}".to_string();
    };
    let token_type_ids: Vec<i32> = if req.token_type_ids.is_empty() {
        vec![0i32; input_ids.len()]
    } else if req.token_type_ids.len() == input_ids.len() {
        req.token_type_ids.clone()
    } else {
        return format!("{{\"error\":\"token_type_ids length {} != input_ids length {}\"}}",
            req.token_type_ids.len(), input_ids.len());
    };
    let mut bert = bert_mu.lock().unwrap();
    if input_ids.len() > bert.max_seq {
        return format!("{{\"error\":\"input length {} exceeds max_pos {}\"}}",
            input_ids.len(), bert.max_seq);
    }
    if input_ids.is_empty() {
        return "{\"error\":\"empty input after tokenization\"}".to_string();
    }
    let t = std::time::Instant::now();
    let emb = bert.embed(&input_ids, &token_type_ids);
    eprintln!("[serve] /v1/embeddings: {} tokens -> {}-dim in {:.3}s",
        input_ids.len(), emb.len(), t.elapsed().as_secs_f32());

    // OpenAI shape: {"object":"list","model":..,"data":[{"object":"embedding",
    // "index":0,"embedding":[...]}],"usage":{"prompt_tokens":N,"total_tokens":N}}
    let mut emb_json = String::with_capacity(emb.len() * 16);
    emb_json.push('[');
    for (i, v) in emb.iter().enumerate() {
        if i > 0 { emb_json.push(','); }
        emb_json.push_str(&format!("{:.7}", v));
    }
    emb_json.push(']');
    format!(
        "{{\"object\":\"list\",\"model\":\"{}\",\"data\":[{{\"object\":\"embedding\",\"index\":0,\"embedding\":{}}}],\"usage\":{{\"prompt_tokens\":{},\"total_tokens\":{}}}}}",
        model_name, emb_json, req.input_ids.len(), req.input_ids.len())
}

#[cfg(not(feature = "cuda"))]
fn render_embeddings_json(_state: &ServerState, _req: &EmbeddingsRequest) -> String {
    "{\"error\":\"aether-serve built without --features cuda\"}".to_string()
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

fn find_key_float(s: &str, key: &str) -> Option<f32> {
    let pat = format!("\"{}\"", key);
    let i = s.find(&pat)?;
    let after_key = &s[i + pat.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    let end = after_colon.find(|c: char|
        !c.is_ascii_digit() && c != '-' && c != '.' && c != 'e' && c != 'E' && c != '+'
    ).unwrap_or(after_colon.len());
    after_colon[..end].parse().ok()
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

/// Parse OpenAI `logit_bias` map.  Field shape: `{"50256": -100, ...}`
/// — JSON object with stringified integer ids → float biases.  We do
/// a substring-driven scan since we don't have a real JSON parser.
fn find_logit_bias(s: &str) -> std::collections::HashMap<usize, f32> {
    let mut out = std::collections::HashMap::new();
    let key = "\"logit_bias\"";
    let Some(i) = s.find(key) else { return out; };
    let after = &s[i + key.len()..];
    let Some(colon) = after.find(':') else { return out; };
    let after_colon = after[colon + 1..].trim_start();
    if !after_colon.starts_with('{') { return out; }
    // Body of the object lives between { and matching }.  Naive scan —
    // good enough since logit_bias values are numbers, no nesting.
    let body_start = (after.as_ptr() as usize) - (s.as_ptr() as usize)
                    + colon + 1 + (after[colon + 1..].len() - after_colon.len()) + 1;
    let mut depth = 1i32;
    let bytes = s.as_bytes();
    let mut end = body_start;
    while end < bytes.len() {
        match bytes[end] {
            b'{' => depth += 1,
            b'}' => { depth -= 1; if depth == 0 { break; } }
            _ => {}
        }
        end += 1;
    }
    let body = &s[body_start..end];
    // Walk pairs:  "id": float, ...
    let mut cursor = 0;
    while cursor < body.len() {
        let Some(q1) = body[cursor..].find('"') else { break; };
        let abs1 = cursor + q1 + 1;
        let Some(q2_rel) = body[abs1..].find('"') else { break; };
        let abs2 = abs1 + q2_rel;
        let id: usize = match body[abs1..abs2].parse() { Ok(v) => v, Err(_) => { cursor = abs2 + 1; continue; } };
        let after_q2 = &body[abs2 + 1..];
        let Some(colon) = after_q2.find(':') else { break; };
        let after_colon = after_q2[colon + 1..].trim_start();
        let end_v = after_colon.find(|c: char| c == ',' || c == '}' || c == '\n')
            .unwrap_or(after_colon.len());
        if let Ok(v) = after_colon[..end_v].trim().parse::<f32>() {
            out.insert(id, v);
        }
        cursor = abs2 + 1 + colon + 1 + (after_q2[colon + 1..].len() - after_colon.len()) + end_v;
    }
    out
}

/// Parse OpenAI `stop` field.  Accepts either a JSON string or a JSON
/// array of strings.  Returns empty Vec when absent / malformed.
fn find_stop_strings(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let key = "\"stop\"";
    let Some(i) = s.find(key) else { return out; };
    let after = &s[i + key.len()..];
    let Some(colon) = after.find(':') else { return out; };
    let after_colon = after[colon + 1..].trim_start();
    if after_colon.starts_with('"') {
        // Single string.  Reuse find_key_string semantics — find the
        // matching closing quote.
        let q1_abs = (after.as_ptr() as usize) - (s.as_ptr() as usize)
                    + colon + 1 + (after[colon + 1..].len() - after_colon.len()) + 1;
        let bytes = s.as_bytes();
        let mut j = q1_abs;
        while j < bytes.len() && !(bytes[j] == b'"' && bytes[j - 1] != b'\\') { j += 1; }
        if j < bytes.len() {
            out.push(unescape_json_string(&s[q1_abs..j]));
        }
    } else if after_colon.starts_with('[') {
        // Array of strings.  Walk pairs of unescaped quotes.
        let arr_start = (after.as_ptr() as usize) - (s.as_ptr() as usize)
                    + colon + 1 + (after[colon + 1..].len() - after_colon.len()) + 1;
        let bytes = s.as_bytes();
        let mut cursor = arr_start;
        let mut depth = 1i32;
        while cursor < bytes.len() && depth > 0 {
            match bytes[cursor] {
                b'[' => depth += 1,
                b']' => { depth -= 1; if depth == 0 { break; } cursor += 1; continue; }
                b'"' => {
                    let q1 = cursor + 1;
                    let mut j = q1;
                    while j < bytes.len() && !(bytes[j] == b'"' && bytes[j - 1] != b'\\') { j += 1; }
                    if j < bytes.len() {
                        out.push(unescape_json_string(&s[q1..j]));
                        cursor = j + 1;
                        continue;
                    } else { break; }
                }
                _ => {}
            }
            cursor += 1;
        }
    }
    out
}

/// Parse `messages: [{role:..., content:...}, ...]` into ordered (role,
/// content) pairs.  Tolerant of whitespace / property ordering — assumes
/// each role appears before its content in the message object.  Falls
/// back gracefully (returns Vec::new()) on malformed input;  the chat-
/// template path also has a fallback to plain-text encode.
fn find_messages_pairs(s: &str) -> Vec<(String, String)> {
    let key = "\"messages\"";
    let Some(i) = s.find(key) else { return Vec::new(); };
    let mut cursor = i + key.len();
    let mut out: Vec<(String, String)> = Vec::new();
    let bytes = s.as_bytes();
    loop {
        let Some(rel) = s[cursor..].find("\"role\"") else { break; };
        let role_at = cursor + rel + "\"role\"".len();
        // Skip to value of role.
        let after_role = &s[role_at..];
        let Some(colon) = after_role.find(':') else { break; };
        let after_colon = after_role[colon + 1..].trim_start();
        if !after_colon.starts_with('"') { cursor = role_at; continue; }
        let q1 = role_at + colon + 1 + (after_role[colon + 1..].len() - after_colon.len()) + 1;
        let mut j = q1;
        while j < bytes.len() && !(bytes[j] == b'"' && bytes[j - 1] != b'\\') { j += 1; }
        if j >= bytes.len() { break; }
        let role = unescape_json_string(&s[q1..j]);
        cursor = j + 1;
        // Find matching content.
        let Some(rel) = s[cursor..].find("\"content\"") else { break; };
        let cont_at = cursor + rel + "\"content\"".len();
        let after_cont = &s[cont_at..];
        let Some(colon) = after_cont.find(':') else { break; };
        let after_colon = after_cont[colon + 1..].trim_start();
        if !after_colon.starts_with('"') { cursor = cont_at; continue; }
        let q1 = cont_at + colon + 1 + (after_cont[colon + 1..].len() - after_colon.len()) + 1;
        let mut j = q1;
        while j < bytes.len() && !(bytes[j] == b'"' && bytes[j - 1] != b'\\') { j += 1; }
        if j >= bytes.len() { break; }
        let content = unescape_json_string(&s[q1..j]);
        cursor = j + 1;
        out.push((role, content));
    }
    out
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
    /// FR-19.4-extra-tenant: shared KV pool, kept alive for the lifetime of
    /// the server when --pool-blocks > 0.  Future-session work routes new
    /// concurrent requests through additional sessions bound to this pool.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pool: Option<std::sync::Arc<SharedKvPool>>,
    /// FR-19.5-extra-deep — continuous-batching scheduler.  When
    /// `cli.max_concurrent > 1`, non-streaming /v1/chat/completions
    /// requests are dispatched through this instead of the per-request
    /// `session` Mutex.  `session` stays None in that mode (the
    /// scheduler owns the QwenSession internally).
    #[cfg(feature = "cuda")]
    scheduler: Option<std::sync::Arc<aether_rt::batched_serving::BatchScheduler>>,
    /// FR-17-extra-bert-fwd: BERT/BGE encoder for /v1/embeddings, loaded
    /// when --bge-gguf is supplied.  Held under a Mutex so concurrent
    /// /v1/embeddings calls serialise.  (BertSession holds no state between
    /// requests beyond loaded weights; the per-request alloc/free of
    /// activations is internal.)
    #[cfg(feature = "cuda")]
    bert: Option<std::sync::Mutex<BertSession>>,
    /// WordPiece tokenizer built from the same bge GGUF as `bert`.  Stateless
    /// + immutable after construction; shared by reference across requests.
    #[cfg(feature = "cuda")]
    bert_tokenizer: Option<WordPieceTokenizer>,
}

impl ServerState {
    fn new(mut cli: Cli) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
            // FR-19.5-extra-deep: --max-concurrent > 1 IMPLIES paged-KV
            // + a SharedKvPool sized for max_concurrent * blocks_per_slot.
            // Default 1 keeps every legacy single-session caller bit-
            // identical (no scheduler, no pool unless --pool-blocks).
            if cli.max_concurrent > 1 {
                if cli.pool_blocks == 0 {
                    cli.pool_blocks = cli.max_concurrent as i32 * cli.blocks_per_slot;
                }
                cli.paged = true;
            }
            let pool = if cli.pool_blocks > 0 {
                eprintln!("[aether-serve] allocating SharedKvPool: {} blocks × 4 tokens = {} token capacity",
                    cli.pool_blocks, cli.pool_blocks * 4);
                Some(SharedKvPool::new(cli.pool_blocks, 4))
            } else { None };

            let session = match &cli.gguf {
                Some(path) => {
                    eprintln!("[aether-serve] loading GGUF: {}{}", path,
                        if cli.paged { " (paged KV mode)" } else { "" });
                    let t = std::time::Instant::now();
                    let mut s = if cli.tp > 1 {
                        // FR-x-extra-tp — tensor-parallel session.  Constructs a
                        // TpSession which runtime-detects NCCL + multi-GPU
                        // availability.  Falls back to TP=1 (single-GPU) with a
                        // warning if unavailable.  The inner QwenSession is the
                        // underlying compute engine in both cases; multi-GPU
                        // execution wires up once the cuda.rs multi-context
                        // refactor lands (see tensor_parallel::TP_GAPS).
                        if pool.is_some() {
                            return Err(
                                "--tp > 1 with --pool-blocks is not yet supported; \
                                 SharedKvPool multi-context refactor is filed alongside \
                                 TP_GAPS::CUDA_MULTI_CONTEXT".into());
                        }
                        let tp = if cli.paged {
                            aether_rt::tensor_parallel::TpSession::new_paged(path, cli.tp)
                        } else {
                            aether_rt::tensor_parallel::TpSession::new(path, cli.tp)
                        }.map_err(|e| format!("TpSession construction failed: {}", e))?;
                        eprintln!("[aether-serve] {}", tp.diag());
                        tp.into_inner()
                    } else if let Some(p) = &pool {
                        QwenSession::new_paged_with_pool(path, p.clone())?
                    } else if cli.paged {
                        QwenSession::new_paged(path)?
                    } else {
                        QwenSession::new(path)?
                    };
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

            // FR-19.5-extra-deep: when max_concurrent > 1 and the session
            // is pool-backed, lift the QwenSession out of the legacy
            // Mutex wrap and hand it to a BatchScheduler.  Streaming
            // requests still need a Mutex<QwenSession> to call directly
            // — Phase-1 fallback is to keep streaming on the legacy
            // single-session path; when the scheduler is active and a
            // streaming request arrives, we return 503 with a clear
            // message.  Non-streaming chat completions route through
            // the scheduler.
            let (session, scheduler) = if cli.max_concurrent > 1 {
                let Some(mu) = session else {
                    return Err(
                        "--max-concurrent N > 1 requires --gguf (no model loaded)".into());
                };
                let inner = mu.into_inner().map_err(|e| format!("session poisoned: {}", e))?;
                if !inner.is_pool_backed() {
                    return Err(format!(
                        "--max-concurrent {} requires a pool-backed session — \
                         did --pool-blocks default get suppressed?  is_pool_backed=false",
                        cli.max_concurrent));
                }
                let sched = aether_rt::batched_serving::BatchScheduler::new(
                    inner, cli.max_concurrent)?;
                eprintln!("[aether-serve] BatchScheduler online: max_concurrent={}", cli.max_concurrent);
                (None, Some(std::sync::Arc::new(sched)))
            } else {
                (session, None)
            };
            let (bert, bert_tokenizer) = match &cli.bge_gguf {
                Some(path) => {
                    eprintln!("[aether-serve] loading BGE GGUF: {}", path);
                    let t = std::time::Instant::now();
                    let s = BertSession::from_gguf(path)?;
                    eprintln!("[aether-serve] BGE loaded in {:.2}s (d_model={} layers={})",
                        t.elapsed().as_secs_f32(), s.cfg.d_model, s.cfg.n_layers);
                    // Build the WordPiece tokenizer from the same GGUF so
                    // /v1/embeddings can accept raw text input.
                    let tok = unsafe {
                        let h = aether_gguf_open(
                            path.as_ptr() as i64, path.len() as std::os::raw::c_int);
                        if h < 0 {
                            return Err(format!("tokenizer GGUF reopen failed: {}", h));
                        }
                        let t = WordPieceTokenizer::from_gguf(h)?;
                        aether_gguf_close(h);
                        t
                    };
                    eprintln!("[aether-serve] BGE tokenizer: vocab loaded (CLS={} SEP={} UNK={})",
                        tok.cls_id, tok.sep_id, tok.unk_id);
                    (Some(std::sync::Mutex::new(s)), Some(tok))
                }
                None => (None, None),
            };
            return Ok(ServerState { cli, session, pool, scheduler, bert, bert_tokenizer });
        }
        #[cfg(not(feature = "cuda"))]
        {
            if cli.gguf.is_some() || cli.bge_gguf.is_some() {
                return Err("--gguf / --bge-gguf require building with --features cuda".into());
            }
            Ok(ServerState { cli })
        }
    }
}

// ----------------------------------------------------------------------------
// Transport trait — uniform read/write over plain TCP or TLS.
// ----------------------------------------------------------------------------

trait Transport {
    unsafe fn read(&mut self, dst: &mut [u8]) -> Result<usize, &'static str>;
    unsafe fn write(&mut self, src: &[u8]) -> Result<usize, &'static str>;
}

struct PlainTcp { pub fd: i64 }

impl Transport for PlainTcp {
    unsafe fn read(&mut self, dst: &mut [u8]) -> Result<usize, &'static str> {
        let n = aether_tcp_recv(self.fd, dst.as_mut_ptr() as i64, dst.len() as i64);
        if n < 0 { Err("tcp recv failed") } else { Ok(n as usize) }
    }
    unsafe fn write(&mut self, src: &[u8]) -> Result<usize, &'static str> {
        let n = aether_tcp_send(self.fd, src.as_ptr() as i64, src.len() as i64);
        if n != src.len() as i64 { Err("tcp send short") } else { Ok(n as usize) }
    }
}

impl Transport for TlsStream {
    unsafe fn read(&mut self, dst: &mut [u8]) -> Result<usize, &'static str> {
        TlsStream::read_app(self, dst)
    }
    unsafe fn write(&mut self, src: &[u8]) -> Result<usize, &'static str> {
        TlsStream::write_app(self, src)
    }
}

/// Read bytes until we have full HTTP/1.1 headers + the declared Content-Length
/// body, returning the assembled buffer (headers + body) or an error.
/// `prefix` is bytes already peeked off the wire — prepended into the buffer
/// so the parser sees them.
unsafe fn read_full_http_request_with_prefix(
    t: &mut dyn Transport, max: usize, prefix: &[u8],
) -> Result<Vec<u8>, &'static str> {
    let mut buf: Vec<u8> = Vec::with_capacity(8192);
    buf.extend_from_slice(prefix);
    let mut tmp = vec![0u8; 8192];
    let mut header_end = find_crlf_crlf(&buf).map(|p| p + 4);
    while header_end.is_none() {
        if buf.len() >= max { return Err("request too large"); }
        let n = t.read(&mut tmp)?;
        if n == 0 { return Err("eof before headers"); }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = find_crlf_crlf(&buf) { header_end = Some(p + 4); }
    }
    let body_start = header_end.unwrap();
    let head = &buf[..body_start];
    let content_length = parse_content_length(head).unwrap_or(0);
    let need = body_start + content_length;
    while buf.len() < need {
        if buf.len() >= max { return Err("request body too large"); }
        let n = t.read(&mut tmp)?;
        if n == 0 { return Err("eof in body"); }
        buf.extend_from_slice(&tmp[..n]);
    }
    buf.truncate(need);
    Ok(buf)
}

/// HTTP/2 (h2c, prior-knowledge mode) server.  Caller has already consumed
/// the 24-byte client preface from the transport.  We send our SETTINGS frame
/// + ACK the client's SETTINGS, then loop: read HEADERS frame, optionally
/// DATA frames, dispatch through the existing JSON parser, and emit
/// HEADERS+DATA response frames.  Single concurrent stream per connection
/// supported; multi-stream HEADERS-interleave handled.
unsafe fn handle_request_h2(state: &ServerState, t: &mut dyn Transport) {
    // 1) Send our SETTINGS frame.
    let mut out = Vec::new();
    http2::build_settings(&mut out, &[
        (http2::SETTINGS_HEADER_TABLE_SIZE, 0), // we don't index dynamically
        (http2::SETTINGS_MAX_CONCURRENT_STREAMS, 16),
        (http2::SETTINGS_INITIAL_WINDOW_SIZE, 65535),
        (http2::SETTINGS_MAX_FRAME_SIZE, 16384),
    ]);
    if t.write(&out).is_err() { return; }

    // 2) Frame loop.
    let mut in_buf: Vec<u8> = Vec::with_capacity(16384);
    let mut tmp = vec![0u8; 16384];
    // Per-stream state: hpack-decoded headers + accumulated DATA body.
    use std::collections::HashMap;
    struct StreamState {
        headers: Vec<(Vec<u8>, Vec<u8>)>,
        body: Vec<u8>,
        end_stream: bool,
        end_headers: bool,
    }
    let mut streams: HashMap<u32, StreamState> = HashMap::new();

    loop {
        // Try to parse a complete frame from in_buf.
        while let Some((frame, consumed)) = http2::Frame::parse(&in_buf) {
            in_buf.drain(..consumed);
            let ft = frame.frame_type;
            if ft == http2::FRAME_SETTINGS {
                if (frame.flags & http2::FLAG_ACK) == 0 {
                    let mut ack = Vec::new();
                    http2::build_settings_ack(&mut ack);
                    let _ = t.write(&ack);
                }
            } else if ft == http2::FRAME_PING {
                if (frame.flags & http2::FLAG_ACK) == 0 && frame.payload.len() == 8 {
                    let mut ping = Vec::new();
                    let mut payload = [0u8; 8];
                    payload.copy_from_slice(&frame.payload);
                    http2::build_ping_ack(&mut ping, &payload);
                    let _ = t.write(&ping);
                }
            } else if ft == http2::FRAME_WINDOW_UPDATE || ft == http2::FRAME_PRIORITY {
                // Accept and ignore.
            } else if ft == http2::FRAME_HEADERS {
                let sid = frame.stream_id;
                let end_stream = (frame.flags & http2::FLAG_END_STREAM) != 0;
                let end_headers = (frame.flags & http2::FLAG_END_HEADERS) != 0;
                let mut hpack_payload = frame.payload.clone();
                if (frame.flags & http2::FLAG_PRIORITY) != 0 {
                    if hpack_payload.len() < 5 { send_goaway(t, sid, 1); return; }
                    hpack_payload.drain(..5);
                }
                if (frame.flags & http2::FLAG_PADDED) != 0 {
                    if hpack_payload.is_empty() { send_goaway(t, sid, 1); return; }
                    let pad_len = hpack_payload[0] as usize;
                    hpack_payload.drain(..1);
                    if pad_len > hpack_payload.len() { send_goaway(t, sid, 1); return; }
                    hpack_payload.truncate(hpack_payload.len() - pad_len);
                }
                let headers = match http2::hpack_decode_headers(&hpack_payload) {
                    Some(h) => h,
                    None => { send_goaway(t, sid, 9); return; }
                };
                let entry = streams.entry(sid).or_insert(StreamState {
                    headers: Vec::new(), body: Vec::new(),
                    end_stream: false, end_headers: false,
                });
                entry.headers = headers;
                entry.end_stream = end_stream;
                entry.end_headers = end_headers;
                if end_headers && end_stream {
                    if let Some(s) = streams.remove(&sid) {
                        dispatch_h2_stream(state, t, sid, s.headers, s.body);
                    }
                }
            } else if ft == http2::FRAME_DATA {
                let sid = frame.stream_id;
                let mut payload = frame.payload.clone();
                if (frame.flags & http2::FLAG_PADDED) != 0 {
                    if payload.is_empty() { send_goaway(t, sid, 1); return; }
                    let pad_len = payload[0] as usize;
                    payload.drain(..1);
                    if pad_len > payload.len() { send_goaway(t, sid, 1); return; }
                    payload.truncate(payload.len() - pad_len);
                }
                if let Some(entry) = streams.get_mut(&sid) {
                    entry.body.extend_from_slice(&payload);
                    if (frame.flags & http2::FLAG_END_STREAM) != 0 {
                        entry.end_stream = true;
                        if entry.end_headers {
                            let s = streams.remove(&sid).unwrap();
                            dispatch_h2_stream(state, t, sid, s.headers, s.body);
                        }
                    }
                }
            } else if ft == http2::FRAME_RST_STREAM {
                streams.remove(&frame.stream_id);
            } else if ft == http2::FRAME_GOAWAY {
                return;
            }
            // Unknown frame types: ignored per RFC.
        }
        // Need more bytes.
        let n = match t.read(&mut tmp) {
            Ok(n) => n,
            Err(_) => return,
        };
        if n == 0 { return; }
        in_buf.extend_from_slice(&tmp[..n]);
    }
}

unsafe fn send_goaway(t: &mut dyn Transport, last_stream_id: u32, error_code: u32) {
    let mut buf = Vec::new();
    http2::build_goaway(&mut buf, last_stream_id, error_code);
    let _ = t.write(&buf);
}

/// Dispatch one h2 stream — headers + body are owned (moved out of the
/// per-stream state in handle_request_h2).
unsafe fn dispatch_h2_stream(
    state: &ServerState, t: &mut dyn Transport,
    stream_id: u32,
    headers: Vec<(Vec<u8>, Vec<u8>)>,
    body: Vec<u8>,
) {
    let req = http2::build_h2_request(stream_id, headers);
    eprintln!("[serve] h2 {} {} body_len={}",
        std::str::from_utf8(&req.method).unwrap_or("?"),
        std::str::from_utf8(&req.path).unwrap_or("?"),
        body.len());
    let path_str = std::str::from_utf8(&req.path).unwrap_or("").to_string();
    let method_str = std::str::from_utf8(&req.method).unwrap_or("").to_string();
    let is_chat_path = path_str == "/v1/chat/completions";
    match (method_str.as_str(), path_str.as_str()) {
        ("GET", "/health") => write_h2_text(t, stream_id, 200, b"ok"),
        ("GET", "/v1/models") => {
            let body = format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"aether\"}}]}}",
                state.cli.model);
            write_h2_json(t, stream_id, 200, body.as_bytes());
        }
        ("GET", "/props") => {
            let body = render_props_json(state);
            write_h2_json(t, stream_id, 200, body.as_bytes());
        }
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            let resp = match parse_body(&body, state.cli.max_tokens_default) {
                Ok(mut req) => {
                    // FR-x-extra-text-encode: same wiring as the HTTP/1.1
                    // handle_completion_t path.  Map text → token ids if
                    // the client didn't supply prompt_ids directly.
                    if req.prompt_ids.is_empty() {
                        #[cfg(feature = "cuda")]
                        {
                            if let Some(sched) = &state.scheduler {
                                let (encode_input, used_template): (Option<String>, bool) =
                                    if is_chat_path && !req.messages.is_empty() {
                                        match sched.apply_chat_template(&req.messages) {
                                            Some(s) => (Some(s), true),
                                            None => (req.text_prompt.clone(), false),
                                        }
                                    } else if is_chat_path {
                                        (req.text_prompt.clone(), false)
                                    } else {
                                        (req.raw_prompt.clone().or_else(|| req.text_prompt.clone()), false)
                                    };
                                if let Some(text) = encode_input {
                                    req.prompt_ids = if used_template {
                                        sched.encode_text_with_specials(&text)
                                    } else {
                                        sched.encode_text(&text)
                                    };
                                }
                            } else if let Some(sess_mu) = &state.session {
                                let sess = sess_mu.lock().unwrap();
                                let (encode_input, used_template): (Option<String>, bool) = if is_chat_path && !req.messages.is_empty() {
                                    match sess.apply_chat_template(&req.messages) {
                                        Some(s) => (Some(s), true),
                                        None => (req.text_prompt.clone(), false),
                                    }
                                } else if is_chat_path {
                                    (req.text_prompt.clone(), false)
                                } else {
                                    (req.raw_prompt.clone().or_else(|| req.text_prompt.clone()), false)
                                };
                                if let Some(text) = encode_input {
                                    req.prompt_ids = if used_template {
                                        sess.encode_text_with_specials(&text)
                                    } else {
                                        sess.encode_text(&text)
                                    };
                                }
                            }
                        }
                    }
                    if req.prompt_ids.is_empty() {
                        format!("{{\"error\":\"text encode failed or no model loaded\"}}")
                    } else {
                        // FR-x-extra: vocab guard — see handle_completion_t for rationale.
                        #[cfg(feature = "cuda")]
                        let oob = if let Some(sched) = &state.scheduler {
                            let v = sched.vocab();
                            req.prompt_ids.iter().find(|&&i| i >= v).copied().map(|i| (i, v))
                        } else if let Some(sess_mu) = &state.session {
                            let v = sess_mu.lock().unwrap().cfg.vocab;
                            req.prompt_ids.iter().find(|&&i| i >= v).copied().map(|i| (i, v))
                        } else {
                            None
                        };
                        #[cfg(not(feature = "cuda"))]
                        let oob: Option<(usize, usize)> = None;
                        if let Some((bad, vocab)) = oob {
                            format!("{{\"error\":\"prompt_ids contains id {} out of vocab {}\"}}", bad, vocab)
                        } else {
                            render_completion_json(state, &req)
                        }
                    }
                }
                Err(e) => format!("{{\"error\":\"{}\"}}", e),
            };
            write_h2_json(t, stream_id, 200, resp.as_bytes());
        }
        ("POST", "/v1/embeddings") => {
            let resp = match parse_embeddings_body(&body) {
                Ok(req) => render_embeddings_json(state, &req),
                Err(e) => format!("{{\"error\":\"{}\"}}", e),
            };
            write_h2_json(t, stream_id, 200, resp.as_bytes());
        }
        _ => write_h2_text(t, stream_id, 404, b"not found"),
    }
}

unsafe fn write_h2_text(t: &mut dyn Transport, stream_id: u32, code: u16, msg: &[u8]) {
    let resp = http2::H2Response {
        status: code, body: msg.to_vec(),
        content_type: b"text/plain".to_vec(),
    };
    let mut out = Vec::new();
    http2::write_h2_response(&mut out, stream_id, &resp);
    let _ = t.write(&out);
}
unsafe fn write_h2_json(t: &mut dyn Transport, stream_id: u32, code: u16, body: &[u8]) {
    let resp = http2::H2Response {
        status: code, body: body.to_vec(),
        content_type: b"application/json".to_vec(),
    };
    let mut out = Vec::new();
    http2::write_h2_response(&mut out, stream_id, &resp);
    let _ = t.write(&out);
}

/// Render the OpenAI completion JSON for an H2 request (shares the cuda /
/// stub branching with the HTTP/1.1 path).
unsafe fn render_completion_json(state: &ServerState, req: &JsonBody) -> String {
    let generated_text: String;
    let prompt_tokens = req.prompt_ids.len() as c_int;
    let completion_tokens: c_int;
    #[cfg(feature = "cuda")]
    {
        if let Some(sched) = &state.scheduler {
            // FR-19.5-extra-deep — scheduler-routed path for H2 clients.
            let stop = state.cli.stop_token.or_else(|| {
                let eos = sched.eos_token();
                if eos >= 0 { Some(eos as usize) } else { None }
            });
            let params = aether_rt::serving::SamplingParams {
                temperature: req.temperature.unwrap_or(state.cli.default_temperature),
                top_p: req.top_p.unwrap_or(state.cli.default_top_p),
                top_k: req.top_k.unwrap_or(0),
                presence_penalty: req.presence_penalty.unwrap_or(state.cli.default_presence_penalty),
                frequency_penalty: req.frequency_penalty.unwrap_or(state.cli.default_frequency_penalty),
                seed: req.seed,
                logit_bias: req.logit_bias.clone(),
            };
            let ids = match sched.generate_blocking(
                req.prompt_ids.clone(), req.max_tokens, stop,
                params, req.stop_strings.clone(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    return format!("{{\"error\":\"scheduler error: {}\"}}", e);
                }
            };
            completion_tokens = ids.len() as c_int;
            let text = sched.decode_ids(&ids);
            generated_text = if text.is_empty() { format_id_list(&ids) } else { text };
        } else {
            match &state.session {
                Some(sess_mu) => {
                    let mut sess = sess_mu.lock().unwrap();
                    let stop = state.cli.stop_token.or_else(|| {
                        if sess.eos_token >= 0 { Some(sess.eos_token as usize) } else { None }
                    });
                    let params = aether_rt::serving::SamplingParams {
                        temperature: req.temperature.unwrap_or(state.cli.default_temperature),
                        top_p: req.top_p.unwrap_or(state.cli.default_top_p),
                        top_k: req.top_k.unwrap_or(0),
                        presence_penalty: req.presence_penalty.unwrap_or(state.cli.default_presence_penalty),
                        frequency_penalty: req.frequency_penalty.unwrap_or(state.cli.default_frequency_penalty),
                        seed: req.seed,
                        logit_bias: req.logit_bias.clone(),
                    };
                    let ids = sess.generate_sampled_v2(
                        &req.prompt_ids, req.max_tokens, stop, &params, &req.stop_strings);
                    completion_tokens = ids.len() as c_int;
                    let text = sess.decode_ids(&ids);
                    generated_text = if text.is_empty() { format_id_list(&ids) } else { text };
                }
                None => {
                    generated_text = "[aether-serve stub: --gguf not supplied]".into();
                    completion_tokens = 0;
                }
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        generated_text = "[aether-serve stub: built without --features cuda]".into();
        completion_tokens = 0;
        let _ = state;
    }
    let resp_id = b"chatcmpl-aether-serve-1";
    let escaped = json_escape(&generated_text);
    let mut json_buf = vec![0u8; 65536];
    let n = aether_openai_render_completion(
        resp_id.as_ptr() as *const c_void, resp_id.len() as c_int,
        state.cli.model.as_ptr() as *const c_void, state.cli.model.len() as c_int,
        escaped.as_ptr() as *const c_void, escaped.len() as c_int,
        prompt_tokens, completion_tokens,
        json_buf.as_mut_ptr() as *mut c_void, json_buf.len() as c_int,
    );
    if n <= 0 { return "{}".into(); }
    String::from_utf8_lossy(&json_buf[..n as usize]).into_owned()
}


fn find_crlf_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_content_length(head: &[u8]) -> Option<usize> {
    let s = std::str::from_utf8(head).ok()?;
    for line in s.split("\r\n") {
        let mut parts = line.splitn(2, ':');
        let k = parts.next()?.trim();
        if !k.eq_ignore_ascii_case("content-length") { continue; }
        let v = parts.next()?.trim();
        return v.parse().ok();
    }
    None
}

/// Auto-detect HTTP/2 (prior-knowledge h2c) vs HTTP/1.1 by peeking at the
/// first 24 bytes of the inbound stream.  HTTP/2 connections begin with the
/// 24-byte preface "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n" (RFC 7540 §3.5).
unsafe fn handle_request(state: &ServerState, t: &mut dyn Transport) {
    let mut peek = vec![0u8; 24];
    let mut got = 0;
    while got < 24 {
        let n = match t.read(&mut peek[got..]) {
            Ok(n) => n,
            Err(e) => { eprintln!("[serve] preface read: {}", e); return; }
        };
        if n == 0 { break; }
        got += n;
    }
    if got >= http2::CONNECTION_PREFACE.len() &&
       &peek[..http2::CONNECTION_PREFACE.len()] == http2::CONNECTION_PREFACE
    {
        eprintln!("[serve] h2c connection accepted");
        handle_request_h2(state, t);
        return;
    }
    // Otherwise treat as HTTP/1.1 — prepend the peeked bytes back into the
    // request buffer.
    let req_bytes = match read_full_http_request_with_prefix(t, 1 << 20, &peek[..got]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[serve] {}", e);
            let _ = send_text_t(t, 400, e);
            return;
        }
    };

    // Parse method + path.
    let mut strings = vec![0u8; 512];
    let mut m_len: c_int = 0;
    let mut p_len: c_int = 0;
    let body_off = aether_http_parse_request(
        req_bytes.as_ptr() as *const c_void, req_bytes.len() as c_int,
        strings.as_mut_ptr() as *mut c_void, strings.len() as c_int,
        &mut m_len, &mut p_len,
    );
    if body_off <= 0 {
        let _ = send_text_t(t, 400, "bad request");
        return;
    }
    let method = std::str::from_utf8(&strings[..m_len as usize]).unwrap_or("").to_string();
    let path = std::str::from_utf8(&strings[m_len as usize..(m_len + p_len) as usize]).unwrap_or("").to_string();
    let body = &req_bytes[body_off as usize..];
    eprintln!("[serve] {} {} body_len={}", method, path, body.len());

    match (method.as_str(), path.as_str()) {
        ("GET", "/health") => { let _ = send_text_t(t, 200, "ok"); }
        ("GET", "/v1/models") => handle_list_models_t(state, t),
        ("GET", "/props") => handle_props_t(state, t),
        ("POST", "/v1/chat/completions") => {
            handle_completion_t(state, t, body, /*is_chat=*/true);
        }
        ("POST", "/v1/completions") => {
            handle_completion_t(state, t, body, /*is_chat=*/false);
        }
        ("POST", "/v1/embeddings") => handle_embeddings_t(state, t, body),
        _ => { let _ = send_text_t(t, 404, "not found"); }
    }
}

/// FR-17-extra-bert-fwd — HTTP/1.1 handler for /v1/embeddings.
unsafe fn handle_embeddings_t(state: &ServerState, t: &mut dyn Transport, body: &[u8]) {
    let req = match parse_embeddings_body(body) {
        Ok(r) => r,
        Err(e) => { let _ = send_text_t(t, 400, e); return; }
    };
    let json = render_embeddings_json(state, &req);
    let _ = send_json_t(t, 200, &json);
}

unsafe fn handle_list_models_t(state: &ServerState, t: &mut dyn Transport) {
    let body = format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"aether\"}}]}}",
        state.cli.model);
    let _ = send_json_t(t, 200, &body);
}

/// llama-server compatibility — /props returns server defaults and
/// model identity so load balancers / probes can discriminate
/// deployments without a chat round-trip.
unsafe fn handle_props_t(state: &ServerState, t: &mut dyn Transport) {
    let body = render_props_json(state);
    let _ = send_json_t(t, 200, &body);
}

fn render_props_json(state: &ServerState) -> String {
    let model_name = &state.cli.model;
    let port = state.cli.port;
    let max_default = state.cli.max_tokens_default;
    let t_def  = state.cli.default_temperature;
    let p_def  = state.cli.default_top_p;
    let pp_def = state.cli.default_presence_penalty;
    let fp_def = state.cli.default_frequency_penalty;
    format!(
        "{{\
\"runtime\":\"aether\",\
\"model\":\"{}\",\
\"port\":{},\
\"default_generation_settings\":{{\
\"temperature\":{},\"top_p\":{},\"top_k\":0,\
\"presence_penalty\":{},\"frequency_penalty\":{},\
\"max_tokens\":{}\
}},\
\"chat_template\":\"per-arch built-in\",\
\"supports\":[\"chat_completions\",\"completions\",\"embeddings\",\"streaming\",\"top_k\",\"logit_bias\",\"seed\",\"stop_strings\"]\
}}",
        model_name, port, t_def, p_def, pp_def, fp_def, max_default)
}

unsafe fn handle_completion_t(state: &ServerState, t: &mut dyn Transport, body: &[u8], is_chat: bool) {
    let mut req = match parse_body(body, state.cli.max_tokens_default) {
        Ok(r) => r,
        Err(e) => { let _ = send_text_t(t, 400, e); return; }
    };

    // /v1/chat/completions:  prefer chat-template apply over messages.
    // /v1/completions:  raw prompt continuation, NO chat template (the
    //   legacy OpenAI endpoint expects the server to model-continue the
    //   given text verbatim).
    if req.prompt_ids.is_empty() {
        #[cfg(feature = "cuda")]
        {
            // FR-19.5-extra-deep: scheduler mode delegates encoding to the
            // scheduler's shared session; legacy mode keeps the
            // Mutex<QwenSession> path.
            let (encode_input, used_template, have_session): (Option<String>, bool, bool) =
                if let Some(sched) = &state.scheduler {
                    if is_chat && !req.messages.is_empty() {
                        match sched.apply_chat_template(&req.messages) {
                            Some(rendered) => (Some(rendered), true, true),
                            None => (req.text_prompt.clone(), false, true),
                        }
                    } else if is_chat {
                        (req.text_prompt.clone(), false, true)
                    } else {
                        (req.raw_prompt.clone().or_else(|| req.text_prompt.clone()), false, true)
                    }
                } else if let Some(sess_mu) = &state.session {
                    let sess = sess_mu.lock().unwrap();
                    let pair = if is_chat && !req.messages.is_empty() {
                        match sess.apply_chat_template(&req.messages) {
                            Some(rendered) => (Some(rendered), true),
                            None => (req.text_prompt.clone(), false),
                        }
                    } else if is_chat {
                        (req.text_prompt.clone(), false)
                    } else {
                        (req.raw_prompt.clone().or_else(|| req.text_prompt.clone()), false)
                    };
                    (pair.0, pair.1, true)
                } else {
                    (None, false, false)
                };
            if !have_session {
                let _ = send_text_t(t, 503,
                    "no model loaded (start with --gguf) — text encode unavailable");
                return;
            }
            let Some(text) = encode_input else {
                let _ = send_text_t(t, 400, "body has no usable prompt");
                return;
            };
            let ids = if let Some(sched) = &state.scheduler {
                if used_template {
                    sched.encode_text_with_specials(&text)
                } else {
                    sched.encode_text(&text)
                }
            } else if let Some(sess_mu) = &state.session {
                let sess = sess_mu.lock().unwrap();
                if used_template {
                    sess.encode_text_with_specials(&text)
                } else {
                    sess.encode_text(&text)
                }
            } else {
                Vec::new()
            };
            if ids.is_empty() {
                let _ = send_text_t(t, 500,
                    "text encode failed (no tokenizer loaded? or vocab gap) — pass prompt_ids");
                return;
            }
            req.prompt_ids = ids;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = is_chat;
            let _ = send_text_t(t, 501,
                "text encode requires --features cuda build");
            return;
        }
    }

    // FR-x-extra: validate prompt_ids against the model's vocab BEFORE
    // dispatching into generate().  Otherwise an out-of-vocab id from a
    // stale or cross-model client (e.g. GLM BOS 154822 sent to a
    // Qwen2.5 server with vocab=152064) panics deep in
    // dequant_embd_row and takes the server down for ~10s while systemd
    // restarts it.  Return 400 instead.
    #[cfg(feature = "cuda")]
    if !req.prompt_ids.is_empty() {
        let vocab_opt: Option<usize> = if let Some(sched) = &state.scheduler {
            Some(sched.vocab())
        } else if let Some(sess_mu) = &state.session {
            Some(sess_mu.lock().unwrap().cfg.vocab)
        } else {
            None
        };
        if let Some(vocab) = vocab_opt {
            if let Some(&bad) = req.prompt_ids.iter().find(|&&i| i >= vocab) {
                let msg = format!(
                    "prompt_ids contains token id {} which is out of vocab (vocab_size={}); pass ids in [0, {})",
                    bad, vocab, vocab);
                let _ = send_text_t(t, 400, &msg);
                return;
            }
        }
    }

    if req.stream {
        // FR-19.5-extra-deep Phase 2: streaming over the BatchScheduler
        // now drains a per-token `StreamEvent` channel from the worker
        // thread.  Without a scheduler, the legacy single-session SSE
        // chunker handles it.
        #[cfg(feature = "cuda")]
        if state.scheduler.is_some() {
            handle_completion_streaming_scheduler_t(state, t, &req);
            return;
        }
        handle_completion_streaming_t(state, t, &req);
        return;
    }

    let generated_text: String;
    let prompt_tokens = req.prompt_ids.len() as c_int;
    let completion_tokens: c_int;

    #[cfg(feature = "cuda")]
    {
        // FR-19.5-extra-deep: scheduler mode routes generation through
        // BatchScheduler::generate_blocking; legacy mode keeps the
        // direct Mutex<QwenSession>::generate_sampled_v2 path.
        if let Some(sched) = &state.scheduler {
            let stop = state.cli.stop_token.or_else(|| {
                let eos = sched.eos_token();
                if eos >= 0 { Some(eos as usize) } else { None }
            });
            let params = aether_rt::serving::SamplingParams {
                temperature: req.temperature.unwrap_or(state.cli.default_temperature),
                top_p: req.top_p.unwrap_or(state.cli.default_top_p),
                top_k: req.top_k.unwrap_or(0),
                presence_penalty: req.presence_penalty.unwrap_or(state.cli.default_presence_penalty),
                frequency_penalty: req.frequency_penalty.unwrap_or(state.cli.default_frequency_penalty),
                seed: req.seed,
                logit_bias: req.logit_bias.clone(),
            };
            let t0 = std::time::Instant::now();
            let ids = match sched.generate_blocking(
                req.prompt_ids.clone(), req.max_tokens, stop, params, req.stop_strings.clone(),
            ) {
                Ok(v) => v,
                Err(e) => {
                    let _ = send_text_t(t, 500, &format!("scheduler error: {}", e));
                    return;
                }
            };
            let secs = t0.elapsed().as_secs_f32();
            eprintln!("[serve/sched] {} tokens in {:.3}s = {:.2} tok/s",
                ids.len(), secs, ids.len() as f32 / secs.max(1e-6));
            completion_tokens = ids.len() as c_int;
            let text = sched.decode_ids(&ids);
            generated_text = if text.is_empty() { format_id_list(&ids) } else { text };
        } else {
            match &state.session {
                Some(sess_mu) => {
                    let mut sess = sess_mu.lock().unwrap();
                    let stop = state.cli.stop_token.or_else(|| {
                        if sess.eos_token >= 0 { Some(sess.eos_token as usize) } else { None }
                    });
                    let t = std::time::Instant::now();
                    let params = aether_rt::serving::SamplingParams {
                        temperature: req.temperature.unwrap_or(state.cli.default_temperature),
                        top_p: req.top_p.unwrap_or(state.cli.default_top_p),
                        top_k: req.top_k.unwrap_or(0),
                        presence_penalty: req.presence_penalty.unwrap_or(state.cli.default_presence_penalty),
                        frequency_penalty: req.frequency_penalty.unwrap_or(state.cli.default_frequency_penalty),
                        seed: req.seed,
                        logit_bias: req.logit_bias.clone(),
                    };
                    let ids = sess.generate_sampled_v2(
                        &req.prompt_ids, req.max_tokens, stop, &params, &req.stop_strings);
                    let secs = t.elapsed().as_secs_f32();
                    eprintln!("[serve] generated {} tokens in {:.3}s = {:.2} tok/s (T={:.2} top_p={:.2} top_k={} pp={:.2} fp={:.2})",
                        ids.len(), secs, ids.len() as f32 / secs.max(1e-6),
                        params.temperature, params.top_p, params.top_k, params.presence_penalty, params.frequency_penalty);
                    completion_tokens = ids.len() as c_int;
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
        let _ = send_text_t(t, 500, "render failed");
        return;
    }

    let mut http_buf = vec![0u8; 131072];
    let n_http = aether_http_write_response_200(
        json_buf.as_ptr() as *const c_void, n_json,
        http_buf.as_mut_ptr() as *mut c_void, http_buf.len() as c_int,
    );
    if n_http <= 0 {
        let _ = send_text_t(t, 500, "http write failed");
        return;
    }

    let _ = t.write(&http_buf[..n_http as usize]);
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
/// per generated token via HTTP/1.1 chunked transfer encoding.
unsafe fn handle_completion_streaming_t(state: &ServerState, t: &mut dyn Transport, req: &JsonBody) {
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";
    let _ = t.write(headers.as_bytes());

    let mut send_chunk = |t: &mut dyn Transport, s: &str| {
        let hex_len = format!("{:x}\r\n", s.len());
        let _ = t.write(hex_len.as_bytes());
        let _ = t.write(s.as_bytes());
        let _ = t.write(b"\r\n");
    };

    #[cfg(feature = "cuda")]
    {
        if let Some(sess_mu) = &state.session {
            let mut sess = sess_mu.lock().unwrap();
            let stop = state.cli.stop_token.or_else(|| {
                if sess.eos_token >= 0 { Some(sess.eos_token as usize) } else { None }
            });
            let params = aether_rt::serving::SamplingParams {
                temperature: req.temperature.unwrap_or(state.cli.default_temperature),
                top_p: req.top_p.unwrap_or(state.cli.default_top_p),
                top_k: req.top_k.unwrap_or(0),
                presence_penalty: req.presence_penalty.unwrap_or(state.cli.default_presence_penalty),
                frequency_penalty: req.frequency_penalty.unwrap_or(state.cli.default_frequency_penalty),
                seed: req.seed,
                logit_bias: req.logit_bias.clone(),
            };
            let stop_strings = req.stop_strings.clone();
            // Stream the sampled generation token-by-token.  Mirror
            // generate_sampled_v2 semantics exactly (penalties, top_k,
            // top_p, stop strings).
            sess.reset();
            sess.prefill(&req.prompt_ids);
            let mut last = *req.prompt_ids.last().unwrap();
            let mut rng = params.seed.unwrap_or_else(aether_rt::serving::seed_rng_external);
            if rng == 0 { rng = aether_rt::serving::seed_rng_external(); }
            let mut seen: std::collections::HashMap<usize, u32> =
                std::collections::HashMap::new();
            let mut running = String::new();
            for _ in 0..req.max_tokens {
                let mut logits = sess.step_logits(last);
                if !params.logit_bias.is_empty() {
                    aether_rt::serving::apply_logit_bias(&mut logits, &params.logit_bias);
                }
                if params.presence_penalty != 0.0 || params.frequency_penalty != 0.0 {
                    aether_rt::serving::apply_repetition_penalty(
                        &mut logits, &seen, params.presence_penalty, params.frequency_penalty);
                }
                let id = if params.temperature <= 0.0 {
                    aether_rt::serving::argmax_external(&logits)
                } else {
                    aether_rt::serving::sample_from_logits_v2(
                        &mut logits, params.temperature, params.top_p, params.top_k, &mut rng)
                };
                if Some(id) == stop { break; }
                *seen.entry(id).or_insert(0) += 1;
                let piece = sess.decode_ids(&[id]);
                running.push_str(&piece);
                let escaped = json_escape(&piece);
                let chunk = format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
                    escaped);
                send_chunk(t, &chunk);
                last = id;
                if !stop_strings.is_empty() && stop_strings.iter().any(|s| running.ends_with(s)) {
                    break;
                }
            }
            send_chunk(t, "data: [DONE]\n\n");
        } else {
            send_chunk(t, "data: {\"error\":\"--gguf not supplied\"}\n\n");
            send_chunk(t, "data: [DONE]\n\n");
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = state; let _ = req;
        send_chunk(t, "data: {\"error\":\"built without --features cuda\"}\n\n");
        send_chunk(t, "data: [DONE]\n\n");
    }

    let _ = t.write(b"0\r\n\r\n");
}

/// FR-19.5-extra-deep Phase 2 — SSE streaming over the BatchScheduler.
/// Submits a streaming request, then drains the worker's per-token
/// `StreamEvent` channel, writing one OpenAI-compatible `data:` delta
/// chunk per generated token and `[DONE]` on completion.  Chunk format
/// is byte-identical to the legacy single-session SSE path so existing
/// clients (incl. the OpenAI SDK) work unchanged regardless of
/// --max-concurrent.
#[cfg(feature = "cuda")]
unsafe fn handle_completion_streaming_scheduler_t(
    state: &ServerState, t: &mut dyn Transport, req: &JsonBody,
) {
    use aether_rt::batched_serving::StreamEvent;
    let Some(sched) = &state.scheduler else {
        let _ = send_text_t(t, 500, "scheduler unavailable");
        return;
    };
    let stop = state.cli.stop_token.or_else(|| {
        let eos = sched.eos_token();
        if eos >= 0 { Some(eos as usize) } else { None }
    });
    let params = aether_rt::serving::SamplingParams {
        temperature: req.temperature.unwrap_or(state.cli.default_temperature),
        top_p: req.top_p.unwrap_or(state.cli.default_top_p),
        top_k: req.top_k.unwrap_or(0),
        presence_penalty: req.presence_penalty.unwrap_or(state.cli.default_presence_penalty),
        frequency_penalty: req.frequency_penalty.unwrap_or(state.cli.default_frequency_penalty),
        seed: req.seed,
        logit_bias: req.logit_bias.clone(),
    };
    // Submit BEFORE writing any HTTP status — an admission failure here
    // can still surface as a clean 503 rather than a half-open SSE body.
    let rx = match sched.submit_streaming(
        req.prompt_ids.clone(), req.max_tokens, stop, params, req.stop_strings.clone(),
    ) {
        Ok(rx) => rx,
        Err(e) => { let _ = send_text_t(t, 503, &format!("scheduler: {}", e)); return; }
    };

    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n";
    let _ = t.write(headers.as_bytes());
    let send_chunk = |t: &mut dyn Transport, s: &str| {
        let hex_len = format!("{:x}\r\n", s.len());
        let _ = t.write(hex_len.as_bytes());
        let _ = t.write(s.as_bytes());
        let _ = t.write(b"\r\n");
    };

    while let Ok(ev) = rx.recv() {
        match ev {
            StreamEvent::Token { piece, .. } => {
                let escaped = json_escape(&piece);
                let chunk = format!(
                    "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
                    escaped);
                send_chunk(t, &chunk);
            }
            StreamEvent::Done { .. } => {
                send_chunk(t, "data: [DONE]\n\n");
                break;
            }
            StreamEvent::Error(e) => {
                let escaped = json_escape(&e);
                send_chunk(t, &format!("data: {{\"error\":\"{}\"}}\n\n", escaped));
                send_chunk(t, "data: [DONE]\n\n");
                break;
            }
        }
    }
    // Channel closed without a terminal event (worker dropped): still
    // terminate the SSE stream cleanly.
    let _ = t.write(b"0\r\n\r\n");
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

unsafe fn send_text_t(t: &mut dyn Transport, code: i32, msg: &str) -> Result<(), &'static str> {
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code, http_status_text(code), msg.len(), msg);
    t.write(resp.as_bytes()).map(|_| ())
}

unsafe fn send_json_t(t: &mut dyn Transport, code: i32, body: &str) -> Result<(), &'static str> {
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        code, http_status_text(code), body.len(), body);
    t.write(resp.as_bytes()).map(|_| ())
}

fn http_status_text(code: i32) -> &'static str {
    match code {
        200 => "OK", 400 => "Bad Request", 404 => "Not Found",
        500 => "Internal Server Error", 501 => "Not Implemented",
        _ => "Unknown",
    }
}

/// FR-17-extra-runtime-shape probe.  Open the GGUF, run `ModelConfig::from_gguf`,
/// print the detected shape, exit.  No weight upload.  Lets the operator
/// confirm a new model is detected correctly before attempting to serve it.
#[cfg(feature = "cuda")]
unsafe fn run_probe(path: &str) {
    use aether_rt::{aether_gguf_open, aether_gguf_close};
    use aether_rt::serving::ModelConfig;
    use std::os::raw::c_int;
    eprintln!("[probe] opening {}", path);
    let h = aether_gguf_open(path.as_ptr() as i64, path.len() as c_int);
    if h < 0 { eprintln!("[probe] gguf open failed: {}", h); std::process::exit(1); }
    let cfg = ModelConfig::from_gguf(h);
    println!("Detected model shape:");
    println!("  architecture : {}", cfg.arch);
    println!("  n_layers     : {}", cfg.n_layers);
    println!("  d_model      : {}", cfg.d_model);
    println!("  n_q_heads    : {}", cfg.n_q_heads);
    println!("  n_kv_heads   : {}", cfg.n_kv_heads);
    println!("  head_dim     : {}", cfg.head_dim);
    println!("  d_kv         : {}", cfg.d_kv);
    println!("  d_ff         : {}", cfg.d_ff);
    println!("  vocab        : {}", cfg.vocab);
    println!("  rope_base    : {}", cfg.rope_base);
    println!("  norm_eps     : {:.2e}", cfg.norm_eps);
    if cfg.kv_lora_rank > 0 {
        println!();
        println!("MLA (Multi-head Latent Attention) detected:");
        println!("  kv_lora_rank      : {}", cfg.kv_lora_rank);
        println!("  q_lora_rank       : {} {}",
            cfg.q_lora_rank,
            if cfg.q_lora_rank == 0 { "(direct Q, no LoRA)" } else { "" });
        println!("  qk_head_dim       : {}  (qk_nope+qk_rope)", cfg.qk_head_dim);
        println!("  qk_rope_head_dim  : {}", cfg.qk_rope_head_dim);
        println!("  v_head_dim        : {}", cfg.v_head_dim);
        println!("  leading_dense_blks: {}", cfg.leading_dense_blocks);
        println!("  n_shared_experts  : {}", cfg.n_shared_experts);
    }
    if cfg.sliding_window > 0 {
        println!("  sliding_window: {}", cfg.sliding_window);
    }
    if cfg.yarn_factor > 1.0 {
        let mscale = 1.0 + cfg.yarn_log_multiplier * cfg.yarn_factor.ln();
        println!();
        println!("YaRN RoPE scaling detected:");
        println!("  yarn_factor         : {}  (s, context multiplier)", cfg.yarn_factor);
        println!("  yarn_log_multiplier : {}", cfg.yarn_log_multiplier);
        println!("  yarn_orig_ctx       : {}", cfg.yarn_orig_ctx);
        println!("  yarn_beta_fast      : {}", cfg.yarn_beta_fast);
        println!("  yarn_beta_slow      : {}", cfg.yarn_beta_slow);
        println!("  → attention mscale  : {:.4} (Q*K^T scores ×{:.4})", mscale, mscale * mscale);
    }
    println!();
    // Kernel-constraint check (mirrors new_with_mode's checks).
    let mut violations = Vec::<String>::new();
    if cfg.head_dim == 0 || cfg.head_dim % 32 != 0 || cfg.head_dim > 256 {
        violations.push(format!("head_dim={} not supported (need multiple of 32, ≤ 256)", cfg.head_dim));
    }
    if cfg.n_kv_heads == 0 || cfg.n_q_heads % cfg.n_kv_heads != 0 {
        violations.push(format!("n_q_heads({}) % n_kv_heads({}) != 0", cfg.n_q_heads, cfg.n_kv_heads));
    }
    if cfg.d_model == 0 || cfg.d_model % 256 != 0 {
        violations.push(format!("d_model({}) must be a multiple of 256 (Q4_K super-block input dim)", cfg.d_model));
    }
    if cfg.d_ff == 0 || cfg.d_ff % 256 != 0 {
        violations.push(format!("d_ff({}) must be a multiple of 256 (Q4_K super-block input dim)", cfg.d_ff));
    }
    if violations.is_empty() {
        println!("All shape constraints satisfied.");
    } else {
        println!("Shape constraint violations:");
        for v in &violations { println!("  - {}", v); }
    }
    // Architecture compatibility — the kernels today implement Qwen2.5-style
    // dense attention + dense FFN with GQA.  Other arches need additional
    // per-arch work even if the shape passes the SHAPE constraints above.
    println!();
    println!("Architecture compatibility:");
    let (loadable, notes): (bool, &[&str]) = match cfg.arch.as_str() {
        "qwen2" => (true, &[
            "  ✅ Qwen2.5 dense attention + GQA + dense FFN.",
            "     Today's kernel surface implements this arch directly.",
            "     Loadable for any Qwen2.5 variant (7B verified; 14B/32B should work — needs GGUF).",
        ]),
        "qwen3" => (true, &[
            "  ✅ Qwen3 dense attention + GQA + Q/K RMS norm + dense FFN.",
            "     Per-head Q/K RMS norm + no-bias loading shipped (FR-17-extra-qwen3-fwd).",
            "     Verified on Qwen3-8B GGUF — generating real text @ 11 tok/s.",
        ]),
        "qwen3vl" => (true, &[
            "  ✅ Qwen3-VL text-only LLM body is identical to Qwen3 (verified on this hardware).",
            "     Vision tower not exposed today; image inputs require a separate forward path.",
        ]),
        "qwen3moe" => (true, &[
            "  ✅ Mixture-of-Experts code path shipped (FR-17-extra-moe-fwd).",
            "     Loads ffn_gate_inp + ffn_*_exps tensors; new fused_q4k_expert_matmul_seq1",
            "     kernel does per-expert dispatch with concatenated weight slices.",
            "     Forward pass: router matmul + host top-k + n_used × (gate, up, silu*mul, down)",
            "     + weighted accumulate.  CUDA graph capture skipped for MoE (host top-k).",
            "     Verification needs more GPU memory than this 8 GB card (30B is ~17 GB Q4_K_M).",
        ]),
        "deepseek2" => (false, &[
            "  🟡 DeepSeek-V2 MLA kernels SHIPPED (FR-17-extra-mla-fwd, partial):",
            "      - paged_attention_mla_devarg — split per-head dims for Q/K (192) vs V",
            "        (128), CPU↔GPU bit-witnessed in cuda_attention_mla_parity.rs.",
            "      - paged_append_kv_mla_devarg — independent K/V row strides, also",
            "        witnessed.",
            "      - ModelConfig reads kv_lora_rank / q_lora_rank / qk_head_dim /",
            "        qk_rope_head_dim / v_head_dim / leading_dense_blocks /",
            "        n_shared_experts from GGUF metadata.",
            "      - load_block reads attn_kv_a_mqa / attn_kv_a_norm / attn_kv_b and the",
            "        optional attn_q_a / attn_q_a_norm / attn_q_b LoRA tensors.",
            "     REMAINING for end-to-end: (a) c_kv projection + per-head decompression",
            "     glue kernels, (b) partial-dim RoPE on Q_rope (per-head, qk_rope_head_dim)",
            "     and K_rope_shared (single broadcast vec).  block_forward_devarg detects",
            "     MLA layout today and panics with a clear pointer rather than producing",
            "     garbage activations.  MoE FFN is shipped via FR-17-extra-moe-fwd (above).",
            "  ✅ Q4_0 dispatch — landed (FR-17-extra-q4_0-fwd).  18-byte 32-elem",
            "     blocks (f16 scale + 16 nibble-packed bytes); fused matmul",
            "     kernel + dispatch_matmul wiring + GGUF upload all support",
            "     dtype=2.  CPU↔GPU witnessed in cuda_q4_0_matmul_parity.rs.",
            "     End-to-end V2-Lite Q4_0 still needs >8 GB GPU (cnc P100s).",
            "  ✅ YaRN RoPE scaling — landed (FR-17-extra-mla-fwd YaRN).  Per-",
            "     frequency-dim scale factor (ramp between beta_fast=32 and",
            "     beta_slow=1) + attention mscale = 1 + log_mult*ln(s) applied",
            "     to Q*K scores.  CPU↔GPU witnessed in cuda_yarn_rope_parity.rs.",
            "  ✅ MoE shared experts — landed.  load_block reads ffn_{gate,up,",
            "     down}_shexp tensors when expert_shared_count > 0; moe_ffn_",
            "     forward runs them as a single FUSED MLP at hidden dim",
            "     n_shared*expert_ff_dim and adds the result at weight 1.0.",
            "     Witnessed in cuda_moe_shared_expert_parity.rs.",
            "  ⚠ Q4_K unaligned-d_ff fallback (d_ff=10944, expert_ff=1408 both",
            "     non-multiples of 256) — pending.",
        ]),
        "gemma3" => (true, &[
            "  ✅ Gemma3 dispatch shipped (FR-17-extra-gemma-fwd):",
            "      1) head_dim-flexible attention (paged_attention_flex_devarg) —",
            "         handles head_dim ∈ [1, 256] including 168 (verified on synthetic).",
            "      2) Sliding-window attention via sliding_window arg in the flex kernel.",
            "         Restricts t-range to [max(0, cur_seq-W), cur_seq).",
            "      3) Pre+post-attention RMS norm — load_block reads",
            "         post_attention_norm.weight + post_ffw_norm.weight via _opt loader;",
            "         block_forward_devarg applies them after Q/O proj and after FFN down.",
            "     Requires --paged mode (contiguous flex variant is a follow-on).",
            "     End-to-end verification needs >8 GB GPU (27B model).",
        ]),
        "llama" => (false, &[
            "  ⚠ Llama is close to Qwen2.5 (no attention biases, no Q/K norm).  Should be",
            "     a small variant of the existing kernels — drop the `bias_add` calls for",
            "     attn_q/k/v after the matmul.  FR-17-extra-llama-fwd.",
        ]),
        _ => (false, &[
            "  ❓ Unknown architecture — requires per-arch implementation work.",
        ]),
    };
    for line in notes { println!("{}", line); }
    if !loadable && violations.is_empty() {
        println!();
        println!("→ shape OK, but arch-specific kernel work needed before this model loads.");
    } else if loadable && violations.is_empty() {
        println!();
        println!("→ READY: model would load with `aether-serve --gguf <this>`.");
    }

    // Tensor dtype histogram — quant-aware loaders need to know what's in
    // the file.  Today's matmul kernels handle dtype 12 (Q4_K) + 14 (Q6_K)
    // only; F16 / F32 / other Q-types need additional work.
    let n_tensors = aether_rt::aether_gguf_n_tensors(h);
    let mut hist = std::collections::BTreeMap::<i32, usize>::new();
    let mut name_examples = std::collections::BTreeMap::<i32, String>::new();
    let mut name_buf = [0u8; 256];
    for i in 0..n_tensors {
        let dt = aether_rt::aether_gguf_get_tensor_dtype(h, i);
        *hist.entry(dt).or_default() += 1;
        name_examples.entry(dt).or_insert_with(|| {
            let n = aether_rt::aether_gguf_get_tensor_name(h, i, name_buf.as_mut_ptr() as i64, 256);
            if n > 0 {
                String::from_utf8_lossy(&name_buf[..n as usize]).to_string()
            } else { format!("<tensor {}>", i) }
        });
    }
    println!();
    println!("Tensor dtype histogram ({} tensors total):", n_tensors);
    let dtype_name = |dt: i32| -> &'static str {
        match dt {
            0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1",
            6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
            10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
            14 => "Q6_K", 15 => "Q8_K", 30 => "BF16",
            _ => "?",
        }
    };
    for (dt, count) in &hist {
        let supported = matches!(*dt, 0 | 12 | 14);
        let mark = if supported { "✅" } else { "⚠" };
        let example = name_examples.get(dt).map(|s| s.as_str()).unwrap_or("");
        println!("  {} dtype {:3} ({:6}): {:4} tensors  e.g. {}",
            mark, dt, dtype_name(*dt), count, example);
    }
    let unsupported_count: usize = hist.iter()
        .filter(|(dt, _)| !matches!(**dt, 0 | 12 | 14))
        .map(|(_, c)| *c).sum();
    if unsupported_count > 0 {
        println!();
        println!("→ {} tensor(s) use dtypes outside the Q4_K + Q6_K + F32 set supported today.",
            unsupported_count);
        println!("  FR-17-extra-f16-fwd: add F16 (dtype 1) weight + matmul path.");
    }
    aether_gguf_close(h);
}

fn main() {
    let cli = parse_cli();
    #[cfg(feature = "cuda")]
    if cli.probe {
        let path = cli.gguf.as_ref().unwrap_or_else(|| {
            eprintln!("[aether-serve] --probe requires --gguf <path>");
            std::process::exit(2);
        });
        unsafe { run_probe(path); }
        std::process::exit(0);
    }
    let tls_on = cli.tls;
    let tls_cn = cli.tls_cn.clone();
    let state = match ServerState::new(cli) {
        Ok(s) => s,
        Err(e) => { eprintln!("[aether-serve] startup error: {}", e); std::process::exit(1); }
    };
    let state = std::sync::Arc::new(state);
    unsafe {
        // FR-x-extra: bind via aether_tcp_listen_addr so --bind can switch
        // between 0.0.0.0 (default, reachable from podman / other hosts)
        // and 127.0.0.1 (loopback only).  The legacy `aether_tcp_listen`
        // hardcodes 127.0.0.1 — the log line used to lie about 0.0.0.0
        // (kokonoe substrate-swap finding #3).
        let bind_bytes = state.cli.bind.as_bytes();
        let listener = aether_tcp_listen_addr(
            bind_bytes.as_ptr() as i64, bind_bytes.len() as std::os::raw::c_int,
            state.cli.port,
        );
        // Bare-port fallback path retained so older deployments that only
        // know the loopback semantics still build green.
        let _ = aether_tcp_listen;
        if listener < 0 {
            eprintln!("[aether-serve] failed to bind {}:{}", state.cli.bind, state.cli.port);
            std::process::exit(1);
        }
        let bound_port = aether_tcp_listener_port(listener);
        let scheme = if tls_on { "https" } else { "http" };
        eprintln!("[aether-serve] listening on {}:{} ({}, model={}, concurrent=on)",
            state.cli.bind, bound_port, scheme, state.cli.model);
        if tls_on {
            eprintln!("[aether-serve] TLS 1.3 enabled; fresh self-signed Ed25519 cert per session (CN={})", tls_cn);
            eprintln!("[aether-serve] try:");
            eprintln!("  curl -k --tlsv1.3 https://localhost:{}/health", bound_port);
            eprintln!("  curl -k --tlsv1.3 https://localhost:{}/v1/models", bound_port);
        } else {
            eprintln!("[aether-serve] try:");
            eprintln!("  curl http://localhost:{}/v1/models", bound_port);
            eprintln!("  curl http://localhost:{}/health", bound_port);
        }

        loop {
            let stream = aether_tcp_accept_one(listener);
            if stream < 0 {
                eprintln!("[serve] accept returned {} (continuing)", stream);
                continue;
            }
            // Spawn a thread per accepted connection.  The QwenSession behind
            // ServerState is Mutex-wrapped, so threads serialize on actual
            // forward-pass work (single GPU), but can run HTTP/TLS / decode
            // concurrently to accept latency.
            let state_clone = state.clone();
            let tls_cn_clone = tls_cn.clone();
            std::thread::spawn(move || {
                unsafe {
                    if tls_on {
                        let mut tls = TlsStream::accept(stream, &tls_cn_clone);
                        match tls.handshake() {
                            Ok(_) => handle_request(&state_clone, &mut tls),
                            Err(e) => eprintln!("[serve] tls handshake failed: {}", e),
                        }
                    } else {
                        let mut plain = PlainTcp { fd: stream };
                        handle_request(&state_clone, &mut plain);
                    }
                    aether_tcp_stream_close(stream);
                }
            });
        }
        #[allow(unreachable_code)]
        { aether_tcp_close(listener); }
    }
}
