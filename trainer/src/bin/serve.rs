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
    aether_random_bytes,
    tls13::TlsServerSession,
    http2,
};

#[cfg(feature = "cuda")]
use aether_rt::serving::{QwenSession, SharedKvPool};

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
            "-h" | "--help" => {
                eprintln!("aether-serve [--port N] [--model NAME] [--gguf PATH] [--max-tokens N] [--stop-token ID|none] [--warmup N] [--tls] [--tls-cn NAME] [--paged]");
                eprintln!();
                eprintln!("  Listens on 0.0.0.0:port for OpenAI-compat /v1/chat/completions.");
                eprintln!("  --gguf points at any Qwen2.5-7B-Instruct Q4_K_M model file.");
                eprintln!("  --warmup N runs N synthetic decode steps on startup to drive");
                eprintln!("    the GPU into P0/P2 power state and pre-capture the graph.");
                eprintln!("  --tls enables TLS 1.3 (self-signed Ed25519 cert generated on startup");
                eprintln!("        with --tls-cn as the cert CN; default port becomes 8443).");
                eprintln!("  --paged routes K/V through paged_append_kv_devarg +");
                eprintln!("        paged_attention_seq1_devarg (FR-19.4-extra) — identity");
                eprintln!("        page table, bit-identical token output to contiguous mode.");
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
    /// FR-19.4-extra-tenant: shared KV pool, kept alive for the lifetime of
    /// the server when --pool-blocks > 0.  Future-session work routes new
    /// concurrent requests through additional sessions bound to this pool.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pool: Option<std::sync::Arc<SharedKvPool>>,
}

impl ServerState {
    fn new(cli: Cli) -> Result<Self, String> {
        #[cfg(feature = "cuda")]
        {
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
                    let mut s = if let Some(p) = &pool {
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
            return Ok(ServerState { cli, session, pool });
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
    match (method_str.as_str(), path_str.as_str()) {
        ("GET", "/health") => write_h2_text(t, stream_id, 200, b"ok"),
        ("GET", "/v1/models") => {
            let body = format!(
                "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"aether\"}}]}}",
                state.cli.model);
            write_h2_json(t, stream_id, 200, body.as_bytes());
        }
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            let resp = match parse_body(&body, state.cli.max_tokens_default) {
                Ok(req) => render_completion_json(state, &req),
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
        match &state.session {
            Some(sess_mu) => {
                let mut sess = sess_mu.lock().unwrap();
                let stop = state.cli.stop_token.or_else(|| {
                    if sess.eos_token >= 0 { Some(sess.eos_token as usize) } else { None }
                });
                let ids = sess.generate(&req.prompt_ids, req.max_tokens, stop);
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
        ("POST", "/v1/chat/completions") | ("POST", "/v1/completions") => {
            handle_completion_t(state, t, body)
        }
        _ => { let _ = send_text_t(t, 404, "not found"); }
    }
}

unsafe fn handle_list_models_t(state: &ServerState, t: &mut dyn Transport) {
    let body = format!(
        "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"aether\"}}]}}",
        state.cli.model);
    let _ = send_json_t(t, 200, &body);
}

unsafe fn handle_completion_t(state: &ServerState, t: &mut dyn Transport, body: &[u8]) {
    let req = match parse_body(body, state.cli.max_tokens_default) {
        Ok(r) => r,
        Err(e) => { let _ = send_text_t(t, 400, e); return; }
    };

    if req.text_prompt.is_some() && req.prompt_ids.is_empty() {
        let _ = send_text_t(t, 501,
            "text-prompt encode not wired yet; pass prompt_ids (token ids) for now");
        return;
    }

    if req.stream {
        handle_completion_streaming_t(state, t, &req);
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
                send_chunk(t, &chunk);
                last = id;
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
            "  ⚠ DeepSeek-V2/Coder MoE FFN is shipped via FR-17-extra-moe-fwd above.",
            "     REMAINING: Multi-head Latent Attention (MLA) — KV is projected into a",
            "     low-dim latent space, then decompressed for attention.  Different tensor",
            "     layout (attn_kv_a + attn_kv_b instead of attn_k + attn_v).  Needs a new",
            "     attention kernel.  FR-17-extra-mla-fwd (multi-session).",
        ]),
        "gemma3" => (false, &[
            "  ⚠ Gemma3 has THREE blockers:",
            "      1) head_dim=168 — our attention_seq1 lays out per_lane = head_dim>>5",
            "         and assumes head_dim is a multiple of 32.  Need a head_dim-flexible",
            "         attention kernel variant (per_lane = ceil(head_dim/32) + bounds check).",
            "      2) Sliding-window attention — attention scope is local-window, not full",
            "         causal.  Needs a sw-aware attention variant.",
            "      3) Gemma3-specific RMSNorm placement (pre + post-attention norm vs",
            "         Qwen's single attn_norm).  Block forward needs rewiring.",
            "     All three are individually tractable but combine into a separate forward",
            "     path.  FR-17-extra-gemma-fwd (multi-session).",
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
        let listener = aether_tcp_listen(state.cli.port);
        if listener < 0 {
            eprintln!("[aether-serve] failed to bind port {}", state.cli.port);
            std::process::exit(1);
        }
        let bound_port = aether_tcp_listener_port(listener);
        let scheme = if tls_on { "https" } else { "http" };
        eprintln!("[aether-serve] listening on 0.0.0.0:{} ({}, model={}, concurrent=on)",
            bound_port, scheme, state.cli.model);
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
