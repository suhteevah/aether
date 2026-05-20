//! End-to-end HTTP wire-up for matt-voice's serving deploy.
//!
//! Demonstrates the full request/response chain through the existing
//! Aether runtime fns -- TCP listen/accept, HTTP parse, OpenAI
//! response render -- without paying the 24 GB Qwen2.5-7B load cost.
//! Uses a SIMULATED generation step (echoes the prompt) so the test
//! runs in milliseconds; the real-Qwen forward chain shipped in the
//! qwen25_autoregressive_* tests is a drop-in for the simulate step.
//!
//! This is the matt-voice deploy proof for the "serve" half:
//!   curl -d '{"prompt":"...","max_tokens":N}' /v1/chat/completions
//!   -> JSON with generated text
//!
//! The aether-serve BINARY (`trainer/src/bin/serve.rs`) ties this
//! together with the real Qwen forward + tokenizer for a runnable
//! `aether-serve --gguf <path> --port <N>` deploy.

use std::os::raw::c_int;
use std::ffi::c_void;

use aether_rt::{
    aether_tcp_listen, aether_tcp_listener_port, aether_tcp_accept_one,
    aether_tcp_send, aether_tcp_recv, aether_tcp_close, aether_tcp_stream_close,
    aether_http_parse_request, aether_http_write_response_200,
    aether_openai_render_completion,
};

/// Server side: accept one connection, parse the HTTP request,
/// generate (simulated) content, render the OpenAI response, send.
unsafe fn handle_one_request(listener: i64) {
    let stream = aether_tcp_accept_one(listener);
    assert!(stream >= 0, "accept returned {}", stream);

    // Read request bytes -- with HTTP/1.1 + Content-Length we'd parse
    // headers properly, but for this wire-up test we read until the
    // body marker.
    let mut req_buf = vec![0u8; 4096];
    let got = aether_tcp_recv(stream, req_buf.as_mut_ptr() as i64, req_buf.len() as i64);
    assert!(got > 0, "recv returned {}", got);
    let req_bytes = &req_buf[..got as usize];

    // Parse the request.
    let mut strings = vec![0u8; 256];
    let mut m_len: c_int = 0;
    let mut p_len: c_int = 0;
    let body_off = aether_http_parse_request(
        req_bytes.as_ptr() as *const c_void, got as c_int,
        strings.as_mut_ptr() as *mut c_void, strings.len() as c_int,
        &mut m_len, &mut p_len,
    );
    assert!(body_off > 0, "parse failed: {}", body_off);
    let method = std::str::from_utf8(&strings[..m_len as usize]).unwrap();
    let path = std::str::from_utf8(&strings[m_len as usize .. (m_len + p_len) as usize]).unwrap();
    let body = &req_bytes[body_off as usize..];
    eprintln!("[server] {} {} body={:?}", method, path,
        std::str::from_utf8(body).unwrap_or("<bin>"));

    // Generate (simulated): for the wire-up proof, the assistant
    // content is a fixed string. In the real aether-serve binary the
    // forward chain from qwen25_autoregressive_cuda.rs lives here.
    //
    // Pretend we generated 5 tokens from a 4-token prompt.
    let generated_text = "Hello, world! I'm a 2";
    let prompt_tokens = 4;
    let completion_tokens = 5;

    // Render OpenAI response.
    let resp_id = b"chatcmpl-aether-1";
    let model = b"matt-voice";
    let mut json_buf = vec![0u8; 2048];
    let n_json = aether_openai_render_completion(
        resp_id.as_ptr() as *const c_void, resp_id.len() as c_int,
        model.as_ptr() as *const c_void, model.len() as c_int,
        generated_text.as_ptr() as *const c_void, generated_text.len() as c_int,
        prompt_tokens, completion_tokens,
        json_buf.as_mut_ptr() as *mut c_void, json_buf.len() as c_int,
    );
    assert!(n_json > 0, "render returned {}", n_json);

    // Wrap in HTTP/1.1 response.
    let mut http_buf = vec![0u8; 4096];
    let n_http = aether_http_write_response_200(
        json_buf.as_ptr() as *const c_void, n_json,
        http_buf.as_mut_ptr() as *mut c_void, http_buf.len() as c_int,
    );
    assert!(n_http > 0);

    // Send.
    let sent = aether_tcp_send(stream, http_buf.as_ptr() as i64, n_http as i64);
    assert_eq!(sent, n_http as i64);

    aether_tcp_stream_close(stream);
}

#[test]
fn aether_serve_http_end_to_end() {
    unsafe {
        let listener = aether_tcp_listen(0);
        assert!(listener >= 0);
        let port = aether_tcp_listener_port(listener);
        assert!(port > 0);
        eprintln!("[test] aether-serve listening on port {}", port);

        // Server thread.
        let t = std::thread::spawn(move || unsafe { handle_one_request(listener); });

        // Client side.
        std::thread::sleep(std::time::Duration::from_millis(100));
        use std::io::{Read, Write};
        let mut sock = std::net::TcpStream::connect(("127.0.0.1", port as u16))
            .expect("client connect");
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\n\
             Host: localhost\r\n\
             Content-Type: application/json\r\n\
             Content-Length: 56\r\n\r\n\
             {{\"prompt_ids\":[9707,11,1879,0],\"max_tokens\":5}}"
        );
        sock.write_all(request.as_bytes()).unwrap();
        sock.flush().unwrap();

        let mut resp = Vec::new();
        sock.read_to_end(&mut resp).unwrap();
        let resp_str = std::str::from_utf8(&resp).expect("response not utf8");
        eprintln!("[client] response:\n{}", resp_str);

        // Verify response shape.
        assert!(resp_str.starts_with("HTTP/1.1 200 OK"), "bad status line");
        assert!(resp_str.contains("Content-Length:"), "missing Content-Length");
        let body_off = resp_str.find("\r\n\r\n").unwrap() + 4;
        let body = &resp_str[body_off..];
        assert!(body.contains("\"object\":\"chat.completion\""), "wrong object kind");
        assert!(body.contains("\"model\":\"matt-voice\""), "wrong model name");
        assert!(body.contains("\"role\":\"assistant\""), "wrong message role");
        assert!(body.contains("\"content\":\"Hello, world! I'm a 2\""),
            "generated content not in response");
        assert!(body.contains("\"prompt_tokens\":4"));
        assert!(body.contains("\"completion_tokens\":5"));

        t.join().unwrap();
        aether_tcp_close(listener);

        eprintln!("[OK] aether-serve HTTP wire-up verified end-to-end");
    }
}
