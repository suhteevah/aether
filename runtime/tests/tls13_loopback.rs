//! TLS 1.3 self-loopback integration test.
//!
//! Wires an in-process `TlsServerSession` to a matching `client_for_test::TestClient`
//! and runs the full handshake + 1 round-trip of application data.
//!
//! Verifies (witness):
//!   1. ClientHello -> ServerHello flight roundtrips through the record layer
//!   2. The encrypted server flight (EE || Cert || CV || Finished) decrypts
//!   3. Server CertificateVerify Ed25519 signature passes verify
//!   4. Server Finished MAC matches client-side recomputation
//!   5. Client Finished is accepted by the server
//!   6. Both sides arrive at matching app traffic keys
//!   7. App-data records exchange in both directions and decrypt to original bytes
//!
//! roadmap: P19.1-extra

use aether_rt::tls13::{client_for_test::TestClient, record, TlsServerSession, State};
use aether_rt::tls13::REC_APPLICATION_DATA;

/// Drive a full TLS 1.3 handshake + 1 round-trip of app data.
#[test]
fn tls13_full_loopback() {
    // ---- Deterministic test material ----
    let server_seed = [0x11u8; 32];
    let server_random = [0x22u8; 32];
    let server_x25519_priv = [0x33u8; 32];

    let client_x25519_priv = [0x44u8; 32];
    let client_random = [0x55u8; 32];
    let client_session_id: Vec<u8> = (0..32u8).collect();

    // ---- Server ----
    let mut server = TlsServerSession::new(
        &server_seed,
        &server_random,
        &server_x25519_priv,
        "aether-loopback",
        b"\x01",
    );
    assert_eq!(server.state(), State::ExpectClientHello);

    // ---- Client builds ClientHello ----
    let client = TestClient::new(client_x25519_priv, client_random, client_session_id.clone());
    let ch_record = client.build_client_hello_record();
    println!("[loopback] ClientHello record: {} bytes", ch_record.len());

    // ---- Feed CH to server ----
    let app_out = server.feed(&ch_record).expect("server should accept ClientHello");
    assert!(app_out.is_empty(), "no app data yet");
    assert_eq!(server.state(), State::SentServerFlight);

    // ---- Drain server flight ----
    let server_flight = server.take_outbound();
    println!("[loopback] Server flight: {} bytes", server_flight.len());
    assert!(!server_flight.is_empty());

    // ---- Client processes flight, builds Finished ----
    let (client_app_keys, server_app_keys, client_fin_rec, _transcript) =
        client
            .process_server_flight_and_build_client_finished(&server_flight, &ch_record)
            .expect("client should accept server flight");
    println!("[loopback] Client Finished record: {} bytes", client_fin_rec.len());

    // ---- Feed Finished to server ----
    let app_out = server.feed(&client_fin_rec).expect("server should accept client Finished");
    assert!(app_out.is_empty(), "no app data yet");
    assert_eq!(server.state(), State::Connected, "handshake complete");

    // ---- App-data round-trip: server -> client ----
    let s_to_c_msg = b"Hello from the Aether TLS 1.3 server!";
    server.send_app_data(s_to_c_msg).unwrap();
    let server_app_record = server.take_outbound();
    println!("[loopback] Server app record: {} bytes (plaintext {} bytes)",
             server_app_record.len(), s_to_c_msg.len());
    // Decrypt on the client side with server_app_keys @ seq 0.
    let (inner_type, plain, _consumed) =
        record::open_record(&server_app_record, &server_app_keys.key, &server_app_keys.iv, 0)
            .expect("client must decrypt server app data");
    assert_eq!(inner_type, REC_APPLICATION_DATA);
    assert_eq!(plain, s_to_c_msg);

    // ---- App-data round-trip: client -> server ----
    let c_to_s_msg = b"Hello from the Aether TLS 1.3 client!";
    let mut client_app_record = Vec::new();
    record::seal_record(
        &mut client_app_record,
        &client_app_keys.key, &client_app_keys.iv, 0,
        REC_APPLICATION_DATA, c_to_s_msg,
    );
    println!("[loopback] Client app record: {} bytes", client_app_record.len());
    let recovered = server.feed(&client_app_record).expect("server must accept client app data");
    assert_eq!(recovered, c_to_s_msg);

    println!("[loopback] TLS 1.3 handshake + 2 app data records — PASS");
}

/// Negative test: a tampered server flight (single byte flipped) must fail the
/// client's CV signature or MAC checks.
#[test]
fn tls13_tampered_server_flight_rejected() {
    let server_seed = [0x11u8; 32];
    let server_random = [0x22u8; 32];
    let server_x25519_priv = [0x33u8; 32];
    let client_x25519_priv = [0x44u8; 32];
    let client_random = [0x55u8; 32];
    let client_session_id: Vec<u8> = (0..16u8).collect();

    let mut server = TlsServerSession::new(
        &server_seed, &server_random, &server_x25519_priv,
        "aether-loopback", b"\x02",
    );
    let client = TestClient::new(client_x25519_priv, client_random, client_session_id);
    let ch_record = client.build_client_hello_record();
    server.feed(&ch_record).unwrap();
    let mut flight = server.take_outbound();
    // Flip one byte in the encrypted second record (skip the 5-byte SH header
    // and skip the first record).
    // First record = SH (cleartext). Find its length.
    let sh_len = u16::from_be_bytes([flight[3], flight[4]]) as usize;
    let after_sh = 5 + sh_len;
    // Encrypted record starts at after_sh; flip a byte in its ciphertext.
    let target = after_sh + 6; // arbitrary inside ciphertext
    flight[target] ^= 1;
    let result = client
        .process_server_flight_and_build_client_finished(&flight, &ch_record);
    assert!(result.is_err(), "tampered flight must be rejected");
}
