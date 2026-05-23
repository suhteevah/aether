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

use aether_rt::tls13::{client_for_test::TestClient, record, TlsServerSession, State, TlsClientSession, ClientState};
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

/// Drive both halves via the public TlsServerSession + TlsClientSession state
/// machines (no direct use of client_for_test).
#[test]
fn tls13_client_session_loopback() {
    let server_seed = [0xa1u8; 32];
    let server_random = [0xa2u8; 32];
    let server_x25519_priv = [0xa3u8; 32];
    let client_x25519_priv = [0xa4u8; 32];
    let client_random = [0xa5u8; 32];
    let client_session_id: Vec<u8> = (0..16u8).collect();

    let mut server = TlsServerSession::new(
        &server_seed, &server_random, &server_x25519_priv,
        "aether-client-session-test", b"\x03",
    );
    let mut client = TlsClientSession::new(client_x25519_priv, client_random, client_session_id);
    assert_eq!(client.state(), ClientState::ExpectServerFlight);

    // Client → server: ClientHello
    let to_server = client.take_outbound();
    assert!(!to_server.is_empty());
    let app1 = server.feed(&to_server).unwrap();
    assert!(app1.is_empty());
    assert_eq!(server.state(), State::SentServerFlight);

    // Server → client: full flight
    let to_client = server.take_outbound();
    assert!(!to_client.is_empty());
    let app2 = client.feed(&to_client).unwrap();
    assert!(app2.is_empty());
    assert!(client.is_handshake_done(), "client should be Connected after server flight");

    // Client → server: client Finished
    let fin = client.take_outbound();
    assert!(!fin.is_empty());
    let app3 = server.feed(&fin).unwrap();
    assert!(app3.is_empty());
    assert_eq!(server.state(), State::Connected);

    // Application data round-trip (both ways).
    let s2c = b"server->client via TlsClientSession";
    server.send_app_data(s2c).unwrap();
    let bytes = server.take_outbound();
    let got = client.feed(&bytes).unwrap();
    assert_eq!(got, s2c);

    let c2s = b"client->server via TlsClientSession";
    client.send_app_data(c2s).unwrap();
    let bytes = client.take_outbound();
    let got = server.feed(&bytes).unwrap();
    assert_eq!(got, c2s);
}

/// ALPN negotiation loopback: client offers ["h2","http/1.1"], server supports
/// ["http/1.1","h2"], result = "http/1.1" (server preference).
#[test]
fn tls13_alpn_negotiation() {
    use aether_rt::tls13::TlsServerSession;
    let server_seed = [0xb1u8; 32];
    let server_random = [0xb2u8; 32];
    let server_x25519_priv = [0xb3u8; 32];
    let client_x25519_priv = [0xb4u8; 32];
    let client_random = [0xb5u8; 32];

    let server_alpn = vec![b"http/1.1".to_vec(), b"h2".to_vec()];
    let client_alpn = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    let mut server = TlsServerSession::new_with_alpn(
        &server_seed, &server_random, &server_x25519_priv,
        "aether-alpn-test", b"\x05",
        server_alpn,
    );
    let mut client = TlsClientSession::new_with_alpn(
        client_x25519_priv, client_random, (0..16u8).collect(),
        client_alpn,
    );

    // Drive handshake.
    let to_server = client.take_outbound();
    server.feed(&to_server).unwrap();
    let to_client = server.take_outbound();
    client.feed(&to_client).unwrap();
    assert!(client.is_handshake_done());
    let fin = client.take_outbound();
    server.feed(&fin).unwrap();
    assert_eq!(server.state(), State::Connected);

    // Both sides see the same negotiated protocol — server's first preference
    // that also appears in client's list = "http/1.1".
    assert_eq!(server.negotiated_alpn(), Some(b"http/1.1".as_slice()),
        "server picked: {:?}", server.negotiated_alpn());
    assert_eq!(client.negotiated_alpn(), Some(b"http/1.1".as_slice()),
        "client saw: {:?}", client.negotiated_alpn());
}

/// ALPN where client offers "h2" only and server supports "h2" too -> h2 wins.
#[test]
fn tls13_alpn_h2_only() {
    use aether_rt::tls13::TlsServerSession;
    let mut server = TlsServerSession::new_with_alpn(
        &[0xc1u8; 32], &[0xc2u8; 32], &[0xc3u8; 32],
        "aether-alpn-h2", b"\x06",
        vec![b"h2".to_vec(), b"http/1.1".to_vec()],
    );
    let mut client = TlsClientSession::new_with_alpn(
        [0xc4u8; 32], [0xc5u8; 32], (0..8u8).collect(),
        vec![b"h2".to_vec()],
    );
    server.feed(&client.take_outbound()).unwrap();
    client.feed(&server.take_outbound()).unwrap();
    server.feed(&client.take_outbound()).unwrap();
    assert_eq!(server.negotiated_alpn(), Some(b"h2".as_slice()));
    assert_eq!(client.negotiated_alpn(), Some(b"h2".as_slice()));
}

/// Cert chain / trust-anchor verifier: client configured with the server's
/// SPKI pubkey as a trusted anchor; handshake completes and self-sig is
/// verified.  Then run with a WRONG anchor and verify it's rejected.
#[test]
fn tls13_trust_anchor_positive() {
    use aether_rt::tls13::TlsServerSession;
    use aether_rt::tls13::client_for_test::verify_self_signed_ed25519_cert;

    // Build server.  Compute the SPKI pubkey it will use (Ed25519 derive from seed).
    let seed = [0xe1u8; 32];
    let mut server_pub = [0u8; 32];
    unsafe {
        aether_rt::aether_ed25519_derive_public(
            seed.as_ptr() as *const std::ffi::c_void,
            server_pub.as_mut_ptr() as *mut std::ffi::c_void,
        );
    }
    let mut server = TlsServerSession::new(
        &seed, &[0xe2u8; 32], &[0xe3u8; 32], "aether-trust-test", b"\x08",
    );

    // Build client with that pubkey as the lone trust anchor.
    let mut client = TlsClientSession::new_full(
        [0xe4u8; 32], [0xe5u8; 32], (0..8u8).collect(),
        Vec::new(),                     // no ALPN
        vec![server_pub],               // trust anchor
    );

    server.feed(&client.take_outbound()).unwrap();
    let to_client = server.take_outbound();
    // First, sanity-check our self-sig verifier on the server's own cert.
    // We extract the cert DER directly from x509::self_sign_ed25519.
    let (cert_der, _) = aether_rt::tls13::x509::self_sign_ed25519(&seed, "aether-trust-test", b"\x08");
    assert!(verify_self_signed_ed25519_cert(&cert_der),
        "the cert builder must emit verifiable self-signatures");

    client.feed(&to_client).expect("client must accept server with valid trust anchor");
    server.feed(&client.take_outbound()).unwrap();
    assert!(client.is_handshake_done());
}

#[test]
fn tls13_trust_anchor_negative() {
    use aether_rt::tls13::TlsServerSession;
    let mut server = TlsServerSession::new(
        &[0xf1u8; 32], &[0xf2u8; 32], &[0xf3u8; 32], "aether-bad-trust", b"\x09",
    );
    // Client configured with a completely different anchor.
    let mut client = TlsClientSession::new_full(
        [0xf4u8; 32], [0xf5u8; 32], (0..8u8).collect(),
        Vec::new(),
        vec![[0xdeu8; 32]],  // bogus trust anchor
    );
    server.feed(&client.take_outbound()).unwrap();
    let to_client = server.take_outbound();
    let result = client.feed(&to_client);
    assert!(result.is_err(),
        "client must reject server when SPKI doesn't match any trust anchor; got {:?}", result);
}

/// ALPN where there's no overlap -> server emits no ALPN extension, both sides
/// negotiated_alpn() = None.
#[test]
fn tls13_alpn_no_overlap() {
    use aether_rt::tls13::TlsServerSession;
    let mut server = TlsServerSession::new_with_alpn(
        &[0xd1u8; 32], &[0xd2u8; 32], &[0xd3u8; 32],
        "aether-alpn-none", b"\x07",
        vec![b"h2".to_vec()],
    );
    let mut client = TlsClientSession::new_with_alpn(
        [0xd4u8; 32], [0xd5u8; 32], (0..8u8).collect(),
        vec![b"smtp".to_vec()],
    );
    server.feed(&client.take_outbound()).unwrap();
    client.feed(&server.take_outbound()).unwrap();
    server.feed(&client.take_outbound()).unwrap();
    assert_eq!(server.negotiated_alpn(), None);
    assert_eq!(client.negotiated_alpn(), None);
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
