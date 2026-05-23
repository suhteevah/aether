//! TLS 1.3 server (RFC 8446) on top of Aether's runtime crypto.
//!
//! Profile (minimum viable):
//!   - Cipher suite: TLS_CHACHA20_POLY1305_SHA256 (0x13_03)
//!   - Key exchange: X25519 (group 0x001d) only
//!   - Signature alg: ed25519 (0x0807) only
//!   - Server-side only, self-signed Ed25519 cert
//!   - No 0-RTT, no PSK, no resumption, no ALPN, no early data
//!   - No HelloRetryRequest; ClientHello must include an X25519 key_share
//!     directly.
//!
//! Architecture:
//!   - `record::*`           — TLSPlaintext + TLSCiphertext codec, per-record
//!                             nonce derivation
//!   - `key_schedule::Sched` — Early/Handshake/Master secret derivation +
//!                             traffic keys per §7.1
//!   - `handshake::*`        — encode/decode for ClientHello, ServerHello,
//!                             EncryptedExtensions, Certificate,
//!                             CertificateVerify, Finished
//!   - `x509`                — minimal self-signed Ed25519 cert generator
//!   - `TlsServerSession`    — state machine driver (transport-agnostic)
//!   - FFI surface           — `aether_tls13_*` for `.aether` callers

use crate::{
    aead_chacha20_poly1305_open, aead_chacha20_poly1305_seal,
    hkdf_extract, hmac_sha256, sha256, x25519_scalar_mult,
};
use std::os::raw::{c_int, c_void};

// ============================================================================
// Constants
// ============================================================================

pub const CIPHER_SUITE_CHACHA20_POLY1305_SHA256: u16 = 0x1303;
pub const NAMED_GROUP_X25519: u16 = 0x001d;
pub const SIG_ALG_ED25519: u16 = 0x0807;

pub const TLS_VERSION_1_3: u16 = 0x0304;
pub const TLS_VERSION_1_2_LEGACY: u16 = 0x0303;

pub const HS_CLIENT_HELLO: u8 = 1;
pub const HS_SERVER_HELLO: u8 = 2;
pub const HS_ENCRYPTED_EXTENSIONS: u8 = 8;
pub const HS_CERTIFICATE: u8 = 11;
pub const HS_CERTIFICATE_VERIFY: u8 = 15;
pub const HS_FINISHED: u8 = 20;

pub const REC_HANDSHAKE: u8 = 0x16;
pub const REC_APPLICATION_DATA: u8 = 0x17;
pub const REC_ALERT: u8 = 0x15;
pub const REC_CHANGE_CIPHER_SPEC: u8 = 0x14;

const HASH_LEN: usize = 32; // SHA-256
const KEY_LEN: usize = 32;
const IV_LEN: usize = 12;

// ============================================================================
// HKDF-Expand-Label (RFC 8446 §7.1) — Rust-internal
// ============================================================================

pub(crate) fn hkdf_expand_label(secret: &[u8; 32], label: &[u8], context: &[u8], length: usize) -> Vec<u8> {
    let mut full_label = Vec::with_capacity(6 + label.len());
    full_label.extend_from_slice(b"tls13 ");
    full_label.extend_from_slice(label);
    assert!(full_label.len() <= 255);
    assert!(context.len() <= 255);
    assert!(length <= 0xffff);
    let mut info = Vec::with_capacity(2 + 1 + full_label.len() + 1 + context.len());
    info.extend_from_slice(&(length as u16).to_be_bytes());
    info.push(full_label.len() as u8);
    info.extend_from_slice(&full_label);
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    crate::hkdf_expand(secret, &info, length)
}

/// Derive-Secret(Secret, Label, Messages) = HKDF-Expand-Label(Secret, Label, Hash(Messages), Hash.length)
pub(crate) fn derive_secret(secret: &[u8; 32], label: &[u8], transcript: &[u8]) -> [u8; 32] {
    let th = sha256(transcript);
    let v = hkdf_expand_label(secret, label, &th, HASH_LEN);
    let mut out = [0u8; 32]; out.copy_from_slice(&v); out
}

// ============================================================================
// Record layer
// ============================================================================

pub mod record {
    use super::*;

    /// TLSPlaintext / TLSCiphertext header (5 bytes).
    /// type(1) || legacy_record_version(2) || length(2)
    pub fn write_record_header(out: &mut Vec<u8>, opaque_type: u8, len: usize) {
        out.push(opaque_type);
        out.extend_from_slice(&TLS_VERSION_1_2_LEGACY.to_be_bytes());
        out.extend_from_slice(&(len as u16).to_be_bytes());
    }

    /// Wrap a plaintext fragment in a TLSPlaintext record.
    pub fn write_plaintext_record(out: &mut Vec<u8>, opaque_type: u8, fragment: &[u8]) {
        assert!(fragment.len() <= 0x4000);
        write_record_header(out, opaque_type, fragment.len());
        out.extend_from_slice(fragment);
    }

    /// Build the per-record nonce: 12-byte IV XOR seq (right-aligned big-endian).
    pub fn build_nonce(iv: &[u8; IV_LEN], seq: u64) -> [u8; IV_LEN] {
        let mut n = *iv;
        let s = seq.to_be_bytes();
        for i in 0..8 { n[IV_LEN - 8 + i] ^= s[i]; }
        n
    }

    /// AAD for an encrypted record: opaque_type(0x17) || 0x0303 || length(2-byte BE)
    /// where length = inner_plaintext_len + 1 (inner_type byte) + 16 (tag).
    pub fn build_aad(ct_total_len: usize) -> [u8; 5] {
        let mut a = [0u8; 5];
        a[0] = REC_APPLICATION_DATA;
        a[1] = (TLS_VERSION_1_2_LEGACY >> 8) as u8;
        a[2] = (TLS_VERSION_1_2_LEGACY & 0xff) as u8;
        a[3] = (ct_total_len >> 8) as u8;
        a[4] = (ct_total_len & 0xff) as u8;
        a
    }

    /// Seal an inner plaintext under the given key+IV+seq, wrapped as a TLSCiphertext.
    /// `inner_type` is the real content type (usually REC_HANDSHAKE or REC_APPLICATION_DATA).
    /// Appended to plaintext per §5.2 (no padding for now).
    /// Output is the full TLSCiphertext record (header + encrypted body).
    pub fn seal_record(
        out: &mut Vec<u8>,
        key: &[u8; KEY_LEN], iv: &[u8; IV_LEN], seq: u64,
        inner_type: u8, inner_plaintext: &[u8],
    ) {
        let mut inner = Vec::with_capacity(inner_plaintext.len() + 1);
        inner.extend_from_slice(inner_plaintext);
        inner.push(inner_type);
        let ct_total_len = inner.len() + 16;
        let aad = build_aad(ct_total_len);
        let nonce = build_nonce(iv, seq);
        let ct_and_tag = aead_chacha20_poly1305_seal(key, &nonce, &aad, &inner);
        write_record_header(out, REC_APPLICATION_DATA, ct_total_len);
        out.extend_from_slice(&ct_and_tag);
    }

    /// Open a single TLSCiphertext record (input starts at the 5-byte header).
    /// Returns (inner_type, inner_plaintext, bytes_consumed) on success.
    pub fn open_record(
        input: &[u8],
        key: &[u8; KEY_LEN], iv: &[u8; IV_LEN], seq: u64,
    ) -> Option<(u8, Vec<u8>, usize)> {
        if input.len() < 5 { return None; }
        let opaque_type = input[0];
        if opaque_type != REC_APPLICATION_DATA { return None; }
        let len = u16::from_be_bytes([input[3], input[4]]) as usize;
        if input.len() < 5 + len { return None; }
        let ct_body = &input[5..5 + len];
        let aad = build_aad(len);
        let nonce = build_nonce(iv, seq);
        let mut plain = aead_chacha20_poly1305_open(key, &nonce, &aad, ct_body)?;
        // Strip trailing zero padding then peel off the inner_type byte.
        while plain.last() == Some(&0) { plain.pop(); }
        let inner_type = plain.pop()?;
        Some((inner_type, plain, 5 + len))
    }

    /// Parse a single TLSPlaintext record header (cleartext path: ClientHello).
    /// Returns (opaque_type, body, bytes_consumed).
    pub fn parse_plaintext(input: &[u8]) -> Option<(u8, &[u8], usize)> {
        if input.len() < 5 { return None; }
        let opaque_type = input[0];
        let len = u16::from_be_bytes([input[3], input[4]]) as usize;
        if input.len() < 5 + len { return None; }
        Some((opaque_type, &input[5..5 + len], 5 + len))
    }
}

// ============================================================================
// Key schedule (RFC 8446 §7.1)
// ============================================================================

pub mod key_schedule {
    use super::*;

    pub struct Schedule {
        pub early_secret: [u8; 32],
        pub handshake_secret: [u8; 32],
        pub master_secret: [u8; 32],
        pub client_hs_traffic_secret: [u8; 32],
        pub server_hs_traffic_secret: [u8; 32],
        pub client_app_traffic_secret: [u8; 32],
        pub server_app_traffic_secret: [u8; 32],
    }

    pub struct DirKeys {
        pub key: [u8; KEY_LEN],
        pub iv: [u8; IV_LEN],
    }

    impl Schedule {
        /// Build the schedule up through the handshake-traffic-secrets.
        /// `transcript_ch_sh` = ClientHello bytes || ServerHello bytes (handshake-msg bodies as written on wire).
        pub fn handshake_phase(
            dhe_shared: &[u8; 32],
            transcript_ch_sh: &[u8],
        ) -> Schedule {
            // Early Secret = HKDF-Extract(salt=0, IKM=PSK=0).
            let zero32 = [0u8; 32];
            let es_v = hkdf_extract(&[0u8; 32], &zero32);
            let mut early_secret = [0u8; 32]; early_secret.copy_from_slice(&es_v);

            // Derive-Secret(es, "derived", "") -> salt for handshake secret.
            let derived_es = derive_secret(&early_secret, b"derived", b"");

            // Handshake Secret = HKDF-Extract(salt=derived_es, IKM=DHE).
            let hs_v = hkdf_extract(&derived_es, dhe_shared);
            let mut handshake_secret = [0u8; 32]; handshake_secret.copy_from_slice(&hs_v);

            // Per-direction handshake traffic secrets.
            let chs = derive_secret(&handshake_secret, b"c hs traffic", transcript_ch_sh);
            let shs = derive_secret(&handshake_secret, b"s hs traffic", transcript_ch_sh);

            // Master secret derivation uses Derive-Secret(handshake_secret, "derived", "") as salt.
            let derived_hs = derive_secret(&handshake_secret, b"derived", b"");
            let ms_v = hkdf_extract(&derived_hs, &zero32);
            let mut master_secret = [0u8; 32]; master_secret.copy_from_slice(&ms_v);

            // App traffic secrets filled in later (need transcript through server Finished).
            Schedule {
                early_secret,
                handshake_secret,
                master_secret,
                client_hs_traffic_secret: chs,
                server_hs_traffic_secret: shs,
                client_app_traffic_secret: [0u8; 32],
                server_app_traffic_secret: [0u8; 32],
            }
        }

        /// Fill in app traffic secrets given the transcript through the server Finished.
        pub fn finalize_app_secrets(&mut self, transcript_through_server_fin: &[u8]) {
            self.client_app_traffic_secret =
                derive_secret(&self.master_secret, b"c ap traffic", transcript_through_server_fin);
            self.server_app_traffic_secret =
                derive_secret(&self.master_secret, b"s ap traffic", transcript_through_server_fin);
        }
    }

    /// Derive (key, iv) from a traffic secret.
    pub fn traffic_keys(secret: &[u8; 32]) -> DirKeys {
        let k_v = hkdf_expand_label(secret, b"key", &[], KEY_LEN);
        let i_v = hkdf_expand_label(secret, b"iv", &[], IV_LEN);
        let mut k = [0u8; KEY_LEN]; k.copy_from_slice(&k_v);
        let mut i = [0u8; IV_LEN]; i.copy_from_slice(&i_v);
        DirKeys { key: k, iv: i }
    }

    /// Compute the Finished MAC: HMAC(finished_key, Transcript-Hash(...)).
    /// finished_key = HKDF-Expand-Label(BaseKey, "finished", "", Hash.length).
    pub fn finished_mac(base_key: &[u8; 32], transcript: &[u8]) -> [u8; 32] {
        let fin_key_v = hkdf_expand_label(base_key, b"finished", &[], HASH_LEN);
        let th = sha256(transcript);
        hmac_sha256(&fin_key_v, &th)
    }
}

// ============================================================================
// Handshake messages
// ============================================================================

pub mod handshake {
    use super::*;

    /// Wrap a handshake-message body in the 4-byte header (msg_type || u24 length).
    pub fn wrap(msg_type: u8, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + body.len());
        out.push(msg_type);
        let len = body.len();
        out.push((len >> 16) as u8);
        out.push((len >> 8) as u8);
        out.push(len as u8);
        out.extend_from_slice(body);
        out
    }

    /// Parsed ClientHello — only fields we need.
    pub struct ParsedClientHello {
        pub random: [u8; 32],
        pub legacy_session_id: Vec<u8>,
        pub x25519_key_share: [u8; 32],
        pub offers_chacha20_poly1305_sha256: bool,
        pub offers_tls13: bool,
        pub offers_x25519: bool,
        pub offers_ed25519: bool,
        /// ALPN (RFC 7301) protocol names the client advertises, in order.
        pub alpn_offered: Vec<Vec<u8>>,
    }

    /// Parse a ClientHello message body (after the 4-byte handshake header has been stripped).
    pub fn parse_client_hello(body: &[u8]) -> Option<ParsedClientHello> {
        let mut r = Reader::new(body);
        let legacy_version = r.read_u16()?;
        if legacy_version != TLS_VERSION_1_2_LEGACY { return None; }
        let mut random = [0u8; 32];
        random.copy_from_slice(r.read_bytes(32)?);
        let sid_len = r.read_u8()? as usize;
        let legacy_session_id = r.read_bytes(sid_len)?.to_vec();
        let cs_len = r.read_u16()? as usize;
        let cs_body = r.read_bytes(cs_len)?;
        let mut offers_chacha20 = false;
        let mut i = 0;
        while i + 2 <= cs_body.len() {
            let cs = u16::from_be_bytes([cs_body[i], cs_body[i + 1]]);
            if cs == CIPHER_SUITE_CHACHA20_POLY1305_SHA256 { offers_chacha20 = true; }
            i += 2;
        }
        let comp_len = r.read_u8()? as usize;
        let _comp_body = r.read_bytes(comp_len)?;
        let ext_total_len = r.read_u16()? as usize;
        let ext_body = r.read_bytes(ext_total_len)?;

        let mut offers_tls13 = false;
        let mut offers_x25519 = false;
        let mut offers_ed25519 = false;
        let mut x25519_key_share = [0u8; 32];
        let mut found_key_share = false;
        let mut alpn_offered: Vec<Vec<u8>> = Vec::new();

        let mut er = Reader::new(ext_body);
        while er.remaining() > 0 {
            let ext_type = er.read_u16()?;
            let ext_len = er.read_u16()? as usize;
            let ext_data = er.read_bytes(ext_len)?;
            match ext_type {
                0x002b => {
                    // supported_versions: 1-byte length || list of u16 versions
                    if ext_data.is_empty() { return None; }
                    let vlen = ext_data[0] as usize;
                    let mut j = 1;
                    while j + 2 <= 1 + vlen && j + 2 <= ext_data.len() {
                        let v = u16::from_be_bytes([ext_data[j], ext_data[j + 1]]);
                        if v == TLS_VERSION_1_3 { offers_tls13 = true; }
                        j += 2;
                    }
                }
                0x000a => {
                    // supported_groups: u16 length || list of u16 groups
                    if ext_data.len() < 2 { return None; }
                    let glen = u16::from_be_bytes([ext_data[0], ext_data[1]]) as usize;
                    let mut j = 2;
                    while j + 2 <= 2 + glen && j + 2 <= ext_data.len() {
                        let g = u16::from_be_bytes([ext_data[j], ext_data[j + 1]]);
                        if g == NAMED_GROUP_X25519 { offers_x25519 = true; }
                        j += 2;
                    }
                }
                0x000d => {
                    // signature_algorithms: u16 length || list of u16 algs
                    if ext_data.len() < 2 { return None; }
                    let slen = u16::from_be_bytes([ext_data[0], ext_data[1]]) as usize;
                    let mut j = 2;
                    while j + 2 <= 2 + slen && j + 2 <= ext_data.len() {
                        let sa = u16::from_be_bytes([ext_data[j], ext_data[j + 1]]);
                        if sa == SIG_ALG_ED25519 { offers_ed25519 = true; }
                        j += 2;
                    }
                }
                0x0033 => {
                    // key_share: u16 total length || list of KeyShareEntry {group(u16), data(u16-len)}
                    if ext_data.len() < 2 { return None; }
                    let total = u16::from_be_bytes([ext_data[0], ext_data[1]]) as usize;
                    let mut kr = Reader::new(&ext_data[2..2 + total]);
                    while kr.remaining() > 0 {
                        let g = kr.read_u16()?;
                        let dlen = kr.read_u16()? as usize;
                        let d = kr.read_bytes(dlen)?;
                        if g == NAMED_GROUP_X25519 && d.len() == 32 {
                            x25519_key_share.copy_from_slice(d);
                            found_key_share = true;
                        }
                    }
                }
                0x0010 => {
                    // ALPN (RFC 7301): u16 list_len || ProtocolName<1..255>+ where each = u8 len || bytes
                    if ext_data.len() < 2 { return None; }
                    let list_len = u16::from_be_bytes([ext_data[0], ext_data[1]]) as usize;
                    let mut pr = Reader::new(&ext_data[2..2 + list_len]);
                    while pr.remaining() > 0 {
                        let nlen = pr.read_u8()? as usize;
                        let name = pr.read_bytes(nlen)?;
                        alpn_offered.push(name.to_vec());
                    }
                }
                _ => {}
            }
        }
        if !found_key_share { return None; }
        Some(ParsedClientHello {
            random,
            legacy_session_id,
            x25519_key_share,
            offers_chacha20_poly1305_sha256: offers_chacha20,
            offers_tls13,
            offers_x25519,
            offers_ed25519,
            alpn_offered,
        })
    }

    /// Encode a ServerHello body.
    /// Returns the body (without the 4-byte handshake header).
    pub fn encode_server_hello(
        random: &[u8; 32],
        legacy_session_id: &[u8],
        server_x25519_pub: &[u8; 32],
    ) -> Vec<u8> {
        let mut body = Vec::with_capacity(128);
        body.extend_from_slice(&TLS_VERSION_1_2_LEGACY.to_be_bytes());
        body.extend_from_slice(random);
        // legacy_session_id_echo: 1-byte length + body (echo CH's value).
        body.push(legacy_session_id.len() as u8);
        body.extend_from_slice(legacy_session_id);
        body.extend_from_slice(&CIPHER_SUITE_CHACHA20_POLY1305_SHA256.to_be_bytes());
        body.push(0); // legacy_compression_method = null

        // Extensions: supported_versions(tls 1.3) + key_share(x25519, server pub).
        let mut exts = Vec::with_capacity(64);
        // supported_versions (server-side): 2-byte type, 2-byte length, body=u16 version.
        exts.extend_from_slice(&0x002bu16.to_be_bytes());
        exts.extend_from_slice(&2u16.to_be_bytes());
        exts.extend_from_slice(&TLS_VERSION_1_3.to_be_bytes());
        // key_share (server-side): single KeyShareEntry.
        let ks_entry_len = 2 + 2 + 32; // group + len + 32-byte pub
        exts.extend_from_slice(&0x0033u16.to_be_bytes());
        exts.extend_from_slice(&(ks_entry_len as u16).to_be_bytes());
        exts.extend_from_slice(&NAMED_GROUP_X25519.to_be_bytes());
        exts.extend_from_slice(&32u16.to_be_bytes());
        exts.extend_from_slice(server_x25519_pub);

        body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
        body.extend_from_slice(&exts);
        body
    }

    /// EncryptedExtensions body.  In TLS 1.3 ALPN moves from ServerHello to
    /// EncryptedExtensions (RFC 8446 §4.4 + RFC 7301).  If `alpn_chosen`
    /// is Some, emit an ALPN extension with exactly that one protocol.
    pub fn encode_encrypted_extensions(alpn_chosen: Option<&[u8]>) -> Vec<u8> {
        let mut exts = Vec::new();
        if let Some(proto) = alpn_chosen {
            // Inner ALPN: u16 list_len || u8 proto_len || proto_bytes
            let inner_len = 1 + proto.len();
            // Outer: u16 type=0x0010 || u16 ext_len || (u16 inner_list_len || u8 plen || proto)
            exts.extend_from_slice(&0x0010u16.to_be_bytes());
            exts.extend_from_slice(&((2 + inner_len) as u16).to_be_bytes());
            exts.extend_from_slice(&(inner_len as u16).to_be_bytes());
            exts.push(proto.len() as u8);
            exts.extend_from_slice(proto);
        }
        let mut body = Vec::with_capacity(2 + exts.len());
        body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
        body.extend_from_slice(&exts);
        body
    }

    /// Certificate body — single CertificateEntry with `cert_der`, no extensions.
    /// Format: certificate_request_context<0..2^8-1> || certificate_list<0..2^24-1>
    /// certificate_list entry = cert_data<1..2^24-1> || extensions<0..2^16-1>
    pub fn encode_certificate(cert_der: &[u8]) -> Vec<u8> {
        let mut body = Vec::with_capacity(8 + cert_der.len());
        body.push(0); // certificate_request_context length 0
        // u24 length for the certificate_list:
        let entry_len = 3 + cert_der.len() + 2; // cert_len(u24) + cert_data + ext_len(u16,=0)
        body.push((entry_len >> 16) as u8);
        body.push((entry_len >> 8) as u8);
        body.push(entry_len as u8);
        // cert_data length (u24)
        body.push((cert_der.len() >> 16) as u8);
        body.push((cert_der.len() >> 8) as u8);
        body.push(cert_der.len() as u8);
        body.extend_from_slice(cert_der);
        // extensions length = 0 (u16)
        body.extend_from_slice(&[0, 0]);
        body
    }

    /// CertificateVerify body for SignatureScheme + 64-byte signature.
    pub fn encode_certificate_verify(sig_alg: u16, signature: &[u8]) -> Vec<u8> {
        let mut body = Vec::with_capacity(4 + signature.len());
        body.extend_from_slice(&sig_alg.to_be_bytes());
        body.extend_from_slice(&(signature.len() as u16).to_be_bytes());
        body.extend_from_slice(signature);
        body
    }

    /// Finished body = 32-byte verify_data (HMAC).
    pub fn encode_finished(verify_data: &[u8; 32]) -> Vec<u8> {
        verify_data.to_vec()
    }

    /// Construct the bytes signed by the server in CertificateVerify per §4.4.3:
    ///   64 spaces || "TLS 1.3, server CertificateVerify" || 0x00 || Transcript-Hash(handshake_messages)
    pub fn server_cert_verify_signed_content(transcript: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + 33 + 1 + 32);
        out.extend_from_slice(&[0x20u8; 64]);
        out.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        out.push(0x00);
        out.extend_from_slice(&sha256(transcript));
        out
    }

    /// Tiny zero-copy reader used while parsing.
    pub struct Reader<'a> { buf: &'a [u8], pos: usize }
    impl<'a> Reader<'a> {
        pub fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }
        pub fn remaining(&self) -> usize { self.buf.len() - self.pos }
        pub fn read_u8(&mut self) -> Option<u8> {
            if self.remaining() < 1 { return None; }
            let v = self.buf[self.pos]; self.pos += 1; Some(v)
        }
        pub fn read_u16(&mut self) -> Option<u16> {
            if self.remaining() < 2 { return None; }
            let v = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
            self.pos += 2; Some(v)
        }
        pub fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
            if self.remaining() < n { return None; }
            let s = &self.buf[self.pos..self.pos + n];
            self.pos += n;
            Some(s)
        }
    }
}

// ============================================================================
// Minimal X.509 self-signed Ed25519 cert (DER)
// ============================================================================

pub mod x509 {
    use super::*;

    fn der_len_encode(out: &mut Vec<u8>, len: usize) {
        if len < 0x80 {
            out.push(len as u8);
        } else if len < 0x100 {
            out.push(0x81);
            out.push(len as u8);
        } else if len < 0x10000 {
            out.push(0x82);
            out.push((len >> 8) as u8);
            out.push(len as u8);
        } else {
            out.push(0x83);
            out.push((len >> 16) as u8);
            out.push((len >> 8) as u8);
            out.push(len as u8);
        }
    }

    /// Wrap a body in TAG || LENGTH || body.
    pub fn der_tag(tag: u8, body: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + body.len());
        out.push(tag);
        der_len_encode(&mut out, body.len());
        out.extend_from_slice(body);
        out
    }

    // Common Name "aether-tls13" — Name = SEQUENCE { SET { SEQUENCE { OID(CN), UTF8String("...") } } }
    fn name_seq(cn: &str) -> Vec<u8> {
        // OID 2.5.4.3 (commonName) = 06 03 55 04 03
        let oid_cn = [0x06u8, 0x03, 0x55, 0x04, 0x03];
        let cn_utf8 = der_tag(0x0c, cn.as_bytes());
        let mut attr_body = Vec::new();
        attr_body.extend_from_slice(&oid_cn);
        attr_body.extend_from_slice(&cn_utf8);
        let attr = der_tag(0x30, &attr_body);             // SEQUENCE
        let set = der_tag(0x31, &attr);                    // SET
        der_tag(0x30, &set)                                // outer SEQUENCE (RDNSequence)
    }

    // AlgorithmIdentifier { OID 1.3.101.112 (Ed25519) } — RFC 8410.
    // Encoded as: SEQUENCE { OBJECT IDENTIFIER 1.3.101.112 }
    fn ed25519_alg_id() -> Vec<u8> {
        // OID 1.3.101.112 = 06 03 2b 65 70
        let oid = [0x06u8, 0x03, 0x2b, 0x65, 0x70];
        der_tag(0x30, &oid)
    }

    // SubjectPublicKeyInfo = SEQUENCE { algorithm, BIT STRING { 0x00 || public_key } }
    fn ed25519_spki(pub_key: &[u8; 32]) -> Vec<u8> {
        let alg = ed25519_alg_id();
        let mut bit_str_body = Vec::with_capacity(33);
        bit_str_body.push(0x00); // 0 unused bits
        bit_str_body.extend_from_slice(pub_key);
        let bit_str = der_tag(0x03, &bit_str_body);
        let mut body = Vec::new();
        body.extend_from_slice(&alg);
        body.extend_from_slice(&bit_str);
        der_tag(0x30, &body)
    }

    // Validity = SEQUENCE { UTCTime "240101000000Z", UTCTime "490101000000Z" }
    // Fixed window 2024-01-01 to 2049-01-01.
    fn validity_fixed() -> Vec<u8> {
        let nb = der_tag(0x17, b"240101000000Z");
        let na = der_tag(0x17, b"490101000000Z");
        let mut body = Vec::new();
        body.extend_from_slice(&nb);
        body.extend_from_slice(&na);
        der_tag(0x30, &body)
    }

    /// Build TBSCertificate body (everything inside the outer Certificate SEQUENCE
    /// before signatureAlgorithm + signatureValue).
    fn tbs_certificate(serial: &[u8], pub_key: &[u8; 32], cn: &str) -> Vec<u8> {
        // version [0] EXPLICIT INTEGER (2 means v3)
        let version_int = der_tag(0x02, &[0x02]);
        let version = der_tag(0xa0, &version_int);
        // serial INTEGER
        let mut serial_body = Vec::with_capacity(serial.len() + 1);
        // DER INTEGERs are signed; if high bit of first byte is 1, prepend 0x00.
        if serial.first().map_or(false, |&b| b & 0x80 != 0) {
            serial_body.push(0x00);
        }
        serial_body.extend_from_slice(serial);
        let serial_der = der_tag(0x02, &serial_body);
        let sig_alg = ed25519_alg_id();
        let issuer = name_seq(cn);
        let validity = validity_fixed();
        let subject = name_seq(cn);
        let spki = ed25519_spki(pub_key);

        let mut body = Vec::new();
        body.extend_from_slice(&version);
        body.extend_from_slice(&serial_der);
        body.extend_from_slice(&sig_alg);
        body.extend_from_slice(&issuer);
        body.extend_from_slice(&validity);
        body.extend_from_slice(&subject);
        body.extend_from_slice(&spki);
        der_tag(0x30, &body)
    }

    /// Generate a self-signed Ed25519 X.509 cert.
    /// `seed`: 32-byte Ed25519 private key seed; `cn`: common-name string.
    /// Returns the cert DER and the 32-byte public key.
    pub fn self_sign_ed25519(seed: &[u8; 32], cn: &str, serial: &[u8]) -> (Vec<u8>, [u8; 32]) {
        let mut pub_key = [0u8; 32];
        unsafe {
            let n = crate::aether_ed25519_derive_public(
                seed.as_ptr() as *const c_void,
                pub_key.as_mut_ptr() as *mut c_void,
            );
            assert_eq!(n, 32);
        }
        let tbs = tbs_certificate(serial, &pub_key, cn);
        // Sign TBS with Ed25519.
        let mut sig = [0u8; 64];
        unsafe {
            let n = crate::aether_ed25519_sign(
                seed.as_ptr() as *const c_void,
                pub_key.as_ptr() as *const c_void,
                tbs.as_ptr() as *const c_void,
                tbs.len() as c_int,
                sig.as_mut_ptr() as *mut c_void,
            );
            assert_eq!(n, 64);
        }
        let sig_alg = ed25519_alg_id();
        let mut sig_body = Vec::with_capacity(65);
        sig_body.push(0x00); // unused-bits = 0
        sig_body.extend_from_slice(&sig);
        let sig_bit_str = der_tag(0x03, &sig_body);
        let mut outer_body = Vec::new();
        outer_body.extend_from_slice(&tbs);
        outer_body.extend_from_slice(&sig_alg);
        outer_body.extend_from_slice(&sig_bit_str);
        let cert = der_tag(0x30, &outer_body);
        (cert, pub_key)
    }
}

// ============================================================================
// Server state machine
// ============================================================================

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum State {
    ExpectClientHello,
    SentServerFlight,
    Connected,
    Closed,
}

pub struct TlsServerSession {
    state: State,
    transcript: Vec<u8>,
    sched: Option<key_schedule::Schedule>,
    server_seed: [u8; 32],
    server_cert_pub: [u8; 32],
    server_cert_der: Vec<u8>,
    server_random: [u8; 32],
    server_x25519_priv: [u8; 32],
    server_x25519_pub: [u8; 32],
    /// ALPN protocols this server accepts, in preference order.
    /// Empty = no ALPN support; server won't emit the extension.
    supported_alpn: Vec<Vec<u8>>,
    /// Negotiated ALPN protocol (filled in after ClientHello).
    alpn_chosen: Option<Vec<u8>>,

    client_hs_keys: Option<key_schedule::DirKeys>,
    server_hs_keys: Option<key_schedule::DirKeys>,
    client_app_keys: Option<key_schedule::DirKeys>,
    server_app_keys: Option<key_schedule::DirKeys>,

    client_hs_seq: u64,
    server_hs_seq: u64,
    client_app_seq: u64,
    server_app_seq: u64,

    transcript_at_server_fin: Vec<u8>,
    server_finished_verify_data: [u8; 32],

    /// Bytes queued to send to the peer.
    out_buf: Vec<u8>,
    /// Inbound bytes received but not yet consumed as a complete record.
    /// Lets `feed()` tolerate streaming TCP that may deliver partial records.
    pending: Vec<u8>,
}

impl TlsServerSession {
    /// Build a session with the given Ed25519 seed for cert signing and a
    /// deterministic-or-random X25519 ephemeral key + server random.
    /// In tests we pass deterministic seeds; in production these come from RNG.
    pub fn new(
        ed25519_seed: &[u8; 32],
        server_random: &[u8; 32],
        x25519_priv: &[u8; 32],
        cn: &str,
        serial: &[u8],
    ) -> Self {
        Self::new_with_alpn(ed25519_seed, server_random, x25519_priv, cn, serial, Vec::new())
    }

    /// Construct a server session that will negotiate ALPN against the given
    /// `supported_alpn` list (server's preference order — picks the first
    /// match from `alpn_offered` that the server supports).
    pub fn new_with_alpn(
        ed25519_seed: &[u8; 32],
        server_random: &[u8; 32],
        x25519_priv: &[u8; 32],
        cn: &str,
        serial: &[u8],
        supported_alpn: Vec<Vec<u8>>,
    ) -> Self {
        let (cert_der, cert_pub) = x509::self_sign_ed25519(ed25519_seed, cn, serial);
        let mut bp = [0u8; 32]; bp[0] = 9;
        let server_x25519_pub = x25519_scalar_mult(x25519_priv, &bp);

        Self {
            state: State::ExpectClientHello,
            transcript: Vec::new(),
            sched: None,
            server_seed: *ed25519_seed,
            server_cert_pub: cert_pub,
            server_cert_der: cert_der,
            server_random: *server_random,
            server_x25519_priv: *x25519_priv,
            server_x25519_pub,
            supported_alpn,
            alpn_chosen: None,
            client_hs_keys: None,
            server_hs_keys: None,
            client_app_keys: None,
            server_app_keys: None,
            client_hs_seq: 0,
            server_hs_seq: 0,
            client_app_seq: 0,
            server_app_seq: 0,
            transcript_at_server_fin: Vec::new(),
            server_finished_verify_data: [0u8; 32],
            out_buf: Vec::new(),
            pending: Vec::new(),
        }
    }

    pub fn state(&self) -> State { self.state }
    pub fn is_handshake_done(&self) -> bool { self.state == State::Connected }

    /// The ALPN protocol the server picked, if any (set after ClientHello).
    pub fn negotiated_alpn(&self) -> Option<&[u8]> { self.alpn_chosen.as_deref() }

    /// Drain pending outbound bytes.
    pub fn take_outbound(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.out_buf)
    }

    /// Feed inbound TCP bytes.  Buffers partial records across calls.
    /// Returns Ok(decrypted_app_data) or Err(&str) on protocol error.
    pub fn feed(&mut self, input: &[u8]) -> Result<Vec<u8>, &'static str> {
        self.pending.extend_from_slice(input);
        let mut decrypted_app: Vec<u8> = Vec::new();
        loop {
            if self.pending.len() < 5 { break; } // need at least the record header
            let len = u16::from_be_bytes([self.pending[3], self.pending[4]]) as usize;
            if self.pending.len() < 5 + len { break; } // partial record body
            // Snapshot the full record bytes so we don't borrow self.pending across the body.
            let record_total = 5 + len;
            let record_bytes: Vec<u8> = self.pending[..record_total].to_vec();
            // Drain consumed bytes from pending up front.
            self.pending.drain(..record_total);
            match self.state {
                State::ExpectClientHello => {
                    let (rt, body, _consumed) = record::parse_plaintext(&record_bytes)
                        .ok_or("malformed ClientHello record")?;
                    if rt != REC_HANDSHAKE { return Err("expected handshake record"); }
                    self.handle_client_hello_record(body)?;
                }
                State::SentServerFlight => {
                    // Tolerate a CCS record (RFC 8446 §5 middlebox-compat).
                    if record_bytes[0] == REC_CHANGE_CIPHER_SPEC { continue; }
                    let keys = self.client_hs_keys.as_ref().ok_or("no client hs keys")?;
                    let seq = self.client_hs_seq;
                    let (inner_type, plain, _consumed) =
                        record::open_record(&record_bytes, &keys.key, &keys.iv, seq)
                            .ok_or("decryption failure on client Finished")?;
                    self.client_hs_seq += 1;
                    if inner_type != REC_HANDSHAKE { return Err("expected handshake inner type"); }
                    self.handle_client_finished(&plain)?;
                }
                State::Connected => {
                    if record_bytes[0] == REC_CHANGE_CIPHER_SPEC { continue; }
                    let keys = self.client_app_keys.as_ref().ok_or("no client app keys")?;
                    let seq = self.client_app_seq;
                    let (inner_type, plain, _consumed) =
                        record::open_record(&record_bytes, &keys.key, &keys.iv, seq)
                            .ok_or("decryption failure on app data")?;
                    self.client_app_seq += 1;
                    match inner_type {
                        REC_APPLICATION_DATA => decrypted_app.extend_from_slice(&plain),
                        REC_ALERT => { self.state = State::Closed; }
                        _ => {} // post-handshake messages (NewSessionTicket, KeyUpdate) ignored
                    }
                }
                State::Closed => return Err("session closed"),
            }
        }
        Ok(decrypted_app)
    }

    fn handle_client_hello_record(&mut self, hs_record_body: &[u8]) -> Result<(), &'static str> {
        // hs_record_body may contain one or more handshake messages concatenated.
        // For ClientHello we expect exactly one.
        if hs_record_body.len() < 4 { return Err("CH too short"); }
        let msg_type = hs_record_body[0];
        if msg_type != HS_CLIENT_HELLO { return Err("expected ClientHello"); }
        let body_len = (hs_record_body[1] as usize) << 16
                     | (hs_record_body[2] as usize) << 8
                     | (hs_record_body[3] as usize);
        if hs_record_body.len() != 4 + body_len { return Err("CH body length mismatch"); }
        let parsed = handshake::parse_client_hello(&hs_record_body[4..])
            .ok_or("malformed ClientHello")?;
        if !parsed.offers_tls13 { return Err("client doesn't support TLS 1.3"); }
        if !parsed.offers_chacha20_poly1305_sha256 {
            return Err("client doesn't support TLS_CHACHA20_POLY1305_SHA256");
        }
        if !parsed.offers_x25519 { return Err("client doesn't offer X25519"); }
        // ed25519 advertise is optional: some test clients may not include it; we still sign.

        // Add ClientHello to transcript.
        self.transcript.extend_from_slice(hs_record_body);

        // Build ServerHello.
        let sh_body = handshake::encode_server_hello(
            &self.server_random,
            &parsed.legacy_session_id,
            &self.server_x25519_pub,
        );
        let sh_full = handshake::wrap(HS_SERVER_HELLO, &sh_body);
        self.transcript.extend_from_slice(&sh_full);

        // Emit ServerHello (cleartext TLSPlaintext record).
        record::write_plaintext_record(&mut self.out_buf, REC_HANDSHAKE, &sh_full);

        // Compute DHE shared secret = scalar_mult(server_priv, client_pub).
        let dhe = x25519_scalar_mult(&self.server_x25519_priv, &parsed.x25519_key_share);

        // Build handshake-phase schedule from transcript (CH || SH).
        let sched = key_schedule::Schedule::handshake_phase(&dhe, &self.transcript);
        let server_hs_keys = key_schedule::traffic_keys(&sched.server_hs_traffic_secret);
        let client_hs_keys = key_schedule::traffic_keys(&sched.client_hs_traffic_secret);

        // ALPN negotiation: pick the first server-supported protocol that the
        // client also offered.  Order = server preference.
        self.alpn_chosen = None;
        for srv in &self.supported_alpn {
            if parsed.alpn_offered.iter().any(|c| c == srv) {
                self.alpn_chosen = Some(srv.clone());
                break;
            }
        }

        // Build the server's encrypted flight: EncryptedExtensions || Certificate || CertificateVerify || Finished.
        let ee_body = handshake::encode_encrypted_extensions(self.alpn_chosen.as_deref());
        let ee = handshake::wrap(HS_ENCRYPTED_EXTENSIONS, &ee_body);
        self.transcript.extend_from_slice(&ee);

        let cert_msg = handshake::wrap(HS_CERTIFICATE, &handshake::encode_certificate(&self.server_cert_der));
        self.transcript.extend_from_slice(&cert_msg);

        // CertificateVerify: sign Transcript-Hash(handshake_messages_so_far).
        let cv_signed_content = handshake::server_cert_verify_signed_content(&self.transcript);
        let mut sig = [0u8; 64];
        unsafe {
            let n = crate::aether_ed25519_sign(
                self.server_seed.as_ptr() as *const c_void,
                self.server_cert_pub.as_ptr() as *const c_void,
                cv_signed_content.as_ptr() as *const c_void,
                cv_signed_content.len() as c_int,
                sig.as_mut_ptr() as *mut c_void,
            );
            if n != 64 { return Err("ed25519 sign failed"); }
        }
        let cv = handshake::wrap(HS_CERTIFICATE_VERIFY,
            &handshake::encode_certificate_verify(SIG_ALG_ED25519, &sig));
        self.transcript.extend_from_slice(&cv);

        // Finished MAC over transcript so far.
        let server_fin_vd = key_schedule::finished_mac(&sched.server_hs_traffic_secret, &self.transcript);
        let fin_msg = handshake::wrap(HS_FINISHED, &handshake::encode_finished(&server_fin_vd));
        self.transcript.extend_from_slice(&fin_msg);
        self.server_finished_verify_data = server_fin_vd;

        // Concat the four inner handshake messages and emit as ONE encrypted record.
        // (Could split, but this is legal and simpler.)
        let mut inner = Vec::with_capacity(ee.len() + cert_msg.len() + cv.len() + fin_msg.len());
        inner.extend_from_slice(&ee);
        inner.extend_from_slice(&cert_msg);
        inner.extend_from_slice(&cv);
        inner.extend_from_slice(&fin_msg);
        record::seal_record(
            &mut self.out_buf,
            &server_hs_keys.key, &server_hs_keys.iv, self.server_hs_seq,
            REC_HANDSHAKE, &inner,
        );
        self.server_hs_seq += 1;

        // Capture transcript snapshot used for app-traffic-secret derivation.
        self.transcript_at_server_fin = self.transcript.clone();
        // Finalize app traffic secrets.
        let mut sched = sched;
        sched.finalize_app_secrets(&self.transcript_at_server_fin);
        let server_app_keys = key_schedule::traffic_keys(&sched.server_app_traffic_secret);
        let client_app_keys = key_schedule::traffic_keys(&sched.client_app_traffic_secret);

        self.client_hs_keys = Some(client_hs_keys);
        self.server_hs_keys = Some(server_hs_keys);
        self.client_app_keys = Some(client_app_keys);
        self.server_app_keys = Some(server_app_keys);
        self.sched = Some(sched);
        self.state = State::SentServerFlight;
        Ok(())
    }

    fn handle_client_finished(&mut self, hs_record_body: &[u8]) -> Result<(), &'static str> {
        if hs_record_body.len() < 4 { return Err("Finished too short"); }
        if hs_record_body[0] != HS_FINISHED { return Err("expected Finished"); }
        let body_len = (hs_record_body[1] as usize) << 16
                     | (hs_record_body[2] as usize) << 8
                     | (hs_record_body[3] as usize);
        if hs_record_body.len() != 4 + body_len { return Err("Finished body length mismatch"); }
        if body_len != HASH_LEN { return Err("Finished body must be 32 bytes"); }
        let sched = self.sched.as_ref().ok_or("no schedule")?;
        // Client Finished MAC was computed over transcript through server Finished.
        let expected = key_schedule::finished_mac(
            &sched.client_hs_traffic_secret,
            &self.transcript_at_server_fin,
        );
        let recv = &hs_record_body[4..4 + 32];
        let mut diff = 0u8;
        for i in 0..32 { diff |= expected[i] ^ recv[i]; }
        if diff != 0 { return Err("client Finished verify_data mismatch"); }
        // Add client Finished to transcript (for app-traffic-secret-1 in future updates).
        self.transcript.extend_from_slice(hs_record_body);
        self.state = State::Connected;
        Ok(())
    }

    /// Encrypt an application-data buffer and append to the outbound queue.
    pub fn send_app_data(&mut self, plaintext: &[u8]) -> Result<(), &'static str> {
        if self.state != State::Connected { return Err("not connected"); }
        let keys = self.server_app_keys.as_ref().ok_or("no server app keys")?;
        let seq = self.server_app_seq;
        record::seal_record(
            &mut self.out_buf, &keys.key, &keys.iv, seq,
            REC_APPLICATION_DATA, plaintext,
        );
        self.server_app_seq += 1;
        Ok(())
    }
}

// ============================================================================
// FFI surface
// ============================================================================

/// Build an HKDF-Expand-Label tls13-style output.  See aether_tls13_hkdf_expand_label
/// in lib.rs — this is a Rust-mirror so .aether programs can reach the same fn.
/// (lib.rs already exports the FFI symbol; this module's FFI surface adds
/// session-level entry points only.)

// Owned table of sessions for .aether code that doesn't want to manage raw pointers.
struct SessionTable {
    items: Vec<Option<Box<TlsServerSession>>>,
}

use std::cell::UnsafeCell;
struct SessionTableCell(UnsafeCell<SessionTable>);
unsafe impl Sync for SessionTableCell {}

static SESSIONS: SessionTableCell = SessionTableCell(UnsafeCell::new(SessionTable { items: Vec::new() }));

#[inline]
unsafe fn sessions() -> &'static mut SessionTable { &mut *SESSIONS.0.get() }

/// Create a TLS server session.  Returns an i64 handle (>=0) or -1.
/// `ed25519_seed_ptr`: 32 bytes; `server_random_ptr`: 32 bytes; `x25519_priv_ptr`: 32 bytes;
/// `cn_ptr`/`n_cn`: ASCII subject CN; `serial_ptr`/`n_serial`: cert serial.
#[no_mangle] pub unsafe extern "C" fn aether_tls13_server_new(
    ed25519_seed_ptr: *const c_void,
    server_random_ptr: *const c_void,
    x25519_priv_ptr: *const c_void,
    cn_ptr: *const c_void, n_cn: c_int,
    serial_ptr: *const c_void, n_serial: c_int,
) -> i64 {
    if ed25519_seed_ptr.is_null() || server_random_ptr.is_null() || x25519_priv_ptr.is_null() {
        return -1;
    }
    if cn_ptr.is_null() || serial_ptr.is_null() || n_cn < 0 || n_serial < 0 { return -1; }
    let seed = {
        let s = std::slice::from_raw_parts(ed25519_seed_ptr as *const u8, 32);
        let mut a = [0u8; 32]; a.copy_from_slice(s); a
    };
    let rand = {
        let s = std::slice::from_raw_parts(server_random_ptr as *const u8, 32);
        let mut a = [0u8; 32]; a.copy_from_slice(s); a
    };
    let priv_ = {
        let s = std::slice::from_raw_parts(x25519_priv_ptr as *const u8, 32);
        let mut a = [0u8; 32]; a.copy_from_slice(s); a
    };
    let cn_bytes = std::slice::from_raw_parts(cn_ptr as *const u8, n_cn as usize);
    let serial_bytes = std::slice::from_raw_parts(serial_ptr as *const u8, n_serial as usize);
    let cn_str = match std::str::from_utf8(cn_bytes) {
        Ok(s) => s, Err(_) => return -1,
    };
    let sess = TlsServerSession::new(&seed, &rand, &priv_, cn_str, serial_bytes);
    let t = sessions();
    for (i, slot) in t.items.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(sess)); return i as i64; }
    }
    t.items.push(Some(Box::new(sess)));
    (t.items.len() - 1) as i64
}

/// Feed inbound bytes into a session.  Writes any decrypted app-data plaintext
/// to `out_app`/`max_app` and returns its length.  Returns -1 on protocol error,
/// -2 on bad handle.
#[no_mangle] pub unsafe extern "C" fn aether_tls13_server_feed(
    handle: i64,
    in_ptr: *const c_void, n_in: c_int,
    out_app: *mut c_void, max_app: c_int,
) -> c_int {
    let t = sessions();
    if handle < 0 || (handle as usize) >= t.items.len() { return -2; }
    let Some(sess) = t.items[handle as usize].as_mut() else { return -2; };
    if in_ptr.is_null() || n_in < 0 { return -1; }
    let buf = std::slice::from_raw_parts(in_ptr as *const u8, n_in as usize);
    let plain = match sess.feed(buf) {
        Ok(p) => p, Err(_) => return -1,
    };
    if plain.len() > max_app as usize { return -1; }
    if !plain.is_empty() && !out_app.is_null() {
        let o = std::slice::from_raw_parts_mut(out_app as *mut u8, plain.len());
        o.copy_from_slice(&plain);
    }
    plain.len() as c_int
}

/// Take pending outbound bytes from a session into out/max_out.  Returns bytes written.
#[no_mangle] pub unsafe extern "C" fn aether_tls13_server_take_outbound(
    handle: i64, out: *mut c_void, max_out: c_int,
) -> c_int {
    let t = sessions();
    if handle < 0 || (handle as usize) >= t.items.len() { return -2; }
    let Some(sess) = t.items[handle as usize].as_mut() else { return -2; };
    let v = sess.take_outbound();
    if v.len() > max_out as usize { return -1; }
    if !v.is_empty() && !out.is_null() {
        std::slice::from_raw_parts_mut(out as *mut u8, v.len()).copy_from_slice(&v);
    }
    v.len() as c_int
}

/// Send app data; bytes go to the outbound queue (call take_outbound to drain).
#[no_mangle] pub unsafe extern "C" fn aether_tls13_server_send(
    handle: i64, app: *const c_void, n_app: c_int,
) -> c_int {
    let t = sessions();
    if handle < 0 || (handle as usize) >= t.items.len() { return -2; }
    let Some(sess) = t.items[handle as usize].as_mut() else { return -2; };
    if app.is_null() || n_app < 0 { return -1; }
    let buf = std::slice::from_raw_parts(app as *const u8, n_app as usize);
    match sess.send_app_data(buf) {
        Ok(_) => n_app, Err(_) => -1,
    }
}

/// Is the handshake done?  Returns 1 if Connected, 0 otherwise.
#[no_mangle] pub unsafe extern "C" fn aether_tls13_server_is_done(handle: i64) -> c_int {
    let t = sessions();
    if handle < 0 || (handle as usize) >= t.items.len() { return -1; }
    let Some(sess) = t.items[handle as usize].as_ref() else { return -1; };
    if sess.is_handshake_done() { 1 } else { 0 }
}

/// Free a session and return its slot to the pool.
#[no_mangle] pub unsafe extern "C" fn aether_tls13_server_free(handle: i64) -> c_int {
    let t = sessions();
    if handle < 0 || (handle as usize) >= t.items.len() { return -1; }
    t.items[handle as usize] = None;
    0
}

/// One-shot self-loopback smoke entry for the `.aether` audit witness.
/// Runs ClientHello -> server flight -> client Finished -> 1 round of
/// app data in process, all in Rust, all deterministic.  Returns 42 on
/// success, a small positive sentinel on failure (callable directly
/// from `.aether`).
#[no_mangle] pub unsafe extern "C" fn aether_tls13_self_loopback_smoke() -> c_int {
    use client_for_test::TestClient;
    let server_seed = [0x11u8; 32];
    let server_random = [0x22u8; 32];
    let server_x25519_priv = [0x33u8; 32];
    let client_x25519_priv = [0x44u8; 32];
    let client_random = [0x55u8; 32];
    let client_session_id: Vec<u8> = (0..32u8).collect();
    let mut server = TlsServerSession::new(
        &server_seed, &server_random, &server_x25519_priv,
        "aether-smoke", b"\x07",
    );
    if server.state() != State::ExpectClientHello { return 1; }
    let client = TestClient::new(client_x25519_priv, client_random, client_session_id);
    let ch = client.build_client_hello_record();
    if server.feed(&ch).is_err() { return 2; }
    let flight = server.take_outbound();
    if flight.is_empty() { return 3; }
    let (c_app, s_app, c_fin, _t) = match client
        .process_server_flight_and_build_client_finished(&flight, &ch)
    {
        Ok(v) => v, Err(_) => return 4,
    };
    if server.feed(&c_fin).is_err() { return 5; }
    if !server.is_handshake_done() { return 6; }
    let msg = b"smoke";
    if server.send_app_data(msg).is_err() { return 7; }
    let s_rec = server.take_outbound();
    let (it, plain, _n) = match record::open_record(&s_rec, &s_app.key, &s_app.iv, 0) {
        Some(v) => v, None => return 8,
    };
    if it != REC_APPLICATION_DATA || plain != msg { return 9; }
    let mut c_rec = Vec::new();
    record::seal_record(&mut c_rec, &c_app.key, &c_app.iv, 0, REC_APPLICATION_DATA, msg);
    match server.feed(&c_rec) {
        Ok(p) if p == msg => 42,
        _ => 10,
    }
}

// ============================================================================
// TlsClientSession — streaming client state machine.
//
// `TlsClientSession::new(...)` -> ExpectServerHello (after a ClientHello is
// queued in out_buf).  feed(server_bytes) advances through the server flight,
// verifies CV signature + server Finished MAC, emits client Finished, and
// transitions to Connected.  send_app_data + take_outbound mirror the server.
//
// Internally reuses `client_for_test::TestClient` for the actual byte-shovelling
// (build_client_hello_record + process_server_flight_and_build_client_finished),
// but exposes a clean state-machine façade analogous to TlsServerSession.
// ============================================================================

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ClientState {
    ExpectServerFlight,
    Connected,
    Closed,
}

pub struct TlsClientSession {
    state: ClientState,
    client: client_for_test::TestClient,
    client_hello_record: Vec<u8>,
    /// Bytes received from server but not yet a complete flight.
    pending: Vec<u8>,
    /// Bytes to send to server.
    out_buf: Vec<u8>,
    /// App data not yet read by caller.
    app_buf: Vec<u8>,

    server_app_keys: Option<key_schedule::DirKeys>,
    client_app_keys: Option<key_schedule::DirKeys>,
    server_app_seq: u64,
    client_app_seq: u64,
}

impl TlsClientSession {
    /// Construct a client and queue the ClientHello in the outbound buffer.
    /// Caller should `take_outbound()` and send those bytes to the server, then
    /// `feed()` server responses until `is_handshake_done()`.
    pub fn new(x25519_priv: [u8; 32], random: [u8; 32], session_id: Vec<u8>) -> Self {
        Self::new_with_alpn(x25519_priv, random, session_id, Vec::new())
    }

    pub fn new_with_alpn(
        x25519_priv: [u8; 32], random: [u8; 32], session_id: Vec<u8>,
        alpn_offered: Vec<Vec<u8>>,
    ) -> Self {
        Self::new_full(x25519_priv, random, session_id, alpn_offered, Vec::new())
    }

    /// Full constructor: trust anchors are Ed25519 SPKI pubkeys the client
    /// will accept on the server's cert.  Empty = trust-on-first-use.
    pub fn new_full(
        x25519_priv: [u8; 32], random: [u8; 32], session_id: Vec<u8>,
        alpn_offered: Vec<Vec<u8>>,
        trust_anchors: Vec<[u8; 32]>,
    ) -> Self {
        let client = client_for_test::TestClient::new_full(x25519_priv, random, session_id, alpn_offered, trust_anchors);
        let ch = client.build_client_hello_record();
        Self {
            state: ClientState::ExpectServerFlight,
            client,
            client_hello_record: ch.clone(),
            pending: Vec::new(),
            out_buf: ch,
            app_buf: Vec::new(),
            server_app_keys: None,
            client_app_keys: None,
            server_app_seq: 0,
            client_app_seq: 0,
        }
    }

    /// The ALPN protocol the server picked (matched against our offered list).
    pub fn negotiated_alpn(&self) -> Option<&[u8]> {
        self.client.alpn_chosen.get().and_then(|i| self.client.alpn_offered.get(i).map(|v| v.as_slice()))
    }

    pub fn state(&self) -> ClientState { self.state }
    pub fn is_handshake_done(&self) -> bool { self.state == ClientState::Connected }
    pub fn take_outbound(&mut self) -> Vec<u8> { std::mem::take(&mut self.out_buf) }

    /// Feed inbound server bytes.  Returns Ok(decrypted_app_data) or Err on protocol error.
    pub fn feed(&mut self, input: &[u8]) -> Result<Vec<u8>, &'static str> {
        self.pending.extend_from_slice(input);
        match self.state {
            ClientState::ExpectServerFlight => {
                // Walk records: we need ServerHello (cleartext) + 1+ encrypted records ending in Finished.
                // The existing process_server_flight_and_build_client_finished expects the FULL flight.
                // Heuristic: keep buffering until the first encrypted record contains Finished
                // (i.e., when the encrypted record's last decrypted handshake msg is type 20).
                // Simple approximation: try to process; if it returns "fin peel" type errors, wait.
                let attempt = self.client.process_server_flight_and_build_client_finished(
                    &self.pending, &self.client_hello_record);
                match attempt {
                    Ok((c_app, s_app, c_fin, _transcript)) => {
                        self.out_buf.extend_from_slice(&c_fin);
                        self.server_app_keys = Some(s_app);
                        self.client_app_keys = Some(c_app);
                        self.state = ClientState::Connected;
                        self.pending.clear(); // server flight consumed
                        Ok(Vec::new())
                    }
                    Err(e) => {
                        // Distinguish "need more bytes" vs "fatal".  process_server_flight_...
                        // returns errors like "no SH record" / "fin peel" when bytes are missing.
                        // Heuristic: if message mentions "peel" or "no SH" or "open server flight failed",
                        // treat as transient (need more data).
                        if e.contains("peel") || e.contains("no SH") || e.contains("open server flight failed") {
                            Ok(Vec::new())
                        } else {
                            Err("client flight processing failed")
                        }
                    }
                }
            }
            ClientState::Connected => {
                let keys = self.server_app_keys.as_ref().ok_or("no server app keys")?;
                let mut decrypted = Vec::new();
                loop {
                    if self.pending.len() < 5 { break; }
                    let len = u16::from_be_bytes([self.pending[3], self.pending[4]]) as usize;
                    if self.pending.len() < 5 + len { break; }
                    let rec_total = 5 + len;
                    let rec: Vec<u8> = self.pending[..rec_total].to_vec();
                    self.pending.drain(..rec_total);
                    if rec[0] == REC_CHANGE_CIPHER_SPEC { continue; }
                    let (inner_type, plain, _n) =
                        record::open_record(&rec, &keys.key, &keys.iv, self.server_app_seq)
                            .ok_or("client app decrypt failed")?;
                    self.server_app_seq += 1;
                    match inner_type {
                        REC_APPLICATION_DATA => decrypted.extend_from_slice(&plain),
                        REC_ALERT => { self.state = ClientState::Closed; }
                        _ => {} // post-handshake messages ignored
                    }
                }
                self.app_buf.extend_from_slice(&decrypted);
                Ok(decrypted)
            }
            ClientState::Closed => Err("session closed"),
        }
    }

    /// Encrypt + queue application-data bytes for sending.
    pub fn send_app_data(&mut self, plaintext: &[u8]) -> Result<(), &'static str> {
        if self.state != ClientState::Connected { return Err("not connected"); }
        let keys = self.client_app_keys.as_ref().ok_or("no client app keys")?;
        const CHUNK: usize = 16 * 1024 - 32;
        let mut i = 0;
        while i < plaintext.len() {
            let take = (plaintext.len() - i).min(CHUNK);
            record::seal_record(&mut self.out_buf, &keys.key, &keys.iv,
                self.client_app_seq, REC_APPLICATION_DATA, &plaintext[i..i + take]);
            self.client_app_seq += 1;
            i += take;
        }
        Ok(())
    }
}

// ============================================================================
// FFI for TlsClientSession
// ============================================================================

struct ClientTable {
    items: Vec<Option<Box<TlsClientSession>>>,
}
struct ClientTableCell(UnsafeCell<ClientTable>);
unsafe impl Sync for ClientTableCell {}
static CLIENTS: ClientTableCell = ClientTableCell(UnsafeCell::new(ClientTable { items: Vec::new() }));
#[inline] unsafe fn clients() -> &'static mut ClientTable { &mut *CLIENTS.0.get() }

#[no_mangle] pub unsafe extern "C" fn aether_tls13_client_new(
    x25519_priv_ptr: *const c_void,
    random_ptr: *const c_void,
    session_id_ptr: *const c_void, n_session_id: c_int,
) -> i64 {
    if x25519_priv_ptr.is_null() || random_ptr.is_null() || n_session_id < 0 { return -1; }
    let priv_ = {
        let s = std::slice::from_raw_parts(x25519_priv_ptr as *const u8, 32);
        let mut a = [0u8; 32]; a.copy_from_slice(s); a
    };
    let rand = {
        let s = std::slice::from_raw_parts(random_ptr as *const u8, 32);
        let mut a = [0u8; 32]; a.copy_from_slice(s); a
    };
    let sid = if session_id_ptr.is_null() || n_session_id == 0 { Vec::new() }
              else { std::slice::from_raw_parts(session_id_ptr as *const u8, n_session_id as usize).to_vec() };
    let c = TlsClientSession::new(priv_, rand, sid);
    let t = clients();
    for (i, slot) in t.items.iter_mut().enumerate() {
        if slot.is_none() { *slot = Some(Box::new(c)); return i as i64; }
    }
    t.items.push(Some(Box::new(c)));
    (t.items.len() - 1) as i64
}

#[no_mangle] pub unsafe extern "C" fn aether_tls13_client_feed(
    handle: i64, in_ptr: *const c_void, n_in: c_int,
    out_app: *mut c_void, max_app: c_int,
) -> c_int {
    let t = clients();
    if handle < 0 || (handle as usize) >= t.items.len() { return -2; }
    let Some(sess) = t.items[handle as usize].as_mut() else { return -2; };
    let buf = std::slice::from_raw_parts(in_ptr as *const u8, n_in as usize);
    let plain = match sess.feed(buf) { Ok(p) => p, Err(_) => return -1, };
    if plain.len() > max_app as usize { return -1; }
    if !plain.is_empty() && !out_app.is_null() {
        std::slice::from_raw_parts_mut(out_app as *mut u8, plain.len()).copy_from_slice(&plain);
    }
    plain.len() as c_int
}

#[no_mangle] pub unsafe extern "C" fn aether_tls13_client_take_outbound(
    handle: i64, out: *mut c_void, max_out: c_int,
) -> c_int {
    let t = clients();
    if handle < 0 || (handle as usize) >= t.items.len() { return -2; }
    let Some(sess) = t.items[handle as usize].as_mut() else { return -2; };
    let v = sess.take_outbound();
    if v.len() > max_out as usize { return -1; }
    if !v.is_empty() && !out.is_null() {
        std::slice::from_raw_parts_mut(out as *mut u8, v.len()).copy_from_slice(&v);
    }
    v.len() as c_int
}

#[no_mangle] pub unsafe extern "C" fn aether_tls13_client_send(
    handle: i64, app: *const c_void, n_app: c_int,
) -> c_int {
    let t = clients();
    if handle < 0 || (handle as usize) >= t.items.len() { return -2; }
    let Some(sess) = t.items[handle as usize].as_mut() else { return -2; };
    let buf = std::slice::from_raw_parts(app as *const u8, n_app as usize);
    match sess.send_app_data(buf) { Ok(_) => n_app, Err(_) => -1 }
}

#[no_mangle] pub unsafe extern "C" fn aether_tls13_client_is_done(handle: i64) -> c_int {
    let t = clients();
    if handle < 0 || (handle as usize) >= t.items.len() { return -1; }
    let Some(sess) = t.items[handle as usize].as_ref() else { return -1; };
    if sess.is_handshake_done() { 1 } else { 0 }
}

#[no_mangle] pub unsafe extern "C" fn aether_tls13_client_free(handle: i64) -> c_int {
    let t = clients();
    if handle < 0 || (handle as usize) >= t.items.len() { return -1; }
    t.items[handle as usize] = None;
    0
}

// ============================================================================
// Self-test client (kept as a public module — `TlsClientSession` builds on top
// of `client_for_test::TestClient`).
// ============================================================================

pub mod client_for_test {
    use super::*;

    pub struct TestClient {
        pub priv_x25519: [u8; 32],
        pub pub_x25519: [u8; 32],
        pub random: [u8; 32],
        pub session_id: Vec<u8>,
        /// ALPN protocol names to offer; empty = no ALPN extension.
        pub alpn_offered: Vec<Vec<u8>>,
        /// ALPN chosen by the server (filled after processing flight).
        pub alpn_chosen: std::cell::Cell<Option<usize>>,
        /// FR-19.1-extra-cert: Ed25519 public keys the client trusts.
        /// When non-empty, the server's cert SPKI MUST appear here AND the
        /// cert must self-verify (signed by its own SPKI key) for the
        /// handshake to proceed.  Empty = trust on first use (accept any
        /// self-signed cert).
        pub trust_anchors: Vec<[u8; 32]>,
    }

    impl TestClient {
        pub fn new(priv_x25519: [u8; 32], random: [u8; 32], session_id: Vec<u8>) -> Self {
            Self::new_with_alpn(priv_x25519, random, session_id, Vec::new())
        }
        pub fn new_with_alpn(
            priv_x25519: [u8; 32], random: [u8; 32], session_id: Vec<u8>,
            alpn_offered: Vec<Vec<u8>>,
        ) -> Self {
            Self::new_full(priv_x25519, random, session_id, alpn_offered, Vec::new())
        }
        pub fn new_full(
            priv_x25519: [u8; 32], random: [u8; 32], session_id: Vec<u8>,
            alpn_offered: Vec<Vec<u8>>,
            trust_anchors: Vec<[u8; 32]>,
        ) -> Self {
            let mut bp = [0u8; 32]; bp[0] = 9;
            let pub_x25519 = x25519_scalar_mult(&priv_x25519, &bp);
            Self { priv_x25519, pub_x25519, random, session_id, alpn_offered,
                   alpn_chosen: std::cell::Cell::new(None), trust_anchors }
        }
        /// Returns the alpn_offered index that the server picked, if any.
        pub fn negotiated_alpn_index(&self) -> Option<usize> { self.alpn_chosen.get() }

        /// Construct a real ClientHello record carrying our key share + supported_versions etc.
        pub fn build_client_hello_record(&self) -> Vec<u8> {
            let mut body = Vec::with_capacity(256);
            body.extend_from_slice(&TLS_VERSION_1_2_LEGACY.to_be_bytes());
            body.extend_from_slice(&self.random);
            body.push(self.session_id.len() as u8);
            body.extend_from_slice(&self.session_id);
            // cipher suites: 1 suite -> 2 bytes
            body.extend_from_slice(&2u16.to_be_bytes());
            body.extend_from_slice(&CIPHER_SUITE_CHACHA20_POLY1305_SHA256.to_be_bytes());
            // legacy compression methods: 1 byte length, null compression
            body.push(1); body.push(0);
            // extensions
            let mut exts = Vec::with_capacity(64);
            // supported_versions (client form): 1-byte len + list of u16 versions
            exts.extend_from_slice(&0x002bu16.to_be_bytes());
            exts.extend_from_slice(&3u16.to_be_bytes()); // ext length = 3
            exts.push(2); // versions list length
            exts.extend_from_slice(&TLS_VERSION_1_3.to_be_bytes());
            // supported_groups: u16 length || u16 x25519
            exts.extend_from_slice(&0x000au16.to_be_bytes());
            exts.extend_from_slice(&4u16.to_be_bytes()); // ext length = 4
            exts.extend_from_slice(&2u16.to_be_bytes()); // list length = 2
            exts.extend_from_slice(&NAMED_GROUP_X25519.to_be_bytes());
            // signature_algorithms: u16 length || u16 ed25519
            exts.extend_from_slice(&0x000du16.to_be_bytes());
            exts.extend_from_slice(&4u16.to_be_bytes());
            exts.extend_from_slice(&2u16.to_be_bytes());
            exts.extend_from_slice(&SIG_ALG_ED25519.to_be_bytes());
            // key_share (client form): u16 total length || KeyShareEntry list
            let entry_total = 2 + 2 + 32;
            exts.extend_from_slice(&0x0033u16.to_be_bytes());
            exts.extend_from_slice(&((entry_total + 2) as u16).to_be_bytes());
            exts.extend_from_slice(&(entry_total as u16).to_be_bytes());
            exts.extend_from_slice(&NAMED_GROUP_X25519.to_be_bytes());
            exts.extend_from_slice(&32u16.to_be_bytes());
            exts.extend_from_slice(&self.pub_x25519);
            // ALPN (RFC 7301): u16 type=0x0010 || u16 ext_len || u16 list_len || (u8 plen || proto)+
            if !self.alpn_offered.is_empty() {
                let mut inner = Vec::new();
                for p in &self.alpn_offered {
                    inner.push(p.len() as u8);
                    inner.extend_from_slice(p);
                }
                let ext_body_len = 2 + inner.len();
                exts.extend_from_slice(&0x0010u16.to_be_bytes());
                exts.extend_from_slice(&(ext_body_len as u16).to_be_bytes());
                exts.extend_from_slice(&(inner.len() as u16).to_be_bytes());
                exts.extend_from_slice(&inner);
            }
            body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
            body.extend_from_slice(&exts);

            let hs = handshake::wrap(HS_CLIENT_HELLO, &body);
            let mut rec = Vec::with_capacity(5 + hs.len());
            record::write_plaintext_record(&mut rec, REC_HANDSHAKE, &hs);
            rec
        }

        /// Drive a full server flight: parse ServerHello, derive keys, decrypt
        /// EE/Cert/CV/Finished, verify, send our Finished record.
        /// Returns (client_app_keys, server_app_keys, client_finished_record, transcript_through_server_fin).
        pub fn process_server_flight_and_build_client_finished(
            &self,
            server_flight: &[u8],
            client_hello_record: &[u8],
        ) -> Result<(key_schedule::DirKeys, key_schedule::DirKeys, Vec<u8>, Vec<u8>), String> {
            // 1) Parse SH record.
            let (rt1, sh_record_body, consumed1) = record::parse_plaintext(server_flight)
                .ok_or_else(|| "no SH record".to_string())?;
            if rt1 != REC_HANDSHAKE { return Err("first record not handshake".into()); }
            if sh_record_body.len() < 4 { return Err("SH too short".into()); }
            if sh_record_body[0] != HS_SERVER_HELLO { return Err("not SH".into()); }
            let sh_len = (sh_record_body[1] as usize) << 16
                       | (sh_record_body[2] as usize) << 8
                       | (sh_record_body[3] as usize);
            if sh_record_body.len() != 4 + sh_len { return Err("SH len mismatch".into()); }
            let sh_body = &sh_record_body[4..];

            let mut r = handshake::Reader::new(sh_body);
            let _legacy_ver = r.read_u16().ok_or("sh ver")?;
            let _server_random = r.read_bytes(32).ok_or("sh rand")?.to_vec();
            let sid_echo_len = r.read_u8().ok_or("sh sid len")? as usize;
            let _sid_echo = r.read_bytes(sid_echo_len).ok_or("sh sid")?.to_vec();
            let _cs = r.read_u16().ok_or("sh cs")?;
            let _comp = r.read_u8().ok_or("sh comp")?;
            let ext_total = r.read_u16().ok_or("sh ext total")? as usize;
            let ext_body = r.read_bytes(ext_total).ok_or("sh ext body")?;

            let mut server_x25519_pub = [0u8; 32];
            let mut er = handshake::Reader::new(ext_body);
            while er.remaining() > 0 {
                let t = er.read_u16().ok_or("ext type")?;
                let l = er.read_u16().ok_or("ext len")? as usize;
                let d = er.read_bytes(l).ok_or("ext data")?;
                if t == 0x0033 {
                    // server key_share: KeyShareEntry { group, len, data }
                    if d.len() < 36 { return Err("ks len".into()); }
                    let group = u16::from_be_bytes([d[0], d[1]]);
                    if group != NAMED_GROUP_X25519 { return Err("server didn't pick x25519".into()); }
                    let dlen = u16::from_be_bytes([d[2], d[3]]) as usize;
                    if dlen != 32 { return Err("ks data len".into()); }
                    server_x25519_pub.copy_from_slice(&d[4..36]);
                }
            }

            // 2) Compute DHE shared.
            let dhe = x25519_scalar_mult(&self.priv_x25519, &server_x25519_pub);

            // Transcript = ClientHello msg || ServerHello msg (handshake-msg bodies, not the records).
            // Client hello record body is the inside of the 5-byte record header.
            let (_rt_ch, ch_record_body, _ch_consumed) = record::parse_plaintext(client_hello_record)
                .ok_or("ch record")?;
            let mut transcript = Vec::new();
            transcript.extend_from_slice(ch_record_body);
            transcript.extend_from_slice(sh_record_body);

            let sched = key_schedule::Schedule::handshake_phase(&dhe, &transcript);
            let server_hs_keys = key_schedule::traffic_keys(&sched.server_hs_traffic_secret);
            let client_hs_keys = key_schedule::traffic_keys(&sched.client_hs_traffic_secret);

            // 3) Decrypt the server's encrypted flight (one or more records).
            let mut pos = consumed1;
            let mut server_hs_seq = 0u64;
            let mut decrypted = Vec::new();
            // Server flight should fit in one record but tolerate more.
            while pos < server_flight.len() {
                if server_flight[pos] == REC_CHANGE_CIPHER_SPEC {
                    let (_t, _b, consumed) = record::parse_plaintext(&server_flight[pos..])
                        .ok_or("ccs")?;
                    pos += consumed;
                    continue;
                }
                let (inner_type, plain, consumed) =
                    record::open_record(&server_flight[pos..], &server_hs_keys.key, &server_hs_keys.iv, server_hs_seq)
                        .ok_or_else(|| "open server flight failed".to_string())?;
                if inner_type != REC_HANDSHAKE { return Err("inner not hs".into()); }
                decrypted.extend_from_slice(&plain);
                server_hs_seq += 1;
                pos += consumed;
                // Heuristic: if we just decrypted a Finished, stop.
                // Cheap parse: walk the messages we just got.
                let mut walk = 0;
                let mut saw_fin = false;
                while walk + 4 <= decrypted.len() {
                    let mt = decrypted[walk];
                    let ml = (decrypted[walk + 1] as usize) << 16
                           | (decrypted[walk + 2] as usize) << 8
                           | (decrypted[walk + 3] as usize);
                    if walk + 4 + ml > decrypted.len() { break; }
                    if mt == HS_FINISHED { saw_fin = true; }
                    walk += 4 + ml;
                }
                if saw_fin { break; }
            }

            // 4) Walk EE/Cert/CV/Finished out of `decrypted`, appending each to transcript as we go.
            //    Verify CertificateVerify signature and server Finished MAC.
            let mut idx = 0;
            // EE
            let (ee_full, n1) = peel_hs_msg(&decrypted[idx..]).ok_or("ee peel")?;
            if ee_full[0] != HS_ENCRYPTED_EXTENSIONS { return Err("not EE".into()); }
            // Parse EE body (after the 4-byte HS header) for ALPN.
            if ee_full.len() >= 6 {
                let ee_body = &ee_full[4..];
                let list_len = u16::from_be_bytes([ee_body[0], ee_body[1]]) as usize;
                if ee_body.len() >= 2 + list_len {
                    let mut ep = 2;
                    while ep + 4 <= 2 + list_len {
                        let etype = u16::from_be_bytes([ee_body[ep], ee_body[ep + 1]]);
                        let elen = u16::from_be_bytes([ee_body[ep + 2], ee_body[ep + 3]]) as usize;
                        if ep + 4 + elen > ee_body.len() { break; }
                        if etype == 0x0010 && elen >= 3 {
                            // ALPN: u16 inner_list_len || u8 plen || proto bytes
                            let inner_list_len = u16::from_be_bytes([ee_body[ep + 4], ee_body[ep + 5]]) as usize;
                            if inner_list_len >= 1 && ep + 4 + 2 + 1 <= ee_body.len() {
                                let plen = ee_body[ep + 6] as usize;
                                if ep + 4 + 2 + 1 + plen <= ee_body.len() {
                                    let proto = &ee_body[ep + 7 .. ep + 7 + plen];
                                    if let Some(i) = self.alpn_offered.iter().position(|p| p.as_slice() == proto) {
                                        self.alpn_chosen.set(Some(i));
                                    }
                                }
                            }
                        }
                        ep += 4 + elen;
                    }
                }
            }
            transcript.extend_from_slice(ee_full);
            idx += n1;
            // Cert
            let (cert_full, n2) = peel_hs_msg(&decrypted[idx..]).ok_or("cert peel")?;
            if cert_full[0] != HS_CERTIFICATE { return Err("not Cert".into()); }
            // Extract pub key + the cert DER itself for trust-anchor check.
            let server_cert_body = &cert_full[4..];
            let (server_cert_pub, server_cert_der) =
                extract_ed25519_pub_and_der_from_cert_msg_body(server_cert_body)
                    .ok_or("extract pub failed")?;

            // Trust-anchor verification (FR-19.1-extra-cert).  When the client
            // was configured with `trust_anchors`, the server's cert SPKI
            // public key MUST be one of them AND the cert must self-verify
            // (signed by its own SPKI — true for our self-signed deployment).
            if !self.trust_anchors.is_empty() {
                if !self.trust_anchors.iter().any(|a| a == &server_cert_pub) {
                    return Err("server cert pubkey not in trust anchors".into());
                }
                if !verify_self_signed_ed25519_cert(&server_cert_der) {
                    return Err("server cert self-signature failed".into());
                }
            }
            transcript.extend_from_slice(cert_full);
            idx += n2;
            // CV
            let (cv_full, n3) = peel_hs_msg(&decrypted[idx..]).ok_or("cv peel")?;
            if cv_full[0] != HS_CERTIFICATE_VERIFY { return Err("not CV".into()); }
            // cv body = u16 sig_alg || u16 len || sig
            if cv_full.len() < 4 + 4 { return Err("cv too short".into()); }
            let sig_alg = u16::from_be_bytes([cv_full[4], cv_full[5]]);
            let sig_len = u16::from_be_bytes([cv_full[6], cv_full[7]]) as usize;
            if sig_alg != SIG_ALG_ED25519 || sig_len != 64 { return Err("bad sig alg/len".into()); }
            let mut sig = [0u8; 64]; sig.copy_from_slice(&cv_full[8..8 + 64]);
            let signed = handshake::server_cert_verify_signed_content(&transcript);
            let verify_ok = unsafe {
                crate::aether_ed25519_verify(
                    server_cert_pub.as_ptr() as *const c_void,
                    signed.as_ptr() as *const c_void, signed.len() as c_int,
                    sig.as_ptr() as *const c_void,
                )
            };
            if verify_ok != 0 { return Err("CV signature failed verify".into()); }
            transcript.extend_from_slice(cv_full);
            idx += n3;
            // Finished
            let (fin_full, _n4) = peel_hs_msg(&decrypted[idx..]).ok_or("fin peel")?;
            if fin_full[0] != HS_FINISHED { return Err("not Finished".into()); }
            let expected_fin = key_schedule::finished_mac(
                &sched.server_hs_traffic_secret,
                &transcript,
            );
            if fin_full[4..4 + 32] != expected_fin { return Err("server Finished MAC mismatch".into()); }
            transcript.extend_from_slice(fin_full);

            // 5) App secrets now derivable.
            let mut sched = sched;
            sched.finalize_app_secrets(&transcript);
            let server_app_keys = key_schedule::traffic_keys(&sched.server_app_traffic_secret);
            let client_app_keys = key_schedule::traffic_keys(&sched.client_app_traffic_secret);

            // 6) Build our client Finished using client_hs_traffic_secret over current transcript.
            let client_fin_vd = key_schedule::finished_mac(
                &sched.client_hs_traffic_secret,
                &transcript,
            );
            let client_fin_msg = handshake::wrap(HS_FINISHED, &handshake::encode_finished(&client_fin_vd));
            // Wrap as encrypted handshake record using client_hs_keys @ seq 0.
            let mut client_fin_record = Vec::new();
            record::seal_record(
                &mut client_fin_record,
                &client_hs_keys.key, &client_hs_keys.iv, 0,
                REC_HANDSHAKE, &client_fin_msg,
            );

            Ok((client_app_keys, server_app_keys, client_fin_record, transcript))
        }
    }

    /// Peel one handshake message out of a buffer (full message = 4 header + body).
    /// Returns (full_msg_slice, bytes_consumed).
    fn peel_hs_msg(buf: &[u8]) -> Option<(&[u8], usize)> {
        if buf.len() < 4 { return None; }
        let body_len = (buf[1] as usize) << 16 | (buf[2] as usize) << 8 | (buf[3] as usize);
        let total = 4 + body_len;
        if buf.len() < total { return None; }
        Some((&buf[..total], total))
    }

    /// Verify a self-signed Ed25519 X.509 certificate: extracts the SPKI
    /// pubkey, the TBS DER, and the signature, then runs ed25519_verify.
    /// Returns true iff the cert is signed by its own SPKI key.
    pub fn verify_self_signed_ed25519_cert(cert_der: &[u8]) -> bool {
        // Cert: SEQUENCE { tbs, sigAlg, BIT STRING sig }
        let Some((tag, body, _total)) = der_peel(cert_der) else { return false; };
        if tag != 0x30 { return false; }
        // First child = TBSCertificate.  We need its FULL DER bytes (incl. SEQ
        // header) since that's what was signed.
        let Some((tbs_tag, _tbs_body, tbs_total)) = der_peel(body) else { return false; };
        if tbs_tag != 0x30 { return false; }
        let tbs_full = &body[..tbs_total];

        // Walk TBS children to find SPKI (7th child after version/serial/sigAlg/
        // issuer/validity/subject).
        let tbs_inner = &body[..tbs_total];
        let Some((_, tbs_children, _)) = der_peel(tbs_inner) else { return false; };
        let mut p = tbs_children;
        for _ in 0..6 {
            let Some((_t, _b, n)) = der_peel(p) else { return false; };
            p = &p[n..];
        }
        let Some((spki_tag, spki_body, _)) = der_peel(p) else { return false; };
        if spki_tag != 0x30 { return false; }
        let Some((_alg_tag, _alg_body, alg_n)) = der_peel(spki_body) else { return false; };
        let bit_str = &spki_body[alg_n..];
        let Some((bs_tag, bs_body, _)) = der_peel(bit_str) else { return false; };
        if bs_tag != 0x03 || bs_body.len() < 33 || bs_body[0] != 0 { return false; }
        let mut pub_key = [0u8; 32];
        pub_key.copy_from_slice(&bs_body[1..33]);

        // Outer second child = sigAlg (skip), third child = BIT STRING signature.
        let mut q = &body[tbs_total..];
        let Some((_a_tag, _a_body, a_n)) = der_peel(q) else { return false; };
        q = &q[a_n..];
        let Some((sig_tag, sig_body, _)) = der_peel(q) else { return false; };
        if sig_tag != 0x03 || sig_body.len() != 65 || sig_body[0] != 0 { return false; }
        let sig = &sig_body[1..65];

        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(sig);

        unsafe {
            crate::aether_ed25519_verify(
                pub_key.as_ptr() as *const c_void,
                tbs_full.as_ptr() as *const c_void, tbs_full.len() as c_int,
                sig_arr.as_ptr() as *const c_void,
            ) == 0
        }
    }

    /// Walk the Certificate message body and return (pubkey, full_cert_der).
    fn extract_ed25519_pub_and_der_from_cert_msg_body(body: &[u8]) -> Option<([u8; 32], Vec<u8>)> {
        if body.len() < 1 { return None; }
        let ctx_len = body[0] as usize;
        if body.len() < 1 + ctx_len + 3 { return None; }
        let p = 1 + ctx_len;
        let list_len = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | (body[p + 2] as usize);
        let p = p + 3;
        if body.len() < p + list_len { return None; }
        let cert_len = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | (body[p + 2] as usize);
        let p = p + 3;
        if body.len() < p + cert_len + 2 { return None; }
        let cert_der = &body[p..p + cert_len];
        let pubkey = extract_ed25519_pub_from_cert_der(cert_der)?;
        Some((pubkey, cert_der.to_vec()))
    }

    /// Just walk a cert DER and pull the Ed25519 pubkey from the SPKI.
    fn extract_ed25519_pub_from_cert_der(cert_der: &[u8]) -> Option<[u8; 32]> {
        let (tag, body0, _) = der_peel(cert_der)?;
        if tag != 0x30 { return None; }
        let (ttag, tbs_body, _) = der_peel(body0)?;
        if ttag != 0x30 { return None; }
        let mut p = tbs_body;
        for _ in 0..6 {
            let (_t, _b, n) = der_peel(p)?;
            p = &p[n..];
        }
        let (spki_tag, spki_body, _) = der_peel(p)?;
        if spki_tag != 0x30 { return None; }
        let (_alg_tag, _alg_body, alg_n) = der_peel(spki_body)?;
        let bit_str = &spki_body[alg_n..];
        let (bs_tag, bs_body, _) = der_peel(bit_str)?;
        if bs_tag != 0x03 || bs_body.len() < 33 || bs_body[0] != 0 { return None; }
        let mut out = [0u8; 32];
        out.copy_from_slice(&bs_body[1..33]);
        Some(out)
    }

    /// Walk the Certificate message body and extract the Ed25519 SPKI public key.
    /// Body layout (per encode_certificate):
    ///   u8 ctx_len(=0) || u24 list_len ||
    ///     u24 cert_len || cert_der || u16 ext_len(=0)
    /// `cert_der` is a DER SEQUENCE whose first child is TBSCertificate; we walk
    /// the TBS to find the SPKI (last child) and pull the 32-byte BIT STRING contents.
    fn extract_ed25519_pub_from_cert_msg_body(body: &[u8]) -> Option<[u8; 32]> {
        if body.len() < 1 { return None; }
        let ctx_len = body[0] as usize;
        if body.len() < 1 + ctx_len + 3 { return None; }
        let p = 1 + ctx_len;
        let list_len = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | (body[p + 2] as usize);
        let p = p + 3;
        if body.len() < p + list_len { return None; }
        // First (only) CertificateEntry: u24 cert_len || cert_der || u16 ext_len
        let cert_len = ((body[p] as usize) << 16) | ((body[p + 1] as usize) << 8) | (body[p + 2] as usize);
        let p = p + 3;
        if body.len() < p + cert_len + 2 { return None; }
        let cert_der = &body[p..p + cert_len];
        // Walk DER tree.
        // cert_der: SEQUENCE { tbs, sigAlg, sigValue }
        let (tag, body0, _consumed) = der_peel(cert_der)?;
        if tag != 0x30 { return None; }
        // Peel TBS (first child).
        let (ttag, tbs_body, n1) = der_peel(body0)?;
        if ttag != 0x30 { return None; }
        let _ = n1;
        // Walk TBS children — SPKI is the 7th child (version, serial, sigAlg, issuer, validity, subject, spki).
        let mut p = tbs_body;
        for _ in 0..6 {
            let (_t, _b, n) = der_peel(p)?;
            p = &p[n..];
        }
        let (spki_tag, spki_body, _spki_total) = der_peel(p)?;
        if spki_tag != 0x30 { return None; }
        // spki_body: SEQUENCE { algorithm, BIT STRING { 0x00 || key } }
        let (_alg_tag, _alg_body, alg_n) = der_peel(spki_body)?;
        let bit_str = &spki_body[alg_n..];
        let (bs_tag, bs_body, _bs_total) = der_peel(bit_str)?;
        if bs_tag != 0x03 || bs_body.len() < 33 { return None; }
        // bs_body[0] is unused-bits count (0 for our cert).
        if bs_body[0] != 0 { return None; }
        let mut out = [0u8; 32];
        out.copy_from_slice(&bs_body[1..33]);
        Some(out)
    }

    /// Peel one DER TLV: returns (tag, body, total_consumed).
    fn der_peel(buf: &[u8]) -> Option<(u8, &[u8], usize)> {
        if buf.len() < 2 { return None; }
        let tag = buf[0];
        let first_len = buf[1];
        let (len, len_n) = if first_len & 0x80 == 0 {
            (first_len as usize, 1usize)
        } else {
            let n = (first_len & 0x7f) as usize;
            if n == 0 || n > 4 || buf.len() < 2 + n { return None; }
            let mut acc = 0usize;
            for i in 0..n { acc = (acc << 8) | (buf[2 + i] as usize); }
            (acc, 1 + n)
        };
        if buf.len() < 1 + len_n + len { return None; }
        Some((tag, &buf[1 + len_n..1 + len_n + len], 1 + len_n + len))
    }
}

// ============================================================================
// Unit tests (in-tree)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_nonce_xor() {
        let iv = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c];
        let n0 = record::build_nonce(&iv, 0);
        assert_eq!(n0, iv);
        let n1 = record::build_nonce(&iv, 1);
        let expected1 = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0d];
        assert_eq!(n1, expected1);
        let n256 = record::build_nonce(&iv, 256);
        let expected256 = [0x01u8, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0a, 0x0c];
        assert_eq!(n256, expected256);
    }

    /// RFC 7539 §2.8.2 — known ciphertext + tag for the canonical test vector.
    /// Catches any self-consistent bug in our chacha20-poly1305 impl.
    #[test]
    fn aead_rfc7539_2_8_2_vector() {
        let key: [u8; 32] = [
            0x80,0x81,0x82,0x83,0x84,0x85,0x86,0x87,
            0x88,0x89,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,
            0x90,0x91,0x92,0x93,0x94,0x95,0x96,0x97,
            0x98,0x99,0x9a,0x9b,0x9c,0x9d,0x9e,0x9f,
        ];
        let nonce: [u8; 12] = [
            0x07,0x00,0x00,0x00,0x40,0x41,0x42,0x43,
            0x44,0x45,0x46,0x47,
        ];
        let aad: [u8; 12] = [
            0x50,0x51,0x52,0x53,0xc0,0xc1,0xc2,0xc3,
            0xc4,0xc5,0xc6,0xc7,
        ];
        let plain = b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
        let expected_ct_tag = hex_decode(concat!(
            "d31a8d34648e60db7b86afbc53ef7ec2",
            "a4aded51296e08fea9e2b5a736ee62d6",
            "3dbea45e8ca9671282fafb69da92728b",
            "1a71de0a9e060b2905d6a5b67ecd3b36",
            "92ddbd7f2d778b8c9803aee328091b58",
            "fab324e4fad675945585808b4831d7bc",
            "3ff4def08e4b7a9de576d26586cec64b",
            "6116",
            // Poly1305 tag (RFC 7539 §2.8.2):
            "1ae10b594f09e26a7e902ecbd0600691",
        ));
        let got = aead_chacha20_poly1305_seal(&key, &nonce, &aad, plain);
        if got != expected_ct_tag {
            // Find first differing byte for triage.
            for i in 0..got.len().min(expected_ct_tag.len()) {
                if got[i] != expected_ct_tag[i] {
                    panic!("RFC 7539 mismatch at byte {}: got {:02x} expected {:02x}\n  got first 32 = {}\n  exp first 32 = {}",
                        i, got[i], expected_ct_tag[i],
                        got[..32.min(got.len())].iter().map(|b| format!("{:02x}", b)).collect::<String>(),
                        expected_ct_tag[..32.min(expected_ct_tag.len())].iter().map(|b| format!("{:02x}", b)).collect::<String>(),
                    );
                }
            }
            panic!("length mismatch: got {} expected {}", got.len(), expected_ct_tag.len());
        }
    }

    fn hex_decode(s: &str) -> Vec<u8> {
        let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        let mut out = Vec::with_capacity(s.len() / 2);
        let bytes = s.as_bytes();
        for i in (0..bytes.len()).step_by(2) {
            let h = char::from(bytes[i]).to_digit(16).unwrap() as u8;
            let l = char::from(bytes[i + 1]).to_digit(16).unwrap() as u8;
            out.push((h << 4) | l);
        }
        out
    }

    #[test]
    fn aead_seal_open_roundtrip() {
        let key = [0x42u8; 32];
        let iv = [0x11u8; 12];
        let nonce = record::build_nonce(&iv, 5);
        let aad = b"associated-data";
        let plain = b"Hello, TLS 1.3.";
        let sealed = aead_chacha20_poly1305_seal(&key, &nonce, aad, plain);
        assert_eq!(sealed.len(), plain.len() + 16);
        let opened = aead_chacha20_poly1305_open(&key, &nonce, aad, &sealed).unwrap();
        assert_eq!(opened, plain);
        // Tamper: bit-flip in tag should fail.
        let mut t = sealed.clone(); *t.last_mut().unwrap() ^= 1;
        assert!(aead_chacha20_poly1305_open(&key, &nonce, aad, &t).is_none());
    }

    #[test]
    fn hkdf_expand_label_shape() {
        let secret = [0x55u8; 32];
        let okm = hkdf_expand_label(&secret, b"key", &[], 32);
        assert_eq!(okm.len(), 32);
        // Determinism.
        let again = hkdf_expand_label(&secret, b"key", &[], 32);
        assert_eq!(okm, again);
        // Different label -> different output.
        let other = hkdf_expand_label(&secret, b"iv", &[], 32);
        assert_ne!(okm, other);
    }

    #[test]
    fn self_signed_cert_smoke() {
        let seed = [0xaau8; 32];
        let (cert, pub_key) = x509::self_sign_ed25519(&seed, "aether-test", b"\x01");
        // Cert must start with 0x30 (SEQUENCE).
        assert_eq!(cert[0], 0x30);
        // Pub key 32 bytes.
        assert_eq!(pub_key.len(), 32);
        // Cert must contain the pub-key bytes verbatim (BIT STRING body).
        let mut found = false;
        for i in 0..cert.len() - 33 {
            if cert[i] == 0x00 && &cert[i + 1..i + 33] == &pub_key {
                found = true; break;
            }
        }
        assert!(found, "pub key not embedded in cert");
    }
}
