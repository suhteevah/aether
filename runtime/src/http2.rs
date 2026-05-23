//! HTTP/2 framing + HPACK static table (RFC 7540 + RFC 7541).
//!
//! Minimum viable server profile:
//!   - Frames: DATA, HEADERS, SETTINGS, WINDOW_UPDATE, PING, GOAWAY, RST_STREAM
//!   - HPACK: static table only, no dynamic table, no Huffman encode (decode
//!     accepted), variable-length integers per §5.1
//!   - Stream multiplexing: stream ID tracking, half-close, end-of-stream flag
//!   - Settings: SETTINGS_HEADER_TABLE_SIZE (we send 0; no dynamic indexing),
//!     SETTINGS_MAX_CONCURRENT_STREAMS, SETTINGS_INITIAL_WINDOW_SIZE
//!
//! Not yet:
//!   - PUSH_PROMISE / CONTINUATION (rare; can stitch into HEADERS if needed)
//!   - PRIORITY frames (deprecated by RFC 9113)
//!   - HPACK dynamic table indexing
//!   - HPACK Huffman encoding (we still DECODE Huffman to interop with curl)
//!   - Flow-control credit on outbound (we assume credits are large)
//!
//! roadmap: P19.2-extra

use std::os::raw::c_int;

pub const FRAME_DATA: u8 = 0x0;
pub const FRAME_HEADERS: u8 = 0x1;
pub const FRAME_PRIORITY: u8 = 0x2;
pub const FRAME_RST_STREAM: u8 = 0x3;
pub const FRAME_SETTINGS: u8 = 0x4;
pub const FRAME_PUSH_PROMISE: u8 = 0x5;
pub const FRAME_PING: u8 = 0x6;
pub const FRAME_GOAWAY: u8 = 0x7;
pub const FRAME_WINDOW_UPDATE: u8 = 0x8;
pub const FRAME_CONTINUATION: u8 = 0x9;

pub const FLAG_END_STREAM: u8 = 0x1;
pub const FLAG_END_HEADERS: u8 = 0x4;
pub const FLAG_PADDED: u8 = 0x8;
pub const FLAG_PRIORITY: u8 = 0x20;
pub const FLAG_ACK: u8 = 0x1;

pub const SETTINGS_HEADER_TABLE_SIZE: u16 = 0x1;
pub const SETTINGS_ENABLE_PUSH: u16 = 0x2;
pub const SETTINGS_MAX_CONCURRENT_STREAMS: u16 = 0x3;
pub const SETTINGS_INITIAL_WINDOW_SIZE: u16 = 0x4;
pub const SETTINGS_MAX_FRAME_SIZE: u16 = 0x5;
pub const SETTINGS_MAX_HEADER_LIST_SIZE: u16 = 0x6;

/// HTTP/2 connection preface (24 bytes) sent by the client.
pub const CONNECTION_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

// ============================================================================
// Frame codec
// ============================================================================

pub struct Frame {
    pub length: u32,        // 24 bits
    pub frame_type: u8,
    pub flags: u8,
    pub stream_id: u32,     // 31 bits
    pub payload: Vec<u8>,
}

impl Frame {
    pub fn encode(&self, out: &mut Vec<u8>) {
        let len = self.payload.len() as u32;
        assert!(len < (1 << 24), "frame length overflow");
        out.push((len >> 16) as u8);
        out.push((len >> 8) as u8);
        out.push(len as u8);
        out.push(self.frame_type);
        out.push(self.flags);
        out.push(((self.stream_id >> 24) & 0x7f) as u8);
        out.push((self.stream_id >> 16) as u8);
        out.push((self.stream_id >> 8) as u8);
        out.push(self.stream_id as u8);
        out.extend_from_slice(&self.payload);
    }

    /// Parse a single frame.  Returns (frame, bytes_consumed) or None if incomplete.
    pub fn parse(input: &[u8]) -> Option<(Frame, usize)> {
        if input.len() < 9 { return None; }
        let length = ((input[0] as u32) << 16) | ((input[1] as u32) << 8) | (input[2] as u32);
        let frame_type = input[3];
        let flags = input[4];
        let stream_id = (((input[5] & 0x7f) as u32) << 24)
                      | ((input[6] as u32) << 16)
                      | ((input[7] as u32) << 8)
                      | (input[8] as u32);
        let total = 9 + length as usize;
        if input.len() < total { return None; }
        Some((Frame {
            length,
            frame_type,
            flags,
            stream_id,
            payload: input[9..total].to_vec(),
        }, total))
    }
}

// ============================================================================
// HPACK static table (RFC 7541 Appendix A, 61 entries).
// ============================================================================

pub static HPACK_STATIC: &[(&str, &str)] = &[
    (":authority", ""),
    (":method", "GET"),
    (":method", "POST"),
    (":path", "/"),
    (":path", "/index.html"),
    (":scheme", "http"),
    (":scheme", "https"),
    (":status", "200"),
    (":status", "204"),
    (":status", "206"),
    (":status", "304"),
    (":status", "400"),
    (":status", "404"),
    (":status", "500"),
    ("accept-charset", ""),
    ("accept-encoding", "gzip, deflate"),
    ("accept-language", ""),
    ("accept-ranges", ""),
    ("accept", ""),
    ("access-control-allow-origin", ""),
    ("age", ""),
    ("allow", ""),
    ("authorization", ""),
    ("cache-control", ""),
    ("content-disposition", ""),
    ("content-encoding", ""),
    ("content-language", ""),
    ("content-length", ""),
    ("content-location", ""),
    ("content-range", ""),
    ("content-type", ""),
    ("cookie", ""),
    ("date", ""),
    ("etag", ""),
    ("expect", ""),
    ("expires", ""),
    ("from", ""),
    ("host", ""),
    ("if-match", ""),
    ("if-modified-since", ""),
    ("if-none-match", ""),
    ("if-range", ""),
    ("if-unmodified-since", ""),
    ("last-modified", ""),
    ("link", ""),
    ("location", ""),
    ("max-forwards", ""),
    ("proxy-authenticate", ""),
    ("proxy-authorization", ""),
    ("range", ""),
    ("referer", ""),
    ("refresh", ""),
    ("retry-after", ""),
    ("server", ""),
    ("set-cookie", ""),
    ("strict-transport-security", ""),
    ("transfer-encoding", ""),
    ("user-agent", ""),
    ("vary", ""),
    ("via", ""),
    ("www-authenticate", ""),
];

/// HPACK integer decode per RFC 7541 §5.1.
/// `prefix_bits` is N (the number of bits available in the first byte).
/// Returns (value, bytes_consumed) or None.
pub fn hpack_decode_int(input: &[u8], prefix_bits: u8) -> Option<(u64, usize)> {
    if input.is_empty() { return None; }
    let max_prefix = (1u64 << prefix_bits) - 1;
    let first = (input[0] as u64) & max_prefix;
    if first < max_prefix { return Some((first, 1)); }
    let mut value = max_prefix;
    let mut m = 0u32;
    let mut idx = 1;
    while idx < input.len() {
        let b = input[idx];
        value += ((b & 0x7f) as u64) << m;
        idx += 1;
        if (b & 0x80) == 0 { return Some((value, idx)); }
        m += 7;
        if m >= 64 { return None; } // overflow
    }
    None
}

/// HPACK integer encode per §5.1.
pub fn hpack_encode_int(out: &mut Vec<u8>, prefix_bits: u8, prefix_high: u8, value: u64) {
    let max_prefix = (1u64 << prefix_bits) - 1;
    if value < max_prefix {
        out.push(prefix_high | (value as u8));
        return;
    }
    out.push(prefix_high | (max_prefix as u8));
    let mut v = value - max_prefix;
    while v >= 128 {
        out.push(((v & 0x7f) as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

/// HPACK string decode (§5.2).  Returns (decoded_string, bytes_consumed).
pub fn hpack_decode_string(input: &[u8]) -> Option<(Vec<u8>, usize)> {
    if input.is_empty() { return None; }
    let huffman = (input[0] & 0x80) != 0;
    let (length, len_bytes) = hpack_decode_int(input, 7)?;
    let start = len_bytes;
    let end = start + length as usize;
    if end > input.len() { return None; }
    let raw = &input[start..end];
    let s = if huffman {
        huffman_decode(raw)?
    } else {
        raw.to_vec()
    };
    Some((s, end))
}

/// HPACK string encode (raw, no Huffman).  We never advertise Huffman bits on output;
/// we only need to decode it on input for interop with `curl --http2-prior-knowledge`.
pub fn hpack_encode_string(out: &mut Vec<u8>, s: &[u8]) {
    hpack_encode_int(out, 7, 0x00, s.len() as u64);
    out.extend_from_slice(s);
}

// ============================================================================
// HPACK Huffman decoder (RFC 7541 Appendix B).
//
// Static Huffman code for the 256-byte alphabet + EOS.  We hard-code a tree
// walk over a fixed code-length table.  The decoder reads MSB-first.
// ============================================================================

/// (code, code_length) for each symbol 0..=256.  Symbol 256 = EOS (invalid in data).
static HUFFMAN: [(u32, u8); 257] = [
    (0x1ff8, 13), (0x7fffd8, 23), (0xfffffe2, 28), (0xfffffe3, 28), (0xfffffe4, 28),
    (0xfffffe5, 28), (0xfffffe6, 28), (0xfffffe7, 28), (0xfffffe8, 28), (0xffffea, 24),
    (0x3ffffffc, 30), (0xfffffe9, 28), (0xfffffea, 28), (0x3ffffffd, 30), (0xfffffeb, 28),
    (0xfffffec, 28), (0xfffffed, 28), (0xfffffee, 28), (0xfffffef, 28), (0xffffff0, 28),
    (0xffffff1, 28), (0xffffff2, 28), (0x3ffffffe, 30), (0xffffff3, 28), (0xffffff4, 28),
    (0xffffff5, 28), (0xffffff6, 28), (0xffffff7, 28), (0xffffff8, 28), (0xffffff9, 28),
    (0xffffffa, 28), (0xffffffb, 28), (0x14, 6), (0x3f8, 10), (0x3f9, 10), (0xffa, 12),
    (0x1ff9, 13), (0x15, 6), (0xf8, 8), (0x7fa, 11), (0x3fa, 10), (0x3fb, 10), (0xf9, 8),
    (0x7fb, 11), (0xfa, 8), (0x16, 6), (0x17, 6), (0x18, 6), (0x0, 5), (0x1, 5), (0x2, 5),
    (0x19, 6), (0x1a, 6), (0x1b, 6), (0x1c, 6), (0x1d, 6), (0x1e, 6), (0x1f, 6), (0x5c, 7),
    (0xfb, 8), (0x7ffc, 15), (0x20, 6), (0xffb, 12), (0x3fc, 10), (0x1ffa, 13), (0x21, 6),
    (0x5d, 7), (0x5e, 7), (0x5f, 7), (0x60, 7), (0x61, 7), (0x62, 7), (0x63, 7), (0x64, 7),
    (0x65, 7), (0x66, 7), (0x67, 7), (0x68, 7), (0x69, 7), (0x6a, 7), (0x6b, 7), (0x6c, 7),
    (0x6d, 7), (0x6e, 7), (0x6f, 7), (0x70, 7), (0x71, 7), (0x72, 7), (0xfc, 8), (0x73, 7),
    (0xfd, 8), (0x1ffb, 13), (0x7fff0, 19), (0x1ffc, 13), (0x3ffc, 14), (0x22, 6),
    (0x7ffd, 15), (0x3, 5), (0x23, 6), (0x4, 5), (0x24, 6), (0x5, 5), (0x25, 6), (0x26, 6),
    (0x27, 6), (0x6, 5), (0x74, 7), (0x75, 7), (0x28, 6), (0x29, 6), (0x2a, 6), (0x7, 5),
    (0x2b, 6), (0x76, 7), (0x2c, 6), (0x8, 5), (0x9, 5), (0x2d, 6), (0x77, 7), (0x78, 7),
    (0x79, 7), (0x7a, 7), (0x7b, 7), (0x7ffe, 15), (0x7fc, 11), (0x3ffd, 14), (0x1ffd, 13),
    (0xffffffc, 28), (0xfffe6, 20), (0x3fffd2, 22), (0xfffe7, 20), (0xfffe8, 20),
    (0x3fffd3, 22), (0x3fffd4, 22), (0x3fffd5, 22), (0x7fffd9, 23), (0x3fffd6, 22),
    (0x7fffda, 23), (0x7fffdb, 23), (0x7fffdc, 23), (0x7fffdd, 23), (0x7fffde, 23),
    (0xffffeb, 24), (0x7fffdf, 23), (0xffffec, 24), (0xffffed, 24), (0x3fffd7, 22),
    (0x7fffe0, 23), (0xffffee, 24), (0x7fffe1, 23), (0x7fffe2, 23), (0x7fffe3, 23),
    (0x7fffe4, 23), (0x1fffdc, 21), (0x3fffd8, 22), (0x7fffe5, 23), (0x3fffd9, 22),
    (0x7fffe6, 23), (0x7fffe7, 23), (0xffffef, 24), (0x3fffda, 22), (0x1fffdd, 21),
    (0xfffe9, 20), (0x3fffdb, 22), (0x3fffdc, 22), (0x7fffe8, 23), (0x7fffe9, 23),
    (0x1fffde, 21), (0x7fffea, 23), (0x3fffdd, 22), (0x3fffde, 22), (0xfffff0, 24),
    (0x1fffdf, 21), (0x3fffdf, 22), (0x7fffeb, 23), (0x7fffec, 23), (0x1fffe0, 21),
    (0x1fffe1, 21), (0x3fffe0, 22), (0x1fffe2, 21), (0x7fffed, 23), (0x3fffe1, 22),
    (0x7fffee, 23), (0x7fffef, 23), (0xfffea, 20), (0x3fffe2, 22), (0x3fffe3, 22),
    (0x3fffe4, 22), (0x7ffff0, 23), (0x3fffe5, 22), (0x3fffe6, 22), (0x7ffff1, 23),
    (0x3ffffe0, 26), (0x3ffffe1, 26), (0xfffeb, 20), (0x7fff1, 19), (0x3fffe7, 22),
    (0x7ffff2, 23), (0x3fffe8, 22), (0x1ffffec, 25), (0x3ffffe2, 26), (0x3ffffe3, 26),
    (0x3ffffe4, 26), (0x7ffffde, 27), (0x7ffffdf, 27), (0x3ffffe5, 26), (0xfffff1, 24),
    (0x1ffffed, 25), (0x7fff2, 19), (0x1fffe3, 21), (0x3ffffe6, 26), (0x7ffffe0, 27),
    (0x7ffffe1, 27), (0x3ffffe7, 26), (0x7ffffe2, 27), (0xfffff2, 24), (0x1fffe4, 21),
    (0x1fffe5, 21), (0x3ffffe8, 26), (0x3ffffe9, 26), (0xffffffd, 28), (0x7ffffe3, 27),
    (0x7ffffe4, 27), (0x7ffffe5, 27), (0xfffec, 20), (0xfffff3, 24), (0xfffed, 20),
    (0x1fffe6, 21), (0x3fffe9, 22), (0x1fffe7, 21), (0x1fffe8, 21), (0x7ffff3, 23),
    (0x3fffea, 22), (0x3fffeb, 22), (0x1ffffee, 25), (0x1ffffef, 25), (0xfffff4, 24),
    (0xfffff5, 24), (0x3ffffea, 26), (0x7ffff4, 23), (0x3ffffeb, 26), (0x7ffffe6, 27),
    (0x3ffffec, 26), (0x3ffffed, 26), (0x7ffffe7, 27), (0x7ffffe8, 27), (0x7ffffe9, 27),
    (0x7ffffea, 27), (0x7ffffeb, 27), (0xffffffe, 28), (0x7ffffec, 27), (0x7ffffed, 27),
    (0x7ffffee, 27), (0x7ffffef, 27), (0x7fffff0, 27), (0x3ffffee, 26),
    (0x3fffffff, 30), // EOS
];

/// Decode HPACK Huffman bytes.  Implemented as bit-by-bit code matching against
/// the static `HUFFMAN` table.  This is slow (O(input * 257)) but small enough
/// for header sizes typically <1 KB.
fn huffman_decode(src: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(src.len() * 2);
    let mut acc: u64 = 0;
    let mut bits: u32 = 0;
    let mut idx = 0;
    while idx < src.len() || bits >= 5 {
        // Feed bits.
        while bits < 30 && idx < src.len() {
            acc = (acc << 8) | (src[idx] as u64);
            bits += 8;
            idx += 1;
        }
        // Try matching a code from longest down to shortest?  Actually we go
        // shortest-first so we exit early on common chars.
        let mut matched = false;
        for sym in 0..256 {
            let (code, len) = HUFFMAN[sym];
            if (len as u32) <= bits {
                let top = (acc >> (bits - len as u32)) as u32 & ((1u32 << len) - 1);
                if top == code {
                    out.push(sym as u8);
                    bits -= len as u32;
                    acc &= (1u64 << bits) - 1;
                    matched = true;
                    break;
                }
            }
        }
        if !matched {
            // Remaining bits should be all-1 EOS padding (≤7 bits).
            if bits == 0 { return Some(out); }
            if bits <= 7 {
                let mask = (1u64 << bits) - 1;
                if (acc & mask) == mask { return Some(out); }
            }
            return None;
        }
    }
    Some(out)
}

// ============================================================================
// HPACK decode/encode of HEADERS payload
// ============================================================================

/// Decode the HEADERS payload into a list of (name, value) pairs.
/// We DO NOT maintain a dynamic table; literal-with-incremental-indexing
/// entries are accepted but not stored.
pub fn hpack_decode_headers(payload: &[u8]) -> Option<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut out = Vec::new();
    let mut idx = 0;
    while idx < payload.len() {
        let first = payload[idx];
        if (first & 0x80) != 0 {
            // 6.1 Indexed Header Field (1xxxxxxx)
            let (i, n) = hpack_decode_int(&payload[idx..], 7)?;
            idx += n;
            if i == 0 { return None; }
            let i = i as usize;
            if i <= HPACK_STATIC.len() {
                let (n, v) = HPACK_STATIC[i - 1];
                out.push((n.as_bytes().to_vec(), v.as_bytes().to_vec()));
            } else {
                // dynamic table index — not supported; treat as empty.
                return None;
            }
        } else if (first & 0xc0) == 0x40 {
            // 6.2.1 Literal Header Field with Incremental Indexing (01xxxxxx)
            let (i, n) = hpack_decode_int(&payload[idx..], 6)?;
            idx += n;
            let name = if i == 0 {
                let (s, n2) = hpack_decode_string(&payload[idx..])?;
                idx += n2;
                s
            } else if (i as usize) <= HPACK_STATIC.len() {
                HPACK_STATIC[i as usize - 1].0.as_bytes().to_vec()
            } else {
                return None;
            };
            let (value, nv) = hpack_decode_string(&payload[idx..])?;
            idx += nv;
            out.push((name, value));
        } else if (first & 0xe0) == 0x20 {
            // 6.3 Dynamic Table Size Update (001xxxxx) — accept and ignore.
            let (_size, n) = hpack_decode_int(&payload[idx..], 5)?;
            idx += n;
        } else {
            // 6.2.2 Literal without Indexing (0000xxxx)
            // 6.2.3 Literal Never Indexed (0001xxxx)
            let (i, n) = hpack_decode_int(&payload[idx..], 4)?;
            idx += n;
            let name = if i == 0 {
                let (s, n2) = hpack_decode_string(&payload[idx..])?;
                idx += n2;
                s
            } else if (i as usize) <= HPACK_STATIC.len() {
                HPACK_STATIC[i as usize - 1].0.as_bytes().to_vec()
            } else {
                return None;
            };
            let (value, nv) = hpack_decode_string(&payload[idx..])?;
            idx += nv;
            out.push((name, value));
        }
    }
    Some(out)
}

/// Encode HEADERS payload from a list of (name, value) pairs.  Always uses
/// literal-without-indexing form (4-bit prefix) with name & value sent
/// uncompressed.  Status pseudo-header is looked up in the static table.
pub fn hpack_encode_headers(headers: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64);
    for (name, value) in headers {
        // Try static-table lookup by exact (name,value) match first.
        if let Some(idx) = static_index_full_match(name, value) {
            hpack_encode_int(&mut out, 7, 0x80, idx as u64);
            continue;
        }
        // Try name-only match.
        if let Some(idx) = static_index_name_match(name) {
            hpack_encode_int(&mut out, 4, 0x00, idx as u64);
            hpack_encode_string(&mut out, value);
        } else {
            // New name + value.
            hpack_encode_int(&mut out, 4, 0x00, 0);
            hpack_encode_string(&mut out, name);
            hpack_encode_string(&mut out, value);
        }
    }
    out
}

fn static_index_full_match(name: &[u8], value: &[u8]) -> Option<usize> {
    for (i, (n, v)) in HPACK_STATIC.iter().enumerate() {
        if n.as_bytes() == name && v.as_bytes() == value { return Some(i + 1); }
    }
    None
}

fn static_index_name_match(name: &[u8]) -> Option<usize> {
    for (i, (n, _)) in HPACK_STATIC.iter().enumerate() {
        if n.as_bytes() == name { return Some(i + 1); }
    }
    None
}

// ============================================================================
// Helpers for building common server-side frames
// ============================================================================

pub fn build_settings_ack(out: &mut Vec<u8>) {
    Frame { length: 0, frame_type: FRAME_SETTINGS, flags: FLAG_ACK, stream_id: 0, payload: vec![] }.encode(out);
}

pub fn build_settings(out: &mut Vec<u8>, settings: &[(u16, u32)]) {
    let mut payload = Vec::with_capacity(settings.len() * 6);
    for (id, value) in settings {
        payload.extend_from_slice(&id.to_be_bytes());
        payload.extend_from_slice(&value.to_be_bytes());
    }
    Frame { length: payload.len() as u32, frame_type: FRAME_SETTINGS, flags: 0, stream_id: 0, payload }.encode(out);
}

pub fn build_headers(out: &mut Vec<u8>, stream_id: u32, end_stream: bool, hpack_payload: Vec<u8>) {
    let flags = FLAG_END_HEADERS | (if end_stream { FLAG_END_STREAM } else { 0 });
    Frame { length: hpack_payload.len() as u32, frame_type: FRAME_HEADERS, flags, stream_id, payload: hpack_payload }.encode(out);
}

pub fn build_data(out: &mut Vec<u8>, stream_id: u32, end_stream: bool, data: Vec<u8>) {
    let flags = if end_stream { FLAG_END_STREAM } else { 0 };
    Frame { length: data.len() as u32, frame_type: FRAME_DATA, flags, stream_id, payload: data }.encode(out);
}

pub fn build_goaway(out: &mut Vec<u8>, last_stream_id: u32, error_code: u32) {
    let mut payload = Vec::with_capacity(8);
    payload.extend_from_slice(&last_stream_id.to_be_bytes());
    payload.extend_from_slice(&error_code.to_be_bytes());
    Frame { length: 8, frame_type: FRAME_GOAWAY, flags: 0, stream_id: 0, payload }.encode(out);
}

pub fn build_ping_ack(out: &mut Vec<u8>, opaque: &[u8; 8]) {
    Frame { length: 8, frame_type: FRAME_PING, flags: FLAG_ACK, stream_id: 0, payload: opaque.to_vec() }.encode(out);
}

pub fn build_window_update(out: &mut Vec<u8>, stream_id: u32, increment: u32) {
    let mut payload = Vec::with_capacity(4);
    payload.extend_from_slice(&increment.to_be_bytes());
    Frame { length: 4, frame_type: FRAME_WINDOW_UPDATE, flags: 0, stream_id, payload }.encode(out);
}

// ============================================================================
// One-shot HTTP/2 request -> response loop, for testing.
//
// Drives a server-side state machine where each open stream may receive
// HEADERS (and optional DATA) from the client; the server responds with
// HEADERS + DATA. No multiplexing semantics enforced beyond what's needed
// for one request per stream.
// ============================================================================

pub struct H2Request {
    pub stream_id: u32,
    pub method: Vec<u8>,
    pub path: Vec<u8>,
    pub authority: Vec<u8>,
    pub body: Vec<u8>,
    pub headers: Vec<(Vec<u8>, Vec<u8>)>,
}

pub struct H2Response {
    pub status: u16,
    pub body: Vec<u8>,
    pub content_type: Vec<u8>,
}

/// Find pseudo-headers and build an H2Request from a HEADERS payload.
pub fn build_h2_request(stream_id: u32, header_list: Vec<(Vec<u8>, Vec<u8>)>) -> H2Request {
    let mut method = Vec::new();
    let mut path = Vec::new();
    let mut authority = Vec::new();
    let mut headers = Vec::new();
    for (n, v) in header_list {
        match n.as_slice() {
            b":method" => method = v,
            b":path" => path = v,
            b":authority" => authority = v,
            b":scheme" => {}
            _ => headers.push((n, v)),
        }
    }
    H2Request { stream_id, method, path, authority, body: Vec::new(), headers }
}

/// Encode a response (HEADERS + DATA frames) for the given stream.
pub fn write_h2_response(out: &mut Vec<u8>, stream_id: u32, resp: &H2Response) {
    let status_str = resp.status.to_string();
    let cl = resp.body.len().to_string();
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b":status", status_str.as_bytes()),
        (b"content-type", &resp.content_type),
        (b"content-length", cl.as_bytes()),
    ];
    let hpack_payload = hpack_encode_headers(&pairs);
    build_headers(out, stream_id, false, hpack_payload);
    build_data(out, stream_id, true, resp.body.clone());
}

// ============================================================================
// FFI surface
// ============================================================================

/// Encode `count` frames into `out`.  Simple smoke entry for the witness.
#[no_mangle] pub unsafe extern "C" fn aether_http2_self_loopback_smoke() -> c_int {
    // Build a SETTINGS frame, parse it back, verify fields.
    let mut buf = Vec::new();
    build_settings(&mut buf, &[
        (SETTINGS_MAX_CONCURRENT_STREAMS, 100),
        (SETTINGS_INITIAL_WINDOW_SIZE, 65535),
    ]);
    let (f, n) = match Frame::parse(&buf) { Some(t) => t, None => return 1, };
    if n != buf.len() || f.frame_type != FRAME_SETTINGS || f.payload.len() != 12 { return 2; }

    // HPACK round-trip a small response header set.
    let pairs: Vec<(&[u8], &[u8])> = vec![
        (b":status", b"200"),
        (b"content-type", b"application/json"),
        (b"content-length", b"3"),
    ];
    let enc = hpack_encode_headers(&pairs);
    let dec = match hpack_decode_headers(&enc) { Some(d) => d, None => return 3, };
    if dec.len() != 3 { return 4; }
    if dec[0].0 != b":status" || dec[0].1 != b"200" { return 5; }
    if dec[1].0 != b"content-type" || dec[1].1 != b"application/json" { return 6; }
    if dec[2].0 != b"content-length" || dec[2].1 != b"3" { return 7; }

    // Huffman: decode a known-good encoded string for "www.example.com" (RFC 7541 §C.4.1).
    // Encoded bytes: f1 e3 c2 e5 f2 3a 6b a0 ab 90 f4 ff
    let huff = &[0xf1u8, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff];
    let decoded = match huffman_decode(huff) { Some(d) => d, None => return 8, };
    if decoded != b"www.example.com" { return 9; }

    42
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_encode_decode_roundtrip() {
        let f = Frame {
            length: 4, frame_type: FRAME_WINDOW_UPDATE, flags: 0, stream_id: 7,
            payload: vec![0, 0, 0xff, 0xff],
        };
        let mut buf = Vec::new();
        f.encode(&mut buf);
        assert_eq!(buf.len(), 9 + 4);
        let (g, n) = Frame::parse(&buf).unwrap();
        assert_eq!(n, buf.len());
        assert_eq!(g.frame_type, FRAME_WINDOW_UPDATE);
        assert_eq!(g.stream_id, 7);
        assert_eq!(g.payload, vec![0, 0, 0xff, 0xff]);
    }

    #[test]
    fn hpack_int_codec() {
        // RFC 7541 §C.1 examples.
        let mut out = Vec::new();
        hpack_encode_int(&mut out, 5, 0x00, 10);
        assert_eq!(out, vec![10]);
        let (v, n) = hpack_decode_int(&out, 5).unwrap();
        assert_eq!((v, n), (10, 1));

        let mut out = Vec::new();
        hpack_encode_int(&mut out, 5, 0x00, 1337);
        assert_eq!(out, vec![31, 154, 10]);
        let (v, n) = hpack_decode_int(&out, 5).unwrap();
        assert_eq!((v, n), (1337, 3));
    }

    #[test]
    fn huffman_decode_www_example_com() {
        // RFC 7541 §C.4.1
        let enc: &[u8] = &[0xf1, 0xe3, 0xc2, 0xe5, 0xf2, 0x3a, 0x6b, 0xa0, 0xab, 0x90, 0xf4, 0xff];
        let dec = huffman_decode(enc).unwrap();
        assert_eq!(dec, b"www.example.com");
    }

    #[test]
    fn hpack_decode_indexed_static() {
        // ":method GET" via 6.1 indexed (idx 2).
        let payload = vec![0x82];
        let decoded = hpack_decode_headers(&payload).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].0, b":method");
        assert_eq!(decoded[0].1, b"GET");
    }

    #[test]
    fn hpack_round_trip_request_headers() {
        let pairs: Vec<(&[u8], &[u8])> = vec![
            (b":method", b"POST"),
            (b":path", b"/v1/chat/completions"),
            (b":scheme", b"https"),
            (b":authority", b"localhost:8443"),
            (b"content-type", b"application/json"),
            (b"content-length", b"123"),
        ];
        let enc = hpack_encode_headers(&pairs);
        let dec = hpack_decode_headers(&enc).unwrap();
        assert_eq!(dec.len(), pairs.len());
        for (i, (n, v)) in pairs.iter().enumerate() {
            assert_eq!(dec[i].0, *n, "name mismatch at {}", i);
            assert_eq!(dec[i].1, *v, "value mismatch at {}", i);
        }
    }

    #[test]
    fn smoke_returns_42() {
        unsafe { assert_eq!(aether_http2_self_loopback_smoke(), 42); }
    }
}
