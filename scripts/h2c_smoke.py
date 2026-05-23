"""Minimal h2c client to smoke-test aether-serve's HTTP/2 server-side wire.

No external h2 library — implements the bare minimum: connection preface,
SETTINGS exchange, HEADERS frame using HPACK static-table indexed name +
literal value form, read the server's HEADERS + DATA frames, decode the
small static-indexed :status header.

Usage:
    python h2c_smoke.py <port> <method> <path>
"""
import socket
import struct
import sys


def encode_int(prefix_bits, prefix_hi, value):
    max_pfx = (1 << prefix_bits) - 1
    if value < max_pfx:
        return bytes([prefix_hi | value])
    out = bytes([prefix_hi | max_pfx])
    v = value - max_pfx
    while v >= 128:
        out += bytes([(v & 0x7f) | 0x80])
        v >>= 7
    out += bytes([v])
    return out


def encode_string(s):
    b = s.encode("utf-8")
    return encode_int(7, 0x00, len(b)) + b


def encode_literal_no_index(name_idx, value):
    """6.2.2 — literal w/o indexing.  `name_idx` is a static-table index;
    `value` is raw."""
    return encode_int(4, 0x00, name_idx) + encode_string(value)


def encode_headers(method, path):
    # Use static-table indices for names; literal values for path/authority.
    # :method GET=2, POST=3 (static-indexed if matches; else literal-with-name).
    out = b""
    if method == "GET":
        out += bytes([0x82])  # :method GET
    elif method == "POST":
        out += bytes([0x83])  # :method POST
    else:
        out += encode_literal_no_index(2, method)
    out += bytes([0x86])  # :scheme http (=6)
    out += encode_literal_no_index(1, "localhost")  # :authority -> idx 1
    if path == "/":
        out += bytes([0x84])  # :path /
    else:
        out += encode_literal_no_index(4, path)
    return out


def build_frame(frame_type, flags, stream_id, payload):
    length = len(payload)
    header = bytes([
        (length >> 16) & 0xff, (length >> 8) & 0xff, length & 0xff,
        frame_type, flags,
        (stream_id >> 24) & 0x7f, (stream_id >> 16) & 0xff,
        (stream_id >> 8) & 0xff, stream_id & 0xff,
    ])
    return header + payload


def read_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_frame(sock):
    hdr = read_exact(sock, 9)
    if hdr is None:
        return None
    length = (hdr[0] << 16) | (hdr[1] << 8) | hdr[2]
    ft = hdr[3]
    flags = hdr[4]
    sid = ((hdr[5] & 0x7f) << 24) | (hdr[6] << 16) | (hdr[7] << 8) | hdr[8]
    payload = read_exact(sock, length) if length > 0 else b""
    return ft, flags, sid, payload


def decode_int(buf, prefix_bits):
    max_pfx = (1 << prefix_bits) - 1
    first = buf[0] & max_pfx
    if first < max_pfx:
        return first, 1
    value = max_pfx
    m = 0
    idx = 1
    while idx < len(buf):
        b = buf[idx]
        value += (b & 0x7f) << m
        idx += 1
        if (b & 0x80) == 0:
            return value, idx
        m += 7
    raise ValueError("hpack int overflow")


def decode_string(buf):
    huff = (buf[0] & 0x80) != 0
    n, consumed = decode_int(buf, 7)
    s = buf[consumed:consumed + n]
    consumed += n
    if huff:
        # We don't bother decoding Huffman in this smoke test.
        return f"<huff {n} bytes>", consumed
    return s.decode("utf-8", errors="replace"), consumed


def decode_headers(payload):
    out = []
    i = 0
    STATIC = [
        (":authority", ""), (":method", "GET"), (":method", "POST"),
        (":path", "/"), (":path", "/index.html"),
        (":scheme", "http"), (":scheme", "https"),
        (":status", "200"), (":status", "204"), (":status", "206"),
        (":status", "304"), (":status", "400"), (":status", "404"),
        (":status", "500"),
    ]
    while i < len(payload):
        b = payload[i]
        if b & 0x80:
            idx, c = decode_int(payload[i:], 7)
            i += c
            if 1 <= idx <= len(STATIC):
                out.append(STATIC[idx - 1])
            else:
                out.append((f"<dyn {idx}>", ""))
        elif (b & 0xc0) == 0x40:
            idx, c = decode_int(payload[i:], 6)
            i += c
            if idx == 0:
                name, cc = decode_string(payload[i:]); i += cc
            elif 1 <= idx <= len(STATIC):
                name = STATIC[idx - 1][0]
            else:
                name = f"<dyn {idx}>"
            value, cv = decode_string(payload[i:]); i += cv
            out.append((name, value))
        elif (b & 0xe0) == 0x20:
            _size, c = decode_int(payload[i:], 5); i += c
        else:
            idx, c = decode_int(payload[i:], 4)
            i += c
            if idx == 0:
                name, cc = decode_string(payload[i:]); i += cc
            elif 1 <= idx <= len(STATIC):
                name = STATIC[idx - 1][0]
            else:
                name = f"<dyn {idx}>"
            value, cv = decode_string(payload[i:]); i += cv
            out.append((name, value))
    return out


def main():
    port = int(sys.argv[1])
    method = sys.argv[2]
    path = sys.argv[3]
    body = sys.argv[4].encode("utf-8") if len(sys.argv) > 4 else b""

    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    # 1. Send preface.
    s.sendall(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
    # 2. Send our SETTINGS (empty).
    s.sendall(build_frame(0x4, 0, 0, b""))

    # 3. Build HEADERS frame.
    hpack = encode_headers(method, path)
    flags = 0x04  # END_HEADERS
    if not body:
        flags |= 0x01  # END_STREAM
    s.sendall(build_frame(0x1, flags, 1, hpack))
    if body:
        s.sendall(build_frame(0x0, 0x01, 1, body))  # DATA END_STREAM

    # 4. Read frames until we have HEADERS+DATA for stream 1.
    got_headers = False
    while True:
        frame = parse_frame(s)
        if frame is None:
            break
        ft, fl, sid, payload = frame
        if ft == 0x4 and (fl & 0x1) == 0:
            # SETTINGS from server -> ACK.
            s.sendall(build_frame(0x4, 0x1, 0, b""))
        elif ft == 0x1 and sid == 1:
            hs = decode_headers(payload)
            print("HEADERS:", hs)
            got_headers = True
        elif ft == 0x0 and sid == 1:
            print(f"DATA ({len(payload)} bytes):", payload[:200])
            if fl & 0x1:
                break  # END_STREAM
        elif ft == 0x7:
            print("GOAWAY:", payload.hex())
            break

    s.close()


if __name__ == "__main__":
    main()
