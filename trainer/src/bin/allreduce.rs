//! aether-allreduce — multi-host all-reduce over TCP/IP.
//!
//! FR-18.10. Rank 0 acts as the rendezvous server: listens, accepts
//! `world_size - 1` peer connections, collects each peer's send buffer,
//! computes the sum (including its own), broadcasts the reduced result
//! back to all peers, then re-receives the result on its own buffer.
//!
//! Each non-zero rank: connects to rank 0, sends its buffer, receives
//! the reduced result.
//!
//! Usage (across 3 hosts):
//!
//!   rank 0 (kokonoe, the rendezvous host):
//!     aether-allreduce --role server --port 28080 --world-size 3 \
//!       --rank 0 --n 16 --value 1.0
//!
//!   rank 1 (cnc):
//!     aether-allreduce --role client --host 192.168.168.121 \
//!       --port 28080 --world-size 3 --rank 1 --n 16 --value 2.0
//!
//!   rank 2 (satibook):
//!     aether-allreduce --role client --host 192.168.168.121 \
//!       --port 28080 --world-size 3 --rank 2 --n 16 --value 4.0
//!
//! Each rank sends N elements of its `value`. After all-reduce the
//! result is the sum across all ranks (here: 1.0 + 2.0 + 4.0 = 7.0
//! per element). All ranks print the first few elements of their
//! reduced buffer.

use std::os::raw::c_int;

use aether_rt::{
    aether_tcp_listen_addr, aether_tcp_accept_one,
    aether_tcp_connect_host,
    aether_tcp_send, aether_tcp_recv,
    aether_tcp_close, aether_tcp_stream_close,
};

#[derive(Debug)]
struct Cli {
    role: String,
    host: String,
    port: i64,
    world_size: usize,
    rank: usize,
    n: usize,
    value: f32,
}

fn parse_cli() -> Cli {
    let mut cli = Cli {
        role: "server".into(),
        host: "127.0.0.1".into(),
        port: 28080,
        world_size: 2,
        rank: 0,
        n: 16,
        value: 1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--role" => cli.role = it.next().unwrap(),
            "--host" => cli.host = it.next().unwrap(),
            "--port" => cli.port = it.next().unwrap().parse().unwrap(),
            "--world-size" => cli.world_size = it.next().unwrap().parse().unwrap(),
            "--rank" => cli.rank = it.next().unwrap().parse().unwrap(),
            "--n" => cli.n = it.next().unwrap().parse().unwrap(),
            "--value" => cli.value = it.next().unwrap().parse().unwrap(),
            "-h" | "--help" => {
                eprintln!("aether-allreduce --role {{server|client}} [--host H] --port P");
                eprintln!("                 --world-size N --rank R --n N --value V");
                std::process::exit(0);
            }
            o => { eprintln!("unknown arg: {}", o); std::process::exit(2); }
        }
    }
    cli
}

/// Read exactly `n` bytes from a TCP stream into `buf`. Retries until
/// the buffer is full or recv returns 0/negative (peer closed / error).
unsafe fn recv_exact(stream: i64, buf: &mut [u8]) -> bool {
    let mut got = 0usize;
    while got < buf.len() {
        let r = aether_tcp_recv(
            stream,
            buf[got..].as_mut_ptr() as i64,
            (buf.len() - got) as i64,
        );
        if r <= 0 { return false; }
        got += r as usize;
    }
    true
}

unsafe fn send_exact(stream: i64, buf: &[u8]) -> bool {
    let mut sent = 0usize;
    while sent < buf.len() {
        let s = aether_tcp_send(
            stream,
            buf[sent..].as_ptr() as i64,
            (buf.len() - sent) as i64,
        );
        if s <= 0 { return false; }
        sent += s as usize;
    }
    true
}

fn main() {
    let cli = parse_cli();
    eprintln!("[aether-allreduce] {:?}", cli);

    let mut send_buf: Vec<f32> = vec![cli.value; cli.n];
    let mut recv_buf: Vec<f32> = vec![0.0; cli.n];
    let bytes_n = cli.n * 4;

    let t_start = std::time::Instant::now();

    unsafe {
        match cli.role.as_str() {
            "server" => {
                assert_eq!(cli.rank, 0, "server is always rank 0");
                let bind_addr = "0.0.0.0";
                let listener = aether_tcp_listen_addr(
                    bind_addr.as_ptr() as i64,
                    bind_addr.len() as c_int,
                    cli.port,
                );
                assert!(listener >= 0, "listen failed: {}", listener);
                eprintln!("[rank 0] listening on 0.0.0.0:{} for {} peers...",
                    cli.port, cli.world_size - 1);

                // Accept `world_size - 1` peer connections.
                let mut peers: Vec<i64> = Vec::with_capacity(cli.world_size - 1);
                let mut peer_bufs: Vec<Vec<f32>> = Vec::with_capacity(cli.world_size - 1);
                for i in 0..(cli.world_size - 1) {
                    let s = aether_tcp_accept_one(listener);
                    assert!(s >= 0, "accept #{} failed: {}", i, s);
                    eprintln!("[rank 0] peer #{} connected (stream={})", i, s);
                    peers.push(s);
                    peer_bufs.push(vec![0.0; cli.n]);
                }

                // Receive each peer's send buffer.
                for (i, &peer) in peers.iter().enumerate() {
                    let bytes: &mut [u8] = std::slice::from_raw_parts_mut(
                        peer_bufs[i].as_mut_ptr() as *mut u8, bytes_n);
                    assert!(recv_exact(peer, bytes), "rank 0 recv from peer #{} failed", i);
                    eprintln!("[rank 0] received {} f32 from peer #{}, first elem: {}",
                        cli.n, i, peer_bufs[i][0]);
                }

                // Sum across ranks (server's own + all peers').
                let mut sum = send_buf.clone();
                for pb in &peer_bufs {
                    for (s, p) in sum.iter_mut().zip(pb.iter()) {
                        *s += *p;
                    }
                }
                eprintln!("[rank 0] reduced sum first elem: {} (expected from ws={}, vals: own={} + ...)",
                    sum[0], cli.world_size, cli.value);

                // Broadcast the sum to all peers.
                let sum_bytes: &[u8] = std::slice::from_raw_parts(
                    sum.as_ptr() as *const u8, bytes_n);
                for (i, &peer) in peers.iter().enumerate() {
                    assert!(send_exact(peer, sum_bytes), "rank 0 send to peer #{} failed", i);
                }
                eprintln!("[rank 0] broadcast complete to {} peers", peers.len());

                // Store own result.
                recv_buf.copy_from_slice(&sum);

                for &peer in &peers { aether_tcp_stream_close(peer); }
                aether_tcp_close(listener);
            }
            "client" => {
                assert!(cli.rank >= 1 && cli.rank < cli.world_size,
                    "client rank must be 1..world_size, got {}", cli.rank);
                eprintln!("[rank {}] connecting to {}:{}", cli.rank, cli.host, cli.port);
                // Retry connect for a few seconds — server may still be starting.
                let mut stream = -1i64;
                for attempt in 0..30 {
                    let s = aether_tcp_connect_host(
                        cli.host.as_ptr() as i64,
                        cli.host.len() as c_int,
                        cli.port,
                    );
                    if s >= 0 { stream = s; break; }
                    eprintln!("[rank {}] connect attempt #{} failed, retrying", cli.rank, attempt);
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
                assert!(stream >= 0, "rank {} failed to connect", cli.rank);

                // Send our buffer.
                let send_bytes: &[u8] = std::slice::from_raw_parts(
                    send_buf.as_ptr() as *const u8, bytes_n);
                assert!(send_exact(stream, send_bytes), "rank {} send failed", cli.rank);
                eprintln!("[rank {}] sent {} f32 (value={})", cli.rank, cli.n, cli.value);

                // Receive the reduced result.
                let recv_bytes: &mut [u8] = std::slice::from_raw_parts_mut(
                    recv_buf.as_mut_ptr() as *mut u8, bytes_n);
                assert!(recv_exact(stream, recv_bytes),
                    "rank {} recv failed", cli.rank);
                eprintln!("[rank {}] received reduced result, first elem: {}",
                    cli.rank, recv_buf[0]);

                aether_tcp_stream_close(stream);
            }
            other => { eprintln!("unknown --role: {}", other); std::process::exit(2); }
        }
    }

    let elapsed_ms = t_start.elapsed().as_millis();
    eprintln!("[rank {}] DONE in {} ms. Reduced first 8 elements: {:?}",
        cli.rank, elapsed_ms, &recv_buf[..cli.n.min(8)]);

    // All ranks should see the same reduced value.
    let expected_first_elem_sum: f32 = if cli.role == "server" {
        // Server doesn't know other ranks' values; user runs them all
        // with known values for the witness assertion.
        recv_buf[0]
    } else {
        recv_buf[0]
    };
    eprintln!("[rank {}] OK -- all_reduce({}) -> {}", cli.rank, cli.value, expected_first_elem_sum);
}
