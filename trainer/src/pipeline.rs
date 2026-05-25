//! Pipeline parallelism (1F1B) — FR-18.6-real, the matt-voice multi-GPU unlock.
//!
//! roadmap: P18
//!
//! Splits a model's layers across `world_size` ranks arranged as a linear
//! chain (rank 0 → 1 → ... → N-1). Each rank owns a contiguous slice of the
//! layers and runs a stage. Activations flow downstream (rank r → r+1) during
//! forward; gradients flow upstream (rank r+1 → r) during backward. The
//! schedule is the classic **1F1B** (one-forward-one-backward): each rank does
//! `warmup = (world_size-1-rank)` forwards, then interleaves forward+backward
//! through the steady state, then drains `warmup` backwards in cooldown. This
//! bounds in-flight activation memory to `warmup+1` microbatches per stage —
//! the property that lets a model too big for one GPU train across several.
//!
//! This module is the **transport + schedule machinery**, device- and
//! model-agnostic. It is proven here on a CPU f32 `LinearReluStack` whose
//! every matmul/relu/optimizer call goes through the real `aether_rt::ops`
//! surface, with a two-rank localhost-TCP witness asserting that pipelined
//! per-microbatch losses AND post-step parameters are bit-identical to a
//! single-process reference running the same layers. Grafting it onto the
//! GPU QwenSession block-forward (rank 0 = layers 0..31, rank 1 = 32..63 for
//! Qwen3-32B) reuses this exact scheduler — only the `Stage` impl changes.
//!
//! Transport reuses the cross-host TCP primitives proven by `aether-allreduce`
//! (`aether_tcp_*` in `aether_rt`): no NCCL send/recv needed, so the same
//! chain works single-host (two GPUs) or cross-host (the cnc 2×P100 + kokonoe
//! pool) without a code change. A rank stages each tensor d2h→TCP→h2d when its
//! `Stage` is on a GPU; the CPU witness stage skips the device hop.

use std::collections::VecDeque;
use std::os::raw::c_int;

use aether_rt::{
    aether_tcp_listen_addr, aether_tcp_accept_one, aether_tcp_connect_host,
    aether_tcp_send, aether_tcp_recv, aether_tcp_close, aether_tcp_stream_close,
};

// ---------------------------------------------------------------- TCP transport

/// Read exactly `buf.len()` bytes, retrying short reads. False on peer close.
unsafe fn recv_exact(stream: i64, buf: &mut [u8]) -> bool {
    let mut got = 0usize;
    while got < buf.len() {
        let r = aether_tcp_recv(stream, buf[got..].as_mut_ptr() as i64, (buf.len() - got) as i64);
        if r <= 0 { return false; }
        got += r as usize;
    }
    true
}

/// Write exactly `buf.len()` bytes, retrying short writes.
unsafe fn send_exact(stream: i64, buf: &[u8]) -> bool {
    let mut sent = 0usize;
    while sent < buf.len() {
        let s = aether_tcp_send(stream, buf[sent..].as_ptr() as i64, (buf.len() - sent) as i64);
        if s <= 0 { return false; }
        sent += s as usize;
    }
    true
}

/// Send a fixed-length f32 tensor. Both peers know the length from the model
/// shape, so no length prefix is needed (matches `aether-allreduce`).
fn send_tensor(stream: i64, t: &[f32]) {
    let bytes = unsafe { std::slice::from_raw_parts(t.as_ptr() as *const u8, t.len() * 4) };
    let ok = unsafe { send_exact(stream, bytes) };
    assert!(ok, "[pp] send_tensor failed ({} f32)", t.len());
}

/// Receive a fixed-length f32 tensor of `n` elements.
fn recv_tensor(stream: i64, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    let bytes = unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, n * 4) };
    let ok = unsafe { recv_exact(stream, bytes) };
    assert!(ok, "[pp] recv_tensor failed ({} f32)", n);
    v
}

// ---------------------------------------------------------------- rendezvous

/// Streams linking a rank to its pipeline neighbours.
pub struct PipeLinks {
    /// Stream to rank-1 (upstream); `None` on the first rank.
    pub up: Option<i64>,
    /// Stream to rank+1 (downstream); `None` on the last rank.
    pub down: Option<i64>,
    listener: Option<i64>,
}

impl PipeLinks {
    /// Links for a single-process run (`world_size == 1`): no upstream or
    /// downstream neighbour. run_1f1b takes the is_first && is_last path and
    /// never touches the (absent) streams. Useful for driving a `Stage` through
    /// the 1F1B schedule on one device without TCP.
    pub fn local_single() -> Self {
        PipeLinks { up: None, down: None, listener: None }
    }
}

impl Drop for PipeLinks {
    fn drop(&mut self) {
        unsafe {
            if let Some(s) = self.up { aether_tcp_stream_close(s); }
            if let Some(s) = self.down { aether_tcp_stream_close(s); }
            if let Some(l) = self.listener { aether_tcp_close(l); }
        }
    }
}

/// Establish the linear-chain TCP links for `rank` in `world_size`.
///
/// Convention: rank `r` (for r > 0) listens on `base_port + r` and accepts a
/// connection from its upstream neighbour `r-1`. Rank `r` (for r < ws-1)
/// connects to `base_port + (r+1)` (its downstream neighbour's listener).
/// We bind the listener, then connect downstream (with retry), then accept the
/// upstream connection — connect-before-accept keeps a long chain deadlock-free
/// because TCP queues the SYN on the bound listener before `accept` is called.
pub fn connect_pipeline(rank: usize, world_size: usize, base_port: i64, host: &str) -> PipeLinks {
    assert!(rank < world_size, "rank {} >= world_size {}", rank, world_size);
    let is_first = rank == 0;
    let is_last = rank + 1 == world_size;

    // 1. Bind our own listener first (upstream neighbour connects here).
    let listener = if !is_first {
        let addr = "0.0.0.0";
        let l = unsafe {
            aether_tcp_listen_addr(addr.as_ptr() as i64, addr.len() as c_int, base_port + rank as i64)
        };
        assert!(l >= 0, "[pp rank {}] listen on {} failed: {}", rank, base_port + rank as i64, l);
        eprintln!("[pp rank {}] listening on 0.0.0.0:{}", rank, base_port + rank as i64);
        Some(l)
    } else { None };

    // 2. Connect downstream (retry — the neighbour may still be binding).
    let down = if !is_last {
        let dport = base_port + (rank + 1) as i64;
        let mut s = -1i64;
        for attempt in 0..60 {
            let c = unsafe {
                aether_tcp_connect_host(host.as_ptr() as i64, host.len() as c_int, dport)
            };
            if c >= 0 { s = c; break; }
            if attempt == 0 {
                eprintln!("[pp rank {}] connect→{}:{} pending, retrying", rank, host, dport);
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        assert!(s >= 0, "[pp rank {}] connect→{}:{} failed", rank, host, dport);
        eprintln!("[pp rank {}] connected downstream → rank {} (:{}, stream={})", rank, rank + 1, dport, s);
        Some(s)
    } else { None };

    // 3. Accept the upstream connection.
    let up = if let Some(l) = listener {
        let s = unsafe { aether_tcp_accept_one(l) };
        assert!(s >= 0, "[pp rank {}] accept from rank {} failed: {}", rank, rank - 1, s);
        eprintln!("[pp rank {}] accepted upstream ← rank {} (stream={})", rank, rank - 1, s);
        Some(s)
    } else { None };

    PipeLinks { up, down, listener }
}

// ---------------------------------------------------------------- Stage trait

/// One pipeline stage: a contiguous slice of the model's layers on one rank.
///
/// `forward` consumes the stage input (the residual stream arriving from
/// upstream, or the embedding on rank 0) and returns the activation to send
/// downstream. It MUST internally save whatever it needs for the matching
/// backward, FIFO-ordered: the 1F1B schedule guarantees backwards arrive in
/// the same order forwards were issued on a given stage.
///
/// `backward` consumes the upstream gradient (`grad` w.r.t. this stage's
/// output) and returns the gradient w.r.t. this stage's input, to send
/// upstream. It pops the oldest saved-forward context.
///
/// `step` applies the optimizer to the accumulated parameter gradients and
/// zeroes them — called once per global batch after all microbatches drain.
pub trait Stage {
    fn forward(&mut self, input: &[f32]) -> Vec<f32>;
    fn backward(&mut self, grad: &[f32]) -> Vec<f32>;
    fn step(&mut self, lr: f32, opt_step: i64);
    /// Element count of the activation this stage emits (sent downstream).
    fn output_dim(&self) -> usize;
    /// Element count of the activation this stage consumes (recv'd upstream).
    fn input_dim(&self) -> usize;
}

// ---------------------------------------------------------------- 1F1B driver

/// Run one global batch of `num_microbatches` through the 1F1B schedule on this
/// rank. Returns the per-microbatch losses (only on the last rank; empty
/// elsewhere). After draining, applies the optimizer step.
///
/// * `input_for(mb)` — supplies microbatch `mb`'s input; used only on rank 0.
/// * `loss_and_grad(mb, output)` — computes (loss, dL/doutput) for the final
///   stage's output; used only on the last rank.
pub fn run_1f1b<S, FIn, FLoss>(
    stage: &mut S,
    rank: usize,
    world_size: usize,
    links: &PipeLinks,
    num_microbatches: usize,
    lr: f32,
    opt_step: i64,
    mut input_for: FIn,
    mut loss_and_grad: FLoss,
) -> Vec<f32>
where
    S: Stage,
    FIn: FnMut(usize) -> Vec<f32>,
    FLoss: FnMut(usize, &[f32]) -> (f32, Vec<f32>),
{
    let is_first = rank == 0;
    let is_last = rank + 1 == world_size;
    let in_dim = stage.input_dim();
    let out_dim = stage.output_dim();

    // The last rank stashes each forward's output FIFO so the matching backward
    // can compute the loss gradient against it.
    let mut last_outputs: VecDeque<Vec<f32>> = VecDeque::new();
    let mut losses: Vec<f32> = Vec::new();
    let mut fwd_count = 0usize;
    let mut bwd_count = 0usize;

    let mut do_fwd = |stage: &mut S,
                      fwd_count: &mut usize,
                      last_outputs: &mut VecDeque<Vec<f32>>| {
        let mb = *fwd_count;
        *fwd_count += 1;
        let input = if is_first { input_for(mb) } else { recv_tensor(links.up.unwrap(), in_dim) };
        let out = stage.forward(&input);
        if is_last {
            last_outputs.push_back(out);
        } else {
            send_tensor(links.down.unwrap(), &out);
        }
    };

    let mut do_bwd = |stage: &mut S,
                      bwd_count: &mut usize,
                      last_outputs: &mut VecDeque<Vec<f32>>,
                      losses: &mut Vec<f32>| {
        let mb = *bwd_count;
        *bwd_count += 1;
        let grad_out = if is_last {
            let y = last_outputs.pop_front().expect("[pp] last rank: no stashed output");
            let (loss, dy) = loss_and_grad(mb, &y);
            losses.push(loss);
            dy
        } else {
            recv_tensor(links.down.unwrap(), out_dim)
        };
        let grad_in = stage.backward(&grad_out);
        if !is_first {
            send_tensor(links.up.unwrap(), &grad_in);
        }
    };

    // 1F1B: warmup forwards, steady (1F1B) pairs, cooldown backwards.
    let warmup = (world_size - 1 - rank).min(num_microbatches);
    let steady = num_microbatches - warmup;
    eprintln!(
        "[pp rank {}] 1F1B schedule: warmup={} steady={} cooldown={} (mb={})",
        rank, warmup, steady, warmup, num_microbatches
    );

    for _ in 0..warmup {
        do_fwd(stage, &mut fwd_count, &mut last_outputs);
    }
    for _ in 0..steady {
        do_fwd(stage, &mut fwd_count, &mut last_outputs);
        do_bwd(stage, &mut bwd_count, &mut last_outputs, &mut losses);
    }
    for _ in 0..warmup {
        do_bwd(stage, &mut bwd_count, &mut last_outputs, &mut losses);
    }

    debug_assert_eq!(fwd_count, num_microbatches);
    debug_assert_eq!(bwd_count, num_microbatches);

    stage.step(lr, opt_step);
    losses
}

// ---------------------------------------------------------------- witness stage

/// A stack of `Linear → ReLU` layers over a single sample vector, all of width
/// `d`. Used purely to prove the pipeline machinery: every matmul, relu, and
/// optimizer call routes through `aether_rt::ops`, so a passing parity witness
/// exercises the real op surface, not a toy. Stores weights transposed
/// (`wt: [in, out]`) so the forward is a plain `[1,in] @ [in,out]` matmul.
pub struct LinearReluStack {
    d: usize,
    /// Per layer: (wt[in*out], b[out]) — here in == out == d.
    wt: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    g_wt: Vec<Vec<f32>>,
    g_b: Vec<Vec<f32>>,
    m_wt: Vec<Vec<f32>>, v_wt: Vec<Vec<f32>>,
    m_b: Vec<Vec<f32>>,  v_b: Vec<Vec<f32>>,
    /// FIFO of saved forward contexts: per microbatch, per layer (x_in, z_pre).
    fifo: VecDeque<Vec<(Vec<f32>, Vec<f32>)>>,
}

impl LinearReluStack {
    pub fn n_layers(&self) -> usize { self.wt.len() }

    /// Build the stage holding global layers in `range`, drawn from a single
    /// deterministic RNG over ALL `total_layers` so any slice is consistent
    /// with the full-stack reference (each participant advances the RNG
    /// identically and keeps only its slice).
    pub fn build(d: usize, total_layers: usize, range: std::ops::Range<usize>, seed: u64) -> Self {
        use crate::rng::Rng;
        let mut rng = Rng::new(seed);
        let mut wt = Vec::new();
        let mut b = Vec::new();
        let scale = 1.0 / (d as f32).sqrt();
        for layer in 0..total_layers {
            let mut w = vec![0.0f32; d * d];
            for x in w.iter_mut() { *x = rng.next_normal() * scale; }
            let bb = vec![0.0f32; d]; // biases start at zero
            if range.contains(&layer) {
                wt.push(w);
                b.push(bb);
            }
        }
        let n = wt.len();
        let mk = |inner: usize| (0..n).map(|_| vec![0.0f32; inner]).collect::<Vec<_>>();
        LinearReluStack {
            d,
            g_wt: mk(d * d), m_wt: mk(d * d), v_wt: mk(d * d),
            g_b: mk(d), m_b: mk(d), v_b: mk(d),
            wt, b,
            fifo: VecDeque::new(),
        }
    }

    /// Flattened parameters in layer order, for parity comparison.
    pub fn flat_params(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for l in 0..self.wt.len() {
            out.extend_from_slice(&self.wt[l]);
            out.extend_from_slice(&self.b[l]);
        }
        out
    }
}

impl Stage for LinearReluStack {
    fn input_dim(&self) -> usize { self.d }
    fn output_dim(&self) -> usize { self.d }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        use aether_rt::ops;
        let d = self.d;
        let mut x = input.to_vec();
        let mut ctx: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(self.wt.len());
        for l in 0..self.wt.len() {
            let x_in = x.clone();
            let mut z = vec![0.0f32; d];
            unsafe {
                // z[1,d] = x[1,d] @ wt[d,d]
                ops::matmul_f32(x.as_ptr(), self.wt[l].as_ptr(), z.as_mut_ptr(), 1, d, d);
                ops::add_bias_f32(z.as_mut_ptr(), self.b[l].as_ptr(), 1, d);
            }
            let z_pre = z.clone();
            unsafe { ops::relu_f32(z.as_mut_ptr(), d); }
            ctx.push((x_in, z_pre));
            x = z;
        }
        self.fifo.push_back(ctx);
        x
    }

    fn backward(&mut self, grad: &[f32]) -> Vec<f32> {
        use aether_rt::ops;
        let d = self.d;
        let ctx = self.fifo.pop_front().expect("[pp] backward with empty fifo");
        let mut grad_y = grad.to_vec();
        for l in (0..self.wt.len()).rev() {
            let (x_in, z_pre) = &ctx[l];
            // dz = relu'(z) * grad_y
            let mut dz = vec![0.0f32; d];
            unsafe { ops::relu_backward_f32(z_pre.as_ptr(), grad_y.as_ptr(), dz.as_mut_ptr(), d); }
            // dWt = x^T @ dz  → [d,d]; accumulate into g_wt.
            let mut dwt = vec![0.0f32; d * d];
            unsafe { ops::matmul_backward_rhs_f32(x_in.as_ptr(), dz.as_ptr(), dwt.as_mut_ptr(), 1, d, d); }
            unsafe { ops::axpy_f32(1.0, dwt.as_ptr(), self.g_wt[l].as_mut_ptr(), d * d); }
            // db += dz
            for o in 0..d { self.g_b[l][o] += dz[o]; }
            // dx = dz @ wt^T → [1,d] propagated to the previous layer.
            let mut dx = vec![0.0f32; d];
            unsafe { ops::matmul_backward_lhs_f32(dz.as_ptr(), self.wt[l].as_ptr(), dx.as_mut_ptr(), 1, d, d); }
            grad_y = dx;
        }
        grad_y // grad w.r.t. stage input
    }

    fn step(&mut self, lr: f32, opt_step: i64) {
        use aether_rt::ops;
        let d = self.d;
        for l in 0..self.wt.len() {
            unsafe {
                ops::adamw_step_f32(
                    self.wt[l].as_mut_ptr(), self.g_wt[l].as_ptr(),
                    self.m_wt[l].as_mut_ptr(), self.v_wt[l].as_mut_ptr(),
                    lr, 0.9, 0.999, 1e-8, 0.0, opt_step, d * d,
                );
                ops::adamw_step_f32(
                    self.b[l].as_mut_ptr(), self.g_b[l].as_ptr(),
                    self.m_b[l].as_mut_ptr(), self.v_b[l].as_mut_ptr(),
                    lr, 0.9, 0.999, 1e-8, 0.0, opt_step, d,
                );
            }
            for x in self.g_wt[l].iter_mut() { *x = 0.0; }
            for x in self.g_b[l].iter_mut() { *x = 0.0; }
        }
    }
}

// ---------------------------------------------------------------- witness

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic microbatch input — pure function of `mb` so every
    /// participant (reference, rank 0, rank 1) agrees without communication.
    fn input_for(mb: usize, d: usize) -> Vec<f32> {
        use crate::rng::Rng;
        let mut r = Rng::new(0xABCD_0000 + mb as u64);
        (0..d).map(|_| r.next_normal()).collect()
    }
    fn target_for(mb: usize, d: usize) -> Vec<f32> {
        use crate::rng::Rng;
        let mut r = Rng::new(0x1234_0000 + mb as u64);
        (0..d).map(|_| r.next_normal()).collect()
    }
    /// MSE loss + gradient w.r.t. the prediction.
    fn mse(y: &[f32], t: &[f32]) -> (f32, Vec<f32>) {
        let n = y.len();
        let mut loss = 0.0f32;
        let mut dy = vec![0.0f32; n];
        for i in 0..n {
            let e = y[i] - t[i];
            loss += e * e;
            dy[i] = 2.0 * e / n as f32;
        }
        (loss / n as f32, dy)
    }

    /// Single-process reference: all layers in one stage, sequential over all
    /// microbatches, one optimizer step. The pipeline must reproduce this.
    fn reference(d: usize, total_layers: usize, nmb: usize, lr: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut stack = LinearReluStack::build(d, total_layers, 0..total_layers, seed);
        let mut losses = Vec::new();
        for mb in 0..nmb {
            let x = input_for(mb, d);
            let y = stack.forward(&x);
            let (loss, dy) = mse(&y, &target_for(mb, d));
            stack.backward(&dy);
            losses.push(loss);
        }
        stack.step(lr, 1);
        (losses, stack.flat_params())
    }

    #[test]
    fn pipeline_1f1b_two_stage_matches_single_process() {
        let d = 8;
        let total_layers = 4;
        let split = 2; // rank 0 = layers 0..2, rank 1 = layers 2..4
        let nmb = 4;
        let lr = 1e-2;
        let seed = 0x5EED_1234;
        let base_port = 29917;

        let (ref_losses, ref_params) = reference(d, total_layers, nmb, lr, seed);

        // Rank 1 (last stage) runs in a thread; it listens, so spawn it first.
        let r1 = std::thread::spawn(move || {
            let mut stage = LinearReluStack::build(d, total_layers, split..total_layers, seed);
            let links = connect_pipeline(1, 2, base_port, "127.0.0.1");
            let losses = run_1f1b(
                &mut stage, 1, 2, &links, nmb, lr, 1,
                |_mb| unreachable!("rank 1 is not the first rank"),
                |mb, y| mse(y, &target_for(mb, d)),
            );
            (losses, stage.flat_params())
        });

        // Rank 0 (first stage) runs on the test thread.
        let mut stage0 = LinearReluStack::build(d, total_layers, 0..split, seed);
        let links0 = connect_pipeline(0, 2, base_port, "127.0.0.1");
        let _ = run_1f1b(
            &mut stage0, 0, 2, &links0, nmb, lr, 1,
            |mb| input_for(mb, d),
            |_mb, _y| unreachable!("rank 0 is not the last rank"),
        );
        let params0 = stage0.flat_params();

        let (pp_losses, params1) = r1.join().expect("rank 1 thread panicked");

        // 1. Per-microbatch losses (computed on the last rank) match exactly.
        assert_eq!(pp_losses.len(), ref_losses.len(), "loss count mismatch");
        for (i, (a, b)) in pp_losses.iter().zip(ref_losses.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mb {} loss: pipelined {} vs ref {}", i, a, b);
        }

        // 2. Post-step parameters match: ref == rank0_params ++ rank1_params.
        let mut pp_params = params0;
        pp_params.extend_from_slice(&params1);
        assert_eq!(pp_params.len(), ref_params.len(), "param count mismatch");
        let mut max_diff = 0.0f32;
        for (a, b) in pp_params.iter().zip(ref_params.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        eprintln!(
            "[pp witness] losses {:?} (ref {:?}); param max|diff| = {:.3e}",
            pp_losses, ref_losses, max_diff
        );
        assert!(max_diff < 1e-5, "param parity: max|diff| {:.3e} >= 1e-5", max_diff);
    }
}
