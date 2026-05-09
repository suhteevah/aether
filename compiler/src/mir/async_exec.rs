//! `async` / Future trait scaffolding + minimal single-threaded executor.
//!
//! Phase 6.10 — model of how `async fn` lowers and how an executor drives
//! the resulting state machine. The Future is represented as an enum-of-
//! states; the executor polls it until `Poll::Ready`.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Poll<T> { Pending, Ready(T) }

/// A trivial Future that yields its value after `delay` polls.
/// Models the state-machine an `async fn` desugars into.
#[derive(Debug, Clone)]
pub struct DelayFuture {
    pub delay: u32,
    pub polls: u32,
    pub value: i64,
}

impl DelayFuture {
    pub fn new(delay: u32, value: i64) -> Self {
        Self { delay, polls: 0, value }
    }

    pub fn poll(&mut self) -> Poll<i64> {
        self.polls += 1;
        if self.polls >= self.delay { Poll::Ready(self.value) }
        else { Poll::Pending }
    }
}

/// A composite Future that awaits A, then B, then returns A.value + B.value.
#[derive(Debug, Clone)]
pub struct ChainFuture {
    pub a: DelayFuture,
    pub b: DelayFuture,
    pub a_done: Option<i64>,
}

impl ChainFuture {
    pub fn new(a: DelayFuture, b: DelayFuture) -> Self {
        Self { a, b, a_done: None }
    }

    pub fn poll(&mut self) -> Poll<i64> {
        if self.a_done.is_none() {
            match self.a.poll() {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(v) => self.a_done = Some(v),
            }
        }
        match self.b.poll() {
            Poll::Pending => Poll::Pending,
            Poll::Ready(v) => Poll::Ready(self.a_done.unwrap() + v),
        }
    }
}

/// Executor: spin-poll a Future to completion. Returns the final value
/// + the total poll count (useful for verifying the polling rhythm).
pub fn block_on_delay(mut f: DelayFuture) -> (i64, u32) {
    loop {
        if let Poll::Ready(v) = f.poll() { return (v, f.polls); }
    }
}

pub fn block_on_chain(mut f: ChainFuture) -> i64 {
    loop {
        if let Poll::Ready(v) = f.poll() { return v; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_future_polls_n_times() {
        let f = DelayFuture::new(5, 42);
        let (v, polls) = block_on_delay(f);
        assert_eq!(v, 42);
        assert_eq!(polls, 5);
    }

    #[test]
    fn chain_future_awaits_both_arms() {
        let a = DelayFuture::new(2, 10);
        let b = DelayFuture::new(3, 32);
        assert_eq!(block_on_chain(ChainFuture::new(a, b)), 42);
    }

    #[test]
    fn pending_then_ready_transition() {
        let mut f = DelayFuture::new(2, 7);
        assert_eq!(f.poll(), Poll::Pending);
        assert_eq!(f.poll(), Poll::Ready(7));
    }
}
