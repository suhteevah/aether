//! Minimal SplitMix64 RNG. No external crates; deterministic across runs.

#[derive(Clone)]
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) } }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        // Uniform [0, 1)
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Box-Muller normal sample (mean 0, std 1).
    pub fn next_normal(&mut self) -> f32 {
        loop {
            let u: f32 = self.next_f32();
            if u > 1e-10 {
                let v: f32 = self.next_f32();
                return (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
            }
        }
    }

    pub fn gen_range(&mut self, hi: usize) -> usize {
        (self.next_u64() % hi as u64) as usize
    }
}
