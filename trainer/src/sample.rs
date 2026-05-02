//! Top-k temperature sampling. Pure Rust, no rand crate.

use crate::rng::Rng;

/// Sample one token from logits with temperature + top-k.
pub fn sample_topk(logits: &[f32], temperature: f32, top_k: usize, rng: &mut Rng) -> i32 {
    let n = logits.len();
    let temp = temperature.max(1e-5);
    let scaled: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &l)| (l / temp, i)).collect();

    let mut sorted = scaled.clone();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.min(n).max(1);
    let cutoff = sorted[k - 1].0;

    let mut probs = vec![0.0f32; n];
    let mut mx = f32::NEG_INFINITY;
    for &(v, i) in &scaled {
        if v >= cutoff {
            probs[i] = v;
            if v > mx { mx = v; }
        } else {
            probs[i] = f32::NEG_INFINITY;
        }
    }
    let mut sum = 0.0f32;
    for v in probs.iter_mut() {
        if v.is_finite() { *v = (*v - mx).exp(); sum += *v; } else { *v = 0.0; }
    }
    if sum <= 0.0 { return 0; }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if acc >= r { return i as i32; }
    }
    (n - 1) as i32
}
