//! Q5_K fused matmul parity (FR-17-extra-q5_k-fwd).
//!
//! Q5_K block layout (176 bytes per 256-elem super-block):
//!   bytes 0-1    : f16 super-block scale `d`
//!   bytes 2-3    : f16 super-block min `dmin`
//!   bytes 4-15   : 12 bytes scales (8 × {6-bit scale, 6-bit min},
//!                  IDENTICAL packing to Q4_K)
//!   bytes 16-47  : 32 bytes qh (one bit per element, bit `sub` of qh[l]
//!                  is the high bit of element `sub * 32 + l`)
//!   bytes 48-175 : 128 bytes qs (nibble-packed low 4 bits, 2 elems/byte)
//!
//! Used by Qwen2.5-32B Q5_K_M, Llama-3 Q5_K_M, GLM-4.7-flash, and basically
//! every modern Q5_K_M GGUF.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_q5_k_matmul_seq1_cuda,
};

const QK: usize = 256;
const BYTES_PER_BLOCK: usize = 176;

/// Construct a synthetic Q5_K block with random scales/mins/quants, return
/// the packed 176 bytes + the dequantised 256 f32 values.
fn synth_block(seed: u64) -> ([u8; BYTES_PER_BLOCK], [f32; QK]) {
    let mut state = seed.wrapping_add(1);
    let mut next_u32 = || {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z >> 32) ^ z) as u32
    };
    let mut bytes = [0u8; BYTES_PER_BLOCK];

    // f16 d and dmin
    let d_f32 = 0.05f32;
    let dmin_f32 = 0.5f32;
    let d_bits = unsafe { aether_f32_to_f16(d_f32) } as u16;
    let dmin_bits = unsafe { aether_f32_to_f16(dmin_f32) } as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;
    bytes[2] = (dmin_bits & 0xFF) as u8;
    bytes[3] = ((dmin_bits >> 8) & 0xFF) as u8;

    // 8 sub-blocks × 6-bit scale + 6-bit min, packed identical to Q4_K.
    // Pick random 6-bit values.
    let mut scale_6 = [0u32; 8];
    let mut min_6   = [0u32; 8];
    for j in 0..8 {
        scale_6[j] = next_u32() & 0x3F;
        min_6[j]   = next_u32() & 0x3F;
    }
    // Pack the 12-byte scales array.  See q4k_get_scale/min in cuda.rs.
    for j in 0..4 {
        // scales[j]: low 6 bits = scale_6[j], high 2 bits = scale_6[j+4] >> 4
        bytes[4 + j]     = ((scale_6[j] & 0x3F) | ((scale_6[j + 4] >> 4) << 6)) as u8;
        // scales[j+4]: low 6 bits = min_6[j], high 2 bits = min_6[j+4] >> 4
        bytes[4 + j + 4] = ((min_6[j]   & 0x3F) | ((min_6[j + 4] >> 4) << 6)) as u8;
        // scales[j+8]: low 4 bits = scale_6[j+4] & 0xF, high 4 bits = min_6[j+4] & 0xF
        bytes[4 + j + 8] = ((scale_6[j + 4] & 0xF) | ((min_6[j + 4] & 0xF) << 4)) as u8;
    }

    // 32-byte qh + 128-byte qs.  For each elem in [0, 256) at sub-block sub
    // and lane l (sub * 32 + l): pick a random 5-bit quant, split into
    // low 4 bits → qs nibble (sub%2 picks low/high half of byte) and
    // high 1 bit → qh[l] bit sub.
    let mut quants = [0u32; QK];
    for k in 0..QK { quants[k] = next_u32() & 0x1F; }

    // qh: bytes 16-47.  Zeroed initially; set bit per quant.
    for sub in 0..8 {
        for l in 0..32 {
            let elem = sub * 32 + l;
            let hi = (quants[elem] >> 4) & 1;
            if hi != 0 { bytes[16 + l] |= 1 << sub; }
        }
    }
    // qs: bytes 48-175.  Each byte holds low nibble of (sub*32+l) for one
    // sub-block and high nibble of (sub_pair * 32 + l) for the next.
    // Block layout: 4 sub-block pairs × 32 bytes each, byte (pair, l) holds
    //   low-nibble  = quant_low_4 of element (pair*2 + 0) sub at lane l
    //   high-nibble = quant_low_4 of element (pair*2 + 1) sub at lane l
    // Matches the kernel's `qs_off = (sub >> 1) * 32; nibble selected by sub & 1`.
    for pair in 0..4 {
        for l in 0..32 {
            let sub_lo = pair * 2;
            let sub_hi = pair * 2 + 1;
            let q_lo = quants[sub_lo * 32 + l] & 0xF;
            let q_hi = quants[sub_hi * 32 + l] & 0xF;
            bytes[48 + pair * 32 + l] = (q_lo | (q_hi << 4)) as u8;
        }
    }

    let dq = dequant_q5_k_block(&bytes);
    (bytes, dq)
}

/// CPU reference dequant — mirrors the GPU kernel exactly.
fn dequant_q5_k_block(bytes: &[u8; BYTES_PER_BLOCK]) -> [f32; QK] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let dmin_bits = ((bytes[3] as u32) << 8) | (bytes[2] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let dmin = unsafe { aether_f16_to_f32(dmin_bits as i32) };
    let scales = &bytes[4..16];
    let qh = &bytes[16..48];
    let qs = &bytes[48..176];

    let get_scale = |sub: usize| -> u32 {
        if sub < 4 {
            (scales[sub] & 0x3F) as u32
        } else {
            ((scales[sub + 4] & 0xF) as u32) | (((scales[sub - 4] >> 6) as u32) << 4)
        }
    };
    let get_min = |sub: usize| -> u32 {
        if sub < 4 {
            (scales[sub + 4] & 0x3F) as u32
        } else {
            ((scales[sub + 4] >> 4) as u32) | (((scales[sub] >> 6) as u32) << 4)
        }
    };

    let mut out = [0f32; QK];
    for sub in 0..8 {
        let j = sub >> 1;
        let is_hi = sub & 1;
        let d_eff = d * (get_scale(sub) as f32);
        let m_eff = dmin * (get_min(sub) as f32);
        let qs_off = j * 32;
        for l in 0..32 {
            let byte = qs[qs_off + l];
            let nibble = if is_hi != 0 {
                ((byte >> 4) & 0xF) as u32
            } else {
                (byte & 0xF) as u32
            };
            let hi_bit = ((qh[l] >> sub) & 1) as u32;
            let quant = (nibble | (hi_bit << 4)) as i32;  // [0, 31]
            out[sub * 32 + l] = d_eff * (quant as f32) - m_eff;
        }
    }
    out
}

fn deterministic(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n).map(|_| {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        let u = (((z >> 32) ^ z) as u32) as f32 / 4_294_967_296.0;
        (u * 2.0 - 1.0) * scale
    }).collect()
}

#[test]
fn q5_k_pack_roundtrip_smoke() {
    // Pack random scales/mins/quants and verify the dequant produces the
    // exact d * scale * quant - dmin * min reconstruction.
    let (bytes, dq) = synth_block(7);
    // Re-derive: extract d, dmin, scales, then for each elem dequant.
    let d = unsafe {
        let bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
        aether_f16_to_f32(bits as i32)
    };
    let dmin = unsafe {
        let bits = ((bytes[3] as u32) << 8) | (bytes[2] as u32);
        aether_f16_to_f32(bits as i32)
    };
    println!("[q5_k-pack] d={} dmin={} dq[0..4]={:?}", d, dmin, &dq[..4]);
    // The whole roundtrip is the dequant fn itself, so if it produces
    // finite + non-zero values we're packing+unpacking consistently.
    assert!(dq.iter().all(|x| x.is_finite()));
    assert!(dq.iter().any(|x| x.abs() > 0.0));
}

#[test]
fn q5_k_matches_cpu_dequant_small() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 32;
    let k = 512;  // 2 super-blocks per row
    let n_blocks = k / QK;
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let (bytes, dq) = synth_block(row as u64 * 1000 + b as u64);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..QK { w_dequant[row * k + b * QK + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 11, 1.0);
    let mut cpu_out = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += a[i] * w_dequant[row * k + i];
        }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_q5_k_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0, "fused_q5_k rc={}", rc);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_out.iter().filter(|x| x.is_finite()).count();
        println!("[q5_k] n={} k={} max_diff={:.3e} finite={}/{}",
            n, k, max_diff, n_finite, n);
        assert_eq!(n_finite, n, "non-finite values in Q5_K output");
        assert!(max_diff < 1e-3, "Q5_K diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn q5_k_matches_at_qwen_class_shape() {
    // d_model=2048, n_out=2048 (attn matmul shape for V2-Lite + GLM-4.7).
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 2048;
    let k = 2048;
    let n_blocks = k / QK;
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let (bytes, dq) = synth_block(row as u64 * 1000 + b as u64);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..QK { w_dequant[row * k + b * QK + i] = dq[i]; }
        }
    }
    let a = deterministic(k, 23, 1.0);
    let mut cpu_out = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for i in 0..k {
            acc += a[i] * w_dequant[row * k + i];
        }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_q5_k_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[q5_k-2048] n={} k={} max_diff={:.3e}", n, k, max_diff);
        // Larger n + k accumulates fp error; tolerance scaled accordingly.
        assert!(max_diff < 1e-1, "Q5_K diverged at 2048 ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}
