//! IQ4_NL fused matmul parity (FR-17-extra-iq4_nl-fwd).
//!
//! IQ4_NL block layout (18 bytes per 32-elem block):
//!   bytes 0-1  : f16 scale `d`
//!   bytes 2-17 : 16-byte nibble-packed indices into a 16-entry codebook
//!                of signed int8 values:
//!                  [-127, -104, -83, -65, -49, -35, -22, -10,
//!                      1,   13,  25,  38,  53,  69,  89, 113]
//!   byte i holds: low nibble = index for elem i, high nibble = index for elem i+16
//!   dequant: y[j] = d * codebook[index]
//!
//! Used by cnc's glm-4.7-flash-UD-IQ3_XXS for ~72 tensors.
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_iq4_nl_matmul_seq1_cuda,
};

const BLOCK_K: usize = 32;
const BYTES_PER_BLOCK: usize = 18;

const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
];

fn synth_block(seed: u64) -> ([u8; BYTES_PER_BLOCK], [f32; BLOCK_K]) {
    let mut state = seed.wrapping_add(1);
    let mut next_u32 = || {
        let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
        state = z;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        ((z >> 32) ^ z) as u32
    };
    let mut bytes = [0u8; BYTES_PER_BLOCK];
    let d_f32 = 0.05f32;
    let d_bits = unsafe { aether_f32_to_f16(d_f32) } as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;
    // Reference dq must use the F16-roundtripped d (what the kernel reads),
    // not the F32 original — otherwise pack vs unpack diverges by ~5e-5 × max|kval|.
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let mut indices = [0u32; BLOCK_K];
    for k in 0..BLOCK_K { indices[k] = next_u32() & 0xF; }
    for i in 0..16 {
        let lo = indices[i] & 0xF;
        let hi = indices[i + 16] & 0xF;
        bytes[2 + i] = (lo | (hi << 4)) as u8;
    }
    let mut dq = [0f32; BLOCK_K];
    for k in 0..BLOCK_K {
        dq[k] = d * (KVALUES_IQ4NL[indices[k] as usize] as f32);
    }
    (bytes, dq)
}

fn unpack_iq4_nl(bytes: &[u8; BYTES_PER_BLOCK]) -> [f32; BLOCK_K] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let mut out = [0f32; BLOCK_K];
    for i in 0..16 {
        let byte = bytes[2 + i];
        let idx_lo = (byte & 0xF) as usize;
        let idx_hi = ((byte >> 4) & 0xF) as usize;
        out[i]      = d * (KVALUES_IQ4NL[idx_lo] as f32);
        out[i + 16] = d * (KVALUES_IQ4NL[idx_hi] as f32);
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
fn iq4_nl_pack_roundtrip_smoke() {
    let (bytes, dq_via_pack) = synth_block(7);
    let dq_via_unpack = unpack_iq4_nl(&bytes);
    let max_diff = dq_via_pack.iter().zip(dq_via_unpack.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("[iq4_nl-pack] pack vs unpack max_diff={:.3e}", max_diff);
    assert!(max_diff < 1e-9);
}

#[test]
fn iq4_nl_matches_cpu_dequant_small() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 32;
    let k = 256;  // 8 IQ4_NL blocks per row
    let n_blocks = k / BLOCK_K;
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let (bytes, dq) = synth_block(row as u64 * 1000 + b as u64);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..BLOCK_K { w_dequant[row * k + b * BLOCK_K + i] = dq[i]; }
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
        let rc = aether_op_fused_iq4_nl_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0, "fused_iq4_nl rc={}", rc);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_out.iter().filter(|x| x.is_finite()).count();
        println!("[iq4_nl] n={} k={} max_diff={:.3e} finite={}/{}",
            n, k, max_diff, n_finite, n);
        assert_eq!(n_finite, n);
        assert!(max_diff < 1e-3, "IQ4_NL diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn iq4_nl_matches_at_glm_class_shape() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 2048;
    let k = 2048;
    let n_blocks = k / BLOCK_K;
    let mut w_packed = vec![0u8; n * n_blocks * BYTES_PER_BLOCK];
    let mut w_dequant = vec![0f32; n * k];
    for row in 0..n {
        for b in 0..n_blocks {
            let (bytes, dq) = synth_block(row as u64 * 1000 + b as u64);
            let off = (row * n_blocks + b) * BYTES_PER_BLOCK;
            w_packed[off..off + BYTES_PER_BLOCK].copy_from_slice(&bytes);
            for i in 0..BLOCK_K { w_dequant[row * k + b * BLOCK_K + i] = dq[i]; }
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
        let rc = aether_op_fused_iq4_nl_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[iq4_nl-2048] max_diff={:.3e}", max_diff);
        assert!(max_diff < 1e-1, "IQ4_NL at GLM shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}
