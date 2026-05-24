//! IQ3_S fused matmul parity (FR-17-extra-iq3_s-fwd).
//!
//! cnc's GLM-4.7-flash-UD-IQ3_XXS uses IQ3_S for ~44 tensors (the higher-
//! precision "selected" sibling of IQ3_XXS).  Format: 110-byte 256-element
//! block.
//!
//!   bytes 0-1     : f16 super-block scale `d`
//!   bytes 2-65    : 64 bytes qs   (low 8 bits of codebook index)
//!   bytes 66-73   : 8 bytes  qh   (high bit (bit 8) of codebook index;
//!                                   1 byte per 32-elem sub-block, supplying
//!                                   2 bits per lane × 4 lanes)
//!   bytes 74-105  : 32 bytes signs (direct 8-bit sign patterns — one per
//!                                   8-weight lane × 4 lanes × 8 sub-blocks)
//!   bytes 106-109 : 4 bytes  scales (4-bit unsigned per sub-block × 8)
//!
//! Per-sub-block:
//!   scale_nib = (scales[ib32/2] >> (4*(ib32&1))) & 0xF
//!   db        = d * (1 + 2 * scale_nib)   // odd ints 1, 3, ..., 31
//!
//! Compared to IQ3_XXS, IQ3_S uses a 512-entry grid (vs 256), direct 8-bit
//! signs (vs ksigns_iq2xs indirection), and odd-integer per-sub-block scales
//! (vs the 0.5 * (0.5 + scale) factor).
//!
//! roadmap: P17.5

#![cfg(feature = "cuda")]

use aether_rt::{aether_f32_to_f16, aether_f16_to_f32};
use aether_rt::cuda::{
    aether_dev_init, aether_dev_alloc_f32, aether_dev_free_f32,
    aether_dev_alloc_u8, aether_dev_free_u8,
    aether_dev_h2d_f32, aether_dev_d2h_f32, aether_dev_h2d_u8, aether_dev_sync,
    aether_op_fused_iq3_s_matmul_seq1_cuda,
};

const QK: usize = 256;
const BYTES_PER_BLOCK: usize = 110;

// llama.cpp's iq3s_grid: 9-bit index → packed 4-uint8 quant pattern (u32).
// Same bytes embedded in the CUDA kernel source — kept in sync by hand.
const IQ3S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

fn next_u32(state: &mut u64) -> u32 {
    let mut z = state.wrapping_add(0x9E3779B97F4A7C15);
    *state = z;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    ((z >> 32) ^ z) as u32
}

/// Build a synthetic 110-byte IQ3_S block.  Returns (block_bytes, dequant).
fn synth_block(seed: u64) -> ([u8; BYTES_PER_BLOCK], [f32; QK]) {
    let mut state = seed.wrapping_add(1);
    let mut bytes = [0u8; BYTES_PER_BLOCK];

    // f16 d ≈ 0.05 (typical for 3-bit weights — odd-int scales explode fast).
    let d = 0.05f32;
    let d_bits = unsafe { aether_f32_to_f16(d) } as u16;
    bytes[0] = (d_bits & 0xFF) as u8;
    bytes[1] = ((d_bits >> 8) & 0xFF) as u8;

    // 64 bytes qs.
    for i in 0..64 {
        bytes[2 + i] = (next_u32(&mut state) & 0xFF) as u8;
    }
    // 8 bytes qh.
    for i in 0..8 {
        bytes[66 + i] = (next_u32(&mut state) & 0xFF) as u8;
    }
    // 32 bytes signs (direct 8-bit patterns).
    for i in 0..32 {
        bytes[74 + i] = (next_u32(&mut state) & 0xFF) as u8;
    }
    // 4 bytes scales (4-bit per sub-block × 8).
    for i in 0..4 {
        let lo = next_u32(&mut state) & 0xF;
        let hi = next_u32(&mut state) & 0xF;
        bytes[106 + i] = (lo | (hi << 4)) as u8;
    }

    let dq = dequant_iq3_s_block(&bytes);
    (bytes, dq)
}

/// CPU reference — mirrors the GPU kernel exactly.
fn dequant_iq3_s_block(bytes: &[u8; BYTES_PER_BLOCK]) -> [f32; QK] {
    let d_bits = ((bytes[1] as u32) << 8) | (bytes[0] as u32);
    let d = unsafe { aether_f16_to_f32(d_bits as i32) };
    let qs     = &bytes[2..66];
    let qh     = &bytes[66..74];
    let signs  = &bytes[74..106];
    let scales = &bytes[106..110];
    let mut out = [0f32; QK];

    for ib32 in 0..8 {
        let scale_nib = ((scales[ib32 >> 1] as u32) >> (4 * (ib32 & 1))) & 0xF;
        let db = d * (1.0 + 2.0 * scale_nib as f32);
        let qh_byte = qh[ib32] as u32;

        for l in 0..4 {
            let idx1 = (qs[ib32 * 8 + 2 * l + 0] as u32)
                | ((qh_byte << (8 - 2 * l)) & 256);
            let idx2 = (qs[ib32 * 8 + 2 * l + 1] as u32)
                | ((qh_byte << (7 - 2 * l)) & 256);
            let grid1 = IQ3S_GRID[idx1 as usize];
            let grid2 = IQ3S_GRID[idx2 as usize];
            let sign  = signs[ib32 * 4 + l] as u32;

            for j in 0..4 {
                let q0 = ((grid1 >> (8 * j)) & 0xFF) as u32;
                let q1 = ((grid2 >> (8 * j)) & 0xFF) as u32;
                let s0 = if (sign & (1 << (j + 0))) != 0 { -1.0 } else { 1.0 };
                let s1 = if (sign & (1 << (j + 4))) != 0 { -1.0 } else { 1.0 };
                out[32 * ib32 + 8 * l + j + 0] = db * (q0 as f32) * s0;
                out[32 * ib32 + 8 * l + j + 4] = db * (q1 as f32) * s1;
            }
        }
    }
    out
}

fn deterministic(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n).map(|_| {
        let u = (next_u32(&mut state) as f32) / 4_294_967_296.0;
        (u * 2.0 - 1.0) * scale
    }).collect()
}

#[test]
fn iq3_s_pack_roundtrip_smoke() {
    // synth_block calls dequant_iq3_s_block — a second invocation must agree
    // bit-for-bit (sanity check that the reference dequant is deterministic
    // wrt the packed bytes).
    let (bytes, dq_a) = synth_block(13);
    let dq_b = dequant_iq3_s_block(&bytes);
    let max_diff = dq_a.iter().zip(dq_b.iter())
        .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("[iq3_s-pack] roundtrip max_diff={:.3e}", max_diff);
    assert!(max_diff == 0.0, "roundtrip not deterministic");
}

#[test]
fn iq3_s_matches_cpu_dequant_small() {
    unsafe { assert_eq!(aether_dev_init(), 0); }
    let n = 32;
    let k = 256;  // exactly one IQ3_S block per row
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
        for i in 0..k { acc += a[i] * w_dequant[row * k + i]; }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_iq3_s_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0, "fused_iq3_s rc={}", rc);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let n_finite = gpu_out.iter().filter(|x| x.is_finite()).count();
        println!("[iq3_s] n={} k={} max_diff={:.3e} finite={}/{}",
            n, k, max_diff, n_finite, n);
        assert_eq!(n_finite, n, "non-finite values in IQ3_S output");
        assert!(max_diff < 1e-3,
            "IQ3_S matmul diverged from CPU reference ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}

#[test]
fn iq3_s_matches_at_glm_47_flash_class_shape() {
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
        for i in 0..k { acc += a[i] * w_dequant[row * k + i]; }
        cpu_out[row] = acc;
    }
    unsafe {
        let a_dev = aether_dev_alloc_f32(k as i32);
        let w_dev = aether_dev_alloc_u8((n * n_blocks * BYTES_PER_BLOCK) as i32);
        let o_dev = aether_dev_alloc_f32(n as i32);
        aether_dev_h2d_f32(a.as_ptr() as i64, a_dev, k as i32);
        aether_dev_h2d_u8(w_packed.as_ptr() as i64, w_dev,
            (n * n_blocks * BYTES_PER_BLOCK) as i32);
        let rc = aether_op_fused_iq3_s_matmul_seq1_cuda(
            a_dev, w_dev, o_dev, n as i32, n_blocks as i32);
        assert_eq!(rc, 0);
        aether_dev_sync();
        let mut gpu_out = vec![0f32; n];
        aether_dev_d2h_f32(o_dev, gpu_out.as_mut_ptr() as i64, n as i32);
        let max_diff = cpu_out.iter().zip(gpu_out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("[iq3_s-2048] max_diff={:.3e}", max_diff);
        assert!(max_diff < 1.0,
            "IQ3_S at GLM-class shape diverged ({:.3e})", max_diff);
        aether_dev_free_f32(a_dev); aether_dev_free_u8(w_dev); aether_dev_free_f32(o_dev);
    }
}
