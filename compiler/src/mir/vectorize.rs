//! Loop vectorization detection.
//!
//! Phase 10.6 — recognises the simplest auto-vectorisable shape:
//! `for i in 0..N { c[i] = a[i] OP b[i]; }` with no dependencies between
//! iterations. The pass returns a `VectorPlan` describing the SIMD width
//! and unroll factor the codegen should emit; the actual register-class
//! lowering (xmm/ymm/zmm) sits in the asm backend.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdWidth {
    Sse128 = 4,    // 4 × f32
    Avx256 = 8,    // 8 × f32
    Avx512 = 16,   // 16 × f32
}

#[derive(Debug, Clone)]
pub struct VectorPlan {
    pub width: SimdWidth,
    pub unroll: u32,
    pub trip_count: u32,
    pub remainder: u32,
}

#[derive(Debug, Clone)]
pub struct LoopShape {
    pub trip_count: u32,
    pub has_loop_carried_dep: bool,
    pub body_op_count: u32,
}

pub fn plan(shape: &LoopShape, target: SimdWidth) -> Option<VectorPlan> {
    if shape.has_loop_carried_dep { return None; }
    if shape.trip_count == 0 { return None; }
    let lane = target as u32;
    let unroll = if shape.body_op_count <= 2 { 4 }
                 else if shape.body_op_count <= 4 { 2 }
                 else { 1 };
    let main = (shape.trip_count / lane) * lane;
    let remainder = shape.trip_count - main;
    Some(VectorPlan { width: target, unroll, trip_count: main, remainder })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_trip_no_remainder() {
        let p = plan(&LoopShape { trip_count: 256, has_loop_carried_dep: false, body_op_count: 1 }, SimdWidth::Avx256).unwrap();
        assert_eq!(p.trip_count, 256);
        assert_eq!(p.remainder, 0);
        assert_eq!(p.unroll, 4);
    }

    #[test]
    fn unaligned_trip_has_remainder() {
        let p = plan(&LoopShape { trip_count: 250, has_loop_carried_dep: false, body_op_count: 3 }, SimdWidth::Avx256).unwrap();
        assert_eq!(p.trip_count, 248);
        assert_eq!(p.remainder, 2);
        assert_eq!(p.unroll, 2);
    }

    #[test]
    fn dep_blocks_vectorisation() {
        assert!(plan(&LoopShape { trip_count: 256, has_loop_carried_dep: true, body_op_count: 1 }, SimdWidth::Avx256).is_none());
    }
}
