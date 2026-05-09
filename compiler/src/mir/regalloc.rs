//! Linear-scan register allocator.
//!
//! Phase 10.3 — assigns physical registers to SSA virtual values.
//! Today the asm backend stores every local in a stack slot. This pass
//! computes live ranges, sorts them by start point, sweeps once
//! (Poletto/Sarkar 1999), and assigns each live range to the first
//! free physical register; spills the longest-active range when the
//! pool is exhausted.
//!
//! Limited to a small fixed register pool here (callee-saved scratch:
//! r10, r11, r12, r13). Backend integration sits behind the existing
//! stack-slot path and is enabled by `--ra` (TODO: wire flag).

use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiveRange {
    pub vreg: u32,
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loc {
    Reg(u8),
    Spill(u32),
}

pub struct Allocator {
    pub pool: Vec<u8>,
}

impl Allocator {
    pub fn new(pool: Vec<u8>) -> Self { Self { pool } }

    pub fn allocate(&self, ranges: &mut [LiveRange]) -> Vec<(u32, Loc)> {
        ranges.sort_by_key(|r| r.start);
        let mut out = Vec::with_capacity(ranges.len());
        let mut active: Vec<(LiveRange, u8)> = Vec::new();
        let mut free: Vec<u8> = self.pool.clone();
        let mut next_spill: u32 = 0;
        for &r in ranges.iter() {
            // Expire ranges whose end < r.start.
            let still_active: Vec<(LiveRange, u8)> = active
                .into_iter()
                .filter(|(act, reg)| {
                    if act.end < r.start { free.push(*reg); false } else { true }
                })
                .collect();
            active = still_active;
            if let Some(reg) = free.pop() {
                active.push((r, reg));
                out.push((r.vreg, Loc::Reg(reg)));
            } else {
                // Spill the longest-active range OR the new one — pick
                // whichever ends LATER (Poletto's heuristic).
                let (idx, &(longest, longest_reg)) = active
                    .iter().enumerate()
                    .max_by_key(|(_, (lr, _))| lr.end)
                    .unwrap();
                if longest.end > r.end {
                    out.push((longest.vreg, Loc::Spill(next_spill)));
                    next_spill += 1;
                    active[idx] = (r, longest_reg);
                    out.push((r.vreg, Loc::Reg(longest_reg)));
                } else {
                    out.push((r.vreg, Loc::Spill(next_spill)));
                    next_spill += 1;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_overlap_uses_one_reg() {
        let alloc = Allocator::new(vec![10, 11, 12]);
        let mut r = vec![
            LiveRange { vreg: 0, start: 0, end: 1 },
            LiveRange { vreg: 1, start: 2, end: 3 },
        ];
        let out = alloc.allocate(&mut r);
        // Both vregs land in the same physical reg since they don't overlap.
        let regs: HashSet<_> = out.iter().filter_map(|(_, l)| match l {
            Loc::Reg(r) => Some(*r), _ => None
        }).collect();
        assert_eq!(regs.len(), 1);
    }

    #[test]
    fn overflow_spills() {
        let alloc = Allocator::new(vec![10]);  // pool of 1
        let mut r = vec![
            LiveRange { vreg: 0, start: 0, end: 10 },
            LiveRange { vreg: 1, start: 1, end: 5 },
        ];
        let out = alloc.allocate(&mut r);
        // One must spill.
        let spilled = out.iter().filter(|(_, l)| matches!(l, Loc::Spill(_))).count();
        assert!(spilled >= 1);
    }
}
