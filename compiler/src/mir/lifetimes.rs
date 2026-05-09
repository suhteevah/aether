//! Borrow checker — flow-sensitive lifetime analysis.
//!
//! Phase 6.3 — tracks borrows of stack variables and rejects:
//!   * Mutable aliasing: two simultaneous `&mut x` references.
//!   * Read-write conflict: `&mut x` overlapping `&x`.
//!   * Use-after-move: reading a variable after it was consumed.
//!
//! Operates on a linear sequence of `BorrowEvent`s (issued by the AST→MIR
//! lowering). Uses a small per-place state machine; the implementation
//! is the kernel of Rust's NLL borrow checker.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorrowKind { Shared, Mut }

#[derive(Debug, Clone)]
pub enum BorrowEvent {
    Borrow { place: String, kind: BorrowKind, id: u32 },
    EndBorrow { id: u32 },
    Move    { place: String },
    Use     { place: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaceState {
    pub shared_borrows: u32,
    pub mut_borrows:    u32,
    pub moved:          bool,
}

#[derive(Default)]
pub struct Checker {
    pub places:  HashMap<String, PlaceState>,
    pub borrows: HashMap<u32, (String, BorrowKind)>,
    pub errors:  Vec<String>,
}

impl Checker {
    pub fn run(events: &[BorrowEvent]) -> Vec<String> {
        let mut c = Checker::default();
        for ev in events { c.step(ev); }
        c.errors
    }

    fn step(&mut self, ev: &BorrowEvent) {
        match ev {
            BorrowEvent::Borrow { place, kind, id } => {
                let st = self.places.entry(place.clone()).or_insert(PlaceState {
                    shared_borrows: 0, mut_borrows: 0, moved: false,
                });
                if st.moved {
                    self.errors.push(format!("borrow of moved value `{}`", place));
                    return;
                }
                match kind {
                    BorrowKind::Mut => {
                        if st.mut_borrows > 0 || st.shared_borrows > 0 {
                            self.errors.push(format!(
                                "cannot borrow `{}` as mutable: already borrowed", place));
                        }
                        st.mut_borrows += 1;
                    }
                    BorrowKind::Shared => {
                        if st.mut_borrows > 0 {
                            self.errors.push(format!(
                                "cannot borrow `{}` as shared: already mutably borrowed", place));
                        }
                        st.shared_borrows += 1;
                    }
                }
                self.borrows.insert(*id, (place.clone(), *kind));
            }
            BorrowEvent::EndBorrow { id } => {
                if let Some((place, kind)) = self.borrows.remove(id) {
                    if let Some(st) = self.places.get_mut(&place) {
                        match kind {
                            BorrowKind::Mut    => st.mut_borrows = st.mut_borrows.saturating_sub(1),
                            BorrowKind::Shared => st.shared_borrows = st.shared_borrows.saturating_sub(1),
                        }
                    }
                }
            }
            BorrowEvent::Move { place } => {
                let st = self.places.entry(place.clone()).or_insert(PlaceState {
                    shared_borrows: 0, mut_borrows: 0, moved: false,
                });
                if st.shared_borrows + st.mut_borrows > 0 {
                    self.errors.push(format!(
                        "cannot move out of `{}` while it is borrowed", place));
                }
                st.moved = true;
            }
            BorrowEvent::Use { place } => {
                if let Some(st) = self.places.get(place) {
                    if st.moved {
                        self.errors.push(format!("use of moved value `{}`", place));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(place: &str, k: BorrowKind, id: u32) -> BorrowEvent {
        BorrowEvent::Borrow { place: place.into(), kind: k, id }
    }

    #[test]
    fn double_mut_borrow_rejected() {
        let evs = vec![
            b("x", BorrowKind::Mut, 1),
            b("x", BorrowKind::Mut, 2),
        ];
        let errs = Checker::run(&evs);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("already borrowed"));
    }

    #[test]
    fn shared_then_mut_rejected() {
        let evs = vec![
            b("x", BorrowKind::Shared, 1),
            b("x", BorrowKind::Mut, 2),
        ];
        let errs = Checker::run(&evs);
        assert_eq!(errs.len(), 1);
    }

    #[test]
    fn end_borrow_releases_lock() {
        let evs = vec![
            b("x", BorrowKind::Mut, 1),
            BorrowEvent::EndBorrow { id: 1 },
            b("x", BorrowKind::Mut, 2),
        ];
        assert!(Checker::run(&evs).is_empty());
    }

    #[test]
    fn use_after_move_rejected() {
        let evs = vec![
            BorrowEvent::Move { place: "x".into() },
            BorrowEvent::Use  { place: "x".into() },
        ];
        let errs = Checker::run(&evs);
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("moved"));
    }
}
