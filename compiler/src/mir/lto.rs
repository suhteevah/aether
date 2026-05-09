//! Whole-program LTO bookkeeping.
//!
//! Phase 10.9 — collects per-crate fn signatures + call edges into a
//! single `LtoGraph`, then runs cross-module DCE: any fn with no caller
//! AND not exported is unreachable and gets dropped before final link.
//!
//! A real implementation would also do cross-module inlining decisions
//! and per-callsite specialization; this scaffold computes the
//! reachability set, which is the prerequisite.

use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct CrateUnit {
    pub name: String,
    pub fns: Vec<FnSummary>,
}

#[derive(Debug, Clone)]
pub struct FnSummary {
    pub name: String,
    /// Called fn names by FQN (`crate::name`).
    pub callees: Vec<String>,
    pub exported: bool,
}

#[derive(Default)]
pub struct LtoGraph {
    pub units: Vec<CrateUnit>,
}

impl LtoGraph {
    pub fn add(&mut self, unit: CrateUnit) { self.units.push(unit); }

    /// Returns the set of fn FQNs reachable from any exported fn, plus
    /// the exported fns themselves. Anything not in this set is
    /// dead-on-link.
    pub fn reachable(&self) -> HashSet<String> {
        let mut by_fqn: HashMap<String, &FnSummary> = HashMap::new();
        for u in &self.units {
            for f in &u.fns {
                by_fqn.insert(format!("{}::{}", u.name, f.name), f);
            }
        }
        let mut work: VecDeque<String> = by_fqn.iter()
            .filter(|(_, f)| f.exported)
            .map(|(k, _)| k.clone())
            .collect();
        let mut seen: HashSet<String> = work.iter().cloned().collect();
        while let Some(fqn) = work.pop_front() {
            if let Some(f) = by_fqn.get(&fqn) {
                for c in &f.callees {
                    if !seen.contains(c) {
                        seen.insert(c.clone());
                        work.push_back(c.clone());
                    }
                }
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fns(name: &str, callees: &[&str], exported: bool) -> FnSummary {
        FnSummary {
            name: name.into(),
            callees: callees.iter().map(|s| s.to_string()).collect(),
            exported,
        }
    }

    #[test]
    fn reachable_through_one_hop() {
        let mut g = LtoGraph::default();
        g.add(CrateUnit { name: "main".into(), fns: vec![
            fns("entry", &["main::helper"], true),
            fns("helper", &["main::leaf"], false),
            fns("leaf", &[], false),
            fns("dead", &[], false),
        ]});
        let r = g.reachable();
        assert!(r.contains("main::entry"));
        assert!(r.contains("main::helper"));
        assert!(r.contains("main::leaf"));
        assert!(!r.contains("main::dead"));
    }

    #[test]
    fn cross_crate_reach() {
        let mut g = LtoGraph::default();
        g.add(CrateUnit { name: "a".into(), fns: vec![
            fns("entry", &["b::work"], true),
        ]});
        g.add(CrateUnit { name: "b".into(), fns: vec![
            fns("work", &[], false),
            fns("unused", &[], false),
        ]});
        let r = g.reachable();
        assert!(r.contains("b::work"));
        assert!(!r.contains("b::unused"));
    }
}
