//! Roadmap-aware audit dimension.
//!
//! Each `tests/runtime/*.aether` file MAY include a top-of-file marker:
//!
//!   // roadmap: P7.3, P10.6
//!
//! to declare which roadmap items it witnesses. This module parses those
//! markers, cross-references them against `docs/ROADMAP_V2.md` (the source
//! of truth for which items exist), and prints a phase-by-phase summary
//! of (item count, witnessed item count, missing items).
//!
//! Exit code stays informational — missing witnesses don't fail the audit
//! (the work is intentionally still ahead of us). They just surface in the
//! report so we can see the curve close.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct RoadmapItem {
    pub id: String,        // e.g. "P7.3"
    pub phase: u32,        // 7
    pub effort: String,    // "L"
    pub title: String,     // "The full op surface"
}

#[derive(Debug)]
pub struct RoadmapReport {
    pub items: Vec<RoadmapItem>,
    /// item id → list of test file basenames that witness it
    pub witnesses: BTreeMap<String, Vec<String>>,
    /// per-phase (total_items, witnessed_items)
    pub phase_progress: BTreeMap<u32, (usize, usize)>,
}

pub fn run(root: &Path) -> RoadmapReport {
    let mut items = parse_roadmap(&root.join("docs").join("ROADMAP_V2.md"));
    items.extend(parse_roadmap(&root.join("docs").join("ROADMAP_V3.md")));
    items.extend(parse_roadmap(&root.join("docs").join("ROADMAP_V4.md")));
    let witnesses = scan_witnesses(root);

    let mut phase_progress: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
    let mut witnessed_ids: BTreeSet<String> = witnesses.keys().cloned().collect();
    for it in &items {
        let entry = phase_progress.entry(it.phase).or_insert((0, 0));
        entry.0 += 1;
        if witnessed_ids.contains(&it.id) { entry.1 += 1; }
    }
    // Drop witnesses that don't correspond to a known item (typo guard).
    let known: BTreeSet<String> = items.iter().map(|i| i.id.clone()).collect();
    let witnesses: BTreeMap<String, Vec<String>> = witnesses.into_iter()
        .filter(|(k, _)| known.contains(k)).collect();
    witnessed_ids.retain(|k| known.contains(k));

    RoadmapReport { items, witnesses, phase_progress }
}

fn parse_roadmap(path: &Path) -> Vec<RoadmapItem> {
    let Ok(text) = std::fs::read_to_string(path) else { return Vec::new(); };
    let mut items = Vec::new();
    let mut cur_phase = 0u32;
    for line in text.lines() {
        // `# Phase N — title` (catches both en-dash and hyphen-minus).
        if let Some(rest) = line.strip_prefix("# Phase ") {
            if let Some(num_end) = rest.find(|c: char| !c.is_ascii_digit()) {
                if let Ok(n) = rest[..num_end].parse::<u32>() {
                    cur_phase = n;
                    continue;
                }
            }
        }
        // `## N.M Title (EFFORT)` or `## N.M Title (EFFORT, depends X)`.
        if let Some(rest) = line.strip_prefix("## ") {
            // Expect N.M-shaped numeric prefix.
            let mut chars = rest.chars();
            let first = chars.next().unwrap_or(' ');
            if !first.is_ascii_digit() { continue; }
            // Find the space after the numeric ID.
            if let Some(sp) = rest.find(' ') {
                let id_part = &rest[..sp];
                if !id_part.contains('.') { continue; }
                let title_full = rest[sp..].trim();
                // Effort is in trailing parens — `(L)` / `(L, depends X)`.
                let (title, effort) = match title_full.rfind(" (") {
                    Some(p) => {
                        let after = title_full[p + 2..].trim_end_matches(')');
                        let eff = after.split([',', ' ']).next().unwrap_or("").trim().to_string();
                        (title_full[..p].trim().to_string(), eff)
                    }
                    None => (title_full.to_string(), String::new()),
                };
                items.push(RoadmapItem {
                    id: format!("P{}.{}", cur_phase, id_part.trim_start_matches(&format!("{}.", cur_phase))),
                    phase: cur_phase,
                    effort,
                    title,
                });
            }
        }
    }
    items
}

fn scan_witnesses(root: &Path) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let dir = root.join("tests").join("runtime");
    let Ok(entries) = std::fs::read_dir(&dir) else { return out; };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) != Some("aether") { continue; }
        let Ok(src) = std::fs::read_to_string(&p) else { continue; };
        let basename = p.file_stem().and_then(|x| x.to_str()).unwrap_or("?").to_string();
        for line in src.lines().take(10) {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("// roadmap:") {
                for tag in rest.split(',') {
                    let tag = tag.trim().to_string();
                    if !tag.is_empty() {
                        out.entry(tag).or_default().push(basename.clone());
                    }
                }
                break;
            }
        }
    }
    out
}

pub fn report_text(r: &RoadmapReport, w: &mut impl std::io::Write) -> std::io::Result<()> {
    writeln!(w, "=== roadmap progress ({} items) ===", r.items.len())?;
    if r.items.is_empty() {
        writeln!(w, "  (no items parsed — docs/ROADMAP_V2.md not found or empty)")?;
        return Ok(());
    }
    for (phase, (total, done)) in &r.phase_progress {
        let pct = if *total > 0 { (done * 100) / total } else { 0 };
        writeln!(w, "  Phase {}: {}/{} witnessed  ({}%)", phase, done, total, pct)?;
    }
    let total_items: usize = r.phase_progress.values().map(|(t, _)| t).sum();
    let total_done:  usize = r.phase_progress.values().map(|(_, d)| d).sum();
    let pct = if total_items > 0 { (total_done * 100) / total_items } else { 0 };
    writeln!(w, "  ---- TOTAL: {}/{} ({}%) ----", total_done, total_items, pct)?;
    writeln!(w, "  (witnessed = ≥1 test in tests/runtime/ tagged `// roadmap: <id>`)")?;
    Ok(())
}
