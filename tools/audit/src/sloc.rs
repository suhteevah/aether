//! SLOC accounting per crate / per file. Counts lines that are not blank
//! and not whole-line comments. Sufficient for tracking growth; not a
//! formal coverage metric.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Default, Clone, Debug)]
pub struct FileStats {
    pub lines_total: u32,
    pub lines_code: u32,
    pub lines_comment: u32,
    pub lines_blank: u32,
}

#[derive(Default)]
pub struct SlocReport {
    pub by_file: BTreeMap<PathBuf, FileStats>,
    pub by_crate: BTreeMap<String, FileStats>,
}

impl SlocReport {
    pub fn merge(&mut self, krate: String, path: PathBuf, fs: FileStats) {
        let agg = self.by_crate.entry(krate).or_default();
        agg.lines_total += fs.lines_total;
        agg.lines_code += fs.lines_code;
        agg.lines_comment += fs.lines_comment;
        agg.lines_blank += fs.lines_blank;
        self.by_file.insert(path, fs);
    }

    pub fn total(&self) -> FileStats {
        let mut agg = FileStats::default();
        for fs in self.by_crate.values() {
            agg.lines_total += fs.lines_total;
            agg.lines_code += fs.lines_code;
            agg.lines_comment += fs.lines_comment;
            agg.lines_blank += fs.lines_blank;
        }
        agg
    }
}

pub fn count_workspace(root: &Path) -> SlocReport {
    let mut report = SlocReport::default();
    walk(root, root, &mut report);
    report
}

fn walk(root: &Path, dir: &Path, report: &mut SlocReport) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if matches!(name.as_ref(), "target" | ".git" | "checkpoints" | "node_modules") {
            continue;
        }
        if name.starts_with('.') { continue; }
        if path.is_dir() {
            walk(root, &path, report);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "rs" | "aether") {
                count_file(root, &path, report);
            }
        }
    }
}

fn count_file(root: &Path, path: &Path, report: &mut SlocReport) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return,
    };
    let mut fs = FileStats::default();
    for raw in text.lines() {
        fs.lines_total += 1;
        let t = raw.trim();
        if t.is_empty() {
            fs.lines_blank += 1;
        } else if t.starts_with("//") || t.starts_with("///") || t.starts_with("//!")
            || t.starts_with("/*") || t.starts_with('*') {
            fs.lines_comment += 1;
        } else {
            fs.lines_code += 1;
        }
    }
    let rel = path.strip_prefix(root).unwrap_or(path).to_path_buf();
    let krate = rel.iter().next()
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_else(|| "<root>".into());
    report.merge(krate, rel, fs);
}
