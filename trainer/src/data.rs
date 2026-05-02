//! Byte-stream dataset. The on-disk file IS the token stream — vocab=256
//! means no tokenizer artefact, just bytes.

use std::fs::File;
use std::io::Read;
use std::path::Path;
use crate::rng::Rng;

pub struct ByteDataset {
    pub data: Vec<u8>,
    pub seq_len: usize,
}

impl ByteDataset {
    /// Read a corpus from disk. The path is canonicalised and must resolve to
    /// a regular file (no symlink chain, no directory). Size capped at 4 GiB
    /// to bound memory and prevent accidental ingestion of, say, /dev/zero.
    pub fn from_file(path: &Path, seq_len: usize) -> std::io::Result<Self> {
        const MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
        let canon = path.canonicalize()?;
        let meta = std::fs::symlink_metadata(&canon)?;
        if !meta.file_type().is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("not a regular file: {:?}", canon),
            ));
        }
        if meta.len() > MAX_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("corpus too large ({} bytes, max {})", meta.len(), MAX_BYTES),
            ));
        }
        let mut f = File::open(&canon)?;
        let mut buf = Vec::with_capacity(meta.len() as usize);
        f.read_to_end(&mut buf)?;
        Ok(Self { data: buf, seq_len })
    }

    /// Synthetic dataset for smoke tests: a simple repeating pattern that
    /// the model can plausibly learn.
    pub fn synthetic(n_bytes: usize, seq_len: usize) -> Self {
        let pat = b"the quick brown fox jumps over the lazy dog. ";
        let mut data = Vec::with_capacity(n_bytes);
        while data.len() < n_bytes { data.extend_from_slice(pat); }
        data.truncate(n_bytes);
        Self { data, seq_len }
    }

    pub fn n_starts(&self) -> usize {
        self.data.len().saturating_sub(self.seq_len + 1)
    }

    /// Sample a batch. Returns flat `ids` and `labels`, each of length
    /// `batch * seq_len`. labels[i] = ids[i+1].
    pub fn sample_batch(&self, batch: usize, rng: &mut Rng) -> (Vec<i32>, Vec<i32>) {
        assert!(self.n_starts() > 0, "dataset too small for seq_len");
        let n = self.n_starts();
        let mut ids = vec![0i32; batch * self.seq_len];
        let mut labels = vec![0i32; batch * self.seq_len];
        for bi in 0..batch {
            let start = rng.gen_range(n);
            for i in 0..self.seq_len {
                ids[bi * self.seq_len + i] = self.data[start + i] as i32;
                labels[bi * self.seq_len + i] = self.data[start + i + 1] as i32;
            }
        }
        (ids, labels)
    }
}
