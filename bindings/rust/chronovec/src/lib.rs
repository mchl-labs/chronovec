//! A snapshot-isolated approximate-nearest-neighbour index.
//!
//! ChronoVec is built for collections that change. Every version carries an
//! MVCC interval, so a query can read the index as of a point in time, and
//! deletion reclaims space on a bounded budget rather than accumulating
//! tombstones.
//!
//! ```no_run
//! use chronovec::{Index, Metric};
//!
//! let index = Index::builder(3).metric(Metric::Cosine).build()?;
//! let t1 = index.insert(1, &[0.1, 0.2, 0.3])?;
//! index.delete(1)?;
//!
//! assert!(index.search(&[0.1, 0.2, 0.3], 10)?.is_empty());
//! assert_eq!(index.search_as_of(&[0.1, 0.2, 0.3], 10, t1)?.len(), 1);
//! # Ok::<(), chronovec::Error>(())
//! ```
//!
//! # Threading
//!
//! Searches are lock-free over immutable snapshots, and mutations are
//! thread-safe but serialise internally. Every method therefore takes `&self`:
//! share one index with `Arc<Index>` and query it from many threads without an
//! external lock. This is *lock-free reads*, not parallel multiwriter progress.

use std::ffi::{CStr, CString};
use std::fmt;
use std::path::Path;
use std::ptr::NonNull;

use chronovec_sys as sys;

pub mod collection;

/// Distance metric. Cosine inputs are normalised by the index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    Cosine,
    L2,
}

impl Metric {
    fn raw(self) -> std::os::raw::c_int {
        match self {
            Metric::Cosine => sys::CV_COSINE,
            Metric::L2 => sys::CV_L2,
        }
    }
}

/// A logical point in time. Returned by writes; accepted by [`Index::search_as_of`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Snapshot(pub u64);

/// One search result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub id: i64,
    pub distance: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The vector length did not match the index dimensionality.
    Dimension { expected: usize, actual: usize },
    /// A path could not be represented as a C string.
    InvalidPath,
    /// The native layer reported a failure.
    Native(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Dimension { expected, actual } => {
                write!(f, "expected a {expected}-dimensional vector, got {actual}")
            }
            Error::InvalidPath => write!(f, "path is not valid for the native API"),
            Error::Native(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for Error {}

fn last_error() -> Error {
    // SAFETY: cv_last_error returns a thread-local NUL-terminated string.
    let raw = unsafe { sys::cv_last_error() };
    if raw.is_null() {
        return Error::Native("unknown native error".into());
    }
    Error::Native(
        unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Configuration for a new [`Index`]. Adding options here never breaks callers.
#[derive(Debug, Clone)]
pub struct Builder {
    dimensions: usize,
    metric: Metric,
    page_capacity: usize,
    nprobe: usize,
    screening: bool,
    rerank_factor: usize,
}

impl Builder {
    /// Distance metric. Defaults to cosine.
    pub fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }
    /// Vectors per page. Larger pages scan more per probe but split less often.
    pub fn page_capacity(mut self, capacity: usize) -> Self {
        self.page_capacity = capacity;
        self
    }
    /// Default number of pages probed per query. Higher is more accurate and slower.
    pub fn nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }
    /// Compressed candidate screening. On by default; disable to save memory.
    pub fn screening(mut self, enabled: bool) -> Self {
        self.screening = enabled;
        self
    }
    /// How many candidates per result are rescored exactly. Defaults to 4.
    pub fn rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor;
        self
    }

    pub fn build(self) -> Result<Index, Error> {
        let flags = if self.screening {
            sys::CV_ENABLE_INT8_SCREENING
        } else {
            0
        };
        // SAFETY: all arguments are plain values; the pointer is checked below.
        let raw = unsafe {
            sys::cv_create_with_options(
                self.dimensions,
                self.metric.raw(),
                self.page_capacity,
                self.nprobe,
                flags,
                self.rerank_factor,
            )
        };
        NonNull::new(raw)
            .map(|handle| Index {
                handle,
                dimensions: self.dimensions,
            })
            .ok_or_else(last_error)
    }
}

/// A snapshot-isolated vector index.
pub struct Index {
    handle: NonNull<sys::cv_index>,
    dimensions: usize,
}

// SAFETY: the C ABI documents searches as lock-free over immutable snapshots and
// mutations as thread-safe but serialised, so shared references are sound across
// threads and the handle may be dropped on any thread.
unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Index {
    /// Start configuring an index of the given dimensionality.
    pub fn builder(dimensions: usize) -> Builder {
        Builder {
            dimensions,
            metric: Metric::Cosine,
            page_capacity: 256,
            nprobe: 16,
            screening: true,
            rerank_factor: 4,
        }
    }

    fn check(&self, vector: &[f32]) -> Result<(), Error> {
        if vector.len() == self.dimensions {
            Ok(())
        } else {
            Err(Error::Dimension {
                expected: self.dimensions,
                actual: vector.len(),
            })
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// The current logical clock. `Snapshot(clock + 1)` sees all committed writes.
    pub fn clock(&self) -> Snapshot {
        Snapshot(unsafe { sys::cv_clock(self.handle.as_ptr()) })
    }

    /// Insert or replace `id`, returning the snapshot at which it became visible.
    pub fn insert(&self, id: i64, vector: &[f32]) -> Result<Snapshot, Error> {
        self.check(vector)?;
        let mut committed = 0u64;
        // SAFETY: `vector` is checked to have `dimensions` elements.
        let rc =
            unsafe { sys::cv_insert(self.handle.as_ptr(), id, vector.as_ptr(), 0, &mut committed) };
        if rc == 0 {
            Ok(Snapshot(committed))
        } else {
            Err(last_error())
        }
    }

    /// Insert many vectors. Equivalent to repeated [`Index::insert`], and the
    /// natural shape for ingestion pipelines.
    pub fn insert_many<'a, I>(&self, items: I) -> Result<Snapshot, Error>
    where
        I: IntoIterator<Item = (i64, &'a [f32])>,
    {
        let mut last = self.clock();
        for (id, vector) in items {
            last = self.insert(id, vector)?;
        }
        Ok(last)
    }

    /// Delete `id`. The version stops being visible immediately; its space is
    /// returned by [`Index::vacuum`].
    pub fn delete(&self, id: i64) -> Result<Snapshot, Error> {
        let mut committed = 0u64;
        let rc = unsafe { sys::cv_delete(self.handle.as_ptr(), id, 0, &mut committed) };
        if rc == 0 {
            Ok(Snapshot(committed))
        } else {
            Err(last_error())
        }
    }

    /// Nearest neighbours of `query` in the current state.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>, Error> {
        let mut out = Vec::new();
        self.search_into(query, k, None, &mut out)?;
        Ok(out)
    }

    /// Nearest neighbours as the index existed at `snapshot`.
    pub fn search_as_of(
        &self,
        query: &[f32],
        k: usize,
        snapshot: Snapshot,
    ) -> Result<Vec<Hit>, Error> {
        let mut out = Vec::new();
        self.search_into(query, k, Some(snapshot), &mut out)?;
        Ok(out)
    }

    /// Search, reusing `out`'s allocation. Use this in hot loops to keep the
    /// query path allocation-free.
    pub fn search_into(
        &self,
        query: &[f32],
        k: usize,
        snapshot: Option<Snapshot>,
        out: &mut Vec<Hit>,
    ) -> Result<usize, Error> {
        self.check(query)?;
        out.clear();
        if k == 0 {
            return Ok(0);
        }
        let mut ids = vec![0i64; k];
        let mut distances = vec![0f32; k];
        // The C ABI treats snapshot 0 as "now". Passing u64::MAX instead makes
        // the visibility test `snap < end` fail for live records, whose end is
        // itself u64::MAX, and every query returns nothing.
        let snap = snapshot.map(|s| s.0).unwrap_or(0);
        // SAFETY: `query` is checked; the output buffers hold `k` elements each.
        let found = unsafe {
            sys::cv_search(
                self.handle.as_ptr(),
                query.as_ptr(),
                k,
                snap,
                0,
                ids.as_mut_ptr(),
                distances.as_mut_ptr(),
            )
        };
        out.extend((0..found).map(|i| Hit {
            id: ids[i],
            distance: distances[i],
        }));
        Ok(found)
    }

    /// A snapshot just after every currently committed write. Pass this to
    /// [`Index::vacuum`] to reclaim everything eligible right now.
    pub fn horizon(&self) -> Snapshot {
        Snapshot(self.clock().0 + 1)
    }

    /// Physically reclaim at most `budget` versions no longer visible to any
    /// snapshot at or after `oldest`. Bounded: it does not scan the index.
    pub fn vacuum(&self, oldest: Snapshot, budget: usize) -> usize {
        unsafe { sys::cv_vacuum(self.handle.as_ptr(), oldest.0, budget) }
    }

    /// Write a checksummed checkpoint, replacing `path` atomically.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error> {
        let raw = CString::new(path.as_ref().to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidPath)?;
        let rc = unsafe { sys::cv_save_checkpoint(self.handle.as_ptr(), raw.as_ptr()) };
        if rc == 0 {
            Ok(())
        } else {
            Err(last_error())
        }
    }

    /// Restore an index from a checkpoint written by [`Index::save`].
    pub fn load(path: impl AsRef<Path>) -> Result<Index, Error> {
        let raw = CString::new(path.as_ref().to_string_lossy().as_bytes())
            .map_err(|_| Error::InvalidPath)?;
        let handle = NonNull::new(unsafe { sys::cv_load_checkpoint(raw.as_ptr()) })
            .ok_or_else(last_error)?;
        let dimensions = unsafe { sys::cv_dimensions(handle.as_ptr()) };
        Ok(Index { handle, dimensions })
    }

    /// Index statistics: live vectors, pages, reclaimed versions, and so on.
    pub fn stats(&self) -> sys::cv_stats {
        let mut out = sys::cv_stats::default();
        unsafe { sys::cv_get_stats(self.handle.as_ptr(), &mut out) };
        out
    }
}

impl Drop for Index {
    fn drop(&mut self) {
        unsafe { sys::cv_destroy(self.handle.as_ptr()) }
    }
}
