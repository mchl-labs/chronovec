//! Node.js native addon for ChronoVec, via napi-rs wrapping the existing Rust
//! `chronovec` crate rather than a fourth independent FFI implementation over
//! the C ABI directly.
//!
//! Two tiers, matching the same two-tier shape as Rust and Python:
//! - Raw (`Index`, this file): the same vocabulary as the C ABI, Rust's
//!   `chronovec::Index`, Python's `chronovec.Index`, and the Go binding
//!   (`insert`/`delete`/`search`/`vacuum`/`save`/`load`/`clock`, spelled
//!   camelCase per JS convention).
//! - Ergonomic (`Collection`, in `collection.rs`): the same vocabulary as
//!   Python's `chronovec.Collection` (`add`/`query`/`snapshot`/`vacuum`) and
//!   Rust's `chronovec::collection::Collection`, over string-or-int ids,
//!   metadata, and a filter DSL. Glue over the existing Rust `Collection`,
//!   not a new implementation.
//!
//! ids are `i64`, bridged through napi-rs as JS `BigInt` rather than
//! `number`: a JS `number` cannot losslessly represent the full `int64`
//! range the engine actually stores (ids can exceed
//! `Number.MAX_SAFE_INTEGER`), and silently truncating one would be a
//! correctness bug, not a convenience trade worth making.

#![deny(clippy::all)]

use std::sync::Arc;

use napi::bindgen_prelude::*;
use napi_derive::napi;

pub mod collection;

fn to_js_err(e: chronovec::Error) -> Error {
    Error::from_reason(e.to_string())
}

/// Every id and snapshot value crosses the boundary as a JS `BigInt`, not
/// `number`: napi-rs's `i64` type maps to a JS `number` (verified empirically
/// -- it doesn't even accept a `BigInt` argument, since a plain `number`
/// can't represent it), and a JS `number` cannot losslessly hold the full
/// `int64` range the engine actually uses. `stable_id()` in the Rust and
/// Python `Collection` layers hashes strings into that full range, well past
/// `Number.MAX_SAFE_INTEGER` -- so silently truncating here would be a real,
/// silent correctness bug the day a Node caller's ids come from that hash
/// rather than a small literal.
fn bigint_to_i64(b: BigInt) -> Result<i64> {
    let (value, lossless) = b.get_i64();
    if !lossless {
        return Err(Error::from_reason(
            "id/snapshot value does not fit in a 64-bit signed integer",
        ));
    }
    Ok(value)
}

/// Distance metric: `"cosine"` or `"l2"`.
fn parse_metric(metric: Option<String>) -> Result<chronovec::Metric> {
    match metric.as_deref() {
        None | Some("cosine") => Ok(chronovec::Metric::Cosine),
        Some("l2") => Ok(chronovec::Metric::L2),
        Some(other) => Err(Error::from_reason(format!(
            "unknown metric {other:?}, expected \"cosine\" or \"l2\""
        ))),
    }
}

/// Options for [`Index`]. Every field is optional; omitted fields use the
/// same defaults as the Rust crate's own `Index::builder`.
#[napi(object)]
#[derive(Default)]
pub struct IndexOptions {
    pub metric: Option<String>,
    pub page_capacity: Option<u32>,
    pub nprobe: Option<u32>,
    pub screening: Option<bool>,
    pub rerank_factor: Option<u32>,
}

/// One search result.
#[napi(object)]
pub struct Hit {
    pub id: BigInt,
    pub distance: f64,
}

/// Index-wide counters: live vectors, pages, reclaimed versions, and so on.
///
/// Plain `number`, unlike `id`/snapshot values: these are diagnostic counts
/// (pages, live vectors, reclaimed versions), not values a caller round-trips
/// back into the API, and reaching `Number.MAX_SAFE_INTEGER` here means
/// tens of petabytes of vectors -- not a realistic precision concern.
#[napi(object)]
pub struct Stats {
    pub clock: i64,
    pub pages: i64,
    pub live_vectors: i64,
    pub retained_versions: i64,
    pub allocated_slots: i64,
    pub splits: i64,
    pub reclaimed_versions: i64,
    pub merges: i64,
}

/// A snapshot-isolated approximate-nearest-neighbour index.
///
/// Searches are lock-free over immutable snapshots; mutations are
/// thread-safe but serialise internally, matching the underlying C ABI's own
/// contract -- documented here for completeness, though Node's single
/// JS-thread execution model means it rarely comes up directly the way it
/// would calling this from multiple native threads.
#[napi]
pub struct Index {
    inner: Arc<chronovec::Index>,
}

#[napi]
impl Index {
    #[napi(constructor)]
    pub fn new(dimensions: u32, options: Option<IndexOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();
        let mut builder =
            chronovec::Index::builder(dimensions as usize).metric(parse_metric(opts.metric)?);
        if let Some(pc) = opts.page_capacity {
            builder = builder.page_capacity(pc as usize);
        }
        if let Some(np) = opts.nprobe {
            builder = builder.nprobe(np as usize);
        }
        if let Some(s) = opts.screening {
            builder = builder.screening(s);
        }
        if let Some(rf) = opts.rerank_factor {
            builder = builder.rerank_factor(rf as usize);
        }
        let inner = builder.build().map_err(to_js_err)?;
        Ok(Index {
            inner: Arc::new(inner),
        })
    }

    #[napi]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    /// The current logical clock. `BigInt(clock()) + 1n` sees all committed writes.
    #[napi]
    pub fn clock(&self) -> BigInt {
        BigInt::from(self.inner.clock().0 as i64)
    }

    /// Insert or replace `id`, returning the snapshot at which it became visible.
    #[napi]
    pub fn insert(&self, id: BigInt, vector: Float32Array) -> Result<BigInt> {
        let id = bigint_to_i64(id)?;
        let snap = self.inner.insert(id, &vector).map_err(to_js_err)?;
        Ok(BigInt::from(snap.0 as i64))
    }

    /// Delete `id`. Its space is returned by `vacuum`.
    #[napi]
    pub fn delete(&self, id: BigInt) -> Result<BigInt> {
        let id = bigint_to_i64(id)?;
        let snap = self.inner.delete(id).map_err(to_js_err)?;
        Ok(BigInt::from(snap.0 as i64))
    }

    /// The `k` nearest neighbours of `query` in the current state.
    #[napi]
    pub fn search(&self, query: Float32Array, k: u32) -> Result<Vec<Hit>> {
        self.search_impl(query, k, None)
    }

    /// The `k` nearest neighbours as the index existed at `snapshot`.
    #[napi(js_name = "searchAsOf")]
    pub fn search_as_of(&self, query: Float32Array, k: u32, snapshot: BigInt) -> Result<Vec<Hit>> {
        let snapshot = bigint_to_i64(snapshot)?;
        self.search_impl(query, k, Some(chronovec::Snapshot(snapshot as u64)))
    }

    fn search_impl(
        &self,
        query: Float32Array,
        k: u32,
        snapshot: Option<chronovec::Snapshot>,
    ) -> Result<Vec<Hit>> {
        let hits = match snapshot {
            Some(s) => self.inner.search_as_of(&query, k as usize, s),
            None => self.inner.search(&query, k as usize),
        }
        .map_err(to_js_err)?;
        Ok(hits
            .into_iter()
            .map(|h| Hit {
                id: BigInt::from(h.id),
                distance: h.distance as f64,
            })
            .collect())
    }

    /// Physically reclaim at most `budget` versions no longer visible to any
    /// snapshot at or after `oldest`. Bounded: it does not scan the index.
    #[napi]
    pub fn vacuum(&self, oldest: BigInt, budget: u32) -> Result<u32> {
        let oldest = bigint_to_i64(oldest)?;
        Ok(self
            .inner
            .vacuum(chronovec::Snapshot(oldest as u64), budget as usize) as u32)
    }

    /// Write a checksummed checkpoint, replacing `path` atomically.
    #[napi]
    pub fn save(&self, path: String) -> Result<()> {
        self.inner.save(path).map_err(to_js_err)
    }

    /// Restore an index from a checkpoint written by `save`.
    #[napi(factory)]
    pub fn load(path: String) -> Result<Self> {
        let inner = chronovec::Index::load(path).map_err(to_js_err)?;
        Ok(Index {
            inner: Arc::new(inner),
        })
    }

    /// Index statistics: live vectors, pages, reclaimed versions, and so on.
    #[napi]
    pub fn stats(&self) -> Stats {
        let s = self.inner.stats();
        Stats {
            clock: s.clock as i64,
            pages: s.pages as i64,
            live_vectors: s.live_vectors as i64,
            retained_versions: s.retained_versions as i64,
            allocated_slots: s.allocated_slots as i64,
            splits: s.splits as i64,
            reclaimed_versions: s.reclaimed_versions as i64,
            merges: s.merges as i64,
        }
    }
}
