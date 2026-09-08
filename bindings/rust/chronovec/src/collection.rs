//! An ergonomic layer over [`crate::Index`]: string-or-int ids, metadata, filtering,
//! and snapshots -- the same vocabulary as Python's `chronovec.Collection`
//! (`add`/`query`/`snapshot`/`vacuum`), not the raw engine's (`insert`/`search`/`clock`).
//!
//! [`crate::Index`] already mirrors the raw tier across every language binding
//! (C ABI, Rust, Python's `Index`) with identical names. This module is the
//! Rust half of the *other* tier -- the one Python users reach for by default
//! -- so a caller moving between the two languages finds the same two-tier
//! shape and the same names within each tier, not a flattened, renamed API.
//!
//! ```no_run
//! use chronovec::collection::{Collection, Value};
//! use std::collections::HashMap;
//!
//! let col = Collection::new(3)?;
//! let mut meta = HashMap::new();
//! meta.insert("lang".to_string(), Value::Str("en".into()));
//! col.add("doc-a", &[0.1, 0.2, 0.3], Some(meta), Some("hello".into()))?;
//!
//! let hits = col.query(&[0.1, 0.2, 0.3], 5, Default::default())?;
//! assert_eq!(hits[0].id, chronovec::collection::Id::Str("doc-a".into()));
//! # Ok::<(), chronovec::Error>(())
//! ```

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Mutex;

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

use crate::{Error, Index, Snapshot};

/// A record id: as given by the caller, before hashing into the engine's `i64` key.
///
/// Integers pass through to the engine unchanged -- a caller already using
/// int ids keeps them readable in `Index::stats()`. Strings are hashed with
/// [`stable_id`]: deterministic across processes and across every language
/// binding, so a record keeps its identity across a restart and across
/// Python/Rust/Go/Node all touching the same index.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Id {
    Str(String),
    Int(i64),
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Id::Str(s) => write!(f, "{s}"),
            Id::Int(i) => write!(f, "{i}"),
        }
    }
}

impl From<&str> for Id {
    fn from(s: &str) -> Self {
        Id::Str(s.to_string())
    }
}
impl From<String> for Id {
    fn from(s: String) -> Self {
        Id::Str(s)
    }
}
impl From<i64> for Id {
    fn from(i: i64) -> Self {
        Id::Int(i)
    }
}

/// Maps an id onto the `i64` the engine uses.
///
/// Must match `chronovec.collection.stable_id()` in the Python package
/// byte-for-byte: BLAKE2b, digest_size 8, big-endian, right-shifted one bit
/// so the top bit is always clear (the engine writes -1 into an empty slot,
/// and this guarantees a hashed id never collides with it). A record added
/// from Python and read from Rust -- or the reverse -- must resolve to the
/// same engine key, or the two bindings are silently looking at different
/// records under the same string id.
pub fn stable_id(id: &Id) -> i64 {
    match id {
        Id::Int(i) => *i,
        Id::Str(s) => {
            let mut hasher = Blake2bVar::new(8).expect("8 is a valid BLAKE2b output size");
            hasher.update(s.as_bytes());
            let mut digest = [0u8; 8];
            hasher
                .finalize_variable(&mut digest)
                .expect("buffer is exactly 8 bytes");
            (u64::from_be_bytes(digest) >> 1) as i64
        }
    }
}

/// A metadata value. Deliberately just the JSON-like scalars Python's
/// `dict[str, Any]` metadata realistically holds -- no arrays, no nesting.
/// The Python side has no static type to translate here, so the Rust side
/// declares only what the filter operators below can actually reason about.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_string())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Str(s)
    }
}
impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i)
    }
}
impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}
impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

fn compare(have: &Value, want: &Value) -> Option<std::cmp::Ordering> {
    match (have, want) {
        (Value::Int(a), Value::Int(b)) => a.partial_cmp(b),
        (Value::Float(a), Value::Float(b)) => a.partial_cmp(b),
        (Value::Int(a), Value::Float(b)) => (*a as f64).partial_cmp(b),
        (Value::Float(a), Value::Int(b)) => a.partial_cmp(&(*b as f64)),
        (Value::Str(a), Value::Str(b)) => a.partial_cmp(b),
        // A caller comparing a bool, or comparing across incompatible types,
        // gets "doesn't match" rather than a panic -- the same behaviour as
        // an absent field, since the comparison is meaningless either way.
        _ => None,
    }
}

/// A metadata filter, evaluated after the vector search over a widened
/// candidate set -- same semantics as Python's `where=` dict. The operator
/// vocabulary is identical (`$eq $ne $gt $gte $lt $lte $in $nin $contains
/// $regex $and $or`); the shape is a Rust enum instead of a dict of string
/// keys because that is what "the same operators, idiomatically" means in a
/// statically typed language.
#[derive(Debug, Clone)]
pub enum Filter {
    Eq(String, Value),
    Ne(String, Value),
    Gt(String, Value),
    Gte(String, Value),
    Lt(String, Value),
    Lte(String, Value),
    In(String, Vec<Value>),
    NotIn(String, Vec<Value>),
    Contains(String, String),
    Regex(String, regex_lite::Regex),
    And(Vec<Filter>),
    Or(Vec<Filter>),
}

fn matches(metadata: &HashMap<String, Value>, filter: &Filter) -> bool {
    match filter {
        Filter::Eq(key, want) => metadata.get(key) == Some(want),
        Filter::Ne(key, want) => metadata.get(key) != Some(want),
        Filter::Gt(key, want) => metadata
            .get(key)
            .and_then(|have| compare(have, want))
            .is_some_and(|o| o.is_gt()),
        Filter::Gte(key, want) => metadata
            .get(key)
            .and_then(|have| compare(have, want))
            .is_some_and(|o| !o.is_lt()),
        Filter::Lt(key, want) => metadata
            .get(key)
            .and_then(|have| compare(have, want))
            .is_some_and(|o| o.is_lt()),
        Filter::Lte(key, want) => metadata
            .get(key)
            .and_then(|have| compare(have, want))
            .is_some_and(|o| !o.is_gt()),
        Filter::In(key, options) => metadata.get(key).is_some_and(|have| options.contains(have)),
        Filter::NotIn(key, options) => {
            !metadata.get(key).is_some_and(|have| options.contains(have))
        }
        Filter::Contains(key, needle) => match metadata.get(key) {
            Some(Value::Str(have)) => have.contains(needle.as_str()),
            _ => false,
        },
        Filter::Regex(key, re) => match metadata.get(key) {
            Some(Value::Str(have)) => re.is_match(have),
            _ => false,
        },
        Filter::And(parts) => parts.iter().all(|f| matches(metadata, f)),
        Filter::Or(parts) => parts.iter().any(|f| matches(metadata, f)),
    }
}

/// One result from [`Collection::query`], [`Collection::get`], or [`Collection::get_where`].
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    pub id: Id,
    /// `None` for a record returned by `get`/`get_where` -- there was no query vector.
    pub distance: Option<f32>,
    pub metadata: HashMap<String, Value>,
    pub document: Option<String>,
}

/// Options for [`Collection::query`]. `Default::default()` is "now, no filter,
/// default overfetch" -- the equivalent of Python's `col.query(v, k)` with no
/// further keyword arguments.
#[derive(Default)]
pub struct QueryOptions<'a> {
    pub snapshot: Option<Snapshot>,
    pub filter: Option<&'a Filter>,
    /// How many candidates past `k` to fetch before applying `filter`. `0`
    /// (the default) means the crate's own default, not "fetch zero".
    pub overfetch: usize,
}

type Version = (Snapshot, HashMap<String, Value>, Option<String>);

/// Everything [`Index`] does not carry, behind one lock.
///
/// Deliberately one `Mutex` guarding all four maps rather than four
/// independent ones: `query` needs `names` while resolving `history` per
/// hit, and `vacuum` needs `history` while pruning `names` -- two different
/// orderings over the same two structures. Four separate locks taken in
/// inconsistent order across methods is a real deadlock under concurrent
/// use, which is exactly the case `&self` (mirroring [`Index`]'s own
/// thread-safety promise) exists to support. One lock has no ordering to
/// get wrong. The critical sections here are small HashMap operations, not
/// the vector search itself (which stays lock-free in the engine), so the
/// coarser granularity costs little.
#[derive(Default)]
struct Sidecar {
    history: HashMap<i64, Vec<Version>>,
    names: HashMap<i64, Id>,
    live: HashSet<i64>,
    deleted_at: HashMap<i64, Snapshot>,
}

/// Vectors with ids, metadata, filtering, and snapshots.
///
/// Everything [`Index`] does not carry -- metadata, documents, the mapping
/// from a caller's id to the engine's `i64` key -- is kept here, versioned
/// the same way Python's `Collection` keeps it: a snapshot taken before an
/// update can still resolve the metadata as it was, not the metadata that
/// happens to be current when the read runs.
pub struct Collection {
    index: Index,
    state: Mutex<Sidecar>,
}

impl Collection {
    /// A collection of the given dimensionality with default index options
    /// (cosine metric, page capacity 256, nprobe 16 -- same defaults as
    /// [`Index::builder`]). For anything beyond the defaults, configure an
    /// [`Index`] with [`Index::builder`] and pass it to [`Collection::wrap`].
    pub fn new(dimensions: usize) -> Result<Collection, Error> {
        Ok(Collection::wrap(Index::builder(dimensions).build()?))
    }

    /// Wrap an already-built [`Index`]. Use this when index-level options
    /// beyond [`Collection::new`]'s defaults are needed (metric, a WAL path,
    /// ...), or to add the ergonomic layer over an index built elsewhere.
    pub fn wrap(index: Index) -> Collection {
        Collection {
            index,
            state: Mutex::new(Sidecar::default()),
        }
    }

    /// The metadata/document in force at `snapshot` (or now) for one key,
    /// read from an already-locked [`Sidecar`] -- never takes the lock
    /// itself, so callers holding it can call this without deadlocking.
    fn version_at_locked(
        state: &Sidecar,
        key: i64,
        snapshot: Option<Snapshot>,
    ) -> (HashMap<String, Value>, Option<String>) {
        let Some(versions) = state.history.get(&key) else {
            return (HashMap::new(), None);
        };
        match snapshot {
            None => versions
                .last()
                .map(|(_, m, d)| (m.clone(), d.clone()))
                .unwrap_or_default(),
            Some(at) => versions
                .iter()
                .rev()
                .find(|(stamp, _, _)| *stamp <= at)
                .map(|(_, m, d)| (m.clone(), d.clone()))
                .or_else(|| versions.first().map(|(_, m, d)| (m.clone(), d.clone())))
                .unwrap_or_default(),
        }
    }

    /// Insert or replace one record.
    ///
    /// `metadata`/`document` are carried forward from the previous version
    /// when `None` -- matching Python's `Collection.add()`, which carries
    /// forward when its `metadatas`/`documents` arguments are omitted
    /// entirely. Pass `Some(HashMap::new())` to explicitly clear metadata to
    /// empty (as opposed to `None`, which keeps whatever it was). There is
    /// no way to explicitly clear an existing document back to "no
    /// document" while keeping the metadata -- neither can Python's
    /// `Collection.add()`, whose per-item `documents` type is `str`, not
    /// `str | None`, once the `documents` list itself is given.
    pub fn add(
        &self,
        id: impl Into<Id>,
        vector: &[f32],
        metadata: Option<HashMap<String, Value>>,
        document: Option<String>,
    ) -> Result<Snapshot, Error> {
        let id = id.into();
        let key = stable_id(&id);
        let stamp = self.index.insert(key, vector)?;
        let mut state = self.state.lock().unwrap();
        let (metadata, document) = match (metadata, document) {
            (Some(m), Some(d)) => (m, Some(d)),
            (Some(m), None) => {
                let (_, carried_doc) = Self::version_at_locked(&state, key, None);
                (m, carried_doc)
            }
            (None, Some(d)) => {
                let (carried_meta, _) = Self::version_at_locked(&state, key, None);
                (carried_meta, Some(d))
            }
            (None, None) => Self::version_at_locked(&state, key, None),
        };
        state
            .history
            .entry(key)
            .or_default()
            .push((stamp, metadata, document));
        state.names.insert(key, id);
        state.live.insert(key);
        state.deleted_at.remove(&key);
        Ok(stamp)
    }

    /// Insert or replace many records. Equivalent to repeated [`Collection::add`].
    pub fn add_many<'a, I>(&self, items: I) -> Result<Snapshot, Error>
    where
        I: IntoIterator<
            Item = (
                Id,
                &'a [f32],
                Option<HashMap<String, Value>>,
                Option<String>,
            ),
        >,
    {
        let mut last = self.index.clock();
        for (id, vector, metadata, document) in items {
            last = self.add(id, vector, metadata, document)?;
        }
        Ok(last)
    }

    /// Change an existing record's metadata without touching its vector.
    /// Errors if `id` is not currently live.
    pub fn update_metadata(
        &self,
        id: impl Into<Id>,
        metadata: HashMap<String, Value>,
    ) -> Result<(), Error> {
        let id = id.into();
        let key = stable_id(&id);
        let mut state = self.state.lock().unwrap();
        if !state.live.contains(&key) {
            return Err(Error::Native(format!("unknown id: {id}")));
        }
        let stamp = self.index.clock();
        let (current, document) = Self::version_at_locked(&state, key, None);
        let mut merged = current;
        merged.extend(metadata);
        state
            .history
            .entry(key)
            .or_default()
            .push((stamp, merged, document));
        Ok(())
    }

    /// Remove one record. Returns whether it was live.
    pub fn delete(&self, id: impl Into<Id>) -> Result<bool, Error> {
        let id = id.into();
        let key = stable_id(&id);
        if !self.state.lock().unwrap().live.remove(&key) {
            return Ok(false);
        }
        let stamp = self.index.delete(key)?;
        self.state.lock().unwrap().deleted_at.insert(key, stamp);
        Ok(true)
    }

    /// Remove every live record matching `filter`. Returns their ids.
    pub fn delete_where(&self, filter: &Filter) -> Result<Vec<Id>, Error> {
        // Candidates and per-record metadata are read under one lock
        // acquisition each -- not held across the `index.delete` native
        // call, so a concurrent reader is never blocked on a vector delete.
        let candidates: Vec<(i64, HashMap<String, Value>)> = {
            let state = self.state.lock().unwrap();
            state
                .live
                .iter()
                .map(|&key| (key, Self::version_at_locked(&state, key, None).0))
                .collect()
        };
        let mut removed = Vec::new();
        for (key, metadata) in candidates {
            if !matches(&metadata, filter) {
                continue;
            }
            self.index.delete(key)?;
            let stamp = self.index.clock();
            let mut state = self.state.lock().unwrap();
            state.live.remove(&key);
            state.deleted_at.insert(key, stamp);
            if let Some(id) = state.names.get(&key).cloned() {
                removed.push(id);
            }
        }
        Ok(removed)
    }

    /// Nearest records to `query`, optionally filtered and optionally as of
    /// a past snapshot.
    pub fn query(
        &self,
        query: &[f32],
        k: usize,
        options: QueryOptions,
    ) -> Result<Vec<Record>, Error> {
        let overfetch = if options.overfetch == 0 {
            8
        } else {
            options.overfetch
        };
        let want = if options.filter.is_some() {
            k * overfetch.max(1)
        } else {
            k
        };
        let hits = match options.snapshot {
            Some(s) => self.index.search_as_of(query, want, s)?,
            None => self.index.search(query, want)?,
        };
        let state = self.state.lock().unwrap();
        let mut out = Vec::with_capacity(k);
        for hit in hits {
            let (metadata, document) = Self::version_at_locked(&state, hit.id, options.snapshot);
            if let Some(filter) = options.filter {
                if !matches(&metadata, filter) {
                    continue;
                }
            }
            out.push(Record {
                id: state.names.get(&hit.id).cloned().unwrap_or(Id::Int(hit.id)),
                distance: Some(hit.distance),
                metadata,
                document,
            });
            if out.len() == k {
                break;
            }
        }
        Ok(out)
    }

    /// Fetch one live record by id, without a vector search.
    pub fn get(&self, id: impl Into<Id>) -> Option<Record> {
        let id = id.into();
        let key = stable_id(&id);
        let state = self.state.lock().unwrap();
        if !state.live.contains(&key) {
            return None;
        }
        let (metadata, document) = Self::version_at_locked(&state, key, None);
        Some(Record {
            id,
            distance: None,
            metadata,
            document,
        })
    }

    /// Fetch every live record matching `filter`, without a vector search.
    pub fn get_where(&self, filter: &Filter) -> Vec<Record> {
        let state = self.state.lock().unwrap();
        let mut out = Vec::new();
        for &key in state.live.iter() {
            let (metadata, document) = Self::version_at_locked(&state, key, None);
            if !matches(&metadata, filter) {
                continue;
            }
            out.push(Record {
                id: state.names.get(&key).cloned().unwrap_or(Id::Int(key)),
                distance: None,
                metadata,
                document,
            });
        }
        out
    }

    pub fn count(&self) -> usize {
        self.state.lock().unwrap().live.len()
    }

    pub fn dimensions(&self) -> usize {
        self.index.dimensions()
    }

    /// A token for the present, readable later via [`QueryOptions::snapshot`].
    pub fn snapshot(&self) -> Snapshot {
        self.index.clock()
    }

    /// Reclaim versions no reader at or after `keep_snapshot` can still
    /// reach. `None` reclaims everything currently unreachable.
    pub fn vacuum(&self, keep_snapshot: Option<Snapshot>) -> usize {
        let horizon = keep_snapshot.unwrap_or_else(|| self.index.horizon());
        let freed = self.index.vacuum(horizon, 0);
        self.prune_history(horizon);
        freed
    }

    fn prune_history(&self, horizon: Snapshot) {
        let mut state = self.state.lock().unwrap();
        let keys: Vec<i64> = state.history.keys().copied().collect();
        for key in keys {
            if let Some(versions) = state.history.get_mut(&key) {
                let keep_from = versions
                    .iter()
                    .rposition(|(stamp, _, _)| *stamp <= horizon)
                    .unwrap_or(0);
                versions.drain(..keep_from);
            }
            if let Some(gone) = state.deleted_at.get(&key) {
                if *gone <= horizon {
                    state.history.remove(&key);
                    state.names.remove(&key);
                    state.deleted_at.remove(&key);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_id_matches_python() {
        // Generated with: python -c "from chronovec.collection import
        // stable_id; print(stable_id('a'))" -- one fixture per input class
        // (ascii, hyphenated, spaced, empty, unicode) so a hashing bug that
        // only shows up on non-ASCII input, or on the empty string, is caught.
        let cases: &[(&str, i64)] = &[
            ("a", 2340832890917691671),
            ("doc-1", 3132883647696894624),
            ("user-preference-42", 4816184505566365106),
            ("hello world", 4882774824043272360),
            ("", 8238016292129134938),
            ("unicode-café-🎉", 4281077293084237012),
        ];
        for (input, expected) in cases {
            let got = stable_id(&Id::Str(input.to_string()));
            assert_eq!(got, *expected, "mismatch for {input:?}");
        }
    }

    #[test]
    fn int_id_passes_through() {
        assert_eq!(stable_id(&Id::Int(42)), 42);
    }

    fn meta(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn filter_eq_and_boolean_composition() {
        let m = meta(&[
            ("lang", Value::Str("en".into())),
            ("year", Value::Int(2024)),
        ]);
        assert!(matches(
            &m,
            &Filter::Eq("lang".into(), Value::Str("en".into()))
        ));
        assert!(!matches(
            &m,
            &Filter::Eq("lang".into(), Value::Str("fr".into()))
        ));
        assert!(matches(
            &m,
            &Filter::And(vec![
                Filter::Eq("lang".into(), Value::Str("en".into())),
                Filter::Gte("year".into(), Value::Int(2020)),
            ])
        ));
        assert!(!matches(
            &m,
            &Filter::And(vec![
                Filter::Eq("lang".into(), Value::Str("en".into())),
                Filter::Gte("year".into(), Value::Int(2025)),
            ])
        ));
        assert!(matches(
            &m,
            &Filter::Or(vec![
                Filter::Eq("lang".into(), Value::Str("fr".into())),
                Filter::Eq("lang".into(), Value::Str("en".into())),
            ])
        ));
    }

    #[test]
    fn filter_missing_key_never_matches_comparisons() {
        let m = meta(&[("lang", Value::Str("en".into()))]);
        assert!(!matches(&m, &Filter::Gt("year".into(), Value::Int(2000))));
        assert!(!matches(&m, &Filter::Eq("year".into(), Value::Int(2000))));
        assert!(matches(&m, &Filter::Ne("year".into(), Value::Int(2000))));
    }

    #[test]
    fn filter_contains_and_in_and_regex() {
        let m = meta(&[
            ("title", Value::Str("ChronoVec MVCC".into())),
            ("year", Value::Int(2024)),
        ]);
        assert!(matches(
            &m,
            &Filter::Contains("title".into(), "MVCC".into())
        ));
        assert!(!matches(
            &m,
            &Filter::Contains("title".into(), "SQL".into())
        ));
        assert!(matches(
            &m,
            &Filter::In("year".into(), vec![Value::Int(2023), Value::Int(2024)])
        ));
        assert!(!matches(
            &m,
            &Filter::NotIn("year".into(), vec![Value::Int(2023), Value::Int(2024)])
        ));
        let re = regex_lite::Regex::new(r"^Chrono.*").unwrap();
        assert!(matches(&m, &Filter::Regex("title".into(), re.clone())));
        let re_no = regex_lite::Regex::new(r"^SQL.*").unwrap();
        assert!(!matches(&m, &Filter::Regex("title".into(), re_no)));
    }

    #[test]
    fn add_query_snapshot_roundtrip() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "en".into())])),
            Some("first".into()),
        )
        .unwrap();
        let before = col.snapshot();
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "fr".into())])),
            Some("second".into()),
        )
        .unwrap();

        let now = col.query(&[1.0, 0.0, 0.0], 1, Default::default()).unwrap();
        assert_eq!(now[0].document.as_deref(), Some("second"));

        let then = col
            .query(
                &[1.0, 0.0, 0.0],
                1,
                QueryOptions {
                    snapshot: Some(before),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(then[0].document.as_deref(), Some("first"));
    }

    #[test]
    fn query_with_filter_excludes_non_matching() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "en".into())])),
            None,
        )
        .unwrap();
        col.add(
            "b",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "fr".into())])),
            None,
        )
        .unwrap();

        let filter = Filter::Eq("lang".into(), Value::Str("fr".into()));
        let hits = col
            .query(
                &[1.0, 0.0, 0.0],
                5,
                QueryOptions {
                    filter: Some(&filter),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, Id::Str("b".into()));
    }

    #[test]
    fn delete_removes_from_count_and_get() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add("a", &[1.0, 0.0, 0.0], Some(HashMap::new()), None)
            .unwrap();
        assert_eq!(col.count(), 1);
        assert!(col.delete("a").unwrap());
        assert_eq!(col.count(), 0);
        assert!(col.get("a").is_none());
        assert!(!col.delete("a").unwrap()); // already gone
    }

    #[test]
    fn delete_where_removes_matching_only() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("kind", "hyp".into())])),
            None,
        )
        .unwrap();
        col.add(
            "b",
            &[0.0, 1.0, 0.0],
            Some(meta(&[("kind", "fact".into())])),
            None,
        )
        .unwrap();

        let removed = col
            .delete_where(&Filter::Eq("kind".into(), Value::Str("hyp".into())))
            .unwrap();
        assert_eq!(removed, vec![Id::Str("a".into())]);
        assert_eq!(col.count(), 1);
        assert!(col.get("b").is_some());
    }

    #[test]
    fn update_metadata_merges_and_is_visible_after() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "en".into())])),
            Some("doc".into()),
        )
        .unwrap();
        col.update_metadata("a", meta(&[("score", Value::Int(5))]))
            .unwrap();
        let r = col.get("a").unwrap();
        assert_eq!(r.metadata.get("lang"), Some(&Value::Str("en".into())));
        assert_eq!(r.metadata.get("score"), Some(&Value::Int(5)));
        assert_eq!(r.document.as_deref(), Some("doc")); // unchanged
    }

    #[test]
    fn update_metadata_unknown_id_errors() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        assert!(col.update_metadata("missing", HashMap::new()).is_err());
    }

    #[test]
    fn add_with_none_metadata_carries_forward_previous() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "en".into())])),
            Some("v1".into()),
        )
        .unwrap();
        // Re-add with metadata=None, document=None: the vector changes, but
        // the previous metadata and document must survive unchanged.
        col.add("a", &[0.0, 1.0, 0.0], None, None).unwrap();
        let r = col.get("a").unwrap();
        assert_eq!(r.metadata.get("lang"), Some(&Value::Str("en".into())));
        assert_eq!(r.document.as_deref(), Some("v1"));
    }

    #[test]
    fn add_with_some_metadata_replaces_even_if_empty() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "en".into())])),
            None,
        )
        .unwrap();
        // Some(empty map) is an explicit replace, not "keep" -- distinct
        // from None.
        col.add("a", &[1.0, 0.0, 0.0], Some(HashMap::new()), None)
            .unwrap();
        let r = col.get("a").unwrap();
        assert!(r.metadata.is_empty());
    }

    #[test]
    fn add_can_change_metadata_while_carrying_forward_document() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "en".into())])),
            Some("original".into()),
        )
        .unwrap();
        col.add(
            "a",
            &[1.0, 0.0, 0.0],
            Some(meta(&[("lang", "fr".into())])),
            None, // carry forward the document
        )
        .unwrap();
        let r = col.get("a").unwrap();
        assert_eq!(r.metadata.get("lang"), Some(&Value::Str("fr".into())));
        assert_eq!(r.document.as_deref(), Some("original"));
    }

    #[test]
    fn add_first_version_with_none_metadata_is_empty_not_an_error() {
        let col = Collection::wrap(Index::builder(3).build().unwrap());
        // Nothing to carry forward on a genuinely new id -- None just means
        // "no metadata", same as it would for any other empty default.
        col.add("a", &[1.0, 0.0, 0.0], None, None).unwrap();
        let r = col.get("a").unwrap();
        assert!(r.metadata.is_empty());
        assert!(r.document.is_none());
    }
}
