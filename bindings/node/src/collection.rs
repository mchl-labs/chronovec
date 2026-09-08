//! Ergonomic `Collection` binding, matching Python's vocabulary
//! (`add`/`query`/`snapshot`/`vacuum`) rather than the raw `Index` tier's
//! (`insert`/`search`/`clock`) exposed alongside it in `lib.rs`.
//!
//! This is glue, not new logic: `chronovec::collection::Collection` already
//! exists in the Rust crate (built and tested this session) with string-or-int
//! ids, metadata, a filter DSL matching Python's `where=` operator set, and
//! snapshot-isolated queries. Everything here is type bridging between that
//! and JS -- the reason this binding was cheap relative to a from-scratch
//! implementation (e.g. Go's raw tier).
//!
//! `where` filters and metadata cross the boundary as plain JS objects
//! (`{lang: "en"}`, `{year: {$gte: 2024}}`) via `serde_json::Value`, mirroring
//! Python's `where=` dict shape exactly rather than Rust's `Filter` enum --
//! that enum is the right *Rust* idiom, but a JS/TS caller coming from
//! Python's Collection or Chroma-style filters expects an object literal,
//! not a discriminated union type they'd have to construct by hand.

use std::collections::HashMap;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::Value as Json;

use chronovec::collection::{
    self as cv, Collection as RustCollection, Filter, Id as CvId, QueryOptions, Value as CvValue,
};

fn to_js_err(e: chronovec::Error) -> Error {
    Error::from_reason(e.to_string())
}

fn either_to_id(id: Either<String, BigInt>) -> Result<CvId> {
    match id {
        Either::A(s) => Ok(CvId::Str(s)),
        Either::B(b) => {
            let (value, lossless) = b.get_i64();
            if !lossless {
                return Err(Error::from_reason(
                    "integer id does not fit in a 64-bit signed integer",
                ));
            }
            Ok(CvId::Int(value))
        }
    }
}

fn id_to_js(id: &CvId) -> Either<String, BigInt> {
    match id {
        CvId::Str(s) => Either::A(s.clone()),
        CvId::Int(i) => Either::B(BigInt::from(*i)),
    }
}

/// A scalar JSON value -> the metadata `Value` the Rust Collection stores.
/// Arrays, objects, and null aren't supported metadata values -- same
/// restriction the Rust and Python Collection layers already have, since the
/// filter operators below can't reason about anything else.
fn json_to_cv_value(v: &Json) -> Result<CvValue> {
    match v {
        Json::String(s) => Ok(CvValue::Str(s.clone())),
        Json::Bool(b) => Ok(CvValue::Bool(*b)),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(CvValue::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(CvValue::Float(f))
            } else {
                Err(Error::from_reason(format!("unsupported number: {n}")))
            }
        }
        other => Err(Error::from_reason(format!(
            "unsupported metadata value {other}: only string, number, and boolean are allowed"
        ))),
    }
}

fn cv_value_to_json(v: &CvValue) -> Json {
    match v {
        CvValue::Str(s) => Json::String(s.clone()),
        CvValue::Int(i) => Json::Number((*i).into()),
        CvValue::Float(f) => serde_json::Number::from_f64(*f).map_or(Json::Null, Json::Number),
        CvValue::Bool(b) => Json::Bool(*b),
    }
}

fn json_to_metadata(v: &Json) -> Result<HashMap<String, CvValue>> {
    let Json::Object(map) = v else {
        return Err(Error::from_reason("metadata must be a plain object"));
    };
    map.iter()
        .map(|(k, v)| Ok((k.clone(), json_to_cv_value(v)?)))
        .collect()
}

/// `None` means "not provided" and passes straight through as `None` --
/// [`Collection::add`] carries forward the previous version's metadata for
/// a `None`, the same as omitting the argument entirely does in Python's
/// `Collection.add()`. `Some(json)` is an explicit replacement, parsed the
/// same way [`json_to_metadata`] does (including `Some(Json::Object({}))`,
/// which explicitly clears metadata to empty -- distinct from `None`).
fn json_to_metadata_opt(v: Option<Json>) -> Result<Option<HashMap<String, CvValue>>> {
    v.map(|json| json_to_metadata(&json)).transpose()
}

fn metadata_to_json(m: &HashMap<String, CvValue>) -> Json {
    Json::Object(
        m.iter()
            .map(|(k, v)| (k.clone(), cv_value_to_json(v)))
            .collect(),
    )
}

/// Parses a `where=`-shaped JS object into a [`Filter`]. A plain object is an
/// implicit AND of its keys; a key's value is either a scalar (`$eq`
/// shorthand) or an object of operators; `$and`/`$or` take an array of
/// nested filter objects. Mirrors `chronovec/collection.py`'s `matches()`
/// parsing exactly, operator-for-operator.
fn parse_filter(v: &Json) -> Result<Filter> {
    let Json::Object(map) = v else {
        return Err(Error::from_reason("where must be a plain object"));
    };
    let mut parts = Vec::with_capacity(map.len());
    for (key, condition) in map {
        if key == "$and" {
            parts.push(Filter::And(parse_filter_array(condition)?));
            continue;
        }
        if key == "$or" {
            parts.push(Filter::Or(parse_filter_array(condition)?));
            continue;
        }
        parts.push(parse_condition(key, condition)?);
    }
    Ok(if parts.len() == 1 {
        parts.remove(0)
    } else {
        Filter::And(parts)
    })
}

fn parse_filter_array(v: &Json) -> Result<Vec<Filter>> {
    let Json::Array(items) = v else {
        return Err(Error::from_reason(
            "$and/$or must be an array of filter objects",
        ));
    };
    items.iter().map(parse_filter).collect()
}

fn parse_condition(key: &str, condition: &Json) -> Result<Filter> {
    let Json::Object(ops) = condition else {
        // Bare scalar: `{lang: "en"}` means `{lang: {$eq: "en"}}`.
        return Ok(Filter::Eq(key.to_string(), json_to_cv_value(condition)?));
    };
    // An object with no recognised operator key is itself treated as an
    // equality target only when empty; otherwise every key must be a known
    // operator, same as Python's `matches()` raising on an unknown one.
    let mut result: Option<Filter> = None;
    for (op, want) in ops {
        let filter = match op.as_str() {
            "$eq" => Filter::Eq(key.to_string(), json_to_cv_value(want)?),
            "$ne" => Filter::Ne(key.to_string(), json_to_cv_value(want)?),
            "$gt" => Filter::Gt(key.to_string(), json_to_cv_value(want)?),
            "$gte" => Filter::Gte(key.to_string(), json_to_cv_value(want)?),
            "$lt" => Filter::Lt(key.to_string(), json_to_cv_value(want)?),
            "$lte" => Filter::Lte(key.to_string(), json_to_cv_value(want)?),
            "$in" => Filter::In(key.to_string(), parse_value_array(want)?),
            "$nin" => Filter::NotIn(key.to_string(), parse_value_array(want)?),
            "$contains" => Filter::Contains(
                key.to_string(),
                want.as_str()
                    .ok_or_else(|| Error::from_reason("$contains needs a string"))?
                    .to_string(),
            ),
            "$regex" => Filter::Regex(
                key.to_string(),
                regex_lite::Regex::new(
                    want.as_str()
                        .ok_or_else(|| Error::from_reason("$regex needs a string"))?,
                )
                .map_err(|e| Error::from_reason(format!("invalid $regex: {e}")))?,
            ),
            other => return Err(Error::from_reason(format!("unknown operator {other:?}"))),
        };
        result = Some(match result {
            None => filter,
            Some(existing) => Filter::And(vec![existing, filter]),
        });
    }
    result.ok_or_else(|| Error::from_reason(format!("empty operator object for key {key:?}")))
}

fn parse_value_array(v: &Json) -> Result<Vec<CvValue>> {
    let Json::Array(items) = v else {
        return Err(Error::from_reason("$in/$nin needs an array"));
    };
    items.iter().map(json_to_cv_value).collect()
}

/// One result from `query`, `get`, or `getWhere`.
#[napi(object)]
pub struct Record {
    pub id: Either<String, BigInt>,
    pub distance: Option<f64>,
    pub metadata: Json,
    pub document: Option<String>,
}

fn cv_record_to_js(r: cv::Record) -> Record {
    Record {
        id: id_to_js(&r.id),
        distance: r.distance.map(|d| d as f64),
        metadata: metadata_to_json(&r.metadata),
        document: r.document,
    }
}

/// Options for [`Collection::query`]. All fields optional; omitted means
/// "now, no filter, default overfetch" -- the equivalent of Python's
/// `col.query(v, k)` with no further keyword arguments.
#[napi(object)]
#[derive(Default)]
pub struct QueryJsOptions {
    pub snapshot: Option<BigInt>,
    pub r#where: Option<Json>,
    pub overfetch: Option<u32>,
}

/// Vectors with string-or-int ids, metadata, filtering, and snapshots --
/// same vocabulary as Python's `chronovec.Collection`.
#[napi]
pub struct Collection {
    inner: RustCollection,
}

#[napi]
impl Collection {
    /// A collection of the given dimensionality with default index options
    /// (cosine metric). For custom index options, build a raw `Index` first
    /// and wrap it -- not yet exposed through this binding; use the Rust
    /// crate directly if you need it today.
    #[napi(constructor)]
    pub fn new(dimensions: u32) -> Result<Self> {
        Ok(Collection {
            inner: RustCollection::new(dimensions as usize).map_err(to_js_err)?,
        })
    }

    #[napi]
    pub fn dimensions(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    #[napi]
    pub fn count(&self) -> u32 {
        self.inner.count() as u32
    }

    /// A token for the present, readable later via `query`'s `snapshot` option.
    #[napi]
    pub fn snapshot(&self) -> BigInt {
        BigInt::from(self.inner.snapshot().0 as i64)
    }

    /// Insert or replace one record.
    ///
    /// `metadata`/`document` are carried forward from the previous version
    /// when omitted (`undefined`) -- matching Python's `Collection.add()`,
    /// which carries forward when its `metadatas`/`documents` arguments are
    /// omitted entirely. Pass `{}` to explicitly clear metadata to empty
    /// (as opposed to omitting it, which keeps whatever it was). There is
    /// no way to explicitly clear an existing document back to no document
    /// while keeping the metadata -- neither can Python's `Collection.add()`.
    #[napi]
    pub fn add(
        &self,
        id: Either<String, BigInt>,
        vector: Float32Array,
        metadata: Option<Json>,
        document: Option<String>,
    ) -> Result<BigInt> {
        let id = either_to_id(id)?;
        let metadata = json_to_metadata_opt(metadata)?;
        let snap = self
            .inner
            .add(id, &vector, metadata, document)
            .map_err(to_js_err)?;
        Ok(BigInt::from(snap.0 as i64))
    }

    /// Change an existing record's metadata without touching its vector.
    /// Throws if `id` is not currently live.
    #[napi(js_name = "updateMetadata")]
    pub fn update_metadata(&self, id: Either<String, BigInt>, metadata: Json) -> Result<()> {
        let id = either_to_id(id)?;
        let metadata = json_to_metadata(&metadata)?;
        self.inner.update_metadata(id, metadata).map_err(to_js_err)
    }

    /// Remove one record. Returns whether it was live.
    #[napi]
    pub fn delete(&self, id: Either<String, BigInt>) -> Result<bool> {
        let id = either_to_id(id)?;
        self.inner.delete(id).map_err(to_js_err)
    }

    /// Remove every live record matching `where`. Returns their ids.
    #[napi(js_name = "deleteWhere")]
    pub fn delete_where(&self, filter: Json) -> Result<Vec<Either<String, BigInt>>> {
        let filter = parse_filter(&filter)?;
        let removed = self.inner.delete_where(&filter).map_err(to_js_err)?;
        Ok(removed.iter().map(id_to_js).collect())
    }

    /// Nearest records to `query`, optionally filtered and optionally as of
    /// a past snapshot.
    #[napi]
    pub fn query(
        &self,
        query: Float32Array,
        k: u32,
        options: Option<QueryJsOptions>,
    ) -> Result<Vec<Record>> {
        let options = options.unwrap_or_default();
        let snapshot = options
            .snapshot
            .map(|b| {
                let (v, lossless) = b.get_i64();
                if lossless {
                    Ok(chronovec::Snapshot(v as u64))
                } else {
                    Err(Error::from_reason(
                        "snapshot does not fit in a 64-bit signed integer",
                    ))
                }
            })
            .transpose()?;
        let filter = options.r#where.as_ref().map(parse_filter).transpose()?;
        let hits = self
            .inner
            .query(
                &query,
                k as usize,
                QueryOptions {
                    snapshot,
                    filter: filter.as_ref(),
                    overfetch: options.overfetch.unwrap_or(0) as usize,
                },
            )
            .map_err(to_js_err)?;
        Ok(hits.into_iter().map(cv_record_to_js).collect())
    }

    /// Fetch one live record by id, without a vector search.
    #[napi]
    pub fn get(&self, id: Either<String, BigInt>) -> Result<Option<Record>> {
        let id = either_to_id(id)?;
        Ok(self.inner.get(id).map(cv_record_to_js))
    }

    /// Fetch every live record matching `where`, without a vector search.
    #[napi(js_name = "getWhere")]
    pub fn get_where(&self, filter: Json) -> Result<Vec<Record>> {
        let filter = parse_filter(&filter)?;
        Ok(self
            .inner
            .get_where(&filter)
            .into_iter()
            .map(cv_record_to_js)
            .collect())
    }

    /// Reclaim versions no reader at or after `keepSnapshot` can still
    /// reach. Omitted, reclaims everything currently unreachable.
    #[napi]
    pub fn vacuum(&self, keep_snapshot: Option<BigInt>) -> Result<u32> {
        let keep = keep_snapshot
            .map(|b| {
                let (v, lossless) = b.get_i64();
                if lossless {
                    Ok(chronovec::Snapshot(v as u64))
                } else {
                    Err(Error::from_reason(
                        "keepSnapshot does not fit in a 64-bit signed integer",
                    ))
                }
            })
            .transpose()?;
        Ok(self.inner.vacuum(keep) as u32)
    }
}
