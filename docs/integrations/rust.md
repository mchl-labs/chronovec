# Rust bindings

ChronoVec provides Rust bindings via `chronovec-sys` (raw C ABI) and a safe wrapper crate.

## Location

```
bindings/rust/
├── chronovec-sys/     # raw unsafe FFI bindings over the C ABI
└── chronovec/         # safe Rust wrapper
```

## Building

Published to crates.io: `cargo add chronovec` builds the bundled native
core automatically.

To build from this repository instead (for contributing to the bindings
themselves), build the native library first,

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then:

```bash
cd bindings/rust
cargo build
cargo clippy
cargo fmt --check
```

## Usage: raw tier (`Index`)

The same vocabulary as the C ABI and Python's `Index`: `insert`/`delete`/`search`/`vacuum`/
`save`/`load`/`clock`, over `i64` ids and bare `&[f32]` vectors.

```rust
use chronovec::{Index, Metric};

let index = Index::builder(384).metric(Metric::Cosine).nprobe(16).build()?;
index.insert(1, &embedding)?;
index.insert(2, &other_embedding)?;

let results = index.search(&query, 10)?;
for hit in results {
    println!("{} {:.4}", hit.id, hit.distance);
}

index.save("my_index.cvec")?;
let restored = Index::load("my_index.cvec")?;
```

## Usage: ergonomic tier (`collection::Collection`)

The same vocabulary as Python's `Collection` (`add`/`query`/`snapshot`/`vacuum`), over
string-or-int ids, metadata, documents, and a filter DSL with the same operator set
(`Eq`/`Ne`/`Gt`/`Gte`/`Lt`/`Lte`/`In`/`NotIn`/`Contains`/`Regex`/`And`/`Or`).

```rust
use chronovec::collection::{Collection, Filter, QueryOptions, Value};
use std::collections::HashMap;

let col = Collection::new(384)?;

let mut meta = HashMap::new();
meta.insert("lang".to_string(), Value::Str("en".into()));
col.add("doc-a", &embedding, Some(meta), Some("hello".into()))?;

let before = col.snapshot();
// None carries forward the previous version's metadata/document, matching
// Python's Collection.add() -- only the vector changes here.
col.add("doc-a", &corrected_embedding, None, None)?;
// Some(HashMap::new()) replaces metadata with empty instead:
col.add("doc-a", &corrected_embedding, Some(HashMap::new()), None)?;

// Now
let now = col.query(&query, 5, Default::default())?;
// As of `before`
let then = col.query(&query, 5, QueryOptions { snapshot: Some(before), ..Default::default() })?;

// Filtered
let filter = Filter::Eq("lang".into(), Value::Str("en".into()));
let filtered = col.query(&query, 5, QueryOptions { filter: Some(&filter), ..Default::default() })?;
```

`Collection::new` uses `Index`'s defaults; for custom index options build an `Index` with
`Index::builder(...)` and pass it to `Collection::wrap`.

## Time travel example

```bash
cargo run --example time_travel
```

## C ABI direct access

If you need features not yet exposed in the safe wrapper, bind directly via `chronovec-sys`:

```rust
use chronovec_sys::*;

unsafe {
    let idx = cv_create(384, CV_COSINE as i32, 256, 16);
    // ...
    cv_destroy(idx);
}
```

See the [C ABI header](https://github.com/mchl-labs/chronovec/blob/main/bindings/rust/chronovec-sys/native/include/chronovec.h) for the full C ABI.

## ABI stability

`bindings/rust/chronovec-sys/native/include/chronovec.h` carries an explicit semantic version (`CV_ABI_VERSION_MAJOR`/
`MINOR`/`PATCH`, currently 0.1.0): MAJOR for breaking changes, MINOR for additive ones, PATCH
for none. `tools/abi_gate.py` enforces this in CI: a breaking change to the header that
doesn't bump MAJOR fails the build. Every binding (Python, Rust, Go, Node) links against this
same header, so a version bump here is the signal to check for a breaking change across all
of them.
