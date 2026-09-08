# Node.js bindings

ChronoVec provides a native Node.js addon via [napi-rs](https://napi.rs), wrapping the
existing Rust `chronovec` crate rather than a fourth independent FFI implementation over
the C ABI directly: the Rust safe wrapper does the unsafe work once; this is glue on top
of it.

Two tiers, matching Rust and Python:

- **Raw (`Index`)**: the same vocabulary as the C ABI, Rust's `chronovec::Index`,
  Python's `chronovec.Index`, and the Go binding
  (`insert`/`delete`/`search`/`vacuum`/`save`/`load`/`clock`, spelled camelCase per JS
  convention).
- **Ergonomic (`Collection`)**: the same vocabulary as Python's `chronovec.Collection`
  and Rust's `chronovec::collection::Collection` (`add`/`query`/`snapshot`/`vacuum`), over
  string-or-int ids, metadata, and a filter DSL. This was a cheap fast-follow specifically
  for this binding: `chronovec::collection::Collection` already existed in the Rust crate
  this wraps, so exposing it needed only new napi glue, not new logic, unlike Go, which
  has no Rust crate to reuse and would need the Collection logic built from scratch.

## Location

```
bindings/node/
├── Cargo.toml       # chronovec-node crate: napi-rs wrapping chronovec::Index/Collection
├── build.rs
├── package.json
├── src/
│   ├── lib.rs        # raw Index tier
│   └── collection.rs # ergonomic Collection tier
├── index.js          # committed: generated multi-platform loader
├── index.d.ts        # committed: generated TypeScript declarations
└── __test__/
    ├── index.test.js
    └── collection.test.js
```

## Building

The native `libchronovec` library must be built first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then, from `bindings/node`:

```bash
npm install
npm run build          # release build
npm run build:debug    # debug build, faster to iterate
npm test
```

## Usage

```ts
import { Index } from "@chronovec/native";

const idx = new Index(384, { metric: "cosine" });

const t1 = idx.insert(1n, embedding);       // ids are bigint, not number
idx.insert(2n, otherEmbedding);

for (const hit of idx.search(query, 10)) {
  console.log(hit.id, hit.distance);
}

idx.delete(1n);
idx.search(query, 10);           // id 1 is gone
idx.searchAsOf(query, 10, t1);   // id 1 is visible

idx.save("index.cvec");
const restored = Index.load("index.cvec");
```

```ts
import { Collection } from "@chronovec/native";

const col = new Collection(384);

col.add("doc-a", embedding, { lang: "en" }, "the user prefers dark mode");
const before = col.snapshot();
// Omitting metadata/document carries forward the previous version's value,
// matching Python's Collection.add() -- only the vector changes here.
col.add("doc-a", correctedEmbedding);
// An explicit, empty object replaces metadata with empty instead:
col.add("doc-a", correctedEmbedding, {});

col.query(query, 5);                              // now
col.query(query, 5, { snapshot: before });         // as of `before`

// where= is a plain object, matching Python's Collection.query(where=...) shape --
// not a Rust-style Filter enum, since that's not how a JS caller expects to write it.
col.query(query, 5, { where: { lang: "en" } });
col.query(query, 5, { where: { year: { $gte: 2024 } } });
col.query(query, 5, { where: { $and: [{ lang: "en" }, { year: { $gte: 2024 } }] } });

col.deleteWhere({ lang: "fr" });
```

### Why ids are `bigint`, not `number`

`stable_id()`, used by the Rust and Python `Collection` layers to turn a string id into
the engine's `int64` key, hashes into the full 64-bit range, routinely past
`Number.MAX_SAFE_INTEGER` (2^53). A JS `number` cannot represent that range losslessly.
napi-rs's plain `i64` type maps to a JS `number` and would silently misbehave on a real
hashed id, so every id and snapshot value here crosses the boundary as `bigint` instead:
verified with a round-trip test using an id past `MAX_SAFE_INTEGER`
(`__test__/index.test.js`). `Index.stats()` fields stay plain `number`: they're diagnostic
counts (pages, live vectors), not values you round-trip back into the API, and hitting
`MAX_SAFE_INTEGER` there means tens of petabytes of vectors.

## Running tests on macOS without `npm test`

macOS strips `DYLD_*` environment variables across npm's process-spawn chain (a
System Integrity Protection behavior), so `DYLD_LIBRARY_PATH=... npm test` may fail to
find `libchronovec.dylib` even though the same variable set on a direct `node` invocation
works. If `npm test` fails to load the native binding locally, run the test file directly:

```bash
DYLD_LIBRARY_PATH="$(pwd)/../../build" node --test __test__/index.test.js
```

This does not affect Linux (no equivalent stripping of `LD_LIBRARY_PATH`) or a properly
packaged release, where the platform-specific native binary ships without needing an
environment variable at all.

## ABI stability

`bindings/rust/chronovec-sys/native/include/chronovec.h` carries an explicit semantic version (`CV_ABI_VERSION_MAJOR`/
`MINOR`/`PATCH`): MAJOR for breaking changes, MINOR for additive ones, PATCH for none.
`tools/abi_gate.py` enforces this in CI. Every binding (Python, Rust, Go, Node) links
against this same header, though Node reaches it indirectly, through the Rust crate.
