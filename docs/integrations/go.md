# Go bindings

ChronoVec provides a Go binding via cgo, directly over the C ABI: no separate raw/safe
split like Rust's `chronovec-sys`/`chronovec`, since Go's `error` return values already
give the safety a Rust `unsafe` boundary buys.

Two tiers, matching Rust, Python, and Node:

- **Raw (`Index`)**: the same vocabulary as the C ABI, Rust's `chronovec::Index`, and
  Python's `chronovec.Index` (`Insert`/`Delete`/`Search`/`Vacuum`/`Save`/`Load`/`Clock`),
  over `int64` ids and `[]float32` vectors.
- **Ergonomic (`Collection`)**: the same vocabulary as Python's `chronovec.Collection` and
  Rust's `chronovec::collection::Collection` (`Add`/`Query`/`Snapshot`/`Vacuum`), over
  `any`-typed string-or-int ids, `map[string]any` metadata, and a filter DSL built from
  plain Go maps matching Python's `where=` dict shape. Unlike Node (which wraps the
  existing Rust `Collection`), this is a from-scratch implementation: no Rust crate for Go
  to reuse via cgo, so `StableID` and the filter DSL are native Go, verified against the
  same six Python-generated hash fixtures used in the Rust binding's test.

## Location

```
bindings/go/
├── go.mod
└── chronovec/
    ├── chronovec.go         # raw Index tier
    ├── chronovec_test.go
    ├── collection.go        # ergonomic Collection tier
    └── collection_test.go
```

## Building

Requires Go 1.25+: `golang.org/x/crypto` (used for `StableID`'s BLAKE2b hashing) declares
that as its own minimum as of the version this module pins. The raw `Index` tier alone
would work on the older `go 1.21` this module originally targeted; the floor moved with
the `Collection` tier's dependency, not for any Index-tier reason.

The native library must be built first:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then, from `bindings/go`:

```bash
go vet ./...
go test ./...
gofmt -l .    # should print nothing
```

By default the package links against `../../../build` relative to its own source
directory. Override with the standard `CGO_CFLAGS`/`CGO_LDFLAGS` environment variables if
the native library lives elsewhere:

```bash
CGO_CFLAGS="-I/path/to/bindings/rust/chronovec-sys/native/include" CGO_LDFLAGS="-L/path/to/build -lchronovec" go test ./...
```

## Usage

```go
import "github.com/mchl-labs/chronovec/bindings/go/chronovec"

idx, err := chronovec.New(384, chronovec.DefaultOptions())
if err != nil {
    log.Fatal(err)
}
defer idx.Close()

t1, _ := idx.Insert(1, embedding)
idx.Insert(2, otherEmbedding)

hits, _ := idx.Search(query, 10)
for _, hit := range hits {
    fmt.Println(hit.ID, hit.Distance)
}

idx.Delete(1)
idx.Search(query, 10)               // id 1 is gone
idx.SearchAsOf(query, 10, t1)       // id 1 is visible

idx.Save("index.cvec")
restored, _ := chronovec.Load("index.cvec")
```

```go
col, err := chronovec.NewCollection(384)
if err != nil {
    log.Fatal(err)
}
defer col.Close()

originalDoc := "original doc"
col.Add("doc-a", embedding, map[string]any{"lang": "en"}, &originalDoc)
before := col.Snapshot()
// nil metadata/document carries forward the previous version's value,
// matching Python's Collection.add() -- only the vector changes here.
col.Add("doc-a", correctedEmbedding, nil, nil)
// An explicit, non-nil empty map replaces metadata with empty instead:
col.Add("doc-a", correctedEmbedding, map[string]any{}, nil)

col.Query(query, 5, chronovec.QueryOptions{})                          // now
col.Query(query, 5, chronovec.QueryOptions{Snapshot: &before})         // as of `before`

// Where is a plain map, matching Python's Collection.query(where=...) shape.
col.Query(query, 5, chronovec.QueryOptions{
    Where: map[string]any{"lang": "en"},
})
col.Query(query, 5, chronovec.QueryOptions{
    Where: map[string]any{"year": map[string]any{"$gte": 2024}},
})

col.DeleteWhere(map[string]any{"lang": "fr"})
```

## Threading

Searches are lock-free over immutable snapshots; mutations are thread-safe but serialise
internally. An `*Index` may be shared across goroutines without an external lock; this is
lock-free reads, not parallel multiwriter progress, matching the C ABI's own contract.

## Resource lifetime

`Close()` releases the native handle and should be called explicitly (typically via
`defer`). A `runtime.SetFinalizer` backstop exists so a forgotten `Close` doesn't leak past
Go's GC, but it runs at an unpredictable time, so don't rely on it in a program that creates
many indices.

## ABI stability

`bindings/rust/chronovec-sys/native/include/chronovec.h` carries an explicit semantic version (`CV_ABI_VERSION_MAJOR`/
`MINOR`/`PATCH`): MAJOR for breaking changes, MINOR for additive ones, PATCH for none.
`tools/abi_gate.py` enforces this in CI. Every binding (Python, Rust, Go, Node) links
against this same header.
