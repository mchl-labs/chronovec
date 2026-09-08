# Support and maturity matrix

ChronoVec intentionally exposes a wide ecosystem surface, but the surfaces do
not all have identical guarantees. This matrix is the release policy: it keeps
the API broad while making the supported path clear.

| Surface | Level | What is covered |
|---|---|---|
| Python native `Index` | Stable | Native SIMD core, snapshots, batches, persistence, reclamation, and the main regression suite. |
| Python `Client` / `Collection` / `AsyncCollection` | Stable | Named local persistence, metadata, filters, history, RAG workflows, and framework-facing usage. `AsyncCollection` is a thread-offloading wrapper. |
| C ABI | Stable | Versioned boundary used by the bindings, ABI gate, native tests, and sanitizer coverage. |
| Rust | Supported alpha | Raw and ergonomic collections, bundled native build, unit tests, doctests, and CI. Published to crates.io as `chronovec-sys` and `chronovec`. |
| Node.js | Supported alpha | N-API addon building platform binaries for Linux x64/arm64, macOS arm64, and Windows x64. Published to npm as `@chronovec/native`. |
| Go | Supported alpha | Raw cgo binding and ergonomic collection. Requires a system C++ library or an explicitly configured library path. |
| SQLite | Integration | Loadable extension and persistence smoke tests. |
| LangChain / LlamaIndex | Integration | Adapter tests against the actual framework interfaces. |
| DuckDB | Experimental | Useful adapter, but currently Python/GIL-bound and not a thread-safety or performance commitment. |

## Compatibility policy

The Python API and C ABI are the compatibility anchors. Changes to the C ABI
must pass `tools/abi_gate.py` and bump the ABI major version for breaking
changes. Binding changes must preserve the shared raw vocabulary and add a
cross-language test when behavior is intended to match Python.

Performance claims are release claims only when the benchmark JSON, exact
dataset identity, configuration, environment, and confidence intervals are
published together. A static build-once query benchmark must not be used to
claim superiority on mutable workloads, and vice versa.
