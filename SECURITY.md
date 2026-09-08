# Security policy

## Supported versions

ChronoVec's Python API and C ABI are production-supported. Rust, Go, and
Node.js bindings are supported alpha surfaces; DuckDB remains experimental.

## Reporting a vulnerability

Do not report security vulnerabilities in public GitHub issues.

To report a vulnerability privately, email the address in [`MAINTAINERS.md`](MAINTAINERS.md). We aim to acknowledge reports within 72 hours and provide a fix timeline within 7 days.

## Scope

- **In scope:** Memory safety bugs in the C++ core, unsafe deserialization in the checkpoint/WAL format, data leakage between MVCC snapshots, injection via the SQLite virtual table interface.
- **Out of scope:** Denial-of-service via malformed vectors (the library is intended for trusted callers), issues in third-party dependencies.

## Notes

ChronoVec processes arbitrary float arrays from callers. Do not expose the index directly to untrusted input without validation at your application boundary. The WAL and checkpoint files are not encrypted; treat them as sensitive if they contain sensitive embeddings.

`Collection.load()` and `AgentMemory.load()` also restore Python pickle
sidecars. Load checkpoint files only from trusted producers; checksum
validation detects corruption or mismatched generations but does not make
pickle deserialization safe for untrusted input.
