# Changelog

All notable changes are documented here.

## Upcoming

### Testing & Quality
- `streambench --mode filtered` now takes `--label-partition`, so
  `label_partition=True` (one label to a page, previously covered only by a
  purity test) can be measured for recall instead of just correctness. At a
  fixed probe budget (nprobe 96, no ladder rematching, which was adding noise
  to an earlier pass at this measurement) the default filtered mode's recall
  erodes under churn at 10% selectivity (0.885 fresh, 0.803 after 3 rounds of
  25% churn on SIFT-128 at 200,000 live vectors) because insert placement is
  geometry only, so a tag that started concentrated in a few pages spreads
  into more of them as churn replaces records. `label_partition` loses about
  20x less recall to the same churn trace (1.000 fresh, 0.996 after churn),
  confirmed by an exact page count from `page_label_profile`: 1,233 of 2,053
  pages truly carried the tag under the default mode versus 206 of 2,053
  under `label_partition`. Documented in `docs/performance.md` under
  "Filtered search recall under churn," along with the tradeoff: the
  per-label page floor was not exercised by this measurement, since it used
  only two label values.

## 1.0.0 (2026-09-07)

First public release.

### Core engine
- Page-structured MVCC index with lock-free immutable query snapshots: every
  record carries a version interval, so a query reads a consistent
  point-in-time view with no locking, even while writes continue
  concurrently.
- Bounded reclamation (`vacuum`) physically removes expired versions
  incrementally, on a caller-specified budget, instead of a full rebuild.
- Crash recovery via write-ahead log replay, and an optional disk-backed,
  kernel-evictable vector arena for corpora larger than memory.
- Metadata filtering with page-level skipping: a page whose labels can't
  match the filter is skipped whole, so filtered search gets faster as the
  filter narrows.

### Python API
- `Collection`: string ids, metadata, a filter DSL (`$eq`/`$ne`/`$gt`/`$gte`/
  `$lt`/`$lte`/`$in`/`$nin`/`$contains`/`$regex`/`$and`/`$or`), snapshot
  reads, `save()`/`load()` checkpoints.
- `AsyncCollection`: an asyncio-native wrapper mirroring the full
  `Collection` surface.
- `Client`: named, auto-checkpointing collections on disk, plus a
  `chronovec` inspection CLI (`list`/`inspect`).
- `Index`: the raw engine (int64 ids, NumPy vectors) for maximum throughput.
- A provider-neutral embedding interface: pass a plain batch callable,
  `CustomEmbedding` (separate document/query embedders), any LangChain- or
  LlamaIndex-shaped embedder object directly, or the optional
  `SentenceTransformerEmbedding`/`ProviderEmbedding` (LiteLLM-backed,
  covering OpenAI/Cohere/Bedrock/Azure and others) adapters.

### Agent memory
- `AgentMemory`: branch, speculate, and merge or discard, with full snapshot
  isolation. Defaults to a private branch-delta engine: a branch's writes
  live in their own small native index and never touch main's pages,
  centroids, or routing until `merge()`, so there's no fixed
  concurrent-branch ceiling. The prior shared-label engine remains available
  (`branch_engine="labels"`, 63-branch ceiling) for compatibility.
- `branch_page_capacity` sizes a branch's native index independently of
  main's, for workloads with many small, short-lived branches (e.g. one
  branch per tree-search node).
- Checkpoint/resume: `save()`/`load()` persist main plus every still-open
  branch atomically, so an orchestrator can reattach to an in-flight branch
  after a restart (`AgentMemory.get_branch`).

### Integrations
- LangChain `VectorStore` subclass with branching support.
- LlamaIndex node storage adapter.
- LangGraph adapter (`LangGraphMemory`): maps one graph `thread_id` to one
  isolated ChronoVec branch.
- DuckDB adapter and a persistent SQLite virtual table (experimental).
- A worked Language Agent Tree Search (LATS) pattern
  (`examples/lats_chronovec.py`, `examples/lats_benchmark.py`) showing
  branch-delta memory used for tree search: not a packaged integration, a
  pattern to copy and adapt.

### Other language bindings
- Rust (`chronovec-sys` + `chronovec` on crates.io): a raw `Index` tier and
  an ergonomic `Collection` tier matching Python's vocabulary.
- Go (`bindings/go`): raw `Index` tier via cgo, plus an ergonomic
  `Collection` built from scratch.
- Node.js (`@chronovec/native` on npm): a native addon via napi-rs wrapping
  the Rust crate, with a raw tier and an ergonomic `Collection`.

### Engineering
CI covers Linux/macOS/Windows across Python 3.11-3.13, C++ sanitizers
(ASan/UBSan/TSan), an ABI-compatibility gate over the C header, and an
equal-recall performance regression gate. See
[Performance](docs/performance.md) for benchmark methodology and results
against chromadb, faiss, hnswlib, and other engines.
