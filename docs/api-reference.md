# API reference

## `Index` (alias for `NativeChronoVecIndex`)

The primary index class. Backed by the C++20 native library.

```python
from chronovec import Index
```

### Constructor

```python
Index(
    dimensions: int,
    *,
    metric: str = "cosine",           # "cosine" or "l2"
    page_capacity: int = 256,          # vectors per page
    nprobe: int = 16,                  # pages visited per query
    screening: bool = True,            # 4-bit screening codes
    adaptive_bounds: bool = False,     # exact-safe page pruning (opt-in)
    labels: bool = False,              # enable per-record label filtering
    arena_path: str | None = None,     # disk-backed float payload
    max_vectors: int | None = None,    # required when arena_path is set
    wal_path: str | None = None,       # write-ahead log for crash recovery
    wal_sync: bool = True,             # fsync after each WAL record
    threads: int = 1,                  # query thread pool size
    rerank_factor: int = 4,            # exact-rerank candidate multiplier
)
```

Also opens an existing WAL when `wal_path` points to an existing log file: recovery is automatic.

### Methods

#### `insert(item_id, vector, timestamp=None) → int`

Insert a single vector. Returns the commit timestamp.

- `item_id`: `int`, must fit in `int64`.
- `vector`: `np.ndarray[float32]`, shape `(dimensions,)`.

#### `insert_many(ids, vectors, labels=None) → int`

Batch insert. One commit timestamp for the whole batch: atomic to readers.

- `ids`: array-like of `int64`.
- `vectors`: `np.ndarray[float32]`, shape `(n, dimensions)`.
- `labels`: optional `np.ndarray[uint64]`, shape `(n,)`.

#### `delete(item_id, timestamp=None) → int`

Mark a record expired. Returns the commit timestamp.

#### `delete_many(ids) → int`

Delete multiple records. Each deletion receives its own commit timestamp; use
separate snapshots when a caller needs a stable read across the operation.

#### `search(vector, k=10, snapshot=None, nprobe=None, predicate=None, adaptive=True) → list[SearchResult]`

Approximate nearest-neighbour search.

- `snapshot`: `int | None`, query as of this timestamp. `None` means latest committed.
- `nprobe`: overrides the index default for this query.
- `predicate`: label bitmask for filtered search (requires `labels=True` on construction).

Returns a list of `SearchResult(id: int, distance: float)`.

#### `search_batch(vectors, k=10, snapshot=None) → list[list[SearchResult]]`

Query multiple vectors in one call, sharing the snapshot.

#### `vacuum(oldest_snapshot, budget_pages=None, budget_versions=None) → int`

Reclaim expired versions older than `oldest_snapshot`. Returns the number of versions reclaimed.

- `budget_versions`: maximum versions to reclaim in this call.
- `budget_pages`: maximum pages to inspect.

#### `snapshot() → int`

Return the current committed timestamp. Use as the `snapshot=` argument to read the index as of now.

#### `clock → int` (property)

The committed timestamp. Identical to `snapshot()`.

#### `save(path: str) → None`

Write a checksummed checkpoint. Atomic on POSIX (write to temp, rename).

#### `Index.load(path: str) → Index` (classmethod)

Restore from a checkpoint. Validates checksum.

#### `flush_vectors() → None`

Make disk-backed pages clean and evictable by the kernel.

#### `stats() → dict`

Index statistics: page count, live vectors, capacity amplification, bytes per live vector, etc.

#### `close() → None`

Release native resources. Called automatically on `__del__`.

---

## `Client` / `PersistentClient`

The recommended local entry point. A client directory contains named
collections, their checksummed checkpoints, and a small JSON registry.

```python
from chronovec import Client

client = Client("./data")
collection = client.get_or_create_collection("docs", dimensions=384)
same_collection = client.get_collection("docs")
names = client.list_collections()
```

Successful `add`, `update`, `delete`, and `vacuum` calls on collections opened
by a client are checkpointed automatically. `PersistentClient` is an alias for
`Client`. Use `chronovec list ./data` and `chronovec inspect ./data docs --json`
to inspect a local store without writing application code.

## `Collection`

High-level API with string IDs, metadata, and rich filtering. Backed by `Index`.

```python
from chronovec import Collection
```

### Constructor

```python
Collection(
    dimensions: int,
    *,
    metric: str = "cosine",                 # "cosine" or "l2"
    name: str = "chronovec",
    embedding_function=None,                # batch callable or EmbeddingFunction
    wal_path: str | None = None,            # path for crash-recovery WAL
    arena_path: str | None = None,          # path for disk-backed float payload
    max_vectors: int = 0,                   # required when arena_path is set
    growth_factor: float = 2.0,             # multiplier on auto-grow
    **index_options,                        # forwarded to Index (nprobe, page_capacity, …)
)
```

When both `wal_path` and `arena_path` are set, the collection auto-grows the arena when
it fills: closes the index, removes the arena file, and reopens with `max_vectors * growth_factor`,
letting the WAL replay refill the larger arena automatically.

### Methods

#### `add(ids, *, embeddings=None, documents=None, metadatas=None) → list[str | int]`

Insert or replace records. `ids` may be `str` or `int` (or a mix). Returns the id list.

Passing `documents` requires `embedding_function` on construction. A plain
callable receives `Sequence[str]` and returns one vector per string. For
separate document and query behavior, pass `CustomEmbedding` or an object with
`embed_documents(texts)` and `embed_query(text)` methods:

```python
from chronovec import Collection, CustomEmbedding

embedding = CustomEmbedding(
    embed_documents=my_embed_documents,
    embed_query=my_embed_query,
)
collection = Collection(dimensions=384, embedding_function=embedding)
```

LlamaIndex-style `get_text_embedding_batch` and `get_query_embedding` methods
are also recognized. ChronoVec leaves provider selection and model lifecycle
to the application.

For local Sentence Transformers models, the optional
`SentenceTransformerEmbedding(model_or_name, **encode_kwargs)` adapter is
available after `pip install "chronovec[sentence-transformers]"`. It uses
`encode_document`/`encode_query` when the model provides them and otherwise
falls back to `encode`.

For hosted providers, `ProviderEmbedding(provider, model, api_key, **kwargs)`
wraps [LiteLLM](https://docs.litellm.ai/) so one ChronoVec adapter covers every
provider LiteLLM supports (OpenAI, Cohere, Bedrock, Azure, ...) without
ChronoVec depending on any of them directly, or needing an update every time a
new provider shows up. Install it with `pip install "chronovec[litellm]"` and
pass the secret explicitly, usually as `api_key=os.environ["PROVIDER_API_KEY"]`.
Optional `document_kwargs` and `query_kwargs` are passed only to their
respective embedding calls. `CustomEmbedding` remains the zero-dependency
option for anything LiteLLM doesn't cover.

`upsert` is an alias; behaviour is identical.

#### `update(ids, *, embeddings=None, documents=None, metadatas=None) → None`

Update existing records. Raises `KeyError` for unknown ids. Accepts the same kwargs as `add`.

#### `delete(ids=None, *, where=None) → list[str | int]`

Remove by id, by metadata filter, or both. Returns the ids of what was removed.
Raises `ValueError` if both are `None` (refusing to delete everything).

#### `query(query=None, k=10, *, query_text=None, where=None, snapshot=None, overfetch=8) → list[Record]`

Nearest records, optionally filtered and optionally as of a past snapshot.

- `query`: a pre-computed `np.ndarray[float32]` vector. **Never pass a plain string here.**
- `query_text`: text string, embedded automatically when `embedding_function` is set. Passing a bare string as the first positional arg when no `embedding_function` is set raises a helpful `ValueError`.
- `where`: metadata filter (see below).
- `snapshot`: query as of this timestamp token. `None` = latest.
- `overfetch`: widens the candidate set before filtering; raise it when few results pass the filter.

Returns `list[Record]` where each `Record` has `.id`, `.distance`, `.metadata`, `.document`.

#### `get(ids=None, *, where=None, limit=None) → list[Record]`

Fetch records by id or metadata without a vector search.

#### `peek(n=5) → list[Record]`

Return up to `n` live records in insertion order. Useful for quick inspection.

#### `count() → int`

Number of live records.

#### `snapshot() → int`

Capture the current clock value. Pass it as `snapshot=` to any future `query()` call to
read the index as it was at this moment.

#### `vacuum(*, keep_snapshot=None) → int`

Reclaim expired versions. Returns the number of versions freed.

- `keep_snapshot`: oldest snapshot any caller still holds. Versions older than this are dropped.
  `None` means drop everything currently unreachable.

#### `save(path: str | Path) → None`

Durably checkpoint the collection to disk:

- `<path>.cvec`: native index checkpoint (atomic write-then-rename on POSIX).
- `<path>.meta`: pickle sidecar with id/metadata/document state.

Restore with `Collection.load()`. The restored collection is heap-resident; pass
`wal_path`/`arena_path` after loading to re-enable those features.

#### `Collection.load(path, *, embedding_function=None) → Collection` (classmethod)

Restore from a checkpoint created with `save()`. Pass `embedding_function` again if
text queries are needed: it is not serialised.

#### `flush_vectors() → int`

Mark disk-backed vector pages clean so the kernel may evict them. Returns bytes made
evictable, or `0` when not disk-backed. **BETA.**

#### `stats() → dict`

Index statistics forwarded from the native layer.

#### `close() → None`

Release native resources. Also available as a context manager (`with Collection(...) as col:`).

---

## `AsyncCollection`

`asyncio`-native wrapper around `Collection`. Every method that touches the index runs
via `asyncio.to_thread` so it never blocks the event loop.

```python
from chronovec import AsyncCollection
```

### Constructor

Accepts the same parameters as `Collection`.

```python
AsyncCollection(
    dimensions: int,
    *,
    metric: str = "cosine",
    name: str = "chronovec",
    embedding_function=None,
    wal_path: str | None = None,
    arena_path: str | None = None,
    max_vectors: int = 0,
    growth_factor: float = 2.0,
    **index_options,
)
```

### Async methods

`add`, `upsert`, `update`, `delete`, `query`, `get`, `peek`, `vacuum`, `save`, `close`:
all have the same signatures as `Collection` but are `async`.

### Sync methods

`snapshot()`, `count()`, `flush_vectors()`: synchronous (read an atomic int or make a
syscall; no need for the thread pool).

### Class method

#### `AsyncCollection.load(path, *, embedding_function=None) → AsyncCollection`

Synchronous: WAL replay happens inside the native load. Returns an `AsyncCollection`
wrapping the restored `Collection`.

### Context manager

```python
async with AsyncCollection(384, embedding_function=embed) as col:
    await col.add(ids=["a"], documents=["hello"])
    results = await col.query(query_text="hello", k=3)
```

### `.sync` property

Exposes the underlying `Collection` for callers that need the synchronous interface.

### Filter expressions (`where`)

```python
# Simple equality
where={"lang": "en"}                             # $eq implied
where={"lang": {"$eq": "en"}}

# Comparison
where={"year": {"$gt": 2023}}
where={"score": {"$gte": 0.8, "$lt": 1.0}}      # not yet: combine in $and

# Set membership
where={"lang": {"$in": ["en", "fr"]}}
where={"lang": {"$nin": ["de"]}}

# String matching
where={"title": {"$contains": "MVCC"}}
where={"title": {"$regex": "^vec.*"}}

# Boolean composition
where={"$and": [{"lang": "en"}, {"year": {"$gte": 2024}}]}
where={"$or":  [{"lang": "en"}, {"lang": "fr"}]}
```

---

## `AgentMemory`

Snapshot-isolated memory store with speculative branching and historical forks.

```python
from chronovec import AgentMemory
```

### Constructor

```python
AgentMemory(
    dimensions: int,
    *,
    metric: str = "cosine",
    page_capacity: int = 256,
    branch_page_capacity: int | None = None,
    nprobe: int = 16,
    checkpoint_path: str | None = None,
    branch_engine: str = "delta",
    branch_merge_strategy: str | None = None,
)
```

`branch_engine="delta"` (the default) gives every branch a private native
delta index: branch writes never touch main pages, centroids, or routing, and
are merged into main only on `merge()`. `branch_engine="labels"` is the
previous implementation, kept for compatibility: it partitions one shared
index with an attribute label per branch and is limited to 63 concurrent
branches (one label bit each, bit 0 reserved for main). The delta engine has
no corresponding branch-count limit.

`branch_merge_strategy` controls what `merge()` does when a branch and main
both changed the same id after the branch's fork point. It defaults to
`"last_writer_wins"` (matching the label engine's behavior: the branch's
value wins silently). Pass `"fail_on_conflict"` to raise
`BranchConflictError` instead of overwriting a concurrent main change; only
valid with `branch_engine="delta"`.

`branch_page_capacity` sizes every branch's private delta index independently
of main's `page_capacity`; only valid with `branch_engine="delta"`. Each
native page preallocates its full capacity's worth of storage up front (see
[Architecture](architecture)), so `page_capacity` is a real tradeoff for
main (too small and a large corpus needs enough pages to hurt routing/query
cost), but a branch that will only ever hold a handful of records pays that
same preallocation for no benefit. Set `branch_page_capacity` lower (it
defaults to `None`, meaning branches use `page_capacity` too, unchanged from
before this parameter existed) for a workload with many short-lived,
small branches (a tree search over branch-per-node trajectories, for
example) without affecting main's own page sizing at all. If a branch you
sized this way ends up growing large, it pays the small-page-count-for-a-
given-size routing cost that motivated `page_capacity` being large for main
in the first place: this parameter is about right-sizing to a branch's
*actual* usage, not a free lower bound.

### Methods

#### `add(item_id, vector, **payload) → int`

Insert a record with arbitrary keyword metadata.

#### `delete(item_id) → int`

Mark a record expired.

#### `search(vector, k=10) → list[tuple[SearchResult, Record]]`

Search within the current view (main or branch).

#### `snapshot() → int`

Return the current committed point in the memory timeline. Pass it to
`branch(..., snapshot=...)` to start a non-destructive retry or exploration
from an earlier state.

This is a read boundary, not an in-place rollback. Passing the value to
`search(..., as_of=...)` reads historical state; it does not undo later writes.
To continue writing from that state, create a branch from the snapshot.

#### `branch(name: str, *, snapshot: int | None = None) → _View`

Create a speculative branch from the current committed state, or from the
specified historical snapshot. The branch sees the chosen main-memory state
plus its own writes and local deletes; later main writes are not visible to it.
Returns a `_View` that supports `add`, `delete`, `search`, `merge`, `discard`.

Up to 63 concurrent branches with `branch_engine="labels"`; no such limit
with the default `branch_engine="delta"`.

#### `get_branch(name: str) → _View`

Return an active branch by name. This is intended for workflow recovery after
`AgentMemory.load(...)`, when an external orchestrator needs to reattach to a
branch that was open when the memory checkpoint was written. It raises
`BranchError` if the branch does not exist or has already been merged/discarded.

#### `save(path) → None`

Write a generation of native vectors and agent payload/branch state, then
atomically publish a manifest to select that generation's files. An
incomplete or mismatched generation is rejected on load.

With the default `branch_engine="delta"`, the manifest is
`<path>.branchable.manifest`, selecting a `.main.cvec` file plus one
`.branch-N.cvec` delta file per active branch. With `branch_engine="labels"`,
it is `<path>.manifest`, selecting a single `.cvec` and `.meta` file; there
is no separate per-branch delta, since branch writes share the main index.

Pass `checkpoint_path=` to the constructor to repeat this checkpoint after
each successful memory or branch mutation.

The checkpoint is a durable restart point, not a rollback transaction. It
contains AgentMemory's payload and branch metadata in addition to the native
vectors. The core `Index` and `Collection` APIs separately support optional
WAL-based crash recovery; AgentMemory's checkpoint manifest does not turn
arbitrary Python-side operations into a WAL transaction.

Checkpoint sidecars are Python pickle files. Load only checkpoints produced by
your application or another trusted source; the manifest checksum detects
mix-ups and corruption but is not a sandbox for untrusted deserialization.

#### `AgentMemory.load(path, *, checkpoint_path=None) → AgentMemory`

Restore a checkpoint, including active branches. The embedding/vector state is
restored; no external embedding function is involved at this low-level API.

#### `purge(oldest_snapshot) → int`

Garbage-collect versions older than `oldest_snapshot`. Active branch fork
points are protected automatically; purge raises `ValueError` if the requested
horizon would invalidate one, so discard or merge those branches first.

### Branch lifecycle

```python
branch = memory.branch("plan-a")
branch.add(...)
branch.search(...)   # sees main + branch writes

branch.merge()       # atomically visible promotion to main
# or
branch.discard()     # abandons branch writes; purge later reclaims versions
```

Branch discard is the non-destructive way to abandon speculation. Merge holds
the AgentMemory state gate so high-level readers see the promotion as one
before-or-after transition. It is not an in-place rollback transaction; failed
operations do not rewind already accepted native mutations, so snapshot reads
and checkpoint loads remain the mechanisms for historical inspection and
restart.

With the default `branch_engine="delta"`, `merge()` checks every id the
branch touched against main's current version at merge time. Under the
default `branch_merge_strategy="last_writer_wins"`, a conflicting id is
silently overwritten with the branch's value. Construct with
`branch_merge_strategy="fail_on_conflict"` if a branch and main might touch
the same id concurrently and a silent overwrite would be a data-loss bug in
your application; that raises `BranchConflictError` (with the conflicting
ids) instead, and leaves the branch open so you can retry after inspecting
`branch.conflicts()`.

---

## `Record`

Returned by `Collection.query()`, `Collection.get()`, and `Collection.peek()`.

```python
@dataclass
class Record:
    id: str | int          # original string or integer id
    distance: float | None # distance from query vector (None for get/peek results)
    metadata: dict         # metadata dict as passed to add()
    document: str | None   # document text as passed to add()
```

Supports attribute access (`r.id`) and dict-style key access (`r["id"]`).

---

## `SearchResult`

Returned by the raw `Index.search()` and `Index.search_batch()`.

```python
@dataclass
class SearchResult:
    id: int
    distance: float
```

---

## `estimate_memory`

```python
from chronovec import estimate_memory

info = estimate_memory(
    dimensions=768,
    n_vectors=5_000_000,
    ram_budget_bytes=32 * 10**9,
)
# info.heap_resident_bytes, info.disk_backed_resident_bytes,
# info.reduction_factor, info.recommend_disk
```

---

## `exact_search`

```python
from chronovec import exact_search

results = exact_search(query, corpus, k=10, metric="cosine")
```

SIMD-accelerated brute-force search. Useful as a ground-truth baseline.

---

## `stable_id`

```python
from chronovec import stable_id

int_id = stable_id("my-string-id")  # deterministic int64 hash
```

Used internally by `Collection` to map string IDs to the native `int64` key space.
