# Migrating from Chroma

ChronoVec's high-level `Collection` API is intentionally familiar to Chroma
users: records have ids, documents, embeddings, metadata, and filtered vector
queries. The main difference is that ChronoVec treats versions, snapshots, and
deletion as first-class behavior for data that changes.

## Concept mapping

| Chroma | ChronoVec |
|---|---|
| `PersistentClient(path)` | `Client(path)` |
| `get_or_create_collection(name)` | `get_or_create_collection(name, dimensions=...)` |
| `collection.add(...)` | `collection.add(...)` |
| `collection.upsert(...)` | `collection.add(...)` or `collection.upsert(...)` |
| `collection.query(..., n_results=k)` | `collection.query(..., k=k)` |
| `where=...` metadata filters | `where=...` metadata filters |
| `collection.delete(...)` | `collection.delete(ids=..., where=...)` |
| Rebuild or replace a collection | `snapshot()` and historical queries |

ChronoVec requires the vector dimension when creating a collection. If an
`embedding_function` is supplied, text queries and document-only writes can
use it; otherwise pass embeddings explicitly.

## Before: Chroma

```python
import chromadb

client = chromadb.PersistentClient(path="./data")
collection = client.get_or_create_collection("docs")
collection.add(
    ids=["doc1", "doc2"],
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],
)
results = collection.query(query_texts=["find document1"], n_results=2)
```

## After: ChronoVec

```python
from chronovec import Client

client = Client("./data")
collection = client.get_or_create_collection("docs", dimensions=384)
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[embedding1, embedding2],
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],
)
results = collection.query(query_vector, k=2, where={"source": "notion"})
```

For text-in/text-out usage, provide an embedder once:

```python
from chronovec import Collection

collection = Collection(dimensions=384, embedding_function=my_embed)
collection.add(ids=["doc1"], documents=["This is document1"])
results = collection.query(query_text="find document1", k=2)
```

## What ChronoVec adds

### Query an earlier state

Capture a snapshot before an update and use it for reproducible retrieval:

```python
before = collection.snapshot()
collection.add(
    ids=["doc1"],
    embeddings=[corrected_embedding],
    documents=["The corrected document"],
)

latest = collection.query(query_vector, k=5)
historical = collection.query(query_vector, k=5, snapshot=before)
```

### Rewind and retry agent memory

For agent workflows, start a branch from an earlier snapshot. Branch updates
and deletes stay isolated until the branch is merged:

```python
from chronovec import AgentMemory

memory = AgentMemory(384)
memory.add("fact", fact_embedding, text="confirmed fact")
checkpoint = memory.snapshot()

memory.add("plan", bad_plan_embedding, text="unverified plan")
retry = memory.branch("retry", snapshot=checkpoint)
retry.add("plan", good_plan_embedding, text="corrected plan")
retry.merge()
```

### Delete without an immediate rebuild

Logical deletion takes effect immediately. Call `vacuum()` separately to
reclaim expired versions with an explicit budget:

```python
collection.delete(ids=["doc1"])
collection.vacuum(keep_snapshot=collection.snapshot())
```

## Migration notes

- Chroma's collection dimension is inferred from the first insert; ChronoVec's
  dimension is declared at construction so invalid vectors fail earlier.
- ChronoVec's `query()` returns a list of `Record` objects, not Chroma's nested
  result dictionaries.
- `Client` checkpoints successful collection mutations locally. Use
  `Collection.save()` and `Collection.load()` for explicit checkpoints.
- ChronoVec is an embedded library today. It does not provide Chroma Cloud's
  hosted service, multi-tenant HTTP API, or managed operational plane.
- Start with the Python `Client`/`Collection` path. Other language bindings
  are supported alpha surfaces with separate compatibility guarantees.
