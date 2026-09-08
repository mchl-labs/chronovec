# Streaming and high-churn collections

ChronoVec is designed for collections that change continuously. This page covers patterns for CDC (change-data-capture) streams, high-frequency agent writes, and large-scale dataset turnovers.

## Running the example

```bash
python examples/streaming_updates.py
```

## Core pattern: insert, delete, vacuum loop

```python
from chronovec import Index
import numpy as np

dim = 128
index = Index(dim, metric="cosine", page_capacity=256, nprobe=16)
live = {}  # id → vector, tracks the current live set

def ingest(uid, vector):
    if uid in live:
        index.delete(uid)
    index.insert(uid, vector)
    live[uid] = vector

def evict(uid):
    if uid in live:
        index.delete(uid)
        del live[uid]

# Continuously mutate and periodically vacuum
for epoch in range(5):
    for i in range(10_000):
        uid = np.random.randint(0, 5000)
        if np.random.rand() < 0.3:
            evict(uid)
        else:
            ingest(uid, np.random.rand(dim).astype("float32"))

    # Vacuum on a schedule: O(budget_versions), not O(corpus)
    reclaimed = index.vacuum(
        oldest_snapshot=index.clock + 1,
        budget_versions=5000,
    )
    print(f"epoch {epoch}: reclaimed {reclaimed} versions, "
          f"amplification ~{index.stats()['capacity_amplification']:.2f}x")
```

## Amplification stays flat

ChronoVec's consolidation algorithm holds capacity amplification flat across full dataset turnovers:

- Merge candidate groups, not pairs (pairwise merging cannot reach the fill factor the capacity allows).
- Drop empty pages immediately after vacuum empties them.
- Bulk-load consolidates itself so a freshly built index that is never vacuumed doesn't accumulate pages.

At 200k SIFT-128 vectors, 5 full turnovers: amplification 1.276x → 1.331x (flat), 1,000,000 expired versions reclaimed.

## Concurrent reads during writes

Readers never block during writes:

```python
import threading

index = Index(dim, metric="cosine")

def reader():
    for _ in range(1000):
        results = index.search(np.random.rand(dim).astype("float32"), k=10)
        assert results is not None

def writer():
    for i in range(500):
        index.insert(i, np.random.rand(dim).astype("float32"))

threads = [threading.Thread(target=reader) for _ in range(4)]
threads.append(threading.Thread(target=writer))
for t in threads:
    t.start()
for t in threads:
    t.join()
```

## CDC stream pattern

```python
from chronovec import Collection

coll = Collection(dimensions=768, metric="cosine")

def on_insert(event):
    coll.add(
        ids=[event["id"]],
        embeddings=[embed(event["text"])],
        metadatas=[event["metadata"]],
    )

def on_delete(event):
    coll.delete(ids=[event["id"]])
    coll.vacuum(oldest_snapshot=coll.snapshot(), budget_versions=64)

def on_update(event):
    # Re-adding an existing id replaces it at one commit timestamp. Readers
    # see the old version or the new version, never a delete/insert gap.
    coll.add(
        ids=[event["id"]],
        embeddings=[embed(event["new_text"])],
        metadatas=[event["new_metadata"]],
    )
```

## Write-ahead log for durability

```python
index = Index(dim, wal_path="stream.wal")
# All inserts/deletes are logged before being applied.
# Recovery is automatic: open the same wal_path to replay.
```

## Batch writes for throughput

```python
ids = np.arange(10_000, dtype=np.int64)
vectors = np.random.rand(10_000, dim).astype("float32")

# One commit timestamp for the whole batch: atomic to readers
ts = index.insert_many(ids, vectors)
```
