# RAG with time travel

ChronoVec's snapshot reads let you query a knowledge base as it existed at any past point in time. This is useful for:

- **Reproducible evaluation:** run the same retrieval query as of a specific date and compare results.
- **Debugging retrieval regressions:** "what did the retriever return last Tuesday?"
- **Staged rollouts:** query a corpus as of the previous stable state while ingesting updates.
- **Multi-version documents:** maintain multiple versions of a document and query each version's embedding independently.

## Running the example

```bash
python examples/rag_time_travel.py
```

## Core pattern

```python
from chronovec import Collection
import numpy as np

coll = Collection(dimensions=768)

# Load initial corpus
coll.add(
    ids=["doc-1", "doc-2"],
    embeddings=[embed("Paris is the capital of France"),
                embed("The Louvre is a museum in Paris")],
    documents=["Paris is the capital of France",
               "The Louvre is a museum in Paris"],
)

# Capture a snapshot before the update
before = coll.snapshot()

# Update a document
coll.add(
    ids=["doc-1"],
    embeddings=[embed("Paris is the capital of France (updated)")],
    documents=["Paris is the capital of France (updated)"],
)

# Query latest
latest = coll.query(embed("France capital"), k=5)
print(latest[0].document)  # "Paris is the capital of France (updated)"

# Query as of the snapshot: returns old version
historical = coll.query(embed("France capital"), k=5, snapshot=before)
print(historical[0].document)  # "Paris is the capital of France"
```

## Staged rollout pattern

```python
# Production service always queries at the `stable` snapshot
stable = coll.snapshot()

# Ingest a batch of updates in the background
ingest_new_documents(coll)

# Evaluate new documents on a test set
eval_results = evaluate_retrieval(coll, test_queries)  # latest snapshot

if eval_results.recall >= threshold:
    stable = coll.snapshot()  # promote: production now uses new documents
# else: stable stays as-is; users are unaffected
```

## Timestamp pinning

Snapshots are integers (MVCC timestamps). Pin a snapshot for as long as you hold a reference to it:

```python
pinned = coll.snapshot()

# ... do a long evaluation job ...

results = coll.query(q, snapshot=pinned)  # consistent throughout the job
```

Pinned snapshots block reclamation of versions newer than the pin. Call `coll.vacuum(oldest_snapshot=pinned)` only when you are done with the pinned view.

## LlamaIndex: version-pinned retriever

```python
from chronovec.integrations.llamaindex import ChronoVecLlamaStore

store = ChronoVecLlamaStore(dimensions=768)
snapshot = store.snapshot()

# retrieve as of `snapshot` regardless of later inserts
results = store.query(query, snapshot=snapshot)
```
