# Compliance deletion (GDPR, right to erasure)

ChronoVec's bounded reclamation makes it straightforward to implement GDPR right-to-erasure requirements: delete a user's vectors, reclaim the space, and verify the data is gone, within a bounded number of operations.

## The problem with other indexes

- **HNSW / hnswlib:** `remove_ids` marks tombstones but does not reclaim memory. Rebuilding the index is the only way to physically remove data.
- **faiss:** `remove_ids` scans every inverted list. Cost is O(corpus) per call; slows tenfold as the corpus grows tenfold.
- **ChronoVec:** `delete` is O(1), `vacuum` reclaims at most `budget_versions` records per call: bounded by design.

## Example: user data erasure

```bash
python examples/compliance_deletion.py
```

```python
from chronovec import Index
import numpy as np

dim = 128
index = Index(dim, metric="cosine", page_capacity=256, nprobe=16)

# Insert a user's data
user_ids = list(range(100, 200))
for uid in user_ids:
    index.insert(uid, np.random.rand(dim).astype("float32"))

# Query before deletion: user records appear in results
before = index.search(np.random.rand(dim).astype("float32"), k=50)
user_hits_before = [r for r in before if r.id in set(user_ids)]
assert len(user_hits_before) > 0

# Delete all user records
for uid in user_ids:
    index.delete(uid)

# Vacuum: physically reclaim, on a budget
# oldest_snapshot=index.clock+1 means "reclaim anything deleted before now"
reclaimed = index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=200)
assert reclaimed == len(user_ids)

# Query after vacuum: user records are gone
after = index.search(np.random.rand(dim).astype("float32"), k=50)
user_hits_after = [r for r in after if r.id in set(user_ids)]
assert len(user_hits_after) == 0  # physically removed, not tombstoned

# Verify via time travel: even historical queries should be limited
# by setting the snapshot to after the vacuum
snapshot_after = index.clock
after_tv = index.search(np.random.rand(dim).astype("float32"), k=50,
                        snapshot=snapshot_after)
user_hits_tv = [r for r in after_tv if r.id in set(user_ids)]
assert len(user_hits_tv) == 0
```

## Bounded cost guarantee

`vacuum(oldest_snapshot, budget_versions=N)` reclaims **at most N versions** per call. Cost is O(N), not O(corpus). This means:

- You can issue deletions continuously and vacuum on a schedule.
- The vacuum call duration is predictable and bounded.
- You never need to rebuild the index to reclaim space.

## Statement for an auditor

After `delete` + `vacuum(oldest_snapshot=index.clock+1, budget_versions=N)`:

> "The user's vectors were logically deleted at time T. `vacuum` physically reclaimed the underlying storage at time T+1 (within N operations). A query at any snapshot ≥ T+1 will not return the user's records. The data is no longer present in the index."

## Multi-tenant erasure

For multi-tenant systems, use the `Collection` API with a `tenant_id` metadata field:

```python
from chronovec import Collection

coll = Collection(dimensions=768)
coll.add(
    ids=[f"user-{uid}-doc-{i}" for i in range(50)],
    embeddings=[...],
    metadatas=[{"tenant_id": uid} for _ in range(50)],
)

# Erase all records for tenant 42
coll.delete(where={"tenant_id": 42})
coll.vacuum(oldest_snapshot=coll.snapshot())
```
