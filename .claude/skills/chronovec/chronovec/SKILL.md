---
name: chronovec
description: "Use when building with ChronoVec: adding vectors, querying, taking snapshots, branching for speculative reasoning, or wiring up agent memory. Covers Collection, AgentMemory, and raw Index."
---

# ChronoVec: Agent Usage Guide

ChronoVec is an embeddable vector index with **snapshot isolation**, **bounded deletion**, and **branchable memory**. It is designed for agents that write continuously, need to query the past, or explore speculative reasoning paths without polluting their main memory.

---

## Which class to use

| You need... | Use |
|---|---|
| String IDs, metadata filters, `snapshot()` | `Collection` |
| Speculative branches, fork/merge/discard | `AgentMemory` |
| Maximum throughput, int64 IDs only | `Index` (raw engine) |

The `Collection` API is the right default. `AgentMemory` adds branching on top of it. Raw `Index` is for callers who never need metadata or string IDs.

---

## Collection: quick reference

```python
from chronovec import Collection

# With an embedding function (text to vector): a plain batch callable works,
# or pass CustomEmbedding/SentenceTransformerEmbedding/ProviderEmbedding
# (see "Embedding functions" below) when documents and queries need different
# handling.
col = Collection(dimensions=384, embedding_function=my_embed_fn)

# Add
col.add(ids=["mem-1", "mem-2"],
        documents=["user prefers dark mode", "meeting at 3pm"],
        metadatas=[{"source": "prefs"}, {"source": "calendar"}])

# Query: ALWAYS pass query_text or a pre-computed vector, never a raw string
# as the first positional arg (that slot expects np.ndarray)
results = col.query(query_text="user interface preferences", k=5)
results = col.query(my_vector, k=5)                        # pre-computed vector

# Filter
results = col.query(query_text="meeting", k=5,
                    where={"source": "calendar"})

# Iterate results
for r in results:
    print(r.id, r.distance, r.document, r.metadata)

# Take a snapshot *before* writes you want to query later
t = col.snapshot()
col.add(ids=["mem-3"], documents=["task completed"])

# Query as of t: mem-3 is invisible
old_results = col.query(query_text="task", k=5, snapshot=t)

# Reclaim space. Pass the oldest snapshot any reader still holds.
col.vacuum(keep_snapshot=t)
```

### Filter operators

```python
# Equality
where={"lang": "en"}
where={"lang": {"$eq": "en"}}

# Comparisons
where={"score": {"$gte": 0.8}}

# Membership
where={"tag": {"$in": ["agent", "memory"]}}

# Text match
where={"body": {"$contains": "GDPR"}}
where={"path": {"$regex": r"^docs/"}}

# Boolean
where={"$and": [{"lang": "en"}, {"score": {"$gte": 0.5}}]}
where={"$or": [{"tag": "urgent"}, {"priority": {"$eq": 1}}]}
```

### Persistence

Three persistence options, composable:

```python
# 1. WAL: crash recovery (replay on restart, nothing lost)
col = Collection(dimensions=384, wal_path="agent_memory.wal",
                 embedding_function=my_embed_fn)

# 2. Arena: disk-backed float payload (kernel-evictable mmap)
col = Collection(dimensions=384, wal_path="mem.wal",
                 arena_path="mem.arena", max_vectors=100_000,
                 embedding_function=my_embed_fn)
# When arena is full: auto-grows to max_vectors * growth_factor (default 2.0)
# via WAL replay; requires BOTH wal_path and arena_path to be set.

# 3. save / load: explicit checkpoint (portable, no WAL dependency)
col.save("snapshot")              # writes snapshot.cvec + snapshot.meta
col2 = Collection.load("snapshot", embedding_function=my_embed_fn)
# Restored collection is heap-resident; pass wal_path/arena_path again
# if you want to re-enable those on the restored instance.
```

### Embedding functions

`embedding_function` accepts, in order of what you'd reach for:

```python
# A plain batch callable: used for both documents and queries (original API)
col = Collection(384, embedding_function=lambda texts: my_model.encode(texts))

# CustomEmbedding: when the model wants different handling for indexing vs.
# retrieval (instruction prefixes, different endpoints, etc.)
from chronovec import CustomEmbedding
embedder = CustomEmbedding(embed_documents=my_embed_documents,
                           embed_query=my_embed_query)
col = Collection(384, embedding_function=embedder)

# Any object with embed_documents/embed_query (LangChain-shaped) or
# get_text_embedding_batch/get_query_embedding (LlamaIndex-shaped) also works
# directly, no need to wrap it.
```

Optional adapters, each its own extra so you don't pull in a dependency you don't need:

```python
# Local model, no network calls: pip install chronovec[sentence-transformers]
from chronovec import SentenceTransformerEmbedding
embedder = SentenceTransformerEmbedding("sentence-transformers/all-MiniLM-L6-v2")

# Hosted provider via LiteLLM: pip install chronovec[litellm]
from chronovec import ProviderEmbedding
embedder = ProviderEmbedding(provider="openai", model="text-embedding-3-small",
                             api_key=os.environ["OPENAI_API_KEY"])
```

**Footgun:** a single batch callable (or `CustomEmbedding` with no `embed_query`) is used symmetrically: the same function embeds both documents and queries. If retrieval quality looks off with a model that documents different instructions for query vs. document embedding (common with BGE/E5/Nomic/Voyage-style models), that's the first thing to check: pass `embed_query` explicitly.

---

## AgentMemory: branching for speculative reasoning

Use when the agent needs to explore a hypothesis without polluting the main timeline.

```python
from chronovec import AgentMemory
import numpy as np

mem = AgentMemory(dimensions=384)

# Write to main: string or int IDs both work
mem.add("pref-theme", base_vector, text="user prefers dark mode")

# Fork: creates an isolated branch
plan = mem.branch("hypothesis-a")
plan.add(2, other_vector, text="speculative: user wants light mode")

# Branch sees main + its own writes
hits = plan.search(query_vector, k=5)  # returns list[tuple[SearchResult, Record]]
for result, record in hits:
    print(record.id, result.distance, record.payload)

# Main never sees speculation
main_hits = mem.search(query_vector, k=5)  # same tuple structure

# Discard (bounded reclamation, no rebuild)
plan.discard()

# Or fold into main
plan2 = mem.branch("confirmed")
plan2.add(3, vec3, text="confirmed fact")
plan2.merge()
```

**Limits:** branches are flat (no nested forks) regardless of engine. `AgentMemory` defaults to a private delta index per branch (`branch_engine="delta"`), which has no branch-count ceiling. The legacy `branch_engine="labels"` engine caps out at 63 concurrent branches and raises `BranchError`, not silent truncation, when the limit is hit.

**Merge conflicts:** `merge()` defaults to `branch_merge_strategy="last_writer_wins"`: if main changed an id the branch also touched, the branch's value silently wins. Construct with `branch_merge_strategy="fail_on_conflict"` (only with the default delta engine) to raise `BranchConflictError` on those ids instead, so a real conflict doesn't get lost.

**Many small branches:** each branch's private delta index preallocates its full `page_capacity` up front (that's a real tradeoff for main: too few pages per capacity unit hurts routing at scale, see `docs/architecture.md`), so a workload with many short-lived, tiny branches (a tree-search agent, one branch per node) pays that same preallocation for no benefit. `AgentMemory(..., page_capacity=256, branch_page_capacity=8)` sizes branches independently of main: main keeps whatever `page_capacity` its own corpus needs, only new branches use the smaller setting. Only valid with `branch_engine="delta"`.

### Query the past

```python
t = mem.clock          # snapshot token
mem.add(4, vec4, text="new write")

# Search as of t: id 4 is invisible
old_hits = mem.search(query_vector, as_of=t)

# Search state at the moment a branch was created
hits = plan.as_of_fork(query_vector)
```

### Cleanup

```python
# Release vector space for deleted/discarded records
mem.purge()                          # drop metadata no snapshot can reach
```

---

## Key footguns

### 1. Passing a string as the first arg to `Collection.query()`

```python
# WRONG: first positional arg is np.ndarray, not text
col.query("what did the user say?", k=5)   # confusing error or garbage results

# RIGHT
col.query(query_text="what did the user say?", k=5)
col.query(precomputed_vector, k=5)
```

### 2. Taking a snapshot *after* the writes you want to capture

```python
# WRONG: t captures the state after the write
col.add(ids=["a"], documents=["important fact"])
t = col.snapshot()
col.query(query_text="fact", snapshot=t)   # t includes the write, not historical

# RIGHT: t captures the state before the write
t = col.snapshot()
col.add(ids=["a"], documents=["important fact"])
col.query(query_text="fact", snapshot=t)   # "a" is invisible
```

### 3. Calling methods on a closed branch

```python
plan.discard()
plan.search(query_vector)   # raises BranchError, check branch lifecycle
```

### 4. `AgentMemory.search()` returns tuples, not bare Records

```python
# WRONG: `r` is a tuple, not a Record
for r in mem.search(query_vector):
    print(r.id)      # AttributeError

# RIGHT
for hit, record in mem.search(query_vector):
    print(record.id, hit.distance)
```

---

## LangChain / LlamaIndex adapters

```python
# LangChain
from chronovec.integrations.langchain import ChronoVecVectorStore
store = ChronoVecVectorStore(embedding=embeddings, dimensions=384)
store.add_texts(["some text"])
docs = store.similarity_search("query", k=4)

# Speculative branch inside LangChain
with store.branch("hypothesis") as scratch:
    scratch.add_texts(["speculative fact"])
    scratch.similarity_search("query")   # sees base + speculation
# branch is auto-discarded on exit

# LlamaIndex
from chronovec.integrations.llamaindex import ChronoVecLlamaStore
```

---

## LangGraph adapter

```python
# LangGraph: one graph thread_id maps to one ChronoVec branch
from chronovec import AgentMemory
from chronovec.integrations import LangGraphMemory

memory = AgentMemory(dimensions=384)
branches = LangGraphMemory(memory)
candidate = branches.open("candidate-a")   # idempotent across node replays
candidate.add("new-fact", embedding, text="candidate fact")
branches.merge("candidate-a")              # or branches.discard("candidate-a")
```

`open()` reattaches to an already-open branch (safe for LangGraph's node
replay) and also to a same-named branch restored by `AgentMemory.load` after
a process restart, via the new `AgentMemory.get_branch()`. `cleanup(thread_id)`
releases one branch after a failed run; `close()` releases every branch the
adapter still owns.

---

## Tree-search agents (LATS): a pattern, not a packaged integration

There's no `chronovec.integrations` module for this; `examples/_lats_engine.py`
is a complete, self-contained `PersistentLATS`/`LATSNode` implementation
built directly on `AgentMemory.branch()`, shared by `examples/lats_chronovec.py`
(a small explicable walkthrough) and `examples/lats_benchmark.py` (scale,
concurrency, checkpoint/resume, and a baseline comparison); copy it into
your own code rather than importing the example scripts.

```python
# Every explored tree node gets its own O(1) live branch delta: just the one
# action/observation it contributes, not its whole path. Full trajectory
# context is reconstructed at read time, not by rewriting ancestors' data.
branch = memory.branch(f"node-{node_id}")     # from examples/_lats_engine.py
node.trajectory_search(query_vector, k=10)    # merges this node + every ancestor's branch
...
promoted = winner_branch.merge()              # merges the winner
loser_branch.discard()                        # discards every other trajectory
```

`evaluate(branch, node)` gets the trajectory's own live branch; searching it
sees both this node's own writes and whatever long-term memory the tree was
seeded with, so a judge can retrieve the applicable policy per step (e.g. by
nearest-neighbour search against seeded runbooks) instead of hardcoding
per-action rules; call `node.trajectory_search(...)` instead when the check
needs the *whole* trajectory's evidence, not just this step's. See
`examples/lats_chronovec.py` for a worked judge built this way, and
`LATSStats` for the lifecycle-cost fields the search tracks (branch
create/discard cost, peak live branches, ...).

`PersistentLATS.run_async(concurrency=N)` runs several rollouts' tool/LLM
calls concurrently (`execute`/`evaluate` may be plain sync callables,
offloaded to a thread, or real async ones, awaited directly, see
`_call_maybe_async`). `search.save(path)` / `PersistentLATS.resume(path,
embed=..., execute=..., evaluate=...)` checkpoint and reattach a search's
`AgentMemory` and tree metadata across a process boundary; the caller
supplies the callables again, since they aren't serializable. See
`examples/lats_benchmark.py` for a configurable depth/branching-factor/budget
comparison against a naive copy-per-trajectory baseline, and for a real,
measured caveat: sizing branches at main's own `page_capacity=256` costs more
fixed per-branch memory than a 2-record LATS branch needs. Pass
`branch_page_capacity=8` to `AgentMemory` for a many-small-branch workload
like this: it sizes only new branches, leaving main's own `page_capacity`
(and the routing/query performance it protects at real corpus scale)
untouched.

---

## Snapshot pattern for agent evaluation loops

```python
# Typical agent memory pattern:
# 1. Record the state before a run
# 2. Let the agent write freely
# 3. Audit or roll back to the pre-run state

col = Collection(384, embedding_function=embed)

pre_run = col.snapshot()

# ... agent runs, adds memories ...

# Reproduce what the retriever saw before the run
audit_results = col.query(query_text="decision context", snapshot=pre_run)

# Free space from this run's writes if rolling back
col.vacuum(keep_snapshot=pre_run)
```
