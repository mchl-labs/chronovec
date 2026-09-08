---
layout: home

hero:
  name: "ChronoVec"
  image:
    light: /chronovec-logo-lockup-light.png
    dark: /chronovec-logo-lockup.png
    alt: "ChronoVec logo"
  text: "Vector memory that thinks in time"
  tagline: "Vector memory for data that changes"
  actions:
    - theme: brand
      text: Get started
      link: /getting-started
    - theme: alt
      text: Agent memory demo
      link: /use-cases/agent-memory
    - theme: alt
      text: GitHub
      link: https://github.com/mchl-labs/chronovec

features:
  - icon: 🕐
    title: Query the past
    details: Every record carries a version interval. Query the index as it existed at any past moment (for audit, reproducible evaluation, or retrieval regression debugging).
    link: /use-cases/rag-with-history
    linkText: RAG with time travel

  - icon: 🌿
    title: Branch and speculate
    details: Fork agent memory, explore a hypothesis, then discard or merge, all without rebuilding the index. Each branch gets its own private delta index by default, with full snapshot isolation and no fixed branch-count ceiling.
    link: /use-cases/agent-memory
    linkText: Agent memory guide

  - icon: 🗑️
    title: Delete for real, on a budget
    details: vacuum() physically reclaims expired records at a bounded cost, no full scan and no rebuild. Historical snapshots taken before the delete still read correctly.
    link: /use-cases/compliance
    linkText: Compliance & erasure

  - icon: ⚡
    title: Write without blocking reads
    details: Writes never block reads. A reader snapshots an immutable view and queries it lock-free, even while a batch is in progress.
    link: /architecture
    linkText: How MVCC works

  - icon: 🔌
    title: Drop-in integrations
    details: LangChain VectorStore, LlamaIndex store, asyncio-native AsyncCollection, DuckDB and SQLite adapters. The branching API threads through all of them.
    link: /integrations/langchain
    linkText: View integrations

  - icon: 💾
    title: Survive a crash
    details: Writes are logged before they are applied. A torn tail from a crash mid-append is detected by CRC, dropped, and the file truncated cleanly on restart.
    link: /getting-started#persistence
    linkText: WAL persistence

---

## Support and maturity

See the [support and maturity matrix](support-matrix) for the recommended
production path, binding status, and compatibility policy.

## Install

```bash
pip install chronovec
```

> Building from source requires a C++20 compiler and CMake (`pip install .` handles that automatically).

See the [Chroma migration guide](migration-from-chroma) for the quickest path
from a familiar vector-store API to versioned local memory.

## Quickstart

Run the offline agent-memory demo:

```bash
pip install chronovec
python examples/agent_memory.py
```

It retries a mistaken memory update on an isolated branch, merges the corrected
result, and keeps the original timeline queryable. For mutable RAG with
document corrections, see the [RAG with history](use-cases/rag-with-history)
guide.

For graph-based candidate evaluation, install the optional LangGraph adapter
and run `python examples/langgraph_branching_memory.py`. See the
[LangGraph integration guide](integrations/langgraph) for the branch lifecycle
and checkpointing contract.

For tree-search agents, run `python examples/lats_chronovec.py`: a worked
pattern (not a packaged integration) where every explored Language Agent Tree
Search trajectory gets its own live branch delta, so nodes can be revisited
and scored without rebuilding state, and only the winning trajectory is
merged. See the [tree-search agents guide](use-cases/lats).

## Five minutes

```python
from chronovec import Collection

col = Collection(dimensions=384, embedding_function=my_embed)

# Add with text: embedding happens automatically
col.add(ids=["a", "b"],
        documents=["the user prefers dark mode", "meeting at 3pm Friday"],
        metadatas=[{"kind": "pref"}, {"kind": "event"}])

# Query by text or by vector
results = col.query(query_text="appearance settings", k=3)

# Take a snapshot before a correction
t = col.snapshot()
col.add(ids=["a"], documents=["the user prefers light mode"])

# Travel back: "a" still says dark mode here
old = col.query(query_text="appearance settings", k=1, snapshot=t)
```

```python
from chronovec import AgentMemory

mem = AgentMemory(384)
mem.add("fact-1", embedding, text="user writes Python")

# Speculate on a branch
guess = mem.branch("hypothesis")
guess.add("guess-1", other_embedding, text="user might prefer TypeScript")

guess.search(query)   # sees both
mem.search(query)     # never saw the speculation

guess.discard()       # abandon speculation; purge later at a safe horizon
```
