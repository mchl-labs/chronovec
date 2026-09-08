# Agent memory with branching

ChronoVec's `AgentMemory` and branching API are designed for agentic workloads where the agent needs to:

- Maintain memory across agent interactions
- Speculate (explore a hypothesis without committing it)
- Rewind (start a clean branch from what the agent knew at a previous step)
- Erase records with a bounded cost (GDPR, multi-tenancy)

## Running the example

```bash
python examples/agent_memory.py
```

## Core pattern: rewind, retry, and merge

```python
from chronovec import AgentMemory
import numpy as np

memory = AgentMemory(dimensions=384)

# Establish base facts
memory.add(1, np.random.rand(384).astype("float32"), text="Paris is the capital of France")
memory.add(2, np.random.rand(384).astype("float32"), text="The Eiffel Tower is in Paris")

# Open a speculative branch
plan = memory.branch("plan-a")
plan.add(3, np.random.rand(384).astype("float32"), text="speculative: relocate capital")

# Branch sees base + speculation
print(len(plan.search(np.random.rand(384).astype("float32"), k=10)))  # 3

# Main never sees the speculation
print(len(memory.search(np.random.rand(384).astype("float32"), k=10)))  # 2

# Discard: abandon speculation; reclaim later at a safe horizon
plan.discard()
```

`AgentMemory` can be checkpointed explicitly with `memory.save("agent")` and
restored with `AgentMemory.load("agent")`. Checkpoints include payloads and
active branch metadata as well as vectors. A manifest atomically selects a
complete native checkpoint and metadata generation; incomplete or mismatched
generations are rejected on load.

## Rewind and retry from an earlier interaction

Capture a snapshot before a risky tool call or reasoning step. If the future
turns out to be wrong, fork from that snapshot and continue there. The later
main-memory writes remain in the original timeline and do not contaminate the
retry branch.

```python
checkpoint = memory.snapshot()
memory.add("bad-plan", embedding, text="an unverified plan")

retry = memory.branch("retry", snapshot=checkpoint)
retry.add("good-plan", embedding, text="a corrected plan")
retry.search(query)  # sees the clean checkpoint plus the corrected plan
memory.search(query)  # remains on the original timeline
```

The complete runnable example is in
[`examples/agent_memory.py`](https://github.com/mchl-labs/chronovec/blob/main/examples/agent_memory.py). It shows a
mistaken preference update, a retry from the clean checkpoint, and a merge of
the corrected branch while the pre-interaction state remains queryable.

## Many concurrent branches

Each branch gets its own private native delta index by default, so branch
count isn't capped and idle branches cost only their own metadata; main
pages, centroids, and routing are untouched until a branch merges:

```python
branches = [memory.branch(f"plan-{i}") for i in range(200)]
for b in branches:
    b.discard()
```

(`AgentMemory(..., branch_engine="labels")` restores the previous
shared-index implementation, which caps out at 63 concurrent branches.)

## Merge: commit speculation to main

```python
branch = memory.branch("research")
branch.add(4, embedding, text="discovered: important context")
branch.merge()  # promotes all branch writes to main
# now main sees record 4
```

## Time travel: query past beliefs

```python
t1 = memory.add(5, embedding, text="initial belief")
# ... time passes, belief is corrected ...
memory.delete(5)
memory.add(5, new_embedding, text="corrected belief")

# query as of t1: returns old belief
old_results = memory.search(query, k=10, as_of=t1)
```

## LangChain: branching memory in a chain

```python
from chronovec.integrations.langchain import ChronoVecVectorStore

store = ChronoVecVectorStore(embedding=embeddings, dimensions=1536)
store.add_texts(["user prefers dark mode"])

with store.branch("hypothesis") as scratch:
    scratch.add_texts(["user might prefer light mode"])
    retriever = scratch.as_retriever(search_kwargs={"k": 3})
    # run retrieval over speculative + confirmed memory
# hypothesis is discarded when the block exits
```

## Garbage collection

```python
# After merging or discarding branches, reclaim space
memory.purge(oldest_snapshot=memory.snapshot())
```
