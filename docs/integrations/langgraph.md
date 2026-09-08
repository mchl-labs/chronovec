# LangGraph

ChronoVec can give a LangGraph execution an isolated semantic-memory branch.
This is useful when a graph explores several candidate plans, policies, or
agent trajectories and should publish only the selected result.

`LangGraphMemory` itself has no dependency on the `langgraph` package --
it only maps thread ids to branches, framework-agnostic by design. The
optional extra is for actually building and running a real LangGraph graph
(as in the example below), not for importing the adapter:

```bash
pip install "chronovec[langgraph]"
```

The `LangGraphMemory` adapter maps one LangGraph `thread_id` to one
ChronoVec branch:

```python
from chronovec import AgentMemory
from chronovec.integrations import LangGraphMemory

memory = AgentMemory(dimensions=384)
branches = LangGraphMemory(memory)

candidate = branches.open("candidate-a")
candidate.add("new-fact", embedding, text="candidate fact")

# Publish only after the graph evaluator accepts the candidate.
branches.merge("candidate-a")
# Or: branches.discard("candidate-a")
```

`open()` is idempotent for an existing thread, which matters when LangGraph
replays a node from a checkpoint. It also reattaches to an active same-named
branch restored by `AgentMemory.load`, so a workflow can resume after a
process restart when both the graph checkpoint and ChronoVec checkpoint are
durable. A failed run can call `cleanup(thread_id)`; `close()` discards every
branch still owned by the adapter. LangGraph’s checkpointer stores graph state,
while ChronoVec provides snapshot-isolated semantic state and atomic
merge/discard.

The runnable example is
[`examples/langgraph_branching_memory.py`](https://github.com/mchl-labs/chronovec/blob/main/examples/langgraph_branching_memory.py).
It evaluates narrow, correct, and unsafe refund-policy candidates, then merges
only the accepted policy into the main memory. It uses a deterministic local
embedding, so no model key or external service is required.
