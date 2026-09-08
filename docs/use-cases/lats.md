# Tree-search agents (LATS)

[`examples/lats_chronovec.py`](https://github.com/mchl-labs/chronovec/blob/main/examples/lats_chronovec.py)
is a complete, runnable Language Agent Tree Search (LATS) implementation built
directly on the default `AgentMemory` branch-delta engine, and
[`examples/lats_benchmark.py`](https://github.com/mchl-labs/chronovec/blob/main/examples/lats_benchmark.py)
is its engineering-facing complement: configurable depth/branching-factor/
budget, a head-to-head against a naive "copy-per-trajectory" baseline, and a
concurrency sweep. Both import their shared engine from
[`examples/_lats_engine.py`](https://github.com/mchl-labs/chronovec/blob/main/examples/_lats_engine.py).
There is no "LATS integration" to install: this is demo code, not a
`chronovec.integrations` module. Copy and adapt it for your own search rather
than importing it.

Every expanded tree node owns a live private ANN delta: `memory.branch(name)`.
At search completion, `publish(winner)` atomically merges the winning
trajectory's deltas and discards every other one.

```python
from chronovec import AgentMemory
# LATSAction, LATSNode, PersistentLATS are defined in examples/_lats_engine.py;
# copy them into your own module rather than importing the example script.

memory = AgentMemory(dimensions=384)
search = PersistentLATS(
    memory,
    actions=[LATSAction("inspect", "Inspect replica lag"), ...],
    embed=embed,
    execute=run_tool,
    evaluate=judge_trajectory,
    max_depth=3,
    max_iterations=256,
)
winner = search.run()
search.publish(winner)
```

`execute(path)` returns the observation for the newly expanded action.
`evaluate(branch, node)` receives that node's own live branch and can perform
semantic retrieval (including `node.trajectory_search(...)` for full-path
evidence, see below), invoke an LLM judge, or combine tool evidence into a
value in `[0, 1]`. The runner uses UCT selection and backpropagates each leaf
value through the tree. A fully-expanded leaf can be re-selected for the rest
of the search budget; `evaluate` is cached per node so revisiting one never
re-runs the judge or re-searches its branch.

### Delta reuse, not rematerialization

`AgentMemory` branches are flat: forking always happens from the shared main
state, never from another branch (see `branch()` in the
[API reference](../api-reference)). That rules out a literal branch-of-a-branch
tree. This engine gets the same
effect a different way: **every node's branch holds only the one action and
observation that node itself contributes** (not its whole path) and full
trajectory context is reconstructed at read time by walking the node's parent
chain and merging each ancestor's own branch search
(`LATSNode.trajectory_search`). Branch *creation* is therefore O(1) per node
regardless of tree depth, never O(depth). `lats_benchmark.py` measures this
directly against a baseline that has no such trick available (see below).

### Why the branch matters here, not just the tree search

`evaluate` receiving a *live, searchable branch* rather than a static
trajectory summary is the point of building LATS on `AgentMemory`. The
incident-response demo (`examples/lats_chronovec.py`) seeds four safety
runbooks into long-term memory before the search starts. Its judge does not
hardcode which action each runbook governs: for every step, it embeds that
step's own action text, searches a branch (any branch works for this, since
runbooks live in main and every branch sees main), and only enforces a
runbook's precondition or prohibition if the *nearest retrieved* record turns
out to name that action. Remove a runbook from memory, or add a new one, and
the judge's behavior changes with it: nothing about compliance is baked into
per-action `if` branches. The final trajectory quality is a cosine-similarity
score, via `node.trajectory_search`, between the *whole trajectory's* own
recorded evidence and the incident's resolution goal, not a hardcoded string
match, and not limited to what any one branch alone recorded.

The demo explores 85 nodes across 60 distinct complete trajectories and keeps
all 85 branch deltas live at once (more than the legacy label engine's
63-branch limit), then merges only `inspect → route traffic → validate
rollback`. Running it prints, step by step, which runbook was retrieved for
the winning trajectory, plus node/leaf counts, peak live branches, branch
lifecycle cost, the goal-alignment score, and the final publication result.

### Scale, concurrency, and a baseline comparison

`examples/lats_benchmark.py` is where the "does this actually scale" claim
gets measured rather than asserted:

```bash
python examples/lats_benchmark.py --depth 4 --branching-factor 6 --iterations 400
```

It runs the delta-branch engine and a copy-per-trajectory baseline (a fresh
`Collection`, holding a full copy of the base corpus plus every ancestor's
own writes, rebuilt for every node: what you'd have to do without a native
branching primitive) in separate subprocesses, and reports explored nodes,
node-creation time (total and average), records written per node, peak
resident memory, wall-clock time, and whether each found the true best
trajectory. It also prints a concrete, real finding, not just a favorable
one: sizing branches at main's own `page_capacity` preallocates far more
native page space per branch than a 2-record LATS branch ever needs, so
hundreds of live branches carry a real fixed memory cost at that setting.
`AgentMemory(..., branch_page_capacity=8)` sizes branches independently of
main (the benchmark measures the effect); main's own `page_capacity`, and
the routing/query performance it protects at real corpus scale, is
completely unaffected; only new branches use the smaller setting.

The same script's concurrency sweep runs `PersistentLATS.run_async` at
several concurrency levels against a synthetic per-step latency, to show
concurrent rollouts actually overlapping slow tool/LLM calls rather than
paying for them serially. `execute`/`evaluate` may be plain sync callables
(offloaded to a thread) or real async ones (e.g. an LLM client's own async
method) interchangeably; see `_call_maybe_async` in `_lats_engine.py`.

### Checkpoint and resume

`PersistentLATS.save(path)` checkpoints the `AgentMemory` (main plus every
still-open branch (`AgentMemory.save` already persists active branches) and
the tree's metadata (visits, values, parent/child structure) to a JSON
sidecar. `PersistentLATS.resume(path, embed=..., execute=..., evaluate=...)`
reloads both and reattaches every not-yet-published-or-discarded node to its
live branch, ready to keep spending search budget
(`resumed.max_iterations += more; resumed.run()`) or to `publish`/
`discard_all` to conclude it. The callables themselves aren't serialized:
the caller supplies them again, since a real tool/LLM client can't be pickled
meaningfully across a process boundary.

For very large or expensive searches, callers may run separate `PersistentLATS`
instances per task, cap `max_iterations`, or call `discard_all()` to release a
search without publication.
