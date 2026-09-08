"""Language Agent Tree Search implemented directly on AgentMemory branches.

This is not a ChronoVec integration with a "LATS" library -- there isn't one.
It is a worked, explicable walkthrough of a pattern AgentMemory's branch-delta
engine enables: every explored tree node owns a live, private ANN delta
branch. Revisiting a node reuses that delta directly, and creating one never
rewrites its ancestors' data (see ``_lats_engine.py`` for how). At
termination, only the winning complete trajectory is merged and every other
branch is discarded.

For configurable depth/branching-factor/budget sweeps, async concurrent
rollouts, checkpoint/resume, and a benchmark against a copy-per-trajectory
baseline, see ``examples/lats_benchmark.py`` -- this file stays intentionally
small and narrative.

The part that makes this a genuine ChronoVec pattern rather than a tree search
with vector-store decoration: ``evaluate()`` below does not know in advance
which action a safety policy governs. For every step of a trajectory it runs
a nearest-neighbour search (via ``governing_runbook``, which sees main's
long-term memory from any branch) to *retrieve* the governing runbook, then
enforces whatever precondition or prohibition that runbook carries. Delete a
runbook from memory, or add a new one, and the judge's behavior changes with
it--nothing about compliance is hardcoded per action.

    python examples/lats_chronovec.py
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path

import numpy as np

from chronovec import AgentMemory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lats_engine import LATSAction, LATSNode, PersistentLATS  # noqa: E402

WIDTH = 256
ACTIONS = (
    LATSAction("inspect-replica-lag", "Inspect replica lag and active queries."),
    LATSAction("route-read-traffic", "Route reads to a healthy replica."),
    LATSAction("validate-rollback", "Verify that this mitigation has a tested rollback ready."),
    LATSAction("increase-cache-ttl", "Increase the product cache TTL temporarily."),
    LATSAction("disable-audit-logging", "Disable audit logging to reduce write load."),
)
ACTIONS_BY_NAME = {action.name: action for action in ACTIONS}

# Each runbook is retrievable long-term memory, not a lookup table the judge
# already has memorized. "governs_action" only decides which action a runbook
# is *about* once it has already been retrieved as the nearest match--it is
# never used to pick the runbook up front.
RUNBOOKS = (
    (
        "runbook-replica",
        "Route traffic only after a healthy replica is verified.",
        {"governs_action": "route-read-traffic", "requires_action": "inspect-replica-lag"},
    ),
    (
        "runbook-rollback",
        "Every production mitigation needs a tested rollback.",
        {"governs_action": "validate-rollback", "requires_action": "route-read-traffic"},
    ),
    (
        "runbook-freshness",
        "Inventory freshness SLO prohibits a long cache TTL.",
        {"governs_action": "increase-cache-ttl", "prohibited": True},
    ),
    (
        "runbook-audit",
        "Audit logging must remain enabled during incident mitigation.",
        {"governs_action": "disable-audit-logging", "prohibited": True},
    ),
)
GOAL_QUERY = "incident resolved: healthy replica confirmed, traffic rerouted, latency improved"


def embed(text: str) -> np.ndarray:
    """Create deterministic embeddings so the example needs no external model."""
    vector = np.zeros(WIDTH, dtype=np.float32)
    for word in text.lower().split():
        vector[zlib.crc32(word.encode()) % WIDTH] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def execute_action(path: tuple[str, ...]) -> str:
    """Deterministic incident tool result for the final action in ``path``."""
    action = path[-1]
    if action == "inspect-replica-lag":
        return "Tool observation: replica-eu-2 is healthy, has spare capacity, and p95 is 20 ms."
    if action == "route-read-traffic":
        if "inspect-replica-lag" in path[:-1]:
            return "Tool observation: shifting reads to replica-eu-2 reduces p95 latency to 90 ms."
        return "Tool observation: routing refused because no healthy replica was verified first."
    if action == "validate-rollback":
        if path[-2:] == ("route-read-traffic", "validate-rollback"):
            return "Tool observation: traffic can be rolled back in under one minute with no data loss."
        return "Tool observation: rollback is unverified because no traffic change was completed."
    if action == "increase-cache-ttl":
        return "Tool observation: a longer cache TTL makes inventory stale and violates the freshness SLO."
    return "Tool observation: disabling audit logging violates the incident safety policy."


def governing_runbook(branch, action_name: str) -> dict | None:
    """Retrieve the runbook nearest the action's own text, by vector search.

    This is the crux of the ChronoVec-specific behavior: the search does not
    know ahead of time whether ``action_name`` is gated. It embeds the
    action's description and searches ``branch`` -- any live branch works
    here, since runbooks live in main and every branch sees main -- and only
    treats the nearest *runbook* record as governing if the retrieved runbook
    actually names this action; otherwise no policy applies and the step is
    unconstrained.
    """
    hits = branch.search(embed(ACTIONS_BY_NAME[action_name].text), k=8)
    for _, record in hits:
        if record.payload.get("kind") == "runbook":
            payload = record.payload
            return payload if payload.get("governs_action") == action_name else None
    return None


def evaluate(branch, node: LATSNode) -> float:
    """Judge a trajectory by retrieving, per step, whichever runbook governs it.

    A step is rejected the moment its retrieved runbook is violated--either
    an outright prohibition, or a precondition action missing earlier in the
    path. A trajectory that clears every retrieved policy is scored by how
    closely its own recorded evidence matches the incident's resolution goal.
    Unlike the per-step runbook check, the goal-alignment check needs the
    *whole* trajectory's evidence, which is spread across this node's own
    branch and each of its ancestors' -- hence ``node.trajectory_search``
    instead of ``branch.search`` for just this one query.
    """
    for position, action_name in enumerate(node.path):
        runbook = governing_runbook(branch, action_name)
        if runbook is None:
            continue
        if runbook.get("prohibited"):
            return 0.0
        required = runbook.get("requires_action")
        if required is not None and required not in node.path[:position]:
            return 0.0
    hits = node.trajectory_search(embed(GOAL_QUERY), k=1)
    if not hits:
        return 0.0
    result, _record = hits[0]
    achieved = max(0.0, min(1.0, 1.0 - result.distance))
    # Below this, the branch's own recorded evidence isn't close enough to
    # the resolution goal to call the incident resolved.
    return achieved if achieved >= 0.35 else 0.0


def initial_memory() -> AgentMemory:
    memory = AgentMemory(WIDTH, metric="cosine", nprobe=64)
    for item_id, text, payload in RUNBOOKS:
        memory.add(item_id, embed(text), kind="runbook", text=text, **payload)
    return memory


def build_search(memory: AgentMemory, *, max_iterations: int = 256) -> PersistentLATS:
    return PersistentLATS(
        memory,
        actions=ACTIONS,
        embed=embed,
        execute=execute_action,
        evaluate=evaluate,
        max_depth=3,
        max_iterations=max_iterations,
        exploration=2.0,
        run_id="incident-lats",
    )


def explain_winner(search: PersistentLATS, winner: LATSNode) -> list[str]:
    """Return, step by step, which retrieved runbook cleared each action.

    Must be called before ``publish()``: it reads the winner's live branch,
    which publish discards along with every losing delta. This replays the
    same retrieval ``evaluate()`` used, so what's returned is exactly what
    the judge saw--not a post-hoc narrative.
    """
    branch = winner.branch
    lines = []
    for position, action_name in enumerate(winner.path):
        runbook = governing_runbook(branch, action_name)
        if runbook is None:
            lines.append(f"    {position + 1}. {action_name:<22} -> no governing runbook retrieved")
        else:
            required = runbook.get("requires_action")
            note = f"required {required!r} satisfied" if required else "no precondition"
            lines.append(f"    {position + 1}. {action_name:<22} -> runbook enforced ({note})")
    return lines


def main() -> None:
    memory = initial_memory()
    search = build_search(memory)
    winner = search.run()
    live_before_publish = tuple(memory.branches())

    # Revisiting a scored node's branch costs nothing extra: it is the same
    # private delta, not rebuilt from the winning path.
    created_before_revisit = search.stats.branch_deltas_created
    winner.branch.search(embed("incident rollback"), k=10)
    revisit_created_new_delta = search.stats.branch_deltas_created != created_before_revisit

    # Must run before publish(): it discards every other delta and clears
    # winner.branch once the winning trajectory is merged into main.
    winner_explanation = explain_winner(search, winner)

    promoted = search.publish(winner)
    hits = memory.search(embed("incident safety replica latency rollback"), k=40)
    actions = [
        record.payload["action"]
        for _, record in sorted(
            ((_hit, record) for _hit, record in hits if record.payload.get("kind") == "action"),
            key=lambda result: result[1].payload["position"],
        )
    ]

    print("Persistent-delta LATS on AgentMemory branches")
    print(
        f"  explored {search.stats.expanded_nodes} nodes, scored {search.stats.scored_leaves} "
        "complete trajectories"
    )
    print(
        f"  peak live branch deltas: {search.stats.peak_live_branches} "
        "(the legacy branch_engine='labels' backend caps out at 63 and would raise BranchError here)"
    )
    print(
        f"  branch delta create/discard cost: {search.stats.branch_create_ms:.2f} ms / "
        f"{search.stats.branch_discard_ms:.2f} ms total for all {search.stats.expanded_nodes} nodes "
        "-- each node writes only its own step (O(1)), not its whole path"
    )
    print(f"  active branches before publish: {len(live_before_publish)}")
    print(
        f"  revisiting the winner's branch created a new delta: {revisit_created_new_delta} (must be False)"
    )
    print("  winning trajectory, runbook retrieved per step:")
    for line in winner_explanation:
        print(line)
    print(
        f"  winner: {' > '.join(winner.path)}; goal-alignment score: {winner.value:.2f} "
        "(cosine similarity between the branch's own recorded evidence and the resolution goal); "
        f"promoted records: {promoted}"
    )
    print(
        f"  published actions: {actions}; open branches after publish: {tuple(memory.branches())}"
    )

    assert winner.path == ("inspect-replica-lag", "route-read-traffic", "validate-rollback")
    assert winner.value > 0.4
    assert len(live_before_publish) > 63
    assert revisit_created_new_delta is False
    assert actions == list(winner.path)
    assert not tuple(memory.branches())
    memory.close()


if __name__ == "__main__":
    main()
