"""Contract tests for persistent branch-delta LATS execution."""

import runpy
from pathlib import Path

import pytest


def _demo():
    path = Path(__file__).resolve().parent.parent / "examples" / "lats_chronovec.py"
    return runpy.run_path(str(path))


def _published_actions(memory, embed):
    hits = memory.search(embed("incident safety replica latency rollback"), k=40)
    return [
        record.payload["action"]
        for _, record in sorted(
            ((_hit, record) for _hit, record in hits if record.payload.get("kind") == "action"),
            key=lambda result: result[1].payload["position"],
        )
    ]


def test_persistent_lats_keeps_more_than_legacy_label_limit_live_and_publishes_one_leaf():
    namespace = _demo()
    memory = namespace["initial_memory"]()
    search = namespace["build_search"](memory)
    winner = search.run()

    assert search.stats.expanded_nodes == 85
    # 60 distinct complete trajectories (5*4*3 orderings), not 85: a
    # fully-expanded leaf gets re-selected by UCT for the rest of the
    # search budget but must not be re-scored each time.
    assert search.stats.scored_leaves == 60
    assert search.stats.peak_live_branches == 85
    assert len(tuple(memory.branches())) == 85
    assert winner.path == ("inspect-replica-lag", "route-read-traffic", "validate-rollback")
    assert winner.value > 0.4
    assert winner.branch_name is not None

    # Revisiting a node uses its existing private delta instead of rebuilding it.
    before = search.stats.branch_deltas_created
    winner.branch.search(namespace["embed"]("incident rollback"), k=10)
    assert search.stats.branch_deltas_created == before

    assert search.publish(winner) == 6
    assert _published_actions(memory, namespace["embed"]) == list(winner.path)
    assert tuple(memory.branches()) == ()
    # publish() merges every node along the winning path's own branch (3
    # nodes at depth 1/2/3), not just the leaf's -- each contributes one
    # action/observation pair, so 82 = 85 - 3 branches are discarded instead.
    assert search.stats.branch_deltas_discarded == 82
    memory.close()


def test_lats_requires_budget_to_reach_a_complete_trajectory():
    namespace = _demo()
    memory = namespace["initial_memory"]()
    search = namespace["build_search"](memory, max_iterations=1)
    with pytest.raises(RuntimeError, match="before reaching"):
        search.run()
    search.discard_all()
    memory.close()


def test_governing_runbook_is_retrieved_from_memory_not_hardcoded():
    """The judge's policy comes from what's in memory, not from a lookup table."""
    namespace = _demo()
    memory = namespace["initial_memory"]()
    governing_runbook = namespace["governing_runbook"]

    with_policy = memory.branch("with-policy")
    assert governing_runbook(with_policy, "route-read-traffic") is not None
    with_policy.discard()

    memory.delete("runbook-replica")
    without_policy = memory.branch("without-policy")
    assert governing_runbook(without_policy, "route-read-traffic") is None
    without_policy.discard()
    memory.close()


def test_evaluate_rejects_a_precondition_skipped_out_of_order():
    namespace = _demo()
    memory = namespace["initial_memory"]()
    embed = namespace["embed"]
    execute_action = namespace["execute_action"]
    actions_by_name = namespace["ACTIONS_BY_NAME"]

    # route-read-traffic before inspect-replica-lag: retrieval finds
    # runbook-replica governs the routing step and requires the inspection
    # first, so this must be rejected regardless of the tool's own wording.
    path = ("route-read-traffic", "inspect-replica-lag", "validate-rollback")
    branch = memory.branch("out-of-order")
    for position in range(len(path)):
        action = path[position]
        observation = execute_action(path[: position + 1])
        branch.add(
            f"a{position}",
            embed(actions_by_name[action].text),
            kind="action",
            action=action,
            position=position,
            text=actions_by_name[action].text,
        )
        branch.add(
            f"o{position}",
            embed(observation),
            kind="observation",
            action=action,
            position=position,
            text=observation,
        )
    node = namespace["LATSNode"](99, path, tuple(), None)
    assert namespace["evaluate"](branch, node) == 0.0
    branch.discard()
    memory.close()


def test_lats_releases_all_deltas_when_no_candidate_is_published():
    namespace = _demo()
    memory = namespace["initial_memory"]()
    search = namespace["build_search"](memory, max_iterations=32)
    winner = search.run()
    assert winner.branch_name is not None
    assert tuple(memory.branches())
    search.discard_all()
    assert tuple(memory.branches()) == ()
    assert search.stats.branch_deltas_discarded == search.stats.branch_deltas_created
    memory.close()
