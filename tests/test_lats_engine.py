"""Engine-level tests for examples/_lats_engine.py.

These use a small synthetic domain (not the incident-response narrative in
lats_chronovec.py) to isolate engine correctness -- delta reuse, concurrent
rollouts, checkpoint/resume -- from that example's specific judge logic.
"""

from __future__ import annotations

import asyncio
import runpy
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from _lats_engine import LATSAction, PersistentLATS  # noqa: E402

DIMENSIONS = 8
ACTIONS = tuple(LATSAction(f"a{i}", f"action {i}") for i in range(4))


def embed(text: str) -> np.ndarray:
    vector = np.zeros(DIMENSIONS, dtype=np.float32)
    vector[hash(text) % DIMENSIONS] = 1.0
    return vector


def execute(path: tuple[str, ...]) -> str:
    return f"observation for {path}"


BEST_PATH = ("a0", "a1")


def evaluate_best_is_all_a0(_branch, node) -> float:
    """Exactly one path scores 1.0 (no ties), so winner selection is unambiguous
    regardless of exploration order -- what sync vs. async need to agree on."""
    return 1.0 if node.path == BEST_PATH else 0.0


def _fresh_memory():
    from chronovec import AgentMemory

    return AgentMemory(DIMENSIONS, metric="cosine")


def _build(memory, **overrides):
    kwargs = dict(
        actions=ACTIONS,
        embed=embed,
        execute=execute,
        evaluate=evaluate_best_is_all_a0,
        max_depth=2,
        max_iterations=64,
        run_id="engine-test",
    )
    kwargs.update(overrides)
    return PersistentLATS(memory, **kwargs)


def test_expand_writes_only_its_own_step_not_the_whole_path():
    """Each node's branch holds exactly its own action+observation, not the path."""
    memory = _fresh_memory()
    search = _build(memory)
    search.run()
    zero_query = np.zeros(DIMENSIONS, dtype=np.float32)
    zero_query[0] = 1.0
    for node in search.nodes[1:]:
        assert node.branch is not None
        # Main has no records of its own here, so this branch's search sees
        # only what this one node itself wrote -- exactly 2, at any depth.
        hits = node.branch.search(zero_query, k=100)
        assert len(hits) == 2, f"node at depth {len(node.path)} has {len(hits)} records, expected 2"
    search.discard_all()
    memory.close()


def test_trajectory_search_merges_ancestor_chain_and_dedupes_main():
    memory = _fresh_memory()
    memory.add("shared-fact", embed("shared"), text="shared")
    search = _build(memory, max_depth=3, max_iterations=200)
    winner = search.run()
    query = embed("shared")
    hits = winner.trajectory_search(query, k=100)
    ids = [record.id for _, record in hits]
    # One record per ancestor step (up to 3, since max_depth=3) plus the
    # single main record, deduplicated -- not counted once per ancestor.
    assert ids.count("shared-fact") == 1
    assert len(ids) == len(set(ids))
    assert len(hits) == 2 * len(winner.path) + 1
    search.discard_all()
    memory.close()


def test_run_async_reaches_the_same_winner_as_run():
    memory_sync = _fresh_memory()
    search_sync = _build(memory_sync)
    winner_sync = search_sync.run()

    memory_async = _fresh_memory()
    search_async = _build(memory_async)
    winner_async = asyncio.run(search_async.run_async(concurrency=4))

    assert winner_sync.path == winner_async.path
    assert winner_sync.value == winner_async.value == 1.0
    assert search_sync.stats.expanded_nodes == search_async.stats.expanded_nodes
    assert search_sync.stats.scored_leaves == search_async.stats.scored_leaves

    search_sync.discard_all()
    search_async.discard_all()
    memory_sync.close()
    memory_async.close()


def test_run_async_spends_its_full_budget_under_contention():
    """A regression guard for a real bug: with a wider/deeper tree and
    concurrency > 1, a batch wave can land on a node another reservation in
    the same wave is already sitting on. The fix must route the rest of the
    batch to other work instead of one contention hit silently ending the
    whole search early -- test_run_async_reaches_the_same_winner_as_run's
    4-action, depth-2 tree is too small to ever hit this path (it's usually
    exhausted within a wave or two, before contention becomes likely), so it
    passed even with the bug present.
    """
    wide_actions = tuple(LATSAction(f"a{i}", f"action {i}") for i in range(6))
    # depth 3 with 6 actions has 6 + 6*5 + 6*5*4 = 156 possible nodes -- well
    # above the budget below, so the tree never exhausts and can't mask
    # contention by running out of new work to find anyway.
    budget = 40

    def build(memory, concurrency):
        return PersistentLATS(
            memory,
            actions=wide_actions,
            embed=embed,
            execute=execute,
            evaluate=evaluate_best_is_all_a0,
            max_depth=3,
            max_iterations=budget,
            run_id=f"contention-{concurrency}",
        )

    memory_sync = _fresh_memory()
    search_sync = build(memory_sync, 1)
    search_sync.run()
    sync_expanded = search_sync.stats.expanded_nodes
    search_sync.discard_all()
    memory_sync.close()

    # UCT's exploration term can legitimately spend an iteration re-visiting
    # an already-scored terminal instead of creating a new node -- which one
    # depends on traversal timing, so sync and async need not land on the
    # exact same count. A small tolerance still catches the actual bug this
    # guards: the batch silently giving up mid-wave, losing most of a
    # concurrency run's budget (the real failure here explored 26-52 out of
    # a 60-node budget, not 1 or 2 short of it).
    tolerance = 3
    for concurrency in (4, 16):
        memory_async = _fresh_memory()
        search_async = build(memory_async, concurrency)
        asyncio.run(search_async.run_async(concurrency=concurrency))
        assert search_async.stats.expanded_nodes >= sync_expanded - tolerance, (
            f"concurrency={concurrency} explored only {search_async.stats.expanded_nodes} "
            f"nodes out of a budget that reached {sync_expanded} synchronously -- the batch "
            f"is likely giving up early under contention instead of routing to other work"
        )
        search_async.discard_all()
        memory_async.close()


def test_run_async_overlaps_slow_tool_calls():
    """Concurrency should measurably hide per-call latency, not just execute correctly."""
    delay = 0.02

    async def slow_execute(path: tuple[str, ...]) -> str:
        await asyncio.sleep(delay)
        return f"observation for {path}"

    def build(memory, concurrency):
        search = PersistentLATS(
            memory,
            actions=ACTIONS,
            embed=embed,
            execute=slow_execute,
            evaluate=evaluate_best_is_all_a0,
            max_depth=2,
            max_iterations=20,
            run_id=f"latency-{concurrency}",
        )
        return search

    memory_serial = _fresh_memory()
    search_serial = build(memory_serial, 1)
    started = time.perf_counter()
    asyncio.run(search_serial.run_async(concurrency=1))
    serial_elapsed = time.perf_counter() - started
    search_serial.discard_all()
    memory_serial.close()

    memory_concurrent = _fresh_memory()
    search_concurrent = build(memory_concurrent, 8)
    started = time.perf_counter()
    asyncio.run(search_concurrent.run_async(concurrency=8))
    concurrent_elapsed = time.perf_counter() - started
    search_concurrent.discard_all()
    memory_concurrent.close()

    assert concurrent_elapsed < serial_elapsed * 0.6


def test_save_and_resume_round_trip(tmp_path):
    memory = _fresh_memory()
    search = _build(memory, max_iterations=6)
    search.run()
    still_open = [node for node in search.nodes if node.branch is not None]
    assert still_open

    checkpoint = tmp_path / "lats-checkpoint"
    search.save(checkpoint)
    memory.close()

    resumed = PersistentLATS.resume(
        checkpoint, embed=embed, execute=execute, evaluate=evaluate_best_is_all_a0
    )
    assert resumed.run_id == search.run_id
    assert resumed.max_depth == search.max_depth
    assert resumed.stats.expanded_nodes == search.stats.expanded_nodes
    assert len(resumed.nodes) == len(search.nodes)

    reattached = [node for node in resumed.nodes if node.branch_name is not None]
    assert len(reattached) == len(still_open)
    for node in reattached:
        # The reattached branch is immediately searchable, not just present.
        assert node.branch is not None
        node.branch.search(embed("probe"), k=1)

    # A resumed search can keep spending budget and reach a conclusion.
    resumed.max_iterations += 64
    winner = resumed.run()
    assert winner.path == BEST_PATH
    resumed.discard_all()
    resumed.memory.close()


def test_lats_chronovec_demo_still_runs_end_to_end():
    path = Path(__file__).resolve().parent.parent / "examples" / "lats_chronovec.py"
    runpy.run_path(str(path), run_name="__main__")
