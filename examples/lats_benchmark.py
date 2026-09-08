"""Benchmarks: does ChronoVec's branch-delta LATS actually scale?

``lats_chronovec.py`` is deliberately small: five domain actions, depth 3, an
explicable narrative. This script is the engineering-facing complement --
configurable depth/branching-factor/search budget, a head-to-head against a
"copy-per-trajectory" baseline (what you'd have to do without a native
branching primitive: copy the whole accumulated corpus into a fresh
`Collection` for every expanded node), and a concurrency section showing
`PersistentLATS.run_async` actually overlapping slow tool/LLM calls rather
than paying for them serially.

Usage:
    python examples/lats_benchmark.py
    python examples/lats_benchmark.py --depth 4 --branching-factor 6 --iterations 400
    python examples/lats_benchmark.py --concurrency 1 4 16 --rollout-latency-ms 20

The delta-branch and copy-per-trajectory runs execute in separate
subprocesses (`--engine delta` / `--engine copy`, printing one JSON line):
`resource.getrusage`'s peak-RSS counter is monotonic for the life of a
process, so running both approaches back to back in one interpreter would
let whichever ran first quietly set the high-water mark for both.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lats_engine import LATSAction, PersistentLATS  # noqa: E402

from chronovec import AgentMemory, Collection  # noqa: E402

WIDTH = 64


def embed(text: str) -> np.ndarray:
    """Same deterministic bag-of-words scheme as the other examples."""
    vector = np.zeros(WIDTH, dtype=np.float32)
    for word in text.lower().split():
        vector[zlib.crc32(word.encode()) % WIDTH] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def resident_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024)


def make_actions(branching_factor: int) -> tuple[LATSAction, ...]:
    return tuple(
        LATSAction(f"action-{i}", f"synthetic action {i} keyword{i}")
        for i in range(branching_factor)
    )


def make_best_path(depth: int) -> tuple[str, ...]:
    """The unique correct answer: pick actions 0, 1, 2, ... in order.

    Exactly one leaf scores 1.0 regardless of depth/branching-factor, so
    "result quality" (did the search find it) stays a meaningful, unambiguous
    signal at any scale swept.
    """
    return tuple(f"action-{i}" for i in range(depth))


def make_execute():
    def execute(path: tuple[str, ...]) -> str:
        return f"observation for {path[-1]} at depth {len(path)}"

    return execute


def make_evaluate(best_path: tuple[str, ...]):
    def evaluate(_branch, node) -> float:
        return 1.0 if node.path == best_path else 0.0

    return evaluate


def base_corpus(size: int) -> list[tuple[str, np.ndarray, dict]]:
    """Shared long-term memory every trajectory must see, regardless of engine."""
    return [
        (
            f"fact-{i}",
            embed(f"shared long term fact number {i}"),
            {"kind": "fact", "text": f"fact {i}"},
        )
        for i in range(size)
    ]


# -- delta-branch engine (ChronoVec AgentMemory) ------------------------------


def run_delta(
    depth: int,
    branching_factor: int,
    iterations: int,
    corpus_size: int,
    page_capacity: int = 256,
    branch_page_capacity: int | None = None,
) -> dict:
    actions = make_actions(branching_factor)
    best_path = make_best_path(depth)
    memory = AgentMemory(
        WIDTH,
        metric="cosine",
        page_capacity=page_capacity,
        branch_page_capacity=branch_page_capacity,
    )
    for item_id, vector, payload in base_corpus(corpus_size):
        memory.add(item_id, vector, **payload)

    search = PersistentLATS(
        memory,
        actions=actions,
        embed=embed,
        execute=make_execute(),
        evaluate=make_evaluate(best_path),
        max_depth=depth,
        max_iterations=iterations,
        run_id="benchmark",
    )
    started = time.perf_counter()
    winner = search.run()
    elapsed = time.perf_counter() - started
    search.discard_all()
    memory.close()
    return {
        "engine": "delta",
        "page_capacity": page_capacity,
        "branch_page_capacity": branch_page_capacity,
        "elapsed_s": elapsed,
        "expanded_nodes": search.stats.expanded_nodes,
        "scored_leaves": search.stats.scored_leaves,
        "peak_live_branches": search.stats.peak_live_branches,
        "records_created": search.stats.branch_deltas_created * 2,
        "peak_records_per_node": 2,
        "create_total_ms": search.stats.branch_create_ms,
        "create_avg_ms": search.stats.branch_create_ms / max(search.stats.branch_deltas_created, 1),
        "found_best_path": winner.path == best_path,
        "peak_resident_mb": resident_mb(),
    }


# -- copy-per-trajectory baseline ---------------------------------------------
#
# The same tree policy as PersistentLATS (deterministic, so given identical
# actions/execute/evaluate/depth/budget it visits the same nodes), but with no
# branching primitive: every expanded node gets a brand-new Collection holding
# a full copy of the base corpus plus every ancestor's own action/observation.
# This is what "no delta reuse" actually costs.


class _CopyNode:
    __slots__ = (
        "node_id",
        "path",
        "parent",
        "children",
        "visits",
        "total_value",
        "value",
        "record_count",
    )

    def __init__(self, node_id, path, parent):
        self.node_id = node_id
        self.path = path
        self.parent = parent
        self.children: list[_CopyNode] = []
        self.visits = 0
        self.total_value = 0.0
        self.value: float | None = None
        self.record_count = 0

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


def run_copy(depth: int, branching_factor: int, iterations: int, corpus_size: int) -> dict:
    actions = make_actions(branching_factor)
    actions_by_name = {a.name: a for a in actions}
    best_path = make_best_path(depth)
    base = base_corpus(corpus_size)
    execute = make_execute()

    root = _CopyNode(0, (), None)
    nodes = [root]
    leaves: list[_CopyNode] = []
    stats = {"copy_create_ms": 0.0, "records_created": 0, "peak_records_per_node": 0}

    def available(node: _CopyNode) -> list[LATSAction]:
        expanded = {child.path[-1] for child in node.children}
        return [a for a in actions if a.name not in node.path and a.name not in expanded]

    def terminal(node: _CopyNode) -> bool:
        return len(node.path) >= depth or not [a for a in actions if a.name not in node.path]

    def select() -> _CopyNode:
        node = root
        while not terminal(node):
            if available(node):
                return node
            parent_visits = max(node.visits, 1)
            node = max(
                node.children,
                key=lambda c: (
                    c.mean_value + 1.4 * math.sqrt(math.log(parent_visits) / max(c.visits, 1))
                ),
            )
        return node

    def materialize(parent: _CopyNode) -> _CopyNode:
        action = available(parent)[0]
        path = (*parent.path, action.name)
        node = _CopyNode(len(nodes), path, parent)
        parent.children.append(node)
        nodes.append(node)
        observation = execute(path)

        # No branch-of-a-branch, and no branch primitive at all here: rebuild
        # the whole trajectory's corpus from scratch, exactly what a system
        # without ChronoVec's branching would have to do per candidate.
        started = time.perf_counter()
        collection = Collection(dimensions=WIDTH)
        for item_id, vector, payload in base:
            collection.add(ids=[item_id], embeddings=[vector], metadatas=[payload])
        for position, action_name in enumerate(path):
            step_observation = (
                observation if position == len(path) - 1 else f"observation for {action_name}"
            )
            collection.add(
                ids=[f"node-{node.node_id}-action-{position}"],
                embeddings=[embed(actions_by_name[action_name].text)],
                metadatas=[{"kind": "action", "action": action_name, "position": position}],
            )
            collection.add(
                ids=[f"node-{node.node_id}-observation-{position}"],
                embeddings=[embed(step_observation)],
                metadatas=[{"kind": "observation", "action": action_name, "position": position}],
            )
        node.record_count = len(base) + 2 * len(path)
        stats["copy_create_ms"] += (time.perf_counter() - started) * 1_000
        stats["records_created"] += node.record_count
        stats["peak_records_per_node"] = max(stats["peak_records_per_node"], node.record_count)
        return node

    def backpropagate(node: _CopyNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    started = time.perf_counter()
    for _ in range(iterations):
        selected = select()
        node = selected if terminal(selected) else materialize(selected)
        if terminal(node):
            # Mirrors PersistentLATS._score: only ever scored once, the first
            # time this leaf is reached, however many more times UCT
            # reselects it for the rest of the budget.
            first_visit = node.value is None
            if first_visit:
                node.value = 1.0 if node.path == best_path else 0.0
            backpropagate(node, node.value)
            if first_visit:
                leaves.append(node)
    elapsed = time.perf_counter() - started
    if not leaves:
        raise RuntimeError("search budget ended before reaching a complete trajectory")
    winner = max(leaves, key=lambda n: (n.value or 0.0, n.visits))

    return {
        "engine": "copy",
        "elapsed_s": elapsed,
        "expanded_nodes": len(nodes) - 1,
        "scored_leaves": len(leaves),
        "peak_live_branches": None,
        "records_created": stats["records_created"],
        "peak_records_per_node": stats["peak_records_per_node"],
        "create_total_ms": stats["copy_create_ms"],
        "create_avg_ms": stats["copy_create_ms"] / max(len(nodes) - 1, 1),
        "found_best_path": winner.path == best_path,
        "peak_resident_mb": resident_mb(),
    }


# -- async concurrency section -------------------------------------------------


async def _run_concurrency_sweep(
    depth: int, branching_factor: int, iterations: int, latency_s: float, levels: list[int]
) -> list[dict]:
    async def slow_execute(path: tuple[str, ...]) -> str:
        await asyncio.sleep(latency_s)
        return f"observation for {path[-1]}"

    actions = make_actions(branching_factor)
    best_path = make_best_path(depth)
    results = []
    for concurrency in levels:
        memory = AgentMemory(WIDTH, metric="cosine")
        search = PersistentLATS(
            memory,
            actions=actions,
            embed=embed,
            execute=slow_execute,
            evaluate=make_evaluate(best_path),
            max_depth=depth,
            max_iterations=iterations,
            run_id=f"concurrency-{concurrency}",
        )
        started = time.perf_counter()
        winner = await search.run_async(concurrency=concurrency)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "concurrency": concurrency,
                "elapsed_s": elapsed,
                "nodes": search.stats.expanded_nodes,
                "found_best_path": winner.path == best_path,
            }
        )
        search.discard_all()
        memory.close()
    return results


def print_comparison(delta: dict, copy: dict) -> None:
    def row(label, d_val, c_val, fmt="{}"):
        print(f"  {label:<28} {fmt.format(d_val):>16} {fmt.format(c_val):>16}")

    print(f"\n  {'metric':<28} {'delta-branch':>16} {'copy-per-traj':>16}")
    row("explored nodes", delta["expanded_nodes"], copy["expanded_nodes"])
    row("scored leaves", delta["scored_leaves"], copy["scored_leaves"])
    row("node creation, total (ms)", delta["create_total_ms"], copy["create_total_ms"], "{:.2f}")
    row("node creation, avg (ms)", delta["create_avg_ms"], copy["create_avg_ms"], "{:.4f}")
    row("records written per node", delta["peak_records_per_node"], copy["peak_records_per_node"])
    row("total records written", delta["records_created"], copy["records_created"])
    row("peak resident memory (MB)", delta["peak_resident_mb"], copy["peak_resident_mb"], "{:.1f}")
    row("wall clock (s)", delta["elapsed_s"], copy["elapsed_s"], "{:.3f}")
    row("found the true best path", delta["found_best_path"], copy["found_best_path"])
    speedup = copy["create_total_ms"] / max(delta["create_total_ms"], 1e-9)
    print(
        f"\n  delta-branch node creation is {speedup:.1f}x faster in total than copy-per-trajectory"
    )
    print(
        "  and, unlike copy-per-trajectory, its per-node cost (records written) does not grow with depth."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--branching-factor", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--corpus-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--rollout-latency-ms", type=float, default=15.0)
    parser.add_argument("--engine", choices=["delta", "copy"], default=None)
    parser.add_argument(
        "--page-capacity",
        type=int,
        default=256,
        help="AgentMemory's main page_capacity (default 256, same as the library default). "
        "Only used by --engine delta.",
    )
    parser.add_argument(
        "--branch-page-capacity",
        type=int,
        default=None,
        help="AgentMemory's branch_page_capacity (default: unset, meaning branches use "
        "--page-capacity too, same as before this parameter existed). Only used by --engine delta.",
    )
    args = parser.parse_args()

    if args.engine is not None:
        # Subprocess mode: run exactly one engine, print one JSON line, exit.
        if args.engine == "delta":
            result = run_delta(
                args.depth,
                args.branching_factor,
                args.iterations,
                args.corpus_size,
                args.page_capacity,
                args.branch_page_capacity,
            )
        else:
            result = run_copy(args.depth, args.branching_factor, args.iterations, args.corpus_size)
        print(json.dumps(result))
        return

    print(
        f"LATS scale benchmark: depth={args.depth} branching_factor={args.branching_factor} "
        f"iterations={args.iterations} corpus_size={args.corpus_size}"
    )
    print(
        "running delta-branch and copy-per-trajectory in separate processes for clean peak-memory readings..."
    )
    forwarded = [
        "--depth",
        str(args.depth),
        "--branching-factor",
        str(args.branching_factor),
        "--iterations",
        str(args.iterations),
        "--corpus-size",
        str(args.corpus_size),
    ]
    delta_proc = subprocess.run(
        [sys.executable, __file__, "--engine", "delta", *forwarded],
        capture_output=True,
        text=True,
        check=True,
    )
    copy_proc = subprocess.run(
        [sys.executable, __file__, "--engine", "copy", *forwarded],
        capture_output=True,
        text=True,
        check=True,
    )
    delta = json.loads(delta_proc.stdout.strip().splitlines()[-1])
    copy = json.loads(copy_proc.stdout.strip().splitlines()[-1])
    print_comparison(delta, copy)

    # Every branch here ever holds exactly 2 records, however deep the tree
    # -- so sizing branches with the same page_capacity as main preallocates
    # far more native page space per branch than this one needs. This is a
    # real, measured cost, not a rounding error: with hundreds of live
    # branches, that fixed per-branch floor dominates. branch_page_capacity
    # sizes branches independently of main, so main's own page_capacity (and
    # therefore its page-count-driven routing/query performance at real
    # corpus scale) is completely unaffected by tuning this.
    tuned_proc = subprocess.run(
        [
            sys.executable,
            __file__,
            "--engine",
            "delta",
            "--page-capacity",
            str(args.page_capacity),
            "--branch-page-capacity",
            "8",
            *forwarded,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    tuned = json.loads(tuned_proc.stdout.strip().splitlines()[-1])
    print(
        f"\n  tip: branches sized at page_capacity={args.page_capacity} (main's own setting) cost "
        f"this run {delta['peak_resident_mb']:.1f} MB peak resident; branch_page_capacity=8 "
        f"(sized for this workload's 2-record branches, main's page_capacity={args.page_capacity} "
        f"untouched) costs {tuned['peak_resident_mb']:.1f} MB for the same {tuned['expanded_nodes']} "
        "branches. Set branch_page_capacity for many-small-branch workloads like this one -- it "
        "doesn't cost main anything, unlike lowering page_capacity globally would."
    )

    print(
        f"\nasync concurrency sweep: {args.rollout_latency_ms:.0f} ms simulated tool-call latency per step"
    )
    concurrency_results = asyncio.run(
        _run_concurrency_sweep(
            args.depth,
            args.branching_factor,
            args.iterations,
            args.rollout_latency_ms / 1000.0,
            args.concurrency,
        )
    )
    baseline = concurrency_results[0]["elapsed_s"]
    for result in concurrency_results:
        speedup = baseline / max(result["elapsed_s"], 1e-9)
        print(
            f"  concurrency={result['concurrency']:<3} "
            f"nodes={result['nodes']:<4} "
            f"found_best_path={str(result['found_best_path']):<5} "
            f"wall_clock={result['elapsed_s']:.2f}s "
            f"speedup_vs_concurrency_1={speedup:.2f}x"
        )

    assert delta["found_best_path"], "delta-branch engine did not find the true best path"
    assert copy["found_best_path"], "copy-per-trajectory baseline did not find the true best path"
    assert all(result["found_best_path"] for result in concurrency_results), (
        "an async concurrency level did not find the true best path"
    )
    # Every concurrency level should explore roughly the same amount of the
    # tree -- the actual invariant that matters here: a batch under
    # contention must route to other work instead of silently ending the
    # search early (see PersistentLATS.run_async). Wall-clock speedup is a
    # real, visible benefit at realistic tool-call latencies -- the printed
    # table above demonstrates it -- but isn't sound to hard-assert on at
    # this tiny synthetic scale: a few ms of simulated latency is easily
    # swamped by scheduling noise on a shared runner, so a strict wall-clock
    # comparison here would end up testing runner noise, not the algorithm.
    baseline_nodes = concurrency_results[0]["nodes"]
    for result in concurrency_results:
        assert result["nodes"] >= baseline_nodes - 1, (
            f"concurrency={result['concurrency']} explored only {result['nodes']} nodes, "
            f"vs {baseline_nodes} at concurrency=1 -- looks like the search gave up early"
        )


if __name__ == "__main__":
    main()
