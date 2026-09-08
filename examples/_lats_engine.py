"""Shared Language Agent Tree Search engine for the LATS examples.

This is demo code, not a public ChronoVec API: copy and adapt it for your own
search rather than importing it from application code. It backs both
``lats_chronovec.py`` (a small, explicable incident-response walkthrough) and
``lats_benchmark.py`` (scale, concurrency, and baseline-comparison benchmarks).

AgentMemory branches are flat -- forking always happens from the shared main
state, never from another branch (see the "Limits" note on
``AgentMemory.branch`` and the SKILL.md guide). That rules out a literal
branch-of-a-branch tree. This engine gets the same effect -- each tree node
paying only for what *it* adds, not for re-copying every ancestor's writes --
by giving every node its own tiny private branch (exactly the one action and
observation that node itself contributes) and reconstructing full-trajectory
context in Python by walking the node's parent chain and merging each
ancestor's own branch search (see ``LATSNode.trajectory_search``). Branch
creation is therefore O(1) per node regardless of tree depth, not O(depth).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chronovec import AgentMemory

Embed = Callable[[str], Any]
Execute = Callable[[tuple[str, ...]], str]
Evaluate = Callable[[Any, "LATSNode"], float]


@dataclass(frozen=True)
class LATSAction:
    """One action made available to the search policy."""

    name: str
    text: str


@dataclass
class LATSNode:
    """A trajectory node backed by one active ChronoVec branch."""

    node_id: int
    path: tuple[str, ...]
    observations: tuple[str, ...]
    # parent/children form a reference cycle (a child's parent lists that
    # child back). Comparing or repr-ing them with the dataclass defaults
    # would recurse through that cycle forever, so both are excluded.
    parent: LATSNode | None = field(compare=False, repr=False)
    branch_name: str | None = None
    branch: Any | None = field(default=None, compare=False, repr=False)
    visits: int = 0
    total_value: float = 0.0
    value: float | None = None
    children: list[LATSNode] = field(default_factory=list, compare=False, repr=False)

    @property
    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def trajectory_search(self, vector, k: int = 10) -> list[tuple[Any, Any]]:
        """Search this node's own branch plus every ancestor's, merged.

        Each node's branch only ever holds the one action/observation pair it
        contributed itself; the shared long-term memory (main) plus every
        ancestor along the path is what reconstructs full trajectory context.
        Results are deduplicated by record id (main-visible records, e.g.
        seeded runbooks, would otherwise appear once per ancestor searched)
        and merged by distance.
        """
        best: dict[Any, tuple[Any, Any]] = {}
        node: LATSNode | None = self
        while node is not None:
            if node.branch is not None:
                for result, record in node.branch.search(vector, k=k):
                    current = best.get(record.id)
                    if current is None or result.distance < current[0].distance:
                        best[record.id] = (result, record)
            node = node.parent
        return sorted(best.values(), key=lambda pair: pair[0].distance)[:k]


@dataclass
class LATSStats:
    expanded_nodes: int = 0
    scored_leaves: int = 0
    branch_deltas_created: int = 0
    branch_deltas_discarded: int = 0
    peak_live_branches: int = 0
    branch_create_ms: float = 0.0
    branch_discard_ms: float = 0.0


def _sidecar_path(path: str | Path) -> Path:
    return Path(str(path) + ".lats.json")


async def _call_maybe_async(fn: Callable[..., Any], *args: Any) -> Any:
    """Call a user-supplied callable, awaiting it if it is itself async.

    This is what makes ``execute``/``evaluate`` genuinely pluggable tool/LLM
    interfaces: pass a plain sync function (offloaded to a thread so it never
    blocks the event loop) or a real async client method (e.g. an LLM SDK's
    ``.acomplete``) interchangeably.
    """
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args)
    return await asyncio.to_thread(fn, *args)


class PersistentLATS:
    """MCTS where every node's trajectory lives in its own AgentMemory branch.

    ``execute`` returns the observation recorded for a newly expanded action.
    ``evaluate`` receives the live branch and complete node, and should return
    a value in the inclusive range ``[0, 1]``. It can retrieve semantic
    evidence from the branch (including ``node.trajectory_search`` for full
    ancestor context), call a judge, or inspect tool observations. Both may
    be plain sync callables or async ones -- see ``_call_maybe_async``.
    """

    def __init__(
        self,
        memory: AgentMemory,
        *,
        actions: Sequence[LATSAction],
        embed: Embed,
        execute: Execute,
        evaluate: Evaluate,
        max_depth: int,
        max_iterations: int,
        exploration: float = 1.4,
        run_id: str | None = None,
    ) -> None:
        if not actions:
            raise ValueError("actions must not be empty")
        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        names = [action.name for action in actions]
        if len(names) != len(set(names)):
            raise ValueError("action names must be unique")
        self.memory = memory
        self.actions = tuple(actions)
        self._actions = {action.name: action for action in actions}
        self.embed = embed
        self.execute = execute
        self.evaluate = evaluate
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.exploration = exploration
        self.run_id = run_id or f"lats-{uuid.uuid4().hex[:10]}"
        self.root = LATSNode(0, (), (), None)
        self.nodes: list[LATSNode] = [self.root]
        self.stats = LATSStats()
        # Persistent across calls (and across save/resume): run()/run_async()
        # can be called more than once to extend a search's budget, and each
        # call must still be able to return the best leaf found by any
        # earlier call, not just ones scored during this particular call.
        self.leaves: list[LATSNode] = []

    # -- tree policy -------------------------------------------------------

    def _available(self, node: LATSNode) -> list[LATSAction]:
        expanded = {child.path[-1] for child in node.children}
        return [
            action
            for action in self.actions
            if action.name not in node.path and action.name not in expanded
        ]

    def _terminal(self, node: LATSNode) -> bool:
        return len(node.path) >= self.max_depth or not [
            action for action in self.actions if action.name not in node.path
        ]

    def _select(self) -> LATSNode:
        node = self.root
        while not self._terminal(node):
            if self._available(node):
                return node
            parent_visits = max(node.visits, 1)
            node = max(
                node.children,
                key=lambda child: (
                    child.mean_value
                    + self.exploration * math.sqrt(math.log(parent_visits) / max(child.visits, 1))
                ),
            )
        return node

    def _branch_name(self, node_id: int) -> str:
        return f"{self.run_id}-node-{node_id}"

    # -- synchronous expansion ---------------------------------------------

    def _expand(self, parent: LATSNode) -> LATSNode:
        action = self._available(parent)[0]
        node = LATSNode(len(self.nodes), (*parent.path, action.name), parent.observations, parent)
        parent.children.append(node)
        self.nodes.append(node)
        observation = self.execute(node.path)
        node.observations = (*parent.observations, observation)
        self._materialize_branch(node, action, observation)
        return node

    def _materialize_branch(self, node: LATSNode, action: LATSAction, observation: str) -> None:
        """Create the node's own O(1) delta: just its action and observation.

        This is the crux of "delta reuse rather than rematerialization": a
        naive branch-per-trajectory design would rewrite the whole path into
        every new node's branch, so write cost grows with depth. Writing only
        the node's own step, and reconstructing context by walking the
        parent chain at read time (``LATSNode.trajectory_search``), keeps
        write cost constant per node however deep or wide the tree grows.
        """
        started = time.perf_counter()
        branch_name = self._branch_name(node.node_id)
        branch = self.memory.branch(branch_name)
        position = len(node.parent.path) if node.parent is not None else 0
        branch.add(
            f"{branch_name}-action",
            self.embed(action.text),
            kind="action",
            action=action.name,
            position=position,
            text=action.text,
        )
        branch.add(
            f"{branch_name}-observation",
            self.embed(observation),
            kind="observation",
            action=action.name,
            position=position,
            text=observation,
        )
        node.branch_name = branch_name
        node.branch = branch
        self.stats.expanded_nodes += 1
        self.stats.branch_deltas_created += 1
        self.stats.branch_create_ms += (time.perf_counter() - started) * 1_000
        self.stats.peak_live_branches = max(
            self.stats.peak_live_branches, len(tuple(self.memory.branches()))
        )

    def _score(self, node: LATSNode) -> float:
        # A fully-expanded terminal node can be re-selected by UCT for the
        # rest of the search budget; caching avoids re-running evaluate()
        # (and its branch search) once a leaf already has a value.
        if node.value is not None:
            return node.value
        if node.branch is None:
            raise RuntimeError("a live branch is required to score a trajectory")
        value = float(self.evaluate(node.branch, node))
        if not 0.0 <= value <= 1.0:
            raise ValueError("evaluate must return a value between 0 and 1")
        node.value = value
        self.stats.scored_leaves += 1
        return value

    @staticmethod
    def _backpropagate(node: LATSNode, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent

    def run(self) -> LATSNode:
        """Run MCTS for the configured budget and return the best scored leaf.

        Calling this again (optionally after raising ``max_iterations`` first)
        spends more budget on the same tree and still considers every leaf
        ever scored, not just ones found during this particular call.
        """
        for _ in range(self.max_iterations):
            selected = self._select()
            node = selected if self._terminal(selected) else self._expand(selected)
            if self._terminal(node):
                first_visit = node.value is None
                value = self._score(node)
                self._backpropagate(node, value)
                if first_visit:
                    self.leaves.append(node)
        if not self.leaves:
            raise RuntimeError("search budget ended before reaching a complete trajectory")
        return max(self.leaves, key=lambda node: (node.value or 0.0, node.visits))

    # -- concurrent rollout execution ---------------------------------------

    async def _materialize_branch_async(self, node: LATSNode, action: LATSAction) -> None:
        started = time.perf_counter()
        observation = await _call_maybe_async(self.execute, node.path)
        node.observations = (*node.parent.observations, observation)  # type: ignore[union-attr]
        branch_name = self._branch_name(node.node_id)
        branch = await asyncio.to_thread(self.memory.branch, branch_name)
        position = len(node.parent.path)  # type: ignore[union-attr]
        await asyncio.to_thread(
            branch.add,
            f"{branch_name}-action",
            self.embed(action.text),
            kind="action",
            action=action.name,
            position=position,
            text=action.text,
        )
        await asyncio.to_thread(
            branch.add,
            f"{branch_name}-observation",
            self.embed(observation),
            kind="observation",
            action=action.name,
            position=position,
            text=observation,
        )
        node.branch_name = branch_name
        node.branch = branch
        # Plain attribute writes below run on the event loop thread when this
        # coroutine resumes -- asyncio never runs two coroutines' bodies at
        # once, so unlike the awaited calls above, these need no lock.
        self.stats.expanded_nodes += 1
        self.stats.branch_deltas_created += 1
        self.stats.branch_create_ms += (time.perf_counter() - started) * 1_000
        self.stats.peak_live_branches = max(
            self.stats.peak_live_branches, len(tuple(self.memory.branches()))
        )

    async def _score_async(self, node: LATSNode) -> float:
        if node.value is not None:
            return node.value
        if node.branch is None:
            raise RuntimeError("a live branch is required to score a trajectory")
        value = float(await _call_maybe_async(self.evaluate, node.branch, node))
        if not 0.0 <= value <= 1.0:
            raise ValueError("evaluate must return a value between 0 and 1")
        node.value = value
        self.stats.scored_leaves += 1
        return value

    def _reserve(self, selected: LATSNode) -> tuple[LATSNode, LATSAction]:
        action = self._available(selected)[0]
        node = LATSNode(
            len(self.nodes), (*selected.path, action.name), selected.observations, selected
        )
        selected.children.append(node)
        self.nodes.append(node)
        return node, action

    def _select_for_batch(self, avoid: frozenset[int]) -> LATSNode | None:
        """Same top-down UCT descent as ``_select()``, but able to route
        around nodes another reservation in this same batch wave is already
        sitting on. ``_select()`` is a pure function of tree state, so
        simply retrying it while nothing has changed always lands on the
        same blocked node; comparing UCT scores across *different* parents
        to pick some other node globally would be unsound instead (UCT only
        means what it means when comparing true siblings under one shared
        parent's visit count). So this walks best-child-first like
        ``_select()`` normally does, but backtracks to the next-best sibling
        on a dead end (every descendant of the best child is itself blocked
        or avoided) instead of giving up -- a single greedy path can walk
        into a subtree that's entirely blocked while a merely lower-scored
        sibling subtree sits untouched. With ``avoid`` empty and nothing
        pending this wave, every node is reachable on the first try and this
        behaves exactly like ``_select()``.

        ``avoid`` holds terminal nodes already claimed earlier in the
        current wave, so the same one isn't claimed twice. A node reserved
        but not yet materialized (``branch is None``, not root) is always
        excluded, since its own observation isn't final yet and neither it
        nor a child claimed under it is safe to select right now -- and it
        never has children of its own yet regardless, having just been
        created this wave, so excluding it costs nothing to explore.
        """

        def score(node: LATSNode, parent_visits: int) -> float:
            return node.mean_value + self.exploration * math.sqrt(
                math.log(parent_visits) / max(node.visits, 1)
            )

        def visit(node: LATSNode) -> LATSNode | None:
            reachable = node is self.root or node.branch is not None
            if reachable and node.node_id not in avoid and self._available(node):
                return node
            if self._terminal(node):
                return node if reachable and node.node_id not in avoid else None
            # Try children best-UCT-first, same as the plain descent, but on
            # a dead end (every descendant down that child is itself blocked
            # or avoided) backtrack and try the next-best sibling instead of
            # giving up -- a single greedy path can walk into a subtree
            # that's entirely blocked while a perfectly good, merely
            # lower-scored, sibling subtree sits untouched.
            parent_visits = max(node.visits, 1)
            candidates = sorted(
                (c for c in node.children if c.node_id not in avoid and c.branch is not None),
                key=lambda child: score(child, parent_visits),
                reverse=True,
            )
            for child in candidates:
                found = visit(child)
                if found is not None:
                    return found
            return None

        return visit(self.root)

    async def run_async(self, *, concurrency: int = 8) -> LATSNode:
        """Run MCTS with up to ``concurrency`` rollouts executing at once.

        Real tool calls and LLM judges are I/O-bound; running them
        concurrently hides that latency instead of paying for it serially.
        Each of the ``max_iterations`` budget units is still exactly one
        selection (matching ``run()``'s semantics one-for-one); this only
        batches the expensive "materialize the picked node" step.

        Selecting and reserving a node's *action* is synchronous and cheap
        (pure Python), so a whole batch is reserved up front -- this is what
        lets later selections within the same batch see earlier reservations
        and avoid claiming the same action twice: reserving a child mutates
        its parent's ``children``, which ``_available()`` reads live, so a
        parent with several open actions (root, most often) is naturally
        still eligible for the rest of the wave. A terminal node isn't
        mutated by being picked, though, so it goes in ``claimed_terminals``
        to stop the wave from claiming the same one repeatedly instead of
        moving on to the rest of the batch.
        """
        remaining = self.max_iterations
        while remaining > 0:
            batch_size = min(concurrency, remaining)
            immediate: list[LATSNode] = []
            pending: list[tuple[LATSNode, LATSAction]] = []
            claimed_terminals: set[int] = set()
            count = 0
            while count < batch_size:
                selected = self._select_for_batch(frozenset(claimed_terminals))
                if selected is None:
                    break
                count += 1
                if self._terminal(selected):
                    claimed_terminals.add(selected.node_id)
                    immediate.append(selected)
                else:
                    pending.append(self._reserve(selected))
            if count == 0:
                break
            remaining -= count
            if pending:
                await asyncio.gather(
                    *(self._materialize_branch_async(node, action) for node, action in pending)
                )
                for node, _action in pending:
                    if self._terminal(node):
                        immediate.append(node)
            for node in immediate:
                first_visit = node.value is None
                value = await self._score_async(node)
                self._backpropagate(node, value)
                if first_visit:
                    self.leaves.append(node)
        if not self.leaves:
            raise RuntimeError("search budget ended before reaching a complete trajectory")
        return max(self.leaves, key=lambda node: (node.value or 0.0, node.visits))

    # -- publication ---------------------------------------------------------

    def _ancestor_chain(self, node: LATSNode) -> list[LATSNode]:
        chain = []
        while node is not None:
            chain.append(node)
            node = node.parent
        return list(reversed(chain))

    def publish(self, winner: LATSNode) -> int:
        """Atomically merge the winning trajectory's deltas, discard the rest.

        Every node along the winning path owns its own branch (one action and
        observation each), so publishing the trajectory means merging each of
        those, not just the leaf's.
        """
        if winner.branch is None:
            raise ValueError("winner does not have a live branch")
        promoted = 0
        for node in self._ancestor_chain(winner):
            if node.branch is not None:
                promoted += node.branch.merge()
                node.branch = None
                node.branch_name = None
        for node in self.nodes:
            if node.branch is not None:
                self._discard(node)
        return promoted

    def discard_all(self) -> None:
        """Release all trajectory deltas without publishing any result."""
        for node in self.nodes:
            if node.branch is not None:
                self._discard(node)

    def _discard(self, node: LATSNode) -> None:
        started = time.perf_counter()
        node.branch.discard()
        self.stats.branch_discard_ms += (time.perf_counter() - started) * 1_000
        self.stats.branch_deltas_discarded += 1
        node.branch_name = None
        node.branch = None

    # -- checkpoint / resume --------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Checkpoint the AgentMemory (main plus every open branch) and the
        tree metadata needed to resume the search in a later process.

        ``embed``/``execute``/``evaluate`` are callables and are not
        serialized; the caller supplies them again to :meth:`resume`.
        """
        self.memory.save(path)
        payload = {
            "run_id": self.run_id,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "exploration": self.exploration,
            "actions": [{"name": a.name, "text": a.text} for a in self.actions],
            "stats": asdict(self.stats),
            "nodes": [
                {
                    "node_id": node.node_id,
                    "path": list(node.path),
                    "observations": list(node.observations),
                    "parent_id": node.parent.node_id if node.parent is not None else None,
                    "branch_name": node.branch_name,
                    "visits": node.visits,
                    "total_value": node.total_value,
                    "value": node.value,
                }
                for node in self.nodes
            ],
        }
        _sidecar_path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def resume(
        cls,
        path: str | Path,
        *,
        embed: Embed,
        execute: Execute,
        evaluate: Evaluate,
    ) -> PersistentLATS:
        """Reload a checkpoint written by :meth:`save` and reattach its branches.

        Any node whose branch had not yet been merged or discarded at save
        time comes back with a live, immediately searchable branch -- the
        search can keep calling :meth:`run`/:meth:`run_async` to spend more
        of its budget, or :meth:`publish`/:meth:`discard_all` to conclude it.
        """
        memory = AgentMemory.load(path)
        payload = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
        actions = tuple(LATSAction(item["name"], item["text"]) for item in payload["actions"])
        search = cls(
            memory,
            actions=actions,
            embed=embed,
            execute=execute,
            evaluate=evaluate,
            max_depth=payload["max_depth"],
            max_iterations=payload["max_iterations"],
            exploration=payload["exploration"],
            run_id=payload["run_id"],
        )
        for key, value in payload["stats"].items():
            setattr(search.stats, key, value)
        by_id: dict[int, LATSNode] = {0: search.root}
        for item in payload["nodes"]:
            if item["node_id"] == 0:
                # The root always exists already (created by __init__); it
                # just needs its accumulated visit/value stats restored too.
                search.root.visits = item["visits"]
                search.root.total_value = item["total_value"]
                search.root.value = item["value"]
                continue
            parent = by_id[item["parent_id"]]
            node = LATSNode(
                item["node_id"],
                tuple(item["path"]),
                tuple(item["observations"]),
                parent,
            )
            node.branch_name = item["branch_name"]
            node.visits = item["visits"]
            node.total_value = item["total_value"]
            node.value = item["value"]
            if node.branch_name is not None:
                node.branch = memory.get_branch(node.branch_name)
            parent.children.append(node)
            by_id[node.node_id] = node
        search.nodes = [by_id[node_id] for node_id in sorted(by_id)]
        # Every scored node is, by construction, a terminal leaf that was
        # scored at some point (`_score`/`_score_async` are the only place
        # `.value` is ever set) -- rebuilding `leaves` from that, rather than
        # persisting it separately, means run()/run_async() after resume
        # still consider leaves found before the checkpoint, not just new ones.
        search.leaves = [node for node in search.nodes if node.value is not None]
        return search
