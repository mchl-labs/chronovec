"""The shared operation trace.

Every engine replays byte-identical operations in the same order, so a
difference in the report is the engine's and not the workload's. Victim
positions are carried rather than looked up by value, which keeps replay linear
in the number of replacements instead of quadratic in the live set.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Epoch:
    positions: np.ndarray   # where in the live set each replacement lands
    retired_ids: np.ndarray
    new_ids: np.ndarray
    source_rows: np.ndarray


@dataclass
class Trace:
    initial_ids: np.ndarray
    initial_rows: np.ndarray
    epochs: list[Epoch] = field(default_factory=list)

    @property
    def live(self) -> int:
        return int(self.initial_ids.shape[0])

    @property
    def replacements_per_epoch(self) -> int:
        return int(self.epochs[0].positions.shape[0]) if self.epochs else 0


def build_trace(pool: int, live: int, epochs: int, turnover: float,
                seed: int = 42) -> Trace:
    """Replace `turnover` of the live set each epoch, drawing from `pool` rows."""
    if live > pool:
        raise ValueError(f"live set {live} exceeds the pool of {pool} vectors")
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool)
    trace = Trace(initial_ids=np.arange(live, dtype=np.int64),
                  initial_rows=order[:live].copy())
    live_ids = trace.initial_ids.copy()
    cursor, next_id = live, live
    per_epoch = max(1, int(live * turnover))
    for _ in range(epochs):
        positions = rng.choice(live, size=per_epoch, replace=False)
        rows = np.empty(per_epoch, dtype=np.int64)
        for i in range(per_epoch):
            if cursor >= pool:
                cursor = 0          # wrap the pool rather than run dry
            rows[i] = order[cursor]
            cursor += 1
        new_ids = np.arange(next_id, next_id + per_epoch, dtype=np.int64)
        next_id += per_epoch
        trace.epochs.append(Epoch(positions.copy(), live_ids[positions].copy(),
                                  new_ids, rows))
        live_ids[positions] = new_ids
    return trace


@dataclass
class OpsPhase:
    """One homogeneous burst of a single operation kind.

    Kinds are kept separate rather than mixed so a report can say *which*
    operation an engine is slow at. A mixed trace averages that away.
    """
    kind: str                       # insert | update | delete | replace
    ids: np.ndarray                 # ids the operation acts on
    source_rows: np.ndarray | None  # pool rows supplying vectors, None for delete
    retired_ids: np.ndarray | None  # replace only: the ids being retired


@dataclass
class OpsTrace:
    initial_ids: np.ndarray
    initial_rows: np.ndarray
    phases: list[OpsPhase] = field(default_factory=list)

    @property
    def live(self) -> int:
        return int(self.initial_ids.shape[0])


def build_ops_trace(pool: int, live: int, batch: int, cycles: int = 3,
                    seed: int = 42) -> OpsTrace:
    """A trace that exercises every mutation an engine can be asked for.

    The replace-only trace built by :func:`build_trace` measures one shape:
    retire an id, add a different id, live set constant. Real workloads also
    grow, shrink, and rewrite a vector under an id that already exists. Update
    in place is the interesting one -- it is the operation most engines have to
    emulate as delete plus insert, and the only one where an MVCC index can keep
    the old version readable.

    Each cycle runs insert, update, delete, replace in that order, so the live
    set returns to `live` at the end of every cycle and phases stay comparable
    across cycles.
    """
    if live > pool:
        raise ValueError(f"live set {live} exceeds the pool of {pool} vectors")
    if batch > live:
        raise ValueError(f"batch {batch} exceeds the live set {live}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool)
    trace = OpsTrace(initial_ids=np.arange(live, dtype=np.int64),
                     initial_rows=order[:live].copy())
    live_ids = trace.initial_ids.copy()
    cursor, next_id = live, live

    def draw(count: int) -> np.ndarray:
        nonlocal cursor
        rows = np.empty(count, dtype=np.int64)
        for position in range(count):
            if cursor >= pool:
                cursor = 0          # wrap the pool rather than run dry
            rows[position] = order[cursor]
            cursor += 1
        return rows

    for _ in range(cycles):
        # Grow.
        ids = np.arange(next_id, next_id + batch, dtype=np.int64)
        next_id += batch
        trace.phases.append(OpsPhase("insert", ids, draw(batch), None))
        live_ids = np.concatenate([live_ids, ids])

        # Rewrite the vector under an id that already exists.
        chosen = rng.choice(live_ids.shape[0], size=batch, replace=False)
        trace.phases.append(
            OpsPhase("update", live_ids[chosen].copy(), draw(batch), None))

        # Shrink.
        chosen = rng.choice(live_ids.shape[0], size=batch, replace=False)
        victims = live_ids[chosen].copy()
        trace.phases.append(OpsPhase("delete", victims, None, None))
        live_ids = np.delete(live_ids, chosen)

        # Retire and add, the shape the original trace measures.
        chosen = rng.choice(live_ids.shape[0], size=batch, replace=False)
        retired = live_ids[chosen].copy()
        fresh = np.arange(next_id, next_id + batch, dtype=np.int64)
        next_id += batch
        trace.phases.append(OpsPhase("replace", fresh, draw(batch), retired))
        live_ids[chosen] = fresh
    return trace


def build_mixed_trace(pool: int, live: int, batch: int, cycles: int = 3,
                      seed: int = 42) -> OpsTrace:
    """Interleave the operation kinds instead of running them in bursts.

    :func:`build_ops_trace` runs each kind as a homogeneous burst, which is what
    makes per-operation timing readable. Real traffic is not sorted by
    operation, and an engine can be fine on each kind alone and bad on the
    mixture: a delete that tombstones cheaply and an insert that reuses
    tombstoned slots interact, and a batching layer that assumes homogeneous
    runs stops batching at all.

    Generated against the live set as it actually evolves rather than by
    slicing an ordered trace. Slicing looks equivalent and is not: the ordered
    trace picks its update targets after the whole insert burst has landed, so
    an interleaved update slice would reference ids a later insert slice has
    not added yet.

    Each cycle issues the same total work as the ops trace, so the two are
    directly comparable.
    """
    if live > pool:
        raise ValueError(f"live set {live} exceeds the pool of {pool} vectors")
    if batch < 8:
        raise ValueError("mixed trace needs a batch of at least 8")
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool)
    trace = OpsTrace(initial_ids=np.arange(live, dtype=np.int64),
                     initial_rows=order[:live].copy())
    live_ids = trace.initial_ids.copy()
    cursor, next_id = live, live
    slices = 8
    per_slice = max(1, batch // slices)

    def draw(count: int) -> np.ndarray:
        nonlocal cursor
        rows = np.empty(count, dtype=np.int64)
        for position in range(count):
            if cursor >= pool:
                cursor = 0
            rows[position] = order[cursor]
            cursor += 1
        return rows

    for _ in range(cycles):
        for _ in range(slices):
            ids = np.arange(next_id, next_id + per_slice, dtype=np.int64)
            next_id += per_slice
            trace.phases.append(OpsPhase("insert", ids, draw(per_slice), None))
            live_ids = np.concatenate([live_ids, ids])

            chosen = rng.choice(live_ids.shape[0], size=per_slice, replace=False)
            trace.phases.append(
                OpsPhase("update", live_ids[chosen].copy(), draw(per_slice), None))

            chosen = rng.choice(live_ids.shape[0], size=per_slice, replace=False)
            trace.phases.append(
                OpsPhase("delete", live_ids[chosen].copy(), None, None))
            live_ids = np.delete(live_ids, chosen)

            chosen = rng.choice(live_ids.shape[0], size=per_slice, replace=False)
            retired = live_ids[chosen].copy()
            fresh = np.arange(next_id, next_id + per_slice, dtype=np.int64)
            next_id += per_slice
            trace.phases.append(
                OpsPhase("replace", fresh, draw(per_slice), retired))
            live_ids[chosen] = fresh
    return trace
