"""The epoch loop: replay the trace, measure what changes."""

from __future__ import annotations

import platform
import time
from typing import Any, Callable, Sequence

import numpy as np

from .engines import ENGINES, Engine
from .trace import OpsTrace, Trace, build_trace

SCHEMA_VERSION = 1


def exact_truth(base: np.ndarray, ids: np.ndarray, queries: np.ndarray, k: int,
                metric: str, chunk: int = 256) -> np.ndarray:
    """Top-k ids against the *current* live set, recomputed every epoch."""
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        block = queries[start:stop]
        if metric == "cosine":
            distances = 1.0 - block @ base.T
        else:
            distances = (np.sum(block * block, axis=1)[:, None]
                         - 2.0 * (block @ base.T)
                         + np.sum(base * base, axis=1)[None, :])
        take = min(k, base.shape[0])
        cols = np.argpartition(distances, take - 1, axis=1)[:, :take]
        rows = np.arange(stop - start)[:, None]
        out[start:stop] = ids[cols[rows, np.argsort(distances[rows, cols], axis=1)]]
    return out


def _latency(samples: list[float]) -> dict[str, float]:
    a = np.asarray(samples)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "qps": float(len(a) / a.sum()) if a.sum() else 0.0}


def run_engine(factory: Callable[..., Engine], base: np.ndarray, trace: Trace,
               queries: np.ndarray, k: int, metric: str,
               options: dict[str, Any], mode: str = "bulk") -> dict[str, Any]:
    engine = factory(dim=base.shape[1], metric=metric, live=trace.live, **options)
    if mode == "interleaved" and not getattr(engine, "supports_interleaved", True):
        engine.close()
        raise RuntimeError(
            f"{engine.name} must rebuild to express a turnover, so interleaved "
            f"single-operation mutation is not meaningful for it")
    live_ids = trace.initial_ids.copy()
    live_rows = trace.initial_rows.copy()
    epochs: list[dict[str, Any]] = []

    def measure(number: int, write_s: float, operations: int) -> dict[str, Any]:
        start = time.perf_counter()
        freed = engine.maintain()
        maintenance_s = time.perf_counter() - start
        truth = exact_truth(base[live_rows], live_ids, queries, k, metric)
        latencies, hits = [], 0.0
        for index in range(queries.shape[0]):
            start = time.perf_counter()
            found = engine.search(queries[index], k)
            latencies.append((time.perf_counter() - start) * 1000.0)
            hits += len(set(found) & set(truth[index].tolist())) / k
        return {"epoch": number, "live_vectors": int(live_ids.shape[0]),
                "write_operations": operations, "write_s": write_s,
                "write_ops_per_s": operations / write_s if write_s else 0.0,
                "maintenance_s": maintenance_s, "reclaimed": freed,
                "total_s": write_s + maintenance_s,
                "recall_at_k": hits / queries.shape[0],
                "latency": _latency(latencies), "stats": engine.stats()}

    start = time.perf_counter()
    engine.build(base[trace.initial_rows], trace.initial_ids)
    epochs.append(measure(0, time.perf_counter() - start, trace.live))

    for number, epoch in enumerate(trace.epochs, start=1):
        next_ids = live_ids.copy()
        next_rows = live_rows.copy()
        next_ids[epoch.positions] = epoch.new_ids
        next_rows[epoch.positions] = epoch.source_rows
        start = time.perf_counter()
        if mode == "interleaved":
            source = base[epoch.source_rows]
            for position in range(epoch.retired_ids.shape[0]):
                engine.replace_one(int(epoch.retired_ids[position]),
                                   int(epoch.new_ids[position]), source[position])
        else:
            engine.apply(epoch.retired_ids, epoch.new_ids, base[epoch.source_rows],
                         next_ids, base[next_rows])
        write_s = time.perf_counter() - start
        live_ids, live_rows = next_ids, next_rows
        epochs.append(measure(number, write_s, 2 * epoch.positions.shape[0]))

    result = {"engine": engine.name, "can_delete": engine.can_delete,
              "needs_rebuild": engine.needs_rebuild, "mode": mode,
              "epochs": epochs}
    engine.close()
    return result


def run_benchmark(base: np.ndarray, queries: np.ndarray, trace: Trace, *,
                  engines: list[str], k: int = 10, metric: str = "l2",
                  options: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    options = options or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__},
        "workload": {"live_vectors": trace.live, "epochs": len(trace.epochs),
                     "replacements_per_epoch": trace.replacements_per_epoch,
                     "dimensions": int(base.shape[1]), "queries": int(queries.shape[0]),
                     "k": k, "metric": metric,
                     "note": "one operation trace replayed against every engine"},
        "engines": {},
    }
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine(
                ENGINES[name], base, trace, queries, k, metric, options.get(name, {}))
        except Exception as error:  # a missing or broken engine must not stop the run
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report


def _next_live(live_ids: np.ndarray, live_rows: np.ndarray,
               phase: "OpsPhase") -> tuple[np.ndarray, np.ndarray]:
    """The live set a phase produces, computed outside any timed region."""
    if phase.kind == "insert":
        return (np.concatenate([live_ids, phase.ids]),
                np.concatenate([live_rows, phase.source_rows]))
    if phase.kind == "delete":
        keep = ~np.isin(live_ids, phase.ids)
        return live_ids[keep], live_rows[keep]
    next_ids, next_rows = live_ids.copy(), live_rows.copy()
    position = {int(v): i for i, v in enumerate(live_ids)}
    if phase.kind == "update":
        for one, row in zip(phase.ids, phase.source_rows):
            next_rows[position[int(one)]] = row
        return next_ids, next_rows
    for old, new, row in zip(phase.retired_ids, phase.ids, phase.source_rows):
        at = position[int(old)]
        next_ids[at], next_rows[at] = new, row
    return next_ids, next_rows


def run_engine_ops(factory: Callable[..., Engine], base: np.ndarray,
                   trace: "OpsTrace", queries: np.ndarray, k: int, metric: str,
                   options: dict[str, Any]) -> dict[str, Any]:
    """Replay a per-operation trace, timing each operation kind separately.

    :func:`run_engine` measures replacement only. This measures insert, update,
    delete and replace apart from one another, so a report can say which
    operation an engine is bad at instead of averaging it into a single number.
    """
    engine = factory(dim=base.shape[1], metric=metric, live=trace.live, **options)
    live_ids = trace.initial_ids.copy()
    live_rows = trace.initial_rows.copy()
    phases: list[dict[str, Any]] = []

    def measure(kind: str, number: int, write_s: float, operations: int,
                note: str | None = None) -> dict[str, Any]:
        # Reclamation runs after every phase and is timed apart from the write,
        # so an engine that defers cleanup indefinitely is visible as such
        # rather than simply cheap.
        start = time.perf_counter()
        freed = engine.maintain()
        maintenance_s = time.perf_counter() - start
        truth = exact_truth(base[live_rows], live_ids, queries, k, metric)
        latencies, hits = [], 0.0
        for index in range(queries.shape[0]):
            start = time.perf_counter()
            found = engine.search(queries[index], k)
            latencies.append((time.perf_counter() - start) * 1000.0)
            hits += len(set(found) & set(truth[index].tolist())) / k
        row = {"phase": number, "kind": kind,
               "live_vectors": int(live_ids.shape[0]),
               "write_operations": operations, "write_s": write_s,
               "write_ops_per_s": operations / write_s if write_s else 0.0,
               "maintenance_s": maintenance_s, "reclaimed": freed,
               "total_s": write_s + maintenance_s,
               "recall_at_k": hits / queries.shape[0],
               "latency": _latency(latencies), "stats": engine.stats()}
        if note:
            row["note"] = note
        return row

    start = time.perf_counter()
    engine.build(base[trace.initial_rows], trace.initial_ids)
    phases.append(measure("build", 0, time.perf_counter() - start, trace.live))

    for number, phase in enumerate(trace.phases, start=1):
        note = None
        # Worked out before timing, so bookkeeping is never inside the region.
        next_ids, next_rows = _next_live(live_ids, live_rows, phase)

        # An engine without a native operation is not skipped. Skipping would
        # leave its contents inconsistent with the trace every later phase
        # assumes, and it would also flatter the engine: a user holding an
        # immutable index has to rebuild, and that is the real cost. So the
        # fallback is a rebuild of the resulting live set, timed like any other
        # write and labelled as such.
        try:
            start = time.perf_counter()
            if phase.kind == "insert":
                if not engine.can_grow:
                    raise NotImplementedError("fixed size")
                engine.insert_many(phase.ids, base[phase.source_rows])
            elif phase.kind == "update":
                engine.update_many(phase.ids, base[phase.source_rows])
                if not engine.can_update_in_place:
                    note = "emulated as delete then insert"
            elif phase.kind == "delete":
                engine.delete_many(phase.ids)
            else:
                engine.apply(phase.retired_ids, phase.ids,
                             base[phase.source_rows], next_ids, base[next_rows])
            write_s = time.perf_counter() - start
        except NotImplementedError:
            start = time.perf_counter()
            engine.build(base[next_rows], next_ids)
            write_s = time.perf_counter() - start
            note = f"rebuilt: no native {phase.kind}"

        live_ids, live_rows = next_ids, next_rows
        phases.append(measure(phase.kind, number, write_s,
                              int(phase.ids.shape[0]), note))

    result = {"engine": engine.name, "mode": "ops",
              "capabilities": {"can_delete": engine.can_delete,
                               "can_grow": engine.can_grow,
                               "can_update_in_place": engine.can_update_in_place,
                               "needs_rebuild": engine.needs_rebuild},
              "phases": phases}
    engine.close()
    return result


def run_ops_benchmark(base: np.ndarray, queries: np.ndarray, trace: "OpsTrace", *,
                      engines: list[str], k: int = 10, metric: str = "l2",
                      options: dict[str, dict[str, Any]] | None = None
                      ) -> dict[str, Any]:
    options = options or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__},
        "workload": {"live_vectors": trace.live, "phases": len(trace.phases),
                     "dimensions": int(base.shape[1]),
                     "queries": int(queries.shape[0]), "k": k, "metric": metric,
                     "note": "insert, update, delete and replace timed apart"},
        "engines": {},
    }
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine_ops(
                ENGINES[name], base, trace, queries, k, metric, options.get(name, {}))
        except Exception as error:
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report


def run_engine_concurrent(factory: Callable[..., Engine], base: np.ndarray,
                          trace: "OpsTrace", queries: np.ndarray, k: int,
                          metric: str, options: dict[str, Any],
                          readers: int = 2) -> dict[str, Any]:
    """Mutate on one thread while others search, and see what reads cost.

    Every other mode stops the world to measure: writes finish, then queries
    run. That is the one condition under which a lock-free reader and a
    lock-taking one are indistinguishable, and it is not the condition an agent
    memory or a CDC pipeline ever runs in.

    Recall is not well posed here -- the corpus changes underneath the query, so
    there is no single correct answer to compare against. What is well posed is
    an *anchor*: an id the trace never touches. Its vector is in the index
    before the run starts and is never rewritten or removed, so a search for it
    must find it at every instant, whatever else is being written. A reader that
    misses an anchor saw a torn index; a reader that returns an id the trace
    never issued saw memory that was being rewritten under it.
    """
    engine = factory(dim=base.shape[1], metric=metric, live=trace.live, **options)
    if not engine.supports_concurrent_reads:
        engine.close()
        raise RuntimeError(
            f"{engine.name} does not claim searches are safe during a write; "
            f"running it here would measure a crash, not a latency")

    touched: set[int] = set()
    for phase in trace.phases:
        touched.update(int(x) for x in phase.ids)
        if phase.retired_ids is not None:
            touched.update(int(x) for x in phase.retired_ids)
    known = set(int(x) for x in trace.initial_ids) | touched
    anchors = [i for i, one in enumerate(trace.initial_ids)
               if int(one) not in touched]
    if len(anchors) < 32:
        engine.close()
        raise RuntimeError("trace leaves too few untouched ids to anchor on")
    anchor_positions = np.array(anchors[:512])
    anchor_ids = trace.initial_ids[anchor_positions]
    anchor_vectors = base[trace.initial_rows[anchor_positions]]

    engine.build(base[trace.initial_rows], trace.initial_ids)

    # An anchor can also be missed simply because the search is approximate.
    # Measuring the miss rate on a quiescent index first separates "this engine
    # does not return every anchor" from "this engine loses anchors while being
    # written to", which are different claims.
    def quiescent_miss_rate() -> float:
        return sum(1 for position in range(len(anchor_ids))
                   if int(anchor_ids[position]) not in engine.search(
                       anchor_vectors[position], k)) / len(anchor_ids)

    quiescent_before = quiescent_miss_rate()

    import threading
    stop = threading.Event()
    reports: list[dict[str, Any]] = []

    def reader(seed: int) -> None:
        rng = np.random.default_rng(seed)
        latencies: list[float] = []
        missed = unknown = served = 0
        while not stop.is_set():
            pick = int(rng.integers(len(anchor_ids)))
            start = time.perf_counter()
            try:
                found = engine.search(anchor_vectors[pick], k)
            except Exception:
                # A read that cannot complete during a write is itself the
                # result; it is counted, not raised.
                missed += 1
                continue
            latencies.append((time.perf_counter() - start) * 1000.0)
            served += 1
            if int(anchor_ids[pick]) not in found:
                missed += 1
            unknown += sum(1 for one in found if one not in known)
        reports.append({"searches": served, "anchor_misses": missed,
                        "unknown_ids": unknown,
                        "latency": _latency(latencies) if latencies else {}})

    threads = [threading.Thread(target=reader, args=(100 + n,), daemon=True)
               for n in range(readers)]
    # The window readers actually ran for, wall clock. Dividing their count by
    # the sum of the write phases instead was wrong and flattered us: readers
    # also run through maintenance and the bookkeeping between phases, so the
    # denominator was shorter than the window they served in.
    window_started = time.perf_counter()
    for thread in threads:
        thread.start()

    live_ids, live_rows = trace.initial_ids.copy(), trace.initial_rows.copy()
    phases: list[dict[str, Any]] = []
    try:
        for number, phase in enumerate(trace.phases, start=1):
            # The live set has to advance on every phase kind, not only on
            # replace: a later phase retires ids that an earlier insert added,
            # and tracking only replace leaves them unknown.
            next_ids, next_rows = _next_live(live_ids, live_rows, phase)
            start = time.perf_counter()
            if phase.kind == "insert":
                engine.insert_many(phase.ids, base[phase.source_rows])
            elif phase.kind == "update":
                engine.update_many(phase.ids, base[phase.source_rows])
            elif phase.kind == "delete":
                engine.delete_many(phase.ids)
            else:
                engine.apply(phase.retired_ids, phase.ids,
                             base[phase.source_rows], next_ids, base[next_rows])
            write_s = time.perf_counter() - start
            live_ids, live_rows = next_ids, next_rows
            phases.append({"phase": number, "kind": phase.kind,
                           "write_operations": int(phase.ids.shape[0]),
                           "write_s": write_s,
                           "write_ops_per_s": int(phase.ids.shape[0]) / write_s
                           if write_s else 0.0})
            engine.maintain()
    finally:
        stop.set()
        window = time.perf_counter() - window_started
        for thread in threads:
            thread.join(timeout=30)

    # Measured again at the end, because churn legitimately changes an
    # approximate index's recall. Comparing the under-load rate only against
    # the rate before the run reports concurrency damage where there is none:
    # measured 1.28% under load against 0.50% before and 1.25% after, so all
    # of the difference was the index having been rewritten, not torn.
    quiescent_after = quiescent_miss_rate()
    served = sum(r["searches"] for r in reports)
    elapsed = window
    reads = {
        "reader_threads": readers,
        "searches": served,
        "read_qps": served / elapsed if elapsed else 0.0,
        "window_s": elapsed,
        "write_s": sum(p["write_s"] for p in phases),
        "anchor_misses": sum(r["anchor_misses"] for r in reports),
        "anchor_miss_rate": (sum(r["anchor_misses"] for r in reports) / served
                             if served else 0.0),
        "quiescent_anchor_miss_rate_before": quiescent_before,
        "quiescent_anchor_miss_rate_after": quiescent_after,
        # The comparison that means something: how far the under-load rate sits
        # outside the range the index reaches on its own.
        "excess_over_quiescent": max(
            0.0, (sum(r["anchor_misses"] for r in reports) / served
                  if served else 0.0) - max(quiescent_before, quiescent_after)),
        "unknown_ids": sum(r["unknown_ids"] for r in reports),
        "latency_p50_ms": float(np.mean([r["latency"]["p50_ms"]
                                         for r in reports if r["latency"]])),
        "latency_p99_ms": float(np.max([r["latency"]["p99_ms"]
                                        for r in reports if r["latency"]])),
    }
    result = {"engine": engine.name, "mode": "concurrent",
              "reads": reads, "phases": phases}
    engine.close()
    return result


def run_concurrent_benchmark(base: np.ndarray, queries: np.ndarray,
                             trace: "OpsTrace", *, engines: list[str],
                             k: int = 10, metric: str = "l2", readers: int = 2,
                             options: dict[str, dict[str, Any]] | None = None
                             ) -> dict[str, Any]:
    options = options or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__},
        "workload": {"live_vectors": trace.live, "phases": len(trace.phases),
                     "dimensions": int(base.shape[1]), "k": k, "metric": metric,
                     "reader_threads": readers,
                     "note": "searches run continuously while one thread mutates"},
        "engines": {},
    }
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine_concurrent(
                ENGINES[name], base, trace, queries, k, metric,
                options.get(name, {}), readers)
        except Exception as error:
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report


def run_engine_snapshot(factory: Callable[..., Engine], base: np.ndarray,
                        trace: "OpsTrace", queries: np.ndarray, k: int,
                        metric: str, options: dict[str, Any]) -> dict[str, Any]:
    """Pin a point in time, write past it, and keep reading the pinned view.

    This is the capability almost nothing else has, so most of the value is in
    reporting what it costs rather than that it exists. A retained snapshot is
    precisely what stops an MVCC index reclaiming: every version the snapshot
    can still see has to be kept. So the run holds one open across the whole
    trace, watches space grow, then releases it and watches the space come
    back. An implementation that cannot reclaim after release has a leak, and
    one whose pinned reads drift has not got isolation at all.
    """
    engine = factory(dim=base.shape[1], metric=metric, live=trace.live, **options)
    if not engine.supports_snapshots:
        engine.close()
        raise RuntimeError(f"{engine.name} cannot pin a point in time to read")

    engine.build(base[trace.initial_rows], trace.initial_ids)
    pinned = engine.snapshot()
    engine.retain(pinned)
    # The answer the pinned view must keep giving, no matter what is written.
    pinned_truth = exact_truth(base[trace.initial_rows], trace.initial_ids,
                               queries, k, metric)

    def recall(finder: Callable[[np.ndarray], list[int]],
               truth: np.ndarray) -> tuple[float, dict[str, float]]:
        latencies, hits = [], 0.0
        for index in range(queries.shape[0]):
            start = time.perf_counter()
            found = finder(queries[index])
            latencies.append((time.perf_counter() - start) * 1000.0)
            hits += len(set(found) & set(truth[index].tolist())) / k
        return hits / queries.shape[0], _latency(latencies)

    live_ids, live_rows = trace.initial_ids.copy(), trace.initial_rows.copy()
    phases: list[dict[str, Any]] = []
    for number, phase in enumerate(trace.phases, start=1):
        next_ids, next_rows = _next_live(live_ids, live_rows, phase)
        if phase.kind == "insert":
            engine.insert_many(phase.ids, base[phase.source_rows])
        elif phase.kind == "update":
            engine.update_many(phase.ids, base[phase.source_rows])
        elif phase.kind == "delete":
            engine.delete_many(phase.ids)
        else:
            engine.apply(phase.retired_ids, phase.ids, base[phase.source_rows],
                         next_ids, base[next_rows])
        live_ids, live_rows = next_ids, next_rows
        freed = engine.maintain()

        pinned_recall, pinned_latency = recall(
            lambda q: engine.search_at(q, k, pinned), pinned_truth)
        current_truth = exact_truth(base[live_rows], live_ids, queries, k, metric)
        current_recall, current_latency = recall(
            lambda q: engine.search(q, k), current_truth)
        phases.append({
            "phase": number, "kind": phase.kind,
            "live_vectors": int(live_ids.shape[0]), "reclaimed": freed,
            "pinned_recall_at_k": pinned_recall,
            "pinned_p50_ms": pinned_latency["p50_ms"],
            "current_recall_at_k": current_recall,
            "current_p50_ms": current_latency["p50_ms"],
            "stats": engine.stats()})

    # Release and reclaim. What comes back here is what the snapshot was
    # costing; if nothing does, the retention was a leak.
    before = engine.stats()
    engine.retain(None)
    start = time.perf_counter()
    freed = engine.maintain()
    release_s = time.perf_counter() - start
    result = {"engine": engine.name, "mode": "snapshot",
              "pinned_at": pinned, "phases": phases,
              "release": {"reclaimed": freed, "seconds": release_s,
                          "stats_before": before, "stats_after": engine.stats()}}
    engine.close()
    return result


def run_snapshot_benchmark(base: np.ndarray, queries: np.ndarray,
                           trace: "OpsTrace", *, engines: list[str], k: int = 10,
                           metric: str = "l2",
                           options: dict[str, dict[str, Any]] | None = None
                           ) -> dict[str, Any]:
    options = options or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__},
        "workload": {"live_vectors": trace.live, "phases": len(trace.phases),
                     "dimensions": int(base.shape[1]), "k": k, "metric": metric,
                     "note": "one snapshot held open across the whole trace"},
        "engines": {},
    }
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine_snapshot(
                ENGINES[name], base, trace, queries, k, metric, options.get(name, {}))
        except Exception as error:
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report


def run_engine_equal_recall(factory: Callable[..., Engine], base: np.ndarray,
                            trace: "OpsTrace", queries: np.ndarray, k: int,
                            metric: str, options: dict[str, Any],
                            targets: tuple[float, ...] = (0.90, 0.95, 0.99)
                            ) -> dict[str, Any]:
    """Sweep each engine's search knob and compare only at matched recall.

    Every other number in this benchmark runs each engine at its own defaults,
    which compares tuning choices rather than engines: whichever is configured
    to answer an easier question looks faster. A throughput figure is only
    meaningful next to the recall it was achieved at.

    Write throughput is reported too, but it does not move with the search
    knob -- it is a property of the structure the engine built. It is listed
    beside the recall curve that structure produces, which is the honest way to
    put the two together, not multiplied into a single score.
    """
    engine = factory(dim=base.shape[1], metric=metric, live=trace.live, **options)
    start = time.perf_counter()
    engine.build(base[trace.initial_rows], trace.initial_ids)
    build_s = time.perf_counter() - start

    # An insert burst after the build, timed apart: build can train or bulk-load
    # on a path a steady-state insert never takes.
    insert_phase = next((p for p in trace.phases if p.kind == "insert"), None)
    insert_ops_per_s = None
    if insert_phase is not None:
        try:
            start = time.perf_counter()
            engine.insert_many(insert_phase.ids, base[insert_phase.source_rows])
            insert_ops_per_s = (int(insert_phase.ids.shape[0])
                                / (time.perf_counter() - start))
            live_ids = np.concatenate([trace.initial_ids, insert_phase.ids])
            live_rows = np.concatenate([trace.initial_rows,
                                        insert_phase.source_rows])
        except NotImplementedError:
            live_ids, live_rows = trace.initial_ids, trace.initial_rows
    else:
        live_ids, live_rows = trace.initial_ids, trace.initial_rows

    truth = exact_truth(base[live_rows], live_ids, queries, k, metric)
    ladder = engine.search_ladder or (None,)
    curve: list[dict[str, Any]] = []
    for setting in ladder:
        if setting is not None:
            engine.set_search_param(setting)
        # A warm-up pass, then the best of several timed ones. A single stall
        # anywhere on the machine otherwise corrupts one point of the curve, and
        # it corrupts it downward, which is indistinguishable from a real
        # regression until you notice the curve is no longer monotone. Taking
        # the best pass rejects interference without inventing throughput.
        for index in range(queries.shape[0]):
            engine.search(queries[index], k)
        best: list[float] | None = None
        hits = 0.0
        for attempt in range(3):
            latencies = []
            round_hits = 0.0
            for index in range(queries.shape[0]):
                began = time.perf_counter()
                found = engine.search(queries[index], k)
                latencies.append(time.perf_counter() - began)
                round_hits += len(set(found) & set(truth[index].tolist())) / k
            if best is None or sum(latencies) < sum(best):
                best = latencies
            hits = round_hits
        latencies = best or []
        total = sum(latencies)
        curve.append({"setting": setting, "recall_at_k": hits / queries.shape[0],
                      "qps": len(latencies) / total if total else 0.0,
                      "p50_ms": float(np.percentile(np.asarray(latencies), 50) * 1000),
                      "p99_ms": float(np.percentile(np.asarray(latencies), 99) * 1000)})

    # A knob that trades recall for speed must produce a monotone curve. If it
    # does not, the measurement is contaminated and the comparison built on it
    # would be wrong, so the report says so rather than quietly ranking on it.
    monotone = all(curve[i]["qps"] >= curve[i + 1]["qps"] - 1e-9
                   for i in range(len(curve) - 1))

    # The cheapest setting that clears each target. A target the engine cannot
    # reach at any setting is reported as unreached rather than approximated.
    at_target: dict[str, Any] = {}
    for want in targets:
        reached = [point for point in curve if point["recall_at_k"] >= want]
        at_target[f"{want:.2f}"] = (
            min(reached, key=lambda point: point["setting"] or 0)
            if reached else {"unreached": True,
                             "best_recall": max(p["recall_at_k"] for p in curve)})

    result = {"engine": engine.name, "mode": "equal-recall",
              "qps_curve_monotone": monotone,
              "build_ops_per_s": trace.live / build_s if build_s else 0.0,
              "insert_ops_per_s": insert_ops_per_s,
              "tunable": bool(engine.search_ladder),
              "curve": curve, "at_target": at_target}
    engine.close()
    return result


def run_equal_recall_benchmark(base: np.ndarray, queries: np.ndarray,
                               trace: "OpsTrace", *, engines: list[str],
                               k: int = 10, metric: str = "l2",
                               targets: tuple[float, ...] = (0.90, 0.95, 0.99),
                               options: dict[str, dict[str, Any]] | None = None
                               ) -> dict[str, Any]:
    options = options or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"platform": platform.platform(), "machine": platform.machine(),
                        "python": platform.python_version(), "numpy": np.__version__},
        "workload": {"live_vectors": trace.live, "dimensions": int(base.shape[1]),
                     "k": k, "metric": metric, "targets": list(targets),
                     "note": "search knob swept; engines compared at equal recall"},
        "engines": {},
    }
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine_equal_recall(
                ENGINES[name], base, trace, queries, k, metric,
                options.get(name, {}), targets)
        except Exception as error:
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report


def run_engine_batching(factory: Callable[..., Engine], base: np.ndarray,
                        trace: "OpsTrace", queries: np.ndarray, k: int,
                        metric: str, options: dict[str, Any]) -> dict[str, Any]:
    """The same work, once through the batch API and once an item at a time.

    Batch throughput and single-item throughput are different products. An
    ingestion job cares about the first; an agent writing one memory per turn,
    or a CDC stream, only ever gets the second. Reporting one number hides
    which an engine is good at, and engines differ enormously here: a batch API
    that is internally a loop looks identical to one that is not, until it is
    measured against its own single-item path.
    """
    results: dict[str, Any] = {}
    name = ""
    for shape in ("batched", "single"):
        engine = factory(dim=base.shape[1], metric=metric, live=trace.live,
                         **options)
        name = engine.name
        engine.build(base[trace.initial_rows], trace.initial_ids)
        live_ids, live_rows = trace.initial_ids.copy(), trace.initial_rows.copy()
        timings: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for phase in trace.phases:
            if phase.kind not in ("insert", "delete"):
                continue
            rows = base[phase.source_rows] if phase.source_rows is not None else None
            start = time.perf_counter()
            if shape == "batched":
                if phase.kind == "insert":
                    engine.insert_many(phase.ids, rows)
                else:
                    engine.delete_many(phase.ids)
            else:
                if phase.kind == "insert":
                    for position, one in enumerate(phase.ids):
                        engine.insert_one(int(one), rows[position])
                else:
                    for one in phase.ids:
                        engine.delete_one(int(one))
            timings.setdefault(phase.kind, []).append(time.perf_counter() - start)
            counts[phase.kind] = counts.get(phase.kind, 0) + int(phase.ids.shape[0])
            live_ids, live_rows = _next_live(live_ids, live_rows, phase)
        results[shape] = {
            kind: counts[kind] / sum(spent) if sum(spent) else 0.0
            for kind, spent in timings.items()}
        engine.close()

    speedup = {}
    for kind, batched in results["batched"].items():
        single = results["single"].get(kind) or 0.0
        speedup[kind] = batched / single if single else None
    return {"engine": name,
            "mode": "batching", "batched_ops_per_s": results["batched"],
            "single_ops_per_s": results["single"], "batch_speedup": speedup}


def run_batching_benchmark(base: np.ndarray, queries: np.ndarray,
                           trace: "OpsTrace", *, engines: list[str], k: int = 10,
                           metric: str = "l2",
                           options: dict[str, dict[str, Any]] | None = None
                           ) -> dict[str, Any]:
    options = options or {}
    report = {"schema_version": SCHEMA_VERSION,
              "workload": {"live_vectors": trace.live,
                           "dimensions": int(base.shape[1]), "metric": metric,
                           "note": "same work through the batch API and one "
                                   "item at a time"},
              "engines": {}}
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine_batching(
                ENGINES[name], base, trace, queries, k, metric,
                options.get(name, {}))
        except Exception as error:
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report


def run_dimension_sweep(make_data: Callable[[int], tuple[np.ndarray, np.ndarray]],
                        dimensions: Sequence[int], *, engines: list[str],
                        live: int, batch: int, k: int = 10, metric: str = "l2",
                        options: dict[str, dict[str, Any]] | None = None
                        ) -> dict[str, Any]:
    """Run the per-operation benchmark across several vector widths.

    Everything here had been measured at d=128. Width is not a scaling factor
    that divides out: it moves distance computation relative to bookkeeping,
    so an engine whose cost is per-record looks better as d grows and one whose
    cost is per-dimension looks worse. A single width cannot tell you which
    you are looking at.
    """
    from .trace import build_ops_trace
    options = options or {}
    report = {"schema_version": SCHEMA_VERSION,
              "workload": {"live_vectors": live, "batch": batch,
                           "dimensions": list(dimensions), "metric": metric,
                           "k": k, "note": "per-operation timing at each width"},
              "by_dimension": {}}
    for dim in dimensions:
        base, queries = make_data(dim)
        trace = build_ops_trace(base.shape[0], live, batch, 1)
        report["by_dimension"][str(dim)] = run_ops_benchmark(
            base, queries, trace, engines=engines, k=k, metric=metric,
            options=options)["engines"]
    return report


def _draw_tags(count: int, fraction: float, wanted: int, other: int,
               layout: str, rng: np.random.Generator,
               vectors: np.ndarray | None = None) -> np.ndarray:
    """Assign the filter tag to `fraction` of `count` records.

    The layout decides which of the two skipping mechanisms is even reachable,
    so it is a parameter rather than a fixed choice.

    `random` scatters the tag uniformly. Every page then holds a mixture, so a
    page-level union test almost never rules a page out -- at 1% with 256
    records to a page, 92% of pages still contain a match. Whatever speedup
    survives here comes from rejecting individual records before their
    distance is computed, and this is the worst case for any index that groups
    by tag.

    `clustered` gives the tag to a contiguous run, which is what a tenant id,
    an owner, or a time bucket actually looks like: records sharing a tag are
    written together and land together. This is the realistic case and the one
    where a page-level test can rule out whole pages at once.

    `semantic` gives the tag to the records nearest a randomly chosen anchor
    vector. This is the layout that actually tests page-level skipping, and
    the previous two do not: an index that groups records by similarity places
    a contiguous *id* range across many pages, so `clustered` is scattered
    from the index's point of view even though it looks grouped from the
    caller's. A tenant whose corpus is topically coherent -- one language, one
    product catalogue, one customer's documents -- correlates with the vector
    space, and that is the only correlation this structure can exploit.

    Reporting only one of these would be a choice about which mechanism to
    credit. All three are run.
    """
    tags = np.full(count, other, dtype=np.int64)
    marked = int(round(count * fraction))
    if marked <= 0:
        return tags
    if layout == "semantic" and vectors is not None:
        anchor = vectors[int(rng.integers(0, vectors.shape[0]))]
        gap = np.linalg.norm(vectors - anchor, axis=1)
        tags[np.argpartition(gap, marked - 1)[:marked]] = wanted
    elif layout == "clustered":
        start = int(rng.integers(0, max(1, count - marked + 1)))
        tags[start:start + marked] = wanted
    else:
        tags[rng.choice(count, size=marked, replace=False)] = wanted
    return tags


def _match_recall(engine, queries, k, metric, live_ids, live_vectors,
                  live_tags, wanted: int, target: float) -> dict[str, Any]:
    """Walk the engine's search ladder to the cheapest point clearing `target`.

    If no rung clears it, the best rung is reported along with the shortfall
    rather than a failure: an engine that cannot reach the target on this
    workload has told us something, and dropping it from the table would hide
    exactly the case worth knowing about.
    """
    ladder = getattr(engine, "search_ladder", ()) or ()
    best = None
    for rung in ladder:
        try:
            engine.set_search_param(rung)
        except NotImplementedError:
            break
        attempt = _measure_filtered(engine, queries, k, metric, live_ids,
                                    live_vectors, live_tags, wanted)
        attempt["search_param"] = rung
        if best is None or attempt["recall_at_k"] > best["recall_at_k"]:
            best = attempt
        if attempt["recall_at_k"] >= target:
            attempt["met_target"] = True
            return attempt
    if best is None:
        best = _measure_filtered(engine, queries, k, metric, live_ids,
                                 live_vectors, live_tags, wanted)
        best["search_param"] = 0
    best["met_target"] = best["recall_at_k"] >= target
    return best


def _measure_filtered(engine, queries, k, metric, live_ids, live_vectors,
                      live_tags, wanted: int) -> dict[str, Any]:
    """One filtered measurement against the corpus as it stands right now.

    Truth is recomputed from the live set on every call rather than carried
    forward, because under churn the matching subset changes with every round
    and a stale truth would score an engine on neighbours that no longer exist.
    """
    keep = live_tags == wanted
    matching = live_ids[keep]
    truth = exact_truth(live_vectors[keep], matching, queries, k, metric)
    allowed = set(matching.tolist())
    attainable = min(k, matching.shape[0])

    latencies, hits, short = [], 0.0, 0
    for index in range(queries.shape[0]):
        begin = time.perf_counter()
        found = engine.search_filtered(queries[index], k, wanted)
        latencies.append((time.perf_counter() - begin) * 1000.0)
        wrong = [i for i in found if i not in allowed]
        if wrong:
            raise RuntimeError(
                f"{engine.name} returned {len(wrong)} records that do not "
                f"match the filter")
        hits += len(set(found) & set(truth[index].tolist())) / k
        if len(found) < attainable:
            short += 1
    # The control. Without it, "filtered search got 2x slower after churn"
    # cannot be told apart from "every search got 2x slower after churn",
    # which is a different claim about a different part of the engine. What
    # the filter is worth is the ratio between these two, not either alone.
    plain = []
    for index in range(queries.shape[0]):
        begin = time.perf_counter()
        engine.search(queries[index], k)
        plain.append((time.perf_counter() - begin) * 1000.0)

    return {"matching_records": int(matching.shape[0]),
            "fetch_width": engine.filter_fetch_width(),
            "unfiltered_latency": _latency(plain),
            "filter_speedup": (float(np.percentile(plain, 50))
                               / float(np.percentile(latencies, 50))),
            "recall_at_k": hits / queries.shape[0],
            # How often the engine could not fill k. An over-fetching engine
            # starts failing this as the filter narrows; a native one does not.
            "short_result_rate": short / queries.shape[0],
            "latency": _latency(latencies)}


def run_engine_filtered(factory: Callable[..., Engine], base: np.ndarray,
                        trace: "OpsTrace", queries: np.ndarray, k: int,
                        metric: str, options: dict[str, Any],
                        selectivities: Sequence[float] = (0.5, 0.1, 0.01),
                        churn_rounds: int = 0, churn_fraction: float = 0.25,
                        layout: str = "random",
                        target_recall: float = 0.95) -> dict[str, Any]:
    """Filtered search, swept over how selective the filter is, then churned.

    Selectivity is the axis that separates the two ways of doing this. An
    engine that filters inside the search skips work as the filter narrows: a
    page or a node that cannot match is never visited. An engine that has no
    filter forces the caller to over-fetch and discard, so narrowing the filter
    makes it do *more* work and eventually return fewer than k results however
    wide it fetches.

    Churn is the second axis, and the one a static benchmark hides. Skipping
    work by tag depends on records that share a tag being grouped together;
    mutation is exactly what disturbs that grouping, so an advantage measured
    on a freshly built index is not evidence of an advantage on a live one.
    Each round retires `churn_fraction` of the live set and inserts the same
    number of fresh records with freshly drawn tags, then re-measures.

    Recall is measured against the exact nearest neighbours **that satisfy the
    filter**, which is the only ground truth the question has. Comparing
    against the unfiltered neighbours would score every engine on how many
    matching records happen to be near, not on whether it found them.
    """
    probe = factory(dim=base.shape[1], metric=metric, live=trace.live, **options)
    style = getattr(probe, "filtering", "none")
    name = probe.name
    probe.close()
    if style == "none":
        raise RuntimeError(
            f"{name} cannot restrict a search to an attribute")

    initial_ids = trace.initial_ids
    initial_vectors = base[trace.initial_rows]
    rng = np.random.default_rng(17)
    wanted = 1
    plan = (build_trace(base.shape[0], initial_ids.shape[0], churn_rounds,
                        churn_fraction, seed=101) if churn_rounds else None)
    out: list[dict[str, Any]] = []

    for fraction in selectivities:
        # A fresh index per selectivity. Reusing one made every engine that
        # appends rather than rebuilding load the corpus again on the second
        # pass -- usearch refused outright on duplicate keys, and the others
        # would have silently measured a doubled index.
        engine = factory(dim=base.shape[1], metric=metric, live=trace.live,
                         **options)
        # One tag marks `fraction` of the corpus; the rest carry a different
        # tag, so the filter is a genuine restriction rather than a no-op.
        live_ids = initial_ids.copy()
        live_vectors = initial_vectors.copy()
        live_tags = _draw_tags(live_ids.shape[0], fraction, wanted, 2,
                               layout, rng, live_vectors)
        # Sized before the build so a post-filtering engine is asked the
        # question it can actually answer.
        engine.set_filter_selectivity(fraction, k, live_ids.shape[0])
        try:
            engine.build_filtered(live_ids, live_vectors, live_tags)
        except NotImplementedError as unsupported:
            out.append({"selectivity": fraction,
                        "unsupported": str(unsupported)})
            engine.close()
            continue

        # Matched on recall, not on the search parameter. A filter shrinks
        # the pool a fixed probe width draws from, so the same nprobe buys
        # less recall filtered than unfiltered, and by a different amount for
        # each engine and each selectivity. Comparing at a fixed setting
        # therefore compares operating points: at nprobe 32 this index
        # answered a 10% semantic filter at recall 0.76 and looked fast for
        # it. Each engine is walked up its own ladder to the first point that
        # clears the target, and that is the point reported.
        first = _match_recall(engine, queries, k, metric, live_ids,
                              live_vectors, live_tags, wanted, target_recall)
        rounds = [dict(first, round=0, churned=0.0)]
        turned = 0.0
        for number in range(churn_rounds):
            epoch = plan.epochs[number]
            positions = epoch.positions
            fresh_vectors = base[epoch.source_rows]
            fresh_tags = _draw_tags(positions.shape[0], fraction, wanted, 2,
                                    layout, rng, fresh_vectors)
            try:
                engine.delete_filtered(live_ids[positions])
                engine.insert_filtered(epoch.new_ids, fresh_vectors,
                                       fresh_tags)
            except NotImplementedError as unsupported:
                rounds.append({"round": number + 1,
                               "unsupported": str(unsupported)})
                break
            live_ids = live_ids.copy()
            live_vectors = live_vectors.copy()
            live_tags = live_tags.copy()
            live_ids[positions] = epoch.new_ids
            live_vectors[positions] = fresh_vectors
            live_tags[positions] = fresh_tags
            turned += churn_fraction
            rounds.append(dict(
                _measure_filtered(engine, queries, k, metric, live_ids,
                                  live_vectors, live_tags, wanted),
                round=number + 1, churned=turned))
        engine.close()
        first = rounds[0]
        entry = {"selectivity": fraction, "rounds": rounds}
        # The static numbers stay at the top level so a report that does not
        # churn reads exactly as it did before churn existed.
        entry.update({key: first[key] for key in
                      ("matching_records", "fetch_width", "recall_at_k",
                       "short_result_rate", "latency", "unfiltered_latency",
                       "filter_speedup", "search_param", "met_target")})
        final = [r for r in rounds if "latency" in r][-1]
        if final is not first:
            entry["after_churn"] = {
                "churned": final["churned"],
                "recall_at_k": final["recall_at_k"],
                "short_result_rate": final["short_result_rate"],
                "p50_ms": final["latency"]["p50_ms"],
                # Above 1.0 means churn made filtered search slower, which is
                # the failure mode this whole mode exists to catch.
                "p50_ratio": (final["latency"]["p50_ms"]
                              / first["latency"]["p50_ms"]
                              if first["latency"]["p50_ms"] else 0.0),
                "filter_speedup": final["filter_speedup"],
                # The same ratio for an unfiltered search. If this moved as
                # much as p50_ratio did, churn slowed the engine down and did
                # not erode the filter specifically.
                "unfiltered_p50_ratio": (
                    final["unfiltered_latency"]["p50_ms"]
                    / first["unfiltered_latency"]["p50_ms"]
                    if first["unfiltered_latency"]["p50_ms"] else 0.0),
                "recall_delta": final["recall_at_k"] - first["recall_at_k"]}
        out.append(entry)
    return {"engine": name, "mode": "filtered", "filtering": style,
            "layout": layout, "target_recall": target_recall,
            "churn_rounds": churn_rounds,
            "churn_fraction": churn_fraction, "selectivities": out}


def run_filtered_benchmark(base: np.ndarray, queries: np.ndarray,
                           trace: "OpsTrace", *, engines: list[str],
                           k: int = 10, metric: str = "l2",
                           options: dict[str, dict[str, Any]] | None = None,
                           selectivities: Sequence[float] = (0.5, 0.1, 0.01),
                           churn_rounds: int = 0,
                           churn_fraction: float = 0.25,
                           layout: str = "random",
                           target_recall: float = 0.95) -> dict[str, Any]:
    options = options or {}
    report = {
        "schema_version": SCHEMA_VERSION,
        "environment": {"platform": platform.platform(),
                        "machine": platform.machine(),
                        "python": platform.python_version(),
                        "numpy": np.__version__},
        "workload": {"live_vectors": trace.live, "k": k, "metric": metric,
                     "dimensions": int(base.shape[1]),
                     "note": "filtered search swept over selectivity, "
                             "each engine matched on recall",
                     "tag_layout": layout, "target_recall": target_recall},
        "engines": {},
    }
    for name in engines:
        if name not in ENGINES:
            report["engines"][name] = {"skipped": f"unknown engine {name!r}"}
            continue
        try:
            report["engines"][name] = run_engine_filtered(
                ENGINES[name], base, trace, queries, k, metric,
                options.get(name, {}), selectivities=selectivities,
                churn_rounds=churn_rounds, churn_fraction=churn_fraction,
                layout=layout, target_recall=target_recall)
        except Exception as error:
            report["engines"][name] = {"skipped": f"{type(error).__name__}: {error}"}
    return report
