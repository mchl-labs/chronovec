"""Streaming benchmark: how an index behaves over repeated dataset turnover.

ann-benchmarks measures a static index -- build once, query forever -- which is
why every mature system scores well on it and it cannot separate them on the
axis that matters for changing data. This harness measures the other axis.

The workload is a shared operation trace: the same (delete, insert) pairs, in
the same order, are replayed against every engine, so differences are the
engines' and not the trace's. After each epoch the live set has changed, so
exact ground truth is recomputed against the *current* live set rather than
reused from the dataset file.

Reported per epoch, per engine:
  write throughput and maintenance time
  Recall@k against freshly recomputed ground truth  (does quality drift?)
  query latency p50/p95/p99                          (does the tail degrade?)
  space: allocated slots, capacity amplification, bytes per live vector
  versions reclaimed                                 (is deletion bounded?)
And once at the end: checkpoint save/load time and post-recovery recall.

A system passes this benchmark by being *boring*: flat recall, flat latency,
flat space, across many turnovers.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, field

import numpy as np

SCHEMA_VERSION = 1


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def exact_truth(
    base: np.ndarray, ids: np.ndarray, queries: np.ndarray, k: int, angular: bool, chunk: int = 256
) -> np.ndarray:
    """Top-k ids per query against the current live set."""
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        block = queries[start:stop]
        if angular:
            distances = 1.0 - block @ base.T
        else:
            distances = (
                np.sum(block * block, axis=1)[:, None]
                - 2.0 * (block @ base.T)
                + np.sum(base * base, axis=1)[None, :]
            )
        take = min(k, base.shape[0])
        cols = np.argpartition(distances, take - 1, axis=1)[:, :take]
        rows = np.arange(stop - start)[:, None]
        cols = cols[rows, np.argsort(distances[rows, cols], axis=1)]
        out[start:stop] = ids[cols]
    return out


def latency_summary(samples: list[float]) -> dict[str, float]:
    array = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "max_ms": float(array.max()),
        "qps": float(len(array) / array.sum()) if array.sum() else 0.0,
    }


@dataclass
class Trace:
    """A replayable turnover trace, identical for every engine."""

    initial_ids: np.ndarray
    initial_rows: np.ndarray
    # (positions, retired ids, new ids, source rows). Positions are carried so
    # replaying an epoch is O(replacements); locating each retired id by value
    # would make the replay quadratic in the live-set size.
    epochs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=list
    )


def build_trace(pool: int, live: int, epochs: int, fraction: float, seed: int) -> Trace:
    rng = np.random.default_rng(seed)
    order = rng.permutation(pool)
    initial_rows = order[:live]
    trace = Trace(initial_ids=np.arange(live, dtype=np.int64), initial_rows=initial_rows)
    live_ids = trace.initial_ids.copy()
    live_rows = initial_rows.copy()
    cursor = live
    next_id = live
    per_epoch = max(1, int(live * fraction))
    for _ in range(epochs):
        victims = rng.choice(live, size=per_epoch, replace=False)
        rows = np.empty(per_epoch, dtype=np.int64)
        for i in range(per_epoch):
            if cursor >= pool:  # wrap the pool if it is exhausted
                cursor = 0
            rows[i] = order[cursor]
            cursor += 1
        new_ids = np.arange(next_id, next_id + per_epoch, dtype=np.int64)
        next_id += per_epoch
        trace.epochs.append((victims.copy(), live_ids[victims].copy(), new_ids, rows))
        live_ids[victims] = new_ids
        live_rows[victims] = rows
    return trace


def run_chronovec(base, trace, queries, k, angular, args, report):
    from chronovec import Index

    index = Index(
        base.shape[1],
        metric="cosine" if angular else "l2",
        page_capacity=args.page_capacity,
        nprobe=args.nprobe,
        screening=not args.disable_screening,
    )
    start = time.perf_counter()
    for item_id, row in zip(trace.initial_ids, trace.initial_rows):
        index.insert(int(item_id), base[row])
    build_s = time.perf_counter() - start

    live_ids = trace.initial_ids.copy()
    live_rows = trace.initial_rows.copy()
    epochs = []

    def measure(epoch, write_s, maintenance_s, reclaimed, ops):
        truth = exact_truth(base[live_rows], live_ids, queries, k, angular)
        latencies, hits = [], 0.0
        for qi in range(queries.shape[0]):
            t0 = time.perf_counter()
            found = index.search(queries[qi], k=k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            hits += len({r.id for r in found} & set(truth[qi].tolist())) / k
        stats = index.stats()
        return {
            "epoch": epoch,
            "live_vectors": int(live_ids.shape[0]),
            "write_ops": ops,
            "write_s": write_s,
            "write_ops_per_s": (ops / write_s) if write_s else 0.0,
            "maintenance_s": maintenance_s,
            "reclaimed_versions": reclaimed,
            "recall_at_k": hits / queries.shape[0],
            "latency": latency_summary(latencies),
            "allocated_slots": int(stats.get("allocated_slots", 0)),
            "capacity_amplification": float(stats.get("capacity_amplification", 0.0)),
            "pages": int(stats.get("pages", 0)),
        }

    epochs.append(measure(0, build_s, 0.0, 0, int(live_ids.shape[0])))

    for number, (positions, old_ids, new_ids, rows) in enumerate(trace.epochs, start=1):
        reclaimed = 0
        maintenance_s = 0.0
        start = time.perf_counter()
        for position, (old_id, new_id, row) in enumerate(zip(old_ids, new_ids, rows)):
            index.delete(int(old_id))
            index.insert(int(new_id), base[row])
            if args.maintenance_every and position % args.maintenance_every == 0:
                t0 = time.perf_counter()
                reclaimed += index.vacuum(
                    oldest_snapshot=index.clock + 1, budget_versions=args.maintenance_budget
                )
                maintenance_s += time.perf_counter() - t0
        write_s = time.perf_counter() - start
        # settle any versions the budget did not reach this epoch
        t0 = time.perf_counter()
        reclaimed += index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
        maintenance_s += time.perf_counter() - t0
        live_ids[positions] = new_ids
        live_rows[positions] = rows
        epochs.append(measure(number, write_s, maintenance_s, reclaimed, 2 * len(old_ids)))

    recovery = {}
    if args.checkpoint:
        import os
        import tempfile

        path = os.path.join(tempfile.gettempdir(), "chronovec_streaming.cvec")
        t0 = time.perf_counter()
        index.save(path)
        recovery["save_s"] = time.perf_counter() - t0
        recovery["bytes"] = os.path.getsize(path)
        t0 = time.perf_counter()
        restored = type(index).load(path)
        recovery["load_s"] = time.perf_counter() - t0
        truth = exact_truth(base[live_rows], live_ids, queries, k, angular)
        hits = sum(
            len({r.id for r in restored.search(queries[qi], k=k)} & set(truth[qi].tolist())) / k
            for qi in range(queries.shape[0])
        )
        recovery["recall_after_recovery"] = hits / queries.shape[0]
        restored.close()
        os.remove(path)

    index.close()
    report["engines"]["chronovec"] = {"epochs": epochs, "recovery": recovery}


def run_hnswlib(base, trace, queries, k, angular, args, report):
    try:
        import hnswlib
    except Exception as exc:  # pragma: no cover - optional baseline
        report["engines"]["hnswlib"] = {"skipped": str(exc)}
        return

    live = int(trace.initial_ids.shape[0])
    index = hnswlib.Index(space="cosine" if angular else "l2", dim=base.shape[1])
    index.init_index(
        max_elements=live + args.hnsw_headroom,
        M=args.hnsw_m,
        ef_construction=args.hnsw_ef_construction,
        random_seed=42,
        allow_replace_deleted=True,
    )
    index.set_ef(args.hnsw_ef_search)
    start = time.perf_counter()
    index.add_items(base[trace.initial_rows], trace.initial_ids, num_threads=1)
    build_s = time.perf_counter() - start

    live_ids = trace.initial_ids.copy()
    live_rows = trace.initial_rows.copy()
    epochs = []

    def measure(epoch, write_s, ops):
        truth = exact_truth(base[live_rows], live_ids, queries, k, angular)
        latencies, hits = [], 0.0
        for qi in range(queries.shape[0]):
            t0 = time.perf_counter()
            labels, _ = index.knn_query(queries[qi : qi + 1], k=k)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            hits += len(set(labels[0].tolist()) & set(truth[qi].tolist())) / k
        return {
            "epoch": epoch,
            "live_vectors": int(live_ids.shape[0]),
            "write_ops": ops,
            "write_s": write_s,
            "write_ops_per_s": (ops / write_s) if write_s else 0.0,
            "recall_at_k": hits / queries.shape[0],
            "latency": latency_summary(latencies),
            "serialized_bytes_per_live_vector": None,
        }

    epochs.append(measure(0, build_s, live))
    for number, (positions, old_ids, new_ids, rows) in enumerate(trace.epochs, start=1):
        start = time.perf_counter()
        for old_id in old_ids:
            index.mark_deleted(int(old_id))
        index.add_items(base[rows], new_ids, num_threads=1, replace_deleted=True)
        write_s = time.perf_counter() - start
        live_ids[positions] = new_ids
        live_rows[positions] = rows
        epochs.append(measure(number, write_s, 2 * len(old_ids)))

    report["engines"]["hnswlib"] = {
        "epochs": epochs,
        "note": "deleted-slot reuse enabled; the stronger bounded-capacity baseline",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="ann-benchmarks HDF5 file")
    parser.add_argument("--live", type=int, default=200000, help="live vectors held constant")
    parser.add_argument("--epochs", type=int, default=5, help="turnover epochs")
    parser.add_argument(
        "--turnover",
        type=float,
        default=1.0,
        help="fraction of the live set replaced per epoch (1.0 = full turnover)",
    )
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--angular", action="store_true")
    parser.add_argument("--page-capacity", type=int, default=256)
    parser.add_argument("--nprobe", type=int, default=96)
    parser.add_argument("--disable-screening", action="store_true")
    parser.add_argument("--maintenance-every", type=int, default=64)
    parser.add_argument("--maintenance-budget", type=int, default=64)
    parser.add_argument("--checkpoint", action="store_true", help="measure save/load recovery")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=100)
    parser.add_argument("--hnsw-ef-search", type=int, default=128)
    parser.add_argument("--hnsw-headroom", type=int, default=0)
    parser.add_argument("--skip-hnswlib", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import h5py

    with h5py.File(args.dataset, "r") as handle:
        base = np.asarray(handle["train"][:], dtype=np.float32)
        queries = np.asarray(handle["test"][: args.queries], dtype=np.float32)
    if args.angular:
        base, queries = normalize(base), normalize(queries)
    if args.live > base.shape[0]:
        raise SystemExit(f"--live {args.live} exceeds dataset size {base.shape[0]}")

    trace = build_trace(base.shape[0], args.live, args.epochs, args.turnover, args.seed)
    report = {
        "schema_version": SCHEMA_VERSION,
        "config": vars(args),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "workload": {
            "live_vectors": args.live,
            "epochs": args.epochs,
            "turnover_fraction": args.turnover,
            "replacements_per_epoch": int(trace.epochs[0][0].shape[0]) if trace.epochs else 0,
            "note": "identical operation trace replayed against every engine",
        },
        "engines": {},
    }

    run_chronovec(base, trace, queries, args.k, args.angular, args, report)
    if not args.skip_hnswlib:
        run_hnswlib(base, trace, queries, args.k, args.angular, args, report)

    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)

    print(
        f"\n{'engine':<12}{'epoch':>6}{'recall@k':>10}{'write ops/s':>13}"
        f"{'p99 ms':>9}{'cap amp':>9}"
    )
    for name, payload in report["engines"].items():
        if "epochs" not in payload:
            print(f"{name:<12}  skipped: {payload.get('skipped')}")
            continue
        for row in payload["epochs"]:
            amp = row.get("capacity_amplification")
            print(
                f"{name:<12}{row['epoch']:>6}{row['recall_at_k']:>10.4f}"
                f"{row['write_ops_per_s']:>13.0f}{row['latency']['p99_ms']:>9.3f}"
                f"{(f'{amp:.3f}' if amp else '-'):>9}"
            )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
