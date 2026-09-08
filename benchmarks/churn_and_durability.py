"""What only this engine claims: bounded space under churn, and durability.

Neither has a faiss counterpart, so these are not comparisons. They are the
measurements that decide whether the claims in README.md are true, taken on
real vectors at a realistic size.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import shutil
import tempfile
import time

import numpy as np


def resident_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024)


def load(path: str, live: int, queries: int):
    import h5py

    with h5py.File(path, "r") as handle:
        pool = np.asarray(handle["train"][: int(live * 2.2)], dtype=np.float32)
        test = np.asarray(handle["test"][:queries], dtype=np.float32)
    return pool, test


def recall_of(index, queries, truth, k):
    return sum(
        len({r.id for r in index.search(queries[i], k)} & set(truth[i].tolist()))
        for i in range(len(queries))
    ) / (k * len(queries))


def churn(pool, queries, live, cycles, k):
    """Sustained full turnover, watching space and recall rather than speed."""
    from chronovec import Index

    index = Index(pool.shape[1], metric="l2", page_capacity=256, nprobe=64)
    index.insert_many(np.arange(live), pool[:live])
    ids, nxt, rows = np.arange(live), live, np.arange(live)
    out = []
    for cycle in range(cycles):
        base = pool[rows]
        scores = (
            (queries**2).sum(1)[:, None] - 2.0 * (queries @ base.T) + (base * base).sum(1)[None, :]
        )
        near = np.argpartition(scores, k, axis=1)[:, :k]
        truth = ids[near]
        stats = index.stats()
        out.append(
            {
                "cycle": cycle,
                "live": stats["live_vectors"],
                "pages": stats["pages"],
                "amplification": stats["capacity_amplification"],
                "bytes_per_live_vector": (
                    stats["tracked_index_bytes"] / max(1, stats["live_vectors"])
                ),
                "resident_mb": resident_mb(),
                "recall_at_k": recall_of(index, queries, truth, k),
            }
        )
        # Replace the whole live set, then reclaim.
        index.delete_many(ids)
        rows = (np.arange(nxt, nxt + live)) % pool.shape[0]
        ids = np.arange(nxt, nxt + live)
        index.insert_many(ids, pool[rows])
        nxt += live
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
        gc.collect()
    index.close()
    return out


def durability(pool, live):
    """What a write-ahead log costs, and that recovery reproduces the index."""
    from chronovec import Index

    out = []
    for label, options in (
        ("none", {}),
        ("wal", {"wal_sync": False}),
        ("wal+fsync", {"wal_sync": True}),
    ):
        workspace = tempfile.mkdtemp(prefix="cvdur-")
        if label != "none":
            options["wal_path"] = os.path.join(workspace, "index.wal")
        index = Index(pool.shape[1], metric="l2", page_capacity=256, nprobe=64, **options)
        started = time.perf_counter()
        index.insert_many(np.arange(live), pool[:live])
        batch = (time.perf_counter() - started) / live * 1e6
        started = time.perf_counter()
        for position in range(2000):
            index.insert(live + position, pool[live + position])
        single = (time.perf_counter() - started) / 2000 * 1e6
        row = {"mode": label, "insert_batch_us": batch, "insert_single_us": single}
        if label != "none":
            row["log_bytes"] = os.path.getsize(options["wal_path"])
            del index  # no clean shutdown
            started = time.perf_counter()
            recovered = Index(pool.shape[1], metric="l2", page_capacity=256, nprobe=64, **options)
            row["recovery_s"] = time.perf_counter() - started
            row["recovered_live"] = recovered.stats()["live_vectors"]
            row["expected_live"] = live + 2000
            recovered.close()
        else:
            index.close()
        shutil.rmtree(workspace, ignore_errors=True)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--live", type=int, default=200000)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="benchmarks/churn.json")
    args = parser.parse_args()

    pool, queries = load(args.dataset, args.live, args.queries)
    report = {"dataset": args.dataset, "live": args.live, "k": args.k}

    report["churn"] = churn(pool, queries, args.live, args.cycles, args.k)
    print(f"{'cycle':>6}{'pages':>8}{'amp':>7}{'B/vec':>8}{'rss MB':>9}{'recall':>9}")
    for row in report["churn"]:
        print(
            f"{row['cycle']:>6}{row['pages']:>8,}{row['amplification']:>7.2f}"
            f"{row['bytes_per_live_vector']:>8.0f}{row['resident_mb']:>9.0f}"
            f"{row['recall_at_k']:>9.4f}"
        )

    report["durability"] = durability(pool, args.live)
    print(
        f"\n{'mode':>10}{'batch us':>10}{'single us':>11}{'log MB':>9}"
        f"{'recovery s':>12}{'recovered':>11}"
    )
    for row in report["durability"]:
        print(
            f"{row['mode']:>10}{row['insert_batch_us']:>10.2f}"
            f"{row['insert_single_us']:>11.2f}"
            f"{row.get('log_bytes', 0) / 1e6:>9.0f}"
            f"{row.get('recovery_s', 0):>12.2f}"
            f"{row.get('recovered_live', '-'):>11}"
        )

    with open(args.output, "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
