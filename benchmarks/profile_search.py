#!/usr/bin/env python3
"""Break down native ChronoVec search work without relying on a sampling profiler."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_benchmark import load_ann_benchmarks, recall_at_k

from chronovec import NativeChronoVecIndex


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=20_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--probes", type=int, nargs="+", default=[24, 32, 40])
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--dataset-limit", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    if args.dataset:
        values, queries, truth, metric = load_ann_benchmarks(
            args.dataset, args.queries, 10, args.dataset_limit
        )
        args.vectors, args.dimensions, args.queries = len(values), values.shape[1], len(queries)
    else:
        metric = "cosine"
        centers = normalize(rng.normal(size=(32, args.dimensions)).astype(np.float32))
        assignments = rng.integers(0, len(centers), size=args.vectors)
        values = normalize(
            centers[assignments] + 0.18 * rng.normal(size=(args.vectors, args.dimensions))
        ).astype(np.float32)
        queries = normalize(
            values[rng.choice(args.vectors, args.queries, replace=False)]
            + 0.03 * rng.normal(size=(args.queries, args.dimensions))
        ).astype(np.float32)
        scores = queries @ values.T
        truth = np.argpartition(-scores, kth=9, axis=1)[:, :10]
    report = {}
    for screening, adaptive in ((False, False), (False, True), (True, False)):
        index = NativeChronoVecIndex(
            args.dimensions,
            metric=metric,
            nprobe=max(args.probes),
            screening=screening,
            adaptive_bounds=adaptive,
        )
        for item_id, vector in enumerate(values):
            index.insert(item_id, vector)
        policy = "adaptive_bound" if adaptive else "screened" if screening else "exact"
        report[policy] = {}
        for probes in args.probes:
            totals = None
            found = []
            for query in queries:
                result = index.search(query, 10, nprobe=probes, adaptive=adaptive)
                found.append([row.id for row in result])
                metrics = index.last_search_metrics()
                if totals is None:
                    totals = {key: 0 for key in metrics}
                for key, value in metrics.items():
                    totals[key] += value
            averaged = {key: value / len(queries) for key, value in totals.items()}
            averaged["core_qps"] = 1e9 / averaged["total_ns"]
            averaged["recall_at_10"] = recall_at_k(np.asarray(found, dtype=np.int64), truth)
            averaged["candidate_pruning_fraction"] = 1.0 - averaged["visible_candidates"] / len(
                values
            )
            report[policy][str(probes)] = averaged
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
