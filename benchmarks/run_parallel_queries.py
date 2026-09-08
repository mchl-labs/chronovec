#!/usr/bin/env python3
"""Measure query-level parallel scaling over immutable native snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_benchmark import environment_summary, latency_summary, normalize

from chronovec import NativeChronoVecIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=96)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--queries-per-thread", type=int, default=2_000)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(177)
    values = normalize(rng.normal(size=(args.vectors, args.dimensions))).astype(np.float32)
    index = NativeChronoVecIndex(args.dimensions, nprobe=args.nprobe, adaptive_bounds=args.adaptive)
    for item_id, vector in enumerate(values):
        index.insert(item_id, vector)

    def worker(seed):
        local = np.random.default_rng(seed)
        latencies = []
        for _ in range(args.queries_per_thread):
            query = values[local.integers(0, len(values))]
            started = time.perf_counter()
            index.search(query, 10, adaptive=args.adaptive)
            latencies.append(time.perf_counter() - started)
        return latencies

    variants = {}
    for thread_count in args.threads:
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=thread_count) as pool:
            rows = list(pool.map(worker, range(thread_count)))
        elapsed = time.perf_counter() - started
        latencies = [latency for row in rows for latency in row]
        variants[str(thread_count)] = {
            "aggregate_qps": len(latencies) / elapsed,
            "wall_s": elapsed,
            "latency": latency_summary(latencies),
        }
    report = {
        "schema_version": 1,
        "environment": environment_summary(),
        "config": {**vars(args), "output": str(args.output) if args.output else None},
        "variants": variants,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
