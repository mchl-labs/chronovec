#!/usr/bin/env python3
"""Mixed read/write benchmark with a reproducible prefilled native index."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_benchmark import environment_summary, latency_summary

from chronovec import NativeChronoVecIndex


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--dimensions", type=int, default=96)
    parser.add_argument("--readers", type=int, default=8)
    parser.add_argument("--writers", type=int, default=1)
    parser.add_argument("--queries-per-reader", type=int, default=10_000)
    parser.add_argument("--writes", type=int, default=20_000)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup-queries", type=int, default=20)
    args = parser.parse_args()
    if args.writers < 1 or args.writes > args.vectors:
        parser.error("writers must be positive and writes must not exceed vectors")
    rng = np.random.default_rng(42)
    values = normalize(rng.normal(size=(args.vectors + args.writes, args.dimensions)))
    index = NativeChronoVecIndex(args.dimensions, nprobe=args.nprobe)
    build_started = time.perf_counter()
    for item_id in range(args.vectors):
        index.insert(item_id, values[item_id])
    build_s = time.perf_counter() - build_started

    def read_worker(seed):
        local = np.random.default_rng(seed)
        for _ in range(args.warmup_queries):
            index.search(values[local.integers(0, args.vectors)], 10)
        latencies = []
        for _ in range(args.queries_per_reader):
            query = values[local.integers(0, args.vectors)]
            started = time.perf_counter()
            index.search(query, 10)
            latencies.append(time.perf_counter() - started)
        return latencies

    def write_worker(start_offset, end_offset):
        started = time.perf_counter()
        replacement_latencies = []
        maintenance_s = 0.0
        for offset in range(start_offset, end_offset):
            victim = offset % args.vectors
            replacement_started = time.perf_counter()
            index.delete(victim)
            index.insert(victim, values[args.vectors + offset])
            replacement_latencies.append(time.perf_counter() - replacement_started)
            if offset % 128 == 127:
                maintenance_started = time.perf_counter()
                index.vacuum(index.clock + 1, budget_versions=128)
                maintenance_s += time.perf_counter() - maintenance_started
        return time.perf_counter() - started, replacement_latencies, maintenance_s

    started = time.perf_counter()
    boundaries = np.linspace(0, args.writes, args.writers + 1, dtype=int)
    with ThreadPoolExecutor(max_workers=args.readers + args.writers) as pool:
        readers = [pool.submit(read_worker, seed) for seed in range(args.readers)]
        writers = [
            pool.submit(write_worker, int(boundaries[i]), int(boundaries[i + 1]))
            for i in range(args.writers)
        ]
        per_reader = [future.result() for future in readers]
        write_results = [future.result() for future in writers]
    maintenance_started = time.perf_counter()
    index.vacuum(index.clock + 1)
    final_maintenance_s = time.perf_counter() - maintenance_started
    latencies = [item for reader in per_reader for item in reader]
    replacement_latencies = [item for _, rows, _ in write_results for item in rows]
    maintenance_s = sum(row[2] for row in write_results) + final_maintenance_s
    write_seconds = max(row[0] for row in write_results)
    elapsed = time.perf_counter() - started
    query_metrics = latency_summary(latencies)
    reader_qps = [len(reader) / sum(reader) for reader in per_reader]
    report = {
        "schema_version": 3,
        "config": vars(args) | {"output": str(args.output) if args.output else None},
        "environment": environment_summary(),
        "build_s": build_s,
        "build_vectors_per_s": args.vectors / build_s,
        "elapsed_s": elapsed,
        "aggregate_query_qps": len(latencies) / elapsed,
        "query_latency": query_metrics,
        "reader_qps_min": float(min(reader_qps)),
        "reader_qps_mean": float(np.mean(reader_qps)),
        "reader_qps_max": float(max(reader_qps)),
        "writer_operations_per_s": 2 * args.writes / write_seconds,
        "replacement_pairs_per_s": args.writes / write_seconds,
        "replacement_latency": latency_summary(replacement_latencies),
        "maintenance_s": maintenance_s,
        "stats": index.stats(),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
