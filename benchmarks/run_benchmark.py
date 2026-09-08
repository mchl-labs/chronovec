#!/usr/bin/env python3
"""Reproducible static and streaming benchmark for ChronoVec.

ScaNN is optional because official wheels are not available on every Python /
architecture combination. A missing baseline is reported, never substituted by
a homemade implementation under the ScaNN name.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Allow direct execution from a source checkout without requiring installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chronovec import ChronoVecIndex, NativeChronoVecIndex, exact_search


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def ground_truth(
    base: np.ndarray, ids: np.ndarray, queries: np.ndarray, k: int, metric: str = "cosine"
) -> np.ndarray:
    result = np.empty((len(queries), k), dtype=np.int64)
    batch = (
        128
        if metric == "cosine"
        else max(1, min(128, 64_000_000 // max(1, len(base) * base.shape[1] * 4)))
    )
    for start in range(0, len(queries), batch):
        if metric == "cosine":
            scores = queries[start : start + batch] @ base.T
        else:
            scores = -np.sum(
                (queries[start : start + batch, None, :] - base[None, :, :]) ** 2, axis=2
            )
        positions = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        ordered = np.take_along_axis(scores, positions, axis=1)
        order = np.argsort(-ordered, axis=1)
        result[start : start + batch] = ids[positions[np.arange(len(positions))[:, None], order]]
    return result


def load_ann_benchmarks(
    path: Path, query_count: int, k: int, vector_limit: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load an ann-benchmarks HDF5 cosine or Euclidean dataset."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("HDF5 datasets require the 'benchmark' extra (h5py)") from exc
    with h5py.File(path, "r") as dataset:
        distance = dataset.attrs.get("distance", "angular")
        if isinstance(distance, bytes):
            distance = distance.decode()
        distance = str(distance).lower()
        if distance not in {"angular", "cosine", "euclidean", "l2"}:
            raise ValueError(f"unsupported dataset distance {distance!r}")
        metric = "cosine" if distance in {"angular", "cosine"} else "l2"
        missing = {name for name in ("train", "test", "neighbors") if name not in dataset}
        if missing:
            raise ValueError(f"dataset is missing required arrays: {sorted(missing)}")
        count = min(query_count, len(dataset["test"]))
        train_count = (
            min(vector_limit, len(dataset["train"])) if vector_limit else len(dataset["train"])
        )
        base = np.asarray(dataset["train"][:train_count], dtype=np.float32)
        queries = np.asarray(dataset["test"][:count], dtype=np.float32)
        if metric == "cosine":
            base, queries = normalize(base), normalize(queries)
        if train_count == len(dataset["train"]):
            truth = np.asarray(dataset["neighbors"][:count, :k], dtype=np.int64)
        else:
            truth = ground_truth(base, np.arange(train_count, dtype=np.int64), queries, k, metric)
    if truth.shape[1] < k:
        raise ValueError(
            f"dataset ground truth has only {truth.shape[1]} neighbors; requested k={k}"
        )
    return base, queries, truth, metric


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.mean(
            [
                len(set(row).intersection(expected)) / len(expected)
                for row, expected in zip(found, truth)
            ]
        )
    )


def quality_summary(found: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    values = np.asarray(
        [
            len(set(row).intersection(expected)) / len(expected)
            for row, expected in zip(found, truth)
        ],
        dtype=np.float64,
    )
    return {
        "recall_at_k": float(values.mean()),
        "recall_p05": float(np.percentile(values, 5)),
        "recall_p50": float(np.percentile(values, 50)),
        "recall_p95": float(np.percentile(values, 95)),
        "perfect_query_fraction": float(np.mean(values == 1.0)),
    }


def latency_summary(latencies: list[float]) -> dict[str, float]:
    seconds = np.asarray(latencies, dtype=np.float64)
    values = seconds * 1_000
    return {
        "queries": int(len(values)),
        "total_query_s": float(seconds.sum()),
        "qps": float(len(values) / seconds.sum()),
        "mean_ms": float(values.mean()),
        "stddev_ms": float(values.std()),
        "min_ms": float(values.min()),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "p999_ms": float(np.percentile(values, 99.9)),
        "max_ms": float(values.max()),
    }


def rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def environment_summary() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }


def average_native_profile(samples: list[dict[str, int]], corpus_size: int) -> dict[str, float]:
    if not samples:
        return {}
    averaged = {key: float(np.mean([sample[key] for sample in samples])) for key in samples[0]}
    total = averaged.get("total_ns", 0.0)
    averaged["core_qps"] = 1e9 / total if total else 0.0
    averaged["candidate_fraction_of_corpus"] = averaged["visible_candidates"] / corpus_size
    if total:
        averaged["routing_time_fraction"] = averaged["routing_ns"] / total
        averaged["screening_time_fraction"] = averaged["screening_ns"] / total
        averaged["rerank_time_fraction"] = averaged["rerank_ns"] / total
    return averaged


def query_chronovec(
    index: ChronoVecIndex,
    queries: np.ndarray,
    k: int,
    nprobe: int,
    warmup: int = 0,
    corpus_size: int = 1,
    adaptive: bool = False,
    linear_routing: bool = False,
) -> tuple[np.ndarray, dict]:
    search_options = {"nprobe": nprobe, "adaptive": adaptive}
    if isinstance(index, NativeChronoVecIndex):
        search_options["linear_routing"] = linear_routing
    for query in queries[: min(warmup, len(queries))]:
        index.search(query, k, **search_options)
    rows, latencies = [], []
    profiles = []
    for query in queries:
        started = time.perf_counter()
        result = index.search(query, k, **search_options)
        latencies.append(time.perf_counter() - started)
        rows.append([item.id for item in result] + [-1] * (k - len(result)))
        if hasattr(index, "last_search_metrics"):
            profiles.append(index.last_search_metrics())
    summary = latency_summary(latencies)
    if profiles:
        summary["native_profile"] = average_native_profile(profiles, corpus_size)
    return np.asarray(rows, dtype=np.int64), summary


def query_exact_scan(
    base: np.ndarray, queries: np.ndarray, k: int, warmup: int = 0, metric: str = "cosine"
) -> tuple[np.ndarray, dict]:
    ids = np.arange(len(base), dtype=np.int64)
    for query in queries[: min(warmup, len(queries))]:
        exact_search(base, query[None, :], k, ids=ids, metric=metric)
    rows, latencies = [], []
    for query in queries:
        started = time.perf_counter()
        result = exact_search(base, query[None, :], k, ids=ids, metric=metric)[0]
        latencies.append(time.perf_counter() - started)
        rows.append([item.id for item in result])
    return np.asarray(rows, dtype=np.int64), latency_summary(latencies)


def query_hnsw(index, queries: np.ndarray, k: int, warmup: int = 0) -> tuple[np.ndarray, dict]:
    for query in queries[: min(warmup, len(queries))]:
        index.knn_query(query, k=k)
    rows, latencies = [], []
    for query in queries:
        started = time.perf_counter()
        labels, _ = index.knn_query(query, k=k)
        latencies.append(time.perf_counter() - started)
        rows.append(labels[0])
    return np.asarray(rows, dtype=np.int64), latency_summary(latencies)


def benchmark(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    dataset_mode = args.dataset is not None
    if dataset_mode:
        base, queries, truth, metric = load_ann_benchmarks(
            args.dataset, args.queries, args.k, args.dataset_limit
        )
        args.vectors, args.dimensions, args.queries = len(base), base.shape[1], len(queries)
        centers = None
    else:
        # Clustered data makes routing meaningful while shifted churn tests
        # whether routing adapts when the distribution moves.
        metric = "cosine"
        centers = normalize(rng.normal(size=(32, args.dimensions)))
        assignments = rng.integers(0, len(centers), size=args.vectors)
        base = normalize(
            centers[assignments] + 0.18 * rng.normal(size=(args.vectors, args.dimensions))
        )
        queries = normalize(
            base[rng.choice(args.vectors, args.queries, replace=False)]
            + 0.03 * rng.normal(size=(args.queries, args.dimensions))
        )
    ids = np.arange(args.vectors, dtype=np.int64)
    if not dataset_mode:
        truth = ground_truth(base, ids, queries, args.k, metric)
    if args.churn >= args.vectors:
        raise ValueError("--churn must be smaller than the loaded dataset")
    config = {
        key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()
    }
    report: dict = {
        "schema_version": 3,
        "config": config,
        "environment": environment_summary(),
        "dataset": {
            "mode": "ann-benchmarks" if dataset_mode else "synthetic_clustered",
            "vectors": int(len(base)),
            "dimensions": int(base.shape[1]),
            "queries": int(len(queries)),
            "file_bytes": args.dataset.stat().st_size if args.dataset else None,
            "metric": metric,
        },
        "initial": {},
        "streaming": {},
        "notes": [],
    }

    chrono = None
    if not args.skip_reference:
        chrono = ChronoVecIndex(
            args.dimensions,
            metric=metric,
            page_capacity=args.page_capacity,
            nprobe=max(args.probes),
        )
        memory_before = rss_bytes()
        started = time.perf_counter()
        for item_id, vector in zip(ids, base, strict=True):
            chrono.insert(int(item_id), vector)
        chrono_build = time.perf_counter() - started
        memory_after = rss_bytes()
        report["initial"]["chronovec"] = {
            "build_s": chrono_build,
            "build_vectors_per_s": len(base) / chrono_build,
            "rss_before_bytes": memory_before,
            "rss_after_bytes": memory_after,
            "rss_delta_bytes": memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None,
            "variants": {},
        }
        for nprobe in args.probes:
            found, latency = query_chronovec(
                chrono, queries, args.k, nprobe, args.warmup_queries, len(base)
            )
            report["initial"]["chronovec"]["variants"][str(nprobe)] = {
                **latency,
                **quality_summary(found, truth),
            }
        report["initial"]["chronovec"]["stats"] = chrono.stats()

    native = None
    try:
        native = NativeChronoVecIndex(
            args.dimensions,
            metric=metric,
            page_capacity=args.page_capacity,
            nprobe=max(args.probes),
            screening=not args.disable_native_screening,
            adaptive_bounds=args.adaptive_frontier,
            rerank_factor=args.native_rerank_factor,
        )
        memory_before = rss_bytes()
        started = time.perf_counter()
        for item_id, vector in zip(ids, base, strict=True):
            native.insert(int(item_id), vector)
        native_build = time.perf_counter() - started
        memory_after = rss_bytes()
        native_stats = native.stats()
        native_stats["tracked_payload_bytes"] = (
            native_stats["vector_payload_bytes"] + native_stats["screening_bytes"]
        )
        native_stats["tracked_payload_bytes_per_live_vector"] = native_stats[
            "tracked_payload_bytes"
        ] / max(1, native_stats["live_vectors"])
        native_stats["tracked_index_bytes_per_live_vector"] = native_stats[
            "tracked_index_bytes"
        ] / max(1, native_stats["live_vectors"])
        report["initial"]["chronovec_native"] = {
            "build_s": native_build,
            "build_vectors_per_s": len(base) / native_build,
            "rss_before_bytes": memory_before,
            "rss_after_bytes": memory_after,
            "rss_delta_bytes": memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None,
            "variants": {},
            "stats": native_stats,
        }
        if args.checkpoint_benchmark:
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "index.cvec"
                started = time.perf_counter()
                native.save(checkpoint)
                checkpoint_s = time.perf_counter() - started
                started = time.perf_counter()
                recovered = NativeChronoVecIndex.load(checkpoint)
                recovery_s = time.perf_counter() - started
                report["initial"]["chronovec_native"]["checkpoint"] = {
                    "save_s": checkpoint_s,
                    "load_s": recovery_s,
                    "bytes": checkpoint.stat().st_size,
                    "bytes_per_live_vector": checkpoint.stat().st_size / len(base),
                    "recovered_stats_match": recovered.stats() == native.stats(),
                }
                recovered.close()
        for nprobe in args.probes:
            found, latency = query_chronovec(
                native, queries, args.k, nprobe, args.warmup_queries, len(base)
            )
            report["initial"]["chronovec_native"]["variants"][str(nprobe)] = {
                **latency,
                **quality_summary(found, truth),
            }
            if args.adaptive_frontier:
                found, latency = query_chronovec(
                    native, queries, args.k, nprobe, args.warmup_queries, len(base), adaptive=True
                )
                report["initial"]["chronovec_native"]["variants"][f"{nprobe}:adaptive"] = {
                    **latency,
                    **quality_summary(found, truth),
                }
            if args.linear_routing_ablation:
                found, latency = query_chronovec(
                    native,
                    queries,
                    args.k,
                    nprobe,
                    args.warmup_queries,
                    len(base),
                    linear_routing=True,
                )
                report["initial"]["chronovec_native"]["variants"][f"{nprobe}:linear"] = {
                    **latency,
                    **quality_summary(found, truth),
                }
    except RuntimeError as exc:
        report["initial"]["chronovec_native"] = {"skipped": str(exc)}

    if args.exact_scan:
        found, latency = query_exact_scan(base, queries, args.k, args.warmup_queries, metric)
        report["initial"]["exact_simd_scan"] = {**latency, **quality_summary(found, truth)}

    hnsw = None
    try:
        import hnswlib

        hnsw = hnswlib.Index(space="cosine" if metric == "cosine" else "l2", dim=args.dimensions)
        hnsw.init_index(
            max_elements=args.vectors,
            ef_construction=args.hnsw_ef_construction,
            M=args.hnsw_m,
            allow_replace_deleted=True,
        )
        memory_before = rss_bytes()
        started = time.perf_counter()
        hnsw.add_items(base, ids, num_threads=1)
        hnsw_build = time.perf_counter() - started
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hnsw.bin"
            hnsw.save_index(str(path))
            serialized_bytes = path.stat().st_size
        memory_after = rss_bytes()
        report["initial"]["hnswlib"] = {
            "build_s": hnsw_build,
            "build_vectors_per_s": len(base) / hnsw_build,
            "rss_before_bytes": memory_before,
            "rss_after_bytes": memory_after,
            "rss_delta_bytes": memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None,
            "serialized_bytes": serialized_bytes,
            "serialized_bytes_per_live_vector": serialized_bytes / len(base),
            "m": args.hnsw_m,
            "ef_construction": args.hnsw_ef_construction,
            "variants": {},
        }
        for ef_search in args.hnsw_ef_search:
            hnsw.set_ef(ef_search)
            found, latency = query_hnsw(hnsw, queries, args.k, args.warmup_queries)
            report["initial"]["hnswlib"]["variants"][str(ef_search)] = {
                **latency,
                **quality_summary(found, truth),
            }
    except ImportError as exc:
        report["initial"]["hnswlib"] = {"skipped": str(exc)}

    searcher = None
    scann_config = None
    try:
        import scann

        leaves = max(16, int(np.sqrt(args.vectors)))
        leaves_to_search = max(1, leaves // 20)
        reorder = max(args.k * 10, 100)
        started = time.perf_counter()
        searcher = (
            scann.scann_ops_pybind.builder(
                base, args.k, "dot_product" if metric == "cosine" else "squared_l2"
            )
            .tree(
                num_leaves=leaves,
                num_leaves_to_search=leaves_to_search,
                training_sample_size=min(args.vectors, 250_000),
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(reorder)
            .build()
        )
        scann_build = time.perf_counter() - started
        rows, latencies = [], []
        for query in queries:
            query_start = time.perf_counter()
            neighbors, _ = searcher.search(query, final_num_neighbors=args.k)
            latencies.append(time.perf_counter() - query_start)
            rows.append(ids[np.asarray(neighbors)])
        report["initial"]["scann"] = {
            "build_s": scann_build,
            "build_vectors_per_s": len(base) / scann_build,
            **latency_summary(latencies),
            **quality_summary(np.asarray(rows), truth),
            "num_leaves": leaves,
            "num_leaves_to_search": leaves_to_search,
            "reorder": reorder,
        }
        scann_config = (leaves, leaves_to_search, reorder)
    except (ImportError, AttributeError, RuntimeError) as exc:
        report["initial"]["scann"] = {"skipped": f"official ScaNN unavailable: {exc}"}

    # Replace a random subset with a deliberately shifted population.
    delete_ids = rng.choice(ids, args.churn, replace=False)
    new_ids = np.arange(args.vectors, args.vectors + args.churn, dtype=np.int64)
    if dataset_mode:
        new_vectors = base[delete_ids].copy()
        new_vectors[:, : min(8, args.dimensions)] += 0.35
        new_vectors = new_vectors + 0.03 * rng.normal(size=new_vectors.shape)
        if metric == "cosine":
            new_vectors = normalize(new_vectors)
    else:
        shifted_centers = centers.copy()
        shifted_centers[:, : min(8, args.dimensions)] += 0.8
        shifted_centers = normalize(shifted_centers)
        new_assignments = rng.integers(0, len(shifted_centers), size=args.churn)
        new_vectors = normalize(
            shifted_centers[new_assignments] + 0.18 * rng.normal(size=(args.churn, args.dimensions))
        )

    live_mask = ~np.isin(ids, delete_ids)
    live_vectors = np.vstack((base[live_mask], new_vectors))
    live_ids = np.concatenate((ids[live_mask], new_ids))
    shifted_queries = live_vectors[
        rng.choice(len(live_vectors), args.queries, replace=False)
    ] + 0.03 * rng.normal(size=(args.queries, args.dimensions))
    if metric == "cosine":
        shifted_queries = normalize(shifted_queries)
    shifted_truth = ground_truth(live_vectors, live_ids, shifted_queries, args.k, metric)
    if chrono is not None:
        started = time.perf_counter()
        reclaimed = 0
        replacement_latencies = []
        maintenance_s = 0.0
        for operation, (old_id, new_id, vector) in enumerate(
            zip(delete_ids, new_ids, new_vectors, strict=True), start=1
        ):
            replacement_started = time.perf_counter()
            chrono.delete(int(old_id))
            chrono.insert(int(new_id), vector)
            replacement_latencies.append(time.perf_counter() - replacement_started)
            if operation % args.maintenance_every == 0:
                maintenance_started = time.perf_counter()
                reclaimed += chrono.vacuum(
                    chrono.clock + 1, budget_versions=args.maintenance_budget
                )
                maintenance_s += time.perf_counter() - maintenance_started
        maintenance_started = time.perf_counter()
        reclaimed += chrono.vacuum(chrono.clock + 1)
        maintenance_s += time.perf_counter() - maintenance_started
        chrono_update_s = time.perf_counter() - started
        report["streaming"]["chronovec"] = {
            "update_s": chrono_update_s,
            "updates_per_s": float((2 * args.churn) / chrono_update_s),
            "replacement_pairs_per_s": float(args.churn / chrono_update_s),
            "replacement_latency": latency_summary(replacement_latencies),
            "maintenance_s": maintenance_s,
            "reclaimed_versions": reclaimed,
            "variants": {},
            "stats": chrono.stats(),
        }
        for nprobe in args.probes:
            found, latency = query_chronovec(
                chrono, shifted_queries, args.k, nprobe, args.warmup_queries, len(live_vectors)
            )
            report["streaming"]["chronovec"]["variants"][str(nprobe)] = {
                **latency,
                **quality_summary(found, shifted_truth),
            }

    if native is not None:
        started = time.perf_counter()
        native_reclaimed = 0
        replacement_latencies = []
        maintenance_s = 0.0
        for operation, (old_id, new_id, vector) in enumerate(
            zip(delete_ids, new_ids, new_vectors, strict=True), start=1
        ):
            replacement_started = time.perf_counter()
            native.delete(int(old_id))
            native.insert(int(new_id), vector)
            replacement_latencies.append(time.perf_counter() - replacement_started)
            if operation % args.maintenance_every == 0:
                maintenance_started = time.perf_counter()
                native_reclaimed += native.vacuum(
                    native.clock + 1, budget_versions=args.maintenance_budget
                )
                maintenance_s += time.perf_counter() - maintenance_started
        maintenance_started = time.perf_counter()
        native_reclaimed += native.vacuum(native.clock + 1)
        maintenance_s += time.perf_counter() - maintenance_started
        native_update_s = time.perf_counter() - started
        native_streaming_stats = native.stats()
        native_streaming_stats["tracked_payload_bytes"] = (
            native_streaming_stats["vector_payload_bytes"]
            + native_streaming_stats["screening_bytes"]
        )
        native_streaming_stats["tracked_payload_bytes_per_live_vector"] = native_streaming_stats[
            "tracked_payload_bytes"
        ] / max(1, native_streaming_stats["live_vectors"])
        native_streaming_stats["tracked_index_bytes_per_live_vector"] = native_streaming_stats[
            "tracked_index_bytes"
        ] / max(1, native_streaming_stats["live_vectors"])
        report["streaming"]["chronovec_native"] = {
            "update_s": native_update_s,
            "updates_per_s": float((2 * args.churn) / native_update_s),
            "replacement_pairs_per_s": float(args.churn / native_update_s),
            "replacement_latency": latency_summary(replacement_latencies),
            "maintenance_s": maintenance_s,
            "reclaimed_versions": native_reclaimed,
            "variants": {},
            "stats": native_streaming_stats,
        }
        for nprobe in args.probes:
            found, latency = query_chronovec(
                native, shifted_queries, args.k, nprobe, args.warmup_queries, len(live_vectors)
            )
            report["streaming"]["chronovec_native"]["variants"][str(nprobe)] = {
                **latency,
                **quality_summary(found, shifted_truth),
            }
            if args.adaptive_frontier:
                found, latency = query_chronovec(
                    native,
                    shifted_queries,
                    args.k,
                    nprobe,
                    args.warmup_queries,
                    len(live_vectors),
                    adaptive=True,
                )
                report["streaming"]["chronovec_native"]["variants"][f"{nprobe}:adaptive"] = {
                    **latency,
                    **quality_summary(found, shifted_truth),
                }
            if args.linear_routing_ablation:
                found, latency = query_chronovec(
                    native,
                    shifted_queries,
                    args.k,
                    nprobe,
                    args.warmup_queries,
                    len(live_vectors),
                    linear_routing=True,
                )
                report["streaming"]["chronovec_native"]["variants"][f"{nprobe}:linear"] = {
                    **latency,
                    **quality_summary(found, shifted_truth),
                }

    if hnsw is not None:
        started = time.perf_counter()
        for old_id in delete_ids:
            hnsw.mark_deleted(int(old_id))
        hnsw.add_items(new_vectors, new_ids, num_threads=1, replace_deleted=True)
        hnsw_update_s = time.perf_counter() - started
        report["streaming"]["hnswlib"] = {
            "update_s": hnsw_update_s,
            "updates_per_s": float((2 * args.churn) / hnsw_update_s),
            "replacement_pairs_per_s": float(args.churn / hnsw_update_s),
            "variants": {},
            "note": "hnswlib deleted-slot reuse is enabled; this is the stronger bounded-capacity baseline",
        }
        for ef_search in args.hnsw_ef_search:
            hnsw.set_ef(ef_search)
            found, latency = query_hnsw(hnsw, shifted_queries, args.k, args.warmup_queries)
            report["streaming"]["hnswlib"]["variants"][str(ef_search)] = {
                **latency,
                **quality_summary(found, shifted_truth),
            }

    if searcher is not None and scann_config is not None:
        leaves, leaves_to_search, reorder = scann_config
        started = time.perf_counter()
        rebuilt = (
            scann.scann_ops_pybind.builder(
                live_vectors, args.k, "dot_product" if metric == "cosine" else "squared_l2"
            )
            .tree(
                num_leaves=leaves,
                num_leaves_to_search=leaves_to_search,
                training_sample_size=min(len(live_vectors), 250_000),
            )
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(reorder)
            .build()
        )
        rebuild_s = time.perf_counter() - started
        rows, latencies = [], []
        for query in shifted_queries:
            query_start = time.perf_counter()
            neighbors, _ = rebuilt.search(query, final_num_neighbors=args.k)
            latencies.append(time.perf_counter() - query_start)
            rows.append(live_ids[np.asarray(neighbors)])
        report["streaming"]["scann"] = {
            "rebuild_s": rebuild_s,
            **latency_summary(latencies),
            **quality_summary(np.asarray(rows), shifted_truth),
            "note": "ScaNN has no in-place delete/update path here; the live index was rebuilt",
        }

    report["notes"].append(
        "chronovec is the NumPy reference; chronovec_native, hnswlib, and ScaNN use native compiled kernels."
    )
    if dataset_mode:
        truth_source = (
            "published neighbors"
            if not args.dataset_limit
            else "exact truth recomputed for the limited train subset"
        )
        report["notes"].append(f"Dataset mode: {args.dataset.name}; {truth_source}.")
    report["notes"].append(
        "Results establish behavior and bottlenecks, not production superiority."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=20_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--churn", type=int, default=4_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--page-capacity", type=int, default=256)
    parser.add_argument("--probes", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=100)
    parser.add_argument("--hnsw-ef-search", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--maintenance-every", type=int, default=64)
    parser.add_argument("--maintenance-budget", type=int, default=64)
    parser.add_argument("--disable-native-screening", action="store_true")
    parser.add_argument("--native-rerank-factor", type=int, default=4)
    parser.add_argument(
        "--adaptive-frontier", action="store_true", help="also benchmark safe page-radius pruning"
    )
    parser.add_argument(
        "--exact-scan", action="store_true", help="include the native SIMD brute-force baseline"
    )
    parser.add_argument(
        "--checkpoint-benchmark",
        action="store_true",
        help="measure native checkpoint size and direct recovery",
    )
    parser.add_argument(
        "--linear-routing-ablation",
        action="store_true",
        help="compare graph routing with an exact all-centroid page frontier",
    )
    parser.add_argument("--warmup-queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        help="ann-benchmarks HDF5 angular dataset; uses all train vectors and published truth",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=0,
        help="limit train vectors and recompute exact truth (0 uses the full dataset)",
    )
    parser.add_argument(
        "--skip-reference", action="store_true", help="skip the slow NumPy reference implementation"
    )
    args = parser.parse_args()
    if args.dataset is None and args.churn >= args.vectors:
        parser.error("--churn must be smaller than --vectors")
    report = benchmark(args)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
