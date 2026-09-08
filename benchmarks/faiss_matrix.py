"""ChronoVec against faiss-ivfflat, across widths, scales, and every operation.

Only two engines, so the matrix can be dense rather than broad. Both are pinned
to one thread: faiss takes every OpenMP thread it can find otherwise, and an
unpinned comparison measures core count.

Batched and single-item are reported apart because they are different products.
An ingestion job only cares about the first; an agent writing one memory per
turn only ever gets the second.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import time

import numpy as np


def clustered(n: int, dim: int, seed: int = 7, clusters: int = 256) -> np.ndarray:
    """Clustered vectors, not uniform.

    Uniform Gaussians have no neighbourhood structure at high d: every point is
    roughly equidistant, every engine scores about 0.55 recall, and the
    measurement becomes the curse of dimensionality rather than the index.

    The cluster count is fixed and does not scale with n. An earlier version
    planted n/500 clusters, which quietly rigged the comparison: faiss was
    given nlist = sqrt(live), so past about 30k vectors there were more true
    clusters than inverted lists, k-means had to merge distinct clusters into
    one list, and faiss's recall per probe collapsed. That produced a 13x
    "win" at 500k that reversed to a 0.45x loss at 10k -- the ratio was
    tracking the planted cluster count, not the engines.
    """
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dim)).astype(np.float32) * 3.0
    who = rng.integers(0, clusters, size=n)
    return (centres[who] + rng.normal(scale=1.0, size=(n, dim))).astype(np.float32)


def metric_of(path: str) -> str:
    """ann-benchmarks names carry the metric, so it is read rather than guessed."""
    name = str(path).lower()
    return "cosine" if "angular" in name else "l2"


def real_dataset(path: str, dim: int, live: int, queries: int, metric: str = "l2"):
    """Real vectors, which is what the headline comparison should use.

    Synthetic data can flatter either engine depending on how its structure
    lines up with a fixed partition count. Real vectors cannot be tuned by
    accident.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        need = int(live * 1.3)
        base = np.asarray(handle["train"][:need], dtype=np.float32)
        test = np.asarray(handle["test"][:queries], dtype=np.float32)
    if base.shape[1] != dim:
        raise ValueError(f"{path} is d={base.shape[1]}, expected {dim}")
    if metric == "cosine":
        base, test = normalise(base), normalise(test)
    return base, test


def normalise(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (values / norms).astype(np.float32)


def truth_for(base, queries, k, metric="l2"):
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], 64):
        stop = min(start + 64, queries.shape[0])
        block = queries[start:stop]
        if metric == "cosine":
            # Unit vectors, so ranking by -inner product is ranking by cosine
            # distance, and matches what both engines are asked for.
            scores = -(block @ base.T)
        else:
            scores = (
                (block * block).sum(1)[:, None]
                - 2.0 * (block @ base.T)
                + (base * base).sum(1)[None, :]
            )
        cols = np.argpartition(scores, k, axis=1)[:, :k]
        rows = np.arange(stop - start)[:, None]
        out[start:stop] = cols[rows, np.argsort(scores[rows, cols], axis=1)]
    return out


class Chrono:
    name = "chronovec"
    # Mapped and still writable, which is the difference from faiss here.
    writable_on_disk = True

    def __init__(
        self,
        dim,
        live,
        nprobe,
        storage="heap",
        lanes=1,
        workspace=None,
        page_capacity=256,
        metric="l2",
        **_,
    ):
        from chronovec import Index

        options = {}
        if storage == "mmap":
            options["arena_path"] = str(workspace / "chronovec.arena")
            # Sized for the pre-consolidation peak, which is several times the
            # live count on data that splits a lot.
            options["max_vectors"] = live * 6
        self.page_capacity = page_capacity
        self.ix = Index(
            dim, metric=metric, page_capacity=page_capacity, nprobe=nprobe, threads=lanes, **options
        )
        self.nprobe = nprobe

    def build(self, ids, vectors):
        self.ix.insert_many(ids, vectors)

    def insert_many(self, ids, vectors):
        self.ix.insert_many(ids, vectors)

    def delete_many(self, ids):
        self.ix.delete_many(ids)

    def update_many(self, ids, vectors):
        self.ix.insert_many(ids, vectors)

    def insert_one(self, i, v):
        self.ix.insert(int(i), v)

    def delete_one(self, i):
        self.ix.delete(int(i))

    def search(self, q, k, probe):
        return [r.id for r in self.ix.search(q, k, nprobe=probe)]

    def search_batch(self, block, k, probe):
        ids, _ = self.ix.search_many(block, k, nprobe=probe, as_arrays=True)
        return ids

    def maintain(self):
        self.ix.vacuum(oldest_snapshot=self.ix.clock + 1, budget_versions=0)

    def probes(self):
        return (8, 16, 32, 64, 96, 128, 192)

    def seal(self):
        pass  # already mapped, and stays writable

    def close(self):
        self.ix.close()


class Faiss:
    name = "faiss-ivfflat"

    # An mmap-backed faiss index is read-only, and a write to one does not
    # raise: it aborts the process with an uncaught C++ exception. So writes
    # are not attempted in that configuration rather than caught.
    writable_on_disk = False

    def __init__(
        self, dim, live, nprobe, nlist=0, storage="heap", lanes=1, workspace=None, metric="l2", **_
    ):
        import faiss

        # One OpenMP thread always. Concurrency here comes from concurrent
        # queries, which is what a serving workload produces and what both
        # engines can do; letting faiss also fan a single query across cores
        # would be comparing two different things.
        faiss.omp_set_num_threads(1)
        self.storage = storage
        self.workspace = workspace
        self.faiss = faiss
        # nlist is swept rather than fixed at sqrt(live). Whether faiss has
        # more or fewer lists than the data has clusters dominates its
        # recall-per-probe, so pinning it to one guess measures the guess.
        self.nlist = nlist or max(16, int(np.sqrt(live)))
        # Cosine is inner product over unit vectors, which the loader has
        # already normalised; using L2 there would rank by a different metric
        # than the ground truth is computed in.
        inner = metric == "cosine"
        quantizer = faiss.IndexFlatIP(dim) if inner else faiss.IndexFlatL2(dim)
        self.ix = faiss.IndexIVFFlat(
            quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT if inner else faiss.METRIC_L2
        )
        self.ix.nprobe = nprobe

    def build(self, ids, vectors):
        if not self.ix.is_trained:
            self.ix.train(vectors)
        self.ix.add_with_ids(vectors, ids.astype(np.int64))

    def insert_many(self, ids, vectors):
        self.ix.add_with_ids(vectors, ids.astype(np.int64))

    def delete_many(self, ids):
        self.ix.remove_ids(self.faiss.IDSelectorBatch(ids.astype(np.int64)))

    def update_many(self, ids, vectors):  # no native update
        self.delete_many(ids)
        self.insert_many(ids, vectors)

    def insert_one(self, i, v):
        self.ix.add_with_ids(v[None, :], np.array([i], dtype=np.int64))

    def delete_one(self, i):
        self.ix.remove_ids(self.faiss.IDSelectorBatch(np.array([i], dtype=np.int64)))

    def search(self, q, k, probe):
        self.ix.nprobe = probe
        _, labels = self.ix.search(q[None, :], k)
        return [int(x) for x in labels[0] if x >= 0]

    def search_batch(self, block, k, probe):
        self.ix.nprobe = probe
        _, labels = self.ix.search(block, k)
        return labels

    def maintain(self):
        pass

    def probes(self):
        return (1, 2, 4, 8, 16, 32, 64, 128)

    def seal(self):
        """Move the built index onto disk, after which it is read-only."""
        if self.storage != "mmap":
            return
        path = str(self.workspace / "faiss.index")
        self.faiss.write_index(self.ix, path)
        self.ix = self.faiss.read_index(path, self.faiss.IO_FLAG_MMAP)

    def close(self):
        pass


def timed(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


def frontier(curve, field="qps"):
    """Best qps seen at or above each recall target, over all configurations."""
    out = {}
    for target in (0.90, 0.95, 0.99):
        reached = [p for p in curve if p["recall"] >= target]
        out[f"{target:.2f}"] = max(p[field] for p in reached) if reached else None
    return out


def concurrent_qps(engine, queries, k, probe, workers, batched):
    """Queries served per second with `workers` issuing them at once.

    Reported both one query at a time and as a batch, because they are
    different workloads and the engines differ enormously between them. Online
    serving gets one query at a time; an offline scoring job can hand over the
    whole set. Measuring faiss only per-query understates it by 2.8x -- its
    Python binding costs about 26us a call against an engine time of 15us --
    and measuring only batched hides that our binding costs 8us against 53us,
    so the per-call path is where we are relatively strongest.

    Concurrency is concurrent queries rather than one query fanned across
    cores: that is what a serving workload produces and both engines can do it.
    """

    def serve(rows):
        if batched:
            engine.search_batch(queries[rows], k, probe)
        else:
            for row in rows:
                engine.search(queries[row], k, probe)

    if workers <= 1:
        started = time.perf_counter()
        serve(np.arange(queries.shape[0]))
        return queries.shape[0] / (time.perf_counter() - started)

    import threading

    chunks = np.array_split(np.arange(queries.shape[0]), workers)
    threads = [threading.Thread(target=serve, args=(chunk,)) for chunk in chunks]
    started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return queries.shape[0] / (time.perf_counter() - started)


def measure(
    engine_class,
    dim,
    live,
    pool,
    queries,
    truth,
    k,
    single_n,
    storage="heap",
    workers=1,
    workspace=None,
    metric="l2",
    **kwargs,
):
    engine = engine_class(
        dim, live, 64, storage=storage, lanes=workers, workspace=workspace, metric=metric, **kwargs
    )
    out = {"engine": engine.name, "storage": storage, "workers": workers, "metric": metric}
    out.update(kwargs)

    ids = np.arange(live)
    out["build_ops_s"] = live / timed(engine.build, ids, pool[:live])
    engine.maintain()
    # After this point an engine that is read-only on disk is read-only.
    engine.seal()
    writable = storage != "mmap" or engine.writable_on_disk
    out["writable"] = writable

    # Recall/qps curve, so throughput is only ever quoted at a matched recall.
    curve = []
    for probe in engine.probes():
        for q in queries[:20]:
            engine.search(q, k, probe)  # warm
        # Recall is measured sequentially; throughput at the requested
        # concurrency. Mixing them would charge recall accounting to the
        # threaded timing.
        hits = 0.0
        for index in range(queries.shape[0]):
            found = engine.search(queries[index], k, probe)
            hits += len(set(found) & set(truth[index].tolist())) / k
        single = max(concurrent_qps(engine, queries, k, probe, workers, False) for _ in range(2))
        batch = max(concurrent_qps(engine, queries, k, probe, workers, True) for _ in range(2))
        curve.append(
            {"probe": probe, "recall": hits / queries.shape[0], "qps": single, "qps_batched": batch}
        )
    out["curve"] = curve
    if not writable:
        engine.close()
        return out

    batch = max(1, live // 10)
    fresh_ids = np.arange(live, live + batch)
    out["insert_batch_ops_s"] = batch / timed(
        engine.insert_many, fresh_ids, pool[live : live + batch]
    )
    out["update_batch_ops_s"] = batch / timed(
        engine.update_many, ids[:batch], pool[live : live + batch]
    )
    out["delete_batch_ops_s"] = batch / timed(engine.delete_many, fresh_ids)
    start = time.perf_counter()
    engine.delete_many(ids[:batch])
    engine.insert_many(np.arange(live * 2, live * 2 + batch), pool[:batch])
    out["replace_batch_ops_s"] = (2 * batch) / (time.perf_counter() - start)
    engine.maintain()

    # Single-item, on a bounded sample: the per-call path, not the bulk one.
    n = min(single_n, batch)
    single_ids = np.arange(live * 3, live * 3 + n)
    start = time.perf_counter()
    for position in range(n):
        engine.insert_one(int(single_ids[position]), pool[position])
    out["insert_single_ops_s"] = n / (time.perf_counter() - start)
    start = time.perf_counter()
    for position in range(n):
        engine.delete_one(int(single_ids[position]))
    out["delete_single_ops_s"] = n / (time.perf_counter() - start)
    engine.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", type=int, nargs="+", default=[128, 384, 768])
    parser.add_argument("--scales", type=int, nargs="+", default=[10_000, 100_000, 500_000])
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--single", type=int, default=2000)
    parser.add_argument("--output", default="benchmarks/faiss_matrix.json")
    parser.add_argument("--dataset", default="", help="HDF5 file of real vectors; overrides --dims")
    parser.add_argument(
        "--storage",
        nargs="+",
        default=["heap", "mmap"],
        choices=["heap", "mmap"],
        help="resident or memory-mapped payload",
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[1, 4], help="concurrent query threads"
    )
    args = parser.parse_args()

    report = {"k": args.k, "queries": args.queries, "threads": 1, "cells": []}
    for dim in args.dims:
        for live in args.scales:
            metric = metric_of(args.dataset) if args.dataset else "l2"
            if args.dataset:
                pool, queries = real_dataset(args.dataset, dim, live, args.queries, metric)
                if pool.shape[0] < live:
                    print(f"skipping n={live}: dataset has {pool.shape[0]} rows")
                    continue
            else:
                pool = clustered(int(live * 2.5), dim)
                queries = clustered(args.queries, dim, seed=99)
            truth = truth_for(pool[:live], queries, args.k, metric)
            root = int(np.sqrt(live))
            # Sweep our page capacity as well as faiss's list count. Tuning one
            # side and pinning the other was the same bias in reverse: every
            # faiss cell was its best configuration and ours was a single
            # guess.
            plans = [(Chrono, {"page_capacity": capacity}) for capacity in (128, 256, 512, 1024)]
            # Sweep faiss's list count. Which side of the data's own cluster
            # count it falls on dominates its recall per probe, so one guess
            # measures the guess rather than the engine.
            for factor in (1, 2, 4, 8):
                plans.append((Faiss, {"nlist": max(16, root * factor)}))
            # Every combination of storage and concurrency, for both engines,
            # so neither is compared only in the configuration that suits it.
            configurations = [
                (storage, workers) for storage in args.storage for workers in args.workers
            ]
            for storage, workers in configurations:
                for engine_class, kwargs in plans:
                    workspace = pathlib.Path(tempfile.mkdtemp(prefix="cvbench-"))
                    started = time.perf_counter()
                    cell = measure(
                        engine_class,
                        dim,
                        live,
                        pool,
                        queries,
                        truth,
                        args.k,
                        args.single,
                        storage=storage,
                        workers=workers,
                        workspace=workspace,
                        metric=metric,
                        **kwargs,
                    )
                    shutil.rmtree(workspace, ignore_errors=True)
                    cell.update(dim=dim, live=live, seconds=round(time.perf_counter() - started, 1))
                    report["cells"].append(cell)
                    tag = (
                        f"{cell['engine']}(nlist={cell['nlist']})"
                        if "nlist" in cell
                        else f"{cell['engine']}(cap={cell['page_capacity']})"
                    )
                    reached = frontier(cell["curve"])
                    at95 = reached["0.95"]
                    writes = (
                        f"{cell['insert_batch_ops_s']:>10,.0f}/s"
                        if cell.get("writable")
                        else "  read-only"
                    )
                    print(
                        f"d={dim} n={live} {storage:>5}/{workers}t "
                        f"{tag:26s} qps@.95="
                        f"{(f'{at95:,.0f}' if at95 else 'unreached'):>10} "
                        f"ins_batch={writes} ({cell['seconds']}s)",
                        flush=True,
                    )
            with open(args.output, "w") as handle:
                json.dump(report, handle, indent=1)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
