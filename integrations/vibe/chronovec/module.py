"""ChronoVec for VIBE (https://github.com/vector-index-bench/vibe).

VIBE measures build and static query. That is deliberately not what ChronoVec
is built for -- its subject is an index that keeps changing -- but the static
number is the one everybody runs, so it belongs in the standard harness rather
than in one written by this project. The mutation, concurrency, disk and
snapshot axes VIBE does not cover are measured separately by `streambench`.

Copy this directory to `vibe/algorithms/chronovec/`.
"""

import numpy as np

from ..base.module import BaseANN


class ChronoVec(BaseANN):
    """A page-structured MVCC index, measured here with none of that exercised.

    `page_capacity` is the build parameter and `nprobe` the query one. The
    capacity is swept rather than fixed: how it sits against the data's own
    cluster structure moves recall per probe substantially, and pinning it
    would report the guess rather than the index.
    """

    def __init__(self, metric, page_capacity=256, rerank_factor=8):
        # VIBE's "ip" and "normalized" are inner product over unit vectors,
        # which is what cosine is here.
        self.metric = {
            "ip": "cosine",
            "normalized": "cosine",
            "cosine": "cosine",
            "euclidean": "l2",
        }[metric]
        self.page_capacity = page_capacity
        self.rerank_factor = rerank_factor
        self.nprobe = 64

    def fit(self, X):
        from chronovec import Index

        X = np.ascontiguousarray(X, dtype=np.float32)
        self.index = Index(
            X.shape[1],
            metric=self.metric,
            page_capacity=self.page_capacity,
            nprobe=self.nprobe,
            rerank_factor=self.rerank_factor,
        )
        self.index.insert_many(np.arange(X.shape[0], dtype=np.int64), X)
        # A bulk load leaves the pages its splits produced. Consolidating is
        # part of building, not part of querying, so it happens here where the
        # build timer covers it.
        self.index.vacuum(oldest_snapshot=self.index.clock + 1,
                          budget_versions=0)

    def set_query_arguments(self, nprobe):
        self.nprobe = nprobe

    def query(self, v, n):
        found = self.index.search(np.asarray(v, dtype=np.float32), n,
                                  nprobe=self.nprobe)
        return np.array([r.id for r in found], dtype=np.int64)

    def batch_query(self, X, n):
        # One crossing for the whole set rather than one per query, which is
        # worth about 12% here. Not threaded: the base class would use a
        # ThreadPool, and VIBE runs single-threaded comparisons.
        ids, _ = self.index.search_many(
            np.ascontiguousarray(X, dtype=np.float32), n,
            nprobe=self.nprobe, as_arrays=True)
        self.res = ids

    def get_batch_results(self):
        return np.array([[i for i in row if i != -1] for row in self.res])

    def get_additional(self):
        stats = self.index.stats()
        return {
            "pages": stats["pages"],
            "capacity_amplification": stats["capacity_amplification"],
            "tracked_index_bytes": stats["tracked_index_bytes"],
        }

    def __str__(self):
        return "ChronoVec(page_capacity=%d, rerank_factor=%d, nprobe=%d)" % (
            self.page_capacity, self.rerank_factor, self.nprobe)
