"""Engine adapters.

Adding an engine means implementing :class:`Engine` and registering it. Declare
capabilities honestly: an engine that cannot delete is not disqualified, it is
made to rebuild, and the rebuild cost appears in the report where a reader can
see it.
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Engine(Protocol):
    name: str
    can_delete: bool          # supports removing a vector without a rebuild
    needs_rebuild: bool       # must rebuild the whole index to express a turnover

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None: ...
    # Bulk turnover: the whole epoch's delta at once.
    def apply(self, retired: np.ndarray, new_ids: np.ndarray,
              new_vectors: np.ndarray, live_ids: np.ndarray,
              live_vectors: np.ndarray) -> None: ...
    # Interleaved turnover: one replacement at a time, which is what agent
    # memory and change-data-capture actually look like. The two modes rank
    # engines differently -- FAISS removes in one O(N) pass over its inverted
    # lists, which is superb in bulk and expensive per call -- so an engine
    # should be judged on the mode matching its intended workload.
    def replace_one(self, retired_id: int, new_id: int,
                    vector: np.ndarray) -> None: ...
    # The individual operations. Replacement is only one of the shapes a
    # changing corpus produces; growth, shrink, and rewrite-under-the-same-id
    # are the others, and they are not interchangeable.
    def insert_many(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...
    def delete_many(self, ids: np.ndarray) -> None: ...
    def update_many(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...
    # Reclamation, timed apart from the operation that created the garbage.
    def maintain(self) -> int: ...
    def search(self, query: np.ndarray, k: int) -> list[int]: ...
    def stats(self) -> dict[str, Any]: ...
    def close(self) -> None: ...


class _Base:
    # Threads the engine may use. Defaults to one for everybody, because the
    # engines do not agree on a default and the disagreement is invisible:
    # faiss takes every OpenMP thread it can find (ten on this machine),
    # usearch's threads=0 means all cores, hnswlib defaults to all but was
    # pinned to one here, and chronovec serialises writers behind a mutex.
    # Comparing those against each other measures core count, not indexes.
    threads = 1

    can_delete = True
    needs_rebuild = False
    supports_interleaved = True
    # Whether the engine offers a real update: rewrite the vector under an id
    # that already exists. Most do not, and emulate it as delete plus insert.
    # That is a legitimate implementation, but it is a different cost and a
    # different consistency story -- there is a window with no vector under the
    # id -- so it is reported rather than hidden.
    can_update_in_place = False
    can_grow = True     # accepts ids beyond the initial live-set size
    # Whether the engine claims searches may run while another thread mutates.
    # Off by default: most ANN libraries document concurrent reads as safe and
    # concurrent read-with-write as not, and a wrong claim here is a crash or a
    # torn result, not a slow number. Engines run isolated, so an engine that
    # claims this and then aborts is reported as aborting.
    supports_concurrent_reads = False
    # Whether a reader can pin a point in time and keep reading it while the
    # index moves on. Almost nothing else offers this, so it is reported as a
    # capability rather than scored -- and its cost is measured, because a
    # pinned snapshot is exactly what stops an MVCC index reclaiming space.
    supports_snapshots = False
    # Whether the engine can restrict a search to records carrying an
    # attribute, and whether it does so inside the search or by over-fetching
    # and discarding afterwards. The difference decides whether a selective
    # filter costs more work or less, so it is reported rather than assumed.
    filtering = "none"        # none | native | post

    def build_filtered(self, ids, vectors, tags) -> None:
        """Load with one integer tag per record. Default: filtering unsupported."""
        raise NotImplementedError(f"{self.name} cannot filter a search")

    def search_filtered(self, query: np.ndarray, k: int, tag: int) -> list[int]:
        raise NotImplementedError(f"{self.name} cannot filter a search")

    # Mutation of a *tagged* corpus. A filtered read on a static index is the
    # easy case; the question that matters is whether the filter still pays
    # once the corpus has been churned, because every engine reorganises under
    # mutation and the reorganisation decides whether records sharing a tag
    # stay grouped tightly enough to skip in bulk.
    def insert_filtered(self, ids, vectors, tags) -> None:
        raise NotImplementedError(f"{self.name} cannot insert a tagged record")

    def delete_filtered(self, ids) -> None:
        """Remove records and forget their tags."""
        self.delete_many(np.asarray(ids, dtype=np.int64))

    # How wide a post-filtering engine has to cast its net. Fixing this at a
    # constant is the easy way to make post-filtering look broken: at 1% a
    # constant 16x over-fetch returns nothing usable, but that is the harness
    # under-fetching, not the engine failing. A caller who knows the
    # selectivity sizes the fetch as k/selectivity plus a margin for the
    # matches not being uniformly distributed among the neighbours, which is
    # what a competent integration does. The engine is then correct, and what
    # the comparison measures is what that correctness costs.
    def set_filter_selectivity(self, fraction: float, k: int,
                               corpus: int) -> None:
        """Tell the engine how selective the coming filter is. Native: unused."""

    # Reported alongside the latency so the fetch width behind a post-filtered
    # number is visible rather than implied.
    def filter_fetch_width(self) -> int:
        return 0

    # The knob that trades recall for query speed. Comparing engines at their
    # own defaults compares tuning choices, not engines: an engine set to a
    # cheaper operating point looks faster and is simply answering a different
    # question. Sweeping this is what makes a throughput number mean something.
    search_ladder: tuple[int, ...] = ()

    def set_search_param(self, value: int) -> None:
        raise NotImplementedError(f"{self.name} has no tunable search parameter")

    def snapshot(self) -> int:
        raise NotImplementedError(f"{self.name} has no snapshots")

    def search_at(self, query: np.ndarray, k: int, snapshot: int) -> list[int]:
        raise NotImplementedError(f"{self.name} has no snapshots")

    def retain(self, snapshot: int | None) -> None:
        """Hold `snapshot` readable, blocking reclamation of anything it needs."""
        raise NotImplementedError(f"{self.name} has no snapshots")

    # Whether the engine exposes a genuine batch write, or whether its "batch"
    # API is a loop. Batching is where most of the difference between engines
    # on write throughput comes from, so it is reported, not assumed.
    batched_writes = True

    def insert_many(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Add ids that do not exist yet. Default is the build path."""
        self.build(vectors, ids)

    def insert_one(self, one: int, vector: np.ndarray) -> None:
        """A single insert through whatever single-item path the engine has.

        Engines that only offer a bulk API pay a batch of one here, which is
        the honest cost of using them one record at a time.
        """
        self.insert_many(np.array([one], dtype=np.int64), vector[None, :])

    def delete_one(self, one: int) -> None:
        self.delete_many(np.array([one], dtype=np.int64))

    def delete_many(self, ids: np.ndarray) -> None:
        raise NotImplementedError(f"{self.name} cannot delete")

    def update_many(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        """Rewrite the vector under existing ids.

        The default emulates it, which is what an engine without a real update
        forces a caller to write. `can_update_in_place` says which happened.
        """
        self.delete_many(ids)
        self.insert_many(ids, vectors)

    def maintain(self) -> int:
        """Reclaim space made dead by the last operation. Returns units freed.

        Timed apart from the operation itself. Engines differ in *when* they pay
        for a delete: hnswlib marks a tombstone and moves on, which is why its
        delete is nearly free and its space is not. Charging reclamation to
        whichever phase happens to trigger it would compare bookkeeping
        schedules rather than engines.
        """
        return 0

    def replace_one(self, retired_id: int, new_id: int, vector: np.ndarray) -> None:
        """One replacement. Default routes through the bulk path with a batch of
        one, which is the honest cost for an engine offering only bulk APIs."""
        self.apply(np.array([retired_id], dtype=np.int64),
                   np.array([new_id], dtype=np.int64),
                   vector[None, :], None, None)

    def stats(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        pass


class ChronoVecEngine(_Base):
    name = "chronovec"
    can_update_in_place = True
    # Reaches past 256 because a filtered search on a large index needs it:
    # at 500k with a 10% filter over 1186 eligible pages, recall runs 0.75,
    # 0.87, 0.95, 0.99 at 64, 128, 256, 512. Stopping at 256 reported that
    # workload as an engine that could not reach the target, when it was the
    # ladder that could not reach it.
    search_ladder = (4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
                     1024)
    # Readers take a snapshot and never a lock; this is the design claim the
    # whole index exists to make, so it is the thing most worth testing.
    supports_concurrent_reads = True
    supports_snapshots = True
    # Each page carries the union of its records' labels, so a page that
    # cannot satisfy the filter is skipped without touching a record.
    filtering = "native"

    def __init__(self, dim: int, metric: str, nprobe: int = 96,
                 page_capacity: int = 256, maintenance_every: int = 64,
                 maintenance_budget: int = 64, threads: int = 1,
                 label_partition: bool = False,
                 **_: Any) -> None:
        # Writers serialise behind one mutex, so the write path is
        # single-threaded whatever is asked for. Recorded, not silently ignored.
        self.threads = int(threads)
        from chronovec import Index
        self._dimensions, self._metric = dim, metric
        self._page_capacity = page_capacity
        # Only used by build_filtered: strict mode is a filtered-search
        # question, unlabelled builds have no partition to keep pure.
        self._label_partition = bool(label_partition)
        if self._label_partition:
            self.filtering = "native-part"
        self._index = Index(dim, metric=metric, page_capacity=page_capacity,
                            nprobe=nprobe)
        self._nprobe = nprobe
        self._every = maintenance_every
        self._budget = maintenance_budget
        self.reclaimed = 0
        self._retained: int | None = None

    def build(self, vectors, ids):
        self._index.insert_many(ids, vectors)

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        # Bulk delete then bulk insert, matching the shape of the batch APIs the
        # other engines expose, so the comparison is of engines and not of
        # binding overhead. Reclamation is not done here: it is `maintain`, and
        # the runner times it separately.
        self._index.delete_many(retired)
        self._index.insert_many(new_ids, new_vectors)

    def maintain(self):
        # Anything a retained snapshot can still see must survive. With nothing
        # retained the horizon is the present and every dead version goes.
        horizon = self._retained if self._retained is not None \
            else self._index.clock + 1
        freed = self._index.vacuum(oldest_snapshot=horizon, budget_versions=0)
        self.reclaimed += freed
        return freed

    def snapshot(self):
        return int(self._index.clock)

    def search_at(self, query, k, snapshot):
        return [r.id for r in self._index.search(query, k=k, nprobe=self._nprobe,
                                                 snapshot=snapshot)]

    def retain(self, snapshot):
        self._retained = snapshot

    def replace_one(self, retired_id, new_id, vector):
        self._index.delete(int(retired_id))
        self._index.insert(int(new_id), vector)

    def set_search_param(self, value):
        self._nprobe = int(value)

    def build_filtered(self, ids, vectors, tags):
        from chronovec import Index
        self._index.close()
        self._index = Index(self._dimensions, metric=self._metric,
                            page_capacity=self._page_capacity,
                            nprobe=self._nprobe, labels=True,
                            label_partition=self._label_partition)
        self._index.insert_many(ids, vectors,
                                labels=np.asarray(tags, dtype=np.uint64))

    def search_filtered(self, query, k, tag):
        return [r.id for r in self._index.search(
            query, k=k, nprobe=self._nprobe, require_all=int(tag))]

    def insert_filtered(self, ids, vectors, tags):
        self._index.insert_many(np.asarray(ids, dtype=np.int64), vectors,
                                labels=np.asarray(tags, dtype=np.uint64))

    def insert_one(self, one, vector):
        self._index.insert(int(one), vector)

    def delete_one(self, one):
        self._index.delete(int(one))

    def insert_many(self, ids, vectors):
        self._index.insert_many(ids, vectors)

    def delete_many(self, ids):
        self._index.delete_many(ids)

    def update_many(self, ids, vectors):
        # insert_many on an existing id closes the old version and opens a new
        # one under the same id, in one operation. The old version stays
        # readable from a snapshot taken before the write.
        self._index.insert_many(ids, vectors)

    def search(self, query, k):
        return [r.id for r in self._index.search(query, k=k, nprobe=self._nprobe)]

    def stats(self):
        s = self._index.stats()
        live = s.get("live_vectors") or 1
        tracked = s.get("tracked_index_bytes") or 0
        return {"pages": s.get("pages"),
                "capacity_amplification": s.get("capacity_amplification"),
                # Derived here: the index reports totals, and there is no
                # per-vector key. Asking for one returned None and the report
                # printed an empty column instead of a number.
                "bytes_per_live_vector": tracked / live,
                "tracked_index_bytes": tracked,
                # Versions kept alive because something can still read them.
                # This is what a pinned snapshot actually costs.
                "retained_versions": s.get("retained_versions"),
                "reclaimed_versions": self.reclaimed}

    def close(self):
        self._index.close()


class HnswlibEngine(_Base):
    name = "hnswlib"
    can_update_in_place = True
    # knn_query takes a filter callable, applied per candidate during the
    # graph walk rather than afterwards.
    filtering = "native"
    # add_items takes an array but inserts one node at a time: measured batch
    # speedup over its own single-item path is 1.0x, against 5.3x for chronovec
    # and 11.2x for faiss. The API is batched; the write is not.
    batched_writes = False
    search_ladder = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)

    def __init__(self, dim: int, metric: str, live: int, m: int = 16,
                 ef_construction: int = 100, ef_search: int = 128,
                 threads: int = 1, **_: Any) -> None:
        self.threads = int(threads)
        import hnswlib
        self._index = hnswlib.Index(space="cosine" if metric == "cosine" else "l2", dim=dim)
        self._index.init_index(max_elements=live, M=m, ef_construction=ef_construction,
                               random_seed=42, allow_replace_deleted=True)
        self._index.set_ef(ef_search)

    def build(self, vectors, ids):
        self._index.add_items(vectors, ids, num_threads=self.threads)

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        for old in retired:
            self._index.mark_deleted(int(old))
        self._index.add_items(new_vectors, new_ids, num_threads=self.threads, replace_deleted=True)

    def replace_one(self, retired_id, new_id, vector):
        self._index.mark_deleted(int(retired_id))
        self._index.add_items(vector[None, :], np.array([new_id]), num_threads=self.threads,
                              replace_deleted=True)

    def search(self, query, k):
        labels, _ = self._index.knn_query(query[None, :], k=k)
        return [int(x) for x in labels[0]]

    def set_search_param(self, value):
        self._index.set_ef(int(value))

    def insert_many(self, ids, vectors):
        # max_elements was sized for the initial live set, so growth costs a
        # resize. That is a real cost of the engine and is left in the timing.
        needed = self._index.get_current_count() + len(ids)
        if needed > self._index.get_max_elements():
            self._index.resize_index(int(needed * 1.2) + 1)
        self._index.add_items(vectors, ids, num_threads=self.threads, replace_deleted=True)

    def delete_many(self, ids):
        for one in ids:
            self._index.mark_deleted(int(one))

    def update_many(self, ids, vectors):
        # hnswlib overwrites an existing label in place on add_items.
        self._index.add_items(vectors, ids, num_threads=self.threads)

    def build_filtered(self, ids, vectors, tags):
        self._tags = {int(i): int(t) for i, t in zip(ids, tags)}
        self.build(vectors, ids)

    def search_filtered(self, query, k, tag):
        try:
            labels, _ = self._index.knn_query(
                query[None, :], k=k,
                filter=lambda i: self._tags.get(int(i)) == int(tag))
        except TypeError as unsupported:      # older wheels have no filter=
            raise NotImplementedError(
                "this hnswlib build has no filter callback") from unsupported
        return [int(x) for x in labels[0]]

    def insert_filtered(self, ids, vectors, tags):
        for one, tag in zip(ids, tags):
            self._tags[int(one)] = int(tag)
        self.insert_many(np.asarray(ids, dtype=np.int64), vectors)

    def delete_filtered(self, ids):
        for one in ids:
            self._tags.pop(int(one), None)
        self.delete_many(np.asarray(ids, dtype=np.int64))

    def stats(self):
        return {"note": "deleted-slot reuse enabled"}


class FaissFlatEngine(_Base):
    """IVF-Flat with id mapping: FAISS's dynamic configuration."""
    name = "faiss-ivfflat"
    filtering = "post"
    search_ladder = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128)

    def __init__(self, dim: int, metric: str, live: int, nlist: int = 0,
                 nprobe: int = 32, threads: int = 1, **_: Any) -> None:
        import faiss
        self.threads = int(threads)
        # Without this faiss quietly uses every core, which is where its
        # apparent insert advantage over single-threaded engines came from.
        faiss.omp_set_num_threads(self.threads)
        self._faiss = faiss
        self._metric = metric
        nlist = nlist or max(16, int(np.sqrt(live)))
        quantizer = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
        # IVF carries ids natively. Wrapping it in IndexIDMap2 and removing
        # through the wrapper trips an assertion inside FAISS and aborts the
        # process, so the IVF index is used directly.
        index = faiss.IndexIVFFlat(quantizer, dim, nlist,
                                   faiss.METRIC_INNER_PRODUCT if metric == "cosine"
                                   else faiss.METRIC_L2)
        index.nprobe = nprobe
        self._base = index
        self._index = index

    def build(self, vectors, ids):
        if not self._base.is_trained:
            self._base.train(vectors)
        self._index.add_with_ids(vectors, ids.astype(np.int64))

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        self._index.remove_ids(self._faiss.IDSelectorBatch(retired.astype(np.int64)))
        self._index.add_with_ids(new_vectors, new_ids.astype(np.int64))

    def search(self, query, k):
        _, labels = self._index.search(query[None, :], k)
        return [int(x) for x in labels[0] if x >= 0]

    def set_search_param(self, value):
        self._base.nprobe = int(value)

    def insert_many(self, ids, vectors):
        self._index.add_with_ids(vectors, ids.astype(np.int64))

    def delete_many(self, ids):
        self._index.remove_ids(self._faiss.IDSelectorBatch(ids.astype(np.int64)))

    def build_filtered(self, ids, vectors, tags):
        self._tags = {int(i): int(t) for i, t in zip(ids, tags)}
        self.build(vectors, ids)

    # Sized from the selectivity rather than fixed. The margin covers the
    # matching records not being spread evenly through the ranking: at 1% a
    # bare k/selectivity fetch lands short about half the time.
    _fetch = 0

    def set_filter_selectivity(self, fraction, k, corpus):
        import math
        if fraction <= 0:
            self._fetch = int(corpus)
            return
        self._fetch = int(min(corpus, max(k, math.ceil(k / fraction * 3))))

    def filter_fetch_width(self):
        return int(self._fetch)

    def search_filtered(self, query, k, tag):
        # No filter in the index, so the caller widens the search and discards.
        # That is the honest cost of filtering with faiss-ivfflat: it can be
        # made correct, but the work scales as k/selectivity and the discarded
        # majority is paid for in full.
        _, labels = self._index.search(query[None, :], self._fetch or k)
        return [int(x) for x in labels[0]
                if x >= 0 and self._tags.get(int(x)) == int(tag)][:k]

    def insert_filtered(self, ids, vectors, tags):
        for one, tag in zip(ids, tags):
            self._tags[int(one)] = int(tag)
        self.insert_many(np.asarray(ids, dtype=np.int64), vectors)

    def delete_filtered(self, ids):
        for one in ids:
            self._tags.pop(int(one), None)
        self.delete_many(np.asarray(ids, dtype=np.int64))

    def stats(self):
        return {"nlist": self._base.nlist, "nprobe": self._base.nprobe}


class FaissHnswEngine(_Base):
    """FAISS HNSW: no delete, so a turnover costs a full rebuild."""
    name = "faiss-hnsw"
    can_delete = False
    needs_rebuild = True
    can_grow = False
    supports_interleaved = False

    def __init__(self, dim: int, metric: str, m: int = 16,
                 ef_construction: int = 100, ef_search: int = 128,
                 threads: int = 1, **_: Any) -> None:
        import faiss
        self.threads = int(threads)
        faiss.omp_set_num_threads(self.threads)
        self._faiss = faiss
        self._dim, self._metric = dim, metric
        self._m, self._efc, self._efs = m, ef_construction, ef_search
        self._index = None

    def _fresh(self):
        index = self._faiss.IndexHNSWFlat(
            self._dim, self._m,
            self._faiss.METRIC_INNER_PRODUCT if self._metric == "cosine"
            else self._faiss.METRIC_L2)
        index.hnsw.efConstruction = self._efc
        index.hnsw.efSearch = self._efs
        return self._faiss.IndexIDMap2(index)

    def build(self, vectors, ids):
        self._index = self._fresh()
        self._index.add_with_ids(vectors, ids)

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        # No removal: the only way to express a turnover is to rebuild.
        self.build(live_vectors, live_ids)

    def search(self, query, k):
        _, labels = self._index.search(query[None, :], k)
        return [int(x) for x in labels[0] if x >= 0]


class UsearchEngine(_Base):
    name = "usearch"
    filtering = "post"
    # usearch documents concurrent add and search.
    supports_concurrent_reads = True

    def __init__(self, dim: int, metric: str, live: int, connectivity: int = 16,
                 threads: int = 1, **_: Any) -> None:
        # usearch defaults threads=0, which means every core.
        self.threads = int(threads)
        from usearch.index import Index as USIndex
        self._index = USIndex(ndim=dim, metric="cos" if metric == "cosine" else "l2sq",
                              connectivity=connectivity)

    def build(self, vectors, ids):
        self._index.add(ids, vectors, threads=self.threads)

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        self._index.remove(retired.astype(np.int64))
        self._index.add(new_ids, new_vectors, threads=self.threads)

    def search(self, query, k):
        matches = self._index.search(query, k, threads=self.threads)
        return [int(x) for x in matches.keys]

    def insert_many(self, ids, vectors):
        self._index.add(ids, vectors, threads=self.threads)

    def delete_many(self, ids):
        self._index.remove(ids.astype(np.int64))

    def build_filtered(self, ids, vectors, tags):
        self._tags = {int(i): int(t) for i, t in zip(ids, tags)}
        self.build(vectors, ids)

    # Sized from the selectivity rather than fixed. The margin covers the
    # matching records not being spread evenly through the ranking: at 1% a
    # bare k/selectivity fetch lands short about half the time.
    _fetch = 0

    def set_filter_selectivity(self, fraction, k, corpus):
        import math
        if fraction <= 0:
            self._fetch = int(corpus)
            return
        self._fetch = int(min(corpus, max(k, math.ceil(k / fraction * 3))))

    def filter_fetch_width(self):
        return int(self._fetch)

    def search_filtered(self, query, k, tag):
        matches = self._index.search(query, self._fetch or k,
                                     threads=self.threads)
        return [int(x) for x in matches.keys
                if self._tags.get(int(x)) == int(tag)][:k]

    def insert_filtered(self, ids, vectors, tags):
        for one, tag in zip(ids, tags):
            self._tags[int(one)] = int(tag)
        self.insert_many(np.asarray(ids, dtype=np.int64), vectors)

    def delete_filtered(self, ids):
        for one in ids:
            self._tags.pop(int(one), None)
        self.delete_many(np.asarray(ids, dtype=np.int64))

    def stats(self):
        return {"size": len(self._index)}


class ChromaEngine(_Base):
    """Chroma: what a lot of agent-memory code actually reaches for."""
    name = "chroma"
    can_update_in_place = True
    filtering = "native"
    supports_concurrent_reads = True

    def __init__(self, dim: int, metric: str, **_: Any) -> None:
        import chromadb
        self._dir = tempfile.mkdtemp(prefix="streambench-chroma-")
        self._client = chromadb.PersistentClient(path=self._dir)
        self._collection = self._client.create_collection(
            name="bench",
            metadata={"hnsw:space": "cosine" if metric == "cosine" else "l2"})

    def build(self, vectors, ids):
        step = 5000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._collection.add(
                ids=[str(int(i)) for i in ids[start:stop]],
                embeddings=[v.tolist() for v in vectors[start:stop]])

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        step = 5000
        for start in range(0, len(retired), step):
            stop = min(start + step, len(retired))
            self._collection.delete(ids=[str(int(i)) for i in retired[start:stop]])
            self._collection.add(
                ids=[str(int(i)) for i in new_ids[start:stop]],
                embeddings=[v.tolist() for v in new_vectors[start:stop]])

    def search(self, query, k):
        got = self._collection.query(query_embeddings=[query.tolist()], n_results=k)
        return [int(x) for x in got["ids"][0]]

    def insert_many(self, ids, vectors):
        step = 5000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._collection.add(
                ids=[str(int(i)) for i in ids[start:stop]],
                embeddings=[v.tolist() for v in vectors[start:stop]])

    def delete_many(self, ids):
        step = 5000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._collection.delete(ids=[str(int(i)) for i in ids[start:stop]])

    def update_many(self, ids, vectors):
        step = 5000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._collection.update(
                ids=[str(int(i)) for i in ids[start:stop]],
                embeddings=[v.tolist() for v in vectors[start:stop]])

    def build_filtered(self, ids, vectors, tags):
        step = 5000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._collection.add(
                ids=[str(int(i)) for i in ids[start:stop]],
                embeddings=[v.tolist() for v in vectors[start:stop]],
                metadatas=[{"tag": int(t)} for t in tags[start:stop]])

    def search_filtered(self, query, k, tag):
        got = self._collection.query(query_embeddings=[query.tolist()],
                                     n_results=k, where={"tag": int(tag)})
        return [int(x) for x in got["ids"][0]]

    def insert_filtered(self, ids, vectors, tags):
        step = 5000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._collection.add(
                ids=[str(int(i)) for i in ids[start:stop]],
                embeddings=[v.tolist() for v in vectors[start:stop]],
                metadatas=[{"tag": int(t)} for t in tags[start:stop]])

    def stats(self):
        return {"count": self._collection.count()}

    def close(self):
        shutil.rmtree(self._dir, ignore_errors=True)


class QdrantEngine(_Base):
    """Qdrant, against a real server.

    Not the in-process client. `QdrantClient(":memory:")` is a pure-Python
    fallback that scans exhaustively; benchmarking it would report a number
    Qdrant never produces and call it Qdrant. This talks to the actual engine
    over HTTP, which means the figures include client and transport cost --
    that is what using Qdrant costs, and it is stated rather than subtracted.
    """
    name = "qdrant"
    can_update_in_place = True     # upsert is the native write
    filtering = "native"
    supports_concurrent_reads = True
    search_ladder = (16, 32, 64, 128, 256, 512)

    def __init__(self, dim: int, metric: str, url: str = "",
                 threads: int = 1, **_: Any) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self.threads = int(threads)
        self._url = url or "http://localhost:6333"
        self._client = QdrantClient(url=self._url, timeout=120)
        # Fail here, not at the first query, so an absent server reads as a
        # skip with a reason instead of a wall of transport errors.
        try:
            self._client.get_collections()
        except Exception as unreachable:
            raise RuntimeError(
                f"no qdrant server at {self._url}: {unreachable}. Start one "
                f"with: docker run -p 6333:6333 qdrant/qdrant") from unreachable
        self._name = "streambench"
        self._models = __import__("qdrant_client.models", fromlist=["models"])
        self._client.recreate_collection(
            collection_name=self._name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE if metric == "cosine" else Distance.EUCLID))
        self._ef = 128

    def _points(self, ids: np.ndarray, vectors: np.ndarray):
        return [self._models.PointStruct(id=int(i), vector=v.tolist())
                for i, v in zip(ids, vectors)]

    def build(self, vectors, ids):
        step = 2000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._client.upsert(collection_name=self._name, wait=True,
                                points=self._points(ids[start:stop],
                                                    vectors[start:stop]))

    def insert_many(self, ids, vectors):
        self.build(vectors, ids)

    def update_many(self, ids, vectors):
        self.build(vectors, ids)      # upsert covers both

    def delete_many(self, ids):
        step = 2000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._client.delete(
                collection_name=self._name, wait=True,
                points_selector=self._models.PointIdsList(
                    points=[int(i) for i in ids[start:stop]]))

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        self.delete_many(retired)
        self.insert_many(new_ids, new_vectors)

    def set_search_param(self, value):
        self._ef = int(value)

    def search(self, query, k):
        got = self._client.query_points(
            collection_name=self._name, query=query.tolist(), limit=k,
            search_params=self._models.SearchParams(hnsw_ef=self._ef))
        return [int(point.id) for point in got.points]

    def build_filtered(self, ids, vectors, tags):
        step = 2000
        for start in range(0, len(ids), step):
            stop = min(start + step, len(ids))
            self._client.upsert(
                collection_name=self._name, wait=True,
                points=[self._models.PointStruct(
                    id=int(i), vector=v.tolist(), payload={"tag": int(t)})
                    for i, v, t in zip(ids[start:stop], vectors[start:stop],
                                       tags[start:stop])])

    def insert_filtered(self, ids, vectors, tags):
        self.build_filtered(ids, vectors, tags)

    def search_filtered(self, query, k, tag):
        condition = self._models.Filter(must=[self._models.FieldCondition(
            key="tag",
            match=self._models.MatchValue(value=int(tag)))])
        got = self._client.query_points(
            collection_name=self._name, query=query.tolist(), limit=k,
            query_filter=condition,
            search_params=self._models.SearchParams(hnsw_ef=self._ef))
        return [int(point.id) for point in got.points]

    def stats(self):
        info = self._client.get_collection(self._name)
        return {"points": info.points_count, "hnsw_ef": self._ef,
                "note": "server over HTTP; transport cost included"}

    def close(self):
        try:
            self._client.delete_collection(self._name)
        except Exception:
            pass


class AnnoyEngine(_Base):
    """Annoy: immutable once built, so every turnover is a rebuild."""
    name = "annoy"
    can_delete = False
    needs_rebuild = True
    can_grow = False
    supports_interleaved = False

    def __init__(self, dim: int, metric: str, trees: int = 32, **_: Any) -> None:
        self._dim = dim
        self._metric = "angular" if metric == "cosine" else "euclidean"
        self._trees = trees
        self._index = None
        self._ids: np.ndarray | None = None

    @staticmethod
    def _self_check() -> None:
        """Refuse to report numbers from a build that does not work.

        Some annoy wheels (arm64 macOS, CPython 3.12 at the time of writing)
        return a single neighbour for any k. Publishing the resulting recall of
        zero would misrepresent annoy rather than measure it.
        """
        import numpy as np
        from annoy import AnnoyIndex
        probe = AnnoyIndex(8, "euclidean")
        rng = np.random.default_rng(0)
        for position in range(64):
            probe.add_item(position, rng.standard_normal(8).tolist())
        probe.build(8)
        found = probe.get_nns_by_vector(rng.standard_normal(8).tolist(), 5)
        if len(found) < 5:
            raise RuntimeError(
                f"annoy build is not functional here: asked for 5 neighbours, "
                f"got {len(found)}")

    def build(self, vectors, ids):
        from annoy import AnnoyIndex
        if self._index is None:
            self._self_check()
        index = AnnoyIndex(self._dim, self._metric)
        for position, vector in enumerate(vectors):
            index.add_item(position, vector.tolist())
        index.build(self._trees)
        self._index, self._ids = index, np.asarray(ids)

    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors):
        self.build(live_vectors, live_ids)

    def search(self, query, k):
        positions = self._index.get_nns_by_vector(query.tolist(), k)
        return [int(self._ids[p]) for p in positions]

    def stats(self):
        return {"trees": self._trees}


ENGINES: dict[str, Callable[..., Engine]] = {
    e.name: e for e in (ChronoVecEngine, HnswlibEngine, FaissFlatEngine,
                        FaissHnswEngine, UsearchEngine, ChromaEngine, QdrantEngine,
                        AnnoyEngine)
}


def available_engines() -> dict[str, bool]:
    """Which engines can actually be constructed in this environment."""
    modules = {"chronovec": "chronovec", "hnswlib": "hnswlib",
               "faiss-ivfflat": "faiss", "faiss-hnsw": "faiss",
               "usearch": "usearch", "chroma": "chromadb",
               "qdrant": "qdrant_client", "annoy": "annoy"}
    out = {}
    for name, module in modules.items():
        try:
            __import__(module)
            out[name] = True
        except Exception:
            out[name] = False
    return out


CAPABILITIES = ("can_delete", "can_grow", "can_update_in_place",
                "batched_writes", "supports_concurrent_reads",
                "supports_snapshots", "needs_rebuild")


def capability_matrix(names: list[str] | None = None) -> dict[str, dict[str, bool]]:
    """What each engine can do, read off the classes without constructing them.

    Worth printing beside any results table. A benchmark that fills in every
    cell -- by emulating an update as delete-plus-insert, or by rebuilding
    where an engine has no delete -- makes engines look equivalent when they
    are not, because the number is there and the footnote is easy to miss.
    """
    chosen = names or list(ENGINES)
    out: dict[str, dict[str, bool]] = {}
    for name in chosen:
        engine = ENGINES.get(name)
        if engine is None:
            continue
        out[name] = {flag: bool(getattr(engine, flag)) for flag in CAPABILITIES}
    return out


def format_capability_matrix(names: list[str] | None = None) -> str:
    matrix = capability_matrix(names)
    short = {"can_delete": "delete", "can_grow": "grow",
             "can_update_in_place": "update", "batched_writes": "batch",
             "supports_concurrent_reads": "read|write",
             "supports_snapshots": "snapshot", "needs_rebuild": "rebuilds"}
    columns = [short[flag] for flag in CAPABILITIES]
    lines = ["  " + f"{'engine':<15}" + "".join(f"{c:>11}" for c in columns)]
    for name, flags in matrix.items():
        cells = []
        for flag in CAPABILITIES:
            value = flags[flag]
            if flag == "needs_rebuild":
                cells.append("yes" if value else "-")
            else:
                cells.append("yes" if value else "NO")
        lines.append("  " + f"{name:<15}" + "".join(f"{c:>11}" for c in cells))
    return "\n".join(lines)
