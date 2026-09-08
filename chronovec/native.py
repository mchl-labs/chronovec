from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from .index import SearchResult


class _CvVersion(ctypes.Structure):
    _fields_ = [
        ("major", ctypes.c_uint8),
        ("minor", ctypes.c_uint8),
        ("patch", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]


class _Stats(ctypes.Structure):
    _fields_ = [
        ("clock", ctypes.c_uint64),
        ("pages", ctypes.c_uint64),
        ("live_vectors", ctypes.c_uint64),
        ("retained_versions", ctypes.c_uint64),
        ("allocated_slots", ctypes.c_uint64),
        ("splits", ctypes.c_uint64),
        ("reclaimed_versions", ctypes.c_uint64),
        ("merges", ctypes.c_uint64),
        ("routing_full_rebuilds", ctypes.c_uint64),
        ("routing_incremental_updates", ctypes.c_uint64),
        ("routing_centroid_bytes", ctypes.c_uint64),
        ("routing_edge_count", ctypes.c_uint64),
    ]


class _Filter(ctypes.Structure):
    _fields_ = [
        ("require_all", ctypes.c_uint64),
        ("require_any", ctypes.c_uint64),
        ("exclude", ctypes.c_uint64),
    ]


class _SearchMetrics(ctypes.Structure):
    _fields_ = [
        ("directory_pages", ctypes.c_uint64),
        ("centroid_scores", ctypes.c_uint64),
        ("routed_pages", ctypes.c_uint64),
        ("physical_candidates", ctypes.c_uint64),
        ("visible_candidates", ctypes.c_uint64),
        ("mvcc_filtered", ctypes.c_uint64),
        ("sketch_scores", ctypes.c_uint64),
        ("exact_scores", ctypes.c_uint64),
        ("result_count", ctypes.c_uint64),
        ("bound_pruned_pages", ctypes.c_uint64),
        ("preparation_ns", ctypes.c_uint64),
        ("routing_ns", ctypes.c_uint64),
        ("screening_ns", ctypes.c_uint64),
        ("rerank_ns", ctypes.c_uint64),
        ("result_materialization_ns", ctypes.c_uint64),
        ("total_ns", ctypes.c_uint64),
    ]


def _library_path() -> Path:
    override = os.environ.get("CHRONOVEC_LIBRARY")
    if override:
        return Path(override)
    name = (
        "chronovec.dll"
        if sys.platform == "win32"
        else "libchronovec.dylib"
        if sys.platform == "darwin"
        else "libchronovec.so"
    )
    pkg = Path(__file__).resolve().parent
    candidates = [
        pkg / name,  # pip install . (scikit-build-core)
        pkg.parent / "build" / name,  # manual cmake build from source root
        pkg.parent / "build" / "lib" / name,  # some cmake generator layouts
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise RuntimeError(
        f"native ChronoVec library not found ({name}); "
        "run `pip install .` or `cmake -S . -B build && cmake --build build -j`"
    )


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_library_path()))
    f32p = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    lib.cv_create.argtypes = [ctypes.c_size_t, ctypes.c_int, ctypes.c_size_t, ctypes.c_size_t]
    lib.cv_create.restype = ctypes.c_void_p
    lib.cv_create_with_options.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_size_t,
    ]
    lib.cv_create_with_options.restype = ctypes.c_void_p
    lib.cv_destroy.argtypes = [ctypes.c_void_p]
    lib.cv_insert.argtypes = [ctypes.c_void_p, ctypes.c_int64, f32p, ctypes.c_uint64, u64p]
    lib.cv_delete.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64, u64p]
    lib.cv_search.argtypes = [
        ctypes.c_void_p,
        f32p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_size_t,
        i64p,
        f32p,
    ]
    lib.cv_search.restype = ctypes.c_size_t
    lib.cv_search_with_options.argtypes = [
        ctypes.c_void_p,
        f32p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_uint32,
        i64p,
        f32p,
    ]
    lib.cv_delete_batch.argtypes = [ctypes.c_void_p, i64p, ctypes.c_size_t, u64p]
    lib.cv_delete_batch.restype = ctypes.c_size_t
    lib.cv_apply_changes.argtypes = [
        ctypes.c_void_p,
        i64p,
        ctypes.c_size_t,
        i64p,
        f32p,
        ctypes.c_size_t,
        u64p,
    ]
    lib.cv_apply_changes.restype = ctypes.c_int
    lib.cv_insert_batch.argtypes = [ctypes.c_void_p, i64p, f32p, ctypes.c_size_t, u64p]
    lib.cv_insert_batch.restype = ctypes.c_size_t
    lib.cv_flush_vectors.argtypes = [ctypes.c_void_p]
    lib.cv_flush_vectors.restype = ctypes.c_size_t
    lib.cv_create_disk_backed.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.cv_create_disk_backed.restype = ctypes.c_void_p
    lib.cv_create_with_wal.argtypes = [
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.cv_create_with_wal.restype = ctypes.c_void_p
    lib.cv_set_threads.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.cv_set_threads.restype = None
    lib.cv_set_auto_consolidate.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.cv_set_auto_consolidate.restype = None
    lib.cv_insert_batch_labeled.argtypes = [
        ctypes.c_void_p,
        i64p,
        f32p,
        u64p,
        ctypes.c_size_t,
        u64p,
    ]
    lib.cv_insert_batch_labeled.restype = ctypes.c_size_t
    lib.cv_search_batch.argtypes = [
        ctypes.c_void_p,
        f32p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_uint32,
        i64p,
        f32p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cv_search_batch.restype = ctypes.c_size_t
    lib.cv_search_filtered.argtypes = [
        ctypes.c_void_p,
        f32p,
        ctypes.c_size_t,
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(_Filter),
        i64p,
        f32p,
    ]
    lib.cv_search_filtered.restype = ctypes.c_size_t
    lib.cv_insert_labeled.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        f32p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        u64p,
    ]
    lib.cv_insert_labeled.restype = ctypes.c_int
    lib.cv_search_with_options.restype = ctypes.c_size_t
    lib.cv_exact_search_f32.argtypes = [
        f32p,
        i64p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        f32p,
        ctypes.c_size_t,
        i64p,
        f32p,
    ]
    lib.cv_exact_search_f32.restype = ctypes.c_size_t
    lib.cv_vacuum.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    lib.cv_vacuum.restype = ctypes.c_size_t
    lib.cv_save_checkpoint.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.cv_load_checkpoint.argtypes = [ctypes.c_char_p]
    lib.cv_load_checkpoint.restype = ctypes.c_void_p
    for name in ("cv_dimensions", "cv_page_capacity", "cv_default_nprobe", "cv_rerank_factor"):
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_void_p]
        function.restype = ctypes.c_size_t
    lib.cv_metric.argtypes = [ctypes.c_void_p]
    lib.cv_metric.restype = ctypes.c_int
    lib.cv_flags.argtypes = [ctypes.c_void_p]
    lib.cv_flags.restype = ctypes.c_uint32
    lib.cv_clock.argtypes = [ctypes.c_void_p]
    lib.cv_clock.restype = ctypes.c_uint64
    lib.cv_get_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Stats)]
    lib.cv_page_label_profile.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_Filter),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.cv_page_label_profile.restype = ctypes.c_size_t
    lib.cv_get_last_search_metrics.argtypes = [ctypes.POINTER(_SearchMetrics)]
    lib.cv_last_error.restype = ctypes.c_char_p
    lib.cv_version.argtypes = []
    lib.cv_version.restype = _CvVersion
    return lib


class NativeChronoVecIndex:
    """ctypes binding over the portable ChronoVec C ABI."""

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "cosine",
        page_capacity: int = 256,
        nprobe: int = 16,
        screening: bool = True,
        adaptive_bounds: bool = False,
        rerank_factor: int = 4,
        labels: bool = False,
        label_partition: bool = False,
        arena_path: str | os.PathLike[str] | None = None,
        max_vectors: int = 0,
        wal_path: str | os.PathLike[str] | None = None,
        wal_sync: bool = True,
        threads: int = 1,
        auto_consolidate: bool = True,
    ) -> None:
        self.dimensions = dimensions
        self.metric = metric
        self.nprobe = nprobe
        self.screening = screening
        self.adaptive_bounds = adaptive_bounds
        self.labels = labels
        self._lib = _load()
        metric_id = 0 if metric == "cosine" else 1 if metric == "l2" else -1
        flags = (
            int(screening)
            | (int(adaptive_bounds) << 1)
            | (int(labels) << 2)
            | (int(label_partition) << 3)
        )
        if wal_path is not None:
            # Both create and recover: an existing log is replayed first. An
            # arena may be combined -- it is a spill file, not a record, so it
            # is recreated empty and refilled by the replay.
            if arena_path is not None and max_vectors <= 0:
                raise ValueError("max_vectors must be positive when arena_path is set")
            self._handle = self._lib.cv_create_with_wal(
                dimensions,
                metric_id,
                page_capacity,
                nprobe,
                flags,
                rerank_factor,
                str(wal_path).encode(),
                int(bool(wal_sync)),
                str(arena_path).encode() if arena_path is not None else None,
                max_vectors,
            )
        elif arena_path is not None:
            if max_vectors <= 0:
                raise ValueError("max_vectors must be positive when arena_path is set")
            self._handle = self._lib.cv_create_disk_backed(
                dimensions,
                metric_id,
                page_capacity,
                nprobe,
                flags,
                rerank_factor,
                str(arena_path).encode(),
                max_vectors,
            )
        else:
            self._handle = self._lib.cv_create_with_options(
                dimensions, metric_id, page_capacity, nprobe, flags, rerank_factor
            )
        if not self._handle:
            raise RuntimeError(self._error())
        if threads != 1:
            self._lib.cv_set_threads(self._handle, max(1, int(threads)))
        if not auto_consolidate:
            self._lib.cv_set_auto_consolidate(self._handle, 0)
        self.auto_consolidate = bool(auto_consolidate)
        self.threads = max(1, int(threads))

    def _error(self) -> str:
        value = self._lib.cv_last_error()
        return value.decode() if value else "unknown native error"

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self._lib.cv_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def _vector(self, vector) -> np.ndarray:
        value = np.ascontiguousarray(vector, dtype=np.float32)
        if value.shape != (self.dimensions,):
            raise ValueError(f"expected ({self.dimensions},), got {value.shape}")
        return value

    @property
    def clock(self) -> int:
        return int(self._lib.cv_clock(self._handle))

    def insert(self, item_id: int, vector, *, timestamp: int | None = None, label: int = 0) -> int:
        value = self._vector(vector)
        committed = ctypes.c_uint64()
        pointer = value.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        if label:
            result = self._lib.cv_insert_labeled(
                self._handle, item_id, pointer, label, timestamp or 0, ctypes.byref(committed)
            )
        else:
            result = self._lib.cv_insert(
                self._handle, item_id, pointer, timestamp or 0, ctypes.byref(committed)
            )
        if result:
            raise RuntimeError(self._error())
        return int(committed.value)

    def insert_many(self, ids, vectors, labels=None) -> int:
        """Insert many vectors in one call, returning how many were stored.

        The batch is published at one MVCC timestamp, so readers see all of
        it or none of it. This also avoids the per-call binding overhead of a
        Python loop over ``insert()``.
        """
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        block = np.ascontiguousarray(vectors, dtype=np.float32)
        if block.ndim != 2 or block.shape[1] != self.dimensions:
            raise ValueError(f"expected vectors of shape (n, {self.dimensions}), got {block.shape}")
        if ids.shape[0] != block.shape[0]:
            raise ValueError("ids and vectors must have the same length")
        committed = ctypes.c_uint64()
        if labels is None:
            stored = int(
                self._lib.cv_insert_batch(
                    self._handle,
                    ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    block.shape[0],
                    ctypes.byref(committed),
                )
            )
        else:
            if not self.labels:
                raise ValueError("labels require labels=True at index creation")
            marks = np.ascontiguousarray(labels, dtype=np.uint64)
            if marks.shape != (block.shape[0],):
                raise ValueError(f"expected {block.shape[0]} labels, got {marks.shape}")
            stored = int(
                self._lib.cv_insert_batch_labeled(
                    self._handle,
                    ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                    block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    marks.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                    block.shape[0],
                    ctypes.byref(committed),
                )
            )
        if stored != block.shape[0]:
            raise RuntimeError(self._error())
        return stored

    def delete_many(self, ids) -> int:
        """Delete many ids in one call. Returns how many were removed."""
        block = np.ascontiguousarray(ids, dtype=np.int64)
        committed = ctypes.c_uint64()
        return int(
            self._lib.cv_delete_batch(
                self._handle,
                block.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                block.shape[0],
                ctypes.byref(committed),
            )
        )

    def apply_changes(self, delete_ids, upsert_ids, upsert_vectors) -> int:
        """Atomically publish deletes and upserts at one MVCC timestamp."""
        deletes = np.ascontiguousarray(delete_ids, dtype=np.int64)
        ids = np.ascontiguousarray(upsert_ids, dtype=np.int64)
        vectors = np.ascontiguousarray(upsert_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (ids.shape[0], self.dimensions):
            raise ValueError(
                f"expected vectors of shape ({ids.shape[0]}, {self.dimensions}), got {vectors.shape}"
            )
        if np.intersect1d(deletes, ids).size:
            raise ValueError("an id cannot be deleted and upserted in one apply_changes call")
        committed = ctypes.c_uint64()
        result = self._lib.cv_apply_changes(
            self._handle,
            deletes.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            deletes.shape[0],
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids.shape[0],
            ctypes.byref(committed),
        )
        if result:
            raise RuntimeError(self._error())
        return int(committed.value)

    def delete(self, item_id: int, *, timestamp: int | None = None) -> int:
        committed = ctypes.c_uint64()
        result = self._lib.cv_delete(self._handle, item_id, timestamp or 0, ctypes.byref(committed))
        if result:
            raise KeyError(self._error())
        return int(committed.value)

    def search(
        self,
        vector,
        k: int = 10,
        *,
        snapshot: int | None = None,
        nprobe: int | None = None,
        adaptive: bool = False,
        linear_routing: bool = False,
        require_all: int = 0,
        require_any: int = 0,
        exclude: int = 0,
    ) -> list[SearchResult]:
        """Nearest neighbours, optionally restricted by attribute label.

        A record matches when it carries every bit in ``require_all``, at least
        one bit in ``require_any`` (when non-zero), and no bit in ``exclude``.
        Pages whose label union cannot match are skipped whole.
        """
        if not getattr(self, "_handle", None):
            # Without this a closed index answers every query with an empty
            # list, which is indistinguishable from an index that is merely
            # empty. Writes already raise; reads must too.
            raise RuntimeError("index is closed")
        if snapshot == 0:
            # 0 is the C ABI's "as of now", so it cannot also mean "before
            # anything was written". An empty index reports clock 0, which
            # makes `index.clock` look like a usable token before the first
            # write and silently return the present instead.
            raise ValueError(
                "snapshot=0 means 'now' in the underlying ABI, not 'before "
                "the first write'; take a snapshot after at least one write"
            )
        value = self._vector(vector)
        if adaptive and not self.adaptive_bounds:
            raise ValueError("adaptive search requires adaptive_bounds=True at index creation")
        filtering = bool(require_all or require_any or exclude)
        if filtering and not self.labels:
            raise ValueError("filtering requires labels=True at index creation")
        ids = np.empty(k, dtype=np.int64)
        distances = np.empty(k, dtype=np.float32)
        flags = int(adaptive) | (int(linear_routing) << 1)
        args = (
            self._handle,
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k,
            snapshot or 0,
            nprobe or self.nprobe,
            flags,
        )
        if filtering:
            spec = _Filter(require_all, require_any, exclude)
            count = self._lib.cv_search_filtered(
                *args,
                ctypes.byref(spec),
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        else:
            count = self._lib.cv_search_with_options(
                *args,
                ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        return [SearchResult(int(ids[i]), float(distances[i])) for i in range(count)]

    def flush_vectors(self) -> int:
        """Flush disk-backed vectors so the kernel may evict them. BETA.

        Returns bytes made evictable, or 0 when the index is not disk backed.
        Eviction happens under memory pressure; resident memory does not
        necessarily drop immediately.
        """
        return int(self._lib.cv_flush_vectors(self._handle))

    def vacuum(self, oldest_snapshot: int, *, budget_versions: int = 0) -> int:
        return int(self._lib.cv_vacuum(self._handle, oldest_snapshot, budget_versions))

    def save(self, path: str | os.PathLike[str]) -> None:
        """Durably and atomically checkpoint the complete native index."""
        encoded = os.fsencode(path)
        if self._lib.cv_save_checkpoint(self._handle, encoded):
            raise RuntimeError(self._error())

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> NativeChronoVecIndex:
        """Restore an index without replaying its mutation history."""
        instance = cls.__new__(cls)
        instance._lib = _load()
        instance._handle = instance._lib.cv_load_checkpoint(os.fsencode(path))
        if not instance._handle:
            raise RuntimeError(instance._error())
        instance.dimensions = int(instance._lib.cv_dimensions(instance._handle))
        instance.metric = "cosine" if instance._lib.cv_metric(instance._handle) == 0 else "l2"
        instance.nprobe = int(instance._lib.cv_default_nprobe(instance._handle))
        instance.screening = bool(instance._lib.cv_flags(instance._handle) & 1)
        instance.adaptive_bounds = bool(instance._lib.cv_flags(instance._handle) & 2)
        instance.labels = bool(instance._lib.cv_flags(instance._handle) & 4)
        return instance

    def search_many(
        self,
        queries,
        k: int = 10,
        *,
        snapshot: int | None = None,
        nprobe: int | None = None,
        adaptive: bool = False,
        linear_routing: bool = False,
        as_arrays: bool = False,
    ) -> list[list[SearchResult]] | tuple[np.ndarray, np.ndarray]:
        """Answer many queries in one call.

        Identical results to :meth:`search` per query; what it saves is the
        per-call crossing, measured at 7.9 us of a 61.2 us query. All the
        queries see the same snapshot, so the batch is a consistent read.

        Concurrency is left to the caller: readers are lock-free, so running
        several of these on separate threads scales, and parallelising inside
        would oversubscribe a caller that already does.

        `as_arrays` returns `(ids, distances)` as `(n, k)` arrays instead of
        result objects, padded with -1 and infinity. Building ten objects per
        query is about half the remaining per-query cost, so a numeric pipeline
        that does not want them should not pay for them.
        """
        if not getattr(self, "_handle", None):
            raise RuntimeError("index is closed")
        if snapshot == 0:
            raise ValueError(
                "snapshot=0 means 'now' in the underlying ABI, not 'before "
                "the first write'; take a snapshot after at least one write"
            )
        block = np.ascontiguousarray(queries, dtype=np.float32)
        if block.ndim == 1:
            block = block[None, :]
        if block.ndim != 2 or block.shape[1] != self.dimensions:
            raise ValueError(f"expected queries of shape (n, {self.dimensions}), got {block.shape}")
        if adaptive and not self.adaptive_bounds:
            raise ValueError("adaptive search requires adaptive_bounds=True at index creation")
        rows = block.shape[0]
        if rows == 0 or k == 0:
            if as_arrays:
                return (np.empty((rows, k), dtype=np.int64), np.empty((rows, k), dtype=np.float32))
            return [[] for _ in range(rows)]
        ids = np.empty(rows * k, dtype=np.int64)
        distances = np.empty(rows * k, dtype=np.float32)
        found = (ctypes.c_size_t * rows)()
        flags = int(adaptive) | (int(linear_routing) << 1)
        self._lib.cv_search_batch(
            self._handle,
            block.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rows,
            k,
            snapshot or 0,
            nprobe or self.nprobe,
            flags,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            found,
        )
        if as_arrays:
            return ids.reshape(rows, k), distances.reshape(rows, k)
        out = []
        for row in range(rows):
            count = found[row]
            base = row * k
            out.append(
                [
                    SearchResult(int(ids[base + position]), float(distances[base + position]))
                    for position in range(count)
                ]
            )
        return out

    def stats(self) -> dict[str, int | float]:
        stats = _Stats()
        if self._lib.cv_get_stats(self._handle, ctypes.byref(stats)):
            raise RuntimeError(self._error())
        result: dict[str, int | float] = {
            field[0]: int(getattr(stats, field[0])) for field in stats._fields_
        }
        live = int(result["live_vectors"])
        result["capacity_amplification"] = result["allocated_slots"] / live if live else 0.0
        result["vector_payload_bytes"] = result["allocated_slots"] * self.dimensions * 4
        result["screening_bytes"] = (
            result["allocated_slots"] * self.dimensions if self.screening else 0
        )
        result["tracked_index_bytes"] = (
            result["vector_payload_bytes"]
            + result["screening_bytes"]
            + result["routing_centroid_bytes"]
            + result["routing_edge_count"] * 4
        )
        return result

    def page_label_profile(
        self, *, require_all: int = 0, require_any: int = 0, exclude: int = 0, snapshot: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Live and filter-matching record counts, per page.

        How tightly records sharing a label are packed decides how much a
        filtered search can skip, and neither recall nor latency shows it: an
        index that scatters one label across every page still answers
        correctly, and reads as merely slow.

        Returns two numpy arrays of equal length -- live records per page, and
        how many of those match the filter.
        """
        import numpy as np

        pages = self._lib.cv_page_label_profile(self._handle, None, 0, None, None, 0)
        live = np.zeros(pages, dtype=np.uint32)
        matching = np.zeros(pages, dtype=np.uint32)
        spec = _Filter(require_all, require_any, exclude)
        self._lib.cv_page_label_profile(
            self._handle,
            ctypes.byref(spec),
            snapshot,
            live.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            matching.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            pages,
        )
        return live, matching

    def last_search_metrics(self) -> dict[str, int]:
        metrics = _SearchMetrics()
        if self._lib.cv_get_last_search_metrics(ctypes.byref(metrics)):
            raise RuntimeError(self._error())
        return {field[0]: int(getattr(metrics, field[0])) for field in metrics._fields_}

    def __enter__(self) -> NativeChronoVecIndex:
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def library_version() -> tuple[int, int, int]:
    """Return (major, minor, patch) of the loaded native library's C ABI.

    Compare ``major`` against ``CV_ABI_VERSION_MAJOR`` (currently 0) to detect
    an incompatible build.  An exact match on major is sufficient for forward
    compatibility within a minor version series.
    """
    v = _load().cv_version()
    return (int(v.major), int(v.minor), int(v.patch))


def exact_search(
    vectors, queries, k: int = 10, *, metric: str = "cosine", ids=None
) -> list[list[SearchResult]]:
    """Run the native SIMD exact-scan baseline over contiguous float32 data."""
    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
    if matrix.ndim != 2 or query_matrix.ndim != 2 or query_matrix.shape[1] != matrix.shape[1]:
        raise ValueError("vectors and queries must be compatible 2D arrays")
    if metric not in {"cosine", "l2"}:
        raise ValueError("metric must be 'cosine' or 'l2'")
    library = _load()
    ids = (
        np.arange(len(matrix), dtype=np.int64)
        if ids is None
        else np.ascontiguousarray(ids, dtype=np.int64)
    )
    if ids.shape != (len(matrix),):
        raise ValueError("ids must contain one int64 per vector")
    results = []
    for query in query_matrix:
        out_ids = np.empty(k, dtype=np.int64)
        distances = np.empty(k, dtype=np.float32)
        count = library.cv_exact_search_f32(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            len(matrix),
            matrix.shape[1],
            0 if metric == "cosine" else 1,
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            k,
            out_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        results.append(
            [SearchResult(int(out_ids[index]), float(distances[index])) for index in range(count)]
        )
    return results


def estimate_memory(
    dimensions: int,
    vectors: int,
    *,
    screening: bool = True,
    capacity_amplification: float = 1.78,
    ram_budget_bytes: int | None = None,
) -> dict[str, object]:
    """Estimate resident memory, and say whether disk backing is worth it.

    Disk backing is not a general improvement. It moves the float payload to a
    mapping so the kernel can evict cold vectors, which helps only when that
    payload does not comfortably fit in RAM. Below that point it adds a file, a
    fixed mapping and writeback pressure for nothing.

    `capacity_amplification` defaults to the value measured at 1M SIFT; pass
    your own from ``index.stats()`` for a tighter estimate.

    Returns byte estimates for both modes plus ``recommend_disk``, which is only
    true when a RAM budget is supplied and the heap estimate exceeds 70% of it.
    """
    if dimensions <= 0 or vectors < 0:
        raise ValueError("dimensions must be positive and vectors non-negative")
    slots = vectors * capacity_amplification
    floats = int(slots * 4 * dimensions)
    codes = int(slots * (dimensions // 2 + 8)) if screening else 0
    heap = floats + codes
    resident_on_disk = codes if screening else 0
    recommend = False
    headroom = None
    if ram_budget_bytes:
        recommend = heap > 0.7 * ram_budget_bytes
        headroom = ram_budget_bytes - heap
    return {
        "live_vectors": vectors,
        "allocated_slots": int(slots),
        "float_payload_bytes": floats,
        "code_bytes": codes,
        "heap_resident_bytes": heap,
        "disk_backed_resident_bytes": resident_on_disk,
        "disk_arena_bytes": floats,
        "reduction_factor": (heap / resident_on_disk) if resident_on_disk else float("inf"),
        "ram_budget_bytes": ram_budget_bytes,
        "headroom_bytes": headroom,
        "recommend_disk": recommend,
        "note": (
            "Disk backing suits read-heavy or bulk-load-then-serve use. "
            "Copy-on-write dirties a whole page per insert, so sustained "
            "high-rate mutation generates more writeback than a commodity SSD "
            "absorbs."
        ),
    }
