from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

MAX_TS = np.iinfo(np.int64).max
Metric = Literal["cosine", "l2"]


@dataclass(frozen=True)
class SearchResult:
    id: int
    distance: float


class _Page:
    """Fixed-capacity vector page.

    Pages are the unit of routing, splitting, reclamation, and locking.  The
    implementation uses mutable NumPy arrays for a useful prototype; a native
    implementation can publish immutable page/delta descriptors with CAS.
    """

    def __init__(self, page_id: int, capacity: int, dimensions: int) -> None:
        self.page_id = page_id
        self.capacity = capacity
        self.dimensions = dimensions
        self.ids = np.full(capacity, -1, dtype=np.int64)
        self.vectors = np.empty((capacity, dimensions), dtype=np.float32)
        self.begin_ts = np.zeros(capacity, dtype=np.int64)
        self.end_ts = np.zeros(capacity, dtype=np.int64)
        self.occupied = np.zeros(capacity, dtype=bool)
        self.centroid = np.zeros(dimensions, dtype=np.float32)
        self.radius = 0.0
        self.generation = 0
        self._sum = np.zeros(dimensions, dtype=np.float64)
        self._current_count = 0
        self.lock = threading.RLock()

    def free_slot(self) -> int | None:
        free = np.flatnonzero(~self.occupied)
        return int(free[0]) if free.size else None

    def visible_mask(self, snapshot: int) -> np.ndarray:
        return self.occupied & (self.begin_ts <= snapshot) & (snapshot < self.end_ts)

    def current_mask(self) -> np.ndarray:
        return self.visible_mask(MAX_TS - 1)

    def recompute_geometry(self, metric: Metric) -> None:
        mask = self.current_mask()
        if not np.any(mask):
            mask = self.occupied
        if not np.any(mask):
            self.centroid.fill(0)
            self.radius = 0.0
            self._sum.fill(0)
            self._current_count = 0
            return
        vectors = self.vectors[mask]
        self._sum = vectors.sum(axis=0, dtype=np.float64)
        self._current_count = vectors.shape[0]
        center = vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
        if metric == "cosine":
            norm = float(np.linalg.norm(center))
            if norm:
                center /= norm
            distances = 1.0 - vectors @ center
        else:
            distances = np.linalg.norm(vectors - center, axis=1)
        self.centroid = center
        self.radius = float(np.max(distances, initial=0.0))
        self.generation += 1

    def add_to_geometry(self, vector: np.ndarray, metric: Metric) -> None:
        """Incrementally update routing geometry after an insertion.

        Exact geometry is restored at split/vacuum boundaries. Between those
        points the radius is conservative for L2 and heuristic for cosine; the
        current query implementation uses centroid ordering, not radius-based
        termination.
        """
        old_center = self.centroid.copy()
        self._sum += vector
        self._current_count += 1
        center = (self._sum / self._current_count).astype(np.float32)
        if metric == "cosine":
            norm = float(np.linalg.norm(center))
            if norm:
                center /= norm
            point_distance = float(1.0 - vector @ center)
            center_shift = float(1.0 - old_center @ center) if self._current_count > 1 else 0.0
        else:
            point_distance = float(np.linalg.norm(vector - center))
            center_shift = (
                float(np.linalg.norm(old_center - center)) if self._current_count > 1 else 0.0
            )
        self.centroid = center
        self.radius = max(self.radius + center_shift, point_distance)
        self.generation += 1

    def remove_from_geometry(self, vector: np.ndarray, metric: Metric) -> None:
        if self._current_count <= 1:
            self._sum.fill(0)
            self._current_count = 0
            self.centroid.fill(0)
            self.generation += 1
            return
        self._sum -= vector
        self._current_count -= 1
        center = (self._sum / self._current_count).astype(np.float32)
        if metric == "cosine":
            norm = float(np.linalg.norm(center))
            if norm:
                center /= norm
        self.centroid = center
        # Radius remains conservative/stale until bounded maintenance.
        self.generation += 1

    def clone_empty(self, page_id: int) -> _Page:
        return _Page(page_id, self.capacity, self.dimensions)


class _ReferenceIndex:
    """Page-structured dynamic approximate-nearest-neighbor index.

    The prototype focuses on the proposed research invariants:
    bounded pages, local splits, MVCC visibility, bounded vacuum work, and
    predicate-aware page probing. It is not yet a lock-free native engine.
    """

    def __init__(
        self,
        dimensions: int,
        *,
        metric: Metric = "cosine",
        page_capacity: int = 512,
        nprobe: int = 8,
        split_iterations: int = 5,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if page_capacity < 8:
            raise ValueError("page_capacity must be at least 8")
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        self.dimensions = dimensions
        self.metric = metric
        self.page_capacity = page_capacity
        self.nprobe = nprobe
        self.split_iterations = split_iterations
        self._pages: tuple[_Page, ...] = ()
        self._locations: dict[int, list[tuple[_Page, int]]] = {}
        self._clock = 0
        self._next_page_id = 0
        self._retired: deque[tuple[int, int, int]] = deque()
        self._directory_lock = threading.RLock()
        self._stats = {
            "inserts": 0,
            "deletes": 0,
            "splits": 0,
            "vacuumed_versions": 0,
            "searches": 0,
        }

    @property
    def clock(self) -> int:
        return self._clock

    def _tick(self, timestamp: int | None) -> int:
        with self._directory_lock:
            if timestamp is None:
                self._clock += 1
            elif timestamp <= self._clock:
                raise ValueError("timestamps must increase monotonically")
            else:
                self._clock = timestamp
            return self._clock

    def _prepare(self, vector: Iterable[float] | np.ndarray) -> np.ndarray:
        value = np.asarray(vector, dtype=np.float32)
        if value.shape != (self.dimensions,):
            raise ValueError(f"expected vector shape ({self.dimensions},), got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("vectors must contain only finite values")
        value = value.copy()
        if self.metric == "cosine":
            norm = float(np.linalg.norm(value))
            if norm == 0:
                raise ValueError("cosine vectors must be non-zero")
            value /= norm
        return value

    def _new_page(self) -> _Page:
        page = _Page(self._next_page_id, self.page_capacity, self.dimensions)
        self._next_page_id += 1
        return page

    def _page_distances(self, vector: np.ndarray, pages: tuple[_Page, ...]) -> np.ndarray:
        centers = np.vstack([page.centroid for page in pages])
        if self.metric == "cosine":
            return 1.0 - centers @ vector
        return np.linalg.norm(centers - vector, axis=1)

    def _choose_page(self, vector: np.ndarray) -> _Page:
        with self._directory_lock:
            if not self._pages:
                page = self._new_page()
                self._pages = (page,)
                return page
            pages = self._pages
        distances = self._page_distances(vector, pages)
        page = pages[int(np.argmin(distances))]
        with page.lock:
            if page.free_slot() is not None:
                return page
        # Preserve geometric locality: split the chosen page rather than
        # spilling into an unrelated page that happens to have a free slot.
        self._split_page(page)
        return self._choose_page(vector)

    def insert(
        self,
        item_id: int,
        vector: Iterable[float] | np.ndarray,
        *,
        timestamp: int | None = None,
    ) -> int:
        value = self._prepare(vector)
        ts = self._tick(timestamp)
        # Upsert semantics: close the current version at the same commit time.
        self._close_current_version(item_id, ts, required=False)
        while True:
            page = self._choose_page(value)
            with page.lock:
                slot = page.free_slot()
                if slot is None:
                    continue
                page.ids[slot] = int(item_id)
                page.vectors[slot] = value
                page.begin_ts[slot] = ts
                page.end_ts[slot] = MAX_TS
                page.occupied[slot] = True
                page.add_to_geometry(value, self.metric)
                with self._directory_lock:
                    self._locations.setdefault(int(item_id), []).append((page, slot))
                    self._stats["inserts"] += 1
                return ts

    def _close_current_version(self, item_id: int, ts: int, *, required: bool) -> None:
        with self._directory_lock:
            locations = list(self._locations.get(int(item_id), ()))
        found = False
        for page, slot in reversed(locations):
            with page.lock:
                if (
                    page.occupied[slot]
                    and page.ids[slot] == item_id
                    and page.end_ts[slot] == MAX_TS
                ):
                    page.end_ts[slot] = ts
                    page.remove_from_geometry(page.vectors[slot], self.metric)
                    with self._directory_lock:
                        self._retired.append((ts, int(item_id), int(page.begin_ts[slot])))
                    found = True
                    break
        if required and not found:
            raise KeyError(item_id)

    def delete(self, item_id: int, *, timestamp: int | None = None) -> int:
        ts = self._tick(timestamp)
        self._close_current_version(int(item_id), ts, required=True)
        with self._directory_lock:
            self._stats["deletes"] += 1
        return ts

    def _split_page(self, page: _Page) -> None:
        with self._directory_lock:
            if page not in self._pages:
                return
            with page.lock:
                slots = np.flatnonzero(page.occupied)
                if slots.size < 2:
                    return
                vectors = page.vectors[slots]
                # Deterministic farthest-pair initialization.
                seed_a = vectors[0]
                if self.metric == "cosine":
                    first_dist = 1.0 - vectors @ seed_a
                else:
                    first_dist = np.linalg.norm(vectors - seed_a, axis=1)
                seed_b = vectors[int(np.argmax(first_dist))]
                centers = np.vstack((seed_a, seed_b)).copy()
                labels = np.zeros(slots.size, dtype=np.int8)
                for _ in range(self.split_iterations):
                    if self.metric == "cosine":
                        scores = vectors @ centers.T
                        labels = np.argmax(scores, axis=1).astype(np.int8)
                    else:
                        d0 = np.linalg.norm(vectors - centers[0], axis=1)
                        d1 = np.linalg.norm(vectors - centers[1], axis=1)
                        labels = (d1 < d0).astype(np.int8)
                    if np.all(labels == labels[0]):
                        labels[np.argmax(first_dist)] = 1 - labels[0]
                    for label in (0, 1):
                        group = vectors[labels == label]
                        if group.size:
                            centers[label] = group.mean(axis=0)
                            if self.metric == "cosine":
                                norm = np.linalg.norm(centers[label])
                                if norm:
                                    centers[label] /= norm

                left, right = self._new_page(), self._new_page()
                remapped: list[tuple[int, _Page, int]] = []
                for raw_slot, raw_label in zip(slots.tolist(), labels.tolist(), strict=True):
                    source_slot = int(raw_slot)
                    target = left if int(raw_label) == 0 else right
                    target_slot = target.free_slot()
                    assert target_slot is not None
                    target.ids[target_slot] = page.ids[source_slot]
                    target.vectors[target_slot] = page.vectors[source_slot]
                    target.begin_ts[target_slot] = page.begin_ts[source_slot]
                    target.end_ts[target_slot] = page.end_ts[source_slot]
                    target.occupied[target_slot] = True
                    remapped.append((int(page.ids[source_slot]), target, target_slot))
                left.recompute_geometry(self.metric)
                right.recompute_geometry(self.metric)

            new_pages = [candidate for candidate in self._pages if candidate is not page]
            new_pages.extend((left, right))
            self._pages = tuple(new_pages)
            # Repair only IDs held by the split page rather than rebuilding the
            # global map. This is bounded by page capacity.
            for item_id, target, target_slot in remapped:
                old_locations = self._locations.get(item_id, [])
                kept = [
                    (old_page, old_slot)
                    for old_page, old_slot in old_locations
                    if old_page is not page
                ]
                kept.append((target, target_slot))
                self._locations[item_id] = kept
            self._stats["splits"] += 1

    def _rebuild_locations_locked(self) -> None:
        locations: dict[int, list[tuple[_Page, int]]] = {}
        for page in self._pages:
            with page.lock:
                for slot in np.flatnonzero(page.occupied):
                    locations.setdefault(int(page.ids[slot]), []).append((page, int(slot)))
        self._locations = locations

    def search(
        self,
        vector: Iterable[float] | np.ndarray,
        k: int = 10,
        *,
        snapshot: int | None = None,
        nprobe: int | None = None,
        predicate: Callable[[int], bool] | None = None,
        adaptive: bool = True,
    ) -> list[SearchResult]:
        if k <= 0:
            return []
        query = self._prepare(vector)
        snap = self._clock if snapshot is None else int(snapshot)
        with self._directory_lock:
            pages = self._pages
            self._stats["searches"] += 1
        if not pages:
            return []
        page_distances = self._page_distances(query, pages)
        order = np.argsort(page_distances)
        initial_probe = min(len(pages), max(1, nprobe or self.nprobe))
        probe_count = initial_probe
        processed = 0
        id_batches: list[np.ndarray] = []
        vector_batches: list[np.ndarray] = []
        candidate_count = 0

        while True:
            for page_offset in order[processed:probe_count]:
                page = pages[int(page_offset)]
                with page.lock:
                    mask = page.visible_mask(snap)
                    ids = page.ids[mask].copy()
                    vectors = page.vectors[mask].copy()
                if predicate is not None and ids.size:
                    keep = np.fromiter((predicate(int(i)) for i in ids), dtype=bool, count=ids.size)
                    ids, vectors = ids[keep], vectors[keep]
                if not ids.size:
                    continue
                id_batches.append(ids)
                vector_batches.append(vectors)
                candidate_count += ids.size
            processed = probe_count
            if candidate_count >= k or not adaptive or probe_count == len(pages):
                break
            probe_count = min(len(pages), max(probe_count + 1, probe_count * 2))

        if not id_batches:
            return []
        ids = np.concatenate(id_batches)
        vectors = np.concatenate(vector_batches)
        if self.metric == "cosine":
            distances = 1.0 - vectors @ query
        else:
            distances = np.linalg.norm(vectors - query, axis=1)
        take = min(k, ids.size)
        positions = np.argpartition(distances, take - 1)[:take]
        positions = positions[np.argsort(distances[positions])]
        return [
            SearchResult(int(ids[position]), float(distances[position])) for position in positions
        ]

    def vacuum(
        self,
        oldest_snapshot: int,
        *,
        budget_pages: int | None = None,
        budget_versions: int | None = None,
    ) -> int:
        """Physically reclaim versions invisible to every permitted snapshot.

        Retired versions are ordered by commit timestamp, so reclamation does
        not scan the index. ``budget_pages`` remains as a compatibility alias
        and maps to a conservative version budget.
        """
        if budget_versions is None and budget_pages is not None:
            budget_versions = budget_pages * self.page_capacity
        reclaimed = 0
        with self._directory_lock:
            changed_pages: set[_Page] = set()
            while self._retired and self._retired[0][0] < oldest_snapshot:
                if budget_versions is not None and reclaimed >= budget_versions:
                    break
                end_ts, item_id, begin_ts = self._retired.popleft()
                locations = self._locations.get(item_id, [])
                for page, slot in locations:
                    with page.lock:
                        if (
                            page.occupied[slot]
                            and page.ids[slot] == item_id
                            and page.begin_ts[slot] == begin_ts
                            and page.end_ts[slot] == end_ts
                        ):
                            page.occupied[slot] = False
                            page.ids[slot] = -1
                            page.begin_ts[slot] = 0
                            page.end_ts[slot] = 0
                            changed_pages.add(page)
                            reclaimed += 1
                            break
            for page in changed_pages:
                with page.lock:
                    page.recompute_geometry(self.metric)
            if reclaimed:
                self._rebuild_locations_locked()
                self._stats["vacuumed_versions"] += reclaimed
        return reclaimed

    def stats(self) -> dict[str, int | float]:
        with self._directory_lock:
            occupied = live = dead = 0
            for page in self._pages:
                with page.lock:
                    occupied += int(np.count_nonzero(page.occupied))
                    live += int(np.count_nonzero(page.current_mask()))
            dead = occupied - live
            allocated_slots = len(self._pages) * self.page_capacity
            bytes_per_page = (
                self.page_capacity * 8
                + self.page_capacity * self.dimensions * 4
                + self.page_capacity * 8 * 2
                + self.page_capacity
                + self.dimensions * 4
            )
            return {
                **self._stats,
                "clock": self._clock,
                "pages": len(self._pages),
                "live_vectors": live,
                "retained_versions": dead,
                "occupied_slots": occupied,
                "allocated_slots": allocated_slots,
                "allocated_page_bytes": len(self._pages) * bytes_per_page,
                "space_amplification": (occupied / live) if live else 0.0,
                "capacity_amplification": (allocated_slots / live) if live else 0.0,
            }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with self._directory_lock:
            pages = self._pages
            np.savez(
                path,
                dimensions=np.array(self.dimensions),
                metric=np.array(self.metric),
                page_capacity=np.array(self.page_capacity),
                nprobe=np.array(self.nprobe),
                clock=np.array(self._clock),
                ids=np.stack([p.ids for p in pages])
                if pages
                else np.empty((0, self.page_capacity), dtype=np.int64),
                vectors=np.stack([p.vectors for p in pages])
                if pages
                else np.empty((0, self.page_capacity, self.dimensions), dtype=np.float32),
                begin_ts=np.stack([p.begin_ts for p in pages])
                if pages
                else np.empty((0, self.page_capacity), dtype=np.int64),
                end_ts=np.stack([p.end_ts for p in pages])
                if pages
                else np.empty((0, self.page_capacity), dtype=np.int64),
                occupied=np.stack([p.occupied for p in pages])
                if pages
                else np.empty((0, self.page_capacity), dtype=bool),
            )

    @classmethod
    def load(cls, path: str | Path) -> ChronoVecIndex:
        data = np.load(Path(path), allow_pickle=False)
        metric = str(data["metric"])
        if metric not in ("cosine", "l2"):
            raise ValueError(f"unsupported metric in checkpoint: {metric!r}")
        metric_value = cast(Metric, metric)
        index = cls(
            int(data["dimensions"]),
            metric=metric_value,
            page_capacity=int(data["page_capacity"]),
            nprobe=int(data["nprobe"]),
        )
        with index._directory_lock:
            index._clock = int(data["clock"])
            restored = []
            for page_idx in range(data["ids"].shape[0]):
                page = index._new_page()
                page.ids[:] = data["ids"][page_idx]
                page.vectors[:] = data["vectors"][page_idx]
                page.begin_ts[:] = data["begin_ts"][page_idx]
                page.end_ts[:] = data["end_ts"][page_idx]
                page.occupied[:] = data["occupied"][page_idx]
                page.recompute_geometry(index.metric)
                restored.append(page)
            index._pages = tuple(restored)
            index._rebuild_locations_locked()
            retired = []
            for page in index._pages:
                for slot in np.flatnonzero(page.occupied & (page.end_ts != MAX_TS)):
                    retired.append(
                        (int(page.end_ts[slot]), int(page.ids[slot]), int(page.begin_ts[slot]))
                    )
            index._retired = deque(sorted(retired))
        return index


# Backward-compatible alias. Import _ReferenceIndex directly for new code.
ChronoVecIndex = _ReferenceIndex
