"""A collection API: string ids, metadata, and filtering, over the fast core.

:class:`~chronovec.Index` is the engine. It takes int64 ids and bare float
arrays, which is what makes it fast and what makes it awkward: real
applications have string ids, carry metadata beside each vector, and want to
filter on it. Reaching for Chroma to get those and losing snapshot isolation to
get them is a bad trade, so they are provided here.

Nothing in this module is on the hot path of a raw
:meth:`~chronovec.Index.search`. The engine is unchanged; this is a layer over
it, and anyone who wants the last microsecond can keep using the engine
directly.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import EmbeddingInput, as_embedding_function
from .native import NativeChronoVecIndex

Embedder = EmbeddingInput

#: Operators accepted inside a ``where`` clause, mirroring the vocabulary most
#: vector stores use so a filter can be moved across without rewriting.
_OPERATORS = {
    "$eq": lambda have, want: have == want,
    "$ne": lambda have, want: have != want,
    "$gt": lambda have, want: have is not None and have > want,
    "$gte": lambda have, want: have is not None and have >= want,
    "$lt": lambda have, want: have is not None and have < want,
    "$lte": lambda have, want: have is not None and have <= want,
    "$in": lambda have, want: have in want,
    "$nin": lambda have, want: have not in want,
    "$contains": lambda have, want: have is not None and want in have,
    "$regex": lambda have, want: have is not None and re.search(want, str(have)) is not None,
}


def stable_id(value: str | int) -> int:
    """Map an id onto the int64 the engine uses.

    Integers pass through, so a caller already using int ids keeps them
    readable in the engine's own stats. Strings are hashed with blake2b
    truncated to 63 bits: deterministic across processes, so a record keeps its
    identity across restarts, and never negative, so it cannot collide with the
    -1 the engine writes into an empty slot.
    """
    if isinstance(value, bool):
        raise TypeError("id must be a string or integer, not bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if not isinstance(value, str):
        raise TypeError(f"id must be a string or integer, got {type(value).__name__}")
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


def matches(metadata: Mapping[str, Any], where: Mapping[str, Any] | None) -> bool:
    """Whether one record's metadata satisfies a ``where`` clause."""
    if not where:
        return True
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    if not isinstance(where, Mapping):
        raise TypeError("where must be a mapping")
    for key, condition in where.items():
        if key == "$and":
            if not isinstance(condition, Sequence) or isinstance(condition, (str, bytes)):
                raise ValueError("$and expects a sequence of filter mappings")
            if not all(matches(metadata, part) for part in condition):
                return False
            continue
        if key == "$or":
            if not isinstance(condition, Sequence) or isinstance(condition, (str, bytes)):
                raise ValueError("$or expects a sequence of filter mappings")
            if not any(matches(metadata, part) for part in condition):
                return False
            continue
        have = metadata.get(key)
        if isinstance(condition, Mapping):
            for operator, want in condition.items():
                check = _OPERATORS.get(operator)
                if check is None:
                    raise ValueError(f"unknown operator {operator!r}")
                if operator in {"$in", "$nin"} and (
                    isinstance(want, (str, bytes, Mapping)) or not isinstance(want, Sequence)
                ):
                    raise ValueError(f"{operator} expects a sequence")
                if not check(have, want):
                    return False
        elif have != condition:
            return False
    return True


@dataclass
class Record:
    """One result. Attribute access, and indexable for dict-style callers."""

    id: str | int
    # Named as the engine names it. A second vocabulary for the same number is
    # how "score" comes to mean similarity in one place and distance in another.
    distance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    document: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


class Collection:
    """Vectors with ids, metadata and filtering, plus snapshots.

    >>> collection = Collection(dimensions=3, metric="l2")
    >>> _ = collection.add(ids=["a", "b"], embeddings=[[1, 0, 0], [0, 1, 0]],
    ...                    metadatas=[{"lang": "en"}, {"lang": "fr"}])
    >>> [r.id for r in collection.query([1, 0, 0], k=1)]
    ['a']
    >>> [r.id for r in collection.query([1, 0, 0], k=1, where={"lang": "fr"})]
    ['b']
    """

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "cosine",
        name: str = "chronovec",
        embedding_function: EmbeddingInput | None = None,
        wal_path: str | None = None,
        arena_path: str | None = None,
        max_vectors: int = 0,
        growth_factor: float = 2.0,
        checkpoint_path: str | os.PathLike[str] | None = None,
        **index_options: Any,
    ) -> None:
        self.name = name
        self.dimensions = int(dimensions)
        self._embedder = (
            as_embedding_function(embedding_function) if embedding_function is not None else None
        )
        self._wal_path = wal_path
        self._arena_path = arena_path
        self._max_vectors = int(max_vectors)
        self._growth_factor = float(growth_factor)
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not np.isfinite(self._growth_factor) or self._growth_factor <= 1:
            raise ValueError("growth_factor must be greater than 1")
        # Stored so _grow_arena can recreate the native index with the same
        # options (minus arena/wal/max_vectors, which are managed separately).
        self._index_kwargs: dict[str, Any] = dict(metric=metric, **index_options)
        if wal_path is not None:
            index_options["wal_path"] = wal_path
        if arena_path is not None:
            index_options["arena_path"] = arena_path
            index_options["max_vectors"] = max_vectors
        self._index = NativeChronoVecIndex(dimensions, metric=metric, **index_options)
        # Everything the engine does not carry, keyed by the engine's int id.
        #
        # Versioned, not overwritten. The engine keeps old vector versions
        # readable from a snapshot; if the metadata beside them were replaced
        # in place, a snapshot read would return a historical vector dressed in
        # current metadata. That is worse than not offering snapshots, because
        # it looks like it worked.
        self._history: dict[int, list[tuple[int, dict[str, Any], str | None]]] = {}
        self._names: dict[int, str | int] = {}
        self._live: set[int] = set()
        # When each removed record stopped being live. Without it, pruning
        # cannot tell whether a reader at the horizon can still see a deleted
        # record, and dropping its history too early makes a snapshot read
        # return a vector with no id, no document and no metadata -- the
        # snapshot appears to work and returns nothing useful.
        self._deleted_at: dict[int, int] = {}

    # -- helpers ---------------------------------------------------------
    def _vectors(self, embeddings: Any, documents: Sequence[str] | None, count: int) -> np.ndarray:
        if embeddings is None:
            if documents is None or self._embedder is None:
                raise ValueError(
                    "pass embeddings, or construct the collection with an "
                    "embedding_function and pass documents"
                )
            embeddings = self._embedder.embed_documents(list(documents))
        block = np.asarray(embeddings, dtype=np.float32)
        if block.ndim == 1:
            block = block[None, :]
        if block.shape != (count, self.dimensions):
            raise ValueError(
                f"expected {count} vectors of width {self.dimensions}, got {block.shape}"
            )
        return np.ascontiguousarray(block)

    @staticmethod
    def _listify(value: Any) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes, Mapping)):
            return [value]
        return list(value)

    def _version(self, key: int, snapshot: int | None) -> tuple[dict[str, Any], str | None]:
        """The metadata and document in force at `snapshot`, or now."""
        versions = self._history.get(key)
        if not versions:
            return {}, None
        if snapshot is None:
            return versions[-1][1], versions[-1][2]
        for stamp, metadata, document in reversed(versions):
            if stamp <= snapshot:
                return metadata, document
        # The engine surfaced a version older than anything recorded here.
        return versions[0][1], versions[0][2]

    # -- writing ---------------------------------------------------------
    def add(
        self,
        ids: Sequence[str | int] | str | int,
        *,
        embeddings: Any = None,
        documents: Sequence[str] | str | None = None,
        metadatas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> list[str | int]:
        """Insert records. An id that already exists is replaced.

        Replacement is a real update in the engine -- one operation that closes
        the old version and opens a new one -- so a reader on an older snapshot
        still sees what was there before.
        """
        names = self._listify(ids) or []
        documents = self._listify(documents)
        metadatas = self._listify(metadatas)
        if not names:
            return []
        if documents is not None and len(documents) != len(names):
            raise ValueError("documents must be the same length as ids")
        if metadatas is not None and len(metadatas) != len(names):
            raise ValueError("metadatas must be the same length as ids")
        if metadatas is not None and any(not isinstance(item, Mapping) for item in metadatas):
            raise TypeError("every metadata item must be a mapping")
        vectors = self._vectors(embeddings, documents, len(names))

        keys = np.array([stable_id(one) for one in names], dtype=np.int64)
        seen: dict[int, int] = {}
        for position, key in enumerate(keys):
            if int(key) in seen:
                raise ValueError(f"id {names[position]!r} appears twice in one call")
            seen[int(key)] = position
        try:
            self._index.insert_many(keys, vectors)
        except RuntimeError as exc:
            if "arena is full" not in str(exc):
                raise
            if not (self._arena_path and self._wal_path and self._growth_factor > 1):
                raise MemoryError(
                    f"vector arena is full ({self._max_vectors} vectors); "
                    "increase max_vectors at construction, or set both "
                    "wal_path and growth_factor to enable automatic growth"
                ) from exc
            self._grow_arena()
            self._index.insert_many(keys, vectors)
        # One stamp for the batch. Snapshots are only obtainable between calls,
        # so no reader can land inside a batch and see a partial one.
        stamp = int(self._index.clock)
        for position, key in enumerate(keys):
            key = int(key)
            previous = self._history.get(key)
            carried_meta = dict(previous[-1][1]) if previous else {}
            carried_doc = previous[-1][2] if previous else None
            metadata = dict(metadatas[position]) if metadatas is not None else carried_meta
            document = documents[position] if documents is not None else carried_doc
            self._history.setdefault(key, []).append((stamp, metadata, document))
            self._names[key] = names[position]
            self._live.add(key)
            self._deleted_at.pop(key, None)  # alive again
        self._checkpoint()
        return names

    upsert = add  # the engine has no separate insert-if-absent

    def update(
        self,
        ids: Sequence[str | int] | str | int,
        *,
        embeddings: Any = None,
        documents: Sequence[str] | str | None = None,
        metadatas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> None:
        """Change existing records. Unknown ids raise rather than being added."""
        names = self._listify(ids) or []
        missing = [one for one in names if stable_id(one) not in self._live]
        if missing:
            raise KeyError(f"unknown ids: {missing[:5]}")
        if embeddings is None and documents is None:
            # Metadata-only edit: nothing for the engine to do.
            metadatas = self._listify(metadatas)
            if metadatas is None:
                raise ValueError("nothing to update")
            if len(metadatas) != len(names):
                raise ValueError("metadatas must be the same length as ids")
            if any(not isinstance(item, Mapping) for item in metadatas):
                raise TypeError("every metadata item must be a mapping")
            stamp = int(self._index.clock)
            for position, one in enumerate(names):
                key = stable_id(one)
                current, document = self._version(key, None)
                merged = dict(current)
                merged.update(metadatas[position])
                self._history[key].append((stamp, merged, document))
            self._checkpoint()
            return
        self.add(names, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def delete(
        self,
        ids: Sequence[str | int] | str | int | None = None,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[str | int]:
        """Remove by id, by metadata filter, or both. Returns what was removed."""
        if ids is None and where is None:
            raise ValueError("pass ids or where; refusing to delete everything")
        chosen = (
            [stable_id(one) for one in (self._listify(ids) or [])]
            if ids is not None
            else list(self._live)
        )
        removed: list[str | int] = []
        keys: list[int] = []
        for key in chosen:
            if key not in self._live:
                continue
            if where and not matches(self._version(key, None)[0], where):
                continue
            keys.append(key)
            removed.append(self._names[key])
        if keys:
            self._index.delete_many(np.array(keys, dtype=np.int64))
            stamp = int(self._index.clock)
            # History is kept. A snapshot taken before the delete can still
            # read these vectors, and it must be able to read their metadata.
            for key in keys:
                self._live.discard(key)
                self._deleted_at[key] = stamp
            self._checkpoint()
        return removed

    # -- reading ---------------------------------------------------------
    def query(
        self,
        query: Any = None,
        k: int = 10,
        *,  # noqa: PYI041
        query_text: str | None = None,
        where: Mapping[str, Any] | None = None,
        snapshot: int | None = None,
        overfetch: int = 8,
        **search_options: Any,
    ) -> list[Record]:
        """Nearest records, optionally filtered and optionally as of a snapshot.

        A ``where`` clause is applied after the search, over a widened
        candidate set. That is honest about what it is: the engine ranks by
        distance, and a filter that excludes almost everything can return fewer
        than `k`. `overfetch` controls how much wider the search goes; raising
        it trades work for a better chance of filling `k`.
        """
        if isinstance(query, str):
            if self._embedder is None:
                raise ValueError(
                    "received a string as query; pass query_text= for text "
                    "search, or construct the collection with an "
                    "embedding_function so strings are embedded automatically"
                )
            query_text, query = query, None
        if query is None:
            if query_text is None or self._embedder is None:
                raise ValueError("pass query, or query_text with an embedding_function")
            query = self._embedder.embed_query(query_text)
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dimensions:
            raise ValueError(f"query has width {vector.shape[0]}, expected {self.dimensions}")
        want = k * max(1, overfetch) if where else k
        found = self._index.search(vector, k=want, snapshot=snapshot, **search_options)
        out: list[Record] = []
        for hit in found:
            key = int(hit.id)
            metadata, document = self._version(key, snapshot)
            if where and not matches(metadata, where):
                continue
            out.append(
                Record(
                    id=self._names.get(key, key),
                    distance=hit.distance,
                    metadata=dict(metadata),
                    document=document,
                )
            )
            if len(out) == k:
                break
        return out

    def get(
        self,
        ids: Sequence[str | int] | str | int | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Record]:
        """Fetch by id or metadata, without a vector search."""
        chosen = (
            [stable_id(one) for one in (self._listify(ids) or [])]
            if ids is not None
            else list(self._live)
        )
        out: list[Record] = []
        for key in chosen:
            if key not in self._live:
                continue
            metadata, document = self._version(key, None)
            if where and not matches(metadata, where):
                continue
            out.append(Record(id=self._names[key], metadata=dict(metadata), document=document))
            if limit is not None and len(out) >= limit:
                break
        return out

    def peek(self, n: int = 5) -> list[Record]:
        return self.get(limit=n)

    def count(self) -> int:
        return len(self._live)

    # -- time ------------------------------------------------------------
    def snapshot(self) -> int:
        """A token for the present, readable later via ``query(snapshot=...)``.

        This is the thing the collection API exists to keep. Everything above
        is convenience; this is not available anywhere else.
        """
        return int(self._index.clock)

    def vacuum(self, *, keep_snapshot: int | None = None) -> int:
        """Reclaim versions no reader can reach. Returns versions freed.

        Prunes this layer's history as well as the engine's versions. Without
        that the sidecar grew without bound -- 5,000 updates to one id left
        5,001 metadata versions alive after repeated vacuums, because the
        engine reclaimed its side and nothing reclaimed this one.
        """
        horizon = keep_snapshot if keep_snapshot is not None else self._index.clock + 1
        freed = self._index.vacuum(oldest_snapshot=horizon, budget_versions=0)
        self._prune_history(horizon)
        self._checkpoint()
        return freed

    def _checkpoint(self) -> None:
        """Persist a client-managed collection after a successful mutation."""
        if self._checkpoint_path is not None:
            self.save(self._checkpoint_path)

    def _prune_history(self, horizon: int) -> None:
        """Drop sidecar versions no reader at or after `horizon` can reach.

        The newest version at or below the horizon is the one such a reader
        resolves to, so it stays; everything older than it is unreachable.
        Records whose last version is gone from the engine and unreachable here
        are dropped entirely.
        """
        for key in list(self._history):
            versions = self._history[key]
            keep_from = 0
            for position, (stamp, _, _) in enumerate(versions):
                if stamp <= horizon:
                    keep_from = position
                else:
                    break
            if keep_from:
                del versions[:keep_from]
            # Drop a deleted record only once the horizon has passed the
            # moment it was deleted. Testing the last *write* instead was
            # wrong: a record written at t=5 and deleted at t=10 is alive at a
            # horizon of 7, and dropping it there lost its id and metadata
            # while the engine still returned its vector.
            gone = self._deleted_at.get(key)
            if gone is not None and gone <= horizon:
                del self._history[key]
                self._names.pop(key, None)
                self._deleted_at.pop(key, None)

    # -- lifecycle -------------------------------------------------------
    def _grow_arena(self) -> None:
        """Double the arena capacity via WAL replay into a fresh, larger file.

        Closes the current index (which syncs the WAL), removes the old arena
        file (the WAL is the source of truth and will repopulate it), then
        opens a new index with max_vectors * growth_factor. The WAL replays
        automatically on open, refilling the new, larger arena.
        """
        new_max = int(self._max_vectors * self._growth_factor)
        self._index.close()
        arena_path = self._arena_path
        if arena_path is not None and os.path.exists(arena_path):
            os.unlink(arena_path)
        self._index = NativeChronoVecIndex(
            self.dimensions,
            arena_path=self._arena_path,
            max_vectors=new_max,
            wal_path=self._wal_path,
            **self._index_kwargs,
        )
        self._max_vectors = new_max

    def flush_vectors(self) -> int:
        """Flush disk-backed vector pages so the kernel may evict them. BETA.

        Returns bytes made evictable, or 0 when not disk-backed.
        """
        return self._index.flush_vectors()

    def save(self, path: str | os.PathLike[str]) -> None:
        """Durably checkpoint the collection to disk.

        Saves the native index to ``<path>.cvec`` and the id/metadata sidecar
        to ``<path>.meta``. Restore with :meth:`load`.

        The restored collection is always heap-resident; pass ``wal_path``
        and ``arena_path`` after loading if you want to re-enable those.
        """
        p = Path(path)
        self._index.save(p.with_suffix(".cvec"))
        meta = {
            "format_version": 1,
            "name": self.name,
            "dimensions": self.dimensions,
            "_history": self._history,
            "_names": self._names,
            "_live": self._live,
            "_deleted_at": self._deleted_at,
        }
        target = p.with_suffix(".meta")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], *, embedding_function: EmbeddingInput | None = None
    ) -> Collection:
        """Restore a collection saved with :meth:`save`.

        Loads the native index from ``<path>.cvec`` and the sidecar from
        ``<path>.meta``. The ``embedding_function`` must be passed again if
        you need text queries: it is not serialised.
        """
        p = Path(path)
        native = NativeChronoVecIndex.load(p.with_suffix(".cvec"))
        meta = pickle.loads(p.with_suffix(".meta").read_bytes())
        if meta.get("format_version", 0) not in {0, 1}:
            raise ValueError(
                f"unsupported collection checkpoint format: {meta.get('format_version')}"
            )
        col: Collection = cls.__new__(cls)
        col.name = meta["name"]
        col.dimensions = meta["dimensions"]
        col._embedder = (
            as_embedding_function(embedding_function) if embedding_function is not None else None
        )
        col._index = native
        col._history = meta["_history"]
        col._names = meta["_names"]
        col._live = meta["_live"]
        col._deleted_at = meta["_deleted_at"]
        col._wal_path = None
        col._arena_path = None
        col._max_vectors = 0
        col._growth_factor = 2.0
        col._index_kwargs = {}
        col._checkpoint_path = None
        return col

    def stats(self) -> dict[str, Any]:
        return self._index.stats()

    def close(self) -> None:
        self._index.close()

    def __enter__(self) -> Collection:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __len__(self) -> int:
        return self.count()

    def __contains__(self, one: str | int) -> bool:
        return stable_id(one) in self._live

    def __repr__(self) -> str:
        return f"Collection(name={self.name!r}, dimensions={self.dimensions}, count={self.count()})"
