"""Async wrapper around Collection for use in asyncio applications.

Every method that touches the index runs in a thread pool via
``asyncio.to_thread`` so it never blocks the event loop. The underlying
Collection is thread-safe for concurrent reads; writes serialize on the MVCC
lock inside the native library.

    col = AsyncCollection(384, embedding_function=my_embed, wal_path="mem.wal")

    async with col:
        await col.add(ids=["a"], documents=["the user prefers dark mode"])
        results = await col.query(query_text="theme", k=5)
        for r in results:
            print(r.id, r.distance, r.document)

``snapshot()`` and ``count()`` are synchronous: they read an atomic integer
and never need the thread pool.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from .collection import Collection, Embedder, Record


class AsyncCollection:
    """``Collection`` with an async interface for asyncio applications."""

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "cosine",
        name: str = "chronovec",
        embedding_function: Embedder | None = None,
        wal_path: str | None = None,
        arena_path: str | None = None,
        max_vectors: int = 0,
        growth_factor: float = 2.0,
        **index_options: Any,
    ) -> None:
        self._col = Collection(
            dimensions,
            metric=metric,
            name=name,
            embedding_function=embedding_function,
            wal_path=wal_path,
            arena_path=arena_path,
            max_vectors=max_vectors,
            growth_factor=growth_factor,
            **index_options,
        )

    # -- writing ---------------------------------------------------------

    async def add(
        self,
        ids: Sequence[str | int] | str | int,
        *,
        embeddings: Any = None,
        documents: Sequence[str] | str | None = None,
        metadatas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> list[str | int]:
        return await asyncio.to_thread(
            self._col.add,
            ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    upsert = add

    async def update(
        self,
        ids: Sequence[str | int] | str | int,
        *,
        embeddings: Any = None,
        documents: Sequence[str] | str | None = None,
        metadatas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ) -> None:
        return await asyncio.to_thread(
            self._col.update,
            ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    async def delete(
        self,
        ids: Sequence[str | int] | str | int | None = None,
        *,
        where: Mapping[str, Any] | None = None,
    ) -> list[str | int]:
        return await asyncio.to_thread(self._col.delete, ids, where=where)

    # -- reading ---------------------------------------------------------

    async def query(
        self,
        query: Any = None,
        k: int = 10,
        *,
        query_text: str | None = None,
        where: Mapping[str, Any] | None = None,
        snapshot: int | None = None,
        overfetch: int = 8,
        **search_options: Any,
    ) -> list[Record]:
        return await asyncio.to_thread(
            self._col.query,
            query,
            k,
            query_text=query_text,
            where=where,
            snapshot=snapshot,
            overfetch=overfetch,
            **search_options,
        )

    async def get(
        self,
        ids: Sequence[str | int] | str | int | None = None,
        *,
        where: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[Record]:
        return await asyncio.to_thread(self._col.get, ids, where=where, limit=limit)

    async def peek(self, n: int = 5) -> list[Record]:
        return await asyncio.to_thread(self._col.peek, n)

    # -- time ------------------------------------------------------------

    def snapshot(self) -> int:
        """Return the current clock value. Synchronous: reads an atomic int."""
        return self._col.snapshot()

    async def vacuum(self, *, keep_snapshot: int | None = None) -> int:
        return await asyncio.to_thread(self._col.vacuum, keep_snapshot=keep_snapshot)

    def flush_vectors(self) -> int:
        """Flush disk-backed vector pages so the kernel may evict them. BETA."""
        return self._col.flush_vectors()

    async def save(self, path: str) -> None:
        """Checkpoint the collection to disk. See :meth:`Collection.save`."""
        await asyncio.to_thread(self._col.save, path)

    @classmethod
    def load(cls, path: str, *, embedding_function: Embedder | None = None) -> AsyncCollection:
        """Restore a collection saved with :meth:`save`.

        Synchronous: WAL replay happens inside the native load call.
        Pass ``embedding_function`` again if you need text queries.
        """
        col = Collection.load(path, embedding_function=embedding_function)
        instance: AsyncCollection = cls.__new__(cls)
        instance._col = col
        return instance

    # -- introspection ---------------------------------------------------

    def count(self) -> int:
        return self._col.count()

    def stats(self) -> dict[str, Any]:
        return self._col.stats()

    @property
    def name(self) -> str:
        return self._col.name

    @property
    def dimensions(self) -> int:
        return self._col.dimensions

    @property
    def sync(self) -> Collection:
        """The underlying synchronous Collection for callers that need it."""
        return self._col

    # -- lifecycle -------------------------------------------------------

    async def close(self) -> None:
        await asyncio.to_thread(self._col.close)

    async def __aenter__(self) -> AsyncCollection:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def __len__(self) -> int:
        return self._col.count()

    def __contains__(self, one: str | int) -> bool:
        return one in self._col

    def __repr__(self) -> str:
        return repr(self._col).replace("Collection(", "AsyncCollection(", 1)
