"""LangChain ``VectorStore`` backed by ChronoVec.

Beyond the standard interface this exposes :meth:`ChronoVecVectorStore.branch`,
which has no LangChain equivalent: it forks the store so an agent can write
speculatively and then discard or merge the branch without rebuilding an index.

    store = ChronoVecVectorStore(embedding=embeddings, dimensions=384)
    store.add_texts(["the user prefers dark mode"])

    with store.branch("hypothesis") as scratch:
        scratch.add_texts(["the user might want light mode"])
        scratch.similarity_search("theme")     # sees both
    store.similarity_search("theme")           # speculation was discarded
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .core import ChronoVecStore, StoredDocument


def _document_class():
    try:
        from langchain_core.documents import Document
    except ImportError as exc:  # pragma: no cover - exercised only without langchain
        raise ImportError(
            "langchain-core is required for chronovec.integrations.langchain; "
            "install it with `pip install langchain-core`"
        ) from exc
    return Document


def _base_class():
    """Subclass LangChain's VectorStore when it is installed.

    Duck typing is not enough here. `as_retriever` lives on the base class, and
    LCEL chains and agent toolkits type-check what they are handed, so a store
    that merely has the right method names cannot be dropped into a chain --
    which is the entire point of shipping an adapter. Without langchain-core
    the class still exists and works standalone; only the base is dropped.
    """
    try:
        from langchain_core.vectorstores import VectorStore
    except ImportError:  # pragma: no cover - exercised only without langchain
        return object
    return VectorStore


class ChronoVecVectorStore(_base_class()):  # type: ignore[misc]
    """LangChain-compatible vector store with branching.

    Implements the subset of ``VectorStore`` that agent workflows use:
    ``add_texts``, ``delete``, ``similarity_search``,
    ``similarity_search_with_score`` and ``from_texts``.
    """

    def __init__(
        self, embedding: Any, dimensions: int, *, metric: str = "cosine", nprobe: int = 32
    ) -> None:
        self.embedding = embedding
        self._store = ChronoVecStore(embedding, dimensions, metric=metric, nprobe=nprobe)

    # -- VectorStore surface ---------------------------------------------
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        *,
        ids: Sequence[str] | None = None,
        **_: Any,
    ) -> list[str]:
        return self._store.add_texts(texts, metadatas, ids)

    @property
    def embeddings(self) -> Any:
        """LangChain reads this to reuse the embedder for queries."""
        return self.embedding

    def delete(self, ids: Sequence[str] | None = None, **_: Any) -> bool:
        if not ids:
            return False
        return self._store.delete(ids) > 0

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Any]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Any, float]]:
        Document = _document_class()
        found: list[StoredDocument] = self._store.similarity_search(
            query, k, view=kwargs.get("view"), as_of=kwargs.get("as_of")
        )
        return [
            (
                Document(page_content=d.text, metadata={**d.metadata, "id": d.id}),
                float(d.score if d.score is not None else 0.0),
            )
            for d in found
        ]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        dimensions: int | None = None,
        **kwargs: Any,
    ) -> ChronoVecVectorStore:
        if dimensions is None:
            probe = embedding.embed_documents([texts[0] if texts else ""])
            dimensions = len(probe[0])
        store = cls(embedding, dimensions, **kwargs)
        store.add_texts(texts, metadatas)
        return store

    # -- the part LangChain has no equivalent for -------------------------
    @contextmanager
    def branch(self, name: str) -> Iterator[_BranchedStore]:
        """Fork the store for the duration of the block.

        Writes inside the block are invisible to the main store and to sibling
        branches. Leaving the block discards them unless ``keep()`` was called.
        """
        view = self._store.branch(name)
        wrapper = _BranchedStore(self._store, view)
        try:
            yield wrapper
        finally:
            wrapper._finish()

    def as_of(self, snapshot: int) -> _SnapshotStore:
        """A read-only view of the store as it was at ``snapshot``."""
        return _SnapshotStore(self._store, snapshot)

    @property
    def clock(self) -> int:
        return self._store.memory.clock

    def close(self) -> None:
        self._store.close()


class _BranchedStore:
    """The store as seen from inside a branch."""

    def __init__(self, store: ChronoVecStore, view: Any) -> None:
        self._store = store
        self._view = view
        self._keep = False
        self._done = False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        *,
        ids: Sequence[str] | None = None,
    ) -> list[str]:
        return self._store.add_texts(texts, metadatas, ids, view=self._view)

    def similarity_search(self, query: str, k: int = 4) -> list[StoredDocument]:
        return self._store.similarity_search(query, k, view=self._view)

    def keep(self) -> None:
        """Promote this branch's writes to the main store on exit."""
        self._keep = True

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        self._view.merge() if self._keep else self._view.discard()


class _SnapshotStore:
    """Read-only view of the store at a past snapshot."""

    def __init__(self, store: ChronoVecStore, snapshot: int) -> None:
        self._store = store
        self._snapshot = snapshot

    def similarity_search(self, query: str, k: int = 4) -> list[StoredDocument]:
        return self._store.similarity_search(query, k, as_of=self._snapshot)
