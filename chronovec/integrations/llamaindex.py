"""LlamaIndex vector store backed by ChronoVec.

Implements the ``add`` / ``query`` / ``delete`` surface LlamaIndex expects, and
adds branching and snapshot reads, which LlamaIndex has no primitive for.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .core import ChronoVecStore, _stable_id


def _base_class():
    """Subclass LlamaIndex's store base when it is installed.

    Duck typing is not enough. `VectorStoreIndex.from_vector_store` and the
    retriever machinery type-check what they are handed, so a store with the
    right method names but the wrong base cannot be plugged into an index --
    which is the only reason to ship an adapter. Without llama-index-core the
    class still exists and works standalone; only the base is dropped.

    The base is a pydantic model, so the subclass declares its fields and calls
    super().__init__ before touching anything of its own.
    """
    try:
        from llama_index.core.vector_stores.types import BasePydanticVectorStore
    except ImportError:  # pragma: no cover - exercised only without llama-index
        return object
    return BasePydanticVectorStore


class ChronoVecLlamaStore(_base_class()):  # type: ignore[misc]
    """LlamaIndex-compatible store.

    Nodes arrive already embedded, so no embedder is required; the callable
    handed to :class:`ChronoVecStore` is only used for text queries and raises
    if one is attempted without an embedder.
    """

    # False, and deliberately: the store returns ids and similarities, not
    # reconstructed TextNodes. Claiming to store text makes LlamaIndex expect
    # nodes back from query() and skip its own docstore, so the retriever gets
    # nothing. Text is still kept here and readable through the branch API.
    stores_text: bool = False
    is_embedding_query: bool = True

    def __init__(
        self, dimensions: int, *, metric: str = "cosine", nprobe: int = 32, embed: Any = None
    ) -> None:
        base = type(self).__mro__[1]
        if base is not object:
            # Pydantic owns __init__ on the base, and attribute assignment
            # before it runs is rejected.
            base.__init__(self, stores_text=False, is_embedding_query=True)  # type: ignore[misc]

        def _embed(texts: Sequence[str]):
            if embed is None:
                raise ValueError(
                    "this store was created without an embedder; query by "
                    "embedding via query(), or pass embed= to use text queries"
                )
            return embed(list(texts))

        self._store = ChronoVecStore(_embed, dimensions, metric=metric, nprobe=nprobe)
        self._nprobe = nprobe

    @property
    def client(self) -> Any:
        return self._store.memory

    # -- LlamaIndex surface ----------------------------------------------
    def add(self, nodes: Sequence[Any], **_: Any) -> list[str]:
        """Store nodes that already carry embeddings."""
        added: list[str] = []
        view = self._store.memory.main
        for node in nodes:
            embedding = getattr(node, "embedding", None)
            if embedding is None:
                raise ValueError("node has no embedding; embed nodes before add()")
            node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
            if node_id is None:
                raise ValueError("node has no id")
            text = getattr(node, "text", "") or ""
            metadata = dict(getattr(node, "metadata", {}) or {})
            internal = _stable_id(str(node_id))
            view.add(internal, np.asarray(embedding, dtype=np.float32), text=text, **metadata)
            self._store._ids[internal] = str(node_id)
            added.append(str(node_id))
        return added

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        self._store.delete([ref_doc_id])

    def query(self, query: Any, **_: Any) -> Any:
        """Answer a ``VectorStoreQuery``.

        Returns a lightweight result object with ``ids`` and ``similarities``;
        callers holding real LlamaIndex types can map ids back to nodes.
        """
        embedding = getattr(query, "query_embedding", None)
        top_k = getattr(query, "similarity_top_k", 10) or 10
        if embedding is None:
            raise ValueError("query_embedding is required")
        hits = self._store.memory.main.search(
            np.asarray(embedding, dtype=np.float32), k=top_k, nprobe=self._nprobe
        )
        ids = [self._store._ids.get(hit.id, str(hit.id)) for hit, _ in hits]
        # LlamaIndex expects similarity, not distance
        similarities = [float(1.0 - hit.distance) for hit, _ in hits]
        try:
            from llama_index.core.vector_stores.types import VectorStoreQueryResult
        except ImportError:  # pragma: no cover - only without llama-index
            return _QueryResult(ids=ids, similarities=similarities)
        # The real type, not a look-alike: the retriever reads these fields off
        # a VectorStoreQueryResult and a duck-typed stand-in does not reach it.
        return VectorStoreQueryResult(nodes=None, ids=ids, similarities=similarities)

    # -- the parts LlamaIndex has no equivalent for -----------------------
    @contextmanager
    def branch(self, name: str) -> Iterator[Any]:
        view = self._store.branch(name)
        keep = {"value": False}

        class _Branch:
            def add_embedding(self_inner, node_id: str, embedding, **metadata: Any):
                internal = _stable_id(str(node_id))
                view.add(internal, np.asarray(embedding, dtype=np.float32), **metadata)
                self._store._ids[internal] = str(node_id)

            def search(self_inner, embedding, k: int = 10):
                return view.search(np.asarray(embedding, dtype=np.float32), k=k)

            def keep(self_inner) -> None:
                keep["value"] = True

        try:
            yield _Branch()
        finally:
            view.merge() if keep["value"] else view.discard()

    def close(self) -> None:
        self._store.close()


class _QueryResult:
    """Minimal stand-in for ``VectorStoreQueryResult``."""

    def __init__(self, ids: list[str], similarities: list[float]) -> None:
        self.ids = ids
        self.similarities = similarities
        self.nodes = None

    def __repr__(self) -> str:
        return f"_QueryResult(ids={self.ids!r}, similarities={self.similarities!r})"
