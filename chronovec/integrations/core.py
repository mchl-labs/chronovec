"""Framework-agnostic document store used by every integration adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..embeddings import EmbeddingInput, as_embedding_function
from ..memory import AgentMemory, _View

Embedder = EmbeddingInput


@dataclass
class StoredDocument:
    """A document as the adapters see it."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None


def _stable_id(value: str) -> int:
    """Map an arbitrary string id onto the int64 the index uses.

    Blake2b truncated to 63 bits: deterministic across processes, so a document
    keeps its identity across restarts, and non-negative so it never collides
    with the -1 the index uses for an empty slot.
    """
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


class ChronoVecStore:
    """Text-in, text-out document store with branching.

    Adapters wrap this; it does not depend on any framework.
    """

    def __init__(
        self,
        embed: Embedder,
        dimensions: int,
        *,
        metric: str = "cosine",
        nprobe: int = 32,
        memory: AgentMemory | None = None,
    ) -> None:
        self._embedder = as_embedding_function(embed)
        self._memory = memory or AgentMemory(dimensions, metric=metric, nprobe=nprobe)
        self._nprobe = nprobe
        self._ids: dict[int, str] = {}

    # -- writing ---------------------------------------------------------
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[str] | None = None,
        view: _View | None = None,
    ) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must be the same length as texts")
        if ids is not None and len(ids) != len(texts):
            raise ValueError("ids must be the same length as texts")
        vectors = np.asarray(self._embedder.embed_documents(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("embedder must return one vector per text")
        target = view or self._memory.main
        assigned: list[str] = []
        for position, text in enumerate(texts):
            external = ids[position] if ids else f"doc-{_stable_id(text):x}"
            internal = _stable_id(external)
            payload = dict(metadatas[position]) if metadatas else {}
            target.add(internal, vectors[position], text=text, **payload)
            self._ids[internal] = external
            assigned.append(external)
        return assigned

    def delete(self, ids: Iterable[str]) -> int:
        removed = 0
        for external in ids:
            internal = _stable_id(external)
            try:
                self._memory.delete(internal)
                removed += 1
            except RuntimeError:
                continue
        return removed

    # -- reading ---------------------------------------------------------
    def similarity_search(
        self, query: str, k: int = 4, *, view: _View | None = None, as_of: int | None = None
    ) -> list[StoredDocument]:
        vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        target = view or self._memory.main
        hits = target.search(vector, k=k, nprobe=self._nprobe, as_of=as_of)
        out: list[StoredDocument] = []
        for hit, record in hits:
            payload = dict(record.payload)
            text = payload.pop("text", "")
            out.append(
                StoredDocument(
                    id=self._ids.get(hit.id, str(hit.id)),
                    text=text,
                    metadata=payload,
                    score=hit.distance,
                )
            )
        return out

    # -- branching -------------------------------------------------------
    def branch(self, name: str) -> _View:
        """Fork the store. Writes on the returned view stay invisible to main."""
        return self._memory.branch(name)

    @property
    def memory(self) -> AgentMemory:
        return self._memory

    def close(self) -> None:
        self._memory.close()
