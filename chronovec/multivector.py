"""Multi-vector (ColBERT-style late interaction) retrieval. BETA.

Late interaction scores a query's token vectors against a document's token
vectors with Chamfer similarity: for each query token, the best matching
document token, summed:

    Chamfer(Q, D) = sum_{q in Q} max_{d in D} <q, d>

Done natively this needs a different index. MUVERA (Google Research, 2024)
avoids that by reducing multi-vector retrieval to *single-vector* search: it
builds one Fixed Dimensional Encoding (FDE) per document whose inner product
approximates Chamfer, retrieves with an ordinary ANN index, and reranks the
shortlist exactly.

That composes with ChronoVec rather than changing it. The FDE is just a vector,
so everything already built applies unchanged: MVCC versions a document's tokens
atomically, attribute filters work, and bounded reclamation covers token sets.

How the encoding works. Random hyperplanes split the space into 2**k_sim
buckets; a token lands in the bucket named by the signs of its projections.
Query buckets *sum* their tokens and document buckets *average* theirs, so their
inner product accumulates one query-token-to-document-token match per bucket;
an approximation of the per-token max. Repeating with independent hyperplanes
and concatenating reduces the variance of that approximation.

Two strategies are provided and both rerank exactly; the default is *not* the
FDE one, for a measured reason.

Indexing one token per vector and gathering the documents owning the nearest
tokens -- how ColBERT and PLAID actually retrieve -- reaches parity with exact
Chamfer (top-1 1.000 over 5,000 documents) at 3.6x the speed of brute force,
provided ``nprobe`` is scaled to the number of *tokens* rather than documents.
That last point is the one that bites: 5,000 documents of 32 tokens is 160,000
indexed vectors, and probing as if there were 5,000 costs recall (top-1 0.70 at
nprobe 64, 1.000 at nprobe 512).

The MUVERA FDE path is offered but not default. On the synthetic corpus in
``experiments/multivector_eval.py`` its inner product correlates with exact
Chamfer at Pearson r of about 0.55, which is too weak to rank reliably at a
small shortlist, and adding dimensions does not fix it (r plateaus near 0.59 at
262,144 dimensions). That corpus may be adversarial -- real contextual token
embeddings are far more clustered than synthetic ones -- so the path is kept and
labelled rather than deleted. Measure on your own data before choosing it.

Beta: the API may change, token vectors are held in memory rather than in
SQLite, and no real ColBERT checkpoint has been evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .native import NativeChronoVecIndex


def chamfer(query_tokens: np.ndarray, document_tokens: np.ndarray) -> float:
    """Exact late-interaction score. The thing FDE approximates."""
    if document_tokens.size == 0 or query_tokens.size == 0:
        return 0.0
    return float((query_tokens @ document_tokens.T).max(axis=1).sum())


class FixedDimensionalEncoder:
    """Turns a variable-length token set into one fixed-length vector."""

    def __init__(
        self,
        dimensions: int,
        *,
        k_sim: int = 4,
        repetitions: int = 8,
        projection_dim: int | None = None,
        seed: int = 0,
        fill_empty: bool = True,
    ) -> None:
        if k_sim < 1 or k_sim > 16:
            raise ValueError("k_sim must be between 1 and 16")
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        self.dimensions = dimensions
        self.k_sim = k_sim
        self.repetitions = repetitions
        self.buckets = 1 << k_sim
        self.fill_empty = fill_empty
        rng = np.random.default_rng(seed)
        # One SimHash matrix per repetition.
        self._hyperplanes = rng.standard_normal((repetitions, k_sim, dimensions)).astype(np.float32)
        self._inner = dimensions
        self._projection = None
        if projection_dim is not None and projection_dim < dimensions:
            self._projection = rng.standard_normal(
                (repetitions, dimensions, projection_dim)
            ).astype(np.float32) / np.sqrt(projection_dim)
            self._inner = projection_dim
        self._bit_weights = (1 << np.arange(k_sim)).astype(np.int64)

    @property
    def fde_dimensions(self) -> int:
        return self.repetitions * self.buckets * self._inner

    def _bucket_of(self, tokens: np.ndarray, repetition: int) -> np.ndarray:
        signs = (tokens @ self._hyperplanes[repetition].T) >= 0
        return (signs * self._bit_weights).sum(axis=1)

    def _project(self, block: np.ndarray, repetition: int) -> np.ndarray:
        if self._projection is None:
            return block
        return block @ self._projection[repetition]

    def encode_query(self, tokens: np.ndarray) -> np.ndarray:
        """Query side: buckets accumulate, so every token contributes."""
        tokens = np.ascontiguousarray(tokens, dtype=np.float32)
        out = np.zeros((self.repetitions, self.buckets, self._inner), dtype=np.float32)
        for r in range(self.repetitions):
            where = self._bucket_of(tokens, r)
            block = np.zeros((self.buckets, self.dimensions), dtype=np.float32)
            np.add.at(block, where, tokens)
            out[r] = self._project(block, r)
        return out.reshape(-1)

    def encode_document(self, tokens: np.ndarray) -> np.ndarray:
        """Document side: buckets average, so a long document is not favoured
        merely for being long."""
        tokens = np.ascontiguousarray(tokens, dtype=np.float32)
        out = np.zeros((self.repetitions, self.buckets, self._inner), dtype=np.float32)
        for r in range(self.repetitions):
            where = self._bucket_of(tokens, r)
            block = np.zeros((self.buckets, self.dimensions), dtype=np.float32)
            counts = np.zeros(self.buckets, dtype=np.int64)
            np.add.at(block, where, tokens)
            np.add.at(counts, where, 1)
            occupied = counts > 0
            block[occupied] /= counts[occupied, None]
            if self.fill_empty and not occupied.all() and occupied.any():
                block = self._fill(block, occupied)
            out[r] = self._project(block, r)
        return out.reshape(-1)

    def _fill(self, block: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        """Give an empty bucket the value of the nearest occupied one.

        Without this a query token can land in a bucket the document never
        populated and contribute nothing, understating the true Chamfer score.
        Nearest is by Hamming distance between bucket codes, which is the
        natural metric here: adjacent codes differ by one hyperplane.
        """
        codes = np.arange(self.buckets, dtype=np.int64)
        empty = np.nonzero(~occupied)[0]
        filled = np.nonzero(occupied)[0]
        distances = (
            np.bitwise_count(codes[empty, None] ^ codes[None, filled])
            if hasattr(np, "bitwise_count")
            else np.array([[bin(int(a) ^ int(b)).count("1") for b in filled] for a in codes[empty]])
        )
        block[empty] = block[filled[np.argmin(distances, axis=1)]]
        return block


@dataclass
class MultiVectorHit:
    doc_id: int
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class MultiVectorIndex:
    """Late-interaction retrieval over ChronoVec. BETA.

    Two candidate-generation strategies, both reranked with exact Chamfer:

    ``"tokens"`` (default) indexes every token as its own vector and gathers the
    documents owning the nearest tokens to each query token. This is how ColBERT
    and PLAID retrieve, and it is what this implementation measures well on.

    ``"fde"`` uses the MUVERA encoding above: one vector per document, so the
    index holds far fewer vectors. Measured on the synthetic corpus in
    ``experiments/multivector_eval.py`` its inner product correlates with exact
    Chamfer at Pearson r of roughly 0.55, which is too weak to rank reliably at
    a small shortlist. It is offered because that corpus may be adversarial for
    it -- real contextual token embeddings have very different geometry -- but
    it is not the default, and anyone choosing it should measure on their own
    data first.
    """

    def __init__(
        self,
        dimensions: int,
        *,
        strategy: str = "tokens",
        encoder: FixedDimensionalEncoder | None = None,
        nprobe: int = 32,
        page_capacity: int = 256,
        **encoder_kwargs: Any,
    ) -> None:
        if strategy not in ("tokens", "fde"):
            raise ValueError("strategy must be 'tokens' or 'fde'")
        self.strategy = strategy
        self.dimensions = dimensions
        self.encoder = None
        if strategy == "fde":
            self.encoder = encoder or FixedDimensionalEncoder(dimensions, **encoder_kwargs)
            indexed_dimensions = self.encoder.fde_dimensions
        else:
            indexed_dimensions = dimensions
        self._index = NativeChronoVecIndex(
            indexed_dimensions,
            metric="cosine",
            page_capacity=page_capacity,
            nprobe=nprobe,
        )
        self._tokens: dict[int, np.ndarray] = {}
        self._payloads: dict[int, dict[str, Any]] = {}
        self._owner: dict[int, int] = {}  # token slot -> document
        self._slots: dict[int, list[int]] = {}
        self._next_slot = 0
        self._nprobe = nprobe

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def add(self, doc_id: int, token_vectors, **payload: Any) -> int:
        """Index a document given its token vectors."""
        tokens = np.ascontiguousarray(token_vectors, dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[1] != self.dimensions:
            raise ValueError(
                f"expected token vectors of shape (n, {self.dimensions}), got {tokens.shape}"
            )
        if self.strategy == "fde":
            encoder = self.encoder
            if encoder is None:
                raise RuntimeError("fde strategy requires an encoder")
            committed = self._index.insert(doc_id, self._unit(encoder.encode_document(tokens)))
        else:
            slots = []
            committed = self._index.clock
            for token in tokens:
                slot = self._next_slot
                self._next_slot += 1
                committed = self._index.insert(slot, token)
                self._owner[slot] = doc_id
                slots.append(slot)
            self._slots[doc_id] = slots
        self._tokens[doc_id] = tokens
        self._payloads[doc_id] = dict(payload)
        return committed

    def delete(self, doc_id: int) -> int:
        if self.strategy == "fde":
            committed = self._index.delete(doc_id)
        else:
            committed = self._index.clock
            for slot in self._slots.pop(doc_id, []):
                committed = self._index.delete(slot)
                self._owner.pop(slot, None)
        self._tokens.pop(doc_id, None)
        self._payloads.pop(doc_id, None)
        return committed

    def search(
        self,
        query_tokens,
        k: int = 10,
        *,
        candidates: int | None = None,
        nprobe: int | None = None,
        snapshot: int | None = None,
    ) -> list[MultiVectorHit]:
        """Retrieve by FDE, then rerank the shortlist with exact Chamfer.

        `candidates` controls the shortlist; it defaults to 10x k because the
        FDE is an approximation and the exact rescoring is cheap relative to it.
        """
        tokens = np.ascontiguousarray(query_tokens, dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[1] != self.dimensions:
            raise ValueError(
                f"expected token vectors of shape (n, {self.dimensions}), got {tokens.shape}"
            )
        shortlist = candidates or max(10 * k, 50)
        probes = nprobe or self._nprobe
        if self.strategy == "fde":
            encoder = self.encoder
            if encoder is None:
                raise RuntimeError("fde strategy requires an encoder")
            fde = self._unit(encoder.encode_query(tokens))
            found = self._index.search(fde, k=shortlist, nprobe=probes, snapshot=snapshot)
            documents = [hit.id for hit in found if hit.id in self._tokens]
        else:
            # Each query token votes for the documents owning its nearest tokens;
            # a document only has to be found by one token to reach reranking.
            per_token = max(1, shortlist // max(1, len(tokens)))
            seen: dict[int, None] = {}
            for token in tokens:
                for hit in self._index.search(token, k=per_token, nprobe=probes, snapshot=snapshot):
                    owner = self._owner.get(hit.id)
                    if owner is not None and owner in self._tokens:
                        seen.setdefault(owner)
            documents = list(seen)
        scored = [
            MultiVectorHit(
                doc, chamfer(tokens, self._tokens[doc]), dict(self._payloads.get(doc, {}))
            )
            for doc in documents
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    @property
    def clock(self) -> int:
        return self._index.clock

    def close(self) -> None:
        self._index.close()

    def __enter__(self) -> MultiVectorIndex:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
