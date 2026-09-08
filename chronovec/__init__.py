"""ChronoVec: a snapshot-isolated vector index with bounded reclamation."""

from .async_collection import AsyncCollection
from .client import Client, PersistentClient
from .collection import Collection, Record, stable_id
from .embeddings import (
    CustomEmbedding,
    EmbeddingFunction,
    EmbeddingInput,
    ProviderEmbedding,
    SentenceTransformerEmbedding,
    as_embedding_function,
)
from .index import ChronoVecIndex, SearchResult, _ReferenceIndex
from .memory import AgentMemory, BranchError

try:
    from .native import NativeChronoVecIndex, estimate_memory, exact_search, library_version

    Index = NativeChronoVecIndex
except RuntimeError as _e:
    import warnings

    warnings.warn(
        f"Native ChronoVec library not available ({_e}). "
        "Falling back to the pure-Python reference index; performance will be lower. "
        "Run `pip install .` from the source root to build the native library.",
        stacklevel=2,
    )
    NativeChronoVecIndex = None  # type: ignore[assignment,misc]
    Index = _ReferenceIndex  # type: ignore[assignment,misc]

    def estimate_memory(*args, **kwargs) -> dict:  # type: ignore[misc]
        raise RuntimeError("estimate_memory requires the native library")

    def exact_search(*args, **kwargs) -> list:  # type: ignore[misc]
        raise RuntimeError("exact_search requires the native library")

    def library_version(*args, **kwargs) -> tuple:  # type: ignore[misc]
        raise RuntimeError("library_version requires the native library")


__all__ = [
    "Index",
    "AsyncCollection",
    "Client",
    "PersistentClient",
    "Collection",
    "Record",
    "stable_id",
    "EmbeddingFunction",
    "EmbeddingInput",
    "CustomEmbedding",
    "SentenceTransformerEmbedding",
    "ProviderEmbedding",
    "as_embedding_function",
    "AgentMemory",
    "BranchError",
    "_ReferenceIndex",
    "ChronoVecIndex",  # backward-compatible alias for _ReferenceIndex
    "NativeChronoVecIndex",
    "SearchResult",
    "estimate_memory",
    "exact_search",
    "library_version",
]
