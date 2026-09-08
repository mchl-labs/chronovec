"""Adapters that let agent frameworks use ChronoVec as their vector store.

The adapters are deliberately thin. Everything they need that is not
framework-specific lives in :mod:`chronovec.integrations.core`, which is tested
directly, so the framework classes stay small enough to read in one screen.

The reason to ship these is not convenience. An engine that is merely faster can
be swapped for a faster one. An engine whose semantics an application depends on
cannot: ChronoVec exposes memory branching that other stores have no primitive
for, and once a workflow is built on it, replacing the engine means rewriting
the workflow.
"""

from .core import ChronoVecStore, StoredDocument
from .langgraph import LangGraphMemory

__all__ = [
    "ChronoVecStore",
    "LangGraphMemory",
    "StoredDocument",
]
