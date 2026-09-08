"""A five-minute mutable RAG demo with no model or service dependencies.

Run from the repository root:

    python examples/mutable_rag.py

Replace ``hash_embed`` with your real embedding model in production. The
important part of the example is the storage behavior: update a document,
keep serving the latest state, and reproduce the old retrieval result from a
snapshot captured before the update.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from chronovec import Collection

DIMENSIONS = 64


def hash_embed(texts: list[str]) -> list[list[float]]:
    """Small deterministic local embedder so the demo runs offline."""
    rows: list[list[float]] = []
    for text in texts:
        vector = np.zeros(DIMENSIONS, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            position = int.from_bytes(digest[:4], "big") % DIMENSIONS
            vector[position] += 1.0
        rows.append(vector.tolist())
    return rows


def show(label: str, results) -> None:
    print(label)
    for result in results:
        print(f"  {result.id}: {result.document}")


with Collection(dimensions=DIMENSIONS, metric="cosine", embedding_function=hash_embed) as memory:
    memory.add(
        ids=["deploy", "privacy", "retries"],
        documents=[
            "Deploy the service with PostgreSQL in production.",
            "User data is retained for thirty days.",
            "Retry failed jobs three times with exponential backoff.",
        ],
        metadatas=[
            {"team": "platform", "status": "current"},
            {"team": "privacy", "status": "current"},
            {"team": "platform", "status": "current"},
        ],
    )

    before = memory.snapshot()
    show("Before the correction:", memory.query(query_text="production database", k=2))

    # A document correction is an MVCC update, not an in-place overwrite.
    memory.add(
        ids=["deploy"],
        documents=["Deploy the service with SQLite for the local edge runtime."],
        metadatas=[{"team": "platform", "status": "corrected"}],
    )

    show("Latest retrieval:", memory.query(query_text="local edge database", k=2))
    show(
        "Replayed from the pre-correction snapshot:",
        memory.query(query_text="production database", k=2, snapshot=before),
    )

    print(f"\nCurrent records: {memory.count()} | Snapshot token: {before}")
