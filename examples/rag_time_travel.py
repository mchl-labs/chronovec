"""RAG with time travel: query a knowledge base as it existed at a past moment.

Run: python examples/rag_time_travel.py

Use cases:
- Reproducible evaluation: run the same retrieval as of a specific date.
- Retrieval regression debugging: "what did the retriever return last Tuesday?"
- Staged rollouts: query at the previous stable state while ingesting updates.

No API keys or model required: a deterministic bag-of-words embedder is used.
"""

from __future__ import annotations

import zlib

import numpy as np

from chronovec import Collection

WIDTH = 256


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for text in texts:
        v = np.zeros(WIDTH, dtype=np.float32)
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % WIDTH] += 1.0
        norm = np.linalg.norm(v)
        out.append((v / norm if norm else v).tolist())
    return out


def show(title: str, records: list) -> None:
    print(f"  {title}")
    for r in records:
        print(f"    - {r.document or r.id}")
    if not records:
        print("    (nothing)")


def main() -> None:
    coll = Collection(dimensions=WIDTH, metric="cosine", embedding_function=embed)

    print("1. Build the initial corpus.\n")
    coll.add(
        ids=["doc-paris", "doc-louvre", "doc-eiffel"],
        documents=[
            "Paris is the capital of France",
            "The Louvre is a museum in Paris",
            "The Eiffel Tower is an iron lattice tower in Paris",
        ],
        metadatas=[{"topic": "geography"}, {"topic": "art"}, {"topic": "architecture"}],
    )

    before = coll.snapshot()  # pin the current state
    show("initial query for 'Paris capital':", coll.query(query_text="Paris capital", k=3))

    print("\n2. Update a document (correction or new version).\n")
    coll.add(
        ids=["doc-paris"],
        documents=["Paris is the capital of France and a global cultural hub"],
        metadatas=[{"topic": "geography", "updated": True}],
    )

    show("latest query for 'Paris capital':", coll.query(query_text="Paris capital", k=1))

    show(
        "same query as of the snapshot (before update):",
        coll.query(query_text="Paris capital", k=1, snapshot=before),
    )

    print("\n3. Staged rollout: evaluate before promoting the snapshot.\n")
    test_queries = ["Paris France capital", "cultural hub", "iron tower"]
    changed_results = 0
    for q in test_queries:
        latest = coll.query(query_text=q, k=1)
        historical = coll.query(query_text=q, k=1, snapshot=before)
        if latest and historical and latest[0].document != historical[0].document:
            changed_results += 1
        print(f"  query: {q!r}")
        print(f"    latest:     {latest[0].document if latest else '(none)'}")
        print(f"    historical: {historical[0].document if historical else '(none)'}")
    # A real evaluation would use labeled queries. This toy decision shows the
    # operational pattern: promote a tested snapshot explicitly.
    if changed_results:
        stable = coll.snapshot()
        print(f"  decision: promote tested snapshot {stable} ({changed_results} changed result(s))")
    else:
        stable = before
        print(f"  decision: keep stable snapshot {stable}")

    print("\n4. Reclaim versions older than our pinned snapshot.\n")
    freed = coll.vacuum(keep_snapshot=stable)
    print(f"  freed {freed} versions (snapshot is still pinned at {stable})")

    freed2 = coll.vacuum()
    print(f"  freed {freed2} more versions after releasing the pin")

    coll.close()


if __name__ == "__main__":
    main()
