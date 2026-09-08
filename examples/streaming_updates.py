"""High-churn streaming: continuous insert/delete/vacuum loop.

Run: python examples/streaming_updates.py

Demonstrates that capacity amplification stays flat under sustained churn:
pages are compacted after vacuum, expired versions are reclaimed, and memory
does not grow without bound over multiple full dataset turnovers.

No API keys or model required.
"""

from __future__ import annotations

import numpy as np

from chronovec import Index

DIM = 64
N_LIVE = 2_000
N_EPOCHS = 5
REPLACEMENTS_PER_EPOCH = 1_000


def main() -> None:
    index = Index(
        DIM,
        metric="l2",
        page_capacity=128,
        nprobe=8,
    )

    # Seed the index with an initial live set
    print(f"Seeding {N_LIVE} vectors...")
    rng = np.random.default_rng(42)
    ids = np.arange(N_LIVE, dtype=np.int64)
    vectors = rng.standard_normal((N_LIVE, DIM)).astype(np.float32)
    index.insert_many(ids, vectors)

    live = set(ids.tolist())
    vectors_by_id = {int(vector_id): vector for vector_id, vector in zip(ids, vectors)}
    next_id = int(N_LIVE)

    print(f"\nStreaming {N_EPOCHS} epochs × {REPLACEMENTS_PER_EPOCH} replacements\n")
    print(f"{'epoch':>5}  {'ops/s':>8}  {'cap amp':>8}  {'reclaimed':>10}  {'query recall':>12}")
    print("-" * 60)

    import time

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.perf_counter()
        for _ in range(REPLACEMENTS_PER_EPOCH):
            # Evict a random live record
            victim = int(rng.choice(list(live)))
            index.delete(victim)
            live.discard(victim)
            vectors_by_id.pop(victim)

            # Insert a new one
            new_vec = rng.standard_normal(DIM).astype(np.float32)
            index.insert(next_id, new_vec)
            live.add(next_id)
            vectors_by_id[next_id] = new_vec
            next_id += 1

        elapsed = time.perf_counter() - t0
        ops_per_s = REPLACEMENTS_PER_EPOCH / elapsed

        # Vacuum on a budget
        reclaimed = index.vacuum(
            oldest_snapshot=index.clock + 1,
            budget_versions=REPLACEMENTS_PER_EPOCH * 2,
        )

        stats = index.stats()
        cap_amp = stats.get("capacity_amplification", float("nan"))

        # Spot-check self-recall using known vectors. Random-query hit rates
        # are not recall.
        sample_ids = rng.choice(list(live), size=min(20, len(live)), replace=False)
        recall_hits = 0
        for sid in sample_ids:
            # The vector for `sid` may have changed; just verify it appears in search
            query = vectors_by_id[int(sid)]
            results = {r.id for r in index.search(query, k=10)}
            if int(sid) in results:
                recall_hits += 1
        recall = recall_hits / len(sample_ids) if sample_ids.size else 0.0

        print(f"{epoch:>5}  {ops_per_s:>8.0f}  {cap_amp:>8.2f}x  {reclaimed:>10}  {recall:>12.3f}")

    print(f"\nFinal live set: {len(live)} vectors")
    print(f"Final stats: {index.stats()}")

    index.close()


if __name__ == "__main__":
    main()
