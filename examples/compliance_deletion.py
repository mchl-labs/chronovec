"""GDPR erasure / compliance deletion: delete + vacuum + verify.

Run: python examples/compliance_deletion.py

Demonstrates that:
- Deleted records are gone from all queries after vacuum, including time-travel queries.
- Reclamation is bounded: vacuum reclaims at most budget_versions records per call.
- Vacuum with oldest_snapshot=clock+1 physically destroys data across all snapshots.

Important: physical reclamation intentionally invalidates snapshots that would
otherwise expose the deleted data. Deletion takes priority over time travel.
No API keys or model required.
"""

from __future__ import annotations

import numpy as np

from chronovec import Index

DIM = 64


def main() -> None:
    rng = np.random.default_rng(0)
    index = Index(DIM, metric="cosine", page_capacity=64, nprobe=8)

    # Insert a background corpus (not the user we will erase)
    N_BACKGROUND = 500
    bg_ids = np.arange(0, N_BACKGROUND, dtype=np.int64)
    bg_vecs = rng.standard_normal((N_BACKGROUND, DIM)).astype(np.float32)
    index.insert_many(bg_ids, bg_vecs)

    # Insert the target user's records
    USER_OFFSET = 10_000
    N_USER = 50
    user_ids = np.arange(USER_OFFSET, USER_OFFSET + N_USER, dtype=np.int64)
    user_vecs = rng.standard_normal((N_USER, DIM)).astype(np.float32)
    index.insert_many(user_ids, user_vecs)
    user_id_set = set(user_ids.tolist())

    before_deletion = index.clock  # snapshot: user records are visible here

    print(f"Inserted {N_USER} user records (ids {USER_OFFSET}-{USER_OFFSET + N_USER - 1})")

    # Verify user records appear before deletion
    query = rng.standard_normal(DIM).astype(np.float32)
    before_results = index.search(query, k=N_USER + 10)
    user_before = [r for r in before_results if r.id in user_id_set]
    print(f"User records visible before deletion: {len(user_before)}")
    assert len(user_before) > 0, "Expected user records to be visible"

    # --- Erasure request received ---
    print("\nProcessing erasure request...")

    # Step 1: Logical deletion, O(1) per record
    index.delete_many(user_ids)
    print(f"  Logical deletion of {N_USER} records: done")

    # Step 2: Physical reclamation, bounded cost
    reclaimed = index.vacuum(
        oldest_snapshot=index.clock + 1,  # reclaim anything deleted before now
        budget_versions=N_USER * 2,  # bounded: at most this many records
    )
    print(f"  Vacuum reclaimed {reclaimed} versions")

    after_deletion = index.clock

    # Step 3: Verify: user records are gone from all queries at current snapshot
    after_results = index.search(query, k=N_USER + 10, snapshot=after_deletion)
    user_after = [r for r in after_results if r.id in user_id_set]
    print(f"  User records visible after deletion: {len(user_after)}")
    assert len(user_after) == 0, f"Expected 0 user records, got {len(user_after)}"

    # Step 4: After vacuum, even time-travel queries cannot recover the data.
    # Vacuum with oldest_snapshot=clock+1 reclaims all deleted records, including
    # those that were visible at before_deletion. Physical erasure takes priority.
    historical = index.search(query, k=N_USER + 10, snapshot=before_deletion)
    user_historical = [r for r in historical if r.id in user_id_set]
    print(
        f"\nTime travel to before deletion (post-vacuum): {len(user_historical)} user records visible"
    )
    print("  (physically reclaimed: data is gone from all snapshots)")
    assert len(user_historical) == 0

    final = index.search(query, k=N_USER + 10, snapshot=after_deletion)
    user_final = [r for r in final if r.id in user_id_set]
    assert len(user_final) == 0

    print("\nCompliance statement:")
    print(f"  The user's {N_USER} vectors were logically deleted, then physically")
    print(f"  reclaimed by vacuum ({reclaimed} records, bounded: no full-index scan).")
    print("  No query at any snapshot can recover the data after vacuum.")
    print("  Done.")

    index.close()


if __name__ == "__main__":
    main()
