"""Concurrent read under sustained write load.

Tests that multiple readers can run concurrently with an active writer
without errors or data races. Complements the lighter concurrency tests
in test_native.py and test_acid.py.
"""

import threading

import numpy as np

from chronovec import NativeChronoVecIndex


def clustered(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(8, dim)).astype(np.float32)
    return (centres[rng.integers(0, 8, n)] + rng.normal(scale=0.5, size=(n, dim))).astype(
        np.float32
    )


def test_eight_readers_one_writer_no_errors():
    """8 reader threads × 1000 queries each, concurrent with a writer of 500 inserts."""
    dim = 32
    n_seed = 200
    n_writes = 500
    n_reads_per_thread = 1000
    n_reader_threads = 8

    vectors_seed = clustered(n_seed, dim, seed=1)
    vectors_new = clustered(n_writes, dim, seed=2)
    queries = clustered(n_reads_per_thread, dim, seed=3)

    index = NativeChronoVecIndex(dim, page_capacity=32, nprobe=8)
    ids_seed = np.arange(n_seed, dtype=np.int64)
    index.insert_many(ids_seed, vectors_seed)

    errors = []
    read_counts = []

    def reader(thread_id: int) -> None:
        count = 0
        try:
            for i in range(n_reads_per_thread):
                q = queries[i % len(queries)]
                results = index.search(q, k=5)
                assert results is not None, "search returned None"
                count += 1
        except Exception as exc:
            errors.append(exc)
        read_counts.append(count)

    def writer() -> None:
        ids_new = np.arange(n_seed, n_seed + n_writes, dtype=np.int64)
        try:
            for i in range(n_writes):
                index.insert(int(ids_new[i]), vectors_new[i])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(n_reader_threads)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent access: {errors}"
    assert all(c == n_reads_per_thread for c in read_counts), (
        f"Some readers did not complete all queries: {read_counts}"
    )
    # All seed vectors + all new vectors should be live
    assert index.stats()["live_vectors"] == n_seed + n_writes
    index.close()


def test_concurrent_readers_see_consistent_snapshots():
    """Readers holding a snapshot see the same result throughout their read."""
    dim = 16
    n = 100

    index = NativeChronoVecIndex(dim, page_capacity=32, nprobe=8)
    rng = np.random.default_rng(5)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    index.insert_many(np.arange(n, dtype=np.int64), vectors)

    snapshot = index.clock
    query = vectors[0]

    # Expected result at snapshot
    expected = [r.id for r in index.search(query, k=5, snapshot=snapshot)]

    inconsistencies = []

    def reader_with_snapshot():
        for _ in range(200):
            result = [r.id for r in index.search(query, k=5, snapshot=snapshot)]
            if result != expected:
                inconsistencies.append(result)

    def writer():
        for i in range(100):
            index.insert(n + i, rng.standard_normal(dim).astype(np.float32))
            index.delete(i)

    threads = [threading.Thread(target=reader_with_snapshot) for _ in range(4)]
    threads.append(threading.Thread(target=writer))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not inconsistencies, (
        f"Snapshot reads were inconsistent: {len(inconsistencies)} mismatches"
    )
    index.close()
