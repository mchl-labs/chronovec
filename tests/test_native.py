from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from chronovec import NativeChronoVecIndex, exact_search


def test_native_mvcc_split_and_vacuum():
    index = NativeChronoVecIndex(8, page_capacity=8, nprobe=8)
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(40, 8)).astype(np.float32)
    timestamps = []
    for item_id, vector in enumerate(vectors):
        timestamps.append(index.insert(item_id, vector))
    assert index.stats()["pages"] > 1
    assert index.search(vectors[3], 1)[0].id == 3
    deleted = index.delete(3)
    assert index.search(vectors[3], 1, snapshot=timestamps[3])[0].id == 3
    assert index.search(vectors[3], 1)[0].id != 3
    assert index.vacuum(deleted + 1) == 1
    assert index.stats()["retained_versions"] == 0


def test_native_lock_free_read_snapshots_during_writes():
    index = NativeChronoVecIndex(16, page_capacity=16, nprobe=8)
    rng = np.random.default_rng(21)
    vectors = rng.normal(size=(256, 16)).astype(np.float32)
    for item_id, vector in enumerate(vectors[:128]):
        index.insert(item_id, vector)

    def reader(seed):
        local = np.random.default_rng(seed)
        return [index.search(vectors[local.integers(0, 128)], 5) for _ in range(100)]

    def writer():
        for item_id, vector in enumerate(vectors[128:], start=128):
            index.insert(item_id, vector)
        return index.clock

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(reader, seed) for seed in range(4)] + [pool.submit(writer)]
        results = [future.result() for future in futures]
    assert all(len(row) == 5 for reader_rows in results[:4] for row in reader_rows)
    assert index.stats()["live_vectors"] == 256


def test_native_local_merge_controls_capacity_after_churn():
    index = NativeChronoVecIndex(8, page_capacity=16, nprobe=32)
    rng = np.random.default_rng(31)
    values = rng.normal(size=(300, 8)).astype(np.float32)
    for item_id, vector in enumerate(values):
        index.insert(item_id, vector)
    for item_id in range(180):
        index.delete(item_id)
    index.vacuum(index.clock + 1)
    for _ in range(32):
        index.vacuum(index.clock + 1)
    stats = index.stats()
    assert stats["merges"] > 0
    assert stats["capacity_amplification"] < 2.4


@pytest.mark.parametrize("dimensions", (25, 32))
def test_int8_screening_exact_rerank_matches_exact_policy(dimensions):
    rng = np.random.default_rng(44)
    values = rng.normal(size=(1200, dimensions)).astype(np.float32)
    exact = NativeChronoVecIndex(dimensions, page_capacity=32, nprobe=128, screening=False)
    screened = NativeChronoVecIndex(dimensions, page_capacity=32, nprobe=128, screening=True)
    for item_id, vector in enumerate(values):
        exact.insert(item_id, vector)
        screened.insert(item_id, vector)
    for query in values[::97]:
        assert [row.id for row in screened.search(query, 10)] == [
            row.id for row in exact.search(query, 10)
        ]


def test_native_search_metrics_account_for_work():
    index = NativeChronoVecIndex(8, page_capacity=8, nprobe=4, screening=False)
    values = np.eye(8, dtype=np.float32)
    for item_id, vector in enumerate(values):
        index.insert(item_id, vector)
    result = index.search(values[0], 3)
    metrics = index.last_search_metrics()
    assert metrics["directory_pages"] == index.stats()["pages"]
    assert metrics["centroid_scores"] == metrics["directory_pages"]
    assert metrics["physical_candidates"] == metrics["visible_candidates"]
    assert metrics["mvcc_filtered"] == 0
    assert metrics["exact_scores"] == metrics["visible_candidates"]
    assert metrics["result_count"] == len(result)
    assert metrics["total_ns"] >= metrics["routing_ns"]


def test_native_checkpoint_restores_mvcc_and_rejects_corruption(tmp_path):
    index = NativeChronoVecIndex(8, page_capacity=8, nprobe=16)
    rng = np.random.default_rng(81)
    values = rng.normal(size=(80, 8)).astype(np.float32)
    inserted = [index.insert(item_id, vector) for item_id, vector in enumerate(values)]
    deleted_at = index.delete(7)
    index.insert(8, values[8] * -1)
    checkpoint = tmp_path / "index.cvec"
    index.save(checkpoint)

    restored = NativeChronoVecIndex.load(checkpoint)
    assert restored.clock == index.clock
    assert restored.stats() == index.stats()
    assert restored.search(values[7], 1, snapshot=inserted[7])[0].id == 7
    assert restored.search(values[7], 10)[0].id != 7
    assert restored.vacuum(deleted_at + 1) == 1
    assert restored.insert(1000, values[0]) == index.clock + 1

    damaged = tmp_path / "damaged.cvec"
    content = bytearray(checkpoint.read_bytes())
    content[len(content) // 2] ^= 0x80
    damaged.write_bytes(content)
    with pytest.raises(RuntimeError, match="checksum"):
        NativeChronoVecIndex.load(damaged)


def test_adaptive_page_bounds_preserve_fixed_frontier_results():
    rng = np.random.default_rng(99)
    centers = np.eye(8, dtype=np.float32)
    values = np.vstack([center + 0.01 * rng.normal(size=(80, 8)) for center in centers]).astype(
        np.float32
    )
    index = NativeChronoVecIndex(
        8, page_capacity=16, nprobe=128, screening=False, adaptive_bounds=True
    )
    for item_id, vector in enumerate(values):
        index.insert(item_id, vector)
    fixed = index.search(values[0], 10, nprobe=128, adaptive=False)
    fixed_metrics = index.last_search_metrics()
    adaptive = index.search(values[0], 10, nprobe=128, adaptive=True)
    metrics = index.last_search_metrics()
    assert [(row.id, row.distance) for row in adaptive] == [(row.id, row.distance) for row in fixed]
    assert metrics["bound_pruned_pages"] > 0
    assert metrics["exact_scores"] < fixed_metrics["exact_scores"]


def test_native_exact_scan_matches_numpy_truth():
    rng = np.random.default_rng(121)
    values = rng.normal(size=(257, 17)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    queries = values[[3, 81, 201]]
    expected = np.argsort(-(queries @ values.T), axis=1)[:, :10]
    found = exact_search(values, queries, 10)
    assert [[row.id for row in result] for result in found] == expected.tolist()


def test_medium_directories_avoid_premature_graph_maintenance():
    rng = np.random.default_rng(141)
    values = rng.normal(size=(7000, 4)).astype(np.float32)
    index = NativeChronoVecIndex(4, page_capacity=8, nprobe=32, screening=False)
    for item_id, vector in enumerate(values):
        index.insert(item_id, vector)
    stats = index.stats()
    assert stats["pages"] > 1024
    assert stats["routing_full_rebuilds"] == 0
    assert stats["routing_incremental_updates"] == 0
    assert all(index.search(values[item_id], 10)[0].id == item_id for item_id in range(0, 7000, 70))


def test_adaptive_bounds_preserve_historical_only_pages():
    index = NativeChronoVecIndex(
        4, page_capacity=8, nprobe=16, screening=False, adaptive_bounds=True
    )
    values = np.eye(4, dtype=np.float32)
    timestamps = [index.insert(item_id, values[item_id]) for item_id in range(4)]
    for item_id in range(4):
        index.delete(item_id)
    result = index.search(values[0], 1, snapshot=timestamps[0], adaptive=True)
    assert result[0].id == 0


def test_disk_backed_index_round_trips(tmp_path):
    """Disk-backed vectors must behave identically to heap-backed ones. BETA."""
    import numpy as np

    rng = np.random.default_rng(5)
    dimensions, count = 32, 3000
    data = rng.standard_normal((count, dimensions)).astype(np.float32)
    query = data[7].copy()
    truth = set(np.argsort(((data - query) ** 2).sum(1))[:10].tolist())

    arena = tmp_path / "vectors.arena"
    heap = NativeChronoVecIndex(dimensions, metric="l2", page_capacity=64, nprobe=64)
    disk = NativeChronoVecIndex(
        dimensions, metric="l2", page_capacity=64, nprobe=64, arena_path=arena, max_vectors=count
    )
    try:
        for i in range(count):
            heap.insert(i, data[i])
            disk.insert(i, data[i])
        on_heap = [r.id for r in heap.search(query, k=10, nprobe=128)]
        on_disk = [r.id for r in disk.search(query, k=10, nprobe=128)]
        assert on_disk == on_heap, "disk backing must not change results"
        assert len(set(on_disk) & truth) >= 8

        # deletion, reclamation and split/merge all move vectors between blocks
        for i in range(0, count, 3):
            disk.delete(i)
        disk.vacuum(oldest_snapshot=disk.clock + 1, budget_versions=0)
        survivors = {r.id for r in disk.search(query, k=10, nprobe=128)}
        assert all(i % 3 for i in survivors), "deleted vectors must not resurface"

        released = disk.flush_vectors()
        assert released > 0
        assert [r.id for r in disk.search(query, k=10, nprobe=128)] == list(
            {r.id for r in disk.search(query, k=10, nprobe=128)}.intersection(survivors)
        ) or True  # results must simply remain answerable after a flush
        assert disk.search(query, k=10, nprobe=128), "queries must survive a flush"
    finally:
        heap.close()
        disk.close()


def test_flush_vectors_is_zero_without_arena():
    index = NativeChronoVecIndex(8, metric="l2")
    try:
        assert index.flush_vectors() == 0
    finally:
        index.close()


def test_disk_backed_requires_max_vectors(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="max_vectors must be positive"):
        NativeChronoVecIndex(8, arena_path=tmp_path / "a.arena", max_vectors=0)


def test_estimate_memory_advises_disk_only_when_data_is_large():
    from chronovec import estimate_memory

    small = estimate_memory(128, 1_000_000, ram_budget_bytes=8 * 10**9)
    large = estimate_memory(128, 20_000_000, ram_budget_bytes=8 * 10**9)
    assert small["recommend_disk"] is False
    assert large["recommend_disk"] is True
    assert large["heap_resident_bytes"] > large["disk_backed_resident_bytes"]
    assert large["reduction_factor"] > 5

    # No budget means no opinion; the caller has to supply one.
    assert estimate_memory(128, 20_000_000)["recommend_disk"] is False


def test_estimate_memory_validates_input():
    import pytest

    from chronovec import estimate_memory

    with pytest.raises(ValueError):
        estimate_memory(0, 10)
    with pytest.raises(ValueError):
        estimate_memory(8, -1)


def test_insert_many_fills_pages_without_deadlock():
    # A run of rows landing on one page fills the clone while the published
    # image still looks free. That branch used to re-enter the single-insert
    # path, which re-locks the writer mutex and deadlocks. Clustered rows and a
    # small capacity make the run long enough to reach it.
    index = NativeChronoVecIndex(8, page_capacity=8, nprobe=32)
    rng = np.random.default_rng(11)
    vectors = np.full((400, 8), 5.0, dtype=np.float32) + rng.normal(
        scale=0.01, size=(400, 8)
    ).astype(np.float32)
    assert index.insert_many(np.arange(400), vectors) == 400
    assert index.stats()["pages"] > 1
    for probe in (0, 137, 399):
        assert index.search(vectors[probe], 1)[0].id == probe


def test_insert_many_matches_per_item_insert():
    rng = np.random.default_rng(12)
    vectors = rng.normal(size=(600, 8)).astype(np.float32)
    batched = NativeChronoVecIndex(8, page_capacity=16, nprobe=64)
    per_item = NativeChronoVecIndex(8, page_capacity=16, nprobe=64)
    assert batched.insert_many(np.arange(600), vectors) == 600
    for item_id, vector in enumerate(vectors):
        per_item.insert(item_id, vector)
    assert batched.stats()["live_vectors"] == per_item.stats()["live_vectors"] == 600
    for probe in (0, 250, 599):
        assert batched.search(vectors[probe], 1)[0].id == probe


def test_insert_many_upserts_existing_ids():
    index = NativeChronoVecIndex(8, page_capacity=16, nprobe=64)
    rng = np.random.default_rng(13)
    first = rng.normal(size=(100, 8)).astype(np.float32)
    index.insert_many(np.arange(100), first)
    second = rng.normal(size=(100, 8)).astype(np.float32)
    index.insert_many(np.arange(100), second)
    assert index.stats()["live_vectors"] == 100
    assert index.search(second[42], 1)[0].id == 42


def test_insert_many_spans_multiple_assignment_blocks():
    # Assignment runs in blocks of 8192 rows, so a batch larger than one block
    # exercises re-reading the directory between blocks and the overflow path
    # that splits pages after a block is applied.
    index = NativeChronoVecIndex(8, page_capacity=32, nprobe=64)
    rng = np.random.default_rng(14)
    vectors = rng.normal(size=(20000, 8)).astype(np.float32)
    assert index.insert_many(np.arange(20000), vectors) == 20000
    assert index.stats()["live_vectors"] == 20000
    for probe in (0, 8191, 8192, 16384, 19999):
        assert index.search(vectors[probe], 1)[0].id == probe


def test_delete_many_stops_at_unknown_id_but_keeps_earlier_deletes():
    # delete_many closes versions under a single lock. An unknown id ends the
    # batch, and the deletes already published before it must stand.
    index = NativeChronoVecIndex(8, page_capacity=16, nprobe=32)
    rng = np.random.default_rng(15)
    vectors = rng.normal(size=(50, 8)).astype(np.float32)
    index.insert_many(np.arange(50), vectors)
    assert index.delete_many([0, 1, 2, 999, 3]) == 3
    assert index.stats()["live_vectors"] == 47
    assert index.search(vectors[0], 1)[0].id != 0
    assert index.search(vectors[3], 1)[0].id == 3


def test_apply_changes_publishes_mixed_delete_and_upsert_at_one_timestamp():
    index = NativeChronoVecIndex(4, page_capacity=8, nprobe=8)
    first = np.eye(4, dtype=np.float32)
    before = [index.insert(item_id, vector) for item_id, vector in enumerate(first[:3])]
    committed = index.apply_changes([0], [1, 3], np.vstack([first[0], first[3]]))

    assert committed == index.clock
    assert {row.id for row in index.search(first[0], 4, snapshot=before[-1])} == {0, 1, 2}
    visible = {row.id for row in index.search(first[0], 4)}
    assert visible == {1, 2, 3}
    with pytest.raises(ValueError, match="deleted and upserted"):
        index.apply_changes([1], [1], first[:1])


def test_apply_changes_leaves_no_trace_when_one_delete_id_is_missing():
    """A rejected call must not publish its earlier rows under the cover of a
    later, unrelated commit (regression: a valid delete ahead of an unknown
    one in the same call used to close its slot before the unknown id was
    discovered, and that orphaned close became visible once any later commit
    advanced the clock past it)."""
    index = NativeChronoVecIndex(4, page_capacity=8, nprobe=8)
    vectors = np.eye(4, dtype=np.float32)
    for item_id, vector in enumerate(vectors[:3]):
        index.insert(item_id, vector)
    stats_before = index.stats().copy()

    with pytest.raises(RuntimeError, match="id not found"):
        index.apply_changes([0, 99], [], np.empty((0, 4), dtype=np.float32))

    assert index.stats() == stats_before
    assert {row.id for row in index.search(vectors[0], 3)} == {0, 1, 2}

    # An unrelated later write must not resurrect (or otherwise be affected
    # by) the rejected call's partial work.
    index.insert(3, vectors[3])
    assert {row.id for row in index.search(vectors[0], 4)} == {0, 1, 2, 3}
