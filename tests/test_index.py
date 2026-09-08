import numpy as np

from chronovec import ChronoVecIndex


def test_insert_search_delete_and_snapshot():
    index = ChronoVecIndex(2, metric="l2", page_capacity=8, nprobe=2)
    t1 = index.insert(1, [0.0, 0.0])
    index.insert(2, [10.0, 10.0])
    t3 = index.delete(1)

    assert index.search([0.0, 0.0], 1, snapshot=t1)[0].id == 1
    assert all(result.id != 1 for result in index.search([0.0, 0.0], 2, snapshot=t3))
    assert index.vacuum(t3 + 1) == 1


def test_upsert_preserves_old_snapshot():
    index = ChronoVecIndex(2, metric="l2", page_capacity=8, nprobe=2)
    old_ts = index.insert(7, [0.0, 0.0])
    index.insert(7, [5.0, 5.0])
    assert index.search([0.0, 0.0], 1, snapshot=old_ts)[0].id == 7
    assert index.search([5.0, 5.0], 1)[0].id == 7


def test_splits_and_recalls_nearest_neighbors():
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(100, 8)).astype(np.float32)
    index = ChronoVecIndex(8, page_capacity=16, nprobe=4)
    for item_id, vector in enumerate(vectors):
        index.insert(item_id, vector)
    assert index.stats()["pages"] > 1
    for item_id in range(20):
        assert index.search(vectors[item_id], 1)[0].id == item_id


def test_persistence_round_trip(tmp_path):
    index = ChronoVecIndex(3, metric="cosine", page_capacity=8)
    index.insert(1, [1, 0, 0])
    index.insert(2, [0, 1, 0])
    path = tmp_path / "index.npz"
    index.save(path)
    restored = ChronoVecIndex.load(path)
    assert restored.search([1, 0, 0], 1)[0].id == 1
