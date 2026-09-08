"""The memory-mapped arena.

Vectors are 80% of an index's payload and the screening codes the other 20%,
so mapping only the vectors left a fifth of a large index pinned in memory.
Both are mapped now, and these check that the mapping is real rather than
silently falling back to the heap.
"""

import numpy as np
import pytest

from chronovec import Index


def clustered(n, dim, seed=3):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(32, dim)).astype(np.float32) * 3.0
    return (centres[rng.integers(0, 32, n)] + rng.normal(scale=1.0, size=(n, dim))).astype(
        np.float32
    )


def test_both_the_vectors_and_the_codes_land_in_mapped_files(tmp_path):
    path = tmp_path / "arena.bin"
    vectors = clustered(4000, 64)
    index = Index(64, page_capacity=64, nprobe=32, metric="l2", arena_path=path, max_vectors=40000)
    index.insert_many(np.arange(4000), vectors)
    assert path.exists() and path.stat().st_size > 0
    codes = tmp_path / "arena.bin.codes"
    assert codes.exists() and codes.stat().st_size > 0
    assert index.search(vectors[7], 1)[0].id == 7
    index.close()
    # Both are spill files, not persistence: they go when the index does.
    assert not path.exists() and not codes.exists()


def test_a_mapped_index_answers_the_same_as_a_heap_one(tmp_path):
    vectors = clustered(4000, 64)
    heap = Index(64, page_capacity=64, nprobe=32, metric="l2")
    heap.insert_many(np.arange(4000), vectors)
    mapped = Index(
        64,
        page_capacity=64,
        nprobe=32,
        metric="l2",
        arena_path=tmp_path / "a.bin",
        max_vectors=40000,
    )
    mapped.insert_many(np.arange(4000), vectors)
    assert heap.stats()["pages"] == mapped.stats()["pages"]
    for probe in (0, 1500, 3999):
        assert [r.id for r in heap.search(vectors[probe], 5)] == [
            r.id for r in mapped.search(vectors[probe], 5)
        ]
    heap.close()
    mapped.close()


def test_churn_on_a_mapped_index_recycles_blocks(tmp_path):
    # Blocks are returned when the last clone holding them drops. If they were
    # not, sustained churn would exhaust the arena.
    vectors = clustered(6000, 32)
    index = Index(
        32,
        page_capacity=64,
        nprobe=32,
        metric="l2",
        arena_path=tmp_path / "a.bin",
        max_vectors=24000,
    )
    index.insert_many(np.arange(3000), vectors[:3000])
    for cycle in range(6):
        index.delete_many(np.arange(cycle * 500, cycle * 500 + 500))
        index.insert_many(np.arange(3000 + cycle * 500, 3500 + cycle * 500), vectors[3000:3500])
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
    assert index.stats()["live_vectors"] == 3000
    index.close()


def test_flushing_reports_both_arenas(tmp_path):
    vectors = clustered(2000, 64)
    index = Index(
        64,
        page_capacity=64,
        nprobe=32,
        metric="l2",
        arena_path=tmp_path / "a.bin",
        max_vectors=20000,
    )
    index.insert_many(np.arange(2000), vectors)
    vector_bytes = (tmp_path / "a.bin").stat().st_size
    evictable = index.flush_vectors()
    # More than the vector arena alone, because the codes arena counts too.
    assert evictable > vector_bytes
    assert index.search(vectors[3], 1)[0].id == 3  # still readable after
    index.close()


def test_an_arena_too_small_says_so_rather_than_corrupting(tmp_path):
    index = Index(
        32, page_capacity=64, nprobe=32, metric="l2", arena_path=tmp_path / "a.bin", max_vectors=200
    )
    with pytest.raises(RuntimeError, match="arena is full"):
        index.insert_many(np.arange(5000), clustered(5000, 32))
    index.close()
