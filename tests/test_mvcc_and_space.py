"""MVCC visibility, disk backing, and the space bound.

Three claims that are easy to demonstrate once and hard to keep true: that a
version is visible exactly on [begin, end), that a mapped index behaves like a
resident one, and that space is a function of live data rather than of history.
"""

import numpy as np
import pytest
from invariants import assert_no_duplicate_ids, check

from chronovec import Index


def data(n=400, dim=8, seed=6):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def index(**kwargs):
    kwargs.setdefault("metric", "l2")
    kwargs.setdefault("page_capacity", 32)
    kwargs.setdefault("nprobe", 64)
    return Index(kwargs.pop("dim", 8), **kwargs)


# ------------------------------------------------------------------- MVCC
def test_a_version_is_visible_exactly_on_its_half_open_interval():
    v = data(10)
    with index() as ix:
        ix.insert(2, v[1])  # a survivor
        born = ix.insert(1, v[0])
        assert ix.search(v[0], 1, snapshot=born)[0].id == 1  # at begin
        died = ix.delete(1)
        # Visible up to but not including the closing timestamp.
        assert ix.search(v[0], 2, snapshot=died - 1)[0].id == 1
        assert 1 not in {r.id for r in ix.search(v[0], 2, snapshot=died)}


def test_an_overwrite_keeps_both_versions_readable_at_their_own_times():
    v = data(10)
    with index() as ix:
        first = ix.insert(1, v[0])
        second = ix.insert(1, v[5])  # same id, new vector
        assert ix.search(v[0], 1, snapshot=first)[0].id == 1
        assert ix.search(v[5], 1, snapshot=second)[0].id == 1
        # And only one of them is live now.
        assert ix.stats()["live_vectors"] == 1
        assert_no_duplicate_ids(ix, v[5], 10)


def test_a_snapshot_older_than_the_horizon_is_what_blocks_reclamation():
    v = data(300)
    with index() as ix:
        ix.insert_many(np.arange(200), v[:200])
        pinned = ix.clock
        ix.delete_many(np.arange(100))
        assert ix.vacuum(oldest_snapshot=pinned, budget_versions=0) == 0
        assert ix.stats()["retained_versions"] > 0
        freed = ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=0)
        assert freed == 100
        assert ix.stats()["retained_versions"] == 0


def test_reclamation_respects_its_budget():
    v = data(400)
    with index() as ix:
        ix.insert_many(np.arange(300), v[:300])
        ix.delete_many(np.arange(300))
        total = 0
        for _ in range(3):
            freed = ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=32)
            assert freed <= 32, "budget exceeded"
            total += freed
        assert 0 < total <= 96


def test_the_horizon_moves_only_forward_for_a_reader():
    v = data(200)
    with index() as ix:
        ix.insert_many(np.arange(100), v[:100])
        early = ix.clock
        ix.insert_many(np.arange(100, 150), v[100:150])
        late = ix.clock
        assert late > early
        # An older snapshot never gains rows written after it.
        assert not ({r.id for r in ix.search(v[0], 200, snapshot=early)} & set(range(100, 150)))


# ------------------------------------------------------------------- disk
def test_a_mapped_index_answers_identically_to_a_resident_one(tmp_path):
    v = data(1500, 16)
    with (
        index(dim=16, page_capacity=64) as resident,
        index(dim=16, page_capacity=64, arena_path=tmp_path / "a.bin", max_vectors=40000) as mapped,
    ):
        resident.insert_many(np.arange(1500), v)
        mapped.insert_many(np.arange(1500), v)
        assert resident.stats()["pages"] == mapped.stats()["pages"]
        for probe in (0, 700, 1499):
            assert [r.id for r in resident.search(v[probe], 5)] == [
                r.id for r in mapped.search(v[probe], 5)
            ]


def test_both_payloads_reach_the_mapping(tmp_path):
    arena = tmp_path / "b.bin"
    with index(dim=16, page_capacity=64, arena_path=arena, max_vectors=40000) as ix:
        ix.insert_many(np.arange(1200), data(1200, 16))
        assert arena.stat().st_size > 0
        assert (tmp_path / "b.bin.codes").stat().st_size > 0


def test_an_exhausted_arena_fails_loudly_and_leaves_a_usable_index(tmp_path):
    # The failure that matters: running out of mapped space must not corrupt
    # what is already there.
    with index(dim=8, page_capacity=32, arena_path=tmp_path / "c.bin", max_vectors=300) as ix:
        ix.insert_many(np.arange(200), data(200))
        check(ix, 200, where="before exhaustion")
        with pytest.raises(RuntimeError, match="arena is full"):
            ix.insert_many(np.arange(200, 20000), data(19800))
        # Still answers, still consistent.
        assert ix.search(data(200)[5], 1)[0].id == 5
        assert ix.stats()["live_vectors"] >= 200


def test_a_mapped_index_survives_churn_without_leaking_blocks(tmp_path):
    v = data(3000, 16)
    with index(dim=16, page_capacity=64, arena_path=tmp_path / "d.bin", max_vectors=24000) as ix:
        ix.insert_many(np.arange(1500), v[:1500])
        for cycle in range(8):
            ix.delete_many(np.arange(cycle * 100, cycle * 100 + 100))
            ix.insert_many(
                np.arange(1500 + cycle * 100, 1600 + cycle * 100),
                v[1500 + cycle * 100 : 1600 + cycle * 100],
            )
            ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=0)
        check(ix, 1500, where="after mapped churn")


# ------------------------------------------------------- bounded footprint
def test_space_is_a_function_of_live_data_not_of_history():
    live = 3000
    pool = data(live * 3, 8, seed=8)
    with index(page_capacity=32) as ix:
        ix.insert_many(np.arange(live), pool[:live])
        seen, ids, nxt = [], np.arange(live), live
        for _ in range(8):
            ix.delete_many(ids)
            ids = np.arange(nxt, nxt + live)
            ix.insert_many(ids, pool[np.arange(nxt, nxt + live) % len(pool)])
            nxt += live
            ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=0)
            stats = check(ix, live, where="turnover")
            seen.append(stats["capacity_amplification"])
        # Flat, not merely finite. A slow climb is still unbounded.
        assert max(seen) < 2.5, seen
        assert seen[-1] <= min(seen) * 1.25, seen


def test_a_held_snapshot_costs_space_and_gives_it_back():
    v = data(1200)
    with index(page_capacity=32) as ix:
        ix.insert_many(np.arange(600), v[:600])
        lean = ix.stats()["tracked_index_bytes"] / 600
        pinned = ix.clock
        for cycle in range(4):
            ix.delete_many(np.arange(cycle * 100, cycle * 100 + 100))
            ix.insert_many(
                np.arange(600 + cycle * 100, 700 + cycle * 100),
                v[600 + cycle * 100 : 700 + cycle * 100],
            )
            ix.vacuum(oldest_snapshot=pinned, budget_versions=0)
        held = ix.stats()["tracked_index_bytes"] / ix.stats()["live_vectors"]
        assert held > lean, "a pinned snapshot should cost space"
        ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=0)
        released = ix.stats()["tracked_index_bytes"] / ix.stats()["live_vectors"]
        assert released < held, "releasing the pin should return it"


def test_an_emptied_index_shrinks_back():
    v = data(2000)
    with index(page_capacity=32) as ix:
        ix.insert_many(np.arange(1500), v[:1500])
        full = ix.stats()["pages"]
        ix.delete_many(np.arange(1500))
        ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=0)
        assert ix.stats()["live_vectors"] == 0
        assert ix.stats()["pages"] < full / 4, "emptied index kept its pages"
        ix.insert(1, v[0])  # and still works
        assert ix.search(v[0], 1)[0].id == 1


# ------------------------------------------------------- screening quality
def test_recall_is_monotone_in_nprobe_on_realistic_data():
    # Probing more pages must return more of the true neighbours. If it does
    # not, the exact-rescore shortlist is the binding constraint rather than
    # the probe count, and a user paying for more probes gets nothing.
    #
    # This is checked on spread data rather than tight clusters on purpose:
    # tight Gaussian clusters make within-cluster distances nearly equal, the
    # estimator's ranking goes noisy, and recall plateaus for reasons that
    # belong to the data. `rerank_factor` is the knob for that case.
    rng = np.random.default_rng(12)
    n, dim = 40000, 32
    pool = rng.normal(size=(n, dim)).astype(np.float32)
    queries = pool[rng.integers(0, n, 60)] + rng.normal(scale=0.1, size=(60, dim)).astype(
        np.float32
    )
    with Index(dim, metric="l2", page_capacity=256, nprobe=64) as ix:
        ix.insert_many(np.arange(n), pool)
        d2 = (queries**2).sum(1)[:, None] - 2 * queries @ pool.T + (pool**2).sum(1)[None, :]
        truth = np.argpartition(d2, 10, axis=1)[:, :10]
        seen = []
        for probe in (8, 16, 32, 64, 128):
            hits = sum(
                len(
                    {r.id for r in ix.search(queries[i], 10, nprobe=probe)} & set(truth[i].tolist())
                )
                for i in range(len(queries))
            )
            seen.append(hits / (10 * len(queries)))
        assert all(b >= a - 0.005 for a, b in zip(seen, seen[1:])), seen
        assert seen[-1] > seen[0], f"more probes bought nothing: {seen}"
