"""Edge cases for the native engine.

Written to find bugs rather than to confirm the happy path, so several of these
probe values the implementation has a special meaning for: -1 is the empty-slot
marker, snapshot 0 means "now" across the C ABI, and the cosine metric cannot
normalise a zero vector.
"""

import numpy as np
import pytest

from chronovec import Index


def make(dim=4, **kwargs):
    kwargs.setdefault("metric", "l2")
    kwargs.setdefault("page_capacity", 8)
    kwargs.setdefault("nprobe", 32)
    return Index(dim, **kwargs)


# -- empty and near-empty ------------------------------------------------
def test_search_on_an_empty_index_returns_nothing():
    with make() as index:
        assert index.search(np.zeros(4, dtype=np.float32), 10) == []
        assert index.stats()["live_vectors"] == 0


def test_k_larger_than_the_live_set_returns_what_exists():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        found = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 100)
        assert [r.id for r in found] == [1]


def test_k_of_zero_returns_nothing_rather_than_everything():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        assert index.search(np.array([1, 0, 0, 0], dtype=np.float32), 0) == []


def test_an_index_emptied_by_deletes_still_answers():
    with make() as index:
        for i in range(20):
            index.insert(i, np.eye(4, dtype=np.float32)[i % 4])
        index.delete_many(np.arange(20))
        assert index.stats()["live_vectors"] == 0
        assert index.search(np.array([1, 0, 0, 0], dtype=np.float32), 5) == []
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
        assert index.search(np.array([1, 0, 0, 0], dtype=np.float32), 5) == []


# -- ids at the boundaries -----------------------------------------------
def test_negative_ids_round_trip():
    with make() as index:
        index.insert(-42, np.array([1, 0, 0, 0], dtype=np.float32))
        assert index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)[0].id == -42


def test_the_empty_slot_marker_is_usable_as_an_id():
    # -1 is what the engine writes into a vacated slot. An id of -1 must not be
    # confused with an empty slot.
    with make() as index:
        index.insert(-1, np.array([1, 0, 0, 0], dtype=np.float32))
        index.insert(2, np.array([0, 1, 0, 0], dtype=np.float32))
        found = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 2)
        assert found[0].id == -1
        assert index.stats()["live_vectors"] == 2
        index.delete(-1)
        assert index.stats()["live_vectors"] == 1


def test_very_large_ids_survive_the_int64_boundary():
    big = 2**62
    with make() as index:
        index.insert(big, np.array([1, 0, 0, 0], dtype=np.float32))
        assert index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)[0].id == big


def test_reinserting_the_same_id_replaces_rather_than_duplicates():
    with make() as index:
        index.insert(7, np.array([1, 0, 0, 0], dtype=np.float32))
        index.insert(7, np.array([0, 1, 0, 0], dtype=np.float32))
        assert index.stats()["live_vectors"] == 1
        assert index.search(np.array([0, 1, 0, 0], dtype=np.float32), 1)[0].id == 7


# -- malformed input ------------------------------------------------------
def test_configuration_errors_say_what_is_wrong():
    with pytest.raises(RuntimeError, match="page_capacity must be at least 8"):
        Index(4, metric="l2", page_capacity=4)
    with pytest.raises(RuntimeError, match="dimensions must be positive"):
        Index(0, metric="l2")


def test_wrong_width_is_rejected():
    with make() as index:
        with pytest.raises(Exception):
            index.insert(1, np.array([1, 0], dtype=np.float32))
        with pytest.raises(Exception):
            index.search(np.array([1, 0], dtype=np.float32), 1)


def test_a_zero_vector_is_rejected_under_cosine_and_fine_under_l2():
    with make(metric="cosine") as index, pytest.raises(Exception):
        index.insert(1, np.zeros(4, dtype=np.float32))
    with make(metric="l2") as index:
        index.insert(1, np.zeros(4, dtype=np.float32))
        assert index.search(np.zeros(4, dtype=np.float32), 1)[0].id == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_do_not_corrupt_the_index(bad):
    # Whether they are rejected or stored is the engine's choice; what must not
    # happen is that a later ordinary query stops working.
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        try:
            index.insert(2, np.array([bad, 0, 0, 0], dtype=np.float32))
        except Exception:
            pass
        found = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)
        assert found and found[0].id in (1, 2)
        assert index.stats()["live_vectors"] >= 1


def test_deleting_an_unknown_id_is_an_error_not_a_crash():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        with pytest.raises(Exception):
            index.delete(999)
        assert index.stats()["live_vectors"] == 1


# -- degenerate geometry --------------------------------------------------
def test_identical_vectors_split_and_remain_retrievable():
    # Splitting picks two seeds by distance; with every vector identical there
    # is no distance to pick by.
    with make(page_capacity=8) as index:
        same = np.array([1, 0, 0, 0], dtype=np.float32)
        for i in range(40):
            index.insert(i, same)
        assert index.stats()["live_vectors"] == 40
        found = index.search(same, 10)
        assert len(found) == 10


def test_one_dimensional_vectors_work():
    with make(dim=1) as index:
        for i in range(30):
            index.insert(i, np.array([float(i)], dtype=np.float32))
        assert index.search(np.array([29.0], dtype=np.float32), 1)[0].id == 29


def test_the_smallest_allowed_page_capacity_still_splits_correctly():
    with make(page_capacity=8) as index:
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(50, 4)).astype(np.float32)
        index.insert_many(np.arange(50), vectors)
        stats = index.stats()
        assert stats["live_vectors"] == 50
        # It split -- 50 rows cannot fit one page of 8 -- and consolidation
        # then packed them back to near the minimum the capacity allows.
        assert stats["pages"] > 1
        assert stats["pages"] <= 50 // 8 + 5
        for probe in (0, 25, 49):
            assert index.search(vectors[probe], 1)[0].id == probe


# -- snapshots ------------------------------------------------------------
def test_a_snapshot_of_zero_is_refused_because_the_abi_reads_it_as_now():
    # An empty index reports clock 0, so `index.clock` looks like a usable
    # token before the first write and would silently return the present.
    with make() as index:
        assert index.clock == 0
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        with pytest.raises(ValueError, match="means 'now'"):
            index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1, snapshot=0)


def test_a_snapshot_taken_between_writes_hides_the_later_one():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        between = index.clock
        index.insert(2, np.array([1, 0, 0, 0], dtype=np.float32))
        assert len(index.search(np.array([1, 0, 0, 0], dtype=np.float32), 5)) == 2
        early = index.search(np.array([1, 0, 0, 0], dtype=np.float32), 5, snapshot=between)
        assert [r.id for r in early] == [1]


def test_a_snapshot_beyond_the_clock_behaves_like_the_present():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        far = index.clock + 1_000_000
        assert index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1, snapshot=far)[0].id == 1


def test_vacuum_with_nothing_to_reclaim_is_a_no_op():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        assert index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0) == 0
        assert index.stats()["live_vectors"] == 1


def test_a_pinned_snapshot_blocks_reclamation_and_releasing_it_frees():
    with make() as index:
        for i in range(20):
            index.insert(i, np.eye(4, dtype=np.float32)[i % 4] * (i + 1))
        pinned = index.clock
        index.delete_many(np.arange(10))
        assert index.vacuum(oldest_snapshot=pinned, budget_versions=0) == 0
        assert index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0) == 10


# -- batch paths ----------------------------------------------------------
def test_empty_batches_are_accepted():
    with make() as index:
        assert (
            index.insert_many(np.array([], dtype=np.int64), np.zeros((0, 4), dtype=np.float32)) == 0
        )
        assert index.delete_many(np.array([], dtype=np.int64)) == 0


def test_a_batch_of_one_matches_a_single_insert():
    with make() as a, make() as b:
        vector = np.array([1, 0, 0, 0], dtype=np.float32)
        a.insert(1, vector)
        b.insert_many(np.array([1]), vector[None, :])
        assert a.stats()["live_vectors"] == b.stats()["live_vectors"] == 1
        assert a.search(vector, 1)[0].id == b.search(vector, 1)[0].id == 1


def test_a_batch_larger_than_one_block_with_every_row_identical():
    # Crosses the 8192-row assignment block with the worst possible geometry.
    with make(page_capacity=16) as index:
        same = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (9000, 1))
        assert index.insert_many(np.arange(9000), same) == 9000
        assert index.stats()["live_vectors"] == 9000


def test_mismatched_batch_lengths_are_rejected():
    with make() as index, pytest.raises(Exception):
        index.insert_many(np.arange(3), np.zeros((2, 4), dtype=np.float32))


# -- lifecycle ------------------------------------------------------------
def test_using_a_closed_index_raises_rather_than_answering_emptily():
    # A closed index used to return [] from search, which is indistinguishable
    # from an index that is merely empty.
    index = make()
    index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
    index.close()
    with pytest.raises(RuntimeError, match="closed"):
        index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1)
    with pytest.raises(RuntimeError):
        index.insert(2, np.array([0, 1, 0, 0], dtype=np.float32))


def test_closing_twice_is_safe():
    index = make()
    index.close()
    index.close()


def test_sustained_full_turnover_does_not_grow_the_index():
    # The claim the whole index rests on. Empty pages were never removed from
    # the directory, so under repeated full turnover the page count climbed
    # without bound while the live count stayed flat.
    live = 4000
    with make(page_capacity=32, dim=8) as index:
        rng = np.random.default_rng(21)
        pool = rng.normal(size=(live * 4, 8)).astype(np.float32)
        index.insert_many(np.arange(live), pool[:live])
        ids = np.arange(live)
        after_first = None
        nxt = live
        for cycle in range(6):
            index.delete_many(ids)
            ids = np.arange(nxt, nxt + live)
            index.insert_many(ids, pool[np.arange(nxt, nxt + live) % len(pool)])
            nxt += live
            index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
            stats = index.stats()
            assert stats["live_vectors"] == live
            if cycle == 0:
                after_first = stats["capacity_amplification"]
            else:
                # Flat, not merely finite: a slow climb is still unbounded.
                assert stats["capacity_amplification"] <= after_first * 1.25


def test_pages_emptied_by_deletion_are_removed_from_the_directory():
    with make(page_capacity=8, dim=4) as index:
        rng = np.random.default_rng(22)
        vectors = rng.normal(size=(400, 4)).astype(np.float32)
        index.insert_many(np.arange(400), vectors)
        full = index.stats()["pages"]
        index.delete_many(np.arange(350))
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
        assert index.stats()["live_vectors"] == 50
        assert index.stats()["pages"] < full / 2


def test_the_last_page_survives_emptying_the_index():
    with make(page_capacity=8, dim=4) as index:
        rng = np.random.default_rng(23)
        vectors = rng.normal(size=(60, 4)).astype(np.float32)
        index.insert_many(np.arange(60), vectors)
        index.delete_many(np.arange(60))
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
        assert index.stats()["pages"] >= 1
        index.insert(999, vectors[0])  # still has somewhere to land
        assert index.search(vectors[0], 1)[0].id == 999


def test_a_batch_insert_is_atomic_to_a_concurrent_reader():
    # Rows used to become visible as their pages were published, each with its
    # own timestamp, so a reader sampling the clock mid-batch saw a fraction of
    # the batch. Allocation and commit are separate now: one timestamp for the
    # call, published after every page is.
    import threading

    rng = np.random.default_rng(31)
    with make(dim=8, page_capacity=32) as index:
        index.insert_many(np.arange(2000), rng.normal(size=(2000, 8)).astype(np.float32))
        batch = rng.normal(size=(4000, 8)).astype(np.float32)
        observed, stop = [], threading.Event()

        def reader():
            while not stop.is_set():
                snapshot = index.clock
                if not snapshot:
                    continue
                observed.append(
                    sum(1 for r in index.search(batch[0], 1000, snapshot=snapshot) if r.id >= 2000)
                )

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        index.insert_many(np.arange(2000, 6000), batch)
        stop.set()
        thread.join(timeout=10)

        assert observed, "reader never sampled"
        # Two outcomes only: nothing of the batch, or all of it. Any value in
        # between is a torn read.
        assert len(set(observed)) <= 2, f"partial batches seen: {sorted(set(observed))}"
        assert min(observed) == 0 or len(set(observed)) == 1


def test_an_uncommitted_timestamp_is_never_handed_out_as_a_snapshot():
    with make() as index:
        index.insert(1, np.array([1, 0, 0, 0], dtype=np.float32))
        # The clock a caller reads is the committed one, so a snapshot taken
        # from it can always be read at.
        snapshot = index.clock
        assert (
            index.search(np.array([1, 0, 0, 0], dtype=np.float32), 1, snapshot=snapshot)[0].id == 1
        )


def test_a_batch_delete_and_insert_keep_their_own_visibility():
    with make(dim=4, page_capacity=8) as index:
        vectors = np.eye(4, dtype=np.float32)[np.arange(40) % 4] * (1 + np.arange(40)[:, None])
        index.insert_many(np.arange(40), vectors)
        before = index.clock
        index.delete_many(np.arange(20))
        after_delete = index.clock
        assert after_delete > before
        assert index.stats()["live_vectors"] == 20
        # The snapshot before the delete still sees everything.
        found = index.search(vectors[0], 40, snapshot=before)
        assert len([r for r in found if r.id < 20]) > 0


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
def test_the_lane_count_changes_speed_and_nothing_else(lanes):
    # The apply phase runs in parallel across pages. The index it builds must
    # not depend on how the work divided: unsorted split order made the page
    # count differ by lane count, which is a reproducibility bug even though
    # the answers were right.
    rng = np.random.default_rng(41)
    vectors = rng.normal(size=(6000, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32, threads=lanes) as index:
        assert index.insert_many(np.arange(6000), vectors) == 6000
        stats = index.stats()
        # The page count is asserted, not just the answers: unsorted split
        # order once made it differ by lane count, which is a reproducibility
        # bug even when every query is right.
        assert (stats["live_vectors"], stats["pages"]) == (6000, 283)
        for probe in (0, 2999, 5999):
            assert index.search(vectors[probe], 1)[0].id == probe


def test_parallel_apply_handles_more_lanes_than_pages():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(20, 4)).astype(np.float32)
    with make(page_capacity=32, threads=16) as index:
        assert index.insert_many(np.arange(20), vectors) == 20
        assert index.stats()["pages"] == 1
        assert index.search(vectors[7], 1)[0].id == 7


def test_a_bulk_load_consolidates_itself_without_being_asked():
    # Merging used to run only inside vacuum, so an index that was bulk loaded
    # and never vacuumed kept every page its splits produced. That is what
    # everyone does first, and the bloat was invisible until measured.
    rng = np.random.default_rng(51)
    vectors = rng.normal(size=(20000, 16)).astype(np.float32)
    with make(dim=16, page_capacity=64) as index:
        index.insert_many(np.arange(20000), vectors)
        tidy = index.stats()["capacity_amplification"]
    with make(dim=16, page_capacity=64, auto_consolidate=False) as index:
        index.insert_many(np.arange(20000), vectors)
        untidy = index.stats()["capacity_amplification"]
    assert tidy < untidy
    assert tidy < 2.0


def test_turning_consolidation_off_still_leaves_a_correct_index():
    rng = np.random.default_rng(52)
    vectors = rng.normal(size=(5000, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32, auto_consolidate=False) as index:
        index.insert_many(np.arange(5000), vectors)
        assert index.stats()["live_vectors"] == 5000
        for probe in (0, 2500, 4999):
            assert index.search(vectors[probe], 1)[0].id == probe
        # And a later vacuum still cleans up what was left.
        before = index.stats()["capacity_amplification"]
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
        assert index.stats()["capacity_amplification"] < before


def test_a_small_insert_does_not_pay_for_consolidation():
    # Only batches large enough to have split much are charged for it.
    with make(dim=8, page_capacity=64) as index:
        rng = np.random.default_rng(53)
        vectors = rng.normal(size=(2000, 8)).astype(np.float32)
        index.insert_many(np.arange(2000), vectors)
        pages = index.stats()["pages"]
        index.insert_many(np.arange(2000, 2010), vectors[:10])
        assert index.stats()["live_vectors"] == 2010
        assert index.stats()["pages"] >= pages - 1


# -- batched queries ------------------------------------------------------
def test_search_many_matches_searching_one_at_a_time():
    rng = np.random.default_rng(61)
    vectors = rng.normal(size=(2000, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32) as index:
        index.insert_many(np.arange(2000), vectors)
        queries = vectors[[0, 500, 999, 1500]]
        one = [[r.id for r in index.search(q, 10)] for q in queries]
        many = [[r.id for r in row] for row in index.search_many(queries, 10)]
        assert one == many


def test_search_many_as_arrays_pads_and_matches():
    rng = np.random.default_rng(62)
    vectors = rng.normal(size=(50, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32) as index:
        index.insert_many(np.arange(50), vectors)
        ids, distances = index.search_many(vectors[:3], 80, as_arrays=True)
        assert ids.shape == (3, 80) and distances.shape == (3, 80)
        # Only 50 records exist, so every row is padded past that.
        assert (ids[:, 50:] == -1).all()
        assert np.isinf(distances[:, 50:]).all()
        assert list(ids[0][:1]) == [0]


def test_search_many_honours_a_snapshot_for_the_whole_batch():
    rng = np.random.default_rng(63)
    vectors = rng.normal(size=(200, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32) as index:
        index.insert_many(np.arange(100), vectors[:100])
        pinned = index.clock
        index.insert_many(np.arange(100, 200), vectors[100:])
        rows = index.search_many(vectors[:4], 50, snapshot=pinned)
        assert all(all(r.id < 100 for r in row) for row in rows)


def test_search_many_rejects_the_same_mistakes_search_does():
    with make(dim=8, page_capacity=32) as index:
        index.insert_many(
            np.arange(50), np.random.default_rng(64).normal(size=(50, 8)).astype(np.float32)
        )
        with pytest.raises(ValueError, match="expected queries"):
            index.search_many(np.zeros((2, 4), dtype=np.float32), 5)
        with pytest.raises(ValueError, match="means 'now'"):
            index.search_many(np.zeros((1, 8), dtype=np.float32), 5, snapshot=0)
        assert index.search_many(np.zeros((0, 8), dtype=np.float32), 5) == []


def test_search_many_on_a_closed_index_raises():
    index = make(dim=8, page_capacity=32)
    index.insert_many(
        np.arange(20), np.random.default_rng(65).normal(size=(20, 8)).astype(np.float32)
    )
    index.close()
    with pytest.raises(RuntimeError, match="closed"):
        index.search_many(np.zeros((1, 8), dtype=np.float32), 5)


def test_labelled_bulk_insert_filters_correctly():
    rng = np.random.default_rng(71)
    vectors = rng.normal(size=(3000, 8)).astype(np.float32)
    labels = np.array([1 << (i % 4) for i in range(3000)], dtype=np.uint64)
    with make(dim=8, page_capacity=64, labels=True) as index:
        assert index.insert_many(np.arange(3000), vectors, labels=labels) == 3000
        found = [r.id for r in index.search(vectors[9], 40, require_all=1 << 1)]
        assert found and all(i % 4 == 1 for i in found)
        excluded = [r.id for r in index.search(vectors[9], 40, exclude=1 << 1)]
        assert excluded and all(i % 4 != 1 for i in excluded)


def test_labels_on_an_index_without_them_are_refused():
    rng = np.random.default_rng(72)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32) as index, pytest.raises(ValueError, match="labels=True"):
        index.insert_many(np.arange(20), vectors, labels=np.ones(20, dtype=np.uint64))


def test_mismatched_label_count_is_refused():
    rng = np.random.default_rng(73)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    with make(dim=8, page_capacity=32, labels=True) as index:
        with pytest.raises(ValueError, match="expected 20 labels"):
            index.insert_many(np.arange(20), vectors, labels=np.ones(5, dtype=np.uint64))


def test_labelled_batch_matches_labelled_single_insert():
    rng = np.random.default_rng(74)
    vectors = rng.normal(size=(500, 8)).astype(np.float32)
    labels = np.array([1 << (i % 3) for i in range(500)], dtype=np.uint64)
    with (
        make(dim=8, page_capacity=32, labels=True) as batched,
        make(dim=8, page_capacity=32, labels=True) as singly,
    ):
        batched.insert_many(np.arange(500), vectors, labels=labels)
        for i in range(500):
            singly.insert(i, vectors[i], label=int(labels[i]))
        for probe in (0, 250, 499):
            a = {r.id for r in batched.search(vectors[probe], 20, require_all=1)}
            b = {r.id for r in singly.search(vectors[probe], 20, require_all=1)}
            assert a == b
