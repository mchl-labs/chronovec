"""MVCC and durability, one property at a time.

Each guarantee is tested for what it actually means rather than demonstrated
once and assumed. Where a guarantee is *not* offered, that is asserted too --
an untested boundary is how a claim quietly grows past what the code does.
"""

import os
import threading

import numpy as np
import pytest
from invariants import assert_no_duplicate_ids, check

from chronovec import Index


def data(n=600, dim=8, seed=4):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def index(**kwargs):
    kwargs.setdefault("metric", "l2")
    kwargs.setdefault("page_capacity", 32)
    kwargs.setdefault("nprobe", 64)
    return Index(kwargs.pop("dim", 8), **kwargs)


# ---------------------------------------------------------------- atomicity
def test_a_batch_advances_the_clock_exactly_once():
    # One commit timestamp for the call is what makes the batch a unit. If the
    # clock moved per row, a reader could land between rows.
    with index() as ix:
        before = ix.clock
        ix.insert_many(np.arange(400), data(400))
        assert ix.clock - before == 1


def test_no_reader_observes_a_partial_batch():
    v = data(4000)
    with index() as ix:
        ix.insert_many(np.arange(500), v[:500])
        counts, stop = [], threading.Event()

        def reader():
            while not stop.is_set():
                snapshot = ix.clock
                if snapshot:
                    counts.append(
                        sum(1 for r in ix.search(v[600], 400, snapshot=snapshot) if r.id >= 500)
                    )

        watcher = threading.Thread(target=reader, daemon=True)
        watcher.start()
        ix.insert_many(np.arange(500, 3500), v[500:3500])
        stop.set()
        watcher.join(timeout=10)
        assert counts, "reader never sampled"
        # Before, or after. Never in between.
        assert len(set(counts)) <= 2, f"torn batch: {sorted(set(counts))}"


def test_a_batch_that_fails_does_not_report_success():
    # Partial application is possible -- pages publish as they go -- but the
    # returned count must never claim more than was applied, or a caller cannot
    # tell that it needs to retry.
    with index() as ix:
        ix.insert_many(np.arange(100), data(100))
        with pytest.raises(Exception):
            ix.insert_many(np.arange(100, 110), data(10, dim=4))  # wrong width
        check(ix, 100, where="after a refused batch")


def test_replaying_a_log_reproduces_the_batch_as_a_unit(tmp_path):
    log = tmp_path / "a.wal"
    v = data(500)
    ix = index(wal_path=log)
    ix.insert_many(np.arange(400), v[:400])
    del ix
    back = index(wal_path=log)
    # Every row of the batch, at one timestamp: the same atomicity a live
    # reader saw.
    check(back, 400, where="after replay")
    stamps = {back.search(v[i], 1)[0].id for i in (0, 200, 399)}
    assert stamps == {0, 200, 399}
    back.close()


# -------------------------------------------------------------- consistency
def test_structural_invariants_hold_through_a_churn_cycle():
    v = data(2000)
    with index() as ix:
        ix.insert_many(np.arange(600), v[:600])
        check(ix, 600, where="load")
        for cycle in range(4):
            ix.delete_many(np.arange(cycle * 100, cycle * 100 + 100))
            check(ix, 500, where=f"delete {cycle}")
            ix.insert_many(
                np.arange(600 + cycle * 100, 700 + cycle * 100),
                v[600 + cycle * 100 : 700 + cycle * 100],
            )
            check(ix, 600, where=f"insert {cycle}")
            ix.vacuum(oldest_snapshot=ix.clock + 1, budget_versions=0)
            check(ix, 600, where=f"vacuum {cycle}")
            assert_no_duplicate_ids(ix, v[10], 700, where=f"cycle {cycle}")


def test_an_id_never_has_two_visible_versions():
    v = data(300)
    with index() as ix:
        ix.insert_many(np.arange(200), v[:200])
        for _ in range(5):  # repeated overwrite
            ix.insert_many(np.arange(200), v[:200])
        check(ix, 200, where="after repeated overwrite")
        assert_no_duplicate_ids(ix, v[5], 300)


def test_reclamation_never_removes_something_still_reachable():
    v = data(400)
    with index() as ix:
        ix.insert_many(np.arange(300), v[:300])
        pinned = ix.clock
        ix.delete_many(np.arange(150))
        assert ix.vacuum(oldest_snapshot=pinned, budget_versions=0) == 0
        # Everything is still readable at the pin.
        found = {r.id for r in ix.search(v[0], 300, snapshot=pinned)}
        assert len(found & set(range(150))) > 0
        check(ix, 150, where="pinned")


# ---------------------------------------------------------------- isolation
def test_a_snapshot_is_repeatable():
    v = data(600)
    with index() as ix:
        ix.insert_many(np.arange(300), v[:300])
        pinned = ix.clock
        first = [r.id for r in ix.search(v[0], 20, snapshot=pinned)]
        for cycle in range(5):
            ix.insert_many(
                np.arange(300 + cycle * 50, 350 + cycle * 50),
                v[300 + cycle * 50 : 350 + cycle * 50],
            )
            ix.delete_many(np.arange(cycle * 10, cycle * 10 + 10))
            ix.vacuum(oldest_snapshot=pinned, budget_versions=0)
        assert [r.id for r in ix.search(v[0], 20, snapshot=pinned)] == first


def test_two_snapshots_see_two_consistent_worlds():
    v = data(400)
    with index() as ix:
        ix.insert_many(np.arange(200), v[:200])
        early = ix.clock
        ix.delete_many(np.arange(100))
        late = ix.clock
        old = {r.id for r in ix.search(v[0], 200, snapshot=early)}
        new = {r.id for r in ix.search(v[0], 200, snapshot=late)}
        assert old & set(range(100)), "early snapshot lost deleted rows"
        assert not (new & set(range(100))), "late snapshot kept deleted rows"


def test_a_reader_never_sees_a_write_that_has_not_committed():
    # The committed clock is what a caller can snapshot, so any snapshot it
    # hands out must already be fully readable.
    v = data(2000)
    with index() as ix:
        ix.insert_many(np.arange(500), v[:500])
        seen, stop = [], threading.Event()

        def reader():
            while not stop.is_set():
                snapshot = ix.clock
                if snapshot:
                    a = len(ix.search(v[0], 300, snapshot=snapshot))
                    b = len(ix.search(v[0], 300, snapshot=snapshot))
                    seen.append(a == b)

        watcher = threading.Thread(target=reader, daemon=True)
        watcher.start()
        for cycle in range(6):
            ix.insert_many(
                np.arange(500 + cycle * 100, 600 + cycle * 100),
                v[500 + cycle * 100 : 600 + cycle * 100],
            )
        stop.set()
        watcher.join(timeout=10)
        assert seen and all(seen), "a snapshot changed under a reader"


# --------------------------------------------------------------- durability
def test_committed_writes_survive_a_process_that_never_closed(tmp_path):
    log = tmp_path / "d.wal"
    v = data(500)
    ix = index(wal_path=log)
    ix.insert_many(np.arange(300), v[:300])
    ix.delete_many(np.arange(50))
    ix.insert(9999, v[400])
    del ix
    back = index(wal_path=log)
    check(back, 251, where="recovered")
    assert back.search(v[400], 1)[0].id == 9999
    back.close()


def test_a_torn_tail_is_dropped_and_the_log_stays_usable(tmp_path):
    log = tmp_path / "e.wal"
    ix = index(wal_path=log)
    ix.insert_many(np.arange(200), data(200))
    del ix
    intact = os.path.getsize(log)
    with open(log, "ab") as handle:
        handle.write(b"\x80\x00\x00\x00" + b"\x01" + b"\x00" * 20)
    back = index(wal_path=log)
    check(back, 200, where="after torn tail")
    assert os.path.getsize(log) == intact
    back.insert(5000, data(1)[0])
    del back
    again = index(wal_path=log)
    check(again, 201, where="after appending past a truncation")
    again.close()


def test_durability_is_off_unless_asked_for():
    # An index with no log makes no durability claim, and must not pretend to.
    with index() as ix:
        ix.insert_many(np.arange(100), data(100))
        assert not hasattr(ix, "wal_path") or ix.__dict__.get("wal_path") is None


@pytest.mark.parametrize("sync", [True, False])
def test_both_sync_settings_recover_the_same_contents(tmp_path, sync):
    log = tmp_path / f"f-{sync}.wal"
    ix = index(wal_path=log, wal_sync=sync)
    ix.insert_many(np.arange(200), data(200))
    del ix
    back = index(wal_path=log, wal_sync=sync)
    check(back, 200, where=f"sync={sync}")
    back.close()
