"""Write-ahead logging and crash recovery.

Without a log a crash loses every write since the last checkpoint, which for an
index whose premise is a corpus that keeps changing is not a footnote. These
tests kill the index without a clean shutdown and check what comes back.
"""

import os

import numpy as np
import pytest

from chronovec import Index


def data(n=400, dim=8, seed=5):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def open_at(path, dim=8, **kwargs):
    kwargs.setdefault("page_capacity", 32)
    kwargs.setdefault("nprobe", 32)
    kwargs.setdefault("metric", "l2")
    return Index(dim, wal_path=path, **kwargs)


def test_writes_survive_a_process_that_never_shut_down(tmp_path):
    path = tmp_path / "log.wal"
    vectors = data()
    index = open_at(path)
    index.insert_many(np.arange(300), vectors[:300])
    index.delete_many(np.arange(50))
    index.insert(1000, vectors[300])
    del index  # no close, no checkpoint

    recovered = open_at(path)
    assert recovered.stats()["live_vectors"] == 251
    assert recovered.search(vectors[100], 1)[0].id == 100  # survived
    assert recovered.search(vectors[0], 1)[0].id != 0  # deleted stayed deleted
    assert recovered.search(vectors[300], 1)[0].id == 1000  # last write kept
    recovered.close()


def test_recovery_is_repeatable(tmp_path):
    path = tmp_path / "log.wal"
    vectors = data(n=100)
    first = open_at(path)
    first.insert_many(np.arange(80), vectors[:80])
    del first
    second = open_at(path)
    second.insert(500, vectors[80])
    del second
    third = open_at(path)
    assert third.stats()["live_vectors"] == 81
    assert third.search(vectors[80], 1)[0].id == 500
    third.close()


def test_a_torn_tail_is_dropped_rather_than_replayed(tmp_path):
    # A crash mid-append leaves a partial record. It describes a change that
    # never committed, so replay must stop there -- and must truncate, or every
    # later replay stops at the same place for ever.
    path = tmp_path / "log.wal"
    vectors = data(n=60)
    index = open_at(path)
    index.insert_many(np.arange(50), vectors[:50])
    del index

    intact = os.path.getsize(path)
    with open(path, "ab") as handle:  # half a record
        handle.write(b"\x40\x00\x00\x00" + b"\x01" + b"\x00" * 12)
    assert os.path.getsize(path) > intact

    recovered = open_at(path)
    assert recovered.stats()["live_vectors"] == 50
    assert os.path.getsize(path) == intact  # truncated back
    recovered.insert(900, vectors[50])
    del recovered

    again = open_at(path)
    assert again.stats()["live_vectors"] == 51  # append after truncation works
    assert again.search(vectors[50], 1)[0].id == 900
    again.close()


def test_a_corrupted_record_body_stops_replay_at_that_point(tmp_path):
    path = tmp_path / "log.wal"
    vectors = data(n=40)
    index = open_at(path)
    for position in range(20):
        index.insert(position, vectors[position])
    del index

    with open(path, "r+b") as handle:  # flip a byte in the middle
        handle.seek(os.path.getsize(path) // 2)
        byte = handle.read(1)
        handle.seek(os.path.getsize(path) // 2)
        handle.write(bytes([byte[0] ^ 0xFF]))

    recovered = open_at(path)
    # Some prefix survives; nothing past the damage is replayed, and it does
    # not raise.
    assert 0 <= recovered.stats()["live_vectors"] < 20
    recovered.close()


def test_an_empty_or_absent_log_is_a_normal_empty_index(tmp_path):
    absent = open_at(tmp_path / "missing.wal")
    assert absent.stats()["live_vectors"] == 0
    absent.close()
    empty = tmp_path / "empty.wal"
    empty.write_bytes(b"")
    index = open_at(empty)
    assert index.stats()["live_vectors"] == 0
    index.close()


def test_a_log_from_a_different_width_is_refused(tmp_path):
    path = tmp_path / "log.wal"
    index = open_at(path, dim=8)
    index.insert_many(np.arange(10), data(n=10, dim=8))
    del index
    with pytest.raises(RuntimeError, match="different width"):
        open_at(path, dim=16)


def test_durability_and_a_mapped_index_work_together(tmp_path):
    # The arena is a spill file, not a record: it holds what the pages
    # currently need and is discarded on close. The log is what says what the
    # index contains, so recovery makes a fresh arena and the replay refills
    # it. Nothing has to survive in the arena across a crash.
    log = tmp_path / "log.wal"
    arena = tmp_path / "arena.bin"
    vectors = data(n=600, dim=8)

    index = Index(
        8,
        page_capacity=32,
        nprobe=32,
        metric="l2",
        wal_path=log,
        arena_path=arena,
        max_vectors=8000,
    )
    index.insert_many(np.arange(500), vectors[:500])
    index.delete_many(np.arange(100))
    assert arena.exists()
    del index  # crash: no clean shutdown
    assert not arena.exists()  # spill file went with it
    assert log.exists() and log.stat().st_size > 0

    recovered = Index(
        8,
        page_capacity=32,
        nprobe=32,
        metric="l2",
        wal_path=log,
        arena_path=arena,
        max_vectors=8000,
    )
    assert recovered.stats()["live_vectors"] == 400
    assert arena.exists()  # rebuilt by the replay
    assert recovered.search(vectors[250], 1)[0].id == 250
    assert recovered.search(vectors[0], 1)[0].id != 0
    recovered.close()


def test_an_arena_without_a_size_is_refused(tmp_path):
    with pytest.raises(ValueError, match="max_vectors must be positive"):
        Index(8, metric="l2", wal_path=tmp_path / "l.wal", arena_path=tmp_path / "a.bin")


def test_syncing_every_commit_costs_something_and_can_be_turned_off(tmp_path):
    # Not a performance assertion, a semantics one: both settings must produce
    # the same recovered contents after a clean handover.
    for sync in (True, False):
        path = tmp_path / f"log-{sync}.wal"
        index = open_at(path, wal_sync=sync)
        index.insert_many(np.arange(50), data(n=50))
        del index
        recovered = open_at(path, wal_sync=sync)
        assert recovered.stats()["live_vectors"] == 50
        recovered.close()
