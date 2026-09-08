import threading

import numpy as np
import pytest

from chronovec.branchable_memory import BranchableMemory, BranchConflictError
from chronovec.memory import BranchError


def unit(rng, dimensions=16):
    vector = rng.standard_normal(dimensions).astype(np.float32)
    return vector / np.linalg.norm(vector)


def seen(view, vector):
    return {hit.id for hit, _ in view.search(vector, k=20)}


def test_branch_uses_a_private_delta_without_changing_main_index():
    rng = np.random.default_rng(100)
    with BranchableMemory(16, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base, text="base")
        stats_before = memory._index.stats().copy()
        branch = memory.branch("draft")
        branch.add(2, base, text="private")

        assert memory._index.stats() == stats_before
        assert memory._index.labels is False
        assert branch._state.delta.stats()["live_vectors"] == 1
        assert seen(memory, base) == {1}
        assert seen(branch, base) == {1, 2}


def test_branch_page_capacity_sizes_branch_deltas_independently_of_main():
    rng = np.random.default_rng(108)
    # Every branch here only ever holds a handful of records: a small
    # branch_page_capacity avoids paying main's page_capacity worth of
    # preallocated native storage per branch, without touching main's own
    # page sizing (and therefore its page-count-driven routing cost) at all.
    with BranchableMemory(16, nprobe=64, page_capacity=256, branch_page_capacity=8) as memory:
        for i in range(300):
            memory.add(i, unit(rng), text=f"main-{i}")
        main_pages_at_256 = memory._index.stats()["pages"]

        branch = memory.branch("tiny")
        for i in range(300):
            branch.add(1_000 + i, unit(rng), text=f"branch-{i}")
        branch_pages_at_8 = branch._state.delta.stats()["pages"]

        # Same record count, very different page counts: main used capacity
        # 256 (few, large pages), the branch used capacity 8 (many, tiny
        # pages) -- proving the two are sized independently, not sharing
        # one setting the way they did before this parameter existed.
        assert branch_pages_at_8 > main_pages_at_256 * 5


def test_branch_page_capacity_defaults_to_matching_main_unchanged():
    rng = np.random.default_rng(112)
    with BranchableMemory(16, nprobe=64, page_capacity=32) as memory:
        assert memory.branch_page_capacity is None
        branch = memory.branch("draft")
        for i in range(80):
            branch.add(i, unit(rng), text=f"branch-{i}")
        # page_capacity=32 applied to the branch too, exactly as it did
        # before branch_page_capacity existed: every page holds 32 slots,
        # so the allocated total is always a multiple of 32.
        assert branch._state.delta.stats()["allocated_slots"] % 32 == 0
        assert branch._state.delta.stats()["pages"] >= 3


def test_main_view_is_compatible_with_agent_memory_callers():
    rng = np.random.default_rng(109)
    with BranchableMemory(16, nprobe=64) as memory:
        vector = unit(rng)
        assert memory.main is memory
        memory.main.add("fact", vector, text="main-view")
        assert memory.main.search(vector, k=1)[0][1].payload == {"text": "main-view"}


def test_branch_isolation_and_historical_base_are_exact():
    rng = np.random.default_rng(101)
    with BranchableMemory(16, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        checkpoint = memory.snapshot()
        memory.add(2, base)
        left = memory.branch("left", snapshot=checkpoint)
        right = memory.branch("right", snapshot=checkpoint)
        left.add(3, base)

        assert seen(memory, base) == {1, 2}
        assert seen(left, base) == {1, 3}
        assert seen(right, base) == {1}
        assert {hit.id for hit, _ in left.as_of_fork(base, k=20)} == {1}


def test_branch_update_and_delete_mask_the_forked_base_only():
    rng = np.random.default_rng(102)
    with BranchableMemory(16, nprobe=64) as memory:
        original = unit(rng)
        revised = unit(rng)
        memory.add(1, original, text="original")
        memory.add(2, original, text="remove")
        branch = memory.branch("retry")
        branch.add(1, revised, text="revised")
        branch.delete(2)

        assert memory.search(original, k=1)[0][1].payload["text"] == "original"
        assert branch.search(revised, k=1)[0][1].payload["text"] == "revised"
        assert 2 not in seen(branch, original)
        assert 2 in seen(memory, original)


def test_branch_search_widens_the_base_fetch_only_by_deletes_not_updates(monkeypatch):
    """An update replaces its stale base entry in place (via the delta_hits
    merge), so it never shrinks the visible candidate pool the way a delete
    does. Padding the base fetch by the branch's full write count instead of
    just its deletes would make an update-heavy branch's search cost scale
    with total branch history rather than with what was actually deleted."""
    rng = np.random.default_rng(120)
    with BranchableMemory(16, nprobe=64) as memory:
        vectors = [unit(rng) for _ in range(20)]
        for item_id, vector in enumerate(vectors):
            memory.add(item_id, vector, text=f"original-{item_id}")
        branch = memory.branch("update-heavy")
        for item_id in range(20):
            branch.add(item_id, vectors[item_id], text=f"revised-{item_id}")
        branch.delete(0)

        requested_k: list[int] = []
        original_search = memory._index.search

        def capturing_search(*args, **kwargs):
            requested_k.append(kwargs["k"])
            return original_search(*args, **kwargs)

        monkeypatch.setattr(memory._index, "search", capturing_search)
        results = branch.search(vectors[5], k=3)

        assert requested_k == [3 + 1]  # k + len(state.deleted), not k + 20 updates
        assert {record.payload["text"] for _, record in results} <= {
            f"revised-{i}" for i in range(1, 20)
        }


def test_merge_promotes_delta_and_deletions_then_releases_resources():
    rng = np.random.default_rng(103)
    with BranchableMemory(16, nprobe=64) as memory:
        original = unit(rng)
        revised = unit(rng)
        memory.add(1, original, text="original")
        memory.add(2, original, text="remove")
        branch = memory.branch("retry")
        branch.add(1, revised, text="revised")
        branch.delete(2)

        assert branch.diff() == {"upserts": (1,), "deletes": (2,)}
        assert branch.merge() == 1
        assert memory.search(revised, k=1)[0][1].payload["text"] == "revised"
        assert seen(memory, original) == {1}
        assert list(memory.branches()) == []
        with pytest.raises(BranchError):
            branch.search(revised)


def test_merge_reports_write_write_conflicts_and_keeps_branch_usable():
    rng = np.random.default_rng(104)
    with BranchableMemory(16, nprobe=64) as memory:
        original = unit(rng)
        main_value = unit(rng)
        branch_value = unit(rng)
        memory.add("preference", original, text="base")
        branch = memory.branch("retry")
        branch.add("preference", branch_value, text="branch")
        memory.add("preference", main_value, text="main")

        assert branch.conflicts() == ("preference",)
        with pytest.raises(BranchConflictError, match="preference"):
            branch.merge()
        assert branch.search(branch_value, k=1)[0][1].payload["text"] == "branch"
        assert memory.search(main_value, k=1)[0][1].payload["text"] == "main"


def test_merge_is_atomic_to_branchable_memory_readers(monkeypatch):
    rng = np.random.default_rng(105)
    with BranchableMemory(16, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("atomic")
        branch.add(2, base)
        branch.add(3, base)

        entered = threading.Event()
        release = threading.Event()
        original_apply_changes = memory._index.apply_changes

        def paused_apply_changes(*args, **kwargs):
            entered.set()
            assert release.wait(2), "merge did not resume in time"
            return original_apply_changes(*args, **kwargs)

        monkeypatch.setattr(memory._index, "apply_changes", paused_apply_changes)
        errors = []
        merger = threading.Thread(target=lambda: _capture(errors, branch.merge), daemon=True)
        merger.start()
        assert entered.wait(2), "merge did not start promotion"

        observed = []
        reader = threading.Thread(target=lambda: observed.append(seen(memory, base)), daemon=True)
        reader.start()
        reader.join(timeout=0.05)
        assert reader.is_alive(), "readers must not observe a partially merged state"

        release.set()
        merger.join(timeout=2)
        reader.join(timeout=2)
        assert not errors
        assert observed == [{1, 2, 3}]


def test_checkpoint_restores_active_private_delta_and_payloads(tmp_path):
    rng = np.random.default_rng(106)
    checkpoint = tmp_path / "branchable"
    base = unit(rng)
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add(1, base, text="base")
        branch = memory.branch("retry")
        branch.add(2, base, text="private")
        memory.save(checkpoint)

    restored = BranchableMemory.load(checkpoint)
    try:
        retry = restored.get_branch("retry")
        assert seen(restored, base) == {1}
        assert seen(retry, base) == {1, 2}
        assert retry.search(base, k=2)[-1][1].payload["text"] == "private"
        retry.merge()
        assert seen(restored, base) == {1, 2}
    finally:
        restored.close()


def test_checkpoint_round_trips_branch_page_capacity(tmp_path):
    rng = np.random.default_rng(113)
    checkpoint = tmp_path / "branchable"
    with BranchableMemory(16, nprobe=64, branch_page_capacity=8) as memory:
        memory.add(1, unit(rng), text="base")
        branch = memory.branch("tiny")
        for i in range(30):
            branch.add(10 + i, unit(rng), text=f"branch-{i}")
        memory.save(checkpoint)

    restored = BranchableMemory.load(checkpoint)
    try:
        assert restored.branch_page_capacity == 8
        # A new branch created after resume still uses the restored setting,
        # not main's page_capacity (16, the class default in this test file).
        fresh = restored.branch("another")
        fresh.add(999, unit(rng), text="fresh")
        assert fresh._state.delta.stats()["allocated_slots"] % 8 == 0
    finally:
        restored.close()


def test_old_checkpoint_without_branch_page_capacity_field_still_loads(tmp_path):
    """A checkpoint written before this field existed has no such key."""
    import hashlib
    import json
    import pickle

    rng = np.random.default_rng(114)
    checkpoint = tmp_path / "branchable"
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add(1, unit(rng), text="base")
        memory.save(checkpoint)

    metadata_path = next(checkpoint.parent.glob(f"{checkpoint.name}.*.meta"))
    meta = pickle.loads(metadata_path.read_bytes())
    meta["format_version"] = 1
    del meta["branch_page_capacity"]
    metadata_path.write_bytes(pickle.dumps(meta))

    manifest_path = checkpoint.with_suffix(".branchable.manifest")
    manifest = json.loads(manifest_path.read_text())
    manifest["checksums"][metadata_path.name] = hashlib.sha256(
        metadata_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    restored = BranchableMemory.load(checkpoint)
    try:
        assert restored.branch_page_capacity is None
    finally:
        restored.close()


def test_checkpoint_path_tracks_branch_lifecycle_and_removes_old_generation(tmp_path):
    rng = np.random.default_rng(107)
    checkpoint = tmp_path / "branchable"
    with BranchableMemory(16, nprobe=64, checkpoint_path=checkpoint) as memory:
        branch = memory.branch("discard")
        branch.add(1, unit(rng), text="temporary")
        branch.discard()

    restored = BranchableMemory.load(checkpoint)
    try:
        assert list(restored.branches()) == []
        assert restored.search(unit(rng), k=10) == []
    finally:
        restored.close()
    assert len(list(tmp_path.iterdir())) == 3


def test_purge_protects_active_branch_forks_and_reclaims_after_release():
    rng = np.random.default_rng(108)
    vector = unit(rng)
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add(1, vector)
        branch = memory.branch("protect")
        memory.add(1, vector)

        with pytest.raises(ValueError, match="active branch fork point"):
            memory.purge()
        branch.discard()
        assert memory.purge(oldest_snapshot=memory.snapshot() + 1) >= 1


def test_checkpoint_rejects_mixed_or_corrupted_generation(tmp_path):
    rng = np.random.default_rng(110)
    checkpoint = tmp_path / "branchable"
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add(1, unit(rng))
        memory.save(checkpoint)

    manifest_path = checkpoint.with_suffix(".branchable.manifest")
    manifest = manifest_path.read_text()
    manifest_path.write_text(manifest.replace('"checksums": {', '"checksums": {"bad": "0",'))
    with pytest.raises(ValueError, match="checkpoint does not match"):
        BranchableMemory.load(checkpoint)


def test_failed_checkpoint_leaves_the_previous_complete_generation_loadable(tmp_path, monkeypatch):
    rng = np.random.default_rng(111)
    checkpoint = tmp_path / "branchable"
    first, second = unit(rng), unit(rng)
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add(1, first)
        memory.save(checkpoint)
        memory.add(2, second)

        def fail_manifest(*_args, **_kwargs):
            raise RuntimeError("manifest unavailable")

        monkeypatch.setattr(memory, "_write_json", fail_manifest)
        with pytest.raises(RuntimeError, match="manifest unavailable"):
            memory.save(checkpoint)

    restored = BranchableMemory.load(checkpoint)
    try:
        assert seen(restored, first) == {1}
    finally:
        restored.close()


def test_concurrent_isolated_branch_merges_and_reads_remain_consistent():
    rng = np.random.default_rng(112)
    base = unit(rng)
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add(0, base)
        errors = []
        start = threading.Barrier(5, timeout=3)

        def writer(worker):
            try:
                start.wait()
                for offset in range(8):
                    branch = memory.branch(f"{worker}-{offset}")
                    branch.add(1000 + worker * 100 + offset, base)
                    branch.merge()
            except BaseException as error:  # surfaced by caller
                errors.append(error)

        def reader():
            try:
                start.wait()
                for _ in range(40):
                    assert 0 in seen(memory, base)
            except BaseException as error:  # surfaced by caller
                errors.append(error)

        threads = [
            threading.Thread(target=writer, args=(worker,), daemon=True) for worker in range(4)
        ]
        threads.append(threading.Thread(target=reader, daemon=True))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        assert not errors
        assert len(seen(memory, base)) == 20  # query returns its requested top-k


def test_randomized_branch_visibility_matches_a_materialized_reference():
    """Exercise updates/deletes in arbitrary order against a simple reference model."""
    rng = np.random.default_rng(113)
    vectors = [unit(rng) for _ in range(96)]
    with BranchableMemory(16, page_capacity=8, nprobe=64) as memory:
        base = {item_id: vectors[item_id] for item_id in range(12)}
        for item_id, vector in base.items():
            memory.add(item_id, vector, generation="base")
        branch = memory.branch("random")
        expected = dict(base)

        for step in range(80):
            item_id = int(rng.integers(0, 20))
            if rng.random() < 0.32:
                branch.delete(item_id)
                expected.pop(item_id, None)
            else:
                vector = vectors[int(rng.integers(12, len(vectors)))]
                branch.add(item_id, vector, generation=step)
                expected[item_id] = vector

            query = vectors[int(rng.integers(0, len(vectors)))]
            actual = {hit.id for hit, _ in branch.search(query, k=64, nprobe=64)}
            assert actual == set(expected), f"visibility diverged at randomized step {step}"
            assert {hit.id for hit, _ in memory.search(query, k=64, nprobe=64)} == set(base)


def test_restored_branch_keeps_conflict_fence_and_multi_branch_isolation(tmp_path):
    rng = np.random.default_rng(114)
    checkpoint = tmp_path / "branchable"
    original, main_value, left_value, right_value = [unit(rng) for _ in range(4)]
    with BranchableMemory(16, nprobe=64) as memory:
        memory.add("shared", original)
        left = memory.branch("left")
        right = memory.branch("right")
        left.add("shared", left_value)
        right.add("right-only", right_value)
        memory.add("shared", main_value)
        memory.save(checkpoint)

    restored = BranchableMemory.load(checkpoint)
    try:
        left = restored.get_branch("left")
        right = restored.get_branch("right")
        assert left.conflicts() == ("shared",)
        with pytest.raises(BranchConflictError):
            left.merge()
        assert {record.id for _, record in right.search(right_value, k=20)} == {
            "shared",
            "right-only",
        }
        right.merge()
        assert {record.id for _, record in restored.search(right_value, k=20)} == {
            "shared",
            "right-only",
        }
    finally:
        restored.close()


def _capture(errors, operation):
    try:
        operation()
    except BaseException as error:  # surfaced by caller
        errors.append(error)
