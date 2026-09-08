import json
import threading
from typing import Any

import numpy as np
import pytest

from chronovec.memory import MAX_BRANCHES, AgentMemory, BranchError


def unit(rng, d=32):
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


def test_branch_writes_are_invisible_to_main_and_siblings():
    rng = np.random.default_rng(0)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base, text="fact")
        a = memory.branch("a")
        b = memory.branch("b")
        a.add(2, base, text="a-speculation")
        b.add(3, base, text="b-speculation")

        seen = lambda view: {h.id for h, _ in view.search(base, k=10)}
        assert seen(memory.main) == {1}, "main must not see branch writes"
        assert seen(a) == {1, 2}, "a sees base plus its own"
        assert seen(b) == {1, 3}, "b sees base plus its own, not a's"


def test_discard_removes_branch_writes():
    rng = np.random.default_rng(1)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("explore")
        for i in range(2, 12):
            branch.add(i, unit(rng))
        assert len({h.id for h, _ in branch.search(base, k=20)}) > 1
        branch.discard()
        assert {h.id for h, _ in memory.search(base, k=20)} == {1}
        with pytest.raises(BranchError):
            branch.add(99, base)


def test_purge_reclaims_native_versions_at_safe_horizon():
    rng = np.random.default_rng(12)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        vector = unit(rng)
        before = memory.add(1, vector, text="temporary")
        memory.delete(1)
        retained_before = memory._index.stats()["retained_versions"]

        assert memory.purge(oldest_snapshot=memory.snapshot() + 1) == 1
        assert memory._index.stats()["retained_versions"] < retained_before
        assert memory.search(vector, k=1) == []
        assert memory.search(vector, k=1, as_of=before) == []


def test_purge_protects_active_branch_fork_point():
    rng = np.random.default_rng(13)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        vector = unit(rng)
        memory.add(1, vector, text="before branch")
        branch = memory.branch("retry")
        memory.add(1, vector, text="after branch")

        with pytest.raises(ValueError, match="active branch fork point"):
            memory.purge()

        assert branch.search(vector, k=1)[0][1].payload["text"] == "before branch"
        assert memory.purge(oldest_snapshot=branch.base_snapshot + 1) == 0


def test_failed_discard_keeps_branch_open_for_retry(monkeypatch):
    rng = np.random.default_rng(11)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        branch = memory.branch("explore")
        branch.add(1, unit(rng))

        def fail(_view):
            raise RuntimeError("native cleanup unavailable")

        monkeypatch.setattr(memory, "_discard", fail)
        with pytest.raises(RuntimeError, match="native cleanup unavailable"):
            branch.discard()
        assert branch.search(unit(rng), k=1)


def test_checkpoint_failure_after_discard_release_closes_view(monkeypatch):
    with AgentMemory(8, checkpoint_path=None, nprobe=64, branch_engine="labels") as memory:
        branch = memory.branch("discard-me")
        branch.add(1, unit(np.random.default_rng(111), d=8))

        def fail_checkpoint():
            raise RuntimeError("checkpoint unavailable")

        monkeypatch.setattr(memory, "_checkpoint", fail_checkpoint)
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            branch.discard()
        with pytest.raises(BranchError):
            branch.search(unit(np.random.default_rng(112), d=8), k=1)


def test_checkpoint_failure_after_merge_release_closes_view(monkeypatch):
    with AgentMemory(8, checkpoint_path=None, nprobe=64, branch_engine="labels") as memory:
        branch = memory.branch("merge-me")
        branch.add(1, unit(np.random.default_rng(113), d=8))

        def fail_checkpoint():
            raise RuntimeError("checkpoint unavailable")

        monkeypatch.setattr(memory, "_checkpoint", fail_checkpoint)
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            branch.merge()
        with pytest.raises(BranchError):
            branch.search(unit(np.random.default_rng(114), d=8), k=1)


def test_merge_promotes_writes_to_main():
    rng = np.random.default_rng(2)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("keep")
        branch.add(2, base, text="worth keeping")
        assert {h.id for h, _ in memory.search(base, k=10)} == {1}
        assert branch.merge() == 1
        promoted = {h.id for h, _ in memory.search(base, k=10)}
        assert promoted == {1, 2}, "merged writes must be visible on main"


def test_merge_is_atomic_to_high_level_readers(monkeypatch):
    rng = np.random.default_rng(21)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("atomic-merge")
        branch.add(2, base)
        branch.add(3, base)

        entered = threading.Event()
        release = threading.Event()
        original_delete = memory._delete_engine

        def pause_before_delete(engine_id):
            entered.set()
            assert release.wait(2), "merge did not resume in time"
            original_delete(engine_id)

        monkeypatch.setattr(memory, "_delete_engine", pause_before_delete)
        merge_errors = []
        merge_thread = threading.Thread(
            target=lambda: _run_and_capture(merge_errors, branch.merge), daemon=True
        )
        merge_thread.start()
        assert entered.wait(2), "merge did not reach its first promotion"

        observed = []
        reader = threading.Thread(
            target=lambda: observed.append({h.id for h, _ in memory.search(base, k=10)}),
            daemon=True,
        )
        reader.start()
        reader.join(timeout=0.05)
        assert reader.is_alive(), "reader should wait while a merge is in progress"

        release.set()
        merge_thread.join(timeout=2)
        reader.join(timeout=2)
        assert not merge_errors
        assert observed == [{1, 2, 3}]


def _run_and_capture(errors, operation):
    try:
        operation()
    except BaseException as error:  # surfaced by the assertion in the caller
        errors.append(error)


def test_branch_can_read_its_fork_point():
    rng = np.random.default_rng(3)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        base = unit(rng)
        memory.add(1, base, text="original")
        branch = memory.branch("t")
        memory.delete(1)
        assert {h.id for h, _ in memory.search(base, k=10)} == set()
        assert {h.id for h, _ in branch.as_of_fork(base, k=10)} == {1}


def test_branch_delete_does_not_affect_main():
    rng = np.random.default_rng(30)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("retry")
        branch.delete(1)

        assert {h.id for h, _ in memory.search(base, k=10)} == {1}
        assert {h.id for h, _ in branch.search(base, k=10)} == set()


def test_branch_from_historical_snapshot_excludes_future_main_writes():
    rng = np.random.default_rng(31)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        checkpoint = memory.snapshot()
        memory.add(2, base)
        branch = memory.branch("retry", snapshot=checkpoint)

        assert {h.id for h, _ in branch.search(base, k=10)} == {1}
        assert {h.id for h, _ in memory.search(base, k=10)} == {1, 2}


def test_branch_can_start_from_the_initial_empty_snapshot():
    rng = np.random.default_rng(311)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("empty-retry", snapshot=0)
        branch.add(2, base)

        assert {h.id for h, _ in branch.search(base, k=10)} == {2}
        assert {h.id for h, _ in memory.search(base, k=10)} == {1}


def test_branch_overlay_is_visible_on_historical_base():
    rng = np.random.default_rng(32)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        checkpoint = memory.snapshot()
        memory.add(2, base)
        branch = memory.branch("retry", snapshot=checkpoint)
        branch.add(3, base)

        assert {h.id for h, _ in branch.search(base, k=10)} == {1, 3}


def test_branch_update_does_not_replace_main_version():
    rng = np.random.default_rng(33)
    with AgentMemory(32, nprobe=64) as memory:
        original = unit(rng)
        revised = unit(rng)
        memory.add(1, original, text="original")
        branch = memory.branch("retry")
        branch.add(1, revised, text="revised")

        main_hit, main_record = memory.search(original, k=1)[0]
        branch_hit, branch_record = branch.search(revised, k=1)[0]
        assert main_hit.id == branch_hit.id == 1
        assert main_record.payload["text"] == "original"
        assert branch_record.payload["text"] == "revised"


def test_branch_merge_promotes_update_and_deletion():
    rng = np.random.default_rng(34)
    with AgentMemory(32, nprobe=64) as memory:
        original = unit(rng)
        revised = unit(rng)
        memory.add(1, original, text="original")
        memory.add(2, original, text="to remove")
        branch = memory.branch("retry")
        branch.add(1, revised, text="revised")
        branch.delete(2)
        assert branch.merge() == 1

        assert memory.search(revised, k=1)[0][1].payload["text"] == "revised"
        assert {h.id for h, _ in memory.search(original, k=10)} == {1}


def test_deleted_branch_overlay_does_not_leak_after_merge():
    rng = np.random.default_rng(35)
    with AgentMemory(32, nprobe=64) as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("retry")
        branch.add(2, base)
        branch.delete(2)
        branch.merge()

        assert {h.id for h, _ in memory.search(base, k=10)} == {1}


def test_failed_merge_keeps_branch_open_for_retry(monkeypatch):
    rng = np.random.default_rng(351)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        base = unit(rng)
        memory.add(1, base)
        branch = memory.branch("retry")
        branch.add(2, base)

        def fail(_view):
            raise RuntimeError("checkpoint unavailable")

        monkeypatch.setattr(memory, "_merge", fail)
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            branch.merge()
        assert branch.search(base, k=10)


def test_checkpoint_restores_payloads_and_active_branch(tmp_path):
    rng = np.random.default_rng(36)
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        base = unit(rng)
        memory.add(1, base, text="base")
        checkpoint = memory.snapshot()
        branch = memory.branch("retry", snapshot=checkpoint)
        branch.add(2, base, text="retry")
        memory.save(tmp_path / "agent")
        assert (tmp_path / "agent.manifest").exists()

    restored = AgentMemory.load(tmp_path / "agent")
    try:
        assert restored.snapshot() == 2
        assert {h.id for h, _ in restored.search(base, k=10)} == {1}
        branch = restored._branches["retry"]
        assert {h.id for h, _ in branch.search(base, k=10)} == {1, 2}
        assert branch.search(base, k=10)[-1][1].payload["text"] == "retry"
    finally:
        restored.close()


def test_checkpoint_bad_interaction_rewind_retry_merge_end_to_end(tmp_path):
    rng = np.random.default_rng(362)
    vector = unit(rng)
    checkpoint = tmp_path / "before-interaction"

    with AgentMemory(32, nprobe=64) as memory:
        memory.add("preference", vector, text="dark mode")
        before = memory.snapshot()
        memory.save(checkpoint)

        # The live timeline receives a bad tool result after the checkpoint.
        memory.add("preference", vector, text="light mode (bad tool result)")

        # Rewind from the clean interaction boundary and retry in isolation.
        retry = memory.branch("retry", snapshot=before)
        retry.add("preference", vector, text="dark mode (corrected retry)")
        assert retry.search(vector, k=1)[0][1].payload["text"] == "dark mode (corrected retry)"
        assert memory.search(vector, k=1)[0][1].payload["text"] == "light mode (bad tool result)"

        retry.merge()
        assert memory.search(vector, k=1)[0][1].payload["text"] == "dark mode (corrected retry)"
        assert memory.search(vector, k=1, as_of=before)[0][1].payload["text"] == "dark mode"

    restored = AgentMemory.load(checkpoint)
    try:
        assert restored.search(vector, k=1)[0][1].payload["text"] == "dark mode"
    finally:
        restored.close()


def test_checkpoint_persists_discarded_branch_lifecycle(tmp_path):
    rng = np.random.default_rng(361)
    checkpoint = tmp_path / "agent"
    with AgentMemory(32, checkpoint_path=checkpoint, nprobe=64) as memory:
        branch = memory.branch("discard-me")
        branch.add(1, unit(rng), text="speculation")
        branch.discard()

    restored = AgentMemory.load(checkpoint)
    try:
        assert list(restored.branches()) == []
        assert restored.search(unit(rng), k=10) == []
    finally:
        restored.close()


def test_checkpoint_rejects_manifest_with_mismatched_native_generation(tmp_path):
    with AgentMemory(32, branch_engine="labels") as memory:
        memory.add(1, unit(np.random.default_rng(37)), text="safe")
        memory.save(tmp_path / "agent")

    manifest_path = tmp_path / "agent.manifest"
    manifest = json.loads(manifest_path.read_text())
    manifest["native_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match its manifest"):
        AgentMemory.load(tmp_path / "agent")


def test_checkpoint_does_not_accumulate_superseded_generations(tmp_path):
    rng = np.random.default_rng(362)
    checkpoint = tmp_path / "agent"
    with AgentMemory(16, checkpoint_path=checkpoint, nprobe=8, branch_engine="labels") as memory:
        for i in range(10):
            memory.add(i, unit(rng, d=16), text=f"item-{i}")

    files = sorted(tmp_path.iterdir())
    assert len(files) == 3, (
        f"only the current generation and its manifest should remain, found {[f.name for f in files]}"
    )
    assert checkpoint.with_suffix(".manifest") in files

    restored = AgentMemory.load(checkpoint)
    try:
        assert len(restored._records) == 10
    finally:
        restored.close()


def test_concurrent_save_and_mutation_does_not_race(tmp_path):
    rng = np.random.default_rng(363)
    with AgentMemory(16, nprobe=8, branch_engine="labels") as memory:
        errors: list[BaseException] = []
        stop = threading.Event()

        def writer() -> None:
            i = 0
            while not stop.is_set():
                try:
                    memory.add(i, unit(rng, d=16), text=f"item-{i}")
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                i += 1

        def saver() -> None:
            i = 0
            while not stop.is_set():
                try:
                    memory.save(tmp_path / f"snapshot-{i % 3}")
                except BaseException as error:  # noqa: BLE001
                    errors.append(error)
                i += 1

        threads = [
            threading.Thread(target=writer, daemon=True),
            threading.Thread(target=saver, daemon=True),
            threading.Thread(target=saver, daemon=True),
        ]
        for t in threads:
            t.start()
        stop.wait(timeout=0.5)
        stop.set()
        for t in threads:
            t.join(timeout=3)

        assert not errors, f"concurrent add()/save() must not race: {errors}"


def test_concurrent_reads_are_not_serialized():
    """Two AgentMemory.search() calls must be able to run concurrently.

    A single exclusive lock across every operation once made this deadlock
    outright: thread A would enter the native call and block on the barrier
    waiting for thread B, but thread B could never even reach the barrier
    because it was stuck acquiring the same lock thread A still held.
    """
    rng = np.random.default_rng(40)
    with AgentMemory(16, nprobe=8, branch_engine="labels") as memory:
        memory.add(1, unit(rng, d=16))
        original_search = memory._index.search
        barrier = threading.Barrier(2, timeout=2)

        def synchronized_search(*args: Any, **kwargs: Any) -> Any:
            barrier.wait()
            return original_search(*args, **kwargs)

        memory._index.search = synchronized_search  # type: ignore[method-assign]

        errors: list[BaseException] = []
        results: list[Any] = []

        def run() -> None:
            try:
                results.append(memory.search(unit(np.random.default_rng(41), d=16), k=1))
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=run, daemon=True) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3)

        assert not errors, f"concurrent reads must overlap, not deadlock: {errors}"
        assert len(results) == 2


def test_historical_branch_search_fills_k_despite_branch_deletes():
    """Deleting a few candidates on a branch must not starve the result below k.

    Each side of a historical-branch search fetches only a shortlist before
    filtering out branch-local deletes, so the shortlist must be padded by
    the number of deletions or the filtered result silently falls short of
    k even when enough real candidates exist.
    """
    rng = np.random.default_rng(364)
    base = unit(rng)

    def near(scale: float):
        v = base + rng.standard_normal(32).astype(np.float32) * scale
        return v / np.linalg.norm(v)

    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        for i in range(15):
            memory.add(i, near(0.01 * (i + 1)))
        checkpoint = memory.snapshot()
        branch = memory.branch("retry", snapshot=checkpoint)
        for i in (0, 1, 2):
            branch.delete(i)

        results = branch.search(base, k=10)
        assert len(results) == 10, (
            f"12 valid candidates remain after 3 branch deletes; requested k=10, got {len(results)}"
        )
        assert {0, 1, 2}.isdisjoint(h.id for h, _ in results)


def test_payload_round_trips():
    rng = np.random.default_rng(4)
    with AgentMemory(32, nprobe=64) as memory:
        v = unit(rng)
        memory.add(7, v, text="hello", source="unit-test")
        ((_, record),) = memory.search(v, k=1)
        assert record.payload == {"text": "hello", "source": "unit-test"}
        assert record.branch == "main"


def test_branch_bits_are_recycled_and_bounded():
    with AgentMemory(32, nprobe=64, branch_engine="labels") as memory:
        for i in range(MAX_BRANCHES):
            memory.branch(f"b{i}")
        with pytest.raises(BranchError, match="branch labels are in use"):
            memory.branch("one-too-many")
        memory._branches["b0"].discard()
        memory.branch("recycled")


def test_duplicate_branch_name_rejected():
    with AgentMemory(32, nprobe=64) as memory:
        memory.branch("x")
        with pytest.raises(BranchError):
            memory.branch("x")
        with pytest.raises(BranchError):
            memory.branch("main")
