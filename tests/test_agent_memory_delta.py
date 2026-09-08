import numpy as np
import pytest

from chronovec import AgentMemory
from chronovec.branchable_memory import BranchConflictError


def unit(rng):
    value = rng.standard_normal(16).astype(np.float32)
    return value / np.linalg.norm(value)


def ids(view, vector):
    return {hit.id for hit, _ in view.search(vector, k=20)}


def test_agent_memory_default_delta_engine_preserves_public_workflow(tmp_path):
    rng = np.random.default_rng(401)
    vector = unit(rng)
    checkpoint = tmp_path / "memory"
    with AgentMemory(16, nprobe=64, checkpoint_path=checkpoint) as memory:
        memory.main.add(1, vector, text="base")
        branch = memory.branch("retry")
        branch.add(2, vector, text="candidate")

        assert ids(memory, vector) == {1}
        assert ids(branch, vector) == {1, 2}
        branch.merge()
        assert ids(memory, vector) == {1, 2}

    restored = AgentMemory.load(checkpoint)
    try:
        assert ids(restored, vector) == {1, 2}
    finally:
        restored.close()


def test_agent_memory_dimensions_is_set_for_the_delta_engine(tmp_path):
    checkpoint = tmp_path / "memory"
    with AgentMemory(16, nprobe=64, checkpoint_path=checkpoint) as memory:
        assert memory.dimensions == 16
        memory.add(1, unit(np.random.default_rng(404)), text="base")

    restored = AgentMemory.load(checkpoint)
    try:
        assert restored.dimensions == 16
    finally:
        restored.close()


def test_agent_memory_default_engine_is_private_delta_while_labels_remain_available():
    with AgentMemory(16, nprobe=64) as memory:
        assert memory._delta_engine is not None
        assert memory._delta_engine._index.labels is False
    with AgentMemory(16, nprobe=64, branch_engine="labels") as memory:
        assert memory._delta_engine is None
        assert memory._index.labels is True


def test_agent_memory_branch_page_capacity_only_sizes_branches_not_main():
    rng = np.random.default_rng(403)
    with AgentMemory(16, nprobe=64, page_capacity=64, branch_page_capacity=8) as memory:
        for i in range(200):
            memory.add(i, unit(rng), text=f"main-{i}")
        main_pages = memory._delta_engine._index.stats()["pages"]

        branch = memory.branch("scratch")
        for i in range(200):
            branch.add(1_000 + i, unit(rng), text=f"branch-{i}")
        branch_pages = memory._delta_engine._branches["scratch"].delta.stats()["pages"]

        # Same record count in both, but main used page_capacity=64 while the
        # branch used branch_page_capacity=8 -- independent settings, not one
        # shared knob.
        assert branch_pages > main_pages * 3


def test_agent_memory_rejects_branch_page_capacity_with_labels_engine():
    with pytest.raises(ValueError, match="branch_page_capacity requires branch_engine='delta'"):
        AgentMemory(16, branch_engine="labels", branch_page_capacity=8)


def test_delta_agent_memory_preserves_legacy_last_writer_wins_merge():
    rng = np.random.default_rng(402)
    first, main_value, branch_value = unit(rng), unit(rng), unit(rng)
    with AgentMemory(16, nprobe=64, branch_engine="delta") as memory:
        memory.add("shared", first, text="base")
        branch = memory.branch("candidate")
        branch.add("shared", branch_value, text="branch")
        memory.add("shared", main_value, text="main")
        branch.merge()
        assert memory.search(branch_value, k=1)[0][1].payload["text"] == "branch"


def test_delta_agent_memory_can_request_conflict_aware_merges():
    rng = np.random.default_rng(403)
    first, main_value, branch_value = unit(rng), unit(rng), unit(rng)
    with AgentMemory(
        16, nprobe=64, branch_engine="delta", branch_merge_strategy="fail_on_conflict"
    ) as memory:
        memory.add("shared", first)
        branch = memory.branch("candidate")
        branch.add("shared", branch_value)
        memory.add("shared", main_value)
        with pytest.raises(BranchConflictError):
            branch.merge()
