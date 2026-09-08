"""Contract tests for the optional LangGraph/ChronoVec integration."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("langgraph")

from chronovec import AgentMemory, BranchError
from chronovec.integrations import LangGraphMemory


def test_thread_branch_is_idempotent_and_lifecycle_is_explicit():
    memory = AgentMemory(3)
    adapter = LangGraphMemory(memory)
    first = adapter.open("run-1")
    assert adapter.open("run-1") is first
    first.add("speculation", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert tuple(adapter.names()) == ("run-1",)
    assert adapter.merge("run-1") == 1
    assert tuple(adapter.names()) == ()
    assert memory.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1)[0][1].id == "speculation"
    with pytest.raises(BranchError, match="no open"):
        adapter.get("run-1")
    memory.close()


def test_adapter_can_reattach_to_an_active_persisted_branch():
    memory = AgentMemory(3)
    original = LangGraphMemory(memory)
    view = original.open("resume-1")
    view.add("speculation", np.array([1.0, 0.0, 0.0], dtype=np.float32))

    recovered = LangGraphMemory(memory)
    resumed = recovered.open("resume-1")
    assert resumed.name == view.name == "resume-1"
    assert resumed.base_snapshot == view.base_snapshot
    assert recovered.merge("resume-1") == 1
    assert tuple(memory.branches()) == ()
    memory.close()


def test_discard_and_close_never_publish_speculation():
    memory = AgentMemory(3)
    adapter = LangGraphMemory(memory)
    adapter.open("discarded").add("bad", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    adapter.open("still-open").add("also-bad", np.array([0.0, 1.0, 0.0], dtype=np.float32))
    adapter.discard("discarded")
    adapter.close()
    assert tuple(memory.branches()) == ()
    assert memory.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=3) == []
    memory.close()


def test_langgraph_demo_uses_real_checkpoints_and_cleans_branches():
    path = Path(__file__).resolve().parent.parent / "examples" / "langgraph_branching_memory.py"
    namespace = runpy.run_path(str(path))
    memory = namespace["initial_memory"]()
    runner = namespace["BranchRunner"](memory)
    graph = namespace["build_graph"](runner)
    assert graph is not None

    state = {
        "branch": "checkpointed-run",
        "policy": "Refunds under 100 euros can be approved automatically after order verification.",
        "retrieved": "",
        "evidence": [],
        "accepted": False,
    }
    config = {"configurable": {"thread_id": state["branch"]}}
    result = graph.invoke(state, config=config)
    history = list(graph.get_state_history(config))

    assert result["accepted"] is True
    assert "under 100" in result["retrieved"]
    assert len(history) >= 4
    assert tuple(runner.branches.names()) == ()
    assert tuple(memory.branches()) == ()
    assert any(
        "under 100" in record.payload["text"]
        for _, record in memory.search(namespace["embed"]("automatic refund for 75 euros"), k=3)
    )
    memory.close()


def test_failed_graph_run_cleans_its_open_thread_branch():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "examples" / "langgraph_branching_memory.py")
    )
    memory = AgentMemory(256)
    runner = namespace["BranchRunner"](memory)

    class FailingGraph:
        def invoke(self, state, config):
            runner.write_candidate(state)
            raise RuntimeError("simulated evaluator failure")

    with pytest.raises(RuntimeError, match="simulated"):
        namespace["run_langgraph"](FailingGraph(), runner)
    assert tuple(runner.branches.names()) == ()
    assert tuple(memory.branches()) == ()
    memory.close()


def test_candidate_evaluation_uses_its_own_branch_evidence():
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parent.parent / "examples" / "langgraph_branching_memory.py")
    )
    memory = namespace["initial_memory"]()
    runner = namespace["BranchRunner"](memory)
    correct = {
        "branch": "correct",
        "policy": "Refunds under 100 euros can be approved automatically after order verification.",
        "retrieved": "",
        "evidence": [],
        "accepted": False,
    }
    unsafe = {
        "branch": "unsafe",
        "policy": "All refunds can be approved automatically without order verification.",
        "retrieved": "",
        "evidence": [],
        "accepted": False,
    }
    runner.write_candidate(correct)
    runner.write_candidate(correct)
    runner.write_candidate(unsafe)
    correct_result = runner.retrieve_and_evaluate(correct)
    unsafe_result = runner.retrieve_and_evaluate(unsafe)

    assert correct_result["accepted"] is True
    assert unsafe_result["accepted"] is False
    assert correct["policy"] in correct_result["evidence"]
    assert unsafe["policy"] not in correct_result["evidence"]
    assert unsafe["policy"] in unsafe_result["evidence"]
    candidate_records = [
        record
        for _, record in runner.branches.get("correct").search(namespace["embed"]("refund"), k=10)
        if record.payload.get("source") == "candidate"
    ]
    assert len(candidate_records) == 1
    runner.branches.discard("correct")
    runner.branches.discard("unsafe")
    memory.close()
