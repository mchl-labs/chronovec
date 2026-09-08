"""Run a LangGraph-style retrieval experiment with ChronoVec branches.

Run with the real LangGraph integration:

    pip install "chronovec[langgraph]"
    python examples/langgraph_branching_memory.py

The embedding is deterministic and no model key is required. Each candidate
trajectory gets an isolated ChronoVec memory branch. The graph writes and
searches that branch, evaluates the retrieved policy evidence, then merges the
winner and discards the losers. The in-memory LangGraph checkpointer is for the
demo only; production resumes need a durable LangGraph checkpointer and a
durable ChronoVec memory. If LangGraph is not installed, the same workflow runs
with the offline runner below so the example remains executable from a source
checkout with only ChronoVec installed.
"""

from __future__ import annotations

import zlib
from typing import TypedDict

import numpy as np

from chronovec import AgentMemory
from chronovec.integrations import LangGraphMemory

WIDTH = 256


def embed(text: str) -> np.ndarray:
    """Create a deterministic stand-in embedding for the demo."""
    vector = np.zeros(WIDTH, dtype=np.float32)
    for word in text.lower().split():
        vector[zlib.crc32(word.encode()) % WIDTH] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


class CandidateState(TypedDict):
    branch: str
    policy: str
    retrieved: str
    evidence: list[str]
    accepted: bool


class BranchRunner:
    """Connect graph nodes to one ChronoVec branch per graph run."""

    def __init__(self, memory: AgentMemory) -> None:
        self.memory = memory
        self.branches = LangGraphMemory(memory)

    def write_candidate(self, state: CandidateState) -> CandidateState:
        branch = self.branches.open(state["branch"])
        existing = branch.search(embed(state["policy"]), k=10)
        if any(
            record.payload.get("source") == "candidate"
            and record.payload.get("candidate_id") == state["branch"]
            for _, record in existing
        ):
            return state
        branch.add(
            "refund-policy",
            embed(state["policy"]),
            text=state["policy"],
            source="candidate",
            candidate_id=state["branch"],
        )
        return state

    def retrieve_and_evaluate(self, state: CandidateState) -> CandidateState:
        branch = self.branches.get(state["branch"])
        hits = branch.search(embed("authorized automatic refund limit order verification"), k=5)
        retrieved = hits[0][1].payload["text"] if hits else "(nothing)"
        evidence = [record.payload.get("text", "") for _, record in hits]
        joined = " ".join(evidence).lower()
        candidate_evidence = next(
            (
                record.payload.get("text", "").lower()
                for _, record in hits
                if record.payload.get("source") == "candidate"
            ),
            "",
        )
        # This stands in for an application evaluator or LLM judge. It is
        # deterministic, but its decision depends on evidence retrieved from
        # this branch: the authorized policy, the business target, and the
        # candidate policy must all be present and mutually consistent.
        accepted = (
            "authorized automatic refund policy" in joined
            and "business target" in joined
            and "under 100 euros" in candidate_evidence
            and "order verification" in candidate_evidence
            and "all refunds" not in candidate_evidence
        )
        return {**state, "retrieved": retrieved, "evidence": evidence, "accepted": accepted}

    def publish_or_discard(self, state: CandidateState) -> CandidateState:
        if state["accepted"]:
            self.branches.merge(state["branch"])
        else:
            self.branches.discard(state["branch"])
        return state


def build_graph(runner: BranchRunner):
    """Build a LangGraph workflow without making it a package dependency."""
    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        return None

    graph = StateGraph(CandidateState)
    graph.add_node("write_candidate", runner.write_candidate)
    graph.add_node("retrieve_and_evaluate", runner.retrieve_and_evaluate)
    graph.add_node("publish", runner.publish_or_discard)
    graph.add_node("discard", runner.publish_or_discard)
    graph.add_edge(START, "write_candidate")
    graph.add_edge("write_candidate", "retrieve_and_evaluate")
    graph.add_conditional_edges(
        "retrieve_and_evaluate",
        lambda state: "publish" if state["accepted"] else "discard",
        {"publish": "publish", "discard": "discard"},
    )
    graph.add_edge("publish", END)
    graph.add_edge("discard", END)
    return graph.compile(checkpointer=InMemorySaver())


def initial_memory() -> AgentMemory:
    memory = AgentMemory(dimensions=WIDTH, metric="cosine", nprobe=64)
    memory.add(
        "refund-policy",
        embed("Refunds require manual review."),
        text="Refunds require manual review.",
        source="production",
    )
    memory.add(
        "authorized-refund-policy",
        embed(
            "Authorized automatic refund policy: approve refunds under 100 euros after order verification."
        ),
        text="Authorized automatic refund policy: approve refunds under 100 euros after order verification.",
        source="policy-control",
    )
    memory.add(
        "refund-business-target",
        embed(
            "Business target: automatically approve eligible refunds through the authorized 100-euro limit."
        ),
        text="Business target: automatically approve eligible refunds through the authorized 100-euro limit.",
        source="product-requirement",
    )
    return memory


CANDIDATES = [
    ("narrow", "Refunds under 50 euros can be approved automatically after order verification."),
    ("correct", "Refunds under 100 euros can be approved automatically after order verification."),
    ("unsafe", "All refunds can be approved automatically without order verification."),
]


def run_offline(runner: BranchRunner) -> list[CandidateState]:
    results = []
    for name, policy in CANDIDATES:
        state: CandidateState = {
            "branch": f"candidate-{name}",
            "policy": policy,
            "retrieved": "",
            "evidence": [],
            "accepted": False,
        }
        state = runner.write_candidate(state)
        state = runner.retrieve_and_evaluate(state)
        state = runner.publish_or_discard(state)
        results.append(state)
    return results


def run_langgraph(graph, runner: BranchRunner) -> list[CandidateState]:
    results = []
    for name, policy in CANDIDATES:
        state: CandidateState = {
            "branch": f"candidate-{name}",
            "policy": policy,
            "retrieved": "",
            "evidence": [],
            "accepted": False,
        }
        # Each candidate has a distinct LangGraph execution. ChronoVec
        # supplies the corresponding semantic-memory branch.
        try:
            results.append(
                graph.invoke(
                    state,
                    config={"configurable": {"thread_id": state["branch"]}},
                )
            )
        except BaseException:
            # A failed graph run must not leave speculative memory behind.
            runner.branches.cleanup(state["branch"])
            raise
    return results


def main() -> None:
    memory = initial_memory()
    runner = BranchRunner(memory)
    graph = build_graph(runner)
    results = run_langgraph(graph, runner) if graph is not None else run_offline(runner)

    print("LangGraph + ChronoVec transactional retrieval demo")
    print(f"  execution backend: {'LangGraph' if graph is not None else 'offline runner'}")
    print("  main before candidates: Refunds require manual review.")
    for result in results:
        outcome = "MERGED" if result["accepted"] else "DISCARDED"
        print(f"  {result['branch']}: {outcome}; branch retrieved: {result['retrieved']}")

    main_hits = memory.search(embed("authorized automatic refund limit order verification"), k=10)
    main_texts = [record.payload.get("text", "") for _, record in main_hits]
    print(f"  main after evaluation: {main_texts[0] if main_texts else '(nothing)'}")
    assert any("under 100" in text for text in main_texts)
    assert not any("under 50" in text for text in main_texts)
    assert not any("All refunds" in text for text in main_texts)
    memory.close()


if __name__ == "__main__":
    main()
