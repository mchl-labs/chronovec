"""Validate a retrieval correction before atomically publishing it.

Run with:

    python examples/retrieval_change_validation.py

The main corpus continues answering production queries while the candidate
policy is indexed, tested, and then promoted from an isolated ANN branch.
"""

from __future__ import annotations

import numpy as np

from chronovec import AgentMemory


def vector(*values: float) -> np.ndarray:
    value = np.asarray(values, dtype=np.float32)
    return value / np.linalg.norm(value)


def answer(view, query: np.ndarray) -> str:
    _, record = view.search(query, k=1)[0]
    return record.payload["text"]


def main() -> None:
    refund_query = vector(1.0, 0.0, 0.0, 0.0)
    shipping_query = vector(0.0, 1.0, 0.0, 0.0)
    with AgentMemory(4, nprobe=16) as memory:
        memory.add(
            "refund-policy",
            refund_query,
            text="Refunds are available within 14 days.",
            revision="v1",
        )
        memory.add(
            "shipping-policy",
            shipping_query,
            text="Standard shipping takes 3-5 days.",
            revision="v1",
        )

        candidate = memory.branch("refund-v2")
        candidate.add(
            "refund-policy",
            refund_query,
            text="Refunds are available within 30 days.",
            revision="v2-candidate",
        )

        # Production remains on v1 while the candidate is evaluated.
        assert answer(memory.main, refund_query).endswith("14 days.")
        assert answer(candidate, refund_query).endswith("30 days.")
        assert answer(candidate, shipping_query).endswith("3-5 days.")
        print("main during validation:", answer(memory.main, refund_query))
        print("candidate evaluation:", answer(candidate, refund_query))

        candidate.merge()
        assert answer(memory.main, refund_query).endswith("30 days.")
        print("main after atomic publication:", answer(memory.main, refund_query))


if __name__ == "__main__":
    main()
