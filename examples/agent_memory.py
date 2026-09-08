"""Show an agent recovering from a bad interaction without polluting memory.

Run: python examples/agent_memory.py

The agent checkpoints its memory before a risky step, writes the mistaken
future to main, then retries from the checkpoint on an isolated branch. The
corrected branch is merged while the original timeline remains queryable.
No API keys or model are required: the deterministic embedder keeps the demo
reproducible and puts the memory lifecycle in view.
"""

from __future__ import annotations

import zlib

import numpy as np

from chronovec import AgentMemory

WIDTH = 256


def embed(text: str) -> np.ndarray:
    """Create a deterministic stand-in embedding for one piece of text."""
    vector = np.zeros(WIDTH, dtype=np.float32)
    for word in text.lower().split():
        vector[zlib.crc32(word.encode()) % WIDTH] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def show(title: str, results) -> None:
    print(f"  {title}")
    for _, record in results:
        print(f"    - {record.id}: {record.payload.get('text', '')}")
    if not results:
        print("    (nothing)")


def main() -> None:
    with AgentMemory(dimensions=WIDTH, metric="cosine", nprobe=64) as memory:
        print("1. The agent records a confirmed preference.\n")
        memory.add(
            "pref-theme",
            embed("the user prefers dark mode interfaces"),
            text="the user prefers dark mode interfaces",
        )
        checkpoint = memory.snapshot()

        print("2. A risky interaction writes a mistaken future.\n")
        memory.add(
            "pref-theme",
            embed("the user prefers light mode interfaces"),
            text="the user prefers light mode interfaces, mistaken update",
        )
        show("main memory after the mistaken update:", memory.search(embed("mode preference"), k=2))

        print("\n3. Rewind to the checkpoint and retry in isolation.\n")
        retry = memory.branch("retry", snapshot=checkpoint)
        retry.add(
            "pref-theme",
            embed("the user prefers dark mode interfaces"),
            text="the user prefers dark mode interfaces, corrected after retry",
        )
        show("retry branch sees the corrected future:", retry.search(embed("mode preference"), k=2))
        show("main remains on the original timeline:", memory.search(embed("mode preference"), k=2))

        print("\n4. Merge the successful retry; the old timeline remains readable.\n")
        retry.merge()
        show("main after merging the retry:", memory.search(embed("mode preference"), k=2))
        show(
            "memory at the pre-interaction checkpoint:",
            memory.search(embed("mode preference"), k=2, as_of=checkpoint),
        )


if __name__ == "__main__":
    main()
