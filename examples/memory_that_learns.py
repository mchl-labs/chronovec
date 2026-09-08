"""A small, human-readable agent-memory story with no model or API key.

Run: python examples/memory_that_learns.py

The assistant stores a hypothesis, observes contradictory evidence, and can
compare what it believed before and after correcting itself. The whole demo is
inside main so importing the module has no side effects.
"""

from __future__ import annotations

import zlib

import numpy as np

from chronovec import Collection

WIDTH = 256


def embed(texts: list[str]) -> list[list[float]]:
    """Create deterministic stand-in embeddings for an offline demo."""
    result = []
    for text in texts:
        vector = np.zeros(WIDTH, dtype=np.float32)
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % WIDTH] += 1.0
        norm = np.linalg.norm(vector)
        result.append((vector / norm if norm else vector).tolist())
    return result


def show(label: str, records: list) -> None:
    print(f"  {label}")
    for record in records:
        print(f"    - {record.document} [{record.metadata.get('source', '')}]")
    if not records:
        print("    (nothing)")


def main() -> None:
    memory = Collection(dimensions=WIDTH, embedding_function=embed, name="assistant")
    try:
        print("1. Learn confirmed facts.\n")
        memory.add(
            ids=["language", "editor"],
            documents=["the user writes Python", "the user uses Vim for quick edits"],
            metadatas=[{"source": "observed"}, {"source": "observed"}],
        )
        show("What the assistant knows:", memory.query("user programming editor", k=2))

        print("\n2. Store an uncertain hypothesis.\n")
        memory.add(
            ids=["shell-guess"],
            documents=["hypothesis: the user probably uses zsh"],
            metadatas=[{"source": "hypothesis"}],
        )
        while_hypothesising = memory.snapshot()
        show("What it believes for now:", memory.query("user shell terminal", k=2))

        print("\n3. New evidence corrects the hypothesis.\n")
        memory.delete(where={"source": "hypothesis"})
        memory.add(
            ids=["shell-actual"],
            documents=["the user uses fish shell, confirmed from a pasted script"],
            metadatas=[{"source": "confirmed"}],
        )
        corrected = memory.snapshot()
        show("What it knows now:", memory.query("user shell terminal", k=2))

        print("\n4. Compare the two points in time.\n")
        show(
            f"At snapshot {while_hypothesising}:",
            memory.query("user shell terminal", k=2, snapshot=while_hypothesising),
        )
        show(
            f"At snapshot {corrected}:",
            memory.query("user shell terminal", k=2, snapshot=corrected),
        )

        print("\n5. Reclaim the obsolete hypothesis after releasing the old snapshot.\n")
        freed = memory.vacuum()
        print(f"  reclaimed {freed} obsolete version(s)")
        print("  Done.")
    finally:
        memory.close()


if __name__ == "__main__":
    main()
