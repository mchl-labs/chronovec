"""Invariants an index must satisfy at rest, and a model to check it against.

Used by the ACID and MVCC suites. The point is to assert properties rather than
outcomes: a test that checks "search returns id 3" passes on an index whose
bookkeeping is quietly wrong, and the bugs found in this project have all been
of that shape -- correct answers over inconsistent state, or state that was
consistent but growing.
"""

from __future__ import annotations

import numpy as np


def check(index, expected_live: int | None = None, *, where: str = "") -> dict:
    """Assert the structural invariants and return the stats that were checked."""
    stats = index.stats()
    tag = f" ({where})" if where else ""

    live = stats["live_vectors"]
    pages = stats["pages"]
    capacity = stats["allocated_slots"]

    if expected_live is not None:
        assert live == expected_live, f"live {live} != expected {expected_live}{tag}"

    # A page holds at most its capacity, and slots are only ever allocated in
    # whole pages, so allocated slots must be a multiple of the page count.
    assert pages >= 1, f"directory is empty{tag}"
    assert capacity >= live, f"allocated {capacity} < live {live}{tag}"
    assert capacity % pages == 0, f"allocated {capacity} is not {pages} whole pages{tag}"

    # Live plus retained must fit in what is allocated: a version that is
    # readable has to occupy a slot.
    retained = stats["retained_versions"]
    assert live + retained <= capacity, (
        f"live {live} + retained {retained} > allocated {capacity}{tag}"
    )

    # Amplification is the ratio the space claim is made in terms of.
    if live:
        expected_amp = capacity / live
        assert abs(stats["capacity_amplification"] - expected_amp) < 0.01, (
            f"amplification {stats['capacity_amplification']} != {expected_amp}{tag}"
        )
    return stats


def live_ids(index, probe: np.ndarray, ceiling: int) -> set[int]:
    """Every id the index will return for a wide search. Small indexes only."""
    return {r.id for r in index.search(probe, ceiling)}


def assert_no_duplicate_ids(index, probe: np.ndarray, ceiling: int, where: str = "") -> None:
    """One live version per id.

    MVCC keeps several versions of a record; at most one may be visible at a
    given snapshot. A duplicate here means two versions share an open interval.
    """
    found = [r.id for r in index.search(probe, ceiling)]
    assert len(found) == len(set(found)), (
        f"duplicate ids returned{' (' + where + ')' if where else ''}: "
        f"{[i for i in found if found.count(i) > 1][:5]}"
    )
