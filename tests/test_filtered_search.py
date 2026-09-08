"""Filtered search must answer the filtered question, not a cheaper one.

These cover a failure that the rest of the suite could not see: a filtered
search that probes a fixed number of pages returns whatever happened to match
inside them. That looks like a *fast* query rather than a wrong one, so
latency-based tests and unfiltered recall tests both pass while the answer is
badly incomplete. On SIFT at 1% selectivity it returned 1.6 of 10 requested
neighbours and no test noticed.

The layout of the filtered attribute is the axis that exposes it, so it is a
parameter here rather than a fixed choice. A scattered attribute leaves every
page holding a mixture, so the nearest pages contain matches and a fixed probe
set gets away with it. An attribute that correlates with position in the
embedding space -- a tenant, a language, a product line, which is what people
actually filter on -- puts every match in a region the query's own
neighbourhood need not reach.
"""

from __future__ import annotations

import numpy as np
import pytest

from chronovec import Index

DIM = 32
COUNT = 6000
MATCH, OTHER = 1, 2


def _corpus(seed: int = 0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((COUNT, DIM), dtype=np.float32), np.arange(COUNT, dtype=np.int64))


def _tags(vectors: np.ndarray, fraction: float, layout: str, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    tags = np.full(vectors.shape[0], OTHER, dtype=np.uint64)
    marked = max(1, int(vectors.shape[0] * fraction))
    if layout == "semantic":
        anchor = vectors[rng.integers(0, vectors.shape[0])]
        gap = np.linalg.norm(vectors - anchor, axis=1)
        chosen = np.argpartition(gap, marked - 1)[:marked]
    else:
        chosen = rng.choice(vectors.shape[0], size=marked, replace=False)
    tags[chosen] = MATCH
    return tags


def _truth(vectors, ids, tags, query, k):
    keep = tags == MATCH
    matching, keys = vectors[keep], ids[keep]
    order = np.argsort(np.linalg.norm(matching - query, axis=1))[:k]
    return set(keys[order].tolist())


@pytest.mark.parametrize("layout", ["random", "semantic"])
@pytest.mark.parametrize("fraction", [0.5, 0.1, 0.01])
def test_filtered_search_fills_k(layout, fraction):
    """k results whenever k matching records exist, at any probe width.

    The probe width is deliberately far too small for the corpus. A filtered
    search may not quietly return fewer results because it stopped looking:
    when the filter matches more than k records, the caller asked for k and
    there are k to give.
    """
    vectors, ids = _corpus()
    tags = _tags(vectors, fraction, layout)
    assert int((tags == MATCH).sum()) > 10
    index = Index(DIM, metric="l2", nprobe=4, labels=True)
    index.insert_many(ids, vectors, labels=tags)
    try:
        for query in vectors[:25]:
            found = index.search(query, k=10, nprobe=4, require_all=MATCH)
            assert len(found) == 10, f"{layout} at {fraction:.0%} returned {len(found)} of 10"
            assert all(tags[r.id] == MATCH for r in found)
    finally:
        index.close()


@pytest.mark.parametrize("layout", ["random", "semantic"])
def test_filtered_recall_is_reached_by_probing_wider(layout):
    """Recall has to be reachable, and reachable by the documented knob.

    A filter shrinks the pool a fixed probe width draws from, so filtered
    recall at a given nprobe is below unfiltered recall at the same nprobe.
    That is expected and is why the benchmark matches recall rather than
    nprobe. What is not acceptable is a ceiling: if widening the probe set
    stops improving the answer, no amount of tuning gets a caller a correct
    one.
    """
    vectors, ids = _corpus(seed=3)
    tags = _tags(vectors, 0.1, layout, seed=3)
    index = Index(DIM, metric="l2", nprobe=8, labels=True)
    index.insert_many(ids, vectors, labels=tags)
    try:
        queries = vectors[:30]
        best = 0.0
        for probes in (8, 64, 512):
            hits = 0.0
            for query in queries:
                found = {r.id for r in index.search(query, k=10, nprobe=probes, require_all=MATCH)}
                hits += len(found & _truth(vectors, ids, tags, query, 10)) / 10
            best = max(best, hits / len(queries))
        assert best >= 0.95, f"{layout} plateaued at recall {best:.3f}"
    finally:
        index.close()


def test_filter_never_returns_a_non_matching_record():
    """The filter is a guarantee, not a ranking hint."""
    vectors, ids = _corpus(seed=7)
    tags = _tags(vectors, 0.05, "semantic", seed=7)
    index = Index(DIM, metric="l2", nprobe=64, labels=True)
    index.insert_many(ids, vectors, labels=tags)
    try:
        for query in vectors[:40]:
            for record in index.search(query, k=10, nprobe=64, require_all=MATCH):
                assert tags[record.id] == MATCH
    finally:
        index.close()


def test_fewer_matches_than_k_returns_what_exists():
    """Short results are correct when the filter genuinely matches few rows.

    The paired case to `test_filtered_search_fills_k`: widening the probe set
    to satisfy a filter must stop at the directory rather than spin when the
    answer really is smaller than k.
    """
    vectors, ids = _corpus(seed=11)
    tags = np.full(COUNT, OTHER, dtype=np.uint64)
    tags[[3, 17, 400]] = MATCH
    index = Index(DIM, metric="l2", nprobe=4, labels=True)
    index.insert_many(ids, vectors, labels=tags)
    try:
        found = index.search(vectors[0], k=10, nprobe=4, require_all=MATCH)
        assert {r.id for r in found} == {3, 17, 400}
    finally:
        index.close()


def test_filtered_search_survives_churn():
    """The filter must still hold after the pages have been reorganised.

    Page-level skipping depends on records sharing a tag staying grouped, and
    mutation is what disturbs that grouping. An advantage measured on a
    freshly built index is not evidence about a live one.
    """
    vectors, ids = _corpus(seed=13)
    tags = _tags(vectors, 0.1, "semantic", seed=13)
    index = Index(DIM, metric="l2", nprobe=32, labels=True)
    index.insert_many(ids, vectors, labels=tags)
    rng = np.random.default_rng(23)
    live_tags = tags.copy()
    live = np.ones(COUNT, dtype=bool)
    try:
        for round_number in range(3):
            present = np.flatnonzero(live)
            retire = rng.choice(present, size=present.shape[0] // 4, replace=False)
            for one in retire:
                index.delete(int(one))
            live[retire] = False
            live_tags = live_tags.copy()
            live_tags[retire] = 0
            for query in vectors[:20]:
                found = index.search(query, k=10, nprobe=32, require_all=MATCH)
                assert all(live_tags[r.id] == MATCH for r in found), (
                    f"round {round_number} returned a retired or non-matching record"
                )
    finally:
        index.close()


# --- label partitioning -------------------------------------------------
#
# Keeping one label to a page is what turns the label union from a test that
# rarely fires into one that rules out most of the directory. The properties
# below are what the mode promises; the reason it is opt-in is that it buys
# them by spending pages and giving up placement freedom.


def _partitioned(vectors, ids, tags, **kwargs):
    index = Index(DIM, metric="l2", labels=True, label_partition=True, **kwargs)
    index.insert_many(ids, vectors, labels=tags)
    return index


@pytest.mark.parametrize("layout", ["random", "semantic"])
def test_partitioning_leaves_no_page_holding_two_labels(layout):
    """The invariant the whole mode rests on.

    A single mixed page is enough to make the union test fail for every query
    that touches it, and nothing else in the system notices: the search still
    returns the right records, just after reading pages it should have skipped.
    """
    vectors, ids = _corpus(seed=31)
    tags = _tags(vectors, 0.1, layout, seed=31)
    index = _partitioned(vectors, ids, tags)
    try:
        live, matching = index.page_label_profile(require_all=MATCH)
        mixed = ((matching > 0) & (matching < live)).sum()
        assert mixed == 0, f"{mixed} pages hold both labels"
    finally:
        index.close()


def test_partitioning_survives_churn_and_vacuum():
    """Merging is where purity would be lost, and lost silently.

    Consolidation groups pages by distance. Left to do that under
    partitioning it would join pages carrying different labels and undo the
    placement rule at the first vacuum, while every query kept returning
    correct results.
    """
    vectors, ids = _corpus(seed=37)
    tags = _tags(vectors, 0.1, "semantic", seed=37)
    index = _partitioned(vectors, ids, tags)
    rng = np.random.default_rng(41)
    live_ids = ids.copy()
    nxt = COUNT
    try:
        for round_number in range(3):
            retire = rng.choice(COUNT, size=COUNT // 4, replace=False)
            index.delete_many(live_ids[retire].astype(np.int64))
            fresh_ids = np.arange(nxt, nxt + retire.shape[0], dtype=np.int64)
            fresh = vectors[rng.choice(COUNT, size=retire.shape[0])]
            fresh_tags = np.full(retire.shape[0], OTHER, dtype=np.uint64)
            fresh_tags[rng.random(retire.shape[0]) < 0.1] = MATCH
            index.insert_many(fresh_ids, fresh, labels=fresh_tags)
            live_ids = live_ids.copy()
            live_ids[retire] = fresh_ids
            nxt += retire.shape[0]
            index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
            live, matching = index.page_label_profile(require_all=MATCH)
            mixed = ((matching > 0) & (matching < live)).sum()
            assert mixed == 0, f"round {round_number} left {mixed} pages holding both labels"
    finally:
        index.close()


def test_partitioning_narrows_the_pages_a_filter_must_visit():
    """The payoff, stated as a measurement rather than assumed.

    Without it a selective label is spread thinly: at 1% of a 500k corpus its
    records occupied 485 pages where 20 would hold them. The assertion is
    deliberately loose -- the point is the order of magnitude, not a number
    that would need revisiting whenever packing changes.
    """
    vectors, ids = _corpus(seed=43)
    tags = _tags(vectors, 0.05, "random", seed=43)
    spread = Index(DIM, metric="l2", labels=True)
    spread.insert_many(ids, vectors, labels=tags)
    packed = _partitioned(vectors, ids, tags)
    try:
        _, loose = spread.page_label_profile(require_all=MATCH)
        _, tight = packed.page_label_profile(require_all=MATCH)
        assert (tight > 0).sum() * 2 <= (loose > 0).sum(), (
            f"partitioned visits {(tight > 0).sum()} pages, unpartitioned {(loose > 0).sum()}"
        )
    finally:
        spread.close()
        packed.close()


def test_partitioning_is_off_unless_asked_for():
    """Default placement is unchanged, which is what makes this safe to add."""
    vectors, ids = _corpus(seed=47)
    tags = _tags(vectors, 0.1, "semantic", seed=47)
    index = Index(DIM, metric="l2", labels=True)
    index.insert_many(ids, vectors, labels=tags)
    try:
        live, matching = index.page_label_profile(require_all=MATCH)
        assert ((matching > 0) & (matching < live)).sum() > 0
    finally:
        index.close()


def test_partitioned_search_is_still_correct():
    """Purity is a performance property and must not become a correctness one."""
    vectors, ids = _corpus(seed=53)
    tags = _tags(vectors, 0.05, "semantic", seed=53)
    index = _partitioned(vectors, ids, tags)
    try:
        hits = 0.0
        queries = vectors[:30]
        for query in queries:
            found = {r.id for r in index.search(query, k=10, nprobe=256, require_all=MATCH)}
            assert all(tags[one] == MATCH for one in found)
            assert len(found) == 10
            hits += len(found & _truth(vectors, ids, tags, query, 10)) / 10
        assert hits / len(queries) >= 0.95
    finally:
        index.close()
