"""Every capability the documentation claims, exercised.

Docs drift away from code silently. Each test here corresponds to a specific
promise made in README.md, DESIGN.md or RESULTS.md, so a claim that stops being
true fails the suite rather than misleading a reader.
"""

import numpy as np
import pytest

import chronovec
from chronovec import Collection, Index


def vectors(n=200, dim=8, seed=1):
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


# -- README: "Query the past" ------------------------------------------------
def test_claim_query_the_past():
    v = vectors(50)
    with Index(8, metric="l2", nprobe=32) as index:
        index.insert(2, v[1])  # a survivor, so "now" has an answer
        stamp = index.insert(1, v[0])
        index.delete(1)
        now = index.search(v[0], 10)
        assert now and now[0].id == 2
        assert index.search(v[0], 10, snapshot=stamp)[0].id == 1


# -- README: "Delete for real, on a budget" ----------------------------------
def test_claim_bounded_reclamation():
    v = vectors(400)
    with Index(8, metric="l2", page_capacity=32, nprobe=32) as index:
        index.insert_many(np.arange(400), v)
        index.delete_many(np.arange(200))
        freed = index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=64)
        # At most the budget, and it actually reclaimed rather than no-oping.
        assert 0 < freed <= 64
        assert index.stats()["live_vectors"] == 200


# -- README: "Survive a crash" ----------------------------------------------
def test_claim_crash_recovery(tmp_path):
    log = tmp_path / "c.wal"
    v = vectors(120)
    index = Index(8, metric="l2", page_capacity=32, nprobe=32, wal_path=log)
    index.insert_many(np.arange(100), v[:100])
    del index  # no clean shutdown
    recovered = Index(8, metric="l2", page_capacity=32, nprobe=32, wal_path=log)
    assert recovered.stats()["live_vectors"] == 100
    recovered.close()


# -- README: a batch is atomic to readers ------------------------------------
def test_claim_batch_atomicity_single_timestamp():
    v = vectors(500)
    with Index(8, metric="l2", page_capacity=32, nprobe=32) as index:
        before = index.clock
        index.insert_many(np.arange(400), v[:400])
        after = index.clock
        # One commit for the call: the clock advances once, not 400 times.
        assert after - before == 1


# -- README: Collection surface ---------------------------------------------
def test_claim_collection_string_ids_metadata_and_filters():
    with Collection(dimensions=4, metric="l2") as c:
        c.add(
            ids=["x", "y"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadatas=[{"lang": "en", "n": 1}, {"lang": "fr", "n": 2}],
        )
        assert {r.id for r in c.query([1, 0, 0, 0], k=2)} == {"x", "y"}
        for where, expect in [
            ({"lang": "en"}, {"x"}),
            ({"n": {"$gte": 2}}, {"y"}),
            ({"$or": [{"lang": "fr"}, {"n": 1}]}, {"x", "y"}),
        ]:
            assert {r.id for r in c.query([1, 0, 0, 0], k=2, where=where)} == expect


def test_claim_collection_snapshot_survives_an_overwrite():
    with Collection(dimensions=4, metric="l2") as c:
        c.add(ids=["x"], embeddings=[[1, 0, 0, 0]], documents=["old"])
        before = c.snapshot()
        c.add(ids=["x"], embeddings=[[0, 1, 0, 0]], documents=["new"])
        assert c.query([1, 0, 0, 0], k=1, snapshot=before)[0].document == "old"
        assert c.query([0, 1, 0, 0], k=1)[0].document == "new"


# -- README: disk backing, both payloads mapped ------------------------------
def test_claim_both_payloads_are_mapped(tmp_path):
    arena = tmp_path / "a.bin"
    with Index(
        16, metric="l2", page_capacity=64, nprobe=32, arena_path=arena, max_vectors=20000
    ) as index:
        index.insert_many(np.arange(1000), vectors(1000, 16))
        assert arena.exists()
        assert (tmp_path / "a.bin.codes").exists()  # screening mapped too


def test_claim_durability_and_mapping_compose(tmp_path):
    index = Index(
        16,
        metric="l2",
        page_capacity=64,
        nprobe=32,
        wal_path=tmp_path / "c.wal",
        arena_path=tmp_path / "a.bin",
        max_vectors=20000,
    )
    index.insert_many(np.arange(500), vectors(500, 16))
    del index
    recovered = Index(
        16,
        metric="l2",
        page_capacity=64,
        nprobe=32,
        wal_path=tmp_path / "c.wal",
        arena_path=tmp_path / "a.bin",
        max_vectors=20000,
    )
    assert recovered.stats()["live_vectors"] == 500
    recovered.close()


# -- DESIGN: bounded space under churn ---------------------------------------
def test_claim_amplification_stays_bounded_under_churn():
    live = 3000
    pool = vectors(live * 3, 8, seed=2)
    with Index(8, metric="l2", page_capacity=32, nprobe=32) as index:
        index.insert_many(np.arange(live), pool[:live])
        ids, nxt, seen = np.arange(live), live, []
        for _ in range(5):
            index.delete_many(ids)
            ids = np.arange(nxt, nxt + live)
            index.insert_many(ids, pool[np.arange(nxt, nxt + live) % len(pool)])
            nxt += live
            index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
            seen.append(index.stats()["capacity_amplification"])
        assert max(seen) < 2.5, seen
        assert seen[-1] <= seen[0] * 1.25, seen  # flat, not merely finite


# -- README: threads help reads, and lanes do not change the answer ----------
def test_claim_lane_count_does_not_change_the_index():
    v = vectors(4000, 8, seed=3)
    shapes = []
    for lanes in (1, 4):
        with Index(8, metric="l2", page_capacity=32, nprobe=32, threads=lanes) as index:
            index.insert_many(np.arange(4000), v)
            stats = index.stats()
            shapes.append((stats["live_vectors"], stats["pages"]))
    assert shapes[0] == shapes[1]


# -- README: the public surface is what the docs name ------------------------
def test_claim_public_api_is_exported():
    for name in (
        "Index",
        "Collection",
        "Record",
        "AgentMemory",
        "SearchResult",
        "exact_search",
        "estimate_memory",
        "stable_id",
        "Client",
        "PersistentClient",
    ):
        assert hasattr(chronovec, name), name


@pytest.mark.parametrize(
    "module,cls",
    [
        ("chronovec.integrations.langchain", "ChronoVecVectorStore"),
        ("chronovec.integrations.llamaindex", "ChronoVecLlamaStore"),
    ],
)
def test_claim_framework_adapters_import(module, cls):
    import importlib

    assert hasattr(importlib.import_module(module), cls)
