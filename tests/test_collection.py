import numpy as np
import pytest

from chronovec import Collection, Record, stable_id


def make(dimensions=4, **kwargs):
    return Collection(dimensions=dimensions, metric="l2", **kwargs)


def test_string_ids_survive_a_round_trip_and_are_stable():
    # The engine takes int64. A caller with string ids should never see that.
    assert stable_id("doc-a") == stable_id("doc-a")
    assert stable_id("doc-a") != stable_id("doc-b")
    assert stable_id(7) == 7  # ints pass through unhashed
    assert stable_id("doc-a") >= 0  # never collides with the empty marker
    with make() as collection:
        collection.add(ids=["doc-a"], embeddings=[[1, 0, 0, 0]])
        assert collection.query([1, 0, 0, 0], k=1)[0].id == "doc-a"


def test_stable_id_rejects_ambiguous_types():
    with pytest.raises(TypeError, match="string or integer"):
        stable_id(1.5)
    with pytest.raises(TypeError, match="not bool"):
        stable_id(True)


def test_metadata_filters_cover_the_operator_vocabulary():
    with make() as collection:
        collection.add(
            ids=["a", "b", "c"],
            embeddings=[[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.2, 0, 0]],
            metadatas=[
                {"lang": "en", "year": 2021, "tag": "alpha"},
                {"lang": "fr", "year": 2023, "tag": "beta"},
                {"lang": "en", "year": 2024, "tag": "gamma"},
            ],
        )
        q = [1, 0, 0, 0]
        assert {r.id for r in collection.query(q, k=3, where={"lang": "en"})} == {"a", "c"}
        assert {r.id for r in collection.query(q, k=3, where={"year": {"$gte": 2023}})} == {
            "b",
            "c",
        }
        assert {r.id for r in collection.query(q, k=3, where={"year": {"$lt": 2023}})} == {"a"}
        assert {r.id for r in collection.query(q, k=3, where={"lang": {"$in": ["fr"]}})} == {"b"}
        assert {r.id for r in collection.query(q, k=3, where={"lang": {"$ne": "en"}})} == {"b"}
        assert {r.id for r in collection.query(q, k=3, where={"tag": {"$regex": "^g"}})} == {"c"}
        assert {
            r.id for r in collection.query(q, k=3, where={"$or": [{"lang": "fr"}, {"year": 2024}]})
        } == {"b", "c"}
        assert {
            r.id
            for r in collection.query(
                q, k=3, where={"$and": [{"lang": "en"}, {"year": {"$gt": 2022}}]}
            )
        } == {"c"}
        with pytest.raises(ValueError, match="unknown operator"):
            collection.query(q, k=1, where={"year": {"$bogus": 1}})


def test_metadata_lengths_and_types_fail_with_actionable_errors():
    with make() as collection:
        with pytest.raises(ValueError, match="same length"):
            collection.add(ids=["a", "b"], embeddings=[[1, 0, 0, 0]], metadatas=[{}])
        with pytest.raises(TypeError, match="every metadata item"):
            collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]], metadatas=["bad"])


def test_a_filter_matching_a_missing_key_excludes_rather_than_raises():
    with make() as collection:
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]], metadatas=[{}])
        assert collection.query([1, 0, 0, 0], k=1, where={"lang": "en"}) == []
        assert collection.query([1, 0, 0, 0], k=1, where={"n": {"$gt": 1}}) == []


def test_add_replaces_an_existing_id_and_older_snapshots_still_read_it():
    with make() as collection:
        collection.add(ids=["a", "b"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]])
        before = collection.snapshot()
        collection.add(ids=["a"], embeddings=[[0, 0, 1, 0]])
        assert collection.count() == 2  # replaced, not duplicated
        assert collection.query([1, 0, 0, 0], k=1)[0].id == "b"
        assert collection.query([1, 0, 0, 0], k=1, snapshot=before)[0].id == "a"


def test_update_refuses_unknown_ids_and_can_edit_metadata_alone():
    with make() as collection:
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]], metadatas=[{"n": 1}])
        with pytest.raises(KeyError):
            collection.update(ids=["nope"], metadatas=[{"n": 2}])
        collection.update(ids=["a"], metadatas=[{"n": 2}])
        assert collection.get(["a"])[0].metadata["n"] == 2
        with pytest.raises(ValueError, match="nothing to update"):
            collection.update(ids=["a"])


def test_delete_by_filter_and_the_refusal_to_delete_everything():
    with make() as collection:
        collection.add(
            ids=["a", "b"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            metadatas=[{"lang": "en"}, {"lang": "fr"}],
        )
        with pytest.raises(ValueError, match="refusing to delete everything"):
            collection.delete()
        assert collection.delete(where={"lang": "fr"}) == ["b"]
        assert collection.count() == 1
        assert collection.delete(ids=["missing"]) == []  # silent, not an error


def test_shape_and_duplicate_mistakes_are_caught_at_the_call():
    with make() as collection:
        with pytest.raises(ValueError, match="expected 1 vectors of width 4"):
            collection.add(ids=["a"], embeddings=[[1, 0]])
        with pytest.raises(ValueError, match="same length"):
            collection.add(
                ids=["a", "b"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]], metadatas=[{"x": 1}]
            )
        with pytest.raises(ValueError, match="appears twice"):
            collection.add(ids=["a", "a"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]])
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]])
        with pytest.raises(ValueError, match="width 2"):
            collection.query([1, 0], k=1)


def test_embedding_function_is_used_for_documents_and_query_text():
    def embed(texts):
        return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    with make(embedding_function=embed) as collection:
        collection.add(ids=["short", "longer"], documents=["ab", "abcdef"])
        found = collection.query(query_text="abc", k=1)
        assert found[0].id == "short"
        assert found[0].document == "ab"


def test_missing_embeddings_without_an_embedder_says_so():
    with make() as collection:
        with pytest.raises(ValueError, match="embedding_function"):
            collection.add(ids=["a"], documents=["text"])
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]])
        with pytest.raises(ValueError, match="embedding_function"):
            collection.query(query_text="text")


def test_overfetch_lets_a_selective_filter_still_fill_k():
    rng = np.random.default_rng(4)
    with make(dimensions=8) as collection:
        ids = [f"d{i}" for i in range(400)]
        vectors = rng.normal(size=(400, 8)).astype(np.float32)
        # One in twenty carries the tag, so k=5 needs a much wider search.
        metadatas = [{"tag": "keep" if i % 20 == 0 else "drop"} for i in range(400)]
        collection.add(ids=ids, embeddings=vectors, metadatas=metadatas)
        wide = collection.query(vectors[0], k=5, where={"tag": "keep"}, overfetch=80)
        narrow = collection.query(vectors[0], k=5, where={"tag": "keep"}, overfetch=1)
        assert len(wide) >= len(narrow)
        assert all(r.metadata["tag"] == "keep" for r in wide)


def test_scalar_arguments_are_accepted_where_a_sequence_is_natural():
    with make() as collection:
        collection.add(ids="solo", embeddings=[[1, 0, 0, 0]], metadatas={"n": 1})
        assert collection.count() == 1
        assert collection.get("solo")[0].metadata == {"n": 1}


def test_record_supports_attribute_and_item_access():
    record = Record(id="a", distance=0.5, metadata={"k": "v"})
    assert record.id == "a" and record["id"] == "a"
    assert record["distance"] == 0.5 and record["metadata"] == {"k": "v"}


def test_empty_collection_answers_rather_than_raising():
    with make() as collection:
        assert collection.count() == 0
        assert collection.query([1, 0, 0, 0], k=5) == []
        assert collection.get() == [] and collection.peek() == []
        assert collection.add(ids=[]) == []


def test_vacuum_can_be_held_back_by_a_snapshot():
    with make() as collection:
        collection.add(
            ids=[f"d{i}" for i in range(50)],
            embeddings=np.eye(4, dtype=np.float32)[np.arange(50) % 4]
            + 0.01 * np.arange(50)[:, None],
        )
        pinned = collection.snapshot()
        collection.delete(ids=[f"d{i}" for i in range(25)])
        assert collection.vacuum(keep_snapshot=pinned) == 0  # still readable
        assert collection.vacuum() > 0  # now reclaimable


def test_snapshot_reads_return_the_metadata_of_that_time_not_of_now():
    # The bug this guards: the engine keeps old vector versions readable, so if
    # the metadata beside them were overwritten in place a snapshot read would
    # return a historical vector dressed in current metadata -- which looks
    # like it worked.
    with make() as collection:
        collection.add(
            ids=["a"], embeddings=[[1, 0, 0, 0]], documents=["dark mode"], metadatas=[{"v": 1}]
        )
        before = collection.snapshot()
        collection.add(
            ids=["a"], embeddings=[[1, 0, 0, 0]], documents=["light mode"], metadatas=[{"v": 2}]
        )
        now = collection.query([1, 0, 0, 0], k=1)[0]
        then = collection.query([1, 0, 0, 0], k=1, snapshot=before)[0]
        assert (now.document, now.metadata["v"]) == ("light mode", 2)
        assert (then.document, then.metadata["v"]) == ("dark mode", 1)


def test_a_deleted_record_is_still_readable_with_its_metadata_from_a_snapshot():
    with make() as collection:
        collection.add(
            ids=["a", "b"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]], documents=["alpha", "beta"]
        )
        before = collection.snapshot()
        collection.delete(ids=["a"])
        assert collection.count() == 1
        assert "a" not in collection
        gone = collection.query([1, 0, 0, 0], k=1)
        assert gone[0].id == "b"
        back = collection.query([1, 0, 0, 0], k=1, snapshot=before)[0]
        assert (back.id, back.document) == ("a", "alpha")


def test_add_without_metadata_carries_the_previous_version_forward():
    with make() as collection:
        collection.add(
            ids=["a"], embeddings=[[1, 0, 0, 0]], documents=["text"], metadatas=[{"keep": True}]
        )
        collection.add(ids=["a"], embeddings=[[0, 1, 0, 0]])
        record = collection.get(["a"])[0]
        assert record.metadata == {"keep": True} and record.document == "text"


def test_vacuum_prunes_the_sidecar_history_it_no_longer_needs():
    # Without this the layer leaked: the engine reclaimed its versions and
    # nothing reclaimed the metadata beside them.
    with make() as collection:
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]])
        for n in range(200):
            collection.add(ids=["a"], embeddings=[[float(n % 3), 1, 0, 0]], metadatas=[{"n": n}])
        assert len(collection._history[stable_id("a")]) == 201
        collection.vacuum()
        assert len(collection._history[stable_id("a")]) == 1
        assert collection.query([0, 1, 0, 0], k=1)[0].metadata["n"] == 199


def test_pruning_keeps_what_a_pinned_snapshot_still_needs():
    with make() as collection:
        collection.add(ids=["x"], embeddings=[[1, 0, 0, 0]], documents=["old"])
        pinned = collection.snapshot()
        collection.add(ids=["x"], embeddings=[[0, 1, 0, 0]], documents=["new"])
        collection.vacuum(keep_snapshot=pinned)
        assert collection.query([1, 0, 0, 0], k=1, snapshot=pinned)[0].document == "old"
        assert collection.query([0, 1, 0, 0], k=1)[0].document == "new"


def test_a_record_deleted_after_a_pin_keeps_its_identity_at_that_pin():
    # Pruning tested the last *write* rather than the delete, so a record
    # written at t=5 and deleted at t=10 was dropped at a horizon of 7 -- where
    # it is still alive. The engine kept returning its vector while the id came
    # back as a raw hash with no document and no metadata.
    with make() as collection:
        collection.add(
            ids=["a", "b"],
            embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]],
            documents=["alpha", "beta"],
            metadatas=[{"k": 1}, {"k": 2}],
        )
        pinned = collection.snapshot()
        collection.delete(ids=["a"])
        collection.vacuum(keep_snapshot=pinned)
        found = collection.query([1, 0, 0, 0], k=1, snapshot=pinned)[0]
        assert found.id == "a"
        assert found.document == "alpha"
        assert found.metadata == {"k": 1}


def test_history_of_a_deleted_record_goes_once_nothing_can_reach_it():
    with make() as collection:
        collection.add(ids=["a", "b"], embeddings=[[1, 0, 0, 0], [0, 1, 0, 0]])
        collection.delete(ids=["a"])
        collection.vacuum()
        assert stable_id("a") not in collection._history
        assert stable_id("b") in collection._history
        assert collection.count() == 1


def test_deleting_then_re_adding_an_id_makes_it_live_again():
    with make() as collection:
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]], documents=["first"])
        collection.delete(ids=["a"])
        assert "a" not in collection
        collection.add(ids=["a"], embeddings=[[1, 0, 0, 0]], documents=["second"])
        assert "a" in collection and collection.count() == 1
        collection.vacuum()
        # Re-adding cleared the deletion, so the vacuum must not drop it.
        assert collection.get(["a"])[0].document == "second"
        assert collection.query([1, 0, 0, 0], k=1)[0].id == "a"
