"""Complex filter expressions in Collection.query().

The existing test_collection.py covers simple operators. These tests focus on
deeply nested boolean compositions and edge cases that the operator vocabulary
must handle correctly.
"""

from chronovec import Collection


def make():
    coll = Collection(dimensions=4, metric="l2")
    coll.add(
        ids=["a", "b", "c", "d", "e"],
        embeddings=[
            [1.0, 0, 0, 0],
            [0.9, 0.1, 0, 0],
            [0.8, 0.2, 0, 0],
            [0.7, 0.3, 0, 0],
            [0.6, 0.4, 0, 0],
        ],
        metadatas=[
            {"lang": "en", "year": 2021, "score": 0.9, "tags": ["python", "ml"]},
            {"lang": "fr", "year": 2022, "score": 0.7, "tags": ["ml"]},
            {"lang": "en", "year": 2023, "score": 0.5, "tags": ["python"]},
            {"lang": "de", "year": 2024, "score": 0.3, "tags": ["rust"]},
            {"lang": "en", "year": 2023, "score": 0.8, "tags": ["python", "rust"]},
        ],
    )
    return coll


def test_or_of_two_and_clauses():
    coll = make()
    q = [1, 0, 0, 0]
    # (lang=en AND year>=2023) OR (lang=fr AND score<0.8)
    result = coll.query(
        q,
        k=10,
        where={
            "$or": [
                {"$and": [{"lang": "en"}, {"year": {"$gte": 2023}}]},
                {"$and": [{"lang": "fr"}, {"score": {"$lt": 0.8}}]},
            ]
        },
    )
    ids = {r.id for r in result}
    assert "c" in ids  # lang=en, year=2023
    assert "e" in ids  # lang=en, year=2023
    assert "b" in ids  # lang=fr, score=0.7
    assert "a" not in ids  # lang=en but year=2021
    assert "d" not in ids  # lang=de


def test_and_with_four_conditions():
    coll = make()
    q = [1, 0, 0, 0]
    # lang=en AND year>=2021 AND year<=2023 AND score>=0.5
    result = coll.query(
        q,
        k=10,
        where={
            "$and": [
                {"lang": "en"},
                {"year": {"$gte": 2021}},
                {"year": {"$lte": 2023}},
                {"score": {"$gte": 0.5}},
            ]
        },
    )
    ids = {r.id for r in result}
    assert "a" in ids  # lang=en, year=2021, score=0.9
    assert "c" in ids  # lang=en, year=2023, score=0.5
    assert "e" in ids  # lang=en, year=2023, score=0.8
    assert "b" not in ids  # lang=fr
    assert "d" not in ids  # lang=de


def test_regex_combined_with_numeric_gt():
    coll = make()
    q = [1, 0, 0, 0]
    # lang matches /^e/ AND score > 0.6
    result = coll.query(
        q,
        k=10,
        where={
            "$and": [
                {"lang": {"$regex": "^e"}},
                {"score": {"$gt": 0.6}},
            ]
        },
    )
    ids = {r.id for r in result}
    assert "a" in ids  # lang=en, score=0.9
    assert "e" in ids  # lang=en, score=0.8
    assert "c" not in ids  # lang=en but score=0.5
    assert "b" not in ids  # lang=fr (does not match ^e ... wait fr starts with f)


def test_impossible_filter_returns_empty():
    coll = make()
    q = [1, 0, 0, 0]
    # lang=en AND lang=fr: cannot both be true
    result = coll.query(q, k=10, where={"$and": [{"lang": "en"}, {"lang": "fr"}]})
    assert result == []


def test_ne_and_nin_compose_correctly():
    coll = make()
    q = [1, 0, 0, 0]
    # lang not in [en, fr] AND year != 2024
    result = coll.query(
        q,
        k=10,
        where={
            "$and": [
                {"lang": {"$nin": ["en", "fr"]}},
                {"year": {"$ne": 2024}},
            ]
        },
    )
    ids = {r.id for r in result}
    assert ids == set()  # only de (d) qualifies on lang, but d has year=2024


def test_contains_operator_on_string_metadata():
    coll = make()
    # Add a record with a string containing a substring
    coll.add(
        ids=["f"],
        embeddings=[[0.5, 0.5, 0, 0]],
        metadatas=[{"description": "multiversion concurrency control"}],
    )
    q = [0.5, 0.5, 0, 0]
    result = coll.query(q, k=1, where={"description": {"$contains": "concurrency"}})
    assert result and result[0].id == "f"


def test_filter_on_missing_key_excludes_record():
    coll = make()
    q = [1, 0, 0, 0]
    # Filter on a key that does not exist in any record
    result = coll.query(q, k=10, where={"nonexistent_key": "value"})
    assert result == []


def test_or_where_one_branch_matches_all():
    coll = make()
    q = [1, 0, 0, 0]
    # lang=en OR lang=fr OR lang=de: should match all 5 records
    result = coll.query(q, k=10, where={"$or": [{"lang": "en"}, {"lang": "fr"}, {"lang": "de"}]})
    assert len(result) == 5
