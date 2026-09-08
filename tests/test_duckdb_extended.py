"""Extended DuckDB adapter tests.

The original test_duckdb.py is a 19-line smoke test. These tests cover
additional behaviour: large-k queries, custom column names, snapshot reads,
and custom function names.
"""

import json

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")

from chronovec import ChronoVecIndex  # noqa: E402
from chronovec.duckdb_adapter import ChronoDuckDBAdapter  # noqa: E402


def make_adapter(dim=4, n=10):
    conn = duckdb.connect()
    index = ChronoVecIndex(dim)
    for i in range(n):
        v = np.zeros(dim, dtype=np.float32)
        v[i % dim] = 1.0
        index.insert(i, v)
    return conn, index, ChronoDuckDBAdapter(conn, index)


def test_k_larger_than_corpus_returns_what_exists():
    conn, index, adapter = make_adapter(n=5)
    adapter.register()
    payload = conn.execute(
        "SELECT chronovec_search([1.0, 0.0, 0.0, 0.0]::FLOAT[], 100, NULL)"
    ).fetchone()[0]
    results = json.loads(payload)
    assert len(results) <= 5
    assert all("id" in r and "distance" in r for r in results)


def test_custom_id_and_vector_column_names():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE docs(doc_id BIGINT, vec FLOAT[])")
    conn.execute("INSERT INTO docs VALUES (42, [1.0, 0.0]), (99, [0.0, 1.0])")
    index = ChronoVecIndex(2)
    adapter = ChronoDuckDBAdapter(conn, index)
    assert adapter.ingest_table("docs", id_column="doc_id", vector_column="vec") == 2
    adapter.register()
    payload = conn.execute("SELECT chronovec_search([1.0, 0.0]::FLOAT[], 1, NULL)").fetchone()[0]
    results = json.loads(payload)
    assert results[0]["id"] == 42


def test_snapshot_argument_is_forwarded():
    conn = duckdb.connect()
    index = ChronoVecIndex(2)
    adapter = ChronoDuckDBAdapter(conn, index)
    t1 = index.insert(1, np.array([1.0, 0.0], dtype=np.float32))
    index.delete(1)
    adapter.register()

    # Without snapshot: id 1 is gone
    payload_now = conn.execute("SELECT chronovec_search([1.0, 0.0]::FLOAT[], 5, NULL)").fetchone()[
        0
    ]
    ids_now = {r["id"] for r in json.loads(payload_now)}
    assert 1 not in ids_now

    # With snapshot at t1: id 1 is visible
    payload_past = conn.execute(
        f"SELECT chronovec_search([1.0, 0.0]::FLOAT[], 5, {t1}::BIGINT)"
    ).fetchone()[0]
    ids_past = {r["id"] for r in json.loads(payload_past)}
    assert 1 in ids_past


def test_custom_function_name():
    conn = duckdb.connect()
    conn.execute("CREATE TABLE items(id BIGINT, embedding FLOAT[])")
    conn.execute("INSERT INTO items VALUES (7, [0.5, 0.5])")
    index = ChronoVecIndex(2)
    adapter = ChronoDuckDBAdapter(conn, index, function_name="cv_find")
    adapter.ingest_table("items")
    adapter.register()
    payload = conn.execute("SELECT cv_find([0.5, 0.5]::FLOAT[], 1, NULL)").fetchone()[0]
    results = json.loads(payload)
    assert results[0]["id"] == 7


def test_unsafe_identifier_is_rejected():
    conn = duckdb.connect()
    index = ChronoVecIndex(2)
    adapter = ChronoDuckDBAdapter(conn, index)
    with pytest.raises(ValueError, match="unsafe SQL identifier"):
        adapter.ingest_table("items; DROP TABLE items")
