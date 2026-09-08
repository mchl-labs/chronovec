import json

import duckdb

from chronovec import ChronoVecIndex
from chronovec.duckdb_adapter import ChronoDuckDBAdapter


def test_duckdb_udf_and_ingest():
    connection = duckdb.connect()
    connection.execute("CREATE TABLE items(id BIGINT, embedding FLOAT[])")
    connection.execute("INSERT INTO items VALUES (1, [1, 0]), (2, [0, 1])")
    index = ChronoVecIndex(2)
    adapter = ChronoDuckDBAdapter(connection, index)
    assert adapter.ingest_table("items") == 2
    adapter.register()
    payload = connection.execute(
        "SELECT chronovec_search([1.0, 0.0]::FLOAT[], 1, NULL)"
    ).fetchone()[0]
    assert json.loads(payload)[0]["id"] == 1
