# DuckDB integration (experimental)

ChronoVec provides a beta DuckDB UDF adapter that exposes vector search via SQL.

> **Status:** Experimental. Not thread-safe. Not covered by the main correctness claims. Test coverage is limited. Use in production at your own risk.

## Installation

```bash
pip install "chronovec[duckdb]"
```

## Usage

```python
import duckdb
from chronovec.duckdb_adapter import ChronoDuckDBAdapter

adapter = ChronoDuckDBAdapter(dimensions=384, metric="cosine")
conn = duckdb.connect()
adapter.register(conn)

# Insert via the adapter
adapter.insert(1, embedding_array)

# Query via SQL
results = conn.execute("""
    SELECT id, distance
    FROM chronovec_search(?, 10)
    ORDER BY distance
""", [query_embedding]).fetchall()
```

## Ingesting a table

```python
conn.execute("""
    CREATE TABLE docs (id INTEGER, text VARCHAR, embedding FLOAT[384])
""")
# ... populate docs ...

adapter.ingest_table(conn, "docs", id_col="id", embedding_col="embedding")

results = conn.execute("""
    SELECT id, distance
    FROM chronovec_search(?, 5)
""", [query]).fetchall()
```

## Limitations

- Not thread-safe: the adapter holds the Python GIL during search.
- Metadata filtering is not exposed via SQL (use the Python `Collection` API instead).
- Not suitable for concurrent writers.
- Performance numbers for this adapter are not part of the benchmarked claims.
