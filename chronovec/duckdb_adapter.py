from __future__ import annotations

import contextlib
import json
from typing import Any

import numpy as np

from .index import _ReferenceIndex

# Backward-compatible name used in existing type annotations.
ChronoVecIndex = _ReferenceIndex


class ChronoDuckDBAdapter:
    """Python-level DuckDB integration. **Experimental**: not yet production-ready.

    Known limitations:
    - Not thread-safe: depends on the Python GIL; concurrent queries will race.
    - Minimal test coverage compared to the core index and other integrations.
    - API surface may change before promotion to stable.

    Use ``asyncio.to_thread()`` or an explicit lock when calling from async code
    or from multiple threads.

    This is an adapter, not a native DuckDB index extension. It registers a
    scalar UDF returning matching IDs and offers explicit table sync helpers.
    """

    def __init__(
        self, connection: Any, index: _ReferenceIndex, *, function_name: str = "chronovec_search"
    ) -> None:
        self.connection = connection
        self.index = index
        self.function_name = function_name

    def register(self) -> None:
        import duckdb

        def search_json(vector: list[float], k: int, snapshot: int | None = None) -> str:
            results = self.index.search(
                np.asarray(vector, dtype=np.float32), int(k), snapshot=snapshot
            )
            return json.dumps(
                [{"id": result.id, "distance": result.distance} for result in results]
            )

        with contextlib.suppress(Exception):
            self.connection.remove_function(self.function_name)
        self.connection.create_function(
            self.function_name,
            search_json,
            [duckdb.list_type("FLOAT"), "BIGINT", "BIGINT"],
            "VARCHAR",
            null_handling="special",
        )

    def ingest_table(
        self, table: str, *, id_column: str = "id", vector_column: str = "embedding"
    ) -> int:
        # Identifiers cannot be bound as parameters, so restrict them to simple
        # SQL identifiers before interpolation.
        for value in (table, id_column, vector_column):
            if not value.replace("_", "").isalnum():
                raise ValueError(f"unsafe SQL identifier: {value!r}")
        rows = self.connection.execute(
            f'SELECT "{id_column}", "{vector_column}" FROM "{table}"'
        ).fetchall()
        for item_id, vector in rows:
            self.index.insert(int(item_id), vector)
        return len(rows)
