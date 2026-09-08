# SQLite virtual table

ChronoVec exposes a SQLite virtual table interface. SQLite's WAL provides durability; ChronoVec maintains a native cache rebuilt from committed SQLite state on reopen.

## Building the extension

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target chronovec_sqlite
```

This produces `build/libchronovec_sqlite.so` (Linux) or `build/libchronovec_sqlite.dylib` (macOS).

## Loading the extension

```python
import sqlite3

conn = sqlite3.connect("memory.db")
conn.enable_load_extension(True)
conn.load_extension("./build/libchronovec_sqlite")
```

Or in the SQLite shell:
```sql
.load ./build/libchronovec_sqlite
```

## Creating a virtual table

```sql
CREATE VIRTUAL TABLE memories USING chronovec(
    dim=384,
    metric=cosine,
    capacity=256,
    nprobe=16
);
```

## Inserting vectors

```python
import struct

def pack_vector(v):
    return struct.pack(f"{len(v)}f", *v)

conn.execute(
    "INSERT INTO memories(id, vector) VALUES (?, ?)",
    (42, pack_vector(embedding))
)
conn.commit()
```

## Searching

```sql
SELECT id, distance
FROM memories
WHERE query = :packed_query
  AND k = 10
ORDER BY distance;
```

```python
results = conn.execute(
    "SELECT id, distance FROM memories WHERE query=? AND k=?",
    (pack_vector(query), 10)
).fetchall()
```

## Persistence and recovery

SQLite's pager and WAL handle persistence. On `conn.close()` + `conn = sqlite3.connect(...)`, the ChronoVec native cache is reconstructed from the committed SQLite state: no explicit save/load required.

## dqlite deployment

The SQLite shadow log is compatible with dqlite's Raft replication: shadow log entries are ordinary SQLite transactions and are replicated automatically. Every server must call `chronovec_register_vtab(sqlite3*)` before opening the database. See the [`native/sqlite/` source directory](https://github.com/mchl-labs/chronovec/tree/main/native/sqlite) for the static registration API.

> **Note:** Three-node dqlite leader-failover is untested. Do not claim distributed recovery until leader-failover and follower-rebuild are validated.
