"""Run in an environment whose SQLite enables loadable extensions."""

import os
import sqlite3
import struct
import tempfile

extension = os.environ.get("CHRONOVEC_SQLITE_EXTENSION", "build/chronovec_sqlite.so")
with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "vectors.db")
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(extension)
    db.execute("CREATE VIRTUAL TABLE vec USING chronovec(dim=3, capacity=8, nprobe=4)")
    one = struct.pack("fff", 1, 0, 0)
    two = struct.pack("fff", 0, 1, 0)
    db.execute("INSERT INTO vec(id, vector) VALUES (?, ?)", (1, one))
    db.execute("INSERT INTO vec(id, vector) VALUES (?, ?)", (2, two))
    db.commit()
    assert db.execute("SELECT id FROM vec WHERE query=? AND k=1", (one,)).fetchone()[0] == 1
    db.close()

    # The native page graph was destroyed. Reopening must replay only the
    # durable SQLite operation log and produce the same result.
    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(extension)
    assert db.execute("SELECT id FROM vec WHERE query=? AND k=1", (one,)).fetchone()[0] == 1
    db.execute("BEGIN")
    db.execute("INSERT INTO vec(id, vector) VALUES (?, ?)", (3, one))
    db.rollback()
    assert db.execute("SELECT count(*) FROM vec WHERE query=? AND k=10", (one,)).fetchone()[0] == 2
    db.execute("DELETE FROM vec WHERE id=2")
    db.commit()
    db.close()

    db = sqlite3.connect(path)
    db.enable_load_extension(True)
    db.load_extension(extension)
    assert db.execute("SELECT count(*) FROM vec WHERE query=? AND k=10", (one,)).fetchone()[0] == 1
    db.close()

print("persistent virtual table reopen and rollback: ok")
