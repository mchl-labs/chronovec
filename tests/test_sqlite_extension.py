import sqlite3
from pathlib import Path

import numpy as np


def test_sqlite_extension():
    build = Path(__file__).parents[1] / "build"
    extension = next(
        (
            path
            for path in (build / "chronovec_sqlite.so", build / "chronovec_sqlite.dylib")
            if path.exists()
        ),
        None,
    )
    if extension is None:
        __import__("pytest").skip("SQLite extension has not been built")
    db = sqlite3.connect(":memory:")
    if not hasattr(db, "enable_load_extension"):
        __import__("pytest").skip("Python sqlite3 was built without loadable-extension support")
    db.enable_load_extension(True)
    # The default entrypoint keeps this portable across Python sqlite3 builds.
    db.load_extension(str(extension))
    assert db.execute("SELECT chronovec_create('items', 2, 'cosine', 8, 4)").fetchone()[0] == 1
    db.execute(
        "SELECT chronovec_insert('items', 1, ?)", (np.array([1, 0], dtype=np.float32).tobytes(),)
    )
    db.execute(
        "SELECT chronovec_insert('items', 2, ?)", (np.array([0, 1], dtype=np.float32).tobytes(),)
    )
    result = db.execute(
        "SELECT chronovec_search('items', ?, 1)", (np.array([1, 0], dtype=np.float32).tobytes(),)
    ).fetchone()[0]
    assert '"id":1' in result
