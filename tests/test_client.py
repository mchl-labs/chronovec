import json

import numpy as np
import pytest

from chronovec import Client, PersistentClient
from chronovec.cli import main


def test_client_persists_named_collection(tmp_path):
    client = Client(tmp_path / "db")
    collection = client.get_or_create_collection("docs", dimensions=3, metric="l2")
    collection.add(
        ids=["a"], embeddings=np.array([[1, 0, 0]], dtype=np.float32), metadatas=[{"v": 1}]
    )
    collection.update("a", metadatas={"v": 2})
    assert client.list_collections() == ["docs"]
    collection.close()

    reopened = PersistentClient(tmp_path / "db")
    restored = reopened.get_collection("docs")
    assert reopened.get_collection("docs") is restored
    assert restored.count() == 1
    assert restored.get("a")[0].metadata == {"v": 2}
    restored.delete("a")
    restored.close()
    empty = PersistentClient(tmp_path / "db").get_collection("docs")
    assert empty.count() == 0
    empty.close()


def test_client_requires_dimensions_for_new_collection(tmp_path):
    with pytest.raises(ValueError, match="dimensions"):
        Client(tmp_path).get_or_create_collection("docs")


def test_cli_lists_and_inspects(tmp_path, capsys):
    client = Client(tmp_path / "db")
    collection = client.get_or_create_collection("docs", dimensions=3)
    collection.add(ids=["a"], embeddings=[[1, 0, 0]])
    collection.close()

    assert main(["list", str(tmp_path / "db")]) == 0
    assert capsys.readouterr().out.strip() == "docs"
    assert main(["inspect", str(tmp_path / "db"), "docs", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["name"] == "docs"
    assert report["count"] == 1
