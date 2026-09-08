"""Adapter tests.

LangChain and LlamaIndex are not installed here, so these exercise the shared
core and the framework-independent behaviour of the adapters. The CI job that
installs the real packages is what verifies interface conformance.
"""

import numpy as np
import pytest

from chronovec.integrations import ChronoVecStore


class FakeEmbedder:
    """Deterministic bag-of-characters embedding: no model, still meaningful."""

    dimensions = 32

    def __call__(self, texts):
        out = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for char in text.lower():
                out[row, ord(char) % self.dimensions] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out

    def embed_documents(self, texts):
        return self(texts).tolist()

    def embed_query(self, text):
        return self([text])[0].tolist()


def make_store():
    embed = FakeEmbedder()
    return ChronoVecStore(embed, embed.dimensions, nprobe=64)


def test_add_and_search_round_trip():
    store = make_store()
    store.add_texts(
        ["dark mode preferred", "likes espresso"], metadatas=[{"kind": "pref"}, {"kind": "pref"}]
    )
    found = store.similarity_search("dark mode preferred", k=1)
    assert found[0].text == "dark mode preferred"
    assert found[0].metadata["kind"] == "pref"
    store.close()


def test_ids_are_stable_across_instances():
    a, b = make_store(), make_store()
    assert a.add_texts(["x"], ids=["fixed"]) == b.add_texts(["x"], ids=["fixed"])
    a.close()
    b.close()


def test_branch_isolates_then_discards():
    store = make_store()
    store.add_texts(["committed fact"])
    branch = store.branch("scratch")
    store.add_texts(["speculative guess"], view=branch)

    in_branch = {d.text for d in store.similarity_search("guess", k=10, view=branch)}
    on_main = {d.text for d in store.similarity_search("guess", k=10)}
    assert "speculative guess" in in_branch
    assert "speculative guess" not in on_main

    branch.discard()
    after = {d.text for d in store.similarity_search("guess", k=10)}
    assert after == {"committed fact"}
    store.close()


def test_snapshot_read_sees_deleted_document():
    store = make_store()
    ids = store.add_texts(["will be removed"])
    before = store.memory.clock
    store.delete(ids)
    assert store.similarity_search("will be removed", k=5) == []
    revived = store.similarity_search("will be removed", k=5, as_of=before)
    assert [d.text for d in revived] == ["will be removed"]
    store.close()


def test_mismatched_metadata_length_rejected():
    store = make_store()
    with pytest.raises(ValueError, match="same length"):
        store.add_texts(["a", "b"], metadatas=[{"only": 1}])
    store.close()


def test_langchain_adapter_reports_missing_dependency():
    # Only meaningful without langchain-core. With it installed the adapter is
    # expected to work, and tests/test_langchain_integration.py covers that.
    pytest.importorskip.__module__  # keep the import used
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip(
            "langchain-core is installed; the working path is tested "
            "in test_langchain_integration.py"
        )
    from chronovec.integrations.langchain import ChronoVecVectorStore

    store = ChronoVecVectorStore(FakeEmbedder(), 32)
    store.add_texts(["hello"])
    with pytest.raises(ImportError, match="langchain-core is required"):
        store.similarity_search("hello")
    store.close()
