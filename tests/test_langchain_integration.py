"""The LangChain adapter against the real langchain-core interfaces.

Duck typing is not enough: `as_retriever` lives on the base class and LCEL
chains type-check what they are handed, so an adapter that merely has the right
method names cannot be dropped into a chain.
"""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.vectorstores import VectorStore  # noqa: E402

from chronovec.integrations.langchain import ChronoVecVectorStore  # noqa: E402


class Embeddings:
    """Deterministic stand-in. Length and first letter give two usable axes."""

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [float(len(text)), float(ord(text[0]) if text else 0), 1.0, 0.0]


def store():
    return ChronoVecVectorStore(embedding=Embeddings(), dimensions=4, metric="cosine")


def test_it_is_a_real_vectorstore_not_a_lookalike():
    assert issubclass(ChronoVecVectorStore, VectorStore)
    assert isinstance(store(), VectorStore)


def test_add_texts_and_similarity_search_return_documents():
    s = store()
    ids = s.add_texts(
        ["dark mode preferred", "tabs not spaces"], metadatas=[{"k": "a"}, {"k": "b"}]
    )
    assert len(ids) == 2
    found = s.similarity_search("dark mode preferred", k=1)
    assert isinstance(found[0], Document)
    assert found[0].page_content == "dark mode preferred"
    assert found[0].metadata["k"] == "a"


def test_similarity_search_with_score_returns_pairs():
    s = store()
    s.add_texts(["alpha"])
    pairs = s.similarity_search_with_score("alpha", k=1)
    document, score = pairs[0]
    assert isinstance(document, Document) and isinstance(score, float)


def test_as_retriever_works_which_is_the_point_of_the_base_class():
    s = store()
    s.add_texts(["dark mode preferred", "tabs not spaces"])
    retriever = s.as_retriever(search_kwargs={"k": 1})
    got = retriever.invoke("dark mode preferred")
    assert [d.page_content for d in got] == ["dark mode preferred"]


def test_delete_removes_and_reports():
    s = store()
    ids = s.add_texts(["alpha", "beta"])
    assert s.delete([ids[0]]) is True
    assert s.delete([]) is False
    assert [d.page_content for d in s.similarity_search("alpha", k=5)] == ["beta"]


def test_from_texts_classmethod_builds_a_populated_store():
    s = ChronoVecVectorStore.from_texts(["alpha", "beta"], Embeddings(), dimensions=4)
    assert len(s.similarity_search("alpha", k=5)) == 2


def test_similarity_search_calls_embed_query_not_embed_documents():
    # LangChain embedders commonly use different instructions/prefixes for
    # indexing vs. retrieval; the store must route through the matching method.
    calls: list[tuple[str, object]] = []

    class AsymmetricEmbeddings:
        def embed_documents(self, texts):
            calls.append(("documents", list(texts)))
            return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

        def embed_query(self, text):
            calls.append(("query", text))
            return [0.0, 1.0, 0.0, 0.0]

    s = ChronoVecVectorStore(embedding=AsymmetricEmbeddings(), dimensions=4, metric="l2")
    s.add_texts(["alpha"])
    s.similarity_search("find alpha", k=1)

    assert calls == [("documents", ["alpha"]), ("query", "find alpha")]


def test_branching_is_the_thing_langchain_has_no_equivalent_for():
    s = store()
    s.add_texts(["the user prefers dark mode"])
    with s.branch("hypothesis") as scratch:
        scratch.add_texts(["the user might want light mode"])
        assert len(scratch.similarity_search("mode", k=5)) == 2
    # Speculation discarded without rebuilding anything.
    assert len(s.similarity_search("mode", k=5)) == 1
