import sys
from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from chronovec import (
    Collection,
    CustomEmbedding,
    ProviderEmbedding,
    SentenceTransformerEmbedding,
    as_embedding_function,
)


def test_custom_embedding_can_use_distinct_document_and_query_functions():
    calls: list[tuple[str, object]] = []

    def documents(texts: Sequence[str]):
        calls.append(("documents", list(texts)))
        return [[float(len(texts)), 0.0] for _ in texts]

    def query(text: str):
        calls.append(("query", text))
        return [0.0, 1.0]

    embedder = CustomEmbedding(documents, query)
    with Collection(2, metric="l2", embedding_function=embedder) as collection:
        collection.add(ids=["a", "b"], documents=["one", "two"])
        collection.query(query_text="find", k=1)

    assert calls == [("documents", ["one", "two"]), ("query", "find")]


def test_original_batch_callable_still_supports_document_writes_and_queries():
    def embed(texts: Sequence[str]):
        return [[float(len(text)), 0.0] for text in texts]

    with Collection(2, metric="l2", embedding_function=embed) as collection:
        collection.add(ids=["a"], documents=["hello"])
        assert collection.query(query_text="hello", k=1)[0].id == "a"


def test_langchain_shaped_object_is_supported_without_importing_langchain():
    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    normalized = as_embedding_function(FakeEmbeddings())
    assert normalized.embed_documents(["a"]) == [[1.0, 0.0]]
    assert normalized.embed_query("a") == [1.0, 0.0]


def test_llamaindex_shaped_object_is_supported_without_importing_llamaindex():
    class FakeEmbedding:
        def get_text_embedding_batch(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def get_query_embedding(self, text):
            return [1.0, 0.0]

    normalized = as_embedding_function(FakeEmbedding())
    assert normalized.embed_documents(["a"]) == [[1.0, 0.0]]
    assert normalized.embed_query("a") == [1.0, 0.0]


def test_sentence_transformer_adapter_uses_specialized_encode_methods_when_available():
    class FakeModel:
        def encode(self, texts, **kwargs):
            raise AssertionError("specialized methods should be preferred")

        def encode_document(self, texts, **kwargs):
            assert kwargs == {"normalize_embeddings": True}
            return [[1.0, 0.0] for _ in texts]

        def encode_query(self, text, **kwargs):
            assert kwargs == {"normalize_embeddings": True}
            return [0.0, 1.0]

    embedder = SentenceTransformerEmbedding(FakeModel(), normalize_embeddings=True)
    assert embedder.embed_documents(["a"]) == [[1.0, 0.0]]
    assert embedder.embed_query("a") == [0.0, 1.0]


def test_sentence_transformer_adapter_falls_back_to_encode():
    class FakeModel:
        def encode(self, values, **kwargs):
            return [[1.0, 0.0] for _ in values] if isinstance(values, list) else [1.0, 0.0]

    embedder = SentenceTransformerEmbedding(FakeModel())
    assert embedder.embed_documents(["a"]) == [[1.0, 0.0]]
    assert embedder.embed_query("a") == [1.0, 0.0]


def test_provider_adapter_uses_explicit_secret_and_separate_options(monkeypatch):
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        return {"data": [{"embedding": [1.0, 0.0]} for _ in kwargs["input"]]}

    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(embedding=fake_embedding))
    embedder = ProviderEmbedding(
        "test-provider",
        "test-model",
        api_key="secret",
        document_kwargs={"input_type": "search_document"},
        query_kwargs={"input_type": "search_query"},
    )

    assert embedder.embed_documents(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert embedder.embed_query("q") == [1.0, 0.0]
    assert calls == [
        {
            "model": "test-provider/test-model",
            "input": ["a", "b"],
            "api_key": "secret",
            "input_type": "search_document",
        },
        {
            "model": "test-provider/test-model",
            "input": ["q"],
            "api_key": "secret",
            "input_type": "search_query",
        },
    ]


def test_invalid_embedding_object_has_actionable_error():
    with pytest.raises(TypeError, match="batch callable"):
        as_embedding_function(object())

    class Incomplete:
        def embed_documents(self, texts):
            return []

    with pytest.raises(TypeError, match="embed_query"):
        as_embedding_function(Incomplete())
