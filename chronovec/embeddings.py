"""Provider-neutral embedding interfaces for the high-level ChronoVec API.

ChronoVec does not pick a model provider, manage model downloads, or read
credentials from config or the environment on its own. `CustomEmbedding` and
the duck-typed LangChain/LlamaIndex adapters below add zero dependencies.
`ProviderEmbedding` is an optional convenience layer over LiteLLM for hosted
providers -- it exists so ChronoVec doesn't hand-write and maintain a client
per provider, not to make ChronoVec a provider itself -- and it still takes
the secret explicitly from the caller on every construction.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

Vector = Sequence[float]
DocumentEmbedder = Callable[[Sequence[str]], Sequence[Vector]]
QueryEmbedder = Callable[[str], Vector]


@runtime_checkable
class EmbeddingFunction(Protocol):
    """The embedding contract used by :class:`chronovec.Collection`.

    Document and query methods are separate because some models use different
    instructions or prefixes for indexing and retrieval.
    """

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> Vector:
        """Embed one search query."""


EmbeddingInput = EmbeddingFunction | DocumentEmbedder


class CustomEmbedding:
    """Adapt application functions to ChronoVec's embedding interface.

    ``embed_query`` is optional. When omitted, the document function is called
    with a one-item batch for text queries, preserving the original ChronoVec
    callable API.

    The object is callable for backwards compatibility, so it can also be
    passed to integrations that expect a batch embedding function.
    """

    def __init__(
        self,
        embed_documents: DocumentEmbedder,
        embed_query: QueryEmbedder | None = None,
    ) -> None:
        if not callable(embed_documents):
            raise TypeError("embed_documents must be callable")
        if embed_query is not None and not callable(embed_query):
            raise TypeError("embed_query must be callable")
        self._embed_documents = embed_documents
        self._embed_query = embed_query

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self._embed_documents(texts)

    def embed_query(self, text: str) -> Vector:
        if self._embed_query is not None:
            return self._embed_query(text)
        return self._embed_documents([text])[0]

    def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self.embed_documents(texts)


class SentenceTransformerEmbedding:
    """Use a local `sentence-transformers` model with ChronoVec.

    ``model`` may be a model name/path or an already loaded
    ``SentenceTransformer`` instance. The dependency is imported only when a
    model name is supplied, so importing ChronoVec does not load an ML runtime.
    """

    def __init__(self, model: str | Any, **encode_kwargs: Any) -> None:
        if isinstance(model, str):
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "SentenceTransformerEmbedding requires the optional dependency; "
                    "install it with `pip install chronovec[embeddings]`"
                ) from exc
            model = SentenceTransformer(model)
        if not callable(getattr(model, "encode", None)):
            raise TypeError("model must be a model name or SentenceTransformer-like object")
        self.model = model
        self.encode_kwargs = dict(encode_kwargs)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        encoder = getattr(self.model, "encode_document", self.model.encode)
        return encoder(list(texts), **self.encode_kwargs)

    def embed_query(self, text: str) -> Vector:
        encoder = getattr(self.model, "encode_query", self.model.encode)
        return encoder(text, **self.encode_kwargs)

    def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self.embed_documents(texts)


class ProviderEmbedding:
    """Use a hosted embedding provider through a thin LiteLLM pass-through.

    LiteLLM already speaks every hosted provider's embedding API, so this
    wrapper only adapts its response shape to ChronoVec's
    ``embed_documents``/``embed_query`` contract -- it does not add its own
    provider-specific logic, and installing it does not make ChronoVec depend
    on LiteLLM unless this class is used.

    ``provider`` identifies the backend (for example ``"openai"`` or
    ``"cohere"``), while ``model`` is the provider's model name. Pass the
    secret explicitly, usually from an environment variable. Extra keyword
    arguments are passed to both calls; use ``document_kwargs`` and
    ``query_kwargs`` for provider-specific options.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str,
        document_kwargs: Mapping[str, Any] | None = None,
        query_kwargs: Mapping[str, Any] | None = None,
        **embedding_kwargs: Any,
    ) -> None:
        if not isinstance(provider, str) or not provider:
            raise TypeError("provider must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise TypeError("model must be a non-empty string")
        if not isinstance(api_key, str) or not api_key:
            raise TypeError("api_key must be a non-empty string")
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.embedding_kwargs = dict(embedding_kwargs)
        self.document_kwargs = dict(document_kwargs or {})
        self.query_kwargs = dict(query_kwargs or {})

    @staticmethod
    def _response_vectors(response: Any) -> list[Vector]:
        data = response.get("data") if isinstance(response, Mapping) else response.data
        if data is None:
            raise ValueError("embedding provider response did not contain data")
        vectors: list[Vector] = []
        for item in data:
            vector = item.get("embedding") if isinstance(item, Mapping) else item.embedding
            if vector is None:
                raise ValueError("embedding provider response did not contain an embedding")
            vectors.append(vector)
        return vectors

    def _embed(self, values: list[str], options: Mapping[str, Any]) -> list[Vector]:
        try:
            from litellm import embedding
        except ImportError as exc:
            raise ImportError(
                "ProviderEmbedding requires the optional dependency; install it with "
                "`pip install chronovec[embeddings]`"
            ) from exc
        response = embedding(
            model=f"{self.provider}/{self.model}",
            input=values,
            api_key=self.api_key,
            **self.embedding_kwargs,
            **options,
        )
        return self._response_vectors(response)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self._embed(list(texts), self.document_kwargs)

    def embed_query(self, text: str) -> Vector:
        return self._embed([text], self.query_kwargs)[0]

    def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self.embed_documents(texts)


class _MethodEmbedding:
    """Small wrapper around a foreign object's document/query methods."""

    def __init__(self, documents: DocumentEmbedder, query: QueryEmbedder) -> None:
        self._documents = documents
        self._query = query

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self._documents(texts)

    def embed_query(self, text: str) -> Vector:
        return self._query(text)

    def __call__(self, texts: Sequence[str]) -> Sequence[Vector]:
        return self.embed_documents(texts)


def as_embedding_function(value: Any) -> EmbeddingFunction:
    """Normalize a custom callable or compatible embedding object.

    Accepted inputs are:

    * an object with ``embed_documents`` and ``embed_query`` (for example the
      standard embedding shape used by LangChain);
    * an object with LlamaIndex's ``get_text_embedding_batch`` and
      ``get_query_embedding`` methods; or
    * the original ChronoVec callable contract, accepting a batch of strings.

    No optional provider package is imported by this function.
    """
    documents = getattr(value, "embed_documents", None)
    query = getattr(value, "embed_query", None)
    if callable(documents):
        if not callable(query):
            raise TypeError("embedding object must provide callable embed_query")
        return _MethodEmbedding(documents, query)

    documents = getattr(value, "get_text_embedding_batch", None)
    query = getattr(value, "get_query_embedding", None)
    if callable(documents) and callable(query):
        return _MethodEmbedding(documents, query)

    if callable(value):
        return CustomEmbedding(value)

    raise TypeError(
        "embedding_function must be a batch callable, or provide "
        "embed_documents/embed_query methods"
    )
