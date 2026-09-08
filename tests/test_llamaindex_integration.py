"""The LlamaIndex adapter against the real llama-index-core interfaces.

Duck typing is not enough: `VectorStoreIndex.from_vector_store` and the
retriever machinery type-check what they are handed, so a store with the right
method names and the wrong base cannot be plugged into an index -- which is the
only reason to ship an adapter.
"""

import numpy as np
import pytest

pytest.importorskip("llama_index.core")

from llama_index.core.schema import TextNode  # noqa: E402
from llama_index.core.vector_stores.types import (  # noqa: E402
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from chronovec.integrations.llamaindex import ChronoVecLlamaStore  # noqa: E402

DIM = 4


def store():
    return ChronoVecLlamaStore(dimensions=DIM, metric="l2")


def node(node_id, vector, text="", **metadata):
    return TextNode(id_=node_id, text=text, embedding=list(map(float, vector)), metadata=metadata)


def test_it_is_a_real_vector_store_not_a_lookalike():
    assert issubclass(ChronoVecLlamaStore, BasePydanticVectorStore)
    assert isinstance(store(), BasePydanticVectorStore)


def test_add_returns_the_ids_it_stored():
    s = store()
    ids = s.add([node("a", [1, 0, 0, 0], "alpha"), node("b", [0, 1, 0, 0], "beta")])
    assert ids == ["a", "b"]


def test_query_returns_the_real_result_type_with_ids_and_similarities():
    s = store()
    s.add([node("a", [1, 0, 0, 0]), node("b", [0, 1, 0, 0])])
    result = s.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=1))
    assert isinstance(result, VectorStoreQueryResult)
    assert result.ids == ["a"]
    assert len(result.similarities) == 1


def test_stores_text_is_false_so_llamaindex_uses_its_own_docstore():
    # query() returns ids, not reconstructed nodes. Claiming to store text
    # makes LlamaIndex expect nodes back and skip its docstore, and the
    # retriever then gets nothing.
    # Instance access, not class: on a pydantic model the fields are not class
    # attributes.
    assert store().stores_text is False


def test_delete_removes_a_node():
    s = store()
    s.add([node("a", [1, 0, 0, 0]), node("b", [0, 1, 0, 0])])
    s.delete("a")
    result = s.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=5))
    assert "a" not in result.ids and "b" in result.ids


def test_a_node_without_an_embedding_is_refused_with_a_reason():
    s = store()
    with pytest.raises(ValueError, match="embed nodes before"):
        s.add([TextNode(id_="a", text="no vector")])


def test_branching_is_the_thing_llamaindex_has_no_equivalent_for():
    s = store()
    s.add([node("a", [1, 0, 0, 0])])
    with s.branch("hypothesis") as scratch:
        scratch.add_embedding("guess", np.array([0.9, 0.1, 0, 0], dtype=np.float32))
    after = s.query(VectorStoreQuery(query_embedding=[1.0, 0.0, 0.0, 0.0], similarity_top_k=5))
    assert "guess" not in after.ids  # speculation discarded
    assert "a" in after.ids
