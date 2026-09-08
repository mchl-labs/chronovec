# LlamaIndex integration

ChronoVec implements the LlamaIndex `VectorStore` contract.

## Installation

```bash
pip install "chronovec[integrations]"
```

## Basic usage

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from chronovec.integrations.llamaindex import ChronoVecLlamaStore

store = ChronoVecLlamaStore(dimensions=768)
storage_context = StorageContext.from_defaults(vector_store=store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("what is MVCC?")
```

## Direct node operations

```python
# Add nodes manually
nodes = [
    TextNode(text="ChronoVec uses MVCC", metadata={"topic": "architecture"}),
    TextNode(text="Bounded reclamation", metadata={"topic": "deletion"}),
]
store.add(nodes)

# Query with metadata filtering
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter

results = store.query(
    VectorStoreQuery(
        query_embedding=embedding,
        similarity_top_k=5,
        filters=MetadataFilters(
            filters=[ExactMatchFilter(key="topic", value="architecture")]
        ),
    )
)
```

## Snapshot reads (ChronoVec-specific)

```python
snapshot = store.snapshot()
# ... more nodes added ...
# query as of the snapshot
results = store.query(query, snapshot=snapshot)
```

## API surface

`ChronoVecLlamaStore` implements:

- `add(nodes: list[BaseNode]) → list[str]`
- `delete(ref_doc_id: str, **kwargs) → None`
- `query(query: VectorStoreQuery, **kwargs) → VectorStoreQueryResult`
- `snapshot() → int` (ChronoVec-specific)
