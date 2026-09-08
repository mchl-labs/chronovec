# LangChain integration

ChronoVec provides a full `VectorStore` subclass for LangChain, including the branching API for speculative LCEL chains.

## Installation

```bash
pip install "chronovec[integrations]"
```

## Basic usage

```python
from langchain_openai import OpenAIEmbeddings
from chronovec.integrations.langchain import ChronoVecVectorStore

embeddings = OpenAIEmbeddings()
store = ChronoVecVectorStore(embedding=embeddings, dimensions=1536)

store.add_texts(
    ["the user prefers dark mode", "the user speaks French"],
    metadatas=[{"type": "preference"}, {"type": "language"}],
)

docs = store.similarity_search("what language does the user speak?", k=3)
docs_with_scores = store.similarity_search_with_score("dark mode", k=3)
```

## LCEL retriever

```python
retriever = store.as_retriever(search_kwargs={"k": 5})

chain = retriever | format_docs | llm | StrOutputParser()
chain.invoke("what are the user's preferences?")
```

## Branching (ChronoVec-specific)

Use branching to evaluate speculative additions without committing them to main memory:

```python
with store.branch("hypothesis") as scratch:
    scratch.add_texts(["speculative: user might prefer light mode"])
    results = scratch.similarity_search("theme preference")
    # results include the speculative doc

results = store.similarity_search("theme preference")
# speculative doc is not here: it was discarded when the context exited
```

The branch is automatically discarded if you do not call `merge()` before the context exits.

```python
with store.branch("confirmed") as branch:
    branch.add_texts(["confirmed preference"])
    branch.merge()  # promote to main

# confirmed preference is now in the main store
```

## `from_texts` classmethod

```python
store = ChronoVecVectorStore.from_texts(
    texts=["doc one", "doc two"],
    embedding=embeddings,
    dimensions=1536,
)
```

## Deletion

```python
# delete returns the number of records deleted
n = store.delete(ids=["id-1", "id-2"])
```

## API surface

`ChronoVecVectorStore` implements the full `langchain_core.vectorstores.VectorStore` interface:

- `add_texts(texts, metadatas=None, ids=None) → list[str]`
- `similarity_search(query, k=4, filter=None) → list[Document]`
- `similarity_search_with_score(query, k=4) → list[tuple[Document, float]]`
- `as_retriever(**kwargs) → VectorStoreRetriever`
- `delete(ids) → int`
- `from_texts(texts, embedding, **kwargs) → ChronoVecVectorStore` (classmethod)
- `branch(name) → contextmanager` (ChronoVec-specific)
