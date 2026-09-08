# Getting started

## Requirements

- Python 3.11+
- C++20 compiler (clang 14+ or GCC 13+)
- CMake 3.18+
- NumPy 1.26+

## Installation

### From PyPI

```bash
pip install chronovec
```

Pre-built wheels cover CPython 3.11-3.13 on Linux x86-64, macOS arm64, and Windows x86-64. If a wheel is not available for your platform, `pip` builds the native library from source, which requires C++20 and CMake.

### Quickstart

Start with an in-memory collection for easy prototyping:

```python
from chronovec import Collection

collection = Collection(dimensions=3)
collection.add(
    ids=["doc1", "doc2"],
    embeddings=[[1, 0, 0], [0, 1, 0]],
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],
)

results = collection.query(
    [1, 0, 0],
    k=2,
    # where={"source": "notion"},
    # snapshot=collection.snapshot(),
)
```

### Text embedding

ChronoVec accepts a batch embedding callable for backwards compatibility, or
the provider-neutral `CustomEmbedding` adapter when document and query text
need different model instructions:

```python
from chronovec import Collection, CustomEmbedding

embedder = CustomEmbedding(
    embed_documents=my_embed_documents,
    embed_query=my_embed_query,
)
collection = Collection(dimensions=384, embedding_function=embedder)
collection.add(ids=["doc1"], documents=["This is document1"])
results = collection.query(query_text="find document one", k=1)
```

Compatible objects with `embed_documents`/`embed_query` are accepted directly.
LlamaIndex-style `get_text_embedding_batch`/`get_query_embedding` objects are
also recognized without making either provider a ChronoVec dependency.

For a local Sentence Transformers model, install the optional adapter:

```bash
pip install "chronovec[sentence-transformers]"
```

```python
from chronovec import Collection, SentenceTransformerEmbedding

embedding = SentenceTransformerEmbedding("sentence-transformers/all-MiniLM-L6-v2")
collection = Collection(384, embedding_function=embedding)
```

For hosted providers, ChronoVec wraps [LiteLLM](https://docs.litellm.ai/) so
one adapter covers OpenAI, Cohere, Bedrock, Azure, and the rest of LiteLLM's
provider list:

```bash
pip install "chronovec[litellm]"
```

Pass the provider secret explicitly from your environment:

```python
import os

from chronovec import Collection, ProviderEmbedding

embedding = ProviderEmbedding(
    provider="openai",
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
collection = Collection(1536, embedding_function=embedding)
```

Add persistence with a `Client`, which opens (or creates) named collections
that checkpoint themselves to disk after every mutation, no separate save
call:

```python
from chronovec import Client

client = Client("./data")  # omit the path for "./.chronovec"
collection = client.get_or_create_collection("docs", dimensions=3)
collection.add(ids=["doc1"], embeddings=[[1, 0, 0]], documents=["This is document1"])
```

Reopen the same directory with `Client("./data")` and call
`get_collection("docs")`. `get_collection`, `list_collections`, and
`delete_collection` are also available. `PersistentClient` is an explicit
alias for teams that prefer the longer name. For local inspection:

```bash
chronovec list ./data
chronovec inspect ./data docs --json
```

### Building from source

The single-step install: `pip install .` triggers cmake automatically via scikit-build-core, compiles the native library, and installs the Python package in one shot.

```bash
git clone https://github.com/mchl-labs/chronovec
cd chronovec
pip install ".[dev]"       # build + install + dev dependencies
```

For in-place development (editable install, rebuilds on source change):

```bash
pip install -e ".[dev]"
```

Alternatively, build with cmake directly and use the build directory:

```bash
cmake --preset release     # configure (uses CMakePresets.json)
cmake --build build -j     # compile
pip install -e ".[dev]"    # install Python package pointing at build/
```

Or use `make`:

```bash
make dev
```

### Verifying the installation

```bash
python -c "from chronovec import Index; print('ok')"
python -m pytest tests/ -q
```

### Optional extras

```bash
pip install "chronovec[integrations]"  # LangChain + LlamaIndex
pip install "chronovec[duckdb]"        # DuckDB UDF adapter
```

---

## 3-minute tour

### Insert, search, delete

```python
from chronovec import Index

index = Index(384, metric="cosine", page_capacity=256, nprobe=16)

# Insert returns a commit timestamp
t1 = index.insert(1, embedding_a)
index.insert(2, embedding_b)

# Search returns SearchResult(id, distance) objects
for hit in index.search(query, k=5):
    print(hit.id, hit.distance)

# Delete marks the record expired
index.delete(1)

# Vacuum physically reclaims up to budget_versions expired records
index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=64)
```

### Time travel: query the past

```python
t1 = index.insert(1, embedding)
index.delete(1)

index.search(query, k=5)               # id 1 is gone
index.search(query, k=5, snapshot=t1)  # id 1 is visible
```

### Persistence

```python
index.save("my_index.cvec")                        # atomic, checksummed
restored = Index.load("my_index.cvec")

# With crash recovery via write-ahead log
index = Index(384, wal_path="index.wal")
```

### String IDs and metadata filtering

```python
from chronovec import Collection

coll = Collection(dimensions=768, metric="cosine")
coll.add(
    ids=["doc-1", "doc-2"],
    embeddings=[v1, v2],
    metadatas=[{"lang": "en", "year": 2024}, {"lang": "fr", "year": 2023}],
    documents=["first doc", "second doc"],
)

results = coll.query(q, k=5, where={"lang": "en"})
results = coll.query(q, k=5, where={"year": {"$gte": 2024}})
results = coll.query(q, k=5, where={"$or": [{"lang": "en"}, {"lang": "fr"}]})
```

### Agent memory with branching

```python
from chronovec import AgentMemory

memory = AgentMemory(384)
memory.add("fact-1", base_embedding, text="established fact")  # str or int ids

branch = memory.branch("hypothesis")
branch.add("spec-1", spec_embedding, text="speculative")

for hit, record in branch.search(query):   # sees both; returns (SearchResult, Record)
    print(hit.distance, record.id, record.payload)
memory.search(query)   # sees only established fact
branch.discard()       # abandons speculative writes; purge later reclaims versions
```

### Collection persistence: checkpoint, disk-backed arena, async

```python
# Checkpoint / restore: native index + id/metadata sidecar
coll.save("snapshot")                                    # snapshot.cvec + snapshot.meta
restored = Collection.load("snapshot", embedding_function=my_embed)

# Disk-backed arena that grows automatically when it fills
coll = Collection(768, wal_path="mem.wal", arena_path="mem.arena",
                  max_vectors=100_000, growth_factor=2.0)

# asyncio / FastAPI: same API, every method awaited
from chronovec import AsyncCollection

acoll = AsyncCollection(768, embedding_function=my_embed)
results = await acoll.query(query_text="...", k=5)
```

> **Claude Code users:** a skill file at `.claude/skills/chronovec/chronovec/SKILL.md` teaches coding agents the full API: snapshot timing, branch lifecycle, filter syntax, and footguns. It activates automatically when relevant, or invoke it with `/chronovec`.

---

## Build options

| CMake flag | Effect |
|---|---|
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build (default for `make build`) |
| `-DCHRONOVEC_SANITIZE_ADDRESS=ON` | ASan + UBSan (development) |
| `-DCHRONOVEC_SANITIZE_THREAD=ON` | ThreadSanitizer (development) |
| `-DCHRONOVEC_BUILD_FUZZER=ON` | libFuzzer harness |
| `-DCHRONOVEC_USE_BLAS=ON` | BLAS for batched inserts (auto-detected on macOS) |

## Platform notes

- **macOS (Apple Silicon):** Fully supported and performance-tuned. Uses Apple Accelerate for BLAS.
- **Linux (x86-64):** Built and tested in CI. Performance numbers are from ARM; x86-64 numbers are a roadmap item.
- **Windows:** Native library builds in CI (MSVC). Python tests pass. Performance not characterized.
