# streambench

A benchmark for vector indexes whose data changes.

`ann-benchmarks` measures a static index: build once, query forever. Every mature
system is tuned for it, which is exactly why it cannot separate them on the axis
that matters once documents are edited, users exercise deletion rights, or an
agent writes memories continuously.

streambench measures the other axis.

```bash
pip install streambench            # plus whichever engines you want to compare
python -m streambench --list-engines
python -m streambench \
  --dataset sift-128-euclidean.hdf5 \
  --live 100000 --epochs 5 --turnover 1.0 \
  --engines chronovec hnswlib faiss-ivfflat usearch chroma \
  --output report.json
```

## What it measures

One operation trace is generated and replayed against **every** engine, so a
difference in the report belongs to the engine and not the workload. After each
epoch the live set has changed, so exact ground truth is **recomputed against the
current live set** rather than reused from the dataset file; reusing it is the
most common way a streaming benchmark quietly measures the wrong thing.

Per epoch, per engine:

| | |
|---|---|
| `recall_at_k` | against freshly recomputed truth: does quality drift? |
| `write_ops_per_s` | delete+insert throughput |
| `latency` | mean, p50, p95, p99: does the tail degrade? |
| `stats` | engine-specific: pages, space amplification, versions reclaimed |

**A system passes by being boring.** Flat recall, flat latency, flat space,
across many complete turnovers. A system that starts fast and degrades over five
turnovers is worse than one that is slightly slower and stays put.

## Capability is part of the result

Not every index can delete. Rather than omit those engines, streambench makes
them express a turnover the only way they can (a full rebuild) and reports the
cost in the same column as everyone else's incremental update. The `del` column
says which is which. An engine that cannot delete is not disqualified; it is
measured honestly.

## Fairness notes

These matter more than the numbers, and they are easy to get wrong.

- **Batch APIs.** Engines that accept a batched call get one; engines driven
  per-item pay per-item binding overhead that has nothing to do with the index.
  An early run of this harness showed one engine at 2.8M ops/s purely because it
  received one C call for 10,000 vectors while another took 10,000 Python calls.
  Adapters should use each engine's natural bulk API.
- **Different work per operation.** An IVF append and an MVCC insert that
  versions, reclaims and maintains page structure are not the same operation.
  Throughput columns are not directly comparable without reading what the engine
  actually guarantees; that is why `stats` and the capability flags are reported
  alongside.
- **Process isolation.** Each engine runs in its own process. A native engine can
  abort outright (FAISS trips a C++ assertion on some remove paths) and a
  benchmark that dies partway through is worse than one that records the failure
  and continues.
- **Broken builds.** An adapter that cannot function should refuse rather than
  report. The annoy adapter self-checks and skips with a reason, because some
  wheels return a single neighbour for any `k`, and publishing the resulting
  zero would misrepresent annoy rather than measure it.

## Adding an engine

Implement the `Engine` protocol and register it in `ENGINES`:

```python
class MyEngine:
    name = "mine"
    can_delete = True          # can it remove without a rebuild?
    needs_rebuild = False      # must it rebuild to express a turnover?

    def __init__(self, dim, metric, live, **options): ...
    def build(self, vectors, ids): ...
    def apply(self, retired, new_ids, new_vectors, live_ids, live_vectors): ...
    def search(self, query, k) -> list[int]: ...
    def stats(self) -> dict: ...
    def close(self): ...
```

`apply` receives both the delta and the full resulting live set, so an engine
that cannot delete can rebuild from `live_ids`/`live_vectors` without the harness
special-casing it.

Pull requests adding engines are welcome, including ones that beat the engine
this harness was written alongside.

## License

Apache-2.0, matching the repository it ships from.
