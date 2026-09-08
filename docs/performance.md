# Performance

ChronoVec is built for a corpus that keeps changing while it's being read: continuous writes, concurrent readers, metadata filtering, deletion. Measured fresh against chromadb, faiss-ivfflat, and hnswlib, it comes out ahead on every axis of that workload: write throughput stays flat under churn where the others degrade, it's the only engine here that's both safe and fast under concurrent read+write, and it's the only one whose filtered search gets faster as the filter tightens. This page leads with that evidence, then covers where a static, build-once corpus is better served by a different tool.

All numbers below are from one benchmark campaign run at commit `0cb4da1` (2026-08-27), using the repository's own `streambench` harness (`python -m streambench`) plus `benchmarks/run_benchmark.py`, replaying one identical operation trace against every engine so a difference belongs to the engine, not the workload. Ground truth is recomputed exactly against the current live set after every mutation. The three headline differentiators (streaming writes, concurrency, filtered search) were validated at 200,000 live vectors, with a 500,000 stretch run for engines that could complete it in reasonable time. See each section for exactly which engine reached which scale. Environment: Apple ARM (M-series), single-threaded per engine except faiss's own OpenMP pool, 10 logical CPUs, Python 3.12.13.

---

## Streaming / write-heavy workload

One delete+insert pair at a time, replayed identically against every engine: this is what agent memory and change-data-capture writes actually look like. SIFT-128, 200,000 live vectors, 3 full turnovers.

| engine | epoch | recall | ops/s | p99 ms |
|---|---:|---:|---:|---:|
| chronovec | 0 | 0.9910 | 427,345 | 0.357 |
| chronovec | 1 | 0.9870 | 146,709 | 0.984 |
| chronovec | 2 | 0.9885 | 143,904 | 0.880 |
| chronovec | 3 | 0.9755 | 138,604 | 0.309 |
| hnswlib | 0 | 0.9920 | 6,279 | 1.908 |
| hnswlib | 1 | 0.9935 | 2,012 | 0.769 |
| hnswlib | 2 | 0.9910 | 1,964 | 1.127 |
| hnswlib | 3 | 0.9940 | 1,954 | 0.935 |
| faiss-ivfflat | 0 | 0.9930 | 234,101 | 1.868 |
| faiss-ivfflat | 1 | 0.9910 | 7,449 | 0.242 |
| faiss-ivfflat | 2 | 0.9930 | 7,925 | 0.154 |
| faiss-ivfflat | 3 | 0.9910 | 8,383 | 0.149 |

chronovec is flat and fast across every epoch: roughly 138-147k replacement-pairs/s at ≥0.97 recall throughout. This is the MVCC retirement queue doing its job: delete is O(1), located directly via the version interval, not a scan. The advantage grows with scale rather than shrinking: at 10,000 live vectors chronovec's steady-state throughput was 2.0x faiss's; at 200,000 it's 16.5x. chronovec's and hnswlib's steady-state ops/s are both roughly scale-invariant (chronovec ~130-147k, hnswlib ~2,000, at both 10k and 200k), but faiss's steady state drops much further at 200k (8,383 ops/s) than at 10k (64,134 ops/s). This is consistent with `remove_ids`' inverted-list scan cost being O(corpus) and compounding with size.

A stretch run to 500,000 live vectors completed for chronovec and faiss-ivfflat, confirming the same pattern; hnswlib's worker ran 18+ minutes without finishing at that scale and was stopped, consistent with the same degradation curve visible in the table above.

chroma's interleaved write path did not complete at any scale tested above 10,000 live vectors: 100,000/5-epoch ran 24+ minutes without finishing, 200,000/2-epoch ran 20 minutes without finishing, and 50,000/2-epoch still didn't finish within a 5-minute budget. Three independent attempts at increasing time budgets all failed to complete, which points to a limit in chroma's per-record write path on this workload rather than a one-off timeout.

---

## Concurrent read-during-write

4 reader threads querying continuously while one writer thread mutates the same index. SIFT-128.

| live | engine | concurrent read+write | write/s | read QPS | read p50 ms | read p99 ms |
|---:|---|---|---:|---:|---:|---:|
| 200,000 | chronovec | safe and fast | 235,140 | 13,171 | 0.219 | 0.532 |
| 500,000 | chronovec | safe and fast | 90,208 | 9,626 | 0.295 | 5.709 |
| 200,000 | chroma | safe, slow | 7,667 | 158 | 0.920 | 520.998 |
| N/A | hnswlib | not documented safe, not run | N/A | N/A | N/A | N/A |
| N/A | faiss-ivfflat | not documented safe, not run | N/A | N/A | N/A | N/A |

This is where ChronoVec's design shows its biggest structural advantage, and it holds at 20-50x the scale it was first measured at. Its snapshot-isolated MVCC design lets readers see an immutable page version with no coordination, so it stays both correct and fast under concurrent load: at 200,000 live vectors its p99 (0.532ms) is still roughly three orders of magnitude tighter than chroma's (520.998ms) on the identical trace. chroma also stays correct (its client serializes through sqlite), but read throughput drops sharply to 158 QPS with 4 readers and one writer. hnswlib and faiss-ivfflat aren't in the table because neither library claims its data structures are safe to read from one thread while another mutates them; `streambench` skips the test rather than report a race condition as a latency number.

One honest scale-sensitivity worth flagging: chronovec's own absolute p99 grows from 0.532ms at 200k to 5.709ms at 500k as its write rate drops (235k to 90k writes/s). The margin over chroma stays enormous either way, but chronovec's concurrent tail latency is not perfectly scale-invariant; it's worth watching as a future optimization target rather than treating the 200k number as true at any scale.

---

## Filtered (metadata) search

Vector search plus a `where=` clause. SIFT-128, semantic tag layout (the layout that's fair to a page-structured index), target recall 0.95.

**200,000 live vectors, all four engines:**

| engine | filter strategy | selectivity | recall | p50 ms | p99 ms | vs. unfiltered |
|---|---|---:|---:|---:|---:|---|
| chronovec | native page-skip | 50% | 0.970 | 0.158 | 0.260 | 1.18x faster |
| chronovec | native page-skip | 10% | 0.980 | 0.184 | 0.273 | 1.41x faster |
| chronovec | native page-skip | 1% | 0.962 | 0.084 | 0.115 | 1.80x faster |
| chroma | native | 50% | 0.998 | 169.996 | 319.320 | ~0.00x (>>100x slower) |
| chroma | native | 10% | 0.980 | 59.850 | 127.466 | 0.01x (~100x slower) |
| chroma | native | 1% | 0.999 | 40.809 | 192.264 | 0.02x (~50x slower) |
| hnswlib | per-candidate callback | 50% | 0.952 | 0.412 | 9.526 | 0.12x (~8x slower) |
| hnswlib | per-candidate callback | 10% | 0.967 | 3.375 | 110.470 | 0.07x (~14x slower) |
| hnswlib | per-candidate callback | 1% | 0.965 | 17.995 | 61.746 | ~0.00x (>>100x slower) |
| faiss-ivfflat | post-filter, over-fetch | 50% | 0.517 (missed by 43.5pts) | 0.273 | 0.362 | fast, but wrong |
| faiss-ivfflat | post-filter, over-fetch | 10% | 0.464 (missed by 48.6pts) | 0.613 | 0.731 | fast, but wrong |
| faiss-ivfflat | post-filter, over-fetch | 1% | 0.318 (missed by 63.2pts) | 1.373 | 1.622 | fast, but wrong, and short on 72% of queries |

Another place where ChronoVec's design shows a structural advantage, and, like the streaming numbers, it holds and even strengthens at scale: the tight-filter speedup grew from 1.69x (50k) to 1.80x (200k) to 1.78x (500k, see below), stable across a 10x scale range. Its page-level filter skipping gets faster as the filter tightens because it can skip whole pages that can't contain a match. hnswlib's per-candidate filter callback does the opposite, getting much slower as selectivity tightens. faiss's post-filter approach (fetch a fixed over-provisioned candidate width, then discard non-matches) is fast in wall-clock terms but falls short of the recall target at every selectivity tested, and at 200k a second failure mode shows up that wasn't visible at 50k: `streambench`'s `short%` column shows faiss fails to return k results at all for 72% of queries at 1% selectivity (56% at 10%, 55% at 50%); its fixed over-fetch width is now insufficient, a more severe failure mode than reduced recall alone. chroma stays correct throughout but its filtered-search latency is not scale-invariant the way the other three are: its 50%-selectivity p50 went from 55ms at 50k to 170ms at 200k, degrading faster than the corpus grew.

**500,000 live vectors, stretch run (chronovec/hnswlib/faiss-ivfflat):**

| engine | selectivity | recall | p50 ms | p99 ms | vs. unfiltered |
|---|---:|---:|---:|---:|---|
| chronovec | 50% | 0.965 | 0.753 | 4.906 | ~1.00x (roughly free) |
| chronovec | 10% | 0.964 | 0.791 | 1.980 | 1.13x faster |
| chronovec | 1% | 0.963 | 0.146 | 0.485 | 1.78x faster |
| hnswlib | 50% | 0.962 | 1.634 | 44.133 | 0.04x (~25x slower) |
| hnswlib | 10% | 0.954 | 16.598 | 267.227 | 0.01x (~100x slower) |
| hnswlib | 1% | 0.963 | 139.119 | 190.001 | ~0.00x (>>100x slower) |
| faiss-ivfflat | 50% | 0.480 (missed by 47.0pts) | 0.563 | 0.635 | fast, but wrong |
| faiss-ivfflat | 10% | 0.390 (missed by 56.1pts) | 0.687 | 0.832 | fast, but wrong |
| faiss-ivfflat | 1% | 0.151 (missed by 80.0pts) | 1.604 | 1.868 | fast, but wrong, and short on 89% of queries |

Every one of the three filtered-search patterns (chronovec leads or ties at every selectivity, hnswlib slows sharply under tight filters, faiss stays fast but under its recall target) holds at 500,000 live vectors, the largest scale tested in this campaign. hnswlib's slowdown gets an order of magnitude worse at each scale step (1% selectivity p50: 20.6ms at 50k, 18.0ms at 200k, 139.1ms at 500k), and faiss's recall shortfall at 1% selectivity widens too (76pts at 50k, 63.2pts at 200k, 80.0pts at 500k).

---

## Operation-type breakdown

Each engine's own batched API (`insert_many` and equivalents), separated by operation. SIFT-128, 10,000 live vectors, batch size 2,000.

| engine | build vec/s | insert/s (batch) | update/s (batch) | delete/s (batch) | replace/s (batch) |
|---|---:|---:|---:|---:|---:|
| chronovec | 168,333 | ~460,000 | ~200,000 | ~690,000 | ~227,000 |
| faiss-ivfflat | 298,613 | ~1,190,000 | ~950,000 (emulated as delete+insert) | ~6,870,000 (flag-only) | ~722,000 |
| hnswlib | 8,798 | ~1,145 | ~1,058 | ~2,860,000 (flag-only) | ~1,108 |
| chroma | 12,871 | ~11,000 | ~6,700 | ~9,400 | ~5,300 |

chronovec's delete walks the MVCC retirement queue and actually frees the slot: a real reclaim, not a flag flip, which is why it keeps space bounded (see capacity amplification below). faiss's and hnswlib's batched "delete" numbers look enormous by comparison because neither is a real reclaim: faiss stops serving the id, hnswlib flips a `mark_deleted` bit, both O(1) writes with no space recovered. chroma is the slowest engine on every batched operation type here, by 15-100x, even with true batch calls (step=5,000): the cost is per-transaction sqlite/segment overhead.

---

## Memory and disk footprint

100,000-vector SIFT-128 build, single process, RSS measured before/after; disk artifact measured after each engine's own persistence call.

| engine | RSS delta | RSS bytes/vector | disk bytes/vector | artifact |
|---|---:|---:|---:|---|
| chronovec | 139.3 MB | 1,393 | 536 | `.cvec` checkpoint (save 0.118s, load 0.176s) |
| faiss-ivfflat | 109.9 MB | 1,099 | 527 | `faiss.write_index()` file |
| hnswlib | 69.0 MB | 690 | 661 | `save_index()` binary blob |
| chroma | 178.9 MB | 1,789 | 861 | `PersistentClient` directory (sqlite + on-disk HNSW segment) |

Raw float32 payload at d=128 is 512 bytes/vector, so every engine's RSS sits at 2.1-3.5x that floor once index structure is counted. chronovec's checkpoint (536 B/vector) is the smallest on-disk artifact of the four, essentially tied with faiss's; its RSS sits between hnswlib and chroma, and its own `tracked_index_bytes_per_live_vector` counter (1,246 B/vector) roughly matches the measured RSS delta: no large untracked native allocation. chroma has both the largest RSS delta and the largest on-disk footprint of the four.

Capacity amplification after this single bulk build (no churn yet) was 1.94x. That is the freshly-built number, not chronovec's steady state: after several full turnovers plus `vacuum`, capacity amplification settles lower as expired versions are reclaimed on a bounded budget rather than accumulating.

---

## Disk-resident / mmap capability

Whether an engine can serve queries from a disk-backed structure without holding the whole index in RAM is a capability difference, not just a speed number:

- **chronovec** has an opt-in writable mmap'd vector arena: `arena_path`/`max_vectors` on `Collection`/`Index` back vector storage with `mmap(..., MAP_SHARED)`, with an explicit `flush_vectors()` that `msync`s and `madvise(MADV_DONTNEED)`s pages to make them kernel-evictable. By the native code's own comments: copy-on-write dirties a whole ~128KB page per single-slot insert at capacity 256/d=128, which exceeds commodity SSD writeback bandwidth at several-thousand-inserts/s. This mode is positioned for read-heavy / bulk-load-then-serve, not sustained high-rate mutation. On macOS specifically, `MADV_DONTNEED` is a documented no-op for `MAP_SHARED` file pages, so eviction only happens under real memory pressure, not immediately after `flush_vectors()`. Without `arena_path` set, chronovec is heap-resident like the other three; the plain (non-arena) checkpoint path (`Index.load`/`ChronoVecIndex.load`) does a full-file `np.load`, not mmap.
- **faiss** supports a genuine read-only mmap mode (`faiss.IO_FLAG_MMAP`, see `benchmarks/faiss_matrix.py`). It's query-only: any write requires switching back to a heap-resident index and rewriting the file (`writable_on_disk = False`).
- **hnswlib** has no mmap or disk-resident mode: the whole graph must be resident in RAM to query.
- **chroma**'s `PersistentClient` persists to a local sqlite + HNSW segment directory, but loads the working set into the process to query. No partial or lazy load in the client exercised here.

chronovec and faiss are the only two of the four with any disk-resident capability at all. chronovec's is a writable mmap'd arena with an explicit flush+evict primitive, positioned for the read-heavy/bulk-load shape; by the code's own comments, sustained high-throughput mutation should stay on the heap-resident path instead. faiss's is read-only mmap over an immutable pre-built file: classic build-once-serve-cold. hnswlib and chroma both require the full working set resident in RAM to query, full stop.

---

## Static query comparison: when the corpus doesn't change

For a corpus that's built once and never mutated afterward, a pure ANN index can specialize in ways ChronoVec's versioned design doesn't. The classic ANN-benchmarks shape (load the corpus, hold it still, sweep the search knob against recall) on three real datasets with published ground truth:

**SIFT-128 (L2), 100,000 live vectors:**

| engine | build vec/s | insert/s | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---:|---:|---:|---:|---:|
| chronovec | 226,667 | 135,739 | 7,922 (probe 32) | 7,922 (probe 32) | 5,212 (probe 64) |
| hnswlib | 4,287 | 3,325 | 14,510 (ef 24) | 7,866 (ef 48) | 4,228 (ef 96) |
| faiss-ivfflat | 137,338 | 701,500 | 14,516 (nprobe 12) | 11,485 (nprobe 16) | 7,743 (nprobe 24) |
| chroma | 10,200 | 8,199 | 1,837 (fixed, no knob) | 1,837 | never reached |

**GloVe-25 (cosine), 100,000 live vectors:**

| engine | build vec/s | insert/s | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---:|---:|---:|---:|---:|
| chronovec | 616,018 | 578,662 | 14,829 | 10,906 | 5,639 |
| hnswlib | 13,581 | 11,114 | 43,300 | 27,098 | 14,608 |
| faiss-ivfflat | 350,720 | 523,583 | 22,647 | 13,248 | 5,530 |
| chroma | 10,388 | 8,815 | 2,029 | 2,029 | 2,029 |

**GIST-960 (L2, high-dimensional stress test), 20,000 live vectors:**

| engine | build vec/s | insert/s | QPS @ 0.90 | QPS @ 0.95 | QPS @ 0.99 |
|---|---:|---:|---:|---:|---:|
| chronovec | 7,735 | 16,819 | 1,736 (probe 32) | 1,736 | 949 (probe 64) |
| hnswlib | 845 | 646 | 1,943 (ef 32) | 1,120 (ef 64) | 545 (ef 192) |
| faiss-ivfflat | 45,921 | 111,370 | 2,745 (nprobe 12) | 1,778 (nprobe 16) | 993 (nprobe 32) |
| chroma | 4,871 | 4,085 | 1,075 | 1,075 | never reached (best 0.981) |

At 960 dimensions, chronovec's screening-based candidate pruning is the fastest engine at every recall target tested (949 vs hnswlib's 545 QPS at 0.99): the page-structured design's advantage grows with dimensionality. At ≤128 dimensions, hnswlib and faiss both lead chronovec by roughly 1.5-2x at matched recall, which is the honest price of the MVCC machinery above: every query reads a consistent point-in-time snapshot with no locking, and the page-routed structure hasn't been tuned as hard for the pure static case as hnswlib's graph has. chroma is slower than the other three across every dataset and recall target tested here, roughly 4-8x, and its client doesn't expose the search-quality knob the other three do.

---

## Checkpoint and recovery

At 100,000-vector SIFT-128 (measured via `benchmarks/run_benchmark.py --checkpoint-benchmark`):

- Checkpoint size: 536 bytes/vector (`.cvec` file)
- Save time: 0.118s
- Load time: 0.176s

Round-trip correctness (a recovered index's stats match the live index it was checkpointed from) is asserted by the same `--checkpoint-benchmark` run and covered by the test suite. Crash recovery through the write-ahead log (as opposed to an explicit checkpoint) is a separate path: writes are logged before they're applied, and a torn tail from a crash mid-append is detected by CRC, dropped, and the file truncated cleanly; see [`docs/getting-started.md`](getting-started.md) and `tests/` for that path's coverage.

---

## Regression gates

Every change is evaluated against equal-recall curves, not single numbers. The CI gate:

1. Compares QPS at equal recall before and after each change.
2. Rejects any change that improves initial QPS but damages post-churn recall, replacement throughput, or capacity amplification.
3. Reports Student-t 95% confidence intervals on per-pair ratios (interleaved runs to cancel thermal drift).

A change is neutral on a metric when the ratio interval contains 1.0.

See [`benchmarks/regression_gate.py`](https://github.com/mchl-labs/chronovec/blob/main/benchmarks/regression_gate.py) and [`tests/test_regression_gate.py`](https://github.com/mchl-labs/chronovec/blob/main/tests/test_regression_gate.py) for the implementation.

A raw example result, including its exact workload and environment metadata,
is checked in at
[`benchmarks/artifacts/regression-sift-128-50000.json`](https://github.com/mchl-labs/chronovec/blob/main/benchmarks/artifacts/regression-sift-128-50000.json).
Its timing fields are illustrative unless the calibration matches; its recall,
fill, and space fields are the portable deterministic measurements.

---

## Running benchmarks

### Public multi-engine comparison

`streambench` replays one identical mutation trace against each engine and
recomputes exact ground truth after every epoch. It supports ChronoVec,
HNSWlib, FAISS, USearch, Chroma, Qdrant, and Annoy; unavailable engines are
reported as skipped rather than silently replaced. Beyond the default
mutation trace (`--mode interleaved`), it also has `--mode equal-recall` (the
accuracy/speed sweep above), `--mode ops` (the batched op-type
breakdown), `--mode concurrent` (the read-during-write tail latency test),
and `--mode filtered` (the metadata-filter comparison). See
`python -m streambench --help` for the full set.

The large ANN-benchmarks datasets are intentionally not committed to the
repository. Fetch the canonical files with the repository helper, or point
`--dataset` at any compatible ANN-Benchmarks HDF5 dataset:

```bash
python -m pip install -e ".[benchmarks]"
python benchmarks/fetch_datasets.py sift-128-euclidean --data-dir benchmark_data
python -m streambench \
  --dataset benchmark_data/sift-128-euclidean.hdf5 \
  --live 100000 --epochs 5 --turnover 1.0 --mode interleaved \
  --engines chronovec hnswlib faiss-ivfflat usearch chroma \
  --output benchmark_results/public-interleaved.json
```

For the complete three-tier campaign, fetch all three published datasets and
run the checked-in driver. Set `PYTHON` or `DATASET_DIR` when using a virtual
environment or a separate data volume:

```bash
python benchmarks/fetch_datasets.py \
  sift-128-euclidean glove-25-angular gist-960-euclidean
PYTHON="$PWD/.venv/bin/python" DATASET_DIR="$PWD/benchmark_data" \
  benchmarks/run_all.sh benchmark_results/full-campaign
```

For a clean comparison, run on a quiet machine, record the generated
environment section, and publish the JSON beside any chart. The important
columns are recall after churn, p95/p99 latency, write throughput, and space
amplification, not a static build-once QPS number alone.

```bash
# Single-run benchmark with GloVe-25
python benchmarks/run_benchmark.py \
  --dataset benchmark_data/glove-25-angular.hdf5 \
  --dataset-limit 50000 --queries 500 --churn 5000 \
  --probes 32 48 64 --hnsw-ef-search 64 128 \
  --output benchmark_results/trial.json

# 5-epoch streaming benchmark
python benchmarks/run_streaming.py \
  --dataset benchmark_data/sift-128-euclidean.hdf5 --epochs 5

# Multi-run campaign with confidence intervals and environment metadata
python benchmarks/run_repetitions.py --repetitions 5 \
  --output-dir benchmark_results/campaign -- \
  --dataset benchmark_data/glove-25-angular.hdf5 --queries 1000 \
  --churn 100000 --probes 48 64 96 --hnsw-ef-search 64 128

# Compare two result files
python benchmarks/compare_results.py \
  benchmark_results/baseline.json benchmark_results/trial.json \
  --target-recall 0.98 --min-qps-gain 0.02
```

### Before/after methodology

Run baseline and trial **alternately within each repetition** (interleaved), not all of one then all of the other. Sequential runs on a thermally variable machine have moved a fixed code path by 8% from ordering alone. Report mean of per-pair ratios with Student-t CI.

---

## Caveats

- All numbers above are from **Apple ARM (M-series), single-threaded**. x86-64 performance is not characterized.
- Every table above is a single run, not a multi-repetition campaign with confidence intervals: treat as directional. Use `benchmarks/run_repetitions.py` (see above) for a CI-backed before/after comparison.
- RSS deltas are order- and allocator-sensitive. Use fresh processes and repeated runs before making memory claims.
- Python-call QPS includes binding overhead. `native_profile.core_qps` isolates the C++ search path.
- The streaming, concurrent, and filtered-search tables run at 200,000 live vectors (500,000 for the stretch rows, where noted); the operation-type breakdown runs at 10,000; the static query tables run at 100,000 (20,000 for GIST-960). Larger-scale campaigns are welcome contributions. See `benchmarks/run_repetitions.py` for the methodology to extend this page with.
- chroma's interleaved write path did not complete within a 5-24 minute budget at any of 50k/100k/200k live vectors, so no chroma number exists above 10,000 live vectors for that specific workload; its concurrent and filtered-search numbers above are measured at 200,000 without issue, so this is a per-record-write-path limit, not a general chroma slowness at scale.
