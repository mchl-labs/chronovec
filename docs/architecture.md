# Architecture

ChronoVec is built around one central guarantee: a reader never blocks a writer, a writer never blocks a reader, and every read is consistent: no partial writes, no phantom deletes.

This document explains how the design achieves that guarantee and what the tradeoffs are.

---

## Core model: page-structured MVCC

The index is a collection of **pages**. Each page holds a fixed number of vectors and their metadata. Pages are immutable once published: a write clones the affected page, modifies the clone, and atomically publishes the new version via compare-and-swap. Readers hold a reference to the page they are reading; the old version lives until all references to it are dropped.

Every slot (vector entry) carries a `[begin_ts, end_ts)` interval:
- `begin_ts`: the timestamp at which this version was inserted
- `end_ts`: the timestamp at which it was logically deleted (or `∞` if still live)

A query at snapshot `t` sees exactly the slots where `begin_ts ≤ t < end_ts`. This is standard MVCC: the same model used by Postgres and SQLite, applied to an ANN index.

---

## Lock-free reads

Readers operate without any lock:

1. Acquire the current committed timestamp (`cv_clock`).
2. Snapshot the immutable directory (a pointer to an array of page descriptors).
3. For each page consulted, dereference its current descriptor: an atomic load.
4. Read vector data from the shared payload buffers.

Writers clone, modify, and CAS-publish. If the CAS fails (another writer beat them), they retry. Directory splits and merges, the ID-to-version map, and operation ordering use a single writer mutex, but that mutex is never held during a read path.

The correct claim is **lock-free reads with CAS-published pages**, not lock-free multiwriter mutation. Writers serialize.

---

## Two-level routing

With fewer than 16,384 pages, ChronoVec uses **linear centroid ranking**: score all page centroids against the query and take the top-`nprobe`. This is faster than graph traversal at small scale and avoids early recall cliffs.

Past 16,384 pages, queries traverse a **bounded-degree routing graph** over centroids, starting from four entry points and evaluating a bounded frontier of `max(64, 6·nprobe)` candidates. The crossover is measured, not aesthetic: on a 2,792-page index, linear scoring averaged 20.25k QPS versus 10.67k for graph traversal at identical recall.

The routing graph is built incrementally. Splits reconnect only replaced/new nodes; merges remap adjacency. Centroid descriptors are stored in a contiguous immutable array, refreshed by a rotating 1/64 partition update per split.

---

## Compressed scoring (residual quantization)

Each page stores **4-bit residual codes** alongside the float vectors:

- A per-page centroid is subtracted from each vector before quantization.
- Each component is packed to 4 bits with a per-vector peak scale.
- The reconstructed distance is `<c, u_q> / <c, u>`, debiased.

This gives 72 bytes per vector at 128 dimensions versus 512 for float32. Recall is identical to 8-bit; the win is memory, not speed.

A search over a page proceeds:
1. Score all visible vectors with the 4-bit codes (fast, cache-friendly).
2. Keep the top candidate set.
3. Exactly rerank the candidate set using the full float vectors.

Step 1 is the **screening** phase, step 3 is **reranking**. On a 20k/64d workload:

| Phase | avg candidates | avg time |
|---|---:|---:|
| Routing | 135 centroids | 4.8 µs |
| Int8 screening | 4,757 visible vectors | 21.0 µs |
| Exact reranking | 40 vectors | 3.0 µs |

---

## Bounded reclamation

`vacuum(oldest_snapshot, budget_versions)` is the garbage collector:

1. Walk the ordered retirement queue for versions with `end_ts < oldest_snapshot`.
2. Reclaim at most `budget_versions` entries per call.
3. After reclamation, trigger **consolidation**: merge underfull pages toward the count the occupied slots justify, with headroom so the next insert does not immediately split again.

Three invariants hold space flat:
- **Groups, not pairs:** merge a destination page from as many near-source pages as fit (pairwise merging cannot reach the fill factor the capacity allows).
- **Drop empty pages:** a page with no occupied slots is removed from the directory (the actual leak in early versions).
- **Bulk-load consolidation:** a large batch consolidates itself after finishing, because vacuum may never be called on a freshly built index.

Together these hold capacity amplification at 1.58x across 30 full turnovers on SIFT-128 at 1M vectors.

---

## Durability: write-ahead log

The WAL is optional and off unless you pass `wal_path=`. When enabled:

1. Records are length-prefixed and CRC'd before they are applied.
2. A torn tail from a crash mid-append is detected by CRC, dropped, and the file truncated.
3. Recovery replays the log from the beginning; replay reproduces the same
   committed insert-batch visibility a live reader sees.

The WAL composes with the disk-backed arena: the arena is a spill file (reconstructed from the log on recovery), not a record: the log is the canonical source of truth.

---

## Batch atomicity

`insert_many` gets a single commit timestamp, published once after every page
is updated. A concurrent reader that samples the clock mid-batch sees the
insert batch entire or not at all. `delete_many` currently applies its row
deletions one at a time, so callers needing a stable view across a delete batch
must pin a snapshot before starting it. With per-row timestamps, a 4,000-row
insert had nine of twelve concurrent readers observing between 315 and 658 rows,
showing partial batches. That is not a correctness property ChronoVec offers for
`insert_many`.

## Snapshot reads are not rollback

An MVCC snapshot is a read boundary, not a command that rewinds the live index.
`search(..., snapshot=t)` answers what was visible at committed timestamp `t`;
it does not remove writes made after `t`, restore the current index, or undo a
WAL record. A checkpoint is likewise a durable image to load later, not an
in-place rollback operation.

For speculative writes, use `AgentMemory.branch(name, snapshot=t)`. The branch
starts from the historical read boundary and keeps later writes isolated from
main until the caller explicitly merges it. Discarding the branch abandons its
logical writes; it does not rewrite the main timeline. `merge()` promotes the
branch's records to main. `AgentMemory` holds a high-level state gate across
that promotion, so its readers observe either the pre-merge branch state or the
complete post-merge state, never a partially promoted branch. This is atomic
visibility, not a general rollback transaction: a failed merge does not rewind
native mutations already accepted by the engine, and callers should retry from
a known snapshot or reload the last complete checkpoint.

## Checkpoint and WAL boundaries

The core `Index`/`Collection` WAL is an optional crash-recovery log. A WAL record
is flushed before its mutation is applied; recovery replays complete records and
drops a torn final record. WAL durability therefore covers the core native
index, subject to the configured filesystem and `wal_sync` policy. It does not
make arbitrary Python-side metadata or multiple unrelated files one transaction.

`AgentMemory.save()` addresses that boundary with a generation-specific native
checkpoint, a checksummed metadata sidecar, and a manifest published last.
`AgentMemory.load()` selects only a complete matching generation. With
`checkpoint_path=`, successful memory and branch mutations trigger this
checkpoint path. It is a durable restart point, not an implicit WAL-backed
rollback log.

The AgentMemory metadata sidecar uses Python pickle because payloads may contain
arbitrary Python values. Checkpoint files must therefore be treated as trusted
input: the manifest checksum provides integrity/matching protection, not safe
execution of an untrusted pickle.

---

## Known design limitations

| Limitation | Why | Roadmap |
|---|---|---|
| Single writer | Writers serialize on the MVCC directory lock. Lock-free reads, not multiwriter mutation. | Multi-writer is a research-scope item. |
| Pure-Python reference remains available | `ChronoVecIndex` (index.py) is a reference implementation, slower and without native SIMD. It remains available for tests and portability, but is not the recommended production backend. | Keep internal in future major API cleanup. |
| DuckDB adapter not thread-safe | Minimal test coverage, Python GIL dependency. | Promote to stable after test parity. |
| Async is wrapper-based | `AsyncCollection` uses `asyncio.to_thread()`; it preserves the synchronous API contract but is not a native async engine. | Keep the wrapper stable; add native async only if profiling demonstrates a need. |
| No HTTP/REST mode | Library only. | A FastAPI `examples/server.py` is a natural contribution. |
| x86-64 performance is not a release claim | AVX2 kernels are enabled when the compiler supports them, but the published performance campaign is Apple ARM only. | Add representative x86 measurements before making x86 speed claims. |

The support levels are intentionally asymmetric. Python's native `Index`,
`Client`, `Collection`, and the C ABI are stable compatibility surfaces. Rust,
Node, and Go remain supported alpha bindings with release CI; SQLite,
LangChain, and LlamaIndex are integration surfaces. DuckDB remains
experimental. A broad API surface helps adoption, but it does not mean every
adapter has the same compatibility or performance guarantees.

---

## Performance improvement proposals

These proposals are grounded in the current C++ implementation (`bindings/rust/chronovec-sys/native/src/chronovec.cpp`, `bindings/rust/chronovec-sys/native/src/distance.cpp`). Each one names the exact bottleneck, its projected impact, and an explicit non-degradation analysis, because a change that wins on one axis while hurting another is not an improvement.

---

### P1: AVX2 and AVX-512 SIMD for x86-64 distance kernels

**Status.** Shipped. `distance.cpp` has ARM NEON paths and AVX2 paths for the four hot kernels (`dot`, `l2sq`, `int8_dot`, `int4_dot`). CMake enables AVX2 when the compiler supports it; builds without AVX2 retain the scalar fallback.

The screening hot-loop calls `int4_dot` on every visible vector in every probed page. At 128 dimensions the scalar path runs 128 multiply-adds per call; AVX2 processes 32 int16 lanes per instruction, reducing instruction count for the same operation.

**Change.** The implementation uses `#ifdef __AVX2__` paths in `distance.cpp`:
- `int4_dot`: unpack nibbles to int8 with `_mm256_and_si256` / shift, accumulate with `_mm256_maddubs_epi16` into int32, horizontal sum.
- `int8_dot`: directly `_mm256_maddubs_epi16` on int8×int8; identical layout.
- `dot` and `l2sq`: `_mm256_fmadd_ps` on float32 (straightforward 8-wide FMA).

Gate on `__AVX2__` at compile time and `cpuid` at runtime to avoid illegal instruction faults on deployment targets that differ from the build machine.

**Projected impact.** Screening throughput ×3-8× on x86-64 (typical Linux server). Query latency falls by roughly `screening_ns / total_query_ns`, which from the profiling table above is 21 µs out of ~29 µs, a significant fraction of total query time. **Recall is unaffected**: the kernel computes the same value as the scalar path, merely faster.

**Non-degradation analysis.**
- Recall: unchanged (same math, different instruction count).
- Insert throughput: unchanged (inserts do not call `int4_dot`).
- MVCC correctness: unchanged (no data structures modified).
- Memory: unchanged (no extra storage).
- ARM/macOS: unchanged (NEON paths remain active, the AVX2 block is `#ifdef`-gated and unreachable).

**Validation.** The recall regression gate and cross-platform build matrix cover the implementation. A representative x86 performance campaign is still pending, so AVX2 is an opt-in portability/performance capability rather than a published speed claim. Enable it only when the deployment fleet is known to support AVX2; runtime dispatch is future work.

---

### P2: O(1) free-slot allocation via bitset

**Bottleneck.** `Page::free_slot()` scans the `occupied[]` array linearly to find the first unused slot. `occupied` is `bool[capacity]` with capacity up to 256: 256 iterations per insert, 256 cache-line reads in the worst case. This is called on every insert and on every page split.

**Change.** Replace `bool occupied[capacity]` with a `uint64_t occupied_bits[4]` bitset (256 bits for capacity ≤ 256). `free_slot()` becomes:

```cpp
for (int word = 0; word < 4; ++word) {
    if (~occupied_bits[word]) {
        return word * 64 + __builtin_ctzll(~occupied_bits[word]);
    }
}
return -1;  // full
```

This is 4 iterations and a hardware count-trailing-zeros instruction regardless of capacity. Setting and clearing a slot is a single bit-OR or bit-AND.

**Projected impact.** Reduces `free_slot()` from O(capacity) = O(256) to O(1). Insert throughput improvement is modest on single-thread (dominated by vector copy and BLAS assignment), but significant under concurrent workloads where the writer mutex is held longer; shorter critical-section means higher contention headroom. Page splits, which call `free_slot()` once per slot during reconstruction, see a 64× reduction in slot-scan work.

**Non-degradation analysis.**
- Recall: unchanged (routing and scoring are unaffected).
- Read latency: unchanged (readers never touch `occupied_bits`).
- Memory: four `uint64_t` per page replaces 256 `bool`. The struct shrinks.
- MVCC correctness: CAS-published pages are still immutable after publication; the bitset lives in the mutable staging copy that writers hold before CAS, same as before.

**Validation.** Run the ACID test suite (`tests/test_acid.py`, `tests/test_mvcc_and_space.py`); verify all `filled` iterators still match the bitset. Stress with TSan to confirm no race on the staging copy.

---

### P3: 4-bit routing centroid quantization

**Bottleneck.** The two-level coarse routing stage (`coarse_centroids`, `centroid_codes`) uses int8 per dimension per page. At 7,000 pages and 128 dimensions that is 7,000 × 128 = 896 KB of centroid data read per query. The routing score loop over all sqrt(7000) ≈ 84 groups requires 84 × 128 = 10,752 int8 multiplies just to pick which groups to probe.

**Change.** Quantize group centroids from int8 to 4 bits (two centroids per byte) using the same nibble-pack scheme already in use for per-record codes. The group centroid array halves from 896 KB to 448 KB. The routing dot-product loop reuses `int4_dot`, already written and NEON/AVX2-accelerated.

**Projected impact.** Halves the memory bandwidth of the routing pass. At 896 KB the array spills out of L2 cache on most CPUs; at 448 KB it fits in L2 (typical L2 is 256 KB-1 MB). On a warm cache the routing phase takes ~4.8 µs (from the profiling table); a hot-L2 routing pass is expected to drop to ~2-3 µs. The improvement compounds with P1 since `int4_dot` is the same kernel.

**Non-degradation analysis.**
- Recall: the routing pass is approximate by design (pages not reached contribute nothing to recall already). 4-bit centroid quantization introduces slightly more routing error than int8, which could cause marginally more misrouting. **Gating:** hold recall fixed at existing baselines (0.99 at `nprobe=32` on SIFT-128). If recall drops even 0.001 at any tested `nprobe`, fall back to int8 for that dimension range or increase `nprobe` by 1 to compensate.
- Insert throughput: routing is query-only; inserts are unaffected.
- Memory: net reduction.
- Correctness: quantization is lossy only for routing rank, not for the scored results; recall gates the acceptable loss.

**Validation.** Run the full recall suite at nprobe ∈ {4, 8, 16, 32, 64} on ANN-benchmarks datasets (SIFT-128, GIST-960). Fail the proposal if any recall point regresses beyond 0.001.

---

### P4: Hardware-accelerated WAL CRC

**Bottleneck.** The WAL CRC is a byte-by-byte software loop (`crc32_byte` called once per byte). On a 4 KB insert record this is 4,096 iterations of a table-lookup loop. ARM has the `__crc32cb` / `__crc32cw` hardware instructions (Cortex-A53+); x86-64 has `_mm_crc32_u8` / `_mm_crc32_u64` (SSE4.2). Both reduce the loop to one instruction per byte (and the 64-bit variant to one instruction per 8 bytes).

**Change.** Replace the software loop with:

```cpp
#if defined(__ARM_FEATURE_CRC32)
    while (len >= 8) { crc = __crc32cd(crc, *ptr8++); len -= 8; }
    while (len--) crc = __crc32cb(crc, *ptr++);
#elif defined(__SSE4_2__)
    while (len >= 8) { crc = _mm_crc32_u64(crc, *ptr8++); len -= 8; }
    while (len--) crc = _mm_crc32_u8(crc, *ptr++);
#else
    /* existing software path */
#endif
```

**Projected impact.** On a 4 KB WAL record: software CRC runs ~4,096 iterations; hardware CRC (64-bit variant) runs ~512. On ARM M1, `__crc32cd` throughput is ~1 cycle/8 bytes: the CRC becomes negligible in the insert path. This directly reduces write latency when WAL is enabled.

**Non-degradation analysis.**
- CRC value: must be identical to the software CRC for a valid WAL. **Critical:** the Castagnoli polynomial (CRC-32C) must match if the existing code uses CRC-32C, or the standard IEEE polynomial (CRC-32) if it uses that. Read the existing table before choosing the hardware instruction variant to ensure byte-exact compatibility with existing WAL files.
- Recall: unchanged (WAL is write-only).
- Recovery: recovery reads the same CRC and validates the same byte sequence: value unchanged.
- Portability: `#else` preserves the existing software path on targets without hardware CRC.

**Validation.** Write a round-trip test: write a WAL with hardware CRC, read it back with software CRC (or vice versa) and verify both accept the same records. Run the WAL recovery tests under ASan.

---

### P5: Tiered storage: codes in RAM, floats via mmap

**Bottleneck.** At 1M SIFT-128 vectors with `arena_path=`, both float vectors (4 bytes × 128 × 1M = 512 MB) and 4-bit residual codes (8 bytes per record at 128d = 8 MB) are memory-mapped. Cold queries page-fault on the float payload during reranking. But the 4-bit codes are read during screening for every visible candidate; mapping them cold causes page faults in the hot loop.

**Change.** Split the arena into two files: `vectors.arena` (floats, mmap, evictable) and `codes.arena` (4-bit codes, locked in RAM via `mlock` or loaded eagerly on open). The codes file for 1M SIFT-128 vectors is 8 MB, trivial to hold in RAM. Float vectors stay mmap'd and cold; only the reranking phase (`at most k × rerank_factor` vectors, default 40 for k=10) triggers page faults.

**Projected impact.** Screening becomes fault-free: the 21 µs screening phase runs entirely from RAM. Reranking faults at most `rerank` times (default 40 for k=10) instead of the full candidate set. On a 1M-vector cold-start workload, query latency drops from fault-dominated (potentially milliseconds) to the RAM-screening baseline (~21 µs + fault cost for ~40 floats).

**Non-degradation analysis.**
- Recall: unchanged (codes and floats are identical, just stored differently).
- Memory: codes file (8 MB at 1M × 128d) replaces part of what was mmap'd. The codes were previously counted against virtual memory; now they count against resident RSS. At 1M SIFT-128 this is an 8 MB RSS increase in exchange for eliminating page faults in the hot path: an acceptable trade. `mlock` should be guarded by `RLIMIT_MEMLOCK` and fall back to mmap if the limit is exceeded, preserving current behavior.
- Insert throughput: negligible change (code writes are sequential appends).
- Portability: `mlock` is POSIX; Windows uses `VirtualLock`. The existing arena abstraction can gate this per-platform.

**Validation.** Add a benchmark that queries a 1M-vector cold-start index with `arena_path=`, measure P50/P99 latency before and after. Verify no recall regression. Test `mlock` failure path (simulate by setting `RLIMIT_MEMLOCK=0` in a test) to confirm fallback is silent.

---

### P6: Bounded-degree graph routing elimination for write-heavy workloads

**Bottleneck.** Above `ROUTING_GRAPH_THRESHOLD = 16384` pages, `reconnect_node()` is called on every split. It runs an O(pages × d) neighbor scan to repair graph edges for the split and new nodes. At 20,000 pages and 128 dimensions that is 2.56M float multiplies per split, paid on the write path. The graph was added because linear centroid scanning at >16k pages costs more than graph traversal at query time, but the write cost is unbounded as page count grows.

**Change (option A: defer graph, extend flat scan).** Raise `ROUTING_GRAPH_THRESHOLD` and replace the linear scan with a quantized flat scan using the int4/int8 centroid codes from P3. At 20,000 pages × 128d with int8 centroids: 20,000 × 128 = 2.56M int8 multiplies ≈ 0.64 ms on NEON. With int4 (P3): 0.32 ms. For workloads that do not have 16k+ pages this proposal has no effect.

**Change (option B: lazy graph rebuild).** Keep the graph but rebuild it asynchronously after a configurable number of splits, rather than per-split. New pages are added to a "pending" list and scored by linear scan until the graph is rebuilt. This bounds write latency at the cost of a higher constant on queries during the rebuild window.

**Recommendation.** Option A (raise the threshold + P3) is strictly better for write-heavy workloads and adds no query-latency regression if the quantized flat scan is fast enough. Option B is safer for read-heavy workloads that need consistent query latency at high page counts.

**Projected impact.** Option A: eliminates the O(pages × d) reconnect cost entirely below the (raised) threshold: split latency returns to the base insert cost. Option B: caps per-split work at zero; rebuild is amortized over N splits and can run on a background thread.

**Non-degradation analysis.**
- Recall: for option A, the flat quantized scan at 20k pages is equivalent to the linear scan that existed before the graph: recall is the same as the pre-graph baseline at those page counts. For option B, recall during the rebuild window may dip slightly if the pending-list linear scan is less precise; this is bounded to the rebuild interval.
- Query latency: flat scan at 20k pages costs ~0.32-0.64 ms with P1+P3. Graph traversal at the same count costs < 0.1 ms. Option A increases query latency at high page counts. Only choose it if the write-latency win outweighs the query-latency cost: this is a workload-specific decision, not a universal improvement.
- MVCC correctness: graph repair happens inside the writer lock; deferring it does not change which readers see which pages.

**Validation.** Run the routing benchmark (currently in `tests/test_benchmark_compare.py`) at 20k and 50k pages. Record QPS and recall before and after. Only merge if query latency does not regress beyond 10% at the write-heavy workload's nprobe setting.

---

### Summary table

| ID | Target metric | Change | Risk to other metrics | Status |
|---|---|---|---|---|
| P1 | Query throughput (x86) | AVX2 `int4_dot`, `int8_dot`, `dot`, `l2sq` | None (same computation) | **Shipped** (`-mavx2` opt-in; NEON unchanged) |
| P2 | Insert throughput | `filled.size()` replaces O(capacity) fullness checks | None | **Shipped** |
| P3 | Routing latency / bandwidth | 4-bit nibble centroid codes (was int8) | Recall: gated ≤ 0.001 | **Shipped** (0.990 recall at 7521 pages) |
| P4 | WAL write latency | Table-driven CRC-32 (8× faster, same polynomial) | None (backward-compatible) | **Shipped** |
| P5 | Cold-start latency / memory | RecordStore on heap, float VectorStore via mmap | +8 MB RSS per 1M×128d | **Pending** (requires per-platform `mlock`/`VirtualLock` and arena split) |
| P6 | Write latency at >16k pages | int4 centroid codes in `nearest_pages` reconnect | Query latency at >16k pages | **Pending** (testability requires a >16k-page index) |

**P5 path:** `VectorStore` and `RecordStore` are already separate allocations. A `pin_codes=True` flag in `cv_create_with_options` that forces `RecordStore` to heap regardless of `arena_path` is the minimal implementation.

**P6 path:** `nearest_pages()` (`bindings/rust/chronovec-sys/native/src/chronovec.cpp:1084`) does a full float scan. Replacing `distance()` there with the int4 centroid scoring from P3 would halve `reconnect_node` memory bandwidth. The routing graph only activates above `ROUTING_GRAPH_THRESHOLD = 16384` pages; validate recall at that scale before shipping.

---

## SQLite integration

`CREATE VIRTUAL TABLE ... USING chronovec(dim=384, metric=cosine, ...)` creates an append-only shadow table in SQLite. SQLite's pager and WAL provide durability. On reopen, ChronoVec replays the shadow log to reconstruct the native cache. Rollback marks the cache dirty; the next query rebuilds from committed SQLite state.

The shadow log is canonical; the native pages are a cache.

---

## Disk-backed arena

For collections that don't fit in RAM, `arena_path=` moves the float payload and 4-bit codes to memory-mapped files:

```python
index = Index(768, arena_path="vectors.arena", max_vectors=10_000_000)
```

Both the vectors (80% of payload) and the codes (20%) are mapped so the kernel can evict cold pages. Insert dirties one vector, not a page: copy-on-write clones share the float payload. `flush_vectors()` makes pages clean and evictable.

Capacity estimate (1.58x amplification at 1M SIFT):

| dimensions | ~vectors in 8 GB RAM | ~vectors in 32 GB RAM |
|---:|---:|---:|
| 128 | 7.7M | 30.8M |
| 768 | 1.3M | 5.2M |
