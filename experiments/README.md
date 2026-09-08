# v2 design experiments

Two offline experiments run before touching the engine, to falsify the two
candidate theses cheaply. Both are reproducible from the checked-in datasets.

## 1. `overlap_simulation.py`: bounded overlap. RESULT: FALSIFIED.

Question: can replicating boundary vectors into neighbouring pages cut the
candidate count enough to matter?

Method: with exact reranking a true neighbour is recovered iff it is present in
the probed candidate set, so Recall@k reduces to set membership and the whole
curve is computable without distance work. Replicas are admitted by distance to
the Voronoi bisector between a vector's primary page and a rival (a true length
in the embedding space, unlike ratio margins, which collapse under distance
concentration in high dimension). The admission threshold is a quantile chosen so
that overlap hits an exact target, making the budget a hard resource.

```bash
python experiments/overlap_simulation.py \
  --dataset benchmark_data/sift-128-euclidean.hdf5 --queries 500 \
  --overlaps 1.0 1.15 1.25 1.4 1.75 --output experiments/results/sift1m_kmeans.json
```

SIFT-128, 1M vectors, 3,906 pages, candidates needed for Recall@10 >= 0.98:

| overlap | candidates | vs 1.0x |
|---|---:|---:|
| 1.00x | ~26,074 (P=96) | N/A |
| 1.15x | 20,421 (P=64) | 0.78x |
| 1.25x | 22,344 (P=64) | 0.86x |
| 1.40x | 25,222 (P=64) | 0.97x |
| 1.75x | 24,037 (P=48) | 0.92x |

Best case is ~20% fewer candidates at 1.15x overlap, degrading past that:
replicas inflate page sizes as fast as they cut probe counts. The plan's gate
required 4x. **Thesis rejected before any implementation cost.**

A secondary finding: proper k-means partitioning needs ~26k candidates where the
engine's greedy incremental placement measured 30.5k. Partition quality is worth
~15%, not a leap either. Candidate count is near a floor for centroid-routed
partition indexes; it cannot be fixed by partitioning or overlap.

## 2. `quant_simulation.py`: compressed candidate scan. RESULT: CONFIRMED.

Question: if the candidate count cannot fall, can each candidate get much
cheaper without losing recall?

Method: end-to-end simulation of the proposed search path: rank pages, gather
candidates, estimate distances from compressed codes, rerank the best `budget`
exactly. The fp32 mode is a control that must reproduce the candidate-set
ceiling exactly; it does, validating the residual decomposition
`||q-v||^2 = ||q-c||^2 + ||v-c||^2 - 2||q-c||*||v-c||*<u_q,u_v>`.

```bash
python experiments/quant_simulation.py \
  --dataset benchmark_data/sift-128-euclidean.hdf5 --queries 300 \
  --probes 64 96 128 192 --modes fp32 rabitq sq2 sq3 sq4 \
  --output experiments/results/quant_sift1m.json
```

SIFT-128, 1M vectors, P=192 (51,404 candidates, ceiling 0.9980):

| code | bytes/vector | rerank 50 | 100 | 200 | 400 | 800 |
|---|---:|---:|---:|---:|---:|---:|
| fp32 (control) | 512 | 0.9973 | 0.9973 | 0.9973 | 0.9973 | 0.9973 |
| RaBitQ 1-bit | 24 | 0.9177 | 0.9683 | 0.9930 | 0.9967 | 0.9980 |
| 2-bit | 40 | 0.9813 | 0.9947 | 0.9977 | 0.9973 | 0.9977 |
| 3-bit | 56 | 0.9973 | 0.9967 | 0.9973 | 0.9973 | 0.9967 |
| 4-bit | 72 | 0.9973 | 0.9973 | 0.9973 | 0.9967 | 0.9973 |

3-bit codes are lossless against the fp32 control at every probe level with a
rerank budget of 50: 9.1x fewer scanned bytes at zero recall cost. RaBitQ
1-bit reaches the ceiling at a budget of 400-800 for 21x fewer bytes.

### Rotation is not required above 3 bits

```bash
python experiments/quant_simulation.py ... --no-rotation
```

| code | rerank 20 (rotated) | rerank 20 (identity) |
|---|---:|---:|
| 3-bit | 0.9905 | 0.9745 |
| 4-bit | 0.9960 | 0.9960 |
| 8-bit | 0.9960 | 0.9960 |

At 4 bits and above a random rotation buys nothing. This removes the rotation
matrix, its checkpoint serialisation, its determinism requirements and its
per-query cost from the design. It matters only if the 1-2 bit regime is
pursued later for a memory-first tier.

## 3. `metadata_history_simulation.py`: delta-chain metadata history. RESULT: CONDITIONALLY CONFIRMED.

Question: `Collection._history` (`chronovec/collection.py`) stores a full dict
copy on every version, including a metadata-only `update()` that changes one
key out of many -- inspired by a Cloudflare DNS-cache write-up on cutting
per-entry memory by storing only what differs from an inferable default
instead of copying the whole record. Does chaining deltas (a full "keyframe"
dict every 16 versions, changed-keys-only in between) cut real memory here,
and at what read cost?

Method: simulate N ids each receiving partial `update()` calls that touch a
`touch`-sized subset of a `width`-key metadata dict, the shape of an
agent/streaming workload that edits a status field on a long-lived record.
Measure actual heap bytes with `tracemalloc` (one representation on the heap
at a time) and the wall-clock cost of resolving current metadata for every id
-- the real cost `Collection.get`/`query` pay on every read. Two delta
encodings are compared: a nested dict per version, and a flat tuple of
`(key, value)` pairs (CPython's dict container has a fixed overhead that a
1-2 key `changed` dict pays in full).

```bash
python experiments/metadata_history_simulation.py \
  --ids 10000 --width 16 --updates 40 --touch 1 2 4 8 16 \
  --keyframe-interval 16 --repeats 7
python experiments/metadata_history_simulation.py \
  --ids 5000 --width 64 --updates 40 --touch 1 2 4 8 --keyframe-interval 16 --repeats 5
```

16-key metadata (typical small record), bytes/id relative to full-copy history:

| keys touched / 16 | delta (dict) | delta (flat tuple) | resolve slowdown |
|---:|---:|---:|---:|
| 1 | 0.72x | **0.58x** | 3.9-4.2x |
| 2 | 0.74x | 0.71x | 3.5x |
| 4 | 0.77x | 0.92x | 3.3-4.4x |
| 8 | 0.91x | 1.23x | 3.1-3.7x |
| 16 (full rewrite) | 1.09x | 1.59x | 4.0-5.7x |

64-key metadata (wide record), same sweep:

| keys touched / 64 | delta (dict) | delta (flat tuple) |
|---:|---:|---:|
| 1 | 0.32x | **0.28x** |
| 2 | 0.33x | 0.32x |
| 4 | 0.36x | 0.41x |
| 8 | 0.45x | 0.58x |

Confirmed, but narrower than the naive "only one key changed, why copy
sixteen" intuition suggested: CPython's dict container (hash table + entries
array) costs ~500-700 bytes regardless of how many keys it holds, so even a
1-key delta pays most of a full dict's fixed overhead -- the flat-tuple
encoding exists specifically to dodge that container cost, and it wins only
while `touch` stays small (crosses over to *worse than the naive full copy*
past roughly a quarter of the keys touched per update, both encodings do).
The technique earns its complexity only in the regime of wide metadata with
sparse per-update touches (30-70% smaller) -- narrow metadata or
touch-most-of-the-keys updates make it a wash or a net loss. Every regime
pays a 3-4x wall-clock cost to resolve current metadata (still
microseconds/id in absolute terms, but real, and paid on every
`Collection.get`/`query` call, not just the rare wide-metadata write).

Not implemented in `Collection`: the win is real but conditional on workload
shape chronovec has no data on yet (how wide is metadata in practice, how
sparse are updates), and adopting it means a chain-aware `vacuum`/
`_prune_history`, a checkpoint format bump, and a slower read path for every
caller, including the ones with narrow metadata that this doesn't help.
Worth revisiting if a real workload shows up with wide, sparsely-updated
metadata and history memory actually shows up in a profile.
