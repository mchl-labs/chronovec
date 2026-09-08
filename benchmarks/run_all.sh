#!/usr/bin/env bash
# The full comparison, in three tiers.
#
# Each tier answers a different question and they are separate because they
# fail separately: a missing engine should not cost the head-to-head, and a
# long tail-latency run should not have to be repeated to re-check a recall
# curve. Every tier writes JSON, so a partial run is still usable.
#
# Usage: benchmarks/run_all.sh [output-directory]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-benchmarks/results}"
mkdir -p "$OUT"
export PYTHONPATH="$PWD"
PY="${PYTHON:-python}"
DATA="${DATASET_DIR:-benchmark_data}"

log() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------- tier A
# Head to head against faiss across every configuration: resident or mapped,
# one thread or several, queries one at a time or in a batch, with both
# engines swept over their own build parameter so neither is compared at a
# guess while the other is at its best.
for SET in sift-128-euclidean:128 glove-25-angular:25 gist-960-euclidean:960; do
  NAME="${SET%%:*}"; DIM="${SET##*:}"
  [ -f "$DATA/$NAME.hdf5" ] || { echo "skipping $NAME: not present"; continue; }
  log "tier A: $NAME (d=$DIM)"
  $PY benchmarks/faiss_matrix.py --dataset "$DATA/$NAME.hdf5" \
      --dims "$DIM" --scales 50000 200000 1000000 \
      --queries 500 --single 2000 \
      --storage heap mmap --workers 1 4 8 \
      --output "$OUT/A-$NAME.json"
done

# ---------------------------------------------------------------- tier B
# The rest of the field, on the workloads that separate engines rather than
# tuning: each operation timed apart, traffic interleaved rather than sorted
# by operation, readers running while a writer works, and a snapshot held
# open. Engines that cannot do an operation are reported as such.
ENGINES="chronovec faiss-ivfflat hnswlib usearch qdrant chroma annoy faiss-hnsw"
for MODE in ops mixed concurrent snapshot equal-recall; do
  log "tier B: $MODE"
  $PY -m streambench --dataset "$DATA/sift-128-euclidean.hdf5" \
      --mode "$MODE" --live 200000 --batch 20000 --epochs 2 \
      --queries 500 --readers 4 --engines $ENGINES \
      --output "$OUT/B-$MODE.json"
done

# ---------------------------------------------------------------- tier C
# What only this engine claims: space that stays a function of live data
# under sustained churn, and the price of durability.
log "tier C: churn and durability"
$PY benchmarks/churn_and_durability.py --dataset "$DATA/sift-128-euclidean.hdf5" \
    --live 200000 --cycles 12 --output "$OUT/C-churn.json"

log "done"
echo "results in $OUT"
