#!/bin/sh
set -eu

docker build -f Dockerfile.benchmark -t chronovec-benchmark .
mkdir -p benchmark_results
docker run --rm \
  -v "$(pwd)/benchmark_results:/work/benchmark_results" \
  chronovec-benchmark \
  --vectors "${VECTORS:-20000}" \
  --dimensions "${DIMENSIONS:-64}" \
  --queries "${QUERIES:-300}" \
  --churn "${CHURN:-4000}" \
  --page-capacity "${PAGE_CAPACITY:-256}" \
  --probes 4 8 16 32 \
  --output benchmark_results/linux_scann.json

