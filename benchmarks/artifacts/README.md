# Public benchmark artifacts

This directory contains raw JSON output from the repository's reproducible
regression workload. Each result records the dataset path, workload
configuration, commit, working-tree state, machine, calibration, and measured metrics. Timing
comparisons are meaningful only when the calibration and environment are
comparable; deterministic recall, fill, and space metrics remain useful across
machines.

To reproduce or create a new artifact:

```bash
python -m pip install -e ".[benchmarks]"
python benchmarks/fetch_datasets.py sift-128-euclidean --data-dir benchmark_data
python benchmarks/regression_gate.py \
  --dataset benchmark_data/sift-128-euclidean.hdf5 \
  --live 50000 --queries 300 --repeats 2 \
  --output benchmarks/artifacts/regression-run.json
```

The command does not update the regression baseline unless `--update` is also
passed. Large source datasets are intentionally not committed; the fetch
helper records the canonical ANN-Benchmarks source and verifies the download.
