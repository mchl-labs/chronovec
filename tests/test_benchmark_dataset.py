import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from run_benchmark import load_ann_benchmarks


@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is an optional benchmark dependency"
)
def test_load_ann_benchmarks_uses_published_truth(tmp_path):
    import h5py

    path = tmp_path / "tiny-angular.hdf5"
    with h5py.File(path, "w") as dataset:
        dataset.attrs["distance"] = "angular"
        dataset.create_dataset("train", data=np.eye(4, dtype=np.float32))
        dataset.create_dataset("test", data=np.eye(4, dtype=np.float32)[:2])
        dataset.create_dataset("neighbors", data=np.array([[0, 1], [1, 0]], dtype=np.int64))
    base, queries, truth, metric = load_ann_benchmarks(path, query_count=1, k=2)
    assert base.shape == (4, 4)
    assert queries.shape == (1, 4)
    assert truth.tolist() == [[0, 1]]
    assert metric == "cosine"


@pytest.mark.skipif(
    importlib.util.find_spec("h5py") is None, reason="h5py is an optional benchmark dependency"
)
def test_load_ann_benchmarks_supports_euclidean_metric(tmp_path):
    import h5py

    path = tmp_path / "tiny-l2.hdf5"
    with h5py.File(path, "w") as dataset:
        dataset.attrs["distance"] = "euclidean"
        dataset.create_dataset("train", data=np.eye(2, dtype=np.float32))
        dataset.create_dataset("test", data=np.eye(2, dtype=np.float32))
        dataset.create_dataset("neighbors", data=np.array([[0], [1]], dtype=np.int64))
    base, queries, truth, metric = load_ann_benchmarks(path, query_count=1, k=1)
    assert metric == "l2"
    assert truth.tolist() == [[0]]
