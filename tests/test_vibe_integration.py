"""The VIBE adapter against VIBE's own base class contract.

VIBE (https://github.com/vector-index-bench/vibe) is the maintained successor
to ann-benchmarks, which now points at it. It measures build and static query
only, so it covers the axis this project is weakest on and none of the ones it
is built for -- which is exactly why the static number should come from a
harness this project did not write.

The adapter is tested here rather than only inside VIBE, because a container
run is a slow way to find a typo.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture(scope="module")
def module():
    """Import the adapter with a stand-in for VIBE's package layout.

    The adapter uses `from ..base.module import BaseANN`, so the two parent
    packages have to exist. Reproducing VIBE's real base class here keeps the
    contract honest: `query` is what must be implemented, `batch_query` fills
    `self.res`, and `get_batch_results` reads it.
    """
    from multiprocessing.pool import ThreadPool

    class BaseANN:
        def batch_query(self, X, n):
            pool = ThreadPool()
            self.res = pool.map(lambda q: self.query(q, n), X)

        def get_batch_results(self):
            return self.res

        def get_additional(self):
            return {}

        def __str__(self):
            return self.name

    root = types.ModuleType("vibe_stub")
    root.__path__ = []
    base_pkg = types.ModuleType("vibe_stub.base")
    base_pkg.__path__ = []
    base_mod = types.ModuleType("vibe_stub.base.module")
    base_mod.BaseANN = BaseANN
    sys.modules.update(
        {"vibe_stub": root, "vibe_stub.base": base_pkg, "vibe_stub.base.module": base_mod}
    )

    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent / "integrations" / "vibe" / "chronovec" / "module.py"
    )
    # Named as if it sat at vibe/algorithms/chronovec/module.py, so `..base`
    # resolves to the stub base package exactly as it will resolve to VIBE's.
    chronovec_pkg = types.ModuleType("vibe_stub.chronovec")
    chronovec_pkg.__path__ = []
    sys.modules["vibe_stub.chronovec"] = chronovec_pkg
    spec = importlib.util.spec_from_file_location("vibe_stub.chronovec.module", path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules["vibe_stub.chronovec.module"] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def data(n=3000, dim=16, seed=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, dim)).astype(np.float32)


@pytest.mark.parametrize(
    "metric,expected",
    [
        ("euclidean", "l2"),
        ("cosine", "cosine"),
        ("ip", "cosine"),
        ("normalized", "cosine"),
    ],
)
def test_every_metric_vibe_passes_is_mapped(module, metric, expected):
    assert module.ChronoVec(metric).metric == expected


def test_fit_then_query_returns_the_true_neighbour(module):
    X = data()
    engine = module.ChronoVec("euclidean", page_capacity=128)
    engine.fit(X)
    engine.set_query_arguments(64)
    for probe in (0, 1500, 2999):
        assert engine.query(X[probe], 10)[0] == probe


def test_batch_query_matches_single_query(module):
    X = data()
    engine = module.ChronoVec("euclidean", page_capacity=128)
    engine.fit(X)
    engine.set_query_arguments(64)
    queries = X[[0, 100, 2000]]
    one = [list(engine.query(q, 10)) for q in queries]
    engine.batch_query(queries, 10)
    assert [list(row) for row in engine.get_batch_results()] == one


def test_batch_results_drop_the_padding(module):
    # Fewer records than k, so every row is padded and VIBE must not see -1.
    X = data(n=40, dim=8)
    engine = module.ChronoVec("euclidean", page_capacity=32)
    engine.fit(X)
    engine.set_query_arguments(32)
    engine.batch_query(X[:3], 100)
    for row in engine.get_batch_results():
        assert len(row) == 40 and all(i >= 0 for i in row)


def test_cosine_ranks_by_cosine(module):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(500, 8)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    engine = module.ChronoVec("cosine", page_capacity=64)
    engine.fit(X)
    engine.set_query_arguments(64)
    assert engine.query(X[42], 5)[0] == 42


def test_additional_reports_what_the_build_produced(module):
    X = data()
    engine = module.ChronoVec("euclidean", page_capacity=128)
    engine.fit(X)
    extra = engine.get_additional()
    assert extra["pages"] >= 1
    assert extra["capacity_amplification"] > 0
    assert extra["tracked_index_bytes"] > 0


def test_str_names_the_configuration(module):
    engine = module.ChronoVec("euclidean", page_capacity=512)
    engine.set_query_arguments(32)
    text = str(engine)
    assert "512" in text and "32" in text


def test_the_config_grid_matches_the_constructor(module):
    from pathlib import Path

    yaml = pytest.importorskip("yaml")

    path = (
        Path(__file__).resolve().parent.parent
        / "integrations"
        / "vibe"
        / "chronovec"
        / "config.yml"
    )
    config = yaml.safe_load(path.read_text())
    entry = config["float"]["any"][0]
    assert entry["constructor"] == "ChronoVec"
    assert entry["module"] == "vibe.algorithms.chronovec"
    for group in entry["run_groups"].values():
        # Every argument the grid sweeps must be one the constructor accepts,
        # and every query argument one set_query_arguments takes.
        for key in group["args"]:
            module.ChronoVec("euclidean", **{key: group["args"][key][0]})
        assert list(group["query_args"]) == ["nprobe"]
