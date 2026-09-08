"""compare_revisions.py's command construction, without real builds/subprocess.

measure() shells out to cmake and `uv run ... regression_gate.py`, which is
too slow for the fast suite (real worktree checkouts and native builds). What
actually matters here is provable without any of that: which flags measure()
puts on the regression_gate.py command line for each of its two call shapes.
"""

import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import compare_revisions as cr  # noqa: E402


def args():
    return Namespace(dataset=Path("data.hdf5"), live=1000, queries=10, k=5, repeats=1)


def test_measuring_the_base_revision_uses_measure_only_not_the_static_baseline(
    monkeypatch, tmp_path
):
    """Without --measure-only, the base worktree's own regression_gate.py
    falls back to comparing against whatever static regression_baseline.json
    happens to be checked out there -- unrelated to what this call measures,
    and a spurious "regression" against it crashed the whole comparison for
    a reason that had nothing to do with the base revision being measured."""
    calls = []
    monkeypatch.setattr(cr, "run", lambda command, *, cwd, env=None: calls.append(command))

    cr.measure(tmp_path, tmp_path / "base.json", args())

    gate_call = calls[-1]
    assert "--measure-only" in gate_call
    assert "--baseline" not in gate_call


def test_measuring_the_current_tree_compares_against_the_fresh_base(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cr, "run", lambda command, *, cwd, env=None: calls.append(command))
    base_result = tmp_path / "base.json"

    cr.measure(tmp_path, tmp_path / "current.json", args(), baseline=base_result)

    gate_call = calls[-1]
    assert "--measure-only" not in gate_call
    assert "--baseline" in gate_call
    assert str(base_result) in gate_call
