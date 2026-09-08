import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from compare_results import compare, qps_at_recall


def engine(points, updates=100.0, capacity=1.5):
    return {
        "initial": {"chronovec_native": {"variants": points}},
        "streaming": {
            "chronovec_native": {
                "variants": points,
                "updates_per_s": updates,
                "stats": {"capacity_amplification": capacity},
            }
        },
    }


def test_qps_at_recall_interpolates_curve():
    value = qps_at_recall(
        {"variants": {"a": {"recall_at_k": 0.9, "qps": 100}, "b": {"recall_at_k": 1.0, "qps": 50}}},
        0.95,
    )
    assert value == pytest.approx(75)


def test_compare_rejects_post_churn_query_regression():
    baseline = engine(
        {"a": {"recall_at_k": 0.97, "qps": 100}, "b": {"recall_at_k": 0.99, "qps": 80}}
    )
    trial = engine(
        {"a": {"recall_at_k": 0.97, "qps": 105}, "b": {"recall_at_k": 0.99, "qps": 75}},
        updates=110,
        capacity=1.4,
    )
    result = compare(
        baseline,
        trial,
        target_recall=0.98,
        min_qps_gain=0.02,
        max_update_regression=0.02,
        max_capacity_regression=0,
    )
    assert not result["passed"]
