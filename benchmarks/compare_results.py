#!/usr/bin/env python3
"""Gate an algorithm trial against a benchmark baseline at equal recall."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def qps_at_recall(engine: dict, target: float) -> float | None:
    points = sorted(
        (float(item["recall_at_k"]), float(item["qps"]))
        for item in engine.get("variants", {}).values()
        if "recall_at_k" in item and "qps" in item
    )
    if not points or target < points[0][0] or target > points[-1][0]:
        return None
    for recall, qps in points:
        if recall == target:
            return qps
    for (left_recall, left_qps), (right_recall, right_qps) in zip(points, points[1:]):
        if left_recall <= target <= right_recall:
            weight = (target - left_recall) / (right_recall - left_recall)
            return left_qps + weight * (right_qps - left_qps)
    return None


def compare(
    baseline: dict,
    trial: dict,
    *,
    target_recall: float,
    min_qps_gain: float,
    max_update_regression: float,
    max_capacity_regression: float,
) -> dict:
    checks = []
    for phase in ("initial", "streaming"):
        before = qps_at_recall(baseline[phase]["chronovec_native"], target_recall)
        after = qps_at_recall(trial[phase]["chronovec_native"], target_recall)
        passed = before is not None and after is not None and after >= before * (1 + min_qps_gain)
        checks.append(
            {
                "name": f"{phase}_equal_recall_qps",
                "passed": passed,
                "baseline": before,
                "trial": after,
                "relative_change": after / before - 1 if before and after else None,
            }
        )

    before_updates = float(baseline["streaming"]["chronovec_native"]["updates_per_s"])
    after_updates = float(trial["streaming"]["chronovec_native"]["updates_per_s"])
    checks.append(
        {
            "name": "replacement_throughput",
            "passed": after_updates >= before_updates * (1 - max_update_regression),
            "baseline": before_updates,
            "trial": after_updates,
            "relative_change": after_updates / before_updates - 1,
        }
    )

    before_capacity = float(
        baseline["streaming"]["chronovec_native"]["stats"]["capacity_amplification"]
    )
    after_capacity = float(
        trial["streaming"]["chronovec_native"]["stats"]["capacity_amplification"]
    )
    checks.append(
        {
            "name": "capacity_amplification",
            "passed": after_capacity <= before_capacity * (1 + max_capacity_regression),
            "baseline": before_capacity,
            "trial": after_capacity,
            "relative_change": after_capacity / before_capacity - 1,
        }
    )
    return {
        "passed": all(check["passed"] for check in checks),
        "target_recall": target_recall,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--target-recall", type=float, default=0.98)
    parser.add_argument("--min-qps-gain", type=float, default=0.02)
    parser.add_argument("--max-update-regression", type=float, default=0.02)
    parser.add_argument("--max-capacity-regression", type=float, default=0.0)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()),
        json.loads(args.trial.read_text()),
        target_recall=args.target_recall,
        min_qps_gain=args.min_qps_gain,
        max_update_regression=args.max_update_regression,
        max_capacity_regression=args.max_capacity_regression,
    )
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
