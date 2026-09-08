#!/usr/bin/env python3
"""Run fresh-process benchmark repetitions and summarize 95% confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def summarize(reports):
    rows = [dict(flatten(report)) for report in reports]
    shared = set.intersection(*(set(row) for row in rows))
    result = {}
    # Student-t 97.5% critical values for 2..30 samples; normal thereafter.
    critical = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    count = len(rows)
    t_value = critical.get(count, 1.96 if count > 30 else 2.0)
    for path in sorted(shared):
        values = np.asarray([row[path] for row in rows], dtype=np.float64)
        if np.all(np.isfinite(values)):
            standard_error = float(values.std(ddof=1) / math.sqrt(count)) if count > 1 else 0.0
            result[path] = {
                "mean": float(values.mean()),
                "ci95_low": float(values.mean() - t_value * standard_error),
                "ci95_high": float(values.mean() + t_value * standard_error),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    benchmark_args = (
        args.benchmark_args[1:] if args.benchmark_args[:1] == ["--"] else args.benchmark_args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for repetition in range(args.repetitions):
        output = args.output_dir / f"run-{repetition + 1}.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("run_benchmark.py")),
            *benchmark_args,
            "--output",
            str(output),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        reports.append(json.loads(output.read_text()))
    campaign = {
        "schema_version": 2,
        "repetitions": args.repetitions,
        "command": [str(Path(__file__).with_name("run_benchmark.py")), *benchmark_args],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "runs": [
            str(args.output_dir / f"run-{index + 1}.json") for index in range(args.repetitions)
        ],
        "confidence_intervals": summarize(reports),
    }
    destination = args.output_dir / "summary.json"
    destination.write_text(json.dumps(campaign, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
