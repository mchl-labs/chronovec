"""Run one engine in its own process and emit its report as JSON.

Isolation is not tidiness. A native engine can abort the process outright --
FAISS trips a C++ assertion on some remove paths -- and a benchmark that dies
partway through is worse than one that records a failure and continues.
"""

from __future__ import annotations

import json
import sys

from .cli import load
from .runner import (ENGINES, run_engine, run_engine_concurrent,
                     run_engine_equal_recall, run_engine_filtered,
                     run_engine_ops, run_engine_snapshot)
from .trace import build_mixed_trace, build_ops_trace, build_trace


def main() -> None:
    spec = json.loads(sys.argv[1])
    base, queries = load(spec["dataset"], spec["queries"], spec["metric"])
    if spec.get("mode") == "filtered":
        trace = build_ops_trace(base.shape[0], spec["live"], spec["batch"], 1,
                                spec["seed"])
        result = run_engine_filtered(
            ENGINES[spec["engine"]], base, trace, queries, spec["k"],
            spec["metric"], spec.get("options", {}),
            selectivities=spec.get("selectivities", (0.5, 0.1, 0.01)),
            churn_rounds=spec.get("churn_rounds", 0),
            churn_fraction=spec.get("churn_fraction", 0.25),
            layout=spec.get("layout", "random"),
            target_recall=spec.get("target_recall", 0.95))
    elif spec.get("mode") == "equal-recall":
        trace = build_ops_trace(base.shape[0], spec["live"], spec["batch"], 1,
                                spec["seed"])
        result = run_engine_equal_recall(ENGINES[spec["engine"]], base, trace,
                                         queries, spec["k"], spec["metric"],
                                         spec.get("options", {}))
    elif spec.get("mode") == "snapshot":
        trace = build_ops_trace(base.shape[0], spec["live"], spec["batch"],
                                spec["epochs"], spec["seed"])
        result = run_engine_snapshot(ENGINES[spec["engine"]], base, trace,
                                     queries, spec["k"], spec["metric"],
                                     spec.get("options", {}))
    elif spec.get("mode") == "concurrent":
        trace = build_ops_trace(base.shape[0], spec["live"], spec["batch"],
                                spec["epochs"], spec["seed"])
        result = run_engine_concurrent(ENGINES[spec["engine"]], base, trace,
                                       queries, spec["k"], spec["metric"],
                                       spec.get("options", {}),
                                       spec.get("readers", 3))
    elif spec.get("mode") in ("ops", "mixed"):
        build = build_mixed_trace if spec["mode"] == "mixed" else build_ops_trace
        trace = build(base.shape[0], spec["live"], spec["batch"],
                      spec["epochs"], spec["seed"])
        result = run_engine_ops(ENGINES[spec["engine"]], base, trace, queries,
                                spec["k"], spec["metric"],
                                spec.get("options", {}))
    else:
        trace = build_trace(base.shape[0], spec["live"], spec["epochs"],
                            spec["turnover"], spec["seed"])
        result = run_engine(ENGINES[spec["engine"]], base, trace, queries,
                            spec["k"], spec["metric"], spec.get("options", {}),
                            spec.get("mode", "bulk"))
    sys.stdout.write("@@RESULT@@" + json.dumps(result))


if __name__ == "__main__":
    main()
