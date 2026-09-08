"""Command line: python -m streambench --dataset ... --engines ..."""

from __future__ import annotations

import argparse
import json

import numpy as np

from .engines import available_engines, format_capability_matrix
from .runner import run_benchmark
from .trace import build_mixed_trace, build_ops_trace, build_trace


def normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return values / norms


def load(path: str, queries: int, metric: str):
    import h5py
    with h5py.File(path, "r") as handle:
        base = np.asarray(handle["train"][:], dtype=np.float32)
        test = np.asarray(handle["test"][:queries], dtype=np.float32)
    if metric == "cosine":
        base, test = normalize(base), normalize(test)
    return base, test


def _run_isolated(engine: str, args: argparse.Namespace) -> dict:
    """Run one engine in a child process so a native crash cannot end the run."""
    import subprocess
    import sys
    spec = {"engine": engine, "dataset": args.dataset, "queries": args.queries,
            "metric": args.metric, "live": args.live, "epochs": args.epochs,
            "turnover": args.turnover, "seed": args.seed, "k": args.k,
            "mode": args.mode, "batch": args.batch, "readers": args.readers,
            "selectivities": list(args.selectivities),
            "churn_rounds": args.churn_rounds,
            "churn_fraction": args.churn_fraction,
            "layout": args.tag_layout,
            "target_recall": args.target_recall}
    finished = subprocess.run([sys.executable, "-m", "streambench._worker",
                               json.dumps(spec)],
                              capture_output=True, text=True)
    marker = finished.stdout.find("@@RESULT@@")
    if marker >= 0:
        return json.loads(finished.stdout[marker + len("@@RESULT@@"):])
    detail = (finished.stderr or finished.stdout).strip().splitlines()
    reason = detail[-1] if detail else f"exit code {finished.returncode}"
    return {"skipped": reason, "exit_code": finished.returncode}


def _run_ops(args: argparse.Namespace, base: np.ndarray,
             queries: np.ndarray) -> None:
    """Per-operation mode: report each mutation kind on its own line.

    `mixed` runs the same total work with the kinds interleaved rather than in
    homogeneous bursts, because an engine can be fine on each kind alone and
    bad on the mixture.
    """
    build = build_mixed_trace if args.mode == "mixed" else build_ops_trace
    trace = build(base.shape[0], args.live, args.batch, args.epochs, args.seed)
    report = {"schema_version": 1,
              "workload": {"live_vectors": trace.live, "batch": args.batch,
                           "phases": len(trace.phases), "mode": args.mode,
                           "dimensions": int(base.shape[1]),
                           "metric": args.metric, "dataset": args.dataset,
                           "k": args.k, "queries": int(queries.shape[0])},
              "engines": {}}
    del base, queries
    for name in args.engines:
        report["engines"][name] = _run_isolated(name, args)

    print(f"\n{args.live:,} live vectors, {args.batch:,} operations per phase, "
          f"{args.epochs} cycles of insert/update/delete/replace\n")
    print("what each engine can actually do:")
    print(format_capability_matrix(args.engines) + "\n")
    print(f"{'engine':<16}{'phase':>9}{'live':>9}{'ops/s':>12}"
          f"{'maint ms':>10}{'recall':>9}  note")
    for name, payload in report["engines"].items():
        if "phases" not in payload:
            print(f"{name:<16}  skipped: {payload['skipped']}")
            continue
        capability = payload["capabilities"]
        print(f"{name:<16}  delete={capability['can_delete']} "
              f"grow={capability['can_grow']} "
              f"update_in_place={capability['can_update_in_place']}")
        for row in payload["phases"]:
            print(f"{'':<16}{row['kind']:>9}{row['live_vectors']:>9,}"
                  f"{row['write_ops_per_s']:>12,.0f}"
                  f"{row['maintenance_s'] * 1000:>10.1f}"
                  f"{row['recall_at_k']:>9.4f}  {row.get('note', '')}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.output}")


def _run_concurrent(args: argparse.Namespace, base: np.ndarray,
                    queries: np.ndarray) -> None:
    """Read-during-write: what a lock-free reader is actually for."""
    trace = build_ops_trace(base.shape[0], args.live, args.batch, args.epochs,
                            args.seed)
    report = {"schema_version": 1,
              "workload": {"live_vectors": trace.live, "batch": args.batch,
                           "phases": len(trace.phases), "mode": "concurrent",
                           "reader_threads": args.readers,
                           "dimensions": int(base.shape[1]),
                           "metric": args.metric, "dataset": args.dataset,
                           "k": args.k},
              "engines": {}}
    del base, queries
    for name in args.engines:
        report["engines"][name] = _run_isolated(name, args)

    print(f"\n{args.live:,} live vectors, {args.readers} reader threads "
          f"searching while one thread mutates\n")
    print("what each engine can actually do:")
    print(format_capability_matrix(args.engines) + "\n")
    print(f"{'engine':<16}{'write/s':>11}{'read qps':>10}{'p50 ms':>9}"
          f"{'p99 ms':>10}{'anchor miss':>13}{'quiet b/a':>13}{'excess':>9}{'unknown':>9}")
    for name, payload in report["engines"].items():
        if "reads" not in payload:
            print(f"{name:<16}  skipped: {payload['skipped']}")
            continue
        reads = payload["reads"]
        writes = (sum(p["write_operations"] for p in payload["phases"])
                  / sum(p["write_s"] for p in payload["phases"]))
        print(f"{name:<16}{writes:>11,.0f}{reads['read_qps']:>10,.0f}"
              f"{reads['latency_p50_ms']:>9.3f}{reads['latency_p99_ms']:>10.3f}"
              f"{reads['anchor_miss_rate']:>13.4f}"
              f"{reads['quiescent_anchor_miss_rate_before']:.4f}/"
              f"{reads['quiescent_anchor_miss_rate_after']:.4f}"
              f"{reads['excess_over_quiescent']:>9.4f}"
              f"{reads['unknown_ids']:>9}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.output}")


def _run_snapshot(args: argparse.Namespace, base: np.ndarray,
                  queries: np.ndarray) -> None:
    """Hold one snapshot open across the trace and report what it costs."""
    trace = build_ops_trace(base.shape[0], args.live, args.batch, args.epochs,
                            args.seed)
    report = {"schema_version": 1,
              "workload": {"live_vectors": trace.live, "batch": args.batch,
                           "phases": len(trace.phases), "mode": "snapshot",
                           "dimensions": int(base.shape[1]),
                           "metric": args.metric, "dataset": args.dataset,
                           "k": args.k},
              "engines": {}}
    del base, queries
    for name in args.engines:
        report["engines"][name] = _run_isolated(name, args)

    print(f"\n{args.live:,} live vectors, one snapshot pinned for the whole run\n")
    print("what each engine can actually do:")
    print(format_capability_matrix(args.engines) + "\n")
    for name, payload in report["engines"].items():
        if "phases" not in payload:
            print(f"{name:<16}  skipped: {payload['skipped']}")
            continue
        print(f"{name} pinned at t={payload['pinned_at']}")
        print(f"{'phase':>10}{'live':>9}{'pinned':>9}{'now':>9}"
              f"{'pin ms':>9}{'now ms':>9}{'bytes/vec':>11}{'retained':>10}")
        for row in payload["phases"]:
            stats = row["stats"]
            print(f"{row['kind']:>10}{row['live_vectors']:>9,}"
                  f"{row['pinned_recall_at_k']:>9.4f}"
                  f"{row['current_recall_at_k']:>9.4f}"
                  f"{row['pinned_p50_ms']:>9.3f}{row['current_p50_ms']:>9.3f}"
                  f"{stats['bytes_per_live_vector']:>11.0f}"
                  f"{stats['retained_versions']:>10,}")
        release = payload["release"]
        print(f"  release: reclaimed {release['reclaimed']:,} in "
              f"{release['seconds'] * 1000:.0f}ms, bytes/vector "
              f"{release['stats_before']['bytes_per_live_vector']:.0f} -> "
              f"{release['stats_after']['bytes_per_live_vector']:.0f}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.output}")


def _run_equal_recall(args: argparse.Namespace, base: np.ndarray,
                      queries: np.ndarray) -> None:
    """Compare engines only where their recall matches."""
    trace = build_ops_trace(base.shape[0], args.live, args.batch, 1, args.seed)
    report = {"schema_version": 1,
              "workload": {"live_vectors": trace.live, "mode": "equal-recall",
                           "dimensions": int(base.shape[1]),
                           "metric": args.metric, "dataset": args.dataset,
                           "k": args.k, "queries": int(queries.shape[0])},
              "engines": {}}
    del base, queries
    for name in args.engines:
        report["engines"][name] = _run_isolated(name, args)

    print(f"\n{args.live:,} live vectors, search knob swept, "
          f"engines compared where recall matches\n")
    print(f"{'engine':<15}{'build/s':>10}{'insert/s':>11}   recall targets")
    for name, payload in report["engines"].items():
        if "at_target" not in payload:
            print(f"{name:<15}  skipped: {payload['skipped']}")
            continue
        pieces = []
        for want, got in payload["at_target"].items():
            pieces.append(f"{want}: unreached (best {got['best_recall']:.3f})"
                          if got.get("unreached")
                          else f"{want}: {got['qps']:,.0f}qps@{got['setting']}")
        print(f"{name:<15}{payload['build_ops_per_s']:>10,.0f}"
              f"{(payload['insert_ops_per_s'] or 0):>11,.0f}   "
              + "  ".join(pieces))
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.output}")


def _run_filtered(args: argparse.Namespace, base: np.ndarray,
                  queries: np.ndarray) -> None:
    """Filtered search, swept over how selective the filter is."""
    trace = build_ops_trace(base.shape[0], args.live, args.batch, 1, args.seed)
    report = {"schema_version": 1,
              "workload": {"live_vectors": trace.live, "mode": "filtered",
                           "tag_layout": args.tag_layout,
                           "dimensions": int(base.shape[1]),
                           "metric": args.metric, "dataset": args.dataset,
                           "k": args.k, "queries": int(queries.shape[0])},
              "engines": {}}
    del base, queries
    for name in args.engines:
        report["engines"][name] = _run_isolated(name, args)

    print(f"\n{args.live:,} live vectors, one attribute filter laid out "
          f"{args.tag_layout}, swept over how much of the corpus it matches"
          + (f", then {args.churn_rounds} rounds of "
             f"{args.churn_fraction:.0%} churn" if args.churn_rounds else "")
          + "\n")
    print("what each engine can actually do:")
    print(format_capability_matrix(args.engines) + "\n")
    header = (f"{'engine':<15}{'mode':<8}{'match':>7}{'param':>7}"
              f"{'recall':>9}{'short':>7}{'fetch':>8}{'p50 ms':>9}"
              f"{'p99 ms':>9}{'vs all':>8}")
    if args.churn_rounds:
        header += f"{'churned':>9}{'p50 x':>8}{'all x':>8}{'d recall':>10}"
    print(header)
    for name, payload in report["engines"].items():
        if "selectivities" not in payload:
            print(f"{name:<15}  skipped: {payload['skipped']}")
            continue
        for row in payload["selectivities"]:
            if "unsupported" in row:
                print(f"{name:<15}{payload['filtering']:<8}"
                      f"{row['selectivity']:>6.0%}  {row['unsupported'][:40]}")
                continue
            line = (f"{name:<15}{payload['filtering']:<8}"
                    f"{row['selectivity']:>6.0%}"
                    f"{row.get('search_param', 0):>7}"
                    + ("*" if not row.get("met_target", True) else " ")
                    + f"{row['recall_at_k']:>8.4f}"
                    f"{row['short_result_rate']:>7.0%}"
                    + (f"{row.get('fetch_width', 0):>8,}"
                       if row.get('fetch_width') else f"{'-':>8}")
                    + f"{row['latency']['p50_ms']:>9.3f}"
                    f"{row['latency']['p99_ms']:>9.3f}"
                    f"{row['filter_speedup']:>8.2f}")
            churned = row.get("after_churn")
            if churned:
                line += (f"{churned['churned']:>8.0%}"
                         f"{churned['p50_ratio']:>8.2f}"
                         f"{churned['unfiltered_p50_ratio']:>8.2f}"
                         f"{churned['recall_delta']:>+10.4f}")
            elif args.churn_rounds:
                stalled = next((r["unsupported"] for r in row["rounds"]
                                if "unsupported" in r), "")
                line += f"   {stalled[:30]}" if stalled else ""
            print(line)
    if args.churn_rounds:
        print("\n'vs all' is unfiltered p50 over filtered p50: above 1.0 means "
              "the filter\nsaves work rather than costing it. 'p50 x' is "
              "filtered latency after churn\nover latency on the freshly built "
              "index, and 'all x' is the same ratio for\nan unfiltered search "
              "-- if they moved together, churn slowed the engine down\nand "
              "did not erode the filter specifically.")
    print(f"\nEach engine is shown at the cheapest setting on its own search "
          f"ladder that\nreaches recall {args.target_recall:.2f}; 'param' is "
          f"that setting. A '*' means the\nengine never reached the target "
          f"and is shown at its best. Matching the setting\ninstead of the "
          f"recall would compare operating points rather than engines: a\n"
          f"filter shrinks the pool a given setting draws from, and by a "
          f"different amount\nfor each engine and each selectivity.")
    print("\n'short' is the fraction of queries that could not return k "
          "results, and\n'fetch' is how many candidates a post-filtering "
          "engine had to pull to get k\nafter discarding. That width is sized "
          "from the selectivity, so those engines\nanswer correctly and the "
          "comparison is what the correctness costs them.")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="streambench", description=__doc__)
    parser.add_argument("--dataset", required=True, help="ann-benchmarks HDF5 file")
    parser.add_argument("--live", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--turnover", type=float, default=1.0,
                        help="fraction of the live set replaced per epoch")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    parser.add_argument("--engines", nargs="+", default=["chronovec", "hnswlib"])
    parser.add_argument("--list-engines", action="store_true")
    parser.add_argument("--mode",
                        choices=["bulk", "interleaved", "ops", "mixed",
                                 "concurrent", "snapshot", "equal-recall",
                                 "filtered"],
                        default="bulk",
                        help="bulk replays an epoch's delta in one call; "
                             "interleaved does one replacement at a time, which "
                             "is what agent memory and CDC look like; ops times "
                             "insert, update, delete and replace separately; "
                             "concurrent searches while one thread mutates; "
                             "snapshot pins a point in time and reads it; "
                             "equal-recall sweeps the search knob and compares "
                             "engines only where their recall matches; "
                             "filtered restricts a search to an attribute and "
                             "sweeps how selective the filter is")
    parser.add_argument("--readers", type=int, default=3,
                        help="reader threads in --mode concurrent")
    parser.add_argument("--batch", type=int, default=5000,
                        help="operations per phase in --mode ops")
    parser.add_argument("--selectivities", nargs="+", type=float,
                        default=[0.5, 0.1, 0.01],
                        help="in --mode filtered, the fractions of the corpus "
                             "the filter matches")
    parser.add_argument("--target-recall", type=float, default=0.95,
                        help="in --mode filtered, the recall each engine is "
                             "tuned up to before its latency is compared")
    parser.add_argument("--tag-layout",
                        choices=("random", "clustered", "semantic"),
                        default="random",
                        help="in --mode filtered, how the filtered attribute "
                             "is distributed. 'random' scatters it, which "
                             "defeats any page-level grouping and is the worst "
                             "case; 'clustered' gives it to a contiguous run "
                             "of ids; 'semantic' gives it to records near an "
                             "anchor vector, which is the only layout that "
                             "correlates with how a similarity index places "
                             "records and so the only one that tests "
                             "page-level skipping")
    parser.add_argument("--churn-rounds", type=int, default=0,
                        help="in --mode filtered, rounds of mutation to apply "
                             "after the first measurement, re-measuring after "
                             "each one")
    parser.add_argument("--churn-fraction", type=float, default=0.25,
                        help="fraction of the live set replaced per churn round")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    if args.list_engines:
        for name, ready in available_engines().items():
            print(f"  {name:16s} {'available' if ready else 'not installed'}")
        return

    base, queries = load(args.dataset, args.queries, args.metric)
    if args.mode in ("ops", "mixed", "concurrent", "snapshot", "equal-recall",
                     "filtered"):
        {"ops": _run_ops, "mixed": _run_ops, "concurrent": _run_concurrent,
         "snapshot": _run_snapshot, "equal-recall": _run_equal_recall,
         "filtered": _run_filtered}[args.mode](args, base, queries)
        return
    trace = build_trace(base.shape[0], args.live, args.epochs, args.turnover, args.seed)
    report = {
        "schema_version": 1,
        "workload": {"live_vectors": trace.live, "epochs": len(trace.epochs),
                     "replacements_per_epoch": trace.replacements_per_epoch,
                     "dimensions": int(base.shape[1]), "metric": args.metric,
                     "queries": int(queries.shape[0]), "k": args.k,
                     "turnover": args.turnover, "dataset": args.dataset,
                     "mode": args.mode,
                     "note": "one operation trace replayed against every engine"},
        "engines": {},
    }
    del base, queries          # each worker loads its own copy
    for name in args.engines:
        report["engines"][name] = _run_isolated(name, args)

    shape = report["workload"]
    print(f"\n{args.live:,} live vectors, {args.epochs} epochs at "
          f"{args.turnover:.0%} turnover, d={shape['dimensions']}, "
          f"metric={args.metric}, mode={args.mode}\n")
    print(f"{'engine':<16}{'del':>5}{'epoch':>7}{'recall':>9}{'ops/s':>11}{'p99 ms':>9}")
    for name, payload in report["engines"].items():
        if "epochs" not in payload:
            print(f"{name:<16}  skipped: {payload['skipped']}")
            continue
        mark = "yes" if payload["can_delete"] else "no"
        for row in payload["epochs"]:
            print(f"{name:<16}{mark:>5}{row['epoch']:>7}{row['recall_at_k']:>9.4f}"
                  f"{row['write_ops_per_s']:>11.0f}{row['latency']['p99_ms']:>9.3f}")
            mark = ""
    if args.output:
        with open(args.output, "w") as handle:
            json.dump(report, handle, indent=1)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
