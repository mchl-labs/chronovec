"""A fast, fixed workload that says whether a change made things worse.

This is not the comparison benchmark. That one takes hours, runs against other
engines, and answers "where do we stand". This answers a different question --
"did this commit cost anything" -- and it has to be fast enough to run on every
change, or it will only ever catch a regression weeks after it landed.

Three things it does that the comparison benchmark got wrong repeatedly:

* Throughput is measured **at equal recall**, never at a fixed nprobe. A change
  that packs pages fuller scans more rows per probe, which looks like an 18%
  query regression at fixed nprobe and is a 15% improvement at equal recall.
  That exact mistake was made and had to be undone.
* Every metric is reported, not just the failing one, because most real changes
  are trades. "query -5%, memory -20%" is a decision for a person to make; a
  bare FAIL is not.
* Runs are best-of-N after a warm-up, and the gate refuses to compare when its
  own repeats disagree by more than the tolerance it would judge by -- a noisy
  machine should abstain rather than report a verdict it cannot support.

Usage:
    python benchmarks/regression_gate.py --update    # record a baseline
    python benchmarks/regression_gate.py             # judge against it
    python benchmarks/regression_gate.py --output results/run.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
from pathlib import Path

import numpy as np

BASELINE = Path(__file__).parent / "regression_baseline.json"

# Per metric: how much worse is tolerated, and which direction is better.
# Throughput tolerances are wide because the machine is not quiet; recall and
# space are tight because they are properties of the algorithm rather than of
# the machine, and should not move at all without someone meaning it.
TOLERANCE = {
    "build_us_per_row": (0.15, "lower"),
    "insert_single_us": (0.15, "lower"),
    "delete_single_us": (0.15, "lower"),
    "vacuum_s": (0.20, "lower"),
    "qps_at_recall_0.90": (0.12, "higher"),
    "qps_at_recall_0.95": (0.12, "higher"),
    "qps_at_recall_0.99": (0.12, "higher"),
    "best_recall": (0.01, "higher"),
    "amplification": (0.05, "lower"),
    "bytes_per_live_vector": (0.05, "lower"),
    "amplification_after_churn": (0.08, "lower"),
    "recall_after_churn": (0.02, "higher"),
    # Filtered search. Recall and fill are held tightly because the failure
    # mode here is silent -- an index that gives up early looks faster -- so
    # the latency entries are deliberately loose and exist to price a recall
    # change, not to gate on their own.
    "filtered_random_10pct_recall": (0.02, "higher"),
    "filtered_random_1pct_recall": (0.02, "higher"),
    "filtered_semantic_10pct_recall": (0.02, "higher"),
    "filtered_semantic_1pct_recall": (0.02, "higher"),
    "filtered_random_10pct_filled": (0.01, "higher"),
    "filtered_random_1pct_filled": (0.01, "higher"),
    "filtered_semantic_10pct_filled": (0.01, "higher"),
    "filtered_semantic_1pct_filled": (0.01, "higher"),
    "filtered_random_10pct_us": (0.25, "lower"),
    "filtered_random_1pct_us": (0.25, "lower"),
    "filtered_semantic_10pct_us": (0.25, "lower"),
    "filtered_semantic_1pct_us": (0.25, "lower"),
}


def calibration_ms() -> float:
    """A fixed amount of arithmetic, unrelated to anything under test.

    The gate compares timings against a baseline recorded on some earlier run
    of this machine, which is only meaningful if the machine is in the same
    state. Twice in one day a loaded machine produced a confident five-metric
    regression report here, and the existing abstention could not see it: that
    rule fires when repeats disagree, and under *sustained* load every repeat
    is slow together, so the spread stays small and the verdict reads WORSE.

    This measures the machine rather than the code. If it has drifted from the
    baseline, timings are not comparable and no timing verdict is worth
    printing, however consistent the repeats were.
    """
    values = np.arange(1 << 16, dtype=np.float64)
    started = time.perf_counter()
    for _ in range(80):
        values = values * 1.000001 + 0.5
        float(values.sum())
    return (time.perf_counter() - started) * 1000.0


def resident_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if os.uname().sysname == "Darwin" else 1024)


def load(dataset: str, live: int, queries: int):
    import h5py

    with h5py.File(dataset, "r") as handle:
        pool = np.asarray(handle["train"][: live * 2], dtype=np.float32)
        test = np.asarray(handle["test"][:queries], dtype=np.float32)
    return pool, test


def exact_truth(base, queries, k):
    scores = (queries**2).sum(1)[:, None] - 2.0 * (queries @ base.T) + (base * base).sum(1)[None, :]
    return np.argpartition(scores, k, axis=1)[:, :k]


def recall_curve(index, queries, truth, k, ladder):
    """(recall, qps) at each probe count, best of two timed passes."""
    curve = []
    for probe in ladder:
        for q in queries[:20]:
            index.search(q, k, nprobe=probe)
        hits = sum(
            len({r.id for r in index.search(queries[i], k, nprobe=probe)} & set(truth[i].tolist()))
            for i in range(len(queries))
        )
        best = None
        for _ in range(2):
            started = time.perf_counter()
            index.search_many(queries, k, nprobe=probe, as_arrays=True)
            spent = time.perf_counter() - started
            best = spent if best is None else min(best, spent)
        curve.append((hits / (k * len(queries)), len(queries) / best))
    return curve


def at_recall(curve, target):
    reached = [qps for recall, qps in curve if recall >= target]
    return max(reached) if reached else None


def once(pool, queries, live, k) -> dict:
    from chronovec import Index

    base = pool[:live]
    truth = exact_truth(base, queries, k)
    out = {}

    index = Index(base.shape[1], metric="l2", page_capacity=256, nprobe=64)
    started = time.perf_counter()
    index.insert_many(np.arange(live), base)
    out["build_us_per_row"] = (time.perf_counter() - started) / live * 1e6

    stats = index.stats()
    out["amplification"] = stats["capacity_amplification"]
    out["bytes_per_live_vector"] = stats["tracked_index_bytes"] / max(1, stats["live_vectors"])

    curve = recall_curve(index, queries, truth, k, (8, 16, 32, 48, 64, 96, 128, 192))
    for target in (0.90, 0.95, 0.99):
        out[f"qps_at_recall_{target:.2f}"] = at_recall(curve, target)
    out["best_recall"] = max(recall for recall, _ in curve)

    started = time.perf_counter()
    for position in range(2000):
        index.insert(live + position, pool[live + position])
    out["insert_single_us"] = (time.perf_counter() - started) / 2000 * 1e6

    started = time.perf_counter()
    for position in range(2000):
        index.delete(live + position)
    out["delete_single_us"] = (time.perf_counter() - started) / 2000 * 1e6

    started = time.perf_counter()
    index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
    out["vacuum_s"] = time.perf_counter() - started

    # Space and recall after churn, which is where the regressions this project
    # actually had would show up. A build-only gate would have missed every one.
    ids = np.arange(live)
    nxt = live
    for _ in range(3):
        index.delete_many(ids)
        ids = np.arange(nxt, nxt + live)
        index.insert_many(ids, pool[np.arange(nxt, nxt + live) % len(pool)])
        nxt += live
        index.vacuum(oldest_snapshot=index.clock + 1, budget_versions=0)
    stats = index.stats()
    out["amplification_after_churn"] = stats["capacity_amplification"]
    churned = pool[np.arange(nxt - live, nxt) % len(pool)]
    churn_truth = ids[exact_truth(churned, queries, k)]
    hits = sum(
        len({r.id for r in index.search(queries[i], k)} & set(churn_truth[i].tolist()))
        for i in range(len(queries))
    )
    out["recall_after_churn"] = hits / (k * len(queries))
    index.close()

    out.update(filtered(base, queries, k))
    gc.collect()
    return out


def filtered(base, queries, k) -> dict:
    """Filtered recall and latency, under both layouts of the filtered value.

    This block exists because the gate did not have it and a filtered search
    that returned 1.6 of 10 requested neighbours passed every other metric
    here. Recall is the metric that catches it: an index that stops looking
    early gets *faster*, so latency alone reads the failure as an improvement.

    Both layouts are measured because they exercise different machinery.
    `random` scatters the value so every page holds a mixture and only
    per-record rejection ahead of the distance can pay. `semantic` gives it to
    the records around one anchor, which is what a tenant or a language looks
    like and is the only layout under which whole pages can be ruled out. A
    change that helps one can quietly cost the other.
    """
    from chronovec import Index

    out = {}
    count = base.shape[0]
    for layout in ("random", "semantic"):
        for fraction, tag in ((0.10, "10pct"), (0.01, "1pct")):
            rng = np.random.default_rng(4)
            marked = max(k + 1, int(count * fraction))
            tags = np.full(count, 2, dtype=np.uint64)
            if layout == "semantic":
                anchor = base[rng.integers(0, count)]
                gap = np.linalg.norm(base - anchor, axis=1)
                chosen = np.argpartition(gap, marked - 1)[:marked]
            else:
                chosen = rng.choice(count, size=marked, replace=False)
            tags[chosen] = 1

            index = Index(base.shape[1], metric="l2", page_capacity=256, nprobe=64, labels=True)
            index.insert_many(np.arange(count), base, labels=tags)
            matching, keys = base[chosen], np.arange(count)[chosen]
            truth = [
                set(keys[np.argsort(np.linalg.norm(matching - q, axis=1))[:k]].tolist())
                for q in queries
            ]
            started = time.perf_counter()
            found = [{r.id for r in index.search(q, k, require_all=1)} for q in queries]
            elapsed = time.perf_counter() - started
            stem = f"filtered_{layout}_{tag}"
            out[f"{stem}_recall"] = sum(len(f & t) for f, t in zip(found, truth)) / (
                k * len(queries)
            )
            # Whether k was returned at all, kept separate from recall: an
            # index can return k wrong records or the right records short, and
            # the two have different causes.
            out[f"{stem}_filled"] = sum(len(f) == k for f in found) / len(queries)
            out[f"{stem}_us"] = elapsed / len(queries) * 1e6
            index.close()
    return out


def measure(dataset, live, queries_n, k, repeats) -> dict:
    pool, queries = load(dataset, live, queries_n)
    runs = [once(pool, queries, live, k) for _ in range(repeats)]
    merged = {}
    for key in runs[0]:
        values = [run[key] for run in runs if run[key] is not None]
        if not values:
            merged[key] = None
            continue
        better = TOLERANCE.get(key, (0.1, "higher"))[1]
        merged[key] = max(values) if better == "higher" else min(values)
        # How far the repeats disagreed, so the gate can abstain when the
        # machine is too noisy to support a verdict.
        if len(values) > 1 and merged[key]:
            merged[f"{key}__spread"] = (max(values) - min(values)) / abs(merged[key])
    merged["resident_mb"] = resident_mb()
    # Taken last, so it reflects the machine as it was while measuring rather
    # than before the run started.
    merged["_calibration_ms"] = min(calibration_ms() for _ in range(3))
    return merged


def commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def working_tree_dirty() -> bool:
    try:
        return bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except Exception:
        return True


def host_identity() -> str:
    """A stable, non-identifying host token for benchmark comparisons.

    ``platform.platform()`` includes the operating-system patch version, so a
    routine OS update used to make the gate claim a baseline came from another
    machine.  Hashing the host name and hardware architecture distinguishes
    hosts without recording the host name in a committed artifact.
    """
    source = "|".join((platform.node(), platform.machine(), platform.processor()))
    return hashlib.sha256(source.encode()).hexdigest()[:16]


# How far the calibration may drift before timings stop being comparable.
# Wide enough not to fire on ordinary variation, narrow enough to catch the
# load that produced a 35% shift across every timing metric at once.
CALIBRATION_TOLERANCE = 0.15

# Metrics that measure the machine as much as the code. A drifted calibration
# invalidates these; it says nothing about recall, fill or space, which are
# deterministic and stay judged.
TIMING_METRICS = tuple(
    key
    for key in TOLERANCE
    if key.endswith(("_us", "_s", "_ms", "_us_per_row")) or key.startswith("qps_")
)


def judge(current: dict, baseline: dict) -> int:
    was_cal, now_cal = baseline.get("_calibration_ms"), current.get("_calibration_ms")
    drift = (now_cal - was_cal) / was_cal if was_cal and now_cal else 0.0
    calibration_drifted = abs(drift) > CALIBRATION_TOLERANCE
    if calibration_drifted:
        print(
            f"Machine calibration has moved {drift:+.1%} against the "
            f"baseline ({was_cal:.1f}ms -> {now_cal:.1f}ms).\n"
            f"Timing metrics are not comparable on this run and are reported "
            f"without a verdict.\nRecall, fill and space are deterministic "
            f"and still judged.\n"
        )

    # A same-machine calibration probe (a generic numpy arithmetic loop) does
    # not stand in for cross-machine comparability: two genuinely different
    # machines (e.g. the baseline recorded on a developer's Mac, judged
    # against a GitHub Actions Linux runner) can calibrate as "close enough"
    # on that generic probe while the actual compiled engine's timing differs
    # by an order of magnitude from real hardware/OS/compiler differences --
    # this was observed in CI as every timing metric reading 40-580% "WORSE"
    # against a macOS baseline, with recall/fill/space (deterministic,
    # unaffected by machine) all exactly unchanged. A host mismatch, by
    # either signal available, withholds timing verdicts unconditionally.
    baseline_host = baseline.get("_host_id")
    if baseline_host is not None:
        host_mismatched = baseline_host != current.get("_host_id")
    else:
        # Legacy baseline predates _host_id; _machine (OS family only) is
        # the coarsest signal available, but still catches the case that
        # actually happened here.
        host_mismatched = baseline.get("_machine") != current.get("_machine")
    if host_mismatched:
        print(
            f"Baseline was recorded on a different machine "
            f"({baseline.get('_machine', '?')} vs. {current.get('_machine', '?')}); "
            "timing metrics are not comparable on this run and are reported "
            "without a verdict.\nRecall, fill and space are deterministic and "
            "still judged.\n"
        )
    untrusted = calibration_drifted or host_mismatched
    print(f"{'metric':<34}{'baseline':>12}{'current':>12}{'change':>10}  verdict")
    worse, noisy = [], []
    for key, (tolerance, better) in TOLERANCE.items():
        was, now = baseline.get(key), current.get(key)
        if was is None or now is None:
            # A metric the baseline predates still gets its current value
            # printed. Hiding it made a newly added metric indistinguishable
            # from one that failed to measure, which is the opposite of what
            # adding it was for.
            shown = f"{now:,.4g}" if now is not None else "-"
            # Reaching an equal-recall target is itself a quality guarantee.
            # Treating a target that used to be reachable as merely
            # informational let a complete loss of the 0.99 frontier pass the
            # gate.  A target absent from an older baseline remains a new,
            # unjudged measurement; one absent from both is likewise only
            # diagnostic.
            lost_target = was is not None and now is None
            label = (
                "WORSE (not reached)"
                if lost_target
                else "new"
                if was is None and now is not None
                else "not reached"
            )
            print(f"{key:<34}{'-':>12}{shown:>12}{'-':>10}  {label}")
            if lost_target:
                worse.append((key, float("-inf")))
            continue
        change = (now - was) / was
        improved = change >= 0 if better == "higher" else change <= 0
        magnitude = abs(change)
        spread = current.get(f"{key}__spread", 0.0)
        if untrusted and key in TIMING_METRICS:
            print(f"{key:<34}{was:>12,.4g}{now:>12,.4g}{change:>+9.1%}  untrusted")
            continue
        if improved or magnitude <= tolerance:
            verdict = "ok" if improved or magnitude < tolerance / 2 else "within"
        elif spread > tolerance:
            verdict = "NOISY"
            noisy.append(key)
        else:
            verdict = "WORSE"
            worse.append((key, change))
        print(f"{key:<34}{was:>12,.4g}{now:>12,.4g}{change:>+9.1%}  {verdict}")

    print()
    if noisy:
        print("Repeats disagreed by more than the tolerance on: " + ", ".join(noisy))
        print("The machine is too busy to support a verdict on those. Re-run.")
    if host_mismatched:
        print(
            "\nCompare against a baseline recorded on this same machine to trust any timing here."
        )
    elif calibration_drifted:
        print("\nRe-run on an idle machine before trusting any timing here.")
    if worse:
        print("Regressions:")
        for key, change in worse:
            print(f"  {key} {change:+.1%}")

        # An improvement is a move beyond a floor in the better direction.
        # Testing only the sign counted anything that barely moved -- a metric
        # 1.9% *worse* was being reported as improved.
        def moved_better(key: str, floor: float = 0.02) -> bool:
            was_, now_ = baseline.get(key), current.get(key)
            if not was_ or now_ is None:
                return False
            shift = (now_ - was_) / was_
            return shift > floor if TOLERANCE[key][1] == "higher" else shift < -floor

        improvements = [key for key in TOLERANCE if moved_better(key)]
        if improvements:
            print("Improved at the same time: " + ", ".join(improvements))
            print("This is a trade, not a failure. Decide whether it is worth it.")
        return 1
    print("No regression beyond tolerance.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="benchmark_data/sift-128-euclidean.hdf5")
    parser.add_argument("--live", type=int, default=50000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="compare against this freshly measured result instead of the committed baseline",
    )
    parser.add_argument(
        "--update", action="store_true", help="record the current numbers as the baseline"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the measured result as JSON without changing the baseline",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="emit a raw measurement without consulting or updating any baseline",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    current = measure(args.dataset, args.live, args.queries, args.k, args.repeats)
    current["_commit"] = commit()
    current["_working_tree_dirty"] = working_tree_dirty()
    # platform.system() only -- "Darwin"/"Linux"/"Windows", not the full
    # platform.platform() string, which bakes in the OS patch version and
    # CPU architecture and would fingerprint the specific device in a
    # committed artifact. host_identity() below is the actual same-machine
    # check; this is just enough to explain a cross-OS-family comparison.
    current["_machine"] = platform.system()
    current["_host_id"] = host_identity()
    current["_python"] = platform.python_version()
    current["_numpy"] = np.__version__
    current["_dataset"] = str(Path(args.dataset))
    current["_live"] = args.live
    current["_queries"] = args.queries
    current["_k"] = args.k
    current["_repeats"] = args.repeats
    current["_seconds"] = round(time.perf_counter() - started, 1)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"wrote raw result to {args.output}")
    print(f"measured in {current['_seconds']}s at {current['_commit']}\n")

    if args.measure_only:
        return 0

    baseline_path = args.baseline or BASELINE
    if args.update or not baseline_path.exists():
        baseline_path.write_text(json.dumps(current, indent=1, sort_keys=True))
        print(f"wrote baseline to {baseline_path}")
        if not args.update:
            print("(no baseline existed; nothing to compare against)")
        return 0

    baseline = json.loads(baseline_path.read_text())
    print(f"baseline from {baseline.get('_commit', '?')} on {baseline.get('_machine', '?')}\n")
    # judge() detects a host mismatch itself (via _host_id, or _machine for a
    # legacy baseline predating it) and withholds timing verdicts for it,
    # the same way it withholds them for same-machine calibration drift.
    return judge(current, baseline)


if __name__ == "__main__":
    raise SystemExit(main())
