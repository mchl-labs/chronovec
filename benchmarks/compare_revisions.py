"""Compare a base revision and the current tree on one host, back-to-back.

The committed regression baseline is useful history, but it cannot establish a
timing result after an OS update, a hardware power-state change, or a long time
gap.  This runner measures the requested base commit in a temporary worktree,
then treats that raw measurement as the current tree's baseline.  Both runs
therefore use the same host, dataset, Python invocation, and gate workload.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from regression_gate import host_identity


def run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def library_path(build: Path) -> Path:
    if sys.platform == "darwin":
        return build / "libchronovec.dylib"
    if sys.platform == "win32":
        return build / "chronovec.dll"
    return build / "libchronovec.so"


def measure(
    repo: Path, output: Path, args: argparse.Namespace, baseline: Path | None = None
) -> None:
    build = repo / ".benchmark-build"
    run(["cmake", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], cwd=repo)
    run(["cmake", "--build", str(build), "--parallel"], cwd=repo)
    env = os.environ.copy()
    env["CHRONOVEC_LIBRARY"] = str(library_path(build))
    command = [
        "uv",
        "run",
        "--with",
        "h5py",
        "python",
        "benchmarks/regression_gate.py",
        "--dataset",
        str(args.dataset),
        "--live",
        str(args.live),
        "--queries",
        str(args.queries),
        "--k",
        str(args.k),
        "--repeats",
        str(args.repeats),
        "--output",
        str(output),
    ]
    if baseline is not None:
        command.extend(["--baseline", str(baseline)])
        run(command, cwd=repo, env=env)
        return

    # A raw measurement, not a judged comparison: without --measure-only,
    # the base revision's own regression_gate.py falls back to comparing
    # against whatever static regression_baseline.json happens to be
    # checked out in this worktree -- unrelated to what this call is
    # measuring, and a spurious "regression" against it would crash this
    # entire comparison for a reason that has nothing to do with the base
    # revision actually being measured.
    command.append("--measure-only")
    run(command, cwd=repo, env=env)


def stamp_host_identity(result: Path) -> None:
    """Backfill the stable host token when the base predates that field."""
    value = json.loads(result.read_text())
    value.setdefault("_host_id", host_identity())
    result.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="base commit or ref (default: HEAD)")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("benchmark_data/sift-128-euclidean.hdf5"),
    )
    parser.add_argument("--live", type=int, default=50_000)
    parser.add_argument("--queries", type=int, default=300)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--keep-worktree", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="preserve base.json and current.json here for a defaulting-gate artifact",
    )
    args = parser.parse_args()
    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    args.dataset = args.dataset.resolve()
    if args.output_dir is not None:
        args.output_dir = (root / args.output_dir).resolve()
    if not args.dataset.is_file():
        raise SystemExit(f"benchmark dataset does not exist: {args.dataset}")
    temporary = Path(tempfile.mkdtemp(prefix="chronovec-revision-gate-"))
    base_tree = temporary / "base"
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        base_result = args.output_dir / "base.json"
        current_result = args.output_dir / "current.json"
    else:
        base_result = temporary / "base.json"
        current_result = temporary / "current.json"
    try:
        run(["git", "worktree", "add", "--detach", str(base_tree), args.base], cwd=root)
        measure(base_tree, base_result, args)
        stamp_host_identity(base_result)
        measure(root, current_result, args, baseline=base_result)
        if args.output_dir is not None:
            print(f"\nfresh base measurement: {base_result}")
            print(f"fresh current measurement: {current_result}")
        else:
            print("\nfresh revision comparison complete (pass --output-dir to retain raw JSON).")
    finally:
        if not args.keep_worktree:
            subprocess.run(["git", "worktree", "remove", "--force", str(base_tree)], cwd=root)
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
