"""Fetch the public ANN-Benchmarks datasets used by the comparison suite.

The large files stay out of Git, but a named download command keeps the public
benchmark reproducible instead of making each contributor guess a URL or file
name. Downloads use a temporary sibling and are renamed only after completion,
so an interrupted transfer is never mistaken for a usable dataset.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

DATASETS = {
    "sift-128-euclidean": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove-25-angular": "https://ann-benchmarks.com/glove-25-angular.hdf5",
    "gist-960-euclidean": "https://ann-benchmarks.com/gist-960-euclidean.hdf5",
}


def fetch(name: str, destination: Path) -> Path:
    """Download one canonical dataset and return its completed path."""
    target = destination / f"{name}.hdf5"
    temporary = target.with_suffix(f"{target.suffix}.partial")
    request = Request(DATASETS[name], headers={"User-Agent": "chronovec-benchmarks/1"})
    print(f"downloading {name} -> {target}")
    try:
        with urlopen(request) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=sorted(DATASETS),
        help="canonical dataset names to download",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark_data"),
        help="directory for HDF5 files (default: benchmark_data)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace files that are already present",
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        target = args.data_dir / f"{name}.hdf5"
        if target.exists() and not args.force:
            print(f"already present: {target} (use --force to replace)")
            continue
        fetch(name, args.data_dir)


if __name__ == "__main__":
    main()
