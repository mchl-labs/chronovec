"""Metadata-history memory experiment: full-copy vs. delta-chain versioning.

Question: `Collection._history` (chronovec/collection.py) stores a full dict
copy on every version, including a metadata-only `update()` that changes one
key out of many. Cloudflare's DNS cache write-up (blog.cloudflare.com,
2024-11-11) cut per-entry memory by storing only what differs from an inferable
default instead of a full copy each time. Does the same idea -- a delta chain
with periodic keyframes, instead of a full dict per version -- cut real bytes
here, and at what read cost?

Method: simulate N ids, each starting with a W-key metadata dict and receiving
`updates` partial updates that each touch `touch` keys (the rest carry
forward unchanged) -- the shape of an agent/streaming workload that edits a
status field on long-lived records. Measure actual heap bytes via tracemalloc
(one representation on the heap at a time, so neither run's measurement
includes the other) and the wall time to resolve every id's current metadata,
which is the real cost the read path (`Collection.get`/`query`) pays.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from typing import Any

_MISSING = object()

Chain = list[tuple[str, Any]]


def _random_metadata(width: int, rng: random.Random) -> dict[str, Any]:
    kinds = (
        lambda: rng.randint(0, 10_000),
        lambda: round(rng.random() * 100, 3),
        lambda: rng.choice(("pending", "active", "done", "failed", "queued")),
        lambda: f"tag-{rng.randint(0, 50)}",
    )
    return {f"k{i}": rng.choice(kinds)() for i in range(width)}


def _mutate(metadata: dict[str, Any], touch: int, rng: random.Random) -> dict[str, Any]:
    """One partial update: `touch` keys get new values, the rest carry forward."""
    out = dict(metadata)
    for key in rng.sample(list(out), min(touch, len(out))):
        out[key] = rng.choice((rng.randint(0, 10_000), f"tag-{rng.randint(0, 50)}"))
    return out


def build_naive(
    ids: list[int], width: int, updates: int, touch: int, rng: random.Random
) -> dict[int, list[dict[str, Any]]]:
    """Mirrors `Collection._history` today: every version is a full dict copy."""
    history: dict[int, list[dict[str, Any]]] = {}
    for key in ids:
        meta = _random_metadata(width, rng)
        versions = [dict(meta)]
        for _ in range(updates):
            meta = _mutate(meta, touch, rng)
            versions.append(dict(meta))
        history[key] = versions
    return history


def build_delta(
    ids: list[int],
    width: int,
    updates: int,
    touch: int,
    interval: int,
    rng: random.Random,
    *,
    flat: bool = False,
) -> dict[int, Chain]:
    """One full keyframe every `interval` versions; every other version stores
    only the keys that changed since the previous version.

    `flat=True` stores the changed keys as a tuple of (key, value) pairs
    instead of a nested dict -- CPython's dict container has a fixed overhead
    (hash table + entries array) that a 1-2 key `changed` dict pays in full,
    which is worth avoiding once the win is being measured in bytes.
    """
    history: dict[int, Chain] = {}
    for key in ids:
        meta = _random_metadata(width, rng)
        chain: Chain = [("full", dict(meta))]
        for i in range(1, updates + 1):
            new_meta = _mutate(meta, touch, rng)
            if i % interval == 0:
                chain.append(("full", dict(new_meta)))
            else:
                pairs = tuple((k, v) for k, v in new_meta.items() if meta.get(k, _MISSING) != v)
                removed = tuple(k for k in meta if k not in new_meta)
                changed = pairs if flat else dict(pairs)
                chain.append(("delta", (changed, removed)))
            meta = new_meta
        history[key] = chain
    return history


def resolve_naive(history: dict[int, list[dict[str, Any]]], key: int) -> dict[str, Any]:
    return dict(history[key][-1])


def resolve_delta(history: dict[int, Chain], key: int) -> dict[str, Any]:
    chain = history[key]
    j = len(chain) - 1
    while chain[j][0] != "full":
        j -= 1
    out = dict(chain[j][1])
    for _kind, payload in chain[j + 1 :]:
        changed, removed = payload
        if isinstance(changed, tuple):
            for k, v in changed:
                out[k] = v
        else:
            out.update(changed)
        for k in removed:
            out.pop(k, None)
    return out


def _traced_bytes(build, *args, **kwargs) -> tuple[Any, int]:
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    structure = build(*args, **kwargs)
    gc.collect()
    current, _peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return structure, current


def _resolve_seconds(history, ids: list[int], resolve, repeats: int) -> float:
    """Best-of-`repeats`, to keep GC pauses and scheduling noise from
    dominating a measurement that is a few milliseconds wide."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        for key in ids:
            resolve(history, key)
        best = min(best, time.perf_counter() - t0)
    return best


def run_one(
    ids: list[int],
    width: int,
    updates: int,
    touch: int,
    interval: int,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    naive, naive_bytes = _traced_bytes(build_naive, ids, width, updates, touch, random.Random(seed))
    naive_resolve_s = _resolve_seconds(naive, ids, resolve_naive, repeats)
    sample = ids[: min(200, len(ids))]
    expected = {key: resolve_naive(naive, key) for key in sample}
    del naive
    gc.collect()

    row: dict[str, Any] = {
        "touch": touch,
        "naive_bytes_per_id": naive_bytes / len(ids),
        "naive_resolve_s": naive_resolve_s,
    }
    for label, flat in (("delta", False), ("delta_flat", True)):
        structure, nbytes = _traced_bytes(
            build_delta, ids, width, updates, touch, interval, random.Random(seed), flat=flat
        )
        resolve_s = _resolve_seconds(structure, ids, resolve_delta, repeats)
        correct = all(resolve_delta(structure, key) == expected[key] for key in sample)
        del structure
        gc.collect()
        row[f"{label}_bytes_per_id"] = nbytes / len(ids)
        row[f"{label}_ratio"] = nbytes / naive_bytes
        row[f"{label}_resolve_s"] = resolve_s
        row[f"{label}_resolve_slowdown"] = (
            resolve_s / naive_resolve_s if naive_resolve_s else float("nan")
        )
        row[f"{label}_correct"] = correct
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, default=5_000)
    ap.add_argument("--width", type=int, default=12, help="metadata keys per record")
    ap.add_argument("--updates", type=int, default=30, help="partial updates per id after the add")
    ap.add_argument(
        "--touch", type=int, nargs="+", default=[1, 3, 12], help="keys changed per update; sweep"
    )
    ap.add_argument("--keyframe-interval", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=7, help="best-of-N for resolve timing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="experiments/results/metadata_history_sim.json")
    a = ap.parse_args()

    ids = list(range(a.ids))
    print(
        f"ids={a.ids} width={a.width} updates={a.updates} keyframe_interval={a.keyframe_interval}"
    )
    rows = []
    for touch in a.touch:
        row = run_one(ids, a.width, a.updates, touch, a.keyframe_interval, a.seed, a.repeats)
        rows.append(row)
        print(f"touch={touch:3d}/{a.width}  naive={row['naive_bytes_per_id']:7.0f} B/id")
        for label in ("delta", "delta_flat"):
            print(
                f"    {label:10s} {row[f'{label}_bytes_per_id']:7.0f} B/id  "
                f"ratio={row[f'{label}_ratio']:.3f}  "
                f"resolve {row['naive_resolve_s'] * 1e3:.1f}ms -> {row[f'{label}_resolve_s'] * 1e3:.1f}ms "
                f"({row[f'{label}_resolve_slowdown']:.2f}x)  "
                f"correct={row[f'{label}_correct']}"
            )

    with open(a.output, "w") as f:
        json.dump(
            {
                "ids": a.ids,
                "width": a.width,
                "updates": a.updates,
                "keyframe_interval": a.keyframe_interval,
                "seed": a.seed,
                "results": rows,
            },
            f,
            indent=1,
        )
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
