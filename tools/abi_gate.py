"""Fails a PR that breaks the C ABI without bumping CV_ABI_VERSION_MAJOR.

bindings/rust/chronovec-sys/native/include/chronovec.h already documents a versioning promise in its own
comments -- MAJOR for breaking changes, MINOR for additive ones, PATCH for
none. That promise was prose only: nothing enforced it, so a signature edit
or a struct-field reorder could ship silently and every language binding
built on this header (Rust today; Go and Node as of this change) would break
at link time or, worse, at a call site with a garbled struct.

This does the same shape of thing as benchmarks/regression_gate.py: parse a
snapshot of the current state, diff it against a committed baseline, and fail
loudly only on the thing that actually matters -- a breaking change with no
MAJOR bump. Additive changes (new function, new enum value) are always safe
to ship and only ask for a MINOR bump, so a missed one is reported, not
failed on: enforcing it strictly would make the gate cry wolf on every
feature addition, and a gate that fails on harmless changes gets skipped.

Usage:
    python tools/abi_gate.py             # judge the header against the baseline
    python tools/abi_gate.py --update    # record a new baseline (after a
                                          # deliberate, version-bumped change)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HEADER = Path(__file__).parent.parent / "bindings" / "rust" / "chronovec-sys" / "native" / "include" / "chronovec.h"
BASELINE = Path(__file__).parent.parent / "native" / "abi_baseline.json"

_VERSION_RE = re.compile(r"#define CV_ABI_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)")
_FUNC_RE = re.compile(r"CV_API\s+([^;]+?)\s*;", re.DOTALL)
_STRUCT_RE = re.compile(r"typedef\s+struct\s+\w*\s*\{([^}]*)\}\s*(\w+)\s*;", re.DOTALL)
_ENUM_RE = re.compile(r"enum\s*\{([^}]*)\}\s*;", re.DOTALL)
_FIELD_RE = re.compile(r"([\w \t\*]+?)\s+(\w+)\s*;")


def _normalize(sig: str) -> str:
    """Collapse the multi-line-wrapped whitespace ruff/clang-format leaves."""
    return re.sub(r"\s+", " ", sig).strip()


def parse_header(text: str) -> dict:
    """A structural snapshot of the ABI: function signatures, struct layouts,
    enum values, and the declared version. Comments are stripped first so a
    doc-only edit (the common case) never touches the snapshot.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)

    version = {m.group(1): int(m.group(2)) for m in _VERSION_RE.finditer(text)}

    functions = {}
    for m in _FUNC_RE.finditer(text):
        sig = _normalize(m.group(1))
        name_match = re.search(r"(\w+)\s*\(", sig)
        if name_match:
            functions[name_match.group(1)] = sig

    structs = {}
    for m in _STRUCT_RE.finditer(text):
        body, name = m.group(1), m.group(2)
        # Lists, not tuples: baseline comes back from JSON as lists-of-lists,
        # and comparing a freshly parsed tuple against a JSON list is always
        # unequal even when nothing changed.
        fields = [[_normalize(f_type), f_name] for f_type, f_name in _FIELD_RE.findall(body)]
        structs[name] = fields

    enum_values = {}
    for m in _ENUM_RE.finditer(text):
        next_value = 0
        for entry in m.group(1).split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                name, raw = (part.strip() for part in entry.split("=", 1))
                next_value = int(raw, 0)
            else:
                name = entry
            enum_values[name] = next_value
            next_value += 1

    return {
        "version": version,
        "functions": functions,
        "structs": structs,
        "enum_values": enum_values,
    }


def diff(current: dict, baseline: dict) -> tuple[list[str], list[str]]:
    """Returns (breaking, additive) as lists of human-readable descriptions."""
    breaking: list[str] = []
    additive: list[str] = []

    base_fns, cur_fns = baseline["functions"], current["functions"]
    for name in base_fns:
        if name not in cur_fns:
            breaking.append(f"function removed: {name}")
        elif cur_fns[name] != base_fns[name]:
            breaking.append(
                f"function signature changed: {name}\n"
                f"    was: {base_fns[name]}\n"
                f"    now: {cur_fns[name]}"
            )
    for name in cur_fns:
        if name not in base_fns:
            additive.append(f"function added: {name}")

    base_structs, cur_structs = baseline["structs"], current["structs"]
    for name in base_structs:
        if name not in cur_structs:
            breaking.append(f"struct removed: {name}")
        elif cur_structs[name] != base_structs[name]:
            breaking.append(
                f"struct layout changed: {name}\n"
                f"    was: {base_structs[name]}\n"
                f"    now: {cur_structs[name]}"
            )
    for name in cur_structs:
        if name not in base_structs:
            additive.append(f"struct added: {name}")

    base_enum, cur_enum = baseline["enum_values"], current["enum_values"]
    for name, value in base_enum.items():
        if name not in cur_enum:
            breaking.append(f"enum value removed: {name}")
        elif cur_enum[name] != value:
            breaking.append(f"enum value changed: {name} was {value}, now {cur_enum[name]}")
    for name in cur_enum:
        if name not in base_enum:
            additive.append(f"enum value added: {name}")

    return breaking, additive


def judge(current: dict, baseline: dict) -> int:
    breaking, additive = diff(current, baseline)

    if not breaking and not additive:
        print("No ABI change.")
        return 0

    major_bumped = current["version"]["MAJOR"] > baseline["version"]["MAJOR"]
    minor_bumped = current["version"]["MINOR"] > baseline["version"]["MINOR"] or major_bumped

    if additive:
        print("Additive ABI changes:")
        for line in additive:
            print(f"  + {line}")
        if not minor_bumped:
            print(
                "  note: CV_ABI_VERSION_MINOR was not bumped for an additive "
                "change (not a failure, just a reminder)."
            )

    if breaking:
        print("Breaking ABI changes:")
        for line in breaking:
            print(f"  ! {line}")
        if not major_bumped:
            print(
                "\nFAIL: breaking ABI change without a CV_ABI_VERSION_MAJOR bump. "
                "Every language binding built on this header (Rust, Go, Node) "
                "assumes MAJOR is stable across builds. Either bump "
                "CV_ABI_VERSION_MAJOR in bindings/rust/chronovec-sys/native/include/chronovec.h, or this "
                "wasn't meant to be a breaking change."
            )
            return 1
        print("\nMAJOR was bumped -- breaking change acknowledged.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update", action="store_true", help="record the current header as baseline"
    )
    args = parser.parse_args()

    current = parse_header(HEADER.read_text())

    if args.update:
        BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"Baseline updated: {BASELINE}")
        return 0

    if not BASELINE.exists():
        print(f"No baseline at {BASELINE} -- run with --update to create one.")
        return 1

    baseline = json.loads(BASELINE.read_text())
    return judge(current, baseline)


if __name__ == "__main__":
    sys.exit(main())
