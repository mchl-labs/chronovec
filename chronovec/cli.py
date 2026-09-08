"""The ``chronovec`` command-line interface."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .client import Client


def _describe(client: Client, name: str) -> dict[str, Any]:
    collection = client.get_collection(name)
    try:
        return {
            "name": collection.name,
            "dimensions": collection.dimensions,
            "count": collection.count(),
            "stats": collection.stats(),
        }
    finally:
        collection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chronovec", description="Inspect local ChronoVec collections"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="list collections in a client directory")
    list_parser.add_argument("path", nargs="?", default=".chronovec")
    inspect_parser = subparsers.add_parser(
        "inspect", help="show collection metadata and engine stats"
    )
    inspect_parser.add_argument("path", nargs="?", default=".chronovec")
    inspect_parser.add_argument("name", nargs="?")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    client = Client(args.path)
    if args.command == "list":
        for name in client.list_collections():
            print(name)
        return 0
    names = [args.name] if args.name else client.list_collections()
    descriptions = [_describe(client, name) for name in names]
    if args.as_json:
        print(json.dumps(descriptions[0] if args.name else descriptions, indent=2, sort_keys=True))
    else:
        for description in descriptions:
            print(
                f"{description['name']}: {description['count']} vectors, "
                f"{description['dimensions']} dimensions"
            )
            print(json.dumps(description["stats"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
