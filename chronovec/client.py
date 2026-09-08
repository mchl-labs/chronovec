"""Small persistent client facade for local ChronoVec applications."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .collection import Collection, Embedder

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REGISTRY = "collections.json"


class Client:
    """Manage named, automatically checkpointed local collections.

    ``Client()`` stores collections in ``.chronovec`` in the current directory;
    pass ``path`` to choose another directory. Mutations are checkpointed after
    they commit, so reopening the client requires no separate save call.
    """

    def __init__(self, path: str | Path = ".chronovec") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._registry_path = self.path / _REGISTRY
        self._registry = self._read_registry()
        self._open: dict[str, Collection] = {}

    def _read_registry(self) -> dict[str, dict[str, Any]]:
        if not self._registry_path.exists():
            return {}
        try:
            value = json.loads(self._registry_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read ChronoVec client registry: {self._registry_path}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid ChronoVec client registry: {self._registry_path}")
        return value

    def _write_registry(self) -> None:
        temporary = self._registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._registry, indent=2, sort_keys=True) + "\n")
        temporary.replace(self._registry_path)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("collection name must be 1-128 chars: letters, numbers, _, ., or -")
        return name

    def _base(self, name: str) -> Path:
        return self.path / name

    def get_or_create_collection(
        self,
        name: str = "default",
        *,
        dimensions: int | None = None,
        metric: str = "cosine",
        embedding_function: Embedder | None = None,
        **index_options: Any,
    ) -> Collection:
        """Open a named collection, creating and persisting it if necessary."""
        name = self._validate_name(name)
        if name in self._open:
            collection = self._open[name]
            if dimensions is not None and int(dimensions) != collection.dimensions:
                raise ValueError(
                    f"collection {name!r} already has dimensions={collection.dimensions}"
                )
            return collection
        config = self._registry.get(name)
        base = self._base(name)
        if config is not None:
            if dimensions is not None and int(dimensions) != config["dimensions"]:
                raise ValueError(
                    f"collection {name!r} already has dimensions={config['dimensions']}"
                )
            collection = Collection.load(base, embedding_function=embedding_function)
            collection._checkpoint_path = base
            self._open[name] = collection
            return collection
        if dimensions is None:
            raise ValueError("dimensions is required when creating a collection")
        collection = Collection(
            int(dimensions),
            metric=metric,
            name=name,
            embedding_function=embedding_function,
            checkpoint_path=base,
            **index_options,
        )
        self._registry[name] = {
            "dimensions": int(dimensions),
            "metric": metric,
            "index_options": index_options,
        }
        self._write_registry()
        collection.save(base)
        self._open[name] = collection
        return collection

    def get_collection(
        self, name: str, *, embedding_function: Embedder | None = None
    ) -> Collection:
        """Open an existing collection or raise ``KeyError``."""
        name = self._validate_name(name)
        if name not in self._registry:
            raise KeyError(f"collection {name!r} does not exist")
        return self.get_or_create_collection(name, embedding_function=embedding_function)

    def list_collections(self) -> list[str]:
        """Return collection names in stable order."""
        return sorted(self._registry)

    def delete_collection(self, name: str) -> None:
        """Close and remove one collection and its checkpoint files."""
        name = self._validate_name(name)
        if name not in self._registry:
            raise KeyError(f"collection {name!r} does not exist")
        collection = self._open.pop(name, None)
        if collection is None:
            collection = self.get_collection(name)
        collection.close()
        base = self._base(name)
        for suffix in (".cvec", ".meta"):
            base.with_suffix(suffix).unlink(missing_ok=True)
        self._registry.pop(name)
        self._write_registry()

    def reset(self) -> None:
        """Delete every collection managed by this client."""
        for name in list(self._registry):
            self.delete_collection(name)

    def close(self) -> None:
        """Close all collections opened through this client."""
        for collection in self._open.values():
            collection.close()
        self._open.clear()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


PersistentClient = Client

__all__ = ["Client", "PersistentClient"]
