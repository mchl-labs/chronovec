"""Private branch-local ANN deltas for transactional speculative retrieval.

BranchableMemory is deliberately separate from chronovec.memory. AgentMemory
uses it as its default backend while retaining a label-based compatibility
engine. This layer gives every branch a private native delta index and reads it against an
immutable snapshot of the main index. Branch writes never enter the main index
until a successful merge.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .collection import stable_id
from .memory import BranchError, Record
from .native import NativeChronoVecIndex, SearchResult


class BranchConflictError(BranchError):
    """A branch changed a record that main modified after its fork."""


class _RWLock:
    """Re-entrant reader/writer gate for high-level atomic publication."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int | None = None
        self._depth = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        owner = threading.get_ident()
        with self._condition:
            while self._writer is not None and self._writer != owner:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        owner = threading.get_ident()
        with self._condition:
            if self._writer == owner:
                self._depth += 1
            else:
                while self._writer is not None or self._readers:
                    self._condition.wait()
                self._writer = owner
                self._depth = 1
        try:
            yield
        finally:
            with self._condition:
                self._depth -= 1
                if self._depth == 0:
                    self._writer = None
                    self._condition.notify_all()


def _copy_record(record: Record, *, branch: str | None = None) -> Record:
    return Record(
        record.id,
        dict(record.payload),
        record.branch if branch is None else branch,
        record.deleted_at,
    )


@dataclass
class _BranchState:
    """All mutable branch state is private to this branch."""

    owner: BranchableMemory
    name: str
    base_snapshot: int
    delta: NativeChronoVecIndex
    records: dict[int, Record] = field(default_factory=dict)
    vectors: dict[int, np.ndarray] = field(default_factory=dict)
    deleted: set[int] = field(default_factory=set)
    expected_versions: dict[int, int] = field(default_factory=dict)
    closed: bool = False

    def touch(self, item_id: int) -> None:
        self.expected_versions.setdefault(
            item_id, self.owner._version_at(item_id, self.base_snapshot)
        )


class Branch:
    """An isolated, immediately searchable view over a main-index snapshot."""

    def __init__(self, state: _BranchState) -> None:
        self._state = state

    @property
    def name(self) -> str:
        return self._state.name

    @property
    def base_snapshot(self) -> int:
        return self._state.base_snapshot

    def add(self, item_id: str | int, vector, **payload: Any) -> int:
        self._check_open()
        return self._state.owner._branch_add(self._state, item_id, vector, payload)

    def delete(self, item_id: str | int) -> int:
        self._check_open()
        return self._state.owner._branch_delete(self._state, item_id)

    def search(
        self, vector, k: int = 10, *, nprobe: int | None = None, as_of: int | None = None
    ) -> list[tuple[SearchResult, Record]]:
        self._check_open()
        return self._state.owner._branch_search(self._state, vector, k, nprobe, as_of)

    def as_of_fork(
        self, vector, k: int = 10, *, nprobe: int | None = None
    ) -> list[tuple[SearchResult, Record]]:
        return self.search(vector, k, nprobe=nprobe, as_of=self.base_snapshot)

    def diff(self) -> dict[str, tuple[str | int, ...]]:
        self._check_open()
        return self._state.owner._diff(self._state)

    def conflicts(self) -> tuple[str | int, ...]:
        self._check_open()
        return self._state.owner._conflicts(self._state)

    def merge(self, *, strategy: str | None = None) -> int:
        self._check_open()
        try:
            count = self._state.owner._merge_branch(
                self._state, strategy or self._state.owner.merge_strategy
            )
        except BaseException:
            if self._state.name not in self._state.owner._branches:
                self._state.closed = True
            raise
        self._state.closed = True
        return count

    def discard(self) -> int:
        self._check_open()
        try:
            self._state.owner._discard_branch(self._state)
        except BaseException:
            if self._state.name not in self._state.owner._branches:
                self._state.closed = True
            raise
        self._state.closed = True
        return 0

    def _check_open(self) -> None:
        if self._state.closed:
            raise BranchError(f"branch {self._state.name!r} has already been discarded or merged")

    def __repr__(self) -> str:
        return f"<Branch {self.name!r} forked at {self.base_snapshot}>"


class BranchableMemory:
    """Experimental branch-local ANN memory.

    The main index is an ordinary unlabeled NativeChronoVecIndex. A branch owns
    an independent delta index and sees main at fork plus delta minus deletes.
    This keeps branch writes out of main pages, routing, and reclamation.
    """

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "cosine",
        page_capacity: int = 256,
        branch_page_capacity: int | None = None,
        nprobe: int = 16,
        checkpoint_path: str | os.PathLike[str] | None = None,
        merge_strategy: str = "fail_on_conflict",
    ) -> None:
        if merge_strategy not in {"fail_on_conflict", "last_writer_wins"}:
            raise ValueError("merge_strategy must be 'fail_on_conflict' or 'last_writer_wins'")
        self.dimensions = int(dimensions)
        self.metric = metric
        self.page_capacity = int(page_capacity)
        self.branch_page_capacity = (
            None if branch_page_capacity is None else int(branch_page_capacity)
        )
        self.nprobe = int(nprobe)
        self.merge_strategy = merge_strategy
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._index = self._new_index()
        self._records: dict[int, Record] = {}
        self._history: dict[int, list[tuple[int, Record]]] = {}
        self._vectors: dict[int, np.ndarray] = {}
        self._versions: dict[int, list[int]] = {}
        self._branches: dict[str, _BranchState] = {}
        self._state_lock = _RWLock()

    def _new_index(self, page_capacity: int | None = None) -> NativeChronoVecIndex:
        return NativeChronoVecIndex(
            self.dimensions,
            metric=self.metric,
            page_capacity=self.page_capacity if page_capacity is None else page_capacity,
            nprobe=self.nprobe,
            labels=False,
        )

    @property
    def clock(self) -> int:
        with self._state_lock.read():
            return self._index.clock

    def snapshot(self) -> int:
        return self.clock

    def branches(self) -> Iterator[str]:
        with self._state_lock.read():
            return iter(tuple(self._branches))

    def get_branch(self, name: str) -> Branch:
        """Reopen an active branch, including one restored from a checkpoint."""
        with self._state_lock.read():
            try:
                return Branch(self._branches[name])
            except KeyError as error:
                raise BranchError(f"branch {name!r} does not exist") from error

    def add(self, item_id: str | int, vector, **payload: Any) -> int:
        with self._state_lock.write():
            committed = self._main_add(item_id, vector, payload)
            self._checkpoint()
            return committed

    def delete(self, item_id: str | int) -> int:
        with self._state_lock.write():
            committed = self._main_delete(stable_id(item_id))
            self._checkpoint()
            return committed

    def search(
        self, vector, k: int = 10, *, nprobe: int | None = None, as_of: int | None = None
    ) -> list[tuple[SearchResult, Record]]:
        with self._state_lock.read():
            return self._main_search(vector, k, nprobe, as_of)

    def branch(self, name: str, *, snapshot: int | None = None) -> Branch:
        with self._state_lock.write():
            if name == "main" or name in self._branches:
                raise BranchError(f"branch {name!r} already exists")
            base = self._index.clock if snapshot is None else int(snapshot)
            if base < 0 or base > self._index.clock:
                raise ValueError(f"snapshot must be between 0 and {self._index.clock}, got {base}")
            state = _BranchState(self, name, base, self._new_index(self.branch_page_capacity))
            self._branches[name] = state
            self._checkpoint()
            return Branch(state)

    @property
    def main(self) -> BranchableMemory:
        """The main view, provided for AgentMemory API compatibility."""
        return self

    def _main_add(self, item_id: str | int, vector, payload: dict[str, Any]) -> int:
        engine_id = stable_id(item_id)
        value = self._vector(vector)
        committed = self._index.insert(engine_id, value)
        record = Record(item_id, dict(payload), "main")
        self._records[engine_id] = record
        self._history.setdefault(engine_id, []).append((committed, _copy_record(record)))
        self._vectors[engine_id] = value
        self._versions.setdefault(engine_id, []).append(committed)
        return committed

    def _main_delete(self, engine_id: int) -> int:
        committed = self._index.delete(engine_id)
        if engine_id in self._records:
            self._records[engine_id].deleted_at = committed
        self._versions.setdefault(engine_id, []).append(committed)
        return committed

    def _main_search(
        self, vector, k: int, nprobe: int | None, snapshot: int | None
    ) -> list[tuple[SearchResult, Record]]:
        hits = self._index.search(vector, k=k, nprobe=nprobe, snapshot=snapshot)
        result = []
        for hit in hits:
            record = self._record_at(hit.id, snapshot)
            if record is not None:
                result.append((SearchResult(hit.id, hit.distance), record))
        return result

    def _branch_add(
        self, state: _BranchState, item_id: str | int, vector, payload: dict[str, Any]
    ) -> int:
        with self._state_lock.write():
            engine_id = stable_id(item_id)
            state.touch(engine_id)
            value = self._vector(vector)
            stamp = state.delta.insert(engine_id, value)
            state.records[engine_id] = Record(item_id, dict(payload), state.name)
            state.vectors[engine_id] = value
            state.deleted.discard(engine_id)
            self._checkpoint()
            return stamp

    def _branch_delete(self, state: _BranchState, item_id: str | int) -> int:
        with self._state_lock.write():
            engine_id = stable_id(item_id)
            state.touch(engine_id)
            if engine_id in state.records:
                state.delta.delete(engine_id)
                state.records.pop(engine_id, None)
                state.vectors.pop(engine_id, None)
            state.deleted.add(engine_id)
            # Branch-local history is never externally addressable, so deleted
            # delta versions can be reclaimed immediately.
            state.delta.vacuum(oldest_snapshot=state.delta.clock + 1, budget_versions=0)
            self._checkpoint()
            return state.delta.clock

    def _branch_search(
        self,
        state: _BranchState,
        vector,
        k: int,
        nprobe: int | None,
        as_of: int | None,
    ) -> list[tuple[SearchResult, Record]]:
        with self._state_lock.read():
            if as_of is not None and as_of != state.base_snapshot:
                raise ValueError("branch snapshots are immutable; use as_of_fork() or omit as_of")
            if as_of == state.base_snapshot:
                return self._main_search(vector, k, nprobe, state.base_snapshot)

            # Only a delete permanently removes a candidate the base search
            # could otherwise supply, so only deletes need to widen the base
            # fetch. An update doesn't shrink the pool: it still contributes
            # exactly one entry, just a newer one, which the delta_hits loop
            # below overwrites the stale base entry with by id -- padding by
            # the branch's full write count instead of just its deletes would
            # make base search cost scale with total branch history rather
            # than with what was actually deleted.
            base_hits = (
                []
                if state.base_snapshot == 0
                else self._index.search(
                    vector,
                    k=k + len(state.deleted),
                    nprobe=nprobe,
                    snapshot=state.base_snapshot,
                )
            )
            delta_hits = state.delta.search(vector, k=k, nprobe=nprobe)
            merged: dict[int, tuple[SearchResult, Record]] = {}
            for hit in base_hits:
                if hit.id in state.deleted:
                    continue
                record = self._record_at(hit.id, state.base_snapshot)
                if record is not None:
                    merged[hit.id] = (SearchResult(hit.id, hit.distance), record)
            for hit in delta_hits:
                record = state.records.get(hit.id)
                if record is not None:
                    merged[hit.id] = (SearchResult(hit.id, hit.distance), _copy_record(record))
            return sorted(merged.values(), key=lambda pair: pair[0].distance)[:k]

    def _diff(self, state: _BranchState) -> dict[str, tuple[str | int, ...]]:
        return {
            "upserts": tuple(state.records[item_id].id for item_id in sorted(state.records)),
            "deletes": tuple(
                self._logical_id(item_id, state.base_snapshot) for item_id in sorted(state.deleted)
            ),
        }

    def _conflicts(self, state: _BranchState) -> tuple[str | int, ...]:
        with self._state_lock.read():
            return self._conflicts_unlocked(state)

    def _conflicts_unlocked(self, state: _BranchState) -> tuple[str | int, ...]:
        return tuple(
            self._logical_id(item_id, state.base_snapshot)
            for item_id, expected in state.expected_versions.items()
            if self._versions.get(item_id, [0])[-1] != expected
        )

    def _merge_branch(self, state: _BranchState, strategy: str) -> int:
        with self._state_lock.write():
            conflicts = self._conflicts_unlocked(state)
            if strategy == "fail_on_conflict" and conflicts:
                raise BranchConflictError(
                    f"branch {state.name!r} conflicts with main changes to: "
                    + ", ".join(map(str, conflicts))
                )
            if strategy not in {"fail_on_conflict", "last_writer_wins"}:
                raise ValueError("unknown branch merge strategy")

            promoted = sorted(state.records)
            deleted = sorted(
                item_id
                for item_id in state.deleted
                if (current := self._records.get(item_id)) is not None and current.live
            )
            vectors = (
                np.stack([state.vectors[item_id] for item_id in promoted])
                if promoted
                else np.empty((0, self.dimensions), dtype=np.float32)
            )
            # One native commit timestamp publishes every affected page. The
            # Python gate includes the accompanying record metadata in that
            # same visible operation for BranchableMemory callers.
            committed = self._index.apply_changes(deleted, promoted, vectors)
            for item_id in deleted:
                self._records[item_id].deleted_at = committed
                self._versions.setdefault(item_id, []).append(committed)
            for item_id in promoted:
                record = _copy_record(state.records[item_id], branch="main")
                record.deleted_at = None
                self._records[item_id] = record
                self._history.setdefault(item_id, []).append((committed, _copy_record(record)))
                self._vectors[item_id] = state.vectors[item_id]
                self._versions.setdefault(item_id, []).append(committed)

            self._release(state)
            self._checkpoint()
            return len(promoted)

    def _discard_branch(self, state: _BranchState) -> None:
        with self._state_lock.write():
            self._release(state)
            self._checkpoint()

    def _release(self, state: _BranchState) -> None:
        self._branches.pop(state.name, None)
        state.delta.close()

    def _vector(self, vector) -> np.ndarray:
        value = np.ascontiguousarray(vector, dtype=np.float32)
        if value.shape != (self.dimensions,):
            raise ValueError(f"expected ({self.dimensions},), got {value.shape}")
        return value

    def _record_at(self, engine_id: int, snapshot: int | None) -> Record | None:
        if snapshot is None:
            record = self._records.get(engine_id)
            return _copy_record(record) if record is not None else None
        for stamp, record in reversed(self._history.get(engine_id, [])):
            if stamp <= snapshot:
                return _copy_record(record)
        return None

    def _version_at(self, engine_id: int, snapshot: int) -> int:
        for version in reversed(self._versions.get(engine_id, [])):
            if version <= snapshot:
                return version
        return 0

    def _logical_id(self, engine_id: int, snapshot: int) -> str | int:
        record = self._record_at(engine_id, snapshot)
        return record.id if record is not None else engine_id

    def purge(self, oldest_snapshot: int | None = None) -> int:
        """Reclaim main-index history without invalidating active branch forks."""
        with self._state_lock.write():
            horizon = self._index.clock + 1 if oldest_snapshot is None else int(oldest_snapshot)
            protected = min(
                (state.base_snapshot + 1 for state in self._branches.values()), default=0
            )
            if self._branches and horizon > protected:
                raise ValueError(
                    "purge horizon would invalidate an active branch fork point; "
                    "discard or merge branches first"
                )
            reclaimed = self._index.vacuum(oldest_snapshot=horizon, budget_versions=0)
            stale = [
                item_id
                for item_id, record in self._records.items()
                if record.deleted_at is not None and record.deleted_at < horizon
            ]
            for item_id in stale:
                self._records.pop(item_id, None)
                self._history.pop(item_id, None)
                self._vectors.pop(item_id, None)
                self._versions.pop(item_id, None)
            self._checkpoint()
            return reclaimed

    def _checkpoint(self) -> None:
        if self._checkpoint_path is not None:
            self.save(self._checkpoint_path)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically checkpoint main metadata plus every active delta index.

        All files are written under a new generation.  The manifest is the
        commit point, so a crash can restore either the previous complete
        generation or this complete one, never a mixture.
        """
        with self._state_lock.write():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path = path.with_suffix(".branchable.manifest")
            previous = self._read_generation_files(manifest_path)
            generation = uuid.uuid4().hex
            native_path = path.with_name(f"{path.name}.{generation}.main.cvec")
            metadata_path = path.with_name(f"{path.name}.{generation}.meta")
            self._index.save(native_path)

            branch_meta = []
            files = [native_path, metadata_path]
            for position, state in enumerate(self._branches.values()):
                delta_path = path.with_name(f"{path.name}.{generation}.branch-{position}.cvec")
                state.delta.save(delta_path)
                files.append(delta_path)
                branch_meta.append(
                    {
                        "name": state.name,
                        "base_snapshot": state.base_snapshot,
                        "delta": delta_path.name,
                        "records": state.records,
                        "vectors": state.vectors,
                        "deleted": state.deleted,
                        "expected_versions": state.expected_versions,
                    }
                )
            meta = {
                "format_version": 2,
                "generation": generation,
                "dimensions": self.dimensions,
                "metric": self.metric,
                "page_capacity": self.page_capacity,
                "branch_page_capacity": self.branch_page_capacity,
                "nprobe": self.nprobe,
                "merge_strategy": self.merge_strategy,
                "records": self._records,
                "history": self._history,
                "vectors": self._vectors,
                "versions": self._versions,
                "branches": branch_meta,
            }
            self._write_pickle(metadata_path, meta)
            checksums = {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in files}
            self._write_json(
                manifest_path,
                {
                    "format_version": 1,
                    "generation": generation,
                    "native": native_path.name,
                    "metadata": metadata_path.name,
                    "checksums": checksums,
                },
            )
            self._remove_generation_files(path.parent, previous)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> BranchableMemory:
        """Restore a checkpoint, including active private branch deltas."""
        path = Path(path)
        manifest_path = path.with_suffix(".branchable.manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1:
            raise ValueError(
                f"unsupported BranchableMemory manifest format: {manifest.get('format_version')}"
            )
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict):
            raise ValueError("BranchableMemory manifest has no checksums")
        files: dict[str, Path] = {}
        for name, expected in checksums.items():
            candidate = path.parent / name
            if candidate.parent != path.parent:
                raise ValueError(
                    "BranchableMemory manifest points outside its checkpoint directory"
                )
            try:
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(
                    "BranchableMemory checkpoint does not match its manifest"
                ) from error
            if digest != expected:
                raise ValueError("BranchableMemory checkpoint does not match its manifest")
            files[name] = candidate
        native_path = files.get(manifest.get("native"))
        metadata_path = files.get(manifest.get("metadata"))
        if native_path is None or metadata_path is None:
            raise ValueError("BranchableMemory manifest is missing its main checkpoint files")
        meta = pickle.loads(metadata_path.read_bytes())
        if meta.get("format_version") not in {1, 2} or meta.get("generation") != manifest.get(
            "generation"
        ):
            raise ValueError("BranchableMemory metadata does not match its manifest")

        memory: BranchableMemory = cls.__new__(cls)
        memory.dimensions = int(meta["dimensions"])
        memory.metric = meta["metric"]
        memory.page_capacity = int(meta["page_capacity"])
        # Absent on a format_version=1 checkpoint (predates this field):
        # None means "branches use the same page_capacity as main", exactly
        # matching that checkpoint's actual behavior.
        branch_page_capacity = meta.get("branch_page_capacity")
        memory.branch_page_capacity = (
            None if branch_page_capacity is None else int(branch_page_capacity)
        )
        memory.nprobe = int(meta["nprobe"])
        memory.merge_strategy = meta.get("merge_strategy", "fail_on_conflict")
        memory._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        memory._index = NativeChronoVecIndex.load(native_path)
        memory._records = meta["records"]
        memory._history = meta["history"]
        memory._vectors = meta["vectors"]
        memory._versions = meta["versions"]
        memory._branches = {}
        memory._state_lock = _RWLock()
        for item in meta["branches"]:
            delta_path = files.get(item["delta"])
            if delta_path is None:
                memory.close()
                raise ValueError("BranchableMemory branch delta is missing from its manifest")
            state = _BranchState(
                memory,
                item["name"],
                int(item["base_snapshot"]),
                NativeChronoVecIndex.load(delta_path),
                records=item["records"],
                vectors=item["vectors"],
                deleted=set(item["deleted"]),
                expected_versions=dict(item["expected_versions"]),
            )
            memory._branches[state.name] = state
        return memory

    @staticmethod
    def _read_generation_files(manifest_path: Path) -> tuple[str, ...] | None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = tuple(manifest.get("checksums", {}))
        except (OSError, ValueError):
            return None
        return files or None

    @staticmethod
    def _remove_generation_files(directory: Path, names: tuple[str, ...] | None) -> None:
        if names is None:
            return
        for name in names:
            candidate = directory / name
            if candidate.parent != directory:
                continue
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _write_pickle(path: Path, value: Any) -> None:
        BranchableMemory._write_atomic(
            path, pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), "wb"
        )

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        BranchableMemory._write_atomic(
            path, (json.dumps(value, sort_keys=True) + "\n").encode(), "wb"
        )

    @staticmethod
    def _write_atomic(path: Path, value: bytes, mode: str) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, mode) as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def close(self) -> None:
        with self._state_lock.write():
            for state in tuple(self._branches.values()):
                state.delta.close()
            self._branches.clear()
            self._index.close()

    def __enter__(self) -> BranchableMemory:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
