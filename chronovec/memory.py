"""Branchable agent memory with private delta branches by default.

An agent that explores a line of reasoning writes memories it may need to
abandon. Without an isolated branch, the options are to pollute memory or to
rebuild the index. ChronoVec has the primitives to make it a fork instead, and
they are the two things this engine has to make that practical:

* **snapshots** fix the point a branch diverged, so a branch can always read the
  state as it was at the fork;
* **private delta indexes** keep concurrent branches from contaminating each
  other, because branch writes never enter the serving index before merge.

Snapshots give read isolation; private delta indexes keep branch writes out of
the serving timeline. The legacy label implementation remains available via
``branch_engine="labels"`` for compatibility with existing deployments.

    memory = AgentMemory(384)
    memory.add("pref-theme", embedding, text="user prefers dark mode")

    plan = memory.branch("plan-a")          # fork
    plan.add(2, other, text="speculative")
    plan.search(query)                      # sees base + its own writes
    memory.search(query)                    # main never saw the speculation
    plan.discard()                          # abandon speculation; purge later

The legacy label implementation has a 63-branch limit (one label bit each, bit
0 reserved for main); private deltas have no corresponding label-bit limit.
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
from typing import TYPE_CHECKING, Any

from .collection import stable_id
from .native import NativeChronoVecIndex, SearchResult

if TYPE_CHECKING:
    from .branchable_memory import BranchableMemory

MAIN_BIT = 1 << 0
MAX_BRANCHES = 63


class _RWLock:
    """Serializes writers against everyone; lets readers overlap with readers.

    AgentMemory.merge/discard/purge span several native and Python-side
    mutations that a concurrent reader must never see half-applied, so
    writers still need full exclusivity. But two concurrent
    AgentMemory.search() calls don't conflict with each other or with the
    native index's own lock-free reader path -- serializing those too (a
    plain Lock/RLock) made concurrent search throughput *worse* than
    single-threaded, since every reader queued behind the others for no
    reason. Writers are reentrant per-thread so a mutator can call back into
    another write-locked method (merge -> _delete, _checkpoint -> save)
    without deadlocking on its own lock.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer: int | None = None
        self._writer_depth = 0

    @contextmanager
    def read(self) -> Iterator[None]:
        me = threading.get_ident()
        with self._cond:
            while self._writer is not None and self._writer != me:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write(self) -> Iterator[None]:
        me = threading.get_ident()
        with self._cond:
            if self._writer == me:
                self._writer_depth += 1
            else:
                while self._writer is not None or self._readers > 0:
                    self._cond.wait()
                self._writer = me
                self._writer_depth = 1
        try:
            yield
        finally:
            with self._cond:
                self._writer_depth -= 1
                if self._writer_depth == 0:
                    self._writer = None
                    self._cond.notify_all()


class BranchError(RuntimeError):
    """Raised for branch lifecycle misuse, rather than failing silently."""


@dataclass
class Record:
    """What was stored alongside a vector.

    Payloads outlive deletion. The index keeps a deleted version visible to any
    snapshot taken before the delete, so dropping its metadata immediately would
    make historical reads return a vector with no text attached. The entry is
    retained until :meth:`AgentMemory.purge` decides no snapshot can reach it.
    """

    id: str | int
    payload: dict[str, Any] = field(default_factory=dict)
    branch: str = "main"
    deleted_at: int | None = None

    @property
    def live(self) -> bool:
        return self.deleted_at is None


class _View:
    """A readable, writable view of memory restricted to one branch."""

    def __init__(self, store: AgentMemory, name: str, bit: int, base: int) -> None:
        self._store = store
        self.name = name
        self.bit = bit
        self.base_snapshot = base
        self._deleted: set[int] = set()
        self._overlay_ids: dict[int, int] = {}
        self._closed = False

    # -- writing ---------------------------------------------------------
    def add(self, item_id: str | int, vector, **payload: Any) -> int:
        """Store a vector on this branch. Returns the commit snapshot."""
        self._check_open()
        return self._store._add(item_id, vector, self.bit, self.name, payload, view=self)

    def delete(self, item_id: str | int) -> int:
        self._check_open()
        return self._store._delete(item_id, view=self)

    # -- reading ---------------------------------------------------------
    def search(
        self, vector, k: int = 10, *, nprobe: int | None = None, as_of: int | None = None
    ) -> list[tuple[SearchResult, Record]]:
        """Nearest neighbours visible to this branch.

        Visible means: written on main, or written on this branch. Records
        belonging to sibling branches are excluded by label, so speculation in
        one branch is invisible to another.
        """
        self._check_open()
        return self._store._search(vector, k, self, nprobe, as_of)

    def as_of_fork(
        self, vector, k: int = 10, *, nprobe: int | None = None
    ) -> list[tuple[SearchResult, Record]]:
        """Search the state as it was when this branch was created."""
        return self.search(vector, k, nprobe=nprobe, as_of=self.base_snapshot)

    # -- lifecycle -------------------------------------------------------
    def discard(self) -> int:
        """Abandon every write made on this branch.

        The writes become unreachable immediately but remain reclaimable MVCC
        versions until :meth:`AgentMemory.purge` is called with a safe horizon.
        """
        self._check_open()
        try:
            reclaimed = self._store._discard(self)
        except BaseException:
            # _discard releases the view only after native cleanup. If the
            # follow-up checkpoint fails, the object must not outlive the
            # store's released branch label.
            if self.name not in self._store._branches:
                self._closed = True
            raise
        self._closed = True
        return reclaimed

    def merge(self) -> int:
        """Fold this branch's writes into main. Returns records promoted.

        Implemented as delete-and-reinsert under the main label, because a
        record's label is fixed at insert. The rewrite is bounded by the size of
        the branch, not of the index.
        """
        self._check_open()
        try:
            promoted = self._store._merge(self)
        except BaseException:
            if self.name not in self._store._branches:
                self._closed = True
            raise
        self._closed = True
        return promoted

    def _check_open(self) -> None:
        if self._closed:
            raise BranchError(f"branch {self.name!r} has already been discarded or merged")

    def __repr__(self) -> str:
        return f"<branch {self.name!r} forked at {self.base_snapshot}>"


class AgentMemory:
    """Vector memory that can fork, explore, and recover historical state."""

    _delta_engine: BranchableMemory | None

    def __init__(
        self,
        dimensions: int,
        *,
        metric: str = "cosine",
        page_capacity: int = 256,
        branch_page_capacity: int | None = None,
        nprobe: int = 16,
        checkpoint_path: str | os.PathLike[str] | None = None,
        branch_engine: str = "delta",
        branch_merge_strategy: str | None = None,
    ) -> None:
        if branch_engine == "delta":
            # Imported lazily: BranchableMemory depends on Record and
            # BranchError from this module, while this façade keeps the stable
            # AgentMemory import path unchanged.
            from .branchable_memory import BranchableMemory

            self._delta_engine = BranchableMemory(
                dimensions,
                metric=metric,
                page_capacity=page_capacity,
                branch_page_capacity=branch_page_capacity,
                nprobe=nprobe,
                checkpoint_path=checkpoint_path,
                merge_strategy=branch_merge_strategy or "last_writer_wins",
            )
            self.dimensions = dimensions
            return
        if branch_engine != "labels":
            raise ValueError("branch_engine must be 'labels' or 'delta'")
        if branch_merge_strategy is not None:
            raise ValueError("branch_merge_strategy requires branch_engine='delta'")
        if branch_page_capacity is not None:
            raise ValueError("branch_page_capacity requires branch_engine='delta'")
        self._delta_engine = None
        self._index = NativeChronoVecIndex(
            dimensions,
            metric=metric,
            page_capacity=page_capacity,
            nprobe=nprobe,
            labels=True,
        )
        self.dimensions = dimensions
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._records: dict[int, Record] = {}
        self._history: dict[int, list[tuple[int, Record]]] = {}
        self._vectors: dict[int, Any] = {}
        self._branches: dict[str, _View] = {}
        self._used_bits = MAIN_BIT
        self._state_lock = _RWLock()
        self._main = _View(self, "main", MAIN_BIT, 0)

    # -- main-branch convenience ----------------------------------------
    def add(self, item_id: str | int, vector, **payload: Any) -> int:
        if self._delta_engine is not None:
            return self._delta_engine.add(item_id, vector, **payload)
        return self._main.add(item_id, vector, **payload)

    def delete(self, item_id: str | int) -> int:
        if self._delta_engine is not None:
            return self._delta_engine.delete(item_id)
        return self._main.delete(item_id)

    def search(
        self, vector, k: int = 10, *, nprobe: int | None = None, as_of: int | None = None
    ) -> list[tuple[SearchResult, Record]]:
        if self._delta_engine is not None:
            return self._delta_engine.search(vector, k, nprobe=nprobe, as_of=as_of)
        return self._main.search(vector, k, nprobe=nprobe, as_of=as_of)

    @property
    def clock(self) -> int:
        if self._delta_engine is not None:
            return self._delta_engine.clock
        with self._state_lock.read():
            return self._index.clock

    def snapshot(self) -> int:
        """Return the current committed point in the memory timeline."""
        if self._delta_engine is not None:
            return self._delta_engine.snapshot()
        with self._state_lock.read():
            return self._index.clock

    @property
    def main(self) -> _View:
        if self._delta_engine is not None:
            return self._delta_engine.main  # type: ignore[return-value]
        return self._main

    def branches(self) -> Iterator[str]:
        if self._delta_engine is not None:
            return self._delta_engine.branches()
        with self._state_lock.read():
            return iter(tuple(self._branches))

    def get_branch(self, name: str) -> _View:
        """Return an active branch by name for workflow recovery or inspection.

        A caller that owns the branch lifecycle can reattach after restoring a
        persisted memory checkpoint. The returned view has the same lifecycle
        rules as a view returned by :meth:`branch`.
        """
        if self._delta_engine is not None:
            return self._delta_engine.get_branch(name)  # type: ignore[return-value]
        with self._state_lock.read():
            try:
                return self._branches[name]
            except KeyError as exc:
                raise BranchError(f"branch {name!r} does not exist or is closed") from exc

    # -- branching -------------------------------------------------------
    def branch(self, name: str, *, snapshot: int | None = None) -> _View:
        """Fork memory at the current or an earlier committed snapshot."""
        if self._delta_engine is not None:
            return self._delta_engine.branch(name, snapshot=snapshot)  # type: ignore[return-value]
        with self._state_lock.write():
            if name in self._branches or name == "main":
                raise BranchError(f"branch {name!r} already exists")
            base = self._index.clock if snapshot is None else int(snapshot)
            if base < 0 or base > self._index.clock:
                raise ValueError(f"snapshot must be between 0 and {self._index.clock}, got {base}")
            bit = self._allocate_bit()
            view = _View(self, name, bit, base)
            self._branches[name] = view
            self._checkpoint()
            return view

    def _allocate_bit(self) -> int:
        for position in range(1, MAX_BRANCHES + 1):
            candidate = 1 << position
            if not self._used_bits & candidate:
                self._used_bits |= candidate
                return candidate
        raise BranchError(
            f"all {MAX_BRANCHES} branch labels are in use; discard or merge one first"
        )

    # -- internals -------------------------------------------------------
    def _add(
        self,
        item_id: str | int,
        vector,
        bit: int,
        branch: str,
        payload: dict[str, Any],
        *,
        view: _View | None = None,
    ) -> int:
        with self._state_lock.write():
            return self._add_unlocked(item_id, vector, bit, branch, payload, view=view)

    def _add_unlocked(
        self,
        item_id: str | int,
        vector,
        bit: int,
        branch: str,
        payload: dict[str, Any],
        *,
        view: _View | None = None,
    ) -> int:
        external_id = stable_id(item_id)
        engine_id = external_id if bit == MAIN_BIT else self._overlay_id(external_id, bit)
        committed = self._index.insert(engine_id, vector, label=bit)
        self._records[engine_id] = Record(item_id, dict(payload), branch)
        self._history.setdefault(engine_id, []).append(
            (committed, Record(item_id, dict(payload), branch))
        )
        self._vectors[engine_id] = vector
        if view is not None and view.bit != MAIN_BIT:
            view._overlay_ids[external_id] = engine_id
            view._deleted.discard(external_id)
        self._checkpoint()
        return committed

    @staticmethod
    def _overlay_id(external_id: int, bit: int) -> int:
        """Give a branch version its own native identity.

        The native index treats an insert with an existing id as an update on
        the shared timeline. Branch overlays must instead coexist with the
        main version until the branch is merged.
        """
        digest = hashlib.blake2b(f"{external_id}:{bit}".encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") >> 1

    def _delete(self, item_id: str | int, *, view: _View | None = None) -> int:
        with self._state_lock.write():
            engine_id = stable_id(item_id)
            if view is not None and view.bit != MAIN_BIT:
                view._deleted.add(engine_id)
                self._checkpoint()
                return self._index.clock
            committed = self._index.delete(engine_id)
            record = self._records.get(engine_id)
            if record is not None:
                record.deleted_at = committed
            self._checkpoint()
            return committed

    def _search(
        self, vector, k: int, view: _View, nprobe: int | None, as_of: int | None
    ) -> list[tuple[SearchResult, Record]]:
        with self._state_lock.read():
            return self._search_unlocked(vector, k, view, nprobe, as_of)

    def _search_unlocked(
        self, vector, k: int, view: _View, nprobe: int | None, as_of: int | None
    ) -> list[tuple[SearchResult, Record]]:
        if view.bit == MAIN_BIT:
            hits = self._index.search(
                vector,
                k=k,
                nprobe=nprobe,
                snapshot=as_of,
                exclude=self._used_bits & ~MAIN_BIT,
            )
            return [self._materialize(hit, as_of) for hit in hits]

        # A historical branch is the main view at its fork plus its own overlay.
        # The native filter can express either half efficiently, while combining
        # the two result sets here keeps future main writes out of the branch.
        #
        # Each side is fetched as a shortlist and then filtered by `hidden`
        # below, so a plain k-sized fetch can come up short of k even when
        # enough real candidates exist: every branch-local delete can only
        # remove a candidate this call already fetched, so padding the
        # shortlist by len(view._deleted) exactly covers that worst case
        # without guessing at a multiplier.
        cutoff = view.base_snapshot if as_of is None else min(view.base_snapshot, as_of)
        shortlist = k + len(view._deleted)
        base_hits = (
            []
            if cutoff == 0
            else self._index.search(
                vector,
                k=shortlist,
                nprobe=nprobe,
                snapshot=cutoff,
                exclude=self._used_bits & ~MAIN_BIT,
            )
        )
        overlay_hits = (
            []
            if as_of == 0
            else self._index.search(
                vector,
                k=shortlist,
                nprobe=nprobe,
                snapshot=as_of,
                require_any=view.bit,
            )
        )
        by_id: dict[int, tuple[SearchResult, int | None]] = {}
        for hit in base_hits:
            record = self._record_at(hit.id, cutoff)
            if record is not None:
                by_id[stable_id(record.id)] = (hit, cutoff)
        for hit in overlay_hits:
            record = self._record_at(hit.id, as_of)
            if record is not None:
                by_id[stable_id(record.id)] = (hit, as_of)
        hidden = view._deleted
        visible: list[tuple[SearchResult, int | None]] = [
            value for external_id, value in by_id.items() if external_id not in hidden
        ]
        sorted_hits: list[tuple[SearchResult, int | None]] = sorted(
            visible,
            key=lambda value: value[0].distance,
        )
        return [self._materialize(hit, snapshot) for hit, snapshot in sorted_hits[:k]]

    def _record_at(self, engine_id: int, snapshot: int | None) -> Record | None:
        if snapshot is None:
            return self._records.get(engine_id)
        versions = self._history.get(engine_id, [])
        for stamp, record in reversed(versions):
            if stamp <= snapshot:
                return record
        return None

    def _materialize(self, hit: SearchResult, snapshot: int | None) -> tuple[SearchResult, Record]:
        record = self._record_at(hit.id, snapshot) or Record(hit.id)
        return SearchResult(stable_id(record.id), hit.distance), record

    def _release(self, view: _View) -> None:
        self._used_bits &= ~view.bit
        self._branches.pop(view.name, None)

    def _ids_on(self, view: _View) -> list[int]:
        return [i for i, r in self._records.items() if r.branch == view.name and r.live]

    def _discard(self, view: _View) -> int:
        with self._state_lock.write():
            for item_id in self._ids_on(view):
                self._delete_engine(item_id)
            self._release(view)
            self._checkpoint()
            return 0

    def _merge(self, view: _View) -> int:
        with self._state_lock.write():
            branch_ids = self._ids_on(view)
            promoted = [
                item_id
                for item_id in branch_ids
                if stable_id(self._records[item_id].id) not in view._deleted
            ]
            for item_id in promoted:
                vector = self._vectors[item_id]
                record = self._records[item_id]
                main_id = stable_id(record.id)
                self._delete_engine(item_id)
                self._index.insert(main_id, vector, label=MAIN_BIT)
                committed = self._index.clock
                self._records[main_id] = Record(record.id, dict(record.payload), "main")
                self._history.setdefault(main_id, []).append(
                    (committed, Record(record.id, dict(record.payload), "main"))
                )
                self._vectors[main_id] = vector
            for item_id in branch_ids:
                if item_id not in promoted:
                    self._delete_engine(item_id)
            for external_id in view._deleted:
                if external_id in self._records and self._records[external_id].live:
                    self._delete(external_id)
            self._release(view)
            self._checkpoint()
            return len(promoted)

    def _delete_engine(self, engine_id: int) -> None:
        self._index.delete(engine_id)
        record = self._records.get(engine_id)
        if record is not None:
            record.deleted_at = self._index.clock

    def purge(self, oldest_snapshot: int | None = None) -> int:
        """Reclaim versions older than a caller-supplied safe snapshot horizon.

        The same horizon is applied to native vectors and Python payload
        history. Callers must keep the oldest snapshot any reader may still
        use; active branch fork points are protected automatically. The
        default reclaims everything that is no longer current and no active
        branch can still read. Returns the number of native versions removed.
        """
        if self._delta_engine is not None:
            return self._delta_engine.purge(oldest_snapshot)
        with self._state_lock.write():
            horizon = self._index.clock + 1 if oldest_snapshot is None else oldest_snapshot
            protected_horizon = min(
                (view.base_snapshot + 1 for view in self._branches.values()),
                default=0,
            )
            if horizon > protected_horizon and self._branches:
                raise ValueError(
                    "purge horizon would invalidate an active branch fork point; "
                    "discard or merge branches first"
                )
            reclaimed = self._index.vacuum(oldest_snapshot=horizon, budget_versions=0)
            stale = [
                i
                for i, r in self._records.items()
                if r.deleted_at is not None and r.deleted_at < horizon
            ]
            for item_id in stale:
                self._records.pop(item_id, None)
                self._history.pop(item_id, None)
                self._vectors.pop(item_id, None)
            self._checkpoint()
            return reclaimed

    def _checkpoint(self) -> None:
        """Persist the complete memory only when a checkpoint path is configured."""
        if self._checkpoint_path is not None:
            self.save(self._checkpoint_path)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Atomically publish a native index and agent-memory checkpoint.

        A generation contains the native checkpoint and metadata sidecar. The
        manifest is published last, so a crash leaves the previous complete
        generation selected; a checksum prevents a mixed pair from loading.
        Once the new manifest is durably published, the generation it
        replaces is deleted -- this can run on every mutation when
        ``checkpoint_path`` is set, so leaving each superseded generation on
        disk would grow the checkpoint directory without bound.

        Takes the same write lock as every mutator, so a concurrent add,
        delete, branch, merge, discard, or purge cannot be serialized into
        this call mid-mutation.
        """
        if self._delta_engine is not None:
            self._delta_engine.save(path)
            return
        with self._state_lock.write():
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path = path.with_suffix(".manifest")
            previous_generation = self._read_generation_files(manifest_path)
            generation = uuid.uuid4().hex
            native_path = path.with_name(f"{path.name}.{generation}.cvec")
            metadata_path = path.with_name(f"{path.name}.{generation}.meta")
            self._index.save(native_path)
            meta = {
                "format_version": 2,
                "generation": generation,
                "dimensions": self.dimensions,
                "records": self._records,
                "history": self._history,
                "vectors": self._vectors,
                "branches": [
                    {
                        "name": view.name,
                        "bit": view.bit,
                        "base_snapshot": view.base_snapshot,
                        "deleted": view._deleted,
                        "overlay_ids": view._overlay_ids,
                    }
                    for view in self._branches.values()
                ],
            }
            self._write_pickle(metadata_path, meta)
            manifest = {
                "format_version": 1,
                "generation": generation,
                "native": native_path.name,
                "metadata": metadata_path.name,
                "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
            }
            self._write_json(manifest_path, manifest)
            self._remove_generation_files(path.parent, previous_generation)

    @staticmethod
    def _read_generation_files(manifest_path: Path) -> tuple[str, str] | None:
        """The (native, metadata) filenames the current manifest points at, if any."""
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        native = manifest.get("native")
        metadata = manifest.get("metadata")
        if not native or not metadata:
            return None
        return native, metadata

    @staticmethod
    def _remove_generation_files(directory: Path, names: tuple[str, str] | None) -> None:
        """Best-effort cleanup of a superseded checkpoint generation.

        Called only after the new manifest is durably published, so these
        files are provably no longer the selected generation. A failed
        unlink leaves an orphan rather than raising -- the save it follows
        already succeeded, and a stray old file is safe, just wasted space.
        """
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
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        checkpoint_path: str | os.PathLike[str] | None = None,
    ) -> AgentMemory:
        """Restore an `AgentMemory` checkpoint, including active branches."""
        path = Path(path)
        delta_manifest = path.with_suffix(".branchable.manifest")
        if delta_manifest.exists():
            from .branchable_memory import BranchableMemory

            memory: AgentMemory = cls.__new__(cls)
            memory._delta_engine = BranchableMemory.load(path, checkpoint_path=checkpoint_path)
            memory.dimensions = memory._delta_engine.dimensions
            return memory
        manifest_path = path.with_suffix(".manifest")
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format_version") != 1:
                raise ValueError(
                    f"unsupported AgentMemory manifest format: {manifest.get('format_version')}"
                )
            native_path = path.parent / manifest["native"]
            metadata_path = path.parent / manifest["metadata"]
            if native_path.parent != path.parent or metadata_path.parent != path.parent:
                raise ValueError("AgentMemory manifest points outside its checkpoint directory")
            if hashlib.sha256(native_path.read_bytes()).hexdigest() != manifest["native_sha256"]:
                raise ValueError("AgentMemory native checkpoint does not match its manifest")
        else:
            # Read checkpoints created before manifest publication was added.
            native_path = path.with_suffix(".cvec")
            metadata_path = path.with_suffix(".meta")
        native = NativeChronoVecIndex.load(native_path)
        meta = pickle.loads(metadata_path.read_bytes())
        if manifest_path.exists() and meta.get("generation") != manifest["generation"]:
            raise ValueError("AgentMemory metadata does not match its manifest")
        if meta.get("format_version") not in {1, 2}:
            raise ValueError(
                f"unsupported AgentMemory checkpoint format: {meta.get('format_version')}"
            )
        memory = cls.__new__(cls)
        memory._delta_engine = None
        memory._index = native
        memory.dimensions = int(meta["dimensions"])
        memory._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        memory._records = meta["records"]
        memory._history = meta.get("history", {})
        memory._vectors = meta["vectors"]
        memory._branches = {}
        memory._used_bits = MAIN_BIT
        memory._state_lock = _RWLock()
        memory._main = _View(memory, "main", MAIN_BIT, 0)
        for item in meta["branches"]:
            view = _View(memory, item["name"], int(item["bit"]), int(item["base_snapshot"]))
            view._deleted = set(item["deleted"])
            view._overlay_ids = dict(item["overlay_ids"])
            memory._branches[view.name] = view
            memory._used_bits |= view.bit
        return memory

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        if self._delta_engine is not None:
            self._delta_engine.close()
            return
        self._index.close()

    def __enter__(self) -> AgentMemory:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
