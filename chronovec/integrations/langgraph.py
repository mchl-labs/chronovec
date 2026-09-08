"""LangGraph lifecycle adapter for :class:`chronovec.AgentMemory`.

LangGraph checkpoints workflow state; this adapter checkpoints the semantic
memory decision separately by mapping each graph ``thread_id`` to one
ChronoVec branch.  Keeping that mapping here makes graph retries idempotent
and ensures failed runs can release their branch without reaching into
``AgentMemory`` internals.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..memory import AgentMemory, BranchError, _View


class LangGraphMemory:
    """Map LangGraph execution/thread ids to isolated ChronoVec branches."""

    def __init__(self, memory: AgentMemory) -> None:
        self.memory = memory
        self._views: dict[str, _View] = {}

    def open(self, thread_id: str, *, snapshot: int | None = None) -> _View:
        """Open or return the branch for a graph thread.

        Returning an existing view is intentional: LangGraph may replay a
        node after a checkpoint, and replay must not allocate a second branch.
        """
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("thread_id must be a non-empty string")
        view = self._views.get(thread_id)
        if view is not None:
            if snapshot is not None and snapshot != view.base_snapshot:
                raise BranchError(f"thread {thread_id!r} already has a different snapshot")
            return view
        try:
            view = self.memory.branch(thread_id, snapshot=snapshot)
        except BranchError as branch_error:
            # A durable AgentMemory may already contain the branch after a
            # process restart. Reattach rather than allocating a duplicate.
            try:
                view = self.memory.get_branch(thread_id)
            except BranchError:
                raise branch_error from None
            if snapshot is not None and snapshot != view.base_snapshot:
                raise BranchError(
                    f"thread {thread_id!r} already has a different snapshot"
                ) from None
        self._views[thread_id] = view
        return view

    def get(self, thread_id: str) -> _View:
        """Return an open branch or raise a clear lifecycle error."""
        try:
            return self._views[thread_id]
        except KeyError as exc:
            raise BranchError(f"no open ChronoVec branch for thread {thread_id!r}") from exc

    def merge(self, thread_id: str) -> int:
        """Publish a thread's branch and remove its registry entry."""
        view = self.get(thread_id)
        try:
            return view.merge()
        finally:
            if thread_id not in tuple(self.memory.branches()):
                self._views.pop(thread_id, None)

    def discard(self, thread_id: str) -> int:
        """Discard a thread's branch and remove its registry entry."""
        view = self.get(thread_id)
        try:
            return view.discard()
        finally:
            if thread_id not in tuple(self.memory.branches()):
                self._views.pop(thread_id, None)

    def cleanup(self, thread_id: str) -> None:
        """Best-effort cleanup for an interrupted or failed graph run."""
        if thread_id not in self._views:
            return
        self.discard(thread_id)

    def names(self) -> Iterator[str]:
        """Return currently open graph thread ids, mainly for observability."""
        return iter(tuple(self._views))

    def close(self) -> None:
        """Discard all open branches, preserving the adapter's ownership rule."""
        for thread_id in tuple(self._views):
            self.discard(thread_id)
