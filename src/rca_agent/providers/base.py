"""Abstract base protocols for RCA Agent providers.

Any log provider — local files, remote APIs, cloud sinks — must satisfy
the LogProvider interface.  Any Git provider must satisfy GitProvider.
Any tracing backend must satisfy TraceProvider.
All are structural protocols so implementations do not need to inherit
from them explicitly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from rca_agent.models.git_models import Commit, CommitDiff, GitCommitQuery
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.trace_models import (
    Span,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
)


@runtime_checkable
class LogProvider(Protocol):
    """Structural protocol for all log provider implementations."""

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        """Return log entries matching *query*, sorted ascending by timestamp."""
        ...

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        """Return all log entries sharing a trace ID, sorted by timestamp."""
        ...

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        """Return the single log entry with the given stable ID, or None."""
        ...


@runtime_checkable
class GitProvider(Protocol):
    """Structural protocol for all Git provider implementations.

    All operations are **read-only**.  Implementations must never expose
    write operations or arbitrary shell access.
    """

    def get_recent_commits(self, limit: int = 20) -> list[Commit]:
        """Return the *limit* most recent commits, newest first."""
        ...

    def get_commit(self, commit_id: str) -> Commit:
        """Return a single commit by its full or abbreviated SHA-1.

        Raises
        ------
        ValueError
            If *commit_id* is not found or is invalid.
        """
        ...

    def get_diff(self, commit_id: str) -> list[CommitDiff]:
        """Return per-file diffs for the given commit.

        Raises
        ------
        ValueError
            If *commit_id* is not found or is invalid.
        """
        ...

    def get_files_changed(self, commit_id: str) -> list[str]:
        """Return repository-relative paths of files changed in *commit_id*.

        Raises
        ------
        ValueError
            If *commit_id* is not found or is invalid.
        """
        ...

    def search_commits(self, query: GitCommitQuery) -> list[Commit]:
        """Return commits matching the search criteria in *query*."""
        ...

    def get_commits_between(
        self, start_time: "datetime", end_time: "datetime"
    ) -> list[Commit]:
        """Return commits whose author date falls within [start_time, end_time]."""
        ...


@runtime_checkable
class TraceProvider(Protocol):
    """Structural protocol for all distributed trace provider implementations.

    Implementations query a tracing backend (Jaeger, Grafana Tempo, Zipkin, …)
    and return generic ``Trace`` / ``Span`` objects.  All operations are
    **read-only**.
    """

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return the complete trace for *trace_id*, or ``None`` if not found.

        Parameters
        ----------
        trace_id:
            A 16-byte (32 hex character) trace identifier.

        Returns
        -------
        Trace | None
            The full trace including all spans, or ``None`` when the trace
            does not exist in the backend.
        """
        ...

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        """Return traces matching the criteria in *query*.

        Results are ordered newest-first (by root span start time).
        Implementations must respect ``query.limit``.
        """
        ...

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Return all spans for *trace_id*, or an empty list if not found.

        This is a convenience wrapper around ``get_trace()`` for callers
        that only need the span list.
        """
        ...

    def get_failed_spans(self, trace_id: str) -> list[Span]:
        """Return only the ERROR-status spans for *trace_id*.

        Returns an empty list when the trace does not exist or has no errors.
        This is the primary entry point for failure investigation workflows.
        """
        ...
