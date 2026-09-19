"""Abstract base protocol for log providers.

Any log provider — local files, remote APIs, cloud sinks — must satisfy
this interface so that the RCA Agent remains source-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult


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
