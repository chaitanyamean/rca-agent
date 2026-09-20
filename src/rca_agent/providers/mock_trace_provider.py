"""MockTraceProvider — deterministic in-memory trace provider for tests.

Usage::

    from rca_agent.providers.mock_trace_provider import MockTraceProvider
    from rca_agent.models.trace_models import Trace, Span, TraceStatus
    from datetime import datetime, timezone

    NOW = datetime.now(timezone.utc)

    provider = MockTraceProvider()
    provider.add_trace(Trace(trace_id="abc123", spans=[
        Span(
            trace_id="abc123", span_id="s1", service_name="rke-backend",
            operation_name="POST /api/pay", start_time=NOW, end_time=NOW,
            duration_ms=5_100.0, status=TraceStatus.ERROR,
            status_message="pool exhausted",
        )
    ]))

    result = provider.get_trace("abc123")
    failed = provider.get_failed_spans("abc123")

MockTraceProvider is used:
* In unit tests that exercise the RCA workflow with trace evidence.
* In integration tests for the RCA Agent that need controlled trace inputs
  without a running Jaeger backend.
* Never in production — production uses JaegerTraceProvider.
"""

from __future__ import annotations

import logging
from datetime import datetime

from rca_agent.models.trace_models import (
    Span,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
    TraceStatus,
)

logger = logging.getLogger(__name__)


class MockTraceProvider:
    """In-memory ``TraceProvider`` for tests.

    Stores traces in a dict keyed by ``trace_id``.  All operations are
    synchronous and deterministic.

    Parameters
    ----------
    traces:
        Optional list of traces to pre-load at construction time.
    """

    def __init__(self, traces: list[Trace] | None = None) -> None:
        self._traces: dict[str, Trace] = {}
        for trace in (traces or []):
            self.add_trace(trace)

    # ------------------------------------------------------------------
    # Write operations (test setup only — not part of TraceProvider protocol)
    # ------------------------------------------------------------------

    def add_trace(self, trace: Trace) -> None:
        """Register a trace so it can be retrieved by ID or search."""
        self._traces[trace.trace_id] = trace

    def clear(self) -> None:
        """Remove all traces from the store."""
        self._traces.clear()

    @property
    def trace_count(self) -> int:
        """Number of traces currently in the store."""
        return len(self._traces)

    # ------------------------------------------------------------------
    # TraceProvider protocol implementation
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return the trace with *trace_id*, or ``None`` if not found."""
        return self._traces.get(trace_id)

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        """Return traces matching *query*, newest-first, up to ``query.limit``."""
        results: list[Trace] = list(self._traces.values())

        # Filter by service name
        if query.service:
            results = [
                t for t in results
                if query.service in t.service_names
            ]

        # Filter by operation name (any span matches)
        if query.operation:
            results = [
                t for t in results
                if any(s.operation_name == query.operation for s in t.spans)
            ]

        # Filter by time window (root span start time)
        if query.start_time:
            start = _ensure_utc(query.start_time)
            results = [
                t for t in results
                if t.root_span and t.root_span.start_time >= start
            ]
        if query.end_time:
            end = _ensure_utc(query.end_time)
            results = [
                t for t in results
                if t.root_span and t.root_span.start_time <= end
            ]

        # Filter by minimum duration
        if query.min_duration_ms is not None:
            results = [
                t for t in results
                if (t.total_duration_ms >= query.min_duration_ms
                    or any(s.duration_ms >= query.min_duration_ms for s in t.spans))
            ]

        # Filter by status
        if query.status == TraceStatus.ERROR:
            results = [t for t in results if t.error_spans]
        elif query.status == TraceStatus.OK:
            results = [t for t in results if not t.error_spans]

        # Filter by tags (any span must have all requested tags)
        if query.tags:
            filtered: list[Trace] = []
            for trace in results:
                for span in trace.spans:
                    if all(
                        str(span.attributes.get(k, "")) == v
                        for k, v in query.tags.items()
                    ):
                        filtered.append(trace)
                        break
            results = filtered

        # Sort newest-first by root span start time (stable for determinism)
        results.sort(
            key=lambda t: t.root_span.start_time if t.root_span else datetime.min,
            reverse=True,
        )

        # Apply limit
        results = results[: query.limit]

        return TraceSearchResult(traces=results, total=len(results), query=query)

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Return all spans for *trace_id*, or an empty list if not found."""
        trace = self.get_trace(trace_id)
        return trace.spans if trace else []

    def get_failed_spans(self, trace_id: str) -> list[Span]:
        """Return only ERROR-status spans for *trace_id*."""
        trace = self.get_trace(trace_id)
        return trace.error_spans if trace else []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_utc(dt: datetime) -> datetime:
    from datetime import timezone
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
