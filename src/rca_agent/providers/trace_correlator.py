"""Deterministic trace ↔ log correlation utilities.

Purpose
-------
RKE (and any OTEL-instrumented service) injects ``trace_id`` and ``span_id``
into every structured log line via the SLF4J MDC.  These IDs match the
identifiers in the distributed trace stored in Jaeger.

This module provides utilities for:
1. Finding all log entries that belong to a specific trace.
2. Finding all log entries that belong to a specific span within a trace.
3. Building a ``TraceLogCorrelationMap`` — an index that groups logs by
   (trace_id, span_id) for efficient lookup during RCA investigation.
4. Identifying which spans produced errors (cross-referenced with logs).

All correlation is **deterministic** — the result depends only on the
content of the logs and traces, never on LLM inference.  The LLM receives
the correlation output as pre-computed evidence; it does not decide which
logs match which spans.

Correlation contract
--------------------
A log entry is correlated to a span when:
    ``log_entry.trace_id == span.trace_id``  AND
    ``log_entry.extra_fields.get("span_id") == span.span_id``

A log entry is correlated to a trace when:
    ``log_entry.trace_id == trace.trace_id``

These comparisons are string equality — no fuzzy matching, no LLM involvement.

Usage::

    from rca_agent.providers.trace_correlator import TraceLogCorrelator
    from rca_agent.providers.local_log_provider import LocalLogProvider
    from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider

    log_provider = LocalLogProvider("/path/to/logs")
    trace_provider = JaegerTraceProvider("http://localhost:16686")

    correlator = TraceLogCorrelator(log_provider, trace_provider)

    # All logs for a trace
    logs_for_trace = correlator.logs_for_trace("abc123def456")

    # Logs for a specific span
    logs_for_span = correlator.logs_for_span("abc123def456", "span001")

    # Build a full index for a set of trace IDs
    index = correlator.build_index(["abc123def456", "def789abc012"])
    for span_id, log_ids in index.span_to_logs.items():
        print(f"Span {span_id}: {len(log_ids)} correlated log(s)")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rca_agent.models.log_entry import LogEntry, LogSearchQuery
from rca_agent.models.trace_models import Span, Trace

logger = logging.getLogger(__name__)


@dataclass
class TraceLogCorrelationMap:
    """Index linking spans to their correlated log entries.

    Built by ``TraceLogCorrelator.build_index()`` for efficient lookups
    during evidence correlation.

    All mappings are keyed by string IDs (trace_id or span_id).
    """

    # trace_id → list of correlated LogEntry objects
    trace_to_logs: dict[str, list[LogEntry]] = field(default_factory=dict)

    # span_id → list of correlated LogEntry objects (subset of trace logs)
    span_to_logs: dict[str, list[LogEntry]] = field(default_factory=dict)

    # span_id → list of Span objects (for reverse lookup)
    spans: dict[str, Span] = field(default_factory=dict)

    def logs_for_trace(self, trace_id: str) -> list[LogEntry]:
        """Return all logs correlated to *trace_id*."""
        return self.trace_to_logs.get(trace_id, [])

    def logs_for_span(self, span_id: str) -> list[LogEntry]:
        """Return all logs correlated to *span_id*."""
        return self.span_to_logs.get(span_id, [])

    def spans_with_correlated_logs(self) -> list[Span]:
        """Return all spans that have at least one correlated log entry."""
        return [
            self.spans[sid]
            for sid in self.span_to_logs
            if sid in self.spans and self.span_to_logs[sid]
        ]

    def error_spans_with_logs(self) -> list[tuple[Span, list[LogEntry]]]:
        """Return (span, logs) pairs for ERROR spans that have correlated logs."""
        result: list[tuple[Span, list[LogEntry]]] = []
        for span in self.spans.values():
            if span.is_error:
                logs = self.span_to_logs.get(span.span_id, [])
                if logs:
                    result.append((span, logs))
        return result

    def format_summary(self) -> str:
        """Return a compact human-readable summary for use in RCA prompts."""
        total_logs = sum(len(v) for v in self.trace_to_logs.values())
        total_spans_with_logs = len(
            [sid for sid, logs in self.span_to_logs.items() if logs]
        )
        lines = [
            f"Trace/log correlation: "
            f"{len(self.trace_to_logs)} trace(s), "
            f"{total_logs} correlated log(s), "
            f"{total_spans_with_logs} span(s) with logs"
        ]
        for span_id, logs in self.span_to_logs.items():
            if logs:
                span = self.spans.get(span_id)
                svc = span.service_name if span else "?"
                op = span.operation_name if span else span_id
                lines.append(
                    f"  Span {span_id[:8]} [{svc}/{op}]: "
                    f"{len(logs)} log(s)"
                )
        return "\n".join(lines)


class TraceLogCorrelator:
    """Correlates distributed traces with structured log entries.

    This is a pure-function utility — it reads from the providers but
    never writes, never calls the LLM, and never modifies any state.

    Parameters
    ----------
    log_provider:
        Any ``LogProvider`` implementation (local file, RKE, etc.).
    """

    def __init__(self, log_provider: object) -> None:
        self._log = log_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def logs_for_trace(self, trace_id: str) -> list[LogEntry]:
        """Return all log entries carrying *trace_id*.

        Uses ``LogProvider.get_logs_by_trace_id()`` — the only correct
        place to do this filtering.  Never uses the LLM.
        """
        result = self._log.get_logs_by_trace_id(trace_id)
        return result.entries

    def logs_for_span(self, trace_id: str, span_id: str) -> list[LogEntry]:
        """Return log entries matching both *trace_id* and *span_id*.

        Filters the trace's logs by ``extra_fields.span_id``.
        This is a deterministic, exact-match operation.
        """
        trace_logs = self.logs_for_trace(trace_id)
        return [
            e for e in trace_logs
            if e.extra_fields.get("span_id") == span_id
        ]

    def build_index(
        self,
        traces: list[Trace],
    ) -> TraceLogCorrelationMap:
        """Build a full correlation index for the given traces.

        For each trace, fetches all correlated log entries from the
        log provider and groups them by (trace_id, span_id).

        Parameters
        ----------
        traces:
            The traces to index.  Typically retrieved from a ``TraceProvider``
            during the investigation workflow.

        Returns
        -------
        TraceLogCorrelationMap
            Indexed map ready for efficient evidence assembly.
        """
        index = TraceLogCorrelationMap()

        for trace in traces:
            # Fetch all logs for this trace in one call
            trace_logs = self.logs_for_trace(trace.trace_id)
            index.trace_to_logs[trace.trace_id] = trace_logs

            # Register all spans in the index
            for span in trace.spans:
                index.spans[span.span_id] = span

            # Group logs by span_id
            for log_entry in trace_logs:
                raw_span_id = log_entry.extra_fields.get("span_id")
                if raw_span_id:
                    sid = str(raw_span_id)
                    index.span_to_logs.setdefault(sid, []).append(log_entry)

            logger.debug(
                "trace_correlator: trace %s → %d log(s) across %d span(s)",
                trace.trace_id[:16],
                len(trace_logs),
                len(trace.spans),
            )

        return index

    def correlate_error_spans(
        self,
        traces: list[Trace],
    ) -> list[dict]:
        """Return a list of evidence dicts for error spans correlated with logs.

        Each dict contains:
        ``trace_id``, ``span_id``, ``service``, ``operation``, ``duration_ms``,
        ``error_message``, ``correlated_logs`` (list of log messages).

        This is the primary output consumed by the RCA workflow for trace-backed
        root cause generation.  It provides provenance without LLM involvement.
        """
        index = self.build_index(traces)
        results: list[dict] = []

        for trace in traces:
            for span in trace.error_spans:
                correlated = index.logs_for_span(span.span_id)
                results.append({
                    "trace_id": trace.trace_id,
                    "span_id": span.span_id,
                    "service": span.service_name,
                    "operation": span.operation_name,
                    "duration_ms": span.duration_ms,
                    "error_message": (
                        span.exception_message or span.status_message or "unknown error"
                    ),
                    "correlated_log_count": len(correlated),
                    "correlated_logs": [e.message[:120] for e in correlated[:5]],
                    "statement_type": "FACT",  # directly observed
                })

        return results

    def correlate_slow_spans(
        self,
        traces: list[Trace],
    ) -> list[dict]:
        """Return evidence dicts for slow spans correlated with logs.

        Similar to ``correlate_error_spans`` but for performance-degrading spans.
        Uses the configurable threshold from ``settings.trace_slow_threshold_ms``.
        """
        index = self.build_index(traces)
        results: list[dict] = []

        for trace in traces:
            for span in trace.slow_spans:
                if span.is_error:
                    continue  # covered by correlate_error_spans
                correlated = index.logs_for_span(span.span_id)
                db_info = {}
                for attr_key in ("db.system", "db.operation", "db.statement"):
                    if attr_key in span.attributes:
                        db_info[attr_key] = str(span.attributes[attr_key])[:80]
                results.append({
                    "trace_id": trace.trace_id,
                    "span_id": span.span_id,
                    "service": span.service_name,
                    "operation": span.operation_name,
                    "duration_ms": span.duration_ms,
                    "db_attributes": db_info,
                    "correlated_log_count": len(correlated),
                    "correlated_logs": [e.message[:120] for e in correlated[:5]],
                    "statement_type": "FACT",
                })

        return results
