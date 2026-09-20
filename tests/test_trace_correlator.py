"""Tests for the TraceLogCorrelator and TraceLogCorrelationMap.

Coverage
--------
1.  logs_for_trace returns matching entries
2.  logs_for_trace returns empty for unknown trace_id
3.  logs_for_span filters by span_id from extra_fields
4.  logs_for_span returns empty when span_id not in extra_fields
5.  build_index groups logs by trace and span
6.  build_index handles multiple traces
7.  build_index handles traces with no correlated logs
8.  TraceLogCorrelationMap.error_spans_with_logs returns correct pairs
9.  TraceLogCorrelationMap.spans_with_correlated_logs filters correctly
10. TraceLogCorrelationMap.format_summary contains key info
11. correlate_error_spans: one error span with correlated log
12. correlate_error_spans: error span with no log still included
13. correlate_slow_spans: slow span captured with db attributes
14. correlate_slow_spans: error spans excluded (not double-counted)
15. Correlation is exact-match — different trace_id is not correlated
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.trace_models import Span, Trace, TraceStatus
from rca_agent.providers.trace_correlator import TraceLogCorrelationMap, TraceLogCorrelator

NOW = datetime.now(timezone.utc)
TRACE_A = "aaaa1111bbbb2222"
TRACE_B = "cccc3333dddd4444"
SPAN_1 = "span0001"
SPAN_2 = "span0002"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(
    message: str,
    trace_id: str = TRACE_A,
    span_id: str | None = SPAN_1,
    level: str = "ERROR",
) -> LogEntry:
    data: dict = {
        "timestamp": NOW.isoformat(),
        "service": "rke-backend",
        "level": level,
        "message": message,
        "trace_id": trace_id,
    }
    if span_id:
        data["span_id"] = span_id
    return LogEntry.from_raw_line(json.dumps(data))


def _span(
    span_id: str = SPAN_1,
    trace_id: str = TRACE_A,
    duration_ms: float = 5.0,
    status: TraceStatus = TraceStatus.UNSET,
    parent_id: str | None = None,
) -> Span:
    return Span(
        trace_id=trace_id, span_id=span_id, parent_span_id=parent_id,
        service_name="rke-backend", operation_name="POST /api/test",
        start_time=NOW, end_time=NOW, duration_ms=duration_ms, status=status,
    )


def _trace(
    trace_id: str = TRACE_A,
    spans: list[Span] | None = None,
) -> Trace:
    return Trace(trace_id=trace_id, spans=spans or [_span(trace_id=trace_id)])


class FakeLogProvider:
    """Returns pre-loaded entries for specific trace_ids."""

    def __init__(self, entries: list[LogEntry]) -> None:
        self._entries = entries

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        matching = [e for e in self._entries if e.trace_id == trace_id]
        return LogSearchResult(entries=matching, query=LogSearchQuery(trace_id=trace_id))

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        return LogSearchResult(entries=self._entries, query=query)

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        return next((e for e in self._entries if e.id == log_id), None)


# ---------------------------------------------------------------------------
# 1–4. logs_for_trace / logs_for_span
# ---------------------------------------------------------------------------

class TestLogsForTrace:
    def test_returns_matching_entries(self) -> None:
        log = _log("db error", trace_id=TRACE_A)
        provider = FakeLogProvider([log])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_trace(TRACE_A)
        assert len(result) == 1
        assert result[0].trace_id == TRACE_A

    def test_returns_empty_for_unknown_trace(self) -> None:
        provider = FakeLogProvider([])
        correlator = TraceLogCorrelator(provider)
        assert correlator.logs_for_trace("unknown-trace") == []

    def test_only_returns_matching_trace(self) -> None:
        log_a = _log("error in A", trace_id=TRACE_A)
        log_b = _log("error in B", trace_id=TRACE_B)
        provider = FakeLogProvider([log_a, log_b])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_trace(TRACE_A)
        assert all(e.trace_id == TRACE_A for e in result)
        assert len(result) == 1


class TestLogsForSpan:
    def test_filters_by_span_id(self) -> None:
        log1 = _log("span 1 error", trace_id=TRACE_A, span_id=SPAN_1)
        log2 = _log("span 2 error", trace_id=TRACE_A, span_id=SPAN_2)
        provider = FakeLogProvider([log1, log2])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_span(TRACE_A, SPAN_1)
        assert len(result) == 1
        assert result[0].extra_fields.get("span_id") == SPAN_1

    def test_returns_empty_when_no_span_id_in_extras(self) -> None:
        log = _log("no span id", trace_id=TRACE_A, span_id=None)
        provider = FakeLogProvider([log])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_span(TRACE_A, SPAN_1)
        assert result == []

    def test_different_span_not_returned(self) -> None:
        log = _log("span 2 only", trace_id=TRACE_A, span_id=SPAN_2)
        provider = FakeLogProvider([log])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_span(TRACE_A, SPAN_1)
        assert result == []


# ---------------------------------------------------------------------------
# 5–7. build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_groups_logs_by_trace(self) -> None:
        log_a1 = _log("a error 1", trace_id=TRACE_A, span_id=SPAN_1)
        log_a2 = _log("a error 2", trace_id=TRACE_A, span_id=SPAN_2)
        log_b1 = _log("b error", trace_id=TRACE_B, span_id=SPAN_1)
        provider = FakeLogProvider([log_a1, log_a2, log_b1])
        correlator = TraceLogCorrelator(provider)
        index = correlator.build_index([_trace(TRACE_A), _trace(TRACE_B)])
        assert len(index.trace_to_logs[TRACE_A]) == 2
        assert len(index.trace_to_logs[TRACE_B]) == 1

    def test_groups_logs_by_span(self) -> None:
        log1 = _log("e1", trace_id=TRACE_A, span_id=SPAN_1)
        log2 = _log("e2", trace_id=TRACE_A, span_id=SPAN_2)
        provider = FakeLogProvider([log1, log2])
        correlator = TraceLogCorrelator(provider)
        trace = _trace(TRACE_A, [
            _span(SPAN_1, TRACE_A),
            _span(SPAN_2, TRACE_A),
        ])
        index = correlator.build_index([trace])
        assert len(index.span_to_logs.get(SPAN_1, [])) == 1
        assert len(index.span_to_logs.get(SPAN_2, [])) == 1

    def test_trace_with_no_logs(self) -> None:
        provider = FakeLogProvider([])
        correlator = TraceLogCorrelator(provider)
        index = correlator.build_index([_trace(TRACE_A)])
        assert index.trace_to_logs[TRACE_A] == []
        assert SPAN_1 in index.spans


# ---------------------------------------------------------------------------
# 8–10. TraceLogCorrelationMap methods
# ---------------------------------------------------------------------------

class TestCorrelationMap:
    def _build_map(self) -> TraceLogCorrelationMap:
        err_span = _span(SPAN_1, status=TraceStatus.ERROR, duration_ms=5_100.0)
        ok_span = _span(SPAN_2)
        log1 = _log("error in span1", trace_id=TRACE_A, span_id=SPAN_1)

        index = TraceLogCorrelationMap()
        index.trace_to_logs[TRACE_A] = [log1]
        index.span_to_logs[SPAN_1] = [log1]
        index.spans[SPAN_1] = err_span
        index.spans[SPAN_2] = ok_span
        return index

    def test_error_spans_with_logs(self) -> None:
        index = self._build_map()
        pairs = index.error_spans_with_logs()
        assert len(pairs) == 1
        span, logs = pairs[0]
        assert span.span_id == SPAN_1
        assert len(logs) == 1

    def test_spans_with_correlated_logs(self) -> None:
        index = self._build_map()
        spans = index.spans_with_correlated_logs()
        span_ids = {s.span_id for s in spans}
        assert SPAN_1 in span_ids
        assert SPAN_2 not in span_ids

    def test_format_summary_contains_trace_info(self) -> None:
        index = self._build_map()
        summary = index.format_summary()
        assert "1 trace" in summary.lower() or "trace" in summary.lower()
        assert "1 correlated log" in summary.lower() or "log" in summary.lower()


# ---------------------------------------------------------------------------
# 11–14. correlate_error_spans / correlate_slow_spans
# ---------------------------------------------------------------------------

class TestCorrelateErrorSpans:
    def test_error_span_with_log_included(self) -> None:
        log = _log("pool exhausted", trace_id=TRACE_A, span_id=SPAN_1)
        provider = FakeLogProvider([log])
        correlator = TraceLogCorrelator(provider)
        err_span = _span(SPAN_1, status=TraceStatus.ERROR, duration_ms=5_000.0)
        trace = _trace(TRACE_A, [err_span])
        results = correlator.correlate_error_spans([trace])
        assert len(results) == 1
        assert results[0]["trace_id"] == TRACE_A
        assert results[0]["span_id"] == SPAN_1
        assert results[0]["correlated_log_count"] == 1
        assert results[0]["statement_type"] == "FACT"

    def test_error_span_without_log_still_included(self) -> None:
        provider = FakeLogProvider([])
        correlator = TraceLogCorrelator(provider)
        err_span = _span(SPAN_1, status=TraceStatus.ERROR)
        trace = _trace(TRACE_A, [err_span])
        results = correlator.correlate_error_spans([trace])
        assert len(results) == 1
        assert results[0]["correlated_log_count"] == 0

    def test_no_error_spans_returns_empty(self) -> None:
        provider = FakeLogProvider([])
        correlator = TraceLogCorrelator(provider)
        trace = _trace(TRACE_A, [_span(SPAN_1)])
        assert correlator.correlate_error_spans([trace]) == []


class TestCorrelateSlowSpans:
    def test_slow_span_captured(self) -> None:
        provider = FakeLogProvider([])
        correlator = TraceLogCorrelator(provider)
        slow = Span(
            trace_id=TRACE_A, span_id=SPAN_1, service_name="rke-backend",
            operation_name="SELECT *", start_time=NOW, end_time=NOW,
            duration_ms=5_000.0, status=TraceStatus.UNSET,
            attributes={"db.system": "postgresql", "db.operation": "SELECT"},
        )
        trace = _trace(TRACE_A, [slow])
        results = correlator.correlate_slow_spans([trace])
        assert len(results) == 1
        assert results[0]["duration_ms"] == 5_000.0
        assert results[0]["db_attributes"]["db.system"] == "postgresql"

    def test_error_spans_excluded_from_slow(self) -> None:
        """Error spans should not appear in slow span results (avoid double-counting)."""
        provider = FakeLogProvider([])
        correlator = TraceLogCorrelator(provider)
        error_slow = Span(
            trace_id=TRACE_A, span_id=SPAN_1, service_name="rke-backend",
            operation_name="op", start_time=NOW, end_time=NOW,
            duration_ms=5_000.0, status=TraceStatus.ERROR,
        )
        trace = _trace(TRACE_A, [error_slow])
        results = correlator.correlate_slow_spans([trace])
        assert results == []


# ---------------------------------------------------------------------------
# 15. Exact-match guarantee
# ---------------------------------------------------------------------------

class TestExactMatchCorrelation:
    def test_different_trace_id_not_correlated(self) -> None:
        log = _log("error", trace_id=TRACE_B)
        provider = FakeLogProvider([log])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_trace(TRACE_A)
        assert result == []

    def test_partial_trace_id_not_matched(self) -> None:
        # Only the first 8 chars of TRACE_A
        partial = TRACE_A[:8]
        log = _log("error", trace_id=partial)
        provider = FakeLogProvider([log])
        correlator = TraceLogCorrelator(provider)
        result = correlator.logs_for_trace(TRACE_A)
        assert result == []
