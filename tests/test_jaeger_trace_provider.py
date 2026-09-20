"""Tests for JaegerTraceProvider and the trace model layer.

Coverage
--------
1.  Successful trace — happy path, all span types
2.  Failed trace — ERROR span detected
3.  Database failure trace — slow + error JDBC span
4.  Missing trace — 404 from Jaeger returns None
5.  Multiple spans — parent-child structure preserved
6.  Nested spans — children_of() traversal
7.  Slow span detection (>= 1 000 ms)
8.  Network error — httpx.RequestError → graceful None
9.  Malformed JSON — graceful None
10. search_traces — happy path, limit honoured
11. search_traces — network error returns empty result
12. get_trace_spans — delegates to get_trace
13. get_failed_spans — returns only ERROR spans
14. Span model: duration_ms computed from start/end when 0
15. Span model: is_root / is_error / is_slow properties
16. Trace model: root_span, error_spans, slow_spans, service_names
17. Trace model: total_duration_ms
18. Trace model: format_summary contains expected fragments
19. TraceStatus derivation: otel.status_code tag
20. TraceStatus derivation: error=true tag
21. TraceStatus derivation: http.status_code >= 500
22. EvidenceCorrelator: error span becomes FACT TRACE evidence
23. EvidenceCorrelator: slow span becomes FACT TRACE evidence
24. EvidenceCorrelator: normal span becomes INFERENCE TRACE evidence
25. EvidenceCorrelator: empty traces returns 0 trace evidence pieces
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import httpx

from rca_agent.models.trace_models import (
    Span,
    SpanEvent,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
    TraceStatus,
)
from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider


# ---------------------------------------------------------------------------
# Helpers — build minimal Jaeger API response dicts
# ---------------------------------------------------------------------------

def _us(dt: datetime) -> int:
    """Convert a UTC datetime to Jaeger-style microsecond epoch."""
    return int(dt.timestamp() * 1_000_000)


def _jaeger_span(
    span_id: str = "aabb1122",
    operation: str = "POST /api/test",
    start_us: int = 1_700_000_000_000_000,
    duration_us: int = 15_000,
    parent_id: str | None = None,
    process_id: str = "p1",
    tags: list[dict] | None = None,
    logs: list[dict] | None = None,
    error: bool = False,
) -> dict:
    refs = []
    if parent_id:
        refs.append({"refType": "CHILD_OF", "traceID": "abc", "spanID": parent_id})
    raw_tags = tags or []
    if error:
        raw_tags = [{"key": "error", "value": True, "type": "bool"}] + raw_tags
    return {
        "traceID": "abc123def456abc1",
        "spanID": span_id,
        "operationName": operation,
        "startTime": start_us,
        "duration": duration_us,
        "processID": process_id,
        "references": refs,
        "tags": raw_tags,
        "logs": logs or [],
    }


def _jaeger_trace(spans: list[dict], trace_id: str = "abc123def456abc1") -> dict:
    return {
        "traceID": trace_id,
        "spans": spans,
        "processes": {
            "p1": {"serviceName": "rke-backend", "tags": []},
            "p2": {"serviceName": "postgres", "tags": []},
        },
    }


def _jaeger_response(traces: list[dict]) -> dict:
    return {"data": traces, "total": len(traces), "errors": None}


def _mock_http_response(payload: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

START_US = 1_700_000_000_000_000   # fixed epoch microseconds for determinism
DURATION_FAST_US = 5_000            # 5 ms  — normal span
DURATION_SLOW_US = 2_000_000        # 2 s   — slow span


@pytest.fixture
def provider():
    """JaegerTraceProvider pointed at a fake URL; HTTP client is mocked per test."""
    return JaegerTraceProvider(base_url="http://fake-jaeger:16686", timeout_seconds=5.0)


# ===========================================================================
# 1. Successful trace — happy path
# ===========================================================================

class TestSuccessfulTrace:
    def test_returns_trace_object(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("span1", "POST /api/sales/cash"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        assert isinstance(result, Trace)
        assert result.trace_id == "abc123def456abc1"

    def test_span_fields_populated(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("span1", "POST /api/sales/cash"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        span = result.spans[0]
        assert span.span_id == "span1"
        assert span.operation_name == "POST /api/sales/cash"
        assert span.service_name == "rke-backend"
        assert span.duration_ms == pytest.approx(15.0)   # 15_000 us → 15 ms
        assert span.is_root is True


# ===========================================================================
# 2. Failed trace — ERROR span detected
# ===========================================================================

class TestFailedTrace:
    def test_error_span_status(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("err1", "POST /api/error", error=True),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        assert len(result.error_spans) == 1
        assert result.error_spans[0].is_error is True

    def test_otel_status_code_error(self, provider: JaegerTraceProvider) -> None:
        tags = [{"key": "otel.status_code", "value": "ERROR", "type": "string"},
                {"key": "otel.status_description", "value": "pool exhausted", "type": "string"}]
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("err2", "db.query", tags=tags),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        span = result.spans[0]
        assert span.status == TraceStatus.ERROR
        assert span.status_message == "pool exhausted"

    def test_error_spans_property(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("ok1", "GET /api/health"),
            _jaeger_span("err1", "POST /api/pay", error=True),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        assert len(result.error_spans) == 1
        assert result.error_spans[0].span_id == "err1"


# ===========================================================================
# 3. Database failure trace — slow + error JDBC span
# ===========================================================================

class TestDatabaseFailureTrace:
    def test_slow_db_span_detected(self, provider: JaegerTraceProvider) -> None:
        db_tags = [
            {"key": "db.system", "value": "postgresql", "type": "string"},
            {"key": "db.operation", "value": "SELECT", "type": "string"},
            {"key": "db.statement", "value": "SELECT pg_sleep(5)", "type": "string"},
        ]
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span(
                "db1", "SELECT pg_sleep(?)",
                duration_us=DURATION_SLOW_US,
                process_id="p2",
                tags=db_tags,
            ),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        slow = result.slow_spans
        assert len(slow) == 1
        assert slow[0].attributes["db.system"] == "postgresql"
        assert slow[0].duration_ms >= 1_000.0

    def test_error_and_slow_span_both_captured(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("s1", "fast", duration_us=DURATION_FAST_US),
            _jaeger_span("s2", "slow-db", duration_us=DURATION_SLOW_US, error=True),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        assert len(result.error_spans) == 1
        assert len(result.slow_spans) == 1
        assert result.error_spans[0].span_id == "s2"


# ===========================================================================
# 4. Missing trace — 404 from Jaeger
# ===========================================================================

class TestMissingTrace:
    def test_returns_none_on_404(self, provider: JaegerTraceProvider) -> None:
        mock_resp = _mock_http_response({}, status_code=404)
        with patch.object(provider._client, "get", return_value=mock_resp):
            result = provider.get_trace("nonexistent")
        assert result is None

    def test_empty_trace_id_returns_none(self, provider: JaegerTraceProvider) -> None:
        result = provider.get_trace("")
        assert result is None

    def test_empty_data_returns_none(self, provider: JaegerTraceProvider) -> None:
        mock_resp = _mock_http_response({"data": []})
        with patch.object(provider._client, "get", return_value=mock_resp):
            result = provider.get_trace("abc")
        assert result is None


# ===========================================================================
# 5. Multiple spans — structure preserved
# ===========================================================================

class TestMultipleSpans:
    def test_all_spans_returned(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("root", "POST /api/sales/cash"),
            _jaeger_span("child1", "sale.create", parent_id="root"),
            _jaeger_span("child2", "SELECT farmers", parent_id="child1"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        assert len(result.spans) == 3

    def test_parent_child_ids_preserved(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("root", "POST /api/sales/cash"),
            _jaeger_span("child1", "sale.create", parent_id="root"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        root = result.root_span
        assert root is not None
        assert root.span_id == "root"
        child = next(s for s in result.spans if s.span_id == "child1")
        assert child.parent_span_id == "root"


# ===========================================================================
# 6. Nested spans — children_of() traversal
# ===========================================================================

class TestNestedSpans:
    def test_children_of_root(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("root", "HTTP"),
            _jaeger_span("c1", "sale.create", parent_id="root"),
            _jaeger_span("c2", "farmer.lookup", parent_id="root"),
            _jaeger_span("gc1", "SELECT", parent_id="c1"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        children = result.children_of("root")
        assert len(children) == 2
        assert {s.span_id for s in children} == {"c1", "c2"}

        grandchildren = result.children_of("c1")
        assert len(grandchildren) == 1
        assert grandchildren[0].span_id == "gc1"


# ===========================================================================
# 7. Slow span detection
# ===========================================================================

class TestSlowSpanDetection:
    def test_exactly_1000ms_is_slow(self) -> None:
        span = Span(
            trace_id="abc", span_id="s1", service_name="svc",
            operation_name="op", duration_ms=1_000.0,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )
        assert span.is_slow is True

    def test_999ms_is_not_slow(self) -> None:
        span = Span(
            trace_id="abc", span_id="s1", service_name="svc",
            operation_name="op", duration_ms=999.0,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert span.is_slow is False

    def test_slow_spans_property_filters_correctly(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("fast", "fast-op", duration_us=5_000),
            _jaeger_span("slow", "slow-op", duration_us=5_000_000),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")

        assert result is not None
        slow = result.slow_spans
        assert len(slow) == 1
        assert slow[0].span_id == "slow"


# ===========================================================================
# 8. Network error — graceful degradation
# ===========================================================================

class TestNetworkError:
    def test_get_trace_returns_none_on_request_error(
        self, provider: JaegerTraceProvider
    ) -> None:
        with patch.object(
            provider._client, "get",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = provider.get_trace("abc")
        assert result is None

    def test_search_traces_returns_empty_on_request_error(
        self, provider: JaegerTraceProvider
    ) -> None:
        with patch.object(
            provider._client, "get",
            side_effect=httpx.ConnectError("connection refused"),
        ):
            result = provider.search_traces(TraceSearchQuery())
        assert result.traces == []
        assert result.total == 0


# ===========================================================================
# 9. Malformed JSON — graceful degradation
# ===========================================================================

class TestMalformedJSON:
    def test_get_trace_returns_none_on_json_error(
        self, provider: JaegerTraceProvider
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("not json")
        with patch.object(provider._client, "get", return_value=mock_resp):
            result = provider.get_trace("abc")
        assert result is None


# ===========================================================================
# 10. search_traces — happy path
# ===========================================================================

class TestSearchTraces:
    def test_returns_multiple_traces(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([
            _jaeger_trace([_jaeger_span("s1", "GET /api/health")], trace_id="trace1"),
            _jaeger_trace([_jaeger_span("s2", "POST /api/pay")], trace_id="trace2"),
        ])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.search_traces(TraceSearchQuery(service="rke-backend", limit=10))

        assert len(result.traces) == 2
        assert result.total == 2

    def test_error_status_filter(self, provider: JaegerTraceProvider) -> None:
        """search_traces with status=ERROR returns only traces with error spans."""
        good_trace = _jaeger_trace([_jaeger_span("s1", "ok")], trace_id="good")
        bad_trace = _jaeger_trace(
            [_jaeger_span("s2", "bad", error=True)], trace_id="bad"
        )
        raw = _jaeger_response([good_trace, bad_trace])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.search_traces(
                TraceSearchQuery(status=TraceStatus.ERROR)
            )

        assert len(result.traces) == 1
        assert result.traces[0].trace_id == "bad"


# ===========================================================================
# 11. search_traces — network error
# ===========================================================================

class TestSearchTracesNetworkError:
    def test_returns_empty_result(self, provider: JaegerTraceProvider) -> None:
        with patch.object(
            provider._client, "get",
            side_effect=httpx.ConnectError("refused"),
        ):
            result = provider.search_traces(TraceSearchQuery())
        assert isinstance(result, TraceSearchResult)
        assert result.traces == []


# ===========================================================================
# 12. get_trace_spans
# ===========================================================================

class TestGetTraceSpans:
    def test_returns_span_list(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("s1", "op1"),
            _jaeger_span("s2", "op2", parent_id="s1"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            spans = provider.get_trace_spans("abc123def456abc1")

        assert len(spans) == 2

    def test_returns_empty_when_trace_missing(self, provider: JaegerTraceProvider) -> None:
        with patch.object(
            provider._client, "get", return_value=_mock_http_response({}, status_code=404)
        ):
            spans = provider.get_trace_spans("missing")
        assert spans == []


# ===========================================================================
# 13. get_failed_spans
# ===========================================================================

class TestGetFailedSpans:
    def test_returns_only_error_spans(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("ok", "healthy"),
            _jaeger_span("fail", "broken", error=True),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            failed = provider.get_failed_spans("abc123def456abc1")

        assert len(failed) == 1
        assert failed[0].span_id == "fail"

    def test_returns_empty_when_no_errors(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([
            _jaeger_span("ok", "healthy"),
        ])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            failed = provider.get_failed_spans("abc123def456abc1")
        assert failed == []


# ===========================================================================
# 14-18. Span / Trace model properties (no HTTP needed)
# ===========================================================================

def _make_span(
    span_id: str = "s1",
    parent_id: str | None = None,
    duration_ms: float = 5.0,
    status: TraceStatus = TraceStatus.UNSET,
) -> Span:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return Span(
        trace_id="trace1", span_id=span_id, parent_span_id=parent_id,
        service_name="svc", operation_name="op",
        start_time=t0, end_time=t0,
        duration_ms=duration_ms, status=status,
    )


class TestSpanModelProperties:
    def test_duration_computed_from_start_end(self) -> None:
        t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc)  # 2 seconds
        span = Span(
            trace_id="t", span_id="s", service_name="svc",
            operation_name="op", start_time=t0, end_time=t1,
            duration_ms=0.0,
        )
        assert span.duration_ms == pytest.approx(2_000.0)

    def test_is_root_true_when_no_parent(self) -> None:
        assert _make_span().is_root is True

    def test_is_root_false_when_has_parent(self) -> None:
        assert _make_span(parent_id="p1").is_root is False

    def test_is_error(self) -> None:
        assert _make_span(status=TraceStatus.ERROR).is_error is True
        assert _make_span(status=TraceStatus.OK).is_error is False

    def test_is_slow_threshold(self) -> None:
        assert _make_span(duration_ms=1_000.0).is_slow is True
        assert _make_span(duration_ms=999.9).is_slow is False


class TestTraceModelProperties:
    def test_root_span_identified(self) -> None:
        trace = Trace(trace_id="t", spans=[
            _make_span("root"),
            _make_span("child", parent_id="root"),
        ])
        assert trace.root_span is not None
        assert trace.root_span.span_id == "root"

    def test_error_spans_filter(self) -> None:
        trace = Trace(trace_id="t", spans=[
            _make_span("ok"),
            _make_span("err", status=TraceStatus.ERROR),
        ])
        assert len(trace.error_spans) == 1
        assert trace.error_spans[0].span_id == "err"

    def test_slow_spans_filter(self) -> None:
        trace = Trace(trace_id="t", spans=[
            _make_span("fast", duration_ms=5.0),
            _make_span("slow", duration_ms=2_000.0),
        ])
        assert len(trace.slow_spans) == 1

    def test_service_names_deduplicated_and_sorted(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        spans = [
            Span(trace_id="t", span_id=f"s{i}", service_name=svc,
                 operation_name="op", start_time=t0, end_time=t0, duration_ms=1.0)
            for i, svc in enumerate(["svc-b", "svc-a", "svc-b"])
        ]
        trace = Trace(trace_id="t", spans=spans)
        assert trace.service_names == ["svc-a", "svc-b"]

    def test_total_duration_ms(self) -> None:
        t0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        t1 = datetime(2024, 1, 1, 0, 0, 3, tzinfo=timezone.utc)  # 3 seconds
        spans = [
            Span(trace_id="t", span_id="s1", service_name="svc",
                 operation_name="op", start_time=t0, end_time=t1, duration_ms=3_000.0),
        ]
        trace = Trace(trace_id="t", spans=spans)
        assert trace.total_duration_ms == pytest.approx(3_000.0)

    def test_format_summary_contains_trace_id(self) -> None:
        trace = Trace(trace_id="abc123def456abc1", spans=[_make_span()])
        summary = trace.format_summary()
        assert "abc123def456ab" in summary


# ===========================================================================
# 19-21. TraceStatus derivation
# ===========================================================================

class TestTraceStatusDerivation:
    def test_otel_status_code_error_tag(self, provider: JaegerTraceProvider) -> None:
        tags = [{"key": "otel.status_code", "value": "ERROR", "type": "string"}]
        raw = _jaeger_response([_jaeger_trace([_jaeger_span("s1", "op", tags=tags)])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")
        assert result is not None
        assert result.spans[0].status == TraceStatus.ERROR

    def test_error_bool_tag(self, provider: JaegerTraceProvider) -> None:
        raw = _jaeger_response([_jaeger_trace([_jaeger_span("s1", "op", error=True)])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")
        assert result is not None
        assert result.spans[0].status == TraceStatus.ERROR

    def test_http_status_500_is_error(self, provider: JaegerTraceProvider) -> None:
        tags = [{"key": "http.status_code", "value": 500, "type": "int64"}]
        raw = _jaeger_response([_jaeger_trace([_jaeger_span("s1", "POST /", tags=tags)])])
        with patch.object(provider._client, "get", return_value=_mock_http_response(raw)):
            result = provider.get_trace("abc123def456abc1")
        assert result is not None
        assert result.spans[0].status == TraceStatus.ERROR


# ===========================================================================
# 22-25. EvidenceCorrelator with trace evidence
# ===========================================================================

class TestEvidenceCorrelatorTraces:
    """Tests that EvidenceCorrelator converts Trace objects to Evidence correctly."""

    def _make_trace_with_error_span(self) -> Trace:
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return Trace(trace_id="trace-err", spans=[
            Span(
                trace_id="trace-err", span_id="root", service_name="rke-backend",
                operation_name="POST /api/pay", start_time=t0, end_time=t0,
                duration_ms=5.0, status=TraceStatus.UNSET,
            ),
            Span(
                trace_id="trace-err", span_id="db1", service_name="rke-backend",
                operation_name="SELECT pg_sleep(?)", start_time=t0, end_time=t0,
                duration_ms=5_000.0, status=TraceStatus.ERROR,
                status_message="pool exhausted",
                parent_span_id="root",
            ),
        ])

    def _make_trace_with_slow_span(self) -> Trace:
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return Trace(trace_id="trace-slow", spans=[
            Span(
                trace_id="trace-slow", span_id="root", service_name="rke-backend",
                operation_name="POST /api/list", start_time=t0, end_time=t0,
                duration_ms=2.0, status=TraceStatus.UNSET,
            ),
            Span(
                trace_id="trace-slow", span_id="db1", service_name="rke-backend",
                operation_name="SELECT *", start_time=t0, end_time=t0,
                duration_ms=3_000.0, status=TraceStatus.UNSET,
                attributes={"db.system": "postgresql"},
                parent_span_id="root",
            ),
        ])

    def _make_normal_trace(self) -> Trace:
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        return Trace(trace_id="trace-ok", spans=[
            Span(
                trace_id="trace-ok", span_id="root", service_name="rke-backend",
                operation_name="GET /api/health", start_time=t0, end_time=t0,
                duration_ms=5.0, status=TraceStatus.OK,
            ),
        ])

    def test_error_span_becomes_fact_trace_evidence(self) -> None:
        from rca_agent.agents.evidence_correlator import EvidenceCorrelator
        from rca_agent.models.evidence import EvidenceType
        from rca_agent.models.rca_result import EvidenceStatement

        correlator = EvidenceCorrelator()
        result = correlator.correlate(traces=[self._make_trace_with_error_span()])

        trace_ev = [e for e in result.evidence if e.evidence_type == EvidenceType.TRACE]
        assert len(trace_ev) >= 1
        error_ev = [e for e in trace_ev if e.statement_type == EvidenceStatement.FACT]
        assert len(error_ev) >= 1
        assert any("pool exhausted" in e.description or "SELECT" in e.description
                   for e in error_ev)

    def test_slow_span_becomes_fact_trace_evidence(self) -> None:
        from rca_agent.agents.evidence_correlator import EvidenceCorrelator
        from rca_agent.models.evidence import EvidenceType
        from rca_agent.models.rca_result import EvidenceStatement

        correlator = EvidenceCorrelator()
        result = correlator.correlate(traces=[self._make_trace_with_slow_span()])

        trace_ev = [e for e in result.evidence if e.evidence_type == EvidenceType.TRACE]
        fact_ev = [e for e in trace_ev if e.statement_type == EvidenceStatement.FACT]
        assert len(fact_ev) >= 1
        assert any("slow" in e.description.lower() or "3000" in e.description
                   for e in fact_ev)

    def test_normal_span_becomes_inference_evidence(self) -> None:
        from rca_agent.agents.evidence_correlator import EvidenceCorrelator
        from rca_agent.models.evidence import EvidenceType
        from rca_agent.models.rca_result import EvidenceStatement

        correlator = EvidenceCorrelator()
        result = correlator.correlate(traces=[self._make_normal_trace()])

        trace_ev = [e for e in result.evidence if e.evidence_type == EvidenceType.TRACE]
        assert len(trace_ev) == 1
        assert trace_ev[0].statement_type == EvidenceStatement.INFERENCE

    def test_empty_traces_returns_no_trace_evidence(self) -> None:
        from rca_agent.agents.evidence_correlator import EvidenceCorrelator
        from rca_agent.models.evidence import EvidenceType

        correlator = EvidenceCorrelator()
        result = correlator.correlate(traces=[])

        trace_ev = [e for e in result.evidence if e.evidence_type == EvidenceType.TRACE]
        assert trace_ev == []

    def test_trace_evidence_source_ref_is_trace_id(self) -> None:
        from rca_agent.agents.evidence_correlator import EvidenceCorrelator
        from rca_agent.models.evidence import EvidenceType

        correlator = EvidenceCorrelator()
        result = correlator.correlate(traces=[self._make_trace_with_error_span()])

        trace_ev = [e for e in result.evidence if e.evidence_type == EvidenceType.TRACE]
        assert all(e.source_ref == "trace-err" for e in trace_ev)
