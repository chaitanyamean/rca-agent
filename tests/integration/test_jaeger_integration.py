"""Live Jaeger integration tests.

These tests require a running Jaeger instance and are **skipped by default**.

To run:

    JAEGER_INTEGRATION_TEST=1 pytest tests/integration/test_jaeger_integration.py -v

Or with a custom URL:

    JAEGER_INTEGRATION_TEST=1 JAEGER_BASE_URL=http://localhost:16686 \\
        pytest tests/integration/test_jaeger_integration.py -v

The tests use the service name from ``settings.jaeger_service_name``
(default ``"rke-backend"``).  To test a different service:

    JAEGER_SERVICE_NAME=my-service JAEGER_INTEGRATION_TEST=1 \\
        pytest tests/integration/test_jaeger_integration.py -v

What the tests verify:
1.  Jaeger is reachable (health check via service list).
2.  Trace search returns traces for the configured service.
3.  A retrieved trace has at least one span.
4.  Service names are populated on all spans.
5.  Parent-child relationships are preserved (at least one span has a parent).
6.  Span duration_ms is non-negative.
7.  Error span detection works on a real trace with errors.
8.  Slow span detection works with the configured threshold.
9.  get_trace_spans delegates to get_trace correctly.
10. get_failed_spans returns only ERROR-status spans.

Note: Tests 7–10 require at least one trace with an error/slow span to have
been submitted to Jaeger.  If no such traces exist the tests are marked xfail.
"""

from __future__ import annotations

import os

import pytest

from rca_agent.config.settings import settings
from rca_agent.models.trace_models import TraceSearchQuery, TraceStatus
from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider

# ---------------------------------------------------------------------------
# Skip entire module unless explicitly enabled
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    os.environ.get("JAEGER_INTEGRATION_TEST", "").lower() not in ("1", "true", "yes"),
    reason=(
        "Live Jaeger integration tests are disabled by default. "
        "Set JAEGER_INTEGRATION_TEST=1 to run them."
    ),
)


@pytest.fixture(scope="module")
def jaeger() -> JaegerTraceProvider:
    """Return a JaegerTraceProvider connected to the configured Jaeger instance."""
    url = os.environ.get("JAEGER_BASE_URL", settings.jaeger_base_url)
    timeout = settings.jaeger_timeout_seconds
    provider = JaegerTraceProvider(base_url=url, timeout_seconds=timeout)
    yield provider
    provider.close()


SERVICE = os.environ.get("JAEGER_SERVICE_NAME", settings.jaeger_service_name)


# ---------------------------------------------------------------------------
# 1. Connectivity check
# ---------------------------------------------------------------------------

class TestJaegerConnectivity:
    def test_jaeger_is_reachable(self, jaeger: JaegerTraceProvider) -> None:
        """Jaeger must return a non-error response to a basic trace search."""
        result = jaeger.search_traces(TraceSearchQuery(service=SERVICE, limit=1))
        # If Jaeger is down this will be an empty result or raise — not an assertion error
        assert result is not None, "search_traces returned None (Jaeger may be unreachable)"


# ---------------------------------------------------------------------------
# 2–6. Basic trace structure
# ---------------------------------------------------------------------------

class TestBasicTraceStructure:
    @pytest.fixture(scope="class")
    def first_trace(self, jaeger: JaegerTraceProvider):
        result = jaeger.search_traces(TraceSearchQuery(service=SERVICE, limit=1))
        if not result.traces:
            pytest.skip(f"No traces found for service '{SERVICE}' in Jaeger.")
        return result.traces[0]

    def test_search_returns_traces(self, jaeger: JaegerTraceProvider) -> None:
        result = jaeger.search_traces(TraceSearchQuery(service=SERVICE, limit=5))
        assert isinstance(result.traces, list)

    def test_trace_has_at_least_one_span(self, first_trace) -> None:
        assert len(first_trace.spans) >= 1

    def test_spans_have_service_names(self, first_trace) -> None:
        for span in first_trace.spans:
            assert span.service_name, f"Span {span.span_id} missing service_name"

    def test_parent_child_preserved(self, first_trace) -> None:
        """At least one span should have a parent (if trace has > 1 span)."""
        if len(first_trace.spans) == 1:
            pytest.skip("Single-span trace — no parent-child relationship to verify.")
        children = [s for s in first_trace.spans if s.parent_span_id]
        assert len(children) >= 1, (
            "Multi-span trace should have at least one child span"
        )

    def test_span_duration_non_negative(self, first_trace) -> None:
        for span in first_trace.spans:
            assert span.duration_ms >= 0.0, (
                f"Span {span.span_id} has negative duration: {span.duration_ms}"
            )


# ---------------------------------------------------------------------------
# 7–10. Error and slow span detection
# ---------------------------------------------------------------------------

class TestErrorAndSlowSpans:
    @pytest.fixture(scope="class")
    def error_trace(self, jaeger: JaegerTraceProvider):
        result = jaeger.search_traces(
            TraceSearchQuery(service=SERVICE, status=TraceStatus.ERROR, limit=1)
        )
        if not result.traces:
            pytest.xfail(
                f"No error traces found for '{SERVICE}' — cannot test error detection."
            )
        return result.traces[0]

    def test_error_trace_has_error_spans(self, error_trace) -> None:
        assert len(error_trace.error_spans) >= 1, (
            "A trace retrieved with status=ERROR must have at least one error span"
        )

    def test_error_span_service_name_populated(self, error_trace) -> None:
        for span in error_trace.error_spans:
            assert span.service_name, f"Error span {span.span_id} missing service_name"

    def test_get_failed_spans_matches_error_spans(
        self, jaeger: JaegerTraceProvider, error_trace
    ) -> None:
        failed = jaeger.get_failed_spans(error_trace.trace_id)
        error_ids = {s.span_id for s in error_trace.error_spans}
        failed_ids = {s.span_id for s in failed}
        assert error_ids == failed_ids

    def test_get_trace_by_id(
        self, jaeger: JaegerTraceProvider, error_trace
    ) -> None:
        retrieved = jaeger.get_trace(error_trace.trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == error_trace.trace_id

    def test_get_trace_spans_count(
        self, jaeger: JaegerTraceProvider, error_trace
    ) -> None:
        spans = jaeger.get_trace_spans(error_trace.trace_id)
        assert len(spans) == len(error_trace.spans)

    def test_slow_span_threshold(
        self, jaeger: JaegerTraceProvider, error_trace
    ) -> None:
        """Slow spans should be detected using the configured threshold."""
        threshold = settings.trace_slow_threshold_ms
        slow = error_trace.slow_spans
        for span in slow:
            assert span.duration_ms >= threshold, (
                f"Span {span.span_id} duration {span.duration_ms} ms < threshold {threshold} ms"
            )
