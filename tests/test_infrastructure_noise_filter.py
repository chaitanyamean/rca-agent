"""Tests for infrastructure-noise filtering in TraceErrorDetector and JaegerMonitor.

Acceptance criteria verified:
1.  Bare POST + network.peer.address + no client.address → infrastructure noise
2.  Bare GET  + network.peer.address + no client.address → infrastructure noise
3.  Bare POST + network.peer.address + client.address present → NOT noise
4.  POST /api/test/incidents/db-pool-exhaustion + client.address → NOT noise
5.  POST /api/test/incidents/db-pool-exhaustion + error status → RCA triggered
6.  Normal application GET route → RCA behavior unchanged
7.  Successful (non-error) traces → ignored (unchanged behavior)
8.  Noise traces never trigger RCA, but ARE marked as processed (deduplication)
9.  Existing JaegerMonitor tests continue passing
10. Existing Phase 1–4 tests continue passing (regression guard)
"""

from __future__ import annotations

from datetime import datetime, timezone
from rca_agent.models.trace_models import Span, Trace, TraceStatus
from rca_agent.monitor.trace_error_detector import TraceErrorDetector
from rca_agent.monitor.config import MonitorConfig
from rca_agent.monitor.jaeger_monitor import JaegerMonitor
from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry
from rca_agent.models.rca_result import RCAResult, RCAStatus

_NOW = datetime(2026, 9, 20, 17, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _span(
    operation: str,
    attributes: dict | None = None,
    status: TraceStatus = TraceStatus.ERROR,
    trace_id: str = "t001",
    span_id: str = "s001",
    service: str = "test-svc",
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        service_name=service,
        operation_name=operation,
        start_time=_NOW,
        end_time=_NOW,
        duration_ms=200.0,
        status=status,
        attributes=attributes or {},
    )


def _trace(trace_id: str, *spans: Span) -> Trace:
    return Trace(trace_id=trace_id, spans=list(spans))


def _noise_trace(
    operation: str = "POST",
    peer: str = "172.22.0.4",
    trace_id: str = "noise-001",
) -> Trace:
    """Build a realistic OTEL exporter noise trace."""
    return _trace(trace_id, _span(
        operation,
        attributes={
            "error": True,
            "error.type": "java.io.IOException",
            "network.peer.address": peer,
            "http.request.method": operation,
        },
        trace_id=trace_id,
    ))


def _app_trace(
    operation: str = "POST /api/test/incidents/db-pool-exhaustion",
    trace_id: str = "app-001",
) -> Trace:
    """Build a realistic application-layer error trace."""
    return _trace(trace_id, _span(
        operation,
        attributes={
            "error": True,
            "error.type": "500",
            "client.address": "192.168.65.1",
            "http.status_code": 500,
            "http.request.method": "POST",
        },
        trace_id=trace_id,
    ))


class _FakeClient:
    def __init__(self, traces: list[Trace]):
        self._traces = traces
        self.search_calls = 0

    def search_traces(self, q):
        from rca_agent.models.trace_models import TraceSearchResult
        self.search_calls += 1
        return TraceSearchResult(traces=self._traces, total=len(self._traces), query=q)

    def list_services(self) -> list[str]:
        return []


class _CountingTrigger:
    def __init__(self):
        self.calls: list[str] = []

    def trigger(self, trace: Trace) -> RCAResult:
        self.calls.append(trace.trace_id)
        return RCAResult(
            incident_id=f"auto-{trace.trace_id[:8]}",
            status=RCAStatus.PARTIAL,
            summary="Fake RCA",
            confidence=0.5,
        )


def _monitor(traces: list[Trace]) -> tuple[JaegerMonitor, _CountingTrigger]:
    trigger = _CountingTrigger()
    monitor = JaegerMonitor(
        client=_FakeClient(traces),
        detector=TraceErrorDetector(),
        registry=ProcessedTraceRegistry(),
        trigger=trigger,
        config=MonitorConfig(jaeger_url="http://fake:16686"),
    )
    return monitor, trigger


# ===========================================================================
# 1–2: Bare HTTP method + peer + no client → noise
# ===========================================================================

class TestIsInfrastructureNoise:
    def test_post_with_peer_no_client_is_noise(self) -> None:
        """Acceptance criterion 1."""
        d = TraceErrorDetector()
        trace = _noise_trace(operation="POST", peer="172.22.0.4")
        assert d.is_infrastructure_noise(trace) is True

    def test_get_with_peer_no_client_is_noise(self) -> None:
        """Acceptance criterion 2."""
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "GET",
            attributes={"error": True, "network.peer.address": "10.0.0.5"},
        ))
        assert d.is_infrastructure_noise(trace) is True

    def test_put_with_peer_no_client_is_noise(self) -> None:
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "PUT",
            attributes={"error": True, "network.peer.address": "10.0.0.5"},
        ))
        assert d.is_infrastructure_noise(trace) is True

    def test_delete_with_peer_no_client_is_noise(self) -> None:
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "DELETE",
            attributes={"net.peer.name": "otel-collector"},
        ))
        assert d.is_infrastructure_noise(trace) is True

    def test_patch_with_peer_no_client_is_noise(self) -> None:
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "PATCH",
            attributes={"net.peer.ip": "192.168.1.50"},
        ))
        assert d.is_infrastructure_noise(trace) is True

    def test_okhttp_connection_shutdown_is_noise(self) -> None:
        """Real-world ConnectionShutdownException pattern."""
        d = TraceErrorDetector()
        trace = _trace("f5d9", _span(
            "POST",
            attributes={
                "error": True,
                "error.type": "okhttp3.internal.http2.ConnectionShutdownException",
                "network.peer.address": "172.22.0.4",
                "http.request.method": "POST",
            },
            trace_id="f5d9",
        ))
        assert d.is_infrastructure_noise(trace) is True

    def test_server_address_attribute_is_noise(self) -> None:
        """OTel semantic conventions >= 1.23 use server.address instead of network.peer.address."""
        d = TraceErrorDetector()
        trace = _trace("ccd8", _span(
            "POST",
            attributes={
                "error": True,
                "error.type": "java.io.IOException",
                "server.address": "otel-collector",
                "server.port": 4318,
                "span.kind": "client",
            },
            trace_id="ccd8",
        ))
        assert d.is_infrastructure_noise(trace) is True

    def test_io_exception_exporter_is_noise(self) -> None:
        """java.io.IOException from OTEL exporter pattern."""
        d = TraceErrorDetector()
        trace = _trace("aabb", _span(
            "POST",
            attributes={
                "error": True,
                "error.type": "java.io.IOException",
                "network.peer.address": "172.22.0.4",
            },
            trace_id="aabb",
        ))
        assert d.is_infrastructure_noise(trace) is True


# ===========================================================================
# 3: Bare POST + peer + client.address present → NOT noise
# ===========================================================================

class TestNotNoise:
    def test_bare_post_with_peer_and_client_is_not_noise(self) -> None:
        """Acceptance criterion 3 — client.address present means inbound request."""
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "POST",
            attributes={
                "error": True,
                "network.peer.address": "172.22.0.4",
                "client.address": "192.168.65.1",  # inbound client
            },
        ))
        assert d.is_infrastructure_noise(trace) is False

    def test_app_route_with_client_is_not_noise(self) -> None:
        """Acceptance criterion 4 — full route always means application span."""
        d = TraceErrorDetector()
        trace = _app_trace()
        assert d.is_infrastructure_noise(trace) is False

    def test_get_route_is_not_noise(self) -> None:
        """GET /health or similar never classified as noise."""
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "GET /health",
            attributes={"client.address": "192.168.1.1"},
        ))
        assert d.is_infrastructure_noise(trace) is False

    def test_post_route_no_peer_is_not_noise(self) -> None:
        """Bare POST with no peer address is ambiguous — not classified as noise."""
        d = TraceErrorDetector()
        trace = _trace("t", _span(
            "POST",
            attributes={"error": True, "http.status_code": 500},
        ))
        assert d.is_infrastructure_noise(trace) is False

    def test_route_with_slash_is_not_noise(self) -> None:
        """Any operation_name containing a slash is definitively not a bare method."""
        d = TraceErrorDetector()
        for op in [
            "POST /api/test/incidents/historical",
            "GET /api/health",
            "DELETE /api/users/123",
            "PUT /api/config/1",
        ]:
            trace = _trace("t", _span(
                op,
                attributes={"network.peer.address": "10.0.0.1", "error": True},
            ))
            assert d.is_infrastructure_noise(trace) is False, (
                f"Operation {op!r} incorrectly classified as noise"
            )

    def test_empty_trace_is_not_noise(self) -> None:
        """Trace with no spans returns False (not noise, not error)."""
        d = TraceErrorDetector()
        assert d.is_infrastructure_noise(Trace(trace_id="empty", spans=[])) is False

    def test_ok_span_is_not_noise(self) -> None:
        """Non-error spans are not noise (they're not errors at all)."""
        d = TraceErrorDetector()
        trace = _trace("t", _span("POST", attributes={}, status=TraceStatus.OK))
        assert d.is_infrastructure_noise(trace) is False

    def test_root_span_determines_classification(self) -> None:
        """When the root span is a real app span, the trace is not noise
        even if a child span has peer-only attributes."""
        d = TraceErrorDetector()
        root = _span(
            "POST /api/test/incidents/db-pool-exhaustion",
            attributes={"client.address": "192.168.65.1", "error": True},
            trace_id="t-multi",
            span_id="root",
        )
        child = _span(
            "POST",
            attributes={"network.peer.address": "172.22.0.4"},
            trace_id="t-multi",
            span_id="child",
        )
        trace = Trace(trace_id="t-multi", spans=[root, child])
        # root span is the real app span — NOT noise
        assert d.is_infrastructure_noise(trace) is False


# ===========================================================================
# 4–5: Application errors trigger RCA, noise does not
# ===========================================================================

class TestMonitorFilterBehavior:
    def test_app_error_triggers_rca(self) -> None:
        """Acceptance criterion 5 — application error must reach the trigger."""
        app = _app_trace("POST /api/test/incidents/db-pool-exhaustion", "app-abc")
        monitor, trigger = _monitor([app])
        count = monitor.run_once()
        assert count == 1
        assert "app-abc" in trigger.calls

    def test_noise_trace_does_not_trigger_rca(self) -> None:
        """Noise traces must NEVER call trigger.trigger()."""
        monitor, trigger = _monitor([_noise_trace(trace_id="noise-skip")])
        count = monitor.run_once()
        assert count == 0
        assert trigger.calls == []

    def test_noise_trace_is_marked_processed(self) -> None:
        """Even suppressed noise traces are marked so they don't log repeatedly."""
        from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry
        registry = ProcessedTraceRegistry()
        trigger = _CountingTrigger()
        monitor = JaegerMonitor(
            client=_FakeClient([_noise_trace(trace_id="noise-mark")]),
            detector=TraceErrorDetector(),
            registry=registry,
            trigger=trigger,
            config=MonitorConfig(jaeger_url="http://x"),
        )
        monitor.run_once()
        # Noise trace marked in registry — won't log again
        assert registry.is_processed("noise-mark")

    def test_noise_and_app_in_same_poll(self) -> None:
        """Only the application trace triggers RCA when both types arrive together."""
        traces = [
            _noise_trace(trace_id="noise-mixed"),
            _app_trace(trace_id="app-mixed"),
        ]
        monitor, trigger = _monitor(traces)
        count = monitor.run_once()
        assert count == 1
        assert "app-mixed" in trigger.calls
        assert "noise-mixed" not in trigger.calls

    def test_multiple_noise_traces_all_suppressed(self) -> None:
        """All noise traces in a poll are suppressed."""
        traces = [_noise_trace(trace_id=f"noise-{i}") for i in range(5)]
        monitor, trigger = _monitor(traces)
        count = monitor.run_once()
        assert count == 0
        assert trigger.calls == []

    def test_multiple_app_errors_all_trigger(self) -> None:
        """Multiple distinct application errors all trigger RCA."""
        traces = [
            _app_trace("POST /api/test/incidents/db-pool-exhaustion", "app-pool"),
            _app_trace("POST /api/test/incidents/backend-error", "app-backend"),
        ]
        monitor, trigger = _monitor(traces)
        count = monitor.run_once()
        assert count == 2
        assert "app-pool" in trigger.calls
        assert "app-backend" in trigger.calls

    def test_noise_does_not_consume_rca_budget(self) -> None:
        """Noise traces returning 0 from run_once is the expected contract."""
        traces = [_noise_trace(trace_id="budget-noise")]
        monitor, trigger = _monitor(traces)
        assert monitor.run_once() == 0

    def test_ok_trace_still_ignored(self) -> None:
        """Acceptance criterion 7 — successful traces remain ignored."""
        ok_trace = Trace(trace_id="ok-t", spans=[
            _span("GET /health", attributes={}, status=TraceStatus.OK)
        ])
        monitor, trigger = _monitor([ok_trace])
        count = monitor.run_once()
        assert count == 0
        assert trigger.calls == []


# ===========================================================================
# Noise reason string
# ===========================================================================

class TestNoiseReason:
    def test_reason_contains_operation(self) -> None:
        d = TraceErrorDetector()
        trace = _noise_trace(operation="POST")
        reason = d.noise_reason(trace)
        assert "POST" in reason
        assert "client.address=absent" in reason

    def test_reason_contains_peer_address(self) -> None:
        d = TraceErrorDetector()
        trace = _noise_trace(peer="172.22.0.4")
        reason = d.noise_reason(trace)
        assert "172.22.0.4" in reason

    def test_reason_on_non_noise_is_still_safe(self) -> None:
        """noise_reason() must not crash even for non-noise traces."""
        d = TraceErrorDetector()
        trace = _app_trace()
        reason = d.noise_reason(trace)
        assert isinstance(reason, str)


# ===========================================================================
# Regression: existing tests untouched
# ===========================================================================

class TestRegressionExistingBehavior:
    def test_is_error_trace_unchanged(self) -> None:
        """is_error_trace() still works exactly as before."""
        d = TraceErrorDetector()
        error = _trace("t", _span("POST", attributes={}, status=TraceStatus.ERROR))
        ok = _trace("t", _span("POST", attributes={}, status=TraceStatus.OK))
        unset = _trace("t", _span("POST", attributes={}, status=TraceStatus.UNSET))
        assert d.is_error_trace(error) is True
        assert d.is_error_trace(ok) is False
        assert d.is_error_trace(unset) is False

    def test_deduplication_still_prevents_double_rca(self) -> None:
        """Same application trace in two polls still triggers RCA only once."""
        app = _app_trace("POST /api/test/incidents/db-pool-exhaustion", "dedup-app")
        client = _FakeClient([app])
        trigger = _CountingTrigger()
        registry = ProcessedTraceRegistry()
        monitor = JaegerMonitor(
            client=client,
            detector=TraceErrorDetector(),
            registry=registry,
            trigger=trigger,
            config=MonitorConfig(jaeger_url="http://x"),
        )
        monitor.run_once()
        monitor.run_once()
        assert trigger.calls.count("dedup-app") == 1

    def test_rca_failure_does_not_stop_monitor(self) -> None:
        """RCA exception on application trace must not crash the monitor."""
        class _FailTrigger:
            def trigger(self, t: Trace) -> RCAResult:
                raise RuntimeError("RCA exploded")

        monitor = JaegerMonitor(
            client=_FakeClient([_app_trace("fail-app")]),
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=_FailTrigger(),
            config=MonitorConfig(jaeger_url="http://x"),
        )
        # Must not raise
        count = monitor.run_once()
        assert count == 1  # attempted (True returned before exception)

    def test_jaeger_unavailable_does_not_crash(self) -> None:
        class _FailClient:
            def search_traces(self, q):
                from rca_agent.models.trace_models import TraceSearchResult
                raise ConnectionError("Jaeger down")
            def list_services(self): return []

        trigger = _CountingTrigger()
        monitor = JaegerMonitor(
            client=_FailClient(),
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=trigger,
            config=MonitorConfig(jaeger_url="http://x"),
        )
        count = monitor.run_once()
        assert count == 0

    def test_existing_phase4_evaluation_unaffected(self) -> None:
        """Phase 4 evaluation imports correctly after changes."""
        from evaluation.datasets.golden_dataset import GOLDEN_DATASET
        assert len(GOLDEN_DATASET) == 6


# ===========================================================================
# Duplicate-logging fix: noise trace logged once, silently skipped after
# ===========================================================================

class TestNoiseDuplicateSuppressionFix:
    """Verifies the fix for repeated logging of infra-noise traces.

    Root cause: is_infrastructure_noise() was checked before is_processed(),
    so the noise-log branch ran on every poll for the same trace ID.
    Fix: is_processed() now runs first; subsequent polls are silently skipped.
    """

    def _make_monitor(self, traces_per_poll: list[Trace]):
        """Build a monitor whose fake client always returns the same trace list."""
        class _FixedClient:
            def __init__(self, ts):
                self._ts = ts
                self.call_count = 0
            def search_traces(self, q):
                from rca_agent.models.trace_models import TraceSearchResult
                self.call_count += 1
                return TraceSearchResult(traces=self._ts, total=len(self._ts), query=q)
            def list_services(self): return []

        trigger = _CountingTrigger()
        client = _FixedClient(traces_per_poll)
        monitor = JaegerMonitor(
            client=client,
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=trigger,
            config=MonitorConfig(jaeger_url="http://x"),
        )
        return monitor, trigger, client

    def test_noise_trace_logged_only_on_first_poll(self, caplog) -> None:
        """First poll: ignore message logged once. Second poll: silence."""
        import logging
        noise = _noise_trace(trace_id="dup-noise-001")
        monitor, trigger, _ = self._make_monitor([noise])

        with caplog.at_level(logging.INFO, logger="rca_agent.monitor.jaeger_monitor"):
            monitor.run_once()   # first poll — should log the ignore message
            first_msgs = [r.message for r in caplog.records]
            caplog.clear()

            monitor.run_once()   # second poll — should be silent (DEBUG only)
            second_msgs = [r.message for r in caplog.records
                          if r.levelno >= logging.INFO]

        assert any("ignoring infrastructure" in m for m in first_msgs), (
            "First poll must log the ignore message"
        )
        assert not any("ignoring infrastructure" in m for m in second_msgs), (
            "Second poll must NOT log the ignore message again"
        )

    def test_noise_trace_never_triggers_rca_across_polls(self) -> None:
        """RCA must never be called for a noise trace, even across many polls."""
        noise = _noise_trace(trace_id="dup-noise-002")
        monitor, trigger, _ = self._make_monitor([noise])

        for _ in range(5):
            monitor.run_once()

        assert trigger.calls == [], (
            "Noise trace must never trigger RCA across any number of polls"
        )

    def test_noise_trace_is_processed_after_first_poll(self) -> None:
        """After the first poll, the noise trace ID must be in the registry."""
        noise = _noise_trace(trace_id="dup-noise-003")
        registry = ProcessedTraceRegistry()
        trigger = _CountingTrigger()

        class _Client:
            def search_traces(self, q):
                from rca_agent.models.trace_models import TraceSearchResult
                return TraceSearchResult(traces=[noise], total=1, query=q)
            def list_services(self): return []

        monitor = JaegerMonitor(
            client=_Client(),
            detector=TraceErrorDetector(),
            registry=registry,
            trigger=trigger,
            config=MonitorConfig(jaeger_url="http://x"),
        )

        assert not registry.is_processed("dup-noise-003")
        monitor.run_once()
        assert registry.is_processed("dup-noise-003"), (
            "Noise trace must be marked processed after first poll"
        )

    def test_app_error_still_triggers_rca_once(self) -> None:
        """Real application error: triggered on first poll, silently deduped after."""
        app = _app_trace("POST /api/test/incidents/db-pool-exhaustion", "dup-app-001")
        monitor, trigger, _ = self._make_monitor([app])

        for _ in range(3):
            monitor.run_once()

        assert trigger.calls.count("dup-app-001") == 1, (
            "Application error must trigger RCA exactly once, not per-poll"
        )
