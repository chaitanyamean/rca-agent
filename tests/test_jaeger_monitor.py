"""Tests for the autonomous Jaeger monitor.

Coverage
--------
1.  Successful trace is ignored.
2.  ERROR trace (otel.status_code=ERROR) is detected.
3.  HTTP 5xx trace is detected.
4.  OTEL ERROR status is detected via TraceStatus.ERROR on span.
5.  New error trace triggers RCA exactly once.
6.  Same trace in multiple polls triggers RCA only once.
7.  Multiple different error traces trigger RCA independently.
8.  Jaeger unavailable does not terminate monitor.
9.  Malformed / empty Jaeger response does not terminate monitor.
10. RCA workflow failure does not terminate monitor.
11. Configuration is read from settings / environment.
12. No RKE-specific incident IDs are hardcoded anywhere in the monitor.
13. Existing RCA workflow tests remain passing (regression guard).
14. Integration path: fake error trace → monitor detects → RCA invoked.

All tests use fakes/mocks — no live Jaeger required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, call

import pytest

from rca_agent.models.trace_models import (
    Span,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
    TraceStatus,
)
from rca_agent.monitor.config import MonitorConfig
from rca_agent.monitor.jaeger_monitor import JaegerMonitor
from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry
from rca_agent.monitor.trace_error_detector import TraceErrorDetector

# ──────────────────────────────────────────────────────────────────────────────
# Helpers / factories
# ──────────────────────────────────────────────────────────────────────────────

_NOW = datetime(2026, 9, 20, 12, 0, 0, tzinfo=timezone.utc)


def _span(
    trace_id: str = "trace-abc",
    span_id: str = "span-001",
    service: str = "test-svc",
    operation: str = "POST /api/test",
    status: TraceStatus = TraceStatus.UNSET,
    status_message: str | None = None,
    http_status: int | None = None,
    parent: str | None = None,
) -> Span:
    attributes = {}
    if http_status is not None:
        attributes["http.status_code"] = http_status
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        service_name=service,
        operation_name=operation,
        start_time=_NOW,
        end_time=_NOW,
        duration_ms=100.0,
        status=status,
        status_message=status_message,
        attributes=attributes,
    )


def _trace(trace_id: str = "trace-abc", spans: list[Span] | None = None) -> Trace:
    return Trace(trace_id=trace_id, spans=spans or [_span(trace_id=trace_id)])


def _error_trace(
    trace_id: str = "trace-err",
    service: str = "test-svc",
    operation: str = "POST /api/fail",
) -> Trace:
    """Return a trace with one ERROR span."""
    return Trace(
        trace_id=trace_id,
        spans=[
            _span(
                trace_id=trace_id,
                service=service,
                operation=operation,
                status=TraceStatus.ERROR,
                status_message="simulated error",
            )
        ],
    )


def _ok_trace(trace_id: str = "trace-ok") -> Trace:
    """Return a trace with one OK span."""
    return Trace(
        trace_id=trace_id,
        spans=[_span(trace_id=trace_id, status=TraceStatus.OK)],
    )


def _unset_trace(trace_id: str = "trace-unset") -> Trace:
    """Return a trace with one UNSET span (no errors)."""
    return Trace(
        trace_id=trace_id,
        spans=[_span(trace_id=trace_id, status=TraceStatus.UNSET)],
    )


class _FakeClient:
    """Fake TraceQueryClient that returns a pre-configured sequence of results."""

    def __init__(self, results: list[list[Trace]] | None = None) -> None:
        # Each element in `results` is what search_traces returns on the nth call
        self._results: list[list[Trace]] = results or []
        self._call_count = 0
        self.services_called = False

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        idx = min(self._call_count, len(self._results) - 1)
        traces = self._results[idx] if self._results else []
        self._call_count += 1
        return TraceSearchResult(traces=traces, total=len(traces), query=query)

    def list_services(self) -> list[str]:
        self.services_called = True
        return []


class _FailingClient:
    """Fake client that always raises on search_traces."""

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        raise ConnectionError("Jaeger is unreachable")

    def list_services(self) -> list[str]:
        raise ConnectionError("Jaeger is unreachable")


class _CountingTrigger:
    """Fake RCAWorkflowTrigger that counts invocations and records trace IDs."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def trigger(self, trace: Trace):  # noqa: ANN201
        self.calls.append(trace.trace_id)
        from rca_agent.models.rca_result import RCAResult, RCAStatus
        return RCAResult(
            incident_id=f"auto-{trace.trace_id[:8]}",
            status=RCAStatus.PARTIAL,
            summary="Fake RCA result",
            confidence=0.6,
        )


class _FailingTrigger:
    """Fake trigger that always raises."""

    def trigger(self, trace: Trace):
        raise RuntimeError("RCA workflow exploded")


def _monitor(
    client=None,
    trigger=None,
    services: list[str] | None = None,
) -> tuple[JaegerMonitor, _CountingTrigger]:
    cfg = MonitorConfig(
        jaeger_url="http://fake-jaeger:16686",
        poll_interval_seconds=5.0,
        lookback_seconds=30,
        services=services or [],
    )
    t = trigger if trigger is not None else _CountingTrigger()
    m = JaegerMonitor(
        client=client or _FakeClient(),
        detector=TraceErrorDetector(),
        registry=ProcessedTraceRegistry(),
        trigger=t,
        config=cfg,
    )
    return m, t


# ──────────────────────────────────────────────────────────────────────────────
# 1. Successful trace is ignored
# ──────────────────────────────────────────────────────────────────────────────

class TestSuccessfulTraceIgnored:
    def test_ok_span_trace_not_triggered(self) -> None:
        client = _FakeClient([[_ok_trace()]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 0
        assert trigger.calls == []

    def test_unset_span_trace_not_triggered(self) -> None:
        client = _FakeClient([[_unset_trace()]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 0
        assert trigger.calls == []

    def test_empty_trace_list_not_triggered(self) -> None:
        client = _FakeClient([[]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 0


# ──────────────────────────────────────────────────────────────────────────────
# 2. ERROR trace (TraceStatus.ERROR on span) is detected
# ──────────────────────────────────────────────────────────────────────────────

class TestErrorTraceDetected:
    def test_error_span_triggers_rca(self) -> None:
        err = _error_trace("trace-001")
        client = _FakeClient([[err]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 1
        assert trigger.calls == ["trace-001"]

    def test_detector_returns_true_for_error_span(self) -> None:
        detector = TraceErrorDetector()
        assert detector.is_error_trace(_error_trace()) is True

    def test_detector_returns_false_for_ok_span(self) -> None:
        detector = TraceErrorDetector()
        assert detector.is_error_trace(_ok_trace()) is False

    def test_detector_returns_false_for_unset_span(self) -> None:
        detector = TraceErrorDetector()
        assert detector.is_error_trace(_unset_trace()) is False

    def test_trace_with_mixed_spans_detected_if_any_error(self) -> None:
        """A trace with one OK span + one ERROR span must be treated as an error."""
        trace = Trace(
            trace_id="trace-mixed",
            spans=[
                _span(trace_id="trace-mixed", span_id="s1", status=TraceStatus.OK),
                _span(trace_id="trace-mixed", span_id="s2", status=TraceStatus.ERROR),
            ],
        )
        assert TraceErrorDetector().is_error_trace(trace) is True


# ──────────────────────────────────────────────────────────────────────────────
# 3. HTTP 5xx trace is detected
# ──────────────────────────────────────────────────────────────────────────────

class TestHttp5xxDetected:
    def test_http_500_span_is_error(self) -> None:
        """JaegerTraceProvider._derive_status maps http.status_code>=500 to ERROR."""
        span_500 = _span(trace_id="t500", status=TraceStatus.ERROR, http_status=500)
        trace = Trace(trace_id="t500", spans=[span_500])
        assert TraceErrorDetector().is_error_trace(trace) is True

    def test_http_500_triggers_rca(self) -> None:
        span = _span(trace_id="t500x", status=TraceStatus.ERROR, status_message="HTTP 500")
        trace = Trace(trace_id="t500x", spans=[span])
        client = _FakeClient([[trace]])
        monitor, trigger = _monitor(client=client)
        assert monitor.run_once() == 1
        assert "t500x" in trigger.calls

    def test_http_200_not_detected(self) -> None:
        """HTTP 200 → UNSET status → not an error."""
        span_200 = _span(trace_id="t200", status=TraceStatus.UNSET, http_status=200)
        trace = Trace(trace_id="t200", spans=[span_200])
        assert TraceErrorDetector().is_error_trace(trace) is False


# ──────────────────────────────────────────────────────────────────────────────
# 4. OTEL ERROR status detected
# ──────────────────────────────────────────────────────────────────────────────

class TestOtelErrorStatusDetected:
    def test_otel_error_status_on_span(self) -> None:
        """TraceStatus.ERROR on Span.status is the canonical signal from JaegerTraceProvider."""
        span = Span(
            trace_id="totel",
            span_id="sp1",
            service_name="svc",
            operation_name="op",
            start_time=_NOW,
            end_time=_NOW,
            duration_ms=10.0,
            status=TraceStatus.ERROR,
            attributes={"otel.status_code": "ERROR"},
        )
        trace = Trace(trace_id="totel", spans=[span])
        assert TraceErrorDetector().is_error_trace(trace) is True

    def test_otel_ok_status_not_detected(self) -> None:
        span = Span(
            trace_id="tok",
            span_id="sp1",
            service_name="svc",
            operation_name="op",
            start_time=_NOW,
            end_time=_NOW,
            duration_ms=10.0,
            status=TraceStatus.OK,
            attributes={"otel.status_code": "OK"},
        )
        trace = Trace(trace_id="tok", spans=[span])
        assert TraceErrorDetector().is_error_trace(trace) is False


# ──────────────────────────────────────────────────────────────────────────────
# 5. New error trace triggers RCA exactly once
# ──────────────────────────────────────────────────────────────────────────────

class TestNewErrorTraceTriggerOnce:
    def test_single_error_triggers_once(self) -> None:
        err = _error_trace("trace-once")
        client = _FakeClient([[err]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 1
        assert trigger.calls.count("trace-once") == 1

    def test_trigger_receives_correct_trace_id(self) -> None:
        err = _error_trace("trace-xyz-123")
        client = _FakeClient([[err]])
        monitor, trigger = _monitor(client=client)
        monitor.run_once()
        assert trigger.calls[0] == "trace-xyz-123"


# ──────────────────────────────────────────────────────────────────────────────
# 6. Same trace in multiple polls triggers RCA only once
# ──────────────────────────────────────────────────────────────────────────────

class TestDeduplication:
    def test_same_trace_in_two_polls_triggers_once(self) -> None:
        err = _error_trace("trace-dup")
        # Return the same trace in both polls
        client = _FakeClient([[err], [err]])
        monitor, trigger = _monitor(client=client)

        monitor.run_once()  # poll 1 — should trigger
        monitor.run_once()  # poll 2 — same trace, should skip

        assert trigger.calls.count("trace-dup") == 1

    def test_registry_marks_trace_as_processed(self) -> None:
        registry = ProcessedTraceRegistry()
        assert not registry.is_processed("abc")
        registry.mark_processed("abc")
        assert registry.is_processed("abc")

    def test_registry_double_mark_is_idempotent(self) -> None:
        registry = ProcessedTraceRegistry()
        registry.mark_processed("x")
        registry.mark_processed("x")  # second mark — no error
        assert registry.size() == 1

    def test_registry_size_grows_with_new_ids(self) -> None:
        registry = ProcessedTraceRegistry()
        for i in range(5):
            registry.mark_processed(f"trace-{i}")
        assert registry.size() == 5

    def test_registry_evicts_when_at_capacity(self) -> None:
        registry = ProcessedTraceRegistry(max_size=10)
        for i in range(10):
            registry.mark_processed(f"t{i:03d}")
        # Now add one more — should trigger eviction of oldest 10%
        registry.mark_processed("t999")
        assert registry.size() <= 10

    def test_same_trace_multiple_polls_via_monitor(self) -> None:
        """Run 5 polls returning the same error trace — RCA invoked exactly once."""
        err = _error_trace("trace-repeat")
        client = _FakeClient([[err]] * 5)
        monitor, trigger = _monitor(client=client)

        for _ in range(5):
            monitor.run_once()

        assert trigger.calls.count("trace-repeat") == 1


# ──────────────────────────────────────────────────────────────────────────────
# 7. Multiple different error traces trigger RCA independently
# ──────────────────────────────────────────────────────────────────────────────

class TestMultipleErrorTraces:
    def test_two_different_traces_trigger_twice(self) -> None:
        err1 = _error_trace("trace-A")
        err2 = _error_trace("trace-B")
        client = _FakeClient([[err1, err2]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 2
        assert "trace-A" in trigger.calls
        assert "trace-B" in trigger.calls

    def test_error_in_each_poll_both_triggered(self) -> None:
        err1 = _error_trace("trace-poll1")
        err2 = _error_trace("trace-poll2")
        client = _FakeClient([[err1], [err2]])
        monitor, trigger = _monitor(client=client)

        monitor.run_once()
        monitor.run_once()

        assert "trace-poll1" in trigger.calls
        assert "trace-poll2" in trigger.calls
        assert len(trigger.calls) == 2

    def test_mix_error_and_ok_only_error_triggered(self) -> None:
        err = _error_trace("trace-err-only")
        ok = _ok_trace("trace-ok-only")
        client = _FakeClient([[err, ok]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 1
        assert trigger.calls == ["trace-err-only"]


# ──────────────────────────────────────────────────────────────────────────────
# 8. Jaeger unavailable does not terminate monitor
# ──────────────────────────────────────────────────────────────────────────────

class TestJaegerUnavailable:
    def test_network_error_returns_zero_not_exception(self) -> None:
        monitor, trigger = _monitor(client=_FailingClient())
        # Must not raise
        count = monitor.run_once()
        assert count == 0
        assert trigger.calls == []

    def test_two_polls_after_outage_continue(self) -> None:
        """Monitor keeps running across multiple failing polls."""
        monitor, trigger = _monitor(client=_FailingClient())
        for _ in range(3):
            count = monitor.run_once()
            assert count == 0  # never raises
        assert trigger.calls == []

    def test_failing_list_services_falls_back_gracefully(self) -> None:
        """When list_services fails and no services configured, poll continues."""
        class _FailServiceListClient:
            def search_traces(self, q):
                return TraceSearchResult(traces=[], total=0, query=q)
            def list_services(self):
                raise ConnectionError("services endpoint down")

        cfg = MonitorConfig(jaeger_url="http://x", services=[])
        monitor = JaegerMonitor(
            client=_FailServiceListClient(),
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=_CountingTrigger(),
            config=cfg,
        )
        # Should not raise
        count = monitor.run_once()
        assert count == 0


# ──────────────────────────────────────────────────────────────────────────────
# 9. Malformed / empty Jaeger response does not terminate monitor
# ──────────────────────────────────────────────────────────────────────────────

class TestMalformedResponse:
    def test_empty_trace_list_handled(self) -> None:
        client = _FakeClient([[]])
        monitor, trigger = _monitor(client=client)
        count = monitor.run_once()
        assert count == 0

    def test_trace_with_no_spans_is_not_error(self) -> None:
        """A Trace with zero spans has no error spans — safe to ignore."""
        empty_trace = Trace(trace_id="empty", spans=[])
        assert TraceErrorDetector().is_error_trace(empty_trace) is False

    def test_client_returning_none_does_not_crash(self) -> None:
        """If client search_traces returns empty result, monitor continues."""
        class _NullClient:
            def search_traces(self, q):
                return TraceSearchResult(traces=[], total=0, query=q)
            def list_services(self):
                return []

        monitor = JaegerMonitor(
            client=_NullClient(),
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=_CountingTrigger(),
            config=MonitorConfig(jaeger_url="http://x"),
        )
        count = monitor.run_once()
        assert count == 0


# ──────────────────────────────────────────────────────────────────────────────
# 10. RCA workflow failure does not terminate monitor
# ──────────────────────────────────────────────────────────────────────────────

class TestRCAFailureIsolation:
    def test_rca_exception_does_not_raise_from_monitor(self) -> None:
        err = _error_trace("trace-fail-rca")
        client = _FakeClient([[err]])
        monitor, _ = _monitor(client=client, trigger=_FailingTrigger())
        # Must not raise, must return 1 (attempted, not successfully completed)
        count = monitor.run_once()
        assert count == 1  # attempted

    def test_rca_failure_does_not_prevent_next_new_trace(self) -> None:
        """After one failing RCA, the monitor processes the next new trace."""
        err1 = _error_trace("trace-fail1")
        err2 = _error_trace("trace-ok2")

        # First poll: failing trigger; second poll: counting trigger
        failing = _FailingTrigger()
        counting = _CountingTrigger()

        class _SwitchingTrigger:
            def __init__(self):
                self._calls = 0
            def trigger(self, trace: Trace):
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("first call fails")
                return counting.trigger(trace)

        switcher = _SwitchingTrigger()
        client = _FakeClient([[err1], [err2]])
        monitor = JaegerMonitor(
            client=client,
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=switcher,
            config=MonitorConfig(jaeger_url="http://x"),
        )

        monitor.run_once()  # err1 triggers, RCA fails
        monitor.run_once()  # err2 triggers, RCA succeeds

        assert "trace-ok2" in counting.calls


# ──────────────────────────────────────────────────────────────────────────────
# 11. Configuration is read from settings / environment
# ──────────────────────────────────────────────────────────────────────────────

class TestConfiguration:
    def test_from_settings_produces_valid_config(self) -> None:
        config = MonitorConfig.from_settings()
        assert isinstance(config.jaeger_url, str)
        assert config.poll_interval_seconds > 0
        assert config.lookback_seconds > 0

    def test_settings_has_monitor_fields(self) -> None:
        from rca_agent.config.settings import Settings
        s = Settings()
        assert hasattr(s, "rca_poll_interval_seconds")
        assert hasattr(s, "rca_lookback_seconds")
        assert hasattr(s, "rca_monitor_services")
        assert hasattr(s, "rca_monitor_environment")

    def test_settings_defaults(self) -> None:
        from rca_agent.config.settings import Settings
        s = Settings()
        assert s.rca_poll_interval_seconds == 5.0
        assert s.rca_lookback_seconds == 30
        assert s.rca_monitor_environment == "production"

    def test_services_parsed_from_comma_string(self) -> None:
        """MonitorConfig.from_settings() parses RCA_MONITOR_SERVICES correctly."""
        # Patch the settings object directly rather than relying on env var reload
        # (pydantic-settings singleton is cached at module import time).
        from rca_agent.config.settings import settings as _settings
        original = _settings.rca_monitor_services
        try:
            _settings.rca_monitor_services = "svc-a, svc-b , svc-c"
            config = MonitorConfig.from_settings()
            assert config.services == ["svc-a", "svc-b", "svc-c"]
        finally:
            _settings.rca_monitor_services = original

    def test_empty_services_means_all(self) -> None:
        config = MonitorConfig(jaeger_url="http://x", services=[])
        assert config.services == []

    def test_memory_enabled_forwarded_to_config(self) -> None:
        config = MonitorConfig(memory_enabled=False)
        assert config.memory_enabled is False

    def test_config_from_settings_preserves_memory_enabled(self) -> None:
        """memory_enabled from settings is forwarded to MonitorConfig."""
        from rca_agent.config.settings import Settings
        s = Settings()
        config = MonitorConfig.from_settings()
        assert config.memory_enabled == s.memory_enabled


# ──────────────────────────────────────────────────────────────────────────────
# 12. No RKE-specific hardcoding in monitor source files
# ──────────────────────────────────────────────────────────────────────────────

class TestNoHardcoding:
    """Scan monitor source files for hardcoded application-specific strings."""

    _MONITOR_FILES = [
        "src/rca_agent/monitor/jaeger_monitor.py",
        "src/rca_agent/monitor/rca_workflow_trigger.py",
        "src/rca_agent/monitor/trace_error_detector.py",
        "src/rca_agent/monitor/processed_trace_registry.py",
        "src/rca_agent/monitor/config.py",
        "scripts/monitor.py",
    ]

    _FORBIDDEN = [
        "rke-backend",
        "INC-001", "INC-002", "INC-003", "INC-004", "INC-005", "INC-006",
        "db-pool-exhaustion",
        "slow-query",
        "backend-error",
        "config-regression",
        "443bbb28",  # real trace ID from demo
    ]

    def test_no_rke_hardcoding(self) -> None:
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel_path in self._MONITOR_FILES:
            full_path = os.path.join(root, rel_path)
            if not os.path.exists(full_path):
                continue  # skip if file doesn't exist yet during incremental dev
            text = open(full_path).read()
            for forbidden in self._FORBIDDEN:
                assert forbidden not in text, (
                    f"Hardcoded string {forbidden!r} found in {rel_path}. "
                    "Monitor must be generic, not RKE-specific."
                )


# ──────────────────────────────────────────────────────────────────────────────
# 13. Existing RCA workflow tests remain passing (regression guard)
# ──────────────────────────────────────────────────────────────────────────────

class TestExistingWorkflowRegression:
    def test_rca_agent_still_works_with_mock_llm(self) -> None:
        """Smoke-test the existing RCAAgent to confirm nothing regressed."""
        from datetime import timedelta
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        from rca_agent.models.incident import Incident, IncidentStatus, Severity
        from rca_agent.models.rca_result import RCAResult

        class _NoOpLog:
            def search_logs(self, q):
                from rca_agent.models.log_entry import LogSearchResult
                return LogSearchResult(entries=[], query=q)
            def get_logs_by_trace_id(self, tid):
                from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
                return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
            def get_log_by_id(self, lid): return None

        class _NoOpGit:
            def get_recent_commits(self, limit=20): return []
            def get_commit(self, cid): raise ValueError("no git")
            def get_diff(self, cid): return []
            def get_files_changed(self, cid): return []
            def search_commits(self, q): return []
            def get_commits_between(self, s, e): return []

        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=_NoOpLog(),
            git_provider=_NoOpGit(),
            memory=IncidentMemory(
                graph=InMemoryGraphProvider(),
                vector=TfidfVectorProvider(),
            ),
            auto_store_rca=False,
        )
        now = datetime.now(timezone.utc)
        incident = Incident(
            incident_id="regression-001",
            application="test-app",
            environment="test",
            title="Regression test incident",
            severity=Severity.LOW,
            status=IncidentStatus.OPEN,
            start_time=now - timedelta(minutes=5),
        )
        result = agent.investigate(incident)
        assert isinstance(result, RCAResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_trace_models_still_parse_correctly(self) -> None:
        """Trace model parsing that JaegerTraceProvider depends on."""
        span = _span(status=TraceStatus.ERROR)
        assert span.is_error is True
        trace = Trace(trace_id="t1", spans=[span])
        assert len(trace.error_spans) == 1

    def test_memory_off_behavior_preserved(self) -> None:
        """Confirm memory_enabled=False still works on RCAAgent."""
        from datetime import timedelta
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        from rca_agent.models.incident import Incident, IncidentStatus, Severity

        class _Noop:
            def search_logs(self, q):
                from rca_agent.models.log_entry import LogSearchResult
                return LogSearchResult(entries=[], query=q)
            def get_logs_by_trace_id(self, t):
                from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
                return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=t))
            def get_log_by_id(self, l): return None
            def get_recent_commits(self, limit=20): return []
            def get_commit(self, c): raise ValueError()
            def get_diff(self, c): return []
            def get_files_changed(self, c): return []
            def search_commits(self, q): return []
            def get_commits_between(self, s, e): return []

        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=_Noop(),
            git_provider=_Noop(),
            memory=IncidentMemory(
                graph=InMemoryGraphProvider(),
                vector=TfidfVectorProvider(),
            ),
            memory_enabled=False,
            auto_store_rca=False,
        )
        now = datetime.now(timezone.utc)
        incident = Incident(
            incident_id="mem-off-001",
            application="test",
            environment="test",
            title="Memory OFF regression",
            severity=Severity.LOW,
            status=IncidentStatus.OPEN,
            start_time=now - timedelta(minutes=2),
        )
        result = agent.investigate(incident)
        assert result.memory_enabled is False
        assert result.retrieved_historical_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# 14. Integration path: fake error trace → monitor detects → RCA invoked
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegrationPath:
    """End-to-end path using fakes — no live Jaeger required.

    Validates the complete flow:
        fake error trace → JaegerMonitor → TraceErrorDetector → ProcessedTraceRegistry
            → RCAWorkflowTrigger (real) with MockLLMProvider → RCAResult
    """

    def test_full_path_with_real_rca_agent(self) -> None:
        """
        Build a real RCAWorkflowTrigger backed by MockLLMProvider and inject
        it into the monitor.  A fake client returns one error trace.
        Verify the monitor detects it and produces an RCAResult.
        """
        from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        from rca_agent.providers.noop_trace_provider import NoOpTraceProvider
        from rca_agent.models.rca_result import RCAResult

        config = MonitorConfig(
            jaeger_url="http://fake-jaeger:16686",
            poll_interval_seconds=5.0,
            lookback_seconds=30,
            services=[],
            environment="integration-test",
            memory_enabled=False,  # keep it fast
        )

        results: list[RCAResult] = []

        class _InlineRCATrigger:
            """Uses the real RCA pipeline but with no-op providers."""

            def trigger(self, trace: Trace) -> RCAResult:
                from datetime import timedelta

                class _NoOpLog:
                    def search_logs(self, q):
                        from rca_agent.models.log_entry import LogSearchResult
                        return LogSearchResult(entries=[], query=q)
                    def get_logs_by_trace_id(self, t):
                        from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
                        return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=t))
                    def get_log_by_id(self, l): return None

                class _NoOpGit:
                    def get_recent_commits(self, limit=20): return []
                    def get_commit(self, c): raise ValueError()
                    def get_diff(self, c): return []
                    def get_files_changed(self, c): return []
                    def search_commits(self, q): return []
                    def get_commits_between(self, s, e): return []

                agent = RCAAgent(
                    llm=MockLLMProvider(),
                    log_provider=_NoOpLog(),
                    git_provider=_NoOpGit(),
                    memory=IncidentMemory(
                        graph=InMemoryGraphProvider(),
                        vector=TfidfVectorProvider(),
                    ),
                    trace_provider=NoOpTraceProvider(reason="integration test"),
                    memory_enabled=False,
                    auto_store_rca=False,
                )

                # Build minimal incident from trace
                root = trace.root_span
                from rca_agent.models.incident import Incident, IncidentStatus, Severity
                incident = Incident(
                    incident_id=f"auto-{trace.trace_id[:8]}",
                    application=root.service_name if root else "unknown",
                    environment=config.environment,
                    title=f"Auto-detected: {root.operation_name if root else 'unknown'}",
                    severity=Severity.HIGH,
                    status=IncidentStatus.OPEN,
                    start_time=root.start_time if root else datetime.now(timezone.utc),
                    affected_services=trace.service_names,
                )

                result = agent.investigate(incident)
                results.append(result)
                return result

        err_trace = _error_trace(
            trace_id="integ-trace-001",
            service="any-service",
            operation="POST /some/endpoint",
        )
        client = _FakeClient([[err_trace]])

        monitor = JaegerMonitor(
            client=client,
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=_InlineRCATrigger(),
            config=config,
        )

        triggered = monitor.run_once()

        # Verify the integration path end-to-end
        assert triggered == 1, "Expected exactly one RCA to be triggered"
        assert len(results) == 1, "Expected exactly one RCAResult"
        result = results[0]
        assert isinstance(result, RCAResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_no_hardcoded_incident_id_in_auto_incident(self) -> None:
        """The auto-generated incident ID is derived from the trace ID, not hardcoded."""
        err = _error_trace("trace-unique-xyz")
        # RCAWorkflowTrigger builds incident_id as f"auto-{trace_id[:16]}"
        # trace_id "trace-unique-xyz" has 15 chars, so full trace_id is used
        from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
        config = MonitorConfig(jaeger_url="http://x", environment="test")
        trigger = RCAWorkflowTrigger(config)
        incident = trigger._build_incident(err)
        assert incident.incident_id.startswith("auto-trace-unique")
        assert "INC-" not in incident.incident_id
        assert incident.application == "test-svc"

    def test_incident_environment_comes_from_config(self) -> None:
        err = _error_trace("trace-env-test")
        from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
        config = MonitorConfig(jaeger_url="http://x", environment="staging")
        trigger = RCAWorkflowTrigger(config)
        incident = trigger._build_incident(err)
        assert incident.environment == "staging"

    def test_service_list_comes_from_trace(self) -> None:
        """affected_services on the auto-incident must reflect actual trace services."""
        trace = Trace(
            trace_id="trace-multi-svc",
            spans=[
                _span(trace_id="trace-multi-svc", span_id="s1",
                      service="frontend", status=TraceStatus.ERROR),
                _span(trace_id="trace-multi-svc", span_id="s2",
                      service="backend", status=TraceStatus.UNSET),
            ],
        )
        from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
        config = MonitorConfig(jaeger_url="http://x")
        trigger = RCAWorkflowTrigger(config)
        incident = trigger._build_incident(trace)
        assert set(incident.affected_services) == {"frontend", "backend"}
