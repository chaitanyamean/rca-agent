"""Tests for the Node 2 log-retrieval strategy fix.

When an exact primary trace_id exists AND trace-correlated logs are found,
the broad general time-window search must be skipped entirely.

Acceptance criteria:
1. primary_trace_id exists + correlated logs > 0  → search_logs() NOT called.
2. primary_trace_id exists + correlated logs = 0  → search_logs() IS called.
3. primary_trace_id absent                        → search_logs() IS called (unchanged).
4. Correlated logs are present in the final raw_logs state.
5. Unrelated logs NOT added when correlated logs exist.
6. Existing trace retrieval tests pass (regression).
7. Existing autonomous monitor tests pass (regression).
8. Existing Phase 1–4 tests pass (regression).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.agents.rca_graph import build_rca_graph
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.incident import Incident, IncidentStatus, Severity
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.rca_result import RCAResult
from rca_agent.models.trace_models import (
    Span, Trace, TraceSearchQuery, TraceSearchResult, TraceStatus,
)
from rca_agent.providers.noop_trace_provider import NoOpTraceProvider

_NOW = datetime(2026, 9, 20, 17, 0, 0, tzinfo=timezone.utc)
_TRACE_ID = "0489378683549a43ccfaf9ad28e384ac"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _SpyLogProvider:
    """Log provider that records every call to search_logs / get_logs_by_trace_id."""

    def __init__(
        self,
        correlated: list[LogEntry] | None = None,
        general_error: list[LogEntry] | None = None,
        general_warn: list[LogEntry] | None = None,
    ) -> None:
        self._correlated = correlated or []
        self._general_error = general_error or []
        self._general_warn = general_warn or []
        self.search_calls: list[LogSearchQuery] = []
        self.trace_calls: list[str] = []

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        self.search_calls.append(query)
        entries = (
            self._general_error
            if (query.level or "").upper() == "ERROR"
            else self._general_warn
        )
        return LogSearchResult(entries=entries, query=query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        self.trace_calls.append(trace_id)
        return LogSearchResult(
            entries=self._correlated,
            query=LogSearchQuery(trace_id=trace_id),
        )

    def get_log_by_id(self, lid: str): return None


def _make_log(trace_id: str, level: str = "ERROR", message: str = "test") -> LogEntry:
    return LogEntry.from_raw_line(json.dumps({
        "timestamp": _NOW.isoformat(),
        "level": level,
        "service": "rke-backend",
        "message": message,
        "traceId": trace_id,
    }))


def _unrelated_log(i: int = 0) -> LogEntry:
    """A log entry with a DIFFERENT trace ID — simulates noise from another incident."""
    return _make_log(f"unrelated-trace-{i:04d}", message=f"unrelated log {i}")


def _correlated_log(trace_id: str = _TRACE_ID, i: int = 0) -> LogEntry:
    return _make_log(trace_id, level="ERROR", message=f"cascade failure step {i}")


class _FakeTraceProvider:
    def __init__(self, trace: Trace | None = None):
        self._trace = trace

    def get_trace(self, tid: str) -> Trace | None:
        if self._trace and self._trace.trace_id == tid:
            return self._trace
        return None

    def search_traces(self, q: TraceSearchQuery) -> TraceSearchResult:
        ts = [self._trace] if self._trace else []
        return TraceSearchResult(traces=ts, total=len(ts), query=q)

    def get_trace_spans(self, tid): return []
    def get_failed_spans(self, tid): return []


class _NoOpGit:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, c): raise ValueError()
    def get_diff(self, c): return []
    def get_files_changed(self, c): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


def _error_trace(trace_id: str = _TRACE_ID) -> Trace:
    span = Span(
        trace_id=trace_id, span_id="s001", service_name="rke-backend",
        operation_name="POST /api/test/incidents/cascade",
        start_time=_NOW, end_time=_NOW, duration_ms=1500.0,
        status=TraceStatus.ERROR, status_message="HTTP 500",
        attributes={"client.address": "192.168.65.1", "error": True},
    )
    return Trace(trace_id=trace_id, spans=[span])


def _incident(trace_id: str | None = _TRACE_ID) -> Incident:
    return Incident(
        incident_id=f"auto-{(trace_id or 'noid')[:8]}",
        application="rke-backend",
        environment="test",
        title="CASCADE incident",
        severity=Severity.HIGH,
        status=IncidentStatus.OPEN,
        start_time=_NOW - timedelta(minutes=1),
        trace_id=trace_id,
    )


def _run_node2(
    log_provider: _SpyLogProvider,
    trace_provider=None,
    incident: Incident | None = None,
) -> dict:
    """Invoke the full graph and return the final LangGraph state."""
    graph = build_rca_graph(
        llm=MockLLMProvider(),
        log_provider=log_provider,
        git_provider=_NoOpGit(),
        memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
        trace_provider=trace_provider,
        memory_enabled=False,
    )
    inc = incident or _incident()
    return graph.invoke({"incident": inc, "memory_enabled": False})


# ===========================================================================
# 1. primary_trace_id + correlated > 0 → search_logs NOT called
# ===========================================================================

class TestCorrelatedLogsSkipsGeneralSearch:
    def test_search_logs_not_called_when_correlated_found(self) -> None:
        """When correlated logs exist, search_logs() must not be called."""
        logs = [_correlated_log(i=i) for i in range(8)]
        spy = _SpyLogProvider(correlated=logs)
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        _run_node2(spy, tp, _incident(_TRACE_ID))

        assert spy.search_calls == [], (
            f"search_logs() was called {len(spy.search_calls)} times but should not be "
            "called when 8 trace-correlated logs are available"
        )

    def test_trace_call_made_for_correct_trace_id(self) -> None:
        """get_logs_by_trace_id must be called with the incident's trace_id."""
        logs = [_correlated_log()]
        spy = _SpyLogProvider(correlated=logs)
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        _run_node2(spy, tp, _incident(_TRACE_ID))

        assert _TRACE_ID in spy.trace_calls

    def test_final_raw_logs_contains_only_correlated(self) -> None:
        """raw_logs in the state must contain only the 8 correlated entries."""
        corr_logs = [_correlated_log(i=i) for i in range(8)]
        unrelated = [_unrelated_log(i) for i in range(5)]
        spy = _SpyLogProvider(
            correlated=corr_logs,
            general_error=unrelated,   # these must NOT appear
        )
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        state = _run_node2(spy, tp, _incident(_TRACE_ID))
        raw_logs = state.get("raw_logs", [])

        # All returned logs must carry the primary trace_id
        for entry in raw_logs:
            assert entry.trace_id == _TRACE_ID, (
                f"Unrelated log (trace_id={entry.trace_id!r}) appeared in raw_logs"
            )

        assert len(raw_logs) == 8

    def test_unrelated_logs_not_added(self) -> None:
        """When correlated logs exist, no unrelated logs must be included."""
        corr_logs = [_correlated_log(i=i) for i in range(3)]
        noise_logs = [_unrelated_log(i) for i in range(20)]
        spy = _SpyLogProvider(
            correlated=corr_logs,
            general_error=noise_logs[:10],
            general_warn=noise_logs[10:],
        )
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        state = _run_node2(spy, tp, _incident(_TRACE_ID))
        raw_logs = state.get("raw_logs", [])

        unrelated_found = [e for e in raw_logs if e.trace_id != _TRACE_ID]
        assert unrelated_found == [], (
            f"{len(unrelated_found)} unrelated log(s) slipped into raw_logs"
        )


# ===========================================================================
# 2. primary_trace_id + correlated = 0 → search_logs IS called as fallback
# ===========================================================================

class TestFallbackWhenNoCorrelatedLogs:
    def test_search_logs_called_when_correlated_empty(self) -> None:
        """When get_logs_by_trace_id returns 0, search_logs() must run."""
        spy = _SpyLogProvider(correlated=[])  # empty correlated
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        _run_node2(spy, tp, _incident(_TRACE_ID))

        assert len(spy.search_calls) >= 1, (
            "search_logs() should run as fallback when correlated logs are empty"
        )

    def test_search_logs_called_when_trace_not_found(self) -> None:
        """When exact trace is not in Jaeger, no correlated logs → fallback."""
        spy = _SpyLogProvider(correlated=[])
        tp = _FakeTraceProvider(trace=None)  # trace not found

        _run_node2(spy, tp, _incident(_TRACE_ID))

        assert len(spy.search_calls) >= 1


# ===========================================================================
# 3. primary_trace_id absent → search_logs unchanged
# ===========================================================================

class TestNoTraceIdUsesGeneralSearch:
    def test_search_logs_called_when_no_trace_id(self) -> None:
        """Without a trace_id, the original general search runs."""
        spy = _SpyLogProvider()
        inc = _incident(trace_id=None)

        _run_node2(spy, trace_provider=None, incident=inc)

        assert len(spy.search_calls) >= 1, (
            "search_logs() must run when incident has no trace_id"
        )

    def test_get_logs_by_trace_id_not_called_without_trace_id(self) -> None:
        spy = _SpyLogProvider()
        inc = _incident(trace_id=None)

        _run_node2(spy, trace_provider=None, incident=inc)

        assert spy.trace_calls == []

    def test_no_trace_provider_still_runs_general_search(self) -> None:
        """Manual API usage without Jaeger must still fall back to general search."""
        spy = _SpyLogProvider(general_error=[_make_log("x", "ERROR", "manual")])
        inc = _incident(trace_id=None)

        state = _run_node2(spy, trace_provider=NoOpTraceProvider(reason="disabled"), incident=inc)

        assert len(spy.search_calls) >= 1


# ===========================================================================
# 4. Correlated logs survive to final state
# ===========================================================================

class TestCorrelatedLogsInState:
    def test_correlated_log_messages_in_raw_logs(self) -> None:
        """The message text from correlated logs must reach raw_logs."""
        corr_logs = [
            _make_log(_TRACE_ID, "ERROR", "RatingEngine connection timeout"),
            _make_log(_TRACE_ID, "ERROR", "PricingService downstream failure"),
        ]
        spy = _SpyLogProvider(correlated=corr_logs)
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        state = _run_node2(spy, tp, _incident(_TRACE_ID))
        messages = [e.message for e in state.get("raw_logs", [])]

        assert "RatingEngine connection timeout" in messages
        assert "PricingService downstream failure" in messages

    def test_eight_correlated_logs_all_reach_state(self) -> None:
        """All 8 cascade logs must appear in raw_logs."""
        messages = [
            "=== INCIDENT TRIGGER: INC-005 — cascade ===",
            "[IncidentController] Dispatching order to PricingService",
            "[PricingService] Requesting rate from RatingEngine",
            "[RatingEngine] Connecting to external rate feed",
            "[RatingEngine] Connection to external rate feed timed out after 500 ms",
            "[PricingService] Downstream failure from RatingEngine",
            "[IncidentController] Upstream failure: PricingService unavailable",
            "Simulation failure triggered: incidentId=INC-005",
        ]
        corr_logs = [_make_log(_TRACE_ID, "ERROR" if i >= 4 else "WARN", msg)
                     for i, msg in enumerate(messages)]
        spy = _SpyLogProvider(correlated=corr_logs)
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        state = _run_node2(spy, tp, _incident(_TRACE_ID))
        raw_msgs = [e.message for e in state.get("raw_logs", [])]

        for msg in messages:
            assert msg in raw_msgs, f"Expected log message not in raw_logs: {msg!r}"


# ===========================================================================
# 5. Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_correlated_log_provider_exception_falls_back_to_general(self) -> None:
        """If get_logs_by_trace_id raises, general search must still run."""
        class _FailingLog(_SpyLogProvider):
            def get_logs_by_trace_id(self, tid):
                raise ConnectionError("log server down")

        spy = _FailingLog(general_error=[_unrelated_log()])
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        # Must not crash; must fall back to general search
        state = _run_node2(spy, tp, _incident(_TRACE_ID))
        assert isinstance(state.get("rca_result"), RCAResult)
        assert len(spy.search_calls) >= 1

    def test_zero_total_logs_produces_valid_rca_result(self) -> None:
        """Even with no logs at all, the workflow must complete."""
        spy = _SpyLogProvider(correlated=[], general_error=[], general_warn=[])
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        state = _run_node2(spy, tp, _incident(_TRACE_ID))
        assert isinstance(state.get("rca_result"), RCAResult)


# ===========================================================================
# 6. Regression: full investigation with correlated logs produces a result
# ===========================================================================

class TestRegressionCascadeInvestigation:
    def test_cascade_incident_completes_with_correlated_logs(self) -> None:
        """Full RCA agent run with 8 cascade logs should not crash."""
        cascade_logs = [
            _make_log(_TRACE_ID, "WARN", "=== INCIDENT TRIGGER: INC-005 — cascade ==="),
            _make_log(_TRACE_ID, "ERROR", "[RatingEngine] Connection timeout after 500ms"),
            _make_log(_TRACE_ID, "ERROR", "[PricingService] Downstream failure from RatingEngine"),
            _make_log(_TRACE_ID, "ERROR", "[IncidentController] Upstream failure: PricingService"),
        ]
        spy = _SpyLogProvider(correlated=cascade_logs)
        tp = _FakeTraceProvider(_error_trace(_TRACE_ID))

        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=spy,
            git_provider=_NoOpGit(),
            memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
            trace_provider=tp,
            memory_enabled=False,
            auto_store_rca=False,
        )
        result = agent.investigate(_incident(_TRACE_ID))

        assert isinstance(result, RCAResult)
        assert 0.0 <= result.confidence <= 1.0
        # General search must not have been called
        assert spy.search_calls == []
        # All 4 cascade logs must be in raw state (checked via trace_calls made)
        assert _TRACE_ID in spy.trace_calls

    def test_backward_compat_no_trace_id_incident(self) -> None:
        """Incidents without trace_id (manual API usage) still work."""
        spy = _SpyLogProvider(general_error=[_unrelated_log(0)])
        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=spy,
            git_provider=_NoOpGit(),
            memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
            memory_enabled=False,
            auto_store_rca=False,
        )
        result = agent.investigate(_incident(trace_id=None))
        assert isinstance(result, RCAResult)
        assert len(spy.search_calls) >= 1
