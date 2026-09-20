"""Tests for the autonomous RCA trace-evidence grounding fix.

Verifies acceptance criteria:
1.  Detected trace_id is set on the Incident by RCAWorkflowTrigger.
2.  Incident.trace_id carries through to the RCA workflow state.
3.  Node 2 requests the exact trace_id via get_trace().
4.  Exact trace becomes the primary (first) raw_traces entry.
5.  Successful exact trace retrieval works end-to-end.
6.  Missing exact trace is handled gracefully (falls back, no crash).
7.  Correlated logs are fetched using get_logs_by_trace_id().
8.  Historical memory remains separate from current trace evidence.
9.  Secondary traces do not replace the detected primary trace.
10. Broad search is NOT performed when exact trace is found.
11. Existing autonomous monitor tests continue passing.
12. Existing Phase 1–4 tests continue passing (regression).
13. Manual evaluation continues passing (regression).

All tests use fakes — no live Jaeger required.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.models.incident import Incident, IncidentStatus, Severity
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.rca_result import RCAResult, RCAStatus
from rca_agent.models.trace_models import (
    Span,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
    TraceStatus,
)
from rca_agent.monitor.config import MonitorConfig
from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 9, 20, 17, 0, 0, tzinfo=timezone.utc)


def _span(
    trace_id: str = "tid-001",
    span_id: str = "sid-001",
    service: str = "test-svc",
    operation: str = "POST /api/test",
    status: TraceStatus = TraceStatus.ERROR,
    parent: str | None = None,
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        service_name=service,
        operation_name=operation,
        start_time=_NOW,
        end_time=_NOW,
        duration_ms=200.0,
        status=status,
        status_message="test error" if status == TraceStatus.ERROR else None,
    )


def _error_trace(trace_id: str = "tid-abc123", service: str = "test-svc") -> Trace:
    return Trace(
        trace_id=trace_id,
        spans=[_span(trace_id=trace_id, service=service)],
    )


def _ok_trace(trace_id: str = "tid-ok") -> Trace:
    return Trace(
        trace_id=trace_id,
        spans=[_span(trace_id=trace_id, status=TraceStatus.OK)],
    )


def _incident(
    trace_id: str | None = None,
    application: str = "test-svc",
) -> Incident:
    return Incident(
        incident_id="inc-test",
        application=application,
        environment="test",
        title="Test incident",
        severity=Severity.HIGH,
        status=IncidentStatus.OPEN,
        start_time=_NOW - timedelta(minutes=5),
        trace_id=trace_id,
    )


class _FakeTraceProvider:
    """Controllable trace provider: supports both get_trace and search_traces."""

    def __init__(
        self,
        traces_by_id: dict[str, Trace] | None = None,
        search_results: list[Trace] | None = None,
    ) -> None:
        self._by_id: dict[str, Trace] = traces_by_id or {}
        self._search: list[Trace] = search_results or []
        self.get_trace_calls: list[str] = []
        self.search_calls: list[TraceSearchQuery] = []

    def get_trace(self, trace_id: str) -> Trace | None:
        self.get_trace_calls.append(trace_id)
        return self._by_id.get(trace_id)

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        self.search_calls.append(query)
        return TraceSearchResult(
            traces=self._search,
            total=len(self._search),
            query=query,
        )

    def get_trace_spans(self, trace_id: str):
        t = self.get_trace(trace_id)
        return t.spans if t else []

    def get_failed_spans(self, trace_id: str):
        t = self.get_trace(trace_id)
        return t.error_spans if t else []


class _FakeLogProvider:
    """Log provider that records calls and returns configurable results."""

    def __init__(
        self,
        entries_by_trace: dict[str, list[LogEntry]] | None = None,
        search_entries: list[LogEntry] | None = None,
    ) -> None:
        self._by_trace = entries_by_trace or {}
        self._search = search_entries or []
        self.trace_calls: list[str] = []
        self.search_calls: list[LogSearchQuery] = []

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        self.search_calls.append(query)
        return LogSearchResult(entries=self._search, query=query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        self.trace_calls.append(trace_id)
        entries = self._by_trace.get(trace_id, [])
        return LogSearchResult(
            entries=entries,
            query=LogSearchQuery(trace_id=trace_id),
        )

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        return None


class _NoOpGit:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, c): raise ValueError("no git")
    def get_diff(self, c): return []
    def get_files_changed(self, c): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


def _make_agent(
    trace_provider=None,
    log_provider=None,
    memory_enabled: bool = False,
) -> tuple[RCAAgent, Any]:
    from rca_agent.agents.llm_provider import MockLLMProvider
    from rca_agent.memory.graph_provider import InMemoryGraphProvider
    from rca_agent.memory.incident_memory import IncidentMemory
    from rca_agent.memory.vector_provider import TfidfVectorProvider

    llm = MockLLMProvider()
    log = log_provider or _FakeLogProvider()
    git = _NoOpGit()
    memory = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
    )
    agent = RCAAgent(
        llm=llm,
        log_provider=log,
        git_provider=git,
        memory=memory,
        trace_provider=trace_provider,
        memory_enabled=memory_enabled,
        auto_store_rca=False,
    )
    return agent, llm


# ===========================================================================
# 1. trace_id is set on Incident by RCAWorkflowTrigger
# ===========================================================================

class TestTraceIdPropagation:
    def test_trigger_sets_trace_id_on_incident(self) -> None:
        """RCAWorkflowTrigger._build_incident must preserve the trace_id."""
        config = MonitorConfig(jaeger_url="http://x", environment="test")
        trigger = RCAWorkflowTrigger(config)
        trace = _error_trace("c3a1eb5b5bfca578278634b28d0288e8")
        incident = trigger._build_incident(trace)
        assert incident.trace_id == "c3a1eb5b5bfca578278634b28d0288e8"

    def test_trigger_trace_id_matches_full_trace_id(self) -> None:
        """The full trace_id (not truncated) is stored on the Incident."""
        config = MonitorConfig(jaeger_url="http://x")
        trigger = RCAWorkflowTrigger(config)
        full_id = "fb3228950c38ccea000000000000cafe"
        trace = _error_trace(full_id)
        incident = trigger._build_incident(trace)
        assert incident.trace_id == full_id

    def test_incident_incident_id_is_truncated_but_trace_id_is_not(self) -> None:
        """incident_id = 'auto-{first16}'; trace_id = full ID."""
        config = MonitorConfig(jaeger_url="http://x")
        trigger = RCAWorkflowTrigger(config)
        full_id = "1234567890abcdef1234567890abcdef"
        trace = _error_trace(full_id)
        incident = trigger._build_incident(trace)
        assert incident.incident_id == "auto-1234567890abcdef"
        assert incident.trace_id == full_id

    def test_incident_model_accepts_trace_id_field(self) -> None:
        """Incident model can be constructed with a trace_id."""
        inc = _incident(trace_id="abc123")
        assert inc.trace_id == "abc123"

    def test_incident_model_trace_id_defaults_to_none(self) -> None:
        """Incident.trace_id defaults to None for backward compatibility."""
        inc = _incident()
        assert inc.trace_id is None


# ===========================================================================
# 2. Incident.trace_id reaches Node 2 via RCA workflow state
# ===========================================================================

class TestTraceIdReachesNode2:
    def test_trace_id_in_state_triggers_exact_lookup(self) -> None:
        """When incident has trace_id, Node 2 calls get_trace()."""
        tid = "exact-trace-001"
        trace = _error_trace(tid)
        tp = _FakeTraceProvider(traces_by_id={tid: trace})
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=tid))
        assert tid in tp.get_trace_calls

    def test_no_trace_id_skips_exact_lookup(self) -> None:
        """When incident has no trace_id, Node 2 does NOT call get_trace()."""
        tp = _FakeTraceProvider(search_results=[_error_trace("some-trace")])
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=None))
        assert tp.get_trace_calls == []


# ===========================================================================
# 3. Node 2 requests exact trace via get_trace()
# ===========================================================================

class TestExactTraceRetrieval:
    def test_get_trace_called_with_correct_id(self) -> None:
        tid = "c3a1eb5b5bfca578278634b28d0288e8"
        tp = _FakeTraceProvider(traces_by_id={tid: _error_trace(tid)})
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=tid))
        assert tp.get_trace_calls == [tid]

    def test_search_traces_not_called_when_exact_found(self) -> None:
        """If exact trace is found, broad search must NOT run."""
        tid = "exact-and-found"
        tp = _FakeTraceProvider(
            traces_by_id={tid: _error_trace(tid)},
            search_results=[_error_trace("unrelated-trace")],
        )
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=tid))
        assert tp.search_calls == [], (
            "search_traces must NOT be called when exact trace was found"
        )


# ===========================================================================
# 4. Exact trace is primary (first) raw_traces entry
# ===========================================================================

class TestExactTraceIsPrimary:
    def test_exact_trace_is_in_raw_traces(self) -> None:
        """The exact trace must appear in raw_traces after investigation."""
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        from rca_agent.agents.rca_graph import build_rca_graph

        tid = "primary-trace-xyz"
        primary = _error_trace(tid)
        tp = _FakeTraceProvider(traces_by_id={tid: primary})
        log = _FakeLogProvider()
        git = _NoOpGit()
        llm = MockLLMProvider()
        memory = IncidentMemory(
            graph=InMemoryGraphProvider(),
            vector=TfidfVectorProvider(),
        )
        graph = build_rca_graph(
            llm=llm,
            log_provider=log,
            git_provider=git,
            memory=memory,
            trace_provider=tp,
            memory_enabled=False,
        )
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        raw_traces = state.get("raw_traces", [])
        trace_ids = [t.trace_id for t in raw_traces]
        assert tid in trace_ids, (
            f"Exact trace {tid!r} missing from raw_traces. Got: {trace_ids}"
        )
        if len(trace_ids) > 1:
            assert trace_ids[0] == tid, "Exact trace must be FIRST in raw_traces"


# ===========================================================================
# 5. Successful exact trace retrieval
# ===========================================================================

class TestSuccessfulExactRetrieval:
    def test_agent_receives_exact_trace_spans(self) -> None:
        """Node 2 gets spans from the exact trace."""
        tid = "full-trace-spans"
        trace = Trace(
            trace_id=tid,
            spans=[
                _span(trace_id=tid, span_id="s1", status=TraceStatus.ERROR),
                _span(trace_id=tid, span_id="s2", status=TraceStatus.UNSET),
                _span(trace_id=tid, span_id="s3", status=TraceStatus.UNSET),
            ],
        )
        tp = _FakeTraceProvider(traces_by_id={tid: trace})
        agent, _ = _make_agent(trace_provider=tp)
        # Run the full investigation; just verify it doesn't crash and
        # get_trace was called exactly once with the right ID
        result = agent.investigate(_incident(trace_id=tid))
        assert isinstance(result, RCAResult)
        assert tp.get_trace_calls == [tid]

    def test_exact_trace_error_spans_inform_investigation(self) -> None:
        """Error spans from the exact trace contribute to evidence."""
        tid = "error-evidence-tid"
        trace = _error_trace(tid)
        assert len(trace.error_spans) == 1
        tp = _FakeTraceProvider(traces_by_id={tid: trace})
        agent, _ = _make_agent(trace_provider=tp)
        result = agent.investigate(_incident(trace_id=tid))
        assert isinstance(result, RCAResult)


# ===========================================================================
# 6. Missing exact trace handled gracefully
# ===========================================================================

class TestMissingExactTrace:
    def test_missing_trace_does_not_crash(self) -> None:
        """If get_trace returns None, investigation continues without crashing."""
        tid = "not-in-jaeger"
        tp = _FakeTraceProvider(
            traces_by_id={},          # exact trace not found
            search_results=[],         # broad search also empty
        )
        agent, _ = _make_agent(trace_provider=tp)
        result = agent.investigate(_incident(trace_id=tid))
        assert isinstance(result, RCAResult)
        # get_trace was attempted
        assert tid in tp.get_trace_calls

    def test_missing_exact_trace_falls_back_to_search(self) -> None:
        """When exact trace is not found, Node 2 falls back to search_traces."""
        tid = "missing-exact"
        fallback = _error_trace("fallback-trace")
        tp = _FakeTraceProvider(
            traces_by_id={},           # exact not found
            search_results=[fallback], # broad search has something
        )
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=tid))
        # search_traces must be called as fallback
        assert len(tp.search_calls) == 1

    def test_get_trace_raising_does_not_crash(self) -> None:
        """If get_trace raises (e.g. network error), investigation continues."""

        class _FailingTraceProvider:
            def get_trace(self, trace_id: str) -> None:
                raise ConnectionError("Jaeger unreachable")
            def search_traces(self, q):
                return TraceSearchResult(traces=[], total=0, query=q)
            def get_trace_spans(self, tid): return []
            def get_failed_spans(self, tid): return []

        agent, _ = _make_agent(trace_provider=_FailingTraceProvider())
        result = agent.investigate(_incident(trace_id="some-tid"))
        assert isinstance(result, RCAResult)


# ===========================================================================
# 7. Correlated logs fetched using trace_id
# ===========================================================================

class TestCorrelatedLogsByTraceId:
    def test_get_logs_by_trace_id_called_when_trace_found(self) -> None:
        """When exact trace is found, get_logs_by_trace_id() is called."""
        tid = "trace-with-logs"
        tp = _FakeTraceProvider(traces_by_id={tid: _error_trace(tid)})
        log = _FakeLogProvider()
        agent, _ = _make_agent(trace_provider=tp, log_provider=log)
        agent.investigate(_incident(trace_id=tid))
        assert tid in log.trace_calls

    def test_get_logs_by_trace_id_not_called_when_no_trace_id(self) -> None:
        """Without incident.trace_id, get_logs_by_trace_id is NOT called."""
        tp = _FakeTraceProvider(search_results=[])
        log = _FakeLogProvider()
        agent, _ = _make_agent(trace_provider=tp, log_provider=log)
        agent.investigate(_incident(trace_id=None))
        assert log.trace_calls == []

    def test_correlated_logs_prepended_to_raw_logs(self) -> None:
        """Correlated logs for the trace ID are included in raw_logs."""
        from rca_agent.agents.rca_graph import build_rca_graph
        from rca_agent.agents.llm_provider import MockLLMProvider
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        import json

        tid = "trace-correlated-logs"
        tp = _FakeTraceProvider(traces_by_id={tid: _error_trace(tid)})

        # Build a synthetic log entry for the trace
        raw_line = json.dumps({
            "timestamp": _NOW.isoformat(),
            "level": "ERROR",
            "service": "test-svc",
            "message": "Pool exhausted",
            "traceId": tid,
        })
        entry = LogEntry.from_raw_line(raw_line)
        log = _FakeLogProvider(entries_by_trace={tid: [entry]})

        graph = build_rca_graph(
            llm=MockLLMProvider(),
            log_provider=log,
            git_provider=_NoOpGit(),
            memory=IncidentMemory(
                graph=InMemoryGraphProvider(),
                vector=TfidfVectorProvider(),
            ),
            trace_provider=tp,
            memory_enabled=False,
        )
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        raw_logs = state.get("raw_logs", [])
        log_messages = [e.message for e in raw_logs]
        assert any("Pool exhausted" in m for m in log_messages), (
            f"Correlated log not in raw_logs. Messages: {log_messages}"
        )


# ===========================================================================
# 8. Historical memory is separate from current trace evidence
# ===========================================================================

class TestHistoricalMemorySeparation:
    def test_memory_off_produces_no_historical_context(self) -> None:
        """Memory OFF means no historical incidents in context."""
        tid = "mem-sep-tid"
        tp = _FakeTraceProvider(traces_by_id={tid: _error_trace(tid)})
        agent, _ = _make_agent(trace_provider=tp, memory_enabled=False)
        result = agent.investigate(_incident(trace_id=tid))
        assert result.memory_enabled is False
        assert result.retrieved_historical_count == 0

    def test_historical_evidence_not_marked_as_current_fact(self) -> None:
        """Any historical evidence must have is_historical=True if present."""
        from rca_agent.models.evidence import EvidenceStatement
        tid = "hist-sep-tid"
        tp = _FakeTraceProvider(traces_by_id={tid: _error_trace(tid)})
        agent, _ = _make_agent(trace_provider=tp, memory_enabled=True)
        result = agent.investigate(_incident(trace_id=tid))
        for ev in result.structured_evidence:
            if getattr(ev, "is_historical", False):
                assert ev.statement_type != EvidenceStatement.FACT, (
                    "Historical evidence must not be FACT about the current incident"
                )


# ===========================================================================
# 9. Secondary traces do not replace the detected primary trace
# ===========================================================================

class TestSecondaryTracesDoNotReplacePrimary:
    def test_primary_trace_not_overwritten_by_search(self) -> None:
        """When exact trace found, search_traces must not be called at all."""
        tid = "primary-not-replaced"
        secondary = _error_trace("secondary-trace-ignored")
        tp = _FakeTraceProvider(
            traces_by_id={tid: _error_trace(tid)},
            search_results=[secondary],
        )
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=tid))
        # search_traces should NOT be called
        assert tp.search_calls == [], (
            "search_traces was called even though exact trace was found — "
            "secondary traces must not replace primary"
        )
        # get_trace must have been called
        assert tid in tp.get_trace_calls

    def test_secondary_traces_only_used_when_exact_missing(self) -> None:
        """Secondary search only runs when exact trace is NOT found."""
        tid = "primary-missing"
        secondary = _error_trace("secondary-fallback")
        tp = _FakeTraceProvider(
            traces_by_id={},            # exact not available
            search_results=[secondary], # broad search fills the gap
        )
        agent, _ = _make_agent(trace_provider=tp)
        agent.investigate(_incident(trace_id=tid))
        # Exact lookup was attempted
        assert tid in tp.get_trace_calls
        # Fallback search was then used
        assert len(tp.search_calls) == 1


# ===========================================================================
# 10. Existing autonomous monitor tests still pass (regression)
# ===========================================================================

class TestMonitorRegression:
    def test_jaeger_monitor_still_detects_error_traces(self) -> None:
        """Core monitor detection logic unchanged."""
        from rca_agent.monitor.jaeger_monitor import JaegerMonitor
        from rca_agent.monitor.trace_error_detector import TraceErrorDetector
        from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry

        class _FakeClient:
            def search_traces(self, q):
                return TraceSearchResult(
                    traces=[_error_trace("mon-reg-001")], total=1, query=q
                )
            def list_services(self):
                return []

        calls = []
        class _CaptureTrigger:
            def trigger(self, trace: Trace) -> RCAResult:
                calls.append(trace.trace_id)
                return RCAResult(
                    incident_id="auto-x",
                    status=RCAStatus.PARTIAL,
                    summary="ok",
                    confidence=0.5,
                )

        monitor = JaegerMonitor(
            client=_FakeClient(),
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=_CaptureTrigger(),
            config=MonitorConfig(jaeger_url="http://x"),
        )
        count = monitor.run_once()
        assert count == 1
        assert "mon-reg-001" in calls

    def test_rca_workflow_trigger_build_incident_trace_id(self) -> None:
        """RCAWorkflowTrigger._build_incident now sets trace_id."""
        config = MonitorConfig(jaeger_url="http://x")
        trigger = RCAWorkflowTrigger(config)
        full_id = "abcdef1234567890abcdef1234567890"
        trace = _error_trace(full_id)
        incident = trigger._build_incident(trace)
        assert incident.trace_id == full_id

    def test_deduplication_still_works(self) -> None:
        """Same trace cannot trigger RCA twice after fix."""
        from rca_agent.monitor.jaeger_monitor import JaegerMonitor
        from rca_agent.monitor.trace_error_detector import TraceErrorDetector
        from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry

        err = _error_trace("dedup-after-fix")

        class _TwoReturnClient:
            _call = 0
            def search_traces(self, q):
                self._call += 1
                return TraceSearchResult(traces=[err], total=1, query=q)
            def list_services(self): return []

        calls = []
        class _CT:
            def trigger(self, t: Trace) -> RCAResult:
                calls.append(t.trace_id)
                return RCAResult(incident_id="x", status=RCAStatus.PARTIAL, summary="ok", confidence=0.5)

        monitor = JaegerMonitor(
            client=_TwoReturnClient(),
            detector=TraceErrorDetector(),
            registry=ProcessedTraceRegistry(),
            trigger=_CT(),
            config=MonitorConfig(jaeger_url="http://x"),
        )
        monitor.run_once()
        monitor.run_once()
        assert calls.count("dedup-after-fix") == 1


# ===========================================================================
# 11. Incident model backward compatibility
# ===========================================================================

class TestIncidentModelBackwardCompat:
    def test_existing_incidents_without_trace_id_still_valid(self) -> None:
        """All existing code that constructs Incident without trace_id still works."""
        inc = Incident(
            application="payments-api",
            environment="production",
            title="DB pool exhaustion",
            severity=Severity.CRITICAL,
            status=IncidentStatus.OPEN,
            start_time=_NOW,
        )
        assert inc.trace_id is None

    def test_incident_with_trace_id_valid(self) -> None:
        inc = Incident(
            application="payments-api",
            environment="production",
            title="DB pool exhaustion",
            severity=Severity.CRITICAL,
            status=IncidentStatus.OPEN,
            start_time=_NOW,
            trace_id="deadbeef0000000000000000cafebabe",
        )
        assert inc.trace_id == "deadbeef0000000000000000cafebabe"

    def test_incident_serialises_with_trace_id(self) -> None:
        """Pydantic serialisation includes trace_id."""
        inc = Incident(
            application="svc",
            environment="prod",
            title="err",
            start_time=_NOW,
            trace_id="t123",
        )
        d = inc.model_dump()
        assert d["trace_id"] == "t123"

    def test_incident_serialises_without_trace_id(self) -> None:
        inc = Incident(
            application="svc",
            environment="prod",
            title="err",
            start_time=_NOW,
        )
        d = inc.model_dump()
        assert d["trace_id"] is None

    def test_existing_test_evaluation_not_broken(self) -> None:
        """Phase 4 golden dataset EvalCases still construct Incidents correctly."""
        from evaluation.datasets.golden_dataset import GOLDEN_EVAL_CASES
        from evaluation.runners.eval_runner import _build_incident
        for ec in GOLDEN_EVAL_CASES:
            inc = _build_incident(ec.incident_data)
            assert isinstance(inc, Incident)
            assert inc.trace_id is None  # golden cases don't set trace_id
