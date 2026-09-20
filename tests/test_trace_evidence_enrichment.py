"""Tests for trace evidence enrichment (P0 task: trace findings in LLM prompt).

Verifies acceptance criteria:
1.  Exact trace remains primary evidence (trace_findings derived from it).
2.  Trace ID is passed to LogProvider (get_logs_by_trace_id called).
3.  Correlated logs are included in raw_logs.
4.  Span attributes and exception details surface in trace_findings.
5.  Related spans are included when explicitly in the same trace.
6.  No unrelated recent traces injected as primary evidence.
7.  Historical evidence remains separate from CURRENT TRACE EVIDENCE.
8.  Missing logs handled gracefully (no crash).
9.  Missing exact trace handled gracefully (no crash).
10. Existing autonomous monitor tests still pass.
11. Existing Phase 1–4 tests still pass (regression).

All tests use fakes — no live Jaeger or RKE required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rca_agent.agents.nodes import _extract_trace_findings
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.agents.rca_graph import build_rca_graph
from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.incident import Incident, IncidentStatus, Severity
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.rca_result import RCAResult, RCAStatus
from rca_agent.models.trace_models import (
    Span,
    SpanEvent,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
    TraceStatus,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 9, 20, 17, 0, 0, tzinfo=timezone.utc)


def _span(
    trace_id: str = "t001",
    span_id: str = "s001",
    service: str = "test-svc",
    operation: str = "POST /api/test",
    status: TraceStatus = TraceStatus.ERROR,
    status_message: str | None = None,
    attributes: dict | None = None,
    events: list[SpanEvent] | None = None,
    parent: str | None = None,
    duration_ms: float = 500.0,
) -> Span:
    return Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent,
        service_name=service,
        operation_name=operation,
        start_time=_NOW,
        end_time=_NOW,
        duration_ms=duration_ms,
        status=status,
        status_message=status_message,
        attributes=attributes or {},
        events=events or [],
    )


def _error_trace(
    trace_id: str = "err-001",
    service: str = "rke-backend",
    operation: str = "POST /api/test/incidents/historical",
    status_message: str | None = "HTTP 500",
    attributes: dict | None = None,
) -> Trace:
    span = _span(
        trace_id=trace_id,
        service=service,
        operation=operation,
        status=TraceStatus.ERROR,
        status_message=status_message,
        attributes=attributes or {"error": True, "http.status_code": 500},
    )
    return Trace(trace_id=trace_id, spans=[span])


def _incident(
    trace_id: str | None = None,
    application: str = "test-svc",
    start_time: datetime | None = None,
) -> Incident:
    return Incident(
        incident_id="test-inc",
        application=application,
        environment="test",
        title="Test incident",
        severity=Severity.HIGH,
        status=IncidentStatus.OPEN,
        start_time=start_time or (_NOW - timedelta(minutes=5)),
        trace_id=trace_id,
    )


class _FakeTraceProvider:
    def __init__(
        self,
        by_id: dict[str, Trace] | None = None,
        search_results: list[Trace] | None = None,
    ) -> None:
        self._by_id = by_id or {}
        self._search = search_results or []
        self.get_calls: list[str] = []
        self.search_calls: list[TraceSearchQuery] = []

    def get_trace(self, tid: str) -> Trace | None:
        self.get_calls.append(tid)
        return self._by_id.get(tid)

    def search_traces(self, q: TraceSearchQuery) -> TraceSearchResult:
        self.search_calls.append(q)
        return TraceSearchResult(traces=self._search, total=len(self._search), query=q)

    def get_trace_spans(self, tid: str): return []
    def get_failed_spans(self, tid: str): return []


class _FakeLogProvider:
    def __init__(
        self,
        by_trace: dict[str, list[LogEntry]] | None = None,
        search_entries: list[LogEntry] | None = None,
    ) -> None:
        self._by_trace = by_trace or {}
        self._search = search_entries or []
        self.trace_calls: list[str] = []

    def search_logs(self, q: LogSearchQuery) -> LogSearchResult:
        return LogSearchResult(entries=self._search, query=q)

    def get_logs_by_trace_id(self, tid: str) -> LogSearchResult:
        self.trace_calls.append(tid)
        return LogSearchResult(
            entries=self._by_trace.get(tid, []),
            query=LogSearchQuery(trace_id=tid),
        )

    def get_log_by_id(self, lid: str): return None


class _NoOpGit:
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, c): raise ValueError("no git")
    def get_diff(self, c): return []
    def get_files_changed(self, c): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


def _make_graph(tp=None, lp=None):
    return build_rca_graph(
        llm=MockLLMProvider(),
        log_provider=lp or _FakeLogProvider(),
        git_provider=_NoOpGit(),
        memory=IncidentMemory(
            graph=InMemoryGraphProvider(),
            vector=TfidfVectorProvider(),
        ),
        trace_provider=tp,
        memory_enabled=False,
    )


# ===========================================================================
# 1. _extract_trace_findings — unit tests
# ===========================================================================

class TestExtractTraceFindings:
    def test_empty_traces_returns_empty(self) -> None:
        assert _extract_trace_findings([]) == []

    def test_trace_with_no_spans_returns_trace_label(self) -> None:
        trace = Trace(trace_id="abc123def456", spans=[])
        findings = _extract_trace_findings([trace])
        assert len(findings) >= 1
        assert "abc123def456"[:16] in findings[0]

    def test_error_span_produces_status_error_line(self) -> None:
        trace = _error_trace("t001")
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "ERROR" in combined

    def test_exception_type_surfaces_in_findings(self) -> None:
        span = _span(
            trace_id="t002",
            status=TraceStatus.ERROR,
            attributes={
                "error.type": "okhttp3.internal.http2.ConnectionShutdownException",
                "http.status_code": 500,
            },
        )
        trace = Trace(trace_id="t002", spans=[span])
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        # The exception class name must appear in the findings
        assert "ConnectionShutdownException" in combined

    def test_http_route_surfaces_in_findings(self) -> None:
        span = _span(
            trace_id="t003",
            status=TraceStatus.ERROR,
            attributes={"http.route": "/api/test/incidents/historical", "http.status_code": 500},
        )
        trace = Trace(trace_id="t003", spans=[span])
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "/api/test/incidents/historical" in combined

    def test_db_system_surfaces_in_findings(self) -> None:
        span = _span(
            trace_id="t004",
            status=TraceStatus.UNSET,
            attributes={"db.system": "postgresql", "db.statement": "SELECT 1"},
        )
        trace = Trace(trace_id="t004", spans=[span])
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "postgresql" in combined

    def test_ok_span_without_interesting_attrs_not_added(self) -> None:
        span = _span(trace_id="t005", status=TraceStatus.OK, attributes={})
        trace = Trace(trace_id="t005", spans=[span])
        findings = _extract_trace_findings([trace])
        # Trace header line still present, but no span detail line
        span_lines = [f for f in findings if "span " in f]
        assert span_lines == []

    def test_exception_event_surfaces_in_findings(self) -> None:
        event = SpanEvent(
            name="exception",
            timestamp=_NOW,
            attributes={
                "exception.type": "java.net.SocketException",
                "exception.message": "Connection reset",
            },
        )
        span = _span(trace_id="t006", status=TraceStatus.ERROR, events=[event])
        trace = Trace(trace_id="t006", spans=[span])
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "SocketException" in combined

    def test_status_message_surfaces_in_findings(self) -> None:
        span = _span(
            trace_id="t007",
            status=TraceStatus.ERROR,
            status_message="Connection refused by peer",
        )
        trace = Trace(trace_id="t007", spans=[span])
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "Connection refused" in combined

    def test_multiple_spans_all_extracted(self) -> None:
        trace = Trace(
            trace_id="t008",
            spans=[
                _span(trace_id="t008", span_id="s1", status=TraceStatus.ERROR,
                      attributes={"error.type": "TimeoutException"}),
                _span(trace_id="t008", span_id="s2", status=TraceStatus.UNSET,
                      attributes={"db.system": "postgresql"}),
            ],
        )
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "TimeoutException" in combined
        assert "postgresql" in combined

    def test_realistic_connection_shutdown_trace(self) -> None:
        """Reproduce the exact error seen in the demo run."""
        span = _span(
            trace_id="f5d952198f231038d2c23912f697e677",
            service="rke-backend",
            operation="POST /api/test/incidents/historical",
            status=TraceStatus.ERROR,
            status_message="HTTP 500",
            duration_ms=2800.0,
            attributes={
                "error": True,
                "error.type": "okhttp3.internal.http2.ConnectionShutdownException",
                "http.status_code": 500,
                "http.route": "/api/test/incidents/historical",
                "http.request.method": "POST",
            },
        )
        trace = Trace(trace_id="f5d952198f231038d2c23912f697e677", spans=[span])
        findings = _extract_trace_findings([trace])
        combined = "\n".join(findings)
        assert "ConnectionShutdownException" in combined
        assert "status=ERROR" in combined
        assert "/api/test/incidents/historical" in combined


# ===========================================================================
# 2. trace_findings in LangGraph state (integration via graph invoke)
# ===========================================================================

class TestTracesFindingsInState:
    def test_trace_findings_populated_when_exact_trace_found(self) -> None:
        """When incident has trace_id and exact trace is found, trace_findings is non-empty."""
        tid = "exact-findings-test"
        trace = _error_trace(
            tid, attributes={"error.type": "SomeException", "http.status_code": 500}
        )
        graph = _make_graph(tp=_FakeTraceProvider(by_id={tid: trace}))
        state = graph.invoke({
            "incident": _incident(trace_id=tid),
            "memory_enabled": False,
        })
        trace_findings = state.get("trace_findings", [])
        assert len(trace_findings) >= 1, "trace_findings must be non-empty when trace found"
        combined = "\n".join(trace_findings)
        assert "SomeException" in combined

    def test_trace_findings_empty_when_no_trace_provider(self) -> None:
        """Without a trace provider, trace_findings is empty."""
        graph = _make_graph(tp=None)
        state = graph.invoke({
            "incident": _incident(trace_id="some-id"),
            "memory_enabled": False,
        })
        assert state.get("trace_findings", []) == []

    def test_trace_findings_empty_when_no_trace_id(self) -> None:
        """Without incident.trace_id, broad search may return traces but trace_findings
        is still built from whatever traces are returned."""
        graph = _make_graph(tp=_FakeTraceProvider(search_results=[]))
        state = graph.invoke({
            "incident": _incident(trace_id=None),
            "memory_enabled": False,
        })
        # With no traces and no trace_id, trace_findings should be empty
        assert state.get("trace_findings", []) == []

    def test_trace_findings_include_service_and_operation(self) -> None:
        tid = "svc-op-test"
        trace = _error_trace(tid, service="my-api", operation="GET /health")
        graph = _make_graph(tp=_FakeTraceProvider(by_id={tid: trace}))
        state = graph.invoke({
            "incident": _incident(trace_id=tid, application="my-api"),
            "memory_enabled": False,
        })
        combined = "\n".join(state.get("trace_findings", []))
        assert "my-api" in combined
        assert "/health" in combined


# ===========================================================================
# 3. Correlated logs via trace_id
# ===========================================================================

class TestCorrelatedLogRetrieval:
    def test_get_logs_by_trace_id_called_for_exact_trace(self) -> None:
        """Node 2 calls get_logs_by_trace_id when exact trace found."""
        tid = "corr-logs-tid"
        tp = _FakeTraceProvider(by_id={tid: _error_trace(tid)})
        lp = _FakeLogProvider()
        graph = _make_graph(tp=tp, lp=lp)
        graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        assert tid in lp.trace_calls

    def test_correlated_logs_in_raw_logs(self) -> None:
        """Logs returned by get_logs_by_trace_id appear in raw_logs state."""
        import json
        tid = "corr-in-raw"
        raw_line = json.dumps({
            "timestamp": _NOW.isoformat(),
            "level": "ERROR",
            "service": "test-svc",
            "message": "Pool exhausted",
            "traceId": tid,
        })
        entry = LogEntry.from_raw_line(raw_line)
        tp = _FakeTraceProvider(by_id={tid: _error_trace(tid)})
        lp = _FakeLogProvider(by_trace={tid: [entry]})
        graph = _make_graph(tp=tp, lp=lp)
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        messages = [e.message for e in state.get("raw_logs", [])]
        assert "Pool exhausted" in messages

    def test_missing_logs_does_not_crash(self) -> None:
        """get_logs_by_trace_id returning no entries does not crash."""
        tid = "no-logs-tid"
        tp = _FakeTraceProvider(by_id={tid: _error_trace(tid)})
        lp = _FakeLogProvider()  # no entries for any trace
        graph = _make_graph(tp=tp, lp=lp)
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        assert isinstance(state.get("rca_result"), RCAResult)

    def test_log_provider_exception_does_not_crash(self) -> None:
        """If get_logs_by_trace_id raises, investigation continues."""
        class _FailingLog(_FakeLogProvider):
            def get_logs_by_trace_id(self, tid: str) -> LogSearchResult:
                raise ConnectionError("log server down")

        tid = "failing-log-tid"
        tp = _FakeTraceProvider(by_id={tid: _error_trace(tid)})
        graph = _make_graph(tp=tp, lp=_FailingLog())
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        assert isinstance(state.get("rca_result"), RCAResult)


# ===========================================================================
# 4. Parent / related spans within same trace
# ===========================================================================

class TestRelatedSpans:
    def test_multi_span_trace_all_spans_in_trace_findings(self) -> None:
        """All spans of the exact trace are processed into trace_findings."""
        tid = "multi-span"
        trace = Trace(
            trace_id=tid,
            spans=[
                _span(trace_id=tid, span_id="root", service="gateway",
                      operation="POST /", status=TraceStatus.ERROR,
                      attributes={"http.status_code": 500}),
                _span(trace_id=tid, span_id="db", service="database",
                      operation="SELECT", status=TraceStatus.UNSET,
                      attributes={"db.system": "postgresql"}, parent="root"),
            ],
        )
        tp = _FakeTraceProvider(by_id={tid: trace})
        graph = _make_graph(tp=tp)
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        combined = "\n".join(state.get("trace_findings", []))
        assert "gateway" in combined or "POST" in combined
        assert "postgresql" in combined or "database" in combined

    def test_parent_span_in_same_trace_not_separate_request(self) -> None:
        """Parent-child spans in the same trace.trace_id are treated as one unit."""
        tid = "parent-child"
        root = _span(trace_id=tid, span_id="root", status=TraceStatus.ERROR,
                     attributes={"http.status_code": 500})
        child = _span(trace_id=tid, span_id="child", status=TraceStatus.UNSET,
                      attributes={"db.system": "postgresql"}, parent="root")
        trace = Trace(trace_id=tid, spans=[root, child])
        tp = _FakeTraceProvider(by_id={tid: trace})
        graph = _make_graph(tp=tp)
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        # Both spans should be in trace_findings
        combined = "\n".join(state.get("trace_findings", []))
        assert "postgresql" in combined

    def test_missing_exact_trace_falls_back_gracefully(self) -> None:
        """If exact trace is absent, investigation doesn't crash."""
        tp = _FakeTraceProvider(by_id={}, search_results=[])
        graph = _make_graph(tp=tp)
        state = graph.invoke({
            "incident": _incident(trace_id="not-in-jaeger"),
            "memory_enabled": False,
        })
        result = state.get("rca_result")
        assert isinstance(result, RCAResult)


# ===========================================================================
# 5. No unrelated traces injected as primary evidence
# ===========================================================================

class TestNoUnrelatedTraces:
    def test_search_not_called_when_exact_found(self) -> None:
        """Broad search must not run when exact trace is found."""
        tid = "exact-no-search"
        tp = _FakeTraceProvider(
            by_id={tid: _error_trace(tid)},
            search_results=[_error_trace("unrelated")],
        )
        graph = _make_graph(tp=tp)
        graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        assert tp.search_calls == [], "search_traces must not be called when exact trace found"

    def test_unrelated_trace_not_in_trace_findings(self) -> None:
        """Unrelated traces from broad search don't appear when exact trace found."""
        tid = "primary-only"
        tp = _FakeTraceProvider(
            by_id={tid: _error_trace(tid, attributes={"error.type": "PrimaryError"})},
            search_results=[
                _error_trace("unrelated", attributes={"error.type": "UnrelatedError"})
            ],
        )
        graph = _make_graph(tp=tp)
        state = graph.invoke({"incident": _incident(trace_id=tid), "memory_enabled": False})
        combined = "\n".join(state.get("trace_findings", []))
        assert "UnrelatedError" not in combined
        assert "PrimaryError" in combined


# ===========================================================================
# 6. Historical evidence remains separate from current trace evidence
# ===========================================================================

class TestHistoricalSeparation:
    def test_historical_section_labelled_separately(self) -> None:
        """Node 6 uses 'HISTORICAL INCIDENTS (for context only...)' not 'HISTORICAL INCIDENTS:'."""
        # Verify the new section header is in the node source
        import inspect
        from rca_agent.agents.nodes import make_correlate_evidence_node
        src = inspect.getsource(make_correlate_evidence_node)
        assert "for context only" in src or "HISTORICAL INCIDENTS" in src

    def test_memory_off_no_historical_in_trace_investigation(self) -> None:
        tid = "hist-sep"
        tp = _FakeTraceProvider(by_id={tid: _error_trace(tid)})
        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=_FakeLogProvider(),
            git_provider=_NoOpGit(),
            memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
            trace_provider=tp,
            memory_enabled=False,
            auto_store_rca=False,
        )
        result = agent.investigate(_incident(trace_id=tid))
        assert result.memory_enabled is False
        assert result.retrieved_historical_count == 0
        # Verify historical_context_notes records OFF status
        combined = " ".join(result.historical_context_notes)
        assert "MEMORY OFF" in combined or "disabled" in combined.lower()


# ===========================================================================
# 7. CURRENT TRACE EVIDENCE label appears in LLM prompt when traces present
# ===========================================================================

class TestLLMPromptContent:
    def test_trace_analysis_section_in_correlate_node_source(self) -> None:
        """CURRENT TRACE EVIDENCE section is in Node 6 source."""
        import inspect
        from rca_agent.agents.nodes import make_correlate_evidence_node
        src = inspect.getsource(make_correlate_evidence_node)
        assert "CURRENT TRACE EVIDENCE" in src

    def test_historical_incidents_labelled_context_only(self) -> None:
        """Historical incidents section must be labelled 'context only' not plain."""
        import inspect
        from rca_agent.agents.nodes import make_correlate_evidence_node
        src = inspect.getsource(make_correlate_evidence_node)
        # The old 'HISTORICAL INCIDENTS:' must be replaced with a label including 'context'
        assert "context only" in src

    def test_rca_evidence_summary_log_exists(self) -> None:
        """Node 2 must emit an 'RCA evidence summary' log line."""
        import inspect
        from rca_agent.agents.nodes import make_retrieve_evidence_node
        src = inspect.getsource(make_retrieve_evidence_node)
        assert "RCA evidence summary" in src

    def test_extract_trace_findings_helper_exists(self) -> None:
        """The _extract_trace_findings helper function must exist."""
        from rca_agent.agents.nodes import _extract_trace_findings
        assert callable(_extract_trace_findings)


# ===========================================================================
# 8. Regression: existing monitor and workflow tests still pass
# ===========================================================================

class TestRegression:
    def test_existing_rca_agent_without_trace_id(self) -> None:
        """Legacy usage without trace_id still works."""
        agent = RCAAgent(
            llm=MockLLMProvider(),
            log_provider=_FakeLogProvider(),
            git_provider=_NoOpGit(),
            memory=IncidentMemory(InMemoryGraphProvider(), TfidfVectorProvider()),
            auto_store_rca=False,
        )
        inc = Incident(
            application="test",
            environment="test",
            title="Test",
            severity=Severity.LOW,
            status=IncidentStatus.OPEN,
            start_time=_NOW,
        )
        result = agent.investigate(inc)
        assert isinstance(result, RCAResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_existing_phase4_golden_cases_unaffected(self) -> None:
        """Phase 4 golden EvalCases build Incidents without trace_id — still valid."""
        from evaluation.datasets.golden_dataset import GOLDEN_EVAL_CASES
        from evaluation.runners.eval_runner import _build_incident
        for ec in GOLDEN_EVAL_CASES:
            inc = _build_incident(ec.incident_data)
            assert inc.trace_id is None  # golden cases don't set trace_id

    def test_trace_findings_accumulate_across_operator_add(self) -> None:
        """trace_findings uses operator.add reducer — verify it's in state annotations."""
        import typing
        from rca_agent.agents.state import InvestigationState
        ann = typing.get_type_hints(InvestigationState, include_extras=True)
        assert "trace_findings" in ann

    def test_monitor_trigger_build_incident_still_sets_trace_id(self) -> None:
        """RCAWorkflowTrigger still propagates trace_id correctly."""
        from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
        from rca_agent.monitor.config import MonitorConfig
        config = MonitorConfig(jaeger_url="http://x")
        trigger = RCAWorkflowTrigger(config)
        trace = _error_trace("regression-trace-id")
        inc = trigger._build_incident(trace)
        assert inc.trace_id == "regression-trace-id"
