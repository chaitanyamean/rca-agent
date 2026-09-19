"""Tests for the RCA Agent — Phase 6.

All tests use:
* MockLLMProvider     — deterministic, no network calls
* InMemoryGraphProvider + TfidfVectorProvider — no Neo4j or PostgreSQL
* In-memory log / git fixtures — no file I/O or subprocess calls

Test scenarios
--------------
1. Clear database failure        — logs show SQLException → COMPLETE/PARTIAL, high confidence
2. Deployment regression         — recent commit correlates with incident
3. Insufficient evidence         — no logs, no commits → INSUFFICIENT_EVIDENCE + unknowns
4. Conflicting evidence          — two contradictory candidates, low confidence
5. Historical similar incident   — memory contains matching past incident

Keywords used in MockLLMProvider match text that actually appears in node prompts
(verified by probing the real prompt content at runtime).

Node prompt identifiers:
  "Analyse this incident"    → understand_incident
  "Log entries:"             → analyze_logs
  "Recent commits:"          → inspect_git
  "Similar historical"       → search_historical (only fires when results exist)
  "All findings:"            → correlate_evidence
  "key:\\ncandidates"        → generate_candidate
  "selected_index"           → validate_candidate
  "Root cause:"              → generate_rca
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from rca_agent.agents.llm_provider import MockLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.git_models import ChangeType, Commit, CommitDiff, CommitFile, GitCommitQuery
from rca_agent.models.incident import (
    Incident,
    IncidentError,
    IncidentStatus,
    IncidentSymptom,
    Severity,
)
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.rca_result import EvidenceStatement, RCAStatus

NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------

def _make_log_entry(
    message: str,
    level: str = "ERROR",
    service: str = "payments-api",
    exception: str | None = None,
    minutes_ago: int = 5,
) -> LogEntry:
    ts = NOW - timedelta(minutes=minutes_ago)
    raw = json.dumps({
        "timestamp": ts.isoformat(),
        "service": service,
        "level": level,
        "message": message,
        "exception": exception,
        "traceId": "trace-001",
        "endpoint": "/api/payments",
        "status": 500,
    })
    return LogEntry.from_raw_line(raw)


def _make_commit(
    short_id: str = "abc1234",
    subject: str = "fix: update database query",
    minutes_ago: int = 30,
    files: list[str] | None = None,
) -> Commit:
    ts = NOW - timedelta(minutes=minutes_ago)
    changed = [
        CommitFile(file_path=f, change_type=ChangeType.MODIFIED)
        for f in (files or ["app/database.py"])
    ]
    return Commit(
        commit_id=short_id * 6,
        short_id=short_id,
        author="dev",
        author_email="dev@example.com",
        timestamp=ts,
        message=subject,
        subject=subject,
        files_changed=changed,
    )


class FakeLogProvider:
    def __init__(self, entries: list[LogEntry] | None = None) -> None:
        self._entries = entries or []

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        return LogSearchResult(entries=self._entries, query=query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        matching = [e for e in self._entries if e.trace_id == trace_id]
        return LogSearchResult(entries=matching, query=LogSearchQuery(trace_id=trace_id))

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        return next((e for e in self._entries if e.id == log_id), None)


class FakeGitProvider:
    def __init__(self, commits: list[Commit] | None = None) -> None:
        self._commits = commits or []

    def get_recent_commits(self, limit: int = 20) -> list[Commit]:
        return self._commits[:limit]

    def get_commit(self, commit_id: str) -> Commit:
        for c in self._commits:
            if c.commit_id == commit_id or c.short_id == commit_id:
                return c
        raise ValueError(f"Commit not found: {commit_id}")

    def get_diff(self, commit_id: str) -> list[CommitDiff]:
        c = self.get_commit(commit_id)
        return [
            CommitDiff(
                commit_id=c.commit_id, file_path=f.file_path,
                change_type=f.change_type, additions=5, deletions=3,
                patch="@@ -1,3 +1,5 @@\n+new line\n context\n",
            )
            for f in c.files_changed
        ]

    def get_files_changed(self, commit_id: str) -> list[str]:
        return [f.file_path for f in self.get_commit(commit_id).files_changed]

    def search_commits(self, query: GitCommitQuery) -> list[Commit]:
        if query.keyword:
            return [c for c in self._commits if query.keyword.lower() in c.message.lower()]
        return self._commits

    def get_commits_between(self, start_time: datetime, end_time: datetime) -> list[Commit]:
        return [c for c in self._commits if start_time <= c.timestamp <= end_time]


def _make_memory(seed_incidents: bool = False) -> IncidentMemory:
    mem = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=0.05,
        auto_link_similar=True,
    )
    if seed_incidents:
        past = Incident(
            incident_id="INC-HIST-001",
            application="payments-api",
            environment="production",
            title="Database connection pool exhaustion",
            description="The payments API ran out of PostgreSQL connections causing HTTP 500 errors.",
            severity=Severity.CRITICAL,
            status=IncidentStatus.RESOLVED,
            start_time=NOW - timedelta(days=30),
        )
        mem.store_incident(past)
    return mem


def _make_agent(
    llm: MockLLMProvider,
    logs: list[LogEntry] | None = None,
    commits: list[Commit] | None = None,
    seed_memory: bool = False,
) -> RCAAgent:
    return RCAAgent(
        llm=llm,
        log_provider=FakeLogProvider(logs or []),
        git_provider=FakeGitProvider(commits or []),
        memory=_make_memory(seed_incidents=seed_memory),
        max_log_entries=50,
        max_commits=20,
        similar_incidents_top_k=3,
    )


def _make_incident(
    incident_id: str = "INC-TEST",
    title: str = "Payments API returning HTTP 500",
    description: str = "All payment requests are failing with 500 errors.",
    application: str = "payments-api",
    severity: Severity = Severity.CRITICAL,
) -> Incident:
    return Incident(
        incident_id=incident_id,
        application=application,
        environment="production",
        title=title,
        description=description,
        severity=severity,
        status=IncidentStatus.OPEN,
        start_time=NOW - timedelta(hours=1),
        affected_services=[application],
        symptoms=[IncidentSymptom(
            description="HTTP 500 errors on all POST /api/payments",
            observed_at=NOW - timedelta(hours=1),
            service=application,
        )],
        errors=[IncidentError(
            error_type="SQLException",
            message="Connection pool exhausted: no available connections",
            service=application,
            count=500,
        )],
    )


# ---------------------------------------------------------------------------
# LLM response factories
# ---------------------------------------------------------------------------

def _r_understand(terms: list[str] | None = None) -> str:
    return json.dumps({
        "incident_summary": "Payments API is failing with database errors.",
        "key_search_terms": terms or ["database", "connection", "pool", "SQLException"],
        "investigation_plan": "Check logs; inspect commits; search history.",
    })


def _r_logs_db() -> str:
    return json.dumps({
        "findings": ["FACT: Multiple SQLException errors in logs"],
        "error_patterns": ["SQLException", "connection pool exhausted"],
        "evidence": [{
            "statement_type": "FACT",
            "description": "SQLException: connection pool exhausted in logs",
            "source_type": "log", "source_ref": "trace-001",
        }],
    })


def _r_logs_empty() -> str:
    return json.dumps({"findings": ["No logs."], "error_patterns": [], "evidence": []})


def _r_git_regression(sha: str = "abc1234") -> str:
    return json.dumps({
        "findings": [f"FACT: Commit {sha} modified database config"],
        "suspicious_commits": [sha],
        "evidence": [{
            "statement_type": "FACT",
            "description": f"Commit {sha} changed database connection configuration",
            "source_type": "git_commit", "source_ref": sha,
        }],
    })


def _r_git_empty() -> str:
    return json.dumps({"findings": ["No commits."], "suspicious_commits": [], "evidence": []})


def _r_historical_found() -> str:
    return json.dumps({"findings": [
        "INFERENCE: INC-HIST-001 had identical symptoms — resolved by pool size increase."
    ]})


def _r_correlate(summary: str = "DB pool exhausted after a code change.") -> str:
    return json.dumps({
        "correlation_summary": summary,
        "evidence": [{
            "statement_type": "FACT",
            "description": "Log errors and commit changes both point to database layer",
            "source_type": "log", "source_ref": "trace-001",
        }],
        "affected_services": ["payments-api"],
    })


def _r_correlate_conflicting() -> str:
    return json.dumps({
        "correlation_summary": "Conflicting signals: DB overload vs code change",
        "evidence": [
            {"statement_type": "FACT", "description": "SQLException in logs",
             "source_type": "log", "source_ref": "log-1"},
            {"statement_type": "INFERENCE", "description": "DB may be independently overloaded",
             "source_type": "log", "source_ref": None},
        ],
        "affected_services": ["payments-api"],
    })


def _r_correlate_empty() -> str:
    return json.dumps({"correlation_summary": "No evidence.", "evidence": [], "affected_services": []})


def _r_candidate_db() -> str:
    return json.dumps({"candidates": [{
        "summary": "Slow database query exhausted the connection pool",
        "category": "code_bug", "component": "payments-api/database",
        "confidence": 0.85, "statement_type": "FACT",
        "supporting_evidence": ["trace-001", "abc1234"],
        "contradicting_evidence": [],
    }]})


def _r_candidate_deploy() -> str:
    return json.dumps({"candidates": [{
        "summary": "Deployment introduced a configuration change that broke DB connections",
        "category": "config_change", "component": "payments-api/config",
        "confidence": 0.75, "statement_type": "FACT",
        "supporting_evidence": ["abc1234"], "contradicting_evidence": [],
    }]})


def _r_candidate_conflicting() -> str:
    return json.dumps({"candidates": [
        {"summary": "Database server overloaded", "category": "infrastructure",
         "confidence": 0.45, "statement_type": "INFERENCE",
         "supporting_evidence": [], "contradicting_evidence": ["commit_shows_bug"]},
        {"summary": "Code bug introduced slow query", "category": "code_bug",
         "confidence": 0.40, "statement_type": "INFERENCE",
         "supporting_evidence": ["commit_shows_bug"], "contradicting_evidence": ["db_overload"]},
    ]})


def _r_candidate_empty() -> str:
    return json.dumps({"candidates": []})


def _r_validate(idx: int = 0, confidence: float = 0.85, stmt: str = "FACT") -> str:
    return json.dumps({
        "selected_index": idx,
        "adjusted_confidence": confidence,
        "validation_notes": ["Candidate supported by direct log evidence"],
        "statement_type": stmt,
    })


def _r_validate_none() -> str:
    return json.dumps({
        "selected_index": None, "adjusted_confidence": 0.15,
        "validation_notes": ["Insufficient evidence"], "statement_type": "UNKNOWN",
    })


def _r_validate_conflicting() -> str:
    return json.dumps({
        "selected_index": 0, "adjusted_confidence": 0.35,
        "validation_notes": [
            "Conflicting evidence — cannot confirm",
            "contradicts available commit evidence",
        ],
        "statement_type": "INFERENCE",
    })


def _r_rca(summary: str = "The payments API failed due to DB connection pool exhaustion.") -> str:
    return json.dumps({
        "summary": summary,
        "contributing_factors": ["No pool monitoring", "Missing query timeout"],
        "unknowns": [],
        "recommended_next_steps": ["Increase pool size", "Add query timeout", "Add alerting"],
        "affected_services": ["payments-api"],
    })


def _r_rca_insufficient() -> str:
    return json.dumps({
        "summary": "Root cause could not be determined — no logs or commits available.",
        "contributing_factors": [],
        "unknowns": ["No logs found", "No commits available", "Cannot identify component"],
        "recommended_next_steps": ["Enable structured logging", "Check log retention"],
        "affected_services": [],
    })


def _r_rca_conflicting() -> str:
    return json.dumps({
        "summary": "Evidence is contradictory — DB overload and code bug both possible.",
        "contributing_factors": ["Conflicting signals"],
        "unknowns": ["Cannot confirm DB overload vs code change"],
        "recommended_next_steps": ["Review DB metrics", "Inspect commit diff"],
        "affected_services": ["payments-api"],
    })


# ---------------------------------------------------------------------------
# Pre-built response dicts using real prompt keywords
# ---------------------------------------------------------------------------

def _responses_db_failure() -> dict[str, str]:
    return {
        "Analyse this incident": _r_understand(),
        "Log entries:": _r_logs_db(),
        "Recent commits:": _r_git_regression(),
        "Similar historical": _r_historical_found(),
        "All findings:": _r_correlate(),
        "key:\ncandidates": _r_candidate_db(),
        "selected_index": _r_validate(idx=0, confidence=0.85),
        "Root cause:": _r_rca(),
    }


def _responses_deployment() -> dict[str, str]:
    return {
        "Analyse this incident": _r_understand(["config", "deployment", "gateway"]),
        "Log entries:": json.dumps({
            "findings": ["FACT: 502 errors after config deploy"],
            "error_patterns": ["BadGatewayError"],
            "evidence": [{"statement_type": "FACT",
                          "description": "502 correlates with config deployment",
                          "source_type": "log", "source_ref": "log-502"}],
        }),
        "Recent commits:": _r_git_regression("def5678"),
        "All findings:": _r_correlate("Config change broke upstream health check."),
        "key:\ncandidates": _r_candidate_deploy(),
        "selected_index": _r_validate(idx=0, confidence=0.78),
        "Root cause:": _r_rca("Gateway config change broke upstream health check."),
    }


def _responses_insufficient() -> dict[str, str]:
    return {
        "Analyse this incident": _r_understand(),
        "Log entries:": _r_logs_empty(),
        "Recent commits:": _r_git_empty(),
        "All findings:": _r_correlate_empty(),
        "key:\ncandidates": _r_candidate_empty(),
        "selected_index": _r_validate_none(),
        "Root cause:": _r_rca_insufficient(),
    }


def _responses_conflicting() -> dict[str, str]:
    return {
        "Analyse this incident": _r_understand(),
        "Log entries:": _r_logs_db(),
        "Recent commits:": _r_git_regression(),
        "All findings:": _r_correlate_conflicting(),
        "key:\ncandidates": _r_candidate_conflicting(),
        "selected_index": _r_validate_conflicting(),
        "Root cause:": _r_rca_conflicting(),
    }


def _responses_historical() -> dict[str, str]:
    return {
        "Analyse this incident": _r_understand(),
        "Log entries:": _r_logs_db(),
        "Recent commits:": _r_git_empty(),
        "Similar historical": _r_historical_found(),
        "All findings:": _r_correlate(),
        "key:\ncandidates": _r_candidate_db(),
        "selected_index": _r_validate(idx=0, confidence=0.80),
        "Root cause:": _r_rca("Based on INC-HIST-001, this is a DB pool exhaustion issue."),
    }


# ---------------------------------------------------------------------------
# Scenario 1 — Clear database failure
# ---------------------------------------------------------------------------

class TestClearDatabaseFailure:
    """Given clear log evidence of a database error, agent should return COMPLETE or PARTIAL."""

    def test_returns_rca_result(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_db_failure()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
            commits=[_make_commit()],
        )
        result = agent.investigate(_make_incident())
        assert result is not None
        assert result.incident_id == "INC-TEST"

    def test_status_complete_or_partial(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_db_failure()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
            commits=[_make_commit()],
        )
        result = agent.investigate(_make_incident())
        assert result.status in (RCAStatus.COMPLETE, RCAStatus.PARTIAL)

    def test_confidence_above_threshold(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_db_failure()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
            commits=[_make_commit()],
        )
        result = agent.investigate(_make_incident())
        assert result.confidence >= 0.4

    def test_rca_has_summary(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_db_failure()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
        )
        result = agent.investigate(_make_incident())
        assert result.summary and len(result.summary) > 10

    def test_rca_has_required_fields(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_db_failure()))
        result = agent.investigate(_make_incident())
        assert isinstance(result.evidence, list)
        assert isinstance(result.similar_incidents, list)
        assert isinstance(result.contributing_factors, list)
        assert isinstance(result.recommended_next_steps, list)
        assert isinstance(result.unknowns, list)

    def test_evidence_statement_types_valid(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_db_failure()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
        )
        result = agent.investigate(_make_incident())
        for ev in result.evidence:
            assert ev.statement_type in EvidenceStatement


# ---------------------------------------------------------------------------
# Scenario 2 — Deployment regression
# ---------------------------------------------------------------------------

class TestDeploymentRegression:
    """Given a recent commit near the incident time, agent should identify deployment cause."""

    def test_deployment_completes(self) -> None:
        incident = _make_incident(
            incident_id="INC-DEPLOY",
            title="API Gateway 502 after config deploy",
            description="Gateway returning 502 after config deployment.",
            application="api-gateway",
        )
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_deployment()),
            logs=[_make_log_entry("Bad gateway", level="ERROR", service="api-gateway")],
            commits=[_make_commit("def5678", "chore: update health check path", files=["config/gateway.yml"])],
        )
        result = agent.investigate(incident)
        assert result.status in (RCAStatus.COMPLETE, RCAStatus.PARTIAL)

    def test_valid_result(self) -> None:
        incident = _make_incident(
            incident_id="INC-DEPLOY",
            title="API Gateway 502 after config deploy",
            application="api-gateway",
        )
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_deployment()),
            commits=[_make_commit("def5678", "chore: update health check path")],
        )
        result = agent.investigate(incident)
        assert result.incident_id == "INC-DEPLOY"
        assert 0.0 <= result.confidence <= 1.0

    def test_recommended_steps(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_deployment()))
        result = agent.investigate(_make_incident(incident_id="INC-DEPLOY2"))
        assert isinstance(result.recommended_next_steps, list)


# ---------------------------------------------------------------------------
# Scenario 3 — Insufficient evidence
# ---------------------------------------------------------------------------

class TestInsufficientEvidence:
    """With no logs and no commits, agent must return low confidence + unknowns."""

    def test_status_insufficient_or_partial(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_insufficient()), logs=[], commits=[])
        result = agent.investigate(_make_incident(incident_id="INC-EMPTY"))
        assert result.status in (RCAStatus.INSUFFICIENT_EVIDENCE, RCAStatus.PARTIAL)

    def test_confidence_low(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_insufficient()), logs=[], commits=[])
        result = agent.investigate(_make_incident(incident_id="INC-EMPTY"))
        assert result.confidence < 0.5

    def test_unknowns_populated(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_insufficient()), logs=[], commits=[])
        result = agent.investigate(_make_incident(incident_id="INC-EMPTY"))
        assert len(result.unknowns) > 0

    def test_no_fabricated_high_confidence_root_cause(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_insufficient()), logs=[], commits=[])
        result = agent.investigate(_make_incident(incident_id="INC-EMPTY"))
        if result.root_cause is not None:
            assert result.root_cause.confidence < 0.5

    def test_valid_result_structure(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_insufficient()), logs=[], commits=[])
        result = agent.investigate(_make_incident(incident_id="INC-EMPTY"))
        assert result.incident_id == "INC-EMPTY"
        assert result.summary
        assert isinstance(result.recommended_next_steps, list)


# ---------------------------------------------------------------------------
# Scenario 4 — Conflicting evidence
# ---------------------------------------------------------------------------

class TestConflictingEvidence:
    """When evidence is contradictory, agent must return low confidence + unknowns."""

    def test_low_confidence(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_conflicting()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException"),
                  _make_log_entry("DB server CPU 99%", level="WARN")],
            commits=[_make_commit("abc1234", "feat: optimise query")],
        )
        result = agent.investigate(_make_incident(incident_id="INC-CONFLICT"))
        assert result.confidence < 0.7

    def test_status_conflicting_or_partial(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_conflicting()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
        )
        result = agent.investigate(_make_incident(incident_id="INC-CONFLICT"))
        assert result.status in (
            RCAStatus.CONFLICTING_EVIDENCE, RCAStatus.PARTIAL, RCAStatus.INSUFFICIENT_EVIDENCE,
        )

    def test_unknowns_or_low_confidence(self) -> None:
        agent = _make_agent(llm=MockLLMProvider(responses=_responses_conflicting()))
        result = agent.investigate(_make_incident(incident_id="INC-CONFLICT"))
        assert result.confidence < 0.7 or len(result.unknowns) > 0

    def test_evidence_has_statement_types(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_conflicting()),
            logs=[_make_log_entry("DB issue", exception="SQLException")],
        )
        result = agent.investigate(_make_incident(incident_id="INC-CONFLICT"))
        assert len({ev.statement_type for ev in result.evidence}) >= 1


# ---------------------------------------------------------------------------
# Scenario 5 — Historical similar incident retrieved
# ---------------------------------------------------------------------------

class TestHistoricalSimilarIncident:
    """When memory contains a matching past incident, agent should reference it."""

    def test_similar_incidents_list_present(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_historical()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
            seed_memory=True,
        )
        result = agent.investigate(_make_incident(
            incident_id="INC-CURRENT",
            title="Database connection pool exhausted in payments service",
            description="PostgreSQL connections exhausted causing 500 errors",
        ))
        assert isinstance(result.similar_incidents, list)

    def test_reasonable_confidence_with_history(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_historical()),
            logs=[_make_log_entry("Connection pool exhausted", exception="SQLException")],
            seed_memory=True,
        )
        result = agent.investigate(_make_incident(
            incident_id="INC-CURRENT2",
            title="Payments DB pool exhaustion",
            description="DB connection pool exhausted — same as INC-HIST-001",
        ))
        assert result.confidence > 0.0
        assert result.status in (RCAStatus.COMPLETE, RCAStatus.PARTIAL)

    def test_summary_produced(self) -> None:
        agent = _make_agent(
            llm=MockLLMProvider(responses=_responses_historical()),
            seed_memory=True,
        )
        result = agent.investigate(_make_incident(incident_id="INC-HIST-TEST"))
        assert result.summary


# ---------------------------------------------------------------------------
# Core agent unit tests
# ---------------------------------------------------------------------------

class TestRCAAgentCore:
    def test_agent_builds_without_error(self) -> None:
        assert _make_agent(llm=MockLLMProvider()) is not None

    def test_investigate_returns_rca_result_type(self) -> None:
        from rca_agent.models.rca_result import RCAResult
        result = _make_agent(llm=MockLLMProvider()).investigate(_make_incident())
        assert isinstance(result, RCAResult)

    def test_result_incident_id_matches(self) -> None:
        result = _make_agent(llm=MockLLMProvider()).investigate(
            _make_incident(incident_id="UNIQUE-ID")
        )
        assert result.incident_id == "UNIQUE-ID"

    def test_confidence_always_in_valid_range(self) -> None:
        result = _make_agent(llm=MockLLMProvider()).investigate(_make_incident())
        assert 0.0 <= result.confidence <= 1.0

    def test_status_is_valid_enum(self) -> None:
        result = _make_agent(llm=MockLLMProvider()).investigate(_make_incident())
        assert result.status in RCAStatus

    def test_mock_llm_called_multiple_times(self) -> None:
        llm = MockLLMProvider()
        _make_agent(llm=llm).investigate(_make_incident())
        # retrieve_evidence node does not call LLM; analyze/git nodes skip LLM when no data
        # understand + correlate + generate_candidate + validate + generate_rca = at least 4
        assert len(llm.call_log) >= 4

    def test_investigation_notes_accumulated(self) -> None:
        result = _make_agent(llm=MockLLMProvider()).investigate(_make_incident())
        assert len(result.investigation_notes) >= 1


class TestMockLLMProvider:
    def test_default_response_returned_on_no_match(self) -> None:
        llm = MockLLMProvider(default_response='{"key": "val"}')
        assert '"key"' in llm.complete([{"role": "user", "content": "unrelated"}])

    def test_keyword_response_matched(self) -> None:
        llm = MockLLMProvider(responses={"database": '{"found": "db"}'})
        assert '"found"' in llm.complete([{"role": "user", "content": "check database errors"}])

    def test_call_log_tracks_all_calls(self) -> None:
        llm = MockLLMProvider()
        llm.complete([{"role": "user", "content": "call 1"}])
        llm.complete([{"role": "user", "content": "call 2"}])
        assert len(llm.call_log) == 2

    def test_model_name_is_mock(self) -> None:
        assert MockLLMProvider().model_name == "mock-llm"

    def test_satisfies_protocol(self) -> None:
        from rca_agent.agents.llm_provider import LLMProvider
        assert isinstance(MockLLMProvider(), LLMProvider)
