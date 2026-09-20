"""POST /incidents/investigate — the core RCA investigation endpoint.

Request flow
------------
1. Validate the ``InvestigationRequest`` payload (Pydantic).
2. Build an ``Incident`` domain object.
3. Wire providers (log, git, memory) from settings.
4. Run ``RCAAgent.investigate()``.
5. Map the ``RCAResult`` to ``InvestigationResponse``.
6. Persist the report to ``FileReportStore``.
7. Return the response.

Provider wiring
---------------
The endpoint builds *real* providers when paths are configured, or falls back
to no-op stubs so the service always starts successfully even without a live
Git repo or log directory.

Security
--------
The ``require_api_key`` dependency enforces auth when
``settings.api_key_enabled`` is True.

Rate limiting
-------------
The ``@limiter.limit()`` decorator applies the investigation-specific rate
limit from ``settings.rate_limit_investigate``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status

from rca_agent.api.auth import require_api_key
from rca_agent.api.middleware import limiter
from rca_agent.config.settings import settings
from rca_agent.agents.llm_provider import MockLLMProvider, ResilientLLMProvider
from rca_agent.agents.rca_agent import RCAAgent
from rca_agent.memory.graph_provider import InMemoryGraphProvider
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.report_store import FileReportStore
from rca_agent.memory.vector_provider import TfidfVectorProvider
from rca_agent.models.incident import (
    Incident,
    IncidentStatus,
    IncidentSymptom,
    Severity,
)
from rca_agent.models.investigation import (
    EvidenceSummary,
    InvestigationReport,
    InvestigationRequest,
    InvestigationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/incidents", tags=["investigation"])
_report_store = FileReportStore()


# ---------------------------------------------------------------------------
# No-op stubs for when providers are not configured
# ---------------------------------------------------------------------------

class _NoOpLogProvider:
    """Returns empty results when no log directory is configured."""
    from rca_agent.models.log_entry import LogSearchQuery, LogSearchResult
    def search_logs(self, q):
        from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
        return LogSearchResult(entries=[], query=q)
    def get_logs_by_trace_id(self, tid):
        from rca_agent.models.log_entry import LogSearchResult, LogSearchQuery
        return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
    def get_log_by_id(self, lid): return None


class _NoOpGitProvider:
    """Returns empty results when no repository is configured."""
    def get_recent_commits(self, limit=20): return []
    def get_commit(self, cid): raise ValueError(f"No git provider configured: {cid}")
    def get_diff(self, cid): return []
    def get_files_changed(self, cid): return []
    def search_commits(self, q): return []
    def get_commits_between(self, s, e): return []


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _build_log_provider(application: str):
    """Return a log provider based on settings, or the no-op stub."""
    if not settings.log_dir or not Path(settings.log_dir).exists():
        logger.debug("Log directory not configured or missing — using no-op provider")
        return _NoOpLogProvider()
    try:
        from rca_agent.providers.local_log_provider import LocalLogProvider
        return LocalLogProvider(
            log_path=settings.log_dir,
            max_lines=settings.log_max_lines,
        )
    except Exception as exc:
        logger.warning("Could not build log provider: %s", exc)
        return _NoOpLogProvider()


def _build_git_provider():
    """Return a git provider based on settings, or the no-op stub."""
    if not settings.git_repo_path or not Path(settings.git_repo_path).exists():
        logger.debug("Git repo path not configured or missing — using no-op provider")
        return _NoOpGitProvider()
    try:
        from rca_agent.providers.local_git_provider import LocalGitProvider
        return LocalGitProvider(
            repo_path=settings.git_repo_path,
            max_commits=settings.git_max_commits,
        )
    except Exception as exc:
        logger.warning("Could not build git provider: %s", exc)
        return _NoOpGitProvider()


def _build_trace_provider():
    """Return a JaegerTraceProvider when Jaeger is configured, or None.

    Returns None when ``settings.jaeger_base_url`` is empty so the
    investigation continues without trace evidence rather than failing.
    """
    if not settings.jaeger_base_url or not settings.jaeger_base_url.strip():
        logger.debug("Jaeger base URL not configured — trace provider disabled")
        return None
    try:
        from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider
        return JaegerTraceProvider(
            base_url=settings.jaeger_base_url,
            timeout_seconds=settings.jaeger_timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Could not build trace provider: %s", exc)
        return None


def _build_llm_provider():
    """Return the configured LLM provider wrapped with resilience."""
    if settings.llm_provider == "mock":
        inner = MockLLMProvider()
    elif settings.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI  # type: ignore[import]
            from rca_agent.agents.llm_provider import LangchainLLMProvider
            model = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            inner = LangchainLLMProvider(model)
        except ImportError:
            logger.warning("langchain_openai not installed — falling back to mock LLM")
            inner = MockLLMProvider()
    else:
        logger.warning("Unknown LLM provider %r — using mock", settings.llm_provider)
        inner = MockLLMProvider()

    return ResilientLLMProvider(
        inner=inner,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        wait_seconds=settings.llm_retry_wait_seconds,
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/investigate",
    response_model=InvestigationResponse,
    status_code=status.HTTP_200_OK,
    summary="Investigate an incident",
    description=(
        "Run an AI-powered root cause analysis investigation for the given incident. "
        "Returns a structured RCA report with evidence, confidence, and next steps."
    ),
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(settings.rate_limit_investigate)
async def investigate(
    request: Request,         # SlowAPI requires this name exactly
    body: InvestigationRequest,
) -> InvestigationResponse:
    """Investigate an incident and return a structured RCA."""
    investigation_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    logger.info(
        "Investigation started",
        extra={
            "investigation_id": investigation_id,
            "incident_id": body.incident_id,
            "application": body.application,
            "environment": body.environment,
        },
    )

    # Build domain incident
    incident = Incident(
        incident_id=body.incident_id,
        application=body.application,
        environment=body.environment,
        title=body.derive_title(),
        description=body.description,
        severity=Severity(body.severity),
        status=IncidentStatus.OPEN,
        start_time=body.start_time,
        end_time=body.end_time,
        affected_services=body.affected_services,
        symptoms=[
            IncidentSymptom(description=s, observed_at=body.start_time)
            for s in body.symptoms
        ],
    )

    # Wire providers
    llm = _build_llm_provider()
    log_provider = _build_log_provider(body.application)
    git_provider = _build_git_provider()
    trace_provider = _build_trace_provider()
    memory = IncidentMemory(
        graph=InMemoryGraphProvider(),
        vector=TfidfVectorProvider(),
        similarity_threshold=settings.vector_similarity_threshold,
    )

    agent = RCAAgent(
        llm=llm,
        log_provider=log_provider,
        git_provider=git_provider,
        memory=memory,
        max_log_entries=settings.agent_max_log_entries,
        max_commits=settings.agent_max_commits,
        similar_incidents_top_k=settings.agent_similar_incidents_top_k,
        trace_provider=trace_provider,
    )

    # Run investigation with provider failure isolation
    t_start = time.perf_counter()
    try:
        result = agent.investigate(incident)
    except Exception as exc:
        logger.exception(
            "Investigation failed",
            extra={"investigation_id": investigation_id, "error": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Investigation failed: {exc}",
        ) from exc
    duration = time.perf_counter() - t_start

    # Token tracking
    tokens_estimated = 0
    if hasattr(llm, "total_chars"):
        tokens_estimated = llm.total_chars // 4

    # Map to API response
    evidence_summaries = [
        EvidenceSummary(
            evidence_type=getattr(ev, "evidence_type", type(ev).__name__).value
            if hasattr(getattr(ev, "evidence_type", None), "value")
            else str(getattr(ev, "evidence_type", "unknown")),
            source=getattr(ev, "source", ""),
            description=getattr(ev, "description", "")[:200],
            statement_type=getattr(
                getattr(ev, "statement_type", None), "value", str(getattr(ev, "statement_type", "UNKNOWN"))
            ),
            source_ref=getattr(ev, "source_ref", None),
            confidence=getattr(ev, "confidence", None),
        )
        for ev in result.structured_evidence[:20]
    ]

    response = InvestigationResponse(
        investigation_id=investigation_id,
        incident_id=result.incident_id,
        status=result.status.value,
        summary=result.summary,
        root_cause=result.root_cause.summary if result.root_cause else None,
        root_cause_category=result.root_cause.category if result.root_cause else None,
        confidence=result.confidence,
        affected_services=result.affected_services,
        evidence=evidence_summaries,
        similar_incidents=result.similar_incidents,
        contributing_factors=result.contributing_factors,
        unknowns=result.unknowns,
        recommended_next_steps=result.recommended_next_steps,
        investigation_started_at=started_at,
        prompt_version=settings.prompt_version,
        tokens_estimated=tokens_estimated,
        duration_seconds=round(duration, 3),
    )

    # Persist report
    try:
        report = InvestigationReport(
            investigation_id=investigation_id,
            incident_id=body.incident_id,
            request=body,
            response=response,
            prompt_version=settings.prompt_version,
            model_name=llm.model_name,
        )
        _report_store.save(report)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not persist investigation report: %s", exc)

    logger.info(
        "Investigation complete",
        extra={
            "investigation_id": investigation_id,
            "status": result.status.value,
            "confidence": result.confidence,
            "duration_seconds": duration,
            "tokens_estimated": tokens_estimated,
        },
    )

    return response
