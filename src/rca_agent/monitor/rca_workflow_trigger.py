"""RCAWorkflowTrigger — translates a detected error Trace into an RCA investigation.

Responsibility
--------------
When the monitor detects a new error trace, this component:
  1. Constructs an ``Incident`` domain object from the trace metadata.
  2. Builds all required providers (LLM, log, git, trace, memory) using the
     same factory functions already present in the existing API route.
  3. Invokes ``RCAAgent.investigate()`` — the EXISTING RCA workflow.
  4. Logs and returns the result.

This component deliberately reuses the existing provider factories and
``RCAAgent`` without modification.  No second RCA pipeline is introduced.

Memory ON/OFF
-------------
The ``memory_enabled`` flag from ``MonitorConfig`` is forwarded to the
``RCAAgent`` constructor, preserving the Phase 3 experiment switch.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rca_agent.models.incident import Incident, IncidentStatus, IncidentSymptom, Severity
from rca_agent.models.rca_result import RCAResult
from rca_agent.models.trace_models import Trace
from rca_agent.monitor.config import MonitorConfig

logger = logging.getLogger(__name__)


class RCAWorkflowTrigger:
    """Invokes the existing RCA workflow for a newly detected error trace.

    Parameters
    ----------
    config:
        Monitor configuration.  ``environment`` and ``memory_enabled`` are
        forwarded to the incident and agent respectively.
    """

    def __init__(self, config: MonitorConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trigger(self, trace: Trace) -> RCAResult:
        """Build an Incident from *trace* and run the RCA workflow.

        Parameters
        ----------
        trace:
            A fully parsed error trace from ``JaegerTraceProvider``.

        Returns
        -------
        RCAResult
            The structured RCA produced by the existing LangGraph workflow.
        """
        incident = self._build_incident(trace)
        logger.info(
            "RCAWorkflowTrigger: triggering investigation "
            "incident_id=%s trace_id=%s service=%s",
            incident.incident_id,
            trace.trace_id,
            "/".join(trace.service_names),
        )

        agent = self._build_agent(trace)
        result = agent.investigate(incident)

        logger.info(
            "RCAWorkflowTrigger: investigation complete "
            "incident_id=%s status=%s confidence=%.2f",
            incident.incident_id,
            result.status.value,
            result.confidence,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: incident construction
    # ------------------------------------------------------------------

    def _build_incident(self, trace: Trace) -> Incident:
        """Construct an ``Incident`` from trace metadata.

        All values come from the trace itself — no hardcoded service names,
        incident IDs, or scenario-specific logic.
        """
        root = trace.root_span
        service = root.service_name if root else (trace.service_names[0] if trace.service_names else "unknown")
        operation = root.operation_name if root else "unknown"
        start_time = root.start_time if root else datetime.now(timezone.utc)

        # Build a human-readable title from what the trace tells us
        error_spans = trace.error_spans
        if error_spans:
            first_error = error_spans[0]
            status_hint = (
                first_error.status_message
                or first_error.exception_message
                or "error"
            )
            title = f"[{service}] {operation} — {status_hint[:100]}"
        else:
            title = f"[{service}] {operation} — error trace detected"

        symptoms: list[IncidentSymptom] = []
        for span in error_spans[:5]:
            desc = (
                f"ERROR span '{span.operation_name}' in {span.service_name}"
                + (f": {span.status_message[:120]}" if span.status_message else "")
            )
            symptoms.append(
                IncidentSymptom(description=desc, observed_at=span.start_time, service=span.service_name)
            )

        return Incident(
            incident_id=f"auto-{trace.trace_id[:16]}",
            application=service,
            environment=self._config.environment,
            title=title,
            description=(
                f"Automatically detected by RCA-Agent Jaeger monitor. "
                f"Trace ID: {trace.trace_id}. "
                f"Services: {', '.join(trace.service_names)}. "
                f"Error spans: {len(error_spans)}."
            ),
            severity=Severity.HIGH,
            status=IncidentStatus.OPEN,
            start_time=start_time,
            affected_services=trace.service_names,
            symptoms=symptoms,
            trace_id=trace.trace_id,  # primary evidence anchor
        )

    # ------------------------------------------------------------------
    # Internal: agent construction (reuses existing factories)
    # ------------------------------------------------------------------

    def _build_agent(self, trace: Trace):
        """Build a fully wired ``RCAAgent`` using the existing provider factories."""
        from rca_agent.agents.llm_factory import build_llm_provider
        from rca_agent.agents.rca_agent import RCAAgent
        from rca_agent.memory.graph_provider import InMemoryGraphProvider
        from rca_agent.memory.incident_memory import IncidentMemory
        from rca_agent.memory.vector_provider import TfidfVectorProvider
        from rca_agent.config.settings import settings
        from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider
        from rca_agent.providers.noop_trace_provider import NoOpTraceProvider

        llm = build_llm_provider()

        # Reuse the existing log/git factories from the API route
        log_provider = self._build_log_provider()
        git_provider = self._build_git_provider()

        # For the investigation, point at the same Jaeger so the workflow
        # can fetch the full trace details and correlated spans.
        jaeger_url = self._config.jaeger_url
        if jaeger_url:
            trace_provider = JaegerTraceProvider(
                base_url=jaeger_url,
                timeout_seconds=self._config.jaeger_timeout_seconds,
            )
        else:
            trace_provider = NoOpTraceProvider(reason="jaeger_url not configured in MonitorConfig")

        memory = IncidentMemory(
            graph=InMemoryGraphProvider(),
            vector=TfidfVectorProvider(),
            similarity_threshold=settings.vector_similarity_threshold,
        )

        return RCAAgent(
            llm=llm,
            log_provider=log_provider,
            git_provider=git_provider,
            memory=memory,
            max_log_entries=settings.agent_max_log_entries,
            max_commits=settings.agent_max_commits,
            similar_incidents_top_k=settings.agent_similar_incidents_top_k,
            trace_provider=trace_provider,
            memory_enabled=self._config.memory_enabled,
            auto_store_rca=True,
        )

    @staticmethod
    def _build_log_provider():
        """Return a real log provider if configured, otherwise a no-op stub.

        Source selection (``settings.log_source``):
        - ``'docker'`` — DockerLogProvider reading container stdout/stderr.
                         Requires Docker to be running and the container to be
                         named by ``settings.log_docker_container``.
        - ``'file'`` (default) — LocalLogProvider reading NDJSON files from
                         ``settings.log_dir``.
        """
        from rca_agent.config.settings import settings
        from rca_agent.models.log_entry import LogSearchQuery, LogSearchResult

        class _NoOpLog:
            def search_logs(self, q): return LogSearchResult(entries=[], query=q)
            def get_logs_by_trace_id(self, tid):
                return LogSearchResult(entries=[], query=LogSearchQuery(trace_id=tid))
            def get_log_by_id(self, lid): return None

        source = (settings.log_source or "file").lower().strip()
        application_fallback = settings.log_docker_container

        if source == "docker":
            try:
                from rca_agent.providers.docker_log_provider import DockerLogProvider
                container = settings.log_docker_container or application_fallback
                provider = DockerLogProvider(
                    container_name=container,
                    since_minutes=settings.log_docker_since_minutes,
                )
                logger.info(
                    "RCAWorkflowTrigger: using DockerLogProvider "
                    "container=%s since=%dm",
                    container, settings.log_docker_since_minutes,
                )
                return provider
            except Exception as exc:
                logger.warning(
                    "RCAWorkflowTrigger: could not build DockerLogProvider: %s "
                    "— falling back to no-op", exc
                )
                return _NoOpLog()

        # Default: file-based NDJSON provider
        if not settings.log_dir or not Path(settings.log_dir).exists():
            return _NoOpLog()
        try:
            from rca_agent.providers.local_log_provider import LocalLogProvider
            return LocalLogProvider(
                log_path=settings.log_dir,
                max_lines=settings.log_max_lines,
            )
        except Exception as exc:
            logger.warning("RCAWorkflowTrigger: could not build log provider: %s — using no-op", exc)
            return _NoOpLog()

    @staticmethod
    def _build_git_provider():
        """Return a real git provider if configured, otherwise a no-op stub."""
        from rca_agent.config.settings import settings

        class _NoOpGit:
            def get_recent_commits(self, limit=20): return []
            def get_commit(self, cid): raise ValueError(f"no git: {cid}")
            def get_diff(self, cid): return []
            def get_files_changed(self, cid): return []
            def search_commits(self, q): return []
            def get_commits_between(self, s, e): return []

        if not settings.git_repo_path or not Path(settings.git_repo_path).exists():
            return _NoOpGit()
        try:
            from rca_agent.providers.local_git_provider import LocalGitProvider
            return LocalGitProvider(
                repo_path=settings.git_repo_path,
                max_commits=settings.git_max_commits,
            )
        except Exception as exc:
            logger.warning("RCAWorkflowTrigger: could not build git provider: %s — using no-op", exc)
            return _NoOpGit()
