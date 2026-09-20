"""RCAAgent — the public entry point for automated incident investigation.

This is the only class external code should import from the agents package.
It wires together all dependencies and exposes a single ``investigate()``
method that runs the full LangGraph workflow and returns a typed ``RCAResult``.

Usage::

    from rca_agent.agents.rca_agent import RCAAgent
    from rca_agent.agents.llm_provider import MockLLMProvider
    from rca_agent.memory.graph_provider import InMemoryGraphProvider
    from rca_agent.memory.vector_provider import TfidfVectorProvider
    from rca_agent.memory.incident_memory import IncidentMemory

    agent = RCAAgent(
        llm=MockLLMProvider(),
        log_provider=my_log_provider,
        git_provider=my_git_provider,
        memory=IncidentMemory(
            graph=InMemoryGraphProvider(),
            vector=TfidfVectorProvider(),
        ),
    )
    result = agent.investigate(incident)
    print(result.status, result.confidence, result.summary)
"""

from __future__ import annotations

import logging

from rca_agent.agents.llm_provider import LLMProvider
from rca_agent.agents.rca_graph import build_rca_graph
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.memory.rca_memory_writer import RCAMemoryWriter
from rca_agent.models.incident import Incident
from rca_agent.models.rca_result import RCAResult, RCAStatus
from rca_agent.providers.base import GitProvider, LogProvider, TraceProvider

logger = logging.getLogger(__name__)


class RCAAgent:
    """Automated Root Cause Analysis agent.

    The agent runs a fixed, auditable LangGraph workflow that:
    1. Understands the incident context.
    2. Retrieves evidence from logs and Git.
    3. Searches historical incidents in memory.
    4. Correlates evidence, proposes, and validates root causes.
    5. Returns a fully structured ``RCAResult``.

    Parameters
    ----------
    llm:
        LLM provider.  Use ``MockLLMProvider`` for tests.
    log_provider:
        Satisfies ``LogProvider`` protocol.
    git_provider:
        Satisfies ``GitProvider`` protocol.
    memory:
        Populated ``IncidentMemory`` with historical incidents.
    max_log_entries:
        Cap on log entries retrieved per tool call.
    max_commits:
        Cap on recent commits retrieved.
    similar_incidents_top_k:
        Number of similar historical incidents to retrieve.
    trace_provider:
        Optional ``TraceProvider`` (e.g. ``JaegerTraceProvider``).
    auto_store_rca:
        When True (default), the completed RCA is automatically persisted
        to ``memory`` via ``RCAMemoryWriter`` at the end of ``investigate()``.
        Set to False to disable automatic persistence (e.g. for dry-runs).
        Note: auto_store_rca is automatically forced to False when
        memory_enabled=False to prevent Memory-OFF results from polluting
        the historical corpus.
    memory_enabled:
        When True (default), Node 5 queries historical incident memory and
        injects retrieved incidents as contextual evidence.
        When False, Node 5 is replaced with a no-op — no historical evidence
        is retrieved, and auto_store_rca is implicitly suppressed.
        This is the primary experimental variable for Phase 3.
    """

    def __init__(
        self,
        llm: LLMProvider,
        log_provider: LogProvider,
        git_provider: GitProvider,
        memory: IncidentMemory,
        *,
        max_log_entries: int = 50,
        max_commits: int = 20,
        similar_incidents_top_k: int = 5,
        trace_provider: TraceProvider | None = None,
        auto_store_rca: bool = True,
        memory_enabled: bool = True,
    ) -> None:
        self._graph = build_rca_graph(
            llm=llm,
            log_provider=log_provider,
            git_provider=git_provider,
            memory=memory,
            max_log_entries=max_log_entries,
            max_commits=max_commits,
            similar_incidents_top_k=similar_incidents_top_k,
            trace_provider=trace_provider,
            memory_enabled=memory_enabled,
        )
        self._llm = llm
        self._memory = memory
        self._memory_enabled = memory_enabled
        # When memory is disabled, auto-store is also suppressed: we must not
        # write a Memory-OFF investigation result into memory, because it would
        # pollute the historical corpus used by the Memory-ON condition.
        self._auto_store_rca = auto_store_rca and memory_enabled
        self._memory_writer = RCAMemoryWriter(memory)

    def investigate(self, incident: Incident) -> RCAResult:
        """Run the full RCA investigation workflow for *incident*.

        Parameters
        ----------
        incident:
            The incident to investigate.  Must have at minimum ``title``,
            ``application``, ``environment``, and ``start_time``.

        Returns
        -------
        RCAResult
            A fully structured root cause analysis.  ``status`` and
            ``confidence`` indicate the reliability of the findings.
            ``unknowns`` lists what could not be determined.
        """
        logger.info(
            "RCAAgent: starting investigation for incident %s (%s)",
            incident.incident_id,
            incident.title,
        )

        initial_state: dict = {"incident": incident, "memory_enabled": self._memory_enabled}

        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.exception("RCAAgent: graph execution failed: %s", exc)
            # Return a safe fallback result rather than propagating
            return RCAResult(
                incident_id=incident.incident_id,
                status=RCAStatus.INSUFFICIENT_EVIDENCE,
                summary=f"Investigation failed due to an internal error: {exc}",
                confidence=0.0,
                unknowns=["Graph execution error — see investigation_notes."],
                investigation_notes=[f"FATAL: {exc}"],
            )

        result: RCAResult | None = final_state.get("rca_result")

        if result is None:
            logger.warning("RCAAgent: graph completed but no RCAResult produced.")
            return RCAResult(
                incident_id=incident.incident_id,
                status=RCAStatus.INSUFFICIENT_EVIDENCE,
                summary="Investigation completed but no result was produced.",
                confidence=0.0,
                unknowns=["No RCAResult generated — check agent logs."],
            )

        logger.info(
            "RCAAgent: investigation complete — status=%s confidence=%.2f",
            result.status.value,
            result.confidence,
        )

        # Persist the completed RCA to long-term incident memory so it is
        # available as historical evidence for future investigations.
        if self._auto_store_rca:
            try:
                self._memory_writer.store(incident, result)
            except Exception as exc:  # noqa: BLE001
                # Memory write failures must never break the investigation
                # response — the RCA result is still returned to the caller.
                logger.warning(
                    "RCAAgent: could not persist RCA to memory (non-fatal): %s", exc
                )

        return result
