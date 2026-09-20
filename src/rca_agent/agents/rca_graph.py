"""LangGraph StateGraph definition for the RCA investigation workflow.

The graph enforces a fixed, explicit sequence of nodes.  There are no
conditional branches that could send execution to unexpected places — the
workflow always follows the same path:

    START
      → understand_incident
      → retrieve_evidence
      → analyze_logs
      → inspect_git_changes
      → search_historical
      → correlate_evidence
      → generate_candidate
      → validate_candidate
      → generate_rca
    END

This is intentional.  A fixed workflow is easier to audit, test, and
reason about than a dynamically-routed agent that decides its own path.

Usage::

    from rca_agent.agents.rca_graph import build_rca_graph
    graph = build_rca_graph(llm, log_provider, git_provider, memory)
    result = graph.invoke({"incident": incident})
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from rca_agent.agents.llm_provider import LLMProvider
from rca_agent.agents.nodes import (
    make_analyze_logs_node,
    make_correlate_evidence_node,
    make_generate_candidate_node,
    make_generate_rca_node,
    make_inspect_git_node,
    make_retrieve_evidence_node,
    make_search_historical_node,
    make_understand_incident_node,
    make_validate_candidate_node,
)
from rca_agent.agents.state import InvestigationState
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.providers.base import GitProvider, LogProvider, TraceProvider


def build_rca_graph(
    llm: LLMProvider,
    log_provider: LogProvider,
    git_provider: GitProvider,
    memory: IncidentMemory,
    *,
    max_log_entries: int = 50,
    max_commits: int = 20,
    similar_incidents_top_k: int = 5,
    trace_provider: TraceProvider | None = None,
):
    """Construct and compile the RCA investigation LangGraph.

    Parameters
    ----------
    llm:
        Any ``LLMProvider`` implementation.
    log_provider:
        Any ``LogProvider`` implementation.
    git_provider:
        Any ``GitProvider`` implementation.
    memory:
        Populated ``IncidentMemory`` instance (graph + vector).
    max_log_entries:
        Cap on log entries fetched per tool call.
    max_commits:
        Cap on recent commits fetched.
    similar_incidents_top_k:
        Number of historical incidents to retrieve.
    trace_provider:
        Optional ``TraceProvider`` implementation.  When provided, Node 2
        retrieves distributed traces from the backend (e.g. Jaeger) and
        includes them as TRACE evidence in the correlation pipeline.

    Returns
    -------
    CompiledGraph
        A LangGraph compiled graph ready for ``invoke()`` or ``stream()``.
    """
    graph = StateGraph(InvestigationState)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    graph.add_node(
        "understand_incident",
        make_understand_incident_node(llm),
    )
    graph.add_node(
        "retrieve_evidence",
        make_retrieve_evidence_node(
            llm, log_provider, git_provider,
            max_log_entries=max_log_entries,
            max_commits=max_commits,
            trace_provider=trace_provider,
        ),
    )
    graph.add_node(
        "analyze_logs",
        make_analyze_logs_node(llm),
    )
    graph.add_node(
        "inspect_git_changes",
        make_inspect_git_node(llm, git_provider),
    )
    graph.add_node(
        "search_historical",
        make_search_historical_node(llm, memory, top_k=similar_incidents_top_k),
    )
    graph.add_node(
        "correlate_evidence",
        make_correlate_evidence_node(llm),
    )
    graph.add_node(
        "generate_candidate",
        make_generate_candidate_node(llm),
    )
    graph.add_node(
        "validate_candidate",
        make_validate_candidate_node(llm),
    )
    graph.add_node(
        "generate_rca",
        make_generate_rca_node(llm),
    )

    # ------------------------------------------------------------------
    # Wire edges — fixed linear sequence, no dynamic routing
    # ------------------------------------------------------------------
    graph.add_edge(START, "understand_incident")
    graph.add_edge("understand_incident", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "analyze_logs")
    graph.add_edge("analyze_logs", "inspect_git_changes")
    graph.add_edge("inspect_git_changes", "search_historical")
    graph.add_edge("search_historical", "correlate_evidence")
    graph.add_edge("correlate_evidence", "generate_candidate")
    graph.add_edge("generate_candidate", "validate_candidate")
    graph.add_edge("validate_candidate", "generate_rca")
    graph.add_edge("generate_rca", END)

    return graph.compile()
