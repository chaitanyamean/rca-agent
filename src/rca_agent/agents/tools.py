"""Controlled investigation tools for the RCA Agent.

Security model
--------------
* Every tool accepts only typed, validated parameters.
* No tool exposes arbitrary shell access or filesystem paths.
* Tool functions are plain Python callables — not LangChain tools — so they
  cannot be invoked via prompt injection.
* Providers (LogProvider, GitProvider, IncidentMemory) are injected at
  agent construction time; tools receive them as explicit arguments.
* All results are typed Pydantic models — no raw strings from external sources
  are passed directly to the LLM.

Tool catalogue
--------------
1. search_logs             — search structured logs by criteria
2. get_logs_by_trace_id    — retrieve all logs for a trace
3. get_recent_commits      — fetch the N most recent Git commits
4. get_commit              — fetch a single commit by SHA
5. get_git_diff            — fetch per-file diffs for a commit
6. search_historical_incidents — semantic similarity search in memory
7. get_incident_by_id      — retrieve a historical incident node
8. get_service_information — return services affected by a historical incident
"""

from __future__ import annotations

import logging
from datetime import datetime

from rca_agent.models.git_models import Commit, CommitDiff, GitCommitQuery
from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.models.memory_models import IncidentNode, ServiceNode, SimilarIncident
from rca_agent.memory.incident_memory import IncidentMemory
from rca_agent.providers.base import GitProvider, LogProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log tools
# ---------------------------------------------------------------------------

def search_logs(
    log_provider: LogProvider,
    *,
    service: str | None = None,
    level: str | None = None,
    keyword: str | None = None,
    trace_id: str | None = None,
    endpoint: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    max_results: int = 50,
) -> LogSearchResult:
    """Search structured logs with validated filter parameters.

    Returns a ``LogSearchResult`` containing matching ``LogEntry`` objects.
    The result is capped at ``max_results`` entries.
    """
    query = LogSearchQuery(
        service=service,
        level=level,
        keyword=keyword,
        trace_id=trace_id,
        endpoint=endpoint,
        start_time=start_time,
        end_time=end_time,
    )
    result = log_provider.search_logs(query)
    # Cap results to avoid overwhelming the LLM context
    if len(result.entries) > max_results:
        from rca_agent.models.log_entry import LogSearchResult as LSR
        result = LSR(entries=result.entries[:max_results], query=result.query)
    logger.debug("search_logs: %d entries matched", result.total)
    return result


def get_logs_by_trace_id(
    log_provider: LogProvider,
    trace_id: str,
) -> LogSearchResult:
    """Return all log entries sharing a distributed trace ID."""
    result = log_provider.get_logs_by_trace_id(trace_id)
    logger.debug("get_logs_by_trace_id(%r): %d entries", trace_id, result.total)
    return result


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------

def get_recent_commits(
    git_provider: GitProvider,
    limit: int = 20,
) -> list[Commit]:
    """Return the *limit* most recent commits from the repository, newest first."""
    commits = git_provider.get_recent_commits(limit=limit)
    logger.debug("get_recent_commits: %d commits fetched", len(commits))
    return commits


def get_commit(
    git_provider: GitProvider,
    commit_id: str,
) -> Commit | None:
    """Return a single commit by its full or abbreviated SHA-1.

    Returns None (rather than raising) so the agent can handle missing
    commits gracefully without crashing the graph.
    """
    try:
        commit = git_provider.get_commit(commit_id)
        logger.debug("get_commit(%r): found", commit_id)
        return commit
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_commit(%r): not found — %s", commit_id, exc)
        return None


def get_git_diff(
    git_provider: GitProvider,
    commit_id: str,
) -> list[CommitDiff]:
    """Return per-file diffs for the given commit.

    Returns an empty list if the commit is not found.
    """
    try:
        diffs = git_provider.get_diff(commit_id)
        logger.debug("get_git_diff(%r): %d file diffs", commit_id, len(diffs))
        return diffs
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_git_diff(%r): error — %s", commit_id, exc)
        return []


# ---------------------------------------------------------------------------
# Memory / historical incident tools
# ---------------------------------------------------------------------------

def search_historical_incidents(
    memory: IncidentMemory,
    query: str,
    top_k: int = 5,
    exclude_ids: set[str] | None = None,
) -> list[SimilarIncident]:
    """Retrieve the most semantically similar historical incidents.

    Parameters
    ----------
    query:
        Free-form description of the current incident or its symptoms.
    top_k:
        Maximum number of results to return.
    exclude_ids:
        Incident IDs to exclude (e.g. the current incident).
    """
    results = memory.find_similar_incidents(
        query=query,
        top_k=top_k,
        exclude_ids=exclude_ids or set(),
    )
    logger.debug("search_historical_incidents: %d similar incidents found", len(results))
    return results


def get_incident_by_id(
    memory: IncidentMemory,
    incident_id: str,
) -> IncidentNode | None:
    """Return the graph node for a historical incident, or None if not found."""
    node = memory.get_incident(incident_id)
    logger.debug("get_incident_by_id(%r): %s", incident_id, "found" if node else "not found")
    return node


def get_service_information(
    memory: IncidentMemory,
    incident_id: str,
) -> list[ServiceNode]:
    """Return service nodes affected by the given historical incident."""
    services = memory.find_related_services(incident_id)
    logger.debug(
        "get_service_information(%r): %d services", incident_id, len(services)
    )
    return services


# ---------------------------------------------------------------------------
# Tool registry (for documentation and testing)
# ---------------------------------------------------------------------------

TOOL_NAMES: list[str] = [
    "search_logs",
    "get_logs_by_trace_id",
    "get_recent_commits",
    "get_commit",
    "get_git_diff",
    "search_historical_incidents",
    "get_incident_by_id",
    "get_service_information",
]
