"""NoOpTraceProvider — satisfies TraceProvider when tracing is not configured.

Purpose
-------
When an application does not have distributed tracing infrastructure, the RCA
Agent should proceed without trace evidence rather than failing.

``NoOpTraceProvider`` satisfies the ``TraceProvider`` protocol and returns
structured empty results.  It:

* Does NOT throw unhandled exceptions.
* Does NOT crash the RCA workflow.
* Does NOT invent traces or fabricate trace data.
* Returns clearly empty ``TraceSearchResult`` objects with ``total=0``.

The difference between ``NoOpTraceProvider`` and ``None``
--------------------------------------------------------
Passing ``trace_provider=None`` to the agent currently means "tracing was not
configured, skip it silently."  ``NoOpTraceProvider`` is an explicit object
that can also carry a reason string — useful when tracing was configured but
the provider could not be instantiated (e.g. httpx not installed, bad URL).

Selecting the right provider
-----------------------------
Use the factory ``build_trace_provider()`` which reads ``settings.trace_provider_type``:

    ``trace_provider_type = "jaeger"``  → ``JaegerTraceProvider``
    ``trace_provider_type = "none"``    → ``NoOpTraceProvider``
    ``trace_provider_type = "auto"``    → ``JaegerTraceProvider`` if ``jaeger_base_url``
                                          is set, else ``NoOpTraceProvider``
    ``trace_provider_type = "mock"``    → ``MockTraceProvider`` (tests only)

Usage::

    from rca_agent.providers.noop_trace_provider import NoOpTraceProvider

    provider = NoOpTraceProvider(reason="Application does not use distributed tracing")
    result = provider.search_traces(TraceSearchQuery())
    assert result.traces == []
    assert result.total == 0
"""

from __future__ import annotations

import logging

from rca_agent.models.trace_models import (
    Span,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
)

logger = logging.getLogger(__name__)


class NoOpTraceProvider:
    """A trace provider that always returns empty results.

    Satisfies the ``TraceProvider`` structural protocol so it can be injected
    anywhere a real provider is expected.

    Parameters
    ----------
    reason:
        Human-readable explanation for why tracing is not available.
        Logged at DEBUG level on each call.
    """

    def __init__(self, reason: str = "tracing not configured") -> None:
        self._reason = reason
        logger.debug("NoOpTraceProvider created: %s", reason)

    @property
    def reason(self) -> str:
        """The reason tracing is unavailable."""
        return self._reason

    # ------------------------------------------------------------------
    # TraceProvider protocol implementation
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Trace | None:
        """Always returns ``None`` — no traces are available."""
        logger.debug("NoOpTraceProvider.get_trace(%r): %s", trace_id, self._reason)
        return None

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        """Always returns an empty result — no traces are available."""
        logger.debug("NoOpTraceProvider.search_traces: %s", self._reason)
        return TraceSearchResult(traces=[], total=0, query=query)

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Always returns an empty list — no traces are available."""
        logger.debug("NoOpTraceProvider.get_trace_spans(%r): %s", trace_id, self._reason)
        return []

    def get_failed_spans(self, trace_id: str) -> list[Span]:
        """Always returns an empty list — no traces are available."""
        logger.debug("NoOpTraceProvider.get_failed_spans(%r): %s", trace_id, self._reason)
        return []
