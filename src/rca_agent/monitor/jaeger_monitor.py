"""JaegerMonitor — autonomous Jaeger polling loop for RCA-Agent.

Responsibility
--------------
Polls Jaeger on a configurable interval, detects new error traces
deterministically, deduplicates across overlapping polls, and triggers the
existing RCA workflow for each newly discovered failure.

Architecture
------------

    JaegerMonitor
         │
         │ polls via
         ▼
    JaegerTraceProvider        (existing — reused, not duplicated)
         │
         │ returns Trace objects
         ▼
    TraceErrorDetector         (deterministic — no LLM)
         │
         │ error traces only
         ▼
    ProcessedTraceRegistry     (deduplication)
         │
         │ new traces only
         ▼
    RCAWorkflowTrigger         (invokes existing RCAAgent)
         │
         ▼
    RCAResult

Failure handling
----------------
* Jaeger unavailable: logged as WARNING, poll skipped, monitoring continues.
* Malformed response: JaegerTraceProvider already handles this gracefully.
* RCA failure: logged as ERROR, monitoring continues.
* KeyboardInterrupt: clean shutdown with final log message.

Nothing in this file is hardcoded to a specific application, service name,
or incident ID.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from rca_agent.models.trace_models import Trace, TraceSearchQuery, TraceSearchResult
from rca_agent.monitor.config import MonitorConfig
from rca_agent.monitor.processed_trace_registry import ProcessedTraceRegistry
from rca_agent.monitor.rca_workflow_trigger import RCAWorkflowTrigger
from rca_agent.monitor.trace_error_detector import TraceErrorDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstractions (allow test doubles without network)
# ---------------------------------------------------------------------------

@runtime_checkable
class TraceQueryClient(Protocol):
    """Minimal interface used by ``JaegerMonitor`` to fetch traces.

    ``JaegerTraceProvider`` already satisfies this protocol — the monitor
    depends on the interface, not the concrete class.
    """

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        """Return traces matching *query*."""
        ...

    def list_services(self) -> list[str]:
        """Return all service names known to the backend.

        Implementations that don't support this may return an empty list.
        """
        ...


# ---------------------------------------------------------------------------
# Main monitor class
# ---------------------------------------------------------------------------

class JaegerMonitor:
    """Polls Jaeger for error traces and triggers RCA automatically.

    Parameters
    ----------
    client:
        Anything satisfying ``TraceQueryClient`` — normally a
        ``JaegerTraceProvider``.
    detector:
        Determines whether a Trace is an error.
    registry:
        Tracks already-processed trace IDs.
    trigger:
        Invokes the RCA workflow.
    config:
        Runtime configuration (URLs, intervals, services, etc.).
    """

    def __init__(
        self,
        client: TraceQueryClient,
        detector: TraceErrorDetector,
        registry: ProcessedTraceRegistry,
        trigger: RCAWorkflowTrigger,
        config: MonitorConfig,
    ) -> None:
        self._client = client
        self._detector = detector
        self._registry = registry
        self._trigger = trigger
        self._config = config

    # ------------------------------------------------------------------
    # Factory — build from project settings (used by the CLI entrypoint)
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls) -> "JaegerMonitor":
        """Construct a fully wired ``JaegerMonitor`` from project settings.

        This is the main factory used by ``scripts/monitor.py``.
        """
        from rca_agent.providers.jaeger_trace_provider import JaegerTraceProvider

        config = MonitorConfig.from_settings()

        # Extend JaegerTraceProvider with list_services so it satisfies the protocol
        client = _JaegerClientAdapter(
            JaegerTraceProvider(
                base_url=config.jaeger_url,
                timeout_seconds=config.jaeger_timeout_seconds,
            )
        )
        detector = TraceErrorDetector()
        registry = ProcessedTraceRegistry()
        trigger = RCAWorkflowTrigger(config)

        return cls(
            client=client,
            detector=detector,
            registry=registry,
            trigger=trigger,
            config=config,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the autonomous monitoring loop.

        Runs indefinitely until interrupted by ``KeyboardInterrupt`` (Ctrl-C)
        or until an unrecoverable fatal error occurs.

        All per-poll errors are caught and logged — the loop never terminates
        due to a Jaeger outage or a single failing RCA investigation.
        """
        cfg = self._config
        services_label = ", ".join(cfg.services) if cfg.services else "ALL"

        logger.info("=" * 60)
        logger.info("RCA-Agent Autonomous Jaeger Monitor starting")
        logger.info("  Jaeger URL          : %s", cfg.jaeger_url)
        logger.info("  Poll interval       : %.1f seconds", cfg.poll_interval_seconds)
        logger.info("  Lookback window     : %d seconds", cfg.lookback_seconds)
        logger.info("  Services monitored  : %s", services_label)
        logger.info("  Environment label   : %s", cfg.environment)
        logger.info("  Memory enabled      : %s", cfg.memory_enabled)
        logger.info("=" * 60)

        try:
            while True:
                self._poll_once()
                time.sleep(cfg.poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("RCA-Agent Monitor: shutting down (KeyboardInterrupt)")

    def run_once(self) -> int:
        """Execute a single poll cycle and return the number of RCAs triggered.

        Used by tests and one-shot scripts.
        """
        return self._poll_once()

    # ------------------------------------------------------------------
    # Internal: single poll cycle
    # ------------------------------------------------------------------

    def _poll_once(self) -> int:
        """Run one poll cycle.  Returns the number of RCA investigations triggered."""
        now = datetime.now(timezone.utc)
        since = now - timedelta(seconds=self._config.lookback_seconds)

        logger.debug(
            "JaegerMonitor: polling — lookback=%ds window=[%s, %s]",
            self._config.lookback_seconds,
            since.strftime("%H:%M:%S"),
            now.strftime("%H:%M:%S"),
        )

        services_to_query = self._config.services or self._discover_services()

        triggered = 0
        for service in (services_to_query if services_to_query else [None]):
            triggered += self._poll_service(service, since, now)

        return triggered

    def _discover_services(self) -> list[str]:
        """Ask Jaeger for all known service names.

        Returns an empty list on failure so the caller falls back to
        a service-agnostic query.
        """
        try:
            services = self._client.list_services()
            if services:
                logger.debug("JaegerMonitor: discovered %d services", len(services))
            return services
        except Exception as exc:  # noqa: BLE001
            logger.warning("JaegerMonitor: could not list services: %s", exc)
            return []

    def _poll_service(
        self,
        service: str | None,
        since: datetime,
        until: datetime,
    ) -> int:
        """Poll for error traces for *service* (or all services if None).

        Returns the number of RCA investigations triggered.
        """
        query = TraceSearchQuery(
            service=service,
            start_time=since,
            end_time=until,
            limit=50,
        )

        try:
            result = self._client.search_traces(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "JaegerMonitor: Jaeger query failed (service=%s): %s — will retry next poll",
                service or "ALL",
                exc,
            )
            return 0

        total = len(result.traces)
        if total:
            logger.debug(
                "JaegerMonitor: found %d trace(s) for service=%s",
                total,
                service or "ALL",
            )

        triggered = 0
        for trace in result.traces:
            if self._handle_trace(trace):
                triggered += 1

        return triggered

    def _handle_trace(self, trace: Trace) -> bool:
        """Process a single trace.  Returns True if RCA was triggered."""
        # Skip non-error traces (deterministic, no LLM)
        if not self._detector.is_error_trace(trace):
            return False

        # Skip already-processed traces (applies to both application errors
        # and infrastructure noise — prevents repeated logging on overlapping
        # lookback windows).
        if self._registry.is_processed(trace.trace_id):
            logger.debug(
                "JaegerMonitor: trace %s already processed — skipping",
                trace.trace_id[:16],
            )
            return False

        # Skip infrastructure/exporter noise — these are outbound spans from
        # the OTEL SDK failing to deliver to the collector.  They are NOT
        # application errors and have no application logs.
        # Mark as processed FIRST so subsequent polls skip them silently.
        if self._detector.is_infrastructure_noise(trace):
            self._registry.mark_processed(trace.trace_id)
            reason = self._detector.noise_reason(trace)
            logger.info(
                "JaegerMonitor: ignoring infrastructure/exporter error trace\n"
                "  Trace ID  : %s\n"
                "  Operation : %s\n"
                "  Reason    : %s",
                trace.trace_id,
                trace.root_span.operation_name if trace.root_span else "unknown",
                reason,
            )
            return False

        # New error trace — mark immediately to prevent concurrent re-processing
        self._registry.mark_processed(trace.trace_id)

        service = trace.root_span.service_name if trace.root_span else (
            trace.service_names[0] if trace.service_names else "unknown"
        )
        operation = trace.root_span.operation_name if trace.root_span else "unknown"
        error_summary = self._detector.error_summary(trace)

        logger.info(
            "JaegerMonitor: NEW error trace detected\n"
            "  Trace ID  : %s\n"
            "  Service   : %s\n"
            "  Operation : %s\n"
            "  Errors    : %s",
            trace.trace_id,
            service,
            operation,
            error_summary or "(see trace)",
        )
        logger.info("JaegerMonitor: triggering RCA workflow for trace %s", trace.trace_id[:16])

        # Invoke the existing RCA workflow — failures must NOT kill the monitor
        try:
            result = self._trigger.trigger(trace)
            logger.info(
                "JaegerMonitor: RCA complete — trace=%s status=%s confidence=%.2f",
                trace.trace_id[:16],
                result.status.value,
                result.confidence,
            )
            if result.root_cause:
                logger.info(
                    "JaegerMonitor: root cause — [%s] %s",
                    result.root_cause.category,
                    result.root_cause.summary[:120],
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "JaegerMonitor: RCA workflow failed for trace %s: %s — monitoring continues",
                trace.trace_id[:16],
                exc,
            )

        return True


# ---------------------------------------------------------------------------
# Adapter: wraps JaegerTraceProvider to add list_services()
# ---------------------------------------------------------------------------

class _JaegerClientAdapter:
    """Adds ``list_services()`` to ``JaegerTraceProvider`` so it satisfies
    ``TraceQueryClient`` without modifying the existing provider."""

    def __init__(self, provider) -> None:  # JaegerTraceProvider
        self._provider = provider

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        return self._provider.search_traces(query)

    def list_services(self) -> list[str]:
        """Call Jaeger ``GET /api/services`` and return the service name list."""
        try:
            resp = self._provider._client.get("/api/services")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [s for s in data if isinstance(s, str)]
        except Exception as exc:  # noqa: BLE001
            logger.debug("_JaegerClientAdapter.list_services: %s", exc)
        return []
