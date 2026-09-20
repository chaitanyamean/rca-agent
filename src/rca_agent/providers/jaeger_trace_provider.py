"""JaegerTraceProvider — queries Jaeger's HTTP API for distributed traces.

Design goals
------------
* **Backend-agnostic output**: all returned objects are generic ``Trace``
  and ``Span`` models — no Jaeger types leak to callers.
* **No hardcoded addresses**: ``base_url`` comes from configuration
  (``settings.jaeger_base_url`` / ``JAEGER_BASE_URL`` env var).
* **Graceful degradation**: network errors and 404s return ``None``
  or empty results — they never raise at the call-site.  The agent
  continues without traces rather than crashing.
* **Read-only**: all operations are GET requests.  No writes, no side effects.
* **Application-agnostic**: knows nothing about RKE or any other target app.

Jaeger HTTP API used
--------------------
The provider uses Jaeger's public JSON query API (v1):

    GET {base_url}/api/traces/{traceID}
        → single trace (all spans)

    GET {base_url}/api/traces?service=…&start=…&end=…&limit=…&operation=…&tags=…
        → list of traces matching the query (times are Unix microseconds)

    GET {base_url}/api/services
        → list of known service names (health check)

These endpoints are available in Jaeger v1.x all-in-one and have been
stable since Jaeger 1.6.  They are not the gRPC/Protobuf query API.

Timestamp handling
------------------
Jaeger stores and returns all timestamps as **microseconds** since the Unix
epoch.  The ``Span`` model validators accept these and convert to UTC datetime.

Port layout (default)
---------------------
``JAEGER_BASE_URL`` should point to the Jaeger UI/query HTTP port, which is
16686 by default.  The OTLP ingest ports (4317/4318) are separate and are
not used by this provider.

Example configuration::

    JAEGER_BASE_URL=http://localhost:16686   # local development
    JAEGER_BASE_URL=http://jaeger:16686      # Docker Compose network
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from rca_agent.models.trace_models import (
    Span,
    SpanEvent,
    Trace,
    TraceSearchQuery,
    TraceSearchResult,
    TraceStatus,
)

logger = logging.getLogger(__name__)

# How long to wait for a Jaeger HTTP response before giving up (seconds).
_DEFAULT_TIMEOUT = 10.0


class JaegerTraceProvider:
    """Queries the Jaeger HTTP API and returns generic trace/span models.

    Parameters
    ----------
    base_url:
        Root URL of the Jaeger query service, e.g. ``http://localhost:16686``.
        Do not include a trailing slash.
    timeout_seconds:
        HTTP request timeout in seconds.  Defaults to 10.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = _DEFAULT_TIMEOUT,
    ) -> None:
        # Strip trailing slash so we can always write f"{self._base}/{path}"
        self._base = base_url.rstrip("/")
        self._timeout = timeout_seconds
        # Shared synchronous httpx client (re-used across calls for efficiency)
        self._client = httpx.Client(
            base_url=self._base,
            timeout=self._timeout,
            headers={"Accept": "application/json"},
        )

    # ------------------------------------------------------------------
    # Public API (satisfies TraceProvider protocol)
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> Trace | None:
        """Return the complete trace for *trace_id*, or ``None`` if not found.

        Calls ``GET /api/traces/{traceID}``.
        """
        if not trace_id or not trace_id.strip():
            logger.debug("JaegerTraceProvider.get_trace: empty trace_id")
            return None

        url = f"/api/traces/{trace_id.strip()}"
        try:
            resp = self._client.get(url)
        except httpx.RequestError as exc:
            logger.warning(
                "JaegerTraceProvider.get_trace: network error for %s: %s", trace_id, exc
            )
            return None

        if resp.status_code == 404:
            logger.debug("JaegerTraceProvider.get_trace: trace %s not found", trace_id)
            return None

        if resp.status_code != 200:
            logger.warning(
                "JaegerTraceProvider.get_trace: unexpected status %d for %s",
                resp.status_code,
                trace_id,
            )
            return None

        try:
            payload = resp.json()
        except Exception as exc:
            logger.warning("JaegerTraceProvider.get_trace: JSON parse error: %s", exc)
            return None

        raw_traces = payload.get("data", [])
        if not raw_traces:
            logger.debug("JaegerTraceProvider.get_trace: empty data for %s", trace_id)
            return None

        return self._parse_trace(raw_traces[0])

    def search_traces(self, query: TraceSearchQuery) -> TraceSearchResult:
        """Return traces matching *query*.

        Calls ``GET /api/traces`` with the appropriate query parameters.
        Results are ordered newest-first.
        """
        params = self._build_search_params(query)

        try:
            resp = self._client.get("/api/traces", params=params)
        except httpx.RequestError as exc:
            logger.warning("JaegerTraceProvider.search_traces: network error: %s", exc)
            return TraceSearchResult(traces=[], total=0, query=query)

        if resp.status_code != 200:
            logger.warning(
                "JaegerTraceProvider.search_traces: unexpected status %d", resp.status_code
            )
            return TraceSearchResult(traces=[], total=0, query=query)

        try:
            payload = resp.json()
        except Exception as exc:
            logger.warning("JaegerTraceProvider.search_traces: JSON parse error: %s", exc)
            return TraceSearchResult(traces=[], total=0, query=query)

        raw_traces = payload.get("data", [])
        traces: list[Trace] = []
        for raw in raw_traces:
            try:
                trace = self._parse_trace(raw)
                if trace:
                    traces.append(trace)
            except Exception as exc:  # noqa: BLE001
                logger.debug("JaegerTraceProvider.search_traces: skipping malformed trace: %s", exc)

        # Apply local post-filters that Jaeger's API doesn't support natively
        if query.status == TraceStatus.ERROR:
            traces = [t for t in traces if t.error_spans]
        if query.min_duration_ms is not None:
            traces = [t for t in traces if t.total_duration_ms >= query.min_duration_ms]

        return TraceSearchResult(traces=traces, total=len(traces), query=query)

    def get_trace_spans(self, trace_id: str) -> list[Span]:
        """Return all spans for *trace_id*, or an empty list if not found."""
        trace = self.get_trace(trace_id)
        if trace is None:
            return []
        return trace.spans

    def get_failed_spans(self, trace_id: str) -> list[Span]:
        """Return only the ERROR-status spans for *trace_id*."""
        trace = self.get_trace(trace_id)
        if trace is None:
            return []
        return trace.error_spans

    # ------------------------------------------------------------------
    # Internal: Jaeger → generic model translation
    # ------------------------------------------------------------------

    def _parse_trace(self, raw: dict[str, Any]) -> Trace | None:
        """Parse a single Jaeger trace object into a generic ``Trace``."""
        trace_id = raw.get("traceID", "")
        if not trace_id:
            return None

        # Jaeger includes a ``processes`` dict mapping processID → {serviceName, tags}
        processes: dict[str, dict[str, Any]] = raw.get("processes", {})

        spans: list[Span] = []
        for raw_span in raw.get("spans", []):
            try:
                span = self._parse_span(raw_span, trace_id, processes)
                if span:
                    spans.append(span)
            except Exception as exc:  # noqa: BLE001
                logger.debug("JaegerTraceProvider._parse_trace: skipping span: %s", exc)

        return Trace(trace_id=trace_id, spans=spans)

    def _parse_span(
        self,
        raw: dict[str, Any],
        trace_id: str,
        processes: dict[str, dict[str, Any]],
    ) -> Span | None:
        """Parse a Jaeger span dict into a generic ``Span``."""
        span_id = raw.get("spanID", "")
        if not span_id:
            return None

        # ---- Service name ------------------------------------------------
        process_id = raw.get("processID", "")
        process = processes.get(process_id, {})
        service_name = process.get("serviceName", "unknown")

        # ---- Timestamps (Jaeger uses microseconds) -----------------------
        start_us: int = raw.get("startTime", 0)
        duration_us: int = raw.get("duration", 0)

        start_time = datetime.fromtimestamp(start_us / 1_000_000.0, tz=timezone.utc)
        end_time = datetime.fromtimestamp(
            (start_us + duration_us) / 1_000_000.0, tz=timezone.utc
        )
        duration_ms = duration_us / 1_000.0

        # ---- Parent span ID (first CHILD_OF reference) -------------------
        parent_span_id: str | None = None
        for ref in raw.get("references", []):
            if ref.get("refType") == "CHILD_OF":
                parent_span_id = ref.get("spanID") or None
                break

        # ---- Attributes (Jaeger calls them "tags") -----------------------
        attributes: dict[str, Any] = {}
        for tag in raw.get("tags", []):
            key = tag.get("key", "")
            value = tag.get("value")
            if key:
                attributes[key] = value

        # Also include process-level tags
        for tag in process.get("tags", []):
            key = tag.get("key", "")
            value = tag.get("value")
            if key and key not in attributes:
                attributes[key] = value

        # ---- Status (derive from tags / http.status_code) ----------------
        status, status_message = self._derive_status(raw, attributes)

        # ---- Events (Jaeger "logs") ---------------------------------------
        events: list[SpanEvent] = []
        for log_entry in raw.get("logs", []):
            try:
                ev = self._parse_span_event(log_entry)
                if ev:
                    events.append(ev)
            except Exception:  # noqa: BLE001
                pass

        return Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            service_name=service_name,
            operation_name=raw.get("operationName", "unknown"),
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            status=status,
            status_message=status_message,
            attributes=attributes,
            events=events,
        )

    @staticmethod
    def _derive_status(
        raw: dict[str, Any],
        attributes: dict[str, Any],
    ) -> tuple[TraceStatus, str | None]:
        """Derive the OTel TraceStatus from Jaeger span tags.

        Jaeger doesn't have a native OTel status concept; it uses a mix of:
        - ``error=true`` tag (set by Jaeger's own instrumentation)
        - ``otel.status_code`` tag (set by OTEL SDK)
        - ``http.status_code`` tag (>= 500 = error)
        """
        status_message: str | None = None

        # OTel SDK sets otel.status_code / otel.status_description explicitly
        otel_status = str(attributes.get("otel.status_code", "")).upper()
        if otel_status == "ERROR":
            status_message = str(attributes.get("otel.status_description", "")) or None
            return TraceStatus.ERROR, status_message
        if otel_status == "OK":
            return TraceStatus.OK, None

        # Jaeger's own error tag
        if attributes.get("error") is True or str(attributes.get("error", "")).lower() == "true":
            # Look for exception in span logs
            for log_entry in raw.get("logs", []):
                for field in log_entry.get("fields", []):
                    if field.get("key") == "error.object":
                        status_message = str(field.get("value", ""))[:200]
                        break
                    if field.get("key") == "message":
                        status_message = str(field.get("value", ""))[:200]
            return TraceStatus.ERROR, status_message

        # HTTP status code
        http_status = attributes.get("http.status_code")
        if http_status is not None:
            try:
                code = int(http_status)
                if code >= 500:
                    return TraceStatus.ERROR, f"HTTP {code}"
                if code >= 400:
                    return TraceStatus.ERROR, f"HTTP {code}"
                return TraceStatus.OK, None
            except (ValueError, TypeError):
                pass

        return TraceStatus.UNSET, None

    @staticmethod
    def _parse_span_event(raw: dict[str, Any]) -> SpanEvent | None:
        """Parse a Jaeger span log entry into a ``SpanEvent``."""
        timestamp_us = raw.get("timestamp", 0)
        if not timestamp_us:
            return None

        fields: dict[str, Any] = {}
        event_name = "log"
        for field in raw.get("fields", []):
            key = field.get("key", "")
            value = field.get("value")
            if key == "event":
                event_name = str(value)
            elif key:
                fields[key] = value

        return SpanEvent(
            name=event_name,
            timestamp=timestamp_us,  # validator handles microsecond conversion
            attributes=fields,
        )

    # ------------------------------------------------------------------
    # Internal: query parameter construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_search_params(query: TraceSearchQuery) -> dict[str, Any]:
        """Convert a ``TraceSearchQuery`` to Jaeger HTTP query parameters."""
        params: dict[str, Any] = {"limit": query.limit}

        if query.service:
            params["service"] = query.service

        if query.operation:
            params["operation"] = query.operation

        if query.start_time:
            # Jaeger expects microseconds since Unix epoch
            start = query.start_time
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            params["start"] = int(start.timestamp() * 1_000_000)

        if query.end_time:
            end = query.end_time
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            params["end"] = int(end.timestamp() * 1_000_000)

        if query.min_duration_ms is not None:
            # Jaeger minDuration is in microseconds as a string like "1000us" or "1ms"
            params["minDuration"] = f"{int(query.min_duration_ms * 1000)}us"

        if query.tags:
            # Jaeger accepts tags as JSON: '{"key":"value"}'
            import json
            params["tags"] = json.dumps(query.tags)

        return params

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client.  Call when done if not using
        the provider as a context manager."""
        self._client.close()

    def __enter__(self) -> "JaegerTraceProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
