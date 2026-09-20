"""TraceErrorDetector — deterministic error classification for Trace objects.

Design contract
---------------
* This component makes NO LLM calls.
* Classification is entirely deterministic: given the same Trace, it always
  returns the same result.
* It reuses the existing ``Span.is_error`` property, which in turn uses the
  ``TraceStatus`` enum produced by ``JaegerTraceProvider._derive_status()``.
  That method already handles:
    - ``otel.status_code = ERROR``
    - ``error = true`` span tag
    - ``http.status_code >= 500``
  So this component just aggregates those signals at the Trace level.

A ``Trace`` is considered an error trace when any of these are true:
  1. At least one span has ``status == TraceStatus.ERROR``.
  2. The root span has ``status == TraceStatus.ERROR``.  (Redundant with 1
     but explicit for clarity.)

Successful traces (all spans UNSET or OK) are ignored.

Infrastructure noise filtering
-------------------------------
Some ERROR traces are not application failures — they are outbound spans
produced by the OpenTelemetry exporter inside the monitored service when it
fails to deliver spans to the collector.  These MUST NOT trigger RCA because:
  * The application never logs under these trace IDs.
  * The error is at the telemetry layer, not the business layer.
  * Triggering RCA on them produces ``insufficient_evidence`` every time.

The discriminator is deterministic (no LLM):
  bare HTTP method (POST/GET/…) in ``operation_name``   (no URL path)
  AND ``network.peer.address`` present in span attributes  (outbound peer)
  AND ``client.address`` absent from span attributes       (not an inbound request)
"""

from __future__ import annotations

from rca_agent.models.trace_models import Trace, TraceStatus

# Bare HTTP method names — operation names that consist of only these words
# (no path component) are candidates for infrastructure-noise classification.
_BARE_HTTP_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"})


class TraceErrorDetector:
    """Classifies whether a Trace represents a failure.

    This class has no state — all methods are effectively static.  It exists
    as a class (rather than a module-level function) so it can be easily
    replaced with a test double in unit tests.
    """

    def is_error_trace(self, trace: Trace) -> bool:
        """Return True if *trace* contains at least one ERROR span.

        Parameters
        ----------
        trace:
            A fully parsed ``Trace`` object (from ``JaegerTraceProvider``
            or a test fake).

        Returns
        -------
        bool
            True  — one or more spans have ``status == TraceStatus.ERROR``.
            False — all spans are ``UNSET`` or ``OK``.
        """
        return any(span.status == TraceStatus.ERROR for span in trace.spans)

    def is_infrastructure_noise(self, trace: Trace) -> bool:
        """Return True if *trace* is an infrastructure/exporter span that must not trigger RCA.

        Deterministic check — no LLM involvement.

        An ERROR trace is classified as infrastructure noise when the root span
        (or, if there is no root span, the first error span) satisfies ALL of:

        1. ``operation_name`` is a bare HTTP method with no path component
           (e.g. ``"POST"``, ``"GET"``).  Application spans always carry a
           route such as ``"POST /api/orders"``.

        2. ``network.peer.address`` is present in span attributes.
           This indicates an *outbound* connection to a peer (e.g. the OTEL
           Collector at ``172.x.x.x``).

        3. ``client.address`` is absent from span attributes.
           Application spans recording *inbound* requests always carry this
           attribute (the caller's IP / hostname).

        This combination uniquely identifies spans produced by the
        OpenTelemetry SDK's exporter (OkHttp / gRPC) when it fails to
        deliver telemetry to the collector.  No application-layer span
        satisfies all three conditions simultaneously.

        The check is generic — it does not reference specific IP addresses,
        service names, exception class names, or incident IDs.

        Parameters
        ----------
        trace:
            A fully parsed ``Trace`` object.

        Returns
        -------
        bool
            True  — the trace is infrastructure/exporter noise; skip it.
            False — the trace is a candidate application error.
        """
        # Use root span; fall back to first error span if no explicit root
        span = trace.root_span
        if span is None:
            error_spans = trace.error_spans
            if not error_spans:
                return False
            span = error_spans[0]

        attrs = span.attributes

        # Condition 1: bare HTTP method (no path in operation_name)
        op = span.operation_name.strip()
        if op not in _BARE_HTTP_METHODS:
            return False

        # Condition 2: outbound peer address present
        has_peer = (
            attrs.get("network.peer.address") is not None
            or attrs.get("net.peer.name") is not None
            or attrs.get("net.peer.ip") is not None
            or attrs.get("server.address") is not None    # OTel semantic conventions >= 1.23
        )
        if not has_peer:
            return False

        # Condition 3: inbound client address absent
        has_client = attrs.get("client.address") is not None
        if has_client:
            return False

        return True

    def noise_reason(self, trace: Trace) -> str:
        """Return a human-readable reason string for an infrastructure-noise trace.

        Only meaningful when ``is_infrastructure_noise()`` returns True.
        Used for diagnostic logging.
        """
        span = trace.root_span or (trace.error_spans[0] if trace.error_spans else None)
        if span is None:
            return "no spans"
        op = span.operation_name.strip()
        peer = (
            span.attributes.get("network.peer.address")
            or span.attributes.get("net.peer.name")
            or span.attributes.get("net.peer.ip")
            or span.attributes.get("server.address")
            or "unknown"
        )
        return (
            f"outbound HTTP span: operation={op!r} "
            f"network.peer.address={peer} "
            f"client.address=absent — likely OTEL exporter failure"
        )

    def error_summary(self, trace: Trace) -> str:
        """Return a compact one-line summary of the errors in *trace*.

        Used for logging.  Returns an empty string if no errors found.
        """
        error_spans = [s for s in trace.spans if s.status == TraceStatus.ERROR]
        if not error_spans:
            return ""

        parts: list[str] = []
        for span in error_spans[:3]:  # cap to avoid huge log lines
            msg = span.status_message or span.exception_message or "ERROR"
            parts.append(f"{span.service_name}/{span.operation_name}: {msg[:80]}")

        suffix = f" (+{len(error_spans) - 3} more)" if len(error_spans) > 3 else ""
        return "; ".join(parts) + suffix
