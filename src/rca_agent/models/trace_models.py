"""Generic distributed trace models for the RCA Agent.

These models are backend-agnostic — they represent the logical structure
of an OpenTelemetry trace regardless of whether the source is Jaeger,
Zipkin, Grafana Tempo, or any other compatible backend.

Design principles
-----------------
* No Jaeger-specific field names leak into these models.
* All timestamps are normalised to UTC-aware ``datetime`` objects.
* ``duration_ms`` is always a non-negative float (milliseconds).
* ``attributes`` and ``events`` are plain dicts/lists so the models do not
  depend on any particular tracing SDK schema.
* ``TraceStatus`` mirrors the OpenTelemetry span status codes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TraceStatus(str, Enum):
    """OpenTelemetry span status codes.

    See: https://opentelemetry.io/docs/concepts/signals/traces/#span-status
    """
    UNSET = "UNSET"    # No status set — the default.
    OK = "OK"          # The operation completed successfully.
    ERROR = "ERROR"    # The operation failed.


class SpanEvent(BaseModel):
    """A timestamped log event recorded on a span.

    Corresponds to OTel span events (formerly "logs").
    """
    name: str = Field(description="Event name (e.g. 'exception', 'db.query.start').")
    timestamp: datetime = Field(description="UTC timestamp when the event was recorded.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value pairs associated with the event.",
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, (int, float)):
            # microseconds → seconds (Jaeger uses microsecond epoch timestamps)
            seconds = v / 1_000_000.0 if v > 1e12 else v
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)
        raise ValueError(f"Cannot parse timestamp from: {v!r}")


class Span(BaseModel):
    """A single operation within a distributed trace.

    Maps directly to an OpenTelemetry span.  Field names follow OTel
    conventions; the ``JaegerTraceProvider`` translates from Jaeger's
    internal naming when constructing these objects.
    """

    trace_id: str = Field(description="W3C-format trace identifier (16-byte hex string).")
    span_id: str = Field(description="Span identifier (8-byte hex string).")
    parent_span_id: str | None = Field(
        default=None,
        description="Parent span identifier, or None for root spans.",
    )
    service_name: str = Field(description="Name of the service that produced this span.")
    operation_name: str = Field(
        description="Name of the operation (e.g. 'POST /api/sales/cash', 'sale.create').",
    )
    start_time: datetime = Field(description="UTC start time of the span.")
    end_time: datetime = Field(description="UTC end time of the span.")
    duration_ms: float = Field(
        ge=0.0,
        description="Span duration in milliseconds.",
    )
    status: TraceStatus = Field(
        default=TraceStatus.UNSET,
        description="Span status: UNSET, OK, or ERROR.",
    )
    status_message: str | None = Field(
        default=None,
        description="Human-readable status message (populated on ERROR spans).",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Span attributes (key-value pairs).  Common keys: "
            "http.method, http.route, http.status_code, db.system, "
            "db.statement, db.operation, sale.type, tenant.id."
        ),
    )
    events: list[SpanEvent] = Field(
        default_factory=list,
        description="Timestamped events recorded during the span (e.g. exception details).",
    )

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _ensure_utc(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
            return v.astimezone(timezone.utc)
        if isinstance(v, (int, float)):
            # Jaeger stores timestamps as microseconds since Unix epoch
            seconds = v / 1_000_000.0 if v > 1e12 else v
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(timezone.utc)
        raise ValueError(f"Cannot parse timestamp from: {v!r}")

    @model_validator(mode="after")
    def _compute_duration(self) -> "Span":
        """Compute duration_ms from start/end if it was not provided."""
        if self.duration_ms == 0.0 and self.end_time > self.start_time:
            delta_ms = (self.end_time - self.start_time).total_seconds() * 1000.0
            object.__setattr__(self, "duration_ms", delta_ms)
        return self

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def is_root(self) -> bool:
        """True if this span has no parent (i.e. it is the trace entry point)."""
        return self.parent_span_id is None

    @property
    def is_error(self) -> bool:
        """True if the span status is ERROR."""
        return self.status == TraceStatus.ERROR

    @property
    def is_slow(self) -> bool:
        """True if the span duration exceeds 1 000 ms (1 second).

        This threshold matches typical database statement timeout warnings
        and is conservative enough to avoid false positives on normal spans.
        """
        return self.duration_ms >= 1_000.0

    @property
    def exception_message(self) -> str | None:
        """Return the first exception message recorded as a span event, if any."""
        for event in self.events:
            if "exception" in event.name.lower():
                return (
                    event.attributes.get("exception.message")
                    or event.attributes.get("exception.type")
                    or event.name
                )
        return None


class Trace(BaseModel):
    """A complete distributed trace — a tree of ``Span`` objects.

    A trace represents the full journey of a single request through all
    services and components that handled it.
    """

    trace_id: str = Field(description="Unique trace identifier shared by all spans.")
    spans: list[Span] = Field(default_factory=list, description="All spans in this trace.")

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def root_span(self) -> Span | None:
        """Return the root span (the one with no parent), or None."""
        for span in self.spans:
            if span.is_root:
                return span
        return None

    @property
    def error_spans(self) -> list[Span]:
        """Return all spans with status ERROR."""
        return [s for s in self.spans if s.is_error]

    @property
    def slow_spans(self) -> list[Span]:
        """Return all spans with duration ≥ 1 000 ms."""
        return [s for s in self.spans if s.is_slow]

    @property
    def service_names(self) -> list[str]:
        """Return a deduplicated, sorted list of service names in this trace."""
        return sorted({s.service_name for s in self.spans})

    @property
    def total_duration_ms(self) -> float:
        """Return the end-to-end duration of the trace in milliseconds.

        Computed as the time between the earliest span start and the
        latest span end across all spans.
        """
        if not self.spans:
            return 0.0
        earliest = min(s.start_time for s in self.spans)
        latest = max(s.end_time for s in self.spans)
        return (latest - earliest).total_seconds() * 1000.0

    def get_span(self, span_id: str) -> Span | None:
        """Return the span with the given ID, or None."""
        return next((s for s in self.spans if s.span_id == span_id), None)

    def children_of(self, span_id: str) -> list[Span]:
        """Return all direct child spans of the given span."""
        return [s for s in self.spans if s.parent_span_id == span_id]

    def format_summary(self) -> str:
        """Return a compact human-readable summary for use in RCA prompts."""
        root = self.root_span
        lines = [
            f"Trace {self.trace_id[:16]}  "
            f"duration={self.total_duration_ms:.0f}ms  "
            f"spans={len(self.spans)}  "
            f"errors={len(self.error_spans)}  "
            f"services={','.join(self.service_names)}"
        ]
        if root:
            lines.append(
                f"  Root: [{root.status.value}] {root.operation_name}  "
                f"{root.duration_ms:.0f}ms"
            )
        for span in self.error_spans[:3]:
            lines.append(
                f"  ERROR: {span.service_name}/{span.operation_name}  "
                f"{span.duration_ms:.0f}ms"
                + (f"  msg={span.status_message[:80]}" if span.status_message else "")
            )
        for span in self.slow_spans[:3]:
            if not span.is_error:  # already reported above
                lines.append(
                    f"  SLOW: {span.service_name}/{span.operation_name}  "
                    f"{span.duration_ms:.0f}ms"
                )
        return "\n".join(lines)


class TraceSearchQuery(BaseModel):
    """Parameters for a trace search request.

    All fields are optional — absent fields are not used as filters.
    """

    service: str | None = Field(
        default=None,
        description="Filter to traces containing at least one span from this service.",
    )
    operation: str | None = Field(
        default=None,
        description="Filter to traces containing a span with this operation name.",
    )
    start_time: datetime | None = Field(
        default=None,
        description="Only return traces that started at or after this time (UTC).",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Only return traces that started at or before this time (UTC).",
    )
    min_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Only return traces with total duration ≥ this value (ms).",
    )
    status: TraceStatus | None = Field(
        default=None,
        description="Filter to traces with at least one span of this status.",
    )
    tags: dict[str, str] | None = Field(
        default=None,
        description="Filter to traces where any span has all of these tag key-value pairs.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of traces to return.",
    )


class TraceSearchResult(BaseModel):
    """The result of a trace search operation."""

    traces: list[Trace] = Field(default_factory=list)
    total: int = Field(default=0, description="Total number of matching traces found.")
    query: TraceSearchQuery = Field(description="The query that produced this result.")
