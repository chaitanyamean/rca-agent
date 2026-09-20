"""Pydantic models for structured log entries and search operations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Core domain model
# ---------------------------------------------------------------------------

class LogEntry(BaseModel):
    """A single structured log line, validated against the expected schema.

    Fields map directly to the JSON keys emitted by target applications.
    Unknown extra fields are preserved in ``extra_fields`` so no information
    is silently dropped.
    """

    # Computed stable identifier — sha256 of the raw JSON line so the same
    # log event always has the same ID regardless of how it was loaded.
    id: str = Field(description="Stable SHA-256 hash of the raw log line.")

    timestamp: datetime = Field(description="Log event time (ISO-8601, UTC-normalised).")
    service: str = Field(description="Name of the originating service.")
    level: str = Field(description="Log severity level (INFO / WARN / ERROR / DEBUG …).")
    message: str = Field(description="Human-readable log message.")

    # Optional but common fields
    trace_id: str | None = Field(default=None, alias="traceId", description="Distributed trace ID.")
    endpoint: str | None = Field(default=None, description="HTTP endpoint path.")
    method: str | None = Field(default=None, description="HTTP method (GET, POST …).")
    status: int | None = Field(default=None, description="HTTP response status code.")
    exception: str | None = Field(default=None, description="Exception class name.")

    # Bucket for any extra fields present in the raw log
    extra_fields: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional fields from the raw log line.",
    )

    model_config = {"populate_by_name": True}

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_and_normalise_timestamp(cls, v: Any) -> datetime:
        """Accept ISO-8601 strings and ensure the result is timezone-aware UTC."""
        if isinstance(v, datetime):
            dt = v
        elif isinstance(v, str):
            # Replace trailing Z with +00:00 for fromisoformat compatibility
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        else:
            raise ValueError(f"Cannot parse timestamp: {v!r}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @field_validator("level", mode="before")
    @classmethod
    def _normalise_level(cls, v: Any) -> str:
        """Uppercase the level string for consistent filtering."""
        if isinstance(v, str):
            return v.upper()
        raise ValueError(f"Level must be a string, got {type(v)}")

    @classmethod
    def from_raw_line(cls, raw: str) -> "LogEntry":
        """Parse a single NDJSON line into a LogEntry.

        Raises
        ------
        ValueError
            If the line is not valid JSON or fails Pydantic validation.
        """
        raw = raw.strip()
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        # Compute a stable ID from the raw bytes
        entry_id = hashlib.sha256(raw.encode()).hexdigest()

        # Normalize alternative timestamp field names before validation.
        # RKE / Logstash / ECS use "@timestamp"; the model expects "timestamp".
        if "@timestamp" in data and "timestamp" not in data:
            data = dict(data)  # don't mutate the original
            data["timestamp"] = data.pop("@timestamp")

        # Separate known fields from extra
        known = {
            "id", "timestamp", "service", "level", "message",
            "traceId", "trace_id", "endpoint", "method", "status", "exception",
            "@timestamp", "@version",
        }
        extra = {k: v for k, v in data.items() if k not in known}

        return cls.model_validate({**data, "id": entry_id, "extra_fields": extra})


# ---------------------------------------------------------------------------
# Search / query models
# ---------------------------------------------------------------------------

class LogSearchQuery(BaseModel):
    """Parameters for filtering log entries."""

    start_time: datetime | None = Field(
        default=None,
        description="Include only entries at or after this time (UTC).",
    )
    end_time: datetime | None = Field(
        default=None,
        description="Include only entries at or before this time (UTC).",
    )
    service: str | None = Field(default=None, description="Filter by service name (exact match).")
    level: str | None = Field(default=None, description="Filter by log level (case-insensitive).")
    trace_id: str | None = Field(default=None, description="Filter by distributed trace ID.")
    endpoint: str | None = Field(default=None, description="Filter by HTTP endpoint (exact match).")
    keyword: str | None = Field(
        default=None,
        description="Case-insensitive substring search over the message field.",
    )

    @field_validator("level", mode="before")
    @classmethod
    def _normalise_level(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            return v.upper()
        raise ValueError(f"level must be a string, got {type(v)}")

    @model_validator(mode="after")
    def _validate_time_range(self) -> "LogSearchQuery":
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("start_time must be before end_time")
        return self


class LogSearchResult(BaseModel):
    """Container for search results returned by a log provider."""

    entries: list[LogEntry] = Field(default_factory=list)
    total: int = Field(default=0, description="Number of entries returned.")
    query: LogSearchQuery = Field(description="The query that produced this result.")

    @model_validator(mode="after")
    def _sync_total(self) -> "LogSearchResult":
        self.total = len(self.entries)
        return self
