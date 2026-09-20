"""Observability capability and evidence availability models.

These models answer two related but distinct questions:

1. **ObservabilityCapabilities** — which evidence sources are *configured*
   for this application/investigation?  (static, set at startup)

2. **EvidenceSourceStatus** — what actually happened when we tried to retrieve
   evidence from each source during a specific investigation?  (dynamic,
   recorded per-run)

The distinction matters:

* ``logs_available=True`` but ``EvidenceAvailability.FAILED`` means the log
  provider is configured but the log directory was unreadable during this run.

* ``traces_available=False`` and ``EvidenceAvailability.NOT_CONFIGURED`` means
  there is no tracing infrastructure — this is expected and normal.

* ``traces_available=True`` but ``EvidenceAvailability.FAILED`` means Jaeger
  was configured but unreachable — this is an unexpected failure that should be
  surfaced in the RCA report.

Design rules
------------
* These models are application-agnostic.  RKE is just one example.
* ``metrics_available`` and ``deployment_available`` exist as capability flags
  but their providers are NOT implemented in this phase.
* The models use the project's standard Pydantic conventions.
* No RKE-specific logic lives here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Availability status for a single evidence source during one investigation
# ---------------------------------------------------------------------------

class EvidenceAvailability(str, Enum):
    """The retrieval status of a single evidence source during an investigation.

    These values are mutually exclusive and cover all possible outcomes.
    """

    AVAILABLE = "available"
    """Evidence was successfully retrieved and is non-empty."""

    AVAILABLE_EMPTY = "available_empty"
    """Provider is configured and reachable but returned zero evidence items."""

    NOT_CONFIGURED = "not_configured"
    """No provider is configured for this source.  This is expected and normal
    when the application does not have that observability capability."""

    FAILED = "failed"
    """The provider is configured but the retrieval attempt failed (network
    error, timeout, permission error, etc.).  This is unexpected."""

    SKIPPED = "skipped"
    """The source was intentionally skipped (e.g. max_commits=0)."""


class EvidenceSourceStatus(BaseModel):
    """The observed status of one evidence source during an investigation."""

    source: str = Field(
        description="Evidence source name: 'logs', 'traces', 'git', 'memory'."
    )
    availability: EvidenceAvailability = Field(
        description="What happened when this source was queried."
    )
    item_count: int = Field(
        default=0,
        ge=0,
        description="Number of evidence items retrieved (0 when not available).",
    )
    detail: str = Field(
        default="",
        description=(
            "Human-readable detail: error message, number of items retrieved, "
            "or explanation of why the source was not queried."
        ),
    )

    @property
    def produced_evidence(self) -> bool:
        """True if this source contributed at least one evidence item."""
        return self.availability == EvidenceAvailability.AVAILABLE and self.item_count > 0

    @property
    def is_failure(self) -> bool:
        """True only when the source is configured but retrieval failed."""
        return self.availability == EvidenceAvailability.FAILED

    def format_line(self) -> str:
        """Return a single-line human-readable status for inclusion in reports."""
        icon = {
            EvidenceAvailability.AVAILABLE: "✓",
            EvidenceAvailability.AVAILABLE_EMPTY: "○",
            EvidenceAvailability.NOT_CONFIGURED: "–",
            EvidenceAvailability.FAILED: "✗",
            EvidenceAvailability.SKIPPED: "·",
        }.get(self.availability, "?")
        detail = f" ({self.detail})" if self.detail else ""
        return f"  {icon} {self.source}: {self.availability.value}{detail}"


# ---------------------------------------------------------------------------
# Static capability declaration (configured at startup)
# ---------------------------------------------------------------------------

class ObservabilityCapabilities(BaseModel):
    """Which evidence sources are configured for an application.

    These are *static* declarations — they describe the observability
    infrastructure available for the target application, not the dynamic
    outcome of a specific investigation.

    Fields marked ``False`` map to ``EvidenceAvailability.NOT_CONFIGURED``.
    Fields marked ``True`` may still produce ``AVAILABLE_EMPTY`` or
    ``FAILED`` outcomes at runtime if the provider is reachable but returns
    no data or encounters an error.

    Application-agnostic — RKE is just one example target.
    """

    # Core evidence sources
    logs_available: bool = Field(
        default=True,
        description=(
            "True when a log provider is configured (directory, API, etc.). "
            "A fully observable application always has logs."
        ),
    )
    traces_available: bool = Field(
        default=False,
        description=(
            "True when a distributed tracing backend (Jaeger, Tempo, etc.) "
            "is configured.  OpenTelemetry instrumentation is optional."
        ),
    )
    git_available: bool = Field(
        default=True,
        description=(
            "True when a Git repository path is configured and accessible. "
            "Useful for correlating incidents with code changes."
        ),
    )

    # Future evidence sources (providers NOT yet implemented)
    metrics_available: bool = Field(
        default=False,
        description=(
            "True when a metrics backend (Prometheus, CloudWatch, etc.) "
            "is configured.  NOT IMPLEMENTED — reserved for a future phase."
        ),
    )
    deployment_available: bool = Field(
        default=False,
        description=(
            "True when deployment records are accessible. "
            "NOT IMPLEMENTED — reserved for a future phase."
        ),
    )

    @property
    def active_sources(self) -> list[str]:
        """Return the names of all enabled evidence sources."""
        sources = []
        if self.logs_available:
            sources.append("logs")
        if self.traces_available:
            sources.append("traces")
        if self.git_available:
            sources.append("git")
        if self.metrics_available:
            sources.append("metrics")
        if self.deployment_available:
            sources.append("deployments")
        return sources

    @property
    def total_sources(self) -> int:
        return len(self.active_sources)

    def describe(self) -> str:
        """Return a compact human-readable capability summary."""
        parts = [
            f"logs={'✓' if self.logs_available else '✗'}",
            f"traces={'✓' if self.traces_available else '✗'}",
            f"git={'✓' if self.git_available else '✗'}",
        ]
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Dynamic evidence summary (recorded per investigation run)
# ---------------------------------------------------------------------------

class InvestigationEvidenceSummary(BaseModel):
    """Records the actual evidence retrieval outcomes for one investigation.

    This is the *dynamic* counterpart to ``ObservabilityCapabilities``.
    It is populated by Node 2 (retrieve_evidence) and consumed by Node 9
    (generate_rca) to produce accurate provenance in the final report.

    Having both a static capability declaration and a dynamic summary
    means the RCA report can distinguish:
    - "traces were not configured" from "trace retrieval failed"
    - "git returned no commits" from "git provider was not configured"
    """

    sources: list[EvidenceSourceStatus] = Field(default_factory=list)

    def add(
        self,
        source: str,
        availability: EvidenceAvailability,
        item_count: int = 0,
        detail: str = "",
    ) -> None:
        self.sources.append(EvidenceSourceStatus(
            source=source,
            availability=availability,
            item_count=item_count,
            detail=detail,
        ))

    def get(self, source: str) -> EvidenceSourceStatus | None:
        return next((s for s in self.sources if s.source == source), None)

    @property
    def has_any_evidence(self) -> bool:
        """True if at least one source produced evidence items."""
        return any(s.produced_evidence for s in self.sources)

    @property
    def failed_sources(self) -> list[EvidenceSourceStatus]:
        """Sources that are configured but failed to return evidence."""
        return [s for s in self.sources if s.is_failure]

    @property
    def active_sources(self) -> list[EvidenceSourceStatus]:
        """Sources that produced at least one evidence item."""
        return [s for s in self.sources if s.produced_evidence]

    def format_provenance(self) -> str:
        """Return a human-readable evidence provenance block for RCA reports."""
        lines = ["Evidence Sources:"]
        for status in self.sources:
            lines.append(status.format_line())
        if self.failed_sources:
            lines.append(
                f"\nNote: {len(self.failed_sources)} configured source(s) failed "
                f"to return evidence — investigation may be incomplete."
            )
        return "\n".join(lines)

    def to_unknowns(self) -> list[str]:
        """Generate unknowns entries for the RCA report based on source status."""
        unknowns: list[str] = []
        for status in self.sources:
            if status.availability == EvidenceAvailability.NOT_CONFIGURED:
                # Not configured is expected — mention briefly
                unknowns.append(
                    f"{status.source.capitalize()} evidence was not configured "
                    f"for this investigation."
                )
            elif status.availability == EvidenceAvailability.FAILED:
                unknowns.append(
                    f"{status.source.capitalize()} evidence retrieval failed: "
                    f"{status.detail or 'unknown error'}. "
                    f"RCA may be incomplete."
                )
        return unknowns
