"""Observability pipeline health checker for the RCA Agent.

Provides a lightweight mechanism to verify the full observability pipeline:

    Is Jaeger configured?
        ↓
    Is Jaeger reachable?
        ↓
    Can traces be queried?
        ↓
    Are traces being generated for the target service?

This is the P2 observability health check described in Phase 2 requirements.
It is intentionally simple — no large monitoring dashboard, just fast deterministic
checks that can be run from the CLI, a test suite, or a diagnostic endpoint.

Usage (CLI)::

    python -m rca_agent.providers.jaeger_health_checker

Usage (Python)::

    from rca_agent.providers.jaeger_health_checker import JaegerHealthChecker
    checker = JaegerHealthChecker("http://localhost:16686")
    report = checker.check_all(service_name="rke-backend")
    print(report.summary())
    if not report.overall_healthy:
        print("Observability pipeline has issues")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Result of a single health check."""
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"    # check not applicable given prior failure
    WARN = "WARN"    # passes but with a warning


@dataclass
class HealthCheckResult:
    """Result of one individual health check."""
    name: str
    status: CheckStatus
    detail: str = ""
    duration_ms: float = 0.0

    def format_line(self) -> str:
        icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "·", "WARN": "⚠"}.get(self.status.value, "?")
        detail = f" — {self.detail}" if self.detail else ""
        return f"  {icon} {self.name}: {self.status.value}{detail}"


@dataclass
class HealthReport:
    """Aggregated health check results for the full observability pipeline."""
    checks: list[HealthCheckResult] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def overall_healthy(self) -> bool:
        return all(c.status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)
                   for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)

    def summary(self) -> str:
        lines = [
            f"Observability Health Report — {self.checked_at.strftime('%Y-%m-%dT%H:%M:%SZ')}",
            f"Overall: {'HEALTHY ✓' if self.overall_healthy else 'UNHEALTHY ✗'}  "
            f"({self.pass_count} passed, {self.fail_count} failed)",
            "",
        ]
        for check in self.checks:
            lines.append(check.format_line())
        return "\n".join(lines)


class JaegerHealthChecker:
    """Checks the Jaeger + OTEL pipeline health.

    Parameters
    ----------
    jaeger_base_url:
        Base URL of the Jaeger HTTP query API, e.g. ``http://localhost:16686``.
    timeout_seconds:
        HTTP request timeout per check.
    """

    def __init__(
        self,
        jaeger_base_url: str,
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._url = jaeger_base_url.rstrip("/")
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(
        self,
        service_name: str = "rke-backend",
        collector_health_url: str | None = None,
    ) -> HealthReport:
        """Run all health checks and return an aggregated report.

        Parameters
        ----------
        service_name:
            The service to look for in Jaeger (e.g. ``"rke-backend"``).
        collector_health_url:
            Optional OTEL Collector health endpoint.  When provided, an extra
            check verifies the collector is reachable.
        """
        report = HealthReport()

        # Check 1 — Jaeger URL configured
        if not self._url:
            report.checks.append(HealthCheckResult(
                "jaeger_configured",
                CheckStatus.FAIL,
                detail="JAEGER_BASE_URL is not set",
            ))
            # All subsequent checks are meaningless
            for name in ["jaeger_reachable", "jaeger_services", "service_traces"]:
                report.checks.append(HealthCheckResult(name, CheckStatus.SKIP,
                                                        detail="skipped — Jaeger not configured"))
            return report

        report.checks.append(HealthCheckResult(
            "jaeger_configured", CheckStatus.PASS,
            detail=f"url={self._url}",
        ))

        # Check 2 — Jaeger reachable
        reachable, reachable_ms, reachable_detail = self._check_jaeger_reachable()
        report.checks.append(HealthCheckResult(
            "jaeger_reachable",
            CheckStatus.PASS if reachable else CheckStatus.FAIL,
            detail=reachable_detail,
            duration_ms=reachable_ms,
        ))

        if not reachable:
            for name in ["jaeger_services", "service_traces"]:
                report.checks.append(HealthCheckResult(name, CheckStatus.SKIP,
                                                        detail="skipped — Jaeger unreachable"))
            if collector_health_url:
                report.checks.append(self._check_collector(collector_health_url))
            return report

        # Check 3 — Jaeger has services
        has_service, svc_ms, svc_detail = self._check_service_known(service_name)
        report.checks.append(HealthCheckResult(
            "jaeger_services",
            CheckStatus.PASS if has_service else CheckStatus.WARN,
            detail=svc_detail,
            duration_ms=svc_ms,
        ))

        # Check 4 — Recent traces exist for the service
        has_traces, trace_ms, trace_detail = self._check_recent_traces(service_name)
        report.checks.append(HealthCheckResult(
            "service_traces",
            CheckStatus.PASS if has_traces else CheckStatus.WARN,
            detail=trace_detail,
            duration_ms=trace_ms,
        ))

        # Check 5 (optional) — OTEL Collector reachable
        if collector_health_url:
            report.checks.append(self._check_collector(collector_health_url))

        return report

    def check_jaeger_reachable(self) -> bool:
        """Fast reachability test. Returns True if Jaeger responds."""
        ok, _, _ = self._check_jaeger_reachable()
        return ok

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_jaeger_reachable(self) -> tuple[bool, float, str]:
        import time
        t0 = time.perf_counter()
        try:
            import httpx
            resp = httpx.get(f"{self._url}/api/services", timeout=self._timeout)
            ms = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                return True, ms, f"responded {resp.status_code} in {ms:.0f}ms"
            return False, ms, f"unexpected status {resp.status_code}"
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            return False, ms, f"{type(exc).__name__}: {exc}"

    def _check_service_known(self, service_name: str) -> tuple[bool, float, str]:
        import time
        t0 = time.perf_counter()
        try:
            import httpx
            resp = httpx.get(f"{self._url}/api/services", timeout=self._timeout)
            ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                return False, ms, f"status {resp.status_code}"
            services = resp.json().get("data", [])
            if service_name in services:
                return True, ms, f"service '{service_name}' found ({len(services)} total)"
            return False, ms, (
                f"service '{service_name}' NOT found. "
                f"Known services: {services[:5]}"
            )
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            return False, ms, f"{type(exc).__name__}: {exc}"

    def _check_recent_traces(self, service_name: str) -> tuple[bool, float, str]:
        import time
        t0 = time.perf_counter()
        try:
            import httpx
            resp = httpx.get(
                f"{self._url}/api/traces",
                params={"service": service_name, "limit": 1},
                timeout=self._timeout,
            )
            ms = (time.perf_counter() - t0) * 1000
            if resp.status_code != 200:
                return False, ms, f"status {resp.status_code}"
            traces = resp.json().get("data", [])
            if traces:
                return True, ms, f"{len(traces)} recent trace(s) found for '{service_name}'"
            return False, ms, (
                f"no traces found for '{service_name}' — "
                "is RKE running and sending telemetry?"
            )
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            return False, ms, f"{type(exc).__name__}: {exc}"

    def _check_collector(self, health_url: str) -> HealthCheckResult:
        import time
        t0 = time.perf_counter()
        try:
            import httpx
            resp = httpx.get(health_url, timeout=self._timeout)
            ms = (time.perf_counter() - t0) * 1000
            ok = resp.status_code == 200
            return HealthCheckResult(
                "otel_collector_reachable",
                CheckStatus.PASS if ok else CheckStatus.FAIL,
                detail=f"status={resp.status_code} in {ms:.0f}ms",
                duration_ms=ms,
            )
        except Exception as exc:
            ms = (time.perf_counter() - t0) * 1000
            return HealthCheckResult(
                "otel_collector_reachable",
                CheckStatus.FAIL,
                detail=f"{type(exc).__name__}: {exc}",
                duration_ms=ms,
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import sys
    from rca_agent.config.settings import settings
    from rca_agent.utils.logging import setup_logging
    setup_logging()

    base_url = settings.jaeger_base_url or "http://localhost:16686"
    service = settings.jaeger_service_name or "rke-backend"

    print(f"\nRCA Agent — Observability Health Check")
    print(f"Jaeger:  {base_url}")
    print(f"Service: {service}\n")

    checker = JaegerHealthChecker(base_url, timeout_seconds=settings.jaeger_timeout_seconds)
    report = checker.check_all(
        service_name=service,
        collector_health_url="http://localhost:13133/health",
    )
    print(report.summary())
    sys.exit(0 if report.overall_healthy else 1)


if __name__ == "__main__":
    _main()
