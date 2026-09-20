"""Persistent RCA report storage.

Reports are stored as JSON files under ``settings.reports_dir``.
File naming: ``<investigation_id>.json``

This is an intentionally simple file-based store suitable for local/demo use.
A future phase can swap it for a database-backed implementation by implementing
the ``ReportStore`` protocol.

Usage::

    from rca_agent.memory.report_store import FileReportStore
    store = FileReportStore()
    store.save(report)
    report = store.get("some-investigation-id")
    all_reports = store.list_recent(limit=10)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from rca_agent.config.settings import settings
from rca_agent.models.investigation import InvestigationReport

logger = logging.getLogger(__name__)


@runtime_checkable
class ReportStore(Protocol):
    """Protocol for investigation report persistence."""

    def save(self, report: InvestigationReport) -> Path:
        """Persist the report and return its storage path."""
        ...

    def get(self, investigation_id: str) -> InvestigationReport | None:
        """Return the report for *investigation_id*, or None."""
        ...

    def list_recent(self, limit: int = 20) -> list[InvestigationReport]:
        """Return the *limit* most recent reports, newest first."""
        ...


class FileReportStore:
    """File-based JSON report store.

    One file per investigation under ``reports_dir/``.
    """

    def __init__(self, reports_dir: str | Path | None = None) -> None:
        self._dir = Path(reports_dir or settings.reports_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, report: InvestigationReport) -> Path:
        path = self._dir / f"{report.investigation_id}.json"
        path.write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Investigation report saved",
            extra={
                "investigation_id": report.investigation_id,
                "incident_id": report.incident_id,
                "path": str(path),
            },
        )
        return path

    def get(self, investigation_id: str) -> InvestigationReport | None:
        path = self._dir / f"{investigation_id}.json"
        if not path.exists():
            return None
        try:
            return InvestigationReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load report %s: %s", investigation_id, exc)
            return None

    def list_recent(self, limit: int = 20) -> list[InvestigationReport]:
        """Return the most recent reports sorted by creation time (newest first)."""
        files = sorted(
            self._dir.glob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]
        reports: list[InvestigationReport] = []
        for f in files:
            try:
                reports.append(
                    InvestigationReport.model_validate_json(f.read_text(encoding="utf-8"))
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping unreadable report file %s: %s", f.name, exc)
        return reports
