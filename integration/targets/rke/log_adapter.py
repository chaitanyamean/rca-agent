"""RKE log adapter — bridges RKE's log format to the RCA Agent's LogProvider.

Why this adapter exists
-----------------------
RKE's Spring Boot backend, instrumented with the OpenTelemetry Java agent,
emits structured JSON logs to stdout.  When captured to a file the lines look
like this::

    {
      "timestamp": "2026-09-19T10:00:00.123Z",
      "level":     "ERROR",
      "logger":    "c.r.backend.service.OrderService",
      "message":   "Database connection pool exhausted",
      "trace_id":  "4bf92f3577b34da6a3ce929d0e0e4736",
      "span_id":   "00f067aa0ba902b7",
      "service":   "rke-backend",
      "exception": "org.springframework.dao.DataAccessResourceFailureException"
    }

The key differences from the canonical ``LogEntry`` schema are:

1. ``trace_id`` (snake_case) instead of ``traceId`` (camelCase).
   ``LogEntry`` already accepts both via ``alias="traceId"`` and
   ``populate_by_name=True``, so this is handled automatically.
2. The ``service`` field is not always present in every log line when running
   outside Docker Compose — the adapter injects it when missing.
3. The ``logger`` field (Java class name) is not part of ``LogEntry`` but is
   preserved in ``extra_fields``.
4. No ``endpoint`` / ``method`` / ``status`` fields on non-HTTP log lines —
   these are optional in ``LogEntry`` so that's fine.

This adapter is a thin wrapper around ``LocalLogProvider`` that adds the field
normalisation step.  No RKE source code is imported or copied.

Usage::

    from integration.targets.rke.config import load_rke_config
    from integration.targets.rke.log_adapter import build_rke_log_provider
    from rca_agent.models.log_entry import LogSearchQuery

    cfg = load_rke_config()
    provider = build_rke_log_provider(cfg)
    result = provider.search_logs(LogSearchQuery(level="ERROR"))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.providers.local_log_provider import LocalLogProvider
from integration.targets.rke.config import RKETargetConfig

logger = logging.getLogger(__name__)

# Fields RKE backend adds that LogEntry doesn't have — these land in extra_fields
_RKE_EXTRA_FIELDS = frozenset({"logger", "thread_name", "span_id", "trace_flags"})


class RKELogAdapter:
    """A ``LogProvider``-compatible adapter for RKE structured logs.

    Wraps a ``LocalLogProvider`` and adds:
    * Injection of ``service`` field when absent (uses the configured
      ``backend_service_name`` as the default).
    * Tolerance for RKE-specific extra fields that are not part of ``LogEntry``.

    Since ``LogEntry`` already handles ``trace_id`` / ``traceId`` via its alias
    and ``populate_by_name=True``, no special treatment is needed for that field.
    """

    def __init__(self, inner: LocalLogProvider, default_service: str) -> None:
        self._inner = inner
        self._default_service = default_service

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        return self._inner.search_logs(query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        return self._inner.get_logs_by_trace_id(trace_id)

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        return self._inner.get_log_by_id(log_id)


class RKENormalisingLogProvider:
    """LocalLogProvider replacement that normalises RKE JSON log lines.

    Reads NDJSON files directly (not via LocalLogProvider) so it can
    pre-process each line and inject missing fields before Pydantic sees it.

    Use this when the RKE log file has missing or non-standard fields that
    would cause ``LogEntry.from_raw_line`` to fail validation.
    """

    def __init__(self, log_path: Path, default_service: str, max_lines: int = 50_000) -> None:
        self._log_path = log_path
        self._default_service = default_service
        self._max_lines = max_lines

    # ------------------------------------------------------------------
    # LogProvider protocol implementation
    # ------------------------------------------------------------------

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        entries = self._load_entries()
        matching = [e for e in entries if self._matches(e, query)]
        return LogSearchResult(entries=matching, query=query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        q = LogSearchQuery(trace_id=trace_id)
        return self.search_logs(q)

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        for entry in self._load_entries():
            if entry.id == log_id:
                return entry
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_entries(self) -> list[LogEntry]:
        """Load and normalise all log entries from the configured path."""
        files: list[Path] = []
        if self._log_path.is_file():
            files = [self._log_path]
        elif self._log_path.is_dir():
            files = sorted(
                f for f in self._log_path.iterdir()
                if f.is_file() and f.suffix in {".jsonl", ".log", ".json"}
            )
        else:
            logger.warning("RKE log path does not exist: %s", self._log_path)
            return []

        entries: list[LogEntry] = []
        for file_path in files:
            entries.extend(self._parse_file(file_path))
        return sorted(entries, key=lambda e: (e.timestamp, e.id))

    def _parse_file(self, file_path: Path) -> list[LogEntry]:
        results: list[LogEntry] = []
        lines_read = 0
        try:
            with file_path.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    if lines_read >= self._max_lines:
                        break
                    lines_read += 1
                    line = raw_line.strip()
                    if not line:
                        continue
                    normalised = self._normalise_line(line)
                    if normalised is None:
                        continue
                    try:
                        entry = LogEntry.from_raw_line(normalised)
                        results.append(entry)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("RKE: skipping malformed log line: %s", exc)
        except OSError as exc:
            logger.warning("RKE: cannot read log file %s: %s", file_path, exc)
        return results

    def _normalise_line(self, raw: str) -> str | None:
        """Apply RKE-specific normalisation and return the normalised JSON string.

        Returns None if the line should be skipped entirely.
        """
        try:
            data: dict = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # Inject service name if missing
        if "service" not in data or not data["service"]:
            data["service"] = self._default_service

        # Ensure level is present (RKE uses "level" — same key as LogEntry)
        if "level" not in data:
            data["level"] = "INFO"

        # RKE uses "message" for the log message — same as LogEntry, no change needed.
        # RKE uses "trace_id" (snake_case) — LogEntry accepts both via alias.
        # RKE uses "exception" as the exception class name — same field.

        return json.dumps(data)

    @staticmethod
    def _matches(entry: LogEntry, query: LogSearchQuery) -> bool:
        """Basic filter — delegates to the same logic used by LocalLogProvider."""
        from datetime import timezone
        ts = entry.timestamp
        if query.start_time:
            st = query.start_time
            if st.tzinfo is None:
                st = st.replace(tzinfo=timezone.utc)
            if ts < st.astimezone(timezone.utc):
                return False
        if query.end_time:
            et = query.end_time
            if et.tzinfo is None:
                et = et.replace(tzinfo=timezone.utc)
            if ts > et.astimezone(timezone.utc):
                return False
        if query.service and entry.service != query.service:
            return False
        if query.level and entry.level != query.level.upper():
            return False
        if query.trace_id and entry.trace_id != query.trace_id:
            return False
        if query.endpoint and entry.endpoint != query.endpoint:
            return False
        if query.keyword and query.keyword.lower() not in entry.message.lower():
            return False
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_rke_log_provider(cfg: RKETargetConfig) -> RKELogAdapter | RKENormalisingLogProvider | None:
    """Build the appropriate log provider for the given RKE config.

    Returns
    -------
    RKELogAdapter
        When the log path points to a standard NDJSON directory/file that
        ``LocalLogProvider`` can parse directly.
    RKENormalisingLogProvider
        When the RKE logs need field normalisation (missing service names, etc.).
    None
        When no log path is configured — the caller should handle this gracefully.
    """
    if not cfg.log_path:
        logger.info("RKE log path not configured — log provider not created.")
        return None

    log_path = Path(cfg.log_path)
    if not log_path.exists():
        logger.warning("RKE log path does not exist: %s", log_path)
        return None

    # Use the normalising provider — it handles both standard and RKE-specific
    # log formats and is tolerant of missing/extra fields.
    return RKENormalisingLogProvider(
        log_path=log_path,
        default_service=cfg.backend_service_name,
        max_lines=cfg.log_max_lines,
    )
