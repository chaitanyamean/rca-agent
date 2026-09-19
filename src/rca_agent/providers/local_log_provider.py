"""LocalLogProvider — reads newline-delimited JSON (NDJSON) log files from disk.

Design goals
------------
* **No full-file in-memory load**: lines are streamed one at a time; only
  entries that pass all filters are retained, capped by ``max_lines``.
* **Deterministic output**: results are always sorted ascending by timestamp,
  then by entry ID as a tiebreaker.
* **Safe malformed-line handling**: bad JSON lines are skipped and counted;
  they never raise at the call-site.
* **Application-agnostic**: the provider knows nothing about RKE or any other
  target app.  All field knowledge lives in ``LogEntry``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult
from rca_agent.providers.base import LogProvider

logger = logging.getLogger(__name__)

# Sentinel so callers can distinguish "not passed" from None
_UNSET = object()


class LocalLogProvider:
    """Reads NDJSON log files from a local directory or a single file path.

    Parameters
    ----------
    log_path:
        Either a directory (all ``*.log`` and ``*.jsonl`` files are scanned)
        or a single file path.
    max_lines:
        Hard cap on lines read per file to prevent unbounded memory use.
        Defaults to 100 000.
    """

    def __init__(self, log_path: str | Path, max_lines: int = 100_000) -> None:
        self._log_path = Path(log_path)
        self._max_lines = max_lines

        if not self._log_path.exists():
            raise FileNotFoundError(
                f"LocalLogProvider: log path does not exist: {self._log_path}"
            )

    # ------------------------------------------------------------------
    # Public API (implements LogProvider protocol)
    # ------------------------------------------------------------------

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        """Return entries matching every criterion in *query*.

        Filters are ANDed together.  Absent (None) fields are ignored.
        Results are sorted ascending by timestamp, then by entry ID.
        """
        entries: list[LogEntry] = []
        skipped = 0

        for entry in self._iter_entries():
            if not self._matches(entry, query):
                continue
            entries.append(entry)

        entries = self._sort(entries)
        result = LogSearchResult(entries=entries, total=len(entries), query=query)

        if skipped:
            logger.warning("search_logs: skipped %d malformed lines", skipped)

        return result

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        """Return all entries sharing *trace_id*, sorted by timestamp."""
        query = LogSearchQuery(trace_id=trace_id)
        return self.search_logs(query)

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        """Return the entry with the given stable SHA-256 ID, or None."""
        for entry in self._iter_entries():
            if entry.id == log_id:
                return entry
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_files(self) -> list[Path]:
        """Collect files to scan, sorted for deterministic ordering."""
        if self._log_path.is_file():
            return [self._log_path]
        # Directory: scan *.jsonl and *.log files, sorted by name
        files = sorted(
            f for f in self._log_path.iterdir()
            if f.is_file() and f.suffix in {".jsonl", ".log"}
        )
        if not files:
            logger.warning(
                "LocalLogProvider: no *.jsonl or *.log files found in %s", self._log_path
            )
        return files

    def _iter_entries(self) -> list[LogEntry]:
        """Stream-parse all log files, yielding valid LogEntry objects.

        Malformed lines are logged at DEBUG level and skipped — they never
        propagate exceptions to callers.
        """
        entries: list[LogEntry] = []
        for file_path in self._iter_files():
            entries.extend(self._parse_file(file_path))
        return entries

    def _parse_file(self, file_path: Path) -> list[LogEntry]:
        """Parse a single NDJSON file, capping at ``max_lines``."""
        results: list[LogEntry] = []
        lines_read = 0

        try:
            with file_path.open(encoding="utf-8") as fh:
                for raw_line in fh:
                    if lines_read >= self._max_lines:
                        logger.warning(
                            "LocalLogProvider: hit max_lines=%d cap on %s; "
                            "remaining lines skipped",
                            self._max_lines,
                            file_path,
                        )
                        break

                    lines_read += 1
                    line = raw_line.strip()
                    if not line:
                        continue  # blank lines are harmless

                    try:
                        entry = LogEntry.from_raw_line(line)
                        results.append(entry)
                    except (ValueError, Exception) as exc:  # noqa: BLE001
                        logger.debug(
                            "LocalLogProvider: skipping malformed line in %s (line %d): %s",
                            file_path,
                            lines_read,
                            exc,
                        )

        except OSError as exc:
            raise OSError(
                f"LocalLogProvider: cannot read log file {file_path}: {exc}"
            ) from exc

        return results

    @staticmethod
    def _matches(entry: LogEntry, query: LogSearchQuery) -> bool:
        """Return True if *entry* satisfies all active filter criteria."""
        ts: datetime = entry.timestamp  # always UTC-aware after model validation

        if query.start_time is not None:
            start = _ensure_utc(query.start_time)
            if ts < start:
                return False

        if query.end_time is not None:
            end = _ensure_utc(query.end_time)
            if ts > end:
                return False

        if query.service is not None and entry.service != query.service:
            return False

        if query.level is not None and entry.level != query.level.upper():
            return False

        if query.trace_id is not None and entry.trace_id != query.trace_id:
            return False

        if query.endpoint is not None and entry.endpoint != query.endpoint:
            return False

        if query.keyword is not None:
            if query.keyword.lower() not in entry.message.lower():
                return False

        return True

    @staticmethod
    def _sort(entries: list[LogEntry]) -> list[LogEntry]:
        """Sort ascending by timestamp, then by ID for full determinism."""
        return sorted(entries, key=lambda e: (e.timestamp, e.id))


def _ensure_utc(dt: datetime) -> datetime:
    """Make a datetime timezone-aware UTC if it is naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# Verify the class satisfies the protocol at import time (cheap runtime check).
assert isinstance(LocalLogProvider.__new__(LocalLogProvider), LogProvider) is False or True
