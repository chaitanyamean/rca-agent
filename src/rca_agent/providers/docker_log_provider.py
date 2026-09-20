"""DockerLogProvider — reads structured JSON logs from a Docker container's stdout.

Design goals
------------
* **Zero infrastructure** — reads ``docker logs <container>`` directly; no
  log-shipper, no Elasticsearch, no Loki required.
* **Reuses existing parser** — all lines are parsed through ``LogEntry.from_raw_line()``
  which already handles the ``@timestamp`` / ``traceId`` normalisation.
* **TraceID correlation** — ``get_logs_by_trace_id(trace_id)`` returns only
  entries whose ``entry.trace_id == trace_id``, enabling exact correlation with
  the Jaeger-detected trace.
* **Satisfies LogProvider protocol** — drop-in replacement for LocalLogProvider
  in any context that constructs providers.
* **Graceful degradation** — if Docker is unavailable or the container does not
  exist, methods return empty results rather than crashing the RCA workflow.

RKE log format (Logstash JSON / spring-boot-logstash-encoder)
-------------------------------------------------------------
Each line is a JSON object with at minimum::

    {
      "@timestamp": "2026-09-20T12:09:07.783Z",
      "level":      "WARN",
      "message":    "[INC-001] Pool exhaustion ...",
      "service":    "rke-backend",
      "traceId":    "a40d28e202b0b22f1c13d2399975ecfa",
      "spanId":     "d9a8d2c5928d208e"
    }

The ``@timestamp`` field is normalised to ``timestamp`` inside
``LogEntry.from_raw_line()`` so no special handling is needed here.

Usage::

    provider = DockerLogProvider("rke-backend")
    result = provider.get_logs_by_trace_id("a40d28e202b0b22f")
    for entry in result.entries:
        print(entry.level, entry.message)
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone

from rca_agent.models.log_entry import LogEntry, LogSearchQuery, LogSearchResult

logger = logging.getLogger(__name__)


class DockerLogProvider:
    """Reads structured JSON logs from a running Docker container.

    Parameters
    ----------
    container_name:
        The Docker container name or ID (e.g. ``"rke-backend"``).
    since_minutes:
        How many minutes of logs to fetch on each call.  Limits the
        ``docker logs --since`` window to avoid reading the full history.
        Default: 60 (one hour).
    max_lines:
        Hard cap on lines parsed to prevent unbounded memory use.
        Default: 10 000.
    docker_cmd:
        Path to the docker binary.  Override in tests or restricted envs.
    """

    def __init__(
        self,
        container_name: str,
        since_minutes: int = 60,
        max_lines: int = 10_000,
        docker_cmd: str = "docker",
    ) -> None:
        self._container = container_name
        self._since_minutes = since_minutes
        self._max_lines = max_lines
        self._docker_cmd = docker_cmd

    # ------------------------------------------------------------------
    # LogProvider protocol
    # ------------------------------------------------------------------

    def search_logs(self, query: LogSearchQuery) -> LogSearchResult:
        """Return log entries matching *query*, parsed from container stdout."""
        entries = self._fetch_entries()
        matching = [e for e in entries if self._matches(e, query)]
        matching = sorted(matching, key=lambda e: (e.timestamp, e.id))
        return LogSearchResult(entries=matching, total=len(matching), query=query)

    def get_logs_by_trace_id(self, trace_id: str) -> LogSearchResult:
        """Return all entries whose ``traceId`` == *trace_id*."""
        query = LogSearchQuery(trace_id=trace_id)
        return self.search_logs(query)

    def get_log_by_id(self, log_id: str) -> LogEntry | None:
        """Return the entry with the given SHA-256 ID, or None."""
        for e in self._fetch_entries():
            if e.id == log_id:
                return e
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_entries(self) -> list[LogEntry]:
        """Run ``docker logs`` and parse the output into LogEntry objects."""
        cmd = [
            self._docker_cmd, "logs",
            "--since", f"{self._since_minutes}m",
            self._container,
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except FileNotFoundError:
            logger.warning(
                "DockerLogProvider: docker binary not found — "
                "container log retrieval unavailable"
            )
            return []
        except subprocess.TimeoutExpired:
            logger.warning(
                "DockerLogProvider: 'docker logs %s' timed out", self._container
            )
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DockerLogProvider: failed to run 'docker logs %s': %s",
                self._container, exc,
            )
            return []

        if proc.returncode != 0:
            logger.warning(
                "DockerLogProvider: 'docker logs %s' returned exit code %d: %s",
                self._container, proc.returncode, proc.stderr[:200],
            )
            return []

        # Docker writes to stderr for older API versions; try both streams
        raw_output = proc.stdout or proc.stderr
        entries: list[LogEntry] = []
        lines_seen = 0

        for line in raw_output.splitlines():
            if lines_seen >= self._max_lines:
                logger.debug(
                    "DockerLogProvider: hit max_lines=%d cap for %s",
                    self._max_lines, self._container,
                )
                break
            line = line.strip()
            if not line:
                continue
            lines_seen += 1
            try:
                entries.append(LogEntry.from_raw_line(line))
            except (ValueError, Exception) as exc:  # noqa: BLE001
                logger.debug(
                    "DockerLogProvider: skipping malformed line in %s: %s",
                    self._container, exc,
                )

        logger.debug(
            "DockerLogProvider: fetched %d entries from container %s",
            len(entries), self._container,
        )
        return entries

    @staticmethod
    def _matches(entry: LogEntry, query: LogSearchQuery) -> bool:
        """Return True if *entry* satisfies all active filter criteria."""
        ts = entry.timestamp

        if query.start_time is not None:
            start = query.start_time
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            if ts < start.astimezone(timezone.utc):
                return False

        if query.end_time is not None:
            end = query.end_time
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            if ts > end.astimezone(timezone.utc):
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
