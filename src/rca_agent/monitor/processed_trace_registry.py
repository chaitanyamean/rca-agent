"""ProcessedTraceRegistry — deduplication store for the Jaeger monitor.

Responsibility
--------------
Track which trace IDs have already triggered an RCA investigation so the
same error trace cannot trigger RCA more than once across overlapping polls.

Current implementation
----------------------
In-memory set.  Suitable for the autonomous demo — survives as long as the
Python process runs.  If RCA-Agent restarts, the registry is reset and traces
from the recent lookback window may be re-processed once.

Replacement contract
--------------------
Any persistent implementation (Redis, SQLite, PostgreSQL) only needs to
satisfy the same three-method interface used by ``JaegerMonitor``.  No
other component depends on this class's internals.

Thread safety
-------------
All mutations are protected by a ``threading.Lock`` so the registry is safe
when the monitor runs its polling loop from a background thread.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ProcessedTraceRegistry:
    """In-memory deduplication registry for processed trace IDs.

    Parameters
    ----------
    max_size:
        Maximum number of trace IDs to retain.  When the registry grows
        beyond this limit, the oldest 10 % of entries are evicted to keep
        memory bounded.  Default: 10 000.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._processed: dict[str, None] = {}  # ordered dict preserves insertion order
        self._lock = threading.Lock()
        self._max_size = max_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_processed(self, trace_id: str) -> bool:
        """Return True if *trace_id* has already been processed."""
        with self._lock:
            return trace_id in self._processed

    def mark_processed(self, trace_id: str) -> None:
        """Record *trace_id* as processed.

        If the registry is at capacity, the oldest 10 % of entries are
        evicted before inserting.
        """
        with self._lock:
            if trace_id in self._processed:
                return
            if len(self._processed) >= self._max_size:
                self._evict()
            self._processed[trace_id] = None
            logger.debug("ProcessedTraceRegistry: marked %s as processed", trace_id)

    def size(self) -> int:
        """Return the current number of tracked trace IDs."""
        with self._lock:
            return len(self._processed)

    def clear(self) -> None:
        """Remove all entries.  Primarily useful for testing."""
        with self._lock:
            self._processed.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Remove the oldest 10 % of entries.  Must be called with lock held."""
        evict_count = max(1, self._max_size // 10)
        oldest_keys = list(self._processed.keys())[:evict_count]
        for key in oldest_keys:
            del self._processed[key]
        logger.debug(
            "ProcessedTraceRegistry: evicted %d old entries (size was %d)",
            evict_count,
            len(self._processed) + evict_count,
        )
