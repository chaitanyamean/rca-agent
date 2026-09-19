"""Structured application logging.

Supports two output formats controlled by ``Settings.log_format``:

* ``json``  — machine-readable JSON lines (default, good for production).
* ``text``  — human-readable coloured output (good for local development).

Usage::

    from rca_agent.utils.logging import setup_logging, get_logger

    setup_logging()                         # call once at startup
    logger = get_logger(__name__)
    logger.info("server started", port=8000)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from rca_agent.config.settings import settings

# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

try:
    import json as _json

    class _JsonFormatter(logging.Formatter):
        """Emit a single JSON object per log record."""

        def format(self, record: logging.LogRecord) -> str:  # noqa: A003
            log_obj: dict[str, Any] = {
                "timestamp": self.formatTime(record, self.datefmt),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_obj["exc_info"] = self.formatException(record.exc_info)
            # Attach any extra fields passed via extra={...}
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "levelname", "levelno", "pathname",
                    "filename", "module", "exc_info", "exc_text", "stack_info",
                    "lineno", "funcName", "created", "msecs", "relativeCreated",
                    "thread", "threadName", "processName", "process", "message",
                    "taskName",
                }:
                    log_obj[key] = value
            return _json.dumps(log_obj, default=str)

except ImportError:  # pragma: no cover — stdlib json always present
    _JsonFormatter = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Configure the root logger.  Call exactly once at application startup."""
    level = logging.getLevelName(settings.log_level.upper())

    handler = logging.StreamHandler(sys.stdout)

    if settings.log_format == "json" and _JsonFormatter is not None:
        handler.setFormatter(_JsonFormatter())
    else:
        fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any existing handlers so that calling setup_logging() in tests
    # doesn't accumulate duplicate handlers.
    root.handlers.clear()
    root.addHandler(handler)

    # Quieten noisy third-party loggers.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Always use ``__name__`` as the *name* argument."""
    return logging.getLogger(name)
