"""Production middleware for the RCA Agent API.

Middleware stack (applied in reverse registration order):
1. RequestIDMiddleware  — assigns a UUID to every request for log correlation
2. StructuredLoggingMiddleware — logs method, path, status, latency as JSON
3. SlowAPI rate limiter — registered on the app via Limiter (in app.py)

Rate limiting
-------------
Uses ``slowapi`` (a Starlette/FastAPI port of ``flask-limiter``).
The limiter is keyed on the client IP address.  Limits are defined in
settings and can be overridden per endpoint with the ``@limiter.limit()``
decorator.

All limits apply only when ``settings.rate_limit_enabled`` is True.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from rca_agent.config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SlowAPI limiter — single instance shared across the app
# ---------------------------------------------------------------------------

limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    default_limits=[settings.rate_limit_default],
)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------

class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assigns a UUID ``X-Request-ID`` to every incoming request.

    The ID is:
    * Added to the response headers.
    * Available in route handlers via ``request.state.request_id``.
    * Included in all structured log records for the duration of the request.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Structured access-logging middleware
# ---------------------------------------------------------------------------

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Emits one structured log record per HTTP request.

    Fields logged:
    * ``method``      — HTTP verb
    * ``path``        — URL path
    * ``status_code`` — HTTP response status
    * ``duration_ms`` — Request processing time in milliseconds
    * ``request_id``  — Correlation ID from ``RequestIDMiddleware``
    * ``client_ip``   — Remote address
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)

        request_id = getattr(request.state, "request_id", "-")
        client_ip = request.client.host if request.client else "-"

        logger.info(
            "%s %s %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
                "client_ip": client_ip,
            },
        )
        return response
