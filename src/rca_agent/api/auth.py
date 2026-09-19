"""API key authentication for the RCA Agent.

Security model
--------------
This is a lightweight API-key scheme suitable for local and demo deployments.
It is NOT intended as a replacement for a full OAuth2/OIDC solution in
multi-tenant production environments.

Behaviour
---------
* When ``settings.api_key_enabled`` is False (default in development), all
  requests pass through unauthenticated.
* When enabled, every request must include the ``X-API-Key`` header (or the
  configured header name) with the value matching ``settings.api_key``.
* Endpoints that are explicitly excluded (e.g. ``/health``, ``/docs``) skip
  the check regardless of the enabled flag.

Usage::

    # In a route
    from fastapi import Depends
    from rca_agent.api.auth import require_api_key

    @router.post("/incidents/investigate")
    def investigate(request: ..., _: None = Depends(require_api_key)):
        ...
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from rca_agent.config.settings import settings

logger = logging.getLogger(__name__)

# Public paths that never require authentication
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
})

_api_key_scheme = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False,        # we handle the 401 ourselves for better messages
)


async def require_api_key(
    request: Request,
    api_key: str | None = Depends(_api_key_scheme),
) -> None:
    """FastAPI dependency that enforces API key authentication.

    Raises HTTP 401 when authentication is enabled and the key is missing or
    incorrect.  Always passes when ``api_key_enabled`` is False.
    """
    if not settings.api_key_enabled:
        return

    if request.url.path in _PUBLIC_PATHS:
        return

    if api_key is None:
        logger.warning(
            "API request rejected — missing %s header",
            settings.api_key_header,
            extra={"path": request.url.path},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                f"Missing API key. Include '{settings.api_key_header}' header."
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key):
        logger.warning(
            "API request rejected — invalid API key",
            extra={"path": request.url.path},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
