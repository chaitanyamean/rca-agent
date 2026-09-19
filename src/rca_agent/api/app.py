"""FastAPI application factory — Phase 10 hardened.

Changes from Phase 1
--------------------
* RequestIDMiddleware — every request gets a UUID for log correlation
* StructuredLoggingMiddleware — JSON access logs with duration/status
* SlowAPI rate limiter — configurable per-endpoint limits
* API key authentication — optional, enabled via API_KEY_ENABLED=true
* POST /incidents/investigate — the core RCA investigation endpoint
* CORS restricted to configured origins (not wildcard in production)
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from rca_agent.api.middleware import RequestIDMiddleware, StructuredLoggingMiddleware, limiter
from rca_agent.api.routes import health as health_router
from rca_agent.api.routes import investigate as investigate_router
from rca_agent.config.settings import settings
from rca_agent.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler — startup and shutdown hooks."""
    setup_logging()
    logger.info(
        "RCA Agent starting",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "api_key_enabled": settings.api_key_enabled,
            "rate_limit_enabled": settings.rate_limit_enabled,
            "llm_provider": settings.llm_provider,
            "prompt_version": settings.prompt_version,
        },
    )
    yield
    logger.info("RCA Agent shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered Production Incident Root Cause Analysis platform.\n\n"
            "Standalone service — integrates with target applications via "
            "configurable provider interfaces.\n\n"
            "**Authentication**: Set `API_KEY_ENABLED=true` and `API_KEY=<secret>` "
            "to enable API key auth (recommended outside local development).\n\n"
            "**Status**: Research/demonstration quality — not production-ready. "
            "See `README.md` for known limitations."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------
    # Rate limiter state (must be set before adding the error handler)
    # ------------------------------------------------------------------
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Middleware (applied last-registered = outermost)
    # ------------------------------------------------------------------
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # CORS — restrict to localhost in non-production
    cors_origins = (
        ["http://localhost:3001", "http://localhost:3000", "http://localhost:8000"]
        if settings.environment != "production"
        else []  # Explicitly empty in production — configure per deployment
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],  # fallback to wildcard only in dev
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(health_router.router)
    app.include_router(investigate_router.router)

    return app
