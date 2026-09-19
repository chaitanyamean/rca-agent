"""FastAPI application factory.

Import ``create_app`` and call it to obtain a fully configured
``FastAPI`` instance.  The factory pattern makes it easy to create
isolated app instances in tests without side effects.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rca_agent.api.routes import health as health_router
from rca_agent.config.settings import settings
from rca_agent.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:  # noqa: ARG001
    """Application lifespan handler — startup and shutdown hooks."""
    setup_logging()
    logger.info(
        "RCA Agent starting",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
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
            "AI Production Incident Root Cause Analysis platform. "
            "Standalone service — integrates with target applications via external APIs."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # tighten per environment in a future phase
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    app.include_router(health_router.router)

    return app
