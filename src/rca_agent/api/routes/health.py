"""Health-check endpoint.

``GET /health`` is the canonical liveness probe for the service.
It requires no authentication and returns HTTP 200 as long as the
application process is running and able to handle requests.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from rca_agent.config.settings import settings
from rca_agent.models.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns HTTP 200 with service metadata when the application is running.",
)
def get_health() -> JSONResponse:
    """Return service health status."""
    body = HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )
    return JSONResponse(content=body.model_dump(), status_code=200)
