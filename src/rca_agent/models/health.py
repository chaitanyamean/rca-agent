"""Pydantic response models for the health endpoint."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body returned by ``GET /health``."""

    status: str = Field(description="Service status.  Always 'ok' when reachable.")
    version: str = Field(description="Running application version.")
    environment: str = Field(description="Runtime environment label.")
