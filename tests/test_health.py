"""Unit tests for GET /health."""

import pytest
from fastapi.testclient import TestClient

from rca_agent.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Return a synchronous test client wrapping a fresh app instance."""
    app = create_app()
    return TestClient(app)


class TestGetHealth:
    """Tests for the liveness probe endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_content_type_is_json(self, client: TestClient) -> None:
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_body_has_status_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_body_has_version(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_body_has_environment(self, client: TestClient) -> None:
        response = client.get("/health")
        data = response.json()
        assert "environment" in data
        assert isinstance(data["environment"], str)
