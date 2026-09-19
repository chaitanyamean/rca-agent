"""Shared pytest fixtures for the rca-agent test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rca_agent.providers.local_log_provider import LocalLogProvider

# Root of the fixtures directory relative to this file
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Return the absolute path to tests/fixtures/."""
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def sample_log_provider(fixtures_dir: Path) -> LocalLogProvider:
    """Provider backed by the realistic sample_logs.jsonl fixture."""
    return LocalLogProvider(log_path=fixtures_dir / "sample_logs.jsonl")


@pytest.fixture(scope="session")
def malformed_log_provider(fixtures_dir: Path) -> LocalLogProvider:
    """Provider backed by the malformed_logs.jsonl fixture (contains bad JSON lines)."""
    return LocalLogProvider(log_path=fixtures_dir / "malformed_logs.jsonl")
