"""Application entry point.

Run locally with::

    uvicorn rca_agent.main:app --reload

Or via the installed script::

    rca-agent
"""

import uvicorn

from rca_agent.api.app import create_app
from rca_agent.config.settings import settings

# Module-level ``app`` so that uvicorn / gunicorn can reference it directly.
app = create_app()


def run() -> None:
    """Start the uvicorn server programmatically (used by the CLI entry-point)."""
    uvicorn.run(
        "rca_agent.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
