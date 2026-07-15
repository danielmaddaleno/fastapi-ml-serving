"""Shared pytest fixtures."""

import pytest

from app.main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def app():
    """A fresh FastAPI app per test, with its lifespan actually run.

    httpx.ASGITransport talks straight to the ASGI app and does not send the
    lifespan startup/shutdown events on its own, so app.state.registry would
    never get set without driving the lifespan here manually.
    """
    fastapi_app = create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        yield fastapi_app
