"""Shared pytest fixtures."""

import pytest

from app.config import settings
from app.main import create_app
from scripts.train_toy_model import train


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def model_path(tmp_path, monkeypatch):
    """Path the lifespan will look for the trained artifact at.

    Pinned per test so the suite behaves the same whether or not
    artifacts/model.joblib happens to exist in the working copy.
    """
    path = tmp_path / "model.joblib"
    monkeypatch.setattr(settings, "model_path", str(path))
    return path


@pytest.fixture
async def app(model_path):
    """A fresh FastAPI app per test, with its lifespan actually run.

    httpx.ASGITransport talks straight to the ASGI app and does not send the
    lifespan startup/shutdown events on its own, so app.state.registry would
    never get set without driving the lifespan here manually.

    No artifact exists at model_path, so this app serves the dummy only.
    """
    fastapi_app = create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        yield fastapi_app


@pytest.fixture
async def app_with_model(model_path):
    """The same app, but with a trained artifact on disk before startup."""
    train(model_path, random_state=0)
    fastapi_app = create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        yield fastapi_app
