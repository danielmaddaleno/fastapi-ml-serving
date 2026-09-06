"""Tests for the health and readiness endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import create_app


@pytest.mark.anyio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "dummy"
    assert data["model_loaded"] is False


@pytest.mark.anyio
async def test_health_reports_the_trained_model(app_with_model):
    async with AsyncClient(transport=ASGITransport(app=app_with_model), base_url="http://test") as client:
        resp = await client.get("/health")
    data = resp.json()
    assert data["model_loaded"] is True
    assert data["version"] == "production"


@pytest.mark.anyio
async def test_ready_is_503_with_only_the_dummy(app):
    # The readiness probe must signal not-ready with a non-2xx status, or k8s
    # keeps routing traffic to a pod whose trained model never loaded. The
    # dummy is always in memory, so it must not count as ready.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["ready"] is False


@pytest.mark.anyio
async def test_ready_is_200_once_the_trained_model_loads(app_with_model):
    async with AsyncClient(transport=ASGITransport(app=app_with_model), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@pytest.mark.anyio
async def test_app_starts_when_the_artifact_is_unreadable(model_path):
    # A corrupt artifact used to raise out of the lifespan. The service should
    # come up on the dummy and report not-ready instead.
    model_path.write_bytes(b"this is not a joblib artifact")

    fastapi_app = create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        transport = ASGITransport(app=fastapi_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            ready = await client.get("/ready")
            predicted = await client.post("/predict", json={"features": [1.0, 3.0]})

    assert ready.status_code == 503
    assert predicted.status_code == 200
    assert predicted.json()["model_version"] == "dummy"
