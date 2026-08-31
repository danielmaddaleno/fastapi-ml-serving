"""Tests for the health and readiness endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient


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
    assert resp.json()["model_loaded"] is True


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
