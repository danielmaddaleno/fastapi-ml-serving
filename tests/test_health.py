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
    assert data["model_loaded"] is True


@pytest.mark.anyio
async def test_ready(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


@pytest.mark.anyio
async def test_ready_returns_503_when_model_not_loaded(app):
    # The readiness probe must signal not-ready with a non-2xx status, or k8s
    # keeps routing traffic to a pod whose model never loaded. Empty the
    # registry to reach that state and confirm /ready answers 503, not 200.
    app.state.registry.unload_all()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ready")
    assert resp.status_code == 503
    assert resp.json()["ready"] is False
