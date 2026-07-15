"""Tests for the /predict and /reload endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_predict_success(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert "latency_ms" in data
    assert data["model_version"] == "dummy"


@pytest.mark.anyio
async def test_predict_empty_features(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": []})
    assert resp.status_code == 422  # validation error


@pytest.mark.anyio
async def test_predict_unknown_version(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"features": [1.0], "model_version": "nonexistent"},
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_reload_unknown_version(app):
    # "dummy" was built in memory, not loaded from a file, so there is
    # nothing on disk to re-read.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/reload", params={"version": "dummy"})
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_reload_missing_version(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/reload", params={"version": "nope"})
    assert resp.status_code == 404
