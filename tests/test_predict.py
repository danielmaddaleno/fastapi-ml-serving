"""Tests for the /predict and /reload endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings


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
async def test_predict_non_finite_features_rejected(app):
    # Infinity/NaN can only arrive as raw JSON literals (json.loads accepts
    # them). The validator rejects the vector, and the response must stay a
    # clean 422 rather than crashing the error handler while echoing "inf".
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            content='{"features": [1.0, Infinity]}',
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "features"]


@pytest.mark.anyio
async def test_predict_unknown_version(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"features": [1.0], "model_version": "nonexistent"},
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_predict_wrong_feature_count_is_422(app, tmp_path):
    # Load a real model that expects a fixed number of features (the breast
    # cancer pipeline expects 30), then send a vector of the wrong length.
    # sklearn raises ValueError, which the route should surface as a 422
    # client error rather than a 500.
    from scripts.train_toy_model import train

    model_path = tmp_path / "model.joblib"
    train(model_path, random_state=0)
    app.state.registry.load("production", model_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict",
            json={"features": [1.0, 2.0, 3.0], "model_version": "production"},
        )
    assert resp.status_code == 422


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


@pytest.mark.anyio
async def test_reload_requires_token_when_configured(app, monkeypatch):
    # With ML_RELOAD_TOKEN set, the auth check runs before the handler, so a
    # request with no credentials is rejected (401) regardless of version.
    monkeypatch.setattr(settings, "reload_token", "s3cret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/reload", params={"version": "production"})
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_reload_rejects_wrong_token(app, monkeypatch):
    monkeypatch.setattr(settings, "reload_token", "s3cret")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/reload",
            params={"version": "production"},
            headers={"Authorization": "Bearer wrong"},
        )
    assert resp.status_code == 401


@pytest.mark.anyio
async def test_reload_accepts_matching_token(app_with_model, monkeypatch):
    # The matching bearer token gets past the gate, and the production model
    # loaded by the fixture is re-read from its path.
    monkeypatch.setattr(settings, "reload_token", "s3cret")
    async with AsyncClient(transport=ASGITransport(app=app_with_model), base_url="http://test") as client:
        resp = await client.post(
            "/reload",
            params={"version": "production"},
            headers={"Authorization": "Bearer s3cret"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"reloaded": True, "version": "production"}
