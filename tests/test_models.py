"""Tests for the GET /models version-listing endpoint."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_models_lists_only_the_dummy_before_an_artifact_loads(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["versions"] == ["dummy"]
    assert data["default"] == "dummy"


@pytest.mark.anyio
async def test_models_lists_the_trained_version_and_marks_it_default(app_with_model):
    async with AsyncClient(transport=ASGITransport(app=app_with_model), base_url="http://test") as client:
        resp = await client.get("/models")
    data = resp.json()
    assert data["versions"] == ["dummy", "production"]
    assert data["default"] == "production"


@pytest.mark.anyio
async def test_a_version_from_models_can_be_pinned_on_predict(app_with_model):
    # The point of listing versions is that a caller can pin one and have it
    # accepted, rather than guess and get a 404.
    async with AsyncClient(transport=ASGITransport(app=app_with_model), base_url="http://test") as client:
        listed = await client.get("/models")
        version = listed.json()["versions"][-1]
        features = [0.0] * 30
        predicted = await client.post("/predict", json={"model_version": version, "features": features})
    assert predicted.status_code == 200
    assert predicted.json()["model_version"] == version
