"""Tests for the /predict/batch endpoint and ModelRegistry.predict_batch."""

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.models.dummy import DummyModel
from app.models.registry import ModelRegistry


@pytest.mark.anyio
async def test_batch_scores_every_row_in_order(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict/batch",
            json={"instances": [[0.0, 1.0], [2.0, 2.0], [10.0, 0.0]]},
        )
    assert resp.status_code == 200
    data = resp.json()
    # The dummy returns the per-row mean, so the response pins down both the
    # values and that order is preserved across the batch.
    assert data["predictions"] == [0.5, 2.0, 5.0]
    assert data["model_version"] == "dummy"
    assert "latency_ms" in data


@pytest.mark.anyio
async def test_batch_single_row_matches_predict(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        one = await client.post("/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
        batch = await client.post("/predict/batch", json={"instances": [[1.0, 2.0, 3.0, 4.0]]})
    assert batch.json()["predictions"] == [one.json()["prediction"]]


@pytest.mark.anyio
async def test_batch_empty_is_422(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict/batch", json={"instances": []})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_ragged_rows_rejected(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict/batch", json={"instances": [[1.0, 2.0], [3.0]]})
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_non_finite_rejected(app):
    # Infinity only arrives as a raw JSON literal; it must be refused with a
    # clean 422, the same as the single-vector endpoint does.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict/batch",
            content='{"instances": [[1.0, Infinity]]}',
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_batch_unknown_version_is_404(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/predict/batch",
            json={"instances": [[1.0]], "model_version": "nonexistent"},
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_batch_wrong_feature_count_is_422(app_with_model):
    # The production pipeline expects 30 features; sklearn raises ValueError on
    # the wrong width, which must surface as a 422 for the whole batch.
    async with AsyncClient(transport=ASGITransport(app=app_with_model), base_url="http://test") as client:
        resp = await client.post(
            "/predict/batch",
            json={"instances": [[1.0, 2.0, 3.0]], "model_version": "production"},
        )
    assert resp.status_code == 422


def test_registry_predict_batch_one_call_per_batch():
    registry = ModelRegistry()
    registry.load_default()

    preds = registry.predict_batch([[1.0, 3.0], [4.0, 4.0]])

    assert preds == [2.0, 4.0]


def test_registry_predict_batch_unknown_version_raises_keyerror():
    registry = ModelRegistry()
    registry.load_default()

    with pytest.raises(KeyError):
        registry.predict_batch([[1.0]], version="nonexistent")


def test_registry_predict_batch_matches_single_predict():
    registry = ModelRegistry()
    registry.load_default()

    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    batched = registry.predict_batch(rows)
    one_by_one = [registry.predict(row) for row in rows]

    assert batched == one_by_one


def test_dummy_predict_handles_a_2d_batch():
    # The registry stacks the batch into one 2-D array; the model has to return
    # one prediction per row, not collapse the whole matrix to a single value.
    model = DummyModel()
    out = model.predict(np.array([[1.0, 3.0], [10.0, 20.0]]))
    assert out.tolist() == [2.0, 15.0]
