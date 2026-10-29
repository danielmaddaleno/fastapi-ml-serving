"""Tests for the request body size limit."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import create_app
from app.middleware.body_limit import BodySizeLimitMiddleware


@pytest.fixture
async def small_cap_app(model_path, monkeypatch):
    """An app whose body cap is small enough to trip from a test payload."""
    monkeypatch.setattr(settings, "max_request_bytes", 200)
    fastapi_app = create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        yield fastapi_app


@pytest.mark.anyio
async def test_oversized_body_rejected(small_cap_app):
    # The feature list alone is well over the 200-byte cap, so the request is
    # refused with 413 and never reaches the model.
    big = {"features": [0.0] * 500}
    async with AsyncClient(transport=ASGITransport(app=small_cap_app), base_url="http://test") as client:
        resp = await client.post("/predict", json=big)
    assert resp.status_code == 413


@pytest.mark.anyio
async def test_body_within_limit_passes(small_cap_app):
    async with AsyncClient(transport=ASGITransport(app=small_cap_app), base_url="http://test") as client:
        resp = await client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_zero_cap_disables_the_limit(model_path, monkeypatch):
    monkeypatch.setattr(settings, "max_request_bytes", 0)
    fastapi_app = create_app()
    async with fastapi_app.router.lifespan_context(fastapi_app):
        async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as client:
            resp = await client.post("/predict", json={"features": [0.0] * 5000})
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_streaming_body_capped_without_content_length():
    # The header short-circuit is bypassed here: the inner app reads the body
    # chunk by chunk and the scope carries no Content-Length, so the cap has to
    # hold on the running byte count alone. Two 8-byte chunks exceed the cap of
    # 10 on the second read, before the inner app can respond.
    async def inner(scope, receive, send):
        more = True
        while more:
            message = await receive()
            more = message.get("more_body", False)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = BodySizeLimitMiddleware(inner, max_bytes=10)
    chunks = [
        {"type": "http.request", "body": b"abcdefgh", "more_body": True},
        {"type": "http.request", "body": b"ijklmnop", "more_body": False},
    ]

    async def receive():
        return chunks.pop(0)

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": []}
    await middleware(scope, receive, send)

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
