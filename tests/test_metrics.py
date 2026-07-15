"""Tests for the /metrics endpoint and PrometheusMiddleware."""

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.anyio
async def test_metrics_returns_prometheus_text(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    body = resp.text
    assert "# HELP http_requests_total" in body
    assert "# TYPE http_requests_total counter" in body


@pytest.mark.anyio
async def test_metrics_counts_requests_by_endpoint(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/health")
        await client.get("/health")
        resp = await client.get("/metrics")

    body = resp.text
    assert 'http_requests_total{endpoint="GET_/health"} 2' in body


@pytest.mark.anyio
async def test_metrics_state_does_not_leak_across_apps(app):
    # A second app, built the same way conftest builds `app`, must start
    # with its own empty counters rather than inheriting the first app's
    # request count. This is what moving PrometheusMiddleware's counters
    # off the class and onto the instance actually buys us.
    from app.main import create_app

    other_app = create_app()
    async with other_app.router.lifespan_context(other_app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/health")

        async with AsyncClient(transport=ASGITransport(app=other_app), base_url="http://test") as client:
            resp = await client.get("/metrics")

    assert 'endpoint="GET_/health"' not in resp.text
