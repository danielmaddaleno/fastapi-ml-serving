# -*- coding: utf-8 -*-
"""Metrics — core implementation."""
"""Lightweight Prometheus-style metrics middleware."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collects request count and latency histograms per endpoint.

    Exposes a ``/metrics`` endpoint in Prometheus text format.
    """

    _request_count: dict[str, int] = defaultdict(int)
    _latency_sum: dict[str, float] = defaultdict(float)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Serve metrics endpoint
        if request.url.path == "/metrics":
            return self._render_metrics()

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        label = f"{request.method}_{request.url.path}"
        PrometheusMiddleware._request_count[label] += 1
        PrometheusMiddleware._latency_sum[label] += elapsed

        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response

    @staticmethod
    def _render_metrics() -> PlainTextResponse:
        lines: list[str] = [
            "# HELP http_requests_total Total request count",
            "# TYPE http_requests_total counter",
        ]
        for label, count in PrometheusMiddleware._request_count.items():
            lines.append(f'http_requests_total{{endpoint="{label}"}} {count}')

        lines += [
            "# HELP http_request_latency_seconds_sum Cumulative latency",
            "# TYPE http_request_latency_seconds_sum counter",
        ]
        for label, total in PrometheusMiddleware._latency_sum.items():
            lines.append(
                f'http_request_latency_seconds_sum{{endpoint="{label}"}} {total:.6f}'
            )

        return PlainTextResponse("\n".join(lines) + "\n")
