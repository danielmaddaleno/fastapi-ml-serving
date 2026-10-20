"""Lightweight Prometheus-style metrics middleware."""

from __future__ import annotations

import time
from collections import defaultdict

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp

# Upper bounds (seconds) for the latency histogram, matching
# prometheus_client.Histogram's defaults so existing dashboards and
# histogram_quantile() queries transfer unchanged.
_LATENCY_BUCKETS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_LATENCY_BUCKET_LABELS: tuple[str, ...] = (
    "0.005",
    "0.01",
    "0.025",
    "0.05",
    "0.1",
    "0.25",
    "0.5",
    "1.0",
    "2.5",
    "5.0",
    "10.0",
)


def _escape_label_value(value: str) -> str:
    """Escape a string for use as a Prometheus label value.

    The exposition format requires backslash, double-quote, and newline to be
    escaped inside a label value. Request paths can legitimately contain a
    double-quote (e.g. a probe hitting ``/pre"dict``), and interpolating one
    raw would close the ``endpoint="..."`` quote early and corrupt the whole
    ``/metrics`` payload, so every scrape after it fails to parse.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Collects request count and a latency histogram per endpoint.

    Exposes a ``/metrics`` endpoint in Prometheus text format. Counters live
    on the middleware instance, not on the class, so each app (and each test
    that builds its own app) gets an independent set of counters instead of
    sharing global state across every FastAPI instance in the process.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._request_count: dict[str, int] = defaultdict(int)
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_buckets: dict[str, list[int]] = defaultdict(lambda: [0] * len(_LATENCY_BUCKETS))

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path == "/metrics":
            return self._render_metrics()

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        # Label by the matched route template (e.g. "/predict"), which the
        # router records on the scope during call_next, rather than the raw
        # request path. An unmatched request (404) has no route, so bucket all
        # of them under one label instead of minting a fresh Prometheus time
        # series per URL. Otherwise anyone probing random paths would grow the
        # /metrics payload without bound and slow every scrape.
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "<unmatched>"
        label = f"{request.method}_{endpoint}"
        self._request_count[label] += 1
        self._latency_sum[label] += elapsed
        self._observe_latency(label, elapsed)

        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response

    def _observe_latency(self, label: str, elapsed: float) -> None:
        # Buckets are cumulative (le = less-than-or-equal), so an observation
        # lands in every bound it is under, not just the tightest one.
        counts = self._latency_buckets[label]
        for i, bound in enumerate(_LATENCY_BUCKETS):
            if elapsed <= bound:
                counts[i] += 1

    def _render_metrics(self) -> PlainTextResponse:
        lines: list[str] = [
            "# HELP http_requests_total Total request count",
            "# TYPE http_requests_total counter",
        ]
        for label, count in self._request_count.items():
            lines.append(f'http_requests_total{{endpoint="{_escape_label_value(label)}"}} {count}')

        lines += [
            "# HELP http_request_latency_seconds Request latency distribution",
            "# TYPE http_request_latency_seconds histogram",
        ]
        for label, total in self._request_count.items():
            endpoint = _escape_label_value(label)
            for le, bucket_count in zip(_LATENCY_BUCKET_LABELS, self._latency_buckets[label]):
                lines.append(f'http_request_latency_seconds_bucket{{endpoint="{endpoint}",le="{le}"}} {bucket_count}')
            # The +Inf bucket holds every observation, so it equals the count.
            lines.append(f'http_request_latency_seconds_bucket{{endpoint="{endpoint}",le="+Inf"}} {total}')
            lines.append(f'http_request_latency_seconds_sum{{endpoint="{endpoint}"}} {self._latency_sum[label]:.6f}')
            lines.append(f'http_request_latency_seconds_count{{endpoint="{endpoint}"}} {total}')

        return PlainTextResponse("\n".join(lines) + "\n")
