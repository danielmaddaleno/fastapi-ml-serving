# Metrics

`GET /metrics` is served directly by `PrometheusMiddleware`
(`app/middleware/metrics.py`), no `prometheus_client` dependency involved.
It tracks two things per `METHOD_path` label: how many requests hit that
route, and their cumulative latency in seconds.

Real output after a health check, two predictions, and a reload:

```
# HELP http_requests_total Total request count
# TYPE http_requests_total counter
http_requests_total{endpoint="GET_/health"} 1
http_requests_total{endpoint="POST_/predict"} 2
http_requests_total{endpoint="POST_/reload"} 1
# HELP http_request_latency_seconds_sum Cumulative latency
# TYPE http_request_latency_seconds_sum counter
http_request_latency_seconds_sum{endpoint="GET_/health"} 0.001565
http_request_latency_seconds_sum{endpoint="POST_/predict"} 0.003935
http_request_latency_seconds_sum{endpoint="POST_/reload"} 0.001722
```

## What is and isn't here

- Counters, not histograms. `http_request_latency_seconds_sum` is a running
  total, so you can derive average latency (`sum / count`) but there is no
  bucketed distribution and no `_count`/`_bucket` series. Good enough for a
  small service, not a drop-in replacement for `prometheus_client`'s
  `Histogram`.
- The `/metrics` route itself is excluded from its own counters (it returns
  early in `dispatch()` before the timing code runs).
- Counters live on the `PrometheusMiddleware` instance, which FastAPI
  creates once per app via `add_middleware`. Two apps in the same process
  (as in the test suite, where every test builds its own app) never share
  counters. Restarting the process resets everything, same as
  `prometheus_client`'s default in-process registry.

## Scraping it

Nothing exotic, it is plain Prometheus text format on a normal route:

```yaml
scrape_configs:
  - job_name: fastapi-ml-serving
    static_configs:
      - targets: ["localhost:8000"]
    metrics_path: /metrics
```
