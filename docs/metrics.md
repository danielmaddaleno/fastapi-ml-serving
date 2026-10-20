# Metrics

`GET /metrics` is served directly by `PrometheusMiddleware`
(`app/middleware/metrics.py`), no `prometheus_client` dependency involved.
Per `METHOD_path` label it tracks how many requests hit that route and a
latency histogram (bucketed distribution, running sum, and count).

Real output after a health check and two predictions (buckets trimmed for
brevity):

```
# HELP http_requests_total Total request count
# TYPE http_requests_total counter
http_requests_total{endpoint="GET_/health"} 1
http_requests_total{endpoint="POST_/predict"} 2
# HELP http_request_latency_seconds Request latency distribution
# TYPE http_request_latency_seconds histogram
http_request_latency_seconds_bucket{endpoint="GET_/health",le="0.005"} 1
http_request_latency_seconds_bucket{endpoint="GET_/health",le="0.01"} 1
http_request_latency_seconds_bucket{endpoint="GET_/health",le="+Inf"} 1
http_request_latency_seconds_sum{endpoint="GET_/health"} 0.001565
http_request_latency_seconds_count{endpoint="GET_/health"} 1
http_request_latency_seconds_bucket{endpoint="POST_/predict",le="0.005"} 2
http_request_latency_seconds_bucket{endpoint="POST_/predict",le="+Inf"} 2
http_request_latency_seconds_sum{endpoint="POST_/predict"} 0.003935
http_request_latency_seconds_count{endpoint="POST_/predict"} 2
```

The bucket bounds are prometheus_client's default set
(`0.005 ... 10.0` seconds plus `+Inf`), so a dashboard can compute
percentiles with the usual query:

```promql
histogram_quantile(0.95, sum by (le) (rate(http_request_latency_seconds_bucket[5m])))
```

## What is and isn't here

- A histogram, exposition-compatible with `prometheus_client`'s `Histogram`
  (same `_bucket`/`_sum`/`_count` series and default bounds), but with the
  caveats below rather than a full reimplementation of that library.
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
