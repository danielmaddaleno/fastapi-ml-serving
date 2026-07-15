![Tests](https://github.com/danielmaddaleno/fastapi-ml-serving/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# fastapi-ml-serving

A small FastAPI service for serving a scikit-learn model: a versioned model
registry, health/readiness probes, hot reload from disk, and a hand-rolled
Prometheus `/metrics` endpoint. No async I/O to speak of (the model lives in
memory), but inference still runs off the event loop in a worker thread so
one slow prediction doesn't stall every other request in flight.

## Run it

```bash
pip install -r requirements.txt
python scripts/train_toy_model.py   # optional: writes artifacts/model.joblib
uvicorn app.main:app --reload
```

The training step is optional. Skip it and the app still starts, serving a
built-in dummy model (`app/models/dummy.py`, returns the mean of the input
vector) under version `"dummy"`. Run it and you also get a real
`LogisticRegression` fit on scikit-learn's breast cancer dataset, loaded as
version `"production"`.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.5, 0.3, 0.8]}'
# {"prediction":0.425,"model_version":"dummy","latency_ms":0.22}
```

The dummy model accepts any feature vector length. The trained one expects
the 30 features from `sklearn.datasets.load_breast_cancer`:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_version": "production", "features": [17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
# {"prediction":0.0,"model_version":"production","latency_ms":0.63}
```

## Endpoints

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Status, whether a model is loaded, the default model version |
| `GET` | `/ready` | Plain readiness boolean, for k8s probes |
| `POST` | `/predict` | Run inference, optionally pinning `model_version` |
| `POST` | `/reload?version=production` | Re-read a model from its original file path, no restart |
| `GET` | `/metrics` | Request counts and cumulative latency, Prometheus text format |

`PredictionResponse` from the OpenAPI schema FastAPI generates at `/docs`:

```json
{
  "title": "PredictionResponse",
  "type": "object",
  "required": ["prediction", "model_version", "latency_ms"],
  "properties": {
    "prediction": {"type": "number", "title": "Prediction"},
    "model_version": {"type": "string", "title": "Model Version"},
    "latency_ms": {"type": "number", "title": "Latency Ms"}
  }
}
```

## How a request flows

1. `app/main.py` builds the `FastAPI` app. On startup, the lifespan handler
   loads the dummy model, then tries `settings.model_path`
   (`artifacts/model.joblib`, overridable via `ML_MODEL_PATH`) and loads
   that too if the file is there.
2. `POST /predict` validates the body against `PredictionRequest`
   (`app/schemas.py`), then hands off to `ModelRegistry.predict`
   (`app/models/registry.py`) via `asyncio.to_thread` so the sklearn call
   doesn't block the event loop.
3. `POST /reload` calls `joblib.load` again on the same path a version was
   first loaded from. Useful after retraining without a redeploy; it does
   not discover new files, only re-reads known ones.
4. `PrometheusMiddleware` (`app/middleware/metrics.py`) wraps every request,
   timing it and bumping a per-endpoint counter, unless the path is
   `/metrics` itself.

See [docs/metrics.md](docs/metrics.md) for the actual `/metrics` output and
what it does and doesn't track.

## Project layout

```
app/
  main.py             FastAPI app factory and lifespan (model loading)
  config.py           Settings, read from ML_-prefixed env vars
  schemas.py          Request / response Pydantic models
  models/
    registry.py       ModelRegistry: load, reload, predict, version
    dummy.py          Built-in fallback model
  routes/
    predict.py        POST /predict, POST /reload
    health.py         GET /health, GET /ready
  middleware/
    metrics.py        Prometheus-text /metrics
scripts/
  train_toy_model.py  Trains and writes artifacts/model.joblib
tests/                pytest + httpx, one file per router/module
Dockerfile            Multi-stage build
docker-compose.yml
```

## Configuration

Everything is an environment variable, `ML_`-prefixed, read by
`app/config.py`'s `Settings` (pydantic-settings):

| Variable | Default | Meaning |
|---|---|---|
| `ML_MODEL_PATH` | `artifacts/model.joblib` | Where the lifespan looks for a trained model |
| `ML_MODEL_VERSION` | `1.0.0` | Free-text version label, not currently surfaced anywhere |
| `ML_LOG_LEVEL` | `INFO` | Passed through to `logging` |
| `ML_WORKERS` | `1` | Not read by the app itself; set uvicorn's `--workers` separately if you scale out |
| `ML_ENABLE_METRICS` | `true` | Not currently wired to anything, `/metrics` is always mounted |

`ML_WORKERS` and `ML_ENABLE_METRICS` exist on `Settings` but nothing reads
them yet. Listed here rather than hidden, since that's the honest state of
the code.

## Docker

```bash
python scripts/train_toy_model.py   # optional, see docker-compose.yml
docker compose up --build
```

`Dockerfile` is two stages: a `builder` that installs dependencies with
`pip install --user`, and a `runtime` stage that copies only that installed
`.local` directory plus `app/` into a fresh `python:3.11-slim`, running as a
non-root user. No compilers or build headers end up in the final image.
`artifacts/` is not baked in at build time (it's gitignored, may not exist
yet); `docker-compose.yml` mounts it read-only instead, so retraining on the
host and restarting the container is enough to pick up a new model.

## Development

```bash
make install-dev  # requirements-dev.txt + editable install
make train         # writes artifacts/model.joblib
make test          # pytest -v
make lint          # flake8 + mypy
make format        # black + isort
```

`requirements.txt` is runtime-only (what the app imports to run);
`requirements-dev.txt` adds test and lint tooling on top of it.

## Known gaps

- `ModelRegistry` isn't thread-safe against concurrent writers. Fine for a
  single `/reload` call at a time, not fine if you fire several
  concurrently, or run multiple uvicorn workers that each hold their own
  in-memory copy anyway (reloading one doesn't reload the others).
- Metrics are counters only, no latency histogram/percentiles. See
  [docs/metrics.md](docs/metrics.md).
- No auth on `/reload`. It only re-reads a path already loaded at startup,
  but that's still an unauthenticated POST in a demo service; add a check
  before using this pattern for anything real.
- No request size limit, a huge feature vector is accepted and processed as is.

## License

MIT
