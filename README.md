![Tests](https://github.com/danielmaddaleno/fastapi-ml-serving/actions/workflows/tests.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# 🚀 fastapi-ml-serving

Production-ready template for serving ML models behind a FastAPI service with async inference, health checks, Prometheus metrics, and Docker support.

## Features

| Feature | Detail |
|---|---|
| **Async inference** | Non-blocking prediction endpoint using `asyncio` |
| **Model registry** | Hot-swap models without restarting the service |
| **Health & readiness** | `/health` and `/ready` probes for Kubernetes |
| **Prometheus metrics** | Request latency, prediction counts, error rates |
| **Input validation** | Pydantic v2 request/response schemas |
| **Docker** | Multi-stage build, ~120 MB final image |
| **CI-ready** | pytest + httpx for integration tests |

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.5, 0.3, 0.8]}'
```

## Project structure

```
app/
├── main.py            # FastAPI application factory
├── config.py          # Pydantic settings
├── models/
│   ├── registry.py    # Model loading & versioning
│   └── dummy.py       # Example sklearn model
├── schemas.py         # Request / response models
├── routes/
│   ├── predict.py     # POST /predict
│   └── health.py      # GET /health, /ready
├── middleware/
│   └── metrics.py     # Prometheus instrumentation
tests/
├── test_predict.py
├── test_health.py
Dockerfile
docker-compose.yml
requirements.txt
```

## Docker

```bash
docker compose up --build
# API on :8000, Prometheus metrics on :8000/metrics
```

## License

MIT


## Installation

```bash
git clone https://github.com/danielmaddaleno/fastapi-ml-serving.git
cd fastapi-ml-serving
pip install -r requirements.txt
```

## Usage

See `docs/` for detailed examples.

## Configuration

Configuration files live in `configs/`. Copy the sample and edit.

## Development

```bash
make install  # Install deps
make test     # Run tests
make lint     # Linters
```

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Roadmap

- [ ] Improve test coverage
- [ ] Add benchmarks
- [ ] Docker support

## Acknowledgements

Built with Python and open-source tools.
