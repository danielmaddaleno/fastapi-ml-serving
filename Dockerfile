# syntax=docker/dockerfile:1

# ---- builder: compile wheels, nothing else ships from this stage ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- runtime: just the installed packages plus app code ----
FROM python:3.11-slim AS runtime

# curl is only here so the docker-compose healthcheck can hit /health.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY app/ app/

# artifacts/model.joblib is not baked into the image: it is gitignored and
# not guaranteed to exist at build time. Mount it at run time instead (see
# docker-compose.yml) or bake your own COPY in a derived image. Without it
# the app falls back to the built-in dummy model (app/models/registry.py).
RUN mkdir -p artifacts

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
