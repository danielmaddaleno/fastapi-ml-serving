"""FastAPI ML serving application."""

import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.middleware.metrics import PrometheusMiddleware
from app.models.registry import ModelRegistry
from app.routes import health, predict


def _stringify_non_finite(obj):
    """Recursively replace inf/NaN floats with their string form.

    A non-finite float can only reach us as a raw JSON ``Infinity``/``NaN``
    literal, since ``json.loads`` accepts those tokens. The features validator
    rejects such input, but the default error handler echoes the offending
    value back verbatim and then blows up on ``json.dumps`` (inf/NaN are not
    valid JSON), turning a clean 422 into a 500. Coerce them to strings so the
    error body stays serializable.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_non_finite(v) for v in obj]
    return obj


async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = _stringify_non_finite(jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": errors})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the built-in dummy model, then the trained artifact if present."""
    registry = ModelRegistry()
    registry.load_default()
    registry.load_if_present("production", settings.model_path)
    app.state.registry = registry
    yield
    registry.unload_all()


def _configure_logging() -> None:
    """Apply the configured ``ML_LOG_LEVEL`` to the ``app`` package logger.

    Our modules log via ``logging.getLogger(__name__)`` (all children of the
    ``app`` logger), but nothing was reading ``settings.log_level``, so the
    setting had no effect. Set the level on the parent logger so the children
    inherit it. An unrecognized level name falls back to INFO rather than
    raising at startup.
    """
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.getLogger("app").setLevel(level)


def create_app() -> FastAPI:
    _configure_logging()
    app = FastAPI(
        title="ML Serving API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_middleware(PrometheusMiddleware)
    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, tags=["inference"])
    return app


app = create_app()
