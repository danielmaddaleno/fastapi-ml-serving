# -*- coding: utf-8 -*-
"""FastAPI ML serving application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models.registry import ModelRegistry
from app.routes import health, predict
from app.middleware.metrics import PrometheusMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default model on startup, cleanup on shutdown."""
    registry = ModelRegistry()
    registry.load_default()
    app.state.registry = registry
    yield
    registry.unload_all()


def create_app() -> FastAPI:
    app = FastAPI(
        title="ML Serving API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(PrometheusMiddleware)
    app.include_router(health.router, tags=["health"])
    app.include_router(predict.router, tags=["inference"])
    return app


app = create_app()
