# -*- coding: utf-8 -*-
"""Health & readiness probes."""

from fastapi import APIRouter, Request

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    registry = request.app.state.registry
    return HealthResponse(
        status="ok",
        model_loaded=registry.is_ready,
        version=registry.default_version,
    )


@router.get("/ready")
async def ready(request: Request):
    if request.app.state.registry.is_ready:
        return {"ready": True}
    return {"ready": False}
