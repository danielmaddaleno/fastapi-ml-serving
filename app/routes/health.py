"""Health and readiness probes."""

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

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
    # A readiness probe has to signal not-ready with a non-2xx status, or the
    # orchestrator (k8s) reads any 200 as "ready" and keeps routing traffic to
    # a pod whose model has not loaded. Return 503 until the registry is ready.
    is_ready = request.app.state.registry.is_ready
    code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content={"ready": is_ready})
