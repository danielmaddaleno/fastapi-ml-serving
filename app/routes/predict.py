"""Prediction and model-reload endpoints."""

import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, Request

from app.schemas import PredictionRequest, PredictionResponse, ReloadResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest):
    registry = request.app.state.registry
    start = time.perf_counter()
    try:
        # registry.predict() is a synchronous, CPU-bound sklearn call. Running
        # it inline would block the event loop for the duration of inference,
        # so it goes to a worker thread and the route awaits the result.
        prediction = await asyncio.to_thread(registry.predict, payload.features, payload.model_version)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        # sklearn raises ValueError when the feature vector does not match what
        # the model expects (most often the wrong number of features). That is
        # a client input problem, so surface it as a 422 rather than a 500.
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        # Anything past the KeyError/ValueError cases above is unexpected. Log
        # it server-side for debugging, but do not echo the raw exception text
        # back to the client, since it can expose internal details.
        logger.exception("Unexpected error during inference")
        raise HTTPException(status_code=500, detail="Internal error during inference")

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        prediction=prediction,
        model_version=payload.model_version or registry.default_version,
        latency_ms=round(latency_ms, 3),
    )


@router.post("/reload", response_model=ReloadResponse)
async def reload_model(request: Request, version: str = "production"):
    """Re-read a model from disk without restarting the process.

    Lets you retrain and drop a new `artifacts/model.joblib` in place, then
    pick up the change with a single call instead of a deploy.
    """
    registry = request.app.state.registry
    reloaded = await asyncio.to_thread(registry.reload, version)
    if not reloaded:
        raise HTTPException(
            status_code=404,
            detail=f"Model version '{version}' was not loaded from a file",
        )
    return ReloadResponse(reloaded=True, version=version)
