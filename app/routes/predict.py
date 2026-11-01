"""Prediction and model-reload endpoints."""

import asyncio
import logging
import math
import secrets
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.config import settings
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
    ReloadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _inference_error(exc: Exception) -> HTTPException:
    """Map an exception raised by a registry inference call to an HTTP error.

    KeyError means the pinned version was never loaded (404). ValueError is
    sklearn rejecting the input, most often a wrong feature count (422).
    Anything else is unexpected: log it server-side, but keep the raw text out
    of the response since it can expose internals.
    """
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=422, detail=str(exc))
    logger.exception("Unexpected error during inference")
    return HTTPException(status_code=500, detail="Internal error during inference")


def _require_reload_token(authorization: str | None = Header(default=None)) -> None:
    """Gate /reload behind a bearer token when ML_RELOAD_TOKEN is set.

    Unset, the endpoint stays open so the demo keeps working. Set, each reload
    must send Authorization: Bearer <token>, compared in constant time so a
    wrong token leaks nothing through timing.
    """
    expected = settings.reload_token
    if not expected:
        return
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing reload token")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
    registry = request.app.state.registry
    start = time.perf_counter()
    try:
        # registry.predict() is a synchronous, CPU-bound sklearn call. Running
        # it inline would block the event loop for the duration of inference,
        # so it goes to a worker thread and the route awaits the result.
        prediction = await asyncio.to_thread(registry.predict, payload.features, payload.model_version)
    except Exception as exc:
        raise _inference_error(exc) from exc

    if not math.isfinite(prediction):
        # A model can emit NaN/inf under numerical instability even when the
        # input vector is finite. Pydantic serializes such a value to JSON
        # null, so the client would get a 200 with prediction=null and no
        # signal that anything went wrong. Surface it as a server error
        # instead of handing back a silent bad answer.
        version = payload.model_version or registry.default_version
        logger.error("Model %s produced a non-finite prediction", version)
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction")

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        prediction=prediction,
        model_version=payload.model_version or registry.default_version,
        latency_ms=round(latency_ms, 3),
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: Request, payload: BatchPredictionRequest) -> BatchPredictionResponse:
    """Score many feature vectors in one request.

    One round trip and one vectorized model.predict for the whole batch,
    instead of a request per row. Version resolution and error mapping match
    /predict, and predictions come back in the order the instances were sent.
    """
    registry = request.app.state.registry
    start = time.perf_counter()
    try:
        predictions = await asyncio.to_thread(registry.predict_batch, payload.instances, payload.model_version)
    except Exception as exc:
        raise _inference_error(exc) from exc

    if any(not math.isfinite(p) for p in predictions):
        # One bad row would serialize to a JSON null in the middle of the list,
        # a silent wrong answer the caller can't tell from a real one. Fail the
        # whole batch instead, same as the single-prediction path does.
        version = payload.model_version or registry.default_version
        logger.error("Model %s produced a non-finite prediction in a batch", version)
        raise HTTPException(status_code=500, detail="Model produced a non-finite prediction")

    latency_ms = (time.perf_counter() - start) * 1000
    return BatchPredictionResponse(
        predictions=predictions,
        model_version=payload.model_version or registry.default_version,
        latency_ms=round(latency_ms, 3),
    )


@router.post("/reload", response_model=ReloadResponse, dependencies=[Depends(_require_reload_token)])
async def reload_model(request: Request, version: str = "production") -> ReloadResponse:
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
