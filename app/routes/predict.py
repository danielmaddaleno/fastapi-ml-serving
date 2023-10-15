"""Prediction endpoint."""

import time

from fastapi import APIRouter, HTTPException, Request

from app.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, payload: PredictionRequest):
    registry = request.app.state.registry
    start = time.perf_counter()
    try:
        prediction = registry.predict(
            payload.features, version=payload.model_version
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        prediction=prediction,
        model_version=payload.model_version or registry.default_version,
        latency_ms=round(latency_ms, 3),
    )
