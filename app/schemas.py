# -*- coding: utf-8 -*-
"""Schemas — core implementation."""
"""Pydantic request / response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=1, description="Numeric feature vector"
    )
    model_version: str | None = Field(
        None, description="Optional model version override"
    )


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
