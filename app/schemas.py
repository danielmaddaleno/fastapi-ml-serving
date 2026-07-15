"""Pydantic request / response schemas."""

from math import isfinite

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1, description="Numeric feature vector")
    model_version: str | None = Field(None, description="Optional model version override")

    @field_validator("features")
    @classmethod
    def _features_must_be_finite(cls, v: list[float]) -> list[float]:
        # NaN/inf slip through `list[float]` but poison inference: the model
        # returns NaN and the JSON response can't encode it. Reject them here.
        if not all(isfinite(x) for x in v):
            raise ValueError("features must be finite (no NaN or infinity)")
        return v


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ReloadResponse(BaseModel):
    reloaded: bool
    version: str
