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


class BatchPredictionRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1, description="Feature vectors to score in one call")
    model_version: str | None = Field(None, description="Optional model version override")

    @field_validator("instances")
    @classmethod
    def _instances_must_be_finite_and_rectangular(cls, v: list[list[float]]) -> list[list[float]]:
        # A batch is one matrix handed to a single model.predict, so the rows
        # have to share a width; a ragged batch would otherwise surface as an
        # opaque numpy error. Finiteness is rejected for the same reason as the
        # single-vector case: NaN/inf poison inference and can't be serialized.
        width = len(v[0])
        if width == 0:
            raise ValueError("each instance must have at least one feature")
        for row in v:
            if len(row) != width:
                raise ValueError("every instance must have the same number of features")
            if not all(isfinite(x) for x in row):
                raise ValueError("features must be finite (no NaN or infinity)")
        return v


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ReloadResponse(BaseModel):
    reloaded: bool
    version: str
