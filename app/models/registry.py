# -*- coding: utf-8 -*-
"""Registry — core implementation."""
"""Model registry – load, cache, and version ML models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Any object that exposes a sklearn-style predict method."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class ModelRegistry:
    """Thread-safe in-memory model store with versioning."""

    def __init__(self) -> None:
        self._models: dict[str, Predictor] = {}
        self._default_version: str | None = None

    # ------------------------------------------------------------------
    def load(self, version: str, path: str | Path) -> None:
        import joblib

        model = joblib.load(path)
        self._models[version] = model
        if self._default_version is None:
            self._default_version = version
        logger.info("Loaded model v%s from %s", version, path)

    def load_default(self) -> None:
        """Load a built-in dummy model for demo / health-check purposes."""
        from app.models.dummy import DummyModel

        dummy = DummyModel()
        self._models["dummy"] = dummy
        self._default_version = "dummy"
        logger.info("Loaded built-in dummy model")

    # ------------------------------------------------------------------
    def predict(self, features: list[float], version: str | None = None) -> float:
        v = version or self._default_version
        if v is None or v not in self._models:
            raise KeyError(f"Model version '{v}' not found")
        X = np.array(features).reshape(1, -1)
        return float(self._models[v].predict(X)[0])

    @property
    def is_ready(self) -> bool:
        return len(self._models) > 0

    @property
    def default_version(self) -> str:
        return self._default_version or "none"

    def unload_all(self) -> None:
        self._models.clear()
        self._default_version = None
