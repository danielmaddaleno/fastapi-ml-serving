"""Model registry: load, cache, and version ML models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)


class Predictor(Protocol):
    """Any object that exposes a sklearn-style predict method."""

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class ModelRegistry:
    """In-memory model store with versioning.

    Not safe for concurrent writes from multiple threads/processes; the app
    only mutates it from the lifespan handler and the single-worker /reload
    endpoint, so a lock has not been needed so far.
    """

    def __init__(self) -> None:
        self._models: dict[str, Predictor] = {}
        self._default_version: str | None = None
        self._paths: dict[str, Path] = {}

    # ------------------------------------------------------------------
    def load(self, version: str, path: str | Path) -> None:
        import joblib

        model = joblib.load(path)
        self._models[version] = model
        self._paths[version] = Path(path)
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

    def load_if_present(self, version: str, path: str | Path) -> bool:
        """Load a model from ``path`` if the file exists.

        Used at startup for the trained artifact: it may not exist yet on a
        fresh clone (the training script has not been run), and the app
        should still come up with the dummy model in that case.
        """
        p = Path(path)
        if not p.is_file():
            logger.info("No model artifact at %s, skipping v%s", p, version)
            return False
        self.load(version, p)
        return True

    def reload(self, version: str) -> bool:
        """Re-read a previously loaded model from its original path.

        Powers ``POST /reload``: swap in a freshly trained artifact without
        restarting the process. Returns False if ``version`` was never
        loaded from a file (e.g. the built-in dummy model).
        """
        path = self._paths.get(version)
        if path is None:
            return False
        self.load(version, path)
        return True

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

    @property
    def versions(self) -> list[str]:
        return sorted(self._models)

    def unload_all(self) -> None:
        self._models.clear()
        self._paths.clear()
        self._default_version = None
