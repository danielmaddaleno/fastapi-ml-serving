"""Dummy model for testing. Returns the mean of the feature vector."""

import numpy as np


class DummyModel:
    """Sklearn-compatible predictor that returns np.mean(X, axis=1)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        # asarray pins the result to an ndarray: np.mean is typed as
        # returning Any because it yields a scalar when axis is None.
        return np.asarray(np.mean(X, axis=1))
