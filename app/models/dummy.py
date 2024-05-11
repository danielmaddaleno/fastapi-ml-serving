# -*- coding: utf-8 -*-
"""Dummy — core implementation."""
"""Dummy model for testing – returns the mean of the feature vector."""

import numpy as np


class DummyModel:
    """Sklearn-compatible predictor that returns np.mean(X, axis=1)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean(X, axis=1)
