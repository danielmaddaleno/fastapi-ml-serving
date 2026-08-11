"""Tests for the built-in DummyModel.

DummyModel is what the service falls back to at startup before a trained
artifact exists, so it is what /predict and /health answer with on a fresh
clone. The route-level tests exercise it indirectly; these pin down its
numeric contract directly: predict returns the per-row mean and keeps one
output per input row.
"""

import numpy as np

from app.models.dummy import DummyModel


def test_predict_returns_row_mean_for_single_sample():
    model = DummyModel()

    result = model.predict(np.array([[1.0, 2.0, 3.0, 4.0]]))

    assert result.shape == (1,)
    assert result[0] == 2.5


def test_predict_returns_one_mean_per_row_for_a_batch():
    model = DummyModel()

    result = model.predict(np.array([[1.0, 3.0], [10.0, 20.0]]))

    assert result.shape == (2,)
    np.testing.assert_allclose(result, [2.0, 15.0])
