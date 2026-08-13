"""Tests for ModelRegistry, including a real trained model round-trip.

test_load_real_model trains the same LogisticRegression pipeline the app
serves in production (via scripts/train_toy_model.py's train()), persists it
with joblib, and loads it back through the registry. It does not depend on
artifacts/model.joblib already existing on disk, so it passes on a fresh
clone before anyone has run the training script.
"""

from pathlib import Path

import pytest

from app.models.registry import ModelRegistry
from scripts.train_toy_model import train


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("artifacts")
    return out_dir / "model.joblib"


def test_load_real_model(trained_model_path):
    train(trained_model_path, random_state=0)
    assert trained_model_path.is_file()

    registry = ModelRegistry()
    registry.load("production", trained_model_path)

    assert registry.is_ready
    assert "production" in registry.versions
    # The breast cancer dataset has 30 features; predict() must accept a
    # real feature vector of that shape and return a class label (0 or 1).
    features = [0.0] * 30
    prediction = registry.predict(features, version="production")
    assert prediction in (0.0, 1.0)


def test_load_if_present_missing_file(tmp_path):
    registry = ModelRegistry()
    missing = tmp_path / "does-not-exist.joblib"

    loaded = registry.load_if_present("production", missing)

    assert loaded is False
    assert not registry.is_ready


def test_load_if_present_existing_file(trained_model_path):
    registry = ModelRegistry()

    loaded = registry.load_if_present("production", trained_model_path)

    assert loaded is True
    assert registry.default_version == "production"


def test_reload_rereads_from_original_path(trained_model_path):
    registry = ModelRegistry()
    registry.load("production", trained_model_path)

    reloaded = registry.reload("production")

    assert reloaded is True


def test_reload_unknown_version_returns_false():
    registry = ModelRegistry()
    registry.load_default()

    # "dummy" was built in memory, so there's no path to re-read it from.
    assert registry.reload("dummy") is False


def test_unload_all_clears_paths_too(trained_model_path):
    registry = ModelRegistry()
    registry.load("production", trained_model_path)

    registry.unload_all()

    assert not registry.is_ready
    assert registry.reload("production") is False


def test_predict_unknown_version_raises_keyerror():
    registry = ModelRegistry()
    registry.load_default()

    # The /predict route relies on this KeyError to answer 404 for a pinned
    # version that was never loaded; a different exception type would leak
    # through as a 500 instead.
    with pytest.raises(KeyError):
        registry.predict([1.0, 2.0], version="nonexistent")


def test_predict_with_no_models_raises_keyerror():
    registry = ModelRegistry()

    # Before any model is loaded there is no default version to fall back on,
    # so an unqualified predict must fail loudly rather than dereference None.
    with pytest.raises(KeyError):
        registry.predict([1.0, 2.0])
