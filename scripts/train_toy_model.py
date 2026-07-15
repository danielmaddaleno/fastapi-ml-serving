"""Train and persist the model the app serves at startup.

Fits a plain LogisticRegression on the breast cancer Wisconsin dataset
bundled with scikit-learn (569 rows, 30 numeric features, binary target).
It is real measured data, not something made up for this repo, and it is
small enough to train in under a second, which is the point: this is a demo
serving path, not a model worth writing home about.

Run it from the repo root:

    python scripts/train_toy_model.py

It writes artifacts/model.joblib. The app loads that file as the
"production" model version on startup, alongside the built-in dummy model,
if it exists (app/models/registry.py, ModelRegistry.load_if_present).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(output_path: Path, random_state: int = 0) -> Pipeline:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=random_state
    )

    # The 30 features are measured on very different scales (areas vs.
    # smoothness ratios), so scaling first is what lets lbfgs converge
    # instead of just cranking up max_iter.
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"held-out accuracy: {accuracy:.3f} ({len(X_test)} test rows)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"wrote {output_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/model.joblib"),
        help="Where to write the trained model (default: artifacts/model.joblib)",
    )
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    train(args.output, random_state=args.random_state)


if __name__ == "__main__":
    main()
