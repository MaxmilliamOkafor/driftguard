"""Train and retrain the demand model."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .drift import rmse


def train_model(X: np.ndarray, y: np.ndarray, seed: int = 7) -> RandomForestRegressor:
    """Fit a fresh model on the given window."""
    model = RandomForestRegressor(n_estimators=120, max_depth=8, random_state=seed, n_jobs=1)
    model.fit(X, y)
    return model


def evaluate(model, X: np.ndarray, y: np.ndarray) -> float:
    return rmse(y, model.predict(X))
