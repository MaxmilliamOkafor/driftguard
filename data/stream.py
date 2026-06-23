"""Deterministic synthetic data stream with injectable drift.

Models a demand-forecasting feed: four features drive a target via a linear
relationship plus noise. At ``drift_at`` the *concept* changes — the mapping
from features to target shifts — which is exactly the failure mode a monitor
must catch. Because everything is seeded, the demo is fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_FEATURES = 4
_BASE_WEIGHTS = np.array([2.0, -1.0, 0.5, 3.0])
_BASE_BIAS = 5.0
# Post-drift concept: weights and bias shift, breaking the incumbent model.
_DRIFT_WEIGHTS = np.array([2.0, -1.0, 2.5, -1.5])
_DRIFT_BIAS = 9.0


@dataclass
class Batch:
    step: int
    X: np.ndarray
    y: np.ndarray
    drifted: bool


class DriftingStream:
    """Yields batches; concept drifts at ``drift_at`` and stays drifted."""

    def __init__(
        self,
        seed: int = 7,
        batch_size: int = 200,
        n_batches: int = 30,
        drift_at: int = 12,
        noise: float = 1.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.drift_at = drift_at
        self.noise = noise

    def _make(self, step: int, n: int) -> Batch:
        X = self.rng.normal(0.0, 1.0, size=(n, N_FEATURES))
        drifted = step >= self.drift_at
        w = _DRIFT_WEIGHTS if drifted else _BASE_WEIGHTS
        b = _DRIFT_BIAS if drifted else _BASE_BIAS
        y = X @ w + b + self.rng.normal(0.0, self.noise, size=n)
        return Batch(step=step, X=X, y=y, drifted=drifted)

    def reference(self, n: int = 2000) -> Batch:
        """A pre-drift reference sample used as the drift baseline."""
        return self._make(step=0, n=n)

    def __iter__(self):
        for step in range(self.n_batches):
            yield self._make(step=step, n=self.batch_size)
