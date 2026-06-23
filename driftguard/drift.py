"""Drift detection: data drift (PSI + KS) and concept drift (windowed error).

The closed-loop decisions are driven by these deterministic statistics. An
Evidently HTML report can be generated alongside for human inspection
(optional dependency), but the automation never depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a single feature. >0.2 conventionally signals meaningful drift."""
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_pct = np.clip(ref_hist / ref_hist.sum(), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@dataclass
class DriftReport:
    data_drift: bool
    concept_drift: bool
    max_psi: float
    n_ks_flagged: int
    window_rmse: float
    baseline_rmse: float

    @property
    def any_drift(self) -> bool:
        return self.data_drift or self.concept_drift


def data_drift_report(
    reference_X: np.ndarray,
    current_X: np.ndarray,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.01,
) -> tuple[bool, float, int]:
    """Per-feature PSI + KS test; flags drift if any feature crosses a threshold."""
    n_features = reference_X.shape[1]
    psis, ks_flags = [], 0
    for j in range(n_features):
        psis.append(population_stability_index(reference_X[:, j], current_X[:, j]))
        _, p = stats.ks_2samp(reference_X[:, j], current_X[:, j])
        if p < ks_alpha:
            ks_flags += 1
    max_psi = max(psis) if psis else 0.0
    flagged = (max_psi > psi_threshold) or (ks_flags > 0)
    return flagged, max_psi, ks_flags


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def detect(
    reference_X: np.ndarray,
    current_X: np.ndarray,
    window_rmse: float,
    baseline_rmse: float,
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.01,
    perf_tolerance: float = 1.25,
) -> DriftReport:
    """Combine data-drift and concept-drift signals into one report."""
    data_flag, max_psi, ks_flags = data_drift_report(
        reference_X, current_X, psi_threshold, ks_alpha
    )
    concept_flag = window_rmse > baseline_rmse * perf_tolerance
    return DriftReport(
        data_drift=data_flag,
        concept_drift=concept_flag,
        max_psi=round(max_psi, 4),
        n_ks_flagged=ks_flags,
        window_rmse=round(window_rmse, 4),
        baseline_rmse=round(baseline_rmse, 4),
    )
