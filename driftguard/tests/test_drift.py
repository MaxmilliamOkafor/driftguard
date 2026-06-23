"""Drift detection unit tests."""

import numpy as np

from driftguard.drift import data_drift_report, detect, population_stability_index


def test_psi_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(0, 1, 5000)
    assert population_stability_index(a, b) < 0.1


def test_psi_high_for_shifted_distribution():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    b = rng.normal(3, 1, 5000)
    assert population_stability_index(a, b) > 0.25


def test_no_data_drift_when_identical():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (2000, 4))
    Y = rng.normal(0, 1, (2000, 4))
    flagged, _, _ = data_drift_report(X, Y)
    assert flagged is False


def test_concept_drift_flag_from_performance():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (500, 4))
    rep = detect(X, X, window_rmse=5.0, baseline_rmse=1.0, perf_tolerance=1.25)
    assert rep.concept_drift is True
    assert rep.any_drift is True
