"""Validation + promotion decision tests."""

import numpy as np

from driftguard.retrain import train_model
from driftguard.validate import validate_candidate


def _data(seed, w):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (800, 4))
    y = X @ np.array(w) + rng.normal(0, 1, 800)
    return X, y


def test_better_candidate_is_promoted():
    Xc, yc = _data(0, [2, -1, 2.5, -1.5])  # new concept
    Xo, yo = _data(1, [2, -1, 0.5, 3.0])  # old concept
    incumbent = train_model(Xo, yo)  # trained on old concept
    candidate = train_model(Xc, yc)  # trained on new concept
    res = validate_candidate(candidate, incumbent, Xc, yc)
    assert res.promote is True
    assert res.candidate_rmse < res.incumbent_rmse


def test_worse_candidate_is_rejected():
    Xc, yc = _data(0, [2, -1, 2.5, -1.5])
    good = train_model(Xc, yc)
    rng = np.random.default_rng(9)
    junk_X = rng.normal(0, 1, (200, 4))
    junk_y = rng.normal(0, 1, 200)
    junk = train_model(junk_X, junk_y)  # trained on noise
    res = validate_candidate(junk, good, Xc, yc)
    assert res.promote is False
