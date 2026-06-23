"""Candidate validation: a new model is only worth promoting if it beats the
incumbent on a holdout drawn from the *current* distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .retrain import evaluate


@dataclass
class ValidationResult:
    candidate_rmse: float
    incumbent_rmse: float
    improvement: float
    promote: bool


def validate_candidate(
    candidate, incumbent, X_holdout: np.ndarray, y_holdout: np.ndarray, min_gain: float = 0.05
) -> ValidationResult:
    """Promote only if the candidate improves RMSE by at least ``min_gain`` (relative)."""
    cand = evaluate(candidate, X_holdout, y_holdout)
    inc = evaluate(incumbent, X_holdout, y_holdout)
    improvement = (inc - cand) / inc if inc else 0.0
    return ValidationResult(
        candidate_rmse=round(cand, 4),
        incumbent_rmse=round(inc, 4),
        improvement=round(improvement, 4),
        promote=improvement >= min_gain,
    )
