"""Environment-driven configuration for DriftGuard."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment: str = os.getenv("DRIFTGUARD_EXPERIMENT", "driftguard")
    registered_model: str = os.getenv("DRIFTGUARD_MODEL", "driftguard-demand")

    # Stream
    seed: int = int(os.getenv("DRIFTGUARD_SEED", "7"))
    batch_size: int = int(os.getenv("DRIFTGUARD_BATCH", "600"))
    n_batches: int = int(os.getenv("DRIFTGUARD_NBATCHES", "24"))
    drift_at: int = int(os.getenv("DRIFTGUARD_DRIFT_AT", "10"))

    # Detection thresholds
    psi_threshold: float = float(os.getenv("DRIFTGUARD_PSI", "0.2"))
    ks_alpha: float = float(os.getenv("DRIFTGUARD_KS_ALPHA", "0.01"))
    perf_tolerance: float = float(os.getenv("DRIFTGUARD_PERF_TOL", "1.25"))


def get_settings() -> Settings:
    return Settings()
