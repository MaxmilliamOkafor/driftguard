"""End-to-end self-healing loop test."""

import os
import tempfile

from driftguard.config import Settings
from driftguard.monitor import run_loop


def test_loop_detects_and_recovers(monkeypatch):
    # isolate the registry + mlflow store in a temp dir
    tmp = tempfile.mkdtemp()
    monkeypatch.chdir(tmp)
    s = Settings(
        mlflow_tracking_uri=f"file:{os.path.join(tmp, 'mlruns')}",
        n_batches=20,
        drift_at=10,
        batch_size=600,
    )
    summary = run_loop(s)
    assert summary.drift_detected_step is not None
    assert summary.detection_lead_time is not None and summary.detection_lead_time <= 2
    assert summary.recovered_step is not None
    assert summary.human_interventions == 0
    assert summary.performance_retained_pct is not None
    assert summary.performance_retained_pct > 70
