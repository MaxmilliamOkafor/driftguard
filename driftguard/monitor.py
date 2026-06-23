"""The closed self-healing loop: monitor -> retrain -> validate -> promote/rollback.

No human in the loop. Every event is appended to an event log and logged to
MLflow. Running this module prints a JSON summary with the headline metrics.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from .config import Settings, get_settings
from .drift import detect, rmse
from .promote import decide_and_apply
from .registry import Registry
from .retrain import evaluate, train_model
from .validate import validate_candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("driftguard")


@dataclass
class Event:
    step: int
    kind: str  # baseline | drift_detected | retrain | promoted | rolled_back | ok
    detail: dict = field(default_factory=dict)


@dataclass
class LoopSummary:
    baseline_rmse: float
    drift_injected_step: int
    drift_detected_step: int | None
    recovered_step: int | None
    detection_lead_time: int | None
    recovery_time: int | None
    rmse_after_recovery: float | None
    performance_retained_pct: float | None
    human_interventions: int
    n_versions: int
    events: list[dict]


def _mlflow(settings: Settings):
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.experiment)
        return mlflow
    except Exception as exc:  # noqa: BLE001
        logger.warning("MLflow unavailable (%s); continuing without tracking", exc)
        return None


def run_loop(settings: Settings | None = None) -> LoopSummary:
    s = settings or get_settings()
    from data.stream import DriftingStream  # local import to avoid hard dep at import

    stream = DriftingStream(
        seed=s.seed, batch_size=s.batch_size, n_batches=s.n_batches, drift_at=s.drift_at
    )
    mlflow = _mlflow(s)

    # --- baseline ---
    ref = stream.reference(n=2000)
    split = int(0.8 * len(ref.y))
    base_model = train_model(ref.X[:split], ref.y[:split], seed=s.seed)
    baseline_rmse = evaluate(base_model, ref.X[split:], ref.y[split:])

    registry = Registry()
    v1 = registry.register(base_model, baseline_rmse, step=0, stage="candidate")
    registry.promote(v1.version)

    events: list[Event] = [Event(0, "baseline", {"rmse": round(baseline_rmse, 4)})]
    if mlflow:
        with mlflow.start_run(run_name="baseline"):
            mlflow.log_metric("rmse", baseline_rmse)
            mlflow.log_param("model_version", v1.version)

    drift_detected_step: int | None = None
    recovered_step: int | None = None
    rmse_after_recovery: float | None = None

    for batch in stream:
        prod = registry.production()
        model = registry.load(prod.version)
        window_rmse = rmse(batch.y, model.predict(batch.X))

        report = detect(
            reference_X=ref.X,
            current_X=batch.X,
            window_rmse=window_rmse,
            baseline_rmse=baseline_rmse,
            psi_threshold=s.psi_threshold,
            ks_alpha=s.ks_alpha,
            perf_tolerance=s.perf_tolerance,
        )

        if mlflow:
            with mlflow.start_run(run_name=f"monitor_step_{batch.step}"):
                mlflow.log_metric("window_rmse", window_rmse, step=batch.step)
                mlflow.log_metric("max_psi", report.max_psi, step=batch.step)
                mlflow.log_metric("concept_drift", int(report.concept_drift), step=batch.step)

        if not report.any_drift:
            events.append(Event(batch.step, "ok", {"rmse": round(window_rmse, 4)}))
            if (
                drift_detected_step is not None
                and recovered_step is None
                and window_rmse <= baseline_rmse * s.perf_tolerance
            ):
                recovered_step = batch.step
                rmse_after_recovery = window_rmse
            continue

        # --- drift! self-heal ---
        if drift_detected_step is None:
            drift_detected_step = batch.step
        events.append(
            Event(
                batch.step,
                "drift_detected",
                {
                    "max_psi": report.max_psi,
                    "window_rmse": round(window_rmse, 4),
                    "concept": report.concept_drift,
                    "data": report.data_drift,
                },
            )
        )

        # Retrain on the full recent (drifted) window; validate on a fresh
        # holdout drawn from the CURRENT distribution.
        holdout = stream._make(step=batch.step, n=400)  # noqa: SLF001 - same generator
        candidate = train_model(batch.X, batch.y, seed=s.seed)
        cand_v = registry.register(
            candidate,
            evaluate(candidate, holdout.X, holdout.y),
            step=batch.step,
            stage="candidate",
        )
        events.append(Event(batch.step, "retrain", {"candidate_version": cand_v.version}))
        result = validate_candidate(candidate, model, holdout.X, holdout.y)
        decision = decide_and_apply(registry, cand_v.version, result)
        events.append(Event(batch.step, decision, asdict(result)))

        if mlflow:
            with mlflow.start_run(run_name=f"heal_step_{batch.step}"):
                mlflow.log_metrics(
                    {
                        "candidate_rmse": result.candidate_rmse,
                        "incumbent_rmse": result.incumbent_rmse,
                        "improvement": result.improvement,
                    }
                )
                mlflow.log_param("decision", decision)

        if decision == "promoted":
            new_prod = registry.load(registry.production().version)
            post_rmse = rmse(holdout.y, new_prod.predict(holdout.X))
            if post_rmse <= baseline_rmse * s.perf_tolerance and recovered_step is None:
                recovered_step = batch.step
                rmse_after_recovery = post_rmse

    detection_lead = drift_detected_step - s.drift_at if drift_detected_step is not None else None
    recovery_time = (
        recovered_step - drift_detected_step
        if recovered_step is not None and drift_detected_step is not None
        else None
    )
    retained = (
        round(100.0 * baseline_rmse / rmse_after_recovery, 1) if rmse_after_recovery else None
    )

    return LoopSummary(
        baseline_rmse=round(baseline_rmse, 4),
        drift_injected_step=s.drift_at,
        drift_detected_step=drift_detected_step,
        recovered_step=recovered_step,
        detection_lead_time=detection_lead,
        recovery_time=recovery_time,
        rmse_after_recovery=round(rmse_after_recovery, 4) if rmse_after_recovery else None,
        performance_retained_pct=retained,
        human_interventions=0,
        n_versions=len(registry.lineage()),
        events=[asdict(e) for e in events],
    )


if __name__ == "__main__":  # pragma: no cover
    summary = run_loop()
    d = asdict(summary)
    d["events"] = f"{len(d['events'])} events (omitted)"
    print(json.dumps(d, indent=2))
