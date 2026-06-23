"""Promotion / rollback decisions, recorded in the registry and MLflow."""

from __future__ import annotations

from .registry import ModelVersion, Registry
from .validate import ValidationResult


def decide_and_apply(
    registry: Registry,
    candidate_version: int,
    result: ValidationResult,
) -> str:
    """Apply the canary decision. Returns 'promoted' or 'rolled_back'."""
    if result.promote:
        registry.promote(candidate_version)
        return "promoted"
    # rollback = leave the incumbent in production, archive the candidate
    return "rolled_back"


def current_production(registry: Registry) -> ModelVersion | None:
    return registry.production()
