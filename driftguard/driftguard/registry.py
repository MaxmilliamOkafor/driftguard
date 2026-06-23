"""A small model registry with MLflow experiment tracking.

MLflow's *file* store can't host the full Model Registry (that needs a DB
backend), so we keep a lightweight JSON registry for version + production
pointer and joblib for artifacts, while logging every train/validate/promote
event to MLflow for metrics and lineage. This keeps the demo runnable on a free
tier with zero services, without giving up experiment tracking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib

REG_DIR = Path("models")
REG_FILE = REG_DIR / "registry.json"


@dataclass
class ModelVersion:
    version: int
    path: str
    rmse: float
    created_step: int
    stage: str  # "production" | "archived" | "candidate"


class Registry:
    def __init__(self, root: Path = REG_DIR) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = root / "registry.json"
        if self.file.exists():
            self._state = json.loads(self.file.read_text())
        else:
            self._state = {"next_version": 1, "versions": []}

    def _save(self) -> None:
        self.file.write_text(json.dumps(self._state, indent=2))

    def register(self, model, rmse: float, step: int, stage: str = "candidate") -> ModelVersion:
        v = self._state["next_version"]
        path = str(self.root / f"model_v{v}.joblib")
        joblib.dump(model, path)
        mv = ModelVersion(version=v, path=path, rmse=rmse, created_step=step, stage=stage)
        self._state["versions"].append(asdict(mv))
        self._state["next_version"] = v + 1
        self._save()
        return mv

    def production(self) -> ModelVersion | None:
        for d in self._state["versions"]:
            if d["stage"] == "production":
                return ModelVersion(**d)
        return None

    def load(self, version: int):
        for d in self._state["versions"]:
            if d["version"] == version:
                return joblib.load(d["path"])
        raise KeyError(f"version {version} not found")

    def promote(self, version: int) -> None:
        for d in self._state["versions"]:
            if d["stage"] == "production":
                d["stage"] = "archived"
            if d["version"] == version:
                d["stage"] = "production"
        self._save()

    def lineage(self) -> list[ModelVersion]:
        return [ModelVersion(**d) for d in self._state["versions"]]
