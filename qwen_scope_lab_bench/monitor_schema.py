"""Saveable artifact for a feature-based behavior monitor (the detection counterpart to a recipe).

Mirrors ``recipe_schema.FeatureRecipe`` and reuses its ``TargetBehavior`` / ``ModelMetadata``.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Self

from .recipe_schema import ModelMetadata, RecipeValidationError, TargetBehavior, _slug, utc_now_iso

MONITOR_SCHEMA_VERSION = "0.1.0"
MONITOR_STATUSES = {"draft", "benchmarked", "validated", "failed"}


@dataclass
class BehaviorMonitor:
    monitor_id: str
    behavior: TargetBehavior
    model: ModelMetadata
    layer: int
    features: list[int]
    combine: str = "max"
    threshold: float = 0.0
    top_k: int = 3
    evaluation: dict[str, Any] = field(default_factory=dict)
    examples: list[dict[str, Any]] = field(default_factory=list)
    status: str = "draft"
    schema_version: str = MONITOR_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    created_by: str = "qwen-scope"
    discovery: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, behavior: TargetBehavior, model: ModelMetadata, layer: int, features: list[int], *,
               threshold: float = 0.0, top_k: int = 3, combine: str = "max",
               evaluation: dict | None = None, examples: list | None = None,
               discovery: dict | None = None, status: str = "draft", monitor_id: str | None = None) -> Self:
        m = cls(monitor_id=monitor_id or "", behavior=behavior, model=model, layer=int(layer),
                features=[int(f) for f in features], threshold=float(threshold), top_k=int(top_k), combine=combine,
                evaluation=evaluation or {}, examples=examples or [], discovery=discovery or {}, status=status)
        m.monitor_id = monitor_id or m.compute_id()
        m.validate()
        return m

    def compute_id(self) -> str:
        return f"{_slug(self.behavior.name)}_monitor_l{self.layer}_v1"

    def validate(self) -> None:
        if self.schema_version != MONITOR_SCHEMA_VERSION:
            raise RecipeValidationError(f"unsupported monitor schema_version: {self.schema_version}")
        if self.status not in MONITOR_STATUSES:
            raise RecipeValidationError(f"invalid monitor status: {self.status}")
        if not self.monitor_id or "/" in self.monitor_id or ".." in self.monitor_id:
            raise RecipeValidationError("monitor_id must be a simple non-empty id")
        if not self.behavior.name or not self.behavior.description:
            raise RecipeValidationError("missing behavior")
        if not self.model.model_id:
            raise RecipeValidationError("missing model id")
        if not self.features:
            raise RecipeValidationError("monitor requires at least one feature")
        if any(int(f) < 0 for f in self.features):
            raise RecipeValidationError("invalid feature id")
        if self.combine not in {"max"}:
            raise RecipeValidationError(f"invalid combine mode: {self.combine}")
        if self.status == "validated":
            if (self.evaluation.get("validation_decision") or {}).get("status") != "validated":
                raise RecipeValidationError("validated monitor requires a validated evaluation decision")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["behavior"] = asdict(self.behavior)
        d["model"] = asdict(self.model)
        return d

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw = deepcopy(data)
        m = cls(monitor_id=raw["monitor_id"], behavior=TargetBehavior(**raw["behavior"]),
                model=ModelMetadata(**raw["model"]), layer=raw["layer"], features=raw["features"],
                combine=raw.get("combine", "max"), threshold=raw.get("threshold", 0.0), top_k=raw.get("top_k", 3),
                evaluation=raw.get("evaluation", {}), examples=raw.get("examples", []),
                status=raw.get("status", "draft"), schema_version=raw.get("schema_version", MONITOR_SCHEMA_VERSION),
                created_at=raw.get("created_at", utc_now_iso()), created_by=raw.get("created_by", "qwen-scope"),
                discovery=raw.get("discovery", {}), provenance=raw.get("provenance", {}))
        m.validate()
        return m

    @classmethod
    def from_json(cls, value: str) -> Self:
        return cls.from_dict(json.loads(value))

    def to_markdown(self) -> str:
        self.validate()
        ev = self.evaluation or {}
        vd = ev.get("validation_decision") or {}
        feats = ", ".join(f"#{f}" for f in self.features)
        return (
            f"# Monitor: {self.monitor_id}\n\nStatus: `{self.status}`\n\n"
            f"## Behavior\n\n**{self.behavior.name}**: {self.behavior.description}\n\n"
            f"## Detector\n\n- Layer {self.layer}; features {feats}; combine `{self.combine}`; threshold {self.threshold}\n\n"
            f"## Held-out evaluation\n\n"
            f"- AUC: {ev.get('auc')}\n- precision: {ev.get('precision')}\n- recall: {ev.get('recall')}\n"
            f"- F1: {ev.get('f1')}\n- FPR: {ev.get('fpr')}\n- random-feature control AUC: {ev.get('control_auc')}\n"
            f"- n_pos / n_neg: {ev.get('n_pos')} / {ev.get('n_neg')}\n"
            f"- verdict: `{vd.get('status', self.status)}` — {vd.get('reason', '')}\n"
        )
