"""Saveable artifact for a residual-space linear probe (the detector that beat the SAE feature).

Mirrors ``monitor_schema.BehaviorMonitor`` and reuses its ``TargetBehavior`` / ``ModelMetadata``.
Stores the probe *direction* (a residual-stream vector) + bias + calibrated threshold, so it can
be reloaded to score new text or to steer (CAA) along the same direction.
"""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Self

from .recipe_schema import ModelMetadata, RecipeValidationError, TargetBehavior, _slug, utc_now_iso

PROBE_SCHEMA_VERSION = "0.1.0"
PROBE_STATUSES = {"draft", "benchmarked", "validated", "failed"}
PROBE_METHODS = {"diffmeans", "logistic", "ensemble"}


@dataclass
class LinearProbe:
    probe_id: str
    behavior: TargetBehavior
    model: ModelMetadata
    layer: int
    direction: list[float]
    bias: float = 0.0
    threshold: float = 0.0
    method: str = "diffmeans"
    on_policy: bool = False
    evaluation: dict[str, Any] = field(default_factory=dict)
    status: str = "draft"
    schema_version: str = PROBE_SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    created_by: str = "qwen-scope"
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, behavior: TargetBehavior, model: ModelMetadata, layer: int, direction: list[float], *,
               bias: float = 0.0, threshold: float = 0.0, method: str = "diffmeans", on_policy: bool = False,
               evaluation: dict | None = None, status: str = "draft", probe_id: str | None = None) -> Self:
        p = cls(probe_id=probe_id or "", behavior=behavior, model=model, layer=int(layer),
                direction=[float(x) for x in direction], bias=float(bias), threshold=float(threshold),
                method=method, on_policy=bool(on_policy), evaluation=evaluation or {}, status=status)
        p.probe_id = probe_id or p.compute_id()
        p.validate()
        return p

    def compute_id(self) -> str:
        tag = "onpolicy" if self.on_policy else self.method
        return f"{_slug(self.behavior.name)}_probe_{tag}_l{self.layer}_v1"

    def validate(self) -> None:
        if self.schema_version != PROBE_SCHEMA_VERSION:
            raise RecipeValidationError(f"unsupported probe schema_version: {self.schema_version}")
        if self.status not in PROBE_STATUSES:
            raise RecipeValidationError(f"invalid probe status: {self.status}")
        if not self.probe_id or "/" in self.probe_id or ".." in self.probe_id:
            raise RecipeValidationError("probe_id must be a simple non-empty id")
        if not self.behavior.name or not self.behavior.description:
            raise RecipeValidationError("missing behavior")
        if not self.model.model_id:
            raise RecipeValidationError("missing model id")
        if not self.direction:
            raise RecipeValidationError("probe requires a non-empty direction vector")
        if self.method not in PROBE_METHODS:
            raise RecipeValidationError(f"invalid probe method: {self.method}")
        if self.status == "validated":
            if (self.evaluation.get("validation_decision") or {}).get("status") != "validated":
                raise RecipeValidationError("validated probe requires a validated evaluation decision")

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
        p = cls(probe_id=raw["probe_id"], behavior=TargetBehavior(**raw["behavior"]),
                model=ModelMetadata(**raw["model"]), layer=raw["layer"], direction=raw["direction"],
                bias=raw.get("bias", 0.0), threshold=raw.get("threshold", 0.0), method=raw.get("method", "diffmeans"),
                on_policy=raw.get("on_policy", False), evaluation=raw.get("evaluation", {}),
                status=raw.get("status", "draft"), schema_version=raw.get("schema_version", PROBE_SCHEMA_VERSION),
                created_at=raw.get("created_at", utc_now_iso()), created_by=raw.get("created_by", "qwen-scope"),
                provenance=raw.get("provenance", {}))
        p.validate()
        return p

    @classmethod
    def from_json(cls, value: str) -> Self:
        return cls.from_dict(json.loads(value))

    def to_markdown(self) -> str:
        self.validate()
        ev = self.evaluation or {}
        vd = ev.get("validation_decision") or {}
        return (
            f"# Probe: {self.probe_id}\n\nStatus: `{self.status}`\n\n"
            f"## Behavior\n\n**{self.behavior.name}**: {self.behavior.description}\n\n"
            f"## Detector\n\n- Layer {self.layer}; method `{self.method}`{' (on-policy)' if self.on_policy else ''}; "
            f"d={len(self.direction)}; threshold {self.threshold}\n\n"
            f"## Held-out evaluation\n\n"
            f"- AUC: {ev.get('auc')}\n- precision: {ev.get('precision')}\n- recall: {ev.get('recall')}\n"
            f"- F1: {ev.get('f1')}\n- TPR@FPR: {ev.get('tpr_at_fpr')}\n- label-shuffled control AUC: {ev.get('control_auc')}\n"
            f"- verdict: `{vd.get('status', self.status)}` — {vd.get('reason', '')}\n"
        )
