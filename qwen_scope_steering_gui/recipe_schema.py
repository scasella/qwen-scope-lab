from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Self


SCHEMA_VERSION = "0.1.0"
RECIPE_STATUSES = {"draft", "candidate", "benchmarked", "validated", "failed", "blocked"}


class RecipeValidationError(ValueError):
    pass


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "recipe"


@dataclass
class TargetBehavior:
    name: str
    description: str
    positive_description: str = ""
    negative_description: str = ""


@dataclass
class ModelMetadata:
    model_id: str
    sae_id: str
    dtype: str = ""
    config_name: str = ""


@dataclass
class Intervention:
    layer: int
    feature_id: int
    strength: float
    sign: str = "positive"
    injection_mode: str = "all_positions"
    position_policy: str = "all_generated_tokens"


@dataclass
class ManifoldSpec:
    """A concept-manifold steer: traverse a concept's residual-stream manifold from
    ``source`` to ``target`` at ``layer`` along ``path`` (manifold | linear | pullback).
    Unlike a feature Intervention, this replaces the concept token's residual with a point
    on the fitted manifold, so it needs no SAE feature id."""
    concept: str
    source: str
    target: str
    layer: int
    path: str = "manifold"
    n_waypoints: int = 5


@dataclass
class DiscoveryMetadata:
    method: str = "manual"
    positive_prompts: list[str] = field(default_factory=list)
    negative_prompts: list[str] = field(default_factory=list)
    candidate_layers: list[int] = field(default_factory=list)
    candidate_count: int = 0
    ranking_metric: str = ""


@dataclass
class FeatureRecipe:
    recipe_id: str
    target_behavior: TargetBehavior
    model: ModelMetadata
    interventions: list[Intervention]
    kind: str = "feature"  # "feature" (SAE-feature steer) | "manifold" (concept-manifold steer)
    manifold: "ManifoldSpec | None" = None
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=utc_now_iso)
    created_by: str = "qwen-scope"
    status: str = "draft"
    discovery: DiscoveryMetadata = field(default_factory=DiscoveryMetadata)
    benchmark: dict[str, Any] = field(default_factory=lambda: {"status": "draft", "prompt_set_id": "", "methods_compared": [], "metrics": {}, "controls": {}, "summary": ""})
    examples: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=lambda: {"json_path": "", "markdown_path": "", "results_path": ""})
    provenance: dict[str, Any] = field(default_factory=lambda: {"git_commit": "", "command": "", "modal_gpu": "", "python_version": "", "torch_version": "", "transformers_version": "", "seed": 0})

    @classmethod
    def create(
        cls,
        target_behavior: TargetBehavior,
        model: ModelMetadata,
        interventions: list[Intervention],
        *,
        recipe_id: str | None = None,
        created_by: str = "qwen-scope",
        discovery: DiscoveryMetadata | None = None,
    ) -> Self:
        provisional = cls(
            recipe_id=recipe_id or "",
            target_behavior=target_behavior,
            model=model,
            interventions=interventions,
            created_by=created_by,
            discovery=discovery or DiscoveryMetadata(),
        )
        provisional.recipe_id = recipe_id or provisional.compute_id()
        provisional.validate()
        return provisional

    @classmethod
    def create_manifold(
        cls,
        target_behavior: TargetBehavior,
        model: ModelMetadata,
        manifold: "ManifoldSpec",
        *,
        recipe_id: str | None = None,
        created_by: str = "qwen-scope",
        discovery: DiscoveryMetadata | None = None,
    ) -> Self:
        provisional = cls(
            recipe_id=recipe_id or "",
            target_behavior=target_behavior,
            model=model,
            interventions=[],
            kind="manifold",
            manifold=manifold,
            created_by=created_by,
            discovery=discovery or DiscoveryMetadata(method="manifold"),
        )
        provisional.recipe_id = recipe_id or provisional.compute_id()
        provisional.validate()
        return provisional

    def compute_id(self) -> str:
        if self.kind == "manifold" and self.manifold:
            m = self.manifold
            return f"{_slug(m.concept)}_{_slug(m.source)}_to_{_slug(m.target)}_l{m.layer}_v1"
        if not self.interventions:
            return f"{_slug(self.target_behavior.name)}_v1"
        first = self.interventions[0]
        return f"{_slug(self.target_behavior.name)}_l{first.layer}_f{first.feature_id}_v1"

    def validate(self, config_metadata: dict[str, Any] | None = None) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RecipeValidationError(f"unsupported schema_version: {self.schema_version}")
        if self.status not in RECIPE_STATUSES:
            raise RecipeValidationError(f"invalid recipe status: {self.status}")
        if not self.recipe_id or "/" in self.recipe_id or ".." in self.recipe_id:
            raise RecipeValidationError("recipe_id must be a simple non-empty id")
        if not self.target_behavior.name or not self.target_behavior.description:
            raise RecipeValidationError("missing target behavior")
        if not self.model.model_id:
            raise RecipeValidationError("missing model id")
        if config_metadata and config_metadata.get("model_id") and config_metadata["model_id"] != self.model.model_id:
            raise RecipeValidationError("recipe model id does not match config metadata")
        if self.kind == "manifold":
            self._validate_manifold(config_metadata)
        else:
            self._validate_feature(config_metadata)
        if self.status == "validated":
            self._validate_validated_evidence()

    def _validate_feature(self, config_metadata: dict[str, Any] | None = None) -> None:
        if not self.model.sae_id:
            raise RecipeValidationError("missing SAE id")
        if not self.interventions:
            raise RecipeValidationError("at least one intervention is required")
        num_layers = int(config_metadata.get("num_layers", 0)) if config_metadata else None
        d_sae = int(config_metadata.get("d_sae", 0)) if config_metadata else None
        if config_metadata and config_metadata.get("sae_id") and config_metadata["sae_id"] != self.model.sae_id:
            raise RecipeValidationError("recipe SAE id does not match config metadata")
        for intervention in self.interventions:
            if intervention.layer < 0 or (num_layers is not None and num_layers > 0 and intervention.layer >= num_layers):
                raise RecipeValidationError("invalid layer")
            if intervention.feature_id < 0 or (d_sae is not None and d_sae > 0 and intervention.feature_id >= d_sae):
                raise RecipeValidationError("invalid feature id")
            if not -100.0 <= float(intervention.strength) <= 100.0:
                raise RecipeValidationError("invalid strength")
            if intervention.sign not in {"positive", "negative", "zero", "control"}:
                raise RecipeValidationError(f"invalid intervention sign: {intervention.sign}")

    def _validate_manifold(self, config_metadata: dict[str, Any] | None = None) -> None:
        m = self.manifold
        if m is None:
            raise RecipeValidationError("manifold recipe requires a manifold spec")
        if not m.concept or not m.source or not m.target:
            raise RecipeValidationError("manifold spec needs concept, source and target")
        if m.path not in {"manifold", "linear", "pullback"}:
            raise RecipeValidationError(f"invalid manifold path: {m.path}")
        if int(m.n_waypoints) < 2:
            raise RecipeValidationError("manifold spec needs at least 2 waypoints")
        num_layers = int(config_metadata.get("num_layers", 0)) if config_metadata else None
        if m.layer < 0 or (num_layers is not None and num_layers > 0 and m.layer >= num_layers):
            raise RecipeValidationError("invalid layer")

    def _validate_validated_evidence(self) -> None:
        decision = self.benchmark.get("validation_decision") or {}
        if decision.get("status") != "validated":
            raise RecipeValidationError("validated recipes require validated benchmark decision evidence")
        if self.kind == "manifold":
            legs = self.benchmark.get("legs") or {}
            if not {"manifold", "linear", "pullback"} & set(legs):
                raise RecipeValidationError("validated manifold recipes require benchmark legs (manifold/linear/pullback energies)")
            if not self.examples:
                raise RecipeValidationError("validated recipes require before/after examples")
            return
        required_methods = {
            "unsteered_baseline",
            "prompt_only",
            "steering_only",
            "prompt_plus_steering",
            "zero_strength_control",
            "random_feature_control",
            "negative_strength_control",
        }
        methods = set(self.benchmark.get("methods_compared") or [])
        if not required_methods.issubset(methods):
            raise RecipeValidationError("validated recipes require all required benchmark methods")
        controls = self.benchmark.get("controls") or {}
        for control in ("zero_strength_control", "random_feature_control", "negative_strength_control"):
            if control not in controls:
                raise RecipeValidationError(f"validated recipes require {control} evidence")
        if not self.examples:
            raise RecipeValidationError("validated recipes require before/after examples")

    def mark_unvalidated(self) -> None:
        self.status = "draft"
        self.benchmark["status"] = "draft"

    def mark_candidate(self) -> None:
        if not self.benchmark.get("methods_compared"):
            raise RecipeValidationError("candidate recipes require at least one benchmark run")
        self.status = "candidate"
        self.benchmark["status"] = "candidate"

    def mark_benchmarked(self, validation_decision: dict[str, Any] | None = None) -> None:
        self.status = "benchmarked"
        self.benchmark["status"] = "benchmarked"
        if validation_decision:
            self.benchmark["validation_decision"] = validation_decision

    def mark_validated(self) -> None:
        previous = self.status
        self.status = "validated"
        self.benchmark["status"] = "validated"
        try:
            self.validate()
        except Exception:
            self.status = previous
            raise

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["target_behavior"] = asdict(self.target_behavior)
        data["model"] = asdict(self.model)
        data["interventions"] = [asdict(intervention) for intervention in self.interventions]
        data["discovery"] = asdict(self.discovery)
        return data

    def to_json(self) -> str:
        self.validate()
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        raw = deepcopy(data)
        recipe = cls(
            schema_version=raw.get("schema_version", SCHEMA_VERSION),
            recipe_id=raw["recipe_id"],
            created_at=raw.get("created_at", utc_now_iso()),
            created_by=raw.get("created_by", "qwen-scope"),
            status=raw.get("status", raw.get("benchmark", {}).get("status", "draft")),
            target_behavior=TargetBehavior(**raw["target_behavior"]),
            model=ModelMetadata(**raw["model"]),
            interventions=[Intervention(**item) for item in raw.get("interventions", [])],
            kind=raw.get("kind", "feature"),
            manifold=ManifoldSpec(**raw["manifold"]) if raw.get("manifold") else None,
            discovery=DiscoveryMetadata(**raw.get("discovery", {})),
            benchmark=raw.get("benchmark", {}),
            examples=raw.get("examples", []),
            limitations=raw.get("limitations", []),
            side_effects=raw.get("side_effects", []),
            artifacts=raw.get("artifacts", {"json_path": "", "markdown_path": "", "results_path": ""}),
            provenance=raw.get("provenance", {"git_commit": "", "command": "", "modal_gpu": "", "python_version": "", "torch_version": "", "transformers_version": "", "seed": 0}),
        )
        recipe.validate()
        return recipe

    @classmethod
    def from_json(cls, value: str) -> Self:
        return cls.from_dict(json.loads(value))

    def to_markdown(self) -> str:
        self.validate()
        if self.kind == "manifold" and self.manifold:
            m = self.manifold
            steer_header = "Manifold Steer"
            legs = self.benchmark.get("legs") or {}
            leg_rows = "".join(
                f"| {name} | {legs[name].get('mean_energy', '—')} | {legs[name].get('recovered_r', '—')} |\n"
                for name in ("manifold", "linear", "pullback") if name in legs
            )
            leg_table = f"\n\n| path | behavior-energy (lower=more faithful) | recovers ℳ_h (higher=traces geometry) |\n|---|---|---|\n{leg_rows}" if leg_rows else ""
            intervention_lines = (
                f"- Concept `{m.concept}`: steer **{m.source} → {m.target}** along the **{m.path}** path "
                f"at layer {m.layer} ({m.n_waypoints} waypoints)" + leg_table
            )
        else:
            steer_header = "Interventions"
            intervention_lines = "\n".join(
                f"- Layer {i.layer}, feature {i.feature_id}, strength {i.strength}, mode `{i.injection_mode}`"
                for i in self.interventions
            )
        examples = "\n".join(
            f"### Example {index}\n\nPrompt: `{example.get('prompt', '')}`\n\nUnsteered:\n\n```text\n{example.get('unsteered', '')}\n```\n\nSteered:\n\n```text\n{example.get('steered', '')}\n```"
            for index, example in enumerate(self.examples, start=1)
        )
        limitations = "\n".join(f"- {item}" for item in self.limitations) or "- None recorded."
        side_effects = "\n".join(f"- {item}" for item in self.side_effects) or "- None recorded."
        decision = self.benchmark.get("validation_decision") or {"status": self.status, "reason": self.benchmark.get("summary", "")}
        return (
            f"# Recipe: {self.recipe_id}\n\n"
            f"Status: `{self.status}`\n\n"
            f"## Target Behavior\n\n"
            f"**{self.target_behavior.name}**: {self.target_behavior.description}\n\n"
            f"Positive: {self.target_behavior.positive_description or 'Not specified.'}\n\n"
            f"Negative: {self.target_behavior.negative_description or 'Not specified.'}\n\n"
            f"## Model and SAE\n\n"
            f"- Model: `{self.model.model_id}`\n"
            f"- SAE: `{self.model.sae_id}`\n"
            f"- Dtype: `{self.model.dtype}`\n"
            f"- Config: `{self.model.config_name}`\n\n"
            f"## {steer_header}\n\n{intervention_lines}\n\n"
            f"## Benchmark Summary\n\n"
            f"- Prompt set: `{self.benchmark.get('prompt_set_id', '')}`\n"
            f"- Methods: {', '.join(self.benchmark.get('methods_compared', []))}\n"
            f"- Summary: {self.benchmark.get('summary', '')}\n"
            f"- Validation status: `{decision.get('status', self.status)}`\n"
            f"- Reason: {decision.get('reason', '')}\n\n"
            f"## Examples\n\n{examples or 'No examples recorded.'}\n\n"
            f"## Limitations\n\n{limitations}\n\n"
            f"## Side Effects\n\n{side_effects}\n\n"
            f"## Provenance\n\n"
            f"```json\n{json.dumps(self.provenance, indent=2, sort_keys=True)}\n```\n"
        )
