from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .benchmark_controls import REQUIRED_METHODS, MethodSpec, build_method_specs
from .benchmark_metrics import aggregate_metrics, score_for_objective, text_metrics
from .config import config_to_dict
from .generation import generate_text
from .prompt_sets import format_prompt_only
from .recipe_schema import FeatureRecipe, Intervention, ModelMetadata, TargetBehavior, utc_now_iso


class GenerationBackend(Protocol):
    d_sae: int

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int = 0) -> dict[str, Any]:
        ...

    def steer(
        self,
        prompt: str,
        intervention: Intervention,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int = 0,
    ) -> dict[str, Any]:
        ...


@dataclass
class ServiceGenerationBackend:
    service: Any

    @property
    def d_sae(self) -> int:
        return int(self.service.config.d_sae)

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int = 0) -> dict[str, Any]:
        bundle = self.service.ensure_model()
        start = time.perf_counter()
        text, _ = generate_text(bundle, prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        return {"text": text, "latency_seconds": time.perf_counter() - start}

    def steer(
        self,
        prompt: str,
        intervention: Intervention,
        *,
        max_new_tokens: int,
        temperature: float,
        seed: int = 0,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        result = self.service.steer(
            prompt,
            intervention.layer,
            intervention.feature_id,
            intervention.strength,
            max_new_tokens,
            temperature,
            intervention.injection_mode,
        )
        result["text"] = result.get("steered_text", "")
        result["latency_seconds"] = time.perf_counter() - start
        return result


class EchoGenerationBackend:
    def __init__(self, d_sae: int = 32768):
        self.d_sae = d_sae

    def generate(self, prompt: str, *, max_new_tokens: int, temperature: float, seed: int = 0) -> dict[str, Any]:
        if "json" in prompt.lower():
            text = '{"name":"Ada","age":31}'
        elif "concise" in prompt.lower() or "no more than" in prompt.lower():
            text = "Concise answer."
        else:
            text = f"{prompt} This is a deliberately longer baseline response."
        return {"text": text[: max(1, max_new_tokens * 20)], "latency_seconds": 0.0}

    def steer(self, prompt: str, intervention: Intervention, *, max_new_tokens: int, temperature: float, seed: int = 0) -> dict[str, Any]:
        base = self.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)["text"]
        if intervention.strength == 0:
            text = base
        elif intervention.strength < 0:
            text = f"{base} Additional caveats and extra words."
        elif "json" in prompt.lower():
            text = '{"name":"Ada","age":31}'
        else:
            text = "Concise steered answer."
        return {
            "text": text,
            "steered_text": text,
            "unsteered_text": base,
            "hook_fired": intervention.strength != 0,
            "hidden_delta_norm": abs(float(intervention.strength)) if intervention.strength != 0 else 0.0,
            "logits_delta_norm": abs(float(intervention.strength)) * 2.0 if intervention.strength != 0 else 0.0,
            "latency_seconds": 0.0,
        }


def benchmark_id() -> str:
    return f"bench_{utc_now_iso().replace('-', '').replace(':', '').replace('Z', '').replace('T', '_')}"


def _run_method(
    backend: GenerationBackend,
    spec: MethodSpec,
    prompt: str,
    *,
    prompt_only_instruction: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> dict[str, Any]:
    effective_prompt = format_prompt_only(prompt, prompt_only_instruction) if spec.prompt_only else prompt
    if spec.intervention is None:
        result = backend.generate(effective_prompt, max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
        return {"output": result.get("text", ""), "raw": result}
    result = backend.steer(effective_prompt, spec.intervention, max_new_tokens=max_new_tokens, temperature=temperature, seed=seed)
    return {"output": result.get("text") or result.get("steered_text", ""), "raw": result}


def validate_benchmark_result(result: dict[str, Any], *, objective: str = "maximize_rule_score", tolerance: float = 0.05) -> dict[str, Any]:
    aggregate = result.get("aggregate_metrics", {})
    missing = [method for method in REQUIRED_METHODS if method not in aggregate]
    reasons = []
    if missing:
        reasons.append(f"missing required methods: {missing}")
    baseline = score_for_objective(aggregate.get("unsteered_baseline", {}), objective)
    prompt_only = score_for_objective(aggregate.get("prompt_only", {}), objective)
    steering = max(
        score_for_objective(aggregate.get("steering_only", {}), objective),
        score_for_objective(aggregate.get("prompt_plus_steering", {}), objective),
    )
    random_control = score_for_objective(aggregate.get("random_feature_control", {}), objective)
    negative = score_for_objective(aggregate.get("negative_strength_control", {}), objective)
    zero = score_for_objective(aggregate.get("zero_strength_control", {}), objective)
    hooks = [
        metrics
        for prompt_result in result.get("per_prompt_results", [])
        for method, metrics in prompt_result.get("metrics", {}).items()
        if method in {"steering_only", "prompt_plus_steering"}
    ]
    if steering <= baseline:
        reasons.append("steering did not improve over unsteered baseline")
    if steering + tolerance < prompt_only:
        reasons.append("steering was worse than prompt-only beyond tolerance")
    if random_control >= steering - tolerance:
        reasons.append("random-feature control was similar to steering")
    if negative > steering:
        reasons.append("negative-strength control beat steering")
    if abs(zero - baseline) > max(tolerance, 0.1):
        reasons.append("zero-strength control diverged from unsteered baseline")
    if any(not item.get("hook_fired") or item.get("hidden_delta_norm", 0) <= 0 for item in hooks):
        reasons.append("one or more steered generations did not fire hook with positive hidden delta")
    if any(item.get("empty_output") or item.get("excessive_repetition_flag") for item in hooks):
        reasons.append("coherence proxy failed for one or more steered generations")
    if not result.get("config", {}).get("held_out", True):
        reasons.append("held-out validation prompts were not marked as used")
    status = "validated" if not reasons else "benchmarked"
    return {
        "status": status,
        "passed": not reasons,
        "reason": "All conservative validation gates passed." if not reasons else "; ".join(reasons),
        "scores": {
            "unsteered_baseline": baseline,
            "prompt_only": prompt_only,
            "best_steering": steering,
            "random_feature_control": random_control,
            "negative_strength_control": negative,
            "zero_strength_control": zero,
        },
    }


def add_control_delta_metrics(aggregate: dict[str, dict[str, Any]], objective: str) -> None:
    baseline = score_for_objective(aggregate.get("unsteered_baseline", {}), objective)
    steering = max(
        score_for_objective(aggregate.get("steering_only", {}), objective),
        score_for_objective(aggregate.get("prompt_plus_steering", {}), objective),
    )
    if "zero_strength_control" in aggregate:
        zero = score_for_objective(aggregate["zero_strength_control"], objective)
        aggregate["zero_strength_control"]["zero_strength_delta"] = zero - baseline
    if "random_feature_control" in aggregate:
        random_control = score_for_objective(aggregate["random_feature_control"], objective)
        aggregate["random_feature_control"]["random_control_delta"] = random_control - steering
    if "negative_strength_control" in aggregate:
        negative = score_for_objective(aggregate["negative_strength_control"], objective)
        aggregate["negative_strength_control"]["negative_strength_delta"] = negative - steering


def run_benchmark(
    recipe: FeatureRecipe,
    prompts: list[dict[str, Any]],
    backend: GenerationBackend,
    *,
    prompt_only_instruction: str = "",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
    objective: str = "maximize_rule_score",
    prompt_set_id: str = "inline",
    required_terms: list[str] | None = None,
    forbidden_terms: list[str] | None = None,
    max_length_chars: int | None = None,
    held_out: bool = True,
) -> dict[str, Any]:
    recipe.validate()
    intervention = recipe.interventions[0]
    methods = build_method_specs(intervention, d_sae=backend.d_sae, seed=seed)
    per_prompt = []
    by_method_metrics: dict[str, list[dict[str, Any]]] = {method.name: [] for method in methods}
    for prompt_row in prompts:
        outputs = {}
        raw_results = {}
        metrics = {}
        for spec in methods:
            try:
                call = _run_method(
                    backend,
                    spec,
                    prompt_row["prompt"],
                    prompt_only_instruction=prompt_only_instruction,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    seed=seed,
                )
                output = call["output"]
                raw = call["raw"]
                method_metrics = text_metrics(
                    output,
                    required_terms=required_terms,
                    forbidden_terms=forbidden_terms,
                    max_length_chars=max_length_chars,
                )
                method_metrics["latency_seconds"] = float(raw.get("latency_seconds", 0.0))
                method_metrics["hook_fired"] = bool(raw.get("hook_fired", False))
                method_metrics["hidden_delta_norm"] = float(raw.get("hidden_delta_norm", 0.0) or 0.0)
                method_metrics["logits_delta_norm"] = float(raw.get("logits_delta_norm", 0.0) or 0.0)
            except Exception as exc:
                output = ""
                raw = {"error": str(exc)}
                method_metrics = text_metrics("")
                method_metrics["generation_error"] = str(exc)
            outputs[spec.name] = output
            raw_results[spec.name] = raw
            metrics[spec.name] = method_metrics
            by_method_metrics[spec.name].append(method_metrics)
        per_prompt.append(
            {
                "prompt_id": prompt_row.get("id", ""),
                "prompt": prompt_row["prompt"],
                "outputs": outputs,
                "raw_results": raw_results,
                "metrics": metrics,
            }
        )
    aggregate = {method: aggregate_metrics(rows) for method, rows in by_method_metrics.items()}
    add_control_delta_metrics(aggregate, objective)
    result = {
        "benchmark_id": benchmark_id(),
        "recipe_id": recipe.recipe_id,
        "created_at": utc_now_iso(),
        "config": {
            "model_id": recipe.model.model_id,
            "sae_id": recipe.model.sae_id,
            "prompt_set_id": prompt_set_id,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "objective": objective,
            "held_out": held_out,
        },
        "methods": [method.name for method in methods],
        "aggregate_metrics": aggregate,
        "per_prompt_results": per_prompt,
        "pass_fail": {"passed": False, "reasons": []},
    }
    decision = validate_benchmark_result(result, objective=objective)
    result["validation_decision"] = decision
    result["pass_fail"] = {"passed": bool(decision["passed"]), "reasons": [] if decision["passed"] else [decision["reason"]]}
    return result


def attach_benchmark_to_recipe(recipe: FeatureRecipe, result: dict[str, Any]) -> FeatureRecipe:
    recipe.benchmark.update(
        {
            "status": result.get("validation_decision", {}).get("status", "benchmarked"),
            "prompt_set_id": result.get("config", {}).get("prompt_set_id", ""),
            "methods_compared": result.get("methods", []),
            "metrics": result.get("aggregate_metrics", {}),
            "controls": {
                key: result.get("aggregate_metrics", {}).get(key, {})
                for key in ("zero_strength_control", "random_feature_control", "negative_strength_control")
            },
            "summary": result.get("validation_decision", {}).get("reason", ""),
            "validation_decision": result.get("validation_decision", {}),
        }
    )
    if result.get("per_prompt_results"):
        recipe.examples = [
            {
                "prompt": row["prompt"],
                "unsteered": row["outputs"].get("unsteered_baseline", ""),
                "steered": row["outputs"].get("steering_only", ""),
                "prompt_only": row["outputs"].get("prompt_only", ""),
                "prompt_plus_steering": row["outputs"].get("prompt_plus_steering", ""),
            }
            for row in result["per_prompt_results"][:3]
        ]
    if recipe.benchmark["status"] == "validated":
        recipe.mark_validated()
    else:
        recipe.mark_benchmarked(recipe.benchmark.get("validation_decision"))
    return recipe


def save_benchmark_result(result: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def recipe_from_manual(
    *,
    config: Any,
    config_path: str,
    target_behavior: str,
    target_description: str,
    layer: int,
    feature_id: int,
    strength: float,
) -> FeatureRecipe:
    target = target_behavior.replace("_", " ")
    recipe = FeatureRecipe.create(
        target_behavior=TargetBehavior(
            name=target_behavior,
            description=target_description or f"Steer toward {target}.",
            positive_description=f"More {target}.",
            negative_description=f"Less {target}.",
        ),
        model=ModelMetadata(
            model_id=config.model_id,
            sae_id=config.sae_id,
            dtype=config.torch_dtype,
            config_name=config_path,
        ),
        interventions=[Intervention(layer=layer, feature_id=feature_id, strength=strength)],
        created_by="qwen-scope-bench",
    )
    recipe.validate(config_to_dict(config))
    return recipe
