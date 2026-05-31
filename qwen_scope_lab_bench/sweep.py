from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark import GenerationBackend, run_benchmark
from .benchmark_metrics import score_for_objective
from .recipe_schema import FeatureRecipe, Intervention, utc_now_iso


DEFAULT_STRENGTHS = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0]


def strength_grid(strengths: list[float] | None = None) -> list[float]:
    values = list(strengths or DEFAULT_STRENGTHS)
    if 0.0 not in values:
        values.append(0.0)
    return sorted(set(float(value) for value in values))


def select_best_setting(results: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> tuple[float, float]:
        metrics = row.get("aggregate_metrics", {}).get("prompt_plus_steering") or row.get("aggregate_metrics", {}).get("steering_only", {})
        return (score_for_objective(metrics, objective), -abs(float(row["strength"])))

    if not results:
        return {}
    best = max(results, key=key)
    return {
        "layer": best["layer"],
        "feature_id": best["feature_id"],
        "strength": best["strength"],
        "reason": f"best {objective} score with deterministic tie-break toward smaller absolute strength",
    }


def run_strength_sweep(
    recipe: FeatureRecipe,
    prompts: list[dict[str, Any]],
    backend: GenerationBackend,
    *,
    strengths: list[float] | None = None,
    prompt_only_instruction: str = "",
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 0,
    objective: str = "maximize_rule_score",
    prompt_set_id: str = "inline",
) -> dict[str, Any]:
    base = recipe.interventions[0]
    results = []
    for strength in strength_grid(strengths):
        candidate = FeatureRecipe.from_dict(recipe.to_dict())
        candidate.interventions = [
            Intervention(
                layer=base.layer,
                feature_id=base.feature_id,
                strength=float(strength),
                sign="zero" if strength == 0 else ("positive" if strength > 0 else "negative"),
                injection_mode=base.injection_mode,
                position_policy=base.position_policy,
            )
        ]
        bench = run_benchmark(
            candidate,
            prompts,
            backend,
            prompt_only_instruction=prompt_only_instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            objective=objective,
            prompt_set_id=prompt_set_id,
        )
        results.append(
            {
                "layer": base.layer,
                "feature_id": base.feature_id,
                "strength": float(strength),
                "aggregate_metrics": bench["aggregate_metrics"],
                "pass_fail": bench["pass_fail"],
                "validation_decision": bench["validation_decision"],
            }
        )
    return {
        "recipe_id": recipe.recipe_id,
        "sweep_id": f"sweep_{utc_now_iso().replace('-', '').replace(':', '').replace('Z', '').replace('T', '_')}",
        "strengths": strength_grid(strengths),
        "results": results,
        "best_setting": select_best_setting(results, objective),
    }


def save_sweep_result(result: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)
