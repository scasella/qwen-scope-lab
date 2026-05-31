from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark import EchoGenerationBackend, GenerationBackend, attach_benchmark_to_recipe, run_benchmark
from .benchmark_metrics import score_for_objective
from .candidate_search import CandidateFeature, fake_inspection, rank_candidates_from_inspections
from .prompt_sets import load_prompt_set, load_text_examples
from .recipe_schema import DiscoveryMetadata, FeatureRecipe, Intervention, ModelMetadata, TargetBehavior
from .recipe_store import RecipeStore
from .sweep import run_strength_sweep


def _inspect_examples(service: Any | None, examples: list[str], layer: int, positive: bool) -> list[dict[str, Any]]:
    if service is None:
        return [fake_inspection(example, layer, positive) for example in examples]
    return [service.inspect_prompt(example, layer=layer, top_k=10, max_seq_len=128) for example in examples]


def search_candidate_features(
    *,
    service: Any | None,
    positive_examples: list[str],
    negative_examples: list[str],
    candidate_layers: list[int],
    candidate_count: int,
) -> list[CandidateFeature]:
    all_candidates: list[CandidateFeature] = []
    for layer in candidate_layers:
        positive = _inspect_examples(service, positive_examples, layer, True)
        negative = _inspect_examples(service, negative_examples, layer, False)
        all_candidates.extend(
            rank_candidates_from_inspections(
                layer=layer,
                positive_inspections=positive,
                negative_inspections=negative,
                limit=candidate_count,
            )
        )
    return sorted(all_candidates, key=lambda item: (-item.combined_score, item.layer, item.feature_id))[:candidate_count]


def _recipe_for_candidate(
    *,
    candidate: CandidateFeature,
    target_name: str,
    target_description: str,
    config: Any,
    config_path: str,
    positive_examples: list[str],
    negative_examples: list[str],
    candidate_layers: list[int],
    candidate_count: int,
    strength: float = 4.0,
) -> FeatureRecipe:
    return FeatureRecipe.create(
        target_behavior=TargetBehavior(
            name=target_name,
            description=target_description,
            positive_description="; ".join(positive_examples[:3]),
            negative_description="; ".join(negative_examples[:3]),
        ),
        model=ModelMetadata(
            model_id=config.model_id,
            sae_id=config.sae_id,
            dtype=config.torch_dtype,
            config_name=config_path,
        ),
        interventions=[Intervention(layer=candidate.layer, feature_id=candidate.feature_id, strength=strength)],
        created_by="qwen-scope-autopilot",
        discovery=DiscoveryMetadata(
            method="contrastive_prompt_search",
            positive_prompts=positive_examples,
            negative_prompts=negative_examples,
            candidate_layers=candidate_layers,
            candidate_count=candidate_count,
            ranking_metric="combined = contrast * log(1 + frequency_positive / (epsilon + frequency_negative))",
        ),
    )


def select_best_candidate(results: list[dict[str, Any]], objective: str) -> dict[str, Any]:
    if not results:
        return {}

    def key(row: dict[str, Any]) -> tuple[float, float, int]:
        aggregate = row["benchmark"]["aggregate_metrics"]
        steering_score = max(
            score_for_objective(aggregate.get("steering_only", {}), objective),
            score_for_objective(aggregate.get("prompt_plus_steering", {}), objective),
        )
        prompt_only = score_for_objective(aggregate.get("prompt_only", {}), objective)
        return (steering_score, steering_score - prompt_only, -int(row["candidate"]["feature_id"]))

    return max(results, key=key)


def run_autopilot(
    *,
    config: Any,
    config_path: str,
    target_name: str,
    target_description: str,
    positive_examples: list[str],
    negative_examples: list[str],
    validation_prompts: list[dict[str, Any]],
    candidate_layers: list[int],
    candidate_count: int,
    objective: str,
    backend: GenerationBackend | None = None,
    service: Any | None = None,
    output_dir: str | Path | None = None,
    prompt_only_instruction: str = "",
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    backend = backend or EchoGenerationBackend(d_sae=int(config.d_sae))
    candidates = search_candidate_features(
        service=service,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        candidate_layers=candidate_layers,
        candidate_count=candidate_count,
    )
    candidate_results = []
    for candidate in candidates:
        recipe = _recipe_for_candidate(
            candidate=candidate,
            target_name=target_name,
            target_description=target_description,
            config=config,
            config_path=config_path,
            positive_examples=positive_examples,
            negative_examples=negative_examples,
            candidate_layers=candidate_layers,
            candidate_count=candidate_count,
            strength=4.0,
        )
        benchmark = run_benchmark(
            recipe,
            validation_prompts,
            backend,
            prompt_only_instruction=prompt_only_instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            objective=objective,
            prompt_set_id="autopilot_validation",
        )
        candidate_results.append({"candidate": candidate.to_dict(), "recipe": recipe, "benchmark": benchmark})
    best = select_best_candidate(candidate_results, objective)
    if not best:
        raise RuntimeError("autopilot found no candidates")
    best_recipe: FeatureRecipe = best["recipe"]
    sweep = run_strength_sweep(
        best_recipe,
        validation_prompts,
        backend,
        prompt_only_instruction=prompt_only_instruction,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        objective=objective,
        prompt_set_id="autopilot_validation",
    )
    best_recipe = attach_benchmark_to_recipe(best_recipe, best["benchmark"])
    if best_recipe.status != "validated":
        best_recipe.status = "candidate"
        best_recipe.benchmark["status"] = "candidate"
    output_paths: dict[str, str] = {}
    if output_dir is not None:
        recipe_dir = Path(output_dir)
        best_recipe.recipe_id = recipe_dir.name
        best_recipe.artifacts = {"json_path": "", "markdown_path": "", "results_path": ""}
        store = RecipeStore(recipe_dir.parent)
        store.save(best_recipe, benchmark_results=best["benchmark"], examples=best_recipe.examples)
        (recipe_dir / "candidate_features.json").write_text(
            json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (recipe_dir / "sweep_results.json").write_text(json.dumps(sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output_paths = {
            "recipe_json": str(recipe_dir / "recipe.json"),
            "recipe_markdown": str(recipe_dir / "recipe.md"),
            "benchmark_results": str(recipe_dir / "benchmark_results.json"),
            "candidate_features": str(recipe_dir / "candidate_features.json"),
            "sweep_results": str(recipe_dir / "sweep_results.json"),
        }
    prompt_only_score = score_for_objective(best["benchmark"]["aggregate_metrics"].get("prompt_only", {}), objective)
    steering_score = max(
        score_for_objective(best["benchmark"]["aggregate_metrics"].get("steering_only", {}), objective),
        score_for_objective(best["benchmark"]["aggregate_metrics"].get("prompt_plus_steering", {}), objective),
    )
    return {
        "candidate_features": [candidate.to_dict() for candidate in candidates],
        "candidate_benchmarks": [
            {
                "candidate": row["candidate"],
                "recipe_id": row["recipe"].recipe_id,
                "validation_decision": row["benchmark"]["validation_decision"],
                "aggregate_metrics": row["benchmark"]["aggregate_metrics"],
            }
            for row in candidate_results
        ],
        "best_candidate": best["candidate"],
        "best_recipe": best_recipe.to_dict(),
        "benchmark": best["benchmark"],
        "sweep": sweep,
        "output_paths": output_paths,
        "warning": "" if steering_score > prompt_only_score else "No candidate beat prompt-only on the configured objective.",
    }


def run_autopilot_from_files(
    *,
    config: Any,
    config_path: str,
    target_name: str,
    target_description: str,
    positive_examples_path: str | Path,
    negative_examples_path: str | Path,
    validation_prompts_path: str | Path,
    candidate_layers: list[int],
    candidate_count: int,
    objective: str,
    output_dir: str | Path,
    backend: GenerationBackend | None = None,
    service: Any | None = None,
    prompt_only_instruction: str = "",
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int = 0,
) -> dict[str, Any]:
    return run_autopilot(
        config=config,
        config_path=config_path,
        target_name=target_name,
        target_description=target_description,
        positive_examples=load_text_examples(positive_examples_path),
        negative_examples=load_text_examples(negative_examples_path),
        validation_prompts=load_prompt_set(validation_prompts_path),
        candidate_layers=candidate_layers,
        candidate_count=candidate_count,
        objective=objective,
        backend=backend,
        service=service,
        output_dir=output_dir,
        prompt_only_instruction=prompt_only_instruction,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
