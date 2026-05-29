from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_steering_gui.benchmark import ServiceGenerationBackend, attach_benchmark_to_recipe, recipe_from_manual, run_benchmark, save_benchmark_result
from qwen_scope_steering_gui.config import load_config
from qwen_scope_steering_gui.prompt_sets import load_prompt_set
from qwen_scope_steering_gui.recipe_schema import FeatureRecipe
from qwen_scope_steering_gui.recipe_store import RecipeStore
from qwen_scope_steering_gui.service import SteeringService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe")
    parser.add_argument("--config")
    parser.add_argument("--target-behavior")
    parser.add_argument("--target-description", default="")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--feature-id", type=int)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--prompt-set", required=True)
    parser.add_argument("--prompt-only-instruction", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--objective", default="maximize_rule_score")
    parser.add_argument("--save-recipe", action="store_true")
    args = parser.parse_args()

    if args.recipe:
        recipe = FeatureRecipe.from_json(Path(args.recipe).read_text(encoding="utf-8"))
        config_path = args.config or recipe.model.config_name
        if not config_path:
            raise SystemExit("--config is required when recipe lacks model.config_name")
    else:
        required = (args.config, args.target_behavior, args.layer, args.feature_id, args.strength)
        if any(value is None for value in required):
            raise SystemExit("manual recipe requires --config --target-behavior --layer --feature-id --strength")
        config_path = args.config
        cfg = load_config(config_path)
        recipe = recipe_from_manual(
            config=cfg,
            config_path=config_path,
            target_behavior=args.target_behavior,
            target_description=args.target_description,
            layer=args.layer,
            feature_id=args.feature_id,
            strength=args.strength,
        )

    service = SteeringService.from_config_path(config_path)
    backend = ServiceGenerationBackend(service)
    prompts = load_prompt_set(args.prompt_set)
    result = run_benchmark(
        recipe,
        prompts,
        backend,
        prompt_only_instruction=args.prompt_only_instruction,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
        objective=args.objective,
        prompt_set_id=args.prompt_set,
    )
    output = args.output or f"reports/{recipe.recipe_id}_bench.json"
    save_benchmark_result(result, output)
    recipe = attach_benchmark_to_recipe(recipe, result)
    recipe.artifacts["results_path"] = output
    if args.save_recipe or args.recipe:
        RecipeStore().save(recipe, benchmark_results=result, examples=recipe.examples)
    print(json.dumps({"benchmark_path": output, "recipe_id": recipe.recipe_id, "validation_decision": result["validation_decision"]}, indent=2))


if __name__ == "__main__":
    main()
