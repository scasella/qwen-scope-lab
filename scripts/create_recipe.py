from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_steering_gui.benchmark import recipe_from_manual
from qwen_scope_steering_gui.config import load_config
from qwen_scope_steering_gui.recipe_store import RecipeStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-behavior", required=True)
    parser.add_argument("--target-description", default="")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--strength", type=float, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    recipe = recipe_from_manual(
        config=cfg,
        config_path=args.config,
        target_behavior=args.target_behavior,
        target_description=args.target_description,
        layer=args.layer,
        feature_id=args.feature_id,
        strength=args.strength,
    )
    saved = RecipeStore().save(recipe)
    print(json.dumps({"recipe_id": saved.recipe_id, "recipe_json": saved.artifacts["json_path"]}, indent=2))


if __name__ == "__main__":
    main()
