from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.recipe_schema import FeatureRecipe
from qwen_scope_lab.recipe_store import RecipeStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-id")
    parser.add_argument("--recipe")
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.recipe:
        recipe = FeatureRecipe.from_json(Path(args.recipe).read_text(encoding="utf-8"))
        output = Path(args.output or Path(args.recipe).with_suffix(".md"))
        output.write_text(recipe.to_markdown(), encoding="utf-8")
        print(output)
    else:
        if not args.recipe_id:
            raise SystemExit("--recipe-id or --recipe is required")
        print(RecipeStore().export_markdown(args.recipe_id))


if __name__ == "__main__":
    main()
