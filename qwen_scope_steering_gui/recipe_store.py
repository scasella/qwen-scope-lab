from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .recipe_schema import FeatureRecipe


class RecipeStore:
    def __init__(self, root: str | Path = "recipes"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _recipe_dir(self, recipe_id: str) -> Path:
        if not recipe_id or "/" in recipe_id or "\\" in recipe_id or ".." in recipe_id:
            raise ValueError("invalid recipe_id")
        path = (self.root / recipe_id).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("recipe path escapes store root")
        return path

    def save(
        self,
        recipe: FeatureRecipe,
        *,
        benchmark_results: dict[str, Any] | None = None,
        examples: list[dict[str, Any]] | None = None,
    ) -> FeatureRecipe:
        recipe.validate()
        directory = self._recipe_dir(recipe.recipe_id)
        directory.mkdir(parents=True, exist_ok=True)
        recipe.artifacts["json_path"] = str(directory / "recipe.json")
        recipe.artifacts["markdown_path"] = str(directory / "recipe.md")
        if benchmark_results is not None:
            recipe.artifacts["results_path"] = str(directory / "benchmark_results.json")
        (directory / "recipe.json").write_text(recipe.to_json(), encoding="utf-8")
        (directory / "recipe.md").write_text(recipe.to_markdown(), encoding="utf-8")
        if benchmark_results is not None:
            (directory / "benchmark_results.json").write_text(json.dumps(benchmark_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if examples is not None:
            with (directory / "examples.jsonl").open("w", encoding="utf-8") as f:
                for example in examples:
                    f.write(json.dumps(example, sort_keys=True) + "\n")
        return recipe

    def load(self, recipe_id: str) -> FeatureRecipe:
        path = self._recipe_dir(recipe_id) / "recipe.json"
        return FeatureRecipe.from_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*/recipe.json")):
            try:
                recipe = FeatureRecipe.from_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            m = recipe.manifold
            rows.append(
                {
                    "recipe_id": recipe.recipe_id,
                    "status": recipe.status,
                    "kind": recipe.kind,
                    "target_behavior": recipe.target_behavior.name,
                    "model_id": recipe.model.model_id,
                    "layer": (m.layer if m else (recipe.interventions[0].layer if recipe.interventions else None)),
                    "feature_id": recipe.interventions[0].feature_id if recipe.interventions else None,
                    "concept": m.concept if m else None,
                    "source": m.source if m else None,
                    "target": m.target if m else None,
                    "manifold_path": m.path if m else None,
                    "path": str(path),
                }
            )
        return rows

    def search(self, query: str = "", status: str | None = None) -> list[dict[str, Any]]:
        needle = query.lower().strip()
        results = []
        for row in self.list():
            if status and status != "all" and row["status"] != status:
                continue
            recipe = self.load(row["recipe_id"])
            haystack = " ".join(
                [
                    recipe.recipe_id,
                    recipe.status,
                    recipe.target_behavior.name,
                    recipe.target_behavior.description,
                    str(recipe.interventions[0].feature_id if recipe.interventions else ""),
                    str(recipe.interventions[0].layer if recipe.interventions else ""),
                    " ".join([recipe.manifold.concept, recipe.manifold.source, recipe.manifold.target] if recipe.manifold else []),
                    " ".join(recipe.limitations),
                    recipe.benchmark.get("summary", ""),
                ]
            ).lower()
            if not needle or needle in haystack:
                results.append(row)
        return results

    def export_markdown(self, recipe_id: str) -> str:
        recipe = self.load(recipe_id)
        path = self._recipe_dir(recipe_id) / "recipe.md"
        path.write_text(recipe.to_markdown(), encoding="utf-8")
        return str(path)
