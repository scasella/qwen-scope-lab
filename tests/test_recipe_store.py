import pytest

from qwen_scope_lab_bench.recipe_schema import FeatureRecipe, Intervention, ModelMetadata, TargetBehavior
from qwen_scope_lab_bench.recipe_store import RecipeStore


def make_recipe():
    return FeatureRecipe.create(
        target_behavior=TargetBehavior(name="json_validity", description="Return valid JSON."),
        model=ModelMetadata(model_id="model", sae_id="sae"),
        interventions=[Intervention(layer=1, feature_id=7, strength=4.0)],
    )


def test_store_save_load_list_search_and_export(tmp_path):
    store = RecipeStore(tmp_path)
    recipe = store.save(make_recipe())

    loaded = store.load(recipe.recipe_id)
    assert loaded.recipe_id == recipe.recipe_id
    assert store.list()[0]["recipe_id"] == recipe.recipe_id
    assert store.search("json")[0]["recipe_id"] == recipe.recipe_id
    markdown = store.export_markdown(recipe.recipe_id)
    assert markdown.endswith("recipe.md")


def test_store_rejects_path_traversal(tmp_path):
    store = RecipeStore(tmp_path)

    with pytest.raises(ValueError):
        store.load("../escape")
