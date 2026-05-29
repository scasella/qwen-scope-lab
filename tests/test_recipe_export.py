from qwen_scope_steering_gui.recipe_schema import FeatureRecipe, Intervention, ModelMetadata, TargetBehavior


def test_markdown_export_contains_required_review_fields():
    recipe = FeatureRecipe.create(
        target_behavior=TargetBehavior(name="json_validity", description="Return valid JSON."),
        model=ModelMetadata(model_id="model", sae_id="sae"),
        interventions=[Intervention(layer=1, feature_id=7, strength=4.0)],
    )
    recipe.benchmark["summary"] = "Benchmark summary"
    markdown = recipe.to_markdown()

    assert "model" in markdown
    assert "sae" in markdown
    assert "feature 7" in markdown
    assert "strength 4.0" in markdown
    assert "Benchmark summary" in markdown
    assert "Status" in markdown
