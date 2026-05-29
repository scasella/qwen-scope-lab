import pytest

from qwen_scope_steering_gui.recipe_schema import FeatureRecipe, Intervention, ManifoldSpec, ModelMetadata, RecipeValidationError, TargetBehavior


def valid_recipe():
    return FeatureRecipe.create(
        target_behavior=TargetBehavior(name="concise_answers", description="Make answers concise."),
        model=ModelMetadata(model_id="model", sae_id="sae", dtype="float32", config_name="configs/fake_test.yaml"),
        interventions=[Intervention(layer=1, feature_id=2, strength=4.0)],
    )


def test_valid_recipe_round_trips_json():
    recipe = valid_recipe()
    loaded = FeatureRecipe.from_json(recipe.to_json())

    assert loaded.recipe_id == "concise_answers_l1_f2_v1"
    assert loaded.model.model_id == "model"


def test_missing_model_id_fails():
    recipe = valid_recipe()
    recipe.model.model_id = ""

    with pytest.raises(RecipeValidationError, match="model id"):
        recipe.validate()


def test_invalid_feature_id_fails_with_config_metadata():
    recipe = valid_recipe()
    recipe.interventions[0].feature_id = 999

    with pytest.raises(RecipeValidationError, match="feature id"):
        recipe.validate({"model_id": "model", "sae_id": "sae", "num_layers": 2, "d_sae": 10})


def test_validated_status_without_evidence_fails():
    recipe = valid_recipe()
    recipe.status = "validated"

    with pytest.raises(RecipeValidationError, match="validated benchmark"):
        recipe.validate()


def test_recipe_id_is_deterministic():
    assert valid_recipe().compute_id() == valid_recipe().compute_id()


def manifold_recipe():
    return FeatureRecipe.create_manifold(
        target_behavior=TargetBehavior(name="days_of_week: Monday→Thursday", description="Steer days Monday to Thursday."),
        model=ModelMetadata(model_id="model", sae_id="sae"),
        manifold=ManifoldSpec(concept="days_of_week", source="Monday", target="Thursday", layer=3, path="pullback", n_waypoints=5),
    )


def test_manifold_recipe_id_has_no_doubled_slug():
    recipe = manifold_recipe()
    assert recipe.recipe_id == "days_of_week_monday_to_thursday_l3_v1"
    assert recipe.kind == "manifold" and recipe.interventions == []


def test_manifold_recipe_round_trips_and_validates():
    loaded = FeatureRecipe.from_json(manifold_recipe().to_json())
    assert loaded.kind == "manifold"
    assert loaded.manifold.concept == "days_of_week" and loaded.manifold.layer == 3


def test_manifold_recipe_requires_spec_fields():
    recipe = manifold_recipe()
    recipe.manifold.target = ""
    with pytest.raises(RecipeValidationError, match="concept, source and target"):
        recipe.validate()
