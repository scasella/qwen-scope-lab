from qwen_scope_lab.benchmark import EchoGenerationBackend
from qwen_scope_lab.recipe_schema import FeatureRecipe, Intervention, ModelMetadata, TargetBehavior
from qwen_scope_lab.sweep import run_strength_sweep, select_best_setting, strength_grid


def make_recipe():
    return FeatureRecipe.create(
        target_behavior=TargetBehavior(name="concise", description="Be concise."),
        model=ModelMetadata(model_id="model", sae_id="sae"),
        interventions=[Intervention(layer=1, feature_id=2, strength=4.0)],
    )


def test_strength_grid_includes_zero():
    assert 0.0 in strength_grid([2, 4])


def test_sweep_serializes_and_selects_best():
    result = run_strength_sweep(make_recipe(), [{"id": "p1", "prompt": "Answer concisely."}], EchoGenerationBackend(), strengths=[0, 4])

    assert result["results"]
    assert result["best_setting"]["strength"] in {0.0, 4.0}


def test_best_setting_tie_breaks_deterministically():
    best = select_best_setting(
        [
            {"layer": 1, "feature_id": 2, "strength": 8.0, "aggregate_metrics": {"steering_only": {"json_validity": 1.0}}},
            {"layer": 1, "feature_id": 2, "strength": 4.0, "aggregate_metrics": {"steering_only": {"json_validity": 1.0}}},
        ],
        "json_validity",
    )

    assert best["strength"] == 4.0
