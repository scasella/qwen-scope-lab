from qwen_scope_lab.autopilot import run_autopilot
from qwen_scope_lab.benchmark import EchoGenerationBackend, attach_benchmark_to_recipe, run_benchmark
from qwen_scope_lab.config import load_config
from qwen_scope_lab.recipe_schema import FeatureRecipe, Intervention, ModelMetadata, RecipeValidationError, TargetBehavior
from qwen_scope_lab.recipe_store import RecipeStore


def make_recipe():
    return FeatureRecipe.create(
        target_behavior=TargetBehavior(name="json_validity", description="Return valid JSON."),
        model=ModelMetadata(model_id="model", sae_id="sae"),
        interventions=[Intervention(layer=1, feature_id=7, strength=4.0)],
    )


def test_bench_calls_required_methods_and_refuses_weak_validation():
    result = run_benchmark(make_recipe(), [{"id": "p1", "prompt": "Return JSON."}], EchoGenerationBackend(), objective="maximize_json_validity")

    assert set(result["methods"]) >= {
        "unsteered_baseline",
        "prompt_only",
        "steering_only",
        "prompt_plus_steering",
        "zero_strength_control",
        "random_feature_control",
        "negative_strength_control",
    }
    assert result["validation_decision"]["status"] in {"benchmarked", "validated"}
    assert "zero_strength_delta" in result["aggregate_metrics"]["zero_strength_control"]
    assert "random_control_delta" in result["aggregate_metrics"]["random_feature_control"]
    assert "negative_strength_delta" in result["aggregate_metrics"]["negative_strength_control"]


def test_autopilot_produces_candidate_recipe_and_store_artifact(tmp_path):
    cfg = load_config("configs/fake_test.yaml")
    result = run_autopilot(
        config=cfg,
        config_path="configs/fake_test.yaml",
        target_name="json_validity",
        target_description="Return valid JSON.",
        positive_examples=['{"a":1}'],
        negative_examples=["not json"],
        validation_prompts=[{"id": "p1", "prompt": "Return JSON."}],
        candidate_layers=[1],
        candidate_count=2,
        objective="maximize_json_validity",
        output_dir=tmp_path / "json_recipe",
    )

    assert result["output_paths"]["recipe_json"].endswith("recipe.json")
    assert (tmp_path / "json_recipe" / "recipe.json").exists()
    assert result["best_recipe"]["status"] in {"candidate", "validated"}


def test_validation_gate_rejects_claim_without_evidence():
    recipe = make_recipe()
    recipe.status = "validated"

    try:
        recipe.validate()
    except RecipeValidationError:
        pass
    else:
        raise AssertionError("validated recipe without evidence should fail")
