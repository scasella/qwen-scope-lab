import logging

from qwen_scope_lab_bench.benchmark import EchoGenerationBackend, run_benchmark
from qwen_scope_lab_bench.recipe_schema import FeatureRecipe, Intervention, ModelMetadata, TargetBehavior


def test_bench_does_not_log_secrets(monkeypatch, caplog):
    secret = "unit-test-openai-secret-value"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    caplog.set_level(logging.INFO)
    recipe = FeatureRecipe.create(
        target_behavior=TargetBehavior(name="json", description="JSON."),
        model=ModelMetadata(model_id="model", sae_id="sae"),
        interventions=[Intervention(layer=1, feature_id=7, strength=4.0)],
    )

    run_benchmark(recipe, [{"id": "p1", "prompt": "Return JSON."}], EchoGenerationBackend())

    assert secret not in caplog.text
