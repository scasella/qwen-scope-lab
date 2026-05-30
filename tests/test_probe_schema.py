import pytest

from qwen_scope_steering_gui.probe_schema import LinearProbe
from qwen_scope_steering_gui.recipe_schema import ModelMetadata, RecipeValidationError, TargetBehavior


def _behavior():
    return TargetBehavior(name="sycophancy", description="Detect sycophancy via a residual probe.")


def _model():
    return ModelMetadata(model_id="Qwen/Qwen3.5-2B", sae_id="dev/sae", dtype="float32", config_name="x")


def test_create_compute_id_and_roundtrip():
    p = LinearProbe.create(_behavior(), _model(), layer=12, direction=[0.1, 0.2, 0.3], bias=0.5,
                           threshold=1.0, method="diffmeans", on_policy=True)
    assert p.probe_id == "sycophancy_probe_onpolicy_l12_v1"
    back = LinearProbe.from_json(p.to_json())
    assert back.direction == [0.1, 0.2, 0.3] and back.bias == 0.5 and back.on_policy is True


def test_validated_requires_validated_decision():
    with pytest.raises(RecipeValidationError):
        LinearProbe.create(_behavior(), _model(), layer=12, direction=[0.1], status="validated",
                           evaluation={"validation_decision": {"status": "benchmarked"}})


def test_empty_direction_rejected():
    with pytest.raises(RecipeValidationError):
        LinearProbe.create(_behavior(), _model(), layer=12, direction=[])


def test_markdown_renders():
    p = LinearProbe.create(_behavior(), _model(), layer=12, direction=[0.1, 0.2],
                           evaluation={"auc": 1.0, "validation_decision": {"status": "benchmarked", "reason": "r"}})
    md = p.to_markdown()
    assert "Probe: sycophancy_probe" in md and "AUC" in md
