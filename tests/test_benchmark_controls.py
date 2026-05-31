from qwen_scope_lab_bench.benchmark_controls import build_method_specs, random_feature_id, stable_method_names
from qwen_scope_lab_bench.recipe_schema import Intervention


def test_control_methods_are_constructed():
    methods = build_method_specs(Intervention(layer=1, feature_id=5, strength=4.0), d_sae=100, seed=0)
    names = [method.name for method in methods]

    assert names == stable_method_names()
    assert next(method for method in methods if method.name == "zero_strength_control").intervention.strength == 0.0
    assert next(method for method in methods if method.name == "negative_strength_control").intervention.strength == -4.0


def test_random_feature_control_is_seeded_and_excludes_feature():
    first = random_feature_id(100, excluded_feature_id=5, seed=123)
    second = random_feature_id(100, excluded_feature_id=5, seed=123)

    assert first == second
    assert first != 5
