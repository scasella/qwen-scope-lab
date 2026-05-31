import pytest

pytest.importorskip("gradio")  # the legacy Gradio app is an opt-in extra (`.[gradio]`)

from app import build_demo


def _components(demo):
    return list(demo.blocks.values())


def test_gui_contains_workbench_tabs():
    demo = build_demo("configs/fake_test.yaml")

    tab_labels = {getattr(component, "label", None) for component in _components(demo)}

    assert {
        "Inspect prompt features",
        "Compare prompts",
        "Steer generation",
        "Bench",
        "Autopilot",
        "Recipe Library",
    }.issubset(tab_labels)


def test_validated_action_is_hidden_until_benchmark_passes():
    demo = build_demo("configs/fake_test.yaml")

    buttons = [
        component
        for component in _components(demo)
        if getattr(component, "value", None) == "Mark Recipe Validated"
    ]

    assert len(buttons) == 1
    assert buttons[0].visible is False
