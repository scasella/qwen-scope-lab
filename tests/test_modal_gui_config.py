import pytest

pytest.importorskip("modal")  # Modal is an opt-in extra (`.[modal]`)

import modal_app


def test_gui_scaledown_window_is_interactive_not_permanent():
    assert modal_app.GUI_SCALEDOWN_WINDOW_SECONDS == 300


def test_default_gui_target_is_cheap_2b_l4():
    profile = modal_app.select_gui_profile(None)
    assert modal_app.DEFAULT_GUI_TARGET == "2b-l4"
    assert profile.name == "2b-l4"
    assert profile.gpu == "L4"
    assert profile.config_path.endswith("qwen35_2b_dev_l0_100.yaml")


def test_gui_target_aliases_preserve_explicit_27b_choices():
    assert modal_app.select_gui_profile("27b").name == "27b-a100"
    assert modal_app.select_gui_profile("27b-a100").gpu == "A100-80GB"
    assert modal_app.select_gui_profile("h100").name == "27b-h100"


def test_invalid_gui_target_lists_supported_choices():
    try:
        modal_app.select_gui_profile("all")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("invalid GUI target should raise ValueError")
    assert "QWEN_GUI_TARGET" in message
    assert "2b-l4" in message
    assert "27b-a100" in message
    assert "27b-h100" in message
