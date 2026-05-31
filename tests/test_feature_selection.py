import pytest

from qwen_scope_lab_bench.feature_selection import select_active_feature


def test_select_active_feature_picks_highest_activation():
    inspection = {
        "top_features_by_token": [
            {"token_index": 0, "token_text": "A", "features": [{"feature_id": 1, "activation": 0.5}]},
            {"token_index": 1, "token_text": "B", "features": [{"feature_id": 2, "activation": 3.0}]},
            {"token_index": 2, "token_text": "C", "features": [{"feature_id": 3, "activation": 2.0}]},
        ]
    }
    selected = select_active_feature(inspection)
    assert selected == {"feature_id": 2, "activation": 3.0, "token_index": 1, "token_text": "B"}


def test_select_active_feature_fails_on_empty_inspection():
    with pytest.raises(ValueError, match="no active features"):
        select_active_feature({"top_features_by_token": []})
