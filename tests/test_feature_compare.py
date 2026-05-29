from qwen_scope_steering_gui.feature_compare import contrast_features, token_activation_summary


def inspection(prompt, rows):
    return {"prompt": prompt, "top_features_by_token": rows}


def test_contrast_features_includes_token_summaries():
    positive = inspection(
        "pos",
        [
            {
                "token_index": 0,
                "token_text": "A",
                "features": [{"feature_id": 1, "activation": 2.0}, {"feature_id": 2, "activation": 3.0}],
            }
        ],
    )
    negative = inspection(
        "neg",
        [
            {
                "token_index": 0,
                "token_text": "B",
                "features": [{"feature_id": 1, "activation": 0.5}, {"feature_id": 3, "activation": 4.0}],
            }
        ],
    )

    result = contrast_features(positive, negative, limit=2)
    assert result["positive_token_summary"][0]["top_feature_ids"] == [1, 2]
    assert result["negative_token_summary"][0]["max_activation"] == 4.0
    assert result["positive_stronger"][0]["feature_id"] == 2
    assert result["negative_stronger"][0]["feature_id"] == 3


def test_token_activation_summary_handles_empty_features():
    result = token_activation_summary(inspection("x", [{"token_index": 0, "token_text": "x", "features": []}]))
    assert result[0]["max_activation"] == 0.0
