from app import _flatten_inspection, _rows_to_frame


def test_flatten_inspection_returns_tabular_frames():
    result = {
        "top_features_by_token": [
            {
                "token_index": 0,
                "token_text": "The",
                "features": [
                    {"feature_id": 7, "activation": 1.25},
                    {"feature_id": 9, "activation": 0.5},
                ],
            }
        ]
    }

    token_rows, feature_rows, returned = _flatten_inspection(result)

    assert list(token_rows.columns) == ["token_index", "token_text", "top_features"]
    assert token_rows.to_dict("records") == [{"token_index": 0, "token_text": "The", "top_features": "7:1.250, 9:0.500"}]
    assert list(feature_rows.columns) == ["token_index", "token_text", "feature_id", "activation"]
    assert feature_rows.to_dict("records") == [
        {"token_index": 0, "token_text": "The", "feature_id": 7, "activation": 1.25},
        {"token_index": 0, "token_text": "The", "feature_id": 9, "activation": 0.5},
    ]
    assert returned is result


def test_rows_to_frame_preserves_columns_for_empty_results():
    frame = _rows_to_frame([], ["feature_id", "difference"])

    assert list(frame.columns) == ["feature_id", "difference"]
    assert frame.empty
