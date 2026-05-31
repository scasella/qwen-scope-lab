from qwen_scope_lab.feature_labels import label_feature


def test_label_feature_disabled_without_api_keys(monkeypatch):
    for name in ("MODEL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    result = label_feature({"layer": 1, "feature_id": 2, "top_activating_tokens": ["Paris"]})
    assert result["enabled"] is False
    assert result["speculative"] is True


def test_label_feature_returns_speculative_label_with_key(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    result = label_feature({"layer": 1, "feature_id": 2, "top_activating_tokens": ["Paris", "France"]})
    assert result["enabled"] is True
    assert result["label"] == "Speculative: Paris / France"
    assert result["speculative"] is True
