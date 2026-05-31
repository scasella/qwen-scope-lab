import logging

from qwen_scope_lab_bench.safety import redact_mapping, redact_text


def test_secret_values_redacted(monkeypatch, caplog):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret_123456")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-123456")
    with caplog.at_level(logging.INFO):
        logging.getLogger("test").info("token=%s", redact_text("hf_test_secret_123456"))
        logging.getLogger("test").info("mapping=%s", redact_mapping({"api_key": "sk-test-secret-123456"}))
    assert "hf_test_secret_123456" not in caplog.text
    assert "sk-test-secret-123456" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_non_secret_hf_cache_name_is_not_redacted():
    assert redact_text("hf_cache_dir") == "hf_cache_dir"
