from pathlib import Path

import pytest

from qwen_scope_lab_bench.config import load_config


def test_all_shipped_configs_parse():
    for path in Path("configs").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.d_model > 0
        assert cfg.d_sae > 0
        assert 0 <= cfg.default_layer < cfg.num_layers


def test_invalid_config_fails_clearly(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model_id: fake\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        load_config(path)
