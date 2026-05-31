from pathlib import Path

import pytest
import torch

from qwen_scope_lab.config import load_config
from qwen_scope_lab.sae_loader import LazySAELoader, layer_filename, load_sae_file, validate_sae_state


def fake_state(d_model=4, d_sae=6):
    return {
        "W_enc": torch.zeros(d_sae, d_model),
        "W_dec": torch.zeros(d_model, d_sae),
        "b_enc": torch.zeros(d_sae),
        "b_dec": torch.zeros(d_model),
    }


def test_layer_filename_resolution():
    assert layer_filename(12) == "layer12.sae.pt"
    with pytest.raises(ValueError):
        layer_filename(-1)


def test_shape_validation_catches_mismatch():
    cfg = load_config("configs/fake_test.yaml")
    state = fake_state()
    validate_sae_state(state, cfg)
    state["W_dec"] = torch.zeros(5, 6)
    with pytest.raises(ValueError, match="W_dec shape"):
        validate_sae_state(state, cfg)


def test_lru_cache_evicts_layers(monkeypatch, tmp_path):
    cfg = load_config("configs/fake_test.yaml")
    paths = {}
    for layer in (0, 1):
        path = tmp_path / f"layer{layer}.sae.pt"
        torch.save(fake_state(), path)
        paths[layer] = path

    loader = LazySAELoader(cfg)
    monkeypatch.setattr(loader, "resolve_layer_path", lambda layer: Path(paths[layer]))
    first = loader.load_layer(0)
    second = loader.load_layer(1)
    assert first.layer == 0
    assert second.layer == 1
    assert loader.cached_layers == [1]


def test_download_error_mentions_token_when_missing(monkeypatch):
    from unittest.mock import MagicMock

    from huggingface_hub.utils import HfHubHTTPError

    cfg = load_config("configs/fake_test.yaml")
    loader = LazySAELoader(cfg)

    def fail_download(**_kwargs):
        # Simulate a 401 from the Hub. huggingface_hub changed its HTTP backend
        # (requests -> httpx) across versions, so build the error backend-agnostically.
        raise HfHubHTTPError("401 Client Error", response=MagicMock(status_code=401))

    monkeypatch.setattr("qwen_scope_lab.sae_loader.hf_hub_download", fail_download)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        loader.resolve_layer_path(0)
