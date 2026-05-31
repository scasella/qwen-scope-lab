from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

from .config import SteeringConfig
from .env import get_hf_token


REQUIRED_KEYS = ("W_enc", "W_dec", "b_enc", "b_dec")


@dataclass
class SAELayer:
    layer: int
    path: Path
    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor

    def to_device(self, device: torch.device | str, dtype: torch.dtype | None = None) -> "SAELayer":
        kwargs = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return SAELayer(
            layer=self.layer,
            path=self.path,
            W_enc=self.W_enc.to(**kwargs),
            W_dec=self.W_dec.to(**kwargs),
            b_enc=self.b_enc.to(**kwargs),
            b_dec=self.b_dec.to(**kwargs),
        )


def layer_filename(layer: int) -> str:
    if layer < 0:
        raise ValueError("layer must be non-negative")
    return f"layer{layer}.sae.pt"


def validate_sae_state(state: dict[str, Any], config: SteeringConfig) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(f"SAE checkpoint missing keys: {missing}")

    expected = {
        "W_enc": (config.d_sae, config.d_model),
        "W_dec": (config.d_model, config.d_sae),
        "b_enc": (config.d_sae,),
        "b_dec": (config.d_model,),
    }
    for key, shape in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{key} must be a torch.Tensor")
        if tuple(value.shape) != shape:
            raise ValueError(f"{key} shape {tuple(value.shape)} does not match expected {shape}")


def load_sae_file(path: str | Path, layer: int, config: SteeringConfig) -> SAELayer:
    state = torch.load(Path(path), map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"SAE checkpoint {path} must be a dict")
    validate_sae_state(state, config)
    return SAELayer(
        layer=layer,
        path=Path(path),
        W_enc=state["W_enc"].detach().cpu(),
        W_dec=state["W_dec"].detach().cpu(),
        b_enc=state["b_enc"].detach().cpu(),
        b_dec=state["b_dec"].detach().cpu(),
    )


class LazySAELoader:
    def __init__(self, config: SteeringConfig):
        self.config = config
        self._cache: OrderedDict[int, SAELayer] = OrderedDict()

    @property
    def cached_layers(self) -> list[int]:
        return list(self._cache)

    def resolve_layer_path(self, layer: int) -> Path:
        if not 0 <= layer < self.config.num_layers:
            raise ValueError(f"layer {layer} is outside [0, {self.config.num_layers})")
        filename = layer_filename(layer)
        token = get_hf_token()
        try:
            path = hf_hub_download(
                repo_id=self.config.sae_id,
                filename=filename,
                cache_dir=self.config.hf_cache_dir,
                token=token,
                local_files_only=self.config.local_files_only,
            )
        except GatedRepoError as exc:
            raise RuntimeError(
                f"Hugging Face access is gated for {self.config.sae_id}; set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in .env with access."
            ) from exc
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Could not access SAE repo {self.config.sae_id}; check the repo id and Hugging Face token permissions."
            ) from exc
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"SAE layer {filename} is not present in local cache and downloads are disabled for {self.config.sae_id}."
            ) from exc
        except HfHubHTTPError as exc:
            hint = " Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in .env if this repo requires authentication." if not token else ""
            raise RuntimeError(f"Failed to download {filename} from {self.config.sae_id}.{hint}") from exc
        return Path(path)

    def load_layer(self, layer: int) -> SAELayer:
        if layer in self._cache:
            self._cache.move_to_end(layer)
            return self._cache[layer]
        path = self.resolve_layer_path(layer)
        loaded = load_sae_file(path, layer, self.config)
        self._cache[layer] = loaded
        while len(self._cache) > self.config.sae_cache_max_layers:
            self._cache.popitem(last=False)
        return loaded
