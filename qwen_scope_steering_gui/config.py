from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


_VALID_DTYPES = {"float16", "bfloat16", "float32"}


@dataclass(frozen=True)
class SteeringConfig:
    model_id: str
    sae_id: str
    top_k: int
    num_layers: int
    d_model: int
    d_sae: int
    default_layer: int
    default_max_new_tokens: int
    torch_dtype: str
    device: str
    sae_cache_max_layers: int
    hf_cache_dir: str
    trust_remote_code: bool = True
    device_map: str | None = "auto"
    low_cpu_mem_usage: bool = True
    local_files_only: bool = False
    notebook_path: str = "feature_notebook.json"

    def validate(self) -> None:
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.sae_id:
            raise ValueError("sae_id is required")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0 <= self.default_layer < self.num_layers:
            raise ValueError("default_layer must be in [0, num_layers)")
        if self.d_model <= 0 or self.d_sae <= 0:
            raise ValueError("d_model and d_sae must be positive")
        if self.default_max_new_tokens <= 0:
            raise ValueError("default_max_new_tokens must be positive")
        if self.torch_dtype not in _VALID_DTYPES:
            raise ValueError(f"torch_dtype must be one of {sorted(_VALID_DTYPES)}")
        if self.sae_cache_max_layers <= 0:
            raise ValueError("sae_cache_max_layers must be positive")


def load_config(path: str | Path) -> SteeringConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    allowed = {field.name for field in fields(SteeringConfig)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{path} has unknown config fields: {sorted(unknown)}")

    missing = [field.name for field in fields(SteeringConfig) if field.default is MISSING and field.default_factory is MISSING and field.name not in raw]
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")

    cfg = SteeringConfig(**raw)
    cfg.validate()
    return cfg


def config_to_dict(config: SteeringConfig) -> dict[str, Any]:
    return {field.name: getattr(config, field.name) for field in fields(SteeringConfig)}
