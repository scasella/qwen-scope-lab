from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from .config import SteeringConfig
from .env import get_hf_token
from .safety import redact_mapping

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    dtype: torch.dtype


def torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def select_device(config: SteeringConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests cuda but torch.cuda.is_available() is false")
    return torch.device(config.device)


def gpu_memory_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "total_gb": round(props.total_memory / 1024**3, 2),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 1024**3, 2),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 1024**3, 2),
    }


def parameter_device_summary(model: Any) -> dict[str, Any]:
    devices: dict[str, int] = {}
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        devices[str(param.device)] = devices.get(str(param.device), 0) + count
    return {
        "parameter_count": total,
        "parameter_count_b": round(total / 1_000_000_000, 3),
        "parameter_devices": {device: round(count / max(total, 1), 4) for device, count in devices.items()},
    }


def load_model(config: SteeringConfig) -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype(config.torch_dtype)
    device = select_device(config)
    token = get_hf_token()

    log.info("loading tokenizer/model: %s", redact_mapping({"model_id": config.model_id}))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        token=token,
        cache_dir=config.hf_cache_dir,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "cache_dir": config.hf_cache_dir,
        "token": token,
        "trust_remote_code": config.trust_remote_code,
        "low_cpu_mem_usage": config.low_cpu_mem_usage,
        "local_files_only": config.local_files_only,
    }
    if device.type == "cuda" and config.device_map:
        kwargs["device_map"] = config.device_map
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
    if "device_map" not in kwargs:
        model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    log.info(
        "model loaded: %s",
        redact_mapping(
            {
                "model_id": config.model_id,
                "dtype": str(dtype),
                "device": str(device),
                **parameter_device_summary(model),
                **gpu_memory_summary(),
            }
        ),
    )
    return ModelBundle(tokenizer=tokenizer, model=model, device=device, dtype=dtype)
