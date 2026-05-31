from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class HookTrace:
    fired_count: int = 0
    hidden_delta_norm: float = 0.0

    @property
    def hook_fired(self) -> bool:
        return self.fired_count > 0


def transformer_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("model", "model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
    ]
    for path in candidates:
        target = model
        for attr in path:
            target = getattr(target, attr, None)
            if target is None:
                break
        if target is not None:
            return target
    raise ValueError("could not locate transformer layers on model")


def layer_module(model: Any, layer: int) -> torch.nn.Module:
    layers = transformer_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer {layer} is outside available layer count {len(layers)}")
    return layers[layer]


def _hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def _replace_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def register_capture_hook(model: Any, layer: int, capture: dict[str, torch.Tensor], *, to_cpu: bool = True) -> torch.utils.hooks.RemovableHandle:
    def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        residual = _hidden_from_output(output).detach()
        capture["residual"] = residual.cpu() if to_cpu else residual

    return layer_module(model, layer).register_forward_hook(_hook)


def apply_steering_to_hidden(
    hidden: torch.Tensor,
    steering_vector: torch.Tensor,
    strength: float,
    mode: str = "all_positions",
) -> tuple[torch.Tensor, float]:
    if mode != "all_positions":
        raise ValueError(f"unsupported steering mode: {mode}")
    vector = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
    delta = strength * vector
    steered = hidden + delta.view(*([1] * (hidden.ndim - 1)), hidden.shape[-1])
    delta_norm = float((steered - hidden).float().norm().item())
    return steered, delta_norm


def register_steering_hook(
    model: Any,
    layer: int,
    steering_vector: torch.Tensor,
    strength: float,
    mode: str,
    trace: HookTrace,
) -> Any:
    if getattr(model, "is_mlx_runtime", False):  # local Apple-Silicon (MLX) backend: swap the block
        return model.install_steer(layer, steering_vector, strength, trace)

    def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = _hidden_from_output(output)
        steered, delta_norm = apply_steering_to_hidden(hidden, steering_vector, strength, mode)
        trace.fired_count += 1
        trace.hidden_delta_norm += delta_norm
        return _replace_hidden(output, steered)

    return layer_module(model, layer).register_forward_hook(_hook)


def register_replace_hook(
    model: Any,
    layer: int,
    replacement: torch.Tensor,
    position: int,
    trace: HookTrace,
) -> Any:
    """Manifold-steering intervention: overwrite the residual at a single token
    position with ``replacement`` (a point on the fitted manifold). Fires only when the
    position is within the current sequence — i.e. the prompt forward pass — so during
    cached generation (seq len 1) it is a no-op and the KV cache propagates the edit."""
    if getattr(model, "is_mlx_runtime", False):  # local Apple-Silicon (MLX) backend: position-replace swap
        return model.install_replace(layer, replacement, position, trace)

    def _hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = _hidden_from_output(output)
        if not 0 <= position < hidden.shape[1]:
            return output
        vector = replacement.to(device=hidden.device, dtype=hidden.dtype)
        new_hidden = hidden.clone()
        new_hidden[:, position, :] = vector
        trace.fired_count += 1
        trace.hidden_delta_norm += float((new_hidden - hidden).float().norm().item())
        return _replace_hidden(output, new_hidden)

    return layer_module(model, layer).register_forward_hook(_hook)
