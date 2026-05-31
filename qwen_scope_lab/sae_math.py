from __future__ import annotations

import torch


def compute_pre_activations(residual: torch.Tensor, w_enc: torch.Tensor, b_enc: torch.Tensor) -> torch.Tensor:
    if residual.shape[-1] != w_enc.shape[1]:
        raise ValueError(f"residual hidden size {residual.shape[-1]} does not match W_enc second dim {w_enc.shape[1]}")
    if b_enc.shape != (w_enc.shape[0],):
        raise ValueError(f"b_enc shape {tuple(b_enc.shape)} does not match W_enc first dim {w_enc.shape[0]}")
    return residual @ w_enc.transpose(-1, -2) + b_enc


def topk_features(residual: torch.Tensor, w_enc: torch.Tensor, b_enc: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pre_acts = compute_pre_activations(residual, w_enc, b_enc)
    k = min(top_k, pre_acts.shape[-1])
    return pre_acts.topk(k, dim=-1)


def steering_delta_norm(hidden: torch.Tensor, steering_vector: torch.Tensor, strength: float) -> float:
    expanded = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
    delta = strength * expanded
    shape_factor = 1
    for size in hidden.shape[:-1]:
        shape_factor *= size
    return float(delta.float().norm().item() * (shape_factor ** 0.5))
