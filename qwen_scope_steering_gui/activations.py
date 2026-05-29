from __future__ import annotations

from typing import Any

import torch

from .config import SteeringConfig
from .hooks import register_capture_hook
from .model_loader import ModelBundle
from .sae_loader import SAELayer
from .sae_math import topk_features


def extract_prompt_features(
    bundle: ModelBundle,
    sae: SAELayer,
    config: SteeringConfig,
    prompt: str,
    layer: int,
    top_k: int | None = None,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    tokenizer = bundle.tokenizer
    model = bundle.model
    encoded = tokenizer(prompt, return_tensors="pt", truncation=bool(max_seq_len), max_length=max_seq_len)
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.device)

    capture: dict[str, torch.Tensor] = {}
    handle = register_capture_hook(model, layer, capture, to_cpu=False)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    residual = capture.get("residual")
    if residual is None:
        raise RuntimeError(f"capture hook did not fire for layer {layer}")
    residual = residual[0].float()
    w_enc = sae.W_enc.to(device=residual.device, dtype=torch.float32)
    b_enc = sae.b_enc.to(device=residual.device, dtype=torch.float32)
    vals, idx = topk_features(residual, w_enc, b_enc, top_k or config.top_k)
    vals = vals.detach().cpu()
    idx = idx.detach().cpu()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().tolist())

    rows = []
    for token_index, token_text in enumerate(tokens):
        features = [
            {"feature_id": int(feature_id), "activation": float(value)}
            for value, feature_id in zip(vals[token_index].tolist(), idx[token_index].tolist(), strict=True)
        ]
        rows.append({"token_index": token_index, "token_text": token_text, "features": features})

    return {
        "prompt": prompt,
        "layer": layer,
        "tokens": tokens,
        "top_features_by_token": rows,
        "metadata": {
            "model_id": config.model_id,
            "sae_id": config.sae_id,
            "top_k": top_k or config.top_k,
            "d_model": config.d_model,
            "d_sae": config.d_sae,
        },
    }
