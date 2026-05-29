from __future__ import annotations

from typing import Any

import torch

from .config import SteeringConfig
from .hooks import HookTrace, register_replace_hook, register_steering_hook
from .model_loader import ModelBundle
from .sae_loader import SAELayer


def _decode_new_text(tokenizer: Any, input_len: int, generated_ids: torch.Tensor) -> str:
    return tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)


def generate_text(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, torch.Tensor]:
    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.device)
    do_sample = temperature > 0
    generate_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        generated = bundle.model.generate(**generate_kwargs)
    return _decode_new_text(tokenizer, input_ids.shape[1], generated), input_ids


def sequence_perplexity(bundle: ModelBundle, prompt: str, continuation: str) -> float | None:
    """Perplexity of `continuation` given `prompt` (one scoring forward pass). Lower =
    more fluent/natural — our proxy for the paper's off-manifold "naturalness" energy."""
    if not continuation:
        return None
    tokenizer = bundle.tokenizer
    full_ids = tokenizer(prompt + continuation, return_tensors="pt")["input_ids"].to(bundle.device)
    p_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    if full_ids.shape[1] <= p_len:
        return None
    with torch.no_grad():
        logits = bundle.model(input_ids=full_ids).logits[0].float()  # [seq, vocab]
    vocab = logits.shape[-1]
    logprobs = torch.log_softmax(logits[:-1], dim=-1)               # predicts token at pos+1
    targets = full_ids[0, 1:].clamp(max=vocab - 1)                  # clamp guards toy-model OOV ids
    cont = logprobs[p_len - 1:].gather(1, targets[p_len - 1:].unsqueeze(1)).squeeze(1)
    if cont.numel() == 0:
        return None
    return float(torch.exp(-cont.mean()).item())


def logits_delta_norm(
    bundle: ModelBundle,
    prompt: str,
    layer: int,
    steering_vector: torch.Tensor,
    strength: float,
    mode: str,
) -> float | None:
    tokenizer = bundle.tokenizer
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(bundle.device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(bundle.device)

    with torch.no_grad():
        base = bundle.model(input_ids=input_ids, attention_mask=attention_mask).logits.detach()
    trace = HookTrace()
    handle = register_steering_hook(bundle.model, layer, steering_vector, strength, mode, trace)
    try:
        with torch.no_grad():
            steered = bundle.model(input_ids=input_ids, attention_mask=attention_mask).logits.detach()
    finally:
        handle.remove()
    return float((steered - base).float().norm().item())


def manifold_generate(
    bundle: ModelBundle,
    prompt: str,
    layer: int,
    replacement_vector: torch.Tensor,
    position: int,
    max_new_tokens: int,
    temperature: float,
    compute_unsteered: bool = True,
) -> dict[str, Any]:
    """Generate with the residual at ``position`` replaced by a manifold point."""
    unsteered_text = None
    if compute_unsteered:
        unsteered_text, _ = generate_text(bundle, prompt, max_new_tokens, temperature)
    trace = HookTrace()
    handle = register_replace_hook(bundle.model, layer, replacement_vector, position, trace)
    try:
        steered_text, _ = generate_text(bundle, prompt, max_new_tokens, temperature)
    finally:
        handle.remove()
    return {
        "unsteered_text": unsteered_text,
        "steered_text": steered_text,
        "hook_fired": trace.hook_fired,
        "hidden_delta_norm": trace.hidden_delta_norm,
    }


def steer_generation(
    bundle: ModelBundle,
    sae: SAELayer,
    config: SteeringConfig,
    prompt: str,
    layer: int,
    feature_id: int,
    strength: float,
    max_new_tokens: int,
    temperature: float,
    mode: str = "all_positions",
    compute_logits_delta: bool = True,
) -> dict[str, Any]:
    if not 0 <= feature_id < sae.W_dec.shape[1]:
        raise ValueError(f"feature_id must be in [0, {sae.W_dec.shape[1]})")
    unsteered_text, _ = generate_text(bundle, prompt, max_new_tokens, temperature)
    steering_vector = sae.W_dec[:, feature_id]
    trace = HookTrace()
    handle = register_steering_hook(bundle.model, layer, steering_vector, strength, mode, trace)
    try:
        steered_text, _ = generate_text(bundle, prompt, max_new_tokens, temperature)
    finally:
        handle.remove()

    logit_norm = None
    if compute_logits_delta:
        try:
            logit_norm = logits_delta_norm(bundle, prompt, layer, steering_vector, strength, mode)
        except Exception:
            logit_norm = None

    return {
        "prompt": prompt,
        "layer": layer,
        "feature_id": feature_id,
        "strength": strength,
        "mode": mode,
        "unsteered_text": unsteered_text,
        "steered_text": steered_text,
        "hook_fired": trace.hook_fired,
        "hidden_delta_norm": trace.hidden_delta_norm,
        "logits_delta_norm": logit_norm,
        "metadata": {
            "model_id": config.model_id,
            "sae_id": config.sae_id,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }
