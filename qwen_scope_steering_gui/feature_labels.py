from __future__ import annotations

from typing import Any

from .env import has_model_api_key


def label_feature(payload: dict[str, Any]) -> dict[str, Any]:
    if not has_model_api_key():
        return {"enabled": False, "label": None, "speculative": True, "note": "No model API key is configured."}
    tokens = payload.get("top_activating_tokens") or []
    label = " / ".join(str(token) for token in tokens[:3]) or f"feature {payload.get('feature_id')}"
    return {
        "enabled": True,
        "label": f"Speculative: {label}",
        "speculative": True,
        "note": "Tentative label from provided examples/tokens. Core steering does not depend on hosted APIs.",
    }
