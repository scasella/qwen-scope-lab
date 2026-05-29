from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any


SECRET_ENV_NAMES = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "MODEL_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
)

_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
]


def known_secret_values() -> list[str]:
    return [value for name in SECRET_ENV_NAMES if (value := os.environ.get(name))]


def redact_text(text: str) -> str:
    redacted = text
    for secret in known_secret_values():
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.upper() in SECRET_ENV_NAMES or any(token in key.upper() for token in ("TOKEN", "KEY", "SECRET")):
            out[key] = "[REDACTED]" if value else value
        elif isinstance(value, str):
            out[key] = redact_text(value)
        else:
            out[key] = value
    return out
