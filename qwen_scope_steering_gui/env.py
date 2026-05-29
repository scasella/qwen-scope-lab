from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN")


def load_environment(env_path: str | Path | None = ".env") -> None:
    if env_path is None:
        return
    path = Path(env_path)
    if path.exists():
        load_dotenv(path, override=False)


def get_hf_token() -> str | None:
    for name in HF_TOKEN_NAMES:
        value = os.environ.get(name)
        if value:
            return value
    return None


def has_model_api_key() -> bool:
    return any(os.environ.get(name) for name in ("MODEL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"))
