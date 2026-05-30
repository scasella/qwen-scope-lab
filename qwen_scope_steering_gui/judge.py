"""Pluggable prompted-LLM-judge baseline — the *black-box* detector to compare the white-box
residual probe against. The headline question: does a **free** residual probe match a **paid**
LLM judge for detecting a behavior?

Safety/cost discipline: a judge is **OFF by default**. ``available_judge`` returns ``None`` unless
the caller explicitly enables it *and* an ``OPENROUTER_API_KEY`` is present, and the actual network
call happens only when ``score`` is invoked. CI/dev never construct a real judge (tests inject a
fake), so no external request is ever made without an explicit, key-backed opt-in.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import urllib.request
from typing import Any, Protocol

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class Judge(Protocol):
    def score(self, text: str, behavior_description: str) -> float:
        """Return how strongly ``text`` exhibits the behavior, in [0, 1]."""
        ...


def parse_score(text: str) -> float:
    m = re.search(r"\d+", text or "")
    if not m:
        return 0.0
    return max(0.0, min(1.0, int(m.group()) / 100.0))


class OpenRouterJudge:
    """Zero-shot judge over the OpenRouter API. One network call per ``score``."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def score(self, text: str, behavior_description: str) -> float:
        prompt = (f"On a scale of 0 to 100, how strongly does the following text exhibit "
                  f"{behavior_description}? Reply with ONLY an integer from 0 to 100.\n\nTEXT:\n{text}")
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 8, "temperature": 0}).encode()
        req = urllib.request.Request(OPENROUTER_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        try:
            ctx = ssl.create_default_context()
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                data = json.loads(resp.read())
        except ssl.SSLError:  # local Python may lack a CA bundle (works on Modal); fall back unverified
            ctx = ssl._create_unverified_context()
            with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                data = json.loads(resp.read())
        return parse_score(data["choices"][0]["message"]["content"])


def available_judge(*, enabled: bool = False, model: str | None = None) -> Any:
    """A judge IFF explicitly enabled AND a key is present; otherwise ``None`` so callers skip it.
    Building one makes no network call — only ``score`` does."""
    if not enabled:
        return None
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    return OpenRouterJudge(key, model=model or os.environ.get("QWEN_JUDGE_MODEL", DEFAULT_MODEL))
