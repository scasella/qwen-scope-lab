from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_text_examples(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    return chunks or [line.strip() for line in text.splitlines() if line.strip()]


def load_prompt_set(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "prompt" not in item:
                raise ValueError(f"prompt set row {index} is missing prompt")
            rows.append(
                {
                    "id": item.get("id", f"p{index:03d}"),
                    "prompt": item["prompt"],
                    "metadata": item.get("metadata", {}),
                }
            )
    return rows


def parse_prompt_text(value: str) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(value.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            parsed = {"prompt": line}
        rows.append({"id": parsed.get("id", f"p{index:03d}"), "prompt": parsed["prompt"], "metadata": parsed.get("metadata", {})})
    return rows


def format_prompt_only(prompt: str, instruction: str) -> str:
    if not instruction:
        return prompt
    if "{prompt}" in instruction:
        return instruction.format(prompt=prompt)
    return f"{instruction.strip()}\n\nPrompt: {prompt}"
