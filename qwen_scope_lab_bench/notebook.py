from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_NOTEBOOK = {"features": []}


def load_notebook(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return dict(DEFAULT_NOTEBOOK)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(f"{path} is not a feature notebook")
    return data


def save_notebook_entry(path: str | Path, entry: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    data = load_notebook(path)
    data["features"].append(entry)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    return data
