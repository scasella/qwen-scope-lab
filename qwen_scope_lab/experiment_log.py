"""Append-only research trail for the bench.

Distinct from the recipe store: a *recipe* is a validated artifact; an *experiment* is a
record of something tried and its outcome — including negatives. Every job (and the sync
benchmark/autopilot runs) appends one compact line here, so an agent (or human) can review
the full research history. Mirrors the tiny-store style of ``recipe_store.py``.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class ExperimentLog:
    def __init__(self, root: str | Path = "experiments"):
        self.root = Path(root)
        self.path = self.root / "log.jsonl"

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        self.root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return rec

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def read(self) -> list[dict[str, Any]]:
        return self.tail(10**9)
