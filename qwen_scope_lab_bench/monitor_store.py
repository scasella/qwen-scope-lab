"""Filesystem store for behavior monitors (mirrors ``recipe_store.RecipeStore``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .monitor_schema import BehaviorMonitor


class MonitorStore:
    def __init__(self, root: str | Path = "monitors"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _monitor_dir(self, monitor_id: str) -> Path:
        if not monitor_id or "/" in monitor_id or "\\" in monitor_id or ".." in monitor_id:
            raise ValueError("invalid monitor_id")
        path = (self.root / monitor_id).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("monitor path escapes store root")
        return path

    def save(self, monitor: BehaviorMonitor) -> BehaviorMonitor:
        monitor.validate()
        directory = self._monitor_dir(monitor.monitor_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "monitor.json").write_text(monitor.to_json(), encoding="utf-8")
        (directory / "monitor.md").write_text(monitor.to_markdown(), encoding="utf-8")
        return monitor

    def load(self, monitor_id: str) -> BehaviorMonitor:
        path = self._monitor_dir(monitor_id) / "monitor.json"
        return BehaviorMonitor.from_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*/monitor.json")):
            try:
                m = BehaviorMonitor.from_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            ev = m.evaluation or {}
            rows.append({"monitor_id": m.monitor_id, "status": m.status, "behavior": m.behavior.name,
                         "layer": m.layer, "n_features": len(m.features), "auc": ev.get("auc"),
                         "f1": ev.get("f1"), "model_id": m.model.model_id})
        return rows

    def search(self, query: str = "", status: str | None = None) -> list[dict[str, Any]]:
        needle = query.lower().strip()
        out = []
        for row in self.list():
            if status and status != "all" and row["status"] != status:
                continue
            hay = f"{row['monitor_id']} {row['behavior']}".lower()
            if not needle or needle in hay:
                out.append(row)
        return out
