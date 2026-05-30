"""Filesystem store for linear probes (mirrors ``monitor_store.MonitorStore``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .probe_schema import LinearProbe


class ProbeStore:
    def __init__(self, root: str | Path = "probes"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _probe_dir(self, probe_id: str) -> Path:
        if not probe_id or "/" in probe_id or "\\" in probe_id or ".." in probe_id:
            raise ValueError("invalid probe_id")
        path = (self.root / probe_id).resolve()
        root = self.root.resolve()
        if root not in path.parents and path != root:
            raise ValueError("probe path escapes store root")
        return path

    def save(self, probe: LinearProbe) -> LinearProbe:
        probe.validate()
        directory = self._probe_dir(probe.probe_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "probe.json").write_text(probe.to_json(), encoding="utf-8")
        (directory / "probe.md").write_text(probe.to_markdown(), encoding="utf-8")
        return probe

    def load(self, probe_id: str) -> LinearProbe:
        path = self._probe_dir(probe_id) / "probe.json"
        return LinearProbe.from_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, Any]]:
        rows = []
        for path in sorted(self.root.glob("*/probe.json")):
            try:
                p = LinearProbe.from_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            ev = p.evaluation or {}
            rows.append({"probe_id": p.probe_id, "status": p.status, "behavior": p.behavior.name,
                         "layer": p.layer, "method": p.method, "on_policy": p.on_policy,
                         "auc": ev.get("auc"), "f1": ev.get("f1"), "model_id": p.model.model_id})
        return rows

    def search(self, query: str = "", status: str | None = None) -> list[dict[str, Any]]:
        needle = query.lower().strip()
        out = []
        for row in self.list():
            if status and status != "all" and row["status"] != status:
                continue
            hay = f"{row['probe_id']} {row['behavior']} {row['method']}".lower()
            if not needle or needle in hay:
                out.append(row)
        return out
