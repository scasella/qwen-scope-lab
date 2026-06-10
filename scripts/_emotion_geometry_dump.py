"""Dump the fitted 3D geometry of the emotion manifolds for the visual write-up.

For each emotion concept (at the layer the experiment selected) this saves the real
fitted coordinates the GUI's 3D view uses: per-value centroids (`points_3d`), the dense
spline (`curve_3d`), and the actual 6-waypoint walk paths (`path_3d`) for both the
manifold spline and the linear chord. No new claims — pure geometry extraction for
`docs/writeups/emotion-manifold-centroid-routing.html`.

Run:  python scripts/_emotion_geometry_dump.py
Out:  reports/emotion_manifold/geometry_3d.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwen_scope_lab.mlx_backend import build_mlx_service  # noqa: E402

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"
# (concept, layer chosen by the experiment's sweep)
SPECS = [
    ("emotion_arousal", 12),
    ("emotion_valence_intensity", 16),
    ("emotion_fear", 16),
]
N_WAYPOINTS = 6


def main() -> None:
    t0 = time.time()
    print(f"[geo] loading MLX model {DEFAULT_MLX_MODEL} (no SAE) ...", flush=True)
    service = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)

    out: dict = {"model_id": DEFAULT_MLX_MODEL, "n_waypoints": N_WAYPOINTS, "concepts": {}}
    for name, layer in SPECS:
        print(f"[geo] {name} @ L{layer}", flush=True)
        fit = service.manifold_fit(name, layer)
        items = fit["items"]
        src, tgt = items[0], items[-1]
        paths = {}
        for path in ("manifold", "linear"):
            steer = service.manifold_steer(name, tgt, layer=layer, source=src,
                                           n_waypoints=N_WAYPOINTS, max_new_tokens=1,
                                           path=path, compute_unsteered=False,
                                           compute_energy=False)
            paths[path] = steer["path_3d"]
        out["concepts"][name] = {
            "label": fit["label"], "layer": layer, "items": items,
            "points_3d": fit["points_3d"], "curve_3d": fit["curve_3d"],
            "path_3d": paths, "quality": fit["quality"],
        }

    dest = ROOT / "reports" / "emotion_manifold" / "geometry_3d.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[geo] wrote {dest} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
