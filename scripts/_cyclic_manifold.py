"""Cyclic manifold test — does the days-of-week RING do something a linear probe provably can't?

The recovery diagnostic showed manifold steering is dominated by a linear chord for ordinal concepts:
a straight line interpolates the behavior through the middle. But a straight chord between two points
on a RING cuts across the empty interior — so cyclic concepts are the one place a 1-D linear baseline
should break. This tests both halves of the claim on the clean days-of-week ring:

  READING  : can a single linear direction represent the cyclic order? (ring-adjacency of the best
             1-D projection vs the manifold's 2-D ring fit). A line cannot close a loop — expected to fail.
  ROUTING  : walk Monday -> Friday AROUND the ring (manifold arc) vs the straight chord across the
             interior, argmax agreement on the intermediate days, with shuffled + random controls.
             This is the cyclic routing test never run before.

Uses a position-readout prompt ("Today is {item}. So today is ___") so the readout reports the
injected day, not its successor (the registry's "Tomorrow is" prompt would offset by one).

Run:  python scripts/_cyclic_manifold.py
Out:  reports/manifold_census/cyclic.json  (+ geometry for the ring plot)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import qwen_scope_lab.concept_presets as cp  # noqa: E402
from qwen_scope_lab.concept_presets import Concept  # noqa: E402
from qwen_scope_lab.mlx_backend import build_mlx_service  # noqa: E402
from scripts._emotion_manifold import register, routing_analysis, shuffled_concept  # noqa: E402

MODEL = "mlx-community/Qwen3.5-2B-bf16"
LAYERS = [8, 12, 14, 16]
READOUT = "full_string"

DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAYS_RING = Concept("days_ring", "Days of the week (ring)", "cyclic", DAYS,
    ("Today is {item}", "The meeting is on {item}", "See you {item}", "It happened last {item}",
     "Every {item}", "By {item}"),
    "Today is {item}. So today is", best_layer=14)
SRC, TGT, NWP = 0, 4, 5  # Monday -> Friday around the ring (chord cuts the interior)


def ring_adjacency_kd(centroids_pca, k):
    """Fraction of points whose nearest neighbour (in the first k PCA dims) is a cyclic neighbour.
    k=1 is the single-linear-direction baseline (cannot close a ring); k=2 is the plane the ring lives in."""
    n = len(centroids_pca)
    xy = np.asarray(centroids_pca)[:, :k]
    xy = xy - xy.mean(0)
    dd = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    np.fill_diagonal(dd, np.inf)
    nn = dd.argmin(1)
    return float(np.mean([(nn[i] == (i - 1) % n) or (nn[i] == (i + 1) % n) for i in range(n)]))


def main():
    t0 = time.time()
    print(f"[cyclic] loading {MODEL} ...", flush=True)
    service = build_mlx_service(MODEL, default_layer=12)
    register(DAYS_RING)
    register(shuffled_concept(DAYS_RING))

    # ---- layer sweep: pick by manifold ring-adjacency (2-D) ----
    sweep = []
    for layer in LAYERS:
        service._build_manifold(DAYS_RING.name, layer)
        service._build_behavior_manifold(DAYS_RING, layer, READOUT)
        c2 = service._manifold_cache[(DAYS_RING.name, layer)]["centroids_pca"]
        adj1 = ring_adjacency_kd(c2, 1)
        adj2 = ring_adjacency_kd(c2, 2)
        sweep.append({"layer": layer, "ring_adj_1d_linear": round(adj1, 4), "ring_adj_2d_manifold": round(adj2, 4)})
        print(f"[cyclic]   L{layer}: 1d-linear ring-adj={adj1:.3f}  2d-manifold ring-adj={adj2:.3f}", flush=True)
    best = max(sweep, key=lambda s: s["ring_adj_2d_manifold"])
    layer = best["layer"]
    print(f"[cyclic]   -> best layer {layer}", flush=True)

    # ---- ROUTING around the ring: Mon -> Fri ----
    man_rate, man_per, _ = routing_analysis(service, DAYS_RING.name, layer, SRC, TGT, NWP, "manifold")
    lin_rate, lin_per, _ = routing_analysis(service, DAYS_RING.name, layer, SRC, TGT, NWP, "linear")
    sc = cp.get_concept(DAYS_RING.name + "_shuf")
    sh_src, sh_tgt = sc.items.index(DAYS[SRC]), sc.items.index(DAYS[TGT])
    service._build_manifold(sc.name, layer); service._build_behavior_manifold(sc, layer, READOUT)
    shuf_rate, _, _ = routing_analysis(service, sc.name, layer, sh_src, sh_tgt, NWP, "manifold")
    rnd_rate, rnd_per, _ = routing_analysis(service, DAYS_RING.name, layer, SRC, TGT, NWP, "manifold", random_seed=29)
    routing_win = bool(man_rate > lin_rate and man_rate > shuf_rate)
    print(f"[cyclic] routing Mon->Fri: manifold={man_rate:.2f} linear={lin_rate:.2f} "
          f"shuffled={shuf_rate:.2f} random={rnd_rate:.2f} -> win={routing_win}", flush=True)
    print(f"[cyclic]   manifold walk: {[w['induced'] for w in man_per]}", flush=True)
    print(f"[cyclic]   linear  walk: {[w['induced'] for w in lin_per]}", flush=True)

    # ---- geometry for the ring plot ----
    fit = service.manifold_fit(DAYS_RING.name, layer)
    paths = {}
    for path in ("manifold", "linear"):
        st = service.manifold_steer(DAYS_RING.name, DAYS[TGT], layer=layer, source=DAYS[SRC],
                                    n_waypoints=NWP, max_new_tokens=1, path=path,
                                    compute_unsteered=False, compute_energy=False)
        paths[path] = [[round(float(c), 3) for c in p] for p in st["path_3d"]]

    out = {
        "model_id": MODEL, "concept": "days_ring", "layer": layer, "readout": READOUT,
        "route": f"{DAYS[SRC]} -> {DAYS[TGT]} (around the ring)", "n_waypoints": NWP,
        "reading": {"best_layer_sweep": sweep,
                    "ring_adj_1d_linear": best["ring_adj_1d_linear"],
                    "ring_adj_2d_manifold": best["ring_adj_2d_manifold"],
                    "claim": "a single linear direction cannot represent cyclic order; the 2-D manifold ring can"},
        "routing": {"manifold": round(man_rate, 4), "linear": round(lin_rate, 4),
                    "shuffled": round(shuf_rate, 4), "random": round(rnd_rate, 4),
                    "routing_win": routing_win,
                    "manifold_waypoints": man_per, "linear_waypoints": lin_per, "random_waypoints": rnd_per},
        "geometry": {"label": fit["label"], "layer": layer, "kind": fit["kind"], "items": list(DAYS),
                     "points_3d": [{"value": p["value"], "index": p["index"],
                                    "xyz": [round(float(c), 3) for c in p["xyz"]]} for p in fit["points_3d"]],
                     "curve_3d": [[round(float(c), 3) for c in p] for p in fit["curve_3d"][::2]],
                     "path_3d": paths},
    }
    dest = ROOT / "reports" / "manifold_census"
    (dest / "cyclic.json").write_text(json.dumps(out, indent=1))
    print(f"\n[cyclic] READING: 1d-linear ring-adj={best['ring_adj_1d_linear']} vs manifold(2d)={best['ring_adj_2d_manifold']}")
    print(f"[cyclic] ROUTING: manifold={man_rate:.2f} vs linear={lin_rate:.2f} (win={routing_win})")
    print(f"[cyclic] wrote {dest/'cyclic.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
