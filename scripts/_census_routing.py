"""Routing + energy test + 3D-geometry dump for the census-discovered clean manifolds.

Reuses the emotion-experiment machinery verbatim (`scripts/_emotion_manifold.py`): for each
clean ORDINAL concept at its registry best_layer, computes
  - energy: manifold vs linear chord (full_string read-out)  [gate b: faithfulness]
  - centroid-routing: manifold / linear / shuffled-order / random-direction agreement [gate c]
and dumps the real fitted 3D geometry (centroids, spline, manifold+linear walk paths) for the
interactive atlas page. Answers: is there a second centroid-routing win beyond arousal?

Run:  python scripts/_census_routing.py
Out:  reports/manifold_census/routing.json   (verdicts + per-waypoint walks)
      reports/manifold_census/geometry_3d.json  (points_3d / curve_3d / path_3d per concept)
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
from qwen_scope_lab.mlx_backend import build_mlx_service  # noqa: E402
# reuse the emotion experiment's helpers exactly
from scripts._emotion_manifold import isometry_r, register, routing_analysis, shuffled_concept  # noqa: E402

MODEL = "mlx-community/Qwen3.5-2B-bf16"
N_WAYPOINTS = 6
READOUT = "full_string"

# clean ORDINAL concepts from the census (time_of_day is cyclic → geometry only, no ordinal routing)
ROUTE = ["priority", "age_stage", "difficulty", "certainty", "distance", "quantity",
         "wealth", "hardness", "quality", "weight", "formality"]
GEOMETRY_ONLY = ["time_of_day"]


def r3(x):
    return round(float(x), 3)


def dump_geometry(service, name, layer):
    fit = service.manifold_fit(name, layer)
    items = fit["items"]
    paths = {}
    for path in ("manifold", "linear"):
        steer = service.manifold_steer(name, items[-1], layer=layer, source=items[0],
                                       n_waypoints=N_WAYPOINTS, max_new_tokens=1,
                                       path=path, compute_unsteered=False, compute_energy=False)
        paths[path] = [[r3(c) for c in p] for p in steer["path_3d"]]
    return {
        "label": fit["label"], "layer": layer, "kind": fit["kind"], "items": items,
        "points_3d": [{"value": p["value"], "index": p["index"], "xyz": [r3(c) for c in p["xyz"]]}
                      for p in fit["points_3d"]],
        "curve_3d": [[r3(c) for c in p] for p in fit["curve_3d"][::2]],
        "path_3d": paths, "quality": fit["quality"],
    }


def main():
    t0 = time.time()
    print(f"[route] loading {MODEL} (no SAE) ...", flush=True)
    service = build_mlx_service(MODEL, default_layer=12)

    routing_results, geometry = [], {"model_id": MODEL, "n_waypoints": N_WAYPOINTS, "concepts": {}}

    for name in ROUTE:
        c = cp.get_concept(name)
        register(c)
        register(shuffled_concept(c))  # the shuffled-order control variant
        layer = c.best_layer
        src_i, tgt_i = 0, len(c.items) - 1

        service._build_manifold(c.name, layer)
        service._build_behavior_manifold(c, layer, READOUT)
        iso = isometry_r(service, c.name, layer, READOUT)

        comp = service.manifold_compare(c.name, c.items[tgt_i], layer=layer, source=c.items[src_i],
                                        n_waypoints=N_WAYPOINTS, max_new_tokens=8, behavior_readout=READOUT)
        m_e, l_e = comp["manifold"]["mean_energy"], comp["linear"]["mean_energy"]
        gap = round(l_e - m_e, 4) if (m_e is not None and l_e is not None) else None
        faithful = bool(m_e is not None and l_e is not None and m_e < l_e)

        man_rate, man_per, _ = routing_analysis(service, c.name, layer, src_i, tgt_i, N_WAYPOINTS, "manifold")
        lin_rate, lin_per, _ = routing_analysis(service, c.name, layer, src_i, tgt_i, N_WAYPOINTS, "linear")
        sc = cp.get_concept(c.name + "_shuf")
        sh_src, sh_tgt = sc.items.index(c.items[src_i]), sc.items.index(c.items[tgt_i])
        service._build_manifold(sc.name, layer)
        service._build_behavior_manifold(sc, layer, READOUT)
        shuf_rate, _, _ = routing_analysis(service, sc.name, layer, sh_src, sh_tgt, N_WAYPOINTS, "manifold")
        rnd_rate, rnd_per, _ = routing_analysis(service, c.name, layer, src_i, tgt_i, N_WAYPOINTS, "manifold", random_seed=29)

        routing_win = bool(man_rate > lin_rate and man_rate > shuf_rate)
        verdict = "centroid_routing" if (routing_win and faithful) else (
            "routes_not_faithful" if routing_win else ("faithful_not_routing" if faithful else "no_win"))
        print(f"[route] {name:12s} L{layer:>2} iso={iso:.3f} energy_gap={gap} "
              f"man={man_rate:.2f} lin={lin_rate:.2f} shuf={shuf_rate:.2f} rnd={rnd_rate:.2f} -> {verdict} ({time.time()-t0:.0f}s)", flush=True)

        routing_results.append({
            "concept": name, "label": c.label, "layer": layer, "items": list(c.items),
            "source": c.items[src_i], "target": c.items[tgt_i],
            "isometry_r": r3(iso) if iso is not None else None,
            "energy": {"manifold": m_e, "linear": l_e, "gap_linear_minus_manifold": gap, "manifold_more_faithful": faithful},
            "routing": {"manifold": round(man_rate, 4), "linear": round(lin_rate, 4),
                        "shuffled": round(shuf_rate, 4), "random": round(rnd_rate, 4), "routing_win": routing_win},
            "verdict": verdict,
            "manifold_waypoints": man_per, "linear_waypoints": lin_per, "random_waypoints": rnd_per,
        })
        geometry["concepts"][name] = dump_geometry(service, name, layer)

    # geometry-only (cyclic) concepts
    for name in GEOMETRY_ONLY:
        c = cp.get_concept(name)
        register(c)
        service._build_manifold(c.name, c.best_layer)
        geometry["concepts"][name] = dump_geometry(service, name, c.best_layer)
        print(f"[geom]  {name:12s} L{c.best_layer} (cyclic, geometry only)", flush=True)

    n_route = sum(r["verdict"] == "centroid_routing" for r in routing_results)
    n_faith = sum(r["energy"]["manifold_more_faithful"] for r in routing_results)
    routing_results.sort(key=lambda r: (r["verdict"] != "centroid_routing", -r["routing"]["manifold"]))

    dest = ROOT / "reports" / "manifold_census"
    (dest / "routing.json").write_text(json.dumps(
        {"model_id": MODEL, "readout": READOUT, "n_waypoints": N_WAYPOINTS,
         "n_routing_wins": n_route, "n_faithful": n_faith, "n_tested": len(ROUTE),
         "results": routing_results}, indent=1))
    (dest / "geometry_3d.json").write_text(json.dumps(geometry, indent=0))

    print(f"\n[route] {n_route}/{len(ROUTE)} centroid-routing wins, {n_faith}/{len(ROUTE)} manifold-more-faithful")
    print(f"[route] wrote {dest}/routing.json + geometry_3d.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
