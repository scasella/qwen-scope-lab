"""Manifold zoo validation script for survivor candidates from the cyclic/fold ideation pass.

This mirrors scripts/_cyclic_atlas.py: for each candidate, sweep the same real MLX 2B layers by
2-D ring adjacency, then test centroid routing against linear, shuffled-order, and random-direction
controls. The gate is unchanged: manifold > linear AND manifold > shuffled.

Run:  python scripts/_manifold_zoo.py
Out:  reports/manifold_census/manifold_zoo.json  (+ per-concept geometry for 3D plots)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import qwen_scope_lab.concept_presets as cp  # noqa: E402
from qwen_scope_lab.concept_presets import Concept  # noqa: E402
from qwen_scope_lab.mlx_backend import build_mlx_service  # noqa: E402
from scripts._cyclic_manifold import ring_adjacency_kd  # noqa: E402
from scripts._emotion_manifold import register, routing_analysis, shuffled_concept  # noqa: E402

MODEL = "mlx-community/Qwen3.5-2B-bf16"
LAYERS = [6, 8, 12, 14, 16]
READOUT = "full_string"

RINGS = [
    # (concept, src_i, tgt_i, n_waypoints)
    (Concept("chinese_zodiac_ring", "Chinese zodiac (ring)", "cyclic",
             ("Rat", "Ox", "Tiger", "Rabbit", "Dragon", "Snake", "Horse",
              "Goat", "Monkey", "Rooster", "Dog", "Pig"),
             ("The zodiac animal is {item}", "She was born in the year of the {item}",
              "His Chinese zodiac sign is {item}", "The calendar marks the year of the {item}",
              "A person born that year is a {item}", "The lunar zodiac animal is {item}"),
             "The Chinese zodiac animal is {item}. So the zodiac animal is", best_layer=None),
     0, 6, 7),
    (Concept("political_horseshoe_path", "Political horseshoe (fold)", "ordinal",
             ("communist", "socialist", "liberal", "centrist", "conservative",
              "reactionary", "fascist"),
             ("The ideology is {item}", "The political tradition is {item}",
              "The platform is described as {item}", "The political label is {item}",
              "The movement is considered {item}", "The analyst labels the ideology as {item}"),
             "The ideology is {item}. So the ideology is", best_layer=None),
     0, 6, 7),
    (Concept("card_rank_ring", "Card ranks (ring)", "cyclic",
             ("Ace", "Two", "Three", "Four", "Five", "Six", "Seven",
              "Eight", "Nine", "Ten", "Jack", "Queen", "King"),
             ("The card rank is {item}", "She drew a card ranked {item}",
              "The face value is {item}", "The playing card is a {item}",
              "In ace-low rank order, the card is {item}",
              "In ace-high rank order, the card is {item}"),
             "The Ace-through-King card rank is {item}. The rank is", best_layer=None),
     0, 6, 7),
    (Concept("solfege_ring", "Solfege (ring)", "cyclic",
             ("Do", "Re", "Mi", "Fa", "Sol", "La", "Ti"),
             ("The solfege syllable is {item}", "The scale degree syllable is {item}",
              "The music lesson reached {item}", "The choir practiced {item}",
              "In movable-do solfege, the syllable is {item}",
              "The octave-scale solfege note is {item}"),
             "The solfege syllable is {item}. The syllable is", best_layer=None),
     0, 4, 5),
    (Concept("cell_cycle_ring", "Cell cycle (ring)", "cyclic",
             ("interphase", "prophase", "prometaphase", "metaphase",
              "anaphase", "telophase", "cytokinesis"),
             ("The microscope diagram labels the phase as {item}",
              "The biology worksheet shows the cell in {item}",
              "The cell-cycle chart marks this phase {item}",
              "The observed division stage is {item}",
              "The lab notebook records the phase as {item}",
              "The current cell-cycle label is {item}"),
             "The cell-cycle phase is {item}. So the phase is", best_layer=None),
     0, 4, 5),
    (Concept("engine_cycle_ring", "Engine cycle (ring)", "cyclic",
             ("intake", "compression", "ignition", "combustion", "expansion", "exhaust"),
             ("The engine diagram labels this step {item}",
              "The piston-cycle chart shows {item}",
              "In the motor sequence, the step is {item}",
              "The mechanical phase being shown is {item}",
              "The current engine-cycle label is {item}",
              "The cylinder is at the step called {item}"),
             "The engine cycle step is {item}. So the current step is", best_layer=None),
     0, 3, 4),
]


def run_ring(service, concept, src, tgt, nwp):
    register(concept)
    register(shuffled_concept(concept))

    sweep = []
    for layer in LAYERS:
        service._build_manifold(concept.name, layer)
        c2 = service._manifold_cache[(concept.name, layer)]["centroids_pca"]
        adj1 = ring_adjacency_kd(c2, 1)
        adj2 = ring_adjacency_kd(c2, 2)
        sweep.append({"layer": layer, "ring_adj_1d_linear": round(adj1, 4),
                      "ring_adj_2d_manifold": round(adj2, 4)})
        print(f"[zoo]   {concept.name} L{layer}: 1d={adj1:.3f} 2d={adj2:.3f}", flush=True)
    best = max(sweep, key=lambda s: (s["ring_adj_2d_manifold"], -s["layer"]))
    layer = best["layer"]
    service._build_behavior_manifold(concept, layer, READOUT)

    items = concept.items
    man_rate, man_per, _ = routing_analysis(service, concept.name, layer, src, tgt, nwp, "manifold")
    lin_rate, lin_per, _ = routing_analysis(service, concept.name, layer, src, tgt, nwp, "linear")
    sc = cp.get_concept(concept.name + "_shuf")
    sh_src, sh_tgt = sc.items.index(items[src]), sc.items.index(items[tgt])
    service._build_manifold(sc.name, layer)
    service._build_behavior_manifold(sc, layer, READOUT)
    shuf_rate, _, _ = routing_analysis(service, sc.name, layer, sh_src, sh_tgt, nwp, "manifold")
    rnd_rate, _, _ = routing_analysis(service, concept.name, layer, src, tgt, nwp, "manifold", random_seed=29)
    win = bool(man_rate > lin_rate and man_rate > shuf_rate)
    print(f"[zoo] {concept.name} L{layer} routing {items[src]}->{items[tgt]}: "
          f"manifold={man_rate:.2f} linear={lin_rate:.2f} shuffled={shuf_rate:.2f} "
          f"random={rnd_rate:.2f} -> win={win}", flush=True)
    print(f"[zoo]   manifold walk: {[w['induced'] for w in man_per]}", flush=True)
    print(f"[zoo]   linear  walk: {[w['induced'] for w in lin_per]}", flush=True)

    fit = service.manifold_fit(concept.name, layer)
    return {
        "concept": concept.name, "label": concept.label, "layer": layer,
        "items": list(items), "steer_prompt": concept.steer_prompt,
        "route": {"src": src, "tgt": tgt, "n_waypoints": nwp,
                  "desc": f"{items[src]} -> {items[tgt]} (around the ring)"},
        "reading": {"sweep": sweep, "ring_adj_1d_linear": best["ring_adj_1d_linear"],
                    "ring_adj_2d_manifold": best["ring_adj_2d_manifold"]},
        "routing": {"manifold": round(man_rate, 4), "linear": round(lin_rate, 4),
                    "shuffled": round(shuf_rate, 4), "random": round(rnd_rate, 4),
                    "routing_win": win,
                    "manifold_waypoints": man_per, "linear_waypoints": lin_per},
        "geometry": {"points_3d": [{"value": p["value"], "index": p["index"],
                                    "xyz": [round(float(c), 3) for c in p["xyz"]]}
                                   for p in fit["points_3d"]],
                     "curve_3d": [[round(float(c), 3) for c in p] for p in fit["curve_3d"][::2]]},
    }


def main():
    t0 = time.time()
    print(f"[zoo] loading {MODEL} ...", flush=True)
    service = build_mlx_service(MODEL, default_layer=12)
    results = []
    for concept, src, tgt, nwp in RINGS:
        t1 = time.time()
        results.append(run_ring(service, concept, src, tgt, nwp))
        print(f"[zoo] {concept.name} done in {time.time()-t1:.0f}s\n", flush=True)

    wins = [r["concept"] for r in results if r["routing"]["routing_win"]]
    out = {"model_id": MODEL, "readout": READOUT, "layers_swept": LAYERS,
           "n_rings": len(results), "n_routing_wins": len(wins), "wins": wins,
           "note": ("Survivor candidates from the manifold-zoo ideation pass, encoded with the "
                    "same protocol as cyclic_atlas.json. Routing waypoints are the spline at "
                    "integer item params = the item centroids, walked the named direction; "
                    "linear = straight PCA chord. The political horseshoe candidate is an "
                    "exploratory ordinal fold/path, not a cyclic topology claim."),
           "results": results}
    dest = ROOT / "reports" / "manifold_census" / "manifold_zoo.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[zoo] wins: {len(wins)}/{len(results)} -> {wins}")
    print(f"[zoo] wrote {dest} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
