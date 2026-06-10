"""Cyclic atlas — do OUR OWN cyclic concepts (not Goodfire's days-of-week example) beat linear steering?

The days-of-week ring demonstrated the cyclic exception, but days-of-week is the worked example in
Goodfire's manifold papers. This sweeps the same protocol over six net-new cyclic concepts from our
census/atlas work, each with a position-readout prompt (the model reports the injected value, not its
successor):

  time_of_day   dawn -> evening        (6-ring,  5 waypoints)
  hues          red -> blue            (6-ring,  5 waypoints, color wheel)
  compass       North -> South         (8-ring,  5 waypoints)
  moon_phases   new moon -> full moon  (8-ring,  5 waypoints)
  months        January -> July        (12-ring, 7 waypoints — the chord is a diameter)
  zodiac        Aries -> Libra         (12-ring, 7 waypoints)

Per concept: layer sweep picks the best 2-D ring fit (vs the 1-D linear-direction baseline), then
routing around the ring (manifold spline at integer item params = the item centroids) vs the straight
PCA chord, with shuffled-order and random-direction controls. Same gate as always:
manifold > linear AND manifold > shuffled.

Run:  python scripts/_cyclic_atlas.py
Out:  reports/manifold_census/cyclic_atlas.json  (+ per-ring geometry for the 3D plots)
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
    (Concept("time_ring", "Time of day (ring)", "cyclic",
             ("dawn", "morning", "noon", "afternoon", "evening", "night"),
             ("It was {item}", "By {item}", "Every {item}", "Around {item}", "Late {item}",
              "We met at {item}"),
             "It is {item} right now. The time of day is", best_layer=None),
     0, 4, 5),
    (Concept("hues_ring", "Color wheel (ring)", "cyclic",
             ("red", "orange", "yellow", "green", "blue", "purple"),
             ("The paint is {item}", "She wore {item}", "The wall was painted {item}",
              "A splash of {item}", "The logo is {item}", "Dyed {item}"),
             "The paint is {item}. The color is", best_layer=None),
     0, 4, 5),
    (Concept("compass_ring", "Compass (ring)", "cyclic",
             ("North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"),
             ("Head {item}", "Facing {item}", "It lies to the {item}", "Travel {item}",
              "Winds from the {item}", "Bearing {item}"),
             "The ship is sailing {item}. Its heading is", best_layer=None),
     0, 4, 5),
    (Concept("moon_ring", "Moon phases (ring)", "cyclic",
             ("new moon", "waxing crescent", "first quarter", "waxing gibbous",
              "full moon", "waning gibbous", "last quarter", "waning crescent"),
             ("The moon is {item}", "Tonight brings a {item}", "We saw the {item}",
              "The calendar shows a {item}", "A {item} rose", "Under the {item}"),
             "The moon tonight is {item}. The moon phase is", best_layer=None),
     0, 4, 5),
    (Concept("months_ring", "Months (ring)", "cyclic",
             ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"),
             ("The month is {item}", "We met in {item}", "It was a cold {item}",
              "Born in {item}", "Since last {item}", "By {item}"),
             "The month is {item}. So the current month is", best_layer=None),
     0, 6, 7),
    (Concept("zodiac_ring", "Zodiac (ring)", "cyclic",
             ("Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra",
              "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"),
             ("She is a {item}", "Born under {item}", "His sign is {item}",
              "The horoscope for {item}", "A typical {item}", "True to {item}"),
             "She was born under {item}. Her zodiac sign is", best_layer=None),
     0, 6, 7),
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
        print(f"[atlas]   {concept.name} L{layer}: 1d={adj1:.3f} 2d={adj2:.3f}", flush=True)
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
    print(f"[atlas] {concept.name} L{layer} routing {items[src]}->{items[tgt]}: "
          f"manifold={man_rate:.2f} linear={lin_rate:.2f} shuffled={shuf_rate:.2f} "
          f"random={rnd_rate:.2f} -> win={win}", flush=True)
    print(f"[atlas]   manifold walk: {[w['induced'] for w in man_per]}", flush=True)
    print(f"[atlas]   linear  walk: {[w['induced'] for w in lin_per]}", flush=True)

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
    print(f"[atlas] loading {MODEL} ...", flush=True)
    service = build_mlx_service(MODEL, default_layer=12)
    results = []
    for concept, src, tgt, nwp in RINGS:
        t1 = time.time()
        results.append(run_ring(service, concept, src, tgt, nwp))
        print(f"[atlas] {concept.name} done in {time.time()-t1:.0f}s\n", flush=True)

    wins = [r["concept"] for r in results if r["routing"]["routing_win"]]
    out = {"model_id": MODEL, "readout": READOUT, "layers_swept": LAYERS,
           "n_rings": len(results), "n_routing_wins": len(wins), "wins": wins,
           "note": ("Net-new cyclic concepts (days-of-week is the worked example in Goodfire's "
                    "manifold papers and is kept as the credited baseline in cyclic.json). "
                    "Routing waypoints are the ring spline at integer item params = the item "
                    "centroids, walked the named direction; linear = straight PCA chord."),
           "results": results}
    dest = ROOT / "reports" / "manifold_census" / "cyclic_atlas.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[atlas] wins: {len(wins)}/{len(results)} -> {wins}")
    print(f"[atlas] wrote {dest} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
