"""Card ranks: does the ace-wrap actually exist? The seam-crossing test.

The card_rank_ring win (Ace->Seven, 1.00 vs linear 0.29, shuffled 0.57) never crossed the
King->Ace seam, so it cannot by itself prove the ring closes — an open Ace..King LINE routes an
interior walk just as well. Two measurements settle it:

  CLOSURE  : in d_model space, is the King->Ace centroid gap comparable to ordinary adjacent
             gaps (ring stitched) or an outlier (open line)?
  SEAM WALK: route Jack -> Three THE SHORT WAY AROUND, through the seam
             (Jack, Queen, King, Ace, Two, Three — 6 waypoints, params 10..15 evaluated mod 13).
             * A ring routes it: Queen, King, Ace, Two induced in order.
             * A LINE cannot: the straight chord from Jack back to Three runs through
               Ten..Four in REVERSE; no 1-D line path can produce Queen->King->Ace->Two.
             Controls: linear PCA chord, shuffled-order ring, random direction (matched norm).

Run:  python scripts/_card_seam.py
Out:  reports/manifold_census/card_seam.json  (+ geometry for the seam plot)
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
from scripts._emotion_manifold import register, shuffled_concept  # noqa: E402

MODEL = "mlx-community/Qwen3.5-2B-bf16"
LAYER = 6  # the validated card_rank_ring layer
READOUT = "full_string"
NAME = "card_rank_ring"
SRC, TGT, NWP = 10, 2, 6  # Jack -> Three across the seam: params 10..15 (mod 13)


def seam_routing(service, concept_name, layer, src_i, tgt_i, nwp, arm, random_seed=None):
    """routing_analysis with mod-n spline evaluation so the walk can cross the periodic seam."""
    import torch
    cobj = cp.get_concept(concept_name)
    manifold, _, layer = service._build_manifold(concept_name, layer)
    items, n = manifold["items"], manifold["n_items"]
    spline, pca, cpca = manifold["spline"], manifold["pca"], manifold["centroids_pca"]
    prompt = cobj.steer_prompt.format(item=items[src_i])
    bundle = service.ensure_model()
    position = service._locate_item_position(bundle.tokenizer, cobj.steer_prompt, items[src_i], prompt)
    src_cent = manifold["centroids_dmodel"][src_i]

    d_fwd = (tgt_i - src_i) % n                      # forward (seam-crossing) parameter distance
    us = list(np.linspace(float(src_i), float(src_i + d_fwd), nwp))
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    per, matches = [], 0
    for u in us:
        if arm == "linear":
            t = (u - src_i) / d_fwd if d_fwd else 1.0
            pca_pt = (1 - t) * cpca[src_i] + t * cpca[tgt_i]   # straight chord, no wrap
        else:
            pca_pt = spline(float(u) % n)
        rep = pca.inverse_transform(np.asarray(pca_pt).reshape(1, -1))[0]
        vec = torch.tensor(rep, dtype=torch.float32)
        if rng is not None:
            disp = rep - src_cent
            d = rng.standard_normal(len(disp))
            d = d / (np.linalg.norm(d) + 1e-9) * np.linalg.norm(disp)
            vec = torch.tensor(src_cent + d, dtype=torch.float32)
        q = service._value_string_distribution(prompt, layer, vec, position, cobj)
        argmax = int(np.argmax(q))
        expected = int(round(u)) % n
        match = (argmax == expected)
        matches += int(match)
        per.append({"param": round(float(u), 2), "expected_idx": expected, "expected": items[expected],
                    "induced_idx": argmax, "induced": items[argmax],
                    "match": bool(match), "p_induced": round(float(q[argmax]), 4)})
    return matches / len(per), per


def main():
    t0 = time.time()
    print(f"[seam] loading {MODEL} ...", flush=True)
    service = build_mlx_service(MODEL, default_layer=12)
    concept = cp.get_concept(NAME)
    register(shuffled_concept(concept))

    manifold, _, layer = service._build_manifold(NAME, LAYER)
    service._build_behavior_manifold(concept, layer, READOUT)
    items, n = manifold["items"], manifold["n_items"]
    cd = np.asarray(manifold["centroids_dmodel"])

    # ---- CLOSURE: is the King->Ace gap an ordinary adjacent gap, or an outlier? ----
    adj = [float(np.linalg.norm(cd[i] - cd[(i + 1) % n])) for i in range(n)]
    seam_gap, nonseam = adj[n - 1], adj[: n - 1]
    closure_ratio = seam_gap / float(np.mean(nonseam))
    rank_of_seam = int(sorted(adj, reverse=True).index(seam_gap)) + 1
    dist = np.linalg.norm(cd[:, None, :] - cd[None, :, :], axis=2)
    nonadj = [float(dist[i, j]) for i in range(n) for j in range(i + 1, n)
              if (j - i) % n not in (1, n - 1)]
    seam_vs_nonadj = seam_gap / float(np.mean(nonadj))
    def nns(i, k=3):
        order = np.argsort(dist[i])
        return [(items[j], round(float(dist[i, j]), 1)) for j in order[1:k + 1]]
    ace_nn, king_nn = nns(0), nns(n - 1)
    king_rank_from_ace = int(np.argsort(dist[0]).tolist().index(n - 1))  # 1 = nearest
    print(f"[seam] CLOSURE: King->Ace gap {seam_gap:.1f} vs mean adjacent {np.mean(nonseam):.1f} "
          f"-> ratio {closure_ratio:.2f} (seam is #{rank_of_seam} largest of {n} adjacent gaps)", flush=True)
    print(f"[seam]   vs mean NON-adjacent {np.mean(nonadj):.1f} -> seam/nonadj {seam_vs_nonadj:.2f}", flush=True)
    print(f"[seam]   Ace's nearest neighbours: {ace_nn} (King is #{king_rank_from_ace})", flush=True)
    print(f"[seam]   King's nearest neighbours: {king_nn}", flush=True)

    # ---- SEAM WALK: Jack -> Three through Queen, King, Ace, Two ----
    man_rate, man_per = seam_routing(service, NAME, layer, SRC, TGT, NWP, "manifold")
    lin_rate, lin_per = seam_routing(service, NAME, layer, SRC, TGT, NWP, "linear")
    sc = cp.get_concept(NAME + "_shuf")
    service._build_manifold(sc.name, layer)
    service._build_behavior_manifold(sc, layer, READOUT)
    sh_src, sh_tgt = sc.items.index(items[SRC]), sc.items.index(items[TGT])
    shuf_rate, shuf_per = seam_routing(service, sc.name, layer, sh_src, sh_tgt, NWP, "manifold")
    rnd_rate, rnd_per = seam_routing(service, NAME, layer, SRC, TGT, NWP, "manifold", random_seed=29)
    win = bool(man_rate > lin_rate and man_rate > shuf_rate)
    print(f"[seam] ROUTING {items[SRC]}->{items[TGT]} (across the seam): "
          f"manifold={man_rate:.2f} linear={lin_rate:.2f} shuffled={shuf_rate:.2f} "
          f"random={rnd_rate:.2f} -> win={win}", flush=True)
    print(f"[seam]   manifold walk: {[w['induced'] for w in man_per]}", flush=True)
    print(f"[seam]   linear  walk: {[w['induced'] for w in lin_per]}", flush=True)

    # ---- geometry for the seam plot: full ring + the wrapped seam arc + the chord ----
    fit = service.manifold_fit(NAME, layer)
    us = np.linspace(SRC, SRC + (TGT - SRC) % n, 60)
    pca3 = np.asarray([manifold["spline"](float(u) % n)[:3] for u in us])
    out = {
        "model_id": MODEL, "concept": NAME, "layer": layer, "readout": READOUT,
        "route": {"src": SRC, "tgt": TGT, "n_waypoints": NWP,
                  "desc": f"{items[SRC]} -> {items[TGT]} (across the King->Ace seam)"},
        "closure": {"seam_gap": round(seam_gap, 2), "mean_adjacent_nonseam": round(float(np.mean(nonseam)), 2),
                    "closure_ratio": round(closure_ratio, 3), "seam_rank_among_adjacent": rank_of_seam,
                    "mean_nonadjacent": round(float(np.mean(nonadj)), 2),
                    "seam_vs_nonadjacent": round(seam_vs_nonadj, 3),
                    "ace_nearest_neighbours": ace_nn, "king_nearest_neighbours": king_nn,
                    "king_rank_from_ace": king_rank_from_ace,
                    "adjacent_gaps": [round(a, 2) for a in adj],
                    "claim": "ratio ~1 = the ring is stitched; >>1 = an open line wearing a cyclic fit"},
        "routing": {"manifold": round(man_rate, 4), "linear": round(lin_rate, 4),
                    "shuffled": round(shuf_rate, 4), "random": round(rnd_rate, 4), "routing_win": win,
                    "manifold_waypoints": man_per, "linear_waypoints": lin_per,
                    "shuffled_waypoints": shuf_per, "random_waypoints": rnd_per},
        "geometry": {"label": "Card ranks (ring)", "layer": layer, "kind": "cyclic", "items": list(items),
                     "points_3d": [{"value": p["value"], "index": p["index"],
                                    "xyz": [round(float(c), 3) for c in p["xyz"]]} for p in fit["points_3d"]],
                     "curve_3d": [[round(float(c), 3) for c in p] for p in fit["curve_3d"][::2]],
                     "seam_arc_3d": [[round(float(c), 3) for c in p] for p in pca3]},
    }
    dest = ROOT / "reports" / "manifold_census" / "card_seam.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"[seam] wrote {dest} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
