"""Emotion-space manifold steering audit (real Qwen3.5-2B, MLX, on-device).

Goodfire-style concept-manifold steering applied to ORDERED EMOTION concepts, à la Anthropic's
emotion-vectors work. Tests whether the manifold (centroid-routing) path matches emotional
transitions better than the straight linear chord, and whether the manifold path actually routes
*through* the intermediate emotions where the chord skips them.

Gates (preregistered, see report.md):
  (a) clean manifold     : isometry r >= 0.9
  (b) manifold faithful  : full_string energy gap (linear - manifold) > 0 on a MAJORITY of concepts
  (c) centroid-routing   : intermediate-waypoint argmax-agreement(manifold) > linear AND > shuffled

Controls:
  - random-direction at matched strength (replace concept residual with a random vector matched in
    norm to the manifold waypoint displacement) -> energy should be high / agreement ~chance
  - shuffled-ordering    : fit the manifold with the emotion order permuted; routing-through-
    intermediates must NOT survive shuffling, else it is a fitting artifact

Run:  python3 scripts/_emotion_manifold.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qwen_scope_lab.concept_presets as cp
from qwen_scope_lab.concept_presets import Concept
from qwen_scope_lab.mlx_backend import build_mlx_service

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"

# ---- emotion concepts (item lists + {item} carrier templates in the house style) ----
_FEEL_TEMPLATES = ("She felt {item}", "He seemed {item}", "I am {item}",
                   "A {item} mood", "They looked {item}", "Feeling {item}")

EMOTION_CONCEPTS = [
    Concept(
        "emotion_valence_intensity", "Anger->joy valence-intensity line", "ordinal",
        ("furious", "angry", "annoyed", "calm", "content", "delighted", "euphoric"),
        _FEEL_TEMPLATES, "Right now she feels {item}. Honestly, she is", best_layer=None),
    Concept(
        "emotion_fear", "Fear gradation", "ordinal",
        ("terrified", "afraid", "anxious", "uneasy", "calm"),
        _FEEL_TEMPLATES, "Right now he feels {item}. Honestly, he is", best_layer=None),
    Concept(
        "emotion_arousal", "Low->high arousal line", "ordinal",
        ("numb", "bored", "calm", "alert", "excited", "frantic"),
        _FEEL_TEMPLATES, "Right now they feel {item}. Honestly, they are", best_layer=None),
]


def register(concept: Concept):
    cp._ALL[concept.name] = concept


def shuffled_concept(c: Concept, seed: int = 13) -> Concept:
    rng = np.random.default_rng(seed)
    perm = list(c.items)
    rng.shuffle(perm)
    sc = Concept(c.name + "_shuf", c.label + " (shuffled order)", c.kind, tuple(perm),
                 c.templates, c.steer_prompt, best_layer=c.best_layer)
    return sc


# ---- isometry helper (matches scripts/_c05_mlx_audit.py) ----
def _cumdist(pts, metric):
    n = len(pts)
    seg = [metric(pts[k], pts[k + 1]) for k in range(n - 1)]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.abs(cum[:, None] - cum[None, :])


def isometry_r(service, concept_name, layer, readout):
    eucl = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    hell = lambda a, b: float(np.linalg.norm(np.sqrt(np.clip(a, 0, None)) - np.sqrt(np.clip(b, 0, None))) / np.sqrt(2))
    mk = service._manifold_cache.get((concept_name, layer))
    bk = service._behavior_cache.get((concept_name, layer, readout))
    if mk is None or bk is None:
        return None
    d_h = _cumdist(np.asarray(mk["centroids_pca"]), eucl)
    d_y = _cumdist(np.asarray(bk["centroids"]), hell)
    iu = np.triu_indices(d_h.shape[0], 1)
    a, bb = d_h[iu], d_y[iu]
    return float(np.corrcoef(a, bb)[0, 1]) if (a.std() > 0 and bb.std() > 0) else None


# ---- per-waypoint centroid-routing analysis ----
def waypoint_replacements(service, concept_name, layer, src_i, tgt_i, n_waypoints, path):
    """Return list of (u_ideal_index, replacement_dmodel_vector) for each waypoint, mirroring
    manifold_steer's geometry exactly (ordinal concepts only)."""
    import torch
    manifold, _, layer = service._build_manifold(concept_name, layer)
    cpca, spline, pca = manifold["centroids_pca"], manifold["spline"], manifold["pca"]
    us = list(np.linspace(float(src_i), float(tgt_i), max(2, n_waypoints)))
    steps = len(us)
    out = []
    for wi, u in enumerate(us):
        t = wi / (steps - 1) if steps > 1 else 1.0
        if path == "linear":
            pca_pt = (1 - t) * cpca[src_i] + t * cpca[tgt_i]
            ideal = (1 - t) * src_i + t * tgt_i
        else:
            pca_pt = spline(float(u))
            ideal = float(u)
        rep = pca.inverse_transform(np.asarray(pca_pt).reshape(1, -1))[0]
        out.append((ideal, torch.tensor(rep, dtype=torch.float32)))
    return out, manifold


def routing_analysis(service, concept_name, layer, src_i, tgt_i, n_waypoints, path,
                     random_seed=None, random_norm=None):
    """For each waypoint, induce the distribution over the concept values and record:
       expected intermediate index (rounded ideal), induced argmax index, and whether they match.
    Returns (agreement_rate, per_waypoint list, mean displacement norm)."""
    import torch
    cobj = cp.get_concept(concept_name)
    manifold, _, layer = service._build_manifold(concept_name, layer)
    items = manifold["items"]
    prompt = cobj.steer_prompt.format(item=items[src_i])
    bundle = service.ensure_model()
    position = service._locate_item_position(bundle.tokenizer, cobj.steer_prompt, items[src_i], prompt)
    src_cent = manifold["centroids_dmodel"][src_i]

    reps, _ = waypoint_replacements(service, concept_name, layer, src_i, tgt_i, n_waypoints, path)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    per, matches, norms = [], 0, []
    for ideal, rep in reps:
        vec = rep
        if rng is not None:  # random-direction control at matched displacement norm
            disp = rep.numpy() - src_cent
            d = rng.standard_normal(len(disp))
            d = d / (np.linalg.norm(d) + 1e-9) * np.linalg.norm(disp)
            vec = torch.tensor(src_cent + d, dtype=torch.float32)
        norms.append(float(np.linalg.norm(vec.numpy() - src_cent)))
        q = service._value_string_distribution(prompt, layer, vec, position, cobj)
        argmax = int(np.argmax(q))
        expected = int(round(ideal))
        match = (argmax == expected)
        matches += int(match)
        per.append({"expected_idx": expected, "expected": items[expected],
                    "induced_idx": argmax, "induced": items[argmax],
                    "match": bool(match), "p_induced": round(float(q[argmax]), 4)})
    return matches / len(per), per, round(float(np.mean(norms)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=[8, 12, 16])
    ap.add_argument("--n-waypoints", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=6)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[emo] loading MLX model {DEFAULT_MLX_MODEL} (no SAE) ...", flush=True)
    service = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)

    for c in EMOTION_CONCEPTS:
        register(c)
        register(shuffled_concept(c))

    results = []
    for c in EMOTION_CONCEPTS:
        print(f"\n[emo] === {c.name} ({c.label}) ===", flush=True)
        src_i, tgt_i = 0, len(c.items) - 1

        # ---- layer sweep: pick the layer with the best isometry r (full_string) ----
        sweep = []
        for layer in args.layers:
            service._build_manifold(c.name, layer)
            service._build_behavior_manifold(cp.get_concept(c.name), layer, "full_string")
            r = isometry_r(service, c.name, layer, "full_string")
            q = service._manifold_quality(service._manifold_cache[(c.name, layer)])
            sweep.append({"layer": layer, "isometry_r_full_string": round(r, 4) if r is not None else None,
                          "manifold_quality": q})
            print(f"[emo]   layer {layer}: iso_r={r:.4f} {q['metric_name']}={q['metric']}" if r is not None
                  else f"[emo]   layer {layer}: iso_r=NA", flush=True)
        best = max(sweep, key=lambda s: (s["isometry_r_full_string"] or -1))
        layer = best["layer"]
        print(f"[emo]   -> best layer {layer} (iso_r={best['isometry_r_full_string']})", flush=True)

        # ---- manifold vs linear energy (full_string) ----
        comp = service.manifold_compare(c.name, c.items[tgt_i], layer=layer, source=c.items[src_i],
                                        n_waypoints=args.n_waypoints, max_new_tokens=args.max_new_tokens,
                                        behavior_readout="full_string")
        m_e, l_e = comp["manifold"]["mean_energy"], comp["linear"]["mean_energy"]
        gap = round(l_e - m_e, 4) if (m_e is not None and l_e is not None) else None
        manifold_more_faithful = bool(m_e is not None and l_e is not None and m_e < l_e)
        print(f"[emo]   energy: manifold={m_e} linear={l_e} gap={gap} faithful={manifold_more_faithful}", flush=True)

        # ---- centroid-routing: per-waypoint argmax agreement ----
        man_rate, man_per, man_norm = routing_analysis(service, c.name, layer, src_i, tgt_i,
                                                       args.n_waypoints, "manifold")
        lin_rate, lin_per, lin_norm = routing_analysis(service, c.name, layer, src_i, tgt_i,
                                                       args.n_waypoints, "linear")
        # shuffled control: fit manifold on permuted order, route src->tgt in shuffled index space
        sc = cp.get_concept(c.name + "_shuf")
        # map the same emotional endpoints into the shuffled item list
        sh_src = sc.items.index(c.items[src_i])
        sh_tgt = sc.items.index(c.items[tgt_i])
        service._build_manifold(sc.name, layer)
        service._build_behavior_manifold(sc, layer, "full_string")
        shuf_rate, shuf_per, _ = routing_analysis(service, sc.name, layer, sh_src, sh_tgt,
                                                  args.n_waypoints, "manifold")
        # random-direction control (matched norm to manifold waypoint displacement)
        rnd_rate, rnd_per, rnd_norm = routing_analysis(service, c.name, layer, src_i, tgt_i,
                                                       args.n_waypoints, "manifold",
                                                       random_seed=29)
        centroid_routing = bool(man_rate > lin_rate and man_rate > shuf_rate)
        print(f"[emo]   routing agreement: manifold={man_rate:.2f} linear={lin_rate:.2f} "
              f"shuffled={shuf_rate:.2f} random={rnd_rate:.2f} -> routing={centroid_routing}", flush=True)

        results.append({
            "concept": c.name, "label": c.label, "kind": c.kind,
            "items": list(c.items), "source": c.items[src_i], "target": c.items[tgt_i],
            "layer_sweep": sweep, "best_layer": layer,
            "isometry_r_full_string": best["isometry_r_full_string"],
            "manifold_quality": best["manifold_quality"],
            "energy": {"manifold": m_e, "linear": l_e, "gap_linear_minus_manifold": gap,
                       "manifold_more_faithful": manifold_more_faithful,
                       "behavior_readout": "full_string"},
            "energy_first_token": None,  # filled below
            "routing": {
                "manifold_agreement": round(man_rate, 4), "linear_agreement": round(lin_rate, 4),
                "shuffled_agreement": round(shuf_rate, 4), "random_agreement": round(rnd_rate, 4),
                "centroid_routing": centroid_routing,
                "manifold_waypoints": man_per, "linear_waypoints": lin_per,
                "shuffled_waypoints": shuf_per, "random_waypoints": rnd_per,
                "manifold_disp_norm": man_norm, "linear_disp_norm": lin_norm, "random_disp_norm": rnd_norm,
            },
            "samples": {
                "unsteered": comp["unsteered_text"],
                "manifold_final": comp["manifold"]["steered_text"],
                "linear_final": comp["linear"]["steered_text"],
            },
        })
        # first-token energy for continuity (as the C05 doc requests)
        comp_ft = service.manifold_compare(c.name, c.items[tgt_i], layer=layer, source=c.items[src_i],
                                           n_waypoints=args.n_waypoints, max_new_tokens=args.max_new_tokens,
                                           behavior_readout="first_token")
        mf, lf = comp_ft["manifold"]["mean_energy"], comp_ft["linear"]["mean_energy"]
        results[-1]["energy_first_token"] = {
            "manifold": mf, "linear": lf,
            "gap_linear_minus_manifold": round(lf - mf, 4) if (mf is not None and lf is not None) else None,
            "manifold_more_faithful": bool(mf is not None and lf is not None and mf < lf)}

    # ---- gate verdicts ----
    n = len(results)
    iso_ok = [r for r in results if (r["isometry_r_full_string"] or 0) >= 0.9]
    faithful = [r for r in results if r["energy"]["manifold_more_faithful"]]
    routing = [r for r in results if r["routing"]["centroid_routing"]]
    gates = {
        "gate_a_clean_manifold": {
            "threshold": "isometry_r_full_string >= 0.9",
            "pass_concepts": [r["concept"] for r in iso_ok], "n_pass": len(iso_ok), "n_total": n,
            "verdict": "PASS" if len(iso_ok) >= 1 else "FAIL",
            "majority": len(iso_ok) > n / 2},
        "gate_b_manifold_faithful": {
            "threshold": "full_string energy gap > 0 on a majority of concepts",
            "pass_concepts": [r["concept"] for r in faithful], "n_pass": len(faithful), "n_total": n,
            "verdict": "PASS" if len(faithful) > n / 2 else "FAIL"},
        "gate_c_centroid_routing": {
            "threshold": "manifold argmax-agreement > linear AND > shuffled",
            "pass_concepts": [r["concept"] for r in routing], "n_pass": len(routing), "n_total": n,
            "verdict": "PASS" if len(routing) > n / 2 else "FAIL"},
    }

    out = {
        "model_id": DEFAULT_MLX_MODEL,
        "experiment": "emotion-space manifold steering (Goodfire-style; emotion-vectors framing)",
        "date": time.strftime("%Y-%m-%d"),
        "params": {"layers_swept": args.layers, "n_waypoints": args.n_waypoints,
                   "max_new_tokens": args.max_new_tokens, "behavior_readout": "full_string",
                   "shuffle_seed": 13, "random_seed": 29},
        "gates": gates,
        "results": results,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    os.makedirs("reports/emotion_manifold", exist_ok=True)
    with open("reports/emotion_manifold/verdict.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[emo] wrote reports/emotion_manifold/verdict.json", flush=True)
    print(json.dumps(gates, indent=2))
    print(f"[emo] wall-clock {out['wall_clock_sec']}s", flush=True)
    return out


if __name__ == "__main__":
    main()
