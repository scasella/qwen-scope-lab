"""Recovery diagnostic: are the 'non-steerable' clean manifolds actually useful at a softer bar?

The census routing test used the harshest possible metric — greedy-argmax agreement, beating BOTH a
linear chord AND a strong shuffled-order control. Only 2/11 cleared it. This re-scores the same 11
clean ordinals (plus arousal as a gold-router anchor and fear as a known-dead anchor) at a sweep of
layers with FOUR metrics per arm, then buckets each manifold by what it can actually do:

  - argmax_agree   : greedy-argmax == expected intermediate            (the original harsh metric)
  - top2_agree     : expected intermediate in the induced top-2        (temperature would surface it)
  - p_expected     : mean probability mass on the expected intermediate (soft hit rate)
  - com_corr       : corr(center-of-mass of induced dist, ideal index)  (graded monotonic routing)
  - mean_energy    : distance of induced dist to the behavior manifold  (distribution faithfulness)

Arms: manifold / linear chord / shuffled-order control / random-direction control. Best CONTROL
layer per concept = the layer maximizing manifold com_corr (vs the census's best-FIT layer). Buckets:
  hard_steer  : argmax_man > argmax_lin AND > argmax_shuf            (the strict gate)
  soft_steer  : com_corr_man >= 0.7 AND > com_corr_lin AND > com_corr_shuf  (graded order-specific)
  monitor_only: neither — clean geometry, no order-specific routing at any granularity

Run:  python scripts/_census_recovery.py
Out:  reports/manifold_census/recovery.json
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
from scripts._emotion_manifold import register, shuffled_concept, waypoint_replacements  # noqa: E402

MODEL = "mlx-community/Qwen3.5-2B-bf16"
LAYERS = [6, 8, 12, 16, 20]
N_WP = 6
READOUT = "full_string"

ORDINALS = ["priority", "age_stage", "difficulty", "certainty", "distance", "quantity",
            "wealth", "hardness", "quality", "weight", "formality"]
ANCHORS = {"emotion_arousal": 12, "emotion_fear": 16}  # gold-router / known-dead references at their layer


def arm_distributions(service, name, layer, src_i, tgt_i, path, random_seed=None):
    """Return [(ideal_float, q_array)] over n waypoints for the given arm."""
    import torch
    cobj = cp.get_concept(name)
    manifold, _, layer = service._build_manifold(name, layer)
    items = manifold["items"]
    prompt = cobj.steer_prompt.format(item=items[src_i])
    bundle = service.ensure_model()
    pos = service._locate_item_position(bundle.tokenizer, cobj.steer_prompt, items[src_i], prompt)
    src_cent = manifold["centroids_dmodel"][src_i]
    reps, _ = waypoint_replacements(service, name, layer, src_i, tgt_i, N_WP, path)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    out = []
    for ideal, rep in reps:
        vec = rep
        if rng is not None:
            disp = rep.numpy() - src_cent
            d = rng.standard_normal(len(disp))
            d = d / (np.linalg.norm(d) + 1e-9) * np.linalg.norm(disp)
            vec = torch.tensor(src_cent + d, dtype=torch.float32)
        q = np.asarray(service._value_string_distribution(prompt, layer, vec, pos, cobj), dtype=float)
        out.append((float(ideal), q))
    return out


def metrics(service, dists, beh):
    ideals = np.array([round(i) for i, _ in dists])
    ideal_f = np.array([i for i, _ in dists])
    com, argm, top2, pexp, energy = [], 0, 0, [], []
    for (i_f, q), iexp in zip(dists, ideals):
        n = len(q)
        com.append(float(np.dot(np.arange(n), q)))
        am = int(np.argmax(q))
        argm += int(am == iexp)
        order = np.argsort(q)
        top2 += int(iexp in order[-2:])
        pexp.append(float(q[iexp]))
        if beh is not None:
            energy.append(service._behavior_energy(beh, q))
    com = np.array(com)
    com_corr = float(np.corrcoef(ideal_f, com)[0, 1]) if com.std() > 1e-9 else 0.0
    return {"argmax_agree": round(argm / len(dists), 4), "top2_agree": round(top2 / len(dists), 4),
            "p_expected": round(float(np.mean(pexp)), 4), "com_corr": round(com_corr, 4),
            "mean_energy": round(float(np.mean(energy)), 5) if energy else None}


def score_concept(service, name, layers):
    c = cp.get_concept(name)
    register(c); register(shuffled_concept(c))
    sc = cp.get_concept(name + "_shuf")
    src_i, tgt_i = 0, len(c.items) - 1
    sh_src, sh_tgt = sc.items.index(c.items[src_i]), sc.items.index(c.items[tgt_i])
    per_layer = []
    for layer in layers:
        service._build_manifold(c.name, layer); service._build_behavior_manifold(c, layer, READOUT)
        beh = service._behavior_cache.get((c.name, layer, READOUT))
        man = metrics(service, arm_distributions(service, c.name, layer, src_i, tgt_i, "manifold"), beh)
        lin = metrics(service, arm_distributions(service, c.name, layer, src_i, tgt_i, "linear"), beh)
        rnd = metrics(service, arm_distributions(service, c.name, layer, src_i, tgt_i, "manifold", random_seed=29), beh)
        service._build_manifold(sc.name, layer); service._build_behavior_manifold(sc, layer, READOUT)
        beh_s = service._behavior_cache.get((sc.name, layer, READOUT))
        shuf = metrics(service, arm_distributions(service, sc.name, layer, sh_src, sh_tgt, "manifold"), beh_s)
        per_layer.append({"layer": layer, "manifold": man, "linear": lin, "shuffled": shuf, "random": rnd})
    best = max(per_layer, key=lambda L: L["manifold"]["com_corr"])
    m, l, s = best["manifold"], best["linear"], best["shuffled"]
    hard = bool(m["argmax_agree"] > l["argmax_agree"] and m["argmax_agree"] > s["argmax_agree"])
    soft = bool(m["com_corr"] >= 0.7 and m["com_corr"] > l["com_corr"] and m["com_corr"] > s["com_corr"])
    bucket = "hard_steer" if hard else ("soft_steer" if soft else "monitor_only")
    return {"concept": name, "label": c.label, "n_items": len(c.items),
            "best_control_layer": best["layer"], "fit_layer": c.best_layer,
            "manifold": m, "linear": l, "shuffled": s, "random": best["random"],
            "hard_steer": hard, "soft_steer": soft, "bucket": bucket, "per_layer": per_layer}


def main():
    t0 = time.time()
    print(f"[recover] loading {MODEL} ...", flush=True)
    service = build_mlx_service(MODEL, default_layer=12)

    results = []
    for name in ORDINALS:
        r = score_concept(service, name, LAYERS)
        m = r["manifold"]
        print(f"[recover] {name:12s} L{r['best_control_layer']:>2}(fit {r['fit_layer']:>2}) "
              f"argmax={m['argmax_agree']:.2f} com_corr={m['com_corr']:+.2f} top2={m['top2_agree']:.2f} "
              f"p_exp={m['p_expected']:.2f} -> {r['bucket']} ({time.time()-t0:.0f}s)", flush=True)
        results.append(r)

    anchors = []
    for name, layer in ANCHORS.items():
        r = score_concept(service, name, [layer])
        m = r["manifold"]
        print(f"[anchor]  {name:18s} L{layer} argmax={m['argmax_agree']:.2f} com_corr={m['com_corr']:+.2f} "
              f"-> {r['bucket']} ({time.time()-t0:.0f}s)", flush=True)
        anchors.append(r)

    by = {b: [r["concept"] for r in results if r["bucket"] == b] for b in ("hard_steer", "soft_steer", "monitor_only")}
    recovered = len(by["hard_steer"]) + len(by["soft_steer"])
    dest = ROOT / "reports" / "manifold_census"
    (dest / "recovery.json").write_text(json.dumps(
        {"model_id": MODEL, "layers_swept": LAYERS, "readout": READOUT,
         "buckets": by, "n_recovered": recovered, "n_tested": len(ORDINALS),
         "thresholds": {"hard": "argmax_man > linear AND > shuffled",
                        "soft": "com_corr_man >= 0.7 AND > linear AND > shuffled"},
         "results": results, "anchors": anchors}, indent=1))
    print(f"\n[recover] buckets: hard={by['hard_steer']} | soft={by['soft_steer']} | monitor={by['monitor_only']}")
    print(f"[recover] recovered {recovered}/{len(ORDINALS)} (was 2 hard); wrote {dest/'recovery.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
