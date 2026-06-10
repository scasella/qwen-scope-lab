"""Census: which NET-NEW concepts form clean geometric manifolds in Qwen3.5-2B?

Follows the centroid-routing line. For a broad batch of candidate ordered/cyclic concepts
(none already in the registry), fit the residual-stream manifold at each of several layers,
and score it two ways:

  - order recovery     : abs_spearman (ordinal) / ring_adjacency (cyclic) from `_manifold_quality`
  - activation↔behavior: isometry r (full_string read-out), the paper's metric

A concept is CLEAN if, at its best layer, both order-recovery >= 0.9 AND isometry r >= 0.9;
PARTIAL if order-recovery >= 0.75 or isometry r >= 0.85; else DIFFUSE. This only needs the
*fit* (geometry), so multi-token value phrases are fine — the C05 read-out caveat affects
steering faithfulness, not manifold fitting.

Run:  python scripts/_manifold_census.py
Out:  reports/manifold_census/census.json  (+ a ranked table to stdout)
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

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"
LAYERS = [6, 8, 12, 16, 20]
READOUT = "full_string"

_G = ("It was {item}", "The level was {item}", "Rated {item}", "Quite {item}", "Remarkably {item}", "Marked {item}")

# ---- net-new candidates (none already in CONCEPTS / _ATLAS_EXTRA) ----
CANDIDATES = [
    # physical / perceptual scales
    Concept("weight", "Weight", "ordinal",
        ("weightless", "light", "medium", "heavy", "massive"), _G, "It feels {item}"),
    Concept("hardness", "Hardness", "ordinal",
        ("mushy", "soft", "firm", "hard", "rigid"), _G, "The material is {item}"),
    Concept("wetness", "Wetness", "ordinal",
        ("bone-dry", "dry", "damp", "wet", "soaked"), _G, "The ground is {item}"),
    Concept("spiciness", "Spiciness", "ordinal",
        ("bland", "mild", "medium", "spicy", "fiery"), _G, "The dish is {item}"),
    Concept("distance", "Distance", "ordinal",
        ("adjacent", "near", "moderate", "far", "remote"), _G, "The place is {item}"),
    Concept("pain", "Pain level", "ordinal",
        ("painless", "mild", "uncomfortable", "painful", "agonizing"), _G, "The feeling is {item}"),
    # abstract / social scales
    Concept("wealth", "Wealth", "ordinal",
        ("destitute", "poor", "modest", "comfortable", "affluent", "wealthy"), _G, "They are {item}"),
    Concept("certainty", "Certainty", "ordinal",
        ("impossible", "doubtful", "possible", "probable", "certain"), _G, "The outcome is {item}"),
    Concept("difficulty", "Difficulty", "ordinal",
        ("trivial", "easy", "moderate", "hard", "impossible"), _G, "The task is {item}"),
    Concept("priority", "Priority", "ordinal",
        ("trivial", "low", "medium", "high", "critical"), _G, "The priority is {item}"),
    Concept("formality", "Formality", "ordinal",
        ("casual", "informal", "neutral", "formal", "ceremonial"), _G, "The tone is {item}"),
    Concept("politeness", "Politeness", "ordinal",
        ("rude", "blunt", "neutral", "polite", "deferential"), _G, "The reply was {item}"),
    Concept("quality", "Quality", "ordinal",
        ("terrible", "poor", "average", "good", "excellent"), _G, "The work is {item}"),
    Concept("quantity", "Quantity", "ordinal",
        ("none", "few", "some", "many", "most", "all"), _G, "The amount is {item}"),
    # named ordered series
    Concept("age_stage", "Life stage", "ordinal",
        ("infant", "child", "teenager", "adult", "elderly"),
        ("They are a {item}", "A {item} person", "The {item} years", "As a {item}", "Being a {item}"), "They are a {item}"),
    Concept("planets", "Planets by distance", "ordinal",
        ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"),
        ("The planet is {item}", "Orbiting {item}", "A probe to {item}", "Beyond {item}", "Near {item}"), "The next planet after {item} is"),
    Concept("belt_rank", "Martial-arts belts", "ordinal",
        ("white", "yellow", "orange", "green", "blue", "brown", "black"),
        ("A {item} belt", "Earned the {item} belt", "Wearing {item}", "Promoted to {item}", "The {item} rank"), "The belt is {item}"),
    # cyclic
    Concept("seasons", "Seasons", "cyclic",
        ("winter", "spring", "summer", "autumn"),
        ("The season is {item}", "It was {item}", "During {item}", "A {item} day", "By {item}"), "The season is {item}"),
    Concept("time_of_day", "Time of day", "cyclic",
        ("dawn", "morning", "noon", "afternoon", "evening", "night"),
        ("It was {item}", "By {item}", "Every {item}", "Around {item}", "Late {item}"), "The time is {item}"),
    Concept("moon_phases", "Moon phases", "cyclic",
        ("new", "crescent", "quarter", "gibbous", "full"),
        ("The moon was {item}", "A {item} moon", "During the {item} moon", "By the {item} moon", "Under a {item} moon"), "The moon is {item}"),
]


def isometry_r(service, name, layer):
    eucl = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    hell = lambda a, b: float(np.linalg.norm(np.sqrt(np.clip(a, 0, None)) - np.sqrt(np.clip(b, 0, None))) / np.sqrt(2))
    mk = service._manifold_cache.get((name, layer))
    bk = service._behavior_cache.get((name, layer, READOUT))
    if mk is None or bk is None:
        return None
    def cumdist(pts, metric):
        n = len(pts)
        seg = [metric(pts[k], pts[k + 1]) for k in range(n - 1)]
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        return np.abs(cum[:, None] - cum[None, :])
    d_h = cumdist(np.asarray(mk["centroids_pca"]), eucl)
    d_y = cumdist(np.asarray(bk["centroids"]), hell)
    iu = np.triu_indices(d_h.shape[0], 1)
    a, b = d_h[iu], d_y[iu]
    return float(np.corrcoef(a, b)[0, 1]) if (a.std() > 0 and b.std() > 0) else None


def classify(order, iso):
    if order is None or iso is None:
        return "error"
    if order >= 0.9 and iso >= 0.9:
        return "clean"
    if order >= 0.75 or iso >= 0.85:
        return "partial"
    return "diffuse"


def main():
    t0 = time.time()
    print(f"[census] loading {DEFAULT_MLX_MODEL} (no SAE) ...", flush=True)
    service = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)

    results = []
    for c in CANDIDATES:
        cp._ALL[c.name] = c  # register so service can fetch it
        sweep = []
        for layer in LAYERS:
            try:
                service._build_manifold(c.name, layer)
                service._build_behavior_manifold(c, layer, READOUT)
                q = service._manifold_quality(service._manifold_cache[(c.name, layer)])
                iso = isometry_r(service, c.name, layer)
                sweep.append({"layer": layer, "order_metric_name": q["metric_name"],
                              "order": q["metric"], "iso_r": round(iso, 4) if iso is not None else None})
            except Exception as e:  # noqa: BLE001
                sweep.append({"layer": layer, "error": str(e)[:120]})
        ok = [s for s in sweep if "order" in s and s["order"] is not None]
        best = max(ok, key=lambda s: s["order"]) if ok else None
        verdict = classify(best["order"], best["iso_r"]) if best else "error"
        results.append({"concept": c.name, "label": c.label, "kind": c.kind, "n_items": len(c.items),
                        "best_layer": best["layer"] if best else None,
                        "order_metric_name": best["order_metric_name"] if best else None,
                        "order": best["order"] if best else None,
                        "iso_r": best["iso_r"] if best else None,
                        "verdict": verdict, "sweep": sweep})
        b = best or {}
        print(f"[census] {c.name:16s} {c.kind:7s} -> {verdict:8s} "
              f"L{b.get('layer','?'):>2} order={b.get('order','?')} iso={b.get('iso_r','?')} ({time.time()-t0:.0f}s)", flush=True)

    order_rank = {"clean": 0, "partial": 1, "diffuse": 2, "error": 3}
    results.sort(key=lambda r: (order_rank[r["verdict"]], -(r["order"] or 0)))
    n_clean = sum(r["verdict"] == "clean" for r in results)
    n_partial = sum(r["verdict"] == "partial" for r in results)

    dest = ROOT / "reports" / "manifold_census"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "census.json").write_text(json.dumps(
        {"model_id": DEFAULT_MLX_MODEL, "layers_swept": LAYERS, "readout": READOUT,
         "n_candidates": len(CANDIDATES), "n_clean": n_clean, "n_partial": n_partial,
         "thresholds": {"clean": "order>=0.9 AND iso>=0.9", "partial": "order>=0.75 OR iso>=0.85"},
         "results": results}, indent=1))

    print(f"\n[census] {n_clean} clean, {n_partial} partial, "
          f"{len(CANDIDATES)-n_clean-n_partial} diffuse/error of {len(CANDIDATES)} new candidates")
    print(f"[census] wrote {dest/'census.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
