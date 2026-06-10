"""C05 — real-2B (MLX, on-device) behavior_readout audit: first_token vs full_string.

Builds an MLX-backed SteeringService (no SAE needed — energy/isometry are SAE-free) and runs
manifold_compare twice per concept, reporting whether the manifold-more-faithful verdict FLIPS
between the first-token and full-string behavior read-outs, with isometry_r and tokenization
metadata. Scratch/diagnostic script (underscore-prefixed); run locally on Apple Silicon:

    python3 scripts/_c05_mlx_audit.py            # default 2B instruct pairing
    python3 scripts/_c05_mlx_audit.py --base     # the base model the SAE was trained on
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root for serve_web

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.mlx_backend import build_mlx_service

# Mirror serve_web's MLX pairing (kept in sync; this scratch script avoids importing the root entrypoint).
DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"
BASE_MLX_MODEL = "Qwen/Qwen3.5-2B-Base"

# (concept, layer, source, target, risk)
SPECS = [
    ("agreement", 8, "strongly disagree", "strongly agree", "multi_token"),
    ("education", 8, "kindergarten", "doctorate", "multi_token"),
    ("rank", 20, "private", "general", "control"),
    ("valence", 16, "miserable", "ecstatic", "control"),
    ("size", 16, "tiny", "enormous", "control"),
    ("days_of_week", 14, "Monday", "Thursday", "control"),
]


def _cumdist(pts, metric):
    n = len(pts)
    seg = [metric(pts[k], pts[k + 1]) for k in range(n - 1)]
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.abs(cum[:, None] - cum[None, :])


def _isometry(service, concept, layer, readout):
    eucl = lambda a, b: float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    hell = lambda a, b: float(np.linalg.norm(np.sqrt(np.clip(a, 0, None)) - np.sqrt(np.clip(b, 0, None))) / np.sqrt(2))
    mk = service._manifold_cache.get((concept, layer))
    bk = service._behavior_cache.get((concept, layer, readout))
    if mk is None or bk is None:
        return None
    d_h = _cumdist(np.asarray(mk["centroids_pca"]), eucl)
    d_y = _cumdist(np.asarray(bk["centroids"]), hell)
    iu = np.triu_indices(d_h.shape[0], 1)
    a, bb = d_h[iu], d_y[iu]
    return float(np.corrcoef(a, bb)[0, 1]) if (a.std() > 0 and bb.std() > 0) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true", help="use the base model (--mlx-base pairing)")
    ap.add_argument("--n-waypoints", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    repo = BASE_MLX_MODEL if args.base else DEFAULT_MLX_MODEL
    print(f"[c05] loading MLX model {repo} (no SAE) …", flush=True)
    service = build_mlx_service(repo, default_layer=12)

    rows, n_flips = [], 0
    for concept, layer, source, target, risk in SPECS:
        print(f"[c05] {concept} (layer {layer}, {risk}) …", flush=True)
        try:
            cobj = get_concept(concept)
            first_ids = service._concept_token_ids(cobj)
            conts = service._value_continuation_ids(cobj)
            row = {"concept": concept, "layer": layer, "risk": risk,
                   "tokenization": {
                       "first_token_collisions": len(first_ids) - len(set(first_ids)),
                       "mean_tokens_per_value": round(float(np.mean([len(c) for c in conts])), 3),
                       "multi_token": bool(any(len(c) > 1 for c in conts))}}
            verdicts = {}
            for readout in ("first_token", "full_string"):
                c = service.manifold_compare(concept, target, layer=layer, source=source,
                                             n_waypoints=args.n_waypoints, max_new_tokens=args.max_new_tokens,
                                             behavior_readout=readout)
                m_e, l_e = c["manifold"]["mean_energy"], c["linear"]["mean_energy"]
                gap = round(l_e - m_e, 4) if (m_e is not None and l_e is not None) else None
                mf = bool(m_e is not None and l_e is not None and m_e < l_e)
                verdicts[readout] = mf
                iso = _isometry(service, concept, layer, readout)
                row[readout] = {"manifold_energy": m_e, "linear_energy": l_e, "energy_gap": gap,
                                "manifold_more_faithful": mf,
                                "isometry_r": round(iso, 4) if iso is not None else None}
            row["verdict_flip"] = bool(verdicts["first_token"] != verdicts["full_string"])
            n_flips += int(row["verdict_flip"])
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append({"concept": concept, "error": repr(exc)})
            print(f"[c05]   ERROR {concept}: {exc!r}", flush=True)

    out = {"model_id": repo, "audit": "behavior_readout first_token vs full_string (C05)",
           "n_verdict_flips": n_flips, "results": rows}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
