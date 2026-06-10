"""C05 follow-up — the FULL MANIFOLD.md §5 manifold-vs-linear table under BOTH read-outs.

Re-runs every well-defined §5 census concept (integers degenerate->skipped) with
manifold_compare under first_token AND full_string read-outs, same layer/waypoints/params
for both, for a self-consistent corrected energy ledger. Records per concept and read-out:
layer, manifold_energy, linear_energy, energy_gap, manifold_more_faithful, isometry_r,
plus tokenization stats and verdict_flip. Optional emotion section (single-token; bit-identical).

    python3 scripts/_c05_full_table.py            # default 2B instruct pairing
    python3 scripts/_c05_full_table.py --emotion  # also run the 3 emotion concepts
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.mlx_backend import build_mlx_service

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"
BASE_MLX_MODEL = "Qwen/Qwen3.5-2B-Base"

# (concept, layer, source, target, note)  — layers = each concept's preset best_layer,
# matching the C05 audit SPECS and §5 isometry-row layers where discoverable.
# integers_0_20 is the §5 degenerate concept (->NA) and is intentionally skipped.
SPECS = [
    ("days_of_week", 14, "Monday", "Thursday", "census"),
    ("rank", 20, "private", "general", "census"),
    ("size", 16, "tiny", "enormous", "census"),
    ("agreement", 8, "strongly disagree", "strongly agree", "census"),
    ("education", 8, "kindergarten", "doctorate", "census"),
    ("valence", 16, "miserable", "ecstatic", "census"),
]

EMOTION_SPECS = [
    ("emotion_arousal", 12, "numb", "frantic", "emotion"),
    ("emotion_valence_intensity", 8, "furious", "euphoric", "emotion"),
    ("emotion_fear", 8, "terrified", "calm", "emotion"),
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


def run_specs(service, specs, n_waypoints, max_new_tokens):
    rows = []
    for concept, layer, source, target, note in specs:
        print(f"[c05-full] {concept} (layer {layer}, {note}) …", flush=True)
        try:
            cobj = get_concept(concept)
            first_ids = service._concept_token_ids(cobj)
            conts = service._value_continuation_ids(cobj)
            row = {"concept": concept, "layer": layer, "note": note,
                   "source": source, "target": target,
                   "tokenization": {
                       "first_token_collisions": len(first_ids) - len(set(first_ids)),
                       "mean_tokens_per_value": round(float(np.mean([len(c) for c in conts])), 3),
                       "multi_token": bool(any(len(c) > 1 for c in conts))}}
            verdicts = {}
            for readout in ("first_token", "full_string"):
                c = service.manifold_compare(concept, target, layer=layer, source=source,
                                             n_waypoints=n_waypoints, max_new_tokens=max_new_tokens,
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
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append({"concept": concept, "error": repr(exc)})
            print(f"[c05-full]   ERROR {concept}: {exc!r}", flush=True)
    return rows


def tally(rows, key):
    wins = [r["concept"] for r in rows if "error" not in r and r[key]["manifold_more_faithful"]]
    return wins, len([r for r in rows if "error" not in r])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", action="store_true")
    ap.add_argument("--emotion", action="store_true", help="also run the 3 emotion concepts")
    ap.add_argument("--n-waypoints", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=4)
    args = ap.parse_args()

    repo = BASE_MLX_MODEL if args.base else DEFAULT_MLX_MODEL
    print(f"[c05-full] loading MLX model {repo} (no SAE) …", flush=True)
    service = build_mlx_service(repo, default_layer=12)

    census = run_specs(service, SPECS, args.n_waypoints, args.max_new_tokens)
    emotion = run_specs(service, EMOTION_SPECS, args.n_waypoints, args.max_new_tokens) if args.emotion else []

    ft_wins, n = tally(census, "first_token")
    fs_wins, _ = tally(census, "full_string")
    n_flips = sum(int(r.get("verdict_flip", False)) for r in census)

    out = {
        "model_id": repo,
        "audit": "C05 follow-up — full §5 manifold-vs-linear table, both read-outs",
        "params": {"n_waypoints": args.n_waypoints, "max_new_tokens": args.max_new_tokens,
                   "temperature": 0.0, "skipped": ["integers_0_20 (degenerate -> NA)"]},
        "census": {
            "n_concepts": n,
            "first_token": {"manifold_wins": len(ft_wins), "concepts": sorted(ft_wins)},
            "full_string": {"manifold_wins": len(fs_wins), "concepts": sorted(fs_wins)},
            "n_verdict_flips": n_flips,
            "results": census,
        },
        "emotion": {"results": emotion} if args.emotion else None,
    }

    os.makedirs("reports/manifold_c05", exist_ok=True)
    path = "reports/manifold_c05/full_table_full_string.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[c05-full] wrote {path}", flush=True)
    print(f"[c05-full] first_token: manifold beats linear {len(ft_wins)}/{n}: {sorted(ft_wins)}", flush=True)
    print(f"[c05-full] full_string: manifold beats linear {len(fs_wins)}/{n}: {sorted(fs_wins)}", flush=True)
    return out


if __name__ == "__main__":
    main()
