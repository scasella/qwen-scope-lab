"""C09 salvage #2 — does an earlier injection layer make manifold-steered TEXT move?

The live C09 result: at each concept's geometry-optimal layer, greedy text is invariant along the
source->target manifold sweep (distinct texts/sweep ~1.0), so text-SFT can't carry the provenance
signal. This sweeps the injection layer and measures, per layer, how much the steered TEXT actually
varies (distinct texts per sweep) and how often manifold != linear text. If some layer makes text
move materially, text-SFT becomes viable there. Generation-only (no energy) for speed. Local MLX.

    python3 scripts/_c09_layer_sweep.py
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.mlx_backend import build_mlx_service

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", nargs="*", default=["rank", "education"])
    ap.add_argument("--layers", nargs="*", type=int, default=[4, 8, 12, 16, 20, 24])
    ap.add_argument("--n-waypoints", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--out", default="reports/manifold_c09/layer_sweep.json")
    ap.add_argument("--model", default=DEFAULT_MLX_MODEL, help="MLX repo or HF repo (mlx_lm converts on load)")
    args = ap.parse_args()

    print(f"[sweep] loading MLX {args.model} …", flush=True)
    service = build_mlx_service(args.model, default_layer=12)
    num_layers = service.bundle.model.num_layers
    results = {}
    for cname in args.concepts:
        c = get_concept(cname)
        target = c.items[-1]
        carriers = [(t, s) for t in c.templates[:2] for s in c.items[:3]]  # 2 templates x 3 sources
        results[cname] = {}
        for layer in args.layers:
            if layer >= num_layers:
                continue
            distinct, man_ne_lin, n = [], 0, 0
            for tmpl, src in carriers:
                prompt = tmpl.format(item=src)
                try:
                    man = service.manifold_steer(cname, target, layer=layer, source=src, prompt=prompt,
                                                 n_waypoints=args.n_waypoints, max_new_tokens=args.max_new_tokens,
                                                 path="manifold", compute_unsteered=False, compute_energy=False)
                    lin = service.manifold_steer(cname, target, layer=layer, source=src, prompt=prompt,
                                                 n_waypoints=args.n_waypoints, max_new_tokens=args.max_new_tokens,
                                                 path="linear", compute_unsteered=False, compute_energy=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"[sweep]   skip {cname} L{layer} {prompt!r}: {exc!r}", flush=True)
                    continue
                man_texts = [w["text"] for w in man["waypoints"]]
                lin_texts = [w["text"] for w in lin["waypoints"]]
                distinct.append(len(set(man_texts)))
                man_ne_lin += int(set(man_texts) != set(lin_texts))
                n += 1
            if n:
                results[cname][layer] = {
                    "n_carriers": n,
                    "mean_distinct_texts_per_sweep": round(sum(distinct) / n, 3),
                    "frac_manifold_text_differs_from_linear": round(man_ne_lin / n, 3),
                }
                r = results[cname][layer]
                print(f"[sweep] {cname:10} L{layer:<2} distinct/sweep={r['mean_distinct_texts_per_sweep']:<5} "
                      f"man!=lin={r['frac_manifold_text_differs_from_linear']}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
