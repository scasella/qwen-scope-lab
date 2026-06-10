"""C09 held-out transfer eval (MLX, on-device).

The C09 question made concrete: after training on geometry-gated manifold data, does the model —
*with no hook* — lean toward the target concept value on HELD-OUT carrier templates? For the base
model and each arm's LoRA adapter, this reads the model's distribution over the ordered concept
values (the same value-distribution machinery the behavior-energy uses) on held-out carriers and
reports target-value mass, normalized expected position (0=source end, 1=target end), and target
rank. The arm whose adapter shifts position/mass toward the target most has transferred best.

    python3 scripts/_c09_mlx_eval.py --concept rank --readout first_token \
        --out reports/manifold_c09/rank --eval-template-index 4
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.mlx_backend import build_mlx_service

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"


def _distribution(service, prompt, layer, concept, readout):
    if readout == "full_string":
        return np.asarray(service._value_string_distribution(prompt, layer, None, 0, concept), dtype=float)
    token_ids = service._concept_token_ids(concept)
    return np.asarray(service._output_distribution(prompt, layer, None, 0, token_ids), dtype=float)


def _eval_one(service, concept, layer, readout, eval_prompts):
    n = len(concept.items)
    tgt = n - 1
    masses, positions, ranks = [], [], []
    for prompt in eval_prompts:
        p = _distribution(service, prompt, layer, concept, readout)
        if p.sum() <= 0:
            continue
        p = p / p.sum()
        masses.append(float(p[tgt]))
        positions.append(float(np.dot(np.arange(n), p) / (n - 1)))   # 0=source end .. 1=target end
        ranks.append(1 + int(np.sum(p > p[tgt])))                    # 1 = target is the argmax
    return {
        "n_eval": len(masses),
        "target_mass": round(float(np.mean(masses)), 4) if masses else None,
        "expected_position": round(float(np.mean(positions)), 4) if positions else None,
        "target_rank": round(float(np.mean(ranks)), 3) if ranks else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="rank")
    ap.add_argument("--readout", choices=["first_token", "full_string"], default="first_token")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--eval-template-index", type=int, default=4, help="held-out carrier template")
    ap.add_argument("--arms", nargs="*", default=["gated_manifold", "ungated_manifold", "linear"])
    args = ap.parse_args()

    concept = get_concept(args.concept)
    layer = args.layer if args.layer is not None else (concept.best_layer or 8)
    out_root = Path(args.out or f"reports/manifold_c09/{args.concept}")
    tmpl = concept.templates[args.eval_template_index]
    # held-out carriers: the held-out template applied to every source value (hook-free transfer probe)
    eval_prompts = [tmpl.format(item=s) for s in concept.items[:-1]]
    print(f"[eval] concept={args.concept} layer={layer} readout={args.readout} "
          f"held-out template={tmpl!r} ({len(eval_prompts)} prompts)", flush=True)

    results = {}
    # base (no adapter)
    print("[eval] base (no adapter) …", flush=True)
    base_svc = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)
    results["base"] = _eval_one(base_svc, concept, layer, args.readout, eval_prompts)
    del base_svc

    for arm in args.arms:
        adapter = out_root / arm / "adapter"
        if not (adapter / "adapters.safetensors").exists():
            print(f"[eval] {arm}: no adapter at {adapter} — skipping", flush=True)
            results[arm] = {"error": "no_adapter"}
            continue
        print(f"[eval] {arm} adapter …", flush=True)
        svc = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12, adapter_path=str(adapter))
        results[arm] = _eval_one(svc, concept, layer, args.readout, eval_prompts)
        del svc

    base = results.get("base", {})
    out = {"concept": args.concept, "layer": layer, "readout": args.readout,
           "held_out_template": tmpl, "target": concept.items[-1], "arms": results,
           "delta_expected_position_vs_base": {
               a: (round(results[a]["expected_position"] - base["expected_position"], 4)
                   if results.get(a, {}).get("expected_position") is not None and base.get("expected_position") is not None else None)
               for a in args.arms}}
    (out_root / "eval_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
