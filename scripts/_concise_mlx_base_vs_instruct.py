#!/usr/bin/env python3
"""Did the base-SAE-on-instruct mismatch cause the v0.1 concision FALSE NEGATIVE?

One-variable controlled test. The concision steer is an SAE *feature* (#29073 @ L12), so — unlike the
probe-based truth-holding/sentiment steers — it is the one place the base-trained-SAE-on-instruct mismatch
sits in the causal path. Here we hold the steer spec FIXED (the exact `concise_l12_f29073_v1` recipe, same
base-trained SAE) and swap ONLY the model: the faithful pairing (Qwen3.5-2B-**Base**, what the SAE was trained
on) vs the original pairing (the **instruct** model). Then we run the real steering-to-data generate→filter and
compare keep-rate / collapse-rate across a strength sweep.

Reading: if concision steering produces kept (shorter, non-collapsed, content-preserving) pairs on the BASE
pairing where the instruct pairing gave ~0, the original negative was a mismatch artifact. If both collapse,
the negative stands. Local MLX only — no Tinker/Modal/network beyond the cached model weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.benchmark import ServiceGenerationBackend
from qwen_scope_lab.experiments import steering_distill as sd
from qwen_scope_lab.mlx_backend import build_mlx_service

SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"   # base-trained SAE, used in BOTH pairings
D_SAE = 32768
LAYER = 12
RECIPE = "recipes/concise_autopilot/recipe.json"        # concise_l12_f29073_v1 (feature 29073)
PROMPTS = "data/experiments/steering_distill/prompts.jsonl"
MODELS = [("base_faithful", "Qwen/Qwen3.5-2B-Base"), ("instruct_original", "mlx-community/Qwen3.5-2B-bf16")]
STRENGTHS = [4.0, 6.0, 8.0]
MAX_NEW_TOKENS = 64


def _prompts() -> list[dict]:
    return [json.loads(l) for l in Path(PROMPTS).read_text().splitlines() if l.strip()]


def run() -> dict:
    prompts = _prompts()
    base_recipe = sd.load_recipe(RECIPE)
    cfg = sd.DistillConfig(target="concise", max_length_ratio=1.0, min_content_overlap=0.5, concise_ref_tokens=80)
    params = sd.GenParams(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0, seed=0)

    results: dict = {"recipe": RECIPE, "sae": SAE_REPO, "layer": LAYER, "feature_id": 29073, "rows": [], "errors": {}}
    examples: list[dict] = []

    for label, repo in MODELS:
        try:
            service = build_mlx_service(repo, default_layer=LAYER, d_sae=D_SAE, sae_repo=SAE_REPO, top_k=64)
            backend = ServiceGenerationBackend(service)
            meta = {"model_id": repo, "sae_id": SAE_REPO, "num_layers": service.config.num_layers, "d_sae": D_SAE}
        except Exception as exc:  # honest: report a load failure, never fabricate a result
            results["errors"][label] = f"{type(exc).__name__}: {exc}"
            print(f"[{label}] MODEL LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue

        for strength in STRENGTHS:
            spec = sd.SteerSpec.from_recipe(base_recipe)
            spec.strength = float(strength)
            spec.model_id = spec.model_id or repo
            spec.sae_id = spec.sae_id or SAE_REPO
            pairs = sd.generate_pairs(spec, prompts, backend, params)
            res = sd.distill_pairs(pairs, spec, cfg, params)
            m = res["metrics"]
            # mean length ratio + mean concision delta over all scored pairs
            scored = res.get("all", [])
            lr = [r["scores"]["length_ratio"] for r in scored if r.get("scores")]
            dl = [r["scores"]["delta"] for r in scored if r.get("scores")]
            row = {"model": label, "repo": repo, "strength": strength, "n": m["n_prompts"], "n_kept": m["n_kept"],
                   "keep_rate": m["keep_rate"], "collapse_rate": m["collapse_rate"],
                   "mean_length_ratio": round(sum(lr) / len(lr), 3) if lr else None,
                   "mean_concision_delta": round(sum(dl) / len(dl), 4) if dl else None}
            results["rows"].append(row)
            print(f"[{label}] s={strength}: n={row['n']} kept={row['n_kept']} keep_rate={row['keep_rate']} "
                  f"collapse_rate={row['collapse_rate']} len_ratio={row['mean_length_ratio']} "
                  f"Δconcision={row['mean_concision_delta']}", flush=True)
            # grab 2 example steered outputs at the recipe strength for eyeballing
            if strength == 4.0:
                for r in scored[:2]:
                    examples.append({"model": label, "prompt": r.get("prompt", "")[:80],
                                     "unsteered": (r.get("unsteered", "") or "")[:160],
                                     "steered": (r.get("steered", "") or "")[:160]})
        del service, backend  # free the model before loading the next

    results["examples"] = examples
    out = Path("reports/sae_audit/concise_base_vs_instruct.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("\nWROTE", out, flush=True)
    return results


if __name__ == "__main__":
    run()
