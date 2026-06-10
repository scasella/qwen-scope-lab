"""C09 live generation (MLX, on-device) — compile REAL manifold steering into geometry-gated data.

For each ordinal concept, sweep every source value -> the top value along the residual manifold,
across the concept's carrier templates, calling the real 2B via manifold_compare (with the C05
behavior_readout: full_string for multi-token concepts). Compile each per-prompt payload through the
torch-free manifold_distill gate and split into three equal-size arms:

  gated_manifold   — manifold waypoints kept by the geometry gate (energy <= linear chord)
  ungated_manifold — ALL manifold waypoints (text-quality only) — isolates whether the GATE matters
  linear           — linear-chord waypoints (the baseline)

Writes, per concept, mlx_lm-ready {train,valid,test}.jsonl per arm (carrier-template held-out split)
plus the full provenance/metrics. No external services, no API keys. Scratch script (underscored).

    python3 scripts/_c09_mlx_generate.py --concept rank      --readout first_token
    python3 scripts/_c09_mlx_generate.py --concept education --readout full_string
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_scope_lab.concept_presets import get_concept
from qwen_scope_lab.experiments import manifold_distill as md
from qwen_scope_lab.mlx_backend import build_mlx_service
from qwen_scope_lab.experiments.steering_distill import is_collapsed, is_empty

DEFAULT_MLX_MODEL = "mlx-community/Qwen3.5-2B-bf16"


def _clean_sft(records):
    """mlx_lm chat format: {"messages":[user,assistant]} with non-empty, non-collapsed completion."""
    out = []
    for r in records:
        text = r["steered_text"]
        if is_empty(text) or is_collapsed(text)[0]:
            continue
        out.append({"messages": [{"role": "user", "content": r["prompt"]},
                                 {"role": "assistant", "content": text}]})
    return out


def _split(records_by_template, train_t, valid_t, test_t):
    tr = [r for t in train_t for r in records_by_template.get(t, [])]
    va = [r for t in valid_t for r in records_by_template.get(t, [])]
    te = [r for t in test_t for r in records_by_template.get(t, [])]
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept", default="rank")
    ap.add_argument("--readout", choices=["first_token", "full_string"], default="first_token")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--n-waypoints", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--min-recovered-r", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    concept = get_concept(args.concept)
    layer = args.layer if args.layer is not None else (concept.best_layer or 8)
    target = concept.items[-1]
    sources = list(concept.items[:-1])
    templates = list(concept.templates)
    out_root = Path(args.out or f"reports/manifold_c09/{args.concept}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[c09] loading MLX {DEFAULT_MLX_MODEL} (no SAE) …", flush=True)
    service = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)

    gate = md.GateConfig(min_recovered_r=args.min_recovered_r)
    spec = md.ManifoldDataSpec.explicit(concept=args.concept, source=sources[0], target=target,
                                        layer=layer, n_waypoints=args.n_waypoints, behavior_readout=args.readout)
    spec.source_kind = "live_mlx"

    # arm -> template_index -> list[clean sft record]; plus raw graded records for provenance
    arms = {"gated_manifold": {}, "ungated_manifold": {}, "linear": {}}
    all_graded = []
    per_prompt_gap = []
    for ti, tmpl in enumerate(templates):
        for src in sources:
            prompt = tmpl.format(item=src)
            try:
                payload = service.manifold_compare(args.concept, target, layer=layer, source=src, prompt=prompt,
                                                   n_waypoints=args.n_waypoints, max_new_tokens=args.max_new_tokens,
                                                   behavior_readout=args.readout)
            except Exception as exc:  # noqa: BLE001
                print(f"[c09]   skip {prompt!r}: {exc!r}", flush=True)
                continue
            graded = md.compile_payload(payload, spec, gate)
            all_graded.extend(graded["all"])
            m_e = payload["manifold"]["mean_energy"]
            l_e = payload["linear"]["mean_energy"]
            if m_e is not None and l_e is not None:
                per_prompt_gap.append(l_e - m_e)
            man = [r for r in graded["all"] if r["path"] == "manifold"]
            lin = [r for r in graded["all"] if r["path"] == "linear"]
            arms["gated_manifold"].setdefault(ti, []).extend(_clean_sft([r for r in man if r["keep"]]))
            arms["ungated_manifold"].setdefault(ti, []).extend(_clean_sft(man))
            arms["linear"].setdefault(ti, []).extend(_clean_sft(lin))
        print(f"[c09] template {ti} '{tmpl}' done", flush=True)

    # carrier-template held-out split: train on 0..n-3, valid n-2, test n-1 (held-out carriers)
    n_t = len(templates)
    train_t, valid_t, test_t = list(range(n_t - 2)), [n_t - 2], [n_t - 1]
    summary = {"concept": args.concept, "layer": layer, "readout": args.readout, "target": target,
               "n_sources": len(sources), "n_templates": n_t,
               "mean_energy_gap_manifold_minus_linear": round(sum(per_prompt_gap) / len(per_prompt_gap), 4) if per_prompt_gap else None,
               "n_prompts_with_manifold_better": sum(1 for g in per_prompt_gap if g > 0),
               "n_prompts_total": len(per_prompt_gap), "arms": {}}
    # equal-size across arms (cap train to the smallest arm's train count)
    train_counts = {a: sum(len(v) for t, v in by_t.items() if t in train_t) for a, by_t in arms.items()}
    cap = min(train_counts.values()) if train_counts else 0
    for arm, by_t in arms.items():
        tr, va, te = _split(by_t, train_t, valid_t, test_t)
        tr = tr[:cap]  # equal-size training across arms
        d = out_root / arm
        d.mkdir(parents=True, exist_ok=True)
        for name, rows in (("train", tr), ("valid", va or te), ("test", te or va)):
            with (d / f"{name}.jsonl").open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary["arms"][arm] = {"train": len(tr), "valid": len(va), "test": len(te), "dir": str(d)}

    (out_root / "pairs_all.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_graded), encoding="utf-8")
    (out_root / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
