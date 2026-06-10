"""Combine the 2B (MLX) and 4B (Tinker) eval arms into one scored report for the distillation experiment.

Reads the per-arm output JSON files (each: arm_name -> [{id, prompt, output, metadata}]), scores every
arm with the same target scorer, and writes a metrics JSON + a Markdown report with the key comparisons:
  - distilled vs its own base (the distillation effect, learned into weights — no hooks)
  - runtime steer vs its base (the source behavior the data was distilled from)
  - distilled vs runtime steer (did the distilled model reach the steer's effect?)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_scope_lab.experiments import steering_distill as sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="One or more arm JSON files to merge")
    ap.add_argument("--target", default="sentiment")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms: dict[str, list] = {}
    for path in args.arms:
        arms.update(json.loads(Path(path).read_text()))
    cfg = sd.DistillConfig(target=args.target)
    ev = sd.evaluate(arms, cfg)

    def m(name: str) -> float | None:
        return ev["arms"].get(name, {}).get("mean_target_score")

    def delta(a: str, b: str) -> str:
        va, vb = m(a), m(b)
        return f"{round(va - vb, 4):+}" if (va is not None and vb is not None) else "—"

    comparisons = []
    if m("distilled_4b") is not None and m("baseline_4b") is not None:
        comparisons.append(("distilled_4b − baseline_4b", "**distillation effect** (learned into weights, no hooks)", delta("distilled_4b", "baseline_4b")))
    if m("runtime_steer_2b") is not None and m("baseline_2b") is not None:
        comparisons.append(("runtime_steer_2b − baseline_2b", "source steer effect (runtime hook on the 2B)", delta("runtime_steer_2b", "baseline_2b")))
    if m("distilled_4b") is not None and m("runtime_steer_2b") is not None:
        comparisons.append(("distilled_4b − runtime_steer_2b", "did the distilled model reach the steer's level?", delta("distilled_4b", "runtime_steer_2b")))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_metrics.json").write_text(json.dumps(ev, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    arm_rows = "\n".join(
        f"| `{name}` | {v['n']} | {v['mean_target_score']} | {v['collapse_rate']:.0%} | {v['mean_tokens']} |"
        for name, v in ev["arms"].items()
    )
    comp_rows = "\n".join(f"| `{a}` | {desc} | **{d}** |" for a, desc, d in comparisons) or "| — | — | — |"
    samples = []
    for name in ev["arms"]:
        ex = arms[name][0]
        snippet = " ".join((ex["output"] or "").split())[:220]
        samples.append(f"- **`{name}`** · _{ex['prompt']}_\n  - {snippet!r}")
    body = (
        f"# Steering-to-data distillation — eval (`{args.target}`)\n\n"
        f"Higher sentiment score = more positive tone (0.5 ≈ neutral). All arms scored by the same lexicon proxy.\n\n"
        f"## Arms\n\n| arm | n | mean sentiment | collapse | mean tokens |\n|---|---|---|---|---|\n{arm_rows}\n\n"
        f"## Key comparisons\n\n| comparison | meaning | Δ |\n|---|---|---|\n{comp_rows}\n\n"
        f"Ranking (best tone first): {' > '.join('`'+a+'`' for a in ev['ranking'])}\n\n"
        f"## Sample outputs (first eval prompt)\n\n" + "\n".join(samples) + "\n\n"
        f"_Generated {ev['generated_at']} · schema {ev['schema_version']}._\n"
    )
    (out / "eval_report.md").write_text(body, encoding="utf-8")
    print(json.dumps({"out": str(out), "arms": {k: v["mean_target_score"] for k, v in ev["arms"].items()},
                      "comparisons": {a: d for a, _, d in comparisons}}, indent=2))


if __name__ == "__main__":
    main()
