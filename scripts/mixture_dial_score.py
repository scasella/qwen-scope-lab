#!/usr/bin/env python3
"""Post-hoc, class-sliced dual-target scorer for the Mixture Dial validation pilot.

The paid step (mixture_dial_validate.py --execute) trains a LoRA per arm and writes the
sampled held-out outputs to <run>/arms/<arm>/<seed>/eval_arms.json (arm_name -> [{id, prompt,
output, metadata}]). This scorer reads those, and for the DISTILLED arm of each run scores:
  - truth-holding  = mean `deference`  over A_factual scenarios
  - calibration    = mean `calibrated` over B_unknowable + C_subjective scenarios
then aggregates across seeds and prints the arm comparison + the pre-registered kill checks
(K3 vs truth_only, K2 vs naive_stratified). Scoring is offline/lexicon-based — no API, no cost.

Usage:
    python3 scripts/mixture_dial_score.py /tmp/mixture_pilot_run
    python3 scripts/mixture_dial_score.py /tmp/mixture_pilot_run --distilled-key distilled_4b
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

from qwen_scope_lab.experiments import steering_distill as sd

CALIB_CLASSES = {"B_unknowable", "C_subjective"}


def _score(rows: list[dict], target_name: str, classes: set[str] | None) -> float | None:
    cfg = sd.DistillConfig(target=target_name)
    target = sd.get_target(target_name, cfg)
    vals = []
    for r in rows:
        meta = r.get("metadata", {}) or {}
        if classes is not None and meta.get("behavioral_class") not in classes:
            continue
        vals.append(float(target.score(r.get("output", ""), meta)))
    return round(statistics.mean(vals), 4) if vals else None


def score_arm_seed(eval_arms_path: Path, distilled_key: str | None) -> dict | None:
    data = json.loads(eval_arms_path.read_text())
    key = distilled_key
    if key is None:
        key = next((k for k in data if "distill" in k.lower()), None)
    if key is None or key not in data:
        return None
    rows = data[key]
    return {
        "n": len(rows),
        "deference_A": _score(rows, "deference", {"A_factual"}),
        "calibrated_BC": _score(rows, "calibrated", CALIB_CLASSES),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--distilled-key", default=None, help="arm key in eval_arms.json (default: first containing 'distill')")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(Path(args.run_dir) / "arms" / "*" / "*" / "eval_arms.json")))
    if not paths:
        print(f"no eval_arms.json under {args.run_dir}/arms/*/*/ — has the --execute run finished?")
        return

    # arm -> list of per-seed metric dicts
    per_arm: dict[str, list[dict]] = {}
    for p in paths:
        arm = Path(p).parts[-3]
        m = score_arm_seed(Path(p), args.distilled_key)
        if m:
            per_arm.setdefault(arm, []).append(m)

    def agg(arm: str, field: str) -> float | None:
        vals = [d[field] for d in per_arm.get(arm, []) if d.get(field) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    arms = sorted(per_arm)
    print("\n=== Mixture Dial pilot — distilled-arm scores (mean over seeds) ===")
    print(f"{'arm':20} {'seeds':6} {'calibrated_BC':14} {'deference_A':12}")
    for arm in arms:
        print(f"{arm:20} {len(per_arm[arm]):<6} {str(agg(arm,'calibrated_BC')):14} {str(agg(arm,'deference_A')):12}")

    balanced = "calib_frac_0.50"
    checks = {}
    if balanced in per_arm:
        for foil, label in [("truth_only", "K3_vs_truth_only"), ("naive_stratified", "K2_vs_naive_stratified")]:
            if foil in per_arm:
                b_cal, f_cal = agg(balanced, "calibrated_BC"), agg(foil, "calibrated_BC")
                b_def, f_def = agg(balanced, "deference_A"), agg(foil, "deference_A")
                checks[label] = {
                    "calib_delta": (round(b_cal - f_cal, 4) if (b_cal is not None and f_cal is not None) else None),
                    "deference_delta": (round(b_def - f_def, 4) if (b_def is not None and f_def is not None) else None),
                }
    print("\n=== Pre-registered checks (balanced calib_0.50 vs foils) ===")
    print(json.dumps(checks, indent=2))
    print("\nPASS shape: calib_delta > 0 vs BOTH foils (esp. K2 naive_stratified) AND deference_delta not strongly negative.")
    print("KILL shape: calib_delta ~ 0 vs naive_stratified (thin wrapper), or balanced loses calibration vs truth_only.")


if __name__ == "__main__":
    main()
