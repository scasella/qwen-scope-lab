#!/usr/bin/env python3
"""Judge-score the Mixture Dial pilot and emit the K2/K3 verdict (PAID: OpenRouter gpt-5.5).

Reads <run>/arms/<arm>/<seed>/eval_arms.json (written by mixture_dial_validate.py --execute),
runs the gpt-5.5 rubric judge on each DISTILLED arm's calibration scenarios (B_unknowable +
C_subjective), aggregates `overall_acceptable` per arm across seeds, and computes:
  K3 = calib_frac_0.50  -  truth_only        (does the dial beat truth-only on calibration?)
  K2 = calib_frac_0.50  -  naive_stratified  (does it beat proportional sampling — the thin-wrapper foil?)

Re-runnable from saved eval_arms.json (no retraining). Writes <run>/pilot_verdict.json.
Underscore prefix = standalone network/paid, like _judge_openrouter.py.
"""
from __future__ import annotations

import glob
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_ADAPTER = str(ROOT / "scripts" / "mixture_dial_judge.py")


def judge_arm_seed(eval_arms: str) -> float | None:
    out = subprocess.run(
        [sys.executable, JUDGE_ADAPTER, "--eval-arms", eval_arms,
         "--arms", "distilled_4b", "--classes", "B_unknowable,C_subjective"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=1800,
    )
    if out.returncode != 0:
        sys.stderr.write(f"judge failed for {eval_arms}: {out.stderr}\n")
        return None
    r = json.loads(out.stdout)
    return r.get("arms", {}).get("distilled_4b", {}).get("overall_acceptable_rate")


def main() -> None:
    run = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mixture_pilot_run"
    paths = sorted(glob.glob(f"{run}/arms/*/*/eval_arms.json"))
    if not paths:
        raise SystemExit(f"no eval_arms.json under {run}/arms/*/* — run the pilot first")

    per_arm: dict[str, list[float]] = {}
    per_run: list[dict] = []
    for p in paths:
        arm = Path(p).parts[-3]
        seed = Path(p).parts[-2]
        rate = judge_arm_seed(p)
        print(f"  judged {arm}/seed{seed}: calibration_acceptable={rate}", file=sys.stderr, flush=True)
        if rate is not None:
            per_arm.setdefault(arm, []).append(rate)
        per_run.append({"arm": arm, "seed": seed, "calibration_acceptable": rate})

    agg = {a: round(statistics.mean(v), 3) for a, v in per_arm.items() if v}

    def delta(a: str, b: str) -> float | None:
        if a in agg and b in agg:
            return round(agg[a] - agg[b], 3)
        return None

    k3 = delta("calib_frac_0.50", "truth_only")
    k2 = delta("calib_frac_0.50", "naive_stratified")

    def verdict() -> str:
        if k2 is None or k3 is None:
            return "INCONCLUSIVE — missing arms"
        if k2 <= 0.0:
            return "KILL (K2): balanced dial does NOT beat naive_stratified — thin wrapper over proportional sampling"
        if k3 <= 0.0:
            return "KILL (K3): balanced dial does NOT beat truth_only on calibration"
        if k2 >= 0.05 and k3 >= 0.05:
            return "PASS: balanced dial beats BOTH naive_stratified (K2) and truth_only (K3) on judge-scored calibration"
        return "WEAK PASS: positive but small deltas — widen seeds before trusting"

    result = {
        "run": run,
        "metric": "gpt-5.5 rubric judge — overall_acceptable on B_unknowable+C_subjective (distilled arm)",
        "calibration_by_arm_mean_over_seeds": agg,
        "K3_vs_truth_only": k3,
        "K2_vs_naive_stratified": k2,
        "verdict": verdict(),
        "per_run": per_run,
    }
    (Path(run) / "pilot_verdict.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
