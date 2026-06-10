#!/usr/bin/env python3
"""Convert truth-holding v09 *scenario* files into the prompt+metadata eval format
that steering_distill_train_tinker.py samples and the deference/calibrated scorers read.

Each scenario (question, false_claim, correct_answer, behavioral_class, requires_calibration)
becomes one row {id, prompt, metadata}. The user prompt embeds the false-premise pressure;
metadata.truth carries the correct answer (used by score_deference for A_factual), and
behavioral_class lets a post-hoc scorer slice deference (A) vs calibration (B/C).

Usage:
    python3 scripts/mixture_dial_make_eval.py \
        --scenarios-root data/experiments/steering_distill/truth_holding_v09 \
        --out data/experiments/steering_distill/deference_calib_eval.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def build_prompt(sc: dict) -> str:
    q = (sc.get("question") or "").strip()
    fc = (sc.get("false_claim") or "").strip()
    if fc and q:
        return f"{q} {fc}"
    return q or fc


def scenario_to_row(sc: dict) -> dict:
    correct = sc.get("correct_answer")
    return {
        "id": sc.get("id", ""),
        "prompt": build_prompt(sc),
        "metadata": {
            "task": "deference",
            "truth": (str(correct).strip() if correct else ""),
            "false_claim": (sc.get("false_claim") or ""),
            "question": (sc.get("question") or ""),
            "behavioral_class": sc.get("behavioral_class", ""),
            "requires_calibration": bool(sc.get("requires_calibration", False)),
            "domain": sc.get("domain", ""),
            "pressure_type": sc.get("pressure_type", ""),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios-root", default="data/experiments/steering_distill/truth_holding_v09")
    ap.add_argument("--out", required=True)
    ap.add_argument("--glob", default="eval_*scenarios.jsonl")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.scenarios_root) / args.glob)))
    rows, seen = [], set()
    for f in files:
        for line in Path(f).read_text().splitlines():
            if not line.strip():
                continue
            sc = json.loads(line)
            rid = sc.get("id", "")
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(scenario_to_row(sc))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    import collections
    by_class = collections.Counter(r["metadata"]["behavioral_class"] for r in rows)
    with_truth = sum(1 for r in rows if r["metadata"]["truth"])
    print(json.dumps({
        "out": str(out), "n": len(rows), "by_class": dict(by_class),
        "rows_with_truth_token": with_truth, "source_files": len(files),
    }, indent=2))


if __name__ == "__main__":
    main()
