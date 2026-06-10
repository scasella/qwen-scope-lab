"""Invoke the 27B truth-holding showdown Modal function and save its results JSON locally.

Real GPU run (H100). Network + Modal credentials required; kept outside the test path.

    python scripts/_run_27b_showdown_modal.py --out reports/steering_distill/th_v06_27b_showdown/modal_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", default="0.25,0.5,1,2,3,4")
    ap.add_argument("--layer-strategy", default="low_mid_high")
    ap.add_argument("--n-sweep", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import modal_app

    print("launching 27B showdown on Modal (H100)…", flush=True)
    with modal_app.app.run():
        res = modal_app.truth_holding_27b_showdown.remote(
            strengths=args.strengths, layer_strategy=args.layer_strategy,
            n_sweep=args.n_sweep, max_new_tokens=args.max_new_tokens,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sweep = res.get("sweep", [])
    print(f"wrote {out}", flush=True)
    print(json.dumps({"layers": res.get("layers"), "probe_auc_by_layer": res.get("probe_auc_by_layer"),
                      "n_sweep_conditions": len(sweep), "best_condition": res.get("best_condition"),
                      "prompt_only_n": len(res.get("prompt_only_rows", [])),
                      "steer_full_n": len(res.get("steer_full_rows", []))}, indent=2))


if __name__ == "__main__":
    main()
