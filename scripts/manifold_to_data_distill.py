#!/usr/bin/env python3
"""Manifold-to-data provenance compiler CLI (C09 — first build step).

Compile concept-manifold steering payloads into provenance-stamped SFT/preference data, keeping
only the on-manifold samples whose geometry (behavior-energy ≤ linear chord, recovered_r ≥ threshold)
clears the gate. This first build step ships a deterministic, model-free smoke that proves the
schema/gate/export path; a live mode that pulls real payloads from a running lab service is a TODO.

    # model-free smoke (writes reports/steering_distill/manifold_smoke)
    python3 scripts/manifold_to_data_distill.py synthetic-smoke
    python3 scripts/manifold_to_data_distill.py synthetic-smoke --scenario fail --out reports/tmp/manifold_fail

See docs/experiments/MANIFOLD_TO_DATA_PROVENANCE.md for the preregistration + falsification gate.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from qwen_scope_lab.experiments import manifold_distill as md


def cmd_synthetic_smoke(args: argparse.Namespace) -> int:
    spec = md.ManifoldDataSpec.explicit(concept="rank", source="private", target="general", layer=20)
    spec.source_kind = "synthetic"
    gate = md.GateConfig(min_recovered_r=args.min_recovered_r, energy_margin=args.energy_margin)
    graded = md.compile_payload(md.build_synthetic_payload(args.scenario), spec, gate)
    paths = md.write_outputs(args.out, spec, graded, gate)
    m = graded["metrics"]
    print(json.dumps({"scenario": args.scenario, "out": args.out,
                      "n_records": m["n_records"], "n_kept": m["n_kept"],
                      "n_sft": m["n_sft"], "n_preference": m["n_preference"],
                      "equal_size_n_per_arm": m["equal_size_n_per_arm"],
                      "reject_reason_counts": m["reject_reason_counts"], "paths": paths}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic-smoke", help="compile a synthetic manifold payload (no model/network)")
    s.add_argument("--scenario", choices=["win", "fail"], default="win")
    s.add_argument("--out", default="reports/steering_distill/manifold_smoke")
    s.add_argument("--min-recovered-r", type=float, default=0.5)
    s.add_argument("--energy-margin", type=float, default=0.0)
    s.set_defaults(func=cmd_synthetic_smoke)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
