"""C09 DISTRIBUTION-distillation falsification gate (MLX, on-device).

Applies the preregistered C09 gate (docs/experiments/MANIFOLD_TO_DATA_PROVENANCE.md) to the
soft-label-KL distilled LoRAs. For base + each arm, on HELD-OUT carrier templates, it reads the
model's value distribution and computes two transfer metrics:

  expected_position   E[index]/(n-1) over the value distribution, averaged across held-out source
                      carriers — 0=source end, 1=target end. "Did the LoRA lean the model toward the
                      target value with no hook?"  (delta vs base = the transfer score)
  order_corr          Spearman rho between the intended waypoint order (source..target) and the
                      model's expected_position on the held-out *target-filled-at-each-value* carriers
                      — the prereg "order correlation" primary. Higher = monotonic transfer.

Gate (REFUTE if any fails, per prereg): on rank AND education, geometry-gated manifold must beat
BOTH prompt_only AND linear by >= +0.05 on the macro transfer score. Also refute if the geometry
arms do not beat the shuffled_label control (the gain is not geometry then), or if the stress concept
passes the same energy gate.

    python3 scripts/_c09_distill_gate.py
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
GATE_MARGIN = 0.05
ARMS = ["gated_manifold", "ungated_manifold", "linear", "prompt_only", "shuffled_label"]
# prereg primary concepts + stress case
CONCEPTS = {"rank": ("first_token", 20), "education": ("full_string", 8), "days_of_week": ("first_token", 14)}


def _distribution(service, prompt, layer, concept, readout):
    if readout == "full_string":
        return np.asarray(service._value_string_distribution(prompt, layer, None, 0, concept), dtype=float)
    token_ids = service._concept_token_ids(concept)
    return np.asarray(service._output_distribution(prompt, layer, None, 0, token_ids), dtype=float)


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def _eval_arm(service, concept, layer, readout, templates):
    """expected_position averaged over held-out source carriers; order_corr over per-value positions."""
    n = len(concept.items)
    positions_by_value = {}  # value index -> mean expected_position across held-out templates
    masses = []
    for vi, val in enumerate(concept.items):
        pos_list = []
        for tmpl in templates:
            prompt = tmpl.format(item=val)
            p = _distribution(service, prompt, layer, concept, readout)
            if p.sum() <= 0:
                continue
            p = p / p.sum()
            pos_list.append(float(np.dot(np.arange(n), p) / (n - 1)))
            if vi < n - 1:  # source-side mass toward target (matches existing eval's source carriers)
                masses.append(float(p[n - 1]))
        if pos_list:
            positions_by_value[vi] = float(np.mean(pos_list))
    # macro transfer: mean expected_position over source values (0..n-2), held-out templates
    src_positions = [positions_by_value[vi] for vi in range(n - 1) if vi in positions_by_value]
    expected_position = float(np.mean(src_positions)) if src_positions else None
    # order correlation: does the model's position track the intended value order?
    idxs = sorted(positions_by_value)
    order_corr = _spearman(idxs, [positions_by_value[i] for i in idxs]) if len(idxs) >= 2 else None
    return {"expected_position": round(expected_position, 4) if expected_position is not None else None,
            "order_corr": round(order_corr, 4) if order_corr is not None else None,
            "target_mass_sources": round(float(np.mean(masses)), 4) if masses else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="reports/manifold_c09/distribution_distill")
    ap.add_argument("--concepts", nargs="*", default=list(CONCEPTS))
    ap.add_argument("--eval-template-index", type=int, default=4, help="held-out carrier (last template)")
    args = ap.parse_args()

    root = Path(args.root)
    all_results = {}
    for cname in args.concepts:
        readout, layer = CONCEPTS[cname]
        concept = get_concept(cname)
        eval_idx = min(args.eval_template_index, len(concept.templates) - 1)
        templates = [concept.templates[eval_idx]]  # held-out carrier
        print(f"\n[gate] {cname} L{layer} readout={readout} held-out template={templates[0]!r}", flush=True)
        res = {}
        print("[gate] base …", flush=True)
        base_svc = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12)
        res["base"] = _eval_arm(base_svc, concept, layer, readout, templates)
        del base_svc
        for arm in ARMS:
            adapter = root / cname / arm / "adapter"
            if not (adapter / "adapters.safetensors").exists():
                res[arm] = {"error": "no_adapter"}
                continue
            print(f"[gate] {arm} …", flush=True)
            svc = build_mlx_service(DEFAULT_MLX_MODEL, default_layer=12, adapter_path=str(adapter))
            res[arm] = _eval_arm(svc, concept, layer, readout, templates)
            del svc

        base_pos = res["base"]["expected_position"]
        deltas = {a: (round(res[a]["expected_position"] - base_pos, 4)
                      if res.get(a, {}).get("expected_position") is not None and base_pos is not None else None)
                  for a in ARMS}
        all_results[cname] = {"layer": layer, "readout": readout, "held_out_template": templates[0],
                              "arms": res, "delta_expected_position_vs_base": deltas}
        (root / cname / "gate_eval.json").write_text(
            json.dumps(all_results[cname], indent=2) + "\n", encoding="utf-8")
        print(json.dumps(deltas, indent=2), flush=True)

    # ---- apply the preregistered gate ----
    def d(c, a):
        return all_results[c]["delta_expected_position_vs_base"].get(a)

    verdict = {"experiment": "C09 distribution (soft-label KL) distillation — falsification gate",
               "gate_margin": GATE_MARGIN, "concepts": {}, "checks": {}}
    refuted_reasons = []
    for c in ("rank", "education"):
        if c not in all_results:
            continue
        gm, pp, ln, sh = (d(c, "gated_manifold"), d(c, "prompt_only"), d(c, "linear"), d(c, "shuffled_label"))
        beats_prompt = gm is not None and pp is not None and (gm - pp) >= GATE_MARGIN
        beats_linear = gm is not None and ln is not None and (gm - ln) >= GATE_MARGIN
        beats_shuffle = gm is not None and sh is not None and (gm - sh) >= GATE_MARGIN
        verdict["concepts"][c] = {
            "gated_manifold_delta": gm, "prompt_only_delta": pp, "linear_delta": ln, "shuffled_label_delta": sh,
            "gated_minus_prompt": round(gm - pp, 4) if gm is not None and pp is not None else None,
            "gated_minus_linear": round(gm - ln, 4) if gm is not None and ln is not None else None,
            "gated_minus_shuffled": round(gm - sh, 4) if gm is not None and sh is not None else None,
            "beats_prompt_only_by_margin": beats_prompt, "beats_linear_by_margin": beats_linear,
            "beats_shuffled_control_by_margin": beats_shuffle,
            "order_corr_gated": all_results[c]["arms"].get("gated_manifold", {}).get("order_corr"),
        }
        if not (beats_prompt and beats_linear):
            refuted_reasons.append(f"{c}: gated_manifold did not beat prompt_only AND linear by +{GATE_MARGIN} "
                                   f"(Δgated={gm}, Δprompt={pp}, Δlinear={ln})")
        if not beats_shuffle:
            refuted_reasons.append(f"{c}: gated_manifold did not beat the shuffled-label control by +{GATE_MARGIN} "
                                   f"(Δgated={gm}, Δshuffled={sh}) — the gain is not geometry-driven")

    verdict["result"] = "REFUTED" if refuted_reasons else "SUPPORTED"
    verdict["refuted_reasons"] = refuted_reasons
    (Path(args.root) / "distill_verdict.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print("\n==== VERDICT ====")
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
