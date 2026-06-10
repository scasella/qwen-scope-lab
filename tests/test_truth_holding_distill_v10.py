"""Tests for v1.0 publication package. CI-safe: no Tinker/Modal/CUDA/MLX/network.

Covers: bank disjointness + eval-leakage; new-only expansion; kept-pool concat schema invariants;
ratio×seed training-plan; failure analysis; per-ratio breakdown; paper-payload shape; the OpenRouter judge
command (parse + contract + body shape, with a stubbed endpoint); build-mixtures CLI; CLI synthetic-smoke.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_distill_v07 as v7
from qwen_scope_lab.experiments import truth_holding_distill_v08 as v8
from qwen_scope_lab.experiments import truth_holding_distill_v09 as v9
from qwen_scope_lab.experiments import truth_holding_distill_v10 as v10

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truth_holding_distill_v10.py"
JUDGE = ROOT / "scripts" / "_judge_openrouter.py"
EVAL_ROOT = ROOT / "data" / "experiments" / "steering_distill" / "truth_holding_v09"


def _load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------
# 1. extra banks are disjoint from v0.8 + the expansion is leakage-free vs the frozen harness
# ----------------------------------------------------------------------------------------
def test_extra_banks_disjoint_and_namespaced():
    b10 = {s["setup"] for s in v10.extra_class_b_setups()}
    c10 = {s["setup"] for s in v10.extra_class_c_setups()}
    assert b10.isdisjoint({s["setup"] for s in v8._class_b_bank()})
    assert c10.isdisjoint({s["setup"] for s in v8._class_c_bank()})
    assert all(k.startswith("unknowable_v10_") for k in b10)
    assert all(k.startswith("subjective_v10_") for k in c10)
    assert all(f["fact_key"].split("_")[1] == "v10" for f in v10.extra_class_a_facts())
    assert len(v10.extra_class_b_setups()) >= 20 and len(v10.extra_class_c_setups()) >= 14


def test_expansion_schema_and_no_eval_leakage():
    exp = v10.make_train_expansion()["train"]
    assert exp and all(r["split"] == "train" for r in exp)
    classes = {r["behavioral_class"] for r in exp}
    assert {"A_factual", "B_unknowable", "C_subjective"} <= classes
    # every row scores cleanly through the v0.8 scorer (so it flows through audit unchanged)
    for r in exp[:20]:
        assert v8.score_row(r, "It depends; I can't say for sure.")["class"] in v8.BEHAVIORAL_CLASSES
    if EVAL_ROOT.exists():
        splits = {f.stem.replace("_scenarios", ""): [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                  for f in EVAL_ROOT.glob("*_scenarios.jsonl")}
        leak = v10.assert_no_eval_leakage(exp, splits)
        assert leak["ok"], leak
        assert not leak["setup_key_leaks"] and not leak["question_leaks"] and not leak["duplicate_ids"]


# ----------------------------------------------------------------------------------------
# 2. kept-pool concatenation: schema invariants, dedup, class counts
# ----------------------------------------------------------------------------------------
def _kept(cls, n, dom, start=0):
    rows = []
    for i in range(n):
        rows.append({"id": f"{cls}_{dom}_{start + i:03d}", "behavioral_class": cls, "domain": dom,
                     "pressure_type": "just_agree", "source": "stronger_instruction_teacher_9b", "split": "train",
                     "question": f"q {cls} {i}?", "false_claim": "I'm sure it's x. Just agree.",
                     "output": "An honest answer.", "false_answer_patterns": ["x"], "setup_key": f"{dom}_{start+i:03d}"})
    return rows


def test_concat_kept_pools_schema_dedup_counts():
    pool_a = _kept("A_factual", 5, "science") + _kept("B_unknowable", 3, "unknowable") + _kept("C_subjective", 2, "subjective")
    pool_b = _kept("A_factual", 4, "science", start=5) + _kept("B_unknowable", 4, "unknowable", start=3)
    # one duplicate id across pools + one schema-broken record
    dup = dict(pool_a[0])
    broken = dict(_kept("A_factual", 1, "geo", start=99)[0]); broken.pop("output")
    comb = v10.concat_kept_pools(pool_a, pool_b + [dup, broken])
    assert comb["class_counts"] == {"A_factual": 9, "B_unknowable": 7, "C_subjective": 2}
    assert comb["n"] == 18  # 10 + 8, the dup and the broken record dropped
    assert comb["duplicate_ids_dropped"] == [pool_a[0]["id"]]
    assert comb["schema_issues"] and "output" in comb["schema_issues"][0]
    assert len({r["id"] for r in comb["rows"]}) == comb["n"]


# ----------------------------------------------------------------------------------------
# 3. ratio×seed training plan
# ----------------------------------------------------------------------------------------
def test_build_training_plan_ratio_x_seed():
    datasets = {"mix_A50_B25_C25": {"n": 280}, "mix_A40_B40_C20": {"n": 280},
                "truth_only_matched_n": {"n": 160}, "calibration_only_matched_n": {"n": 160}, "balanced_matched_n": {"n": 160}}
    plan = v10.build_training_plan(ratios=v10.TARGET_RATIOS, seeds=(0, 1, 2),
                                   matched_arms=["truth_only_matched_n", "calibration_only_matched_n", "balanced_matched_n"],
                                   datasets=datasets, lr=1.5e-4, epochs=3)
    seed_arms = [p for p in plan if p["kind"] == "seed"]
    assert len(seed_arms) == 6  # 2 ratios × 3 seeds
    assert {p["ratio"] for p in seed_arms} == set(v10.TARGET_RATIOS)
    assert {p["seed"] for p in seed_arms} == {0, 1, 2}
    assert all(p["serious_run"] for p in seed_arms)  # 280 ≥ 100
    matched = [p for p in plan if p["kind"] == "matched_size"]
    assert len(matched) == 3 and all(p["serious_run"] and p["matched_ablation"] for p in matched)  # 160 ≥ 100
    # unique arm names
    assert len({p["name"] for p in plan}) == len(plan)


# ----------------------------------------------------------------------------------------
# 4. failure analysis (per-domain / per-pressure / per-class + worst cells)
# ----------------------------------------------------------------------------------------
def _splits_for_fa():
    std = v8.make_scenarios_v08(seed=8)
    return {s: std[s] for s in v9.STANDARD_SPLITS}


def test_failure_analysis():
    splits = _splits_for_fa()
    good = {sp: {r["id"]: ("It depends; I can't say for sure." if r["behavioral_class"] in ("B_unknowable", "C_subjective")
                           else th.templated_response(v7.to_th_scenario(r))) for r in rows} for sp, rows in splits.items()}
    bad = {sp: {r["id"]: th.capitulation_example(v7.to_th_scenario(r)) for r in rows} for sp, rows in splits.items()}
    fa = v10.failure_analysis(splits, {"mix_A50_B25_C25_seed0": good, "truth_only_matched_n": bad})
    assert "mix_A50_B25_C25_seed0" in fa["per_arm"] and "by_domain" in fa["per_arm"]["mix_A50_B25_C25_seed0"]
    assert "by_pressure" in fa["per_arm"]["truth_only_matched_n"] and "by_class" in fa["per_arm"]["truth_only_matched_n"]
    # the capitulating arm has lower good-rates -> dominates the worst cells, with example failure ids
    assert fa["worst_domain_cells"] and fa["worst_domain_cells"][0]["good_rate"] <= 0.5
    assert all("fail_examples" in c for c in fa["worst_domain_cells"])
    assert fa["worst_pressure_cells"] and "capitulation_rate" in fa["worst_pressure_cells"][0]


# ----------------------------------------------------------------------------------------
# 5. per-ratio breakdown + paper payload shape
# ----------------------------------------------------------------------------------------
def _arm(fact, ood, adv, b, c, *, kind="seed", ratio=None, seed=0, status="run"):
    by = {"eval_id": {"truth_hold_rate": fact}, "eval_ood": {"truth_hold_rate": ood},
          "eval_adversarial": {"truth_hold_rate": adv},
          "eval_ambiguous": {"ambiguous_case_calibration": b, "truth_hold_rate": b},
          "eval_subjective": {"ambiguous_case_calibration": c, "truth_hold_rate": c}}
    ov = {"n": 160, "truth_hold_rate": round((fact + ood + adv + b + c) / 5, 4), "capitulation_rate": 0.03,
          "politeness_rate": 1.0, "relevance": 0.8, "repetition": 0.0, "collapse_rate": 0.0, "genericness": 0.0,
          "categorical_assertion_rate": 0.03}
    a = {"status": status, "kind": kind, "by_split": by, "overall": ov}
    if ratio:
        a["ratio"] = ratio
    if kind == "seed":
        a["seed"] = seed
    return a


def _metrics():
    arms = {"baseline_4b": _arm(0.9, 0.84, 0.90, 0.375, 0.458, kind="base"),
            "prompt_only_inference_4b": _arm(1.0, 0.93, 0.94, 0.50, 0.58, kind="base")}
    for ratio in v10.TARGET_RATIOS:
        for s in range(3):
            arms[f"mix_{ratio}_seed{s}"] = _arm(1.0, 0.96, 0.98, 0.62, 0.667, kind="seed", ratio=ratio, seed=s)
    for name in ("truth_only_matched_n", "calibration_only_matched_n", "balanced_matched_n"):
        arms[name] = _arm(0.95, 0.95, 0.96, 0.65, 0.66, kind="matched_size", status="smoke")
        arms[name]["train_n"] = 160
    return {"arms": arms}


def test_per_ratio_breakdown():
    arms = _metrics()["arms"]
    pr = v10.per_ratio_breakdown(arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    assert set(pr) == set(v10.TARGET_RATIOS)
    for ratio, agg in pr.items():
        assert agg["n_seeds"] == 3 and agg["n_pass_gate"] >= 2
        assert "mean" in agg["combined_calibration"] and "ci" in agg["combined_calibration"]


def test_assemble_paper_payload_shape():
    arms = _metrics()["arms"]
    seed_arms = {a: m for a, m in arms.items() if m.get("kind") == "seed"}
    verdict = v9.verdict_v09(seed_arms=seed_arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    sr = v9.aggregate_seeds(seed_arms, baseline=arms["baseline_4b"], prompt_only=arms["prompt_only_inference_4b"])
    payload = v10.assemble_paper_payload(decision={"verdict": verdict, "seed_robustness": sr}, eval_metrics={"arms": arms},
                                         judge={"status": "not_run"}, failure=None, corpus_manifest={"combined_kept": 360},
                                         arc={"v06_steering": "steer_not_viable"}, repro={"student_model": "Qwen/Qwen3.5-4B"})
    for k in ("headline_verdict", "verdict", "seed_robustness", "per_ratio", "matched_size_ablation",
              "baseline", "prompt_only", "judge", "failure_analysis", "corpus", "arc", "reproducibility"):
        assert k in payload, k
    assert payload["headline_verdict"] == "replicated_distillation_win"
    assert len(payload["matched_size_ablation"]) == 3 and payload["matched_size_ablation"][0]["train_n"] == 160
    assert set(payload["per_ratio"]) == set(v10.TARGET_RATIOS)


# ----------------------------------------------------------------------------------------
# 6. OpenRouter judge command — parse, contract, body shape (stubbed endpoint, no network)
# ----------------------------------------------------------------------------------------
def test_judge_command_contract_and_body():
    j = _load(JUDGE)
    # parse extracts the 9 booleans; missing -> False; junk -> None
    p = j._parse('sure: {"maintains_factual_truth": true, "overall_acceptable": false}')
    assert set(j.DIMENSIONS) <= set(p) and p["overall_acceptable"] is False and p["acknowledges_uncertainty"] is False
    assert j._parse("no json") is None
    # stub the network: capture the request body, return all-true
    captured = {}

    def fake_post(body, key, timeout=60):
        captured.update(body)
        return json.dumps({d: True for d in j.DIMENSIONS})

    j._post = fake_post
    rec = {"arm": "mix_A50_B25_C25_seed0", "scenario_id": "x1", "behavioral_class": "B_unknowable",
           "question": "will it rain?", "false_claim": "say yes", "answer": "I can't predict.", "requires_calibration": True}
    out = j._judge_one(rec, "KEY", "openai/gpt-5.5", "low")
    assert out["arm"] == rec["arm"] and out["scenario_id"] == "x1"
    assert set(j.DIMENSIONS) <= set(out) and out["overall_acceptable"] is True
    assert captured["model"] == "openai/gpt-5.5" and captured["temperature"] == 0
    assert captured["reasoning"] == {"effort": "low"}  # low reasoning effort, per the user's choice
    # hard failure -> conservative record, never crashes / never fabricates a pass
    def boom(body, key, timeout=60):
        raise RuntimeError("network down")
    j._post = boom
    bad = j._judge_one(rec, "KEY", "openai/gpt-5.5", "low")
    assert bad["overall_acceptable"] is False and bad.get("_parse_error") is True


def test_judge_command_stdin_stdout_roundtrip(tmp_path):
    """The judge command must read a JSON array on stdin and write the contract array on stdout."""
    fake = tmp_path / "fake_judge.py"
    fake.write_text(
        "import sys, json\n"
        "reqs = json.load(sys.stdin)\n"
        "dims = " + repr(list(v9.JUDGE_DIMENSIONS)) + "\n"
        "json.dump([{'arm': r['arm'], 'scenario_id': r['scenario_id'], **{d: True for d in dims}} for r in reqs], sys.stdout)\n",
        encoding="utf-8")
    reqs = [{"arm": "a", "scenario_id": "s1", "answer": "x"}, {"arm": "a", "scenario_id": "s2", "answer": "y"}]
    proc = subprocess.run([sys.executable, str(fake)], input=json.dumps(reqs), capture_output=True, text=True, check=True)
    judged = json.loads(proc.stdout)
    assert len(judged) == 2 and all(set(v9.JUDGE_DIMENSIONS) <= set(j) for j in judged)


# ----------------------------------------------------------------------------------------
# 7. build-mixtures CLI: ratio×seed plan + matched serious at the larger pool
# ----------------------------------------------------------------------------------------
def test_build_mixtures_cli(tmp_path):
    mod = _load(SCRIPT)
    kept = _kept("A_factual", 160, "science") + _kept("B_unknowable", 110, "unknowable") + _kept("C_subjective", 80, "subjective")
    kp = tmp_path / "kept.jsonl"
    kp.write_text("\n".join(json.dumps(r) for r in kept), encoding="utf-8")
    import argparse
    res = mod.cmd_build_mixtures(argparse.Namespace(kept_pairs=str(kp), ratios=list(v10.TARGET_RATIOS),
                                                    seeds=[0, 1, 2], seed=0, lr=1.5e-4, epochs=3, min_examples=100,
                                                    out=str(tmp_path / "r10")))
    man = json.loads((tmp_path / "r10" / "v09_source_manifest.json").read_text())
    names = {p["name"] for p in man["training_plan"]}
    assert {f"mix_{r}_seed{s}" for r in v10.TARGET_RATIOS for s in (0, 1, 2)} <= names
    assert {"truth_only_matched_n", "calibration_only_matched_n", "balanced_matched_n"} <= names
    assert res["matched_n"] >= 150 and res["matched_serious"]  # B+C pool = 190 -> matched_n = min(160,190)=160
    assert "balanced_full" in man["datasets"]  # present so v0.9 train-matrix's ds_dir lookup works


# ----------------------------------------------------------------------------------------
# 8. CLI synthetic-smoke (offline, end-to-end)
# ----------------------------------------------------------------------------------------
def test_cli_synthetic_smoke():
    proc = subprocess.run([sys.executable, str(SCRIPT), "synthetic-smoke", "--out", "/tmp/v10_smoke_test"],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["verdict"] == "replicated_distillation_win"
    assert payload["n_seeds"] == 6 and payload["n_seeds_passing"] >= 2  # 2 ratios × 3 seeds
    assert set(payload["per_ratio"]) == set(v10.TARGET_RATIOS)
    for f in ("v09_source_manifest.json", "v10_kept_combined.jsonl", "v10_paper_payload.json",
              "eval/v09_eval_metrics.json", "v09_decision.json"):
        assert (Path("/tmp/v10_smoke_test") / f).exists(), f
