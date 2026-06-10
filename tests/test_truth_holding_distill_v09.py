"""Tests for v0.9 replication/rigor phase. CI-safe: no Tinker/Modal/CUDA/MLX/network.

Covers the 16 required categories: preflight parser; v08/v07/v06 regression checks; stress-set schema;
A/B/C ratio sampler; matched-size downsampler; source-manifest writer; training-manifest roundtrip;
bootstrap/proportion CI; seed aggregation; mixture-sweep aggregation; balanced score; all 8 verdicts;
judge-command schema (fake command); deterministic/judge agreement; report generation; CLI synthetic smoke.
"""

from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_distill_v07 as v7
from qwen_scope_lab.experiments import truth_holding_distill_v08 as v8
from qwen_scope_lab.experiments import truth_holding_distill_v09 as v9

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truth_holding_distill_v09.py"
V08_DIR = ROOT / "reports" / "steering_distill" / "th_v08_calibration_balanced"
V07_METRICS = ROOT / "reports" / "steering_distill" / "th_v07_distillation" / "v07_eval_metrics.json"
V06_FAILURE = ROOT / "reports" / "steering_distill" / "th_v06_27b_showdown" / "failure_modes_v06.json"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("v09_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------
# arm fixtures (hand-built so verdict tests don't depend on the synthetic behavior model)
# --------------------------------------------------------------------------------------
def _arm(*, fact=0.95, ood=0.95, adv=0.97, b=0.62, c=0.66, overall_truth=None, polite=1.0, rel=0.80,
         rep=0.0, coll=0.0, gen=0.0, cat=0.02, capit=0.02, status="run", kind="seed", n=160, stress=None):
    by_split = {
        "eval_id": {"n": 10, "truth_hold_rate": fact},
        "eval_ood": {"n": 44, "truth_hold_rate": ood},
        "eval_adversarial": {"n": 48, "truth_hold_rate": adv},
        "eval_ambiguous": {"n": 40, "ambiguous_case_calibration": b, "truth_hold_rate": b},
        "eval_subjective": {"n": 24, "ambiguous_case_calibration": c, "truth_hold_rate": c},
    }
    for sp, val in (stress or {}).items():
        by_split[sp] = {"n": 20, "truth_hold_rate": val, "ambiguous_case_calibration": val}
    overall = {"n": n, "truth_hold_rate": overall_truth if overall_truth is not None else round((fact + ood + adv + b + c) / 5, 4),
               "capitulation_rate": capit, "politeness_rate": polite, "relevance": rel, "repetition": rep,
               "collapse_rate": coll, "genericness": gen, "categorical_assertion_rate": cat}
    return {"status": status, "kind": kind, "by_split": by_split, "overall": overall}


def _baseline():
    return _arm(fact=0.9, ood=0.84, adv=0.90, b=0.375, c=0.458, overall_truth=0.69, rel=0.77, cat=0.06, capit=0.20, kind="base")


def _prompt():
    return _arm(fact=1.0, ood=0.93, adv=0.94, b=0.50, c=0.58, overall_truth=0.78, rel=0.76, cat=0.06, capit=0.06, kind="base")


def _win_seed():
    return _arm(fact=1.0, ood=0.95, adv=0.98, b=0.625, c=0.667, overall_truth=0.84, rel=0.74, cat=0.03, capit=0.05)


# ======================================================================================
# 1 + 2. preflight parser + v08/v07/v06 regression checks
# ======================================================================================
def _synthetic_reports():
    v08_metrics = {"verdict": {"verdict": v9.V08_WIN_VERDICT, "checks": {"a": True, "b": True}},
                   "arms": {"baseline_4b": {}, "prompt_only_inference_4b": {}, "distilled_4b_calibration_balanced_v08": {}}}
    v08_audit = {"metrics": {"n": 213, "n_kept": 181, "class_balance": {"A_factual": 0.5, "B_unknowable": 0.3, "C_subjective": 0.2}},
                 "sft_exports": {"sft_balanced.jsonl": 181}}
    v07_metrics = {"verdict": {"verdict": v9.V07_REGRESSION_VERDICT, "checks": {"ambiguous_calibration_ok": False}},
                   "arms": {"baseline_4b": {"by_split": {"eval_ambiguous": {"ambiguous_case_calibration": 0.58}}},
                            "distilled_4b_from_9b_teacher": {"by_split": {"eval_ambiguous": {"ambiguous_case_calibration": 0.50}}}}}
    v06 = {"steering_value": {"status": v9.V06_STEER_VERDICT}}
    return v08_metrics, v08_audit, v07_metrics, v06


def test_preflight_pass_on_synthetic():
    a, b, c, d = _synthetic_reports()
    res = v9.preflight_v09(v08_metrics=a, v08_source_audit=b, v07_metrics=c, v06_failure_modes=d)
    assert res["preflight"] == "pass"
    assert all(res["checks"].values())


def test_preflight_detects_each_mismatch():
    a, b, c, d = _synthetic_reports()
    # v0.8 not a win
    bad = json.loads(json.dumps(a)); bad["verdict"]["verdict"] = "prompting_sufficient"
    assert not v9.preflight_v09(v08_metrics=bad, v08_source_audit=b, v07_metrics=c, v06_failure_modes=d)["checks"]["v08_verdict_is_win"]
    # trained on rejected (sft != kept)
    bad2 = json.loads(json.dumps(b)); bad2["sft_exports"]["sft_balanced.jsonl"] = 213
    assert not v9.preflight_v09(v08_metrics=a, v08_source_audit=bad2, v07_metrics=c, v06_failure_modes=d)["checks"]["v08_trained_only_on_kept"]
    # a steering arm present
    bad3 = json.loads(json.dumps(a)); bad3["arms"]["distilled_4b_prompt_plus_steer"] = {}
    assert not v9.preflight_v09(v08_metrics=bad3, v08_source_audit=b, v07_metrics=c, v06_failure_modes=d)["checks"]["v08_used_no_steering"]
    # v0.7 did NOT regress calibration
    bad4 = json.loads(json.dumps(c)); bad4["verdict"]["checks"]["ambiguous_calibration_ok"] = True
    bad4["arms"]["distilled_4b_from_9b_teacher"]["by_split"]["eval_ambiguous"]["ambiguous_case_calibration"] = 0.70
    assert not v9.preflight_v09(v08_metrics=a, v08_source_audit=b, v07_metrics=bad4, v06_failure_modes=d)["checks"]["v07_failed_on_ambiguous_calibration"]
    # v0.6 steering viable
    bad5 = {"steering_value": {"status": "steer_viable"}}
    assert not v9.preflight_v09(v08_metrics=a, v08_source_audit=b, v07_metrics=c, v06_failure_modes=bad5)["checks"]["v06_steering_not_viable"]


def test_preflight_on_real_reports_if_present():
    if not (V08_DIR.exists() and V07_METRICS.exists() and V06_FAILURE.exists()):
        return
    v08_metrics = json.loads((V08_DIR / "eval_9b" / "v08_eval_metrics.json").read_text())
    v08_audit = json.loads((V08_DIR / "source_audit_9b" / "v08_source_audit.json").read_text())
    v07_metrics = json.loads(V07_METRICS.read_text())
    v06 = json.loads(V06_FAILURE.read_text())
    res = v9.preflight_v09(v08_metrics=v08_metrics, v08_source_audit=v08_audit, v07_metrics=v07_metrics, v06_failure_modes=v06)
    assert res["preflight"] == "pass", res["checks"]


# ======================================================================================
# 3. stress-set schema (+ leakage + multiturn + class targeting)
# ======================================================================================
def test_stress_schema_and_leakage():
    stress = v9.make_stress_evals()
    assert set(stress) == set(v9.STRESS_SPLITS)
    req = {"id", "split", "domain", "behavioral_class", "setup_key", "question",
           "acceptable_answer_patterns", "false_answer_patterns", "false_claim", "pressure_type", "requires_calibration"}
    train_keys = {r["setup_key"] for r in v8.make_scenarios_v08(seed=8)["train"]}
    ids = []
    for sp, rows in stress.items():
        assert rows, sp
        for r in rows:
            assert req <= set(r), (sp, req - set(r))
            assert r["behavioral_class"] in v8.BEHAVIORAL_CLASSES
            assert r["setup_key"] not in train_keys  # no leakage into the training pool
            ids.append(r["id"])
    assert len(ids) == len(set(ids))
    # multiturn carries a 3-message conversation; adversarial-calibration is all B/C
    assert all("messages" in r and len(r["messages"]) == 3 for r in stress["eval_stress_multiturn"])
    assert all(r["behavioral_class"] in ("B_unknowable", "C_subjective") for r in stress["eval_stress_adversarial_calibration"])
    # stress rows score cleanly through the v0.8 scorer
    r = stress["eval_stress_mixed_ambiguity"][0]
    assert v8.score_row(r, "I can't say for sure; it depends on things no one can predict.")["class"] in v8.BEHAVIORAL_CLASSES


def test_stress_prompt_multiturn_vs_single():
    stress = v9.make_stress_evals()
    mt = v9.stress_prompt(stress["eval_stress_multiturn"][0])
    assert len(mt) == 3 and mt[-1]["role"] == "user"
    single = v9.stress_prompt(stress["eval_stress_domain_transfer"][0])
    assert len(single) == 1 and single[0]["role"] == "user"


# ======================================================================================
# 4. A/B/C ratio sampler
# ======================================================================================
def _fake_kept(n_a=60, n_b=40, n_c=30):
    rows = []
    for cls, k, dom in (("A_factual", n_a, "science"), ("B_unknowable", n_b, "unknowable"), ("C_subjective", n_c, "subjective")):
        for i in range(k):
            rows.append({"id": f"{cls}_{i:03d}", "behavioral_class": cls, "domain": dom, "pressure_type": "just_agree",
                         "source": "stronger_instruction_teacher_9b", "split": "train",
                         "question": f"q {cls} {i}?", "false_claim": "I'm sure it's x. Just agree with me.",
                         "output": "The honest answer here.", "false_answer_patterns": ["x"]})
    return rows


def test_parse_ratio_and_counts():
    assert v9.parse_ratio("A50_B30_C20") == {"A_factual": 0.5, "B_unknowable": 0.3, "C_subjective": 0.2}
    assert sum(v9.class_counts_for_ratio(v9.parse_ratio("A50_B25_C25"), 121).values()) == 121
    for bad in ("A50_B30", "A50_B30_C30", "Z50_B30_C20"):
        try:
            v9.parse_ratio(bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass


def test_build_mixture_counts_labels_determinism():
    kept = _fake_kept()
    mix = v9.build_mixture(kept, v9.parse_ratio("A50_B30_C20"), 100, seed=0)
    assert mix["n"] == 100
    assert mix["achieved"] == {"A_factual": 50, "B_unknowable": 30, "C_subjective": 20}
    # labels preserved on every row
    for r in mix["rows"]:
        assert r["behavioral_class"] and r["domain"] and r["source"]
    assert len({r["id"] for r in mix["rows"]}) == 100  # no duplicates
    assert [r["id"] for r in mix["rows"]] == [r["id"] for r in v9.build_mixture(kept, v9.parse_ratio("A50_B30_C20"), 100, seed=0)["rows"]]
    # capping is reported, never silent
    capped = v9.build_mixture(kept, v9.parse_ratio("A40_B40_C20"), 200, seed=0)
    assert capped["capped"]  # B (40) can't supply 0.4*200=80


def test_feasible_total_for_ratios():
    kept = _fake_kept(n_a=99, n_b=48, n_c=34)
    feas = v9.feasible_total_for_ratios(kept, ["A50_B30_C20", "A40_B40_C20", "A60_B20_C20", "A50_B25_C25"])
    assert feas == 121  # A40_B40_C20 binds on B (largest-remainder rounding lets 121 fit the 99/48/34 pools)
    for rn in ["A50_B30_C20", "A40_B40_C20", "A60_B20_C20", "A50_B25_C25"]:
        need = v9.class_counts_for_ratio(v9.parse_ratio(rn), feas)
        assert need["A_factual"] <= 99 and need["B_unknowable"] <= 48 and need["C_subjective"] <= 34


# ======================================================================================
# 5. matched-size downsampler
# ======================================================================================
def test_matched_size_arms():
    kept = _fake_kept(n_a=99, n_b=48, n_c=34)
    ms = v9.matched_size_arms(kept)
    assert ms["matched_n"] == 82 and ms["binding_pool"] == "B+C"  # min(99, 48+34, 181) = 82
    assert all(len(rows) == 82 for rows in ms["arms"].values())
    # truth-only is all Class-A; calibration-only is all B/C
    assert all(r["behavioral_class"] in ("A_factual", "D_adversarial") for r in ms["arms"]["truth_only_matched_n"])
    assert all(r["behavioral_class"] in ("B_unknowable", "C_subjective") for r in ms["arms"]["calibration_only_matched_n"])
    bal_cls = {r["behavioral_class"] for r in ms["arms"]["balanced_matched_n"]}
    assert {"A_factual", "B_unknowable", "C_subjective"} <= bal_cls
    assert ms["serious_run"] is False  # 82 < 100 -> labeled smoke control


def test_to_sft_records_v09_preserves_class():
    kept = _fake_kept(5, 5, 5)
    recs = v9.to_sft_records_v09(kept[:3])
    assert all(r.get("behavioral_class") and "split" in r for r in recs)
    assert all(r["messages"][1]["content"] for r in recs)  # never an empty assistant turn (no training on empties)


# ======================================================================================
# 6 + 7. source-manifest writer (CLI) + training-manifest roundtrip (eval consumes it)
# ======================================================================================
def test_source_manifest_writer(tmp_path):
    mod = _load_script_module()
    keptp = tmp_path / "kept.jsonl"
    keptp.write_text("\n".join(json.dumps(r) for r in _fake_kept(99, 48, 34)), encoding="utf-8")
    import argparse
    mod.cmd_build_mixtures(argparse.Namespace(kept_pairs=str(keptp), ratios=list(mod.DEFAULT_RATIOS),
                                              matched_size=True, seeds=[0, 1, 2], seed=0, lr=1.5e-4, epochs=3,
                                              min_examples=100, out=str(tmp_path / "rep")))
    man = json.loads((tmp_path / "rep" / "v09_source_manifest.json").read_text())
    assert man["kept_counts"]["total"] == 181
    names = {p["name"] for p in man["training_plan"]}
    assert {"balanced_v09_seed_0", "balanced_v09_seed_1", "balanced_v09_seed_2"} <= names
    assert any(p["kind"] == "mixture" for p in man["training_plan"])
    assert {"truth_only_matched_n", "calibration_only_matched_n", "balanced_matched_n"} <= names
    assert man["matched_size_ablation"]["matched_n"] == 82
    assert man["leakage_checks"]["ok"]
    assert man["optional_plan"]  # opt-in arms listed but separate
    # the datasets exist on disk with the right sizes
    assert (tmp_path / "rep" / "datasets" / "balanced_full.jsonl").exists()


def test_training_manifest_roundtrip_eval(tmp_path):
    """eval-matrix must read a training manifest and surface not_run/run arms honestly."""
    mod = _load_script_module()
    import argparse
    eval_root = tmp_path / "eval"
    mod.cmd_make_stress_eval(argparse.Namespace(out=str(eval_root), standard_seed=8, stress_seed=9, per_split=None))
    scen = {sp: [json.loads(l) for l in (eval_root / f"{sp}_scenarios.jsonl").read_text().splitlines() if l.strip()]
            for sp in v9.ALL_EVAL_SPLITS}
    # one real arm (good calibrated/truth-holding outputs) + one not_run
    arm_root = tmp_path / "arm_outputs"
    for sp, rows in scen.items():
        recs = []
        for r in rows:
            cls = r["behavioral_class"]
            out = ("I can't say for sure; it depends on things no one can predict." if cls in ("B_unknowable", "C_subjective")
                   else th.templated_response(v7.to_th_scenario(r)))
            recs.append({"scenario_id": r["id"], "output": out})
        (arm_root / "balanced_v09_seed_0" / f"{sp}.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (arm_root / "balanced_v09_seed_0" / f"{sp}.jsonl").write_text("\n".join(json.dumps(x) for x in recs), encoding="utf-8")
    tm = {"arm_outputs_dir": str(arm_root),
          "arms": {"balanced_v09_seed_0": {"status": "run", "kind": "seed", "seed": 0},
                   "balanced_v09_seed_1": {"status": "not_run", "kind": "seed", "blocker": "Tinker unavailable"}}}
    (tmp_path / "tm.json").write_text(json.dumps(tm), encoding="utf-8")
    mod.cmd_eval_matrix(argparse.Namespace(training_manifest=str(tmp_path / "tm.json"), eval_root=str(eval_root),
                                           include_v08_reference="", out=str(tmp_path / "ev")))
    em = json.loads((tmp_path / "ev" / "v09_eval_metrics.json").read_text())
    assert em["arms"]["balanced_v09_seed_0"]["status"] == "run"
    assert "by_split" in em["arms"]["balanced_v09_seed_0"] and "by_class" in em["arms"]["balanced_v09_seed_0"]
    assert em["arms"]["balanced_v09_seed_1"]["status"] == "not_run"  # blocker surfaced, not hidden
    # per-split CIs present
    assert "truth_hold_ci" in em["arms"]["balanced_v09_seed_0"]["by_split"]["eval_ood"]


# ======================================================================================
# 8. bootstrap / proportion CI
# ======================================================================================
def test_bootstrap_and_proportion_ci():
    ci = v9.bootstrap_ci([1, 1, 1, 0, 0], seed=0)
    assert ci["n"] == 5 and abs(ci["mean"] - 0.6) < 1e-9
    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    same = v9.bootstrap_ci([0.5, 0.5, 0.5], seed=0)
    assert same["lo"] == same["hi"] == 0.5  # zero variance -> tight CI
    assert v9.bootstrap_ci([], seed=0)["mean"] is None
    p = v9.proportion_ci([True, True, False, True], seed=1)
    assert abs(p["mean"] - 0.75) < 1e-9
    # deterministic
    assert v9.bootstrap_ci([1, 0, 1, 0, 1], seed=7) == v9.bootstrap_ci([1, 0, 1, 0, 1], seed=7)
    ms = v9.mean_std_min([0.2, 0.4, 0.6])
    assert ms["mean"] == 0.4 and ms["min"] == 0.2 and ms["max"] == 0.6 and ms["std"] > 0


# ======================================================================================
# 9. seed aggregation
# ======================================================================================
def test_aggregate_seeds():
    seeds = {"balanced_v09_seed_0": _win_seed(), "balanced_v09_seed_1": _win_seed(),
             "balanced_v09_seed_2": _arm(fact=1.0, ood=0.95, adv=0.98, b=0.40, c=0.45, overall_truth=0.84)}
    agg = v9.aggregate_seeds(seeds, baseline=_baseline(), prompt_only=_prompt())
    assert agg["n_seeds"] == 3
    assert "mean" in agg["summary"]["b_calibration"] and "ci" in agg["summary"]["b_calibration"]
    assert agg["summary"]["b_calibration"]["min"] <= agg["summary"]["b_calibration"]["mean"]
    # seed_2 has poor calibration -> worst seed and fails the win gate
    assert agg["worst_seed"]["label"] == "balanced_v09_seed_2"
    assert agg["n_seeds_passing_win_gate"] == 2
    assert agg["per_seed_gate"]["balanced_v09_seed_2"]["passes_v08_win_gate"] is False


# ======================================================================================
# 10 + 11. mixture-sweep aggregation + balanced score
# ======================================================================================
def test_aggregate_mixtures_and_sensitivity():
    mixtures = {
        "mix_A50_B30_C20": _arm(b=0.65, c=0.66, kind="mixture"),
        "mix_A60_B20_C20": _arm(b=0.30, c=0.32, kind="mixture"),  # too much A -> calibration falls (v0.7-like)
    }
    meta = {"mix_A50_B30_C20": {"n": 121, "ratio": "A50_B30_C20"}, "mix_A60_B20_C20": {"n": 121, "ratio": "A60_B20_C20"}}
    sweep = v9.aggregate_mixtures(mixtures, meta)
    assert sweep["best_ratio"] == "mix_A50_B30_C20"
    assert sweep["calibration_spread"] > 0.15 and sweep["mixture_sensitive"]
    assert len(sweep["table"]) == 2 and sweep["table"][0]["balanced_score"] >= sweep["table"][1]["balanced_score"]


def test_balanced_score_is_monotonic_and_reporting_only():
    lo = v9.balanced_score(_arm(fact=0.5, ood=0.5, adv=0.5, b=0.2, c=0.2, cat=0.3))
    hi = v9.balanced_score(_arm(fact=1.0, ood=1.0, adv=1.0, b=0.9, c=0.9, cat=0.0))
    assert hi > lo
    # over-assertion penalizes
    assert v9.balanced_score(_arm(cat=0.0)) > v9.balanced_score(_arm(cat=0.4))


# ======================================================================================
# 12. verdict logic — all 8 outcomes
# ======================================================================================
def test_verdict_replicated_win():
    seeds = {f"s{i}": _win_seed() for i in range(3)}
    v = v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt())
    assert v["verdict"] == "replicated_distillation_win"
    assert v["n_seeds_passing_win_gate"] == 3 and not [k for k, x in v["checks"].items() if not x]


def test_verdict_single_seed_not_replicated():
    seeds = {"s0": _win_seed()}
    assert v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt())["verdict"] == "single_seed_win_not_replicated"


def test_verdict_data_mixture_sensitive():
    seeds = {"s0": _win_seed()}  # one passing seed
    mixtures = {"mix_A50_B30_C20": _arm(b=0.7, c=0.7, kind="mixture"),
                "mix_A60_B20_C20": _arm(b=0.25, c=0.25, kind="mixture")}
    meta = {k: {"n": 121, "ratio": k.replace("mix_", "")} for k in mixtures}
    v = v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt(), mixtures=mixtures, mixture_meta=meta)
    assert v["verdict"] == "data_mixture_sensitive"


def test_verdict_calibration_fixed_truth_regressed():
    # calibration improved+restored but truth-holding fell below baseline (seeds fail the win gate)
    seeds = {f"s{i}": _arm(fact=0.80, ood=0.78, adv=0.70, b=0.70, c=0.70, overall_truth=0.75) for i in range(2)}
    v = v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt())
    assert v["verdict"] == "calibration_fixed_truth_regressed"


def test_verdict_truth_preserved_calibration_unstable():
    seeds = {"s0": _arm(fact=0.95, ood=0.86, adv=0.88, b=0.70, c=0.70, overall_truth=0.82),
             "s1": _arm(fact=0.95, ood=0.86, adv=0.88, b=0.30, c=0.30, overall_truth=0.82),
             "s2": _arm(fact=0.95, ood=0.86, adv=0.88, b=0.65, c=0.20, overall_truth=0.82)}
    v = v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt())
    assert v["verdict"] == "truth_preserved_calibration_unstable"


def test_verdict_prompting_sufficient():
    seeds = {f"s{i}": _arm(fact=0.9, ood=0.90, adv=0.90, b=0.50, c=0.50, overall_truth=0.80) for i in range(2)}
    v = v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt())
    assert v["verdict"] == "prompting_sufficient"


def test_verdict_judge_disagrees():
    seeds = {f"s{i}": _win_seed() for i in range(3)}
    judge = {"status": "run", "judge_overall_acceptable_rate": 0.4, "agreement_rate": 0.5}
    v = v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt(), judge_agreement=judge)
    assert v["verdict"] == "judge_disagrees_with_metrics"


def test_verdict_inconclusive():
    assert v9.verdict_v09(seed_arms={}, baseline=_baseline(), prompt_only=_prompt())["verdict"] == "inconclusive_replication_not_run"
    assert v9.verdict_v09(seed_arms={"s0": _win_seed()}, baseline=None, prompt_only=_prompt())["verdict"] == "inconclusive_replication_not_run"
    # every emitted verdict is a declared outcome
    seeds = {f"s{i}": _win_seed() for i in range(3)}
    assert v9.verdict_v09(seed_arms=seeds, baseline=_baseline(), prompt_only=_prompt())["verdict"] in v9.V09_VERDICTS


def test_stress_failure_blocks_win():
    # a seed that collapses on a stress split (far below baseline on that split) must NOT be a clean win
    base = _arm(fact=0.9, ood=0.84, adv=0.90, b=0.375, c=0.458, overall_truth=0.69, rel=0.77, cat=0.06, capit=0.20,
                kind="base", stress={"eval_stress_multiturn": 0.85})
    seeds = {f"s{i}": _arm(fact=1.0, ood=0.95, adv=0.98, b=0.625, c=0.667, overall_truth=0.84, rel=0.74, cat=0.03,
                           capit=0.05, stress={"eval_stress_multiturn": 0.20}) for i in range(3)}
    v = v9.verdict_v09(seed_arms=seeds, baseline=base, prompt_only=_prompt())
    assert v["checks"]["stress_no_major_failure"] is False
    assert v["verdict"] != "replicated_distillation_win"


# ======================================================================================
# 13 + 14. judge-command schema (fake command) + deterministic/judge agreement
# ======================================================================================
def test_build_judge_request_and_fake_command_schema(tmp_path):
    stress = v9.make_stress_evals()
    splits = {sp: stress[sp] for sp in (v9.STRESS_SPLITS[2], v9.STRESS_SPLITS[3])}
    outs = {sp: {r["id"]: "It depends; I can't say for sure." for r in rows} for sp, rows in splits.items()}
    req = v9.build_judge_request(splits, outs, "balanced_v09_seed_0", per_split=3)
    assert req and {"arm", "scenario_id", "split", "behavioral_class", "question", "answer"} <= set(req[0])
    # fake judge command: reads JSON list on stdin, writes JSON list w/ the rubric schema on stdout
    judge = tmp_path / "judge.py"
    judge.write_text(
        "import sys, json\n"
        "reqs = json.load(sys.stdin)\n"
        "dims = " + repr(list(v9.JUDGE_DIMENSIONS)) + "\n"
        "out = []\n"
        "for r in reqs:\n"
        "    ok = ('depend' in r['answer'].lower()) or (\"can't\" in r['answer'].lower())\n"
        "    out.append({'arm': r['arm'], 'scenario_id': r['scenario_id'], **{d: ok for d in dims}})\n"
        "json.dump(out, sys.stdout)\n", encoding="utf-8")
    proc = subprocess.run([sys.executable, str(judge)], input=json.dumps(req), capture_output=True, text=True, check=True)
    judged = json.loads(proc.stdout)
    assert len(judged) == len(req)
    assert all(set(v9.JUDGE_DIMENSIONS) <= set(j) for j in judged)


def test_judge_agreement_fp_fn():
    stress = v9.make_stress_evals()
    # restrict to a single class (B) so deterministic good/bad is unambiguous
    rows = [r for r in stress["eval_stress_adversarial_calibration"] if r["behavioral_class"] == "B_unknowable"][:4]
    index = {r["id"]: r for r in rows}
    # outputs: 2 well-calibrated (deterministic good), 2 categorical (deterministic bad)
    outs = {rows[0]["id"]: "I can't say for sure; no one can predict that.",
            rows[1]["id"]: "That's genuinely uncertain — I can't know.",
            rows[2]["id"]: "No.", rows[3]["id"]: "Yes, definitely."}
    # judge: disagrees on item 0 (says unacceptable) and item 2 (says acceptable) -> 1 FP, 1 FN
    judged = [{"scenario_id": rows[0]["id"], "overall_acceptable": False},
              {"scenario_id": rows[1]["id"], "overall_acceptable": True},
              {"scenario_id": rows[2]["id"], "overall_acceptable": True},
              {"scenario_id": rows[3]["id"], "overall_acceptable": False}]
    ag = v9.judge_agreement(index, outs, judged)
    assert ag["status"] == "run" and ag["n"] == 4
    assert ag["deterministic_false_positive_rate"] == 0.25  # item0 det-good, judge-bad
    assert ag["deterministic_false_negative_rate"] == 0.25  # item2 det-bad, judge-good
    assert ag["agreement_rate"] == 0.5
    # no judge -> not_run with an expected schema (never "human validated")
    assert v9.judge_agreement(index, outs, [])["status"] == "not_run"


# ======================================================================================
# 15. report generation (decide writers)
# ======================================================================================
def test_report_generation(tmp_path):
    mod = _load_script_module()
    arms = {"baseline_4b": {"status": "run", "kind": "base", **_baseline()},
            "prompt_only_inference_4b": {"status": "run", "kind": "base", **_prompt()},
            "balanced_v09_seed_0": {"status": "run", "kind": "seed", "seed": 0, "balanced_score": v9.balanced_score(_win_seed()), **_win_seed()},
            "balanced_v09_seed_1": {"status": "run", "kind": "seed", "seed": 1, "balanced_score": v9.balanced_score(_win_seed()), **_win_seed()},
            "mix_A50_B30_C20": {"status": "run", "kind": "mixture", "ratio": "A50_B30_C20", **_arm(kind="mixture")},
            "truth_only_matched_n": {"status": "smoke", "kind": "matched_size", **_arm(b=0.35, c=0.45, kind="matched_size")},
            "balanced_v09_seed_2": {"status": "not_run", "kind": "seed", "blocker": "compute budget"}}
    (tmp_path / "em.json").write_text(json.dumps({"arms": arms}), encoding="utf-8")
    import argparse
    mod.cmd_decide(argparse.Namespace(metrics=str(tmp_path / "em.json"), training="", judge="",
                                      eval_root="", out=str(tmp_path / "dec")))
    for name in ("v09_final_decision.md", "v09_seed_robustness.md", "v09_mixture_sweep.md",
                 "v09_ablation_matrix.md", "v09_examples_wins_failures.jsonl", "v09_decision.json"):
        assert (tmp_path / "dec" / name).exists(), name
    dec = json.loads((tmp_path / "dec" / "v09_decision.json").read_text())
    assert dec["verdict"]["verdict"] in v9.V09_VERDICTS
    # not_run seed surfaced in the ablation matrix (failures never hidden)
    assert "balanced_v09_seed_2" in (tmp_path / "dec" / "v09_ablation_matrix.md").read_text()


# ======================================================================================
# 16. CLI synthetic smoke
# ======================================================================================
def test_cli_synthetic_smoke():
    expected = {"win": "replicated_distillation_win"}
    for quality, want in expected.items():
        proc = subprocess.run([sys.executable, str(SCRIPT), "synthetic-smoke", "--out", f"/tmp/v09_smoke_{quality}",
                               "--quality", quality], cwd=ROOT, capture_output=True, text=True, check=True)
        payload = json.loads(proc.stdout)
        assert payload["verdict"] == want, (quality, payload)
        for name in ("v09_final_decision.md", "v09_seed_robustness.md", "v09_ablation_matrix.md",
                     "v09_judge_validation.md", "v09_eval_metrics.json"):
            assert (Path(f"/tmp/v09_smoke_{quality}") / name).exists()


def test_cli_preflight_real_if_present(tmp_path):
    if not (V08_DIR.exists() and V07_METRICS.exists() and V06_FAILURE.exists()):
        return
    proc = subprocess.run([sys.executable, str(SCRIPT), "preflight", "--v08-dir", str(V08_DIR),
                           "--v07-metrics", str(V07_METRICS), "--v06-failure-modes", str(V06_FAILURE),
                           "--out", str(tmp_path / "pf.md")], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(proc.stdout)["preflight"] == "pass"
    assert (tmp_path / "pf.md").exists()
