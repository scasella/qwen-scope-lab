"""Tests for v0.7 stronger-teacher distillation. CI-safe: no Tinker/Modal/CUDA/MLX/network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_distill_v07 as v7

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truth_holding_distill_v07.py"
V06 = ROOT / "reports" / "steering_distill" / "th_v06_27b_showdown" / "teacher_showdown_metrics_v06.json"


# 2/3. Scenario generator validity, split separation, no fact leak
def test_scenarios_valid_and_no_leak():
    sp = v7.make_scenarios(n_train=120, n_dev=20, n_eval_id=20, n_eval_ood=40, n_eval_ambiguous=30, n_eval_adversarial=40)
    assert len(sp["train"]) == 120 and len(sp["eval_ood"]) > 0 and len(sp["eval_ambiguous"]) > 0
    # required schema fields present
    r = sp["train"][0]
    assert {"id", "split", "domain", "question", "false_claim", "acceptable_answer_patterns", "false_answer_patterns", "pressure_type"} <= set(r)
    # no fact_key appears in two splits (held-out facts never leak into train)
    keys = {k: {row["fact_key"] for row in v} for k, v in sp.items()}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (keys[names[i]] & keys[names[j]]), (names[i], names[j])
    # ids unique
    ids = [row["id"] for v in sp.values() for row in v]
    assert len(ids) == len(set(ids))
    # deterministic
    sp2 = v7.make_scenarios(n_train=120, n_dev=20, n_eval_id=20, n_eval_ood=40, n_eval_ambiguous=30, n_eval_adversarial=40)
    assert [r["id"] for r in sp["train"]] == [r["id"] for r in sp2["train"]]


def test_domain_balance_under_gate():
    sp = v7.make_scenarios(n_train=160)
    from collections import Counter
    c = Counter(r["domain"] for r in sp["train"])
    assert max(c.values()) / len(sp["train"]) <= 0.35  # domain balance gate


# 4. teacher-output schema loader + adapter
def test_to_th_scenario_adapter():
    sp = v7.make_scenarios(n_train=10, n_dev=2, n_eval_id=2, n_eval_ood=4, n_eval_ambiguous=6, n_eval_adversarial=4)
    scn = v7.to_th_scenario(sp["train"][0])
    assert scn.id and scn.question and scn.false_claim
    amb = v7.to_th_scenario(sp["eval_ambiguous"][0])
    assert amb.ambiguous is True


# 5. source audit gate
def _excellent_outputs(rows):
    return {r["id"]: th.templated_response(v7.to_th_scenario(r)) for r in rows}


def test_audit_gate_passes_excellent_and_blocks_small_and_low():
    sp = v7.make_scenarios(n_train=160)
    au = v7.audit_source(sp["train"], _excellent_outputs(sp["train"]), source="stronger_instruction_teacher_9b")
    elig = v7.training_eligibility(au["metrics"])
    assert au["metrics"]["kept_rate"] >= 0.6 and au["metrics"]["n_kept"] >= 100
    assert elig["eligible"] is True and elig["status"] == "eligible"
    # too few examples -> smoke_only
    small = sp["train"][:20]
    elig_small = v7.training_eligibility(v7.audit_source(small, _excellent_outputs(small), source="t")["metrics"])
    assert elig_small["status"] == "smoke_only" and elig_small["eligible"] is False
    # low kept-rate (capitulating outputs) -> blocked
    caps = {r["id"]: th.capitulation_example(v7.to_th_scenario(r)) for r in sp["train"]}
    elig_low = v7.training_eligibility(v7.audit_source(sp["train"], caps, source="t")["metrics"])
    assert elig_low["status"] == "blocked" and elig_low["eligible"] is False


def test_templated_excluded_from_gate():
    sp = v7.make_scenarios(n_train=160)
    m = v7.audit_source(sp["train"], _excellent_outputs(sp["train"]), source="templated_oracle", is_templated=True)["metrics"]
    elig = v7.training_eligibility(m)
    assert elig["eligible"] is False and elig["status"] == "templated_control_excluded"


# 6. phrase/template concentration warning (templated outputs share trigrams -> gate warns)
def test_phrase_concentration_warns():
    sp = v7.make_scenarios(n_train=120)
    rows = sp["train"]
    m = v7.audit_source(rows, _excellent_outputs(rows), source="t")["metrics"]  # templated -> shared trigrams, all kept
    elig = v7.training_eligibility(m)
    assert any("template trigram" in w for w in elig["warns"])  # template-domination flagged


# 7/8. SFT + preference export schema
def test_exports_schema():
    sp = v7.make_scenarios(n_train=40)
    au = v7.audit_source(sp["train"], _excellent_outputs(sp["train"]), source="stronger_instruction_teacher_9b")
    sft = v7.to_sft_records(au["kept"])
    pref = v7.to_preference_records(au["kept"])
    assert sft and all([m["role"] for m in r["messages"]] == ["user", "assistant"] for r in sft)
    assert all(r["source"] == "stronger_instruction_teacher_9b" and "domain" in r for r in sft)
    assert pref and all(set(r) >= {"prompt", "chosen", "rejected", "source", "domain"} for r in pref)


# 10. eval metric aggregation by split/domain/pressure
def test_eval_aggregation_by_split():
    sp = v7.make_scenarios(n_train=10, n_eval_id=10, n_eval_ood=20, n_eval_ambiguous=20, n_eval_adversarial=20)
    splits = {k: sp[k] for k in ("eval_id", "eval_ood", "eval_ambiguous", "eval_adversarial")}
    good = {k: {r["id"]: th.templated_response(v7.to_th_scenario(r)) for r in v} for k, v in splits.items()}
    ev = v7.evaluate_arm(splits, good)
    assert ev["overall"]["n"] > 0 and ev["overall"]["truth_hold_rate"] >= 0.8
    assert ev["by_split"]["eval_ambiguous"]["ambiguous_case_calibration"] is not None


# 11. verdict logic — all six outcomes
def _arms(base_th, po_th, dist_overall, dist_by_split=None, dist_status="run"):
    base = {"status": "run", "overall": {"truth_hold_rate": base_th, "correctness_rate": 0.5, "capitulation_rate": 0.6,
            "politeness_rate": 1.0, "relevance": 0.8, "repetition": 0.0, "genericness": 0.0, "collapse_rate": 0.0},
            "by_split": {"eval_ood": {"n": 20, "truth_hold_rate": base_th}, "eval_ambiguous": {"n": 20, "ambiguous_case_calibration": 0.5, "overconfidence_rate": 0.2}}}
    po = {"status": "run", "overall": {"truth_hold_rate": po_th, "capitulation_rate": 0.3}, "by_split": {"eval_ood": {"n": 20, "truth_hold_rate": po_th}}}
    dist = {"status": dist_status}
    if dist_status == "run":
        dist["overall"] = dist_overall
        dist["by_split"] = dist_by_split or {"eval_ood": {"n": 20, "truth_hold_rate": dist_overall["truth_hold_rate"]}, "eval_ambiguous": {"n": 20, "ambiguous_case_calibration": 0.6, "overconfidence_rate": 0.1}}
    return {"baseline_4b": base, "prompt_only_inference_4b": po, "distilled_4b_from_9b_teacher": dist}


_GOOD = {"truth_hold_rate": 0.8, "correctness_rate": 0.85, "capitulation_rate": 0.2, "politeness_rate": 1.0, "relevance": 0.8, "repetition": 0.0, "genericness": 0.0, "collapse_rate": 0.0}


def test_verdict_distillation_win():
    # distilled strong everywhere; prompt-only weak OOD -> complements -> win
    arms = _arms(0.3, 0.6, _GOOD, dist_by_split={"eval_ood": {"n": 20, "truth_hold_rate": 0.78}, "eval_ambiguous": {"n": 20, "ambiguous_case_calibration": 0.6, "overconfidence_rate": 0.1}})
    arms["prompt_only_inference_4b"]["by_split"]["eval_ood"]["truth_hold_rate"] = 0.4
    v = v7.distillation_verdict(arms, eligibility={"eligible": True}, n_train_kept=150)
    assert v["verdict"] == "distillation_win"


def test_verdict_prompting_sufficient():
    # true tie: distilled overall == prompt-only overall, equal OOD -> no axis beats -> prompting sufficient
    arms = _arms(0.3, 0.8, _GOOD)
    arms["prompt_only_inference_4b"]["overall"] = dict(_GOOD)
    arms["prompt_only_inference_4b"]["by_split"]["eval_ood"]["truth_hold_rate"] = _GOOD["truth_hold_rate"]
    v = v7.distillation_verdict(arms, eligibility={"eligible": True}, n_train_kept=150)
    assert v["verdict"] == "prompting_sufficient"


def test_verdict_source_good_training_failed():
    weak = dict(_GOOD, truth_hold_rate=0.3, capitulation_rate=0.6)  # didn't beat baseline
    arms = _arms(0.3, 0.6, weak)
    v = v7.distillation_verdict(arms, eligibility={"eligible": True}, n_train_kept=150)
    assert v["verdict"] == "source_good_training_failed"


def test_verdict_negative_overfit():
    arms = _arms(0.3, 0.6, _GOOD, dist_by_split={"eval_ood": {"n": 20, "truth_hold_rate": 0.25}, "eval_ambiguous": {"n": 20, "ambiguous_case_calibration": 0.6, "overconfidence_rate": 0.1}})
    v = v7.distillation_verdict(arms, eligibility={"eligible": True}, n_train_kept=150)
    assert v["verdict"] == "negative_overfit_or_regression"


def test_verdict_training_not_run_and_inconclusive():
    not_run = v7.distillation_verdict(_arms(0.3, 0.6, _GOOD, dist_status="not_run"), eligibility={"eligible": True}, n_train_kept=150)
    assert not_run["verdict"] == "training_not_run_source_ready"
    small = v7.distillation_verdict(_arms(0.3, 0.6, _GOOD), eligibility={"eligible": False, "status": "smoke_only"}, n_train_kept=40)
    assert small["verdict"] == "inconclusive_small_data"


# 1/13. CLI: preflight + synthetic smoke
def test_cli_preflight_and_smoke(tmp_path):
    if V06.exists():
        proc = subprocess.run([sys.executable, str(SCRIPT), "preflight", "--v06-report", str(V06)], cwd=ROOT, capture_output=True, text=True, check=True)
        assert json.loads(proc.stdout)["preflight"] == "pass"
    out = tmp_path / "smoke"
    proc = subprocess.run([sys.executable, str(SCRIPT), "synthetic-smoke", "--out", str(out)], cwd=ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["verdict"] in v7.VERDICTS
    for name in ("v07_eval_metrics.json", "v07_eval_truth_holding.md", "v07_final_decision.md", "v07_examples_wins_failures.jsonl"):
        assert (out / name).exists()


# 9 / 12. training manifest statuses (gate logic) + report generation
def test_training_gate_statuses():
    # eligible -> serious; smoke_only -> labeled smoke; blocked -> no train
    sp = v7.make_scenarios(n_train=160)
    big = v7.training_eligibility(v7.audit_source(sp["train"], _excellent_outputs(sp["train"]), source="t")["metrics"])
    assert big["allow_smoke"] is True and big["eligible"] is True


def test_report_render():
    sp = v7.make_scenarios(n_train=40)
    m = v7.audit_source(sp["train"], _excellent_outputs(sp["train"]), source="t")["metrics"]
    # render via the CLI helper indirectly: ensure metrics carry the fields the report needs
    assert "domain_breakdown" in m and "pressure_breakdown" in m and "phrase_concentration" in m
