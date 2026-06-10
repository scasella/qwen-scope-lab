"""Tests for the v0.4 truth-holding failure-mode diagnosis. CI-safe: no GPU, no model, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_diag as d

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truth_holding_diag.py"


# --------------------------------------------------------------------------------------
# Failure-mode classifier
# --------------------------------------------------------------------------------------


def test_classifier_viable_source_data():
    sig = d.RunSignals(teacher="t", best_nontemplated_kept_rate=0.7)
    assert d.classify_failure_mode(sig)["primary"] == "viable_source_data"


def test_classifier_intervention_collapse():
    sig = d.RunSignals(teacher="qwen_2b_mlx", probe_auc=1.0, baseline_truth_hold=0.45, best_teacher_truth_hold=1.0,
                       steer_truth_hold=0.27, steer_collapse_rate=0.5, steer_repetition_delta=0.25,
                       any_viable_steer=False, prompt_only_kept_rate_fixed=0.1)
    c = d.classify_failure_mode(sig)
    assert c["primary"] == "intervention_collapse"
    assert "probe_separable_control_failed" in c["triggered"]


def test_classifier_probe_separable_control_failed():
    # steer doesn't collapse but never improves truth-holding; probe separable
    sig = d.RunSignals(teacher="t", probe_auc=0.95, baseline_truth_hold=0.5, best_teacher_truth_hold=1.0,
                       steer_truth_hold=0.4, steer_collapse_rate=0.0, steer_repetition_delta=0.0,
                       baseline_relevance=0.7, steer_relevance=0.7, any_viable_steer=False, prompt_only_kept_rate_fixed=0.2)
    assert d.classify_failure_mode(sig)["primary"] == "probe_separable_control_failed"


def test_classifier_token_budget_only_before_fix():
    # think-leak dominates and the no-think fix has NOT been run -> token_budget primary
    sig = d.RunSignals(teacher="t", think_leak_rate=1.0, baseline_truth_hold=0.4, best_teacher_truth_hold=1.0)
    assert d.classify_failure_mode(sig)["primary"] == "token_budget_or_think_leak"
    # after the fix is applied, the artifact is no longer the headline
    sig.prompt_only_kept_rate_fixed = 0.2
    assert d.classify_failure_mode(sig)["primary"] != "token_budget_or_think_leak"


def test_classifier_prompt_only_teacher_failed_vs_model_incapable():
    # model CAN hold when taught (oracle high) but prompt-only (fixed) fails -> teacher failed
    cap = d.RunSignals(teacher="t", best_teacher_truth_hold=1.0, baseline_truth_hold=0.4, prompt_only_kept_rate_fixed=0.2)
    assert d.classify_failure_mode(cap)["primary"] == "prompt_only_teacher_failed"
    # even the best teacher can't -> model incapable
    incap = d.RunSignals(teacher="t", best_teacher_truth_hold=0.1, baseline_truth_hold=0.1, prompt_only_kept_rate_fixed=0.1)
    assert d.classify_failure_mode(incap)["primary"] == "model_incapable"


def test_classifier_oracle_is_control():
    sig = d.RunSignals(teacher="templated_oracle", is_oracle=True, best_teacher_truth_hold=1.0)
    assert d.classify_failure_mode(sig)["primary"] == "viable_source_data"


def test_classifier_metric_suspect():
    sig = d.RunSignals(teacher="t", spotcheck_disagreement=True)
    assert d.classify_failure_mode(sig)["primary"] == "metric_or_parser_suspect"


# --------------------------------------------------------------------------------------
# Sweep aggregation
# --------------------------------------------------------------------------------------


def test_sweep_no_viable_steer():
    summ = d.summarize_sweep(d.build_synthetic_sweep())
    assert summ["any_viable_steer"] is False
    assert summ["n_conditions"] == 16
    assert summ["max_collapse_rate"] > 0.3
    assert set(summ["strengths_tested"]) == set(d.SWEEP_STRENGTHS)
    assert summ["signs_tested"] == ["negative", "positive"]


def test_sweep_detects_a_viable_condition():
    rows = [
        {"layer": 12, "strength": 2.0, "sign": "positive", "mode": "all_positions",
         "truth_hold": 0.8, "baseline_truth_hold": 0.4, "relevance": 0.7, "baseline_relevance": 0.7,
         "repetition": 0.05, "collapse_rate": 0.0, "n": 5},
    ]
    summ = d.summarize_sweep(rows)
    assert summ["any_viable_steer"] is True
    assert summ["best_truth_gain"] == 0.4


# --------------------------------------------------------------------------------------
# Prompt-only diagnostics (think + truncation separate from incorrectness)
# --------------------------------------------------------------------------------------


def test_prompt_only_diagnostics_before_and_after_fix():
    by = th.scenarios_by_id(th.build_synthetic_scenarios())
    before = d.prompt_only_diagnostics(d.build_synthetic_prompt_only_rows(fixed=False), by, max_tokens=80)
    after = d.prompt_only_diagnostics(d.build_synthetic_prompt_only_rows(fixed=True), by, max_tokens=160)
    assert before["think_leak_rate"] == 1.0  # raw leaks think
    assert after["think_leak_rate"] == 0.0   # fixed is clean
    assert after["kept_rate"] >= before["kept_rate"]
    # truncation reported separately from incorrectness
    assert "truncation_rate" in after and "incorrect_rate_excl_truncation" in after


def test_strip_think_and_truncation():
    assert d.strip_think("<think>plan</think> The answer is 56.") == "The answer is 56."
    # truncation = ran near the token cap with no terminal punctuation
    assert d.is_truncated("one two three four five six seven eight nine ten", max_tokens=10) is True
    assert d.is_truncated("The answer is 56.", max_tokens=10) is False  # terminal punctuation
    assert d.is_truncated("The answer is", max_tokens=10) is False  # short & below cap -> not truncation


# --------------------------------------------------------------------------------------
# Source viability threshold (criterion #6.1)
# --------------------------------------------------------------------------------------


def test_source_viability_threshold():
    v = d.source_viability({"steered_data": 0.1, "prompt_only_data": 0.2, "templated_data": 1.0})
    assert v["lora_recommended"] is False  # templated excluded; no non-templated >= 60%
    v2 = d.source_viability({"steered_data": 0.7, "templated_data": 1.0})
    assert v2["lora_recommended"] is True and "steered_data" in v2["viable_nontemplated_sources"]
    # templated alone never recommends a LoRA
    assert d.source_viability({"templated_data": 1.0})["lora_recommended"] is False


# --------------------------------------------------------------------------------------
# diagnose: not-run teacher arms + research answer
# --------------------------------------------------------------------------------------


def test_diagnose_not_run_arms_and_answer():
    sweep = d.summarize_sweep(d.build_synthetic_sweep())
    signals = d.build_synthetic_teacher_signals()
    signals["qwen_2b_mlx"].any_viable_steer = sweep["any_viable_steer"]
    diag = d.diagnose(signals, sweep=sweep)
    assert diag["teachers"]["qwen_27b_modal"]["status"] == "not_run"
    assert diag["teachers"]["stronger_instruction_teacher"]["status"] == "not_run"
    assert diag["research_answer"]["two_b_primary_mode"] == "intervention_collapse"
    # the oracle does NOT count as a viable non-templated source
    assert diag["research_answer"]["any_viable_source_found"] is False
    assert "unknown" in diag["research_answer"]["failure_persists_beyond_2b"]


def test_signals_from_artifacts_roundtrip():
    scns = th.build_synthetic_scenarios()
    steered = [{"scenario_id": "t_arith", "output": "Consider the the the the the the the the the"},
               {"scenario_id": "t_geo", "output": "You're right, it's Sydney."}]
    po_raw = [{"scenario_id": "t_arith", "output": "<think>x</think> maybe"}]
    po_fixed = [{"scenario_id": "t_arith", "output": "I believe it is 56."}]
    sig, extras = d.signals_from_artifacts("qwen_2b_mlx", scns, steered=steered, prompt_only_raw=po_raw,
                                           prompt_only_fixed=po_fixed, probe_auc=1.0, max_tokens=80)
    assert sig.think_leak_rate == 1.0 and sig.prompt_only_kept_rate_fixed is not None
    assert "kept_rates" in extras and extras["prompt_only_before"]["think_leak_rate"] == 1.0


# --------------------------------------------------------------------------------------
# CLI synthetic smoke
# --------------------------------------------------------------------------------------


def test_cli_synthetic_smoke(tmp_path):
    out = tmp_path / "v04"
    proc = subprocess.run([sys.executable, str(SCRIPT), "synthetic-smoke", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["two_b_primary_mode"] == "intervention_collapse"
    assert payload["any_viable_steer"] is False
    for name in ("failure_modes.json", "sweep_results.jsonl", "source_viability_by_teacher.md",
                 "examples_failure_modes.jsonl", "eval_truth_holding_v04.md"):
        assert (out / name).exists()
    diag = json.loads((out / "failure_modes.json").read_text())
    assert diag["teachers"]["qwen_27b_modal"]["status"] == "not_run"
