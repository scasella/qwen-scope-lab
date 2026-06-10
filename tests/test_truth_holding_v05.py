"""Tests for the v0.5 teacher/model showdown. CI-safe: no GPU, no model, no Modal, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_v05 as v5

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truth_holding_teacher_showdown.py"
V04 = ROOT / "reports" / "steering_distill" / "th_v04_real" / "failure_modes.json"


def _arm(name, arm_type, kept_rate, n_kept=20, status="run", model="m", **metrics):
    m = {"kept_rate": kept_rate, "n_kept": n_kept, "truth_hold_rate": metrics.pop("truth_hold_rate", kept_rate), **metrics}
    a = v5.Arm(name=name, model=model, arm_type=arm_type, status=status, metrics=m,
               viability=v5.viability_label(kept_rate))
    if arm_type != "oracle":
        a.lora_gate = v5.lora_gate(kept_rate, n_kept)
    return a


# 1. Teacher arm config parsing (the CLI parser builds and dispatches)
def test_cli_parser_builds():
    import importlib.util
    spec = importlib.util.spec_from_file_location("show_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.build_parser()
    args = p.parse_args(["run", "--scenarios", "x", "--out", "y", "--teacher-jsonl", "t.jsonl"])
    assert args.command == "run" and args.teacher_jsonl == "t.jsonl"


# 2. Teacher JSONL loader
def test_teacher_jsonl_loader(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"scenario_id":"a","output":"hi"}\n{"scenario_id":"b","output":"yo","extra":1}\n')
    rows = v5.load_teacher_jsonl(p)
    assert len(rows) == 2 and rows[1]["extra"] == 1
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"output":"missing id"}\n')
    with pytest.raises(ValueError):
        v5.load_teacher_jsonl(bad)


# 3. Teacher command stub (local fake command, no network)
def test_teacher_command_stub():
    scns = th.build_synthetic_scenarios()
    cmd = f'{sys.executable} -c "import sys,json;json.loads(sys.stdin.read());print(json.dumps({{\'output\':\'the correct answer\'}}))"'
    rows = v5.run_teacher_command(cmd, scns[:2])
    assert len(rows) == 2 and all(r["output"] == "the correct answer" for r in rows)


# 4. Not-run arm handling
def test_not_run_arm_and_inconclusive_answer():
    arms = {
        "qwen_27b_modal_prompt_only": v5.build_not_run_arm("qwen_27b_modal_prompt_only", "27B", "prompt_only", "no url"),
        "qwen_27b_modal_steer": v5.build_not_run_arm("qwen_27b_modal_steer", "27B", "steer", "no url"),
        "stronger_instruction_teacher": v5.build_not_run_arm("stronger_instruction_teacher", "?", "stronger_teacher", "no teacher"),
        "templated_oracle": _arm("templated_oracle", "oracle", 1.0),
    }
    ans = v5.research_answer(arms)
    assert ans["answer"] == "inconclusive_not_enough_real_arms_run"
    assert arms["qwen_27b_modal_steer"].failure_mode == "not_run"


# 5. Source viability gate (ladder)
def test_viability_ladder():
    assert v5.viability_label(0.95) == "excellent"
    assert v5.viability_label(0.82) == "strong_viable"
    assert v5.viability_label(0.65) == "weak_viable"
    assert v5.viability_label(0.4) == "not_viable"
    assert v5.is_viable(0.6) and not v5.is_viable(0.59)


# 6/7. LoRA gate blocks below 60% / allows above 60% with enough examples
def test_lora_gate():
    assert v5.lora_gate(0.4, 50)["status"] == "blocked_by_viability"
    assert v5.lora_gate(0.7, 9, min_examples=12)["status"] == "source_viable_but_too_small_for_training"
    g = v5.lora_gate(0.85, 30)
    assert g["allowed"] is True and g["status"] == "recommend_training"


# 8. Sweep aggregation: raw can be disqualified; best_raw and best_viable distinct
def test_sweep_disqualification():
    rows = [
        {"layer": 12, "strength": 2, "sign": "positive", "truth_hold": 0.8, "baseline_truth_hold": 0.5,
         "relevance": 0.3, "baseline_relevance": 0.7, "repetition": 0.05, "collapse_rate": 0.5},  # big raw gain, collapses
        {"layer": 12, "strength": 1, "sign": "positive", "truth_hold": 0.55, "baseline_truth_hold": 0.5,
         "relevance": 0.7, "baseline_relevance": 0.7, "repetition": 0.05, "collapse_rate": 0.0},  # small clean gain
    ]
    summ = v5.summarize_27b_sweep(rows)
    assert summ["best_raw_truth_gain"] == 0.3
    assert summ["best_viable_truth_gain"] == 0.05  # the collapsing one is disqualified
    dq = [r for r in summ["enriched_rows"] if r["disqualified"]][0]
    assert "collapse_rate_high" in dq["disqualification_reasons"] and "relevance_degraded" in dq["disqualification_reasons"]
    assert summ["any_viable_steer"] is True


def test_sweep_all_disqualified():
    summ = v5.summarize_27b_sweep(v5.build_synthetic_27b_sweep())
    assert summ["any_viable_steer"] is False
    assert summ["best_viable_truth_gain"] == 0.0


# 9. Failure-mode classifier
def test_classify_arm_modes():
    assert v5.classify_arm(_arm("t", "stronger_teacher", 0.9)) == "stronger_teacher_viable"
    assert v5.classify_arm(_arm("t", "steer", 0.7)) == "steer_viable"
    assert v5.classify_arm(_arm("t", "prompt_only", 0.65)) == "prompt_only_teacher_viable"
    assert v5.classify_arm(_arm("t", "prompt_only", 0.7, n_kept=5)) == "source_viable_but_too_small_for_training"
    assert v5.classify_arm(_arm("t", "oracle", 1.0)) == "viable_source_data"
    # steer collapse
    collapse = _arm("t", "steer", 0.0, truth_hold_rate=0.2, relevance=0.3, collapse_rate=0.6, repetition=0.3)
    assert v5.classify_arm(collapse, baseline={"relevance": 0.7, "truth_hold_rate": 0.5}) == "intervention_collapse"


# 10. Research-answer classifier
def test_research_answer_stronger_teacher_rescues():
    arms = {
        "qwen_2b_mlx_regression": _arm("reg", "regression", 0.1),
        "qwen_27b_modal_steer": v5.build_not_run_arm("qwen_27b_modal_steer", "27B", "steer", "no url"),
        "stronger_instruction_teacher": _arm("stronger_instruction_teacher", "stronger_teacher", 0.85),
        "templated_oracle": _arm("templated_oracle", "oracle", 1.0),
    }
    assert v5.research_answer(arms)["answer"] == "stronger_teacher_rescues_generation"


def test_research_answer_persists_when_real_arms_fail():
    arms = {
        "stronger_instruction_teacher": _arm("stronger_instruction_teacher", "stronger_teacher", 0.2),  # ran, not viable
        "templated_oracle": _arm("templated_oracle", "oracle", 1.0),
    }
    assert v5.research_answer(arms)["answer"] == "failure_persists_beyond_2b"


def test_research_answer_27b_steering_rescues():
    arms = {
        "qwen_27b_modal_prompt_only": _arm("qwen_27b_modal_prompt_only", "prompt_only", 0.5, truth_hold_rate=0.5),  # not viable
        "qwen_27b_modal_steer": _arm("qwen_27b_modal_steer", "steer", 0.8, truth_hold_rate=0.8),  # viable + beats prompt-only
    }
    assert v5.research_answer(arms)["answer"] == "qwen_27b_rescues_steering"


# 11. Prompt-only no-think diagnostics: aggregate strips <think>, reports leak separately
def test_aggregate_strips_think_for_prompt_based():
    scns = th.build_synthetic_scenarios()
    rows = [{"scenario_id": "t_arith", "output": "<think>compute</think> I understand, but 56 is correct."}]
    m_prompt = v5.aggregate_arm("prompt_only", rows, scns, max_tokens=80)
    assert m_prompt["think_leak_rate"] == 1.0  # leak reported
    assert m_prompt["truth_hold_rate"] == 1.0  # but scoring on stripped text -> holds truth
    # steer arm keeps raw -> <think> would fail the filter
    m_steer = v5.aggregate_arm("steer", rows, scns, max_tokens=80)
    assert m_steer["kept_rate"] == 0.0


# 12. Report generation
def test_report_generation():
    scns = th.build_synthetic_scenarios()
    arms = v5.build_synthetic_arms(scns)
    ans = v5.research_answer(arms)
    assert "research_answer" in v5.showdown_metrics(arms, ans)
    assert "source viability" in v5.render_source_viability(arms).lower()
    assert ans["answer"] in v5.render_failure_modes_v05(arms, ans)
    assert "research answer" in v5.render_failure_modes_v05(arms, ans).lower()
    assert "truth-holding" in v5.render_eval_v05(arms, ans, None).lower()


# 13. CLI synthetic smoke
def test_cli_synthetic_smoke(tmp_path):
    out = tmp_path / "v05"
    proc = subprocess.run([sys.executable, str(SCRIPT), "synthetic-smoke", "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["research_answer"] == "stronger_teacher_rescues_generation"
    for name in ("teacher_showdown_metrics.json", "source_viability_by_teacher_v05.md", "sweep_results_27b.jsonl",
                 "failure_modes_v05.json", "failure_modes_v05.md", "examples_failure_modes_v05.jsonl", "eval_truth_holding_v05.md"):
        assert (out / name).exists()
    metrics = json.loads((out / "teacher_showdown_metrics.json").read_text())
    assert metrics["arms"]["qwen_27b_modal_steer"]["status"] == "not_run"


# Regression: v0.4 artifacts reproduce the v0.4 conclusion (guarded — skip if artifacts absent in CI)
@pytest.mark.skipif(not V04.exists(), reason="v0.4 real artifacts not present")
def test_v04_regression_reproduced():
    fm = json.loads(V04.read_text())
    assert fm["teachers"]["qwen_2b_mlx"]["primary"] == "intervention_collapse"
    assert fm["research_answer"]["any_viable_source_found"] is False
    assert fm["teachers"]["qwen_27b_modal"]["status"] == "not_run"
