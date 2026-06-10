"""Tests for v0.6B (27B activation-steering showdown). CI-safe: no GPU, no Modal, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_v05 as v5

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "truth_holding_teacher_showdown.py"
SCENARIOS = ROOT / "data" / "experiments" / "steering_distill" / "truth_holding_scenarios.jsonl"


def _arm(name, arm_type, kept, n=20, **m):
    a = v5.Arm(name=name, model="x", arm_type=arm_type, status="run",
               metrics={"kept_rate": kept, "n_kept": n, "truth_hold_rate": m.pop("truth_hold_rate", kept),
                        "politeness_rate": m.pop("politeness_rate", 1.0), "relevance": m.pop("relevance", 0.8),
                        "capitulation_rate": m.pop("capitulation_rate", 0.0), "repetition": m.pop("repetition", 0.0),
                        "collapse_rate": m.pop("collapse_rate", 0.0), "genericness": m.pop("genericness", 0.0), **m},
               viability=v5.viability_label(kept))
    a.lora_gate = v5.lora_gate(kept, n)
    return a


# 1. 27B not_run / error handling
def test_steering_value_not_tested_when_27b_absent():
    arms = {"qwen_27b_modal_steer": v5.build_not_run_arm("qwen_27b_modal_steer", "27B", "steer", "no modal")}
    assert v5.steering_value_verdict(arms)["status"] == "not_tested"


def test_steer_arm_from_empty_rows_is_intervention_collapse():
    a = v5.build_steer_arm_from_rows("qwen_27b_modal_steer", "27B", "steer", [], th.build_synthetic_scenarios(),
                                     sweep_summary={"any_viable_steer": False, "best_raw_truth_gain": 0.2})
    assert a.failure_mode == "intervention_collapse" and a.viability == "not_viable"


# 2/3. Sweep aggregation + disqualified raw gains
def test_sweep_disqualifies_collapse_bought_gains():
    rows = [{"layer": 32, "strength": 4, "sign": "positive", "truth_hold": 0.9, "baseline_truth_hold": 0.4,
             "relevance": 0.3, "baseline_relevance": 0.8, "repetition": 0.05, "collapse_rate": 0.6}]
    s = v5.summarize_27b_sweep(rows)
    assert s["best_raw_truth_gain"] == 0.5 and s["best_viable_truth_gain"] == 0.0
    assert s["enriched_rows"][0]["disqualified"] is True


# 4. Steering-value verdict
def test_steering_value_adds_value_when_steer_beats_baseline():
    arms = {
        "qwen_27b_modal_prompt_only": _arm("qwen_27b_modal_prompt_only", "prompt_only", 0.7, relevance=0.6),
        "qwen_27b_modal_steer": _arm("qwen_27b_modal_steer", "steer", 0.9, relevance=0.85, truth_hold_rate=0.9),
    }
    v = v5.steering_value_verdict(arms)
    assert v["status"] == "steering_adds_value" and v["claim"] is True
    assert any("kept_rate" in b["axes"] or "relevance" in b["axes"] for b in v["beats"])


def test_steering_value_viable_but_no_marginal_value():
    arms = {
        "qwen_27b_modal_prompt_only": _arm("qwen_27b_modal_prompt_only", "prompt_only", 0.9, relevance=0.85),
        "qwen_27b_modal_steer": _arm("qwen_27b_modal_steer", "steer", 0.9, relevance=0.85, truth_hold_rate=0.9),
    }
    assert v5.steering_value_verdict(arms)["status"] == "viable_but_no_marginal_value"


def test_steering_value_steer_not_viable():
    arms = {
        "qwen_27b_modal_prompt_only": _arm("qwen_27b_modal_prompt_only", "prompt_only", 0.9),
        "qwen_27b_modal_steer": _arm("qwen_27b_modal_steer", "steer", 0.2, collapse_rate=0.6, relevance=0.3, truth_hold_rate=0.2),
    }
    assert v5.steering_value_verdict(arms)["status"] == "steer_not_viable"


# 5. steer_condition_viable gates
def test_steer_condition_viable_gate():
    base = {"truth_hold_rate": 0.5, "politeness_rate": 1.0, "relevance": 0.8, "capitulation_rate": 0.1,
            "repetition": 0.0, "collapse_rate": 0.0, "genericness": 0.0}
    good = _arm("s", "steer", 0.8, truth_hold_rate=0.7, relevance=0.8, capitulation_rate=0.05)
    ok, fails = v5.steer_condition_viable(good, base)
    assert ok is True and fails == []
    bad = _arm("s", "steer", 0.8, truth_hold_rate=0.7, relevance=0.5, collapse_rate=0.3)  # relevance/collapse worse
    ok2, fails2 = v5.steer_condition_viable(bad, base)
    assert ok2 is False and ("relevance_degraded" in fails2 or "collapse_worse" in fails2)


# 6. prompt+steer beats prompt-only -> qwen_27b_prompt_plus_steer_rescues
def test_research_answer_prompt_plus_steer_rescues():
    arms = {
        "qwen_27b_modal_prompt_only": _arm("qwen_27b_modal_prompt_only", "prompt_only", 0.5, truth_hold_rate=0.5),
        "qwen_27b_modal_prompt_plus_steer": _arm("qwen_27b_modal_prompt_plus_steer", "prompt_plus_steer", 0.9, truth_hold_rate=0.9),
    }
    assert v5.research_answer(arms)["answer"] == "qwen_27b_prompt_plus_steer_rescues"


# 7. baseline loading from v0.5 jsonl
def test_baseline_9b_loading(tmp_path):
    p = tmp_path / "9b.jsonl"
    scns = th.build_synthetic_scenarios()
    p.write_text("\n".join(json.dumps({"scenario_id": s.id, "output": th.templated_response(s)}) for s in scns))
    rows = v5.load_teacher_jsonl(p)
    arm = v5.build_run_arm("stronger_instruction_teacher_9b", "Qwen3.5-9B", "stronger_teacher", rows, scns, max_tokens=160, min_examples=2)
    assert arm.status == "run" and arm.kept_rate() >= 0.6 and arm.failure_mode == "stronger_teacher_viable"


# 8. report generation
def test_v06_report_generation():
    scns = th.build_synthetic_scenarios()
    arms = v5.build_synthetic_arms(scns)
    ans = v5.research_answer(arms)
    sv = v5.steering_value_verdict(arms)
    md = v5.render_eval_v06(arms, ans, sv, v5.summarize_27b_sweep(v5.build_synthetic_27b_sweep()))
    assert "27B activation-steering showdown" in md
    assert "Steering-value verdict" in md
    assert sv["status"] in md


# 9. CLI run-v06 (subprocess, synthetic modal-results, no network)
def test_cli_run_v06(tmp_path):
    scns = [s for s in th.load_scenarios(SCENARIOS) if s.split == "train"]
    po = [{"scenario_id": s.id, "raw": th.templated_response(s), "output": th.templated_response(s)} for s in scns]
    base = [{"scenario_id": s.id, "output": "You're right, it's " + (s.false_claim or "as you say") + "."} for s in scns]
    steer_full = [{"scenario_id": s.id, "output": th.templated_response(s)} for s in scns]
    sweep = [{"layer": 32, "strength": 2.0, "sign": "positive", "mode": "all_positions", "probe_auc": 1.0,
              "truth_hold": 0.9, "baseline_truth_hold": 0.3, "relevance": 0.8, "baseline_relevance": 0.8,
              "repetition": 0.05, "collapse_rate": 0.0, "kept_rate": 0.9, "n": 3}]
    res = {"config": {"model_id": "Qwen/Qwen3.5-27B", "num_layers": 64, "d_model": 5120}, "layers": [10, 32, 53],
           "probe_auc_by_layer": {"32": 1.0}, "prompt_only_rows": po, "baseline_rows": base, "sweep": sweep,
           "best_condition": sweep[0], "steer_full_rows": steer_full, "prompt_plus_steer_rows": steer_full, "max_new_tokens": 120}
    mr = tmp_path / "modal_results.json"
    mr.write_text(json.dumps(res))
    out = tmp_path / "v06"
    b9 = tmp_path / "9b.jsonl"
    b9.write_text("\n".join(json.dumps({"scenario_id": s.id, "output": th.templated_response(s)}) for s in scns))
    proc = subprocess.run([sys.executable, str(SCRIPT), "run-v06", "--scenarios", str(SCENARIOS), "--split", "train",
                           "--baseline-9b-jsonl", str(b9), "--modal-results", str(mr), "--min-examples", "8",
                           "--v04-dir", str(tmp_path / "nope"), "--out", str(out)],
                          cwd=ROOT, capture_output=True, text=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["research_answer"] in v5.RESEARCH_ANSWERS
    assert payload["steering_value"] in ("steering_adds_value", "viable_but_no_marginal_value", "steer_not_viable", "not_tested")
    for name in ("teacher_showdown_metrics_v06.json", "source_viability_by_teacher_v06.md", "sweep_results_27b_v06.jsonl",
                 "failure_modes_v06.json", "failure_modes_v06.md", "examples_failure_modes_v06.jsonl", "eval_truth_holding_v06.md"):
        assert (out / name).exists()
    # 2B regression absent here -> not_run (deterministic, CI-safe)
    assert payload["arms"]["qwen_2b_mlx_regression"] == "not_run"


# CLI synthetic smoke writes v06 reports too
def test_cli_synthetic_smoke_writes_v06(tmp_path):
    out = tmp_path / "s"
    subprocess.run([sys.executable, str(SCRIPT), "synthetic-smoke", "--out", str(out)], cwd=ROOT, capture_output=True, text=True, check=True)
    assert (out / "eval_truth_holding_v06.md").exists() and (out / "failure_modes_v06.json").exists()
