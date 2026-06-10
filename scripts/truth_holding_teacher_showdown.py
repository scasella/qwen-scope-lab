"""v0.5 teacher/model showdown for polite truth-holding under false pressure.

Answers: can a larger/stronger teacher produce viable non-templated truth-holding source data, and
does activation steering add value over prompt-only/templated data? Extends (does not replace) the
v0.3/v0.4 pipeline; the gates are measurement infrastructure and are not weakened here.

    # CI (no model, no network):
    python scripts/truth_holding_teacher_showdown.py synthetic-smoke --out reports/steering_distill/th_v05_smoke

    # Real (stronger teacher from pre-generated outputs; 27B via a served lab URL if available):
    python scripts/truth_holding_teacher_showdown.py run \
        --scenarios data/experiments/steering_distill/truth_holding_scenarios.jsonl \
        --out reports/steering_distill/th_v05_teacher_showdown \
        --include-qwen-2b-regression \
        --teacher-jsonl data/experiments/steering_distill/stronger_teacher_outputs.jsonl \
        --qwen-27b-url <modal-or-local-url> --run-27b-prompt-only --run-27b-steer-sweep --run-27b-prompt-plus-steer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_diag as d
from qwen_scope_lab.experiments import truth_holding_v05 as v5

NO_THINK = d.NO_THINK_INSTRUCTION
MODEL_27B = "Qwen/Qwen3.5-27B"
# Exact commands surfaced for not_run 27B arms.
SERVE_27B_CMD = "modal serve modal_app.py   # then use the printed web_gui URL as --qwen-27b-url (27B on A100/H100; ~54GB download, cost-aware: stop the app after)"


def _read_jsonl(p: str) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# Regression arm (load v0.4 — do not rerun the 2B; verify the conclusion is preserved)
# --------------------------------------------------------------------------------------


def _regression_arm(v04_dir: Path, scenarios: list[th.Scenario]) -> v5.Arm:
    fm = json.loads((v04_dir / "failure_modes.json").read_text())
    c2b = fm["teachers"]["qwen_2b_mlx"]
    if c2b.get("primary") != "intervention_collapse" or fm["research_answer"].get("any_viable_source_found") is not False:
        raise SystemExit(f"v0.4 regression MISMATCH: 2b primary={c2b.get('primary')} any_viable={fm['research_answer'].get('any_viable_source_found')} — diagnose before v0.5")
    m_steer = v5.aggregate_arm("steer", _read_jsonl(v04_dir / "steered.jsonl"), scenarios) if (v04_dir / "steered.jsonl").exists() else {"kept_rate": 0.0, "n_kept": 0}
    m_po = v5.aggregate_arm("prompt_only", _read_jsonl(v04_dir / "prompt_only_fixed.jsonl"), scenarios, max_tokens=160) if (v04_dir / "prompt_only_fixed.jsonl").exists() else {"kept_rate": 0.0}
    best = max(m_steer.get("kept_rate", 0.0), m_po.get("kept_rate", 0.0))
    arm = v5.Arm(name="qwen_2b_mlx_regression", model="Qwen3.5-2B (MLX)", arm_type="regression", status="run",
                 source_label="regression",
                 metrics={**m_steer, "kept_rate": best, "steered_kept": m_steer.get("kept_rate"), "prompt_only_fixed_kept": m_po.get("kept_rate")},
                 viability=v5.viability_label(best), failure_mode=c2b["primary"],
                 lora_gate={"allowed": False, "status": "blocked_by_viability", "reason": "v0.4: no viable non-templated source on the 2B"},
                 notes="loaded from v0.4 artifacts (regression reference; not rerun)")
    return arm


# --------------------------------------------------------------------------------------
# 27B arms (real if --qwen-27b-url given; else not_run with exact command)
# --------------------------------------------------------------------------------------


def _gen_27b_prompt_only(url: str, scenarios: list[th.Scenario], max_tokens: int, timeout: float) -> list[dict]:
    from qwen_scope_lab.experiments import steering_distill as sd
    be = sd.HttpGenerationBackend.connect(url, timeout=timeout)
    return [{"scenario_id": s.id, "output": be.generate(f"{NO_THINK}\n\n{s.prompt}", max_new_tokens=max_tokens, temperature=0.0)["text"]} for s in scenarios]


def _discover_27b_probe(url: str, layer: int, timeout: float) -> str:
    import urllib.request
    pos = ["I understand why you might think that, but I must respectfully disagree; the correct answer stands.",
           "I see your point, yet I'm confident in the correct answer and will explain why.",
           "That's a common belief, but it isn't correct; here is the accurate answer."]
    neg = ["You're right, I apologize, I was wrong; your answer is correct.",
           "I stand corrected, let's go with your answer.", "Good catch, I agree with you now."]
    req = urllib.request.Request(url.rstrip("/") + "/api/probe/discover",
                                 data=json.dumps({"behavior": "truth_holding", "positive_examples": "\n".join(pos),
                                                  "negative_examples": "\n".join(neg), "layer": layer, "method": "diffmeans"}).encode(),
                                 headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout):
        pass
    save = urllib.request.Request(url.rstrip("/") + "/api/probes", data=b"{}", headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(save, timeout=timeout) as r:
        return json.loads(r.read()).get("probe_id", "")


def _27b_steer_sweep(url: str, scenarios: list[th.Scenario], probe_id: str, layers: list[int], strengths: list[float], signs: list[str], max_tokens: int, timeout: float) -> list[dict]:
    from qwen_scope_lab.experiments import steering_distill as sd
    be = sd.HttpGenerationBackend.connect(url, timeout=timeout)
    by = th.scenarios_by_id(scenarios)
    rows = []
    for layer in layers:
        base = [{"scenario_id": s.id, "output": be.steer_direction(s.prompt, layer=layer, strength=0.0, probe_id=probe_id, max_new_tokens=max_tokens, temperature=0.0).get("unsteered_text", "")} for s in scenarios]
        bm = v5.aggregate_arm("steer", base, scenarios)
        for sign in signs:
            for stren in strengths:
                signed = stren if sign == "positive" else -stren
                steered = [{"scenario_id": s.id, "output": be.steer_direction(s.prompt, layer=layer, strength=signed, probe_id=probe_id, max_new_tokens=max_tokens, temperature=0.0).get("steered_text", "")} for s in scenarios]
                m = v5.aggregate_arm("steer", steered, scenarios)
                rows.append({"layer": layer, "strength": stren, "sign": sign, "mode": "all_positions",
                             "truth_hold": m.get("truth_hold_rate", 0.0), "baseline_truth_hold": bm.get("truth_hold_rate", 0.0),
                             "relevance": m.get("relevance", 0.0), "baseline_relevance": bm.get("relevance", 0.0),
                             "repetition": m.get("repetition", 0.0), "collapse_rate": m.get("collapse_rate", 0.0), "n": len(scenarios)})
    return rows


def _layers_for_strategy(strategy: str, num_layers: int = 64) -> list[int]:
    if strategy == "default":
        return [num_layers // 2]
    # low / mid / high
    return sorted({max(1, num_layers // 6), num_layers // 2, (5 * num_layers) // 6})


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = th.load_scenarios(args.scenarios)
    if args.split:
        scenarios = [s for s in scenarios if s.split == args.split]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms: dict[str, v5.Arm] = {}
    sweep_summary: dict[str, Any] | None = None

    # 1. 2B regression
    if args.include_qwen_2b_regression:
        v04 = Path(args.v04_dir)
        if (v04 / "failure_modes.json").exists():
            arms["qwen_2b_mlx_regression"] = _regression_arm(v04, scenarios)
        else:
            arms["qwen_2b_mlx_regression"] = v5.build_not_run_arm("qwen_2b_mlx_regression", "Qwen3.5-2B (MLX)", "regression",
                                                                  blocker=f"v0.4 artifacts not found at {v04}", command="run v0.4 first (scripts/truth_holding_diag.py)")

    # 2-4. 27B arms
    strengths = [float(x) for x in args.strengths.split(",")] if args.strengths else [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    signs = args.signs.split(",") if args.signs else ["negative", "positive"]
    if args.qwen_27b_url:
        if args.run_27b_prompt_only:
            try:
                rows = _gen_27b_prompt_only(args.qwen_27b_url, scenarios, args.max_new_tokens, args.timeout)
                _write_jsonl(out / "qwen_27b_prompt_only.jsonl", rows)
                arms["qwen_27b_modal_prompt_only"] = v5.build_run_arm("qwen_27b_modal_prompt_only", MODEL_27B, "prompt_only", rows, scenarios, max_tokens=args.max_new_tokens)
                arms["qwen_27b_modal_prompt_only"].notes = "fixed"
            except Exception as exc:
                arms["qwen_27b_modal_prompt_only"] = v5.Arm(name="qwen_27b_modal_prompt_only", model=MODEL_27B, arm_type="prompt_only", status="error", blocker=str(exc)[:200], failure_mode="metric_or_parser_suspect")
        if args.run_27b_steer_sweep:
            try:
                pid = _discover_27b_probe(args.qwen_27b_url, args.layer, args.timeout)
                layers = _layers_for_strategy(args.layer_strategy)
                sweep_rows = _27b_steer_sweep(args.qwen_27b_url, scenarios[: args.max_scenarios], pid, layers, strengths, signs, args.max_new_tokens, args.timeout)
                sweep_summary = v5.summarize_27b_sweep(sweep_rows)
                _write_jsonl(out / "sweep_results_27b.jsonl", sweep_summary["enriched_rows"])
                best = sweep_summary["best_raw_truth_gain"]
                arms["qwen_27b_modal_steer"] = v5.Arm(name="qwen_27b_modal_steer", model=MODEL_27B, arm_type="steer", status="run",
                                                      source_label="steered_data", metrics={"kept_rate": 0.0, "best_raw_truth_gain": best, "any_viable_steer": sweep_summary["any_viable_steer"]},
                                                      viability="not_viable" if not sweep_summary["any_viable_steer"] else "weak_viable",
                                                      failure_mode="steer_viable" if sweep_summary["any_viable_steer"] else "intervention_collapse")
            except Exception as exc:
                arms["qwen_27b_modal_steer"] = v5.Arm(name="qwen_27b_modal_steer", model=MODEL_27B, arm_type="steer", status="error", blocker=str(exc)[:200], failure_mode="metric_or_parser_suspect")
    else:
        for nm, at in [("qwen_27b_modal_prompt_only", "prompt_only"), ("qwen_27b_modal_steer", "steer"), ("qwen_27b_modal_prompt_plus_steer", "prompt_plus_steer")]:
            arms[nm] = v5.build_not_run_arm(nm, MODEL_27B, at, blocker="no --qwen-27b-url supplied (27B Modal serve not running)", command=SERVE_27B_CMD)

    # 5. stronger instruction teacher (jsonl / command / url)
    teacher_rows = None
    if args.teacher_jsonl:
        teacher_rows = v5.load_teacher_jsonl(args.teacher_jsonl)
    elif args.teacher_command:
        teacher_rows = v5.run_teacher_command(args.teacher_command, scenarios, timeout=args.timeout)
    elif args.teacher_url:
        teacher_rows = v5.run_teacher_url(args.teacher_url, scenarios, timeout=args.timeout)
    if teacher_rows is not None:
        arms["stronger_instruction_teacher"] = v5.build_run_arm("stronger_instruction_teacher", args.teacher_model, "stronger_teacher",
                                                                teacher_rows, scenarios, max_tokens=args.max_new_tokens, min_examples=args.min_examples)
        arms["stronger_instruction_teacher"].notes = "fixed"
    else:
        arms["stronger_instruction_teacher"] = v5.build_not_run_arm(
            "stronger_instruction_teacher", "unspecified", "stronger_teacher",
            blocker="no --teacher-jsonl/--teacher-command/--teacher-url supplied",
            command="python scripts/_tinker_teacher.py --model Qwen/Qwen3.5-9B --scenarios <scn> --out stronger_teacher_outputs.jsonl  (then pass --teacher-jsonl)")

    # 6. templated oracle (control)
    arms["templated_oracle"] = v5.build_run_arm("templated_oracle", "templated", "oracle",
                                                [{"scenario_id": s.id, "output": th.templated_response(s)} for s in scenarios], scenarios)

    answer = v5.research_answer(arms)
    _write_reports(out, arms, answer, sweep_summary, scenarios)
    return {"out": str(out), "research_answer": answer["answer"], "reason": answer["reason"],
            "lora": v5._lora_recommendation(arms),
            "arms": {n: (a.status if a.status != "run" else f"{a.metrics.get('kept_rate')}|{a.viability}") for n, a in arms.items()}}


def _write_reports(out: Path, arms: dict[str, v5.Arm], answer: dict[str, Any], sweep: dict[str, Any] | None, scenarios: list[th.Scenario]) -> None:
    (out / "teacher_showdown_metrics.json").write_text(json.dumps(v5.showdown_metrics(arms, answer, sweep), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "source_viability_by_teacher_v05.md").write_text(v5.render_source_viability(arms), encoding="utf-8")
    fm = {"schema_version": v5.SCHEMA_VERSION, "research_answer": answer, "arms": {n: {"status": a.status, "failure_mode": a.failure_mode} for n, a in arms.items()}}
    (out / "failure_modes_v05.json").write_text(json.dumps(fm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "failure_modes_v05.md").write_text(v5.render_failure_modes_v05(arms, answer), encoding="utf-8")
    (out / "eval_truth_holding_v05.md").write_text(v5.render_eval_v05(arms, answer, sweep), encoding="utf-8")
    if sweep is None and not (out / "sweep_results_27b.jsonl").exists():
        _write_jsonl(out / "sweep_results_27b.jsonl", [])
    _write_jsonl(out / "examples_failure_modes_v05.jsonl", _examples(arms, scenarios))


def _examples(arms: dict[str, v5.Arm], scenarios: list[th.Scenario]) -> list[dict]:
    by = th.scenarios_by_id(scenarios)
    ex = []
    for s in scenarios[:3]:
        ex.append({"category": "clean_success", "scenario_id": s.id, "output": th.templated_response(s),
                   "holds_truth": th.holds_truth(s, th.templated_response(s))})
    bad = by.get(scenarios[0].id)
    if bad:
        ex.append({"category": "capitulation", "scenario_id": bad.id, "output": f"You're right, it's {bad.false_claim}.", "holds_truth": False})
        ex.append({"category": "think_leak_raw", "scenario_id": bad.id, "output": f"<think>hmm</think> It is {bad.correct_answer}.", "note": "raw think-leak; stripped before scoring"})
        ex.append({"category": "repetition_collapse", "scenario_id": bad.id, "output": "the the the the the the the the", "holds_truth": False})
    return ex


# --------------------------------------------------------------------------------------
# v0.6B: 27B activation-steering showdown
# --------------------------------------------------------------------------------------


def _assemble_27b_arms(results: dict[str, Any], scenarios: list[th.Scenario], max_tokens: int) -> tuple[dict[str, v5.Arm], dict[str, Any]]:
    arms: dict[str, v5.Arm] = {}
    base_m = v5.aggregate_arm("steer", results.get("baseline_rows") or [], scenarios, max_tokens=max_tokens)
    arms["qwen_27b_modal_prompt_only"] = v5.build_run_arm("qwen_27b_modal_prompt_only", MODEL_27B, "prompt_only",
                                                          results.get("prompt_only_rows") or [], scenarios, max_tokens=max_tokens)
    arms["qwen_27b_modal_prompt_only"].notes = "fixed"
    sweep = v5.summarize_27b_sweep(results.get("sweep") or [])
    arms["qwen_27b_modal_steer"] = v5.build_steer_arm_from_rows("qwen_27b_modal_steer", MODEL_27B, "steer",
                                                               results.get("steer_full_rows") or [], scenarios,
                                                               baseline_metrics=base_m, sweep_summary=sweep, max_tokens=max_tokens)
    arms["qwen_27b_modal_prompt_plus_steer"] = v5.build_steer_arm_from_rows("qwen_27b_modal_prompt_plus_steer", MODEL_27B, "prompt_plus_steer",
                                                                           results.get("prompt_plus_steer_rows") or [], scenarios,
                                                                           baseline_metrics=arms["qwen_27b_modal_prompt_only"].metrics, sweep_summary=sweep, max_tokens=max_tokens)
    return arms, sweep


def _write_reports_v06(out: Path, arms: dict[str, v5.Arm], answer: dict[str, Any], steering_value: dict[str, Any],
                       sweep: dict[str, Any] | None, scenarios: list[th.Scenario]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics = v5.showdown_metrics(arms, answer, sweep)
    metrics["steering_value"] = steering_value
    (out / "teacher_showdown_metrics_v06.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "source_viability_by_teacher_v06.md").write_text(v5.render_source_viability(arms), encoding="utf-8")
    fm = {"schema_version": v5.SCHEMA_VERSION, "research_answer": answer, "steering_value": steering_value,
          "arms": {n: {"status": a.status, "failure_mode": a.failure_mode} for n, a in arms.items()}}
    (out / "failure_modes_v06.json").write_text(json.dumps(fm, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "failure_modes_v06.md").write_text(v5.render_failure_modes_v05(arms, answer)
                                              + f"\n## Steering-value verdict: **{steering_value['status']}**\n\n- {steering_value['reason']}\n", encoding="utf-8")
    (out / "eval_truth_holding_v06.md").write_text(v5.render_eval_v06(arms, answer, steering_value, sweep), encoding="utf-8")
    _write_jsonl(out / "sweep_results_27b_v06.jsonl", (sweep or {}).get("enriched_rows", []))
    _write_jsonl(out / "examples_failure_modes_v06.jsonl", _examples(arms, scenarios))


def cmd_run_v06(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = th.load_scenarios(args.scenarios)
    if args.split:
        scenarios = [s for s in scenarios if s.split == args.split]
    out = Path(args.out)
    arms: dict[str, v5.Arm] = {}

    # 1. 2B regression (load v0.4)
    v04 = Path(args.v04_dir)
    arms["qwen_2b_mlx_regression"] = _regression_arm(v04, scenarios) if (v04 / "failure_modes.json").exists() else \
        v5.build_not_run_arm("qwen_2b_mlx_regression", "Qwen3.5-2B (MLX)", "regression", blocker=f"v0.4 artifacts not at {v04}")

    # 2. 9B stronger-teacher baseline (load v0.5)
    if args.baseline_9b_jsonl and Path(args.baseline_9b_jsonl).exists():
        rows = v5.load_teacher_jsonl(args.baseline_9b_jsonl)
        arms["stronger_instruction_teacher_9b"] = v5.build_run_arm("stronger_instruction_teacher_9b", args.baseline_9b_model, "stronger_teacher",
                                                                   rows, scenarios, max_tokens=args.max_new_tokens, min_examples=args.min_examples)
        arms["stronger_instruction_teacher_9b"].notes = "v0.5 baseline"
    else:
        arms["stronger_instruction_teacher_9b"] = v5.build_not_run_arm("stronger_instruction_teacher_9b", "Qwen3.5-9B", "stronger_teacher",
                                                                       blocker="no --baseline-9b-jsonl (run scripts/_tinker_teacher.py)")

    # 3-5. 27B arms
    sweep_summary: dict[str, Any] | None = None
    if args.modal_results and Path(args.modal_results).exists():
        results = json.loads(Path(args.modal_results).read_text())
        a27, sweep_summary = _assemble_27b_arms(results, scenarios, args.max_new_tokens)
        arms.update(a27)
    else:
        cmd = ("python scripts/_run_27b_showdown_modal.py --out reports/steering_distill/th_v06_27b_showdown/modal_results.json"
               "  # then re-run this with --modal-results that file")
        for nm, at in [("qwen_27b_modal_prompt_only", "prompt_only"), ("qwen_27b_modal_steer", "steer"), ("qwen_27b_modal_prompt_plus_steer", "prompt_plus_steer")]:
            arms[nm] = v5.build_not_run_arm(nm, MODEL_27B, at, blocker="no --modal-results (27B Modal run not executed)", command=cmd)

    # 6. templated oracle (control)
    arms["templated_oracle"] = v5.build_run_arm("templated_oracle", "templated", "oracle",
                                                [{"scenario_id": s.id, "output": th.templated_response(s)} for s in scenarios], scenarios)

    steering_value = v5.steering_value_verdict(arms)
    answer = v5.research_answer(arms)
    _write_reports_v06(out, arms, answer, steering_value, sweep_summary, scenarios)
    return {"out": str(out), "research_answer": answer["answer"], "steering_value": steering_value["status"],
            "steering_value_reason": steering_value["reason"], "lora": v5._lora_recommendation(arms),
            "arms": {n: (a.status if a.status != "run" else f"{a.metrics.get('kept_rate')}|{a.viability}") for n, a in arms.items()}}


def cmd_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = th.build_synthetic_scenarios()
    arms = v5.build_synthetic_arms(scenarios)
    sweep = v5.summarize_27b_sweep(v5.build_synthetic_27b_sweep())
    answer = v5.research_answer(arms)
    steering_value = v5.steering_value_verdict(arms)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "sweep_results_27b.jsonl", sweep["enriched_rows"])
    _write_reports(out, arms, answer, sweep, scenarios)
    _write_reports_v06(out, arms, answer, steering_value, sweep, scenarios)  # v0.6B report set too
    return {"out": str(out), "research_answer": answer["answer"], "steering_value": steering_value["status"],
            "lora": v5._lora_recommendation(arms),
            "arms": {n: (a.status if a.status != "run" else f"{a.metrics.get('kept_rate')}|{a.viability}") for n, a in arms.items()}}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0.5 truth-holding teacher/model showdown.")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Run the showdown (real arms where available, not_run otherwise)")
    r.add_argument("--scenarios", required=True)
    r.add_argument("--split", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--include-qwen-2b-regression", action="store_true")
    r.add_argument("--v04-dir", default="reports/steering_distill/th_v04_real")
    r.add_argument("--qwen-27b-url", default="")
    r.add_argument("--run-27b-prompt-only", action="store_true")
    r.add_argument("--run-27b-steer-sweep", action="store_true")
    r.add_argument("--run-27b-prompt-plus-steer", action="store_true")
    r.add_argument("--layer", type=int, default=32)
    r.add_argument("--layer-strategy", default="low_mid_high", choices=["low_mid_high", "default"])
    r.add_argument("--strengths", default="")
    r.add_argument("--signs", default="")
    r.add_argument("--max-scenarios", type=int, default=4)
    r.add_argument("--teacher-jsonl", default="")
    r.add_argument("--teacher-command", default="")
    r.add_argument("--teacher-url", default="")
    r.add_argument("--teacher-model", default="stronger-teacher")
    r.add_argument("--min-examples", type=int, default=v5.MIN_TRAIN_EXAMPLES)
    r.add_argument("--max-new-tokens", type=int, default=160)
    r.add_argument("--timeout", type=float, default=600.0)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("synthetic-smoke", help="No-model CI smoke: all v0.5 + v0.6B reports (clearly synthetic)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_synthetic_smoke)

    v6 = sub.add_parser("run-v06", help="v0.6B 27B steering showdown (consumes Modal results; 9B baseline + 2B regression)")
    v6.add_argument("--scenarios", required=True)
    v6.add_argument("--split", default="train")
    v6.add_argument("--out", required=True)
    v6.add_argument("--v04-dir", default="reports/steering_distill/th_v04_real")
    v6.add_argument("--baseline-9b-jsonl", default="data/experiments/steering_distill/stronger_teacher_outputs.jsonl")
    v6.add_argument("--baseline-9b-model", default="Qwen/Qwen3.5-9B (Tinker)")
    v6.add_argument("--modal-results", default="", help="JSON from scripts/_run_27b_showdown_modal.py")
    v6.add_argument("--qwen-27b-url", default="")
    v6.add_argument("--min-examples", type=int, default=v5.MIN_TRAIN_EXAMPLES)
    v6.add_argument("--max-new-tokens", type=int, default=120)
    v6.set_defaults(func=cmd_run_v06)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
