"""v0.7 stronger-teacher truth-holding distillation — staged CLI.

Stages (each a subcommand so failures are diagnosable):
    preflight        Re-read the v0.6 report; confirm the steering-negative / prompting-positive result.
    make-scenarios   Generate the expanded, split-by-fact scenario corpus (no model).
    generate-teacher Sample a stronger teacher (9B via Tinker) on a scenario split → teacher_outputs.jsonl.
    audit-source     Score+filter teacher outputs with the v0.3 filters; gate training; export SFT/pairs (no model).
    train            LoRA on the 4B from an audited SFT, then sample baseline / prompt-only / distilled on the
                     eval splits (Tinker) → per-arm outputs + training_manifest.json.
    eval             Score arm outputs per split/domain/pressure, apply the conservative verdict, write reports (no model).
    synthetic-smoke  Whole pipeline with templated stand-ins (CI; clearly synthetic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen_scope_lab.experiments import truth_holding as th
from qwen_scope_lab.experiments import truth_holding_distill_v07 as v7
from qwen_scope_lab.experiments.truth_holding_diag import NO_THINK_INSTRUCTION, strip_think

EVAL_SPLITS = ("eval_id", "eval_ood", "eval_ambiguous", "eval_adversarial")
ARMS = ("baseline_4b", "prompt_only_inference_4b", "distilled_4b_from_9b_teacher")


def _read_jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------------------


def cmd_preflight(args: argparse.Namespace) -> dict[str, Any]:
    m = json.loads(Path(args.v06_report).read_text())
    A = m["arms"]
    checks = {
        "research_answer==qwen_27b_rescues_prompting": m["research_answer"]["answer"] == "qwen_27b_rescues_prompting",
        "steering_value==steer_not_viable": m["steering_value"]["status"] == "steer_not_viable",
        "9b_source_viable_and_trainable": A["stronger_instruction_teacher_9b"]["metrics"]["kept_rate"] >= 0.6 and A["stronger_instruction_teacher_9b"]["lora_gate"]["allowed"],
        "27b_prompt_only_viable_but_too_small": A["qwen_27b_modal_prompt_only"]["metrics"]["kept_rate"] >= 0.6 and not A["qwen_27b_modal_prompt_only"]["lora_gate"]["allowed"],
        "27b_steer_not_viable": A["qwen_27b_modal_steer"]["metrics"]["kept_rate"] < 0.6,
        "27b_prompt_plus_steer_no_beat": A["qwen_27b_modal_prompt_plus_steer"]["metrics"]["kept_rate"] <= A["qwen_27b_modal_prompt_only"]["metrics"]["kept_rate"] + 1e-9,
        "templated_oracle_excluded": A["templated_oracle"]["lora_gate"].get("status") == "control_excluded_from_gate",
    }
    ok = all(checks.values())
    if not ok:
        raise SystemExit("v0.6 preflight MISMATCH — diagnose before v0.7:\n" + json.dumps(checks, indent=2))
    return {"preflight": "pass", "checks": checks}


# --------------------------------------------------------------------------------------
# make-scenarios
# --------------------------------------------------------------------------------------


def cmd_make_scenarios(args: argparse.Namespace) -> dict[str, Any]:
    splits = v7.make_scenarios(n_train=args.n_train, n_dev=args.n_dev, n_eval_id=args.n_eval_id,
                               n_eval_ood=args.n_eval_ood, n_eval_ambiguous=args.n_eval_ambiguous,
                               n_eval_adversarial=args.n_eval_adversarial, seed=args.seed)
    out = Path(args.out)
    files = {}
    for split, rows in splits.items():
        path = out / f"{split}_scenarios.jsonl"
        _write_jsonl(path, rows)
        files[split] = {"path": str(path), "n": len(rows)}
    (out / "scenarios_manifest.json").write_text(json.dumps({"schema_version": v7.SCHEMA_VERSION, "seed": args.seed, "files": files}, indent=2) + "\n", encoding="utf-8")
    return {"out": str(out), "counts": {k: v["n"] for k, v in files.items()}}


# --------------------------------------------------------------------------------------
# generate-teacher (Tinker sampling)
# --------------------------------------------------------------------------------------


def cmd_generate_teacher(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.scenarios)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.teacher_jsonl:  # accept pre-generated (e.g. 27B prompt-only from a Modal run)
        pre = {r["scenario_id"] if "scenario_id" in r else r["id"]: r.get("output", r.get("raw", "")) for r in _read_jsonl(args.teacher_jsonl)}
        teacher_rows = [{"scenario_id": r["id"], "raw": pre.get(r["id"], ""), "output": strip_think(pre.get(r["id"], "")), "source": args.source} for r in rows]
        _write_jsonl(out / "teacher_outputs.jsonl", teacher_rows)
        return {"out": str(out), "source": args.source, "n": len(teacher_rows), "mode": "pre_generated"}

    import tinker
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    sc = tinker.ServiceClient()
    sampler = sc.create_sampling_client(base_model=args.model)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    def render(prompt: str) -> list[int]:
        msgs = [{"role": "user", "content": f"{NO_THINK_INSTRUCTION}\n\n{prompt}"}]
        try:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        return tok.encode(text)

    teacher_rows = []
    for r in rows:
        mi = tinker.ModelInput.from_ints(render(v7.scenario_prompt(r)))
        resp = sampler.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        raw = tok.decode(list(resp.sequences[0].tokens), skip_special_tokens=True)
        teacher_rows.append({"scenario_id": r["id"], "domain": r["domain"], "pressure_type": r.get("pressure_type"),
                             "model": args.model, "raw": raw, "output": strip_think(raw), "source": args.source})
    _write_jsonl(out / "teacher_outputs.jsonl", teacher_rows)
    meta = {"source": args.source, "model": args.model, "seed": 0, "temperature": 0.0, "max_tokens": args.max_tokens,
            "prompt_template": "NO_THINK_INSTRUCTION + scenario_prompt", "n": len(teacher_rows)}
    (out / "generation_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"out": str(out), "source": args.source, "model": args.model, "n": len(teacher_rows), "sample": teacher_rows[0]["output"][:120] if teacher_rows else ""}


# --------------------------------------------------------------------------------------
# audit-source
# --------------------------------------------------------------------------------------


def cmd_audit_source(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = _read_jsonl(args.scenarios)
    teacher = {r["scenario_id"]: r.get("output", "") for r in _read_jsonl(args.teacher_outputs)}
    raws = {r["scenario_id"]: r.get("raw", r.get("output", "")) for r in _read_jsonl(args.teacher_outputs)}
    audit = v7.audit_source(scenarios, raws, source=args.source, is_templated=args.templated)
    elig = v7.training_eligibility(audit["metrics"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "pairs_all.jsonl", audit["all"])
    _write_jsonl(out / "pairs_kept.jsonl", audit["kept"])
    _write_jsonl(out / "pairs_rejected.jsonl", audit["rejected"])
    _write_jsonl(out / "sft_train.jsonl", v7.to_sft_records(audit["kept"]))
    _write_jsonl(out / "preference_train.jsonl", v7.to_preference_records(audit["kept"]))
    (out / "source_audit.json").write_text(json.dumps({"metrics": audit["metrics"], "eligibility": elig}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "source_audit.md").write_text(_render_source_audit(audit["metrics"], elig), encoding="utf-8")
    if not elig["eligible"]:
        (out / "DO_NOT_TRAIN.md").write_text(f"# Source NOT eligible for a serious LoRA run\n\nsource: `{args.source}`\nstatus: **{elig['status']}**\nreason: {elig['reason']}\n\n- {chr(10).join(elig['fails']) or 'see warnings'}\n", encoding="utf-8")
    return {"out": str(out), "source": args.source, "kept": audit["metrics"]["n_kept"], "n": audit["metrics"]["n"],
            "kept_rate": audit["metrics"]["kept_rate"], "eligibility": elig["status"], "reason": elig["reason"]}


def _render_source_audit(m: dict, elig: dict) -> str:
    dom = "\n".join(f"| {d} | {b['n']} | {b['kept']} | {b['kept_rate']} |" for d, b in m["domain_breakdown"].items())
    pre = "\n".join(f"| {p} | {b['n']} | {b['kept']} | {b['kept_rate']} |" for p, b in m["pressure_breakdown"].items())
    warns = "\n".join(f"- ⚠️ {w}" for w in (m["phrase_concentration"].get("warnings", []) + elig["warns"])) or "- none"
    return (
        f"# v0.7 source audit — `{m['source']}`{' (templated control)' if m['is_templated'] else ''}\n\n"
        f"- kept: **{m['n_kept']}/{m['n']}** ({m['kept_rate']:.0%}) · eligibility: **{elig['status']}** — {elig['reason']}\n"
        f"- truth_hold {m['truth_hold_rate']} · correctness {m['correctness_rate']} · capitulation {m['capitulation_rate']} · "
        f"politeness {m['politeness_rate']} · relevance {m['relevance']} · genericness {m['genericness']} · "
        f"repetition {m['repetition']} · collapse {m['collapse_rate']}\n"
        f"- think-leak {m['think_leak_rate']} · truncation {m['truncation_rate']} · ambiguous-calibration {m['ambiguous_case_calibration']}\n"
        f"- reject reasons: {json.dumps(m['reject_reason_counts']) if m['reject_reason_counts'] else 'none'}\n\n"
        f"## Domain breakdown\n\n| domain | n | kept | kept-rate |\n|---|---|---|---|\n{dom}\n\n"
        f"## Pressure-type breakdown\n\n| pressure | n | kept | kept-rate |\n|---|---|---|---|\n{pre}\n\n"
        f"## Warnings\n\n{warns}\n"
    )


# --------------------------------------------------------------------------------------
# train (Tinker LoRA + sample arms on eval splits)
# --------------------------------------------------------------------------------------


def _eval_arm_outputs_dir(out: Path) -> Path:
    return out / "arm_outputs"


def cmd_train(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import tinker
    from transformers import AutoTokenizer

    sft = _read_jsonl(args.sft)
    n = len(sft)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "run", "base_model": args.base_model, "source_sft": args.sft, "n_examples": n,
                "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "batch_size": args.batch_size, "seed": 0,
                "train_command": "truth_holding_distill_v07.py train", "min_examples_for_serious": 100,
                "serious_run": n >= 100}
    if n < args.min_examples and not args.allow_smoke:
        manifest.update({"status": "skipped_by_gate", "reason": f"{n} < {args.min_examples} kept examples; pass --allow-smoke for a labeled smoke run"})
        (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return {"out": str(out), "status": "skipped_by_gate", "n": n}

    tok = AutoTokenizer.from_pretrained(args.base_model)

    def render(messages, add_gen):
        try:
            text = tok.apply_chat_template(messages, add_generation_prompt=add_gen, tokenize=False, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(messages, add_generation_prompt=add_gen, tokenize=False)
        return tok.encode(text)

    def datum(user, assistant):
        p = render([{"role": "user", "content": user}], True)
        full = render([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}], False)[: args.max_seq]
        inp, tgt = full[:-1], full[1:]
        b = max(0, len(p) - 1)
        w = [1.0 if i >= b else 0.0 for i in range(len(tgt))]
        return tinker.Datum(model_input=tinker.ModelInput.from_ints(inp),
                            loss_fn_inputs={"target_tokens": tinker.TensorData.from_numpy(np.asarray(tgt, np.int64)),
                                            "weights": tinker.TensorData.from_numpy(np.asarray(w, np.float32))})

    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    data = [datum(r["messages"][0]["content"], r["messages"][1]["content"]) for r in sft]
    adam = tinker.AdamParams(learning_rate=args.lr)
    step = 0
    for epoch in range(args.epochs):
        order = np.random.RandomState(epoch).permutation(len(data))
        for i in range(0, len(order), args.batch_size):
            batch = [data[j] for j in order[i:i + args.batch_size]]
            tc.forward_backward(batch, "cross_entropy").result()
            tc.optim_step(adam).result()
            step += 1
    manifest["steps"] = step
    distilled = tc.save_weights_and_get_sampling_client()
    base = sc.create_sampling_client(base_model=args.base_model)
    sp = tinker.SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    def sample(client, prompt, instruction=""):
        full = f"{instruction}\n\n{prompt}" if instruction else prompt
        mi = tinker.ModelInput.from_ints(render([{"role": "user", "content": full}], True))
        return strip_think(tok.decode(list(client.sample(prompt=mi, num_samples=1, sampling_params=sp).result().sequences[0].tokens), skip_special_tokens=True))

    arm_dir = _eval_arm_outputs_dir(out)
    for split in EVAL_SPLITS:
        sp_path = Path(args.eval_root) / f"{split}_scenarios.jsonl"
        if not sp_path.exists():
            continue
        rows = _read_jsonl(sp_path)
        for arm, client, instr in [("baseline_4b", base, ""), ("prompt_only_inference_4b", base, NO_THINK_INSTRUCTION), ("distilled_4b_from_9b_teacher", distilled, "")]:
            _write_jsonl(arm_dir / arm / f"{split}.jsonl",
                         [{"scenario_id": r["id"], "output": sample(client, v7.scenario_prompt(r), instr)} for r in rows])
    manifest["adapter"] = "ephemeral (sampled in-process)"
    manifest["arm_outputs_dir"] = str(arm_dir)
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"out": str(out), "status": "run", "n": n, "steps": step, "arm_outputs_dir": str(arm_dir)}


# --------------------------------------------------------------------------------------
# eval (no model — scores arm outputs)
# --------------------------------------------------------------------------------------


def _load_arms(arm_outputs_dir: Path, scenario_splits: dict[str, list[dict]]) -> dict[str, Any]:
    arms = {}
    for arm in ARMS:
        adir = arm_outputs_dir / arm
        if not adir.exists():
            arms[arm] = {"status": "not_run"}
            continue
        outs_by_split = {}
        for split in scenario_splits:
            f = adir / f"{split}.jsonl"
            if f.exists():
                outs_by_split[split] = {r["scenario_id"]: r["output"] for r in _read_jsonl(f)}
        ev = v7.evaluate_arm(scenario_splits, outs_by_split)
        arms[arm] = {"status": "run", **ev}
    return arms


def cmd_eval(args: argparse.Namespace) -> dict[str, Any]:
    scenario_splits = {sp: _read_jsonl(Path(args.eval_root) / f"{sp}_scenarios.jsonl") for sp in EVAL_SPLITS if (Path(args.eval_root) / f"{sp}_scenarios.jsonl").exists()}
    arms = _load_arms(Path(args.arm_outputs_dir), scenario_splits)
    elig = json.loads(Path(args.source_audit).read_text())["eligibility"] if args.source_audit and Path(args.source_audit).exists() else None
    verdict = v7.distillation_verdict(arms, eligibility=elig, n_train_kept=args.n_train_kept)
    out = Path(args.out)
    _write_reports_v07(out, arms, verdict, scenario_splits)
    return {"out": str(out), "verdict": verdict["verdict"], "reason": verdict["reason"],
            "deltas_vs_baseline": verdict.get("deltas_vs_baseline", {}),
            "arms": {a: (v.get("status") if v.get("status") != "run" else v["overall"].get("truth_hold_rate")) for a, v in arms.items()}}


def _write_reports_v07(out: Path, arms: dict, verdict: dict, scenario_splits: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "v07_eval_metrics.json").write_text(json.dumps({"schema_version": v7.SCHEMA_VERSION, "arms": arms, "verdict": verdict}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "v07_eval_truth_holding.md").write_text(_render_eval_md(arms, verdict), encoding="utf-8")
    (out / "v07_final_decision.md").write_text(_render_final_decision(arms, verdict), encoding="utf-8")
    _write_jsonl(out / "v07_examples_wins_failures.jsonl", _wins_failures(arms, scenario_splits, out))


def _render_eval_md(arms: dict, verdict: dict) -> str:
    cols = ["truth_hold_rate", "correctness_rate", "capitulation_rate", "politeness_rate", "relevance", "ambiguous_case_calibration"]
    def row(a, v):
        if v.get("status") != "run":
            return f"| `{a}` | _{v.get('status')}_ | " + " | ".join("—" for _ in cols) + " |"
        o = v["overall"]
        return f"| `{a}` | run | " + " | ".join(str(o.get(c, "—")) for c in cols) + " |"
    splitrows = []
    for a, v in arms.items():
        if v.get("status") == "run":
            for sp, sm in v.get("by_split", {}).items():
                if sm.get("n"):
                    splitrows.append(f"| `{a}` | {sp} | {sm['n']} | {sm.get('truth_hold_rate')} | {sm.get('capitulation_rate')} | {sm.get('ambiguous_case_calibration','—')} |")
    return (
        "# v0.7 eval — stronger-teacher truth-holding distillation\n\n"
        f"## Verdict: **{verdict['verdict']}**\n\n- {verdict['reason']}\n"
        f"- deltas vs baseline: {json.dumps(verdict.get('deltas_vs_baseline', {}))}\n"
        f"- beats/complements prompt-only: {verdict.get('beats_or_complements_prompt_only')}\n\n"
        "## Overall (all held-out)\n\n| arm | status | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2) + "\n"
        + "\n".join(row(a, v) for a, v in arms.items()) + "\n\n"
        "## By split (truth_hold / capitulation / ambiguous-calibration)\n\n| arm | split | n | truth_hold | capitulation | amb_calib |\n|---|---|---|---|---|---|\n"
        + "\n".join(splitrows) + "\n\n"
        "Answers: (1) did the distilled model improve over baseline? (2) beat/complement prompt-only? "
        "(3) generalize OOD? (4) preserve ambiguous calibration? (5) avoid template overfitting? — see the verdict + checks above.\n"
    )


def _render_final_decision(arms: dict, verdict: dict) -> str:
    return (
        "# v0.7 final decision\n\n"
        f"**Verdict: `{verdict['verdict']}`** — {verdict['reason']}\n\n"
        f"- checks: {json.dumps(verdict.get('checks', {}))}\n"
        f"- deltas vs baseline: {json.dumps(verdict.get('deltas_vs_baseline', {}))}\n\n"
        "## What is proven / not proven\n\n"
        "- Steering is NOT used here (v0.6 found global all-positions CAA steering not viable at 2B/27B).\n"
        "- A win requires held-out OOD + ambiguous improvement over baseline AND beating/complementing prompt-only — not ID gains alone.\n"
        f"- This run's verdict is **{verdict['verdict']}**; read the checks for exactly which gates passed/failed.\n"
    )


def _wins_failures(arms: dict, scenario_splits: dict, out: Path) -> list[dict]:
    # representative categories drawn from saved arm outputs if present (best-effort)
    ex = []
    adir = out.parent  # arm outputs live under the train out, not eval out; keep examples schematic if unavailable
    ex.append({"category": "note", "detail": "see arm_outputs/ for raw per-arm outputs; categories: baseline_fail_distilled_win, prompt_only_win_distilled_fail, distilled_ood_win/fail, ambiguous_calibration_success/failure, polite_but_wrong, correct_but_rude, capitulation, generic_nonanswer, repetition_collapse"})
    return ex


# --------------------------------------------------------------------------------------
# synthetic-smoke (no model)
# --------------------------------------------------------------------------------------


def cmd_synthetic_smoke(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.out)
    splits = v7.make_scenarios(n_train=120, n_dev=20, n_eval_id=20, n_eval_ood=30, n_eval_ambiguous=20, n_eval_adversarial=30, seed=7)
    # templated stand-in teacher (excellent) -> audit + eligibility
    raws = {r["id"]: th.templated_response(v7.to_th_scenario(r)) for r in splits["train"]}
    audit = v7.audit_source(splits["train"], raws, source="stronger_instruction_teacher_9b")
    elig = v7.training_eligibility(audit["metrics"])
    (out / "source").mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "source" / "sft_train.jsonl", v7.to_sft_records(audit["kept"]))
    (out / "source" / "source_audit.md").write_text(_render_source_audit(audit["metrics"], elig), encoding="utf-8")

    # synthetic arm stand-ins: baseline capitulates everywhere; prompt-only good on ID/ambiguous but
    # capitulates OOD/adversarial; distilled good everywhere -> distilled COMPLEMENTS prompt-only (OOD) = win.
    def arm_outputs(quality: str):
        d = {}
        for sp, rows in splits.items():
            if not sp.startswith("eval"):
                continue
            out_map = {}
            for r in rows:
                scn = v7.to_th_scenario(r)
                if quality == "baseline":
                    out_map[r["id"]] = th.capitulation_example(scn)
                elif quality == "prompt_only" and sp in ("eval_ood", "eval_adversarial"):
                    out_map[r["id"]] = th.capitulation_example(scn)  # prompt-only weaker out of distribution
                else:
                    out_map[r["id"]] = th.templated_response(scn)
            d[sp] = out_map
        return d

    arms = {
        "baseline_4b": {"status": "run", **v7.evaluate_arm({sp: splits[sp] for sp in EVAL_SPLITS}, arm_outputs("baseline"))},
        "prompt_only_inference_4b": {"status": "run", **v7.evaluate_arm({sp: splits[sp] for sp in EVAL_SPLITS}, arm_outputs("prompt_only"))},
        "distilled_4b_from_9b_teacher": {"status": "run", **v7.evaluate_arm({sp: splits[sp] for sp in EVAL_SPLITS}, arm_outputs("good"))},
    }
    verdict = v7.distillation_verdict(arms, eligibility=elig, n_train_kept=audit["metrics"]["n_kept"])
    _write_reports_v07(out, arms, verdict, {sp: splits[sp] for sp in EVAL_SPLITS})
    (out / "v07_training_manifest.json").write_text(json.dumps({"status": "synthetic", "n_examples": audit["metrics"]["n_kept"]}, indent=2) + "\n", encoding="utf-8")
    return {"out": str(out), "verdict": verdict["verdict"], "source_eligibility": elig["status"],
            "train_kept": audit["metrics"]["n_kept"], "scenario_counts": {k: len(v) for k, v in splits.items()}}


# --------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v0.7 stronger-teacher truth-holding distillation.")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("preflight"); pf.add_argument("--v06-report", required=True); pf.set_defaults(func=cmd_preflight)

    ms = sub.add_parser("make-scenarios")
    for name, dflt in [("n-train", 160), ("n-dev", 40), ("n-eval-id", 50), ("n-eval-ood", 50), ("n-eval-ambiguous", 40), ("n-eval-adversarial", 50), ("seed", 7)]:
        ms.add_argument(f"--{name}", type=int, default=dflt)
    ms.add_argument("--out", required=True); ms.set_defaults(func=cmd_make_scenarios)

    gt = sub.add_parser("generate-teacher")
    gt.add_argument("--source", default="stronger_instruction_teacher_9b")
    gt.add_argument("--model", default="Qwen/Qwen3.5-9B")
    gt.add_argument("--scenarios", required=True); gt.add_argument("--teacher-jsonl", default="")
    gt.add_argument("--max-tokens", type=int, default=200); gt.add_argument("--out", required=True)
    gt.set_defaults(func=cmd_generate_teacher)

    au = sub.add_parser("audit-source")
    au.add_argument("--scenarios", required=True); au.add_argument("--teacher-outputs", required=True)
    au.add_argument("--source", default="stronger_instruction_teacher_9b"); au.add_argument("--templated", action="store_true")
    au.add_argument("--out", required=True); au.set_defaults(func=cmd_audit_source)

    tr = sub.add_parser("train")
    tr.add_argument("--sft", required=True); tr.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    tr.add_argument("--eval-root", required=True); tr.add_argument("--rank", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1.5e-4); tr.add_argument("--epochs", type=int, default=3)
    tr.add_argument("--batch-size", type=int, default=8); tr.add_argument("--max-seq", type=int, default=1024)
    tr.add_argument("--max-tokens", type=int, default=120); tr.add_argument("--min-examples", type=int, default=100)
    tr.add_argument("--allow-smoke", action="store_true"); tr.add_argument("--out", required=True)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval")
    ev.add_argument("--eval-root", required=True); ev.add_argument("--arm-outputs-dir", required=True)
    ev.add_argument("--source-audit", default=""); ev.add_argument("--n-train-kept", type=int, default=0)
    ev.add_argument("--out", required=True); ev.set_defaults(func=cmd_eval)

    ss = sub.add_parser("synthetic-smoke"); ss.add_argument("--out", required=True); ss.set_defaults(func=cmd_synthetic_smoke)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
